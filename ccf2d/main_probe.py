from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import polars as pl
from argclz import AbstractParser, argument
from brainglobe_atlasapi import BrainGlobeAtlas
from neuralib.atlas.ccf.matrix import slice_transform_helper
from neuralib.atlas.util import ALLEN_CCF_10um_BREGMA
from neuralib.atlas.view import get_slice_view
from neuralib.util.verbose import fprint, print_save

from ccf2d.core import (TerminalLog, boundary_mask, ccf_mm_to_plane_point, plane_point_to_ccf_mm,
                        read_oriented, region_name, rotate)

__all__ = ['ProbeOptions']

# superficial = dorsal (top of brain, small DV); deep = ventral. ProbeRenderCLI wants dorsal first.
MARKS = ('superficial', 'deep')
_POINT = {'superficial': 'dorsal', 'deep': 'ventral'}


class ProbeOptions(AbstractParser):
    DESCRIPTION = ('Reconstruct probe shanks from dye labels on registered slices (napari). '
                   'Pick superficial+deep per shank across serial sections, then render with brainrender.')

    directory: Path | None = argument(
        '-D', '--directory',
        default=None,
        help='folder of serial sections (steps through them; reads <dir>/transformations/<stem>_transform.json)'
    )

    raw_image: Path | None = argument(
        '-I', '--image',
        default=None,
        help='single registered histology image (alternative to -D)'
    )

    transform_dir: Path | None = argument(
        '--transform-dir',
        default=None,
        help='where the *_transform.json live (default: <image-dir>/transformations)'
    )

    output: Path | None = argument(
        '-O', '--output',
        default=None,
        help='output points csv (default: <dir>/probe_shanks.csv)'
    )

    depth: int | None = argument(
        '--depth',
        default=None,
        help='implant depth (µm); if set, render adds the theoretical track, else dye-only'
    )

    interval: int | None = argument(
        '--interval',
        default=None,
        help='shank interval (µm) for a multi-shank theoretical track'
    )

    _IMG_EXT = {'.tif', '.tiff', '.png', '.jpg', '.jpeg'}

    def _list_images(self, d: Path) -> list[Path]:
        return sorted(p for p in Path(d).iterdir() if p.suffix.lower() in self._IMG_EXT)

    def run(self):
        files = self._list_images(self.directory) if self.directory else []
        if files and self.raw_image is None:
            self.raw_image = files[0]
        if self.raw_image is None:
            raise ValueError('provide a registered image via -I/--image or a folder via -D/--directory')

        base = self.raw_image.parent
        self._tdir = self.transform_dir or base / 'transformations'
        self._out = self.output or base / 'probe_shanks.csv'

        first = self._transform_path(self.raw_image)
        if not first.exists():
            raise FileNotFoundError(
                f'no native registration {first} — register with `ccf2d register` first '
                f'(legacy MATLAB .mat is not supported)')
        self._launch_napari(files)

    def _transform_path(self, img: Path) -> Path:
        return self._tdir / f'{img.stem}_transform.json'

    def _launch_napari(self, files: list[Path]):
        import napari
        from magicgui.widgets import (ComboBox, Container, Label, LineEdit, PushButton,
                                      Select, SpinBox)

        viewer = napari.Viewer(title='ccf2d probe')
        viewer.text_overlay.visible = True
        viewer.text_overlay.font_size = 18
        viewer.text_overlay.color = 'yellow'

        from napari.utils import Colormap

        bg = BrainGlobeAtlas('allen_mouse_10um')
        # region picker drives the brainrender; show "ACR — full name" but keep the acronym as value
        region_pairs = [(f"{a} — {bg.structures[a]['name']}", a)
                        for a in sorted(bg.structures.acronym_to_id_map)]

        # pts[(shank, 'dorsal'|'ventral')] = (AP, DV, ML) in bregma-relative mm
        state: dict = {'pts': {}, 'files': files, 'cursor': 0, 'plane': None,
                       'view': None, 'plane_off': None, 'name': None, 'ann_img': None, 'hover_id': None}
        # pts[(shank, point)] = (AP, DV, ML) mm; crosses are reconstructed from these per slice

        warped_layer = None  # created on first load (re-created to switch grayscale<->RGB safely)
        bound_layer = viewer.add_image(np.zeros((10, 10)), name='boundaries',
                                       colormap='red', blending='additive', opacity=0.7)
        hcmap = Colormap([[0, 0, 0], [0.3, 0.6, 1.0]], name='highlight')  # region-under-cursor fill
        highlight_layer = viewer.add_image(np.zeros((10, 10), dtype=float), name='region_highlight',
                                           colormap=hcmap, blending='additive', opacity=0.4)
        click_layer = viewer.add_points(name='clicks', face_color='cyan', symbol='cross',
                                        size=20, ndim=2)

        def set_histology(img: np.ndarray):
            # napari can't flip grayscale<->RGB in place; re-create it (reorder() puts it at bottom)
            nonlocal warped_layer
            if warped_layer is not None and warped_layer in viewer.layers:
                viewer.layers.remove(warped_layer)
            warped_layer = viewer.add_image(img, name='histology', colormap='gray')

        def reorder():
            # bottom -> top: histology, hover highlight, atlas boundaries, click points
            stack = [warped_layer, highlight_layer, bound_layer, click_layer]
            for i, lyr in enumerate(l for l in stack if l is not None and l in viewer.layers):
                viewer.layers.move(viewer.layers.index(lyr), i)

        # scrolling terminal-style log (same as register): every status.value = msg appends a line
        status_label = Label(value='')
        status_label.native.setStyleSheet(
            'font-family: Menlo, Consolas, monospace; font-size: 12px; '
            'color: #b9f27c; background: #11131a; padding: 6px;')
        status_label.native.setWordWrap(True)
        status = TerminalLog(status_label)
        status.value = 'load a registered slice to begin'
        summary = Label(value='')
        summary.native.setWordWrap(True)
        summary.native.setStyleSheet('font-family: Menlo, Consolas, monospace; font-size: 12px;')

        views: dict = {}  # cache (plane, res) -> (reference_view, annotation_view); volume load is slow

        def get_views(plane: str, res: int):
            key = (plane, res)
            if key not in views:
                views[key] = (get_slice_view('reference', plane, resolution=res),
                              get_slice_view('annotation', plane, resolution=res))
            return views[key]

        def redraw_crosses():
            # reconstruct each point's pixel from its CCF coord; show it if it lands on this slice
            view, off = state['view'], state['plane_off']
            here = []
            if view is not None and off is not None:
                for ccf in state['pts'].values():
                    plane, x, y = ccf_mm_to_plane_point(ccf, project_index=view.project_index,
                                                        resolution=view.resolution)
                    xi, yi = int(round(x)), int(round(y))
                    if 0 <= yi < off.shape[0] and 0 <= xi < off.shape[1] and abs(off[yi, xi] - plane) <= 1:
                        here.append([y, x])
            click_layer.data = np.array(here) if here else np.empty((0, 2))

        def load_slice(img: Path):
            tp = self._transform_path(img)
            if not tp.exists():
                status.value = f'no registration for {img.name} — skip or register it first'
                return

            meta = json.loads(tp.read_text())
            plane, res = meta['plane'], int(meta['resolution'])
            idx, dw, dh = int(meta['slice_index']), int(meta['dw']), int(meta['dh'])
            oimg = rotate(read_oriented(img, meta.get('flip_lr', False), meta.get('flip_ud', False)),
                          float(meta.get('rotate', 0.0)))
            _, trans = slice_transform_helper(oimg, np.array(meta['matrix'], float), plane_type=plane)

            od = lambda v: v + 1 if v != 0 else 0
            view, ann = get_views(plane, res)  # cached: avoids reloading atlas volumes per slice
            sp = view.plane_at(idx).with_offset(od(dw), od(dh))
            ann_sp = ann.plane_at(idx).with_offset(od(dw), od(dh))

            ann_img = ann_sp.image
            state.update(plane=plane, view=view, plane_off=sp.plane_offset, name=img.stem,
                         ann_img=ann_img, hover_id=None)
            set_histology(trans)
            bound_layer.data = boundary_mask(ann_img)
            highlight_layer.data = np.zeros(ann_img.shape, dtype=float)  # stale on plane change
            redraw_crosses()  # restore crosses picked on this slice (e.g. after CSV reload)
            reorder()
            viewer.reset_view()  # camera was fitted to the tiny placeholder; refit to the slice
            status.value = f'{img.name}: click the dye, marking {mark_w.value} of shank {shank_w.value}'

        def refresh_summary():
            pts = state['pts']
            if not pts:
                summary.value = 'no points yet'
                return
            rows = ['<tr><th>shank</th><th>point</th><th>AP</th><th>DV</th><th>ML</th></tr>']
            for shank in sorted({s for s, _ in pts}):
                for point in ('dorsal', 'ventral'):
                    if (shank, point) in pts:
                        ap, dv, ml = pts[(shank, point)]
                        # shank number is a link: click to remove the whole shank
                        rows.append(f'<tr><td align="center"><a href="del:{shank}">{shank} ✕</a></td>'
                                    f'<td>{point}</td><td align="right">{ap:.2f}</td>'
                                    f'<td align="right">{dv:.2f}</td><td align="right">{ml:.2f}</td></tr>')
            summary.value = ('<table border=1 cellspacing=0 cellpadding=4>' + ''.join(rows)
                             + '</table><br><i>AP, DV, ML in mm · click a shank ✕ to remove it</i>')

        def on_table_link(href: str):
            if not href.startswith('del:'):
                return
            shank = int(href.split(':', 1)[1])
            from qtpy.QtWidgets import QMessageBox
            if QMessageBox.question(None, 'Remove shank?',
                                    f'Remove shank {shank} (both points)?') != QMessageBox.Yes:
                return
            for point in ('dorsal', 'ventral'):
                state['pts'].pop((shank, point), None)
            refresh_summary()
            redraw_crosses()
            status.value = f'removed shank {shank}'

        summary.native.setOpenExternalLinks(False)
        summary.native.linkActivated.connect(on_table_link)

        @viewer.mouse_drag_callbacks.append
        def on_click(_v, event):
            dragged = False
            yield
            while event.type == 'mouse_move':
                dragged = True
                yield
            if dragged or state['view'] is None:
                return
            y, x = event.position
            off = state['plane_off']
            if not (0 <= y < off.shape[0] and 0 <= x < off.shape[1]):
                return
            ccf = plane_point_to_ccf_mm(off[int(y), int(x)], x, y,
                                        project_index=state['view'].project_index,
                                        resolution=state['view'].resolution,
                                        bregma_10um=tuple(ALLEN_CCF_10um_BREGMA))
            key = (int(shank_w.value), _POINT[mark_w.value])
            state['pts'][key] = ccf
            redraw_crosses()
            status.value = (f'shank {shank_w.value} {mark_w.value} -> '
                            f'AP {ccf[0]:.2f}, DV {ccf[1]:.2f}, ML {ccf[2]:.2f} mm')
            refresh_summary()

        @viewer.mouse_move_callbacks.append
        def on_move(_v, event):
            ann = state['ann_img']
            if ann is None:
                return
            y, x = event.position
            viewer.text_overlay.text = region_name(ann, bg.structures, y, x)
            rid = (int(ann[int(y), int(x)])
                   if 0 <= y < ann.shape[0] and 0 <= x < ann.shape[1] else 0)
            if rid != state['hover_id']:
                state['hover_id'] = rid
                highlight_layer.data = (ann == rid).astype(float) if rid else np.zeros(ann.shape)

        shank_w = SpinBox(label='shank', value=1, min=1, max=64)
        mark_w = ComboBox(label='marking', choices=list(MARKS), value=MARKS[0])
        _COLORS = ['red', 'salmon', 'darkred', 'cyan', 'yellow', 'magenta',
                   'lime', 'green', 'blue', 'orange', 'white', 'black']

        # atlas regions to render: pick region(s) in the (filterable) list, choose a color, Add.
        # region_colors keeps each region with its own color, in render order.
        region_colors: dict[str, str] = {}
        search_w = LineEdit(label='filter')
        region_w = Select(label='regions', choices=region_pairs)
        region_color_w = ComboBox(label='color', choices=_COLORS, value='red')
        add_region_btn = PushButton(text='+ Add region(s)')
        region_lbl = Label(value='no regions')
        region_lbl.native.setOpenExternalLinks(False)

        def apply_filter(*_):
            q = search_w.value.strip().lower()
            region_w.choices = [p for p in region_pairs if q in p[0].lower()] if q else list(region_pairs)

        def refresh_region_table():
            if not region_colors:
                region_lbl.value = 'no regions'
                return
            rows = ['<tr><th>region</th><th>color</th></tr>']
            for acr, c in region_colors.items():
                rows.append(f'<tr><td><a href="rdel:{acr}">{acr} ✕</a></td>'
                            f'<td><font color="{c}">■</font> {c}</td></tr>')
            region_lbl.value = '<table border=1 cellspacing=0 cellpadding=3>' + ''.join(rows) + '</table>'

        def add_regions(*_):
            for acr in region_w.value:
                region_colors[acr] = region_color_w.value  # add or recolor
            refresh_region_table()

        def on_region_link(href: str):
            if href.startswith('rdel:'):
                region_colors.pop(href[len('rdel:'):], None)
                refresh_region_table()

        search_w.changed.connect(apply_filter)
        add_region_btn.changed.connect(add_regions)
        region_lbl.native.linkActivated.connect(on_region_link)
        refresh_region_table()

        # per-shank dye color: pick a shank (above), then set its color here
        shank_colors: dict[int, str] = {}  # shank -> dye color; default red
        probe_color_w = ComboBox(label='shank color', choices=_COLORS, value='red')

        def store_shank_color(*_):
            shank_colors[int(shank_w.value)] = probe_color_w.value

        def show_shank_color(*_):
            probe_color_w.value = shank_colors.get(int(shank_w.value), 'red')

        probe_color_w.changed.connect(store_shank_color)
        shank_w.changed.connect(show_shank_color)

        _CSV_COLS = {'AP_location', 'DV_location', 'ML_location', 'probe_idx', 'point'}

        def load_csv_points(path: Path):
            try:
                df = pl.read_csv(path)
            except Exception as e:  # noqa: BLE001 - surface any read error in the GUI
                status.value = f'could not read {path.name}: {e}'
                return
            if not _CSV_COLS <= set(df.columns):
                status.value = f'{path.name}: not a probe CSV (needs {", ".join(sorted(_CSV_COLS))})'
                return
            for r in df.iter_rows(named=True):
                state['pts'][(int(r['probe_idx']), r['point'])] = (
                    r['AP_location'], r['DV_location'], r['ML_location'])
            refresh_summary()
            redraw_crosses()  # crosses are reconstructed from the loaded coordinates
            status.value = f'loaded {len({s for s, _ in state["pts"]})} shank(s) from {path.name}'

        def save_csv() -> Path | None:
            shanks = sorted({s for s, _ in state['pts']})
            rows = []
            for s in shanks:
                for point in ('dorsal', 'ventral'):  # dorsal first: ProbeRenderCLI reshape order
                    if (s, point) not in state['pts']:
                        status.value = f'shank {s} missing its {point} point'
                        return None
                    ap, dv, ml = state['pts'][(s, point)]
                    rows.append({'AP_location': ap, 'DV_location': dv, 'ML_location': ml,
                                 'probe_idx': s, 'point': point})
            if not rows:
                status.value = 'no points to save'
                return None
            self._out.parent.mkdir(parents=True, exist_ok=True)
            pl.DataFrame(rows).write_csv(self._out)
            print_save(self._out)
            status.value = f'saved {len(shanks)} shank(s) -> {self._out.name}'
            return self._out

        def on_render():
            csv = save_csv()
            if csv is None:
                return
            plane = state['plane'] or 'coronal'
            shanks = sorted({s for s, _ in state['pts']})  # same order as the saved CSV rows
            colors = ','.join(shank_colors.get(s, 'red') for s in shanks)  # one per shank
            cmd = [sys.executable, '-m', 'neuralib.atlas.brainrender.probe',
                   '--file', str(csv), '--plane-type', plane, '--probe-color', colors]
            if self.depth is None:
                cmd.append('--dye')
            else:
                cmd += ['--depth', str(self.depth)]
                if self.interval is not None:
                    cmd += ['--interval', str(self.interval)]
            regions = list(region_colors)  # insertion order = render order
            if regions:  # show the added atlas regions as 3D meshes, each with its own color
                cmd += ['--region', ','.join(regions)]
                cmd += ['--region-color', ','.join(region_colors[r] for r in regions)]
            status.value = (f'rendering ({len(shanks)} shank(s): {colors}'
                            + (f', {len(regions)} region(s)' if regions else '')
                            + ') in a separate window...')
            subprocess.Popen(cmd)

        def step(delta: int):
            files = state['files']
            if not files:
                status.value = 'single-image mode — nothing to step'
                return
            state['cursor'] = int(np.clip(state['cursor'] + delta, 0, len(files) - 1))
            load_slice(files[state['cursor']])

        prev_btn = PushButton(text='◀ Prev slice')
        prev_btn.changed.connect(lambda *_: step(-1))
        next_btn = PushButton(text='Next slice ▶')
        next_btn.changed.connect(lambda *_: step(+1))

        def on_load_csv():
            from qtpy.QtWidgets import QFileDialog
            path, _ = QFileDialog.getOpenFileName(caption='Load probe points CSV',
                                                  filter='CSV (*.csv);;All files (*)')
            if path:
                load_csv_points(Path(path))

        load_btn = PushButton(text='Load CSV')
        load_btn.changed.connect(lambda *_: on_load_csv())
        save_btn = PushButton(text='Save CSV')
        save_btn.changed.connect(lambda *_: save_csv())
        render_btn = PushButton(text='Render (brainrender)')
        render_btn.changed.connect(lambda *_: on_render())

        def header(text):
            lbl = Label(value=text)
            lbl.native.setStyleSheet('font-weight: bold; color: #88c0d0; padding-top: 6px;')
            return lbl

        def srow(*ws):
            return Container(widgets=list(ws), layout='horizontal', labels=False)

        panel = Container(widgets=[
            header('Slice'), srow(prev_btn, next_btn),
            header('Dye point'), shank_w, mark_w, probe_color_w,
            header('Accumulated'), summary,
            header('Render'), search_w, region_w, srow(region_color_w, add_region_btn),
            region_lbl, render_btn,
            header('CSV'), srow(load_btn, save_btn),
            status_label,
        ])
        viewer.window.add_dock_widget(panel, area='right', name='probe')

        if state['files']:
            load_slice(state['files'][0])
        elif self.raw_image:
            load_slice(self.raw_image)
        if self._out.exists():
            load_csv_points(self._out)  # auto-resume the session CSV
        refresh_summary()

        fprint('probe: pick superficial+deep dye per shank (step slices as needed), then Render')
        napari.run()


if __name__ == '__main__':
    ProbeOptions().main()
