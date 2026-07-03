from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl
from argclz import argument
from brainglobe_atlasapi import BrainGlobeAtlas
from neuralib.atlas.ccf.matrix import slice_transform_helper
from neuralib.atlas.util import ALLEN_CCF_10um_BREGMA
from neuralib.util.verbose import fprint

from regrender._core import (boundary_mask, ccf_mm_to_plane_point, ccf_mm_to_voxel, load_transform,
                        plane_point_to_ccf_mm, read_oriented, rotate)
from regrender._app import RegionPicker, SliceReconstructOptions

__all__ = ['ProbeOptions']

# superficial = dorsal (top of brain, small DV); deep = ventral. ProbeRenderCLI wants dorsal first.
MARKS = ('superficial', 'deep')
_POINT = {'superficial': 'dorsal', 'deep': 'ventral'}
_COLORS = ['red', 'salmon', 'darkred', 'cyan', 'yellow', 'magenta',
           'lime', 'green', 'blue', 'orange', 'white', 'black']
SHADER_STYLES = ['plastic', 'cartoon', 'metallic', 'shiny', 'glossy']


def _shank_table_html(pts: dict[tuple[int, str], tuple[float, float, float]],
                      src: dict[tuple[int, str], str] | None = None) -> str:
    """HTML table of picked points; each shank number is a ``del:<n>`` link to remove it."""
    if not pts:
        return 'no points yet'
    src = src or {}
    rows = ['<tr><th>shank</th><th>point</th><th>AP</th><th>DV</th><th>ML</th><th>source</th></tr>']
    for shank in sorted({s for s, _ in pts}):
        for point in ('dorsal', 'ventral'):
            if (shank, point) in pts:
                ap, dv, ml = pts[(shank, point)]
                rows.append(f'<tr><td align="center"><a href="del:{shank}">{shank} ✕</a></td>'
                            f'<td>{point}</td><td align="right">{ap:.2f}</td>'
                            f'<td align="right">{dv:.2f}</td><td align="right">{ml:.2f}</td>'
                            f'<td>{src.get((shank, point), "")}</td></tr>')
    return ('<table border=1 cellspacing=0 cellpadding=4>' + ''.join(rows)
            + '</table><br><i>AP, DV, ML in mm · click a shank ✕ to remove it</i>')


def _render_command(csv: Path, plane: str, shanks: list[int], shank_colors: dict[int, str],
                    region_colors: dict[str, str], depth: int | None, interval: int | None,
                    style: str = 'plastic', no_root: bool = False, hemisphere: str = 'both') -> list[str]:
    """``ProbeRenderCLI`` argv (after ``-m``): per-shank dye colors, optional track, region meshes."""
    cmd = ['neuralib.atlas.brainrender.probe',
           '--file', str(csv), '--plane-type', plane, '--style', style, '--hemisphere', hemisphere,
           '--probe-color', ','.join(shank_colors.get(s, 'red') for s in shanks)]
    if no_root:
        cmd.append('--no-root')
    if depth is None:
        cmd.append('--dye')
    else:
        cmd += ['--depth', str(depth)]
        if interval is not None:
            cmd += ['--interval', str(interval)]
    if region_colors:  # insertion order = render order; each region carries its own color
        cmd += ['--region', ','.join(region_colors)]
        cmd += ['--region-color', ','.join(region_colors.values())]
    return cmd


class ProbeOptions(SliceReconstructOptions):
    DESCRIPTION = ('Reconstruct probe shanks from dye labels on registered slices (napari). '
                   'Pick superficial+deep per shank across serial sections, then render with brainrender.')

    depth: int | None = argument(
        '--depth', default=None,
        help='implant depth (µm); if set, render adds the theoretical track, else dye-only')

    interval: int | None = argument(
        '--interval', default=None,
        help='shank interval (µm) for a multi-shank theoretical track')

    def run(self):
        files = self._resolve_paths('probe/probe_shanks.csv')
        self._launch_napari(files)

    def _launch_napari(self, files: list[Path]):
        import napari
        from napari.utils import Colormap
        from magicgui.widgets import CheckBox, ComboBox, Container, Label, PushButton, SpinBox

        viewer = napari.Viewer(title='regrender probe')
        viewer.text_overlay.visible = True
        viewer.text_overlay.font_size = 18
        viewer.text_overlay.color = 'yellow'

        bg = BrainGlobeAtlas('allen_mouse_10um', check_latest=False)

        # pts[(shank, 'dorsal'|'ventral')] = (AP, DV, ML) bregma-relative mm; crosses re-derived per slice
        # src[(shank, 'dorsal'|'ventral')] = source slice stem the point was picked on
        state: dict = {'pts': {}, 'src': {}, 'files': files, 'cursor': 0, 'plane': None,
                       'view': None, 'plane_off': None, 'name': None, 'ann_img': None, 'hover_id': None,
                       'mode': 'single', 'tiles': [], 'grid': None, 'warped': None}

        warped_layer = None  # created on first load (re-created to switch grayscale<->RGB safely)
        bound_layer = viewer.add_image(np.zeros((10, 10)), name='boundaries',
                                       colormap='red', blending='additive', opacity=0.7)
        hcmap = Colormap([[0, 0, 0], [0.3, 0.6, 1.0]], name='highlight')  # region-under-cursor fill
        highlight_layer = viewer.add_image(np.zeros((10, 10), dtype=float), name='region_highlight',
                                           colormap=hcmap, blending='additive', opacity=0.4)
        click_layer = viewer.add_points(name='clicks', face_color='cyan', symbol='cross',
                                        size=20, ndim=2,
                                        features={'label': np.empty(0, dtype='<U8')},
                                        text={'string': '{label}', 'color': 'cyan', 'size': 12,
                                              'translation': [-14, 0]})

        def _set_hist_layer(display: np.ndarray):
            # napari can't flip grayscale<->RGB in place; re-create it (reorder() puts it at bottom)
            nonlocal warped_layer
            if warped_layer is not None and warped_layer in viewer.layers:
                viewer.layers.remove(warped_layer)
            warped_layer = viewer.add_image(display, name='histology', colormap='gray')

        def set_histology(img: np.ndarray):
            state['warped'] = img  # keep the full image so the channel selector can re-slice it
            _set_hist_layer(self.channel_view(img, channel_w.value))

        def reorder():
            # bottom -> top: histology, hover highlight, atlas boundaries, click points
            stack = [warped_layer, highlight_layer, bound_layer, click_layer]
            for i, lyr in enumerate(l for l in stack if l is not None and l in viewer.layers):
                viewer.layers.move(viewer.layers.index(lyr), i)

        status_label, status = self.make_status_log()
        status.value = 'load a registered slice to begin'
        summary = Label(value='')
        summary.native.setWordWrap(True)
        summary.native.setStyleSheet('font-family: Menlo, Consolas, monospace; font-size: 12px;')
        slice_lbl = Label(value='no slice loaded')  # current file (i/N) shown in the Slice section
        slice_lbl.native.setWordWrap(True)

        get_views = self.make_view_cache()

        def frames() -> list[dict]:
            # tiles to reconstruct crosses onto: N in all-slices view, 1 (origin 0,0) in single view
            if state['mode'] == 'all':
                return state['tiles']
            if state['view'] is None:
                return []
            return [{'view': state['view'], 'plane_off': state['plane_off'], 'origin': (0, 0)}]

        def redraw_crosses():
            # reconstruct each point's pixel from its CCF coord; show it on every frame it lands on
            here, labels, keys = [], [], []
            for fr in frames():
                view, off, (oy, ox) = fr['view'], fr['plane_off'], fr['origin']
                for (shank, point), ccf in state['pts'].items():
                    plane, x, y = ccf_mm_to_plane_point(ccf, project_index=view.project_index,
                                                        resolution=view.resolution)
                    xi, yi = int(round(x)), int(round(y))
                    if 0 <= yi < off.shape[0] and 0 <= xi < off.shape[1] and abs(off[yi, xi] - plane) <= 1:
                        here.append([oy + y, ox + x])
                        labels.append(f'{shank} ({point[0]})')  # e.g. "0 (d)" / "0 (v)"
                        keys.append((shank, point))
            state['cross_here'], state['cross_keys'] = here, keys  # parallel to displayed points
            state['syncing'] = True  # guard: our own data write must not re-enter the delete handler
            click_layer.data = np.array(here) if here else np.empty((0, 2))
            click_layer.features = {'label': np.array(labels, dtype='<U8')}
            state['syncing'] = False

        def on_crosses_edited(_event=None):
            # deleting a cross in napari only mutates the layer; mirror it back into state + table
            if state.get('syncing'):
                return
            survivors = {(round(y, 3), round(x, 3)) for y, x in click_layer.data}
            removed = [k for k, (y, x) in zip(state.get('cross_keys', []), state.get('cross_here', []))
                       if (round(y, 3), round(x, 3)) not in survivors]
            if not removed:
                return
            for k in removed:
                state['pts'].pop(k, None)
                state['src'].pop(k, None)
            refresh_summary()
            redraw_crosses()
            status.value = f'removed {len(removed)} point(s)'

        def compute_slice(img: Path) -> dict | None:
            # warp one registered slice to atlas space; None if it has no saved registration
            tp = self._transform_path(img)
            if not tp.exists():
                return None
            meta = load_transform(tp)
            plane, res = meta['plane'], int(meta['resolution'])
            idx, dw, dh = int(meta['slice_index']), int(meta['dw']), int(meta['dh'])
            oimg = rotate(read_oriented(img, meta.get('flip_lr', False), meta.get('flip_ud', False)),
                          float(meta.get('rotate', 0.0)))
            _, trans = slice_transform_helper(oimg, np.array(meta['matrix'], float), plane_type=plane)
            od = lambda v: v + 1 if v != 0 else 0
            view, ann = get_views(plane, res)  # cached: avoids reloading atlas volumes per slice
            sp = view.plane_at(idx).with_offset(od(dw), od(dh))
            ann_sp = ann.plane_at(idx).with_offset(od(dw), od(dh))
            return {'trans': trans, 'view': view, 'plane_off': sp.plane_offset,
                    'ann_img': ann_sp.image, 'plane': plane}

        def load_slice(img: Path):
            r = compute_slice(img)
            if r is None:
                status.value = f'no registration for {img.name} — skip or register it first'
                return
            view, ann_img = r['view'], r['ann_img']
            state.update(plane=r['plane'], view=view, plane_off=r['plane_off'], name=img.stem,
                         ann_img=ann_img, hover_id=None)
            set_histology(r['trans'])
            bound_layer.visible = highlight_layer.visible = True  # may have been hidden by all-view
            bound_layer.data = boundary_mask(ann_img)
            highlight_layer.data = np.zeros(ann_img.shape, dtype=float)  # stale on plane change
            redraw_crosses()  # restore crosses picked on this slice (e.g. after CSV reload)
            reorder()
            viewer.reset_view()  # camera was fitted to the tiny placeholder; refit to the slice
            files = state['files']
            pos = f'{state["cursor"] + 1}/{len(files)}  ' if files else ''
            slice_lbl.value = f'{pos}{img.name}'
            status.value = f'{img.name}: click the dye, marking {mark_w.value} of shank {shank_w.value}'

        def refresh_summary():
            summary.value = _shank_table_html(state['pts'], state['src'])

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
                state['src'].pop((shank, point), None)
            refresh_summary()
            redraw_crosses()
            status.value = f'removed shank {shank}'

        summary.native.setOpenExternalLinks(False)
        summary.native.linkActivated.connect(on_table_link)
        click_layer.events.data.connect(on_crosses_edited)

        def frame_at(y: float, x: float) -> dict | None:
            # which slice was clicked: the single frame, or the mosaic tile under (y, x)
            if state['mode'] == 'all':
                if not state['tiles']:
                    return None
                h, w, cols = state['grid']
                row, col = int(y) // h, int(x) // w
                idx = row * cols + col
                return state['tiles'][idx] if 0 <= col < cols and 0 <= idx < len(state['tiles']) else None
            return frames()[0] if frames() else None

        @viewer.mouse_drag_callbacks.append
        def on_click(_v, event):
            dragged = False
            yield
            while event.type == 'mouse_move':
                dragged = True
                yield
            if dragged:
                return
            y, x = event.position
            fr = frame_at(y, x)
            if fr is None:
                return
            view, off, (oy, ox) = fr['view'], fr['plane_off'], fr['origin']
            ly, lx = y - oy, x - ox  # pixel within this slice
            if not (0 <= ly < off.shape[0] and 0 <= lx < off.shape[1]):
                return
            ccf = plane_point_to_ccf_mm(off[int(ly), int(lx)], lx, ly,
                                        project_index=view.project_index,
                                        resolution=view.resolution,
                                        bregma_10um=tuple(ALLEN_CCF_10um_BREGMA))
            key = (int(shank_w.value), _POINT[mark_w.value])
            state['pts'][key] = ccf
            state['src'][key] = fr.get('name', state['name'])  # slice this point was picked on
            redraw_crosses()
            status.value = (f'shank {shank_w.value} {mark_w.value} -> '
                            f'AP {ccf[0]:.2f}, DV {ccf[1]:.2f}, ML {ccf[2]:.2f} mm')
            refresh_summary()

        viewer.mouse_move_callbacks.append(self.make_hover(viewer, state, bg.structures, highlight_layer))

        shank_w = SpinBox(label='shank', value=1, min=1, max=64)
        mark_w = ComboBox(label='marking', choices=list(MARKS), value=MARKS[0])
        channel_w = ComboBox(label='channel', choices=['merge', 'R', 'G', 'B'], value='merge')

        def on_channel(*_):
            if state['mode'] == 'all':
                show_all()  # rebuild the mosaic in the new channel
            elif state.get('warped') is not None:
                set_histology(state['warped'])  # re-slice current image; keep camera
                reorder()

        channel_w.changed.connect(on_channel)

        regions = RegionPicker(bg.structures)
        style_w = ComboBox(label='style', choices=SHADER_STYLES, value=SHADER_STYLES[0])
        hemisphere_w = ComboBox(label='hemisphere', choices=['both', 'left', 'right'], value='both')
        no_root_w = CheckBox(label='no root (hide brain)', value=False)

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
                key = (int(r['probe_idx']), r['point'])
                state['pts'][key] = (r['AP_location'], r['DV_location'], r['ML_location'])
                state['src'][key] = r.get('source', '') or ''  # column optional on older CSVs
            refresh_summary()
            redraw_crosses()  # crosses are reconstructed from the loaded coordinates
            status.value = f'loaded {len({s for s, _ in state["pts"]})} shank(s) from {path.name}'

        def build_rows() -> list[dict] | None:
            shanks = sorted({s for s, _ in state['pts']})
            rows = []
            for s in shanks:
                for point in ('dorsal', 'ventral'):  # dorsal first: ProbeRenderCLI reshape order
                    if (s, point) not in state['pts']:
                        status.value = f'shank {s} missing its {point} point'
                        return None
                    ap, dv, ml = state['pts'][(s, point)]
                    rows.append({'AP_location': ap, 'DV_location': dv, 'ML_location': ml,
                                 'probe_idx': s, 'point': point,
                                 'source': state['src'].get((s, point), '')})
            if not rows:
                status.value = 'no points to save'
                return None
            return rows

        def save_csv() -> Path | None:
            rows = build_rows()
            if rows is None:
                return None
            self._out.parent.mkdir(parents=True, exist_ok=True)
            pl.DataFrame(rows).write_csv(self._out)
            status.value = f'saved {len({r["probe_idx"] for r in rows})} shank(s) -> {self._out}'
            return self._out

        def on_render():
            rows = build_rows()  # render from current points without touching the session CSV
            if rows is None:
                return
            import tempfile
            csv = Path(tempfile.gettempdir()) / 'regrender_probe_render.csv'
            pl.DataFrame(rows).write_csv(csv)
            shanks = sorted({s for s, _ in state['pts']})  # same order as the saved CSV rows
            cmd = _render_command(csv, state['plane'] or 'coronal', shanks, shank_colors,
                                  regions.colors, self.depth, self.interval, style_w.value,
                                  no_root_w.value, hemisphere_w.value)
            self.launch_render(cmd, status,
                               f'rendering ({len(shanks)} shank(s)'
                               + (f', {len(regions.colors)} region(s)' if regions.colors else '')
                               + ') in a separate window...')

        def region_profile():
            # for each shank, sample dorsal->ventral and show which region each depth band is in
            shanks = sorted({s for s, _ in state['pts']})
            tracks = []  # (shank, runs, dv_span); runs = [(dv0, dv1, acronym), ...] in mm
            for s in shanks:
                d, v = state['pts'].get((s, 'dorsal')), state['pts'].get((s, 'ventral'))
                if d is None or v is None:
                    status.value = f'shank {s} missing its dorsal/ventral point'
                    return
                d, v = np.array(d, float), np.array(v, float)
                ts = np.linspace(0, 1, 256)
                dvs = d[1] + (v[1] - d[1]) * ts  # DV (mm) at each sample, dorsal -> ventral
                dv_span = abs(v[1] - d[1])
                acrs = []
                for t in ts:
                    try:
                        a = bg.structure_from_coords(ccf_mm_to_voxel(tuple(d + (v - d) * t)),
                                                     microns=False, as_acronym=True)
                    except Exception:  # noqa: BLE001 - outside the annotated volume
                        a = 'out'
                    acrs.append(a)
                runs, i = [], 0
                while i < len(ts):  # collapse consecutive samples of the same region into one band
                    j = i
                    while j < len(ts) and acrs[j] == acrs[i]:
                        j += 1
                    # extend to the next band's boundary (dvs[j]) so bands abut with no gaps
                    dv_end = dvs[j] if j < len(ts) else dvs[-1]
                    runs.append((dvs[i], dv_end, acrs[i]))
                    i = j
                tracks.append((s, runs, dv_span))
            if not tracks:
                status.value = 'no shanks to plot'
                return

            def acr_color(a: str):
                try:
                    r, g, b = bg.structures[a]['rgb_triplet']
                    return r / 255, g / 255, b / 255
                except KeyError:
                    return 0.85, 0.85, 0.85  # 'out' / root / unknown

            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(1.6 * len(tracks) + 1, 6))
            for xi, (s, runs, dv_span) in enumerate(tracks):
                for dv0, dv1, a in runs:
                    lo, hi = min(dv0, dv1), max(dv0, dv1)
                    ax.bar(xi, hi - lo, bottom=lo, width=0.8, color=acr_color(a),
                           edgecolor='white', linewidth=0.5)
                    ax.text(xi, (lo + hi) / 2, a, ha='center', va='center', fontsize=7)
            ax.set_xticks(range(len(tracks)))
            ax.set_xticklabels([f'shank {s}' for s, _, _ in tracks])
            ax.set_ylabel('DV from bregma (mm)')
            ax.set_title('Probe region profile')
            ax.invert_yaxis()  # dorsal (smaller DV) at top
            fig.tight_layout()
            self._out.parent.mkdir(parents=True, exist_ok=True)
            out = self._out.parent / 'probe_region_profile.png'
            fig.savefig(out, dpi=150)
            status.value = f'region profile -> {out}'
            plt.show(block=False)

        def invert_ml():
            if not state['pts']:
                status.value = 'no points to flip'
                return
            for k, (ap, dv, ml) in list(state['pts'].items()):
                state['pts'][k] = (ap, dv, -ml)  # midline is ML=0; mirror to the other hemisphere
            refresh_summary()
            redraw_crosses()
            status.value = 'flipped ML -> other hemisphere'

        def show_all():
            # tile every registered slice into one canvas; store per-tile geometry for click mapping
            files = state['files'] or ([self.raw_image] if self.raw_image else [])
            items = [(f.stem, r) for f in files if (r := compute_slice(f)) is not None]
            if not items:
                status.value = 'no registered slices to show'
                return
            imgs = [self.channel_view(r['trans'], channel_w.value) for _, r in items]
            tail = imgs[0].shape[2:]  # () for grayscale, (3,) / (4,) for RGB(A)
            th = max(i.shape[0] for i in imgs)
            tw = max(i.shape[1] for i in imgs)
            cols = int(np.ceil(np.sqrt(len(imgs))))
            rows = int(np.ceil(len(imgs) / cols))
            big = np.zeros((rows * th, cols * tw) + tail, dtype=imgs[0].dtype)
            bmask = np.zeros((rows * th, cols * tw), dtype=bool)  # atlas boundaries, tiled the same way
            tiles = []
            for i, ((name, r), im) in enumerate(zip(items, imgs)):
                oy, ox = (i // cols) * th, (i % cols) * tw
                big[oy:oy + im.shape[0], ox:ox + im.shape[1]] = im
                b = boundary_mask(r['ann_img'])
                bmask[oy:oy + b.shape[0], ox:ox + b.shape[1]] = b
                tiles.append({'view': r['view'], 'plane_off': r['plane_off'],
                              'origin': (oy, ox), 'name': name})
            state.update(tiles=tiles, grid=(th, tw, cols), view=None, plane_off=None, ann_img=None)
            _set_hist_layer(big)
            bound_layer.data = bmask
            bound_layer.visible = True
            highlight_layer.visible = False  # pick-only: static boundaries, but no hover highlight
            redraw_crosses()
            reorder()
            viewer.reset_view()
            slice_lbl.value = f'all slices ({len(tiles)})'
            status.value = f'all-slices view: click any slice to mark {mark_w.value} of shank {shank_w.value}'

        def refresh_view():
            if state['mode'] == 'all':
                show_all()
            elif state['files']:
                load_slice(state['files'][state['cursor']])
            elif self.raw_image:
                load_slice(self.raw_image)

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
        view_w = ComboBox(label='view', choices=['single', 'all'], value='single')

        def on_mode(*_):
            state['mode'] = view_w.value
            single = view_w.value == 'single'
            prev_btn.enabled = next_btn.enabled = single  # stepping only makes sense in single view
            refresh_view()

        view_w.changed.connect(on_mode)

        def on_load_csv():
            from qtpy.QtWidgets import QFileDialog
            path, _ = QFileDialog.getOpenFileName(caption='Load probe points CSV',
                                                  filter='CSV (*.csv);;All files (*)')
            if path:
                load_csv_points(Path(path))

        invert_btn = PushButton(text='Invert ML (flip hemisphere)')
        invert_btn.changed.connect(lambda *_: invert_ml())
        load_btn = PushButton(text='Load CSV')
        load_btn.changed.connect(lambda *_: on_load_csv())
        save_btn = PushButton(text='Save CSV')
        save_btn.changed.connect(lambda *_: save_csv())
        render_btn = PushButton(text='Render (brainrender)')
        render_btn.changed.connect(lambda *_: on_render())
        profile_btn = PushButton(text='Region profile plot')
        profile_btn.changed.connect(lambda *_: region_profile())

        panel = Container(widgets=[
            self.header('Slice'), slice_lbl, view_w, self.srow(prev_btn, next_btn),
            self.header('Dye point'), shank_w, mark_w, channel_w, probe_color_w,
            self.header('Accumulated'), summary, invert_btn,
            self.header('Regions'), *regions.widgets, profile_btn,
            self.header('Render'), self.srow(style_w, hemisphere_w), no_root_w, render_btn,
            self.header('CSV'), self.srow(load_btn, save_btn),
            status_label,
        ])
        self.add_scroll_dock(viewer, panel, 'probe')

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
