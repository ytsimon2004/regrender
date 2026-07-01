"""ROI labeling + projection.

Workflow order is **roi → register → project**. ROIs (cells) are labelled on the *full
resolution* raw histology (you need the detail to see them) and saved in **raw pixel**
coords, independent of any registration. Once the slice is registered (`ccf2d register`),
the saved transform is applied to those raw coords to project them into the down-sampled
Allen atlas / CCF space, then reconstructed with brainrender. Projection is a single
function used by both the GUI "Project + Render" button and the headless ``--project`` CLI.
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import polars as pl
from argclz import argument
from brainglobe_atlasapi import BrainGlobeAtlas
from neuralib.atlas.util import ALLEN_CCF_10um_BREGMA
from neuralib.util.verbose import fprint, print_save

from neuralib.atlas.ccf.matrix import slice_transform_helper

from ccf2d.core import (boundary_mask, plane_point_to_ccf_mm, raw_points_to_atlas, read_oriented,
                        region_name, rotate)
from ccf2d.slice_app import RegionPicker, SliceReconstructOptions

__all__ = ['RoiOptions', 'project_raw_rois']

_COLORS = ['orange', 'magenta', 'red', 'cyan', 'yellow', 'lime', 'green', 'blue', 'white', 'dimgray']
SHADER_STYLES = ['plastic', 'cartoon', 'metallic', 'shiny', 'glossy']

RAW_COLS = ('slice', 'x', 'y', 'raw_h', 'raw_w')  # saved raw-ROI csv schema (channel optional, added on save)
# per-channel cross / render color; 'merge' overridden by the GUI ROI-color picker
_CH_COLOR = {'merge': 'cyan', 'R': 'red', 'G': 'green', 'B': 'blue'}


def project_raw_rois(rows, get_views, structures, transform_for):
    """Project raw-pixel ROIs into Allen CCF mm.

    :param rows: dicts with keys ``slice, x, y, raw_h, raw_w`` (raw image pixels + shape), optional ``channel``.
    :param get_views: ``(plane, res) -> (reference_view, annotation_view)`` (see make_view_cache).
    :param structures: BrainGlobe structures (for region acronyms).
    :param transform_for: ``slice_stem -> meta dict`` (the saved transform) or ``None`` if unregistered.
    :return: ``(ccf_rows, missing_slices)`` — ccf_rows have AP/DV/ML_location + region + source + channel.
    """
    by_slice: dict[str, list] = defaultdict(list)
    for r in rows:
        by_slice[r['slice']].append(r)

    ccf_rows, missing = [], []
    for stem, rs in by_slice.items():
        meta = transform_for(stem)
        if meta is None:
            missing.append(stem)
            continue
        plane, res = meta['plane'], int(meta['resolution'])
        idx, dw, dh = int(meta['slice_index']), int(meta['dw']), int(meta['dh'])
        view, ann = get_views(plane, res)
        od = lambda v: v + 1 if v != 0 else 0
        off = view.plane_at(idx).with_offset(od(dw), od(dh)).plane_offset
        ann_img = ann.plane_at(idx).with_offset(od(dw), od(dh)).image

        raw_shape = (int(rs[0]['raw_h']), int(rs[0]['raw_w']))  # constant per slice
        atlas = raw_points_to_atlas(
            [[r['x'], r['y']] for r in rs], matrix=np.array(meta['matrix'], float),
            raw_shape=raw_shape, plane=plane, rotate_deg=float(meta.get('rotate', 0.0)),
            flip_lr=meta.get('flip_lr', False), flip_ud=meta.get('flip_ud', False))
        for r, (ax, ay) in zip(rs, atlas):  # rs and atlas stay aligned (same order)
            xi, yi = int(round(ax)), int(round(ay))
            if not (0 <= yi < off.shape[0] and 0 <= xi < off.shape[1]):
                continue  # mapped outside the atlas plane
            ap, dv, ml = plane_point_to_ccf_mm(off[yi, xi], ax, ay, project_index=view.project_index,
                                               resolution=view.resolution,
                                               bregma_10um=tuple(ALLEN_CCF_10um_BREGMA))
            acr = region_name(ann_img, structures, ay, ax).split(' — ')[0]
            ccf_rows.append({'AP_location': ap, 'DV_location': dv, 'ML_location': ml,
                             'region': acr, 'source': stem, 'channel': r.get('channel', 'merge')})
    return ccf_rows, missing


def write_channel_csvs(ccf_rows, ccf_csv: Path) -> list[tuple[str, Path]]:
    """Split projected rows into one CCF csv per channel (so each renders as its own colored cloud).
    Returns ``[(channel, path), ...]`` in stable channel order."""
    groups: dict[str, list] = defaultdict(list)
    for r in ccf_rows:
        groups[r.get('channel', 'merge')].append(r)
    cols = ('AP_location', 'DV_location', 'ML_location', 'region', 'source')
    out = []
    for ch in sorted(groups):
        p = ccf_csv.with_name(f'{ccf_csv.stem}_{ch}.csv')
        pl.DataFrame([{k: r[k] for k in cols} for r in groups[ch]]).write_csv(p)
        out.append((ch, p))
    return out


def _render_argv(channel_files: list[tuple[str, Path]], radius: float, ch_color: dict[str, str],
                 region_colors: dict[str, str], style: str, no_root: bool, hemisphere: str) -> list[str]:
    """``RoiRenderCLI`` argv (after ``-m``): one --file per channel + matching colors + region meshes."""
    cmd = ['neuralib.atlas.brainrender.roi']
    colors = []
    for ch, p in channel_files:
        cmd += ['--file', str(p)]
        colors.append(ch_color.get(ch, 'orange'))
    cmd += ['--roi-colors', ','.join(colors), '--roi-radius', str(radius),
            '--style', style, '--hemisphere', hemisphere]
    if no_root:
        cmd.append('--no-root')
    if region_colors:
        cmd += ['--region', ','.join(region_colors)]
        cmd += ['--region-color', ','.join(region_colors.values())]
    return cmd


class RoiOptions(SliceReconstructOptions):
    DESCRIPTION = ('Label ROIs (cells) on the full-res raw histology and save them in raw pixel '
                   'coords. After registration, project them into Allen CCF space and render '
                   '(GUI button or `--project`).')

    project: bool = argument('--project',
                             help='headless: project a saved raw-ROI CSV into CCF space (no GUI)')
    render: bool = argument('--render', help='with --project, also launch brainrender')
    roi_radius: float = argument('--roi-radius', default=30, help='rendered ROI sphere radius (µm)')
    roi_color: str = argument('--roi-color', default='orange', help='rendered ROI point color')

    def run(self):
        if self.project:
            self._run_project_headless()
            return
        files = self._resolve_paths('roi_points_raw.csv', require_transform=False)
        self._launch_napari(files)

    def _ccf_out(self) -> Path:
        return self._out.with_name('roi_points_ccf.csv')

    def _transform_for(self, stem: str):
        p = self._tdir / f'{stem}_transform.json'
        return json.loads(p.read_text()) if p.exists() else None

    def _project_and_save(self, rows, get_views, structures, status=None):
        """Shared projection: raw rows -> combined CCF csv. Returns ``(path, ccf_rows)`` or ``None``."""
        ccf_rows, missing = project_raw_rois(rows, get_views, structures, self._transform_for)
        msg = lambda m: status.__setattr__('value', m) if status is not None else fprint(m)
        if missing:
            msg(f'no registration for slice(s): {", ".join(sorted(set(missing)))} — register them first')
        if not ccf_rows:
            msg('nothing projected (no registered ROIs)')
            return None
        out = self._ccf_out()
        out.parent.mkdir(parents=True, exist_ok=True)
        pl.DataFrame(ccf_rows).write_csv(out)  # combined artifact (with channel column)
        print_save(out)
        msg(f'projected {len(ccf_rows)} ROI(s) -> {out.name}')
        return out, ccf_rows

    def _run_project_headless(self):
        # resolve the raw csv input + transform dir without needing the images
        raw_csv = self.output or (
            (self.directory or (self.raw_image.parent if self.raw_image else Path.cwd()))
            / 'roi_points_raw.csv')
        base = self.directory or (self.raw_image.parent if self.raw_image else raw_csv.parent)
        self._tdir = self.transform_dir or base / 'transformations'
        self._out = raw_csv
        if not raw_csv.exists():
            raise FileNotFoundError(f'no raw-ROI csv {raw_csv} — label ROIs with `ccf2d roi` first')

        df = pl.read_csv(raw_csv)
        if not set(RAW_COLS) <= set(df.columns):
            raise ValueError(f'{raw_csv.name}: not a raw-ROI csv (needs {", ".join(RAW_COLS)})')
        get_views = self.make_view_cache()
        structures = BrainGlobeAtlas('allen_mouse_10um').structures
        res = self._project_and_save(df.to_dicts(), get_views, structures)
        if res is not None and self.render:
            path, ccf_rows = res
            ch_files = write_channel_csvs(ccf_rows, path)
            ch_color = {**_CH_COLOR, 'merge': self.roi_color}
            self.launch_render(_render_argv(ch_files, self.roi_radius, ch_color, {},
                                            SHADER_STYLES[0], False, 'both'))
            fprint(f'rendering {len(ch_files)} channel(s) in a separate window...')

    def _launch_napari(self, files: list[Path]):
        import napari
        from magicgui.widgets import CheckBox, ComboBox, Container, Label, PushButton

        viewer = napari.Viewer(title='ccf2d roi')
        viewer.camera.mouse_pan = False  # left-drag must add ROIs, not pan the slice; scroll still zooms
        viewer.text_overlay.visible = True  # region name under cursor (verify mode)
        viewer.text_overlay.font_size = 18
        viewer.text_overlay.color = 'yellow'

        bg = BrainGlobeAtlas('allen_mouse_10um')
        get_views = self.make_view_cache()  # lazy: atlas volumes load only on first Project/Verify

        # rois[id] = {'slice': stem, 'raw': (y, x)}; shapes[stem] = (H, W)
        state: dict = {'rois': {}, 'next_id': 1, 'shapes': {}, 'files': files, 'cursor': 0,
                       'name': None, 'path': None, 'raw_img': None, 'verify': False,
                       'ann_img': None, 'hover_id': None}

        from napari.utils import Colormap
        raw_layer = None  # image layer, re-created per slice (grayscale<->RGB safe)
        roi_layer = viewer.add_points(name='rois', face_color='cyan', symbol='cross', size=20, ndim=2)
        hcmap = Colormap([[0, 0, 0], [0.3, 0.6, 1.0]], name='highlight')
        highlight_layer = viewer.add_image(np.zeros((10, 10), dtype=float), name='region_highlight',
                                           colormap=hcmap, blending='additive', opacity=0.4)
        bound_layer = viewer.add_image(np.zeros((10, 10)), name='boundaries', colormap='red',
                                       blending='additive', opacity=0.7)
        highlight_layer.visible = bound_layer.visible = False  # atlas overlays: verify mode only
        viewer.layers.selection.active = roi_layer
        roi_layer.mode = 'add'

        def select_roi_layer():
            viewer.layers.selection.active = roi_layer  # adding the image layer steals focus
            roi_layer.mode = 'add'

        def set_raw(img: np.ndarray):
            nonlocal raw_layer
            if raw_layer is not None and raw_layer in viewer.layers:
                viewer.layers.remove(raw_layer)
            raw_layer = viewer.add_image(img, name='histology', colormap='gray')

        def reorder():  # bottom -> top: image, region highlight, atlas boundaries, ROI crosses
            for i, lyr in enumerate(l for l in (raw_layer, highlight_layer, bound_layer, roi_layer)
                                    if l is not None and l in viewer.layers):
                viewer.layers.move(viewer.layers.index(lyr), i)

        status_label, status = self.make_status_log()
        status.value = 'load a raw slice and click cells (no registration needed yet)'
        summary = Label(value='')
        summary.native.setWordWrap(True)
        summary.native.setStyleSheet('font-family: Menlo, Consolas, monospace; font-size: 12px;')
        slice_lbl = Label(value='no slice loaded')
        slice_lbl.native.setWordWrap(True)

        def set_crosses(yx_list, colors=None):
            # programmatic refresh of the layer; guarded so it doesn't re-enter sync_from_layer
            state['loading'] = True
            roi_layer.selected_data = set()  # drop any selection; stale indices crash on data swap
            roi_layer.data = np.array(yx_list) if yx_list else np.empty((0, 2))
            if yx_list and colors is not None:
                roi_layer.face_color = colors  # per-point (per-channel) colors
            state['loading'] = False

        def sync_from_layer(*_):
            # user added/removed points in raw add mode; resync this (slice, channel) subset only
            if state.get('loading') or state['verify'] or state['name'] is None:
                return
            stem, ch = state['name'], channel_w.value
            state['rois'] = {i: r for i, r in state['rois'].items()
                             if not (r['slice'] == stem and r.get('channel', 'merge') == ch)}
            for y, x in roi_layer.data:
                rid = state['next_id']
                state['next_id'] += 1
                state['rois'][rid] = {'slice': stem, 'raw': (float(y), float(x)), 'channel': ch}
            refresh_summary()
            status.value = f'{stem} [{ch}]: {len(roi_layer.data)} ROI(s)'

        roi_layer.events.data.connect(sync_from_layer)
        viewer.mouse_move_callbacks.append(self.make_hover(viewer, state, bg.structures, highlight_layer))

        def _slice_rois(channel=None):
            # ROIs on the current slice; channel=None -> all channels, else just that channel
            return [r for r in state['rois'].values() if r['slice'] == state['name']
                    and (channel is None or r.get('channel', 'merge') == channel)]

        _CH = {'R': 0, 'G': 1, 'B': 2}

        def channel_view(img):
            # show one channel as grayscale (easier to see cells in a merge); coords are unchanged
            c = channel_w.value
            if c == 'merge' or img.ndim != 3:
                return img
            idx = _CH.get(c, 0)
            return img[..., idx] if idx < img.shape[2] else img

        def display_raw():
            # annotation view: native raw image, click to add ROIs in the active channel's set
            ch = channel_w.value
            state['ann_img'] = None
            set_raw(channel_view(state['raw_img']))
            highlight_layer.visible = bound_layer.visible = False
            rois = _slice_rois(ch)
            color = _CH_COLOR.get(ch, 'cyan')
            set_crosses([r['raw'] for r in rois], [color] * len(rois))
            roi_layer.current_face_color = color  # newly clicked ROIs take the channel color
            reorder()
            select_roi_layer()
            viewer.camera.mouse_pan = pan_w.value
            status.value = (f'{state["path"].name} [{ch}]: click to add cells (scroll=zoom) — '
                            f'{len(rois)} in this channel')

        def display_warped():
            # verify view: transformed (atlas-aligned) image + boundaries + projected ROI crosses
            meta = self._transform_for(state['name'])
            if meta is None:
                status.value = f'no registration for {state["name"]} — register it first'
                verify_w.value = False  # handler switches back to the raw view
                return
            plane, res = meta['plane'], int(meta['resolution'])
            idx, dw, dh = int(meta['slice_index']), int(meta['dw']), int(meta['dh'])
            oimg = rotate(read_oriented(state['path'], meta.get('flip_lr', False),
                                        meta.get('flip_ud', False)), float(meta.get('rotate', 0.0)))
            _, warped = slice_transform_helper(oimg, np.array(meta['matrix'], float), plane_type=plane)
            od = lambda v: v + 1 if v != 0 else 0
            view, ann = get_views(plane, res)
            ann_img = ann.plane_at(idx).with_offset(od(dw), od(dh)).image
            state['ann_img'], state['hover_id'] = ann_img, None
            set_raw(warped)
            bound_layer.data = boundary_mask(ann_img)
            highlight_layer.data = np.zeros(ann_img.shape, dtype=float)
            highlight_layer.visible = bound_layer.visible = True

            rois, shape = _slice_rois(), state['shapes'].get(state['name'])  # all channels, colored per channel
            here, cols = [], []
            if rois and shape is not None:  # raw (y,x) -> atlas (x,y); cross goes at (y_atlas, x_atlas)
                atlas = raw_points_to_atlas([[r['raw'][1], r['raw'][0]] for r in rois],
                                            matrix=np.array(meta['matrix'], float), raw_shape=shape,
                                            plane=plane, rotate_deg=float(meta.get('rotate', 0.0)),
                                            flip_lr=meta.get('flip_lr', False), flip_ud=meta.get('flip_ud', False))
                here = [[ay, ax] for ax, ay in atlas]
                cols = [_CH_COLOR.get(r.get('channel', 'merge'), 'cyan') for r in rois]
            set_crosses(here, cols)
            roi_layer.mode = 'pan_zoom'  # atlas space is read-only
            reorder()
            viewer.camera.mouse_pan = True
            status.value = f'{state["name"]}: {len(here)} ROI(s) on the transformed slice (verify)'

        def display_slice():
            (display_warped if state['verify'] else display_raw)()

        def load_slice(img: Path):
            state['path'], state['name'] = img, img.stem
            state['raw_img'] = read_oriented(img, flip_lr=False, flip_ud=False)  # native; flips at projection
            state['shapes'][img.stem] = state['raw_img'].shape[:2]
            files = state['files']
            pos = f'{state["cursor"] + 1}/{len(files)}  ' if files else ''
            slice_lbl.value = f'{pos}{img.name}'
            display_slice()
            viewer.reset_view()

        def refresh_summary():
            if not state['rois']:
                summary.value = 'no ROIs yet'
                return
            per_ch = defaultdict(int)
            for r in state['rois'].values():
                per_ch[r.get('channel', 'merge')] += 1
            chans = '  '.join(f'{c}:{per_ch[c]}' for c in sorted(per_ch))
            summary.value = f'<b>total {len(state["rois"])}</b><br>{chans}'

        def remove_last():
            if state['verify']:
                status.value = 'switch off Verify to edit ROIs'
                return
            if state['name'] is None or len(roi_layer.data) == 0:
                status.value = 'no ROI on this slice to undo'
                return
            roi_layer.data = roi_layer.data[:-1]  # triggers sync_from_layer to resync state

        regions = RegionPicker(bg.structures)
        style_w = ComboBox(label='style', choices=SHADER_STYLES, value=SHADER_STYLES[0])
        hemisphere_w = ComboBox(label='hemisphere', choices=['both', 'left', 'right'], value='both')
        no_root_w = CheckBox(label='no root (hide brain)', value=False)
        color_w = ComboBox(label='ROI color', choices=_COLORS,
                           value=self.roi_color if self.roi_color in _COLORS else _COLORS[0])

        def rows_from_state() -> list[dict]:
            out = []
            for r in state['rois'].values():
                h, w = state['shapes'].get(r['slice'], (None, None))
                y, x = r['raw']
                out.append({'slice': r['slice'], 'x': x, 'y': y, 'raw_h': h, 'raw_w': w,
                            'channel': r.get('channel', 'merge')})
            return out

        def save_csv() -> Path | None:
            if not state['rois']:
                status.value = 'no ROIs to save'
                return None
            self._out.parent.mkdir(parents=True, exist_ok=True)
            pl.DataFrame(rows_from_state()).write_csv(self._out)
            print_save(self._out)
            status.value = f'saved {len(state["rois"])} raw ROI(s) -> {self._out.name}'
            return self._out

        def load_csv_points(path: Path):
            try:
                df = pl.read_csv(path)
            except Exception as e:  # noqa: BLE001 - surface any read error in the GUI
                status.value = f'could not read {path.name}: {e}'
                return
            if not set(RAW_COLS) <= set(df.columns):
                status.value = f'{path.name}: not a raw-ROI csv (needs {", ".join(RAW_COLS)})'
                return
            for r in df.iter_rows(named=True):
                rid = state['next_id']
                state['next_id'] += 1
                state['rois'][rid] = {'slice': r['slice'], 'raw': (r['y'], r['x']),
                                      'channel': r.get('channel', 'merge')}  # back-compat: old csvs -> merge
                state['shapes'].setdefault(r['slice'], (r['raw_h'], r['raw_w']))
            refresh_summary()
            if state['name'] is not None:
                display_slice()  # refresh crosses in whichever view is active
            status.value = f'loaded {len(df)} raw ROI(s) from {path.name}'

        def on_project():
            save_csv()  # persist raw coords first
            res = self._project_and_save(rows_from_state(), get_views, bg.structures, status)
            if res is not None:
                path, ccf_rows = res
                ch_files = write_channel_csvs(ccf_rows, path)  # one colored cloud per channel
                ch_color = {**_CH_COLOR, 'merge': color_w.value}
                self.launch_render(
                    _render_argv(ch_files, self.roi_radius, ch_color, regions.colors,
                                 style_w.value, no_root_w.value, hemisphere_w.value),
                    status, f'rendering {len(ccf_rows)} ROI(s) across {len(ch_files)} channel(s)...')

        def step(delta: int):
            files = state['files']
            if not files:
                status.value = 'single-image mode — nothing to step'
                return
            state['cursor'] = int(np.clip(state['cursor'] + delta, 0, len(files) - 1))
            load_slice(files[state['cursor']])

        pan_w = CheckBox(label='Pan (left-drag moves slice)', value=False)
        pan_w.changed.connect(lambda *_: setattr(viewer.camera, 'mouse_pan', pan_w.value)
                              if not state['verify'] else None)
        verify_w = CheckBox(label='Verify (warped + atlas)', value=False)
        channel_w = ComboBox(label='channel', choices=['merge', 'R', 'G', 'B'], value='merge')
        channel_w.changed.connect(lambda *_: display_raw()
                                  if state['name'] is not None and not state['verify'] else None)

        def on_verify(*_):
            state['verify'] = verify_w.value
            if state['name'] is not None:
                display_slice()
                viewer.reset_view()

        verify_w.changed.connect(on_verify)

        prev_btn = PushButton(text='◀ Prev slice')
        prev_btn.changed.connect(lambda *_: step(-1))
        next_btn = PushButton(text='Next slice ▶')
        next_btn.changed.connect(lambda *_: step(+1))
        undo_btn = PushButton(text='Undo last ROI')
        undo_btn.changed.connect(lambda *_: remove_last())

        def on_load_csv():
            from qtpy.QtWidgets import QFileDialog
            path, _ = QFileDialog.getOpenFileName(caption='Load raw-ROI CSV',
                                                  filter='CSV (*.csv);;All files (*)')
            if path:
                load_csv_points(Path(path))

        load_btn = PushButton(text='Load CSV')
        load_btn.changed.connect(lambda *_: on_load_csv())
        save_btn = PushButton(text='Save CSV')
        save_btn.changed.connect(lambda *_: save_csv())
        project_btn = PushButton(text='Project + Render (needs registration)')
        project_btn.changed.connect(lambda *_: on_project())

        panel = Container(widgets=[
            self.header('Slice'), slice_lbl, self.srow(prev_btn, next_btn), verify_w,
            self.header('ROIs'), summary, undo_btn, pan_w, channel_w, color_w,
            self.header('Regions'), *regions.widgets,
            self.header('Render'), self.srow(style_w, hemisphere_w), no_root_w, project_btn,
            self.header('CSV'), self.srow(load_btn, save_btn),
            status_label,
        ])
        self.add_scroll_dock(viewer, panel, 'roi')

        if state['files']:
            load_slice(state['files'][0])
        elif self.raw_image:
            load_slice(self.raw_image)
        if self._out.exists():
            load_csv_points(self._out)  # auto-resume the session csv
        refresh_summary()

        fprint('roi: click cells on raw slices, Save, then Project + Render after registering')
        napari.run()


if __name__ == '__main__':
    RoiOptions().main()
