from __future__ import annotations

from pathlib import Path

import cv2
import imageio.v3 as iio
import numpy as np
from argclz import AbstractParser, argument
from brainglobe_atlasapi import BrainGlobeAtlas
from neuralib.atlas.ccf.matrix import SLICE_DIMENSION_10um
from neuralib.atlas.typing import PLANE_TYPE
from neuralib.atlas.view import get_slice_view
from neuralib.imglib.transform import apply_transformation
from neuralib.util.verbose import fprint, print_save

from regrender.core import (TerminalLog, boundary_mask, estimate_transform, load_transform, read_oriented,
                        region_name, rotate, save_transform, to_uint8)

__all__ = ['RegisterOptions']


class RegisterOptions(AbstractParser):
    DESCRIPTION = 'Interactively register a histology slice to the Allen CCF (napari)'

    raw_image: Path | None = argument(
        '-I', '--image',
        default=None,
        help='histology image path (optional; can also load it from the GUI)'
    )

    cut_plane: PLANE_TYPE = argument(
        '-P', '--plane-type',
        default='coronal',
        help='cutting orientation'
    )

    resolution: int = argument('--resolution', default=10, help='atlas resolution (um)')

    output_dir: Path | None = argument(
        '-O', '--output-dir',
        default=None,
        help='output directory (default: <image-dir>/transformations)'
    )

    name: str | None = argument('--name', default=None, help='output name (default: image stem)')

    flip_lr: bool = argument('--flip-lr', help='flip histology left-right before registration')
    flip_ud: bool = argument('--flip-ud', help='flip histology up-down before registration')
    affine: bool = argument('--affine', help='use affine instead of projective transform')
    boundary_color: str = argument(
        '--boundary-color',
        default='orange',
        help='annotation boundary overlay color (matplotlib name or #hex)'
    )
    load: Path | None = argument(
        '--load',
        default=None,
        help='resume from a saved *_transform.json (restores points + index/dw/dh/rotate/flips)'
    )
    directory: Path | None = argument(
        '-D', '--directory',
        default=None,
        help='folder of serial sections; step through them with Prev/Next in the GUI'
    )

    _IMG_EXT = {'.tif', '.tiff', '.png', '.jpg', '.jpeg'}

    def _list_images(self, d: Path) -> list[Path]:
        return sorted(p for p in Path(d).iterdir() if p.suffix.lower() in self._IMG_EXT)

    def run(self):
        if self.cut_plane not in SLICE_DIMENSION_10um:
            raise ValueError(f'plane {self.cut_plane!r} not supported yet '
                             f'(available: {list(SLICE_DIMENSION_10um)})')

        load = load_transform(self.load) if self.load else None
        if load:  # resume: preprocessing must match the saved session
            self.flip_lr = load.get('flip_lr', self.flip_lr)
            self.flip_ud = load.get('flip_ud', self.flip_ud)
            if self.raw_image is None:
                raise ValueError('--load needs the matching image via -I/--image')

        files = self._list_images(self.directory) if self.directory else []
        if files and self.raw_image is None:
            self.raw_image = files[0]

        base = self.raw_image.parent if self.raw_image else Path.cwd()
        out_dir = self.output_dir or base / 'transformations'
        name = self.name or (self.raw_image.stem if self.raw_image else 'untitled')

        view = get_slice_view('reference', self.cut_plane, resolution=self.resolution, check_latest=False)
        atlas_w = int(view.width)

        self._oriented = self._read_oriented(self.raw_image)  # None until an image is loaded
        self._launch_napari(view, atlas_w, out_dir, name, load, files)

    def _read_oriented(self, path: Path | None) -> np.ndarray | None:
        """read an image with the current flips (pre-rotate, pre-resize), or None"""
        return None if path is None else read_oriented(path, self.flip_lr, self.flip_ud)

    # (width, height) anatomical axes per plane, for tilt labels
    AXIS = {'coronal': ('ML', 'DV'), 'sagittal': ('AP', 'DV')}

    def _launch_napari(self, ref_view, atlas_w: int, out_dir: Path, name: str,
                       load: dict | None = None, files: list[Path] | None = None):
        import napari
        from magicgui.widgets import CheckBox, ComboBox, Container, Label, PushButton, SpinBox

        load = load or {}
        dim = SLICE_DIMENSION_10um[self.cut_plane]

        def make_hist(angle: float) -> np.ndarray:
            if self._oriented is None:
                return np.zeros((dim[1], dim[0]))  # placeholder until an image is loaded
            return cv2.resize(rotate(self._oriented, angle), dim)

        angle0 = float(load.get('rotate', 0.0))
        state = {'index': int(load.get('slice_index', int(ref_view.n_planes) // 2)),
                 'dw': int(load.get('dw', 0)), 'dh': int(load.get('dh', 0)),
                 'expect': 'atlas', 'ann': None, 'hist': make_hist(angle0),
                 'name': name, 'out_dir': out_dir, 'path': self.raw_image,
                 'files': files or [], 'cursor': 0}
        if state['files'] and self.raw_image in state['files']:
            state['cursor'] = state['files'].index(self.raw_image)

        ann_view = get_slice_view('annotation', self.cut_plane, resolution=self.resolution, check_latest=False)
        structures = BrainGlobeAtlas(f'allen_mouse_{self.resolution}um', check_latest=False).structures

        def plane_image() -> np.ndarray:
            i, dw, dh = state['index'], state['dw'], state['dh']
            od = lambda v: v + 1 if v != 0 else 0
            state['ann'] = ann_view.plane_at(i).with_offset(od(dw), od(dh)).image
            ref_plane = ref_view.plane_at(i).with_offset(od(dw), od(dh))
            state['ref_mm'] = ref_plane.reference_value
            return ref_plane.image

        def pts_kw(color):
            return dict(face_color=color, border_color=color, symbol='cross', size=20, ndim=2,
                        features={'n': np.empty(0, dtype='<U3')},
                        text={'string': '{n}', 'color': color, 'size': 12, 'translation': [-12, 0]})

        from matplotlib.colors import to_rgb
        from napari.utils import Colormap
        brgb = to_rgb(self.boundary_color)  # (r, g, b) in 0..1
        state['brgb'] = brgb
        bcmap = Colormap([[0, 0, 0], list(brgb)], name='boundary')  # 0 -> transparent (additive), 1 -> color

        viewer = napari.Viewer(title=f'regrender register — {name}')
        viewer.text_overlay.visible = True
        viewer.text_overlay.font_size = 18
        viewer.text_overlay.color = 'yellow'
        atlas_layer = viewer.add_image(plane_image(), name='atlas', colormap='gray')
        bound_layer = viewer.add_image(boundary_mask(state['ann']), name='boundaries',
                                       colormap=bcmap, blending='additive', opacity=0.9)
        # filled overlay of the atlas region under the cursor (updated in on_move)
        hcmap = Colormap([[0, 0, 0], [0.3, 0.6, 1.0]], name='highlight')
        highlight_layer = viewer.add_image(np.zeros(state['ann'].shape, dtype=float),
                                           name='region_highlight', colormap=hcmap,
                                           blending='additive', opacity=0.4)
        state['hover_id'] = None
        hist_layer = viewer.add_image(state['hist'], name='histology', translate=(0, atlas_w))

        # reference xy grid over the histology (toggled off), 100 px spacing. drawn as an
        # image on the pixel grid (nearest) so it scales with the slice and is zoom-stable
        gw, gh = dim
        step = 100
        grid_img = np.zeros((gh, gw), dtype=float)
        grid_img[::step, :] = 1
        grid_img[:, ::step] = 1
        grid_img[-1, :] = grid_img[:, -1] = 1
        grid_layer = viewer.add_image(grid_img, name='xy_grid', translate=(0, atlas_w),
                                      colormap='gray', blending='additive', opacity=0.5,
                                      interpolation2d='nearest')
        grid_layer.visible = False

        atlas_pts = viewer.add_points(name='atlas_pts', **pts_kw('red'))
        slice_pts = viewer.add_points(name='slice_pts', **pts_kw('cyan'))

        def set_histology(img: np.ndarray):
            # re-create the layer so napari re-detects rgb/ndim (grayscale<->RGB safely), keep its position
            nonlocal hist_layer
            idx = viewer.layers.index(hist_layer)
            viewer.layers.remove(hist_layer)
            hist_layer = viewer.add_image(img, name='histology', translate=(0, atlas_w))
            viewer.layers.move(len(viewer.layers) - 1, idx)

        def renumber(layer):
            layer.features = {'n': np.array([str(i + 1) for i in range(len(layer.data))], dtype='<U3')}

        def add_pt(layer, pos):
            layer.data = np.vstack([layer.data, pos]) if len(layer.data) else np.array([pos])
            renumber(layer)
            sync_orient_lock()

        @viewer.mouse_drag_callbacks.append
        def on_click(_v, event):
            dragged = False
            yield
            while event.type == 'mouse_move':
                dragged = True
                yield
            if dragged or not pick_w.value:
                return
            y, x = event.position  # world coords (row, col)
            n = len(atlas_pts.data) + 1
            if state['expect'] == 'atlas':
                if x >= atlas_w:
                    status.value = f'expected an ATLAS click (left side) for pt {n}'
                    return
                add_pt(atlas_pts, (y, x))
                state['expect'] = 'slice'
                status.value = f'now click the matching point on the histology (slice pt {n})'
            else:
                if x < atlas_w:
                    status.value = f'expected a SLICE click (right side) for pt {len(slice_pts.data) + 1}'
                    return
                add_pt(slice_pts, (y, x))
                state['expect'] = 'atlas'
                status.value = f'pair {len(slice_pts.data)} set — click atlas landmark {len(slice_pts.data) + 1}'

        w_axis, h_axis = self.AXIS.get(self.cut_plane, ('w', 'h'))
        res = self.resolution
        plane_w = ComboBox(label='plane', choices=list(SLICE_DIMENSION_10um), value=self.cut_plane)
        idx_w = SpinBox(label='slice index (voxel)', value=state['index'], min=0, max=int(ref_view.n_planes) - 1)
        dw_w = SpinBox(label=f'dw / {w_axis} tilt (voxel)', value=state['dw'], min=-200, max=200)
        dh_w = SpinBox(label=f'dh / {h_axis} tilt (voxel)', value=state['dh'], min=-200, max=200)
        rot_w = SpinBox(label='rotate (deg)', value=angle0, min=-180, max=180)
        flip_lr_w = CheckBox(label='flip L-R', value=self.flip_lr)
        flip_ud_w = CheckBox(label='flip U-D', value=self.flip_ud)
        pick_w = CheckBox(label='pick points', value=True)
        grid_w = CheckBox(label='xy grid', value=False)
        grid_w.changed.connect(lambda *_: setattr(grid_layer, 'visible', grid_w.value))

        colors = ['orange', 'red', 'cyan', 'yellow', 'magenta', 'lime', 'white', 'blue']
        if self.boundary_color not in colors:
            colors = [self.boundary_color] + colors
        color_w = ComboBox(label='boundary color', choices=colors, value=self.boundary_color)

        def set_bcolor(*_):
            state['brgb'] = to_rgb(color_w.value)
            cm = Colormap([[0, 0, 0], list(state['brgb'])], name='boundary')
            bound_layer.colormap = cm

        color_w.changed.connect(set_bcolor)

        def set_rotation(*_):
            state['hist'] = make_hist(rot_w.value)
            hist_layer.data = state['hist']
            slice_pts.data = np.empty((0, 2))  # slice points are stale once the image rotates
            renumber(slice_pts)
            state['expect'] = 'atlas' if len(atlas_pts.data) == 0 else (
                'slice' if len(atlas_pts.data) > len(slice_pts.data) else 'atlas')
            status.value = f'rotated {rot_w.value}° — re-pick the slice points'

        rot_w.changed.connect(set_rotation)

        def set_flip(*_):
            self.flip_lr, self.flip_ud = flip_lr_w.value, flip_ud_w.value
            self._oriented = self._read_oriented(state['path'])  # flips applied at read time
            state['hist'] = make_hist(rot_w.value)
            set_histology(state['hist'])
            slice_pts.data = np.empty((0, 2))  # slice points are stale once the image flips
            renumber(slice_pts)
            state['expect'] = 'atlas' if len(atlas_pts.data) == len(slice_pts.data) else 'slice'
            status.value = 'flipped — re-pick the slice points'

        flip_lr_w.changed.connect(set_flip)
        flip_ud_w.changed.connect(set_flip)

        for w in (plane_w, rot_w, flip_lr_w, flip_ud_w, idx_w, dw_w, dh_w):
            w.tooltip = 'clear points (or Re-register) to change the atlas plane / orientation'

        def sync_orient_lock():
            # plane-first: the plane, atlas index/tilt and slice orientation are fixed once any
            # point is picked, since changing them moves the plane out from under the points
            locked = bool(len(atlas_pts.data) or len(slice_pts.data))
            for w in (plane_w, rot_w, flip_lr_w, flip_ud_w, idx_w, dw_w, dh_w):
                w.enabled = not locked

        def info_text() -> str:
            return (f"index {state['index']} = {state.get('ref_mm', '?')} mm from Bregma   ·   "
                    f"dw {state['dw'] * res} µm, dh {state['dh'] * res} µm")

        info_w = Label(value=info_text())
        img_lbl = Label(value='no image loaded')  # current file (i/N) shown in the Image section
        img_lbl.native.setWordWrap(True)
        # scrolling terminal-style log: every status.value = msg appends a line (monospace)
        status_label = Label(value='')
        status_label.native.setStyleSheet(
            'font-family: Menlo, Consolas, monospace; font-size: 12px; '
            'color: #b9f27c; background: #11131a; padding: 6px;')
        status_label.native.setWordWrap(True)
        from qtpy.QtWidgets import QSizePolicy
        status_label.native.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        status = TerminalLog(status_label)
        status.value = 'click an atlas landmark (left), then its match on the slice (right)'

        def reset_highlight():
            highlight_layer.data = np.zeros(state['ann'].shape, dtype=float)
            state['hover_id'] = None

        def refresh(*_):
            state['index'], state['dw'], state['dh'] = idx_w.value, dw_w.value, dh_w.value
            atlas_layer.data = plane_image()
            bound_layer.data = boundary_mask(state['ann'])
            reset_highlight()  # annotation plane changed; stale mask
            info_w.value = info_text()

        idx_w.changed.connect(refresh)
        dw_w.changed.connect(refresh)
        dh_w.changed.connect(refresh)

        @viewer.mouse_move_callbacks.append
        def on_move(_v, event):
            y, x = event.position
            ann = state['ann']
            viewer.text_overlay.text = region_name(ann, structures, y, x)
            rid = (int(ann[int(y), int(x)])
                   if 0 <= y < ann.shape[0] and 0 <= x < ann.shape[1] else 0)
            if rid != state['hover_id']:
                state['hover_id'] = rid
                # ponytail: full-plane mask (~1e6 px), recomputed only when crossing a region edge
                highlight_layer.data = (ann == rid).astype(float) if rid else np.zeros(ann.shape)

        def collect() -> tuple[np.ndarray, np.ndarray]:
            # napari points are (row, col); convert to (x, y), un-translate the slice side
            a = atlas_pts.data[:, ::-1]
            s = (slice_pts.data - np.array([0, atlas_w]))[:, ::-1]
            return s, a

        def on_preview():
            s, a = collect()
            try:
                m = estimate_transform(s, a, affine=self.affine)
            except ValueError as e:
                status.value = f'preview failed: {e}'
                return
            # warp the histology into atlas space and overlay it on the atlas (left) side, under the
            # real atlas boundaries -> you see the transformed histology with the straight boundaries
            warped = apply_transformation(state['hist'], m)
            if 'preview_transformed' in viewer.layers:
                viewer.layers.remove('preview_transformed')  # re-create (grayscale<->RGB safe)
            pv = viewer.add_image(warped, name='preview_transformed', opacity=1.0)
            # order bottom->top: atlas, transformed histology, boundaries, points
            viewer.layers.move(viewer.layers.index(pv), viewer.layers.index(bound_layer))
            for layer in (atlas_pts, slice_pts):
                viewer.layers.move(viewer.layers.index(layer), len(viewer.layers) - 1)
            status.value = 'preview on: transformed histology under the atlas boundaries — toggle off/on to refresh after moving points'

        def on_exit_preview():
            if 'preview_transformed' in viewer.layers:
                viewer.layers.remove('preview_transformed')
            status.value = 'preview closed'

        def confirm(title: str, text: str) -> bool:
            from qtpy.QtWidgets import QMessageBox
            return QMessageBox.question(None, title, text) == QMessageBox.Yes

        def on_save():
            if self._oriented is None:
                status.value = 'load an image first'
                return
            s, a = collect()
            try:
                m = estimate_transform(s, a, affine=self.affine)
            except ValueError as e:
                status.value = f'save failed: {e}'
                return
            out_dir, name = state['out_dir'], state['name']
            js = out_dir / f'{name}_transform.json'
            if js.exists() and not confirm('Overwrite registration?',
                                           f'{js.name} already exists. Overwrite it?'):
                status.value = 'save cancelled'
                return
            contrast = tuple(float(v) for v in hist_layer.contrast_limits)
            print_save(save_transform(m, output_dir=out_dir, name=name, plane=self.cut_plane,
                                      resolution=self.resolution, slice_index=state['index'],
                                      dw=state['dw'], dh=state['dh'], slice_xy=s, atlas_xy=a,
                                      rotate=rot_w.value, flip_lr=self.flip_lr, flip_ud=self.flip_ud,
                                      contrast=contrast))

            # warped histology in atlas space, and a copy with the boundaries burned in.
            # bake the layer's contrast window so the .tif matches what you see.
            warped = to_uint8(apply_transformation(state['hist'], m), contrast)
            trans_path = out_dir / f'{name}_transformed.tif'
            iio.imwrite(trans_path, warped)
            print_save(trans_path)

            rgb = warped if warped.ndim == 3 else np.stack([warped] * 3, axis=-1)
            rgb = rgb[..., :3].copy()
            # use the color currently shown on the left atlas panel so the saved overlay matches
            bcol = np.asarray(bound_layer.colormap.colors[-1])[:3]
            rgb[boundary_mask(state['ann']).astype(bool)] = tuple(int(c * 255) for c in bcol)
            overlay_path = out_dir / f'{name}_overlay.tif'
            iio.imwrite(overlay_path, rgb)
            print_save(overlay_path)

            status.value = f'saved {name} transform (.json) + transformed/overlay .tif'

        def on_undo():
            # remove the most recently added point and restore the alternation state
            if len(atlas_pts.data) > len(slice_pts.data):
                atlas_pts.data = atlas_pts.data[:-1]
                renumber(atlas_pts)
                state['expect'] = 'atlas'
            elif len(slice_pts.data):
                slice_pts.data = slice_pts.data[:-1]
                renumber(slice_pts)
                state['expect'] = 'slice'
            sync_orient_lock()
            status.value = f'{len(slice_pts.data)} complete pair(s); next: {state["expect"]} point'

        def on_clear():
            for layer in (atlas_pts, slice_pts):
                layer.data = np.empty((0, 2))
                renumber(layer)
            state['expect'] = 'atlas'
            sync_orient_lock()
            status.value = 'cleared all points'

        def on_reregister():
            if not (len(atlas_pts.data) or len(slice_pts.data)):
                status.value = 'nothing to re-register — no points loaded'
                return
            if not confirm('Re-register this slice?',
                           'Discard the loaded points and start picking again?\n'
                           '(the saved file is kept until you Save again)'):
                return
            on_clear()  # unlocks plane / index / tilt / orientation
            status.value = 're-registering: plane & orientation unlocked — pick points again'

        def rebuild_plane(*_):
            # plane drives the atlas/annotation views, dimensions and atlas_w; rebuild them all
            nonlocal ref_view, ann_view, atlas_w, dim
            self.cut_plane = plane_w.value
            ref_view = get_slice_view('reference', self.cut_plane, resolution=self.resolution, check_latest=False)
            ann_view = get_slice_view('annotation', self.cut_plane, resolution=self.resolution, check_latest=False)
            atlas_w = int(ref_view.width)
            dim = SLICE_DIMENSION_10um[self.cut_plane]

            state['index'] = min(state['index'], ref_view.n_planes - 1)
            idx_w.max = ref_view.n_planes - 1
            idx_w.value = state['index']
            wa, ha = self.AXIS.get(self.cut_plane, ('w', 'h'))
            dw_w.label, dh_w.label = f'dw / {wa} tilt (voxel)', f'dh / {ha} tilt (voxel)'

            atlas_layer.data = plane_image()
            bound_layer.data = boundary_mask(state['ann'])
            reset_highlight()  # new plane/shape
            state['hist'] = make_hist(rot_w.value)
            set_histology(state['hist'])

            gw, gh = dim  # rebuild the grid for the new dimensions/placement
            g = np.zeros((gh, gw), dtype=float)
            g[::step, :] = 1
            g[:, ::step] = 1
            g[-1, :] = g[:, -1] = 1
            grid_layer.data = g
            grid_layer.translate = (0, atlas_w)

            on_clear()
            info_w.value = info_text()
            status.value = f'plane: {self.cut_plane}'

        plane_w.changed.connect(rebuild_plane)

        def restore_from_meta(meta: dict):
            # set widgets first (their callbacks rebuild the view / clear stale points), then points.
            # plane first: rebuild_plane resets dims/atlas_w/index, so it must run before the rest
            saved_plane = meta.get('plane')
            if saved_plane and saved_plane != plane_w.value:
                plane_w.value = saved_plane  # triggers rebuild_plane
            idx_w.value = int(meta.get('slice_index', idx_w.value))
            dw_w.value = int(meta.get('dw', 0))
            dh_w.value = int(meta.get('dh', 0))
            rot_w.value = float(meta.get('rotate', 0.0))
            flip_lr_w.value = bool(meta.get('flip_lr', False))
            flip_ud_w.value = bool(meta.get('flip_ud', False))
            ax = np.asarray(meta.get('atlas_xy', []), dtype=float).reshape(-1, 2)
            sx = np.asarray(meta.get('slice_xy', []), dtype=float).reshape(-1, 2)
            atlas_pts.data = ax[:, ::-1] if len(ax) else np.empty((0, 2))
            slice_pts.data = sx[:, ::-1] + np.array([0, atlas_w]) if len(sx) else np.empty((0, 2))
            renumber(atlas_pts)
            renumber(slice_pts)
            sync_orient_lock()
            state['expect'] = 'atlas' if len(ax) == len(sx) else 'slice'

        def load_image_path(p: Path):
            p = Path(p)
            state['path'] = p
            self._oriented = self._read_oriented(p)
            state['name'] = self.name or p.stem
            state['out_dir'] = self.output_dir or p.parent / 'transformations'
            state['hist'] = make_hist(rot_w.value)
            set_histology(state['hist'])
            on_clear()
            viewer.title = f'regrender register — {state["name"]}'
            files = state['files']
            pos = f'{files.index(p) + 1}/{len(files)}  ' if p in files else ''
            img_lbl.value = f'{pos}{p.name}'
            js = state['out_dir'] / f'{p.stem}_transform.json'
            if js.exists():
                restore_from_meta(load_transform(js))
                status.value = f'{p.name} — resumed saved registration'
            else:
                status.value = f'loaded {p.name} — pick points'

        def on_load_image():
            from qtpy.QtWidgets import QFileDialog
            path, _ = QFileDialog.getOpenFileName(
                caption='Load histology image',
                filter='Images (*.tif *.tiff *.png *.jpg *.jpeg);;All files (*)')
            if path:
                state['files'] = []  # single image: leave serial mode
                load_image_path(Path(path))

        def on_load_json():
            # reuse another slice's registration (plane/index/tilt/orientation + points) on this image
            from qtpy.QtWidgets import QFileDialog
            path, _ = QFileDialog.getOpenFileName(
                caption='Load transform JSON', filter='JSON (*.json);;All files (*)')
            if path:
                restore_from_meta(load_transform(Path(path)))
                status.value = f'loaded {Path(path).name} — Preview to check, Save to write it here'

        def on_load_dir():
            from qtpy.QtWidgets import QFileDialog
            d = QFileDialog.getExistingDirectory(caption='Load serial-section folder')
            if not d:
                return
            state['files'] = self._list_images(Path(d))
            state['cursor'] = 0
            if not state['files']:
                status.value = 'no images found in folder'
                return
            load_image_path(state['files'][0])
            status.value = f'1/{len(state["files"])}: {state["files"][0].name}'

        def load_slice(delta: int):
            files = state['files']
            if not files:
                status.value = 'not in directory mode — use Load dir'
                return
            state['cursor'] = int(np.clip(state['cursor'] + delta, 0, len(files) - 1))
            p = files[state['cursor']]
            load_image_path(p)
            status.value = f'{state["cursor"] + 1}/{len(files)}: {p.name}' + (
                ' (resumed)' if (state['out_dir'] / f'{p.stem}_transform.json').exists() else '')

        load_btn = PushButton(text='Load image')
        load_btn.changed.connect(on_load_image)
        load_dir_btn = PushButton(text='Load dir (serial)')
        load_dir_btn.changed.connect(on_load_dir)
        load_json_btn = PushButton(text='Load transform (json)')
        load_json_btn.changed.connect(on_load_json)
        prev_btn = PushButton(text='◀ Prev slice')
        prev_btn.changed.connect(lambda *_: load_slice(-1))
        next_btn = PushButton(text='Next slice ▶')
        next_btn.changed.connect(lambda *_: load_slice(+1))
        preview_w = CheckBox(label='preview overlay', value=False)
        preview_w.changed.connect(lambda *_: on_preview() if preview_w.value else on_exit_preview())
        save_btn = PushButton(text='Save transform')
        save_btn.changed.connect(on_save)
        undo_btn = PushButton(text='Undo last point')
        undo_btn.changed.connect(on_undo)
        clear_btn = PushButton(text='Clear all points')
        clear_btn.changed.connect(on_clear)
        reregister_btn = PushButton(text='Re-register (clear & redo)')
        reregister_btn.changed.connect(on_reregister)

        if load.get('slice_xy') or load.get('atlas_xy'):
            restore_from_meta(load)
            status.value = f'resumed: {len(load.get("slice_xy", []))} pair(s) loaded'
        elif state['files']:
            load_slice(0)  # show "i/N" + auto-resume the first serial section
        elif self.raw_image is not None:  # single -I: auto-resume its saved transform like directory mode
            js = out_dir / f'{name}_transform.json'
            if js.exists():
                restore_from_meta(load_transform(js))
                status.value = f'{name} — resumed saved registration'

        def header(text):
            lbl = Label(value=text)
            lbl.native.setStyleSheet('font-weight: bold; color: #88c0d0; padding-top: 6px;')
            return lbl

        def row(*ws):
            return Container(widgets=list(ws), layout='horizontal', labels=False)

        panel = Container(
            widgets=[
                header('Image'), img_lbl, load_btn, load_dir_btn, load_json_btn, row(prev_btn, next_btn),
                header('Atlas plane'), plane_w, idx_w, dw_w, dh_w, info_w,
                header('Orientation'), rot_w, flip_lr_w, flip_ud_w,
                header('Display'), grid_w, color_w,
                header('Points'), pick_w, row(undo_btn, clear_btn), reregister_btn,
                header('Overlay & save'), preview_w, save_btn,
                status_label,
            ]
        )
        panel.native.setStyleSheet('QPushButton { padding: 4px; }')
        viewer.window.add_dock_widget(
            panel, area='right', name='register'
        )
        fprint(f'registering {name}: pick points, Preview to verify, Save when done')
        napari.run()


if __name__ == '__main__':
    RegisterOptions().main()
