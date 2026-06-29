import json
from pathlib import Path

import polars as pl
from argclz import AbstractParser, argument, str_tuple_type, validator
from brainglobe_atlasapi.bg_atlas import BrainGlobeAtlas
from neuralib.atlas.ccf.matrix import load_transform_matrix, SLICE_DIMENSION_10um, slice_transform_helper
from neuralib.atlas.typing import PLANE_TYPE
from neuralib.atlas.view import SlicePlane, get_slice_view
from neuralib.plot import plot_figure

__all__ = ['ViewOptions']


class ViewOptions(AbstractParser):
    DESCRIPTION = 'Options for viewing the registered images'

    GROUP_IO = 'Input/Output Options'

    raw_image: Path = argument(
        '-R', '--raw',
        validator=validator.path.is_exists(),
        group=GROUP_IO,
        help='raw image path (after resize to allen space)'
    )

    trans_matrix: Path | None = argument(
        '-T', '--trans',
        default=None,
        group=GROUP_IO,
        help='transform matrix (3 x 3) .mat file (legacy MATLAB only; native .json carries the matrix)'
    )

    ccf_data: Path = argument(
        '-C', '--ccf',
        validator=validator.path.is_exists(),
        group=GROUP_IO,
        help='registration .json (native) or ccf transformed .mat file (legacy)'
    )

    output: Path | None = argument(
        '--output',
        default=None,
        group=GROUP_IO,
        help='output image path, if not specified, show image'
    )

    # ------------------------------------
    GROUP_VIEW = 'View Options'

    cut_plane: PLANE_TYPE = argument(
        '-P', '--plane-type',
        default='coronal',
        group=GROUP_VIEW,
        help='cutting orientation',
    )

    overlay_only: bool = argument(
        '--overlay',
        group=GROUP_VIEW,
        help='only show image'
    )

    annotation_region: tuple[str, ...] | None = argument(
        '--annotation',
        type=str_tuple_type,
        default=None,
        group=GROUP_VIEW,
        help='annotation brain region'
    )

    # ------------------------------------
    GROUP_VERBOSE = 'Verbose Options'

    print_tree: bool = argument(
        '--print-tree',
        group=GROUP_VERBOSE,
        help='print tree for the available regions for the given source'
    )

    tree_init: str | None = argument(
        '--tree-init',
        default=None,
        group=GROUP_VERBOSE,
        help='init region for the tree print'
    )

    print_name: bool = argument(
        '--print-name',
        group=GROUP_VERBOSE,
        help='print acronym and the corresponding name'
    )

    _stop_render = False

    def run(self):
        self._verbose()
        if not self._stop_render:
            self._run()

    def _verbose(self):
        if self.print_tree:
            from neuralib.atlas.plot import plot_structure_tree
            plot_structure_tree(self.tree_init)
            self._stop_render = True

        if self.print_name:
            from neuralib.util.table import rich_data_frame_table
            bg = BrainGlobeAtlas('allen_mouse_10um', check_latest=True)
            file = bg.root_dir / 'structures.csv'
            df = pl.read_csv(file).select('acronym', 'name')
            rich_data_frame_table(df)
            self._stop_render = True

    def _run(self):
        if self.ccf_data.suffix == '.json':
            # reproduce the registration preprocessing: raw -> flip -> rotate -> resize -> apply
            import numpy as np
            from ccf2d.core import read_oriented, rotate
            meta = json.loads(self.ccf_data.read_text())
            img = read_oriented(self.raw_image, meta.get('flip_lr', False), meta.get('flip_ud', False))
            img = rotate(img, float(meta.get('rotate', 0.0)))
            matrix = np.array(meta['matrix'], dtype=float)
            raw, trans = slice_transform_helper(img, matrix, plane_type=self.cut_plane)
        else:
            if self.trans_matrix is None:
                raise ValueError('a .mat ccf_data (-C) needs the matrix via -T/--trans')
            raw, trans = slice_transform_helper(self.raw_image, self.trans_matrix, plane_type=self.cut_plane)

        x, y = SLICE_DIMENSION_10um[self.cut_plane]
        extent = (-x / 2, x / 2, -y / 2, y / 2)

        plane, title = self._resolve_plane()

        if self.overlay_only:
            with plot_figure(self.output) as ax:
                ax.imshow(trans, extent=extent)
                plane.plot_boundaries(ax=ax, extent=extent, cmap='binary_r', alpha=0.7)
                ax.set_title(title)

        else:
            with plot_figure(self.output, 1, 3) as ax:
                ax[0].imshow(raw)
                ax[0].set_title('resized raw')

                ax[1].imshow(trans, extent=extent)
                plane.plot_boundaries(ax=ax[1], extent=extent, cmap='binary_r', alpha=0.7)
                ax[1].set_title(title)

                regions = list(self.annotation_region) if self.annotation_region else None
                plane.plot(ax=ax[2], annotation_region=regions, extent=extent, boundaries=True)

    def _resolve_plane(self) -> tuple[SlicePlane, str]:
        """Build the reference ``SlicePlane`` + title from either the native ``.json``
        metadata (written by ``ccf2d register``) or the MATLAB ``.mat`` struct."""
        if self.ccf_data.suffix == '.json':
            meta = json.loads(self.ccf_data.read_text())
            dw, dh = int(meta['dw']), int(meta['dh'])
            plane = (get_slice_view('reference', self.cut_plane, resolution=int(meta['resolution']))
                     .plane_at(int(meta['slice_index']))
                     .with_offset(dw + 1 if dw != 0 else 0, dh + 1 if dh != 0 else 0))
            title = '\n'.join([f'{plane.reference_value}mm from Bregma',
                               f'index: {meta["slice_index"]}', f'dw: {dw}', f'dh: {dh}'])
            return plane, title

        matrix = load_transform_matrix(self.ccf_data, self.cut_plane)
        title = '\n'.join([f'{matrix.get_slice_plane().reference_value}mm from Bregma',
                           f'index: {matrix.slice_index}',
                           f'dw: {matrix.delta_xy[0]}', f'dh: {matrix.delta_xy[1]}'])
        return matrix.get_slice_plane(), title
