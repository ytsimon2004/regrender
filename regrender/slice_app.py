"""Shared scaffolding for the napari reconstruction modes (``probe`` and ``roi``).

Both modes step through registered serial sections, pick points that resolve to Allen CCF
mm, paint atlas region boundaries / the region under the cursor, pick atlas regions to render,
and shell out to a brainrender CLI. That common plumbing lives here; each mode subclasses
:class:`SliceReconstructOptions` and supplies only its own image display + click semantics +
render argv. The pure (GUI-free) coordinate/image helpers stay in :mod:`regrender.core`.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
from argclz import AbstractParser, argument
from neuralib.atlas.view import get_slice_view

from regrender.core import TerminalLog, region_name

__all__ = ['SliceReconstructOptions', 'RegionPicker']

_IMG_EXT = {'.tif', '.tiff', '.png', '.jpg', '.jpeg'}


def region_table_html(region_colors: dict[str, str]) -> str:
    """HTML table of regions to render; each acronym is a ``rdel:<acr>`` link to remove it."""
    if not region_colors:
        return 'no regions'
    rows = ['<tr><th>region</th><th>color</th></tr>']
    for acr, c in region_colors.items():
        rows.append(f'<tr><td><a href="rdel:{acr}">{acr} ✕</a></td>'
                    f'<td><font color="{c}">■</font> {c}</td></tr>')
    return '<table border=1 cellspacing=0 cellpadding=3>' + ''.join(rows) + '</table>'


# atlas regions to render as meshes; identical picker in both modes
_REGION_COLORS = ['red', 'salmon', 'darkred', 'cyan', 'yellow', 'magenta',
                  'lime', 'green', 'blue', 'orange', 'white', 'black']


class RegionPicker:
    """Filterable atlas-region picker + colored render list. ``.widgets`` go in the panel;
    ``.colors`` is the ordered ``{acronym: color}`` map (render order = insertion order)."""

    def __init__(self, structures):
        from magicgui.widgets import ComboBox, Label, LineEdit, PushButton, Select
        self._pairs = [(f"{a} — {structures[a]['name']}", a)
                       for a in sorted(structures.acronym_to_id_map)]
        self.colors: dict[str, str] = {}
        self._search = LineEdit(label='filter')
        self._select = Select(label='regions', choices=self._pairs)
        self._select.native.setMinimumHeight(240)
        self._color = ComboBox(label='color', choices=_REGION_COLORS, value='red')
        self._add = PushButton(text='+ Add region(s)')
        self._table = Label(value='no regions')
        self._table.native.setOpenExternalLinks(False)

        self._search.changed.connect(self._apply_filter)
        self._add.changed.connect(self._add_regions)
        self._table.native.linkActivated.connect(self._on_link)
        self._refresh()

    @property
    def widgets(self) -> list:
        return [self._search, self._select, self._color, self._add, self._table]

    def _apply_filter(self, *_):
        q = self._search.value.strip().lower()
        self._select.choices = [p for p in self._pairs if q in p[0].lower()] if q else list(self._pairs)

    def _refresh(self):
        self._table.value = region_table_html(self.colors)

    def _add_regions(self, *_):
        for acr in self._select.value:
            self.colors[acr] = self._color.value  # add or recolor
        self._refresh()

    def _on_link(self, href: str):
        if href.startswith('rdel:'):
            self.colors.pop(href[len('rdel:'):], None)
            self._refresh()


class SliceReconstructOptions(AbstractParser):
    """Base for ``probe``/``roi``: shared CLI args + napari/render plumbing."""

    directory: Path | None = argument(
        '-D', '--directory', default=None,
        help='folder of serial sections (steps through them; reads transformations/<stem>_transform.json)'
    )

    raw_image: Path | None = argument(
        '-I', '--image', default=None, help='single registered histology image (alternative to -D)'
    )

    transform_dir: Path | None = argument(
        '--transform-dir', default=None,
        help='where the *_transform.json live (default: <image-dir>/transformations)'
    )

    output: Path | None = argument(
        '-O', '--output', default=None, help='output points csv (default: <dir>/<mode default>)'
    )

    _IMG_EXT = _IMG_EXT

    def _list_images(self, d: Path) -> list[Path]:
        return sorted(p for p in Path(d).iterdir() if p.suffix.lower() in self._IMG_EXT)

    def _transform_path(self, img: Path) -> Path:
        return self._tdir / f'{img.stem}_transform.json'

    def _resolve_paths(self, default_csv: str, *, require_transform: bool = True) -> list[Path]:
        """Common run() prologue: resolve image(s), transform dir, output csv. Returns the file list.

        ``require_transform=False`` for modes that label before registration exists (roi)."""
        files = self._list_images(self.directory) if self.directory else []
        if files and self.raw_image is None:
            self.raw_image = files[0]
        if self.raw_image is None:
            raise ValueError('provide an image via -I/--image or a folder via -D/--directory')
        base = self.raw_image.parent
        self._tdir = self.transform_dir or base / 'transformations'
        self._out = self.output or base / default_csv
        if require_transform and not self._transform_path(self.raw_image).exists():
            raise FileNotFoundError(
                f'no native registration {self._transform_path(self.raw_image)} — '
                f'register with `regrender register` first')
        return files

    # --- shared napari helpers --------------------------------------------------

    @staticmethod
    def make_status_log():
        """A scrolling terminal-style status: a monospace ``Label`` + its :class:`TerminalLog`.
        Returns ``(label_widget, TerminalLog)``; ``log.value = msg`` appends a line."""
        from magicgui.widgets import Label
        from qtpy.QtWidgets import QSizePolicy
        label = Label(value='')
        label.native.setStyleSheet(
            'font-family: Menlo, Consolas, monospace; font-size: 12px; '
            'color: #b9f27c; background: #11131a; padding: 6px;')
        label.native.setWordWrap(True)
        label.native.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)  # extends with the dock
        return label, TerminalLog(label)

    @staticmethod
    def make_view_cache():
        """``get_views(plane, res) -> (reference_view, annotation_view)``, cached (volume load is slow)."""
        views: dict = {}

        def get_views(plane: str, res: int):
            key = (plane, res)
            if key not in views:
                views[key] = (get_slice_view('reference', plane, resolution=res, check_latest=False),
                              get_slice_view('annotation', plane, resolution=res, check_latest=False))
            return views[key]

        return get_views

    @staticmethod
    def make_hover(viewer, state, structures, highlight_layer):
        """Region-name text overlay + filled highlight of the atlas region under the cursor.
        Reads ``state['ann_img']`` / ``state['hover_id']``. Returns the mouse-move callback."""
        def on_move(_v, event):
            ann = state.get('ann_img')
            if ann is None:
                return
            y, x = event.position
            viewer.text_overlay.text = region_name(ann, structures, y, x)
            rid = (int(ann[int(y), int(x)])
                   if 0 <= y < ann.shape[0] and 0 <= x < ann.shape[1] else 0)
            if rid != state.get('hover_id'):
                state['hover_id'] = rid
                highlight_layer.data = (ann == rid).astype(float) if rid else np.zeros(ann.shape)
        return on_move

    def launch_render(self, argv: list[str], status=None, msg: str = ''):
        """Shell out to a brainrender CLI in a separate window (non-blocking).

        The child inherits our terminal's stdout/stderr, so a crash is visible where regrender
        was launched. brainrender's ``Scene`` re-checks the atlas version over the network (no
        timeout, 5 retries) on every launch, stalling the render for minutes on a slow link — so
        we disable that check in the child before running the CLI."""
        if status is not None:
            status.value = msg
        module, *rest = argv
        boot = (
            'import runpy, sys, brainglobe_atlasapi.bg_atlas as b;'
            'b.BrainGlobeAtlas.check_latest_version = lambda self, *a, **k: None;'
            f'sys.argv = [{module!r}, *sys.argv[1:]];'
            f'runpy.run_module({module!r}, run_name="__main__")'
        )
        subprocess.Popen([sys.executable, '-c', boot, *rest])

    @staticmethod
    def header(text):
        from magicgui.widgets import Label
        lbl = Label(value=text)
        lbl.native.setStyleSheet('font-weight: bold; color: #88c0d0; padding-top: 6px;')
        return lbl

    @staticmethod
    def srow(*ws):
        from magicgui.widgets import Container
        return Container(widgets=list(ws), layout='horizontal', labels=False)

    @staticmethod
    def add_scroll_dock(viewer, panel, name: str):
        from qtpy.QtWidgets import QScrollArea
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)  # panel fills the dock width; scrolls vertically when tall
        scroll.setWidget(panel.native)
        scroll.setMinimumWidth(360)
        viewer.window.add_dock_widget(scroll, area='right', name=name)
