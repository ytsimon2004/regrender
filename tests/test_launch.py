"""Smoke test: every GUI subcommand must build its napari window and dock without crashing.

Catches the "pip install works but the app won't open" class of breakage (missing Qt
binding, napari/magicgui/qtpy API changes). ``napari.run`` is stubbed so no event loop
blocks, and the 6.7 GB 10 µm atlas is swapped for the ~MB 100 µm one, since only
the window construction is under test, not the anatomy.

Needs a display with OpenGL (xvfb on Linux CI; the native platform on macOS, where
``QT_QPA_PLATFORM=offscreen`` has no GL and segfaults). Run with ``pytest -m gui``.
"""
import sys

import pytest

napari = pytest.importorskip('napari')

pytestmark = pytest.mark.gui

SMALL_ATLAS = 'allen_mouse_100um'


@pytest.fixture
def small_atlas(monkeypatch):
    """redirect any allen_mouse_* atlas request to the small 100 µm one"""
    from brainglobe_atlasapi import BrainGlobeAtlas

    orig_init = BrainGlobeAtlas.__init__

    def init(self, atlas_name, *args, **kwargs):
        if atlas_name.startswith('allen_mouse_'):
            atlas_name = SMALL_ATLAS
        orig_init(self, atlas_name, *args, **kwargs)

    monkeypatch.setattr(BrainGlobeAtlas, '__init__', init)
    # neuralib asserts the exact 10 µm volume shape; meaningless for the downsized atlas
    from neuralib.atlas.view import AbstractSliceView
    monkeypatch.setattr(AbstractSliceView, '_check_attrs', lambda self: None)


@pytest.mark.parametrize('command', ['register', 'probe', 'roi'])
def test_launch(command, small_atlas, monkeypatch, tmp_path):
    ran = []
    monkeypatch.setattr(napari, 'run', lambda *a, **k: ran.append(True))
    monkeypatch.chdir(tmp_path)  # any default output/session paths land in the tmp dir
    monkeypatch.setattr(sys, 'argv', ['regrender', command])

    from regrender.__main__ import main

    try:
        main()
        viewer = napari.current_viewer()
        assert ran, f'{command}: never reached napari.run()'
        assert viewer is not None
        assert command in viewer.window.dock_widgets, f'{command}: dock widget missing'
    finally:
        if (v := napari.current_viewer()) is not None:
            v.close()
