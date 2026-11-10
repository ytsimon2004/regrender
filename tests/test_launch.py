"""Smoke test: every GUI subcommand must build its napari window and dock without crashing.

Catches the "pip install works but the app won't open" class of breakage (missing Qt
binding, napari/magicgui/qtpy API changes). ``napari.run`` is stubbed so no event loop
blocks, and the 6.7 GB 10 µm atlas is swapped for a tiny synthetic one written to a temp
dir, since only the window construction is under test, not the anatomy. Fully offline
(the GIN atlas server rejects GitHub runners with 403).

Needs a display with OpenGL (xvfb on Linux CI; the native platform on macOS, where
``QT_QPA_PLATFORM=offscreen`` has no GL and segfaults). Run with ``pytest -m gui``.
"""
import json
import sys
from pathlib import Path

import pytest

napari = pytest.importorskip('napari')

pytestmark = [
    pytest.mark.gui,
    # deprecated napari/Qt API used from *our* code fails the test, so the weekly latest-deps
    # run flags it while the old name still works (rather than when it's finally removed)
    pytest.mark.filterwarnings('error::DeprecationWarning:regrender'),
    pytest.mark.filterwarnings('error::FutureWarning:regrender'),
]

SMALL_ATLAS = 'allen_mouse_100um'


def _write_fake_atlas(root: Path) -> None:
    """a tiny synthetic atlas in brainglobe's on-disk format: real 100 µm shape, 2 regions, random voxels"""
    import numpy as np
    import tifffile

    d = root / f'{SMALL_ATLAS}_v1.2'
    d.mkdir(parents=True)
    shape = (132, 80, 114)  # AP, DV, ML of the real allen_mouse_100um
    (d / 'metadata.json').write_text(json.dumps({
        'name': 'allen_mouse', 'citation': 'synthetic test atlas', 'atlas_link': '', 'species': 'Mus musculus',
        'symmetric': True, 'resolution': [100.0] * 3, 'orientation': 'asr', 'version': '1.2', 'shape': list(shape),
        'trasform_to_bg': np.eye(4).tolist(), 'additional_references': [],
    }))
    (d / 'structures.json').write_text(json.dumps([
        {'acronym': 'root', 'id': 997, 'name': 'root', 'structure_id_path': [997], 'rgb_triplet': [255] * 3},
        {'acronym': 'grey', 'id': 8, 'name': 'Basic cell groups and regions',
         'structure_id_path': [997, 8], 'rgb_triplet': [191, 218, 227]},
    ]))
    rng = np.random.default_rng(0)
    tifffile.imwrite(d / 'reference.tiff', rng.integers(0, 500, shape, dtype=np.uint16))
    ann = np.zeros(shape, dtype=np.uint32)
    ann[20:-20, 10:-10, 10:-10] = 997
    ann[40:-40, 20:-20, 20:-20] = 8
    tifffile.imwrite(d / 'annotation.tiff', ann)


@pytest.fixture(scope='session')
def fake_atlas_dir(tmp_path_factory) -> Path:
    root = tmp_path_factory.mktemp('brainglobe')
    _write_fake_atlas(root)
    return root


@pytest.fixture
def small_atlas(monkeypatch, fake_atlas_dir):
    """redirect any allen_mouse_* atlas request to the offline synthetic one (no GIN download)"""
    from brainglobe_atlasapi import BrainGlobeAtlas

    orig_init = BrainGlobeAtlas.__init__

    def init(self, atlas_name, *args, **kwargs):
        if atlas_name.startswith('allen_mouse_'):
            atlas_name = SMALL_ATLAS
            kwargs.update(brainglobe_dir=fake_atlas_dir, interm_download_dir=fake_atlas_dir, check_latest=False)
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
