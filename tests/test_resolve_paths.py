import pytest

from regrender.main_probe import ProbeOptions


def _make(dirpath, all_stems, registered_stems):
    (dirpath / 'transformations').mkdir()
    for s in all_stems:
        (dirpath / f'{s}.tif').write_bytes(b'')
    for s in registered_stems:
        (dirpath / 'transformations' / f'{s}_transform.json').write_text('{}')


def _opts(**kw):
    o = ProbeOptions()
    o.directory = kw.get('directory')
    o.raw_image = kw.get('raw_image')
    o.transform_dir = None
    o.output = None
    return o


def test_folder_skips_unregistered_and_keeps_registered(tmp_path):
    _make(tmp_path, all_stems=['a', 'b', 'c'], registered_stems=['b', 'c'])  # first is unregistered
    o = _opts(directory=tmp_path)
    files = o._resolve_paths('probe/probe_shanks.csv')
    assert sorted(f.stem for f in files) == ['b', 'c']
    assert o.raw_image.stem == 'b'  # starts on a registered slice, not the missing 'a'


def test_folder_none_registered_raises(tmp_path):
    _make(tmp_path, all_stems=['a', 'b'], registered_stems=[])
    with pytest.raises(FileNotFoundError):
        _opts(directory=tmp_path)._resolve_paths('probe/probe_shanks.csv')


def test_single_unregistered_image_raises(tmp_path):
    _make(tmp_path, all_stems=['a'], registered_stems=[])
    with pytest.raises(FileNotFoundError):
        _opts(raw_image=tmp_path / 'a.tif')._resolve_paths('probe/probe_shanks.csv')
