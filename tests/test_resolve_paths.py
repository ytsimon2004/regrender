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


@pytest.mark.parametrize('cls_path', ['regrender.main_register:RegisterOptions',
                                      'regrender.main_probe:ProbeOptions',
                                      'regrender.main_roi:RoiOptions'])
def test_list_images_natural_order(tmp_path, cls_path):
    # every app must step through a folder in the same, natural order (slice_2 before slice_10)
    import importlib
    mod, cls = cls_path.split(':')
    for name in ['slice_10.tif', 'slice_2.tif', 'slice_1.tif', 'notes.txt']:
        (tmp_path / name).write_bytes(b'')
    o = getattr(importlib.import_module(mod), cls)()
    assert [p.stem for p in o._list_images(tmp_path)] == ['slice_1', 'slice_2', 'slice_10']
