"""Unit test for regrender.main_roi.project_raw_rois (the raw-px -> CCF projection wiring),
stubbing the atlas views so it runs without loading Allen volumes."""
import numpy as np
import polars as pl

from neuralib.atlas.ccf.matrix import SLICE_DIMENSION_10um

from regrender.main_roi import project_raw_rois, write_channel_csvs


class _Plane:
    def __init__(self, off, img):
        self._off, self._img = off, img

    def with_offset(self, a, b):
        return self

    @property
    def plane_offset(self):
        return self._off

    @property
    def image(self):
        return self._img


class _View:
    project_index = (0, 2, 1)
    resolution = 10

    def __init__(self, off, img):
        self._p = _Plane(off, img)

    def plane_at(self, idx):
        return self._p


def _meta():
    return {'matrix': np.eye(3).tolist(), 'plane': 'coronal', 'resolution': 10,
            'slice_index': 0, 'dw': 0, 'dh': 0}


def test_project_raw_rois_basic():
    w, h = SLICE_DIMENSION_10um['coronal']          # raw_shape == dim -> resize identity
    off = np.full((h, w), 7.0)                       # every pixel on voxel-plane 7
    ann = np.zeros((h, w), dtype=int)
    ann[20, 10] = 5                                  # region id 5 at the clicked pixel
    view = _View(off, ann)
    structures = {5: {'acronym': 'X', 'name': 'Xname'}}

    rows = [{'slice': 's1', 'x': 10, 'y': 20, 'raw_h': h, 'raw_w': w, 'channel': 'R'}]
    ccf, missing = project_raw_rois(rows, lambda p, r: (view, view), structures,
                                    lambda stem: _meta())
    assert missing == []
    assert len(ccf) == 1
    r = ccf[0]
    assert r['region'] == 'X' and r['source'] == 's1' and r['channel'] == 'R'
    assert {'AP_location', 'DV_location', 'ML_location'} <= set(r)


def test_write_channel_csvs(tmp_path):
    rows = [{'AP_location': 0.1, 'DV_location': 0.2, 'ML_location': 0.3, 'region': 'X',
             'source': 's1', 'channel': ch} for ch in ('R', 'R', 'G')]
    out = write_channel_csvs(rows, tmp_path)
    assert [c for c, _ in out] == ['G', 'R']  # sorted, one file per channel
    by_ch = dict(out)
    assert by_ch['R'].name == 'roi_ccf_R.csv'
    assert pl.read_csv(by_ch['R']).height == 2 and pl.read_csv(by_ch['G']).height == 1
    assert 'channel' not in pl.read_csv(by_ch['R']).columns  # per-file csv is plain AP/DV/ML+region+source


def test_project_raw_rois_missing_transform():
    rows = [{'slice': 's9', 'x': 1, 'y': 1, 'raw_h': 100, 'raw_w': 100}]
    ccf, missing = project_raw_rois(rows, lambda p, r: None, {}, lambda stem: None)
    assert ccf == [] and missing == ['s9']
