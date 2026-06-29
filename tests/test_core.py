"""Unit tests for ccf2d.core (pure, GUI-free helpers)."""
import json

import cv2
import imageio.v3 as iio
import numpy as np
import pytest

from ccf2d import core


# --- transform -------------------------------------------------------------

def test_estimate_transform_projective_roundtrip():
    slice_xy = np.array([[10, 10], [100, 12], [12, 90], [105, 95], [60, 50]], float)
    m_true = np.array([[1.1, 0.05, 5.0], [0.02, 0.95, -3.0], [1e-4, 0.0, 1.0]])
    atlas_xy = cv2.perspectiveTransform(slice_xy.reshape(-1, 1, 2), m_true).reshape(-1, 2)
    m = core.estimate_transform(slice_xy, atlas_xy)
    mapped = cv2.perspectiveTransform(slice_xy.reshape(-1, 1, 2), m).reshape(-1, 2)
    assert np.allclose(mapped, atlas_xy, atol=1e-3)


def test_estimate_transform_affine_shape():
    slice_xy = np.array([[0, 0], [10, 0], [0, 10]], float)
    atlas_xy = np.array([[1, 1], [11, 1], [1, 11]], float)
    m = core.estimate_transform(slice_xy, atlas_xy, affine=True)
    assert m.shape == (3, 3) and np.allclose(m[2], [0, 0, 1])


def test_estimate_transform_too_few_points():
    pts = np.zeros((3, 2))
    with pytest.raises(ValueError):
        core.estimate_transform(pts, pts)  # projective needs >=4


# --- image preprocessing ---------------------------------------------------

def test_read_oriented_channel_first(tmp_path):
    chw = np.zeros((3, 40, 50), dtype=np.uint8)  # (C, H, W)
    p = tmp_path / 'chw.tif'
    iio.imwrite(p, chw)
    out = core.read_oriented(p)
    assert out.shape == (40, 50, 3)  # normalized to (H, W, C)


def test_read_oriented_flips(tmp_path):
    img = np.arange(12, dtype=np.uint8).reshape(3, 4)
    p = tmp_path / 'g.tif'
    iio.imwrite(p, img)
    assert np.array_equal(core.read_oriented(p, flip_lr=True), np.fliplr(img))
    assert np.array_equal(core.read_oriented(p, flip_ud=True), np.flipud(img))


def test_to_uint8_contrast_window():
    img = np.array([[0, 100, 200, 1000]], dtype=float)
    out = core.to_uint8(img, contrast=(100.0, 200.0))
    assert out.dtype == np.uint8
    assert out[0, 0] == 0 and out[0, 1] == 0 and out[0, 2] == 255 and out[0, 3] == 255


# --- atlas annotation ------------------------------------------------------

def test_boundary_mask():
    ann = np.zeros((4, 4), dtype=int)
    ann[:, 2:] = 1  # vertical edge between col 1 and 2
    b = core.boundary_mask(ann)
    assert b[0, 1] == 1 and b[1, 1] == 1   # edge marked on the left side of the change
    assert b[0, 0] == 0                    # interior not marked


def test_region_name():
    ann = np.array([[0, 315], [997, 0]])
    structures = {315: {'acronym': 'Isocortex', 'name': 'Isocortex'},
                  997: {'acronym': 'root', 'name': 'root'}}
    assert core.region_name(ann, structures, 0, 1) == 'Isocortex — Isocortex'
    assert core.region_name(ann, structures, 0, 0) == ''        # id 0 -> background
    assert core.region_name(ann, structures, 99, 99) == ''      # out of bounds
    assert core.region_name(ann, {}, 1, 0) == 'id 997'          # unknown id


# --- save/load -------------------------------------------------------------

def test_save_transform_roundtrip(tmp_path):
    m = np.eye(3) * [1.0, 2.0, 1.0]
    js = core.save_transform(m, output_dir=tmp_path, name='t', plane='coronal', resolution=10,
                             slice_index=540, dw=1, dh=-2,
                             slice_xy=np.zeros((4, 2)), atlas_xy=np.ones((4, 2)),
                             rotate=12.5, flip_lr=True, contrast=(20.0, 800.0))
    assert js.name == 't_transform.json'
    assert not list(tmp_path.glob('*.npy'))  # single-file output
    meta = json.loads(js.read_text())
    assert np.allclose(np.array(meta['matrix']), m)
    assert meta['rotate'] == 12.5 and meta['flip_lr'] is True
    assert meta['contrast'] == [20.0, 800.0] and meta['slice_index'] == 540
