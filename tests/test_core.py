"""Unit tests for regrender.core (pure, GUI-free helpers)."""
import json

import cv2
import imageio.v3 as iio
import numpy as np
import pytest

from regrender import core


# --- transform -------------------------------------------------------------

def test_estimate_transform_projective_roundtrip():
    slice_xy = np.array([[10, 10], [100, 12], [12, 90], [105, 95], [60, 50]], float)
    m_true = np.array([[1.1, 0.05, 5.0], [0.02, 0.95, -3.0], [1e-4, 0.0, 1.0]])
    atlas_xy = cv2.perspectiveTransform(slice_xy.reshape(-1, 1, 2), m_true).reshape(-1, 2)
    m = core.estimate_transform(slice_xy, atlas_xy)
    mapped = cv2.perspectiveTransform(slice_xy.reshape(-1, 1, 2), m).reshape(-1, 2)
    assert np.allclose(mapped, atlas_xy, atol=1e-3)


def test_raw_points_to_atlas_matches_matrix():
    from neuralib.atlas.ccf.matrix import SLICE_DIMENSION_10um
    plane = 'coronal'
    w, h = SLICE_DIMENSION_10um[plane]  # raw_shape == dim -> resize is identity
    m = np.array([[1.1, 0.05, 5.0], [0.02, 0.95, -3.0], [1e-4, 0.0, 1.0]])
    pts = np.array([[10.0, 20.0], [30.0, 5.0], [100.0, 80.0]])
    expect = cv2.perspectiveTransform(pts.reshape(-1, 1, 2), m).reshape(-1, 2)
    out = core.raw_points_to_atlas(pts, matrix=m, raw_shape=(h, w), plane=plane)
    assert np.allclose(out, expect)


def test_raw_points_to_atlas_flip_lr():
    from neuralib.atlas.ccf.matrix import SLICE_DIMENSION_10um
    plane = 'coronal'
    w, h = SLICE_DIMENSION_10um[plane]
    pts = np.array([[10.0, 20.0], [30.0, 5.0]])
    out = core.raw_points_to_atlas(pts, matrix=np.eye(3), raw_shape=(h, w), plane=plane, flip_lr=True)
    assert np.allclose(out[:, 0], (w - 1) - pts[:, 0])  # x mirrored
    assert np.allclose(out[:, 1], pts[:, 1])            # y unchanged


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


# --- probe reconstruction --------------------------------------------------

def test_plane_point_to_ccf_mm_coronal_bregma():
    # coronal project_index (plane=AP, x=ML, y=DV); at the bregma plane AP must be 0
    ap, dv, ml = core.plane_point_to_ccf_mm(540, 400, 300, project_index=(0, 2, 1),
                                            resolution=10, bregma_10um=(540, 0, 570))
    assert ap == 0.0                    # on the bregma AP plane
    assert dv == 3.0                    # 300 voxel * 10µm = 3000µm below bregma DV(0)
    assert ml == pytest.approx(1.7)     # (570 - 400) * 10µm = 1700µm right of midline

    # round-trip against neuralib's forward transform
    from neuralib.atlas.util import allen_to_brainrender_coord
    out = allen_to_brainrender_coord(np.array([[ap, dv, ml]]))[0]
    assert np.allclose(out, [540 * 10, 300 * 10, 400 * 10])  # back to absolute µm voxels


def test_ccf_mm_to_plane_point_inverts():
    pi, res = (0, 2, 1), 10  # coronal
    for plane_num, x, y in [(540, 400, 300), (700, 123, 456), (200, 1000, 50)]:
        ccf = core.plane_point_to_ccf_mm(plane_num, x, y, project_index=pi, resolution=res)
        p2, x2, y2 = core.ccf_mm_to_plane_point(ccf, project_index=pi, resolution=res)
        assert np.allclose([p2, x2, y2], [plane_num, x, y])


# --- probe render command (pure GUI-free builder) --------------------------

def test_render_command_per_shank_and_region_colors():
    from pathlib import Path
    from regrender.main_probe import _render_command
    cmd = _render_command(Path('p.csv'), 'coronal', shanks=[1, 2],
                          shank_colors={1: 'red', 2: 'blue'},
                          region_colors={'VISp': 'green', 'MOp': 'cyan'},
                          depth=None, interval=None)
    assert cmd[cmd.index('--probe-color') + 1] == 'red,blue'
    assert cmd[cmd.index('--region') + 1] == 'VISp,MOp'
    assert cmd[cmd.index('--region-color') + 1] == 'green,cyan'
    assert '--dye' in cmd and '--depth' not in cmd  # dye-only when no depth


def test_render_command_theoretical_track():
    from pathlib import Path
    from regrender.main_probe import _render_command
    cmd = _render_command(Path('p.csv'), 'coronal', shanks=[1], shank_colors={},
                          region_colors={}, depth=4000, interval=250)
    assert '--dye' not in cmd
    assert cmd[cmd.index('--depth') + 1] == '4000'
    assert cmd[cmd.index('--interval') + 1] == '250'
    assert cmd[cmd.index('--probe-color') + 1] == 'red'  # default when shank has no color
    assert '--region' not in cmd
