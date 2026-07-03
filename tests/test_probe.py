"""Unit tests for the probe region-profile coordinate helper (pure, GUI-free)."""
from regrender import _core as core


def test_ccf_mm_to_voxel_bregma_is_origin():
    # bregma-relative (0, 0, 0) mm must land on the bregma voxel itself
    assert core.ccf_mm_to_voxel((0.0, 0.0, 0.0), bregma_10um=(540, 0, 570)) == (540.0, 0.0, 570.0)


def test_ccf_mm_to_voxel_matches_plane_point_identity():
    # with an identity project_index, it must agree with ccf_mm_to_plane_point's voxel indices
    ccf = (1.23, 0.45, -2.0)
    got = core.ccf_mm_to_voxel(ccf, resolution=10)
    ref = core.ccf_mm_to_plane_point(ccf, project_index=(0, 1, 2), resolution=10)
    assert got == ref


def test_shank_distances_euclidean_um():
    pts = {
        (1, 'dorsal'): (0.0, 0.0, 0.0),
        (1, 'ventral'): (0.0, 3.0, 4.0),  # 5 mm -> 5000 µm
        (2, 'dorsal'): (0.0, 1.0, 0.0),   # shank 2 has no ventral -> excluded
    }
    assert core.shank_distances(pts) == [(1, 5000.0)]
