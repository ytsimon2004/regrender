"""Pure, GUI-free building blocks shared across ccf2d modes.

Everything here is side-effect-light and unit-testable without napari, so future
modes (cell annotation, probe tracks, ...) can reuse the same transform/image/atlas
helpers. The napari front-ends live in the ``main_*`` modules.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import cv2
import imageio.v3 as iio
import numpy as np
from neuralib.atlas.typing import PLANE_TYPE

__all__ = [
    'read_oriented', 'rotate', 'to_uint8', 'boundary_mask', 'region_name',
    'estimate_transform', 'save_transform',
]


# --- image preprocessing ---------------------------------------------------

def read_oriented(path: Path, flip_lr: bool = False, flip_ud: bool = False) -> np.ndarray:
    """read an image, normalize channel-first tiffs to (H, W, C), and apply flips"""
    img = iio.imread(path)
    if img.ndim == 3 and img.shape[0] in (3, 4) and img.shape[-1] not in (3, 4):
        img = np.moveaxis(img, 0, -1)  # (C, H, W) -> (H, W, C)
    if flip_ud:
        img = np.flipud(img)
    if flip_lr:
        img = np.fliplr(img)
    return img


def rotate(img: np.ndarray, deg: float) -> np.ndarray:
    """rotate about the image center, keeping the original shape"""
    if not deg:
        return img
    h, w = img.shape[:2]
    m = cv2.getRotationMatrix2D((w / 2, h / 2), deg, 1.0)
    return cv2.warpAffine(img, m, (w, h))


def to_uint8(img: np.ndarray, contrast: tuple[float, float] | None = None) -> np.ndarray:
    """map to uint8 using a (lo, hi) contrast window, else full min-max"""
    img = np.asarray(img, dtype=float)
    lo, hi = contrast if contrast is not None else (float(img.min()), float(img.max()))
    return (np.clip((img - lo) / ((hi - lo) or 1.0), 0, 1) * 255).astype(np.uint8)


# --- atlas annotation ------------------------------------------------------

def boundary_mask(ann: np.ndarray) -> np.ndarray:
    """region boundaries = pixels where the annotation id changes vs its right/down neighbour"""
    b = np.zeros(ann.shape, dtype=float)
    b[:, :-1] = np.maximum(b[:, :-1], ann[:, :-1] != ann[:, 1:])
    b[:-1, :] = np.maximum(b[:-1, :], ann[:-1, :] != ann[1:, :])
    return b


def region_name(ann: np.ndarray | None, structures: Any, y: float, x: float) -> str:
    """acronym/name of the atlas region at (y, x) in the annotation plane, '' if none"""
    if ann is None or not (0 <= y < ann.shape[0] and 0 <= x < ann.shape[1]):
        return ''
    rid = int(ann[int(y), int(x)])
    if rid == 0:
        return ''
    try:
        s = structures[rid]
        return f"{s['acronym']} — {s['name']}"
    except KeyError:
        return f'id {rid}'


# --- transform -------------------------------------------------------------

def estimate_transform(slice_xy: np.ndarray, atlas_xy: np.ndarray, *, affine: bool = False) -> np.ndarray:
    """Estimate the 3x3 matrix mapping histology (slice) points onto atlas points.

    Matches ``cv2.warpPerspective`` convention: the matrix warps the slice into atlas space.

    :param slice_xy: ``Array[float, [N, 2]]`` (x, y) points on the (resized) histology slice.
    :param atlas_xy: ``Array[float, [N, 2]]`` matched (x, y) points on the atlas plane.
    :param affine: estimate an affine (6 DOF) instead of projective (8 DOF) transform.
    :return: ``Array[float64, [3, 3]]``
    """
    slice_xy = np.asarray(slice_xy, dtype=np.float64)
    atlas_xy = np.asarray(atlas_xy, dtype=np.float64)
    if slice_xy.shape != atlas_xy.shape:
        raise ValueError(f'point count mismatch: {slice_xy.shape} vs {atlas_xy.shape}')

    if affine:
        if len(slice_xy) < 3:
            raise ValueError('affine transform needs >=3 matched point pairs')
        m, _ = cv2.estimateAffine2D(slice_xy, atlas_xy)
        return np.vstack([m, [0, 0, 1]]).astype(np.float64)
    if len(slice_xy) < 4:
        raise ValueError('projective transform needs >=4 matched point pairs')
    m, _ = cv2.findHomography(slice_xy, atlas_xy)
    return m.astype(np.float64)


def save_transform(matrix: np.ndarray, *,
                   output_dir: Path, name: str,
                   plane: PLANE_TYPE, resolution: int,
                   slice_index: int, dw: int, dh: int,
                   slice_xy: np.ndarray, atlas_xy: np.ndarray,
                   rotate: float = 0.0, flip_lr: bool = False, flip_ud: bool = False,
                   contrast: tuple[float, float] | None = None) -> Path:
    """Save the 3x3 matrix and metadata into a single ``.json``. Returns its path.

    ``rotate``/``flip_lr``/``flip_ud`` record the preprocessing so the result can be
    reproduced (raw -> flip -> rotate -> resize -> apply matrix) and the session resumed.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    meta = {
        'matrix': np.asarray(matrix, dtype=float).tolist(),
        'plane': plane,
        'resolution': resolution,
        'slice_index': int(slice_index),
        'dw': int(dw),
        'dh': int(dh),
        'rotate': float(rotate),
        'flip_lr': bool(flip_lr),
        'flip_ud': bool(flip_ud),
        'contrast': list(contrast) if contrast is not None else None,
        'slice_xy': np.asarray(slice_xy, dtype=float).tolist(),
        'atlas_xy': np.asarray(atlas_xy, dtype=float).tolist(),
    }
    js = output_dir / f'{name}_transform.json'
    js.write_text(json.dumps(meta, indent=2))
    return js
