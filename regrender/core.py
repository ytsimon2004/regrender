"""Pure, GUI-free building blocks shared across regrender modes.

Everything here is side-effect-light and unit-testable without napari, so future
modes (cell annotation, probe tracks, ...) can reuse the same transform/image/atlas
helpers. The napari front-ends live in the ``main_*`` modules.
"""
from __future__ import annotations

import json
from collections import deque
from pathlib import Path
from typing import Any, TypedDict

import cv2
import imageio.v3 as iio
import numpy as np
from neuralib.atlas.ccf.matrix import SLICE_DIMENSION_10um
from neuralib.atlas.typing import PLANE_TYPE

__all__ = [
    'read_oriented', 'rotate', 'to_uint8', 'boundary_mask', 'region_name',
    'estimate_transform', 'save_transform', 'plane_point_to_ccf_mm', 'ccf_mm_to_plane_point',
    'raw_points_to_atlas', 'TransformMeta', 'load_transform', 'TerminalLog',
]


def _console():
    """Cached rich Console for the terminal mirror (created lazily to keep import light)."""
    global _CONSOLE
    if _CONSOLE is None:
        from rich.console import Console
        _CONSOLE = Console()
    return _CONSOLE


_CONSOLE = None


class TransformMeta(TypedDict):
    """Schema of a ``*_transform.json`` — the histology→atlas registration written by
    ``save_transform`` and consumed by the probe/roi modes. ``rotate``/``flip_lr``/``flip_ud``
    record the preprocessing so raw points can be replayed into atlas space."""
    matrix: list[list[float]]  # 3x3 homography (resized-slice -> atlas-plane pixels)
    plane: PLANE_TYPE
    resolution: int
    slice_index: int
    dw: int
    dh: int
    rotate: float
    flip_lr: bool
    flip_ud: bool
    contrast: list[float] | None
    slice_xy: list[list[float]]
    atlas_xy: list[list[float]]


def load_transform(path: Path) -> TransformMeta:
    """Read and parse a ``*_transform.json``."""
    return json.loads(Path(path).read_text())


class TerminalLog:
    """Append-only rich-console view over a magicgui Label: ``status.value = msg`` appends a
    timestamped, color-highlighted line (QLabel renders HTML) and scrolls like a terminal,
    keeping the last ``maxlines`` messages. Level (hence color) is inferred from the message;
    pass it explicitly with ``log.log(msg, 'error')`` when the heuristic isn't enough."""

    # level -> (panel HTML hex, rich terminal style)
    _COLORS = {'error': ('#ff6b6b', 'bold red'), 'warning': ('#f2c14e', 'yellow'),
               'save': ('#8bd450', 'green'), 'io': ('#56b6c2', 'cyan'), 'info': ('#d0d0d0', '')}

    def __init__(self, label, maxlines: int = 200, echo: bool = True):
        self._label = label
        self._lines: deque[str] = deque(maxlen=maxlines)
        self._echo = echo  # also mirror each line to the terminal (persistent, copyable log)

    @staticmethod
    def _infer(msg: str) -> str:
        m = msg.lower()
        if any(k in m for k in ('fail', 'error', 'cannot', "n't", 'invalid')):
            return 'error'
        if any(k in m for k in ('missing', 'cancel', 'no points', 'nothing', 'not ', 'stale')):
            return 'warning'
        if any(k in m for k in ('saved', 'save', '->', 'wrote', 'written')):
            return 'save'
        if any(k in m for k in ('load', 'render', 'resum', 'projected')):
            return 'io'
        return 'info'

    def log(self, msg: str, level: str | None = None):
        import html
        from datetime import datetime
        msg = str(msg)
        hexc, style = self._COLORS.get(level or self._infer(msg), self._COLORS['info'])
        ts = datetime.now().strftime('%H:%M:%S')
        self._lines.append(
            f'<span style="color:#6b7280">{ts}</span> '
            f'<span style="color:{hexc}">{html.escape(msg)}</span>')
        self._label.value = '<br>'.join(self._lines)
        if self._echo:  # mirror to terminal via rich (auto-strips color when piped)
            from rich.text import Text
            line = Text()
            line.append(ts + ' ', style='dim')
            line.append(msg, style=style)  # append is literal — no markup injection from msg
            _console().print(line)

    @property
    def value(self) -> str:
        return '\n'.join(self._lines)

    @value.setter
    def value(self, msg: str):
        self.log(msg)


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
    meta: TransformMeta = {
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


# --- probe reconstruction --------------------------------------------------

def plane_point_to_ccf_mm(
        plane_num: float, x: float, y: float, *,
        project_index: tuple[int, int, int],
        resolution: int,
        bregma_10um: tuple[int, int, int] = (540, 0, 570),
) -> tuple[float, float, float]:
    """A clicked atlas-plane pixel ``(x, y)`` on plane number ``plane_num`` -> bregma-relative
    CCF ``(AP, DV, ML)`` in mm.

    This is the inverse of ``neuralib.atlas.util.allen_to_brainrender_coord``, so a CSV of these
    points feeds ``ProbeRenderCLI`` (the existing interpolation + brainrender) directly.

    :param plane_num: voxel plane index the pixel sits on (``slice_index`` + tilt offset).
    :param x: atlas-plane column (the view's ``project_index`` x axis).
    :param y: atlas-plane row (the view's ``project_index`` y axis).
    :param project_index: ``view.project_index`` — (plane, x, y) positions within (AP, DV, ML).
    :param resolution: atlas resolution (µm).
    :param bregma_10um: bregma in 10µm voxels, AP/DV/ML (``ALLEN_CCF_10um_BREGMA``).
    """
    pidx, xidx, yidx = project_index
    idx = [0.0, 0.0, 0.0]
    idx[pidx], idx[xidx], idx[yidx] = plane_num, x, y
    ap, dv, ml = (v * resolution for v in idx)  # voxel -> µm (absolute)
    bap, bdv, bml = (b * 10 for b in bregma_10um)  # 10µm voxel -> µm
    return (bap - ap) / 1000, (dv - bdv) / 1000, (bml - ml) / 1000


def raw_points_to_atlas(
        pts_xy: np.ndarray, *,
        matrix: np.ndarray, raw_shape: tuple[int, int], plane: PLANE_TYPE,
        rotate_deg: float = 0.0, flip_lr: bool = False, flip_ud: bool = False) -> np.ndarray:
    """Forward-map raw-image ``(x, y)`` pixels into atlas-plane ``(x, y)`` pixels.

    Replays register's preprocessing pipeline (read -> flip -> rotate -> resize-to-dim ->
    matrix) on points, so ROIs labelled on the original histology land in the same atlas-plane
    space ``plane_point_to_ccf_mm`` consumes. The matrix was fit on the image resized to
    ``SLICE_DIMENSION_10um[plane]`` (see ``slice_transform_helper``), so resize happens here too.

    :param pts_xy: ``Array[float, [N, 2]]`` (x, y) on the raw image.
    :param matrix: 3x3 homography from the saved ``*_transform.json``.
    :param raw_shape: ``(H, W)`` of the raw image file.
    :param plane: cutting plane (selects the resize dimension).
    :param rotate_deg: rotation recorded at registration (same convention as :func:`rotate`).
    :return: ``Array[float64, [N, 2]]`` atlas-plane (x, y).
    """
    pts = np.asarray(pts_xy, dtype=np.float64).reshape(-1, 2)
    h, w = raw_shape[:2]
    x, y = pts[:, 0].copy(), pts[:, 1].copy()
    if flip_lr:
        x = (w - 1) - x
    if flip_ud:
        y = (h - 1) - y
    if rotate_deg:
        m = cv2.getRotationMatrix2D((w / 2, h / 2), rotate_deg, 1.0)  # same matrix warpAffine applies
        xy = m @ np.vstack([x, y, np.ones_like(x)])
        x, y = xy[0], xy[1]
    dim = SLICE_DIMENSION_10um[plane]  # (width, height)
    x = x * (dim[0] / w)
    y = y * (dim[1] / h)
    src = np.stack([x, y], axis=1).reshape(-1, 1, 2)
    return cv2.perspectiveTransform(src, np.asarray(matrix, dtype=np.float64)).reshape(-1, 2)


def ccf_mm_to_plane_point(
        ccf: tuple[float, float, float], *,
        project_index: tuple[int, int, int],
        resolution: int,
        bregma_10um: tuple[int, int, int] = (540, 0, 570),
) -> tuple[float, float, float]:
    """Inverse of :func:`plane_point_to_ccf_mm`: bregma-relative CCF ``(AP, DV, ML)`` mm ->
    ``(plane_num, x, y)`` in atlas voxels. Lets a saved coordinate be re-placed on a slice
    (the cross belongs on the slice whose ``plane_offset`` at ``(y, x)`` equals ``plane_num``).
    """
    ap_mm, dv_mm, ml_mm = ccf
    bap, bdv, bml = (b * 10 for b in bregma_10um)  # 10µm voxel -> µm
    idx = [(bap - ap_mm * 1000) / resolution,  # AP, DV, ML voxel indices
           (dv_mm * 1000 + bdv) / resolution,
           (bml - ml_mm * 1000) / resolution]
    pidx, xidx, yidx = project_index
    return idx[pidx], idx[xidx], idx[yidx]
