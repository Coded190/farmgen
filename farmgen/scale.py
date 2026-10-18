"""Scale a mesh with one factor on X, Y, and Z.

``size_m`` is the catalog box (width, depth, height). The factor is the
smallest ``size_m / span``, so the mesh keeps its proportions and fits inside
that box. The XY center is the origin and ``min(z)`` is 0.
"""

from __future__ import annotations

import numpy as np

RELATIVE_SIZE_LIMIT = 0.15
# A full span longer than this times the 1st–99th span is a stray vertex.
OUTLIER_RATIO = 1.2


def _keep_mask(points: np.ndarray) -> np.ndarray:
    """Vertices inside the span used to scale. A spike beyond ``OUTLIER_RATIO`` is dropped."""
    full_low = points.min(axis=0)
    full_high = points.max(axis=0)
    full = full_high - full_low
    low = np.percentile(points, 1, axis=0)
    high = np.percentile(points, 99, axis=0)
    robust = np.maximum(high - low, 1e-9)
    use_robust = full > robust * OUTLIER_RATIO
    keep_low = np.where(use_robust, low, full_low)
    keep_high = np.where(use_robust, high, full_high)
    return np.all((points >= keep_low - 1e-9) & (points <= keep_high + 1e-9), axis=1)


def scale_vertices_to_size(vertices: np.ndarray, size_m: tuple[float, float, float]) -> np.ndarray:
    """Kept vertices, scaled by one factor. A stray vertex is not in the result."""
    scaled, _mask, _factor = scale_detail(vertices, size_m)
    return scaled


def scale_detail(
    vertices: np.ndarray, size_m: tuple[float, float, float]
) -> tuple[np.ndarray, np.ndarray, float]:
    """Scaled vertices, the mask of input vertices that were kept, and the factor.

    The factor is ``min(size_m[i] / span[i])``. Every axis is multiplied by it.
    """
    points = np.asarray(vertices, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3 or points.shape[0] == 0:
        raise ValueError(f"vertices must be (N, 3) with N>0, got {points.shape}")
    if not np.isfinite(points).all():
        raise ValueError("vertices contain non-finite values")
    target = np.asarray(size_m, dtype=np.float64)
    if target.shape != (3,) or np.any(target <= 0) or not np.isfinite(target).all():
        raise ValueError(f"size_m must be three positive meters, got {size_m}")

    mask = _keep_mask(points)
    if not np.any(mask):
        raise ValueError("every vertex fell outside the robust span")
    kept = points[mask]
    mins = kept.min(axis=0)
    extent = np.maximum(kept.max(axis=0) - mins, 1e-12)
    if np.any(extent <= 1e-12):
        raise ValueError(f"degenerate bbox extent {extent.tolist()}")
    factor = float(np.min(target / extent))
    scaled = (kept - mins) * factor
    scaled[:, 0] -= (float(scaled[:, 0].min()) + float(scaled[:, 0].max())) / 2.0
    scaled[:, 1] -= (float(scaled[:, 1].min()) + float(scaled[:, 1].max())) / 2.0
    scaled[:, 2] -= float(scaled[:, 2].min())
    return scaled, mask, factor


def kept_faces(faces: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Faces whose three vertices survived the clip, renumbered to the kept order."""
    faces = np.asarray(faces, dtype=np.int64)
    if faces.size == 0:
        return faces.reshape(0, 3)
    index = np.full(int(mask.shape[0]), -1, dtype=np.int64)
    index[np.asarray(mask, dtype=bool)] = np.arange(int(np.count_nonzero(mask)))
    mapped = index[faces]
    return mapped[np.all(mapped >= 0, axis=1)]


def bbox_size(vertices: np.ndarray) -> tuple[float, float, float]:
    points = np.asarray(vertices, dtype=np.float64)
    extent = points.max(axis=0) - points.min(axis=0)
    return (float(extent[0]), float(extent[1]), float(extent[2]))


def relative_size_error(bbox: tuple[float, float, float], size_m: tuple[float, float, float]) -> tuple[float, float, float]:
    errors = []
    for actual, target in zip(bbox, size_m, strict=True):
        if target <= 0:
            raise ValueError(f"size target must be positive, got {target}")
        errors.append(abs(actual - target) / target)
    return (errors[0], errors[1], errors[2])


def size_error_ok(bbox: tuple[float, float, float], size_m: tuple[float, float, float], limit: float = RELATIVE_SIZE_LIMIT) -> bool:
    return max(relative_size_error(bbox, size_m)) <= limit
