"""Scale a mesh so its meter size matches the scene contract.

``size_m`` is width, depth, height along +X, +Y, +Z. After this step the
axis-aligned bbox has that size, the XY center is the origin, and ``min(z)``
is 0 so the mesh rests on the ground plane.
"""

from __future__ import annotations

import numpy as np

RELATIVE_SIZE_LIMIT = 0.15


def scale_vertices_to_size(vertices: np.ndarray, size_m: tuple[float, float, float]) -> np.ndarray:
    points = np.asarray(vertices, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3 or points.shape[0] == 0:
        raise ValueError(f"vertices must be (N, 3) with N>0, got {points.shape}")
    if not np.isfinite(points).all():
        raise ValueError("vertices contain non-finite values")
    target = np.asarray(size_m, dtype=np.float64)
    if target.shape != (3,) or np.any(target <= 0) or not np.isfinite(target).all():
        raise ValueError(f"size_m must be three positive meters, got {size_m}")

    mins = points.min(axis=0)
    maxs = points.max(axis=0)
    extent = maxs - mins
    if np.any(extent <= 1e-12):
        raise ValueError(f"degenerate bbox extent {extent.tolist()}")

    scaled = (points - mins) * (target / extent)
    scaled[:, 0] -= target[0] / 2.0
    scaled[:, 1] -= target[1] / 2.0
    return scaled


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
