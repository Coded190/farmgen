"""Ground mesh checks and the fixed 40 m soil plane.

Thickness is ``max(z) - min(z)`` in meters. The step-1 gate is thickness
under 0.25 m, one mesh, fewer than 500000 triangles, and an XY extent within
5 percent of the requested size. Face counts in the thousands are acceptable.
A flat plane with a few hundred faces passes; the failure mode is millions
of realized plant triangles.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import trimesh

THICKNESS_LIMIT_M = 0.25
FACE_LIMIT = 500_000
EXTENT_TOLERANCE = 0.05
VERTEX_MATCH_M = 1e-4


@dataclass(frozen=True)
class GroundReport:
    meshes: int
    faces: int
    bbox_m: tuple[float, float, float]
    thickness_m: float
    ok: bool
    reason: str


def load_ground(path: Path) -> trimesh.Trimesh:
    loaded = trimesh.load(path, force="mesh", process=False)
    if isinstance(loaded, trimesh.Scene):
        if len(loaded.geometry) != 1:
            raise ValueError(f"{path} contains {len(loaded.geometry)} meshes")
        loaded = next(iter(loaded.geometry.values()))
    if not isinstance(loaded, trimesh.Trimesh):
        raise ValueError(f"{path} did not load as a triangle mesh")
    return loaded


def plot_origin(terrain_m: tuple[float, float], plot_m: tuple[float, float]) -> tuple[float, float]:
    """The plot sits in the middle of the larger terrain mesh."""
    return (
        (float(terrain_m[0]) - float(plot_m[0])) / 2.0,
        (float(terrain_m[1]) - float(plot_m[1])) / 2.0,
    )


def inspect_ground(
    mesh: trimesh.Trimesh,
    size_m: tuple[float, float],
    plot_m: tuple[float, float] | None = None,
    relief_m: float | None = None,
    surround_relief_m: float | None = None,
) -> GroundReport:
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    if vertices.size == 0 or not np.isfinite(vertices).all():
        return GroundReport(1, 0, (0.0, 0.0, 0.0), 0.0, False, "empty or non-finite vertices")
    extent = vertices.max(axis=0) - vertices.min(axis=0)
    bbox = (float(extent[0]), float(extent[1]), float(extent[2]))
    faces = int(len(mesh.faces))
    thickness = bbox[2]
    reasons: list[str] = []
    if faces >= FACE_LIMIT:
        reasons.append(f"faces {faces} >= {FACE_LIMIT}")
    if plot_m is None:
        if thickness >= THICKNESS_LIMIT_M:
            reasons.append(f"thickness_m {thickness:.4f} >= {THICKNESS_LIMIT_M}")
    else:
        reasons.extend(
            _relief_reasons(
                vertices,
                size_m,
                plot_m,
                0.4 if relief_m is None else relief_m,
                4.0 if surround_relief_m is None else surround_relief_m,
            )
        )
    for axis, actual, target in ((0, bbox[0], size_m[0]), (1, bbox[1], size_m[1])):
        if target <= 0 or abs(actual - target) / target > EXTENT_TOLERANCE:
            reasons.append(f"axis {axis} extent {actual:.3f} m is not about {target:.3f} m")
    ok = not reasons
    return GroundReport(1, faces, bbox, thickness, ok, "ok" if ok else "; ".join(reasons))


def _relief_reasons(
    vertices: np.ndarray,
    terrain_m: tuple[float, float],
    plot_m: tuple[float, float],
    relief_m: float,
    surround_relief_m: float,
) -> list[str]:
    inside = _plot_mask(vertices[:, :2], terrain_m, plot_m)
    reasons: list[str] = []
    if not np.any(inside):
        reasons.append("plot contains no terrain vertices")
        return reasons
    plot_z = vertices[inside, 2]
    plot_span = float(plot_z.max() - plot_z.min())
    if plot_span > relief_m + 1e-6:
        reasons.append(f"plot relief {plot_span:.4f} m exceeds {relief_m}")
    outside = ~inside
    if np.any(outside):
        surround_z = vertices[outside, 2]
        surround_span = float(surround_z.max() - surround_z.min())
        if surround_span > surround_relief_m + 1e-6:
            reasons.append(f"surround relief {surround_span:.4f} m exceeds {surround_relief_m}")
    return reasons


def _plot_mask(xy: np.ndarray, terrain_m: tuple[float, float], plot_m: tuple[float, float]) -> np.ndarray:
    origin_x, origin_y = plot_origin(terrain_m, plot_m)
    return (
        (xy[:, 0] >= origin_x - 1e-6)
        & (xy[:, 0] <= origin_x + plot_m[0] + 1e-6)
        & (xy[:, 1] >= origin_y - 1e-6)
        & (xy[:, 1] <= origin_y + plot_m[1] + 1e-6)
    )


def sample_height(
    vertices: np.ndarray,
    faces: np.ndarray,
    xy: tuple[float, float],
) -> float:
    """Height of the terrain mesh at an XY point. The nearest vertex is the fallback."""
    points = np.asarray(vertices, dtype=np.float64)
    tris = np.asarray(faces, dtype=np.int64)
    if points.size == 0 or len(tris) == 0:
        return 0.0
    x, y = float(xy[0]), float(xy[1])
    tri = points[tris]
    ax, ay, az = tri[:, 0, 0], tri[:, 0, 1], tri[:, 0, 2]
    bx, by, bz = tri[:, 1, 0], tri[:, 1, 1], tri[:, 1, 2]
    cx, cy, cz = tri[:, 2, 0], tri[:, 2, 1], tri[:, 2, 2]
    denom = (by - cy) * (ax - cx) + (cx - bx) * (ay - cy)
    valid = np.abs(denom) > 1e-12
    w0 = np.zeros(len(tris), dtype=np.float64)
    w1 = np.zeros(len(tris), dtype=np.float64)
    w0[valid] = ((by[valid] - cy[valid]) * (x - cx[valid]) + (cx[valid] - bx[valid]) * (y - cy[valid])) / denom[valid]
    w1[valid] = ((cy[valid] - ay[valid]) * (x - cx[valid]) + (ax[valid] - cx[valid]) * (y - cy[valid])) / denom[valid]
    w2 = 1.0 - w0 - w1
    inside = valid & (w0 >= -1e-6) & (w1 >= -1e-6) & (w2 >= -1e-6)
    hits = np.flatnonzero(inside)
    if len(hits):
        i = int(hits[0])
        return float(w0[i] * az[i] + w1[i] * bz[i] + w2[i] * cz[i])
    nearest = int(np.argmin((points[:, 0] - x) ** 2 + (points[:, 1] - y) ** 2))
    return float(points[nearest, 2])


def terrain_mesh(
    terrain_m: tuple[float, float] = (120.0, 120.0),
    plot_m: tuple[float, float] = (40.0, 40.0),
    relief_m: float = 0.4,
    surround_relief_m: float = 4.0,
    seed: int = 0,
    divisions: int = 32,
) -> trimesh.Trimesh:
    """Furrows inside the plot and a rise outside it. Rebuilt from ``seed``."""
    if divisions < 1:
        raise ValueError("divisions must be positive")
    phase = float(np.random.default_rng(seed).random()) * 2.0 * np.pi
    width, depth = float(terrain_m[0]), float(terrain_m[1])
    xs = np.linspace(0.0, width, divisions + 1)
    ys = np.linspace(0.0, depth, divisions + 1)
    origin_x, origin_y = plot_origin(terrain_m, plot_m)
    margin = max(min(origin_x, origin_y), 1e-6)
    vertices = np.empty((xs.size * ys.size, 3), dtype=np.float64)
    index = 0
    for y in ys:
        for x in xs:
            inside = origin_x <= x <= origin_x + plot_m[0] and origin_y <= y <= origin_y + plot_m[1]
            if inside:
                z = relief_m * 0.5 * (0.5 + 0.5 * np.sin((2.0 * np.pi * (x - origin_x) / plot_m[0]) + phase))
            else:
                dx = origin_x - x if x < origin_x else max(0.0, x - (origin_x + plot_m[0]))
                dy = origin_y - y if y < origin_y else max(0.0, y - (origin_y + plot_m[1]))
                z = surround_relief_m * min(1.0, float(np.hypot(dx, dy)) / margin)
            vertices[index] = (x, y, z)
            index += 1
    cols = divisions + 1
    faces = []
    for row in range(divisions):
        for col in range(divisions):
            i = row * cols + col
            faces.append([i, i + 1, i + cols])
            faces.append([i + 1, i + cols + 1, i + cols])
    return trimesh.Trimesh(vertices=vertices, faces=np.asarray(faces, dtype=np.int64), process=False)


def flat_ground_mesh(size_m: tuple[float, float], divisions: int = 80) -> trimesh.Trimesh:
    """A single plane of the requested extent, thickness 0, origin at the corner.

    ``divisions`` per side yields ``2 * divisions^2`` triangles. 80 yields
    12800 faces, inside the tens-of-thousands band and far under the failure bar.
    """
    if divisions < 1:
        raise ValueError("divisions must be positive")
    width, depth = size_m
    xs = np.linspace(0.0, width, divisions + 1)
    ys = np.linspace(0.0, depth, divisions + 1)
    grid_x, grid_y = np.meshgrid(xs, ys, indexing="xy")
    vertices = np.column_stack([grid_x.ravel(), grid_y.ravel(), np.zeros(grid_x.size)])
    faces = []
    cols = divisions + 1
    for row in range(divisions):
        for col in range(divisions):
            i = row * cols + col
            faces.append([i, i + 1, i + cols])
            faces.append([i + 1, i + cols + 1, i + cols])
    mesh = trimesh.Trimesh(vertices=vertices, faces=np.asarray(faces, dtype=np.int64), process=False)
    return mesh


def vertex_delta(a: np.ndarray, b: np.ndarray) -> float:
    left = np.asarray(a, dtype=np.float64)
    right = np.asarray(b, dtype=np.float64)
    if left.shape != right.shape:
        raise ValueError(f"vertex shapes differ: {left.shape} vs {right.shape}")
    return float(np.max(np.abs(left - right)))


def same_ground_geometry(a: np.ndarray, b: np.ndarray, limit_m: float = VERTEX_MATCH_M) -> bool:
    return vertex_delta(a, b) < limit_m
