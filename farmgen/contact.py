"""Contact-hole screen for the bottom of a scaled mesh.

Definition used by the reject check:

- Height is the mesh bbox along Z.
- A vertex is in the contact band when ``z <= max(0.02 * height, 0.01 m)``.
- The footprint is rasterized on a 32 by 32 grid over the XY bounding box.
- A cell counts as contact when a contact-band vertex lies in it, or when the
  cell center lies on a triangle whose three vertices are all in the band.
  Face fill is included so a coarse but solid bottom is not marked as a hole;
  an opening with no bottom triangles still counts as empty.
- ``contact_hole_fraction`` is empty cells inside the 2D convex hull of the
  contact band, divided by cells inside that hull.
- A point, an edge, or a hull that contains no cell center has no interior
  cell, so the fraction is 0. No vertices in the band is still 1.
- Reject when the fraction is above 0.25. This is a screen for a hole through
  the contact surface, not a watertightness proof.
"""

from __future__ import annotations

import numpy as np

HOLE_FRACTION_LIMIT = 0.25
GRID = 32


def contact_hole_fraction(
    vertices: np.ndarray,
    faces: np.ndarray,
    grid: int = GRID,
) -> float:
    points = np.asarray(vertices, dtype=np.float64)
    triangles = np.asarray(faces, dtype=np.int64)
    if points.ndim != 2 or points.shape[1] != 3 or points.shape[0] == 0:
        raise ValueError("vertices must be (N, 3)")
    height = float(points[:, 2].max() - points[:, 2].min())
    band = max(0.02 * height, 0.01)
    bottom = points[:, 2] <= (points[:, 2].min() + band)
    if not np.any(bottom):
        return 1.0

    mins = points[:, :2].min(axis=0)
    maxs = points[:, :2].max(axis=0)
    extent = np.maximum(maxs - mins, 1e-9)
    hull = _convex_hull(points[bottom, :2])
    # A point or an edge has no interior. That is a contact, not a hole.
    if hull.shape[0] < 3:
        return 0.0

    occupied = np.zeros((grid, grid), dtype=bool)
    inside = np.zeros((grid, grid), dtype=bool)
    xs = mins[0] + (np.arange(grid) + 0.5) * extent[0] / grid
    ys = mins[1] + (np.arange(grid) + 0.5) * extent[1] / grid

    bottom_ids = np.flatnonzero(bottom)
    for vid in bottom_ids:
        ix = int(np.clip((points[vid, 0] - mins[0]) / extent[0] * grid, 0, grid - 1))
        iy = int(np.clip((points[vid, 1] - mins[1]) / extent[1] * grid, 0, grid - 1))
        occupied[iy, ix] = True

    bottom_faces = []
    if triangles.size:
        for face in triangles:
            if face.shape[0] < 3:
                continue
            if bool(bottom[face[0]] and bottom[face[1]] and bottom[face[2]]):
                bottom_faces.append(points[face[:3], :2])

    for iy, y in enumerate(ys):
        for ix, x in enumerate(xs):
            if _point_in_polygon(x, y, hull):
                inside[iy, ix] = True
            if occupied[iy, ix]:
                continue
            for tri in bottom_faces:
                if _point_in_triangle(x, y, tri):
                    occupied[iy, ix] = True
                    break

    hull_cells = int(inside.sum())
    # The 32x32 grid is over the full XY bbox. A solid contact smaller than
    # one cell (a plant root under a wide canopy) contains no cell center, so
    # the ratio is undefined. There is no interior cell that can be empty.
    if hull_cells == 0:
        return 0.0
    empty = int(np.logical_and(inside, ~occupied).sum())
    return empty / hull_cells


def seal_contact_floor(
    vertices: np.ndarray,
    faces: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float, bool]:
    """Close an open bottom when the contact-hole fraction is above the limit.

    Image-to-3D barns are often an exterior shell: the wall bottoms form a
    ring, so the convex hull of that ring is mostly empty. The fraction
    definition does not change. This adds one triangle fan on z = min(z),
    using that same hull, and reports the fraction again.
    """
    points = np.asarray(vertices, dtype=np.float64)
    triangles = np.asarray(faces, dtype=np.int64)
    hole = contact_hole_fraction(points, triangles)
    if hole <= HOLE_FRACTION_LIMIT:
        return points, triangles, hole, False
    height = float(points[:, 2].max() - points[:, 2].min())
    band = max(0.02 * height, 0.01)
    bottom = points[:, 2] <= (points[:, 2].min() + band)
    hull = _convex_hull(points[bottom, :2])
    if hull.shape[0] < 3:
        return points, triangles, hole, False
    extra = np.column_stack([hull, np.full(hull.shape[0], points[:, 2].min())])
    start = points.shape[0]
    sealed_vertices = np.vstack([points, extra])
    fan = [[start, start + i, start + i + 1] for i in range(1, hull.shape[0] - 1)]
    sealed_faces = np.vstack([triangles, np.asarray(fan, dtype=np.int64)])
    sealed_hole = contact_hole_fraction(sealed_vertices, sealed_faces)
    return sealed_vertices, sealed_faces, sealed_hole, True


def _convex_hull(points: np.ndarray) -> np.ndarray:
    pts = np.unique(np.round(points, decimals=8), axis=0)
    if pts.shape[0] <= 2:
        return pts
    order = np.lexsort((pts[:, 1], pts[:, 0]))
    ordered = pts[order]

    def cross(o, a, b) -> float:
        return float((a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0]))

    lower: list[np.ndarray] = []
    for p in ordered:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    upper: list[np.ndarray] = []
    for p in ordered[::-1]:
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    hull = lower[:-1] + upper[:-1]
    return np.stack(hull, axis=0)


def _point_in_polygon(x: float, y: float, polygon: np.ndarray) -> bool:
    inside = False
    j = polygon.shape[0] - 1
    for i in range(polygon.shape[0]):
        xi, yi = float(polygon[i, 0]), float(polygon[i, 1])
        xj, yj = float(polygon[j, 0]), float(polygon[j, 1])
        intersects = (yi > y) != (yj > y) and x < (xj - xi) * (y - yi) / (yj - yi + 0.0) + xi
        if intersects:
            inside = not inside
        j = i
    return inside


def _point_in_triangle(x: float, y: float, tri: np.ndarray) -> bool:
    def sign(ax, ay, bx, by, cx, cy) -> float:
        return (ax - cx) * (by - cy) - (bx - cx) * (ay - cy)

    d1 = sign(x, y, tri[0, 0], tri[0, 1], tri[1, 0], tri[1, 1])
    d2 = sign(x, y, tri[1, 0], tri[1, 1], tri[2, 0], tri[2, 1])
    d3 = sign(x, y, tri[2, 0], tri[2, 1], tri[0, 0], tri[0, 1])
    has_neg = d1 < 0 or d2 < 0 or d3 < 0
    has_pos = d1 > 0 or d2 > 0 or d3 > 0
    return not (has_neg and has_pos)
