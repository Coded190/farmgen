"""CPU repair after TRELLIS.

The GPU hole fill runs first, inside the TRELLIS process, when that card has
enough free memory. This module fills the same pinholes only when that GPU
call did not run. A loop longer than ``HOLE_PERIMETER_MAX`` stays open either
way, so a silhouette is not closed with a flat sheet.

A manifold edge has one triangle (a boundary) or two (an interior seam). An edge
shared by three or more triangles is non-manifold: the surface branches, so a
later boolean or a collision cook has no single neighborhood there. Image-to-3D
meshes often have many of those edges and still look like the object. This
module duplicates vertices on the extra triangles so each edge has at most two
faces. The triangles stay. Qwen judges the pictures.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import trimesh

from farmgen.mesh_gate import plane_flatness

BESIDE_GAP_M = 0.1
# Same cap CuMesh uses. TRELLIS meshes sit in a 1 m box, so this is a pinhole.
# A larger opening is left open. Fan-filling it adds a sheet across the object.
HOLE_PERIMETER_MAX = 0.03


@dataclass(frozen=True)
class MeshRepair:
    """Counts written next to the kept GLB. A non-zero before-count is not a drop."""

    holes_filled: int
    nonmanifold_before: int
    nonmanifold_after: int
    flatness: float


def _close_holes(mesh: trimesh.Trimesh) -> int:
    """Fan-fill boundary loops. Return how many triangles were added.

    This is the CPU hole close that the GPU fill skips. An edge already shared
    by two or more triangles is not a boundary, so this does not remove
    non-manifold edges.
    """
    directed: set[tuple[int, int]] = set()
    for face in np.asarray(mesh.faces, dtype=np.int64):
        for index in range(3):
            directed.add((int(face[index]), int(face[(index + 1) % 3])))
    followers: dict[int, list[int]] = {}
    for start, end in directed:
        if (end, start) not in directed:
            followers.setdefault(start, []).append(end)
    seen: set[tuple[int, int]] = set()
    present = {_face_key(face) for face in np.asarray(mesh.faces, dtype=np.int64)}
    extra: list[list[int]] = []
    for start, ends in followers.items():
        for end in ends:
            if (start, end) in seen:
                continue
            loop = [start]
            cursor, nxt = start, end
            for _step in range(len(directed) + 2):
                seen.add((cursor, nxt))
                loop.append(nxt)
                if nxt == start:
                    break
                choices = [item for item in followers.get(nxt, []) if (nxt, item) not in seen]
                if not choices:
                    break
                cursor, nxt = nxt, choices[0]
            if len(loop) >= 4 and loop[-1] == start:
                ring = loop[:-1]
                if _loop_perimeter(mesh.vertices, ring) > HOLE_PERIMETER_MAX:
                    continue
                for index in range(1, len(ring) - 1):
                    face = [ring[0], ring[index], ring[index + 1]]
                    key = _face_key(face)
                    if key in present or len(set(face)) < 3:
                        continue
                    present.add(key)
                    extra.append(face)
    if not extra:
        return 0
    mesh.faces = np.vstack([mesh.faces, np.asarray(extra, dtype=np.int64)])
    return len(extra)


def _loop_perimeter(vertices: np.ndarray, ring: list[int]) -> float:
    points = np.asarray(vertices, dtype=np.float64)[np.asarray(ring, dtype=np.int64)]
    nxt = np.roll(points, -1, axis=0)
    return float(np.linalg.norm(nxt - points, axis=1).sum())


def _face_key(face) -> tuple[int, int, int]:
    return tuple(sorted(int(vertex) for vertex in face))


def nonmanifold_edge_count(faces: np.ndarray) -> int:
    """Edges shared by three or more triangles. An open sheet can still score zero."""
    counts: Counter[tuple[int, int]] = Counter()
    for face in np.asarray(faces, dtype=np.int64):
        for index in range(3):
            start = int(face[index])
            end = int(face[(index + 1) % 3])
            counts[(min(start, end), max(start, end))] += 1
    return sum(1 for count in counts.values() if count > 2)


def split_nonmanifold_edges(mesh: trimesh.Trimesh) -> int:
    """Give every edge at most two triangles by copying vertices. Return how many edges were split.

    The copied vertices keep the same position, so the picture does not move.
    Faces beyond the first pair on that edge get their own copies of the two
    endpoints. Edges with one or two faces are left welded.
    """
    faces = np.array(mesh.faces, dtype=np.int64, copy=True)
    verts = np.array(mesh.vertices, dtype=np.float64, copy=True)
    groups: dict[tuple[int, int], list[int]] = {}
    for face_index, face in enumerate(faces):
        for corner in range(3):
            start = int(face[corner])
            end = int(face[(corner + 1) % 3])
            if start == end:
                continue
            groups.setdefault((min(start, end), max(start, end)), []).append(face_index)
    sources: list[int] = []
    split = 0
    for (start, end), face_ids in groups.items():
        unique_faces = list(dict.fromkeys(face_ids))
        if len(unique_faces) <= 2:
            continue
        split += 1
        for face_index in unique_faces[2:]:
            for corner in range(3):
                vid = int(faces[face_index, corner])
                if vid != start and vid != end:
                    continue
                sources.append(vid)
                faces[face_index, corner] = len(verts) + len(sources) - 1
    if not sources:
        return 0
    uv = _per_vertex_uv(mesh)
    colors = _per_vertex_colors(mesh)
    mesh.vertices = np.vstack([verts, verts[np.asarray(sources, dtype=np.int64)]])
    mesh.faces = faces
    if uv is not None:
        mesh.visual.uv = np.vstack([uv, uv[np.asarray(sources, dtype=np.int64)]])
    if colors is not None:
        mesh.visual.vertex_colors = np.vstack([colors, colors[np.asarray(sources, dtype=np.int64)]])
    return split


def hole_fill_on_cpu(marker: dict | None) -> bool:
    """True when the GPU fill did not run, so this process fills pinholes.

    A missing marker is treated as a CPU fill. The GPU report uses
    ``fill=gpu`` only after CuMesh completed the call.
    """
    if not marker:
        return True
    return str(marker.get("fill", "cpu")) != "gpu"


def repair_mesh_file(path: Path, fill_holes: bool = True) -> MeshRepair:
    """Split branching edges, optionally fill pinholes, and write the GLB back.

    ``fill_holes`` is false when the GPU fill already ran. Loops longer than
    ``HOLE_PERIMETER_MAX`` are left open in either case. The file is kept when
    the triangles can be read. ``ValueError`` means the file is unreadable or
    has no faces. A remaining non-manifold count is reported and still kept,
    so a later picture review can decide.
    """
    try:
        loaded = trimesh.load(path, force="mesh", process=False)
    except (OSError, ValueError, RuntimeError) as exc:
        raise ValueError(f"unreadable {exc}") from exc
    if not isinstance(loaded, trimesh.Trimesh) or len(loaded.faces) == 0:
        raise ValueError("empty mesh")
    before = nonmanifold_edge_count(loaded.faces)
    split_nonmanifold_edges(loaded)
    holes = _close_holes(loaded) if fill_holes else 0
    split_nonmanifold_edges(loaded)
    try:
        trimesh.repair.fix_normals(loaded)
    except (ValueError, RuntimeError, ModuleNotFoundError):
        pass
    after = nonmanifold_edge_count(loaded.faces)
    try:
        flatness = plane_flatness(np.asarray(loaded.vertices, dtype=np.float64))
    except ValueError:
        flatness = 0.0
    loaded.export(path)
    return MeshRepair(holes, before, after, flatness)


def _per_vertex_uv(mesh: trimesh.Trimesh) -> np.ndarray | None:
    visual = getattr(mesh, "visual", None)
    uv = getattr(visual, "uv", None)
    if uv is None:
        return None
    array = np.asarray(uv, dtype=np.float64)
    if array.ndim != 2 or array.shape[1] != 2 or len(array) != len(mesh.vertices):
        return None
    return array


def _per_vertex_colors(mesh: trimesh.Trimesh) -> np.ndarray | None:
    visual = getattr(mesh, "visual", None)
    colors = getattr(visual, "vertex_colors", None)
    if colors is None:
        return None
    array = np.asarray(colors)
    if array.ndim != 2 or len(array) != len(mesh.vertices):
        return None
    return array


def xy_hull(vertices: np.ndarray, xy: tuple[float, float], yaw: float) -> np.ndarray:
    """Convex hull of mesh XY, moved to ``xy`` and rotated by ``yaw``, grown by 0.1 m."""
    points = np.asarray(vertices, dtype=np.float64)[:, :2]
    if len(points) < 3:
        raise ValueError("hull needs 3 points")
    hull = _convex_hull_2d(points)
    center = hull.mean(axis=0)
    direction = hull - center
    length = np.linalg.norm(direction, axis=1, keepdims=True)
    length = np.maximum(length, 1e-9)
    hull = hull + direction / length * (BESIDE_GAP_M / 2.0)
    cosine, sine = np.cos(yaw), np.sin(yaw)
    rotation = np.array([[cosine, -sine], [sine, cosine]])
    return (hull - center) @ rotation.T + np.asarray(xy, dtype=np.float64)


def _convex_hull_2d(points: np.ndarray) -> np.ndarray:
    ordered = np.unique(np.round(points, 6), axis=0)
    ordered = ordered[np.lexsort((ordered[:, 1], ordered[:, 0]))]

    def cross(origin, left, right) -> float:
        return (left[0] - origin[0]) * (right[1] - origin[1]) - (left[1] - origin[1]) * (right[0] - origin[0])

    lower: list[np.ndarray] = []
    for point in ordered:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], point) <= 0:
            lower.pop()
        lower.append(point)
    upper: list[np.ndarray] = []
    for point in ordered[::-1]:
        while len(upper) >= 2 and cross(upper[-2], upper[-1], point) <= 0:
            upper.pop()
        upper.append(point)
    ring = lower[:-1] + upper[:-1]
    if len(ring) < 3:
        raise ValueError("hull collapsed")
    return np.vstack(ring)


def polygons_overlap(left: np.ndarray, right: np.ndarray) -> bool:
    """True when two convex polygons intersect. Touching edges do not count."""
    return _separated(left, right) is False and _separated(right, left) is False


def _separated(polygon: np.ndarray, other: np.ndarray) -> bool:
    count = len(polygon)
    for index in range(count):
        edge = polygon[(index + 1) % count] - polygon[index]
        axis = np.array([-edge[1], edge[0]])
        norm = np.linalg.norm(axis)
        if norm <= 1e-12:
            continue
        axis = axis / norm
        first = polygon @ axis
        second = other @ axis
        if first.max() <= second.min() + 1e-8 or second.max() <= first.min() + 1e-8:
            return True
    return False
