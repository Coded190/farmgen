"""Accept one TRELLIS mesh before any copy of it is posed."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from farmgen.critic import MAX_RETRIES
from farmgen.log import log_line
from farmgen.scene import Scene

# Median distance to the best-fit plane, divided by the longest span.
# The 0007 tree sheet measures 0.001. A fence panel measures 0.011.
FLAT_CARD_LIMIT = 0.005
# Share of triangle area whose normal is within 0.95 of X, Y, or Z.
# The 0007 tree and pasture are about 1.0. The barn and silo are about 0.70.
SHEET_FRACTION_LIMIT = 0.85
# Longer plan span divided by the shorter one. A square post is about 1. A rail is longer.
RAIL_ASPECT_MIN = 2.5
_AXIS_NORMAL = 0.95


@dataclass(frozen=True)
class MeshDecision:
    """Qwen's accept flag, the sentence that explains it, and the raw reply."""

    accept: bool
    reason: str = ""
    reply: str = ""


def plane_flatness(vertices: np.ndarray) -> float:
    """Median distance to the best-fit plane, divided by the longest span.

    Outliers that stretch the bounding cube do not hide a sheet: the median
    stays on the sheet. A returned value below ``FLAT_CARD_LIMIT`` is a card.
    """
    points = np.asarray(vertices, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3 or points.shape[0] < 4:
        raise ValueError(f"vertices must be (N, 3) with N>=4, got {points.shape}")
    centered = points - points.mean(axis=0)
    _, _, axes = np.linalg.svd(centered, full_matrices=False)
    distance = np.abs(centered @ axes[-1])
    spans = [float((centered @ axes[index]).max() - (centered @ axes[index]).min()) for index in range(3)]
    longest = max(spans)
    if longest <= 1e-12:
        return 0.0
    return float(np.median(distance) / longest)


def flat_card_reason(path: Path) -> str | None:
    """``flat card flatness=...`` when the GLB is a sheet. An unreadable file is left alone.

    This is a measurement. It does not delete the mesh. Qwen sees the pictures
    and accepts or rejects.
    """
    ratio = _flatness(path)
    if ratio is None or ratio >= FLAT_CARD_LIMIT:
        return None
    return f"flat card flatness={ratio:.4f}"


def sheet_fraction(vertices: np.ndarray, faces: np.ndarray, sample: int = 250_000, seed: int = 1) -> float:
    """Area whose normal lies along an axis, divided by the sampled triangle area.

    A single card and three crossed cards both return about 1. A barn with
    sloped and curved faces stays under ``SHEET_FRACTION_LIMIT``.
    """
    triangles = np.asarray(faces)
    if triangles.ndim != 2 or triangles.shape[0] == 0 or triangles.shape[1] < 3:
        return 0.0
    if triangles.shape[0] > sample:
        choice = np.random.default_rng(seed).choice(triangles.shape[0], sample, replace=False)
        triangles = triangles[choice]
    points = np.asarray(vertices, dtype=np.float64)
    tri = points[triangles[:, :3]]
    normal = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    area = np.linalg.norm(normal, axis=1)
    total = float(area.sum())
    if total <= 0.0:
        return 0.0
    unit = normal / np.maximum(area[:, None], 1e-12)
    aligned = np.zeros(len(area), dtype=bool)
    for axis in np.eye(3):
        aligned |= np.abs(unit @ axis) > _AXIS_NORMAL
    return float(area[aligned].sum() / total)


def thin_yaw_order(span_x: float, span_y: float) -> tuple[float, float, float, float]:
    """Yaw whose picture is the thin edge first. Yaw 0 looks along X, so its width is Y."""
    if float(span_x) <= float(span_y):
        return (90.0, 270.0, 0.0, 180.0)
    return (0.0, 180.0, 90.0, 270.0)


def blocking_post(path: Path) -> str | None:
    """A reason when a perimeter mesh is a post instead of a rail. Unreadable is None.

    Plan aspect is the longer horizontal span divided by the shorter one.
    """
    loaded = _load_mesh(path)
    if loaded is None:
        return None
    vertices = np.asarray(getattr(loaded, "vertices", []))
    if vertices.ndim != 2 or vertices.shape[0] < 4 or vertices.shape[1] != 3:
        return None
    span_x = float(vertices[:, 0].max() - vertices[:, 0].min())
    span_y = float(vertices[:, 1].max() - vertices[:, 1].min())
    short = min(span_x, span_y)
    long = max(span_x, span_y)
    if short <= 1e-9:
        return f"post plan_aspect=inf limit={RAIL_ASPECT_MIN}"
    aspect = long / short
    if aspect >= RAIL_ASPECT_MIN:
        return None
    return f"post plan_aspect={aspect:.2f} limit={RAIL_ASPECT_MIN}"


def blocking_sheet(path: Path) -> str | None:
    """A reason when the GLB is a card or a stack of axis-aligned sheets. Unreadable is None."""
    loaded = _load_mesh(path)
    if loaded is None:
        return None
    vertices = np.asarray(getattr(loaded, "vertices", []))
    faces = np.asarray(getattr(loaded, "faces", []))
    if vertices.ndim != 2 or vertices.shape[0] < 4 or vertices.shape[1] != 3:
        return None
    ratio = plane_flatness(vertices)
    fraction = sheet_fraction(vertices, faces) if faces.ndim == 2 and len(faces) else 0.0
    if ratio >= FLAT_CARD_LIMIT and fraction < SHEET_FRACTION_LIMIT:
        return None
    return f"sheet flatness={ratio:.4f} sheet_fraction={fraction:.3f}"


def mesh_measure_note(path: Path) -> str:
    """Flatness and the non-manifold edge count, for a Qwen prompt. Empty when unread."""
    from farmgen.repair import nonmanifold_edge_count

    ratio = _flatness(path)
    if ratio is None:
        return ""
    import trimesh

    try:
        loaded = trimesh.load(path, force="mesh", process=False)
    except (OSError, ValueError, RuntimeError, TypeError):
        return ""
    vertices = np.asarray(getattr(loaded, "vertices", []))
    faces = np.asarray(getattr(loaded, "faces", []))
    count = nonmanifold_edge_count(faces) if len(faces) else 0
    fraction = sheet_fraction(vertices, faces) if len(faces) else 0.0
    return (
        f"flatness={ratio:.4f} card_limit={FLAT_CARD_LIMIT} sheet_fraction={fraction:.3f} "
        f"sheet_limit={SHEET_FRACTION_LIMIT} nonmanifold_edges={count}. "
        "flatness is the median distance to the best-fit plane divided by the longest span. "
        "sheet_fraction is the share of triangle area whose normal is along X, Y, or Z. "
        "A non-manifold edge is an edge shared by three or more triangles."
    )


def _load_mesh(path: Path):
    import trimesh

    try:
        return trimesh.load(path, force="mesh", process=False)
    except (OSError, ValueError, RuntimeError, TypeError):
        return None


def _flatness(path: Path) -> float | None:
    loaded = _load_mesh(path)
    if loaded is None:
        return None
    vertices = np.asarray(getattr(loaded, "vertices", []))
    if vertices.ndim != 2 or vertices.shape[0] < 4 or vertices.shape[1] != 3:
        return None
    return plane_flatness(vertices)


def fresh_views(mesh: Path, candidates: list[Path]) -> list[Path]:
    """Orbit pictures that are at least as new as the current GLB.

    An older thumb belongs to a previous mesh and must not be judged.
    """
    if not mesh.is_file():
        return []
    stamp = mesh.stat().st_mtime
    return [path for path in candidates if path.is_file() and path.stat().st_mtime >= stamp]


def apply_mesh_gate(
    scene: Scene,
    judge,
    regenerate,
    attempts: int = MAX_RETRIES,
    only: set[str] | None = None,
    rebuild_first: bool = False,
) -> set[str]:
    """Return ids that will be posed. A reject regenerates, then the thumb is judged again.

    ``judge(asset_id, attempt)`` returns a bool or a ``MeshDecision``.
    ``regenerate(asset)`` downloads a new photo and rebuilds that mesh.
    ``only`` limits the ids. ``rebuild_first`` starts with a new photo, which
    is how a scene assessment sends one bad mesh back through this gate.
    After ``attempts`` rejects the mesh is not posed. The reason from each
    call is logged so a later run can be inspected.
    """
    accepted: set[str] = set()
    from farmgen.accept import image_ok, mark_accepted, same_rejected_image

    for asset in scene.assets:
        if only is not None and asset.id not in only:
            continue
        if rebuild_first:
            regenerate(asset)
        ok = False
        reason = ""
        for attempt in range(1, attempts + 1):
            try:
                ok, reason = _decision(judge(asset.id, attempt))
            except (OSError, RuntimeError, ValueError) as exc:
                ok = False
                reason = str(exc)
                log_line(
                    scene.id,
                    scene.seed,
                    "mesh_gate",
                    id=asset.id,
                    attempt=attempt,
                    accept=0,
                    reason=_one_line(reason),
                    error=str(exc),
                )
            else:
                log_line(
                    scene.id,
                    scene.seed,
                    "mesh_gate",
                    id=asset.id,
                    attempt=attempt,
                    accept=int(ok),
                    reason=_one_line(reason),
                )
            sheet = ""
            post = ""
            mesh_file = Path(asset.mesh_path) if asset.mesh_path else None
            if mesh_file is not None and mesh_file.is_file():
                sheet = blocking_sheet(mesh_file) or ""
                if asset.layout == "perimeter":
                    post = blocking_post(mesh_file) or ""
            blocked = sheet or post
            if ok and blocked:
                ok = False
                reason = blocked
                log_line(
                    scene.id,
                    scene.seed,
                    "mesh_gate",
                    id=asset.id,
                    attempt=attempt,
                    accept=0,
                    reason=_one_line(reason),
                    note="sheet" if sheet else "post",
                )
            if ok:
                if asset.mesh_path and Path(asset.mesh_path).is_file():
                    mark_accepted(asset, "mesh")
                    asset.mesh_image_sha = ""
                break
            if blocked:
                from farmgen.run import retire_photo

                retire_photo(scene, asset, reason="sheet_photo" if sheet else "post_photo")
                if attempt < attempts:
                    regenerate(asset)
                continue
            if image_ok(asset):
                from farmgen.run import forget_rejected_mesh

                if not same_rejected_image(asset):
                    forget_rejected_mesh(scene, asset)
                break
            if attempt < attempts:
                regenerate(asset)
        if ok:
            accepted.add(asset.id)
            continue
        if image_ok(asset):
            from farmgen.run import forget_rejected_mesh

            if not same_rejected_image(asset):
                forget_rejected_mesh(scene, asset)
        else:
            asset.mesh_path = None
            asset.instances = []
        log_line(
            scene.id,
            scene.seed,
            "mesh_gate",
            id=asset.id,
            note="dropped",
            accept=0,
            reason=_one_line(reason) or "retries_exhausted",
        )
    return accepted


def _decision(value: object) -> tuple[bool, str]:
    accept = getattr(value, "accept", None)
    if accept is None:
        return bool(value), ""
    reason = getattr(value, "reason", "")
    return bool(accept), "" if reason is None else str(reason)


def _one_line(text: str) -> str:
    return " ".join(text.split())[:400]
