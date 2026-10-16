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
    """``flat card flatness=...`` when the GLB is a sheet. An unreadable file is left alone."""
    import trimesh

    try:
        loaded = trimesh.load(path, force="mesh")
    except (OSError, ValueError, RuntimeError, TypeError):
        return None
    vertices = np.asarray(getattr(loaded, "vertices", []))
    if vertices.ndim != 2 or vertices.shape[0] < 4 or vertices.shape[1] != 3:
        return None
    ratio = plane_flatness(vertices)
    if ratio < FLAT_CARD_LIMIT:
        return f"flat card flatness={ratio:.4f}"
    return None


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
    After ``attempts`` rejects the last mesh is kept and posed. The reason
    from each call is logged so a later run can be inspected.
    """
    accepted: set[str] = set()
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
            if ok:
                break
            if attempt < attempts:
                regenerate(asset)
        if ok:
            accepted.add(asset.id)
            continue
        if reason.startswith("flat card"):
            asset.mesh_path = None
            log_line(
                scene.id,
                scene.seed,
                "mesh_gate",
                id=asset.id,
                note="dropped",
                accept=0,
                reason=_one_line(reason),
            )
            continue
        accepted.add(asset.id)
        log_line(
            scene.id,
            scene.seed,
            "mesh_gate",
            id=asset.id,
            note="kept",
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
