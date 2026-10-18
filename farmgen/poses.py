"""Turn a VLM point on the top image into a stored instance pose."""

from __future__ import annotations

import inspect
import math

from farmgen.log import log_line
from farmgen.placer import Footprint, footprint_issue, ring_slots
from farmgen.scene import Instance, Scene, crop_band_y, plan_extent
from farmgen.views import uv_to_plot_xy


def apply_proposals(scene: Scene, poses: list[dict], problems: list | None = None) -> set[str]:
    """Store every accepted pose. An id is kept when each copy has one.

    A missing copy stays missing so the caller can ask Qwen for that index.
    A footprint that leaves the yard or the ring, or that covers another copy,
    is recorded in ``problems`` and is not stored. Yaw is stored in radians.
    """
    stored: dict[str, dict[int, Instance]] = {
        asset.id: {inst.index: inst for inst in asset.instances} for asset in scene.assets
    }
    for pose in poses:
        asset_id = str(pose.get("id", ""))
        try:
            asset = scene.asset(asset_id)
        except KeyError:
            log_line(scene.id, scene.seed, "propose", id=asset_id, note="unknown asset")
            continue
        u, v, yaw_deg = pose.get("u"), pose.get("v"), pose.get("yaw_deg")
        if u is None or v is None or yaw_deg is None:
            log_line(scene.id, scene.seed, "propose", id=asset_id, note="missing u,v,yaw_deg")
            continue
        try:
            xy = uv_to_plot_xy(float(u), float(v), scene.ground.terrain_m, scene.ground.plot_m)
        except ValueError as exc:
            log_line(scene.id, scene.seed, "propose", id=asset_id, note=str(exc))
            continue
        index = int(pose.get("index", len(stored[asset_id])))
        yaw = math.radians(float(yaw_deg))
        issue = _crop_issue(scene, asset, xy)
        if issue is None:
            issue = footprint_issue(
                xy,
                yaw,
                plan_extent(asset),
                asset.layout,
                scene.ground.plot_m,
                scene.ground.terrain_m,
                _occupied(scene, stored, skip=(asset_id, index)),
            )
        if issue is None:
            issue = _hull_issue(scene, asset, xy, yaw, stored, skip=(asset_id, index))
        if issue is not None:
            note = "overlap" if issue.startswith("overlaps") else "outside"
            log_line(
                scene.id,
                scene.seed,
                "propose",
                id=asset_id,
                instance=index,
                note=note,
                reason=issue,
                xy=f"{xy[0]:.2f},{xy[1]:.2f}",
            )
            if problems is not None:
                problems.append((asset_id, index, issue))
            continue
        stored[asset_id][index] = Instance(index=index, xy=xy, yaw=yaw, z=0.0)
        log_line(
            scene.id,
            scene.seed,
            "propose",
            id=asset_id,
            instance=index,
            xy=f"{xy[0]:.4f},{xy[1]:.4f}",
            yaw_deg=float(yaw_deg),
        )
    kept: set[str] = set()
    for asset in scene.assets:
        have = stored[asset.id]
        needed = set(range(asset.count))
        if needed <= set(have):
            asset.instances = [have[index] for index in range(asset.count)]
            kept.add(asset.id)
            continue
        asset.instances = [have[index] for index in sorted(have) if index in needed]
        if asset.instances:
            log_line(scene.id, scene.seed, "propose", id=asset.id, note="partial", kept=len(asset.instances))
    return kept


def _hull_issue(scene: Scene, asset, xy: tuple[float, float], yaw: float, stored: dict, skip: tuple[str, int]) -> str | None:
    """Convex-hull overlap with a 0.1 m beside gap. An asset without a mesh keeps the box test."""
    from pathlib import Path

    import numpy as np
    import trimesh

    from farmgen.repair import polygons_overlap, xy_hull

    path = Path(asset.mesh_path) if asset.mesh_path else None
    if path is None or not path.is_file():
        return None
    try:
        loaded = trimesh.load(path, force="mesh", process=False)
        mine = xy_hull(np.asarray(loaded.vertices), xy, yaw)
    except (OSError, ValueError, RuntimeError):
        return None
    for other in scene.assets:
        other_path = Path(other.mesh_path) if other.mesh_path else None
        if other_path is None or not other_path.is_file():
            continue
        try:
            other_mesh = trimesh.load(other_path, force="mesh", process=False)
            other_vertices = np.asarray(other_mesh.vertices)
        except (OSError, ValueError, RuntimeError):
            continue
        for index, inst in stored[other.id].items():
            if skip == (other.id, index):
                continue
            theirs = xy_hull(other_vertices, inst.xy, inst.yaw)
            if polygons_overlap(mine, theirs):
                return f"overlaps {other.id}_{index:02d}"
    return None


def _occupied(scene: Scene, stored: dict[str, dict[int, Instance]], skip: tuple[str, int] | None = None) -> list[Footprint]:
    occupied: list[Footprint] = []
    for asset in scene.assets:
        for index, inst in stored[asset.id].items():
            if skip == (asset.id, index):
                continue
            occupied.append(Footprint(asset.id, index, inst.xy, inst.yaw, plan_extent(asset)))
    return occupied


def missing_indexes(scene: Scene, asset_ids: set[str]) -> list[tuple[str, int]]:
    """Copies that still need a Qwen pose."""
    missing: list[tuple[str, int]] = []
    for asset in scene.assets:
        if asset.id not in asset_ids:
            continue
        have = {inst.index for inst in asset.instances}
        for index in range(asset.count):
            if index not in have:
                missing.append((asset.id, index))
    return missing


def seat_visible(scene: Scene, asset_ids: set[str], gap_m: float = 8.0) -> None:
    """Keep footprints that fit, then lattice-seat the copies that do not.

    A yard copy must lie inside the plot and clear every other footprint.
    A surround copy must lie in the ring, outside the plot. ``gap_m`` is unused;
    the check is the footprint, not a center gap.
    """
    del gap_m
    occupied: list[Footprint] = []
    for asset in _seat_order(scene.assets):
        if asset.id not in asset_ids:
            occupied.extend(_footprints(asset))
            continue
        kept: list[Instance] = []
        for inst in asset.instances:
            issue = _crop_issue(scene, asset, inst.xy)
            if issue is None:
                issue = footprint_issue(
                    inst.xy,
                    inst.yaw,
                    plan_extent(asset),
                    asset.layout,
                    scene.ground.plot_m,
                    scene.ground.terrain_m,
                    occupied,
                )
            if issue is None:
                kept.append(inst)
                occupied.append(Footprint(asset.id, inst.index, inst.xy, inst.yaw, plan_extent(asset)))
                continue
            note = "overlap" if issue.startswith("overlaps") else "outside"
            log_line(
                scene.id,
                scene.seed,
                "propose",
                id=asset.id,
                instance=inst.index,
                note=note,
                reason=issue,
                xy=f"{inst.xy[0]:.2f},{inst.xy[1]:.2f}",
            )
        asset.instances = kept
    for asset in _seat_order(scene.assets):
        if asset.id not in asset_ids:
            continue
        have = {inst.index for inst in asset.instances}
        for index in range(asset.count):
            if index in have:
                continue
            xy = _lattice_point(scene, asset, occupied)
            if xy is None:
                remaining = sum(1 for later in range(index, asset.count) if later not in have)
                log_line(
                    scene.id,
                    scene.seed,
                    "propose",
                    id=asset.id,
                    note="unplaced",
                    count=remaining,
                )
                break
            asset.instances.append(Instance(index=index, xy=xy, yaw=0.0, z=0.0))
            occupied.append(Footprint(asset.id, index, xy, 0.0, plan_extent(asset)))
            log_line(
                scene.id,
                scene.seed,
                "propose",
                id=asset.id,
                instance=index,
                note="lattice",
                xy=f"{xy[0]:.2f},{xy[1]:.2f}",
            )
        asset.instances.sort(key=lambda inst: inst.index)


def _seat_order(assets: list) -> list:
    """Trees take the outer ring first. A bush uses a gap those trees left."""

    def rank(asset) -> int:
        if asset.layout != "surround":
            return 1
        prompt = asset.prompt.lower()
        if "tree" in prompt:
            return 0
        if "bush" in prompt or "shrub" in prompt:
            return 2
        return 1

    return sorted(assets, key=rank)


def _footprints(asset) -> list[Footprint]:
    return [Footprint(asset.id, inst.index, inst.xy, inst.yaw, plan_extent(asset)) for inst in asset.instances]


def _crop_issue(scene: Scene, asset, xy: tuple[float, float]) -> str | None:
    """Yard equipment stays north of the crop blocks. A field and a ring are not equipment."""
    if asset.field_m is not None or asset.layout in {"surround", "perimeter"}:
        return None
    band = crop_band_y(scene)
    if band is None or float(xy[1]) >= band:
        return None
    return "inside the crop band"


def _lattice_point(scene: Scene, asset, occupied: list[Footprint]) -> tuple[float, float] | None:
    plot = scene.ground.plot_m
    terrain = scene.ground.terrain_m
    size = plan_extent(asset)
    if asset.layout == "surround":
        candidates = ring_slots(plot, terrain, max(size[0], size[1], 8.0))
    elif asset.layout == "perimeter":
        return None
    else:
        candidates = _yard_centers(plot, size)
        band = crop_band_y(scene)
        if band is not None:
            candidates = [xy for xy in candidates if xy[1] >= band]
    free = [
        xy
        for xy in candidates
        if footprint_issue(xy, 0.0, size, asset.layout, plot, terrain, occupied) is None
    ]
    if not free:
        return None
    if asset.layout == "surround":
        return _widest_gap(free, _ring_points(occupied, plot), (plot[0] / 2.0, plot[1] / 2.0))
    return _farthest_cell(free, [item.xy for item in occupied], (plot[0] / 2.0, plot[1] / 2.0))


def _ring_points(occupied: list[Footprint], plot_m: tuple[float, float]) -> list[tuple[float, float]]:
    width, depth = float(plot_m[0]), float(plot_m[1])
    return [item.xy for item in occupied if item.xy[0] < 0 or item.xy[1] < 0 or item.xy[0] > width or item.xy[1] > depth]


def _farthest_cell(
    free: list[tuple[float, float]],
    occupied: list[tuple[float, float]],
    center: tuple[float, float],
) -> tuple[float, float]:
    if not occupied:
        return min(free, key=lambda xy: math.hypot(xy[0] - center[0], xy[1] - center[1]))

    def clearance(xy: tuple[float, float]) -> float:
        return min(math.hypot(xy[0] - other[0], xy[1] - other[1]) for other in occupied)

    return max(free, key=clearance)


def _widest_gap(
    free: list[tuple[float, float]],
    occupied: list[tuple[float, float]],
    center: tuple[float, float],
) -> tuple[float, float]:
    def angle(xy: tuple[float, float]) -> float:
        return math.atan2(xy[1] - center[1], xy[0] - center[0])

    if not occupied:
        return min(free, key=lambda xy: abs(angle(xy)))
    taken = [angle(xy) for xy in occupied]

    def gap(xy: tuple[float, float]) -> float:
        ang = angle(xy)
        return min(_angle_distance(ang, other) for other in taken)

    return max(free, key=gap)


def _angle_distance(left: float, right: float) -> float:
    delta = abs(left - right) % (2.0 * math.pi)
    return min(delta, 2.0 * math.pi - delta)


def place_fences(scene: Scene, only: set[str] | None = None) -> None:
    """Seat every fence segment on the plot perimeter. Qwen is not asked for each post.

    ``only`` limits which ids are seated. Other fences keep the instances they have.
    """
    from farmgen.placer import perimeter_samples

    for asset in scene.assets:
        if asset.layout != "perimeter":
            continue
        if only is not None and asset.id not in only:
            continue
        samples = perimeter_samples(scene.ground.plot_m, float(asset.size_m[0]), float(asset.size_m[1]))
        asset.instances = [
            Instance(index=index, xy=xy, yaw=yaw, z=0.0) for index, (xy, yaw) in enumerate(samples)
        ]
        asset.count = len(samples)
        log_line(scene.id, scene.seed, "propose", id=asset.id, note="perimeter", count=asset.count)


def place_grids(scene: Scene, only: set[str] | None = None) -> None:
    """Seat a grid asset on a spacing of its own footprint. Occupied cells are skipped.

    Spacing is the longer plan side, so a 1 m plant tiles a 40 m yard in 1,600 cells
    before the barn, silo, and tractor take their cells.
    """
    occupied: list[Footprint] = []
    for asset in scene.assets:
        if asset.layout == "grid":
            continue
        occupied.extend(_footprints(asset))
    for asset in scene.assets:
        if asset.layout != "grid":
            continue
        if only is not None and asset.id not in only:
            continue
        points = _grid_points(scene, asset, occupied)
        asset.instances = [Instance(index=index, xy=xy, yaw=0.0, z=0.0) for index, xy in enumerate(points)]
        asset.count = len(points)
        log_line(scene.id, scene.seed, "propose", id=asset.id, note="grid", count=asset.count)


def _grid_points(scene: Scene, asset, occupied: list[Footprint]) -> list[tuple[float, float]]:
    width, depth = float(scene.ground.plot_m[0]), float(scene.ground.plot_m[1])
    size = plan_extent(asset)
    spacing = max(size[0], size[1], 1e-6)
    points: list[tuple[float, float]] = []
    y = spacing / 2.0
    while y <= depth - spacing / 2.0 + 1e-6:
        x = spacing / 2.0
        while x <= width - spacing / 2.0 + 1e-6:
            xy = (float(x), float(y))
            if (
                footprint_issue(xy, 0.0, size, "grid", scene.ground.plot_m, scene.ground.terrain_m, occupied, margin_m=0.0)
                is None
            ):
                points.append(xy)
                occupied.append(Footprint(asset.id, len(points) - 1, xy, 0.0, size))
            if len(points) >= asset.count:
                return points
            x += spacing
        y += spacing
    return points


def place_fields(scene: Scene, only: set[str] | None = None) -> None:
    """Copy one plant across its field, skipping cells another footprint already uses.

    ``only`` limits which fields are filled. Other fields keep the instances they have.
    """
    occupied: list[Footprint] = []
    for asset in scene.assets:
        if asset.field_m is None:
            occupied.extend(_footprints(asset))
    for asset in scene.assets:
        if asset.field_m is None:
            continue
        if only is not None and asset.id not in only:
            continue
        points = _field_points(scene, asset, occupied)
        asset.instances = [Instance(index=index, xy=xy, yaw=0.0, z=0.0) for index, xy in enumerate(points)]
        asset.count = len(points)
        log_line(
            scene.id,
            scene.seed,
            "propose",
            id=asset.id,
            note="field",
            count=asset.count,
            field_m=f"{asset.field_m[0]:.1f},{asset.field_m[1]:.1f}",
            origin="center" if asset.field_origin is None else f"{asset.field_origin[0]:.1f},{asset.field_origin[1]:.1f}",
        )


def _field_points(scene: Scene, asset, occupied: list[Footprint]) -> list[tuple[float, float]]:
    plot = scene.ground.plot_m
    terrain = scene.ground.terrain_m
    area_w, area_d = float(asset.field_m[0]), float(asset.field_m[1])
    spacing = 2.0
    if asset.field_origin is not None:
        x0, y0 = float(asset.field_origin[0]), float(asset.field_origin[1])
    else:
        x0 = (float(plot[0]) - area_w) / 2.0
        y0 = (float(plot[1]) - area_d) / 2.0
    size = plan_extent(asset)
    points: list[tuple[float, float]] = []
    y = y0 + spacing / 2.0
    while y <= y0 + area_d - spacing / 2.0 + 1e-6:
        x = x0 + spacing / 2.0
        while x <= x0 + area_w - spacing / 2.0 + 1e-6:
            xy = (float(x), float(y))
            if footprint_issue(xy, 0.0, size, "rows", plot, terrain, occupied) is None:
                points.append(xy)
                occupied.append(Footprint(asset.id, len(points), xy, 0.0, size))
            x += spacing
        y += spacing
    return points


def _yard_centers(plot_m: tuple[float, float], size_m: tuple[float, float]) -> list[tuple[float, float]]:
    margin = 1.0
    span_x, span_y = size_m
    step_x = max(span_x, 2.0)
    step_y = max(span_y, 2.0)
    width, depth = float(plot_m[0]), float(plot_m[1])
    centers: list[tuple[float, float]] = []
    x = margin + span_x / 2.0
    while x <= width - margin - span_x / 2.0 + 1e-6:
        y = margin + span_y / 2.0
        while y <= depth - margin - span_y / 2.0 + 1e-6:
            centers.append((x, y))
            y += step_y
        x += step_x
    return centers


def sits_on_another(asset) -> bool:
    """An asset whose prompt puts it on top of something already in the yard."""
    text = f"{asset.id} {asset.prompt}".lower()
    return " on the " in text or " on top" in text or " atop " in text


def fill_poses(scene: Scene, propose, asset_ids: set[str], rounds: int = 2) -> set[str]:
    """Ask Qwen again for any copy that the last reply skipped or rejected.

    ``propose(missing)`` or ``propose(missing, problems)`` returns pose dicts.
    ``problems`` names the footprint failure from the previous reply.
    """
    kept: set[str] = set()
    problems: list[tuple[str, int, str]] = []
    ground_ids = {asset_id for asset_id in asset_ids if not sits_on_another(scene.asset(asset_id))}
    supported_ids = {asset_id for asset_id in asset_ids if asset_id not in ground_ids}
    for group in (ground_ids, supported_ids):
        if group:
            kept.update(_fill_group(scene, propose, group, rounds, problems))
    return kept


def _fill_group(scene, propose, asset_ids: set[str], rounds: int, problems: list) -> set[str]:
    kept: set[str] = set()
    parameters = inspect.signature(propose).parameters
    accepts_problems = len(parameters) >= 2 or any(
        item.kind is inspect.Parameter.VAR_POSITIONAL for item in parameters.values()
    )
    for _round in range(rounds):
        missing = missing_indexes(scene, asset_ids)
        if not missing:
            break
        try:
            poses = propose(missing, list(problems)) if accepts_problems else propose(missing)
        except (OSError, RuntimeError, ValueError) as exc:
            log_line(scene.id, scene.seed, "propose", note="ask_failed", error=str(exc))
            break
        problems.clear()
        apply_proposals(scene, poses, problems)
    for asset_id in asset_ids:
        if not missing_indexes(scene, {asset_id}):
            kept.add(asset_id)
    return kept
