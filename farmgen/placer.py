"""Place asset footprints on the flat plot.

Coordinates are meters from the plot corner ``(0, 0)``. The plot is
``ground.size_m``. A sample is accepted when the yaw-rotated XY rectangle
lies inside the plot and its intersection area with every existing footprint
is 0. Touching edges are allowed. Rotation is about Z only.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from farmgen.log import write_raw
from farmgen.scene import AssetSpec, Instance, Scene

MARGIN_M = 1.0
SURROUND_GAP_M = 2.0
_AREA_EPS = 1e-8


@dataclass(frozen=True)
class Footprint:
    asset_id: str
    index: int
    xy: tuple[float, float]
    yaw: float
    size_m: tuple[float, float]


def place_scene(
    scene: Scene,
    margin_m: float = MARGIN_M,
    keep: set[str] | None = None,
    lattice: bool = True,
) -> list[str]:
    """Place assets that are not already fully posed.

    ``keep`` is the set of asset ids whose instances were proposed and must
    stay. Their footprints still block the lattice. ``lattice=False`` leaves
    an unposed copy where it is instead of inventing a position.
    """
    keep = set(keep or ())
    if not lattice:
        lines: list[str] = []
        for asset in scene.assets:
            if len(asset.instances) >= asset.count:
                continue
            line = (
                f"FARMGEN scene={scene.id} seed={scene.seed} stage=place "
                f"id={asset.id} note=unposed kept={len(asset.instances)}"
            )
            print(line, flush=True)
            write_raw(line)
            lines.append(line)
        return lines
    rng = np.random.default_rng(scene.seed)
    placed: list[Footprint] = []
    lines: list[str] = []
    width, depth = scene.ground.size_m
    rows_index = 0
    surround_done = False
    for asset in scene.assets:
        if asset.id in keep and len(asset.instances) >= asset.count:
            for inst in asset.instances:
                placed.append(Footprint(asset.id, inst.index, inst.xy, inst.yaw, asset.size_m[:2]))
    for asset in scene.assets:
        if asset.id in keep and len(asset.instances) >= asset.count:
            continue
        if asset.layout == "surround":
            if surround_done:
                continue
            group = [
                item
                for item in scene.assets
                if item.layout == "surround" and not (item.id in keep and len(item.instances) >= item.count)
            ]
            if not group:
                surround_done = True
                continue
            grouped = _place_surround_group(group, width, depth, scene.ground.terrain_m, placed)
            for item in group:
                item.instances = grouped[item.id]
                for inst in item.instances:
                    line = (
                        f"FARMGEN scene={scene.id} seed={scene.seed} stage=place "
                        f"placed id={item.id} instance={inst.index} "
                        f"xy={inst.xy[0]:.4f},{inst.xy[1]:.4f} yaw={inst.yaw:.4f}"
                    )
                    print(line, flush=True)
                    write_raw(line)
                    lines.append(line)
            surround_done = True
            continue
        instances = _place_asset(asset, width, depth, scene.ground.terrain_m, margin_m, rng, placed, rows_index)
        if asset.layout == "rows":
            rows_index += 1
        asset.instances = instances
        for inst in instances:
            line = (
                f"FARMGEN scene={scene.id} seed={scene.seed} stage=place "
                f"placed id={asset.id} instance={inst.index} "
                f"xy={inst.xy[0]:.4f},{inst.xy[1]:.4f} yaw={inst.yaw:.4f}"
            )
            print(line, flush=True)
            write_raw(line)
            lines.append(line)
    return lines


def accept_sample(
    xy: tuple[float, float],
    yaw: float,
    size_m: tuple[float, float],
    plot_m: tuple[float, float],
    existing: list[Footprint],
    margin_m: float = MARGIN_M,
) -> bool:
    corners = _corners(xy, yaw, size_m)
    if not _inside_plot(corners, plot_m, margin_m):
        return False
    return not _overlaps(xy, yaw, size_m, existing)


def _overlaps(
    xy: tuple[float, float],
    yaw: float,
    size_m: tuple[float, float],
    existing: list[Footprint],
) -> bool:
    candidate = _corners(xy, yaw, size_m)
    for other in existing:
        other_corners = _corners(other.xy, other.yaw, other.size_m)
        if _polygon_area(_clip(candidate, other_corners)) > _AREA_EPS:
            return True
    return False


def in_surround(
    xy: tuple[float, float],
    size_m: tuple[float, float],
    plot_m: tuple[float, float],
    terrain_m: tuple[float, float],
    gap_m: float = SURROUND_GAP_M,
    yaw: float = 0.0,
) -> bool:
    """True when the footprint is in the ring and outside the plot plus ``gap_m``."""
    from farmgen.ground import plot_origin

    origin_x, origin_y = plot_origin(terrain_m, plot_m)
    local_min = (-origin_x, -origin_y)
    local_max = (terrain_m[0] - origin_x, terrain_m[1] - origin_y)
    corners = _corners(xy, yaw, size_m)
    if np.any(corners[:, 0] < local_min[0]) or np.any(corners[:, 0] > local_max[0]):
        return False
    if np.any(corners[:, 1] < local_min[1]) or np.any(corners[:, 1] > local_max[1]):
        return False
    expanded_min = (-gap_m, -gap_m)
    expanded_max = (plot_m[0] + gap_m, plot_m[1] + gap_m)
    # A corner inside the expanded plot means the copy is too close to the yard.
    inside_gap = (
        (corners[:, 0] >= expanded_min[0])
        & (corners[:, 0] <= expanded_max[0])
        & (corners[:, 1] >= expanded_min[1])
        & (corners[:, 1] <= expanded_max[1])
    )
    return not bool(np.any(inside_gap))


def footprint_issue(
    xy: tuple[float, float],
    yaw: float,
    size_m: tuple[float, float],
    layout: str,
    plot_m: tuple[float, float],
    terrain_m: tuple[float, float],
    existing: list[Footprint],
    margin_m: float = MARGIN_M,
) -> str | None:
    """Return why a footprint cannot sit here. ``None`` means it fits."""
    if layout == "surround":
        if not in_surround(xy, size_m, plot_m, terrain_m, yaw=yaw):
            return "outside the ring"
    else:
        if not _inside_plot(_corners(xy, yaw, size_m), plot_m, margin_m):
            return "outside the yard"
    for other in existing:
        if _overlaps(xy, yaw, size_m, [other]):
            return f"overlaps {other.asset_id}_{other.index:02d}"
    return None


def _place_asset(
    asset: AssetSpec,
    width: float,
    depth: float,
    terrain_m: tuple[float, float],
    margin_m: float,
    rng: np.random.Generator,
    placed: list[Footprint],
    rows_index: int,
) -> list[Instance]:
    if asset.layout == "unique":
        samples = [_sample_unique(asset, width, depth, margin_m, rng, placed)]
    elif asset.layout == "line":
        samples = _line_samples(asset, width, depth, margin_m)
    elif asset.layout == "grid":
        samples = _grid_samples(asset, width, depth, margin_m, cols=3, origin=(28.0, 30.0))
    elif asset.layout == "rows":
        samples = _grid_samples(
            asset, width, depth, margin_m, cols=4, origin=(8.0, 14.0 + rows_index * 8.0)
        )
    elif asset.layout == "surround":
        samples = _surround_samples(asset, (width, depth), terrain_m, rng, placed)
    elif asset.layout == "perimeter":
        samples = perimeter_samples((width, depth), float(asset.size_m[0]), float(asset.size_m[1]))
    else:
        raise ValueError(f"unknown layout {asset.layout}")

    instances: list[Instance] = []
    for index, (xy, yaw) in enumerate(samples):
        if asset.layout == "surround":
            accepted = in_surround(xy, asset.size_m[:2], (width, depth), terrain_m, yaw=yaw) and not _overlaps(
                xy, yaw, asset.size_m[:2], placed
            )
        else:
            accepted = accept_sample(xy, yaw, asset.size_m[:2], (width, depth), placed, margin_m)
        if not accepted:
            raise RuntimeError(
                f"placement rejected id={asset.id} instance={index} xy={xy[0]:.3f},{xy[1]:.3f} yaw={yaw:.3f}"
            )
        placed.append(Footprint(asset.id, index, xy, yaw, asset.size_m[:2]))
        instances.append(Instance(index=index, xy=xy, yaw=yaw))
    if len(instances) != asset.count:
        raise RuntimeError(f"{asset.id} placed {len(instances)} of {asset.count}")
    return instances


def ring_slots(plot_m: tuple[float, float], terrain_m: tuple[float, float], slot_m: float) -> list[tuple[float, float]]:
    """Centers of a square lattice that fills the ring and stays out of the plot."""
    return _ring_lattice((slot_m, slot_m), plot_m, terrain_m, spacing_scale=1.02)


def perimeter_samples(
    plot_m: tuple[float, float],
    length_m: float,
    thickness_m: float,
) -> list[tuple[tuple[float, float], float]]:
    """Fence segments around the plot edge, inside the yard, clear of the corners.

    North and south segments run along X with yaw 0. East and west run along Y
    with yaw pi/2, and they start after the corner segments so footprints do
    not overlap.
    """
    width, depth = float(plot_m[0]), float(plot_m[1])
    length = max(float(length_m), 0.5)
    thickness = max(float(thickness_m), 0.1)
    inset = 1.0 + thickness / 2.0
    span_x = width - 2.0 * inset
    span_y = depth - 2.0 * inset
    samples: list[tuple[tuple[float, float], float]] = []
    for x in _edge_centers(inset, inset + span_x, length):
        samples.append(((x, inset), 0.0))
        samples.append(((x, depth - inset), 0.0))
    corner = length
    for y in _edge_centers(inset + corner, depth - inset - corner, length):
        samples.append(((inset, y), math.pi / 2.0))
        samples.append(((width - inset, y), math.pi / 2.0))
    return samples


def _edge_centers(start: float, end: float, length: float) -> list[float]:
    span = end - start
    if span < length:
        return [start + span / 2.0] if span > 0 else []
    count = max(1, int(span // length))
    step = span / count
    return [start + step * (index + 0.5) for index in range(count)]


def _place_surround_group(
    assets: list[AssetSpec],
    width: float,
    depth: float,
    terrain_m: tuple[float, float],
    placed: list[Footprint],
) -> dict[str, list[Instance]]:
    """Tile the ring with the large assets, then tuck smaller ones into the gaps."""
    plot_m = (width, depth)
    large = [asset for asset in assets if max(asset.size_m[0], asset.size_m[1]) >= 4.0]
    small = [asset for asset in assets if asset not in large]
    assigned: dict[str, list[Instance]] = {}
    if large:
        slot = max(max(asset.size_m[0], asset.size_m[1]) for asset in large)
        centers = ring_slots(plot_m, terrain_m, slot)
        assigned.update(_deal_slots(large, centers, placed))
    for asset in small:
        centers = _ring_lattice(asset.size_m[:2], plot_m, terrain_m, spacing_scale=1.15)
        samples: list[tuple[tuple[float, float], float]] = []
        for xy in centers:
            if len(samples) >= asset.count:
                break
            yaw = 0.0
            if not in_surround(xy, asset.size_m[:2], plot_m, terrain_m, yaw=yaw):
                continue
            occupied = placed + [
                Footprint(asset.id, len(samples), sample[0], sample[1], asset.size_m[:2]) for sample in samples
            ]
            if _overlaps(xy, yaw, asset.size_m[:2], occupied):
                continue
            samples.append((xy, yaw))
        if len(samples) < asset.count:
            raise RuntimeError(f"{asset.id} surround filled {len(samples)} of {asset.count}")
        instances = []
        for index, (xy, yaw) in enumerate(samples):
            placed.append(Footprint(asset.id, index, xy, yaw, asset.size_m[:2]))
            instances.append(Instance(index=index, xy=xy, yaw=yaw))
        assigned[asset.id] = instances
    return assigned


def _deal_slots(
    assets: list[AssetSpec],
    centers: list[tuple[float, float]],
    placed: list[Footprint],
) -> dict[str, list[Instance]]:
    need = {asset.id: asset.count for asset in assets}
    buckets: dict[str, list[tuple[float, float]]] = {asset.id: [] for asset in assets}
    cursor = 0
    ids = [asset.id for asset in assets]
    for xy in centers:
        for _step in range(len(ids)):
            asset_id = ids[cursor % len(ids)]
            cursor += 1
            if len(buckets[asset_id]) < need[asset_id]:
                buckets[asset_id].append(xy)
                break
        if all(len(buckets[asset.id]) >= asset.count for asset in assets):
            break
    assigned: dict[str, list[Instance]] = {}
    by_id = {asset.id: asset for asset in assets}
    for asset in assets:
        if len(buckets[asset.id]) < asset.count:
            raise RuntimeError(
                f"{asset.id} surround filled {len(buckets[asset.id])} of {asset.count}; "
                f"ring has {len(centers)} slots"
            )
        instances = []
        for index, xy in enumerate(buckets[asset.id]):
            placed.append(Footprint(asset.id, index, xy, 0.0, asset.size_m[:2]))
            instances.append(Instance(index=index, xy=xy, yaw=0.0))
        assigned[asset.id] = instances
        by_id[asset.id] = asset
    return assigned


def _ring_lattice(
    size_m: tuple[float, float],
    plot_m: tuple[float, float],
    terrain_m: tuple[float, float],
    spacing_scale: float,
) -> list[tuple[float, float]]:
    from farmgen.ground import plot_origin

    origin_x, origin_y = plot_origin(terrain_m, plot_m)
    pitch_x = size_m[0] * spacing_scale
    pitch_y = size_m[1] * spacing_scale
    half_w = size_m[0] / 2.0
    half_d = size_m[1] / 2.0
    x0 = -origin_x + half_w
    x1 = terrain_m[0] - origin_x - half_w
    y0 = -origin_y + half_d
    y1 = terrain_m[1] - origin_y - half_d
    centers: list[tuple[float, float]] = []
    y = y0
    while y <= y1 + 1e-6:
        x = x0
        while x <= x1 + 1e-6:
            xy = (float(x), float(y))
            if in_surround(xy, size_m, plot_m, terrain_m, yaw=0.0):
                centers.append(xy)
            x += pitch_x
        y += pitch_y
    return centers


def _surround_samples(
    asset: AssetSpec,
    plot_m: tuple[float, float],
    terrain_m: tuple[float, float],
    rng: np.random.Generator,
    placed: list[Footprint],
) -> list[tuple[tuple[float, float], float]]:
    from farmgen.ground import plot_origin

    origin_x, origin_y = plot_origin(terrain_m, plot_m)
    half_w = asset.size_m[0] / 2.0
    half_d = asset.size_m[1] / 2.0
    x0 = -origin_x + half_w
    x1 = terrain_m[0] - origin_x - half_w
    y0 = -origin_y + half_d
    y1 = terrain_m[1] - origin_y - half_d
    samples: list[tuple[tuple[float, float], float]] = []
    for _index in range(asset.count):
        found = None
        for _try in range(400):
            xy = (float(rng.uniform(x0, x1)), float(rng.uniform(y0, y1)))
            yaw = float(rng.uniform(0.0, math.tau))
            if not in_surround(xy, asset.size_m[:2], plot_m, terrain_m, yaw=yaw):
                continue
            provisional = placed + [
                Footprint(asset.id, len(samples), sample[0], sample[1], asset.size_m[:2]) for sample in samples
            ]
            if _overlaps(xy, yaw, asset.size_m[:2], provisional):
                continue
            found = (xy, yaw)
            break
        if found is None:
            raise RuntimeError(f"no surround pose for {asset.id} after 400 samples")
        samples.append(found)
    return samples


def _sample_unique(
    asset: AssetSpec,
    width: float,
    depth: float,
    margin_m: float,
    rng: np.random.Generator,
    placed: list[Footprint],
) -> tuple[tuple[float, float], float]:
    half_w = asset.size_m[0] / 2.0
    half_d = asset.size_m[1] / 2.0
    x0 = margin_m + half_w
    x1 = width - margin_m - half_w
    y0 = margin_m + half_d
    y1 = depth - margin_m - half_d
    if x1 <= x0 or y1 <= y0:
        raise RuntimeError(f"{asset.id} footprint does not fit in the plot")
    for _ in range(400):
        xy = (float(rng.uniform(x0, x1)), float(rng.uniform(y0, y1)))
        yaw = float(rng.uniform(0.0, math.tau))
        if accept_sample(xy, yaw, asset.size_m[:2], (width, depth), placed, margin_m):
            return xy, yaw
    raise RuntimeError(f"no non-overlapping pose for {asset.id} after 400 samples")


def _line_samples(
    asset: AssetSpec,
    width: float,
    depth: float,
    margin_m: float,
) -> list[tuple[tuple[float, float], float]]:
    span = asset.size_m[0] * asset.count
    usable = width - 2.0 * margin_m
    if span > usable + 1e-6:
        raise RuntimeError(f"{asset.id} line length {span:.2f} m exceeds usable {usable:.2f} m")
    y = margin_m + asset.size_m[1] / 2.0
    if y > depth - margin_m:
        raise RuntimeError(f"{asset.id} line does not fit in depth")
    x = margin_m + asset.size_m[0] / 2.0
    return [((x + i * asset.size_m[0], y), 0.0) for i in range(asset.count)]


def _grid_samples(
    asset: AssetSpec,
    width: float,
    depth: float,
    margin_m: float,
    cols: int,
    origin: tuple[float, float],
) -> list[tuple[tuple[float, float], float]]:
    rows = int(math.ceil(asset.count / cols))
    pitch_x = asset.size_m[0] * 1.15
    pitch_y = asset.size_m[1] * 1.15
    origin_x, origin_y = origin
    samples = []
    for index in range(asset.count):
        row, col = divmod(index, cols)
        xy = (origin_x + col * pitch_x, origin_y + row * pitch_y)
        yaw = 0.0
        if xy[0] > width - margin_m or xy[1] > depth - margin_m:
            raise RuntimeError(f"{asset.id} grid cell {index} falls outside the plot: {xy}")
        samples.append((xy, yaw))
    if rows < 1:
        raise RuntimeError(f"{asset.id} grid is empty")
    return samples


def _corners(xy: tuple[float, float], yaw: float, size_m: tuple[float, float]) -> np.ndarray:
    hx, hy = size_m[0] / 2.0, size_m[1] / 2.0
    local = np.array([[-hx, -hy], [hx, -hy], [hx, hy], [-hx, hy]], dtype=np.float64)
    c, s = math.cos(yaw), math.sin(yaw)
    rot = np.array([[c, -s], [s, c]], dtype=np.float64)
    return local @ rot.T + np.asarray(xy, dtype=np.float64)


def _inside_plot(corners: np.ndarray, plot_m: tuple[float, float], margin_m: float) -> bool:
    return bool(
        np.all(corners[:, 0] >= margin_m - 1e-8)
        and np.all(corners[:, 0] <= plot_m[0] - margin_m + 1e-8)
        and np.all(corners[:, 1] >= margin_m - 1e-8)
        and np.all(corners[:, 1] <= plot_m[1] - margin_m + 1e-8)
    )


def _polygon_area(poly: np.ndarray) -> float:
    if poly.shape[0] < 3:
        return 0.0
    x, y = poly[:, 0], poly[:, 1]
    return float(abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))) * 0.5)


def _clip(subject: np.ndarray, clipper: np.ndarray) -> np.ndarray:
    output = subject
    for i in range(clipper.shape[0]):
        if output.shape[0] == 0:
            return output
        a = clipper[i]
        b = clipper[(i + 1) % clipper.shape[0]]
        nxt = []
        prev = output[-1]
        for cur in output:
            if _left(a, b, cur):
                if not _left(a, b, prev):
                    nxt.append(_intersect(prev, cur, a, b))
                nxt.append(cur)
            elif _left(a, b, prev):
                nxt.append(_intersect(prev, cur, a, b))
            prev = cur
        output = np.asarray(nxt, dtype=np.float64) if nxt else np.zeros((0, 2))
    return output


def _left(a: np.ndarray, b: np.ndarray, p: np.ndarray) -> bool:
    return float((b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0])) >= -1e-10


def _intersect(p1: np.ndarray, p2: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    r = p2 - p1
    s = b - a
    denom = float(r[0] * s[1] - r[1] * s[0])
    if abs(denom) < 1e-15:
        return p2
    t = float(((a[0] - p1[0]) * s[1] - (a[1] - p1[1]) * s[0]) / denom)
    return p1 + t * r
