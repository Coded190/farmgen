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
_AREA_EPS = 1e-8


@dataclass(frozen=True)
class Footprint:
    asset_id: str
    index: int
    xy: tuple[float, float]
    yaw: float
    size_m: tuple[float, float]


def place_scene(scene: Scene, margin_m: float = MARGIN_M) -> list[str]:
    rng = np.random.default_rng(scene.seed)
    placed: list[Footprint] = []
    lines: list[str] = []
    width, depth = scene.ground.size_m
    for asset in scene.assets:
        instances = _place_asset(asset, width, depth, margin_m, rng, placed)
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
    candidate = corners
    for other in existing:
        other_corners = _corners(other.xy, other.yaw, other.size_m)
        if _polygon_area(_clip(candidate, other_corners)) > _AREA_EPS:
            return False
    return True


def _place_asset(
    asset: AssetSpec,
    width: float,
    depth: float,
    margin_m: float,
    rng: np.random.Generator,
    placed: list[Footprint],
) -> list[Instance]:
    if asset.layout == "unique":
        samples = [_sample_unique(asset, width, depth, margin_m, rng, placed)]
    elif asset.layout == "line":
        samples = _line_samples(asset, width, depth, margin_m)
    elif asset.layout == "grid":
        samples = _grid_samples(asset, width, depth, margin_m, cols=3, origin=(28.0, 30.0))
    elif asset.layout == "rows":
        samples = _grid_samples(asset, width, depth, margin_m, cols=4, origin=(8.0, 14.0))
    else:
        raise ValueError(f"unknown layout {asset.layout}")

    instances: list[Instance] = []
    for index, (xy, yaw) in enumerate(samples):
        if not accept_sample(xy, yaw, asset.size_m[:2], (width, depth), placed, margin_m):
            raise RuntimeError(
                f"placement rejected id={asset.id} instance={index} xy={xy[0]:.3f},{xy[1]:.3f} yaw={yaw:.3f}"
            )
        placed.append(Footprint(asset.id, index, xy, yaw, asset.size_m[:2]))
        instances.append(Instance(index=index, xy=xy, yaw=yaw))
    if len(instances) != asset.count:
        raise RuntimeError(f"{asset.id} placed {len(instances)} of {asset.count}")
    return instances


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
