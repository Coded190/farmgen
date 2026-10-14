"""Camera angles and filenames for the four-view critic.

This module stays free of Blender so tests can check the contract without a
display. The render worker loads it by file path.
"""

from __future__ import annotations

import math

ASSET_YAWS: tuple[float, ...] = (0.0, 90.0, 180.0, 270.0)
CORNER_COUNT = 4


def asset_view_filename(asset_id: str, yaw_deg: float) -> str:
    return f"{asset_id}_yaw{int(yaw_deg)}.png"


def compose_view_filename(index: int) -> str:
    if index < 0 or index >= CORNER_COUNT:
        raise ValueError(f"corner index {index} is outside 0..{CORNER_COUNT - 1}")
    return f"compose_view_{index}.png"


def orbit_locations(
    center: tuple[float, float, float],
    radius: float,
    yaws: tuple[float, ...] = ASSET_YAWS,
    elevation_deg: float = 25.0,
) -> list[tuple[tuple[float, float, float], tuple[float, float, float], float]]:
    """Cameras looking at ``center`` from ``yaws`` degrees in the XY plane."""
    elev = math.radians(elevation_deg)
    horizontal = radius * math.cos(elev)
    height = radius * math.sin(elev)
    cameras = []
    for yaw in yaws:
        theta = math.radians(yaw)
        location = (
            center[0] + horizontal * math.cos(theta),
            center[1] + horizontal * math.sin(theta),
            center[2] + height,
        )
        cameras.append((location, center, float(yaw)))
    return cameras


def corner_locations(
    size_m: tuple[float, float],
    eye_height_m: float | None = None,
) -> list[tuple[tuple[float, float, float], tuple[float, float, float], int]]:
    """Four cameras outside the rectangle corners, looking at its center."""
    width, depth = float(size_m[0]), float(size_m[1])
    center = (width / 2.0, depth / 2.0, 0.0)
    height = eye_height_m if eye_height_m is not None else max(width, depth) * 0.45
    pull = max(width, depth) * 0.15
    corners = (
        (-pull, -pull),
        (width + pull, -pull),
        (width + pull, depth + pull),
        (-pull, depth + pull),
    )
    cameras = []
    for index, (x, y) in enumerate(corners):
        cameras.append(((x, y, height), center, index))
    return cameras
