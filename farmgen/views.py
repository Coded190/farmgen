"""Camera angles and filenames for the four-view critic.

This module stays free of Blender so tests can check the contract without a
display. The render worker loads it by file path.
"""

from __future__ import annotations

import math

ASSET_YAWS: tuple[float, ...] = (0.0, 90.0, 180.0, 270.0)
CORNER_COUNT = 4
RENDER_WIDTH = 960
RENDER_HEIGHT = 540
ORTHO_MARGIN = 1.08
REFERENCE_TOP = "reference_top.png"


def asset_view_filename(asset_id: str, yaw_deg: float) -> str:
    return f"{asset_id}_yaw{int(yaw_deg)}.png"


def reference_view_filename(index: int) -> str:
    if index < 0 or index >= CORNER_COUNT:
        raise ValueError(f"corner index {index} is outside 0..{CORNER_COUNT - 1}")
    return f"reference_view_{index}.png"


def reference_filenames() -> tuple[str, ...]:
    """The five pictures a splat render or a reference directory must contain."""
    return (REFERENCE_TOP, *(reference_view_filename(index) for index in range(CORNER_COUNT)))


def ortho_scale(size_m: tuple[float, float]) -> float:
    """Blender ortho_scale for a top view. It is the wider image axis, in meters."""
    span = max(float(size_m[0]), float(size_m[1]))
    aspect = RENDER_WIDTH / RENDER_HEIGHT
    return span * aspect * ORTHO_MARGIN


def ortho_view_size(size_m: tuple[float, float]) -> tuple[float, float]:
    """Width and height of the orthographic top image, in meters."""
    scale = ortho_scale(size_m)
    aspect = RENDER_WIDTH / RENDER_HEIGHT
    return scale, scale / aspect


def uv_to_world_xy(u: float, v: float, size_m: tuple[float, float]) -> tuple[float, float]:
    """Map a point on the orthographic top image to terrain meters.

    ``u`` and ``v`` run from 0 to 1. ``v`` 0 is the top of the image, which is
    +Y, because the camera looks down with world +Y as its up axis.
    """
    if u < 0.0 or u > 1.0 or v < 0.0 or v > 1.0:
        raise ValueError(f"u,v must be from 0 to 1, got {u},{v}")
    width, height = ortho_view_size(size_m)
    center_x = float(size_m[0]) / 2.0
    center_y = float(size_m[1]) / 2.0
    x = center_x + (float(u) - 0.5) * width
    y = center_y + (0.5 - float(v)) * height
    return x, y


def uv_to_plot_xy(
    u: float,
    v: float,
    terrain_m: tuple[float, float],
    plot_m: tuple[float, float],
) -> tuple[float, float]:
    """Terrain image point to meters from the plot corner."""
    from farmgen.ground import plot_origin

    world_x, world_y = uv_to_world_xy(u, v, terrain_m)
    origin_x, origin_y = plot_origin(terrain_m, plot_m)
    return world_x - origin_x, world_y - origin_y


def pose_in_bounds(
    xy: tuple[float, float],
    layout: str,
    terrain_m: tuple[float, float],
    plot_m: tuple[float, float],
) -> bool:
    """A yard pose stays in the plot. A surround pose stays in the ring."""
    from farmgen.ground import plot_origin

    x, y = float(xy[0]), float(xy[1])
    if layout == "surround":
        origin_x, origin_y = plot_origin(terrain_m, plot_m)
        world_x, world_y = x + origin_x, y + origin_y
        in_terrain = 0.0 <= world_x <= float(terrain_m[0]) and 0.0 <= world_y <= float(terrain_m[1])
        in_plot = origin_x <= world_x <= origin_x + float(plot_m[0]) and origin_y <= world_y <= origin_y + float(plot_m[1])
        return in_terrain and not in_plot
    return 0.0 <= x <= float(plot_m[0]) and 0.0 <= y <= float(plot_m[1])


def sheet_cameras(
    mins: tuple[float, float, float],
    maxs: tuple[float, float, float],
) -> dict[str, tuple[float, ...] | float | int | str]:
    """Look along the short axis so a vertical sheet shows its face.

    The compose cameras look down Z. A raw Infinigen tile whose short axis is
    Y is edge-on from those cameras. This frame looks along that short axis.
    """
    extent = [float(maxs[index]) - float(mins[index]) for index in range(3)]
    short = int(min(range(3), key=lambda index: extent[index]))
    center = tuple((float(mins[index]) + float(maxs[index])) / 2.0 for index in range(3))
    pull = max(extent) * 1.2
    eye = list(center)
    eye[short] = center[short] + pull
    longs = [index for index in range(3) if index != short]
    corner = list(center)
    corner[longs[0]] = float(maxs[longs[0]]) + extent[longs[0]] * 0.15
    corner[longs[1]] = float(mins[longs[1]]) - extent[longs[1]] * 0.15
    corner[short] = center[short] + pull * 0.35
    span = max(extent[longs[0]], extent[longs[1]], 1e-6)
    return {
        "short_axis": short,
        "up_axis": "Y" if short == 2 else "Z",
        "top_location": tuple(eye),
        "top_target": center,
        "corner_location": tuple(corner),
        "corner_target": center,
        "ortho_scale": span * (RENDER_WIDTH / RENDER_HEIGHT) * ORTHO_MARGIN,
    }


def write_ortho_debug(
    path,
    terrain_m: tuple[float, float] = (120.0, 120.0),
    plot_m: tuple[float, float] = (40.0, 40.0),
) -> None:
    """Draw the ortho frame and mark the image center, which is the terrain center."""
    from PIL import Image, ImageDraw

    image = Image.new("RGB", (RENDER_WIDTH, RENDER_HEIGHT), (230, 230, 230))
    draw = ImageDraw.Draw(image)
    width_m, height_m = ortho_view_size(terrain_m)

    def _pixel(u: float, v: float) -> tuple[int, int]:
        return int(round(u * (RENDER_WIDTH - 1))), int(round(v * (RENDER_HEIGHT - 1)))

    def _uv_for_world(x: float, y: float) -> tuple[float, float]:
        center_x = float(terrain_m[0]) / 2.0
        center_y = float(terrain_m[1]) / 2.0
        u = 0.5 + (x - center_x) / width_m
        v = 0.5 - (y - center_y) / height_m
        return u, v

    from farmgen.ground import plot_origin

    origin_x, origin_y = plot_origin(terrain_m, plot_m)
    corners = (
        (origin_x, origin_y),
        (origin_x + plot_m[0], origin_y),
        (origin_x + plot_m[0], origin_y + plot_m[1]),
        (origin_x, origin_y + plot_m[1]),
    )
    plot_px = [_pixel(*_uv_for_world(x, y)) for x, y in corners]
    draw.polygon(plot_px, outline=(122, 74, 36))
    center = _pixel(0.5, 0.5)
    draw.ellipse((center[0] - 6, center[1] - 6, center[0] + 6, center[1] + 6), fill=(180, 40, 40))
    # Terrain corner (0, 0), the second marked point.
    corner_uv = _uv_for_world(0.0, 0.0)
    corner_px = _pixel(*corner_uv)
    draw.ellipse((corner_px[0] - 6, corner_px[1] - 6, corner_px[0] + 6, corner_px[1] + 6), fill=(20, 80, 160))
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)


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


def eye_locations(
    terrain_m: tuple[float, float],
    plot_m: tuple[float, float],
    eye_height_m: float = 6.0,
) -> list[tuple[tuple[float, float, float], tuple[float, float, float], str]]:
    """Two cameras above the yard, looking at the barn and along the fence.

    Coordinates are terrain meters. The height stays above the ring so the
    camera is not buried in the ground.
    """
    origin_x = (float(terrain_m[0]) - float(plot_m[0])) / 2.0
    origin_y = (float(terrain_m[1]) - float(plot_m[1])) / 2.0
    width, depth = float(plot_m[0]), float(plot_m[1])
    center_x = origin_x + width / 2.0
    center_y = origin_y + depth / 2.0
    height = max(float(eye_height_m), 6.0)
    barn = (center_x, center_y, 1.5)
    # Stand inside the yard, clear of a barn that fills the middle, so the
    # fence and the barn are both in frame.
    return [
        ((origin_x + 6.0, origin_y + 8.0, height + 6.0), barn, "eye_a"),
        ((origin_x + 8.0, origin_y + 3.0, height), (origin_x + width - 8.0, origin_y + 3.0, 1.0), "eye_b"),
    ]
