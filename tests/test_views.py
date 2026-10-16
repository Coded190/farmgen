"""Four yaw cameras and four corner cameras, with the filenames the critic reads."""

from farmgen.views import (
    ASSET_YAWS,
    asset_view_filename,
    compose_view_filename,
    corner_locations,
    orbit_locations,
)


def test_eye_cameras_look_at_the_yard_from_above_the_ring():
    from farmgen.views import eye_locations

    cameras = eye_locations((120.0, 120.0), (40.0, 40.0))
    assert [name for _location, _target, name in cameras] == ["eye_a", "eye_b"]
    for location, target, _name in cameras:
        assert location[2] > 4.0
        assert 40.0 <= location[0] <= 80.0
        assert 40.0 <= location[1] <= 80.0
        assert 40.0 <= target[0] <= 80.0
        assert 40.0 <= target[1] <= 80.0
    assert cameras[0][0][:2] != (14.0, 8.0)


def test_asset_views_are_four_right_angles():
    cameras = orbit_locations((0.0, 0.0, 0.0), radius=4.0)
    assert [yaw for _location, _target, yaw in cameras] == list(ASSET_YAWS)
    names = [asset_view_filename("tractor_01", yaw) for _location, _target, yaw in cameras]
    assert names == [
        "tractor_01_yaw0.png",
        "tractor_01_yaw90.png",
        "tractor_01_yaw180.png",
        "tractor_01_yaw270.png",
    ]
    # Each camera sits on a different side of the target.
    sides = []
    for location, target, _yaw in cameras:
        sides.append((round(location[0] - target[0], 5), round(location[1] - target[1], 5)))
    assert len(set(sides)) == 4


def test_top_image_center_is_the_plot_center_and_a_corner_stays_put():
    from farmgen.views import uv_to_plot_xy, uv_to_world_xy, write_ortho_debug
    from farmgen.paths import output_dir

    terrain = (120.0, 120.0)
    plot = (40.0, 40.0)
    world = uv_to_world_xy(0.5, 0.5, terrain)
    assert world == (60.0, 60.0)
    plot_xy = uv_to_plot_xy(0.5, 0.5, terrain, plot)
    assert plot_xy == (20.0, 20.0)
    corner = uv_to_world_xy(*_uv_for_world(0.0, 0.0, terrain), terrain)
    assert abs(corner[0]) < 1e-6
    assert abs(corner[1]) < 1e-6
    dest = output_dir("farm_000") / "ortho_uv_debug.png"
    write_ortho_debug(dest, terrain, plot)
    assert dest.is_file() and dest.stat().st_size > 1000


def _uv_for_world(x: float, y: float, terrain: tuple[float, float]) -> tuple[float, float]:
    from farmgen.views import ortho_view_size

    width_m, height_m = ortho_view_size(terrain)
    u = 0.5 + (x - terrain[0] / 2.0) / width_m
    v = 0.5 - (y - terrain[1] / 2.0) / height_m
    return u, v


def test_sheet_camera_looks_along_the_short_axis():
    from farmgen.views import sheet_cameras

    frame = sheet_cameras((0.0, -4.0, -120.0), (122.0, 10.0, 0.0))
    assert frame["short_axis"] == 1
    assert frame["up_axis"] == "Z"
    top = frame["top_location"]
    target = frame["top_target"]
    assert top[1] > target[1]


def test_compose_views_are_the_four_corners():
    cameras = corner_locations((120.0, 80.0))
    assert [index for _location, _target, index in cameras] == [0, 1, 2, 3]
    assert [compose_view_filename(index) for index in range(4)] == [
        "compose_view_0.png",
        "compose_view_1.png",
        "compose_view_2.png",
        "compose_view_3.png",
    ]
    xs = [location[0] for location, _target, _index in cameras]
    ys = [location[1] for location, _target, _index in cameras]
    assert min(xs) < 0 < max(xs)
    assert min(ys) < 0 < max(ys)
