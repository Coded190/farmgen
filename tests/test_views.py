"""Four yaw cameras and four corner cameras, with the filenames the critic reads."""

from farmgen.views import (
    ASSET_YAWS,
    asset_view_filename,
    compose_view_filename,
    corner_locations,
    orbit_locations,
)


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
