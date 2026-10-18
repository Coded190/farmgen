from farmgen.placer import Footprint, accept_sample, in_surround, place_scene
from farmgen.scene import starter_scene


def test_placer_log_overlap_and_bounds(capsys):
    scene = starter_scene()
    lines = place_scene(scene)
    captured = capsys.readouterr().out
    assert "placed id=hay_01 instance=3" in captured
    assert any(line.startswith("FARMGEN") and "placed id=hay_01 instance=3" in line for line in lines)

    for asset in scene.assets:
        assert len(asset.instances) == asset.count
        for inst in asset.instances:
            if asset.layout == "surround":
                assert in_surround(
                    inst.xy,
                    asset.size_m[:2],
                    scene.ground.plot_m,
                    scene.ground.terrain_m,
                    yaw=inst.yaw,
                )
            else:
                assert accept_sample(inst.xy, inst.yaw, asset.size_m[:2], scene.ground.size_m, [], margin_m=1.0)

    barn = scene.asset("barn_01")
    overlap = Footprint("barn_01", 0, barn.instances[0].xy, barn.instances[0].yaw, barn.size_m[:2])
    assert not accept_sample(overlap.xy, overlap.yaw, overlap.size_m, scene.ground.size_m, [overlap])
    assert not accept_sample((0.2, 0.2), 0.0, (2.0, 2.0), (40.0, 40.0), [])
    assert not accept_sample((39.5, 20.0), 0.0, (2.0, 2.0), (40.0, 40.0), [])


def test_trees_cover_the_ring_outside_the_yard():
    scene = starter_scene()
    place_scene(scene)
    trees = [asset for asset in scene.assets if asset.id.startswith("tree_")]
    copies = [inst for asset in trees for inst in asset.instances]
    assert len(copies) >= 100
    area = 0.0
    for asset in trees:
        area += asset.size_m[0] * asset.size_m[1] * len(asset.instances)
        for inst in asset.instances:
            assert in_surround(inst.xy, asset.size_m[:2], scene.ground.plot_m, scene.ground.terrain_m, yaw=inst.yaw)
    ring = scene.ground.terrain_m[0] * scene.ground.terrain_m[1] - scene.ground.plot_m[0] * scene.ground.plot_m[1]
    assert area / ring >= 0.7


def test_surround_rejects_a_sample_inside_the_plot():
    plot = (40.0, 40.0)
    terrain = (120.0, 120.0)
    assert not in_surround((20.0, 20.0), (6.0, 6.0), plot, terrain)
    assert in_surround((-20.0, 20.0), (6.0, 6.0), plot, terrain)


def test_fence_copies_lie_on_the_plot_perimeter():
    import math

    from farmgen.placer import footprint_issue
    from farmgen.poses import place_fences
    from farmgen.scene import AssetSpec, GroundSpec, Scene, normalize_catalog

    scene = Scene(
        id="farm_test",
        seed=1,
        brief="a fenced yard",
        ground=GroundSpec(size_m=(40.0, 40.0)),
        assets=[
            AssetSpec(
                id="fence_01",
                prompt="a fence post",
                size_m=(0.1, 0.1, 2.0),
                count=100,
                layout="line",
            )
        ],
    )
    normalize_catalog(scene)
    fence = scene.asset("fence_01")
    assert fence.layout == "perimeter"
    assert fence.size_m[0] >= 2.0
    place_fences(scene)
    points = fence.instances
    assert len(points) == fence.count
    assert len(points) >= 8
    xs = [inst.xy[0] for inst in points]
    ys = [inst.xy[1] for inst in points]
    assert min(xs) < 5.0 and max(xs) > 35.0
    assert min(ys) < 5.0 and max(ys) > 35.0
    occupied = []
    from farmgen.placer import Footprint

    for inst in points:
        issue = footprint_issue(
            inst.xy,
            inst.yaw,
            fence.size_m[:2],
            "perimeter",
            scene.ground.plot_m,
            scene.ground.terrain_m,
            occupied,
        )
        assert issue is None, issue
        occupied.append(Footprint("fence_01", inst.index, inst.xy, inst.yaw, fence.size_m[:2]))
    assert any(abs(inst.yaw - math.pi / 2.0) < 1e-6 for inst in points)


def test_tree_copies_are_spaced_around_the_yard():
    import math

    from farmgen.placer import ring_slots
    from farmgen.poses import seat_visible
    from farmgen.scene import AssetSpec, GroundSpec, Scene, normalize_catalog

    scene = Scene(
        id="farm_test",
        seed=1,
        brief="trees",
        ground=GroundSpec(size_m=(40.0, 40.0)),
        assets=[
            AssetSpec(
                id="tree_ring_01",
                prompt="a ring of trees",
                size_m=(8.0, 8.0, 10.0),
                count=5,
                layout="surround",
            )
        ],
    )
    normalize_catalog(scene)
    tree = scene.asset("tree_ring_01")
    assert tree.prompt.startswith("one leafy")
    assert tree.count == len(ring_slots((40.0, 40.0), (120.0, 120.0), 8.0))
    seat_visible(scene, {tree.id})
    assert len(tree.instances) == tree.count
    angles = [math.atan2(inst.xy[1] - 20.0, inst.xy[0] - 20.0) for inst in tree.instances]
    assert max(angles) - min(angles) > math.pi
    south = [inst for inst in tree.instances if inst.xy[1] < 0.0]
    assert len(south) < len(tree.instances)
    outside = [inst.xy[0] < 0.0 or inst.xy[1] < 0.0 or inst.xy[0] > 40.0 or inst.xy[1] > 40.0 for inst in tree.instances]
    assert all(outside)


def test_a_prompt_that_names_a_tree_and_a_bush_is_stored_as_one_tree():
    from farmgen.placer import ring_slots
    from farmgen.scene import AssetSpec, GroundSpec, Scene, normalize_catalog

    scene = Scene(
        id="farm_test",
        seed=1,
        brief="trees",
        ground=GroundSpec(size_m=(40.0, 40.0)),
        assets=[
            AssetSpec(
                id="tree_ring_01",
                prompt="a ring of trees and bushes around the yard",
                size_m=(8.0, 8.0, 10.0),
                count=5,
                layout="surround",
            ),
            AssetSpec(
                id="bush_ring_01",
                prompt="a ring of bushes around the yard",
                size_m=(6.0, 6.0, 2.0),
                count=4,
                layout="surround",
            ),
        ],
    )
    normalize_catalog(scene)
    tree = scene.asset("tree_ring_01")
    bush = scene.asset("bush_ring_01")
    slots = len(ring_slots((40.0, 40.0), (120.0, 120.0), 8.0))
    assert tree.prompt.startswith("one leafy deciduous tree")
    assert "bush" not in tree.prompt
    assert bush.prompt.startswith("one rounded green bush")
    assert "tree" not in bush.prompt
    assert bush.count == max(4, slots // 4)
    assert tree.count == max(2, slots - bush.count)
    assert tree.count != bush.count


def test_a_yard_hole_is_the_open_cell_not_the_southwest_corner():
    import math

    from farmgen.poses import seat_visible
    from farmgen.scene import AssetSpec, GroundSpec, Scene

    scene = Scene(
        id="farm_test",
        seed=1,
        brief="yard",
        ground=GroundSpec(size_m=(40.0, 40.0)),
        assets=[
            AssetSpec(
                id="silo_01",
                prompt="a silo",
                size_m=(4.0, 4.0, 12.0),
                count=1,
                layout="unique",
            ),
            AssetSpec(
                id="tractor_01",
                prompt="a tractor",
                size_m=(4.0, 2.0, 2.5),
                count=1,
                layout="unique",
            ),
        ],
    )
    seat_visible(scene, {"silo_01"})
    silo = scene.asset("silo_01").instances[0].xy
    assert math.hypot(silo[0] - 20.0, silo[1] - 20.0) < 5.0
    scene.assets[1].instances = []
    seat_visible(scene, {"tractor_01"})
    tractor = scene.asset("tractor_01").instances[0].xy
    assert math.hypot(tractor[0] - silo[0], tractor[1] - silo[1]) > 10.0


def test_a_grid_fills_the_yard_and_a_full_lattice_logs_once(monkeypatch):
    from farmgen.poses import place_grids, seat_visible
    from farmgen.scene import AssetSpec, GroundSpec, Instance, Scene

    scene = Scene(
        id="grid_test",
        seed=1,
        brief="pasture",
        ground=GroundSpec(size_m=(4.0, 4.0), terrain_m=(12.0, 12.0)),
        assets=[
            AssetSpec(
                id="pasture_crop",
                prompt="a green pasture plant",
                size_m=(1.0, 1.0, 0.5),
                count=20,
                layout="grid",
            )
        ],
    )
    place_grids(scene, {"pasture_crop"})
    assert len(scene.asset("pasture_crop").instances) == 16

    blocked = Scene(
        id="grid_test",
        seed=1,
        brief="yard",
        ground=GroundSpec(size_m=(4.0, 4.0), terrain_m=(12.0, 12.0)),
        assets=[
            AssetSpec(
                id="barn_01",
                prompt="a wooden barn",
                size_m=(4.0, 4.0, 4.0),
                count=1,
                layout="unique",
                instances=[Instance(index=0, xy=(2.0, 2.0), yaw=0.0)],
            ),
            AssetSpec(
                id="plant",
                prompt="a plant",
                size_m=(1.0, 1.0, 1.0),
                count=5,
                layout="rows",
            ),
        ],
    )
    notes = []

    def capture(_scene_id, _seed, _stage, **fields):
        notes.append(fields)

    monkeypatch.setattr("farmgen.poses.log_line", capture)
    seat_visible(blocked, {"plant"})
    unplaced = [item for item in notes if item.get("note") == "unplaced"]
    assert unplaced == [{"id": "plant", "note": "unplaced", "count": 5}]
    assert blocked.asset("plant").instances == []


def test_crops_sit_in_the_south_and_clear_the_barn():
    from farmgen.placer import Footprint, footprint_issue
    from farmgen.poses import place_fields
    from farmgen.scene import AssetSpec, GroundSpec, Instance, Scene

    barn = AssetSpec(
        id="barn_01",
        prompt="a wooden barn",
        size_m=(6.0, 6.0, 7.0),
        count=1,
        layout="unique",
        instances=[Instance(index=0, xy=(20.0, 30.0), yaw=0.0)],
    )
    scene = Scene(
        id="farm_test",
        seed=0,
        brief="crops",
        ground=GroundSpec(size_m=(40.0, 40.0)),
        assets=[
            barn,
            AssetSpec(
                id="pasture_crop",
                prompt="a green pasture plant",
                size_m=(0.7, 0.6, 0.5),
                count=1,
                layout="rows",
                field_m=(12.0, 16.0),
                field_origin=(1.0, 2.0),
            ),
            AssetSpec(
                id="corn_01",
                prompt="a corn stalk",
                size_m=(0.6, 0.6, 1.8),
                count=1,
                layout="rows",
                field_m=(12.0, 16.0),
                field_origin=(14.0, 2.0),
            ),
            AssetSpec(
                id="wheat_01",
                prompt="a wheat plant",
                size_m=(0.8, 0.8, 1.0),
                count=1,
                layout="rows",
                field_m=(12.0, 16.0),
                field_origin=(27.0, 2.0),
            ),
        ],
    )
    place_fields(scene)
    occupied = [Footprint("barn_01", 0, barn.instances[0].xy, 0.0, barn.size_m[:2])]
    xs = []
    for asset in scene.assets[1:]:
        assert asset.instances
        assert all(inst.xy[1] < 18.0 for inst in asset.instances)
        xs.append(sum(inst.xy[0] for inst in asset.instances) / len(asset.instances))
        for inst in asset.instances:
            assert (
                footprint_issue(inst.xy, 0.0, asset.size_m[:2], "rows", scene.ground.plot_m, scene.ground.terrain_m, occupied)
                is None
            )
    assert xs[0] < xs[1] < xs[2]


def test_a_pose_inside_the_crop_band_is_rejected():
    from farmgen.poses import apply_proposals, seat_visible
    from farmgen.scene import AssetSpec, GroundSpec, Scene
    from farmgen.views import uv_to_plot_xy

    scene = Scene(
        id="farm_test",
        seed=0,
        brief="yard",
        ground=GroundSpec(size_m=(40.0, 40.0), terrain_m=(120.0, 120.0)),
        assets=[
            AssetSpec(
                id="pasture_crop",
                prompt="a green pasture plant",
                size_m=(1.0, 1.0, 0.5),
                count=1,
                layout="rows",
                field_m=(12.0, 16.0),
                field_origin=(1.0, 2.0),
            ),
            AssetSpec(id="barn_01", prompt="a wooden barn", size_m=(4.0, 4.0, 6.0), count=1, layout="unique"),
        ],
    )
    south = None
    for step in range(21):
        v = step / 20.0
        xy = uv_to_plot_xy(0.5, v, scene.ground.terrain_m, scene.ground.plot_m)
        if 2.0 < xy[1] < 16.0 and 2.0 < xy[0] < 38.0:
            south = (0.5, v)
            break
    assert south is not None
    problems: list = []
    apply_proposals(scene, [{"id": "barn_01", "index": 0, "u": south[0], "v": south[1], "yaw_deg": 0}], problems)
    assert scene.asset("barn_01").instances == []
    assert problems[0][2] == "inside the crop band"
    seat_visible(scene, {"barn_01"})
    assert scene.asset("barn_01").instances[0].xy[1] >= 18.0

