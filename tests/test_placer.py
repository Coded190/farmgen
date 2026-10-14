from farmgen.placer import Footprint, accept_sample, place_scene
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
            assert accept_sample(inst.xy, inst.yaw, asset.size_m[:2], scene.ground.size_m, [], margin_m=1.0)

    barn = scene.asset("barn_01")
    overlap = Footprint("barn_01", 0, barn.instances[0].xy, barn.instances[0].yaw, barn.size_m[:2])
    assert not accept_sample(overlap.xy, overlap.yaw, overlap.size_m, scene.ground.size_m, [overlap])
    assert not accept_sample((0.2, 0.2), 0.0, (2.0, 2.0), (40.0, 40.0), [])
    assert not accept_sample((39.5, 20.0), 0.0, (2.0, 2.0), (40.0, 40.0), [])
