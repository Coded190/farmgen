"""Reference selection and proposed poses. No splat library and no vLLM."""

from pathlib import Path

import pytest

from farmgen.placer import place_scene
from farmgen.poses import apply_proposals, seat_visible
from farmgen.reference import resolve_reference
from farmgen.scene import AssetSpec, GroundSpec, Scene
from farmgen.views import reference_filenames


def _scene() -> Scene:
    return Scene(
        id="farm_test",
        seed=0,
        brief="two props",
        ground=GroundSpec(size_m=(40.0, 40.0)),
        assets=[
            AssetSpec(id="barn_01", prompt="a red wooden barn", size_m=(12.0, 8.0, 7.0), count=1, layout="unique"),
            AssetSpec(id="silo_01", prompt="a metal silo", size_m=(5.0, 5.0, 12.0), count=1, layout="unique"),
        ],
    )


def test_reference_dir_names_a_missing_file_and_splat_does_not_read_it(tmp_path: Path):
    folder = tmp_path / "refs"
    folder.mkdir()
    (folder / "reference_top.png").write_bytes(b"png")
    with pytest.raises(FileNotFoundError, match="reference_view_0.png"):
        resolve_reference("farm_test", (120.0, 120.0), None, folder)

    ply = tmp_path / "field.ply"
    ply.write_bytes(b"ply")
    seen: dict[str, Path] = {}

    def _fake_render(path: Path, directory: Path, terrain_m: tuple[float, float]) -> None:
        seen["ply"] = path
        seen["dir"] = directory
        assert terrain_m == (120.0, 120.0)
        for name in reference_filenames():
            (directory / name).write_bytes(b"png")

    resolved = resolve_reference("farm_test", (120.0, 120.0), ply, folder, render_fn=_fake_render)
    assert seen["ply"] == ply
    assert resolved == seen["dir"]
    assert resolved != folder
    assert all((resolved / name).is_file() for name in reference_filenames())


def test_unposed_copies_stay_empty_when_the_lattice_is_off():
    scene = _scene()
    apply_proposals(scene, [{"id": "barn_01", "index": 0, "u": 0.5, "v": 0.5, "yaw_deg": 90}])
    place_scene(scene, lattice=False)
    assert scene.asset("barn_01").instances[0].xy == (20.0, 20.0)
    assert scene.asset("silo_01").instances == []


def test_proposal_skips_the_lattice_for_that_asset_only():
    scene = _scene()
    kept = apply_proposals(
        scene,
        [{"id": "barn_01", "index": 0, "u": 0.5, "v": 0.5, "yaw_deg": 90}],
    )
    assert kept == {"barn_01"}
    barn_xy = scene.asset("barn_01").instances[0].xy
    assert barn_xy == (20.0, 20.0)
    place_scene(scene, keep=kept)
    assert scene.asset("barn_01").instances[0].xy == barn_xy
    assert len(scene.asset("silo_01").instances) == 1
    assert scene.asset("silo_01").instances[0].xy != barn_xy


def test_stacked_copies_are_seated_apart_and_a_ring_stays_outside_the_yard():
    from farmgen.scene import Instance

    scene = _scene()
    scene.assets.append(
        AssetSpec(
            id="trees_01",
            prompt="a ring of trees",
            size_m=(8.0, 8.0, 6.0),
            count=1,
            layout="surround",
        )
    )
    scene.asset("barn_01").instances = [Instance(index=0, xy=(20.0, 20.0), yaw=0.0)]
    scene.asset("silo_01").instances = [Instance(index=0, xy=(20.0, 20.0), yaw=0.0)]
    seat_visible(scene, {"barn_01", "silo_01", "trees_01"})
    barn = scene.asset("barn_01").instances[0].xy
    silo = scene.asset("silo_01").instances[0].xy
    trees = scene.asset("trees_01").instances[0].xy
    assert barn == (20.0, 20.0)
    assert silo != barn
    assert not (0.0 <= trees[0] <= 40.0 and 0.0 <= trees[1] <= 40.0)
    from farmgen.placer import Footprint, footprint_issue

    blocked = [Footprint("barn_01", 0, barn, 0.0, (12.0, 8.0))]
    assert (
        footprint_issue(silo, 0.0, (5.0, 5.0), "unique", scene.ground.plot_m, scene.ground.terrain_m, blocked)
        is None
    )


def test_no_flags_returns_no_reference():
    assert resolve_reference("farm_test", (120.0, 120.0), None, None) is None
