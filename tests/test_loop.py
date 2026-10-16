"""Redo actions map back onto the module that should run again."""

from farmgen.critic import RedoAction, Verdict
from farmgen.loop import _restart, _review_stage, modules_for
from farmgen.scene import Instance, save_scene, starter_scene


def test_reorient_does_not_schedule_image_or_trellis():
    actions = [RedoAction(action="reorient", id="tractor_01", label="wrong_orientation")]
    assert modules_for(actions) == ["orient", "place", "compose"]


def test_image_redo_comes_before_placement():
    actions = [
        RedoAction(action="redo_place", id="place", label="floating"),
        RedoAction(action="redo_image", id="barn_01", label="missing"),
        RedoAction(action="redo_ground", id="ground", label="not_agricultural"),
    ]
    assert modules_for(actions) == ["ground", "image", "place", "compose"]


def test_a_bad_mesh_regenerates_before_it_is_placed_again():
    actions = [RedoAction(action="redo_asset", id="tree_02", label="missing")]
    assert modules_for(actions) == ["image", "place", "compose"]


def test_move_restamps_and_composes_without_the_lattice():
    actions = [RedoAction(action="move", id="barn_01", label="wrong_place", yaw_deg=15.0, xy=(20.0, 20.0))]
    assert modules_for(actions) == ["move", "compose"]


def test_turn_rerenders_without_replacing_the_placement():
    actions = [RedoAction(action="turn", id="barn_01", label="wrong_yaw", yaw_deg=90)]
    assert modules_for(actions) == ["turn", "compose"]


def test_stage_review_logs_without_clobbering_the_stage_field(tmp_path, monkeypatch):
    scene = starter_scene()
    scene.id = "review_log_smoke"
    path = tmp_path / "scene.json"
    save_scene(scene, path)
    monkeypatch.setattr("farmgen.loop.output_dir", lambda _scene_id: tmp_path)

    def review(stage, _loaded):
        assert stage == "ground"
        return Verdict(entries=[])

    loaded = _review_stage(scene, path, "ground", review, None)
    assert loaded.id == "review_log_smoke"
    assert (tmp_path / "verdict.json").is_file()


def test_pose_only_replace_does_not_call_trellis(tmp_path, monkeypatch):
    scene = starter_scene()
    scene.assets = [scene.assets[0]]
    scene.assets[0].instances = [Instance(index=0, xy=(10.0, 10.0), yaw=0.0)]
    path = tmp_path / "scene.json"
    save_scene(scene, path)

    def boom(*_args, **_kwargs):
        raise AssertionError("trellis")

    monkeypatch.setattr("farmgen.loop.run_assets", boom)
    monkeypatch.setattr("farmgen.loop.run_compose", lambda *_args, **_kwargs: path)
    monkeypatch.setattr("farmgen.run.restamp_heights", lambda *_args, **_kwargs: None)
    action = RedoAction(
        action="move",
        id=scene.assets[0].id,
        label="wrong_place",
        yaw_deg=20.0,
        xy=(12.0, 14.0),
        index=0,
    )
    _restart(scene, path, [action], None)
