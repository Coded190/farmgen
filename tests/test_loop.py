"""Redo actions map back onto the module that should run again."""

from farmgen.critic import RedoAction, Verdict
from farmgen.loop import _restart, _rerun_forward, _review_stage, modules_for, run_pipeline
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


def test_image_redo_keeps_the_photo_judge(tmp_path, monkeypatch):
    scene = starter_scene()
    scene.assets = [scene.assets[0]]
    path = tmp_path / "scene.json"
    save_scene(scene, path)
    seen = {}

    def images(_scene, _path, ids, force=False, photo_judge=None, plant_marker=None):
        seen["ids"] = ids
        seen["photo"] = photo_judge
        seen["plant"] = plant_marker

    monkeypatch.setattr("farmgen.run.run_images", images)
    action = RedoAction(action="redo_image", id=scene.assets[0].id, label="missing")
    _rerun_forward(scene, path, "image", [action], None, photo_judge="photo", plant_marker="plant")
    assert seen["ids"] == [scene.assets[0].id]
    assert seen["photo"] == "photo"
    assert seen["plant"] == "plant"


def test_restart_passes_the_photo_and_mesh_judges(tmp_path, monkeypatch):
    scene = starter_scene()
    scene.assets = [scene.assets[0]]
    path = tmp_path / "scene.json"
    save_scene(scene, path)
    seen = {}

    def assets(*_args, **kwargs):
        seen["photo"] = kwargs.get("photo_judge")
        seen["mesh"] = kwargs.get("mesh_judge")

    monkeypatch.setattr("farmgen.loop.run_assets", assets)
    monkeypatch.setattr("farmgen.loop.run_compose", lambda *_args, **_kwargs: path)
    action = RedoAction(action="redo_image", id=scene.assets[0].id, label="missing")
    _restart(scene, path, [action], None, photo_judge="photo", mesh_judge="mesh")
    assert seen["photo"] == "photo"
    assert seen["mesh"] == "mesh"


def test_each_stage_is_reviewed_before_the_next(tmp_path, monkeypatch):
    scene = starter_scene()
    scene.assets = [scene.assets[0]]
    path = tmp_path / "scene.json"
    save_scene(scene, path)
    monkeypatch.setattr("farmgen.loop.output_dir", lambda _scene_id: tmp_path)
    calls = []

    def mark(name):
        def _inner(*_args, **_kwargs):
            calls.append(name)

        return _inner

    monkeypatch.setattr("farmgen.loop.run_ground", mark("ground"))
    monkeypatch.setattr("farmgen.run.run_images", mark("images"))
    monkeypatch.setattr("farmgen.run.run_meshes", mark("meshes"))
    monkeypatch.setattr("farmgen.run.render_mesh_thumbs", mark("mesh_thumbs"))
    monkeypatch.setattr("farmgen.run.run_scale", mark("scale"))
    monkeypatch.setattr("farmgen.loop.run_place", mark("place"))
    monkeypatch.setattr("farmgen.run.render_place_preview", mark("place_preview"))
    monkeypatch.setattr("farmgen.loop.run_compose", mark("compose"))

    def review(stage, _loaded):
        calls.append(f"review:{stage}")
        return Verdict(entries=[])

    run_pipeline(scene, path, review=review)
    assert calls.index("review:ground") < calls.index("images")
    assert calls.index("review:image") > calls.index("images")
    assert calls.index("review:image") < calls.index("meshes")
    assert calls.index("review:mesh") > calls.index("mesh_thumbs")
    assert calls.index("review:mesh") < calls.index("scale")
    assert calls.index("review:scale") > calls.index("scale")
    assert calls.index("review:place") > calls.index("place")
    assert calls.index("review:place") > calls.index("place_preview")
    assert calls.index("review:place") < calls.index("compose")
    assert calls.index("review:scene") > calls.index("compose")


def test_until_mesh_stops_before_scale(tmp_path, monkeypatch):
    scene = starter_scene()
    scene.id = "until_mesh"
    path = tmp_path / "scene.json"
    save_scene(scene, path)
    monkeypatch.setattr("farmgen.loop.output_dir", lambda _scene_id: tmp_path)
    calls = []

    def mark(name):
        def _inner(*_args, **_kwargs):
            calls.append(name)

        return _inner

    monkeypatch.setattr("farmgen.loop.run_ground", mark("ground"))
    monkeypatch.setattr("farmgen.run.run_images", mark("images"))
    monkeypatch.setattr("farmgen.run.run_meshes", mark("meshes"))
    monkeypatch.setattr("farmgen.run.render_mesh_thumbs", mark("mesh_thumbs"))
    monkeypatch.setattr("farmgen.run.run_scale", mark("scale"))
    monkeypatch.setattr("farmgen.loop.run_place", mark("place"))

    def review(stage, _loaded):
        calls.append(f"review:{stage}")
        return Verdict(entries=[])

    run_pipeline(scene, path, review=review, until="mesh")
    assert "review:mesh" in calls
    assert "scale" not in calls
    assert "place" not in calls


def test_a_grid_is_not_sent_to_qwen():
    from farmgen.loop import _posed_ids

    scene = starter_scene()
    scene.assets[0].layout = "grid"
    assert scene.assets[0].id not in _posed_ids(scene, {asset.id for asset in scene.assets})


def test_a_geometric_copy_has_no_u_coordinate():
    from farmgen.cli import placed_copies

    scene = starter_scene()
    scene.assets[0].instances = [Instance(index=0, xy=(1.0, 2.0), yaw=0.0)]
    assert placed_copies(scene, []) == ""


def test_until_pose_stops_before_place(tmp_path, monkeypatch):
    scene = starter_scene()
    scene.id = "until_pose"
    for asset in scene.assets:
        asset.mesh_path = str(tmp_path / f"{asset.id}.glb")
    path = tmp_path / "scene.json"
    save_scene(scene, path)
    monkeypatch.setattr("farmgen.loop.output_dir", lambda _scene_id: tmp_path)
    calls = []

    def mark(name):
        def _inner(*_args, **_kwargs):
            calls.append(name)

        return _inner

    monkeypatch.setattr("farmgen.loop.run_ground", mark("ground"))
    monkeypatch.setattr("farmgen.run.run_images", mark("images"))
    monkeypatch.setattr("farmgen.run.run_meshes", mark("meshes"))
    monkeypatch.setattr("farmgen.run.render_mesh_thumbs", mark("mesh_thumbs"))
    monkeypatch.setattr("farmgen.run.run_scale", mark("scale"))
    monkeypatch.setattr("farmgen.poses.fill_poses", mark("poses"))
    monkeypatch.setattr("farmgen.poses.place_fences", mark("fences"))
    monkeypatch.setattr("farmgen.poses.seat_visible", mark("seat"))
    monkeypatch.setattr("farmgen.poses.place_fields", mark("fields"))
    monkeypatch.setattr("farmgen.loop.run_place", mark("place"))

    def review(stage, _loaded):
        calls.append(f"review:{stage}")
        return Verdict(entries=[])

    run_pipeline(scene, path, review=review, propose=lambda *_args, **_kwargs: [], until="pose")
    assert "review:scale" in calls
    assert "poses" in calls
    assert "place" not in calls


def test_place_review_uses_the_yard_top_and_the_reference(tmp_path, monkeypatch):
    from farmgen.cli import _stage_review

    scene = starter_scene()
    scene.assets = [scene.assets[0]]
    out = tmp_path / "out"
    out.mkdir()
    (out / "place_top.png").write_bytes(b"png")
    (out / "compose_top.png").write_bytes(b"nope")
    ref = tmp_path / "ref"
    ref.mkdir()
    (ref / "reference_top.png").write_bytes(b"ref")
    monkeypatch.setattr("farmgen.paths.output_dir", lambda _scene_id: out)
    seen = {}

    def verdict(kind, ids, images, note=""):
        seen["kind"] = kind
        seen["ids"] = ids
        seen["images"] = images
        return Verdict(entries=[])

    monkeypatch.setattr("farmgen.workers.vlm_critic.request_verdict", verdict)
    _stage_review("place", scene, ref)
    assert seen["kind"] == "place"
    assert seen["ids"] == [scene.assets[0].id]
    assert [path.name for path in seen["images"]] == ["place_top.png", "reference_top.png"]


def test_a_missing_photo_clears_the_mesh(tmp_path, monkeypatch):
    from farmgen.run import run_meshes

    scene = starter_scene()
    scene.assets = [scene.assets[0]]
    scene.assets[0].image_path = None
    scene.assets[0].mesh_path = str(tmp_path / "old.glb")
    scene.assets[0].instances = [Instance(index=0, xy=(1.0, 1.0), yaw=0.0)]
    path = tmp_path / "scene.json"
    save_scene(scene, path)
    monkeypatch.setattr("farmgen.run.output_dir", lambda _scene_id: tmp_path)
    run_meshes(scene, path, None)
    assert scene.assets[0].mesh_path is None
    assert scene.assets[0].instances == []
