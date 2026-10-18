"""Stage prompts, required catalog words, mesh repair, and support-before-on-top poses."""

import numpy as np
import trimesh

from farmgen.critic import RedoAction, redo_metrics
from farmgen.poses import fill_poses, sits_on_another
from farmgen.repair import polygons_overlap, repair_mesh_file, xy_hull
from farmgen.scene import AssetSpec, GroundSpec, Instance, Scene, missing_required
from farmgen.workers.vlm_critic import verdict_payload


def _scene() -> Scene:
    return Scene(
        seed=0,
        brief="a barn and a bucket on the barn",
        ground=GroundSpec(size_m=(40.0, 40.0)),
        assets=[
            AssetSpec(id="barn_01", prompt="a wooden barn", size_m=(4.0, 4.0, 3.0), count=1),
            AssetSpec(id="bucket_01", prompt="a bucket on the barn", size_m=(1.0, 1.0, 1.0), count=1),
        ],
    )


def test_required_words_come_from_the_brief_only():
    scene = _scene()
    assert missing_required(scene.brief, scene.assets) == []
    assert missing_required("a barn and a silo", scene.assets) == ["silo"]
    assert missing_required("a wooden shed", scene.assets) == []


def test_each_stage_prompt_names_its_own_action():
    ground = verdict_payload("ground", ["ground"], [])["messages"][0]["content"][0]["text"]
    image = verdict_payload("image", ["barn_01"], [])["messages"][0]["content"][0]["text"]
    mesh = verdict_payload("mesh", ["barn_01"], [])["messages"][0]["content"][0]["text"]
    scale = verdict_payload("scale", ["barn_01"], [], note="Contact hole fractions: barn_01=0.000.")["messages"][0]["content"][0]["text"]
    pose = verdict_payload("pose", ["barn_01"], [])["messages"][0]["content"][0]["text"]
    place = verdict_payload("place", ["barn_01"], [], note="Footprint overlaps: none.")["messages"][0]["content"][0]["text"]
    scene = verdict_payload("scene", ["barn_01"], [], note="Contact hole fractions: none. Footprint overlaps: none.")["messages"][0]["content"][0]["text"]
    assert "redo_ground" in ground and "tilled soil" in ground
    assert "redo_image" in image
    assert "redo_asset" in mesh
    assert "wrong_scale" in scale and "reorient" in scale and "barn_01=0.000" in scale
    assert "Do not invent a new point" in pose
    assert "Footprint overlaps" in place
    assert "top view" in scene and "Contact hole" in scene


def test_critique_restarts_the_named_modules_and_logs_the_second_pass(tmp_path, monkeypatch):
    import json

    from farmgen.cli import main
    from farmgen.critic import Verdict, VerdictEntry
    from farmgen.scene import save_scene, starter_scene

    scene = starter_scene()
    scene.assets = scene.assets[:1]
    scene_path = tmp_path / "scene.json"
    save_scene(scene, scene_path)
    verdict_path = tmp_path / "verdict.json"
    verdict_path.write_text(
        json.dumps({"entries": [{"id": "barn_01", "label": "missing", "action": "redo_asset", "score": 8}]}),
        encoding="utf-8",
    )
    restarted = {"n": 0}

    def fake_restart(*_args, **_kwargs):
        restarted["n"] += 1

    def fake_review(_stage, _scene, _reference, appearance=None):
        return Verdict(entries=[VerdictEntry(id="barn_01", label="ok", action=None, score=0)])

    monkeypatch.setattr("farmgen.cli.output_dir", lambda _scene_id: tmp_path)
    monkeypatch.setattr("farmgen.loop._restart", fake_restart)
    monkeypatch.setattr("farmgen.cli._stage_review", fake_review)
    monkeypatch.setattr("farmgen.workers.vlm_critic.stop_server", lambda: None)
    assert main(["critique", "--scene", str(scene_path), "--verdict", str(verdict_path)]) == 0
    assert restarted["n"] == 1
    log = (tmp_path / "farmgen.log").read_text(encoding="utf-8")
    assert "note=second_pass" in log
    assert "redo_rate=0.5000" in log
    assert "second_pass_clear_rate=1.0000" in log


def test_a_missing_second_label_is_not_cleared():
    actions = [RedoAction(action="redo_asset", id="barn_01", label="missing")]
    rate, clear = redo_metrics(actions, n_targets=3, second_labels={})
    assert rate == 1 / 3
    assert clear == 0.0
    _rate, cleared = redo_metrics(actions, n_targets=3, second_labels={"barn_01": "ok"})
    assert cleared == 1.0


def test_ground_copies_are_authored_before_an_asset_that_sits_on_them():
    scene = _scene()
    assert sits_on_another(scene.asset("bucket_01"))
    assert not sits_on_another(scene.asset("barn_01"))
    order: list[set[str]] = []

    def propose(missing, problems=None):
        order.append({asset_id for asset_id, _index in missing})
        return [{"id": asset_id, "index": index, "u": 0.5, "v": 0.5, "yaw_deg": 0} for asset_id, index in missing]

    fill_poses(scene, propose, {"barn_01", "bucket_01"})
    assert order[0] == {"barn_01"}
    assert order[1] == {"bucket_01"}


def test_a_scale_miss_leaves_the_asset_unposed(tmp_path, monkeypatch):
    from farmgen.run import run_scale

    scene = _scene()
    barn = scene.asset("barn_01")
    barn.mesh_path = "missing.glb"
    barn.instances = [Instance(index=0, xy=(10.0, 10.0), yaw=0.0)]
    monkeypatch.setattr("farmgen.run.output_dir", lambda _scene_id: tmp_path)
    monkeypatch.setattr("farmgen.run.cache_path", lambda _key: tmp_path / "nope.glb")
    run_scale(scene, tmp_path / "scene.json", ["barn_01"])
    assert barn.mesh_path is None
    assert barn.instances == []


def test_cpu_repair_fills_a_pinhole_and_splits_a_nonmanifold_edge(tmp_path):
    from farmgen.repair import nonmanifold_edge_count

    wide = trimesh.creation.box(extents=(1.0, 1.0, 1.0))
    wide.faces = wide.faces[:-1]
    large = tmp_path / "large.glb"
    wide.export(large)
    assert repair_mesh_file(large).holes_filled == 0
    pin = trimesh.creation.box(extents=(0.008, 0.008, 0.008))
    pin.faces = pin.faces[:-1]
    small = tmp_path / "pinhole.glb"
    pin.export(small)
    opened = repair_mesh_file(small)
    assert opened.holes_filled > 0
    assert opened.nonmanifold_after == 0
    assert small.is_file()
    vertices = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, -1.0, 0.0],
        ]
    )
    faces = np.array([[0, 1, 2], [0, 1, 3], [0, 1, 4]])
    colors = np.array([[255, 0, 0, 255], [0, 255, 0, 255], [0, 0, 255, 255], [255, 255, 0, 255], [0, 255, 255, 255]])
    broken = tmp_path / "broken.glb"
    sheet = trimesh.Trimesh(vertices, faces, vertex_colors=colors, process=False)
    sheet.export(broken)
    report = repair_mesh_file(broken)
    assert report.nonmanifold_before >= 1
    assert report.nonmanifold_after == 0
    kept = trimesh.load(broken, force="mesh", process=False)
    assert len(kept.faces) >= 3
    assert nonmanifold_edge_count(kept.faces) == 0
    assert broken.is_file()
    skipped = tmp_path / "gpu_filled.glb"
    pin.export(skipped)
    untouched = repair_mesh_file(skipped, fill_holes=False)
    assert untouched.holes_filled == 0
    kept_open = trimesh.load(skipped, force="mesh", process=False)
    assert len(kept_open.faces) == len(pin.faces)


def test_cpu_hole_fill_runs_only_when_the_gpu_fill_did_not():
    from farmgen.repair import hole_fill_on_cpu
    from farmgen.workers.trellis_generate import FILL_FREE_MIN_MIB, fill_device, install_fill_guard

    assert fill_device(FILL_FREE_MIN_MIB - 1) == "cpu"
    assert fill_device(FILL_FREE_MIN_MIB) == "gpu"
    assert hole_fill_on_cpu({"fill": "gpu", "reason": "ok"}) is False
    assert hole_fill_on_cpu({"fill": "cpu", "reason": "cuda_oom"}) is True
    assert hole_fill_on_cpu(None) is True

    calls = {"gpu": 0, "emptied": 0}

    class _Cuda:
        def mem_get_info(self):
            return (100 * 1048576, 10000 * 1048576)

        def memory_allocated(self):
            return (9000 * 1048576)

        def empty_cache(self):
            calls["emptied"] += 1

        class OutOfMemoryError(Exception):
            pass

    class _Torch:
        cuda = _Cuda()
        OutOfMemoryError = _Cuda.OutOfMemoryError

    class _Mesh:
        def fill_holes(self, max_hole_perimeter=3e-2):
            calls["gpu"] += 1
            raise AssertionError("gpu fill ran below the free-memory floor")

    class _Cumesh:
        CuMesh = _Mesh

    low = install_fill_guard(_Cumesh, _Torch)
    _Mesh().fill_holes()
    assert calls["gpu"] == 0
    assert low["fill"] == "cpu"
    assert low["reason"] == "low_free"
    assert low["memory"] == "gpu"
    assert low["gpu_free_mib"] == 100
    assert low["gpu_allocated_mib"] == 9000

    class _FullCuda(_Cuda):
        def mem_get_info(self):
            return (8000 * 1048576, 10000 * 1048576)

        def memory_allocated(self):
            return (2000 * 1048576)

    class _OomTorch:
        cuda = _FullCuda()
        OutOfMemoryError = _Cuda.OutOfMemoryError

    class _OomMesh:
        def fill_holes(self, max_hole_perimeter=3e-2):
            calls["gpu"] += 1
            raise _Cuda.OutOfMemoryError("CUDA out of memory")

    class _OomCumesh:
        CuMesh = _OomMesh

    failed = install_fill_guard(_OomCumesh, _OomTorch)
    _OomMesh().fill_holes()
    assert calls["gpu"] == 1
    assert calls["emptied"] == 1
    assert failed["fill"] == "cpu"
    assert failed["reason"] == "cuda_oom"
    assert failed["memory"] == "gpu"
    assert failed["gpu_free_mib"] == 8000
    assert failed["gpu_total_mib"] == 10000
    assert failed["gpu_allocated_mib"] == 2000


def test_hulls_need_a_tenth_of_a_meter_between_them():
    square = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0]])
    left = xy_hull(square, (0.0, 0.0), 0.0)
    touching = xy_hull(square, (1.05, 0.0), 0.0)
    apart = xy_hull(square, (1.2, 0.0), 0.0)
    assert polygons_overlap(left, touching)
    assert polygons_overlap(left, apart) is False
