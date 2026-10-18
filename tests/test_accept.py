"""Accepted photos and meshes stay out of the generate set."""

from farmgen.cache import file_sha256
from farmgen.critic import RedoAction
from farmgen.scene import save_scene, starter_scene


def _photo(tmp_path, name: str, payload: bytes):
    path = tmp_path / name
    path.write_bytes(payload)
    return path


def test_prepare_keeps_a_passed_mesh_and_skips_a_rejected_photo(tmp_path):
    from farmgen.accept import prepare_generation

    scene = starter_scene()
    barn, silo, tractor = scene.assets[0], scene.assets[1], scene.assets[2]
    scene.assets = [barn, silo, tractor]
    barn_photo = _photo(tmp_path, "barn.png", b"barn-photo")
    barn_mesh = _photo(tmp_path, "barn.glb", b"glb")
    barn.image_path = str(barn_photo)
    barn.mesh_path = str(barn_mesh)
    barn.accepted = ["image", "mesh"]
    silo_photo = _photo(tmp_path, "silo.png", b"silo-photo")
    silo.image_path = str(silo_photo)
    silo.accepted = ["image"]
    silo.mesh_image_sha = file_sha256(silo_photo)
    actions = [
        RedoAction(action="redo_asset", id="barn_01", label="missing"),
        RedoAction(action="redo_asset", id="silo_01", label="missing"),
        RedoAction(action="redo_asset", id="tractor_01", label="missing"),
    ]
    keep, generate, skip = prepare_generation(scene, actions)
    assert keep == ["barn_01"]
    assert generate == ["tractor_01"]
    assert skip == ["silo_01"]
    assert "image" in silo.accepted
    assert silo.image_path == str(silo_photo)


def test_redo_image_clears_the_accepted_photo(tmp_path):
    from farmgen.accept import prepare_generation

    scene = starter_scene()
    barn = scene.assets[0]
    scene.assets = [barn]
    photo = _photo(tmp_path, "barn.png", b"barn-photo")
    barn.image_path = str(photo)
    barn.accepted = ["image", "mesh"]
    barn.mesh_image_sha = "abc"
    keep, generate, skip = prepare_generation(
        scene, [RedoAction(action="redo_image", id="barn_01", label="missing")]
    )
    assert keep == []
    assert generate == ["barn_01"]
    assert skip == []
    assert barn.accepted == []
    assert barn.mesh_image_sha == ""


def test_force_does_not_fetch_an_accepted_photo(tmp_path, monkeypatch):
    from farmgen.run import run_images

    scene = starter_scene()
    scene.assets = [scene.assets[0]]
    photo = tmp_path / "images" / "barn_01.png"
    photo.parent.mkdir()
    photo.write_bytes(b"kept-photo")
    scene.assets[0].image_path = str(photo)
    scene.assets[0].accepted = ["image"]
    monkeypatch.setattr("farmgen.run.output_dir", lambda _scene_id: tmp_path)

    def fetch(*_args, **_kwargs):
        raise AssertionError("photo fetch")

    monkeypatch.setattr("farmgen.web_photo.take_accepted_photo", fetch)
    run_images(scene, tmp_path / "scene.json", ["barn_01"], force=True, photo_judge=lambda *_args: True)
    assert photo.read_bytes() == b"kept-photo"
    assert scene.assets[0].image_path == str(photo)


def test_an_accepted_photo_is_not_replaced_when_the_mesh_fails(tmp_path, monkeypatch):
    from types import SimpleNamespace

    from farmgen.run import _generate_meshes
    from farmgen.scene import AssetSpec, GroundSpec, Scene

    image = tmp_path / "images" / "tree_01.png"
    image.parent.mkdir(parents=True)
    image.write_bytes(b"web-photo")
    scene = Scene(
        id="farm_test",
        seed=1,
        brief="a tree",
        ground=GroundSpec(size_m=(40.0, 40.0)),
        assets=[
            AssetSpec(
                id="tree_01",
                prompt="a deciduous tree",
                size_m=(4.0, 4.0, 8.0),
                count=1,
                image_path=str(image),
                photo_url="http://example.test/tree.jpg",
                accepted=["image"],
            )
        ],
    )
    python = tmp_path / "trellis" / ".venv" / "bin" / "python"
    python.parent.mkdir(parents=True)
    python.write_bytes(b"")
    calls = {"trellis": 0, "flux": 0}

    def fake_run(cmd, cwd=None, env=None):
        calls["trellis"] += 1
        return SimpleNamespace(returncode=1)

    def fake_flux(_scene, assets):
        calls["flux"] += 1

    monkeypatch.setattr("farmgen.run.trellis_dir", lambda: tmp_path / "trellis")
    monkeypatch.setattr("farmgen.run.repo_root", lambda: tmp_path)
    monkeypatch.setattr("farmgen.run._wait_gpu", lambda *_args, **_kwargs: None)
    monkeypatch.setattr("farmgen.run._wait_trellis_devices", lambda *_args, **_kwargs: [0])
    monkeypatch.setattr("farmgen.workers.vlm_critic.stop_server", lambda: None)
    monkeypatch.setattr("farmgen.run.gpu_memory_mib", lambda: 0)
    monkeypatch.setattr("farmgen.run.cache_path", lambda key: tmp_path / "cache" / f"{key}.glb")
    monkeypatch.setattr("farmgen.run.output_dir", lambda _scene_id: tmp_path)
    monkeypatch.setattr("farmgen.run.subprocess.run", fake_run)
    monkeypatch.setattr("farmgen.run._generate_images", fake_flux)
    _generate_meshes(scene, scene.assets)
    asset = scene.asset("tree_01")
    assert calls == {"trellis": 1, "flux": 0}
    assert asset.photo_url == "http://example.test/tree.jpg"
    assert image.read_bytes() == b"web-photo"
    assert asset.mesh_image_sha == file_sha256(image)
    assert "image" in asset.accepted
    assert "mesh" not in asset.accepted


def test_restart_generates_only_assets_that_have_not_passed(tmp_path, monkeypatch):
    from farmgen.loop import _restart

    scene = starter_scene()
    barn, silo, tractor = scene.assets[0], scene.assets[1], scene.assets[2]
    scene.assets = [barn, silo, tractor]
    barn_photo = _photo(tmp_path, "barn.png", b"barn-photo")
    barn_mesh = _photo(tmp_path, "barn.glb", b"glb")
    barn.image_path = str(barn_photo)
    barn.mesh_path = str(barn_mesh)
    barn.accepted = ["image", "mesh"]
    silo_photo = _photo(tmp_path, "silo.png", b"silo-photo")
    silo.image_path = str(silo_photo)
    silo.accepted = ["image"]
    silo.mesh_image_sha = file_sha256(silo_photo)
    path = tmp_path / "scene.json"
    save_scene(scene, path)
    seen = {}

    def assets(_scene, _path, ids, force=False, **_kwargs):
        seen["ids"] = list(ids)
        seen["force"] = force

    monkeypatch.setattr("farmgen.loop.run_assets", assets)
    monkeypatch.setattr("farmgen.loop.run_compose", lambda *_args, **_kwargs: path)
    monkeypatch.setattr("farmgen.loop.run_place", lambda *_args, **_kwargs: None)
    _restart(
        scene,
        path,
        [
            RedoAction(action="redo_asset", id="barn_01", label="missing"),
            RedoAction(action="redo_asset", id="silo_01", label="missing"),
            RedoAction(action="redo_asset", id="tractor_01", label="missing"),
        ],
        None,
    )
    assert seen["ids"] == ["tractor_01"]
    assert seen["force"] is True
    assert barn.mesh_path == str(barn_mesh)
    assert silo.image_path == str(silo_photo)


def test_an_existing_photo_is_kept_when_the_mesh_is_rebuilt(tmp_path, monkeypatch):
    from farmgen.run import keep_existing_photo

    scene = starter_scene()
    scene.id = "keep_photo"
    barn = scene.assets[0]
    barn.image_path = None
    photo = tmp_path / "images" / "barn_01.png"
    photo.parent.mkdir()
    photo.write_bytes(b"png")
    monkeypatch.setattr("farmgen.run.output_dir", lambda _scene_id: tmp_path)
    assert keep_existing_photo(scene, barn) is True
    assert barn.image_path == str(photo)


def test_infinigen_pythonpath_points_at_this_checkout(tmp_path):
    from farmgen.run import _infinigen_env

    root = tmp_path / "infinigen"
    env = _infinigen_env(root)
    assert env["PYTHONPATH"].split(":")[0] == str(root / "src")
    assert "CUDA_HOME" not in env
