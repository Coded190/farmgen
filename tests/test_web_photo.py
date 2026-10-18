"""Web photo selection. The search and the download are fakes, so nothing is fetched."""

from farmgen.web_photo import (
    PHOTO_TRIES,
    fetch_photo,
    ordered_urls,
    search_attempts,
    search_phrases,
    search_query,
    take_accepted_photo,
    urls_from_bing,
    urls_from_commons,
    urls_from_openverse,
    urls_matching_noun,
)


def test_a_run_directory_keeps_the_compose_still(tmp_path, monkeypatch):
    from farmgen.archive import allocate_run_dir, archive_run

    monkeypatch.setattr("farmgen.archive.output_dir", lambda _scene_id: tmp_path)
    (tmp_path / "compose_top.png").write_bytes(b"png")
    images = tmp_path / "images"
    images.mkdir()
    (images / "barn_01.png").write_bytes(b"png")
    first = allocate_run_dir("farm_test")
    second = allocate_run_dir("farm_test")
    archive_run("farm_test", 0, second)
    assert first.name == "0001"
    assert second.name == "0002"
    assert (second / "compose_top.png").read_bytes() == b"png"
    assert not (second / "images").exists()


def test_a_run_records_the_live_image_path(tmp_path, monkeypatch):
    import json

    from farmgen.archive import archive_run
    from farmgen.scene import AssetSpec, GroundSpec, Scene

    monkeypatch.setattr("farmgen.archive.output_dir", lambda _scene_id: tmp_path)
    image = tmp_path / "images" / "barn_01.png"
    image.parent.mkdir()
    image.write_bytes(b"png")
    scene = Scene(
        id="farm_test",
        seed=0,
        brief="a barn",
        ground=GroundSpec(size_m=(40.0, 40.0)),
        assets=[
            AssetSpec(
                id="barn_01",
                prompt="a wooden barn",
                size_m=(8.0, 6.0, 5.0),
                count=1,
                image_path=str(image),
                photo_url="https://example.test/barn.jpg",
                mesh_path=str(tmp_path / "assets" / "barn_01.glb"),
            )
        ],
    )
    run_dir = tmp_path / "runs" / "0001"
    run_dir.mkdir(parents=True)
    archive_run("farm_test", 0, run_dir, scene)
    recorded = json.loads((run_dir / "inputs.json").read_text())
    assert recorded["assets"] == [
        {
            "id": "barn_01",
            "image_path": str(image),
            "photo_url": "https://example.test/barn.jpg",
            "mesh_path": str(tmp_path / "assets" / "barn_01.glb"),
        }
    ]
    assert not (run_dir / "images").exists()


def test_a_stage_record_keeps_the_png_and_the_verdict(tmp_path, monkeypatch):
    import json

    from farmgen.archive import record_stage
    from farmgen.scene import AssetSpec, GroundSpec, Scene

    monkeypatch.setattr("farmgen.archive.output_dir", lambda _scene_id: tmp_path)
    monkeypatch.setattr("farmgen.archive.log_line", lambda *_args, **_kwargs: "")
    picture = tmp_path / "compose_top.png"
    picture.write_bytes(b"png")
    (tmp_path / "farmgen.log").write_text(
        "FARMGEN scene=farm_test seed=0 stage=asset note=uniform factor=2.0000\n",
        encoding="utf-8",
    )
    (tmp_path / "verdict.json").write_text(
        json.dumps({"entries": [{"id": "barn_01", "label": "ok", "score": 9, "action": None}]}),
        encoding="utf-8",
    )
    scene = Scene(
        id="farm_test",
        seed=0,
        brief="a barn",
        ground=GroundSpec(size_m=(40.0, 40.0)),
        assets=[AssetSpec(id="barn_01", prompt="a wooden barn", size_m=(8.0, 6.0, 5.0), count=1)],
    )
    run_dir = tmp_path / "runs" / "0009"
    record_stage(run_dir, "scene", scene, [picture], log_from=0, include_verdict=True)
    folder = run_dir / "stages" / "scene"
    recorded = json.loads((folder / "input.json").read_text())
    assert recorded["brief"] == "a barn"
    assert recorded["assets"][0]["id"] == "barn_01"
    assert recorded["assets"][0]["size_m"] == [8.0, 6.0, 5.0]
    output = json.loads((folder / "output.json").read_text())
    assert output["verdict"] == [{"id": "barn_01", "label": "ok", "score": 9, "action": None}]
    assert output["metrics"][0]["factor"] == "2.0000"
    assert (folder / "images" / "compose_top.png").read_bytes() == b"png"
    assert json.loads((folder / "scene.json").read_text())["assets"][0]["id"] == "barn_01"


def test_the_first_returned_url_is_tried_before_the_second(tmp_path):
    urls = ["http://example.test/a.jpg", "http://example.test/b.jpg"]
    assert ordered_urls(urls, seed=3, asset_id="tree_01", exclude=(urls[0],)) == [urls[1]]

    saved: list[str] = []

    def download(url: str, dest):
        saved.append(url)
        dest.write_bytes(b"not-an-image")
        return True

    path, url = fetch_photo(
        "an oak tree",
        3,
        "tree_01",
        tmp_path / "tree.png",
        search=lambda _query: urls,
        download=download,
    )
    assert path == tmp_path / "tree.png"
    assert url == urls[0]
    again, other = fetch_photo(
        "an oak tree",
        9,
        "barn_01",
        tmp_path / "tree-2.png",
        exclude=(url,),
        search=lambda _query: urls,
        download=download,
    )
    assert again is not None
    assert other == urls[1]
    assert saved == [urls[0], urls[1]]


def test_five_candidates_are_judged_in_bing_order(tmp_path):
    urls = [f"http://example.test/{index}.jpg" for index in range(6)]
    tried: list[int] = []

    def download(url: str, dest):
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(url.encode())
        return True

    def judge(_path, attempt):
        tried.append(attempt)
        return False

    path, url = take_accepted_photo(
        "one deciduous tree",
        1,
        "tree_01",
        tmp_path / "images" / "tree_01.png",
        judge=judge,
        search=lambda _query: urls,
        download=download,
    )
    assert PHOTO_TRIES == 5
    assert path is None
    assert url == ""
    assert tried == [1, 2, 3, 4, 5]
    saved = [
        (tmp_path / "images" / "photo_gate" / f"tree_01_attempt_{attempt}.png").read_bytes()
        for attempt in range(1, 6)
    ]
    assert saved == [url.encode() for url in urls[:5]]


def test_a_long_prompt_keeps_the_object_words():
    prompt = "A small wooden barn with a metal roof, featuring a central entrance and two side doors."
    assert search_query(prompt) == "small wooden barn metal"


def test_search_keeps_the_noun_when_the_long_phrase_has_no_hits():
    silo = "A tall cylindrical metal silo with a conical top, used for storing silage."
    fence = "A continuous perimeter fence enclosing the farm yard, made of wooden posts and rails."
    tractor = "A green farm tractor with a front loader, parked near the barn."
    tree = "A ring of leafy trees and rounded bushes around the yard."
    assert search_attempts(silo) == ["metal silo"]
    assert search_attempts(fence) == ["wooden fence segment"]
    assert search_attempts(tractor) == ["farm tractor"]
    assert search_attempts(tree) == ["deciduous tree"]
    assert "tall cylindrical" not in search_attempts(silo)
    assert "continuous perimeter" not in search_attempts(fence)
    assert "bush" not in search_attempts(tree)[0]


def test_the_search_keeps_the_keyword_and_asks_for_one_object():
    phrases = search_phrases("A tall cylindrical metal silo with a conical top, used for storing silage.")
    assert phrases == ["one metal silo, entire object visible, isolated, plain background"]
    fence = search_phrases(
        "A continuous perimeter fence enclosing the farm yard, made of wooden posts and rails."
    )
    assert fence == ["one wooden fence segment, entire object visible, isolated, plain background"]
    tree = search_phrases("A ring of trees around the yard.")
    assert tree == ["one deciduous tree, entire object visible, isolated, plain background"]


def test_bing_result_html_returns_the_photograph_url():
    page = (
        '<a class="iusc" m="{&quot;murl&quot;:&quot;https://example.test/silo.jpg&quot;,'
        '&quot;t&quot;:&quot;Metal silo&quot;}"></a>'
        '<a class="iusc" m="{&quot;t&quot;:&quot;no url&quot;}"></a>'
    )
    assert urls_from_bing(page) == ["https://example.test/silo.jpg"]


def test_openverse_titles_must_name_the_object():
    results = [
        {"title": "Tall cylindrical Indian baskets", "url": "https://example.test/basket.jpg"},
        {"title": "Metal silo used for maize storage", "url": "https://example.test/silo.jpg", "tags": [{"name": "silo"}]},
        {"title": "skip"},
    ]
    assert urls_matching_noun(results, "silo") == ["https://example.test/silo.jpg"]


def test_openverse_payload_returns_image_urls():
    payload = {"results": [{"title": "Old Red Tractor", "url": "https://example.test/tractor.jpg"}, {"title": "skip"}]}
    assert urls_from_openverse(payload) == ["https://example.test/tractor.jpg"]


def test_commons_payload_returns_bitmap_urls():
    payload = {
        "query": {
            "pages": {
                "1": {"imageinfo": [{"mime": "image/jpeg", "thumburl": "https://example.test/tractor.jpg"}]},
                "2": {"imageinfo": [{"mime": "image/svg+xml", "url": "https://example.test/icon.svg"}]},
            }
        }
    }
    assert urls_from_commons(payload) == ["https://example.test/tractor.jpg"]


def test_a_search_error_is_returned_for_the_log(tmp_path):
    def search(_query):
        raise RuntimeError("duckduckgo_challenge status=202")

    path, note = fetch_photo("a tractor", 0, "tractor_01", tmp_path / "missing.png", search=search)
    assert path is None
    assert note == "error:duckduckgo_challenge status=202"


def test_a_search_miss_returns_no_file(tmp_path):
    path, url = fetch_photo(
        "an oak tree",
        0,
        "tree_01",
        tmp_path / "missing.png",
        search=lambda _query: [],
        download=lambda _url, _dest: True,
    )
    assert path is None
    assert url == ""
    assert not (tmp_path / "missing.png").exists()


def test_a_cluttered_photo_is_skipped_for_the_next_url(tmp_path):
    urls = ["http://example.test/clutter.jpg", "http://example.test/barn.jpg"]

    def download(url: str, dest):
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(url.encode())
        return True

    def judge(path, attempt):
        return attempt == 2

    dest = tmp_path / "images" / "barn_01.png"
    path, url = take_accepted_photo(
        "a wooden barn",
        1,
        "barn_01",
        dest,
        judge=judge,
        search=lambda _query: urls,
        download=download,
    )
    assert path == dest
    assert url in urls
    first = (dest.parent / "photo_gate" / "barn_01_attempt_1.png").read_bytes()
    assert dest.read_bytes() != first


def test_flux_is_used_only_after_the_photo_tries_fail(tmp_path, monkeypatch):
    from farmgen.run import run_images
    from farmgen.scene import AssetSpec, GroundSpec, Scene

    scene = Scene(
        id="farm_test",
        seed=1,
        brief="a barn",
        ground=GroundSpec(size_m=(40.0, 40.0)),
        assets=[AssetSpec(id="barn_01", prompt="a wooden barn", size_m=(8.0, 6.0, 5.0), count=1)],
    )
    monkeypatch.setattr("farmgen.run.output_dir", lambda _scene_id: tmp_path)
    flux: list[str] = []

    def fake_flux(_scene, assets):
        flux.append(assets[0].id)
        dest = tmp_path / "images" / f"{assets[0].id}.png"
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(b"flux-studio")
        assets[0].image_path = str(dest)

    monkeypatch.setattr("farmgen.run._generate_images", fake_flux)

    def download(url: str, dest):
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(b"clutter " + url.encode())
        return True

    def judge(_asset_id, _prompt, path, attempt):
        class Decision:
            accept = attempt == 0 and path.read_bytes() == b"flux-studio"
            reason = "clutter" if not accept else "one barn"
            reply = ""

        return Decision()

    run_images(
        scene,
        tmp_path / "scene.json",
        None,
        search=lambda _query: [f"http://example.test/{index}.jpg" for index in range(4)],
        download=download,
        photo_judge=judge,
    )
    assert flux == ["barn_01"]
    assert scene.asset("barn_01").image_path is not None
    assert scene.asset("barn_01").photo_url in ("", None) or scene.asset("barn_01").image_path.endswith("barn_01.png")
    saved = (tmp_path / "images" / "barn_01.png").read_bytes()
    assert saved == b"flux-studio"
    assert (tmp_path / "photo_gate" / "barn_01_attempt_1.json").is_file()


def test_a_flat_card_is_kept_for_review(tmp_path, monkeypatch):
    from pathlib import Path
    from types import SimpleNamespace

    import numpy as np
    import trimesh

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
            )
        ],
    )
    python = tmp_path / "trellis" / ".venv" / "bin" / "python"
    python.parent.mkdir(parents=True)
    python.write_bytes(b"")
    calls = {"trellis": 0, "flux": 0}

    def _sheet(path):
        grid = np.linspace(-0.5, 0.5, 8)
        verts = np.array([[x, y, 0.0] for x in grid for y in grid])
        faces = []
        n = 8
        for i in range(n - 1):
            for j in range(n - 1):
                a = i * n + j
                faces.append([a, a + 1, a + n])
                faces.append([a + 1, a + n + 1, a + n])
        trimesh.Trimesh(vertices=verts, faces=np.array(faces), process=False).export(path)

    def fake_run(cmd, cwd=None, env=None):
        calls["trellis"] += 1
        dest = Path(cmd[cmd.index("--output") + 1])
        dest.parent.mkdir(parents=True, exist_ok=True)
        if calls["trellis"] == 1:
            _sheet(dest)
        else:
            trimesh.creation.box(extents=(1, 1, 1)).export(dest)
        return SimpleNamespace(returncode=0)

    def fake_flux(_scene, assets):
        calls["flux"] += 1
        studio = tmp_path / "images" / "tree_01.png"
        studio.write_bytes(b"flux-studio")
        assets[0].image_path = str(studio)

    monkeypatch.setattr("farmgen.run.trellis_dir", lambda: tmp_path / "trellis")
    monkeypatch.setattr("farmgen.run.repo_root", lambda: tmp_path)
    monkeypatch.setattr("farmgen.run._wait_gpu", lambda *_args, **_kwargs: None)
    monkeypatch.setattr("farmgen.run._wait_trellis_devices", lambda *_args, **_kwargs: [0])
    monkeypatch.setattr("farmgen.workers.vlm_critic.stop_server", lambda: None)
    monkeypatch.setattr("farmgen.run.gpu_memory_mib", lambda: 0)
    monkeypatch.setattr("farmgen.run.cache_path", lambda key: tmp_path / "cache" / f"{key}.glb")
    monkeypatch.setattr("farmgen.run.subprocess.run", fake_run)
    monkeypatch.setattr("farmgen.run._generate_images", fake_flux)
    _generate_meshes(scene, scene.assets)
    assert calls == {"trellis": 1, "flux": 0}
    assert scene.asset("tree_01").photo_url == "http://example.test/tree.jpg"
    kept = list((tmp_path / "cache").glob("*.glb"))
    assert len(kept) == 1
    assert not list((tmp_path / "cache").glob("*.raw.glb"))


def test_an_empty_foreground_retries_with_flux(tmp_path, monkeypatch):
    from pathlib import Path
    from types import SimpleNamespace

    from farmgen.run import _generate_meshes
    from farmgen.scene import AssetSpec, GroundSpec, Scene

    image = tmp_path / "images" / "barn_01.png"
    image.parent.mkdir(parents=True)
    image.write_bytes(b"web-photo")
    scene = Scene(
        id="farm_test",
        seed=1,
        brief="a barn",
        ground=GroundSpec(size_m=(40.0, 40.0)),
        assets=[
            AssetSpec(
                id="barn_01",
                prompt="a wooden barn",
                size_m=(8.0, 6.0, 5.0),
                count=1,
                image_path=str(image),
                photo_url="http://example.test/barn.jpg",
            )
        ],
    )
    python = tmp_path / "trellis" / ".venv" / "bin" / "python"
    python.parent.mkdir(parents=True)
    python.write_bytes(b"")
    calls = {"trellis": 0, "flux": 0}

    def fake_run(cmd, cwd=None, env=None):
        calls["trellis"] += 1
        if calls["trellis"] == 1:
            return SimpleNamespace(returncode=1)
        Path(cmd[cmd.index("--output") + 1]).write_bytes(b"glb")
        return SimpleNamespace(returncode=0)

    def fake_flux(_scene, assets):
        calls["flux"] += 1
        studio = tmp_path / "images" / "barn_01.png"
        studio.write_bytes(b"flux-studio")
        assets[0].image_path = str(studio)

    monkeypatch.setattr("farmgen.run.trellis_dir", lambda: tmp_path / "trellis")
    monkeypatch.setattr("farmgen.run.repo_root", lambda: tmp_path)
    monkeypatch.setattr("farmgen.run._wait_gpu", lambda *_args, **_kwargs: None)
    monkeypatch.setattr("farmgen.run._wait_trellis_devices", lambda *_args, **_kwargs: [0])
    monkeypatch.setattr("farmgen.workers.vlm_critic.stop_server", lambda: None)
    monkeypatch.setattr("farmgen.run.gpu_memory_mib", lambda: 0)
    monkeypatch.setattr("farmgen.run.cache_path", lambda key: tmp_path / "cache" / f"{key}.glb")
    monkeypatch.setattr("farmgen.run.subprocess.run", fake_run)
    monkeypatch.setattr("farmgen.run._generate_images", fake_flux)
    _generate_meshes(scene, scene.assets)
    assert calls == {"trellis": 2, "flux": 1}
    assert scene.asset("barn_01").photo_url is None
    assert (tmp_path / "cache").is_dir()


def test_a_field_photo_becomes_one_crop_and_many_copies(tmp_path, monkeypatch):
    from PIL import Image

    from farmgen.placer import Footprint, footprint_issue
    from farmgen.poses import place_fields
    from farmgen.run import run_images
    from farmgen.scene import AssetSpec, GroundSpec, Instance, Scene, normalize_catalog

    scene = Scene(
        id="farm_test",
        seed=1,
        brief="a pasture",
        ground=GroundSpec(size_m=(40.0, 40.0)),
        assets=[
            AssetSpec(
                id="barn_01",
                prompt="a red barn",
                size_m=(12.0, 8.0, 7.0),
                count=1,
                instances=[Instance(index=0, xy=(20.0, 20.0), yaw=0.0)],
            ),
            AssetSpec(
                id="pasture_01",
                prompt="a field of green pasture crops",
                size_m=(20.0, 20.0, 0.1),
                count=1,
            ),
        ],
    )
    normalize_catalog(scene)
    pasture = scene.asset("pasture_01")
    assert pasture.field_m == (20.0, 20.0)
    assert pasture.count > 1
    assert pasture.size_m[0] == 1.0
    monkeypatch.setattr("farmgen.run.output_dir", lambda _scene_id: tmp_path)

    def download(_url: str, dest):
        dest.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (80, 80), (20, 140, 40)).save(dest)
        return True

    def marker(_asset_id, _prompt, _path):
        return (0.25, 0.25, 0.5, 0.5)

    def judge(*_args):
        return True

    run_images(
        scene,
        tmp_path / "scene.json",
        ["pasture_01"],
        search=lambda _query: ["http://example.test/field.jpg"],
        download=download,
        photo_judge=judge,
        plant_marker=marker,
    )
    saved = Image.open(tmp_path / "images" / "pasture_01.png")
    assert saved.size[0] < 80
    assert scene.asset("pasture_01").image_path is not None
    assert (tmp_path / "photo_gate" / "pasture_01_field.png").is_file()
    place_fields(scene)
    plants = scene.asset("pasture_01")
    assert plants.count > 1
    barn = Footprint("barn_01", 0, (20.0, 20.0), 0.0, (12.0, 8.0))
    for inst in plants.instances:
        issue = footprint_issue(
            inst.xy,
            inst.yaw,
            plants.size_m[:2],
            "rows",
            scene.ground.plot_m,
            scene.ground.terrain_m,
            [barn],
        )
        assert issue is None, issue

