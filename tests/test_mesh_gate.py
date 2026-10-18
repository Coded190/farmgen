"""A mesh that fails both tries is not posed. A missing copy is asked for again."""

from farmgen.mesh_gate import MeshDecision, apply_mesh_gate
from farmgen.poses import fill_poses
from farmgen.scene import AssetSpec, GroundSpec, Instance, Scene


def _scene() -> Scene:
    return Scene(
        id="farm_test",
        seed=1,
        brief="trees",
        ground=GroundSpec(size_m=(40.0, 40.0)),
        assets=[
            AssetSpec(
                id="tree_01",
                prompt="an oak tree",
                size_m=(4.0, 4.0, 8.0),
                count=2,
                layout="rows",
                instances=[Instance(index=0, xy=(8.0, 8.0), yaw=0.0)],
            )
        ],
    )


def test_a_sheet_is_a_flat_card_and_a_panel_is_not():
    import numpy as np
    import trimesh

    from farmgen.mesh_gate import FLAT_CARD_LIMIT, flat_card_reason, plane_flatness

    grid = np.linspace(-0.5, 0.5, 40)
    sheet = np.array([[x, y, 0.0] for x in grid for y in grid])
    corners = np.array(
        [
            [-0.5, -0.5, -0.5],
            [0.5, -0.5, 0.5],
            [-0.5, 0.5, -0.5],
            [0.5, 0.5, 0.5],
        ]
    )
    card = np.vstack([sheet, corners])
    assert plane_flatness(card) < FLAT_CARD_LIMIT
    panel = np.vstack([sheet, sheet + np.array([0.0, 0.0, 0.02])])
    assert plane_flatness(panel) > FLAT_CARD_LIMIT
    box = trimesh.creation.box(extents=(1.0, 1.0, 1.0))
    assert plane_flatness(box.vertices) > 0.05


def test_a_flat_card_is_dropped_instead_of_kept(tmp_path):
    import trimesh

    from farmgen.mesh_gate import flat_card_reason

    scene = _scene()
    scene.asset("tree_01").mesh_path = str(tmp_path / "tree_01.glb")
    calls = {"n": 0}

    def judge(_asset_id: str, _attempt: int):
        from farmgen.mesh_gate import MeshDecision

        return MeshDecision(False, "flat card flatness=0.0010")

    def regenerate(_asset) -> None:
        calls["n"] += 1

    accepted = apply_mesh_gate(scene, judge, regenerate)
    assert accepted == set()
    assert scene.asset("tree_01").mesh_path is None
    assert calls["n"] == 1
    sheet = trimesh.Trimesh(
        vertices=[[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]],
        faces=[[0, 1, 2], [0, 2, 3]],
        process=False,
    )
    path = tmp_path / "sheet.glb"
    sheet.export(path)
    assert flat_card_reason(path).startswith("flat card")


def test_two_rejects_drop_the_mesh():
    scene = _scene()
    scene.asset("tree_01").mesh_path = "tree_01.glb"
    calls = {"n": 0}
    reasons = []

    def judge(_asset_id: str, attempt: int) -> MeshDecision:
        reasons.append(attempt)
        return MeshDecision(False, f"broken shell attempt {attempt}")

    def regenerate(_asset) -> None:
        calls["n"] += 1

    accepted = apply_mesh_gate(scene, judge, regenerate)
    assert accepted == set()
    assert scene.asset("tree_01").mesh_path is None
    assert scene.asset("tree_01").instances == []
    assert calls["n"] == 1
    assert reasons == [1, 2]


def test_a_card_and_three_planes_are_sheets(tmp_path):
    import numpy as np
    import trimesh

    from farmgen.mesh_gate import SHEET_FRACTION_LIMIT, blocking_sheet, sheet_fraction, thin_yaw_order

    card = trimesh.Trimesh(
        vertices=[[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]],
        faces=[[0, 1, 2], [0, 2, 3]],
        process=False,
    )
    assert sheet_fraction(card.vertices, card.faces) >= SHEET_FRACTION_LIMIT
    planes = []
    for normal in ((0, 0, 1), (0, 1, 0), (1, 0, 0)):
        shift = np.array(normal, dtype=np.float64) * 0.0
        quad = card.vertices + shift
        if normal[0]:
            quad = np.array([[0, 0, 0], [0, 1, 0], [0, 1, 1], [0, 0, 1]], dtype=np.float64)
        elif normal[1]:
            quad = np.array([[0, 0, 0], [1, 0, 0], [1, 0, 1], [0, 0, 1]], dtype=np.float64)
        planes.append(quad)
    vertices = np.vstack(planes)
    faces = []
    for block in range(3):
        base = block * 4
        faces.append([base, base + 1, base + 2])
        faces.append([base, base + 2, base + 3])
    crossed = trimesh.Trimesh(vertices=vertices, faces=np.array(faces), process=False)
    assert sheet_fraction(crossed.vertices, crossed.faces) >= SHEET_FRACTION_LIMIT
    path = tmp_path / "crossed.glb"
    crossed.export(path)
    assert blocking_sheet(path).startswith("sheet")
    assert thin_yaw_order(0.2, 1.0)[0] == 90.0
    assert thin_yaw_order(1.0, 0.2)[0] == 0.0


def test_qwen_accept_does_not_keep_a_sheet(tmp_path):
    import trimesh

    scene = _scene()
    path = tmp_path / "tree_01.glb"
    trimesh.Trimesh(
        vertices=[[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]],
        faces=[[0, 1, 2], [0, 2, 3]],
        process=False,
    ).export(path)
    asset = scene.asset("tree_01")
    asset.mesh_path = str(path)
    asset.image_path = str(tmp_path / "tree_01.png")
    calls = {"n": 0}

    def judge(_asset_id: str, attempt: int) -> MeshDecision:
        if attempt == 1:
            return MeshDecision(True, "a leafy tree")
        return MeshDecision(False, "no mesh")

    def regenerate(_asset) -> None:
        calls["n"] += 1

    accepted = apply_mesh_gate(scene, judge, regenerate)
    assert accepted == set()
    assert calls["n"] == 1
    assert asset.image_path is None


def test_each_accepted_copy_gets_a_pose():
    scene = _scene()
    scene.asset("tree_01").instances = []
    replies = [
        [{"id": "tree_01", "index": 0, "u": 0.5, "v": 0.5, "yaw_deg": 0}],
        [{"id": "tree_01", "index": 1, "u": 0.55, "v": 0.50, "yaw_deg": 15}],
    ]
    asked: list[list[int]] = []

    def propose(missing):
        asked.append([index for _asset_id, index in missing])
        return replies.pop(0)

    kept = fill_poses(scene, propose, {"tree_01"})
    assert kept == {"tree_01"}
    assert asked == [[0, 1], [1]]
    assert [inst.index for inst in scene.asset("tree_01").instances] == [0, 1]


def test_a_square_perimeter_mesh_is_retired(tmp_path):
    import trimesh

    from farmgen.mesh_gate import blocking_post

    scene = _scene()
    asset = scene.asset("tree_01")
    asset.layout = "perimeter"
    asset.prompt = "a wooden fence segment"
    asset.count = 4
    path = tmp_path / "fence.glb"
    trimesh.creation.box(extents=(1.0, 1.0, 1.2)).export(path)
    asset.mesh_path = str(path)
    asset.image_path = str(tmp_path / "fence.png")
    asset.photo_url = "https://example.test/fence.jpg"
    calls = {"n": 0}

    def judge(_asset_id: str, attempt: int) -> MeshDecision:
        if attempt == 1:
            return MeshDecision(True, "a wooden fence")
        return MeshDecision(False, "no mesh")

    def regenerate(_asset) -> None:
        calls["n"] += 1

    accepted = apply_mesh_gate(scene, judge, regenerate)
    assert blocking_post(path) is not None
    assert accepted == set()
    assert calls["n"] == 1
    assert asset.image_path is None
    assert asset.photo_url in asset.seen_photo_urls
    rail = tmp_path / "rail.glb"
    trimesh.creation.box(extents=(2.4, 0.3, 1.2)).export(rail)
    assert blocking_post(rail) is None


def test_a_missing_glb_does_not_read_an_old_thumb(tmp_path):
    import os

    from farmgen.mesh_gate import fresh_views

    thumb = tmp_path / "barn_01_yaw0.png"
    thumb.write_bytes(b"old")
    missing = tmp_path / "missing.glb"
    assert fresh_views(missing, [thumb]) == []
    mesh = tmp_path / "barn.glb"
    mesh.write_bytes(b"mesh")
    os.utime(thumb, (1, 1))
    os.utime(mesh, (100, 100))
    assert fresh_views(mesh, [thumb]) == []
    os.utime(thumb, (200, 200))
    assert fresh_views(mesh, [thumb]) == [thumb]


def test_the_mesh_gate_reads_the_saved_scene(tmp_path, monkeypatch):
    import os

    from farmgen.cli import _mesh_judge
    from farmgen.mesh_gate import MeshDecision
    from farmgen.scene import save_scene
    from farmgen.views import ASSET_YAWS, asset_view_filename

    scene = _scene()
    scene_file = tmp_path / "scene.json"
    save_scene(scene, scene_file)
    out = tmp_path / "out"
    monkeypatch.setattr("farmgen.paths.output_dir", lambda _scene_id: out)
    monkeypatch.setattr(
        "farmgen.workers.vlm_critic.request_mesh_gate",
        lambda *_args, **_kwargs: MeshDecision(True, "one oak tree"),
    )
    judge = _mesh_judge(scene_file)
    glb = out / "assets" / "tree_01.glb"
    glb.parent.mkdir(parents=True)
    glb.write_bytes(b"mesh")
    thumbs = out / "thumbs"
    thumbs.mkdir()
    for yaw in ASSET_YAWS:
        (thumbs / asset_view_filename("tree_01", yaw)).write_bytes(b"view")
    os.utime(glb, (100, 100))
    for yaw in ASSET_YAWS:
        os.utime(thumbs / asset_view_filename("tree_01", yaw), (200, 200))
    scene.asset("tree_01").mesh_path = str(glb)
    save_scene(scene, scene_file)
    decision = judge("tree_01", 1)
    assert decision.accept
    assert decision.reason == "one oak tree"


def test_a_flat_card_is_shown_to_qwen(tmp_path, monkeypatch):
    import os

    import trimesh

    from farmgen.cli import _mesh_judge
    from farmgen.scene import save_scene
    from farmgen.views import ASSET_YAWS, asset_view_filename

    scene = _scene()
    scene_file = tmp_path / "scene.json"
    save_scene(scene, scene_file)
    out = tmp_path / "out"
    seen = {}

    def fake_gate(asset_id, prompt, images, post=None, note=""):
        from farmgen.mesh_gate import MeshDecision

        seen["note"] = note
        seen["images"] = list(images)
        return MeshDecision(True, "one tree, thin but the object is there")

    monkeypatch.setattr("farmgen.paths.output_dir", lambda _scene_id: out)
    monkeypatch.setattr("farmgen.workers.vlm_critic.request_mesh_gate", fake_gate)
    sheet = trimesh.Trimesh(
        vertices=[[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]],
        faces=[[0, 1, 2], [0, 2, 3]],
        process=False,
    )
    glb = out / "assets" / "tree_01.glb"
    glb.parent.mkdir(parents=True)
    sheet.export(glb)
    thumbs = out / "thumbs"
    thumbs.mkdir()
    for yaw in ASSET_YAWS:
        (thumbs / asset_view_filename("tree_01", yaw)).write_bytes(b"view")
    os.utime(glb, (100, 100))
    for yaw in ASSET_YAWS:
        os.utime(thumbs / asset_view_filename("tree_01", yaw), (200, 200))
    scene.asset("tree_01").mesh_path = str(glb)
    save_scene(scene, scene_file)
    decision = _mesh_judge(scene_file)("tree_01", 1)
    assert decision.accept
    assert "flatness=" in seen["note"]
    assert seen["images"]
