"""Qwen's asset list is kept. One unfit asset is repaired, not replaced by the starter farm."""

from pathlib import Path

from farmgen.scene import catalog_or_starter, starter_scene


def test_invalid_catalog_keeps_the_assets_already_on_the_scene():
    scene = starter_scene()
    calls = {"n": 0}

    def ask(_brief: str):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("connection refused")
        return {"assets": [{"id": "bad", "prompt": "a barn", "count": 2, "layout": "unique"}]}

    catalog_or_starter(scene, ask)
    assert calls["n"] == 2
    assert [asset.id for asset in scene.assets] == [asset.id for asset in starter_scene().assets]


def test_valid_catalog_replaces_the_starter_list():
    scene = starter_scene()

    def ask(_brief: str):
        return {
            "assets": [
                {
                    "id": "shed_01",
                    "prompt": "a wooden shed",
                    "size_m": [4, 3, 2.5],
                    "count": 1,
                    "layout": "unique",
                }
            ]
        }

    catalog_or_starter(scene, ask)
    assert [asset.id for asset in scene.assets] == ["shed_01"]
    assert scene.assets[0].size_m == (4.0, 3.0, 2.5)


def test_unplaceable_fence_is_repaired_without_dropping_the_barn(tmp_path):
    scene = starter_scene()

    def ask(prompt: str, _images=None):
        if "Change only fence_01" in prompt:
            return {
                "id": "fence_01",
                "prompt": "a short fence",
                "size_m": [2, 0.2, 1.2],
                "count": 1,
                "layout": "unique",
            }
        return {
            "assets": [
                {
                    "id": "barn_01",
                    "prompt": "a red wooden barn",
                    "size_m": [12, 8, 7],
                    "count": 1,
                    "layout": "unique",
                },
                {
                    "id": "fence_01",
                    "prompt": "a fence",
                    "size_m": [100, 2, 1.5],
                    "count": 1,
                    "layout": "surround",
                },
            ]
        }

    catalog_or_starter(scene, ask, record_dir=tmp_path)
    assert [asset.id for asset in scene.assets] == ["barn_01", "fence_01"]
    assert scene.asset("fence_01").size_m == (2.0, 0.2, 1.2)
    saved = (tmp_path / "catalog_attempt_1.json").read_text(encoding="utf-8")
    assert "100" in saved
    assert "barn_01" in saved


def test_splat_views_are_the_objects_qwen_names():
    scene = starter_scene()
    seen: dict = {}

    def ask(prompt: str, images=None):
        seen["prompt"] = prompt
        seen["images"] = list(images or [])
        return {
            "assets": [
                {
                    "id": "oak_01",
                    "prompt": "an oak tree",
                    "size_m": [4, 4, 8],
                    "count": 1,
                    "layout": "unique",
                }
            ]
        }

    pictures = [Path("reference_top.png")]
    catalog_or_starter(scene, ask, images=pictures)
    assert [asset.id for asset in scene.assets] == ["oak_01"]
    assert seen["images"] == pictures
    assert "Name the objects" in seen["prompt"]
    assert "38" in seen["prompt"]


def test_a_ring_mesh_and_a_yard_slab_are_asked_smaller(tmp_path):
    scene = starter_scene()

    def ask(prompt: str, _images=None):
        if "Change only trees_01" in prompt:
            return {
                "id": "trees_01",
                "prompt": "one oak tree",
                "size_m": [6, 6, 8],
                "count": 4,
                "layout": "surround",
            }
        if "Change only pasture_01" in prompt:
            return {
                "id": "pasture_01",
                "prompt": "a grass patch",
                "size_m": [8, 8, 0.2],
                "count": 1,
                "layout": "unique",
            }
        return {
            "assets": [
                {
                    "id": "trees_01",
                    "prompt": "a ring of trees",
                    "size_m": [38, 38, 5],
                    "count": 1,
                    "layout": "surround",
                },
                {
                    "id": "pasture_01",
                    "prompt": "a pasture",
                    "size_m": [30, 30, 0.1],
                    "count": 1,
                    "layout": "unique",
                },
            ]
        }

    catalog_or_starter(scene, ask, record_dir=tmp_path)
    assert scene.asset("trees_01").count == 4
    assert scene.asset("trees_01").size_m[0] == 6.0
    assert scene.asset("pasture_01").size_m == (8.0, 8.0, 0.2)
