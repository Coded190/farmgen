"""Qwen's asset list is kept. One unfit asset is repaired, not replaced by the starter farm."""

import json
from pathlib import Path

from farmgen.mesh_gate import MeshDecision
from farmgen.scene import catalog_or_starter, starter_scene


def test_catalog_asks_for_an_object_and_its_form():
    from farmgen.scene import catalog_text

    text = catalog_text("a barn and a fence")
    assert "a wooden barn" in text
    assert "one or two words of form" in text
    assert "Do not say where it sits" in text


def test_flux_prompt_asks_for_one_isolated_object():
    from farmgen.image import load_workflow

    text = load_workflow("a wooden barn", 0)["6"]["inputs"]["text"]
    assert text.startswith("a single wooden barn,")
    assert "three-quarter view" in text
    assert "plain light gray background" in text
    assert "no other objects" in text


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
    scene.brief = "a wooden shed"

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
    scene.brief = "a barn and a fence"

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
    scene.brief = "an oak tree"
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
    scene.brief = "trees and a pasture"

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


def _shed(prompt: str, _images=None):
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


def test_catalog_record_keeps_the_prompt_reply_and_normalized_assets(tmp_path):
    scene = starter_scene()
    scene.brief = "a wooden shed"

    class Answer:
        payload = _shed("")
        reply = "raw model text"

    def ask(_prompt: str, _images=None):
        return Answer()

    catalog_or_starter(scene, ask, record_dir=tmp_path)
    record = json.loads((tmp_path / "catalog_record.json").read_text(encoding="utf-8"))
    assert scene.brief in record["prompt"]
    assert record["reply"] == "raw model text"
    assert record["payload"]["assets"][0]["id"] == "shed_01"
    assert record["assets"][0]["id"] == "shed_01"
    assert record["missing"] == []
    assert [asset.id for asset in scene.assets] == ["shed_01"]


def test_a_rejected_catalog_is_generated_again_then_dropped(tmp_path):
    scene = starter_scene()
    starter_ids = [asset.id for asset in scene.assets]
    calls = {"n": 0}

    def ask(prompt: str, images=None):
        calls["n"] += 1
        return _shed(prompt, images)

    def judge(_brief, _assets, _images):
        return MeshDecision(False, "missing the barn")

    catalog_or_starter(scene, ask, judge=judge, record_dir=tmp_path)
    assert calls["n"] == 2
    assert [asset.id for asset in scene.assets] == starter_ids
    gate = json.loads((tmp_path / "catalog_gate.json").read_text(encoding="utf-8"))
    assert gate["accept"] is False
    assert gate["reason"].startswith("missing=")
    assert "barn" in gate["reason"]
    record = json.loads((tmp_path / "catalog_record.json").read_text(encoding="utf-8"))
    assert record["assets"][0]["id"] == "shed_01"


def test_the_second_catalog_is_kept_when_the_gate_accepts_it():
    scene = starter_scene()
    scene.brief = "a wooden shed"
    seen = {"n": 0}

    def judge(_brief, _assets, _images):
        seen["n"] += 1
        return MeshDecision(seen["n"] == 2, "matches the brief" if seen["n"] == 2 else "missing the barn")

    catalog_or_starter(scene, _shed, judge=judge)
    assert seen["n"] == 2
    assert [asset.id for asset in scene.assets] == ["shed_01"]
