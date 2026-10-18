"""Catalog and verdict parsing use a fake HTTP body. No key file is opened."""

from farmgen.critic import SCORE_OK_MAX
from farmgen.scene import starter_scene
from farmgen.workers.vlm_critic import (
    catalog_gate_payload,
    catalog_payload,
    mesh_gate_payload,
    parse_catalog,
    parse_mesh_gate,
    parse_verdict,
    photo_gate_payload,
    poses_payload,
    verdict_payload,
)


def test_photo_gate_accepts_a_plain_or_outdoor_background(tmp_path):
    from PIL import Image

    image = tmp_path / "barn.png"
    Image.new("RGB", (8, 8), (120, 90, 40)).save(image)
    text = photo_gate_payload("barn_01", "a weathered wooden barn", image)["messages"][0]["content"][0]["text"]
    assert "field or sky" in text
    assert "plain background" in text
    assert "watermark by itself is not a reason to reject" in text
    assert "person" in text
    assert "ring" in text


def test_mesh_gate_keeps_the_reason_sentence():
    payload = mesh_gate_payload("silo_01", "a metal silo", [])
    assert "reason" in payload["messages"][0]["content"][0]["text"]
    body = {
        "choices": [
            {
                "message": {
                    "content": '{"id":"silo_01","accept":false,"reason":"The mesh is a broken shell, not one silo."}'
                }
            }
        ]
    }
    decision = parse_mesh_gate(body, "silo_01")
    assert decision.accept is False
    assert decision.reason == "The mesh is a broken shell, not one silo."


def test_catalog_json_is_taken_from_the_message():
    payload = catalog_payload("a small dairy farm")
    assert payload["model"]
    assert "http" not in str(payload)
    body = {
        "choices": [
            {
                "message": {
                    "content": 'Here is the list:\n```json\n{"assets":[{"id":"barn_01","prompt":"a barn","size_m":[12,8,7],"count":1,"layout":"unique"}]}\n```'
                }
            }
        ]
    }
    parsed = parse_catalog(body)
    assert parsed["assets"][0]["id"] == "barn_01"


def test_verdict_json_keeps_scores():
    images_note = verdict_payload("asset", ["tractor_01"], [])
    assert images_note["messages"][0]["content"][0]["type"] == "text"
    body = {
        "choices": [
            {
                "message": {
                    "content": '{"entries":[{"id":"tractor_01","label":"wrong_orientation","action":"reorient","score":3}]}'
                }
            }
        ]
    }
    verdict = parse_verdict(body)
    assert verdict.entries[0].score == 3
    assert verdict.entries[0].score <= SCORE_OK_MAX

    doubled = {
        "choices": [
            {
                "message": {
                    "content": '{"note":1} {"entries":[{"id":"barn_01","label":"wrong_yaw","action":"turn","score":6,"yaw_deg":90}]}'
                }
            }
        ]
    }
    again = parse_verdict(doubled)
    assert again.entries[0].id == "barn_01"
    assert again.entries[0].yaw_deg == 90

    separate = {
        "choices": [
            {
                "message": {
                    "content": '{"id":"barn_01","label":"ok","action":null,"score":2} {"id":"tree_01","label":"wrong_yaw","action":"turn","score":6,"yaw_deg":180}'
                }
            }
        ]
    }
    split = parse_verdict(separate)
    assert [entry.id for entry in split.entries] == ["barn_01", "tree_01"]
    assert split.entries[1].yaw_deg == 180


def test_mesh_and_placement_prompts_stay_on_separate_jobs(tmp_path):
    from PIL import Image

    mesh = mesh_gate_payload("silo_01", "a metal silo", [])["messages"][0]["content"][0]["text"]
    assert "thinnest view" in mesh
    assert "edge-on sheet" in mesh
    place = verdict_payload("place", ["barn_01", "trees_bushes_ring_01"], [])["messages"][0]["content"][0]["text"]
    assert "placement only" in place
    assert "action is null or move" in place
    image = tmp_path / "top.png"
    Image.new("RGB", (8, 8), (80, 140, 60)).save(image)
    poses = poses_payload(["barn_01", "tree_01"], image)["messages"][0]["content"][0]["text"]
    assert "own point" in poses
    assert "green ring" in poses


def test_catalog_gate_accepts_or_rejects_without_rewriting():
    scene = starter_scene()
    text = catalog_gate_payload(scene.brief, scene.assets[:2])["messages"][0]["content"]
    assert scene.brief in text
    assert "Do not rewrite" in text
    assert "absent after normalization" in catalog_gate_payload(scene.brief, scene.assets[:1])["messages"][0]["content"]
    assert "barn_01" in text
    assert "accept" in text
    body = {
        "choices": [
            {"message": {"content": '{"id":"catalog","accept":false,"reason":"The list has no fence."}'}}
        ]
    }
    decision = parse_mesh_gate(body, "catalog")
    assert decision.accept is False
    assert decision.reason == "The list has no fence."


def test_scene_prompt_names_every_catalog_id():
    scene = starter_scene()
    ids = [asset.id for asset in scene.assets]
    text = verdict_payload("scene", ids, [])["messages"][0]["content"][0]["text"]
    for asset_id in ids:
        assert asset_id in text
    assert "tree_01" in text and "bush_01" in text and "crop_01" in text
    assert "wrong_yaw" in text
    assert "redo_asset" in text
    assert "ring" in text
    catalog = catalog_payload(scene.brief)["messages"][0]["content"]
    assert "surround" in catalog
