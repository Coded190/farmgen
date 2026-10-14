"""Catalog and verdict parsing use a fake HTTP body. No key file is opened."""

from farmgen.critic import SCORE_OK_MAX
from farmgen.workers.vlm_critic import catalog_payload, parse_catalog, parse_verdict, verdict_payload


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
