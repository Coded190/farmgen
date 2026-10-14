import json

import pytest

from farmgen.scene import load_scene, save_scene, starter_scene


def test_round_trip(tmp_path):
    scene = starter_scene()
    path = tmp_path / "scene.json"
    save_scene(scene, path)
    loaded = load_scene(path)
    assert loaded.seed == 0
    assert loaded.ground.size_m == (40.0, 40.0)
    assert loaded.asset("barn_01").size_m == (12.0, 8.0, 7.0)
    assert loaded.model_dump() == scene.model_dump()


def test_missing_size_m_is_rejected(tmp_path):
    path = tmp_path / "scene.json"
    path.write_text(
        json.dumps({"id": "farm_000", "seed": 0, "ground": {"material": "tilled_soil"}, "assets": []}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="size_m"):
        load_scene(path)

    path.write_text(
        json.dumps(
            {
                "id": "farm_000",
                "seed": 0,
                "ground": {"size_m": [40, 40], "material": "tilled_soil"},
                "assets": [{"id": "barn_01", "prompt": "a red wooden barn", "count": 1}],
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="size_m"):
        load_scene(path)
