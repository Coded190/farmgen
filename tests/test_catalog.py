"""Qwen catalog JSON is kept only when it passes the scene contract."""

from farmgen.scene import catalog_or_starter, starter_scene


def test_invalid_catalog_falls_back_after_two_tries():
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
