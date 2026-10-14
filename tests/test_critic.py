from farmgen.critic import RedoAction, Verdict, VerdictEntry, plan_redos, redo_metrics
from farmgen.scene import starter_scene


def test_retry_cap_and_unknown_object():
    scene = starter_scene()
    verdict = Verdict(
        entries=[
            VerdictEntry(id="windmill_01", label="missing", action="redo_asset"),
            VerdictEntry(id="barn_01", label="floating", action="redo_asset"),
            VerdictEntry(id="ground", label="not_agricultural", action="redo_ground"),
        ],
        retry_counts={"barn_01": 2},
    )
    actions, ignored = plan_redos(scene, verdict)
    assert any("unknown object type" in item for item in ignored)
    assert any("retry cap id=barn_01" in item for item in ignored)
    assert [action.action for action in actions] == ["redo_ground"]

    rate, clear = redo_metrics(
        [RedoAction(action="redo_ground", id="ground", label="not_agricultural")],
        n_targets=len(scene.assets) + 1,
        second_labels={"ground": "ok"},
    )
    assert rate == 1 / (len(scene.assets) + 1)
    assert clear == 1.0
    _, still = redo_metrics(
        [RedoAction(action="redo_asset", id="hay_01", label="floating")],
        n_targets=len(scene.assets) + 1,
        second_labels={"hay_01": "floating"},
    )
    assert still == 0.0


def test_low_score_does_not_schedule_and_reorient_shares_the_cap():
    scene = starter_scene()
    quiet = Verdict(
        entries=[
            VerdictEntry(id="tractor_01", label="wrong_orientation", action="reorient", score=4),
            VerdictEntry(id="barn_01", label="wrong_orientation", action="reorient", score=5),
            VerdictEntry(id="silo_01", label="missing", action="redo_image", score=8),
            VerdictEntry(id="place", label="floating", action="redo_place", score=7),
        ],
        retry_counts={"barn_01": 2},
    )
    actions, ignored = plan_redos(scene, quiet)
    assert [action.id for action in actions] == ["silo_01", "place"]
    assert any("retry cap id=barn_01" in item for item in ignored)

    shared = Verdict(
        entries=[VerdictEntry(id="tractor_01", label="missing", action="redo_asset", score=9)],
        retry_counts={"tractor_01": 2},
    )
    again, ignored_again = plan_redos(scene, shared)
    assert again == []
    assert any("retry cap id=tractor_01" in item for item in ignored_again)
