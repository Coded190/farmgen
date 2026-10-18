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


def test_a_low_score_still_schedules_a_failed_label_and_reorient_shares_the_cap():
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
    assert [action.id for action in actions] == ["tractor_01", "silo_01", "place"]
    assert any("retry cap id=barn_01" in item for item in ignored)

    shared = Verdict(
        entries=[VerdictEntry(id="tractor_01", label="missing", action="redo_asset", score=9)],
        retry_counts={"tractor_01": 2},
    )
    again, ignored_again = plan_redos(scene, shared)
    assert again == []
    assert any("retry cap id=tractor_01" in item for item in ignored_again)


def test_wrong_place_is_kept_when_the_score_is_acceptable():
    scene = starter_scene()
    verdict = Verdict(
        entries=[
            VerdictEntry(id="barn_01", label="wrong_place", action="move", score=2, u=0.5, v=0.5, yaw_deg=0),
            VerdictEntry(id="tractor_01", label="wrong_yaw", action="turn", score=2, yaw_deg=90),
        ]
    )
    actions, ignored = plan_redos(scene, verdict)
    assert ignored == []
    assert [(action.id, action.action) for action in actions] == [("barn_01", "move"), ("tractor_01", "turn")]


def test_move_keeps_a_point_inside_the_yard_and_drops_one_outside():
    scene = starter_scene()
    inside = Verdict(
        entries=[
            VerdictEntry(id="barn_01", label="wrong_place", action="move", score=8, u=0.5, v=0.5, yaw_deg=15.0),
        ]
    )
    actions, ignored = plan_redos(scene, inside)
    assert ignored == []
    assert actions[0].action == "move"
    assert actions[0].xy == (20.0, 20.0)
    assert actions[0].yaw_deg == 15.0

    outside = Verdict(
        entries=[
            VerdictEntry(id="barn_01", label="wrong_place", action="move", score=8, u=0.0, v=0.0, yaw_deg=10),
        ]
    )
    none, notes = plan_redos(scene, outside)
    assert none == []
    assert any("outside" in item for item in notes)


def test_wrong_yaw_schedules_a_turn_at_a_low_score_too():
    scene = starter_scene()
    loud = Verdict(
        entries=[
            VerdictEntry(id="barn_01", label="wrong_yaw", action="turn", score=6, yaw_deg=90),
        ]
    )
    actions, ignored = plan_redos(scene, loud)
    assert ignored == []
    assert [(action.action, action.yaw_deg) for action in actions] == [("turn", 90)]
    assert all(action.action != "reorient" for action in actions)

    quiet = Verdict(
        entries=[
            VerdictEntry(id="barn_01", label="wrong_yaw", action="turn", score=4, yaw_deg=90),
        ]
    )
    quiet_actions, quiet_notes = plan_redos(scene, quiet)
    assert quiet_notes == []
    assert [(action.action, action.yaw_deg) for action in quiet_actions] == [("turn", 90)]
