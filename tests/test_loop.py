"""Redo actions map back onto the module that should run again."""

from farmgen.critic import RedoAction
from farmgen.loop import modules_for


def test_reorient_does_not_schedule_image_or_trellis():
    actions = [RedoAction(action="reorient", id="tractor_01", label="wrong_orientation")]
    assert modules_for(actions) == ["orient", "place", "compose"]


def test_image_redo_comes_before_placement():
    actions = [
        RedoAction(action="redo_place", id="place", label="floating"),
        RedoAction(action="redo_image", id="barn_01", label="missing"),
        RedoAction(action="redo_ground", id="ground", label="not_agricultural"),
    ]
    assert modules_for(actions) == ["ground", "image", "place", "compose"]
