"""Critic verdicts and the redo actions they schedule.

``farmgen critique --verdict`` still applies a file a person wrote. The local
Qwen client writes the same file. A score at or below 4 schedules nothing.
Contact holes and overlap stay outside this module.
"""

from __future__ import annotations

import json
from pathlib import Path

from pydantic import BaseModel, Field, field_validator

from farmgen.log import log_line
from farmgen.scene import Scene

LABELS = frozenset(
    {
        "ok",
        "missing",
        "wrong_scale",
        "floating",
        "overlapping",
        "not_agricultural",
        "wrong_orientation",
    }
)
ACTIONS = frozenset({"redo_ground", "redo_asset", "reorient", "redo_image", "redo_place"})
# A judge score at or below this value is stored and does not schedule a redo.
SCORE_OK_MAX = 4
MAX_RETRIES = 2
_CATALOG_ACTIONS = frozenset({"redo_asset", "reorient", "redo_image"})


class VerdictEntry(BaseModel):
    id: str
    label: str
    action: str | None = None
    score: int | None = None

    @field_validator("label")
    @classmethod
    def _label(cls, value: str) -> str:
        if value not in LABELS:
            raise ValueError(f"label must be one of {sorted(LABELS)}")
        return value

    @field_validator("action")
    @classmethod
    def _action(cls, value: str | None) -> str | None:
        if value is None:
            return None
        if value not in ACTIONS:
            raise ValueError(f"action must be one of {sorted(ACTIONS)}")
        return value

    @field_validator("score")
    @classmethod
    def _score(cls, value: int | None) -> int | None:
        if value is None:
            return None
        if value < 0 or value > 10:
            raise ValueError("score must be from 0 to 10")
        return value


class Verdict(BaseModel):
    entries: list[VerdictEntry]
    retry_counts: dict[str, int] = Field(default_factory=dict)


class RedoAction(BaseModel):
    action: str
    id: str
    label: str


def load_verdict(path: Path) -> Verdict:
    return Verdict.model_validate(json.loads(path.read_text(encoding="utf-8")))


def save_verdict(verdict: Verdict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(verdict.model_dump_json(indent=2) + "\n", encoding="utf-8")


def plan_redos(scene: Scene, verdict: Verdict) -> tuple[list[RedoAction], list[str]]:
    """Return accepted redos and human-readable ignore reasons.

    ``retry_counts`` is the number of redos already spent. A target at the cap
    is not scheduled again. ``redo_asset`` for an id outside the catalog is
    ignored. ``redo_ground`` is only valid for the ground entry.
    """
    catalog = {asset.id for asset in scene.assets}
    accepted: list[RedoAction] = []
    ignored: list[str] = []
    for entry in verdict.entries:
        if entry.score is not None and entry.score <= SCORE_OK_MAX:
            continue
        if entry.label == "ok" or entry.action is None:
            continue
        target = "ground" if entry.action == "redo_ground" else entry.id
        spent = int(verdict.retry_counts.get(target, 0))
        if entry.action in _CATALOG_ACTIONS and entry.id not in catalog:
            ignored.append(f"ignore unknown object type id={entry.id} action={entry.action}")
            continue
        if entry.action == "redo_place" and entry.id not in catalog and entry.id != "place":
            ignored.append(f"ignore redo_place on id={entry.id}")
            continue
        if entry.action == "redo_ground" and entry.id != "ground":
            ignored.append(f"ignore redo_ground on id={entry.id}")
            continue
        if spent >= MAX_RETRIES:
            ignored.append(f"ignore retry cap id={target} spent={spent}")
            continue
        accepted.append(RedoAction(action=entry.action, id=target, label=entry.label))
    return accepted, ignored


def record_redos(verdict: Verdict, actions: list[RedoAction]) -> Verdict:
    counts = dict(verdict.retry_counts)
    for action in actions:
        counts[action.id] = int(counts.get(action.id, 0)) + 1
    return verdict.model_copy(update={"retry_counts": counts})


def redo_metrics(
    first_actions: list[RedoAction],
    n_targets: int,
    second_labels: dict[str, str],
) -> tuple[float, float]:
    """Redo rate is first-pass redo actions divided by verdict targets.

    ``second_pass_clear_rate`` is the fraction of those redos whose second
    label is no longer the same failure label. A missing second label does
    not count as cleared.
    """
    if n_targets <= 0:
        raise ValueError("n_targets must be positive")
    redo_rate = len(first_actions) / n_targets
    if not first_actions:
        return redo_rate, 1.0
    cleared = 0
    for action in first_actions:
        second = second_labels.get(action.id)
        if second is not None and second != action.label:
            cleared += 1
    return redo_rate, cleared / len(first_actions)


def apply_verdict(scene: Scene, verdict: Verdict, verdict_path: Path) -> list[RedoAction]:
    actions, ignored = plan_redos(scene, verdict)
    for reason in ignored:
        log_line(scene.id, scene.seed, "critic", note=reason)
    for action in actions:
        log_line(scene.id, scene.seed, "critic", action=action.action, id=action.id, label=action.label)
    n_targets = len(scene.assets) + 1
    rate, clear = redo_metrics(actions, n_targets, {})
    log_line(
        scene.id,
        scene.seed,
        "critic",
        redo_rate=rate,
        second_pass_clear_rate=clear,
        redos=len(actions),
        targets=n_targets,
        ignored=len(ignored),
    )
    updated = record_redos(verdict, actions)
    save_verdict(updated, verdict_path)
    return actions
