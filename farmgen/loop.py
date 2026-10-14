"""Restart only the module a verdict names.

Contact holes and footprint overlap stay inside scale and placement. They are
not verdict labels and this loop does not ask the VLM about them.
"""

from __future__ import annotations

from pathlib import Path

from farmgen.critic import MAX_RETRIES, RedoAction, Verdict, plan_redos, record_redos, save_verdict
from farmgen.log import log_line
from farmgen.paths import output_dir
from farmgen.run import reorient_assets, run_assets, run_compose, run_ground, run_place
from farmgen.scene import Scene, catalog_or_starter, load_scene, save_scene

_MODULE_ORDER = ("ground", "image", "orient", "place", "compose")


def modules_for(actions: list[RedoAction]) -> list[str]:
    """Modules to run again, in pipeline order, for a list of accepted redos."""
    needed: set[str] = set()
    for action in actions:
        if action.action == "redo_ground":
            needed.add("ground")
        elif action.action in {"redo_image", "redo_asset"}:
            needed.add("image")
        elif action.action == "reorient":
            needed.add("orient")
        elif action.action == "redo_place":
            needed.add("place")
    if "ground" in needed or "image" in needed or "orient" in needed:
        needed.add("place")
    if needed:
        needed.add("compose")
    return [name for name in _MODULE_ORDER if name in needed]


def run_pipeline(
    scene: Scene,
    scene_file: Path,
    only: list[str] | None = None,
    ask=None,
    judge=None,
) -> Path:
    """Catalog, then ground, assets, place, and compose. Restart failed tools.

    ``ask(brief)`` returns catalog JSON. ``judge(scene)`` returns a Verdict.
    Either may be None when the local server is not available; the handwritten
    starter list and the geometric checks still run.
    """
    if ask is not None:
        catalog_or_starter(scene, ask)
        save_scene(scene, scene_file)
    else:
        log_line(scene.id, scene.seed, "catalog", fallback="starter", reason="no_ask")

    run_ground(scene, scene_file)
    run_assets(scene, scene_file, only)
    scene = load_scene(scene_file)
    run_place(scene, scene_file)
    scene = load_scene(scene_file)
    exported = run_compose(scene, scene_file)

    if judge is None:
        log_line(scene.id, scene.seed, "critic", render="skipped", reason="no_judge")
        return exported

    verdict_path = output_dir(scene.id) / "verdict.json"
    verdict = Verdict(entries=[])
    for _round in range(MAX_RETRIES):
        scene = load_scene(scene_file)
        try:
            fresh = judge(scene)
        except (OSError, RuntimeError, ValueError) as exc:
            log_line(scene.id, scene.seed, "critic", render="skipped", error=str(exc))
            break
        fresh.retry_counts = dict(verdict.retry_counts)
        actions, ignored = plan_redos(scene, fresh)
        for reason in ignored:
            log_line(scene.id, scene.seed, "critic", note=reason)
        verdict = record_redos(fresh, actions)
        save_verdict(verdict, verdict_path)
        if not actions:
            log_line(scene.id, scene.seed, "critic", redos=0)
            break
        log_line(scene.id, scene.seed, "critic", modules=",".join(modules_for(actions)), redos=len(actions))
        _restart(scene, scene_file, actions, only)
        exported = output_dir(scene.id) / "farm.glb"
    return exported


def _restart(scene: Scene, scene_file: Path, actions: list[RedoAction], only: list[str] | None) -> None:
    modules = modules_for(actions)
    image_ids = [action.id for action in actions if action.action in {"redo_image", "redo_asset"}]
    orient_ids = [action.id for action in actions if action.action == "reorient"]
    if "ground" in modules:
        run_ground(scene, scene_file)
    if "image" in modules:
        chosen = image_ids if only is None else [item for item in image_ids if item in only]
        if chosen:
            run_assets(scene, scene_file, chosen, force=True)
    if "orient" in modules and orient_ids:
        turned, exhausted = reorient_assets(scene, scene_file, orient_ids)
        if exhausted:
            run_assets(scene, scene_file, exhausted, force=True)
        if not turned and not exhausted:
            log_line(scene.id, scene.seed, "orient", note="no catalog ids")
    if "place" in modules:
        scene = load_scene(scene_file)
        run_place(scene, scene_file)
    if "compose" in modules:
        scene = load_scene(scene_file)
        run_compose(scene, scene_file)
