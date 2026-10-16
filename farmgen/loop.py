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

_MODULE_ORDER = ("ground", "image", "orient", "place", "turn", "move", "compose")
_STAGE_ACTIONS = {
    "ground": frozenset({"redo_ground"}),
    "image": frozenset({"redo_image"}),
    "mesh": frozenset({"redo_asset"}),
    "scale": frozenset({"reorient"}),
    "place": frozenset({"move"}),
    "scene": frozenset({"move", "turn", "redo_place", "redo_asset", "redo_image", "reorient", "redo_ground"}),
}


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
        elif action.action == "turn":
            needed.add("turn")
        elif action.action == "move":
            needed.add("move")
    if "ground" in needed or "image" in needed or "orient" in needed:
        needed.add("place")
    if needed:
        needed.add("compose")
    return [name for name in _MODULE_ORDER if name in needed]


def _archive(scene: Scene, run_dir: Path | None) -> None:
    if run_dir is None:
        return
    from farmgen.archive import archive_run

    archive_run(scene.id, scene.seed, run_dir, scene)


def _scale_for_views(scene: Scene, scene_file: Path, only: list[str] | None) -> None:
    """Stand each mesh upright and write its four orbit views. A scale error stays in the log."""
    from farmgen.run import run_scale

    try:
        run_scale(scene, scene_file, only)
    except RuntimeError as exc:
        log_line(scene.id, scene.seed, "scale", error=str(exc))


def run_pipeline(
    scene: Scene,
    scene_file: Path,
    only: list[str] | None = None,
    ask=None,
    judge=None,
    review=None,
    propose=None,
    reference: Path | None = None,
    mesh_judge=None,
    photo_judge=None,
    plant_marker=None,
    run_dir: Path | None = None,
) -> Path:
    """Catalog, ground, one mesh per asset, then Qwen poses and one scene assessment.

    ``ask(prompt)`` or ``ask(prompt, images)`` returns catalog JSON. When
    ``reference`` is set, those splat views are the pictures Qwen names objects
    from. ``mesh_judge(asset_id, attempt)`` accepts a mesh before any copy is
    posed. ``propose(scene, reference, missing)`` returns a pose for each
    missing copy. The lattice does not invent positions once ``propose`` is set.
    """
    from farmgen.mesh_gate import apply_mesh_gate
    from farmgen.poses import fill_poses, place_fences, place_fields, seat_visible
    from farmgen.run import run_images, run_meshes, run_scale
    from farmgen.views import reference_filenames

    catalog_images = []
    if reference is not None:
        catalog_images = [reference / name for name in reference_filenames() if (reference / name).is_file()]
    if ask is not None:
        catalog_or_starter(scene, ask, images=catalog_images, record_dir=output_dir(scene.id))
        save_scene(scene, scene_file)
    else:
        log_line(scene.id, scene.seed, "catalog", note="kept", reason="no_ask")

    run_ground(scene, scene_file)
    scene = load_scene(scene_file)
    run_images(scene, scene_file, only, photo_judge=photo_judge, plant_marker=plant_marker)
    _archive(scene, run_dir)
    run_meshes(scene, scene_file, only, photo_judge=photo_judge)
    scene = load_scene(scene_file)
    _scale_for_views(scene, scene_file, only)
    scene = load_scene(scene_file)
    _archive(scene, run_dir)

    def _regen(asset) -> None:
        run_images(scene, scene_file, [asset.id], force=True, photo_judge=photo_judge, plant_marker=plant_marker)
        run_meshes(scene, scene_file, [asset.id], force=True, photo_judge=photo_judge)
        _scale_for_views(scene, scene_file, [asset.id])

    accepted = {asset.id for asset in scene.assets}
    if mesh_judge is not None:
        accepted = apply_mesh_gate(scene, mesh_judge, _regen)
        save_scene(scene, scene_file)
    if accepted:
        _scale_for_views(scene, scene_file, list(accepted))
        scene = load_scene(scene_file)
        accepted = {asset.id for asset in scene.assets if asset.id in accepted and asset.mesh_path}
    if propose is not None and accepted:
        import inspect

        parameters = inspect.signature(propose).parameters
        if len(parameters) >= 4 or any(
            item.kind is inspect.Parameter.VAR_POSITIONAL for item in parameters.values()
        ):
            ask = lambda missing, problems=None: propose(scene, reference, missing, problems)
        else:
            ask = lambda missing, problems=None: propose(scene, reference, missing)
        place_fences(scene)
        posed = _posed_ids(scene, accepted)
        fill_poses(scene, ask, posed)
        seat_visible(scene, posed)
        place_fields(scene)
        save_scene(scene, scene_file)
    run_place(scene, scene_file, lattice=propose is None)
    scene = load_scene(scene_file)
    exported = run_compose(scene, scene_file)
    _archive(scene, run_dir)
    scene = load_scene(scene_file)

    def _on_scene(actions: list[RedoAction]) -> None:
        from farmgen.run import restamp_heights

        moves = [action for action in actions if action.action == "move"]
        for action in actions:
            if action.action != "move":
                log_line(
                    scene.id,
                    scene.seed,
                    "place_gate",
                    id=action.id,
                    note="ignored",
                    action=action.action,
                    reason="mesh already judged",
                )
        if moves:
            before = {asset.id: [inst.model_copy() for inst in asset.instances] for asset in scene.assets}
            _apply_moves(scene, moves)
            _restore_stacked_moves(scene, before)
            seat_visible(scene, {asset.id for asset in scene.assets})
            restamp_heights(scene, scene_file)
            run_compose(scene, scene_file)
            _archive(scene, run_dir)

    if review is not None:
        _review_stage(scene, scene_file, "place", review, only, on_scene=_on_scene)
        return output_dir(scene.id) / "farm.glb"
    if judge is None:
        log_line(scene.id, scene.seed, "critic", render="skipped", reason="no_judge")
        return exported
    return run_critic(scene, scene_file, judge, only)


def _posed_ids(scene: Scene, accepted: set[str]) -> set[str]:
    """Fence segments and field copies are seated geometrically, not by Qwen."""
    skip = {asset.id for asset in scene.assets if asset.layout == "perimeter" or asset.field_m is not None}
    return {asset_id for asset_id in accepted if asset_id not in skip}


def run_critic(scene: Scene, scene_file: Path, judge, only: list[str] | None = None) -> Path:
    """Ask the facing judge and apply turns. Does not rebuild images or meshes."""
    exported = output_dir(scene.id) / "farm.glb"
    verdict_path = output_dir(scene.id) / "verdict.json"
    verdict = Verdict(entries=[])
    for _round in range(MAX_RETRIES):
        scene = load_scene(scene_file)
        try:
            fresh = judge(scene)
        except (OSError, RuntimeError, ValueError) as exc:
            log_line(scene.id, scene.seed, "critic", facing="failed", error=str(exc))
            break
        for entry in fresh.entries:
            log_line(
                scene.id,
                scene.seed,
                "critic",
                facing="yes",
                id=entry.id,
                yaw_deg=-1 if entry.yaw_deg is None else entry.yaw_deg,
                score=-1 if entry.score is None else entry.score,
                label=entry.label,
            )
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


def _review_stage(
    scene: Scene,
    scene_file: Path,
    stage: str,
    review,
    only: list[str] | None,
    on_scene=None,
) -> Scene:
    """Ask once per stage. A scheduled redo reruns that stage, up to the cap."""
    if review is None:
        return scene
    allowed = _STAGE_ACTIONS[stage]
    verdict_path = output_dir(scene.id) / "verdict.json"
    verdict = Verdict(entries=[])
    for _round in range(MAX_RETRIES):
        scene = load_scene(scene_file)
        try:
            fresh = review(stage, scene)
        except (OSError, RuntimeError, ValueError) as exc:
            log_line(scene.id, scene.seed, "critic", review=stage, facing="failed", error=str(exc))
            break
        if fresh is None:
            break
        fresh.retry_counts = dict(verdict.retry_counts)
        actions, ignored = plan_redos(scene, fresh)
        actions = [action for action in actions if action.action in allowed]
        for reason in ignored:
            log_line(scene.id, scene.seed, "critic", review=stage, note=reason)
        verdict = record_redos(fresh, actions)
        save_verdict(verdict, verdict_path)
        if not actions:
            log_line(scene.id, scene.seed, "critic", review=stage, redos=0)
            break
        log_line(scene.id, scene.seed, "critic", review=stage, redos=len(actions))
        if stage == "scene" and on_scene is not None:
            on_scene(actions)
        elif stage == "scene":
            _restart(scene, scene_file, actions, only)
        else:
            _rerun_forward(scene, scene_file, stage, actions, only)
    return load_scene(scene_file)


def _rerun_forward(scene: Scene, scene_file: Path, stage: str, actions: list[RedoAction], only: list[str] | None) -> None:
    """Rerun one stage and stop. Later stages still run in order after this."""
    from farmgen.run import render_mesh_thumbs, render_place_preview, restamp_heights, run_images, run_meshes

    if stage == "ground":
        run_ground(scene, scene_file)
        return
    if stage == "image":
        ids = [action.id for action in actions]
        chosen = ids if only is None else [item for item in ids if item in only]
        if chosen:
            run_images(scene, scene_file, chosen, force=True)
        return
    if stage == "mesh":
        ids = [action.id for action in actions]
        chosen = ids if only is None else [item for item in ids if item in only]
        if chosen:
            run_images(scene, scene_file, chosen, force=True)
            run_meshes(scene, scene_file, chosen)
            render_mesh_thumbs(scene, chosen)
        return
    if stage == "scale":
        ids = [action.id for action in actions if action.action == "reorient"]
        if ids:
            turned, exhausted = reorient_assets(scene, scene_file, ids)
            if exhausted:
                from farmgen.run import run_assets

                run_assets(scene, scene_file, exhausted, force=True)
            if not turned and not exhausted:
                log_line(scene.id, scene.seed, "orient", note="no catalog ids")
        return
    if stage == "place":
        scene = load_scene(scene_file)
        _apply_moves(scene, actions)
        restamp_heights(scene, scene_file)
        render_place_preview(scene)


def _apply_moves(scene: Scene, actions: list[RedoAction]) -> None:
    import math

    for action in actions:
        if action.action != "move" or action.xy is None or action.yaw_deg is None:
            continue
        asset = scene.asset(action.id)
        heading = math.radians(float(action.yaw_deg))
        for inst in asset.instances:
            if action.index is not None and inst.index != action.index:
                continue
            inst.xy = (float(action.xy[0]), float(action.xy[1]))
            inst.yaw = heading
        log_line(
            scene.id,
            scene.seed,
            "critic",
            facing="move",
            id=action.id,
            xy=f"{action.xy[0]:.4f},{action.xy[1]:.4f}",
            yaw_deg=action.yaw_deg,
        )


def _restore_stacked_moves(scene: Scene, before: dict) -> None:
    """Put a copy back when the requested move covers another footprint."""
    from farmgen.placer import Footprint, footprint_issue

    for asset in scene.assets:
        occupied = []
        for other in scene.assets:
            if other.id == asset.id:
                continue
            for inst in other.instances:
                occupied.append(Footprint(other.id, inst.index, inst.xy, inst.yaw, other.size_m[:2]))
        for inst in asset.instances:
            issue = footprint_issue(
                inst.xy,
                inst.yaw,
                asset.size_m[:2],
                asset.layout,
                scene.ground.plot_m,
                scene.ground.terrain_m,
                occupied,
            )
            if issue is None:
                occupied.append(Footprint(asset.id, inst.index, inst.xy, inst.yaw, asset.size_m[:2]))
                continue
            asset.instances = list(before.get(asset.id, []))
            log_line(
                scene.id,
                scene.seed,
                "place_gate",
                id=asset.id,
                note="ignored",
                reason=issue,
            )
            break


def _apply_turns(scene: Scene, actions: list[RedoAction]) -> None:
    import math

    for action in actions:
        if action.action != "turn" or action.yaw_deg is None:
            continue
        asset = scene.asset(action.id)
        heading = math.radians(float(action.yaw_deg))
        for inst in asset.instances:
            inst.yaw = heading
        log_line(scene.id, scene.seed, "critic", facing="turn", id=action.id, yaw_deg=action.yaw_deg)


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
    if "turn" in modules:
        scene = load_scene(scene_file)
        _apply_turns(scene, actions)
        save_scene(scene, scene_file)
    if "move" in modules:
        from farmgen.run import restamp_heights

        scene = load_scene(scene_file)
        _apply_moves(scene, actions)
        restamp_heights(scene, scene_file)
    if "compose" in modules:
        scene = load_scene(scene_file)
        run_compose(scene, scene_file)
