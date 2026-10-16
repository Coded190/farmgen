"""Command line for the FarmGen steps."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from farmgen.critic import apply_verdict, load_verdict
from farmgen.log import log_line, set_log_path
from farmgen.paths import output_dir, scene_path
from farmgen.loop import run_pipeline
from farmgen.run import run_assets, run_compose, run_ground, run_place
from farmgen.scene import load_scene, save_scene, starter_scene


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="farmgen")
    sub = parser.add_subparsers(dest="cmd", required=True)

    init = sub.add_parser("init-scene", help="write the starter scene JSON before generation")
    init.add_argument("--scene-id", default="farm_000")

    ground = sub.add_parser("ground", help="flat soil ground, steps 1-2")
    ground.add_argument("--scene", type=Path, required=True)

    assets = sub.add_parser("assets", help="ComfyUI image then TRELLIS.2 for catalog ids")
    assets.add_argument("--scene", type=Path, required=True)
    assets.add_argument("--only", default="", help="comma-separated asset ids")

    place = sub.add_parser("place", help="write instance transforms into the scene JSON")
    place.add_argument("--scene", type=Path, required=True)

    compose = sub.add_parser("compose", help="one GLB/USD plus Blender proof renders")
    compose.add_argument("--scene", type=Path, required=True)

    critique = sub.add_parser("critique", help="apply a hand-written verdict file")
    critique.add_argument("--scene", type=Path, required=True)
    critique.add_argument("--verdict", type=Path, required=True)

    full = sub.add_parser("run", help="ground, assets, place, compose")
    full.add_argument("--scene", type=Path, required=True)
    full.add_argument("--only", default="")
    full.add_argument("--splat", type=Path, default=None, help="3D Gaussian ply rendered into the five VLM cameras")
    full.add_argument("--reference-dir", type=Path, default=None, help="folder of five reference PNGs, used only when --splat is omitted")

    args = parser.parse_args(argv)
    if args.cmd == "init-scene":
        scene = starter_scene()
        scene.id = args.scene_id
        path = scene_path(args.scene_id)
        save_scene(scene, path)
        print(path)
        return 0

    scene = load_scene(args.scene)
    set_log_path(output_dir(scene.id) / "farmgen.log")
    log_line(scene.id, scene.seed, "start", cmd=args.cmd)

    if args.cmd == "ground":
        run_ground(scene, args.scene)
    elif args.cmd == "assets":
        only = [item for item in args.only.split(",") if item]
        run_assets(scene, args.scene, only or None)
    elif args.cmd == "place":
        run_place(scene, args.scene)
    elif args.cmd == "compose":
        run_compose(scene, args.scene)
    elif args.cmd == "critique":
        verdict = load_verdict(args.verdict)
        apply_verdict(scene, verdict, args.verdict)
    elif args.cmd == "run":
        only = [item for item in args.only.split(",") if item]
        from farmgen.reference import resolve_reference

        pose_reference = None
        if args.splat is not None:
            pose_reference = resolve_reference(scene.id, scene.ground.terrain_m, args.splat, None)
        elif args.reference_dir is not None:
            resolve_reference(scene.id, scene.ground.terrain_m, None, args.reference_dir)
        from farmgen.archive import allocate_run_dir

        appearance = _scene_photos(args.reference_dir)
        run_dir = allocate_run_dir(scene.id)
        log_line(scene.id, scene.seed, "archive", run_dir=str(run_dir), note="start")
        run_pipeline(
            scene,
            args.scene,
            only or None,
            ask=_catalog_ask,
            review=lambda stage, current: _stage_review(stage, current, pose_reference, appearance),
            propose=_propose,
            reference=pose_reference,
            mesh_judge=_mesh_judge(args.scene),
            photo_judge=_photo_judge(scene),
            plant_marker=_plant_marker(),
            run_dir=run_dir,
        )
    else:
        parser.error(args.cmd)
    return 0


def _scene_photos(folder: Path | None) -> list[Path]:
    if folder is None:
        return []
    return sorted(path for path in Path(folder).iterdir() if path.suffix.lower() in {".png", ".jpg", ".jpeg"})


def _catalog_ask(prompt: str, images=None) -> dict:
    from farmgen.workers.vlm_critic import request_catalog

    return request_catalog(prompt, images=images)


def _stage_review(stage: str, scene, reference: Path | None, appearance: list[Path] | None = None):
    from farmgen.paths import output_dir
    from farmgen.review import stage_images
    from farmgen.views import compose_view_filename
    from farmgen.workers.vlm_critic import request_verdict

    if stage == "place":
        out = output_dir(scene.id)
        images = [out / "compose_top.png"]
        images.extend(out / compose_view_filename(index) for index in range(4))
        images = [path for path in images if path.is_file()]
        ids = [asset.id for asset in scene.assets]
        if not images:
            from farmgen.log import log_line
            from farmgen.critic import Verdict

            log_line(scene.id, scene.seed, "place_gate", render="skipped", reason="no image")
            return Verdict(entries=[])
        return request_verdict("place", ids, images)
    ids, images = stage_images(stage, scene, reference)
    for path in appearance or []:
        if path.is_file() and path not in images:
            images.append(path)
    if not images:
        from farmgen.log import log_line

        log_line(scene.id, scene.seed, "critic", review=stage, render="skipped", reason="no image")
        from farmgen.critic import Verdict

        return Verdict(entries=[])
    return request_verdict(stage, ids, images)


def _mesh_judge(scene_file: Path):
    import json

    from farmgen.cache import cache_path
    from farmgen.mesh_gate import MeshDecision, flat_card_reason, fresh_views
    from farmgen.paths import output_dir
    from farmgen.scene import load_scene
    from farmgen.views import ASSET_YAWS, asset_view_filename
    from farmgen.workers.vlm_critic import request_mesh_gate

    def _record(scene_id: str, asset_id: str, attempt: int, decision: MeshDecision) -> None:
        record = output_dir(scene_id) / "mesh_gate" / f"{asset_id}_attempt_{attempt}.json"
        record.parent.mkdir(parents=True, exist_ok=True)
        record.write_text(
            json.dumps(
                {"id": asset_id, "accept": decision.accept, "reason": decision.reason, "reply": decision.reply},
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )

    def judge(asset_id: str, attempt: int) -> MeshDecision:
        scene = load_scene(scene_file)
        asset = scene.asset(asset_id)
        if asset.cache_key:
            reason = flat_card_reason(cache_path(asset.cache_key))
            if reason:
                decision = MeshDecision(False, reason)
                _record(scene.id, asset_id, attempt, decision)
                return decision
        if not asset.mesh_path or not Path(asset.mesh_path).is_file():
            decision = MeshDecision(False, "no mesh")
            _record(scene.id, asset_id, attempt, decision)
            return decision
        thumbs = output_dir(scene.id) / "thumbs"
        candidates = [thumbs / asset_view_filename(asset_id, yaw) for yaw in ASSET_YAWS]
        images = fresh_views(Path(asset.mesh_path), candidates)
        if asset.image_path and Path(asset.image_path).is_file():
            images.append(Path(asset.image_path))
        if len(images) <= (1 if asset.image_path and Path(asset.image_path).is_file() else 0):
            decision = MeshDecision(False, "stale thumb")
        else:
            decision = request_mesh_gate(asset_id, asset.prompt, images)
        _record(scene.id, asset_id, attempt, decision)
        return decision

    return judge


def _plant_marker():
    from farmgen.workers.vlm_critic import request_plant_box

    def marker(asset_id: str, prompt: str, image: Path):
        return request_plant_box(asset_id, prompt, image)

    return marker


def _photo_judge(scene):
    from farmgen.workers.vlm_critic import request_photo_gate

    def judge(asset_id: str, prompt: str, image: Path, _attempt: int):
        return request_photo_gate(asset_id, prompt, image)

    return judge


def _propose(scene, reference: Path | None, missing: list[tuple[str, int]], problems=None):
    from farmgen.paths import output_dir
    from farmgen.views import REFERENCE_TOP
    from farmgen.workers.vlm_critic import request_poses

    if reference is not None:
        image = reference / REFERENCE_TOP
    else:
        image = output_dir(scene.id) / "ground_kept_top.png"
    if not image.is_file():
        raise FileNotFoundError(f"pose image missing: {image.name}")
    ids = sorted({asset_id for asset_id, _index in missing})
    return request_poses(ids, image, required=missing, problems=problems)


if __name__ == "__main__":
    sys.exit(main())
