"""Command line for the FarmGen steps."""

from __future__ import annotations

import argparse
import os
import signal
import sys
from pathlib import Path

from farmgen.critic import apply_verdict, load_verdict
from farmgen.log import log_line, set_log_path
from farmgen.paths import output_dir, scene_path
from farmgen.loop import run_pipeline
from farmgen.run import run_assets, run_compose, run_ground, run_place
from farmgen.scene import load_scene, save_scene, starter_scene


def main(argv: list[str] | None = None) -> int:
    restore = _arm_process_cleanup()
    try:
        return _main(argv)
    finally:
        from farmgen.workers.vlm_critic import stop_server

        stop_server()
        restore()


def _arm_process_cleanup():
    """Stop our vLLM when this process is signaled. The caller restores the old handlers."""
    from farmgen.workers.vlm_critic import stop_server

    previous = {signum: signal.getsignal(signum) for signum in (signal.SIGTERM, signal.SIGINT)}

    def _handle(signum, frame):
        stop_server()
        prior = previous.get(signum, signal.SIG_DFL)
        signal.signal(signum, prior)
        if callable(prior):
            prior(signum, frame)
            return
        os.kill(os.getpid(), signum)

    for signum in previous:
        signal.signal(signum, _handle)

    def _restore() -> None:
        for signum, prior in previous.items():
            signal.signal(signum, prior)

    return _restore


def _main(argv: list[str] | None = None) -> int:
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
    full.add_argument(
        "--until",
        choices=["catalog", "ground", "image", "mesh", "scale", "pose", "place", "scene"],
        default=None,
    )
    full.add_argument("--keep-catalog", action="store_true", help="keep the asset list already stored in the scene")

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
        run_assets(
            scene,
            args.scene,
            only or None,
            photo_judge=_photo_judge(scene),
            plant_marker=_plant_marker(),
            mesh_judge=_mesh_judge(args.scene),
        )
    elif args.cmd == "place":
        run_place(scene, args.scene)
    elif args.cmd == "compose":
        run_compose(scene, args.scene)
    elif args.cmd == "critique":
        from farmgen.critic import redo_metrics
        from farmgen.loop import _restart

        verdict = load_verdict(args.verdict)
        actions = apply_verdict(scene, verdict, args.verdict)
        if actions:
            _restart(
                scene,
                args.scene,
                actions,
                None,
                photo_judge=_photo_judge(scene),
                plant_marker=_plant_marker(),
                mesh_judge=_mesh_judge(args.scene),
                lattice=False,
            )
        scene = load_scene(args.scene)
        second_labels: dict[str, str] = {}
        try:
            fresh = _stage_review("scene", scene, None)
            second_labels = {entry.id: entry.label for entry in fresh.entries}
        except (OSError, RuntimeError, ValueError) as exc:
            log_line(
                scene.id,
                scene.seed,
                "critic",
                note="second_pass_skipped",
                error=str(exc).replace("\n", " "),
            )
        n_targets = len(scene.assets) + 1
        rate, clear = redo_metrics(actions, n_targets, second_labels)
        log_line(
            scene.id,
            scene.seed,
            "critic",
            note="second_pass",
            redo_rate=rate,
            second_pass_clear_rate=clear,
            redos=len(actions),
            targets=n_targets,
        )
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
            ask=None if args.keep_catalog else _catalog_ask,
            review=lambda stage, current: _stage_review(stage, current, pose_reference, appearance),
            propose=_propose,
            reference=pose_reference,
            mesh_judge=_mesh_judge(args.scene),
            photo_judge=_photo_judge(scene),
            plant_marker=_plant_marker(),
            catalog_judge=None if args.keep_catalog else _catalog_judge(),
            run_dir=run_dir,
            until=args.until,
        )
    else:
        parser.error(args.cmd)
    return 0


def _scene_photos(folder: Path | None) -> list[Path]:
    if folder is None:
        return []
    return sorted(path for path in Path(folder).iterdir() if path.suffix.lower() in {".png", ".jpg", ".jpeg"})


def _catalog_ask(prompt: str, images=None):
    from farmgen.workers.vlm_critic import request_catalog

    return request_catalog(prompt, images=images)


def _catalog_judge():
    from farmgen.workers.vlm_critic import request_catalog_gate

    def judge(brief: str, assets, images):
        return request_catalog_gate(brief, assets, images)

    return judge


def _stage_review(stage: str, scene, reference: Path | None, appearance: list[Path] | None = None):
    from farmgen.paths import output_dir
    from farmgen.review import stage_images
    from farmgen.views import REFERENCE_TOP

    if stage == "place":
        out = output_dir(scene.id)
        images = []
        top = out / "place_top.png"
        if top.is_file():
            images.append(top)
        if reference is not None:
            ref = reference / REFERENCE_TOP
            if ref.is_file():
                images.append(ref)
        ids = [asset.id for asset in scene.assets]
        if not images:
            from farmgen.log import log_line
            from farmgen.critic import Verdict

            log_line(scene.id, scene.seed, "place_gate", render="skipped", reason="no image")
            return Verdict(entries=[])
        return _finish_review(scene, stage, ids, images)
    ids, images = stage_images(stage, scene, reference)
    for path in appearance or []:
        if path.is_file() and path not in images:
            images.append(path)
    if not images:
        from farmgen.log import log_line

        log_line(scene.id, scene.seed, "critic", review=stage, render="skipped", reason="no image")
        from farmgen.critic import Verdict

        return Verdict(entries=[])
    return _finish_review(scene, stage, ids, images)


def _finish_review(scene, stage: str, ids: list[str], images: list[Path]):
    import json

    from farmgen.paths import output_dir
    from farmgen.workers.vlm_critic import request_verdict

    note = _quality_note(stage, scene)
    verdict = request_verdict(stage, ids, images, note=note)
    folder = output_dir(scene.id) / "manifest"
    folder.mkdir(parents=True, exist_ok=True)
    payload = {
        "stage": stage,
        "inputs": [str(path) for path in images],
        "outputs": [str(folder / f"{stage}.json")],
        "reply": verdict.reply,
    }
    (folder / f"{stage}.json").write_text(json.dumps(payload, indent=2) + "\n")
    return verdict


def _quality_note(stage: str, scene) -> str:
    from pathlib import Path

    from farmgen.contact import contact_hole_fraction
    from farmgen.ground import load_ground
    from farmgen.placer import footprint_issue

    parts: list[str] = []
    if stage in {"place", "scene", "pose"}:
        from farmgen.placer import Footprint

        occupied = []
        overlaps = []
        for asset in scene.assets:
            for instance in asset.instances:
                issue = footprint_issue(
                    instance.xy,
                    instance.yaw,
                    asset.size_m[:2],
                    asset.layout,
                    scene.ground.plot_m,
                    scene.ground.terrain_m,
                    occupied,
                )
                if issue and issue.startswith("overlaps"):
                    overlaps.append(f"{asset.id} {issue}")
                occupied.append(Footprint(asset.id, instance.index, instance.xy, instance.yaw, asset.size_m[:2]))
        parts.append("Footprint overlaps: " + (", ".join(overlaps) if overlaps else "none") + ".")
    if stage == "mesh":
        from farmgen.cache import cache_path
        from farmgen.mesh_gate import mesh_measure_note
        from farmgen.run import _asset_key

        bits = []
        for asset in scene.assets:
            cached = cache_path(_asset_key(asset))
            if not cached.is_file():
                continue
            measure = mesh_measure_note(cached)
            if measure:
                bits.append(f"{asset.id} {measure}")
        parts.append(
            "Measurements, pictures decide: " + (" | ".join(bits) if bits else "none") + "."
        )
    if stage in {"scale", "scene"}:
        holes = []
        for asset in scene.assets:
            if not asset.mesh_path or not Path(asset.mesh_path).is_file():
                continue
            try:
                mesh = load_ground(Path(asset.mesh_path))
                fraction = contact_hole_fraction(mesh.vertices, mesh.faces)
            except (OSError, ValueError, RuntimeError):
                continue
            holes.append(f"{asset.id}={fraction:.3f}")
        parts.append("Contact hole fractions: " + (", ".join(holes) if holes else "none") + ".")
        parts.append(
            "Target height is the third size_m number, in meters: "
            + ", ".join(f"{asset.id}={asset.size_m[2]:.1f}m" for asset in scene.assets)
            + "."
        )
    return " ".join(parts)


def _thin_yaws(mesh: Path) -> tuple[float, ...]:
    import numpy as np

    from farmgen.mesh_gate import _load_mesh, thin_yaw_order
    from farmgen.views import ASSET_YAWS

    loaded = _load_mesh(mesh)
    vertices = np.asarray(getattr(loaded, "vertices", [])) if loaded is not None else np.zeros((0, 3))
    if vertices.ndim != 2 or vertices.shape[0] < 4 or vertices.shape[1] != 3:
        return ASSET_YAWS
    span = vertices.max(axis=0) - vertices.min(axis=0)
    return thin_yaw_order(float(span[0]), float(span[1]))


def _mesh_judge(scene_file: Path):
    import json

    from farmgen.cache import cache_path
    from farmgen.mesh_gate import MeshDecision, fresh_views, mesh_measure_note, thin_yaw_order
    from farmgen.paths import output_dir
    from farmgen.scene import load_scene
    from farmgen.views import asset_view_filename
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
        measure = ""
        if asset.mesh_path and Path(asset.mesh_path).is_file():
            measure = mesh_measure_note(Path(asset.mesh_path))
        if not measure and asset.cache_key:
            measure = mesh_measure_note(cache_path(asset.cache_key))
        if not asset.mesh_path or not Path(asset.mesh_path).is_file():
            decision = MeshDecision(False, "no mesh")
            _record(scene.id, asset_id, attempt, decision)
            return decision
        thumbs = output_dir(scene.id) / "thumbs"
        yaws = _thin_yaws(Path(asset.mesh_path))
        candidates = [thumbs / asset_view_filename(asset_id, yaw) for yaw in yaws]
        images = fresh_views(Path(asset.mesh_path), candidates)
        if asset.image_path and Path(asset.image_path).is_file():
            images.append(Path(asset.image_path))
        if len(images) <= (1 if asset.image_path and Path(asset.image_path).is_file() else 0):
            decision = MeshDecision(False, "stale thumb")
        else:
            decision = request_mesh_gate(asset_id, asset.prompt, images, note=measure)
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
    placed = placed_copies(scene, missing)
    return request_poses(ids, image, required=missing, problems=problems, placed=placed)


def placed_copies(scene, missing: list[tuple[str, int]]) -> str:
    """Copies that already have u,v. A geometric seat stores xy and is left out of this sentence."""
    skip = set(missing)
    notes = []
    for asset in scene.assets:
        for index, instance in enumerate(asset.instances):
            if (asset.id, index) in skip:
                continue
            data = instance.model_dump()
            u, v = data.get("u"), data.get("v")
            if u is None or v is None:
                continue
            notes.append(f"{asset.id} copy {index} at u={float(u):.2f} v={float(v):.2f}")
    return ", ".join(notes)


if __name__ == "__main__":
    sys.exit(main())
