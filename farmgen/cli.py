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
        run_pipeline(scene, args.scene, only or None, ask=_catalog_ask, judge=_scene_judge)
    else:
        parser.error(args.cmd)
    return 0


def _catalog_ask(brief: str) -> dict:
    from farmgen.workers.vlm_critic import request_catalog

    return request_catalog(brief)


def _scene_judge(scene):
    from farmgen.paths import output_dir
    from farmgen.views import ASSET_YAWS, asset_view_filename, compose_view_filename
    from farmgen.workers.vlm_critic import request_verdict

    out = output_dir(scene.id)
    images = [out / compose_view_filename(index) for index in range(4)]
    missing = [path for path in images if not path.is_file()]
    if missing:
        raise RuntimeError(f"scene views missing: {missing[0].name}")
    ids = ["ground", "place", *[asset.id for asset in scene.assets]]
    verdict = request_verdict("scene", ids, images)
    for asset in scene.assets:
        thumbs = [out / "thumbs" / asset_view_filename(asset.id, yaw) for yaw in ASSET_YAWS]
        if any(not path.is_file() for path in thumbs):
            continue
        asset_verdict = request_verdict("asset", [asset.id], thumbs)
        verdict.entries.extend(asset_verdict.entries)
        picture = out / "images" / f"{asset.id}.png"
        if picture.is_file():
            image_verdict = request_verdict("image", [asset.id], [picture])
            verdict.entries.extend(image_verdict.entries)
    return verdict


if __name__ == "__main__":
    sys.exit(main())
