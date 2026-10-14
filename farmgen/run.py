"""Orchestrate Infinigen, ComfyUI, and TRELLIS.2. Each GPU model runs alone."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import time
from pathlib import Path

import numpy as np
import trimesh

from farmgen.cache import cache_hit, cache_key, cache_path
from farmgen.compose import export_scene
from farmgen.contact import HOLE_FRACTION_LIMIT, seal_contact_floor
from farmgen.gpu import IMAGE_BUDGET_MIB, TRELLIS_BUDGET_MIB, gpu_memory_mib, wait_for_vram
from farmgen.ground import (
    inspect_ground,
    load_ground,
    plot_origin,
    sample_height,
    same_ground_geometry,
    terrain_mesh,
)
from farmgen.image import free_memory, queue_image
from farmgen.log import log_line
from farmgen.orient import apply_rotation, next_rotation
from farmgen.paths import comfy_dir, infinigen_dir, output_dir, repo_root, trellis_dir
from farmgen.placer import place_scene
from farmgen.scale import bbox_size, relative_size_error, scale_vertices_to_size, size_error_ok
from farmgen.scene import Scene, save_scene

COMFY_URL = "http://127.0.0.1:8188"


def run_ground(scene: Scene, scene_file: Path) -> Path:
    out = output_dir(scene.id)
    out.mkdir(parents=True, exist_ok=True)
    work = out / "ground_work"
    export = out / "ground_export"
    mesh_path = out / "ground.obj"
    _stop_vlm()
    _try_infinigen(scene, work, export, mesh_path)
    used_analytic = False
    if not mesh_path.is_file():
        _write_terrain(scene, mesh_path, reason="infinigen produced no mesh")
        used_analytic = True
    report = _inspect_terrain(scene, load_ground(mesh_path))
    _log_ground(scene, report)
    if not report.ok:
        _write_terrain(scene, mesh_path, reason=report.reason)
        used_analytic = True
        report = _inspect_terrain(scene, load_ground(mesh_path))
        _log_ground(scene, report)
    if not report.ok:
        raise RuntimeError(f"ground check failed: {report.reason}")

    original = np.asarray(load_ground(mesh_path).vertices)
    reloaded = np.asarray(load_ground(mesh_path).vertices)
    if not same_ground_geometry(original, reloaded):
        raise RuntimeError("ground mesh changed between loads")
    if used_analytic:
        analytic = np.asarray(_analytic_ground(scene).vertices)
        if not same_ground_geometry(original, analytic):
            raise RuntimeError("terrain vertices changed across builds")
    _render_ground_pair(scene, mesh_path, out)
    log_line(scene.id, scene.seed, "ground", output=str(mesh_path), faces=report.faces, bbox_m=report.bbox_m)
    return mesh_path


def run_assets(scene: Scene, scene_file: Path, only: list[str] | None, force: bool = False) -> None:
    selected = [asset for asset in scene.assets if only is None or asset.id in only]
    if force:
        for asset in selected:
            cached = cache_path(cache_key(asset.prompt, asset.size_m))
            cached.unlink(missing_ok=True)
            cached.with_suffix(".raw.glb").unlink(missing_ok=True)
    missing = [asset for asset in selected if not cache_hit(cache_key(asset.prompt, asset.size_m))]
    if missing:
        _generate_images(scene, missing)
        _generate_meshes(scene, missing)
    failed: list[str] = []
    for asset in selected:
        key = cache_key(asset.prompt, asset.size_m)
        path = cache_path(key)
        if not path.is_file():
            failed.append(asset.id)
            log_line(scene.id, scene.seed, "asset", asset_id=asset.id, error="missing cache")
            continue
        scaled = output_dir(scene.id) / "assets" / f"{asset.id}.glb"
        try:
            _scale_cached(scene, asset, path, scaled)
        except RuntimeError as exc:
            failed.append(asset.id)
            scaled.unlink(missing_ok=True)
            log_line(scene.id, scene.seed, "asset", asset_id=asset.id, error=str(exc))
            continue
        asset.cache_key = key
        asset.mesh_path = str(scaled)
    save_scene(scene, scene_file)
    if failed:
        raise RuntimeError(f"asset failures: {','.join(failed)}")


def run_place(scene: Scene, scene_file: Path) -> None:
    place_scene(scene)
    _stamp_heights(scene)
    save_scene(scene, scene_file)


def run_compose(scene: Scene, scene_file: Path) -> Path:
    out = output_dir(scene.id)
    ground = out / "ground.obj"
    exported = export_scene(scene, ground, out)
    log_line(scene.id, scene.seed, "compose", output=str(exported))
    blender = _find_blender()
    if blender is None:
        log_line(scene.id, scene.seed, "compose", render="skipped", reason="blender missing")
        return exported
    _blender(
        blender,
        [
            "--job",
            "compose",
            "--mesh",
            str(out / "farm.glb"),
            "--output",
            str(out),
            "--size",
            str(scene.ground.terrain_m[0]),
            str(scene.ground.terrain_m[1]),
            "--soil",
            "0" if _ground_has_bake(ground) else "1",
        ],
    )
    return exported


def _try_infinigen(scene: Scene, work: Path, export: Path, mesh_path: Path) -> None:
    root = infinigen_dir()
    python = root / ".venv" / "bin" / "python"
    if not python.is_file():
        log_line(scene.id, scene.seed, "ground", backend="flat_plane", reason="infinigen venv missing")
        return
    _ensure_farm_gin(root)
    env = os.environ.copy()
    env.pop("CUDA_HOME", None)
    cmd = [
        str(python),
        "-m",
        "infinigen_examples.generate_nature",
        "--seed",
        str(scene.seed),
        "--task",
        "coarse",
        "fine_terrain",
        "mesh_save",
        "-g",
        "plain",
        "no_assets",
        "farm_field",
        "--output_folder",
        str(work),
    ]
    log_line(scene.id, scene.seed, "ground", backend="infinigen", cmd=" ".join(cmd))
    result = subprocess.run(cmd, cwd=root, env=env)
    if result.returncode != 0:
        log_line(scene.id, scene.seed, "ground", backend="infinigen", returncode=result.returncode)
        return
    export_cmd = [
        str(python),
        "-m",
        "infinigen.tools.export",
        "--input_folder",
        str(work),
        "--output_folder",
        str(export),
        "-f",
        "obj",
        "-v",
        "-r",
        "512",
    ]
    exported = subprocess.run(export_cmd, cwd=root, env=env)
    if exported.returncode != 0:
        export_cmd[export_cmd.index("-f") + 1] = "usdc"
        subprocess.run(export_cmd, cwd=root, env=env)
    found = _first_mesh(export)
    if found is not None:
        _copy_ground_bundle(found, mesh_path)


def _ensure_farm_gin(root: Path) -> None:
    source = repo_root() / "configs" / "farm_field.gin"
    if not source.is_file():
        return
    destinations = list(root.rglob("configs_nature"))
    if not destinations:
        raise FileNotFoundError(f"no configs_nature directory under {root}")
    for folder in destinations:
        shutil.copy(source, folder / "farm_field.gin")


def _write_terrain(scene: Scene, mesh_path: Path, reason: str) -> None:
    log_line(scene.id, scene.seed, "ground", backend="terrain_mesh", reason=reason)
    mesh = _analytic_ground(scene)
    mesh_path.parent.mkdir(parents=True, exist_ok=True)
    mesh.export(mesh_path)


def _analytic_ground(scene: Scene):
    return terrain_mesh(
        scene.ground.terrain_m,
        scene.ground.plot_m,
        scene.ground.relief_m,
        scene.ground.surround_relief_m,
        seed=scene.seed,
    )


def _inspect_terrain(scene: Scene, mesh):
    return inspect_ground(
        mesh,
        scene.ground.terrain_m,
        plot_m=scene.ground.plot_m,
        relief_m=scene.ground.relief_m,
        surround_relief_m=scene.ground.surround_relief_m,
    )


def _log_ground(scene: Scene, report) -> None:
    log_line(
        scene.id,
        scene.seed,
        "ground_check",
        meshes=report.meshes,
        faces=report.faces,
        bbox_m=report.bbox_m,
        thickness_m=report.thickness_m,
        ok=int(report.ok),
        reason=report.reason,
    )


def _copy_ground_bundle(found: Path, mesh_path: Path) -> None:
    mesh_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(found, mesh_path)
    for sibling in found.parent.iterdir():
        if sibling.resolve() == found.resolve():
            continue
        if sibling.suffix.lower() in {".mtl", ".png", ".jpg", ".jpeg", ".tif", ".tiff", ".exr"}:
            shutil.copy(sibling, mesh_path.parent / sibling.name)


def _ground_has_bake(mesh_path: Path) -> bool:
    material = mesh_path.with_suffix(".mtl")
    if not material.is_file():
        return False
    for line in material.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.lower().startswith("map_"):
            continue
        name = line.split(maxsplit=1)[-1].strip().strip('"')
        if (mesh_path.parent / name).is_file() or Path(name).is_file():
            return True
    return False


def _stamp_heights(scene: Scene) -> None:
    mesh_path = output_dir(scene.id) / "ground.obj"
    if not mesh_path.is_file():
        return
    ground = load_ground(mesh_path)
    origin_x, origin_y = plot_origin(scene.ground.terrain_m, scene.ground.plot_m)
    for asset in scene.assets:
        for inst in asset.instances:
            world = (inst.xy[0] + origin_x, inst.xy[1] + origin_y)
            inst.z = sample_height(ground.vertices, ground.faces, world)
            log_line(
                scene.id,
                scene.seed,
                "place",
                asset_id=asset.id,
                instance=inst.index,
                z=inst.z,
            )


def _render_ground_pair(scene: Scene, mesh_path: Path, out: Path) -> None:
    blender = _find_blender()
    if blender is None:
        log_line(scene.id, scene.seed, "ground_render", render="skipped", reason="blender missing")
        return
    baked = _ground_has_bake(mesh_path)
    seeds = (scene.seed,) if baked else (scene.seed, scene.seed + 1)
    for seed in seeds:
        _blender(
            blender,
            [
                "--job",
                "ground",
                "--mesh",
                str(mesh_path),
                "--output",
                str(out / f"ground_render_seed{seed}"),
                "--seed",
                str(seed),
                "--size",
                str(scene.ground.terrain_m[0]),
                str(scene.ground.terrain_m[1]),
                "--soil",
                "0" if baked else "1",
            ],
        )
    top0 = out / f"ground_render_seed{scene.seed}" / f"ground_top_seed{scene.seed}.png"
    top1 = out / f"ground_render_seed{scene.seed + 1}" / f"ground_top_seed{scene.seed + 1}.png"
    if top0.is_file() and top1.is_file() and top0.read_bytes() == top1.read_bytes():
        raise RuntimeError("soil seeds produced identical images")
    viewport = out / "ground_viewport.png"
    if top0.is_file():
        shutil.copy(top0, viewport)


def _generate_images(scene: Scene, assets) -> None:
    _wait_gpu(scene, IMAGE_BUDGET_MIB)
    before = gpu_memory_mib()
    log_line(scene.id, scene.seed, "gpu", when="before_image", memory_mib=before)
    proc = _start_comfy()
    try:
        for asset in assets:
            dest = output_dir(scene.id) / "images" / f"{asset.id}.png"
            started = time.perf_counter()
            queue_image(COMFY_URL, asset.prompt, scene.seed, dest)
            asset.image_path = str(dest)
            log_line(
                scene.id,
                scene.seed,
                "image",
                asset_id=asset.id,
                prompt=asset.prompt,
                output=str(dest),
                seconds=time.perf_counter() - started,
            )
        free_memory(COMFY_URL)
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            proc.kill()
    after = gpu_memory_mib()
    log_line(scene.id, scene.seed, "gpu", when="after_image", memory_mib=after)


def _generate_meshes(scene: Scene, assets) -> None:
    _wait_gpu(scene, TRELLIS_BUDGET_MIB)
    python = trellis_dir() / ".venv" / "bin" / "python"
    if not python.is_file():
        raise RuntimeError(f"TRELLIS venv missing at {python}")
    before = gpu_memory_mib()
    log_line(scene.id, scene.seed, "gpu", when="before_trellis", memory_mib=before)
    worker = repo_root() / "farmgen" / "workers" / "trellis_generate.py"
    failures: list[str] = []
    for asset in assets:
        if not asset.image_path:
            failures.append(asset.id)
            log_line(scene.id, scene.seed, "asset", asset_id=asset.id, error="missing image_path")
            continue
        key = cache_key(asset.prompt, asset.size_m)
        dest = cache_path(key)
        dest.parent.mkdir(parents=True, exist_ok=True)
        raw = dest.with_suffix(".raw.glb")
        cmd = [
            str(python),
            str(worker),
            "--image",
            asset.image_path,
            "--output",
            str(raw),
            "--seed",
            str(scene.seed),
            "--resolution",
            "512",
        ]
        env = os.environ.copy()
        env["PYTHONPATH"] = str(trellis_dir()) + os.pathsep + env.get("PYTHONPATH", "")
        env.setdefault("CUDA_HOME", str(repo_root() / "toolchains" / "cuda-13.2"))
        env["PATH"] = str(repo_root() / "toolchains" / "cuda-13.2" / "bin") + os.pathsep + env.get("PATH", "")
        result = subprocess.run(cmd, cwd=trellis_dir(), env=env)
        if result.returncode != 0 or not raw.is_file():
            raw.unlink(missing_ok=True)
            failures.append(asset.id)
            log_line(
                scene.id,
                scene.seed,
                "asset",
                asset_id=asset.id,
                error=f"trellis returncode={result.returncode}",
            )
            continue
        raw.replace(dest)
    if failures:
        log_line(scene.id, scene.seed, "asset", failed=",".join(failures))
    after = gpu_memory_mib()
    log_line(scene.id, scene.seed, "gpu", when="after_trellis", memory_mib=after)


def _scale_cached(scene: Scene, asset, source: Path, dest: Path) -> None:
    started = time.perf_counter()
    asset_id = asset.id
    prompt = asset.prompt
    size_m = asset.size_m
    mesh = load_ground(source)
    upright = apply_rotation(np.asarray(mesh.vertices), asset.orient)
    try:
        scaled = scale_vertices_to_size(upright, size_m)
    except ValueError as exc:
        dest.unlink(missing_ok=True)
        raise RuntimeError(f"{asset_id} scale failed: {exc}") from exc
    bbox = bbox_size(scaled)
    if not size_error_ok(bbox, size_m):
        err = relative_size_error(bbox, size_m)
        raise RuntimeError(f"{asset_id} bbox misses size_m by {err}")
    sealed, faces, hole, did_seal = seal_contact_floor(scaled, mesh.faces)
    if hole > HOLE_FRACTION_LIMIT:
        raise RuntimeError(f"{asset_id} contact_hole_fraction={hole:.3f}")
    out = trimesh.Trimesh(vertices=sealed, faces=faces, process=False)
    if mesh.visual is not None:
        out.visual = mesh.visual
    dest.parent.mkdir(parents=True, exist_ok=True)
    out.export(dest)
    elapsed = time.perf_counter() - started
    log_line(
        scene.id,
        scene.seed,
        "asset",
        asset_id=asset_id,
        prompt=prompt,
        output=str(dest),
        bbox_m=tuple(round(axis, 1) for axis in bbox),
        vertices=len(out.vertices),
        seconds=elapsed,
        contact_hole_fraction=round(hole, 4),
        floor_sealed=int(did_seal),
        orient=asset.orient,
    )
    _render_asset_views(scene, asset_id, dest)


def reorient_assets(scene: Scene, scene_file: Path, asset_ids: list[str]) -> tuple[list[str], list[str]]:
    """Try the next axis-aligned candidate. Exhausted ids are returned for a mesh redo."""
    turned: list[str] = []
    exhausted: list[str] = []
    for asset_id in asset_ids:
        asset = scene.asset(asset_id)
        nxt = next_rotation(asset.orient)
        if nxt is None:
            exhausted.append(asset_id)
            log_line(scene.id, scene.seed, "orient", asset_id=asset_id, note="candidates_exhausted")
            continue
        asset.orient = nxt
        source = cache_path(cache_key(asset.prompt, asset.size_m))
        dest = output_dir(scene.id) / "assets" / f"{asset.id}.glb"
        if not source.is_file():
            exhausted.append(asset_id)
            log_line(scene.id, scene.seed, "orient", asset_id=asset_id, error="missing cache")
            continue
        _scale_cached(scene, asset, source, dest)
        asset.mesh_path = str(dest)
        turned.append(asset_id)
        log_line(scene.id, scene.seed, "orient", asset_id=asset_id, orient=nxt)
    save_scene(scene, scene_file)
    return turned, exhausted


def _render_asset_views(scene: Scene, asset_id: str, mesh: Path) -> None:
    blender = _find_blender()
    if blender is None:
        return
    dest = output_dir(scene.id) / "thumbs"
    _blender(
        blender,
        [
            "--job",
            "views",
            "--mesh",
            str(mesh),
            "--output",
            str(dest),
            "--asset-id",
            asset_id,
            "--seed",
            str(scene.seed),
        ],
    )


def _wait_gpu(scene: Scene, need_mib: int) -> None:
    from farmgen.workers.vlm_critic import stop_server

    def on_event(when: str, free: int, required: int) -> None:
        log_line(scene.id, scene.seed, "gpu", when=when, free_mib=free, need_mib=required)

    wait_for_vram(need_mib, stop_own=stop_server, on_event=on_event)


def _stop_vlm() -> None:
    from farmgen.workers.vlm_critic import stop_server

    stop_server()


def _thumbnail(scene: Scene, mesh: Path, dest: Path) -> None:
    blender = _find_blender()
    if blender is None:
        return
    _blender(blender, ["--job", "thumbnail", "--mesh", str(mesh), "--output", str(dest), "--seed", str(scene.seed)])


def _start_comfy() -> subprocess.Popen:
    python = comfy_dir() / ".venv" / "bin" / "python"
    if not python.is_file():
        raise RuntimeError(f"ComfyUI venv missing at {python}")
    proc = subprocess.Popen(
        [str(python), "main.py", "--listen", "127.0.0.1", "--port", "8188"],
        cwd=comfy_dir(),
    )
    import requests

    deadline = time.monotonic() + 180
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(f"ComfyUI exited early with {proc.returncode}")
        try:
            if requests.get(f"{COMFY_URL}/system_stats", timeout=2).ok:
                return proc
        except requests.RequestException:
            time.sleep(1)
    proc.kill()
    raise TimeoutError("ComfyUI did not open port 8188")


def _find_blender() -> Path | None:
    root = infinigen_dir()
    matches = sorted(root.glob("blender*/blender"))
    if matches:
        return matches[0]
    which = shutil.which("blender")
    if which:
        return Path(which)
    bundled = root / ".venv" / "bin" / "python"
    if bundled.is_file():
        return bundled
    return None


def _blender(blender: Path, args: list[str]) -> None:
    script = repo_root() / "farmgen" / "workers" / "blender_render.py"
    if blender.name.startswith("python"):
        cmd = [str(blender), str(script), *args]
    else:
        cmd = [str(blender), "--background", "--python", str(script), "--", *args]
    result = subprocess.run(cmd)
    if result.returncode == 0:
        return
    # bpy 4.2 can save the PNG and then segfault while tearing the process down.
    output = None
    if "--output" in args:
        output = Path(args[args.index("--output") + 1])
    wrote = False
    if output is not None and output.is_file() and output.stat().st_size > 0:
        wrote = True
    if output is not None and output.is_dir() and any(output.rglob("*.png")):
        wrote = True
    if wrote and result.returncode in {139, -11}:
        return
    raise RuntimeError(f"blender render failed: {args} returncode={result.returncode}")


def _first_mesh(folder: Path) -> Path | None:
    if not folder.is_dir():
        return None
    for pattern in ("*.obj", "*.glb", "*.usdc", "*.usd"):
        found = sorted(folder.rglob(pattern))
        if found:
            return found[0]
    return None


def _debug_dump(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
