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

from farmgen.cache import cache_hit, cache_key, cache_path, file_sha256
from farmgen.compose import export_scene
from farmgen.contact import HOLE_FRACTION_LIMIT, seal_contact_floor
from farmgen.gpu import IMAGE_BUDGET_MIB, TRELLIS_BUDGET_MIB, gpu_memory_mib, wait_for_vram
from farmgen.ground import (
    inspect_ground,
    load_ground,
    mesh_has_baked_color,
    paint_field_colors,
    plot_origin,
    prepare_ground_mesh,
    sample_height,
    same_ground_geometry,
    terrain_mesh,
)
from farmgen.image import free_memory, queue_image
from farmgen.log import log_line
from farmgen.mesh_gate import flat_card_reason
from farmgen.orient import apply_rotation, next_rotation
from farmgen.paths import comfy_dir, infinigen_dir, output_dir, repo_root, trellis_dir
from farmgen.placer import place_scene
from farmgen.scale import bbox_size, relative_size_error, scale_vertices_to_size, size_error_ok
from farmgen.scene import Scene, save_scene

COMFY_URL = "http://127.0.0.1:8188"
COPY_FACE_CAP = 16_000


def run_ground(scene: Scene, scene_file: Path) -> Path:
    out = output_dir(scene.id)
    out.mkdir(parents=True, exist_ok=True)
    work = out / "ground_work"
    export = out / "ground_export"
    mesh_path = out / "ground.obj"
    _stop_vlm()
    _try_infinigen(scene, work, export, mesh_path)
    discarded = out / "ground_discarded.obj"
    if mesh_path.is_file():
        shutil.copy(mesh_path, discarded)
        _render_ground_compare(scene, discarded, out, "ground_discarded", soil=False, frame="bbox")
        raw = load_ground(mesh_path)
        prepared = prepare_ground_mesh(
            raw,
            scene.ground.terrain_m,
            scene.ground.plot_m,
            scene.ground.relief_m,
            scene.ground.surround_relief_m,
        )
        _write_ground_mesh(prepared, mesh_path)
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
    if not _ground_has_bake(mesh_path):
        # Infinigen's mud shader is not baked into the OBJ. The exporter writes
        # map_Kd only when a PNG exists; this run has the MTL line and no image.
        painted = paint_field_colors(load_ground(mesh_path), scene.ground.terrain_m, scene.ground.plot_m)
        _write_ground_mesh(painted, mesh_path)
        log_line(scene.id, scene.seed, "ground", material="field_colors", reason="no vertex colors and no texture image")

    original = np.asarray(load_ground(mesh_path).vertices)
    reloaded = np.asarray(load_ground(mesh_path).vertices)
    if not same_ground_geometry(original, reloaded):
        raise RuntimeError("ground mesh changed between loads")
    if used_analytic:
        analytic = np.asarray(_analytic_ground(scene).vertices)
        if not same_ground_geometry(original, analytic):
            raise RuntimeError("terrain vertices changed across builds")
    _render_ground_compare(scene, mesh_path, out, "ground_kept", soil=not _ground_has_bake(mesh_path))
    _render_ground_pair(scene, mesh_path, out)
    _log_ground_compare(scene, discarded, mesh_path)
    log_line(scene.id, scene.seed, "ground", output=str(mesh_path), faces=report.faces, bbox_m=report.bbox_m)
    return mesh_path


def _selected(scene: Scene, only: list[str] | None):
    return [asset for asset in scene.assets if only is None or asset.id in only]


def _asset_key(asset) -> str:
    digest = ""
    if asset.image_path and Path(asset.image_path).is_file():
        digest = file_sha256(Path(asset.image_path))
    return cache_key(asset.prompt, asset.size_m, image_sha256=digest)


def _drop_cache(assets) -> None:
    for asset in assets:
        cached = cache_path(_asset_key(asset))
        cached.unlink(missing_ok=True)
        cached.with_suffix(".raw.glb").unlink(missing_ok=True)


def run_images(
    scene: Scene,
    scene_file: Path,
    only: list[str] | None,
    force: bool = False,
    search=None,
    download=None,
    photo_judge=None,
    plant_marker=None,
) -> None:
    """Save one accepted photo per asset. A field photo is cropped to one plant and kept."""
    from farmgen.web_photo import PHOTO_TRIES, crop_box, take_accepted_photo

    selected = _selected(scene, only)
    if force:
        _drop_cache(selected)
    flux: list = []
    for asset in selected:
        dest = output_dir(scene.id) / "images" / f"{asset.id}.png"
        if dest.is_file() and asset.image_path and not force:
            asset.image_path = str(dest)
            continue
        exclude = tuple(asset.seen_photo_urls) if force else ()
        field = asset.field_m is not None

        def _judge(path: Path, attempt: int, asset=asset):
            decision = photo_judge(asset.id, asset.prompt, path, attempt)
            _record_photo(scene, asset.id, attempt, path, decision)
            return decision

        path, url = take_accepted_photo(
            asset.prompt,
            scene.seed,
            asset.id,
            dest,
            judge=None if field or photo_judge is None else _judge,
            exclude=exclude,
            search=search,
            download=download,
            attempts=PHOTO_TRIES,
            seen=asset.seen_photo_urls,
            gate_dir=output_dir(scene.id) / "photo_gate",
        )
        if path is None:
            log_line(scene.id, scene.seed, "image", asset_id=asset.id, photo="miss", error="rejected")
            flux.append(asset)
            continue
        if field:
            path = _crop_field(scene, asset, path, plant_marker)
        asset.image_path = str(path)
        asset.photo_url = url
        if url and url not in asset.seen_photo_urls:
            asset.seen_photo_urls.append(url)
        log_line(
            scene.id,
            scene.seed,
            "image",
            asset_id=asset.id,
            photo="plant" if field else "web",
            url=url,
            output=str(path),
        )
    if flux:
        _generate_images(scene, flux)
        if photo_judge is not None:
            for asset in flux:
                photo = Path(asset.image_path) if asset.image_path else None
                if photo is None or not photo.is_file():
                    continue
                if asset.field_m is not None:
                    cropped = _crop_field(scene, asset, photo, plant_marker)
                    asset.image_path = str(cropped)
                    log_line(scene.id, scene.seed, "image", asset_id=asset.id, photo="plant", output=str(cropped))
                    continue
                decision = photo_judge(asset.id, asset.prompt, photo, 0)
                _record_photo(scene, asset.id, 0, photo, decision)
                if not _decision_ok(decision):
                    asset.image_path = None
                    log_line(
                        scene.id,
                        scene.seed,
                        "image",
                        asset_id=asset.id,
                        photo="flux_rejected",
                        reason=getattr(decision, "reason", ""),
                    )
                    continue
                log_line(scene.id, scene.seed, "image", asset_id=asset.id, photo="flux", output=str(photo))
    save_scene(scene, scene_file)


def _crop_field(scene: Scene, asset, source: Path, plant_marker) -> Path:
    """Keep one plant from a field photo. A rejected box still leaves a crop on disk."""
    import shutil

    from farmgen.web_photo import crop_box

    gate = output_dir(scene.id) / "photo_gate"
    gate.mkdir(parents=True, exist_ok=True)
    full = gate / f"{asset.id}_field.png"
    if source.resolve() != full.resolve():
        shutil.copy2(source, full)
    box = None
    if plant_marker is not None:
        try:
            box = plant_marker(asset.id, asset.prompt, full)
        except (OSError, RuntimeError, ValueError) as exc:
            log_line(scene.id, scene.seed, "image", asset_id=asset.id, photo="plant_box", error=str(exc))
            box = None
    cropped = crop_box(full, source, box)
    log_line(
        scene.id,
        scene.seed,
        "image",
        asset_id=asset.id,
        photo="crop",
        box="" if box is None else ",".join(f"{value:.3f}" for value in box),
        output=str(cropped),
    )
    return cropped


def _decision_ok(decision) -> bool:
    if isinstance(decision, bool):
        return decision
    return bool(getattr(decision, "accept", False))


def _record_photo(scene: Scene, asset_id: str, attempt: int, path: Path, decision) -> None:
    import json

    accept = _decision_ok(decision)
    reason = "" if isinstance(decision, bool) else str(getattr(decision, "reason", ""))
    reply = "" if isinstance(decision, bool) else str(getattr(decision, "reply", ""))
    name = f"{asset_id}_flux" if attempt == 0 else f"{asset_id}_attempt_{attempt}"
    record = output_dir(scene.id) / "photo_gate" / f"{name}.json"
    record.parent.mkdir(parents=True, exist_ok=True)
    record.write_text(
        json.dumps({"id": asset_id, "accept": accept, "reason": reason, "reply": reply}, indent=2) + "\n",
        encoding="utf-8",
    )
    log_line(
        scene.id,
        scene.seed,
        "image",
        asset_id=asset_id,
        photo="gate",
        attempt=attempt,
        accept=int(accept),
        reason=reason,
    )


def run_meshes(scene: Scene, scene_file: Path, only: list[str] | None, force: bool = False, photo_judge=None) -> None:
    selected = _selected(scene, only)
    if force:
        _drop_cache(selected)
    missing = []
    for asset in selected:
        key = _asset_key(asset)
        cached = cache_path(key)
        reason = flat_card_reason(cached) if cached.is_file() else None
        if reason:
            cached.unlink()
            scaled = output_dir(scene.id) / "assets" / f"{asset.id}.glb"
            scaled.unlink(missing_ok=True)
            asset.mesh_path = None
            log_line(scene.id, scene.seed, "asset", asset_id=asset.id, note="flat_card", reason=reason)
        if not cache_hit(key):
            missing.append(asset)
    if missing:
        _generate_meshes(scene, missing, photo_judge=photo_judge)
    save_scene(scene, scene_file)


def run_scale(scene: Scene, scene_file: Path, only: list[str] | None) -> None:
    selected = _selected(scene, only)
    failed: list[str] = []
    for asset in selected:
        key = _asset_key(asset)
        path = cache_path(key)
        if not path.is_file():
            failed.append(asset.id)
            log_line(scene.id, scene.seed, "asset", asset_id=asset.id, error="missing cache")
            continue
        scaled = output_dir(scene.id) / "assets" / f"{asset.id}.glb"
        if _scaled_is_fresh(asset, scaled, path):
            asset.cache_key = key
            asset.mesh_path = str(scaled)
            log_line(scene.id, scene.seed, "asset", asset_id=asset.id, output=str(scaled), note="scaled_fresh")
            continue
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


def run_assets(scene: Scene, scene_file: Path, only: list[str] | None, force: bool = False) -> None:
    run_images(scene, scene_file, only, force=force)
    run_meshes(scene, scene_file, only)
    run_scale(scene, scene_file, only)


def render_mesh_thumbs(scene: Scene, only: list[str] | None = None) -> None:
    """One picture of the cached mesh before scale, so a raw TRELLIS mesh can be judged."""
    blender = _find_blender()
    out = output_dir(scene.id) / "mesh_thumbs"
    if blender is None:
        log_line(scene.id, scene.seed, "mesh_thumb", render="skipped", reason="blender missing")
        return
    for asset in _selected(scene, only):
        source = cache_path(_asset_key(asset))
        if not source.is_file():
            log_line(scene.id, scene.seed, "mesh_thumb", asset_id=asset.id, render="skipped", reason="no cache")
            continue
        dest = out / f"{asset.id}.png"
        _blender(
            blender,
            ["--job", "thumbnail", "--mesh", str(source), "--output", str(dest), "--asset-id", asset.id],
        )


def render_place_preview(scene: Scene) -> Path | None:
    """Orthographic top of the placed scene, before the four corner renders."""
    out = output_dir(scene.id)
    ground = out / "ground.obj"
    export_scene(scene, ground, out)
    dest = out / "place_top.png"
    blender = _find_blender()
    if blender is None:
        log_line(scene.id, scene.seed, "place_preview", render="skipped", reason="blender missing")
        return None
    _blender(
        blender,
        [
            "--job",
            "top",
            "--mesh",
            str(out / "farm.glb"),
            "--output",
            str(dest),
            "--size",
            str(scene.ground.terrain_m[0]),
            str(scene.ground.terrain_m[1]),
            "--soil",
            "0" if _ground_has_bake(ground) else "1",
        ],
    )
    return dest


def restamp_heights(scene: Scene, scene_file: Path) -> None:
    _stamp_heights(scene)
    save_scene(scene, scene_file)


def run_place(scene: Scene, scene_file: Path, keep: set[str] | None = None, lattice: bool = True) -> None:
    place_scene(scene, keep=keep, lattice=lattice)
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
            "--plot",
            str(scene.ground.plot_m[0]),
            str(scene.ground.plot_m[1]),
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
    mesh = paint_field_colors(_analytic_ground(scene), scene.ground.terrain_m, scene.ground.plot_m)
    _write_ground_mesh(mesh, mesh_path)


def _write_ground_mesh(mesh, mesh_path: Path) -> None:
    mesh_path.parent.mkdir(parents=True, exist_ok=True)
    mesh.export(mesh_path)
    colors = mesh_path.with_suffix(".colors.npy")
    if mesh_has_baked_color(mesh):
        np.save(colors, np.asarray(mesh.visual.vertex_colors))
    elif colors.is_file():
        colors.unlink()


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


def _log_ground_compare(scene: Scene, discarded: Path, kept: Path) -> None:
    if not discarded.is_file() or not kept.is_file():
        return
    discarded_extent, discarded_faces = _mesh_extent(discarded)
    kept_extent, kept_faces = _mesh_extent(kept)
    log_line(
        scene.id,
        scene.seed,
        "ground_compare",
        discarded_m=_extent_text(discarded_extent),
        kept_m=_extent_text(kept_extent),
        discarded_faces=discarded_faces,
        kept_faces=kept_faces,
    )


def _mesh_extent(path: Path) -> tuple[tuple[float, float, float], int]:
    mesh = load_ground(path)
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    extent = vertices.max(axis=0) - vertices.min(axis=0)
    return (float(extent[0]), float(extent[1]), float(extent[2])), int(len(mesh.faces))


def _extent_text(extent: tuple[float, float, float]) -> str:
    return f"{extent[0]:.2f},{extent[1]:.2f},{extent[2]:.2f}"


def render_saved_ground_pair(scene: Scene) -> None:
    """Re-render the discarded and kept objs already on disk. Does not run Infinigen."""
    out = output_dir(scene.id)
    discarded = out / "ground_discarded.obj"
    kept = out / "ground.obj"
    if discarded.is_file():
        _render_ground_compare(scene, discarded, out, "ground_discarded", soil=False, frame="bbox")
    if kept.is_file():
        _render_ground_compare(scene, kept, out, "ground_kept", soil=not _ground_has_bake(kept), frame="terrain")
    _log_ground_compare(scene, discarded, kept)


def _render_ground_compare(
    scene: Scene,
    mesh_path: Path,
    out: Path,
    prefix: str,
    soil: bool,
    frame: str = "terrain",
) -> None:
    blender = _find_blender()
    if blender is None or not mesh_path.is_file():
        log_line(scene.id, scene.seed, "ground_render", render="skipped", prefix=prefix, reason="mesh or blender missing")
        return
    _blender(
        blender,
        [
            "--job",
            "pair",
            "--mesh",
            str(mesh_path),
            "--output",
            str(out),
            "--prefix",
            prefix,
            "--size",
            str(scene.ground.terrain_m[0]),
            str(scene.ground.terrain_m[1]),
            "--soil",
            "1" if soil else "0",
            "--seed",
            str(scene.seed),
            "--frame",
            frame,
        ],
    )


def _ground_has_bake(mesh_path: Path) -> bool:
    if mesh_path.with_suffix(".colors.npy").is_file():
        return True
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


def _generate_meshes(scene: Scene, assets, photo_judge=None) -> None:
    _wait_gpu(scene, TRELLIS_BUDGET_MIB)
    python = trellis_dir() / ".venv" / "bin" / "python"
    if not python.is_file():
        raise RuntimeError(f"TRELLIS venv missing at {python}")
    before = gpu_memory_mib()
    log_line(scene.id, scene.seed, "gpu", when="before_trellis", memory_mib=before)
    worker = repo_root() / "farmgen" / "workers" / "trellis_generate.py"
    failures: list[str] = []
    flux_retry: list = []
    for asset in assets:
        if not asset.image_path:
            failures.append(asset.id)
            log_line(scene.id, scene.seed, "asset", asset_id=asset.id, error="missing image_path")
            continue
        status = _run_trellis(scene, asset, python, worker)
        if status == "ok":
            continue
        if asset.photo_url:
            log_line(scene.id, scene.seed, "asset", asset_id=asset.id, note=status, photo="web")
            asset.image_path = None
            asset.photo_url = None
            flux_retry.append(asset)
            continue
        failures.append(asset.id)
    if flux_retry:
        _generate_images(scene, flux_retry)
        for asset in flux_retry:
            photo = Path(asset.image_path) if asset.image_path else None
            if photo_judge is not None and photo is not None and photo.is_file():
                decision = photo_judge(asset.id, asset.prompt, photo, 0)
                _record_photo(scene, asset.id, 0, photo, decision)
                if not _decision_ok(decision):
                    asset.image_path = None
                    failures.append(asset.id)
                    log_line(scene.id, scene.seed, "image", asset_id=asset.id, photo="flux_rejected")
                    continue
            if not asset.image_path or _run_trellis(scene, asset, python, worker) != "ok":
                failures.append(asset.id)
    if failures:
        log_line(scene.id, scene.seed, "asset", failed=",".join(failures))
    after = gpu_memory_mib()
    log_line(scene.id, scene.seed, "gpu", when="after_trellis", memory_mib=after)


def _run_trellis(scene: Scene, asset, python: Path, worker: Path) -> str:
    """Build one GLB. Return ``ok``, ``empty_foreground``, or ``flat_card``."""
    key = _asset_key(asset)
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
        log_line(
            scene.id,
            scene.seed,
            "asset",
            asset_id=asset.id,
            error=f"trellis returncode={result.returncode}",
        )
        return "empty_foreground"
    reason = flat_card_reason(raw)
    if reason:
        raw.unlink()
        log_line(scene.id, scene.seed, "asset", asset_id=asset.id, note="flat_card", reason=reason)
        return "flat_card"
    raw.replace(dest)
    return "ok"


def _vertex_colors(mesh, vertex_count: int) -> np.ndarray | None:
    """Sample a texture onto vertices. A missing image leaves this empty."""
    visual = getattr(mesh, "visual", None)
    if visual is None or not hasattr(visual, "to_color"):
        return None
    try:
        baked = visual.to_color()
    except Exception:
        return None
    colors = np.asarray(getattr(baked, "vertex_colors", []))
    if colors.ndim != 2 or len(colors) != vertex_count:
        return None
    return colors


def _scaled_is_fresh(asset, scaled: Path, source: Path) -> bool:
    """Reuse a scaled mesh only when it is newer than the cache and under the copy face cap."""
    if not scaled.is_file() or scaled.stat().st_mtime < source.stat().st_mtime:
        return False
    if asset.count < 4:
        return True
    try:
        loaded = trimesh.load(scaled, force="mesh")
    except (OSError, ValueError):
        return False
    return len(getattr(loaded, "faces", [])) <= COPY_FACE_CAP


def _cluster_mesh(mesh, colors: np.ndarray | None, pitch: float):
    """Merge vertices that fall in the same ``pitch`` cell and average their colors."""
    verts = np.asarray(mesh.vertices, dtype=np.float64)
    origin = verts.min(axis=0)
    keys = np.floor((verts - origin) / pitch).astype(np.int64)
    _, inverse = np.unique(keys, axis=0, return_inverse=True)
    counts = np.bincount(inverse).astype(np.float64)
    sums = np.zeros((len(counts), 3), dtype=np.float64)
    np.add.at(sums, inverse, verts)
    faces = inverse[np.asarray(mesh.faces)]
    keep = (faces[:, 0] != faces[:, 1]) & (faces[:, 1] != faces[:, 2]) & (faces[:, 0] != faces[:, 2])
    out = trimesh.Trimesh(vertices=sums / counts[:, None], faces=faces[keep], process=False)
    if colors is not None and len(colors) == len(verts):
        summed = np.zeros((len(counts), colors.shape[1]), dtype=np.float64)
        np.add.at(summed, inverse, np.asarray(colors, dtype=np.float64))
        out.visual.vertex_colors = np.clip(np.rint(summed / counts[:, None]), 0, 255).astype(np.uint8)
    return out


def _cluster_to_cap(mesh, colors: np.ndarray | None, face_cap: int):
    """Pick the finest cell size whose clustered mesh stays at or under ``face_cap``."""
    extent = float(np.max(np.ptp(np.asarray(mesh.vertices), axis=0)))
    if extent <= 0 or face_cap < 1:
        raise RuntimeError(f"cluster cannot reach {face_cap} faces")
    lo = extent / 500.0
    hi = extent / 2.0
    best = None
    while hi < extent * 8:
        coarse = _cluster_mesh(mesh, colors, hi)
        if len(coarse.faces) <= face_cap:
            best = coarse
            break
        hi *= 2
    if best is None:
        raise RuntimeError(f"cluster stopped above {face_cap} faces")
    for _ in range(14):
        pitch = (lo + hi) / 2.0
        trial = _cluster_mesh(mesh, colors, pitch)
        if len(trial.faces) > face_cap:
            lo = pitch
        else:
            best = trial
            hi = pitch
    return best


def _decimate_copy(mesh, colors: np.ndarray | None, face_cap: int):
    """Drop repeated TRELLIS meshes under ``face_cap`` and keep a color per vertex."""
    try:
        reduced = mesh.simplify_quadric_decimation(face_count=face_cap, aggression=10)
    except (TypeError, ValueError) as exc:
        raise RuntimeError(f"decimate to {face_cap} faces failed: {exc}") from exc
    mapped = None
    if colors is not None and len(colors) == len(mesh.vertices):
        mapped = _transfer_colors(np.asarray(mesh.vertices), colors, np.asarray(reduced.vertices))
        reduced.visual.vertex_colors = mapped
    if len(reduced.faces) <= face_cap:
        return reduced
    return _cluster_to_cap(reduced, mapped, face_cap)


def _transfer_colors(src: np.ndarray, colors: np.ndarray, dst: np.ndarray) -> np.ndarray:
    cell = 0.05
    origin = src.min(axis=0)
    src_keys = np.floor((src - origin) / cell).astype(np.int64)
    buckets: dict[tuple[int, int, int], int] = {}
    for index, key in enumerate(src_keys):
        token = (int(key[0]), int(key[1]), int(key[2]))
        buckets.setdefault(token, index)
    out = np.empty((len(dst), colors.shape[1]), dtype=colors.dtype)
    dst_keys = np.floor((dst - origin) / cell).astype(np.int64)
    for index, key in enumerate(dst_keys):
        found = None
        base = (int(key[0]), int(key[1]), int(key[2]))
        for dz in (0, -1, 1):
            for dy in (0, -1, 1):
                for dx in (0, -1, 1):
                    found = buckets.get((base[0] + dx, base[1] + dy, base[2] + dz))
                    if found is not None:
                        break
                if found is not None:
                    break
            if found is not None:
                break
        out[index] = colors[0 if found is None else found]
    return out


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
    colors = _vertex_colors(mesh, len(scaled))
    if colors is not None and did_seal:
        extra = np.repeat(colors[:1], len(sealed) - len(colors), axis=0)
        colors = np.vstack([colors, extra])
    if asset.count >= 4 and len(out.faces) > COPY_FACE_CAP:
        out = _decimate_copy(out, colors, COPY_FACE_CAP)
    elif colors is not None and len(colors) == len(out.vertices):
        out.visual.vertex_colors = colors
    elif mesh.visual is not None and len(getattr(mesh.vertices, "__len__", lambda: 0)()) == len(out.vertices):
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
        faces=len(out.faces),
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
        source = cache_path(_asset_key(asset))
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
    found: list[Path] = []
    for pattern in ("*.obj", "*.glb", "*.usdc", "*.usd"):
        found.extend(folder.rglob(pattern))
    found = [path for path in found if "atmosphere" not in path.name.lower() and not path.name.endswith("_part.obj")]
    if not found:
        return None
    return max(found, key=lambda path: path.stat().st_size)


def _debug_dump(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
