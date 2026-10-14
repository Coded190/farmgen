"""Assemble placed meshes into one GLB. USD is written when ``pxr`` imports."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import trimesh

from farmgen.ground import load_ground, plot_origin, sample_height, terrain_mesh
from farmgen.log import log_line
from farmgen.scene import Scene


def build_scene(scene: Scene, ground_path: Path) -> trimesh.Scene:
    assembled = trimesh.Scene()
    if ground_path.is_file():
        ground = load_ground(ground_path)
    else:
        ground = terrain_mesh(
            scene.ground.terrain_m,
            scene.ground.plot_m,
            scene.ground.relief_m,
            scene.ground.surround_relief_m,
            seed=scene.seed,
        )
    assembled.add_geometry(ground, node_name="ground", geom_name="ground")
    origin_x, origin_y = plot_origin(scene.ground.terrain_m, scene.ground.plot_m)
    for asset in scene.assets:
        if not asset.mesh_path:
            raise ValueError(f"{asset.id} has no mesh_path")
        mesh_path = Path(asset.mesh_path)
        if not mesh_path.is_file():
            raise FileNotFoundError(mesh_path)
        mesh = load_ground(mesh_path)
        for inst in asset.instances:
            transform = np.eye(4)
            yaw = inst.yaw
            c, s = math.cos(yaw), math.sin(yaw)
            transform[:3, :3] = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
            world_xy = (inst.xy[0] + origin_x, inst.xy[1] + origin_y)
            transform[0, 3] = world_xy[0]
            transform[1, 3] = world_xy[1]
            transform[2, 3] = sample_height(ground.vertices, ground.faces, world_xy)
            assembled.add_geometry(
                mesh,
                node_name=f"{asset.id}_{inst.index}",
                geom_name=asset.id,
                transform=transform,
            )
    return assembled


def export_scene(scene: Scene, ground_path: Path, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    assembled = build_scene(scene, ground_path)
    glb_path = out_dir / "farm.glb"
    assembled.export(glb_path)
    usd_path = out_dir / "farm.usdc"
    try:
        write_usd(assembled, usd_path)
    except ImportError:
        log_line(scene.id, scene.seed, "compose", usd="skipped", reason="pxr missing")
        return glb_path
    return usd_path


def usd_stage_metadata() -> dict[str, float | str]:
    return {"upAxis": "Z", "metersPerUnit": 1.0}


def write_usd(assembled: trimesh.Scene, path: Path) -> Path:
    """Meters, Z-up crate. Raises ImportError when usd-core is not installed."""
    from pxr import Usd, UsdGeom

    if path.exists():
        path.unlink()
    stage = Usd.Stage.CreateNew(str(path))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    world = UsdGeom.Xform.Define(stage, "/World")
    stage.SetDefaultPrim(world.GetPrim())
    metadata = usd_stage_metadata()
    if stage.GetMetadata("upAxis") != metadata["upAxis"]:
        raise RuntimeError("usd upAxis is not Z")
    if stage.GetMetadata("metersPerUnit") != metadata["metersPerUnit"]:
        raise RuntimeError("usd metersPerUnit is not 1")
    for node_name in assembled.graph.nodes_geometry:
        transform, geom_name = assembled.graph[node_name]
        geom = assembled.geometry[geom_name]
        points = trimesh.transformations.transform_points(np.asarray(geom.vertices, dtype=np.float64), transform)
        _add_mesh(stage, _usd_name(node_name), points, np.asarray(geom.faces))
    stage.GetRootLayer().Save()
    return path


def _add_mesh(stage, name: str, points: np.ndarray, faces: np.ndarray) -> None:
    from pxr import Gf, UsdGeom, Vt

    mesh = UsdGeom.Mesh.Define(stage, f"/World/{name}")
    mesh.CreatePointsAttr([Gf.Vec3f(float(p[0]), float(p[1]), float(p[2])) for p in points])
    mesh.CreateFaceVertexCountsAttr(Vt.IntArray([3] * len(faces)))
    mesh.CreateFaceVertexIndicesAttr(Vt.IntArray(faces.reshape(-1).tolist()))
    mesh.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
    mesh.CreateDoubleSidedAttr(True)


def _usd_name(name: str) -> str:
    cleaned = "".join(ch if ch.isalnum() or ch == "_" else "_" for ch in name)
    if not cleaned or cleaned[0].isdigit():
        cleaned = f"m_{cleaned}"
    return cleaned
