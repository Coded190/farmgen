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
        colors = ground_path.with_suffix(".colors.npy")
        if colors.is_file():
            # A textured OBJ keeps TextureVisuals if vertex_colors is assigned
            # in place, and the GLB then drops the paint.
            ground.visual = trimesh.visual.ColorVisuals(mesh=ground, vertex_colors=np.load(colors))
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
        if not asset.instances:
            log_line(scene.id, scene.seed, "compose", id=asset.id, note="unplaced")
            continue
        if not asset.mesh_path:
            log_line(scene.id, scene.seed, "compose", id=asset.id, note="no mesh")
            continue
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
        _add_mesh(stage, _usd_name(node_name), points, np.asarray(geom.faces), _vertex_rgb(geom))
    stage.GetRootLayer().Save()
    return path


def _vertex_rgb(geom) -> np.ndarray | None:
    """RGB in 0..1, one color per vertex, when the mesh carries vertex colors."""
    visual = getattr(geom, "visual", None)
    raw = getattr(visual, "vertex_colors", None)
    if raw is None:
        return None
    colors = np.asarray(raw)
    if colors.ndim != 2 or len(colors) != len(geom.vertices) or colors.shape[1] < 3:
        return None
    return np.asarray(colors[:, :3], dtype=np.float64) / 255.0


def _add_mesh(stage, name: str, points: np.ndarray, faces: np.ndarray, colors: np.ndarray | None) -> None:
    from pxr import Gf, Sdf, UsdGeom, UsdShade, Vt

    mesh = UsdGeom.Mesh.Define(stage, f"/World/{name}")
    mesh.CreatePointsAttr([Gf.Vec3f(float(p[0]), float(p[1]), float(p[2])) for p in points])
    mesh.CreateFaceVertexCountsAttr(Vt.IntArray([3] * len(faces)))
    mesh.CreateFaceVertexIndicesAttr(Vt.IntArray(faces.reshape(-1).tolist()))
    mesh.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
    mesh.CreateDoubleSidedAttr(True)
    if colors is not None:
        primvar = UsdGeom.PrimvarsAPI(mesh.GetPrim()).CreatePrimvar(
            "displayColor",
            Sdf.ValueTypeNames.Color3fArray,
            UsdGeom.Tokens.vertex,
        )
        primvar.Set([Gf.Vec3f(float(c[0]), float(c[1]), float(c[2])) for c in colors])
        mean = colors.mean(axis=0)
    else:
        mean = np.array([0.45, 0.42, 0.38])
    material = UsdShade.Material.Define(stage, f"/World/Looks/{name}")
    shader = UsdShade.Shader.Define(stage, f"/World/Looks/{name}/Preview")
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(
        Gf.Vec3f(float(mean[0]), float(mean[1]), float(mean[2]))
    )
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.8)
    shader.CreateOutput("surface", Sdf.ValueTypeNames.Token)
    material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    UsdShade.MaterialBindingAPI(mesh.GetPrim()).Bind(material)


def _usd_name(name: str) -> str:
    cleaned = "".join(ch if ch.isalnum() or ch == "_" else "_" for ch in name)
    if not cleaned or cleaned[0].isdigit():
        cleaned = f"m_{cleaned}"
    return cleaned
