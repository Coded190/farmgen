"""Blender background renders for ground, thumbnails, and the composed scene.

Launch with the Blender binary Infinigen installs:

    blender --background --python farmgen/workers/blender_render.py -- --job ...
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path


def _args() -> argparse.Namespace:
    argv = sys.argv
    if "--" in argv:
        argv = argv[argv.index("--") + 1 :]
    else:
        argv = argv[1:]
    parser = argparse.ArgumentParser()
    parser.add_argument("--job", required=True, choices=["ground", "thumbnail", "views", "compose", "pair", "top"])
    parser.add_argument("--mesh", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--size", type=float, nargs=2, default=[40.0, 40.0])
    parser.add_argument("--plot", type=float, nargs=2, default=None)
    parser.add_argument("--soil", type=int, default=1)
    parser.add_argument("--asset-id", default="asset")
    parser.add_argument("--prefix", default="ground")
    parser.add_argument("--frame", default="terrain", choices=["terrain", "bbox"])
    return parser.parse_args(argv)


def main() -> None:
    import bpy

    args = _args()
    _reset()
    _import_mesh(Path(args.mesh))
    if args.job == "ground":
        _apply_color_file(Path(args.mesh))
        if args.soil:
            _soil_material(args.seed)
        _render_ground_views(Path(args.output), args.seed, tuple(args.size))
    elif args.job == "pair":
        _apply_color_file(Path(args.mesh))
        if args.soil:
            _soil_material(args.seed)
        _render_pair(Path(args.output), tuple(args.size), args.prefix, args.frame)
    elif args.job == "top":
        _vertex_color_materials()
        if args.soil:
            _soil_on_ground(args.seed)
        _render_top(Path(args.output), tuple(args.size))
    elif args.job == "thumbnail":
        _vertex_color_materials()
        _gray_plane()
        _frame_selection()
        _render_still(Path(args.output))
    elif args.job == "views":
        _vertex_color_materials()
        _gray_plane()
        _render_asset_views(Path(args.output), args.asset_id)
    elif args.job == "compose":
        _vertex_color_materials()
        if args.soil:
            _soil_on_ground(args.seed)
        _render_compose(Path(args.output), tuple(args.size), None if args.plot is None else tuple(args.plot))
    else:
        raise SystemExit(args.job)


def _reset() -> None:
    import bpy

    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene
    scene.render.engine = "CYCLES"
    scene.cycles.device = "CPU"
    scene.cycles.samples = 32
    scene.render.resolution_x = 960
    scene.render.resolution_y = 540
    scene.render.image_settings.file_format = "PNG"
    world = bpy.data.worlds.new("farmgen_world")
    world.use_nodes = True
    world.node_tree.nodes["Background"].inputs[0].default_value = (0.55, 0.65, 0.8, 1.0)
    scene.world = world
    sun_data = bpy.data.lights.new("sun", "SUN")
    sun_data.energy = 4.0
    sun = bpy.data.objects.new("sun", sun_data)
    sun.rotation_euler = (math.radians(50), 0.0, math.radians(30))
    scene.collection.objects.link(sun)


def _import_mesh(path: Path) -> None:
    import bpy

    suffix = path.suffix.lower()
    if suffix == ".obj":
        # FarmGen meshes are Z-up. Blender's OBJ importer defaults to Y-up and
        # would stand the ground plane on edge.
        bpy.ops.wm.obj_import(filepath=str(path), forward_axis="Y", up_axis="Z")
    elif suffix in {".glb", ".gltf"}:
        bpy.ops.import_scene.gltf(filepath=str(path))
        # trimesh writes glTF Y-up. This Blender build leaves that axis in place,
        # so height lands on Y and the ground plane slices the mesh. Rotate
        # -90 degrees about X: (x, y, z) -> (x, z, -y), which puts height on Z.
        from mathutils import Matrix

        rotation = Matrix.Rotation(-math.pi / 2.0, 4, "X")
        for obj in list(bpy.context.scene.objects):
            if obj.type == "MESH":
                obj.matrix_world = rotation @ obj.matrix_world
    else:
        raise SystemExit(f"unsupported mesh {path}")


def _soil_material(seed: int) -> None:
    import bpy

    mat = _make_soil(seed)
    for obj in bpy.context.scene.objects:
        if obj.type == "MESH":
            obj.data.materials.clear()
            obj.data.materials.append(mat)


def _make_soil(seed: int):
    import bpy

    mat = bpy.data.materials.new("tilled_soil")
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    nodes.clear()
    out = nodes.new("ShaderNodeOutputMaterial")
    bsdf = nodes.new("ShaderNodeBsdfPrincipled")
    noise = nodes.new("ShaderNodeTexNoise")
    ramp = nodes.new("ShaderNodeValToRGB")
    # 4D noise exposes the W input. The seed changes only this shader, so the
    # exported vertices stay identical across soil seeds.
    noise.noise_dimensions = "4D"
    noise.inputs["Scale"].default_value = 12.0
    noise.inputs["Detail"].default_value = 8.0
    noise.inputs["Distortion"].default_value = 0.2 + (seed % 7) * 0.05
    noise.inputs["W"].default_value = float(seed)
    ramp.color_ramp.elements[0].position = 0.35
    ramp.color_ramp.elements[0].color = (0.28, 0.16, 0.08, 1.0)
    ramp.color_ramp.elements[1].position = 0.7
    ramp.color_ramp.elements[1].color = (0.45, 0.28, 0.12, 1.0)
    links.new(noise.outputs["Fac"], ramp.inputs["Fac"])
    links.new(ramp.outputs["Color"], bsdf.inputs["Base Color"])
    links.new(bsdf.outputs["BSDF"], out.inputs["Surface"])
    bsdf.inputs["Roughness"].default_value = 0.9
    return mat


def _gray_plane() -> None:
    import bpy

    bpy.ops.mesh.primitive_plane_add(size=20.0, location=(0.0, 0.0, 0.0))
    plane = bpy.context.active_object
    mat = bpy.data.materials.new("gray")
    mat.use_nodes = True
    mat.node_tree.nodes["Principled BSDF"].inputs["Base Color"].default_value = (0.6, 0.6, 0.6, 1.0)
    plane.data.materials.append(mat)


def _camera(location, target) -> None:
    import bpy

    data = bpy.data.cameras.new("cam")
    obj = bpy.data.objects.new("cam", data)
    bpy.context.scene.collection.objects.link(obj)
    obj.location = location
    direction = (
        target[0] - location[0],
        target[1] - location[1],
        target[2] - location[2],
    )
    rot = _look_at(direction)
    obj.rotation_euler = rot
    bpy.context.scene.camera = obj


def _look_at(direction):
    import mathutils

    quat = mathutils.Vector(direction).to_track_quat("-Z", "Y")
    return quat.to_euler()


def _render_still(path: Path) -> None:
    import bpy

    path.parent.mkdir(parents=True, exist_ok=True)
    bpy.context.scene.render.filepath = str(path)
    bpy.ops.render.render(write_still=True)


def _selection_center() -> tuple[tuple[float, float, float], float]:
    import bpy
    from mathutils import Vector

    mins = Vector((1e9, 1e9, 1e9))
    maxs = Vector((-1e9, -1e9, -1e9))
    for obj in bpy.context.scene.objects:
        if obj.type != "MESH" or obj.name == "Plane":
            continue
        for corner in obj.bound_box:
            world = obj.matrix_world @ Vector(corner)
            mins = Vector((min(mins.x, world.x), min(mins.y, world.y), min(mins.z, world.z)))
            maxs = Vector((max(maxs.x, world.x), max(maxs.y, world.y), max(maxs.z, world.z)))
    center = (mins + maxs) * 0.5
    span = max(maxs.x - mins.x, maxs.y - mins.y, maxs.z - mins.z, 1.0)
    return (center.x, center.y, center.z), float(span)


def _frame_selection() -> None:
    center, span = _selection_center()
    _camera(
        (center[0] + span * 0.9, center[1] - span * 1.6, center[2] + span * 0.7),
        center,
    )


def _load_views():
    import importlib.util

    path = Path(__file__).resolve().parents[1] / "views.py"
    spec = importlib.util.spec_from_file_location("farmgen_views", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _vertex_color_materials() -> None:
    """Shade glTF COLOR_0. Trimesh writes the attribute and no material, so Blender otherwise stays white."""
    import bpy

    for obj in bpy.context.scene.objects:
        if obj.type != "MESH" or len(obj.data.color_attributes) == 0:
            continue
        mat = bpy.data.materials.new(f"vertex_color_{obj.name}")
        mat.use_nodes = True
        nodes = mat.node_tree.nodes
        links = mat.node_tree.links
        nodes.clear()
        out = nodes.new("ShaderNodeOutputMaterial")
        bsdf = nodes.new("ShaderNodeBsdfPrincipled")
        color = nodes.new("ShaderNodeVertexColor")
        color.layer_name = obj.data.color_attributes[0].name
        links.new(color.outputs["Color"], bsdf.inputs["Base Color"])
        links.new(bsdf.outputs["BSDF"], out.inputs["Surface"])
        bsdf.inputs["Roughness"].default_value = 0.9
        obj.data.materials.clear()
        obj.data.materials.append(mat)


def _apply_color_file(mesh_path: Path) -> None:
    import bpy
    import numpy as np

    colors_path = mesh_path.with_suffix(".colors.npy")
    if not colors_path.is_file():
        return
    colors = np.load(colors_path)
    if colors.ndim != 2 or colors.shape[1] < 3:
        return
    rgb = colors[:, :4] if colors.shape[1] >= 4 else np.concatenate([colors[:, :3], np.full((len(colors), 1), 255)], axis=1)
    rgb = rgb.astype(np.float32)
    if rgb.max() > 1.0:
        rgb /= 255.0
    for obj in bpy.context.scene.objects:
        if obj.type != "MESH" or len(obj.data.vertices) != len(rgb):
            continue
        attr = obj.data.color_attributes.new(name="Col", type="BYTE_COLOR", domain="POINT")
        attr.data.foreach_set("color", rgb.reshape(-1))
        mat = bpy.data.materials.new("baked_ground")
        mat.use_nodes = True
        nodes = mat.node_tree.nodes
        links = mat.node_tree.links
        nodes.clear()
        out = nodes.new("ShaderNodeOutputMaterial")
        bsdf = nodes.new("ShaderNodeBsdfPrincipled")
        color = nodes.new("ShaderNodeVertexColor")
        color.layer_name = "Col"
        links.new(color.outputs["Color"], bsdf.inputs["Base Color"])
        links.new(bsdf.outputs["BSDF"], out.inputs["Surface"])
        bsdf.inputs["Roughness"].default_value = 0.9
        obj.data.materials.clear()
        obj.data.materials.append(mat)


def _render_top(path: Path, size) -> None:
    import bpy

    cx, cy = size[0] / 2.0, size[1] / 2.0
    _camera((cx, cy, max(size) * 1.2), (cx, cy, 0.0))
    cam = bpy.context.scene.camera
    cam.data.type = "ORTHO"
    views = _load_views()
    cam.data.ortho_scale = views.ortho_scale(tuple(size))
    _render_still(path)


def _render_pair(directory: Path, size, prefix: str, frame: str = "terrain") -> None:
    import bpy

    directory.mkdir(parents=True, exist_ok=True)
    if frame == "bbox":
        _render_sheet(directory, prefix)
        return
    cx, cy = size[0] / 2.0, size[1] / 2.0
    _camera((cx, cy, max(size) * 1.2), (cx, cy, 0.0))
    cam = bpy.context.scene.camera
    cam.data.type = "ORTHO"
    views = _load_views()
    cam.data.ortho_scale = views.ortho_scale(tuple(size))
    _render_still(directory / f"{prefix}_top.png")
    location, target, _index = views.corner_locations(tuple(size))[0]
    _camera(location, target)
    _render_still(directory / f"{prefix}_view_0.png")


def _render_sheet(directory: Path, prefix: str) -> None:
    """Frame the mesh's own bbox so a vertical sheet shows its face."""
    import bpy
    from mathutils import Vector

    mins = Vector((1e9, 1e9, 1e9))
    maxs = Vector((-1e9, -1e9, -1e9))
    for obj in bpy.context.scene.objects:
        if obj.type != "MESH":
            continue
        for corner in obj.bound_box:
            world = obj.matrix_world @ Vector(corner)
            mins = Vector((min(mins.x, world.x), min(mins.y, world.y), min(mins.z, world.z)))
            maxs = Vector((max(maxs.x, world.x), max(maxs.y, world.y), max(maxs.z, world.z)))
    views = _load_views()
    frame = views.sheet_cameras((mins.x, mins.y, mins.z), (maxs.x, maxs.y, maxs.z))
    _camera_axis(frame["top_location"], frame["top_target"], str(frame["up_axis"]))
    cam = bpy.context.scene.camera
    cam.data.type = "ORTHO"
    cam.data.ortho_scale = float(frame["ortho_scale"])
    _render_still(directory / f"{prefix}_top.png")
    _camera_axis(frame["corner_location"], frame["corner_target"], str(frame["up_axis"]))
    _render_still(directory / f"{prefix}_view_0.png")


def _camera_axis(location, target, up_axis: str) -> None:
    import bpy
    import mathutils

    data = bpy.data.cameras.new("cam")
    obj = bpy.data.objects.new("cam", data)
    bpy.context.scene.collection.objects.link(obj)
    obj.location = location
    direction = (
        target[0] - location[0],
        target[1] - location[1],
        target[2] - location[2],
    )
    quat = mathutils.Vector(direction).to_track_quat("-Z", up_axis)
    obj.rotation_euler = quat.to_euler()
    bpy.context.scene.camera = obj


def _render_ground_views(directory: Path, seed: int, size) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    cx, cy = size[0] / 2.0, size[1] / 2.0
    _camera((cx, cy, max(size) * 1.2), (cx, cy, 0.0))
    _render_still(directory / f"ground_top_seed{seed}.png")
    _camera((cx, 1.5, 1.5), (cx, cy, 0.5))
    _render_still(directory / f"ground_eye_seed{seed}.png")
    views = _load_views()
    for location, target, index in views.corner_locations(tuple(size)):
        _camera(location, target)
        _render_still(directory / views.compose_view_filename(index))


def _render_asset_views(directory: Path, asset_id: str) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    center, radius = _selection_center()
    views = _load_views()
    for location, target, yaw in views.orbit_locations(center, max(radius * 2.4, 1.0)):
        _camera(location, target)
        _render_still(directory / views.asset_view_filename(asset_id, yaw))


def _soil_on_ground(seed: int) -> None:
    import bpy

    soil = _make_soil(seed)
    for obj in bpy.context.scene.objects:
        if obj.type == "MESH" and obj.name == "ground":
            obj.data.materials.clear()
            obj.data.materials.append(soil)


def _render_compose(directory: Path, size, plot=None) -> None:
    import bpy

    directory.mkdir(parents=True, exist_ok=True)
    cx, cy = size[0] / 2.0, size[1] / 2.0
    # Orthographic top view. A 50 mm lens at 1.2x the plot size only covers
    # about half of a 40 m field, so the fence and hay grid were out of frame.
    _camera((cx, cy, max(size) * 1.2), (cx, cy, 0.0))
    cam = bpy.context.scene.camera
    cam.data.type = "ORTHO"
    # ortho_scale is the wider frame axis. The image is 16:9, so the scale has
    # to grow by width/height before a 40 m plot fits vertically. The fence
    # line sits at y=1 m and was outside a square-ish frame.
    aspect = bpy.context.scene.render.resolution_x / bpy.context.scene.render.resolution_y
    cam.data.ortho_scale = max(size) * aspect * 1.08
    _render_still(directory / "compose_top.png")
    views = _load_views()
    plot_m = tuple(plot) if plot is not None else (min(40.0, float(size[0])), min(40.0, float(size[1])))
    for location, target, name in views.eye_locations(tuple(size), plot_m):
        _camera(location, target)
        _render_still(directory / f"compose_{name}.png")
    # compose.png is the overhead proof: barn, hay, and the fence are separate.
    top = directory / "compose_top.png"
    proof = directory / "compose.png"
    proof.write_bytes(top.read_bytes())
    for location, target, index in views.corner_locations(tuple(size)):
        _camera(location, target)
        _render_still(directory / views.compose_view_filename(index))


if __name__ == "__main__":
    main()
