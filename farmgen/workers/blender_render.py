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
    parser.add_argument("--job", required=True, choices=["ground", "thumbnail", "views", "compose"])
    parser.add_argument("--mesh", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--size", type=float, nargs=2, default=[40.0, 40.0])
    parser.add_argument("--soil", type=int, default=1)
    parser.add_argument("--asset-id", default="asset")
    return parser.parse_args(argv)


def main() -> None:
    import bpy

    args = _args()
    _reset()
    _import_mesh(Path(args.mesh))
    if args.job == "ground":
        if args.soil:
            _soil_material(args.seed)
        _render_ground_views(Path(args.output), args.seed, tuple(args.size))
    elif args.job == "thumbnail":
        _gray_plane()
        _frame_selection()
        _render_still(Path(args.output))
    elif args.job == "views":
        _gray_plane()
        _render_asset_views(Path(args.output), args.asset_id)
    elif args.job == "compose":
        if args.soil:
            _soil_on_ground(args.seed)
        _render_compose(Path(args.output), tuple(args.size))
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


def _render_compose(directory: Path, size) -> None:
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
    # Inside the plot, looking south at the fence line, with the barn beside it.
    _camera((14.0, 8.0, 2.4), (18.0, 1.1, 0.8))
    _render_still(directory / "compose_eye_a.png")
    # Looking at the hay grid, with the silo and tractor in the same view.
    _camera((22.0, 24.0, 2.5), (29.0, 30.0, 1.0))
    _render_still(directory / "compose_eye_b.png")
    # compose.png is the overhead proof: barn, hay, and the fence are separate.
    top = directory / "compose_top.png"
    proof = directory / "compose.png"
    proof.write_bytes(top.read_bytes())
    views = _load_views()
    for location, target, index in views.corner_locations(tuple(size)):
        _camera(location, target)
        _render_still(directory / views.compose_view_filename(index))


if __name__ == "__main__":
    main()
