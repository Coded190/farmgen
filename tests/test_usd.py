"""The composed stage is meters and Z-up when usd-core is installed."""

import json
from pathlib import Path

import numpy as np
import pytest
import trimesh

from farmgen.compose import build_scene, usd_stage_metadata, write_usd
from farmgen.scene import Instance, starter_scene


def test_unplaced_asset_without_a_mesh_is_omitted(tmp_path: Path):
    scene = starter_scene()
    asset = scene.assets[0]
    box = trimesh.creation.box(extents=(1.0, 1.0, 1.0))
    mesh_path = tmp_path / "barn.glb"
    box.export(mesh_path)
    asset.mesh_path = str(mesh_path)
    asset.instances = [Instance(index=0, xy=(2.0, 3.0), yaw=0.0)]
    empty = scene.assets[1]
    empty.mesh_path = None
    empty.instances = []
    scene.assets = [asset, empty]
    assembled = build_scene(scene, tmp_path / "missing-ground.obj")
    assert "silo_01" not in assembled.geometry
    assert asset.id in assembled.geometry


def test_stage_metadata_contract():
    assert usd_stage_metadata() == {"upAxis": "Z", "metersPerUnit": 1.0}


def test_write_usdc_sets_up_axis_and_meters(tmp_path: Path):
    pxr = pytest.importorskip("pxr")
    scene = starter_scene()
    asset = scene.assets[0]
    box = trimesh.creation.box(extents=(1.0, 1.0, 1.0))
    mesh_path = tmp_path / "barn.glb"
    box.export(mesh_path)
    asset.mesh_path = str(mesh_path)
    asset.instances = [Instance(index=0, xy=(2.0, 3.0), yaw=0.0)]
    scene.assets = [asset]
    assembled = build_scene(scene, tmp_path / "missing-ground.obj")
    usd_path = tmp_path / "farm.usdc"
    write_usd(assembled, usd_path)
    stage = pxr.Usd.Stage.Open(str(usd_path))
    assert stage.GetMetadata("upAxis") == "Z"
    assert stage.GetMetadata("metersPerUnit") == 1.0
    assert stage.GetPrimAtPath("/World/ground").IsValid()
    material = stage.GetPrimAtPath("/World/Looks/ground/Preview")
    assert material.IsValid()
    assert material.GetAttribute("info:id").Get() == "UsdPreviewSurface"


def test_ground_paint_exports_color_attribute(tmp_path: Path):
    mesh = trimesh.creation.box(extents=(2.0, 2.0, 0.2))
    ground_path = tmp_path / "ground.obj"
    mesh.export(ground_path)
    colors = np.zeros((len(mesh.vertices), 4), dtype=np.uint8)
    colors[:, 1] = 110
    colors[:, 3] = 255
    np.save(tmp_path / "ground.colors.npy", colors)
    scene = starter_scene()
    scene.assets = []
    assembled = build_scene(scene, ground_path)
    ground = assembled.geometry["ground"]
    assert type(ground.visual).__name__ == "ColorVisuals"
    glb_path = tmp_path / "farm.glb"
    assembled.export(glb_path)
    raw = glb_path.read_bytes()
    chunk_length = int.from_bytes(raw[12:16], "little")
    payload = json.loads(raw[20 : 20 + chunk_length])
    attributes = payload["meshes"][0]["primitives"][0]["attributes"]
    assert "COLOR_0" in attributes
    usd_path = tmp_path / "farm.usdc"
    write_usd(assembled, usd_path)
    pxr = pytest.importorskip("pxr")
    stage = pxr.Usd.Stage.Open(str(usd_path))
    primvar = pxr.UsdGeom.PrimvarsAPI(stage.GetPrimAtPath("/World/ground")).GetPrimvar("displayColor")
    assert primvar.IsDefined()
    painted = primvar.Get()
    assert painted is not None and len(painted) == len(mesh.vertices)
    assert painted[0][1] == pytest.approx(110 / 255, abs=0.01)
