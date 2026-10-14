"""The composed stage is meters and Z-up when usd-core is installed."""

from pathlib import Path

import pytest
import trimesh

from farmgen.compose import build_scene, usd_stage_metadata, write_usd
from farmgen.scene import Instance, starter_scene


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
