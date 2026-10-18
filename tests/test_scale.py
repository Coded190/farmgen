import numpy as np
import trimesh

from farmgen.run import _decimate_copy
from farmgen.scale import bbox_size, scale_vertices_to_size


def test_one_factor_fits_a_cube_inside_the_box():
    corners = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 1.0, 0.0],
            [1.0, 0.0, 1.0],
            [0.0, 1.0, 1.0],
            [1.0, 1.0, 1.0],
        ]
    )
    scaled = scale_vertices_to_size(corners, (12.0, 8.0, 7.0))
    # The height 7 is the limiting side, so every axis uses that factor.
    np.testing.assert_allclose(bbox_size(scaled), (7.0, 7.0, 7.0), atol=1e-9)
    assert float(scaled[:, 2].min()) == 0.0
    assert abs(float(scaled[:, 0].min() + scaled[:, 0].max())) < 1e-9
    assert abs(float(scaled[:, 1].min() + scaled[:, 1].max())) < 1e-9


def test_a_stray_vertex_does_not_set_the_height():
    corners = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 1.0],
            [0.0, 1.0, 1.0],
            [1.0, 1.0, 1.0],
        ]
    )
    stray = np.vstack([np.repeat(corners, 20, axis=0), [[0.5, 0.5, 8.0]]])
    scaled = scale_vertices_to_size(stray, (2.0, 2.0, 2.0))
    assert len(scaled) == 160
    np.testing.assert_allclose(bbox_size(scaled), (2.0, 2.0, 2.0), atol=1e-9)


def test_one_shared_factor_keeps_a_long_box_in_proportion():
    corners = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 2.0],
            [1.0, 1.0, 0.0],
            [1.0, 0.0, 2.0],
            [0.0, 1.0, 2.0],
            [1.0, 1.0, 2.0],
        ]
    )
    scaled = scale_vertices_to_size(corners, (3.0, 12.0, 12.0))
    np.testing.assert_allclose(bbox_size(scaled), (3.0, 3.0, 6.0), atol=1e-9)


def test_decimate_copy_targets_face_count():
    mesh = trimesh.creation.icosphere(subdivisions=4)
    assert len(mesh.faces) > 1000
    colors = np.tile(np.array([[10, 20, 30, 255]], dtype=np.uint8), (len(mesh.vertices), 1))
    reduced = _decimate_copy(mesh, colors, 400)
    assert 0 < len(reduced.faces) <= 400
    assert len(reduced.visual.vertex_colors) == len(reduced.vertices)


def test_decimate_copy_clusters_disconnected_triangles():
    rng = np.random.default_rng(0)
    verts = []
    faces = []
    for _ in range(800):
        base = rng.normal(size=3) * 4.0
        triangle = base + rng.normal(size=(3, 3)) * 0.01
        start = len(verts)
        verts.extend(triangle.tolist())
        faces.append([start, start + 1, start + 2])
    mesh = trimesh.Trimesh(vertices=np.asarray(verts), faces=np.asarray(faces), process=False)
    colors = np.tile(np.array([[4, 5, 6, 255]], dtype=np.uint8), (len(mesh.vertices), 1))
    reduced = _decimate_copy(mesh, colors, 200)
    assert 0 < len(reduced.faces) <= 200
    assert len(reduced.visual.vertex_colors) == len(reduced.vertices)


def test_catalog_size_survives_uniform_scale(tmp_path, monkeypatch):
    from farmgen.run import _scale_cached
    from farmgen.scene import AssetSpec, GroundSpec, Scene

    monkeypatch.setattr("farmgen.run._render_asset_views", lambda *_args, **_kwargs: None)
    monkeypatch.setattr("farmgen.run.log_line", lambda *_args, **_kwargs: "")
    source = tmp_path / "cache.glb"
    box = trimesh.creation.box(extents=(1.0, 1.0, 2.0))
    box.visual.vertex_colors = np.tile(np.array([[10, 20, 30, 255]], dtype=np.uint8), (len(box.vertices), 1))
    box.export(source)
    dest = tmp_path / "scaled.glb"
    asset = AssetSpec(id="silo_01", prompt="a metal silo", size_m=(3.0, 12.0, 12.0), count=1)
    scene = Scene(id="farm_test", seed=0, brief="a silo", ground=GroundSpec(size_m=(40.0, 40.0)), assets=[asset])
    _scale_cached(scene, asset, source, dest)
    assert asset.size_m == (3.0, 12.0, 12.0)
    assert asset.achieved_m is not None
    assert all(float(got) <= float(limit) + 1e-6 for got, limit in zip(asset.achieved_m, asset.size_m, strict=True))
    assert max(asset.achieved_m) > 1.0
