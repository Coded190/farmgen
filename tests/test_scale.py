import numpy as np
import trimesh

from farmgen.run import _decimate_copy
from farmgen.scale import bbox_size, scale_vertices_to_size


def test_unit_box_matches_barn_size():
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
    np.testing.assert_allclose(bbox_size(scaled), (12.0, 8.0, 7.0), atol=1e-9)
    assert float(scaled[:, 2].min()) == 0.0
    assert abs(float(scaled[:, 0].min() + scaled[:, 0].max())) < 1e-9
    assert abs(float(scaled[:, 1].min() + scaled[:, 1].max())) < 1e-9


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
