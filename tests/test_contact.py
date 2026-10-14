import numpy as np

from farmgen.contact import HOLE_FRACTION_LIMIT, contact_hole_fraction, seal_contact_floor


def _open_box():
    vertices = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 1.0],
            [1.0, 1.0, 1.0],
            [0.0, 1.0, 1.0],
        ],
        dtype=np.float64,
    )
    # Four walls and a top. No floor, so the contact hull is an open square.
    faces = np.array(
        [
            [0, 1, 5],
            [0, 5, 4],
            [1, 2, 6],
            [1, 6, 5],
            [2, 3, 7],
            [2, 7, 6],
            [3, 0, 4],
            [3, 4, 7],
            [4, 5, 6],
            [4, 6, 7],
        ],
        dtype=np.int64,
    )
    return vertices, faces


def test_contact_smaller_than_a_grid_cell_is_not_a_hole():
    # Wide canopy, 2 mm root. No 32x32 cell center falls inside the root hull.
    vertices = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.002, 0.0, 0.0],
            [0.001, 0.002, 0.0],
            [-0.5, -0.5, 1.0],
            [0.5, 0.5, 1.0],
        ],
        dtype=np.float64,
    )
    faces = np.array([[0, 1, 2], [0, 1, 3], [1, 2, 4], [2, 0, 3]], dtype=np.int64)
    assert contact_hole_fraction(vertices, faces) == 0.0


def test_open_bottom_is_rejected_until_sealed():
    vertices, faces = _open_box()
    raw = contact_hole_fraction(vertices, faces)
    assert raw > HOLE_FRACTION_LIMIT
    sealed_vertices, sealed_faces, hole, did_seal = seal_contact_floor(vertices, faces)
    assert did_seal
    assert hole <= HOLE_FRACTION_LIMIT
    assert sealed_vertices.shape[0] > vertices.shape[0]
    assert sealed_faces.shape[0] > faces.shape[0]
    assert float(sealed_vertices[:, 2].min()) == 0.0
