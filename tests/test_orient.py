"""Y-up boxes stand on Z before scaling, and the other candidates are pure rotations."""

import numpy as np

from farmgen.orient import CANDIDATE_ORDER, apply_rotation, next_rotation, rotation_matrix
from farmgen.scale import scale_vertices_to_size


def test_y_up_box_stands_on_z_after_scale():
    # Tall along Y, short along Z. That is the TRELLIS pose.
    raw = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 4.0, 0.0],
            [0.0, 0.0, 0.5],
        ]
    )
    upright = apply_rotation(raw, "y_up")
    extents = upright.max(axis=0) - upright.min(axis=0)
    assert extents[2] == max(extents)
    # Source +Y is up, and it stays up.
    assert apply_rotation([[0.0, 1.0, 0.0]], "y_up")[0, 2] > 0
    scaled = scale_vertices_to_size(upright, (2.0, 3.0, 5.0))
    assert scaled[:, 2].min() == 0.0
    scaled_extents = scaled.max(axis=0) - scaled.min(axis=0)
    np.testing.assert_allclose(scaled_extents, [2.0, 3.0, 5.0])


def test_candidates_are_proper_and_cover_each_source_axis():
    source = np.eye(3)
    landed = []
    for name in CANDIDATE_ORDER:
        matrix = rotation_matrix(name)
        assert abs(np.linalg.det(matrix) - 1.0) < 1e-9
        upright = apply_rotation(source, name)
        # The unit vector that landed on +Z.
        landed.append(int(np.argmax(np.abs(upright[:, 2]))))
    assert len(set(landed)) == 3


def test_next_rotation_stops_at_the_end():
    assert next_rotation("y_up") == "z_up"
    assert next_rotation(CANDIDATE_ORDER[-1]) is None
