import numpy as np

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
