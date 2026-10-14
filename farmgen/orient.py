"""Axis-aligned uprighting for generated meshes.

TRELLIS writes Y-up. The default rotation sends ``(x, y, z)`` to ``(x, -z, y)``
so source +Y, the up axis, lands on +Z before scaling. The other candidates put
a different source axis on +Z. None of them calls TRELLIS again.
"""

from __future__ import annotations

import numpy as np

Y_UP = "y_up"

# Proper rotations only (determinant +1). Order is the reorient sequence.
CANDIDATE_ORDER: tuple[str, ...] = ("y_up", "z_up", "neg_y", "neg_z", "x_up", "neg_x")

_MATRICES: dict[str, np.ndarray] = {
    # (x, y, z) -> (x, -z, y) so +Y lands on +Z
    "y_up": np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, -1.0, 0.0]]),
    # already Z-up
    "z_up": np.eye(3),
    # (x, y, z) -> (x, z, -y)
    "neg_y": np.array([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]]),
    # (x, y, z) -> (x, -y, -z)
    "neg_z": np.array([[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]]),
    # (x, y, z) -> (-z, y, x) so +X lands on +Z
    "x_up": np.array([[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]]),
    # (x, y, z) -> (z, y, -x) so -X lands on +Z
    "neg_x": np.array([[0.0, 0.0, -1.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0]]),
}


def rotation_matrix(name: str) -> np.ndarray:
    try:
        matrix = _MATRICES[name]
    except KeyError as exc:
        raise ValueError(f"unknown rotation {name}") from exc
    return np.array(matrix, dtype=np.float64, copy=True)


def apply_rotation(vertices: np.ndarray, name: str) -> np.ndarray:
    points = np.asarray(vertices, dtype=np.float64)
    return points @ rotation_matrix(name)


def next_rotation(name: str) -> str | None:
    """The next candidate after ``name``, or None when the list is exhausted."""
    if name not in CANDIDATE_ORDER:
        raise ValueError(f"unknown rotation {name}")
    index = CANDIDATE_ORDER.index(name) + 1
    if index >= len(CANDIDATE_ORDER):
        return None
    return CANDIDATE_ORDER[index]
