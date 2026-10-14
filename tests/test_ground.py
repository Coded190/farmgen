import numpy as np

from farmgen.contact import contact_hole_fraction
from farmgen.ground import flat_ground_mesh, inspect_ground, sample_height, same_ground_geometry, terrain_mesh


def test_flat_ground_passes_and_matches_across_builds():
    mesh = flat_ground_mesh((40.0, 40.0), divisions=4)
    report = inspect_ground(mesh, (40.0, 40.0))
    assert report.ok
    assert report.meshes == 1
    assert report.faces < 500_000
    assert report.thickness_m < 0.25
    assert same_ground_geometry(mesh.vertices, flat_ground_mesh((40.0, 40.0), divisions=4).vertices)


def test_thick_and_heavy_ground_fail():
    mesh = flat_ground_mesh((40.0, 40.0), divisions=2)
    mesh.vertices[:, 2] = np.linspace(0.0, 2.0, len(mesh.vertices))
    thick = inspect_ground(mesh, (40.0, 40.0))
    assert not thick.ok
    assert thick.thickness_m >= 0.25

    dense = flat_ground_mesh((40.0, 40.0), divisions=2)
    dense.faces = np.vstack([dense.faces] * 200000)
    heavy = inspect_ground(dense, (40.0, 40.0))
    assert not heavy.ok
    assert heavy.faces >= 500_000


def test_contact_hole_rejects_a_missing_center():
    solid = flat_ground_mesh((2.0, 2.0), divisions=8)
    assert contact_hole_fraction(solid.vertices, solid.faces) <= 0.25

    # Border vertices only. The convex hull still covers the square, and the
    # two corner triangles leave the middle empty.
    vertices = np.array(
        [
            [0.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [2.0, 2.0, 0.0],
            [0.0, 2.0, 0.0],
            [0.15, 0.15, 0.0],
            [1.85, 0.15, 0.0],
            [1.85, 1.85, 0.0],
            [0.15, 1.85, 0.0],
        ]
    )
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
        ]
    )
    assert contact_hole_fraction(vertices, faces) > 0.25


def test_plot_relief_stays_under_the_surround():
    mesh = terrain_mesh((120.0, 120.0), (40.0, 40.0), 0.4, 4.0, seed=3, divisions=8)
    report = inspect_ground(
        mesh,
        (120.0, 120.0),
        plot_m=(40.0, 40.0),
        relief_m=0.4,
        surround_relief_m=4.0,
    )
    assert report.ok, report.reason
    again = terrain_mesh((120.0, 120.0), (40.0, 40.0), 0.4, 4.0, seed=3, divisions=8)
    assert np.max(np.abs(mesh.vertices - again.vertices)) < 1e-9
    # A cliff inside the plot fails the relief rule and still has the right extent.
    spiked = terrain_mesh((120.0, 120.0), (40.0, 40.0), 0.4, 4.0, seed=3, divisions=8)
    spiked.vertices[len(spiked.vertices) // 2, 2] = 3.0
    bad = inspect_ground(spiked, (120.0, 120.0), plot_m=(40.0, 40.0), relief_m=0.4, surround_relief_m=4.0)
    assert not bad.ok
    assert "plot relief" in bad.reason


def test_sample_height_reads_a_known_triangle():
    vertices = np.array(
        [
            [0.0, 0.0, 2.0],
            [4.0, 0.0, 2.0],
            [0.0, 4.0, 2.0],
        ]
    )
    faces = np.array([[0, 1, 2]])
    height = sample_height(vertices, faces, (1.0, 1.0))
    assert abs(height - 2.0) < 1e-6
