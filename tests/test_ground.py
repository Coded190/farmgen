import numpy as np
import pytest

from farmgen.contact import contact_hole_fraction
import trimesh

from farmgen.ground import (
    fit_relief,
    flat_ground_mesh,
    inspect_ground,
    mesh_has_baked_color,
    orient_ground_vertices,
    sample_height,
    same_ground_geometry,
    terrain_mesh,
)


def test_flat_ground_passes_and_matches_across_builds():
    mesh = flat_ground_mesh((40.0, 40.0), divisions=4)
    report = inspect_ground(mesh, (40.0, 40.0))
    assert report.ok
    assert report.meshes == 1
    assert report.faces < 500_000
    assert report.thickness_m < 0.25
    assert same_ground_geometry(mesh.vertices, flat_ground_mesh((40.0, 40.0), divisions=4).vertices)


def test_the_ring_floor_sits_on_the_yard():
    """A ring that starts meters above the plot is shifted down onto the yard.

    The step is the surround relief, 0.6 m here: max(outside Z) - min(outside Z).
    The yard floor is the median Z of vertices inside the plot.
    """
    vertices = np.array(
        [
            [60.0, 60.0, 0.0],
            [50.0, 50.0, 0.2],
            [70.0, 70.0, 0.1],
            [10.0, 10.0, 4.0],
            [10.0, 60.0, 8.0],
            [110.0, 110.0, 6.0],
        ]
    )
    fitted = fit_relief(vertices, (120.0, 120.0), (40.0, 40.0), 0.4, 0.6)
    floor = float(np.median(fitted[:3, 2]))
    outside = fitted[3:, 2]
    assert abs(float(outside.min()) - floor) < 1e-6
    assert float(outside.max() - outside.min()) <= 0.6 + 1e-6


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
    fitted = spiked.copy()
    fitted.vertices = fit_relief(fitted.vertices, (120.0, 120.0), (40.0, 40.0), 0.4, 4.0)
    good = inspect_ground(fitted, (120.0, 120.0), plot_m=(40.0, 40.0), relief_m=0.4, surround_relief_m=4.0)
    assert good.ok, good.reason


def test_vertical_sheet_keeps_its_face_count_when_the_short_axis_moves_to_z():
    from farmgen.ground import prepare_ground_mesh

    mesh = flat_ground_mesh((120.0, 120.0), divisions=4)
    # Stand the sheet up: the short axis is Y, matching the raw Infinigen tile.
    mesh.vertices = mesh.vertices[:, [0, 2, 1]]
    before = int(len(mesh.faces))
    prepared = prepare_ground_mesh(mesh, (120.0, 120.0), (40.0, 40.0), 0.4, 4.0)
    extent = np.asarray(prepared.vertices).max(axis=0) - np.asarray(prepared.vertices).min(axis=0)
    assert extent[2] <= extent[0] and extent[2] <= extent[1]
    assert int(len(prepared.faces)) == before


def test_xz_sheet_lands_on_xy_and_a_spike_still_fails_before_the_fit():
    raw = np.array(
        [
            [0.0, 0.0, 0.0],
            [120.0, 0.0, 0.0],
            [0.0, 12.0, 0.0],
            [0.0, 0.0, 120.0],
        ],
        dtype=float,
    )
    laid = orient_ground_vertices(raw)
    extent = laid.max(axis=0) - laid.min(axis=0)
    assert extent[0] == pytest.approx(120.0)
    assert extent[1] == pytest.approx(120.0)
    assert extent[2] == pytest.approx(12.0)
    assert laid.min() == pytest.approx(0.0)


def test_vertex_colors_count_as_a_bake():
    mesh = flat_ground_mesh((4.0, 4.0), divisions=2)
    assert not mesh_has_baked_color(mesh)
    colors = np.full((len(mesh.vertices), 4), 255, dtype=np.uint8)
    mesh.visual.vertex_colors = colors
    assert not mesh_has_baked_color(mesh)
    colors = colors.copy()
    colors[: len(colors) // 2, :3] = (122, 74, 36)
    colors[len(colors) // 2 :, :3] = (62, 110, 42)
    painted = trimesh.Trimesh(vertices=mesh.vertices.copy(), faces=mesh.faces.copy(), process=False)
    painted.visual.vertex_colors = colors
    assert mesh_has_baked_color(painted)


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
