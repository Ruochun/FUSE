"""Analytic geometry and independent nearest-triangle quality checks."""

import json
import subprocess
import sys

import numpy as np
import pytest
import trimesh

import fusemesh
from fusemesh import quality


def triangle(z=0):
    return np.array([[0, 0, z], [1, 0, z], [0, 1, z]], float), np.array([[0, 1, 2]])


def compare(a, b, **kwargs):
    return fusemesh.compare_surfaces(a, b, samples=500, **kwargs)


def test_parallel_triangles_analytic_distance():
    report = compare(triangle(), triangle(0.25), tolerance=0.3, voxel_size=0.1)
    for direction in ("original_to_reconstructed", "reconstructed_to_original", "symmetric"):
        for metric in ("mean", "rms", "p95", "sampled_max"):
            assert report[direction][metric] == pytest.approx(0.25)
        assert report[direction]["coverage"] == 1
    assert report["symmetric"]["f_score"] == 1
    assert report["symmetric_in_voxels"]["rms"] == pytest.approx(2.5)
    assert compare(triangle(), triangle(0.25), tolerance=0.2)["symmetric"]["f_score"] == 0


def test_identical_different_triangulation_and_reproducibility():
    mesh = trimesh.creation.box()
    report = compare(mesh, mesh.subdivide(), seed=17)
    assert report["symmetric"]["sampled_max"] < 1e-14
    assert report == compare(mesh, mesh.subdivide(), seed=17)
    assert report == compare(mesh, mesh.subdivide(), seed=17, batch_size=7)
    assert not report["sampling"]["hausdorff_certified"]
    json.dumps(report, allow_nan=False)


def test_missing_and_extra_component_direction():
    first = trimesh.creation.box()
    second = first.copy()
    second.apply_translation([3, 0, 0])
    both = trimesh.util.concatenate([first, second])
    missing = compare(both, first, tolerance=0.01)
    assert missing["reconstructed_to_original"]["sampled_max"] < 1e-14
    assert 0.4 < missing["original_to_reconstructed"]["coverage"] < 0.6
    extra = compare(first, both, tolerance=0.01)
    assert extra["original_to_reconstructed"]["sampled_max"] < 1e-14
    assert extra["reconstructed_to_original"]["sampled_max"] > 2


@pytest.mark.parametrize("offset", [0, 0.5, 1])
def test_union_removes_buried_faces(offset):
    a = trimesh.creation.box()
    b = a.copy()
    b.apply_translation([offset, 0, 0])
    expected = trimesh.creation.box(extents=[1 + offset, 1, 1])
    expected.apply_translation([offset / 2, 0, 0])
    report = compare([a, b], expected)
    assert report["reference"]["mode"] == "solid_union"
    assert report["symmetric"]["sampled_max"] < 1e-12


def test_union_nested_and_disconnected():
    outer = trimesh.creation.box(extents=[2, 2, 2])
    inner = trimesh.creation.box()
    assert compare([outer, inner], outer)["symmetric"]["sampled_max"] < 1e-12
    inner.apply_translation([5, 0, 0])
    expected = trimesh.util.concatenate([outer, inner])
    assert compare([outer, inner], expected)["symmetric"]["sampled_max"] < 1e-12


def test_union_keeps_cavity():
    outer = trimesh.creation.box(extents=[3, 3, 3])
    inner = trimesh.creation.box()
    inner.invert()
    hollow = trimesh.util.concatenate([outer, inner])
    other = trimesh.creation.box()
    other.apply_translation([5, 0, 0])
    expected = trimesh.util.concatenate([hollow, other])
    assert compare([hollow, other], expected)["symmetric"]["sampled_max"] < 1e-12
    filled = trimesh.util.concatenate([outer, other])
    assert compare([hollow, other], filled)["original_to_reconstructed"]["sampled_max"] > 0.9


def test_point_triangle_query_matches_independent_brute_force():
    mesh = trimesh.creation.icosphere(subdivisions=1)
    points = np.random.default_rng(10).uniform(-2, 2, (100, 3))
    surface = quality._Surface(mesh.triangles)
    try:
        distance, _ = surface.nearest(points)
    finally:
        surface.close()
    expected = trimesh.proximity.closest_point_naive(mesh, points)[1]
    np.testing.assert_allclose(distance, expected, atol=1e-14)


def test_area_weighting_not_triangle_count():
    # One unit-area surface at z=0, one quarter-area surface at z=1.
    a = trimesh.Trimesh(*triangle(), process=False)
    b = trimesh.Trimesh(*triangle(1), process=False)
    b.vertices[:, :2] *= 0.5
    source = trimesh.util.concatenate([a.subdivide().subdivide(), b])
    report = fusemesh.compare_surfaces(source, a, samples=5000, tolerance=0.01)
    assert report["original_to_reconstructed"]["coverage"] == pytest.approx(0.8, abs=0.025)
    assert report["original_to_reconstructed"]["rms"] == pytest.approx(np.sqrt(0.2), abs=0.025)


def test_translation_scaling_and_no_mutation():
    a, b = triangle(), triangle(0.25)
    va, fa = a[0].copy(), a[1].copy()
    compare(a, b)
    np.testing.assert_array_equal(a[0], va)
    np.testing.assert_array_equal(a[1], fa)
    a = (va * 1e-5 + 100, fa)
    b = (b[0] * 1e-5 + 100, b[1])
    assert compare(a, b)["symmetric"]["rms"] == pytest.approx(2.5e-6, rel=1e-7)


def test_files_scene_result_and_cli(tmp_path):
    mesh = trimesh.creation.box()
    source = tmp_path / "box.off"
    mesh.export(source)
    result = fusemesh.ReconstructionResult(mesh.vertices, mesh.faces, {})
    assert compare(source, result)["symmetric"]["sampled_max"] < 1e-14
    scene = trimesh.Scene()
    scene.add_geometry(mesh)
    scene.add_geometry(mesh, transform=trimesh.transformations.translation_matrix([0.5, 0, 0]))
    expected = trimesh.creation.box(extents=[1.5, 1, 1])
    expected.apply_translation([0.25, 0, 0])
    assert compare(scene, expected)["symmetric"]["sampled_max"] < 1e-12
    command = [
        sys.executable,
        "-m",
        "fusemesh.quality",
        str(source),
        "--reconstructed",
        str(source),
        "--samples",
        "100",
        "--tolerance",
        "0.01",
    ]
    proc = subprocess.run(command, capture_output=True, text=True, check=True)
    assert not proc.stderr
    assert json.loads(proc.stdout)["symmetric"]["f_score"] == 1
    output = tmp_path / "report.json"
    subprocess.run(command + ["--output", str(output)], check=True)
    assert json.loads(output.read_text())["sampling"]["samples_per_direction"] == 100


@pytest.mark.parametrize(
    "option,value",
    [
        ("samples", 0),
        ("samples", True),
        ("samples", 1.2),
        ("seed", -1),
        ("tolerance", float("nan")),
        ("tolerance", 0),
        ("voxel_size", -1),
        ("batch_size", 0),
    ],
)
def test_bad_options(option, value):
    with pytest.raises((ValueError, TypeError)):
        fusemesh.compare_surfaces(triangle(), triangle(), **{option: value})


def test_bad_meshes_and_open_union():
    with pytest.raises(ValueError, match="at least one"):
        compare([], triangle())
    with pytest.raises(ValueError, match="exactly one"):
        compare(triangle(), [triangle(), triangle()])
    with pytest.raises(ValueError, match="closed"):
        compare([triangle(), triangle(1)], triangle())
    for pair in [
        (np.zeros((3, 3)), np.array([[0, 1, 2]])),
        (triangle()[0], np.array([[0, 1, 3]])),
        (triangle()[0], np.array([[0.0, 1.0, 2.0]])),
        (triangle()[0] * np.nan, triangle()[1]),
    ]:
        with pytest.raises(ValueError):
            compare(pair, triangle())


def test_missing_union_dependency_has_actionable_error(monkeypatch):
    monkeypatch.setitem(sys.modules, "manifold3d", None)
    mesh = trimesh.creation.box()
    with pytest.raises(ImportError, match=r"fusemesh\[quality\]"):
        compare([mesh, mesh], mesh)
    assert compare(mesh, mesh)["symmetric"]["sampled_max"] < 1e-14


def test_inverted_union_is_rejected():
    a = trimesh.creation.box()
    a.invert()
    with pytest.raises(ValueError, match="positive signed volume"):
        compare([a, trimesh.creation.box()], a)


@pytest.mark.gpu
def test_reconstructed_and_simplified_union_quality():
    a = trimesh.creation.box()
    b = a.copy()
    b.apply_translation([0.5, 0, 0])
    result = fusemesh.reconstruct([a, b], voxel_size=0.15)
    report = compare([a, b], result, tolerance=0.15)
    assert report["symmetric"]["rms"] < 0.06
    assert report["symmetric"]["f_score"] > 0.95
    reduced = fusemesh.simplify(result, max_error=0.04, on_failure="keep")
    report = compare([a, b], reduced, tolerance=0.19)
    assert report["symmetric"]["f_score"] > 0.95


def dirty_triangle(z=0):
    v, f = triangle(z)
    # A collinear face away from the surface, a repeated index, and distinct
    # indices at coincident positions. All must vanish from source AND target.
    extra = np.array([[10, 0, z], [11, 0, z], [12, 0, z], [10, 0, z]])
    return np.vstack((v, extra)), np.vstack((f, [[3, 4, 5], [0, 0, 1], [3, 6, 4]]))


def test_drop_degenerates_both_sides_and_preserve_inputs():
    a, b = dirty_triangle(), dirty_triangle(0.25)
    before = [x.copy() for pair in (a, b) for x in pair]
    with pytest.raises(ValueError, match="original mesh 0: 3 degenerate"):
        compare(a, b)
    with pytest.raises(ValueError, match="reconstructed mesh: 3 degenerate"):
        compare(triangle(), b)
    report = compare(a, b, degenerate_faces="drop", tolerance=0.3)
    expected = compare(triangle(), triangle(0.25), tolerance=0.3)
    for direction in ("original_to_reconstructed", "reconstructed_to_original", "symmetric"):
        assert report[direction] == expected[direction]
    counts = {"input_faces": 4, "removed_faces": 3, "retained_faces": 1}
    assert report["cleaning"] == {
        "degenerate_faces": "drop",
        "originals": [counts],
        "reconstructed": counts,
    }
    for actual, snapshot in zip([x for pair in (a, b) for x in pair], before):
        np.testing.assert_array_equal(actual, snapshot)


@pytest.mark.parametrize("side", ["original", "reconstructed"])
def test_drop_all_degenerate_is_an_error(side):
    bad = (np.zeros((3, 3)), np.array([[0, 1, 2]]))
    a, b = (bad, triangle()) if side == "original" else (triangle(), bad)
    with pytest.raises(ValueError, match=f"{side}.*no valid triangles remain"):
        compare(a, b, degenerate_faces="drop")


def test_drop_does_not_hide_bad_coordinates_or_indices():
    for mesh in [
        (np.full((3, 3), np.nan), np.array([[0, 1, 2]])),
        (triangle()[0], np.array([[0, 0, 9]])),
    ]:
        with pytest.raises(ValueError):
            compare(mesh, triangle(), degenerate_faces="drop")
    with pytest.raises(ValueError, match="degenerate_faces"):
        compare(triangle(), triangle(), degenerate_faces="repair")


def test_drop_retains_thin_positive_area_triangles():
    v = np.array([[0, 0, 0], [1, 0, 0], [0.5, 1e-12, 0]])
    mesh, counts = quality._clean_surface((v, np.array([[0, 1, 2]])), "drop", "test")
    assert counts["removed_faces"] == 0
    assert quality._areas(mesh[0][mesh[1]])[0] > 0


def test_drop_before_union_and_per_original_counts():
    a = trimesh.creation.box()
    b = a.copy()
    b.apply_translation([0.5, 0, 0])
    dirty = (a.vertices, np.vstack((a.faces, [[0, 0, 1]])))
    expected = trimesh.creation.box(extents=[1.5, 1, 1])
    expected.apply_translation([0.25, 0, 0])
    report = compare([dirty, b], expected, degenerate_faces="drop")
    assert report["symmetric"]["sampled_max"] < 1e-12
    assert [c["removed_faces"] for c in report["cleaning"]["originals"]] == [1, 0]
    with pytest.raises(ValueError, match="closed"):
        compare([dirty_triangle(), b], expected, degenerate_faces="drop")


def test_cli_drop_and_error(tmp_path):
    a, b = tmp_path / "a.off", tmp_path / "b.off"
    trimesh.Trimesh(*dirty_triangle(), process=False).export(a)
    trimesh.Trimesh(*dirty_triangle(0.25), process=False).export(b)
    cmd = [
        sys.executable,
        "-m",
        "fusemesh.quality",
        str(a),
        "--reconstructed",
        str(b),
        "--samples",
        "100",
    ]
    rejected = subprocess.run(cmd, capture_output=True, text=True)
    assert rejected.returncode == 2
    assert "degenerate" in rejected.stderr
    accepted = subprocess.run(
        cmd + ["--degenerate-faces", "drop"], capture_output=True, text=True, check=True
    )
    report = json.loads(accepted.stdout)
    assert report["symmetric"]["rms"] == pytest.approx(0.25)
    assert report["cleaning"]["originals"][0]["removed_faces"] == 3
    assert report["cleaning"]["reconstructed"]["removed_faces"] == 3


@pytest.mark.parametrize("scale", [1e-150, 1e150])
def test_drop_is_stable_across_coordinate_scales(scale):
    a, b = dirty_triangle(), dirty_triangle(0.25)
    report = compare((a[0] * scale, a[1]), (b[0] * scale, b[1]), degenerate_faces="drop")
    assert report["cleaning"]["originals"][0]["removed_faces"] == 3
    assert report["cleaning"]["reconstructed"]["removed_faces"] == 3
    assert report["symmetric"]["rms"] / scale == pytest.approx(0.25)
