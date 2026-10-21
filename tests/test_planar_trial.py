"""Optional integration tests for the separately built CGAL planar trial."""

import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
import trimesh
import fusemesh

ROOT = Path(__file__).resolve().parents[1]
HELPER = Path(os.environ.get("FUSE_PLANAR_HELPER", ROOT / "build/planar-native/fuse_planar"))
pytestmark = pytest.mark.skipif(not HELPER.is_file(), reason="standalone CGAL helper not built")


def run_trial(mesh, tmp_path, angle=3, distance=0.00025, expect_success=True, suffix=".ply"):
    source, output, report = [
        tmp_path / name for name in ("input.ply", "output" + suffix, "report.json")
    ]
    mesh.export(source)
    before = source.read_bytes()
    output.write_bytes(b"previous output")
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "benchmarks/trial_planar.py"),
            str(source),
            str(output),
            "--report",
            str(report),
            "--helper",
            str(HELPER),
            "--angle-tolerance",
            str(angle),
            "--plane-tolerance",
            str(distance),
        ],
        capture_output=True,
        text=True,
    )
    assert source.read_bytes() == before
    data = json.loads(report.read_text())
    if not expect_success:
        assert result.returncode != 0
        assert output.read_bytes() == b"previous output"
        return data
    assert result.returncode == 0, result.stdout + result.stderr
    data = data["native"]
    assert data["max_retained_plane_distance"] <= distance
    assert data["max_retained_normal_angle_degrees"] <= angle + 1e-8
    # STL stores independent triangles, so weld coincident vertices on reload.
    reduced = trimesh.load(output, process=suffix.lower() == ".stl")
    assert reduced.is_winding_consistent
    assert reduced.euler_number == mesh.euler_number
    assert reduced.is_watertight == mesh.is_watertight
    return reduced, data


def grid(hole=False, curved=False):
    size = 24
    x, y = np.meshgrid(np.linspace(-0.1, 0.1, size + 1), np.linspace(-0.1, 0.1, size + 1))
    z = 0.0015 * (x / 0.1) ** 2 if curved else np.zeros_like(x)
    v = np.column_stack([x.ravel(), y.ravel(), z.ravel()])
    faces = []
    for j in range(size):
        for i in range(size):
            if hole and 8 <= i < 16 and 8 <= j < 16:
                continue
            a = j * (size + 1) + i
            b = a + 1
            c = a + size + 1
            d = c + 1
            faces.extend([(a, b, d), (a, d, c)])
    mesh = trimesh.Trimesh(v, faces, process=False)
    mesh.remove_unreferenced_vertices()
    return mesh


def check_distance(original, reduced, tolerance):
    report = fusemesh.compare_surfaces(original, reduced, samples=3000, seed=4, tolerance=tolerance)
    assert report["symmetric"]["f_score"] == 1.0, report


@pytest.mark.parametrize("suffix", [".ply", ".obj", ".off", ".stl", ".OBJ"])
def test_planar_box_reduces_to_corners(tmp_path, suffix):
    mesh = trimesh.creation.box().subdivide().subdivide().subdivide()
    reduced, data = run_trial(mesh, tmp_path, suffix=suffix)
    assert len(reduced.faces) == 12 and len(reduced.vertices) == 8
    assert data["rejected_regions"] == 0
    check_distance(mesh, reduced, 1e-8)


def test_plate_hole_is_preserved(tmp_path):
    mesh = grid(hole=True)
    reduced, data = run_trial(mesh, tmp_path)
    assert len(reduced.faces) < len(mesh.faces) // 10
    assert data["euler_characteristic"] == 0
    check_distance(mesh, reduced, 1e-8)


def test_cylinder_does_not_collapse_into_one_plane(tmp_path):
    mesh = trimesh.creation.cylinder(radius=0.1, height=0.2, sections=128)
    reduced, data = run_trial(mesh, tmp_path)
    assert 20 < len(reduced.faces) < len(mesh.faces)
    assert data["final_regions"] > 10
    check_distance(mesh, reduced, 0.001)


def test_small_normal_variation_still_requires_plane_distance(tmp_path):
    mesh = grid(curved=True)
    reduced, data = run_trial(mesh, tmp_path, angle=10, distance=0.0001)
    assert data["final_regions"] > 1
    assert len(reduced.faces) > 2
    check_distance(mesh, reduced, 0.0005)


def test_angle_tolerance_controls_shallow_crease_preservation(tmp_path):
    mesh = grid()
    mesh.vertices[:, 2] = np.abs(mesh.vertices[:, 0]) * np.tan(np.deg2rad(4))
    strict_dir, loose_dir = tmp_path / "strict", tmp_path / "loose"
    strict_dir.mkdir()
    loose_dir.mkdir()
    strict, _ = run_trial(mesh, strict_dir, angle=1, distance=0.02)
    loose, _ = run_trial(mesh, loose_dir, angle=10, distance=0.02)
    assert len(loose.faces) < len(strict.faces)
    check_distance(mesh, strict, 1e-7)
    check_distance(mesh, loose, 0.008)


def test_nonmanifold_input_preserves_previous_output_on_failure(tmp_path):
    a = trimesh.creation.box()
    b = a.copy()
    b.apply_translation([1, 1, 1])
    mesh = trimesh.util.concatenate([a, b])
    mesh.merge_vertices()
    data = run_trial(mesh, tmp_path, expect_success=False)
    assert data["status"] == "failed"


def test_input_cannot_be_overwritten(tmp_path):
    path = tmp_path / "mesh.ply"
    path.write_bytes(b"unchanged")
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "benchmarks/trial_planar.py"),
            str(path),
            str(path),
            "--report",
            str(tmp_path / "report.json"),
            "--helper",
            str(HELPER),
        ],
        capture_output=True,
    )
    assert result.returncode == 2
    assert path.read_bytes() == b"unchanged"


def test_curved_hole_boundary_is_not_straightened_away(tmp_path):
    segments = 256
    theta = np.arange(segments) * 2 * np.pi / segments
    radii = np.linspace(0.05, 0.1, 5)
    vertices = np.array([[r * np.cos(t), r * np.sin(t), 0] for r in radii for t in theta])
    faces = []
    for j in range(len(radii) - 1):
        for i in range(segments):
            a = j * segments + i
            b = j * segments + (i + 1) % segments
            c = a + segments
            d = b + segments
            faces.extend([(a, c, d), (a, d, b)])
    mesh = trimesh.Trimesh(vertices, faces, process=False)
    reduced, data = run_trial(mesh, tmp_path)
    assert data["euler_characteristic"] == 0
    assert len(reduced.faces) < len(mesh.faces)
    _, distance, _ = trimesh.proximity.closest_point_naive(reduced, np.zeros((1, 3)))
    assert distance[0] >= 0.05 - 0.00025
    check_distance(mesh, reduced, 0.0003)


def test_long_straight_boundary_is_reduced_without_changing_shape(tmp_path):
    count = 600
    x = np.linspace(0, 1, count + 1)
    v = np.vstack(
        [
            np.column_stack([x, np.zeros_like(x), np.zeros_like(x)]),
            np.column_stack([x, np.ones_like(x) * 0.1, np.zeros_like(x)]),
        ]
    )
    faces = []
    for i in range(count):
        a = i
        b = i + 1
        c = i + count + 1
        d = c + 1
        faces.extend([(a, b, d), (a, d, c)])
    mesh = trimesh.Trimesh(v, faces, process=False)
    reduced, _ = run_trial(mesh, tmp_path)
    assert len(reduced.faces) < 32
    check_distance(mesh, reduced, 1e-8)
