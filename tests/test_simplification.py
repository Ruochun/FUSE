"""Independent geometry checks and failure-path tests for CPU mesh reduction."""

import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
import trimesh

import fusemesh
from fusemesh import simplification as impl
from geometry_helpers import box, combine, torus


def refined_box(extents=(1, 1, 1), levels=2):
    mesh = trimesh.creation.box(extents=extents)
    for _ in range(levels):
        mesh = mesh.subdivide()
    return mesh


def as_mesh(result):
    return trimesh.Trimesh(result.vertices, result.faces, process=False)


def assert_closed(result, euler=2):
    mesh = as_mesh(result)
    assert mesh.is_watertight and mesh.is_winding_consistent
    assert mesh.euler_number == euler
    assert result.vertices.dtype == np.float64
    assert result.faces.dtype == np.uint64
    return mesh


def sample_surface(mesh):
    # Deterministic barycentric samples independent of the production verifier.
    triangles = np.asarray(mesh.triangles)
    weights = np.array(
        [
            [a, b, 1 - a - b]
            for a in (0, 0.25, 0.5, 0.75, 1)
            for b in (0, 0.25, 0.5, 0.75, 1)
            if a + b <= 1
        ]
    )
    return np.einsum("kj,tjd->tkd", weights, triangles).reshape(-1, 3)


def independent_distance(a, b, limit):
    # Brute-force Trimesh reference avoids the production spatial index/bounds.
    for src, dst in ((a, b), (b, a)):
        points = sample_surface(src)
        for i in range(0, len(points), 32):
            _, distance, _ = trimesh.proximity.closest_point_naive(dst, points[i : i + 32])
            assert distance.max() <= limit + 1e-10


def test_planar_box_reduces_to_twelve_faces_and_preserves_corners():
    original = refined_box(levels=3)
    result = fusemesh.simplify(original, max_error=0.01, target_faces=12)
    mesh = assert_closed(result)
    assert len(result.faces) == 12
    assert mesh.volume == pytest.approx(original.volume, abs=1e-12)
    np.testing.assert_allclose(mesh.bounds, original.bounds, atol=1e-12)
    corners = trimesh.creation.box().vertices
    distances = np.linalg.norm(result.vertices[:, None] - corners[None, :], axis=2)
    assert np.all(distances.min(axis=0) < 1e-12)
    assert np.all(distances.min(axis=1) < 1e-12)
    assert result.metadata["simplification"]["surface_error_bound"] <= 0.01
    independent_distance(original, mesh, 1e-10)


def test_curved_surface_backs_off_face_goal_to_preserve_volume():
    original = trimesh.creation.icosphere(subdivisions=2)
    result = fusemesh.simplify(original, max_error=0.05, target_faces=80)
    mesh = assert_closed(result)
    assert 80 < len(mesh.faces) < len(original.faces)
    assert abs(mesh.volume / original.volume - 1) <= 0.01
    report = result.metadata["simplification"]
    assert not report["target_reached"]
    assert any(not a["accepted"] for a in report["shells"][0]["attempts"])
    independent_distance(original, mesh, 0.05)


def test_thin_solid_does_not_collapse():
    original = refined_box((1, 1, 0.02))
    result = fusemesh.simplify(original, max_error=0.005, target_faces=12)
    mesh = assert_closed(result)
    assert len(mesh.faces) < len(original.faces)
    assert mesh.volume == pytest.approx(original.volume, rel=0.01)
    np.testing.assert_allclose(mesh.bounds, original.bounds, atol=0.005)


@pytest.mark.parametrize("cavity", [False, True])
def test_disconnected_shells_and_cavity_orientation(cavity):
    outer, inner = refined_box((2, 2, 2)), refined_box((0.5, 0.5, 0.5))
    if cavity:
        inner.faces = inner.faces[:, ::-1]
    else:
        inner.vertices += [4, 0, 0]
    v, f = combine((outer.vertices, outer.faces), (inner.vertices, inner.faces))
    result = fusemesh.simplify((v, f), max_error=0.01, target_faces=24)
    mesh = assert_closed(result, euler=4)
    assert len(mesh.faces) == 24
    shells = result.metadata["simplification"]["shells"]
    assert len(shells) == 2
    assert [r["orientation"] for r in shells] == [1, -1 if cavity else 1]
    assert mesh.volume == pytest.approx(outer.volume + inner.volume, rel=1e-10)


def test_torus_handle_survives():
    v, f = torus(nu=24, nv=12)
    result = fusemesh.simplify((v, f), max_error=0.06, target_faces=250, max_volume_change=0.03)
    mesh = assert_closed(result, euler=0)
    assert len(mesh.faces) < len(f)
    independent_distance(trimesh.Trimesh(v, f, process=False), mesh, 0.06)


def test_result_owns_arrays_metadata_and_field():
    mesh = refined_box()
    original = fusemesh.ReconstructionResult(
        np.array(mesh.vertices), np.array(mesh.faces), {"nested": {"value": 7}}, np.ones((2, 2, 2))
    )
    v, f = original.vertices.copy(), original.faces.copy()
    result = fusemesh.simplify(original, max_error=0.01, target_faces=12)
    result.vertices[:] = 99
    result.faces[:] = 0
    result.metadata["nested"]["value"] = 42
    result.field[:] = 0
    np.testing.assert_array_equal(original.vertices, v)
    np.testing.assert_array_equal(original.faces, f)
    assert original.metadata == {"nested": {"value": 7}}
    assert np.all(original.field == 1)


def test_empty_and_no_reduction_requested_do_not_call_backend(monkeypatch):
    def unexpected(*args):
        pytest.fail("backend should not run")

    monkeypatch.setattr(impl, "_decimate", unexpected)
    empty = fusemesh.simplify((np.empty((0, 3)), np.empty((0, 3), dtype=int)), max_error=0.01)
    assert empty.vertices.shape == empty.faces.shape == (0, 3)
    v, f = box()
    result = fusemesh.simplify((v, f), max_error=0.01, target_faces=12)
    np.testing.assert_array_equal(result.vertices, v)
    np.testing.assert_array_equal(result.faces, f)


@pytest.mark.parametrize(
    "options",
    [
        {"max_error": 0},
        {"max_error": -1},
        {"max_error": float("nan")},
        {"max_error": float("inf")},
        {"max_error": True},
        {"max_error": 1e-16},
        {"target_faces": 0},
        {"target_faces": 3},
        {"target_faces": True},
        {"target_faces": 4.5},
        {"feature_angle": 180},
        {"feature_angle": 0},
        {"max_volume_change": -1},
        {"max_checks": 0},
        {"max_checks": True},
        {"on_failure": "ignore"},
    ],
)
def test_invalid_controls(options):
    with pytest.raises((TypeError, ValueError)):
        fusemesh.simplify(box(), **dict({"max_error": 0.01}, **options))


@pytest.mark.parametrize(
    "defect", ["open", "winding", "duplicate", "nan", "index", "float_faces", "pinch"]
)
def test_invalid_meshes_rejected_before_backend(defect, monkeypatch):
    v, f = box()
    if defect == "open":
        f = f[:-1]
    if defect == "winding":
        f[0] = f[0, ::-1]
    if defect == "duplicate":
        f = np.vstack((f, f[0]))
    if defect == "nan":
        v[0, 0] = np.nan
    if defect == "index":
        f[0, 0] = -1
    if defect == "float_faces":
        f = f.astype(float)
    if defect == "pinch":
        v2, f2 = box((1, 1, 1), (2, 2, 2))
        v, f = combine((v, f), (v2, f2))
        f[f == 8] = 6

    def unexpected(*args):
        pytest.fail("invalid input reached backend")

    monkeypatch.setattr(impl, "_decimate", unexpected)
    with pytest.raises(ValueError):
        fusemesh.simplify((v, f), max_error=0.01)


def test_verification_budget_fails_closed_and_keep_is_explicit():
    original = refined_box()
    with pytest.raises(fusemesh.SimplificationError, match="budget"):
        fusemesh.simplify(original, max_error=0.01, target_faces=12, max_checks=1)
    kept = fusemesh.simplify(
        original, max_error=0.01, target_faces=12, max_checks=1, on_failure="keep"
    )
    np.testing.assert_array_equal(kept.vertices, original.vertices)
    np.testing.assert_array_equal(kept.faces, original.faces)
    assert kept.metadata["simplification"]["shells"][0]["status"] == "kept"


@pytest.mark.parametrize("defect", ["open", "inverted", "moved", "no_reduction"])
def test_bad_backend_candidates_are_not_accepted(monkeypatch, defect):
    original = refined_box()

    def bad(v, f, count):
        mesh = trimesh.creation.box()
        cv, cf = np.array(mesh.vertices), np.array(mesh.faces)
        if defect == "open":
            cf = cf[:-1]
        if defect == "inverted":
            cf = cf[:, ::-1]
        if defect == "moved":
            cv += 1
        if defect == "no_reduction":
            return v, f
        return cv, cf

    monkeypatch.setattr(impl, "_decimate", bad)
    with pytest.raises(fusemesh.SimplificationError):
        fusemesh.simplify(original, max_error=0.01, target_faces=12)


def test_distance_verifier_checks_triangle_interiors_and_both_directions():
    # Three tiny target triangles contain the source's vertices, but leave its
    # interior uncovered: a vertex-only distance test would incorrectly pass.
    large = np.array([[[0.0, 0, 0], [1, 0, 0], [0, 1, 0]]])
    tiny = np.array([np.array([p, p + [0.001, 0, 0], p + [0, 0.001, 0]]) for p in large[0]])
    with pytest.raises(fusemesh.SimplificationError, match="deviation"):
        impl._symmetric_bound(large, tiny, 0.01, [1000])
    with pytest.raises(fusemesh.SimplificationError, match="deviation"):
        impl._symmetric_bound(tiny, large, 0.01, [1000])


def test_sharp_edges_checked_even_when_surface_and_volume_pass(monkeypatch):
    original = refined_box()
    candidate = trimesh.creation.icosphere(subdivisions=2)
    candidate.apply_scale((original.volume / candidate.volume) ** (1 / 3))
    monkeypatch.setattr(
        impl, "_decimate", lambda *a: (np.asarray(candidate.vertices), np.asarray(candidate.faces))
    )
    # Need an input denser than the candidate. A generous surface tolerance
    # would allow it, but the cube's crease network has disappeared entirely.
    with pytest.raises(fusemesh.SimplificationError, match="sharp features"):
        fusemesh.simplify(refined_box(levels=3), max_error=1, target_faces=320)


@pytest.mark.parametrize("scale,offset", [(1e-4, 0), (1e4, 1e6)])
def test_world_space_tolerance_and_scale(scale, offset):
    original = refined_box()
    original.vertices = original.vertices * scale + offset
    result = fusemesh.simplify(original, max_error=0.01 * scale, target_faces=12)
    np.testing.assert_allclose(as_mesh(result).bounds, original.bounds, atol=1e-10 * scale)
    assert result.metadata["simplification"]["surface_error_bound"] <= 0.01 * scale


@pytest.mark.gpu
def test_reconstructed_union_simplifies_and_roundtrips(tmp_path):
    a, b = box(), box((0.5, 0.25, 0), (1.5, 1.25, 1))
    original = fusemesh.reconstruct([a, b], 0.12)
    result = fusemesh.simplify(original, max_error=0.03, target_faces=100)
    assert_closed(result)
    assert len(result.faces) < 0.75 * len(original.faces)
    path = tmp_path / "reduced.obj"
    fusemesh.save_mesh(path, result)
    v, f = fusemesh.load_mesh(path)
    assert len(f) == len(result.faces)
    np.testing.assert_allclose(v, result.vertices, atol=1e-7)


@pytest.mark.gpu
def test_file_demo_exports_original_and_simplified(tmp_path):
    source = tmp_path / "box.obj"
    trimesh.creation.box().export(source)
    original, reduced = tmp_path / "original.obj", tmp_path / "reduced.obj"
    script = Path(__file__).resolve().parents[1] / "examples" / "fuse_files.py"
    process = subprocess.run(
        [
            sys.executable,
            str(script),
            str(source),
            "--voxel-size",
            ".12",
            "--output",
            str(reduced),
            "--simplify-error",
            ".03",
            "--target-faces",
            "100",
            "--unsimplified-output",
            str(original),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    report = json.loads(process.stdout)
    _, f0 = fusemesh.load_mesh(original)
    _, f1 = fusemesh.load_mesh(reduced)
    assert len(f1) < len(f0)
    assert report["simplification"]["output_faces"] == len(f1)


def test_backend_cannot_remove_a_handle(monkeypatch):
    v, f = torus(nu=24, nv=12)
    candidate = trimesh.creation.icosphere(subdivisions=1)
    monkeypatch.setattr(
        impl, "_decimate", lambda *a: (np.asarray(candidate.vertices), np.asarray(candidate.faces))
    )
    with pytest.raises(fusemesh.SimplificationError, match="topology"):
        fusemesh.simplify((v, f), max_error=1, target_faces=80)
