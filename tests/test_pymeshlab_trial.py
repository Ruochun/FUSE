"""Optional subprocess tests; set FUSE_PYMESHLAB_PYTHON for a separate environment."""

import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest
import trimesh

ROOT = Path(__file__).resolve().parents[1]
PYTHON = os.environ.get("FUSE_PYMESHLAB_PYTHON", sys.executable)
pytestmark = pytest.mark.skipif(
    not os.environ.get("FUSE_PYMESHLAB_PYTHON") and importlib.util.find_spec("pymeshlab") is None,
    reason="optional PyMeshLab dependency not installed",
)


def run_trial(source, output, report):
    return subprocess.run(
        [
            PYTHON,
            str(ROOT / "benchmarks/trial_pymeshlab.py"),
            str(source),
            str(output),
            "--ratio",
            "0.5",
            "--report",
            str(report),
        ],
        capture_output=True,
        text=True,
    )


@pytest.mark.parametrize("suffix", [".ply", ".obj", ".off", ".stl"])
def test_formats_replace_existing_output(tmp_path, suffix):
    mesh = trimesh.creation.icosphere(subdivisions=2)
    source, output, report = (
        tmp_path / "input.ply",
        tmp_path / ("output" + suffix),
        tmp_path / "report.json",
    )
    mesh.export(source)
    original = source.read_bytes()
    output.write_bytes(b"previous output")
    result = run_trial(source, output, report)
    assert result.returncode == 0, result.stdout + result.stderr
    assert source.read_bytes() == original
    data = json.loads(report.read_text())
    assert data["backend"] == "pymeshlab_quadric_edge_collapse"
    assert data["status"] == "simplified_unvalidated"
    reduced = trimesh.load(output, process=True)
    assert 0 < len(reduced.faces) < len(mesh.faces)
    assert reduced.is_watertight


def test_failed_load_preserves_output(tmp_path):
    source, output, report = tmp_path / "bad.ply", tmp_path / "output.ply", tmp_path / "report.json"
    source.write_text("not a mesh")
    output.write_bytes(b"previous output")
    result = run_trial(source, output, report)
    assert result.returncode != 0
    assert output.read_bytes() == b"previous output"
    assert json.loads(report.read_text())["status"] == "failed"


def test_nonmanifold_vertex_can_be_processed(tmp_path):
    a = trimesh.creation.box().subdivide().subdivide()
    b = a.copy()
    b.apply_translation([1, 1, 1])
    mesh = trimesh.util.concatenate([a, b])
    mesh.merge_vertices()
    source, output, report = (
        tmp_path / "input.ply",
        tmp_path / "output.ply",
        tmp_path / "report.json",
    )
    mesh.export(source)
    result = run_trial(source, output, report)
    assert result.returncode == 0, result.stdout + result.stderr
    data = json.loads(report.read_text())
    assert data["output_faces"] < data["input_faces"]
    assert data["status"] == "simplified_unvalidated"
