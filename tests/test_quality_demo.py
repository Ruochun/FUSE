"""File comparison must not reconstruct or require manifold solids."""

import json
from pathlib import Path
import runpy
import sys

import numpy as np
import pytest
import trimesh

import fusemesh


@pytest.mark.parametrize("drop", [False, True])
@pytest.mark.parametrize("voxel_size", [None, 0.1])
def test_nonmanifold_file_comparison_without_reconstruction(
    tmp_path, monkeypatch, voxel_size, drop
):
    # Three nondegenerate triangles share one edge; the mesh is also open.
    vertices = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, -1, 0], [0.5, 2, 0]])
    faces = np.array([[0, 1, 2], [1, 0, 3], [0, 1, 4]])
    if drop:
        faces = np.vstack((faces, [[0, 0, 1]]))
    original = trimesh.Trimesh(vertices, faces, process=False)
    output = original.copy()
    output.apply_translation([0, 0, 0.25])
    paths = [tmp_path / "original.off", tmp_path / "output.off"]
    for path, mesh in zip(paths, [original, output]):
        mesh.export(path)
    snapshots = [path.read_bytes() for path in paths]

    def forbidden(*args, **kwargs):
        pytest.fail("File comparison must not reconstruct or export meshes")

    monkeypatch.setattr(fusemesh, "reconstruct", forbidden)
    monkeypatch.setattr(fusemesh, "save_mesh", forbidden)
    monkeypatch.setitem(sys.modules, "manifold3d", None)
    script = Path(__file__).resolve().parents[1] / "examples" / "quality_demo.py"
    report_dir = tmp_path / "report"
    argv = [
        str(script),
        "--inputs",
        *map(str, paths),
        "--samples",
        "100",
        "--tolerance",
        "0.3",
        "--output-dir",
        str(report_dir),
    ]
    if drop:
        argv += ["--degenerate-faces", "drop"]
    if voxel_size is not None:
        argv += ["--voxel-size", str(voxel_size)]
    monkeypatch.setattr(sys, "argv", argv)
    runpy.run_path(str(script), run_name="__main__")

    report = json.loads((report_dir / "quality.json").read_text())
    assert report["cleaning"]["originals"][0]["removed_faces"] == int(drop)
    assert report["cleaning"]["reconstructed"]["removed_faces"] == int(drop)
    assert report["symmetric"]["rms"] == pytest.approx(0.25)
    assert report["symmetric"]["f_score"] == 1
    assert report["reference"]["mode"] == "surface"
    assert not report["demo"]["reconstruction_performed"]
    assert report["demo"]["mode"] == "file_comparison"
    assert [path.read_bytes() for path in paths] == snapshots
    assert list(report_dir.iterdir()) == [report_dir / "quality.json"]
    if voxel_size is None:
        assert "symmetric_in_voxels" not in report
    else:
        assert report["symmetric_in_voxels"]["rms"] == pytest.approx(2.5)
