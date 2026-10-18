"""File/object adapters and end-to-end file -> FUSE -> file acceptance."""

from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
import trimesh
import fusemesh
from geometry_helpers import box, combine, mesh_quality


def write_off(path, mesh, *, comments=False):
    v, f = mesh
    with path.open("w") as stream:
        stream.write(("OFF # triangle fixture\n\n" if comments else "OFF\n"))
        stream.write(f"{len(v)} {len(f)} 0\n")
        for vertex in v:
            stream.write(" ".join(format(x, ".17g") for x in vertex) + "\n")
        for face in f:
            stream.write(
                "3 " + " ".join(str(int(i)) for i in face) + (" # face\n" if comments else "\n")
            )
    return path


def assert_mesh_equal(a, b):
    np.testing.assert_array_equal(a[0], b[0])
    np.testing.assert_array_equal(a[1], b[1])


def test_load_off_preserves_arrays_winding_and_unused_vertices(tmp_path):
    v, f = box()
    # No merging, removal of unused vertices, winding fixes, or unit scaling.
    mesh = (np.vstack((v, v[0], [3, 4, 5])), f[:, ::-1])
    path = write_off(tmp_path / "cube.off", mesh, comments=True)
    loaded = fusemesh.load_mesh(path)
    assert_mesh_equal(loaded, mesh)
    assert loaded[0].dtype == np.float64 and loaded[1].dtype == np.uint64
    assert loaded[0].flags.c_contiguous and loaded[1].flags.c_contiguous
    assert loaded[0].flags.owndata and loaded[1].flags.owndata
    assert_mesh_equal(fusemesh.load_meshes(str(path))[0], mesh)
    assert len(fusemesh.load_meshes([path, path])) == 2


def test_off_header_inline_and_colors(tmp_path):
    v, f = box()
    path = write_off(tmp_path / "color.off", (v, f))
    lines = path.read_text().splitlines()
    lines[:2] = ["COFF " + lines[1]]
    for i in range(1, 9):
        lines[i] += " 255 0 0 255"
    path.write_text("\n".join(lines))
    assert_mesh_equal(fusemesh.load_mesh(path), (v, f))


@pytest.mark.parametrize(
    "text,message",
    [
        ("OFF\n3 1 0\n0 0 0\n1 0 0\n0 1 0\n3 0 1 -1\n", "outside"),
        ("OFF\n3 1 0\n0 0 0\n1 0 0\n0 1 0\n3 0 1 3\n", "outside"),
        ("OFF\n3 1 0\n0 0 0\n1 0 0\n", "truncated"),
        ("OFF\n3 1 0\n0 0 0\n1 0 0\n0 1 0\n3 0 1\n", "truncated"),
        ("OFF\n3 1 0\nnan 0 0\n1 0 0\n0 1 0\n3 0 1 2\n", "finite"),
        ("OFF\n-1 0 0\n", "nonnegative"),
        ("OFF\n0 0 0\n0 0 0\n", "beyond"),
        ("OFF\n4 1 0\n0 0 0\n1 0 0\n1 1 0\n0 1 0\n4 0 1 2 3\n", "triangulate"),
    ],
)
def test_bad_off_has_path_and_clear_error(tmp_path, text, message):
    path = tmp_path / "bad.off"
    path.write_text(text)
    with pytest.raises(ValueError, match=message) as exc:
        fusemesh.load_mesh(path)
    assert str(path) in str(exc.value)


def test_missing_file_and_empty_sources(tmp_path):
    with pytest.raises(FileNotFoundError):
        fusemesh.load_mesh(tmp_path / "missing.off")
    assert fusemesh.load_meshes([]) == []
    with pytest.raises(ValueError, match="no mesh"):
        fusemesh.load_meshes(trimesh.Scene())


def test_in_memory_mesh_owns_output(tmp_path):
    v, f = box()
    mesh = trimesh.Trimesh(v.copy(), f.copy(), process=False)
    result = fusemesh.load_mesh(mesh)
    mesh.vertices[:] += 9
    assert_mesh_equal(result, (v, f))
    result[0][:] = 42
    assert not np.all(mesh.vertices == 42)


def instanced_scene():
    v, f = box()
    scene = trimesh.Scene()
    scene.add_geometry(trimesh.Trimesh(v, f, process=False), geom_name="cube", node_name="a")
    transform = np.diag([-1.0, 1.0, 1.0, 1.0])
    transform[:3, 3] = [1.5, 0.25, 0.125]
    scene.graph.update(frame_to="b", matrix=transform, geometry="cube")
    expected = [(v, f), (v @ transform[:3, :3].T + transform[:3, 3], f[:, ::-1])]
    return scene, expected


def test_scene_instances_transforms_and_reflections(tmp_path):
    scene, expected = instanced_scene()
    meshes = fusemesh.load_meshes(scene)
    assert len(meshes) == 2
    for a, b in zip(meshes, expected):
        assert_mesh_equal(a, b)
    assert_mesh_equal((scene.geometry["cube"].vertices, scene.geometry["cube"].faces), box())
    with pytest.raises(ValueError, match="load_meshes"):
        fusemesh.load_mesh(scene)
    path = tmp_path / "instances.glb"
    path.write_bytes(scene.export(file_type="glb"))
    loaded = fusemesh.load_meshes(path)
    assert len(loaded) == 2
    # Loader node order can differ; compare world-space bounds.
    assert sorted(tuple(v.min(axis=0)) for v, _ in loaded) == sorted(
        tuple(v.min(axis=0)) for v, _ in expected
    )


def test_reject_nonmesh_and_singular_scene():
    scene = trimesh.Scene(trimesh.points.PointCloud([[0, 0, 0], [1, 1, 1]]))
    with pytest.raises(ValueError, match="not a triangle mesh"):
        fusemesh.load_meshes(scene)
    scene, _ = instanced_scene()
    transform = np.diag([0.0, 1.0, 1.0, 1.0])
    scene.graph.update(frame_to="b", matrix=transform, geometry="cube")
    with pytest.raises(ValueError, match="singular"):
        fusemesh.load_meshes(scene)


@pytest.mark.gpu
@pytest.mark.parametrize("backend", ["dense", "sparse"])
@pytest.mark.parametrize("case", ["single", "overlap", "contained", "cavity"])
def test_off_loaded_union_matches_direct_arrays(tmp_path, backend, case):
    meshes = [box()]
    components, euler = 1, 2
    if case == "overlap":
        meshes += [box((0.5, 0.25, 0.125), (1.5, 1.25, 1.125))]
    elif case == "contained":
        meshes += [box((0.25, 0.25, 0.25), (0.75, 0.75, 0.75))]
    elif case == "cavity":
        v, f = box((0.25, 0.25, 0.25), (0.75, 0.75, 0.75))
        meshes = [combine(box(), (v, f[:, ::-1]))]
        components, euler = 2, 4
    paths = [write_off(tmp_path / f"{i}.off", mesh) for i, mesh in enumerate(meshes)]
    direct = fusemesh.reconstruct(meshes, 0.1, backend=backend, return_field=True)
    loaded = fusemesh.reconstruct(
        paths[0] if len(paths) == 1 else paths, 0.1, backend=backend, return_field=True
    )
    np.testing.assert_array_equal(loaded.vertices, direct.vertices)
    np.testing.assert_array_equal(loaded.faces, direct.faces)
    np.testing.assert_array_equal(loaded.field, direct.field)
    mesh_quality(loaded, components, euler)
    # The explicit helper produces the same internal descriptors as direct paths.
    explicit = fusemesh.reconstruct(fusemesh.load_meshes(paths), 0.1, backend=backend)
    np.testing.assert_array_equal(explicit.faces, direct.faces)
    output = tmp_path / "result.off"
    fusemesh.save_mesh(output, loaded)
    v, f = fusemesh.load_mesh(output)
    np.testing.assert_allclose(v, loaded.vertices, rtol=0, atol=1e-9)
    np.testing.assert_array_equal(f, loaded.faces)


@pytest.mark.gpu
def test_scene_file_fuses_instances_instead_of_concatenated_parity(tmp_path):
    scene, expected = instanced_scene()
    path = tmp_path / "scene.glb"
    path.write_bytes(scene.export(file_type="glb"))
    direct = fusemesh.reconstruct(expected, 0.125, return_field=True)
    for source in (scene, path):
        result = fusemesh.reconstruct(source, 0.125, return_field=True)
        np.testing.assert_allclose(result.field, direct.field, atol=1e-14)
        np.testing.assert_allclose(result.vertices, direct.vertices, atol=1e-14)
        np.testing.assert_array_equal(result.faces, direct.faces)
        mesh_quality(result, 1, 2)


@pytest.mark.gpu
def test_open_file_is_not_implicitly_repaired(tmp_path):
    v, f = box()
    path = write_off(tmp_path / "open.off", (v, f[:-1]))
    assert len(fusemesh.load_mesh(path)[1]) == len(f) - 1
    with pytest.raises(ValueError, match="open"):
        fusemesh.reconstruct(path, 0.1)


@pytest.mark.gpu
def test_mixed_inputs_and_file_demo(tmp_path):
    meshes = [box(), box((0.5, 0.25, 0.125), (1.5, 1.25, 1.125))]
    paths = [write_off(tmp_path / f"{i}.off", mesh) for i, mesh in enumerate(meshes)]
    mixed = fusemesh.reconstruct([paths[0], trimesh.Trimesh(*meshes[1], process=False)], 0.1)
    direct = fusemesh.reconstruct(meshes, 0.1)
    np.testing.assert_array_equal(mixed.vertices, direct.vertices)
    np.testing.assert_array_equal(mixed.faces, direct.faces)
    output = tmp_path / "union.obj"
    demo = Path(__file__).resolve().parents[1] / "examples/fuse_files.py"
    result = subprocess.run(
        [
            sys.executable,
            str(demo),
            *[str(p) for p in paths],
            "--voxel-size",
            ".1",
            "--output",
            str(output),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    assert output.is_file() and "sparse_cuda" in result.stdout
    v, f = fusemesh.load_mesh(output)
    np.testing.assert_allclose(v, direct.vertices, rtol=0, atol=1e-8)
    np.testing.assert_array_equal(f, direct.faces)


def test_loading_without_visible_gpu(tmp_path):
    import os

    path = write_off(tmp_path / "cube.off", box())
    code = "import sys,fusemesh; v,f=fusemesh.load_mesh(sys.argv[1]); assert v.shape==(8,3) and f.shape==(12,3)"
    subprocess.run(
        [sys.executable, "-c", code, str(path)],
        env={**os.environ, "CUDA_VISIBLE_DEVICES": ""},
        check=True,
    )


@pytest.mark.gpu
def test_file_demo_validation_option(tmp_path):
    import json

    v, f = box()
    bad = f.copy()
    bad[0] = bad[0, ::-1]
    path = write_off(tmp_path / "inconsistent.off", (v, bad))
    output = tmp_path / "result.obj"
    demo = Path(__file__).resolve().parents[1] / "examples/fuse_files.py"
    command = [sys.executable, str(demo), str(path), "--voxel-size", ".1", "--output", str(output)]
    # Default/full rejects inconsistent winding before CUDA reconstruction.
    for flags in ([], ["--validation", "full"]):
        rejected = subprocess.run(command + flags, capture_output=True, text=True)
        assert rejected.returncode != 0 and "orientation" in rejected.stderr
        assert not output.exists()
    accepted = subprocess.run(
        command + ["--validation", "basic"], check=True, capture_output=True, text=True
    )
    assert json.loads(accepted.stdout)["validation"] == "basic"
    # Ray parity is orientation-independent for this closed surface, providing
    # a known reference while testing that the CLI actually bypassed validation.
    direct = fusemesh.reconstruct([(v, f)], 0.1)
    vertices, faces = fusemesh.load_mesh(output)
    np.testing.assert_allclose(vertices, direct.vertices, rtol=0, atol=1e-8)
    np.testing.assert_array_equal(faces, direct.faces)
