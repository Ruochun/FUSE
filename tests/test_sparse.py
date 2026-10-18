"""M2 equivalence, sparsity, inactive-interior, and block-seam acceptance."""

import numpy as np
import pytest
import fusemesh
from geometry_helpers import box, combine, sphere, torus, mesh_quality


def equivalent(meshes, h, **kwargs):
    sparse = fusemesh.reconstruct(meshes, h, backend="sparse", **kwargs)
    dense = fusemesh.reconstruct(meshes, h, backend="dense", **kwargs)
    # Canonical global key order is retained across the sparse/dense backends.
    np.testing.assert_array_equal(sparse.faces, dense.faces)
    np.testing.assert_array_equal(sparse.vertices, dense.vertices)
    if kwargs.get("return_field"):
        np.testing.assert_array_equal(sparse.field, dense.field)
    assert sparse.metadata["backend"] == "sparse_cuda"
    assert sparse.metadata["topology_bytes"] > 0
    return sparse, dense


@pytest.mark.gpu
@pytest.mark.parametrize("kind", ["overlap", "rotated", "cavity", "torus", "tiny", "ties"])
def test_dense_sparse_fields_and_indexed_geometry(kind):
    h = 0.12
    if kind == "overlap":
        meshes = [sphere((0, 0, 0), 1, 2), sphere((0.7, 0.2, 0.3), 0.8, 2)]
    elif kind == "rotated":
        v, f = box((-2, -0.25, -0.3), (2, 0.25, 0.3))
        a, b = 0.61, 0.43
        rz = np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]])
        ry = np.array([[np.cos(b), 0, np.sin(b)], [0, 1, 0], [-np.sin(b), 0, np.cos(b)]])
        meshes = [(v @ rz @ ry, f)]
    elif kind == "cavity":
        v, f = box((0.6, 0.6, 0.6), (1.4, 1.4, 1.4))
        meshes = [combine(box((0, 0, 0), (2, 2, 2)), (v, f[:, ::-1]))]
    elif kind == "torus":
        meshes = [torus()]
    elif kind == "tiny":
        meshes = [box(), box((1.2, 0.21, 0.21), (1.22, 0.23, 0.23))]
    else:
        meshes = [sphere(levels=0)]
        h = 0.4
    equivalent(meshes, h, return_field=True)


@pytest.mark.gpu
@pytest.mark.parametrize("phase", [0, 0.5, 1 - 1e-10])
def test_shared_faces_across_all_block_axes(phase):
    # Surface coordinates deliberately land on/near 8-cell block boundaries.
    # The anchor fixes the global transform independently of the tested shape.
    h = 0.1
    shift = (5.5 + phase) * h
    meshes = [box((0, 0, 0), (0.2, 0.2, 0.2)), box((shift, shift, shift), (2.35 + phase * h,) * 3)]
    sparse, _ = equivalent(meshes, h, return_field=True)
    mesh_quality(sparse, 2, 4)  # all edges twice, opposite winding; Euler/components


@pytest.mark.gpu
def test_deeply_contained_surface_suppressed_without_interior_allocation():
    outer = box((0, 0, 0), (16, 16, 16))
    inner = box((7, 7, 7), (9, 9, 9))
    baseline = fusemesh.reconstruct([outer], 0.25)
    sparse, dense = equivalent([outer, inner], 0.25, return_field=True)
    np.testing.assert_array_equal(sparse.faces, baseline.faces)
    np.testing.assert_array_equal(sparse.vertices, baseline.vertices)
    assert sparse.field[34, 34, 34] < 0  # deep absent region is still inside
    assert sparse.metadata["active_samples"] < np.prod(sparse.metadata["grid_shape"]) * 0.8
    mesh_quality(sparse, 1, 2)


@pytest.mark.gpu
def test_large_empty_domain_crosses_nanovdb_root_tiles_without_dense_allocation():
    # ~4.4 trillion dense samples; surface work is only two small cubes.
    distant = [box(), box((4096, 4096, 4096), (4097, 4097, 4097))]
    sparse = fusemesh.reconstruct(distant, 0.25, max_memory_bytes=16 * 1024**2)
    close = fusemesh.reconstruct(
        [box(), box((8, 8, 8), (9, 9, 9))], 0.25, max_memory_bytes=16 * 1024**2
    )
    mesh_quality(sparse, 2, 4)
    assert sparse.metadata["active_samples"] == close.metadata["active_samples"]
    assert sparse.metadata["candidate_cells"] == close.metadata["candidate_cells"]
    assert sparse.metadata["peak_device_bytes"] < 16 * 1024**2
    assert len(sparse.faces) == len(close.faces)
    with pytest.raises(fusemesh.ResourceError, match="grid"):
        fusemesh.reconstruct(distant, 0.25, backend="dense")
    with pytest.raises(fusemesh.ResourceError, match="diagnostic"):
        fusemesh.reconstruct(distant, 0.25, return_field=True)


@pytest.mark.gpu
def test_sparse_memory_reduction_and_budget_that_dense_cannot_fit():
    meshes = [box((0, 0, 0), (4, 4, 4))]
    sparse, dense = equivalent(meshes, 0.05)
    peak = sparse.metadata["peak_device_bytes"]
    assert peak < dense.metadata["peak_device_bytes"] * 0.75
    constrained = fusemesh.reconstruct(meshes, 0.05, max_memory_bytes=peak)
    np.testing.assert_array_equal(constrained.faces, sparse.faces)
    with pytest.raises(fusemesh.ResourceError, match="max_memory_bytes"):
        fusemesh.reconstruct(meshes, 0.05, backend="dense", max_memory_bytes=peak)


def test_backend_validation():
    with pytest.raises(ValueError, match="backend"):
        fusemesh.reconstruct([], 1, backend="automatic")
    for backend in ("dense", "sparse"):
        result = fusemesh.reconstruct([], 1, backend=backend)
        assert result.metadata["backend"] == backend + "_cuda"
        assert result.metadata["active_samples"] == 0


@pytest.mark.gpu
@pytest.mark.parametrize("seed", range(6))
def test_conservative_candidates_for_oblique_slivers(seed):
    # SAT pruning must retain all crossing cells even for long, oblique,
    # almost-subvoxel sheets. Dense evaluation is an independent coverage oracle.
    rng = np.random.default_rng(seed)
    rotation, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    v, f = box((-2, -0.3, -0.045), (2, 0.3, 0.045))
    meshes = [(v @ rotation, f), sphere((1.6, -0.7, 0.4), 0.12, 1)]
    equivalent(meshes, 0.13, return_field=True)


@pytest.mark.gpu
def test_subvoxel_component_still_allocates_candidates():
    base = fusemesh.reconstruct([box()], 0.1)
    tiny = box((4.01, 4.01, 4.01), (4.02, 4.02, 4.02))
    out = fusemesh.reconstruct([box(), tiny], 0.1)
    assert out.metadata["candidate_blocks"] == base.metadata["candidate_blocks"] + 1
    np.testing.assert_array_equal(out.faces, base.faces)
    np.testing.assert_array_equal(out.vertices, base.vertices)
    # The tiny input has candidate coverage but no negative sample at this h.
    mesh_quality(out, 1, 2)
