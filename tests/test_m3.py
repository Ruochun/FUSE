"""M3 hierarchy correctness, version consistency, and stable API diagnostics."""

from importlib.metadata import version
from pathlib import Path
import gc

import numpy as np
import pytest
import fusemesh
from geometry_helpers import box, box_sdf, sample_positions, mesh_quality


def test_authoritative_version_matches_installed_package_and_native_library():
    expected = (Path(__file__).resolve().parents[1] / "VERSION").read_text().strip()
    assert (
        fusemesh.__version__ == version("fusemesh") == fusemesh.build_info()["version"] == expected
    )


@pytest.mark.parametrize(
    "kwargs,error,message",
    [
        ({"voxel_size": True}, TypeError, "voxel_size"),
        ({"return_field": "false"}, TypeError, "return_field"),
        ({"device": 2**31}, ValueError, "32-bit"),
        ({"device": True}, TypeError, "integer"),
        ({"max_memory_bytes": 2**64}, ValueError, "uint64"),
        ({"max_memory_bytes": False}, TypeError, "integer"),
    ],
)
def test_option_diagnostics_before_cuda(kwargs, error, message):
    options = dict(voxel_size=0.1)
    options.update(kwargs)
    with pytest.raises(error, match=message):
        fusemesh.reconstruct([], **options)


@pytest.mark.gpu
@pytest.mark.parametrize("backend", ["dense", "sparse"])
def test_shape_hierarchy_matches_analytic_union_and_input_permutation(backend):
    bounds = [
        ((x * 1.4, y * 1.4, 0), (x * 1.4 + 1, y * 1.4 + 1, 1)) for x in range(3) for y in range(3)
    ]
    meshes = [box(lo, hi) for lo, hi in bounds]
    out = fusemesh.reconstruct(meshes, 0.2, backend=backend, return_field=True)
    expected = np.minimum.reduce([box_sdf(sample_positions(out), lo, hi) for lo, hi in bounds])
    np.testing.assert_allclose(out.field, expected, atol=3e-8)
    mesh_quality(out, 9, 18)
    reordered = fusemesh.reconstruct(
        [meshes[i] for i in [8, 0, 4, 1, 6, 2, 3, 7, 5]], 0.2, backend=backend
    )
    np.testing.assert_array_equal(out.vertices, reordered.vertices)
    np.testing.assert_array_equal(out.faces, reordered.faces)
    assert out.metadata["input_mesh_count"] == 9
    assert out.metadata["input_triangle_count"] == 108


@pytest.mark.gpu
def test_hierarchy_never_prunes_deep_containing_shape():
    outer = box((0, 0, 0), (8, 8, 8))
    inner = [
        box((x, y, z), (x + 0.5, y + 0.5, z + 0.5)) for x in (2, 4) for y in (2, 4) for z in (2, 4)
    ]
    reference = fusemesh.reconstruct([outer], 0.25)
    for meshes in ([outer] + inner, inner + [outer]):
        result = fusemesh.reconstruct(meshes, 0.25)
        np.testing.assert_array_equal(result.vertices, reference.vertices)
        np.testing.assert_array_equal(result.faces, reference.faces)


@pytest.mark.gpu
def test_result_ownership_and_diagnostic_metadata():
    result = fusemesh.reconstruct([box()], 0.2, return_field=True, validation="basic")
    field = result.field
    expected = field.copy()
    assert result.metadata["validation"] == "basic"
    assert result.metadata["max_memory_bytes"] == 512 * 1024**2
    assert set(result.metadata["topology_stages_ms"]) == {"candidates", "keys", "index"}
    assert all(t >= 0 for t in result.metadata["topology_stages_ms"].values())
    assert (
        sum(result.metadata["topology_stages_ms"].values())
        <= result.metadata["timings_ms"]["topology"]
    )
    del result
    gc.collect()
    np.testing.assert_array_equal(field, expected)
