# FUSE — Fast Union Surface Extraction

C++/CUDA reconstruction of the full boundary of a union of closed triangle
meshes, with a NumPy-facing Python API. **M3 is implemented:** sparse NanoVDB
reconstruction with optimized topology construction and BVH queries, documented
API behavior, and repeatable performance checks. The dense backend remains a
correctness reference. M4 release automation and formatting are configured;
public PyPI wheels await release acceptance. See [release instructions](docs/releasing.md)
and [M4 validation](docs/m4-validation.md).

## Reconstruct a union

```python
import fusemesh

result = fusemesh.reconstruct(
    meshes=[(vertices_a, faces_a), (vertices_b, faces_b)],
    voxel_size=0.05,  # world units
    device=0,
    backend="sparse",  # default; "dense" selects the M1 reference
    max_memory_bytes=512 * 1024**2,
)
vertices = result.vertices  # float64 (N, 3), world coordinates
faces = result.faces        # uint64 (M, 3), indexed triangles
print(result.metadata)
```

Trimesh is installed automatically as the standard file-loading helper. You can
pass a local path, a list of paths, Trimesh/Scene objects, or mix these with array
pairs:

```python
result = fusemesh.reconstruct(["a.off", "b.off"], voxel_size=0.05)
fusemesh.save_mesh("union.obj", result)

# Explicit loading returns our existing (vertices, faces) array pairs:
meshes = fusemesh.load_meshes(["a.off", "b.off"])
result = fusemesh.reconstruct(meshes, voxel_size=0.05)
```

OFF faces must already be triangles. Loading preserves units and does not
repair or weld meshes. Scene instances are separate union inputs with their
world transforms applied; disconnected shells within one geometry remain
one solid. See [mesh loading](docs/mesh-loading.md) for formats and limitations.

Inputs are copied from real numeric vertex arrays and integer face arrays.
The default `validation="full"` checks structural validity, closed edges,
consistent face winding, and manifold vertex links. It does **not** detect
self-intersections: inputs must already represent valid solids. `"basic"` skips
topology checks but has the same supported-input requirements.

Separate inputs can overlap or contain one another. Disconnected components
and enclosed cavity walls are preserved when resolved by the grid. There is no
repair, smoothing, or automatic deletion of small components. Subvoxel walls,
gaps, and components may disappear; voxel spacing is not an error guarantee.
Degenerate contacts and unresolved cell-interior topology are not guaranteed
manifold. See the [API contract](docs/api.md) and [M1 numerical and extraction policy](docs/decisions/0002-m1-dense-baseline.md).

The memory limit covers FUSE device buffers, including scratch and output, but
not CUDA context memory or host arrays. Sparse candidate topology is currently
built on the CPU; field queries and extraction run on the GPU. A scene that exceeds the limit raises
`fusemesh.ResourceError` (a `MemoryError`). Increase voxel spacing or the budget.
Calls are synchronous and release the GIL during native computation. Empty
input returns empty arrays without initializing CUDA.

For diagnostics, `return_field=True` also returns a float64 dense union field
in `(z,y,x)` order, in world-distance units. `metadata['origin']` is sample zero;
`metadata['grid_shape']` is `(x,y,z)`. The result records counts, peak allocated
device bytes, active sample/candidate counts, NanoVDB bytes, and stage timings.
The diagnostic field queries absent samples explicitly, including deep interiors;
absence in NanoVDB never means exterior. Its host array must separately fit
`max_memory_bytes`, so omit `return_field` for large sparse domains.
`backend="dense"` retains the M1 reference; both backends use the same sampling
and extraction policy. See [M2 decisions](docs/decisions/0003-m2-sparse.md). `build_info()` and `smoke_test()` remain
available for build and GPU diagnostics.

## Build and install

Local validation covers CPython 3.9–3.14 on Linux x86-64, CUDA toolkit 12.8,
GCC 11, and an RTX 4060; all six installed wheels pass the reconstruction suite.
See [M4 validation](docs/m4-validation.md). The conservative Linux
driver floor is 570.124.06. Wheels compile SM 80/86/89/90 and compute-90 PTX; only
SM 89 has been executed. Other platforms/drivers are not yet validated.

```sh
git submodule update --init --recursive
python -m pip install build
python -m build
python -m pip install --upgrade dist/fusemesh-0.2.0-*.whl
python -c 'import fusemesh; print(fusemesh.build_info()); print(fusemesh.smoke_test())'
```

When rebuilding without changing the version, use `--force-reinstall --no-deps`
instead of `--upgrade`; pip does not compare wheel contents. Restart any Python
process that has already loaded the extension.

The root [`VERSION`](VERSION) file is the only release-version source. Change
it, rebuild, and reinstall to bump the wheel, Python, and native versions together.

A runnable OBJ demo is available with `python examples/union_demo.py`.
For your own files:

```sh
python examples/fuse_files.py a.off b.off --voxel-size 0.05 --output build/union.obj
```

The intended public command is `pip install fusemesh`, without a CUDA extra.
No package has been published yet. Local wheels statically link the CUDA runtime
and require an NVIDIA GPU/driver, but no toolkit or compiler at runtime. Source
builds require CUDA 12.8, a C++ compiler, and CMake 3.24+ (the Python build can
install CMake). Source archives include the pinned NanoVDB sources. No OpenVDB
CPU library, TBB, upstream Python bindings, or viewers are required.

## Native library and tests

```sh
cmake -S . -B build/native -DFUSE_BUILD_PYTHON=OFF -DFUSE_BUILD_TESTS=ON \
  -DCMAKE_BUILD_TYPE=Release
cmake --build build/native -j2
ctest --test-dir build/native --output-on-failure
python -m pip install pytest
python -m pytest
```

The installed `FUSE::fuse` CMake target exposes
[`fuse/reconstruct.h`](include/fuse/reconstruct.h) independently of Python.
Native development consumers need the CUDA toolkit to link the static library.
Python tests require the installed wheel and a GPU; `pytest -m 'not gpu'` runs
input/metadata checks on machines without a visible GPU.

The tests compare GPU fields with analytic solids and an independent NumPy
solid-angle/distance reference. They check closure, winding, components, Euler
characteristics, volume, surface error, resource limits, and sampling losses.
Native tests exercise all cube sign cases and shared-face extraction. See
[M3 validation](docs/m3-validation.md), [M2 validation](docs/m2-validation.md), [M1 validation](docs/m1-validation.md), and [M0 validation](docs/m0-validation.md).

See [the design](docs/FUSE-design.md),
[distribution requirements](docs/distribution-requirements.md),
[M0 decisions](docs/decisions/0001-m0-build.md), and
[NanoVDB inventory](docs/nanovdb-inventory.md).
FUSE uses BSD-3-Clause; third-party terms are in `licenses/` and wheel metadata.
