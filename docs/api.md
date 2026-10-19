# API reference

The Python entry points are `reconstruct`, `simplify`, `compare_surfaces`,
`load_mesh`, `load_meshes`, `save_mesh`, `build_info`, and `smoke_test`. `ReconstructionResult`,
`ResourceError`, and `SimplificationError` are public types. Names beginning
with `_` are private. Metadata may gain additional keys. Native consumers should
rebuild when changing library versions;
a cross-version C++ ABI is not promised.

## Reconstruction

```python
reconstruct(meshes, voxel_size, *, device=0, max_memory_bytes=512*1024**2,
            validation="full", return_field=False, backend="sparse")
```

| Argument | Contract |
| --- | --- |
| `meshes` | A local path, Trimesh/Scene, or iterable mixing these with `(vertices, faces)` pairs. A single array pair is passed as `[(vertices, faces)]`. An empty iterable is valid. |
| `voxel_size` | Finite positive scalar convertible to float, in input world units; booleans are rejected. One uniform spacing for the scene. |
| `device` | Nonnegative integer fitting signed 32 bits; booleans are rejected. Nonempty reconstruction checks the visible CUDA device range. |
| `max_memory_bytes` | Positive integer fitting uint64; booleans are rejected. Limits explicit FUSE device buffers, including NanoVDB, hierarchy data, scratch, and outputs. |
| `validation` | `"full"` checks finite/index structure plus closed, consistently oriented manifold topology. `"basic"` skips topology checks. Neither detects self-intersections or repairs defective meshes. |
| `return_field` | Boolean (including NumPy bool). Returns a dense diagnostic field when true, even with sparse reconstruction. Its host array must separately fit the memory limit. |
| `backend` | `"sparse"` (default) or `"dense"` reference. No automatic fallback or silent resolution change. |

Array vertices must be real numeric `(N,3)` data, and faces integer `(M,3)`
zero-based indices. Input geometry must represent valid solids even in basic
validation mode. Distinct input meshes are combined by solid union; nested
shells within one mesh use that mesh's even/odd interior rule. The full boundary
includes resolved cavity walls. Unused vertices do not enlarge the grid.

Python snapshots inputs into owned contiguous float64/uint64 buffers before
releasing the GIL. Callers must not mutate inputs while they are being loaded or
copied. Native computation is synchronous and restores the calling thread's
CUDA device on success or failure. Independent concurrent calls have independent
streams, buffers, and resource budgets; there is no cross-call geometry cache.

Empty scenes bypass CUDA. Structural checks still apply to supplied empty mesh
arrays. For nonempty scenes the CUDA runtime/driver must be available.

## Results and ownership

`ReconstructionResult` contains:

- `vertices`: float64 `(N,3)`, world coordinates.
- `faces`: uint64 `(M,3)`, zero-based indexed triangles.
- `field`: `None`, or float64 `(nz,ny,nx)` world-distance values.
- `metadata`: the mapping below.

Arrays are writable and retain their backing storage even after the result
container is deleted. Native reconstruction uses capsule-owned storage;
simplification returns NumPy-owned arrays. Sharing an array does not make it a
view of input arrays or another reconstruction. Face numbering is deterministic on the
validated platform; cross-platform bitwise identity is not part of the contract.

The following metadata describes reconstruction. Simplifying a reconstruction
updates output counts and adds `simplification`; other reconstruction diagnostics
are retained. Simplifying a standalone mesh does not populate grid or GPU fields.

| Metadata | Meaning |
| --- | --- |
| `origin`, `grid_shape`, `voxel_size` | World coordinate of sample zero, sample counts `(nx,ny,nz)`, and spacing. |
| `device`, `backend`, `validation`, `max_memory_bytes` | The effective requested settings; backend values are `sparse_cuda` and `dense_cuda`. |
| `input_mesh_count`, `input_triangle_count` | Number of expanded input descriptors (including empty meshes) and total input triangles. |
| `vertex_count`, `triangle_count` | Output counts. |
| `boundary_mode` | `"full"`. |
| `peak_device_bytes` | Peak simultaneous explicit FUSE CUDA-buffer allocation, including temporary buffers; not total GPU process memory. |
| `candidate_blocks`, `candidate_cells`, `active_samples`, `topology_bytes` | Sparse scheduling blocks, cells, unique samples, and serialized NanoVDB bytes. Dense mode reports dense cells/samples and zero blocks/topology bytes. |
| `timings_ms` | Synchronized wall times for `prepare`, `topology`, `upload`, `field`, `extraction`, and `download`. Diagnostic full-domain queries belong to `download`. |
| `topology_stages_ms` | CPU `candidates`, `keys`, and `index` subtimings **within** `timings_ms.topology`; do not add them again when summing stages. |

Total call latency also includes Python loading/copies, CUDA context startup,
and teardown. Measure end-to-end wall time for application performance. CUDA
context/runtime-managed storage, kernel-local storage, host BVHs/topology,
inputs, and host results are outside the explicit device-buffer limit. A
separate conservative candidate-count guard may reject work before allocation.

## Errors

- `TypeError`: unsupported array dtypes or argument types.
- `ValueError`: invalid options, numeric/index data, topology, transforms, or
  unavailable device index (when CUDA enumeration succeeds).
- `ResourceError` (a Python `MemoryError` subclass): budget/index limits or CUDA
  buffer allocation failure. Allocation-budget messages include requested,
  live, and limit byte counts.
- `RuntimeError`: CUDA runtime/driver failures or extraction/coordinate collapse.
- Filesystem errors retain their normal Python classes. File-parser errors may
  also originate from Trimesh; see [mesh loading](mesh-loading.md).

Exception classes and diagnostic meaning are the contract; complete message
wording is not a machine-readable interface. A failed call does not intentionally
retain device buffers. GPU hardware faults may require restarting the process.

## Version and native API

The root [`VERSION`](../VERSION) file is the sole editable release version.
Use `MAJOR.MINOR.PATCH`. Wheel/sdist metadata and CMake read it
at build time. `fusemesh.__version__`, `build_info()["version"]`, and native
`fuse::version()` report that compiled version. Change the file, rebuild, install
the new wheel, and restart processes that loaded the old extension.

`build_info()` also reports the pinned NanoVDB revision, CUDA compiler version,
compiled architectures and static runtime choice.
Import, loading, version queries, and build information do not initialize CUDA.

Native callers use `fuse/reconstruct.h` and `fuse/version.h`, linked through
`FUSE::fuse`. `MeshView` input pointers must remain valid and unchanged until
return. Results own their vectors. The native library exposes array descriptors,
not Trimesh or NanoVDB types. CMake package-version compatibility is exact.

## Numerical scope

Reconstruction uses uniform sampling and an indexed contour-disk extraction
policy. Voxel size controls resolution; it is not a guaranteed surface-distance
bound. Subvoxel features and unresolved contacts can change topology or disappear.
See [architecture and numerical behavior](FUSE-design.md) for sign conventions,
zero handling, and cell ambiguity decisions.

Reconstruction provides no adaptive resolution, exact Boolean, mesh repair, or
CPU fallback. Simplification has a separate distance-checking API relative to the
mesh supplied to it; it does not establish a bound on reconstruction error.

## Surface simplification

```python
simplify(mesh, *, max_error, target_faces=None, feature_angle=45.0,
         max_volume_change=0.01, max_checks=200000, on_failure="raise")
```

`fusemesh.simplify` accepts a reconstruction result, Trimesh, or array pair and
returns a new `ReconstructionResult` with independent arrays and metadata. It runs
on the CPU and always validates closed manifold topology. This requirement is
part of FUSE's quality contract, not a general restriction of quadric decimation.
`on_failure="keep"` handles rejected candidates; it never bypasses input checks.
`SimplificationError` is a `RuntimeError` subclass. See
[simplification controls and quality checks](simplification.md).

## Loading, export and diagnostics

- `load_mesh(source)` returns one `(vertices, faces)` pair and rejects sources
  containing multiple mesh instances.
- `load_meshes(sources)` accepts one source or an iterable and returns a list of
  pairs. Sources are paths, Trimesh objects, or Scene objects, not raw array pairs.
- `save_mesh(path, result_or_pair)` exports geometry using the filename extension.
  See [supported loading behavior](mesh-loading.md).
- `build_info()` returns compiled build information without initializing CUDA.
- `smoke_test(device=0)` performs a small CUDA check and returns float32 values
  `[3.0, -2.0, 0.5, 7.0]`.

## Surface quality

```python
compare_surfaces(originals, reconstructed, *, samples=10000, seed=0,
                 tolerance=None, voxel_size=None, batch_size=256,
                 degenerate_faces="error")
```

Returns a JSON-compatible dictionary of bidirectional and symmetric surface
distance estimates. See [reconstruction quality](quality.md) for input contracts,
union handling, metric definitions, CLI usage, and sampling limitations.
