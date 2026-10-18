# FUSE 0.2 API contract

The Python entry points are `reconstruct`, `load_mesh`, `load_meshes`,
`save_mesh`, `build_info`, and `smoke_test`. `ReconstructionResult` and
`ResourceError` are public types. Names beginning with `_` are private.
The 0.2 line preserves these signatures and result meanings; metadata may gain
additional keys. Native consumers should rebuild when changing library versions;
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

Arrays are writable and retain their native backing storage even after the
result container is deleted. Sharing an array does not make it a view of input
arrays or another reconstruction. Face numbering is deterministic on the
validated platform; cross-platform bitwise identity is not part of the contract.

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
Use `MAJOR.MINOR.PATCH` (currently `0.2.0`). Wheel/sdist metadata and CMake read it
at build time. `fusemesh.__version__`, `build_info()["version"]`, and native
`fuse::version()` report that compiled version. Change the file, rebuild, install
the new wheel, and restart processes that loaded the old extension.

`build_info()` also reports the pinned NanoVDB revision, CUDA compiler version,
compiled architectures, static runtime choice, and implementation milestone.
Import, loading, version queries, and build information do not initialize CUDA.

Native callers use `fuse/reconstruct.h` and `fuse/version.h`, linked through
`FUSE::fuse`. `MeshView` input pointers must remain valid and unchanged until
return. Results own their vectors. The native library exposes array descriptors,
not Trimesh or NanoVDB types. CMake package-version compatibility is exact.

## Numerical scope

The supported inputs, sign/zero conventions, face ambiguity decisions, and
subvoxel/degenerate-contact limitations remain those documented in
[M1 decisions](decisions/0002-m1-dense-baseline.md) and
[M2 decisions](decisions/0003-m2-sparse.md). Version 0.2 does not add adaptive
resolution, exact Booleans, mesh repair, or an error-bound API. Public PyPI
publishing and full GPU/platform qualification require the M4 acceptance gates;
see [release instructions](releasing.md) and [validation](m4-validation.md).
