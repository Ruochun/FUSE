# Loading and fusing mesh files

Trimesh is a required runtime dependency starting with `fusemesh 0.0.1.dev2`.
A normal wheel installation installs it automatically. The tested version is
Trimesh 4.12.2; the current dependency range is `>=4.12.2,<5`.

```python
import fusemesh

# One file: reconstruct the surface at the requested resolution.
result = fusemesh.reconstruct("model.off", voxel_size=0.05)

# Multiple files: reconstruct their solid union.
result = fusemesh.reconstruct(["a.off", "b.off"], voxel_size=0.05)
fusemesh.save_mesh("union.obj", result)

# Or load first to inspect the arrays before reconstruction.
vertices, faces = fusemesh.load_mesh("model.off")
meshes = fusemesh.load_meshes(["a.off", "b.off"])
result = fusemesh.reconstruct(meshes, voxel_size=0.05)
```

`load_mesh` returns exactly one `(vertices, faces)` pair; it rejects sources with
multiple instances and directs callers to `load_meshes`. `load_meshes` accepts
one source or an iterable of sources and returns a list of pairs. Sources can
be local file paths, `trimesh.Trimesh`, or `trimesh.Scene` objects. Returned arrays
are owned contiguous float64 coordinates and uint64 triangle indices. Loading
requires no GPU. `reconstruct` additionally accepts the original array pairs
and lists mixing array pairs, files, and Trimesh objects. A single array pair
still uses the original `reconstruct([(vertices, faces)], ...)` syntax.

## Geometry semantics

- Coordinates retain their file units; `voxel_size` uses those units. No automatic
  normalization, centering, welding, winding repair, hole filling, or component
  deletion is performed. Colors, textures, and materials are not passed to FUSE.
- Every scene mesh instance becomes a separate union input with its world
  transform applied. Reflections reverse face winding. Invalid/nonfinite or
  singular transforms and non-mesh scene nodes are rejected.
- Disconnected shells **within one geometry** remain together. This preserves
  a cavity represented by nested shells. Separate geometries are interpreted
  as separate solids, so an inner solid does not create a cavity in its outer
  solid. Trimesh's file parser determines geometry grouping; material groups
  are not automatically assembled into a watertight solid.
- Multiple overlapping solids stored as disconnected shells of a single
  geometry are not automatically detected or split. Supply them as separate
  files or mesh objects when they are intended as separate union inputs.
- Loading checks numeric/index structure. Reconstruction performs the existing
  closed, consistently oriented manifold checks by default. It still does not
  detect geometric self-intersections. Open meshes may load, but reconstruction
  rejects them; loading is not a repair operation.

## Formats and triangulation

Triangular ASCII OFF (including COFF headers/colors in `.off` files) is the
initial strict format. A streaming guard checks counts, finite coordinates,
face arity, truncation, and indices before Trimesh reads the geometry. OFF
polygons are rejected with a request to triangulate first. Trimesh's automatic
fan triangulation is not sufficient to guarantee correct concave polygons.

Other local formats are delegated to Trimesh with `process=False` and
`maintain_order=True`; they follow that format reader's indexing, grouping,
and triangulation behavior. Triangulated OBJ and instanced GLB are tested.
Some other formats require Trimesh's optional dependencies, which are not
installed by FUSE. STL triangle soups usually need explicit vertex welding
before FUSE's indexed manifold validation can pass. No general format-support
or arbitrary polygon-triangulation guarantee is implied.

`save_mesh(path, result_or_pair)` exports geometry through Trimesh; the filename
extension chooses the format. Text export precision is format-dependent, so
export/reload is not promised to be bitwise exact. OFF/OBJ round trips are
checked within their text precision. The CUDA reconstruction code is unchanged.

## File-based test command

```sh
python examples/fuse_files.py a.off b.off \
  --voxel-size 0.05 --backend sparse --memory-mib 512 --output build/union.obj
```

One input file also works. Add `--validation basic` to skip reconstruction's
topology checks; `--validation full` is the default. Basic mode retains numeric
and index checks and does not repair the mesh or guarantee correct inside/outside
classification for open or nonmanifold inputs. The selected mode is recorded in
the JSON output.

The command prints input/output counts, loading and
reconstruction wall time, backend stage timings, and peak FUSE device-buffer
bytes. Omit dense diagnostic fields for large workloads. Reconstruction
resamples even a single input mesh; this is not just a format conversion.

## Validation

On 18 October 2026, the `0.0.1.dev2` Linux CPython 3.13 wheel was installed
into the runtime environment with normal dependency resolution, which installed
Trimesh 4.12.2 automatically. **112 tests passed** with CUDA toolkit directories
hidden: 26 I/O tests plus all 86 existing tests, on the RTX 4060. Native CUDA
code is unchanged from M2. The full Python/platform release matrix remains
outstanding.

`tests/test_io.py` covers independent OFF fixtures, comments/colors, malformed
and polygonal faces, owned arrays, preserved unused vertices and winding,
scene instances/reflections, rejected non-mesh/singular scenes, and loading
without a visible GPU. GPU tests compare file-loaded single/overlapping/
contained/cavity meshes against direct array input on **both** sparse and dense
backends, including exact field and indexed-output comparisons. A GLB scene
checks that overlapping instances are fused as separate solids, not concatenated
parity shells. A subprocess exercises the file demo and OBJ reload.

These are generated fixtures, not validation of a user-provided mesh. No user
mesh file was supplied for this change. The existing M1/M2 tests remain part of
the full acceptance suite. See [M2 validation](m2-validation.md) for native
reconstruction coverage and numerical limitations.
