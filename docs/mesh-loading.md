# Loading and fusing mesh files

Trimesh is installed automatically with the wheel and provides the file readers
and exporters. The dependency range is `>=4.12.2,<5`.

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
requires no GPU. `reconstruct` additionally accepts array pairs
and lists mixing array pairs, files, and Trimesh objects. A single array pair
uses `reconstruct([(vertices, faces)], ...)` syntax.

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
- Loading checks numeric/index structure. Reconstruction performs closed, consistently oriented manifold checks by default. It still does not
  detect geometric self-intersections. Open meshes may load, but full reconstruction validation
  rejects them. Basic validation does not make them supported solids.

## Formats and triangulation

Triangular ASCII OFF (including COFF headers/colors in `.off` files) uses a
strict reader guard. A streaming guard checks counts, finite coordinates,
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
checked within their text precision.

## Run from a checkout

```sh
python examples/fuse_files.py a.off b.off \
  --voxel-size 0.05 --backend sparse --memory-mib 512 --output build/union.obj
```

One input file also works.

### Validation when fusing

The file demo defaults to `--validation full`; the Python API defaults to
`validation="full"`. Full mode checks numeric/index validity, degenerate triangles,
and closed, consistently oriented edge- and vertex-manifold topology.

To skip reconstruction's topology checks explicitly:

```bash
python examples/fuse_files.py model.off --voxel-size 0.05 \
  --validation basic --memory-mib 512 --output build/union.obj
```

The equivalent API option is:

```python
result = fusemesh.reconstruct("model.off", voxel_size=0.05, validation="basic")
```

Prefer full validation for normal use. Basic mode retains numeric and index
checks, including rejection of zero-area triangles, but does not repair the mesh
or guarantee correct inside/outside
classification for open or nonmanifold inputs. A successful basic-mode run does
not establish that the input is a valid solid. Neither mode detects geometric
self-intersections. The selected mode is recorded in the result metadata and the
file demo's JSON output. This option controls fusion only; subsequent
simplification still performs its own topology checks.

The file demo defaults to a 16 GiB device-buffer limit; the library
default remains 512 MiB. Set `--memory-mib` explicitly for reproducible runs.
The command prints input/output counts, loading and
reconstruction wall time, backend stage timings, and peak FUSE device-buffer
bytes. Omit dense diagnostic fields for large workloads. Reconstruction
resamples even a single input mesh; this is not just a format conversion.

## Simplify and export both surfaces

```bash
python examples/fuse_files.py model.off --voxel-size 0.05 --memory-mib 512 \
  --simplify-error 0.01 --target-faces 5000 \
  --unsimplified-output original.obj --output reduced.obj
```

The original output is saved before simplification. The two output paths must
be different. `--target-faces` and `--unsimplified-output` require
`--simplify-error`. The command reports simplification counts, bounds, rejected
attempts, and elapsed time in its JSON output.

Simplification always requires a closed manifold mesh, even when reconstruction
used `--validation basic`. Its default failure policy raises an error; use
`--simplify-on-failure keep` to retain a shell when no reduced candidate passes.
This does not allow invalid input. See [simplification](simplification.md) for
volume, crease-angle, and verification-budget controls.
