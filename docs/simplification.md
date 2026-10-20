# Surface mesh simplification

`fusemesh.simplify` is a CPU utility for reconstructed or existing meshes. It reduces
triangles on planar and gently curved regions; output is still a triangle mesh.
The voxel grid, reconstruction algorithm, and default reconstruction output are
unchanged. Sampling resolution remains uniform.

```python
import fusemesh

original = fusemesh.reconstruct(["a.off", "b.obj"], voxel_size=0.05)
reduced = fusemesh.simplify(original, max_error=0.01, target_faces=5000)
fusemesh.save_mesh("original.obj", original)
fusemesh.save_mesh("reduced.obj", reduced)
print(reduced.metadata["simplification"])
```

To simplify an existing closed mesh without GPU reconstruction:

```python
mesh = fusemesh.load_mesh("model.obj")
reduced = fusemesh.simplify(mesh, max_error=0.01, target_faces=5000)
fusemesh.save_mesh("reduced.obj", reduced)
```

Or fuse and simplify using the file demo:

```bash
python examples/fuse_files.py a.off b.obj --voxel-size 0.05 \
  --simplify-error 0.01 --target-faces 5000 \
  --unsimplified-output original.obj --output reduced.obj
```

All tolerances are in the mesh's coordinate units. The example numbers are not
universal defaults; choose them for the size and features of your geometry.

## API and failure behavior

`simplify(mesh, *, max_error, target_faces=None, feature_angle=45.0,
max_volume_change=0.01, max_checks=200000, on_failure="raise")`

- `mesh`: a `ReconstructionResult`, a Trimesh, or a `(vertices, faces)` pair.
  Use `load_mesh` first for a standalone file. Input must be a closed,
  consistently oriented vertex- and edge-manifold triangle mesh. No repair,
  welding, or winding correction is performed. This validation always runs,
  regardless of any earlier reconstruction validation setting. It is a FUSE
  wrapper requirement supporting the topology/volume checks; quadric decimation
  itself is not limited to closed manifold meshes. Open surfaces and arbitrary
  nonmanifold inputs are not supported by this wrapper.
- `max_error`: positive maximum surface and sharp-edge deviation in world units,
  **relative to the mesh passed to this function**, not the original union inputs.
  This does not remove reconstruction error already caused by voxel sampling.
- `target_faces`: a soft goal; defaults to half the input count. It must be at
  least four. Counts are allocated proportionally across connected shells, with
  at least four per shell. The backend can stop above or below the goal.
- `feature_angle`: edges with a dihedral angle at least this many degrees form
  the protected crease network (default 45; valid range is strictly between 0
  and 180). Classification includes a 1e-10
  tolerance in the normal dot product to stabilize threshold-angle creases.
  Feature preservation allows displacement up to `max_error`; it is not an exact
  corner/edge lock or an angular-error guarantee.
- `max_volume_change`: positive maximum relative signed-volume change per shell
  (default 0.01, or 1%). A reversed shell is always rejected. Cavity shells keep
  their inward orientation; disconnected shells are simplified independently.
- `max_checks`: maximum simplex distance checks per candidate, shared by both
  surface directions and both crease directions. Increase this for large meshes
  or tight tolerances. The demo exposes `--simplify-max-checks`. This is a work
  limit, not a wall-clock or host-memory limit.
- `on_failure`: the default raises `SimplificationError` when no reduced candidate
  passes. `"keep"` retains the affected original shell and records `status="kept"`.
  Input errors always raise. The original input is never modified. The demo
  exposes this as `--simplify-on-failure keep`.

For each shell, up to six candidates are tried, moving the requested count halfway
back toward the original after each rejection. These retries do not establish
an optimal triangle count. An already small mesh or a goal at least as large as
its count returns unchanged geometry without invoking the backend.

The result has owned float64 vertices, uint64 faces, and copied metadata. If
present, the original diagnostic voxel field is copied unchanged; it remains a
record of reconstruction, not a field recomputed from the reduced surface.
Reconstruction timing/device-memory metadata also remains historical.
`metadata["simplification"]` records backend/version, input/output counts,
requested controls, whether the count goal was reached, conservative distance
bounds, sampled distances per shell, relative volume changes, rejected attempts,
per-shell status/topology, and elapsed time.

## What the checks establish

Candidates must retain one closed connected shell, consistent winding, manifold
vertex links and edges, and the original Euler characteristic. Together these
preserve the combinatorial genus of each oriented shell. Degenerate and duplicate
triangles are rejected. Signed volume is checked per shell so a small component
cannot be hidden by the volume of a large one.

Surface deviation is checked in **both directions**, including triangle interiors.
For each source triangle, the verifier obtains two upper bounds:

1. Maximum distance of its vertices to a single target triangle. Distance to a
   convex triangle is convex, so this bounds every point of the source triangle.
2. Distance of its centroid to the target surface plus its maximum vertex radius,
   using the 1-Lipschitz property of distance to a set.

If neither bound fits, the source triangle is bisected along its longest edge.
A sampled distance above tolerance rejects the candidate immediately. Unresolved
work at the budget limit also rejects it. The same process on line segments
checks the sharp-edge networks in both directions. Thus vertex-only or random
sample estimates are not presented as a surface-distance bound. Reported upper
bounds can be loose and often lie near the requested tolerance even when actual
error is much smaller.

Calculations use normalized float64 coordinates with numerical margins, **not
exact arithmetic or interval-certified predicates**. Tolerances below 1e-12 of
the mesh extent, with an additional allowance for large coordinate translations,
are rejected. Very small/degenerate triangles may also be rejected.

Self-intersections and shell nesting changes are **not detected**. Combinatorial
topology and distance/volume checks are not an ambient-isotopy guarantee. In
particular, use an error tolerance well below important gaps and wall thicknesses;
these checks do not prove collision-free separation. Inputs must already represent
valid non-self-intersecting solids, as required by reconstruction. Textures,
materials, UVs and other per-vertex/per-face attributes are not transferred.

## Backend and dependencies

The implementation uses the array API from
[fast-simplification](https://github.com/pyvista/fast-simplification), the backend
behind Trimesh's quadric-decimation helper. Calling its array API avoids the
Trimesh wrapper's automatic output processing. FUSE checks the unmodified
candidate independently. Calls through FUSE serialize the upstream backend's
global scratch state. Direct concurrent calls to that backend outside FUSE are
not covered by this lock.

Dependency versions are pinned to tested wheels: fast-simplification 0.1.12 on
Python 3.9 and 0.2.0 on Python 3.10–3.14; Rtree 1.4.1 supplies the spatial index.
They install automatically with a compatible FUSE wheel on Linux x86-64. Rtree's
wheel includes libspatialindex. Neither dependency requires CUDA.

Thin anisotropic meshes can reduce poorly with this backend. Quality checks may
also retain more triangles than requested. See [testing and benchmarks](testing.md)
for reproducible measurements on your own geometry.

## Run checks and benchmarks

```bash
python -m pytest tests/test_simplification.py -q
python -m pytest tests -q
python benchmarks/benchmark_simplify.py --gpu
```

The first command includes two GPU integration tests. Add `-m 'not gpu'` for
CPU-only acceptance. The benchmark without `--gpu` exercises only CPU fixtures.
It runs each case in a fresh process and reports time, process peak RSS, peak-RSS
growth, bounds, volume change and reduction ratio. Peak-RSS growth is a
high-water-mark difference, not a measurement of all allocations.

## Large meshes and host memory

Topology validation uses compact native edge arrays and checks vertex fans without
constructing Python adjacency dictionaries. The initial inspection returns only
connected-shell face indices; reduction proceeds one shell at a time. Volume and
normal calculations use batches, and distance verification builds one direction's
spatial index at a time. These operations run on the CPU and need no GPU memory.

Memory still grows with mesh size. Input/output arrays, the largest shell's
quadric-decimation state, and triangle geometry for distance verification must fit
in host RAM. This is not an out-of-core simplifier, and there is no host-memory cap
in the API. A mesh with one very large shell can remain expensive.

The default `max_checks=200000` is intended for smaller meshes. Verification needs
at least one check per triangle in both the original and candidate shell, plus
sharp-edge checks and any adaptive subdivisions. For million-triangle shells,
increase this budget accordingly; a larger budget permits more work but does not
guarantee acceptance or limit peak memory.

To measure topology inspection separately from decimation and distance checks:

```bash
python benchmarks/benchmark_topology.py model.ply
```

This reports component sizes, inspection time, and whole-process peak RSS
(including mesh loading). Run it in a fresh process for comparable measurements.
