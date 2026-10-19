# Architecture and numerical behavior

FUSE computes an approximate full boundary of the union of triangle-mesh solids.
It uses one uniform isotropic voxel spacing, sparse candidate indexing, CUDA
field queries, and indexed triangle extraction. The Python API also provides
CPU loading, export, mesh simplification, and surface-quality measurement.

## Pipeline

```text
Files / Trimesh / NumPy arrays
          |
CPU loading, validation, normalization and BVH construction
          |
CPU candidate cells and shared NanoVDB sample indexing
          |
CUDA signed union samples
          |
CUDA count/scan and indexed surface extraction
          |
World-space vertices, faces and diagnostics
          |
Optional CPU simplification and export
```

The native public interface uses FUSE-owned array descriptors; NanoVDB types do
not appear in the Python or public C++ mesh API. The extension uses nanobind and
releases the GIL during native reconstruction. Streams, buffers, and memory
budgets are per call. There is no persistent geometry cache or GPU buffer pool.

## Solid and field semantics

Each input mesh defines one solid. A double-precision +X ray-parity classifier
uses projected yz edge functions and a half-open tie convention for shared edges
and vertices. Classification is even/odd **within each input mesh**. Nested shells
within one input can therefore represent cavities. Independent overlapping or
contained solids must be supplied as separate inputs.

For each sample, FUSE finds the nearest triangle distance for each relevant mesh,
assigns a negative sign inside that mesh, and takes the minimum:

```text
F(x) = min_i signed_distance_to_mesh_i(x)
inside_union(x) = any_i inside_mesh_i(x)
```

A containing solid is considered even when its boundary is far from the sample.
This suppresses buried surfaces. Parity over all concatenated inputs would not
implement these union semantics. The minimum field has the union sign but is
not generally the exact distance to the union boundary inside overlaps; it is
not an exact-offset API.

Median-split triangle BVHs accelerate queries. Nearest-distance traversal visits
nearer bounds first; parity uses a separate stackless traversal. Scenes with more
than four solids use a hierarchy over shape bounds, while smaller scenes use a
linear shape loop. Pruning retains potentially containing shapes.

## Coordinates and validation

The scene minimum is subtracted before conversion to grid units. Sample zero
lies 2.5 voxel spacings below that minimum; dimensions are
`ceil(scene_extent / voxel_size) + 6`. Unused vertices do not expand the domain.
Geometry, field values, interpolation, and returned coordinates use double
precision. Faces use uint64 indices.

Full validation checks finite data, valid indices, nondegenerate triangles,
exactly two oppositely directed faces per edge, and connected manifold vertex
links. Basic validation skips topology checks; it does not change the valid-solid
precondition. Neither mode detects self-intersections or repairs inputs.
Coordinates or triangles that collapse at the requested numerical scale are
rejected. Collapsed output triangles produce an error instead of silent removal.

Samples within `1e-12` grid units of zero use a small exterior value of `1e-7`
grid units to avoid collapsed corner intersections. This numerical convention
is not a dilation or offset operation.

## Sparse and dense backends

The default sparse backend finds candidate blocks by conservative triangle/box
overlap tests. Blocks contain 8 × 8 × 8 cells. Candidate cells include their
corner samples, and shared halo samples are deduplicated globally. Coordinate
runs are merged before expansion into canonical z/y/x-ordered keys.

A host-built NanoVDB uint64 grid maps each active sample coordinate to its compact
index plus one. Background zero means missing topology, **not** an exterior field
value. Sample values reside in a separate dense array over active samples only.
The grid is uploaded for CUDA lookup; a thread-local accessor is reused within
each extraction cell. FUSE does not use NanoVDB mesh rasterization or signed flood
fill for this path, and it does not link the OpenVDB CPU library or TBB.

The pinned NanoVDB sources are under `external/openvdb/nanovdb` at revision
`e26f5a569201d3236e0c30580572101022d3d754`. FUSE owns classification, union
semantics, and extraction; NanoVDB supplies sparse spatial indexing.

The dense backend evaluates all samples/cells in the padded domain and serves as
a reference. Both backends use the same field and extraction policy. Sparsity
reduces empty-space work but does not change voxel resolution or guarantee a
speedup on small or nearly full domains.

`return_field=True` produces a dense diagnostic array even for sparse reconstruction.
Missing samples are evaluated explicitly in bounded GPU chunks, including deep
interiors. This costs work proportional to the entire domain and can defeat the
memory advantage of sparse output. See [API memory accounting](api.md).

## Surface extraction

CUDA count-and-scan passes allocate output ranges. Grid-edge crossings receive
shared vertex identities, so neighboring cells reuse vertices without spatial
welding. Sparse cells resolve corner indices through NanoVDB.

Each cube face uses a canonical contour decision. For alternating signs, the
bilinear asymptotic determinant `f0*f2 - f1*f3` selects the pairing; an exact tie
uses the canonical 0–2 diagonal convention. Neighboring cells share that decision.
Closed cell contours are triangulated as disks around the mean of their crossing
vertices. This center is an extraction vertex, not a smoothing pass.

Interior ambiguity is resolved as separate disks. This is not MC33 and does not
guarantee the topology of the continuous trilinear interpolant. Unresolved
cell-interior tunnels, degenerate contacts, subvoxel walls, and narrow gaps can
change topology or disappear. Conservative candidate coverage does not prevent
loss from corner sampling. General manifoldness and absence of self-intersections
are not guaranteed for the extracted mesh.

Face numbering is deterministic on the validated platform; cross-platform
bitwise identity is not promised. Resolution remains uniform. Exact triangle
Booleans, exterior-only envelopes, adaptive refinement, mesh repair, and supplied
implicit-field inputs are not supported.

## Optional simplification

CPU quadric edge-collapse decimation operates on the extracted surface, not on
voxel cells. FUSE independently checks each candidate's surface/crease deviation,
closed manifold topology, orientation, and volume before accepting it. Its error
reference is the unsimplified mesh, so it does not remove sampling error.
See [the simplification contract](simplification.md) for its numerical limits,
verification budget, and failure behavior.

## Surface-quality measurement

`compare_surfaces` runs independently on the CPU. It samples source points
uniformly by triangle area in both directions and queries nearest target
triangles through an Rtree index. Directional and symmetric RMS, percentiles,
and tolerance coverage are estimates, not certified distance bounds.

One original is treated as a surface, which may be open or nonmanifold.
Multiple originals are valid solids combined by Manifold's float64 Boolean union;
this independent reference removes buried surfaces before comparison. Optional
zero-area-face removal acts on snapshots and reports per-input counts. It does
not repair solid topology. See [quality measurement](quality.md) for contracts,
metrics, and numerical limits.

## Resources

All explicit FUSE device buffers, including hierarchy data, NanoVDB, scratch,
scans and outputs, participate in the per-call memory budget. CUDA context and
runtime-managed memory, kernel-local storage, and host memory are excluded.
Sparse candidate construction also uses a budget-derived block-count guard;
it can reject a request before exact device allocation accounting.

Empty scenes bypass CUDA. Nonempty calls synchronize before returning and restore
the calling thread's device. See [API ownership and errors](api.md) and
[testing](testing.md) for reproducible checks.
