# 0002: M1 dense GPU reconstruction baseline

Status: implemented, 18 October 2026. This establishes a correctness reference
for M2, not a sparse or performance-optimized backend.

## Field and solid semantics

Each input is a closed, consistently oriented triangle mesh representing one
solid. Classification is even/odd parity **within each mesh**, followed by the
minimum of the per-mesh signed nearest-triangle distances. Never compute parity
across concatenated inputs. Nested shells within one input represent cavities;
separate overlapping or contained inputs are combined by solid union.

CPU preparation validates numeric/index structure and builds a median-split,
stackless triangle BVH for each mesh. GPU queries use conservative triangle
bounds for distance and ray traversal. Mesh bounds skip provably irrelevant
outside shapes; there is no scene-level shape BVH yet. All mesh sampling and
surface extraction execute in CUDA. CPU work comprises validation, BVH building,
coordinate normalization, result conversion, and output validity checks.

The classifier casts a +X ray using double-precision projected yz edge
functions. A canonical edge direction makes reversed shared-edge evaluations
exact negatives; a half-open top-left convention assigns edge/vertex ties
consistently. Parallel projected triangles contribute no crossings. Closest-point
queries have no finite radius or missing-neighbor fallback. This is a numerical
classifier for valid solids, not exact arithmetic or defective-mesh repair.
Independent solid-angle tests cover ray ties, rotated solids, and overlaps.

Full validation means full *implemented structural/topological validation*:
finite numeric inputs, bounds-safe indices, nonzero triangle area, exactly two
oppositely directed faces per edge, and connected vertex links. It does not
check geometric self-intersections. Basic validation skips only the topology
checks and retains the same valid-solid precondition. No repair is performed.

## Coordinates and dense coverage

Use one isotropic grid for the operation. Subtract the double-precision scene
minimum before scaling into grid units. Sample zero is at scene minimum minus
2.5 voxel spacings; dimensions are `ceil(scene_extent / voxel_size) + 6`.
All cells in this padded domain are evaluated. Unused input vertices do not
expand the domain. Dense coverage is conservative even for a component smaller
than one cell, but corner sampling can still miss that component entirely.

Double precision is used for normalized geometry, field values, interpolation,
and returned world coordinates. Faces use uint64 indices. Coordinates too large
to represent the requested spacing, or triangles collapsing during normalization,
are rejected. Output triangles that collapse after world-coordinate conversion
produce a clear error instead of being silently removed.

Samples with absolute union value at most `1e-12` grid units use an exterior
value of `1e-7` grid units. This is an explicit numerical zero convention to
avoid collapsed corner intersections. It is not a user-facing dilation/offset
mode. No preservation guarantee is made for degenerate contacts or features
near this numerical scale. No smoothing is applied.

## Marching-cubes extraction policy

Use cube edge crossings and a face-contour graph, not an unspecified classic
256-case triangle table. Every face uses canonical increasing in-plane axes.
For alternating signs, the bilinear asymptotic determinant
`f0*f2 - f1*f3` chooses the contour pairing; an exact tie selects the canonical
0–2 diagonal convention. Neighboring cells see the same values and decision.

Each closed contour is triangulated as a disk with a center at its crossing
vertices' mean. This center is an extraction vertex, not a smoothing pass.
Boundary arc orientation is derived from the face normal and scalar gradient,
so adjacent cells produce opposite directions on their shared triangle edge.

**Interior ambiguity is resolved as separate disks.** This is a documented
marching-cubes variant, not MC33 and not a guarantee of preserving the topology
of the continuous trilinear interpolant. A cell containing an unresolved tunnel
may receive a different topology. The baseline targets closed, consistently
oriented output for resolved, nondegenerate inputs. Do not infer general
manifold or self-intersection guarantees from the lookup/contour policy.

CUDA count-and-scan passes allocate exact output ranges. Grid edges have dense
64-bit identities `(sample_index * 3 + axis)`; all incident cells share one
crossing vertex. Per-cell scans allocate center vertices and triangle ranges.
No global append counter or post-hoc spatial weld is used. Identical inputs have
deterministic indexing on the tested platform; cross-platform bitwise identity
is not promised.

## Resource and API contract

`reconstruct(meshes, voxel_size, *, device=0, max_memory_bytes=512*1024**2,
validation='full', return_field=False)` returns a `ReconstructionResult` with
vertices, faces, metadata, and an optional dense field. Inputs are copied to
contiguous float64/uint64 arrays before native computation; callers must not
mutate inputs during conversion. The native phase releases the GIL. Each call
owns its buffers and stream, synchronizes before return, and restores the
caller's device. Empty scenes bypass CUDA and return empty arrays.

The memory limit covers all FUSE-owned device allocations, including CUB scan
scratch and output buffers. A conservative dense preflight uses 48 bytes per
sample, and each actual allocation checks the remaining budget. CUDA context
memory, other processes, host snapshots/BVHs, and host result storage are not
included. Allocation failures and limit violations raise `ResourceError`, a
Python `MemoryError` subclass. Grid lengths are capped so CUB's signed-int scan
counts and all allocation/index products remain valid. The default is a limit,
not a promise that every scene fits.

Diagnostic field order is `(z,y,x)`; `metadata['grid_shape']` is `(x,y,z)`.
Field units and returned coordinates are world units. Metadata includes the
transform, counts, FUSE peak device bytes, and synchronized stage wall timings.
Stage timings exclude Python conversion, CUDA context initialization, and final
resource teardown; measure
end-to-end wall time separately for cold starts.

## Deferred work

NanoVDB remains pinned, built, and smoke-tested; the M1 reconstruction field is
dense. M2 introduces NanoVDB candidate topology and sparse extraction, validated
against this baseline. M3 addresses measured bottlenecks and API stabilization.
Self-intersection detection/repair, generalized winding interpretation, exact
Booleans, adaptive resolution, exterior-only extraction, and direct GPU input
remain outside this implementation. The optional sampled-field entry in the
internal CUDA function exists solely for native extraction validation and is
not a supported supplied-field API.

Algorithm references (implementation is original FUSE code):

- [Nielson and Hamann, The Asymptotic Decider](https://people.eecs.berkeley.edu/~jrs/meshpapers/NielsonHamann.pdf)
- [PBRT, Triangle Meshes](https://www.pbr-book.org/4ed/Shapes/Triangle_Meshes)

See [M1 validation](../m1-validation.md) for evidence and limits.
