# 0003: M2 sparse candidates with shared GPU samples

Status: implemented, 18 October 2026. The default backend is now `sparse`;
`backend="dense"` retains M1 as the correctness reference. The C++ equivalent is
`ReconstructionOptions::sparse` (default true). No automatic backend switching.

## Candidate topology

Candidate construction uses input triangles, never coarse sample signs. In
normalized coordinates, recursively split each triangle's block bounding box
and conservatively reject boxes using triangle/box separating axes. Stop at
8-by-8-by-8-cell blocks. Projection margins include floating-point roundoff and
the M1 numerical zero offset. Merge block identities, clip to the padded domain,
and include all corners of every candidate cell. Buried surfaces initially
contribute candidates; signed union evaluation suppresses them.

Topology construction runs on the **CPU** in this milestone. Geometry
classification, distance queries, counts/scans, and indexed extraction run on
CUDA. This is not an entirely GPU-resident pipeline. Host topology preparation
is separately timed; moving it to CUDA is an M3 optimization candidate if
measurements justify it. No upstream modifications, OpenVDB library, or TBB are
required. NanoVDB's host builder is used only to construct the sparse lookup.

A NanoVDB uint64 grid maps each unique global sample coordinate to its compact
sample index plus one. Background zero means **missing topology**, never a
scalar field value. The NanoVDB grid is uploaded and read by CUDA extraction.
Sample values live in one dense channel over the *active samples only*.
Candidate cell keys and sample keys are sorted in global z/y/x order. Full
scene volume is never allocated or scanned in ordinary sparse reconstruction.

All dimensions fit signed NanoVDB coordinates; checked global linear keys fit
uint64, with headroom for edge arithmetic. Active scan counts retain CUB's
signed-int limits. A large domain is supported when these coordinate/key bounds
and its sparse working set fit; sparsity does not imply unlimited coordinates.

## Interior semantics and extraction

At each active sample, query the per-mesh classifier and untruncated signed
nearest distance, then take the minimum. A containing shape is queried even
when none of its surface samples are nearby. An inactive region is not assigned
an exterior value. This uses design strategy 1 (direct shared-sample queries),
not per-object narrow-band fields or signed flood fill. NanoVDB's flood-fill
root-tile limitation therefore does not apply to this path.

Extraction uses M1's canonical face decisions and contour-disk triangulation.
Every crossing edge is owned by `(compact sample index, axis)`; cells look up
that index through NanoVDB. Halo corners are deduplicated globally before
sampling. Shared values, ambiguity decisions, and vertex identities prevent
block seams without positional welding. An absent edge endpoint suppresses that
edge's count; every candidate cell's eight corners are explicitly allocated.
Conservative candidate coverage includes every cell whose sampled signs cross.

Because keys preserve dense traversal order, the tested backends produce
identical indexed meshes, not just equivalent triangle soups. Cross-platform
bitwise reproducibility is still not promised. All M1 sign, zero, numerical,
contact, subvoxel-feature, and cell-interior ambiguity limitations remain.

## Diagnostics and resources

`return_field=True` retains its dense `(z,y,x)` world-distance array contract.
The sparse backend queries missing regions in bounded GPU chunks, including
deep interiors, and copies its actual extraction values at active samples.
It does not fill absent entries with background. This diagnostic requires work
proportional to the full domain and should be omitted for large sparse scenes.
Its host array must *separately* fit `max_memory_bytes`; it is not part of the
device allocation total. This check occurs before topology construction.

Every device allocation, including NanoVDB, key buffers, CUB scratch, diagnostic
chunks, and outputs, participates in the same per-call budget. Host snapshots,
BVHs, topology construction, and results are excluded. Candidate growth has a
budget-derived block-count cap (`max_memory_bytes / 8192`) to stop pathological
requests before building all host keys; this conservative guard can reject
some scenes before exact device allocation accounting. Actual allocation
failures also raise `ResourceError`. No fallback silently changes resolution.

Metadata adds `candidate_blocks`, `candidate_cells`, `active_samples`, and
`topology_bytes` (serialized NanoVDB device bytes). Dense results report dense
sample/cell counts and zero sparse topology counts/bytes. `timings_ms.topology`
measures host candidate/NanoVDB construction; field and extraction timings
measure synchronized CUDA stages. Dense diagnostic queries are included in
`download`, not in the sparse extraction field timing. Measure total call time
separately for host conversion, CUDA initialization, and teardown.

## Deferred work

M3 should use measured workloads to choose host/GPU topology changes, cached
acceleration, faster NanoVDB access, shape hierarchy pruning, or buffer reuse.
Sparse representation can cost more than dense storage on small or nearly full
candidate domains. This milestone makes no speedup promise. M4 still owns
portable public wheels, CPython 3.9–3.14, GitHub/GPU CI, and publishing.

See [M2 validation](../m2-validation.md) for measured evidence.
