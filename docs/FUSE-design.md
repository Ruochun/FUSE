# FUSE Design Document

Fast Union Surface Extraction

Status: implementation design draft, 18 October 2026

Subsequent distribution requirements and their implementation status are recorded
in [distribution-requirements.md](distribution-requirements.md), including the
CPython 3.9–3.14 target, GitHub wheel builds, and plain `pip install fusemesh`.

M1 implementation choices and validation are recorded in
[decision 0002](decisions/0002-m1-dense-baseline.md) and
[M1 validation](m1-validation.md). M2 sparse implementation choices and evidence
are in [decision 0003](decisions/0003-m2-sparse.md) and
[M2 validation](m2-validation.md). M3 choices and evidence are recorded in
[decision 0004](decisions/0004-m3-performance-api.md),
[M3 validation](m3-validation.md), and the [0.2 API contract](api.md).

FUSE will reconstruct a surface mesh approximating the union of multiple potentially overlapping solids. The implementation will use C++ and CUDA, with NanoVDB as an internal dependency, and expose a Python API distributed as binary wheels on PyPI. This document records the agreed direction, proposed implementation defaults, correctness constraints, and development milestones for implementation agents.

No implementation or benchmark results are implied by this design. Performance targets and final supported platforms must be established using representative workloads.

## 1 Product objective and decisions

The primary objective is fast GPU reconstruction with controllable spatial resolution. Inputs initially consist of triangle meshes representing solids. The architecture should allow analytic shapes and existing implicit fields later, without requiring those representations to be triangulated first.

### Agreed direction

- Product name: FUSE, meaning Fast Union Surface Extraction.
- Implement the reconstruction backend in C++ and CUDA.
- Use NanoVDB as an internal dependency, pinned through a Git submodule of its upstream OpenVDB repository.
- Implement or integrate the missing geometry operations rather than build a new general sparse volume library.
- Keep a path to a standalone Python package with precompiled PyPI wheels.
- Either preserving enclosed cavity boundaries or extracting only the exterior envelope is acceptable to the user. Choose the simpler, faster, robust implementation for supported inputs.

### Proposed initial defaults

- Compute the full boundary of the solid union, including enclosed cavity walls. This avoids an additional exterior-connectivity pass once a signed union field exists.
- Use a uniform, isotropic voxel spacing in a shared world coordinate system.
- Support closed, consistently oriented, valid solid triangle meshes first.
- Extract triangles using a marching cubes implementation with consistent ambiguity handling and shared vertex indexing.
- Offer NumPy inputs and outputs first. Add direct GPU array interoperability later.
- Build a minimal installable wheel at the beginning of development.
- Avoid smoothing by default because it changes the reconstructed geometry.

These defaults are engineering proposals, not measured conclusions. Any change should be recorded with its reason, effect on semantics, and validation evidence.

## 2 Scope and limitations

### First functional release

Accept multiple meshes, combine their solid interiors, and return an indexed triangle mesh at a requested voxel spacing. Support disconnected components, overlaps, containment, and empty inputs. Provide device selection, bounded resource usage or a clear resource error, and useful input diagnostics.

Approximation occurs through field sampling and surface extraction. FUSE is not initially an exact triangle Boolean system. Original faces, UVs, materials, and sharp features are not preserved automatically.

### Deferred capabilities

- Open or defective mesh repair and generalized winding-number solid interpretation.
- Exterior-only envelope extraction and optional cavity removal.
- Analytic primitives, supplied SDFs, occupancy volumes, and point-cloud reconstruction.
- Adaptive output resolution, dual contouring, simplification, and feature preservation.
- Incremental updates for moving or edited inputs.
- Multiple GPUs, CPU reconstruction fallback, non-CUDA GPU backends, and differentiable reconstruction.
- Exact offsets, intersections, and differences.

Do not silently repair inputs or introduce thickness, smooth unions, dilation, or gap closing. Each changes the represented solid and requires an explicit future mode.

## 3 Mathematical semantics

Let each input define a solid S_i and a scalar function f_i(x) that is negative in its interior and positive in its exterior. Define the union S as the union of all S_i and evaluate:

```text
F(x) = min_i f_i(x)
inside_union(x) = any_i inside_i(x)
```

The target is the boundary of S. Surface extraction uses the zero transition of F. For touching surfaces and exact zero samples, use a consistent numerical convention; isolated zero values are not sufficient evidence of a boundary. Degenerate contacts may not admit a manifold surface preserving every contact exactly.

If each f_i is an exact signed distance function, the minimum has the correct union sign and is 1-Lipschitz. It is generally not the exact distance to the union boundary inside overlaps. This is sufficient for reconstructing its boundary, but operations requiring true distances, such as accurate offsets, may need distance rebuilding.

An approximate distance field paired with a heuristic sign classifier must not be assumed to preserve the same Lipschitz guarantee. Sparse pruning may use that guarantee only when the actual field evaluator satisfies it.

### Full boundary and exterior envelope

The full boundary includes enclosed void walls. Extracting it from a valid signed union field requires no separate cavity classification.

An exterior-only mode would classify empty space connected to a padded domain boundary and retain faces adjacent to that space. In a sparse grid, this must account for empty tiles and unallocated regions as well as explicit voxels. It is distinct from the signed flood-fill operation used to propagate existing narrow-band signs.

## 4 System architecture

```text
Python API or native C++ caller
             |
      FUSE public C++ API
             |
 Input validation and scene normalization
             |
 Shape adapters and acceleration structures
             |
 Conservative candidate topology construction
             |
 Shared samples and signed union evaluation
             |
 Sparse indexed surface extraction
             |
 Result validation and optional future simplification
```

NanoVDB owns the internal sparse-grid representation where appropriate. Our code owns solid semantics, union evaluation, extraction integration, resource policy, and public API behavior.

### Public API boundary

Keep NanoVDB classes, CUDA implementation types, and upstream Python bindings out of the public Python API. Prefer FUSE-owned native descriptors so changing storage or upstream versions does not require user code changes.

The C++ library should be usable without Python. Bindings should perform validation and conversion, release the Python GIL during native computation, and translate native errors into useful Python exceptions. Binding code should not contain reconstruction algorithms.

### Internal modules

| Module | Responsibility |
| --- | --- |
| core | Public options, mesh descriptors, result types, errors, coordinate mapping |
| geometry | Shape adapters, solid classification, closest-point queries, acceleration structures |
| volume | NanoVDB integration, candidate topology, sample values, interior representation |
| cuda | Device kernels, scans, compaction, allocation, execution and timing |
| meshing | Cell classification, ambiguity decisions, vertex ownership, indexed face generation |
| python | Nanobind bindings and Python convenience API |
| validation | Geometric checks, analytic fixtures, independent reference comparisons |

## 5 NanoVDB integration

NanoVDB is a C++ sparse-volume library with CUDA tools and Python bindings. Our package will compile its own extension against the selected upstream revision; it does not depend on a separately published NanoVDB wheel.

The following upstream capabilities were identified during design research. Agents must verify their availability, signatures, and behavior at the pinned commit before depending on them.

| Capability | Intended use | Important distinction |
| --- | --- | --- |
| NanoVDB grids and index grids | Sparse topology and associated sample channels | Spatial sparsity does not itself supply solid semantics |
| CUDA MeshToGrid | Candidate topology and optional unsigned mesh distances | Unsigned distances do not classify solid interiors |
| CUDA MergeGrids | Merge sparse allocation across inputs | Topology union is not a Boolean union of scalar fields |
| CUDA SignedFloodFill | Extend established signs into inactive values or tiles | It does not infer signs from unsigned distances |
| GPU topology construction and modification tools | Build or rebuild working grids | Do not assume arbitrary in-place topology insertion |

A complete GPU field-to-mesh extractor was not established in the inspected NanoVDB CUDA tools. Treat surface extraction as a FUSE responsibility until an appropriate existing implementation has been verified.

The inspected SignedFloodFill source documents a root-tile expansion limitation for large enclosed interiors. Determine whether it applies to the pinned version and our data representation. Include large-domain and deeply enclosed-region tests if this tool is used.

### Dependency policy

- Pin an exact upstream commit and record it in build metadata.
- Prefer NanoVDB without the OpenVDB CPU dependency in the shipped backend, unless required functionality justifies it.
- Disable unrelated upstream viewers, examples, bindings, and tools in production builds.
- Prefer an adapter layer over edits to upstream source. Keep unavoidable patches small and documented.
- Audit licenses and redistribution notices for the selected source and shipped runtime components before release.
- An optional OpenVDB CPU dependency may be used in development for independent reference results without becoming a runtime dependency.

## 6 Reconstruction pipeline

### 6.1 Validate and normalize inputs

Validate array shapes, finite coordinates, index bounds, index width, and nonzero triangle area policy. Distinguish cheap structural validation from potentially expensive watertightness, orientation, and self-intersection checks.

The supported baseline is a closed, consistently oriented mesh defining a valid solid. Separate inputs may overlap freely. Handling a self-intersecting individual mesh requires an explicit interior rule and is deferred.

Use one transform from world coordinates to grid coordinates for the entire operation. Consider subtracting a scene origin and retaining a high-precision host transform to reduce precision loss for large coordinate offsets. Record the returned mesh coordinate convention explicitly.

Compute a padded domain so the surface does not terminate at its edge. Padding must cover the selected sampling and extraction stencil. Empty input returns empty arrays without allocating a scene-sized volume.

### 6.2 Build acceleration structures

Use a hierarchy over input shape bounds to reduce the number of shape evaluations per query. Mesh adapters require triangle acceleration for nearest-surface queries and the selected inside/outside algorithm.

Do not assume NanoVDB MeshToGrid provides all query structures required for classification. Select and benchmark an existing C++/CUDA implementation where useful; implement missing queries behind a replaceable interface.

Bounds used to prune distance queries must be conservative. A shape containing a sample may determine the union even if none of its surface-band voxels are present near that sample.

### 6.3 Construct conservative candidate topology

Generate cells or blocks covering potential input surfaces, merge their allocation, and include the sample halo required for extraction. Initially accept extra work around buried surfaces; eliminate it after union evaluation.

Never discover all candidates solely from sign changes at coarse cell corners. Thin objects and small components can fit entirely inside a cell. Use triangle overlap tests, bounds on valid implicit functions, or other conservative candidate generation.

Candidate conservativeness does not guarantee that final fixed-resolution marching cubes preserves subvoxel geometry. Report the sampling limitation and test it explicitly.

Fixed-size blocks such as 8 cubed or 16 cubed cells are possible work scheduling units. These are tuning candidates, not an alternative NanoVDB tree design. NanoVDB tree sparsity should not be confused with adaptive geometric resolution.

### 6.4 Evaluate signed union samples

At shared grid sample positions, evaluate the appropriate shape functions and form their minimum. All inputs must use the same sign convention, units, transform, and truncation policy.

Two implementation strategies are candidates:

1. Query shape adapters directly at the shared candidate samples, avoiding complete per-object fields.
2. Construct per-object signed narrow bands, represent their interior and exterior values correctly, and combine them on shared topology.

Select using correctness and measured memory/query costs. The second strategy must not interpret all absent samples as exterior. An absent sample may be deeply inside another solid and must suppress a buried surface. The first strategy still needs a correct classifier for every relevant containing shape.

Never replace union classification with the sign of the nearest triangle among concatenated meshes. A nearest surface may be internal to the union. Similarly, parity over all concatenated surfaces can cancel overlapping interiors and implement the wrong solid rule.

For the baseline, choose a robust classifier for supported closed meshes, with a deterministic policy for rays through vertices or edges. Winding-number classification is a possible future mode for defective inputs, with a documented threshold and orientation convention. It does not recover an unambiguous intended solid from arbitrary open surfaces.

Finite query-radius failures must be handled explicitly. A failed closest-point query is not automatically an exterior classification.

### 6.5 Extract an indexed surface

Start with marching cubes using consistent shared-face ambiguity decisions. Do not claim watertight or manifold output from an unspecified lookup table alone.

Recommended GPU structure:

1. Evaluate cell cases and output counts.
2. Use scans or compaction to allocate exact output ranges.
3. Assign vertices consistently to global grid edges.
4. Interpolate crossings and emit indexed faces.
5. Apply consistent handling of exact-zero samples, degenerate triangles, and orientation.

Adjacent blocks must share sample values, ambiguity decisions, and vertex identities. A packed edge key must accommodate the supported coordinate range without collisions or overflow. Use a deterministic indexing path where practical; otherwise document which ordering may vary while geometry remains equivalent.

Avoid a single globally contended output counter where a count-and-scan approach is suitable. Check allocation and index overflow before writing buffers.

A dense or chunked implementation may establish a reference pipeline before sparse extraction. Chunking requires halos and global ownership rules; extracting each chunk independently and concatenating it is insufficient.

### 6.6 Return and optionally process results

Return an indexed triangle mesh in world coordinates. Preserve disconnected components. Do not apply default smoothing or delete small components silently.

Future simplification must have a documented geometric error criterion. Projecting vertices toward input surfaces must respect the union boundary; nearest-input-surface projection alone can pull vertices onto buried geometry.

## 7 Resolution and geometry quality

The initial control is voxel_size in world units. A future max_error option requires a measured or proven relationship to the actual algorithm; voxel spacing is not automatically a Hausdorff-error guarantee.

Inputs with walls, gaps, or components below the sampling scale may lose topology. A user requirement to preserve a minimum feature size must become a validated resolution policy, potentially using local refinement. Do not promise preservation from an arbitrary fixed multiplier alone.

A dense 1024 cubed float32 field occupies 4 GiB before scratch or mesh output. This motivates sparse working sets, reusable buffers, and explicit accounting of peak device memory.

Marching cubes is the initial extraction choice. Surface nets may improve face regularity while smoothing features. Dual contouring can preserve sharp features with Hermite data, but adds vertex optimization and connectivity complexity, especially with adaptive resolution. Benchmark these only after the baseline works.

## 8 Alternative approaches and why they are deferred

### Direct triangle Boolean operations

An implementation such as Manifold can serve as a reference or alternative for valid solid meshes. Direct Booleans preserve original planar geometry and intersection curves more closely, but require robust geometric intersections and do not directly solve resolution-controlled remeshing or mixed implicit inputs.

### Surface barriers and exterior flood fill

Conservatively voxelizing surfaces, flooding exterior empty space, and extracting the remaining boundary is a candidate approximate envelope mode. It may avoid some signed-distance work, but voxel gaps can leak, narrow openings can close, and surfaces can jointly enclose regions outside the actual solid union. It is not interchangeable with the exact union semantics.

### Warp prototype

Warp was considered for Python-first development because it provides GPU mesh queries and extraction tools. The selected direction is now a C++/CUDA backend using NanoVDB. Warp may still serve as a prototype or benchmark dependency; it need not be a shipped dependency.

### Fully custom storage

Writing a BVH, sparse volume tree, allocator, mesher, and bindings from scratch offers control but greatly expands maintenance. Replace infrastructure only after a measured limitation justifies it.

## 9 Proposed public interfaces

The following is a sketch, not a frozen API. The proposed distribution and import name fusemesh requires availability and naming checks before publication.

```python
import fusemesh

result = fusemesh.reconstruct(
    meshes=[(vertices_a, faces_a), (vertices_b, faces_b)],
    voxel_size=0.01,
    device=0,
)

vertices = result.vertices
faces = result.faces
```

Specify accepted dtypes and contiguity, copy behavior, output index width, coordinate precision, and validation policy. A reasonable first contract accepts floating-point vertex arrays of shape N by 3 and integer face arrays of shape M by 3, with explicit conversion or rejection rules.

Optional diagnostic metadata should include the requested spacing, grid transform, device, counts, and timing categories when requested. Do not expose unreliable error estimates as guarantees.

The first API may be synchronous. Future asynchronous calls need explicit stream semantics, buffer lifetime rules, and synchronization behavior. GPU input support must define device matching and ownership; accepting a pointer alone is not sufficient.

The C++ API should separate input descriptors, reconstruction options, execution context, and owning result buffers. Avoid requiring a persistent scene object for a one-shot call, but allow cached acceleration structures later.

## 10 Build and wheel distribution

### Proposed repository layout

```text
FUSE/
  CMakeLists.txt
  pyproject.toml
  include/fuse/
  src/core/
  src/geometry/
  src/volume/
  src/cuda/
  src/meshing/
  bindings/
  python/fusemesh/
  external/openvdb/       # pinned Git submodule
  tests/
  benchmarks/
  docs/
```

Use CMake for the native library and extension, nanobind for bindings, and scikit-build-core as the Python build backend. The package builds its own extension and does not require upstream NanoVDB Python bindings.

### Installation contract

For a supported binary wheel, users should need a supported Python version, operating system, NVIDIA GPU, and sufficiently recent driver. They should not need Git, CMake, a host compiler, or nvcc.

Choose a CUDA runtime distribution strategy explicitly. Options include permitted bundling, suitable runtime package dependencies, or permitted static linking. Do not bundle the NVIDIA driver or depend accidentally on toolkit files installed on the build host. Audit the actual extension dependency tree.

Standard wheel tags do not encode the CUDA runtime family. Start with one documented CUDA family rather than several conflicting wheels with indistinguishable compatibility tags. Decide the supported Python versions, Linux baseline, Windows toolchain, GPU architectures, and minimum driver before the first public release.

Compile for an explicit architecture set; do not inherit only the build host GPU. Consider PTX fallback with documented driver implications. Balance coverage against binary size.

Linux and Windows require separate builds and validation. Use an appropriate manylinux build and dependency-repair workflow on Linux, and DLL dependency inspection and repair on Windows. CUDA execution is not a macOS feature of this backend.

Source distributions must include the pinned dependency sources or implement an explicit documented acquisition strategy. A PyPI source archive does not automatically contain an initialized Git submodule. Prefer self-contained source archives for reproducibility.

### Release validation

- Build the wheel in CI with recorded compiler, CUDA, and dependency versions.
- Install the wheel into a clean environment without the development toolkit.
- Import it and execute representative GPU reconstruction.
- Test graceful diagnostics for missing drivers, unsupported GPUs, and allocation failures.
- Verify included licenses, runtime dependencies, and artifact size against current PyPI requirements.
- Check package-name availability before publishing, and use TestPyPI for release rehearsal.

Ordinary build workers can compile without a GPU; runtime correctness and compatibility require GPU test workers.

## 11 Correctness and performance validation

### Geometry fixtures

| Fixture | Required evidence |
| --- | --- |
| Empty input and isolated solid | Defined empty result; correct single-solid reconstruction |
| Disjoint solids | All components retained |
| Partially overlapping solids | Internal overlapping faces suppressed |
| Complete containment | Contained surface suppressed |
| Coincident or nearly coincident surfaces | Stable classification and controlled degeneracy handling |
| Face, edge, and point contacts | Documented behavior for degenerate topology |
| Thin walls and narrow gaps | Resolution-dependent loss characterized |
| Small object inside a coarse cell | Candidate discovery does not falsely claim preservation |
| Enclosed cavities | Full-boundary default preserved |
| Large interior beyond surface bands | Missing samples retain correct interior semantics |
| Geometry across block and tile boundaries | No seams or key collisions |
| Large coordinate offsets | Precision behavior measured |
| Invalid indices, NaNs, open meshes | Rejection or explicit unsupported-input diagnostics |

Use analytic unions where available and an independent CPU implementation for reference. The reference is a comparison tool, not proof that every discrepancy is a FUSE error.

Check finite output, valid indices, triangle degeneracy, orientation, edge incidence, connected components, self-intersections where practical, and approximate symmetric surface distance. Sample triangle interiors as well as vertices for error estimates. Test signed volume and union occupancy where meaningful.

Do not guarantee manifold output for degenerate contacts or unresolved subvoxel features. For well-resolved supported solids without degeneracies, target closed, consistently oriented output and test that contract.

### Performance measurements

Measure upload, validation, acceleration construction, candidate generation, sign queries, distance queries, union evaluation, extraction, download, and total time separately. Synchronize GPU timings appropriately. Report cold-start and warm timings separately.

Record hardware, driver, build configuration, triangle and shape counts, voxel spacing, candidate and active cell counts, output size, overlap distribution, and peak GPU memory including temporary buffers. Compare approaches at equivalent geometry quality and output semantics.

Benchmark repeated queries if cached scenes are added. No real-time claim or speedup target is established yet.

## 12 Development milestones

### Milestone 0 Build and dependency proof

Pin upstream; compile a minimal C++/CUDA library; build and install a wheel; execute a small NanoVDB operation; return a NumPy result. Verify the missing-operation inventory against the selected commit. Record platform, CUDA, compiler, and runtime choices.

Acceptance: the installed wheel runs in a clean test environment on the initial supported GPU platform.

### Milestone 1 Correct baseline

Implement supported mesh validation, shared coordinates, a selected sign classifier, union field evaluation, and a simple dense or chunked extraction path. A CPU reference path may assist debugging but must not be presented as the GPU implementation.

Acceptance: analytic and valid-mesh union fixtures pass at documented resolution, including containment and buried surfaces.

### Milestone 2 Sparse GPU reconstruction

Use NanoVDB candidate topology and shared samples, correct inactive-interior semantics, and sparse indexed extraction. Keep the baseline as a comparison until equivalence is established.

Acceptance: equivalent sampled-field results within numerical tolerance; no block seams; peak memory and stage timings measured.

### Milestone 3 Performance and API stabilization

Optimize only measured bottlenecks. Introduce shape hierarchy pruning, reusable buffers, compacted work, or cached acceleration as justified. Finalize error messages and ownership semantics.

Acceptance: representative workload benchmarks and regression thresholds are recorded; public API behavior is documented.

### Milestone 4 Public wheel release

Add the required platform matrix, runtime dependency handling, clean-install GPU tests, source archive verification, documentation, and release rehearsal. Add authoritative formatting configuration and a project-wide formatter/check utility, and format all first-party code.

Implementation and external release gates: [release instructions](releasing.md) and [M4 validation](m4-validation.md).

Acceptance: users can install the supported wheel and run reconstruction without a compiler or CUDA toolkit.

### Milestone 5 Surface mesh simplification

After M4, add an optional post-reconstruction utility to reduce unnecessary
triangles on planar and gently curved regions. Here, merging fine cells means
simplifying the extracted surface mesh; the sampling grid remains uniform and
adaptive voxel refinement remains deferred. Output remains a triangle mesh.

Start by evaluating quadric edge-collapse decimation through the existing
[Trimesh helper](https://trimesh.org/trimesh.html#trimesh.Trimesh.simplify_quadric_decimation)
and its additional [fast-simplification dependency](https://github.com/pyvista/fast-simplification).
For explicit planar patch merging and retriangulation, evaluate
[CGAL planar patch simplification](https://doc.cgal.org/6.2/PMP_Remeshing/group__PMP__planar__remeshing__grp.html).
Backend selection is pending compatibility, licensing, packaging, and quality
evaluation; no dependency is selected or added by this milestone definition.

Expose explicit simplification controls and report before/after counts and
geometric error. Define world-space error tolerance, sharp-feature protection,
and failure behavior before stabilizing the API. A target face count or quadric
cost alone must not be presented as a guaranteed surface-distance bound.
Keep the unsimplified reconstruction available and simplification opt-in.

Acceptance: meaningful tests on planar boxes, curved surfaces, sharp creases,
thin features, disconnected solids, and cavities demonstrate triangle reduction
while checking geometric deviation, volume change, watertightness, manifoldness,
winding, and preserved topology. Record time and memory overhead; document any
limits of the selected backend and reject results that violate promised checks.
Verify the added dependency against the M4 Python and wheel support matrix.

## 13 Decisions still required

These choices do not block the architecture, but must be resolved before their dependent milestones:

| Decision | Needed by |
| --- | --- |
| Representative workloads, shape counts, triangle counts, spatial scale, latency and memory goals | Baseline benchmarking |
| Exact upstream revision and availability of desired GPU APIs | Milestone 0 |
| Initial platform, Python versions, CUDA family, architecture set, driver minimum | Initial wheel contract |
| Mesh classifier and numerical degeneracy policy | Milestone 1 |
| Direct queries versus per-shape signed fields | Milestone 2 |
| Extraction implementation and ambiguity policy | Milestone 1 |
| Output precision and index-width policy | API stabilization |
| Whether open meshes are a product requirement | Scope expansion |
| Distribution name, project license, and dependency notices | Public release |

Agents should proceed with local, reversible implementation choices within the agreed architecture. Record consequential decisions in docs/decisions with alternatives, rationale, and validation. Do not silently replace union semantics with an approximate envelope or claim an error guarantee from voxel spacing alone.

## 14 Reference sources

Upstream main-branch URLs can change. Implementation records should additionally link the pinned commit used by FUSE.

- [NanoVDB CUDA tools](https://github.com/AcademySoftwareFoundation/openvdb/tree/master/nanovdb/nanovdb/tools/cuda)
- [NanoVDB MeshToGrid source](https://github.com/AcademySoftwareFoundation/openvdb/blob/master/nanovdb/nanovdb/tools/cuda/MeshToGrid.cuh)
- [NanoVDB SignedFloodFill source](https://github.com/AcademySoftwareFoundation/openvdb/blob/master/nanovdb/nanovdb/tools/cuda/SignedFloodFill.cuh)
- [OpenVDB release notes](https://www.openvdb.org/documentation/doxygen/changes.html)
- [Upstream repository and build options](https://github.com/AcademySoftwareFoundation/openvdb)
- [Nanobind packaging guide](https://nanobind.readthedocs.io/en/latest/packaging.html)
- [Scikit build core CMake integration](https://scikit-build-core.readthedocs.io/en/latest/guide/cmakelists.html)
- [NVIDIA CUDA compatibility](https://docs.nvidia.com/deploy/cuda-compatibility/latest/)
- [Warp geometry documentation](https://nvidia.github.io/warp/stable/api_reference/warp_geometry.html)
- [Manifold](https://github.com/elalish/manifold)
