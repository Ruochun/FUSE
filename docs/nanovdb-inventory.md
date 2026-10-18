# NanoVDB capability inventory at the M0 pin

Inspected source: OpenVDB v13.1.0,
`e26f5a569201d3236e0c30580572101022d3d754`.
Paths below are relative to `external/openvdb/nanovdb/nanovdb/`.

M0 runtime-validates host float-grid construction and device access. M2 also
uses host-built uint64 sample-index topology with CUDA lookup; see
[decision 0003](decisions/0003-m2-sparse.md).
Other entries are source-inspected candidates; availability is not evidence of
FUSE semantic correctness or performance. Validate each before production use.

| Capability | Pinned API / evidence | Implication |
| --- | --- | --- |
| Float grids | `tools::build::Grid<float>`, `tools::createNanoGrid<Grid, float>`, `NanoGrid<float>::getAccessor()` | Used by M0 for host construction and CUDA readback. |
| M2 sample lookup | `tools::build::Grid<uint64_t>`, `tools::createNanoGrid<Grid, uint64_t>`, `NanoGrid<uint64_t>::tree().getValue()` | Runtime-tested host construction and CUDA compact sample-index lookup; background is absent topology, not field sign. |
| Index topology | `tools/cuda/TopologyBuilder.cuh`: `TopologyBuilder<BuildT, ResourceT>(cudaStream_t, ResourceT&)` | `BuildTraits<BuildT>::is_onindex` is required. Staged mask/node construction, not arbitrary sparse insertion. |
| Mesh rasterization | `tools/cuda/MeshToGrid.cuh`: `MeshToGrid<BuildT>(const Vec3f*, uint32_t, const Vec3i*, uint32_t, Map, cudaStream_t, ResourceT&)`; `getHandle()`, `getHandleAndUDF()` | Device vertex/index arrays; optional UDF sidecar, no sign classifier. `setNarrowBandWidth()` takes cell units. |
| Topology merge | `tools/cuda/MergeGrids.cuh`: `MergeGrids<BuildT>(const std::vector<const NanoGrid<BuildT>*>&, cudaStream_t, ResourceT&)`, binary overload, `getHandle()` | Active-mask union of index grids, not scalar Boolean union. |
| Signed propagation | `tools/cuda/SignedFloodFill.cuh`: `signedFloodFill<BuildT, ResourceT>(NanoGrid<BuildT>*, bool, cudaStream_t)` | Float build types; propagates existing signs. Does not classify unsigned mesh distances. |
| Topology modification | `DilateGrid.cuh`, `PruneGrid.cuh`, `RefineGrid.cuh`, `CoarsenGrid.cuh` exist | Potential rebuild operations. Not selected or runtime-tested yet. |
| Extraction | No complete marching-cubes/isosurface extractor found in `tools/` | FUSE still owns indexed extraction and ambiguity decisions. |

## Integration findings requiring follow-up

1. SignedFloodFill's header still documents inability to expand root tiles for
   fully enclosed `4096^3` interiors. Its root processing also includes host
   work. Do not use it as an unqualified large-domain sign solution.
2. MeshToGrid's `getHandleAndUDF()` declaration describes voxel-unit distances,
   but `topology::detail::FinalizeSidecarFunctor` multiplies by voxel size and
   emits world-space distances, including the background slot. Before using
   this path, add a non-unit-spacing numerical test and verify transform limits
   and sentinel behavior. Do not infer units from the declaration alone.
3. The CUDA builders require an asynchronous memory resource; several tools
   default to stream-ordered allocations. Resource lifetimes must exceed use.
4. `util/cuda/Util.h` defines `cudaCheck` through `gpuAssert`, which calls
   `exit(code)` on failure. An adapter boundary alone cannot catch process
   termination. Evaluate an upstream-supported error hook or a small documented
   patch before shipping these tool paths in a Python process. M0 uses checked
   FUSE CUDA calls instead and does not modify upstream.
5. A BVH-backed robust solid classifier, signed union evaluation, conservative
   extraction halo policy, shared vertex ownership, and resource limits remain
   FUSE work. Nothing in this inventory changes the design's union semantics.
