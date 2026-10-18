# 0004: M3 measured optimization and the 0.2 API

Status: implemented, 18 October 2026. This milestone preserves the M1/M2 field,
candidate coverage, uniform resolution, and extraction policies while reducing
measured work. Adaptive refinement remains deferred.

## Measurements drive the changes

The benchmark suite includes compact and larger solid blocks, overlapping
triangulated solids, 64 independent solids, a detailed triangulated shell,
a torus, containment, and a very large empty domain. Before/after measurements
use the same input hashes, spacing, and memory policy. A local-file manifest
allows the same runner to include user meshes without checking assets into the
repository. No user mesh was supplied during this implementation.

The pre-M3 baseline identifies CPU topology construction and GPU field queries
as material costs. The implementation changes are:

1. **Merge contiguous coordinate runs before expansion.** Each candidate block
   contributes x-runs for its cell and sample rows. Sort and merge these
   intervals, then emit canonical keys. This replaces sorting every individual
   key and removes duplicated halo samples before allocating the final arrays.
   The resulting key order and topology match M2 exactly, including partially
   clipped blocks and sparse domains crossing NanoVDB root tiles.
2. **Reuse a NanoVDB accessor within each cell.** Corner sampling and edge-ID
   lookup share a thread-local accessor cache instead of restarting at the
   tree root for every lookup. Caches hold no mutable grid data and are never
   shared across threads or calls.
3. **Visit nearer triangle bounds first for nearest-distance queries.** Median
   BVHs retain their existing structure; query ordering establishes tighter
   distance bounds earlier. Ray parity remains a separate stackless traversal
   with unchanged tie rules. The balanced tree and checked triangle limit bound
   traversal depth below 32, allowing a fixed 32-entry pending-node stack.
4. **Use a top-level shape-bounds hierarchy for scenes with more than four
   solids.** Leaves reference up to four shapes. Visit nearer bounds first;
   never prune a potentially containing shape using surface proximity. A
   shape outside a node's bounds cannot be inside any descendant solid. Bounds
   pruning therefore preserves both the signed union minimum and suppression
   of deeply buried surfaces. Small scenes use a separately compiled linear
   kernel, avoiding hierarchy traversal state when it provides no benefit.

No cached scene API, global geometry cache, or persistent CUDA buffer pool is
introduced. Inputs, streams, and resource budgets remain per-call, preserving
ownership and concurrency semantics. Additional performance work can be driven
by the benchmark manifest rather than expanding the public API speculatively.

## API, resources, and diagnostics

The [0.2 API contract](../api.md) specifies supported input forms, dtype and
copying rules, result ownership, error classes, validation modes, concurrency,
diagnostics, and numerical limitations. Invalid boolean/numeric option types
and integer-width overflow fail in Python before native conversion. The native
entry point retains its own option/geometry validation.

Metadata now reports the validation setting, requested memory limit, expanded
input counts, and a breakdown of topology construction into candidates, keys,
and NanoVDB indexing. Subtimings are nested separately to avoid double-counting
the total topology time. Allocation-budget diagnostics report requested, live,
and limit bytes. All explicit GPU buffers, including the new shape hierarchy,
remain accounted for. Kernel-local and runtime-managed GPU storage are excluded
from this explicit-buffer budget, as are host buffers.

## Authoritative version

The root `VERSION` file contains `0.2.0`, without a development suffix.
Scikit-build-core's built-in regex metadata provider reads it for wheel/sdist
metadata. CMake reads the same file before `project()`, tracks it as a configure
dependency, and generates the compiled version string. Python reads that native
version rather than maintaining another literal. `fuse::version()` and the
installed CMake package version expose it to native consumers. Tests compare
source, package metadata, Python, and native versions.

## Regression guardrails

`benchmarks/m3_performance.py` warms CUDA independently, excludes the first
reconstruction call from the median, and records seven warm calls by default.
For the same GPU/driver and identical input/options, `--check` fails on:

- warm median exceeding **1.5 × reference + 2 ms**;
- explicit peak device memory exceeding **1.05 × reference + 4096 bytes**;
- changed output counts, input hashes, options, or workload set.

These initial tolerances allow ordinary WSL/display scheduling noise. They are
engineering regression guards for the recorded hardware, not latency guarantees
or substitutes for geometric tests. Hardware/driver changes require a separate
baseline. Do not refresh baselines automatically after a failure. The numerical
and topology test suites remain the correctness gate.

See [M3 validation](../m3-validation.md) for measured results and remaining limits.
