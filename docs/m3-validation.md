# M3 validation and performance baseline

18 October 2026. Package version **0.2.0**, read from the root `VERSION` file.
See [M3 decisions](decisions/0004-m3-performance-api.md) and the
[0.2 API contract](api.md). Validation uses CPython 3.13.13, GCC 11.4,
CUDA 12.8.93, Linux x86-64 under WSL, and RTX 4060 (SM 89).

## Acceptance evidence

- A full-architecture wheel was built from a fresh build directory, installed,
  and tested with CUDA toolkit directories hidden. **131 Python tests passed**,
  including all existing geometry/I/O cases and new hierarchy, API diagnostics,
  ownership, version-consistency, and benchmark-gate tests.
- **5 native CTest executables passed**, including version consistency and
  768 synthetic sign/magnitude configurations checked on sparse and dense
  extraction. The host contour tests still cover 1,024 configurations.
- Targeted Compute Sanitizer checks reported **zero errors**: memory checks of
  native exhaustive extraction, and memory/initialization checks of the shape
  hierarchy fixtures on both backends, including deep containment.
- An external C++ consumer used the installed `FUSE::fuse` target with exact
  package-version discovery, checked `fuse::version()`, and reconstructed a
  tetrahedron with a diagnostic field (1,080 output triangles).
- The root version, installed distribution metadata, Python `__version__`,
  native build information, and CMake package version agree on **0.2.0**.
- The wheel compiles SM 80/86/89/90 plus compute-90 PTX. Execution remains tested
  only on SM 89. Dependency inspection shows no dynamic CUDA runtime or toolkit
  RPATH/RUNPATH; `auditwheel show` reports the existing glibc 2.34 symbol floor.
  This remains a local Linux wheel, not the M4 public release.

## Benchmark results

The same eight workloads were measured on the same host before and after M3.
The runner warms CUDA separately, records the first reconstruction call, then
uses the median of seven warm calls. Input hashes, options, and output counts
agree between the before/after records. No dense diagnostic field is requested.

| Workload | Before ms | M3 ms | Speedup | M3 device MiB |
| --- | ---: | ---: | ---: | ---: |
| small_cube | 12.28 | 9.63 | 1.28× | 2.60 |
| large_cube | 66.02 | 45.28 | 1.46× | 25.00 |
| overlap_spheres | 78.51 | 45.81 | 1.71× | 4.67 |
| many_solids | 52.07 | 30.29 | 1.72× | 11.64 |
| detailed_shell | 90.46 | 55.96 | 1.62× | 3.25 |
| torus | 45.19 | 26.83 | 1.68× | 3.94 |
| containment | 44.33 | 32.89 | 1.35× | 16.37 |
| distant_cubes | 6.37 | 6.24 | 1.02× | 0.75 |

Raw records:

- [Before M3: 0.0.1.dev2](benchmarks/m3-before-rtx4060.json)
- [M3: 0.2.0](benchmarks/m3-rtx4060.json)

These fixtures represent different geometry/workload classes, not a supplied
production asset corpus. No user mesh was supplied during implementation. The
manifest option below adds real files to the same measurement procedure.
Sparse storage is not always faster or smaller than dense storage on compact
scenes; the dense backend remains available. Near-first BVH traversal and
interval-based key construction improve the measured sparse path without
changing resolution or extraction semantics.

## Regression policy

For the same inputs/options and GPU/driver, the check fails if warm median time
exceeds **1.5 times the reference plus 2 ms**, or explicit peak device memory
exceeds **1.05 times the reference plus 4096 bytes**. It also rejects changed
input hashes, options, output counts, or workload sets. Unit tests verify both
accepted jitter and deliberately injected regressions. A subsequent full
benchmark run passed this gate with no regressions.

Run on the same host with comparable CPU/GPU load and power settings. The tool
checks the GPU/driver signature but does not normalize host CPU speed, display
load, thermal state, or scheduling jitter. Treat a failed timing guard as a
reason to investigate; do not automatically replace the baseline. These are
initial engineering thresholds, not latency guarantees. Geometry tests remain
the correctness gate.

## Reproduce and add real meshes

```sh
python -m build --wheel
python -m pip install --upgrade dist/fusemesh-0.2.0-*.whl
scripts/test_without_toolkit.sh .venv-runtime/bin/python

python benchmarks/m3_performance.py --output build/m3.json
python benchmarks/m3_performance.py \
  --baseline docs/benchmarks/m3-rtx4060.json --check --output build/m3-check.json
```

For your own files, create a manifest with paths relative to the manifest:

```json
[
  {
    "name": "my_part",
    "paths": ["meshes/part.off"],
    "voxel_size": 0.05,
    "validation": "full",
    "backend": "sparse",
    "max_memory_bytes": 536870912
  }
]
```

```sh
python benchmarks/m3_performance.py --manifest workloads.json --output build/my-baseline.json
python benchmarks/m3_performance.py --manifest workloads.json \
  --baseline build/my-baseline.json --check --output build/my-check.json
```

File loading time is recorded separately; the timed reconstruction uses the
loaded array snapshots. For native checks, configure with `FUSE_BUILD_TESTS=ON`,
build, and run CTest as in the README. The new native version test requires no
GPU. Compute Sanitizer examples:

```sh
compute-sanitizer --tool memcheck --error-exitcode 99 build/native/fuse_extraction
compute-sanitizer --tool memcheck --error-exitcode 99 python -m pytest -q tests/test_m3.py -k hierarchy
compute-sanitizer --tool initcheck --error-exitcode 99 python -m pytest -q tests/test_m3.py -k hierarchy
```

## Remaining limits

CPU candidate/NanoVDB construction remains a cost, particularly on small
scenes. No persistent cache or buffer pool is introduced. Device-buffer memory
counts exclude host, runtime/context, and kernel-local storage. Uniform spacing,
valid-solid assumptions, fixed-resolution feature losses, and the existing
cell-interior ambiguity policy remain unchanged. Adaptive refinement is deferred.

Other Python versions, GPUs/drivers, Windows, portable manylinux environments,
GitHub/GPU CI, source-release rehearsal, and public publishing remain M4 work.
The `0.2.0` version does not imply that this compatibility matrix is complete.
