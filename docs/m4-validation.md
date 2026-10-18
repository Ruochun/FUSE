# M4 validation and release acceptance

M4 adds Python 3.9–3.14 packaging, GitHub wheel/native/GPU workflows, release
artifact validation, publishing gates, and project formatting. Version remains
0.2.0. Surface simplification is M5; the reconstruction algorithm is unchanged.

## Local environment

Validation began 18 October 2026 and continued 19 October (Asia/Shanghai).
Linux x86-64 / Ubuntu 22.04 under WSL, GCC 11.4, CUDA 12.8.93, RTX 4060 (SM89),
NVIDIA reported driver 595.97. Local wheels use the `linux_x86_64` tag and the
existing glibc 2.34 symbol floor; they are not portable release wheels.
All builds retain the full SM80/86/89/90 plus compute-90 PTX architecture set.

## Execution evidence

All six full-architecture local wheels were built, installed, and tested with
CUDA toolkit directories hidden. Trimesh was 4.12.2 throughout. These results
establish Python ABI/runtime coverage on this SM89 host, not portable-wheel or
multi-GPU qualification. Logs are under `build/m4-*.log`; wheels under `dist/m4/`.

| CPython | NumPy | Full suite | Time |
| --- | --- | --- | --- |
| 3.9.24 | 2.0.2 | 138 passed | 23.31 s |
| 3.10.19 | 2.2.6 | 138 passed | 23.02 s |
| 3.11.14 | 2.4.6 | 138 passed | 21.24 s |
| 3.12.12 | 2.5.3 | 138 passed | 21.73 s |
| 3.13.13 | 2.5.4 | 138 passed | 24.58 s |
| 3.14.0 | 2.3.4 | 138 passed | 22.23 s |

- Python 3.9 / NumPy 1.23.5 additionally passed all 138 tests (21.31 s).
- Dependency consistency checks passed in all seven runtime environments.
- Extracted source archive: a fresh full-architecture CPython 3.13 wheel built
  without Git/submodules and passed all 138 tests with toolkits hidden.
- Actual missing-driver probe: passed with both `/usr/local` and `/usr/lib/wsl`
  masked in a private namespace; import/metadata/empty input work, nonempty
  reconstruction and smoke report a driver error.
- Standalone CTest: all five tests passed, including native extraction cases.
- CPU-only test selection with `CUDA_VISIBLE_DEVICES=''`: 56 passed, 82 deselected.
- Artifact metadata, included licenses, static runtime dependency checks, and
  `twine check --strict` passed for all six local wheels, the source archive, and source-built wheel.
  The publication gate correctly rejects these nonportable development wheels.

The Python suite covers both reconstruction backends, analytic/reference fields,
closed output topology, loading multiple formats, scene transforms, hierarchy
pruning, ownership/concurrency, failure behavior, and release metadata rejection.
Toolkit-hidden checks run in a private mount namespace with `/usr/local` masked
and CUDA/Python library-path overrides removed. They do not modify the host.

## Formatting

Formatter check mode, actionlint, and ShellCheck passed.
First-party C++/CUDA, Python, and CMake sources were formatted with the pinned
clang-format 18.1.8, Black 25.1.0, and cmake-format 0.6.13 tools. Configuration and
`./.format_all --check` are shared with CI. External OpenVDB sources were not
modified. All 14 modified existing Python files have identical parsed ASTs before/after
formatting; all 19 modified C++/CUDA files exactly match clang-format applied to
the committed originals. Formatting does not change the algorithm or public API.

## External acceptance still required

The local machine has no Docker engine, GitHub runner administration, or registry
publishing session available. The following are configured but not claimed as
executed:

- Digest-pinned manylinux CUDA container builds, auditwheel repair, and hosted
  GitHub Actions execution for all six Python ABIs.
- Driver-only GPU container execution on SM80, SM86, and SM90; forced compute-90
  PTX JIT on SM90; validation of the minimum driver. Local GPU evidence is SM89
  on the current driver, not the entire promised release matrix.
- GitHub protected environments and registry Trusted Publisher setup, final
  redistribution review, TestPyPI upload/install rehearsal, and public PyPI upload.

Public PyPI and TestPyPI project JSON endpoints returned 404 at preparation time.
That is only an availability observation, not a reservation or publishing right.
No upload, remote push, release tag, or public release was performed.

[Release instructions](releasing.md) describe the exact remaining setup and gates.
M4 implementation is reviewable locally; public release acceptance remains pending
those external executions. Do not announce full release qualification yet.
