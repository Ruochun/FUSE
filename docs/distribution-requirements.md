# Distribution requirements and outstanding implementation

Agreed requirements, 18 October 2026. This records the intended release
contract; it does not imply that the full compatibility matrix is implemented
or tested. Architecture and reconstruction milestones are defined in
[FUSE-design.md](FUSE-design.md).

## User installation contract

- The intended installation command is `pip install fusemesh` from PyPI.
  A `[cuda]` extra is unnecessary and will not be required.
- On supported platforms, install a precompiled wheel. Users must not need to
  install the CUDA toolkit, nvcc, CMake, Git, or a C++ compiler themselves.
- Users need a supported NVIDIA GPU and compatible NVIDIA driver. The driver
  is a system prerequisite and is not bundled or installed by the package.
- Retain the current statically linked CUDA runtime approach initially. If
  additional CUDA libraries become necessary, explicitly package their
  permitted runtime components or declare suitable runtime dependencies;
  never rely on toolkit files accidentally available on the build machine.
- Target at least standard CPython 3.9 through 3.14, inclusive. Free-threaded
  Python builds and alternative interpreters are not implied by this target.
- Build release wheels through GitHub Actions. Publish only artifacts that
  satisfy the documented platform, Python, GPU, and driver compatibility policy.

## Current M0 evidence

- CMake builds a standalone C++/CUDA library and a nanobind Python extension;
  scikit-build-core produces the `fusemesh` wheel.
- NanoVDB is pinned to an exact upstream commit. Source archives include its
  sources and do not require fetching or initializing a submodule at build time.
- A local Linux x86-64 / CPython 3.13 wheel was installed and tested on an RTX
  4060 with CUDA toolkit directories hidden. The CUDA runtime is statically
  linked, and dependency inspection found no toolkit runtime search path.
- Native and Python smoke tests pass. The source-archive wheel also passes.
- Third-party license notices are included in wheel metadata.

See [M0 validation](m0-validation.md) for the actual environment, checks, and
limits. M0 established diagnostics only. M1 now provides the dense reconstruction
baseline; see [M1 validation](m1-validation.md). M2 adds default sparse NanoVDB
reconstruction; see [M2 validation](m2-validation.md). M3 adds measured
performance improvements, regression guardrails, and a documented 0.2 API;
see [M3 validation](m3-validation.md). `VERSION` is authoritative for package
and native versioning; the current release version is `0.2.0`. This version
number does not imply public publication or completion of the release matrix.

## File-loading dependency

Trimesh is now a standard Python runtime dependency, installed by ordinary
`pip install fusemesh` (no I/O extra). The adapter currently requires
`trimesh>=4.12.2,<5`; version 4.12.2 was tested on CPython 3.13 and declares
Python >=3.8, so it does not itself rule out the future Python 3.9 target.
The portable-wheel Python matrix remains subject to M4 release acceptance. No Trimesh soft extras or new
native submodules are required for the tested OFF/OBJ/GLB path.
See [mesh loading](mesh-loading.md).

## M4 implementation and remaining acceptance

See [release instructions](releasing.md) for the authoritative platform policy,
workflow setup, formatting commands, and publication procedure. See
[M4 validation](m4-validation.md) for measured local results. A configured CI job
is not a completed release qualification.

| Requirement | Implementation and remaining acceptance | Milestone |
| --- | --- | --- |
| CPython 3.9–3.14 | Metadata and CMake now permit 3.9; six wheel ABIs are configured. Local GPU matrix evidence is in M4 validation. Run the portable-wheel matrix in GitHub before release. | M4 |
| Binding compatibility | Build requirements pin nanobind 2.15.0 and scikit-build-core 0.11.6. NumPy has a Python-3.9-compatible bound; other interpreters resolve compatible distributions. | M4 |
| GitHub wheel automation | `wheels.yml` builds each wheel from a self-contained sdist using cibuildwheel and a digest-pinned CUDA 12.8 manylinux image. Run the workflow on the remote repository. | M4 |
| GPU CI | Driver-only container tests, standalone native tests, GPU identity checks, and evidence uploads are configured for SM80/86/89/90 and six Python versions. Provision the required self-hosted runners and execute the matrix. | M4 |
| Portable wheels | Release baseline is manylinux_2_28 Linux x86-64; auditwheel and artifact checks reject newer glibc or toolkit dependencies. Local Ubuntu wheels remain nonportable development artifacts. Container build execution is still required. | M4 |
| Platform support | Initial release is Linux x86-64 only. Windows, macOS, ARM, PyPy, and free-threaded Python are not part of this matrix. | M4 |
| GPU/driver coverage | CUDA 12.8, SM80/86/89/90 native code and compute-90 PTX; conservative driver floor 570.124.06. Validate all architectures, forced PTX on SM90, and minimum/current drivers. Local execution covers SM89 only. | M4 |
| Toolkit-free execution | Full reconstruction tests run locally with toolkit directories hidden; GitHub GPU jobs use Python containers with no toolkit. Expand execution to all release GPU/platform combinations. | M4 |
| Failure diagnostics | Existing tests cover invalid devices, unavailable GPUs, input errors, and memory budgets. Hosted CPU jobs include an actual missing-driver probe and pre-CUDA validation without GPU passthrough. The missing-driver probe also passed locally with driver directories hidden. Physical old-driver/unsupported-GPU qualification remains external work. | M4 |
| Source release | Sources include pinned NanoVDB, tests, scripts, formatting files, and notices. All CI wheels and native tests build from the archive. Local source-build evidence is recorded separately. | M4 |
| Publication | Artifact checks, Trusted Publishing jobs, tag/version gates, and TestPyPI instructions are implemented. Configure protected environments/registry publishers, review redistribution terms, rehearse TestPyPI installs, then publish. No registry upload has occurred. | M4 |
| Formatting | `.clang-format`, Black settings, CMake formatting settings, pinned tool requirements, and `.format_all --check` are implemented. First-party sources are formatted; external submodules are excluded. | M4 |
| Surface simplification | Optional reduction of triangles on simple surfaces, with geometric/topological validation; no backend selected yet. Adaptive voxel refinement remains deferred. | M5 |

References:

- [Nanobind changelog](https://nanobind.readthedocs.io/en/latest/changelog.html)
- [cibuildwheel CUDA and CI guidance](https://cibuildwheel.pypa.io/en/stable/faq/)
