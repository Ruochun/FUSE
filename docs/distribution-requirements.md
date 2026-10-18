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
limits. M0 exposes diagnostics only, not mesh reconstruction.

## Work still required

| Requirement | Outstanding implementation and acceptance evidence | Milestone |
| --- | --- | --- |
| CPython 3.9–3.14 support | Lower the Python minimum in both `pyproject.toml` and CMake only alongside compatibility work. Select compatible nanobind, NumPy, and build/test dependency versions; build, install, and execute tests for all six Python versions. Current metadata requires 3.10+, and only 3.13 has been tested. | M3–M4 |
| Binding dependency compatibility | Preserve a Python-3.9-compatible binding toolchain. The current nanobind 2.x constraint allows this; nanobind 3.x requires Python 3.10+. Any upgrade must retain the promised Python matrix or explicitly revise the support policy. | M3–M4 |
| GitHub wheel automation | Add GitHub Actions workflows, preferably using cibuildwheel, with pinned build environments, initialized submodules, explicit CUDA architecture targets, recorded tool versions, and uploaded artifacts. No workflow is implemented yet. | M4 |
| GPU CI execution | Provide GPU-capable runners, such as self-hosted runners, to install and exercise the built wheels. Compilation can run without a GPU; import-only checks are insufficient for release acceptance. | M4 |
| Portable Linux wheels | Choose a manylinux baseline, build in an appropriate CUDA-equipped environment, inspect/repair dependencies, and validate clean installations. The current local wheel retains a `linux_x86_64` tag; its observed symbol floor is glibc 2.34. | M4 |
| Platform support | Finalize OS/CPU architecture coverage. If Windows is included in the release matrix, add separate Windows builds, DLL dependency inspection/repair, and GPU tests. CUDA execution on macOS is outside the current backend scope. | M4 |
| GPU and driver support | Finalize and document the architecture set, CUDA family, minimum driver, and PTX policy. Execute compatibility tests on representative supported hardware and drivers; only SM 89 has been run in M0. | M3–M4 |
| Toolkit-free reconstruction | Repeat clean-environment tests with actual reconstruction after M1/M2 and after any native dependency change. Verify both direct dependencies and libraries loaded at runtime. The M0 proof covers only the diagnostic operation. | M1–M4 |
| Failure diagnostics | Extend tests for missing/incompatible drivers, unsupported devices, and resource failures as the reconstruction backend develops. M0 covers invalid devices and no-visible-GPU errors, but not an actual missing-driver host. | M1–M4 |
| Reproducible source releases | Continue verifying that source archives contain pinned native sources and build independently of the checkout. Source builds may require a compiler and CUDA toolkit; the toolkit-free promise applies to supported binary wheels. | M4 |
| Public distribution | Check package-name availability, audit final redistribution terms/notices and artifact size, document compatibility and installation, rehearse on TestPyPI, and configure release publishing. No package has been published. | M4 |

The reconstruction implementation remains the priority for M1–M3. These release
requirements should constrain dependency choices now, without presenting the
unimplemented matrix as supported.

References:

- [Nanobind changelog](https://nanobind.readthedocs.io/en/latest/changelog.html)
- [cibuildwheel CUDA and CI guidance](https://cibuildwheel.pypa.io/en/stable/faq/)
