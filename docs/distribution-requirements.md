# Distribution policy

This page describes the package's build and release targets. For installation
commands, use [Installation](installation.md); for publication procedures, use
the [maintainer guide](releasing.md).

## Wheel targets

- Linux x86-64 with a `manylinux_2_28_x86_64` release baseline.
- Standard CPython 3.9–3.14, with one wheel for each interpreter ABI.
- CUDA 12.8 static runtime, SM80/86/89/90 native code, and compute-90 PTX.
- NVIDIA Linux driver policy: 570.124.06 or newer. GPU runtime coverage and
  minimum-driver execution must be checked independently of successful builds.

The NVIDIA driver is not bundled. A compatible wheel must run without a CUDA
toolkit or compiler. Native C++ development and source builds still require the
toolkit. Windows, macOS, ARM, PyPy, and free-threaded Python are not included in
this matrix. A developer `linux_x86_64` wheel must not be relabeled as manylinux.

## Python dependencies

| Dependency | Constraint | Purpose |
| --- | --- | --- |
| NumPy | `>=1.23`, also `<2.1` on Python 3.9 | Mesh arrays |
| Trimesh | `>=4.12.2,<5` | File loading, export, and geometry helpers |
| fast-simplification | `0.1.12` on Python 3.9; `0.2.0` on Python 3.10+ | CPU decimation |
| Rtree | `1.4.1` | Simplification spatial queries |

These are standard dependencies; no CUDA, loading, or simplification extra is
required. Pip selects a NumPy distribution compatible with the interpreter.
Trimesh formats requiring additional optional dependencies are outside the
built-in loading guarantee. The constraints in [pyproject.toml](../pyproject.toml)
are authoritative.

## Build and release requirements

The root [VERSION](../VERSION) file supplies package and native versions.
Build requirements pin scikit-build-core and nanobind. NanoVDB is pinned in
[CMakeLists.txt](../CMakeLists.txt); source archives contain the required headers
and notices and must build without Git metadata or fetching a submodule.

GitHub Actions builds wheels and native tests from the source archive. Hosted
CPU checks cover import without a driver, input validation, and CPU utilities.
GPU qualification runs the installed artifacts across the declared GPU/Python
matrix in containers without a CUDA toolkit. SM90 qualification also checks
forced PTX JIT. A build passing on a hosted CPU runner does not replace these
GPU checks or a minimum-driver test.

Before publication, artifact checks must verify the version/tag, six wheel ABIs,
source contents, licenses, ELF dependencies, runtime search paths, and glibc
requirements. Registry publishing uses protected GitHub environments and Trusted
Publishing. Runtime components must be distributed under their applicable terms;
see [third-party notices](../licenses/THIRD-PARTY-NOTICES.txt).
