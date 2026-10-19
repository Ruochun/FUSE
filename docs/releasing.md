# Building, validating, and releasing wheels

This is the maintainer guide. Users should start with
[installation](installation.md) and [mesh loading](mesh-loading.md).

## Release matrix

- Linux x86-64, glibc 2.28 or newer (`manylinux_2_28_x86_64`).
- Standard CPython 3.9, 3.10, 3.11, 3.12, 3.13, and 3.14, one wheel per ABI.
  Free-threaded Python, PyPy, Windows, macOS, and ARM are outside this release.
- CUDA 12.8 static runtime; native images for SM 80, 86, 89, 90 and compute-90
  PTX. Release validation requires these four GPU architectures. Forward PTX
  is included but does not establish support for untested future GPUs.
- NVIDIA Linux driver 570.124.06 or newer. This conservative floor follows the
  [CUDA 12.8 Update 1 release notes](https://docs.nvidia.com/cuda/archive/12.8.1/cuda-toolkit-release-notes/index.html),
  rather than promising older-driver CUDA minor-version compatibility.
  Minimum-driver execution must be validated before claiming that floor.
- NumPy starts at 1.23; Python 3.9 uses versions below 2.1. The resolver chooses
  a compatible release on other interpreters (including Python 3.14).
  Trimesh, fast-simplification, and Rtree are normal dependencies; see
  [distribution policy](distribution-requirements.md) for the Python-specific pins.
  The optional `quality` extra installs Manifold for multi-solid reference unions;
  wheel tests include it to exercise this path.
  Build tools pin nanobind 2.15.0 and scikit-build-core 0.11.6 to keep Python 3.9 support reproducible.

After publication, install with `python -m pip install --only-binary=:all: fusemesh`.
GPU reconstruction requires an NVIDIA driver; the CUDA toolkit and a compiler
are not needed for wheel use. Ordinary `pip install fusemesh` also works on supported wheel platforms; unsupported
platforms may attempt a source build. Before publication, install the matching
wheel downloaded from GitHub Actions. An artifact version alone does not imply a
published or fully qualified release.

## Formatting

Install `requirements/format.txt` into a development environment and activate it:

```bash
python -m pip install -r requirements/format.txt
./.format_all
./.format_all --check
```

CI also runs actionlint and ShellCheck from the pinned requirements.
`.clang-format` owns C++/CUDA style, `pyproject.toml` owns Black style, and
`.cmake-format.yaml` owns CMake style. It visits only first-party source directories and CMakeLists.txt;
third-party submodules, build output, and virtual environments are excluded.
Missing tools are errors. `PYTHON` may select the utility's interpreter; formatter
executables must be on PATH. CI uses the same pinned tool versions and check mode.

## GitHub build and GPU checks

`.github/workflows/wheels.yml` runs formatting, builds an sdist, and builds six
wheels from that archive on GitHub-hosted Linux runners. No source checkout or
submodule is used inside the wheel build. cibuildwheel uses the official
manylinux CUDA 12.8 image pinned by manifest digest in `pyproject.toml`, repairs
wheels with auditwheel, and runs a missing-driver probe plus all CPU tests on
the installed artifacts.
A separate job builds the standalone native tests from that same archive; GPU
jobs execute those binaries alongside the Python suites.
The image baseline is glibc 2.28. Do not relabel a wheel built on a newer host.

`check_artifacts.py` checks versions, ABI/platform tags, license notices, source
contents, ELF dependencies, runtime search paths, and the glibc floor. Publishing
requires exactly six wheels and one source archive. Test commands resolve the
installed extension; no in-tree Python package is added to PYTHONPATH.

For GPU acceptance, manually dispatch the workflow on trusted `main` or a `v*`
tag with `publish=none`. Provision self-hosted Linux x86-64 runners with these
labels: `fuse-gpu` plus one of `sm80`, `sm86`, `sm89`, `sm90`. Each needs Docker,
NVIDIA Container Toolkit, a single visible matching GPU, and a compatible driver.
Use dedicated runners; pull-request events never schedule these jobs. A trusted
branch/tag is still trusted code with access to that runner.

Each architecture runs all six Python versions in a fresh Python Debian Bookworm
container, which contains no CUDA toolkit. The wheel is installed only from the
current workflow artifact; dependencies are installed separately. GPU identity,
driver, resolved container identity, installed packages, loaded driver libraries,
and JUnit reports are uploaded. GPU tests must execute, not skip. The SM90 job
also forces PTX JIT for smoke and hierarchy correctness. Older GPUs cannot execute
compute-90 PTX, so their normal tests exercise the native images.

Arrange a runner with the declared minimum driver as well as current drivers
before approving a release. The workflow checks the minimum version but does
not by itself prove execution on the oldest supported driver. Retain this extra
acceptance evidence with the release; do not claim untested GPU/driver coverage.

## Publication setup and rehearsal

Repository owners must configure the following external resources:

1. The four GPU runner labels and runtime access described above.
2. GitHub environments `testpypi` and `pypi`, restricted to release tags, with
   required reviewers. Configure Trusted Publishers on both registries for
   owner `Ruochun`, repository `FUSE`, workflow `wheels.yml`, and the matching
   environment. No long-lived upload token is stored in the repository.
3. Reserve/establish the `fusemesh` project as allowed by the registries. Project-name
   availability must be checked directly with the registry before publication.

Set `VERSION`, commit the exact release sources, and create a matching `vVERSION`
tag. Dispatch that tag with `publish=testpypi`. After all build/GPU gates pass,
the protected environment controls OIDC publication. Install the TestPyPI wheel
in a fresh driver-only environment, choosing dependencies from ordinary PyPI:

```bash
release_version=$(cat VERSION)
python -m pip download --no-deps --only-binary=:all: \
  --index-url https://test.pypi.org/simple/ "fusemesh==$release_version" -d test-wheel
python -m pip install test-wheel/fusemesh-*.whl
python -c 'import fusemesh; print(fusemesh.build_info()); print(fusemesh.smoke_test())'
```

Run the full test suite, including simplification, too. Repeat on the release
matrix. Only then dispatch the same tag with `publish=pypi`. This rebuilds and
retests artifacts; all gates apply again. PyPI versions are immutable; do not overwrite or assume
`pip install` replaces an already installed build of the same version. For local
iterations use `--force-reinstall path/to/wheel.whl`; add `--no-deps` only when
dependencies are already installed and unchanged.

Before enabling publication, review the bundled notices and NVIDIA redistribution
terms against the exact toolkit image. The wheel includes FUSE's BSD license,
NanoVDB, nanobind/robin_map, CUB/libcu++, and CUDA runtime notices. It bundles no
NVIDIA driver and requires no dynamic toolkit libraries. Native consumers/source
builders still need CUDA 12.8 and C++17; this does not change the wheel promise.
