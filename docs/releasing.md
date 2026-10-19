# Building, validating, and releasing wheels

This is the maintainer guide. Users should start with
[installation](installation.md) and [mesh loading](mesh-loading.md).

## Release matrix

- Linux x86-64, glibc 2.28 or newer (`manylinux_2_28_x86_64`).
- Standard CPython 3.9, 3.10, 3.11, 3.12, 3.13, and 3.14, one wheel per ABI.
  Free-threaded Python, PyPy, Windows, macOS, and ARM are outside this release.
- CUDA 12.8 static runtime; native images for SM 80, 86, 89, 90 and compute-90
  PTX. These are compiled targets, not mandatory GPU test jobs. Forward PTX
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

Manual dispatch exposes `publish` (`none`, `testpypi`, or `pypi`) and `gpu`
(`none`, `sm89`, `sm80`, `sm86`, or `sm90`). Both default to `none`. The default
build and publish path uses only GitHub-hosted CPU runners; no GPU runner is
required. CPU tests and artifact checks still run for all six Python versions.
A successful hosted build does not establish GPU runtime compatibility.

To validate on one GPU architecture, select its `gpu` value when dispatching on
trusted `main` or a `v*` tag. Provision a self-hosted Linux x86-64 runner labelled
`fuse-gpu` plus that architecture, for example `sm89` for an RTX 4060. It needs
Docker, NVIDIA Container Toolkit, a single visible matching GPU, and a compatible
driver. The six Python jobs can run sequentially on one runner. Additional GPU
architectures can be checked with separate `publish=none` dispatches. Never run
pull-request code on these persistent runners.

Each selected architecture runs the installed artifacts and native tests in
fresh Python Debian Bookworm containers without a CUDA toolkit. GPU identity,
driver, resolved container identity, installed packages, loaded driver libraries,
and JUnit reports are uploaded. Selecting `sm90` also checks forced PTX JIT;
other architectures use their native images. A selected GPU job must pass before
publication; a failed or cancelled run cannot silently become an unchecked release.
If the requested runner is unavailable, publication waits for it.

Record the GPU and driver combinations actually tested with each release.
Minimum-driver execution and untested architectures remain separate qualification
work; successful compilation alone does not establish that coverage.

## Publication setup and rehearsal

Repository owners configure these external resources once:

1. Create GitHub environments `testpypi` and `pypi`, allowing `main` and release
   tags (`v*`). Add required reviewers where available if an upload should need
   approval after validation.
2. Configure Trusted Publishers on both registries for owner `Ruochun`,
   repository `FUSE`, workflow `wheels.yml`, and the matching environment.
   For a new project, use a pending publisher named `fusemesh`; for an existing
   project, configure publishing as an owner. No long-lived upload token is
   stored in the repository. A pending publisher does not reserve the name.
3. If GPU validation is desired, configure one matching runner as described above.
   This is optional; `gpu=none` needs no self-hosted runner.

Set the root `VERSION` to an unused release version, commit the release sources,
and push `main`. To rehearse, open **Actions → Wheels and release → Run workflow**,
choose `main`, set `publish=testpypi`, and leave `gpu=none` or select your runner's
architecture. Approve the environment deployment if configured. With GitHub CLI:

```bash
gh workflow run wheels.yml --repo Ruochun/FUSE --ref main \
  -f publish=testpypi -f gpu=none
```

The workflow builds and validates six wheels and one source archive, then uploads
them using OIDC. Install the TestPyPI wheel in a fresh environment, resolving
its dependencies from ordinary PyPI:

```bash
release_version=$(cat VERSION)
python -m pip download --no-deps --only-binary=:all: \
  --index-url https://test.pypi.org/simple/ "fusemesh==$release_version" -d test-wheel
python -m pip install test-wheel/fusemesh-*.whl
python -c 'import fusemesh; print(fusemesh.build_info())'
```

Run the CPU tests and, on a compatible GPU, the reconstruction tests and demos.
After verifying the rehearsal, dispatch the same release commit with `publish=pypi`:

```bash
gh workflow run wheels.yml --repo Ruochun/FUSE --ref main \
  -f publish=pypi -f gpu=none
```

For an immutable source reference, create a tag `vVERSION` at that commit and use
it instead of `main` in both dispatches. Tagged publication verifies that the tag
matches `VERSION`. Only manual dispatches on `main` or `v*` tags can publish;
pushes and pull requests never upload to a registry. Production dispatch rebuilds
and retests the artifacts. Ensure `main` has not advanced between rehearsal and
production, or use the tag route.

Once uploaded, users can install `fusemesh` (or `fusemesh[quality]` for multi-solid
quality comparisons). `fusemesh-quality` is installed as a command. Demo scripts
remain in the repository/source archive and must be downloaded separately.
PyPI release filenames cannot be reused; increment `VERSION` for changed builds.
For local iterations use `--force-reinstall path/to/wheel.whl`; add `--no-deps`
only when dependencies are already installed and unchanged.

Before enabling publication, review the bundled notices and NVIDIA redistribution
terms against the exact toolkit image. The wheel includes FUSE's BSD license,
NanoVDB, nanobind/robin_map, CUB/libcu++, and CUDA runtime notices. It bundles no
NVIDIA driver and requires no dynamic toolkit libraries. Native consumers/source
builders still need CUDA 12.8 and C++17; this does not change the wheel promise.
