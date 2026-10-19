# Installation

## Requirements

| Component | Requirement |
| --- | --- |
| Platform | Linux x86-64; portable release wheels target glibc 2.28 or newer |
| Python | Standard CPython 3.9–3.14; install the wheel matching your Python ABI |
| GPU reconstruction | NVIDIA GPU with compatible compiled code and a supported driver |
| Driver policy | NVIDIA Linux driver 570.124.06 or newer; see qualification notes below |
| CPU utilities | Loading, saving, and simplification require no GPU or CUDA initialization |

Wheels compile native code for SM80, SM86, SM89, and SM90, plus compute-90 PTX.
Local GPU execution has been checked on an RTX 4060 (SM89). Compiled architecture
coverage does not establish runtime qualification on every GPU or driver;
minimum-driver and other-architecture qualification belong to the release process.
Check release-specific compatibility information before deploying on another GPU.

Windows, macOS, ARM, PyPy, and free-threaded Python are outside the wheel matrix.
The package provides no CPU reconstruction fallback. Python dependencies install
automatically, including NumPy, Trimesh, fast-simplification, and Rtree.
For [quality comparisons](quality.md) involving multiple original solids, install
the optional `quality` extra (`pip install "fusemesh[quality]"`).

## Install a prebuilt wheel

Open the repository's [wheel workflow](https://github.com/Ruochun/FUSE/actions/workflows/wheels.yml)
and select a successful build of the revision you need. Download the artifact
for your interpreter (`wheel-cp39` for Python 3.9, `wheel-cp313` for Python 3.13,
etc.), extract its ZIP, and install the `.whl` file:

```bash
python -m pip install /path/to/the-matching-wheel.whl
```

Use the actual downloaded filename and one wheel at a time. An Actions artifact
is a build output, not necessarily a published or fully qualified release.
A local `linux_x86_64` wheel may require newer system libraries than a portable
`manylinux_2_28_x86_64` wheel; use it on a compatible build/runtime system.

The CUDA runtime is statically linked into the extension. Users of a compatible
wheel do not need the CUDA toolkit, `nvcc`, CMake, or a C++ compiler. The NVIDIA
driver is a system prerequisite for GPU work and is not installed by pip.

Check the installation:

```bash
python -c 'import fusemesh; print(fusemesh.__version__); print(fusemesh.build_info())'
python -c 'import fusemesh; print(fusemesh.smoke_test())'
```

The first command works without a GPU. The second runs a small CUDA check and
should return `[3.0, -2.0, 0.5, 7.0]`.

## Reinstall a rebuilt wheel

Pip does not compare the contents of wheels with the same version number. To
replace an installed build:

```bash
python -m pip install --force-reinstall /path/to/the-rebuilt-wheel.whl
```

Add `--no-deps` only if its dependencies are already installed and unchanged.
Restart Python processes that loaded the old extension. Run `python -m pip check`
to verify dependency compatibility.

## Build from source

Source builds require a CUDA **12.8.x** toolkit, a compatible C++17 compiler,
and CMake 3.24 or newer. The Python build backend can install CMake and the other
Python build requirements. Select `nvcc` using `CUDACXX` if it is not on PATH.

From a repository checkout:

```bash
git submodule update --init --recursive
python -m pip install build
python -m build --wheel --outdir dist/local
```

Install the matching wheel from `dist/local` using the command above. Use a clean
output directory or select the exact filename if it contains older builds.
Source archives include the pinned NanoVDB headers, so builds from an extracted
archive do not need Git or submodule initialization. No OpenVDB CPU library or
TBB is required.

For a native C++ library without Python bindings:

```bash
cmake -S . -B build/native -DFUSE_BUILD_PYTHON=OFF -DFUSE_BUILD_TESTS=ON \
  -DCMAKE_BUILD_TYPE=Release
cmake --build build/native -j2
ctest --test-dir build/native --output-on-failure
cmake --install build/native --prefix /path/to/install
```

CMake consumers use `find_package(FUSE CONFIG REQUIRED)` and link `FUSE::fuse`.
They need the CUDA toolkit to link the static library. See the [API reference](api.md)
and [maintainer guide](releasing.md) for native ownership and release builds.
