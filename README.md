# FUSE — Fast Union Surface Extraction

C++/CUDA surface reconstruction with NanoVDB and a NumPy-facing Python package.
**Milestone 0:** packaging and GPU dependency proof only. Mesh reconstruction is
not implemented yet.

## Build and run

The initial development target is Linux x86-64, CUDA toolkit 12.8, GCC 11,
Python 3.10+ (tested on 3.13), and an NVIDIA Ampere/Ada/Hopper GPU. The initial
conservative driver requirement is Linux 570.26 or newer. Other configurations
are not yet validated. Wheels are local Linux artifacts, not public manylinux
releases.

```sh
git submodule update --init --recursive
python -m pip install build
python -m build
python -m pip install dist/*.whl
python -c 'import fusemesh; print(fusemesh.build_info()); print(fusemesh.smoke_test())'
```

Expected samples: `float32 [3, -2, 0.5, 7]`. These are values read by a CUDA
kernel from a small NanoVDB grid constructed on the host, including negative
coordinates, multiple leaves, and an inactive background sample. This does not
validate GPU topology construction or surface extraction.

`smoke_test(device=0)` is synchronous, releases the Python GIL, returns an owned
NumPy array, and restores the caller's CUDA device. Import and `build_info()` do
not initialize CUDA. Invalid devices raise `ValueError`; driver/device/runtime
failures raise `RuntimeError`. This diagnostic API is provisional.

The wheel statically links the CUDA runtime and needs an NVIDIA driver, but no
CUDA toolkit or compiler at runtime. The source build needs CUDA 12.8 and a C++
compiler. No OpenVDB CPU library, TBB, upstream Python bindings, or viewers are
built. The source archive includes NanoVDB headers and can build without Git.

## Native library and tests

```sh
cmake -S . -B build/native -DFUSE_BUILD_PYTHON=OFF -DFUSE_BUILD_TESTS=ON
cmake --build build/native -j2
ctest --test-dir build/native --output-on-failure
python -m pip install pytest
python -m pytest
```

CMake 3.24+ is required (the Python build installs it automatically as needed).
The native `FUSE::fuse` target exposes `include/fuse/smoke.h` without Python,
NanoVDB, or CUDA types in its public interface. Native development consumers
need the CUDA toolkit to link the static library.

See [the design](docs/FUSE-design.md), [distribution requirements](docs/distribution-requirements.md),
[M0 decisions](docs/decisions/0001-m0-build.md), and
[dependency inventory](docs/nanovdb-inventory.md). FUSE uses BSD-3-Clause;
third-party terms are in `licenses/` and the wheel's license metadata.
