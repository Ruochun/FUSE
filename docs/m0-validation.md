# M0 validation

Validated locally on 18 October 2026. This is a build/dependency proof, not a
reconstruction or performance result.

## Environment

| Item | Observed value |
| --- | --- |
| OS | Ubuntu 22.04.5 x86-64, WSL NVIDIA driver interface |
| Python | CPython 3.13.13 |
| Compiler | GCC 11.4.0, nvcc 12.8.93 |
| GPU | NVIDIA GeForce RTX 4060, 8 GiB, SM 89 |
| Driver | nvidia-smi reports 595.58.02 / Windows host 595.97 |
| Build tooling | CMake 4.4.4, nanobind 2.15.0, scikit-build-core 0.11.6 |
| NanoVDB | OpenVDB v13.1.0, e26f5a569201d3236e0c30580572101022d3d754 |

## Checks

- Built `fusemesh-0.0.1.dev0-cp313-cp313-linux_x86_64.whl` locally.
- Installed into a separate virtual environment containing NumPy and pytest,
  with no native build tools or binding dependencies installed in that venv.
- Ran six Python tests inside a private Linux mount namespace with all of
  `/usr/local` hidden by an empty filesystem (covering every installed CUDA
  toolkit on this host), toolkit environment variables unset, and `PATH`
  restricted to the runtime venv. Working directory was `/tmp`; the installed
  extension was used. All six tests passed.
- Tested fixed GPU samples, owning result lifetime across calls, concurrent
  calls, invalid devices, and no-visible-device diagnostics. Import and build
  metadata remain available with no GPU visible.
- Built the native library and test executable with `FUSE_BUILD_PYTHON=OFF`;
  CTest passed the GPU readback test.
- Built the self-contained source archive and rebuilt its wheel from the
  extracted archive with no Git metadata or upstream checkout. The archive
  includes NanoVDB sources and the upstream license, excluding unrelated
  OpenVDB source trees. Python build dependencies were already installed.
- Inspected the wheel license metadata: FUSE, NanoVDB, CUDA, nanobind, and
  robin_map notices are included.
- `readelf`/`ldd`: dependencies are the system C/C++ runtime, libm, libgcc, and
  ELF loader. No dynamic cudart, OpenVDB, TBB, or toolkit RPATH/RUNPATH.
  CUDA loads the separately installed NVIDIA driver at runtime.
- `auditwheel show` identifies a `manylinux_2_34_x86_64` symbol floor. The wheel
  intentionally retains its `linux_x86_64` tag: no portable-build or wider
  platform validation is implied.

## Reproduction

```sh
python -m venv .venv-build
.venv-build/bin/python -m pip install build 'nanobind>=2.9,<3' \
  'scikit-build-core>=0.11,<0.12' 'cmake>=3.24' auditwheel
.venv-build/bin/python -m build --no-isolation
python -m venv .venv-runtime
.venv-runtime/bin/python -m pip install numpy pytest dist/*.whl
scripts/test_without_toolkit.sh .venv-runtime/bin/python
.venv-build/bin/cmake -S . -B build/native -DFUSE_BUILD_PYTHON=OFF \
  -DFUSE_BUILD_TESTS=ON -DCMAKE_BUILD_TYPE=Release
.venv-build/bin/cmake --build build/native -j2
.venv-build/bin/ctest --test-dir build/native --output-on-failure
.venv-build/bin/auditwheel show dist/*.whl
```

The namespace check requires Linux user/mount namespace support and assumes
all toolkits are under `/usr/local`; adapt it for other hosts. It does not
modify the host's mounts or uninstall any toolkit.

## Limits

Only the RTX 4060/SM 89 path was executed. SM 80/86/90 native code and compute
90 PTX are compiled but untested on their respective hardware. The documented
minimum driver was not tested. No missing-driver machine, Windows build,
additional Python version, public package upload, or geometry reconstruction
was tested. Those remain later milestones; there is no CPU reconstruction
fallback.
