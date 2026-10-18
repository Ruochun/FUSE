# 0001: M0 build and dependency proof

Status: implemented; 18 October 2026.

## Choices

- Pin OpenVDB v13.1.0 at `e26f5a569201d3236e0c30580572101022d3d754`.
  Use NanoVDB headers directly, with no upstream CMake subproject, OpenVDB CPU,
  TBB, examples, viewers, or upstream Python bindings. No upstream patches.
- C++17/CUDA17, CMake 3.24+, nanobind 2.x, scikit-build-core 0.11.x.
  Native static library and an optional Python extension share the same backend.
- Initial validation platform: Ubuntu 22.04 x86-64, GCC 11.4, CUDA 12.8.93,
  Python 3.13.13, RTX 4060 (SM 89). Python metadata permits 3.10+, but only 3.13
  has been exercised for M0. No Windows or manylinux compatibility claim yet.
- Compile native code for SM 80, 86, 89, 90 and PTX compute 90. This intentionally
  excludes pre-Ampere devices. PTX is a forward compatibility path, not a claim
  that future devices have been tested. Override `FUSE_CUDA_ARCHITECTURES` for
  development; the compiled set is recorded by `build_info()`.
- CUDA 12.8 family only for this proof. Statically link `CUDA::cudart_static`.
  Do not ship the driver or use toolkit runtime search paths. The initial
  conservative Linux driver floor is 570.26 (CUDA 12.8 GA's paired driver), rather
  than relying on all CUDA minor-version compatibility cases. The actual test
  host reports driver 595.58.02 / WSL host 595.97.
- Wheel version `0.0.1.dev0`, provisional distribution/import name `fusemesh`.
  No publication or name-availability claim. M0 wheels carry local Linux tags;
  portable manylinux release builds remain M4.
- Include NanoVDB sources in the sdist. No Git or network fetch is required for
  the native dependency when building that archive. Python build requirements
  still need to be available, as do a compiler and CUDA toolkit.

## Diagnostic implementation

`smoke_test(device=0)` builds a fixed float NanoVDB grid on the CPU, transfers
its serialized buffer to the selected GPU, reads active and background values
using a NanoVDB accessor in a CUDA kernel, and returns an owning NumPy float32
array. Coordinates cross leaf boundaries and include a negative coordinate.

This is a GPU grid-access and packaging proof, not a GPU topology benchmark or
reconstruction implementation. Direct checked CUDA allocation and transfer are
used because upstream CUDA utilities can terminate the process on errors.
FUSE reports runtime failures as exceptions, uses per-call resources, releases
the GIL during native work, and restores the calling thread's current device.

Alternatives: a CPU-only import test cannot prove CUDA execution; a full
MeshToGrid proof would pull topology correctness into M0. Shared cudart would
require runtime packaging/search-path handling. Static cudart is sufficient for
this small diagnostic and simplifies the clean-install dependency check.

## Redistribution

FUSE retains its existing BSD-3-Clause license. NanoVDB at the pinned release is
Apache-2.0. CUDA's Attachment A lists `libcudart_static.a` as redistributable;
the full CUDA terms and binding dependency licenses are shipped in wheel
metadata. A public-release audit of application distribution terms and final
components is still required in M4; this local diagnostic is not published.

Sources: [pinned OpenVDB](https://github.com/AcademySoftwareFoundation/openvdb/tree/e26f5a569201d3236e0c30580572101022d3d754),
[CUDA 12.8 release notes](https://docs.nvidia.com/cuda/archive/12.8.0/cuda-toolkit-release-notes/index.html),
[CUDA 12.8.1 license](https://docs.nvidia.com/cuda/archive/12.8.1/eula/index.html).

Validation results are recorded in [M0 validation](../m0-validation.md).
