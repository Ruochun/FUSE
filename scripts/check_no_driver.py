"""Acceptance probe for an installed wheel on a host with no NVIDIA driver."""

import ctypes

import numpy as np
import fusemesh

try:
    ctypes.CDLL("libcuda.so.1")
except OSError:
    pass
else:
    raise RuntimeError("this acceptance probe requires a host without the NVIDIA driver")

assert fusemesh.build_info()["cuda_runtime"] == "static"
assert fusemesh.reconstruct([], 0.1).vertices.shape == (0, 3)
vertices = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=float)
faces = np.array([[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]], dtype=np.uint64)
for call in (fusemesh.smoke_test, lambda: fusemesh.reconstruct([(vertices, faces)], 0.2)):
    try:
        call()
    except RuntimeError as error:
        assert "FUSE CUDA:" in str(error) and "driver" in str(error).lower(), str(error)
    else:
        raise AssertionError("nonempty CUDA work unexpectedly succeeded without a driver")
print("Import, metadata, empty input, and missing-driver diagnostics passed")
