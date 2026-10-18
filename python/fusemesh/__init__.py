"""Fast Union Surface Extraction: sparse CUDA reconstruction (M2)."""
from __future__ import annotations

from dataclasses import dataclass
import operator
from typing import Optional

import numpy as np

from ._native import ResourceError, _reconstruct, build_info, smoke_test

__version__ = "0.0.1.dev1"
__all__ = ["reconstruct", "ReconstructionResult", "ResourceError", "build_info", "smoke_test"]


@dataclass
class ReconstructionResult:
    """Indexed world-space mesh and diagnostic metadata.

    Vertices are float64 (N, 3), faces are uint64 (M, 3). Optional field samples
    are float64 in (z, y, x) order and world-distance units. Output arrays own
    their backing native storage through a capsule and outlive this container.
    """
    vertices: np.ndarray
    faces: np.ndarray
    metadata: dict
    field: Optional[np.ndarray] = None


def reconstruct(meshes, voxel_size, *, device=0, max_memory_bytes=512 * 1024**2,
                validation="full", return_field=False, backend="sparse"):
    """Approximate the full boundary of the union of closed triangle meshes.

    ``meshes`` is an iterable of (vertices, faces) pairs. Vertices must be real
    numeric (N,3) arrays; faces must be integer (M,3) arrays. Inputs are copied
    to contiguous float64/uint64 snapshots before the GIL is released.

    ``validation='full'`` checks closed, consistently oriented manifold
    topology as well as structural/numeric validity. It does not detect
    self-intersections. ``'basic'`` skips the topology check; the same valid-solid
    precondition still applies. No repair, smoothing, or cavity removal occurs.

    ``max_memory_bytes`` bounds FUSE-owned device buffers, excluding the CUDA
    context and host copies/results. Increase voxel_size if ResourceError is
    raised. Subvoxel features may disappear. ``backend='sparse'`` uses NanoVDB
    candidate topology; ``'dense'`` retains the M1 reference implementation.
    ``return_field`` returns a dense diagnostic field, querying absent samples
    explicitly. Its host array must separately fit max_memory_bytes; sparse
    calls use bounded GPU chunks to fill it. Omit it for very large domains.
    Calls are synchronous. Empty input returns empty arrays without using CUDA.
    """
    if backend not in ("sparse", "dense"):
        raise ValueError("backend must be 'sparse' or 'dense'")
    if validation not in ("full", "basic"):
        raise ValueError("validation must be 'full' or 'basic'")
    if isinstance(device, (bool, np.bool_)):
        raise TypeError("device must be an integer")
    device = operator.index(device)
    if isinstance(max_memory_bytes, (bool, np.bool_)):
        raise TypeError("max_memory_bytes must be an integer")
    max_memory_bytes = operator.index(max_memory_bytes)
    if max_memory_bytes <= 0:
        raise ValueError("max_memory_bytes must be positive")
    snapshots = []
    for i, mesh in enumerate(meshes):
        try:
            vertices, faces = mesh
        except (TypeError, ValueError) as error:
            raise ValueError(f"mesh {i}: expected a (vertices, faces) pair") from error
        vertices, faces = np.asarray(vertices), np.asarray(faces)
        if vertices.ndim != 2 or vertices.shape[1] != 3:
            raise ValueError(f"mesh {i}: vertices must have shape (N, 3)")
        if faces.ndim != 2 or faces.shape[1] != 3:
            raise ValueError(f"mesh {i}: faces must have shape (M, 3)")
        if vertices.dtype.kind not in "fiu":
            raise TypeError(f"mesh {i}: vertices must have a real numeric dtype")
        if faces.dtype.kind not in "iu":
            raise TypeError(f"mesh {i}: faces must have an integer dtype")
        if faces.dtype.kind == "i" and np.any(faces < 0):
            raise ValueError(f"mesh {i}: face indices must be nonnegative")
        snapshots.append((np.array(vertices, dtype=np.float64, order="C", copy=True),
                          np.array(faces, dtype=np.uint64, order="C", copy=True)))
    return ReconstructionResult(**_reconstruct(
        snapshots, float(voxel_size), device, max_memory_bytes,
        validation == "full", bool(return_field), backend == "sparse"))
