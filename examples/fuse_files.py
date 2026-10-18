"""Load local meshes with Trimesh, fuse with CUDA, and export the union surface.

Example: python examples/fuse_files.py a.off b.off --voxel-size 0.05 --output union.obj
"""

import argparse
import json
from pathlib import Path
import time

import fusemesh


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--voxel-size", type=float, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--backend", choices=("sparse", "dense"), default="sparse")
    parser.add_argument(
        "--validation",
        choices=("full", "basic"),
        default="full",
        help="full checks manifold topology; basic skips topology checks without repairing the mesh",
    )
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--memory-mib", type=int, default=16 * 1024)
    args = parser.parse_args()
    start = time.perf_counter()
    meshes = fusemesh.load_meshes(args.inputs)
    loading_ms = (time.perf_counter() - start) * 1000
    start = time.perf_counter()
    result = fusemesh.reconstruct(
        meshes,
        args.voxel_size,
        device=args.device,
        backend=args.backend,
        validation=args.validation,
        max_memory_bytes=args.memory_mib * 1024**2,
    )
    reconstruction_ms = (time.perf_counter() - start) * 1000
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fusemesh.save_mesh(args.output, result)
    print(
        json.dumps(
            dict(
                input_meshes=len(meshes),
                input_triangles=sum(len(f) for _, f in meshes),
                loading_ms=loading_ms,
                reconstruction_ms=reconstruction_ms,
                output=str(args.output.resolve()),
                **result.metadata,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
