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
    parser.add_argument(
        "--simplify-error", type=float, help="maximum surface/crease deviation in mesh units"
    )
    parser.add_argument(
        "--target-faces", type=int, help="soft triangle-count goal for simplification"
    )
    parser.add_argument(
        "--simplify-volume-change",
        type=float,
        default=0.01,
        help="maximum relative volume change per shell (default: 0.01)",
    )
    parser.add_argument(
        "--feature-angle",
        type=float,
        default=45.0,
        help="protect creases at or above this dihedral angle in degrees",
    )
    parser.add_argument(
        "--simplify-max-checks",
        type=int,
        default=200000,
        help="maximum distance-verification simplex checks per candidate",
    )
    parser.add_argument("--simplify-on-failure", choices=("raise", "keep"), default="raise")
    parser.add_argument(
        "--unsimplified-output",
        type=Path,
        help="also save the original reconstruction before simplification",
    )
    args = parser.parse_args()
    if args.simplify_error is None and (args.target_faces is not None or args.unsimplified_output):
        parser.error("--target-faces and --unsimplified-output require --simplify-error")
    if args.unsimplified_output and args.unsimplified_output.resolve() == args.output.resolve():
        parser.error("--unsimplified-output must differ from --output")
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
    if args.unsimplified_output:
        args.unsimplified_output.parent.mkdir(parents=True, exist_ok=True)
        fusemesh.save_mesh(args.unsimplified_output, result)
    if args.simplify_error is not None:
        result = fusemesh.simplify(
            result,
            max_error=args.simplify_error,
            target_faces=args.target_faces,
            feature_angle=args.feature_angle,
            max_volume_change=args.simplify_volume_change,
            on_failure=args.simplify_on_failure,
            max_checks=args.simplify_max_checks,
        )
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
