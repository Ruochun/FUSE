"""Compare original and reconstructed surfaces, or demonstrate reconstruction.

CPU-only file comparison (open and nonmanifold surfaces are accepted):
    python examples/quality_demo.py --inputs original.off reconstructed.obj
With no inputs, reconstruct two boxes; requires the quality extra and NVIDIA GPU:
    python examples/quality_demo.py --output-dir build/quality-demo
Each input file must contain one mesh instance. Comparison always runs on CPU.
"""

import argparse
import json
from pathlib import Path

import trimesh
import fusemesh


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--inputs",
        nargs=2,
        type=Path,
        metavar=("ORIGINAL", "RECONSTRUCTED"),
        help="compare two existing surfaces on CPU; default: reconstruct two generated boxes",
    )
    parser.add_argument(
        "--voxel-size",
        type=float,
        help="voxel normalization for file inputs; reconstruction spacing for boxes (default: 0.1)",
    )
    parser.add_argument("--degenerate-faces", choices=("error", "drop"), default="error")
    parser.add_argument("--samples", type=int, default=10000, help="samples per direction")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--tolerance", type=float, default=0.05, help="distance in mesh units")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--output-dir", type=Path, default=Path("build/quality-demo"))
    args = parser.parse_args()

    if args.inputs:
        reference, reconstructed = [fusemesh.load_mesh(path) for path in args.inputs]
        voxel_size = args.voxel_size
        face_count = len(reconstructed[1])
        exports = []
    else:
        first = trimesh.creation.box()
        second = trimesh.creation.box()
        second.apply_translation([0.55, 0.35, 0.2])
        reference = [first, second]
        voxel_size = 0.1 if args.voxel_size is None else args.voxel_size
        reconstructed = fusemesh.reconstruct(reference, voxel_size=voxel_size, device=args.device)
        face_count = len(reconstructed.faces)
        exports = [
            ("original-a.off", first),
            ("original-b.off", second),
            ("reconstructed.off", reconstructed),
        ]

    # A single original surface needs no manifold validation or Boolean union.
    # The generated-box mode instead compares against their solid-union boundary.
    report = fusemesh.compare_surfaces(
        reference,
        reconstructed,
        samples=args.samples,
        seed=args.seed,
        tolerance=args.tolerance,
        voxel_size=voxel_size,
        degenerate_faces=args.degenerate_faces,
    )
    report["demo"] = {
        "mode": "file_comparison" if args.inputs else "generated_boxes",
        "reconstruction_performed": not bool(args.inputs),
    }
    if args.inputs:
        report["demo"]["inputs"] = [str(path.resolve()) for path in args.inputs]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name, mesh in exports:
        fusemesh.save_mesh(args.output_dir / name, mesh)
    report_path = args.output_dir / "quality.json"
    report_path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")

    print(f"Reconstructed triangles before quality filtering: {face_count:,}")
    cleaning = report["cleaning"]
    for i, counts in enumerate(cleaning["originals"]):
        print(f"Original {i}: dropped {counts['removed_faces']} degenerate faces")
    print(f"Reconstructed: dropped {cleaning['reconstructed']['removed_faces']} degenerate faces")
    for name in ("original_to_reconstructed", "reconstructed_to_original", "symmetric"):
        metrics = report[name]
        print(
            f"{name}: RMS={metrics['rms']:.6g}, p95={metrics['p95']:.6g}, "
            f"sampled max={metrics['sampled_max']:.6g}, coverage={metrics['coverage']:.2%}"
        )
    if voxel_size is not None:
        print(f"Symmetric RMS in voxels: {report['symmetric_in_voxels']['rms']:.6g}")
    print(f"F-score within {args.tolerance:g} mesh units: {report['symmetric']['f_score']:.6f}")
    print("Distances are sampling estimates; sampled max is not a certified Hausdorff distance.")
    print(f"JSON report: {report_path.resolve()}")
    if exports:
        print(f"Generated meshes: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
