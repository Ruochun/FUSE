"""Reproducible reduction/quality/time/RSS measurements; one process per case.

Run: python benchmarks/benchmark_simplify.py [--gpu]
RSS includes the Python process, loaded libraries, and (for fused cases) CUDA.
Peak growth is a high-water-mark difference, not an allocation budget.
"""

import argparse
import json
import resource
import subprocess
import sys
import time

import numpy as np
import trimesh
import fusemesh


def fixture(name):
    if name == "sphere":
        return trimesh.creation.icosphere(subdivisions=3), 320, 0.04
    if name == "thin":
        return trimesh.creation.box(extents=(1, 1, 0.02)).subdivide().subdivide(), 12, 0.005
    box = trimesh.creation.box()
    if name in ("fused_box", "fused_union"):
        meshes = [box]
        if name == "fused_union":
            other = box.copy()
            other.vertices += [0.5, 0.25, 0]
            meshes.append(other)
        return fusemesh.reconstruct(meshes, 0.12), 100, 0.03
    mesh = box.subdivide().subdivide().subdivide()
    if name == "cavity":
        inner = mesh.copy()
        inner.vertices *= 0.5
        inner.faces = inner.faces[:, ::-1]
        mesh = trimesh.util.concatenate([mesh, inner])
        return mesh, 24, 0.01
    return mesh, 12, 0.01


def run(name):
    mesh, count, tolerance = fixture(name)
    before = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    started = time.perf_counter()
    result = fusemesh.simplify(mesh, max_error=tolerance, target_faces=count)
    elapsed = time.perf_counter() - started
    after = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    report = result.metadata["simplification"]
    return dict(
        case=name,
        input_faces=report["input_faces"],
        output_faces=len(result.faces),
        reduction=1 - len(result.faces) / report["input_faces"],
        seconds=elapsed,
        max_error=tolerance,
        surface_error_bound=report["surface_error_bound"],
        feature_error_bound=report["feature_error_bound"],
        max_relative_volume_change=max(s["relative_volume_change"] for s in report["shells"]),
        peak_rss_mib=after / 1024,
        peak_rss_growth_mib=(after - before) / 1024,
        python=sys.version.split()[0],
        numpy=np.__version__,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", action="store_true")
    parser.add_argument(
        "--case", choices=("box", "sphere", "thin", "cavity", "fused_box", "fused_union")
    )
    args = parser.parse_args()
    if args.case:
        print(json.dumps(run(args.case)))
        return
    cases = ["box", "sphere", "thin", "cavity"]
    if args.gpu:
        cases += ["fused_box", "fused_union"]
    results = []
    for case in cases:
        completed = subprocess.run(
            [sys.executable, __file__, "--case", case], capture_output=True, text=True, check=True
        )
        results.append(json.loads(completed.stdout))
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
