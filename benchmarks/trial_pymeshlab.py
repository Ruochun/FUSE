"""Standalone PyMeshLab trial; does not change FUSE's simplification backend.

Install pymeshlab in a separate environment, then run:
  python benchmarks/trial_pymeshlab.py input.ply output.ply --report report.json
The output still needs independent topology and geometric validation.
Existing output files are replaced after the simplified mesh is saved successfully.
"""

import argparse
from importlib.metadata import version
import json
import math
from pathlib import Path
import resource
import tempfile
import time
import traceback


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path, help="existing output is overwritten on success")
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--ratio", type=float, default=0.5)
    parser.add_argument("--fixed-placement", action="store_true")
    parser.add_argument("--memory-gib", type=float, default=20)
    args = parser.parse_args()
    if not 0 < args.ratio < 1:
        parser.error("ratio must be between zero and one")
    if args.input.resolve() == args.output.resolve():
        parser.error("output must be distinct from the input")
    if args.report.resolve() in (args.input.resolve(), args.output.resolve()):
        parser.error("report must be distinct from both mesh files")
    if not math.isfinite(args.memory_gib) or args.memory_gib <= 0:
        parser.error("memory-gib must be finite and positive")
    suffix = args.output.suffix.lower()
    if suffix not in {".ply", ".obj", ".off", ".stl"}:
        parser.error("output format must be .ply, .obj, .off, or .stl")
    limit = int(args.memory_gib * 1024**3)
    resource.setrlimit(resource.RLIMIT_AS, (limit, limit))
    report = dict(
        input=str(args.input.resolve()),
        output=str(args.output.resolve()),
        backend="pymeshlab_quadric_edge_collapse",
        memory_limit_bytes=limit,
        input_repairs_applied=False,
        stages=[],
    )
    started = time.perf_counter()
    phase = "load"
    try:
        import pymeshlab

        report["pymeshlab_version"] = version("pymeshlab")
        ms = pymeshlab.MeshSet()
        t = time.perf_counter()
        print("Loading mesh", flush=True)
        ms.load_new_mesh(str(args.input))
        report.update(
            input_vertices=ms.current_mesh().vertex_number(),
            input_faces=ms.current_mesh().face_number(),
        )
        report["stages"].append(dict(name=phase, seconds=time.perf_counter() - t))
        params = dict(
            targetfacenum=max(4, int(report["input_faces"] * args.ratio)),
            preservetopology=True,
            preserveboundary=True,
            preservenormal=True,
            optimalplacement=not args.fixed_placement,
            planarquadric=True,
            autoclean=True,
        )
        report["parameters"] = params
        phase = "decimation"
        t = time.perf_counter()
        print(json.dumps(dict(phase=phase, parameters=params)), flush=True)
        ms.meshing_decimation_quadric_edge_collapse(**params)
        report["stages"].append(dict(name=phase, seconds=time.perf_counter() - t))
        report.update(
            output_vertices=ms.current_mesh().vertex_number(),
            output_faces=ms.current_mesh().face_number(),
        )
        phase = "save"
        with tempfile.TemporaryDirectory(
            prefix=".fuse-pymeshlab-", dir=args.output.parent
        ) as directory:
            temporary = Path(directory) / ("mesh" + suffix)
            save_options = {"binary": True} if suffix in {".ply", ".stl"} else {}
            ms.save_current_mesh(str(temporary), **save_options)
            temporary.replace(args.output)
        report.update(status="simplified_unvalidated")
    except Exception as e:
        report.update(status="failed", phase=phase, error=repr(e), traceback=traceback.format_exc())
    finally:
        report.update(
            seconds=time.perf_counter() - started,
            peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
        )
        args.report.write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report, indent=2), flush=True)
    return 0 if report["status"] == "simplified_unvalidated" else 1


if __name__ == "__main__":
    raise SystemExit(main())
