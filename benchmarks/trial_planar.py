"""Simplify connected nearly planar regions using angle and distance tolerances.

Requires the standalone CGAL helper; see docs/simplification.md.
Distances use input mesh units. Output count is determined by geometry, not a ratio.
The output still needs independent surface-distance and self-intersection checks.
"""

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import resource
import subprocess
import tempfile
import time
import traceback


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument(
        "output",
        type=Path,
        help="PLY, OBJ, OFF, or STL; an existing output is overwritten only on success",
    )
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--angle-tolerance", type=float, default=3.0, help="degrees, default 3")
    parser.add_argument(
        "--plane-tolerance",
        type=float,
        default=0.00025,
        help="mesh units; default 0.00025 (0.25 mm for metre coordinates)",
    )
    parser.add_argument("--memory-gib", type=float, default=20)
    parser.add_argument(
        "--helper",
        type=Path,
        default=Path(
            os.environ.get(
                "FUSE_PLANAR_HELPER",
                str(Path(__file__).resolve().parents[1] / "build/planar-native/fuse_planar"),
            )
        ),
    )
    args = parser.parse_args()
    if not 0 < args.angle_tolerance < 90:
        parser.error("angle-tolerance must be between 0 and 90 degrees")
    if not math.isfinite(args.plane_tolerance) or args.plane_tolerance <= 0:
        parser.error("plane-tolerance must be finite and positive")
    if not math.isfinite(args.memory_gib) or args.memory_gib <= 0:
        parser.error("memory-gib must be finite and positive")
    if args.input.resolve() == args.output.resolve():
        parser.error("output must be distinct from input")
    for mesh_path in (args.input, args.output):
        if args.report.resolve() == mesh_path.resolve() or (
            args.report.exists() and mesh_path.exists() and args.report.samefile(mesh_path)
        ):
            parser.error("report must be distinct from both mesh files")
    output_suffix = args.output.suffix.lower()
    if output_suffix not in {".ply", ".obj", ".off", ".stl"}:
        parser.error("output format must be .ply, .obj, .off, or .stl")
    if not args.helper.is_file():
        parser.error(
            "CGAL helper not found; build benchmarks/planar_native as documented in docs/simplification.md"
        )
    started = time.perf_counter()
    report = dict(
        input=str(args.input.resolve()),
        output=str(args.output.resolve()),
        backend="cgal_almost_planar_patches",
        input_repairs_applied=False,
        parameters=dict(
            angle_tolerance_degrees=args.angle_tolerance, plane_tolerance=args.plane_tolerance
        ),
        helper_sha256=hashlib.sha256(args.helper.read_bytes()).hexdigest(),
        memory_limit_bytes=int(args.memory_gib * 1024**3),
        validation_scope="manifold connectivity, degeneracy, component count, Euler characteristic, closedness; not self-intersections or continuous surface distance",
    )
    phase = "prepare"
    try:
        resource.setrlimit(
            resource.RLIMIT_AS, (report["memory_limit_bytes"], report["memory_limit_bytes"])
        )
        with tempfile.TemporaryDirectory(
            prefix=".fuse-planar-", dir=args.output.parent
        ) as directory:
            temporary = Path(directory) / ("mesh" + output_suffix)
            stats = Path(directory) / "stats.json"
            command = [
                str(args.helper.resolve()),
                str(args.input.resolve()),
                str(temporary),
                str(args.angle_tolerance),
                str(args.plane_tolerance),
                str(stats),
            ]
            phase = "native simplification"
            log = []
            with subprocess.Popen(
                command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
            ) as process:
                for line in process.stdout:
                    print(line, end="", flush=True)
                    log.append(line.rstrip())
                code = process.wait()
            report.update(native_log=log, native_exit_code=code)
            if code:
                raise RuntimeError(
                    f"CGAL helper exited with {code}; previous output was not replaced"
                )
            report["native"] = json.loads(stats.read_text())
            phase = "save"
            temporary.replace(args.output)
            report["status"] = "simplified_topology_checked"
    except Exception as error:
        report.update(
            status="failed", phase=phase, error=str(error), traceback=traceback.format_exc()
        )
    finally:
        report.update(
            seconds=time.perf_counter() - started,
            native_peak_rss_bytes=resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss * 1024,
        )
        args.report.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        print(json.dumps(report, indent=2, allow_nan=False), flush=True)
    return 0 if report["status"] != "failed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
