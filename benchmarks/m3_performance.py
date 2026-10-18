"""Repeatable performance suite; optional local-file workload manifest.

Run: python benchmarks/m3_performance.py --output build/m3.json
Check: python benchmarks/m3_performance.py --baseline docs/benchmarks/m3-rtx4060.json --check
Manifest: [{"name":"part", "paths":["part.off"], "voxel_size":0.1}]
Paths in manifests are relative to the manifest; loading is timed separately.
"""

import argparse
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import sys
import time

import fusemesh
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tests"))
from geometry_helpers import box, sphere, torus


def workloads():
    return [
        ("small_cube", [box()], 0.04),
        ("large_cube", [box((0, 0, 0), (4, 4, 4))], 0.05),
        (
            "overlap_spheres",
            [
                sphere((x, y, 0), 0.7, 3)
                for x, y in [(-0.35, -0.35), (0.35, -0.35), (-0.35, 0.35), (0.35, 0.35)]
            ],
            0.06,
        ),
        (
            "many_solids",
            [
                box((x * 2, y * 2, z * 2), (x * 2 + 0.5, y * 2 + 0.5, z * 2 + 0.5))
                for x in range(4)
                for y in range(4)
                for z in range(4)
            ],
            0.1,
        ),
        ("detailed_shell", [sphere(levels=5)], 0.08),
        ("torus", [torus()], 0.06),
        ("containment", [box((0, 0, 0), (16, 16, 16)), box((7, 7, 7), (9, 9, 9))], 0.25),
        ("distant_cubes", [box(), box((4096, 4096, 4096), (4097, 4097, 4097))], 0.25),
    ]


def check_report(report, baseline):
    if report["gpu"] != baseline["gpu"]:
        raise ValueError(
            "benchmark GPU/driver differs from baseline; record a platform-specific baseline"
        )
    old = {w["name"]: w for w in baseline["workloads"]}
    failures = []
    for w in report["workloads"]:
        b = old.get(w["name"])
        if b is None:
            failures.append(w["name"] + ": missing baseline")
            continue
        if w["input_sha256"] != b["input_sha256"] or w["options"] != b["options"]:
            failures.append(w["name"] + ": workload changed")
            continue
        # Initial M3 guardrails tolerate ordinary WSL/display scheduling jitter.
        if w["median_ms"] > b["median_ms"] * 1.5 + 2:
            failures.append(w["name"] + ": warm median exceeds 1.5x baseline + 2 ms")
        if w["peak_device_bytes"] > b["peak_device_bytes"] * 1.05 + 4096:
            failures.append(w["name"] + ": device memory exceeds 1.05x baseline + 4096 bytes")
        if w["vertices"] != b["vertices"] or w["triangles"] != b["triangles"]:
            failures.append(w["name"] + ": output counts changed")
    if set(old) != {w["name"] for w in report["workloads"]}:
        failures.append("workload sets differ")
    return failures


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path)
    p.add_argument("--manifest", type=Path)
    p.add_argument("--baseline", type=Path)
    p.add_argument("--check", action="store_true")
    p.add_argument("--repeats", type=int, default=7)
    a = p.parse_args()
    if a.repeats < 3:
        p.error("--repeats must be at least 3")
    if a.check and not a.baseline:
        p.error("--check requires --baseline")
    report = dict(
        version=fusemesh.__version__,
        build=fusemesh.build_info(),
        python=platform.python_version(),
        platform=platform.platform(),
        workloads=[],
    )
    report["gpu"] = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"],
        text=True,
    ).strip()
    cases = [
        dict(
            name=n,
            meshes=m,
            options=dict(
                voxel_size=h, backend="sparse", validation="full", max_memory_bytes=512 * 1024**2
            ),
            loading_ms=0,
        )
        for n, m, h in workloads()
    ]
    if a.manifest:
        for row in json.loads(a.manifest.read_text()):
            start = time.perf_counter()
            meshes = fusemesh.load_meshes([a.manifest.parent / Path(f) for f in row["paths"]])
            cases.append(
                dict(
                    name=row["name"],
                    meshes=meshes,
                    loading_ms=(time.perf_counter() - start) * 1000,
                    options=dict(
                        voxel_size=row["voxel_size"],
                        backend=row.get("backend", "sparse"),
                        validation=row.get("validation", "full"),
                        max_memory_bytes=row.get("max_memory_bytes", 512 * 1024**2),
                    ),
                )
            )
    if len({c["name"] for c in cases}) != len(cases):
        p.error("workload names must be unique")
    # Context startup is excluded from warm performance; preserve its own timing.
    start = time.perf_counter()
    fusemesh.smoke_test()
    report["startup_ms"] = (time.perf_counter() - start) * 1000
    for case in cases:
        meshes = case["meshes"]
        digest = hashlib.sha256()
        for v, f in meshes:
            for x, dtype in ((v, "<f8"), (f, "<u8")):
                arr = np.asarray(x, dtype=dtype)
                digest.update(str(arr.shape).encode())
                digest.update(arr.tobytes())
        samples = []
        for _ in range(a.repeats + 1):
            start = time.perf_counter()
            result = fusemesh.reconstruct(meshes, **case["options"])
            samples.append(dict(total_ms=(time.perf_counter() - start) * 1000, **result.metadata))
        report["workloads"].append(
            dict(
                name=case["name"],
                options=case["options"],
                loading_ms=case["loading_ms"],
                input_sha256=digest.hexdigest(),
                input_triangles=sum(len(f) for _, f in meshes),
                input_shapes=len(meshes),
                vertices=len(result.vertices),
                triangles=len(result.faces),
                peak_device_bytes=max(s["peak_device_bytes"] for s in samples),
                median_ms=float(np.median([s["total_ms"] for s in samples[1:]])),
                first=samples[0],
                runs=samples[1:],
            )
        )
    if a.check:
        report["regressions"] = check_report(report, json.loads(a.baseline.read_text()))
    output = json.dumps(report, indent=2) + "\n"
    if a.output:
        a.output.parent.mkdir(parents=True, exist_ok=True)
        a.output.write_text(output)
    else:
        print(output, end="")
    if report.get("regressions"):
        print("\n".join(report["regressions"]), file=sys.stderr)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
