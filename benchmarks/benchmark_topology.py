"""Measure CPU topology validation in a fresh process, including loading peak RSS.

Run: python benchmarks/benchmark_topology.py model.ply
"""

import argparse
import json
import resource
import time

import numpy as np

import fusemesh
from fusemesh.simplification import _inspect


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mesh")
    args = parser.parse_args()
    vertices, faces = fusemesh.load_mesh(args.mesh)
    # Match simplify's normalized area threshold without copying face geometry.
    if len(faces):
        used = vertices[np.unique(faces)]
        lo, hi = used.min(axis=0), used.max(axis=0)
        del used
        scale = (hi - lo).max()
        if not np.isfinite(scale) or scale <= 0:
            raise ValueError("mesh must have finite positive extent")
        vertices = (vertices - (lo + (hi - lo) / 2)) / scale
    start = time.perf_counter()
    parts, _, _ = _inspect(vertices, faces, components_only=True)
    print(
        json.dumps(
            dict(
                input=args.mesh,
                vertices=len(vertices),
                faces=len(faces),
                components=len(parts),
                component_faces=[len(p) for p in parts],
                inspect_seconds=time.perf_counter() - start,
                peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
