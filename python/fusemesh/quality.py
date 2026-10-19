"""CPU, area-sampled bidirectional point-to-triangle surface distances."""

from __future__ import annotations

from importlib.metadata import version
import argparse
import json
from pathlib import Path

import numpy as np

from .io import _is_source, load_meshes
from .simplification import _Surface, _inspect, _integer, _positive, _volume


def _pair(value):
    if not isinstance(value, (tuple, list)) or len(value) != 2:
        return False
    try:
        return all(np.asarray(x).ndim == 2 and np.asarray(x).shape[1] == 3 for x in value)
    except (ValueError, TypeError):
        return False


def _meshes(value):
    if _is_source(value):
        return load_meshes(value)
    if hasattr(value, "vertices") and hasattr(value, "faces"):
        return [(value.vertices, value.faces)]
    if _pair(value):
        return [value]
    result = []
    for item in value:
        result.extend(_meshes(item))
    return result


def _validate(pair):
    v, f = map(np.asarray, pair)
    if v.ndim != 2 or v.shape[1] != 3 or v.dtype.kind not in "fiu":
        raise ValueError("vertices must be a real numeric (N, 3) array")
    if f.ndim != 2 or f.shape[1] != 3 or f.dtype.kind not in "iu":
        raise ValueError("faces must be an integer (M, 3) array")
    if not len(f) or not np.isfinite(v).all():
        raise ValueError("surfaces must be nonempty with finite vertices")
    if np.any(f < 0) or np.any(f >= len(v)):
        raise ValueError("face index is outside the vertex array")
    return np.array(v, dtype=np.float64), np.array(f, dtype=np.int64)


def _clean_surface(mesh, policy, label):
    """Drop only numerically zero-area faces; do not weld, repair, or thin-filter."""
    v, f = mesh
    triangles = v[f]
    edges = triangles[:, 1:] - triangles[:, :1]
    if not np.isfinite(edges).all():
        raise ValueError(f"{label}: triangle extent exceeds floating-point range")
    # Per-triangle scaling avoids squaring tiny areas or overflowing cross products.
    scales = np.max(np.abs(edges), axis=(1, 2))
    scaled = edges / np.where(scales > 0, scales, 1)[:, None, None]
    cross = np.cross(scaled[:, 0], scaled[:, 1])
    degenerate = np.all(cross == 0, axis=1)
    removed = int(np.count_nonzero(degenerate))
    if removed and policy == "error":
        raise ValueError(
            f"{label}: {removed} degenerate triangles; use degenerate_faces='drop' to ignore them"
        )
    kept = f[~degenerate]
    if not len(kept):
        raise ValueError(f"{label}: no valid triangles remain after dropping degenerate faces")
    # Discard unused vertices in the temporary snapshot so removed line/point
    # features cannot influence normalization of the retained surface.
    used, inverse = np.unique(kept, return_inverse=True)
    cleaned = (v[used], inverse.reshape(-1, 3))
    counts = {"input_faces": len(f), "removed_faces": removed, "retained_faces": len(kept)}
    return cleaned, counts


def _areas(triangles):
    areas = (
        np.hypot.reduce(
            np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]), axis=1
        )
        / 2
    )
    if not np.isfinite(areas).all() or np.any(areas <= 0):
        raise ValueError("surfaces must not contain degenerate triangles")
    return areas


def _union(meshes):
    try:
        import manifold3d as m
    except ImportError as error:
        raise ImportError('multiple originals require pip install "fusemesh[quality]"') from error
    solids = []
    for v, f in meshes:
        _inspect(v, f)
        if _volume(v[f]) <= 0:
            raise ValueError("each original solid must have positive signed volume")
        solid = m.Manifold(m.Mesh64(np.ascontiguousarray(v), np.asarray(f, dtype=np.uint64)))
        if solid.status() != m.Error.NoError:
            raise ValueError(f"invalid original solid: {solid.status()}")
        solids.append(solid)
    union = m.Manifold.batch_boolean(solids, m.OpType.Add)
    if union.status() != m.Error.NoError or union.is_empty():
        raise ValueError(f"reference union failed: {union.status()}")
    mesh = union.to_mesh64()
    return np.asarray(mesh.vert_properties)[:, :3], np.asarray(mesh.tri_verts, dtype=np.int64)


def _distances(source, target, count, rng, batch_size):
    areas = _areas(source)
    # Area-proportional triangle choice followed by uniform barycentric sampling.
    indices = rng.choice(len(source), count, p=areas / areas.sum())
    uv = rng.random((count, 2))
    root = np.sqrt(uv[:, 0])
    weights = np.column_stack((1 - root, root * (1 - uv[:, 1]), root * uv[:, 1]))
    distances = np.empty(count)
    surface = _Surface(target)
    try:
        for start in range(0, count, batch_size):
            stop = min(count, start + batch_size)
            points = np.einsum("ij,ijk->ik", weights[start:stop], source[indices[start:stop]])
            distances[start:stop] = surface.nearest(points)[0]
    finally:
        surface.close()
    return distances


def _stats(distances, tolerance):
    peak = float(np.max(distances))
    return {
        "rms": float(peak * np.sqrt(np.mean((distances / peak) ** 2))) if peak else 0.0,
        "mean": float(peak * np.mean(distances / peak)) if peak else 0.0,
        "p95": float(np.percentile(distances, 95)),
        "sampled_max": float(np.max(distances)),
        "coverage": None if tolerance is None else float(np.mean(distances <= tolerance)),
    }


def compare_surfaces(
    originals,
    reconstructed,
    *,
    samples=10000,
    seed=0,
    tolerance=None,
    voxel_size=None,
    batch_size=256,
    degenerate_faces="error",
):
    """Return a JSON-compatible reconstruction quality report (CPU only).

    Inputs accept paths, Trimesh/Scene objects, array pairs, or ReconstructionResult.
    Originals may be an iterable of these; multiple instances are Boolean-unioned
    using the optional ``quality`` dependency. Reconstructed must contain exactly
    one mesh instance (which may have disconnected components).

    Each direction uses ``samples`` independent area-uniform source points and
    exact nearest-triangle queries in floating point. RMS and percentiles are
    sampling estimates; sampled_max is NOT a certified Hausdorff distance.
    Distances use input units. Coverage is the fraction within tolerance.
    Combined statistics pool equal numbers from both directions, giving each
    surface equal weight regardless of area. A seed makes sampling repeatable.
    Single surfaces need not be closed/manifold. Multiple originals must be valid
    oriented closed solids; self-intersections and shell nesting are not checked.
    ``degenerate_faces='drop'`` removes zero-area/repeated-index triangles from
    temporary copies on both sides and reports counts in ``cleaning``. Default
    ``'error'`` rejects them. No near-zero area threshold or topology repair is
    applied. Each input mesh must retain at least one valid triangle.
    """
    if degenerate_faces not in ("error", "drop"):
        raise ValueError("degenerate_faces must be 'error' or 'drop'")
    samples = _integer(samples, "samples", 1)
    seed = _integer(seed, "seed", 0)
    batch_size = _integer(batch_size, "batch_size", 1)
    if tolerance is not None:
        tolerance = _positive(tolerance, "tolerance")
    if voxel_size is not None:
        voxel_size = _positive(voxel_size, "voxel_size")
    originals = [_validate(p) for p in _meshes(originals)]
    output = [_validate(p) for p in _meshes(reconstructed)]
    if not originals or len(output) != 1:
        raise ValueError("expected at least one original and exactly one reconstructed mesh")
    cleaned = [
        _clean_surface(mesh, degenerate_faces, f"original mesh {i}")
        for i, mesh in enumerate(originals)
    ]
    cleaned_output, output_counts = _clean_surface(
        output[0], degenerate_faces, "reconstructed mesh"
    )
    cleaning = {
        "degenerate_faces": degenerate_faces,
        "originals": [counts for _, counts in cleaned],
        "reconstructed": output_counts,
    }
    # Common normalization improves distance and Boolean accuracy under translations.
    all_meshes = [mesh for mesh, _ in cleaned] + [cleaned_output]
    lo = np.min([v.min(axis=0) for v, _ in all_meshes], axis=0)
    hi = np.max([v.max(axis=0) for v, _ in all_meshes], axis=0)
    scale = float(np.max(hi - lo))
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("mesh extent must be finite and positive")
    normalized = [((v - lo) / scale, f) for v, f in all_meshes]
    for v, f in normalized:
        _areas(v[f])
    reference = _union(normalized[:-1]) if len(originals) > 1 else normalized[0]
    ref_triangles = reference[0][reference[1]]
    out_triangles = normalized[-1][0][normalized[-1][1]]
    _areas(ref_triangles)
    streams = np.random.SeedSequence(seed).spawn(2)
    forward = (
        _distances(
            ref_triangles, out_triangles, samples, np.random.default_rng(streams[0]), batch_size
        )
        * scale
    )
    backward = (
        _distances(
            out_triangles, ref_triangles, samples, np.random.default_rng(streams[1]), batch_size
        )
        * scale
    )
    combined = _stats(np.concatenate((forward, backward)), tolerance)
    a, b = _stats(forward, tolerance), _stats(backward, tolerance)
    if tolerance is not None:
        precision, recall = b["coverage"], a["coverage"]
        combined["f_score"] = (
            2 * precision * recall / (precision + recall) if precision + recall else 0.0
        )
    report = {
        "original_to_reconstructed": a,
        "reconstructed_to_original": b,
        "symmetric": combined,
        "sampling": {
            "samples_per_direction": samples,
            "seed": seed,
            "method": "area_uniform_point_to_triangle",
            "hausdorff_certified": False,
        },
        "reference": {
            "mode": "solid_union" if len(originals) > 1 else "surface",
            "input_meshes": len(originals),
            "faces": len(ref_triangles),
            "union_backend": "manifold3d" if len(originals) > 1 else None,
        },
        "tolerance": tolerance,
        "cleaning": cleaning,
        "versions": {name: version(name) for name in ("numpy", "trimesh", "rtree")},
    }
    if len(originals) > 1:
        report["versions"]["manifold3d"] = version("manifold3d")
    if voxel_size is not None:
        report["voxel_size"] = voxel_size
        report["symmetric_in_voxels"] = {
            key: combined[key] / voxel_size for key in ("rms", "mean", "p95", "sampled_max")
        }
    return report


def main():
    """Command-line JSON report, also available as python -m fusemesh.quality."""
    parser = argparse.ArgumentParser(description=compare_surfaces.__doc__.split("\n")[0])
    parser.add_argument("originals", nargs="+", help="original mesh files (solid union)")
    parser.add_argument("--reconstructed", required=True, help="reconstructed mesh file")
    parser.add_argument("--samples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--tolerance", type=float)
    parser.add_argument("--voxel-size", type=float)
    parser.add_argument("--degenerate-faces", choices=("error", "drop"), default="error")
    parser.add_argument("--output", type=Path, help="JSON output file; default stdout")
    args = parser.parse_args()
    try:
        report = compare_surfaces(
            args.originals,
            args.reconstructed,
            samples=args.samples,
            seed=args.seed,
            tolerance=args.tolerance,
            voxel_size=args.voxel_size,
            degenerate_faces=args.degenerate_faces,
        )
        text = json.dumps(report, indent=2, allow_nan=False) + "\n"
        if args.output:
            args.output.write_text(text, encoding="utf-8")
        else:
            print(text, end="")
    except (ValueError, TypeError, ImportError, OSError) as error:
        parser.exit(2, f"surface comparison failed: {error}\n")


if __name__ == "__main__":
    main()
