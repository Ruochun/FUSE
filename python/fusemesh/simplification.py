"""CPU surface simplification with independent, conservative acceptance checks."""

from __future__ import annotations

from copy import deepcopy
from importlib.metadata import version
import math
import operator
import threading
import time

import numpy as np
import trimesh

# The upstream extension owns process-global scratch buffers.
_BACKEND_LOCK = threading.Lock()


class SimplificationError(RuntimeError):
    """No reduced candidate passed the requested quality checks."""


def _positive(value, name):
    if isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{name} must be a real number, not bool")
    value = float(value)
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be finite and positive")
    return value


def _integer(value, name, minimum):
    if isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{name} must be an integer, not bool")
    value = operator.index(value)
    if value < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return value


def _inspect(vertices, faces):
    """Check closed oriented vertex/edge manifolds and find connected shells."""
    if len(faces) == 0:
        return [], np.empty((0, 2), dtype=np.int64), np.empty((0, 2), dtype=np.int64)
    triangles = vertices[faces]
    areas = np.linalg.norm(
        np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]), axis=1
    )
    if not np.isfinite(areas).all() or np.any(areas <= 1e-28):
        raise ValueError("mesh has degenerate triangles")
    if len(np.unique(np.sort(faces, axis=1), axis=0)) != len(faces):
        raise ValueError("mesh has duplicate triangles")
    edges = {}
    links = {}
    for fi, face in enumerate(faces):
        a, b, c = map(int, face)
        for u, v, opposite in ((a, b, c), (b, c, a), (c, a, b)):
            edges.setdefault((min(u, v), max(u, v)), []).append((fi, u < v))
            links.setdefault(u, {}).setdefault(v, []).append(opposite)
            links[u].setdefault(opposite, []).append(v)
    if any(len(pair) != 2 for pair in edges.values()):
        raise ValueError("mesh must be closed and edge-manifold")
    if any(pair[0][1] == pair[1][1] for pair in edges.values()):
        raise ValueError("mesh winding must be consistent")
    for link in links.values():
        seen, stack = set(), [next(iter(link))]
        while stack:
            u = stack.pop()
            if u not in seen:
                seen.add(u)
                stack.extend(v for v in link[u] if v not in seen)
        if len(seen) != len(link) or any(len(v) != 2 for v in link.values()):
            raise ValueError("mesh must be vertex-manifold")
    adjacent = np.array([[p[0][0], p[1][0]] for p in edges.values()], dtype=np.int64)
    neighbors = [[] for _ in faces]
    for a, b in adjacent:
        neighbors[a].append(b)
        neighbors[b].append(a)
    seen = np.zeros(len(faces), dtype=bool)
    components = []
    for start in range(len(faces)):
        if seen[start]:
            continue
        stack, component = [start], []
        seen[start] = True
        while stack:
            face = stack.pop()
            component.append(face)
            for other in neighbors[face]:
                if not seen[other]:
                    seen[other] = True
                    stack.append(other)
        components.append(np.asarray(sorted(component), dtype=np.int64))
    return components, np.asarray(list(edges), dtype=np.int64), adjacent


def _volume(triangles):
    # Translation reduces cancellation for small shells far from the origin.
    t = triangles - triangles[0, 0]
    return float(np.einsum("ij,ij->i", t[:, 0], np.cross(t[:, 1], t[:, 2])).sum() / 6)


def _sharp_edges(vertices, faces, edges, adjacent, angle):
    triangles = vertices[faces]
    normals = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    normals /= np.linalg.norm(normals, axis=1)[:, None]
    dots = np.einsum("ij,ij->i", normals[adjacent[:, 0]], normals[adjacent[:, 1]])
    return vertices[edges[dots <= math.cos(math.radians(angle)) + 1e-10]]


def _closest(primitives, points):
    if primitives.shape[1] == 3:
        return trimesh.triangles.closest_point(primitives, points)
    start, end = primitives[:, 0], primitives[:, 1]
    edge = end - start
    t = np.einsum("ij,ij->i", points - start, edge) / np.einsum("ij,ij->i", edge, edge)
    return start + np.clip(t, 0, 1)[:, None] * edge


class _Surface:
    """Nearest triangle/segment queries without SciPy or a dense distance matrix."""

    def __init__(self, primitives):
        from rtree import index

        self.primitives = primitives
        bounds = np.concatenate((primitives.min(axis=1), primitives.max(axis=1)), axis=1)
        self.tree = index.Index(
            ((i, tuple(b), None) for i, b in enumerate(bounds)),
            properties=index.Property(dimension=3),
        )

    def close(self):
        self.tree.close()

    def nearest(self, points):
        # Nearest AABB supplies an initial upper bound; all closer primitives
        # must intersect the resulting cube. Refine using actual geometry.
        initial = np.array(
            [next(self.tree.nearest(tuple(p) + tuple(p), 1)) for p in points], dtype=np.int64
        )
        close = _closest(self.primitives[initial], points)
        radius = np.linalg.norm(close - points, axis=1) + 1e-14
        candidates = [
            list(self.tree.intersection(tuple(p - r) + tuple(p + r)))
            for p, r in zip(points, radius)
        ]
        counts = np.array([len(c) for c in candidates])
        ids = np.concatenate(candidates)
        query = np.repeat(points, counts, axis=0)
        distance = np.linalg.norm(_closest(self.primitives[ids], query) - query, axis=1)
        ends = np.cumsum(counts)
        starts = np.r_[0, ends[:-1]]
        best = np.array([a + np.argmin(distance[a:b]) for a, b in zip(starts, ends)])
        return distance[best], ids[best]


def _directed_bound(source, target, tolerance, budget):
    """Bound every point using convexity and the 1-Lipschitz distance function.

    For a source simplex, distance to ONE target simplex is convex, hence its
    maximum is bounded by the maximum vertex distance. A second bound is
    distance(center, target) + max vertex radius. Subdivide ambiguous simplices.
    These are floating-point bounds, not exact-arithmetic certificates.
    """
    upper, sampled = 0.0, 0.0
    # Keep the work queue bounded: chunks plus depth-first subdivision.
    stack = [source[i : i + 256] for i in range(0, len(source), 256)]
    while stack:
        batch = stack.pop()
        budget[0] -= len(batch)
        if budget[0] < 0:
            raise SimplificationError("distance verification budget exhausted")
        center = batch.mean(axis=1)
        distance, ids = target.nearest(center)
        sampled = max(sampled, float(distance.max()))
        if np.any(distance > tolerance):
            raise SimplificationError("surface or sharp-edge deviation exceeds max_error")
        size = batch.shape[1]
        candidate = np.repeat(target.primitives[ids], size, axis=0)
        vertex_distance = np.linalg.norm(
            _closest(candidate, batch.reshape(-1, 3)) - batch.reshape(-1, 3), axis=1
        ).reshape(-1, size)
        radius = np.linalg.norm(batch - center[:, None], axis=2).max(axis=1)
        bound = np.minimum(vertex_distance.max(axis=1), distance + radius) + 1e-13
        accepted = bound <= tolerance
        if np.any(accepted):
            upper = max(upper, float(bound[accepted].max()))
        remaining = batch[~accepted]
        if not len(remaining):
            continue
        if size == 2:
            mid = remaining.mean(axis=1)
            children = np.concatenate(
                (np.stack((remaining[:, 0], mid), axis=1), np.stack((mid, remaining[:, 1]), axis=1))
            )
        else:
            # Longest-edge bisection avoids fourfold work on skinny triangles.
            lengths = np.linalg.norm(remaining - np.roll(remaining, -1, axis=1), axis=2)
            edge = lengths.argmax(axis=1)
            row = np.arange(len(remaining))
            a, b, c = (remaining[row, (edge + j) % 3] for j in range(3))
            mid = (a + b) / 2
            children = np.concatenate(
                (np.stack((a, mid, c), axis=1), np.stack((mid, b, c), axis=1))
            )
        stack.extend(children[i : i + 256] for i in range(0, len(children), 256))
    return upper, sampled


def _symmetric_bound(a, b, tolerance, budget):
    if not len(a) and not len(b):
        return 0.0, 0.0
    if not len(a) or not len(b):
        raise SimplificationError("sharp features were removed or introduced")
    target_a, target_b = _Surface(a), _Surface(b)
    try:
        ab = _directed_bound(a, target_b, tolerance, budget)
        ba = _directed_bound(b, target_a, tolerance, budget)
        return max(ab[0], ba[0]), max(ab[1], ba[1])
    finally:
        target_a.close()
        target_b.close()


def _decimate(vertices, faces, count):
    # Use the array interface: Trimesh's convenience wrapper processes its
    # output by default, which would silently weld/repair candidate geometry.
    import fast_simplification

    with _BACKEND_LOCK:
        v, f = fast_simplification.simplify(
            np.array(vertices, copy=True),
            np.array(faces, dtype=np.int32, copy=True),
            target_count=count,
            agg=5.0,
        )
    return np.asarray(v, dtype=np.float64), np.asarray(f, dtype=np.int64)


def simplify(
    mesh,
    *,
    max_error,
    target_faces=None,
    feature_angle=45.0,
    max_volume_change=0.01,
    max_checks=200000,
    on_failure="raise",
):
    """Reduce a closed triangle mesh on the CPU without changing the input.

    Accepts a ReconstructionResult, Trimesh, or (vertices, faces) pair. Error
    is relative to this mesh, in world units, NOT to the pre-fusion inputs.
    Both directed surface distances are conservatively bounded by adaptive
    subdivision in float64. ``max_checks`` bounds simplex checks per candidate;
    an unverified candidate is never returned as a successful simplification.

    Edges with dihedral angle at least ``feature_angle`` degrees are
    protected by the same symmetric distance bound. ``max_volume_change`` is
    the maximum relative signed-volume change per connected shell. Components,
    Euler characteristic, vertex/edge manifoldness and winding are checked.
    Self-intersections and nesting changes are NOT detected; this is not an
    ambient-isotopy or exact-arithmetic guarantee.

    ``target_faces`` is a soft goal (default: half the input faces). Up to six
    progressively less aggressive candidates are tried per shell. On failure,
    raise SimplificationError; ``on_failure='keep'`` retains the original shell
    and reports it. Metadata records bounds, counts, attempts and timings.
    Original field samples, if any, are copied unchanged.
    """
    from . import ReconstructionResult

    started = time.perf_counter()
    max_error = _positive(max_error, "max_error")
    max_volume_change = _positive(max_volume_change, "max_volume_change")
    feature_angle = _positive(feature_angle, "feature_angle")
    if feature_angle >= 180:
        raise ValueError("feature_angle must be less than 180 degrees")
    max_checks = _integer(max_checks, "max_checks", 1)
    if target_faces is not None:
        target_faces = _integer(target_faces, "target_faces", 4)
    if on_failure not in ("raise", "keep"):
        raise ValueError("on_failure must be 'raise' or 'keep'")
    if isinstance(mesh, (ReconstructionResult, trimesh.Trimesh)):
        vertices, faces = mesh.vertices, mesh.faces
    else:
        try:
            vertices, faces = mesh
        except (TypeError, ValueError) as error:
            raise TypeError("mesh must be a result, Trimesh or (vertices, faces) pair") from error
    vertices, faces = np.asarray(vertices), np.asarray(faces)
    if vertices.ndim != 2 or vertices.shape[1] != 3 or vertices.dtype.kind not in "fiu":
        raise ValueError("vertices must be a real numeric (N, 3) array")
    if faces.ndim != 2 or faces.shape[1] != 3 or faces.dtype.kind not in "iu":
        raise ValueError("faces must be an integer (M, 3) array")
    if not np.isfinite(vertices).all():
        raise ValueError("vertices must be finite")
    if faces.size and (np.any(faces < 0) or np.any(faces >= len(vertices))):
        raise ValueError("face index is outside the vertex array")
    if len(vertices) > np.iinfo(np.int32).max or len(faces) > np.iinfo(np.int32).max:
        raise ValueError("simplification backend requires 32-bit mesh sizes")
    vertices = np.array(vertices, dtype=np.float64, copy=True)
    faces = np.array(faces, dtype=np.int64, copy=True)
    target_faces = target_faces if target_faces is not None else max(4, len(faces) // 2)
    scale, origin = 1.0, np.zeros(3)
    if faces.size:
        used = vertices[np.unique(faces)]
        lo, hi = used.min(axis=0), used.max(axis=0)
        scale = float(np.max(hi - lo))
        if not math.isfinite(scale) or scale <= 0:
            raise ValueError("mesh extent must be finite and positive")
        origin = lo + (hi - lo) / 2
    # Reserve room for normalization/output rounding at large translations.
    rounding = (
        16
        * np.finfo(np.float64).eps
        * max(scale, float(np.max(np.abs(vertices))) if vertices.size else scale)
        / scale
    )
    tolerance = max_error / scale - rounding
    if tolerance < 1e-12:
        raise ValueError(
            "max_error is below the supported precision for this mesh extent/translation"
        )
    normalized = (vertices - origin) / scale
    components, _, _ = _inspect(normalized, faces)
    out_v, out_f, reports, offset = [], [], [], 0
    for component in components:
        used, inverse = np.unique(faces[component], return_inverse=True)
        v, f = normalized[used], inverse.reshape(-1, 3)
        _, edges, adjacent = _inspect(v, f)
        original_volume = _volume(v[f])
        if original_volume == 0 or not math.isfinite(original_volume):
            raise ValueError("each shell must enclose nonzero finite signed volume")
        euler = len(v) - len(edges) + len(f)
        sharp = _sharp_edges(v, f, edges, adjacent, feature_angle)
        count = max(4, int(target_faces * len(f) / max(1, len(faces))))
        report = dict(
            input_faces=len(f),
            output_faces=len(f),
            status="unchanged",
            attempts=[],
            surface_error_bound=0.0,
            sampled_surface_error=0.0,
            feature_error_bound=0.0,
            relative_volume_change=0.0,
            euler_characteristic=euler,
            orientation=1 if original_volume > 0 else -1,
        )
        accepted_v, accepted_f = v, f
        for _ in range(6):
            if count >= len(f):
                break
            attempt = dict(target_faces=count)
            report["attempts"].append(attempt)
            try:
                cv, cf = _decimate(v, f, count)
                if not np.isfinite(cv).all() or cf.size == 0 or cf.min() < 0 or cf.max() >= len(cv):
                    raise SimplificationError("backend produced invalid geometry")
                parts, ce, ca = _inspect(cv, cf)
                if len(cf) >= len(f):
                    raise SimplificationError("backend did not reduce the face count")
                if len(parts) != 1 or len(np.unique(cf)) - len(ce) + len(cf) != euler:
                    raise SimplificationError("shell topology changed")
                volume = _volume(cv[cf])
                change = abs(volume - original_volume) / abs(original_volume)
                if volume * original_volume <= 0 or change > max_volume_change:
                    raise SimplificationError("shell orientation or volume check failed")
                budget = [max_checks]
                bound, sampled = _symmetric_bound(v[f], cv[cf], tolerance, budget)
                feature_bound, _ = _symmetric_bound(
                    sharp, _sharp_edges(cv, cf, ce, ca, feature_angle), tolerance, budget
                )
                accepted_v, accepted_f = cv, cf
                report.update(
                    status="simplified",
                    output_faces=len(cf),
                    surface_error_bound=(bound + rounding) * scale,
                    sampled_surface_error=sampled * scale,
                    feature_error_bound=(feature_bound + rounding) * scale,
                    relative_volume_change=change,
                )
                attempt.update(accepted=True, checks=max_checks - budget[0])
                break
            except (SimplificationError, ValueError) as error:
                attempt.update(accepted=False, reason=str(error))
                count = (count + len(f) + 1) // 2
        if report["attempts"] and report["status"] == "unchanged":
            report["status"] = "kept"
            if on_failure == "raise":
                raise SimplificationError(
                    f"shell {len(reports)}: no candidate passed; "
                    + report["attempts"][-1]["reason"]
                )
        out_v.append(
            accepted_v * scale + origin
            if report["status"] == "simplified"
            else vertices[used].copy()
        )
        out_f.append(accepted_f + offset)
        offset += len(accepted_v)
        reports.append(report)
    result_v = np.concatenate(out_v) if out_v else vertices.copy()
    result_f = np.concatenate(out_f).astype(np.uint64) if out_f else faces.astype(np.uint64)
    metadata = deepcopy(mesh.metadata) if isinstance(mesh, ReconstructionResult) else {}
    metadata.update(vertex_count=len(result_v), triangle_count=len(result_f))
    metadata["simplification"] = dict(
        backend="fast-simplification",
        backend_version=version("fast-simplification"),
        input_vertices=len(vertices),
        input_faces=len(faces),
        output_vertices=len(result_v),
        output_faces=len(result_f),
        target_faces=target_faces,
        target_reached=len(result_f) <= target_faces,
        max_error=max_error,
        feature_angle=feature_angle,
        max_volume_change=max_volume_change,
        max_checks=max_checks,
        surface_error_bound=max((r["surface_error_bound"] for r in reports), default=0.0),
        feature_error_bound=max((r["feature_error_bound"] for r in reports), default=0.0),
        shells=reports,
        elapsed_ms=(time.perf_counter() - started) * 1000,
    )
    field = (
        mesh.field.copy()
        if isinstance(mesh, ReconstructionResult) and mesh.field is not None
        else None
    )
    return ReconstructionResult(result_v, result_f, metadata, field)
