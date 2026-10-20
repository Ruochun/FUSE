"""Compare compact topology against the original independent Python graph checker."""

import numpy as np
import pytest
import trimesh
from fusemesh.simplification import _inspect, _mesh_volume


def reference_inspect(vertices, faces):
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


@pytest.mark.parametrize("kind", ["box", "sphere", "shells", "reversed", "shuffled"])
def test_compact_matches_reference(kind):
    mesh = (
        trimesh.creation.icosphere(subdivisions=2) if kind == "sphere" else trimesh.creation.box()
    )
    if kind == "shells":
        other = mesh.copy()
        other.apply_translation([3, 0, 0])
        mesh = trimesh.util.concatenate([mesh, other])
    v, f = np.array(mesh.vertices), np.array(mesh.faces)
    if kind == "reversed":
        f = f[:, ::-1].copy()
    if kind == "shuffled":
        np.random.default_rng(7).shuffle(f)
    expected, ee, ea = reference_inspect(v, f)
    actual, ae, aa = _inspect(v, f)
    assert [c.tolist() for c in actual] == [c.tolist() for c in expected]

    def pairs(edges, adjacent):
        return {tuple(e): tuple(sorted(a)) for e, a in zip(edges, adjacent)}

    assert pairs(ae, aa) == pairs(ee, ea)
    only, edges, adjacent = _inspect(v, f, components_only=True)
    assert [c.tolist() for c in only] == [c.tolist() for c in expected]
    assert edges.shape == adjacent.shape == (0, 2)
    assert _mesh_volume(v, f) == pytest.approx(mesh.volume * (-1 if kind == "reversed" else 1))


@pytest.mark.parametrize(
    "defect", ["boundary", "winding", "duplicate", "double", "degenerate", "pinch"]
)
def test_invalid_topology_matches_reference(defect):
    mesh = trimesh.creation.box()
    v, f = np.array(mesh.vertices), np.array(mesh.faces)
    if defect == "boundary":
        f = f[:-1]
    if defect == "winding":
        f[0] = f[0, ::-1]
    if defect == "duplicate":
        f = np.vstack([f, f[0]])
    if defect == "double":
        f = np.array([[0, 1, 2], [2, 1, 0]])
    if defect == "degenerate":
        v[f[0, 1]] = v[f[0, 0]]
    if defect == "pinch":
        v = np.vstack([v, v + [2, 2, 2]])
        f = np.vstack([f, f + 8])
        f[f == 8] = 0
    for inspect in (reference_inspect, _inspect):
        with pytest.raises(ValueError):
            inspect(v, f)


def test_empty_and_unused_vertices():
    parts, edges, adjacent = _inspect(np.empty((0, 3)), np.empty((0, 3), dtype=np.int64))
    assert not parts and edges.shape == adjacent.shape == (0, 2)
    mesh = trimesh.creation.box()
    parts, _, _ = _inspect(np.vstack([mesh.vertices, [[100, 0, 0]]]), mesh.faces)
    assert len(parts) == 1


def test_pinched_vertex_inside_one_edge_connected_shell():
    mesh = trimesh.creation.icosphere(subdivisions=2)
    v, f = np.array(mesh.vertices), np.array(mesh.faces)
    # Antipodal vertices have disjoint links but belong to the same face component.
    other = np.argmin(v @ v[0])
    f[f == other] = 0
    for inspect in (reference_inspect, _inspect):
        with pytest.raises(ValueError, match="vertex-manifold"):
            inspect(v, f)


def test_native_rejects_invalid_indices():
    mesh = trimesh.creation.box()
    for index in (-1, len(mesh.vertices)):
        f = np.array(mesh.faces)
        f[0, 0] = index
        with pytest.raises(ValueError, match="index"):
            _inspect(mesh.vertices, f)


def test_insufficient_budget_does_not_build_spatial_index(monkeypatch):
    from fusemesh import simplification as impl

    def unexpected(*args):
        pytest.fail("an impossible budget should fail before allocating an index")

    monkeypatch.setattr(impl, "_Surface", unexpected)
    triangles = trimesh.creation.box().triangles
    with pytest.raises(impl.SimplificationError, match="budget"):
        impl._symmetric_bound(triangles, triangles, 0.01, [len(triangles)])


def test_impossible_shell_budget_does_not_decimate(monkeypatch):
    from fusemesh import simplification as impl

    def unexpected(*args):
        pytest.fail("an impossible budget should fail before decimation")

    monkeypatch.setattr(impl, "_decimate", unexpected)
    mesh = trimesh.creation.box().subdivide()
    with pytest.raises(impl.SimplificationError, match="budget"):
        impl.simplify(mesh, max_error=0.01, max_checks=len(mesh.faces))


def test_batched_geometry_crosses_chunk_boundary():
    from fusemesh.simplification import _sharp_edges

    mesh = trimesh.creation.box()
    count = 5500  # 66,000 triangles: crosses the 65,536-face batch boundary.
    offsets = np.arange(count)[:, None, None] * np.array([3.0, 0, 0])
    v = (mesh.vertices[None, :, :] + offsets + 1e6).reshape(-1, 3)
    f = (mesh.faces[None, :, :] + np.arange(count)[:, None, None] * 8).reshape(-1, 3)
    parts, edges, adjacent = _inspect(v, f)
    assert len(parts) == count
    assert _mesh_volume(v, f) == pytest.approx(count)
    sharp = _sharp_edges(v, f, edges, adjacent, 45)
    assert len(sharp) == 12 * count
    np.testing.assert_allclose(np.linalg.norm(sharp[:, 1] - sharp[:, 0], axis=1), 1)


def test_spatial_index_batch_ids_are_global():
    from fusemesh.simplification import _Surface

    starts = np.column_stack([np.arange(65538) * 3.0, np.zeros((65538, 2))])
    segments = np.stack([starts, starts + [1, 0, 0]], axis=1)
    surface = _Surface(segments)
    try:
        indices = np.array([0, 65535, 65536, 65537])
        distance, ids = surface.nearest(starts[indices] + [0.5, 0.25, 0])
        np.testing.assert_array_equal(ids, indices)
        np.testing.assert_allclose(distance, 0.25)
    finally:
        surface.close()
