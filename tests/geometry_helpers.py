"""Independent NumPy geometry fixtures and reference checks (no FUSE internals)."""

import numpy as np


def box(lo=(0, 0, 0), hi=(1, 1, 1)):
    vertices = np.array(
        [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0], [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]],
        dtype=float,
    )
    faces = np.array(
        [
            [0, 2, 1],
            [0, 3, 2],
            [4, 5, 6],
            [4, 6, 7],
            [0, 1, 5],
            [0, 5, 4],
            [3, 7, 6],
            [3, 6, 2],
            [0, 4, 7],
            [0, 7, 3],
            [1, 2, 6],
            [1, 6, 5],
        ],
        dtype=np.int64,
    )
    return np.asarray(lo) + vertices * (np.asarray(hi) - lo), faces


def combine(*meshes):
    vertices, faces, offset = [], [], 0
    for v, f in meshes:
        vertices.append(v)
        faces.append(f + offset)
        offset += len(v)
    return np.concatenate(vertices), np.concatenate(faces)


def sphere(center=(0, 0, 0), radius=1, levels=3):
    vertices = [
        np.array(p, dtype=float)
        for p in [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)]
    ]
    faces = [(0, 2, 4), (2, 1, 4), (1, 3, 4), (3, 0, 4), (2, 0, 5), (1, 2, 5), (3, 1, 5), (0, 3, 5)]
    for _ in range(levels):
        cache, refined = {}, []

        def midpoint(a, b):
            key = tuple(sorted((a, b)))
            if key not in cache:
                p = vertices[a] + vertices[b]
                cache[key] = len(vertices)
                vertices.append(p / np.linalg.norm(p))
            return cache[key]

        for a, b, c in faces:
            ab, bc, ca = midpoint(a, b), midpoint(b, c), midpoint(c, a)
            refined.extend([(a, ab, ca), (ab, b, bc), (ca, bc, c), (ab, bc, ca)])
        faces = refined
    return np.asarray(vertices) * radius + center, np.asarray(faces, dtype=np.int64)


def torus(major=1.0, minor=0.35, nu=40, nv=20):
    u = np.arange(nu) * 2 * np.pi / nu
    v = np.arange(nv) * 2 * np.pi / nv
    u, v = np.meshgrid(u, v, indexing="ij")
    vertices = np.stack(
        (
            (major + minor * np.cos(v)) * np.cos(u),
            (major + minor * np.cos(v)) * np.sin(u),
            minor * np.sin(v),
        ),
        axis=-1,
    ).reshape(-1, 3)
    faces = []
    for i in range(nu):
        for j in range(nv):
            a = i * nv + j
            b = ((i + 1) % nu) * nv + j
            c = ((i + 1) % nu) * nv + (j + 1) % nv
            d = i * nv + (j + 1) % nv
            faces.extend(((a, b, c), (a, c, d)))
    return vertices, np.asarray(faces, dtype=np.int64)


def box_sdf(points, lo, hi):
    q = np.abs(points - (np.asarray(lo) + hi) * 0.5) - (np.asarray(hi) - lo) * 0.5
    return np.linalg.norm(np.maximum(q, 0), axis=-1) + np.minimum(np.max(q, axis=-1), 0)


def sample_positions(result):
    z, y, x = np.indices(result.field.shape)
    return np.stack((x, y, z), axis=-1) * result.metadata["voxel_size"] + result.metadata["origin"]


def unsigned_distance(points, vertices, faces):
    """Brute-force projected-plane/segment reference, independent of BVH queries."""
    a, b, c = vertices[faces].transpose(1, 0, 2)
    ab, ac = b - a, c - a
    normal = np.cross(ab, ac)
    normal2 = np.einsum("ij,ij->i", normal, normal)
    results = []
    for p in np.asarray(points):
        signed = np.einsum("ij,ij->i", p - a, normal)
        projection = p - normal * (signed / normal2)[:, None]
        inside = np.ones(len(faces), dtype=bool)
        for s, t in ((a, b), (b, c), (c, a)):
            inside &= np.einsum("ij,ij->i", np.cross(t - s, projection - s), normal) >= -1e-14
        distance = np.where(inside, signed * signed / normal2, np.inf)
        for s, t in ((a, b), (b, c), (c, a)):
            edge = t - s
            alpha = np.clip(
                np.einsum("ij,ij->i", p - s, edge) / np.einsum("ij,ij->i", edge, edge), 0, 1
            )
            delta = p - (s + alpha[:, None] * edge)
            distance = np.minimum(distance, np.einsum("ij,ij->i", delta, delta))
        results.append(np.sqrt(np.min(distance)))
    return np.asarray(results)


def reference_field(points, meshes):
    """Solid-angle classification, intentionally different from GPU ray parity."""
    output = np.full(len(points), np.inf)
    for vertices, faces in meshes:
        distances = unsigned_distance(points, vertices, faces)
        triangles = vertices[faces]
        inside = []
        for p in points:
            a, b, c = (triangles - p).transpose(1, 0, 2)
            la, lb, lc = [np.linalg.norm(v, axis=1) for v in (a, b, c)]
            determinant = np.einsum("ij,ij->i", a, np.cross(b, c))
            denominator = (
                la * lb * lc
                + np.einsum("ij,ij->i", a, b) * lc
                + np.einsum("ij,ij->i", b, c) * la
                + np.einsum("ij,ij->i", c, a) * lb
            )
            winding = np.sum(2 * np.arctan2(determinant, denominator))
            inside.append(abs(winding) > 2 * np.pi)
        output = np.minimum(output, np.where(inside, -distances, distances))
    return output


def mesh_quality(result, expected_components=None, expected_euler=None):
    v, f = result.vertices, result.faces.astype(np.int64)
    assert v.ndim == 2 and v.shape[1] == 3 and np.isfinite(v).all()
    assert f.ndim == 2 and f.shape[1] == 3 and len(f) > 0
    assert f.min() >= 0 and f.max() < len(v)
    assert len(np.unique(f)) == len(v), "unreferenced vertices"
    tri = v[f]
    area = np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1)
    assert np.all(area > 0), "degenerate triangle"
    directed = np.concatenate((f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]))
    edges, inverse, counts = np.unique(
        np.sort(directed, axis=1), axis=0, return_inverse=True, return_counts=True
    )
    assert np.all(counts == 2), "open or nonmanifold edge"
    balance = np.bincount(inverse, weights=np.where(directed[:, 0] < directed[:, 1], 1, -1))
    assert np.all(balance == 0), "inconsistent face winding"
    parent = np.arange(len(v))

    def root(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for a, b in edges:
        ra, rb = root(a), root(b)
        if ra != rb:
            parent[ra] = rb
    labels = np.array([root(i) for i in range(len(v))])
    components = np.unique(labels)
    if expected_components is not None:
        assert len(components) == expected_components
    euler = len(v) - len(edges) + len(f)
    if expected_euler is not None:
        assert euler == expected_euler
    # Shift the origin for accurate volumes with large world-coordinate offsets.
    shifted = v - v.mean(axis=0)
    a, b, c = shifted[f].transpose(1, 0, 2)
    face_volumes = np.einsum("ij,ij->i", a, np.cross(b, c)) / 6
    volumes = [face_volumes[labels[f[:, 0]] == label].sum() for label in components]
    return np.asarray(volumes)


def surface_samples(result, count=1000):
    f = result.faces.astype(np.int64)
    idx = np.linspace(0, len(f) - 1, min(count, len(f)), dtype=int)
    tri = result.vertices[f[idx]]
    return np.concatenate(
        (
            result.vertices[:: max(1, len(result.vertices) // count)],
            tri.mean(axis=1),
            0.6 * tri[:, 0] + 0.3 * tri[:, 1] + 0.1 * tri[:, 2],
        )
    )
