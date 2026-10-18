"""Trimesh adapters for local files and in-memory meshes; no implicit repair."""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import trimesh


def _is_source(value):
    return isinstance(value, (str, os.PathLike, trimesh.Trimesh, trimesh.Scene))


def _arrays(mesh):
    """Own snapshots, validating before unsigned conversion or scene transforms."""
    v, f = np.asarray(mesh.vertices), np.asarray(mesh.faces)
    if v.ndim != 2 or v.shape[1] != 3:
        raise ValueError("vertices must have shape (N, 3)")
    if f.ndim != 2 or f.shape[1] != 3:
        raise ValueError("faces must have shape (M, 3)")
    if not np.isfinite(v).all():
        raise ValueError("vertices must be finite")
    if f.dtype.kind not in 'iu':
        raise ValueError("face indices must be integers")
    if f.size and (np.any(f < 0) or np.any(f >= len(v))):
        raise ValueError("face index is outside the vertex array")
    return (np.array(v, dtype=np.float64, order='C', copy=True),
            np.array(f, dtype=np.uint64, order='C', copy=True))


def _check_off(stream):
    """Check OFF face arity before Trimesh's automatic polygon triangulation.

    Trimesh remains the geometry reader. This streaming guard checks record
    counts and indices too, so malformed faces cannot be discarded or repaired.
    """
    def records():
        for line in stream:
            text = line.split('#', 1)[0].strip()
            if text:
                yield text.split()
    rows = records()
    try:
        header = next(rows)
        if header[0] not in ('OFF', 'COFF'):
            raise ValueError("expected an OFF or COFF header")
        counts = header[1:] or next(rows)
        if len(counts) != 3:
            raise ValueError("OFF counts must contain vertices, faces, and edges")
        nv, nf, ne = map(int, counts)
        if min(nv, nf, ne) < 0:
            raise ValueError("OFF counts must be nonnegative")
        for _ in range(nv):
            row = next(rows)
            if len(row) < 3 or not np.isfinite([float(x) for x in row[:3]]).all():
                raise ValueError("OFF vertices must contain three finite coordinates")
        for i in range(nf):
            row = next(rows)
            if int(row[0]) != 3:
                raise ValueError(f"OFF face {i} is not triangular; triangulate the mesh before loading")
            if len(row) < 4:
                raise ValueError(f"OFF face {i} is truncated")
            if any(int(x) < 0 or int(x) >= nv for x in row[1:4]):
                raise ValueError(f"OFF face {i}: index is outside the vertex array")
        if next(rows, None) is not None:
            raise ValueError("OFF contains records beyond its declared counts")
    except StopIteration as error:
        raise ValueError("OFF file is truncated") from error


def _load_source(source):
    if isinstance(source, trimesh.Trimesh):
        return [_arrays(source)]
    if isinstance(source, trimesh.Scene):
        scene = source
        label = 'scene'
    elif isinstance(source, (str, os.PathLike)):
        path = Path(source)
        label = str(path)
        # Opening the file ourselves limits this helper to local files and gives
        # normal FileNotFoundError/PermissionError diagnostics.
        try:
            if path.suffix.lower() == '.off':
                with path.open('r', encoding='utf-8-sig') as stream:
                    _check_off(stream)
            with path.open('rb') as stream:
                scene = trimesh.load_scene(
                    stream, file_type=path.suffix.lstrip('.').lower(),
                    resolver=trimesh.resolvers.FilePathResolver(str(path.resolve())),
                    process=False, maintain_order=True, allow_remote=False)
        except (ValueError, TypeError, IndexError, KeyError) as error:
            raise ValueError(f"cannot load mesh {path}: {error}") from error
    else:
        raise TypeError("expected a local mesh path, trimesh.Trimesh, or trimesh.Scene")

    result = []
    for node in scene.graph.nodes_geometry:
        transform, name = scene.graph[node]
        mesh = scene.geometry[name]
        if not isinstance(mesh, trimesh.Trimesh):
            raise ValueError(f"{label}: node {node!r} is not a triangle mesh")
        v, f = _arrays(mesh)
        transform = np.asarray(transform, dtype=np.float64)
        if (transform.shape != (4, 4) or not np.isfinite(transform).all()
                or not np.array_equal(transform[3], [0, 0, 0, 1])):
            raise ValueError(f"{label}: node {node!r} requires a finite affine transform")
        # Transform positions once per instance, retaining disconnected shells
        # within the geometry. Reflections must also reverse triangle winding.
        sign, _ = np.linalg.slogdet(transform[:3, :3])
        if sign == 0:
            raise ValueError(f"{label}: node {node!r} has a singular transform")
        if not np.array_equal(transform, np.eye(4)):
            v = np.ascontiguousarray(v @ transform[:3, :3].T + transform[:3, 3])
        if sign < 0:
            f = np.ascontiguousarray(f[:, ::-1])
        if not np.isfinite(v).all():
            raise ValueError(f"{label}: transformed vertices must be finite")
        result.append((v, f))
    if not result:
        raise ValueError(f"{label}: no mesh instances found")
    return result


def load_mesh(source):
    """Load exactly one mesh instance as owned (float64 vertices, uint64 faces).

    Accepts a local file path, Trimesh, or single-mesh Scene. Does not split
    disconnected shells, normalize units, weld vertices, or repair topology.
    OFF/COFF faces must already be triangular. Other file formats follow
    Trimesh's parsing/triangulation rules. Use load_meshes for multiple instances.
    Structural validation happens here; solid/topology validation occurs in
    reconstruct. No GPU is required for loading.
    """
    meshes = _load_source(source)
    if len(meshes) != 1:
        raise ValueError(f"expected one mesh instance, found {len(meshes)}; use load_meshes")
    return meshes[0]


def load_meshes(sources):
    """Load a path/object or iterable of them into FUSE (vertices, faces) pairs.

    Each scene instance becomes a separate solid with its world transform
    applied. Instances must individually represent valid closed solids for
    reconstruction; material groups are not automatically joined. Shells within
    one geometry remain together. Empty source iterables return an empty list.
    """
    if _is_source(sources):
        sources = [sources]
    return [mesh for source in sources for mesh in _load_source(source)]


def save_mesh(path, mesh):
    """Export a ReconstructionResult or (vertices, faces) pair through Trimesh.

    The extension selects the format. Geometry only: no materials or textures.
    """
    if hasattr(mesh, 'vertices') and hasattr(mesh, 'faces'):
        vertices, faces = mesh.vertices, mesh.faces
    else:
        vertices, faces = mesh
    trimesh.Trimesh(vertices=vertices, faces=faces, process=False).export(str(path))
