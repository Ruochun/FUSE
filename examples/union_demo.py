"""Union two overlapping cubes and suppress a fully contained third cube.

Run with an installed fusemesh wheel:
    python examples/union_demo.py --output build/demo-union.obj
Open the OBJ in Blender, MeshLab, or another mesh viewer.
"""
import argparse
from pathlib import Path

import numpy as np
import fusemesh


def box(lo, hi):
    corners = np.array([
        [0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
        [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1],
    ], dtype=np.float64)
    faces = np.array([
        [0, 2, 1], [0, 3, 2], [4, 5, 6], [4, 6, 7],
        [0, 1, 5], [0, 5, 4], [3, 7, 6], [3, 6, 2],
        [0, 4, 7], [0, 7, 3], [1, 2, 6], [1, 6, 5],
    ], dtype=np.int64)
    lo, hi = np.asarray(lo), np.asarray(hi)
    return lo + corners * (hi - lo), faces


def write_obj(path, vertices, faces):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w') as stream:
        stream.write('# FUSE GPU union demo\n')
        for x, y, z in vertices:
            stream.write(f'v {x:.17g} {y:.17g} {z:.17g}\n')
        for a, b, c in faces:
            stream.write(f'f {int(a)+1} {int(b)+1} {int(c)+1}\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--voxel-size', type=float, default=0.05)
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--backend', choices=('sparse', 'dense'), default='sparse')
    parser.add_argument('--output', type=Path, default=Path('build/demo-union.obj'))
    args = parser.parse_args()

    outer = [box((0, 0, 0), (1, 1, 1)),
             box((0.55, 0.35, 0.2), (1.55, 1.35, 1.2))]
    inner = box((0.2, 0.2, 0.2), (0.4, 0.4, 0.4))
    result = fusemesh.reconstruct(outer + [inner], args.voxel_size, device=args.device, backend=args.backend)

    # Containment should change neither sampled geometry nor indexing.
    reference = fusemesh.reconstruct(outer, args.voxel_size, device=args.device, backend=args.backend)
    np.testing.assert_array_equal(result.faces, reference.faces)
    np.testing.assert_allclose(result.vertices, reference.vertices, rtol=0, atol=1e-12)
    faces = result.faces.astype(np.int64)
    if not len(faces):
        raise RuntimeError('No surface resolved; try a smaller --voxel-size')
    edges = np.sort(np.concatenate((faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]])), axis=1)
    _, incidence = np.unique(edges, axis=0, return_counts=True)
    if not np.all(incidence == 2):
        raise RuntimeError('Expected a closed mesh with two faces per edge')
    a, b, c = result.vertices[faces].transpose(1, 0, 2)
    volume = np.einsum('ij,ij->i', a, np.cross(b, c)).sum() / 6
    exact_volume = 2 - 0.45 * 0.65 * 0.8
    write_obj(args.output, result.vertices, result.faces)
    print(f'Output: {args.output.resolve()}')
    print(f"Backend: {result.metadata['backend']}")
    print(f'{len(result.vertices):,} vertices; {len(faces):,} triangles')
    print('Checks passed: contained cube suppressed; mesh edges closed')
    print(f'Volume: {volume:.6f}; analytic union: {exact_volume:.6f}; '
          f'relative error: {abs(volume-exact_volume)/exact_volume:.2%}')
    print(f"Peak FUSE device buffers: {result.metadata['peak_device_bytes']/1024**2:.2f} MiB")


if __name__ == '__main__':
    main()
