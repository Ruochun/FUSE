# FUSE — Fast Union Surface Extraction

FUSE reconstructs a triangle mesh approximating the solid union of one or more
closed meshes. It uses CUDA for field evaluation and surface extraction, sparse
NanoVDB indexing to reduce work around empty space, and a Python API for loading,
fusing, simplifying, comparing, and saving meshes.

Choose a uniform voxel size in your mesh's units to control sampling resolution.
FUSE handles overlapping solids, containment, disconnected components, and
resolved cavities. Optional CPU simplification reduces the output triangle count.
It is an approximate reconstruction tool: original triangles and subvoxel features
are not necessarily preserved.

## Install

FUSE is published on PyPI as `fusemesh`. On supported Linux x86-64 systems
with CPython 3.9–3.14, install it with:

```bash
python -m pip install fusemesh
python -c 'import fusemesh; print(fusemesh.build_info())'
```

Reconstruction requires an NVIDIA GPU and driver. Compatible wheel installations
include the CUDA runtime and Python dependencies; no CUDA toolkit, compiler, or
`[cuda]` extra is required. To check GPU execution, run
`python -c 'import fusemesh; print(fusemesh.smoke_test())'`.
Loading, saving, simplification, and quality comparisons run on the CPU without
initializing CUDA.

See [installation and compatibility](docs/installation.md) for GPU requirements,
downloaded wheels, source builds, and reinstalling a changed wheel with the same
version number. Example scripts are in the repository, not in the pip package;
clone or download the repository to run them.

## Fuse mesh files

```python
import fusemesh

result = fusemesh.reconstruct(["a.off", "b.obj"], voxel_size=0.05)
fusemesh.save_mesh("union.obj", result)
print(result.metadata)
```

A single path also works: `fusemesh.reconstruct("model.off", voxel_size=0.05)`.
For arrays, use `reconstruct([(vertices, faces)], voxel_size=0.05)`; each array
pair represents one solid. Returned vertices are float64 `(N, 3)` coordinates
and faces are uint64 `(M, 3)` indices.

From a repository checkout, run the file example:

```bash
python examples/fuse_files.py a.off b.obj \
  --voxel-size 0.05 --memory-mib 512 --output union.obj
```

Fusion defaults to `validation="full"`, which checks numeric/index validity and
closed, consistently oriented manifold topology. To explicitly skip topology
checks, use `validation="basic"` in Python:

```python
result = fusemesh.reconstruct("model.off", voxel_size=0.05, validation="basic")
```

Or pass `--validation basic` to the file example:

```bash
python examples/fuse_files.py model.off --voxel-size 0.05 \
  --validation basic --memory-mib 512 --output union.obj
```

Use full validation by default. Basic mode keeps numeric/index and zero-area
triangle checks but does not repair defects or make open/nonmanifold inputs reliable solids. Neither mode
detects self-intersections. See [mesh validation](docs/mesh-loading.md#validation-when-fusing).

To try generated geometry without supplying a file:

```bash
python examples/union_demo.py
```

## Reduce output triangles

```python
reduced = fusemesh.simplify(result, max_error=0.01, target_faces=5000)
fusemesh.save_mesh("reduced.obj", reduced)
```

Simplification checks surface and sharp-edge deviation, per-shell volume, and
manifold topology. `target_faces` is a soft goal; quality checks take priority.
`max_error` is relative to the mesh being simplified, not the original union
inputs. See [simplification](docs/simplification.md) for controls and limitations.

## Measure surface accuracy

```python
report = fusemesh.compare_surfaces(
    "original.off", "reconstructed.obj", tolerance=0.05,
    degenerate_faces="drop",
)
print(report["symmetric"]["rms"])
print(report["symmetric"]["f_score"])
```

This CPU comparison accepts open or nonmanifold surfaces. The explicit `drop`
option excludes zero-area triangles and reports removal counts; the default is
`error`. Distances use mesh coordinate units. Coverage and F-score depend on the
chosen tolerance, and sampled maxima are not certified Hausdorff bounds.

From a checkout, compare existing files with
`python examples/quality_demo.py --inputs original.off reconstructed.obj`.
Without `--inputs`, the demo reconstructs two boxes and requires a GPU and the
`quality` extra. Multiple original solids also require that extra to construct
an independent Boolean reference union. See [quality comparisons](docs/quality.md)
for installation, metric definitions, and examples.

## Reconstruction requirements and limits

- Inputs must describe closed, consistently oriented manifold solids.
  `validation="basic"` skips topology checks but does not make open or
  nonmanifold solids supported. Self-intersections are not detected or repaired.
- Trimesh loads local files. OFF faces must already be triangles; other formats
  follow their reader's triangulation and grouping rules. See [mesh loading](docs/mesh-loading.md).
- Smaller voxel sizes can resolve finer detail but require more work and memory.
  Thin walls, narrow gaps, and small components may disappear. Voxel size is not
  a guaranteed surface-error bound; output is not guaranteed manifold in
  degenerate or unresolved cases.
- The Python API defaults to a 512 MiB limit on explicit FUSE GPU buffers.
  The file example defaults to 16 GiB; use `--memory-mib` to set it explicitly.
  CUDA context memory and host memory are outside that limit.
- Reconstruction does not perform exact triangle Booleans, mesh repair, adaptive voxel
  refinement, automatic smoothing, or material/texture transfer.

## Documentation

Start with the [documentation index](docs/README.md), or go directly to:

- [Installation and compatibility](docs/installation.md)
- [Mesh loading and file examples](docs/mesh-loading.md)
- [Python and native API](docs/api.md)
- [Surface simplification](docs/simplification.md)
- [Reconstruction quality](docs/quality.md)
- [Troubleshooting](docs/troubleshooting.md)
- [Testing and benchmarks](docs/testing.md)

FUSE uses BSD-3-Clause. Third-party notices are in [licenses](licenses/THIRD-PARTY-NOTICES.txt)
and the installed wheel metadata.
