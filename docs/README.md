# FUSE documentation

FUSE approximates the union of closed triangle-mesh solids by sampling a shared
field and extracting an indexed surface. Reconstruction runs on an NVIDIA GPU;
file loading, export, surface simplification, and quality comparisons run on the CPU.

## Using FUSE

1. [Install from PyPI, a downloaded wheel, or source](installation.md).
2. [Load files and reconstruct a union](mesh-loading.md).
3. [Reduce the output triangle count](simplification.md).
4. [Measure reconstruction quality](quality.md).
5. Consult the [API reference](api.md) or [troubleshooting guide](troubleshooting.md).

Use a voxel size appropriate to your mesh's coordinate units. Sampling can lose
features smaller than that scale, and passing topology validation does not prove
that a mesh is free of self-intersections.

## Technical and contributor references

- [Testing and benchmarks](testing.md): run correctness checks and measure your workload.
- [Architecture and numerical behavior](FUSE-design.md): field semantics, sparse indexing, and extraction.
- [Distribution policy](distribution-requirements.md): wheel targets and runtime dependencies.
- [Maintainer release guide](releasing.md): formatting, CI, GPU qualification, and publishing.
