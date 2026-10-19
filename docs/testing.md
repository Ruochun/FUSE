# Testing and benchmarks

Run commands from a repository checkout after installing a wheel built from the
same sources. Tests import the installed package. Building a new wheel does not
replace an already installed build of the same version; see
[reinstallation](installation.md#reinstall-a-rebuilt-wheel).

## Correctness tests

```bash
python -m pip install pytest
python -m pytest tests -q
```

The full suite needs a compatible NVIDIA GPU and driver. Without a visible GPU:

```bash
python -m pytest tests -m 'not gpu' -q
```

The CPU selection checks loading, validation, metadata, packaging rules, and
simplification. GPU tests cover both reconstruction backends, analytic and
independent reference fields, overlapping/contained solids, cavities, handles,
file integration, resource limits, and resolution-dependent feature loss.
Simplification tests check planar corners, curved surfaces, thin solids, shell
orientation, geometric deviation, topology, and rejected candidates.

Native tests can be built and run using the commands in
[installation](installation.md#build-from-source). They exercise the C++ API,
CUDA smoke checks, and cell extraction configurations.

On a Linux development host with user/mount namespaces enabled,
`scripts/test_without_toolkit.sh /absolute/path/to/venv/bin/python` runs the suite
in a private namespace with `/usr/local` hidden and CUDA/Python path overrides
removed. This assumes the host toolkits live under `/usr/local`; it is not a
universal toolkit-isolation method. Hosted CI and driver-only GPU containers
provide separate package checks; see [the maintainer guide](releasing.md).

## Reconstruction performance

```bash
python benchmarks/m3_performance.py --output build/reconstruction-benchmark.json
```

The runner warms CUDA separately and records warm medians, end-to-end call times,
input hashes, output counts, and explicit device-memory use. Sparse indexing can
cost more than dense sampling for small or nearly full candidate domains; measure
your workload instead of assuming a speedup.

To include local files, create a JSON manifest:

```json
[
  {"name": "part", "paths": ["part.off"], "voxel_size": 0.1}
]
```

Paths are relative to the manifest. Use `--manifest /path/to/manifest.json`;
loading is timed separately. Manifest entries can also set `backend`,
`validation`, and `max_memory_bytes` (default 512 MiB). Use `--repeats` for the
number of timed runs; its minimum is three and its default is seven.

A regression comparison requires the same GPU/driver, inputs, options, and
workload set:

```bash
python benchmarks/m3_performance.py \
  --baseline benchmarks/baselines/reconstruction-rtx4060.json --check
```

The guard rejects warm medians above `1.5 × reference + 2 ms`, explicit device
memory above `1.05 × reference + 4096 bytes`, and changed geometry counts or
workloads. These are regression thresholds, not application latency guarantees.
Keep a separate baseline for each environment and investigate differences before
replacing it. Raw reference reports are stored in `benchmarks/baselines/`.

## Simplification performance

```bash
python benchmarks/benchmark_simplify.py
python benchmarks/benchmark_simplify.py --gpu
```

The first command measures CPU-only fixtures; the second also reconstructs and
simplifies meshes on the GPU-equipped host. Each case runs in a fresh process.
Reported time includes simplification and its checks, but excludes loading and
reconstruction. Peak RSS includes Python and libraries; its growth is a
high-water-mark difference, not a hard allocation budget.

Example measurements on Linux under WSL, Python 3.13.13, NumPy 2.5.4,
fast-simplification 0.2.0, and RTX 4060:

| Fixture | Input → output triangles | max_error | Relative volume change | Time |
| --- | ---: | ---: | ---: | ---: |
| Subdivided box | 768 → 12 | 0.01 | 0 | 0.34 s |
| Icosphere | 1,280 → 800 | 0.04 | 0.27% | 1.62 s |
| Thin box (thickness 0.02) | 192 → 160 | 0.005 | 0 | 2.56 s |
| Box with cavity shell | 1,536 → 24 | 0.01 | 0 | 0.72 s |
| Reconstructed box, spacing 0.12 | 1,536 → 100 | 0.03 | 0.0000601% | 1.59 s |
| Reconstructed overlapping-box union, spacing 0.12 | 2,240 → 1,170 | 0.03 | 0.0000444% | 4.34 s |

These are single measurements on generated fixtures, not guarantees for other
meshes or hardware. Thin geometry can retain many faces even when surfaces are
planar. Conservative error bounds can lie near the requested tolerance while
actual deviation is much smaller. See [the quality contract](simplification.md).
