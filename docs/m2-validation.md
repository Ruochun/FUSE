# M2 validation

18 October 2026. M2 implements the sparse reconstruction path described in
[decision 0003](decisions/0003-m2-sparse.md). Validation uses Linux x86-64,
CPython 3.13.13, CUDA 12.8.93, GCC 11.4, and RTX 4060 (SM 89), as in M1.
The package version is now `0.0.1.dev1` so pip can upgrade the M1 installation.

## Acceptance evidence

- **86 Python tests pass** against the installed full-architecture wheel with
  CUDA toolkit directories hidden in a private mount namespace. The existing
  30 reconstruction tests run on **both backends** (60), plus 20 sparse-specific
  cases and 6 smoke/build cases.
- **4 native CTest executables pass**. The extractor checks 768 synthetic cube
  sign/magnitude configurations on both backends, including NanoVDB leaf,
  internal, and root boundaries. Indexed meshes agree exactly; all edges have
  two oppositely directed incident triangles. Host cell tests cover 1,024 cases.
- Targeted CUDA Compute Sanitizer memory checks of native reconstruction and
  exhaustive extraction, plus initialization checks of native reconstruction,
  report **zero errors**. A memory-check run also covers Python block seams,
  deep containment with dense diagnostic queries, and widely separated solids.
- The wheel contains SM 80/86/89/90 code and compute-90 PTX; execution is tested
  only on SM 89. Dynamic dependency inspection finds no CUDA runtime dependency
  or toolkit RPATH. `auditwheel show` still reports a glibc 2.34 symbol floor;
  this is a local `linux_x86_64` wheel, not the M4 public release.

## Sparse correctness checks

| Check | Evidence |
| --- | --- |
| Dense/sparse equivalence | Full fields and exact indexed meshes agree for overlapping spheres, rotated solids, cavity shells, a torus, tiny components, ray ties, six seeded oblique slivers, and three block-boundary phases. |
| Independent geometric references | The original analytic box fields and independent NumPy solid-angle/distance tests run on both backends, along with volumes, surface errors, components, Euler characteristics, and winding. |
| Shared block faces | Faces on and near 8-cell boundaries in all axes produce closed meshes with shared indices and consistent winding, without welding. Synthetic extraction additionally crosses the NanoVDB root boundary at x=4096. |
| Deep containment | A small cube inside a much larger cube adds candidates but no output surface. Missing diagnostic samples deep inside still evaluate negative. Sparse allocation excludes much of the enclosing cube's interior. |
| Conservative coverage versus resolution | An isolated subvoxel cube increases the candidate-block count even though corner sampling misses its surface. This distinguishes candidate conservativeness from a feature-preservation promise. |
| Large empty domain | Two unit cubes separated by 4096 world units, at spacing 0.25, reconstruct under 16 MiB of device buffers. Active sample/cell counts match the same cubes separated by 8 units. The domain has about 4.4 trillion potential dense samples. Dense reconstruction and dense diagnostic requests reject it explicitly. |
| Memory advantage | A cube at 80 cells per side uses less than 75% of dense peak device memory. Sparse reconstruction succeeds at that measured budget; dense reconstruction rejects the same limit. |
| API and failures | Both backends pass ownership, striding, concurrent-call, resource-failure/retry, empty-scene/no-GPU, scale, large-offset, and invalid-input checks. Unknown backends are rejected. |

## Measurements

Run `python benchmarks/m2_sparse.py`. The raw
[RTX 4060 measurements](benchmarks/m2-rtx4060.json) record four calls per workload
and backend: first call, then three warm calls, with all stage timings and
memory/count metadata. Diagnostics are disabled for these measurements.

| Workload | Dense peak MiB | Sparse peak MiB | Dense warm ms | Sparse warm ms |
| --- | ---: | ---: | ---: | ---: |
| small_cube | 1.84 | 2.60 | 3.59 | 12.32 |
| large_cube | 34.09 | 25.00 | 29.32 | 68.43 |
| overlap_spheres | 3.21 | 4.67 | 73.30 | 81.42 |
| containment | 18.88 | 16.37 | 16.61 | 50.32 |
| distant_cubes | not allocated | 0.75 | — | 7.32 |

These development fixtures establish memory/timing evidence, not performance
regression targets. The first dense call includes CUDA startup; later first
calls are not cold-context measurements. Timings include host topology work
and resource teardown in the total call measurement. Sparse can be slower and
use more memory on small domains: CPU candidate construction and NanoVDB lookup
have costs. M3 should address measured bottlenecks rather than assuming that
sparsity implies speedup. GPU load and clock state affect these timings.

## Reproduce

```sh
python -m build --wheel
python -m pip install --upgrade --no-deps dist/fusemesh-0.0.1.dev1-*.whl
scripts/test_without_toolkit.sh .venv-runtime/bin/python

cmake -S . -B build/native -DFUSE_BUILD_PYTHON=OFF \
  -DFUSE_BUILD_TESTS=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build/native -j2
ctest --test-dir build/native --output-on-failure
compute-sanitizer --tool memcheck --error-exitcode 99 build/native/fuse_reconstruct
compute-sanitizer --tool memcheck --error-exitcode 99 build/native/fuse_extraction
compute-sanitizer --tool initcheck --error-exitcode 99 build/native/fuse_reconstruct
compute-sanitizer --tool memcheck --error-exitcode 99 python -m pytest -q \
  tests/test_sparse.py -k 'deeply or large_empty or shared_faces'
python benchmarks/m2_sparse.py
python examples/union_demo.py --backend sparse
```

Use `--force-reinstall --no-deps` instead of `--upgrade` if rebuilding the same
version. Restart any interpreter that already loaded the native extension.

## Limits

Topology is built on the CPU. Sparse storage is fixed resolution, not adaptive;
subvoxel features and degenerate contacts retain M1 limitations. The extractor
is still a face-decided contour-disk variant, not MC33; geometric
self-intersections and arbitrary cell-interior topology are not guaranteed.
Input self-intersections remain unsupported and undetected. Host construction
and output memory are outside the device budget; the diagnostic host field has
its own size guard. Actual driver absence, physical OOM, other GPUs/drivers,
Windows, CPython 3.9–3.14 coverage, and public distribution remain unvalidated.
