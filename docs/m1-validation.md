# M1 validation

18 October 2026. M1 implements the dense GPU correctness baseline described in
[decision 0002](decisions/0002-m1-dense-baseline.md). The implementation retains
the M0 Linux x86-64 / CPython 3.13.13 / CUDA 12.8.93 / GCC 11.4 / RTX 4060
validation platform. Python-version and public-release requirements remain in
[distribution-requirements.md](distribution-requirements.md).

## Acceptance evidence

- The full-architecture wheel builds with native code for SM 80/86/89/90 and
  compute-90 PTX. Runtime tests execute on SM 89 only.
- **36 Python tests pass** against the installed wheel with every CUDA toolkit
  directory hidden in a private mount namespace, using the M0 clean runtime
  environment. These comprise 30 reconstruction cases and 6 build/smoke cases.
- **4 native CTest executables pass**, including reconstruction through the
  public C++ API without Python, 1,024 host cube-contour configurations, and
  768 GPU cube configurations with padded neighbors. The GPU checks require
  every output edge to have exactly two oppositely directed incident triangles.
- CUDA Compute Sanitizer reports **zero errors** for memory checking of native
  reconstruction and exhaustive extraction, and initialization checking of
  native reconstruction. These are targeted checks, not proof for every input.
- The installed extension still has no dynamic CUDA runtime dependency or
  toolkit RPATH/RUNPATH. `auditwheel show` reports the same glibc 2.34 symbol
  floor as M0; this remains a local Linux wheel, not a public manylinux release.

## Geometry checks

| Fixture / check | Evidence |
| --- | --- |
| Empty scenes and empty meshes | Correct `(0,3)` arrays; no CUDA initialization; optional empty diagnostic field. |
| Single, disjoint, overlapping, contained boxes | Entire GPU fields agree with analytic per-box SDF minima, within the documented zero convention. Correct component counts, closed edges, winding, Euler characteristic, volume, and boundary error. |
| Complete containment and coincident/reversed duplicate inputs | No added buried surfaces; output agrees with the enclosing/single mesh. |
| Enclosed cavity | Two boundary components, negative inner-wall signed volume, positive outer volume; full field agrees with an analytic shell. |
| Rays through edges and vertices | Octahedron field checked against an independent solid-angle classifier and brute-force triangle distances. |
| Rotated overlapping solids | Random grid samples agree with the independent CPU reference; reversing input order preserves sampled geometry/indexing. |
| Sphere | Refinement improves volume accuracy; bidirectional sampled surface errors include triangle-interior samples and analytic sphere points. |
| Torus | Resolved central hole survives: one component and Euler characteristic zero. |
| World-unit scale invariance | Identical topology and normalized geometry at scales `2^-300` and `2^300`; output validity checks avoid world-space area overflow/underflow. |
| Large world offset | Geometry and indexing agree after translating by `(1e8,-2e8,3e8)` within double-coordinate tolerances. |
| Small component, thin wall, narrow gap | Coarse sampling loses the feature; finer sampling resolves it. Conservative dense coverage is not misrepresented as feature preservation. |
| Face/edge/point contacts and almost-touching surfaces | Finite, indexed, deterministic results. No manifold/contact-topology claim for these degeneracies. |
| Invalid inputs | Reject invalid shapes/dtypes/indices, NaNs, zero-area faces, open/inconsistently oriented/nonmanifold topology, disconnected vertex links, and invalid options. |
| Resource limits and cleanup | Reject over-budget grids and later-stage allocations; a subsequent valid call still succeeds and matches the previous result. |
| Ownership / concurrency | Strided and mixed-dtype inputs are converted correctly; output arrays survive result destruction; concurrent calls agree. |
| GPU availability | Clear failures for nonempty reconstruction with no visible GPU; empty reconstruction still works. |

The reference implementation in `tests/geometry_helpers.py` uses NumPy solid
angles and plane/segment distances, rather than the GPU's BVH ray-parity and
triangle Voronoi-region implementation. Analytic box fields provide a second,
simpler oracle. Comparisons are numerical evidence, not exact Boolean proofs.

## Initial timings

Run `python benchmarks/m1_baseline.py` against an installed wheel. Raw values,
input triangle/shape counts, dimensions, output sizes, build metadata, and
per-stage timings are recorded in
[benchmarks/m1-rtx4060.json](benchmarks/m1-rtx4060.json).

| Workload | Grid samples | Output triangles | FUSE peak device MiB | Median of 3 warm calls |
| --- | --- | --- | --- | --- |
| Cube, spacing 0.04 | 31 × 31 × 31 | 15,000 | 1.84 | 3.73 ms |
| Four overlapping triangulated spheres, spacing 0.06 | 41 × 41 × 30 | 22,088 | 3.21 | 78.32 ms |
| Containment, spacing 0.06 | 56 × 56 × 56 | 60,000 | 9.97 | 11.97 ms |

The first call in the fresh process took about 389 ms, including CUDA startup.
Later workloads' first calls are also recorded but do not represent a fresh
CUDA context. These are initial development fixtures, with no latency promise
or regression threshold. GPU clocks, display load, and allocator/runtime
behavior affect timings. Per-stage wall timings synchronize CUDA; total call
time includes Python conversion and startup. Memory counts include all
FUSE-managed temporary/output buffers, but exclude CUDA context and host memory.
The sphere workload identifies field queries as a likely future optimization
focus; optimize only after representative user workloads are agreed.

## Reproduce

```sh
python -m build --wheel
python -m pip install --force-reinstall --no-deps dist/*.whl
python -m pytest -q
# Linux namespace check using a clean runtime venv:
scripts/test_without_toolkit.sh .venv-runtime/bin/python

cmake -S . -B build/native -DFUSE_BUILD_PYTHON=OFF \
  -DFUSE_BUILD_TESTS=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build/native -j2
ctest --test-dir build/native --output-on-failure
compute-sanitizer --tool memcheck --error-exitcode 99 build/native/fuse_reconstruct
compute-sanitizer --tool memcheck --error-exitcode 99 build/native/fuse_extraction
compute-sanitizer --tool initcheck --error-exitcode 99 build/native/fuse_reconstruct
```

## Limits carried into M2

Dense memory scales with domain volume. The face-decided contour-disk extractor
is not MC33 and does not guarantee continuous trilinear interior topology or
absence of geometric self-intersections for arbitrary saddle configurations.
Self-intersecting input meshes remain unsupported and are not detected by the
implemented topology validation. Degenerate contacts and subvoxel features have
no topology guarantee. Actual driver absence, physical OOM, other GPUs/drivers,
Windows, other Python versions, and a public release are not validated here.
