# Reconstruction quality

Compare an original surface (or the union of several solids) with a reconstructed
mesh using CPU point-to-triangle distances. No GPU is needed for this utility.

```python
import fusemesh

report = fusemesh.compare_surfaces(
    ["part-a.off", "part-b.obj"],
    "reconstructed.ply",
    samples=10000,
    seed=0,
    tolerance=0.02,
    voxel_size=0.01,
)
print(report["symmetric"]["rms"])
print(report["symmetric"]["f_score"])
```

For multiple originals, install the Boolean reference dependency:

```bash
python -m pip install 'fusemesh[quality]'
```

For an unpublished local wheel, install `'/path/to/fusemesh-...whl[quality]'`
instead. The extra installs Manifold separately; no external executable is
required. Pip resolves Manifold's interpreter-specific NumPy requirement.
Single-surface comparisons use only the base package dependencies.

## Runnable reconstruction example

[The quality demo](../examples/quality_demo.py) creates two overlapping boxes,
reconstructs their union on the GPU, and evaluates both surface-distance
directions on the CPU. With the current wheel and its `quality` extra installed:

```bash
python examples/quality_demo.py --output-dir build/quality-demo
```

To compare an original mesh with an existing reconstructed/output mesh, pass
exactly two files in that order (each must contain one mesh instance):

```bash
python examples/quality_demo.py --inputs original.off reconstructed.obj \
  --tolerance 0.05 --output-dir build/my-quality-demo
```

This mode runs entirely on the CPU: no reconstruction, GPU, Boolean union, or
`quality` extra is needed. Both meshes may be open or nonmanifold, without an
override flag. They must still contain finite vertices and valid indices.
Degenerate triangles are rejected by default; use `--degenerate-faces drop` to
exclude them. The original surface is compared as supplied, with no repair or
removal of buried faces. Inputs are left untouched; only `quality.json` is
written. This also works with an output mesh from `union_demo.py`, provided you
supply the reference surface you intend to measure it against.

Choose tolerance in your meshes' coordinate units. For file inputs, optional
`--voxel-size` only adds normalized distance metrics; omit it if the reconstruction
spacing is unknown. `--device` applies only to the generated-box mode.

Omitting `--inputs` keeps the generated-box reconstruction example, using a
voxel size of 0.1 unless specified. That mode saves `original-a.off`,
`original-b.off`, `reconstructed.off`, and `quality.json`. Both modes print RMS,
p95, sampled maximum, coverage, and F-score.

To compare a finer reconstruction, keep the sampling settings and tolerance
fixed and use a separate output directory:

```bash
python examples/quality_demo.py --voxel-size 0.05 --samples 10000 --seed 0 \
  --tolerance 0.05 --output-dir build/quality-demo-fine
```

You can also score the saved meshes again without running reconstruction or
requiring a GPU:

```bash
fusemesh-quality build/quality-demo/original-a.off build/quality-demo/original-b.off \
  --reconstructed build/quality-demo/reconstructed.off \
  --samples 10000 --seed 0 --tolerance 0.05 --voxel-size 0.1
```

## Inputs and reference surface

Both arguments accept local file paths, Trimesh objects, `(vertices, faces)`
pairs, or `ReconstructionResult`. Originals also accept an iterable mixing
these types. Scenes expand into transformed mesh instances. The reconstructed
argument must resolve to one mesh instance, which may contain multiple shells.
All coordinates must use the same units and alignment; comparison does not
register, rescale, repair, or modify the inputs.

One original is compared as supplied. Open and nonmanifold surfaces are allowed
on either side, but surfaces must be nonempty with finite vertices, valid indices,
and no degenerate triangles unless `degenerate_faces="drop"` is selected.
Multiple originals are treated as closed solids:
Manifold constructs their Boolean union in float64, independent of FUSE's voxel
reconstruction. Buried and partially buried faces are removed before sampling
and nearest-surface queries. Each solid must have consistent manifold topology,
positive total signed volume, outward outer shells, and inward cavity shells.
The utility checks topology and total volume; it does not certify shell nesting
or detect self-intersections. Supply valid solids. Boolean failures raise errors;
there is no fallback to concatenating the input surfaces.

A single mesh containing overlapping solid shells is compared literally; pass
those solids separately to request a union. Oppositely oriented cavity shells
must remain together in one original mesh.

## Degenerate triangles

By default, quality comparison rejects zero-area triangles. To ignore them in
both input surfaces, use:

```bash
python examples/quality_demo.py --inputs original.off reconstructed.obj \
  --tolerance 0.001 --degenerate-faces drop --output-dir build/quality-cleaned
```

The same option is available on `fusemesh-quality`. In Python:

```python
report = fusemesh.compare_surfaces(
    "original.off", "reconstructed.obj", degenerate_faces="drop"
)
print(report["cleaning"])
```

Cleaning operates on temporary copies and never changes input files or arrays.
It removes repeated-index faces and faces whose cross product is zero in
floating-point arithmetic, including collinear or coincident vertices. There is
no near-zero threshold: thin triangles with positive computed area are retained.
Per-triangle scaling reduces overflow and underflow when identifying zero area.
This is a floating-point check, not an exact-arithmetic degeneracy predicate.

`cleaning.originals` contains one count record per original mesh instance;
`cleaning.reconstructed` contains the output mesh's record. Each reports
`input_faces`, `removed_faces`, and `retained_faces`. The selected policy is
recorded as `cleaning.degenerate_faces`. The demo also prints removal counts.
Either mode rejects invalid coordinates/indices; drop mode fails if **any** input
mesh has no valid triangles left. Dropping faces does not repair topology, so
multiple-original Boolean comparisons still require valid closed solids after
cleaning. This option affects quality scoring, not reconstruction validation.

Dropped faces may represent dangling lines or points. Those features are
excluded from the score, which measures the retained triangle surfaces.

## Reading the report

`original_to_reconstructed` measures missing or displaced reference surface;
`reconstructed_to_original` measures extra or displaced reconstructed surface.
Each direction samples the source uniformly by area and measures distance to
the nearest **triangle**, not to a sampled target point or vertex.

Each directional report and `symmetric` contains:

| Key | Meaning |
| --- | --- |
| `rms` | Root mean square distance; emphasizes larger errors. Smaller is better. |
| `mean` | Mean unsigned surface distance. |
| `p95` | 95% of sampled distances are at or below this value. Smaller is better. |
| `sampled_max` | Largest observed distance, **not a certified Hausdorff distance**. |
| `coverage` | Fraction of samples within `tolerance`, or null if omitted. |

The symmetric report pools equal sample counts from each direction, weighting
the two surfaces equally even when their areas differ. Its RMS is
`sqrt((forward_rms**2 + backward_rms**2) / 2)`. With a tolerance, `f_score` is
the harmonic mean of directional coverages: 1 means every sampled point is
within tolerance, and 0 means at least one direction has zero coverage.
RMS is the primary distance score; F-score is a dimensionless tolerance-based
score. No universal tolerance fits every model. For example, directional
coverages of 90% and 100% give `f_score = 2 * 0.9 * 1.0 / (0.9 + 1.0)`, about
0.9474. Symmetric coverage is their average (95%), while F-score penalizes the
imbalance more strongly. Keep tolerance and sampling settings fixed when
comparing reconstructions.

Distances and tolerance use mesh coordinate units. Optional `voxel_size` adds
`symmetric_in_voxels`; it does not change sampling or the reference. Sampling
settings, reference construction, and dependency versions are included for
reproducibility. Reports are JSON-compatible dictionaries.

These are Monte Carlo estimates. Increasing `samples` improves coverage, but
small defects or thin components can still be missed. A fixed seed repeats a
comparison in the same software environment; different triangulations or
versions can change sampled points. Distance scores do not establish topology,
watertightness, or absence of self-intersections. Check those properties
separately. Boolean reference accuracy is also limited by floating-point
arithmetic. The utility normalizes both inputs together during computation to
reduce numerical sensitivity to scale and translation.

## Command line

```bash
fusemesh-quality part-a.off part-b.obj --reconstructed reconstructed.ply \
  --samples 20000 --seed 42 --tolerance 0.02 --voxel-size 0.01 \
  --output quality.json
```

`python -m fusemesh.quality` accepts the same arguments. Omit `--output` to print
JSON to stdout. Invalid input or a failed union exits with status 2.

The Python defaults are `samples=10000` per direction, `seed=0`,
`tolerance=None`, `voxel_size=None`, `batch_size=256`, and
`degenerate_faces="error"`. Sample count and batch size must be positive integers; seed must be a nonnegative integer. Optional
tolerance and voxel size must be finite and positive. Batch size controls query
batching without changing the sampled points; it is not a hard memory limit.
The spatial index and Boolean reference require host memory proportional to the
geometry, and sampling arrays require memory proportional to the sample count.
