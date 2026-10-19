# Troubleshooting

## A rebuilt wheel still imports the old code

Installing a wheel with the same version may leave the existing installation
in place. Select the wheel matching the active interpreter and run:

```bash
python -m pip install --force-reinstall /path/to/the-rebuilt-wheel.whl
python -m pip check
python -c 'import fusemesh; print(fusemesh.__file__); print(fusemesh.__version__)'
```

Restart notebooks and processes that loaded the extension. Use `python -m pip`
to install into the same interpreter that runs your application. Only add
`--no-deps` when the new wheel's dependencies are already satisfied.

## CUDA is unavailable

`build_info()` does not run a GPU kernel; `smoke_test()` does. Check that the
NVIDIA driver is installed and that the process can see a compatible GPU.
`device=0` refers to the first visible device; `CUDA_VISIBLE_DEVICES` can change
which physical GPU it names. Source-build toolkit availability is separate from
runtime driver availability. See [installation requirements](installation.md).

Loading, saving, `simplify()`, and `compare_surfaces()` can run without a GPU. `reconstruct()` on a
nonempty scene has no CPU fallback.

## Reconstruction or simplification rejects a mesh

Reconstruction and simplification inputs must have finite coordinates, valid triangle indices, nondegenerate faces,
closed edges, consistent winding, and manifold vertex neighborhoods. Common
causes include holes, disconnected copies of shared vertices in STL triangle
soups, or material groups loaded as separate open geometries.

Inspect and repair the input explicitly before passing it to FUSE. The loader
does not weld or repair it. `validation="basic"` skips reconstruction topology
checks but does not define reliable solid semantics for defective meshes.
Simplification always performs its own input validation. Neither validator
detects geometric self-intersections.

## Reconstruction runs out of memory

Increase `voxel_size`, raise `max_memory_bytes` if the GPU has room, or reduce
the workload. The file example's equivalent is `--memory-mib`.
The budget covers explicit FUSE device buffers, not all process/GPU memory.
Host candidate construction has an additional budget-derived growth guard.

Avoid `return_field=True` for large sparse domains: it requests a dense host
array and full-domain sampling even though surface reconstruction is sparse.
There is no automatic coarsening or fallback to another backend.

## Thin walls or gaps disappear

The grid has uniform spacing. Features smaller than the sampling scale may
vanish or change topology. Try a smaller `voxel_size` and compare geometry and
memory usage. Simplification cannot recover detail already lost during
reconstruction. Exact contacts and unresolved cell interiors may still produce
nonmanifold output; see [numerical behavior](FUSE-design.md).

## Simplification returns too many triangles or raises an error

`target_faces` is a goal, not a guarantee. Inspect
`result.metadata["simplification"]` for accepted bounds, counts, and per-shell
attempts. On the default failure path, `SimplificationError` reports the final
reason instead of returning a rejected candidate.

- For a verification-budget failure, increase `max_checks` (file example:
  `--simplify-max-checks`). This permits more checks without relaxing quality.
- For geometric or volume failures, keep more triangles or deliberately choose
  looser tolerances appropriate to the application.
- To retain an unchanged shell when no candidate passes, use `on_failure="keep"`
  (`--simplify-on-failure keep`). Invalid input still raises.

Small error tolerances, sharp creases, and thin anisotropic solids can limit
reduction. Surface checks do not establish freedom from self-intersections or
preserve separation between nearby shells. See [simplification](simplification.md).

## Quality comparison rejects a mesh

Comparing one original surface with one output surface does not require closed
or manifold topology. Use `quality_demo.py --inputs original.off reconstructed.obj`
for this CPU-only mode; no reconstruction occurs.

Zero-area faces are rejected by default. Use `--degenerate-faces drop` in the demo
or quality CLI, or `degenerate_faces="drop"` in Python, to exclude them from both
surfaces. Inspect `report["cleaning"]` for per-mesh removal counts. Input files
are unchanged. Invalid coordinates/indices and meshes with no remaining valid
triangles still raise errors. Thin triangles with positive computed area are
not deliberately removed by a tolerance threshold.

For multiple original solids, install the `quality` extra. Boolean reference
construction still requires valid closed, oriented solids after cleaning.
Dropping zero-area faces does not repair an open or nonmanifold solid. See
[quality inputs and limitations](quality.md).
