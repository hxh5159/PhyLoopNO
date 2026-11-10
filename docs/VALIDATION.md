# Extraction validation

Validation date: 2026-11-10. Source: CDLNO-w `0eece431e4a2019256b048b5f7f67da1283f4b64`. The original checkout remained clean and its remote was unchanged. The new project has an independent Git history.

## Executed checks

| Check | Result |
|---|---|
| Main synthetic suite, `python -m pytest tests -q` | 83 passed; 110 subtests passed |
| Field/Q/K generators, `python -m pytest iclr_vis/weight -q` | 14 passed; 15 subtests passed |
| Expert generators, `python -m pytest iclr_vis/ffn -q` | 8 passed; 8 subtests passed |
| Source V5 vs extracted default core | 12 configurations, initialization/state/parameters/forward/input and parameter gradients bitwise equal |
| Public train routing | 8 tasks * 3 models, all 24 plans parsed |
| Actual saved-sidecar eval/resume routing | 8 tasks * 2 actions, all 16 synthetic-metadata plans parsed |
| Wheel build | Passed; 99 archive entries |
| Wheel tested outside checkout using isolated Python | Model forward and all 24 train plans passed using packaged benchmark files |
| Task mathematical branch preservation | 18 source-branch AST comparisons equal |

The source/core comparison used the untouched source V5 operator/core loaded separately, six native attention variants, both LayerNorm modes, CPU float64 and seed 731. The new default Q/K-independent model is unchanged numerically. New full-sharing and Q/K-sharing ablations separately test actual parameter object ownership and checkpoint reconstruction. The one-expert gate and expert-only residual scale are checked explicitly.

Synthetic training verification covers complete Standard optimizer/scheduler/RNG checkpoint continuation with exact next-step weights/loss/RNG, standalone Car training using its native objective, eight-task model forward/backward, all six attention variants against explicit dense factorization and input gradients, corrupted checkpoint rejection, and portable writer-lock contention. Visualization checks include fresh-process exports, saved normalizers, prediction-fed NS rollout, hook transparency, and bounded CUDA capture where available.

## Environment and limits

Tests used Python 3.10.20, PyTorch 2.12.1+cu130 with CUDA available, NumPy 2.2.6, SciPy 1.15.3 and PyG 2.8.1 on Windows. Two existing torch.jit.script deprecation warnings appeared in the Car runtime test. Figure tests use an explicitly writable temporary/cache directory in this managed environment; shell wrappers are optional and tests invoke Python directly.

No real PDE/industrial datasets were loaded, no submission benchmark was retrained, and no manuscript error value was independently remeasured. Full graph construction requires an appropriate PyG extension (`pyg-lib>=0.6.0` for the tested newer PyG, or torch-cluster on older PyG). That extension was not installed for real-data acceptance here. Synthetic PyG interfaces do not establish raw VTK/mesh dataset completeness or real aerodynamic accuracy.

The wheel excludes datasets, weights, generated figures and historical audit files. Current source files and their original-file hashes are listed in [CLEANUP_MANIFEST.json](CLEANUP_MANIFEST.json). That manifest omits itself and this report to avoid self-reference.

## Deliberate release changes

- Native benchmark data/loss/scheduler/metric calculations are retained; unrelated model dispatch is removed.
- V5 defaults match the final paper's task experts and eight-layer execution schedule. Industrial expert count is explicitly required because it is not specified in the manuscript.
- Full-share and shared-slicing selectors, equal-total-width FFN selection and the 8/16/24/32/40 depth schedule are explicit and recorded.
- Source provenance fingerprints the retained release, without requiring deleted models, historical Git commits or developer snapshots.
- Checkpoint writers use portable exclusive locks and POSIX relative paths; run IDs are short enough for ordinary Windows paths.
- Native package names are retained internally; the public project/API/command is PhyLoopNO. Checkpoint schema 7 does not silently load prior development sidecars.
