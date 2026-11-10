# Paper configurations and reproducibility

## Main model

The authoritative method is the submitted PhyLoopNO manuscript, not the original repository's introductory CDLNO README. Main settings use `qk_sharing=independent`, `core_norm_mode=visit_independent`, dense experts, and operator/expert residual scales `1` and `1/R` in the repeated core. Prefix and suffix expert residuals use scale `1`. Q normalizes over latent tokens, K over spatial points. There is no latent self-attention in this model.

Standard configurations use 8 heads, AdamW, learning rate 0.001 and 500 epochs. Elasticity uses CosineAnnealingLR; the other Standard tasks use OneCycleLR. Task-specific dimensions and expert counts are in the root README. Industrial profiles use 256 channels, 32 latents/head, width-512 experts and 8 heads; AirfRANS uses 400 epochs and Car 200, Adam and batch size 1. Expert count is explicit for industrial tasks because the manuscript does not specify it.

## Ablation selectors

| `--ablation` | Experts/width | Across visits at each core position |
|---|---|---|
| `main` | Task-specific E and native F | Shared input/value/output/experts; independent Q/K, active temperatures, LN and gate |
| `full_share` | One native-width expert | Shared Q/K, temperatures, LN, operator and expert; the inert one-expert gate is also shared |
| `selective_share` | One native-width expert | Main sharing pattern; isolates the single-FFN comparison in Figure 2/Table 8 |
| `shared_slicing` | Main E/F | Shares each Q and K projection and its active temperature; LN and gate stay independent, as in Table 3 |
| `wide_ffn` | One expert of width E*F | Main selective sharing; equivalent to a single width-EF FFN because its one-expert softmax is identically one |

Q and K remain different mappings in every selector. Copying initial Q/K values is not parameter sharing. The default model copies corresponding values across visits into distinct storage; shared variants reference the same module/parameter objects. Different core positions always own different transformations.

`wide_ffn` derives F from the selected profile and explicit native hidden-width/FFN-ratio overrides. `--experts` then describes the original expert count to match; `--expert-width` describes the original per-expert width. E=1 on Darcy makes the wide-FFN replacement identical.

The Airfoil parameter comparison uses `--experts 3`, separately from the four-expert main result. Its full model has 1,457,561 trainable parameters versus 1,765,889 for LinearNO (17.46% fewer). The main four-expert model has 1,623,713 parameters. Parameter savings do not imply lower FLOPs or latency.

## Depth scaling

| Executed depth | Prefix | Stored core positions | Repeats | Suffix |
|---:|---:|---:|---:|---:|
| 8 | 1 | 3 | 2 | 1 |
| 16 | 2 | 6 | 2 | 2 |
| 24 | 2 | 10 | 2 | 2 |
| 32 | 2 | 14 | 2 | 2 |
| 40 | 2 | 18 | 2 | 2 |

For example, `python scripts/run.py elasticity train --depth 24 --data-root /data`. Use `--model linearno` or `--model transolver` with the same `--depth` to scale the baseline. Each depth must be trained independently; depth does not mean repeated physical-time prediction.

`python scripts/depth_sweep.py airfoil --data-root /data` prints the full three-model, five-depth command matrix. Add `--execute` to run it sequentially. Transolver uses its original seed 0 and explicit distinct checkpoint names; the other two families use their own immutable run directories.

## Objectives and metrics

The retained profiles distinguish native model implementation, training objective and evaluation specification. `paper_table8_on_release_model` is the default. `official_release` and `transolver_matched` remain explicit comparison profiles, not guarantees of identical published errors or equal compute.

Four static Standard tasks retain the source project's batch-mean relative-L2 objective (official LinearNO used batch-sum, a 1/B scale difference). Darcy also retains the 0.1 derivative term. NS and Plasticity keep their native batch-sum/time reduction. NS feeds ground truth during training and predictions during evaluation; Plasticity independently queries twenty times, with twenty optimizer updates and one scheduler update per outer batch.

AirfRANS training uses normalized four-channel volume MSE + surface MSE; Car training uses all-point normalized velocity MSE + 0.5*surface-pressure MSE. Physical pressure/velocity relative L2, aerodynamic coefficient errors and Spearman correlations are separate evaluation fields. These objectives match the source project's explicitly selected release-code contracts, whose industrial paper descriptions differ. Do not relabel MSE as relative L2.

## Figures

```bash
python iclr_vis/weight/airfoil_states.py --run-dir /runs/airfoil --data-path /data/fno/airfoil/naca
python iclr_vis/weight/task_states.py pipe --run-dir /runs/pipe --data-path /data/fno/pipe
python iclr_vis/ffn/ffn_states.py elasticity --run-dir /runs/elasticity --data-path /data/fno
```

The generators preserve inference predictions and use saved normalization. Q/K atlas outputs distinguish absolute weights from per-state spatial-maximum normalization; the manuscript atlases use the latter. Expert figures use a common 0-1 gate-weight scale. Expert identities are local to each core position. Temporary observation hooks do not replace model outputs. Generated figures and development audit evidence are excluded from the source release.

## Checkpoints

Training creates an immutable architecture sidecar and committed checkpoint/weight/metadata pairs with checksums. Evaluation and continuation validate metadata before loading tensors. Continuation restores complete training state at saved epoch boundaries and rejects changes to architecture or the original scheduler plan. The extracted schema is revision 7 and records Q/K sharing, paper topology and expert count explicitly; older development checkpoints have no implicit compatibility fallback.
