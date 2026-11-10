# PhyLoopNO

Implementation of **PhyLoopNO: Looped Neural Operator with Physics-Gated Dense Expert Layers**, submitted to ICLR 2027. The model combines LinearNO spatial updates with dense pointwise experts and selective parameter sharing across repeated core executions.

This is a separate, paper-focused extraction of [CDLNO-w](https://github.com/hxh5159/CDLNO-w) at `0eece431e4a2019256b048b5f7f67da1283f4b64`. The original repository is unchanged. This project retains PhyLoopNO, its LinearNO and Transolver baselines, all eight benchmark pipelines, checkpoint/resume support, and the field, Q/K and expert-gate figure generators. Earlier experimental model families and development records are excluded. Details: [scope and source attribution](THIRD_PARTY_NOTICES.md), [data and evaluation](docs/BENCHMARKS.md), [paper configurations](docs/REPRODUCING.md), and [validation](docs/VALIDATION.md).

## Installation

Use Python 3.10 or newer and install a PyTorch build compatible with your hardware first. From the checkout root:

```bash
python -m pip install -e .
python -m pip install -e ".[test]"
```

The industrial tasks additionally require `python -m pip install -e ".[industrial]"` and the graph extension required by the selected PyG version (`pyg-lib>=0.6.0` for newer PyG, or `torch-cluster` for older versions), built for the selected PyTorch/CUDA combination. Use the [PyG installation instructions](https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html). Preserve a working PyTorch/PyG stack rather than reinstalling the old upstream requirements. CUDA is required by the retained six native Standard training entries; standalone models and synthetic tests also run on CPU.

## Train and evaluate

Prepare the datasets using [BENCHMARKS.md](docs/BENCHMARKS.md). Run these commands from the checkout root; `--dry-run` validates configuration and routing without reading data, constructing a model, or creating a run.

```bash
python scripts/run.py airfoil train --data-root /data --dry-run
python scripts/run.py airfoil train_eval --data-root /data --gpu 0
python scripts/run.py darcy train_eval --data-root /data --gpu 0
python scripts/run.py elasticity train_eval --data-root /data --gpu 0
python scripts/run.py pipe train_eval --data-root /data --gpu 0
python scripts/run.py ns train_eval --data-root /data --gpu 0
python scripts/run.py plasticity train_eval --data-root /data --gpu 0
```

`phyloopno` is the default model. Use `--model linearno` for the reproduced backbone or `--model transolver` for the preserved upstream baseline. New LinearNO/PhyLoopNO training reserves a unique run directory and prints its location. Use that exact directory for subsequent evaluation or continuation:

```bash
python scripts/run.py airfoil eval --experiment-dir /runs/exact-run --data-root /data
python scripts/run.py airfoil resume --experiment-dir /runs/exact-run --data-root /data
```

Resume restores optimizer, scheduler, RNG, normalizers and data-loader state at a committed epoch. It retains the original total epoch plan. Native Transolver retains its upstream `--save_name` checkpoint convention and seed 0; it does not gain LinearNO/PhyLoopNO resume archives.

The paper does not specify industrial expert counts. Select them explicitly from the experiment configuration; the following `3` is an example, not a claim about the paper's industrial runs:

```bash
python scripts/run.py airfrans train_eval --experts 3 --data-path /data/AirfRANS/Dataset
python scripts/run.py car train_eval --experts 3 --data-dir /data/mlcfd_data/training_data --cache-dir /data/mlcfd_data/preprocessed_data
```

## Paper model and ablations

Eight-layer models use `1 + 3 * 2 + 1`: one prefix, three core positions visited twice, and one suffix. At the same core position, input/value/output transformations and expert weights are shared; Q/K, active temperatures, gates and LayerNorm are visit-specific. The core expert residual is scaled by `1/R`.

| Task | Channels | Latents/head | Expert width | Experts |
|---|---:|---:|---:|---:|
| Airfoil | 128 | 64 | 128 | 4 |
| Pipe | 128 | 64 | 128 | 3 |
| Plasticity | 128 | 64 | 128 | 4 |
| Darcy | 128 | 64 | 128 | 1 |
| Navier-Stokes | 256 | 32 | 512 | 4 |
| Elasticity | 128 | 64 | 128 | 4 |

```bash
python scripts/run.py airfoil train --ablation full_share --data-root /data
python scripts/run.py airfoil train --ablation selective_share --data-root /data
python scripts/run.py airfoil train --ablation shared_slicing --data-root /data
python scripts/run.py airfoil train --ablation wide_ffn --data-root /data
python scripts/run.py airfoil train --depth 40 --data-root /data
```

These selectors express the manuscript's comparisons; [REPRODUCING.md](docs/REPRODUCING.md) defines their exact ownership and width. New sharing selectors were added during extraction because the source V5 did not expose both sharing ablations. They have synthetic verification; no new benchmark accuracy is claimed.

## Model API and figures

```python
from phyloopno import build_model
model = build_model("elasticity")
# prediction = model(positions, None)  # [B,N,2] -> [B,N,1]
```

`iclr_vis/weight/` generates field and slicing/deslicing plots; `iclr_vis/ffn/` generates dense expert gate maps. Both require an explicit run, saved normalizers and the original dataset. See [figure commands](docs/REPRODUCING.md#figures).

Run `python -m pytest tests -q` for synthetic model, routing, checkpoint, continuation and CLI checks. Full datasets, trained weights and benchmark measurements are not bundled. The extraction has not retrained the submission results. Existing development-repository checkpoints require an explicit migration; old sidecars are not silently treated as the new schema.

The internal `cdlno` and `linearno_loop/v5` package names remain to preserve task imports and tensor-key structure; they expose only the retained paper model and baseline runtime.
