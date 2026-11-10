#!/usr/bin/env python3
"""Read-only V5 Darcy/Elasticity/NS/Pipe final-attention spatial atlases.

Uses native saved normalizers and prediction-fed NS evaluation, without importing
experiment entry points. Plotting and transparent Q/K capture reuse Airfoil's
reviewed utilities. See TASKS.md for paths, data semantics and paper captions.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import sys
import uuid

import airfoil_states as vis

TASKS = ("darcy", "elasticity", "ns", "pipe")
DEFAULT_RUNS = {}
FILES = {
    "darcy": ("piececonst_r421_N1024_smooth1.mat", "piececonst_r421_N1024_smooth2.mat"),
    "elasticity": ("elasticity/Meshes/Random_UnitCell_XY_10.npy", "elasticity/Meshes/Random_UnitCell_sigma_10.npy"),
    "ns": ("NavierStokes_V1e-5_N1200_T20/NavierStokes_V1e-5_N1200_T20.mat",),
    "pipe": ("Pipe_X.npy", "Pipe_Y.npy", "Pipe_Q.npy"),
}
FIELDS = {"darcy": "solution", "elasticity": "stress", "ns": "vorticity", "pipe": "velocity"}
NORMALIZERS = {"darcy": {"input", "output"}, "elasticity": {"output"}, "ns": set(), "pipe": {"input", "output"}}
UNIT_ALGORITHM = "UnitTransformer: mean(dim=(0,1)); std(dim=(0,1))+1e-8"


def parser(task):
    root = vis.REPO / "output" / task / "partial_share_feature_gate_v5"
    default_run = root / DEFAULT_RUNS[task] if task in DEFAULT_RUNS else root
    fno = Path(os.environ.get("CDLNO_FNO_ROOT", str(Path(os.environ.get("CDLNO_DATA_ROOT", str(vis.DATA_ROOT))) / "fno")))
    data = Path(os.environ.get(f"CDLNO_{task.upper()}_ROOT", str(fno / "pipe" if task == "pipe" else fno)))
    p = argparse.ArgumentParser(description=f"{task}: {__doc__}", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--run-dir", type=Path, required=True, help="exact run containing architecture.json")
    p.add_argument("--list-runs", action="store_true", help="list sidecars under --run-dir; no tensors/data or new outputs")
    p.add_argument("--data-path", type=Path, default=data, help="same data root as the native task entry")
    p.add_argument("--checkpoint", default="final", help="final, latest or epoch_XXXX")
    p.add_argument("--sample-index", type=int, default=0, help="index within the task's 200 test samples")
    p.add_argument("--forecast-step", type=int, help="NS only: future step 1..10 to visualize; default 10")
    p.add_argument("--head", default="0", help="zero-based head or all; never averages heads")
    p.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    p.add_argument("--gpu", type=int, default=0, help="index within CUDA_VISIBLE_DEVICES")
    p.add_argument("--output-dir", type=Path, help="new directory; existing paths are never overwritten")
    p.add_argument("--paper", choices=tuple(vis.PAPER_WIDTHS), default="iclr")
    p.add_argument("--width", type=float, help="final width in inches; ICLR 5.5, ICML 6.75")
    p.add_argument("--dpi", type=int, default=400)
    p.add_argument("--columns", type=int, default=8, help="64 states: 8x8; 32 states: 4x8")
    p.add_argument("--cmap", choices=("viridis", "cividis", "coolwarm"), default="viridis")
    p.add_argument("--annotate", action="store_true", help="browsing titles; omit for paper figures")
    vis.add_typography_arguments(p)
    p.add_argument("--individual", action="store_true", help="also export each Q/K state with peak normalization")
    p.add_argument("--preview", action="store_true", help="metadata only; no torch/data/tensor/output creation")
    return p


def list_runs(directory, task):
    from linearno_loop.v5.schema import read_metadata
    directory = Path(directory).expanduser().resolve()
    if not directory.is_dir():
        raise FileNotFoundError(f"Run directory does not exist: {directory}")
    paths = [directory / "architecture.json"] if (directory / "architecture.json").is_file() else sorted(directory.glob("*/architecture.json"))
    rows = []
    for path in paths:
        try:
            m = read_metadata(path); c = m["resolved_config"]
            if c["task"] != task:
                continue
            rows.append(dict(run_dir=str(path.parent), config_hash=c["config_hash"], M=c["actual_M"],
                expert_count=c["expert_count"], topology=c["topology_preset"],
                final_pointer_exists=(path.parent / "checkpoints/final.json").is_file(),
                note="metadata only; weights not verified"))
        except (ValueError, OSError, KeyError) as exc:
            rows.append(dict(run_dir=str(path.parent), error=f"{type(exc).__name__}: {exc}"))
    return rows


def validate_request(task, args, metadata):
    c = metadata["resolved_config"]; spec = c["model_spec"]["constructor_kwargs"]
    if c["architecture"] != "partial_share_feature_gate_v5" or c["task"] != task:
        raise ValueError(f"Expected a V5 {task} checkpoint")
    if spec["space_dim"] != 2 or spec["fun_dim"] != {"darcy": 1, "ns": 10}.get(task, 0) or spec["out_dim"] != 1 or spec["time_input"]:
        raise ValueError("Saved model has an unsupported task input/output contract")
    protocol = metadata["data_spec"]["protocol"]
    if tuple(name.split(":")[0] for name in protocol["files"]) != FILES[task]:
        raise ValueError("Saved task data filenames differ; refusing to reinterpret them")
    expected_split = c["profile_spec"]["values"]["data"]["split"]
    if metadata["data_spec"]["split"] != expected_split:
        raise ValueError("Saved split differs from the task profile")
    if metadata["data_spec"]["runtime"].get("ntest", 200) != 200:
        raise ValueError("Only the native 200-sample test split is supported")
    if set(metadata["normalizer_spec"]["records"]) != NORMALIZERS[task]:
        raise ValueError("Missing/unexpected saved normalizers; fitting replacements is forbidden")
    if not 0 <= args.sample_index < 200:
        raise ValueError("--sample-index must be in 0..199")
    heads = list(range(spec["heads"])) if args.head == "all" else [int(args.head)]
    if any(h < 0 or h >= spec["heads"] for h in heads):
        raise ValueError(f"--head must be in 0..{spec['heads'] - 1}, or all")
    if task == "ns":
        args.forecast_step = 10 if args.forecast_step is None else args.forecast_step
        if not 1 <= args.forecast_step <= 10:
            raise ValueError("--forecast-step must be in 1..10")
    elif args.forecast_step is not None:
        raise ValueError("--forecast-step applies only to NS")
    args.width = vis.PAPER_WIDTHS[args.paper] if args.width is None else args.width
    vis.validate_typography(args)
    if args.gpu < 0 or args.dpi < 72 or not math.isfinite(args.width) or args.width < 4:
        raise ValueError("Require gpu>=0, dpi>=72, finite width>=4")
    if not 1 <= args.columns <= c["actual_M"]:
        raise ValueError("--columns must be between 1 and the checkpoint's actual M")
    if (args.width - .11 - (args.columns - 1) * .025) / args.columns * 72 < 16:
        raise ValueError("Too many columns for visible spatial detail; reduce --columns or increase final --width")
    return heads


def verify_files(task, root, metadata):
    expected = metadata["data_spec"]["checksums"]; actual = {}
    for name in FILES[task]:
        path = root / name
        if not path.is_file():
            raise FileNotFoundError(f"Missing {task} data file: {path}; --data-path uses the native task root")
        if name not in expected:
            raise ValueError(f"No saved data checksum for {name}")
        print(f"Checking dataset SHA-256: {name}", flush=True)
        actual[name] = vis.sha256(path)
        if actual[name] != expected[name]:
            raise ValueError(f"Dataset checksum differs from training: {name}")
    return actual


def saved_normalizers(task, metadata):
    import torch
    from cdlno.linearno.schema import restore_numerical_state
    values = {}
    for name in sorted(NORMALIZERS[task]):
        record = metadata["normalizer_spec"]["records"][name]
        if record["algorithm"] != UNIT_ALGORITHM or record["fit_split"] != "train only" or set(record["states"]) != {"mean", "std"}:
            raise ValueError(f"Unsupported saved {name} normalizer")
        state = {k: torch.as_tensor(restore_numerical_state(v)).cpu() for k, v in record["states"].items()}
        shape = (1, 1, 2) if task == "pipe" and name == "input" else (1, 1)
        if any(tuple(v.shape) != shape or not v.is_floating_point() or not torch.isfinite(v).all() for v in state.values()) or not (state["std"] > 0).all():
            raise ValueError(f"Saved {name} normalizer shape/dtype/std is invalid")
        values[name] = state
    return values


def transform(value, state, *, inverse=False):
    # std already includes the native 1e-8; do not add another epsilon or re-fit.
    mean, std = (state[k].to(value.device) for k in ("mean", "std"))
    return value * std + mean if inverse else (value - mean) / std


def load_sample(task, root, metadata, index):
    import numpy as np
    import torch
    spec = metadata["resolved_config"]["model_spec"]["constructor_kwargs"]
    h, w = spec["grid_height"], spec["grid_width"]
    norms = saved_normalizers(task, metadata)
    fx = None; extras = {}; raw = index
    if task in ("darcy", "ns"):
        from scipy.io import loadmat
        # scipy MAT-v5 reading loads the requested variables, not training fields.
        name = FILES[task][-1]
        data = loadmat(root / name, variable_names=["coeff", "sol"] if task == "darcy" else ["u"])
        if task == "darcy":
            a, u = data["coeff"], data["sol"]
            if a.ndim != 3 or a.shape != u.shape or a.shape[0] < 200:
                raise ValueError("Darcy expects coeff/sol [samples,H,W] in smooth2, at least 200 samples")
            if metadata["data_spec"]["scope"] == "real" and a.shape[1:] != (421, 421):
                raise ValueError("Native Darcy expects the 421x421 grid before stride5")
            coefficient = np.array(a[index, ::5, ::5][:h, :w], dtype=np.float32, copy=True)
            target = np.array(u[index, ::5, ::5][:h, :w], copy=True)
            if coefficient.shape != (h, w) or a[:, ::5, ::5].shape[1:] != (h, w):
                raise ValueError("Saved Darcy grid differs from stride5 data; refusing resampling")
            fx = transform(torch.from_numpy(coefficient.reshape(1, -1)), norms["input"]).unsqueeze(-1)
            extras["coefficient"] = coefficient
        else:
            u = data["u"]
            if u.ndim != 4 or u.shape[0] < 1200 or u.shape[1:3] != (h, w) or u.shape[-1] < 20:
                raise ValueError("NS expects u [>=1200,H,W,>=20], grid matching saved model")
            raw = u.shape[0] - 200 + index  # Native last200, NOT assumed first1200.
            sample = np.array(u[raw, :, :, :20], copy=True)
            if sample.dtype != np.float32:
                raise ValueError("Native NS input must be float32 for this FP32 model; no implicit dtype conversion")
            fx = torch.from_numpy(sample[..., :10].copy().reshape(1, -1, 10))
            target = sample[..., 10:].reshape(-1, 10)
            extras["initial_history"] = sample[..., :10].copy()
        x, y = np.meshgrid(np.linspace(0, 1, w), np.linspace(0, 1, h))
        x, y = x.astype(np.float32), y.astype(np.float32)
        positions = torch.from_numpy(np.stack((x, y), -1).reshape(1, -1, 2))
    elif task == "pipe":
        xx, yy, u = [np.load(root / name, mmap_mode="r", allow_pickle=False) for name in FILES[task]]
        if xx.ndim != 3 or xx.shape != yy.shape or xx.shape[0] < 1200 or xx.shape[1:] != (h, w) or u.ndim != 4 or u.shape[0] != xx.shape[0] or u.shape[1] < 1 or u.shape[2:] != (h, w):
            raise ValueError("Pipe expects X/Y [>=1200,H,W], Q [samples,channels,H,W], native grid")
        raw = 1000 + index  # The entry first truncates to 1200, then takes last200.
        x, y = [np.array(a[raw], dtype=np.float32, copy=True) for a in (xx, yy)]
        target = np.array(u[raw, 0], dtype=np.float32, copy=True)
        positions = transform(torch.from_numpy(np.stack((x, y), -1).reshape(1, -1, 2)), norms["input"])
    else:
        xy, sigma = [np.load(root / name, mmap_mode="r", allow_pickle=False) for name in FILES[task]]
        if xy.ndim != 3 or xy.shape[1] != 2 or xy.shape[2] < 1200 or sigma.shape != (xy.shape[0], xy.shape[2]):
            raise ValueError("Elasticity expects XY [N,2,samples], sigma [N,samples], at least 1200 samples")
        raw = xy.shape[2] - 200 + index
        nodes = np.array(xy[:, :, raw], dtype=np.float32, copy=True)
        x, y = nodes[:, 0].copy(), nodes[:, 1].copy()
        target = np.array(sigma[:, raw], dtype=np.float32, copy=True)
        positions = torch.from_numpy(nodes[None])
    if not all(np.isfinite(a).all() for a in (x, y, target)) or not torch.isfinite(positions).all() or (fx is not None and not torch.isfinite(fx).all()):
        raise ValueError("Non-finite sample or normalized input")
    return dict(x=x, y=y, target=target, positions=positions, fx=fx, normalizers=norms,
                raw_sample_index=raw, source_file=FILES[task][-1], extras=extras)


def infer(task, model, sample, device, forecast_step):
    import numpy as np
    import torch
    positions = sample["positions"].to(device)
    fx = sample["fx"].to(device) if sample["fx"] is not None else None
    extras = dict(sample["extras"])
    if task == "ns":
        predictions = []
        with torch.inference_mode():
            for step in range(1, 11):
                if step == forecast_step:
                    extras["selected_input_history"] = fx[0].cpu().numpy().copy()
                    q, k, prediction, diagnostics = vis.capture_last_attention(model, positions, fx=fx)
                    out = torch.from_numpy(prediction.copy()).to(device).reshape(1, -1, 1)
                else:
                    out = model(positions, fx=fx)
                if not torch.isfinite(out).all():
                    raise ValueError(f"NS non-finite prediction at future step {step}")
                predictions.append(out[0, :, 0].cpu().numpy().copy())
                fx = torch.cat((fx[..., 1:], out), dim=-1)
        rollout = np.stack(predictions, -1)
        target = sample["target"][:, forecast_step - 1]
        extras.update(predicted_rollout=rollout, target_rollout=sample["target"],
                      future_steps=np.arange(1, 11), source_frame_indices=np.arange(10, 20))
        prediction = rollout[:, forecast_step - 1]
    else:
        q, k, encoded, diagnostics = vis.capture_last_attention(model, positions, fx=fx)
        prediction = transform(torch.from_numpy(encoded[None]), sample["normalizers"]["output"], inverse=True)[0].numpy()
        extras["encoded_prediction"] = encoded
        if fx is not None:
            extras["model_input_field"] = fx[0].cpu().numpy().copy()
        target = sample["target"].ravel()
    extras["model_input_positions"] = positions[0].cpu().numpy().copy()
    if not np.isfinite(prediction).all():
        raise ValueError("Non-finite decoded prediction")
    return q, k, np.asarray(target).ravel(), prediction, diagnostics, extras


def point_paint(ax, nodes, values, bounds, *, cmap, norm):
    """Only measured nodes: no invented connectivity across Elasticity's hole."""
    panel_points = ax.get_position().width * ax.figure.get_figwidth() * 72
    size = max(.08, .65 * panel_points**2 / len(nodes))
    artist = ax.scatter(nodes[:, 0], nodes[:, 1], c=values, s=size, marker="o",
                        cmap=cmap, norm=norm, edgecolors="none", linewidths=0, rasterized=True)
    ax.set_xlim(*bounds[0]); ax.set_ylim(*bounds[1])
    ax.set_aspect("equal", adjustable="box"); ax.set_axis_off()
    return artist


def export_plots(task, output, sample, q, k, target, prediction, heads, args, synthetic):
    import numpy as np
    import matplotlib
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize
    x, y = sample["x"], sample["y"]
    mesh = np.stack((x, y), -1) if task == "elasticity" else vis.structured_mesh(x, y)
    painter = point_paint if task == "elasticity" else vis.paint
    bounds = ((float(x.min()), float(x.max())), (float(y.min()), float(y.max())))
    if any(hi <= lo for lo, hi in bounds):
        raise ValueError("Cannot plot a degenerate spatial domain")
    name = "NS" if task == "ns" else task.capitalize()
    prefix = "SYNTHETIC CHECK | " if synthetic else ""
    temporal = f" | step {args.forecast_step}" if task == "ns" else ""
    figures = {}
    for head in heads:
        for kind, weights in (("Q", q[head]), ("K", k[head])):
            for normalization in ("shared", "peak"):
                stem = f"{kind.lower()}_head{head:02d}_full_{normalization}"
                title = (prefix + f"{name} | {kind} | head {head}" + temporal) if (args.annotate or synthetic) else ""
                print("Rendering " + stem, flush=True)
                info = vis.atlas(mesh, weights, kind=kind, normalization=normalization, bounds=bounds,
                    columns=args.columns, title=title, path=output / stem, args=args, painter=painter)
                figures[stem] = dict(**info, bounds=bounds, head=head, kind=kind, normalization=normalization)
            if args.individual:
                directory = output / f"{kind.lower()}_head{head:02d}_individual_peak"; directory.mkdir()
                values, _, _ = vis.display_values(weights, kind, "peak")
                for state in range(weights.shape[1]):
                    with plt.rc_context(vis.style(args)):
                        fig, ax = plt.subplots(figsize=(3., 3.), layout="constrained")
                        painter(ax, mesh, values[:, state], bounds, cmap=args.cmap, norm=Normalize(0, 1))
                        if synthetic:
                            ax.set_title("SYNTHETIC CHECK")
                        elif args.annotate:
                            ax.set_title(f"{kind} weights | peak normalized")
                        vis.save_figure(fig, directory / f"state_{state:02d}", args.dpi); plt.close(fig)
    vis.field_reference(mesh, target, prediction, bounds, output / "field_reference", args,
        (prefix + name + temporal) if (args.annotate or synthetic) else "", field_name=FIELDS[task], painter=painter)
    return figures


def write_caption(task, output, args, heads, rank, synthetic):
    caption = ("SYNTHETIC CHECK; random model and synthetic data. " if synthetic else "")
    caption += (f"Final-layer reconstruction routing weights on {task} "
        f"(test sample {args.sample_index}, head {heads[0]}). All {rank} states retain their original row-major order without panel numbers. "
        r"Panels show $Q_{i,m}/\max_i Q_{i,m}$ on a common 0--1 scale. "
        "Normalization uses the complete spatial domain and compares patterns, not absolute latent strengths. ")
    caption += ("Only original mesh nodes are drawn; no connectivity or filled surface is inferred. " if task == "elasticity"
                else "Colors interpolate within original structured grid cells. ")
    if task == "ns":
        caption += f"Weights are captured at future step {args.forecast_step} of a prediction-fed ten-step rollout. "
    caption += "These are routing weights, not independent physical-field predictions."
    env = "figure*" if args.paper == "icml" else "figure"
    (output / "paper_figure.tex").write_text(
        "% Official conference template + graphicx; keep the template's caption styling.\n"
        "% ICLR official preamble includes times; caption normally 10 TeX pt.\n"
        "% ICML official style loads Times; caption 9 TeX pt. Do not override caption font/size.\n"
        f"% Labels {args.font_size:g}pt, ticks {args.tick_font_size:g}pt at {args.width:g}in. Regenerate instead of shrinking.\n"
        f"\\begin{{{env}}}[t]\n  \\centering\n"
        f"  \\includegraphics[width={args.width:g}in]{{q_head{heads[0]:02d}_full_peak.pdf}}\n"
        f"  \\caption{{{caption}}}\n  \\label{{fig:v5-{task}-routing}}\n\\end{{{env}}}\n", encoding="utf-8")
    (output / "caption.txt").write_text(caption + "\n\nOther files: Q shared shows original reconstruction probabilities; "
        "K shared shows N*K compression weights (uniform=1). K peak divides each state's weights by its own spatial maximum. "
        "Q normalizes across latent states; K normalizes across points. Heads are never averaged.\n", encoding="utf-8")


def relative_l2(prediction, target):
    import numpy as np
    p, t = np.asarray(prediction, dtype=np.float64), np.asarray(target, dtype=np.float64)
    denominator = float(np.linalg.norm(t.ravel()))
    return float(np.linalg.norm((p - t).ravel()) / denominator) if denominator > 0 else None


def main(argv=None):
    tokens = list(sys.argv[1:] if argv is None else argv)
    if not tokens or tokens[0] not in TASKS:
        raise SystemExit("Usage: task_states.py {darcy,elasticity,ns,pipe} [options]")
    task = tokens.pop(0); p = parser(task); args = p.parse_args(tokens)
    args.run_dir = args.run_dir.expanduser().resolve(); args.data_path = args.data_path.expanduser().resolve()
    if args.list_runs:
        print(json.dumps(list_runs(args.run_dir, task), indent=2, ensure_ascii=False)); return 0
    if not (args.run_dir / "architecture.json").is_file():
        p.error(f"--run-dir must identify one exact experiment containing architecture.json: {args.run_dir}. "
                "Use --list-runs to inspect the parent, then pass a specific child with --run-dir; no automatic selection.")
    from linearno_loop.v5.schema import read_metadata
    metadata = read_metadata(args.run_dir / "architecture.json")  # before Torch/data/tensor imports
    heads = validate_request(task, args, metadata)
    cfg = metadata["resolved_config"]
    summary = dict(task=task, run_dir=str(args.run_dir), data_path=str(args.data_path),
        config_hash=cfg["config_hash"], checkpoint=args.checkpoint, M=cfg["actual_M"], heads=heads,
        expert_count=cfg["expert_count"], topology=cfg["topology_preset"],
        core_norm_mode=cfg["core_norm_mode"], sample_index=args.sample_index,
        forecast_step=args.forecast_step, metadata_scope=metadata["data_spec"]["scope"],
        attention_path=f"loop.suffix.{cfg['suffix_blocks'] - 1}.Attn", visit_index=0)
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)
    if args.preview:
        return 0
    font_info = vis.typography(args)
    print(f"Figure font: {font_info['family']}; labels {args.font_size:g} pt, ticks {args.tick_font_size:g} pt; no panel numbers", flush=True)
    if args.output_dir is not None and args.output_dir.expanduser().exists():
        raise FileExistsError("--output-dir already exists; choose a new directory")
    checksums = verify_files(task, args.data_path, metadata)
    import numpy as np
    import torch
    from cdlno.linearno_loop.v5.checkpoint import load_model
    if args.device == "cuda" and (not torch.cuda.is_available() or args.gpu >= torch.cuda.device_count()):
        raise RuntimeError("Requested GPU unavailable; use --device cpu explicitly")
    device = torch.device(f"cuda:{args.gpu}" if args.device == "cuda" else "cpu")
    print("Loading strict checkpoint pair and saved normalization...", flush=True)
    model, loaded = load_model(args.run_dir, selector=args.checkpoint, map_location=device)
    validate_request(task, args, loaded); model.eval()
    sample = load_sample(task, args.data_path, loaded, args.sample_index)
    q, k, target, prediction, diagnostics, extras = infer(task, model, sample, device, args.forecast_step)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    output = (args.output_dir.expanduser().resolve() if args.output_dir else
              Path(__file__).resolve().parent / "outputs" / f"{task}_{cfg['config_hash'][:12]}_{stamp}_{uuid.uuid4().hex[:8]}")
    output.mkdir(parents=True, exist_ok=False)
    metrics = dict(selected_sample_relative_l2=relative_l2(prediction, target))
    if task == "ns":
        steps = [relative_l2(extras["predicted_rollout"][:, i], extras["target_rollout"][:, i]) for i in range(10)]
        metrics.update(rollout_full_relative_l2=relative_l2(extras["predicted_rollout"], extras["target_rollout"]),
                       per_step_relative_l2=steps, mean_step_relative_l2=sum(steps) / 10 if all(v is not None for v in steps) else None)
    report = dict(**summary, status="rendering", epoch=loaded["resume_state"]["epoch"],
        checkpoint_role=loaded["resume_state"]["checkpoint_role"], checkpoint_metadata_hash=loaded["metadata_hash"],
        raw_sample_index=sample["raw_sample_index"], source_file=sample["source_file"], data_sha256=checksums,
        inference=dict(device=str(device), torch=torch.__version__, python=sys.version,
            cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"), dtype="float32", amp=False,
            diagnostic=diagnostics, protocol="prediction feedback; 10 steps" if task == "ns" else "native static eval; saved normalization"),
        metrics=metrics, metrics_scope="one test sample only; not the full-test metric",
        rendering=dict(paper=args.paper, width_inches=args.width, typography=font_info,
            dpi=args.dpi, cmap=args.cmap, columns=args.columns,
            geometry="original nodes only; no inferred triangles" if task == "elasticity" else "native structured cell triangles; Gouraud colors",
            head_aggregation="none", coordinates="raw physical coordinates, not encoded model coordinates",
            peak="divide by each latent's spatial maximum; no minimum subtraction",
            shared="Q unchanged; K multiplied by N; one shared scale per atlas"),
        source_sha256={str(f.relative_to(vis.REPO)): vis.sha256(f) for f in (
            Path(__file__), Path(vis.__file__), vis.REPO / "cdlno/linearno_loop/v5/operator.py",
            vis.REPO / "cdlno/linearno_loop/v5/standard.py")})
    np.savez_compressed(output / "routing_weights.npz", x=sample["x"], y=sample["y"],
        target=target.reshape(sample["x"].shape), prediction=prediction.reshape(sample["x"].shape),
        Q=q[heads], K=k[heads], head_indices=np.asarray(heads), latent_indices=np.arange(cfg["actual_M"]),
        test_index=np.asarray(args.sample_index), raw_sample_index=np.asarray(sample["raw_sample_index"]), **extras)
    try:
        synthetic = loaded["data_spec"]["scope"] != "real"
        report["figures"] = export_plots(task, output, sample, q, k, target, prediction, heads, args, synthetic)
        write_caption(task, output, args, heads, cfg["actual_M"], synthetic)
        report["status"] = "completed"
    except Exception as exc:
        report.update(status="failed", error=f"{type(exc).__name__}: {exc}"); raise
    finally:
        (output / "metadata.json").write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    print(f"Completed. Figures and raw arrays: {output}", flush=True)
    print(f"Single-sample metrics (not a full-test result): {metrics}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
