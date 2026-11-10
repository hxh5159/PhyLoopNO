#!/usr/bin/env python3
"""Read-only V5 dense-FFN weights, loop-visit comparisons and update magnitudes."""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import sys
from types import SimpleNamespace
import uuid

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
for directory in (REPO, HERE.parent / "weight"):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))
import airfoil_states as vis
import task_states as taskvis
from capture import capture_experts, summarize
import render

TASKS = ("airfoil", "darcy", "elasticity", "ns", "pipe")


def parser(task):
    data = vis.parser().get_default("data_path") if task == "airfoil" else taskvis.parser(task).get_default("data_path")
    p = argparse.ArgumentParser(description=f"{task}: {__doc__}", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--list-runs", action="store_true", help="list metadata only; never choose an experiment automatically")
    p.add_argument("--data-path", type=Path, default=data)
    p.add_argument("--checkpoint", default="final", help="final, latest, epoch_XXXX")
    p.add_argument("--sample-index", type=int, default=0)
    p.add_argument("--forecast-step", type=int, help="NS only: future step 1..10, defaults to 10")
    p.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    p.add_argument("--gpu", type=int, default=0, help="index within CUDA_VISIBLE_DEVICES")
    p.add_argument("--output-dir", type=Path, help="new directory only; never overwrites a prior output")
    p.add_argument("--paper", choices=tuple(vis.PAPER_WIDTHS), default="iclr")
    p.add_argument("--width", type=float, help="final printed width in inches")
    p.add_argument("--dpi", type=int, default=400)
    p.add_argument("--cmap", choices=("viridis", "cividis"), default="viridis")
    p.add_argument("--near-xlim", type=float, nargs=2, default=[-.25, 1.25])
    p.add_argument("--near-ylim", type=float, nargs=2, default=[-.5, .5])
    vis.add_typography_arguments(p)
    p.add_argument("--preview", action="store_true", help="metadata only; no torch/data/weights/figures")
    return p


def validate(task, args, metadata):
    # Reuse reviewed task contracts without exposing meaningless attention-head options.
    values = dict(vars(args), head="0", columns=4, full_columns=16, near_columns=8)
    check = SimpleNamespace(**values)
    if task == "airfoil":
        if args.forecast_step is not None:
            raise ValueError("--forecast-step applies only to NS")
        vis.validate_request(check, metadata)
    else:
        taskvis.validate_request(task, check, metadata)
    args.width, args.forecast_step = check.width, check.forecast_step
    if not math.isfinite(args.width):
        raise ValueError("--width must be finite")
    for bound in (args.near_xlim, args.near_ylim):
        if not all(math.isfinite(v) for v in bound) or bound[0] >= bound[1]:
            raise ValueError("View bounds must be finite and increasing")


def load_sample(task, args, metadata):
    import numpy as np
    import torch
    if task == "airfoil":
        x, y, target, checksums = vis.load_sample(args.data_path, metadata, args.sample_index)
        sample = dict(x=x, y=y, target=target, positions=torch.from_numpy(np.stack((x,y), -1).reshape(1,-1,2)),
            fx=None, normalizers={}, raw_sample_index=1000+args.sample_index,
            source_file=vis.DATA_FILES[-1], extras={})
    else:
        checksums = taskvis.verify_files(task, args.data_path, metadata)
        sample = taskvis.load_sample(task, args.data_path, metadata, args.sample_index)
    return sample, checksums


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
                    arrays, rows, prediction, report = capture_experts(model, positions, fx=fx)
                    out = torch.from_numpy(prediction.copy()).to(device).reshape(1,-1,1)
                else:
                    out = model(positions, fx=fx)
                if not torch.isfinite(out).all():
                    raise ValueError(f"Non-finite NS prediction at step {step}")
                predictions.append(out[0,:,0].cpu().numpy().copy())
                fx = torch.cat((fx[...,1:], out), dim=-1)
        rollout = np.stack(predictions, -1)
        prediction = rollout[:, forecast_step-1]
        target = sample["target"][:, forecast_step-1]
        extras.update(predicted_rollout=rollout, target_rollout=sample["target"],
                      future_steps=np.arange(1,11), source_frame_indices=np.arange(10,20))
    else:
        arrays, rows, encoded, report = capture_experts(model, positions, fx=fx)
        prediction = encoded if task == "airfoil" else taskvis.transform(
            torch.from_numpy(encoded[None]), sample["normalizers"]["output"], inverse=True)[0].numpy()
        target = sample["target"].ravel()
        extras["encoded_prediction"] = encoded
        if fx is not None:
            extras["model_input_field"] = fx[0].cpu().numpy().copy()
    extras["model_input_positions"] = positions[0].cpu().numpy().copy()
    if not np.isfinite(prediction).all():
        raise ValueError("Non-finite decoded prediction")
    return arrays, rows, target, prediction, report, extras


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False)+"\n", encoding="utf-8")


def write_csv(path, summary):
    with path.open("w", newline="", encoding="utf-8") as stream:
        fields = ("logical_index", "group", "position", "visit", "expert_index", "mean_gate",
                  "mean_expert_output_norm", "mean_contribution_norm", "expert_scale",
                  "mean_entropy_nats", "mean_normalized_entropy")
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in summary["visits"]:
            for expert in range(summary["expert_count"]):
                values = {f: row[f] for f in fields if f in row}
                for f in ("mean_gate", "mean_expert_output_norm", "mean_contribution_norm"):
                    values[f] = row[f][expert]
                writer.writerow(dict(values, expert_index=expert))


def main(argv=None):
    tokens = list(sys.argv[1:] if argv is None else argv)
    if not tokens or tokens[0] not in TASKS:
        raise SystemExit("Usage: ffn_states.py {airfoil,darcy,elasticity,ns,pipe} [options]")
    task = tokens.pop(0)
    p = parser(task); args = p.parse_args(tokens)
    args.run_dir = args.run_dir.expanduser().resolve()
    args.data_path = args.data_path.expanduser().resolve()
    if args.list_runs:
        print(json.dumps(taskvis.list_runs(args.run_dir, task), indent=2, ensure_ascii=False)); return 0
    if not (args.run_dir/"architecture.json").is_file():
        p.error("--run-dir must name one exact experiment with architecture.json. Use --list-runs for a parent; no automatic run selection.")
    from linearno_loop.v5.schema import read_metadata
    metadata = read_metadata(args.run_dir/"architecture.json")
    validate(task, args, metadata)
    cfg = metadata["resolved_config"]
    info = dict(task=task, run_dir=str(args.run_dir), data_path=str(args.data_path),
        checkpoint=args.checkpoint, config_hash=cfg["config_hash"], sample_index=args.sample_index,
        forecast_step=args.forecast_step, expert_count=cfg["expert_count"], core_norm_mode=cfg["core_norm_mode"],
        topology=cfg["topology_preset"], loop_spec=cfg["loop_spec"], metadata_scope=metadata["data_spec"]["scope"],
        observation="all logical visits; all dense experts; no attention-head axis")
    print(json.dumps(info, indent=2, ensure_ascii=False), flush=True)
    if args.preview:
        return 0
    typography = vis.typography(args)
    print(f"Figure font: {typography['family']}; labels {args.font_size:g}pt, ticks {args.tick_font_size:g}pt", flush=True)
    if args.output_dir is not None and args.output_dir.expanduser().exists():
        raise FileExistsError("--output-dir already exists; use a new path")
    import numpy as np
    import torch
    from cdlno.linearno_loop.v5.checkpoint import load_model
    if args.device == "cuda" and (not torch.cuda.is_available() or args.gpu >= torch.cuda.device_count()):
        raise RuntimeError("Requested CUDA device unavailable; choose --device cpu explicitly")
    device = torch.device(f"cuda:{args.gpu}" if args.device == "cuda" else "cpu")
    print("Loading verified checkpoint pair (strict=True)...", flush=True)
    model, loaded = load_model(args.run_dir, selector=args.checkpoint, map_location=device)
    validate(task, args, loaded); model.eval()
    sample, checksums = load_sample(task, args, loaded)
    arrays, rows, target, prediction, diagnostic, extras = infer(task, model, sample, device, args.forecast_step)
    summary = summarize(arrays, rows)
    if cfg["expert_count"] == 1:
        print("E=1: gate weights are identically 1; normalized entropy is undefined and omitted. Update norms remain informative.", flush=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    output = args.output_dir.expanduser().resolve() if args.output_dir else HERE/"outputs"/f"{task}_{cfg['config_hash'][:12]}_{stamp}_{uuid.uuid4().hex[:8]}"
    # Preserve training archives: even an explicit output override cannot write inside the run.
    if output == args.run_dir or args.run_dir in output.parents:
        raise ValueError("Figure output must be outside the training run")
    output.mkdir(parents=True, exist_ok=False)
    metrics = dict(selected_sample_relative_l2=taskvis.relative_l2(prediction, target))
    if task == "ns":
        steps = [taskvis.relative_l2(extras["predicted_rollout"][:,i], extras["target_rollout"][:,i]) for i in range(10)]
        metrics.update(per_step_relative_l2=steps, mean_step_relative_l2=sum(steps)/10 if all(x is not None for x in steps) else None,
            rollout_full_relative_l2=taskvis.relative_l2(extras["predicted_rollout"], extras["target_rollout"]))
    report = dict(**info, status="rendering", epoch=loaded["resume_state"]["epoch"],
        checkpoint_metadata_hash=loaded["metadata_hash"], checkpoint_role=loaded["resume_state"]["checkpoint_role"],
        raw_sample_index=sample["raw_sample_index"], source_file=sample["source_file"], data_sha256=checksums,
        visits=rows, metrics=metrics, metrics_scope="selected test sample only, not full-test or convergence results",
        inference=dict(torch=torch.__version__, python=sys.version, device=str(device), dtype="float32", amp=False,
                       cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"), diagnostic=diagnostic),
        formulas=dict(gate="softmax_expert(router_visit(LN2_visit(z)))", z="x+Operator_visit(LN1_visit(x))",
            contribution_norm="||scale*pi_e*F_e(u)||_2; core scale=1/R, prefix/suffix scale=1",
            entropy="-sum(pi*log(pi)); normalized by log(E) only for E>1",
            differences="later visit minus visit0 within the same physical bank",
            causal_attribution=False, averaging=summary["averaging"]),
        rendering=dict(typography=typography, width_inches=args.width, paper=args.paper, dpi=args.dpi,
            gate_scale=[0.,1.], delta_scale=[-1.,1.], contribution_scale="one global maximum over all captured visits, points and experts",
            panel_numbers=False, identity_labels="Expert A/B/... local to a physical bank; Visit 1/2/...",
            geometry="original nodes only" if task == "elasticity" else "native structured cell triangles; Gouraud colors"),
        source_sha256={str(f.relative_to(REPO)):vis.sha256(f) for f in (
            HERE/"ffn_states.py", HERE/"capture.py", HERE/"render.py", Path(vis.__file__), Path(taskvis.__file__),
            REPO/"cdlno/linearno_loop/v5/core.py", REPO/"cdlno/linearno_loop/v5/operator.py")})
    try:
        np.savez_compressed(output/"expert_weights.npz", **arrays, **extras,
            x=sample["x"], y=sample["y"], target=target.reshape(sample["x"].shape), prediction=prediction.reshape(sample["x"].shape),
            logical_indices=np.arange(len(rows)), groups=np.array([r["group"] for r in rows]),
            positions=np.array([r["position"] for r in rows]), visits=np.array([r["visit"] for r in rows]),
            expert_indices=np.arange(cfg["expert_count"]), expert_scales=np.array([r["expert_scale"] for r in rows]),
            test_index=np.array(args.sample_index), raw_sample_index=np.array(sample["raw_sample_index"]))
        write_json(output/"expert_summary.json", summary)
        write_csv(output/"expert_summary.csv", summary)
        synthetic = loaded["data_spec"]["scope"] != "real"
        report["figures"] = render.export(task, vis, taskvis, output, sample, arrays, rows, summary, target, prediction, args, synthetic)
        render.write_caption(task, output, rows, args, synthetic, cfg["expert_count"])
        report["status"] = "completed"
    except Exception as exc:
        report.update(status="failed", error=f"{type(exc).__name__}: {exc}"); raise
    finally:
        write_json(output/"metadata.json", report)
    print(f"Completed. Figures and raw expert arrays: {output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
