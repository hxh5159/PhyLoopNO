#!/usr/bin/env python3
"""Read-only V5 Airfoil inference and Figure-10-style Q/K state atlases.

No experiment entry is imported. Temporary hooks observe native projection
outputs; the normal forward, temperature, parameters and checkpoint stay intact.
See README.md for the distinction between Q, K, and Transolver slice weights.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from functools import lru_cache
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import uuid

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

DEFAULT_RUN = None
DATA_ROOT = REPO / 'data'
DATA_FILES = ("NACA_Cylinder_X.npy", "NACA_Cylinder_Y.npy", "NACA_Cylinder_Q.npy")
PAPER_WIDTHS = {"iclr": 5.5, "icml": 6.75}  # 2026 template textwidth, inches
FONT_PT = 9.  # Figure labels, not the formal LaTeX caption.
TICK_PT = 8.
TIMES_FAMILIES = ("Times New Roman", "TeX Gyre Termes", "Nimbus Roman",
                  "Nimbus Roman No9 L", "Liberation Serif", "STIXGeneral")


def add_typography_arguments(p):
    p.add_argument("--font-family", default="auto",
                   help="auto: available Times-family face, then bundled Times-style STIXGeneral; explicit names must exist")
    p.add_argument("--font-size", type=float, default=FONT_PT, help="figure label size in final printed points; not the LaTeX caption")
    p.add_argument("--tick-font-size", type=float, default=TICK_PT, help="colorbar tick size in final printed points")


def validate_typography(args):
    for name in ("font_size", "tick_font_size"):
        value = getattr(args, name)
        if not math.isfinite(value) or not 6 <= value <= 18:
            raise ValueError(f"--{name.replace('_', '-')} must be finite and between 6 and 18 pt")


@lru_cache(maxsize=None)
def resolve_font(requested="auto"):
    """Resolve a real installed face; never silently substitute DejaVu Sans."""
    from matplotlib import font_manager
    candidates = TIMES_FAMILIES if requested == "auto" else (requested,)
    for candidate in candidates:
        try:
            path = font_manager.findfont(font_manager.FontProperties(family=[candidate]), fallback_to_default=False)
        except ValueError:
            continue
        actual = font_manager.FontProperties(fname=path).get_name()
        if actual != candidate:
            continue
        return dict(requested=requested, family=actual, file=path, sha256=sha256(path),
                    times_style_fallback=(requested == "auto" and actual == "STIXGeneral"))
    raise ValueError(f"Requested font {requested!r} is not installed. Use --font-family auto or an installed Times-family font; no silent sans-serif fallback.")


def typography(args):
    return dict(**resolve_font(args.font_family), label_pt=args.font_size, tick_pt=args.tick_font_size,
                panel_numbers=False, formal_caption="official LaTeX template; Times-family; not rasterized into figure")


def sha256(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--run-dir", type=Path, required=True)
    data_default = Path(os.environ.get("CDLNO_AIRFOIL_ROOT", str(
        Path(os.environ.get("CDLNO_DATA_ROOT", str(DATA_ROOT))) / "fno/airfoil/naca")))
    p.add_argument("--data-path", type=Path, default=data_default)
    p.add_argument("--checkpoint", default="final", help="final, latest or epoch_XXXX; never guesses a run")
    p.add_argument("--sample-index", type=int, default=0, help="test-set index 0..199 (raw index = 1000 + index)")
    p.add_argument("--head", default="0", help="one zero-based head index, or all; heads are never averaged")
    p.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    p.add_argument("--gpu", type=int, default=0, help="CUDA index within the current CUDA_VISIBLE_DEVICES")
    p.add_argument("--output-dir", type=Path, help="must not exist; default: this script's outputs/<unique name>")
    p.add_argument("--views", nargs="+", choices=("full", "near"), default=["full", "near"])
    p.add_argument("--near-xlim", nargs=2, type=float, default=[-0.25, 1.25], metavar=("MIN", "MAX"))
    p.add_argument("--near-ylim", nargs=2, type=float, default=[-0.5, 0.5], metavar=("MIN", "MAX"))
    p.add_argument("--full-columns", type=int, default=16, help="Transolver Figure 10 uses 4 rows x 16 columns")
    p.add_argument("--near-columns", type=int, default=8)
    p.add_argument("--paper", choices=("iclr", "icml"), default="iclr",
                   help="full-text-width layout (ICML uses a two-column figure*)")
    p.add_argument("--width", type=float, help="final printed width in inches; default: ICLR 5.5, ICML 6.75")
    p.add_argument("--annotate", action="store_true", help="include browsing titles; omit for paper submission")
    add_typography_arguments(p)
    p.add_argument("--dpi", type=int, default=400)
    p.add_argument("--cmap", choices=("viridis", "cividis", "coolwarm"), default="viridis")
    p.add_argument("--individual", action="store_true", help="also save each near-view, peak-normalized Q/K state as PNG/PDF")
    p.add_argument("--preview", action="store_true", help="read only architecture JSON; no tensors, data reads or output creation")
    return p


def validate_request(args, metadata):
    cfg = metadata["resolved_config"]
    if cfg["architecture"] != "partial_share_feature_gate_v5" or cfg["task"] != "airfoil":
        raise ValueError("This script requires a V5 Airfoil checkpoint")
    if cfg["actual_M"] != 64:
        raise ValueError(f"The requested 64-state atlas requires actual_M=64, saved M={cfg['actual_M']}")
    spec = cfg["model_spec"]["constructor_kwargs"]
    if spec["fun_dim"] != 0 or spec["out_dim"] != 1 or spec["time_input"]:
        raise ValueError("Unsupported Airfoil input/output contract")
    data = metadata["data_spec"]
    if data["protocol"]["normalizer"] != "none" or metadata["normalizer_spec"]["records"]:
        raise ValueError("Airfoil visualization expects the native unnormalized coordinates/field")
    if data["protocol"]["split"] != "first1000 train; next200 test":
        raise ValueError("Unsupported Airfoil split; refusing to guess the test sample")
    for key, expected in (("ntrain", 1000), ("ntest", 200)):
        if data["runtime"].get(key, expected) != expected:
            raise ValueError(f"Saved {key} conflicts with the native Airfoil split")
    if not 0 <= args.sample_index < 200:
        raise ValueError("--sample-index must be in 0..199")
    heads = list(range(spec["heads"])) if args.head == "all" else [int(args.head)]
    if any(h < 0 or h >= spec["heads"] for h in heads):
        raise ValueError(f"--head must be in 0..{spec['heads'] - 1}, or all")
    if args.width is None:
        args.width = PAPER_WIDTHS[args.paper]
    validate_typography(args)
    if args.gpu < 0 or args.dpi < 72 or not math.isfinite(args.width) or args.width < 4:
        raise ValueError("gpu>=0, dpi>=72 and width>=4 are required")
    if not 1 <= args.full_columns <= 64 or not 1 <= args.near_columns <= 64:
        raise ValueError("column counts must be in 1..64")
    for name in ("near_xlim", "near_ylim"):
        limits = getattr(args, name)
        if not all(math.isfinite(v) for v in limits) or limits[0] >= limits[1]:
            raise ValueError(name + " requires finite MIN < MAX")
    return heads


def load_sample(data_path, metadata, sample_index):
    """Memory-map only the requested sample, preserving native order and channel 4."""
    import numpy as np
    cfg = metadata["resolved_config"]
    expected = metadata["data_spec"]["checksums"]
    actual = {}
    for name in DATA_FILES:
        path = Path(data_path) / name
        if not path.is_file():
            raise FileNotFoundError(f"Missing Airfoil data file: {path}")
        if name not in expected:
            raise ValueError(f"Saved checkpoint has no dataset checksum for {name}")
        print(f"Checking dataset SHA-256: {name}", flush=True)
        actual[name] = sha256(path)
        if actual[name] != expected[name]:
            raise ValueError(f"Dataset checksum differs from the training run: {name}")
    arrays = [np.load(Path(data_path) / name, mmap_mode="r", allow_pickle=False) for name in DATA_FILES]
    x, y, fields = arrays
    grid = (cfg["model_spec"]["constructor_kwargs"]["grid_height"],
            cfg["model_spec"]["constructor_kwargs"]["grid_width"])
    if (x.ndim != 3 or x.shape != y.shape or x.shape[0] < 1200 or
            fields.ndim != 4 or fields.shape[0] != x.shape[0] or fields.shape[1] <= 4 or
            tuple(x.shape[1:]) != grid or tuple(fields.shape[2:]) != grid):
        raise ValueError(f"Data grid/layout differs from the saved model {grid}; no implicit resampling is performed")
    index = 1000 + sample_index
    # The real entry casts coordinates and labels to float32 before its forward.
    xx = np.array(x[index], dtype=np.float32, copy=True)
    yy = np.array(y[index], dtype=np.float32, copy=True)
    target = np.array(fields[index, 4], dtype=np.float32, copy=True)
    if not all(np.isfinite(v).all() for v in (xx, yy, target)):
        raise ValueError("The selected sample contains NaN/Inf")
    return xx, yy, target, actual


def capture_last_attention(model, positions, *, fx=None):
    """Observe actual logits/V/native readout; check hook transparency and factor math."""
    import torch
    if model.training:
        raise ValueError("Capture requires model.eval()")
    operator = model.loop.suffix[-1].Attn
    if len(operator.visits) != 1:
        raise ValueError("The last suffix must be a single-visit operator")
    route = operator.visits[0]
    records, counts, handles = {}, {}, []

    def save(name):
        def hook(module, inputs, output):
            counts[name] = counts.get(name, 0) + 1
            records[name] = output.detach().clone()
        return hook

    def readout_hook(module, inputs):
        counts["readout"] = counts.get("readout", 0) + 1
        records["readout"] = inputs[0].detach().clone()

    def rng():
        return [torch.random.get_rng_state()] + (
            torch.cuda.get_rng_state_all() if positions.is_cuda else [])

    initial_rng = rng()
    versions = {name: value._version for name, value in model.state_dict(keep_vars=True).items()}
    with torch.inference_mode():
        baseline = model(positions, fx=fx)
        try:
            handles.append(route.to_q.register_forward_hook(save("q_logits")))
            handles.append(route.to_k.register_forward_hook(save("k_logits")))
            handles.append(operator.to_v.register_forward_hook(save("v")))
            handles.append(operator.to_out.register_forward_pre_hook(readout_hook))
            prediction = model(positions, fx=fx)
        finally:
            for handle in handles:
                handle.remove()
        if not torch.equal(prediction, baseline):
            raise RuntimeError("Observed forward differs from the unobserved forward")
        if counts != {name: 1 for name in ("q_logits", "k_logits", "v", "readout")}:
            raise RuntimeError(f"Expected exactly one final-attention visit, observed {counts}")
        q_logits, k_logits = records["q_logits"], records["k_logits"]
        if operator.variant in ("temp", "conv_temp"):
            tq, tk = route.temperature_q.clamp(.01, 1.), route.temperature_k.clamp(.01, 1.)
        elif operator.variant == "shapenet":
            tq, tk = route.tempreature_q.clamp(.1, 2.), route.tempreature_k.clamp(.1, 2.)
        elif operator.variant in ("plain", "conv", "airfrans"):
            tq = tk = 1.
        else:
            raise ValueError("Unsupported attention variant: " + operator.variant)
        # These detached diagnostics use the native temperature and softmax axes.
        q = (q_logits / tq).softmax(dim=-1)
        k = (k_logits / tk).softmax(dim=-2)
        context = torch.einsum("bhnm,bhnd->bhmd", k, records["v"])
        observed = torch.einsum("bhnm,bhmd->bhnd", q, context)
        observed = observed.transpose(1, 2).reshape_as(records["readout"])
        torch.testing.assert_close(observed, records["readout"], atol=1e-6, rtol=1e-5)
        for value in (q, k, prediction):
            if not torch.isfinite(value).all():
                raise ValueError("NaN/Inf in model output or routing weights")
        q_error = (q.sum(-1) - 1).abs().max().item()
        k_error = (k.sum(-2) - 1).abs().max().item()
        if max(q_error, k_error) > 5e-6:
            raise RuntimeError("Routing normalization check failed")
        if versions != {n: v._version for n, v in model.state_dict(keep_vars=True).items()}:
            raise RuntimeError("Inference modified model parameters/buffers")
        if not all(torch.equal(a, b) for a, b in zip(initial_rng, rng())):
            raise RuntimeError("Evaluation or diagnostics consumed the public Torch RNG")
        diagnostics = dict(hooks=counts, hooked_prediction_bitwise_equal=True,
            parameters_unmodified=True, torch_rng_unchanged=True,
            q_row_sum_max_error=q_error, k_column_sum_max_error=k_error,
            native_readout_max_error=(observed - records["readout"]).abs().max().item(),
            temperature_q=torch.as_tensor(tq).detach().cpu().flatten().tolist(),
            temperature_k=torch.as_tensor(tk).detach().cpu().flatten().tolist())
        return q[0].cpu().numpy(), k[0].cpu().numpy(), prediction[0, :, 0].cpu().numpy(), diagnostics


def structured_mesh(x, y):
    """Triangulate only existing grid cells, never Delaunay across the airfoil hole."""
    import numpy as np
    import matplotlib.tri as mtri
    if x.shape != y.shape or x.ndim != 2 or min(x.shape) < 2:
        raise ValueError("Expected matching structured coordinate grids")
    rows, cols = x.shape
    ii, jj = np.meshgrid(np.arange(rows - 1), np.arange(cols - 1), indexing="ij")
    a = (ii * cols + jj).ravel()
    triangles = np.concatenate([np.stack((a, a + cols, a + cols + 1), axis=1),
                                np.stack((a, a + cols + 1, a + 1), axis=1)])
    xx, yy = x.ravel().astype(float), y.ravel().astype(float)
    p, q, r = triangles.T
    area2 = (xx[q] - xx[p]) * (yy[r] - yy[p]) - (yy[q] - yy[p]) * (xx[r] - xx[p])
    scale = max(float(np.ptp(xx) * np.ptp(yy)), 1.)
    valid = np.abs(area2) > np.finfo(float).eps * scale
    triangles = triangles[valid]
    if not len(triangles):
        raise ValueError("Degenerate mesh: all cell triangles have zero area")
    negative = area2[valid] < 0
    triangles[negative] = triangles[negative][:, [0, 2, 1]]
    return mtri.Triangulation(xx, yy, triangles)


def display_values(weights, kind, normalization):
    """Keep calibrated and within-state contrast products explicitly separate."""
    import numpy as np
    if weights.ndim != 2 or not np.isfinite(weights).all() or np.any(weights < 0):
        raise ValueError("weights must be finite nonnegative [N,M]")
    if kind not in ("Q", "K") or normalization not in ("shared", "peak"):
        raise ValueError("Unknown weight kind/display normalization")
    values = weights.astype(np.float64)
    if normalization == "peak":
        maxima = values.max(axis=0, keepdims=True)
        values = np.divide(values, maxima, out=np.zeros_like(values), where=maxima > 0)
        return values, 1., f"{kind} / maximum per state (shape only)"
    if kind == "K":
        values *= len(values)
    return values, max(float(values.max()), 1e-12), (
        "Q (shared scale)" if kind == "Q" else "N K (shared scale; uniform = 1)")


def style(args):
    family = resolve_font(args.font_family)["family"]
    return {"font.family": [family], "font.size": args.font_size,
            "mathtext.fontset": "stix", "pdf.fonttype": 42, "ps.fonttype": 42,
            "axes.linewidth": .5, "axes.titlesize": args.font_size, "axes.labelsize": args.font_size,
            "xtick.labelsize": args.tick_font_size, "ytick.labelsize": args.tick_font_size,
            "figure.constrained_layout.h_pad": .07,
            "savefig.facecolor": "white", "figure.facecolor": "white", "savefig.bbox": None,
            "text.usetex": False}


def paint(ax, mesh, values, bounds, *, cmap, norm):
    artist = ax.tripcolor(mesh, values, shading="gouraud", cmap=cmap, norm=norm,
                          rasterized=True)
    ax.set_xlim(*bounds[0]); ax.set_ylim(*bounds[1])
    ax.set_aspect("equal", adjustable="box")
    ax.set_axis_off()
    return artist


def save_figure(fig, path, dpi):
    for suffix in ("png", "pdf"):
        # Preserve the declared physical width; tight cropping silently changes
        # the scale of every font when LaTeX includes the file at that width.
        fig.savefig(path.with_suffix("." + suffix), dpi=dpi, bbox_inches=None)


def atlas(mesh, weights, *, kind, normalization, bounds, columns, title, path, args, painter=None):
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize
    values, vmax, label = display_values(weights, kind, normalization)
    count = values.shape[1]
    rows = math.ceil(count / columns)
    ratio = (bounds[1][1] - bounds[1][0]) / (bounds[0][1] - bounds[0][0])
    # No latent numbers or reserved label rows. Font sizes stay independent of
    # panel count and physical width; the compact grid follows the paper atlases.
    margin, gap = .055, .025
    bottom = .27 + (1.4 * args.font_size + 1.4 * args.tick_font_size) / 72
    header = (1.4 * args.font_size / 72 + .09) if title else .035
    cell_width = (args.width - 2 * margin - (columns - 1) * gap) / columns
    grid_height = rows * cell_width * ratio + (rows - 1) * gap
    height = grid_height + bottom + header
    with plt.rc_context(style(args)):
        fig, axes = plt.subplots(rows, columns, figsize=(args.width, height), squeeze=False)
        fig.subplots_adjust(left=margin / args.width, right=1 - margin / args.width,
                            bottom=bottom / height, top=1. - header / height,
                            wspace=gap / cell_width, hspace=gap / (cell_width * ratio))
        norm = Normalize(vmin=0., vmax=vmax)
        for state, ax in enumerate(axes.flat):
            if state >= count:
                ax.set_axis_off(); continue
            (painter or paint)(ax, mesh, values[:, state], bounds, cmap=args.cmap, norm=norm)
        if title:
            fig.suptitle(title, fontsize=args.font_size, y=1. - .07 / height)
        cax = fig.add_axes([.26, (bottom - .22) / height, .48, .07 / height])
        bar = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=args.cmap), cax=cax,
                           orientation="horizontal", ticks=[0., vmax / 2, vmax], format="%.3g")
        bar.outline.set_linewidth(.5)
        bar.ax.tick_params(labelsize=args.tick_font_size, length=2.5, width=.5, pad=2.)
        bar.set_label(label, fontsize=args.font_size, labelpad=3.)
        save_figure(fig, path, args.dpi)
        plt.close(fig)
    return dict(vmin=0., vmax=vmax, transform=label, latent_order=list(range(count)),
                width_inches=args.width, height_inches=height,
                minimum_font_pt=min(args.font_size, args.tick_font_size), panel_numbers=False,
                latent_spatial_max=weights.max(axis=0).tolist(),
                latent_spatial_mean=weights.mean(axis=0).tolist(),
                latent_spatial_std=weights.std(axis=0).tolist())


def field_reference(mesh, target, prediction, bounds, path, args, title, *, field_name="Mach", painter=None):
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize
    low, high = min(target.min(), prediction.min()), max(target.max(), prediction.max())
    high = max(high, low + 1e-8)
    error = np.abs(prediction - target)
    with plt.rc_context(style(args)):
        fig, axes = plt.subplots(1, 3, figsize=(args.width, 2.15), layout="constrained")
        for ax, values, label in zip(axes, (target, prediction, error),
                                     (f"Reference {field_name}", f"Predicted {field_name}", "Absolute error")):
            norm = Normalize(0., max(float(error.max()), 1e-12)) if label == "Absolute error" else Normalize(low, high)
            artist = (painter or paint)(ax, mesh, values, bounds,
                           cmap="magma" if label == "Absolute error" else "viridis", norm=norm)
            # These identify distinct field panels; the overall title is in LaTeX.
            ax.set_title(label, fontsize=args.font_size)
            ticks = [norm.vmin, (norm.vmin + norm.vmax) / 2, norm.vmax]
            bar = fig.colorbar(artist, ax=ax, orientation="horizontal", fraction=.05, pad=.06,
                               ticks=ticks, format="%.3g")
            bar.ax.tick_params(length=2.5, width=.5, pad=2.)
        if title:
            fig.suptitle(title, fontsize=args.font_size)
        save_figure(fig, path, args.dpi)
        plt.close(fig)


def export_plots(output, x, y, q, k, target, prediction, heads, args, *, synthetic=False):
    import matplotlib
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize
    mesh = structured_mesh(x, y)
    full = ((float(x.min()), float(x.max())), (float(y.min()), float(y.max())))
    near = (tuple(args.near_xlim), tuple(args.near_ylim))
    if (near[0][1] <= full[0][0] or near[0][0] >= full[0][1] or
            near[1][1] <= full[1][0] or near[1][0] >= full[1][1]):
        raise ValueError("Near-view bounds do not intersect the mesh")
    prefix = "SYNTHETIC CHECK | " if synthetic else ""
    figures = {}
    for head in heads:
        for kind, all_weights in (("Q", q), ("K", k)):
            weights = all_weights[head]
            for view in dict.fromkeys(args.views):
                bounds = full if view == "full" else near
                for normalization in ("shared", "peak"):
                    name = f"{kind.lower()}_head{head:02d}_{view}_{normalization}"
                    print("Rendering " + name, flush=True)
                    title = (prefix + f"Airfoil | {kind} | head {head} | test {args.sample_index}") if (args.annotate or synthetic) else ""
                    info = atlas(mesh, weights, kind=kind, normalization=normalization, bounds=bounds,
                                 columns=args.full_columns if view == "full" else args.near_columns,
                                 title=title, path=output / name, args=args)
                    figures[name] = dict(**info, bounds=bounds, head=head, kind=kind, normalization=normalization)
            if args.individual:
                directory = output / f"{kind.lower()}_head{head:02d}_individual_peak"
                directory.mkdir()
                values, _, label = display_values(weights, kind, "peak")
                for state in range(weights.shape[-1]):
                    with plt.rc_context(style(args)):
                        fig, ax = plt.subplots(figsize=(3., 2.1), layout="constrained")
                        paint(ax, mesh, values[:, state], near, cmap=args.cmap, norm=Normalize(0, 1))
                        # Identity stays in the filename/NPZ; no state number is
                        # burned into the panel, including individual exports.
                        if synthetic:
                            ax.set_title("SYNTHETIC CHECK", fontsize=args.font_size)
                        elif args.annotate:
                            ax.set_title(f"{kind} weights | peak normalized", fontsize=args.font_size)
                        save_figure(fig, directory / f"state_{state:02d}", args.dpi)
                        plt.close(fig)
    field_reference(mesh, target, prediction, near, output / "field_reference", args,
                    (prefix + f"Airfoil | test sample {args.sample_index}") if (args.annotate or synthetic) else "")
    return figures


def write_latex(output, args, heads, *, synthetic):
    """Let the unmodified conference style own caption font/spacing/numbering."""
    view = args.views[0]
    name = f"q_head{heads[0]:02d}_{view}_peak.pdf"
    environment = "figure*" if args.paper == "icml" else "figure"
    caption = ("SYNTHETIC CHECK; random model and synthetic mesh. " if synthetic else "")
    caption += (
        "Final-layer reconstruction routing weights on Airfoil "
        f"(test sample {args.sample_index}, head {heads[0]}). "
        "The 64 states retain their original order, arranged row by row without panel numbers. "
        r"Each panel shows $Q_{i,m}/\max_i Q_{i,m}$, with the maximum taken over "
        "the complete mesh. The common color scale is 0--1; this normalization compares "
        "spatial patterns, not absolute strengths across latent states. "
        "Colors interpolate within native mesh cells; the airfoil interior has no samples. "
        "These are routing weights, not separate Mach predictions."
    )
    text = (
        "% Use your official conference template and graphicx.\n"
        "% Put this PDF on your graphics search path. Do not override caption styling.\n"
        "% ICLR official preamble: \\usepackage{iclr2026_conference,times}; caption is normally 10 TeX pt.\n"
        "% ICML official style loads Times and sets captions to 9 TeX pt. Do not load another caption font.\n"
        f"% Exported width {args.width:g} in; labels {args.font_size:g} pt; ticks {args.tick_font_size:g} pt.\n"
        "% If a different width is needed, regenerate with --width; do not shrink the PDF.\n"
        f"\\begin{{{environment}}}[t]\n"
        "  \\centering\n"
        f"  \\includegraphics[width={args.width:g}in]{{{name}}}\n"
        f"  \\caption{{{caption}}}\n"
        f"  \\label{{fig:airfoil-v5-q-h{heads[0]}-{view}}}\n"
        f"\\end{{{environment}}}\n"
    )
    (output / "paper_figure.tex").write_text(text, encoding="utf-8")


def main(argv=None):
    args = parser().parse_args(argv)
    # JSON metadata first, BEFORE importing Torch or constructing a tensor model.
    from linearno_loop.v5.schema import read_metadata
    args.run_dir = args.run_dir.expanduser().resolve()
    args.data_path = args.data_path.expanduser().resolve()
    metadata = read_metadata(args.run_dir / "architecture.json")
    heads = validate_request(args, metadata)
    cfg = metadata["resolved_config"]
    summary = dict(run_dir=str(args.run_dir), data_path=str(args.data_path),
        checkpoint=args.checkpoint, config_hash=cfg["config_hash"], heads=heads,
        topology=cfg["topology_preset"], executed_depth=cfg["loop_spec"]["executed_depth"],
        expert_count=cfg["expert_count"], M=cfg["actual_M"], core_norm_mode=cfg["core_norm_mode"],
        sample_index=args.sample_index, raw_sample_index=1000 + args.sample_index,
        attention_path=f"loop.suffix.{cfg['suffix_blocks'] - 1}.Attn", visit_index=0,
        metadata_scope=metadata["data_spec"]["scope"])
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)
    if args.preview:
        return 0
    font_info = typography(args)
    print(f"Figure font: {font_info['family']}; labels {args.font_size:g} pt, ticks {args.tick_font_size:g} pt; no panel numbers", flush=True)
    if args.output_dir is not None and args.output_dir.expanduser().exists():
        raise FileExistsError("--output-dir already exists; choose a new directory")
    import numpy as np
    import torch
    from cdlno.linearno_loop.v5.checkpoint import load_model
    if args.device == "cuda":
        if not torch.cuda.is_available() or args.gpu >= torch.cuda.device_count():
            raise RuntimeError("Requested GPU is unavailable; use --device cpu explicitly if desired")
        device = torch.device(f"cuda:{args.gpu}")
    else:
        device = torch.device("cpu")
    x, y, target, checksums = load_sample(args.data_path, metadata, args.sample_index)
    print("Loading verified checkpoint pair (strict=True)...", flush=True)
    model, loaded = load_model(args.run_dir, selector=args.checkpoint, map_location=device)
    # Also validate the selected epoch, whose immutable fields must match the sidecar.
    validate_request(args, loaded)
    model.eval()
    positions = torch.from_numpy(np.stack((x, y), axis=-1).reshape(1, -1, 2)).to(device)
    print("Running two evaluation forwards to verify observation transparency...", flush=True)
    q, k, prediction, diagnostic = capture_last_attention(model, positions)
    target_flat = target.ravel()
    denominator = float(np.linalg.norm(target_flat.astype(np.float64)))
    relative_l2 = (float(np.linalg.norm(prediction.astype(np.float64) - target_flat) / denominator)
                   if denominator > 0 else None)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    output = (args.output_dir.expanduser().resolve() if args.output_dir else
              Path(__file__).resolve().parent / "outputs" / f"airfoil_{cfg['config_hash'][:12]}_{stamp}_{uuid.uuid4().hex[:8]}")
    output.mkdir(parents=True, exist_ok=False)
    report = dict(**summary, status="rendering", epoch=loaded["resume_state"]["epoch"],
        checkpoint_role=loaded["resume_state"]["checkpoint_role"],
        checkpoint_metadata_hash=loaded["metadata_hash"], data_sha256=checksums,
        inference=dict(device=str(device), torch=torch.__version__, python=sys.version,
                       cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"),
                       dtype="float32", amp=False, mode="eval", diagnostic=diagnostic),
        single_sample_relative_l2=relative_l2,
        formulas=dict(Q="softmax_M(native_q_logits/native_tau_q)",
                      K="softmax_N(native_k_logits/native_tau_k)",
                      peak="W[i,m]/max_i(W[i,m]); no minimum subtraction; contrast only",
                      k_shared="N*K; uniform spatial distribution = 1"),
        rendering=dict(cmap=args.cmap, dpi=args.dpi, width_inches=args.width,
                       paper_template=args.paper, typography=font_info,
                       line_width_pt=.5, pdf_fonttype=42, exact_page_width=True,
                       caption_style="controlled by official LaTeX template via paper_figure.tex",
                       browsing_titles=args.annotate,
                       interpolation="Gouraud within native grid-cell triangles; no Delaunay or field smoothing",
                       head_aggregation="none", latent_order="original index 0..63",
                       full_columns=args.full_columns, near_columns=args.near_columns),
        source_sha256={str(p.relative_to(REPO)): sha256(p) for p in (
            Path(__file__), REPO / "cdlno/linearno_loop/v5/operator.py",
            REPO / "cdlno/linearno_loop/v5/core.py", REPO / "cdlno/linearno_loop/v5/standard.py")})
    np.savez_compressed(output / "routing_weights.npz", x=x, y=y, target=target,
        prediction=prediction.reshape(x.shape), Q=q[heads], K=k[heads],
        head_indices=np.asarray(heads), latent_indices=np.arange(64),
        test_index=np.asarray(args.sample_index), raw_sample_index=np.asarray(1000 + args.sample_index))
    try:
        report["figures"] = export_plots(output, x, y, q, k, target_flat, prediction, heads, args,
                                         synthetic=metadata["data_spec"]["scope"] != "real")
        report["status"] = "completed"
    except Exception as exc:
        report["status"] = "failed"
        report["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        (output / "metadata.json").write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    caption = (
        "Spatial routing weights of the final attention layer of V5 on Airfoil, "
        f"test sample {args.sample_index} (raw index {1000 + args.sample_index}), "
        f"checkpoint epoch {report['epoch']}. All 64 states retain their original row-major order without panel numbers. "
        "Q denotes reconstruction weights (normalized across latents at each point); "
        "K denotes compression weights (normalized across all mesh points for each latent). "
        "Each head is displayed separately. Shared-scale panels show Q or N*K without per-state scaling; "
        "peak-normalized panels show W/max_i(W) to reveal spatial patterns and do not compare absolute state strengths. "
        "Colors interpolate over native mesh cells; the white airfoil interior contains no mesh points. "
        "These are routing weights, not 64 independent Mach predictions or proof of physical specialization.\n"
    )
    (output / "caption.txt").write_text(caption)
    write_latex(output, args, heads, synthetic=metadata["data_spec"]["scope"] != "real")
    print(f"Completed. Figures and raw arrays: {output}", flush=True)
    print(f"Single selected sample relative L2: {relative_l2} (not a full-test metric)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
