"""Publication-width FFN maps; shared typography and native mesh painters."""
from __future__ import annotations


def expert_label(index):
    # Stable local-bank identity, never an alignment across different blocks.
    name = ""
    index += 1
    while index:
        index, digit = divmod(index - 1, 26)
        name = chr(65 + digit) + name
    return "Expert " + name


def grid(vis, painter, mesh, values, bounds, *, path, args, column_labels,
         row_labels, vmin, vmax, label, cmap, synthetic):
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize
    rows, _, columns = values.shape
    left, right, gap = .40, .06, .07
    bottom, top = .70, .31 + (.25 if synthetic else 0.)
    cell_width = (args.width - left - right - gap * (columns - 1)) / columns
    ratio = (bounds[1][1] - bounds[1][0]) / (bounds[0][1] - bounds[0][0])
    cell_height = min(2.6, max(.55, cell_width * ratio))
    height = top + bottom + rows * cell_height + (rows - 1) * .13
    with plt.rc_context(vis.style(args)):
        fig, axes = plt.subplots(rows, columns, figsize=(args.width, height), squeeze=False)
        fig.subplots_adjust(left=left/args.width, right=1-right/args.width,
            bottom=bottom/height, top=1-top/height, wspace=gap/cell_width, hspace=.13/cell_height)
        norm = Normalize(vmin=vmin, vmax=vmax)
        for r in range(rows):
            for c in range(columns):
                ax = axes[r, c]
                painter(ax, mesh, values[r, :, c], bounds, cmap=cmap, norm=norm)
                if r == 0:
                    ax.set_title(column_labels[c], fontsize=args.font_size, pad=6.)
            # Figure text stays outside spatial data even though painter hides axes.
            box = axes[r, 0].get_position()
            fig.text(.12 / args.width, (box.y0+box.y1)/2, row_labels[r],
                     rotation=90, va="center", ha="center", fontsize=args.font_size)
        if synthetic:
            fig.text(.5, 1-.09/height, "SYNTHETIC CHECK", ha="center", va="top", fontsize=args.font_size)
        cax = fig.add_axes([.30, .43/height, .44, .07/height])
        bar = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax,
            orientation="horizontal", ticks=[vmin, (vmin+vmax)/2, vmax], format="%.3g")
        bar.outline.set_linewidth(.5)
        bar.ax.tick_params(labelsize=args.tick_font_size, length=2.5, width=.5, pad=2.)
        bar.set_label(label, fontsize=args.font_size, labelpad=3.)
        vis.save_figure(fig, path, args.dpi)
        plt.close(fig)
    return dict(width_inches=args.width, height_inches=height, vmin=vmin, vmax=vmax,
                row_labels=row_labels, column_labels=column_labels, bounds=bounds)


def mean_heatmap(vis, summary, path, args, synthetic):
    import numpy as np
    import matplotlib.pyplot as plt
    data = np.array([r["mean_gate"] for r in summary["visits"]])
    row_names = [f"{r['group'].capitalize()} {r['position']+1}, visit {r['visit']+1}" for r in summary["visits"]]
    extra_bottom = .35 if data.shape[1] > 4 else 0.
    height = max(2.6, .27 * len(data) + 1.15) + extra_bottom
    with plt.rc_context(vis.style(args)):
        fig, ax = plt.subplots(figsize=(args.width, height))
        fig.subplots_adjust(left=1.45/args.width, right=1-.12/args.width, top=1-(.45 if synthetic else .25)/height, bottom=(.90+extra_bottom)/height)
        artist = ax.imshow(data, vmin=0, vmax=1, cmap=args.cmap, aspect="auto", interpolation="nearest")
        ax.set_yticks(range(len(data)), row_names, fontsize=args.tick_font_size)
        ax.set_xticks(range(data.shape[1]), [expert_label(i) for i in range(data.shape[1])],
                      fontsize=args.tick_font_size, rotation=45 if data.shape[1] > 4 else 0, ha="right" if data.shape[1] > 4 else "center")
        ax.tick_params(length=0, pad=4)
        for spine in ax.spines.values():
            spine.set_linewidth(.5)
        cax = fig.add_axes([.40, .45/height, .38, .07/height])
        bar = fig.colorbar(artist, cax=cax, orientation="horizontal", ticks=[0, .5, 1])
        bar.outline.set_linewidth(.5)
        bar.ax.tick_params(labelsize=args.tick_font_size, length=2.5, width=.5, pad=2)
        bar.set_label("Mean gate weight", fontsize=args.font_size, labelpad=3)
        if synthetic:
            fig.text(.5, 1-.09/height, "SYNTHETIC CHECK", ha="center", va="top", fontsize=args.font_size)
        vis.save_figure(fig, path, args.dpi)
        plt.close(fig)
    return dict(width_inches=args.width, height_inches=height, vmin=0., vmax=1.,
                rows=row_names, expert_alignment="columns are local bank slots, not shared across physical blocks")


def export(task, vis, taskvis, output, sample, arrays, rows, summary, target, prediction, args, synthetic):
    import numpy as np
    import matplotlib
    matplotlib.use("Agg", force=True)
    x, y = sample["x"], sample["y"]
    mesh = np.stack((x, y), -1) if task == "elasticity" else vis.structured_mesh(x, y)
    painter = taskvis.point_paint if task == "elasticity" else vis.paint
    full = ((float(x.min()), float(x.max())), (float(y.min()), float(y.max())))
    views = {"full": full}
    if task == "airfoil":
        views["near"] = (tuple(args.near_xlim), tuple(args.near_ylim))
    if any(hi <= lo for bounds in views.values() for lo, hi in bounds):
        raise ValueError("Cannot plot a degenerate spatial domain")
    count = arrays["probabilities"].shape[-1]
    norm_max = float(arrays["contribution_norm"].max())
    # A truly zero update remains zero. Display range 0..1 is explicitly recorded.
    norm_limit = norm_max if norm_max > 0 else 1.
    owners = list(dict.fromkeys((r["group"], r["position"]) for r in rows))
    result = {}
    for group, position in owners:
        indices = [i for i, r in enumerate(rows) if (r["group"], r["position"]) == (group, position)]
        visits = [f"Visit {rows[i]['visit']+1}" for i in indices]
        prefix = f"{group}_{position:02d}"
        for view, bounds in views.items():
            for start in range(0, count, 4):
                stop = min(start+4, count)
                suffix = f"_experts_{start:02d}_{stop-1:02d}" if count > 4 else ""
                labels = [expert_label(i) for i in range(start, stop)]
                for key, name, maximum, label in (
                    ("probabilities", "gate", 1., "Gate weight"),
                    ("contribution_norm", "update_norm", norm_limit, "Weighted FFN update norm")):
                    stem = f"{prefix}_{name}_{view}{suffix}"
                    print("Rendering " + stem, flush=True)
                    result[stem] = grid(vis, painter, mesh, arrays[key][indices, :, start:stop], bounds,
                        path=output/stem, args=args, column_labels=labels, row_labels=visits,
                        vmin=0., vmax=maximum, label=label, cmap=args.cmap, synthetic=synthetic)
                    result[stem].update(owner=f"loop.{group}.{position}", logical_indices=indices,
                                        expert_indices=list(range(start, stop)))
                if len(indices) > 1:
                    delta = arrays["probabilities"][indices[1:], :, start:stop] - arrays["probabilities"][indices[0], :, start:stop]
                    stem = f"{prefix}_gate_change_{view}{suffix}"
                    result[stem] = grid(vis, painter, mesh, delta, bounds, path=output/stem, args=args,
                        column_labels=labels, row_labels=[v + " - 1" for v in visits[1:]],
                        vmin=-1., vmax=1., label="Change in gate weight", cmap="coolwarm", synthetic=synthetic)
            if count > 1:
                stem = f"{prefix}_entropy_{view}"
                result[stem] = grid(vis, painter, mesh, arrays["normalized_entropy"][indices, :, None], bounds,
                    path=output/stem, args=args, column_labels=["Expert mixing"], row_labels=visits,
                    vmin=0., vmax=1., label="Normalized entropy", cmap=args.cmap, synthetic=synthetic)
    result["mean_gate_weights"] = mean_heatmap(vis, summary, output/"mean_gate_weights", args, synthetic)
    vis.field_reference(mesh, target, prediction, views.get("near", full), output/"field_reference", args,
        "SYNTHETIC CHECK" if synthetic else "", field_name="Mach" if task == "airfoil" else taskvis.FIELDS[task], painter=painter)
    result["field_reference"] = dict(width_inches=args.width, height_inches=2.15)
    return result


def write_caption(task, output, rows, args, synthetic, count):
    # Prefer a core to show reuse; prefix/suffix use independent expert banks.
    row = next(r for r in rows if r["group"] == "core")
    suffix = "_experts_00_03" if count > 4 else ""
    filename = f"core_{row['position']:02d}_gate_full{suffix}.pdf"
    caption = ("SYNTHETIC CHECK; random model and synthetic data. " if synthetic else "")
    caption += (f"Pointwise dense-FFN gate weights on {task}, test sample {args.sample_index}, "
        f"at core position {row['position']+1}. Rows follow the loop visits; columns identify experts "
        "within the same shared expert bank. Each visit has its own router. "
        "Colors show unscaled softmax probabilities on a shared 0--1 scale; all experts execute. ")
    if count > 4:
        caption += "This page shows the first four experts; remaining experts are exported separately. "
    if count == 1:
        caption += "With one expert the gate is identically one, so no allocation contrast is expected. "
    if task == "ns":
        caption += f"The input at future step {args.forecast_step} is obtained by prediction feedback. "
    if task == "elasticity":
        caption += "Only original mesh nodes are drawn; no surface connectivity is inferred. "
    caption += "Gate weights alone do not establish physical specialization or causal importance."
    env = "figure*" if args.paper == "icml" else "figure"
    text = ("% Official conference template + graphicx; leave caption styling unchanged.\n"
        "% ICLR2026 uses times and normal 10 TeX pt captions; ICML2026 uses Times and 9 TeX pt.\n"
        "% Regenerate at the final width instead of shrinking the PDF.\n"
        f"\\begin{{{env}}}[t]\n  \\centering\n"
        f"  \\includegraphics[width={args.width:g}in]{{{filename}}}\n"
        f"  \\caption{{{caption}}}\n  \\label{{fig:v5-{task}-ffn}}\n\\end{{{env}}}\n")
    (output/"paper_figure.tex").write_text(text, encoding="utf-8")
    (output/"caption.txt").write_text(caption + "\n\nOther outputs: update_norm shows ||scale*pi_e*F_e(u)||_2, "
        "using one scale over all captured visits/experts; it is not causal attribution. gate_change is "
        "later visit minus first visit of the same physical block on [-1,1]. Entropy is H(pi)/log(E), "
        "omitted for E=1. Mean gates average nodes equally, not by physical area or volume. "
        "Different blocks have independent expert banks; Expert A is only a local-bank label.\n", encoding="utf-8")
