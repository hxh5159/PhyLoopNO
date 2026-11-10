"""Pure output-field rendering; no model, data loader, or training imports.

Task adapters decode/reduce fields before calling this module. Grid shape is
explicit; point clouds never acquire fabricated connectivity/interpolation.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np


# 7.16 inches = 181.9 mm, suitable for a two-column manuscript figure.
# STIX is bundled with Matplotlib, so no system Times font or TeX is required.
PUBLICATION_STYLE = {'font.family': 'STIXGeneral', 'mathtext.fontset': 'stix',
    'font.size': 8, 'axes.titlesize': 9, 'axes.labelsize': 8,
    'xtick.labelsize': 7, 'ytick.labelsize': 7, 'legend.fontsize': 7,
    'axes.linewidth': .6, 'lines.linewidth': 1., 'pdf.fonttype': 42,
    'ps.fonttype': 42, 'savefig.facecolor': 'white', 'figure.facecolor': 'white'}


def _array(value):
    # Duck-type torch tensors so importing visualization does not import torch.
    if hasattr(value, 'detach'):
        value = value.detach().cpu().numpy()
    result = np.array(value, copy=True)
    if result.dtype.kind not in 'fiu':
        raise ValueError('fields/coordinates must be numeric arrays')
    if not np.isfinite(result).all():
        raise ValueError('nonfinite field/coordinate; figure cannot hide invalid values')
    return result


def _range(lo, hi):
    if hi <= lo:
        delta = max(abs(lo), 1.) * 1e-6
        return float(lo-delta), float(hi+delta)
    return float(lo), float(hi)


@dataclass(frozen=True)
class ColorScale:
    """One field's fixed truth limits and independent absolute-error limit."""
    minimum: float
    maximum: float
    error_maximum: float

    def __post_init__(self):
        if not all(np.isfinite(x) for x in (self.minimum, self.maximum, self.error_maximum)):
            raise ValueError('color scale must be finite')
        if self.maximum <= self.minimum or self.error_maximum <= 0:
            raise ValueError('invalid color scale interval')

    @classmethod
    def from_first_case(cls, truth, prediction):
        truth, prediction = _array(truth), _array(prediction)
        if truth.size == 0 or truth.shape != prediction.shape:
            raise ValueError('truth/prediction must have identical nonempty shapes')
        lo, hi = _range(truth.min(), truth.max())
        err = float(np.abs(prediction.astype(np.float64)-truth).max())
        return cls(lo, hi, err if err > 0 else max(abs(lo), abs(hi), 1.) * 1e-6)

    def to_dict(self):
        return dict(minimum=self.minimum, maximum=self.maximum, error_maximum=self.error_maximum)


def fixed_scales(path, truth, prediction, channel_names):
    """Create once, otherwise read; different channel names are never reused."""
    path = Path(path)
    truth, prediction = _array(truth), _array(prediction)
    if (truth.ndim != 2 or 0 in truth.shape or prediction.shape != truth.shape
            or truth.shape[1] != len(channel_names) or len(set(channel_names)) != len(channel_names)):
        raise ValueError('expected matching [N,C] and C channel names')
    if path.exists():
        saved = json.loads(path.read_text())
        if (saved.get('channel_names') != list(channel_names) or saved.get('version') != 1
                or len(saved.get('scales', [])) != len(channel_names)):
            raise ValueError('existing visualization scale channel/version mismatch')
        return [ColorScale(**v) for v in saved['scales']]
    scales = [ColorScale.from_first_case(truth[:, i], prediction[:, i]) for i in range(truth.shape[1])]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as f:
        json.dump(dict(version=1, channel_names=list(channel_names), scales=[s.to_dict() for s in scales]), f, indent=2)
        f.write('\n')
    return scales


def render_fields(directory, *, coordinates, truth, prediction, channel_names,
                  scales, task, case_id, completed_epoch, grid_shape=None,
                  display_mask=None, units=None, view=(20., -60.), metadata=None,
                  model_name='PhyLoopNO', prediction_coordinates=None, coordinate_labels=('$x$', '$y$')):
    """Write scalar-channel GT/pred/absolute-error panels + original arrays.

    Structured panels preserve explicit [H,W] point order and physical xy.
    Unstructured 2D/3D uses scatter; display masks NEVER subset model inputs.
    Caller supplies fixed scales. No interpolation, clipping of saved arrays,
    or creation of physical units. Returns JSON metrics, not a figure handle.
    """
    xyz, gt, pred = _array(coordinates), _array(truth), _array(prediction)
    if xyz.ndim != 2 or xyz.shape[1] not in (2, 3) or xyz.shape[0] == 0:
        raise ValueError('coordinates must be nonempty [N,2] or [N,3]')
    if gt.ndim != 2 or pred.shape != gt.shape or gt.shape[0] != xyz.shape[0] or gt.shape[1] == 0:
        raise ValueError('truth/prediction must be matching [N,C] fields')
    if len(channel_names) != gt.shape[1] or len(set(channel_names)) != len(channel_names) or len(scales) != gt.shape[1]:
        raise ValueError('unique channel names and fixed scales must match C')
    if type(completed_epoch) is not int or completed_epoch < 1:
        raise ValueError('completed_epoch must be positive integer')
    if not all(isinstance(scale, ColorScale) for scale in scales):
        raise ValueError('scales must contain ColorScale objects')
    units = [''] * gt.shape[1] if units is None else list(units)
    if len(units) != gt.shape[1]:
        raise ValueError('units must match channels')
    if len(coordinate_labels) != 2:
        raise ValueError('two coordinate labels required for the displayed plane')
    mask = np.ones(xyz.shape[0], dtype=bool) if display_mask is None else np.array(display_mask, copy=True)
    if mask.shape != (xyz.shape[0],) or mask.dtype != np.bool_ or not mask.any():
        raise ValueError('display_mask must be nonempty boolean [N]')
    if grid_shape is not None:
        if (len(grid_shape) != 2 or any(type(i) is not int or i < 2 for i in grid_shape)
                or np.prod(grid_shape) != xyz.shape[0] or xyz.shape[1] != 2 or not mask.all()):
            raise ValueError('structured field requires explicit H,W with N=H*W and unmasked xy')
    error = np.abs(pred.astype(np.float64) - gt.astype(np.float64))
    pred_xyz = xyz if prediction_coordinates is None else _array(prediction_coordinates)
    if pred_xyz.shape != xyz.shape:
        raise ValueError('prediction coordinates must preserve point correspondence')
    metrics = []
    for c, (name, scale) in enumerate(zip(channel_names, scales)):
        denominator = np.linalg.norm(gt[:, c].astype(np.float64))
        metrics.append(dict(channel=name, unit=units[c],
            relative_l2=None if denominator == 0 else float(np.linalg.norm(error[:, c])/denominator),
            relative_l2_status='undefined_zero_target' if denominator == 0 else 'defined',
            maximum_absolute_error=float(error[:, c].max()),
            prediction_outside_color_fraction=float(np.mean((pred[:, c] < scale.minimum) | (pred[:, c] > scale.maximum))),
            error_above_color_fraction=float(np.mean(error[:, c] > scale.error_maximum)),
            displayed_prediction_outside_color_fraction=float(np.mean((pred[mask, c] < scale.minimum) | (pred[mask, c] > scale.maximum))),
            displayed_error_above_color_fraction=float(np.mean(error[mask, c] > scale.error_maximum)),
            displayed_mse=float(np.mean(error[mask, c]**2)),
            displayed_relative_l2=(None if np.linalg.norm(gt[mask, c]) == 0 else
                float(np.linalg.norm(error[mask, c])/np.linalg.norm(gt[mask, c]))),
            color_scale=scale.to_dict()))
    record = dict(version=1, task=task, case_id=case_id, completed_epoch=completed_epoch,
        shape=list(gt.shape), grid_shape=None if grid_shape is None else list(grid_shape),
        coordinates_dimension=xyz.shape[1], displayed_points=int(mask.sum()),
        scalar_error='abs(prediction-truth)', channel_metrics=metrics, view=list(view),
        supplied_metadata=metadata or {}, model_name=model_name,
        typography=PUBLICATION_STYLE, figure_width_inches=7.16, png_dpi=600,
        prediction_geometry='separate' if prediction_coordinates is not None else 'shared',
        error_geometry='ground_truth')
    record['coordinate_labels'] = list(coordinate_labels)
    # Fail on invalid metadata before creating output files.
    encoded = json.dumps(record, indent=2, allow_nan=False) + '\n'
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    import matplotlib as mpl
    # rc_context restores all font/export settings even if rendering fails.
    with mpl.rc_context(PUBLICATION_STYLE):
        figure = Figure(figsize=(7.16, 2.25 * gt.shape[1]), layout='constrained')
        FigureCanvasAgg(figure)
        try:
            _draw_fields(figure, xyz, pred_xyz, gt, pred, error, mask, channel_names,
                         scales, units, grid_shape, view, model_name, coordinate_labels)
            figure.savefig(directory / 'fields.png', dpi=600, bbox_inches='tight', pad_inches=.03)
            figure.savefig(directory / 'fields.pdf', dpi=600, bbox_inches='tight', pad_inches=.03)
        finally:
            figure.clear()
    np.savez_compressed(directory / 'fields.npz', coordinates=xyz, prediction_coordinates=pred_xyz,
                        truth=gt, prediction=pred, absolute_error=error, signed_error=pred-gt,
                        display_mask=mask, channel_names=np.array(channel_names))
    (directory / 'metadata.json').write_text(encoded, encoding='utf-8')
    caption = (f'{model_name} predictions for {task}, case {case_id}, after {completed_epoch} training epochs. '
               'Each row shows (left) reference, (middle) prediction, and (right) pointwise absolute error. '
               'Reference and prediction share a color scale; error uses its own scale. '
               'Color limits are fixed across the recorded epochs. '
               'Field values and any saturated colors are documented in the accompanying arrays and metadata.')
    if prediction_coordinates is not None:
        caption += ' Reference and prediction use their respective deformed coordinates; errors use reference coordinates with the original node correspondence.'
    if not mask.all():
        caption += ' Only the stated geometric mask is displayed; it is not a change to model inputs.'
    details = (metadata or {}).get('caption_note', '')
    if (metadata or {}).get('seed') is not None:
        caption += f" Training seed: {metadata['seed']}."
    if details:
        caption += ' ' + details
    (directory / 'caption.txt').write_text(caption + '\n', encoding='utf-8')
    # Escape text for a copyable LaTeX caption without requiring a TeX runtime.
    escapes = {'\\': r'\textbackslash{}', '&': r'\&', '%': r'\%', '$': r'\$', '#': r'\#',
               '_': r'\_', '{': r'\{', '}': r'\}'}
    latex = ''.join(escapes.get(char, char) for char in caption)
    (directory / 'caption.tex').write_text('\\caption{' + latex + '}\n', encoding='utf-8')
    return record


def _draw_fields(figure, xyz, pred_xyz, gt, pred, error, mask, channel_names,
                 scales, units, grid_shape, view, model_name, coordinate_labels):
    for c, name in enumerate(channel_names):
        scale = scales[c]
        for col, (label, values) in enumerate((('Reference', gt[:, c]), (model_name, pred[:, c]), ('Absolute error', error[:, c]))):
            coords = pred_xyz if col == 1 else xyz
            ax = figure.add_subplot(gt.shape[1], 3, c*3+col+1, projection='3d' if xyz.shape[1] == 3 else None)
            lo, hi = (0., scale.error_maximum) if col == 2 else (scale.minimum, scale.maximum)
            color = 'OrRd' if col == 2 else ('RdBu_r' if lo < 0 < hi else 'viridis')
            if grid_shape is not None:
                artist = ax.pcolormesh(coords[:, 0].reshape(grid_shape), coords[:, 1].reshape(grid_shape),
                    values.reshape(grid_shape), shading='gouraud', cmap=color, vmin=lo, vmax=hi, rasterized=True)
                ax.set_aspect('equal')
            elif xyz.shape[1] == 2:
                artist = ax.scatter(coords[mask, 0], coords[mask, 1], c=values[mask], s=2,
                    linewidths=0, cmap=color, vmin=lo, vmax=hi, rasterized=True)
                ax.set_aspect('equal')
            else:
                artist = ax.scatter(coords[mask, 0], coords[mask, 1], coords[mask, 2], c=values[mask], s=1.2,
                    linewidths=0, cmap=color, vmin=lo, vmax=hi, depthshade=False, rasterized=True)
                ax.view_init(elev=view[0], azim=view[1])
                ax.set_proj_type('ortho')
                ax.set_box_aspect(np.maximum(np.ptp(xyz[mask], axis=0), 1e-12))
                ax.set_axis_off()
            if xyz.shape[1] == 2:
                joint = np.concatenate((xyz[mask], pred_xyz[mask]))
                ax.set_xlim(*_range(joint[:, 0].min(), joint[:, 0].max()))
                ax.set_ylim(*_range(joint[:, 1].min(), joint[:, 1].max()))
                ax.set_xlabel(coordinate_labels[0]); ax.set_ylabel(coordinate_labels[1])
                ax.tick_params(length=2, pad=2)
            ax.set_title(f'({chr(97+c*3+col)}) {label}\n{name}', pad=4)
            figure.colorbar(artist, ax=ax, label=units[c], shrink=.8,
                           pad=.025, fraction=.045, extend='max' if col == 2 else 'both')


def render_curves(directory, x, series, *, xlabel, ylabel):
    """Pure diagnostic curve export, with exactly the supplied numerical values."""
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    import matplotlib as mpl
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    x = _array(x)
    values = {name: _array(value) for name, value in series.items()}
    if any(value.shape != x.shape for value in values.values()):
        raise ValueError('curve x and y shapes must match')
    with mpl.rc_context(PUBLICATION_STYLE):
        fig = Figure(figsize=(3.5, 2.3), layout='constrained')
        FigureCanvasAgg(fig)
        try:
            ax = fig.add_subplot(111)
            for name, value in values.items():
                ax.plot(x, value, marker='o', markersize=2, label=name)
            ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)
            ax.legend(frameon=False); ax.grid(alpha=.2)
            for ext in ('pdf', 'png'):
                fig.savefig(directory / ('curve.' + ext), dpi=600, bbox_inches='tight', pad_inches=.03)
            np.savez_compressed(directory / 'curve.npz', x=x, **values)
        finally:
            fig.clear()
