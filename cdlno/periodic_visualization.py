"""Task-aware, observation-only field diagnostics every 50 completed epochs.

No task entry imports, optimizer access, model hooks, architecture changes, or
checkpoint decisions. Fixed cases are prepared inside an RNG-isolated context.
"""
from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
import random

import numpy as np
import torch

from .training_state import isolated_evaluation
from .visualization import ColorScale, render_fields, render_curves


def due(epoch, total_epochs):
    return epoch > 0 and (epoch % 50 == 0 or epoch == total_epochs)


def cpu(value):
    return value.detach().cpu().clone()


def array(value):
    return cpu(value).numpy() if torch.is_tensor(value) else np.array(value, copy=True)


def relative_errors(truth, prediction):
    # Expected [N,C,T]; scalar NS is represented by C=1.
    numerator = np.linalg.norm((prediction - truth).reshape(-1, truth.shape[-1]), axis=0)
    denominator = np.linalg.norm(truth.reshape(-1, truth.shape[-1]), axis=0)
    return np.divide(numerator, denominator, out=np.zeros_like(numerator), where=denominator != 0), denominator


class PeriodicFields:
    def __init__(self, directory, task, model_name, *, seed=None, member=0):
        self.directory = Path(directory) / 'visualizations' / f'member_{member:03d}'
        self.task, self.model_name, self.seed = task, model_name, seed
        self.cases = None

    def after_epoch(self, model, epoch, total_epochs, *, dataset, **kwargs):
        if not due(epoch, total_epochs):
            return None
        event = dict(completed_epoch=epoch, task=self.task, seed=self.seed, status='running')
        try:
            self.directory.mkdir(parents=True, exist_ok=True)
            with isolated_evaluation(model):
                if self.cases is None:
                    # Fixed diagnostics across model/seed runs; training RNG is restored.
                    random.seed(0); np.random.seed(0); torch.manual_seed(0)
                    self.cases = self._prepare(dataset, graph_device=next(model.parameters()).device, **kwargs)
                for index, case in enumerate(self.cases):
                    path = self.directory / f'epoch_{epoch:04d}' / f'case_{index:03d}'
                    path.mkdir(parents=True, exist_ok=False)
                    if self.task in ('car', 'airfrans'):
                        self._graph(model, case, path, index, epoch, **kwargs)
                    else:
                        self._standard(model, case, path, index, epoch, **kwargs)
                event['status'] = 'completed'
        except Exception as error:
            event.update(status='failed', error=f'{type(error).__name__}: {error}')
            logging.getLogger(__name__).warning('%s visualization epoch %s failed: %s',
                                              self.task, epoch, event['error'])
        try:
            with (self.directory / 'events.jsonl').open('a', encoding='utf-8') as stream:
                stream.write(json.dumps(event, allow_nan=False) + '\n')
        except OSError as error:
            logging.getLogger(__name__).warning('Cannot write visualization event: %s', error)
        return event

    def _prepare(self, dataset, **kwargs):
        if len(dataset) < 1:
            raise ValueError('visualization requires a nonempty held-out dataset')
        cases = []
        for i in range(min(2, len(dataset))):
            item = dataset[i]
            if self.task == 'car':
                data, geom = item
                cases.append((data.clone().cpu(), cpu(geom)))
            elif self.task == 'airfrans':
                data = item.clone().cpu()
                count = kwargs['hparams']['subsampling']
                if count > data.x.shape[0]:
                    raise ValueError('AirfRANS visualization sample exceeds graph size')
                # Same uniform without-replacement rule as the original val sample;
                # local RNG fixes the sample without changing the real val loop.
                idx = torch.tensor(random.Random(i).sample(range(data.x.shape[0]), count))
                original_size = data.x.shape[0]
                for name in ('pos', 'x', 'y', 'surf'):
                    setattr(data, name, getattr(data, name)[idx].clone())
                from torch_geometric import nn as nng
                data.edge_index = nng.radius_graph(data.pos.to(kwargs['graph_device']),
                    r=kwargs['hparams']['r'], loop=True,
                    max_num_neighbors=int(kwargs['hparams']['max_neighbors'])).cpu()
                cases.append((data, idx, original_size))
            else:
                cases.append(tuple(cpu(t) for t in item))
        return cases

    def _metadata(self, index, **extra):
        return dict(task=self.task, model=self.model_name, seed=self.seed,
                    held_out_index=index, case_selection='first two held-out cases, never selected by error',
                    **extra)

    def _render(self, path, name, index, epoch, xyz, truth, prediction, channels, *,
                grid_shape=None, mask=None, prediction_coordinates=None, note='', extra=None):
        xyz, truth, prediction = array(xyz), array(truth), array(prediction)
        mask = np.ones(len(xyz), dtype=bool) if mask is None else array(mask).astype(bool)
        if not mask.any():
            raise ValueError(f'{name}: empty display region')
        scale_path = self.directory / 'scales' / f'case_{index:03d}_{name}.json'
        if scale_path.exists():
            saved = json.loads(scale_path.read_text())
            if saved['channels'] != list(channels):
                raise ValueError('visualization channel mismatch')
            scales = [ColorScale(**s) for s in saved['scales']]
        else:
            scales = []
            for channel in range(truth.shape[1]):
                values = truth[mask, channel]
                lo, hi = float(values.min()), float(values.max())
                extent = max(hi-lo, max(abs(lo), abs(hi), 1.) * 1e-6)
                if hi <= lo:
                    lo -= extent; hi += extent
                # Truth-only limits are independent of model/seed/epoch accuracy.
                scales.append(ColorScale(lo, hi, .2*extent))
            scale_path.parent.mkdir(parents=True, exist_ok=True)
            with scale_path.open('x') as stream:
                json.dump(dict(channels=list(channels), scales=[s.to_dict() for s in scales],
                               error_rule='20 percent of reference range; saturation reported, raw values saved'), stream, indent=2)
        metadata = self._metadata(index, caption_note=note, **(extra or {}))
        return render_fields(path / name, coordinates=xyz, truth=truth, prediction=prediction,
            channel_names=channels, scales=scales, task=self.task, case_id=str(index), completed_epoch=epoch,
            grid_shape=grid_shape, display_mask=mask, metadata=metadata, model_name=self.model_name,
            prediction_coordinates=prediction_coordinates,
            coordinate_labels=('$x$', '$z$') if name == 'volume_velocity' else ('$x$', '$y$'))

    def _standard(self, model, case, path, index, epoch, *, grid_shape=None,
                  output_normalizer=None, coordinate_normalizer=None, **kwargs):
        device = next(model.parameters()).device
        batch = [value.unsqueeze(0).to(device) for value in case]
        xyz = batch[0]
        if self.task == 'ns':
            _, window, truth = batch
            outputs = []
            for _ in range(truth.shape[-1]):
                out = model(xyz, window)
                outputs.append(out)
                window = torch.cat((window[..., 1:], out), dim=-1)
            prediction = torch.cat(outputs, dim=-1)
            gt, pred = array(truth[0])[:, None, :], array(prediction[0])[:, None, :]
            frames = sorted(set((0, truth.shape[-1]//2-1, truth.shape[-1]-1)))
            for t in frames:
                self._render(path, f'frame_{t+1:02d}', index, epoch, xyz[0], gt[..., t], pred[..., t],
                    ['Scalar field $u$'], grid_shape=grid_shape,
                    note=f'Forecast step {t+1} of {truth.shape[-1]} with prediction feedback from the original ten-frame input; no truth feedback.')
            self._time_curve(path, np.arange(1, truth.shape[-1]+1), gt, pred, 'Forecast step')
            np.savez_compressed(path / 'trajectory.npz', coordinates=array(xyz[0]),
                                input_window=array(batch[1][0]), truth=gt, prediction=pred)
        elif self.task == 'plasticity':
            _, times, fx, truth = batch
            outputs = [model(xyz, fx, T=times[:, t:t+1]) for t in range(times.shape[-1])]
            prediction = torch.stack(outputs, dim=-1)
            gt, pred = array(truth[0]), array(prediction[0])
            for t in sorted(set((0, times.shape[-1]//2-1, times.shape[-1]-1))):
                geometry = gt[:, :2, t]
                magnitude = np.linalg.norm(gt[:, 2:, t], axis=-1, keepdims=True)
                pred_magnitude = np.linalg.norm(pred[:, 2:, t], axis=-1, keepdims=True)
                self._render(path, f'time_{t+1:02d}', index, epoch, geometry, magnitude, pred_magnitude,
                    ['Deformation magnitude'], grid_shape=grid_shape,
                    prediction_coordinates=pred[:, :2, t],
                    note=f'T = {float(times[0,t]):.6g}. Magnitude uses original output channels 2 and 3; geometry uses channels 0 and 1. No temporal prediction feedback.')
            self._time_curve(path, array(times[0]), gt, pred, 'Time condition $T$')
            np.savez_compressed(path / 'trajectory.npz', input_coordinates=array(xyz[0]),
                                time=array(times[0]), truth=gt, prediction=pred)
        else:
            _, fx, truth = batch
            field = fx.unsqueeze(-1) if self.task == 'darcy' else None
            out = model(xyz, field).squeeze(-1)
            if output_normalizer is not None:
                out = output_normalizer.decode(out)
            coords = xyz if coordinate_normalizer is None else coordinate_normalizer.decode(xyz)
            channels = {'darcy':['Solution $u$'], 'elasticity':['Stress'],
                        'airfoil':['Mach number'], 'pipe':['Velocity component']}[self.task]
            self._render(path, 'field', index, epoch, coords[0], truth[0, :, None], out[0, :, None],
                         channels, grid_shape=grid_shape,
                         note='Fields use the original task output scale; no physical unit is inferred.')
            if self.task == 'airfoil':
                coords = array(coords[0]); region = ((coords[:,0] >= -.25) & (coords[:,0] <= 1.5)
                                                     & (abs(coords[:,1]) <= .5))
                if region.any():
                    self._render(path, 'near_airfoil', index, epoch, coords, truth[0,:,None], out[0,:,None],
                        channels, mask=region, note='Fixed physical window x in [-0.25, 1.5], y in [-0.5, 0.5]; native points, no remeshing.')

    @staticmethod
    def _time_curve(path, times, truth, prediction, xlabel):
        errors, denominator = relative_errors(truth, prediction)
        # Undefined zero-truth ratios are not silently represented as zero.
        valid = denominator != 0
        np.savez_compressed(path / 'time_errors.npz', time=times, relative_l2=errors, defined=valid)
        if valid.any():
            render_curves(path / 'time_error', times[valid], {'Case relative $L_2$':errors[valid]},
                          xlabel=xlabel, ylabel='Relative $L_2$ error')
            caption = ('Single-case relative L2 error versus ' + xlabel.replace('$', '') + '. '
                       'At each time, the norm includes all original spatial nodes and output channels. '
                       'This is a diagnostic trajectory, not the dataset-average evaluation metric. '
                       'Zero-reference norms are marked undefined in time_errors.npz and omitted from the curve.')
            (path / 'time_error/caption.txt').write_text(caption + '\n', encoding='utf-8')
            (path / 'time_error/caption.tex').write_text('\\caption{' + caption.replace('_', r'\_') + '}\n', encoding='utf-8')

    def _graph(self, model, case, path, index, epoch, *, coef_norm=None, **kwargs):
        device = next(model.parameters()).device
        data = case[0].clone().to(device)
        if self.task == 'car':
            prediction = model((data, case[1].clone().to(device)))
        else:
            prediction = model(data)
        gt, pred, xyz, surf = array(data.y), array(prediction), array(data.pos), array(data.surf).astype(bool)
        decoded = coef_norm is not None and len(coef_norm) == 4
        if decoded:
            mean, std = array(coef_norm[2]), array(coef_norm[3])
            gt, pred = gt * (std + 1e-8) + mean, pred * (std + 1e-8) + mean
        base_note = ('Values decoded with the training-set output normalizer. ' if decoded else
                     'Normalized outputs; no output normalization coefficients supplied. ')
        extra = {}
        arrays = dict(coordinates=xyz, truth=gt, prediction=pred, surf=surf)
        if self.task == 'airfrans':
            idx = array(case[1]); arrays['sample_indices'] = idx
            extra = dict(original_nodes=case[2], sampled_nodes=len(idx),
                         index_sha256=hashlib.sha256(idx.tobytes()).hexdigest())
            base_note += 'Fixed sampled validation diagnostic, not the official repeated-sampling/scatter-averaged evaluation. '
            self._render(path, 'fields', index, epoch, xyz, gt, pred,
                         ['$v_x$', '$v_y$', 'Pressure $p$', 'Eddy viscosity $\\nu_t$'], note=base_note, extra=extra)
            region = (xyz[:,0] >= -.25) & (xyz[:,0] <= 1.5) & (abs(xyz[:,1]) <= .5)
            if region.any():
                self._render(path, 'near_airfoil', index, epoch, xyz, gt[:,2:3], pred[:,2:3], ['Pressure $p$'],
                    mask=region, note=base_note+'Fixed near-airfoil physical window.', extra=extra)
        else:
            self._render(path, 'surface_pressure', index, epoch, xyz, gt[:,3:4], pred[:,3:4],
                         ['Surface pressure $p$'], mask=surf, note=base_note+'Only surface pressure is displayed.')
            # Fixed centre-y geometric slab on volume points, not a CFD interpolation.
            volume = ~surf
            if not volume.any():
                raise ValueError('Car visualization has no volume nodes')
            center = float((xyz[volume,1].min()+xyz[volume,1].max())/2)
            width = float(np.ptp(xyz[volume,1]))*.02
            distances = np.abs(xyz[:,1]-center)
            width = max(width, float(distances[volume].min()), 1e-12)
            slab = volume & (distances <= width)
            arrays['volume_display_mask'] = slab
            channels = ['$v_x$', '$v_y$', '$v_z$', 'Speed $|\\mathbf{v}|$']
            fields = np.column_stack((gt[:,:3], np.linalg.norm(gt[:,:3],axis=1)))
            predictions = np.column_stack((pred[:,:3], np.linalg.norm(pred[:,:3],axis=1)))
            self._render(path, 'volume_velocity', index, epoch, xyz[:,[0,2]], fields, predictions,
                channels, mask=slab, note=base_note+f'Volume point-cloud slab |y - {center:.6g}| <= {width:.6g}; horizontal x and vertical z, no mesh interpolation.',
                extra=dict(slab_axis='y', slab_center=center, slab_half_width=width))
        np.savez_compressed(path / 'case.npz', **arrays)
