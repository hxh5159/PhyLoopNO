"""Explicit Car evaluation fields and real-sample drag input for LinearNO only.

Original loader, evaluator and drag helper remain untouched. This module does
not choose the still-unresolved paper TRAINING objective (L0 C08).
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import torch


def fields(prediction, target, surf, coef_norm):
    if not isinstance(prediction, torch.Tensor) or prediction.ndim != 2 or prediction.shape[1] != 4:
        raise ValueError('Car fields require prediction [N,4]')
    if not isinstance(target, torch.Tensor) or target.shape != prediction.shape or target.device != prediction.device:
        raise ValueError('Car target must match prediction shape/device')
    if not isinstance(surf, torch.Tensor) or surf.dtype != torch.bool or surf.shape != (len(prediction),) or surf.device != prediction.device:
        raise ValueError('Car surf must be boolean [N] on prediction device')
    if not surf.any() or surf.all():
        raise ValueError('Car evaluation needs both surface and surrounding-region points')
    if not isinstance(coef_norm, (tuple, list)) or len(coef_norm) != 4:
        raise ValueError('Car evaluation requires saved coef_norm=[mean_in,std_in,mean_out,std_out]')
    mean = torch.as_tensor(coef_norm[2], device=prediction.device, dtype=prediction.dtype)
    std = torch.as_tensor(coef_norm[3], device=prediction.device, dtype=prediction.dtype)
    if mean.shape != (4,) or std.shape != (4,) or not torch.isfinite(mean).all() or not torch.isfinite(std).all() or (std < 0).any():
        raise ValueError('Car saved output mean/std require four finite values and std>=0')
    # Preserve existing evaluation decode exactly, including its no-eps multiply.
    return prediction*std+mean, target*std+mean


def relative_l2(prediction, target):
    """One sample/region: flatten all supplied channels; no stabilizing epsilon."""
    if prediction.shape != target.shape or not target.numel():
        raise ValueError('relative L2 requires equal nonempty prediction/target shapes')
    denominator = torch.linalg.vector_norm(target.reshape(-1))
    if not torch.isfinite(denominator) or denominator <= 0:
        raise ValueError('relative L2 target norm must be finite and nonzero; epsilon is not implicit')
    return torch.linalg.vector_norm((prediction-target).reshape(-1))/denominator


@torch.no_grad()
def field_metrics(prediction, target, surf, coef_norm):
    physical, truth = fields(prediction, target, surf, coef_norm)
    return dict(physical_relative_l2_velocity=relative_l2(physical[~surf,:3],truth[~surf,:3]),
                physical_relative_l2_pressure=relative_l2(physical[surf,3],truth[surf,3]),
                normalized_mse_velocity_components=(prediction[~surf,:3]-target[~surf,:3]).square().mean(0),
                normalized_mse_pressure=(prediction[surf,3]-target[surf,3]).square().mean())


def drag_backend():
    path = Path(__file__).resolve().parents[2]/'Car-Design-ShapeNetCar/utils/drag_coefficient.py'
    spec = importlib.util.spec_from_file_location('_linearno_car_drag_primitives',path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def coefficient(sample_directory, pressure, surface_velocity, *, _backend=None):
    """Old drag arithmetic using actual sample directory and surface velocities.

    No released-bug mode, hardcoded param0, file write, global monkey patch or
    resampling. _backend is a test seam for the existing VTK primitive module.
    """
    sample = Path(sample_directory).resolve()
    press_path, velocity_path = sample/'quadpress_smpl.vtk', sample/'hexvelo_smpl.vtk'
    if not press_path.is_file() or not velocity_path.is_file():
        raise FileNotFoundError(f'Car drag requires both raw VTK files under actual sample path: {sample}')
    pressure, velocity = np.asarray(pressure), np.asarray(surface_velocity)
    if pressure.ndim == 1:
        pressure = pressure[:,None]
    if pressure.ndim != 2 or pressure.shape[1] != 1 or velocity.shape != (len(pressure),3):
        raise ValueError('drag requires matched surface pressure [Ns,1] and surface velocity [Ns,3]')
    if not np.isfinite(pressure).all() or not np.isfinite(velocity).all():
        raise ValueError('drag surface fields must be finite')
    backend = drag_backend() if _backend is None else _backend
    mesh = backend.load_unstructured_grid_data(str(press_path))
    if mesh.GetNumberOfPoints() != len(pressure):
        raise ValueError('surface prediction count does not match raw pressure mesh points')
    normals = backend.get_normal(mesh)
    points = backend.vtk_to_numpy(mesh.GetPoints().GetData())
    area = backend.calculate_pos(points)
    cell_areas = backend.calculate_mesh_cell_area(mesh)
    gradients = backend.calculate_cell_velocity_gradient(mesh,velocity)
    data = backend.vtk.vtkDoubleArray()
    data.SetNumberOfComponents(1); data.SetNumberOfTuples(len(pressure)); data.SetName('my_press')
    for i,value in enumerate(pressure):
        data.SetTuple(i,value)
    mesh.GetPointData().AddArray(data)
    converter = backend.vtk.vtkPointDataToCellData(); converter.SetInputData(mesh); converter.Update()
    cell_pressure = backend.vtk_to_numpy(converter.GetOutput().GetCellData().GetArray('my_press'))
    force = backend.calculate_drag_force(cell_areas,normals[:,-1],cell_pressure,gradients[:,-1],np.array(1.8e-5))
    return (2/(((72/3.6)**2)*area*.3))*force


@torch.no_grad()
def drag_pair(data, prediction, coef_norm, sample_directory, *, coefficient_fn=coefficient):
    physical, truth = fields(prediction,data.y,data.surf,coef_norm)
    surf = data.surf
    pred = coefficient_fn(sample_directory,physical[surf,3:4].cpu().numpy(),physical[surf,:3].cpu().numpy())
    true = coefficient_fn(sample_directory,truth[surf,3:4].cpu().numpy(),truth[surf,:3].cpu().numpy())
    return pred, true
