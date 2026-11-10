"""PhyLoopNO archive aliases and shared optimizer-state validation."""
import torch
from .v5.checkpoint import *

def validate_optimizer_state(saved, optimizer):
    if set(saved) != {'state', 'param_groups'} or len(saved['param_groups']) != len(optimizer.param_groups):
        raise ValueError('resume optimizer parameter groups mismatch')
    ids = []
    for (group, current) in zip(saved['param_groups'], optimizer.param_groups):
        if len(group['params']) != len(current['params']):
            raise ValueError('resume optimizer group size mismatch')
        for (index, p) in zip(group['params'], current['params']):
            if type(index) is not int or index in ids:
                raise ValueError('resume optimizer duplicate/invalid parameter id')
            ids.append(index)
            state = saved['state'].get(index, {})
            if state and set(state) != {'step', 'exp_avg', 'exp_avg_sq', *(['max_exp_avg_sq'] if group['amsgrad'] else [])}:
                raise ValueError('resume optimizer moment keys mismatch')
            for (name, value) in state.items():
                if not isinstance(value, torch.Tensor) or not torch.isfinite(value).all():
                    raise ValueError('nonfinite/invalid optimizer state')
                if name == 'step':
                    if value.ndim != 0 or value.item() < 0:
                        raise ValueError('optimizer step must be nonnegative scalar')
                elif value.shape != p.shape or value.dtype != p.dtype:
                    raise ValueError('resume optimizer moment shape/dtype mismatch')
    if set(saved['state']) - set(ids):
        raise ValueError('resume optimizer contains unknown parameter ids')
