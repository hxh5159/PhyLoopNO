"""Shared strict-state, RNG and atomic-write helpers for training checkpoints."""
from __future__ import annotations

from collections import OrderedDict
from contextlib import contextmanager
from dataclasses import dataclass
import hashlib
from importlib import metadata
import json
import math
import os
from pathlib import Path
import random
import subprocess
import sys
import tempfile
from typing import Mapping

import numpy as np
import torch



VERSION = 1
TASKS = ('darcy', 'elasticity', 'airfoil', 'pipe', 'ns', 'plasticity', 'car', 'airfrans')


class ResumeError(ValueError):
    """Incomplete archive or a mismatch in the requested continuation."""


def _integer(value, name, minimum=0):
    if type(value) is not int or value < minimum:
        raise ResumeError(f'{name} must be an integer >= {minimum}')


def _json_copy(value):
    return json.loads(json.dumps(value, allow_nan=False, sort_keys=True))


def _cpu(value):
    """Owned snapshots: no live parameter, optimizer, or NumPy storage aliases."""
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, np.ndarray):
        return torch.from_numpy(value.copy())
    if isinstance(value, np.generic):
        return _cpu(value.item())
    if isinstance(value, dict):
        out = OrderedDict() if isinstance(value, OrderedDict) else {}
        for key, item in value.items():
            if type(key) not in (str, int):
                raise ResumeError('state mapping keys must be str/int')
            out[key] = _cpu(item)
        if hasattr(value, '_metadata'):
            out._metadata = _cpu(value._metadata)
        return out
    if isinstance(value, (tuple, list)):
        return type(value)(_cpu(item) for item in value)
    if value is None or type(value) in (str, bool, int, float):
        if isinstance(value, float) and not math.isfinite(value):
            raise ResumeError('nonfinite scalar in training state')
        return value
    raise ResumeError(f'unsupported state type: {type(value).__name__}')


def _same(a, b):
    if isinstance(a, torch.Tensor) or isinstance(b, torch.Tensor):
        return (isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor)
                and a.dtype == b.dtype and a.shape == b.shape and torch.equal(a.cpu(), b.cpu()))
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(_same(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        return type(a) is type(b) and len(a) == len(b) and all(_same(x, y) for x, y in zip(a, b))
    return type(a) is type(b) and a == b


def ordered_fingerprint(items):
    """Order-sensitive digest of ALREADY loaded identifiers/arrays (no data IO)."""
    digest = hashlib.sha256()
    def visit(value):
        value = _cpu(value)
        if isinstance(value, torch.Tensor):
            if value.layout != torch.strided:
                raise ResumeError('fingerprint requires dense tensors')
            header = json.dumps([str(value.dtype), list(value.shape)]).encode()
            data = value.contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()
            digest.update(b'tensor' + str(len(header)).encode() + b':' + header)
            digest.update(str(len(data)).encode() + b':' + data)
        elif isinstance(value, dict):
            digest.update(b'dict{')
            for key in sorted(value, key=lambda k: (type(k).__name__, str(k))):
                visit(key); visit(value[key])
            digest.update(b'}')
        elif isinstance(value, (list, tuple)):
            digest.update(type(value).__name__.encode() + b'[')
            for item in value:
                visit(item)
            digest.update(b']')
        else:
            encoded = json.dumps(value, allow_nan=False).encode()
            digest.update(type(value).__name__.encode() + str(len(encoded)).encode() + b':' + encoded)
    for item in items:
        visit(item)
    return digest.hexdigest()


@dataclass(frozen=True)
class TrainingProtocol:
    """Explicit task facts, supplied by future entry adapters, never inferred.

    Counts are per completed epoch. This handles Plasticity's 20 updates per
    batch without changing its scheduler rate. training/data/adapter must be
    JSON mappings; normalizer tensors live separately in the archive.
    """
    task: str
    total_epochs: int
    updates_per_epoch: int
    scheduler_steps_per_epoch: int
    training: dict
    data: dict
    adapter: dict

    def __post_init__(self):
        if self.task not in TASKS:
            raise ResumeError(f'unsupported task: {self.task}')
        for name in ('total_epochs', 'updates_per_epoch', 'scheduler_steps_per_epoch'):
            _integer(getattr(self, name), name, 1)
        for name in ('training', 'data', 'adapter'):
            if not isinstance(getattr(self, name), dict):
                raise ResumeError(f'{name} must be an explicit mapping')
            object.__setattr__(self, name, _json_copy(getattr(self, name)))
        if not {'optimizer', 'scheduler', 'loss', 'batch_size'} <= self.training.keys():
            raise ResumeError('training requires optimizer/scheduler/loss/batch_size')
        _integer(self.training['batch_size'], 'batch_size', 1)
        if not {'split', 'ordered_fingerprint'} <= self.data.keys():
            raise ResumeError('data requires split and ordered_fingerprint')
        fingerprint = self.data['ordered_fingerprint']
        if not isinstance(fingerprint, str) or len(fingerprint) != 64 or any(c not in '0123456789abcdef' for c in fingerprint):
            raise ResumeError('data.ordered_fingerprint must be a SHA256 digest')

    def to_dict(self):
        return _json_copy(dict(version=VERSION, task=self.task, total_epochs=self.total_epochs,
            updates_per_epoch=self.updates_per_epoch, scheduler_steps_per_epoch=self.scheduler_steps_per_epoch,
            training=self.training, data=self.data, adapter=self.adapter))


def capture_rng(generators: Mapping[str, torch.Generator] | None = None):
    numpy_state = np.random.get_state()
    return dict(python=random.getstate(), numpy=dict(algorithm=numpy_state[0],
        keys=torch.tensor(numpy_state[1].astype(np.int64)), position=int(numpy_state[2]),
        has_gauss=int(numpy_state[3]), cached_gaussian=float(numpy_state[4])),
        cpu=torch.get_rng_state().clone(),
        cuda=[v.clone() for v in torch.cuda.get_rng_state_all()] if torch.cuda.is_available() else [],
        generators={k: dict(device=str(g.device), state=g.get_state().clone()) for k, g in (generators or {}).items()})


def _validate_rng(state, generators):
    if not isinstance(state, dict) or set(state) != {'python', 'numpy', 'cpu', 'cuda', 'generators'}:
        raise ResumeError('missing/incompatible RNG state')
    try:
        random.Random().setstate(state['python'])
        n = state['numpy']
        np.random.RandomState().set_state((n['algorithm'], n['keys'].numpy().astype(np.uint32),
                                           n['position'], n['has_gauss'], n['cached_gaussian']))
        torch.Generator().set_state(state['cpu'])
        expected = torch.cuda.device_count() if torch.cuda.is_available() else 0
        if len(state['cuda']) != expected:
            raise ResumeError('CUDA RNG device count differs; exact continuation requires the same visible devices')
        for i, value in enumerate(state['cuda']):
            torch.Generator(device=f'cuda:{i}').set_state(value)
        if state['generators'].keys() != generators.keys():
            raise ResumeError('DataLoader/Sampler generator names differ')
        for name, g in generators.items():
            saved = state['generators'][name]
            if str(g.device) != saved['device']:
                raise ResumeError(f'generator device differs: {name}')
            torch.Generator(device=g.device).set_state(saved['state'])
    except ResumeError:
        raise
    except (KeyError, TypeError, ValueError, RuntimeError, AttributeError) as e:
        raise ResumeError(f'invalid RNG state: {e}') from e


def restore_rng(state, generators: Mapping[str, torch.Generator] | None = None):
    generators = dict(generators or {})
    _validate_rng(state, generators)
    random.setstate(state['python'])
    n = state['numpy']
    np.random.set_state((n['algorithm'], n['keys'].numpy().astype(np.uint32),
                         n['position'], n['has_gauss'], n['cached_gaussian']))
    torch.set_rng_state(state['cpu'])
    if state['cuda']:
        torch.cuda.set_rng_state_all(state['cuda'])
    for name, generator in generators.items():
        generator.set_state(state['generators'][name]['state'])


@contextmanager
def isolated_evaluation(model, *, generators=None):
    """Restore RNG, each module's train/eval flag and buffers, also on exceptions.

    Callbacks must only infer/render, never update parameters or call optimizer.
    The context does not modify gradients, parameters or model forward methods.
    """
    rng = capture_rng(generators)
    flags = [(module, module.training) for module in model.modules()]
    buffers = [(buffer, buffer.detach().clone()) for buffer in model.buffers()]
    try:
        model.eval()
        with torch.no_grad():
            yield
    finally:
        with torch.no_grad():
            for buffer, value in buffers:
                buffer.copy_(value)
        for module, training in flags:
            module.training = training
        restore_rng(rng, generators)


def _qualified(obj):
    cls = type(obj)
    return cls.__module__ + '.' + cls.__qualname__


def _optimizer_signature(model, optimizer):
    if type(optimizer) not in (torch.optim.Adam, torch.optim.AdamW):
        raise ResumeError('epoch continuation currently supports original Adam/AdamW only')
    names = {id(p): name for name, p in model.named_parameters()}
    seen, groups = set(), []
    for group in optimizer.param_groups:
        keys = []
        for p in group['params']:
            if id(p) not in names or id(p) in seen:
                raise ResumeError('optimizer has foreign or repeated parameters')
            seen.add(id(p)); keys.append(names[id(p)])
        groups.append(dict(names=keys, options=_cpu({k: v for k, v in group.items()
                      if k not in ('params', 'lr', 'betas', 'momentum')})))
    return dict(type=_qualified(optimizer), defaults=_cpu(optimizer.defaults), groups=groups)


def _scheduler_signature(scheduler):
    if type(scheduler) not in (torch.optim.lr_scheduler.OneCycleLR, torch.optim.lr_scheduler.CosineAnnealingLR):
        raise ResumeError('epoch continuation supports original OneCycleLR/CosineAnnealingLR only')
    mutable = {'last_epoch', '_step_count', '_last_lr', '_get_lr_called_within_step', '_is_initial'}
    return dict(type=_qualified(scheduler), plan=_cpu({k: v for k, v in scheduler.state_dict().items() if k not in mutable}))


def _file_sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def _fsync_dir(path):
    # Windows does not expose directory fsync through os.open.
    if os.name == 'nt':
        return
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _atomic(path, value, *, json_file=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix='.' + path.name + '.', dir=path.parent)
    try:
        with os.fdopen(fd, 'wb') as f:
            if json_file:
                f.write((json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n').encode())
            else:
                torch.save(value, f)
            f.flush(); os.fsync(f.fileno())
        os.replace(temporary, path)
        _fsync_dir(path.parent)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _model_state_check(saved, model):
    expected = model.state_dict()
    if not isinstance(saved, dict) or saved.keys() != expected.keys():
        raise ResumeError('model state_dict keys differ (strict loading required)')
    for name, value in expected.items():
        other = saved[name]
        if not isinstance(other, torch.Tensor) or other.shape != value.shape or other.dtype != value.dtype:
            raise ResumeError(f'model shape/dtype differs: {name}')
        if other.is_floating_point() and not torch.isfinite(other).all():
            raise ResumeError(f'nonfinite model weight: {name}')


def _environment(model):
    packages = {}
    for name in ('numpy', 'matplotlib', 'torch-geometric'):
        try:
            packages[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            packages[name] = None
    return dict(python=sys.version.split()[0], torch=str(torch.__version__), cuda=torch.version.cuda,
        packages=packages,
        devices=sorted({str(p.device) for p in model.parameters()}),
        dtypes=sorted({str(p.dtype) for p in model.parameters()}),
        deterministic=torch.are_deterministic_algorithms_enabled(),
        matmul_tf32=torch.backends.cuda.matmul.allow_tf32, cudnn_tf32=torch.backends.cudnn.allow_tf32,
        cudnn_benchmark=torch.backends.cudnn.benchmark, cudnn_deterministic=torch.backends.cudnn.deterministic,
        float32_matmul_precision=torch.get_float32_matmul_precision(),
        sdpa_enabled=dict(math=torch.backends.cuda.math_sdp_enabled(), flash=torch.backends.cuda.flash_sdp_enabled(),
                          memory_efficient=torch.backends.cuda.mem_efficient_sdp_enabled(), cudnn=torch.backends.cuda.cudnn_sdp_enabled()),
        source=_source_provenance(model))


def _source_provenance(model):
    # Hash the actual shared code and wrapper modules, including dirty edits.
    # Task adapters additionally supply their loader/entry provenance in extra.
    root = Path(__file__).resolve().parents[1]
    paths = set(Path(__file__).resolve().parent.glob('*.py'))
    for module in model.modules():
        name = type(module).__module__
        filename = getattr(sys.modules.get(name), '__file__', None)
        if filename and not name.startswith('torch.') and Path(filename).is_file():
            paths.add(Path(filename).resolve())
    hashes = {str(p.relative_to(root) if p.is_relative_to(root) else p): _file_sha(p) for p in sorted(paths)}
    try:
        result = subprocess.run(['git', '-C', str(root), 'rev-parse', 'HEAD'],
                                capture_output=True, text=True, timeout=5, check=False)
        head = result.stdout.strip() if result.returncode == 0 else None
    except (OSError, subprocess.TimeoutExpired):
        head = None
    return dict(git_head=head, shared_and_model_source_sha256=hashes)
