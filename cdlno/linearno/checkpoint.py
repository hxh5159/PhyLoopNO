"""Strict native LinearNO epoch pairs, using hxh atomic storage/RNG utilities.

No legacy loader changes or external whole-object pickle loading. Metadata is
validated before construction; epoch pairs commit before latest/final pointers.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import random

import numpy as np
import torch

from cdlno.training_state import _atomic, _same, _model_state_check, capture_rng, restore_rng
from .schema import read_metadata, validate_metadata, pack_state, unpack_state


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def inspect_checkpoint(directory, selector, constructor):
    directory = Path(directory).resolve()
    if selector in ('latest', 'final'):
        pointer = json.loads((directory/'checkpoints'/f'{selector}.json').read_text())
        name = pointer['manifest']
        if Path(name).name != name or sha256(directory/'checkpoints'/name) != pointer['sha256']:
            raise ValueError('checkpoint pointer checksum/path mismatch')
    else:
        import re
        if not re.fullmatch(r'epoch_[0-9]{4,}', selector or ''):
            raise ValueError('--checkpoint requires final, latest or epoch_XXXX')
        name = selector + '.json'
    manifest_path = directory/'checkpoints'/name
    manifest = json.loads(manifest_path.read_text())
    if manifest.get('format') != 'linearno-epoch-pair-v1':
        raise ValueError('checkpoint family/format mismatch')
    for key, subdir, suffix in (('checkpoint', 'checkpoints', '.pt'), ('weights', 'weights', '.pt'),
                                ('metadata', 'checkpoints', '.metadata.json')):
        path = (directory/manifest[key]['path']).resolve()
        if path.parent != directory/subdir or path.name != manifest_path.stem + suffix:
            raise ValueError('checkpoint pair path mismatch')
        if sha256(path) != manifest[key]['sha256']:
            raise ValueError(f'checkpoint {key} checksum mismatch')
    metadata = read_metadata(directory/manifest['metadata']['path'], constructor=constructor)
    if metadata['resume_state']['epoch'] != manifest['epoch']:
        raise ValueError('metadata/manifest epoch mismatch')
    if selector == 'final' and metadata['resume_state']['checkpoint_role'] != 'final':
        raise ValueError('final evaluation requires final checkpoint')
    return metadata, manifest_path


def read_pair(manifest_path, constructor):
    manifest_path = Path(manifest_path)
    directory = manifest_path.parent.parent
    metadata, _ = inspect_checkpoint(directory, manifest_path.stem, constructor)
    manifest = json.loads(manifest_path.read_text())
    saved = torch.load(directory/manifest['checkpoint']['path'], map_location='cpu', weights_only=True)
    weights = torch.load(directory/manifest['weights']['path'], map_location='cpu', weights_only=True)
    if not isinstance(saved, dict) or set(saved) != {'metadata', 'model'} or saved['metadata'] != metadata:
        raise ValueError('checkpoint metadata/payload mismatch')
    if not _same(saved['model'], weights):
        raise ValueError('checkpoint and independent weights differ')
    return metadata, weights


def strict_load(model, state):
    _model_state_check(state, model)
    model.load_state_dict(state, strict=True)


def resume_state(optimizer, scheduler, epoch, steps_per_epoch, total_epochs, generators, sampler):
    return dict(checkpoint_role='final' if epoch == total_epochs else 'epoch',
        selection_split=None, selection_metric=None, epoch=epoch, global_step=epoch*steps_per_epoch,
        optimizer=pack_state(optimizer.state_dict()), scheduler=pack_state(scheduler.state_dict()),
        rng=pack_state(dict(python=random.getstate(), numpy=np.random.get_state(),
                            torch_cpu=torch.get_rng_state(),
                            torch_cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [])),
        dataloader_generators=pack_state({k: dict(device=str(v.device), state=v.get_state()) for k,v in generators.items()}),
        sampler_state=pack_state(sampler))


def restore_random_state(state, generators):
    saved = unpack_state(state['rng']); numpy_rng = saved['numpy']
    native = dict(python=saved['python'], numpy=dict(algorithm=numpy_rng[0],
        keys=torch.tensor(numpy_rng[1].astype(np.int64)), position=numpy_rng[2],
        has_gauss=numpy_rng[3], cached_gaussian=numpy_rng[4]), cpu=saved['torch_cpu'],
        cuda=saved['torch_cuda'], generators=unpack_state(state['dataloader_generators']))
    restore_rng(native, generators)


def save_pair(directory, model, metadata, constructor):
    directory = Path(directory)
    validate_metadata(metadata, constructor=constructor)
    epoch = metadata['resume_state']['epoch']
    if epoch < 1:
        raise ValueError('only completed epochs can be committed')
    ckdir, wdir = directory/'checkpoints', directory/'weights'
    ckdir.mkdir(exist_ok=True); wdir.mkdir(exist_ok=True)
    name = f'epoch_{epoch:04d}'
    manifest_path = ckdir/(name+'.json')
    state = {k: v.detach().cpu().clone() for k,v in model.state_dict().items()}
    if manifest_path.exists():
        old_metadata, old_state = read_pair(manifest_path, constructor)
        if old_metadata != metadata or not _same(state, old_state):
            raise ValueError('refusing to overwrite a committed epoch')
        return manifest_path
    # A per-run exclusive write lock fails closed on concurrent writers.
    from cdlno.runtime import exclusive_lock
    with exclusive_lock(directory/'.checkpoint.lock'):
        if manifest_path.exists():
            raise ValueError('epoch was concurrently committed')
        committed = [int(p.stem.split('_')[1]) for p in ckdir.glob('epoch_*.json')
                     if '.metadata.' not in p.name]
        if committed and epoch <= max(committed):
            raise ValueError('refusing to roll back committed checkpoint progress')
        paths = dict(checkpoint=ckdir/(name+'.pt'), weights=wdir/(name+'.pt'), metadata=ckdir/(name+'.metadata.json'))
        for path in paths.values():
            if path.exists():
                raise ValueError(f'uncommitted file exists; preserve it for inspection: {path}')
        _atomic(paths['checkpoint'], dict(model=state, metadata=metadata))
        _atomic(paths['weights'], state)
        _atomic(paths['metadata'], metadata, json_file=True)
        record = dict(format='linearno-epoch-pair-v1', epoch=epoch,
                      **{k: dict(path=str(p.relative_to(directory)), sha256=sha256(p)) for k,p in paths.items()})
        _atomic(manifest_path, record, json_file=True)
        pointer = dict(manifest=manifest_path.name, sha256=sha256(manifest_path))
        _atomic(ckdir/'latest.json', pointer, json_file=True)
        if metadata['resume_state']['checkpoint_role'] == 'final':
            _atomic(ckdir/'final.json', pointer, json_file=True)
        _atomic(directory/'model.pt', state)  # hxh convenience weights; verified pair is authoritative
    return manifest_path
