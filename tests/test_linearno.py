"""Portable LinearNO baseline and checkpoint checks, independent of old audits."""
import copy
import hashlib
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'PDE-Solving-StandardBenchmark'))

from model.LinearNO import Model
from cdlno.linearno.attention import LinearNOAttention, initialize_release_weights
from cdlno.linearno.airfrans import AirfRANSLinearNO
from cdlno.linearno.shapenet import ShapeNetLinearNO
from cdlno.linearno.profiles import TASKS, PROFILES, resolve_config, validate_resolved
from cdlno.linearno.standard_entry import constructor_kwargs
from cdlno.linearno.checkpoint import (resume_state, restore_random_state, save_pair,
                                       read_pair, inspect_checkpoint, strict_load)
from cdlno.linearno.schema import make_metadata, validate_metadata, unpack_state
from cdlno.training_state import _same
from cdlno.runtime import exclusive_lock


@pytest.mark.parametrize('task', TASKS)
def test_eight_task_profiles_and_small_model_backward(task):
    for profile in PROFILES:
        validate_resolved(resolve_config(task, profile))
    config = resolve_config(task, 'official_release')
    if task in ('airfrans', 'car'):
        cls = AirfRANSLinearNO if task == 'airfrans' else ShapeNetLinearNO
        kwargs = dict(n_hidden=16, n_head=4, n_layers=2, mlp_ratio=1,
                      linearno_rank=8, ref=2)
        model = cls(**kwargs)
        data = SimpleNamespace(x=torch.randn(15, 7), pos=torch.randn(15, 2))
        output = model(data if task == 'airfrans' else (data, None))
        assert output.shape == (15, 4)
    else:
        kwargs = constructor_kwargs(config)
        kwargs.update(n_hidden=16, n_head=4, n_layers=2, mlp_ratio=1,
                      linearno_rank=8, H=3, W=5, ref=2)
        model = Model(**kwargs)
        x = torch.randn(2, 15, kwargs['space_dim'])
        fx = torch.randn(2, 15, kwargs['fun_dim']) if kwargs['fun_dim'] else None
        output = model(x, fx, T=torch.rand(2, 1) if kwargs['Time_Input'] else None)
        assert output.shape == (2, 15, kwargs['out_dim'])
    assert torch.isfinite(output).all()
    output.square().mean().backward()
    assert model.blocks[0].Attn.to_q.weight.grad is not None
    assert model.blocks[0].Attn.to_k.weight.grad is not None
    assert model.blocks[0].Attn.to_q.weight is not model.blocks[0].Attn.to_k.weight


@pytest.mark.parametrize('variant', ('plain', 'temp', 'conv', 'conv_temp', 'airfrans', 'shapenet'))
def test_factorization_matches_explicit_dense_mixing_and_gradients(variant):
    options = dict(dim=12, heads=3, dim_head=4, rank=8, variant=variant)
    if variant in ('conv', 'conv_temp'):
        options.update(H=3, W=5)
    module = LinearNOAttention(**options).double()
    module.apply(initialize_release_weights)
    x = torch.randn(2, 15, 12, dtype=torch.float64, requires_grad=True)
    projected = module.in_project_x(x.transpose(1, 2).reshape(2, 12, 3, 5)
                                    if variant in ('conv', 'conv_temp') else x)
    if variant in ('conv', 'conv_temp'):
        features = projected.reshape(2, 3, 4, 15).transpose(-1, -2)
    else:
        features = projected.reshape(2, 15, 3, 4).transpose(1, 2)
    q, k, v = module.to_q(features), module.to_k(features), module.to_v(features)
    if variant in ('temp', 'conv_temp'):
        q, k = q / module.temperature_q.clamp(.01, 1), k / module.temperature_k.clamp(.01, 1)
    elif variant == 'shapenet':
        q, k = q / module.tempreature_q.clamp(.1, 2), k / module.tempreature_k.clamp(.1, 2)
    q, k = q.softmax(-1), k.softmax(-2)
    mixing = q @ k.transpose(-1, -2)
    torch.testing.assert_close(mixing.sum(-1), torch.ones_like(mixing[..., 0]), atol=1e-12, rtol=1e-12)
    expected = module.to_out((mixing @ v).transpose(1, 2).reshape(2, 15, 12))
    actual = module(x)
    torch.testing.assert_close(actual, expected, atol=1e-11, rtol=1e-10)
    actual_gradient = torch.autograd.grad(actual.square().sum(), x, retain_graph=True)[0]
    dense_gradient = torch.autograd.grad(expected.square().sum(), x)[0]
    torch.testing.assert_close(actual_gradient, dense_gradient, atol=1e-11, rtol=1e-10)


def _step(model, optimizer, scheduler, x, y):
    optimizer.zero_grad()
    (model(x, None) - y).square().mean().backward()
    optimizer.step()
    scheduler.step()


def test_strict_checkpoint_resume_preserves_next_update_and_rejects_tampering(tmp_path):
    kwargs = dict(space_dim=2, fun_dim=0, out_dim=1, n_hidden=16, n_head=4,
                  n_layers=2, linearno_rank=8, linearno_variant='temp')
    model = Model(**kwargs)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=2)
    x, y = torch.randn(2, 7, 2), torch.randn(2, 7, 1)
    _step(model, optimizer, scheduler, x, y)
    config = resolve_config('elasticity', 'official_release')
    data_hash = hashlib.sha256(b'labelled synthetic checkpoint test').hexdigest()
    from cdlno.provenance import provenance
    metadata = make_metadata(
        profile_spec=config, constructor=Model,
        model_spec=dict(class_path='model.LinearNO.Model', constructor_kwargs=kwargs),
        data_spec=dict(split='synthetic only', sampling='fixed tiny arrays', checksums={'synthetic': data_hash}),
        objective_spec=config['values']['objective'], evaluation_spec=config['values']['evaluation'],
        provenance_spec=provenance(), normalizer_spec=dict(policy='none', records={}),
        resume_state=resume_state(optimizer, scheduler, 1, 1, 2, {}, {}), ensemble_manifest=[])
    manifest = save_pair(tmp_path, model, metadata, Model)
    loaded_metadata, weights = read_pair(manifest, Model)
    assert loaded_metadata == metadata
    # A source archive has truthful null Git identity and still validates.
    without_git = copy.deepcopy(metadata)
    without_git['provenance_spec'].update(target_sha=None, base_commit=None, dirty=None)
    from cdlno.linearno.profiles import digest
    without_git['metadata_hash'] = digest({k: v for k, v in without_git.items() if k != 'metadata_hash'})
    validate_metadata(without_git, constructor=Model)
    resumed = Model(**kwargs)
    strict_load(resumed, weights)
    restored_optimizer = torch.optim.AdamW(resumed.parameters(), lr=.001)
    restored_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(restored_optimizer, T_max=2)
    restored_optimizer.load_state_dict(unpack_state(metadata['resume_state']['optimizer']))
    restored_scheduler.load_state_dict(unpack_state(metadata['resume_state']['scheduler']))
    restore_random_state(metadata['resume_state'], {})
    _step(model, optimizer, scheduler, x, y)
    _step(resumed, restored_optimizer, restored_scheduler, x, y)
    assert _same(model.state_dict(), resumed.state_dict())
    assert _same(optimizer.state_dict(), restored_optimizer.state_dict())
    bad_weights = dict(weights)
    bad_weights.pop(next(iter(bad_weights)))
    with pytest.raises(ValueError, match='keys differ'):
        strict_load(resumed, bad_weights)
    weight_file = tmp_path / 'weights' / 'epoch_0001.pt'
    weight_file.write_bytes(weight_file.read_bytes() + b'tampered')
    with pytest.raises(ValueError, match='checksum'):
        inspect_checkpoint(tmp_path, 'latest', Model)


def test_source_provenance_without_development_git_history(tmp_path):
    import cdlno.provenance as source
    (tmp_path / 'cdlno').mkdir()
    script = tmp_path / 'cdlno' / 'model.py'
    script.write_text('VALUE = 1\n', encoding='utf-8')
    with patch.object(source, 'ROOT', tmp_path):
        assert source.code_state() == dict(commit=None, dirty=None)
        first, normalized = source.source_manifest()
        script.write_text('# a comment\nVALUE = 1\n', encoding='utf-8')
        second, same_program = source.source_manifest()
        assert first != second and normalized == same_program
        script.write_text('VALUE = 2\n', encoding='utf-8')
        assert source.source_manifest()[1] != normalized


def test_exclusive_lock_contends_across_processes_and_releases(tmp_path):
    lock = tmp_path / 'writer.lock'
    command = [sys.executable, '-c',
               'import sys; from cdlno.runtime import exclusive_lock; '
               '\nwith exclusive_lock(sys.argv[1]): print("acquired")', str(lock)]
    with exclusive_lock(lock):
        result = subprocess.run(command, cwd=ROOT, text=True, capture_output=True)
        assert result.returncode != 0
    result = subprocess.run(command, cwd=ROOT, text=True, capture_output=True)
    assert result.returncode == 0 and 'acquired' in result.stdout
