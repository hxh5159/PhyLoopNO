"""L1-only profile resolution. This module does not modify any task parser/factory.

Architecture descriptors are NOT model constructor kwargs. Future task wrappers
must explicitly map them to a real, signature-checked constructor (schema.py).
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
from ._profile_data import CATALOG

FAMILY = 'linearno'
CONFIG_VERSION = 1
PROFILES = ('paper_table8_on_release_model', 'official_release', 'transolver_matched')
DEFAULT_PROFILE = PROFILES[0]
TASKS = ('darcy', 'elasticity', 'airfoil', 'pipe', 'ns', 'plasticity', 'car', 'airfrans')
FIELD_SOURCES = ('cli_explicit', 'profile', 'family_default', 'legacy_default', 'integration_contract')
AIRFRANS_OBJECTIVE_CONTRACT = 'airfrans_transolver_mse_v1'
CAR_OBJECTIVE_CONTRACT = 'car_transolver_mse_v1'
_DEFAULT_CONTRACT = object()
FAMILY_DEFAULTS = {'runtime': {'seed': 0, 'device': 'cpu', 'save_name': None},
                   'model': {'dropout': 0.0, 'activation': 'gelu'}}


def canonical_json(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)


def digest(value):
    return hashlib.sha256(canonical_json(value).encode()).hexdigest()


def integer(value, name, minimum=1):
    if type(value) is not int or value < minimum:
        raise ValueError(f'{name} must be an integer >= {minimum}, got {value!r}')


def finite(value, name, minimum=0, exclusive=False):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f'{name} must be finite numeric, got {value!r}')
    if value < minimum or (exclusive and value == minimum):
        raise ValueError(f'{name} is out of range')


def _leaves(value, prefix=''):
    for key, item in value.items():
        path = f'{prefix}.{key}' if prefix else key
        if isinstance(item, dict) and item:
            yield from _leaves(item, path)
        else:
            yield path, item


def _assign(value, path, item):
    keys = path.split('.')
    for key in keys[:-1]:
        value = value[key]
    value[keys[-1]] = copy.deepcopy(item)


def rank_mapping(task, model):
    rank, dh = model['linearno_rank'], model['hidden'] // model['heads']
    if task == 'car':
        if rank % dh:
            raise ValueError('Car release actual M must equal integer key_ratio * head_dim')
        return dict(original_parameter='key_ratio', original_value=rank // dh,
                    semantics='head_dim_multiplier', head_dim=dh, actual_M=rank)
    return dict(original_parameter='slice_num' if task == 'airfrans' else 'key_ratio',
                original_value=rank, semantics='absolute', actual_M=rank)


def validate_model(task, model, *, N=None):
    if task not in TASKS:
        raise ValueError(f'unknown task: {task}')
    for key in ('hidden', 'heads', 'layers', 'linearno_rank', 'ffn_ratio', 'space_dim', 'out_dim', 'ref'):
        integer(model[key], key)
    integer(model['fun_dim'], 'fun_dim', 0)
    if model['hidden'] % model['heads']:
        raise ValueError('hidden must be divisible by heads')
    variants = ('shapenet',) if task == 'car' else ('airfrans',) if task == 'airfrans' else ('plain', 'temp', 'conv', 'conv_temp')
    if model['linearno_variant'] not in variants:
        raise ValueError(f'{task} requires variant in {variants}')
    for key in ('unified_pos', 'time_input'):
        if type(model[key]) is not bool:
            raise ValueError(f'{key} must be bool')
    finite(model['dropout'], 'dropout')
    if model['dropout'] >= 1 or model['activation'] != 'gelu':
        raise ValueError('release contract requires GELU and 0 <= dropout < 1')
    if task in ('car', 'airfrans') and model['time_input']:
        raise ValueError('industrial forward contract supplies no time condition')
    structured = task != 'airfrans' and (model['linearno_variant'].startswith('conv') or model['unified_pos'])
    for key in ('H', 'W'):
        if structured or model[key] is not None:
            integer(model[key], key)
    if N is not None:
        integer(N, 'N')
        if structured and N != model['H'] * model['W']:
            raise ValueError('structured input requires N = H * W')
    rank_mapping(task, model)


def derived_model(task, model):
    variant = model['linearno_variant']
    conv = variant in ('conv', 'conv_temp')
    active = variant in ('temp', 'conv_temp', 'shapenet')
    return dict(rank_mapping=rank_mapping(task, model), q_softmax_axis='M', k_softmax_axis='N',
                qk_independent=True, projections_shared_across_heads=True, qkv_bias=False,
                attention_scale=None, attention_weight_dropout=False, layernorm_eps=1e-5,
                input_projection='conv2d_each_block' if conv else 'linear', kernel_size=3 if conv else None,
                output_projection='linear_gelu_linear' if conv or task == 'car' else 'linear',
                temperature=dict(initial=0.5 if active or task == 'airfrans' else None,
                                 clamp=[0.1, 2.0] if task == 'car' else [0.01, 1.0] if active else None,
                                 used=active, dead_parameter=task == 'airfrans'),
                grid_range=[[-2, 4], [-1.5, 1.5]] if task == 'airfrans' else [[0, 1], [0, 1]],
                position_mode='append_raw_pos_distances' if task == 'airfrans' else 'replace_coordinates',
                official_unused_flags={'linear': True} if task == 'airfrans' else {'isregular': False} if task == 'car' else {},
                initialization='linear_trunc_normal_0.02;LN_1_0;standard_conv_kaiming_normal;placeholder_after_apply')


def resolve_config(task, profile=DEFAULT_PROFILE, *, explicit=None, legacy_defaults=None,
                   contract=_DEFAULT_CONTRACT):
    """Only explicit dotted overrides are applied. Legacy defaults are audit-only.

    Examples: {'model.linearno_rank': 48, 'training.batch_size': 2}.
    No Namespace is accepted: future adapters must track actual argv presence.
    AirfRANS defaults to the user's reviewed Transolver/release MSE objective.
    Explicit contract=None retains the immutable L0 source interpretation for
    audits and validation of metadata created before that decision.
    """
    if task not in TASKS or profile not in PROFILES:
        raise ValueError(f'unknown task/profile: {task}/{profile}')
    if contract is _DEFAULT_CONTRACT:
        contract = {'airfrans': AIRFRANS_OBJECTIVE_CONTRACT, 'car': CAR_OBJECTIVE_CONTRACT}.get(task)
    catalog = CATALOG
    values = copy.deepcopy(FAMILY_DEFAULTS)
    sources = {p: 'family_default' for p, _ in _leaves(values)}
    selected = copy.deepcopy(catalog['official_release'][task])
    for section, fields in catalog['overrides'].get(profile, {}).get(task, {}).items():
        selected[section].update(fields)
    for section, fields in selected.items():
        values.setdefault(section, {}).update(fields)
    sources.update({p: 'profile' for p, _ in _leaves(selected)})
    known = dict(_leaves(values))
    if contract is not None:
        if not ((contract == 'standard_static_l4' and task in ('darcy', 'elasticity', 'airfoil', 'pipe'))
                or (contract == 'standard_temporal_l5' and task in ('ns', 'plasticity'))
                or (contract == 'car_l7' and task == 'car')
                or (contract == CAR_OBJECTIVE_CONTRACT and task == 'car')
                or (contract == AIRFRANS_OBJECTIVE_CONTRACT and task == 'airfrans')):
            raise ValueError('unknown or inapplicable integration contract')
        # Explicit L4 user decision: mean objective, retaining hxh validation
        # cadence. Preserve the versioned release facts when no contract is used.
        changes = {'objective.reduction': 'mean sample-wise flattened ratios per batch; no epsilon',
                   'training.validation_cadence': 'every epoch', 'runtime.device': 'cuda'}
        if task == 'darcy' and profile != 'official_release':
            changes['training.scheduler_epochs'] = 'resolved training.epochs (hxh)'
        if contract == 'standard_temporal_l5':
            # Preserve temporal objectives: batch sum and original time loops.
            changes = {'runtime.device': 'cuda',
                       'training.scheduler_epochs': 'resolved training.epochs (hxh)'}
        if contract == AIRFRANS_OBJECTIVE_CONTRACT:
            # User decision: use the verified official training objective, not
            # an invented completion of the underspecified paper rL2 recipe.
            # Keep model, training-budget and evaluation axes independent.
            changes = {'objective.' + key: value for key, value in
                       catalog['official_release']['airfrans']['objective'].items()}
        if contract == 'car_l7':
            changes = {'evaluation.force_input': 'hxh_fixed_surface_velocity+real_sample_path'}
        if contract == CAR_OBJECTIVE_CONTRACT:
            # Explicit user decision, separate from the underspecified paper
            # objective. Preserve the former car_l7 contract for old metadata.
            changes = {'objective.' + key: value for key, value in
                       catalog['official_release']['car']['objective'].items()}
            changes['evaluation.force_input'] = 'hxh_fixed_surface_velocity+real_sample_path'
        for path, value in changes.items():
            _assign(values, path, value)
            sources[path] = 'integration_contract'
    for path, value in (explicit or {}).items():
        if path == 'model.qk_dim':
            if 'model.linearno_rank' in explicit:
                raise ValueError('supply only linearno_rank or qk_dim, not both')
            path = 'model.linearno_rank'
        if path not in known:
            raise ValueError(f'unknown LinearNO override: {path}')
        _assign(values, path, value)
        sources[path] = 'cli_explicit'
    validate_model(task, values['model'])
    tr = values['training']
    for name in ('epochs', 'batch_size', 'test_batch_size', 'nmodel', 'subsampling'):
        if name in tr:
            integer(tr[name], name)
    for name in ('lr', 'eps'):
        finite(tr[name], name, exclusive=True)
    finite(tr['weight_decay'], 'weight_decay')
    if tr['gradient_clip'] is not None:
        finite(tr['gradient_clip'], 'gradient_clip', exclusive=True)
    integer(values['runtime']['seed'], 'seed', 0)
    if not isinstance(values['runtime']['device'], str) or not values['runtime']['device']:
        raise ValueError('device must be a nonempty string')
    if values['runtime']['save_name'] is not None and not isinstance(values['runtime']['save_name'], str):
        raise ValueError('save_name must be string or None')
    unresolved = [p for p, v in _leaves(values) if isinstance(v, str) and v.startswith('UNRESOLVED')]
    result = dict(family=FAMILY, config_version=CONFIG_VERSION, task=task, profile=profile,
                  values=values, field_sources=sources, derived_model=derived_model(task, values['model']),
                  unresolved=unresolved, ignored_legacy_defaults=copy.deepcopy(legacy_defaults or {}))
    if contract is not None:
        result['integration_contract'] = contract
    result['config_hash'] = digest(result)
    return result


def validate_resolved(config):
    data = copy.deepcopy(config)
    saved = data.pop('config_hash', None)
    if saved != digest(data):
        raise ValueError('resolved config hash mismatch')
    explicit = {p: dict(_leaves(data['values']))[p] for p, s in data['field_sources'].items() if s == 'cli_explicit'}
    expected = resolve_config(data['task'], data['profile'], explicit=explicit,
                              legacy_defaults=data['ignored_legacy_defaults'], contract=data.get('integration_contract'))
    if config != expected:
        raise ValueError('resolved config differs from versioned profile and explicit overrides')
    return copy.deepcopy(config)


def require_resolved_objective(config):
    validate_resolved(config)
    pending = [p for p in config['unresolved'] if p.startswith('objective.')]
    if pending:
        raise ValueError('paper industrial objective requires reviewed decisions: ' + ', '.join(pending))


def parse_overrides(argv):
    """Standalone parser demonstration ONLY; never added to an existing parser."""
    parser = argparse.ArgumentParser(argument_default=argparse.SUPPRESS, allow_abbrev=False)
    parser.add_argument('--linearno-profile', choices=PROFILES)
    parser.add_argument('--linearno-variant', choices=('plain', 'temp', 'conv', 'conv_temp', 'shapenet', 'airfrans'))
    for name in ('rank', 'qk-dim', 'hidden', 'heads', 'layers', 'ffn-ratio', 'batch-size', 'epochs', 'seed'):
        parser.add_argument('--linearno-' + name, type=int)
    for name in ('device', 'save-name'):
        parser.add_argument('--linearno-' + name)
    args = vars(parser.parse_args(argv))
    profile = args.pop('linearno_profile', DEFAULT_PROFILE)
    rename = {'rank': 'model.linearno_rank', 'qk_dim': 'model.qk_dim', 'variant': 'model.linearno_variant',
              'batch_size': 'training.batch_size', 'epochs': 'training.epochs', 'seed': 'runtime.seed',
              'device': 'runtime.device', 'save_name': 'runtime.save_name'}
    return profile, {rename.get(k[9:], 'model.' + k[9:]): v for k, v in args.items()}
