"""LinearNO metadata protocol, not a checkpoint loader or model factory.

No imports of task entries, pickle loading, model construction, or legacy writes.
Metadata-only validation does not certify a runnable model. A future loader MUST
supply the real constructor to validate_model_spec before construction, then load
all weights with strict=True. Published checkpoints contain plain tensor state;
external Python object conversion is outside this protocol.
"""
from __future__ import annotations

import base64
import copy
import hashlib
import inspect
import json
import math
from pathlib import Path
import re

from .profiles import CONFIG_VERSION, FAMILY, canonical_json, digest, integer, validate_resolved

SCHEMA_VERSION = 1
SECTIONS = ('model_spec', 'profile_spec', 'data_spec', 'objective_spec', 'evaluation_spec',
            'provenance_spec', 'normalizer_spec', 'resume_state', 'ensemble_manifest')
EVALUATION_AXES = ('prediction_sampling', 'field_metric', 'force_metric', 'force_input', 'split', 'aggregation')
RESUME_FIELDS = ('checkpoint_role', 'selection_split', 'selection_metric', 'epoch', 'global_step',
                 'optimizer', 'scheduler', 'rng', 'dataloader_generators', 'sampler_state')
PROVENANCE_FIELDS = ('target_sha', 'transolver_sha', 'linearno_sha', 'paper_version', 'paper_sha256',
                     'base_commit', 'dirty', 'source_sha256', 'normalized_patch_sha256',
                     'command', 'environment')


def _keys(value, required, name, *, exact=False):
    if not isinstance(value, dict):
        raise ValueError(f'{name} must be a mapping')
    missing = set(required) - value.keys()
    extra = value.keys() - set(required) if exact else set()
    if missing or extra:
        raise ValueError(f'{name}: missing={sorted(missing)}, unknown={sorted(extra)}')


def _hash(value, name, length=64):
    if not isinstance(value, str) or re.fullmatch(r'[0-9a-f]{%d}' % length, value) is None:
        raise ValueError(f'{name} must be a {length}-digit lowercase hash')


def checkpoint_family(metadata):
    """Missing family stays legacy; do not infer identity from arbitrary keys."""
    if not isinstance(metadata, dict) or 'family' not in metadata:
        return None
    return metadata['family']


def validate_model_spec(spec, constructor=None):
    _keys(spec, ('class_path', 'constructor_kwargs'), 'model_spec', exact=True)
    path, kwargs = spec['class_path'], spec['constructor_kwargs']
    if not isinstance(path, str) or '.' not in path or not all(s.isidentifier() for s in path.split('.')):
        raise ValueError('model_spec.class_path must be a dotted class path')
    if not isinstance(kwargs, dict) or not all(isinstance(k, str) for k in kwargs):
        raise ValueError('constructor_kwargs must be a string-keyed mapping')
    canonical_json(kwargs)
    if constructor is not None:
        if path != constructor.__module__ + '.' + constructor.__qualname__:
            raise ValueError('constructor class_path mismatch')
        signature = inspect.signature(constructor)
        # **kwargs cannot be used to conceal arbitrary profile/runtime fields.
        allowed = {k for k, p in signature.parameters.items() if p.kind in
                   (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)}
        if set(kwargs) - allowed:
            raise ValueError(f'unknown constructor fields: {sorted(set(kwargs) - allowed)}')
        try:
            signature.bind(**kwargs)
        except TypeError as error:
            raise ValueError(f'constructor arguments: {error}') from error
    return copy.deepcopy(spec)


def validate_release_descriptor(spec, config):
    """Validate the fixed official signatures WITHOUT importing reference code.

    L1 has no target LinearNO classes. Metadata without a supplied real constructor
    must identify an audited official descriptor; unknown target paths fail closed.
    Standard's args Namespace is represented by its JSON model field, not imported.
    """
    task, model = config['task'], config['values']['model']
    prefix = 'models' if task in ('car', 'airfrans') else 'model'
    expected_path = prefix + '.LinearAttnNeuralOperator.LinearAttentionNeuralOperator'
    if spec['class_path'] != expected_path:
        raise ValueError('unrecognized constructor descriptor; supply the actual constructor for validation')
    names = dict(space_dim='space_dim', n_layers='layers', n_hidden='hidden', dropout='dropout',
                 n_head='heads', act='activation', mlp_ratio='ffn_ratio', fun_dim='fun_dim',
                 out_dim='out_dim', ref='ref', unified_pos='unified_pos')
    expected = {key: model[field] for key, field in names.items()}
    if task == 'airfrans':
        expected.update(slice_num=model['linearno_rank'], linear=True)
    else:
        expected.update(Time_Input=model['time_input'], H=model['H'], W=model['W'],
                        key_ratio=config['derived_model']['rank_mapping']['original_value'])
        if task == 'car':
            expected['isregular'] = False
        else:
            variant = model['linearno_variant']
            expected['args'] = {'model': 'no_temp' if variant == 'plain' else variant}
    _keys(spec['constructor_kwargs'], expected, 'official constructor_kwargs', exact=True)
    if spec['constructor_kwargs'] != expected:
        raise ValueError('structural constructor fields conflict with resolved profile')


def numerical_state(value):
    """Lossless CPU bytes, including scalar/bfloat16 tensors; no pickle required."""
    import numpy as np
    import torch
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().contiguous()
        if value.layout != torch.strided or value.is_quantized:
            raise ValueError('numerical state requires dense nonquantized tensors')
        raw = value.reshape(-1).view(torch.uint8).numpy().tobytes()
        backend, dtype, shape = 'torch', str(value.dtype).split('.')[-1], list(value.shape)
    else:
        value = np.asarray(value)
        if value.dtype.kind not in 'biufc':
            raise ValueError('numerical state requires numeric/bool dtype, never object')
        raw = value.tobytes(order='C')
        backend, dtype, shape = 'numpy', value.dtype.str, list(value.shape)
    result = dict(backend=backend, dtype=dtype, shape=shape, encoding='base64',
                  data=base64.b64encode(raw).decode('ascii'))
    result['sha256'] = digest(result)
    return result


def restore_numerical_state(spec):
    import numpy as np
    import torch
    _keys(spec, ('backend', 'dtype', 'shape', 'encoding', 'data', 'sha256'), 'numerical state', exact=True)
    payload = {k: v for k, v in spec.items() if k != 'sha256'}
    if spec['sha256'] != digest(payload) or spec['encoding'] != 'base64':
        raise ValueError('numerical state checksum/encoding mismatch')
    if not isinstance(spec['shape'], list):
        raise ValueError('numerical state shape must be list')
    for size in spec['shape']:
        integer(size, 'shape dimension', 0)
    try:
        raw = base64.b64decode(spec['data'], validate=True)
    except Exception as error:
        raise ValueError('invalid numerical base64') from error
    numel = math.prod(spec['shape'])
    if spec['backend'] == 'torch':
        allowed = {str(d).split('.')[-1]: d for d in
                   (torch.bool, torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64,
                    torch.float16, torch.bfloat16, torch.float32, torch.float64, torch.complex64, torch.complex128)}
        if spec['dtype'] not in allowed:
            raise ValueError('unsupported torch numerical dtype')
        dtype = allowed[spec['dtype']]
        if len(raw) != numel * torch.empty((), dtype=dtype).element_size():
            raise ValueError('numerical byte size does not match shape/dtype')
        return (torch.frombuffer(bytearray(raw), dtype=dtype).clone() if raw else
                torch.empty(0, dtype=dtype)).reshape(spec['shape'])
    if spec['backend'] != 'numpy':
        raise ValueError('unknown numerical backend')
    try:
        dtype = np.dtype(spec['dtype'])
    except TypeError as error:
        raise ValueError('invalid numpy dtype') from error
    if dtype.kind not in 'biufc' or len(raw) != numel * dtype.itemsize:
        raise ValueError('numerical byte size/dtype mismatch')
    return np.frombuffer(raw, dtype=dtype).copy().reshape(spec['shape'])


def pack_state(value):
    """Tagged JSON tree preserving tuples, integer optimizer keys and RNG tensors."""
    import numpy as np
    import torch
    if isinstance(value, (torch.Tensor, np.ndarray, np.generic)):
        return {'type': 'numeric', 'value': numerical_state(value)}
    if isinstance(value, dict):
        return {'type': 'dict', 'value': [[pack_state(k), pack_state(v)] for k, v in value.items()]}
    if isinstance(value, (list, tuple)):
        return {'type': 'tuple' if isinstance(value, tuple) else 'list', 'value': [pack_state(v) for v in value]}
    if value is None or type(value) in (bool, int, float, str):
        canonical_json(value)
        return {'type': 'scalar', 'value': value}
    raise ValueError(f'unsupported checkpoint state type: {type(value).__name__}')


def unpack_state(value):
    _keys(value, ('type', 'value'), 'tagged state', exact=True)
    kind, data = value['type'], value['value']
    if kind == 'numeric':
        return restore_numerical_state(data)
    if kind == 'scalar':
        if data is not None and type(data) not in (bool, int, float, str):
            raise ValueError('invalid scalar state')
        canonical_json(data)
        return data
    if kind in ('tuple', 'list'):
        if not isinstance(data, list):
            raise ValueError('invalid sequence state')
        items = [unpack_state(v) for v in data]
        return tuple(items) if kind == 'tuple' else items
    if kind == 'dict':
        if not isinstance(data, list):
            raise ValueError('invalid mapping state')
        result = {}
        for pair in data:
            if not isinstance(pair, list) or len(pair) != 2:
                raise ValueError('invalid mapping item')
            key, item = unpack_state(pair[0]), unpack_state(pair[1])
            if type(key) not in (str, int):
                raise ValueError('checkpoint state keys must be string or integer')
            if key in result:
                raise ValueError('duplicate checkpoint state key')
            result[key] = item
        return result
    raise ValueError('unknown checkpoint state type')


def normalizer_record(named_states, *, fit_split, data_checksum, algorithm):
    """Names such as input.mean/output.std or coef_norm.mean_in must be explicit."""
    if not named_states or not all(isinstance(k, str) and k for k in named_states):
        raise ValueError('normalizer requires nonempty named numerical states')
    return dict(algorithm=algorithm, fit_split=fit_split, data_checksum=data_checksum,
                states={k: numerical_state(v) for k, v in named_states.items()})


def _validate_normalizers(spec):
    _keys(spec, ('policy', 'records'), 'normalizer_spec', exact=True)
    if spec['policy'] not in ('none', 'saved_train_fit') or not isinstance(spec['records'], dict):
        raise ValueError('normalizer policy/records invalid')
    if bool(spec['records']) != (spec['policy'] == 'saved_train_fit'):
        raise ValueError('normalizer policy does not match records')
    for name, record in spec['records'].items():
        if not isinstance(name, str) or not name:
            raise ValueError('normalizer name required')
        _keys(record, ('algorithm', 'fit_split', 'data_checksum', 'states'), 'normalizer record', exact=True)
        if not record['algorithm'] or not record['fit_split'] or not record['states']:
            raise ValueError('normalizer fit provenance and states required')
        _hash(record['data_checksum'], 'normalizer data checksum')
        for state_name, state in record['states'].items():
            if not isinstance(state_name, str) or not state_name:
                raise ValueError('normalizer state names required')
            restore_numerical_state(state)


def _validate_resume(state):
    import random
    import numpy as np
    import torch
    _keys(state, RESUME_FIELDS, 'resume_state')
    if set(state) - set(RESUME_FIELDS) - {'scaler_state'}:
        raise ValueError('unknown resume_state fields')
    if state['checkpoint_role'] not in ('final', 'epoch', 'best_validation'):
        raise ValueError('invalid checkpoint_role')
    if state['checkpoint_role'] == 'best_validation':
        if state['selection_split'] != 'validation' or not state['selection_metric']:
            raise ValueError('best checkpoint requires validation-only selection and metric')
    elif state['selection_split'] is not None or state['selection_metric'] is not None:
        raise ValueError('final/epoch checkpoints cannot have best-selection fields')
    for key in ('epoch', 'global_step'):
        integer(state[key], key, 0)
    for key in ('optimizer', 'scheduler'):
        decoded = unpack_state(state[key])
        if not isinstance(decoded, dict):
            raise ValueError(f'{key} must contain its full state dictionary')
        if key == 'optimizer':
            _keys(decoded, ('state', 'param_groups'), 'optimizer state')
            if not isinstance(decoded['state'], dict) or not isinstance(decoded['param_groups'], list):
                raise ValueError('optimizer state/param_groups have invalid types')
        elif not decoded:
            raise ValueError('scheduler state cannot be empty')
    rng = unpack_state(state['rng'])
    _keys(rng, ('python', 'numpy', 'torch_cpu', 'torch_cuda'), 'rng', exact=True)
    try:
        random.Random(0).setstate(rng['python'])
        np.random.RandomState(0).set_state(rng['numpy'])
        torch.Generator(device='cpu').set_state(rng['torch_cpu'])
    except (ValueError, TypeError, RuntimeError) as error:
        raise ValueError('invalid Python/NumPy/Torch CPU RNG state') from error
    if not isinstance(rng['torch_cuda'], list):
        raise ValueError('torch_cuda RNG must be a device-ordered list (empty on CPU)')
    for item in rng['torch_cuda']:
        if not isinstance(item, torch.Tensor) or item.dtype != torch.uint8 or item.ndim != 1 or not item.numel():
            raise ValueError('invalid CUDA RNG byte state')
    for key in ('dataloader_generators', 'sampler_state'):
        if not isinstance(unpack_state(state[key]), dict):
            raise ValueError(f'{key} must be explicit mapping, empty only when unused')
    for name, item in unpack_state(state['dataloader_generators']).items():
        _keys(item, ('device', 'state'), 'DataLoader generator', exact=True)
        if not isinstance(name, str) or not name or not isinstance(item['device'], str):
            raise ValueError('generator name/device required')
        if item['device'] == 'cpu':
            try:
                torch.Generator().set_state(item['state'])
            except (TypeError, RuntimeError) as error:
                raise ValueError('invalid DataLoader generator state') from error
        elif not re.fullmatch(r'cuda:\d+', item['device']) or not isinstance(item['state'], torch.Tensor) or item['state'].dtype != torch.uint8 or item['state'].ndim != 1:
            raise ValueError('invalid CUDA DataLoader generator descriptor')
    if 'scaler_state' in state and not isinstance(unpack_state(state['scaler_state']), dict):
        raise ValueError('scaler_state must be an explicit state dictionary')


def validate_metadata(metadata, *, constructor=None):
    _keys(metadata, ('family', 'schema_version', 'config_version', 'config_hash', 'metadata_hash', *SECTIONS),
          'LinearNO metadata', exact=True)
    if metadata['family'] != FAMILY:
        raise ValueError('checkpoint family conflict; legacy checkpoints must use their original loader')
    if (type(metadata['schema_version']) is not int or type(metadata['config_version']) is not int or
            metadata['schema_version'] != SCHEMA_VERSION or metadata['config_version'] != CONFIG_VERSION):
        raise ValueError('unsupported LinearNO schema/config version')
    body = {k: v for k, v in metadata.items() if k != 'metadata_hash'}
    if metadata['metadata_hash'] != digest(body):
        raise ValueError('metadata checksum mismatch')
    config = validate_resolved(metadata['profile_spec'])
    if metadata['config_hash'] != config['config_hash']:
        raise ValueError('metadata/profile config hash mismatch')
    validate_model_spec(metadata['model_spec'], constructor)
    if constructor is None:
        validate_release_descriptor(metadata['model_spec'], config)
    if metadata['objective_spec'] != config['values']['objective'] or metadata['evaluation_spec'] != config['values']['evaluation']:
        raise ValueError('objective/evaluation specs disagree with resolved profile')
    _keys(metadata['evaluation_spec'], EVALUATION_AXES, 'evaluation_spec')
    _keys(metadata['data_spec'], ('split', 'checksums', 'sampling'), 'data_spec')
    if not metadata['data_spec']['split'] or not metadata['data_spec']['checksums']:
        raise ValueError('checkpoint requires actual data split/checksums (synthetic tests must be labelled)')
    for name, checksum in metadata['data_spec']['checksums'].items():
        if not name:
            raise ValueError('data checksum name required')
        _hash(checksum, 'data checksum')
    prov = metadata['provenance_spec']
    _keys(prov, PROVENANCE_FIELDS, 'provenance_spec')
    for key in ('transolver_sha', 'linearno_sha'):
        _hash(prov[key], key, 40)
    for key in ('target_sha', 'base_commit'):
        if prov[key] is not None:
            _hash(prov[key], key, 40)
    for key in ('paper_sha256', 'source_sha256', 'normalized_patch_sha256'):
        _hash(prov[key], key)
    if prov['paper_version'] != '2511.06294v3' or (prov['dirty'] is not None and type(prov['dirty']) is not bool):
        raise ValueError('paper version/dirty provenance invalid')
    if not isinstance(prov['command'], list) or not prov['command'] or not isinstance(prov['environment'], dict):
        raise ValueError('command argv and environment mapping required')
    _validate_normalizers(metadata['normalizer_spec'])
    for record in metadata['normalizer_spec']['records'].values():
        if record['data_checksum'] not in metadata['data_spec']['checksums'].values():
            raise ValueError('normalizer checksum is absent from data_spec')
    _validate_resume(metadata['resume_state'])
    members = metadata['ensemble_manifest']
    if not isinstance(members, list):
        raise ValueError('ensemble manifest must be list (empty when not an ensemble)')
    ids = set()
    for order, member in enumerate(members):
        _keys(member, ('member_id', 'order', 'path', 'sha256', 'format'), 'ensemble member', exact=True)
        if member['order'] != order or member['member_id'] in ids or not isinstance(member['member_id'], str):
            raise ValueError('ensemble member id/order invalid')
        ids.add(member['member_id'])
        path = Path(member['path'])
        if path.is_absolute() or '..' in path.parts or not path.name:
            raise ValueError('ensemble path must be a relative descendant')
        if member['format'] != 'state_dict':
            raise ValueError('L1 manifest accepts strict state_dict only; trusted pickle conversion is deferred')
        _hash(member['sha256'], 'ensemble sha256')
    return copy.deepcopy(metadata)


def make_metadata(*, profile_spec, constructor=None, **sections):
    _keys(sections, set(SECTIONS) - {'profile_spec'}, 'metadata sections', exact=True)
    result = dict(family=FAMILY, schema_version=SCHEMA_VERSION, config_version=CONFIG_VERSION,
                  config_hash=profile_spec['config_hash'], profile_spec=copy.deepcopy(profile_spec), **copy.deepcopy(sections))
    result['metadata_hash'] = digest(result)
    return validate_metadata(result, constructor=constructor)


def write_metadata(path, metadata, *, constructor=None):
    """Create a new JSON only; never overwrite an existing training sidecar."""
    checked = validate_metadata(metadata, constructor=constructor)
    with Path(path).open('x', encoding='utf-8') as stream:
        stream.write(canonical_json(checked) + '\n')


def read_metadata(path, *, constructor=None):
    """Read-only, no model/pickle/data loading and no directory creation."""
    def unique_pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError('duplicate JSON metadata key')
            result[key] = value
        return result
    with Path(path).open(encoding='utf-8') as stream:
        value = json.load(stream, object_pairs_hook=unique_pairs)
    return validate_metadata(value, constructor=constructor)


def compare_metadata(saved, requested, *, resume=False, constructor=None):
    """Strict architecture check; runtime/protocol changes are named, never hidden.

    Eval uses the saved objective/evaluation/data/normalizers. Callers must decide
    on each reported protocol change; this function does not mutate or construct.
    Resume rejects protocol changes, while device/path changes are reported.
    """
    saved, requested = validate_metadata(saved, constructor=constructor), validate_metadata(requested, constructor=constructor)
    for key in ('model_spec',):
        if saved[key] != requested[key]:
            raise ValueError(f'structural checkpoint mismatch: {key}')
    for key in ('model',):
        if saved['profile_spec']['values'][key] != requested['profile_spec']['values'][key]:
            raise ValueError(f'structural checkpoint mismatch: {key}')
    if saved['profile_spec']['task'] != requested['profile_spec']['task']:
        raise ValueError('task checkpoint mismatch')
    changes = {k: dict(saved=saved[k], requested=requested[k]) for k in
               ('objective_spec', 'evaluation_spec', 'data_spec', 'normalizer_spec') if saved[k] != requested[k]}
    for key in ('training', 'runtime'):
        a, b = saved['profile_spec']['values'][key], requested['profile_spec']['values'][key]
        if a != b:
            changes[key] = dict(saved=a, requested=b)
    if resume and (set(changes) - {'runtime'}):
        raise ValueError('resume protocol mismatch: ' + ', '.join(sorted(set(changes) - {'runtime'})))
    return changes
