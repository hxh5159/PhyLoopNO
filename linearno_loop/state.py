"""Inspect existing numeric/tagged JSON encodings without NumPy/torch/pickle.

No tensors or RNG generators are restored here. Runtime backend RNG acceptance
and optimizer/model shape checks remain mandatory at the future load boundary.
"""
import base64
from dataclasses import dataclass
import math
import re

from .contracts import LoopSchemaError, exact, integer, json_value, digest

TORCH_BYTES = dict(bool=1, uint8=1, int8=1, int16=2, int32=4, int64=8,
                   float16=2, bfloat16=2, float32=4, float64=8, complex64=8, complex128=16)


def hash_string(value, name, length=64):
    if type(value) is not str or not re.fullmatch('[0-9a-f]{'+str(length)+'}', value):
        raise LoopSchemaError(f'{name}: expected {length}-digit lowercase hash')


def text(value, name):
    if type(value) is not str or not value:
        raise LoopSchemaError(f'{name}: nonempty string required')


def numeric(spec, path):
    exact(spec, {'backend', 'dtype', 'shape', 'encoding', 'data', 'sha256'}, path)
    if spec['encoding'] != 'base64' or spec['sha256'] != digest({k:v for k,v in spec.items() if k!='sha256'}):
        raise LoopSchemaError(f'{path}: numeric checksum/encoding mismatch')
    if type(spec['shape']) is not list:
        raise LoopSchemaError(f'{path}.shape: expected list')
    for d in spec['shape']:
        integer(d, path+'.shape dimension')
    if spec['backend'] == 'torch':
        size = TORCH_BYTES.get(spec['dtype']) if type(spec['dtype']) is str else None
    elif spec['backend'] == 'numpy':
        match = re.fullmatch(r'[<>=|]([biufc])(1|2|4|8|16)', spec['dtype']) if type(spec['dtype']) is str else None
        size = int(match[2]) if match else None
        if match and size not in {'b':(1,), 'i':(1,2,4,8), 'u':(1,2,4,8), 'f':(2,4,8,16), 'c':(8,16)}[match[1]]:
            size = None
    else:
        size = None
    if size is None:
        raise LoopSchemaError(f'{path}.dtype/backend: unsupported numeric encoding')
    try:
        raw = base64.b64decode(spec['data'], validate=True)
    except (ValueError, TypeError) as error:
        raise LoopSchemaError(f'{path}.data: invalid base64') from error
    if len(raw) != math.prod(spec['shape'])*size:
        raise LoopSchemaError(f'{path}: byte count does not match shape/dtype')
    return NumericDescriptor(spec)


@dataclass(frozen=True)
class NumericDescriptor:
    spec: dict


def inspect_state(node, path='state'):
    """Return Python containers with opaque numeric descriptors, never tensors."""
    exact(node, {'type', 'value'}, path)
    kind, value = node['type'], node['value']
    if kind == 'numeric':
        return numeric(value, path+'.value')
    if kind == 'scalar':
        if value is not None and type(value) not in (str, bool, int, float):
            raise LoopSchemaError(path+': invalid scalar')
        json_value(value, path)
        return value
    if kind in ('tuple', 'list'):
        if type(value) is not list:
            raise LoopSchemaError(path+': expected sequence')
        items = [inspect_state(v, f'{path}[{i}]') for i,v in enumerate(value)]
        return tuple(items) if kind == 'tuple' else items
    if kind == 'dict':
        if type(value) is not list:
            raise LoopSchemaError(path+': expected mapping pairs')
        result = {}
        for i, pair in enumerate(value):
            if type(pair) is not list or len(pair) != 2:
                raise LoopSchemaError(path+': expected key/value pair')
            key, item = (inspect_state(v, f'{path}[{i}][{j}]') for j,v in enumerate(pair))
            if type(key) not in (str, int) or key in result:
                raise LoopSchemaError(path+': invalid or duplicate state key')
            result[key] = item
        return result
    raise LoopSchemaError(path+': unknown tagged state type')


def rng_bytes(value, path):
    if not isinstance(value, NumericDescriptor):
        raise LoopSchemaError(path+': requires encoded uint8 RNG vector')
    s = value.spec
    if s['backend'] != 'torch' or s['dtype'] != 'uint8' or len(s['shape']) != 1 or s['shape'][0] < 1:
        raise LoopSchemaError(path+': requires nonempty torch uint8 RNG vector')


def validate_resume(state):
    exact(state, {'checkpoint_role','selection_split','selection_metric','epoch','global_step',
                  'optimizer','scheduler','rng','dataloader_generators','sampler_state'}, 'resume_state')
    if state['checkpoint_role'] not in ('epoch', 'final', 'best_validation'):
        raise LoopSchemaError('resume_state.checkpoint_role: unknown role')
    if state['checkpoint_role'] == 'best_validation':
        if state['selection_split'] != 'validation':
            raise LoopSchemaError('selection_split: best checkpoint requires validation, never test')
        text(state['selection_metric'], 'selection_metric')
    elif state['selection_split'] is not None or state['selection_metric'] is not None:
        raise LoopSchemaError('final/epoch has no checkpoint selection fields')
    integer(state['epoch'], 'epoch'); integer(state['global_step'], 'global_step')
    opt = inspect_state(state['optimizer'], 'optimizer')
    exact(opt, {'state','param_groups'}, 'optimizer')
    if type(opt['state']) is not dict or type(opt['param_groups']) is not list:
        raise LoopSchemaError('optimizer: state mapping and parameter groups required')
    sched = inspect_state(state['scheduler'], 'scheduler')
    if type(sched) is not dict or not sched:
        raise LoopSchemaError('scheduler: nonempty state dictionary required')
    rng = inspect_state(state['rng'], 'rng')
    exact(rng, {'python','numpy','torch_cpu','torch_cuda'}, 'rng')
    # Validate Python state on an isolated object; global RNG never advances.
    import random
    try:
        random.Random(0).setstate(rng['python'])
    except (ValueError, TypeError, OverflowError) as error:
        raise LoopSchemaError('rng.python: invalid state') from error
    n = rng['numpy']
    if (type(n) is not tuple or len(n) != 5 or n[0] != 'MT19937' or
        not isinstance(n[1], NumericDescriptor) or n[1].spec['backend'] != 'numpy' or
        n[1].spec['dtype'] not in ('<u4','>u4','=u4') or n[1].spec['shape'] != [624] or
        type(n[2]) is not int or not 0 <= n[2] <= 624 or type(n[3]) is not int or n[3] not in (0,1) or
        type(n[4]) not in (int,float) or not math.isfinite(n[4])):
        raise LoopSchemaError('rng.numpy: expected complete MT19937 state')
    rng_bytes(rng['torch_cpu'], 'rng.torch_cpu')
    if type(rng['torch_cuda']) is not list:
        raise LoopSchemaError('rng.torch_cuda: expected device-ordered list (empty on CPU)')
    for i,v in enumerate(rng['torch_cuda']):
        rng_bytes(v, f'rng.torch_cuda[{i}]')
    generators = inspect_state(state['dataloader_generators'], 'dataloader_generators')
    sampler = inspect_state(state['sampler_state'], 'sampler_state')
    if type(generators) is not dict or type(sampler) is not dict:
        raise LoopSchemaError('explicit generator/sampler dictionaries required')
    for name, g in generators.items():
        text(name, 'generator name'); exact(g, {'device','state'}, 'generator')
        if type(g['device']) is not str or not re.fullmatch(r'cpu|cuda:\d+', g['device']):
            raise LoopSchemaError('generator device invalid')
        rng_bytes(g['state'], 'generator.state')


def validate_normalizers(spec, checksums):
    exact(spec, {'policy','records'}, 'normalizer_spec')
    if spec['policy'] not in ('none','saved_train_fit') or type(spec['records']) is not dict:
        raise LoopSchemaError('normalizer policy/records invalid')
    if bool(spec['records']) != (spec['policy'] == 'saved_train_fit'):
        raise LoopSchemaError('normalizer policy disagrees with states')
    for name, record in spec['records'].items():
        text(name, 'normalizer name')
        exact(record, {'algorithm','fit_split','data_checksum','states'}, 'normalizer record')
        text(record['algorithm'], 'normalizer.algorithm'); text(record['fit_split'], 'normalizer.fit_split')
        hash_string(record['data_checksum'], 'normalizer.data_checksum')
        if record['data_checksum'] not in checksums.values():
            raise LoopSchemaError('normalizer data checksum absent from data_spec')
        if type(record['states']) is not dict or not record['states']:
            raise LoopSchemaError('normalizer requires named numeric states')
        for key, value in record['states'].items():
            text(key, 'normalizer state name'); numeric(value, f'normalizer.{name}.{key}')
