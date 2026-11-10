"""Strict JSON utilities and public PhyLoopNO CLI fields; no tensor imports."""
from copy import deepcopy
import hashlib
import json
import math
from .v5.contracts import (FAMILY, ARCHITECTURE_EXTENSION, SCHEMA_VERSION,
    CONFIG_VERSION, FORMULA_VERSION, TOPOLOGY_FIELDS, PRESETS, CLASS_PATHS)
RESIDUAL_MODES = ('operator_1_expert_1_over_r',)
HISTORY_FLAGS = ()
CLI_CONTRACT = {
    '--linearno-loop': {'field': 'linearno_loop'},
    '--linearno-loop-topology': {'field': 'topology_preset'},
    '--linearno-loop-prefix-blocks': {'field': 'prefix_blocks'},
    '--linearno-loop-core-blocks': {'field': 'recurrent_core_blocks'},
    '--linearno-loop-repeats': {'field': 'loop_repeats'},
    '--linearno-loop-suffix-blocks': {'field': 'suffix_blocks'},
    '--linearno-loop-residual-mode': {'field': 'residual_mode'},
}
OPTIONS = {'architecture', 'topology_preset', 'executed_depth', *TOPOLOGY_FIELDS,
    'residual_mode', 'core_norm_mode', 'qk_sharing', 'expert_count', 'expert_width', 'actual_M'}
class LoopSchemaError(ValueError):
    """Invalid loop contract; raised before any model/weight loading."""

def integer(value, name, minimum=0):
    if type(value) is not int or value < minimum:
        raise LoopSchemaError(f'{name}: expected integer >= {minimum}, got {value!r}')
    return value

def exact(value, keys, name):
    if type(value) is not dict or any((type(k) is not str for k in value)):
        raise LoopSchemaError(f'{name}: expected JSON object with string keys')
    missing, extra = (set(keys) - value.keys(), value.keys() - set(keys))
    if missing or extra:
        raise LoopSchemaError(f'{name}: missing={sorted(missing)}, unknown={sorted(extra)}')

def json_value(value, path='root'):
    if value is None or type(value) in (str, bool, int):
        return
    if type(value) is float and math.isfinite(value):
        return
    if type(value) is list:
        for i, v in enumerate(value):
            json_value(v, f'{path}[{i}]')
        return
    if type(value) is dict and all((type(k) is str for k in value)):
        for k, v in value.items():
            json_value(v, f'{path}.{k}')
        return
    raise LoopSchemaError(f'{path}: expected finite JSON value, got {type(value).__name__}')

def canonical_json(value):
    json_value(value)
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)

def digest(value):
    return hashlib.sha256(canonical_json(value).encode()).hexdigest()

def differences(expected, actual, path='root'):
    """Field-level diagnostics; bool is never numerically equal to int."""
    if type(expected) is not type(actual):
        return [f'{path}: expected {expected!r}, got {actual!r} (type mismatch)']
    if isinstance(expected, dict):
        result = []
        for k in sorted(expected.keys() | actual.keys()):
            if k not in expected or k not in actual:
                result.append(f'{path}.{k}: missing or unknown field')
            else:
                result.extend(differences(expected[k], actual[k], f'{path}.{k}'))
        return result
    if isinstance(expected, list):
        if len(expected) != len(actual):
            return [f'{path}: list length mismatch']
        return [d for i, (a, b) in enumerate(zip(expected, actual)) for d in differences(a, b, f'{path}[{i}]')]
    return [] if expected == actual else [f'{path}: expected {expected!r}, got {actual!r}']

def require_equal(expected, actual, path):
    errors = differences(expected, actual, path)
    if errors:
        raise LoopSchemaError('\n'.join(errors))

def seal(value, field):
    result = deepcopy(value)
    result[field] = digest({k: v for k, v in result.items() if k != field})
    return result

def read_json(path):

    def pairs(items):
        result = {}
        for k, v in items:
            if k in result:
                raise LoopSchemaError(f'duplicate JSON field: {k}')
            result[k] = v
        return result
    with open(path, encoding='utf-8') as stream:
        result = json.load(stream, object_pairs_hook=pairs)
    json_value(result)
    return result
