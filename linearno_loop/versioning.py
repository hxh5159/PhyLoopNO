"""Dispatch only the published PhyLoopNO architecture, before tensor imports."""
import inspect
from .contracts import OPTIONS, read_json
from .v5 import config, schema
from .v5.contracts import ARCHITECTURE, ARCHITECTURE_EXTENSION

def is_v5(value):
    return (value.get('architecture') == ARCHITECTURE and
            value.get('architecture_extension') == ARCHITECTURE_EXTENSION and
            value.get('architecture_version') == 5)

def api(value):
    if not is_v5(value):
        raise ValueError('checkpoint is not a published PhyLoopNO architecture')
    return schema

def resolve_config(task, profile, *, options, profile_overrides):
    return config.resolve_config(task, profile, options=options, profile_overrides=profile_overrides)

def run_directory_id(value):
    return config.run_directory_id(value)

def read_metadata(path):
    value = read_json(path)
    return api(value).validate_metadata(value)

def validate_metadata(value, **kwargs):
    return api(value).validate_metadata(value, **kwargs)

def restore_config(value, **kwargs):
    return api(value).restore_config(value, **kwargs)

def make_metadata(value, **sections):
    return schema.make_metadata(value, **sections)

def write_metadata(path, value):
    return api(value).write_metadata(path, value)

def validate_constructor(model_spec, constructor, *, version=None):
    return inspect.signature(constructor).bind(**model_spec['constructor_kwargs'])
