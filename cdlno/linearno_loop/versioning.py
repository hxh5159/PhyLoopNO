"""Construction and archives for the published PhyLoopNO architecture."""
import importlib
from linearno_loop.versioning import api

def construction_api(config):
    api(config)
    return importlib.import_module('cdlno.linearno_loop.v5.construction')

def checkpoint_api(config):
    api(config)
    return importlib.import_module('cdlno.linearno_loop.v5.checkpoint')

def construct(config, *, member_seed=None):
    return construction_api(config).build_from_config(config, initialization_seed=member_seed)

def provenance(config, task=None):
    api(config)
    from .v5.provenance import provenance as current
    return current(config, task=task)
