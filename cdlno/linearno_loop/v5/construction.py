"""Validated config-first V5 construction."""
from contextvars import ContextVar
import importlib
import inspect

import torch

from linearno_loop.v5.config import validate_config
from linearno_loop.v5.contracts import ARCHITECTURE, ARCHITECTURE_EXTENSION

_ACTIVE_CONFIG = ContextVar("linearno_loop_v5_active_config", default=None)


def constructor_config(kwargs):
    config = _ACTIVE_CONFIG.get()
    if config is None:
        raise RuntimeError("V5 wrappers require validated config-first construction")
    if config["model_spec"]["constructor_kwargs"] != kwargs:
        raise ValueError("V5 constructor kwargs differ from resolved config")
    return config


class LoopForwardViewV5:
    family = "linearno_loop"
    architecture = ARCHITECTURE
    architecture_extension = ARCHITECTURE_EXTENSION

    @property
    def blocks(self):
        return (self.loop,)


def build_from_config(config, *, initialization_seed=None):
    checked = validate_config(config)
    seed = checked["fair_comparison"]["public_backbone_seed"] if initialization_seed is None else initialization_seed
    if type(seed) is not int or not 0 <= seed < 2**64:
        raise ValueError("initialization seed must fit Torch uint64")
    spec = checked["model_spec"]
    module_name, class_name = spec["class_path"].rsplit(".", 1)
    constructor = getattr(importlib.import_module(module_name), class_name)
    signature = inspect.signature(constructor)
    signature.bind(**spec["constructor_kwargs"])
    token = _ACTIVE_CONFIG.set(checked)
    try:
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(seed)
            model = constructor(**spec["constructor_kwargs"])
    finally:
        _ACTIVE_CONFIG.reset(token)
    model.v5_config = checked
    model.config_hash = checked["config_hash"]
    model.task = checked["task"]
    model.profile = checked["profile"]
    model.topology_preset = checked["topology_preset"]
    model.initialization_seed = seed
    return model
