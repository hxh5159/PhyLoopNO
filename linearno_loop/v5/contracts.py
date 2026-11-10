"""Versioned, JSON-only contracts for the partial-share dense-expert V5."""
from copy import deepcopy
import hashlib
import json
import math

FAMILY = "linearno_loop"
ARCHITECTURE = "partial_share_feature_gate_v5"
ARCHITECTURE_EXTENSION = "partial_share_feature_gate"
ARCHITECTURE_VERSION = 5
CONFIG_VERSION = 7
SCHEMA_VERSION = 7
CHECKPOINT_SCHEMA = ARCHITECTURE
CHECKPOINT_VERSION = 2
CHECKPOINT_FORMAT = "partial-share-feature-gate-v5-pair-v2"
FORMULA_VERSION = "operator1-dense-expert-1-over-r-qk-sharing-v3"
RESIDUAL_MODE = "operator_1_expert_1_over_r"
CORE_NORM_MODES = ("visit_independent", "shared")
DEFAULT_CORE_NORM_MODE = "visit_independent"
QK_SHARING_MODES = ("independent", "shared")
PAPER_EXPERT_COUNTS = {"airfoil": 4, "pipe": 3, "plasticity": 4,
                       "darcy": 1, "ns": 4, "elasticity": 4}
TASKS = ("airfoil", "darcy", "elasticity", "pipe", "ns", "plasticity",
         "airfrans", "car")
TOPOLOGY_FIELDS = ("prefix_blocks", "recurrent_core_blocks", "loop_repeats",
                   "suffix_blocks")
PRESETS = {
    "p1_c3_r2_s1": (1, 3, 2, 1),
    "p2_c2_r2_s2": (2, 2, 2, 2),
}
DEFAULT_TOPOLOGY = "p1_c3_r2_s1"
CLASS_PATHS = {
    "standard": "cdlno.linearno_loop.v5.standard.LoopedStandardModelV5",
    "airfrans": "cdlno.linearno_loop.v5.airfrans.LoopedAirfRANSModelV5",
    "car": "cdlno.linearno_loop.v5.shapenet.LoopedShapeNetModelV5",
}


class V5SchemaError(ValueError):
    pass


def canonical_json(value):
    try:
        return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    except (TypeError, ValueError) as error:
        raise V5SchemaError("value must be finite JSON") from error


def digest(value):
    return hashlib.sha256(canonical_json(value).encode()).hexdigest()


def integer(value, name, minimum=1):
    if type(value) is not int or value < minimum:
        raise V5SchemaError(f"{name} must be integer >= {minimum}")
    return value


def finite(value, name, positive=False):
    if isinstance(value, bool) or type(value) not in (int, float) or not math.isfinite(value):
        raise V5SchemaError(f"{name} must be finite numeric")
    if positive and value <= 0:
        raise V5SchemaError(f"{name} must be positive")
    return value


def seal(value, field="config_hash"):
    body = deepcopy(value)
    body.pop(field, None)
    body[field] = digest(body)
    return body


def require_equal(expected, actual, path):
    if expected != actual:
        raise V5SchemaError(f"{path} mismatch: expected {expected!r}, got {actual!r}")
