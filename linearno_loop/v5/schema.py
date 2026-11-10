"""Strict metadata contract for the V5 checkpoint pair."""
from copy import deepcopy
import json
from pathlib import Path, PurePosixPath

from .config import validate_config
from .contracts import (CHECKPOINT_SCHEMA, CHECKPOINT_VERSION, V5SchemaError,
                        digest)

LOAD_POLICY = dict(metadata_first=True, strict=True, legacy_fallback=False,
                   architecture_migration=False)
EXTERNAL = {"data_spec", "normalizer_spec", "provenance_spec", "resume_state",
            "ensemble_manifest", "parameter_measurement"}


def _exact(value, keys, name):
    if type(value) is not dict or set(value) != set(keys):
        raise V5SchemaError(name + " fields mismatch")


def _hash(value, name):
    if type(value) is not str or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
        raise V5SchemaError(name + " must be SHA-256")


def make_metadata(config, **sections):
    cfg = validate_config(config); _exact(sections, EXTERNAL, "metadata sections")
    meta = dict(
        family=cfg["family"], architecture=cfg["architecture"],
        architecture_family=cfg["architecture_family"],
        architecture_extension=cfg["architecture_extension"],
        architecture_version=cfg["architecture_version"],
        checkpoint_schema=CHECKPOINT_SCHEMA, checkpoint_version=CHECKPOINT_VERSION,
        config_hash=cfg["config_hash"], load_policy=deepcopy(LOAD_POLICY),
        resolved_config=deepcopy(cfg), model_spec=deepcopy(cfg["model_spec"]),
        profile_spec=deepcopy(cfg["profile_spec"]), loop_spec=deepcopy(cfg["loop_spec"]),
        expert_spec=deepcopy(cfg["expert_spec"]),
        temperature_spec=deepcopy(cfg["temperature_spec"]),
        objective_spec=deepcopy(cfg["profile_spec"]["values"]["objective"]),
        evaluation_spec=deepcopy(cfg["profile_spec"]["values"]["evaluation"]),
        **deepcopy(sections),
    )
    meta["metadata_hash"] = digest(meta)
    return validate_metadata(meta)


def validate_metadata(meta):
    keys = {"family", "architecture", "architecture_family", "architecture_extension",
            "architecture_version", "checkpoint_schema", "checkpoint_version", "config_hash",
            "load_policy", "resolved_config", "model_spec", "profile_spec", "loop_spec",
            "expert_spec", "temperature_spec", "objective_spec", "evaluation_spec",
            *EXTERNAL, "metadata_hash"}
    _exact(meta, keys, "metadata")
    try: json.dumps(meta, sort_keys=True, allow_nan=False)
    except (TypeError, ValueError) as error: raise V5SchemaError("metadata must be finite JSON") from error
    if meta["checkpoint_schema"] != CHECKPOINT_SCHEMA or meta["checkpoint_version"] != CHECKPOINT_VERSION:
        raise V5SchemaError("checkpoint schema/version mismatch")
    cfg = validate_config(meta["resolved_config"])
    expected = dict(family=cfg["family"], architecture=cfg["architecture"],
        architecture_family=cfg["architecture_family"], architecture_extension=cfg["architecture_extension"],
        architecture_version=cfg["architecture_version"], config_hash=cfg["config_hash"],
        load_policy=LOAD_POLICY, model_spec=cfg["model_spec"], profile_spec=cfg["profile_spec"],
        loop_spec=cfg["loop_spec"], expert_spec=cfg["expert_spec"],
        temperature_spec=cfg["temperature_spec"],
        objective_spec=cfg["profile_spec"]["values"]["objective"],
        evaluation_spec=cfg["profile_spec"]["values"]["evaluation"])
    for name, value in expected.items():
        if meta[name] != value: raise V5SchemaError(name + " differs from resolved config")
    data = meta["data_spec"]
    _exact(data, {"protocol", "split", "checksums", "sampling", "scope", "runtime"}, "data_spec")
    if data["protocol"] != cfg["profile_spec"]["values"]["data"] or data["scope"] not in ("synthetic", "real"):
        raise V5SchemaError("data protocol/scope mismatch")
    if type(data["checksums"]) is not dict or not data["checksums"] or type(data["runtime"]) is not dict:
        raise V5SchemaError("data checksums/runtime required")
    for value in data["checksums"].values(): _hash(value, "data checksum")
    normalizers = meta["normalizer_spec"]
    if type(normalizers) is not dict or set(normalizers) != {"policy", "records"}:
        raise V5SchemaError("normalizer spec mismatch")
    if type(meta["provenance_spec"]) is not dict or not meta["provenance_spec"]:
        raise V5SchemaError("provenance required")
    state = meta["resume_state"]
    if type(state) is not dict or type(state.get("epoch")) is not int or state["epoch"] < 0 or state.get("checkpoint_role") not in ("epoch", "final"):
        raise V5SchemaError("resume state mismatch")
    from linearno_loop.state import validate_resume, validate_normalizers
    validate_resume(state); validate_normalizers(normalizers, data["checksums"])
    measured = meta["parameter_measurement"]
    _exact(measured, {"status", "total", "trainable", "groups"}, "parameter measurement")
    if measured["status"] != "measured" or sum(measured["groups"].values()) != measured["total"] or measured["trainable"] != measured["total"]:
        raise V5SchemaError("parameter measurement mismatch")
    if type(meta["ensemble_manifest"]) is not list: raise V5SchemaError("ensemble manifest must be list")
    for index, member in enumerate(meta["ensemble_manifest"]):
        _exact(member, {"member_id", "order", "path", "sha256", "format"}, "ensemble member")
        path = PurePosixPath(member["path"])
        if member["order"] != index or member["format"] != "state_dict" or path.is_absolute() or ".." in path.parts:
            raise V5SchemaError("ensemble member mismatch")
        _hash(member["sha256"], "ensemble checksum")
    saved = meta["metadata_hash"]; body = deepcopy(meta); body.pop("metadata_hash")
    if saved != digest(body): raise V5SchemaError("metadata hash mismatch")
    return deepcopy(meta)


def write_metadata(path, meta):
    checked = validate_metadata(meta)
    with Path(path).open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(checked, sort_keys=True, separators=(",", ":")) + "\n")


def read_metadata(path):
    return validate_metadata(json.loads(Path(path).read_text(encoding="utf-8")))


def restore_config(meta, *, explicit=None, runtime=None, strict=True):
    if strict is not True: raise V5SchemaError("strict=True is required")
    checked = validate_metadata(meta); cfg = checked["resolved_config"]
    spec = cfg["loop_spec"]
    expected = dict(task=cfg["task"], profile=cfg["profile"], architecture=cfg["architecture"],
        topology_preset=cfg["topology_preset"], residual_mode=cfg["residual_mode"],
        core_norm_mode=cfg["core_norm_mode"], qk_sharing=cfg["qk_sharing"],
        expert_count=cfg["expert_count"], expert_width=cfg["expert_width"],
        actual_M=cfg["actual_M"], linearno_loop=True, seed=cfg["seed"],
        prefix_blocks=spec["prefix_blocks"],
        recurrent_core_blocks=spec["recurrent_core_blocks"],
        loop_repeats=spec["loop_repeats"], suffix_blocks=spec["suffix_blocks"],
        executed_depth=spec["executed_depth"])
    for key, value in (explicit or {}).items():
        if key not in expected or expected[key] != value:
            raise V5SchemaError("explicit " + key + " conflicts with V5 metadata")
    if set(runtime or {}) - {"device", "experiment_dir"}:
        raise V5SchemaError("unsupported V5 runtime override")
    return dict(config=deepcopy(cfg), runtime_changes=deepcopy(runtime or {}),
                load_policy=deepcopy(LOAD_POLICY))
