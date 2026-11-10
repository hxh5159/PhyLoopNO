"""Pure V5 resolver; imports no torch and accesses no task data."""
from copy import deepcopy
import hashlib

from cdlno.linearno.profiles import DEFAULT_PROFILE, PROFILES, resolve_config as resolve_base
from .contracts import (
    ARCHITECTURE, ARCHITECTURE_EXTENSION, ARCHITECTURE_VERSION, CHECKPOINT_SCHEMA,
    CHECKPOINT_VERSION, CLASS_PATHS, CONFIG_VERSION, CORE_NORM_MODES,
    DEFAULT_CORE_NORM_MODE, DEFAULT_TOPOLOGY, FAMILY, FORMULA_VERSION, PRESETS,
    RESIDUAL_MODE, SCHEMA_VERSION, TASKS, QK_SHARING_MODES, PAPER_EXPERT_COUNTS,
    TOPOLOGY_FIELDS, V5SchemaError, digest, integer, require_equal, seal,
)

OPTIONS = {"architecture", "topology_preset", "executed_depth", *TOPOLOGY_FIELDS,
           "residual_mode", "core_norm_mode", "expert_count", "expert_width",
           "actual_M", "seed", "qk_sharing"}


def _derived_seed(seed, label):
    raw = f"v5:{seed}:{label}".encode()
    return int.from_bytes(hashlib.sha256(raw).digest()[:8], "big") % (2**63)


def _base(task, profile, overrides):
    contract = ("car_transolver_mse_v1" if task == "car" else
                "airfrans_transolver_mse_v1" if task == "airfrans" else
                "standard_temporal_l5" if task in ("ns", "plasticity") else
                "standard_static_l4")
    return resolve_base(task, profile, explicit=overrides, contract=contract)


def _topology(options):
    supplied = set(options) & set(TOPOLOGY_FIELDS)
    preset = options.get("topology_preset")
    depth = options.get("executed_depth")
    if preset is not None and preset not in (*PRESETS, "custom"):
        raise V5SchemaError("unknown V5 topology preset")
    if preset in PRESETS:
        if supplied or depth is not None:
            raise V5SchemaError("preset conflicts with custom topology/depth")
        values = PRESETS[preset]
        name = preset
    elif preset == "custom" or supplied:
        if preset != "custom" or supplied != set(TOPOLOGY_FIELDS) or depth is not None:
            raise V5SchemaError("custom topology requires exactly P/C/R/S")
        values = tuple(options[field] for field in TOPOLOGY_FIELDS)
        name = "custom"
    elif depth is not None:
        integer(depth, "executed_depth", 1)
        if depth <= 4 or (depth - 4) % 2:
            raise V5SchemaError("depth shorthand requires positive integer C=(L-4)/2")
        values = PRESETS[DEFAULT_TOPOLOGY] if depth == 8 else (2, (depth - 4) // 2, 2, 2)
        name = DEFAULT_TOPOLOGY if depth == 8 else "custom"
    else:
        values = PRESETS[DEFAULT_TOPOLOGY]
        name = DEFAULT_TOPOLOGY
    P, C, R, S = values
    for field, value in zip(TOPOLOGY_FIELDS, values):
        integer(value, field, 1)
    return name, dict(zip(TOPOLOGY_FIELDS, values))


def _request(task, profile, options, overrides):
    if task not in TASKS or profile not in PROFILES:
        raise V5SchemaError("unknown task/profile")
    if type(options) is not dict or set(options) - OPTIONS:
        raise V5SchemaError("unknown V5 options: " + str(sorted(set(options) - OPTIONS)))
    if options.get("architecture") != ARCHITECTURE:
        raise V5SchemaError(f"explicit architecture={ARCHITECTURE} is required")
    if type(overrides) is not dict:
        raise V5SchemaError("profile_overrides must be an object")
    for field in ("expert_count", "expert_width", "actual_M", "executed_depth"):
        if field in options:
            integer(options[field], field, 1)
    for field in TOPOLOGY_FIELDS:
        if field in options:
            integer(options[field], field, 1)
    if options.get("residual_mode", RESIDUAL_MODE) != RESIDUAL_MODE:
        raise V5SchemaError("V5 residual rule is fixed to " + RESIDUAL_MODE)
    if options.get("core_norm_mode", DEFAULT_CORE_NORM_MODE) not in CORE_NORM_MODES:
        raise V5SchemaError("core_norm_mode must be one of " + str(CORE_NORM_MODES))
    if options.get("qk_sharing", "independent") not in QK_SHARING_MODES:
        raise V5SchemaError("qk_sharing must be independent or shared")
    if task in ("airfrans", "car") and "expert_count" not in options:
        raise V5SchemaError("industrial expert_count is not specified in the paper; pass it explicitly")


def _assemble(request, base):
    task, profile, options = request["task"], request["profile"], request["options"]
    topology_name, top = _topology(options)
    model = base["values"]["model"]
    if "seed" in options:
        raise V5SchemaError("seed must be resolved through the task runtime profile")
    C, heads = model["hidden"], model["heads"]
    M = options.get("actual_M", model["linearno_rank"])
    E = options.get("expert_count", PAPER_EXPERT_COUNTS.get(task))
    F = options.get("expert_width", C * model["ffn_ratio"])
    core_norm_mode = options.get("core_norm_mode", DEFAULT_CORE_NORM_MODE)
    qk_sharing = options.get("qk_sharing", "independent")
    integer(E, "expert_count"); integer(F, "expert_width"); integer(M, "actual_M")
    effective = {**model, "linearno_rank": M}
    from cdlno.linearno.profiles import validate_model
    validate_model(task, effective)
    P, core, R, S = (top[field] for field in TOPOLOGY_FIELDS)
    unique, executed = P + core + S, P + core * R + S
    class_path = CLASS_PATHS["airfrans" if task == "airfrans" else
                             "car" if task == "car" else "standard"]
    constructor = dict(
        space_dim=model["space_dim"], fun_dim=model["fun_dim"], out_dim=model["out_dim"],
        time_input=model["time_input"], ref=model["ref"], unified_pos=model["unified_pos"],
        hidden_width=C, grid_height=model["H"], grid_width=model["W"], actual_M=M,
        heads=heads, variant=model["linearno_variant"], dropout=model["dropout"],
        activation=model["activation"], expert_count=E, expert_width=F,
        prefix_blocks=P, recurrent_core_blocks=core, loop_repeats=R, suffix_blocks=S,
        core_norm_mode=core_norm_mode, qk_sharing=qk_sharing, architecture=ARCHITECTURE,
    )
    seed = base["values"]["runtime"]["seed"]
    loop = dict(
        task=task, profile=profile, topology_preset=topology_name, **top,
        unique_depth=unique, executed_depth=executed, hidden_width=C, heads=heads,
        head_dim=C // heads, actual_M=M, variant=model["linearno_variant"],
        grid_height=model["H"], grid_width=model["W"],
        expert_count=E, expert_width=F, residual_mode=RESIDUAL_MODE,
        core_norm_mode=core_norm_mode, qk_sharing=qk_sharing,
        router_sharing="shared" if qk_sharing == "shared" and core_norm_mode == "shared" and E == 1 else "independent",
        scale_operator=1.0, scale_expert_core=f"1/{R}", scale_expert_value=1.0/R,
        gate_axis="expert", gate_bias=True, gate_multihead=False,
        gate_temperature=None, dense_experts=True,
        state_partition=dict(
            shared_by_physical_position=(
                ["in_project_x", "to_v", "to_out", "experts"] +
                (["router"] if qk_sharing == "shared" and core_norm_mode == "shared" and E == 1 else []) +
                (["to_q", "to_k", "active_qk_temperature"] if qk_sharing == "shared" else []) +
                (["ln_1", "ln_2"] if core_norm_mode == "shared" else [])
            ),
            independent_by_visit=(
                ([] if qk_sharing == "shared" and core_norm_mode == "shared" and E == 1 else ["router"]) +
                (["to_q", "to_k", "active_qk_temperature"] if qk_sharing == "independent" else []) +
                (["ln_1", "ln_2"] if core_norm_mode == "visit_independent" else [])
            ),
            prefix_suffix="independent_complete_single_visit_blocks",
        ),
        initialization=dict(
            whole_model_apply_count=1,
            q_across_visits="shared_storage" if qk_sharing == "shared" else "same_values_distinct_storage",
            k_across_visits="shared_storage" if qk_sharing == "shared" else "same_values_distinct_storage",
            q_vs_k="independent",
            active_temperature_across_visits="shared_storage" if qk_sharing == "shared" else "same_values_distinct_storage",
            router="weight_and_bias_zero", experts="independent_native_initialization",
            core_norms=("same_values_distinct_storage" if core_norm_mode == "visit_independent"
                        else "shared_by_physical_position"),
        ),
        formula_version=FORMULA_VERSION,
    )
    result = dict(
        family=FAMILY, architecture=ARCHITECTURE, architecture_family=FAMILY,
        architecture_extension=ARCHITECTURE_EXTENSION, architecture_version=ARCHITECTURE_VERSION,
        schema_version=SCHEMA_VERSION, config_version=CONFIG_VERSION,
        checkpoint_schema=CHECKPOINT_SCHEMA, checkpoint_version=CHECKPOINT_VERSION,
        task=task, profile=profile, topology_preset=topology_name, **top,
        residual_mode=RESIDUAL_MODE, core_norm_mode=core_norm_mode, qk_sharing=qk_sharing,
        expert_count=E, expert_width=F, actual_M=M,
        seed=seed, request=deepcopy(request), profile_spec=deepcopy(base), loop_spec=loop,
        expert_spec=dict(count=E, input_width=C, hidden_width=F, output_width=C,
                         activation="GELU", bias=True, dense=True,
                         router=dict(input_width=C, output_width=E, bias=True,
                                     softmax_axis="expert", multihead=False, temperature=None)),
        temperature_spec=dict(variant=model["linearno_variant"],
            active=model["linearno_variant"] in ("temp", "conv_temp", "shapenet"),
            spelling="tempreature_q/k" if task == "car" else "temperature_q/k",
            airfrans_inert_shared=model["linearno_variant"] == "airfrans"),
        model_spec=dict(class_path=class_path, constructor_kwargs=constructor),
        fair_comparison=dict(public_backbone_seed=seed,
            dataloader_generators={"train": _derived_seed(seed, "train_loader"),
                                   "test": _derived_seed(seed, "test_loader")}),
    )
    return seal(result)


def resolve_config(task, profile=DEFAULT_PROFILE, *, options=None, profile_overrides=None):
    options, overrides = deepcopy(options or {}), deepcopy(profile_overrides or {})
    _request(task, profile, options, overrides)
    base_options = deepcopy(overrides)
    if "seed" in options:
        base_options["runtime.seed"] = options["seed"]
        options.pop("seed")
    request = dict(task=task, profile=profile, options=deepcopy(options),
                   profile_overrides=deepcopy(base_options))
    return _assemble(request, _base(task, profile, base_options))


def validate_config(config):
    if type(config) is not dict or config.get("architecture") != ARCHITECTURE:
        raise V5SchemaError("not a V5 configuration")
    saved = config.get("config_hash")
    body = deepcopy(config); body.pop("config_hash", None)
    require_equal(digest(body), saved, "config_hash")
    request = config.get("request")
    if type(request) is not dict or set(request) != {"task", "profile", "options", "profile_overrides"}:
        raise V5SchemaError("invalid saved V5 request")
    expected = resolve_config(request["task"], request["profile"],
                              options=request["options"],
                              profile_overrides=request["profile_overrides"])
    require_equal(expected, config, "resolved_config")
    return deepcopy(config)


def run_directory_id(config):
    c = validate_config(config); s = c["loop_spec"]
    # Full ownership lives in immutable JSON; keep paths usable on Windows.
    return (f"{s['task']}_L{s['executed_depth']}_E{s['expert_count']}"
            f"F{s['expert_width']}_{c['config_hash'][:12]}")
