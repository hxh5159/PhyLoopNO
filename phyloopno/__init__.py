"""Public model construction for the PhyLoopNO paper implementation."""
from linearno_loop.v5.contracts import ARCHITECTURE


def resolve_config(task, *, experts=None, depth=8, expert_width=None,
                   qk_sharing="independent", core_norm_mode="visit_independent",
                   profile="paper_table8_on_release_model", overrides=None):
    """Resolve a paper model without importing Torch or reading a dataset.

    Industrial expert counts must be supplied explicitly: the submission does
    not specify them. ``overrides`` names fields of the native LinearNO profile.
    """
    from linearno_loop.v5.config import resolve_config as resolve
    options = dict(architecture=ARCHITECTURE, executed_depth=depth,
                   qk_sharing=qk_sharing, core_norm_mode=core_norm_mode)
    if experts is not None:
        options["expert_count"] = experts
    if expert_width is not None:
        options["expert_width"] = expert_width
    return resolve(task, profile, options=options, profile_overrides=overrides)


def build_model(task=None, *, config=None, **options):
    """Construct from a validated config, or resolve the requested task first."""
    from cdlno.linearno_loop.v5.construction import build_from_config
    if config is None:
        config = resolve_config(task, **options)
    elif task is not None or options:
        raise ValueError("supply either config or task/options")
    return build_from_config(config)
