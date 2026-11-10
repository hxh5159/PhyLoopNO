"""Detached observations of the unmodified V5 dense-expert forward."""
from __future__ import annotations


def schedule(model):
    result = []
    for group in ("prefix", "core", "suffix"):
        blocks = getattr(model.loop, group)
        for visit in range(model.loop.loop_repeats if group == "core" else 1):
            for position, block in enumerate(blocks):
                result.append((group, position, visit, block))
    return result


def capture_experts(model, positions, *, fx=None):
    """Two identical eval calls, with temporary hooks on the second call only.

    Save [logical_visit,N,E] probabilities/logits/norms, never full expert
    activations. Reconstruct each *actual* point residual for verification.
    Hooks return None, so they cannot replace production outputs.
    """
    import random
    import numpy as np
    import torch
    if model.training or any(m.training for m in model.modules()):
        raise ValueError("Expert capture requires model.eval() on the complete model")
    if positions.ndim != 3 or positions.shape[0] != 1:
        raise ValueError("Expert visualization requires a single sample [1,N,2]")
    plan = schedule(model)
    experts = model.loop.expert_count
    handles, active, rows = [], {}, []
    events = []
    atol, rtol = 1e-6, 1e-5

    def rng():
        return [torch.random.get_rng_state()] + (torch.cuda.get_rng_state_all() if positions.is_cuda else [])

    initial_rng = rng()
    python_rng, numpy_rng = random.getstate(), np.random.get_state()
    versions = {n: t._version for n, t in model.state_dict(keep_vars=True).items()}

    def cpu(value):
        if not torch.isfinite(value).all():
            raise ValueError("NaN/Inf in expert diagnostics")
        return value.detach().cpu().numpy().copy()

    def begin(owner, group, position):
        def hook(module, inputs, kwargs):
            if owner in active:
                raise RuntimeError("Overlapping expert block calls")
            visit = kwargs.get("visit_index", 0)
            scale = kwargs.get("expert_scale", 1.)
            expected = 1. / model.loop.loop_repeats if group == "core" else 1.
            if scale != expected:
                raise RuntimeError("Unexpected expert residual scale")
            expected_event = [(g, p, v) for g, p, v, _ in plan][len(events)]
            if (group, position, visit) != expected_event:
                raise RuntimeError("Unexpected block execution order")
            events.append((group, position, visit))
            active[owner] = dict(group=group, position=position, visit=visit, scale=scale,
                                 finalize=kwargs.get("finalize", False), expert_calls=0,
                                 raw_norm=[], contribution_norm=[])
        return hook

    def norm_input(owner):
        def hook(module, inputs):
            record = active[owner]
            if "z" in record:
                raise RuntimeError("LN2 executed more than once in one visit")
            record["z"] = inputs[0].detach().clone()
        return hook

    def router(owner, visit):
        def hook(module, inputs, output):
            record = active[owner]
            if record["visit"] != visit or "probabilities" in record:
                raise RuntimeError("Wrong router visit or repeated router call")
            if output.shape != (1, positions.shape[1], experts):
                raise ValueError("Router must produce [1,N,E], with no head axis")
            logits = output.detach()
            if not torch.isfinite(logits).all():
                raise ValueError("NaN/Inf in router logits")
            record["logits"] = cpu(logits[0])
            record["probabilities"] = logits.softmax(dim=-1)
            record["mixed"] = torch.zeros_like(record["z"])
        return hook

    def expert(owner, index):
        def hook(module, inputs, output):
            record = active[owner]
            if index != record["expert_calls"]:
                raise RuntimeError("Expert order/count differs from dense execution")
            value = output.detach()
            weighted = record["probabilities"][..., index:index + 1] * value
            record["mixed"] = record["mixed"] + weighted
            record["raw_norm"].append(cpu(torch.linalg.vector_norm(value, dim=-1)[0]))
            record["contribution_norm"].append(cpu(torch.linalg.vector_norm(record["scale"] * weighted, dim=-1)[0]))
            record["expert_calls"] += 1
        return hook

    def final_input(owner):
        def hook(module, inputs):
            active[owner]["before_head"] = inputs[0].detach().clone()
        return hook

    def end(owner):
        def hook(module, inputs, output):
            record = active.pop(owner)
            if record["expert_calls"] != experts:
                raise RuntimeError("Not all dense experts executed")
            expected = record["z"] + record["scale"] * record["mixed"]
            observed = record["before_head"] if record["finalize"] else output.detach()
            torch.testing.assert_close(observed, expected, atol=atol, rtol=rtol)
            probabilities = record["probabilities"]
            error = (probabilities.sum(-1) - 1).abs().max().item()
            if error > 5e-6:
                raise RuntimeError("Expert weights do not sum to one along E")
            rows.append(dict(group=record["group"], position=record["position"], visit=record["visit"],
                scale=record["scale"], probabilities=cpu(probabilities[0]), logits=record["logits"],
                expert_output_norm=np.stack(record["raw_norm"], -1),
                contribution_norm=np.stack(record["contribution_norm"], -1),
                mixed_update_norm=cpu(torch.linalg.vector_norm(record["scale"] * record["mixed"], dim=-1)[0]),
                row_sum_error=error, residual_error=(observed - expected).abs().max().item()))
        return hook

    with torch.inference_mode():
        baseline = model(positions, fx=fx)
        try:
            for group in ("prefix", "core", "suffix"):
                for position, block in enumerate(getattr(model.loop, group)):
                    owner = (group, position)
                    handles.append(block.register_forward_pre_hook(begin(owner, group, position), with_kwargs=True))
                    for norm in (block.ln_2, *block.additional_ln_2):
                        handles.append(norm.register_forward_pre_hook(norm_input(owner)))
                    for visit, route in enumerate(block.visits):
                        handles.append(route.router.register_forward_hook(router(owner, visit)))
                    for index, module in enumerate(block.experts):
                        handles.append(module.register_forward_hook(expert(owner, index)))
                    if block.last_layer:
                        handles.append(block.ln_3.register_forward_pre_hook(final_input(owner)))
                    handles.append(block.register_forward_hook(end(owner)))
            prediction = model(positions, fx=fx)
        finally:
            for handle in handles:
                handle.remove()
            active.clear()
        if not torch.equal(prediction, baseline):
            raise RuntimeError("Expert hooks changed model prediction")
        if len(rows) != len(plan):
            raise RuntimeError("Missing logical block observations")
        if versions != {n: t._version for n, t in model.state_dict(keep_vars=True).items()}:
            raise RuntimeError("Inference modified model state")
        current_rng = rng()
        numpy_now = np.random.get_state()
        if (len(initial_rng) != len(current_rng) or
            not all(torch.equal(a, b) for a, b in zip(initial_rng, current_rng)) or
            python_rng != random.getstate() or numpy_rng[0] != numpy_now[0] or
            not np.array_equal(numpy_rng[1], numpy_now[1]) or numpy_rng[2:] != numpy_now[2:]):
            raise RuntimeError("Observation consumed public RNG")
        prediction_array = cpu(prediction[0, :, 0])
    arrays = {key: np.stack([r[key] for r in rows]) for key in (
        "probabilities", "logits", "expert_output_norm", "contribution_norm", "mixed_update_norm")}
    metadata = [dict(logical_index=i, group=r["group"], position=r["position"], visit=r["visit"],
                     expert_scale=r["scale"], owner=f"loop.{r['group']}.{r['position']}") for i, r in enumerate(rows)]
    report = dict(hooked_prediction_bitwise_equal=True, model_state_unmodified=True,
        public_rng_unchanged=True, hooks_removed=True, logical_visits=len(rows),
        expert_calls=len(rows) * experts, row_sum_max_error=max(r["row_sum_error"] for r in rows),
        residual_max_error=max(r["residual_error"] for r in rows), residual_atol=atol, residual_rtol=rtol)
    return arrays, metadata, prediction_array, report


def summarize(arrays, rows):
    """Equal-node summaries; these are not physical area/volume integrals."""
    import numpy as np
    pi = arrays["probabilities"].astype(np.float64)
    count = pi.shape[-1]
    log_pi = np.zeros_like(pi)
    np.log(pi, out=log_pi, where=pi > 0)
    entropy = -(pi * log_pi).sum(-1)
    arrays["entropy_nats"] = entropy
    if count > 1:
        arrays["normalized_entropy"] = entropy / np.log(count)
    summaries = []
    for index, row in enumerate(rows):
        summaries.append(dict(**row, mean_gate=pi[index].mean(0).tolist(),
            mean_contribution_norm=arrays["contribution_norm"][index].mean(0).tolist(),
            mean_expert_output_norm=arrays["expert_output_norm"][index].mean(0).tolist(),
            mean_mixed_update_norm=float(arrays["mixed_update_norm"][index].mean()),
            mean_entropy_nats=float(entropy[index].mean()),
            mean_normalized_entropy=float(arrays["normalized_entropy"][index].mean()) if count > 1 else None))
    return dict(averaging="equal weight per original spatial node; selected sample only, not area/volume weighted",
        expert_count=count, single_expert=count == 1,
        entropy_note="normalized entropy undefined for E=1; raw entropy is zero" if count == 1 else "entropy/log(E); 0 concentrated, 1 uniform",
        expert_alignment="only within the same physical block across visits; different blocks have independent expert banks",
        visits=summaries)
