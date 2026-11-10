"""Scientific invariants of the released model and paper ablations."""
import importlib
import torch
import pytest
from phyloopno import build_model, resolve_config
from cdlno.linearno_loop.v5.core import V5LoopCore
from cdlno.linearno.attention import initialize_release_weights
from cdlno.linearno_loop.v5.checkpoint import save_pair, load_model


@pytest.mark.parametrize("qk,norms", [("independent", "visit_independent"),
                                      ("shared", "visit_independent"), ("shared", "shared")])
def test_parameter_ownership(qk, norms):
    model = build_model("elasticity", experts=1, qk_sharing=qk, core_norm_mode=norms)
    for block in model.loop.core:
        a, b = block.visits
        assert (a.to_q is b.to_q) == (qk == "shared")
        assert (a.to_k is b.to_k) == (qk == "shared")
        assert a.to_q is not a.to_k
        assert (a.temperature_q is b.temperature_q) == (qk == "shared")
        assert (a.router is b.router) == (qk == "shared" and norms == "shared")
        assert (block.norms_for_visit(0)[0] is block.norms_for_visit(1)[0]) == (norms == "shared")
        assert torch.equal(a.to_q.weight, b.to_q.weight)
        if qk == "independent":
            assert a.to_q.weight.data_ptr() != b.to_q.weight.data_ptr()
    assert model.loop.core[0].Attn.to_v is not model.loop.core[1].Attn.to_v
    assert model.loop.executed_depth == 8 and model.loop.unique_depth == 5


def core(qk="independent"):
    result = V5LoopCore(hidden=8, heads=2, rank=4, variant="temp", dropout=0., H=None, W=None,
        out_dim=1, expert_count=3, expert_width=9, prefix_blocks=1,
        recurrent_core_blocks=3, loop_repeats=2, suffix_blocks=1, qk_sharing=qk)
    result.apply(initialize_release_weights)
    result.finalize_initialization()
    return result


@pytest.mark.parametrize("qk", ["independent", "shared"])
def test_all_experts_run_and_shared_gradients_accumulate(qk):
    model = core(qk)
    calls = []
    handles = [e.register_forward_hook(lambda *unused: calls.append(1))
               for group in (model.prefix, model.core, model.suffix) for b in group for e in b.experts]
    output = model(torch.randn(2, 13, 8))
    output.square().mean().backward()
    assert len(calls) == 8 * 3
    for block in model.core:
        for expert in block.experts:
            assert expert.linear_post.weight.grad is not None
            assert torch.isfinite(expert.linear_post.weight.grad).all()
    for handle in handles:
        handle.remove()


def test_expert_residual_scaling():
    model = core()
    block = model.core[0]
    x = torch.randn(2, 13, 8)
    z = x + block.Attn(block.norms_for_visit(0)[0](x), visit_index=0)
    full = block(x, expert_scale=1.)
    half = block(x, expert_scale=.5)
    torch.testing.assert_close(half - z, .5 * (full - z), atol=2e-7, rtol=1e-5)


def test_single_expert_is_a_single_ffn():
    model = build_model("darcy")
    block = model.loop.core[0]
    x = torch.randn(1, 9, 128)
    assert torch.equal(block.visits[0].router(x).softmax(-1), torch.ones(1, 9, 1))
    assert model.v5_config["expert_count"] == 1


def test_parameter_comparison_is_separate_from_main_configuration():
    three = build_model("airfoil", experts=3)
    four = build_model("airfoil")
    native = importlib.import_module("PDE-Solving-StandardBenchmark.model.LinearNO").Model
    baseline = native(space_dim=2, fun_dim=0, out_dim=1, n_layers=8, n_hidden=128,
                      n_head=8, H=221, W=51, linearno_variant="conv_temp", linearno_rank=64)
    assert sum(p.numel() for p in three.parameters()) == 1457561
    assert sum(p.numel() for p in four.parameters()) == 1623713
    assert sum(p.numel() for p in baseline.parameters()) == 1765889


@pytest.mark.parametrize("qk,norms", [("independent", "visit_independent"),
                                      ("shared", "visit_independent"), ("shared", "shared")])
def test_checkpoint_preserves_sharing_and_predictions(tmp_path, qk, norms):
    model = build_model("elasticity", experts=1, qk_sharing=qk, core_norm_mode=norms)
    model.eval()
    positions = torch.randn(1, 9, 2)
    expected = model(positions, None)
    save_pair(tmp_path / "run", model, model.v5_config)
    loaded, metadata = load_model(tmp_path / "run")
    loaded.eval()
    assert metadata["resolved_config"]["qk_sharing"] == qk
    assert torch.equal(expected, loaded(positions, None))
    visits = loaded.loop.core[0].visits
    assert (visits[0].to_q is visits[1].to_q) == (qk == "shared")
