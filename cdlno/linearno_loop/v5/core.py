"""Task-independent V5 loop core and dense point-expert blocks."""
from __future__ import annotations

import torch
from torch import nn

from .operator import PartialSharedLinearNOOperator


CORE_NORM_MODES = ("visit_independent", "shared")


class DensePointExpert(nn.Module):
    """Native two-Linear point FFN key/layout with configurable hidden width."""
    def __init__(self, hidden, expert_width):
        super().__init__()
        self.linear_pre = nn.Sequential(nn.Linear(hidden, expert_width), nn.GELU())
        self.linear_post = nn.Linear(expert_width, hidden)
        self.linears = nn.ModuleList()

    def forward(self, x):
        return self.linear_post(self.linear_pre(x))


class V5PhysicalBlock(nn.Module):
    def __init__(self, *, hidden, heads, rank, variant, dropout, H, W, out_dim,
                 expert_count, expert_width, visit_count, last_layer,
                 core_norm_mode="visit_independent", qk_sharing="independent"):
        super().__init__()
        if core_norm_mode not in CORE_NORM_MODES:
            raise ValueError("invalid V5 core_norm_mode")
        self.last_layer = bool(last_layer)
        self.visit_count = visit_count
        self.core_norm_mode = core_norm_mode
        self.ln_1 = nn.LayerNorm(hidden)
        self.Attn = PartialSharedLinearNOOperator(
            hidden, heads=heads, rank=rank, variant=variant, dropout=dropout,
            H=H if variant in ("conv", "conv_temp") else None,
            W=W if variant in ("conv", "conv_temp") else None,
            visit_count=visit_count, expert_count=expert_count, qk_sharing=qk_sharing)
        if qk_sharing == "shared" and core_norm_mode == "shared" and expert_count == 1:
            # Full-share single-FFN ablation: even the inert E=1 routers share.
            for visit in self.Attn.visits[1:]:
                visit.router = self.Attn.visits[0].router
        self.ln_2 = nn.LayerNorm(hidden)
        extra_visits = visit_count - 1 if core_norm_mode == "visit_independent" else 0
        self.additional_ln_1 = nn.ModuleList(
            nn.LayerNorm(hidden) for _ in range(extra_visits)
        )
        self.additional_ln_2 = nn.ModuleList(
            nn.LayerNorm(hidden) for _ in range(extra_visits)
        )
        self.experts = nn.ModuleList([DensePointExpert(hidden, expert_width)
                                      for _ in range(expert_count)])
        if self.last_layer:
            self.ln_3 = nn.LayerNorm(hidden)
            self.mlp2 = nn.Linear(hidden, out_dim)

    @property
    def visits(self):
        return self.Attn.visits

    def norms_for_visit(self, visit_index):
        if type(visit_index) is not int or not 0 <= visit_index < self.visit_count:
            raise ValueError("visit_index is outside this V5 block")
        if visit_index == 0 or self.core_norm_mode == "shared":
            return self.ln_1, self.ln_2
        return (self.additional_ln_1[visit_index - 1],
                self.additional_ln_2[visit_index - 1])

    def forward(self, x, *, visit_index=0, expert_scale=1., finalize=False):
        ln_1, ln_2 = self.norms_for_visit(visit_index)
        z = x + self.Attn(ln_1(x), visit_index=visit_index)
        u = ln_2(z)
        probabilities = self.visits[visit_index].router(u).softmax(dim=-1)
        mixed = torch.zeros_like(z)
        for index, expert in enumerate(self.experts):
            mixed = mixed + probabilities[..., index:index + 1] * expert(u)
        result = z + expert_scale * mixed
        if finalize:
            if not self.last_layer:
                raise ValueError("only the final suffix block can finalize")
            return self.mlp2(self.ln_3(result))
        return result


class V5LoopCore(nn.Module):
    def __init__(self, *, hidden, heads, rank, variant, dropout, H, W, out_dim,
                 expert_count, expert_width, prefix_blocks, recurrent_core_blocks,
                 loop_repeats, suffix_blocks, core_norm_mode="visit_independent",
                 qk_sharing="independent"):
        super().__init__()
        for name, value in (("hidden", hidden), ("heads", heads), ("rank", rank),
                            ("expert_count", expert_count), ("expert_width", expert_width),
                            ("prefix_blocks", prefix_blocks),
                            ("recurrent_core_blocks", recurrent_core_blocks),
                            ("loop_repeats", loop_repeats), ("suffix_blocks", suffix_blocks)):
            if type(value) is not int or value < 1:
                raise ValueError(name + " must be a positive integer")
        if hidden % heads:
            raise ValueError("hidden must be divisible by heads")
        if core_norm_mode not in CORE_NORM_MODES:
            raise ValueError("invalid V5 core_norm_mode")
        common = dict(hidden=hidden, heads=heads, rank=rank, variant=variant,
                      dropout=dropout, H=H, W=W, out_dim=out_dim,
                      expert_count=expert_count, expert_width=expert_width,
                      qk_sharing=qk_sharing)
        self.prefix = nn.ModuleList([V5PhysicalBlock(
                                     **common, visit_count=1, last_layer=False,
                                     core_norm_mode=core_norm_mode)
                                     for _ in range(prefix_blocks)])
        self.core = nn.ModuleList([V5PhysicalBlock(
                                   **common, visit_count=loop_repeats, last_layer=False,
                                   core_norm_mode=core_norm_mode)
                                   for _ in range(recurrent_core_blocks)])
        self.suffix = nn.ModuleList([V5PhysicalBlock(
                                      **common, visit_count=1,
                                      core_norm_mode=core_norm_mode,
                                      last_layer=index == suffix_blocks - 1)
                                     for index in range(suffix_blocks)])
        self.prefix_blocks = prefix_blocks
        self.recurrent_core_blocks = recurrent_core_blocks
        self.loop_repeats = loop_repeats
        self.suffix_blocks = suffix_blocks
        self.expert_count = expert_count
        self.expert_width = expert_width
        self.core_norm_mode = core_norm_mode
        self.qk_sharing = qk_sharing
        self.residual_mode = "operator_1_expert_1_over_r"
        self.validate_structure()

    @property
    def unique_depth(self):
        return self.prefix_blocks + self.recurrent_core_blocks + self.suffix_blocks

    @property
    def executed_depth(self):
        return self.prefix_blocks + self.recurrent_core_blocks * self.loop_repeats + self.suffix_blocks

    def finalize_initialization(self):
        for block in (*self.prefix, *self.core, *self.suffix):
            block.Attn.synchronize_visit_initialization()

    def validate_structure(self):
        if len(self.prefix) != self.prefix_blocks or len(self.core) != self.recurrent_core_blocks or len(self.suffix) != self.suffix_blocks:
            raise ValueError("V5 physical block count mismatch")
        seen = set()
        for group, count in ((self.prefix, 1), (self.core, self.loop_repeats), (self.suffix, 1)):
            for block in group:
                if len(block.visits) != count or len(block.experts) != self.expert_count:
                    raise ValueError("V5 visit/expert ownership mismatch")
                expected_norms = count - 1 if (self.core_norm_mode == "visit_independent"
                                                and count > 1) else 0
                if (len(block.additional_ln_1) != expected_norms or
                        len(block.additional_ln_2) != expected_norms):
                    raise ValueError("V5 visit norm ownership mismatch")
                for parameter in block.parameters():
                    if id(parameter) in seen:
                        raise ValueError("V5 parameter registered by more than one owner")
                    seen.add(id(parameter))
        if not self.suffix[-1].last_layer or any(block.last_layer for block in (*self.prefix, *self.core, *self.suffix[:-1])):
            raise ValueError("only final suffix may own the output head")

    def visit_schedule(self):
        result = [("prefix", index, 0) for index in range(self.prefix_blocks)]
        result += [("core", index, round_index) for round_index in range(self.loop_repeats)
                   for index in range(self.recurrent_core_blocks)]
        result += [("suffix", index, 0) for index in range(self.suffix_blocks)]
        return tuple(result)

    def forward(self, x):
        self.validate_structure()
        for block in self.prefix:
            x = block(x, visit_index=0, expert_scale=1.)
        for round_index in range(self.loop_repeats):
            for block in self.core:
                x = block(x, visit_index=round_index, expert_scale=1. / self.loop_repeats)
        for index, block in enumerate(self.suffix):
            x = block(x, visit_index=0, expert_scale=1.,
                      finalize=index == self.suffix_blocks - 1)
        return x
