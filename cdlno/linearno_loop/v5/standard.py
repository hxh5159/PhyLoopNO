"""Standard six-task V5 wrapper preserving the native forward contract."""
import importlib

import torch
from torch import nn

from .construction import LoopForwardViewV5, constructor_config
from .core import V5LoopCore

native = importlib.import_module("PDE-Solving-StandardBenchmark.model.LinearNO")


class LoopedStandardModelV5(LoopForwardViewV5, native.Model):
    def __init__(self, *, space_dim, fun_dim, out_dim, time_input, ref, unified_pos,
                 hidden_width, grid_height, grid_width, actual_M, heads, variant,
                 dropout, activation, expert_count, expert_width, prefix_blocks,
                 recurrent_core_blocks, loop_repeats, suffix_blocks,
                 core_norm_mode, architecture, qk_sharing="independent"):
        kwargs = {key: value for key, value in locals().items() if key != "self"}
        constructor_config(kwargs)
        nn.Module.__init__(self)
        if architecture != self.architecture or activation != "gelu":
            raise ValueError("invalid V5 Standard architecture/activation")
        self.H, self.W, self.ref = grid_height, grid_width, ref
        self.unified_pos = bool(unified_pos)
        self.fun_dim, self.out_dim, self.space_dim = fun_dim, out_dim, space_dim
        self.linearno_variant, self.linearno_rank = variant, actual_M
        if self.unified_pos:
            self.register_buffer("pos", native._unified_positions(grid_height, grid_width, ref), persistent=False)
        channels = fun_dim + (ref * ref if self.unified_pos else space_dim)
        self.preprocess = native.PointwiseMLP(channels, hidden_width * 2, hidden_width)
        self.Time_Input, self.n_hidden = bool(time_input), hidden_width
        if self.Time_Input:
            self.time_fc = nn.Sequential(nn.Linear(hidden_width, hidden_width), nn.SiLU(),
                                         nn.Linear(hidden_width, hidden_width))
        self.loop = V5LoopCore(hidden=hidden_width, heads=heads, rank=actual_M,
            variant=variant, dropout=dropout, H=grid_height, W=grid_width, out_dim=out_dim,
            expert_count=expert_count, expert_width=expert_width, prefix_blocks=prefix_blocks,
            recurrent_core_blocks=recurrent_core_blocks, loop_repeats=loop_repeats,
            suffix_blocks=suffix_blocks, core_norm_mode=core_norm_mode,
            qk_sharing=qk_sharing)
        self.apply(native.initialize_release_weights)
        self.placeholder = nn.Parameter(torch.rand(hidden_width, dtype=torch.float32) / hidden_width)
        self.loop.finalize_initialization()
