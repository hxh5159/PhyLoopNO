"""ShapeNet-Car V5 wrapper preserving tuple and release-rank semantics."""
import torch
from torch import nn

from cdlno.linearno.shapenet import (PointMLP, ShapeNetLinearNO,
                                     initialize_release_weights)
from .construction import LoopForwardViewV5, constructor_config
from .core import V5LoopCore


class LoopedShapeNetModelV5(LoopForwardViewV5, ShapeNetLinearNO):
    def __init__(self, *, space_dim, fun_dim, out_dim, time_input, ref, unified_pos,
                 hidden_width, grid_height, grid_width, actual_M, heads, variant,
                 dropout, activation, expert_count, expert_width, prefix_blocks,
                 recurrent_core_blocks, loop_repeats, suffix_blocks,
                 core_norm_mode, architecture, qk_sharing="independent"):
        kwargs = {key: value for key, value in locals().items() if key != "self"}
        constructor_config(kwargs)
        nn.Module.__init__(self)
        if (architecture != self.architecture or variant != "shapenet" or activation != "gelu"
                or (space_dim + fun_dim, out_dim) != (7, 4)
                or actual_M % (hidden_width // heads)):
            raise ValueError("invalid V5 ShapeNet contract or rank")
        self.H, self.W, self.ref = grid_height, grid_width, ref
        self.unified_pos, self.Time_Input = bool(unified_pos), bool(time_input)
        if self.unified_pos:
            self.register_buffer("pos", self.get_grid(), persistent=False)
        channels = ref * ref if self.unified_pos else space_dim + fun_dim
        self.preprocess = PointMLP(channels, hidden_width * 2, hidden_width)
        self.n_hidden, self.space_dim = hidden_width, space_dim
        if self.Time_Input:
            self.time_fc = nn.Sequential(nn.Linear(hidden_width, hidden_width), nn.SiLU(),
                                         nn.Linear(hidden_width, hidden_width))
        self.loop = V5LoopCore(hidden=hidden_width, heads=heads, rank=actual_M,
            variant=variant, dropout=dropout, H=grid_height, W=grid_width, out_dim=out_dim,
            expert_count=expert_count, expert_width=expert_width, prefix_blocks=prefix_blocks,
            recurrent_core_blocks=recurrent_core_blocks, loop_repeats=loop_repeats,
            suffix_blocks=suffix_blocks, core_norm_mode=core_norm_mode,
            qk_sharing=qk_sharing)
        self.apply(initialize_release_weights)
        self.placeholder = nn.Parameter(torch.rand(hidden_width, dtype=torch.float32) / hidden_width)
        self.loop.finalize_initialization()
