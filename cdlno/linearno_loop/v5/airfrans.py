"""AirfRANS V5 wrapper preserving native Data/reference semantics."""
import numpy as np
import torch
from torch import nn

from cdlno.linearno.airfrans import (AirfRANSLinearNO, PointMLP,
                                     initialize_release_weights)
from .construction import LoopForwardViewV5, constructor_config
from .core import V5LoopCore


class LoopedAirfRANSModelV5(LoopForwardViewV5, AirfRANSLinearNO):
    def __init__(self, *, space_dim, fun_dim, out_dim, time_input, ref, unified_pos,
                 hidden_width, grid_height, grid_width, actual_M, heads, variant,
                 dropout, activation, expert_count, expert_width, prefix_blocks,
                 recurrent_core_blocks, loop_repeats, suffix_blocks,
                 core_norm_mode, architecture, qk_sharing="independent"):
        kwargs = {key: value for key, value in locals().items() if key != "self"}
        constructor_config(kwargs)
        nn.Module.__init__(self)
        if (architecture != self.architecture or variant != "airfrans" or activation != "gelu"
                or time_input or (space_dim, fun_dim, out_dim) != (7, 0, 4)):
            raise ValueError("invalid V5 AirfRANS contract")
        self.__name__ = "PartialShareFeatureGateV5"
        self.ref, self.unified_pos = ref, bool(unified_pos)
        self.preprocess = PointMLP(space_dim + (ref * ref if unified_pos else 0),
                                   hidden_width * 2, hidden_width)
        self.n_hidden, self.space_dim = hidden_width, space_dim
        self.loop = V5LoopCore(hidden=hidden_width, heads=heads, rank=actual_M,
            variant=variant, dropout=dropout, H=grid_height, W=grid_width, out_dim=out_dim,
            expert_count=expert_count, expert_width=expert_width, prefix_blocks=prefix_blocks,
            recurrent_core_blocks=recurrent_core_blocks, loop_repeats=loop_repeats,
            suffix_blocks=suffix_blocks, core_norm_mode=core_norm_mode,
            qk_sharing=qk_sharing)
        self.apply(initialize_release_weights)
        self.placeholder = nn.Parameter(torch.rand(hidden_width, dtype=torch.float32) / hidden_width)
        self.loop.finalize_initialization()
        gx = torch.tensor(np.linspace(-2, 4, ref), dtype=torch.float32)
        gy = torch.tensor(np.linspace(-1.5, 1.5, ref), dtype=torch.float32)
        self.register_buffer("reference", torch.stack(torch.meshgrid(gx, gy, indexing="ij"), -1).reshape(ref * ref, 2), persistent=False)
