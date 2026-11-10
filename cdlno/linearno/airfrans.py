"""AirfRANS LinearNO: Eq.8–9 with task topology of HiPRL release 3f2b80d.

Independent task wrapper composed from the L2-verified attention primitive.
Keeps release state names, unused temperature and initialization order. No task
data loader, loss, profile resolution or checkpoint side effects on import.
"""
from __future__ import annotations

import numpy as np
import torch
from torch import nn

from .attention import LinearNOAttention, initialize_release_weights, _positive_integer


class PointMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.linear_pre = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.GELU())
        self.linear_post = nn.Linear(hidden_dim, output_dim)
        self.linears = nn.ModuleList()

    def forward(self, x):
        return self.linear_post(self.linear_pre(x))


class AirfRANSBlock(nn.Module):
    def __init__(self, hidden, heads, rank, ratio, dropout, out_dim, last_layer):
        super().__init__()
        self.last_layer = last_layer
        self.ln_1 = nn.LayerNorm(hidden)
        self.Attn = LinearNOAttention(hidden, heads=heads, dim_head=hidden//heads,
                                      rank=rank, variant='airfrans', dropout=dropout)
        self.ln_2 = nn.LayerNorm(hidden)
        self.mlp = PointMLP(hidden, hidden*ratio, hidden)
        if last_layer:
            self.ln_3 = nn.LayerNorm(hidden)
            self.mlp2 = nn.Linear(hidden, out_dim)

    def forward(self, x):
        x = self.Attn(self.ln_1(x)) + x
        x = self.mlp(self.ln_2(x)) + x
        return self.mlp2(self.ln_3(x)) if self.last_layer else x


def single_graph(data, n):
    """Respect original Infer_test sampling which retains the source graph ptr."""
    batch, ptr = getattr(data, 'batch', None), getattr(data, 'ptr', None)
    integers = (torch.int32, torch.int64)
    if batch is not None:
        if not isinstance(batch, torch.Tensor) or batch.shape != (n,) or batch.dtype not in integers:
            raise ValueError('LinearNO AirfRANS data.batch must be an integer tensor [N]')
        if torch.any(batch != 0):
            raise ValueError('LinearNO AirfRANS requires one graph per forward; multi-graph batch unsupported')
    if ptr is not None:
        if not isinstance(ptr, torch.Tensor) or ptr.ndim != 1 or ptr.dtype not in integers or ptr.numel() != 2:
            raise ValueError('LinearNO AirfRANS single-graph ptr must contain [0,original_N]')
        if ptr[0] != 0 or ptr[1] < n or (ptr[1] != n and batch is None):
            raise ValueError('LinearNO AirfRANS ptr must match N or an original single graph before sampling')


class AirfRANSLinearNO(nn.Module):
    """forward(Data) -> [N,4]. Actual M is absolute linearno_rank, default32."""
    def __init__(self, space_dim=7, n_layers=8, n_hidden=256, dropout=0., n_head=8,
                 act='gelu', mlp_ratio=2, fun_dim=0, out_dim=4, linearno_rank=32,
                 ref=8, unified_pos=True, linear=True):
        super().__init__()
        for name, value in (('n_layers', n_layers), ('n_hidden', n_hidden), ('n_head', n_head),
                            ('mlp_ratio', mlp_ratio), ('linearno_rank', linearno_rank), ('ref', ref)):
            _positive_integer(value, name)
        if n_hidden % n_head:
            raise ValueError('n_hidden must be divisible by n_head')
        if (space_dim, fun_dim, out_dim) != (7, 0, 4):
            raise ValueError('AirfRANS requires space_dim=7, fun_dim=0, out_dim=4')
        if act != 'gelu':
            raise ValueError('AirfRANS release profile requires act=gelu')
        if unified_pos not in (False, True, 0, 1) or type(linear) is not bool:
            raise ValueError('unified_pos and unused release linear flag must be boolean')
        self.__name__ = 'LinearAttentionNeuralOperator'
        self.ref, self.unified_pos = ref, bool(unified_pos)
        self.preprocess = PointMLP(space_dim + (ref*ref if unified_pos else 0), 2*n_hidden, n_hidden)
        self.n_hidden, self.space_dim = n_hidden, space_dim
        self.blocks = nn.ModuleList([AirfRANSBlock(n_hidden, n_head, linearno_rank, mlp_ratio,
                                        dropout, out_dim, i == n_layers-1) for i in range(n_layers)])
        self.initialize_weights()
        self.placeholder = nn.Parameter((1/n_hidden)*torch.rand(n_hidden, dtype=torch.float))
        # Same float32 NumPy linspaces/x-major flatten as release; no RNG draw.
        gx = torch.tensor(np.linspace(-2, 4, ref), dtype=torch.float)
        gy = torch.tensor(np.linspace(-1.5, 1.5, ref), dtype=torch.float)
        self.register_buffer('reference', torch.stack(torch.meshgrid(gx, gy, indexing='ij'), -1).reshape(ref*ref, 2),
                             persistent=False)

    def initialize_weights(self):
        self.apply(initialize_release_weights)

    def get_grid(self, pos):
        if not isinstance(pos, torch.Tensor) or pos.ndim != 3 or pos.shape[-1] != 2 or min(pos.shape[:2]) < 1:
            raise ValueError('AirfRANS reference distances require raw pos [B,N,2] with B,N>0')
        if not pos.is_floating_point():
            raise ValueError('AirfRANS raw pos must be floating point')
        return ((pos[:, :, None, :] - self.reference.to(device=pos.device)[None, None, :, :])**2).sum(-1).sqrt().contiguous()

    def forward(self, data):
        x, pos = getattr(data, 'x', None), getattr(data, 'pos', None)
        if not isinstance(x, torch.Tensor) or x.ndim != 2 or x.shape[1] != 7 or x.shape[0] < 1:
            raise ValueError('LinearNO AirfRANS data.x must have shape [N,7], N>0')
        if not x.is_floating_point():
            raise ValueError('LinearNO AirfRANS data.x must be floating point')
        if not isinstance(pos, torch.Tensor) or pos.shape != (x.shape[0], 2):
            raise ValueError('LinearNO AirfRANS data.pos must have shape [N,2]; no implicit slicing')
        if pos.device != x.device or pos.dtype != x.dtype:
            raise ValueError('LinearNO AirfRANS data.x and data.pos must share device and floating dtype')
        single_graph(data, x.shape[0])
        features = x.unsqueeze(0)
        if self.unified_pos:
            features = torch.cat((features, self.get_grid(pos.unsqueeze(0))), dim=-1)
        hidden = self.preprocess(features) + self.placeholder[None, None, :]
        for block in self.blocks:
            hidden = block(hidden)
        return hidden[0]
