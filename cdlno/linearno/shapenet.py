"""ShapeNet-Car LinearNO, Eq.8–9 and task topology of release 3f2b80d.

Reuses only the independently verified LinearNO attention. No old Transolver,
data loader, loss or entry import. Stable class path for local whole objects.
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


class ShapeNetBlock(nn.Module):
    def __init__(self, hidden, heads, rank, ratio, dropout, out_dim, last_layer):
        super().__init__()
        self.last_layer = last_layer
        self.ln_1 = nn.LayerNorm(hidden)
        self.Attn = LinearNOAttention(hidden, heads=heads, dim_head=hidden//heads,
                                     rank=rank, variant='shapenet', dropout=dropout)
        self.ln_2 = nn.LayerNorm(hidden)
        self.mlp = PointMLP(hidden, hidden*ratio, hidden)
        if last_layer:
            self.ln_3 = nn.LayerNorm(hidden)
            self.mlp2 = nn.Linear(hidden, out_dim)

    def forward(self, x):
        x = x + self.Attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return self.mlp2(self.ln_3(x)) if self.last_layer else x


def single_graph(data, n):
    """Full-point Car interface: reject graph concatenation before attention."""
    batch, ptr = getattr(data, 'batch', None), getattr(data, 'ptr', None)
    integers = (torch.int32, torch.int64)
    if batch is not None:
        if not isinstance(batch, torch.Tensor) or batch.shape != (n,) or batch.dtype not in integers:
            raise ValueError('ShapeNet LinearNO data.batch must be an integer tensor [N]')
        if torch.any(batch != 0):
            raise ValueError('ShapeNet LinearNO supports one graph per forward; batch>1 is unsupported')
    if ptr is not None:
        if not isinstance(ptr, torch.Tensor) or ptr.ndim != 1 or ptr.dtype not in integers or ptr.numel() != 2:
            raise ValueError('ShapeNet LinearNO single graph ptr must be [0,N]')
        if ptr[0] != 0 or ptr[1] != n:
            raise ValueError('ShapeNet LinearNO full-point ptr must equal [0,N]')


class ShapeNetLinearNO(nn.Module):
    """forward((cfd_data, geom)) -> [N,4]; geom remains release-unused.

    linearno_rank is actual M, not the release multiplier key_ratio. For release
    configurations M/(hidden/heads) is the positive integer original key_ratio.
    Default 3+4 input features preserves the fixed release constructor metadata.
    """
    def __init__(self, space_dim=3, n_layers=8, n_hidden=256, dropout=0., n_head=8,
                 Time_Input=False, act='gelu', mlp_ratio=2, fun_dim=4, out_dim=4,
                 linearno_rank=32, ref=8, unified_pos=False, H=85, W=85, isregular=False):
        super().__init__()
        for name, value in (('n_layers', n_layers), ('n_hidden', n_hidden), ('n_head', n_head),
                            ('mlp_ratio', mlp_ratio), ('linearno_rank', linearno_rank),
                            ('ref', ref), ('H', H), ('W', W)):
            _positive_integer(value, name)
        if n_hidden % n_head:
            raise ValueError('n_hidden must be divisible by n_head')
        if linearno_rank % (n_hidden//n_head):
            raise ValueError('ShapeNet actual M must equal integer key_ratio * head_dim')
        if type(space_dim) is not int or type(fun_dim) is not int or min(space_dim, fun_dim) < 0 or space_dim+fun_dim != 7 or out_dim != 4:
            raise ValueError('ShapeNet requires space_dim+fun_dim=7 and out_dim=4')
        if act != 'gelu':
            raise ValueError('ShapeNet release profile requires act=gelu')
        if any(v not in (False, True, 0, 1) for v in (Time_Input, unified_pos, isregular)):
            raise ValueError('Time_Input, unified_pos and release-unused isregular must be boolean')
        # Release's unified replacement cannot supply the extra fun_dim channels:
        # its wrapper fixes fx=None. Reject that invalid combination explicitly.
        if unified_pos and fun_dim:
            raise ValueError('ShapeNet unified_pos replaces x and requires fun_dim=0 with the tuple wrapper')
        self.H, self.W, self.ref = H, W, ref
        self.unified_pos, self.Time_Input = bool(unified_pos), bool(Time_Input)
        if self.unified_pos:
            self.register_buffer('pos', self.get_grid(), persistent=False)
        self.preprocess = PointMLP(ref*ref if unified_pos else space_dim+fun_dim, n_hidden*2, n_hidden)
        self.n_hidden, self.space_dim = n_hidden, space_dim
        if Time_Input:
            # Official wrapper always T=None: retained only for strict keys.
            self.time_fc = nn.Sequential(nn.Linear(n_hidden, n_hidden), nn.SiLU(), nn.Linear(n_hidden, n_hidden))
        self.blocks = nn.ModuleList([ShapeNetBlock(n_hidden, n_head, linearno_rank, mlp_ratio,
                                    dropout, out_dim, i == n_layers-1) for i in range(n_layers)])
        self.initialize_weights()
        self.placeholder = nn.Parameter((1/n_hidden)*torch.rand(n_hidden, dtype=torch.float))

    def initialize_weights(self):
        self.apply(initialize_release_weights)

    def get_grid(self, batchsize=1):
        _positive_integer(batchsize, 'batchsize')
        def axis(end):
            return torch.tensor(np.linspace(0, 1, end), dtype=torch.float)
        grid = torch.stack(torch.meshgrid(axis(self.H), axis(self.W), indexing='ij'), -1)
        reference = torch.stack(torch.meshgrid(axis(self.ref), axis(self.ref), indexing='ij'), -1)
        distance = ((grid[:, :, None, None, :] - reference[None, None, :, :, :])**2).sum(-1).sqrt()
        return distance.reshape(1, self.H, self.W, self.ref*self.ref).repeat(batchsize, 1, 1, 1).contiguous()

    def forward(self, data):
        if not isinstance(data, (tuple, list)) or len(data) != 2:
            raise ValueError('ShapeNet LinearNO expects (cfd_data, geom)')
        cfd_data, _ = data
        x = getattr(cfd_data, 'x', None)
        if not isinstance(x, torch.Tensor) or x.ndim != 2 or x.shape[0] < 1 or x.shape[1] != 7:
            raise ValueError('ShapeNet LinearNO requires data.x [N,7] with N>0')
        if not x.is_floating_point():
            raise ValueError('ShapeNet LinearNO data.x must be floating point')
        single_graph(cfd_data, x.shape[0])
        if self.unified_pos:
            if x.shape[0] != self.H*self.W:
                raise ValueError('ShapeNet unified_pos requires N=H*W')
            features = self.pos.reshape(1, self.H*self.W, self.ref*self.ref)
        else:
            features = x.unsqueeze(0)
        hidden = self.preprocess(features) + self.placeholder[None, None, :]
        for block in self.blocks:
            hidden = block(hidden)
        return hidden[0]
