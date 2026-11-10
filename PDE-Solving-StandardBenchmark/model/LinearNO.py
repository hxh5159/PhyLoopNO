"""Complete Standard LinearNO, separate from the legacy model registry.

Independent composition of the L2 primitives following LinearNO v3 Eq. 8–9
and HiPRL/LinearNO@3f2b80df Standard release topology/state layout. ``Model``
is the stable task-local export; registration/CLI wiring is a later phase.
Standard ``linearno_rank`` means absolute per-head M (release ``key_ratio``).
"""
from __future__ import annotations

import numpy as np
import torch
from torch import nn

from cdlno.linearno.attention import initialize_release_weights
from .Embedding import timestep_embedding
from .LinearNO_Attention import LinearNO, LinearNO_temp, LinearNO_Conv, LinearNO_Conv_temp


def _integer(value, name, minimum=1):
    if type(value) is not int or value < minimum:
        raise ValueError(f'{name} must be an integer >= {minimum}, got {value!r}')


def _unified_positions(H, W, ref):
    """Release [0,1]^2 distance features, computed without a default CUDA device.

    NumPy linspace then float32 conversion preserves the release's coordinate
    rounding (torch.linspace alone need not produce the same FP32 coordinates).
    """
    gx = torch.tensor(np.linspace(0, 1, H), dtype=torch.float32)
    gy = torch.tensor(np.linspace(0, 1, W), dtype=torch.float32)
    rx = torch.tensor(np.linspace(0, 1, ref), dtype=torch.float32)
    ry = torch.tensor(np.linspace(0, 1, ref), dtype=torch.float32)
    grid = torch.stack((gx[:, None].expand(H, W), gy[None, :].expand(H, W)), dim=-1)
    anchors = torch.stack((rx[:, None].expand(ref, ref), ry[None, :].expand(ref, ref)), dim=-1)
    delta = grid[:, :, None, None, :] - anchors[None, None, :, :, :]
    return delta.square().sum(-1).sqrt().reshape(1, H, W, ref * ref).contiguous()


class PointwiseMLP(nn.Module):
    """The release n_layers=0 MLP, retaining its exact registered key layout."""
    def __init__(self, n_input, n_hidden, n_output):
        super().__init__()
        self.linear_pre = nn.Sequential(nn.Linear(n_input, n_hidden), nn.GELU())
        self.linear_post = nn.Linear(n_hidden, n_output)
        self.linears = nn.ModuleList()

    def forward(self, x):
        return self.linear_post(self.linear_pre(x))


class LinearNOBlock(nn.Module):
    """Two pre-LN residuals; the final block ALSO applies LN and a point head."""
    def __init__(self, *, hidden, heads, rank, variant, dropout, mlp_ratio,
                 H, W, last_layer, out_dim):
        super().__init__()
        self.last_layer = last_layer
        self.ln_1 = nn.LayerNorm(hidden)
        attention = dict(plain=LinearNO, temp=LinearNO_temp,
                         conv=LinearNO_Conv, conv_temp=LinearNO_Conv_temp)[variant]
        kwargs = dict(heads=heads, dim_head=hidden // heads, dropout=dropout, key_ratio=rank)
        if variant in ('conv', 'conv_temp'):
            kwargs.update(H=H, W=W)
        self.Attn = attention(hidden, **kwargs)
        self.ln_2 = nn.LayerNorm(hidden)
        self.mlp = PointwiseMLP(hidden, hidden * mlp_ratio, hidden)
        if last_layer:
            self.ln_3 = nn.LayerNorm(hidden)
            self.mlp2 = nn.Linear(hidden, out_dim)

    def forward(self, features):
        features = self.Attn(self.ln_1(features)) + features
        features = self.mlp(self.ln_2(features)) + features
        if self.last_layer:
            features = self.mlp2(self.ln_3(features))
        return features


class Model(nn.Module):
    """``Model(x, fx, T=None) -> [B,N,out_dim]`` for the six Standard tasks.

    x: [B,N,space_dim]. fx: [B,N,fun_dim], or None when fun_dim=0.
    T: optional [B,1], requiring Time_Input=True. Unified position REPLACES x
    with [0,1]^2 reference distances; it never concatenates both encodings.
    Conv and unified position require N=H*W; other variants accept arbitrary N.
    Variant is independent of the future factory's architecture key --model.
    """
    def __init__(self, space_dim=1, n_layers=5, n_hidden=256, dropout=0., n_head=8,
                 Time_Input=False, act='gelu', mlp_ratio=1, fun_dim=1, out_dim=1,
                 ref=8, unified_pos=False, H=85, W=85, *,
                 linearno_variant='plain', linearno_rank=4):
        super().__init__()
        for name, value in dict(space_dim=space_dim, n_layers=n_layers, n_hidden=n_hidden,
                                n_head=n_head, mlp_ratio=mlp_ratio, out_dim=out_dim,
                                ref=ref, H=H, W=W, linearno_rank=linearno_rank).items():
            _integer(value, name)
        _integer(fun_dim, 'fun_dim', 0)
        if n_hidden % n_head:
            raise ValueError('n_hidden must be divisible by n_head')
        if linearno_variant not in ('plain', 'temp', 'conv', 'conv_temp'):
            raise ValueError('linearno_variant must be plain, temp, conv or conv_temp')
        if act != 'gelu':
            raise ValueError('Standard LinearNO release profiles require act=gelu')
        if type(Time_Input) is not bool or type(unified_pos) is not bool:
            raise ValueError('Time_Input and unified_pos must be bool')
        if Time_Input and n_hidden % 2:
            raise ValueError('Time_Input requires even n_hidden for the release time embedding')
        self.H, self.W, self.ref = H, W, ref
        self.unified_pos = unified_pos
        self.fun_dim, self.out_dim = fun_dim, out_dim
        self.linearno_variant, self.linearno_rank = linearno_variant, linearno_rank
        if unified_pos:
            self.register_buffer('pos', _unified_positions(H, W, ref), persistent=False)
        lift_channels = fun_dim + (ref * ref if unified_pos else space_dim)
        # The order below is part of the scientific initialization contract:
        # native allocations -> ONE whole-model apply -> placeholder draw.
        self.preprocess = PointwiseMLP(lift_channels, n_hidden * 2, n_hidden)
        self.Time_Input, self.n_hidden, self.space_dim = Time_Input, n_hidden, space_dim
        if Time_Input:
            self.time_fc = nn.Sequential(nn.Linear(n_hidden, n_hidden), nn.SiLU(),
                                         nn.Linear(n_hidden, n_hidden))
        self.blocks = nn.ModuleList([
            LinearNOBlock(hidden=n_hidden, heads=n_head, rank=linearno_rank,
                          variant=linearno_variant, dropout=dropout, mlp_ratio=mlp_ratio,
                          H=H, W=W, last_layer=(i == n_layers - 1), out_dim=out_dim)
            for i in range(n_layers)
        ])
        self.initialize_weights()
        self.placeholder = nn.Parameter(torch.rand(n_hidden, dtype=torch.float32) * (1 / n_hidden))

    _init_weights = staticmethod(initialize_release_weights)

    def initialize_weights(self):
        self.apply(self._init_weights)

    def forward(self, x, fx, T=None):
        if not isinstance(x, torch.Tensor) or x.ndim != 3 or not x.is_floating_point():
            raise ValueError('x must be a floating tensor [B,N,space_dim]')
        B, N, channels = x.shape
        if B < 1 or N < 1 or channels != self.space_dim:
            raise ValueError(f'x requires B,N > 0 and space_dim={self.space_dim}; got {tuple(x.shape)}')
        if (self.unified_pos or self.linearno_variant in ('conv', 'conv_temp')) and N != self.H * self.W:
            raise ValueError(f'LinearNO conv/unified position requires N = H * W = {self.H} * {self.W}; got {N}')
        if fx is None:
            if self.fun_dim != 0:
                raise ValueError(f'fx=None requires fun_dim=0; configured fun_dim={self.fun_dim}')
        elif (not isinstance(fx, torch.Tensor) or tuple(fx.shape) != (B, N, self.fun_dim)
              or fx.device != x.device or fx.dtype != x.dtype):
            raise ValueError(f'fx must be [B,N,fun_dim]=[{B},{N},{self.fun_dim}] on x device/dtype')
        if T is not None:
            if not self.Time_Input:
                raise ValueError('T requires Time_Input=True')
            if (not isinstance(T, torch.Tensor) or tuple(T.shape) != (B, 1)
                    or T.device != x.device or not T.is_floating_point()):
                raise ValueError('T must be a floating tensor [B,1] on x device')
        positions = (self.pos.repeat(B, 1, 1, 1).reshape(B, N, self.ref * self.ref)
                     if self.unified_pos else x)
        if fx is None:
            features = self.preprocess(positions) + self.placeholder[None, None, :]
        else:
            features = self.preprocess(torch.cat((positions, fx), dim=-1))
        if T is not None:
            # Release computes sinusoidal features in FP32. Casting only at the
            # Linear boundary also permits an explicitly converted double model.
            time = timestep_embedding(T, self.n_hidden).repeat(1, N, 1)
            features = features + self.time_fc(time.to(dtype=self.time_fc[0].weight.dtype))
        for block in self.blocks:
            features = block(features)
        return features
