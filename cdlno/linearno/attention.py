"""Pure LinearNO attention primitives, independently implemented from Eq. 8–9.

Topology/state names follow HiPRL/LinearNO@3f2b80df task-specific release classes.
No slice normalization/latent attention, residual/FFN, task model, or registration.

Initialization intentionally has the release's two stages: constructors perform
normal PyTorch allocation; the enclosing model applies initialize_release_weights
ONCE after all submodules exist. For a standalone initialized primitive use
``attention.apply(initialize_release_weights)``. Automatically applying it inside
a primitive would change the enclosing release model's RNG/initialization order.
"""
from __future__ import annotations

import math
import torch
from torch import nn
from timm.layers import trunc_normal_

VARIANTS = ('plain', 'temp', 'conv', 'conv_temp', 'airfrans', 'shapenet')


def _positive_integer(value, name):
    if type(value) is not int or value < 1:
        raise ValueError(f'{name} must be a positive integer, got {value!r}')


def initialize_release_weights(module):
    """Apply as model.apply(callback), exactly once at the outer model boundary.

    Matches the fixed release callback; temperatures are untouched. L2 has no
    placeholder; a future full model must create its placeholder AFTER apply.
    """
    if isinstance(module, nn.Linear):
        # Keep the release RNG algorithm; newer torch.nn.init implementations
        # can sample the same distribution using a different random stream.
        trunc_normal_(module.weight, std=0.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, (nn.LayerNorm, nn.BatchNorm1d)):
        nn.init.ones_(module.weight)
        nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Conv2d):
        nn.init.kaiming_normal_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)


class LinearNOAttention(nn.Module):
    """Input/output [B,N,dim]; rank is total M PER HEAD, not head_dim's ratio.

    Task-local wrappers preserve official constructor semantics and state keys.
    Native PyTorch forward hooks on to_q/to_k support tests without a persistent
    diagnostic cache. Normal forward returns only the output tensor.
    """
    def __init__(self, dim, *, heads, dim_head, rank, variant, dropout=0., H=None, W=None, kernel=3):
        super().__init__()
        for name, value in (('dim', dim), ('heads', heads), ('dim_head', dim_head), ('rank', rank)):
            _positive_integer(value, name)
        if dim % heads:
            raise ValueError(f'dim={dim} must be divisible by heads={heads}')
        if variant not in VARIANTS:
            raise ValueError(f'unknown LinearNO attention variant {variant!r}')
        if isinstance(dropout, bool) or not isinstance(dropout, (int, float)) or not math.isfinite(dropout) or not 0 <= dropout <= 1:
            raise ValueError('dropout must be finite and in [0,1]')
        structured = variant in ('conv', 'conv_temp')
        if structured:
            for name, value in (('H', H), ('W', W), ('kernel', kernel)):
                _positive_integer(value, name)
            if kernel % 2 != 1:
                raise ValueError('kernel must be odd to preserve H/W without resampling')
        elif H is not None or W is not None:
            raise ValueError('H/W are only accepted by conv variants')
        self.dim, self.heads, self.dim_head, self.rank = dim, heads, dim_head, rank
        self.variant = variant
        inner_dim = heads * dim_head
        if structured:
            self.H, self.W = H, W
        else:
            if variant == 'airfrans':
                # Release-compatible inert attributes; neither used in forward.
                self.scale = dim_head ** -0.5
                self.softmax = nn.Softmax(dim=-1)
            self.dropout = nn.Dropout(dropout)  # release's unused attention dropout
        if variant in ('temp', 'conv_temp'):
            self.temperature_q = nn.Parameter(torch.full((1, heads, 1, 1), 0.5))
            self.temperature_k = nn.Parameter(torch.full((1, heads, 1, 1), 0.5))
        elif variant == 'shapenet':
            # Intentional release spelling; identity key mapping is reversible.
            self.tempreature_q = nn.Parameter(torch.full((1, heads, 1, 1), 0.5))
            self.tempreature_k = nn.Parameter(torch.full((1, heads, 1, 1), 0.5))
        elif variant == 'airfrans':
            self.temperature = nn.Parameter(torch.full((1, heads, 1, 1), 0.5))
        self.in_project_x = (nn.Conv2d(dim, inner_dim, kernel, stride=1, padding=kernel//2)
                             if structured else nn.Linear(dim, inner_dim))
        # A single small projection of each kind acts identically on every head.
        self.to_q = nn.Linear(dim_head, rank, bias=False)
        self.to_k = nn.Linear(dim_head, rank, bias=False)
        self.to_v = nn.Linear(dim_head, dim_head, bias=False)
        if structured or variant == 'shapenet':
            self.to_out = nn.Sequential(nn.Linear(inner_dim, dim), nn.GELU(),
                                        nn.Linear(dim, dim), nn.Dropout(dropout))
        else:
            self.to_out = nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout))

    def forward(self, x):
        if not isinstance(x, torch.Tensor) or x.ndim != 3:
            raise ValueError('LinearNO attention requires a tensor [B,N,dim]')
        B, N, channels = x.shape
        if B < 1 or N < 1 or channels != self.dim:
            raise ValueError(f'LinearNO attention expected B,N > 0 and dim={self.dim}; got {tuple(x.shape)}')
        if not x.is_floating_point():
            raise TypeError('LinearNO attention input must be floating point')
        if self.variant in ('conv', 'conv_temp'):
            if N != self.H * self.W:
                raise ValueError(f'LinearNO conv requires N = H * W = {self.H} * {self.W}; got N={N}')
            grid = x.transpose(1, 2).reshape(B, channels, self.H, self.W)
            projected = self.in_project_x(grid)
            features = projected.reshape(B, self.heads, self.dim_head, N).transpose(-1, -2)
        else:
            projected = self.in_project_x(x)
            features = projected.reshape(B, N, self.heads, self.dim_head).transpose(1, 2)
            if self.variant == 'airfrans':
                features = features.contiguous()  # release Air layout under AMP
        q_logits, k_logits, values = self.to_q(features), self.to_k(features), self.to_v(features)
        if self.variant in ('temp', 'conv_temp'):
            q_logits = q_logits / self.temperature_q.clamp(0.01, 1.)
            k_logits = k_logits / self.temperature_k.clamp(0.01, 1.)
        elif self.variant == 'shapenet':
            q_logits = q_logits / self.tempreature_q.clamp(0.1, 2.)
            k_logits = k_logits / self.tempreature_k.clamp(0.1, 2.)
        queries = q_logits.softmax(dim=-1)
        keys = k_logits.softmax(dim=-2)
        # Axes: b=batch, h=head, n=point, m=rank, d=value channel.
        context = torch.einsum('bhnm,bhnd->bhmd', keys, values)
        readout = torch.einsum('bhnm,bhmd->bhnd', queries, context)
        merged = readout.transpose(1, 2).reshape(B, N, self.heads * self.dim_head)
        return self.to_out(merged)
