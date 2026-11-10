"""V5 LinearNO operator with a shared body and visit-owned Q/K routing."""
from __future__ import annotations

import torch
from torch import nn

from cdlno.linearno.attention import LinearNOAttention


class VisitRouting(nn.Module):
    def __init__(self, *, dim_head, rank, hidden, expert_count, variant):
        super().__init__()
        self.to_q = nn.Linear(dim_head, rank, bias=False)
        self.to_k = nn.Linear(dim_head, rank, bias=False)
        if variant in ("temp", "conv_temp"):
            self.temperature_q = nn.Parameter(torch.full((1, hidden // dim_head, 1, 1), .5))
            self.temperature_k = nn.Parameter(torch.full((1, hidden // dim_head, 1, 1), .5))
        elif variant == "shapenet":
            self.tempreature_q = nn.Parameter(torch.full((1, hidden // dim_head, 1, 1), .5))
            self.tempreature_k = nn.Parameter(torch.full((1, hidden // dim_head, 1, 1), .5))
        self.router = nn.Linear(hidden, expert_count, bias=True)

    def zero_router(self):
        with torch.no_grad():
            self.router.weight.zero_()
            self.router.bias.zero_()


class PartialSharedLinearNOOperator(nn.Module):
    """Keep native non-Q/K modules once and own Q/K per logical visit."""
    def __init__(self, dim, *, heads, rank, variant, dropout=0., H=None, W=None,
                 visit_count=1, expert_count=2, qk_sharing="independent"):
        super().__init__()
        if type(visit_count) is not int or visit_count < 1:
            raise ValueError("visit_count must be positive")
        if qk_sharing not in ("independent", "shared"):
            raise ValueError("qk_sharing must be independent or shared")
        self.qk_sharing = qk_sharing
        native = LinearNOAttention(dim, heads=heads, dim_head=dim // heads, rank=rank,
                                   variant=variant, dropout=dropout, H=H, W=W)
        self.dim, self.heads, self.dim_head = dim, heads, dim // heads
        self.rank, self.variant = rank, variant
        if variant in ("conv", "conv_temp"):
            self.H, self.W = H, W
        self.in_project_x = native.in_project_x
        self.to_v = native.to_v
        self.to_out = native.to_out
        if variant == "airfrans":
            self.temperature = native.temperature
            self.scale = native.scale
            self.softmax = native.softmax
            self.dropout = native.dropout
        elif variant not in ("conv", "conv_temp"):
            self.dropout = native.dropout
        visits = []
        for index in range(visit_count):
            visit = VisitRouting(dim_head=self.dim_head, rank=rank, hidden=dim,
                                 expert_count=expert_count, variant=variant)
            if index == 0:
                visit.to_q = native.to_q
                visit.to_k = native.to_k
                for name in ("temperature_q", "temperature_k", "tempreature_q", "tempreature_k"):
                    if hasattr(native, name):
                        setattr(visit, name, getattr(native, name))
            visits.append(visit)
        if qk_sharing == "shared":
            # Table 3 ties each map across visits; Q and K remain distinct.
            source = visits[0]
            for visit in visits[1:]:
                visit.to_q, visit.to_k = source.to_q, source.to_k
                for name in ("temperature_q", "temperature_k", "tempreature_q", "tempreature_k"):
                    if hasattr(source, name):
                        setattr(visit, name, getattr(source, name))
        self.visits = nn.ModuleList(visits)

    def synchronize_visit_initialization(self):
        source = self.visits[0]
        with torch.no_grad():
            for visit in self.visits[1:]:
                visit.to_q.weight.copy_(source.to_q.weight)
                visit.to_k.weight.copy_(source.to_k.weight)
                for name in ("temperature_q", "temperature_k", "tempreature_q", "tempreature_k"):
                    if hasattr(source, name):
                        getattr(visit, name).copy_(getattr(source, name))
            for visit in self.visits:
                visit.zero_router()

    def _features(self, x):
        if not isinstance(x, torch.Tensor) or x.ndim != 3:
            raise ValueError("V5 LinearNO operator requires [B,N,C]")
        B, N, C = x.shape
        if B < 1 or N < 1 or C != self.dim or not x.is_floating_point():
            raise ValueError("V5 LinearNO operator input shape/dtype mismatch")
        if self.variant in ("conv", "conv_temp"):
            if N != self.H * self.W:
                raise ValueError("V5 conv operator requires N=H*W")
            grid = x.transpose(1, 2).reshape(B, C, self.H, self.W)
            projected = self.in_project_x(grid)
            return projected.reshape(B, self.heads, self.dim_head, N).transpose(-1, -2)
        projected = self.in_project_x(x)
        features = projected.reshape(B, N, self.heads, self.dim_head).transpose(1, 2)
        return features.contiguous() if self.variant == "airfrans" else features

    def forward(self, x, *, visit_index):
        if type(visit_index) is not int or not 0 <= visit_index < len(self.visits):
            raise ValueError("visit_index is outside this physical operator")
        features = self._features(x)
        visit = self.visits[visit_index]
        q_logits = visit.to_q(features)
        k_logits = visit.to_k(features)
        values = self.to_v(features)
        if self.variant in ("temp", "conv_temp"):
            q_logits = q_logits / visit.temperature_q.clamp(.01, 1.)
            k_logits = k_logits / visit.temperature_k.clamp(.01, 1.)
        elif self.variant == "shapenet":
            q_logits = q_logits / visit.tempreature_q.clamp(.1, 2.)
            k_logits = k_logits / visit.tempreature_k.clamp(.1, 2.)
        queries = q_logits.softmax(dim=-1)
        keys = k_logits.softmax(dim=-2)
        context = torch.einsum("bhnm,bhnd->bhmd", keys, values)
        readout = torch.einsum("bhnm,bhmd->bhnd", queries, context)
        merged = readout.transpose(1, 2).reshape(x.shape[0], x.shape[1], self.dim)
        return self.to_out(merged)
