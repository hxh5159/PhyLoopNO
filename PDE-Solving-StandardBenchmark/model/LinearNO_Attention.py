"""Independent Standard LinearNO primitives; not registered as a full Model.

Official constructor key_ratio is the absolute per-head Q/K feature count M.
Source: HiPRL/LinearNO@3f2b80df, Standard_PDE_Benchmark, Eq. 8–9.
"""
from cdlno.linearno.attention import LinearNOAttention


class LinearNO(LinearNOAttention):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0., key_ratio=4):
        super().__init__(dim, heads=heads, dim_head=dim_head, rank=key_ratio,
                         variant='plain', dropout=dropout)


class LinearNO_temp(LinearNOAttention):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0., key_ratio=4):
        super().__init__(dim, heads=heads, dim_head=dim_head, rank=key_ratio,
                         variant='temp', dropout=dropout)


class LinearNO_Conv(LinearNOAttention):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0., key_ratio=64, H=101, W=31, kernel=3):
        super().__init__(dim, heads=heads, dim_head=dim_head, rank=key_ratio,
                         variant='conv', dropout=dropout, H=H, W=W, kernel=kernel)


class LinearNO_Conv_temp(LinearNOAttention):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0., key_ratio=64, H=101, W=31, kernel=3):
        super().__init__(dim, heads=heads, dim_head=dim_head, rank=key_ratio,
                         variant='conv_temp', dropout=dropout, H=H, W=W, kernel=kernel)
