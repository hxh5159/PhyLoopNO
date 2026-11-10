"""AirfRANS release primitive: absolute M=slice_num (default32), dead temperature.

Not a Data wrapper/full task model. Eq.8–9; HiPRL/LinearNO@3f2b80df AirfRANS.
"""
from cdlno.linearno.attention import LinearNOAttention


class LinearNO(LinearNOAttention):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0., slice_num=32):
        super().__init__(dim, heads=heads, dim_head=dim_head, rank=slice_num,
                         variant='airfrans', dropout=dropout)
