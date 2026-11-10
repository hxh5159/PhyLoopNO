"""ShapeNet release primitive: M=key_ratio*dim_head, original tempreature keys.

Not a tuple/Data wrapper/full task model. Eq.8–9; HiPRL/LinearNO@3f2b80df ShapeNetCar.
"""
from cdlno.linearno.attention import LinearNOAttention, _positive_integer


class LinearNO(LinearNOAttention):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0., key_ratio=4):
        _positive_integer(key_ratio, 'key_ratio')
        _positive_integer(dim_head, 'dim_head')
        super().__init__(dim, heads=heads, dim_head=dim_head, rank=key_ratio*dim_head,
                         variant='shapenet', dropout=dropout)
