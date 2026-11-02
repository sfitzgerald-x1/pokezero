"""Explicit candidate for one independent member write per batch row.

The ordinary indexed scatter stays the default elsewhere. This alternative
uses a dense seat/encounter selector so the compiler can fuse current public
counter/state writes. It caches no values and changes no feature semantics.
"""
import jax.numpy as j


def set_member(array,side,slot,value):
    """Equivalent to array.at[arange(B),side,slot].set(value).

    Original batch/seat/member axes and exact dtype are retained. Negative
    seat/member indices wrap once, as the original scatter does; out-of-range
    destinations select nothing. Callers keep their existing current masks.
    """
    b=array.shape[0]
    if array.ndim<3 or array.shape[1:3]!=(2,6) or value.shape!=(b,)+array.shape[3:] or value.dtype!=array.dtype:
        raise ValueError('same typed batch/seat/member update required')
    side=j.broadcast_to(j.asarray(side),(b,));slot=j.broadcast_to(j.asarray(slot),(b,))
    side=j.where(side<0,side+2,side);slot=j.where(slot<0,slot+6,slot)
    selected=(j.arange(2)[None,:,None]==side[:,None,None])&(j.arange(6)[None,None,:]==slot[:,None,None])
    selected=selected.reshape(selected.shape+(1,)*(array.ndim-3))
    return j.where(selected,value[:,None,None],array)
