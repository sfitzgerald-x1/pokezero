"""Bit-exact digests of encoder inputs, for the depth-2 exactness gate.

Both the depth-2 kernel and the independent two-step oracle hash each leaf's
encoder inputs with this one function, so leaf observations can be compared
for byte equality without moving every leaf to the host. A digest is two
uint32 words: a position-mixed sum and a position-mixed xor of every input
word, keyed by input name. Any differing bit changes both words unless two
independent 32-bit mixes collide at once.
"""
import jax
import jax.numpy as j
import numpy as np

INPUT_KEYS=('numeric_features','categorical_ids','token_type_ids','attention_mask','history_mask')
_GOLDEN=np.uint32(0x9E3779B9);_C1=np.uint32(0x85EBCA6B);_C2=np.uint32(0xC2B2AE35)


def _fmix(h):
    h=h^(h>>16);h=h*_C1;h=h^(h>>13);h=h*_C2;return h^(h>>16)


def words(x):
    """Every bit of one array as a flat uint32 vector (narrow dtypes are zero-extended)."""
    x=j.asarray(x)
    if x.dtype==j.bool_:return x.astype(j.uint32).reshape(-1)
    size=x.dtype.itemsize
    if size==4:return jax.lax.bitcast_convert_type(x,j.uint32).reshape(-1)
    if size==2:return jax.lax.bitcast_convert_type(x,j.uint16).astype(j.uint32).reshape(-1)
    if size==1:return jax.lax.bitcast_convert_type(x,j.uint8).astype(j.uint32).reshape(-1)
    if size==8:return jax.lax.bitcast_convert_type(x,j.uint32).reshape(-1)
    raise ValueError(f'unsupported dtype {x.dtype}')


def digest(inputs):
    """uint32[2] digest of ONE leaf's encoder inputs (a dict keyed by INPUT_KEYS names)."""
    if set(inputs)-set(INPUT_KEYS):raise ValueError('unknown encoder input key')
    total=j.uint32(0);folded=j.uint32(0)
    for index,key in enumerate(INPUT_KEYS):
        if key not in inputs:continue
        w=words(inputs[key]);position=j.arange(w.shape[0],dtype=j.uint32)
        salt=_fmix(j.uint32(index+1)*_GOLDEN)
        total=total+j.sum(_fmix(w^_fmix(position*_GOLDEN+salt)),dtype=j.uint32)
        folded=folded^jax.lax.reduce(_fmix((w+salt)*_C1+position),np.uint32(0),jax.lax.bitwise_xor,(0,))
        total=total+_fmix(j.uint32(w.shape[0])+salt)
    return j.stack((total,folded))


def batched(inputs,axes=1):
    """Digest every leaf of a batch: inputs carry `axes` leading leaf axes; returns [..., 2] uint32."""
    fn=digest
    for _ in range(axes):fn=jax.vmap(fn)
    return fn(inputs)
