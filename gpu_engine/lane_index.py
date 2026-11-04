"""Exact select forms of per-lane point reads and writes on small axes.

``take(x, rows, i1, i2)`` equals ``x[rows, i1, i2]`` and
``put(x, rows, (i1, i2), y)`` equals ``x.at[rows, i1, i2].set(y)`` bit for bit
when ``rows`` is ``arange(B)`` over the leading lane axis, with JAX's default
indexing semantics: signed indices wrap once (i < 0 -> i + n); reads clamp
into range (XLA gather); writes drop rows whose wrapped index is out of range
(FILL_OR_DROP); the update is coerced, promoted with the operand, broadcast to
the slice shape and the result converted back exactly as ``_scatter_impl``.

Written as one-hot selects over the small indexed axes, these fuse into
elementwise kernels, where a gather/scatter pair is otherwise one kernel each
plus an operand copy. Indexed axes whose product exceeds ``LIMIT`` keep the
original indexing. ``rows`` must be the lane iota; callers pass it so its
static shape is checked here.
"""
import itertools
import math

import jax.numpy as j
from jax import lax
from jax._src import dtypes
from jax._src.lax import lax as lax_internal
from jax._src.numpy.util import promote_dtypes

LIMIT = 64


def _wrap(index, size, b):
    index = j.asarray(index)
    if not j.issubdtype(index.dtype, j.integer):raise TypeError('integer lane index required')
    if j.issubdtype(index.dtype, j.signedinteger):
        index = lax.select(index < 0, lax.add(index, lax_internal._const(index, size)), index)
    return j.broadcast_to(index, (b,))


def _check(x, rows, indices):
    if x.ndim < 1 + len(indices):raise ValueError('lane axis plus one axis per index required')
    if j.shape(rows) != (x.shape[0],):raise ValueError('rows must be the lane iota over the leading axis')
    return x.shape[0], x.shape[1:1 + len(indices)]


def _expand(mask, rank):
    return mask.reshape(mask.shape + (1,) * (rank - mask.ndim))


def take(x, rows, *indices):
    """x[rows, *indices] (indices: scalars or [B] integer arrays)."""
    x = j.asarray(x);b, sizes = _check(x, rows, indices)
    if math.prod(sizes) > LIMIT:return x[(rows,) + indices]
    wrapped = [j.clip(_wrap(i, n, b), 0, n - 1) for i, n in zip(indices, sizes)]
    lead = (slice(None),)
    positions = list(itertools.product(*(range(n) for n in sizes)))
    out = x[lead + positions[0]]
    for position in positions[1:]:
        hit = wrapped[0] == position[0]
        for index, p in zip(wrapped[1:], position[1:]):hit = hit & (index == p)
        out = j.where(_expand(hit, out.ndim), x[lead + position], out)
    return out


def put(x, rows, indices, y):
    """x.at[rows, *indices].set(y)."""
    x = j.asarray(x);indices = tuple(indices);b, sizes = _check(x, rows, indices)
    if math.prod(sizes) > LIMIT:return x.at[(rows,) + indices].set(y)
    if (isinstance(y, int) and not isinstance(y, bool) and j.issubdtype(x.dtype, j.integer)
            and j.iinfo(x.dtype).min <= y <= j.iinfo(x.dtype).max):
        y = j.asarray(y, dtype=x.dtype)
    else:y = j.asarray(y)
    dtype = lax.dtype(x);weak = dtypes.is_weakly_typed(x)
    xp, yp = promote_dtypes(x, y)
    yp = j.broadcast_to(yp, (b,) + x.shape[1 + len(indices):])
    wrapped = [_wrap(i, n, b) for i, n in zip(indices, sizes)]
    k = len(indices);hit = None
    for axis, (index, n) in enumerate(zip(wrapped, sizes)):
        shape = [1] * (k + 1);shape[axis + 1] = n
        onehot = index.reshape((b,) + (1,) * k) == j.arange(n, dtype=index.dtype).reshape(shape)
        hit = onehot if hit is None else hit & onehot
    yb = yp.reshape((b,) + (1,) * k + yp.shape[1:])
    out = j.where(_expand(hit, x.ndim), yb, xp)
    return lax_internal._convert_element_type(out, dtype, weak)
