"""Python-rounded observation fractions without runtime device division.

The source encoder evaluates rational features in binary64 before storing
float32. XLA's reciprocal multiplication (including GPU dynamic division)
does not preserve those bits. These bounded integer domains are evaluated on
the host and embedded as constants in the compiled readout.
"""
from functools import lru_cache

import jax
import jax.numpy as j
import numpy as np


@lru_cache(None)
def _normalized_table(divisor, minimum):
    return np.asarray([i / divisor for i in range(minimum, divisor + 1)], np.float32)


def normalized(value, divisor, *, minimum=0):
    """Saturating integer feature, with a single Python-to-float32 rounding."""
    table = j.asarray(_normalized_table(divisor, minimum))
    index = j.clip(value, minimum, divisor).astype(j.int32) - minimum
    return table[index]


@lru_cache(None)
def _ratio_table(limit, pair):
    numerator = np.arange(limit + 1, dtype=np.float64)
    denominator = np.maximum(1, numerator[:, None])
    exact = numerator[None, :] / denominator
    high = exact.astype(np.float32)
    if not pair:
        return high
    return np.stack((high, (exact - high.astype(np.float64)).astype(np.float32)), axis=-1)


def ratio(numerator, denominator, *, limit=714, pair=False):
    """Bounded nonnegative integer ratio; denominator zero has legacy value 1.

    ``pair=True`` retains the binary64 quotient as a signed float32 residual
    for subsequent accumulation, instead of summing already-rounded features.
    Fractions saturate to [0, 1]. Generic public facts retain their full
    positive int32 denominator domain through integer long division, including
    the binary64 rounding performed by Python. No device float division runs.
    """
    table = j.asarray(_ratio_table(limit, pair))
    denominator = j.maximum(1, denominator).astype(j.int32)
    numerator = j.clip(numerator, 0, denominator).astype(j.int32)
    numerator, denominator = j.broadcast_arrays(numerator, denominator)
    def lookup(_):
        return table[j.clip(denominator, 1, limit), j.clip(numerator, 0, limit)]
    def generic(_):
        result = _integer_ratio_pair(numerator, denominator)
        return result if pair else result[..., 0]
    return jax.lax.cond(j.all(denominator <= limit), lookup, generic, operand=None)


def _integer_ratio_pair(numerator, denominator):
    """Python binary64 quotient, split into float32 high and signed low.

    Remainders are below a positive int32 denominator, so doubling them fits
    uint32. Two uint32 limbs hold the 53-bit significand; the 52 fractional
    bits and round-to-even use only exact integer arithmetic.
    """
    n = j.maximum(numerator, 1).astype(j.uint32)
    d = denominator.astype(j.uint32)
    shift = jax.lax.clz(n) - jax.lax.clz(d)
    scaled = n << shift
    extra = scaled < d
    shift = shift + extra.astype(j.uint32)
    scaled = j.where(extra, scaled << j.uint32(1), scaled)
    remainder = scaled - d
    def bit(_, carry):
        r, high, low = carry
        twice = r << j.uint32(1)
        one = twice >= d
        return (j.where(one, twice - d, twice),
                (high << j.uint32(1)) | (low >> j.uint32(31)),
                (low << j.uint32(1)) | one.astype(j.uint32))
    r, high, low = jax.lax.fori_loop(0, 52, bit,
        (remainder, j.zeros_like(n), j.ones_like(n)))
    twice = r << j.uint32(1)
    up = (twice > d) | ((twice == d) & ((low & j.uint32(1)) != 0))
    rounded_low = low + up.astype(j.uint32)
    high = high + (rounded_low < low).astype(j.uint32)
    low = rounded_low
    # Rounded significand may carry into bit 53 (the exact power-of-two case).
    carry = high == j.uint32(1 << 21)
    low = j.where(carry, (low >> j.uint32(1)) | (high << j.uint32(31)), low)
    high = j.where(carry, high >> j.uint32(1), high)
    exponent = -shift.astype(j.int32) + carry.astype(j.int32)
    mantissa = (high << j.uint32(3)) | (low >> j.uint32(29))
    residual = low & j.uint32((1 << 29) - 1)
    up32 = (residual > j.uint32(1 << 28)) | (
        (residual == j.uint32(1 << 28)) & ((mantissa & j.uint32(1)) != 0))
    first = j.ldexp((mantissa + up32.astype(j.uint32)).astype(j.float32), exponent - 23)
    magnitude = j.where(up32, j.uint32(1 << 29) - residual, residual).astype(j.float32)
    second = j.ldexp(j.where(up32, -magnitude, magnitude), exponent - 52)
    result = j.stack((first, second), axis=-1)
    return j.where((numerator > 0)[..., None], result, 0)


def _barrier(value):
    return jax.lax.optimization_barrier(value)


def _two_sum(a, b):
    total = _barrier(a + b)
    recovered = _barrier(total - a)
    left = _barrier(a - _barrier(total - recovered))
    right = _barrier(b - recovered)
    return total, _barrier(left + right)


def add_pair(accumulator, increment):
    """Signed compensated addition, including negative quotient residuals."""
    total, error = _two_sum(accumulator[..., 0], increment[..., 0])
    correction = _barrier(_barrier(accumulator[..., 1] + increment[..., 1]) + error)
    high, low = _two_sum(total, correction)
    return j.stack((high, low), axis=-1)


def value(pair):
    return _barrier(pair[..., 0] + pair[..., 1])


def _two_product(a, b):
    # Dekker split, with barriers preventing XLA from cancelling the residual.
    product = _barrier(a * b)
    def split(x):
        c = _barrier(j.float32(4097) * x)
        high = _barrier(c - _barrier(c - x))
        return high, _barrier(x - high)
    ah, al = split(a)
    bh, bl = split(b)
    error = _barrier(_barrier(ah * bh) - product)
    error = _barrier(error + _barrier(ah * bl))
    error = _barrier(error + _barrier(al * bh))
    error = _barrier(error + _barrier(al * bl))
    return product, error


def scaled_value(pair, divisor):
    """Round a compensated sum only after the source's final normalization."""
    reciprocal = 1.0 / divisor
    rh = np.float32(reciprocal)
    rl = np.float32(reciprocal - float(rh))
    high, error = _two_product(pair[..., 0], j.asarray(rh))
    error = _barrier(error + _barrier(pair[..., 0] * j.asarray(rl)))
    error = _barrier(error + _barrier(pair[..., 1] * j.asarray(rh)))
    error = _barrier(error + _barrier(pair[..., 1] * j.asarray(rl)))
    return _barrier(high + error)
