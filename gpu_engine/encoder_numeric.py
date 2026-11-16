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


def normalized(value, divisor, *, minimum=None):
    """Upper-saturating feature; integer LUT with a precise float transport path.

    Private request stats also admit fractional float32 values. Preserve those
    values, and the source's lack of a lower clamp unless one is specified.
    """
    if int(divisor) != divisor or divisor <= 0:
        raise ValueError('positive integer normalization divisor required')
    divisor = int(divisor)
    lower = 0 if minimum is None else minimum
    clipped = j.minimum(value, divisor)
    if minimum is not None:clipped = j.maximum(clipped, minimum)
    table = j.asarray(_normalized_table(divisor, lower))
    index = j.clip(clipped, lower, divisor).astype(j.int32) - lower
    integral = (clipped >= lower) & (clipped == j.floor(clipped))
    if j.issubdtype(j.asarray(clipped).dtype, j.floating):
        integral &= ~((clipped == 0) & j.signbit(clipped))
    return jax.lax.cond(j.all(integral), lambda _: table[index],
        lambda _: _float_quotient_pair(clipped, j.float32(divisor))[..., 0], None)


def hazard_exposure(count, layers):
    """Full Python expression, rounded once after count * fraction / 6."""
    table = np.asarray([[min(1.0, n * f / 6) for f in (0.0, 1/8, 1/6, 1/4)]
                        for n in range(7)], np.float32)
    return j.asarray(table)[j.clip(count, 0, 6).astype(j.int32),
                            j.clip(layers, 0, 3).astype(j.int32)]


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
    numerator, denominator = j.broadcast_arrays(j.asarray(numerator), j.asarray(denominator))
    if j.issubdtype(denominator.dtype, j.floating):
        bits = jax.lax.bitcast_convert_type(denominator, j.uint32)
        zero = (bits & j.uint32(0x7fffffff)) == 0
    else:
        zero = denominator == 0
    denominator = j.where(zero, 1, denominator)
    def integral_int32(x):
        if j.issubdtype(x.dtype, j.integer):return j.ones_like(x, bool)
        bits = jax.lax.bitcast_convert_type(x, j.uint32) & j.uint32(0x7fffffff)
        subnormal = (bits > 0) & (bits < j.uint32(1 << 23))
        return ~subnormal & (x < float(2**31)) & (x == j.floor(x))
    integer_domain = ((numerator >= 0) & (denominator > 0) &
        integral_int32(numerator) & integral_int32(denominator))
    n = j.clip(numerator, 0, denominator).astype(j.int32)
    d = denominator.astype(j.int32)
    def lookup(_):
        return table[j.clip(d, 1, limit), j.clip(n, 0, limit)]
    def generic(_):
        def floating(_):
            raw = _float_quotient_pair(numerator, denominator)
            high = jax.lax.bitcast_convert_type(raw[..., 0], j.uint32)
            negative = (high >> j.uint32(31)) != 0
            above_one = (high & j.uint32(0x7fffffff)) > j.uint32(0x3f800000)
            one = j.stack((j.ones_like(raw[..., 0]), j.zeros_like(raw[..., 0])), -1)
            return j.where(negative[..., None], 0, j.where(above_one[..., None], one, raw))
        result = jax.lax.cond(j.all(integer_domain),
            lambda _: _integer_ratio_pair(n, d), floating, None)
        return result if pair else result[..., 0]
    return jax.lax.cond(j.all(integer_domain & (denominator <= limit)), lookup, generic, operand=None)


def _float_quotient_pair(numerator, denominator):
    """Divide finite float32 transports through their integer significands."""
    def decode(x):
        bits = jax.lax.bitcast_convert_type(j.asarray(x, j.float32), j.uint32)
        encoded = (bits >> j.uint32(23)) & j.uint32(255)
        mantissa = (bits & j.uint32((1 << 23) - 1)) | j.where(encoded != 0, j.uint32(1 << 23), j.uint32(0))
        exponent = j.where(encoded != 0, encoded.astype(j.int32) - 150, -149)
        return mantissa, exponent, (bits >> j.uint32(31)) != 0
    n, ne, ns = decode(numerator)
    d, de, ds = decode(denominator)
    raw = _integer_ratio_pair(n.astype(j.int32), j.maximum(d, 1).astype(j.int32),
                              exponent_offset=ne - de)
    bits = jax.lax.bitcast_convert_type(raw, j.uint32)
    bits = bits ^ j.where(ns != ds, j.uint32(1 << 31), j.uint32(0))[..., None]
    low = bits[..., 1]
    bits = bits.at[..., 1].set(j.where((low & j.uint32(0x7fffffff)) == 0, j.uint32(0), low))
    return jax.lax.bitcast_convert_type(bits, j.float32)


def _round_shift53(high, low, shift):
    """Round an unsigned two-word significand right by 29 or more bits."""
    amount = j.clip(shift, 1, 63).astype(j.uint32)
    small = amount <= j.uint32(32)
    quotient = j.where(small,
        (high << (j.uint32(32) - amount)) | (low >> amount),
        high >> (amount - j.uint32(32)))
    guard_position = j.where(small, amount - j.uint32(1), amount - j.uint32(33))
    word = j.where(small, low, high)
    guard = ((word >> guard_position) & j.uint32(1)) != 0
    mask = (j.uint32(1) << guard_position) - j.uint32(1)
    sticky = ((word & mask) != 0) | (~small & (low != 0))
    up = guard & (sticky | ((quotient & j.uint32(1)) != 0))
    return j.where(shift > 63, j.uint32(0), quotient + up.astype(j.uint32))


def _float32_bits(high, low, exponent):
    """Round at the final exponent, including subnormals, without float ops."""
    mantissa = _round_shift53(high, low, j.full_like(exponent, 29))
    carry = mantissa == j.uint32(1 << 24)
    biased = exponent + 127 + carry.astype(j.int32)
    mantissa = j.where(carry, mantissa >> j.uint32(1), mantissa)
    normal = (biased.astype(j.uint32) << j.uint32(23)) | (mantissa & j.uint32(0x7fffff))
    normal = j.where(biased >= 255, j.uint32(0x7f800000), normal)
    subnormal = _round_shift53(high, low, 29 + (-126 - exponent))
    return j.where(exponent < -126, subnormal, normal)


def _scaled_integer_bits(magnitude, exponent):
    """Encode magnitude * 2**exponent with one final float32 rounding."""
    leading = 31 - jax.lax.clz(magnitude).astype(j.int32)
    shift = (52 - leading).astype(j.uint32)
    high = j.where(shift >= 32, magnitude << (shift - j.uint32(32)),
                   magnitude >> (j.uint32(32) - shift))
    low = magnitude << shift
    bits = _float32_bits(high, low, exponent + leading)
    return j.where(magnitude != 0, bits, j.uint32(0))


def _integer_ratio_pair(numerator, denominator, *, exponent_offset=0):
    """Python binary64 quotient, split into float32 high and signed low.

    Remainders are below a positive int32 denominator, so doubling them fits
    uint32. Two uint32 limbs hold the 53-bit significand; the 52 fractional
    bits and round-to-even use only exact integer arithmetic.
    """
    n = j.maximum(numerator, 1).astype(j.uint32)
    d = denominator.astype(j.uint32)
    shift = jax.lax.clz(n).astype(j.int32) - jax.lax.clz(d).astype(j.int32)
    scaled = n << j.maximum(shift, 0).astype(j.uint32)
    d = d << j.maximum(-shift, 0).astype(j.uint32)
    extra = scaled < d
    shift = shift + extra.astype(j.int32)
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
    exponent = -shift.astype(j.int32) + carry.astype(j.int32) + exponent_offset
    mantissa = (high << j.uint32(3)) | (low >> j.uint32(29))
    residual = low & j.uint32((1 << 29) - 1)
    up32 = (residual > j.uint32(1 << 28)) | (
        (residual == j.uint32(1 << 28)) & ((mantissa & j.uint32(1)) != 0))
    first = _float32_bits(high, low, exponent)
    magnitude = j.where(up32, j.uint32(1 << 29) - residual, residual)
    second = _scaled_integer_bits(magnitude, exponent - 52)
    second = second | j.where(up32 & (second != 0), j.uint32(1 << 31), j.uint32(0))
    # A subnormal first component's residual is at most half a subnormal ulp;
    # it rounds to zero. Construct bits directly to avoid backend FTZ behavior.
    second = j.where(exponent < -126, j.uint32(0), second)
    result = jax.lax.bitcast_convert_type(j.stack((first, second), axis=-1), j.float32)
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
