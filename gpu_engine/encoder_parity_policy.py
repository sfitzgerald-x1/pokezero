"""Champion input parity: exact scalars, at most one ULP for four sums.

The cumulative sums retain the existing float32 pair representation. Python
accumulates in binary64; a reachable Spikes history differs by one float32 ULP.
The lead approved this exception on 2026-11-16 pending V2's common exact-sum
convention. This is not an absolute-error tolerance for other columns.
"""
from functools import lru_cache
import numpy as np


CUMULATIVE_NAMES = ('LAST_DAMAGE_DEALT', 'LAST_DAMAGE_TAKEN',
                    'SELF_HAZARD_CREDIT', 'OPP_HAZARD_CREDIT')


@lru_cache(maxsize=1)
def cumulative_columns():
    from pokezero import showdown as reference
    return {reference.v4_numeric_index(getattr(reference, 'NUMERIC_' + name)): name
            for name in CUMULATIVE_NAMES}


def float32_ulp_distance(left, right):
    """Ordered binary32 bit distance; invalid/non-lossless inputs exceed 1 ULP."""
    left, right = np.asarray(left), np.asarray(right)
    with np.errstate(over='ignore', invalid='ignore'):
        a, b = left.astype(np.float32), right.astype(np.float32)
        valid = (np.isfinite(left) & np.isfinite(right) &
                 (left == a.astype(np.float64)) & (right == b.astype(np.float64)))
    def ordered(value):
        bits = value.view(np.uint32)
        return np.where(bits >> 31, ~bits, bits ^ np.uint32(0x80000000)).astype(np.int64)
    return np.where(valid, np.abs(ordered(a) - ordered(b)), 2**32)


def compare_numeric(left, right):
    """Return failing cells and ULP distances for complete public29 numeric rows."""
    left, right = np.asarray(left), np.asarray(right)
    if left.shape != right.shape or left.shape[-1] != 132:
        raise ValueError('matching numeric rows with 132 columns required')
    ulps = float32_ulp_distance(left, right)
    signs_differ = np.signbit(left) != np.signbit(right)
    bad = ~np.isfinite(left) | ~np.isfinite(right) | (left != right) | signs_differ
    columns = list(cumulative_columns())
    bad[..., columns] = (ulps[..., columns] > 1) | signs_differ[..., columns]
    return bad, ulps
