"""lane_index take/put are bit-identical to JAX default point indexing."""
import itertools
import unittest
import warnings

import jax
import jax.numpy as j
import numpy as np

from gpu_engine import lane_index


def bits(x):
    x = np.asarray(x)
    return x.view(np.uint8) if x.dtype != bool else x


class LaneIndexTest(unittest.TestCase):
    B = 7

    def operand(self, rng, dtype, trailing):
        shape = (self.B, 2, 6) + trailing
        if dtype == j.bool_:return j.asarray(rng.random(shape) < .5)
        if dtype == j.float32:
            x = rng.standard_normal(shape).astype(np.float32) * 1e3
            flat = x.reshape(-1);flat[::5] = np.nan;flat[1::7] = -0.;flat[2::11] = np.inf;flat[3::13] = 3e7
            return j.asarray(x)
        info = np.iinfo(np.dtype(dtype))
        return j.asarray(rng.integers(max(info.min, -2**31), min(info.max, 2**31 - 1), shape, dtype=np.int64).astype(np.dtype(dtype)))

    def indices(self, rng):
        sides = (0, 1, -1, j.asarray(1), j.asarray(rng.integers(-3, 4, self.B), j.int32),
            j.asarray(rng.integers(0, 2, self.B), j.int32))
        slots = (j.asarray(rng.integers(-8, 9, self.B), j.int32), j.asarray(rng.integers(0, 6, self.B), j.int32),
            j.asarray(rng.integers(0, 9, self.B), j.uint32), 3)
        return sides, slots

    def updates(self, rng, x, trailing):
        shape = (self.B,) + trailing
        yield j.asarray(rng.integers(-2**30, 2**30, shape), j.int32)
        yield j.asarray((rng.standard_normal(shape) * 1e8).astype(np.float32))
        yield j.asarray(rng.random(shape) < .5)
        yield 5
        yield -1
        yield 2.5
        yield True
        if trailing:yield j.asarray(rng.integers(0, 9, trailing[-1:]), j.int32)

    def test_take_and_put_match_jax_indexing(self):
        rng = np.random.default_rng(0);r = j.arange(self.B);checked = 0
        for dtype, trailing in itertools.product((j.bool_, j.int32, j.uint32, j.float32), ((), (4,), (2, 3))):
            x = self.operand(rng, dtype, trailing);sides, slots = self.indices(rng)
            for side, slot in itertools.product(sides, slots):
                with self.subTest(dtype=str(dtype), trailing=trailing, side=str(side), slot=str(slot)):
                    np.testing.assert_array_equal(bits(lane_index.take(x, r, side, slot)), bits(x[r, side, slot]))
                    np.testing.assert_array_equal(bits(lane_index.take(x, r, side)), bits(x[r, side]))
                    for y in self.updates(rng, x, trailing):
                        with warnings.catch_warnings():
                            warnings.simplefilter('ignore', FutureWarning)
                            want = x.at[r, side, slot].set(y);got = lane_index.put(x, r, (side, slot), y)
                            want1 = x.at[r, side].set(y if np.ndim(y) == 0 else j.broadcast_to(j.asarray(y)[:, None] if np.ndim(y) == 1 + len(trailing) else y, (self.B, 6) + trailing))
                            got1 = lane_index.put(x, r, (side,), y if np.ndim(y) == 0 else j.broadcast_to(j.asarray(y)[:, None] if np.ndim(y) == 1 + len(trailing) else y, (self.B, 6) + trailing))
                        self.assertEqual(got.dtype, want.dtype);self.assertEqual(got.shape, want.shape)
                        self.assertEqual(jax.typeof(got).weak_type, jax.typeof(want).weak_type)
                        np.testing.assert_array_equal(bits(got), bits(want))
                        np.testing.assert_array_equal(bits(got1), bits(want1))
                        checked += 1
        self.assertGreater(checked, 1000)

    def test_large_axes_fall_back_and_rows_checked(self):
        rng = np.random.default_rng(2);r = j.arange(self.B)
        x = j.asarray(rng.integers(0, 99, (self.B, 9, 8)), j.int32);a = j.asarray(rng.integers(-9, 9, self.B), j.int32)
        np.testing.assert_array_equal(lane_index.take(x, r, a, a), x[r, a, a])
        np.testing.assert_array_equal(lane_index.put(x, r, (a, a), 7), x.at[r, a, a].set(7))
        with self.assertRaises(ValueError):lane_index.take(x, j.arange(self.B - 1), a[:-1])

    def test_update_matches_read_modify_write(self):
        rng = np.random.default_rng(3);r = j.arange(self.B);checked = 0
        def gate(T):return j.asarray(rng.random((self.B,) + (1,) * len(T)) < .5)
        def ints(T, lo, hi):return j.asarray(rng.integers(lo, hi, (self.B,) + T), j.int32)
        cases = (
            (lambda v, use, value: j.where(use, v + value, v), lambda T: (gate(T), ints(T, -5, 5))),
            (lambda v, use, value: j.minimum(64, j.where(use, v + value, v)), lambda T: (gate(T), ints(T, 0, 90))),
            (lambda v, use, value: j.where(use, value, v), lambda T: (gate(T), j.asarray((rng.standard_normal((self.B,) + T) * 1e8).astype(np.float32)))),
            (lambda v, use, value: j.where(use, value, v), lambda T: (gate(T), ints(T, -2**30, 2**30))),
            (lambda v, use, value: j.where(use, v & value, v), lambda T: (gate(T), ints(T, 0, 2**31 - 1))),
            (lambda v, use, value: j.where(use, j.any(v != 0, axis=-1, keepdims=True) | value, v), lambda T: (gate(T), j.asarray(rng.random((self.B,) + T) < .5))),
            (lambda v, use, value: j.where(use, v + value, v), lambda T: (gate(T), 1)),
            (lambda v, use, value: j.where(use, v + value, v), lambda T: (gate(T), 2.5)),
            (lambda v, use, value: j.where(use, value, v), lambda T: (True, j.asarray(7))),
        )
        for dtype, trailing in itertools.product((j.bool_, j.int32, j.uint32, j.float32), ((), (4,))):
            x = self.operand(rng, dtype, trailing);sides, slots = self.indices(rng)
            for (fn, make), side, slot in itertools.product(cases, sides, slots):
                lanes = make(trailing)
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore', FutureWarning)
                    try:want = x.at[r, side, slot].set(fn(x[r, side, slot], *lanes))
                    except TypeError:continue
                    got = lane_index.update(x, r, (side, slot), fn, *lanes)
                self.assertEqual(got.dtype, want.dtype)
                np.testing.assert_array_equal(bits(got), bits(want));checked += 1
        self.assertGreater(checked, 500)

    def test_under_jit(self):
        rng = np.random.default_rng(1);r = j.arange(self.B)
        x = self.operand(rng, j.float32, (4,));side = j.asarray(rng.integers(-2, 3, self.B), j.int32)
        slot = j.asarray(rng.integers(-7, 8, self.B), j.int32);y = j.asarray(rng.standard_normal((self.B, 4)), j.float32)
        f = jax.jit(lambda x, s, a, y: (lane_index.take(x, r, s, a), lane_index.put(x, r, (s, a), y)))
        g = jax.jit(lambda x, s, a, y: (x[r, s, a], x.at[r, s, a].set(y)))
        for got, want in zip(f(x, side, slot, y), g(x, side, slot, y)):
            np.testing.assert_array_equal(bits(got), bits(want))


if __name__ == '__main__':unittest.main()
