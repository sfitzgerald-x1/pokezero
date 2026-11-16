"""Bitwise Python reference checks for observation arithmetic on any backend."""
import unittest

import jax
import jax.numpy as j
import numpy as np

from gpu_engine import encoder_numeric as en


def same_bits(actual, expected):
    np.testing.assert_array_equal(np.asarray(actual, np.float32).view(np.uint32),
                                  np.asarray(expected, np.float32).view(np.uint32))


class EncoderNumericTests(unittest.TestCase):
    def test_every_saturating_integer_normalization(self):
        for divisor in (1, 2, 3, 4, 5, 6, 8, 15, 64, 100, 200, 714, 1000):
            for minimum in (0, -divisor):
                with self.subTest(divisor=divisor, minimum=minimum):
                    values = np.arange(minimum - 2, divisor + 3, dtype=np.int32)
                    expected = np.asarray([min(1, max(minimum / divisor, int(v) / divisor))
                                           for v in values], np.float32)
                    got = jax.jit(lambda x: en.normalized(x, divisor, minimum=minimum))(values)
                    same_bits(got, expected)

    def test_all_gen3_hp_fractions_and_residuals(self):
        d, n = np.meshgrid(np.arange(1, 715, dtype=np.int32),
                           np.arange(715, dtype=np.int32), indexing='ij')
        n = np.minimum(n, d)
        exact = n.astype(np.float64) / d
        high = exact.astype(np.float32)
        low = (exact - high.astype(np.float64)).astype(np.float32)
        same_bits(jax.jit(en.ratio)(n, d), high)
        pair = jax.jit(lambda a, b: en.ratio(a, b, pair=True))(n, d)
        same_bits(pair[..., 0], high)
        same_bits(pair[..., 1], low)

    def test_full_int32_public_denominators_and_binary64_rounding(self):
        rng = np.random.default_rng(614)
        d = np.concatenate((np.asarray([1, 100, 714, 715, 2048, 2**24 - 1,
             2**24 + 1, 2**30 + 1, 2**31 - 1], np.int32),
             rng.integers(1, 2**31, size=4096, dtype=np.int32)))
        n = rng.integers(0, 2**31, size=len(d), dtype=np.int32) % d
        for numerator in (n, np.ones_like(d), d, d - 1, d // 2, np.zeros_like(d)):
            exact = numerator.astype(np.float64) / d
            high = exact.astype(np.float32)
            low = (exact - high.astype(np.float64)).astype(np.float32)
            pair = jax.jit(lambda a, b: en.ratio(a, b, pair=True))(numerator, d)
            same_bits(pair[..., 0], high)
            same_bits(pair[..., 1], low)

    def test_signed_low_terms_are_retained_until_final_rounding(self):
        # Each sequence exposed an early-rounding or reciprocal-multiply error.
        for numerators, denominator in (([1, 4], 100), ([12, 11, 12, 100], 100),
                ([1] * 997, 100), ([1] * 10000, 2**31 - 1),
                ([1, 17, 43, 111, 2, 6], 307)):
            def accumulate(n):
                parts = en.ratio(n, j.full_like(n, denominator), pair=True)
                total, _ = jax.lax.scan(lambda a, b: (en.add_pair(a, b), None),
                                       j.zeros(2, j.float32), parts)
                return en.value(total), en.scaled_value(total, 6)
            total = sum(int(n) / denominator for n in numerators)
            got, scaled = jax.jit(accumulate)(j.asarray(numerators, j.int32))
            same_bits(got, total)
            same_bits(scaled, total / 6)

    def test_fractional_signed_and_large_float_transports(self):
        numerator = np.asarray([1.5, -1., .5, 3.25, 1e30, 1e30, 100.5], np.float32)
        denominator = np.asarray([2., -2., 1., 7.5, 2e30, 5e30, 714.], np.float32)
        exact = numerator.astype(np.float64) / denominator.astype(np.float64)
        got = jax.jit(en.ratio)(numerator, denominator)
        same_bits(got, np.clip(exact, 0, 1))
        values = np.asarray([100.5, -.5, -1e30, .25, 1e30], np.float32)
        same_bits(jax.jit(lambda x: en.normalized(x, 714))(values),
                  np.minimum(1, values.astype(np.float64) / 714))

    def test_subnormal_outputs_are_rounded_at_the_final_exponent(self):
        tiny = np.float32(np.finfo(np.float32).tiny)
        n = np.asarray([tiny, tiny, tiny, tiny, tiny, tiny], np.float32)
        d = np.asarray([1.5, 3., 6., 714., 2**23, 2**24], np.float32)
        expected = (n.astype(np.float64) / d.astype(np.float64)).astype(np.float32)
        same_bits(jax.jit(en.ratio)(n, d), expected)
        # Include values on both sides of the normal/subnormal boundary and
        # random normal transports whose quotients require subnormal rounding.
        rng = np.random.default_rng(176)
        n = rng.integers(1 << 23, 1 << 25, size=4096, dtype=np.uint32).view(np.float32)
        d = rng.uniform(1, 1000, size=len(n)).astype(np.float32)
        expected = (n.astype(np.float64) / d.astype(np.float64)).astype(np.float32)
        same_bits(jax.jit(en.ratio)(n, d), expected)

    def test_generic_midpoint_remains_within_tolerance_outside_gen3(self):
        denominator = 33554431
        self.assertGreater(denominator, 714)
        def sum_terms():
            a = en.ratio(j.int32(1), j.int32(2), pair=True)
            b = en.ratio(j.int32(1), j.int32(denominator), pair=True)
            return en.value(en.add_pair(a, b))
        got = np.asarray(jax.jit(sum_terms)())
        expected = np.float32(1/2 + 1/denominator)
        self.assertEqual(int(got.view(np.uint32)), 1056964608)
        self.assertEqual(int(expected.view(np.uint32)), 1056964609)
        self.assertLess(abs(float(got) - float(expected)), 1e-6)

    def test_gen3_spikes_sequence_has_exactly_one_ulp_champion_exception(self):
        # Blissey76, Meganium85, Sableye91, Swampert78, Lanturn82,
        # Steelix82: HP IV31/EV85, all grounded with Leftovers. Start Steelix
        # before one Spikes layer; recover to full between these switch-ins.
        hp = (513, 275, 239, 284, 339, 257)
        order = (1,4,1,4,1,4,1,3,4,0,1,2,3,4,0,1,2,3,4,5,
                 0,1,2,3,4,5,0,1,2,3,4,5,0,1,2,3,4,5)
        d = np.asarray([hp[i] for i in order], np.int32)
        self.assertTrue(np.all((d >= 1) & (d <= 714)))
        self.assertTrue(all(a != b for a, b in zip(order, order[1:])))
        n = d // 8
        def accumulate(n, d):
            terms = en.ratio(n, d, pair=True)
            total, _ = jax.lax.scan(lambda a, b: (en.add_pair(a, b), None),
                                   j.zeros(2, j.float32), terms)
            return en.scaled_value(total, 6)
        expected = 0.0
        for numerator, denominator in zip(n, d):
            expected += int(numerator) / int(denominator)
        actual = np.asarray(jax.jit(accumulate)(n, d), np.float32)
        expected = np.asarray(expected / 6, np.float32)
        self.assertEqual(int(actual.view(np.uint32)), 1061706238)
        self.assertEqual(int(expected.view(np.uint32)), 1061706237)
        self.assertEqual(abs(int(actual.view(np.uint32)) - int(expected.view(np.uint32))), 1)

    def test_hazard_exposure_rounds_the_entire_python_expression(self):
        count, layers = np.meshgrid(np.arange(7, dtype=np.int32),
                                    np.arange(4, dtype=np.int32), indexing='ij')
        expected = np.asarray([[n * f / 6 for f in (0., 1/8, 1/6, 1/4)]
                               for n in range(7)], np.float32)
        same_bits(jax.jit(en.hazard_exposure)(count, layers), expected)

    def test_unchanged_small_domain_divisions_are_bitwise_safe(self):
        # public_sleep is owned separately: its /5 saturates at1, and /64 is
        # exact binary scaling. The same proof covers /3 hazard layer counts.
        for divisor in (2, 3, 4, 5, 8, 64):
            values = j.arange(divisor + 1, dtype=j.int32)
            got = jax.jit(lambda x: j.minimum(1, x / divisor))(values)
            same_bits(got, [i / divisor for i in range(divisor + 1)])


if __name__ == '__main__':
    unittest.main()
