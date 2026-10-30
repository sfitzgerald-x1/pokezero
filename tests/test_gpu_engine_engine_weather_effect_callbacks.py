"""Literal source-helper controls; no caller edits, games/models or formatting."""
import unittest
from unittest.mock import patch

import jax
import jax.numpy as j
import numpy as np

from gpu_engine import engine_weather_effect_callbacks as callbacks, weather, mechanics
from gpu_engine import engine_public_events as public
import test_gpu_engine_engine_item_callbacks as fixtures
import test_gpu_engine_weather as weather_fixtures


class WeatherEffectTests(unittest.TestCase):
    def tape(self, batch=1, capacity=16, legacy=False):
        state = fixtures.state(batch)._replace(active=j.broadcast_to(j.array([2, 4], j.int32), (batch, 2)))
        return fixtures.tape(state, capacity, legacy=legacy)

    def same(self, a, b):
        for actual, expected in zip(jax.tree.leaves(a), jax.tree.leaves(b)):
            np.testing.assert_array_equal(actual, expected)

    def words(self, tape, row=0):
        return np.asarray(tape.words[row, :int(tape.count[row])])

    def failure(self, tape, *, move=91, eligible=True):
        return callbacks.same_weather_failure(tape, 0, 2, move,
            weather.SAND, weather.SAND, weather.SAND, eligible=eligible)

    def chip(self, tape, *, old=101, new=95, maximum=101, status=0, reason=weather.SAND, eligible=True):
        return callbacks.weather_chip(tape, 0, 2, old, new, maximum, status, reason, eligible=eligible)

    def test_same_weather_failure_binds_called_move_and_still_constraint(self):
        old = j.arange(1, 5, dtype=j.int32)
        moves = j.array([16777217, 92, 93, 94], j.int32)
        trace = callbacks.same_weather_failure(self.tape(4), j.array([0, 1, 0, 1], j.int32),
            j.array([2, 4, 2, 4], j.int32), moves, old, old, old, eligible=j.ones(4, bool))
        for row in range(4):
            self.assertEqual(self.words(trace, row).tolist(),
                [[15, row % 2, [2, 4][row % 2], -1, -1, row + 1, row + 1, int(moves[row]), row + 1, 1]])
        self.assertEqual(trace.refusal.tolist(), [256] * 4)

    def test_actual_same_weather_set_attempt_has_no_numeric_change(self):
        for clock in (0, 1, 5):
            old = j.array([weather.RAIN], j.int32); before = j.array([clock], j.int32)
            current, after = weather.set_move(old, before, old)
            self.same((current, after), (old, before))
            trace = callbacks.same_weather_failure(self.tape(), 0, 2, 91, old, current, old,
                eligible=j.ones(1, bool))
            self.assertEqual(trace.count.tolist(), [1])
            self.assertEqual(trace.words[0, 0, 9].item(), callbacks.STILL_REQUIRED)

    def test_failure_invalid_weather_identity_and_ineligible_private_slots(self):
        trace = callbacks.same_weather_failure(self.tape(6), j.array([0, 0, 0, 2, 0, 0], j.int32),
            j.array([2, 2, 1, 2, 2, 2], j.int32), j.array([91, 92, 93, 94, 0, 999999], j.int32),
            j.array([0, 1, 1, 1, 1, 1], j.int32), j.array([0, 2, 1, 1, 1, 1], j.int32),
            j.array([0, 1, 1, 1, 1, 1], j.int32), eligible=j.array([True] * 5 + [False]))
        self.assertEqual(trace.count.tolist(), [0] * 6)
        self.assertEqual(trace.refusal.tolist(), [256] * 5 + [0])
        self.assertFalse(bool(trace.words.any()))
        for move in (None, j.array([1.25], j.float32), np.array([2**32 + 1], np.int64)):
            unknown = self.failure(self.tape(), move=move)
            self.assertEqual(unknown.count.tolist(), [0]); self.assertEqual(unknown.refusal.tolist(), [256])

    def test_literal_chip_payload_retains_internal_status_and_never_synthesizes_faint(self):
        trace = callbacks.weather_chip(self.tape(2), j.array([0, 1], j.int32), j.array([2, 4], j.int32),
            j.array([101, 3], j.int32), j.array([95, 0], j.int32), j.array([101, 31], j.int32),
            j.array([mechanics.BRN, mechanics.TOX], j.int32), j.array([weather.SAND, weather.HAIL], j.int32),
            eligible=j.ones(2, bool))
        self.assertEqual(self.words(trace, 0).tolist(), [[16, -1, -1, 0, 2, 101, 95, 101, mechanics.BRN, weather.SAND]])
        self.assertEqual(self.words(trace, 1).tolist(), [[16, -1, -1, 1, 4, 3, 0, 31, mechanics.TOX, weather.HAIL]])
        self.assertEqual(trace.refusal.tolist(), [256, 256])
        self.assertFalse(bool((trace.words[:, :, 0] == public.DIRECT_FAINT).any()))

    def test_actual_chip_primitive_applied_guard_excludes_immune_dead_and_ineligible(self):
        case = weather_fixtures.WeatherTests(); case.setUp()
        old = j.array([101, 101, 0, 101], j.int32); maximum = j.full(4, 101, j.int32)
        types = j.array([[1, 0], [6, 0], [1, 0], [1, 0]], j.int32)  # row1 Rock immunity
        eligible = j.array([True, True, True, False])
        current = weather.chip(case.wt, j.full(4, weather.SAND, j.int32), types,
            j.zeros(4, j.int32), old, maximum, eligible=eligible)
        trace = callbacks.weather_chip(self.tape(4), 0, 2, old, current, maximum,
            j.zeros(4, j.int32), weather.SAND, eligible=eligible & (current < old))
        self.assertEqual(current.tolist(), [95, 101, 0, 101])
        self.assertEqual(trace.count.tolist(), [1, 0, 0, 0])
        self.assertEqual(trace.refusal.tolist(), [256, 0, 0, 0])

    def test_invalid_chip_bounds_status_reason_noop_or_increase_are_unknown_only(self):
        values = [dict(old=0, new=0), dict(old=101, new=101), dict(new=102), dict(new=-1),
            dict(maximum=100), dict(status=7), dict(reason=weather.RAIN), dict(old=j.array([1.5], j.float32))]
        for value in values:
            with self.subTest(value=value):
                trace = self.chip(self.tape(), **value)
                self.assertEqual(trace.count.tolist(), [0]); self.assertEqual(trace.refusal.tolist(), [256])
                self.assertFalse(bool(trace.words.any()))
        no_write = self.chip(self.tape(), old=12345, new=12345, maximum=12345, eligible=False)
        self.assertEqual(no_write.refusal.tolist(), [0]); self.assertFalse(bool(no_write.words.any()))

    def test_all_engine_status_enums_remain_internal_and_supported(self):
        trace = callbacks.weather_chip(self.tape(7), 0, 2, j.full(7, 100, j.int32), j.full(7, 94, j.int32),
            j.full(7, 100, j.int32), j.arange(7, dtype=j.int32), weather.HAIL, eligible=j.ones(7, bool))
        self.assertEqual(trace.count.tolist(), [1] * 7)
        self.assertEqual(trace.words[:, 0, 8].tolist(), list(range(7)))

    def test_caller_write_order_is_preserved_without_sorting_or_poststate_reconstruction(self):
        trace = self.tape(2)
        for sides in (j.array([0, 1], j.int32), j.array([1, 0], j.int32)):
            physical = j.where(sides == 0, 2, 4)
            trace = callbacks.weather_chip(trace, sides, physical, 101, 95, 101, 0, weather.SAND,
                eligible=j.ones(2, bool))
        self.assertEqual(self.words(trace, 0)[:, 3:5].tolist(), [[0, 2], [1, 4]])
        self.assertEqual(self.words(trace, 1)[:, 3:5].tolist(), [[1, 4], [0, 2]])
        self.assertEqual(trace.event_turn[:, :2].tolist(), [[1, 1], [1, 1]])

    def test_current_physical_stream_and_engine_binding_mismatch_cannot_address_a_write(self):
        trace = self.tape(3)
        trace = trace._replace(stream_id=trace.stream_id.at[0].set(999),
            cursor=trace.cursor._replace(physical_active=trace.cursor.physical_active.at[1, 0].set(1),
                engine_binding=trace.cursor.engine_binding.at[2, 0].set(1)))
        refused = self.chip(trace)
        self.assertEqual(refused.count.tolist(), [0, 0, 0]); self.assertEqual(refused.refusal.tolist(), [256] * 3)
        self.assertFalse(bool(refused.words.any()))

    def test_fresh_batch_shape_and_jit_do_not_reuse_a_different_cohort(self):
        compiled = jax.jit(lambda trace, hp: callbacks.weather_chip(trace, 0, 2, hp, hp - 6,
            hp, 0, weather.SAND, eligible=j.ones(hp.shape, bool)))
        for batch in (1, 3, 2):
            initial = self.tape(batch)
            actual = jax.block_until_ready(compiled(initial, j.full(batch, 101, j.int32)))
            self.assertEqual(actual.count.tolist(), [1] * batch)
            self.assertEqual(initial.count.tolist(), [0] * batch)
        mismatch = callbacks.weather_chip(self.tape(2), 0, 2, j.full(3, 101, j.int32),
            j.full(3, 95, j.int32), j.full(3, 101, j.int32), 0, weather.SAND, eligible=j.ones(2, bool))
        self.assertEqual(mismatch.count.tolist(), [0, 0]); self.assertEqual(mismatch.refusal.tolist(), [256, 256])
        with self.assertRaises(ValueError):
            self.chip(self.tape(2), eligible=j.ones(3, bool))
        with self.assertRaises(ValueError):
            self.chip(self.tape()._replace(count=j.zeros(1, j.float32)))

    def test_overflow_legacy_abi_and_central_latch_preserve_other_refusals(self):
        old = self.tape(legacy=True)
        self.same(self.failure(old), old); self.same(self.chip(old), old)
        self.assertEqual(public.CallbackTape._fields, ('words', 'count', 'refusal', 'turn',
            'engine_binding', 'source_binding', 'move_binding', 'producer_binding'))
        initial = self.tape(capacity=1)._replace(refusal=j.array([128], j.int32))
        full = self.chip(self.failure(initial))
        self.assertEqual(full.count.tolist(), [1]); self.assertEqual(full.refusal.tolist(), [128 | 256 | public.OVERFLOW])
        with patch.object(callbacks, '_unknown', side_effect=lambda trace, eligible: trace):
            preserved = self.failure(initial)
        self.assertEqual(preserved.refusal.tolist(), [128])


if __name__ == '__main__':
    unittest.main()
