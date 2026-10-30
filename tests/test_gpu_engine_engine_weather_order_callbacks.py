"""Source-engineering order annotations on literal internal tapes only."""
import unittest
from unittest.mock import patch

import jax
import jax.numpy as j
import numpy as np

from gpu_engine import engine_weather_order_callbacks as order
from gpu_engine import engine_public_events as public, weather, mechanics
import test_gpu_engine_engine_item_callbacks as fixtures


class WeatherOrderTests(unittest.TestCase):
    def tape(self, batch=1, capacity=16, *, legacy=False):
        state = fixtures.state(batch)._replace(active=j.broadcast_to(j.array([2, 4], j.int32), (batch, 2)))
        return fixtures.tape(state, capacity, legacy=legacy)

    def append(self, tape, side=0, physical=2, *, eligible=True, kind=16):
        shape = tape.count.shape
        ints = lambda value: j.broadcast_to(j.asarray(value, j.int32), shape)
        return public._append(tape, kind, ints(-1), ints(-1), ints(side), ints(physical),
            tuple(ints(v) for v in (101, 0, 123, mechanics.TOX, weather.SAND)),
            j.broadcast_to(j.asarray(eligible, bool), shape))

    def annotate(self, tape, *, before=0, side=0, physical=2, speed0=301, speed1=199, eligible=True):
        return order.annotate_weather_chip(tape, before, side, physical, speed0, speed1, eligible=eligible)

    def same(self, actual, expected):
        for a, b in zip(jax.tree.leaves(actual), jax.tree.leaves(expected)):
            np.testing.assert_array_equal(a, b)

    def unchanged_except_latch(self, actual, expected, refusal):
        self.same(actual._replace(refusal=expected.refusal), expected)
        self.assertEqual(actual.refusal.tolist(), refusal)

    def test_just_appended_literal_payload_address_and_stamps_are_preserved(self):
        before = self.tape(2)._replace(refusal=j.array([128, 1], j.int32))
        trace = self.append(before, j.array([0, 1], j.int32), j.array([2, 4], j.int32))
        actual = order.annotate_weather_chip(trace, before.count, j.array([0, 1], j.int32),
            j.array([2, 4], j.int32), j.array([16777217, 99], j.int32), j.array([71, 203], j.int32),
            eligible=j.ones(2, bool))
        self.assertEqual(actual.words[:, 0].tolist(),
            [[16, 16777217, 71, 0, 2, 101, 0, 123, mechanics.TOX, weather.SAND],
             [16, 99, 203, 1, 4, 101, 0, 123, mechanics.TOX, weather.SAND]])
        self.same(actual._replace(words=trace.words, refusal=trace.refusal), trace)
        self.assertEqual(actual.refusal.tolist(), [128 | 256, 1 | 256])

    def test_two_writes_keep_original_order_and_one_before_handler_speed_snapshot(self):
        trace = self.tape()
        for side, physical in ((0, 2), (1, 4)):
            before = trace.count
            trace = self.append(trace, side, physical)
            trace = self.annotate(trace, before=before, side=side, physical=physical, speed0=99, speed1=203)
        # Faster side1 is deliberately second in the engine's unchanged loop.
        # Annotation neither sorts receipts nor inserts a faint after HP0.
        self.assertEqual(trace.count.tolist(), [2])
        self.assertEqual(trace.words[0, :2, 1:5].tolist(), [[99, 203, 0, 2], [99, 203, 1, 4]])
        self.assertFalse(bool((trace.words[:, :, 0] == public.DIRECT_FAINT).any()))
        self.assertEqual(trace.refusal.tolist(), [256])

    def test_ties_are_diagnostic_only_and_remain_unknown(self):
        trace = self.annotate(self.append(self.tape()), speed0=200, speed1=200)
        self.assertEqual(trace.words[0, 0, 1:3].tolist(), [200, 200])
        self.assertEqual(trace.refusal.tolist(), [256]); self.assertEqual(trace.count.tolist(), [1])

    def test_exactly_one_append_no_scan_or_reannotation(self):
        empty = self.tape()
        twice = self.append(self.append(empty))
        unrelated = self.append(empty, kind=15)
        once = self.append(empty)
        for trace, before in ((empty, 0), (twice, 0), (once, -1), (once, 1), (once, 2147483647), (unrelated, 0)):
            with self.subTest(before=before, count=trace.count.tolist()):
                self.unchanged_except_latch(self.annotate(trace, before=before), trace, [256])
        annotated = self.annotate(once)
        self.same(self.annotate(annotated, speed0=1, speed1=2), annotated)

    def test_mixed_actual_write_eligibility_and_row_local_counts(self):
        trace = self.tape(3)
        trace = self.append(trace, eligible=j.array([False, True, False]))
        before = trace.count
        trace = self.append(trace, eligible=j.array([True, True, False]))
        actual = self.annotate(trace, before=before, eligible=j.array([True, True, False]))
        self.assertEqual(actual.count.tolist(), [1, 2, 0])
        self.assertEqual(actual.words[0, 0, 1:3].tolist(), [301, 199])
        self.assertEqual(actual.words[1, 1, 1:3].tolist(), [301, 199])
        np.testing.assert_array_equal(actual.words[1, 0], trace.words[1, 0])
        self.assertEqual(actual.refusal.tolist(), [256, 256, 0])

    def test_invalid_speed_types_values_and_shapes_do_not_mutate(self):
        trace = self.append(self.tape())
        for speed in (0, -1, 2**31, None, True, j.array([1.5], j.float32), np.array([2**32 + 1], np.int64), j.ones(2, j.int32)):
            with self.subTest(speed=str(speed)):
                self.unchanged_except_latch(self.annotate(trace, speed0=speed), trace, [256])
                self.unchanged_except_latch(self.annotate(trace, speed1=speed), trace, [256])
        self.same(self.annotate(trace, speed0=None, eligible=False), trace)

    def test_wrong_victim_or_no_longer_current_physical_is_unknown_only(self):
        trace = self.append(self.tape())
        for value in (dict(side=1, physical=4), dict(side=2), dict(physical=1), dict(side=j.array([0.5], j.float32))):
            self.unchanged_except_latch(self.annotate(trace, **value), trace, [256])
        changed = trace._replace(cursor=trace.cursor._replace(physical_active=trace.cursor.physical_active.at[0, 0].set(1)))
        self.unchanged_except_latch(self.annotate(changed), changed, [256])

    def test_stale_stream_engine_or_event_clock_generation_cannot_be_annotated(self):
        trace = self.append(self.tape())
        variations = [trace._replace(stream_id=j.array([99], j.int32)),
            trace._replace(engine_binding=trace.engine_binding.at[0].set(1)),
            trace._replace(event_turn=trace.event_turn.at[0, 0].add(1)),
            trace._replace(event_phase=trace.event_phase.at[0, 0].set(1)),
            trace._replace(event_generation=trace.event_generation.at[0, 0, 0].add(1)),
            trace._replace(cursor=trace.cursor._replace(initialized=j.array([False]))),
            trace._replace(cursor=trace.cursor._replace(admitted=j.array([False])))]
        for stale in variations:
            self.unchanged_except_latch(self.annotate(stale), stale, [256])

    def test_invalid_receipt_payload_or_absent_actor_contract_is_unknown_only(self):
        trace = self.append(self.tape())
        for column, value in ((1, 0), (2, 10), (5, 0), (6, -1), (6, 101), (7, 100), (8, 7), (9, weather.RAIN)):
            damaged = trace._replace(words=trace.words.at[0, 0, column].set(value))
            self.unchanged_except_latch(self.annotate(damaged), damaged, [256])

    def test_fresh_batch_jit_no_stale_cohort_and_shape_guards(self):
        compiled = jax.jit(lambda trace, before, speed: order.annotate_weather_chip(
            trace, before, 0, 2, speed, speed + 1, eligible=j.ones(speed.shape, bool)))
        for batch in (1, 3, 2):
            before = self.tape(batch); trace = self.append(before)
            actual = jax.block_until_ready(compiled(trace, before.count, j.full(batch, 201, j.int32)))
            self.assertEqual(actual.words[:, 0, 1:3].tolist(), [[201, 202]] * batch)
            self.assertEqual(actual.refusal.tolist(), [256] * batch)
        with self.assertRaises(ValueError):
            self.annotate(self.append(self.tape(2)), eligible=j.ones(3, bool))
        with self.assertRaises(ValueError):
            self.annotate(self.tape()._replace(count=j.zeros(1, j.float32)))
        with self.assertRaises(ValueError):
            self.annotate(self.tape()._replace(cursor=self.tape(2).cursor))

    def test_overflow_legacy_abi_and_helper_specific_latch(self):
        legacy = self.tape(legacy=True)
        self.same(self.annotate(legacy), legacy)
        self.assertEqual(public.CallbackTape._fields, ('words', 'count', 'refusal', 'turn',
            'engine_binding', 'source_binding', 'move_binding', 'producer_binding'))
        trace = self.append(self.tape(capacity=1))
        overflow = self.append(trace)
        self.unchanged_except_latch(self.annotate(overflow, before=1), overflow, [public.OVERFLOW | 256])
        self.unchanged_except_latch(self.annotate(overflow, before=0), overflow, [public.OVERFLOW | 256])
        initial = trace._replace(refusal=j.array([128], j.int32))
        with patch.object(order, '_unknown', side_effect=lambda tape, eligible: tape):
            annotated = self.annotate(initial)
        self.assertEqual(annotated.refusal.tolist(), [128])


if __name__ == '__main__':
    unittest.main()
