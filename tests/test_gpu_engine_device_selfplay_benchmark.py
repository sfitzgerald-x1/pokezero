"""Accounting/action contracts; no model loading or scientific games."""
from typing import NamedTuple
import unittest
import jax
import jax.numpy as j
import numpy as np
from gpu_engine import benchmark_device_selfplay as b


class State(NamedTuple):
    done: object
    turn: object


class DeviceSelfplayBenchmarkTests(unittest.TestCase):
    def test_request_order_switches_map_to_the_engine_physical_bench(self):
        active = j.asarray([[3, 1], [0, 5]], j.int32)
        targets = j.asarray([[[0, 2, 1, 4, 5], [5, 0, 2, 3, 4]],
                             [[1, 2, 3, 4, 5], [0, 1, 2, 3, 4]]], j.int32)
        fn = jax.jit(b.physical_actions)
        np.testing.assert_array_equal(fn(j.asarray([[6, 4], [3, 8]]), targets, active),
                                      [[5, 8], [3, 8]])

    def test_completion_cap_failure_and_padding_are_disjoint_and_never_redrawn(self):
        initial = State(j.zeros(5, bool), j.arange(5, dtype=j.int32))
        s = State(j.asarray([False, False]), j.asarray([0, 0], j.int32))
        c = b.Carry(s, j.asarray([[10], [20]]), j.asarray([0, 1], j.int32),
            j.asarray([0, 1], j.int32), j.asarray(2, j.int32),
            j.zeros(5, bool), j.zeros(5, bool), j.zeros(5, bool),
            j.zeros(5, j.int32), j.zeros(5, j.int32), j.zeros(7, j.int32),
            j.asarray(0, j.int32), j.asarray(0, j.int32), j.asarray(0, j.int32), j.asarray(0, j.uint32))
        advance = jax.jit(lambda c, after, fail: b.retire(c, after, c.trackers+1,
            initial, failure=fail, requested=j.asarray([[True, True], [True, False]]),
            errors=j.asarray([0, 0, 1, 0, 0, 0, 0]), max_requests=2,
            tracker_initialize=lambda state: state.turn[:, None]+100))
        c = advance(c, State(j.asarray([True, False]), j.asarray([9, 8], j.int32)), j.asarray([False, False]))
        np.testing.assert_array_equal(c.ids, [2, 3]); np.testing.assert_array_equal(c.trackers, [[102], [103]])
        np.testing.assert_array_equal(c.completed, [True, False, False, False, False])
        np.testing.assert_array_equal(c.capped, [False, True, False, False, False])
        c = advance(c, State(j.asarray([False, True]), j.asarray([7, 6], j.int32)), j.asarray([True, False]))
        np.testing.assert_array_equal(c.ids, [4, -1])
        c = advance(c, State(j.asarray([True, True]), j.asarray([5, 6], j.int32)), j.asarray([False, False]))
        got = b.summary(c, seconds=2., batch=2, compile_seconds=3., warmup_seconds=1.)
        self.assertEqual((got['completed_games'], got['capped_games'], got['failed_games']), (3, 1, 1))
        self.assertEqual(got['full_games_per_second'], 1.5)
        self.assertEqual((got['unattempted_games'], got['inflight_games']), (0, 0))
        np.testing.assert_array_equal(c.decisions, [2, 1, 2, 1, 2])

    def test_stream_keys_follow_roots_requests_and_seats_not_lane_position(self):
        roots = j.asarray([99, 12], j.uint32); requests = j.asarray([4, 1], j.int32)
        a = b.sample_keys(roots, requests, sampling_seed=33)
        other = b.sample_keys(roots[::-1], requests[::-1], sampling_seed=33)
        np.testing.assert_array_equal(a[::-1], other)
        self.assertFalse(np.array_equal(a[:, 0], a[:, 1]))


if __name__ == '__main__':
    unittest.main()
