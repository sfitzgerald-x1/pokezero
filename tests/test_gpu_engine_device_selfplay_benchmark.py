"""Accounting/action contracts; no model loading or scientific games."""
from typing import NamedTuple
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import jax
import jax.numpy as j
import numpy as np
from gpu_engine import benchmark_device_selfplay as b


class State(NamedTuple):
    done: object
    turn: object


class DeviceSelfplayBenchmarkTests(unittest.TestCase):
    def test_v5_actor_consumes_all29_history_tokens_and_keeps_seat_order(self):
        from gpu_engine import v5_model, engine_device_observation as e
        def obs(values):
            num=j.zeros((2,29,132),j.float32).at[:,28,131].set(j.asarray(values,j.float32))
            return e.Observation(num,j.zeros((2,29,41),j.int32),j.zeros((2,29),j.int32),
                j.ones((2,29),bool),j.ones((2,9),bool),j.zeros((2,5),j.int32),
                j.zeros(2,j.int32),j.zeros(9,j.int32),j.zeros((2,4),j.int32))
        def forward(params,inputs,config,*,compute_dtype):
            self.assertEqual(compute_dtype,'bfloat16')
            self.assertEqual(inputs['numeric_features'].shape,(4,1,29,132))
            values=inputs['numeric_features'][:,0,28,131]
            return v5_model.Output(j.zeros((4,9)),values,j.zeros((4,101)),j.ones(4,bool))
        with patch.object(v5_model,'forward',forward):
            out=b.run_actor({},None,[obs([11,33]),obs([22,44])],actor_kind='v5-public29')
        np.testing.assert_array_equal(out.value,[[11,22],[33,44]])
        self.assertTrue(np.asarray(out.finite).all())

    def test_full_public_step_uses_constructor_carry_and_compressed_move_mapping(self):
        from gpu_engine import champion, transition, engine_device_observation as e
        from gpu_engine import engine_device_trackers as public
        from gpu_engine.engine_device_public_hooks import PublicStepResult
        class FullState(NamedTuple):
            done: object
            turn: object
            hp: object
            active: object
            forced_switch: object
        initial=FullState(j.zeros(3,bool),j.zeros(3,j.int32),j.ones((3,2,6),j.int32),
            j.broadcast_to(j.asarray([3,1],j.int32),(3,2)),j.zeros((3,2),bool))
        initial_public=j.arange(3,dtype=j.int32)[:,None]+100
        tables=SimpleNamespace(belief=j.asarray(42,j.int32))
        refuse_first=[False]
        def encode(state,pub,tables,*,observer_sides,legal_action_mask):
            n=len(observer_sides);side=observer_sides
            numeric=j.zeros((n,29,132),j.float32).at[:,0,0].set(pub[:,0])
            numeric=numeric.at[:,28,131].set(pub[:,0])
            legal=j.zeros((n,9),bool).at[j.arange(n),j.where(side==0,0,4)].set(True)
            return e.Observation(numeric,j.zeros((n,29,41),j.int32),j.zeros((n,29),j.int32),
                j.ones((n,29),bool),legal,j.broadcast_to(j.asarray([5,0,2,3,4],j.int32),(n,5)),
                j.zeros(n,j.int32),j.zeros(len(e.ERROR_KINDS),j.int32),
                j.broadcast_to(j.asarray([3,1,2,0],j.int32),(n,4)))
        def forward(params,*,numeric,categorical,token_types,attention,c,compute_dtype):
            self.assertEqual(compute_dtype,'bfloat16')
            shape=numeric.shape[:2]
            value=j.where(numeric[...,0,0]==100,j.nan,0.) if refuse_first[0] else j.zeros(shape)
            return champion.Output(j.zeros(shape+(9,)),value,j.zeros(shape+(9,)))
        def step(state,actions,table,trackers,*,belief_tables):
            self.assertIs(belief_tables,tables.belief)
            bad=j.any(actions!=j.asarray([3,8]),axis=1)&~state.done
            after=state._replace(done=j.ones_like(state.done),turn=state.turn+(~state.done).astype(j.int32))
            return PublicStepResult(transition.StepResult(after,bad,j.zeros_like(bad)),trackers+10)
        for bad_actor in (False,True):
            refuse_first[0]=bad_actor
            with patch.object(public,'readout',lambda carry,side:carry),patch.object(e,'encode',encode),\
                 patch.object(champion,'forward',forward),patch.object(transition,'step_with_public',step),\
                 patch.object(transition,'action_mask',lambda state,table:j.ones((2,2,9),bool)):
                start=b.initialize(initial,initial_public,2)
                fn=b.make_driver(None,None,tables,sampling_seed=21,chunk_steps=2)
                out=jax.block_until_ready(fn(start,initial,initial_public,j.arange(3,dtype=j.uint32),{}))
            result=b.summary(out,seconds=1.,batch=2,compile_seconds=0.,warmup_seconds=0.)
            self.assertEqual(result['completed_games'],3);self.assertEqual(result['failed_games'],0)
            self.assertEqual(result['actual_requested_views'],6)
            self.assertEqual(result['physical_encoder_forward_rows'],8)
            self.assertEqual(len(result['error_counts']),10)
            self.assertEqual(result['producer_error_views'],2 if bad_actor else 0)
            self.assertEqual(result['error_counts'][-1],2 if bad_actor else 0)
            self.assertGreater(result['all29_observation_checksum_uint32'],0)

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
            j.zeros(5, j.int32), j.zeros(5, j.int32), j.zeros(10, j.int32),
            j.asarray(0, j.int32), j.asarray(0, j.int32), j.asarray(0, j.int32), j.asarray(0, j.uint32),j.asarray(0,j.int32))
        advance = jax.jit(lambda c, after, fail: b.retire(c, after, c.trackers+1,
            initial, initial.turn[:, None]+100, failure=fail,
            requested=j.asarray([[True, True], [True, False]]),
            errors=j.asarray([0, 0, 1, 0, 0, 0, 0, 0, 0, 0]), max_requests=2,error_views=1))
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
