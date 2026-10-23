"""Authored refill accounting fixtures, not scientific measurements."""
import importlib.util
import contextlib
from types import SimpleNamespace
import unittest
from unittest import mock


@unittest.skipUnless(importlib.util.find_spec('jax'),'requires isolated spike JAX')
class RefillTests(unittest.TestCase):
    def test_measurement_only_sampling_and_failed_cleanup_stops_later_batches(self):
        import numpy as np
        from gpu_engine import benchmark_refill as b
        device=mock.Mock(device_kind='authored GPU')
        head=mock.Mock();head.devices.return_value={device}
        def advance(state,ids,counts,cursor,initial):
            cursor=int(cursor)
            values=(state,ids,counts,np.array(cursor+1),np.array([cursor-1]),np.array([True]),np.array([False]))
            return values,np.array([1]),np.array(False),(head,)
        # Fake dependency outputs and clock, not a GPU execution or performance sample.
        for stopped in (True,False):
            with self.subTest(stopped=stopped), contextlib.ExitStack() as stack:
                patch=lambda target,**kw:stack.enter_context(mock.patch.object(*target,**kw))
                patch((b,'sole_gpu'),return_value=device)
                patch((b,'load_bound_pool'),return_value=({'species':range(200000)},{},'pool'))
                identity=patch((b,'cuda_device_uuid'),return_value='bound-uuid')
                patch((b.champion,'load'),return_value=({},None,{}))
                patch((b.transition,'tables_from_metadata'),return_value=None)
                patch((b.transition,'state_from_pool'),return_value={'x':np.arange(2)})
                patch((b.jax,'default_device'),side_effect=lambda _:contextlib.nullcontext())
                patch((b.jax,'device_put'),side_effect=lambda x,_:x)
                patch((b.jax,'block_until_ready'),side_effect=lambda x:x)
                patch((b,'driver'),return_value=advance)
                patch((b.subprocess,'check_output'),side_effect=lambda args,**kw:'source' if 'rev-parse' in args else '')
                # warm start/check/end, measured pre-start/check/end, repeated per batch
                clock=[0,0,2,2.1,2.2,3,5,5.1, 6,6,8,8.1,8.2,9,11,11.1]
                patch((b.time,'perf_counter'),side_effect=clock)
                sampler=patch((b,'DeviceWindow'))
                sampler.return_value.start.side_effect=[3,9]
                sampler.return_value.finish.return_value={'worker_stopped':stopped,'status':'SAMPLED' if stopped else 'PARTIAL'}
                patch((b,'print'),create=True)
                result=b.measure('pool','weights','pool',batches=(1,1),warm_seconds=1,measure_seconds=1,device_telemetry=True)
                self.assertEqual(identity.call_count,1)
                self.assertEqual(sampler.call_count,2 if stopped else 1)
                self.assertNotIn('device_telemetry',result['rows'][0]['warmup'])
                self.assertIn('device_telemetry',result['rows'][0]['measured'])
                self.assertEqual(result['protocol_completed'],stopped)
                if not stopped:
                    self.assertEqual(len(result['rows']),1)
                    self.assertEqual(result['rows'][0]['status'],'TELEMETRY_CLEANUP_FAILED')
                    self.assertIsNone(result['rows'][0]['games_per_second'])

    def test_complete_precedes_cap_and_entire_state_resets_once(self):
        import jax
        import jax.numpy as j
        from gpu_engine.state import empty_state
        from gpu_engine.benchmark_refill import refill
        initial=empty_state(6,seed=19)
        initial=initial._replace(done=j.zeros(6,bool),turn=j.arange(6)*10,
            queued_actor=j.arange(6),hp=j.arange(6)[:,None,None]+initial.hp)
        state=jax.tree.map(lambda x:x[:3],initial)
        state=state._replace(done=j.array([True,False,False]),turn=j.array([99,90,40]))
        values=jax.jit(lambda s,i,r,c:refill(s,i,r,c,initial,max_requests=5))(
            state,j.arange(3),j.array([5,5,4]),j.array(3))
        new,ids,requests,cursor,retired,complete,cap=values
        self.assertEqual(ids.tolist(),[3,4,2]);self.assertEqual(requests.tolist(),[0,0,4])
        self.assertEqual(int(cursor),5);self.assertEqual(retired.tolist(),[0,1,-1])
        self.assertEqual(complete.tolist(),[True,False,False]);self.assertEqual(cap.tolist(),[False,True,False])
        for actual,want in zip(new,initial):
            self.assertTrue(bool(j.array_equal(actual[:2],want[j.array([3,4])])))
        again=refill(new,ids,requests,cursor,initial,max_requests=5)
        self.assertEqual(again[4].tolist(),[-1,-1,-1]);self.assertEqual(int(again[3]),5)

    def test_exhaustion_never_wraps_and_empty_slots_freeze(self):
        import jax.numpy as j
        from gpu_engine.state import empty_state
        from gpu_engine.benchmark_refill import refill
        initial=empty_state(3)._replace(done=j.zeros(3,bool))
        state=empty_state(2)._replace(done=j.ones(2,bool))
        new,ids,requests,cursor,retired,complete,cap=refill(state,j.array([0,1]),j.ones(2,j.int32),j.array(2),initial,max_requests=5)
        self.assertEqual(ids.tolist(),[2,-1]);self.assertEqual(int(cursor),3)
        self.assertEqual(new.done.tolist(),[False,True]);self.assertEqual(retired.tolist(),[0,1])
        again=refill(new,ids,requests,cursor,initial,max_requests=5)
        self.assertEqual(again[4].tolist(),[-1,-1]);self.assertEqual(int(again[3]),3)

    def test_rng_follows_game_id_request_and_seat_not_slot(self):
        import jax.numpy as j
        from gpu_engine.benchmark_refill import policy_keys
        keys=policy_keys(j.array([7,2,19]),j.array([3,9,1]))
        self.assertTrue(bool(j.array_equal(keys[1],policy_keys(j.array([2]),j.array([9]))[0])))
        self.assertFalse(bool(j.array_equal(keys[0,0],keys[0,1])))
        self.assertFalse(bool(j.array_equal(keys[1],policy_keys(j.array([2]),j.array([10]))[0])))

    def test_accounting_keeps_caps_and_right_censoring_separate(self):
        from gpu_engine.benchmark_refill import summarize
        r=summarize([0,1,2],2,1,seconds=2,requests=10,assigned=5,batch=3,turns=80)
        self.assertEqual((r['games_per_second'],r['in_flight_games'],r['capped_games']),(1.,2,1))
        self.assertEqual(r['mean_completed_game_turns'],40.)
        with self.assertRaises(ValueError):summarize([0,0],2,0,seconds=2,requests=10,assigned=3,batch=3,turns=80)

    def test_driver_advances_and_refills_capped_game_with_valid_sampling(self):
        import jax
        import jax.numpy as j
        from gpu_engine.benchmark_refill import driver
        from gpu_engine.champion import Output
        from gpu_engine.transition import MoveTable
        from gpu_engine.state import empty_state
        table=MoveTable(j.array([0,35,50]),j.array([0,1,0]),j.ones(3,bool),j.full(3,100),
            j.ones(3,bool),j.zeros(3,j.int32),j.ones((18,18),j.int32),j.ones((18,18),j.int32),j.array(2))
        s=empty_state(4,seed=7);s=s._replace(hp=j.full_like(s.hp,100),max_hp=j.full_like(s.hp,100),
            species=j.ones_like(s.species),level=j.full_like(s.level,100),stats=j.full_like(s.stats,100),
            moves=j.ones_like(s.moves),pp=j.full_like(s.pp,8),done=j.zeros(4,bool))
        state=jax.tree.map(lambda x:x[:2],s)
        forward=lambda o:Output(j.zeros((2,2,9)),j.zeros((2,2)),j.zeros((2,2,9)))
        values,turns,invalid,heads=driver(forward,table,max_requests=1)(state,j.arange(2),j.zeros(2,j.int32),j.array(2),s)
        new,ids,counts,cursor,ended,done,capped=values
        self.assertFalse(bool(invalid));self.assertEqual(ids.tolist(),[2,3])
        self.assertEqual(capped.tolist(),[True,True]);self.assertEqual(done.tolist(),[False,False])
        self.assertEqual(ended.tolist(),[0,1]);self.assertEqual(counts.tolist(),[0,0])
        for actual,want in zip(new,s):self.assertTrue(bool(j.array_equal(actual,want[2:])))
