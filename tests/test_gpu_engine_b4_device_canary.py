"""Actual device collector algebra with synthetic states and fake actor/step."""
from typing import NamedTuple
from types import SimpleNamespace as NS
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import jax
import jax.numpy as j
import numpy as np
from gpu_engine import b4_device_canary_collection as d, champion, transition, engine_device_observation as e
from gpu_engine.engine_device_public_hooks import PublicStepResult
from pokezero import b4_fidelity_canary as canary, dataset


class State(NamedTuple):
    hp: object
    active: object
    done: object
    forced_switch: object
    turn: object
    winner: object


class Public(NamedTuple):
    turn: object


def fake_chunk(test, *, bad_actor=False):
    b = 2
    state = State(j.full((b,2,6),100,j.int32),j.zeros((b,2),j.int32),j.zeros(b,bool),
        j.array([[False,False],[False,True]]),j.zeros(b,j.int32),j.full(b,-1,j.int32))
    carry = d.Carry(state,Public(j.ones(b,j.int32)),j.array([101,202],j.int32),j.zeros(b,j.int32),j.zeros(b,j.int32),j.full(b,-1,j.int32))
    def encode(state, view, tables, *, observer_sides, legal_action_mask):
        numeric = j.zeros((b,29,132),j.float32).at[:,:,0].set(state.turn[:,None]).at[:,23:,0].set(99.)
        return e.Observation(numeric,j.zeros((b,29,41),j.int32),j.zeros((b,29),j.int32),j.ones((b,29),bool),
            legal_action_mask,j.zeros((b,5),j.int32),j.zeros(b,j.int32),j.zeros(9,j.int32),j.zeros((b,4),j.int32))
    def actor(params,numeric,categorical,token_types,attention,c,*,compute_dtype):
        test.assertEqual(compute_dtype,'bfloat16');test.assertEqual(numeric.shape,(2,2,23,132))
        test.assertEqual(categorical.shape,(2,2,23,41));test.assertEqual(params['marker'].shape,())
        values=j.zeros((2,2)).at[0,0].set(j.nan if bad_actor else .2)
        return champion.Output(j.zeros((2,2,9)),values,j.zeros((2,2,9)))
    def step(state,actions,table,public,*,belief_tables):
        live=~state.done;turn=state.turn+live.astype(j.int32);finished=live&(j.arange(2)==0)&(turn>=2)
        after=state._replace(done=state.done|finished,turn=turn,winner=j.where(finished,0,state.winner),forced_switch=j.zeros((2,2),bool))
        return PublicStepResult(transition.StepResult(after,j.zeros(2,bool),j.zeros(2,j.int32)),Public(public.turn+live.astype(j.int32)))
    prepared=NS(engine_tables=object(),observation_tables=NS(belief=None))
    with patch.object(e,'encode',encode),patch.object(champion,'forward',actor),patch.object(d.trackers,'readout',lambda *a:None),\
            patch.object(e,'map_actions',lambda o,s,side,choice:choice),patch.object(transition,'step_with_public',step),\
            patch.object(transition,'action_mask',lambda s,t:j.zeros((2,2,9),bool).at[:,:,0].set(True)):
        execute=d.make_collector(prepared,champion.Config(),max_requests=3,chunk_steps=4)
        result=jax.device_get(jax.block_until_ready(execute(carry,{'marker':j.array(1)},j.asarray(71,j.uint32))))
    return result


class DeviceCanaryTests(unittest.TestCase):
    def test_compiled_chunk_real23_prefix_requests_tail_and_completion_only_original_cache(self):
        carry,frames=fake_chunk(self)
        self.assertEqual(carry.status.tolist(),[d.COMPLETE,d.CAPPED]);self.assertEqual(carry.frames.tolist(),[2,3])
        self.assertEqual(frames.numeric.shape,(4,2,2,23,132));self.assertFalse((frames.numeric==99.).any())
        self.assertEqual(frames.requested[0].tolist(),[[True,True],[False,True]])
        self.assertFalse(frames.requested[3].any())
        with tempfile.TemporaryDirectory() as folder:
            spool=d.Spool(folder);spool.append(frames);one=spool.game(0)
            game=canary.Game(101,'inert-101',11,(0,1))
            record=d.complete_record(game,one,winner=0,turn=2,duration=1.,source_hash='unit-source',checkpoint_sha='a'*64,archive_sha='b'*64)
            self.assertEqual(len(record.trajectory.steps),4)
            self.assertEqual([s.opponent_action_index for s in record.trajectory.steps],[0]*4)
            self.assertEqual([s.reward for s in record.trajectory.steps],[0.,0.,1.,-1.])
            examples=list(dataset.examples_from_record(record,config=dataset.TrajectoryDatasetConfig(window_size=1,ppo_target_mode='gae',discount=.9999,gae_lambda=.754)))
            self.assertEqual(len(examples),4);self.assertTrue(all(not x.terminal_capped for x in examples))
            self.assertTrue(all(x.observation.schema_version=='pokezero.observation.v4' for x in record.trajectory.steps))
            path=Path(spool.records[0]['numeric']['path']);path.chmod(0o644)
            with self.assertRaisesRegex(ValueError,'spool file changed'):spool.game(0)

    def test_bad_requested_actor_is_refused_without_fallback_training(self):
        carry,frames=fake_chunk(self,bad_actor=True)
        self.assertEqual(carry.status.tolist(),[d.REFUSED,d.CAPPED])
        self.assertFalse(frames.advanced[:,0].any());self.assertEqual(int(carry.winner[1]),-1)
        self.assertEqual(int(carry.frames[0]),1)

    def test_typed_chunk_spool_does_not_reread_other_row_or_drop_one_seat(self):
        _,frames=fake_chunk(self)
        with tempfile.TemporaryDirectory() as folder:
            spool=d.Spool(folder);spool.append(frames)
            for row in (0,1):
                got=spool.game(row)
                for actual, expected in zip(got,frames):np.testing.assert_array_equal(actual,expected[:,row])

if __name__=='__main__':unittest.main()
