"""Synthetic compiled carry ownership; no numerical engine/model/game."""
import ast
from contextlib import ExitStack, contextmanager
from pathlib import Path
import subprocess
from types import SimpleNamespace
from typing import NamedTuple
import unittest
from unittest.mock import patch

import jax
import jax.numpy as j
import numpy as np

from gpu_engine import benchmark_device_selfplay as b


class State(NamedTuple):
    done: object
    turn: object
    hp: object
    active: object
    forced_switch: object
    keys: object
    goal: object


def fixtures():
    n=5
    initial=State(j.zeros(n,bool),j.zeros(n,j.int32),j.ones((n,2,6),j.int32),
        j.zeros((n,2),j.int32),j.zeros((n,2),bool),
        j.arange(n*2,dtype=j.uint32).reshape(n,2)+42,j.asarray([1,7,3,1,2],j.int32))
    shared=j.zeros((n,1),j.int32)
    public={'clock':j.arange(n,dtype=j.int32)[:,None]+100,'same_a':shared,'same_b':shared,'none':None}
    # This intentional alias is immutable and MUST NOT be donated with carry.
    params={'immutable_bank_alias':initial.keys}
    return initial,public,params


@contextmanager
def fake_pipeline():
    from gpu_engine import champion, transition, engine_device_observation as e
    from gpu_engine import engine_device_trackers as public
    from gpu_engine.engine_device_public_hooks import PublicStepResult
    def encode(state,pub,tables,*,observer_sides,legal_action_mask):
        n=len(state.done)
        numeric=j.zeros((n,29,132),j.float32).at[:,0,0].set(pub['clock'][:,0])
        numeric=numeric.at[:,28,131].set(pub['same_a'][:,0]+pub['same_b'][:,0])
        return e.Observation(numeric,j.zeros((n,29,41),j.int32),j.zeros((n,29),j.int32),
            j.ones((n,29),bool),legal_action_mask,j.zeros((n,5),j.int32),
            j.zeros(n,j.int32),j.zeros(len(e.ERROR_KINDS),j.int32),j.zeros((n,4),j.int32))
    def actor(params,config,observations,*,actor_kind):
        n=len(observations[0].error)
        logits=j.zeros((n,2,9),j.float32).at[:,:,0].set(j.sum(params['immutable_bank_alias'])%3)
        return b.ActorResult(logits,j.zeros((n,2)),j.ones((n,2),bool))
    def step(state,actions,table,trackers,*,belief_tables):
        live=~state.done;turn=state.turn+live.astype(j.int32)
        after=state._replace(turn=turn,done=state.done|(turn>=state.goal),
            keys=state.keys+j.where(live[:,None],actions+1,0).astype(j.uint32))
        invalid=(state.goal==3)&live
        updated={**trackers,'clock':trackers['clock']+live[:,None],
            'same_a':trackers['same_a']+live[:,None],
            'same_b':trackers['same_b']+2*live[:,None]}
        return PublicStepResult(transition.StepResult(after,invalid,j.zeros_like(invalid)),updated)
    with ExitStack() as stack:
        stack.enter_context(patch.object(public,'readout',lambda carry,side:carry))
        stack.enter_context(patch.object(e,'encode',encode))
        stack.enter_context(patch.object(e,'map_actions',lambda obs,state,seat,action:action))
        stack.enter_context(patch.object(b,'run_actor',actor))
        stack.enter_context(patch.object(transition,'action_mask',lambda state,table:j.ones((len(state.done),2,9),bool)))
        stack.enter_context(patch.object(transition,'step_with_public',step))
        yield SimpleNamespace(belief=j.asarray(42,j.int32))


class CarryDonationTests(unittest.TestCase):
    def same(self,a,c):
        self.assertEqual(jax.tree.structure(a),jax.tree.structure(c))
        for x,y in zip(jax.tree.leaves(a),jax.tree.leaves(c),strict=True):
            self.assertEqual(x.dtype,y.dtype)
            np.testing.assert_array_equal(x,y)

    def test_actual_compiled_donor_owns_every_alias_and_matches_finite_outputs(self):
        initial,public,params=fixtures();roots=j.arange(5,dtype=j.uint32)+90
        immutable=jax.tree.map(lambda x:np.array(x,copy=True),(initial,public,params,roots))
        start=b.initialize(initial,public,2)
        # Force the duplicate tracker occurrence explicitly: eager slicing
        # may make distinct array objects even from one initial shared value.
        start=start._replace(trackers={**start.trackers,'same_b':start.trackers['same_a']})
        self.assertIs(start.turns,start.decisions)
        self.assertIs(start.trackers['same_a'],start.trackers['same_b'])
        owned=b.own_carry_buffers(start)
        self.assertIsNone(owned.trackers['none'])
        pointers=[x.unsafe_buffer_pointer()for x in jax.tree.leaves(owned)]
        self.assertEqual(len(pointers),len(set(pointers)))
        original={x.unsafe_buffer_pointer()for x in jax.tree.leaves((initial,public,params,start))}
        self.assertTrue(set(pointers).isdisjoint(original))
        with fake_pipeline()as tables:
            regular=b.make_driver(None,None,tables,sampling_seed=81,max_requests=2,chunk_steps=1)
            donor=b.make_driver(None,None,tables,sampling_seed=81,max_requests=2,chunk_steps=1,donate_carry=True)
            regular_compiled=regular.lower(start,initial,public,roots,params).compile()
            donor_compiled=donor.lower(owned,initial,public,roots,params).compile()
            expected,_=b.run_finite(regular_compiled,start,initial,public,roots,params)
            actual,_=b.run_finite(donor_compiled,owned,initial,public,roots,params,donate_carry=True)
        self.same(actual,expected)
        self.same((initial,public,params,roots),immutable)
        report=b.summary(actual,seconds=1.,batch=2,compile_seconds=0.,warmup_seconds=0.)
        self.assertEqual((report['completed_games'],report['capped_games'],report['failed_games']),(3,1,1))
        self.assertEqual(report['unattempted_games'],0);self.assertEqual(report['inflight_games'],0)
        self.assertEqual(report['actual_requested_views'],sum(report['decisions_by_game']))
        self.assertEqual(report['physical_encoder_forward_rows'],4*report['ticks'])
        self.assertEqual(report['producer_error_views'],0)
        # Donated input lifetime expires; original bank and params remain live.
        self.assertTrue(any(x.is_deleted()for x in jax.tree.leaves(owned)))

    def test_measure_pair_records_optin_and_disjoint_owned_starts(self):
        initial,public,params=fixtures();roots=np.arange(5,dtype=np.uint32)+100
        with fake_pipeline()as tables:
            default=b.measure_pair(params,None,None,tables,initial,initial,public,public,
                roots+5,roots,batch=2,sampling_seed=81,max_requests=2,chunk_steps=1)
            donated=b.measure_pair(params,None,None,tables,initial,initial,public,public,
                roots+5,roots,batch=2,sampling_seed=81,max_requests=2,chunk_steps=1,donate_carry=True)
        self.assertIs(default['donate_carry'],False)
        self.assertIsNone(default['carry_ownership_setup_seconds'])
        self.assertEqual(default['carry_ownership_scope'],'LEGACY_NO_DONATION')
        self.assertIs(donated['donate_carry'],True)
        self.assertGreaterEqual(donated['carry_ownership_setup_seconds'],0)
        for report in(default,donated):
            self.assertEqual(report['completed_games'],3)
            self.assertEqual(report['warmup_summary']['completed_games'],3)
        for key in('completed','capped','failed','decisions_by_game','all_final_turns','ticks',
                'actual_requested_views','physical_encoder_forward_rows','error_counts',
                'producer_error_views','all29_observation_checksum_uint32'):
            self.assertEqual(default[key],donated[key],key)
            self.assertEqual(default['warmup_summary'][key],donated['warmup_summary'][key],key)

    def test_donated_host_completion_snapshot_survives_borrowed_buffer_reuse(self):
        initial,public,params=fixtures();start=b.initialize(initial,public,2)
        def run(donate):
            completion=np.zeros(5,bool);calls=[0]
            def compiled(c,*args):
                k=calls[0];calls[0]+=1;completion[min(k,4)]=True
                return c._replace(completed=completion,ids=np.array([-1,-1]if k==4 else[0,-1]),
                    cursor=np.array(5),ticks=np.array(k+1))
            # The second completion occurs after the600s floor. Without an
            # owned seen snapshot the reused first mask hides that progress.
            times=[0.,1.,700.,701.,702.,703.,704.]
            with patch.object(b.jax,'device_get',side_effect=lambda x:x),patch.object(b.time,'perf_counter',side_effect=times):
                return b.run_finite(compiled,start,initial,public,np.arange(5),params,donate_carry=donate)
        with self.assertRaisesRegex(RuntimeError,'no completed game'):run(False)
        completed,elapsed=run(True)
        self.assertTrue(np.asarray(completed.completed).all())
        self.assertEqual(elapsed,704.)

    def test_original_numerical_policy_retirement_bodies_are_exact(self):
        base=subprocess.check_output(['git','show','2589b072534989fdf6e4e6534b9fb14152ec45ea:gpu_engine/benchmark_device_selfplay.py'],cwd=Path(__file__).parents[1])
        current=Path(b.__file__).read_bytes()
        def functions(payload):
            tree=ast.parse(payload)
            # Erase only the new static optional update-route keyword. Its
            # False/default mapping is checked separately by the route tests.
            for node in ast.walk(tree):
                if isinstance(node,ast.Call):node.keywords=[k for k in node.keywords if not(k.arg is None and isinstance(k.value,ast.Name) and k.value.id=='public_kwargs')]
            return {n.name:ast.dump(n,include_attributes=False)for n in ast.walk(tree)if isinstance(n,ast.FunctionDef)}
        old,new=functions(base),functions(current)
        for name in('run_actor','select_rows','request_seats','physical_actions','sample_keys','retire','initialize','summary','advance','chunk','body'):
            self.assertEqual(old[name],new[name],name)


if __name__=='__main__':unittest.main()
