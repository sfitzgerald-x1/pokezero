"""Pure storage/gather and fake-clock controls; no actor or engine execution."""
import ast
from collections import namedtuple
from pathlib import Path
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

import jax
import jax.numpy as j
import numpy as np

from gpu_engine import device_selfplay_profile as p
from gpu_engine.state import empty_state
from gpu_engine.transition import ChanceInputs
from gpu_engine import engine_device_public_hooks as hooks


def fixture():
    # Empty_state is explicitly storage only, not a battle constructor.
    s=empty_state(4);mon=lambda x:j.full((4,2,6),x,j.int32)
    s=s._replace(done=j.array([False,True,True,False]),species=mon(1),level=mon(80),
        hp=j.broadcast_to(j.arange(4,dtype=j.int32)[:,None,None]+50,(4,2,6)),max_hp=mon(100),gender=mon(1))
    public=hooks.initialize(s).trackers
    chance=ChanceInputs(*(j.zeros((4,2,12)if k=='draws'else(4,)if k=='initial_tie'else(4,2),j.int32)
        for k in ChanceInputs._fields))
    return p.FixedInputs(s,public,j.zeros((4,2),j.int32),chance,~s.done)


class Clock:
    def __init__(self):self.time=0
    def __call__(self):self.time+=1000000;return self.time


class FakeJit:
    def __init__(self):self.lowered=[];self.compiled=[];self.called=[]
    def __call__(self,function):
        def lower(*args):
            self.lowered.append((function,args))
            def compile():
                self.compiled.append(function)
                def execute(*actual):
                    self.called.append((function,actual));return function(*actual)
                return execute
            return NS(compile=compile)
        return NS(lower=lower)


class DeviceSelfplayProfileTests(unittest.TestCase):
    def test_one_lower_compile_repeated_identical_input_and_blocked_intervals(self):
        jit=FakeJit();blocks=[];value=np.array([1,2],np.int32)
        def f(x):self.assertIs(x,value);return x+1
        def block(x):blocks.append(x);return x
        out,report=p._measure(f,(value,),repeats=3,warmup=2,jit=jit,block=block,clock=Clock())
        np.testing.assert_array_equal(out,[2,3]);np.testing.assert_array_equal(value,[1,2])
        self.assertEqual(len(jit.lowered),1);self.assertEqual(len(jit.compiled),1);self.assertEqual(len(jit.called),6)
        self.assertIs(blocks[0][0],value);self.assertEqual(len(blocks),7)
        self.assertEqual(report['lowering_wall_ms'],1);self.assertEqual(report['compile_or_cache_load_wall_ms'],1)
        self.assertEqual(report['first_execute_blocked_ms'],1);self.assertEqual(report['warm_execute_blocked_ms'],[1]*3)
        self.assertEqual(report['warm_median_ms'],1);self.assertEqual(report['warmup_execute_blocked_ms'],[1,1])

    def test_tail_contains_each_active_row_once_and_masks_only_duplicate_padding(self):
        s=fixture();before=np.asarray(s.state.done).copy();tail,meta=p.gather_tail(s,capacity=3)
        self.assertEqual(meta['source_rows'],[0,3,1]);self.assertEqual(meta['padding_mask'],[False,False,True])
        self.assertEqual(meta['active_rows'],2);self.assertEqual(meta['padding_rows'],1)
        np.testing.assert_array_equal(tail.state.done,[False,False,True]);np.testing.assert_array_equal(s.state.done,before)
        for old,new in zip(jax.tree.leaves((s.trackers,s.chance,s.actions)),jax.tree.leaves((tail.trackers,tail.chance,tail.actions))):
            np.testing.assert_array_equal(new,np.asarray(old)[[0,3,1]])
        for name in s.state._fields:
            np.testing.assert_array_equal(getattr(tail.state,name),np.asarray(getattr(s.state,name))[[0,3,1]])
        none,status=p.gather_tail(s,capacity=1)
        self.assertIsNone(none);self.assertEqual(status['status'],'UNMEASURED')
        none,status=p.gather_tail(p.FixedInputs(s.state._replace(done=j.ones(4,bool)),s.trackers,s.actions,s.chance,j.zeros(4,bool)))
        self.assertIsNone(none);self.assertEqual(status['reason'],'NO_ACTIVE_ROWS')
        self.assertNotIn('warm_median_ms',status)

    def test_fake_full_tail_phase_counts_output_geometry_and_no_game_claim(self):
        from pokezero.v5_model import V5ModelConfig
        from pokezero.v5_contract import LabelVocabulary
        config=V5ModelConfig(10,'a'*64,LabelVocabulary('b'*64,('surf',),(('gardevoir',1),)))
        snapshot=fixture();jit=FakeJit();chance_seen=[]
        Obs=namedtuple('Obs','error');Actor=namedtuple('Actor','finite')
        Step=namedtuple('Step','state invalid_actions');Public=namedtuple('Public','step trackers')
        def components(*args,**kwargs):
            self.assertIs(kwargs['dense_public_updates'],True)
            def encoder(s,t):return (Obs(error=np.zeros(len(s.done),np.int32)),)*2
            def forward(params,obs):return Actor(finite=np.ones((len(obs[0].error),2),bool))
            def public(s,a,t,chance):
                chance_seen.append(chance);return Public(step=Step(state=s,invalid_actions=np.zeros(len(s.done),bool)),trackers=t)
            def bare(s,a,chance):chance_seen.append(chance);return Step(state=s,invalid_actions=np.zeros(len(s.done),bool))
            def joint(s,a,t,chance,params):return encoder(s,t),forward(params,encoder(s,t)),public(s,a,t,chance)
            return dict(encoder_both_seats=encoder,public29_bf16_forward=forward,step_with_public=public,
                bare_step_same_chance=bare,fixed_actions_joint=joint)
        with patch.object(p,'make_components',side_effect=components),patch('gpu_engine.v5_model.validate_weights'), \
                patch('jax.jit',jit),patch('jax.block_until_ready',side_effect=lambda x:x), \
                patch.object(p.time,'perf_counter_ns',Clock()):
            report=p.profile_fixed(snapshot,params={'inert':np.ones(1)},config=config,engine_tables=object(),
                observation_tables=object(),bindings={'fixture':'not-actual-measurement'},enable_profile=True,
                repeats=2,warmup=1,tail_capacity=3,dense_public_updates=True)
        self.assertEqual(len(jit.compiled),10);self.assertEqual(len(jit.lowered),10)
        self.assertEqual(len(jit.called),40)
        full,tail=report['profiles'];self.assertEqual(full['geometry']['physical_bf16_owner_rows'],8)
        self.assertEqual(tail['geometry']['physical_bf16_owner_rows'],6)
        self.assertEqual(tail['geometry']['active_engine_rows'],2);self.assertEqual(tail['geometry']['actual_requested_owner_rows'],4)
        self.assertEqual(full['public_vs_bare_differing_state_fields'],[])
        self.assertGreater(full['returned_payloads']['fixed_actions_joint']['returned_array_leaves'],60)
        self.assertEqual(full['returned_payloads']['fixed_actions_joint']['nonfinite_elements'],0)
        self.assertEqual(report['played_games'],0);self.assertIsNone(report['full_game_throughput'])
        self.assertFalse(report['qualification']);self.assertEqual(report['sampled_chance_calls'],0)
        self.assertIs(report['dense_public_updates'],True)
        self.assertTrue(all(x is snapshot.chance for x in chance_seen[:12]))
        self.assertEqual(full['phases']['public29_bf16_forward']['warm_execute_blocked_ms'],[1.,1.])

    def test_enablement_and_real_source_paths_no_sampling_no_policy_mutation(self):
        with self.assertRaises(ValueError):p.profile_fixed(None,params=None,config=None,engine_tables=None,
            observation_tables=None,bindings={})
        snapshot=fixture()
        with self.assertRaises(ValueError):p.gather_tail(snapshot,capacity=17)
        wrong=p.FixedInputs(snapshot.state,snapshot.trackers,snapshot.actions,snapshot.chance._replace(draws=j.zeros((4,2,11),j.int32)),snapshot.active_rows)
        with self.assertRaises(ValueError):p.gather_tail(wrong)
        wrong=p.FixedInputs(snapshot.state,snapshot.trackers,snapshot.actions,snapshot.chance,j.ones(4,bool))
        with self.assertRaises(ValueError):p.gather_tail(wrong)
        tree=ast.parse(Path(p.__file__).read_text())
        functions={n.name:n for n in tree.body if isinstance(n,ast.FunctionDef)}
        factory=functions['make_components'];nested={n.name:n for n in factory.body if isinstance(n,ast.FunctionDef)}
        public=next(n for n in ast.walk(nested['public_step'])if isinstance(n,ast.Call))
        self.assertEqual(ast.unparse(public.func),'transition.step_with_public')
        self.assertEqual(ast.unparse(next(k.value for k in public.keywords if k.arg=='chance_inputs')),'chance')
        forward=next(n for n in ast.walk(nested['forward'])if isinstance(n,ast.Call))
        self.assertEqual(ast.unparse(next(k.value for k in forward.keywords if k.arg=='actor_kind')),"'v5-public29'")
        forbidden={'sample_chance_inputs','state_from_pool','initialize_with_public','sample','map_actions','build_v5_model'}
        self.assertFalse(any(isinstance(n,ast.Call)and isinstance(n.func,ast.Attribute)and n.func.attr in forbidden for n in ast.walk(tree)))
        self.assertEqual(ast.unparse(nested['joint'].body[-1].value),'(observations, actor, result)')

    def test_dense_flag_is_explicit_static_and_only_forwarded_to_public_step(self):
        tables=NS(belief='same-source-tables')
        for kwargs,expected in (({},False),({'dense_public_updates':True},True)):
            components=p.make_components('engine','observations-unused','model',**kwargs)
            # Replace only the closure's observation-table object, never an
            # original engine call: this is an inert forwarding control.
            components=p.make_components('engine',tables,'model',**kwargs)
            with patch('gpu_engine.transition.step_with_public',return_value='same-output')as public:
                self.assertEqual(components['step_with_public']('state','actions','trackers','chance'),'same-output')
            self.assertEqual(public.call_args.args,('state','actions','engine','trackers'))
            self.assertEqual(public.call_args.kwargs,dict(belief_tables=tables.belief,chance_inputs='chance',dense_public_updates=expected))
        for flag in (1,None,'true'):
            with self.assertRaises(ValueError):p.make_components(None,None,None,dense_public_updates=flag)


if __name__=='__main__':unittest.main()
