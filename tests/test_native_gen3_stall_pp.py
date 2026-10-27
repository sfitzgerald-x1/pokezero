"""Complete authored stalled-move successors; not a scientific roster replay."""
import importlib.util
import unittest


@unittest.skipUnless(all(importlib.util.find_spec(n) for n in ('jax','poke_engine')),
    'requires isolated JAX and native oracle')
class NativeStallPPTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import jax
        import jax.numpy as j
        from gpu_engine import effects,entry,items,sleep_talk,transition
        from gpu_engine.state import empty_state
        from tests.test_gpu_engine_rest_origin import fixture_components
        _,_,cls.meta,_=fixture_components();names=cls.meta['vocabulary']['moves']
        for name in ('protect','detect','endure'):
            names.append(name);cls.meta['moves'].append({'id':name,'source_type':'Normal','category':'Status','flags':{}})
        n=len(names)
        cls.jax,cls.j,cls.t,cls.empty=jax,j,transition,staticmethod(empty_state)
        priority=j.zeros(n,j.int32).at[n-3:n-1].set(3).at[n-1].set(4)
        cls.table=transition.MoveTable(j.zeros(n,j.int32),j.ones(n,j.int32),j.zeros(n,bool),j.full(n,100),
            j.ones(n,bool),priority,j.ones((18,18),j.int32),j.ones((18,18),j.int32),j.array(5),
            effects=effects.from_metadata(cls.meta),entry=entry.from_metadata(cls.meta),
            items=items.from_metadata(cls.meta),sleep_talk=sleep_talk.from_metadata(cls.meta))

    def fixture(self,side,move='protect',pp=16,counter=2):
        j=self.j;s=self.empty(1,seed=71);names=self.meta['vocabulary']['moves']
        return s._replace(hp=j.full_like(s.hp,101),max_hp=j.full_like(s.max_hp,101),
            stats=j.full_like(s.stats,200).at[...,0].set(101).at[:,side,:,5].set(301).at[:,1-side,:,5].set(199),
            level=j.full_like(s.level,100),types=s.types.at[...,0].set(1),
            species=j.broadcast_to(j.arange(1,7),(1,2,6)),
            moves=j.broadcast_to(j.array([3,1,6,7]),s.moves.shape).at[0,side,0,0].set(names.index(move)),
            pp=j.full_like(s.pp,16).at[0,side,0,0].set(pp),
            protect_counter=s.protect_counter.at[0,side,0].set(counter),
            last_action_kind=j.full_like(s.last_action_kind,2),last_action_index=j.ones_like(s.last_action_index),
            last_move=s.last_move.at[0,:,0].set(1),done=j.zeros(1,bool))

    def compare(self,state,actions,*,cached=True):
        from gpu_engine import chance_consumers as c
        from gpu_engine.fidelity_panel import JaxCaseExecutor,compare_slot,native_executor
        plan=c.plan_for_state(state,0,actions,self.table,self.meta,cap=4096)
        self.assertIsInstance(plan,c.ConsumerPlan,getattr(plan,'gaps',()))
        execute=JaxCaseExecutor(state,actions,self.table,plan,batch_size=64)
        if not cached:
            execute.canonical_cache.canonicalize=lambda host,count,unused_plan: tuple(
                plan.canonical_successor(host,i) for i in range(count))
        result=compare_slot(plan,execute,
            lambda:native_executor(plan),batch_size=64,mass_tolerance=1e-5,probability_tolerance=1e-5)
        self.assertEqual(result['status'],'AGREE',{k:v for k,v in result.items() if k not in ('outcomes','consumer_certificate')})
        return result

    def test_both_seat_full_laws_include_failed_attempt_pp_and_switch_controls(self):
        for side in range(2):
            for move in ('protect','detect','endure'):
                for counter in (2,8):
                    for pp in (1,16):
                        for switched in (False,True):
                            s=self.fixture(side,move,pp,counter);actions=[0,0];actions[1-side]=4 if switched else 0
                            with self.subTest(side=side,move=move,counter=counter,pp=pp,switched=switched):
                                self.compare(s,tuple(actions))

    def test_sleep_talk_called_protect_and_paralysis_gate_full_laws(self):
        for side in range(2):
            s=self.fixture(side)
            s=s._replace(moves=s.moves.at[0,side,0].set(self.j.array([8,2,0,0])),
                pp=s.pp.at[0,side,0,2:].set(0),status=s.status.at[0,side,0].set(5),
                status_counter=s.status_counter.at[0,side,0].set(3),rest_origin=s.rest_origin.at[0,side,0].set(True),
                last_action_index=s.last_action_index.at[0,side].set(0),last_move=s.last_move.at[0,side,0].set(8))
            actions=[0,0];actions[side]=1;self.compare(s,tuple(actions))
            s=self.fixture(side);s=s._replace(status=s.status.at[0,side,0].set(1))
            self.compare(s,(0,0))

    def test_rest_wake_prefix_full_successors(self):
        for side in range(2):
            s=self.fixture(side,pp=1)
            s=s._replace(status=s.status.at[0,side,0].set(5),status_counter=s.status_counter.at[0,side,0].set(1),
                rest_origin=s.rest_origin.at[0,side,0].set(True))
            self.compare(s,(0,0))

    def test_cached_and_uncached_actual_complete_laws_are_identical(self):
        for side in range(2):
            for paralyzed in (False,True):
                with self.subTest(side=side,paralyzed=paralyzed):
                    state=self.fixture(side)
                    if paralyzed:state=state._replace(status=state.status.at[0,side,0].set(1))
                    self.assertEqual(self.compare(state,(0,0)),self.compare(state,(0,0),cached=False))
