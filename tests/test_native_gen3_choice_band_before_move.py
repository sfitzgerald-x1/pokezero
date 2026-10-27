"""Source-adjudicated Choice Band full successors; no roster replay/credit."""
import importlib.util
import unittest


@unittest.skipUnless(all(importlib.util.find_spec(n) for n in ('jax','poke_engine')),
    'requires isolated JAX and native oracle')
class ChoiceBandBeforeMoveTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import jax
        import jax.numpy as j
        from gpu_engine import effects,entry,items,sleep_talk,transition
        from gpu_engine.state import empty_state
        from tests.test_gpu_engine_rest_origin import fixture_components
        _,_,meta,_=fixture_components();meta['vocabulary']['items'].append('choiceband')
        meta['vocabulary']['moves'].append('protect');meta['moves'].append({'id':'protect','source_type':'Normal','category':'Status','flags':{}})
        n=len(meta['vocabulary']['moves']);cls.jax,cls.j,cls.meta,cls.t,cls.empty=jax,j,meta,transition,staticmethod(empty_state)
        cls.table=transition.MoveTable(j.zeros(n,j.int32),j.ones(n,j.int32),j.zeros(n,bool),j.full(n,100),j.ones(n,bool),
            j.zeros(n,j.int32).at[8].set(3),j.ones((18,18),j.int32),j.ones((18,18),j.int32),j.array(5),
            effects=effects.from_metadata(meta),entry=entry.from_metadata(meta),items=items.from_metadata(meta),sleep_talk=sleep_talk.from_metadata(meta))

    def fixture(self,side):
        j=self.j;s=self.empty(1,seed=71)
        return s._replace(hp=j.full_like(s.hp,1000),max_hp=j.full_like(s.max_hp,1000),
            stats=j.full_like(s.stats,200).at[...,0].set(1000).at[:,side,:,5].set(301).at[:,1-side,:,5].set(199),
            level=j.full_like(s.level,100),types=s.types.at[...,0].set(1),species=j.broadcast_to(j.arange(1,7),(1,2,6)),
            moves=j.broadcast_to(j.array([3,1,6,7]),s.moves.shape),pp=j.full_like(s.pp,16),item=s.item.at[0,side,0].set(2),done=j.zeros(1,bool))

    def compare(self,state,actions):
        from gpu_engine import chance_consumers as c
        from gpu_engine.fidelity_panel import JaxCaseExecutor,compare_slot,native_executor
        plan=c.plan_for_state(state,0,actions,self.table,self.meta,cap=4096)
        self.assertIsInstance(plan,c.ConsumerPlan,getattr(plan,'gaps',()))
        result=compare_slot(plan,JaxCaseExecutor(state,actions,self.table,plan,batch_size=64),lambda:native_executor(plan),
            batch_size=64,mass_tolerance=1e-5,probability_tolerance=1e-5)
        self.assertEqual(result['status'],'AGREE',{k:v for k,v in result.items() if k not in ('outcomes','consumer_certificate')})
        return result

    def test_first_freeze_paralysis_and_rest_sleep_blocked_attempts_do_not_lock(self):
        for side in range(2):
            for status in (1,5,6):
                with self.subTest(side=side,status=status):
                    s=self.fixture(side);s=s._replace(status=s.status.at[0,side,0].set(status))
                    if status==5:s=s._replace(status_counter=s.status_counter.at[0,side,0].set(3),rest_origin=s.rest_origin.at[0,side,0].set(True))
                    self.compare(s,(0,0))

    def test_existing_lock_remains_when_frozen_or_sleeping(self):
        for side in range(2):
            for status in (5,6):
                with self.subTest(side=side,status=status):
                    s=self.fixture(side);s=s._replace(choice_move=s.choice_move.at[0,side,0].set(3),last_move=s.last_move.at[0,side,0].set(3),
                        last_action_kind=s.last_action_kind.at[0,side].set(2),last_action_index=s.last_action_index.at[0,side].set(0),status=s.status.at[0,side,0].set(status))
                    if status==5:s=s._replace(status_counter=s.status_counter.at[0,side,0].set(3),rest_origin=s.rest_origin.at[0,side,0].set(True))
                    self.compare(s,(0,0))

    def test_sleep_talk_caller_locks_and_called_moves_cannot_relock_it(self):
        for side in range(2):
            for asleep in (False,True):
                with self.subTest(side=side,asleep=asleep):
                    s=self.fixture(side);s=s._replace(moves=s.moves.at[0,side,0].set(self.j.array([8,2,1,3])))
                    if asleep:s=s._replace(status=s.status.at[0,side,0].set(5),status_counter=s.status_counter.at[0,side,0].set(3),rest_origin=s.rest_origin.at[0,side,0].set(True))
                    actions=[0,0];actions[side]=1;self.compare(s,tuple(actions))

    def test_failed_protect_is_still_accepted_and_locks_its_paid_slot(self):
        for side in range(2):
            s=self.fixture(side);s=s._replace(moves=s.moves.at[0,side,0,0].set(8),pp=s.pp.at[0,side,0,0].set(1),protect_counter=s.protect_counter.at[0,side,0].set(2))
            self.compare(s,(0,0))
