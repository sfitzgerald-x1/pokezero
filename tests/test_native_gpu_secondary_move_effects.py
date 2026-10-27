"""Complete authored secondary laws against Rust; not a scientific roster replay."""
import importlib.util
import unittest


@unittest.skipUnless(all(importlib.util.find_spec(n) for n in ('jax','poke_engine')),
    'requires isolated JAX and native oracle')
class NativeSecondaryMoveEffectsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from tests.test_gpu_engine_secondary_move_effects import components
        from gpu_engine import transition
        from gpu_engine.state import empty_state
        cls.jax,cls.j,cls.meta,cls.table=components(real_accuracy=True);cls.tr=transition;cls.empty=staticmethod(empty_state)

    def fixture(self,side,move,*,source='_none',target='_none',sub=0,stage=0,endure=False,opposing_attack=False):
        j=self.j;s=self.empty(1,seed=701);names=self.meta['vocabulary']['moves'];abilities=self.meta['vocabulary']['abilities']
        return s._replace(hp=j.full_like(s.hp,1000),max_hp=j.full_like(s.max_hp,1000),
            stats=j.full_like(s.stats,200).at[...,0].set(1000).at[:,side,:,5].set(301).at[:,1-side,:,5].set(199),
            level=j.full_like(s.level,100),types=s.types.at[...,0].set(1),species=j.broadcast_to(j.arange(1,7),(1,2,6)),
            moves=j.broadcast_to(j.array([1,3,4,5]),s.moves.shape)
                .at[0,side,0].set(j.array([names.index(move),1,3,4]))
                .at[0,1-side,0].set(j.array([4,1,3,5]) if endure else j.array([9,1,3,4]) if opposing_attack else j.array([1,3,4,5])),pp=j.full_like(s.pp,8),
            boosts=s.boosts.at[0,side,0,:5].set(stage),substitute_hp=s.substitute_hp.at[0,1-side,0].set(sub),
            ability=s.ability.at[0,side,0].set(abilities.index(source)).at[0,1-side,0].set(abilities.index(target)),
            base_ability=s.base_ability.at[0,side,0].set(abilities.index(source)).at[0,1-side,0].set(abilities.index(target)),done=j.zeros(1,bool))\
            ._replace(hp=j.full_like(s.hp,1000).at[0,1-side,0].set(1 if endure else 1000))

    def compare(self,state,actions=(0,0)):
        from gpu_engine import chance_consumers as c
        from gpu_engine.fidelity_panel import JaxCaseExecutor,compare_slot,native_executor
        # Authored compound laws include up to 55,296 cases; this local test
        # allowance does not change the registered scientific panel limit.
        plan=c.plan_for_state(state,0,actions,self.table,self.meta,cap=65536)
        if not isinstance(plan,c.ConsumerPlan):self.fail(repr(getattr(plan,'gaps',())))
        result=compare_slot(plan,JaxCaseExecutor(state,actions,self.table,plan,batch_size=64),
            lambda:native_executor(plan),batch_size=64,mass_tolerance=1e-5,probability_tolerance=1e-5)
        self.assertEqual(result['status'],'AGREE',{k:v for k,v in result.items() if k not in ('outcomes','consumer_certificate')})
        return result

    def test_omni_boost_complete_laws_substitutes_and_target_guards_both_seats(self):
        for side in (0,1):
            for move in ('silverwind','ancientpower'):
                for option in ({},{'source':'serenegrace'},{'target':'shielddust'},{'target':'clearbody'},
                               {'sub':1},{'sub':999},{'stage':5},{'stage':6},{'endure':True},{'target':'static'}):
                    with self.subTest(side=side,move=move,option=option):self.compare(self.fixture(side,move,**option))

    def test_speed_drop_and_burn_complete_laws_with_guarded_targets_both_seats(self):
        for side in (0,1):
            for move in ('mudshot','sacredfire'):
                for option in ({},{'source':'serenegrace'},{'target':'shielddust'},{'target':'clearbody'},
                               {'target':'waterveil'},{'sub':1},{'sub':999}):
                    with self.subTest(side=side,move=move,option=option):self.compare(self.fixture(side,move,**option))

    def test_voluntary_switch_complete_laws_both_seats(self):
        # Switching is a genuine target-entry control, not a new roster.
        for side in (0,1):
            for move in ('silverwind','mudshot','sacredfire'):
                with self.subTest(side=side,move=move):
                    actions=[0,0];actions[1-side]=4;self.compare(self.fixture(side,move),tuple(actions))

    def test_boosted_defense_and_independent_contact_change_later_attack_full_laws(self):
        for side in (0,1):
            for move in ('silverwind','ancientpower'):
                for target in ('_none','static'):
                    with self.subTest(side=side,move=move,target=target):
                        self.compare(self.fixture(side,move,target=target,opposing_attack=True))
