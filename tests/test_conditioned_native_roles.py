"""Authored causal-role comparisons, not real-pool coverage or admission."""
from copy import deepcopy
import importlib.util
from pathlib import Path
import unittest


AVAILABLE=all(importlib.util.find_spec(name) for name in ('jax','numpy','poke_engine','pokezero_search'))
if AVAILABLE:
    import pokezero_search
    AVAILABLE=bool(getattr(pokezero_search,'CONDITIONED_ORACLE_ENABLED',False))


@unittest.skipUnless(AVAILABLE,'requires isolated conditioned-oracle native bindings and JAX')
class ConditionedRoleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import jax
        import jax.numpy as j
        import poke_engine
        from gpu_engine import future_state as f,transition as t,baton,entry,items,secondary
        from gpu_engine.state import empty_state
        spec=importlib.util.spec_from_file_location('resume_fixtures',
            Path(__file__).with_name('test_conditioned_native_resume.py'))
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        module.ConditionedResumeTests.setUpClass()
        cls.fixtures=module.ConditionedResumeTests
        cls.abilities=['swarm','insomnia','synchronize','innerfocus','speedboost','wonderguard',
            'voltabsorb','cutecharm','pressure','runaway','pickup','soundproof','shielddust',
            'intimidate','levitate','naturalcure','plus','earlybird','minus']
        cls.meta=deepcopy(cls.fixtures.meta)
        cls.meta['vocabulary']['abilities']=['_none','static',*cls.abilities,'trace','effectspore']
        cls.meta['vocabulary']['items']=['_none','leftovers','lumberry','choiceband','stick']
        cls.table=cls.fixtures.table._replace(baton=baton.from_metadata(cls.meta),
            entry=entry.from_metadata(cls.meta),items=items.from_metadata(cls.meta),
            secondary=secondary.from_metadata(cls.meta))
        cls.jax,cls.j,cls.engine,cls.f,cls.t=jax,j,poke_engine,f,t
        cls.empty=staticmethod(empty_state);cls.step=staticmethod(jax.jit(t.step_with_chance))

    def setup_case(self,replacing,ability,item,queued_move,*,poison_carrier=False,trace_base=False):
        from gpu_engine import conditioned_future as c
        helper=self.fixtures('test_full_successors_both_seats_saved_secondary_and_no_redraw')
        _,_,certificate,_,chance=helper.fixture(replacing,99,queued_move=queued_move,poison=False,static_at=1)
        base=self.empty(1)
        state=base._replace(**{k:self.j.asarray([v],dtype=getattr(base,k).dtype)
            for k,v in dict(certificate.input_fields).items()})
        ability_id=self.meta['vocabulary']['abilities'].index(ability)
        state=state._replace(ability=state.ability.at[:,replacing,0].set(ability_id),
            item=state.item.at[:,replacing,0].set(self.meta['vocabulary']['items'].index(item)),
            gender=self.j.ones((1,2,6),self.j.int32).at[:,1,:].set(2))
        if poison_carrier:state=state._replace(status=state.status.at[:,replacing,0].set(3))
        # Both parties retain a reached-unsafe ability/item on an untouched
        # bench Pokemon. Full successor comparison must retain it without hooks.
        state=state._replace(ability=state.ability.at[:,:,2].set(self.meta['vocabulary']['abilities'].index('effectspore')),
            item=state.item.at[:,:,2].set(self.meta['vocabulary']['items'].index('choiceband')))
        original=self.f.native_input_from_jax(state,0,self.meta,module=self.engine)
        self.assertIsInstance(original,self.f.NativeInput,original)
        if trace_base:
            state=state._replace(ability=state.ability.at[:,replacing,0].set(1))
            original=self.f._native_input_from_data(self.f._sample(state,0),self.meta,module=self.engine,
                original_static=original.certificate.static_native,
                original_genders=dict(original.certificate.input_fields)['gender'])
            self.assertIsInstance(original,self.f.NativeInput,original)
        actions=self.j.zeros((1,2),self.j.int32)
        paused=self.step(state,actions,self.table,chance).state
        branches=self.engine.generate_instructions(original.state,
            'batonpass' if replacing==0 else queued_move,
            queued_move if replacing==0 else 'batonpass')
        self.assertEqual(len(branches),1)
        native_paused=original.state.apply_instructions(branches[0])
        bound=c.suspended_input_from_jax(paused,0,self.meta,certificate=original.certificate,module=self.engine)
        self.assertIsInstance(bound,c.SuspendedNativeInput,bound)
        # Independently generated native suspension and JAX suspension retain
        # the same complete represented physical/carry key before resumption.
        native_key=c._complete_key(self.f._canonical_native_future(native_paused,
            original.certificate,boundary='suspended'),bound.capsule)
        jax_key=c.canonical_resumed_jax(paused,0,self.meta,native_input=bound)
        self.assertNotIsInstance(native_key,self.f.Refused,native_key)
        self.assertNotIsInstance(jax_key,self.f.Refused,jax_key)
        self.assertEqual(native_key.key,jax_key.key)
        return paused,original.certificate

    def compare(self,state,certificate,replacing):
        from gpu_engine.conditional_law import plan_for_suspension,ConditionalConsumerPlan
        from gpu_engine.fidelity_panel import JaxCaseExecutor,compare_slot
        actions=(4,0) if replacing==0 else (0,4)
        plan=plan_for_suspension(state,0,actions,self.table,self.meta,certificate=certificate,cap=4)
        self.assertIsInstance(plan,ConditionalConsumerPlan,plan)
        return compare_slot(plan,JaxCaseExecutor(state,actions,self.table,plan,batch_size=4),
            plan.native_results,batch_size=4,mass_tolerance=1e-5,probability_tolerance=1e-5)

    def test_actual_carrier_roles_and_inert_benches_match_full_successors(self):
        items=('leftovers','lumberry','choiceband')
        for side in (0,1):
            for move in ('thunderbolt','bodyslam'):
                for i,ability in enumerate(self.abilities):
                    with self.subTest(side=side,move=move,ability=ability,item=items[i%3]):
                        state,certificate=self.setup_case(side,ability,items[i%3],move)
                        result=self.compare(state,certificate,side)
                        self.assertEqual(result['status'],'AGREE',result)

    def test_natural_cure_departure_preserves_full_status_and_carry(self):
        for side in (0,1):
            for move in ('thunderbolt','bodyslam'):
                state,certificate=self.setup_case(side,'naturalcure','choiceband',move,poison_carrier=True)
                result=self.compare(state,certificate,side)
                self.assertEqual(result['status'],'AGREE',result)

    def test_original_trace_departure_exposes_jax_base_restoration_gap(self):
        for side in (0,1):
            state,certificate=self.setup_case(side,'trace','leftovers','thunderbolt',trace_base=True)
            result=self.compare(state,certificate,side)
            self.assertEqual(result['status'],'DIVERGE',result)
            self.assertFalse(result['support_equal'])
            jax_key=next(row['key'] for row in result['outcomes'] if row['jax_probability']=='1')
            native_key=next(row['key'] for row in result['outcomes'] if row['native_probability']==1.)
            expected=f"{self.f.SIDE_NAMES[side]}.pokemon[0].ability"
            self.assertEqual(self.f.field_differences(jax_key,native_key),((expected,'static','trace'),))

    def test_unrepresented_stick_is_a_named_construction_refusal(self):
        helper=self.fixtures('test_full_successors_both_seats_saved_secondary_and_no_redraw')
        _,_,certificate,_,_=helper.fixture(0,99,poison=False)
        base=self.empty(1)
        state=base._replace(**{k:self.j.asarray([v],dtype=getattr(base,k).dtype)
            for k,v in dict(certificate.input_fields).items()})
        state=state._replace(item=state.item.at[:,0,0].set(self.meta['vocabulary']['items'].index('stick')))
        value=self.f.native_input_from_jax(state,0,self.meta,module=self.engine)
        self.assertIsInstance(value,self.f.Refused,value)
        self.assertTrue({gap.code for gap in value.gaps}&{'CONSTRUCTION_FAILED','CONSTRUCTION_NOT_LOSSLESS'})
        self.assertEqual(dict(value.raw_jax)['item'][0][0],self.meta['vocabulary']['items'].index('stick'))


if __name__=='__main__':unittest.main()
