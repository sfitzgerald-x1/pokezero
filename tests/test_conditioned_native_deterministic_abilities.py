"""Authored Pressure/Levitate continuation controls, not pool fidelity credit."""
from copy import deepcopy
import unittest

from tests import test_conditioned_native_roles as roles


@unittest.skipUnless(roles.AVAILABLE, 'requires isolated conditioned native bindings and JAX')
class DeterministicAbilityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        roles.ConditionedRoleTests.setUpClass()
        cls.h = roles.ConditionedRoleTests('test_actual_carrier_roles_and_inert_benches_match_full_successors')
        from gpu_engine import pressure,entry,secondary
        cls.h.meta=deepcopy(cls.h.meta)
        cls.h.meta['vocabulary']['abilities'].append('hustle')
        cls.table = cls.h.table._replace(pressure=pressure.from_metadata(cls.h.meta),
            entry=entry.from_metadata(cls.h.meta),secondary=secondary.from_metadata(cls.h.meta))

    def fixture(self, side, entrant, entrant_base, actor, actor_base, *, move='thunderbolt',
                pp=10, outgoing='swarm', layers=0, hazard_ko=False, poison=False,
                secondary=99, contact=59):
        from gpu_engine import conditioned_future as c
        h=self.h; j=h.j
        helper=h.fixtures('test_full_successors_both_seats_saved_secondary_and_no_redraw')
        _,_,certificate,_,chance=helper.fixture(side,secondary,queued_move=move,
            poison=False,static_at=None,contact_roll=contact)
        empty=h.empty(1)
        state=empty._replace(**{k:j.asarray([v],dtype=getattr(empty,k).dtype)
            for k,v in dict(certificate.input_fields).items()})
        aid=lambda name:h.meta['vocabulary']['abilities'].index(name)
        original=state.ability.at[:,side,0].set(aid(outgoing)).at[:,side,1].set(aid(entrant_base))
        original=original.at[:,1-side,0].set(aid(actor_base))
        state=state._replace(ability=original,base_ability=original,
            pp=state.pp.at[:,1-side,0,0].set(pp),
            gender=j.ones((1,2,6),j.int32).at[:,1,:].set(2),
            spikes=state.spikes.at[:,side].set(layers),
            status=state.status.at[:,1-side,0].set(3 if poison else 0))
        if hazard_ko:
            self.assertGreater(layers,0)
            state=state._replace(hp=state.hp.at[:,side,1].set((0,125,166,250)[layers]))
        source=h.f.native_input_from_jax(state,0,h.meta,module=h.engine)
        self.assertIsInstance(source,h.f.NativeInput,source)
        state=state._replace(ability=state.ability.at[:,side,1].set(aid(entrant))
            .at[:,1-side,0].set(aid(actor)))
        source=h.f._native_input_from_data(h.f._sample(state,0),h.meta,module=h.engine,
            original_static=source.certificate.static_native,
            original_genders=dict(source.certificate.input_fields)['gender'])
        self.assertIsInstance(source,h.f.NativeInput,source)
        initial=j.zeros((1,2),j.int32)
        paused=h.step(state,initial,self.table,chance).state
        branches=h.engine.generate_instructions(source.state,
            'batonpass' if side==0 else move,move if side==0 else 'batonpass')
        self.assertEqual(len(branches),1)
        native=source.state.apply_instructions(branches[0])
        self.assertEqual(native.reverse_instructions(branches[0]).to_string(),source.state.to_string())
        bound=c.suspended_input_from_jax(paused,0,h.meta,certificate=source.certificate,module=h.engine)
        self.assertIsInstance(bound,c.SuspendedNativeInput,bound)
        native_key=c._complete_key(h.f._canonical_native_future(native,source.certificate,
            boundary='suspended'),bound.capsule)
        jax_key=c.canonical_resumed_jax(paused,0,h.meta,native_input=bound)
        self.assertNotIsInstance(native_key,h.f.Refused,native_key)
        self.assertNotIsInstance(jax_key,h.f.Refused,jax_key)
        self.assertEqual(native_key.key,jax_key.key)
        return paused,source.certificate,chance

    def compare(self,state,certificate,side,*,table=None):
        from gpu_engine.conditional_law import plan_for_suspension,ConditionalConsumerPlan
        from gpu_engine.fidelity_panel import JaxCaseExecutor,compare_slot
        table=self.table if table is None else table
        actions=(4,0) if side==0 else (0,4)
        plan=plan_for_suspension(state,0,actions,table,self.h.meta,certificate=certificate,cap=4)
        self.assertTrue(isinstance(plan,ConditionalConsumerPlan),
            [g.code for g in plan.gaps] if isinstance(plan,self.h.f.Refused) else type(plan))
        return compare_slot(plan,JaxCaseExecutor(state,actions,table,plan,batch_size=4),
            plan.native_results,batch_size=4,mass_tolerance=1e-5,probability_tolerance=1e-5)

    def resumed(self,state,chance,side):
        actions=self.h.j.array([[4,0] if side==0 else [0,4]])
        return self.h.step(state,actions,self.table,chance).state

    def test_current_pressure_pp_clamps_and_originals_stay_inert_both_seats(self):
        contexts=[('pressure','hustle','levitate','trace','swarm',2),
            ('static','pressure','_none','levitate','swarm',1),
            ('levitate','pressure','pressure','hustle','swarm',1),
            ('_none','levitate','static','pressure','pressure',1),
            ('pressure','levitate','pressure','pressure','swarm',2)]
        for side in (0,1):
            for move in ('thunderbolt','bodyslam'):
                for pp in (1,2,10):
                    for entrant,base,actor,actor_base,outgoing,cost in contexts:
                        with self.subTest(side=side,move=move,pp=pp,entrant=entrant,base=base,actor=actor):
                            state,cert,chance=self.fixture(side,entrant,base,actor,actor_base,
                                move=move,pp=pp,outgoing=outgoing,poison=True)
                            self.assertEqual(self.compare(state,cert,side)['status'],'AGREE')
                            after=self.resumed(state,chance,side)
                            self.assertEqual(int(after.pp[0,1-side,0,0]),max(0,pp-cost))
                            self.assertTrue(bool(self.h.j.array_equal(after.base_ability,state.base_ability)))

    def test_current_levitate_and_hazard_ko_carry_both_seats_all_layers(self):
        for side in (0,1):
            for layers in range(4):
                for entrant,base in (('levitate','hustle'),('static','levitate'),('pressure','trace')):
                    for ko in (False,True) if layers else (False,):
                        with self.subTest(side=side,layers=layers,entrant=entrant,base=base,ko=ko):
                            state,cert,chance=self.fixture(side,entrant,base,'pressure','hustle',
                                layers=layers,hazard_ko=ko,poison=True)
                            self.assertEqual(self.compare(state,cert,side)['status'],'AGREE')
                            after=self.resumed(state,chance,side)
                            cancelled=ko and entrant!='levitate'
                            self.assertEqual(int(after.pp[0,1-side,0,0]),
                                10 if cancelled else 8 if entrant=='pressure' else 9)
                            if cancelled:
                                self.assertTrue(bool(after.pending_residual[0]))
                                self.assertEqual(after.residual_draws.tolist(),state.residual_draws.tolist())
                                self.assertEqual(after.keys.tolist(),state.keys.tolist())
                            if entrant=='levitate':
                                plain=state._replace(spikes=state.spikes.at[:,side].set(0))
                                reference=self.resumed(plain,chance,side)
                                self.assertEqual(after.hp.tolist(),reference.hp.tolist())

    def test_static_independent_saved_contact_and_secondary_with_pressure_actor(self):
        for side in (0,1):
            for secondary in (29,30):
                for contact in (19,20):
                    state,cert,_=self.fixture(side,'static','pressure','pressure','trace',
                        move='bodyslam',secondary=secondary,contact=contact)
                    self.assertEqual(self.compare(state,cert,side)['status'],'AGREE')

    def test_wrong_jax_pressure_table_is_a_divergence(self):
        for side in (0,1):
            state,cert,_=self.fixture(side,'pressure','hustle','levitate','trace')
            bad=self.table._replace(pressure=self.table.pressure._replace(
                pressure=self.h.j.zeros_like(self.table.pressure.pressure)))
            self.assertEqual(self.compare(state,cert,side,table=bad)['status'],'DIVERGE')


if __name__=='__main__':unittest.main()
