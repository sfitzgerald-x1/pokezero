"""Authored faint-replacement carry laws; no scientific roster replay."""
import importlib.util
import json
import unittest

AVAILABLE = all(importlib.util.find_spec(n) for n in ('jax', 'poke_engine', 'pokezero_search'))
if AVAILABLE:
    import pokezero_search
    AVAILABLE = bool(getattr(pokezero_search, 'CONDITIONED_ORACLE_ENABLED', False))


@unittest.skipUnless(AVAILABLE, 'requires isolated conditioned native bindings and JAX')
class FaintReplacementTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from tests.test_conditioned_native_resume import ConditionedResumeTests
        ConditionedResumeTests.setUpClass.__func__(cls)

    def fixture(self, requested, *, hazard_side=None, layers=0, terminal=False,late=False):
        j=self.j
        s=self.empty(1,seed=315)._replace(hp=j.full((1,2,6),1000),
            max_hp=j.full((1,2,6),1000),stats=j.full((1,2,6,6),200).at[...,0].set(1000),
            species=j.ones((1,2,6),j.int32),level=j.full((1,2,6),100),
            types=j.zeros((1,2,6,2),j.int32).at[...,0].set(1),
            moves=j.full((1,2,6,4),3),pp=j.full((1,2,6,4),10),done=j.array([False]))
        s=s._replace(stats=s.stats.at[:,0,:,5].set(301).at[:,1,:,5].set(101),
            hp=s.hp.at[:,:,1].set(800),item=s.item.at[:,:,1].set(1),
            status=s.status.at[:,:,1].set(3),base_ability=s.ability)
        original=self.f.native_input_from_jax(s,0,self.meta,module=self.engine)
        self.assertIsInstance(original,self.f.NativeInput,original)
        draws=j.zeros((1,2,12),j.int32).at[:,:,4].set(15).at[:,:,5].set(7)
        chance=self.t.ChanceInputs(draws,j.array([[13,47]]),j.array([[9,99]]),
            j.array([[3,59]]),j.array([[0,1]]),j.array([0]))
        s=s._replace(hp=s.hp.at[:,list(requested),0].set(0),
            forced_switch=j.array([[side in requested for side in (0,1)]]),
            pending_residual=j.array([True]),turn=j.array([4]),residual_draws=draws,
            phazing_draws=chance.phazing,secondary_draws=chance.secondary,
            sleep_talk_draws=chance.sleep_talk)
        if hazard_side is not None:
            damage=(0,125,166,250)[layers]
            s=s._replace(spikes=s.spikes.at[:,hazard_side].set(layers),
                hp=s.hp.at[:,hazard_side,1].set(damage),
                stats=s.stats.at[:,hazard_side,1,5].set(401)
                    .at[:,1-hazard_side,1,5].set(201),
                item=s.item.at[:,hazard_side,1].set(0))
            if late:
                s=s._replace(spikes=s.spikes.at[:,1-hazard_side].set(layers),
                    stats=s.stats.at[:,hazard_side,1,5].set(201)
                        .at[:,1-hazard_side,1,5].set(401))
            if terminal:s=s._replace(hp=s.hp.at[:,hazard_side,2:].set(0))
        return s,original.certificate,chance

    def compare(self,state,certificate,actions):
        from gpu_engine import conditioned_future as cf,conditional_law as law
        from gpu_engine.fidelity_panel import compare_slot,JaxCaseExecutor
        bound=cf.suspended_input_from_jax(state,0,self.meta,certificate=certificate,module=self.engine)
        self.assertIsInstance(bound,cf.SuspendedNativeInput,bound)
        plan=law.plan_for_suspension(state,0,actions,self.table,self.meta,certificate=certificate,cap=4)
        self.assertIsInstance(plan,law.ConditionalConsumerPlan,plan)
        result=compare_slot(plan,JaxCaseExecutor(state,actions,self.table,plan,batch_size=4),
            plan.native_results,batch_size=4,mass_tolerance=1e-6,probability_tolerance=1e-6)
        self.assertEqual(result['status'],'AGREE',result)
        return bound,plan

    def test_initial_single_and_joint_faints_have_complete_laws(self):
        for requested in ((0,),(1,),(0,1)):
            with self.subTest(requested=requested):
                s,cert,chance=self.fixture(requested)
                actions=tuple(4 if side in requested else 3 for side in (0,1))
                self.compare(s,cert,actions)
                out=self.step(s,self.j.array([actions]),self.table,chance).state
                self.assertFalse(bool(out.pending_residual[0]))
                self.assertEqual(int(out.turn[0]),4)
                for side in requested:
                    self.assertEqual(int(out.active[0,side]),1)
                    self.assertEqual(int(out.hp[0,side,1]),737)

    def test_joint_hazard_ko_installs_both_and_retains_carry_until_chain_finishes(self):
        from gpu_engine import conditioned_future as cf
        for side in (0,1):
            for layers in (1,2,3):
                with self.subTest(side=side,layers=layers):
                    s,cert,chance=self.fixture((0,1),hazard_side=side,layers=layers)
                    bound,plan=self.compare(s,cert,(4,4))
                    out=self.step(s,self.j.array([[4,4]]),self.table,chance).state
                    self.assertEqual(out.active[0].tolist(),[1,1])
                    self.assertTrue(bool(out.pending_residual[0]))
                    self.assertEqual(int(out.replacement_count[0]),1)
                    self.assertEqual(int(out.hp[0,1-side,1]),800,'other callback/residual must remain cancelled')
                    for name in ('residual_draws','phazing_draws','secondary_draws','sleep_talk_draws','keys'):
                        self.assertTrue(bool(self.j.array_equal(getattr(out,name),getattr(s,name))),name)
                    native=cf.native_successors(bound,(1,1),(0,1))
                    self.assertIsInstance(native,tuple,native)
                    actions=[3,3];actions[side]=5
                    continued=cf.suspended_input_from_jax(out,0,self.meta,certificate=cert,
                        module=self.engine,native_successor=native[0][1])
                    self.assertIsInstance(continued,cf.SuspendedNativeInput,continued)
                    self.compare(out,cert,tuple(actions))
                    final=self.step(out,self.j.array([actions]),self.table,chance).state
                    self.assertEqual(int(final.hp[0,side,2]),1000-(0,125,166,250)[layers])
                    self.assertEqual(int(final.hp[0,1-side,1]),737)
                    self.assertFalse(bool(final.pending_residual[0]))

    def test_joint_terminal_hazard_ko_stops_callbacks_and_residuals_both_seats(self):
        for side in (0,1):
            for layers in (1,2,3):
                with self.subTest(side=side,layers=layers):
                    s,cert,chance=self.fixture((0,1),hazard_side=side,layers=layers,terminal=True)
                    self.compare(s,cert,(4,4))
                    out=self.step(s,self.j.array([[4,4]]),self.table,chance).state
                    self.assertTrue(bool(out.done[0]));self.assertFalse(bool(out.pending_residual[0]))
                    self.assertEqual(int(out.hp[0,1-side,1]),800)

    def test_joint_native_requests_reject_missing_extra_and_dead_choices(self):
        from gpu_engine import conditioned_future as cf
        for requested in ((0,),(1,),(0,1)):
            s,cert,_=self.fixture(requested)
            bound=cf.suspended_input_from_jax(s,0,self.meta,certificate=cert,module=self.engine)
            self.assertIsInstance(bound,cf.SuspendedNativeInput,bound)
            invalid=((None,None),(0,0),(6,1),(True,1))
            if len(requested)==1:invalid=(*invalid,(1,1))
            else:invalid=(*invalid,(1,None))
            for choices in invalid:
                with self.subTest(requested=requested,choices=choices):
                    self.assertIsInstance(cf.native_successors(bound,choices,(0,1)),self.f.Refused)

    def test_fresh_ordinary_draws_cannot_change_joint_resume(self):
        s,cert,chance=self.fixture((0,1));self.compare(s,cert,(4,4))
        first=self.step(s,self.j.array([[4,4]]),self.table,chance).state
        altered=chance._replace(draws=self.j.zeros_like(chance.draws),
            phazing=self.j.full_like(chance.phazing,59),secondary=self.j.zeros_like(chance.secondary),
            sleep_talk=self.j.zeros_like(chance.sleep_talk))
        other=self.step(s,self.j.array([[4,4]]),self.table,altered).state
        for a,b in zip(self.jax.tree.leaves(first),self.jax.tree.leaves(other)):
            self.assertTrue(bool(self.j.array_equal(a,b)))

    def test_late_hazard_ko_keeps_first_completed_callback_and_defers_residual(self):
        for side in (0,1):
            for layers in (1,2,3):
                with self.subTest(side=side,layers=layers):
                    s,cert,chance=self.fixture((0,1),hazard_side=side,layers=layers,late=True)
                    self.compare(s,cert,(4,4))
                    out=self.step(s,self.j.array([[4,4]]),self.table,chance).state
                    damage=(0,125,166,250)[layers]
                    self.assertEqual(int(out.hp[0,1-side,1]),800-damage)
                    self.assertTrue(bool(out.pending_residual[0]))
                    actions=[3,3];actions[side]=5
                    self.compare(out,cert,tuple(actions))
                    final=self.step(out,self.j.array([actions]),self.table,chance).state
                    self.assertEqual(int(final.hp[0,side,2]),1000-damage)
                    self.assertEqual(int(final.hp[0,1-side,1]),737-damage)

    def test_raw_joint_binding_rejects_boolean_indices_and_ties(self):
        from gpu_engine import conditioned_future as cf
        for requested in ((0,),(1,),(0,1)):
            s,cert,_=self.fixture(requested)
            bound=cf.suspended_input_from_jax(s,0,self.meta,certificate=cert,module=self.engine)
            self.assertIsInstance(bound,cf.SuspendedNativeInput,bound)
            capsule=json.dumps(dict(bound.capsule,version=1,schema='pokezero.suspended-turn'))
            choices=tuple(1 if side in requested else None for side in (0,1))
            for bad,ties in (((True,1),(0,1)),((1,False),(0,1)),(choices,(True,1)),(choices,(0,False))):
                with self.subTest(requested=requested,choices=bad,ties=ties):
                    with self.assertRaises((ValueError,TypeError)):
                        pokezero_search.conditioned_resume_joint(bound.raw_native,capsule,bad,ties)
