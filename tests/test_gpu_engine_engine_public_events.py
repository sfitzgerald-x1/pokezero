"""Tiny authored callback/component fixtures, never games or B4 qualification."""
from dataclasses import replace
from copy import deepcopy
import unittest
import numpy as np


class EnginePublicEventsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import test_gpu_engine_public_context as context_fixture
        context_fixture.PublicContextTests.setUpClass.__func__(cls)
        from gpu_engine import engine_public_events as e, transition as t
        from gpu_engine.state import empty_state
        cls.e, cls.t, cls.empty, cls.context_fixture = e, t, staticmethod(empty_state), context_fixture
        names = ['_none','bodyslam','surf','protect','sleeptalk','struggle','not_in_dex']
        powers = [0,40,50,0,0,50,50]
        cls.metadata = dict(vocabulary=dict(moves=names), moves=[None]+[
            dict(id=name,base_power=powers[i],type=1,category='Physical')
            for i,name in enumerate(names[1:],1)])
        j = cls.j
        cls.engine_table = t.MoveTable(j.array(powers,j.int32),j.array([0]+[1]*6,j.int32),
            j.array([False]+[True]*6),j.full((7,),100,j.int32),j.ones((7,),bool),
            j.zeros((7,),j.int32),j.ones((18,18),j.int32),j.ones((18,18),j.int32),j.array(5,j.int32))
        cls.exposure = e.HealthExposure(e.SHOWDOWN,'gen3randombattle',True,False)
        cls.producer = e.compile_catalog(cls.metadata,cls.engine_table,public_moves=cls.catalog,dex=cls.dex,
            exposure=cls.exposure)

    def state(self,batch=1):
        j = self.j
        state = self.empty(batch,seed=29)
        stats = j.full_like(state.stats,200).at[...,0].set(100).at[:,0,:,5].set(300)
        return state._replace(stats=stats,level=j.full_like(state.level,100),
            hp=j.full_like(state.hp,100),max_hp=j.full_like(state.hp,100),
            moves=j.ones_like(state.moves),pp=j.full_like(state.pp,10),
            types=state.types.at[...,0].set(1),done=j.zeros_like(state.done),
            active=j.broadcast_to(j.array([3,5],j.int32),(batch,2)))

    def chance(self,batch=1):
        j = self.j
        draws=j.zeros((batch,2,12),j.int32).at[:,:,4].set(1)
        return self.t.ChanceInputs(draws,*[j.zeros((batch,2),j.int32) for _ in range(4)],j.zeros(batch,j.int32))

    def registry(self):
        from gpu_engine.public_context_protocol import PublicAddress
        return self.e.PublicRegistry({0:(3,PublicAddress(0,0,1)),1:(5,PublicAddress(1,0,1))})

    def project(self,tape,owner=0,**kwargs):
        return self.e.project(tape,battle=0,stream_id=1,owner=owner,next_sequence=2,catalog=self.producer,
            registry=kwargs.pop('registry',self.registry()),expected_binding=self.producer.binding,
            expected_engine_binding=self.j.array(self.producer.engine_binding,self.j.uint32),**kwargs)

    def trace(self,state,actions,**kwargs):
        return self.e.trace_step(state,actions,self.engine_table,catalog=self.producer,
            expected_binding=self.producer.binding,
            expected_engine_binding=np.array(self.producer.engine_binding,np.uint32),**kwargs)

    def same_state(self,a,b):
        for key in a._fields:
            np.testing.assert_array_equal(getattr(a,key),getattr(b,key),err_msg=key)

    def test_same_real_core_and_actual_callback_order_without_private_slot_alias(self):
        state,chance=self.state(),self.chance()
        actions=self.j.zeros((1,2),self.j.int32)
        ordinary=self.t.step_with_chance(state,actions,self.engine_table,chance)
        traced=self.trace(state,actions,chance_inputs=chance)
        self.assertEqual(ordinary._fields,('state','invalid_actions','placeholder_moves'))
        self.same_state(ordinary.state,traced.step.state)
        np.testing.assert_array_equal(ordinary.invalid_actions,traced.step.invalid_actions)
        np.testing.assert_array_equal(ordinary.placeholder_moves,traced.step.placeholder_moves)
        tape=traced.callbacks
        self.assertEqual(tape.words[0,:4,0].tolist(),[self.e.MOVE,self.e.MAIN_HP,self.e.MOVE,self.e.MAIN_HP])
        self.assertEqual(tape.words[0,:4,1:5].tolist(),[[0,3,1,5],[-1,-1,1,5],[1,5,0,3],[-1,-1,0,3]])
        self.assertEqual(tape.count.tolist(),[4])
        for owner in (0,1):
            projected=self.project(tape,owner)
            self.assertEqual([event.sequence for event in projected.events],[2,3,4,5])
            self.assertEqual(projected.events[0].actor_slot,0)
            self.assertEqual(projected.events[0].payload[1],self.catalog.move_id('bodyslam'))
            self.assertNotEqual(projected.events[0].payload[1],1)
            self.assertFalse(projected.full_encoder_admitted)
            self.assertFalse(projected.full_line_consumers_complete)
            self.assertFalse(projected.B4_qualified)
            self.assertTrue(projected.refusal & self.e.UNINSTRUMENTED)

    def test_direct_ko_receipt_order_cancels_foe_and_preserves_zero_fnt(self):
        state=self.state()._replace(hp=self.state().hp.at[0,1,5].set(1),
            status=self.state().status.at[0,1,5].set(4))
        traced=self.trace(state,self.j.zeros((1,2),self.j.int32),chance_inputs=self.chance())
        self.assertEqual(traced.callbacks.count.tolist(),[3])
        self.assertEqual(traced.callbacks.words[0,:3,0].tolist(),[self.e.MOVE,self.e.MAIN_HP,self.e.DIRECT_FAINT])
        self.assertEqual(int(traced.step.state.pp[0,1,5,0]),10)
        for owner in (0,1):
            events=self.project(traced.callbacks,owner).events
            self.assertEqual([int(e.kind) for e in events],[15,10,12])
            self.assertEqual(events[1].payload,(1,0,0,0,1,1,0,0))
            self.assertEqual(events[2].payload,(1,0,0,0,0,0,0,0))

    def test_emitted_words_match_frozen_source_projectors_and_named_consumers(self):
        from gpu_engine import public_context_protocol as context_wire, public_mon_protocol as mon_wire
        from gpu_engine.public_context_protocol import PublicAddress,PublicAddresses
        state=self.state()
        traced=self.trace(state,self.j.zeros((1,2),self.j.int32),chance_inputs=self.chance())
        addresses=PublicAddresses({s:PublicAddress(s,0,1) for s in (0,1)},
            {'p1a: Snorlax':PublicAddress(0,0,1),'p2a: Absol':PublicAddress(1,0,1)})
        for owner in (0,1):
            projection=self.project(traced.callbacks,owner)
            source_events=[]
            for event in projection.events:
                sequence=event.sequence
                if event.kind==self.i.Kind.MOVE:
                    actor='p1a: Snorlax' if event.actor_side==0 else 'p2a: Absol'
                    target='p1a: Snorlax' if event.target_side==0 else 'p2a: Absol'
                    line=f'|move|{actor}|Body Slam|{target}'
                    source_events.extend(context_wire.project_line(line,sequence=sequence,turn=1,
                        addresses=addresses,moves=self.catalog,abilities=self.abilities).events)
                else:
                    target='p1a: Snorlax' if event.target_side==0 else 'p2a: Absol'
                    line=f'|-damage|{target}|{event.payload[1]}/{event.payload[2]}'
                    source_events.extend(mon_wire.project_line(line,sequence=sequence,turn=1,
                        addresses=addresses,species_ids={},source=self.host,damage_metadata=True,
                        active_idents={0:'p1a: Snorlax',1:'p2a: Absol'}).events)
            np.testing.assert_array_equal([e.words() for e in projection.events],[e.words() for e in source_events])
            # This TURN is explicitly prior public fixture evidence, NOT emitted
            # or claimed complete by the engine callback subset.
            events=[self.i.PublicEvent(self.i.Kind.TURN,1,1,self.i.Phase.ACTION),*projection.events]
            fixture=self.context_fixture.PublicContextTests
            initial=fixture.initial(self)
            result=fixture.fold(self,initial,fixture.chunks(self,[events])[0])
            # The minimal context has no damage consumer, so the complete
            # marker must refuse, not silently erase metadata to force PASS.
            self.assertEqual(result.error.tolist(),[self.i.Error.PAYLOAD])
            moves,beliefs,mons=self.fixtures.PublicMovesTests.initial(self)
            for event in projection.events:
                if event.kind==self.i.Kind.MOVE:
                    update=self.p.record_move(moves,beliefs,mons,self.catalog.device,self.host.device,
                        side=event.actor_side,slot=event.actor_slot,encounter=event.payload[0],move_id=event.payload[1],
                        caller=event.payload[2],has_from=bool(event.payload[3]),transformed=False,opposing_pressure=False,
                        wire_target_side=event.target_side)
                    self.assertEqual(update.admitted.tolist(),[True])
                    moves,beliefs=update.state,update.beliefs
                elif event.kind==self.i.Kind.CONDITION:
                    mons,ok=self.m.condition_update(mons,side=event.target_side,slot=event.target_slot,
                        encounter=event.payload[0],numerator=event.payload[1],denominator=event.payload[2],
                        status=event.payload[3],fainted=bool(event.payload[4]))
                    self.assertEqual(ok.tolist(),[True])
                    self.assertEqual(int(mons.hp_numerator[0,event.target_side,0]),event.payload[1])
            self.assertEqual(int(moves.uses[0,0,0,0]),1)
            self.assertEqual(int(moves.uses[0,1,0,0]),1)

    def test_source_hp_percentage_nearfull_status_and_owner_exact(self):
        for hp,maximum,status,shared in ((999,1000,4,(99,100,6)),(1,1000,5,(1,100,1)),
                                       (1000,1000,0,(100,100,0)),(0,1000,4,(0,0,0))):
            tape=self.e.initialize(1,catalog=self.producer)
            tape=self.e.condition(tape,self.j.array([1]),self.j.array([5]),self.j.array([hp]),
                self.j.array([maximum]),self.j.array([status]),eligible=self.j.array([True]))
            peer=self.project(tape,0).events[0]
            own=self.project(tape,1).events[0]
            self.assertEqual(peer.payload[1:4],shared)
            self.assertEqual(own.payload[1:3],(hp,maximum) if hp else (0,0))

    def test_missing_active_identity_unknown_called_and_catalog_drift_refuse(self):
        from gpu_engine.public_context_protocol import PublicAddress
        tape=self.trace(self.state(),self.j.zeros((1,2),self.j.int32),
            chance_inputs=self.chance()).callbacks
        for registry in (self.e.PublicRegistry({}),self.e.PublicRegistry({0:(0,PublicAddress(0,0,1))})):
            with self.assertRaises(ValueError):self.project(tape,registry=registry)
        with self.assertRaises(ValueError):self.project(tape,True)
        counterfeit=replace(self.producer,ids=tuple(reversed(self.producer.ids)))
        with self.assertRaises(ValueError):self.e.check_catalog(counterfeit,self.engine_table,
            expected_binding=self.producer.binding,expected_engine_binding=self.j.array(self.producer.engine_binding,self.j.uint32))
        drift=self.engine_table._replace(power=self.engine_table.power.at[1].add(1))
        with self.assertRaises(ValueError):self.e.check_catalog(self.producer,drift,
            expected_binding=self.producer.binding,expected_engine_binding=self.j.array(self.producer.engine_binding,self.j.uint32))
        for move_id in (4,6):
            unsupported=self.e.move(self.e.initialize(1,catalog=self.producer),self.j.array([0]),self.j.array([3]),self.j.array([5]),
                self.j.array([move_id]),eligible=self.j.array([True]))
            with self.assertRaises(ValueError):self.project(unsupported)
        bad_binding=tape._replace(engine_binding=tape.engine_binding.at[0].add(self.j.uint32(1)))
        with self.assertRaisesRegex(ValueError,'namespace drift'):self.project(bad_binding)
        locked=self.e.move(self.e.initialize(1,catalog=self.producer),self.j.array([0]),self.j.array([3]),self.j.array([5]),
            self.j.array([1]),eligible=self.j.array([True]),unsupported=True)
        with self.assertRaises(ValueError):self.project(locked)

    def test_compiler_exact_source_identity_explicit_exposure_and_supported_rows(self):
        from pokezero.dex import ShowdownDex
        self.assertEqual(len(self.producer.ids),len(self.metadata['vocabulary']['moves']))
        self.assertEqual([bool(x) for x in self.producer.ids],[False,True,True,False,False,True,False])
        bad=ShowdownDex(dict(self.dex.moves,surf=self.dex.moves['bodyslam']),{}, {})
        with self.assertRaisesRegex(ValueError,'identity drift'):
            self.e.compile_catalog(self.metadata,self.engine_table,public_moves=self.catalog,dex=bad,exposure=self.exposure)
        for exposure in (replace(self.exposure,report_percentages=False),replace(self.exposure,report_exact_hp=True),
            replace(self.exposure,report_percentages=1),replace(self.exposure,showdown_revision='0'*40)):
            with self.assertRaises(ValueError):self.e.compile_catalog(self.metadata,self.engine_table,
                public_moves=self.catalog,dex=self.dex,exposure=exposure)
        metadata=deepcopy(self.metadata);metadata['moves'][1]['id']='surf'
        with self.assertRaises(ValueError):self.e.compile_catalog(metadata,self.engine_table,
            public_moves=self.catalog,dex=self.dex,exposure=self.exposure)

    def test_overflow_refuses_entire_projection_and_does_not_change_engine(self):
        state,actions,chance=self.state(),self.j.zeros((1,2),self.j.int32),self.chance()
        ordinary=self.t.step_with_chance(state,actions,self.engine_table,chance)
        result=self.trace(state,actions,chance_inputs=chance,capacity=1)
        self.same_state(ordinary.state,result.step.state)
        self.assertTrue(int(result.callbacks.refusal[0]) & self.e.OVERFLOW)
        with self.assertRaisesRegex(ValueError,'overflow'):self.project(result.callbacks)

    def test_jit_same_core_invalid_done_rows_no_receipts(self):
        state=self.state(2)._replace(done=self.j.array([False,True]))
        actions=self.j.array([[9,0],[-100,100]],self.j.int32)
        compiled=self.jax.jit(lambda s,a,c:self.trace(s,a,chance_inputs=c))
        traced=compiled(state,actions,self.chance(2))
        self.same_state(state,traced.step.state)
        self.assertEqual(traced.callbacks.count.tolist(),[0,0])
        self.assertEqual(traced.step.invalid_actions.tolist(),[True,False])
        self.assertTrue(int(traced.callbacks.refusal[0]) & self.e.INVALID_ACTION)
        # Reuse the same compiled function with live rows. Actual callbacks,
        # including nonzero counts and packed facts, must survive compilation.
        live,actions=self.state(2),self.j.zeros((2,2),self.j.int32)
        positive=compiled(live,actions,self.chance(2))
        ordinary=self.t.step_with_chance(live,actions,self.engine_table,self.chance(2))
        self.same_state(positive.step.state,ordinary.state)
        self.assertEqual(positive.callbacks.count.tolist(),[4,4])
        self.assertEqual(positive.callbacks.words[0,:4,0].tolist(),[1,2,1,2])


if __name__=='__main__':unittest.main()
