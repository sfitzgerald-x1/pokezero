"""Tiny authored actual-core lifecycle fixtures; no games or qualification."""
from copy import deepcopy
from dataclasses import replace
import unittest
import numpy as np


class LifecycleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from test_gpu_engine_engine_public_events import EnginePublicEventsTests as fixture
        fixture.setUpClass.__func__(cls)
        from gpu_engine import engine_lifecycle_callbacks as life,entry
        cls.life,cls.fixture=life,fixture
        cls.metadata=deepcopy(cls.metadata)
        cls.metadata['vocabulary'].update(species=['_none','snorlax','absol','gardevoir'],
            abilities=['_none'],types=['_none','normal','fighting','flying','poison','ground','rock','bug',
                'ghost','steel','fire','water','grass','electric','psychic','ice','dragon','dark'])
        cls.engine_table=cls.engine_table._replace(entry=entry.from_metadata(cls.metadata))
        cls.producer=cls.e.compile_catalog(cls.metadata,cls.engine_table,public_moves=cls.catalog,
            dex=cls.dex,exposure=cls.exposure)

    def state(self,batch=1):
        state=self.fixture.state(self,batch)
        return state._replace(active=self.j.zeros((batch,2),self.j.int32),
            entry_pending=self.j.ones((batch,2),bool),gender=self.j.ones_like(state.gender),
            species=self.j.ones_like(state.species).at[:,1].set(2).at[:,0,1].set(3))

    def chance(self,batch=1):return self.fixture.chance(self,batch)
    def same(self,a,b):self.fixture.same_state(self,a,b)

    def kwargs(self,catalog=None):
        catalog=self.producer if catalog is None else catalog
        return dict(catalog=catalog,expected_binding=catalog.binding,
            expected_engine_binding=np.array(catalog.engine_binding,np.uint32))

    def initial(self,state=None,table=None,catalog=None,capacity=64):
        state=self.state() if state is None else state
        table=self.engine_table if table is None else table
        kwargs=self.kwargs(catalog)
        cursor=self.life.register(self.j.arange(1,state.hp.shape[0]+1,dtype=self.j.int32),**kwargs)
        return self.life.initialize_with_lifecycle(state,table,cursor=cursor,
            tie_coin=self.j.zeros(state.hp.shape[0],self.j.int32),capacity=capacity,**kwargs)

    def step(self,frame,actions,state=None,table=None,catalog=None,capacity=64):
        return self.life.trace_step(frame.state if state is None else state,self.j.array(actions,self.j.int32),
            self.engine_table if table is None else table,cursor=frame.cursor,
            chance_inputs=self.chance(frame.state.hp.shape[0]),capacity=capacity,**self.kwargs(catalog))

    def kinds(self,frame):return frame.callbacks.words[0,:int(frame.callbacks.count[0]),0].tolist()

    def order(self,frame,owner):
        from gpu_engine import engine_request_order as order
        return order.initialize_order(frame.cursor.stream_id,self.j.full_like(frame.cursor.stream_id,owner),
            self.j.full_like(frame.cursor.stream_id,6),engine_binding=frame.cursor.engine_binding)

    def test_initial_reveals_precede_same_legacy_initialize_and_turn1(self):
        state=self.state();frame=self.initial(state)
        legacy=self.t.initialize(state,self.engine_table,tie_coin=self.j.zeros(1,self.j.int32))
        self.same(frame.state,legacy)
        self.assertEqual(self.kinds(frame),[8,8,4])
        self.assertEqual(frame.callbacks.words[0,:2,3:5].tolist(),[[0,0],[1,0]])
        self.assertEqual(frame.callbacks.event_turn[0,:3].tolist(),[0,0,1])
        self.assertEqual(frame.callbacks.event_phase[0,:3].tolist(),[0,0,0])
        self.assertEqual(frame.cursor.generation.tolist(),[[0,0]])
        self.assertEqual(frame.cursor.next_slot.tolist(),[[1,1]])
        self.assertEqual(frame.after_registry.public_slot.tolist(),[[0,0]])
        self.assertEqual(frame.after_registry.encounter.tolist(),[[1,1]])
        self.assertEqual(frame.after_registry.revealed_species_id.tolist(),[[1,2]])
        self.assertEqual(frame.before_registry.established.tolist(),[[False,False]])
        self.assertEqual(frame.after_registry.established.tolist(),[[True,True]])
        self.assertEqual(frame.callbacks.switch_details[0,:2,:3].tolist(),[[1,100,1],[2,100,1]])
        for owner in (0,1):
            result=self.life.apply_installations(self.order(frame,owner),frame)
            self.assertEqual(result.generation.tolist(),[0])
            self.assertEqual(result.request_to_physical.tolist(),[[0,1,2,3,4,5]])

    def test_ordinary_real_install_revisit_and_request_order_use_encounter_order(self):
        frame=self.initial();order=self.order(frame,0)
        first=self.step(frame,[[4,0]])
        legacy=self.t.step_with_chance(frame.state,self.j.array([[4,0]],self.j.int32),self.engine_table,self.chance())
        self.same(first.state,legacy.state)
        self.assertEqual(self.kinds(first),[6,1,2,7,5,4])
        self.assertEqual(first.callbacks.words[0,0,1:5].tolist(),[0,0,0,1])
        self.assertEqual(first.callbacks.switch_details[0,0,3:5].tolist(),[100,100])
        self.assertEqual(first.before_registry.physical_active.tolist(),[[0,0]])
        self.assertEqual(first.after_registry.physical_active.tolist(),[[1,0]])
        self.assertEqual(first.after_registry.public_slot.tolist(),[[1,0]])
        self.assertEqual(first.after_registry.encounter.tolist(),[[2,1]])
        self.assertEqual(first.after_registry.generation.tolist(),[[1,0]])
        self.assertEqual(first.callbacks.event_generation[0,:2].tolist(),[[0,0],[1,0]])
        self.assertEqual(first.callbacks.event_turn[0,:6].tolist(),[1,1,1,1,1,2])
        self.assertEqual(first.callbacks.event_phase[0,:6].tolist(),[0,0,0,0,1,0])
        order=self.life.apply_installations(order,first)
        self.assertEqual(order.request_to_physical.tolist(),[[1,0,2,3,4,5]])
        self.assertEqual(order.generation.tolist(),[1])
        revisit=self.step(first,[[4,0]])
        self.assertEqual(revisit.after_registry.public_slot.tolist(),[[0,0]])
        self.assertEqual(revisit.cursor.next_slot.tolist(),[[2,1]])
        order=self.life.apply_installations(order,revisit)
        self.assertEqual(order.request_to_physical.tolist(),[[0,1,2,3,4,5]])
        self.assertEqual(order.generation.tolist(),[2])

    def test_forced_both_install_before_entries_and_no_extra_upkeep(self):
        frame=self.initial()
        cursor=frame.cursor._replace(phase=self.j.ones(1,self.j.int32))
        state=frame.state._replace(turn=self.j.ones(1,self.j.int32),hp=frame.state.hp.at[:,:,0].set(0))
        frame=frame._replace(state=state,cursor=cursor)
        result=self.step(frame,[[4,4]])
        self.assertEqual(self.kinds(result),[6,6,4])
        self.assertEqual(result.after_registry.generation.tolist(),[[1,1]])
        self.assertEqual(result.callbacks.event_turn[0,:3].tolist(),[1,1,2])
        self.assertEqual(result.callbacks.event_phase[0,:3].tolist(),[1,1,0])
        for owner in (0,1):
            order=self.life.apply_installations(self.order(frame,owner),result)
            self.assertEqual(order.generation.tolist(),[1]);self.assertEqual(order.admitted.tolist(),[True])

    def test_action_faint_pauses_residual_and_replacement_resumes_same_turn(self):
        metadata=deepcopy(self.metadata);metadata['moves'][1]['base_power']=1000
        table=self.engine_table._replace(power=self.engine_table.power.at[1].set(1000))
        catalog=self.e.compile_catalog(metadata,table,public_moves=self.catalog,dex=self.dex,exposure=self.exposure)
        frame=self.initial(table=table,catalog=catalog)
        knockout=self.step(frame,[[0,0]],table=table,catalog=catalog)
        self.assertEqual(self.kinds(knockout),[1,2,3])
        self.assertEqual(knockout.cursor.turn.tolist(),[1]);self.assertEqual(knockout.cursor.phase.tolist(),[0])
        self.assertEqual(knockout.state.pending_residual.tolist(),[True])
        resumed=self.step(knockout,[[0,4]],table=table,catalog=catalog)
        self.assertEqual(self.kinds(resumed),[6,7,5,4])
        self.assertEqual(resumed.callbacks.event_turn[0,:4].tolist(),[1,1,1,2])
        self.assertEqual(resumed.cursor.generation.tolist(),[[0,1]])
        self.assertEqual(resumed.state.turn.tolist(),[1])

    def test_upkeep_after_residual_and_post_residual_replacement_does_not_repeat_it(self):
        frame=self.initial();state=frame.state
        state=state._replace(hp=state.hp.at[0,0,0].set(1),status=state.status.at[0,0,0].set(4),
            moves=state.moves.at[0,1,0].set(3))
        residual=self.step(frame,[[0,0]],state=state)
        self.assertEqual(int(residual.state.hp[0,0,0]),0)
        self.assertEqual(self.kinds(residual)[-2:],[7,5])
        self.assertEqual(residual.cursor.turn.tolist(),[1]);self.assertEqual(residual.cursor.phase.tolist(),[1])
        replaced=self.step(residual,[[4,0]])
        self.assertEqual(self.kinds(replaced),[6,4])
        self.assertEqual(replaced.callbacks.event_phase[0,:2].tolist(),[1,0])
        # Terminal residual does not emit the source's conditional upkeep.
        state=state._replace(hp=state.hp.at[0,0,1:].set(0))
        terminal=self.step(frame,[[0,0]],state=state)
        self.assertTrue(bool(terminal.state.done[0]));self.assertNotIn(5,self.kinds(terminal))

    def test_hazard_cancelled_second_switch_and_invalid_incoming_emit_no_install(self):
        frame=self.initial();state=frame.state
        state=state._replace(spikes=state.spikes.at[0,0].set(3),hp=state.hp.at[0,0,1].set(1))
        cancelled=self.step(frame,[[4,4]],state=state)
        self.assertEqual(self.kinds(cancelled),[6])
        self.assertEqual(cancelled.callbacks.switch_details[0,0,3:5].tolist(),[1,100])
        self.assertEqual(cancelled.cursor.generation.tolist(),[[1,0]])
        self.assertEqual(cancelled.state.active.tolist(),[[1,0]])
        invalid=self.step(frame,[[4,0]],state=frame.state._replace(hp=frame.state.hp.at[0,0,1].set(0)))
        self.assertEqual(self.kinds(invalid),[]);self.assertEqual(invalid.invalid_actions.tolist(),[True])
        self.assertEqual(invalid.cursor.generation.tolist(),[[0,0]])

    def test_gen3_pursuit_outgoing_ko_still_records_actual_installation(self):
        from gpu_engine import entry,pursuit
        from pokezero.dex import ShowdownDex
        metadata=deepcopy(self.metadata)
        metadata['vocabulary']['moves'].append('pursuit')
        metadata['moves'].append(dict(id='pursuit',base_power=40,type=17,category='Special'))
        n=len(metadata['moves'])
        table=self.engine_table._replace(power=self.j.array([0]+[r['base_power'] for r in metadata['moves'][1:]],self.j.int32),
            type=self.j.array([0]+[r['type'] for r in metadata['moves'][1:]],self.j.int32),
            physical=self.j.array([False]+[r['category']=='Physical' for r in metadata['moves'][1:]]),
            accuracy=self.j.full((n,),100,self.j.int32),always_hit=self.j.ones((n,),bool),priority=self.j.zeros((n,),self.j.int32),
            entry=entry.from_metadata(metadata),pursuit=pursuit.from_metadata(metadata))
        dex=ShowdownDex(dict(self.dex.moves,pursuit=replace(self.dex.moves['surf'],id='pursuit',name='Pursuit',
            type='Dark',category='Special',gen3_category='Special',base_power=40)),{}, {})
        public=self.p.compile_moves(self.host,dex,self.vocab)
        catalog=self.e.compile_catalog(metadata,table,public_moves=public,dex=dex,exposure=self.exposure)
        frame=self.initial(table=table,catalog=catalog)
        state=frame.state._replace(moves=frame.state.moves.at[0,0,0,0].set(n-1),hp=frame.state.hp.at[0,1,0].set(1))
        result=self.step(frame,[[0,4]],state=state,table=table,catalog=catalog)
        self.assertEqual(int(result.state.hp[0,1,0]),0)
        self.assertEqual(int(result.state.active[0,1]),1)
        self.assertEqual(self.kinds(result)[0],6)
        self.assertEqual(result.cursor.generation.tolist(),[[0,1]])
        self.assertEqual(result.callbacks.words[0,0,1:5].tolist(),[1,0,1,1])
        # Same publicly revealed species, different actual individual/encounter.
        self.assertEqual(result.after_registry.revealed_species_id.tolist(),[[1,2]])
        self.assertEqual(result.after_registry.encounter.tolist(),[[1,2]])

    def test_unknown_gender_baton_and_overflow_never_establish_false_complete_context(self):
        unknown=self.initial(self.state()._replace(gender=self.j.full_like(self.state().gender,-1)))
        self.assertTrue(int(unknown.callbacks.refusal[0])&self.life.UNKNOWN_DETAILS)
        self.assertEqual(unknown.callbacks.switch_details[0,:2,2].tolist(),[-1,-1])
        frame=self.initial();state=frame.state._replace(baton_pass=frame.state.baton_pass.at[0,0].set(True),
            forced_switch=frame.state.forced_switch.at[0,0].set(True),turn=self.j.ones(1,self.j.int32),
            pending_residual=self.j.ones(1,bool))
        baton=self.step(frame,[[4,0]],state=state)
        self.assertEqual(baton.cursor.generation.tolist(),[[1,0]])
        self.assertEqual(baton.after_registry.established.tolist(),[[False,True]])
        self.assertEqual(baton.cursor.public_slot.tolist(),[[1,0]])
        self.assertTrue(int(baton.callbacks.refusal[0])&self.life.UNKNOWN_BATON)
        overflow=self.step(frame,[[4,0]],capacity=1)
        self.assertTrue(int(overflow.callbacks.refusal[0])&self.e.OVERFLOW)
        self.assertEqual(overflow.cursor.admitted.tolist(),[False])
        self.assertEqual(self.life.apply_installations(self.order(frame,0),overflow).admitted.tolist(),[False])

    def test_positive_jit_and_old_callback_default_path_unchanged(self):
        frame=self.initial();actions=self.j.array([[4,0]],self.j.int32)
        compiled=self.jax.jit(lambda s,c:self.life.trace_step(s,actions,self.engine_table,cursor=c,
            chance_inputs=self.chance(),**self.kwargs()))
        result=compiled(frame.state,frame.cursor)
        self.assertEqual(self.kinds(result),[6,1,2,7,5,4])
        ordinary=self.t.step_with_chance(frame.state,actions,self.engine_table,self.chance())
        self.same(ordinary.state,result.state)
        old=self.e.trace_step(frame.state,actions,self.engine_table,chance_inputs=self.chance(),**self.kwargs())
        self.assertEqual(old.callbacks.words[0,:int(old.callbacks.count[0]),0].tolist(),[1,2])
        self.assertEqual(len(old.step),3)

    def test_stale_missing_install_reinitialize_and_mutable_context_refuse(self):
        frame=self.initial()
        stale=self.step(frame,[[0,0]],state=frame.state._replace(active=frame.state.active.at[0,0].set(1)))
        self.assertEqual(stale.cursor.admitted.tolist(),[False])
        self.assertEqual(stale.cursor.generation.tolist(),[[0,0]])
        self.assertEqual(stale.cursor.physical_active.tolist(),[[0,0]])
        self.assertEqual(stale.after_registry.established.tolist(),[[False,False]])
        self.assertTrue(int(stale.callbacks.refusal[0])&self.life.MISSING_INSTALL)
        repeated=self.life.initialize_with_lifecycle(frame.state,self.engine_table,cursor=frame.cursor,**self.kwargs())
        self.assertEqual(repeated.callbacks.count.tolist(),[0]);self.assertEqual(repeated.cursor.admitted.tolist(),[False])
        self.same(repeated.state,frame.state)
        mutable=frame.cursor._replace(turn=np.array(frame.cursor.turn))
        with self.assertRaises(ValueError):self.life.trace_step(frame.state,self.j.zeros((1,2),self.j.int32),
            self.engine_table,cursor=mutable,chance_inputs=self.chance(),**self.kwargs())

    def test_invalid_and_terminal_rows_do_not_install_or_advance_clocks(self):
        frame=self.initial(self.state(2))
        state=frame.state._replace(done=self.j.array([False,True]))
        result=self.step(frame,[[9,0],[4,4]],state=state)
        self.assertEqual(result.callbacks.count.tolist(),[0,0])
        self.assertEqual(result.cursor.generation.tolist(),[[0,0],[0,0]])
        self.assertEqual(result.cursor.turn.tolist(),[1,1])
        self.assertEqual(result.invalid_actions.tolist(),[True,False])
        self.same(result.state,state)


if __name__=='__main__':unittest.main()
