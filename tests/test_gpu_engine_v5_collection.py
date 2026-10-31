"""Fake original-session steps around real retained public device fixtures.

No constructor/kernel game/model/checkpoint/native simulator is executed.
"""
from contextlib import ExitStack
from dataclasses import replace
import importlib.util
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

import numpy as np


@unittest.skipUnless(importlib.util.find_spec('jax'),'requires optional JAX')
class V5CollectionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import jax
        import jax.numpy as j
        import test_gpu_engine_v5_observation as fixtures
        from gpu_engine import v5_collection as collection
        from gpu_engine import engine_request_order as order, v5_model
        fixtures.V5ObservationTests.setUpClass()
        cls.f=fixtures.V5ObservationTests;cls.jax=jax;cls.j=j;cls.c=collection;cls.order=order;cls.model=v5_model

    def setup_runtime(self,stack,*,batch=1,done=True,failed_input=None,advance_error=False,choice=0):
        c,j=self.c,self.j
        def repeat(value):return j.repeat(value,batch,axis=0) if value.ndim and value.shape[0]==1 else value
        public=self.jax.tree.map(repeat,self.f.state)
        public=public._replace(cursor=public.cursor._replace(stream_id=j.arange(1,batch+1,dtype=j.int32)))
        reports=[];observers=[];requests=[];orders=[];originals=[]
        for owner in (0,1):
            encoded=self.jax.tree.map(repeat,self.f.encoded[owner])
            if failed_input is not None:
                encoded=encoded._replace(observation=encoded.observation._replace(
                    admitted=encoded.observation.admitted.at[failed_input].set(False)))
            reports.append(NS(encoded=encoded,policy_admitted=encoded.observation.admitted,
                basic_error=j.zeros(batch,j.int32),retained_error=j.zeros(batch,j.int32)))
            observers.append(NS(owner=j.full(batch,owner,j.int32),retained=public,retained_healthy=j.ones(batch,bool)))
            own=self.f.encoder.parse_own_requests(self.f.catalog,[self.f.fixture.request(owner)]*batch,owner_sides=[owner]*batch)
            packet=NS(facts=own.actions,admitted=j.ones(batch,bool))
            requests.append(NS(actions=packet))
            orders.append(NS(request_to_physical=j.broadcast_to(j.asarray([1,0,-1,-1,-1,-1],j.int32),(batch,6))))
            originals.append(NS(species=j.broadcast_to(j.asarray([1 if owner==0 else 2,3,1,1,1,1],j.int32),(batch,6))))
        engine=NS(done=j.zeros(batch,bool),winner=j.full(batch,-1,j.int32),turn=j.ones(batch,j.int32),
            hp=j.full((batch,2,6),100,j.int32),max_hp=j.full((batch,2,6),100,j.int32))
        core=NS(cursor=NS(stream_id=j.arange(1,batch+1,dtype=j.int32),physical_to_public=j.full((batch,2,6),-1,j.int32).at[:,:,0].set(0)),
            frame_number=j.zeros(batch,j.int32),engine=engine,requests=tuple(requests),orders=tuple(orders),
            originals=tuple(originals),admitted=j.ones(batch,bool))
        prepared=c.driver.Prepared(j.zeros(8,j.uint32))
        state=c.driver.DriverState(core,tuple(observers));result=c.driver.Result(state,tuple(reports))
        session=c.driver.Session(prepared,state,result)
        current=dict(prepared=prepared,state=state,result=result,constructed=object())
        runtime=dict(catalog=self.f.catalog,request={'metadata':'{"vocabulary":{"species":["","Snorlax","Absol","Gardevoir"]}}'})
        calls=NS(forward_inputs=[],advance=[],routes=[],diagnostics=None)
        def checked_session(actual):
            self.assertIs(actual,session);self.assertIs(actual.state,current['state']);self.assertIs(actual.result,current['result']);return current
        stack.enter_context(patch.object(c.driver,'_session',side_effect=checked_session))
        stack.enter_context(patch.object(c.driver,'_checked',return_value=runtime))
        stack.enter_context(patch.object(c.driver,'_checked_witness'))
        def fake_forward(params,inputs,config,**kwargs):
            calls.forward_inputs.append(set(inputs))
            count=inputs['history_mask'].shape[0]
            logits=j.full((count,9),-100,j.float32).at[:,choice].set(100)
            return self.model.Output(logits,j.full(count,.2,j.float32),None,inputs['history_mask'][:,0])
        stack.enter_context(patch.object(c.v5_model,'forward',side_effect=fake_forward))
        load=stack.enter_context(patch.object(c.v5_model,'load',return_value=({},self.f.adapter.model_config,
            {'weights_archive_sha256':'a'*64})))
        actor=c.load_actor(prepared,'/synthetic-only',expected_archive_sha256='a'*64,
            expected_checkpoint_sha256='b'*64,expected_category_vocabulary_sha256=self.f.adapter.model_config.category_vocabulary_sha256,
            expected_label_vocabulary_sha256=self.f.adapter.model_config.label_vocabulary.sha256,enable_v5=True)
        self.assertEqual(load.call_args.kwargs['expected_archive_sha256'],'a'*64)
        from pokezero.v5_contract import HiddenSet
        truth=tuple(((HiddenSet('snorlax',(0,)),)+(None,)*5,(HiddenSet('absol',(0,)),)+(None,)*5) for _ in range(batch))
        stack.enter_context(patch.object(c,'_pool_truth',return_value=(truth,{'pool_sha256':'f'*64})))
        def route(packet,ledger,choices):
            owner=next(i for i,request in enumerate(session.state.core.requests) if request.actions is packet)
            calls.routes.append((owner,packet,np.asarray(choices).copy()))
            # Request slot0 explicitly maps to physical move action2.
            engine_action=j.where(choices>=0,j.where(choices<4,2,4),-1)
            return self.order.Choice(engine_action,j.where(choices>=4,1,-1),choices>=0)
        stack.enter_context(patch.object(c.actions,'route_choice',side_effect=route))
        def advance(actual,engine_actions,*,active_rows):
            self.assertIs(actual,session)
            calls.advance.append((np.asarray(engine_actions).copy(),np.asarray(active_rows).copy()))
            if advance_error:raise ValueError('synthetic step failure')
            old=session.state.core;active=j.asarray(active_rows)
            hp=old.engine.hp.at[:,1].set(j.where(active[:,None]&done,0,old.engine.hp[:,1]))
            new_engine=NS(**dict(vars(old.engine),done=active&done,winner=j.where(active&done,0,-1),hp=hp))
            core=NS(**dict(vars(old),frame_number=old.frame_number+active.astype(j.int32),engine=new_engine))
            state=c.driver.DriverState(core,session.state.observers)
            result=c.driver.Result(state,session.result.owners)
            session.state=state;session.result=result;current.update(state=state,result=result)
            calls.diagnostics=c.driver.FrameDiagnostics(j.zeros(batch,bool),j.zeros(batch,j.int32),active,active)
            return result
        stack.enter_context(patch.object(c.driver,'advance',side_effect=advance))
        stack.enter_context(patch.object(c.driver,'latest_diagnostics',side_effect=lambda session:calls.diagnostics))
        cells=tuple(c.Cell(i+1,'root'+str(i),70+i,(2*i,2*i+1)) for i in range(batch+1))
        ledger=c.declare_cells(cells)
        return session,actor,ledger,calls,np.asarray([cell.team_indices for cell in cells[:batch]],np.int32)

    def test_same_packet_actor_route_two_owner_semantics_and_literal_terminal(self):
        with ExitStack() as stack:
            session,actor,ledger,calls,indices=self.setup_runtime(stack)
            out=self.c.collect_session(session,actor=actor,pool=object(),team_indices=indices,ledger=ledger,
                max_requests=1,sampling_seed=97,enable_v5=True,include_privileged=True)
            self.assertEqual(out.ledger['counts']['COMPLETE'],1);self.assertEqual(out.ledger['counts']['NOT_ATTEMPTED'],1)
            self.assertEqual(len(out.decisions),2)
            self.assertEqual([x.row.labels.outcome for x in out.decisions],[1,-1])
            for decision in out.decisions:
                self.assertEqual(decision.row.labels.opponent_next_action.identity,'bodyslam')
                self.assertIsNone(decision.row.privileged)  # unseen public slots have no invented correspondence
                self.assertEqual(decision.row.step.action_probability,1.)
                self.assertTrue(decision.row.labels.hidden_sets[0] is not None)
            self.assertTrue(all(keys==set(self.c.MODEL_INPUTS) for keys in calls.forward_inputs))
            np.testing.assert_array_equal(calls.advance[0][0],[[2,2]])
            self.assertEqual([x[0] for x in calls.routes],[0,1])
            import json
            from pokezero import v5_data
            from gpu_engine import v5_collection_data
            first=out.decisions[0]
            restored=v5_data.row_from_dict(json.loads(json.dumps(v5_data.row_to_dict(first.row))))
            round_trip=v5_collection_data.CapturedDecision(restored,first.physical)
            self.assertEqual(round_trip.physical,first.physical)
            self.assertEqual(json.loads(json.dumps(v5_data.row_to_dict(round_trip.row))),
                json.loads(json.dumps(v5_data.row_to_dict(first.row))))
            with self.assertRaises(ValueError):v5_collection_data.CapturedDecision(
                replace(first.row,step=replace(first.row.step,action_probability=.5)),first.physical)
            with self.assertRaises(ValueError):self.c.collect_session(session,actor=actor,pool=object(),team_indices=indices,
                ledger=ledger,max_requests=1,sampling_seed=97,enable_v5=True)
            with self.assertRaises(ValueError):self.c._ledger(replace(ledger))

    def test_switch_semantics_use_actual_physical_target_species(self):
        with ExitStack() as stack:
            session,actor,ledger,calls,indices=self.setup_runtime(stack,choice=4)
            out=self.c.collect_session(session,actor=actor,pool=object(),team_indices=indices,ledger=ledger,
                max_requests=1,sampling_seed=97,enable_v5=True)
            self.assertEqual(len(out.decisions),2)
            self.assertEqual([(d.row.labels.opponent_next_action.kind,d.row.labels.opponent_next_action.identity)
                for d in out.decisions],[('SWITCH','gardevoir')]*2)

    def test_cap_and_post_step_failure_keep_unknowns_and_not_attempted_cells(self):
        for failed in (False,True):
            with self.subTest(failed=failed),ExitStack() as stack:
                session,actor,ledger,calls,indices=self.setup_runtime(stack,done=False,advance_error=failed)
                out=self.c.collect_session(session,actor=actor,pool=object(),team_indices=indices,ledger=ledger,
                    max_requests=1,sampling_seed=97,enable_v5=True)
                self.assertEqual(out.ledger['counts']['FAILED' if failed else 'CAPPED'],1)
                self.assertEqual(out.ledger['counts']['NOT_ATTEMPTED'],1)
                self.assertEqual(len(out.decisions),0 if failed else 2)
                for d in out.decisions:
                    self.assertIsNone(d.row.labels.outcome);self.assertEqual(d.row.labels.pokemon_end,(None,)*12)

    def test_mixed_fixed_roster_input_refusal_preserves_other_root(self):
        with ExitStack() as stack:
            session,actor,ledger,calls,indices=self.setup_runtime(stack,batch=2,failed_input=1)
            out=self.c.collect_session(session,actor=actor,pool=object(),team_indices=indices,ledger=ledger,
                max_requests=1,sampling_seed=97,enable_v5=True)
            self.assertEqual(out.ledger['counts']['COMPLETE'],1)
            self.assertEqual(out.ledger['counts']['REFUSED'],1)
            self.assertEqual(out.ledger['counts']['NOT_ATTEMPTED'],1)
            self.assertEqual(len(out.decisions),2)
            np.testing.assert_array_equal(calls.advance[0][1],[True,False])

    def test_actual_constructor_preparation_failure_remains_attempted_failed(self):
        with ExitStack() as stack:
            session,actor,ledger,calls,indices=self.setup_runtime(stack)
            stack.enter_context(patch.object(self.c,'_pool_truth',side_effect=ValueError('different original pool')))
            out=self.c.collect_session(session,actor=actor,pool=object(),team_indices=indices,ledger=ledger,
                max_requests=1,sampling_seed=97,enable_v5=True)
            self.assertEqual(out.ledger['counts']['FAILED'],1);self.assertEqual(out.ledger['counts']['NOT_ATTEMPTED'],1)
            self.assertFalse(calls.advance);self.assertFalse(calls.forward_inputs)
            self.assertFalse(out.decisions)

    def test_one_unsupported_capture_refuses_only_that_physical_root(self):
        with ExitStack() as stack:
            session,actor,ledger,calls,indices=self.setup_runtime(stack,batch=2)
            original=self.c.data.public_observation
            def capture(*args,**kwargs):
                if kwargs['row']==1:raise ValueError('synthetic unsupported public row')
                return original(*args,**kwargs)
            stack.enter_context(patch.object(self.c.data,'public_observation',side_effect=capture))
            out=self.c.collect_session(session,actor=actor,pool=object(),team_indices=indices,ledger=ledger,
                max_requests=1,sampling_seed=97,enable_v5=True)
            self.assertEqual(out.ledger['counts']['COMPLETE'],1);self.assertEqual(out.ledger['counts']['REFUSED'],1)
            self.assertEqual(out.ledger['counts']['NOT_ATTEMPTED'],1)
            self.assertEqual(len(out.decisions),2);np.testing.assert_array_equal(calls.advance[0][1],[True,False])

    def test_literal_pool_selection_binding_and_birth_fields_without_engine_constructor(self):
        import json
        import test_gpu_engine_engine_private_actions as fixtures
        fixtures.EnginePrivateActionsTests.setUpClass()
        pool=fixtures.EnginePrivateActionsTests.pool
        indices=np.asarray([[0,1]],np.int32);streams=self.j.asarray([71],self.j.int32)
        registrations=tuple(NS(constructor=self.c.actions.retain_constructor_pp(pool,indices,stream_ids=streams,
            owner_sides=self.j.asarray([owner],self.j.int32))) for owner in (0,1))
        fields={actual:self.j.asarray(pool.arrays[source][indices],self.j.int32) for actual,source in (
            ('species','species'),('level','level'),('moves','moves'),('base_ability','ability'),('item','item'),
            ('hidden_power_type','hidden_power_type'),('hidden_power_power','hidden_power_power'))}
        birth=NS(**fields,hp=self.j.zeros((1,2,6),self.j.int32))
        constructed=NS(birth_state=birth,registrations=registrations)
        session=NS(state=NS(core=NS(cursor=NS(stream_id=streams))))
        runtime=dict(request={'metadata':json.dumps(pool.metadata(),sort_keys=True)},catalog=NS(source=self.f.catalog.source))
        with patch.object(self.c.driver.source,'_constructed',return_value=constructed),patch.object(self.c.driver.source,'_registered'):
            truth,receipt=self.c._pool_truth(pool,indices,session,runtime,{'constructed':constructed})
            self.assertEqual(receipt['pool_sha256'],json.loads(pool.receipt_json)['pool_sha256'])
            self.assertEqual(len(truth[0]),2)
            with self.assertRaisesRegex(ValueError,'binding'):
                self.c._pool_truth(pool,indices[:,::-1],session,runtime,{'constructed':constructed})
            birth.moves=birth.moves.at[0,0,0,0].add(1)
            with self.assertRaisesRegex(ValueError,'constructor set differs'):
                self.c._pool_truth(pool,indices,session,runtime,{'constructed':constructed})

    def test_enablement_before_archive_or_session_access_and_duplicate_declaration(self):
        with patch.object(self.c.driver,'_checked') as checked:
            with self.assertRaises(ValueError):self.c.load_actor(None,'/never-read',expected_archive_sha256='a'*64,
                expected_checkpoint_sha256='b'*64,expected_category_vocabulary_sha256='c'*64,expected_label_vocabulary_sha256='d'*64)
            checked.assert_not_called()
        cell=self.c.Cell(1,'root',7,(0,1))
        with self.assertRaises(ValueError):self.c.declare_cells((cell,cell))


if __name__=='__main__':unittest.main()
