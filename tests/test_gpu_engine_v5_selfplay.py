"""Paired fake sessions around original retained device observations only."""
from contextlib import ExitStack
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

import numpy as np

import test_gpu_engine_v5_collection as fixtures
from gpu_engine import v5_selfplay as s


class V5SelfplayTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        fixtures.V5CollectionTests.setUpClass();cls.fixture=fixtures.V5CollectionTests()
        cls.j=fixtures.V5CollectionTests.j;cls.base=fixtures.V5CollectionTests.c

    def setup(self,stack,**kwargs):
        session,learner,ledger,calls,indices=self.fixture.setup_runtime(stack,**kwargs)
        capture=[]
        def history(actual,*,row,owner,history=None):
            self.assertIs(actual,session)
            number=int(np.asarray(actual.state.core.frame_number)[row])
            capture.append((row,owner,number))
            frame=s.history_capture.Frame(number,(),(),(),None,(('fixture-no-FULL-source',1),))
            return s.history_capture.History((id(actual),id(actual.prepared),19),row,owner,
                int(np.asarray(actual.state.core.cursor.stream_id)[row]),self.fixture.f.catalog,(),
                (() if history is None else history.frames)+(frame,))
        stack.enter_context(patch.object(s.history_capture,'capture',side_effect=history))
        def own_static(actual,*,row,owner):
            self.assertIs(actual,session)
            self.assertEqual(int(np.asarray(actual.state.core.frame_number)[row]),0)
            return s.public_world.OwnStatic(int(np.asarray(actual.state.core.cursor.stream_id)[row]),
                owner,(row,owner,0,0,0,0,0,0),tuple(range(7+owner,13+owner)))
        stack.enter_context(patch.object(s.public_world,'capture_own_static',side_effect=own_static))
        return session,learner,ledger,calls,indices,capture

    def collect(self,session,learner,ledger,indices,pairs,**kwargs):
        ended=[]
        result=s.collect_paired(session,learner=learner,actors=pairs,pool=object(),team_indices=indices,
            ledger=ledger,max_requests=1,sampling_seed=97,on_end=lambda *args,**facts:ended.append((args,facts)),
            check_progress=lambda:None,**kwargs)
        return result,ended

    def test_historical_opponent_routes_same_joint_action_but_only_learner_behavior_is_trained(self):
        with ExitStack() as stack:
            session,learner,ledger,calls,indices,capture=self.setup(stack)
            seen=[]
            def forward(params,inputs,config,**kwargs):
                historical='historical' in params
                seen.append((historical,set(inputs),inputs['categorical_ids'].shape))
                count=inputs['history_mask'].shape[0]
                logits=self.j.full((count,9),-100,self.j.float32).at[:,4 if historical else 0].set(100)
                return self.fixture.model.Output(logits,self.j.full(count,.2,self.j.float32),None,inputs['history_mask'][:,0])
            stack.enter_context(patch.object(self.base.v5_model,'forward',side_effect=forward))
            stack.enter_context(patch.object(self.base.v5_model,'load',return_value=({'historical':self.j.ones(())},learner.config,
                {'weights_archive_sha256':'d'*64})))
            historical=self.base.load_actor(session.prepared,'/synthetic-history',expected_archive_sha256='d'*64,
                expected_checkpoint_sha256='e'*64,expected_category_vocabulary_sha256=learner.config.category_vocabulary_sha256,
                expected_label_vocabulary_sha256=learner.config.label_vocabulary.sha256,enable_v5=True)
            out,ended=self.collect(session,learner,ledger,indices,((learner,historical),))
            self.assertEqual(out.ledger['counts']['COMPLETE'],1);self.assertEqual(out.ledger['counts']['NOT_ATTEMPTED'],1)
            self.assertEqual(len(out.decisions),1);decision=out.decisions[0]
            self.assertEqual(decision.row.observation.address.owner,0)
            self.assertEqual((decision.row.labels.opponent_next_action.kind,decision.row.labels.opponent_next_action.identity),('SWITCH','gardevoir'))
            metadata=decision.row.step.metadata['v5_capture']
            self.assertEqual(metadata['actor_archive_sha256'],'a'*64);self.assertEqual(metadata['opponent_archive_sha256'],'d'*64)
            self.assertEqual(decision.row.step.action_probability,1.)
            self.assertIsNotNone(metadata['training_only_peer_own_capture'])
            self.assertTrue(all(keys==set(self.base.MODEL_INPUTS) for _,keys,_ in seen))
            self.assertEqual({x[0] for x in seen},{False,True})
            np.testing.assert_array_equal(calls.advance[0][0],[[2,4]])
            self.assertEqual(capture,[(0,0,0),(0,1,0),(0,0,1),(0,1,1)])
            self.assertEqual(len(ended),1);self.assertGreaterEqual(ended[0][1]['duration_seconds'],0)

    def test_selfplay_both_owned_rows_and_cap_keep_unknown_terminal_and_history(self):
        with ExitStack() as stack:
            session,learner,ledger,calls,indices,capture=self.setup(stack,done=False)
            out,ended=self.collect(session,learner,ledger,indices,((learner,learner),))
            self.assertEqual(len(out.decisions),2);self.assertEqual(out.ledger['counts']['CAPPED'],1)
            for decision in out.decisions:
                self.assertIsNone(decision.row.labels.outcome)
                self.assertEqual(decision.row.labels.pokemon_end,(None,)*12)
                metadata=decision.row.step.metadata['v5_capture']
                self.assertEqual(metadata['actor_archive_sha256'],metadata['opponent_archive_sha256'])
            self.assertEqual(len(out.owning_histories),2);self.assertFalse(out.history_refusals)
            self.assertEqual(len(out.own_statics),2);self.assertFalse(out.own_static_refusals)
            self.assertEqual(ended[0][0],(1,'CAPPED'))
            np.testing.assert_array_equal(calls.advance[0][0],[[2,2]])

    def test_missing_own_static_is_auxiliary_unknown_and_does_not_erase_history_or_actor_rows(self):
        with ExitStack() as stack:
            session,learner,ledger,calls,indices,capture=self.setup(stack)
            static=stack.enter_context(patch.object(s.public_world,'capture_own_static',side_effect=ValueError('missing-own-static')))
            out,ended=self.collect(session,learner,ledger,indices,((learner,learner),))
            self.assertEqual(static.call_count,2) # only the two original constructor0 seats
            self.assertEqual(len(out.decisions),2);self.assertEqual(out.ledger['counts']['COMPLETE'],1)
            self.assertEqual(len(out.owning_histories),2);self.assertFalse(out.history_refusals)
            self.assertFalse(out.own_statics);self.assertEqual(len(out.own_static_refusals),2)

    def test_mixed_root_refusal_no_refill_and_history_failure_does_not_gate_public_actor(self):
        with ExitStack() as stack:
            session,learner,ledger,calls,indices,capture=self.setup(stack,batch=2,failed_input=1)
            stack.enter_context(patch.object(s.history_capture,'capture',side_effect=ValueError('unsupported-history')))
            out,ended=self.collect(session,learner,ledger,indices,((learner,learner),)*2)
            self.assertEqual(out.ledger['counts']['COMPLETE'],1);self.assertEqual(out.ledger['counts']['REFUSED'],1)
            self.assertEqual(out.ledger['counts']['NOT_ATTEMPTED'],1);self.assertEqual(len(out.decisions),2)
            self.assertEqual(len(out.history_refusals),4);self.assertFalse(out.owning_histories)
            np.testing.assert_array_equal(calls.advance[0][1],[True,False])
            self.assertEqual({args[0] for args,_ in ended},{1,2})

    def test_next_declared_prefix_and_D_missing_dependency_fail_before_actor_or_advance(self):
        from dataclasses import replace
        for reversed_order in (False,True):
            with self.subTest(reversed_order=reversed_order),ExitStack() as stack:
                session,learner,ledger,calls,indices,capture=self.setup(stack,batch=2)
                if reversed_order:
                    session.state.core.cursor.stream_id=self.j.asarray([2,1],self.j.int32)
                else:
                    # The existing registry must still hold the exact original
                    # actor; a copied actor/config cannot stand in for D.
                    altered=replace(learner,config=replace(learner.config,arm='D'))
                    learner=altered
                with self.assertRaises((ValueError,s.MissingDependency)):
                    self.collect(session,learner,ledger,indices,((learner,learner),)*2)
                self.assertFalse(calls.forward_inputs);self.assertFalse(calls.advance);self.assertFalse(capture)

    def test_backend_prepares_before_constructor_and_commits_original_caches_and_owner_histories(self):
        from pokezero import v5_run as run, v5_training, v5_loss
        with ExitStack() as stack,tempfile.TemporaryDirectory() as directory:
            session,learner,ledger,calls,indices,capture=self.setup(stack)
            pool=object();backend=s.EngineBackend(session.prepared,pool,compute_dtype='float32')
            stack.enter_context(patch.object(backend,'check_plan'))
            stack.enter_context(patch.object(backend,'_actor',return_value=learner))
            constructed=stack.enter_context(patch.object(self.base.driver,'construct',return_value=session))
            snapshots=dict(archive_sha256='a'*64)
            plan=NS(model=learner.config,loss=v5_loss.V5LossConfig(arm='B'),train=v5_training.TrainConfig(12),max_requests=1,discount=.97)
            batch=run.Batch((run.Game(1,'root0',70,(0,1)),),101,97)
            class Sink:
                path=Path(directory)
                def __init__(self):self.artifacts=[];self.events=[]
                def artifact(self,path,blob,**facts):
                    self.artifacts.append((path,json.loads(blob),facts))
                def emit(self,kind,**facts):self.events.append((kind,facts))
            sink=Sink()
            prepared=backend.prepare_batch(plan,batch,snapshots,
                (dict(stream_id=1,learner_side=0,opponent=snapshots),),sink)
            constructed.assert_not_called();self.assertFalse(calls.forward_inputs)
            actual_cache=[]
            def write_cache(path,captured,**kwargs):
                self.assertEqual(kwargs['discount'],plan.discount)
                actual_cache.extend(captured);Path(path).mkdir(parents=True)
                (Path(path)/'metadata.json').write_text(json.dumps({'source_fixture_rows':len(captured)}))
            stack.enter_context(patch.object(s.data,'write_cache',side_effect=write_cache))
            ended=[];relative,digest=backend.collect(prepared,sink,lambda *a,**kw:ended.append((a,kw)),lambda:None)
            self.assertIs(constructed.call_args.args[0],pool)
            np.testing.assert_array_equal(constructed.call_args.args[1],indices)
            self.assertEqual(constructed.call_args.kwargs['seed'],101)
            np.testing.assert_array_equal(constructed.call_args.kwargs['stream_ids'],[1])
            self.assertEqual(len(actual_cache),2);self.assertEqual(len(ended),1)
            self.assertEqual(len(sink.artifacts),2)
            for _,payload,facts in sink.artifacts:
                self.assertFalse(payload['device_branch_ready']);self.assertEqual([f['number'] for f in payload['frames']],[0,1])
                self.assertEqual(payload['frames'][0]['refusal'],[['fixture-no-FULL-source',1]])
                self.assertNotIn('session',payload);self.assertNotIn('catalog',payload)
                self.assertEqual(payload['owner'],facts['owner'])
                self.assertEqual(payload['own_static']['owner'],facts['owner'])
                self.assertEqual(payload['own_static']['happiness'],list(range(7+facts['owner'],13+facts['owner'])))
                self.assertFalse(payload['own_static_refusals'])
            self.assertEqual(digest,s._file_sha(sink.path/relative/'metadata.json'))


if __name__=='__main__':unittest.main()
