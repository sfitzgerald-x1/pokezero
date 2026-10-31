"""Teacher/collector contract joins on existing fake sessions, not D physics.

Typed label injection tests only downstream behavior/mixing/zero-label stop.
The real producer is separately exercised with original tiny sampler/kernel.
"""
from contextlib import ExitStack
from dataclasses import replace
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

import numpy as np
from pokezero import v5_contract as c,v5_gumbel as g
from pokezero.live_protocol_actor import ActorContract,_vocab_sha,_mask_values
from pokezero.local_showdown import LocalShowdownConfig
from pokezero.observation import ObservationFeatureMasks
from pokezero.showdown import V4_REPLAY_OBSERVATION_SPEC
from gpu_engine import v5_selfplay as s,v5_depth1_provider as p
import test_gpu_engine_v5_selfplay as fixtures


class TeacherCollectionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        fixtures.V5SelfplayTests.setUpClass();cls.f=fixtures.V5SelfplayTests()

    def setup(self,stack):
        session,old,ledger,calls,indices,history=self.f.setup(stack)
        config=replace(old.config,arm='D')
        with patch.object(s.base.v5_model,'load',return_value=({},config,{'weights_archive_sha256':'a'*64})):
            learner=s.base.load_actor(session.prepared,'/untrained-contract-only',expected_archive_sha256='a'*64,
                expected_checkpoint_sha256='b'*64,expected_category_vocabulary_sha256=config.category_vocabulary_sha256,
                expected_label_vocabulary_sha256=config.label_vocabulary.sha256,enable_v5=True)
        catalog=s.base.driver._checked(session.prepared)['catalog'];masks=dict(catalog.masks.__dict__)
        masks['opponent_tendency_stats_block']=masks.pop('stats_block')
        public_config=LocalShowdownConfig(showdown_root='/literal-source-only',set_belief_source=True,
            category_vocab=catalog.vocabulary,observation_spec=V4_REPLAY_OBSERVATION_SPEC,
            feature_masks=ObservationFeatureMasks(**masks))
        contract=replace(ActorContract.champion(),category_vocab_sha256=_vocab_sha(catalog.vocabulary),
            categorical_vocab_size=catalog.vocabulary.size,belief_set_source_hash=catalog.source.source_hash,
            feature_masks=_mask_values(public_config.feature_masks))
        seed=next(i for i in range(64) if g.Recipe(i).selected(c.DecisionAddress(1,0,0)))
        teacher=p.Provider(session.prepared,public_config,contract,g.Recipe(seed))
        metadata=dict(producer_sha256=teacher.producer_sha256,execution_mode='SCALAR_CORRECTNESS_COMPONENT_NOT_PRODUCTION_READY')
        stack.enter_context(patch.object(p.Provider,'require_ready',return_value=metadata))
        return session,learner,ledger,calls,indices,teacher

    def report(self,teacher,context):
        return dict(source_frame_sha256=context.source_frame_sha256,selected=teacher.recipe.selected(context.address),
            address=dict(stream=context.address.stream,frame=context.address.frame,owner=context.address.owner),
            required_pairs=None,unknown_worlds=2)

    def test_typed_teacher_label_preserves_raw_actions_and_mixes_literal_outcome_once(self):
        with ExitStack() as stack:
            session,learner,ledger,calls,indices,teacher=self.setup(stack);seen=[]
            def label(context,**actors):
                seen.append(context)
                self.assertIs(type(context),p.DecisionContext)
                self.assertIs(actors['learner'],learner);self.assertIs(actors['opponent'],learner)
                self.assertFalse(any(hasattr(context,name) for name in ('session','core','pool','peer_capture','physical')))
                self.assertEqual(context.history.owner,context.address.owner)
                report=self.report(teacher,context)
                if not report['selected']:return p.Label('UNSELECTED',None,None,report)
                legal=context.public_observation.legal_action_mask;count=sum(legal)
                improvement=c.RootImprovement(tuple(float(active)/count for active in legal),.25,
                    tuple(.25 if active else None for active in legal),worlds=2)
                return p.Label('LABELED',improvement,None,report)
            stack.enter_context(patch.object(p.Provider,'label',side_effect=label))
            stack.enter_context(patch.object(p.Provider,'require_batched',return_value=dict(source='future-batch-contract-join-only')))
            out,ended=self.f.collect(session,learner,ledger,indices,((learner,learner),),teacher=teacher)
            self.assertEqual(len(out.decisions),2);self.assertEqual(len(seen),2)
            np.testing.assert_array_equal(calls.advance[0][0],[[2,2]]) # unchanged raw actor routes
            self.assertEqual(out.ledger['counts']['COMPLETE'],1)
            self.assertTrue(all(x['retained_training_decision'] for x in out.teacher_reports))
            first=out.decisions[0].row
            self.assertEqual(first.labels.search_status,'LABELED')
            self.assertEqual(first.labels.outcome,1)
            self.assertEqual(first.step.value_estimate,float(np.float32(.2)))
            self.assertEqual(first.step.action_probability,1.)
            self.assertEqual(first.labels.columns(first.observation)['search_value_target'],.625)

    def test_missing_batched_source_refuses_before_constructor_and_zero_labels_never_write_PPO_cache(self):
        from pokezero import v5_run as run,v5_training,v5_loss
        with ExitStack() as stack,tempfile.TemporaryDirectory() as directory:
            session,learner,ledger,calls,indices,teacher=self.setup(stack)
            backend=s.EngineBackend(session.prepared,object(),teacher=teacher,compute_dtype='float32')
            with self.assertRaises(s.MissingDependency):backend.require_teacher()
            self.assertFalse(calls.advance)
            def label(context,**actors):
                report=self.report(teacher,context)
                return p.Label('REFUSED' if report['selected'] else 'UNSELECTED',None,
                    'unknown-required-world' if report['selected'] else None,report)
            stack.enter_context(patch.object(p.Provider,'label',side_effect=label))
            # This cache-stop control models a future connected batch source;
            # it does NOT promote today's scalar component to plan readiness.
            stack.enter_context(patch.object(backend,'require_teacher',return_value=dict(source='unit-join-only')))
            stack.enter_context(patch.object(p.Provider,'require_batched',return_value=dict(source='future-batch-contract-join-only')))
            stack.enter_context(patch.object(backend,'check_plan'))
            stack.enter_context(patch.object(s.base.driver,'construct',return_value=session))
            cache=stack.enter_context(patch.object(s.data,'write_cache'))
            class Sink:
                path=Path(directory)
                def __init__(self):self.artifacts=[];self.events=[]
                def artifact(self,path,blob,**facts):self.artifacts.append((path,json.loads(blob),facts))
                def emit(self,kind,**facts):self.events.append((kind,facts))
            sink=Sink();plan=NS(model=learner.config,loss=v5_loss.V5LossConfig(arm='D'),
                train=v5_training.TrainConfig(12),max_requests=1)
            prepared=dict(plan=plan,batch=run.Batch((run.Game(1,'root0',70,(0,1)),),101,97),learner=learner,
                pairs=((learner,learner),),ledger=ledger,indices=indices,streams=self.f.j.array([1],self.f.j.int32),
                gender=None,teacher=teacher)
            ended=[]
            with self.assertRaisesRegex(s.MissingDependency,'zero usable teacher labels'):
                backend.collect(prepared,sink,lambda *a,**kw:ended.append((a,kw)),lambda:None)
            cache.assert_not_called();self.assertEqual(len(ended),1)
            coverage=[value for _,value,facts in sink.artifacts if facts['kind']=='D_TEACHER_COVERAGE']
            self.assertEqual(len(coverage),1);self.assertEqual(coverage[0]['coverage']['retained_labeled'],0)
            self.assertEqual(coverage[0]['coverage']['eligible'],2)
            self.assertEqual(len(coverage[0]['decisions']),2)
            self.assertEqual(coverage[0]['game_ledger']['counts']['COMPLETE'],1)
