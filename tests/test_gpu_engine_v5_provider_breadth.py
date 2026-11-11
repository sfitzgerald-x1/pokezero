"""Bounded public head breadth/scatter; untrained tiny tensors only."""
import unittest
import jax.numpy as j
import numpy as np
from gpu_engine import v5_depth1_provider as p
from types import SimpleNamespace as NS
from contextlib import ExitStack
from dataclasses import replace
from unittest.mock import patch
from pokezero import v5_contract as c
from gpu_engine import v5_collection as collection,v5_collection_data as data,v5_observation
import test_gpu_engine_v5_depth1_provider as fixture_module


class HeadBreadthTests(unittest.TestCase):
    def test_padding_is_inactive_and_real_rows_preserve_original_order(self):
        count=7;seen=[]
        inputs=dict(numeric_features=j.zeros((count,1,29,132),j.float32).at[:,0,0,0].set(j.arange(count)/10),
            categorical_ids=j.zeros((count,1,29,41),j.int32),token_type_ids=j.zeros((count,1,29),j.int32),
            attention_mask=j.ones((count,1,29),bool),history_mask=j.ones((count,1),bool))
        def head(values):
            seen.append(np.asarray(values['history_mask']).tolist());b=len(values['history_mask'])
            return NS(policy_logits=j.zeros((b,9)),value=values['numeric_features'][:,0,0,0],valid=values['history_mask'][:,0])
        logits,values,valid,batches=p._predict({'forward':head},inputs,max_rows=4)
        np.testing.assert_allclose(values,np.arange(count)/10,atol=1e-7)
        self.assertEqual(logits.shape,(7,9));self.assertTrue(valid.all())
        self.assertEqual(batches,({'active_rows':4,'device_rows':4,'padding':0},{'active_rows':3,'device_rows':4,'padding':1}))
        self.assertEqual(seen[-1],[[True],[True],[True],[False]])
        with self.assertRaises(ValueError):p._predict({'forward':head},inputs,max_rows=3)


class FrameBreadthTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):fixture_module.ProviderTests.setUpClass.__func__(cls)
    setup=fixture_module.ProviderTests.setup

    def test_two_literal_roots_share_reply_forward_and_one_complete_unknown_grid(self):
        with ExitStack() as stack:
            provider,first,(learner,opponent),start=self.setup(stack)
            stream=next(n for n in range(first.address.stream+1,first.address.stream+100)
                if provider.recipe.selected(c.DecisionAddress(n,0,0)))
            # A second literal unit history is independently parsed and encoded
            # under its own stream; no sibling post-branch input is fabricated.
            history=replace(first.history,stream_id=stream,row=1)
            static=replace(first.own_static,stream=stream)
            captured=p.capture.public_root(history,config=self.f.config,contract=self.f.contract,
                battle_id='literal-second-root',seed=31)
            self.assertIsNotNone(captured.root,captured.decoded.refusal_counts)
            root=captured.root
            result=p.worlds.materialize(root,start,prepared=self.f.f.prepared,history=history,own_static=static,seed=5)
            self.assertIsNotNone(result.branch,result.refusals)
            branch=result.branch;current=p.branches.core(branch,self.f.runtime)
            _,encoded=self.f.runtime['encode_fast'](current,p.branches._observer(branch.public,0))
            appended=v5_observation.append_device(encoded,branch.public,learner.adapter,observer_sides=j.array([0],j.int32))
            _,observation=data.public_observation(encoded,appended,row=0,owner=0,address=root.address,adapter=learner.adapter,
                public_species=np.asarray(branch.public.beliefs.species)[0,0])
            logits,value=p._forward(collection._ACTORS[learner],v5_observation.actor_inputs(appended,learner.adapter))
            second=p.DecisionContext(root.address,'c'*64,observation,logits,value,'a'*64,'d'*64,
                history,static,'literal-second-root',31)
            grids=[];corrupt=[False]
            def unknown_grid(batch,adapter,*,max_rows):
                grids.append(batch)
                # This is a transport/refusal control, not an all-grid positive.
                cells=batch.cells
                if corrupt[0]:cells=(cells[1],cells[0],*cells[2:])
                yield p.GridChunk(cells,(),None,np.zeros(len(cells),bool),
                    (('explicit-unit-unknown',),)*len(batch.cells))
            stack.enter_context(patch.object(p,'_grid_executor',return_value=(unknown_grid,'UNIT_UNKNOWN_GRID')))
            sampler=stack.enter_context(patch.object(p.planner,'_gen3_randbat_belief_start_override_result',
                wraps=p.planner._gen3_randbat_belief_start_override_result))
            predict=stack.enter_context(patch.object(p,'_predict',wraps=p._predict))
            labels=provider.label_batch((first,second),learner=learner,opponents=(opponent,opponent))
            self.assertEqual(sampler.call_count,4);self.assertEqual(len(grids),1)
            batch=grids[0];self.assertEqual(len(batch.roots),4)
            self.assertEqual({cell.address for cell in batch.cells},{first.address,second.address})
            self.assertEqual(predict.call_count,1)
            self.assertEqual(predict.call_args.args[1]['history_mask'].shape[0],4)
            self.assertIs(predict.call_args.args[0],collection._ACTORS[opponent])
            for context,label in zip((first,second),labels,strict=True):
                report=label.report
                self.assertEqual(label.status,'REFUSED');self.assertIsNone(label.improvement)
                self.assertEqual(report['source_frame_sha256'],context.source_frame_sha256)
                self.assertTrue(report['required_pairs_known']);self.assertGreater(report['required_pairs'],0)
                self.assertEqual(report['unknown_pairs'],report['required_pairs'])
                self.assertEqual(report['completed_pairs'],0)
                self.assertEqual(report['execution']['reply_policy_batches'][0]['group_rows'],4)
                self.assertEqual(report['execution']['batch_declared_cells'],len(batch.cells))
                self.assertEqual(report['required_pairs'],sum(cell.address==context.address for cell in batch.cells))
            corrupt[0]=True
            rejected=provider.label_batch((first,second),learner=learner,opponents=(opponent,opponent))
            for label in rejected:
                self.assertEqual(label.status,'REFUSED');self.assertIsNone(label.improvement)
                self.assertIn('reordered',label.refusal)
                self.assertEqual(label.report['completed_pairs'],0)
                self.assertEqual(label.report['unknown_pairs'],label.report['known_required_pairs'])
