"""Actual tiny public sampler/kernel; fake untrained heads and fault cells only.

The whole grid is retained. Only one numerical cell is executed in the bounded
unit test; all remaining cells are explicitly injected refusals, never invented
successful sibling observations. This cannot qualify a full D game/provider.
"""
from contextlib import ExitStack
from dataclasses import replace
import unittest
from unittest.mock import patch

import jax
import jax.numpy as j
import numpy as np
from pokezero import v5_contract as c,v5_gumbel as g
from pokezero.randbat import Gen3RandbatSource
from pokezero.v5_model import V5ModelConfig
from gpu_engine import v5_depth1_provider as p,v5_collection as collection,v5_collection_data as data
from gpu_engine import v5_observation,private_team,public_field
import test_gpu_engine_v5_public_world as fixtures


class GridTransportTests(unittest.TestCase):
    def test_unknown_cells_have_no_model_rows_and_scatter_cannot_relabel_them(self):
        cell=p.GridCell(c.DecisionAddress(1,0,0),0,0,0,0)
        p.GridChunk((cell,),(),None,np.array([False]),(('unknown',),))
        with self.assertRaises(ValueError):p.GridChunk((cell,),(0,),None,np.array([False]),(('unknown',),))
        with self.assertRaises(ValueError):p.GridChunk((cell,),(),{'view':'PUBLIC'},np.array([False]),(('unknown',),))
        with self.assertRaises(ValueError):p.GridChunk((cell,),(0,),{'view':'PRIVILEGED'},np.array([True]),((),))
        with self.assertRaises(ValueError):p.GridChunk((cell,),(),None,np.array([False]),(('unknown',),),device_rows=8,active_rows=1,padding=8)

    def test_ordinary_objects_and_ambiguous_action_ids_cannot_be_hypothetical_grid_roots(self):
        address=c.DecisionAddress(1,0,0)
        for changes in (dict(world=True),dict(root_index=-1),dict(root_action=9),dict(reply_action=False)):
            fields=dict(address=address,world=0,root_index=0,root_action=0,reply_action=0);fields.update(changes)
            with self.assertRaises(ValueError):p.GridCell(**fields)
        cell=p.GridCell(address,0,0,0,0)
        with self.assertRaises(ValueError):p.GridBatch((object(),),(None,),(cell,))


class ProviderTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        fixtures.PublicWorldTests.setUpClass();cls.f=fixtures.PublicWorldTests()
        old=cls.f.source;dex=cls.f.catalog.dex
        # The original sampler validates known stats against source metadata.
        # Complete the tiny authored metadata, rather than omit/ignore stats.
        cls.f.__class__.source=Gen3RandbatSource(old.metadata,old.universes,
            move_metadata={name:dict(type=dex.move_info(name).type,category=dex.move_info(name).category)
                for name in cls.f.f.names},
            species_metadata={name:dict(baseStats=dict(dex.species_info(name).base_stats)) for name in old.universes})
        census=c.source_variant_census(cls.f.source)
        labels=c.LabelVocabulary(census['source_rows_sha256'],tuple(cls.f.catalog.moves.names[1:]),census['species_variants'])
        category=np.asarray(private_team.vocabulary_binding(cls.f.catalog.vocabulary),dtype='>u4').tobytes().hex()
        cls.config=V5ModelConfig(cls.f.catalog.vocabulary.size,category,labels,arm='D',width=8,layers=1,
            attention_heads=2,expert_width=8,value_hidden=(8,8))

    def setup(self,stack):
        # Known literal own L79 versus source-sampled L80 avoids a speed tie
        # without changing either side's original IV/EV recipe or source axis.
        f=self.f;root,start,history,static=f.fixture(own_level=79)
        stack.enter_context(patch('subprocess.Popen',side_effect=AssertionError('no Native/simulator')))
        stack.enter_context(patch('pokezero.local_showdown.load_gen3_randbat_source_cached',return_value=f.source))
        stack.enter_context(patch('pokezero.local_showdown.load_showdown_dex_cached',return_value=f.catalog.dex))
        turn_column=public_field.NUMERIC_COLUMNS[1]
        def head(params,inputs,config,**kwargs):
            b=inputs['history_mask'].shape[0];peer='peer' in params
            logits=j.zeros((b,9),j.float32).at[:,1].set(-10000 if peer else 0)
            if not peer:logits=logits.at[:,0].set(100)
            value=j.tanh(10*inputs['numeric_features'][:,0,0,turn_column])
            return collection.v5_model.Output(logits,value,None,inputs['history_mask'][:,0])
        stack.enter_context(patch.object(collection.v5_model,'forward',side_effect=head))
        actors=[]
        for peer,digest in ((False,'a'*64),(True,'d'*64)):
            with patch.object(collection.v5_model,'load',return_value=({'peer' if peer else 'learner':j.ones(())},
                    self.config,{'weights_archive_sha256':digest})):
                actors.append(collection.load_actor(f.f.prepared,'/untrained-unit-only',expected_archive_sha256=digest,
                    expected_checkpoint_sha256='e'*64,expected_category_vocabulary_sha256=self.config.category_vocabulary_sha256,
                    expected_label_vocabulary_sha256=self.config.label_vocabulary.sha256,enable_v5=True))
        result=p.worlds.materialize(root,start,prepared=f.f.prepared,history=history,own_static=static,seed=5)
        self.assertIsNotNone(result.branch,result.refusals);branch=result.branch
        current=p.branches.core(branch,f.runtime)
        _,encoded=f.runtime['encode_fast'](current,p.branches._observer(branch.public,0))
        appended=v5_observation.append_device(encoded,branch.public,actors[0].adapter,observer_sides=j.array([0],j.int32))
        _,observation=data.public_observation(encoded,appended,row=0,owner=0,address=root.address,adapter=actors[0].adapter,
            public_species=np.asarray(branch.public.beliefs.species)[0,0])
        view=v5_observation.actor_inputs(appended,actors[0].adapter)
        logits,value=p._forward(collection._ACTORS[actors[0]],view)
        context=p.DecisionContext(root.address,'b'*64,observation,logits,value,'a'*64,'d'*64,history,static,'literal-provider',31)
        # Selection is a deterministic unit address control, never an eval-bank
        # draw or outcome-dependent retry.
        seed=next(i for i in range(64) if g.Recipe(i).selected(root.address))
        provider=p.Provider(f.f.prepared,f.config,f.contract,g.Recipe(seed))
        return provider,context,actors,start

    def test_original_two_draws_full_reply_grid_one_real_cell_all_other_refusals_retained(self):
        with ExitStack() as stack:
            provider,context,(learner,opponent),start=self.setup(stack)
            original_step=p.branches.step;calls=[]
            def bounded(branch,choices,**kwargs):
                calls.append((branch,choices))
                if len(calls)==1:return original_step(branch,choices,**kwargs)
                return p.branches.Step(None,(),(),('bounded-unit-cell-refusal',),None)
            stack.enter_context(patch.object(p.branches,'step',side_effect=bounded))
            sampler=stack.enter_context(patch.object(p.planner,'_gen3_randbat_belief_start_override_result',
                wraps=p.planner._gen3_randbat_belief_start_override_result))
            materialize=stack.enter_context(patch.object(p.worlds,'materialize',wraps=p.worlds.materialize))
            stack.enter_context(patch.object(collection.driver,'_session',side_effect=AssertionError('no live/peer Session')))
            result=provider.label(context,learner=learner,opponent=opponent)
            self.assertEqual(sampler.call_count,2)
            self.assertEqual(materialize.call_count,2)
            self.assertIs(materialize.call_args_list[0].kwargs['retained'],materialize.call_args_list[1].kwargs['retained'])
            self.assertEqual(result.status,'REFUSED');self.assertIsNone(result.improvement)
            report=result.report;self.assertTrue(report['required_pairs_known'])
            required=sum(slot['required_pairs'] for slot in report['world_slots'])
            self.assertEqual(len(calls),required);self.assertEqual(report['required_pairs'],required)
            self.assertEqual(report['completed_pairs'],1)
            self.assertEqual(report['unknown_pairs'],required-1)
            # Reply1 has numerically underflowed mass and STILL was attempted.
            self.assertTrue(any(choices[1]==1 for _,choices in calls))
            for slot in report['world_slots']:
                self.assertEqual(slot['required_pairs'],report['root_actions']*sum(slot['reply_legal']))
                self.assertEqual(len(slot['joint_values']),9)
            first=report['world_slots'][0]['joint_values'][0][0]
            self.assertIsNotNone(first)
            self.assertNotEqual(first,context.raw_public_value) # CURRENT branch public value head, never root substitute

    def test_missing_world_has_unknown_denominator_no_refill_or_survivor_label(self):
        with ExitStack() as stack:
            provider,context,actors,start=self.setup(stack)
            plans=p.planner.StartPlans((None,None),('missing-clock','missing-clock'))
            sampled=stack.enter_context(patch.object(p.planner,'plan_starts',return_value=plans))
            step=stack.enter_context(patch.object(p.branches,'step',side_effect=AssertionError('no invented worlds')))
            result=provider.label(context,learner=actors[0],opponent=actors[1])
            self.assertEqual(sampled.call_count,1);step.assert_not_called()
            self.assertEqual(result.status,'REFUSED');self.assertIsNone(result.improvement)
            self.assertEqual(result.report['unknown_worlds'],2)
            self.assertFalse(result.report['required_pairs_known']);self.assertIsNone(result.report['required_pairs'])

    def test_unselected_no_history_or_private_peer_and_archive_fault_cannot_make_label(self):
        with ExitStack() as stack:
            provider,context,actors,start=self.setup(stack)
            seed=next(i for i in range(64) if not g.Recipe(i).selected(context.address))
            unselected=p.Provider(self.f.f.prepared,self.f.config,self.f.contract,g.Recipe(seed))
            sampler=stack.enter_context(patch.object(p.planner,'plan_starts',side_effect=AssertionError('no search')))
            result=unselected.label(replace(context,history=None,own_static=None),learner=actors[0],opponent=actors[1])
            self.assertEqual(result.status,'UNSELECTED');sampler.assert_not_called()
            result=provider.label(replace(context,opponent_archive_sha256='f'*64),learner=actors[0],opponent=actors[1])
            self.assertEqual(result.status,'REFUSED');self.assertIn('archive binding',result.refusal)
