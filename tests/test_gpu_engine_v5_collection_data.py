"""Synthetic literal captures and original GAE/cache bridge, no rollout/model."""
from dataclasses import replace
import importlib.util
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np

from gpu_engine import v5_collection_data as data
from pokezero import v5_contract as c, v5_data
from pokezero.randbat import Gen3RandbatVariant
import test_v5_data as fixtures


def captured(frame=9,owner=0,value=.2,physical=(2,0,1,3,4,5,5,-1,-1,-1,-1,-1)):
    row=fixtures.row(frame-9)
    if owner:
        from pokezero.observation import ObservationPerspective
        old=replace(row.step.observation,perspective=ObservationPerspective.from_showdown_slot('p2','p2'))
        address=replace(row.observation.address,owner=owner)
        observation=replace(row.observation,address=address)
        row=replace(row,observation=observation,labels=replace(row.labels,address=address),
            step=replace(row.step,player_id='p2',observation=old))
    row=replace(row,step=replace(row.step,value_estimate=value),source_frame_sha256=hex(frame)[2:].rjust(64,'0'))
    return data.CapturedDecision(row,physical)


class CollectionCaptureTests(unittest.TestCase):
    def test_full_source_fields_hidden_power_and_all_duplicate_rows(self):
        variant=Gen3RandbatVariant('a','set','Swalot','role',80,
            ('Hidden Power Ice','Surf','Rest','Protect'),'Sticky Hold','Leftovers')
        rows=(variant,replace(variant,variant_id='duplicate',role='other'),replace(variant,item='Lum Berry'),
            replace(variant,moves=('Hidden Power Fire','Surf','Rest','Protect')))
        source=SimpleNamespace(species_id=lambda name:1 if name.lower()=='swalot' else 0,
            species_names=('','Swalot'),variants=((),rows))
        arguments=dict(level=80,moves=('surf','hiddenpower','protect','rest'),ability='stickyhold',
            item='leftovers',hidden_power_type='Ice',source=source)
        label=data.matching_hidden_set('Swalot',**arguments)
        self.assertEqual(label.variant_indices,(0,1))
        vocabulary=c.LabelVocabulary('a'*64,('surf',),(('swalot',4),))
        self.assertEqual(label.probabilities(vocabulary)[:4],(.5,.5,0,0))
        for change in (dict(level=81),dict(item=''),dict(ability='liquidooze'),dict(hidden_power_type=''),
                dict(moves=('surf','rest','protect')),
                dict(moves=('hiddenpowerice','surf','rest','protect'),hidden_power_type='Bug')):
            self.assertIsNone(data.matching_hidden_set('Swalot',**dict(arguments,**change)))

    def test_literal_terminal_targets_use_saved_input_physical_order_and_unknown_masks(self):
        row=captured()
        hp=((0,10,20,30,40,50),(60,70,80,90,100,0));maximum=((100,)*6,)*2
        terminal=data.TerminalCapture(100,12,'COMPLETE','f'*64,0,hp,maximum)
        result=data.finish((row,),(terminal,))[0].row
        self.assertEqual(result.labels.outcome,1)
        self.assertEqual(tuple(x.final_hp_fraction if x else None for x in result.labels.pokemon_end),
            (.2,0,.1,.3,.4,.5,0,None,None,None,None,None))
        self.assertFalse(result.labels.pokemon_end[1].survived)
        for status in ('CAPPED','REFUSED','FAILED'):
            unknown=data.finish((row,),(data.TerminalCapture(100,12,status,'e'*64),))[0].row
            self.assertIsNone(unknown.labels.outcome);self.assertEqual(unknown.labels.pokemon_end,(None,)*12)
        with self.assertRaises(ValueError):data.TerminalCapture(100,12,'CAPPED','e'*64,0,hp,maximum)
        with self.assertRaises(ValueError):data.TerminalCapture(100,12,'COMPLETE','e'*64,3,hp,maximum)
        with self.assertRaises(ValueError):data.finish((row,),(replace(terminal,frame=8),))

    def test_source_frame_owner_and_perspective_cannot_cross_join(self):
        row=captured()
        with self.assertRaises(ValueError):data.CapturedDecision(row.row,(0,0,1,2,3,4)+(-1,)*6)
        action=v5_data.semantic_opponent_action(row.row.observation.address,
            replace(row.row.observation.address,owner=1),kind='SWITCH',identity='swalot')
        self.assertEqual(action.identity,'swalot')
        with self.assertRaises(ValueError):v5_data.semantic_opponent_action(row.row.observation.address,
            replace(row.row.observation.address,owner=1,frame=10),kind='MOVE',identity='surf')


@unittest.skipUnless(importlib.util.find_spec('torch'),'requires optional Torch for original GAE')
class CollectionCacheTests(unittest.TestCase):
    def test_complete_original_gae_independent_value_lambda_and_cache_round_trip(self):
        from pokezero.v5_loss import V5LossConfig,gae_targets
        import torch
        rows=(captured(value=.2),captured(10,value=.4),captured(11,value=-.1))
        complete=data.finish(rows,(data.TerminalCapture(100,12,'COMPLETE','f'*64,0,((100,)*6,)*2,((100,)*6,)*2),))
        config=V5LossConfig(arm='B',policy_lambda=.8,value_lambda=.97)
        batches=tuple(data.cache_batches(complete,loss_config=config,batch_size=2))
        advantages,targets=gae_targets(torch.tensor([0.,0.,1.]),torch.tensor([.2,.4,-.1]),torch.tensor([.4,-.1,0.]),
            torch.tensor([False,False,True]),torch.tensor([True,True,False]),policy_lambda=.8,value_lambda=.97)
        np.testing.assert_allclose([x for b in batches for x in b.ppo_advantages],advantages)
        np.testing.assert_allclose([x for b in batches for x in b.ppo_value_targets],targets)
        self.assertTrue(all(x for b in batches for x in b.ppo_value_target_mask))
        self.assertFalse(any(x for b in batches for x in b.opponent_action_mask))
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'cache'
            data.write_cache(path,complete,loss_config=config,batch_size=2)
            restored=tuple(v5_data.iter_cache_batches(path,batch_size=2))
            self.assertEqual(sum(b.batch_size for b in restored),3)
            self.assertTrue(all(x for b in restored for x in b.v5_columns['outcome_target_mask']))

    def test_unknown_terminal_never_becomes_tie_and_last_critic_is_masked(self):
        from pokezero.v5_loss import V5LossConfig
        from pokezero.v5_training import validate_bootstraps
        for status in ('CAPPED','REFUSED'):
            rows=data.finish((captured(value=.2),captured(10,value=.4),captured(11,value=.6)),
                (data.TerminalCapture(100,12,status,'f'*64),))
            batch=next(data.cache_batches(rows,loss_config=V5LossConfig(arm='D'),batch_size=3))
            self.assertEqual(batch.ppo_value_target_mask,(True,True,False))
            self.assertEqual(batch.ppo_advantage_mask,(True,True,False))
            self.assertEqual(batch.v5_columns['outcome_target_mask'],(False,)*3)
            self.assertFalse(any(batch.search_value_target_mask))
            self.assertEqual(batch.step_metadata[0]['v5_bootstrap']['address']['frame'],11)
            self.assertEqual(batch.step_metadata[0]['v5_bootstrap']['value'],.6)
            if status=='CAPPED':validate_bootstraps(batch)
            single=next(data.cache_batches(rows[-1:],loss_config=V5LossConfig(arm='B'),batch_size=1))
            self.assertEqual(single.ppo_value_target_mask,(False,))

    def test_frame_order_and_nonfinite_actual_values_refuse_instead_of_targets(self):
        from pokezero.v5_loss import V5LossConfig
        row=captured()
        with self.assertRaises(ValueError):tuple(data.cache_batches((row,row),loss_config=V5LossConfig(),batch_size=2))
        with self.assertRaises(ValueError):data.capture_decision(row.row.step.observation,row.row.observation,
            battle_id='b',seed=3,turn=1,choice=0,probability=float('nan'),value=.2,
            physical=row.physical,hidden_sets=(None,)*6,source_frame_sha256='e'*64,capture_metadata={})

    def test_gae_requires_one_literal_trajectory_source_namespace_and_terminal(self):
        from pokezero.v5_loss import V5LossConfig
        from unittest.mock import patch
        first,second=data.finish((captured(),captured(10)),(data.TerminalCapture(100,12,'COMPLETE','e'*64,
            0,((100,)*6,)*2,((100,)*6,)*2),))
        different_source=replace(second.row.observation.label_vocabulary,source_binding_sha256='9'*64)
        observation=replace(second.row.observation,posteriors=tuple(replace(p,vocabulary=different_source)
            for p in second.row.observation.posteriors))
        for row in (replace(second.row,battle_id='other'),replace(second.row,seed=second.row.seed+1),
                replace(second.row,format_id='other'),replace(second.row,observation=observation),
                replace(second.row,observation=replace(second.row.observation,category_vocabulary_sha256='8'*64)),
                replace(second.row,terminal_source_sha256='f'*64)):
            with self.subTest(identity=(row.battle_id,row.seed,row.format_id)),patch('pokezero.v5_loss.gae_targets') as gae:
                with self.assertRaisesRegex(ValueError,'same original battle'):
                    tuple(data.cache_batches((first,replace(second,row=row)),loss_config=V5LossConfig(),batch_size=2))
                gae.assert_not_called()


if __name__=='__main__':unittest.main()
