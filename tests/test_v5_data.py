"""Tiny synthetic producer/cache/batch roundtrips; no model, collection or worlds."""
from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from pokezero import dataset as d, v5_contract as c, v5_data as data
from pokezero.trajectory import TrajectoryStep
import test_v5_contract as fixtures


def row(index=0, *, status='IN_PROGRESS', search=False, privileged=False):
    old=fixtures.v4();address=c.DecisionAddress(100,index+9,0)
    posterior=c.PublicPosterior(fixtures.vocabulary(),'blissey',0b101,(.2,.3,.5))
    obs=c.append_belief_tokens(old,(posterior,)*6,address=address,category_vocabulary_sha256='b'*64)
    step=TrajectoryStep('p1',index,old,old.legal_action_mask,0,reward=.3,
        opponent_action_index=4,action_probability=.7,value_estimate=.8)
    labels=c.V5Labels(address,opponent_next_action=c.OpponentAction('MOVE','surf'),
        hidden_sets=(c.HiddenSet('blissey',(0,2)),)+(None,)*5,terminal_status=status,
        outcome=1 if status=='COMPLETE' else None,
        pokemon_end=(c.PokemonEnd(True,.5),)+(None,)*11 if status=='COMPLETE' else (None,)*12,
        root_improvement=fixtures.improvement() if search else None,
        search_status='LABELED' if search else 'UNSELECTED')
    peer=None
    if privileged:
        peer=c.privileged_view(obs,fixtures.v4(owner=1,offset=100),peer_address=replace(address,owner=1),
            opponent_slot_map=tuple(range(6)),peer_category_vocabulary_sha256='b'*64)
    return data.DecisionRow('battle',71,'gen3randombattle',step,obs,labels,'c'*64,peer,
        'd'*64 if status=='COMPLETE' else None)


def base_batch(rows):
    examples=[]
    for r in rows:
        o=r.step.observation
        examples.append(d.TrajectoryExample(r.battle_id,r.seed,r.format_id,r.step.player_id,r.step.turn_index,
            (o.categorical_ids,),(o.numeric_features,),(o.token_type_ids,),(o.attention_mask,),(True,),
            o.legal_action_mask,r.step.action_index,r.step.reward,99.,value_estimate=.8,
            ppo_advantage=77.,ppo_value_target=66.,opponent_action_index=4,action_probability=.7,
            terminal_capped=r.labels.terminal_status=='CAPPED'))
    return d.training_batch_from_examples(examples)


class V5DataTests(unittest.TestCase):
    def test_typed_same_frame_labels_and_trajectory_json_roundtrip(self):
        original=row(status='COMPLETE',search=True,privileged=True)
        encoded=data.row_to_dict(original)
        restored=data.row_from_dict(json.loads(json.dumps(encoded)))
        self.assertEqual(restored,original)
        self.assertEqual(restored.observation.model_inputs(),original.observation.model_inputs())
        self.assertNotEqual(restored.privileged.numeric_features,restored.observation.numeric_features)
        action=data.semantic_opponent_action(original.observation.address,
            replace(original.observation.address,owner=1),kind='SWITCH',identity='blissey')
        self.assertEqual(action.index(original.observation.label_vocabulary),3)
        with self.assertRaisesRegex(ValueError,'same stream/frame'):
            data.semantic_opponent_action(original.observation.address,
                replace(original.observation.address,owner=1,frame=10),kind='MOVE',identity='surf')
        changed=replace(original.step,action_index=1)
        with self.assertRaisesRegex(ValueError,'original behavior'):
            data.attach_batch(base_batch([original]),[replace(original,step=changed)])
        legacy=replace(original.step.observation,schema_version='pokezero.observation.v3')
        with self.assertRaisesRegex(ValueError,'original v4 step schema'):
            replace(original,step=replace(original.step,observation=legacy))

    def test_attach_uses_public29_inputs_and_keeps_behavior_ppo_and_privileged_sidecar_separate(self):
        rows=(row(status='COMPLETE',search=True,privileged=True),row(1,status='CAPPED',search=True))
        base=base_batch(rows);batch=data.attach_batch(base,rows)
        self.assertIsNone(base.v5_columns)
        for name in ('action_indices','rewards','returns','value_estimates','ppo_advantages','ppo_value_targets',
                'action_probabilities','training_weights'):
            self.assertEqual(getattr(batch,name),getattr(base,name))
        self.assertEqual(np.asarray(batch.numeric_features).shape,(2,1,29,132))
        np.testing.assert_array_equal(np.asarray(batch.numeric_features)[0,0,:23],
            np.asarray(base.numeric_features)[0,0])
        self.assertEqual(batch.opponent_action_mask,(False,False))
        self.assertEqual(batch.opponent_action_indices,(-1,-1))
        self.assertEqual(batch.v5_columns['opponent_semantic_action_targets'],(0,0))
        self.assertEqual(batch.v5_columns['outcome_targets'],(1,0.))
        self.assertEqual(batch.v5_columns['outcome_target_mask'],(True,False))
        self.assertEqual(batch.search_policy_target_mask,(True,True))
        self.assertEqual(batch.search_value_target_mask,(True,False))
        self.assertAlmostEqual(batch.search_value_targets[0],.825)
        self.assertEqual(batch.search_value_targets[1],0.)
        self.assertEqual(batch.v5_columns['privileged_view_mask'],(True,False))
        self.assertNotEqual(batch.v5_columns['privileged_numeric_features'][0],batch.numeric_features[0])

    def test_actual_terminal_data_not_returns_or_gae_controls_auxiliary_masks(self):
        original=row()
        done=data.with_terminal(original,address=c.DecisionAddress(100,20,0),status='COMPLETE',outcome=-1,
            pokemon_end=(c.PokemonEnd(False,0.),)*12,source_sha256='e'*64)
        batch=data.attach_batch(base_batch([done]),[done])
        self.assertEqual(batch.returns,(99.,))
        self.assertEqual(batch.ppo_value_targets,(66.,))
        self.assertEqual(batch.v5_columns['outcome_targets'],(-1,))
        self.assertEqual(batch.v5_columns['survival_targets'],((0.,)*12,))
        self.assertEqual(batch.v5_columns['pokemon_end_target_mask'],((True,)*12,))
        capped=data.with_terminal(original,address=c.DecisionAddress(100,20,0),status='CAPPED',source_sha256='e'*64)
        capped_batch=data.attach_batch(base_batch([capped]),[capped])
        self.assertFalse(any(capped_batch.v5_columns['pokemon_end_target_mask'][0]))
        self.assertFalse(capped_batch.v5_columns['outcome_target_mask'][0])
        with self.assertRaises(ValueError):data.with_terminal(original,address=c.DecisionAddress(999,20,0),
            status='COMPLETE',outcome=1,source_sha256='e'*64)
        with self.assertRaises(ValueError):data.with_terminal(original,address=c.DecisionAddress(100,20,0),
            status='REFUSED',outcome=1,source_sha256='e'*64)

    def test_slice_combine_retains_exact_sidecars_and_rejects_namespace_or_legacy_mix(self):
        rows=(row(status='COMPLETE',search=True),row(1,status='CAPPED'),row(2))
        batch=data.attach_batch(base_batch(rows),rows)
        pieces=[d.slice_training_batch(batch,i,i+1) for i in range(3)]
        combined=d._combine_training_batches(pieces)
        np.testing.assert_array_equal(combined.numeric_features,batch.numeric_features)
        for name in data.ARRAY_SHAPES:
            np.testing.assert_array_equal(combined.v5_columns[name],batch.v5_columns[name])
        for name in (*data.STATIC,*data.META_ROWS):
            self.assertEqual(combined.v5_columns[name],batch.v5_columns[name])
        with self.assertRaisesRegex(ValueError,'legacy and v5'):
            d._combine_training_batches([pieces[0],base_batch([rows[0]])])
        changed=dict(pieces[1].v5_columns,category_vocabulary_sha256='f'*64)
        alternate=replace(pieces[1],v5_columns=changed)
        with self.assertRaisesRegex(ValueError,'different v5'):
            d._combine_training_batches([pieces[0],alternate])

    def test_explicit_v5_cache_roundtrip_masks_bindings_and_legacy_reader_refusal(self):
        rows=(row(status='COMPLETE',search=True,privileged=True),row(1,status='CAPPED',search=True))
        batch=data.attach_batch(base_batch(rows),rows)
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'v5'
            metadata=data.write_cache(path,batch)
            self.assertEqual(metadata['schema_version'],data.CACHE_SCHEMA)
            restored=tuple(data.iter_cache_batches(path,batch_size=1))
            merged=d._combine_training_batches(restored)
            for name in (*data.INPUTS,*data.SEARCH,'returns','ppo_value_targets'):
                expected=getattr(batch,name)
                if name in data.SEARCH:
                    # Existing dataset combine uses its original float32/bool
                    # search schema; the literal teacher record stays exact.
                    expected=np.asarray(expected,dtype=np.bool_ if name.endswith('mask') else np.float32)
                np.testing.assert_array_equal(getattr(merged,name),expected)
            for name in data.ARRAY_SHAPES:
                np.testing.assert_array_equal(merged.v5_columns[name],batch.v5_columns[name])
            self.assertEqual(merged.v5_columns['source_frame_sha256'],('c'*64,)*2)
            self.assertEqual(merged.v5_columns['terminal_status'],('COMPLETE','CAPPED'))
            with self.assertRaisesRegex(ValueError,'Unsupported training cache schema'):
                tuple(d.iter_training_cache_batches(path,batch_size=1))
            with self.assertRaises(FileExistsError):data.write_cache(path,batch)
            file=path/'v5__hidden_set_targets.npy'
            with file.open('ab') as stream:stream.write(b'changed')
            with self.assertRaisesRegex(ValueError,'array bytes changed'):
                tuple(data.iter_cache_batches(path,batch_size=1))

    def test_unknown_label_poison_is_ignored_but_active_poison_and_remixed_D_refuse(self):
        original=row(status='CAPPED',search=True)
        batch=data.attach_batch(base_batch([original]),[original])
        cols=dict(batch.v5_columns)
        cols['outcome_targets']=(float('nan'),)
        cols['survival_targets']=((float('nan'),)*12,)
        cols['final_hp_targets']=((float('nan'),)*12,)
        ignored=replace(batch,v5_columns=cols)
        self.assertFalse(ignored.v5_columns['outcome_target_mask'][0])
        active=dict(cols,outcome_target_mask=(True,))
        with self.assertRaises(ValueError):replace(batch,v5_columns=active)
        complete=row(status='COMPLETE',search=True)
        complete_batch=data.attach_batch(base_batch([complete]),[complete])
        with self.assertRaisesRegex(ValueError,'remixed'):
            replace(complete_batch,search_value_targets=(.99,))
        bad=dict(complete_batch.v5_columns,hidden_set_targets=(((1.,)*64,)+(tuple([0.]*64),)*5,))
        with self.assertRaisesRegex(ValueError,'hidden-set distribution'):
            replace(complete_batch,v5_columns=bad)
        with tempfile.TemporaryDirectory() as directory:
            data.write_cache(Path(directory)/'unknown',ignored)
            cached=next(data.iter_cache_batches(Path(directory)/'unknown',batch_size=1))
            self.assertTrue(np.isnan(cached.v5_columns['outcome_targets'][0]))
            self.assertFalse(cached.v5_columns['outcome_target_mask'][0])

    def test_legacy_default_batch_and_v2_cache_path_still_have_no_v5_data(self):
        import test_cache_concat as legacy
        old=base_batch([row()])
        self.assertIsNone(d.slice_training_batch(old,0,1).v5_columns)
        self.assertIsNone(d._combine_training_batches([old,old]).v5_columns)
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            path=legacy.write_cache(root,'legacy',[legacy.rollout_record(11)],config=d.TrajectoryDatasetConfig(window_size=1))
            metadata=json.loads((path/'metadata.json').read_text())
            self.assertEqual(metadata['schema_version'],d.TRAINING_CACHE_SCHEMA_VERSION)
            self.assertFalse(any(name.startswith('v5') for name in metadata))
            self.assertFalse(any(file.name.startswith('v5') for file in path.iterdir()))
            self.assertIsNone(next(d.iter_training_cache_batches(path,batch_size=2)).v5_columns)

    def test_actor_poison_and_missing_original_source_binding_are_rejected(self):
        original=row();batch=data.attach_batch(base_batch([original]),[original])
        values=np.array(batch.numeric_features);values[0,0,0,0]=np.nan
        with self.assertRaisesRegex(ValueError,'public actor'):
            replace(batch,numeric_features=values)
        missing=dict(batch.v5_columns,source_frame_sha256=(None,))
        with self.assertRaisesRegex(ValueError,'SHA256'):
            replace(batch,v5_columns=missing)

    def test_existing_policy_and_value_targets_are_preserved_independently(self):
        original=row();base=base_batch([original])
        value_only=replace(base,search_policy_targets=((float('nan'),)*9,),search_policy_target_mask=(False,),
            search_value_targets=(.4,),search_value_target_mask=(True,))
        with self.assertRaisesRegex(ValueError,'existing search'):
            data.attach_batch(value_only,[original])
        # Matching existing value supervision need not have an existing policy
        # label. New same-frame D data may supply the previously absent policy.
        labeled=row(status='COMPLETE',search=True)
        base=base_batch([labeled])
        matching=replace(base,search_policy_targets=((float('nan'),)*9,),search_policy_target_mask=(False,),
            search_value_targets=(.825,),search_value_target_mask=(True,))
        attached=data.attach_batch(matching,[labeled])
        self.assertEqual(attached.search_value_targets,(.825,))
        self.assertEqual(attached.search_value_target_mask,(True,))
        policy_only=replace(base,search_policy_targets=(labeled.labels.root_improvement.policy_target,),
            search_policy_target_mask=(True,),search_value_targets=(float('nan'),),search_value_target_mask=(False,))
        self.assertTrue(data.attach_batch(policy_only,[labeled]).search_policy_target_mask[0])

    def test_cached_owner_and_cap_facts_cannot_be_rewritten_by_sidecar(self):
        original=row(status='CAPPED');batch=data.attach_batch(base_batch([original]),[original])
        wrong_address=dict(batch.v5_columns['addresses'][0],owner=1)
        with self.assertRaisesRegex(ValueError,'original behavior owner'):
            replace(batch,v5_columns=dict(batch.v5_columns,addresses=(wrong_address,)))
        observed_tie=dict(batch.v5_columns,terminal_status=('COMPLETE',),
            terminal_source_sha256=('e'*64,),outcome_targets=(0.,),outcome_target_mask=(True,))
        with self.assertRaisesRegex(ValueError,'original capped disposition'):
            replace(batch,v5_columns=observed_tie)

    def test_active_private_payload_obeys_original_real_category_role_and_mask_constraints(self):
        original=row(privileged=True);batch=data.attach_batch(base_batch([original]),[original])
        variants=[]
        for value,dtype in ((np.nan,float),(np.inf,float),(True,bool),(1+0j,complex),('1',object)):
            numeric=np.array(batch.v5_columns['privileged_numeric_features'],dtype=dtype)
            numeric[0,0,0,0]=value
            variants.append(('privileged_numeric_features',numeric))
        for value,dtype in ((-1,np.int32),(1.,float),(True,bool)):
            categories=np.array(batch.v5_columns['privileged_categorical_ids'],dtype=dtype)
            categories[0,0,0,0]=value
            variants.append(('privileged_categorical_ids',categories))
        for value,dtype in ((4,np.int32),(7.,float),(-1,np.int32)):
            roles=np.array(batch.v5_columns['privileged_token_type_ids'],dtype=dtype)
            roles[0,0,28]=value
            variants.append(('privileged_token_type_ids',roles))
        mask=np.array(batch.v5_columns['privileged_attention_mask']);mask[0,0,28]=False
        variants.extend((('privileged_attention_mask',mask),
            ('privileged_attention_mask',np.array(batch.v5_columns['privileged_attention_mask'],np.int32))))
        for key,value in variants:
            with self.subTest(key=key,dtype=value.dtype), self.assertRaisesRegex(ValueError,'active privileged'):
                replace(batch,v5_columns=dict(batch.v5_columns,**{key:value}))
        data.validate_batch(batch)

    def test_masked_private_payload_is_ignored_and_does_not_change_public_or_targets(self):
        original=row(status='COMPLETE',search=True);batch=data.attach_batch(base_batch([original]),[original])
        changed=dict(batch.v5_columns,
            privileged_numeric_features=np.full((1,1,29,132),np.nan),
            privileged_categorical_ids=np.full((1,1,29,41),-3.5),
            privileged_token_type_ids=np.full((1,1,29),-9.),
            privileged_attention_mask=np.full((1,1,29),17.))
        ignored=replace(batch,v5_columns=changed)
        for name in (*data.INPUTS,*data.SEARCH,'returns','ppo_value_targets','action_indices'):
            np.testing.assert_array_equal(getattr(ignored,name),getattr(batch,name))
        self.assertEqual(ignored.v5_columns['outcome_targets'],batch.v5_columns['outcome_targets'])
        self.assertFalse(ignored.v5_columns['privileged_view_mask'][0])


if __name__=='__main__':
    unittest.main()
