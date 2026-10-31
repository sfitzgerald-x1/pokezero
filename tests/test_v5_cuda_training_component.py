"""Pure fixture/config/source controls; no Torch import, model, GPU or games."""
import ast
from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from pokezero import v5_cuda_training_component as smoke,v5_data
from pokezero.v5_model import V5ModelConfig
import test_v5_data as fixtures


class ComponentTests(unittest.TestCase):
    def plan(self):
        return dict(schema=smoke.SCHEMA,fixture_kind='NON_GAME',games=0,updates_per_arm=2,rows_per_arm=2,
            unit_schedule_coordinates=dict(start_game=0,end_game=1,total_games=2000000),
            arms=[dict(arm=a,replicate=r)for a,r in smoke.ROLES],
            remote_new_root='/shared/scott/pokezero-v5-cuda-training-component-UNIT')

    def batch(self,arm):
        row=fixtures.row()
        config=V5ModelConfig(128,'b'*64,row.observation.label_vocabulary,arm=arm,width=512,layers=3)
        observation=row.observation
        columns={name:np.asarray([[getattr(observation,name)]]*2)for name in
            ('categorical_ids','numeric_features','token_type_ids','attention_mask')}
        columns.update(history_mask=np.ones((2,1),bool),legal_action_mask=np.asarray([observation.legal_action_mask]*2,bool))
        return smoke.fixture_batch(columns,config,input_sha256='e'*64),columns,config

    def test_explicit_enablement_precedes_input_and_runtime_access(self):
        with patch.object(smoke,'document')as read:
            with self.assertRaisesRegex(ValueError,'enablement'):smoke.execute({},'/never',enable_non_game=False)
            read.assert_not_called()
        with patch('sys.argv',['component','--prespec','/never-read','--out','/never']):
            with self.assertRaisesRegex(ValueError,'before input access'):smoke.main()

    def test_exact_non_game_denominator_and_exclusive_output(self):
        smoke.validate_plan(self.plan())
        for update in (dict(games=1),dict(fixture_kind='GAME'),dict(rows_per_arm=512),
                dict(updates_per_arm=3),dict(arms=list(reversed(self.plan()['arms']))),
                dict(unit_schedule_coordinates=dict(start_game=0,end_game=2000000,total_games=2000000)),
                dict(remote_new_root='/shared/scott/model0')):
            with self.subTest(update=update),self.assertRaises(ValueError):smoke.validate_plan(dict(self.plan(),**update))

    def test_original_fixture_arrays_preserved_no_observed_or_terminal_truth(self):
        for arm in ('A','B','D'):
            batch,columns,_=self.batch(arm)
            v5_data.validate_batch(batch)
            for name in ('categorical_ids','numeric_features','token_type_ids','attention_mask','history_mask','legal_action_mask'):
                np.testing.assert_array_equal(getattr(batch,name),columns[name])
            for name in ('outcome_target_mask','pokemon_end_target_mask','hidden_set_target_mask','opponent_semantic_action_target_mask','privileged_view_mask'):
                self.assertFalse(np.asarray(batch.v5_columns[name]).any())
            self.assertEqual(batch.v5_columns['terminal_status'],('IN_PROGRESS',)*2)
            self.assertEqual(batch.v5_columns['terminal_source_sha256'],(None,None))
            self.assertFalse(np.asarray(batch.search_value_target_mask).any())
            self.assertTrue(all(x['fixture_kind']=='NON_GAME'and x['actual_teacher_eligible']is False for x in batch.step_metadata))
            self.assertTrue(all(x.startswith('NON_GAME_')for x in batch.battle_ids))

    def test_D_unit_policy_tensor_legal_full_denominator_and_cache_is_explicitly_non_game(self):
        batch,_,_=self.batch('D')
        self.assertEqual(tuple(batch.search_policy_target_mask),(True,True))
        for record,legal,target in zip(batch.v5_columns['root_improvement_records'],batch.legal_action_mask,batch.search_policy_targets):
            self.assertEqual(record['worlds'],2)
            self.assertEqual(record['completed_q'],tuple(0. if x else None for x in legal))
            np.testing.assert_allclose(target,np.asarray(legal)/sum(legal))
        with tempfile.TemporaryDirectory()as tmp:
            path=Path(tmp)/'NON_GAME_CACHE';v5_data.write_cache(path,batch)
            restored=list(v5_data.iter_cache_batches(path,batch_size=512))[0]
            np.testing.assert_array_equal(restored.battle_ids,batch.battle_ids)
            self.assertTrue(all(x['fixture_kind']=='NON_GAME'for x in restored.step_metadata))
            v5_data.validate_batch(restored)

    def test_no_production_terminal_or_D_value_validator_is_weakened(self):
        batch,_,_=self.batch('D');cols=dict(batch.v5_columns)
        cols['terminal_status']=('COMPLETE',)*2
        with self.assertRaisesRegex(ValueError,'terminal capture'):replace(batch,v5_columns=cols)
        with self.assertRaisesRegex(ValueError,'independent D policy/value masks'):
            replace(batch,search_value_target_mask=(True,True))
        columns=self.batch('A')[1];columns['legal_action_mask']=np.zeros((2,9),bool)
        with self.assertRaisesRegex(ValueError,'nonempty legal'):smoke.fixture_batch(columns,self.batch('A')[2],input_sha256='e'*64)

    def test_original_managed_memory_trainer_checkpoint_export_api_order_is_literal(self):
        tree=ast.parse(Path(smoke.__file__).read_text())
        execute=next(x for x in tree.body if isinstance(x,ast.FunctionDef)and x.name=='execute')
        calls=sorted((x for x in ast.walk(execute)if isinstance(x,ast.Call)),key=lambda x:x.lineno)
        names=[ast.unparse(x.func)for x in calls]
        required=['worker._source','worker._memory','worker._probe','training.create','training.train_caches',
            'training.save_checkpoint','training.load_checkpoint','training.train_caches','training.save_checkpoint','export.export','export.load']
        selected=[x for x in names if x in set(required)]
        self.assertEqual(selected,required)
        for x in calls:
            if ast.unparse(x.func)=='training.train_caches':
                self.assertEqual(ast.unparse(next(k.value for k in x.keywords if k.arg=='max_updates')),'1')
        self.assertNotIn('forward_v5',names)
        self.assertFalse(any(name!='gc.collect' and name.endswith(('.collect','.collect_paired','.construct','.step_with_chance'))for name in names))


if __name__=='__main__':unittest.main()
