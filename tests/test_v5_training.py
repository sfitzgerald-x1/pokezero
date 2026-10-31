"""Tiny CPU-only optimizer/checkpoint tests; no champion, rollout or world."""
from dataclasses import replace
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import random
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from pokezero import v5_training as training, v5_data
from pokezero.v5_model import V5ModelConfig
from pokezero.v5_loss import V5LossConfig
import test_v5_data as data_fixture
import test_v5_contract as contract_fixture

try:
    import torch
except ImportError:
    torch=None


def configurations(arm='B',*,replicate=0,seed=37,epochs=1,precision='float32'):
    model=V5ModelConfig(128,'b'*64,contract_fixture.vocabulary(),arm=arm,replicate=replicate,
        width=16,layers=1,attention_heads=2,expert_width=12,value_hidden=(16,16),dropout=.2)
    return model,V5LossConfig(arm=arm),training.TrainConfig(seed,batch_size=1,epochs=epochs,
        learning_rate=.001,max_grad_norm=1.,precision=precision)


def batch(arm='B',*,index=0,status='COMPLETE'):
    row=data_fixture.row(index,status=status,search=arm=='D')
    base=data_fixture.base_batch([row])
    base=replace(base,ppo_advantages=(.25,),ppo_value_targets=(.4,),returns=(1.,))
    return v5_data.attach_batch(base,[row])


def assert_tree_equal(test,left,right):
    if torch.is_tensor(left):
        test.assertTrue(torch.equal(left,right))
    elif isinstance(left,dict):
        test.assertEqual(set(left),set(right))
        for key in left:assert_tree_equal(test,left[key],right[key])
    elif isinstance(left,(tuple,list)):
        test.assertEqual(len(left),len(right))
        for a,b in zip(left,right):assert_tree_equal(test,a,b)
    else:test.assertEqual(left,right)


@unittest.skipIf(torch is None,'requires optional local Torch')
class V5TrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(1)

    def test_explicit_enablement_precedes_model_file_access_and_scratch_rng_is_owned(self):
        cfg=configurations()
        with patch.object(training,'_torch') as backend:
            with self.assertRaisesRegex(ValueError,'enablement'):training.create(*cfg)
            with self.assertRaisesRegex(ValueError,'enablement'):training.load_checkpoint('/never-read')
            backend.assert_not_called()
        torch.manual_seed(19);random.seed(19);np.random.seed(19)
        before=training._rng_state(torch,False)
        first=training.create(*cfg,enable_v5=True)
        assert_tree_equal(self,before,training._rng_state(torch,False))
        again=training.create(*cfg,enable_v5=True)
        assert_tree_equal(self,first.model.state_dict(),again.model.state_dict())
        other=training.create(*configurations(replicate=1,seed=38),enable_v5=True)
        self.assertFalse(torch.equal(first.model.category_embedding.weight,other.model.category_embedding.weight))

    def test_first_optimizer_step_changes_small_weights_for_all_four_initial_identities(self):
        for arm,replicate in (('A',0),('B',0),('D',0),('B',1)):
            with self.subTest(arm=arm,replicate=replicate):
                runtime=training.create(*configurations(arm,replicate=replicate),enable_v5=True)
                before={key:value.clone() for key,value in runtime.model.state_dict().items()}
                actual=batch(arm);original_advantage=actual.ppo_advantages
                metrics=training.update(runtime,actual)
                self.assertEqual((runtime.steps,runtime.rows),(1,1))
                self.assertTrue(np.isfinite(metrics['loss']))
                self.assertGreater(metrics['gradient_norm'],0)
                self.assertTrue(any(not torch.equal(before[key],value) for key,value in runtime.model.state_dict().items()))
                self.assertEqual(actual.ppo_advantages,original_advantage)

    def test_cache_checkpoint_resume_is_exact_next_batch_with_dropout_optimizer_and_rng(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);paths=[]
            for i in range(2):
                path=root/('cache'+str(i));v5_data.write_cache(path,batch(index=i));paths.append(path)
            cfg=configurations(epochs=2)
            full=training.create(*cfg,enable_v5=True)
            report=training.train_caches(full,paths)
            self.assertEqual((report['steps'],report['rows'],report['complete']),(4,4,True))
            partial=training.create(*cfg,enable_v5=True)
            first=training.train_caches(partial,paths,max_updates=1)
            self.assertEqual((first['progress']['cache'],first['progress']['batch']),(1,0))
            checkpoint=root/'one.pt';receipt=training.save_checkpoint(checkpoint,partial)
            self.assertEqual(receipt['steps'],1)
            torch.rand(13);np.random.random(13);random.random()
            resumed=training.load_checkpoint(checkpoint,enable_v5=True,expected_model_config=cfg[0],
                expected_loss_config=cfg[1],expected_train_config=cfg[2])
            rest=training.train_caches(resumed,paths)
            self.assertEqual(len(rest['updates']),3)
            self.assertEqual(rest['progress'],report['progress'])
            assert_tree_equal(self,full.model.state_dict(),resumed.model.state_dict())
            assert_tree_equal(self,full.optimizer.state_dict(),resumed.optimizer.state_dict())
            assert_tree_equal(self,full.rng,resumed.rng)
            self.assertEqual(training.train_caches(resumed,paths)['updates'],[])
            with self.assertRaises(FileExistsError):training.save_checkpoint(checkpoint,resumed)

    def test_resume_rejects_namespace_old_checkpoint_cache_order_and_bad_cursor(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);paths=[]
            for i in range(2):
                path=root/('cache'+str(i));v5_data.write_cache(path,batch(index=i));paths.append(path)
            cfg=configurations();runtime=training.create(*cfg,enable_v5=True)
            training.train_caches(runtime,paths,max_updates=1)
            checkpoint=root/'one.pt';training.save_checkpoint(checkpoint,runtime)
            different=replace(cfg[0],category_vocabulary_sha256='f'*64)
            with self.assertRaisesRegex(ValueError,'resume configuration'):
                training.load_checkpoint(checkpoint,enable_v5=True,expected_model_config=different)
            with self.assertRaisesRegex(ValueError,'ordered v5 caches'):
                training.train_caches(runtime,list(reversed(paths)))
            self.assertEqual(runtime.steps,1)
            legacy=root/'legacy.pt';torch.save(dict(schema_version='pokezero.neural_policy.v4'),legacy)
            with self.assertRaisesRegex(ValueError,'same-source v5'):
                training.load_checkpoint(legacy,enable_v5=True)
            value=torch.load(checkpoint,weights_only=True);value['metadata']['progress']['batch']=999
            value['metadata_sha256']=training._sha(value['metadata'])
            bad=root/'bad.pt';torch.save(value,bad)
            with self.assertRaisesRegex(ValueError,'outside original finite roster'):
                training.load_checkpoint(bad,enable_v5=True)
            changed=torch.load(checkpoint,weights_only=True)
            changed['optimizer_state']['param_groups'][0]['lr']=.5
            wrong_optimizer=root/'optimizer.pt';torch.save(changed,wrong_optimizer)
            with self.assertRaisesRegex(ValueError,'optimizer configuration differs'):
                training.load_checkpoint(wrong_optimizer,enable_v5=True)

    def test_capped_critic_needs_literal_later_public_bootstrap_not_return_or_gae(self):
        actual=batch(status='CAPPED');runtime=training.create(*configurations(),enable_v5=True)
        before={key:value.clone() for key,value in runtime.model.state_dict().items()}
        with self.assertRaisesRegex(ValueError,'actual v5_bootstrap'):
            training.update(runtime,actual)
        self.assertEqual(runtime.steps,0)
        assert_tree_equal(self,before,runtime.model.state_dict())
        capture=dict(schema=training.BOOTSTRAP_SCHEMA,address=dict(stream=100,frame=10,owner=0),
            source_frame_sha256='e'*64,value=.2)
        valid=replace(actual,step_metadata=({'v5_bootstrap':capture},))
        self.assertEqual(training.update(runtime,valid)['optimizer_step'],1)
        wrong=dict(capture,address=dict(stream=100,frame=10,owner=1))
        with self.assertRaisesRegex(ValueError,'later original stream/owner'):
            training.update(runtime,replace(actual,step_metadata=({'v5_bootstrap':wrong},)))
        masked=replace(actual,ppo_value_target_mask=(False,))
        self.assertEqual(training.update(runtime,masked)['optimizer_step'],2)
        self.assertFalse(masked.v5_columns['outcome_target_mask'][0])

    def test_cpu_bfloat16_autocast_is_explicit_and_checkpoint_self_describes_it(self):
        runtime=training.create(*configurations(precision='bfloat16'),enable_v5=True)
        self.assertTrue(np.isfinite(training.update(runtime,batch())['loss']))
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'bf16.pt';training.save_checkpoint(path,runtime)
            restored=training.load_checkpoint(path,enable_v5=True)
            self.assertEqual(restored.train_config.precision,'bfloat16')
            assert_tree_equal(self,runtime.model.state_dict(),restored.model.state_dict())

    def test_train_only_cli_scratch_resume_and_opt_in_before_config_access(self):
        from pokezero import v5_cli
        with self.assertRaisesRegex(ValueError,'enablement'):
            v5_cli.main(['train','--config','/not-read','--cache','/not-read','--out','/not-written'])
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);cache=root/'cache';v5_data.write_cache(cache,batch())
            cfg=configurations(epochs=2)
            config=root/'config.json';config.write_text(json.dumps(dict(model=cfg[0].metadata(),
                loss=cfg[1].metadata(),training=training.asdict(cfg[2]))))
            first=root/'first.pt';last=root/'last.pt';summary=root/'summary.json'
            with redirect_stdout(io.StringIO()):
                self.assertEqual(v5_cli.main(['train','--enable-v5','--config',str(config),
                    '--cache',str(cache),'--out',str(first),'--max-updates','1']),0)
                self.assertEqual(v5_cli.main(['train','--enable-v5','--resume',str(first),
                    '--cache',str(cache),'--out',str(last),'--summary',str(summary)]),0)
            report=json.loads(summary.read_text())
            self.assertEqual((report['steps'],report['rows'],report['complete']),(2,2,True))
            with self.assertRaises(FileExistsError):
                v5_cli.main(['train','--enable-v5','--resume',str(first),'--cache',str(cache),'--out',str(last)])


if __name__=='__main__':unittest.main()
