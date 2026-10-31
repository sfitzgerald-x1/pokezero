"""Opt-in declared-game schedule; tiny synthetic CPU data/model only."""
from dataclasses import asdict,replace
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch
import torch
from pokezero import v5_training as t,v5_data,v5_run as run
from pokezero.neural_policy import _learning_rate_for_epoch
import test_v5_training as fixture
import test_v5_run as run_fixture


class ScheduledTrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(1)

    def config(self,epochs=3):
        model,loss,train=fixture.configurations(epochs=epochs)
        return model,loss,replace(train,learning_rate_schedule='mit-thesis')

    def caches(self,root,start=0):
        paths=[]
        for i in range(start,start+2):
            path=root/('cache'+str(i));v5_data.write_cache(path,fixture.batch(index=i));paths.append(path)
        return paths

    def rate(self,cfg,window,epoch):
        return _learning_rate_for_epoch(NS(epochs=cfg.epochs,learning_rate=cfg.learning_rate,
            learning_rate_schedule=cfg.learning_rate_schedule,
            learning_rate_progress_start=window.start_game/window.total_games,
            learning_rate_progress_end=window.end_game/window.total_games,
            learning_rate_warmup_progress=0.,learning_rate_resume_warmup_start_progress=None,
            learning_rate_resume_warmup_end_progress=None,learning_rate_resume_warmup_start_lr=None),epoch)

    def test_existing_epoch_schedule_and_exact_dropout_optimizer_rng_resume(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);paths=self.caches(root);cfg=self.config();window=t.DeclaredGameWindow(0,10,20)
            full=t.create(*cfg,enable_v5=True);report=t.train_caches(full,paths,learning_rate_window=window)
            expected=[self.rate(cfg[2],window,epoch) for epoch in (1,1,2,2,3,3)]
            self.assertEqual([m['learning_rate'] for m in report['updates']],expected)
            self.assertEqual([m['learning_rate_epoch'] for m in report['updates']],[1,1,2,2,3,3])
            partial=t.create(*cfg,enable_v5=True);first=t.train_caches(partial,paths,max_updates=3,learning_rate_window=window)
            self.assertEqual((partial.steps,partial.progress['epoch'],partial.progress['cache']),(3,1,1))
            checkpoint=root/'partial.pt';t.save_checkpoint(checkpoint,partial)
            torch.rand(11)
            resumed=t.load_checkpoint(checkpoint,enable_v5=True,expected_train_config=cfg[2],expected_learning_rate_window=window)
            rest=t.train_caches(resumed,paths,learning_rate_window=window)
            self.assertEqual([m['learning_rate'] for m in rest['updates']],expected[3:])
            for left,right in [(full.model.state_dict(),resumed.model.state_dict()),(full.optimizer.state_dict(),resumed.optimizer.state_dict()),(full.rng,resumed.rng),(full.progress,resumed.progress)]:fixture.assert_tree_equal(self,left,right)
            self.assertEqual(t.train_caches(resumed,paths,learning_rate_window=window)['updates'],[])
            self.assertEqual(resumed.optimizer.param_groups[0]['lr'],expected[-1])
            self.assertEqual(full.learning_rate_state,resumed.learning_rate_state)

    def test_window_cursor_config_and_literal_optimizer_mutants_refuse(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);paths=self.caches(root);cfg=self.config();window=t.DeclaredGameWindow(0,10,20)
            runtime=t.create(*cfg,enable_v5=True)
            with self.assertRaisesRegex(ValueError,'finite train_caches'):t.update(runtime,fixture.batch())
            with self.assertRaisesRegex(ValueError,'window'):t.train_caches(runtime,paths)
            with self.assertRaisesRegex(ValueError,'game zero'):t.train_caches(runtime,paths,learning_rate_window=t.DeclaredGameWindow(1,10,20))
            t.train_caches(runtime,paths,max_updates=3,learning_rate_window=window)
            checkpoint=root/'original.pt';t.save_checkpoint(checkpoint,runtime)
            for w in (None,t.DeclaredGameWindow(0,11,20)):
                with self.assertRaisesRegex(ValueError,'resume window'):t.load_checkpoint(checkpoint,enable_v5=True,expected_learning_rate_window=w)
            for name,value in [('progress',None),('training',None),('group',None)]:
                payload=torch.load(checkpoint,weights_only=True)
                if name=='progress':payload['metadata']['progress']['cache']=0
                if name=='training':payload['metadata']['training']['learning_rate_schedule']='constant'
                if name=='group':payload['optimizer_state']['param_groups'][0]['lr']=cfg[2].learning_rate
                payload['metadata_sha256']=t._sha(payload['metadata']);p=root/(name+'.pt');torch.save(payload,p)
                with self.assertRaises(ValueError):t.load_checkpoint(p,enable_v5=True,expected_train_config=cfg[2],expected_learning_rate_window=window)
            original=runtime.progress['cache'];runtime.progress['cache']=0
            with self.assertRaisesRegex(ValueError,'scheduled finite cache cursor'):t.save_checkpoint(root/'cursor.pt',runtime)
            runtime.progress['cache']=original
            original_cfg=runtime.train_config;runtime.train_config=replace(original_cfg,learning_rate_schedule='constant')
            with self.assertRaisesRegex(ValueError,'original scheduled'):t.save_checkpoint(root/'config.pt',runtime)
            runtime.train_config=original_cfg
            with self.assertRaisesRegex(ValueError,'ordered v5 caches'):t.train_caches(runtime,list(reversed(paths)),learning_rate_window=window)

    def test_contiguous_declared_windows_and_no_cache_refill_or_replay(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);first=self.caches(root);second=self.caches(root,2);cfg=self.config(epochs=2)
            runtime=t.create(*cfg,enable_v5=True);w0=t.DeclaredGameWindow(0,10,20);w1=t.DeclaredGameWindow(10,20,20)
            t.train_caches(runtime,first,learning_rate_window=w0)
            with self.assertRaisesRegex(ValueError,'contiguous'):t.train_caches(runtime,second,learning_rate_window=t.DeclaredGameWindow(11,20,20))
            with self.assertRaisesRegex(ValueError,'replay'):t.train_caches(runtime,first,learning_rate_window=w1)
            r=t.train_caches(runtime,second,max_updates=1,learning_rate_window=w1)
            self.assertEqual((runtime.steps,r['updates'][0]['learning_rate']),(5,self.rate(cfg[2],w1,1)))
            p=root/'second.pt';t.save_checkpoint(p,runtime);resumed=t.load_checkpoint(p,enable_v5=True,expected_learning_rate_window=w1)
            t.train_caches(resumed,second,learning_rate_window=w1)
            self.assertEqual((resumed.steps,resumed.rows,resumed.progress['complete']),(8,8,True))
            self.assertEqual(resumed.optimizer.param_groups[0]['lr'],self.rate(cfg[2],w1,2))

    def test_single_epoch_uses_window_start_and_constant_checkpoint_format_is_unchanged(self):
        for args in [(True,2,4),(0,2.0,4),(2,2,4),(-1,2,4),(0,5,4)]:
            with self.assertRaises(ValueError):t.DeclaredGameWindow(*args)
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);paths=self.caches(root);cfg=self.config(epochs=1);w=t.DeclaredGameWindow(0,4,4)
            runtime=t.create(*cfg,enable_v5=True);r=t.train_caches(runtime,paths,learning_rate_window=w)
            self.assertEqual([m['learning_rate'] for m in r['updates']],[cfg[2].learning_rate]*2)
            constant=t.create(*fixture.configurations(),enable_v5=True);p=root/'constant.pt';t.save_checkpoint(p,constant)
            metadata=torch.load(p,weights_only=True)['metadata']
            self.assertNotIn('learning_rate_state',metadata);self.assertNotIn('learning_rate_schedule',metadata['training'])
            legacy=asdict(constant.train_config);legacy.pop('learning_rate_schedule');self.assertEqual(metadata['training'],legacy)
            t.load_checkpoint(p,enable_v5=True)
            self.assertNotIn('learning_rate_schedule',run_fixture.plan().manifest()['train'])

    def test_resume_inside_cache_keeps_short_last_batch_and_epoch_rate(self):
        from pokezero.dataset import _combine_training_batches
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);path=root/'three-rows'
            v5_data.write_cache(path,_combine_training_batches([fixture.batch(index=i) for i in range(3)]))
            model,loss,cfg=self.config(epochs=2);cfg=replace(cfg,batch_size=2)
            window=t.DeclaredGameWindow(0,3,3)
            full=t.create(model,loss,cfg,enable_v5=True);r=t.train_caches(full,[path],learning_rate_window=window)
            self.assertEqual([m['owner_rows'] for m in r['updates']],[2,1,2,1])
            part=t.create(model,loss,cfg,enable_v5=True);t.train_caches(part,[path],max_updates=1,learning_rate_window=window)
            self.assertEqual((part.progress['batch'],part.rows),(1,2))
            p=root/'part.pt';t.save_checkpoint(p,part)
            resumed=t.load_checkpoint(p,enable_v5=True,expected_learning_rate_window=window)
            t.train_caches(resumed,[path],learning_rate_window=window)
            fixture.assert_tree_equal(self,full.model.state_dict(),resumed.model.state_dict())
            fixture.assert_tree_equal(self,full.optimizer.state_dict(),resumed.optimizer.state_dict())
            self.assertEqual((resumed.steps,resumed.rows),(4,6))

    def test_plan_windows_use_declared_roster_and_scheduled_resume_forwards_exact_window(self):
        plan=run_fixture.plan();plan=replace(plan,train=replace(plan.train,learning_rate_schedule='mit-thesis'))
        self.assertEqual(run.learning_rate_window(plan,0),t.DeclaredGameWindow(0,2,4))
        self.assertEqual(run.learning_rate_window(plan,1),t.DeclaredGameWindow(2,4,4))
        calls=[]
        class Trainer(run_fixture.FakeTraining):
            def train(self,runtime,paths,*,max_updates,learning_rate_window):
                calls.append(learning_rate_window)
                if runtime.progress is not None and runtime.progress['complete']:runtime.progress=None
                return super().train(runtime,paths,max_updates=max_updates)
            def load(self,path,**kwargs):
                self.resume_window=kwargs.get('expected_learning_rate_window')
                return super().load(path,**kwargs)
        harness=run_fixture.V5RunTests();trainer=Trainer()
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)/'run'
            with self.assertRaisesRegex(RuntimeError,'durable checkpoint'):
                harness.execute(root,plan,run_fixture.FakeBackend(),trainer,interrupt=lambda iteration,runtime:iteration==0 and runtime.steps==1)
            harness.execute(root,plan,run_fixture.FakeBackend(),trainer,resume=True)
            self.assertEqual(trainer.resume_window,t.DeclaredGameWindow(0,2,4))
            self.assertEqual(set(calls),{t.DeclaredGameWindow(0,2,4),t.DeclaredGameWindow(2,4,4)})
            self.assertEqual(calls[-3:],[t.DeclaredGameWindow(2,4,4)]*3)

if __name__=='__main__':unittest.main()
