"""Checkpoint declarations are provenance, never an export-time relabelling."""
from dataclasses import replace
import importlib.util
import hashlib
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch


class DeclarationValidationTests(unittest.TestCase):
    def test_create_rejects_non_boolean_before_backend_access(self):
        from pokezero import v2_trainer, v5_training
        for trainer in (v2_trainer, v5_training):
            for value in (None, 0, 1, 'true', [], {}):
                with self.subTest(trainer=trainer.__name__, value=value), patch.object(trainer, '_torch') as backend:
                    with self.assertRaisesRegex(ValueError, 'explicit bool self_moves'):
                        trainer.create(None, None, None, enable_v5=True, self_moves=value)
                    backend.assert_not_called()


@unittest.skipUnless(importlib.util.find_spec('torch'), 'requires optional Torch')
class RuntimeDeclarationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        torch.set_num_threads(1)

    def trainers(self):
        from pokezero import v2_trainer, v5_training
        return v2_trainer, v5_training

    def test_both_trainers_only_add_digest_bound_on_metadata(self):
        from pokezero import v2_trainer, v5_training
        from test_v5_training import configurations
        for trainer, metadata in ((v2_trainer, v2_trainer._v5_metadata), (v5_training, v5_training._metadata)):
            with self.subTest(trainer=trainer.__name__):
                runtime = trainer.create(*configurations(), enable_v5=True)
                off = metadata(runtime)
                self.assertIs(runtime.self_moves, False)
                self.assertNotIn('self_moves', off)
                on = metadata(replace(runtime, self_moves=True))
                self.assertIs(on.pop('self_moves'), True)
                self.assertEqual(on, off)
                for value in (None, 0, 1, 'true'):
                    with self.assertRaisesRegex(ValueError, 'explicit bool self_moves'):
                        metadata(replace(runtime, self_moves=value))

    def test_on_roundtrip_and_export_preserve_declaration(self):
        from pokezero import checkpoint_self_moves as enc
        from test_v5_training import configurations, assert_tree_equal
        for trainer in self.trainers():
            with self.subTest(trainer=trainer.__name__), tempfile.TemporaryDirectory() as tmp:
                root=Path(tmp);runtime=trainer.create(*configurations(),enable_v5=True,self_moves=True)
                saved=trainer.save_checkpoint(root/'on.pt',runtime)
                checked=enc.checked_checkpoint(saved['path'],expected_checkpoint_sha256=saved['sha256'])
                self.assertIs(checked.metadata['self_moves'],True)
                resumed=trainer.load_checkpoint(saved['path'],enable_v5=True,expected_self_moves=True)
                self.assertIs(resumed.self_moves,True)
                assert_tree_equal(self,runtime.model.state_dict(),resumed.model.state_dict())
                assert_tree_equal(self,runtime.optimizer.state_dict(),resumed.optimizer.state_dict())
                assert_tree_equal(self,runtime.rng,resumed.rng)
                again=trainer.save_checkpoint(root/'again.pt',resumed)
                self.assertIs(enc.checked_checkpoint(again['path']).metadata['self_moves'],True)
                from gpu_engine import v5_model
                archive=root/'actor.npz'
                exported=v5_model.export(saved['path'],archive,expected_checkpoint_sha256=saved['sha256'],enable_v5=True)
                self.assertIs(exported['self_moves'],True)
                from pokezero import v2_trainer
                public=v2_trainer.export_actor(saved['path'],root/'public.npz')
                self.assertIs(public['self_moves'],True)
                self.assertEqual(archive.read_bytes(),(root/'public.npz').read_bytes())
                with patch.object(trainer,'create') as create:
                    with self.assertRaises(ValueError):trainer.load_checkpoint(saved['path'],enable_v5=True,expected_self_moves=False)
                    create.assert_not_called()
                with self.assertRaises(ValueError):
                    v2_trainer.export_actor(saved['path'],root/'refused.npz',expected_self_moves=False)
                self.assertFalse((root/'refused.npz').exists())

    def test_off_retained_receipt_survives_serialization_and_is_never_inferred(self):
        from pokezero import checkpoint_self_moves as enc, v2_trainer
        from test_v5_training import configurations
        from gpu_engine import v5_model
        for trainer in self.trainers():
            with self.subTest(trainer=trainer.__name__), tempfile.TemporaryDirectory() as tmp:
                root=Path(tmp);runtime=trainer.create(*configurations(),enable_v5=True)
                saved=trainer.save_checkpoint(root/'off.pt',runtime);before=(root/'off.pt').read_bytes()
                # The producer's retained reference crosses a process/JSON boundary.
                reference=json.loads(json.dumps(saved))['encoding_receipt']
                proof=enc.off_from_receipt(reference)
                self.assertNotIn('self_moves',enc.checked_checkpoint(saved['path']).metadata)
                resumed=trainer.load_checkpoint(saved['path'],enable_v5=True,expected_self_moves=False,off_provenance=proof)
                self.assertIs(resumed.self_moves,False)
                self.assertEqual(before,(root/'off.pt').read_bytes())
                result=v2_trainer.export_actor(saved['path'],root/'off.npz',expected_self_moves=False,off_provenance=proof)
                self.assertNotIn('self_moves',result)
                for expectation in (None,False,True):
                    with patch.object(trainer,'create') as create:
                        with self.assertRaises(ValueError):
                            trainer.load_checkpoint(saved['path'],enable_v5=True,expected_self_moves=expectation)
                        create.assert_not_called()
                    target=root/('refuse-'+str(expectation)+'.npz')
                    with self.assertRaises(ValueError):
                        v5_model.export(saved['path'],target,expected_checkpoint_sha256=saved['sha256'],enable_v5=True,
                            self_moves=expectation)
                    self.assertFalse(target.exists())
                with self.assertRaises(ValueError):
                    trainer.load_checkpoint(saved['path'],enable_v5=True,expected_self_moves=True,off_provenance=proof)
                for reference in ({},dict(path=saved['encoding_receipt']['path'],sha256='a'*64),dict(self_moves=False)):
                    with self.assertRaises(ValueError):enc.off_from_receipt(reference)
                other=trainer.save_checkpoint(root/'other.pt',trainer.create(*configurations(seed=99),enable_v5=True))
                with self.assertRaises(ValueError):
                    trainer.load_checkpoint(other['path'],enable_v5=True,off_provenance=proof)

    def test_malformed_flags_digest_disagreement_and_false_claim_refuse_before_restore(self):
        import torch
        from pokezero import checkpoint_self_moves as enc
        from test_v5_training import configurations
        from gpu_engine import v5_model
        for trainer in self.trainers():
            with self.subTest(trainer=trainer.__name__), tempfile.TemporaryDirectory() as tmp:
                root=Path(tmp);saved=trainer.save_checkpoint(root/'on.pt',trainer.create(*configurations(),enable_v5=True,self_moves=True))
                original=enc.checked_checkpoint(saved['path']).payload
                for i,value in enumerate((False,None,0,1,'true',[],{})):
                    payload=dict(original,metadata=dict(original['metadata'],self_moves=value))
                    payload['metadata_sha256']=enc._sha(payload['metadata'])
                    path=root/f'bad-{i}.pt';torch.save(payload,path)
                    digest=hashlib.sha256(path.read_bytes()).hexdigest()
                    with self.subTest(value=value),patch.object(trainer,'create') as create:
                        with self.assertRaises(ValueError):trainer.load_checkpoint(path,enable_v5=True)
                        create.assert_not_called()
                    with self.assertRaises(ValueError):
                        v5_model.export(path,root/f'bad-{i}.npz',expected_checkpoint_sha256=digest,enable_v5=True)
                    self.assertFalse((root/f'bad-{i}.npz').exists())
                payload=dict(original,metadata=dict(original['metadata'],self_moves=False))
                path=root/'broken-digest.pt';torch.save(payload,path)
                with self.assertRaises(ValueError):trainer.load_checkpoint(path,enable_v5=True)
                for expected in (0,1,'true',[],{}):
                    with self.assertRaises(ValueError):trainer.load_checkpoint(saved['path'],enable_v5=True,expected_self_moves=expected)
                off=trainer.save_checkpoint(root/'off.pt',trainer.create(*configurations(),enable_v5=True))
                with self.assertRaises(ValueError):
                    trainer.load_checkpoint(saved['path'],enable_v5=True,off_provenance=enc.off_from_receipt(off['encoding_receipt']))
                with self.assertRaises(ValueError):
                    trainer.load_checkpoint(saved['path'],enable_v5=True,expected_checkpoint_sha256='a'*64)

    def make_run(self,root,*,self_moves=False,steps=0):
        from pokezero import v5_run, v5_training
        from test_v5_run import plan
        from test_v5_training import configurations
        cfg=configurations()
        document=plan().manifest()
        document.update(model=cfg[0].metadata(),loss=cfg[1].metadata(),train=v5_training.train_config_metadata(cfg[2]))
        if self_moves:document['self_moves']=True
        recipe=SimpleNamespace(self_moves=self_moves,model=cfg[0],loss=cfg[1],train=cfg[2],manifest=lambda:document,
            iterations=plan().iterations,games=plan().games)
        store=v5_run.Store(root,recipe)
        runtime=v5_training.create(*cfg,enable_v5=True,self_moves=self_moves)
        runtime.steps=steps
        checkpoint=store.checkpoint(runtime,iteration=-1 if steps==0 else 0)
        archive_blob=b'authored original initializer archive'
        (root/'unused.npz').write_bytes(archive_blob)
        # The relationship itself is needed for initializer verification; no
        # actual game/model evaluation is involved in this synthetic journal.
        store.emit('SNAPSHOT',iteration=checkpoint['iteration'],checkpoint=checkpoint['path'],
            checkpoint_sha256=checkpoint['sha256'],archive='unused.npz',archive_sha256=hashlib.sha256(archive_blob).hexdigest(),
            model_sha256=v5_run._sha(runtime.model_config.metadata()))
        return store,checkpoint

    def test_historical_off_run_evidence_initializer_and_tampering(self):
        from pokezero import checkpoint_self_moves as enc, v5_run, v5_training
        from gpu_engine.v2_bench_trainer import configurations
        for tamper in ('none','declaration','chain','start','checkpoint','on-unlabelled'):
            with self.subTest(tamper=tamper),tempfile.TemporaryDirectory() as tmp:
                root=Path(tmp)/'run';store,record=self.make_run(root)
                if tamper=='declaration':
                    document=json.loads((root/'declaration.json').read_text());document['self_moves']=True
                    (root/'declaration.json').chmod(0o644);(root/'declaration.json').write_text(json.dumps(document))
                elif tamper in ('chain','start','checkpoint','on-unlabelled'):
                    events=store.events
                    if tamper=='chain':events[1]['previous']='f'*64
                    if tamper=='start':events[0]['payload']['declaration_sha256']='f'*64
                    if tamper=='checkpoint':events[1]['payload']['sha256']='f'*64
                    if tamper=='on-unlabelled':
                        doc=json.loads((root/'declaration.json').read_text());doc['self_moves']=True
                        (root/'declaration.json').chmod(0o644);(root/'declaration.json').write_text(json.dumps(doc))
                        events[0]['payload']['declaration_sha256']=enc._sha(doc)
                    previous=None
                    for i,event in enumerate(events):
                        if tamper!='chain':event['previous']=previous
                        event.pop('sha256',None);event['sha256']=enc._sha(event);previous=event['sha256']
                        f=root/'journal'/f'{i:09d}.json';f.chmod(0o644);f.write_text(json.dumps(event))
                if tamper!='none':
                    with self.assertRaises(ValueError):enc.run_checkpoint(root,record['path'],record['sha256'])
                    continue
                _,actual,proof=enc.run_checkpoint(root,record['path'],record['sha256'])
                self.assertIs(actual,False)
                runtime=v5_training.load_checkpoint(root/record['path'],enable_v5=True,off_provenance=proof)
                self.assertIs(runtime.self_moves,False)
                snapshot=v5_run.Snapshot(**store.matching('SNAPSHOT')[0])
                initializer,proof=enc.original_initializer(root,snapshot)
                _,_,_,_,provenance=configurations(root/record['path'],expected_self_moves=True,
                    off_provenance=proof,weights_only_initializer=initializer)
                self.assertIs(provenance['self_moves'],False)
                with self.assertRaises(ValueError):
                    v5_training.load_checkpoint(root/record['path'],enable_v5=True,expected_self_moves=True,off_provenance=proof)
                with self.assertRaises(ValueError):enc.original_initializer(root,replace(snapshot,iteration=0))
        with tempfile.TemporaryDirectory() as tmp:
            store,record=self.make_run(Path(tmp)/'trained',steps=1)
            snapshot=replace(v5_run.Snapshot(**store.matching('SNAPSHOT')[0]),iteration=-1)
            with self.assertRaises(ValueError):enc.original_initializer(store.path,snapshot)

    def test_store_rejects_runtime_plan_disagreement_before_write(self):
        from pokezero import v5_training
        from test_v5_training import configurations
        with tempfile.TemporaryDirectory() as tmp:
            store,_=self.make_run(Path(tmp)/'run')
            runtime=v5_training.create(*configurations(),enable_v5=True,self_moves=True)
            with patch.object(v5_training,'save_checkpoint') as save:
                with self.assertRaises(ValueError):store.checkpoint(runtime,iteration=1)
                save.assert_not_called()

    def test_bench_runtime_creation_keeps_resolved_encoding(self):
        from pokezero import v2_trainer
        from gpu_engine import v2_bench_trainer as bench
        from test_v5_training import configurations
        class ReachedCreate(Exception):pass
        model,loss,train=configurations()
        args=SimpleNamespace(device='cpu',order='ordered',check_interval=1,optimizer='single')
        for value in (False,True):
            for call in (lambda:bench.parity(None,model,loss,train,{},1,'cpu',True,self_moves=value),
                         lambda:bench.measure(args,None,model,loss,train,{},1,'eager',None,self_moves=value)):
                with self.subTest(self_moves=value),patch.object(v2_trainer,'create',side_effect=ReachedCreate) as create:
                    with self.assertRaises(ReachedCreate):call()
                    self.assertIs(create.call_args.kwargs['self_moves'],value)

    def test_fused_fresh_and_resume_assert_the_plan_before_training(self):
        from pokezero import checkpoint_self_moves as enc, v2_trainer, v5_run
        from gpu_engine import v2_fused
        class ReachedCreate(Exception):pass
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);store,record=self.make_run(root/'reference')
            plan=SimpleNamespace(**dict(store.plan.__dict__,self_moves=True))
            document=dict(store.plan.manifest(),self_moves=True);plan.manifest=lambda:document
            context=dict(plan=plan,preparation_seconds=0,declared={})
            args=['--config','unused.json','--reference-run',str(store.path),'--out',str(root/'out'),
                '--torch-memory-fraction','0']
            with patch.object(v2_fused,'prepare',return_value=context),patch.object(v2_trainer,'create',side_effect=ReachedCreate) as create:
                with self.assertRaises(ReachedCreate):v2_fused.main(args)
                self.assertIs(create.call_args.kwargs['self_moves'],True)
            snapshot=v5_run.Snapshot(**store.matching('SNAPSHOT')[0])
            state=dict(snapshots=[snapshot.__dict__],reports=[],row_order='file',
                declaration_sha256=v5_run._sha(document),checkpoint_encoding_receipts={})
            (root/'out/state.json').write_text(json.dumps(state))
            # A trained resume receives the ON expectation, never the initializer exception.
            with patch.object(v2_fused,'prepare',return_value=context),patch.object(v2_trainer,'load_checkpoint',side_effect=ReachedCreate) as load:
                with self.assertRaises(ReachedCreate):v2_fused.main(args)
                self.assertIs(load.call_args.kwargs['expected_self_moves'],True)
                self.assertNotIn('weights_only_initializer',load.call_args.kwargs)
                self.assertIsNone(load.call_args.kwargs['off_provenance'])


class ChampionDeclarationTests(unittest.TestCase):
    def test_distinct_champion_schema_requires_actual_source_bound_digest(self):
        from pokezero import checkpoint_self_moves as enc, recorded_mirror_roots
        blob=b'controlled synthetic champion bytes';digest=hashlib.sha256(blob).hexdigest()
        original=recorded_mirror_roots.CHAMPION_CONTRACT
        with patch.object(recorded_mirror_roots,'CHAMPION_CONTRACT',dict(original,checkpoint_sha256=digest)):
            self.assertIs(enc.resolve_champion_self_moves(blob,{'model_config':{}}),False)
            self.assertIs(enc.resolve_champion_self_moves(blob,{'model_config':{'self_moves':False}}),False)
            for value in (True,None,0,1,'true',[],{}):
                with self.assertRaises(ValueError):enc.resolve_champion_self_moves(blob,{'model_config':{'self_moves':value}})
            with self.assertRaises(ValueError):enc.resolve_champion_self_moves(blob+b'x',{'model_config':{}})
        self.assertIs(enc.resolve_champion_self_moves(b'on',{'model_config':{'self_moves':True}}),True)
        with self.assertRaises(ValueError):enc.resolve_champion_self_moves(b'off',{'model_config':{'self_moves':False}})
        with self.assertRaises(ValueError):enc.TrustedOffProvenance(None,digest,None,'caller False')
