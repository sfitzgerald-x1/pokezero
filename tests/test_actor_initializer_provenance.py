"""OFF→ON actors require the original untrained checkpoint and exact snapshot."""
from dataclasses import replace
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch


@unittest.skipUnless(importlib.util.find_spec('torch') and importlib.util.find_spec('jax'), 'requires Torch/JAX runtime')
class ActorInitializerTests(unittest.TestCase):
    def setup_run(self,root,*,steps=0):
        from test_self_moves_checkpoint import RuntimeDeclarationTests
        from pokezero.v5_run import Snapshot
        store,record=RuntimeDeclarationTests().make_run(root,steps=steps)
        snapshot=Snapshot(**store.matching('SNAPSHOT')[0])
        plan=SimpleNamespace(model=store.plan.model,self_moves=True)
        return store,record,snapshot,plan

    def load_actors(self,store,snapshot,plan,events=None):
        from gpu_engine import v2_bench_collector,v2_fused,v5_model
        params=object();metadata=dict(checkpoint_sha256=snapshot.checkpoint_sha256,
            weights_archive_sha256=snapshot.archive_sha256)
        args=SimpleNamespace(reference_run=str(store.path))
        with patch.object(v5_model,'load',return_value=(params,plan.model,metadata)):
            return (v2_bench_collector.actor(args,plan,store.events if events is None else events),
                v2_fused.load_actor(plan,snapshot,store.path))

    def test_verified_initializer_allows_bench_and_fused_and_binds_both_digests(self):
        from pokezero import checkpoint_self_moves as enc
        from gpu_engine.private_self_moves import check_run_actor
        with tempfile.TemporaryDirectory() as tmp:
            store,record,snapshot,plan=self.setup_run(Path(tmp)/'run')
            loaded,_=self.load_actors(store,snapshot,plan)
            self.assertEqual(loaded[0],snapshot)
            initializer,_=enc.original_initializer(store.path,snapshot)
            meta=dict(checkpoint_sha256=record['sha256'],weights_archive_sha256=snapshot.archive_sha256)
            check_run_actor(meta,plan_self_moves=True,iteration=-1,context='test',initializer=initializer)
            for changed in (dict(meta,checkpoint_sha256='a'*64),dict(meta,weights_archive_sha256='b'*64)):
                with self.assertRaises(ValueError):
                    check_run_actor(changed,plan_self_moves=True,iteration=-1,context='test',initializer=initializer)
            for iteration,proof in ((-1,None),(-1,object()),(0,initializer)):
                with self.assertRaises(ValueError):
                    check_run_actor(meta,plan_self_moves=True,iteration=iteration,context='test',initializer=proof)

    def test_missing_trained_and_substituted_initializer_evidence_refuses_collection(self):
        from gpu_engine import v2_bench_collector,v2_fused,v5_model
        from pokezero import checkpoint_self_moves as enc
        for mode in ('checkpoint','declaration','archive','trained','substituted-snapshot'):
            with self.subTest(mode=mode),tempfile.TemporaryDirectory() as tmp:
                store,record,snapshot,plan=self.setup_run(Path(tmp)/'run',steps=1 if mode=='trained' else 0)
                if mode in ('checkpoint','declaration','archive'):
                    path={'checkpoint':store.path/record['path'],'declaration':store.path/'declaration.json',
                        'archive':store.path/snapshot.archive}[mode]
                    path.rename(str(path)+'.preserved')
                events=copy.deepcopy(store.events)
                if mode=='trained':
                    previous=None
                    for i,event in enumerate(events):
                        if event['kind'] in ('CHECKPOINT','SNAPSHOT'):event['payload']['iteration']=-1
                        event['previous']=previous;event.pop('sha256',None);event['sha256']=enc._sha(event);previous=event['sha256']
                        path=store.path/'journal'/f'{i:09d}.json';path.chmod(0o644);path.write_text(json.dumps(event))
                    snapshot=replace(snapshot,iteration=-1)
                if mode=='substituted-snapshot':
                    blob=b'substituted actor archive';(store.path/'substituted.npz').write_bytes(blob)
                    snapshot=replace(snapshot,archive='substituted.npz',archive_sha256=hashlib.sha256(blob).hexdigest())
                    for event in events:
                        if event['kind']=='SNAPSHOT':event['payload']=dict(snapshot.__dict__)
                metadata=dict(checkpoint_sha256=snapshot.checkpoint_sha256,weights_archive_sha256=snapshot.archive_sha256)
                with patch.object(v5_model,'load',return_value=(object(),plan.model,metadata)):
                    with self.assertRaisesRegex(ValueError,'verified original initializer'):
                        v2_bench_collector.actor(SimpleNamespace(reference_run=str(store.path)),plan,events)
                    with self.assertRaisesRegex(ValueError,'verified original initializer'):
                        v2_fused.load_actor(plan,snapshot,store.path)
