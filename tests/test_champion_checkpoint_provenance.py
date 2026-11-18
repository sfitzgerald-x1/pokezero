"""Fresh champion-schema writers retain OFF evidence outside checkpoint bytes."""
from dataclasses import replace
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch


@unittest.skipUnless(importlib.util.find_spec('torch') and importlib.util.find_spec('jax'), 'requires Torch/JAX runtime')
class ChampionWriterTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        torch.set_num_threads(1)

    def fixture(self,*,self_moves=False):
        from pokezero import neural_policy as neural
        from test_b4_fidelity_canary import model_config
        config=replace(model_config(),embedding_dim=8,transformer_layers=1,attention_heads=2,feedforward_dim=16,
            self_moves=self_moves)
        model=neural.EntityTokenTransformerPolicy(config)
        result=neural.TransformerTrainingResult(model_config=config,
            training_config=neural.TransformerTrainingConfig(window_size=1),epochs=())
        return neural,config,model,result

    def test_transformer_and_reply_off_writers_preserve_bytes_and_export(self):
        import torch
        from pokezero import checkpoint_self_moves as enc
        from gpu_engine import champion
        neural,config,model,result=self.fixture()
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);prior=root/'prior';current=root/'current';prior.mkdir();current.mkdir()
            neural.save_transformer_checkpoint(prior/'checkpoint.pt',model,result=result)
            reference=enc.save_champion_transformer(current/'checkpoint.pt',model,result=result)
            self.assertEqual((prior/'checkpoint.pt').read_bytes(),(current/'checkpoint.pt').read_bytes())
            proof=enc.off_from_receipt(json.loads(json.dumps(reference)))
            self.assertIsNone(proof.metadata_sha256)
            exported=champion.export(current/'checkpoint.pt',current/'actor.npz',off_provenance=proof)
            self.assertNotIn('self_moves',exported)
            with self.assertRaises(ValueError):champion.export(current/'checkpoint.pt',current/'unsupported.npz')
            self.assertFalse((current/'unsupported.npz').exists())
            payload=torch.load(current/'checkpoint.pt',weights_only=True)
            payload['model_config']=dict(payload['model_config'],policy_id='authored-reply')
            payload['fp_reply']=dict(policy_only=True,view='b')
            torch.save(payload,prior/'reply.pt')
            reference=enc.save_champion_payload(current/'reply.pt',payload,model_config=config)
            self.assertEqual((prior/'reply.pt').read_bytes(),(current/'reply.pt').read_bytes())
            proof=enc.off_from_receipt(reference)
            exported=champion.export(current/'reply.pt',current/'reply.npz',extra=payload['fp_reply'],off_provenance=proof)
            self.assertNotIn('self_moves',exported)
            self.assertEqual(exported['extra'],payload['fp_reply'])
            with self.assertRaises(ValueError):
                champion.export(current/'reply.pt',current/'relabel.npz',self_moves=True,off_provenance=proof)
            with self.assertRaises(FileExistsError):enc.save_champion_payload(current/'reply.pt',payload,model_config=config)
            with self.assertRaises(FileExistsError):enc.save_champion_transformer(current/'checkpoint.pt',model,result=result)

    def test_on_writer_and_malformed_or_unsupported_evidence(self):
        import torch
        from pokezero import checkpoint_self_moves as enc
        from gpu_engine import champion
        neural,config,model,result=self.fixture(self_moves=True)
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);reference=enc.save_champion_transformer(root/'on.pt',model,result=result)
            self.assertIs(champion.export(root/'on.pt',root/'on.npz')['self_moves'],True)
            with self.assertRaises(ValueError):enc.off_from_receipt(reference)
            payload=torch.load(root/'on.pt',weights_only=True)
            for i,value in enumerate((None,0,1,'true',False,[],{})):
                changed=dict(payload,model_config=dict(payload['model_config'],self_moves=value))
                with self.subTest(value=value),self.assertRaises(ValueError):
                    enc.save_champion_payload(root/f'bad-{i}.pt',changed,model_config=config)
                self.assertFalse((root/f'bad-{i}.pt').exists())
            with self.assertRaises(ValueError):
                enc._writer_receipt(root/'unknown.pt',checkpoint_sha256='a'*64,metadata_sha256=None,
                    self_moves=False,checkpoint_schema='unsupported')
            with self.assertRaises(ValueError):
                enc._writer_receipt(root/'wrong-v5.pt',checkpoint_sha256='a'*64,metadata_sha256=None,self_moves=False)

    def test_transformer_writer_binds_actual_unwrapped_model_before_writing(self):
        from pokezero import checkpoint_self_moves as enc
        from gpu_engine import champion
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            for actual_on,result_on in ((True,False),(False,True)):
                for wrapped in (False,True):
                    neural,config,model,result=self.fixture(self_moves=actual_on)
                    result=replace(result,model_config=replace(config,self_moves=result_on))
                    target=root/f'mismatch-{actual_on}-{wrapped}.pt'
                    actual=SimpleNamespace(module=model) if wrapped else model
                    with self.subTest(actual_on=actual_on,result_on=result_on,wrapped=wrapped),\
                            patch.object(neural,'save_transformer_checkpoint') as writer:
                        with self.assertRaisesRegex(ValueError,'differs from runtime configuration'):
                            enc.save_champion_transformer(target,actual,result=result)
                        writer.assert_not_called()
                    self.assertFalse(target.exists())
                    self.assertFalse(Path(str(target)+'.encoding.json').exists())
            for actual_on in (False,True):
                for wrapped in (False,True):
                    _,_,model,result=self.fixture(self_moves=actual_on)
                    actual=SimpleNamespace(module=model) if wrapped else model
                    target=root/f'matching-{actual_on}-{wrapped}.pt'
                    reference=enc.save_champion_transformer(target,actual,result=result)
                    proof=None if actual_on else enc.off_from_receipt(reference)
                    exported=champion.export(target,target.with_suffix('.npz'),off_provenance=proof)
                    self.assertIs(exported.get('self_moves',False),actual_on)

    def test_receipt_schema_and_checkpoint_binding_cannot_be_substituted(self):
        from pokezero import checkpoint_self_moves as enc
        from gpu_engine import champion
        _,config,model,result=self.fixture()
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);reference=enc.save_champion_transformer(root/'off.pt',model,result=result)
            document=json.loads(Path(reference['path']).read_bytes())
            for name,change in [('schema',dict(checkpoint_schema='unsupported')),
                    ('metadata',dict(metadata_sha256='a'*64)),('flag',dict(self_moves=0))]:
                path=root/(name+'.json');blob=json.dumps(dict(document,**change)).encode();path.write_bytes(blob)
                with self.subTest(name=name),self.assertRaises(ValueError):
                    enc.off_from_receipt(dict(path=str(path),sha256=hashlib.sha256(blob).hexdigest()))
            # A V5 receipt with a matching file SHA cannot establish the distinct
            # champion schema's OFF provenance (it binds a V5 metadata digest).
            wrong=enc._writer_receipt(root/'fixture-v5.pt',checkpoint_sha256=document['checkpoint_sha256'],
                metadata_sha256='b'*64,self_moves=False)
            with self.assertRaises(ValueError):
                champion.export(root/'off.pt',root/'cross-schema.npz',off_provenance=enc.off_from_receipt(wrong))
            other=enc.save_champion_transformer(root/'other.pt',model,result=replace(result,model_config=replace(config,policy_id='other')))
            with self.assertRaises(ValueError):
                champion.export(root/'off.pt',root/'wrong-checkpoint.npz',off_provenance=enc.off_from_receipt(other))

    def test_canary_and_reply_export_calls_forward_original_writer_evidence(self):
        import numpy as np
        from pokezero import checkpoint_self_moves as enc
        from gpu_engine import champion,fp_reply_train
        from gpu_engine import b4_canary_collection,b4_device_canary_collection
        class ReachedExport(Exception):pass
        _,config,model,result=self.fixture()
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);reference=enc.save_champion_transformer(root/'off.pt',model,result=result);proof=enc.off_from_receipt(reference)
            for i,module in enumerate((b4_canary_collection,b4_device_canary_collection)):
                backend=object.__new__(module.Backend)
                with self.subTest(backend=module.__name__),patch.object(champion,'export',side_effect=ReachedExport) as export:
                    with self.assertRaises(ReachedExport):
                        backend.collect(None,root/'off.pt',root/f'collect-{i}',dataset_config=None,on_end=None,
                            check_progress=None,off_provenance=proof)
                    self.assertIs(export.call_args.kwargs['off_provenance'],proof)
            with patch.object(fp_reply_train,'predict',return_value=np.ones((1,9))/9),\
                    patch.object(champion,'export',side_effect=ReachedExport) as export:
                with self.assertRaises(ReachedExport):
                    import torch
                    fp_reply_train.archive_parity(root/'off.pt',root/'reply.npz',{},None,np.array([0]),torch,{},off_provenance=proof)
                self.assertIs(export.call_args.kwargs['off_provenance'],proof)
