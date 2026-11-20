"""Digest-matching archives must not coerce malformed encoding declarations."""
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np


@unittest.skipUnless(importlib.util.find_spec('torch') and importlib.util.find_spec('jax'), 'requires Torch/JAX runtime')
class ArchiveDeclarationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        torch.set_num_threads(1)

    def fixture(self,root,kind,on=False):
        from pokezero import checkpoint_self_moves as enc
        from gpu_engine import v5_model,champion,champion_evaluator
        if kind=='v5':
            from pokezero import v5_training
            from test_v5_training import configurations
            runtime=v5_training.create(*configurations(),enable_v5=True,self_moves=on)
            saved=v5_training.save_checkpoint(root/'checkpoint.pt',runtime)
            receipt=v5_model.export(root/'checkpoint.pt',root/'actor.npz',expected_checkpoint_sha256=saved['sha256'],
                enable_v5=True,off_provenance=None if on else enc.off_from_receipt(saved['encoding_receipt']))
            def load(path,digest):
                return v5_model.load(path,enable_v5=True,expected_archive_sha256=digest,
                    expected_checkpoint_sha256=saved['sha256'],expected_category_vocabulary_sha256=runtime.model_config.category_vocabulary_sha256,
                    expected_label_vocabulary_sha256=runtime.model_config.label_vocabulary.sha256)[2]
        else:
            from test_champion_checkpoint_provenance import ChampionWriterTests
            _,_,model,result=ChampionWriterTests().fixture(self_moves=on)
            reference=enc.save_champion_transformer(root/'checkpoint.pt',model,result=result)
            receipt=champion.export(root/'checkpoint.pt',root/'actor.npz',off_provenance=None if on else enc.off_from_receipt(reference))
            def load(path,digest):
                return champion_evaluator.load(path,expected_archive_sha256=digest,
                    expected_checkpoint_sha256=receipt['checkpoint_sha256'],precision='float32').meta
        with np.load(root/'actor.npz',allow_pickle=False) as source:arrays={k:source[k] for k in source.files}
        return arrays,load

    def write(self,path,arrays,metadata):
        changed=dict(arrays,metadata_json=np.frombuffer(json.dumps(metadata,sort_keys=True).encode(),np.uint8))
        np.savez_compressed(path,**changed)
        return hashlib.sha256(path.read_bytes()).hexdigest()

    def test_matching_digest_does_not_admit_non_boolean_flags_or_metadata(self):
        import jax.numpy as jnp
        for kind in ('v5','champion'):
            with self.subTest(kind=kind),tempfile.TemporaryDirectory() as tmp:
                root=Path(tmp);arrays,load=self.fixture(root,kind);metadata=json.loads(arrays['metadata_json'].tobytes())
                documents=[dict(metadata,self_moves=x) for x in (None,0,1,'','false',[],{})]+[None,[],False]
                for i,document in enumerate(documents):
                    path=root/f'bad-{i}.npz';digest=self.write(path,arrays,document)
                    with self.subTest(metadata=document),patch.object(jnp,'asarray',side_effect=AssertionError('device arrays allocated')) as convert:
                        with self.assertRaises(ValueError):load(path,digest)
                        convert.assert_not_called()

    def test_missing_false_and_true_preserve_valid_archive_declarations(self):
        for kind in ('v5','champion'):
            for on in (False,True):
                with self.subTest(kind=kind,on=on),tempfile.TemporaryDirectory() as tmp:
                    root=Path(tmp);arrays,load=self.fixture(root,kind,on=on)
                    path=root/'actor.npz';digest=hashlib.sha256(path.read_bytes()).hexdigest();metadata=load(path,digest)
                    self.assertIs(metadata.get('self_moves',False),on)
                    if not on:
                        self.assertNotIn('self_moves',metadata)
                        explicit=dict(json.loads(arrays['metadata_json'].tobytes()),self_moves=False)
                        path=root/'explicit-off.npz';digest=self.write(path,arrays,explicit)
                        self.assertIs(load(path,digest)['self_moves'],False)

    def test_admission_helpers_and_collection_refuse_before_adapter_or_forward(self):
        from pokezero.checkpoint_self_moves import archive_self_moves
        from gpu_engine import private_self_moves as declarations,v5_collection,v5_model,champion_evaluator
        record=dict(catalog=SimpleNamespace(masks=SimpleNamespace(self_moves=False)))
        for value in (None,0,1,'','false',[],{}):
            metadata=dict(self_moves=value)
            with self.subTest(value=value):
                with self.assertRaises(ValueError):declarations.refuse_unthreaded(metadata,context='unit')
                with self.assertRaises(ValueError):declarations.refuse_search_serving(meta=metadata,context='unit')
                with self.assertRaises(ValueError):
                    champion_evaluator.ChampionEvaluator({},SimpleNamespace(tokens=23),
                        dict(metadata,weights_archive_sha256='a'*64,checkpoint_sha256='b'*64))
                with patch.object(v5_collection.driver,'_checked',return_value=record),\
                        patch.object(v5_collection.driver,'_checked_witness'),\
                        patch.object(v5_model,'load',return_value=({},None,metadata)),\
                        patch.object(v5_collection.v5_observation,'compile_adapter',side_effect=AssertionError('adapter compiled')) as adapter:
                    with self.assertRaises(ValueError):
                        v5_collection.load_actor(object(),'unused.npz',expected_archive_sha256='a'*64,
                            expected_checkpoint_sha256='b'*64,expected_category_vocabulary_sha256='c'*64,
                            expected_label_vocabulary_sha256='d'*64,enable_v5=True)
                    adapter.assert_not_called()
        self.assertIs(archive_self_moves({}),False)
        declarations.refuse_unthreaded(None,context='unit')
        declarations.refuse_search_serving(context='unit')
