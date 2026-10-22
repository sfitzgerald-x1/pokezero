"""Authored forward shape/masking checks, not actual-champion or game parity."""
import importlib.util
from pathlib import Path
import tempfile
import unittest


@unittest.skipUnless(importlib.util.find_spec('jax'),'requires isolated spike JAX')
class ChampionTests(unittest.TestCase):
    def setUp(self):
        import jax
        import jax.numpy as j
        import numpy as np
        from gpu_engine import champion as m
        self.jax,self.j,self.np,self.m=jax,j,np,m
        self.c=m.Config(numeric=4,categorical=3,vocabulary=12,token_types=6,width=8,layers=1,heads=2,feedforward=16)
        rng=np.random.default_rng(701)
        self.p={k:j.asarray(rng.normal(0,.05,s).astype(np.float32)) for k,s in m.shapes(self.c).items()}
        self.num=j.asarray(rng.normal(0,.1,(2,2,23,4)).astype(np.float32))
        self.cat=j.zeros((2,2,23,3),j.int32);self.types=j.zeros((2,2,23),j.int32)
        self.mask=j.ones((2,2,23),bool)

    def call(self,num=None,cat=None,types=None,mask=None):
        return self.m.forward(self.p,self.num if num is None else num,self.cat if cat is None else cat,
            self.types if types is None else types,self.mask if mask is None else mask,self.c)

    def test_shapes_finite_and_eager_jit_batch_prefix(self):
        j=self.j;a=self.call();b=self.jax.jit(lambda n,c,t,m:self.call(n,c,t,m))(self.num,self.cat,self.types,self.mask)
        small=self.call(self.num[:1],self.cat[:1],self.types[:1],self.mask[:1])
        self.assertEqual([x.shape for x in a],[(2,2,9),(2,2),(2,2,9)])
        for x,y,z in zip(a,b,small):
            self.np.testing.assert_allclose(x,y,atol=1e-6,rtol=1e-6)
            self.np.testing.assert_allclose(x[:1],z,atol=1e-6,rtol=1e-6)
            self.assertTrue(bool(j.all(j.isfinite(x))))

    def test_masked_keys_and_pool_ignore_padding_not_action_queries(self):
        mask=self.mask.at[...,1].set(False)
        a=self.call(mask=mask);num=self.num.at[...,1,:].set(1000)
        b=self.call(num=num,mask=mask)
        for x,y in zip(a,b):self.np.testing.assert_allclose(x,y,atol=1e-6,rtol=1e-6)

    def test_category_and_type_clamp_matches_production(self):
        a=self.call(cat=self.cat.at[...,0].set(-1),types=self.types.at[...,0].set(-1))
        b=self.call()
        for x,y in zip(a,b):self.np.testing.assert_allclose(x,y,atol=1e-6)
        a=self.call(cat=self.cat+999,types=self.types+999)
        b=self.call(cat=self.cat+11,types=self.types+5)
        for x,y in zip(a,b):self.np.testing.assert_allclose(x,y,atol=1e-6)

    def test_strict_config_shapes_dtype_keys_and_nonfinite_rejection(self):
        self.m.validate_weights(self.p,self.c)
        for p in ({},dict(self.p,unexpected=self.j.zeros(1)),
                dict(self.p,**{'policy_head.bias':self.j.array([self.j.nan])})):
            with self.assertRaises(ValueError):self.m.validate_weights(p,self.c)
        with self.assertRaises(ValueError):self.m.Config(width=7,heads=2)
        with self.assertRaises(ValueError):self.call(num=self.num[...,:3])

    def test_temperature_one_masking_terminal_sentinel_and_prefix(self):
        j=self.j;out=self.call();legal=j.zeros((2,2,9),bool).at[0,0,3].set(True).at[0,1,7].set(True)
        keys=self.jax.random.split(self.jax.random.PRNGKey(703),4).reshape((2,2,2))
        actions=self.m.sample(out,legal,keys)
        self.assertEqual(actions.tolist(),[[3,7],[-1,-1]])
        again=self.jax.jit(self.m.sample)(out,legal,keys)
        self.assertTrue(bool(j.array_equal(actions,again)))

    def test_missing_activation_is_production_legacy_linear_and_unsupported_rejected(self):
        cfg=dict(window_size=1,numeric_feature_count=4,categorical_feature_count=3,token_count=23,
            categorical_vocab_size=12,token_type_vocab_size=6,embedding_dim=8,transformer_layers=1,
            attention_heads=2,feedforward_dim=16)
        self.assertFalse(self.m.config_from_checkpoint(cfg).tanh_value)
        self.assertTrue(self.m.config_from_checkpoint(dict(cfg,value_activation='tanh')).tanh_value)
        for values in ({'window_size':2},{'temporal_aggregator':'gru'},{'value_activation':'sigmoid'},
                {'value_head_hidden_layers':2}):
            with self.assertRaises(ValueError):self.m.config_from_checkpoint(dict(cfg,**values))

    def test_snapshot_digest_stays_bound_after_pathname_replacement(self):
        import hashlib
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'checkpoint';path.write_bytes(b'original checkpoint bytes')
            blob,digest=self.m.snapshot(path)
            path.write_bytes(b'replacement bytes')
            self.assertEqual(blob,b'original checkpoint bytes')
            self.assertEqual(digest,hashlib.sha256(blob).hexdigest())
            self.assertNotEqual(digest,self.m.snapshot(path)[1])
