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

    def test_forward_root_selection_requires_explicit_repeated_fixture(self):
        from gpu_engine.benchmark_forward import root_indices
        with self.assertRaises(ValueError):root_indices(1024,1024)
        with self.assertRaises(ValueError):root_indices(100001,200000)
        pairs=root_indices(1024,1024,fixture=True)
        self.assertEqual(len(self.np.unique(pairs,axis=0)),512)
        pairs=root_indices(8192,200000)
        self.assertEqual(len(self.np.unique(pairs,axis=0)),8192)

    def test_moe_partition_matches_shared_dense_expert_and_isolates_roles(self):
        from dataclasses import replace
        c=replace(self.c,moe=True)
        p=dict(self.p)
        for layer in range(c.layers):
            prefix=f'encoder.layers.{layer}.'
            for key in ('linear1.weight','linear1.bias','linear2.weight','linear2.bias'):
                value=p.pop(prefix+key)
                for expert in range(3):p[prefix+f'experts.{expert}.'+key]=value
        expected=self.call()
        actual=self.m.forward(p,self.num,self.cat,self.types,self.mask,c)
        for a,b in zip(expected,actual):self.np.testing.assert_allclose(a,b,atol=1e-6)
        # With one layer the field expert cannot change policy action queries.
        key='encoder.layers.0.experts.2.linear2.bias'
        changed=self.m.forward(dict(p,**{key:p[key]+1}),self.num,self.cat,self.types,self.mask,c)
        self.np.testing.assert_array_equal(actual.policy_logits,changed.policy_logits)
        self.assertFalse(self.np.array_equal(actual.value,changed.value))

    def test_rms_attention_value_heads_appended_tokens_and_bf16_are_finite(self):
        from dataclasses import replace
        rng=self.np.random.default_rng(704)
        for head in ('mlp','hl_gauss'):
            c=replace(self.c,tokens=29,norm='rms',moe=True,expert_groups=(),value_head=head,value_hidden=12)
            p={k:self.j.asarray(rng.normal(0,.05,s).astype(self.np.float32)) for k,s in self.m.shapes(c).items()}
            self.m.validate_weights(p,c)
            n=self.j.pad(self.num,((0,0),(0,0),(0,6),(0,0)))
            k=self.j.pad(self.cat,((0,0),(0,0),(0,6),(0,0)))
            t=self.j.pad(self.types,((0,0),(0,0),(0,6)))
            a=self.j.pad(self.mask,((0,0),(0,0),(0,6)),constant_values=True)
            out=self.jax.jit(lambda n,k,t,a:self.m.forward(p,n,k,t,a,c,compute_dtype='bfloat16'))(n,k,t,a)
            self.assertEqual(out.policy_logits.shape,(2,2,9))
            for value in out:
                self.assertEqual(value.dtype,self.j.float32)
                self.assertTrue(self.np.isfinite(value).all())
            if head=='hl_gauss':self.assertTrue(self.np.all(self.np.abs(out.value)<=1))
        for values in ({'norm':'unknown'},{'value_head':'unknown'},{'expert_groups':(0,)*22},
                {'moe':True,'expert_groups':(0,)*23}):
            with self.assertRaises(ValueError):replace(self.c,**values)
        with self.assertRaises(ValueError):self.m.forward(self.p,self.num,self.cat,self.types,self.mask,self.c,compute_dtype='float16')
