"""Real retained-encoder evidence input, default-off identity and weighted v5 tail."""
import hashlib
import json
import unittest
import numpy as np

from gpu_engine import v4_encoder as E,set_posterior_features as F,v5_observation as V5
from tests import test_gpu_engine_v4_encoder as fixtures


class JointFeatures(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        fixtures.V4EncoderTests.setUpClass()
        cls.base=fixtures.V4EncoderTests();cls.base.shared_jit=True
        cls.on=fixtures.V4EncoderTests();cls.on.shared_jit=True
        cls.on.catalog=E.compile_encoder(cls.base.source,cls.base.inputdex,cls.base.vocab,
            checkpoint_sha256='a'*64,feature_masks=dict(cls.base.masks,set_posterior_narrowing=True))
        cls.off_state=cls.base.ingest([*cls.base.public.prefix,'|turn|1'])
        cls.on_state=cls.on.ingest([*cls.base.public.prefix,'|turn|1'])
        cls.owner=0;cls.game='1'*64
        rows={};prior={};source=cls.on.catalog.source
        for slot,sid in enumerate(np.asarray(cls.on_state.beliefs.species)[0,0]):
            if not sid:continue
            from pokezero.randbat import canonical_gen3_randbat_species_id
            name=canonical_gen3_randbat_species_id(source.species_names[sid]);variants=source.variants[sid]
            words=np.asarray(cls.on_state.beliefs.hard)[0,0,slot]
            mask=int(words[0])|(int(words[1])<<32)
            weights=tuple((i+1)*int(bool(mask&(1<<i)))for i in range(len(variants)))
            row=dict(variant_ids=tuple(v.variant_id for v in variants),support=tuple(w>0 for w in weights),weights=weights)
            rows[name]=row
            prior[name]=dict(row,support=(True,)*len(variants),weights=(1,)*len(variants))
        cls.payload=dict(schema='gen3-joint-features-v1',source_hash=source.source_hash,game_sha256=cls.game,
            rows=rows,prior_rows=prior,provenance='fixture')
        cls.payload['payload_sha256']=hashlib.sha256(json.dumps(cls.payload,sort_keys=True,separators=(',',':')).encode()).hexdigest()
        cls.inputs=F.compile_inputs(cls.payload,cls.on.catalog,cls.on_state,owner=0,game_sha256=cls.game)

    def encode(self,fixture,state,post=None):
        own=E.parse_own_requests(fixture.catalog,[fixture.request(0)],owner_sides=[0])
        return E.encode(state,own,fixture.catalog,observer_sides=fixture.j.array([0],fixture.j.int32),set_posterior=post)

    def test_flag_off_preserves_every_encoded_leaf(self):
        plain=self.encode(self.base,self.off_state)
        supplied=self.encode(self.base,self.off_state,self.inputs)
        for a,b in zip(self.base.jax.tree.leaves(plain),self.base.jax.tree.leaves(supplied)):
            np.testing.assert_array_equal(a,b)
        explicit=E.compile_encoder(self.base.source,self.base.inputdex,self.base.vocab,
            checkpoint_sha256='a'*64,feature_masks=dict(self.base.masks,set_posterior_narrowing=False))
        np.testing.assert_array_equal(explicit.binding,self.base.catalog.binding)

    def test_flag_on_uses_weights_and_requires_bound_inputs(self):
        with self.assertRaisesRegex(ValueError,'source-bound set posterior'):
            self.encode(self.on,self.on_state)
        encoded=self.encode(self.on,self.on_state,self.inputs)
        self.assertEqual(encoded.observation.admitted.tolist(),[True])
        np.testing.assert_array_equal(encoded.posterior_weights,self.inputs.weights)
        self.assertFalse(np.array_equal(self.on.catalog.binding,self.base.catalog.binding))
        from pokezero import v5_contract as C
        from pokezero.v5_model import V5ModelConfig
        from gpu_engine.private_team import vocabulary_binding
        source=self.base.public.source;census=C.source_variant_census(source)
        labels=C.LabelVocabulary(census['source_rows_sha256'],tuple(self.on.catalog.moves.names[1:]),census['species_variants'])
        category=np.asarray(vocabulary_binding(self.on.catalog.vocabulary),dtype='>u4').tobytes().hex()
        config=V5ModelConfig(self.on.catalog.vocabulary.size,category,labels,width=8,layers=1,
            attention_heads=2,expert_width=8,value_hidden=(8,8))
        adapter=V5.compile_adapter(self.on.catalog,model_config=config)
        made=V5.append_posterior_device(encoded,self.on_state,adapter,
            observer_sides=self.on.j.array([0],self.on.j.int32),set_posterior=self.inputs)
        self.assertEqual(made.admitted.tolist(),[True])
        np.testing.assert_array_equal(made.posterior_weights[:,:,:self.inputs.weights.shape[-1]],self.inputs.weights)

    def test_foreign_root_and_mutated_rows_fail_closed(self):
        with self.assertRaisesRegex(ValueError,'provenance'):
            F.compile_inputs(self.payload,self.on.catalog,self.on_state,owner=0,game_sha256='2'*64)
        payload=json.loads(json.dumps(self.payload));payload['rows'][next(iter(payload['rows']))]['weights'][0]+=1
        with self.assertRaisesRegex(ValueError,'provenance'):
            F.compile_inputs(payload,self.on.catalog,self.on_state,owner=0,game_sha256=self.game)
