"""Actual Toxic miss/type-no-effect source events; R7 causes stay unidentified.

Tiny original constructor/kernel/public consumers and fake scores only. These
are selected-kernel input cases, not original-physics/B4 qualification.
"""
from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
import jax
import jax.numpy as j
import numpy as np
from pokezero.category_vocab import CategoryVocabulary
from pokezero.dex import ShowdownDex
from pokezero.randbat import Gen3RandbatSource,Gen3RandbatSpeciesUniverse,Gen3RandbatVariant,RandbatSourceMetadata
from gpu_engine import engine_retained_encoder_driver as driver,engine_request_source as request
from gpu_engine import engine_own_facts as own,engine_toxic_callbacks as toxic
from gpu_engine import engine_public_events as public,engine_leftovers_callbacks as leftovers
from gpu_engine import engine_execution_completeness as accounting,engine_kernel_frame_risk as risks
from gpu_engine import public_belief,source_provenance,transition,v4_encoder
import test_gpu_engine_engine_weather_public as fixture
import gpu_engine_eager_witness as eager

class ToxicRuntimeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        fixture.WeatherPublicTests.setUpClass();cls.base=fixture.WeatherPublicTests
        cls.metadata=deepcopy(cls.base.metadata)
        cls.move_id=cls.metadata['vocabulary']['moves'].index('toxic')
        cls.metadata['moves'][cls.move_id]=dict(id='toxic',base_power=0,type=cls.metadata['vocabulary']['types'].index('poison'),
            source_type='Poison',category='Status',pp=10,accuracy=85,priority=0,flags={'protect':1})
        moves=dict(cls.base.catalog.dex.moves)
        moves['toxic']=replace(moves['surf'],id='toxic',name='Toxic',type='Poison',category='Status',gen3_category='Status',
            base_power=0,pp=10,target='normal',status='tox',accuracy=85.,priority=0,effect_label='tox',effect_chance=100,
            self_hp_cost=0.,recoil=False,drain=False,heal=False,boosts={})
        dex=ShowdownDex(moves,cls.base.catalog.dex.species,cls.base.catalog.dex.type_chart)
        cls.names=('toxic','raindance','sunnyday','hail')
        variants={n:Gen3RandbatSpeciesUniverse(n,80,(Gen3RandbatVariant('toxic-'+n,'authored',n,'authored',80,
            cls.names,'Pressure','Leftovers'),)) for n in cls.metadata['vocabulary']['species'][1:]}
        src=Gen3RandbatSource(RandbatSourceMetadata('gen3randombattle',3,None,None,None,'literal-toxic85-fixture'),variants)
        labels=set(cls.base.catalog.vocabulary.tokens)|{'move:toxic','belief:possible_move:toxic','move_effect:tox'}
        cls.catalog=v4_encoder.compile_encoder(public_belief.compile_source(src),dex,
            CategoryVocabulary(tuple(sorted(labels))),checkpoint_sha256='a'*64,feature_masks=cls.base.catalog.masks.__dict__)
        cls.table=transition.tables_from_metadata(cls.metadata)
        producer=public.compile_catalog(cls.metadata,cls.table,public_moves=cls.catalog.moves,dex=dex,
            exposure=public.HealthExposure(public.SHOWDOWN,'gen3randombattle',True,False))
        original=cls.base.f.f;rules=request.load_source_rules(original.root,cls.metadata,node=original.node)
        cls.request=request.compile_request_source(cls.metadata,vocabulary=cls.catalog.vocabulary,dex=dex,
            checkpoint_sha256='a'*64,rules=rules,lifecycle_catalog=producer)
        cls.facts=own.compile_own_facts(cls.request,cls.catalog)
        cls.toxic=toxic.compile_toxic(engine_catalog=producer,metadata=cls.metadata)
        cls.prepared=driver.prepare(cls.request,cls.facts,public_mode='retained-full',
            leftovers_compiled=leftovers.compile_leftovers(cls.metadata,engine_catalog=producer),leftovers_mode='retained',
            account_source_sites=True,track_kernel_risks=True,toxic_compiled=cls.toxic,toxic_mode='kernel-event',
            frame_input_mode='source-witness',fake_forward=lambda _:driver.FakeScores(j.zeros(9),j.array(0,j.int32)))
        cls.runtime=driver._checked(cls.prepared)

    def same(self,a,b):
        for x,y in zip(jax.tree.leaves(a),jax.tree.leaves(b)):np.testing.assert_array_equal(x,y)

    def construct(self):
        arrays=self.base().arrays('pressure','pressure')
        arrays['moves'][:]=np.array([self.move_id]+[self.metadata['vocabulary']['moves'].index(n) for n in self.names[1:]],np.int32)
        arrays['pp'][:]=np.array([16,8,8,16],np.int32)
        with TemporaryDirectory() as tmp:
            p=Path(tmp)/'literal.npz';np.savez_compressed(p,**dict(arrays,metadata_utf8=np.frombuffer(json.dumps(self.metadata).encode(),np.uint8)))
            pool=source_provenance.load_constructor_pool(p)
            session=driver.construct(pool,np.array([[0,1],[0,1]],np.int64),prepared=self.prepared,
                stream_ids=j.array([9701,9702],j.int32),seed=31)
        self.assertEqual(session.state.core.admitted.tolist(),[True,True]);return session

    def advance(self,session,roll,foe_action=8):
        with eager.eager_source_witness(self.prepared):
            before=session.state;frames=[];actual=self.runtime['frame']
            def capture(*args):f=actual(*args);frames.append(f);return f
            draws=j.zeros((2,2,12),j.int32).at[:,:,4].set(1).at[:,0,6].set(roll)
            chance=transition.ChanceInputs(draws,*[j.zeros((2,2),j.int32) for _ in range(4)],j.zeros(2,j.int32))
            acts=j.array([[0,foe_action],[0,foe_action]],j.int32);self.runtime['frame']=capture
            try:result=driver.advance(session,acts,chance_inputs=chance)
            finally:self.runtime['frame']=actual
        legacy=transition.step_with_chance(before.core.engine,acts,self.table,chance)
        self.same(result.state.core.engine,legacy.state);self.assertEqual(len(frames),1)
        return before,result,frames[0]

    def test_actual_miss_and_poison_type_no_effect_both_genuine_drains(self):
        for roll,code in ((99,toxic.MISS),(0,toxic.PLAIN_TYPE_IMMUNITY)):
            with self.subTest(roll=roll):
                before,result,frame=self.advance(self.construct(),roll)
                diagnostic=dict(risk=result.state.core.risks.risk.tolist(),refusal=frame.callbacks.refusal.tolist(),
                    accounting=result.state.core.execution.refusal.tolist(),witness=result.state.core.frame_input_report.refusal.tolist(),
                    reasons=[o.reason.tolist() for o in result.owners])
                self.assertEqual(result.state.core.admitted.tolist(),[True,True],msg=json.dumps(diagnostic))
                self.assertEqual(result.state.core.risks.risk.tolist(),[0,0])
                ledger=result.state.core.execution;selected=(ledger.site==int(accounting.Site.TOXIC_ATTEMPT))&(ledger.serial>0)
                self.assertEqual(selected.sum(axis=1).tolist(),[1,1]);self.assertTrue(bool(j.all(j.where(selected,ledger.stage==3,True))))
                raw=frame.callbacks.words;mask=(j.arange(raw.shape[1])[None,:]<frame.callbacks.count[:,None])&(raw[:,:,0]==toxic.TOXIC_ATTEMPT)
                self.assertEqual(mask.sum(axis=1).tolist(),[1,1]);rows=j.arange(2);at=j.argmax(mask,axis=1);packet=raw[rows,at]
                self.assertEqual(toxic.outcome_device(packet).tolist(),[code,code])
                self.assertEqual(packet[:,7:10].tolist(),[[0,0,0],[0,0,0]])
                self.assertEqual(int(packet[0,4]),5) # actual switched target, not physical0
                for owner in result.owners:
                    self.assertEqual(owner.full_line_consumers_complete.tolist(),[True,True])
                    self.assertEqual(owner.encoded.observation.admitted.tolist(),[True,True])
                    self.assertFalse(bool(owner.B4_qualified.any()))
        plain=risks.compile_risks(self.runtime['request']['lifecycle'],self.runtime['execution_compiled'])
        self.assertTrue(int(plain.device.selected_risk[self.move_id])&risks.FIELD_VOLATILE)
        self.assertEqual(int(self.runtime['kernel_risk_compiled'].device.selected_risk[self.move_id]),0)

    def test_actual_success_retains_literal_kind9_and_counter_but_stays_unknown(self):
        # Foe physical4 is Pikachu, so this is an actual successful apply. The
        # later TOX residual/history path is deliberately not registered here.
        before,result,frame=self.advance(self.construct(),0,foe_action=7)
        raw=frame.callbacks.words;present=j.arange(raw.shape[1])[None,:]<frame.callbacks.count[:,None]
        self.assertTrue(bool(j.all(j.any(present&(raw[:,:,0]==9),axis=1))))
        mask=present&(raw[:,:,0]==toxic.TOXIC_ATTEMPT);at=j.argmax(mask,axis=1);packet=raw[j.arange(2),at]
        self.assertEqual(toxic.outcome_device(packet).tolist(),[toxic.STATUS_WRITE]*2)
        self.assertEqual(packet[:,7:10].tolist(),[[0,4,0],[0,4,0]])
        self.assertEqual(result.state.core.admitted.tolist(),[False,False])
        self.assertTrue(bool(j.all((frame.callbacks.refusal&512)!=0)))  # residual observer absent


def tearDownModule():
    eager.clear()   # drop the consume executables this module compiled
