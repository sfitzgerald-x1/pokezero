"""Genuine owning constructor -> actual self-write -> both drains/encoder.

Tiny source fixture/fake score only, no original-physics or B4 qualification.
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
from gpu_engine import engine_own_facts as own,engine_substitute_callbacks as sub
from gpu_engine import engine_public_events as public,engine_leftovers_callbacks as leftovers
from gpu_engine import engine_execution_completeness as accounting,engine_kernel_frame_risk as risks
from gpu_engine import public_belief,source_provenance,transition,v4_encoder
import test_gpu_engine_engine_weather_public as fixture
import gpu_engine_eager_witness as eager

class SubstituteRuntimeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        fixture.WeatherPublicTests.setUpClass();cls.base=fixture.WeatherPublicTests
        cls.metadata=deepcopy(cls.base.metadata)
        cls.move_id=len(cls.metadata['moves']);cls.metadata['vocabulary']['moves'].append('substitute')
        cls.metadata['moves'].append(dict(id='substitute',base_power=0,type=cls.metadata['vocabulary']['types'].index('normal'),
            source_type='Normal',category='Status',pp=10,accuracy=True,priority=0,flags={}))
        moves=dict(cls.base.catalog.dex.moves)
        moves['substitute']=replace(moves['surf'],id='substitute',name='Substitute',type='Normal',category='Status',
            gen3_category='Status',base_power=0,pp=10,target='self',status=None,accuracy=100.,priority=0,
            effect_label='substitute',effect_chance=100,self_hp_cost=.25,recoil=False,drain=False,heal=False,boosts={})
        dex=ShowdownDex(moves,cls.base.catalog.dex.species,cls.base.catalog.dex.type_chart)
        cls.names=('substitute','raindance','sunnyday','hail')
        variants={n:Gen3RandbatSpeciesUniverse(n,80,(Gen3RandbatVariant('sub-'+n,'authored',n,'authored',80,
            cls.names,'Pressure','Leftovers'),)) for n in cls.metadata['vocabulary']['species'][1:]}
        source=Gen3RandbatSource(RandbatSourceMetadata('gen3randombattle',3,None,None,None,'literal-substitute-fixture'),variants)
        labels=set(cls.base.catalog.vocabulary.tokens)|{'move:substitute','belief:possible_move:substitute','move_effect:substitute'}
        cls.catalog=v4_encoder.compile_encoder(public_belief.compile_source(source),dex,
            CategoryVocabulary(tuple(sorted(labels))),checkpoint_sha256='a'*64,feature_masks=cls.base.catalog.masks.__dict__)
        cls.table=transition.tables_from_metadata(cls.metadata)
        producer=public.compile_catalog(cls.metadata,cls.table,public_moves=cls.catalog.moves,dex=dex,
            exposure=public.HealthExposure(public.SHOWDOWN,'gen3randombattle',True,False))
        original=cls.base.f.f
        rules=request.load_source_rules(original.root,cls.metadata,node=original.node)
        compiled=request.compile_request_source(cls.metadata,vocabulary=cls.catalog.vocabulary,dex=dex,
            checkpoint_sha256='a'*64,rules=rules,lifecycle_catalog=producer)
        facts=own.compile_own_facts(compiled,cls.catalog)
        cls.substitute=sub.compile_substitute(engine_catalog=producer,metadata=cls.metadata)
        cls.prepared=driver.prepare(compiled,facts,public_mode='retained-full',
            leftovers_compiled=leftovers.compile_leftovers(cls.metadata,engine_catalog=producer),leftovers_mode='retained',
            account_source_sites=True,track_kernel_risks=True,substitute_compiled=cls.substitute,
            substitute_mode='kernel-event',frame_input_mode='source-witness',
            fake_forward=lambda _:driver.FakeScores(j.zeros(9),j.array(0,j.int32)))
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
            # Construction runs the witness too: keep it eager.
            with eager.eager_source_witness(self.prepared):
                session=driver.construct(pool,np.array([[0,1],[0,1]],np.int64),prepared=self.prepared,
                    stream_ids=j.array([9601,9602],j.int32),seed=31)
        self.assertEqual(session.state.core.admitted.tolist(),[True,True])
        return session

    def advance(self,session):
        with eager.eager_source_witness(self.prepared):
            previous=session.state;frames=[];actual=self.runtime['frame']
            def capture(*args):
                f=actual(*args);frames.append(f);return f
            chance=transition.ChanceInputs(j.zeros((2,2,12),j.int32).at[:,:,4].set(1),
                *[j.zeros((2,2),j.int32) for _ in range(4)],j.zeros(2,j.int32))
            actions=j.array([[0,8],[0,8]],j.int32)
            self.runtime['frame']=capture
            try:result=driver.advance(session,actions,chance_inputs=chance)
            finally:self.runtime['frame']=actual
        self.assertEqual(len(frames),1)
        legacy=transition.step_with_chance(previous.core.engine,actions,self.table,chance)
        self.same(result.state.core.engine,legacy.state)
        return previous,result,frames[0]

    def test_actual_owning_success_admits_only_after_two_drains_and_existing_sub_refuses(self):
        session=self.construct();before,result,frame=self.advance(session)
        diagnostic=dict(risk=result.state.core.risks.risk.tolist(),refusal=frame.callbacks.refusal.tolist(),
            accounting=result.state.core.execution.refusal.tolist(),
            witness=result.state.core.frame_input_report.refusal.tolist(),
            reasons=[o.reason.tolist() for o in result.owners])
        self.assertEqual(result.state.core.admitted.tolist(),[True,True],msg=json.dumps(diagnostic))
        self.assertEqual(result.state.core.risks.risk.tolist(),[0,0])
        ledger=result.state.core.execution;selected=(ledger.site==int(accounting.Site.SUBSTITUTE_SELF))&(ledger.serial>0)
        self.assertEqual(selected.sum(axis=1).tolist(),[1,1]);self.assertTrue(bool(j.all(j.where(selected,ledger.stage==3,True))))
        for owner in result.owners:
            self.assertEqual(owner.full_line_consumers_complete.tolist(),[True,True])
            self.assertEqual(owner.encoded.observation.admitted.tolist(),[True,True])
            self.assertFalse(bool(owner.B4_qualified.any()))
        plain=risks.compile_risks(self.runtime['request']['lifecycle'],self.runtime['execution_compiled'])
        self.assertTrue(int(plain.device.selected_risk[self.move_id])&risks.SELF_EFFECT)
        self.assertEqual(int(self.runtime['kernel_risk_compiled'].device.selected_risk[self.move_id]),0)
        # Same actual session, next actual attempt with an existing substitute:
        # no manufactured start/damage or generic no-output credit.
        prior,failed,badframe=self.advance(session)
        self.assertEqual(failed.state.core.admitted.tolist(),[False,False])
        self.assertTrue(bool(((failed.state.core.risks.risk&risks.SELF_EFFECT)!=0).all()))
        self.assertTrue(bool((failed.state.core.execution.refusal!=0).all()))
        words=np.asarray(badframe.callbacks.words);present=np.arange(words.shape[1])[None,:]<np.asarray(badframe.callbacks.count)[:,None]
        self.assertFalse(bool(np.any(present&(words[:,:,0]==sub.SUBSTITUTE_CREATED))))


def tearDownModule():
    eager.clear()   # drop the consume executables this module compiled
