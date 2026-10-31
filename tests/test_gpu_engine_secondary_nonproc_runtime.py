"""Literal original metadata/kernel/both FULL folds; fake scores, no games."""
from copy import deepcopy
from dataclasses import replace
import gc
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import jax
import jax.numpy as j
import numpy as np

from pokezero.category_vocab import CategoryVocabulary
from pokezero.dex import ShowdownDex
from pokezero.randbat import Gen3RandbatSource, Gen3RandbatSpeciesUniverse, Gen3RandbatVariant, RandbatSourceMetadata
from gpu_engine import engine_retained_encoder_driver as driver, engine_request_source as request
from gpu_engine import engine_own_facts as own, engine_direct_hit_controls as direct
from gpu_engine import engine_public_events as public, engine_leftovers_callbacks as leftovers
from gpu_engine import engine_execution_completeness as accounting, engine_kernel_frame_risk as risks
from gpu_engine import public_belief, source_provenance, transition, v4_encoder, mechanics
import test_gpu_engine_engine_weather_public as fixture


class SecondaryNonprocRuntimeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        fixture.WeatherPublicTests.setUpClass();cls.base=fixture.WeatherPublicTests
        cls.metadata=deepcopy(cls.base.metadata)
        cls.names=('icebeam','shadowball','rockslide','thunderbolt')
        moves=dict(cls.base.catalog.dex.moves);cls.move_ids=[]
        for name,typ,power,pp,category in (
                ('icebeam','Ice',95,10,'Special'),('shadowball','Ghost',80,15,'Physical'),
                ('rockslide','Rock',75,10,'Physical'),('thunderbolt','Electric',95,15,'Special')):
            cls.move_ids.append(len(cls.metadata['moves']))
            cls.metadata['vocabulary']['moves'].append(name)
            cls.metadata['moves'].append(dict(id=name,base_power=power,
                type=cls.metadata['vocabulary']['types'].index(typ.lower()),source_type=typ,
                category=category,pp=pp,accuracy=90 if name=='rockslide' else 100,priority=0,flags={'protect':1}))
            moves[name]=replace(cls.base.catalog.dex.moves['surf'],id=name,name=name,type=typ,
                category=category,gen3_category=category,base_power=power,pp=pp,target='normal',status=None,
                accuracy=90. if name=='rockslide' else 100.)
        dex=ShowdownDex(moves,cls.base.catalog.dex.species,cls.base.catalog.dex.type_chart)
        variants={name:Gen3RandbatSpeciesUniverse(name,80,tuple(Gen3RandbatVariant(
            f'{name}-{ability}','authored',name,'authored',80,cls.names,ability,'Leftovers')
            for ability in ('Pressure','Synchronize'))) for name in cls.metadata['vocabulary']['species'][1:]}
        source=Gen3RandbatSource(RandbatSourceMetadata('gen3randombattle',3,None,None,None,
            'literal-secondary-nonproc-fixture'),variants)
        labels=set(cls.base.catalog.vocabulary.tokens)
        for name in cls.names:labels|={'move:'+name,'belief:possible_move:'+name}
        labels|={'ability:synchronize','belief:possible_ability:synchronize'}
        cls.catalog=v4_encoder.compile_encoder(public_belief.compile_source(source),dex,
            CategoryVocabulary(tuple(sorted(labels))),checkpoint_sha256='a'*64,
            feature_masks=cls.base.catalog.masks.__dict__)
        cls.table=transition.tables_from_metadata(cls.metadata)
        producer=public.compile_catalog(cls.metadata,cls.table,public_moves=cls.catalog.moves,dex=dex,
            exposure=public.HealthExposure(public.SHOWDOWN,'gen3randombattle',True,False))
        original=cls.base.f.f
        rules=request.load_source_rules(original.root,cls.metadata,node=original.node)
        compiled=request.compile_request_source(cls.metadata,vocabulary=cls.catalog.vocabulary,dex=dex,
            checkpoint_sha256='a'*64,rules=rules,lifecycle_catalog=producer)
        facts=own.compile_own_facts(compiled,cls.catalog)
        cls.direct=direct.compile_direct_hit_controls(engine_catalog=producer)
        cls.prepared=driver.prepare(compiled,facts,public_mode='retained-full',
            leftovers_compiled=leftovers.compile_leftovers(cls.metadata,engine_catalog=producer),
            leftovers_mode='retained',account_source_sites=True,track_kernel_risks=True,
            direct_hit_compiled=cls.direct,direct_hit_mode='kernel-event',frame_input_mode='source-witness',
            fake_forward=lambda _:driver.FakeScores(j.zeros(9),j.array(0,j.int32)))
        cls.runtime=driver._checked(cls.prepared)
        jax.clear_caches();gc.collect()

    def same(self,a,b):
        for x,y in zip(jax.tree.leaves(a),jax.tree.leaves(b)):np.testing.assert_array_equal(x,y)

    def construct(self,*,synchronize=False):
        arrays=self.base().arrays('pressure','pressure')
        arrays['moves'][:]=np.asarray(self.move_ids,np.int32)
        arrays['pp'][:]=np.asarray([16,24,16,24],np.int32)
        if synchronize:arrays['ability'][1,:]=self.metadata['vocabulary']['abilities'].index('synchronize')
        with TemporaryDirectory() as directory:
            path=Path(directory)/'literal.npz'
            np.savez_compressed(path,**dict(arrays,metadata_utf8=np.frombuffer(json.dumps(self.metadata).encode(),np.uint8)))
            pool=source_provenance.load_constructor_pool(path)
            session=driver.construct(pool,np.asarray([[0,1],[0,1]],np.int64),prepared=self.prepared,
                stream_ids=j.array([9301,9302],j.int32),seed=31)
        self.assertEqual(session.state.core.admitted.tolist(),[True,True])
        for owner in session.result.owners:
            self.assertEqual(owner.reason.tolist(),[0,0])
            self.assertEqual(owner.full_line_consumers_complete.tolist(),[True,True])
            self.assertEqual(j.any(owner.encoded.posterior_mask[:,0]!=0,axis=-1).tolist(),[True,True])
        return session

    def advance(self,session,action,secondary):
        before=session.state;frames=[];actual=self.runtime['frame']
        def capture(*args):
            frame=actual(*args);frames.append(frame);return frame
        draws=j.zeros((2,2,12),j.int32).at[:,:,4].set(1)
        chance=transition.ChanceInputs(draws,j.zeros((2,2),j.int32),j.asarray(secondary,j.int32),
            j.zeros((2,2),j.int32),j.zeros((2,2),j.int32),j.zeros(2,j.int32))
        actions=j.array([[action,8],[action,8]],j.int32)
        self.runtime['frame']=capture
        try:result=driver.advance(session,actions,chance_inputs=chance)
        finally:self.runtime['frame']=actual
        self.assertEqual(len(frames),1)
        legacy=transition.step_with_chance(before.core.engine,actions,self.table,chance)
        self.same(result.state.core.engine,legacy.state)  # Includes exact RNG carry.
        return before,result,frames[0]

    def test_icebeam_nonproc_original_kernel_and_both_full_folds_complete(self):
        before,result,frame=self.advance(self.construct(),0,[[99,99],[99,99]])
        diagnostic=dict(risk=result.state.core.risks.risk.tolist(),tape_refusal=frame.callbacks.refusal.tolist(),
            upstream=result.state.core.execution.upstream_refusal.tolist(),
            owner_complete=[o.full_line_consumers_complete.tolist() for o in result.owners])
        self.assertEqual(result.state.core.admitted.tolist(),[True,True],msg=json.dumps(diagnostic))
        self.assertEqual(result.state.core.risks.risk.tolist(),[0,0])
        self.assertEqual(frame.state.status.tolist(),before.core.engine.status.tolist())
        ledger=result.state.core.execution
        self.assertTrue(bool(accounting.report(ledger,tables=self.runtime['execution_compiled'].device).scoped_drained.all()))
        for site in (accounting.Site.DIRECT_HIT,accounting.Site.LEFTOVERS):
            selected=(ledger.site==int(site))&(ledger.serial>0)
            positive=selected&(ledger.no_output_kind==0)
            self.assertEqual(positive.sum(axis=1).tolist(),[1,1])
            self.assertTrue(bool(j.all(j.where(selected,ledger.stage==3,True))))
        self.assertEqual(((ledger.site==int(accounting.Site.LEFTOVERS))&(ledger.no_output_kind==1)
            &(ledger.serial>0)).sum(axis=1).tolist(),[1,1])
        plain=risks.compile_risks(self.runtime['request']['lifecycle'],self.runtime['execution_compiled'])
        for move in self.move_ids:
            self.assertTrue(int(plain.device.selected_risk[move])&risks.FIELD_VOLATILE)
            self.assertEqual(int(self.runtime['kernel_risk_compiled'].device.selected_risk[move]),0)
        prior=frame.callbacks.execution.risks.carry._replace(risk=j.full(2,risks.FIELD_VOLATILE|risks.SELF_HP,j.int32))
        sticky=risks.observe_selected_device(prior,frame.callbacks,j.full(2,self.move_ids[0],j.int32),
            self.runtime['kernel_risk_compiled'].device,eligible=j.ones(2,bool),active=j.ones(2,bool))
        self.assertEqual(sticky.risk.tolist(),[risks.FIELD_VOLATILE|risks.SELF_HP]*2)
        for owner in result.owners:
            self.assertEqual(owner.full_line_consumers_complete.tolist(),[True,True])
            self.assertEqual(owner.encoded.observation.admitted.tolist(),[True,True])
            self.assertFalse(bool(owner.scored.any()));self.assertFalse(bool(owner.B4_qualified.any()))

    def test_actual_major_status_stats_flinch_and_reflection_procs_remain_unknown(self):
        for action,status,synchronize in ((0,mechanics.FRZ,False),(1,None,False),(2,None,False),
                (3,mechanics.PAR,True)):
            with self.subTest(action=action,synchronize=synchronize):
                before,result,frame=self.advance(self.construct(synchronize=synchronize),action,[[0,99],[99,99]])
                self.assertEqual(result.state.core.admitted.tolist(),[False,True])
                if status is not None:
                    target=int(frame.state.active[0,1]);actor=int(frame.state.active[0,0])
                    self.assertEqual(int(frame.state.status[0,1,target]),status)
                    self.assertTrue(int(frame.callbacks.refusal[0])&128)
                    if synchronize:self.assertEqual(int(frame.state.status[0,0,actor]),status)
                else:
                    bit=risks.BOOST if action==1 else risks.FIELD_VOLATILE
                    self.assertTrue(int(result.state.core.risks.risk[0])&bit)
                self.assertEqual(int(result.state.core.risks.risk[1]),0)
                for old,owner in zip(before.observers,result.state.observers):
                    self.same(jax.tree.map(lambda x:x[0],old.retained),jax.tree.map(lambda x:x[0],owner.retained))
                for owner in result.owners:
                    self.assertEqual(owner.full_line_consumers_complete.tolist(),[False,True])
                    self.assertFalse(bool(owner.scored.any()));self.assertFalse(bool(owner.B4_qualified.any()))


if __name__=='__main__':unittest.main()
