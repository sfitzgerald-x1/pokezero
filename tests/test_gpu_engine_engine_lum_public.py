"""Actual Lum receipts joined into pinned full controls; no scientific games."""
import unittest
from unittest.mock import patch
import numpy as np
import jax
import jax.numpy as j
from pokezero.randbat import Gen3RandbatSource, RandbatSourceMetadata, Gen3RandbatSpeciesUniverse, Gen3RandbatVariant
from pokezero.category_vocab import CategoryVocabulary
from gpu_engine import public_belief, v4_encoder as v4, engine_public_events as events
from gpu_engine import engine_public_event_device as base, engine_full_v4_event_device as full
from gpu_engine import engine_item_callbacks as item, public_wire, public_inputs as inputs, mechanics as m
import test_gpu_engine_engine_status_callbacks as fixture


class LumPublicTests(unittest.TestCase):
    kwargs=fixture.StatusCallbacksTests.kwargs
    initial=fixture.StatusCallbacksTests.initial
    chance=fixture.StatusCallbacksTests.chance
    step=fixture.StatusCallbacksTests.step
    same=fixture.StatusCallbacksTests.same
    start=fixture.StatusCallbacksTests.start
    compare=fixture.StatusCallbacksTests.compare

    @classmethod
    def setUpClass(cls):
        fixture.StatusCallbacksTests.setUpClass(); f=fixture.StatusCallbacksTests
        cls.f,cls.metadata,cls.ids,cls.table=f.f,f.metadata,f.ids,f.table
        names=cls.metadata['vocabulary']['species'][1:]
        abilities=cls.metadata['vocabulary']['abilities'][1:]
        variants={name:Gen3RandbatSpeciesUniverse(name,80,tuple(Gen3RandbatVariant(
            name+'-'+held,'fixture',name,'fixture',80,('Surf','Hidden Power Ice'),ability,held)
            for held in ('Leftovers','Lum Berry'))) for name,ability in zip(names,abilities)}
        host=public_belief.compile_source(Gen3RandbatSource(
            RandbatSourceMetadata('gen3randombattle',3,None,None,None,'lum-public-fixture'),variants))
        vocab=CategoryVocabulary(tuple(sorted(set(f.catalog.vocabulary.tokens)|{'item:lumberry','belief:possible_item:lumberry'})))
        cls.catalog=v4.compile_encoder(host,f.catalog.dex,vocab,checkpoint_sha256='a'*64,
            feature_masks=f.catalog.masks.__dict__)
        cls.producer=events.compile_catalog(cls.metadata,cls.table,public_moves=cls.catalog.moves,dex=cls.catalog.dex,
            exposure=events.HealthExposure(events.SHOWDOWN,'gen3randombattle',True,False))
        lower=base.compile_projector(cls.producer,expected_binding=cls.producer.binding,
            expected_engine_binding=np.array(cls.producer.engine_binding,np.uint32))
        cls.compiled=full.compile_adapter(cls.metadata,lower,catalog=cls.catalog)
        cls.fast=staticmethod(jax.jit(lambda s,f,p:full.project_and_fold(s,f,cls.compiled.device,p,cls.catalog)))
        cls.lum=cls.metadata['vocabulary']['items'].index('lumberry')

    def setup_frame(self,physical=0):
        arrays={k:np.array(v,copy=True) for k,v in self.f.arrays.items()}
        arrays['item'][1,physical]=self.lum
        return self.initial(arrays=arrays)

    def test_actual_nonphysical_lum_after_status_exact_all_words_and_public_leaves_both_owners(self):
        frame=self.setup_frame(5)
        actual=self.step(frame,[[0,8],[0,8]])
        for row in range(2):
            words=np.asarray(actual.callbacks.words[row]); at=np.flatnonzero(words[:,0]==item.LUM_CONSUMED)
            self.assertEqual(len(at),1)
            self.assertEqual(words[at[0],3:7].tolist(),[1,5,self.lum,m.PAR])
        for owner in (0,1):
            s,p=self.start(frame,j.full(2,owner,j.int32))
            initial,wire=self.compare(s,frame,p,[public_wire.WireState()]*2)
            result,_=self.compare(initial.projection.state,actual,initial.public,wire)
            self.assertEqual(result.public.mons.status[:,1,1].tolist(),[0,0])
            self.assertFalse(bool(result.projection.full_line_consumers_complete.any()))

    def test_wrong_old_status_and_wrong_item_roll_back_only_bad_row(self):
        frame=self.setup_frame(5);actual=self.step(frame,[[0,8],[0,8]])
        s,p=self.start(frame,j.array([0,1],j.int32));initial=self.fast(s,frame,p)
        index=int(np.flatnonzero(np.asarray(actual.callbacks.words[0,:,0])==item.LUM_CONSUMED)[0])
        for column,value in ((6,m.BRN),(5,0),(8,2),(9,1),(4,0)):
            bad=actual._replace(callbacks=actual.callbacks._replace(words=actual.callbacks.words.at[0,index,column].set(value)))
            result=self.fast(initial.projection.state,bad,initial.public)
            self.assertNotEqual(int(result.projection.error[0]),0)
            self.assertEqual(int(result.projection.error[1]),0)
            self.assertFalse(bool(result.projection.chunk.valid[0].any()))
            self.same(jax.tree.map(lambda x:x[0],initial.public._replace(context=None)),
                jax.tree.map(lambda x:x[0],result.public._replace(context=None)))
            self.assertEqual(int(result.public.cursor.sequence[0]),int(initial.public.cursor.sequence[0]))

    def test_item_namespace_drift_rejected_and_basic_mode_remains_unsupported(self):
        full.check_adapter(self.compiled)
        catalog=self.compiled.causal.items;old=catalog.ids
        from types import MappingProxyType
        try:
            object.__setattr__(catalog,'ids',MappingProxyType(dict(old,lumberry=old['lumberry']+1)))
            with self.assertRaises(ValueError):full.check_adapter(self.compiled)
        finally:object.__setattr__(catalog,'ids',old)
        frame=self.setup_frame();actual=self.step(frame,[[0,0],[0,0]])
        s,p=self.start(frame,j.array([0,1],j.int32));initial=self.fast(s,frame,p)
        disabled=self.compiled.device._replace(causal=self.compiled.device.causal._replace(lum_events=j.array(False)))
        result=full.project_and_fold(initial.projection.state,actual,disabled,initial.public,self.catalog)
        self.assertFalse(bool(result.projection.chunk.valid.any()))
        self.same(result.public,initial.public)

    def test_large_frame_drains_all_chunks_and_late_error_rolls_back_whole_frame(self):
        frame=self.setup_frame();actual=self.step(frame,[[0,0],[0,0]])
        s,p=self.start(frame,j.array([0,1],j.int32))
        initial,wire=self.compare(s,frame,p,[public_wire.WireState()]*2)
        result,_=self.compare(initial.projection.state,actual,initial.public,wire)
        self.assertTrue(bool(result.projection.chunk.more.all()))
        self.assertTrue(bool(result.projection.frames.valid[1].any()))
        self.assertFalse(bool(result.public.cursor.needs_more.any()))
        projection=full.project_frame(initial.projection.state,actual,self.compiled.device)
        damaged=projection.frames._replace(words=projection.frames.words.at[1,0,0,1].add(9))
        with patch.object(full,'project_frame',return_value=projection._replace(frames=damaged)):
            rejected=full.project_and_fold(initial.projection.state,actual,self.compiled.device,initial.public,self.catalog)
        self.assertNotEqual(int(rejected.projection.error[0]),0)
        self.assertEqual(int(rejected.projection.error[1]),0)
        self.assertFalse(bool(rejected.projection.frames.valid[:,0].any()))
        self.assertEqual(int(rejected.public.cursor.sequence[0]),int(initial.public.cursor.sequence[0]))
        self.assertEqual(int(rejected.projection.state.sequence[0]),int(initial.projection.state.sequence[0]))
        self.same(full._commit_public(j.array([False,True]),initial.public,rejected.public),rejected.public)


if __name__=='__main__':unittest.main()
