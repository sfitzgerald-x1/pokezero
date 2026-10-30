"""Actual owning birth -> guarded weather segment -> public HP, no models/games."""
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch
import jax
import jax.numpy as j
import numpy as np
from gpu_engine import engine_retained_encoder_driver as driver,engine_request_source as source
from gpu_engine import engine_own_facts as own,source_provenance,transition
import test_gpu_engine_engine_weather_public as fixture


class WeatherHPRuntimeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        fixture.WeatherPublicTests.setUpClass();cls.f=fixture.WeatherPublicTests
        rules=source.load_source_rules(cls.f.f.f.root,cls.f.metadata,node=cls.f.f.f.node)
        request=source.compile_request_source(cls.f.metadata,vocabulary=cls.f.catalog.vocabulary,
            dex=cls.f.catalog.dex,checkpoint_sha256='a'*64,rules=rules,lifecycle_catalog=cls.f.producer)
        cls.prepared=driver.prepare(request,own.compile_own_facts(request,cls.f.catalog),public_mode='retained-full',
            fake_forward=lambda values:driver.FakeScores(j.zeros(9),j.array(1,j.int32)))

    def construct(self,*,tie=False,lethal=False):
        arrays=self.f().arrays('pressure','pressure')
        arrays['moves'][0,0,:2]=[self.f.ids['hail'],1]
        if tie:arrays['stats'][...,5]=199
        if lethal:arrays['stats'][...,0]=1
        with TemporaryDirectory() as directory:
            path=Path(directory)/'physical.npz'
            np.savez_compressed(path,**dict(arrays,metadata_utf8=np.frombuffer(json.dumps(self.f.metadata).encode(),np.uint8)))
            pool=source_provenance.load_constructor_pool(path)
            return driver.construct(pool,np.tile([[0,1]],(2,1)).astype(np.int64),prepared=self.prepared,
                stream_ids=j.array([801,802],j.int32),seed=31)

    def advance(self,session):
        before=session.state;record=driver._checked(self.prepared);actual=record['frame'];frames=[]
        def capture(*args):
            frame=actual(*args);frames.append(frame);return frame
        record['frame']=capture
        try:result=driver.advance(session,j.tile(j.array([[0,8]],j.int32),(2,1)),chance_inputs=self.f().chance())
        finally:record['frame']=actual
        self.assertEqual(len(frames),1)
        expected=transition.step_with_chance(before.core.engine,j.tile(j.array([[0,8]],j.int32),(2,1)),
            self.f.table,self.f().chance())
        self.f().same(result.state.core.engine,expected.state)
        for owner in result.owners:
            self.assertFalse(bool(owner.scored.any()))
            self.assertFalse(bool(owner.B4_qualified.any()))
        return result,frames[0]

    def test_actual_constructor_profile_orders_nonlethal_weather_and_both_public_owners(self):
        session=self.construct();self.assertEqual(session.state.core.admitted.tolist(),[True,True])
        self.assertEqual(session.state.core.weather_profile.qualified_birth.tolist(),[True,True])
        result,frame=self.advance(session)
        self.assertEqual(result.state.core.admitted.tolist(),[True,True])
        self.assertEqual(frame.callbacks.refusal.tolist(),[1,1])
        for owner in result.owners:self.assertEqual(owner.retained_error.tolist(),[0,0])
        for row in range(2):
            words=np.array(frame.callbacks.words[row,:int(frame.callbacks.count[row])]);chip=words[words[:,0]==16]
            self.assertEqual(chip[:,3].tolist(),[1,0]);self.assertEqual(chip[:,4].tolist(),[5,0])
            self.assertTrue(bool(np.all(chip[:,6]>0)))
        for side,observer in enumerate(result.state.observers):
            for victim in (0,1):
                public=observer.retained.mons;slot=int(public.active[0,victim])
                physical=int(result.state.core.engine.active[0,victim])
                # Later uninstrumented Leftovers is not a public receipt. Check
                # the captured weather write, not final hidden physical HP.
                chip=next(w for w in np.array(frame.callbacks.words[0]) if w[0]==16 and w[3]==victim)
                hp=int(chip[6]);maximum=int(chip[7]);display=hp if side==victim else min(99,(100*hp+maximum-1)//maximum)
                self.assertEqual(int(public.hp_numerator[0,victim,slot]),display)
                self.assertEqual(int(public.hp_denominator[0,victim,slot]),maximum if side==victim else 100)

    def test_actual_ties_and_lethal_segments_preserve_unknown_and_public_rollback(self):
        for kwargs in ({'tie':True},{'lethal':True}):
            session=self.construct(**kwargs);before=session.state
            self.assertEqual(before.core.admitted.tolist(),[True,True])
            result,frame=self.advance(session)
            self.assertTrue(bool(((frame.callbacks.refusal & 256)!=0).all()))
            self.assertEqual(result.state.core.admitted.tolist(),[False,False])
            for old,new in zip(before.observers,result.state.observers):
                self.assertEqual(new.retained.cursor.sequence.tolist(),old.retained.cursor.sequence.tolist())
                self.f().same(new.retained,old.retained)

    def test_intermediate_false_record_is_carried_even_on_empty_weather_segment(self):
        from gpu_engine import engine_weather_hp_segment as segment
        # Fault injection at the internal source seam: a producer returns a
        # sticky refusal with no HP line. A safe final state must not erase it.
        session=self.construct();r=driver._checked(self.prepared);r['frame'].clear_cache()
        actual=segment.order_segment_device
        def latch(*args,**kwargs):
            result=actual(*args,**kwargs)
            return result._replace(record=result.record._replace(
                qualified_birth=j.zeros_like(result.record.qualified_birth)))
        try:
            with patch.object(segment,'order_segment_device',side_effect=latch):
                result=driver.advance(session,j.full((2,2),8,j.int32),chance_inputs=self.f().chance())
            self.assertEqual(result.state.core.weather_profile.qualified_birth.tolist(),[False,False])
            # A later safe frame retains the false origin after the fault hook
            # is removed. No synthetic callback result qualifies model input.
            r['frame'].clear_cache()
            later=driver.advance(session,j.full((2,2),4,j.int32),chance_inputs=self.f().chance())
            self.assertEqual(later.state.core.weather_profile.qualified_birth.tolist(),[False,False])
        finally:r['frame'].clear_cache()

    def test_missing_both_actual_hp_receipts_refuses_and_rolls_back_public(self):
        from gpu_engine import engine_weather_hp_segment as segment
        session=self.construct();before=session.state
        r=driver._checked(self.prepared);r['frame'].clear_cache();actual=segment.order_segment_device
        def omit(tape,before_count,*args,**kwargs):
            # Fault injection at the real post-write seam, with actual states.
            return actual(tape._replace(count=before_count),before_count,*args,**kwargs)
        try:
            with patch.object(segment,'order_segment_device',side_effect=omit):
                result,frame=self.advance(session)
            self.assertTrue(bool(((frame.callbacks.refusal&256)!=0).all()))
            self.assertEqual(result.state.core.admitted.tolist(),[False,False])
            for old,new in zip(before.observers,result.state.observers):self.f().same(old.retained,new.retained)
        finally:r['frame'].clear_cache()


if __name__=='__main__':unittest.main()
