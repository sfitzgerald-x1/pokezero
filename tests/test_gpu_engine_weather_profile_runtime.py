"""Actual birth capture and sticky conditional Speed scope; no model/games."""
from copy import deepcopy
import unittest
from unittest.mock import patch

import jax
import jax.numpy as j
import numpy as np

from gpu_engine import engine_retained_encoder_driver as driver
from gpu_engine import engine_request_source as source,engine_public_events as events
from gpu_engine import engine_own_facts as own,transition,engine_lifecycle_callbacks as lifecycle
import test_gpu_engine_engine_full_retained_driver as fixture


class WeatherProfileRuntimeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        fixture.FullRetainedDriverTests.setUpClass()
        cls.f=fixture.FullRetainedDriverTests.f
        cls.prepared=fixture.FullRetainedDriverTests.prepared

    def construct(self,*,prepared=None,pool=None,batch=2):
        return driver.construct(pool or self.f.pool,np.tile([[0,1]],(batch,1)).astype(np.int64),
            prepared=prepared or self.prepared,stream_ids=j.arange(601,601+batch,dtype=j.int32),seed=37)

    def test_actual_pre_entry_birth_identity_and_current_record_are_carried(self):
        inputs=[];original=lifecycle.initialize_with_lifecycle
        def capture(state,*args,**kwargs):
            inputs.append(state);return original(state,*args,**kwargs)
        with patch.object(lifecycle,'initialize_with_lifecycle',side_effect=capture):session=self.construct()
        constructed=driver._session(session)['constructed']
        self.assertEqual(len(inputs),1);self.assertIs(constructed.birth_state,inputs[0])
        self.assertTrue(bool(constructed.birth_state.entry_pending.all()))
        self.assertFalse(bool(constructed.state.entry_pending.any()))
        record=session.state.core.weather_profile
        self.assertEqual(record.qualified_birth.tolist(),[True,True])
        np.testing.assert_array_equal(record.birth_spe,constructed.birth_state.stats[...,5])
        result=driver.advance(session,j.zeros((2,2),j.int32),
            chance_inputs=fixture.FullRetainedDriverTests().chance())
        self.assertEqual(result.state.core.weather_profile.qualified_birth.tolist(),[True,True])
        for report in result.owners:self.assertFalse(bool(report.scored.any()))

    def test_inactive_actual_unsafe_members_fail_despite_same_bound_metadata(self):
        metadata=deepcopy(self.f.metadata)
        metadata['vocabulary']['items']+=['machobrace','quickclaw','starfberry']
        table=transition.tables_from_metadata(metadata)
        engine=events.compile_catalog(metadata,table,public_moves=self.f.catalog.moves,dex=self.f.catalog.dex,
            exposure=events.HealthExposure(source.SHOWDOWN,'gen3randombattle',True,False))
        rules=source.load_source_rules(self.f.f.root,metadata,node=self.f.f.node)
        request=source.compile_request_source(metadata,vocabulary=self.f.catalog.vocabulary,dex=self.f.catalog.dex,
            checkpoint_sha256='a'*64,rules=rules,lifecycle_catalog=engine)
        prepared=driver.prepare(request,own.compile_own_facts(request,self.f.catalog),public_mode='retained-full',
            fake_forward=lambda values:driver.FakeScores(j.zeros(9),j.array(1,j.int32)))
        arrays={name:np.array(value,copy=True) for name,value in self.f.arrays.items()}
        # Both the safe and injected pools use these SAME exact metadata edges.
        from tempfile import TemporaryDirectory
        from pathlib import Path
        import json
        from gpu_engine import source_provenance
        with TemporaryDirectory() as directory:
            path=Path(directory)/'physical.npz'
            def pool():
                np.savez_compressed(path,**dict(arrays,metadata_utf8=np.frombuffer(json.dumps(metadata).encode(),np.uint8)))
                return source_provenance.load_constructor_pool(path)
            good=self.construct(prepared=prepared,pool=pool())
            self.assertEqual(good.state.core.weather_profile.qualified_birth.tolist(),[True,True])
            for side,physical,item in ((1,5,'machobrace'),(0,4,'quickclaw'),(1,2,'starfberry')):
                arrays['item'][:]=1
                arrays['item'][side,physical]=metadata['vocabulary']['items'].index(item)
                bad=self.construct(prepared=prepared,pool=pool())
                self.assertEqual(bad.state.core.weather_profile.qualified_birth.tolist(),[False,False])
                for report in bad.result.owners:self.assertFalse(bool(report.scored.any()))

    def test_birth_holder_and_profile_edges_reject_replacement_before_advance(self):
        session=self.construct();constructed=driver._session(session)['constructed'];old_birth=constructed.birth_state
        state=session.state
        object.__setattr__(constructed,'birth_state',old_birth._replace(item=j.zeros_like(old_birth.item)))
        try:
            with self.assertRaises(ValueError):driver.advance(session,j.zeros((2,2),j.int32))
            self.assertIs(session.state,state)
        finally:object.__setattr__(constructed,'birth_state',old_birth)
        profile=driver._checked(self.prepared)['speed_profile'];device=profile.device
        object.__setattr__(profile,'device',device._replace(birth_items=j.ones_like(device.birth_items)))
        try:
            with self.assertRaises(ValueError):driver.advance(session,j.zeros((2,2),j.int32))
            self.assertIs(session.state,state)
        finally:object.__setattr__(profile,'device',device)
        driver._checked(self.prepared)

    def test_profile_binding_rows_are_not_globals_at_batch_eight(self):
        session=self.construct(batch=8);before=session.state.core.weather_profile
        draws=j.zeros((8,2,12),j.int32).at[:,:,4].set(1)
        chance=transition.ChanceInputs(draws,*[j.zeros((8,2),j.int32) for _ in range(4)],j.zeros(8,j.int32))
        active=j.array([True,False]*4)
        result=driver.advance(session,j.zeros((8,2),j.int32),active_rows=active,chance_inputs=chance)
        after=result.state.core.weather_profile
        self.assertEqual(after.profile_binding.shape,(8,8))
        for old,new in zip(jax.tree.leaves(before),jax.tree.leaves(after)):
            np.testing.assert_array_equal(np.asarray(new)[~np.asarray(active)],np.asarray(old)[~np.asarray(active)])
        self.assertEqual(result.state.core.frame_number.tolist(),[1,0]*4)


if __name__=='__main__':unittest.main()
