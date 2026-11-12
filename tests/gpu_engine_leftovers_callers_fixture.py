"""Fixture for the actual owning Leftovers write seam, always UNKNOWN; no models/games (not a test module).

LeftoversOwningCallerTests keeps the fixture API of the former test_gpu_engine_leftovers_owning_callers:
setUpClass, same, construct, chance, trace and words. Its tests run in test_gpu_engine_leftovers_caller_frames.
The borrowers are that module and test_gpu_engine_registered_leftovers_driver.
"""
from copy import deepcopy
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import jax
import jax.numpy as j
import numpy as np

from gpu_engine import engine_leftovers_callbacks as leftovers
from gpu_engine import engine_lifecycle_callbacks as life, engine_retained_encoder_driver as driver
from gpu_engine import engine_request_source as source, engine_own_facts as own
from gpu_engine import transition, source_provenance
import test_gpu_engine_engine_weather_public as fixtures


class LeftoversOwningCallerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        fixtures.WeatherPublicTests.setUpClass();cls.f=fixtures.WeatherPublicTests
        cls.metadata=deepcopy(cls.f.metadata)
        cls.leftovers=leftovers.compile_leftovers(cls.metadata,engine_catalog=cls.f.producer)
        rules=source.load_source_rules(cls.f.f.f.root,cls.f.metadata,node=cls.f.f.f.node)
        cls.request=source.compile_request_source(cls.f.metadata,vocabulary=cls.f.catalog.vocabulary,
            dex=cls.f.catalog.dex,checkpoint_sha256='a'*64,rules=rules,lifecycle_catalog=cls.f.producer)
        cls.own=own.compile_own_facts(cls.request,cls.f.catalog)
        fake=lambda _:driver.FakeScores(j.zeros(9),j.array(1,j.int32))
        cls.plain=driver.prepare(cls.request,cls.own,public_mode='retained-full',fake_forward=fake)
        cls.prepared=driver.prepare(cls.request,cls.own,public_mode='retained-full',fake_forward=fake,
            leftovers_compiled=cls.leftovers)
        cls.item=cls.metadata['vocabulary']['items'].index('leftovers')

    def same(self,a,b):
        for x,y in zip(jax.tree.leaves(a),jax.tree.leaves(b)):np.testing.assert_array_equal(x,y)

    def construct(self,b=2,move='hail',prepared=None):
        arrays=self.f().arrays('pressure','pressure')
        arrays['moves'][0,0,:2]=[self.f.ids[move],1]
        with TemporaryDirectory() as directory:
            path=Path(directory)/'tiny.npz'
            np.savez_compressed(path,**dict(arrays,metadata_utf8=np.frombuffer(json.dumps(self.f.metadata).encode(),np.uint8)))
            pool=source_provenance.load_constructor_pool(path)
            return driver.construct(pool,np.tile([[0,1]],(b,1)).astype(np.int64),
                prepared=self.prepared if prepared is None else prepared,
                stream_ids=j.arange(901,901+b,dtype=j.int32),seed=31)

    def chance(self,b=2):
        draws=j.zeros((b,2,12),j.int32).at[:,:,4].set(1)
        return transition.ChanceInputs(draws,*[j.zeros((b,2),j.int32) for _ in range(4)],j.zeros(b,j.int32))

    def trace(self,state,cursor,actions,compiled=True):
        return life.trace_step(state,actions,self.f.table,cursor=cursor,catalog=self.f.producer,
            expected_binding=self.f.producer.binding,expected_engine_binding=np.array(self.f.producer.engine_binding,np.uint32),
            chance_inputs=self.chance(state.hp.shape[0]),leftovers_compiled=self.leftovers if compiled else None)

    @staticmethod
    def words(frame,row):
        return np.asarray(frame.callbacks.words[row,:int(frame.callbacks.count[row])])
