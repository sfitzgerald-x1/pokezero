"""Fixture for owning partial accounting plumbing (not a test module).

ExecutionRuntimePlumbingTests keeps the fixture API of the former test_gpu_engine_execution_runtime_plumbing:
setUpClass (f, plain, prepared), same, construct and chance. Its tests run in
test_gpu_engine_execution_plumbing_closure, test_gpu_engine_execution_plumbing_b8_rows and
test_gpu_engine_owning_status_drain (the heal drain). Its per-case setUp cache release, the leftovers capture
seam and advance_with_capture served only those tests and are not kept.
"""
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import jax
import jax.numpy as j
import numpy as np

from gpu_engine import engine_retained_encoder_driver as d
from gpu_engine import source_provenance
import test_gpu_engine_leftovers_registered_runtime as fixtures


class ExecutionRuntimePlumbingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        fixtures.RegisteredLeftoversRuntimeTests.setUpClass();cls.f=fixtures.RegisteredLeftoversRuntimeTests.f
        cls.plain=fixtures.RegisteredLeftoversRuntimeTests.prepared
        r=d._checked(cls.plain)
        cls.prepared=d.prepare(r['request_compiled'],r['own_compiled'],public_mode='retained-full',
            leftovers_compiled=r['leftovers_compiled'],leftovers_mode='retained',account_source_sites=True,
            fake_forward=lambda _:d.FakeScores(j.ones(9),j.array(1,j.int32)))

    def same(self,a,b):
        for x,y in zip(jax.tree.leaves(a),jax.tree.leaves(b)):np.testing.assert_array_equal(x,y)

    def construct(self,b=2,prepared=None):
        arrays=self.f().arrays('pressure','pressure');arrays['moves'][0,0,:2]=[self.f.ids['hail'],1]
        with TemporaryDirectory() as directory:
            path=Path(directory)/'literal.npz'
            np.savez_compressed(path,**dict(arrays,metadata_utf8=np.frombuffer(json.dumps(self.f.metadata).encode(),np.uint8)))
            pool=source_provenance.load_constructor_pool(path)
            return d.construct(pool,np.tile([[0,1]],(b,1)).astype(np.int64),prepared=prepared or self.prepared,
                stream_ids=j.arange(1101,1101+b,dtype=j.int32),seed=31)

    def chance(self,b=2):
        x=j.zeros((b,2,12),j.int32).at[:,:,4].set(1)
        from gpu_engine import transition
        return transition.ChanceInputs(x,*[j.zeros((b,2),j.int32) for _ in range(4)],j.zeros(b,j.int32))
