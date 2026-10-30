"""Actual constructor/lifecycle calls on tiny authored component arrays only.

Requires the explicitly declared engine_lifecycle_callbacks source dependency.
No generated roster, complete game, model or scientific measurement is run.
"""
from unittest.mock import patch
from dataclasses import replace
import unittest
import jax
import jax.numpy as j
import numpy as np

from gpu_engine import engine_request_source as source
from gpu_engine import engine_private_actions as actions
from gpu_engine import engine_lifecycle_callbacks as lifecycle
import test_gpu_engine_request_source as fixtures


class RequestConstructorLifecycleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        fixtures.RequestSourceTests.setUpClass()
        cls.f=fixtures.RequestSourceTests()

    def construct(self,**kwargs):
        defaults=dict(compiled=self.f.compiled,stream_ids=j.array([71,72],j.int32),seed=31)
        defaults.update(kwargs)
        return source.construct_from_pool(self.f.pool,np.array([[0,1],[0,1]],np.int64),**defaults)

    def test_same_real_constructor_inputs_gender_then_initial_reveals_and_requests(self):
        streams=np.array([71,72],np.int32);indices=np.array([[0,1],[0,1]],np.int64)
        with (patch.object(source.transition,'state_from_pool',wraps=source.transition.state_from_pool) as physical,
                patch.object(actions,'retain_constructor_pp',wraps=actions.retain_constructor_pp) as retained):
            out=source.construct_from_pool(self.f.pool,indices,compiled=self.f.compiled,stream_ids=streams,seed=31)
        self.assertEqual(physical.call_count,1);self.assertEqual(retained.call_count,2)
        self.assertIs(physical.call_args.args[0],self.f.pool.arrays)
        self.assertIsNone(physical.call_args.kwargs['table'])
        for call in retained.call_args_list:
            self.assertIs(call.args[0],self.f.pool)
            self.assertIs(call.args[1],physical.call_args.args[1])
        self.assertEqual(out.callbacks.words[:,:3,0].tolist(),[[8,8,4]]*2)
        self.assertEqual(out.registry.generation.tolist(),[[0,0]]*2)
        self.assertEqual(out.registry.established.tolist(),[[True,True]]*2)
        self.assertTrue(bool(j.all(out.state.gender[out.state.species>0]>=0)))
        for ledger,reg,produced in zip(out.orders,out.registrations,out.requests):
            self.assertEqual(produced.actions.admitted.tolist(),[True,True])
            self.f.compare(out.state,ledger,reg,out.registry)
        emitted=(out.state,out.cursor,out.orders,out.requests,out.callbacks,out.registry)
        baseline=jax.tree.map(lambda x:np.asarray(x).copy(),emitted)
        streams[0]=171;indices[:]=2
        for old,current in zip(jax.tree.leaves(baseline),jax.tree.leaves(emitted)):
            np.testing.assert_array_equal(old,np.asarray(current))
        with self.assertRaises(ValueError):
            source.advance_source(replace(out),j.zeros((2,2),j.int32),compiled=self.f.compiled)

    def test_real_installed_switch_advances_order_before_next_source_request(self):
        old=self.construct()
        out=source.advance_source(old,j.array([[4,0],[4,0]],j.int32),compiled=self.f.compiled)
        self.assertEqual(out.registry.generation.tolist(),[[1,0]]*2)
        self.assertEqual(out.orders[0].generation.tolist(),[1,1])
        self.assertEqual(out.orders[1].generation.tolist(),[0,0])
        for ledger,reg,produced in zip(out.orders,out.registrations,out.requests):
            self.assertEqual(produced.actions.admitted.tolist(),[True,True])
            self.f.compare(out.state,ledger,reg,out.registry)
        self.assertEqual(out.orders[0].request_to_physical[0].tolist(),[1,0,2,-1,-1,-1])

    def test_missing_causal_reveals_never_admit_request_features(self):
        out=self.construct(capacity=1)
        self.assertTrue(bool((out.callbacks.refusal&2).any()))
        for produced in out.requests:
            self.assertFalse(bool(produced.actions.admitted.any()))
            self.assertFalse(bool(produced.actions.features.numeric.any()))
            self.assertFalse(bool(produced.actions.features.categorical.any()))
            self.assertTrue(bool((produced.reason&source.ARRIVAL).all()))


if __name__=='__main__':unittest.main()
