"""Owning risk preparation/row transport; actual callsites tested separately."""
from dataclasses import replace
import unittest
from unittest.mock import patch
import jax
import jax.numpy as j
import numpy as np
from gpu_engine import engine_retained_encoder_driver as driver
from gpu_engine import engine_kernel_frame_risk as risks
import gpu_engine_owning_runtime_fixture as fixture


class KernelRiskDriverPreparationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        fixture.ExecutionRuntimePlumbingTests.setUpClass();cls.base=fixture.ExecutionRuntimePlumbingTests
        cls.plain=cls.base.prepared;r=driver._checked(cls.plain)
        cls.args=(r['request_compiled'],r['own_compiled'])
        cls.options=dict(public_mode='retained-full',leftovers_compiled=r['leftovers_compiled'],
            leftovers_mode='retained',account_source_sites=True,track_kernel_risks=True,
            fake_forward=lambda _:driver.FakeScores(j.zeros(9),j.array(0,j.int32)))
        cls.prepared=driver.prepare(*cls.args,**cls.options)

    def test_original_accounting_risk_compiler_and_fresh_host_guards(self):
        runtime=driver._checked(self.prepared);compiled=runtime['kernel_risk_compiled']
        self.assertIs(risks._COMPILED[compiled][3],runtime['execution_compiled'])
        self.assertIs(risks._COMPILED[compiled][2],runtime['request']['lifecycle'])
        self.assertIsNone(driver._checked(self.plain)['kernel_risk_compiled'])
        self.assertNotEqual(self.prepared.binding,self.plain.binding)
        for bad in (dict(account_source_sites=False),dict(track_kernel_risks=1),dict(public_mode='basic')):
            with self.assertRaises(ValueError):driver.prepare(*self.args,**dict(self.options,**bad))
        for fn in (lambda:jax.jit(lambda:driver.prepare(*self.args,**self.options).binding)(),
                   lambda:jax.jit(lambda dummy:driver.prepare(*self.args,**self.options).binding)(j.array(0))):
            with self.assertRaisesRegex(ValueError,'outside outer JIT'):fn()
        with self.assertRaises(ValueError):risks.check_compiled(replace(compiled))
        with patch.object(risks,'check_compiled',side_effect=ValueError('original risk metadata changed')):
            with self.assertRaisesRegex(ValueError,'original risk metadata changed'):driver._checked(self.prepared)

    def test_b8_core_row_transport_preserves_globals_and_latches_after_binding_drift(self):
        runtime=driver._checked(self.prepared);tables=runtime['kernel_risk_compiled'].device
        carry=risks.initialize_device(j.arange(3401,3409,dtype=j.int32),tables)
        active=j.array([True,False]*4)
        after=carry._replace(frame=carry.frame+1,observed=j.ones(8,j.int32),
            binding=carry.binding.at[7].add(j.uint32(1)))
        # Literal Core row fixture, not a Session or constructor attestation.
        before=driver.Core(j.zeros(8,j.int32),j.zeros(8,j.int32),(),(),(),(),(),
            j.ones(8,bool),j.zeros(8,j.int32),risks=carry)
        candidate=before._replace(engine=j.ones(8,j.int32),risks=after)
        result=jax.jit(driver._core_rows)(active,before,candidate)
        np.testing.assert_array_equal(result.engine,active.astype(j.int32))
        np.testing.assert_array_equal(result.risks.binding,carry.binding)
        np.testing.assert_array_equal(result.risks.accounting_binding,carry.accounting_binding)
        np.testing.assert_array_equal(result.risks.frame,active.astype(j.int32))
        np.testing.assert_array_equal(result.risks.risk,j.where(active,risks.NAMESPACE,0))
        with self.assertRaises(ValueError):driver._risk_rows(active,carry,None)

    def test_row_reducer_keeps_lifetime_risk_and_original_stream_after_corrupt_return(self):
        runtime=driver._checked(self.prepared);tables=runtime['kernel_risk_compiled'].device
        original=risks.initialize_device(j.array([3501,3502],j.int32),tables)
        original=original._replace(risk=j.array([risks.NON_WEATHER,risks.UPSTREAM],j.int32))
        returned=original._replace(risk=j.zeros(2,j.int32),stream_id=j.array([9999,3502],j.int32))
        result=jax.jit(driver._risk_rows)(j.ones(2,bool),original,returned)
        np.testing.assert_array_equal(result.stream_id,original.stream_id)
        self.assertEqual(result.risk.tolist(),[risks.NON_WEATHER|risks.NAMESPACE,risks.UPSTREAM])
        frozen=driver._risk_rows(j.array([False,True]),original,returned)
        for old,new in zip(jax.tree.leaves(original),jax.tree.leaves(frozen)):
            if old.ndim and old.shape[0]==2:np.testing.assert_array_equal(old[0],new[0])


if __name__=='__main__':unittest.main()
