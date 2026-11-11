"""Owning accounting at B=8: all-inactive and mixed-mask advances keep globals and private history; eager frame.

Light replacement coverage for execution_runtime_plumbing's T2.4, through the real constructor and advance
(row selection of the core, both observers and both reports, including accounting commit_rows with the
mixed mask), with the frame phase running the prepared frame closure's own body eagerly
(tests/gpu_engine_eager_frame.py, which traces it under jit once).
"""
import unittest

import jax.numpy as j

from gpu_engine import engine_retained_encoder_driver as d
from gpu_engine_eager_frame import eager_frame
import test_gpu_engine_execution_runtime_plumbing as plumbing


class ExecutionPlumbingB8RowTests(unittest.TestCase):
    construct=plumbing.ExecutionRuntimePlumbingTests.construct
    chance=plumbing.ExecutionRuntimePlumbingTests.chance
    same=plumbing.ExecutionRuntimePlumbingTests.same

    @classmethod
    def setUpClass(cls):
        plumbing.ExecutionRuntimePlumbingTests.setUpClass();base=plumbing.ExecutionRuntimePlumbingTests
        cls.f=base.f;cls.prepared=base.prepared

    def test_b8_zero_and_mixed_rows_preserve_globals_and_private_history(self):
        session=self.construct(8);before=session.result
        with eager_frame(self.prepared):
            result=d.advance(session,j.full((8,2),8,j.int32),active_rows=j.zeros(8,bool),chance_inputs=self.chance(8))
            self.same(result,before)
            mask=j.array([False,True]*4)
            later=d.advance(session,j.full((8,2),8,j.int32),active_rows=mask,chance_inputs=self.chance(8))
        self.assertEqual(later.state.core.execution.frame.tolist(),[1,2]*4)
        self.assertEqual(later.state.core.frame_number.tolist(),[0,1]*4)
        self.same(later.state.core.execution.inventory_binding,before.state.core.execution.inventory_binding)
        for old,new in zip(before.state.observers,later.state.observers):
            self.same(new.accounting.inventory_binding,old.accounting.inventory_binding)
            self.same(d._observer_rows(mask,old,new),new)
        self.same(d._core_rows(mask,before.state.core._replace(originals=None,constructors=None),
            later.state.core._replace(originals=None,constructors=None)),later.state.core._replace(originals=None,constructors=None))


if __name__=='__main__':unittest.main()
