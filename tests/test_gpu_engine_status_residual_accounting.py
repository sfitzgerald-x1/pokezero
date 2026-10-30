"""Exact typed status-span transport, never source/history/drain admission."""
import unittest
import jax.numpy as j
from gpu_engine import engine_execution_completeness as accounting,engine_public_events as events
import test_gpu_engine_engine_execution_completeness as fixture


class StatusResidualAccountingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        fixture.ExecutionAccountingTests.setUpClass();cls.f=fixture.ExecutionAccountingTests()

    def test_current_physical_target_required_and_capture_still_is_not_consumption(self):
        tape=self.f.tape();tape=accounting.begin_on_tape(tape,self.f.ledger(),self.f.tables,active=j.ones(2,bool))
        tape,ticket=accounting.open_attempt(tape,accounting.Site.STATUS_RESIDUAL,0,active=j.ones(2,bool))
        physical=tape.cursor.physical_active[:,0]
        # Explicit trusted synthetic span; this does not claim an actual
        # registered caller, original hooks or owning source history.
        emitted=events._append(tape,20,j.zeros(2,j.int32),j.zeros(2,j.int32),j.zeros(2,j.int32),physical,
            (j.full(2,100,j.int32),j.full(2,90,j.int32),j.full(2,100,j.int32),
             j.full(2,2,j.int32),j.ones(2,j.int32)),j.ones(2,bool))
        captured=accounting.capture_status_residual_write(emitted,ticket,
            requested=j.ones(2,bool),positive_write=j.ones(2,bool))
        self.assertEqual(captured.execution.ledger.stage[:,0].tolist(),[2,2])
        self.assertFalse(bool(accounting.report(captured.execution.ledger,tables=self.f.tables).scoped_drained.any()))
        bad=emitted._replace(words=emitted.words.at[0,0,4].set((physical[0]+1)%6))
        failed=accounting.capture_status_residual_write(bad,ticket,
            requested=j.ones(2,bool),positive_write=j.ones(2,bool))
        self.assertEqual(failed.execution.ledger.stage[:,0].tolist(),[1,2])
        self.assertTrue(int(failed.refusal[0])&accounting.UNKNOWN_EXECUTION)
        self.assertEqual(int(failed.execution.ledger.refusal[1]),0)


if __name__=='__main__':unittest.main()
