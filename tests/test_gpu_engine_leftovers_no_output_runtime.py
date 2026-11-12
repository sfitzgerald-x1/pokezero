"""Specific actual full-HP branch accounting; no original-physics qualification.

The runtime class that ran both owner drains (FullHPOwningRuntimeTests) is covered by
test_gpu_engine_owning_leftovers_no_output.
"""
import unittest
import jax
import jax.numpy as j
from gpu_engine import engine_execution_completeness as accounting
from gpu_engine import engine_leftovers_no_output as no_output
import test_gpu_engine_leftovers_no_output as fixture


class FullHPAccountingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        fixture.LeftoversNoOutputTests.setUpClass();cls.f=fixture.LeftoversNoOutputTests()

    def proof(self,b=2):
        tape,ticket,state,record,event=self.f.boundary(b=b)
        before=self.f.capture(tape,ticket,state,record,event)
        receipt=no_output.finish_device(before,tape,self.f.actual_write(state,event),record,self.f.tables)
        return tape,receipt

    def test_specific_branch_facts_capture_without_generic_empty_span_or_consumption(self):
        tape,receipt=self.proof()
        captured=accounting.capture_leftovers_full_hp(tape,receipt,no_output_tables=self.f.tables)
        ledger=captured.execution.ledger
        self.assertEqual(ledger.stage[:,0].tolist(),[2,2])
        self.assertEqual(ledger.no_output_kind[:,0].tolist(),[1,1])
        self.assertEqual(ledger.end[:,0].tolist(),ledger.start[:,0].tolist())
        self.assertFalse(bool(ledger.claimed.any()))
        self.assertFalse(bool(accounting.report(ledger,tables=self.f.accounting.device).scoped_drained.any()))
        self.f.same(ledger.no_output_binding[:,0],j.broadcast_to(self.f.tables.binding,(2,8)))
        self.f.same(tape.words,captured.words);self.f.same(tape.refusal,captured.refusal)
        spent=accounting.capture_leftovers_full_hp(captured,receipt,no_output_tables=self.f.tables)
        self.assertTrue(bool((spent.refusal&accounting.UNKNOWN_EXECUTION).all()))
        self.assertTrue(bool((spent.execution.ledger.refusal!=0).all()))

    def test_receipt_fields_or_later_storage_mutation_refuse_only_bad_row(self):
        tape,receipt=self.proof()
        for label in ('afterHP','item','side','pairedTarget','reason','prefix','binding'):
            raw=tape;proof=receipt
            if label=='afterHP':proof=proof._replace(after_hp=proof.after_hp.at[0].add(-1))
            elif label=='item':proof=proof._replace(before=proof.before._replace(item=proof.before.item.at[0].set(0)))
            elif label=='side':proof=proof._replace(before=proof.before._replace(side=proof.before.side.at[0].set(1)))
            elif label=='pairedTarget':
                target=(proof.before.physical[0]+1)%6
                proof=proof._replace(before=proof.before._replace(physical=proof.before.physical.at[0].set(target)),
                    after_physical=proof.after_physical.at[0].set(target))
            elif label=='reason':proof=proof._replace(reason=proof.reason.at[0].set(no_output.AFTER))
            elif label=='prefix':raw=raw._replace(words=raw.words.at[0,63,0].set(1))
            else:proof=proof._replace(before=proof.before._replace(binding=proof.before.binding.at[0].add(j.uint32(1))))
            result=accounting.capture_leftovers_full_hp(raw,proof,no_output_tables=self.f.tables)
            self.assertEqual(result.execution.ledger.stage[:,0].tolist(),[1,1] if label=='binding' else [1,2],msg=label)
            self.assertTrue(int(result.refusal[0])&accounting.UNKNOWN_EXECUTION,msg=label)
            if label!='binding':self.assertEqual(int(result.execution.ledger.refusal[1]),0,msg=label)
        with self.assertRaises(ValueError):accounting.capture_leftovers_full_hp(tape,True,no_output_tables=self.f.tables)

    def test_b8_branch_bindings_are_row_data_and_inventory_is_global(self):
        tape,receipt=self.proof(8)
        proof=receipt._replace(reason=receipt.reason.at[1].set(no_output.AFTER))
        result=jax.jit(lambda t,r:accounting.capture_leftovers_full_hp(t,r,no_output_tables=self.f.tables))(tape,proof)
        self.assertEqual(result.execution.ledger.stage[:,0].tolist(),[2,1,2,2,2,2,2,2])
        self.assertEqual(result.execution.ledger.no_output_binding.shape,(8,16,8))
        self.f.same(result.execution.ledger.inventory_binding,tape.execution.ledger.inventory_binding)


if __name__=='__main__':unittest.main()
