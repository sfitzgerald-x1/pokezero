"""Specific actual full-HP branch accounting; no original-physics qualification."""
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
import jax
import jax.numpy as j
import numpy as np
from gpu_engine import engine_execution_completeness as accounting
from gpu_engine import engine_leftovers_no_output as no_output
from gpu_engine import engine_retained_encoder_driver as driver,source_provenance,transition
import test_gpu_engine_leftovers_no_output as fixture
import test_gpu_engine_execution_runtime_plumbing as runtime_fixture


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


class FullHPOwningRuntimeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        runtime_fixture.ExecutionRuntimePlumbingTests.setUpClass()
        cls.base=runtime_fixture.ExecutionRuntimePlumbingTests
        cls.f=cls.base.f;cls.prepared=cls.base.prepared

    def construct(self):
        arrays=self.f().arrays('pressure','pressure')
        arrays['moves'][0,0,:2]=[self.f.ids['raindance'],1]
        with TemporaryDirectory() as directory:
            path=Path(directory)/'tiny.npz'
            np.savez_compressed(path,**dict(arrays,metadata_utf8=np.frombuffer(json.dumps(self.f.metadata).encode(),np.uint8)))
            pool=source_provenance.load_constructor_pool(path)
            return driver.construct(pool,np.array([[0,1],[0,1]],np.int64),prepared=self.prepared,
                stream_ids=j.array([1201,1202],j.int32),seed=31)

    def test_actual_full_hp_attempts_require_genuine_both_owner_drains_and_reject_changed_proof(self):
        session=self.construct();before=session.state.core.engine;r=driver._checked(self.prepared)
        actual=r['frame'];frames=[]
        def capture(*args):
            frame=actual(*args);frames.append(frame);return frame
        actions=j.tile(j.array([[0,8]],j.int32),(2,1));chance=self.base().chance()
        r['frame']=capture
        try:result=driver.advance(session,actions,chance_inputs=chance)
        finally:r['frame']=actual
        self.assertEqual(len(frames),1);frame=frames[0]
        self.base().same(result.state.core.engine,transition.step_with_chance(before,actions,self.f.table,chance).state)
        self.assertEqual(result.state.core.admitted.tolist(),[True,True])
        for row in range(2):
            entered=int(frame.callbacks.execution.ledger.count[row]);self.assertEqual(entered,2)
            self.assertEqual(frame.callbacks.execution.ledger.no_output_kind[row,:entered].tolist(),[1,1])
            self.assertEqual(frame.callbacks.execution.ledger.stage[row,:entered].tolist(),[2,2])
            self.assertFalse(bool((frame.callbacks.words[row,:,0]==17).any()))
            self.assertEqual(result.state.core.execution.stage[row,:entered].tolist(),[3,3])
        self.base().same(result.state.observers[0].accounting,result.state.observers[1].accounting)
        self.assertTrue(bool(accounting.report(result.state.core.execution,tables=r['execution_compiled'].device).scoped_drained.all()))
        for owner in result.owners:self.assertFalse(bool(owner.scored.any()));self.assertFalse(bool(owner.B4_qualified.any()))
        # A malformed proof is fault injection into the actual frame, not a
        # fabricated consumer result or a reset of an existing Session.
        for corruption in ('binding','physical','generation'):
            failed_session=self.construct()
            def corrupt(*args):
                frame=actual(*args);tape=frame.callbacks;ledger=tape.execution.ledger
                if corruption=='binding':
                    ledger=ledger._replace(no_output_binding=ledger.no_output_binding.at[0,0,0].add(j.uint32(1)))
                elif corruption=='physical':
                    ledger=ledger._replace(no_output_facts=ledger.no_output_facts.at[0,0,0].set(
                        (ledger.no_output_facts[0,0,0]+1)%6))
                else:ledger=ledger._replace(generation=ledger.generation.at[0,0,0].add(1))
                return frame._replace(callbacks=tape._replace(execution=tape.execution._replace(ledger=ledger)))
            r['frame']=corrupt
            try:failed=driver.advance(failed_session,actions,chance_inputs=chance)
            finally:r['frame']=actual
            self.assertEqual(failed.state.core.admitted.tolist(),[False,True],msg=corruption)
            self.assertEqual(failed.state.core.execution.stage[0,:2].tolist(),[2,2])
            self.assertEqual(failed.state.core.execution.stage[1,:2].tolist(),[3,3])
            self.assertTrue(int(failed.state.core.execution.refusal[0])&accounting.UNRESOLVED)
            for owner in failed.owners:
                self.assertFalse(bool(owner.encoded.observation.attention_mask[0].any()))
                self.assertFalse(bool(owner.scored.any()));self.assertFalse(bool(owner.B4_qualified.any()))


if __name__=='__main__':unittest.main()
