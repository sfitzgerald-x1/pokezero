"""Actual full-HP Leftovers no-output attempts through both real owner drains; eager frame, no models/games.

Light replacement coverage for leftovers_no_output_runtime's runtime class (FullHPOwningRuntimeTests),
with the expectations of the current kernel (its turn-loop prefix entry). The prepared runtime, constructor
and requests/public/join/encode/score phases are the real ones, compiled once; the frame phase runs the
prepared frame closure's own body eagerly (tests/gpu_engine_eager_frame.py, which also traces it under jit
once). The compiled accounting frame stays covered by the heavy owning_drain_end_to_end module.
"""
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import jax.numpy as j
import numpy as np

from gpu_engine import engine_execution_completeness as accounting
from gpu_engine import engine_lifecycle_callbacks as life
from gpu_engine import engine_retained_encoder_driver as driver, source_provenance, transition
from gpu_engine_eager_frame import eager_call
import test_gpu_engine_execution_runtime_plumbing as runtime_fixture


class OwningLeftoversNoOutputTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Host-only: the same accounting prepared runtime as the heavy module.
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
            frame=eager_call(actual,*args);frames.append(frame);return frame
        actions=j.tile(j.array([[0,8]],j.int32),(2,1));chance=self.base().chance()
        r['frame']=capture
        try:result=driver.advance(session,actions,chance_inputs=chance)
        finally:r['frame']=actual
        self.assertEqual(len(frames),1);frame=frames[0]
        self.base().same(result.state.core.engine,transition.step_with_chance(before,actions,self.f.table,chance).state)
        self.assertEqual(result.state.core.admitted.tolist(),[True,True])
        for row in range(2):
            ledger=frame.callbacks.execution.ledger
            entered=int(ledger.count[row]);self.assertEqual(entered,3)
            self.assertEqual(ledger.site[row,:entered].tolist(),[int(accounting.Site.TURNLOOP_PREFIX),int(accounting.Site.LEFTOVERS),int(accounting.Site.LEFTOVERS)])
            self.assertEqual(ledger.no_output_kind[row,:entered].tolist(),[0,1,1])
            self.assertEqual(ledger.stage[row,:entered].tolist(),[2,2,2])
            prefix=int(ledger.start[row,0]);self.assertEqual(int(ledger.end[row,0]),prefix+1)
            self.assertEqual(frame.callbacks.words[row,prefix].tolist(),[life.TURNLOOP_PREFIX,-1,-1,-1,-1,0,0,0,0,0])
            self.assertFalse(bool((frame.callbacks.words[row,:,0]==17).any()))
            self.assertEqual(result.state.core.execution.stage[row,:entered].tolist(),[3,3,3])
        self.base().same(result.state.observers[0].accounting,result.state.observers[1].accounting)
        self.assertTrue(bool(accounting.report(result.state.core.execution,tables=r['execution_compiled'].device).scoped_drained.all()))
        for owner in result.owners:self.assertFalse(bool(owner.scored.any()));self.assertFalse(bool(owner.B4_qualified.any()))
        # A malformed proof is fault injection into the actual frame, not a
        # fabricated consumer result or a reset of an existing Session.
        for corruption in ('binding','physical','generation'):
            failed_session=self.construct()
            def corrupt(*args):
                frame=eager_call(actual,*args);tape=frame.callbacks;ledger=tape.execution.ledger
                slots=np.flatnonzero(np.asarray(ledger.site[0,:int(ledger.count[0])])==int(accounting.Site.LEFTOVERS))
                self.assertEqual(slots.tolist(),[1,2]);slot=int(slots[0])
                if corruption=='binding':
                    ledger=ledger._replace(no_output_binding=ledger.no_output_binding.at[0,slot,0].add(j.uint32(1)))
                elif corruption=='physical':
                    ledger=ledger._replace(no_output_facts=ledger.no_output_facts.at[0,slot,0].set(
                        (ledger.no_output_facts[0,slot,0]+1)%6))
                else:ledger=ledger._replace(generation=ledger.generation.at[0,slot,0].add(1))
                return frame._replace(callbacks=tape._replace(execution=tape.execution._replace(ledger=ledger)))
            r['frame']=corrupt
            try:failed=driver.advance(failed_session,actions,chance_inputs=chance)
            finally:r['frame']=actual
            self.assertEqual(failed.state.core.admitted.tolist(),[False,True],msg=corruption)
            self.assertEqual(failed.state.core.execution.stage[0,:3].tolist(),[2,2,2])
            self.assertEqual(failed.state.core.execution.stage[1,:3].tolist(),[3,3,3])
            self.assertTrue(int(failed.state.core.execution.refusal[0])&accounting.UNRESOLVED)
            for owner in failed.owners:
                self.assertFalse(bool(owner.encoded.observation.attention_mask[0].any()))
                self.assertFalse(bool(owner.scored.any()));self.assertFalse(bool(owner.B4_qualified.any()))


if __name__=='__main__':unittest.main()
