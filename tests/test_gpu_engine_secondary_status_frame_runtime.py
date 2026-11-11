"""One prospective authored kernel step, real two-owner source witness, no model."""
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import jax
import jax.numpy as j
import numpy as np

from gpu_engine import engine_retained_encoder_driver as driver
from gpu_engine import engine_direct_hit_controls as direct
from gpu_engine import engine_execution_completeness as execution
from gpu_engine import engine_secondary_status_callbacks as secondary
from gpu_engine import engine_source_frame_witness as witness
from gpu_engine import engine_request_source as request
from gpu_engine import source_provenance, transition, mechanics as m
import test_gpu_engine_no_damage_owning as fixtures


class SecondaryStatusFrameRuntimeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        fixtures.NoDamageOwningTests.setUpClass(); cls.f=fixtures.NoDamageOwningTests
        old=cls.f.runtime
        cls.helper=direct.compile_direct_hit_controls(engine_catalog=cls.f.producer,
            metadata=cls.f.metadata,source_rules=request._checked(cls.f.req)['rules'])
        cls.prepared=driver.prepare(cls.f.req,old['own_compiled'],public_mode='retained-full',
            account_source_sites=True,track_kernel_risks=True,frame_input_mode='source-witness',
            leftovers_compiled=old['leftovers_compiled'],leftovers_mode='retained',
            direct_hit_compiled=cls.helper,direct_hit_mode='kernel-event',
            fake_forward=lambda _:driver.FakeScores(j.zeros(9),j.array(0,j.int32)))
        cls.runtime=driver._checked(cls.prepared)

    def test_actual_secondary_receipt_survives_end_turn_and_both_full_consumers(self):
        f=self.f; arrays=f.f.base().arrays('pressure','pressure')
        arrays['moves'][:]=np.array([f.metadata['vocabulary']['moves'].index(n) for n in f.names])
        arrays['pp'][:]=np.array([16,8,24,24]); arrays['item'][:]=0
        # Literal safe nonlethal component operands; opponent acts before PAR.
        arrays['stats'][:,:,0]=300; arrays['stats'][:,:,3]=50; arrays['stats'][:,:,4]=200
        arrays['stats'][0,:,5]=150; arrays['stats'][1,:,5]=220
        for side,name in enumerate(('pikachu','pidgeot')):
            arrays['species'][side,0]=f.metadata['vocabulary']['species'].index(name)
            types=f.dex.species_info(name).types
            arrays['types'][side,0]=[f.metadata['vocabulary']['types'].index(t.lower()) for t in types]+[0]*(2-len(types))
        with TemporaryDirectory() as directory:
            path=Path(directory)/'literal.npz'
            np.savez_compressed(path,**dict(arrays,metadata_utf8=np.frombuffer(json.dumps(f.metadata).encode(),np.uint8)))
            pool=source_provenance.load_constructor_pool(path)
            session=driver.construct(pool,np.tile([[0,1]],(2,1)).astype(np.int64),
                prepared=self.prepared,stream_ids=j.array([28001,28002],j.int32),seed=31)
        self.assertEqual(session.state.core.admitted.tolist(),[True,True])
        before=session.state.core.engine
        draws=j.zeros((2,2,12),j.int32).at[:,:,4].set(1).at[:,:,2].set(99)
        chance=transition.ChanceInputs(draws,j.zeros((2,2),j.int32),
            j.array([[0,99],[99,99]],j.int32),j.zeros((2,2),j.int32),j.zeros((2,2),j.int32),j.zeros(2,j.int32))
        actions=j.array([[3,1],[3,1]],j.int32)
        frames=[]; original=self.runtime['frame']
        def capture(*args,**kwargs):
            frame=original(*args,**kwargs); frames.append(frame); return frame
        self.runtime['frame']=capture
        try: result=driver.advance(session,actions,chance_inputs=chance)
        finally: self.runtime['frame']=original
        frame=frames[0]; tape=frame.callbacks; report=result.state.core.frame_input_report
        self.assertIsNotNone(tape.execution.secondary_status)
        np.testing.assert_array_equal(tape.execution.secondary_status.data.binding,self.helper.device.binding)
        self.assertEqual(j.sum(tape.words[:,:,0]==36,axis=1).tolist(),[1,0])
        self.assertEqual(frame.state.status[:,1,0].tolist(),[m.PAR,m.NONE])
        self.assertGreater(int(tape.cursor.turn[0]),int(tape.event_turn[0,int(j.argmax(tape.words[0,:,0]==36))]))
        plain=jax.jit(lambda state:transition.step_with_chance(state,actions,f.table,chance).state)(before)
        for actual,expected in zip(jax.tree.leaves(frame.state),jax.tree.leaves(plain),strict=True):
            np.testing.assert_array_equal(actual,expected)
        self.assertEqual(result.state.core.admitted.tolist(),[True,True],msg=json.dumps(dict(
            risk=result.state.core.risks.risk.tolist(),witness=report.refusal.tolist(),
            ledger=result.state.core.execution.refusal.tolist(),source=tape.refusal.tolist(),
            owners=[o.reason.tolist() for o in result.owners])))
        self.assertEqual(report.frame_inputs_complete.tolist(),[True,True])
        ledger=result.state.core.execution; tickets=ledger.site==int(execution.Site.SECONDARY_STATUS)
        self.assertEqual(j.sum(tickets,axis=1).tolist(),[1,0])
        self.assertTrue(bool(j.all(j.where(tickets,ledger.stage==3,True))))
        for owner in result.owners:
            self.assertEqual(owner.full_line_consumers_complete.tolist(),[True,True])
            self.assertEqual(owner.encoded.observation.admitted.tolist(),[True,True])
            self.assertFalse(bool(owner.B4_qualified.any()))
        # Missing original source context remains rejected despite intact words.
        missing=tape._replace(execution=tape.execution._replace(secondary_status=None))
        index=j.argmax(tape.words[:,:,0]==36,axis=1)
        self.assertFalse(bool(secondary.owned_pair(missing,index).any()))
        tables=witness.check_compiled(self.runtime['frame_witness_compiled'])
        self.assertEqual(witness._raw_complete(missing,tables).tolist(),[False,True])


if __name__=='__main__':unittest.main()
