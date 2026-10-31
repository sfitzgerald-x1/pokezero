"""Original guarded tiny session/forced request/encoder; fake scores, no model."""
from pathlib import Path
from tempfile import TemporaryDirectory
import json,unittest
import jax
import jax.numpy as j
import numpy as np
from gpu_engine import engine_retained_encoder_driver as driver,engine_direct_hit_controls as hit
from gpu_engine import source_provenance,transition,engine_execution_completeness as ex
import test_gpu_engine_execution_runtime_plumbing as fixture

class DirectKOForcedRuntimeTests(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  fixture.ExecutionRuntimePlumbingTests.setUpClass();cls.base=fixture.ExecutionRuntimePlumbingTests;cls.f=cls.base.f
  original=driver._checked(cls.base.prepared);cls.helper=hit.compile_direct_hit_controls(engine_catalog=original['request']['lifecycle'])
  cls.prepared=driver.prepare(original['request_compiled'],original['own_compiled'],public_mode='retained-full',leftovers_compiled=original['leftovers_compiled'],leftovers_mode='retained',account_source_sites=True,track_kernel_risks=True,direct_hit_compiled=cls.helper,direct_hit_mode='kernel-event',frame_input_mode='source-witness',fake_forward=lambda _:driver.FakeScores(j.ones(9),j.array(1,j.int32)))
 def same(self,a,b):
  for x,y in zip(jax.tree.leaves(a),jax.tree.leaves(b)):np.testing.assert_array_equal(x,y)
 def test_actual_guarded_ko_drains_both_then_original_force_wait_private_encoder(self):
  arrays=self.f().arrays('pressure','pressure');arrays['moves'][0,0,:2]=[self.f.ids['hail'],1]
  arrays['stats'][0,:,1]=600;arrays['stats'][1,:,2]=10
  with TemporaryDirectory() as directory:
   path=Path(directory)/'literal.npz';np.savez_compressed(path,**dict(arrays,metadata_utf8=np.frombuffer(json.dumps(self.f.metadata).encode(),np.uint8)))
   pool=source_provenance.load_constructor_pool(path);session=driver.construct(pool,np.tile([[0,1]],(2,1)).astype(np.int64),prepared=self.prepared,stream_ids=j.arange(5101,5103,dtype=j.int32),seed=31)
  before=session.state;self.assertEqual(before.core.admitted.tolist(),[True,True]);self.assertEqual(session.result.state.core.frame_input_report.frame_inputs_complete.tolist(),[True,True])
  chance=self.base().chance();actions=j.array([[1,0],[1,0]],j.int32);result=driver.advance(session,actions,chance_inputs=chance)
  runtime=driver._checked(self.prepared);plain=transition.step_with_chance(before.core.engine,actions,runtime['request']['engine_table'],chance);self.same(result.state.core.engine,plain.state)
  self.assertEqual(result.state.core.engine.done.tolist(),[False,False]);self.assertEqual(result.state.core.engine.forced_switch.tolist(),[[False,True]]*2);self.assertEqual(result.state.core.engine.pending_residual.tolist(),[True,True])
  self.assertEqual(result.state.core.admitted.tolist(),[True,True]);self.assertEqual(result.state.core.frame_input_report.frame_inputs_complete.tolist(),[True,True]);self.assertEqual(result.state.core.risks.risk.tolist(),[0,0])
  ledger=result.state.core.execution;self.assertEqual(ledger.stage[:,0].tolist(),[3,3]);self.assertTrue(bool(ex.report(ledger,tables=runtime['execution_compiled'].device).scoped_drained.all()))
  for side in (0,1):
   req=result.state.core.requests[side].actions.facts.request
   self.assertEqual(req.force_switch.tolist(),[side==1]*2);self.assertEqual(req.wait.tolist(),[side==0]*2)
   self.assertFalse(bool(result.owners[side].B4_qualified.any()));self.assertTrue(bool(result.owners[side].full_line_consumers_complete.all()))
  self.assertFalse(bool(result.owners[1].full_encoder_admitted.any()))
  self.assertEqual(result.owners[1].encoded.observation.admitted.tolist(),[True,True],msg=str(dict(reason=result.owners[1].reason.tolist(),own=result.owners[1].own.reason.tolist())))
  self.assertTrue(bool(j.isfinite(result.owners[1].encoded.observation.numeric_features).all()))
  self.assertEqual(result.owners[1].encoded.observation.legal_action_mask[:,:4].tolist(),[[False]*4]*2);self.assertTrue(bool(result.owners[1].encoded.observation.legal_action_mask[:,4:].all()))
