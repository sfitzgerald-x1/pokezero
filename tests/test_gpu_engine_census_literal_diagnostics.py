"""Same-source diagnostic data only: no engine/model/game/chance invocation."""
import ast
from collections import namedtuple
from pathlib import Path
import subprocess
import sys
import unittest
import jax.numpy as j
import numpy as np
from gpu_engine import engine_outcome_diagnostics as d,benchmark_capture_gap_census as census
import test_gpu_engine_outcome_diagnostics as fixture

class LiteralCensusDiagnosticsTests(unittest.TestCase):
 def setUp(self):self.f=fixture.OutcomeDiagnosticTests();self.t,self.outcome=self.f.tape()
 def record(self,tape,active):
  b=len(active);side=j.arange(b,dtype=j.int32)%2;physical=side+3
  qualified=j.array([True,False]*4);cached=j.array([True,True,False,False]*2)
  speeds=j.stack((j.arange(b,dtype=j.int32)+100,j.arange(b,dtype=j.int32)+101),axis=1)
  guard=qualified&cached&(tape.refusal==0)
  return d.record_residual_on_tape(tape,side,physical,qualified,cached,speeds,guard,active=active)
 def test_same_literal_operands_refusal_and_risk_identity_overflow_inactive(self):
  active=j.array([True,False]*4);t=self.t
  a=self.record(t,active);b=self.f.record(a,self.outcome,active)
  self.assertIs(b.execution.ledger,t.execution.ledger);self.assertIs(b.execution.risks,t.execution.risks)
  np.testing.assert_array_equal(b.words,t.words);np.testing.assert_array_equal(b.refusal,t.refusal)
  self.assertEqual(b.execution.diagnostics.residual_count.tolist(),[1,0]*4)
  self.assertEqual(b.execution.diagnostics.count.tolist(),[1,0]*4)
  c=self.record(self.record(b,active),active)
  self.assertEqual(c.execution.diagnostics.residual_count.tolist(),[2,0]*4)
  self.assertEqual(c.execution.diagnostics.residual_overflow.tolist(),active.tolist())
  np.testing.assert_array_equal(c.refusal,t.refusal)
  row=d.refused_rows(b.execution.diagnostics,np.array([True]+[False]*7))['rows'][0]
  r=row['residual']['records'][0]
  self.assertEqual((r['side'],r['physical'],r['profile_qualified'],r['cached_speed_valid'],r['end_speed0'],r['end_speed1'],r['guard_valid']),(0,3,1,1,100,101,0))
  self.assertEqual((r['prior_tape_refusal'],r['prior_execution_refusal'],r['prior_kernel_risk']),(30,10,20))
  self.assertEqual((r['stream_id'],r['frame'],r['raw_count'],r['turn'],r['generation0'],r['generation1']),(12001,5,7,1,2,3))
  self.assertEqual(row['records'][0]['move'],91)
 def test_legacy_missing_malformed_and_refused_only_packet_slice(self):
  t=self.t;active=j.ones(8,bool);legacy=d.Capture(*t.execution.diagnostics[:4])
  before=t._replace(execution=t.execution._replace(diagnostics=legacy))
  self.assertIs(self.record(before,active),before)
  a=self.record(self.f.record(t,self.outcome,active),active).execution.diagnostics
  class CountCopies:
   def __init__(self,value):self.value=np.asarray(value);self.shape=self.value.shape;self.dtype=self.value.dtype;self.copies=0
   def __array__(self,dtype=None,copy=None):self.copies+=1;return np.asarray(self.value,dtype=dtype)
   def __getitem__(self,index):raise AssertionError('device row slice/gather')
  wrapped=d.Capture(*(CountCopies(value)for value in a));packets=[{}for _ in range(8)]
  mask=np.array([False,True,False,False,False,False,False,True])
  census._attach_refused_diagnostics(packets,wrapped,mask)
  self.assertEqual([i for i,p in enumerate(packets)if p],[1,7])
  self.assertTrue(all(value.copies==1 for value in wrapped))
  self.assertEqual(packets[7]['literal_diagnostics']['residual']['records'][0]['stream_id'],int(t.stream_id[7]))
  self.assertEqual(packets[7]['literal_diagnostics']['residual']['records'][0]['prior_tape_refusal'],int(t.refusal[7]))
  self.assertEqual(packets[7]['literal_diagnostics']['residual']['records'][0]['prior_execution_refusal'],17)
  self.assertEqual(packets[7]['literal_diagnostics']['residual']['records'][0]['turn'],8)
  census._attach_refused_diagnostics(packets,None,np.array([True]+[False]*7))
  self.assertFalse(packets[0]['literal_diagnostics']['available'])
  malformed=a._replace(residual_words=j.zeros((8,2,len(d.RESIDUAL_FIELDS)),j.float32))
  bad=t._replace(execution=t.execution._replace(diagnostics=malformed))
  self.assertIs(self.record(bad,active),bad)
 def test_original_transition_program_and_guard_exact_after_diagnostic_call_removal(self):
  old=ast.parse(subprocess.check_output(['git','show','5b702e55:gpu_engine/transition.py'],text=True))
  new=ast.parse(Path('gpu_engine/transition.py').read_text())
  class RemoveDiagnostic(ast.NodeTransformer):
   def visit_ImportFrom(self,node):
    return None if any(a.asname=='_residual_diagnostics'for a in node.names)else node
   def visit_Assign(self,node):
    call=node.value
    if isinstance(call,ast.Call)and isinstance(call.func,ast.Attribute)and isinstance(call.func.value,ast.Name)and call.func.value.id=='_residual_diagnostics':return None
    return self.generic_visit(node)
  self.assertEqual(ast.dump(old,include_attributes=False),ast.dump(RemoveDiagnostic().visit(new),include_attributes=False))
  body=Path('gpu_engine/transition.py').read_text();start=body.index('_status_guard_valid=');record=body.index('_public_trace=_residual_diagnostics.record_residual_on_tape');emit=body.index('_public_trace=status_residual_callbacks.record_status_residual_device')
  self.assertLess(start,record);self.assertLess(record,emit)
  for path in ('engine_retained_encoder_driver.py','engine_kernel_frame_risk.py','engine_status_residual_callbacks.py'):
   self.assertEqual(Path('gpu_engine',path).read_bytes(),subprocess.check_output(['git','show','5b702e55:gpu_engine/'+path]))
 def test_cold_import_orders_build_only(self):
  modules=['gpu_engine.benchmark_capture_gap_census','gpu_engine.engine_lifecycle_event_device','gpu_engine.engine_full_v4_event_device']
  for order in (modules,modules[::-1]):
   result=subprocess.run([sys.executable,'-B','-c',';'.join('import '+name for name in order)],capture_output=True,text=True,timeout=60)
   self.assertEqual(result.returncode,0,result.stderr)

if __name__=='__main__':unittest.main()
