"""Inert literal helper/export controls; no engine/FULL/model execution."""
import ast
from copy import deepcopy
from collections import namedtuple
from pathlib import Path
import subprocess,unittest
from types import SimpleNamespace as NS
from unittest.mock import patch
import jax.numpy as j
import numpy as np
from gpu_engine import engine_outcome_diagnostics as d,engine_no_damage_callbacks as nd
from gpu_engine import engine_direct_hit_controls as direct,engine_execution_completeness as ex,engine_public_events as public
import test_gpu_engine_outcome_diagnostics as old_fixture
BASE='11f946c7e2592388064c6f1908ff7f9a849c2b6b'

class NoDamageDiagnosticTests(unittest.TestCase):
 def fixture(self):
  t,_=old_fixture.OutcomeDiagnosticTests().tape();b=8;z=j.zeros(b,j.int32);yes=j.ones(b,bool)
  data=NS(candidates=j.array([False]+[True]*9),move_types=j.ones(10,j.int32)*3,dynamic_type=j.zeros(10,bool),levitate_source=j.array(True),levitate_move=j.array(7,j.int32),ground_type=j.array(3,j.int32),flying_type=j.array(5,j.int32),levitate=j.array([False,False,True]),levitate_items=j.array([True,False]),recoil_immune_move=j.array(-1,j.int32))
  before=nd.Before(yes,z+3,z+2,z,z+71,z+100,j.full(b,2**31+5,j.uint32),z,z+100,~yes,z,z+70,j.stack((z+3,z),axis=1),z+1)
  p=nd.Pending(NS(data=NS(no_damage=data)),None,z,z+2,z+4,z+7,before,NS(blocked=~yes,hp=before.hp,volatile=before.volatile),j.array([True,False]*4),z+7)
  w=direct.HitWrite(z+100,~yes,yes,yes,z+1,z+1,~yes,~yes,~yes,z+71,z+71,z+100)
  kw=dict(scheduled=~yes,code=z,matched=~yes,room=yes,use=~yes,actor_hp=z+100,volatile=before.volatile,sub_hp=z,actor_hp_type=z,actor_hp_power=z+70,target_types=before.target_types,item=z+1,ability=z+2)
  return t,p,w,kw
 def test_literal_known_malformed_zero_capacity_and_refused_host_export(self):
  t,p,w,kw=self.fixture();kw.update(target_types=j.ones(8,j.int32),item=j.ones(8,j.float32));p=p._replace(before=p.before._replace(target_types=None,item=None))
  a=d.record_no_damage_on_tape(t,p,w,**kw)
  self.assertIs(a.execution.ledger,t.execution.ledger);self.assertIs(a.execution.risks,t.execution.risks)
  self.assertIs(a.words,t.words);self.assertIs(a.refusal,t.refusal)
  for x,y in zip(a.execution.diagnostics[:17],t.execution.diagnostics[:17],strict=True):np.testing.assert_array_equal(x,y)
  out=d.refused_rows(a.execution.diagnostics,np.array([True]+[False]*7))['rows'][0]['no_damage_finish'];self.assertEqual(out['count'],1);r=out['records'][0]
  self.assertIsNone(r['target_type0']);self.assertNotIn('target_type0',r['malformed_fields']);self.assertIsNone(r['current_type0']);self.assertIn('current_type0',r['malformed_fields']);self.assertIn('current_item',r['malformed_fields'])
  self.assertEqual(r['finish_matches'],0);self.assertEqual(r['header_code'],0);self.assertEqual(r['volatile_uint32'],2**31+5)
  self.assertIsNone(r['diagnostic_groups']['allowed_item_lookup']);self.assertFalse(r['diagnostic_groups']['finish_matches'])
  self.assertEqual((r['stream_id'],r['prior_execution_refusal'],r['prior_kernel_risk']),(12001,10,20))
  b=d.record_no_damage_on_tape(a,p,w,**kw);c=d.record_no_damage_on_tape(b,p,w,**kw)
  self.assertEqual(c.execution.diagnostics.no_damage_count.tolist(),[2,0]*4);self.assertEqual(c.execution.diagnostics.no_damage_overflow.tolist(),[True,False]*4)
  np.testing.assert_array_equal(c.execution.diagnostics.no_damage_words,b.execution.diagnostics.no_damage_words)
  for capture in (None,d.Capture(*t.execution.diagnostics[:17]),t.execution.diagnostics._replace(no_damage_known=j.zeros((8,2,1),bool))):
   before=t._replace(execution=t.execution._replace(diagnostics=capture));self.assertIs(d.record_no_damage_on_tape(before,p,w,**kw),before)
  legacy=d.Capture(*a.execution.diagnostics[:17]);self.assertNotIn('no_damage_finish',d.refused_rows(legacy,np.array([True]+[False]*7))['rows'][0])
 def test_finish_forwards_already_evaluated_results_and_keeps_old_return(self):
  t,p,w,kw=self.fixture();p=p._replace(before=p.before._replace(target_types=None,item=None));yes=j.ones(8,bool);z=j.zeros(8,j.int32)
  old=ast.parse(subprocess.check_output(['git','show',BASE+':gpu_engine/engine_no_damage_callbacks.py'],text=True));fn=next(n for n in old.body if isinstance(n,ast.FunctionDef)and n.name=='finish');ns=dict(vars(nd));exec(compile(ast.Module(body=[fn],type_ignores=[]),'<old actual finish>','exec'),ns);old_finish=ns['finish']
  FinishContext=namedtuple('FinishContext','ledger tables risks diagnostics')
  t=t._replace(execution=FinishContext(t.execution.ledger,object(),t.execution.risks,t.execution.diagnostics))
  header=j.zeros((8,10),j.int32).at[:,6].set(9)
  functions=dict(scheduled=lambda *a:yes,matches=lambda *a,**k:~yes,words=lambda *a:(header,header))
  for name,value in functions.items():ns[name]=value
  with patch.multiple(nd,**functions),patch.object(public,'_append',side_effect=lambda tape,*a,**k:tape),patch.multiple(ex,_context=lambda tape:tape.execution,unhandled=lambda ledger,*a,**k:ledger,emitted=lambda ledger,*a,**k:ledger,_put_context=lambda tape,ledger,context:tape):
   args=(t,p,w,None,kw['actor_hp'],kw['volatile'],kw['sub_hp'],kw['actor_hp_type'],kw['actor_hp_power'])
   a=nd.finish(*args,target_types=None,item=None,ability=kw['ability']);b=old_finish(*args,target_types=None,item=None,ability=kw['ability'])
  np.testing.assert_array_equal(a.refusal,b.refusal);np.testing.assert_array_equal(a.words,b.words);self.assertIs(a.execution.ledger,b.execution.ledger)
  r=d.refused_rows(a.execution.diagnostics,np.array([True]+[False]*7))['rows'][0]['no_damage_finish']['records'][0]
  self.assertEqual((r['finish_scheduled'],r['header_code'],r['finish_matches']),(1,9,0))
 def test_exact_predicates_source_scope_and_one_host_copy_per_used_array(self):
  old=ast.parse(subprocess.check_output(['git','show',BASE+':gpu_engine/engine_no_damage_callbacks.py'],text=True));new=ast.parse(Path(nd.__file__).read_text());functions=lambda t:{n.name:n for n in t.body if isinstance(n,ast.FunctionDef)};a,b=functions(old),functions(new)
  for name in a:
   if name!='finish':self.assertEqual(ast.dump(a[name]),ast.dump(b[name]),name)
  current=deepcopy(b['finish']);current.body=[n for n in current.body if not(isinstance(n,ast.ImportFrom)and any(x.asname=='diagnostics'for x in n.names))and not(isinstance(n,ast.Assign)and any(isinstance(x,ast.Name)and x.id=='scheduled_result'for x in n.targets))and not(isinstance(n,ast.Assign)and isinstance(n.value,ast.Call)and isinstance(n.value.func,ast.Attribute)and n.value.func.attr=='record_no_damage_on_tape')]
  class OriginalScheduled(ast.NodeTransformer):
   def visit_Name(self,node):return ast.Call(ast.Name('scheduled',ast.Load()),[ast.Name('p',ast.Load()),ast.Name('w',ast.Load())],[])if node.id=='scheduled_result'else node
  self.assertEqual(ast.dump(OriginalScheduled().visit(current)),ast.dump(a['finish']))
  for file in ('transition.py','engine_retained_encoder_driver.py','benchmark_capture_gap_census.py','engine_execution_completeness.py','engine_kernel_frame_risk.py','engine_lifecycle_event_device.py','engine_full_v4_event_device.py'):
   self.assertEqual(Path('gpu_engine',file).read_bytes(),subprocess.check_output(['git','show',('fd3256d2305a0770764e737640fc80157156cb29'if file=='engine_kernel_frame_risk.py'else BASE)+':gpu_engine/'+file]),file)
  t,p,w,kw=self.fixture();capture=d.record_no_damage_on_tape(t,p,w,**kw).execution.diagnostics
  class HostOnly:
   def __init__(self,value):self.value=np.asarray(value);self.shape=self.value.shape;self.dtype=self.value.dtype;self.copies=0
   def __array__(self,dtype=None,copy=None):self.copies+=1;return np.asarray(self.value,dtype=dtype)
   def __getitem__(self,key):raise AssertionError('device row slicing')
  wrapped=d.Capture(*(HostOnly(v)for v in capture));out=d.refused_rows(wrapped,np.array([True,False,True,False,False,False,False,False]));self.assertEqual([x['row']for x in out['rows']],[0,2]);self.assertTrue(all(x.copies==1 for x in wrapped))
