"""Inert source/serialization diagnostics only; no engine turn or FULL."""
import ast,copy,hashlib,subprocess,unittest
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import patch
import jax.numpy as j
import jax
import numpy as np
from gpu_engine import engine_baton_pass_callbacks as baton,engine_outcome_diagnostics as d,state
from gpu_engine import benchmark_capture_gap_census as census
import test_gpu_engine_outcome_diagnostics as fixture
BASE='ab04b1e48396f7424490fb6086d50e3801bdfaa2'

class BatonStartDiagnosticsTests(unittest.TestCase):
 def test_small_jit_records_fresh_operands_without_gate_result_cache(self):
  t,_=fixture.OutcomeDiagnosticTests().tape();active=j.ones(8,bool)
  fast=jax.jit(lambda value:d.record_baton_start_on_tape(t,{'choice_move':value},active=active).execution.diagnostics)
  first=fast(j.full(8,18,j.int32));second=fast(j.zeros(8,j.int32));col=d.BATON_START_FIELDS.index('choice_move')
  self.assertEqual(first.baton_start_words[:,0,col].tolist(),[18]*8)
  self.assertEqual(second.baton_start_words[:,0,col].tolist(),[0]*8)
  self.assertEqual(first.baton_start_count.tolist(),[1]*8);self.assertEqual(second.baton_start_count.tolist(),[1]*8)
 def example(self,transaction=True):
  t,_=fixture.OutcomeDiagnosticTests().tape();b=8;r=j.arange(b);side=r%2;phys=j.zeros(b,j.int32);yes=j.ones(b,bool);z=j.zeros(b,j.int32)
  s=state.empty_state(b)._replace(hp=j.full((b,2,6),253,j.int32),max_hp=j.full((b,2,6),253,j.int32),ability=j.full((b,2,6),44,j.int32),base_ability=j.full((b,2,6),44,j.int32),types=j.ones((b,2,6,2),j.int32),base_types=j.ones((b,2,6,2),j.int32))
  s=s._replace(choice_move=s.choice_move.at[r,side,phys].set(j.where(r%2==0,18,0)),volatile=s.volatile.at[0,0,0].set(j.uint32(0x80000001)))
  data=NS(move_id=j.array(18,j.int32),quiet_abilities=j.ones(81,bool),quiet_items=j.ones(100,bool))
  tr=None if not transaction else NS(history=NS(side=z-1),public=NS(known=j.ones((b,2),bool),pending=j.zeros((b,2),bool),substitute=j.zeros((b,2),bool),boosts=j.zeros((b,2,7),j.int32)))
  got=baton._start_diagnostics(t,s,side,z+18,z+18,data,acts=yes,eligible=r%2==0,transaction=tr,private=None if tr is None else baton.snapshot(s),registered=yes,data_bound=yes,valid=j.zeros(b,bool))
  return t,s,got
 def test_exact_operands_unknowns_mixed_rows_sticky_and_capacity(self):
  t,s,a=self.example();cap=a.execution.diagnostics
  self.assertIs(a.execution.ledger,t.execution.ledger);self.assertIs(a.execution.risks,t.execution.risks)
  for n in t._fields:
   if n!='execution':self.assertIs(getattr(a,n),getattr(t,n))
  self.assertEqual(cap.baton_start_count.tolist(),[1,0]*4)
  item=d.refused_rows(cap,np.array([True]+[False]*7))['rows'][0]['baton_start'];r=item['records'][0]
  self.assertEqual((r['ability'],r['item'],r['choice_move'],r['volatile_uint32']),(44,0,18,0x80000001))
  self.assertEqual((r['hp'],r['maximum'],r['status'],r['counter']),(253,253,0,0))
  self.assertEqual((r['stream_id'],r['frame'],r['raw_count'],r['prior_tape_refusal'],r['prior_execution_refusal'],r['prior_kernel_risk']),(12001,5,7,30,10,20))
  self.assertFalse(r['guard_values']['choice_move_clear']);self.assertFalse(r['guard_values']['volatile_uint32_clear']);self.assertTrue(r['guard_values']['quiet_ability'])
  _,_,missing=self.example(False);r=d.refused_rows(missing.execution.diagnostics,np.array([True]+[False]*7))['rows'][0]['baton_start']['records'][0]
  self.assertEqual(r['transaction_present'],0);self.assertIsNone(r['public_known']);self.assertIsNone(r['history_side']);self.assertIsNone(r['guard_values']['public_known'])
  active=j.array([True,False]*4);values={'choice_move':j.arange(8,dtype=j.int32)}
  second=d.record_baton_start_on_tape(a,values,active=active);third=d.record_baton_start_on_tape(second,values,active=active)
  self.assertEqual(third.execution.diagnostics.baton_start_count.tolist(),[2,0]*4);self.assertEqual(third.execution.diagnostics.baton_start_overflow.tolist(),active.tolist())
  np.testing.assert_array_equal(third.execution.diagnostics.baton_start_words,second.execution.diagnostics.baton_start_words)
  for bad in(None,d.Capture(*cap[:12]),cap._replace(baton_start_known=j.zeros((8,2,len(d.BATON_START_FIELDS)),j.int32))):
   before=t._replace(execution=t.execution._replace(diagnostics=bad));self.assertIs(d.record_baton_start_on_tape(before,values,active=active),before)
 def test_host_copy_legacy_and_additive_source_split_counts(self):
  t,_,a=self.example();cap=a.execution.diagnostics
  old={};exec(compile(subprocess.check_output(['git','show',BASE+':gpu_engine/engine_outcome_diagnostics.py'],text=True),'old-diagnostic','exec'),old)
  legacy=d.Capture(*cap[:12]);mask=np.array([True,False]*4)
  self.assertEqual(d.refused_rows(legacy,mask),old['refused_rows'](old['Capture'](*cap[:12]),mask))
  class Copies:
   def __init__(self,x):self.x=np.asarray(x);self.shape=self.x.shape;self.dtype=self.x.dtype;self.n=0
   def __array__(self,dtype=None,copy=None):self.n+=1;return np.asarray(self.x,dtype=dtype)
   def __getitem__(self,k):raise AssertionError('device row indexing')
  wrapped=d.Capture(*(Copies(v)for v in cap));rows=d.refused_rows(wrapped,mask)['rows'];self.assertEqual([r['row']for r in rows],[0,2,4,6]);self.assertTrue(all(x.n==1 for x in wrapped))
  self.assertEqual(d.refused_rows(wrapped,np.zeros(8,bool))['rows'],[]);self.assertTrue(all(x.n==1 for x in wrapped))
  packet={'unhandled_sites':[39],'context_unknown':False,'literal_diagnostics':rows[0]};plan=census.Plan(max_examples_per_kind=1);streams=np.arange(2048,dtype=np.int32)+1
  x=census.Ledger(plan,streams);y=census.Ledger(plan,streams);active=np.zeros(2048,bool);active[[0,512]]=True;zero=np.zeros(2048,bool)
  with patch.object(census.time,'time',return_value=123.):
   for ledger in(x,y):ledger.mark_capture(active,step=0,packets=lambda r:packet)
   with patch.object(census,'_baton_start_predicates',return_value=[]):x.observe(step=0,active=active,executed=active,engine_failed=zero,done=zero,turns=np.ones(2048,int),packet=lambda r:packet,terminal=lambda r:None)
   y.observe(step=0,active=active,executed=active,engine_failed=zero,done=zero,turns=np.ones(2048,int),packet=lambda r:packet,terminal=lambda r:None)
  xx=x.summary();yy=y.summary()
  for field in('unsupported','unsupported_by_source'):yy[field]=[r for r in yy[field]if not r['kind'].startswith('baton-start-failed-guard:')]
  self.assertEqual(xx,yy)
  self.assertIn('choice_move_clear',census._baton_start_predicates(packet));self.assertEqual(census._baton_start_predicates({}),[])
 def test_observer_only_ast_source_defaults_and_pins(self):
  dump=lambda n:ast.dump(n,include_attributes=False)
  old=ast.parse(subprocess.check_output(['git','show',BASE+':gpu_engine/engine_baton_pass_callbacks.py'],text=True));new=ast.parse(Path('gpu_engine/engine_baton_pass_callbacks.py').read_text())
  class Remove(ast.NodeTransformer):
   def visit_FunctionDef(self,n):return None if n.name=='_start_diagnostics'else self.generic_visit(n)
   def visit_Assign(self,n):
    if isinstance(n.value,ast.Call)and isinstance(n.value.func,ast.Name)and n.value.func.id=='_start_diagnostics':return None
    if any(isinstance(x,ast.Name)and x.id in('prior','data_bound')for x in n.targets):return None
    return self.generic_visit(n)
   def visit_Name(self,n):return ast.Call(ast.Name('_bound',ast.Load()),[ast.Name('tape',ast.Load()),ast.Name('d',ast.Load())],[])if n.id=='data_bound'and isinstance(n.ctx,ast.Load)else n
  self.assertEqual(dump(old),dump(Remove().visit(new)))
  # Historical whole-census delta belongs to this observer component;
  # party-order transport has a separate current behavioral/source review.
  old=ast.parse(subprocess.check_output(['git','show',BASE+':gpu_engine/benchmark_capture_gap_census.py'],text=True));new=ast.parse(subprocess.check_output(['git','show','47f8dd99dd600388ecd72932068c7308f1f8136f:gpu_engine/benchmark_capture_gap_census.py'],text=True))
  class RemoveCensus(ast.NodeTransformer):
   def visit_FunctionDef(self,n):return None if n.name=='_baton_start_predicates'else self.generic_visit(n)
   def visit_For(self,n):return None if any(isinstance(x,ast.Call)and isinstance(x.func,ast.Name)and x.func.id=='_baton_start_predicates'for x in ast.walk(n.iter))else self.generic_visit(n)
   def visit_Call(self,n):n.keywords=[k for k in n.keywords if k.arg!='baton_start_predicate_scope'];return self.generic_visit(n)
  self.assertEqual(dump(old),dump(RemoveCensus().visit(new)))
  for name in('transition.py','state.py','baton.py','engine_execution_completeness.py','engine_kernel_frame_risk.py','engine_retained_encoder_driver.py','engine_full_v4_event_device.py'):
   self.assertEqual(Path('gpu_engine',name).read_bytes(),subprocess.check_output(['git','show',BASE+':gpu_engine/'+name]))
  for f,key in [('engine_source_frame_witness.py','SOURCE_PINS'),('chance_consumers.py','SOURCE_PINS'),('ordinary_law.py','MAPPING_PINS'),('conditional_law.py','MAPPING_PINS')]:
   t=ast.parse(Path('gpu_engine',f).read_text());pins=ast.literal_eval(next(n.value for n in t.body if isinstance(n,ast.Assign)and any(isinstance(x,ast.Name)and x.id==key for x in n.targets)))
   for name,v in pins.items():self.assertEqual(hashlib.sha256(Path('gpu_engine',name).read_bytes()).hexdigest(),v)

if __name__=='__main__':unittest.main()
