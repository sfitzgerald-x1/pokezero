"""Same owning transaction telemetry; inert phases, no engine/model/game."""
import ast,copy,hashlib,itertools,subprocess,unittest
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import patch
import numpy as np
from gpu_engine import engine_retained_encoder_driver as d
from gpu_engine import benchmark_capture_gap_census as census
from test_gpu_engine_builtin_host_timing import BuiltinTimingTests
from test_gpu_engine_census_host_copy import HostCopyTests
BASE='f2f3b9ad725c7f3064f0420e07425ec68e95ec37'

class AtomicDiagnosticsTests(unittest.TestCase):
 def run_transaction(self,*,atomic,builtin=False,callback=False,mutate=False,repeat=False):
  original=d.advance;returned=[];calls=[]
  def actual(session,actions,**kw):
   kw['return_diagnostics']=atomic
   value=original(session,actions,**kw);returned.append(value)
   calls.append(session)
   if repeat:
    returned.append(original(session,actions,**kw))
   return value
  clock=itertools.count()
  with patch.object(d,'advance',side_effect=actual),patch.object(d.time,'perf_counter_ns',side_effect=lambda:next(clock)*1000000):
   trace,record,waits=BuiltinTimingTests().fake(builtin=builtin,callback=callback,mutate=mutate)
  return returned,trace,record,waits,calls

 def test_atomic_and_default_return_same_current_objects_guards_and_waits(self):
  for builtin,callback in ((False,False),(True,False),(False,True)):
   with self.subTest(builtin=builtin,callback=callback):
    normal,oldtrace,old,oldwaits,_=self.run_transaction(atomic=False,builtin=builtin,callback=callback)
    atomic,trace,record,waits,sessions=self.run_transaction(atomic=True,builtin=builtin,callback=callback)
    self.assertIsInstance(normal[0],d.Result)
    pair=atomic[0];self.assertIsInstance(pair,d.AdvanceWithDiagnostics)
    self.assertIs(pair.result,record['result']);self.assertIs(pair.result,sessions[0].result)
    self.assertIs(pair.diagnostics,record['diagnostics'])
    self.assertIs(pair.result.state,record['state']);self.assertIs(pair.result.state,sessions[0].state)
    self.assertEqual(trace,oldtrace);self.assertEqual(waits,oldwaits)
    for field in d.FrameDiagnostics._fields:
     a,b=getattr(pair.diagnostics,field),getattr(old['diagnostics'],field)
     if field=='host_timings':self.assertEqual(a,b)
     else:np.testing.assert_array_equal(a,b)
    self.assertFalse(record['advancing'])

 def test_new_call_is_fresh_and_failure_never_publishes_a_pair(self):
  values,trace,record,waits,_=self.run_transaction(atomic=True,builtin=True,repeat=True)
  self.assertIsNot(values[0].result,values[1].result)
  self.assertIsNot(values[0].diagnostics,values[1].diagnostics)
  self.assertIsNot(values[0].diagnostics.host_timings,values[1].diagnostics.host_timings)
  self.assertIs(record['diagnostics'],values[1].diagnostics)
  self.assertIs(record['result'],values[1].result)
  failed,trace,record,_,_=self.run_transaction(atomic=True,mutate=True)
  self.assertEqual(failed,[]);self.assertNotIn('diagnostics',record)
  self.assertFalse(record['advancing'])
  self.assertEqual(trace,['session','checked','constructed','witness','engine','observer:engine','session','checked'])

 def test_strict_option_and_actual_stale_session_remain_closed(self):
  with patch.object(d,'_session')as checked:
   for value in(0,1,None,'yes',np.bool_(True),np.array(True)):
    with self.subTest(value=value),self.assertRaisesRegex(ValueError,'return_diagnostics'):
     d.advance(None,None,return_diagnostics=value)
   checked.assert_not_called()
  # Genuine current-record identity guard executes BEFORE any phase/source call.
  original=object();result=object();prepared=object();session=d.Session(prepared,original,result)
  d._SESSIONS[session]=dict(prepared=prepared,state=original,result=result,advancing=False)
  session.state=object()
  try:
   with patch.object(d,'_checked')as checked:
    with self.assertRaisesRegex(ValueError,'original current driver session'):
     d.advance(session,np.zeros((1,2),np.int32),return_diagnostics=True)
    checked.assert_not_called()
  finally:del d._SESSIONS[session]

 def test_census_uses_exact_pair_without_checked_getter_or_extra_barrier(self):
  backend,_,_,active,scored=HostCopyTests().make_backend(8,False)
  original=backend.driver.advance;seen=[]
  def advance(*args,**kw):
   seen.append(kw);return original(*args,**kw)
  backend.driver.advance=advance
  backend.driver.latest_diagnostics=lambda _:self.fail('redundant checked getter called')
  with patch.object(census,'_row_select',side_effect=lambda mask,before,after:after),patch.object(census.time,'perf_counter_ns',side_effect=[1000000,2000000]):
   backend.advance(np.zeros((8,2),np.int32),active,scored,'literal-same-chance')
  self.assertEqual(len(seen),1);self.assertIs(seen[0]['return_diagnostics'],True);self.assertIs(seen[0]['collect_timings'],True)

 def test_mixed_census_packets_rows_and_waits_equal_frozen_source(self):
  old=ast.parse(subprocess.check_output(['git','show',BASE+':gpu_engine/benchmark_capture_gap_census.py'],text=True))
  cl=next(n for n in old.body if isinstance(n,ast.ClassDef)and n.name=='_Runtime')
  fn=next(n for n in cl.body if isinstance(n,ast.FunctionDef)and n.name=='advance');fn.name='frozen_advance'
  choose=lambda mask,before,after:after
  env=dict(vars(census),_row_select=choose)
  exec(compile(ast.fix_missing_locations(ast.Module([fn],[])),'frozen-census-advance','exec'),env)
  for b in(8,16):
   for missing in(False,True):
    with self.subTest(batch=b,missing=missing):
     before,_,oldcalls,active,scored=HostCopyTests().make_backend(b,missing)
     after,tracked,calls,_,_=HostCopyTests().make_backend(b,missing)
     for backend in(before,after):
      original=backend.jax.block_until_ready
      def ready(value,original=original,calls=oldcalls if backend is before else calls):
       calls.append(('same-original-ready',));return original(value)
      backend.jax=NS(block_until_ready=ready)
     with patch.object(census,'_row_select',side_effect=choose),patch.object(census.time,'perf_counter_ns',side_effect=[1000000,2000000]):
      expected=env['frozen_advance'](before,np.zeros((b,2),np.int32),active,scored,'literal-chance')
     after.driver.latest_diagnostics=lambda _:self.fail('census must use current returned diagnostics')
     with patch.object(census,'_row_select',side_effect=choose),patch.object(census.time,'perf_counter_ns',side_effect=[1000000,2000000]):
      got=after.advance(np.zeros((b,2),np.int32),active,scored,'literal-chance')
     self.assertEqual(after.last_packet,before.last_packet);self.assertEqual(calls,oldcalls)
     for a,z in zip(got[:2],expected[:2]):np.testing.assert_array_equal(a,z)
     self.assertEqual(got[2],expected[2])
     self.assertTrue(all(v.copies<=1 and v.slices==0 for v in tracked))

 def test_exact_reverse_patch_preserves_original_advancement_and_census(self):
  old=ast.parse(subprocess.check_output(['git','show',BASE+':gpu_engine/engine_retained_encoder_driver.py'],text=True));new=ast.parse(Path(d.__file__).read_text())
  fn=lambda tree,name:next(n for n in tree.body if isinstance(n,ast.FunctionDef)and n.name==name)
  a,b=fn(old,'advance'),copy.deepcopy(fn(new,'advance'))
  old_update=next(n for n in ast.walk(a)if isinstance(n,ast.Call)and ast.unparse(n.func)=='record.update')
  class EraseAtomic(ast.NodeTransformer):
   def visit_FunctionDef(self,n):
    if n.name=='advance':
     pairs=[(a,v)for a,v in zip(n.args.kwonlyargs,n.args.kw_defaults)if a.arg!='return_diagnostics'];n.args.kwonlyargs=[a for a,v in pairs];n.args.kw_defaults=[v for a,v in pairs]
    return self.generic_visit(n)
   def visit_If(self,n):
    if ast.unparse(n.test)=='type(return_diagnostics) is not bool':return None
    return self.generic_visit(n)
   def visit_Assign(self,n):
    if any(isinstance(t,ast.Name)and t.id=='diagnostics'for t in n.targets):return None
    return self.generic_visit(n)
   def visit_Call(self,n):
    if ast.unparse(n.func)=='record.update':return copy.deepcopy(old_update)
    return self.generic_visit(n)
   def visit_Return(self,n):
    if isinstance(n.value,ast.IfExp)and ast.unparse(n.value.test)=='return_diagnostics':return ast.Return(ast.Name('result',ast.Load()))
    return self.generic_visit(n)
  self.assertEqual(ast.dump(a,include_attributes=False),ast.dump(EraseAtomic().visit(b),include_attributes=False))
  old_body={n.name:ast.dump(n,include_attributes=False)for n in old.body if isinstance(n,(ast.FunctionDef,ast.ClassDef))and n.name!='advance'}
  new_body={n.name:ast.dump(n,include_attributes=False)for n in new.body if isinstance(n,(ast.FunctionDef,ast.ClassDef))and n.name not in ('advance','AdvanceWithDiagnostics')}
  self.assertEqual(old_body,new_body)
  old=ast.parse(subprocess.check_output(['git','show',BASE+':gpu_engine/benchmark_capture_gap_census.py'],text=True));new=ast.parse(Path(census.__file__).read_text())
  def body(t):return next(n for n in t.body if isinstance(n,ast.ClassDef)and n.name=='_Runtime')
  a,b=body(old),body(new);oldfn=next(n for n in a.body if isinstance(n,ast.FunctionDef)and n.name=='advance');newfn=next(n for n in b.body if isinstance(n,ast.FunctionDef)and n.name=='advance')
  oldif=next(n for n in oldfn.body if isinstance(n,ast.If)and ast.unparse(n.test)=='kept.any()');newif=next(n for n in newfn.body if isinstance(n,ast.If)and ast.unparse(n.test)=='kept.any()')
  self.assertEqual(ast.unparse(newif.body[1].value.func),'self.driver.advance');self.assertEqual(ast.unparse(newif.body[2]),'result, d = (advanced.result, advanced.diagnostics)');self.assertEqual(ast.unparse(newif.body[3]),'self.jax.block_until_ready(result)')
  self.assertIs(next(k.value.value for k in newif.body[1].value.keywords if k.arg=='return_diagnostics'),True)
  newif.body[1:4]=copy.deepcopy(oldif.body[1:4])
  self.assertEqual(ast.dump(old,include_attributes=False),ast.dump(new,include_attributes=False))
  for name in('transition.py','engine_kernel_frame_risk.py','engine_full_v4_event_device.py','engine_lifecycle_event_device.py','engine_execution_completeness.py','state.py'):
   self.assertEqual(Path('gpu_engine',name).read_bytes(),subprocess.check_output(['git','show',BASE+':gpu_engine/'+name]),name)
  tree=ast.parse(Path('gpu_engine/engine_source_frame_witness.py').read_text());pins=ast.literal_eval(next(n.value for n in tree.body if isinstance(n,ast.Assign)and any(isinstance(x,ast.Name)and x.id=='SOURCE_PINS'for x in n.targets)))
  self.assertEqual(len(pins),85)
  for name,want in pins.items():self.assertEqual(hashlib.sha256(Path('gpu_engine',name).read_bytes()).hexdigest(),want,name)

if __name__=='__main__':unittest.main()
