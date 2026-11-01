"""Literal synthetic monitoring records only; no compilation/model/game run."""
import ast
from contextlib import ExitStack
import importlib.metadata
from pathlib import Path
import subprocess
import threading
import unittest
from unittest.mock import patch
from gpu_engine import jax_compile_observations as c,engine_retained_encoder_driver as d

BASE='4ee452a97eee05016b51e9ee5fc13a648b1fefe4'
BACKEND='/jax/core/compile/backend_compile_duration'
class FakeMonitor:
 def __init__(self):self.spans=[];self.events=[];self.durations=[];self.other=lambda *a,**k:None;self.spans.append(self.other)
 def register_event_time_span_listener(self,x):self.spans.append(x)
 def unregister_event_time_span_listener(self,x):self.spans.remove(x)
 def register_event_listener(self,x):self.events.append(x)
 def unregister_event_listener(self,x):self.events.remove(x)
 def register_event_duration_secs_listener(self,x):self.durations.append(x)
 def unregister_event_duration_listener(self,x):self.durations.remove(x)
 def span(self,*a,**k):
  for fn in list(self.spans):fn(*a,**k)
 def event(self,*a,**k):
  for fn in list(self.events):fn(*a,**k)
 def duration(self,*a,**k):
  for fn in list(self.durations):fn(*a,**k)
class CompileObservationTests(unittest.TestCase):
 def setUp(self):self.m=FakeMonitor();self.p=patch.object(c,'_runtime',return_value=(self.m,dict(jax_version='literal-test',supported=True)));self.p.start()
 def tearDown(self):self.p.stop();self.assertIsNone(c._CURRENT.get());self.assertIsNone(c._PHASE.get())
 def test_cache_duration_nonnegative_finite_literals_only(self):
  o=c.Observer().start()
  try:
   with c.phase('driver.engine'):
    for value in(-.25,float('nan'),float('inf'),-float('inf'),0.,.25):
     self.m.duration('/jax/compilation_cache/cache_retrieval_time_sec',value)
   for record in o.records[:4]:
    self.assertFalse(record['matched']);self.assertIsNone(record['phase']);self.assertIsNone(record['reported_duration_secs']);self.assertEqual(record['reason'],'INVALID_LITERAL_DURATION')
   for record,value in zip(o.records[4:],(0.,.25)):
    self.assertTrue(record['matched']);self.assertEqual(record['reported_duration_secs'],value)
  finally:o.close()
 def test_exact_spans_cache_notifications_unknown_thread_and_fixed_window(self):
  o=c.Observer().start();c.sample_step(8)
  with patch.object(c.time,'time',return_value=10.):
   with c.phase('driver.engine'):
    self.m.duration(BACKEND,2.,fun_name='jit(frame)');self.m.span(BACKEND,11.,13.,fun_name='jit(frame)')
    self.m.event('/jax/compilation_cache/cache_hits');self.m.duration('/jax/compilation_cache/cache_retrieval_time_sec',.2)
    self.m.span(BACKEND,9.,11.,fun_name='previous-call')
    t=threading.Thread(target=lambda:self.m.span(BACKEND,11.,12.,fun_name='other-thread'));t.start();t.join()
  c.sample_step(31)
  self.m.span(BACKEND,15.,16.,fun_name='unscoped')
  c.sample_step(None);o.close();s=o.snapshot(samples=[dict(step=i)for i in range(32)],warmup=8,steps=24)
  self.assertEqual((s['event_count'],s['stored_event_count'],s['matched_event_count'],s['unmatched_event_count']),(6,6,3,3));self.assertEqual(s['duplicate_duration_notifications'],1)
  self.assertEqual(s['records'][0]['duration_secs'],2.);self.assertEqual(s['records'][0]['service'],'compile_or_get_cached_service');self.assertEqual(s['records'][0]['phase'],'driver.engine')
  self.assertIsNone(s['pure_compiler_only_ms']);self.assertEqual(s['fixed_declared_warm_window']['recorded_steps'],list(range(8,32)));self.assertEqual(s['fixed_declared_warm_window']['event_count'],5);self.assertEqual(s['fixed_declared_warm_window']['unassigned_step_events'],1)
  self.assertEqual(self.m.spans,[self.m.other]);self.assertEqual(self.m.events,[]);self.assertEqual(self.m.durations,[])
 def test_missing_events_unsupported_runtime_overflow_and_exception_expiry(self):
  o=c.Observer().start();o.close();s=o.snapshot(samples=[],warmup=8,steps=24);self.assertEqual(s['event_count'],0);self.assertIn('COMPILER_UNKNOWN',s['status']);self.assertIsNone(s['pure_compiler_only_ms'])
  with patch.object(c,'_runtime',return_value=(self.m,dict(supported=False,jax_version='unknown'))):
   o=c.Observer().start();o.close();s=o.snapshot(samples=[],warmup=8,steps=24);self.assertIsNone(s['event_count']);self.assertIsNone(s['matched_event_count']);self.assertEqual(self.m.spans,[self.m.other])
  o=c.Observer(capacity=1).start()
  try:
   with patch.object(c.time,'time',return_value=1.),c.phase('shadow.encode_p1'):
    self.m.span(BACKEND,2.,3.,fun_name='encoder');self.m.span(BACKEND,3.,4.,fun_name='encoder');self.m.span(BACKEND,float('nan'),4.)
    raise ValueError('original phase failure')
  except ValueError:pass
  finally:o.close()
  s=o.snapshot(samples=[],warmup=8,steps=24);self.assertEqual((s['event_count'],s['stored_event_count'],s['dropped_event_count']),(3,1,2));self.assertIn('UNKNOWN',s['status']);self.assertEqual((s['matched_event_count'],s['unmatched_event_count']),(2,1));self.assertEqual(self.m.spans,[self.m.other])
  fresh=c.Observer().start();self.assertEqual(fresh.records,[]);fresh.close()
 def test_driver_same_call_arguments_results_barriers_and_callback_outside_scope(self):
  old=ast.parse(subprocess.check_output(['git','show',BASE+':gpu_engine/engine_retained_encoder_driver.py'],text=True));fn=next(n for n in old.body if isinstance(n,ast.FunctionDef)and n.name=='_timed');ns=dict(vars(d));exec(compile(ast.Module(body=[fn],type_ignores=[]),'original-timed','exec'),ns)
  for observed in (False,True):
   outputs=[]
   for call in (ns['_timed'],d._timed):
    trace=[];arg=object();out=object();o=c.Observer().start()
    def function(value):trace.append(('call',value is arg));return out
    def callback(*args):trace.append(('callback',c._PHASE.get()is None))
    with patch.object(d.jax,'block_until_ready',side_effect=lambda x:trace.append(('ready',x is out))),patch.object(d.time,'perf_counter_ns',side_effect=range(1,20)),patch.object(c.time,'time',return_value=0.):
     got=call(callback if observed else None,'engine',function,arg)
    o.close();self.assertIs(got,out);outputs.append(trace)
   self.assertEqual(outputs[0],outputs[1])
 def test_original_programs_identical_after_observer_only_erasure(self):
  class Erase(ast.NodeTransformer):
   def visit_ImportFrom(self,n):return None if any(v.name=='jax_compile_observations'for v in n.names)else n
   def visit_With(self,n):
    if len(n.items)==1 and isinstance(n.items[0].context_expr,ast.Call)and ast.unparse(n.items[0].context_expr.func)=='compile_observations.phase':return [self.visit(v)for v in n.body]
    return self.generic_visit(n)
   def visit_Expr(self,n):return None if isinstance(n.value,ast.Call)and ast.unparse(n.value.func)=='compile_observations.sample_step'else self.generic_visit(n)
   def visit_Assign(self,n):return None if any(isinstance(t,ast.Name)and t.id=='compile_observer'for t in n.targets)else self.generic_visit(n)
   def visit_If(self,n):
    if ast.unparse(n.test)in('type(collect_compile_events) is not bool','compile_observer is not None'):return None
    return self.generic_visit(n)
   def visit_FunctionDef(self,n):
    if n.name=='measure':
     pairs=[(a,v)for a,v in zip(n.args.kwonlyargs,n.args.kw_defaults)if a.arg!='collect_compile_events'];n.args.kwonlyargs=[a for a,v in pairs];n.args.kw_defaults=[v for a,v in pairs]
    return self.generic_visit(n)
   def visit_For(self,n):
    if isinstance(n.target,ast.Tuple)and ast.unparse(n.target)=='(owner, observer)'and ast.unparse(n.iter)=='enumerate(self.session.state.observers)':n.target=n.target.elts[1];n.iter=n.iter.args[0]
    return self.generic_visit(n)
  for file in('engine_retained_encoder_driver.py','benchmark_capture_gap_census.py'):
   old=ast.parse(subprocess.check_output(['git','show',BASE+':gpu_engine/'+file],text=True));new=Erase().visit(ast.parse(Path('gpu_engine',file).read_text()));self.assertEqual(ast.dump(old,include_attributes=False),ast.dump(new,include_attributes=False),file)
 def test_actual_installed_public_api_and_event_source_contract_without_backend(self):
  self.p.stop()
  try:
   module,binding=c._runtime();self.assertEqual(binding['jax_version'],'0.12.0');self.assertTrue(binding['supported']);self.assertTrue(callable(module.register_event_time_span_listener))
  finally:self.p.start()

if __name__=='__main__':unittest.main()
