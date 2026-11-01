"""Built-in data collector: fake phases only, no model/engine/game execution."""
import ast
from collections import namedtuple
from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace as NS
import subprocess
import unittest
from unittest.mock import patch
import numpy as np
from gpu_engine import engine_retained_encoder_driver as d

BASE='c028e11b20b01f67a6110ac1b3f3e4c3cc2c3efb'
PHASES=('engine','requests','public_pair','encode_p1','admission_p1','encode_p2','admission_p2')

class BuiltinTimingTests(unittest.TestCase):
 def fake(self,*,builtin=False,callback=False,mutate=False):
  Core=namedtuple('Core','engine frame_number admitted requests orders unsafe originals constructors weather_profile execution risks cursor')
  State=namedtuple('State','core observers')
  yes=np.ones(1,bool);engine=NS(done=~yes);core=Core(engine,np.zeros(1,np.int32),yes,(),(),(),(),(),None,None,None,None)
  observers=tuple(NS(retained='public-original',retained_healthy=yes,projection=NS(causal=NS(admitted=yes)))for _ in range(2))
  state=State(core,observers);result=d.Result(state,());session=d.Session(NS(),state,result)
  record=dict(advancing=False,constructed=object(),state=state,result=result);trace=[];poison=[False];waits=[]
  r=dict(execution_compiled=object(),retain_speed=None,public_mode='retained-full',direct_hit_context=None)
  def checked(_):
   trace.append('checked')
   if poison[0]:raise ValueError('actual external static mutation')
   return r
  def frame(*args):
   trace.append('engine');return NS(state=engine,cursor=NS(admitted=yes),callbacks=NS(refusal=np.zeros(1,np.int32),execution=None),weather_profile=None,invalid_actions=np.zeros(1,bool),placeholder_moves=np.zeros(1,np.int32))
  r['frame']=frame
  def consume(r,state,frame,executed,observe,previous,host_timings):
   for name in PHASES[2:]:d._timed(observe,name,lambda:None,_host_timings=host_timings)
   return d.Result(state,())
  def observer(name,ms):
   trace.append('observer:'+name)
   if mutate:poison[0]=True
  with ExitStack()as stack:
   stack.enter_context(patch.object(d,'_session',side_effect=lambda _:trace.append('session')or record))
   stack.enter_context(patch.object(d,'_checked',side_effect=checked))
   stack.enter_context(patch.object(d.source,'_constructed',side_effect=lambda _:trace.append('constructed')))
   stack.enter_context(patch.object(d,'_checked_witness',side_effect=lambda *a,**k:trace.append('witness')))
   stack.enter_context(patch.object(d,'_leech_history',return_value=None))
   stack.enter_context(patch.object(d,'_core_rows',side_effect=lambda active,before,after:after))
   stack.enter_context(patch.object(d,'_consume',side_effect=consume))
   stack.enter_context(patch.object(d.jax,'block_until_ready',side_effect=lambda value:waits.append('ready')))
   if mutate:
    with self.assertRaisesRegex(ValueError,'actual external static mutation'):
     d.advance(session,np.zeros((1,2),np.int32),active_rows=yes,chance_inputs='literal-no-sampling',timing_callback=observer)
    self.assertIs(session.state,state);self.assertFalse(record['advancing'])
   else:
    d.advance(session,np.zeros((1,2),np.int32),active_rows=yes,chance_inputs='literal-no-sampling',timing_callback=observer if callback else None,collect_timings=builtin)
  return trace,record,waits
 def test_builtin_matches_nonobserver_guards_and_original_synchronized_phases(self):
  trace,record,waits=self.fake(builtin=True)
  plain,untimed,no_waits=self.fake()
  external,observed,external_waits=self.fake(callback=True)
  self.assertEqual(trace,plain);self.assertEqual(trace,['session','checked','constructed','witness','engine','checked','session'])
  self.assertEqual(waits,external_waits);self.assertEqual(len(waits),2*len(PHASES));self.assertEqual(no_waits,[])
  self.assertIsNone(untimed['diagnostics'].host_timings)
  times=record['diagnostics'].host_timings
  for name in PHASES:self.assertIn('phase.'+name+'_ms',times);self.assertGreaterEqual(times['phase.'+name+'_ms'],0)
  self.assertNotIn('post_observer_guards_ms',times);self.assertNotIn('timing_observer_ms',times)
  self.assertIn('initial_guards_ms',times);self.assertIn('extra_guards_ms',times);self.assertIn('final_session_guard_ms',times)
  self.assertEqual(external.count('checked'),len(PHASES)+2)
  self.assertEqual(record['diagnostics'].executed_rows.tolist(),observed['diagnostics'].executed_rows.tolist())
  self.assertEqual(record['diagnostics'].active_rows.tolist(),observed['diagnostics'].active_rows.tolist())
 def test_census_opts_in_without_callback_and_copies_original_phase_names(self):
  from gpu_engine import benchmark_capture_gap_census as c
  import test_gpu_engine_census_host_copy as fixtures
  backend,_,_,active,scored=fixtures.HostCopyTests().make_backend(8,False)
  original=backend.driver.advance;observed={}
  def advance(*args,**kwargs):observed.update(kwargs);return original(*args,**kwargs)
  backend.driver.advance=advance
  data=backend.driver.latest_diagnostics(backend.session)
  data.host_timings.update({'phase.engine_ms':5.,'phase.public_pair_ms':2.})
  with patch.object(c,'_row_select',side_effect=lambda mask,before,after:after),patch.object(c.time,'perf_counter_ns',side_effect=[1000000,2000000]):
   _,_,phases=backend.advance(np.zeros((8,2),np.int32),active,scored,'literal-chance')
  self.assertIs(observed['collect_timings'],True);self.assertNotIn('timing_callback',observed)
  self.assertEqual(phases['driver_engine'],5.);self.assertEqual(phases['driver_public_pair'],2.)
  self.assertEqual(phases['driver_host_overlap.phase.engine_ms'],5.)
 def test_external_mutation_and_exclusive_strict_bool_remain_closed(self):
  trace,_,_=self.fake(mutate=True)
  self.assertEqual(trace,['session','checked','constructed','witness','engine','observer:engine','session','checked'])
  with patch.object(d,'_session')as guard:
   for value in(1,0,'yes',None,np.bool_(True)):
    with self.subTest(value=value),self.assertRaises(ValueError):d.advance(None,None,collect_timings=value)
   with self.assertRaises(ValueError):d.advance(None,None,collect_timings=True,timing_callback=lambda *args:None)
   guard.assert_not_called()
 def test_original_guards_callback_body_and_timed_source_exact(self):
  old=ast.parse(subprocess.check_output(['git','show',BASE+':gpu_engine/engine_retained_encoder_driver.py'],text=True));new=ast.parse(Path(d.__file__).read_text())
  fn=lambda tree,name:next(n for n in tree.body if isinstance(n,ast.FunctionDef)and n.name==name)
  for name in('_timed','_host_span','_checked','_checked_witness','_consume','_consume_accounted','construct','latest_diagnostics'):
   self.assertEqual(ast.dump(fn(old,name),include_attributes=False),ast.dump(fn(new,name),include_attributes=False),name)
  a,b=fn(old,'advance'),fn(new,'advance')
  callback=lambda n:next(v for v in ast.walk(n)if isinstance(v,ast.FunctionDef)and v.name=='checked_timing')
  self.assertEqual(ast.dump(callback(a),include_attributes=False),ast.dump(callback(b),include_attributes=False))
  # Erase ONLY this opt-in API/data collector to prove original phase/carry
  # assignments and normal checks remain the identical program.
  class EraseBuiltin(ast.NodeTransformer):
   def visit_FunctionDef(self,node):
    if node.name=='built_in_timing':return None
    node=self.generic_visit(node)
    if node.name=='advance':
     keep=[(a,v)for a,v in zip(node.args.kwonlyargs,node.args.kw_defaults)if a.arg!='collect_timings'];node.args.kwonlyargs=[a for a,v in keep];node.args.kw_defaults=[v for a,v in keep]
     node.body[0]=a.body[0]
    return node
   def visit_If(self,node):
    if ast.unparse(node.test)in('type(collect_timings) is not bool','collect_timings and timing_callback is not None'):return None
    return self.generic_visit(node)
   def visit_Assign(self,node):
    if any(isinstance(t,ast.Name)and t.id=='host_timings'for t in node.targets):node.value=fn(old,'advance').body[1].value
    if any(isinstance(t,ast.Name)and t.id=='observe'for t in node.targets)and isinstance(node.value,ast.IfExp)and ast.unparse(node.value.test)=='collect_timings':node.value=node.value.orelse
    return self.generic_visit(node)
  self.assertEqual(ast.dump(a,include_attributes=False),ast.dump(EraseBuiltin().visit(b),include_attributes=False))

if __name__=='__main__':unittest.main()
