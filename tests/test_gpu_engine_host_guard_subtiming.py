"""Literal host call/clock controls only; no engine, device kernel or model."""
import ast
from contextlib import ExitStack
import copy
import importlib
from itertools import count
from pathlib import Path
import subprocess
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

from gpu_engine import engine_retained_encoder_driver as d

BASE='fd3256d2305a0770764e737640fc80157156cb29'
CALLS=(
 ('own','engine_own_facts','_checked','own_compiled'),
 ('request_source','engine_request_source','_checked','request_compiled'),
 ('public',None,None,'public'),
 ('accounting','engine_execution_completeness','_checked','execution_compiled'),
 ('no_output','engine_leftovers_no_output','check_compiled','no_output_compiled'),
 ('risk','engine_kernel_frame_risk','check_compiled','kernel_risk_compiled'),
 ('calm_mind','engine_calm_mind_callbacks','check_compiled','calm_mind_compiled'),
 ('swords_dance','engine_swords_dance_callbacks','check_compiled','swords_dance_compiled'),
 ('recoil','engine_recoil_callbacks','check_compiled','recoil_compiled'),
 ('toxic','engine_toxic_callbacks','check_compiled','toxic_compiled'),
 ('substitute','engine_substitute_callbacks','check_compiled','substitute_compiled'),
 ('direct_hit','engine_direct_hit_controls','check_compiled','direct_hit_compiled'),
 ('spikes','engine_spikes_callbacks','check_compiled','spikes_compiled'),
 ('thunder_wave','engine_thunder_wave_callbacks','check_compiled','thunder_wave_compiled'),
 ('sleep_powder','engine_sleep_powder_callbacks','check_compiled','sleep_powder_compiled'),
 ('speed_profile','engine_weather_speed_profile','check_profile','speed_profile'),
 ('leftovers','engine_leftovers_callbacks','check_compiled','leftovers_compiled'),
 ('status_residual','engine_status_residual_callbacks','check_compiled','status_residual_compiled'),
)


class HostGuardSubtimingTests(unittest.TestCase):
 def setup_fake(self,stack,*,optional=True,fail=None):
  binding=object();prepared=d.Prepared(binding);runtime={'binding':binding};trace=[]
  for i,(name,module,method,key)in enumerate(CALLS):
   value=object()if optional or i<3 else None;runtime[key]=value
   def check(actual,name=name,value=value):
    self.assertIs(actual,value);trace.append(name)
    if fail is not None and name==fail[0]:raise fail[1]
    return actual
   if module is None:runtime['check_public']=check
   else:stack.enter_context(patch.object(importlib.import_module('gpu_engine.'+module),method,side_effect=check))
  stack.enter_context(patch.dict(d._PREPARED,{prepared:runtime}))
  return prepared,runtime,trace

 def test_original_calls_order_arguments_and_current_optional_omissions(self):
  with ExitStack()as stack:
   prepared,runtime,trace=self.setup_fake(stack)
   clock=stack.enter_context(patch.object(d.time,'perf_counter_ns',side_effect=count(0,1000000)))
   self.assertIs(d._checked(prepared),runtime);self.assertEqual(trace,[x[0]for x in CALLS]);clock.assert_not_called()
   trace.clear();timings={}
   with d._host_span(timings,'initial_guards_ms'):self.assertIs(d._checked(prepared),runtime)
   self.assertEqual(trace,[x[0]for x in CALLS])
   for name,*_ in CALLS:
    prefix='guard_call.initial_guards_ms.'+name
    self.assertEqual(timings[prefix+'_calls'],1);self.assertEqual(timings[prefix+'_ms'],1.)
   self.assertEqual(len(timings),2*len(CALLS)+1);self.assertIsNone(d._HOST_GUARD_TIMING.get())
  with ExitStack()as stack:
   prepared,runtime,trace=self.setup_fake(stack,optional=False);timings={}
   with d._host_span(timings,'extra_guards_ms'):d._checked(prepared)
   self.assertEqual(trace,['own','request_source','public'])
   self.assertNotIn('guard_call.extra_guards_ms.risk_calls',timings)
   self.assertNotIn('guard_call.extra_guards_ms.risk_ms',timings)

 def test_witness_nested_overlap_counts_and_original_current_arguments(self):
  with ExitStack()as stack:
   prepared,runtime,trace=self.setup_fake(stack);compiled=object();runtime['frame_witness_compiled']=compiled
   witness=importlib.import_module('gpu_engine.engine_source_frame_witness');constructed=object();session=object()
   def compiled_call(actual):
    self.assertIs(actual,compiled);trace.append('witness_compiled');return d._checked(prepared)
   def origin_call(actual,**kwargs):
    self.assertIs(actual,compiled);self.assertEqual(tuple(kwargs),('constructed','session'));self.assertIs(kwargs['constructed'],constructed);self.assertIs(kwargs['session'],session)
    trace.append('witness_host_origin');return d._checked(prepared)
   stack.enter_context(patch.object(witness,'check_compiled',side_effect=compiled_call));stack.enter_context(patch.object(witness,'check_host_origin',side_effect=origin_call))
   stack.enter_context(patch.object(d.time,'perf_counter_ns',side_effect=count(0,1000000)))
   timings={}
   with d._host_span(timings,'extra_guards_ms'):
    self.assertIsNone(d._checked_witness(runtime));self.assertIsNone(d._checked_witness(runtime,constructed=constructed,session=session))
   expected=[x[0]for x in CALLS];self.assertEqual(trace,['witness_compiled']+expected+['witness_host_origin']+expected)
   for name,*_ in CALLS:self.assertEqual(timings['guard_call.extra_guards_ms.'+name+'_calls'],2)
   for name in('witness_compiled','witness_host_origin'):
    self.assertEqual(timings['guard_call.extra_guards_ms.'+name+'_calls'],1)
    self.assertGreater(timings['guard_call.extra_guards_ms.'+name+'_ms'],timings['guard_call.extra_guards_ms.own_ms'])
   self.assertIsNone(d._HOST_GUARD_TIMING.get())
   runtime['frame_witness_compiled']=None;before=dict(timings)
   with d._host_span(None,'extra_guards_ms'):self.assertIsNone(d._checked_witness(runtime))
   self.assertEqual(timings,before)

 def test_exception_identity_reset_and_unexecuted_are_not_zero_measurements(self):
  error=ValueError('literal original source refusal');timings={}
  with ExitStack()as stack:
   prepared,runtime,trace=self.setup_fake(stack,fail=('request_source',error))
   stack.enter_context(patch.object(d.time,'perf_counter_ns',side_effect=count(0,1000000)))
   try:
    with d._host_span(timings,'initial_guards_ms'):d._checked(prepared)
   except ValueError as actual:self.assertIs(actual,error)
   else:self.fail('original exception swallowed')
   self.assertEqual(trace,['own','request_source']);self.assertEqual(timings['guard_call.initial_guards_ms.request_source_calls'],1)
   self.assertNotIn('guard_call.initial_guards_ms.public_ms',timings);self.assertIsNone(d._HOST_GUARD_TIMING.get())
   before=dict(timings)
   with self.assertRaises(ValueError):d._checked(prepared)
   self.assertEqual(timings,before)
   with d._host_span(timings,'final_session_guard_ms'):
    with d._guard_span('not_a_guard'):pass
   self.assertFalse(any('not_a_guard'in k for k in timings));self.assertIsNone(d._HOST_GUARD_TIMING.get())
   # Nested timing scope restores the exact prior sink, including on error.
   with d._host_span(timings,'extra_guards_ms'):
    outer=d._HOST_GUARD_TIMING.get()
    with self.assertRaises(ValueError):
     with d._host_span({},'initial_guards_ms'):raise error
    self.assertIs(d._HOST_GUARD_TIMING.get(),outer)
   self.assertIsNone(d._HOST_GUARD_TIMING.get())

 def test_reverse_source_exact_and_existing_census_transports_counts_unchanged(self):
  before=ast.parse(subprocess.check_output(['git','show',BASE+':gpu_engine/engine_retained_encoder_driver.py'],text=True));after=ast.parse(Path(d.__file__).read_text())
  class Erase(ast.NodeTransformer):
   def visit_ImportFrom(self,n):return None if n.module=='contextvars'else n
   def visit_Assign(self,n):return None if any(isinstance(t,ast.Name)and t.id in('_HOST_GUARD_TIMING','token')for t in n.targets)else self.generic_visit(n)
   def visit_FunctionDef(self,n):return None if n.name=='_guard_span'else self.generic_visit(n)
   def visit_With(self,n):
    n=self.generic_visit(n)
    return n.body if len(n.items)==1 and isinstance(n.items[0].context_expr,ast.Call)and isinstance(n.items[0].context_expr.func,ast.Name)and n.items[0].context_expr.func.id=='_guard_span'else n
   def visit_Try(self,n):
    n=self.generic_visit(n)
    if len(n.finalbody)==1 and isinstance(n.finalbody[0],ast.Expr)and ast.unparse(n.finalbody[0].value)=='_HOST_GUARD_TIMING.reset(token)':return n.body
    return n
  self.assertEqual(ast.dump(before,include_attributes=False),ast.dump(Erase().visit(after),include_attributes=False))
  path=Path(d.__file__).with_name('benchmark_capture_gap_census.py');original=subprocess.check_output(['git','show',BASE+':gpu_engine/benchmark_capture_gap_census.py']);self.assertEqual(path.read_bytes(),original)
  node=next(n for n in ast.parse(path.read_text()).body if isinstance(n,ast.FunctionDef)and n.name=='_driver_host_overlap_ms');space={};exec(compile(ast.Module([copy.deepcopy(node)],[]),'<existing literal timing ferry>','exec'),space)
  measured={'guard_call.extra_guards_ms.own_ms':2.5,'guard_call.extra_guards_ms.own_calls':3}
  result=space['_driver_host_overlap_ms'](NS(host_timings=measured));self.assertEqual(result,{'driver_host_overlap.'+k:v for k,v in measured.items()});self.assertNotIn('driver_host_overlap.guard_call.extra_guards_ms.risk_ms',result)


if __name__=='__main__':unittest.main()
