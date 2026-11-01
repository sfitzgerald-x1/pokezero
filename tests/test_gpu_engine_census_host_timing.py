"""Thin host data transport/counters only: no engine/model/game invocation."""
import ast
import copy
import json
from pathlib import Path
from types import SimpleNamespace as NS
import subprocess
import unittest
from unittest import mock
import numpy as np
from gpu_engine import benchmark_capture_gap_census as c

class HostTimingTests(unittest.TestCase):
 def test_original_driver_host_spans_copy_without_calls_or_mutation(self):
  spans={'initial_guards_ms':4.,'advance_wall_ms':7.,'device_call.engine_ms':1.5}
  before=dict(spans);value=c._driver_host_overlap_ms(NS(host_timings=spans))
  self.assertEqual(value,{'driver_host_overlap.'+k:v for k,v in spans.items()})
  self.assertEqual(spans,before);self.assertIsNot(value,spans)
  self.assertEqual(c._driver_host_overlap_ms(NS()),{});self.assertEqual(c._driver_host_overlap_ms(NS(host_timings=None)),{})
  encoded=json.dumps(value,allow_nan=False);self.assertIn('advance_wall_ms',encoded)
  # The already fetched SAME diagnostics supplies spans, with no second guard.
  tree=ast.parse(Path(c.__file__).read_text());cls=next(n for n in tree.body if isinstance(n,ast.ClassDef)and n.name=='_Runtime')
  advance=next(n for n in cls.body if isinstance(n,ast.FunctionDef)and n.name=='advance')
  calls=[n for n in ast.walk(advance)if isinstance(n,ast.Call)and isinstance(n.func,ast.Attribute)and n.func.attr=='latest_diagnostics']
  self.assertEqual(len(calls),1)
 def test_source_union_across_contexts_keeps_engine_and_timing_separate(self):
  ledger=c.Ledger(c.Plan(),np.arange(1,2049,dtype=np.int32))
  for domain,row in [('capture',0),('capture-context',0),('capture-context',1)]:ledger._count('census',domain,'site:5',row,0,{})
  for row in(0,2):ledger._count('census','engine','site:5',row,0,{})
  for domain in('capture','capture-context'):ledger._count('timing-only',domain,'site:5',600,0,{})
  actual=ledger.summary();table={(r['role'],r['source'],r['kind']):r for r in actual['unsupported_by_source']}
  self.assertEqual((table['census','capture-gap','site:5']['occurrences'],table['census','capture-gap','site:5']['games_affected']),(3,2))
  self.assertEqual((table['census','engine','site:5']['occurrences'],table['census','engine','site:5']['games_affected']),(2,2))
  self.assertEqual((table['timing-only','capture-gap','site:5']['occurrences'],table['timing-only','capture-gap','site:5']['timing_rows_affected']),(2,1))
  self.assertEqual(len(actual['unsupported']),5)
 def test_run_original_boundaries_actions_chance_sync_clock_and_warm_flags_identical(self):
  source=subprocess.check_output(['git','show','af95b12a:gpu_engine/benchmark_capture_gap_census.py'],text=True)
  node=next(n for n in ast.parse(source).body if isinstance(n,ast.FunctionDef)and n.name=='_run');node.name='_original_run'
  env=dict(vars(c));exec(compile(ast.fix_missing_locations(ast.Module(body=[node],type_ignores=[])),'original-run','exec'),env)
  class Clock:
   def __init__(self):self.calls=0
   def __call__(self):self.calls+=1;return self.calls*.001
  class NSClock:
   def __init__(self):self.calls=0
   def __call__(self):self.calls+=1;return self.calls*1000000
  class Fake:
   def __init__(self):
    self.engine=NS(done=np.zeros(2048,bool),turn=np.zeros(2048,np.int32));self.calls=[];self.last_packet=[{}for _ in range(2048)];self.n=0
    self.jax=NS(block_until_ready=self.ready)
   def ready(self,value):self.calls.append(('existing-ready',));return value
   def cache_snapshot(self):return {'frame':1}
   def policy(self,active,ledger,step,seed):
    self.calls.append(('policy',step,seed));bad=np.zeros(2048,bool);bad[0]=True;ledger.mark_capture(bad,step=step,packets=lambda row:{'fake':True})
    return np.zeros((2048,2),np.int32),dict(bf16_physical_owner_rows=4096,admitted_policy_owner_rows=int((active&ledger.capture_good).sum())*2)
   def mask(self,state):self.calls.append(('mask',));return np.ones((2048,2,9),bool)
   def requested(self):self.calls.append(('requested',));return np.ones((2048,2),bool)
   def advance(self,actions,active,scored,chance):
    self.calls.append(('advance',actions.tolist(),active.tolist(),scored.tolist(),chance));self.n+=1;self.engine.turn+=active
    if self.n==10:self.engine.done|=active
    self.last_packet=[{}for _ in range(2048)];return active.copy(),np.zeros(2048,bool),dict(original_request_rows=int(scored.sum()),driver_engine=.125)
   def capture_failure(self,active):self.calls.append(('capture_failure',active.tolist()));return np.zeros(2048,bool),np.zeros((2048,2),np.int64)
   def terminal(self,row):return {'done':True,'winner':0}
  ids=np.arange(101,2149,dtype=np.int32);before,after=Fake(),Fake();oldclock,newclock=Clock(),Clock();ns=NSClock()
  with mock.patch('gpu_engine.transition.sample_chance_inputs',return_value='literal-same-once')as sampler,mock.patch.object(c.time,'time',return_value=123.):
   original=env['_original_run'](before,c.Plan(),ids,sampling_seed=31,clock=oldclock);self.assertEqual(sampler.call_count,10)
   sampler.reset_mock()
   with mock.patch.object(c.time,'perf_counter_ns',side_effect=ns):actual=c._run(after,c.Plan(),ids,sampling_seed=31,clock=newclock)
   self.assertEqual(sampler.call_count,10)
  self.assertEqual(before.calls,after.calls);self.assertEqual(oldclock.calls,newclock.calls);self.assertEqual(ns.calls,100)
  self.assertEqual([s['warm_candidate']for s in original['samples']],[s['warm_candidate']for s in actual['samples']])
  self.assertEqual(actual['samples'][0]['phases_ms']['census_host_overlap.chance_sampling_dispatch_wall_ms'],1.)
  stripped=copy.deepcopy(actual)
  for name in('host_timing_scope','unsupported_by_source_scope'):stripped.pop(name)
  for i,sample in enumerate(stripped['samples']):
   self.assertIs(sample.pop('policy_work_skipped'),False);self.assertIsNone(sample.pop('policy_work_skip_reason'))
   self.assertEqual(sample.pop('policy_input_capture_valid_rows'),2048 if i==0 else 2047)
   sample['phases_ms']={k:v for k,v in sample['phases_ms'].items()if not k.startswith('census_host_overlap.')}
  self.assertEqual(stripped,original)
  json.dumps(actual,allow_nan=False)
  for path in('engine_retained_encoder_driver.py','transition.py'):
   self.assertEqual(Path('gpu_engine',path).read_bytes(),subprocess.check_output(['git','show','a8034bc27e083b156f525d69defa673c6908a25b:gpu_engine/'+path]))
  # Both physical sources remain exact at the reviewed current composition;
  # historical ac13 preceded the separately reviewed built-in timing/inputs.

if __name__=='__main__':unittest.main()
