"""Pure readout for predeclared A4 twelve-worker saved-root cost geometry.

No model, simulator, opponent, native search, remote access, or process execution.
FP request windows are labelled elapsed windows, never claimed as CPU busy time.
"""
from __future__ import annotations
import math
import re
import hashlib
import json
import os
import time
import multiprocessing
from contextlib import contextmanager

ARMS=('raw_champion','champion_head_search','refit_head_search')

def require(ok,message):
 if not ok:raise ValueError(message)

def integer(value):return type(value) is int

def finite_nonnegative(value):return type(value) in (int,float) and math.isfinite(value) and value>=0

def nearest_rank_95(values):
 require(bool(values) and all(finite_nonnegative(v) for v in values),'finite nonempty timing observations required')
 return sorted(values)[math.ceil(.95*len(values))-1]

def overlap_seconds(begin,end,intervals):
 """Union coverage and integrated concurrency for available elapsed FP windows."""
 clipped=[(max(begin,a),min(end,b)) for a,b in intervals if max(begin,a)<min(end,b)]
 weighted=sum(b-a for a,b in clipped)
 covered=0;left=right=None
 for a,b in sorted(clipped):
  if left is None:left,right=a,b
  elif a>right:covered+=right-left;left,right=a,b
  else:right=max(right,b)
 if left is not None:covered+=right-left
 return covered/1e9,weighted/1e9

class _NativeCallTiming:
 """Transparent timer around the existing native callable, inherited by fork."""
 def __init__(self, target, emit, clock, cpu_clock, pid):
  self.target=target;self.emit=emit;self.clock=clock;self.cpu_clock=cpu_clock;self.pid=pid;self.calls=0
 def __call__(self,state,search_time_ms,*,threads):
  self.calls+=1;identity=f'{self.pid()}:{self.calls}'
  begin=self.clock();cpu=self.cpu_clock()
  try:result=self.target(state,search_time_ms,threads=threads)
  except BaseException:
   self.emit('A4_NATIVE '+json.dumps({'id':identity,'complete':False,'started_ns':begin,'finished_ns':self.clock(),'cpu_ns':self.cpu_clock()-cpu,'budget_ms':search_time_ms,'threads':threads},sort_keys=True,separators=(',',':')))
   raise
  self.emit('A4_NATIVE '+json.dumps({'id':identity,'complete':True,'started_ns':begin,'finished_ns':self.clock(),'cpu_ns':self.cpu_clock()-cpu,'budget_ms':search_time_ms,'threads':threads,'visits':result.total_visits},sort_keys=True,separators=(',',':')))
  return result

@contextmanager
def native_call_timing(module, emit):
 """Instrument native calls without replacing production FP dispatch/search.

 The actual unmodified ProcessPoolExecutor default must be fork; do not change
 start methods or install a no-fork optimization to make this probe work. The
 production executor must finish inside this context before the hook restores.
 An owner must check one parallel process/thread and bind the exact source.
 """
 require(multiprocessing.get_context().get_start_method()=='fork','actual production pool must inherit the hook through its unchanged fork default')
 original=module.monte_carlo_tree_search
 require(not isinstance(original,_NativeCallTiming),'nested native instrumentation refused')
 module.monte_carlo_tree_search=_NativeCallTiming(original,emit,time.monotonic_ns,time.process_time_ns,os.getpid)
 try:yield
 finally:module.monte_carlo_tree_search=original

def require_serial_native_calls(calls):
 """Observed unmodified ProcessPoolExecutor(max_workers=1) geometry."""
 require(bool(calls) and all(type(c.get('id')) is str and re.fullmatch(r'[0-9]+:[0-9]+',c['id']) for c in calls),'native child PID/call identities required')
 ids=[c['id'].split(':') for c in calls]
 require(len({pid for pid,_ in ids})==1 and [int(n) for _,n in ids]==list(range(1,len(calls)+1)),'one actual child and ordered native calls required')
 require(all(b['started_ns']>=a['finished_ns'] for a,b in zip(calls,calls[1:])),'single production child cannot overlap native samples')

def native_search_log_evidence(text, sample_count, sample_budget_ms):
 """Parse ONE decision-scoped pinned-production log, never execute FP.

 Iterations lines attest completed result futures. Timed native records attest
 actual per-call CPU and elapsed intervals separately from request windows.
 Zero-visit/CPU calls are retained, never manufactured into busy-time evidence.
 """
 require(type(text) is str and len(text.encode())<=16*1024*1024 and integer(sample_count) and sample_count>0 and integer(sample_budget_ms) and sample_budget_ms>0,'bounded log and predeclared native sampling geometry required')
 sampling=re.findall(r'(?m)^.*?Sampling (\d+) battles at (\d+)ms each\s*$',text)
 iterations=re.findall(r'(?m)^.*?Iterations (\d+): (\d+)\s*$',text)
 choices=re.findall(r'(?m)^.*?Choice: (\S.*?)\s*$',text)
 require(sampling==[(str(sample_count),str(sample_budget_ms))],'missing/duplicate/drifted native sample declaration')
 require(len(iterations)==sample_count and {int(i) for i,_ in iterations}==set(range(sample_count)),'incomplete/duplicate native result evidence')
 require(len(choices)==1,'one completed production choice required')
 calls=[json.loads(line) for line in re.findall(r'(?m)^.*?A4_NATIVE (\{.*\})\s*$',text)]
 require(len(calls)==sample_count and len({c.get('id') for c in calls})==sample_count,'complete distinct timed native calls required')
 expected_keys={'id','complete','started_ns','finished_ns','cpu_ns','budget_ms','threads','visits'}
 for c in calls:
  require(set(c)==expected_keys and type(c['id']) is str and re.fullmatch(r'[0-9]+:[0-9]+',c['id']) and c['complete'] is True,'native terminal receipt malformed')
  require(all(integer(c[k]) for k in ('started_ns','finished_ns','cpu_ns','budget_ms','threads','visits')) and 0<c['started_ns']<c['finished_ns'] and c['cpu_ns']>=0 and c['visits']>=0,'native timing/work ledger malformed')
  require(c['budget_ms']==sample_budget_ms and c['threads']==1,'native call configuration drift')
 require_serial_native_calls(calls)
 visits=[int(dict(iterations)[str(i)]) for i in range(sample_count)]
 require(sorted(c['visits'] for c in calls)==sorted(visits),'native/result visit ledger mismatch')
 return {'log_sha256':hashlib.sha256(text.encode()).hexdigest(),'sample_count':sample_count,'sample_budget_ms':sample_budget_ms,'visits_by_sample':visits,'native_calls':calls,'complete':True}

def full_worker_overlap_witnesses(cycles):
 """Unique workers in actual dispatch-to-both-choices intervals, never CPU busy."""
 witnesses=set();peak=0
 # End points are half-open; one finished process cannot overlap its successor.
 for at in sorted({v for _,a,b in cycles for v in (a,b)}):
  live={worker for worker,a,b in cycles if a<=at<b};peak=max(peak,len(live))
  if len(live)==12:witnesses.update(live)
 return peak,witnesses

def readout(plan,rows):
 require(plan['schema']=='pokezero.A4-cost-contention-readout-contract.v2','explicit cost readout contract required')
 roots=plan['root_ids'];workers=plan['workers'];require(len(roots)==20 and len(set(roots))==20,'exact twenty immutable distinct roots required')
 require(len(workers)==12 and len({r['id'] for r in workers})==12,'exact twelve distinct workers required')
 require({r['arm'] for r in workers}==set(ARMS) and all(sum(r['arm']==a for r in workers)==4 for a in ARMS),'four workers per arm required')
 require(plan['cpu']==12 and plan['memory']=='64Gi' and plan['gpu']==1,'declared prospective geometry changed')
 require(plan['algorithmic_cutoff_seconds'] is None and plan['historical_fixed_work_unchanged'] is True,'no new algorithmic deadline')
 require(plan['offline_reference_limit_seconds']==30 and plan['p95_bound_seconds']==15 and plan['offline_reference_justification'],'explicit offline half-limit reference required')
 require(plan['numeric_mode']=='FP32_MHAfastpathOFF_TF32OFF_highest_overrideabsent','all three arms must hold the predeclared numerical mode')
 require(type(plan['fp_source_sha256']) is str and re.fullmatch('[0-9a-f]{64}',plan['fp_source_sha256']),'fixed production opponent source identity required')
 contracts=plan['fp_root_contracts']
 require(len(contracts)==20 and [c['root_id'] for c in contracts]==roots,'exact source-derived FP root contracts required before timing')
 for c in contracts:
  require(c['fp_source_sha256']==plan['fp_source_sha256'],'opponent source differs from fixed plan identity')
  require(type(c['requested']) is bool and integer(c['legal_choice_count']) and c['legal_choice_count']>=0,'source request/actionability evidence required')
  require(c['requested']==(c['legal_choice_count']>0),'source requested/legal coverage mismatch')
  require(all(type(c[k]) is str and re.fullmatch('[0-9a-f]{64}',c[k]) for k in ('boundary_sha256','fp_source_sha256')),'source boundary and FP source identities required')
  require(integer(c['sample_count']) and integer(c['sample_budget_ms']),'predeclared production sample geometry required')
  require((c['sample_count'],c['sample_budget_ms']) in ((1,1000),(2,1000),(2,500),(4,500)) if c['requested'] else (c['sample_count'],c['sample_budget_ms'])==(0,0),'pinned random-battle sample geometry required')
 require(len(rows)==240,'all fixed twenty-root sequences required; no drop/refill')
 mapping={w['id']:w['arm'] for w in workers};by_worker={w:[] for w in mapping};fp=[];cycles=[];eligible_work=[];defects=[];native_intervals=[];requested_work=[]
 for row in rows:
  require(row['worker'] in mapping and row['arm']==mapping[row['worker']],'worker/arm identity mismatch')
  require(integer(row['ordinal']) and 0<=row['ordinal']<20,'fixed root ordinal type/range')
  require(row['root_id']==roots[row['ordinal']],'fixed root ordinal mismatch')
  require(row['input_manifest_sha256']==plan['input_manifest_sha256'] and row['numeric_mode']==plan['numeric_mode'],'root or numerical lineage drift')
  require(all(integer(row[k]) and row[k]==0 for k in ('worker_error','foulplay_error','capture_failures')),'nonzero worker/opponent/capture failure')
  require(all(integer(row[k]) for k in ('foulplay_budget_ms','foulplay_parallelism','foulplay_threads')) and row['foulplay_budget_ms']==1000 and row['foulplay_parallelism']==row['foulplay_threads']==1,'actual Foul Play configuration differs')
  require(all(integer(row[k]) and row[k]>0 for k in ('our_started_ns','our_finished_ns')),'one monotonic host clock and complete terminal stamps required')
  require(row['our_started_ns']<row['our_finished_ns'],'our decision terminal ordering')
  require(type(row['fp_requested']) is bool,'exact recorded opponent requested role required')
  contract=contracts[row['ordinal']]
  require(row['fp_requested']==contract['requested'] and row['fp_root_contract']==contract,'actual row/source actionability drift')
  if row['fp_requested']:
   require(all(integer(row[k]) and row[k]>0 for k in ('fp_request_sent_ns','fp_choice_received_ns')) and row['fp_request_sent_ns']<=row['our_started_ns'] and row['fp_request_sent_ns']<=row['fp_choice_received_ns'],'production overlap ordering not preserved')
  else:require(row['fp_request_sent_ns'] is None and row['fp_choice_received_ns'] is None and row['foulplay_realized_visits']==0,'nonrequested opponent cannot execute a think')
  require(finite_nonnegative(row['foulplay_process_cpu_seconds']) and integer(row['foulplay_realized_visits']) and row['foulplay_realized_visits']>=0,'actual FP CPU/work evidence missing')
  require(row['foulplay_work_status']==('complete' if row['fp_requested'] else 'not_requested'),'incomplete FP attribution; never manufacture zero work')
  evidence=row['foulplay_native_evidence']
  if row['fp_requested']:
   require(type(evidence) is dict and evidence.get('complete') is True,'complete native result receipt required')
   require(evidence.get('sample_count')==contract['sample_count'] and evidence.get('sample_budget_ms')==contract['sample_budget_ms'],'native sampling differs from source-derived root contract')
   require(type(evidence.get('log_sha256')) is str and re.fullmatch('[0-9a-f]{64}',evidence['log_sha256']),'scoped native log identity missing')
   visits=evidence.get('visits_by_sample')
   require(type(visits) is list and len(visits)==contract['sample_count'] and all(integer(v) and v>=0 for v in visits) and sum(visits)==row['foulplay_realized_visits'],'native per-sample visit ledger incomplete')
   cycles.append((row['worker'],row['fp_request_sent_ns'],max(row['our_finished_ns'],row['fp_choice_received_ns'])))
   calls=evidence.get('native_calls')
   require(type(calls) is list and len(calls)==contract['sample_count'] and len({c['id'] for c in calls})==len(calls),'complete distinct native call ledger required')
   for c in calls:
    require(c.get('complete') is True and all(integer(c.get(k)) for k in ('started_ns','finished_ns','cpu_ns','visits','budget_ms','threads')),'complete native call terminal evidence required')
    require(row['fp_request_sent_ns']<=c['started_ns']<c['finished_ns']<=row['fp_choice_received_ns'] and c['cpu_ns']>=0 and c['visits']>=0,'native calls must belong to the actual request interval')
    require(c['budget_ms']==contract['sample_budget_ms'] and c['threads']==1,'actual native call configuration drift')
   require_serial_native_calls(calls)
   require(sorted(c['visits'] for c in calls)==sorted(visits),'actual native/result visit ledger drift')
   require(sum(c['cpu_ns'] for c in calls)/1e9<=row['foulplay_process_cpu_seconds']+1e-6,'native CPU exceeds measured endpoint and reaped children CPU')
   own_work=any(c['cpu_ns']>0 and c['visits']>0 for c in calls)
   requested_work.append((row,own_work))
   native_intervals.extend((c['started_ns'],c['finished_ns']) for c in calls if c['cpu_ns']>0 and c['visits']>0)
   # Complete own results remain mandatory even when a fast raw query finishes
   # before its newly dispatched child starts. Other workers can contend then.
   if contract['legal_choice_count']>1 and not own_work:
    defects.append({'worker':row['worker'],'ordinal':row['ordinal'],'reason':'multi_choice_without_real_own_native_work'})
  else:
   require(evidence is None,'nonrequested root cannot carry a native search receipt')
   cycles.append((row['worker'],row['our_started_ns'],row['our_finished_ns']))
  by_worker[row['worker']].append(row)
  if row['fp_requested']:fp.append((row['fp_request_sent_ns'],row['fp_choice_received_ns']))
 # Source-scheduling definition fixed before actual timing: shared CPU
 # contention can come from any of the twelve workers, never an invented delay.
 for row,own_work in requested_work:
  overlap=any(max(begin,row['our_started_ns'])<min(end,row['our_finished_ns']) for begin,end in native_intervals)
  if own_work and overlap:eligible_work.append(row)
  if row['fp_root_contract']['legal_choice_count']>1 and not overlap:
   defects.append({'worker':row['worker'],'ordinal':row['ordinal'],'reason':'multi_choice_without_shared_native_FP_overlap'})
 for root in roots:
  require(len({r['fp_requested'] for r in rows if r['root_id']==root})==1,'recorded counterpart role changed across replicas/arms')
 for worker,items in by_worker.items():
  require([r['ordinal'] for r in items]==list(range(20)),'missing/duplicate/reordered root; no repair or refill')
  require(all(b['our_started_ns']>=a['our_finished_ns'] and b['our_started_ns']>=(a['fp_choice_received_ns'] or a['our_finished_ns']) and (not b['fp_requested'] or b['fp_request_sent_ns']>=max(a['our_finished_ns'],a['fp_choice_received_ns'] or 0)) for a,b in zip(items,items[1:])),'one worker must finish both choices before the next saved root')
 peak,witnesses=full_worker_overlap_witnesses(cycles)
 if witnesses!=set(mapping):defects.append({'reason':'no_actual_full_twelve_worker_cycle_overlap','witness_workers':sorted(witnesses)})
 for arm in ARMS:
  if not any(r['arm']==arm for r in eligible_work):defects.append({'arm':arm,'reason':'no_actual_busy_native_FP_overlap_witness'})
 result={'schema':'pokezero.A4-cost-contention-readout.v2','unique_roots':20,'workers':12,'rows':240,'all_three_arms_matched_mode':True,'p95_definition':'nearest rank over each distinct root maximum across its four fixed arm replicas','offline_reference_limit_seconds':30,'p95_bound_seconds':15,'algorithmic_cutoff_added':False,'cost_or_panel_admitted':False,'arms':{}}
 for arm in ARMS:
  group=[r for r in rows if r['arm']==arm];timings=[(r['our_finished_ns']-r['our_started_ns'])/1e9 for r in group];root_max={root:max((r['our_finished_ns']-r['our_started_ns'])/1e9 for r in group if r['root_id']==root) for root in roots};covered=weighted=0
  for row in group:
   a,b=overlap_seconds(row['our_started_ns'],row['our_finished_ns'],fp);covered+=a;weighted+=b
  total=sum(timings);p95=nearest_rank_95(list(root_max.values()))
  result['arms'][arm]={'observations':80,'unique_roots':20,'FP_requested_boundaries':sum(r['fp_requested'] for r in group),'per_root_max_seconds':root_max,'p95_seconds':p95,'maximum_seconds':max(timings),'elapsed_FP_request_window_coverage_fraction':covered/total,'mean_concurrent_elapsed_FP_request_windows':weighted/total,'FP_actual_process_cpu_seconds':sum(r['foulplay_process_cpu_seconds'] for r in group),'FP_unrequested_parser_cpu_seconds':sum(r['foulplay_process_cpu_seconds'] for r in group if not r['fp_requested']),'FP_actual_realized_visits':sum(r['foulplay_realized_visits'] for r in group),'elapsed_window_is_not_CPU_busy_time':True}
 result['arithmetic_decision_time_bound_pass']=all(result['arms'][arm]['p95_seconds']<=15 for arm in ARMS[1:])
 result['contention_evidence_eligible']=not defects
 result['contention_defects']=defects
 result['peak_concurrent_worker_cycles']=peak
 result['full_twelve_worker_witnesses']=sorted(witnesses)
 result['actual_busy_native_FP_overlap_rows']=len(eligible_work)
 result['native_overlap_scope']='actual positiveCPU/visits native intervals across the twelve shared workers; own completion/work retained separately'
 result['decision_time_gate_pass']=result['arithmetic_decision_time_bound_pass'] and result['contention_evidence_eligible']
 return result
