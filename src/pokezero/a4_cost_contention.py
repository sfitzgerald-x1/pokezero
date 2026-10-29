"""Pure readout for predeclared A4 twelve-worker saved-root cost geometry.

No model, simulator, opponent, native search, remote access, or process execution.
FP request windows are labelled elapsed windows, never claimed as CPU busy time.
"""
from __future__ import annotations
import math

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

def readout(plan,rows):
 require(plan['schema']=='pokezero.A4-cost-contention-readout-contract.v1','explicit cost readout contract required')
 roots=plan['root_ids'];workers=plan['workers'];require(len(roots)==20 and len(set(roots))==20,'exact twenty immutable distinct roots required')
 require(len(workers)==12 and len({r['id'] for r in workers})==12,'exact twelve distinct workers required')
 require({r['arm'] for r in workers}==set(ARMS) and all(sum(r['arm']==a for r in workers)==4 for a in ARMS),'four workers per arm required')
 require(plan['cpu']==12 and plan['memory']=='64Gi' and plan['gpu']==1,'declared prospective geometry changed')
 require(plan['algorithmic_cutoff_seconds'] is None and plan['historical_fixed_work_unchanged'] is True,'no new algorithmic deadline')
 require(plan['offline_reference_limit_seconds']==30 and plan['p95_bound_seconds']==15 and plan['offline_reference_justification'],'explicit offline half-limit reference required')
 require(plan['numeric_mode']=='FP32_MHAfastpathOFF_TF32OFF_highest_overrideabsent','all three arms must hold the predeclared numerical mode')
 require(len(rows)==240,'all fixed twenty-root sequences required; no drop/refill')
 mapping={w['id']:w['arm'] for w in workers};by_worker={w:[] for w in mapping};fp=[]
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
  if row['fp_requested']:
   require(all(integer(row[k]) and row[k]>0 for k in ('fp_request_sent_ns','fp_choice_received_ns')) and row['fp_request_sent_ns']<=row['our_started_ns'] and row['fp_request_sent_ns']<=row['fp_choice_received_ns'],'production overlap ordering not preserved')
  else:require(row['fp_request_sent_ns'] is None and row['fp_choice_received_ns'] is None and row['foulplay_realized_visits']==0,'nonrequested opponent cannot execute a think')
  require(finite_nonnegative(row['foulplay_process_cpu_seconds']) and integer(row['foulplay_realized_visits']) and row['foulplay_realized_visits']>=0,'actual FP CPU/work evidence missing')
  require(row['foulplay_work_status']==('complete' if row['fp_requested'] else 'not_requested'),'incomplete FP attribution; never manufacture zero work')
  by_worker[row['worker']].append(row)
  if row['fp_requested']:fp.append((row['fp_request_sent_ns'],row['fp_choice_received_ns']))
 for root in roots:
  require(len({r['fp_requested'] for r in rows if r['root_id']==root})==1,'recorded counterpart role changed across replicas/arms')
 for worker,items in by_worker.items():
  require([r['ordinal'] for r in items]==list(range(20)),'missing/duplicate/reordered root; no repair or refill')
  require(all(b['our_started_ns']>=a['our_finished_ns'] and b['our_started_ns']>=(a['fp_choice_received_ns'] or a['our_finished_ns']) and (not b['fp_requested'] or b['fp_request_sent_ns']>=max(a['our_finished_ns'],a['fp_choice_received_ns'] or 0)) for a,b in zip(items,items[1:])),'one worker must finish both choices before the next saved root')
 result={'schema':'pokezero.A4-cost-contention-readout.v1','unique_roots':20,'workers':12,'rows':240,'all_three_arms_matched_mode':True,'p95_definition':'nearest rank over each distinct root maximum across its four fixed arm replicas','offline_reference_limit_seconds':30,'p95_bound_seconds':15,'algorithmic_cutoff_added':False,'cost_or_panel_admitted':False,'arms':{}}
 for arm in ARMS:
  group=[r for r in rows if r['arm']==arm];timings=[(r['our_finished_ns']-r['our_started_ns'])/1e9 for r in group];root_max={root:max((r['our_finished_ns']-r['our_started_ns'])/1e9 for r in group if r['root_id']==root) for root in roots};covered=weighted=0
  for row in group:
   a,b=overlap_seconds(row['our_started_ns'],row['our_finished_ns'],fp);covered+=a;weighted+=b
  total=sum(timings);p95=nearest_rank_95(list(root_max.values()))
  result['arms'][arm]={'observations':80,'unique_roots':20,'FP_requested_boundaries':sum(r['fp_requested'] for r in group),'per_root_max_seconds':root_max,'p95_seconds':p95,'maximum_seconds':max(timings),'elapsed_FP_request_window_coverage_fraction':covered/total,'mean_concurrent_elapsed_FP_request_windows':weighted/total,'FP_actual_process_cpu_seconds':sum(r['foulplay_process_cpu_seconds'] for r in group),'FP_unrequested_parser_cpu_seconds':sum(r['foulplay_process_cpu_seconds'] for r in group if not r['fp_requested']),'FP_actual_realized_visits':sum(r['foulplay_realized_visits'] for r in group),'elapsed_window_is_not_CPU_busy_time':True}
 result['decision_time_gate_pass']=all(result['arms'][arm]['p95_seconds']<=15 for arm in ARMS[1:])
 return result
