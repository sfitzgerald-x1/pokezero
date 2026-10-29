from copy import deepcopy
import unittest
from pokezero.a4_cost_contention import ARMS, readout, overlap_seconds, native_search_log_evidence, _NativeCallTiming, native_call_timing


def fixture():
    arms=ARMS;workers=[{'id':a+str(i),'arm':a} for a in arms for i in range(4)];mode='FP32_MHAfastpathOFF_TF32OFF_highest_overrideabsent';plan={'schema':'pokezero.A4-cost-contention-readout-contract.v2','root_ids':['authored'+str(i) for i in range(20)],'workers':workers,'cpu':12,'memory':'64Gi','gpu':1,'algorithmic_cutoff_seconds':None,'historical_fixed_work_unchanged':True,'offline_reference_limit_seconds':30,'p95_bound_seconds':15,'offline_reference_justification':'explicit prospective half-limit offline reference; never search cutoff','numeric_mode':mode,'input_manifest_sha256':'authored','fp_source_sha256':'f'*64}
    plan['fp_root_contracts']=[{'root_id':root,'requested':i%2==0,'legal_choice_count':2 if i%2==0 else 0,'boundary_sha256':'b'*64,'fp_source_sha256':'f'*64,'sample_count':2 if i%2==0 else 0,'sample_budget_ms':1000 if i%2==0 else 0} for i,root in enumerate(plan['root_ids'])]
    rows=[]
    for worker in workers:
     for i,root in enumerate(plan['root_ids']):
      base=(i*40+1)*10**9;fp_requested=i%2==0;rows.append({'worker':worker['id'],'arm':worker['arm'],'root_id':root,'ordinal':i,'input_manifest_sha256':'authored','numeric_mode':mode,'worker_error':0,'foulplay_error':0,'capture_failures':0,'foulplay_budget_ms':1000,'foulplay_parallelism':1,'foulplay_threads':1,'our_started_ns':base+10**8,'our_finished_ns':base+2100000000,'fp_requested':fp_requested,'fp_request_sent_ns':base if fp_requested else None,'fp_choice_received_ns':base+10**9 if fp_requested else None,'foulplay_process_cpu_seconds':.9 if fp_requested else 0,'foulplay_realized_visits':100 if fp_requested else 0,'foulplay_work_status':'complete' if fp_requested else 'not_requested','fp_root_contract':deepcopy(plan['fp_root_contracts'][i]),'foulplay_native_evidence':{'complete':True,'sample_count':2,'sample_budget_ms':1000,'log_sha256':'a'*64,'visits_by_sample':[50,50],'native_calls':[{'id':'1:'+str(j+1),'complete':True,'started_ns':base+j*400000000+200000000,'finished_ns':base+j*400000000+500000000,'cpu_ns':200000000,'visits':50,'budget_ms':1000,'threads':1} for j in range(2)]} if fp_requested else None})
    return plan, rows


class CostContentionTests(unittest.TestCase):
    def test_complete_geometry_reports_elapsed_coverage_and_worst_replica_roots(self):
        plan, rows=fixture();result=readout(plan,rows)
        self.assertTrue(result['decision_time_gate_pass']);self.assertFalse(result['cost_or_panel_admitted'])
        for arm in result['arms'].values():
            self.assertEqual(arm['p95_seconds'],2)
            self.assertEqual(arm['FP_requested_boundaries'],40)
            self.assertAlmostEqual(arm['elapsed_FP_request_window_coverage_fraction'],.225)
            self.assertAlmostEqual(arm['mean_concurrent_elapsed_FP_request_windows'],2.7)
    def test_all_fixed_root_replicas_contribute_to_conservative_p95(self):
        plan,rows=fixture()
        for row in rows:
            if row['arm']==ARMS[1] and row['ordinal'] in (18,19):
                row['our_finished_ns']=row['our_started_ns']+16*10**9
        self.assertFalse(readout(plan,rows)['decision_time_gate_pass'])
    def test_malformed_or_missing_work_never_passes(self):
        plan,rows=fixture()
        mutations={
            'missing':lambda r:r.pop(),
            'duplicate':lambda r:r.__setitem__(1,deepcopy(r[0])),
            'late_dispatch':lambda r:r[0].update(fp_request_sent_ns=r[0]['our_started_ns']+1),
            'wrong_budget':lambda r:r[0].update(foulplay_budget_ms=500),
            'mode':lambda r:r[0].update(numeric_mode='ON'),
            'CPU_missing':lambda r:r[0].update(foulplay_process_cpu_seconds=None),
            'visits_missing':lambda r:r[0].update(foulplay_realized_visits=None),
            'unmeasured':lambda r:r[0].update(foulplay_work_status='unmeasured'),
            'lineage':lambda r:r[0].update(input_manifest_sha256='changed'),
            'nonrequested_think':lambda r:r[1].update(fp_choice_received_ns=99),
            'failure':lambda r:r[0].update(capture_failures=1),
            'bool_ordinal':lambda r:r[0].update(ordinal=False),
            'bool_failure':lambda r:r[0].update(capture_failures=False),
            'bool_threads':lambda r:r[0].update(foulplay_threads=True),
            'early_next_request':lambda r:r[2].update(fp_request_sent_ns=1),
            'role_drift':lambda r:r[0].update(fp_requested=False,fp_request_sent_ns=None,fp_choice_received_ns=None,foulplay_realized_visits=0,foulplay_work_status='not_requested'),
        }
        for name,mutate in mutations.items():
            changed=deepcopy(rows);mutate(changed)
            with self.subTest(name=name),self.assertRaises((ValueError,KeyError)):readout(plan,changed)
    def test_missing_busy_work_retains_timings_but_never_admits(self):
        for zero,early in ((True,False),(False,True)):
            plan,rows=fixture()
            for row in rows:
                if row['fp_requested']:
                    if zero:
                        row.update(foulplay_process_cpu_seconds=0,foulplay_realized_visits=0)
                        row['foulplay_native_evidence']['visits_by_sample']=[0,0]
                        for c in row['foulplay_native_evidence']['native_calls']:c.update(cpu_ns=0,visits=0)
                    if early:
                        row['fp_choice_received_ns']=row['our_started_ns']
                        for j,c in enumerate(row['foulplay_native_evidence']['native_calls']):c.update(started_ns=row['fp_request_sent_ns']+j*10000000,finished_ns=row['fp_request_sent_ns']+(j+1)*10000000)
            result=readout(plan,rows)
            self.assertFalse(result['decision_time_gate_pass'])
            self.assertTrue(result['arithmetic_decision_time_bound_pass'])
            self.assertEqual(sum(a['observations'] for a in result['arms'].values()),240)
    def test_serialized_workers_cannot_certify_twelve_worker_contention(self):
        plan,rows=fixture()
        for index,worker in enumerate(plan['workers']):
            for row in rows:
                if row['worker']==worker['id']:
                    for key in ('our_started_ns','our_finished_ns','fp_request_sent_ns','fp_choice_received_ns'):
                        if row[key] is not None:row[key]+=index*1000*10**9
                    if row['foulplay_native_evidence']:
                        for c in row['foulplay_native_evidence']['native_calls']:
                            c['started_ns']+=index*1000*10**9;c['finished_ns']+=index*1000*10**9
        result=readout(plan,rows)
        self.assertEqual(result['peak_concurrent_worker_cycles'],1)
        self.assertFalse(result['decision_time_gate_pass'])
        self.assertTrue(result['arithmetic_decision_time_bound_pass'])
    def test_legitimate_single_choice_zero_work_is_retained_not_fabricated(self):
        plan,rows=fixture()
        plan['fp_root_contracts'][0]['legal_choice_count']=1
        for row in rows:
            if row['ordinal']==0:
                row['fp_root_contract']=deepcopy(plan['fp_root_contracts'][0])
                row.update(foulplay_process_cpu_seconds=0,foulplay_realized_visits=0)
                row['foulplay_native_evidence']['visits_by_sample']=[0,0]
                for c in row['foulplay_native_evidence']['native_calls']:c.update(cpu_ns=0,visits=0)
        result=readout(plan,rows)
        self.assertTrue(result['decision_time_gate_pass'])
        self.assertEqual(result['actual_busy_native_FP_overlap_rows'],108)
        self.assertEqual(sum(a['observations'] for a in result['arms'].values()),240)
    def test_all_forced_no_busy_roots_leave_contention_unproved(self):
        plan,rows=fixture()
        for c in plan['fp_root_contracts']:
            if c['requested']:c['legal_choice_count']=1
        for row in rows:
            row['fp_root_contract']=deepcopy(plan['fp_root_contracts'][row['ordinal']])
            if row['fp_requested']:
                row.update(foulplay_process_cpu_seconds=0,foulplay_realized_visits=0)
                row['foulplay_native_evidence']['visits_by_sample']=[0,0]
                for c in row['foulplay_native_evidence']['native_calls']:c.update(cpu_ns=0,visits=0)
        result=readout(plan,rows)
        self.assertFalse(result['contention_evidence_eligible'])
        self.assertEqual(result['rows'],240)
    def test_native_evidence_and_source_actionability_cannot_drift(self):
        for key,value in [('fp_root_contract',{}),('foulplay_native_evidence',{}),('foulplay_realized_visits',99)]:
            plan,rows=fixture();rows[0][key]=value
            with self.subTest(key=key),self.assertRaises((ValueError,KeyError)):readout(plan,rows)
    def test_real_native_log_requires_complete_scoped_sampling_and_results(self):
        import json
        calls=[{'id':'1:'+str(i+1),'complete':True,'started_ns':1+i*3,'finished_ns':3+i*3,'cpu_ns':1,'visits':v,'budget_ms':1000,'threads':1} for i,v in enumerate((43,57))]
        text='INFO Sampling 2 battles at 1000ms each\n'+''.join('INFO A4_NATIVE '+json.dumps(c)+'\n' for c in calls)+'INFO Iterations 1: 43\nINFO Iterations 0: 57\nINFO Choice: splash\n'
        evidence=native_search_log_evidence(text,2,1000)
        self.assertEqual(evidence['visits_by_sample'],[57,43])
        for changed in (text.replace('Iterations 0: 57','Iterations 1: 57'),text.replace('INFO Iterations 0: 57\n',''),text+text,text.replace('1000ms','500ms'),text.replace('INFO Choice: splash\n','')):
            with self.subTest(changed=changed),self.assertRaises(ValueError):native_search_log_evidence(changed,2,1000)
    def test_native_timing_is_transparent_and_records_real_call_only(self):
        from types import SimpleNamespace
        emitted=[];seen=[];clock=iter((10,20));cpu=iter((100,107))
        answer=SimpleNamespace(total_visits=4)
        target=lambda state,budget,threads:(seen.append((state,budget,threads)) or answer)
        hook=_NativeCallTiming(target,emitted.append,lambda:next(clock),lambda:next(cpu),lambda:9)
        self.assertIs(hook('opaque',1000,threads=1),answer)
        self.assertEqual(seen,[('opaque',1000,1)])
        self.assertIn('"cpu_ns":7',emitted[0]);self.assertNotIn('opaque',emitted[0])
    def test_native_hook_restores_target_on_failure_and_refuses_changed_geometry(self):
        from types import SimpleNamespace
        from unittest.mock import patch
        module=SimpleNamespace(monte_carlo_tree_search=object());original=module.monte_carlo_tree_search
        with patch('pokezero.a4_cost_contention.multiprocessing.get_context',return_value=SimpleNamespace(get_start_method=lambda:'fork')):
            with self.assertRaises(RuntimeError):
                with native_call_timing(module,lambda text:None):raise RuntimeError('authored')
        self.assertIs(module.monte_carlo_tree_search,original)
        with patch('pokezero.a4_cost_contention.multiprocessing.get_context',return_value=SimpleNamespace(get_start_method=lambda:'spawn')):
            with self.assertRaises(ValueError):
                with native_call_timing(module,lambda text:None):pass
        self.assertIs(module.monte_carlo_tree_search,original)
    def test_request_overlap_without_native_overlap_remains_ineligible(self):
        plan,rows=fixture()
        for row in rows:
            if row['fp_requested']:
                for j,c in enumerate(row['foulplay_native_evidence']['native_calls']):c.update(started_ns=row['fp_request_sent_ns']+j*10000000,finished_ns=row['fp_request_sent_ns']+(j+1)*10000000)
        result=readout(plan,rows)
        self.assertFalse(result['decision_time_gate_pass'])
        self.assertTrue(result['arithmetic_decision_time_bound_pass'])
    def test_native_failure_keeps_failure_receipt_without_retry(self):
        emitted=[];calls=[];clock=iter((10,20));cpu=iter((100,107))
        def target(*args,**kw):calls.append(1);raise KeyboardInterrupt()
        hook=_NativeCallTiming(target,emitted.append,lambda:next(clock),lambda:next(cpu),lambda:9)
        with self.assertRaises(KeyboardInterrupt):hook('opaque',1000,threads=1)
        self.assertEqual(calls,[1]);self.assertIn('"complete":false',emitted[0])
    def test_native_interval_outside_own_request_refuses(self):
        plan,rows=fixture();rows[0]['foulplay_native_evidence']['native_calls'][0]['started_ns']=1
        with self.assertRaises(ValueError):readout(plan,rows)
    def test_actual_time_pressure_geometry_preserves_same_fixed_1000ms_configuration(self):
        plan,rows=fixture();plan['fp_root_contracts'][0].update(sample_count=1,sample_budget_ms=1000)
        for row in rows:
            if row['ordinal']==0:
                row['fp_root_contract']=deepcopy(plan['fp_root_contracts'][0])
                evidence=row['foulplay_native_evidence'];evidence.update(sample_count=1,visits_by_sample=[100])
                evidence['native_calls']=evidence['native_calls'][:1];evidence['native_calls'][0]['visits']=100
        self.assertTrue(readout(plan,rows)['decision_time_gate_pass'])
    def test_fixed_opponent_source_identity_is_required_across_roots(self):
        plan,rows=fixture();plan['fp_root_contracts'][0]['fp_source_sha256']='e'*64
        for row in rows:
            if row['ordinal']==0:row['fp_root_contract']=deepcopy(plan['fp_root_contracts'][0])
        with self.assertRaises(ValueError):readout(plan,rows)
    def test_observed_single_child_serial_geometry_is_required(self):
        for mutation in ('pid','counter','overlap'):
            plan,rows=fixture();calls=rows[0]['foulplay_native_evidence']['native_calls']
            if mutation=='pid':calls[1]['id']='2:2'
            elif mutation=='counter':calls[1]['id']='1:3'
            else:calls[1]['started_ns']=calls[0]['finished_ns']-1
            with self.subTest(mutation=mutation),self.assertRaises(ValueError):readout(plan,rows)
    def test_fast_raw_query_can_overlap_other_workers_actual_native_activity(self):
        plan,rows=fixture()
        for row in rows:
            if row['arm']==ARMS[0]:row['our_finished_ns']=row['our_started_ns']+50000000
            elif row['fp_requested']:
                base=row['fp_request_sent_ns']
                for j,c in enumerate(row['foulplay_native_evidence']['native_calls']):c.update(started_ns=base+110000000+j*30000000,finished_ns=base+130000000+j*30000000,cpu_ns=10000000)
        # Raw own native calls start200ms after dispatch, after the raw query ends
        # at150ms. Real other-worker native calls at110..160ms supply contention.
        result=readout(plan,rows)
        self.assertTrue(result['decision_time_gate_pass'])
        self.assertEqual(result['actual_busy_native_FP_overlap_rows'],120)
        self.assertEqual(result['rows'],240)
    def test_unrequested_peer_windows_cannot_invent_missing_own_multichoice_work(self):
        plan,rows=fixture()
        for row in rows:
            if row['arm']==ARMS[0] and row['fp_requested']:
                row.update(foulplay_process_cpu_seconds=0,foulplay_realized_visits=0)
                row['foulplay_native_evidence']['visits_by_sample']=[0,0]
                for c in row['foulplay_native_evidence']['native_calls']:c.update(cpu_ns=0,visits=0)
        result=readout(plan,rows)
        self.assertFalse(result['decision_time_gate_pass'])
        self.assertTrue(result['arithmetic_decision_time_bound_pass'])
        self.assertEqual(result['rows'],240)
    def test_union_coverage_does_not_count_overlapping_windows_twice(self):
        self.assertEqual(overlap_seconds(10**9,4*10**9,[(0,3*10**9),(2*10**9,5*10**9)]),(3,4))


if __name__=='__main__':unittest.main()
