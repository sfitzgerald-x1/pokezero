from copy import deepcopy
import unittest
from pokezero.a4_cost_contention import ARMS, readout, overlap_seconds


def fixture():
    arms=ARMS;workers=[{'id':a+str(i),'arm':a} for a in arms for i in range(4)];mode='FP32_MHAfastpathOFF_TF32OFF_highest_overrideabsent';plan={'schema':'pokezero.A4-cost-contention-readout-contract.v1','root_ids':['authored'+str(i) for i in range(20)],'workers':workers,'cpu':12,'memory':'64Gi','gpu':1,'algorithmic_cutoff_seconds':None,'historical_fixed_work_unchanged':True,'offline_reference_limit_seconds':30,'p95_bound_seconds':15,'offline_reference_justification':'explicit prospective half-limit offline reference; never search cutoff','numeric_mode':mode,'input_manifest_sha256':'authored'}
    rows=[]
    for worker in workers:
     for i,root in enumerate(plan['root_ids']):
      base=(i*40+1)*10**9;fp_requested=i%2==0;rows.append({'worker':worker['id'],'arm':worker['arm'],'root_id':root,'ordinal':i,'input_manifest_sha256':'authored','numeric_mode':mode,'worker_error':0,'foulplay_error':0,'capture_failures':0,'foulplay_budget_ms':1000,'foulplay_parallelism':1,'foulplay_threads':1,'our_started_ns':base+10**8,'our_finished_ns':base+2100000000,'fp_requested':fp_requested,'fp_request_sent_ns':base if fp_requested else None,'fp_choice_received_ns':base+10**9 if fp_requested else None,'foulplay_process_cpu_seconds':.9 if fp_requested else 0,'foulplay_realized_visits':100 if fp_requested else 0,'foulplay_work_status':'complete' if fp_requested else 'not_requested'})
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
    def test_union_coverage_does_not_count_overlapping_windows_twice(self):
        self.assertEqual(overlap_seconds(10**9,4*10**9,[(0,3*10**9),(2*10**9,5*10**9)]),(3,4))


if __name__=='__main__':unittest.main()
