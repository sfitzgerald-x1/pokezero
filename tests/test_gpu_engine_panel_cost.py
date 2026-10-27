"""Authored latency/roster accounting; no root generation or scientific work."""
from copy import deepcopy
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from gpu_engine import panel_cost as c,collect_fidelity_panel as collector


def entry():
    result=dict(slots=20,team_start=100,prefix_cycle=5,operation_budget=dict(
        schema=c.BUDGET_SCHEMA,operation=c.TIMING_SCOPE,limit_seconds=60))
    result['cost_admission']=dict(schema=c.SCHEMA,decision_limit_seconds=60,
        timing_scope=c.TIMING_SCOPE,operation_budget_sha256=c.operation_budget(result)[1])
    return result


def completed(path='ordinary',seconds=10):
    return [dict(slot=i,status='COMPLETE',seconds=seconds,path=path) for i in range(20)]


class PanelCostTests(unittest.TestCase):
    def test_ordinary_launch_rejects_cost_mode_before_generation_or_worker(self):
        from gpu_engine import panel_owner
        config=dict(entry(),slots=1000)
        with tempfile.TemporaryDirectory() as directory:
            out=Path(directory)/'must-not-exist'
            with mock.patch.object(collector,'_generate_inputs') as generate, \
                    mock.patch.object(panel_owner.subprocess,'Popen') as spawn:
                with self.assertRaises(ValueError):
                    collector.generate_inputs(None,None,config)
                with self.assertRaises(ValueError):
                    collector.run(config,Path('unused-authored.npz'),out)
                with self.assertRaises(ValueError):
                    panel_owner.own(['unused'],out=out,panel=out/'panel',
                        entry=config,calendar_seconds=1)
                generate.assert_not_called();spawn.assert_not_called()
                self.assertFalse(out.exists())

    def test_exact_twenty_row_cost_roster_and_ordinary_minimum_are_separate(self):
        rows=collector.cost_source_rows(entry())
        self.assertEqual(len(rows),20)
        self.assertEqual(rows[0],dict(slot=0,teams=[100,101],prefix_requests=0))
        self.assertEqual(rows[-1],dict(slot=19,teams=[138,139],prefix_requests=4))
        self.assertEqual(len({n for row in rows for n in row['teams']}),40)
        with self.assertRaises(ValueError):collector.source_rows(20,100,5)
        self.assertEqual(len(collector.source_rows(1000,100,5)),1000)

    def test_cost_preparation_passes_exact_rows_to_shared_generation_without_selecting(self):
        original=object();pool=object();table=object();config=entry()
        with mock.patch.object(collector,'_generate_inputs',return_value='authored sentinel') as generate:
            self.assertEqual(collector.generate_cost_inputs(pool,table,config,
                constructors=original),'authored sentinel')
        args,kw=generate.call_args
        self.assertEqual(args,(pool,table,config))
        self.assertEqual(kw['rows'],collector.cost_source_rows(config))
        self.assertIs(kw['constructors'],original)
        with self.assertRaises(ValueError):collector.generate_cost_inputs(pool,table,config)

    def test_explicit_decision_budget_and_cost_purpose_are_mandatory(self):
        cases=[dict(slots=20,calendar_seconds=28800),dict(entry(),slots=19),dict(entry(),slots=True)]
        for key,value in [('schema','other'),('timing_scope','plan only'),
                ('decision_limit_seconds',True),('decision_limit_seconds',0),
                ('decision_limit_seconds',float('inf'))]:
            cfg=deepcopy(entry());cfg['cost_admission'][key]=value;cases.append(cfg)
        for cfg in cases:
            with self.subTest(cfg=cfg),self.assertRaises(ValueError):c.contract(cfg)

    def test_nearest_rank_nineteen_and_full_roster_not_maximum(self):
        obs=completed();obs[-1]['seconds']=100
        result=c.summarize(entry(),obs)
        self.assertEqual(result['percentile_rank'],19)
        self.assertEqual(result['p95_upper_seconds'],10)
        self.assertEqual(result['threshold_seconds'],30)
        self.assertEqual(result['latency_threshold'],'MET')
        self.assertEqual(result['timed_roots'],20)
        self.assertFalse(result['comparison_cost_qualified'])

    def test_one_missing_completion_is_unknown_without_automatic_percentile_failure(self):
        obs=completed();obs[-1]=dict(slot=19,status='MISSING')
        result=c.summarize(entry(),obs)
        self.assertEqual(result['p95_upper_seconds'],10)
        self.assertEqual(result['latency_threshold'],'MET')
        self.assertFalse(result['minimum_timed_roots_met'])
        self.assertEqual(result['attempt_integrity'],'NOT_ASSESSED_BY_LATENCY_ACCOUNTING')
        self.assertFalse(result['comparison_cost_qualified'])

    def test_proven_censor_bounds_and_unknown_witnesses_remain_distinct(self):
        obs=completed()
        for i in (18,19):obs[i]=dict(slot=i,status='CENSORED',seconds=60,path='conditional')
        result=c.summarize(entry(),obs)
        self.assertEqual(result['p95_lower_seconds'],60)
        self.assertEqual(result['p95_upper_seconds'],'unbounded')
        self.assertEqual(result['latency_threshold'],'FAILED')
        for i in (18,19):obs[i]=dict(slot=i,status='MISSING')
        result=c.summarize(entry(),obs)
        self.assertEqual(result['p95_lower_seconds'],10)
        self.assertEqual(result['latency_threshold'],'UNRESOLVED')
        self.assertEqual(result['p95_upper_seconds'],'unbounded')

    def test_fast_refusals_never_claim_exercised_comparisons(self):
        result=c.summarize(entry(),completed(path='refused',seconds=.001))
        self.assertEqual(result['latency_threshold'],'MET')
        self.assertEqual(result['exercised_paths'],dict(refused=20))
        self.assertEqual(result['unexercised_comparison_paths'],['ordinary','conditional'])
        self.assertFalse(result['comparison_cost_qualified'])
        self.assertFalse(result['scientific_fidelity_qualified'])

    def test_missing_duplicate_changed_and_nonfinite_observations_cannot_pass(self):
        cases=[completed()[:-1],completed()+[completed()[0]]]
        obs=completed();obs[1]['slot']=0;cases.append(obs)
        obs=completed();obs[0]['seconds']=float('nan');cases.append(obs)
        obs=completed();obs[0]=dict(slot=0,status='UNATTEMPTED',seconds=0,path='ordinary');cases.append(obs)
        for obs in cases:
            with self.assertRaises(ValueError):c.summarize(entry(),obs)


if __name__=='__main__':unittest.main()
