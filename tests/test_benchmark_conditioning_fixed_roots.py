import copy
import importlib.util
from pathlib import Path
import unittest
import sys


path = Path(__file__).resolve().parents[1] / 'scripts/benchmark_conditioning_fixed_roots.py'
spec = importlib.util.spec_from_file_location('fixed_conditioning_benchmark', path)
benchmark = importlib.util.module_from_spec(spec)
spec.loader.exec_module(benchmark)


class BenchmarkScopeTests(unittest.TestCase):
    def manifest(self):
        names = ('seed-1167174895-p2-paper_reference',
                 'seed-3495319813-p1-paper_reference', 'seed-474486314-p1-paper_reference')
        return {'execution_schedule': {'deferred': {name: dict(accepted_boundaries=1,
            accepted_prefix_steps={'boundary-000.json.gz': '/unchanged/' + name},
            accepted_prefix_hashes={'boundary-000.json.gz': 'immutable'},
            result_path='/unchanged/' + name + '.json', result_sha256='unscored') for name in names}}}

    def test_exact_existing_failures_and_control_no_redraw(self):
        cases = benchmark.fixed_cases(self.manifest())
        self.assertEqual(len(cases), 4)
        self.assertEqual([case['seed'] for case in cases], [1167174895, 3495319813, 474486314, 3495319813])
        self.assertEqual([case['player'] for case in cases], ['p2','p1','p1','p1'])
        self.assertEqual(cases[-1]['prefix'], [])
        self.assertTrue(all(case['signed_outcome'] is None for case in cases))

    def test_missing_replaced_and_gapped_failures_refuse(self):
        good = self.manifest()
        variants = []
        value = copy.deepcopy(good)
        value['execution_schedule']['deferred'].pop('seed-474486314-p1-paper_reference')
        variants.append(value)
        value = copy.deepcopy(good)
        row = value['execution_schedule']['deferred']['seed-474486314-p1-paper_reference']
        row['accepted_boundaries'] = 2
        variants.append(value)
        value = copy.deepcopy(good)
        row = value['execution_schedule']['deferred']['seed-474486314-p1-paper_reference']
        row['accepted_prefix_steps'] = {'boundary-001.json.gz': 'wrong'}
        variants.append(value)
        for value in variants:
            with self.subTest(value=value), self.assertRaises(ValueError):
                benchmark.fixed_cases(value)

    def test_fourth_refusal_keeps_all_89_hashed_boundaries_and_remains_unscored(self):
        terminal = dict(status='REFUSED', signed_outcome=None, step_hashes={
            f'boundary-{i:03d}.json.gz': str(i) for i in range(89)})
        case = benchmark.slow_baton_case(Path('/retained'), terminal)
        self.assertEqual((case['name'], case['seed'], case['player']),
            ('seed-566883983-p2-paper_reference', 566883983, 'p2'))
        self.assertEqual(len(case['prefix']), 89)
        self.assertIsNone(case['signed_outcome'])
        for bad in (dict(terminal, signed_outcome=-1), dict(terminal, status='COMPLETE'),
                    dict(terminal, step_hashes={}), dict(terminal, step_hashes={
                        **terminal['step_hashes'], 'boundary-089.json.gz': 'extra'})):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                benchmark.slow_baton_case(Path('/retained'), bad)

    def test_predeclared_seed_is_paired_by_case_and_replicate_not_mode(self):
        case = 'seed-3495319813-p1-paper_reference'
        seeds = [benchmark.trial_seed(case, i) for i in range(3)]
        self.assertEqual(len(set(seeds)), 3)
        self.assertEqual(seeds, [benchmark.trial_seed(case, i) for i in range(3)])
        self.assertNotEqual(seeds[0], benchmark.trial_seed('other-fixed-case', 0))

    def test_runtime_requires_explicit_boolean_opt_ins(self):
        from pokezero.mcts_eval.paper_reference_runtime import ShowdownWorkerFactory
        from pokezero.mcts_eval.paper_reference import ReferenceRefusal
        for option in ('history_particles', 'collect_phase_timing'):
            with self.subTest(option=option), self.assertRaises(ReferenceRefusal):
                ShowdownWorkerFactory('checkpoint', 'digest', 'engine', 'source', **{option: 1})

    def test_owned_timeout_reaps_harmless_child_and_descendant(self):
        # This starts only two disposable Python sleepers, never a simulator.
        code = ("import signal,subprocess,sys,time; "
                "signal.signal(signal.SIGTERM,signal.SIG_IGN); "
                "subprocess.Popen([sys.executable,'-c',"
                "'import signal,time; signal.signal(signal.SIGTERM,signal.SIG_IGN); time.sleep(30)']); "
                "time.sleep(30)")
        row = benchmark.isolated_trial(dict(case={'name':'harmless'},mode='fake',replicate=0),
            command=[sys.executable, '-c', code], timeout_seconds=.2)
        self.assertEqual(row['status'], 'BENCHMARK_ERROR')
        self.assertNotEqual(row['actual_child_exit_code'], 0)
        self.assertIn('no retry', row['error'])
        self.assertTrue(row['owned_group_no_live_processes'])

    def trial(self, case='fixed', mode='baseline_instrumented', replicate=0, completed=1):
        status = 'SEARCH_WORK_MEASURED' if completed else 'ZERO_COMPLETED_TRAJECTORIES'
        return dict(case=case, mode=mode, replicate=replicate,
            seed=benchmark.trial_seed(case, replicate), actor_root_key='same-root-' + case,
            status=status, actual_child_exit_code=0 if completed else 1,
            wall_search_seconds=10.1,
            worker_evidence=dict(neural_forwards=3,
                draws=[dict(status='ROOT_VALIDATED') for _ in range(completed)] or [dict(status='DEADLINE_CANCELLED')],
                search_phase_timing=[dict(schema='pokezero.reference-search-phases.v1',
                    phase_timing=dict(root_inference=dict(seconds=.1),
                        world_reconstruction_inclusive=dict(seconds=8.0),
                        forward_search_inclusive=dict(seconds=1.0 if completed else 0.0),
                        world_cleanup=dict(seconds=.1 if completed else 0.0)),
                    completed_trajectories=completed, forward_transitions=2*completed,
                    attempted_world_draws=completed or 1)]))

    def test_disjoint_search_phases_ignore_nested_inclusive_timers(self):
        row = self.trial()
        row['worker_evidence']['draws'][0]['phase_timing'] = {
            'factory_total_inclusive': dict(seconds=8.0),
            'anchor_sampling_inclusive': dict(seconds=7.0)}
        summary = benchmark.measured_trial(row)
        self.assertAlmostEqual(summary['reconstruction_fraction_of_search_wall'], 8/10.1)
        self.assertAlmostEqual(summary['uninstrumented_search_wall_seconds'], .9)

    def test_cancelled_membership_check_and_staged_unavailable_are_explicit(self):
        row = self.trial(completed=0)
        row['worker_evidence']['draws'][0].update(
            conditioning_metrics=dict(counter_scope='certified_staged_path', ordinary_counters_available=False),
            necessary_party_conditioning=dict(schema='pokezero.history-necessary-party.v1',
                complete_team_proposals=10, membership_rejections=7, membership_matches=2))
        summary = benchmark.measured_trial(row)
        self.assertEqual(summary['staged_receipts_without_ordinary_counts'], 1)
        self.assertEqual(summary['ordinary_joint_receipts'], 0)
        self.assertIsNone(summary['observed_ordinary_world_yield_per_started_proposal'])
        self.assertEqual(summary['necessary_party_draws_censored_before_membership_check'], 1)

    def test_ordinary_yield_keeps_censored_proposals_in_denominator(self):
        row = self.trial()
        row['worker_evidence']['draws'][0]['conditioning_metrics'] = dict(
            counter_scope='ordinary_joint_rejection', proposals_started=4,
            materialized_anchors=3, proposals_rejected=2, accepted_worlds=1,
            history_steps_started=5, history_steps_completed=4)
        self.assertEqual(benchmark.measured_trial(row)[
            'observed_ordinary_world_yield_per_started_proposal'], .25)

    def test_work_and_wall_accounting_drift_refuse(self):
        row = self.trial()
        row['worker_evidence']['search_phase_timing'][0]['attempted_world_draws'] = 2
        with self.assertRaises(ValueError):
            benchmark.measured_trial(row)
        row = self.trial()
        row['wall_search_seconds'] = 2
        with self.assertRaises(ValueError):
            benchmark.measured_trial(row)

    def test_every_trial_including_failures_is_required_and_paired(self):
        plan = dict(cases=benchmark.fixed_cases(self.manifest()), replicates=3,
            modes=['baseline_instrumented', benchmark.REPAIR_MODE])
        rows = [self.trial(case['name'], mode, replicate,
                          completed=0 if mode == 'baseline_instrumented' else 3)
                for case in plan['cases'] for mode in plan['modes'] for replicate in range(3)]
        out = benchmark.comparison_readout(plan, rows)
        self.assertEqual(len(out['paired_contrasts']), 12)
        self.assertEqual(len(out['measured_trials']), 24)
        self.assertTrue(all(r['completed_trajectories_repair_minus_baseline'] == 3
                            for r in out['paired_contrasts']))
        self.assertTrue(all(r['median_completed_trajectories'] == 0
            for r in out['fixed_case_groups'] if r['mode'] == 'baseline_instrumented'))
        for variant in (rows[:-1], rows + [rows[0]]):
            with self.assertRaises(ValueError):
                benchmark.comparison_readout(plan, variant)
        rows[0]['seed'] += 1
        with self.assertRaises(ValueError):
            benchmark.comparison_readout(plan, rows)



if __name__ == '__main__':
    unittest.main()
