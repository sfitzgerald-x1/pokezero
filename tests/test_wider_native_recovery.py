"""Adversarial pins for native repair retention and all-historical-score bounds."""
import itertools
from collections import Counter
import gzip
import hashlib
import math
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock
from types import SimpleNamespace

from pokezero.mcts_eval.wider_native_recovery import (
    add_sensitivity, bind_qualified_source, uncertainty_seeds, worst_case_statistics,
    validate_native_qualification,
)
from pokezero.mcts_eval.wider_search import game_identity, study_seed
from pokezero.mcts_eval.followthrough import continuation_seed


def independent_signflip(values):
    """Binomial grouped arithmetic, independent of the production DP/test."""
    quarters = [round(4*v) for v in values]
    # Each equal-magnitude group has binomial coefficients. Zero-valued signs
    # cancel in the probability, so do not put them into the denominator.
    groups = Counter(abs(v) for v in quarters if v)
    distribution = {0: 1}
    for magnitude, count in groups.items():
        changed = Counter()
        for positives in range(count+1):
            total = magnitude*(2*positives-count)
            ways = math.comb(count, positives)
            for previous, previous_ways in distribution.items():
                changed[previous+total] += previous_ways*ways
        distribution = changed
    observed = abs(sum(quarters))
    return sum(ways for total, ways in distribution.items() if abs(total) >= observed)/(2**sum(groups.values()))


def brute(values, indices):
    cases = []
    for assignment in itertools.product(range(-4, 5), repeat=len(indices)):
        changed = list(values)
        for i, quarter in zip(indices, assignment):
            changed[i] = quarter/4
        mean = sum(changed)/len(changed)
        cases.append((independent_signflip(changed), max(-1., mean-math.sqrt(2*math.log(40)/len(changed))), mean))
    return max(c[0] for c in cases), min(c[1] for c in cases), min(c[2] for c in cases)


class NativeRepairSensitivityTests(unittest.TestCase):
    def test_independent_grouped_arithmetic_matches_literal_sign_enumeration(self):
        for values in itertools.product((-.5, 0., .25), repeat=5):
            threshold = abs(sum(values))
            direct = sum(abs(sum(sign*v for sign, v in zip(signs, values))) >= threshold
                         for signs in itertools.product((-1, 1), repeat=len(values)))/(2**len(values))
            self.assertEqual(independent_signflip(values), direct)

    def test_independent_full_64_cluster_counterfactual_arithmetic(self):
        # Include ties, negative differences and all four nonzero magnitudes.
        values = [0., -.25, .5, 1.] + [.5, 1., .25, 0., -.75]*12
        result = worst_case_statistics(values, range(4))
        cases = []
        for assignment in itertools.combinations_with_replacement(range(-4, 5), 4):
            changed = [v/4 for v in assignment] + values[4:]
            mean = sum(changed)/64
            cases.append((independent_signflip(changed),
                          max(-1., mean-math.sqrt(2*math.log(40)/64)), mean))
        self.assertEqual(result['maximum_exact_p'], max(c[0] for c in cases))
        self.assertEqual(result['minimum_bounded_mean_lower'], min(c[1] for c in cases))
        self.assertEqual(result['minimum_mean_win_score_delta'], min(c[2] for c in cases))

    def test_multiset_algorithm_equals_every_ordered_counterfactual(self):
        for values, indices in (([.5, 0., -.25], [0]),
                                ([.5, 0., -.25, 1.], [0, 2]),
                                ([.5, -.5, .25, 0., 1.], [0, 1, 3]),
                                ([.5, -.5, .25, 0., 1.], [0, 1, 2, 3])):
            with self.subTest(indices=indices):
                result = worst_case_statistics(values, indices)
                p, lower, delta = brute(values, indices)
                self.assertEqual(result['maximum_exact_p'], p)
                self.assertEqual(result['minimum_bounded_mean_lower'], lower)
                self.assertEqual(result['minimum_mean_win_score_delta'], delta)
                self.assertEqual(result['ordered_counterfactuals_covered'], 9**len(indices))

    def test_four_uncertain_clusters_cover_all_6561_not_only_extreme_scores(self):
        result = worst_case_statistics([.5]*64, [0, 1, 2, 3])
        self.assertEqual(result['distinct_counterfactual_multisets'], 495)
        self.assertEqual(result['ordered_counterfactuals_covered'], 6561)
        self.assertTrue(result['all_scores_support_advantage'])

    def test_a_modest_effect_cannot_be_promoted_by_green_operational_tests(self):
        result = worst_case_statistics([.25]*64, [0, 1, 2, 3])
        self.assertLess(result['maximum_exact_p'], .05)
        self.assertLessEqual(result['minimum_bounded_mean_lower'], 0)
        self.assertFalse(result['all_scores_support_advantage'])

    def test_original_historical_scores_do_not_select_the_sensitivity_result(self):
        a = worst_case_statistics([1., 1., 1., 1.] + [.5]*60, range(4))
        b = worst_case_statistics([-1., -.5, .25, 0.] + [.5]*60, range(4))
        self.assertEqual(a, b)

    def test_direction_is_not_lost_in_a_two_sided_p_value(self):
        result = worst_case_statistics([-.5]*64, range(4))
        self.assertLess(result['maximum_exact_p'], .05)
        self.assertFalse(result['all_scores_support_advantage'])

    def test_invalid_score_or_roster_is_refused(self):
        for values, indices in (([.3], [0]), ([float('nan')], [0]),
                                ([float('inf')], [0]), ([2.], [0]),
                                ([0.]*5, range(5)), ([0.], [1]), ([0.], []), ([0.], [0,0])):
            with self.subTest(values=values, indices=indices):
                with self.assertRaises(RuntimeError):
                    worst_case_statistics(values, indices)

    def test_all_touched_seeds_are_uncertain_even_if_complete_or_not_showing_struggle(self):
        rows = [dict(seed=10, status='COMPLETE', signed_outcome=1),
                dict(seed=30, status='REFUSED', signed_outcome=None),
                dict(seed=20, status='COMPLETE', signed_outcome=-1)]
        self.assertEqual(uncertainty_seeds(rows, {'seeds':[10,20,30,40]}), [10,20,30])
        rows[0]['signed_outcome'] = -1
        self.assertEqual(uncertainty_seeds(rows, {'seeds':[10,20,30,40]}), [10,20,30])

    def test_unregistered_historical_seed_is_not_silently_dropped(self):
        with self.assertRaises(RuntimeError):
            uncertainty_seeds([dict(seed=42)], {'seeds':[1,2]})

    def test_partial_results_never_get_a_significance_claim(self):
        result = dict(inferential_test_allowed=False, statistically_supported_advantage=False)
        registration = dict(seeds=list(range(64)), repair_retention=dict(uncertain_seed_clusters=[0,1,2,3]))
        add_sensitivity(result, registration)
        self.assertEqual(result['native_repair_sensitivity']['status'], 'INCOMPLETE_NO_INFERENCE')
        self.assertFalse(result['statistically_supported_advantage'])

    def test_full_roster_rule_is_required_even_if_sensitivity_passes(self):
        result = dict(inferential_test_allowed=True, statistically_supported_advantage=False,
                      contrasts=[dict(seed=i,difference=.5) for i in range(64)])
        registration = dict(seeds=list(range(64)), repair_retention=dict(uncertain_seed_clusters=[0,1,2,3]))
        add_sensitivity(result, registration)
        self.assertTrue(result['native_repair_sensitivity']['all_scores_support_advantage'])
        self.assertFalse(result['statistically_supported_advantage'])
        self.assertEqual(result['native_repair_sensitivity']['untouched_seed_conditional_evidence']['seed_clusters'],60)

    def test_full_roster_order_or_duplicate_uncertainty_cannot_drift(self):
        registration = dict(seeds=list(range(64)), repair_retention=dict(uncertain_seed_clusters=[0,0]))
        result = dict(inferential_test_allowed=True, statistically_supported_advantage=True,
                      contrasts=[dict(seed=i,difference=.5) for i in range(64)])
        with self.assertRaises(RuntimeError):
            add_sensitivity(result, registration)
        registration['repair_retention']['uncertain_seed_clusters'] = [0]
        result['contrasts'].reverse()
        with self.assertRaises(RuntimeError):
            add_sensitivity(result, registration)


class QualifiedSourceBindingTests(unittest.TestCase):
    def setUp(self):
        from pokezero.mcts_eval.wider_native_recovery import CONTRACT_KEYS
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.origin, self.current = Path(self.temporary.name)/'old', Path(self.temporary.name)/'new'
        for root in (self.origin, self.current):
            (root/'src').mkdir(parents=True)
            (root/'scripts').mkdir()
            (root/'src/policy.py').write_text('real search contract')
            (root/'scripts/wider_search_comparison.py').write_text(str(root))
        self.qualification = {key: 'unchanged' for key in CONTRACT_KEYS}
        self.qualification.update(source_root=str(self.origin), native_package='immutable-native')
        self.confirmation = dict(self.qualification, source_root=str(self.current))
        self.names = ['src/policy.py', 'scripts/wider_search_comparison.py']

    def bind(self):
        return bind_qualified_source(self.qualification, self.confirmation,
            repo=self.current, git=lambda *a, **k: '\n'.join(self.names),
            sha=lambda p: Path(p).read_text())

    def test_only_disclosed_operational_changes_are_admitted(self):
        self.assertEqual(self.bind(), {str(self.origin/'src/policy.py'): 'real search contract'})

    def test_a_search_policy_change_cannot_hide_behind_new_source_identity(self):
        (self.current/'src/policy.py').write_text('unqualified search change')
        with self.assertRaisesRegex(RuntimeError, 'unqualified semantic change'):
            self.bind()

    def test_new_or_missing_semantic_file_is_not_ignored(self):
        self.names.append('src/new_policy.py')
        (self.current/'src/new_policy.py').write_text('new policy')
        with self.assertRaisesRegex(RuntimeError, 'unqualified semantic change'):
            self.bind()

    def test_deadline_or_checkpoint_change_requires_new_qualification(self):
        for key in ('checkpoint_sha256', 'nominal_decision_seconds', 'incumbent', 'reference'):
            with self.subTest(key=key):
                self.confirmation[key] = 'drift'
                with self.assertRaisesRegex(RuntimeError, 'qualified treatment drift'):
                    self.bind()
                self.confirmation[key] = self.qualification[key]

    def test_loaded_native_package_cannot_differ_from_qualified_package(self):
        self.confirmation['native_package'] = 'different-native'
        with self.assertRaisesRegex(RuntimeError, 'different native binary'):
            self.bind()


class HistoricalRuntimeVerificationTests(unittest.TestCase):
    def test_historical_verifier_uses_historical_source_and_package_not_new_native(self):
        from pokezero.mcts_eval.wider_native_recovery import verify_historical
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)/'registration.json'
            manifest = dict(source_root='/original/source', native_package='/original/native')
            path.write_text(json.dumps(manifest))
            with mock.patch('pokezero.mcts_eval.wider_native_recovery.subprocess.run') as execute:
                self.assertEqual(verify_historical(path), manifest)
            call = execute.call_args
            self.assertTrue(call.kwargs['check'])
            self.assertEqual(call.kwargs['cwd'], Path('/original/source'))
            self.assertEqual(call.kwargs['env']['PYTHONPATH'],
                             '/original/native:/original/source/src:/original/source/scripts')
            self.assertIn('verify(m,source_root=', call.args[0][2])

    def test_historical_verification_failure_is_not_bypassed(self):
        import subprocess
        from pokezero.mcts_eval.wider_native_recovery import verify_historical
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)/'registration.json'
            path.write_text(json.dumps(dict(source_root='/original/source', native_package='/original/native')))
            with mock.patch('pokezero.mcts_eval.wider_native_recovery.subprocess.run',
                            side_effect=subprocess.CalledProcessError(1, 'verification')):
                with self.assertRaises(subprocess.CalledProcessError):
                    verify_historical(path)


class NewNativeQualificationTests(unittest.TestCase):
    def setUp(self):
        from pokezero.mcts_eval.wider_search import analyze
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.path = self.root/'READOUT.json'
        self.registration = dict(phase='QUALIFICATION_NOT_STRENGTH', source_commit='qualified',
            source_root='/qualified/source', input_hashes={}, registered_games=8,
            seeds=[study_seed(i, qualification=True) for i in range(2)])
        self.rows = [dict(seed=seed, subject=seat, arm=arm, status='COMPLETE', signed_outcome=0,
                         identity=game_identity(seed, seat, arm))
                     for seed in self.registration['seeds'] for seat in ('p1','p2')
                     for arm in ('deep_incumbent','paper_reference')]
        self.readout = analyze(self.registration['seeds'], self.rows)
        self.readout.update(source_commit='qualified', input_hashes={},
            phase='QUALIFICATION_NOT_STRENGTH', statistically_supported_advantage=False,
            inferential_test_allowed=False, status='QUALIFICATION_COMPLETE_NOT_STRENGTH')
        self.verify = mock.Mock()

    def validate(self):
        (self.root/'registration.json').write_text(json.dumps(self.registration))
        self.path.write_text(json.dumps(self.readout))
        with mock.patch('pokezero.mcts_eval.wider_native_recovery.bind_qualified_source', return_value={}):
            return validate_native_qualification(self.path, {}, repo=self.root, git=mock.Mock(),
                sha=NativeRecoveryPreparationTests.sha, verify=self.verify,
                bound_rows=lambda *args: self.rows)

    def test_all_eight_durable_games_rederive_exact_qualification(self):
        result = self.validate()
        self.assertEqual(result['qualified_source_commit'], 'qualified')
        self.verify.assert_called_once_with(self.registration, source_root=Path('/qualified/source'))

    def test_forged_complete_label_does_not_qualify_partial_games(self):
        self.rows.pop()
        with self.assertRaisesRegex(RuntimeError, 'qualification incomplete'):
            self.validate()

    def test_refused_cell_never_qualifies_as_loss(self):
        self.rows[0].update(status='REFUSED', signed_outcome=None)
        with self.assertRaisesRegex(RuntimeError, 'qualification incomplete'):
            self.validate()

    def test_selected_or_confirmation_seeds_cannot_replace_qualification(self):
        self.registration['seeds'] = [study_seed(i) for i in range(2)]
        with self.assertRaisesRegex(RuntimeError, 'qualification roster drift'):
            self.validate()

    def test_changed_readout_arithmetic_does_not_qualify(self):
        self.readout['contrasts'][0]['difference'] = 1.
        with self.assertRaisesRegex(RuntimeError, 'readout differs'):
            self.validate()


class QualificationSemanticAuditBindingTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.study = self.root/'study'
        self.study.mkdir()
        self.audit_path = self.root/'qualification-audit.json'
        self.observer = self.root/'validate_games.py'
        self.observer.write_text('fixed observer')
        self.digest = NativeRecoveryPreparationTests.sha(self.observer)
        self.registration = dict(source_commit='qualified', seeds=[study_seed(i, qualification=True) for i in range(2)])
        (self.study/'registration.json').write_text(json.dumps(self.registration))
        self.audit = dict(schema='pokezero.wider-search.read-only-audit.v1',
            observer_sha256=self.digest, source_commit='qualified',
            registration_sha256=NativeRecoveryPreparationTests.sha(self.study/'registration.json'),
            phase='QUALIFICATION_NOT_STRENGTH', registered_games=8, audited_complete_games=8,
            complete_roster_valid=True, strength_inference=False, games=[])
        for seed in self.registration['seeds']:
            for seat in ('p1','p2'):
                for arm in ('deep_incumbent','paper_reference'):
                    identity = game_identity(seed, seat, arm)
                    path = self.study/(identity+'.json')
                    (self.study/identity).mkdir()
                    step = self.study/identity/'boundary-000.json.gz'
                    step.write_bytes(b'bound immutable step')
                    path.write_text(json.dumps(dict(identity=identity, status='COMPLETE',
                        registration_sha256=NativeRecoveryPreparationTests.sha(self.study/'registration.json'),
                        step_hashes={step.name: NativeRecoveryPreparationTests.sha(step)})))
                    self.audit['games'].append(dict(identity=identity,
                        result_sha256=NativeRecoveryPreparationTests.sha(path),
                        status='HASHES_REQUESTS_OPPONENT_POLICY_SEARCH_WITNESSES_AND_TERMINAL_REPLAY_VALID'))

    def bind(self):
        from pokezero.mcts_eval.wider_native_recovery import bind_qualification_audit
        self.audit_path.write_text(json.dumps(self.audit))
        with mock.patch('pokezero.mcts_eval.wider_native_recovery.QUALIFICATION_OBSERVER_HASH', self.digest):
            return bind_qualification_audit(self.audit_path, self.study/'READOUT.json',
                                           sha=NativeRecoveryPreparationTests.sha)

    def test_complete_replay_audit_binds_all_eight_result_hashes(self):
        inputs = self.bind()
        self.assertEqual(len(inputs), 18)
        self.assertIn(str(self.observer), inputs)
        self.assertIn(str(self.audit_path), inputs)

    def test_no_audit_is_not_allowed(self):
        from pokezero.mcts_eval.wider_native_recovery import bind_qualification_audit
        with self.assertRaisesRegex(RuntimeError, 'needs complete semantic replay'):
            bind_qualification_audit(None, self.study/'READOUT.json', sha=NativeRecoveryPreparationTests.sha)

    def test_partial_audit_or_wrong_source_cannot_qualify(self):
        for key, invalid in (('audited_complete_games',3), ('complete_roster_valid',False),
                             ('source_commit','wrong'), ('registration_sha256','wrong'),
                             ('observer_sha256','wrong'), ('strength_inference',True)):
            original = self.audit[key]
            self.audit[key] = invalid
            with self.subTest(key=key), self.assertRaisesRegex(RuntimeError, 'audit binding drift'):
                self.bind()
            self.audit[key] = original

    def test_a_duplicate_game_cannot_hide_a_missing_cell(self):
        self.audit['games'][0] = dict(self.audit['games'][1])
        with self.assertRaisesRegex(RuntimeError, 'audit roster drift'):
            self.bind()

    def test_a_changed_result_or_incomplete_replay_cannot_qualify(self):
        self.audit['games'][0]['result_sha256'] = 'wrong'
        with self.assertRaisesRegex(RuntimeError, 'semantic result drift'):
            self.bind()

    def test_qualification_decision_evidence_is_pinned_not_only_terminal_files(self):
        identity = self.audit['games'][0]['identity']
        step = self.study/identity/'boundary-000.json.gz'
        self.assertIn(str(step), self.bind())
        step.write_bytes(b'changed after semantic replay')
        with self.assertRaisesRegex(RuntimeError, 'semantic decision evidence drift'):
            self.bind()


class NativeRecoveryPreparationTests(unittest.TestCase):
    """The recovery adapter must refuse unqualified or unaudited retention."""
    def setUp(self):
        from pokezero.mcts_eval.wider_native_recovery import CONTRACT_KEYS
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.previous = self.root/'previous'
        self.legacy = self.root/'legacy'
        self.qualified = self.root/'qualified'
        for root in (self.previous, self.legacy, self.qualified):
            root.mkdir()
        self.old = {key: 'fixed' for key in CONTRACT_KEYS}
        self.old.update(phase='FIXED_64_SEED_CONFIRMATION', source_commit='historical-source',
            seeds=[study_seed(i) for i in range(64)], registered_games=256,
            max_boundaries=200, per_game_wall_seconds=2400.)
        self.current = dict(self.old)
        self.original = self.previous/'registration.json'
        self.complete_id = game_identity(self.old['seeds'][0], 'p1', 'paper_reference')
        self.refused_id = game_identity(self.old['seeds'][1], 'p2', 'deep_incumbent')
        self.complete_path = self.legacy/(self.complete_id+'.json')
        self.refused_path = self.previous/(self.refused_id+'.json')
        self.write(self.complete_path, dict(status='COMPLETE'))
        self.write(self.legacy/'registration.json', dict(source_commit='legacy-source'))
        self.old['repair_retention'] = dict(retained_complete={self.complete_id: dict(
            path=str(self.complete_path), registration=str(self.legacy/'registration.json'))})
        self.write(self.original, self.old)
        self.write(self.refused_path, dict(status='REFUSED', signed_outcome=None))
        self.steps = {}
        for root, identity, seed, seat, arm in (
                (self.legacy, self.complete_id, self.old['seeds'][0], 'p1', 'paper_reference'),
                (self.previous, self.refused_id, self.old['seeds'][1], 'p2', 'deep_incumbent')):
            directory = root/identity
            directory.mkdir()
            path = directory/'boundary-000.json.gz'
            domain = f'wider-search:{seed}:{seat}'
            step = dict(boundary=0, chance_seed=continuation_seed(domain, 0, 0, 'chance'),
                actions={seat: 0}, evidence={seat: dict(selector=arm,
                    seed=continuation_seed(domain, 0, 0, 'search'), elapsed_seconds=10.)})
            path.write_bytes(gzip.compress(json.dumps(step).encode()))
            self.steps[identity] = path
        self.rows = [dict(identity=self.complete_id, seed=self.old['seeds'][0], subject='p1',
            arm='paper_reference', status='COMPLETE', signed_outcome=1,
            step_hashes={self.steps[self.complete_id].name: self.sha(self.steps[self.complete_id])}),
            dict(identity=self.refused_id, seed=self.old['seeds'][1], subject='p2',
            arm='deep_incumbent', status='REFUSED', signed_outcome=None,
            step_hashes={self.steps[self.refused_id].name: self.sha(self.steps[self.refused_id])})]
        self.write(self.qualified/'registration.json', dict(source_commit='qualified-source'))
        self.write(self.qualified/'READOUT.json', dict(status='QUALIFICATION_COMPLETE_NOT_STRENGTH'))
        self.probe_path = self.root/'probe-incumbent-struggle-repair-r2.json'
        self.probe = dict(validation='VALID', fresh_source_commit='qualified-source',
            fresh_source_status='', original_registration_sha256=self.sha(self.original),
            fresh_source_and_binary_hashes={}, identity=self.refused_id,
            original_result_sha256=self.sha(self.refused_path),
            prefix_hashes=self.rows[1]['step_hashes'], boundary=1)
        self.certificate_path = self.root/'certificate.json'
        self.certificate = dict(status='EXACT_REFUSED_ROOT_VALID_WITH_REAL_NATIVE_STRUGGLE',
            strength_inference=False, source_commit='qualified-source', source_root='/qualified/source')
        self.write(self.root/'validate_recovered_games.py', 'observer')
        self.write(self.root/'validate_games.py', 'original observer')
        self.audit_path = self.root/'audit.json'
        self.audit = dict(schema='pokezero.wider-search.recovery-read-only-audit.v2',
            source_commit='historical-source', registration_sha256=self.sha(self.original),
            strength_inference=False, observer_sha256=self.sha(self.root/'validate_recovered_games.py'),
            original_observer_sha256=self.sha(self.root/'validate_games.py'), games=[dict(
                identity=self.complete_id, result_sha256=self.sha(self.complete_path),
                status='HASHES_REQUESTS_OPPONENT_POLICY_SEARCH_WITNESSES_AND_TERMINAL_REPLAY_VALID')])

    @staticmethod
    def write(path, data):
        path.write_text(json.dumps(data))

    @staticmethod
    def sha(path):
        return hashlib.sha256(Path(path).read_bytes()).hexdigest()

    def prepare(self):
        from pokezero.mcts_eval.wider_native_recovery import prepare
        self.write(self.probe_path, self.probe)
        self.write(self.certificate_path, dict(self.certificate,
            input_hashes={str(self.probe_path): self.sha(self.probe_path)}))
        self.write(self.audit_path, self.audit)
        real_stat = Path.stat
        def stat(path, *args, **kwargs):
            if path == self.refused_path.parent/self.refused_id:
                return SimpleNamespace(st_birthtime=100.)
            if path == self.refused_path:
                return SimpleNamespace(st_birthtime=120.)
            return real_stat(path, *args, **kwargs)
        with mock.patch('pokezero.mcts_eval.wider_native_recovery.verify_historical', return_value=self.old), \
             mock.patch('pokezero.mcts_eval.wider_native_recovery.validate_native_qualification',
                        return_value=dict(semantic_input_hashes={})), \
             mock.patch('pokezero.mcts_eval.wider_native_recovery.bind_qualification_audit', return_value={}), \
             mock.patch.object(Path, 'stat', stat):
            return prepare(self.previous, self.certificate_path, [self.audit_path],
                self.qualified/'READOUT.json', self.current, repo=self.root,
                git=lambda command, *args, **kwargs: 'qualified-source' if command == 'rev-parse' else '',
                sha=self.sha, verify=mock.Mock(), bound_rows=lambda *args: self.rows,
                step_files=lambda directory: [directory/'boundary-000.json.gz'])

    def test_complete_games_are_flattened_and_only_accepted_incumbent_prefix_resumes(self):
        result = self.prepare()
        binding = result['retained_complete'][self.complete_id]
        self.assertEqual(binding['path'], str(self.complete_path))
        self.assertEqual(binding['source_commit'], 'legacy-source')
        self.assertEqual(result['uncertain_seed_clusters'], self.old['seeds'][:2])
        self.assertEqual(result['resume']['identity'], self.refused_id)
        self.assertEqual(result['resume']['start_boundary'], 1)
        self.assertEqual(result['resume']['elapsed_before_resume'], 21.)
        self.assertIsNone(result['resume']['worker_ordinals'])
        self.assertIn(str(self.steps[self.complete_id]), result['input_hashes'])
        self.assertIn(str(self.steps[self.refused_id]), result['input_hashes'])

    def test_missing_semantic_audit_cannot_be_retained(self):
        self.audit['games'] = []
        with self.assertRaisesRegex(RuntimeError, 'missing semantic audit'):
            self.prepare()

    def test_old_result_hash_drift_cannot_be_retained(self):
        self.audit['games'][0]['result_sha256'] = 'wrong'
        with self.assertRaisesRegex(RuntimeError, 'missing semantic audit'):
            self.prepare()

    def test_native_certificate_cannot_claim_strength(self):
        self.certificate['strength_inference'] = True
        with self.assertRaisesRegex(RuntimeError, 'missing native repair certificate'):
            self.prepare()

    def test_probe_must_bind_exact_refused_result_and_registration(self):
        for key in ('original_result_sha256', 'original_registration_sha256', 'identity'):
            original = self.probe[key]
            self.probe[key] = 'drift'
            with self.subTest(key=key), self.assertRaises(RuntimeError):
                self.prepare()
            self.probe[key] = original

    def test_changed_native_binary_is_refused(self):
        self.probe['fresh_source_and_binary_hashes'] = {str(self.probe_path): 'wrong'}
        with self.assertRaisesRegex(RuntimeError, 'source/binary drift'):
            self.prepare()

    def test_failed_or_scored_cell_cannot_be_resumed(self):
        self.rows[1]['status'] = 'FAILED'
        with self.assertRaisesRegex(RuntimeError, 'unscored incumbent refusal'):
            self.prepare()
        self.rows[1]['status'] = 'REFUSED'
        self.rows[1]['signed_outcome'] = -1
        with self.assertRaisesRegex(RuntimeError, 'unscored incumbent refusal'):
            self.prepare()

    def test_game_cap_and_treatment_cannot_reset_on_recovery(self):
        self.current['per_game_wall_seconds'] = 5000.
        with self.assertRaisesRegex(RuntimeError, 'registered contract'):
            self.prepare()

    def test_prefix_chance_seed_cannot_drift_even_with_new_matching_hashes(self):
        path = self.steps[self.refused_id]
        step = json.loads(gzip.decompress(path.read_bytes()))
        step['chance_seed'] += 1
        path.write_bytes(gzip.compress(json.dumps(step).encode()))
        self.rows[1]['step_hashes'][path.name] = self.sha(path)
        with self.assertRaisesRegex(RuntimeError, 'chance-domain drift'):
            self.prepare()

if __name__ == '__main__':
    unittest.main()
