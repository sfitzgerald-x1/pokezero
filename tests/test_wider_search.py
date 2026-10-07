from types import SimpleNamespace
import importlib.util
import gzip
import itertools
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from pokezero.mcts_eval.wider_search import ARMS, SEATS, analyze, exact_signflip_p, game_identity, play_game, study_seed
from pokezero.mcts_eval.paper_reference import Evaluation


def rows(seeds, reference=1, incumbent=-1):
    return [dict(identity=game_identity(seed, seat, arm), seed=seed, subject=seat, arm=arm,
        status="COMPLETE", signed_outcome=reference if arm == ARMS[1] else incumbent)
        for seed in seeds for arm in ARMS for seat in SEATS]


class WiderSearchTests(unittest.TestCase):
    def test_independent_roster_does_not_select_disagreements_or_overlap_qualification(self):
        confirmation = {study_seed(i) for i in range(64)}
        self.assertEqual(len(confirmation), 64)
        self.assertFalse(confirmation & {study_seed(i, qualification=True) for i in range(2)})

    def test_exact_cluster_p_matches_brute_force(self):
        values = [1., .5, -.25, 0., -1.]
        observed = abs(sum(values))
        brute = sum(abs(sum(a*b for a, b in zip(signs, values))) >= observed
            for signs in itertools.product((-1, 1), repeat=len(values))) / 2**len(values)
        self.assertEqual(exact_signflip_p(values), brute)

    def test_seats_not_counted_as_independent_clusters(self):
        result = analyze(list(range(6)), rows(list(range(6))))
        self.assertEqual(result["complete_seed_clusters"], 6)
        self.assertEqual(result["registered_games"], 24)
        self.assertEqual(result["exact_seed_cluster_signflip_p"], 2 / 2**6)
        # Six favorable clusters still have too-wide a distribution-free mean interval.
        self.assertFalse(result["statistically_supported_advantage"])

    def test_null_and_strong_alternative(self):
        seeds = list(range(64))
        self.assertFalse(analyze(seeds, rows(seeds, 1, 1))["statistically_supported_advantage"])
        self.assertTrue(analyze(seeds, rows(seeds))["statistically_supported_advantage"])

    def test_missing_and_refused_never_scored_or_complete_case_tested(self):
        seeds = [1, 2]
        data = rows(seeds)
        data[0].update(status="REFUSED", signed_outcome=None)
        result = analyze(seeds, data[:-1])
        self.assertEqual(result["complete_seed_clusters"], 0)
        self.assertEqual(result["mean_delta_worst_case_bounds"], [-1., 1.])
        self.assertFalse(result["inferential_test_allowed"])
        data[0]["signed_outcome"] = -1
        with self.assertRaisesRegex(ValueError, "never be scored"):
            analyze(seeds, data)

    def test_duplicate_and_extra_roster_rejected(self):
        with self.assertRaisesRegex(ValueError, "duplicate"):
            analyze([1], rows([1]) + rows([1])[:1])
        with self.assertRaisesRegex(ValueError, "extra"):
            analyze([1], rows([1, 2]))

    def test_searches_first_request_and_all_later_requests(self):
        class Env:
            boundary = 0
            def terminal(self):
                return None if self.boundary < 3 else SimpleNamespace(capped=False, winner="p1")
            def requested_players(self):
                return ("p1", "p2") if self.boundary != 1 else ("p2",)
            def observe(self, actor):
                return SimpleNamespace(actor=actor, legal_action_mask=(True, True))
            def reseed_simulator_rng(self, seed):
                pass
            def step(self, actions):
                self.boundary += 1
        calls = []
        def search(observation, boundary, seed):
            self.assertEqual(observation.actor, "p1")
            calls.append(boundary)
            return 1, {"selector": "search"}
        result = play_game(Env(), subject="p1", decision_id="independent", selector=search,
            opponent=lambda obs: ((0, 1), Evaluation((.5, .5), 0)), emit=lambda row: None,
            max_boundaries=10, wall_seconds=60)
        self.assertEqual(calls, [0, 2])
        self.assertEqual(result["status"], "COMPLETE")

    def test_refusal_does_not_trigger_raw_takeover(self):
        env = SimpleNamespace(terminal=lambda: None, requested_players=lambda: ("p1",),
            observe=lambda actor: SimpleNamespace(legal_action_mask=(True,)))
        def refuse(*args):
            raise RuntimeError("search refused")
        with self.assertRaisesRegex(RuntimeError, "search refused"):
            play_game(env, subject="p1", decision_id="test", selector=refuse, opponent=None,
                emit=lambda row: None, max_boundaries=10, wall_seconds=60)

    def test_replayed_prefix_keeps_boundary_rng_and_selection_counts(self):
        class Env:
            boundary = 2
            def terminal(self):
                return None if self.boundary == 2 else SimpleNamespace(capped=False, winner='p1')
            def requested_players(self):
                return ('p1',)
            def observe(self, actor):
                return SimpleNamespace(legal_action_mask=(True,))
            def reseed_simulator_rng(self, seed):
                self.chance = seed
            def step(self, actions):
                self.boundary += 1
        from pokezero.mcts_eval.followthrough import continuation_seed
        emitted, calls = [], []
        def select(observation, boundary, seed):
            calls.append((boundary, seed))
            return 0, {'selector': 'search'}
        result = play_game(Env(), subject='p1', decision_id='same', selector=select,
            opponent=None, emit=emitted.append, max_boundaries=10, wall_seconds=60,
            start_boundary=2, prior_selections=1, elapsed_before_resume=12.)
        self.assertEqual(calls, [(2, continuation_seed('same', 0, 2, 'search'))])
        self.assertEqual(emitted[0]['chance_seed'], continuation_seed('same', 0, 2, 'chance'))
        self.assertEqual(result['boundaries'], 3)
        self.assertEqual(result['own_decisions'], 2)
        self.assertGreaterEqual(result['elapsed_seconds'], 12.)

    def test_recovery_cannot_reset_whole_game_caps(self):
        env = SimpleNamespace(terminal=lambda: None)
        result = play_game(env, subject='p1', decision_id='same', selector=None,
            opponent=None, emit=None, max_boundaries=10, wall_seconds=60,
            start_boundary=2, prior_selections=1, elapsed_before_resume=61.)
        self.assertEqual(result['status'], 'CAPPED')
        self.assertIsNone(result['signed_outcome'])
        self.assertEqual(result['boundaries'], 2)

    def test_invalid_retained_prefix_positions_refuse(self):
        for start, count, elapsed in ((-1, 0, 0.), (11, 1, 0.), (2, 3, 0.),
                                      (2, True, 0.), (0, 1, 0.), (2, 1, float('nan'))):
            with self.subTest(start=start, count=count, elapsed=elapsed):
                with self.assertRaisesRegex(ValueError, 'prefix'):
                    play_game(None, subject='p1', decision_id='same', selector=None,
                        opponent=None, emit=None, max_boundaries=10, wall_seconds=60,
                        start_boundary=start, prior_selections=count, elapsed_before_resume=elapsed)


class QualificationEvidenceTests(unittest.TestCase):
    def setUp(self):
        spec = importlib.util.spec_from_file_location('wider_driver',
            Path(__file__).resolve().parents[1]/'scripts/wider_search_comparison.py')
        self.driver = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.driver)
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.seeds = [study_seed(i, qualification=True) for i in range(2)]
        self.registration = dict(phase='QUALIFICATION_NOT_STRENGTH', seeds=self.seeds,
            registered_games=8, source_commit='pinned', input_hashes={'checkpoint': 'pinned'})
        self.write('registration.json', self.registration)
        for row in rows(self.seeds):
            cell = self.root/row['identity']
            cell.mkdir()
            self.write(row['identity']+'/boundary-000.json', {'actions': {'p1': 0, 'p2': 0}})
            row.update(registration_sha256=self.driver.sha(self.root/'registration.json'),
                step_hashes={'boundary-000.json': self.driver.sha(cell/'boundary-000.json')})
            self.write(row['identity']+'.json', row)
        result = analyze(self.seeds, self.driver.bound_rows(self.root, self.registration))
        result.update(source_commit='pinned', input_hashes=self.registration['input_hashes'],
            phase=self.registration['phase'], statistically_supported_advantage=False,
            inferential_test_allowed=False, status='QUALIFICATION_COMPLETE_NOT_STRENGTH')
        self.write('READOUT.json', result)

    def write(self, name, value):
        (self.root/name).write_text(json.dumps(value))

    def validate(self):
        with patch.object(self.driver, 'verify'):
            return self.driver.validate_qualification(self.root/'READOUT.json', self.registration)

    def test_canonical_qualification_rederives_and_retains_readout_hash(self):
        self.assertEqual(self.validate()['readout_sha256'], self.driver.sha(self.root/'READOUT.json'))

    def test_default_off_qualification_cannot_qualify_an_opt_in_kernel(self):
        confirmation = dict(self.registration, reference=dict(staged_substitute_conditioning=True))
        with patch.object(self.driver, 'verify'), self.assertRaisesRegex(RuntimeError, 'roster or source'):
            self.driver.validate_qualification(self.root/'READOUT.json', confirmation)

    def test_full_statistics_compress_losslessly_and_never_overwrite(self):
        value = {'statistics_checkpoint': {'Q': {'node': .33333}, 'N': {'node': 7},
            'M': {'a': 4}, 'F': ['trajectory']}, 'receipts': [1, 2, 3]}
        path = self.root/'retained.json.gz'
        self.driver.save_step(path, value)
        self.assertEqual(json.loads(gzip.decompress(path.read_bytes())), value)
        with self.assertRaises(FileExistsError):
            self.driver.save_step(path, value)

    def test_changed_boundary_evidence_cannot_hide_behind_complete_readout(self):
        identity = game_identity(self.seeds[0], 'p1', ARMS[0])
        self.write(identity+'/boundary-000.json', {'actions': {'p1': 1, 'p2': 0}})
        with self.assertRaisesRegex(RuntimeError, 'durable decision evidence drift'):
            self.validate()

    def test_readout_label_alone_cannot_qualify_missing_games(self):
        identity = game_identity(self.seeds[0], 'p1', ARMS[0])
        (self.root/(identity+'.json')).unlink()
        with self.assertRaisesRegex(RuntimeError, 'qualification incomplete'):
            self.validate()

    def test_registered_prefix_combines_losslessly_without_copying_or_duplicate_boundaries(self):
        from pokezero.mcts_eval.wider_recovery import combined_steps
        identity = 'resumed'
        (self.root/identity).mkdir()
        self.write('prefix.json', {'boundary': 0})
        self.write(identity+'/boundary-001.json', {'boundary': 1})
        prefix = {'boundary-000.json': str(self.root/'prefix.json')}
        # Filenames remain part of the invariant, so use a separate parent.
        retained = self.root/'retained'
        retained.mkdir()
        path = retained/'boundary-000.json'
        path.write_text(json.dumps({'boundary': 0}))
        prefix = {path.name: str(path)}
        m = dict(retained_input_hashes={str(path): self.driver.sha(path)},
            repair_retention={'resume': {'identity': identity, 'prefix_steps': prefix}})
        cell = {'identity': identity, 'retained_prefix_steps': prefix}
        paths = combined_steps(self.root, cell, m, sha=self.driver.sha, step_files=self.driver.step_files)
        self.assertEqual([p.name for p in paths], ['boundary-000.json', 'boundary-001.json'])
        self.assertFalse((self.root/identity/path.name).exists())
        self.write(identity+'/boundary-000.json', {'boundary': 0})
        with self.assertRaisesRegex(RuntimeError, 'duplicate'):
            combined_steps(self.root, cell, m, sha=self.driver.sha, step_files=self.driver.step_files)

    def test_prefix_registration_or_hash_drift_cannot_be_hidden(self):
        from pokezero.mcts_eval.wider_recovery import combined_steps
        (self.root/'resumed').mkdir()
        self.write('boundary-000.json', {'boundary': 0})
        path = self.root/'boundary-000.json'
        prefix = {path.name: str(path)}
        cell = {'identity': 'resumed', 'retained_prefix_steps': prefix}
        with self.assertRaisesRegex(RuntimeError, 'unregistered'):
            combined_steps(self.root, cell, {}, sha=self.driver.sha, step_files=self.driver.step_files)
        m = dict(retained_input_hashes={str(path): 'wrong'},
            repair_retention={'resume': {'identity': 'resumed', 'prefix_steps': prefix}})
        with self.assertRaisesRegex(RuntimeError, 'drift'):
            combined_steps(self.root, cell, m, sha=self.driver.sha, step_files=self.driver.step_files)


class RecoverySensitivityTests(unittest.TestCase):
    def apply(self, data):
        from pokezero.mcts_eval.wider_recovery import add_recovery_sensitivity
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'retained-refusal.json'
            path.write_text(json.dumps({'seed': 0}))
            registration = dict(seeds=list(range(64)),
                repair_retention={'resume': {'result_path': str(path)}})
            result = analyze(registration['seeds'], data)
            add_recovery_sensitivity(result, registration, sha=lambda path: 'retained-hash')
            return result

    def test_incomplete_roster_cannot_enter_recovery_inference(self):
        result = self.apply(rows(list(range(64)))[:-1])
        self.assertFalse(result['inferential_test_allowed'])
        self.assertEqual(result['recovery_sensitivity']['status'], 'INCOMPLETE_NO_INFERENCE')

    def test_strong_advantage_survives_every_recovered_seed_score(self):
        result = self.apply(rows(list(range(64))))
        sensitivity = result['recovery_sensitivity']
        self.assertTrue(result['statistically_supported_advantage'])
        self.assertTrue(sensitivity['all_scores_support_advantage'])
        self.assertEqual([row['contrast'] for row in sensitivity['possible_recovered_seed_contrasts']],
                         [quarter/4 for quarter in range(-4, 5)])
        self.assertAlmostEqual(sensitivity['minimum_bounded_mean_lower'],
            result['bounded_mean_confidence_interval'][0]-2/64)

    def test_marginal_advantage_cannot_depend_on_recovered_cluster(self):
        data = rows(list(range(64)))
        for row in data:
            if row['seed'] >= 22:
                row['signed_outcome'] = 1
        result = self.apply(data)
        self.assertTrue(result['preregistered_full_roster_advantage'])
        self.assertFalse(result['statistically_supported_advantage'])
        self.assertLess(result['recovery_sensitivity']['minimum_bounded_mean_lower'], 0.)

    def test_null_is_not_equivalence_or_advantage_after_sensitivity(self):
        result = self.apply(rows(list(range(64)), 1, 1))
        self.assertFalse(result['statistically_supported_advantage'])
        self.assertFalse(result['recovery_sensitivity']['all_scores_support_advantage'])
