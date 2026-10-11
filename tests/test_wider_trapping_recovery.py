"""All five historical clusters remain uncertain, not a dropped fifth seed."""
import itertools
import math
import gzip
import hashlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

from pokezero.mcts_eval.followthrough import continuation_seed

from pokezero.mcts_eval.wider_native_recovery import worst_case_statistics
from pokezero.mcts_eval.wider_trapping_recovery import add_mixed_sensitivity, reference_resume
from test_wider_native_recovery import independent_signflip


class MixedSensitivityTests(unittest.TestCase):
    def test_all_five_clusters_cover_every_59049_ordered_assignment(self):
        values = [0.] * 5 + [.5] * 59
        result = worst_case_statistics(values, range(5), max_uncertain=5)
        self.assertEqual(result['ordered_counterfactuals_covered'], 59049)
        self.assertEqual(result['distinct_counterfactual_multisets'], 1287)
        # Separate grouped-binomial arithmetic, not the production DP.
        cases = []
        for assignment in itertools.combinations_with_replacement(range(-4, 5), 5):
            changed = [q / 4 for q in assignment] + values[5:]
            mean = sum(changed) / 64
            cases.append((independent_signflip(changed), max(-1., mean-math.sqrt(2*math.log(40)/64))))
        self.assertEqual(result['maximum_exact_p'], max(p for p, lower in cases))
        self.assertEqual(result['minimum_bounded_mean_lower'], min(lower for p, lower in cases))

    def test_old_scores_cannot_choose_the_bound_and_six_uncertain_seeds_refuse(self):
        a = worst_case_statistics([1.] * 5 + [.5] * 59, range(5), max_uncertain=5)
        b = worst_case_statistics([-1., 0., -.5, .25, .75] + [.5] * 59, range(5), max_uncertain=5)
        self.assertEqual(a, b)
        with self.assertRaises(RuntimeError):
            worst_case_statistics([0.] * 64, range(6), max_uncertain=5)

    def test_partial_or_original_nonpass_cannot_become_advantage(self):
        registration = dict(seeds=list(range(64)), repair_retention=dict(uncertain_seed_clusters=list(range(5))))
        partial = dict(inferential_test_allowed=False, statistically_supported_advantage=False)
        add_mixed_sensitivity(partial, registration)
        self.assertEqual(partial['mixed_repair_sensitivity']['status'], 'INCOMPLETE_NO_INFERENCE')
        self.assertFalse(partial['statistically_supported_advantage'])
        full = dict(inferential_test_allowed=True, statistically_supported_advantage=False,
                    contrasts=[dict(seed=i, difference=.5) for i in range(64)])
        add_mixed_sensitivity(full, registration)
        self.assertEqual(full['mixed_repair_sensitivity']['untouched_seed_conditional_evidence']['seed_clusters'], 59)
        self.assertFalse(full['statistically_supported_advantage'])

    def test_a_refusal_cannot_become_a_loss_or_wrong_arm_recovery(self):
        for row in (dict(status='REFUSED', signed_outcome=-1, arm='paper_reference'),
                    dict(status='COMPLETE', signed_outcome=1, arm='paper_reference'),
                    dict(status='REFUSED', signed_outcome=None, arm='deep_incumbent')):
            with self.subTest(row=row), self.assertRaises(RuntimeError):
                reference_resume(row, None, [], {}, {}, restore_reference_checkpoint=None, sha=None)


class AcceptedPrefixTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.identity = 'seed-42-p1-paper_reference'
        directory = self.root / self.identity
        directory.mkdir()
        self.path = self.root / (self.identity + '.json')
        self.row = dict(identity=self.identity, seed=42, subject='p1', arm='paper_reference',
                        status='REFUSED', signed_outcome=None,
                        failure_evidence={'not_accepted_draws': 999})
        self.steps = []
        for boundary in range(2):
            counts = [boundary+1, boundary+2]
            witness = dict(worker_pids=[100, 101], decision_id=boundary+1,
                statistics_checkpoint={'battle_id': 'wider-search:' + self.identity},
                result={'world_draws': sum(counts)}, worker_receipts=[
                    dict(worker=i, batch={'world_draws': count}, evidence={'draws': [
                        dict(status='ROOT_VALIDATED', released=True)] * count})
                    for i, count in enumerate(counts)])
            value = dict(boundary=boundary, actions={'p1': 0, 'p2': 0},
                chance_seed=continuation_seed('wider-search:42:p1', 0, boundary, 'chance'),
                evidence={'p1': dict(selector='paper_reference', elapsed_seconds=0.,
                    seed=continuation_seed('wider-search:42:p1', 0, boundary, 'search'), search_evidence=witness)})
            path = directory / f'boundary-{boundary:03d}.json.gz'
            path.write_bytes(gzip.compress(json.dumps(value).encode()))
            self.steps.append(path)
        self.path.write_text(json.dumps(self.row))
        self.old = dict(max_boundaries=200, per_game_wall_seconds=2400., reference={'workers': 2})
        self.sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
        self.probe = dict(identity=self.identity, result_sha256=self.sha(self.path), boundary=2, hidden_ordinal=3)

    def resume(self):
        return reference_resume(self.row, self.path, self.steps, self.old, self.probe,
            restore_reference_checkpoint=lambda checkpoint: SimpleNamespace(**checkpoint), sha=self.sha)

    def mutate(self, boundary, change):
        path = self.steps[boundary]
        value = json.loads(gzip.decompress(path.read_bytes()))
        change(value)
        path.write_bytes(gzip.compress(json.dumps(value).encode()))

    def test_only_accepted_checkpoint_and_all_ordinals_are_restored(self):
        result = self.resume()
        self.assertEqual(result['worker_ordinals'], [3, 5])
        self.assertEqual(result['decision_id'], 2)
        self.assertEqual(result['start_boundary'], 2)
        self.assertEqual(result['prior_selections'], 2)
        self.assertEqual(result['checkpoint_step'], str(self.steps[-1]))
        self.assertGreaterEqual(result['elapsed_before_resume'], 1.)
        self.assertNotIn('failure_evidence', result)

    def test_bad_checkpoint_draws_chance_and_worker_ids_refuse(self):
        changes = [
            lambda v: v.update(chance_seed=-1),
            lambda v: v['evidence']['p1'].update(selector='raw'),
            lambda v: v['evidence']['p1']['search_evidence']['result'].update(world_draws=0),
            lambda v: v['evidence']['p1']['search_evidence']['worker_receipts'][0].update(worker=2),
            lambda v: v['evidence']['p1']['search_evidence']['worker_receipts'][0]['evidence']['draws'][0].update(released=False),
            lambda v: v['evidence']['p1']['search_evidence']['statistics_checkpoint'].update(battle_id='different'),
        ]
        original = self.steps[-1].read_bytes()
        for change in changes:
            with self.subTest(change=change):
                self.mutate(1, change)
                with self.assertRaises(RuntimeError):
                    self.resume()
                self.steps[-1].write_bytes(original)

    def test_cap_and_probe_identity_cannot_reset(self):
        for key, value in [('hidden_ordinal', 99), ('boundary', 1), ('result_sha256', 'wrong')]:
            original = self.probe[key]
            self.probe[key] = value
            with self.subTest(key=key), self.assertRaises(RuntimeError):
                self.resume()
            self.probe[key] = original
        self.old['per_game_wall_seconds'] = 0.
        with self.assertRaisesRegex(RuntimeError, 'wall cap'):
            self.resume()
