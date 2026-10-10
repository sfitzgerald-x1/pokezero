"""Recovery must bind all nineteen proofs, not select convenient historical games."""
import copy
import unittest

from pokezero.mcts_eval.wider_faint_recovery import (
    HISTORICAL_OBSERVER, VALID, validate_prior_audit, validate_resumed_draw,
)


class ResumedDrawTests(unittest.TestCase):
    def setUp(self):
        self.draw = dict(status='ROOT_VALIDATED', released=True, packed_team_sha256='conditional-team',
            materialization_seed=42, pending_policy_conditioning=dict(
                transition_kind='pre-upkeep-faint-replacement', attempts=2, own_prior_action=1,
                current_actor_root_key='exact-root', live_opponent_action_used=False))
        self.probe = dict(witnesses=[dict(draw=copy.deepcopy(self.draw))])

    def test_repaired_conditional_draw_matches_qualified_witness(self):
        validate_resumed_draw([self.draw], self.probe)

    def test_no_completed_worker_zero_batch_does_not_change_scheduling(self):
        validate_resumed_draw([], self.probe)

    def test_world_seed_status_and_conditioning_cannot_drift(self):
        for key, value in (('packed_team_sha256', 'old-unconditional-world'),
                           ('materialization_seed', 43), ('status', 'STARTED'), ('released', False)):
            changed = copy.deepcopy(self.draw)
            changed[key] = value
            with self.subTest(key=key), self.assertRaises(RuntimeError):
                validate_resumed_draw([changed], self.probe)
        changed = copy.deepcopy(self.draw)
        changed['pending_policy_conditioning']['current_actor_root_key'] = 'other'
        with self.assertRaises(RuntimeError):
            validate_resumed_draw([changed], self.probe)


class PriorAuditTests(unittest.TestCase):
    def setUp(self):
        self.old = dict(source_commit='original')
        self.audit = dict(schema='pokezero.wider-search.mixed-trapping-recovery-read-only-audit.v4',
            observer_sha256=HISTORICAL_OBSERVER, source_commit='original', registration_sha256='registration',
            registered_games=256, audited_complete_games=19, strength_inference=False,
            literal_homogeneous_source=False,
            games=[dict(identity=f'game-{i}', status=VALID) for i in range(19)])

    def test_exact_complete_original_inventory_is_required(self):
        self.assertEqual(len(validate_prior_audit(self.audit, self.old, 'registration')), 19)

    def test_partial_or_relabelled_proof_does_not_replace_original_evidence(self):
        for key, value in (('registered_games', 76), ('audited_complete_games', 18),
                           ('source_commit', 'new'), ('registration_sha256', 'other'),
                           ('observer_sha256', 'other'), ('strength_inference', True),
                           ('literal_homogeneous_source', True)):
            changed = copy.deepcopy(self.audit)
            changed[key] = value
            with self.subTest(key=key), self.assertRaises(RuntimeError):
                validate_prior_audit(changed, self.old, 'registration')

    def test_duplicate_or_invalid_game_is_rejected(self):
        for key, value in (('identity', 'game-1'), ('status', 'REFUSED')):
            changed = copy.deepcopy(self.audit)
            changed['games'][0][key] = value
            with self.subTest(key=key), self.assertRaises(RuntimeError):
                validate_prior_audit(changed, self.old, 'registration')


if __name__ == '__main__':
    unittest.main()
