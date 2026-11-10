"""Source-bound replay-to-root checks; no checkpoint, device or simulator."""
import copy
import unittest
from unittest.mock import patch

from gpu_engine import fp_reply_data as data
from tests.test_gpu_engine_fp_reply import (
    FakeEnv, FakePolicy, OUR_SEAT, fake_choice_action, game, two_rounds,
)

SHA = '1' * 64


class SnapshotEnv(FakeEnv):
    def snapshot(self):
        return (self.seed, self.i, copy.deepcopy(self.steps))


class RootReplayTests(unittest.TestCase):
    def setUp(self):
        self.decode = patch.object(data, 'choice_action', fake_choice_action)
        self.decode.start()
        self.addCleanup(self.decode.stop)

    def replay(self, env, source=None, target=1, sha=SHA):
        return data.replay_to_root(env, FakePolicy(), source or game(checkpoint_sha256=SHA), None,
                                   decision_index=target, expected_checkpoint_sha256=sha)

    def test_boundary_is_before_target_action_and_carries_verified_request(self):
        env = SnapshotEnv(two_rounds(), OUR_SEAT)
        root = self.replay(env)
        self.assertEqual(root['snapshot'], (11, 1, [{OUR_SEAT: 0, 'p2': 2}]))
        self.assertEqual((root['decision_index'], root['round_index'], root['prefix_parity_hits']), (1, 1, 1))
        self.assertEqual(root['observation'].legal_action_mask, two_rounds()[1][OUR_SEAT][0])

    def test_first_root_never_consumes_recorded_current_reply(self):
        source = game(checkpoint_sha256=SHA, journal={})
        root = self.replay(SnapshotEnv(two_rounds(), OUR_SEAT), source, target=0)
        self.assertEqual(root['snapshot'], (11, 0, []))

    def test_prefix_and_current_request_mismatch_refuse(self):
        for index in (0, 1):
            with self.subTest(index=index):
                source = game(checkpoint_sha256=SHA)
                source['our_requests'][index] = None
                with self.assertRaisesRegex(data.Mismatch, 'request differs'):
                    self.replay(SnapshotEnv(two_rounds(), OUR_SEAT), source)

    def test_prefix_opponent_choice_and_legality_checks_are_retained(self):
        for entry in (dict(action=2, choice='act 5'), dict(action=8, choice='act 8')):
            with self.subTest(entry=entry):
                source = game(checkpoint_sha256=SHA)
                source['journal'][0].update(entry)
                with self.assertRaises(data.Mismatch):
                    self.replay(SnapshotEnv(two_rounds(), OUR_SEAT), source)

    def test_opponent_only_rounds_do_not_advance_own_decision_index(self):
        original = two_rounds()
        rounds = [{'p2': original[0]['p2']}, {OUR_SEAT: original[0][OUR_SEAT]},
                  {'p2': original[1]['p2']}, {OUR_SEAT: original[1][OUR_SEAT]}]
        source = game(checkpoint_sha256=SHA)
        source['journal'][2] = source['journal'].pop(1)
        source['journal'][2]['round'] = 2
        env = SnapshotEnv(rounds, OUR_SEAT)
        root = self.replay(env, source)
        self.assertEqual((root['decision_index'], root['round_index']), (1, 3))
        self.assertEqual(env.steps, [{'p2': 2}, {OUR_SEAT: 0}, {'p2': 5}])
        source['journal'][1] = dict(round=1, action=2, choice='act 2')
        with self.assertRaisesRegex(data.Mismatch, 'unrequested.*prefix'):
            self.replay(SnapshotEnv(rounds, OUR_SEAT), source)

    def test_source_checkpoint_must_match_before_reset(self):
        env = SnapshotEnv(two_rounds(), OUR_SEAT)
        for source in (game(), game(checkpoint_sha256='2' * 64)):
            with self.assertRaisesRegex(data.Mismatch, 'checkpoint'):
                self.replay(env, source)
        self.assertFalse(hasattr(env, 'i'))

    def test_invalid_or_unreachable_target_is_not_success(self):
        for target in (-1, True, 0.5):
            with self.assertRaises(ValueError):
                self.replay(SnapshotEnv(two_rounds(), OUR_SEAT), target=target)
        with self.assertRaisesRegex(data.Mismatch, 'source has no'):
            self.replay(SnapshotEnv(two_rounds(), OUR_SEAT), target=2)
        with self.assertRaisesRegex(data.Mismatch, 'terminal reached'):
            self.replay(SnapshotEnv(two_rounds()[:1], OUR_SEAT), target=1)
        with self.assertRaises(ValueError):
            self.replay(SnapshotEnv(two_rounds(), OUR_SEAT), sha='')
