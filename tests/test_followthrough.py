"""Follow-through must not degrade to first-action-only or silent raw fallback."""
from types import SimpleNamespace
import unittest

from pokezero.mcts_eval.followthrough import continuation_seed, play_continuation
from pokezero.mcts_eval.paper_reference import Evaluation


class Env:
    def __init__(self, capped=False):
        self.boundary = 0
        self.actions = []
        self.chance = []
        self.capped = capped
    def terminal(self):
        return None if self.boundary < 4 else SimpleNamespace(capped=self.capped, winner='p1', turn_count=4)
    def requested_players(self):
        return ('p1', 'p2') if self.boundary in (0, 3) else ('p2',) if self.boundary == 1 else ('p1',)
    def observe(self, player):
        return SimpleNamespace(player=player, legal_action_mask=(True, True, False))
    def reseed_simulator_rng(self, seed):
        self.chance.append(seed)
    def step(self, actions):
        self.actions.append(actions)
        self.boundary += 1


def opponent(obs):
    assert obs.player == 'p2'
    return (0, 1), Evaluation((.25, .75), 0.)


class FollowThroughTests(unittest.TestCase):
    def test_fixed_first_action_then_search_on_every_own_request_only(self):
        calls, rows = [], []
        def search(obs, boundary, seed):
            self.assertEqual(obs.player, 'p1')
            calls.append((boundary, seed))
            return 1, {'searched': True}
        env = Env()
        result = play_continuation(env, subject='p1', first_action=0, decision_id='root',
            replicate=0, subject_selector=search, opponent_evaluator=opponent, emit=rows.append)
        self.assertEqual(result['status'], 'COMPLETE')
        self.assertEqual(result['signed_outcome'], 1)
        self.assertEqual([b for b, _ in calls], [2, 3])
        self.assertEqual(result['followthrough_decisions'], 2)
        self.assertEqual(env.actions[0]['p1'], 0)
        self.assertTrue(all(row['evidence']['p2']['selector'] == 'champion_full_masked_policy_sample'
            for row in rows if 'p2' in row['evidence']))

    def test_pairing_domains_do_not_depend_on_arm_and_do_not_share_rng_streams(self):
        self.assertEqual(continuation_seed('root', 0, 2, 'chance'), continuation_seed('root', 0, 2, 'chance'))
        self.assertEqual(len({continuation_seed('root', 0, 2, domain)
            for domain in ('chance', 'opponent', 'search')}), 3)
        self.assertNotEqual(continuation_seed('root', 0, 2, 'chance'), continuation_seed('root', 1, 2, 'chance'))

    def test_refusal_propagates_without_raw_fallback_or_another_step(self):
        env = Env()
        def refuse(*args):
            raise RuntimeError('search refused')
        with self.assertRaisesRegex(RuntimeError, 'search refused'):
            play_continuation(env, subject='p1', first_action=0, decision_id='root', replicate=0,
                subject_selector=refuse, opponent_evaluator=opponent, emit=lambda row: None)
        self.assertEqual(len(env.actions), 2)

    def test_caps_are_not_scored_as_wins(self):
        for env, limit in ((Env(), 2), (Env(capped=True), 200)):
            result = play_continuation(env, subject='p1', first_action=0, decision_id='root', replicate=0,
                subject_selector=lambda *args: (1, {}), opponent_evaluator=opponent,
                emit=lambda row: None, max_boundaries=limit)
            self.assertEqual(result['status'], 'CAPPED')
            self.assertIsNone(result['signed_outcome'])


if __name__ == '__main__':
    unittest.main()
