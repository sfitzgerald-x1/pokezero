"""Refresh choices use ONLY newly observed constraints, before bank sampling."""
from types import SimpleNamespace as S
import unittest
from pokezero.mcts_eval.paper_reference import ReferenceRefusal
from pokezero.mcts_eval.paper_reference_prefix_frontier import public_refresh_certificate
from pokezero.mcts_eval.paper_reference_runtime import ShowdownWorkerFactory


def state(lines, player='p1', **private):
    return S(player_id=player, replay=S(public_events=[S(raw_line=line) for line in lines]), **private)


class PublicRefreshTests(unittest.TestCase):
    def decision(self, old, suffix):
        return public_refresh_certificate(state(old), state(old+suffix))

    def test_repeated_moves_damage_and_ordinary_turns_do_not_refresh(self):
        old = ['|turn|1', '|move|p2a: Shuckle|Wrap|p1a: Lugia']
        result = self.decision(old, ['|turn|2', '|move|p2a: Shuckle|Wrap|p1a: Lugia',
            '|-activate|p1a: Lugia|Substitute|[damage]', '|upkeep'])
        self.assertFalse(result['refresh_required'])

    def test_encore_ending_is_a_new_public_constraint_even_in_later_episode(self):
        old = ['|turn|1', '|-end|p1a: Lugia|Encore', '|turn|2']
        result = self.decision(old, ['|-end|p1a: Lugia|Encore', '|upkeep'])
        self.assertTrue(result['refresh_required'])
        self.assertEqual(result['reasons'], [dict(kind='observed_encore_end', actor='p1a: Lugia', value='Encore')])

    def test_new_opponent_and_move_refresh_without_reading_party_or_bank(self):
        old = ['|switch|p2a: Shuckle|Shuckle, L98, M|198/198', '|move|p2a: Shuckle|Wrap|p1a: Lugia']
        new = old + ['|switch|p2a: Marowak|Marowak, L86, M|252/252', '|move|p2a: Marowak|Earthquake|p1a: Lugia']
        one = public_refresh_certificate(state(old, hidden_party=['a'], ess=32), state(new, hidden_party=['b'], survivors=0))
        two = public_refresh_certificate(state(old, hidden_party=['z'], ess=1), state(new, hidden_party=[], survivors=32))
        self.assertEqual(one, two)
        self.assertEqual({r['kind'] for r in one['reasons']}, {'opponent_details', 'opponent_move'})
        self.assertFalse(one['bank_outcome_used'])

    def test_known_switch_and_revealed_trait_do_not_repeat_refresh(self):
        old = ['|switch|p2a: Shuckle|Shuckle, L98, M|198/198', '|-item|p2a: Shuckle|Leftovers']
        self.assertFalse(self.decision(old, old)['refresh_required'])
        self.assertTrue(self.decision(old, ['|-ability|p2a: Shuckle|Sturdy'])['refresh_required'])

    def test_own_new_move_does_not_refresh_hidden_opponent_bank(self):
        self.assertFalse(self.decision(['|turn|1'], ['|move|p1a: Lugia|Psychic|p2a: Shuckle'])['refresh_required'])

    def test_actor_or_history_drift_refuses(self):
        with self.assertRaises(ReferenceRefusal):
            public_refresh_certificate(state(['|turn|1']), state(['|turn|2']))
        with self.assertRaises(ReferenceRefusal):
            public_refresh_certificate(state(['|turn|1']), state(['|turn|1'], player='p2'))

    def test_flag_defaults_off_and_requires_reuse(self):
        self.assertFalse(ShowdownWorkerFactory('a','b','c','d').history_prefix_public_refresh)
        for bad in (True, 1, 'true'):
            with self.assertRaises(ReferenceRefusal):
                ShowdownWorkerFactory('a','b','c','d', history_prefix_public_refresh=bad)
        self.assertTrue(ShowdownWorkerFactory('a','b','c','d', history_particles=32,
            history_prefix_reuse=True, history_prefix_public_refresh=True).history_prefix_public_refresh)


if __name__ == '__main__':
    unittest.main()
