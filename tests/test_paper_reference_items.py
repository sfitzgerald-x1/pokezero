"""Confirmed consumption preserves Recycle history, not just an empty item."""
from dataclasses import replace
from types import SimpleNamespace
import unittest

from _showdown_root import requires_showdown, showdown_root
from pokezero.env import BattleStartOverride
from pokezero.local_showdown import (LocalShowdownConfig, LocalShowdownEnv, LocalShowdownError,
    _public_consumed_item_history, _public_materialization_payload)
from pokezero.showdown_fixture import FixturePokemon, pack_team


class ConsumedItemLedgerTests(unittest.TestCase):
    def history(self, lines):
        return _public_consumed_item_history(SimpleNamespace(replay=SimpleNamespace(
            public_events=[SimpleNamespace(raw_line=line) for line in lines])), 'p1')

    def test_turn_resets_active_flag_but_retains_benched_consumption(self):
        lines = ['|switch|p1a: Feraligatr|Feraligatr|100/100',
            '|-enditem|p1a: Feraligatr|Salac Berry|[eat]']
        self.assertTrue(self.history(lines)['feraligatr']['usedItemThisTurn'])
        self.assertFalse(self.history(lines+['|turn|2'])['feraligatr']['usedItemThisTurn'])
        self.assertTrue(self.history(lines+['|switch|p1a: Pikachu|Pikachu|100/100',
            '|turn|2'])['feraligatr']['usedItemThisTurn'])

    def test_other_seat_unknown_mutations_and_recycle_do_not_discharge_removal(self):
        self.assertEqual(self.history(['|-enditem|p2a: Feraligatr|Salac Berry|[eat]']), {})
        for tail in ['|-item|p1a: Feraligatr|Salac Berry|[from] move: Recycle',
                '|-enditem|p1a: Feraligatr|Leftovers|[from] move: Knock Off',
                '|-enditem|p1a: Feraligatr|Salac Berry|[from] move: Thief']:
            self.assertEqual(self.history(['|-enditem|p1a: Feraligatr|Salac Berry|[eat]', tail]), {})


@requires_showdown()
class ConsumedItemServerTests(unittest.TestCase):
    def test_opt_in_matches_consumed_berry_and_recycle_for_both_actor_seats(self):
        config = LocalShowdownConfig(showdown_root=showdown_root(), set_belief_source=True)
        override = BattleStartOverride(player_teams={
            'p1': pack_team((FixturePokemon(species='Feraligatr', ability='Torrent',
                item='Salac Berry', ivs={'hp': 30}, evs={'hp': 8},
                moves=('Substitute', 'Recycle', 'Splash', 'Recover')),)),
            'p2': pack_team((FixturePokemon(species='Dusclops', ability='Pressure', moves=('Night Shade', 'Splash')),))})
        with LocalShowdownEnv(config) as live, LocalShowdownEnv(config) as restored:
            live.reset_with_start_override(seed=223, start_override=override)
            for _ in range(3):
                live.step({'p1': 0, 'p2': 0})
            for actor in ('p1', 'p2'):
                state = live.public_materialization_state(actor)
                with self.assertRaisesRegex(LocalShowdownError, 'item-state-removed'):
                    restored.materialize_public_world(state=state, start_override=override, seed=19)
                restored.materialize_public_world(state=state, start_override=override, seed=19,
                    reference_consumed_items=True)
                actual = live.snapshot().bridge_snapshot['battle']['sides'][0]['pokemon'][0]
                rebuilt = restored.snapshot().bridge_snapshot['battle']['sides'][0]['pokemon'][0]
                for field in ('item', 'lastItem', 'usedItemThisTurn', 'ateBerry', 'itemKnockedOff'):
                    self.assertEqual(rebuilt[field], actual[field], field)
                restored.step({'p1': 3, 'p2': 1})  # Heal above pinch threshold first.
                restored.step({'p1': 1, 'p2': 1})
                after = restored.snapshot().bridge_snapshot['battle']['sides'][0]['pokemon'][0]
                self.assertEqual(after['item'], 'salacberry')
                self.assertEqual(after['lastItem'], '')
            # A forged/truncated ledger cannot clear even a positive belief flag.
            truncated = replace(state, replay=replace(state.replay, public_events=()))
            payload = _public_materialization_payload(truncated, reference_consumed_items=True)
            self.assertIn('item-state-removed:Feraligatr', payload['sides']['p1']['materializationBlockers'])


if __name__ == '__main__':
    unittest.main()
