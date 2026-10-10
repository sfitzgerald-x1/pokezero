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
    def history(self, lines, player='p1'):
        return _public_consumed_item_history(SimpleNamespace(replay=SimpleNamespace(
            public_events=[SimpleNamespace(raw_line=line) for line in lines])), player)

    def test_white_herb_uses_public_species_not_protocol_nickname_for_both_seats(self):
        for player in ('p1', 'p2'):
            for kind in ('switch', 'drag', 'replace'):
                with self.subTest(player=player, kind=kind):
                    lines = [f'|{kind}|{player}a: Deoxys|Deoxys-Attack, L70|186/186',
                             f'|-enditem|{player}a: Deoxys|White Herb']
                    self.assertEqual(self.history(lines, player), {'deoxysattack': {
                        'id': 'whiteherb', 'usedItemThisTurn': True, 'ateBerry': False}})

    def test_arbitrary_nickname_resolves_only_from_public_switch_details(self):
        lines = ['|switch|p1a: Floof|Feraligatr, L78, M|100/100',
                 '|-enditem|p1a: Floof|Salac Berry|[eat]']
        self.assertIn('feraligatr', self.history(lines))
        self.assertNotIn('floof', self.history(lines))

    def test_truncated_or_malformed_identity_history_never_guesses_species(self):
        for prefix in ([], ['|switch|p1a: Deoxys'],
                       ['|switch|p1a: Deoxys||186/186']):
            with self.subTest(prefix=prefix):
                self.assertEqual(self.history(prefix + ['|-enditem|p1a: Deoxys|White Herb']), {})

    def test_aliased_active_flag_and_bench_history_follow_public_switches(self):
        lines = ['|switch|p1a: Deoxys|Deoxys-Attack, L70|186/186',
                 '|-enditem|p1a: Deoxys|White Herb']
        self.assertFalse(self.history(lines + ['|turn|2'])['deoxysattack']['usedItemThisTurn'])
        benched = lines + ['|switch|p1a: Sparky|Pikachu, L80|100/100', '|turn|2']
        self.assertTrue(self.history(benched)['deoxysattack']['usedItemThisTurn'])
        returned = benched + ['|switch|p1a: Deoxys|Deoxys-Attack, L70|186/186', '|turn|3']
        self.assertFalse(self.history(returned)['deoxysattack']['usedItemThisTurn'])

    def test_aliased_unknown_mutation_invalidates_consumption_certificate(self):
        lines = ['|switch|p1a: Deoxys|Deoxys-Attack, L70|186/186',
                 '|-enditem|p1a: Deoxys|White Herb']
        for tail in ['|-item|p1a: Deoxys|White Herb|[from] move: Recycle',
                     '|-enditem|p1a: Deoxys|Leftovers|[from] move: Knock Off',
                     '|switch|p1a: Deoxys||186/186']:
            with self.subTest(tail=tail):
                self.assertEqual(self.history(lines + [tail]), {})

    def test_same_nickname_on_other_seat_cannot_supply_identity_or_consumption(self):
        lines = ['|switch|p1a: Deoxys|Deoxys-Attack, L70|186/186',
                 '|switch|p2a: Deoxys|Deoxys-Defense, L70|186/186',
                 '|-enditem|p2a: Deoxys|White Herb']
        self.assertEqual(self.history(lines), {})
        self.assertIn('deoxysdefense', self.history(lines, 'p2'))

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
            self.assertEqual(self.history(['|switch|p1a: Feraligatr|Feraligatr|100/100',
                '|-enditem|p1a: Feraligatr|Salac Berry|[eat]', tail]), {})


@requires_showdown()
class ConsumedItemServerTests(unittest.TestCase):
    def test_opt_in_matches_white_herb_on_public_form_alias_for_both_actor_seats(self):
        config = LocalShowdownConfig(showdown_root=showdown_root(), set_belief_source=True)
        override = BattleStartOverride(player_teams={
            'p1': pack_team((FixturePokemon(species='Deoxys-Attack', ability='Pressure',
                item='White Herb', moves=('Superpower', 'Splash')),)),
            'p2': pack_team((FixturePokemon(species='Shuckle', ability='Sturdy',
                moves=('Splash',)),))})
        with LocalShowdownEnv(config) as live, LocalShowdownEnv(config) as restored:
            live.reset_with_start_override(seed=223, start_override=override)
            live.step({'p1': 0, 'p2': 0})
            actual = live.snapshot().bridge_snapshot['battle']['sides'][0]['pokemon'][0]
            self.assertEqual(actual['lastItem'], 'whiteherb')
            self.assertEqual(actual['item'], '')
            for actor in ('p1', 'p2'):
                with self.subTest(actor=actor):
                    state = live.public_materialization_state(actor)
                    self.assertTrue(any(event.raw_line == '|-enditem|p1a: Deoxys|White Herb'
                        for event in state.replay.public_events))
                    with self.assertRaisesRegex(LocalShowdownError, 'item-state-removed'):
                        restored.materialize_public_world(state=state, start_override=override, seed=19)
                    restored.materialize_public_world(state=state, start_override=override, seed=19,
                        reference_consumed_items=True)
                    rebuilt = restored.snapshot().bridge_snapshot['battle']['sides'][0]['pokemon'][0]
                    for field in ('item', 'lastItem', 'usedItemThisTurn', 'ateBerry', 'itemKnockedOff'):
                        self.assertEqual(rebuilt[field], actual[field], field)
                    self.assertEqual(restored.observe(actor).legal_action_mask,
                                     live.observe(actor).legal_action_mask)

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
