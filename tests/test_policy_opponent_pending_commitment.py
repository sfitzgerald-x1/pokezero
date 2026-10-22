"""Authored pending-action contract; never replay a stopped scientific game."""
import json
import os
from pathlib import Path
import subprocess
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from pokezero.poke_engine_adapter import (
    BattleSpec, MoveSpec, PokemonSpec, SideSpec, build_poke_engine_state,
)
from pokezero.engine_world import pending_baton_opponent_action, battle_spec_from_payload, world_battle_spec, EngineWorldUnsupported
import test_engine_world as engine_world_fixtures
import random

try:
    import poke_engine
    import pokezero_search
except ImportError:
    poke_engine = None
    pokezero_search = None

ROOT = Path(__file__).resolve().parents[1]


class PublicCommitmentChronologyTests(unittest.TestCase):
    def test_fast_slow_and_confusion_consumption_in_both_seats(self):
        for passer in ('p1', 'p2'):
            foe = 'p2' if passer == 'p1' else 'p1'
            baton = f'|move|{passer}a: Mew|Baton Pass|{passer}a: Mew'
            for prior, expected in (([], 'PENDING'),
                    ([f'|move|{foe}a: Charmander|Ember|{passer}a: Mew'], 'ALREADY_CONSUMED'),
                    ([f'|cant|{foe}a: Charmander|par'], 'ALREADY_CONSUMED'),
                    ([f'|switch|{foe}a: Ditto|Ditto, L100|100/100'], 'ALREADY_CONSUMED'),
                    ([f'|-activate|{foe}a: Charmander|confusion',
                      f'|-damage|{foe}a: Charmander|90/100|[from] confusion'], 'ALREADY_CONSUMED'),
                    ([f'|-singleturn|{foe}a: Charmander|move: Focus Punch'], 'PENDING'),
                    ([f'|drag|{foe}a: Ditto|Ditto, L100|100/100'], 'AMBIGUOUS')):
                with self.subTest(passer=passer, prior=prior):
                    proof = pending_baton_opponent_action(['|turn|7', *prior, baton], turn=7, passer=passer)
                    self.assertEqual(proof, {'turn': 7, 'passer': passer, 'opponent': foe, 'state': expected})

    def test_truncated_called_duplicate_and_late_histories_refuse(self):
        baton = '|move|p1a: Mew|Baton Pass|p1a: Mew'
        for lines in ([baton], ['|turn|7'], ['|turn|7', baton, baton],
                ['|turn|7', '|turn|7', baton], ['|turn|6', baton],
                ['|turn|7', baton, '|upkeep'], ['|turn|7', baton, '|turn|8'],
                ['|turn|7', baton + '|[from] move: Sleep Talk'],
                ['|turn|7', '|move|p2a: Charmander|Ember|p1a: Mew|[from] move: Metronome', baton],
                ['|turn|7', baton, '|move|p2a: Charmander|Ember|p1a: Ditto'],
                ['|turn|7', '|-activate|p2a: Charmander|unknown-attempt', baton]):
            with self.subTest(lines=lines):
                self.assertEqual(pending_baton_opponent_action(lines, turn=7, passer='p1')['state'], 'AMBIGUOUS')

    def test_target_or_of_annotation_is_not_the_actor(self):
        lines = ['|turn|7', '|-damage|p1a: Mew|90/100|[of] p2a: Charmander',
            '|move|p1a: Mew|Baton Pass|p2a: Charmander']
        self.assertEqual(pending_baton_opponent_action(lines, turn=7, passer='p1')['state'], 'PENDING')

    def test_consumed_materializes_no_commitment_and_missing_proof_refuses(self):
        fixture = engine_world_fixtures.BatonPassBoundaryTests()
        fixture.setUp()
        payload = fixture._bp_payload()
        payload['pendingBatonPassOpponentAction']['state'] = 'ALREADY_CONSUMED'
        world = battle_spec_from_payload(payload, engine_world_fixtures._override(), dex=fixture.dex)
        self.assertFalse(world.spec.side_two.slow_uturn_move)
        self.assertEqual(world.spec.side_two.switch_out_move_second_saved_move, '')
        for proof in (None, {}, {'state': []},
                {'turn': 6, 'passer': 'p1', 'opponent': 'p2', 'state': 'PENDING'},
                {'turn': 7, 'passer': 'p2', 'opponent': 'p1', 'state': 'PENDING'}):
            payload['pendingBatonPassOpponentAction'] = proof
            with self.subTest(proof=proof), self.assertRaises(EngineWorldUnsupported):
                battle_spec_from_payload(payload, engine_world_fixtures._override(), dex=fixture.dex, rng=random.Random(7))

    def test_world_constructor_binds_public_chronology_without_private_queue_access(self):
        fixture = engine_world_fixtures.BatonPassBoundaryTests()
        fixture.setUp()
        payload = fixture._bp_payload()
        payload.pop('pendingBatonPassOpponentAction')
        state = SimpleNamespace(player_id='p1', replay=SimpleNamespace(public_lines=(
            '|turn|7', '|cant|p2a: Snorlax|par', '|move|p1a: Swampert|Baton Pass|p1a: Swampert')))
        with patch('pokezero.local_showdown._public_materialization_payload', return_value=payload):
            world = world_battle_spec(state, engine_world_fixtures._override(), dex=fixture.dex)
        self.assertFalse(world.spec.side_two.slow_uturn_move)
        self.assertEqual(world.spec.side_two.switch_out_move_second_saved_move, '')


def pending_state(waiting_seat, committed=True):
    def mon(name, moves, speed=100, pp=9):
        return PokemonSpec(id=name, level=100, types=('normal',), hp=1000, maxhp=1000,
            attack=100, defense=100, special_attack=100, special_defense=100, speed=speed,
            ability='none', moves=tuple(MoveSpec(id=m, pp=pp) for m in moves))
    passing = SideSpec(pokemon=(mon('mew', ('batonpass', 'splash'), 300),
        mon('ditto', ('splash',))), force_switch=True, baton_passing=True)
    # Native deliberately elides high-PP decrements; the production own-PP
    # ledger repairs those. Low PP isolates this transition's actual charge.
    waiting = SideSpec(pokemon=(mon('charmander', ('ember', 'growl'), pp=9 if committed else 8),
        mon('ditto', ('splash',))), slow_uturn_move=committed,
        switch_out_move_second_saved_move='ember' if committed else '')
    return build_poke_engine_state(BattleSpec(
        side_one=waiting if waiting_seat == 'p1' else passing,
        side_two=waiting if waiting_seat == 'p2' else passing)).to_string()


@unittest.skipUnless(pokezero_search is not None and os.environ.get('POKEZERO_SHOWDOWN_ROOT'),
                     'requires native constructor and pinned Showdown')
class PendingCommitmentTests(unittest.TestCase):
    def test_showdown_wait_preserves_then_executes_committed_move_once(self):
        oracle = json.loads(subprocess.run([os.environ.get('POKEZERO_NODE_BINARY', 'node'),
            str(ROOT / 'tests/policy_opponent_pending_commitment_oracle.cjs'),
            os.environ['POKEZERO_SHOWDOWN_ROOT']], capture_output=True,
            text=True, check=True, timeout=30).stdout)
        self.assertFalse(oracle['scientific_measurement'])
        self.assertFalse(oracle['historical_replay'])
        for row in oracle['rows']:
            with self.subTest(waiting_seat=row['waitingSeat'], fast_pass=row['fastPass']):
                self.assertTrue(row['before']['request']['wait'])
                proof = pending_baton_opponent_action(['|turn|1', *row['before']['events']],
                    turn=1, passer=row['passingSeat'])
                self.assertEqual(proof['state'], 'PENDING' if row['fastPass'] else 'ALREADY_CONSUMED')
                if row['scenario'] == 'confusion':
                    self.assertEqual(row['before']['hp'], 980)
                    self.assertEqual(row['before']['pp'], 9)
                    self.assertEqual(row['after']['pp'], 9)
                    self.assertEqual(row['after']['recipientHP'], 1000)
                    self.assertEqual(row['after']['damage'], [])
                    continue
                self.assertEqual(row['before']['pp'], 9 if row['fastPass'] else 8)
                self.assertEqual(len(row['before']['damage']), 0 if row['fastPass'] else 1)
                self.assertEqual(row['after']['pp'], 8)
                self.assertEqual(row['after']['recipient'], 'ditto')
                self.assertEqual(row['after']['recipientHP'], 950 if row['fastPass'] else 1000)
                self.assertEqual(row['after']['damage'], [{'move': 'ember',
                    'source': row['waitingSeat'], 'target': 'ditto' if row['fastPass'] else 'mew',
                    'targetSeat': row['passingSeat']}])

    def test_native_saved_move_executes_against_recipient_without_resampling(self):
        for seat in ('p1', 'p2'):
            with self.subTest(waiting_seat=seat):
                context = {'p1': ['Charmander', 'Ditto'] if seat == 'p1' else ['Mew', 'Ditto'],
                           'p2': ['Charmander', 'Ditto'] if seat == 'p2' else ['Mew', 'Ditto'],
                           'turn': 1}
                choices = ('ember', 'ditto') if seat == 'p1' else ('ditto', 'ember')
                result = json.loads(pokezero_search.branch_events(pending_state(seat),
                    *choices, json.dumps(context), True, True))
                self.assertGreater(len(result['branches']), 0)
                for branch in result['branches']:
                    moves = [line for line in branch['events'] if line.startswith(f'|move|{seat}a:')]
                    self.assertEqual(len(moves), 1, branch)
                    self.assertIn('|ember|', moves[0].lower())
                    self.assertIn('Ditto', moves[0])
                    post = poke_engine.State.from_string(branch['post_state'])
                    own = post.side_one if seat == 'p1' else post.side_two
                    self.assertEqual(own.pokemon[0].moves[0].pp, 8)

    def test_already_executed_move_is_not_repeated_after_slow_pass(self):
        for seat in ('p1', 'p2'):
            with self.subTest(waiting_seat=seat):
                context = {'p1': ['Charmander', 'Ditto'] if seat == 'p1' else ['Mew', 'Ditto'],
                           'p2': ['Charmander', 'Ditto'] if seat == 'p2' else ['Mew', 'Ditto'],
                           'turn': 1}
                choices = ('none', 'ditto') if seat == 'p1' else ('ditto', 'none')
                result = json.loads(pokezero_search.branch_events(pending_state(seat, False),
                    *choices, json.dumps(context), True, True))
                for branch in result['branches']:
                    self.assertFalse(any(line.startswith(f'|move|{seat}a:') for line in branch['events']))
                    post = poke_engine.State.from_string(branch['post_state'])
                    own = post.side_one if seat == 'p1' else post.side_two
                    self.assertEqual(own.pokemon[0].moves[0].pp, 8)


if __name__ == '__main__':
    unittest.main()
