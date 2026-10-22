"""Authored Trace public-event/request seam; never replay a stopped panel."""
import json
import os
from pathlib import Path
import subprocess
import unittest

from pokezero.poke_engine_adapter import BattleSpec, MoveSpec, PokemonSpec, SideSpec, build_poke_engine_state
from pokezero.policy_opponent_view import public_policy_lines

try:
    import pokezero_search
except ImportError:
    pokezero_search = None

ROOT = Path(__file__).resolve().parents[1]


def native_replacement(seat):
    def mon(name, ability, moves, hp=100):
        return PokemonSpec(id=name, level=100, types=('psychic',), hp=hp, maxhp=100,
            attack=100, defense=100, special_attack=100, special_defense=100, speed=100,
            ability=ability, moves=tuple(MoveSpec(id=m, pp=10) for m in moves))
    holder = SideSpec(pokemon=(mon('wobbuffet', 'shadowtag', ('safeguard', 'counter')),
                              mon('blissey', 'naturalcure', ('softboiled',))))
    tracer = SideSpec(pokemon=(mon('snorlax', 'immunity', ('rest',), hp=0),
                              mon('gardevoir', 'trace', ('psychic', 'calmmind')),
                              mon('blissey', 'naturalcure', ('softboiled',))), force_switch=True)
    return build_poke_engine_state(BattleSpec(
        side_one=holder if seat == 'p1' else tracer,
        side_two=tracer if seat == 'p1' else holder)).to_string()


@unittest.skipUnless(pokezero_search is not None and os.environ.get('POKEZERO_SHOWDOWN_ROOT'),
                     'requires native branch renderer and pinned Showdown source')
class TraceProjectionTests(unittest.TestCase):
    def test_replacement_trace_public_reveal_reaches_child_request_in_both_seats(self):
        result = json.loads(subprocess.run([os.environ.get('POKEZERO_NODE_BINARY', 'node'),
            str(ROOT / 'tests/policy_opponent_trace_oracle.cjs'), os.environ['POKEZERO_SHOWDOWN_ROOT']],
            capture_output=True, text=True, check=True, timeout=30).stdout)
        self.assertFalse(result['scientific_measurement'])
        self.assertFalse(result['historical_cause_established'])
        maximum = {'safeguard': 40, 'counter': 32, 'softboiled': 16}
        for row in result['rows']:
            with self.subTest(seat=row['seat']):
                seat = row['seat']
                other = 'p2' if seat == 'p1' else 'p1'
                expected = (f'|-ability|{other}a: Gardevoir|Shadow Tag|Trace|'
                            f'[from] ability: Trace|[of] {seat}a: Wobbuffet')
                self.assertEqual(row['copied_ability'], 'shadowtag')
                self.assertTrue(row['request']['trapped'])
                self.assertIn(expected, row['after'])
                names = {'p1': ['Wobbuffet', 'Blissey'] if seat == 'p1' else ['Snorlax', 'Gardevoir', 'Blissey'],
                         'p2': ['Wobbuffet', 'Blissey'] if seat == 'p2' else ['Snorlax', 'Gardevoir', 'Blissey']}
                context = json.dumps({**names, 'turn': 1})
                moves = ('none', 'gardevoir') if seat == 'p1' else ('gardevoir', 'none')
                branches = json.loads(pokezero_search.branch_events(native_replacement(seat),
                    *moves, context, True, True))['branches']
                self.assertEqual(len(branches), 1)
                branch = branches[0]
                self.assertFalse(branch['attribution_unsafe'])
                self.assertIn(expected, branch['events'])
                before = list(public_policy_lines(row['before'], hp_visibility={'p1': 'exact', 'p2': 'exact'}))
                supplied = json.loads(pokezero_search.sampled_policy_request(branch['post_state'], seat,
                    names[seat], names[seat], maximum, root=False,
                    public_lines=before + branch['events'], format_id='gen3randombattle'))
                self.assertTrue(supplied['request']['active'][0]['trapped'])
                self.assertEqual(supplied['native_action_indices'], [0, 1])
                # Never accept sampled ability truth in place of a public reveal.
                omitted = [line for line in branch['events'] if line != expected]
                with self.assertRaisesRegex(ValueError, 'legal surfaces differ'):
                    pokezero_search.sampled_policy_request(branch['post_state'], seat,
                        names[seat], names[seat], maximum, root=False,
                        public_lines=before + omitted, format_id='gen3randombattle')


if __name__ == '__main__':
    unittest.main()
