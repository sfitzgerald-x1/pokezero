"""Public constructor regressions from the five retained real decisions."""
import copy
import hashlib
import json
from dataclasses import replace
from pathlib import Path
import unittest

from gpu_engine.phase_d_constructor_rules import own_pp_carry
from gpu_engine import phase_d_tail as tail
from pokezero.live_protocol_actor import _packet
from test_phase_d_tail import storage

FIXTURES = Path(__file__).parent / 'fixtures' / 'phase_d_constructor_walls'


def fixture(name):
    data = json.loads((FIXTURES / (name + '.json')).read_text())
    lines = data['request']['lines']
    digest = hashlib.sha256(''.join(line + '\n' for line in lines).encode()).hexdigest()
    if digest != data['prefix_sha256']:
        raise ValueError('fixture prefix changed')
    return data


class ConstructorWallsTests(unittest.TestCase):
    def test_arrival_without_move_preserves_full_pp(self):
        data = fixture('pp_arrival')
        request = data['request']
        packet = _packet(request['own_request_line'], request['owner'])
        carry = data['own_pp_carry']
        self.assertEqual(own_pp_carry(request['lines'], packet, carry['maximum_pp']), carry['rows'])
        lines = list(request['lines'])
        at = lines.index('|faint|p2a: Pelipper')
        lines.insert(at, '|move|p2a: Pelipper|Surf|p1a: Nosepass')
        with self.assertRaisesRegex(ValueError, 'OWN_PP_LAST_ACTIVE_REQUEST_UNAVAILABLE'):
            own_pp_carry(lines, packet, carry['maximum_pp'])

    def test_departed_ditto_recovers_paid_base_slots(self):
        data = fixture('ditto_transform')
        request = data['request']
        packet = _packet(request['own_request_line'], request['owner'])
        carry = data['own_pp_carry']
        rows = own_pp_carry(request['lines'], packet, carry['maximum_pp'])
        self.assertEqual(rows, carry['rows'])
        ditto = next(i for i, row in enumerate(packet['side']['pokemon']) if row['ident'].endswith(': Ditto'))
        self.assertEqual(rows[ditto], [15, 0, 0, 0])

    def test_fainted_effects_are_cleared_without_mutating_source(self):
        fixture('fainted_effect')
        spec, _ = storage()
        side = replace(spec.side_one, boosts={'attack': 2}, volatile_statuses=('truant',),
                       volatile_status_durations={'truant': 1}, substitute_health=20,
                       side_conditions={'protect': 1, 'toxic_count': 3, 'spikes': 2})
        original = replace(spec, side_one=side)
        projected, cursors = tail.mapping_spec(original, {'pending_residual': True})
        self.assertEqual(projected.side_one.boosts, {'attack': 0})
        self.assertFalse(projected.side_one.volatile_statuses)
        self.assertFalse(projected.side_one.volatile_status_durations)
        self.assertEqual(projected.side_one.substitute_health, 0)
        self.assertEqual(projected.side_one.side_conditions, {'spikes': 2})
        self.assertEqual(original.side_one.boosts, {'attack': 2})
        self.assertEqual(original.side_one.active_index, 0)
        self.assertEqual(projected.side_one.active_index, 1)

    def test_full_passed_substitute_and_damage_refusal(self):
        data = fixture('passed_substitute')
        side = data['owner']
        payload = {'sides': {side: dict(pokemon=[dict(species='Pidgeot', active=True)],
            volatiles=['substitute'], substituteHealthState='full', substituteDepletion=0,
            substituteMinDepletion=0, substituteOriginSpecies='Jolteon')}}
        origins = {}
        self.assertEqual(tail.fresh_substitutes(payload, data['request']['lines'], origins), {side: 'pidgeot'})
        self.assertEqual(origins, {side: 'jolteon'})
        lines = data['request']['lines'] + [f'|-activate|{side}a: Pidgeot|Substitute|[damage]']
        with self.assertRaisesRegex(ValueError, 'PUBLIC_SUBSTITUTE_NOT_FRESH_OR_PASSED'):
            tail.fresh_substitutes(payload, lines, {})
        damaged = copy.deepcopy(payload)
        damaged['sides'][side]['substituteDepletion'] = 1
        with self.assertRaisesRegex(ValueError, 'PUBLIC_SUBSTITUTE_NOT_FRESH_OR_PASSED'):
            tail.fresh_substitutes(damaged, data['request']['lines'], {})

    def test_all_real_fixture_prefixes_are_retained(self):
        for name in ('pp_arrival', 'fainted_effect', 'unown_forme', 'ditto_transform', 'passed_substitute'):
            with self.subTest(name=name):
                data = fixture(name)
                self.assertEqual(data['base']['built'], 0)
                self.assertEqual(data['patched']['built'], 16)
