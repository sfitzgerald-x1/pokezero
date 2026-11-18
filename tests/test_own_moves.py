"""pokezero.own_moves: our own real movesets and exact PP from the owner's protocol history (V2 self_moves).

Pure Python. Owner p1 with an authored team; each case is an owning-wire history ending at the current request. Where
gpu_engine.phase_d_constructor_rules.own_pp_carry (the constructor's world PP) succeeds, own_move_states' PP must equal
it for every non-transformed member (the per-mon port's contract)."""
import json
import unittest

from pokezero.own_moves import move_key, move_label, own_move_states

MAX = {'bodyslam': 24, 'rest': 16, 'sleeptalk': 16, 'curse': 16, 'thunderbolt': 24, 'hiddenpower': 24, 'explosion': 8,
       'hypnosis': 32, 'return': 32, 'toxic': 16, 'transform': 16, 'surf': 24, 'icebeam': 16, 'protect': 16,
       'spikes': 32, 'earthquake': 16, 'swordsdance': 48, 'shadowball': 24}
TEAM = (('Snorlax', ('bodyslam', 'rest', 'sleeptalk', 'curse')),
        ('Gengar', ('thunderbolt', 'hiddenpowerfire', 'explosion', 'hypnosis')),
        ('Zapdos', ('thunderbolt', 'hiddenpowerice', 'return102', 'toxic')),
        ('Ditto', ('transform',)),
        ('Starmie', ('surf', 'icebeam', 'protect', 'thunderbolt')),
        ('Skarmory', ('spikes', 'protect', 'toxic', 'rest')))
START = ('|player|p1|Us||', '|player|p2|Them||', '|gen|3', '|tier|[Gen 3] Random Battle', '|start',
         '|switch|p1a: Snorlax|Snorlax, L80|400/400', '|switch|p2a: Absol|Absol, L80|100/100', '|turn|1')


def maximum(key):
    return MAX[key]


def request(active, pp=None, *, rqid=1, force=False, wait=False, fainted=(), active_moves=None):
    """The owner's request: `active` the active species (None: no active row), `pp` the active's {move id: pp}."""
    rows = []
    for name, moves in TEAM:
        rows.append({'ident': f'p1: {name}', 'details': f'{name}, L80',
                     'condition': '0 fnt' if name in fainted else '300/300', 'active': name == active,
                     'moves': list(active_moves if (name == active and active_moves is not None) else moves),
                     'baseAbility': 'none', 'item': 'leftovers', 'stats': {'atk': 1, 'def': 1, 'spa': 1, 'spd': 1, 'spe': 1}})
    out = {'rqid': rqid, 'side': {'id': 'p1', 'name': 'Us', 'pokemon': rows}}
    if force:out['forceSwitch'] = [True]
    elif wait:out['wait'] = True
    elif active is not None:
        ids = active_moves if active_moves is not None else dict(TEAM)[active]
        out['active'] = [{'moves': [{'move': m, 'id': ('hiddenpower' if m.startswith('hiddenpower') else
                                                        'return' if m.startswith('return') else m),
                                     'pp': (pp or {}).get(m, MAX[move_key(m)]), 'maxpp': MAX[move_key(m)]} for m in ids]}]
    return out


def line(packet):
    return '|request|' + json.dumps(packet)


def states(lines, current):
    return {s.ident: s for s in own_move_states(tuple(lines) + (line(current),), current, maximum_pp=maximum)}


def carry(lines, current):
    from gpu_engine.phase_d_constructor_rules import own_pp_carry
    return own_pp_carry(tuple(lines) + (line(current),), current, MAX)


class MoveIdTests(unittest.TestCase):
    def test_labels_and_keys(self):
        self.assertEqual([move_label(x) for x in ('return102', 'frustration1', 'hiddenpowerfire60', 'hiddenpowerfire',
                                                  'conversion2', 'Body Slam')],
                         ['return', 'frustration', 'hiddenpowerfire', 'hiddenpowerfire', 'conversion2', 'bodyslam'])
        self.assertEqual([move_key(x) for x in ('hiddenpowerfire', 'Hidden Power', 'return102')], ['hiddenpower', 'hiddenpower', 'return'])


class OwnMoveStateTests(unittest.TestCase):
    def assert_matches_carry(self, lines, current, *, skip=()):
        got, want = states(lines, current), carry(lines, current)
        for (name, _), row, pp in zip(TEAM, current['side']['pokemon'], want):
            if name.casefold() in skip:continue
            s = got[name.casefold()]
            by_key = dict(zip([move_key(m) for m in s.moves], s.pp))
            self.assertEqual([by_key[move_key(m)] for m in row['moves']] + [0] * (4 - len(row['moves'])), pp, name)

    def test_never_sent_out_is_full_and_labels_canonical(self):
        r1 = request('Snorlax', rqid=1)
        got = states(START, r1)
        self.assertEqual(got['zapdos'].moves, ('thunderbolt', 'hiddenpowerice', 'return', 'toxic'))
        self.assertEqual(got['zapdos'].pp, (24, 24, 32, 16))
        self.assertEqual(got['gengar'].moves[1], 'hiddenpowerfire')
        self.assertEqual(got['snorlax'].pp, (24, 16, 16, 16))
        self.assert_matches_carry(START, r1)

    def test_snapshot_then_public_charges_pressure_sleep_talk_struggle(self):
        r1 = request('Snorlax', rqid=1)
        lines = START + (line(r1), '|move|p1a: Snorlax|Body Slam|p2a: Absol', '|turn|2')
        r2 = request('Snorlax', {'bodyslam': 23}, rqid=2)
        lines += (line(r2), '|-ability|p2a: Absol|Pressure',
                  # Pressure public: a foe-targeted move costs 2 even with a blank wire target; Curse/Rest never pressured.
                  '|move|p1a: Snorlax|Body Slam||[still]', '|move|p1a: Snorlax|Curse|p1a: Snorlax', '|turn|3',
                  '|move|p1a: Snorlax|Rest|p1a: Snorlax', '|-status|p1a: Snorlax|slp|[from] move: Rest', '|turn|4',
                  '|cant|p1a: Snorlax|slp', '|move|p1a: Snorlax|Sleep Talk|p1a: Snorlax',
                  '|move|p1a: Snorlax|Body Slam|p2a: Absol|[from]Sleep Talk', '|turn|5',
                  '|move|p1a: Snorlax|Struggle|p2a: Absol', '|turn|6',
                  '|drag|p1a: Gengar|Gengar, L80|300/300', '|turn|7')
        r3 = request('Gengar', rqid=3)
        got = states(lines, r3)
        # Snorlax (benched by Whirlwind, no later request): 23 - 2 (Body Slam vs public Pressure) = 21; Curse 15; Rest 15;
        # Sleep Talk 15; the called Body Slam and Struggle uncharged.
        self.assertEqual(dict(zip(got['snorlax'].moves, got['snorlax'].pp)), {'bodyslam': 21, 'rest': 15, 'sleeptalk': 15, 'curse': 15})
        self.assertEqual(got['snorlax'].reason, '')
        self.assert_matches_carry(lines, r3)

    def test_typed_hidden_power_charges_the_generic_key(self):
        r1 = request('Snorlax', rqid=1)
        lines = START + (line(r1), '|switch|p1a: Gengar|Gengar, L80|300/300', '|turn|2')
        r2 = request('Gengar', rqid=2)
        lines += (line(r2), '|move|p1a: Gengar|Hidden Power|p2a: Absol', '|turn|3', '|drag|p1a: Starmie|Starmie, L80|300/300')
        r3 = request('Starmie', rqid=3)
        got = states(lines, r3)
        self.assertEqual(dict(zip(got['gengar'].moves, got['gengar'].pp))['hiddenpowerfire'], 23)
        self.assert_matches_carry(lines, r3)

    def test_spite_and_leppa_make_that_mons_pp_unknown_only(self):
        r1 = request('Snorlax', rqid=1)
        for event in ('|-activate|p1a: Snorlax|move: Spite|Body Slam|4', '|-enditem|p1a: Snorlax|Leppa Berry|[eat]'):
            lines = START + (line(r1), event, '|turn|2')
            r2 = request('Snorlax', {'bodyslam': 20}, rqid=2)
            got = states(lines + (line(r2), '|switch|p1a: Gengar|Gengar, L80|300/300', '|turn|3'), request('Gengar', rqid=3))
            # A later exact request snapshot restates it; benched right after it, the snapshot holds.
            self.assertEqual(got['snorlax'].pp[0], 20)
            got = states(lines + ('|switch|p1a: Gengar|Gengar, L80|300/300', '|turn|3'), request('Gengar', rqid=3))
            self.assertIsNone(got['snorlax'].pp)
            self.assertEqual(got['snorlax'].reason, 'OWN_PP_NONMOVE_CHANGE_UNAVAILABLE')
            self.assertEqual(got['snorlax'].moves, ('bodyslam', 'rest', 'sleeptalk', 'curse'))
            self.assertIsNotNone(got['gengar'].pp)

    def test_transformed_active_reads_the_real_set_and_its_base_pp(self):
        r1 = request('Snorlax', rqid=1)
        lines = START + (line(r1), '|switch|p1a: Ditto|Ditto, L80|300/300', '|turn|2')
        r2 = request('Ditto', rqid=2)
        lines += (line(r2), '|move|p1a: Ditto|Transform|p2a: Absol', '|-transform|p1a: Ditto|p2a: Absol', '|turn|3')
        copied = ('swordsdance', 'shadowball', 'bodyslam', 'protect')
        r3 = request('Ditto', {m: 5 for m in copied}, rqid=3, active_moves=copied)
        got = states(lines, r3)
        self.assertEqual(got['ditto'].moves, ('transform',))
        self.assertEqual(got['ditto'].pp, (15,))
        # After switch-out the base ledger is restored (own_pp_carry agrees for every member then).
        lines += (line(r3), '|switch|p1a: Snorlax|Snorlax, L80|400/400', '|turn|4')
        r4 = request('Snorlax', rqid=4)
        got = states(lines, r4)
        self.assertEqual((got['ditto'].moves, got['ditto'].pp), (('transform',), (15,)))
        self.assert_matches_carry(lines, r4)

    def test_transformed_without_history_is_unknown(self):
        copied = ('swordsdance', 'shadowball', 'bodyslam', 'protect')
        r = request('Ditto', {m: 5 for m in copied}, rqid=9, active_moves=copied)
        lines = ('|start', '|switch|p1a: Ditto|Ditto, L80|300/300', '|-transform|p1a: Ditto|p2a: Absol', line(r))
        got = {s.ident: s for s in own_move_states(lines, r, maximum_pp=maximum)}
        # The only request postdates Ditto's -transform, so it lists the copies: the real set is unknown (never the
        # copies), and the PP with it; the other members' sets are read from that request.
        self.assertEqual((got['ditto'].moves, got['ditto'].pp, got['ditto'].reason), ((), None, 'OWN_REAL_MOVESET_UNAVAILABLE'))
        self.assertEqual(got['snorlax'].moves, ('bodyslam', 'rest', 'sleeptalk', 'curse'))

    def test_fainted_member_and_forced_switch_request(self):
        r1 = request('Snorlax', rqid=1)
        lines = START + (line(r1), '|switch|p1a: Gengar|Gengar, L80|300/300', '|turn|2')
        r2 = request('Gengar', rqid=2)
        lines += (line(r2), '|move|p1a: Gengar|Thunderbolt|p2a: Absol', '|move|p2a: Absol|Shadow Ball|p1a: Gengar',
                  '|-damage|p1a: Gengar|0 fnt', '|faint|p1a: Gengar')
        forced = request(None, rqid=3, force=True, fainted=('Gengar',))
        lines += (line(forced), '|switch|p1a: Starmie|Starmie, L80|300/300', '|turn|3')
        r4 = request('Starmie', rqid=4, fainted=('Gengar',))
        got = states(lines, r4)
        self.assertEqual(dict(zip(got['gengar'].moves, got['gengar'].pp))['thunderbolt'], 23)
        self.assert_matches_carry(lines, r4)

    def test_incomplete_history_refuses_every_member(self):
        r = request('Snorlax', rqid=1)
        got = own_move_states((line(r),), r, maximum_pp=maximum)
        self.assertTrue(all(s.pp is None and s.reason == 'OWN_PP_COMPLETE_UNIQUE_START_HISTORY_REQUIRED' for s in got))
        self.assertEqual(got[0].moves, ('bodyslam', 'rest', 'sleeptalk', 'curse'))


if __name__ == '__main__':
    unittest.main()
