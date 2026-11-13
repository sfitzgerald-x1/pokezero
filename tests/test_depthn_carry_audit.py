"""CPU tests of scripts/depthn_carry_audit.py on synthetic rows: the enumeration and its citations, own moves from the owning wire,
the root-carry markers, the (a)/(b)/(c)/(d) measures and the decision rule, the turn-2 record (prior masses, lane labels from
child states, the learner_max row), and the capture hooks under the real DepthNLiveSearch and re-pricer main (no GPU, no pricer)."""
from contextlib import ExitStack
import hashlib
import io
import json
import math
import os
from pathlib import Path
import re
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from scripts import depthn_carry_audit as A
from tests import _engine_since as since

ROOT = Path(__file__).resolve().parents[1]
VOCAB = ['_none', 'hyperbeam', 'earthquake', 'rest', 'sleeptalk', 'wrap', 'toxic', 'protect', 'solarbeam']
NAME = lambda i: None if i <= 0 else VOCAB[i]


def softmax(logits, actions):
    top = max(logits[a] for a in actions);w = [math.exp(logits[a] - top) for a in actions];t = math.fsum(w)
    return [x / t for x in w]


# Enumeration ----------------------------------------------------------------------------------------------------------------------------
def citations(text):
    """(path, first line) of every gpu_engine file:line citation in `text` (bare names resolve under gpu_engine/)."""
    out = []
    for name, line in re.findall(r'((?:gpu_engine/)?[\w-]+\.(?:py|cjs)):(\d+)', text):
        out.append((ROOT / (name if '/' in name else 'gpu_engine/' + name), int(line)))
    return out


# Every gpu_engine file:line(-line) citation in scripts/depthn_carry_audit.py (the module text: rows, notes and docstring), including the
# continuation ranges after it ("x.py:611-612, 825", "x.py:641-643 and 851-852"), keyed with its path under gpu_engine/, and a token its
# cited lines must contain (an identifier, refusal code or function name there); a token starting with '^' must be on the range's first
# line. A shift of the cited code then fails here instead of drifting silently (the lead, 2026-11-13); a new citation needs its anchor,
# a removed one loses it.
CITATION_ANCHORS = {
    'gpu_engine/baton.py:16': 'batonpass',
    'gpu_engine/constructor_worlds.py:482': 'OPPONENT_RECHARGE_UNREPRESENTED',
    'gpu_engine/constructor_worlds.py:843': 'def fix_recharge',
    'gpu_engine/effects.py:175': 'def stall_success',
    'gpu_engine/effects.py:46': 'destinybond',
    'gpu_engine/encore.py:14-20': '^def from_metadata',
    'gpu_engine/encore.py:33-52': '^def apply',
    'gpu_engine/engine_device_observation.py:290-302': "own['moves']",
    'gpu_engine/engine_request_source.py:261-265': '^_unsafe_history',
    'gpu_engine/engine_request_source.py:341': 'charge_move',
    'gpu_engine/engine_request_source.py:341-342': '^encore',
    'gpu_engine/engine_request_source.py:380': 'LOCK',
    'gpu_engine/entry.py:50': 'spikes',
    'gpu_engine/focus_punch.py:15-19': 'focuspunch',
    'gpu_engine/future_state.py:155': 'lockedmove',
    'gpu_engine/future_state.py:476': 'RECHARGE_UNREPRESENTED',
    'gpu_engine/items.py:27-28': 'trick: jax.Array',
    'gpu_engine/linked_traps.py:12-14': 'meanlook',
    'gpu_engine/linked_traps.py:12-25': 'meanlook',
    'gpu_engine/phase_d_live_worlds.py:278-279': 'PUBLIC_TRANSFORM_NOT_MATERIALIZED',
    'gpu_engine/phase_d_live_worlds.py:344-345': "'perish1'",
    'gpu_engine/phase_d_live_worlds.py:355-356': 'OWN_VOLATILE_OR_LATENT_CLOCK_UNAVAILABLE',
    'gpu_engine/phase_d_live_worlds.py:362': 'DESTINY_BOND',
    'gpu_engine/phase_d_protocol_constructor.py:175': 'load_gen3_randbat_source_cached',
    'gpu_engine/phase_d_protocol_world.py:107-109': '^PUBLIC_ITEM_SWAP_NOT_SUPPORTED',
    'gpu_engine/phase_d_protocol_world.py:178-196': '^_defer_yawn',
    'gpu_engine/phase_d_protocol_world.py:199-214': '^_perish_counts',
    'gpu_engine/phase_d_protocol_world.py:286-297': '^def _write_held_counts',
    'gpu_engine/phase_d_protocol_world.py:430-431': 'approximate_partial_trap_turns=False',
    'gpu_engine/phase_d_protocol_world.py:443': 'map_spec',
    'gpu_engine/public_active.py:217-233': '^PERISH_START',
    'gpu_engine/public_active.py:46': 'perish',
    'gpu_engine/public_refresh_stream.py:17': 'TRANSFORM',
    'gpu_engine/request_source_rules.cjs:57-64': 'altersRequest',
    'gpu_engine/request_source_rules.cjs:59-64': 'TrapPokemon',
    'gpu_engine/request_source_rules.cjs:64': "'encore'",
    'gpu_engine/request_source_rules.cjs:65-66': "^'transform'",
    'gpu_engine/request_source_rules.cjs:67': 'recharge',
    'gpu_engine/residual_order.py:95': 'POSITIONS',
    'gpu_engine/sleep_talk.py:16-22': 'sleeptalk',
    'gpu_engine/solar_beam.py:1': 'Solar Beam',
    'gpu_engine/solar_beam.py:14-15': 'solarbeam',
    'gpu_engine/state.py:14': 'LEECH_SEED',
    'gpu_engine/state.py:146-148': '^def usable_moves',
    'gpu_engine/state.py:16-17': '^FOCUS_PUNCH',
    'gpu_engine/state.py:21': 'DESTINY_BOND',
    'gpu_engine/state.py:24-92': '^class BattleState',
    'gpu_engine/state.py:36-39': 'status_counter',
    'gpu_engine/state.py:38': 'sleep_skipped',
    'gpu_engine/state.py:43': 'choice_move',
    'gpu_engine/state.py:47-48': 'encore_counter',
    'gpu_engine/state.py:49-50': '^charge_move',
    'gpu_engine/state.py:57': 'protect_counter',
    'gpu_engine/state.py:58': 'substitute_hp',
    'gpu_engine/state.py:63': 'trap_source',
    'gpu_engine/state.py:75': 'spikes',
    'gpu_engine/state.py:76-77': '^wish_counter',
    'gpu_engine/state.py:78-79': 'weather_counter',
    'gpu_engine/state.py:84': 'truant_loaf',
    'gpu_engine/state.py:85': 'item_knocked',
    'gpu_engine/state.py:87': 'recharge_counter',
    'gpu_engine/state.py:89': 'perish_counter',
    'gpu_engine/state.py:90': 'yawn_counter',
    'gpu_engine/state.py:91-92': '^partial_trap_counter',
    'gpu_engine/sustain.py:22-27': 'leechseed',
    'gpu_engine/transition.py:1680-1689': '^_trick=',
    'gpu_engine/transition.py:1690-1701': '_white_herb_pair',
    'gpu_engine/transition.py:1729-1737': '^_perish_used',
    'gpu_engine/transition.py:1742-1750': '^_yawn_try',
    'gpu_engine/transition.py:1913-1922': '^seed_target',
    'gpu_engine/transition.py:1950-1955': '^_wrap_set',
    'gpu_engine/transition.py:2527-2547': "^POSITIONS['wrap']",
    'gpu_engine/transition.py:2548-2576': "^POSITIONS['yawn']",
    'gpu_engine/transition.py:2558-2562': '^_yawn_clause',
    'gpu_engine/transition.py:2596-2609': '^_perish_continues',
    'gpu_engine/transition.py:2676-2680': 'truant_loaf=work.truant_loaf',
    'gpu_engine/transition.py:611-612': '^charge_moves',
    'gpu_engine/transition.py:641-643': 'initial_loaf = (table.entry.truant',
    'gpu_engine/transition.py:82-114': '^def tables_from_metadata',
    'gpu_engine/transition.py:825': 'charge_locked',
    'gpu_engine/transition.py:832-836': '~DESTINY_BOND&0xffffffff',
    'gpu_engine/transition.py:840-845': '^_recharge_turn',
    'gpu_engine/transition.py:851-852': 'loafing=table.entry.truant',
    'gpu_engine/transition.py:852': 'loafing=table.entry.truant',
    'gpu_engine/v5_hypothetical_state.py:111': 'DESTINY_BOND',
    'gpu_engine/v5_hypothetical_state.py:113': 'perish_counter',
}
CITATION = re.compile(r'((?:gpu_engine/)?[\w-]+\.(?:py|cjs)):(\d+)(?:-(\d+))?')
CITATION_TAIL = re.compile(r'(?:,\s*|\s+and\s+)(\d+)(?:-(\d+))?(?![\w.-])')


def cited_ranges(text):
    """{key: (path, first, last)} of every citation in `text` and of its continuation ranges (key: path under gpu_engine/ plus
    first[-last], as CITATION_ANCHORS)."""
    out = {}
    for match in CITATION.finditer(text):
        name = match.group(1);path = name if '/' in name else 'gpu_engine/' + name
        ranges = [(int(match.group(2)), int(match.group(3) or match.group(2)))];pos = match.end()
        while (tail := CITATION_TAIL.match(text, pos)):ranges.append((int(tail.group(1)), int(tail.group(2) or tail.group(1))));pos = tail.end()
        for a, b in ranges:out[f'{path}:{a}' + (f'-{b}' if b != a else '')] = (ROOT / path, a, b)
    return out


class EnumerationTests(unittest.TestCase):
    def test_audited_effects_are_unrepresented_pool_moves_with_evidence(self):
        self.assertEqual([e.id for e in A.EFFECTS], ['recharge', 'partial_trap', 'perish_count', 'drowsy', 'destiny_bond', 'transform', 'trick_lock'])
        for e in A.EFFECTS:
            with self.subTest(effect=e.id):
                self.assertFalse(e.represented);self.assertTrue(e.moves and e.pool_species and e.evidence and e.mechanism)
                self.assertTrue(any('gpu_engine/' in x for x in e.evidence))   # an engine citation
        self.assertEqual({e.id for e in A.EFFECTS if e.lock_class}, {'recharge', 'partial_trap', 'transform'})
        self.assertEqual({e.id: e.engine_since for e in A.EFFECTS if e.engine_since}, dict(recharge='8b212312', destiny_bond='286a5ffe', trick_lock='8b212312',
                         partial_trap='f3e71663', perish_count='f3e71663', drowsy='f3e71663'))
        self.assertTrue(all(not e.pool_species for e in A.NOT_IN_POOL))
        self.assertTrue(all(e.represented for e in A.REPRESENTED))
        moves = {m for e in A.NOT_IN_POOL for m in e.moves}
        self.assertTrue({'outrage', 'thrash', 'petaldance', 'uproar', 'rollout', 'iceball', 'bide', 'rage', 'fly', 'dig', 'dive', 'bounce',
            'skullbash', 'razorwind', 'skyattack'} <= moves)

    @unittest.skipUnless(since.IN_GIT, since.SKIP_REASON)
    def test_every_engine_since_is_a_merge_head_contains(self):
        """An effect is in-tree only on an engine at or after its merge: every engine_since is a GitHub pull request merge that changed
        gpu_engine/ and that HEAD contains (tests/_engine_since.problem)."""
        self.assertFalse(since.shallow(), 'a shallow clone cannot check ancestry')
        for e in A.EFFECTS:
            if e.engine_since:
                with self.subTest(effect=e.id, engine_since=e.engine_since):self.assertEqual(since.problem(e.engine_since), '', e.id)

    def test_no_placeholder_merge_sha_remains(self):
        """The placeholder merge SHA is gone from the audit (its engine_since fields and its evidence text) once the merge exists."""
        self.assertNotIn(since.PLACEHOLDER, Path(A.__file__).read_text())

    def test_every_citation_is_anchored_and_its_anchor_is_in_the_cited_lines(self):
        cited = cited_ranges(Path(A.__file__).read_text())
        self.assertEqual(sorted(set(cited) - set(CITATION_ANCHORS)), [], 'a citation without an anchor in CITATION_ANCHORS')
        self.assertEqual(sorted(set(CITATION_ANCHORS) - set(cited)), [], 'an anchor whose citation is gone')
        for key, (path, first, last) in sorted(cited.items()):
            with self.subTest(citation=key):
                lines = path.read_text().splitlines();self.assertLessEqual(last, len(lines))
                token = CITATION_ANCHORS[key];where = lines[first - 1] if token.startswith('^') else '\n'.join(lines[first - 1:last])
                self.assertIn(token.lstrip('^'), where, f'{key} no longer names {token!r}')

    def test_every_engine_citation_names_an_existing_line(self):
        rows = A.enumeration()['rows'];found = 0
        for r in rows:
            for path, line in citations(' '.join(r['evidence']) + ' ' + r['lock_class'] + ' ' + r['note']):
                with self.subTest(effect=r['effect'], path=path.name, line=line):
                    self.assertTrue(path.exists());self.assertLessEqual(line, len(path.read_text().splitlines()))
                found += 1
        self.assertGreater(found, 30)

    def test_the_lock_rule_and_state_fields_cited_are_what_they_say(self):
        rules = (ROOT / 'gpu_engine/request_source_rules.cjs').read_text().splitlines()
        self.assertIn('move.flags.recharge', rules[66]);self.assertIn("'transform'", rules[64]);self.assertIn('TrapPokemon', rules[59])
        source = (ROOT / 'gpu_engine/engine_request_source.py').read_text().splitlines()
        self.assertIn('LOCK', source[379]);self.assertIn('charge_move', source[340])
        state = (ROOT / 'gpu_engine/state.py').read_text().lower()
        # an effect the engine models in-tree (engine_since) names its state; the others still have none
        words = dict(recharge='recharge', perish_count='perish', drowsy='yawn', destiny_bond='destiny', partial_trap='partial', transform='transform')
        for e in A.EFFECTS:
            if e.id in words:(self.assertIn if e.engine_since else self.assertNotIn)(words[e.id], state, e.id)
        traps = (ROOT / 'gpu_engine/linked_traps.py').read_text().splitlines()[12]
        self.assertIn('meanlook', traps);self.assertNotIn('wrap', traps)
        transition = (ROOT / 'gpu_engine/transition.py').read_text()
        for e in A.EFFECTS:
            for move in set(e.moves) & {'hyperbeam', 'wrap', 'perishsong', 'yawn', 'destinybond'}:
                if not e.engine_since:self.assertNotIn(f'"{move}"', transition, move)

    def test_pool_check_against_a_sets_file(self):
        sets = {}
        for e in A.EFFECTS + A.REPRESENTED:
            for s in e.pool_species:sets.setdefault(s, dict(level=80, sets=[dict(role='r', movepool=[])]))['sets'][0]['movepool'] += [m for m in e.moves]
        sets['snorlax'] = dict(level=80, sets=[dict(role='r', movepool=['Body Slam', 'Curse'])])
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'sets.json';path.write_text(json.dumps(sets))
            pool = A.pool_from_sets(path)
            self.assertEqual(A.pool_check(pool), []);self.assertEqual(pool['bodyslam'], ['snorlax'])
            sets['dragonite'] = dict(level=80, sets=[dict(role='r', movepool=['Outrage'])]);sets['shuckle']['sets'][0]['movepool'].remove('wrap')
            sets['ditto']['sets'].append(dict(role='x', movepool=['Hyper Beam']))
            path.write_text(json.dumps(sets));problems = A.pool_check(A.pool_from_sets(path))
        self.assertEqual(len(problems), 3)
        self.assertTrue(any(p.startswith('locked_move') for p in problems) and any(p.startswith('partial_trap') for p in problems)
            and any(p.startswith('recharge') for p in problems))


# Own moves and the prefix ------------------------------------------------------------------------------------------------------------------
def request_line(moves, *, species='Slaking', force=False, side='p1'):
    mon = lambda i: dict(ident=f'{side}: M{i}', details=f'{species if i == 0 else "Mon" + str(i)}, L80, M', condition='100/100', active=i == 0,
        moves=['x'])
    raw = dict(side=dict(id=side, name='a', pokemon=[mon(i) for i in range(6)]))
    if force:raw['forceSwitch'] = [True]
    else:raw['active'] = [dict(moves=[dict(move=m, id=A.normalize(m), pp=5, maxpp=5, disabled=False) for m in moves])]
    return '|request|' + json.dumps(raw)


class RequestTests(unittest.TestCase):
    def test_request_facts(self):
        f = A.request_facts(request_line(['Hyper Beam', 'Earthquake', 'Shadow Ball']))
        self.assertEqual(f, dict(moves=['hyperbeam', 'earthquake', 'shadowball', None], species='slaking'))
        self.assertEqual(A.request_facts(request_line([], force=True))['moves'], [None] * 4)
        self.assertEqual(A.request_facts('|request|not json'[:9] + '{}')['moves'], [None] * 4)
        self.assertEqual(A.request_facts(None), dict(moves=[None] * 4, species=None))

    def test_own_moves_prefer_the_row_then_the_wire(self):
        wire = dict(request=dict(moves=['wrap', None, None, None], species='shuckle'))
        row = dict(root=dict(own_moves=['move:hyperbeam', 'move:rest', 'switch:x']))
        self.assertEqual(A.own_moves_of(row, wire), (['hyperbeam', 'rest', None, None], None, 'row'))
        self.assertEqual(A.own_moves_of(dict(root={}), wire), (['wrap', None, None, None], 'shuckle', 'owning_wire'))
        self.assertEqual(A.own_moves_of(dict(root=None), None), ([None] * 4, None, 'missing'))


class CarriedTests(unittest.TestCase):
    def carried(self, lines, owner='p1'):
        return {k: (v['own'], v['opp']) for k, v in A.carried_effects(lines, owner).items() if v['own'] or v['opp']}

    def test_markers_set_and_clear(self):
        self.assertEqual(self.carried(['|move|p2a: Slaking|Hyper Beam|p1a: X', '|-mustrecharge|p2a: Slaking']), {'recharge': (False, True)})
        self.assertEqual(self.carried(['|-mustrecharge|p2a: Slaking', '|cant|p2a: Slaking|recharge']), {})
        wrap = ['|move|p2a: Shuckle|Wrap|p1a: X', '|-activate|p1a: X|move: Wrap|[of] p2a: Shuckle']
        self.assertEqual(self.carried(wrap), {'partial_trap': (True, False)})
        self.assertEqual(self.carried(wrap + ['|-end|p1a: X|Wrap|[partiallytrapped]']), {})
        self.assertEqual(self.carried(wrap + ['|switch|p2a: Snorlax|Snorlax, L80|100/100']), {})   # the source left
        self.assertEqual(self.carried(wrap, owner='p2'), {'partial_trap': (False, True)})
        perish = ['|-start|p1a: X|perish3|[silent]', '|-start|p2a: Y|perish3|[silent]']
        self.assertEqual(self.carried(perish), {'perish_count': (True, True)})
        self.assertEqual(self.carried(perish + ['|switch|p1a: Z|Z, L80|100/100']), {'perish_count': (False, True)})
        self.assertEqual(self.carried(perish + ['|-start|p2a: Y|perish0', '|faint|p2a: Y', '|-start|p1a: X|perish0']), {})
        yawn = ['|-start|p1a: X|move: Yawn|[of] p2a: Swalot']
        self.assertEqual(self.carried(yawn), {'drowsy': (True, False)});self.assertEqual(self.carried(yawn + ['|-status|p1a: X|slp']), {})
        bond = ['|move|p2a: Gengar|Destiny Bond|p2a: Gengar', '|-singlemove|p2a: Gengar|Destiny Bond']
        self.assertEqual(self.carried(bond), {'destiny_bond': (False, True)})
        self.assertEqual(self.carried(bond + ['|move|p2a: Gengar|Shadow Ball|p1a: X']), {})
        self.assertEqual(self.carried(['|-transform|p2a: Ditto|p1a: X']), {'transform': (False, True)})
        trick = ['|-activate|p1a: X|move: Trick|[of] p2a: Y', '|-item|p2a: Y|Choice Band|[from] move: Trick', '|-item|p1a: X|Leftovers|[from] move: Trick']
        self.assertEqual(self.carried(trick), {'trick_lock': (False, True)})
        self.assertEqual(self.carried(['|-item|p2a: Y|Choice Band']), {})   # a reveal, not a Trick


# Measures -------------------------------------------------------------------------------------------------------------------------------
LOGITS = [1., .5, 0., -1., .2, 0., 0., 0., 0.]
LEGAL = [True, True, True, False, True, False, False, False, False]


def world(i, lanes, *, logits=None, kept=None, species='slaking', priced=True):
    kept = [x is not None for x in lanes] if kept is None else kept
    return dict(world=i, priced=priced, opponent_requested=any(kept), kept=kept, root_peer_logits=logits or [0.] * 9, peer=dict(lanes=lanes),
        opponent=dict(active_species=species), grid=None)


OPP_PLAIN = ['move:earthquake', 'move:rest', None, None, 'switch:x', None, None, None, None]
OPP_BEAM = ['move:hyperbeam', 'move:rest', None, None, 'switch:x', None, None, None, None]


def row(ordinal, decision, *, own=('hyperbeam', 'earthquake', 'rest', 'protect'), search=1, greedy=1, played=None, status='COMPLETE',
        worlds=None, logits=LOGITS, legal=LEGAL, prefix=None, turn=3):
    return dict(ordinal=ordinal, decision_index=decision, prefix_sha256=prefix or f'{ordinal}-{decision}',
        replay=dict(status=status, action_index=search if status == 'COMPLETE' else None, turn=turn),
        live=dict(greedy_action_index=greedy, played_action_index=greedy if played is None else played),
        root=None if status != 'COMPLETE' else dict(root_logits=list(logits), root_legal=list(legal), own_moves=list(own)),
        worlds=[world(0, OPP_PLAIN), world(1, OPP_PLAIN)] if worlds is None else worlds)


class MeasureTests(unittest.TestCase):
    def test_mass_is_a_masked_softmax(self):
        self.assertAlmostEqual(A.mass([0.] * 9, [True] * 4 + [False] * 5, [0]), .25)
        self.assertAlmostEqual(A.mass(LOGITS, LEGAL, [0, 3]), softmax(LOGITS, [0, 1, 2, 4])[0])   # lane 3 is illegal
        self.assertEqual(A.mass(None, LEGAL, [0]), 0.);self.assertEqual(A.mass(LOGITS, [False] * 9, [0]), 0.)

    def test_root_measures(self):
        m = A.root_measures(row(1, 0, search=0, greedy=1, worlds=[world(0, OPP_BEAM, logits=[3.] + [0.] * 8), world(1, OPP_PLAIN)]), None, None)
        f = m['effects']['recharge']
        self.assertTrue(m['priced'] and f['a_own'] and f['a_opp']);self.assertEqual((f['opp_worlds'], f['opp_species']), ([0], ['slaking']))
        self.assertAlmostEqual(f['b1_own'], softmax(LOGITS, [0, 1, 2, 4])[0])
        self.assertAlmostEqual(f['b1_opp'], softmax([3.] + [0.] * 8, [0, 1, 4])[0])   # over the kept replies
        self.assertTrue(f['b1'] and f['b'] and f['c'] and f['search_is']);self.assertFalse(f['greedy_is'])
        self.assertEqual((f['b_source'], f['b2']), ('proxy', None))
        self.assertFalse(any(m['effects']['partial_trap'][k] for k in ('a_own', 'a_opp', 'b', 'c')))
        same = A.root_measures(row(1, 1, search=0, greedy=0), None, None)['effects']['recharge']
        self.assertFalse(same['c']);self.assertTrue(same['search_is'] and same['greedy_is'])   # greedy plays it too: no override into it
        illegal = A.root_measures(row(1, 2, own=('rest', 'earthquake', 'protect', 'hyperbeam')), None, None)['effects']['recharge']
        self.assertFalse(illegal['a_own']);self.assertEqual(illegal['b1_own'], 0.)   # lane 3 is not legal at the root
        refused = A.root_measures(row(1, 3, status='REFUSED'), None, None)
        self.assertFalse(refused['priced']);self.assertFalse(refused['effects']['recharge']['c'])
        live = A.root_measures(row(1, 4, search=1, greedy=1, played=0), None, None)['effects']['recharge']
        self.assertTrue(live['c_live']);self.assertFalse(live['c'])

    def test_greedy_falls_back_to_the_root_prior(self):
        r = row(1, 0);r['live'] = None
        self.assertEqual(A.greedy_of(r), (0, 'root_logits'))
        self.assertEqual(A.greedy_of(row(1, 0, greedy=4)), (4, 'live'))

    def test_turn2_entry_replaces_the_proxy(self):
        entry = dict(worlds=[dict(mass=dict(own={}, opp={'hyperbeam': [[4, .03], [27, .2]]}), own_max={}),
            dict(mass=dict(own={'hyperbeam': [[4, .01]]}, opp={}), own_max={'hyperbeam': [4, 5]})])
        self.assertEqual(A.turn2_masses(entry, ('hyperbeam',)), (.01, .2, 2))
        f = A.root_measures(row(1, 0), None, entry)['effects']['recharge']
        self.assertTrue(f['b2'] and f['b'] and f['b_max']);self.assertEqual((f['b_source'], f['b_max_cells']), ('turn2', 2))
        quiet = dict(worlds=[dict(mass=dict(own={}, opp={'hyperbeam': [[4, .04]]}), own_max={})])
        f = A.root_measures(row(1, 0), None, quiet)['effects']['recharge']
        self.assertTrue(f['b1']);self.assertFalse(f['b2'] or f['b'])   # the turn-2 record decides (b); b1 stays as the proxy column

    def test_unlabelled_turn2_cells_never_clear_b(self):
        """A CAPTURED record with unlabelled cells cannot say (b) is false: without a hit (b) falls back to the proxy, marked; a hit
        still decides; the summary counts the rows and cells."""
        partial = lambda masses: dict(worlds=[dict(mass=dict(own={}, opp=masses), own_max={}, unlabelled=3), dict(mass=dict(own={}, opp={}), own_max={})])
        f = A.root_measures(row(1, 0), None, partial({}))['effects']['recharge']   # Hyper Beam holds ~39% of our root prior: the proxy trips
        self.assertEqual((f['b2'], f['b2_unlabelled_fallback'], f['b'], f['b_source']), (None, True, f['b1'], 'proxy_unlabelled'))
        self.assertTrue(f['b'])
        hit = A.root_measures(row(1, 0), None, partial({'hyperbeam': [[4, .3]]}))['effects']['recharge']
        self.assertEqual((hit['b2'], hit['b2_unlabelled_fallback'], hit['b_source']), (True, False, 'turn2'))
        rows = [row(1, 0), row(1, 1)]
        turn2 = {(1, 0): dict(prefix_sha256='1-0', replay_action_index=1, turn2=dict(status='CAPTURED', **partial({}))),
            (1, 1): dict(prefix_sha256='1-1', replay_action_index=1, turn2=dict(status='CAPTURED', worlds=[dict(mass=dict(own={}, opp={}), own_max={})]))}
        s = A.audit(iter(rows), None, turn2)
        self.assertEqual((s['turn2_join']['unlabelled_rows'], s['turn2_join']['unlabelled_cells']), (1, 3))
        r = s['effects']['recharge']['b']
        self.assertEqual((r['turn2_rows'], r['proxy_rows'], r['unlabelled_fallback_rows']), (1, 1, 1))
        self.assertIn('unlabelled turn-2 cells 1 (3 cells', A.markdown(s))

    def test_decision_rule_and_guard(self):
        rows = [row(1, i, own=('rest', 'earthquake', 'toxic', 'protect')) for i in range(98)]
        rows += [row(2, 0, search=0, greedy=1), row(2, 1, search=0, greedy=1)]   # two overrides into Hyper Beam: 2 / 100
        rows += [row(3, 0, status='REFUSED'), row(1, 0)]   # unpriced; a duplicate
        s = A.audit(iter(rows), None)
        self.assertEqual((s['priced_roots'], s['counts']['rows'], s['counts']['duplicate_rows']), (100, 101, 1))
        r = s['effects']['recharge']
        self.assertEqual((r['c']['roots'], r['coverage']['roots'], r['decision'], r['provisional']), (2, 2, 'NAMED_RESIDUAL', True))
        self.assertEqual(r['b']['source'], 'proxy')
        self.assertEqual(s['recharge_guard']['search_recharge_not_greedy'], 2);self.assertEqual(s['recharge_guard']['search_recharge'], 2)
        rows.append(row(2, 2, search=0, greedy=1))
        r = A.audit(iter(rows), None)['effects']['recharge']
        self.assertEqual((r['coverage']['roots'], r['decision']), (3, 'ESCALATE'))   # 3 / 101 > 2%
        self.assertEqual(A.audit(iter([]), None)['effects']['recharge']['decision'], 'NO_PRICED_ROOTS')

    def test_turn2_join(self):
        rows = [row(1, i, own=('rest', 'earthquake', 'toxic', 'protect'), search=1, greedy=1) for i in range(4)]
        captured = lambda masses: dict(status='CAPTURED', worlds=[dict(mass=dict(own={}, opp=masses), own_max={})])
        turn2 = {(1, 0): dict(prefix_sha256='1-0', replay_action_index=1, turn2=captured({'wrap': [[0, .5]]})),
            (1, 1): dict(prefix_sha256='other', replay_action_index=1, turn2=captured({'wrap': [[0, .5]]})),
            (1, 2): dict(prefix_sha256='1-2', replay_action_index=2, turn2=captured({})),
            (1, 3): dict(prefix_sha256='1-3', replay_action_index=1, turn2=dict(status='ERROR'))}
        s = A.audit(iter(rows), None, turn2)
        self.assertEqual(s['turn2_join'], dict(joined=2, proxy=2, prefix_mismatch=1, not_captured=1, replay_action_differs=1, records=4))
        w = s['effects']['partial_trap']
        self.assertEqual((w['b']['roots'], w['b']['source'], w['b_turn2']['opp'], w['b_turn2']['rows']), (1, 'turn2+proxy', 1, 2))
        self.assertEqual((w['coverage']['roots'], w['decision'], w['provisional']), (1, 'ESCALATE', True))


# The turn-2 record ------------------------------------------------------------------------------------------------------------------------
OWN2 = [1., .2, -.5, 0., .3, 0., 0., 0., 0.]   # turn-2 logits of the learner at cell (0, 4)
OPP2 = [.4, .1, 0., 0., 0., -.2, 0., 0., 0.]


def frame(R=1):
    """A depth-2 frame: cells (0, 4) and (3, 0) expanded in both worlds; at (3, 0) the learner waits. Learner slots at (0, 4):
    actions 0, 4, 1, 2 (ranked); opponent slots 0, 1, 5. The leaf rows make slot 0 (action 0) learner_max's row in world 0 and
    slot 1 (action 4) in world 1."""
    k = 9;z = lambda *s, d=np.float32: np.zeros(s, d)
    f = SimpleNamespace(depth=2, k_own=(k,), k_opp=(k,), pre=z(R, 2, d=np.int32), t1_status=z(R, 2, 9, 9, d=np.int8), t1_value=z(R, 2, 9, 9),
        t1_own_requested=z(R, 2, 9, 9, d=bool), t1_peer_requested=z(R, 2, 9, 9, d=bool), t1_own_logits=z(R, 2, 9, 9, 9), t1_peer_logits=z(R, 2, 9, 9, 9),
        t1_own_actions=np.full((R, 2, 9, 9, k), -1, np.int32), t1_opp_actions=np.full((R, 2, 9, 9, k), -1, np.int32),
        leaf_status=z(R, 2, 81, k, k, d=np.int8), leaf_value=z(R, 2, 81, k, k))
    for r in range(R):
        for j in (0, 1):
            for a, b in ((0, 4), (3, 0)):
                f.t1_status[r, j, a, b] = 2;f.t1_peer_requested[r, j, a, b] = True
                f.t1_peer_logits[r, j, a, b] = OPP2;f.t1_opp_actions[r, j, a, b, :3] = [0, 1, 5]
            f.t1_own_requested[r, j, 0, 4] = True;f.t1_own_logits[r, j, 0, 4] = OWN2;f.t1_own_actions[r, j, 0, 4, :4] = [0, 4, 1, 2]
            f.t1_own_actions[r, j, 3, 0, 0] = 0   # a waiting learner: [0, -1, ...]
            f.leaf_status[r, j, 4, :4, :3] = 4;f.leaf_value[r, j, 4, :4, :3] = .1
            f.leaf_value[r, j, 4, 0 if j == 0 else 1, :3] = .5
            f.leaf_status[r, j, 27, 0, :3] = 4
    return f


def children():
    """P1 children of world rows 0 and 1: lane 0 -> cell 4, lane 1 -> cell 27. World row 0, cell 4: the learner (seat 0) knows
    Hyper Beam, Earthquake, Rest, Sleep Talk; the opponent Wrap, Toxic, Protect. At cell 27 the opponent charges Solar Beam. World
    row 1, cell 4: the opponent has no usable move (Struggle)."""
    n = 2 * 81;lane_cell = np.full((2, 81), 81, np.int32);lane_cell[:, 0] = 4;lane_cell[:, 1] = 27
    moves = np.zeros((n, 2, 4), np.int32);usable = np.zeros((n, 2, 4), bool);charge = np.zeros((n, 2), np.int32)
    for row in (0, 1, 81, 82):
        moves[row, 0] = [1, 2, 3, 4];moves[row, 1] = [5, 6, 7, 0];usable[row, 0] = True;usable[row, 1, :3] = True
    charge[1, 1] = 8;usable[81, 1] = False
    return A.Children(lane_cell, np.zeros((n, 2), np.int32), moves, charge, usable)


class Turn2RecordTests(unittest.TestCase):
    def test_children_rows_follow_lane_cell(self):
        c = children()
        self.assertEqual((c.row(0, 4), c.row(0, 27), c.row(1, 4), c.row(1, 27), c.row(0, 5)), (0, 1, 81, 82, -1))
        self.assertEqual(A.child_lanes(c, 0, 0, NAME), ['hyperbeam', 'earthquake', 'rest', 'sleeptalk'])
        self.assertEqual(A.child_lanes(c, 1, 1, NAME), ['solarbeam', None, None, None])   # the charge move on lane 0
        self.assertEqual(A.child_lanes(c, 81, 1, NAME), ['struggle', None, None, None])

    def test_world_masses_labels_and_learner_max(self):
        f = frame();c = children();watch = {'hyperbeam', 'wrap'}
        w0 = A.world_turn2(f, 0, 0, 0, 0, c, NAME, watch)
        as32 = lambda x: [float(v) for v in np.asarray(x, np.float32)]   # the frame's logits are float32
        own = softmax(as32(OWN2), [0, 4, 1, 2]);opp = softmax(as32(OPP2), [0, 1, 5])
        self.assertEqual((w0['expanded'], w0['unlabelled'], w0['unsolved']), (2, 0, 0))
        self.assertEqual(w0['mass']['own'], {'hyperbeam': [[4, own[0]]]})
        self.assertEqual(w0['mass']['opp'], {'wrap': [[4, opp[0]]]})   # at cell 27 lane 0 is Solar Beam, not Wrap
        self.assertEqual(w0['own_max'], {'hyperbeam': [4]})   # the waiting learner at cell 27 has no row
        self.assertEqual(w0['labels'], dict(own=['earthquake', 'hyperbeam', 'rest'], opp=['solarbeam', 'toxic', 'wrap']))
        w1 = A.world_turn2(f, 0, 1, 1, 0, c, NAME, watch)
        # Struggle at cell 4 (world row 1 charges nothing at cell 27: Wrap there); learner_max plays the switch (action 4) at cell 4
        self.assertEqual((w1['mass']['opp'], w1['own_max']), ({'wrap': [[27, opp[0]]]}, {}))
        self.assertIn('struggle', w1['labels']['opp'])
        # learner_max's row: its opponent-prior average is depthn_backups' backed-up value of the cell
        from gpu_engine import depthn_backups as backups
        for j, slot in ((0, 0), (1, 1)):
            q = math.fsum(p * float(f.leaf_value[0, j, 4, slot, x]) for x, p in enumerate(opp))
            self.assertAlmostEqual(backups.node_value(f, 0, j, 1, 4), q, places=12)
            self.assertEqual(A.learner_max_action(f, 0, j, 0, 4), [0, 4][slot])
        self.assertIsNone(A.learner_max_action(f, 0, 0, 3, 0))
        unlabelled = A.world_turn2(f, 0, 0, 0, 0, None, NAME, watch)
        self.assertEqual((unlabelled['unlabelled'], unlabelled['mass']), (2, dict(own={}, opp={})))
        f.pre[0, 1] = 3;self.assertEqual(A.world_turn2(f, 0, 1, 1, 0, c, NAME, watch)['expanded'], 0)

    def test_record_status(self):
        roots = (dict(status='REFUSED'), dict(status='PRICED', rows=(0,)))
        capture = SimpleNamespace(priced=[(SimpleNamespace(roots=roots), frame())], children=[children()], turn2_notes=[])
        rec = A.turn2_record(capture, 0, None, ('hyperbeam',), name=NAME)
        self.assertEqual((rec['status'], rec['depth'], len(rec['worlds']), rec['worlds'][1]['world']), ('CAPTURED', 2, 2, 1))
        self.assertEqual(A.turn2_record(SimpleNamespace(priced=[], children=[], turn2_notes=[]), 0, None, ())['status'], 'NOT_PRICED')
        self.assertEqual(A.turn2_record(SimpleNamespace(priced=[(SimpleNamespace(roots=()), frame())], children=[], turn2_notes=[]), 0, None, ())['status'],
            'PRICED_ROOTS')
        broken = A.turn2_record(SimpleNamespace(priced=[(SimpleNamespace(roots=roots), object())], children=[], turn2_notes=[]), 0, None, (), name=NAME)
        self.assertEqual(broken['status'], 'ERROR')


class ChildFactsTests(unittest.TestCase):
    def test_child_facts_read_the_flat_children(self):
        import jax
        import jax.numpy as jnp
        from gpu_engine.state import empty_state

        def one(*, active=(0, 0), moves=None, pp=None, charge=None):
            s = empty_state(1)
            s = s._replace(active=jnp.asarray([active], jnp.int32))
            for side, slot in enumerate(active):
                s = s._replace(moves=s.moves.at[0, side, slot].set(jnp.asarray(moves[side])), pp=s.pp.at[0, side, slot].set(jnp.asarray(pp[side])))
                if charge and charge[side]:s = s._replace(charge_move=s.charge_move.at[0, side, slot].set(charge[side]))
            return s
        states = [one(active=(2, 0), moves=([1, 2, 3, 4], [5, 6, 0, 0]), pp=([5] * 4, [5, 0, 0, 0])),
            one(active=(0, 1), moves=([1, 2, 3, 4], [8, 6, 0, 0]), pp=([5] * 4, [5, 5, 0, 0]), charge=(0, 8)),
            one(active=(0, 0), moves=([1, 2, 3, 4], [5, 6, 7, 0]), pp=([0] * 4, [5, 5, 5, 0]))]
        stacked = jax.tree.map(lambda *xs: jnp.stack(xs), *states)   # P1's flat layout: [N, 1, ...]
        lane_cell = np.full((1, 81), 81, np.int32);lane_cell[0, :3] = [4, 27, 30]
        c = A.child_facts(dict(states=stacked, lane_cell=jnp.asarray(lane_cell)))
        self.assertEqual(c.active.tolist(), [[2, 0], [0, 1], [0, 0]])
        self.assertEqual(c.moves[0].tolist(), [[1, 2, 3, 4], [5, 6, 0, 0]])
        self.assertEqual(c.usable[0].tolist(), [[True] * 4, [True, False, False, False]])
        self.assertEqual(c.charge.tolist(), [[0, 0], [0, 8], [0, 0]])
        self.assertEqual(c.row(0, 27), 1)
        self.assertEqual([A.child_lanes(c, i, s, NAME) for i, s in ((1, 1), (2, 0), (0, 1))],
            [['solarbeam', None, None, None], ['struggle', None, None, None], ['wrap', 'toxic', None, None]])


# Capture hooks ---------------------------------------------------------------------------------------------------------------------------
class CaptureHookTests(unittest.TestCase):
    def test_call_sites(self):
        self.assertTrue(A.check_turn2_call_sites());self.assertTrue(A.check_reprice_hooks())

        def bound(self, todo):   # a search holding its own reference to price would escape the wrapper
            from gpu_engine.constructor_worlds import price
            return price(None, None)
        with self.assertRaises(RuntimeError):A.check_turn2_call_sites(search=bound)

    def test_price_wrapper_under_the_live_search(self):
        from tests import test_depthn_rr_reprice as RT
        from gpu_engine import constructor_worlds as cw, device_depthn_pricer as dn
        requests = [RT.body(0), RT.body(1), RT.body(2)]
        recorder = SimpleNamespace(current=None);Capture = A.capture_class(recorder);program = dn.DepthNPricer.__dict__['_program']
        with ExitStack() as stack:
            RT.pricing(stack, still_clock=True);price = cw.price
            plain = RT.fake_search(RT.FakeConstructor(refuse={1}));without = [plain.process_many([r])[0] for r in requests]
            wrapped = RT.fake_search(RT.FakeConstructor(refuse={1}));within, captures = [], []
            for r in requests:
                capture = Capture();self.assertIs(recorder.current, capture)
                with capture.installed():
                    self.assertIsNot(cw.price, price);self.assertIsNot(dn.DepthNPricer.__dict__['_program'], program)
                    within.append(wrapped.process_many([r])[0])
                self.assertIs(cw.price, price);self.assertIs(dn.DepthNPricer.__dict__['_program'], program)
                captures.append(capture)
        self.assertEqual(json.dumps(within, sort_keys=True), json.dumps(without, sort_keys=True))
        self.assertEqual([len(c.priced) for c in captures], [1, 0, 1])   # a batch without requests is never priced
        batch, frame = captures[0].priced[0]
        self.assertEqual([x['rows'] for x in batch.roots], [tuple(range(8))]);self.assertIsInstance(frame, SimpleNamespace)
        self.assertEqual(len(captures[0].decided), 1)   # rr's own capture still runs underneath
        capture = Capture()
        with self.assertRaises(KeyError):
            with capture.installed():raise KeyError('boom')
        self.assertIs(dn.DepthNPricer.__dict__['_program'], program)

    def test_program_wrapper_reads_p1_children(self):
        import jax
        import jax.numpy as jnp
        from gpu_engine import device_depthn_pricer as dn
        from gpu_engine.state import empty_state
        pricer = object.__new__(dn.DepthNPricer)
        pricer.programs, pricer.compiles, pricer.evaluator_suffix, pricer._world_suffix = {}, (), (), ()
        state = empty_state(1);state = state._replace(moves=state.moves.at[0, 1, 0].set(jnp.asarray([5, 6, 0, 0])), pp=state.pp.at[0, 1, 0].set(5))
        stacked = jax.tree.map(lambda x: jnp.stack([x, x]), state)
        lane_cell = jnp.full((1, 81), 81, jnp.int32).at[0, :2].set(jnp.asarray([4, 5]))

        def build():return jax.jit(lambda x: dict(states=jax.tree.map(lambda y: y + 0 * x.astype(y.dtype), stacked), lane_cell=lane_cell + 0 * x))
        recorder = SimpleNamespace(current=None);capture = A.capture_class(recorder)()
        x = jnp.asarray(0, jnp.int32)
        with capture.installed():
            out = pricer._program(('P1', 1), build, x)(x)
            other = pricer._program(('select', 1), build, x)
        self.assertEqual(len(capture.children), 1);self.assertEqual(capture.children[0].row(0, 5), 1)
        self.assertEqual(A.child_lanes(capture.children[0], 1, 1, NAME), ['wrap', 'toxic', None, None])
        self.assertEqual(np.asarray(out['lane_cell']).tolist(), np.asarray(lane_cell).tolist())   # the output is returned unchanged
        self.assertFalse(callable(getattr(other, '__wrapped__', None)));self.assertEqual(len(pricer.programs), 2)
        self.assertNotIn('_program', vars(pricer))

    def test_reprice_argv(self):
        with tempfile.TemporaryDirectory() as tmp:
            header = Path(tmp) / 'header-2.json'
            header.write_text(json.dumps(dict(argv=['--run-directory', 'r', '--arm', 'DN2W16', '--sample', '160', '--sample-seed', '7', '--shard', '2',
                '--shards', '4', '--out', '/old/out', '--tree-commit=abc', '--accept-tree', 'old'])))
            argv = A.reprice_argv(header, ['--max-decisions', '3'], '/new', 'h' * 40)
            self.assertEqual(argv, ['--run-directory', 'r', '--arm', 'DN2W16', '--sample', '160', '--sample-seed', '7', '--shard', '2', '--shards', '4',
                '--max-decisions', '3', '--accept-tree', 'h' * 40, '--out', '/new/replay'])
            self.assertEqual(A.reprice_argv(None, ['--accept-tree', 'mine'], '/o', 'h')[-4:], ['--accept-tree', 'mine', '--out', '/o/replay'])
            with self.assertRaises(ValueError):A.reprice_argv(None, ['--out', 'x'], '/o', 'h')
            with patch('sys.stdout', new_callable=io.StringIO) as out:
                self.assertEqual(A.main(['capture', '--out', str(Path(tmp) / 'cap'), '--like', str(header), '--dry-run']), 0)
            printed = json.loads(out.getvalue())
            self.assertEqual(printed['watch'], ['destinybond', 'hyperbeam', 'perishsong', 'transform', 'trick', 'wrap', 'yawn'])
            self.assertFalse((Path(tmp) / 'cap').exists())

    def test_the_describe_wrapper_passes_the_re_pricers_arguments_through(self):
        """The re-pricer's describe signature since 5a035eb7: describe(capture, peer_lanes, vocabulary, peer_inputs, (ordinal,
        decision_index)), positionally or by keyword; the wrapper hands every argument through and the recorder sees the detail."""
        from scripts import depthn_rr_reprice as rr
        calls, seen = [], []

        def describe(capture, peer_lanes, vocabulary, peer_inputs=None, key=None):
            calls.append((capture, peer_lanes, vocabulary, peer_inputs, key));return dict(detail=len(calls))
        recorder = SimpleNamespace(current=None, described=lambda capture, detail, vocabulary: seen.append((capture, detail, vocabulary)))
        with patch.object(rr, 'describe', describe):
            with A.turn2_hooks(recorder):
                self.assertIsNot(rr.describe, describe)
                first = rr.describe('c', 'p', 'v', 'inputs', (1001, 3));second = rr.describe('c', 'p', 'v', peer_inputs='i', key=(1001, 4))
                third = rr.describe('c', 'p', 'v')   # the older three-argument call still works
            self.assertIs(rr.describe, describe)
        self.assertEqual(calls, [('c', 'p', 'v', 'inputs', (1001, 3)), ('c', 'p', 'v', 'i', (1001, 4)), ('c', 'p', 'v', None, None)])
        self.assertEqual((first, second, third), (dict(detail=1), dict(detail=2), dict(detail=3)))
        self.assertEqual(seen, [('c', dict(detail=1), 'v'), ('c', dict(detail=2), 'v'), ('c', dict(detail=3), 'v')])

    def test_capture_main_writes_turn2_records(self):
        """The re-pricer's own main (its test harness: fake search, fake pricing) under turn2_hooks, cw.price returning a frame."""
        from tests import test_depthn_rr_reprice as RT
        from scripts import depthn_rr_reprice as rr
        from gpu_engine import constructor_worlds as cw
        original = RT.pricing

        def pricing(stack, **keywords):
            original(stack, **keywords);stack.enter_context(patch.object(cw, 'price', return_value=(frame(R=8), None)))
        names = (rr.Capture, rr.describe, rr.decision_row)
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, {'JAX_DEFAULT_MATMUL_PRECISION': 'highest'}), patch.object(RT, 'pricing', pricing):
            out = Path(tmp) / 'cap';out.mkdir();recorder = A.Recorder(out, ('hyperbeam', 'wrap'))
            with A.turn2_hooks(recorder):
                code, rows, *_ = RT.MainTests('test_options').replay(tmp)
            recorder.close()
            records = [json.loads(x) for x in (out / 'turn2.jsonl').read_text().splitlines()]
        self.assertEqual((rr.Capture, rr.describe, rr.decision_row), names)
        self.assertEqual(code, 0)
        self.assertEqual([(r['ordinal'], r['decision_index']) for r in records], [(r['ordinal'], r['decision_index']) for r in rows])
        self.assertEqual([r['prefix_sha256'] for r in records], [r['prefix_sha256'] for r in rows])
        self.assertEqual([r['turn2']['status'] for r in records], ['CAPTURED', 'CAPTURED', 'CAPTURED', 'NOT_PRICED'])
        first = records[0]['turn2']
        self.assertEqual((len(first['worlds']), first['owner'], first['watch']), (16, 0, ['hyperbeam', 'wrap']))
        self.assertEqual((first['worlds'][0]['expanded'], first['worlds'][0]['unlabelled']), (2, 2))   # no P1 ran: nothing labelled
        self.assertEqual(records[0]['replay_action_index'], rows[0]['replay']['action_index'])
        self.assertTrue(all('turn2' not in r for r in rows))   # the re-pricer's rows are its own


# Report end to end ------------------------------------------------------------------------------------------------------------------------
def write_capture(path, decisions, *, owner='p1'):
    """A valid owning capture (phase_d_owning_capture's chain): decisions = [(protocol lines before the request, request line)]."""
    from pokezero import phase_d_owning_capture as owning
    blobs, state, prefix = [], dict(prior='0' * 64, lines=0, requests=0), hashlib.sha256()

    def emit(**fields):
        blob = owning.encoded(dict(schema=owning.SCHEMA, sequence=len(blobs), previous_sha256=state['prior'], **fields))
        blobs.append(blob);state['prior'] = hashlib.sha256(blob).hexdigest()
    emit(event='BEGIN', cell=dict(ordinal=1001), declaration_sha256='d' * 64, namespace='ns', battle_id='b', owner=owner, seed=7, stream_id=5)
    for i, (lines, request) in enumerate(decisions):
        delta = [f'|turn|{i + 1}', *lines, request]
        for x in delta:prefix.update(x.encode() + b'\n')
        emit(event='WIRE', generation=i + 1, start=state['lines'], end=state['lines'] + len(delta), prefix_sha256=prefix.hexdigest(), lines=delta)
        state['lines'] += len(delta);state['requests'] += 1
        emit(event='ROOT', decision_index=i, turn=i + 1, generation=i + 1, request_count=state['requests'], prefix_sha256=prefix.hexdigest(),
            own_request_line=request)
    path.write_bytes(b''.join(b + b'\n' for b in blobs))
    return {r['decision_index']: r for r in owning.read_capture(path)}


class ReportTests(unittest.TestCase):
    def test_report_main(self):
        beam = request_line(['Hyper Beam', 'Earthquake', 'Rest', 'Protect'])
        plain = request_line(['Body Slam', 'Earthquake', 'Rest', 'Protect'], species='Snorlax')
        with tempfile.TemporaryDirectory() as tmp:
            wire = Path(tmp) / 'owning-wire';wire.mkdir()
            roots = write_capture(wire / '1001.jsonl', [([], beam), (['|-mustrecharge|p2a: Slaking'], plain),
                (['|cant|p2a: Slaking|recharge'], beam), ([], plain)])
            rows = [row(1001, 0, search=0, greedy=1), row(1001, 1, search=1, greedy=1), row(1001, 2, search=0, greedy=0), row(1001, 3, status='REFUSED')]
            for r in rows:r['prefix_sha256'] = roots[r['decision_index']]['prefix_sha256'];(r['root'] or {}).pop('own_moves', None)
            rows.append(dict(row(1002, 0, search=0, greedy=1), prefix_sha256='x'));rows[-1]['root'].pop('own_moves')   # no capture: moves unknown
            shard = Path(tmp) / 'shard-0';shard.mkdir()
            (shard / 'rows-0.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in rows))
            t2 = Path(tmp) / 't2';t2.mkdir()
            entry = dict(ordinal=1001, decision_index=1, prefix_sha256=rows[1]['prefix_sha256'], replay_action_index=1,
                turn2=dict(status='CAPTURED', worlds=[dict(mass=dict(own={}, opp={'hyperbeam': [[40, .3]]}), own_max={})]))
            (t2 / 'turn2.jsonl').write_text(json.dumps(entry) + '\n')
            out = Path(tmp) / 'report'
            with patch('sys.stdout', new_callable=io.StringIO):
                code = A.main(['report', '--rows', str(Path(tmp) / 'shard-0'), '--owning-wire', str(wire), '--turn2', str(t2), '--out', str(out)])
            summary = json.loads((out / 'summary.json').read_text());text = (out / 'summary.md').read_text()
        self.assertEqual(code, 0)
        self.assertEqual((summary['priced_roots'], summary['own_move_sources']), (4, dict(owning_wire=3, missing=1)))
        self.assertEqual(set(summary['owning_wire_errors']), {'1002'})   # no capture for battle 1002
        r = summary['effects']['recharge']
        self.assertEqual((r['a']['own'], r['c']['roots'], r['c']['search_is'], r['c']['greedy_is']), (2, 1, 2, 1))
        self.assertEqual((r['b_turn2']['roots'], r['b_turn2']['opp'], r['b']['source']), (1, 1, 'turn2+proxy'))
        self.assertEqual((r['d']['opp'], r['d']['own'], r['d']['known']), (1, 0, 3))   # decision 1's prefix: the opponent must recharge
        self.assertEqual(summary['turn2_join']['joined'], 1)
        self.assertEqual(summary['enumeration']['pool_check'], None)
        # decision 0 by (c), decision 1 by its turn-2 record, decision 2 by the proxy (Hyper Beam holds ~39% of our root prior)
        self.assertEqual((r['coverage']['roots'], r['decision'], r['provisional']), (3, 'ESCALATE', True))
        self.assertIn('| recharge |', text);self.assertIn('## Recharge guard', text);self.assertIn('ESCALATE (PROVISIONAL)', text)
        self.assertEqual(summary['recharge_guard']['search_recharge_not_greedy'], 1)


if __name__ == '__main__':
    unittest.main()
