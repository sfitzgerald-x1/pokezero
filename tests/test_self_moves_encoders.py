"""V2 self_moves: our own real moves and exact PP on the self tokens, Python encoder and B2 (v4_encoder), off and on.

One authored owning-wire battle (owner p1, six own members) over a literal source built from the Showdown dex. Every own
move request carries the active's exact PP (the snapshots pokezero.own_moves reads):
- Snorlax: Body Slam into a foe whose Pressure is public (2 PP), Rest (1, never pressured), Sleep Talk (1) calling Body
  Slam (free), a Struggle line (free), then Curse (1) AFTER its last request and Roared out: snapshot + charge;
- Gengar: Hidden Power (generic on the wire, hiddenpowerfire in the request) into the Pressure foe (2), then fainted;
- Zapdos: active at the final request (exact from it); its Return is spelled return102 in the request;
- three members never sent out (full PP).
Off: the self tokens' move columns stay zero. On: Python and B2 write the same self-move cells (the representation in
ObservationFeatureMasks.self_moves' docstring), on-minus-off is identical in both, and the cells where the two encoders
already differ off (this fixture's authored HP lines) are the same set on. A Spite variant leaves Snorlax's PP unknown
(validity 0; Python, B2's wire refuses the authored line). The active's self-token PP equals its action-token
MOVE_PP_FRACTION.
"""
import importlib.util
import json
import os
import unittest
from unittest import mock

from _showdown_root import requires_showdown, showdown_root_str

SHOWDOWN = requires_showdown('the self-moves fixture compiles its dex and source from a pokemon-showdown checkout')

OWN = [dict(species='Snorlax', level=80, moves=['bodyslam', 'rest', 'sleeptalk', 'curse'], ability='Immunity', item='Leftovers'),
       dict(species='Gengar', level=80, moves=['thunderbolt', 'hiddenpowerfire', 'explosion', 'hypnosis'], ability='Levitate',
            item='Leftovers'),
       dict(species='Zapdos', level=80, moves=['thunderbolt', 'hiddenpowerice', 'return', 'rest'], ability='Pressure',
            item='Leftovers'),
       dict(species='Starmie', level=80, moves=['surf', 'icebeam', 'recover', 'thunderbolt'], ability='Natural Cure',
            item='Leftovers'),
       dict(species='Skarmory', level=80, moves=['spikes', 'whirlwind', 'toxic', 'protect'], ability='Keen Eye', item='Leftovers'),
       dict(species='Blissey', level=80, moves=['softboiled', 'seismictoss', 'toxic', 'protect'], ability='Natural Cure',
            item='Leftovers')]
FOE = [dict(species='Absol', level=80, moves=['swordsdance', 'shadowball', 'quickattack', 'roar'], ability='Pressure',
            item='Leftovers')]
NAMES = [m['species'] for m in OWN]
MAX = {'bodyslam': 24, 'rest': 16, 'sleeptalk': 16, 'curse': 16, 'thunderbolt': 24, 'hiddenpower': 24, 'explosion': 8,
       'hypnosis': 32, 'return': 32, 'surf': 24, 'icebeam': 16, 'recover': 32, 'spikes': 32, 'whirlwind': 32, 'toxic': 16,
       'protect': 16, 'softboiled': 16, 'seismictoss': 32}


def _key(m):
    return 'hiddenpower' if m.startswith('hiddenpower') else m


def request(active, pp=None, *, rqid, force=False, fainted=(), owner=0):
    rows = [{'ident': f'p{owner + 1}: {n}', 'details': f'{n}, L80', 'condition': '0 fnt' if n in fainted else (
             '400/400 slp' if n == 'Snorlax' and rqid >= 6 else '300/300'), 'active': n == active,
             'moves': ['return102' if m == 'return' else m for m in mon['moves']],
             'baseAbility': mon['ability'].lower().replace(' ', ''), 'item': mon['item'].lower(),
             'stats': {'atk': 150, 'def': 160, 'spa': 170, 'spd': 180, 'spe': 190}}
            for n, mon in zip(NAMES, OWN)]
    out = {'rqid': rqid, 'side': {'id': f'p{owner + 1}', 'name': 'Us', 'pokemon': rows}}
    if force:
        out['forceSwitch'] = [True]
    else:
        moves = OWN[NAMES.index(active)]['moves']
        out['active'] = [{'moves': [{'move': m, 'id': _key(m), 'pp': (pp or {}).get(m, MAX[_key(m)]), 'maxpp': MAX[_key(m)],
                                     'target': 'normal', 'disabled': False} for m in moves]}]
    return out


def history(*, spite=False, stop=None):
    """(the owner's lines with its requests, the final request); `stop` ends the history at request rqid `stop`."""
    r = lambda *a, **k: '|request|' + json.dumps(request(*a, **k))
    lines = ['|player|p1|Us||', '|player|p2|Them||', '|gen|3', '|tier|[Gen 3] Random Battle', '|start',
             '|switch|p1a: Snorlax|Snorlax, L80|100/100', '|switch|p2a: Absol|Absol, L80|100/100', '|turn|1',
             r('Snorlax', rqid=1), '|-ability|p2a: Absol|Pressure',
             '|move|p1a: Snorlax|Body Slam|p2a: Absol', '|-damage|p2a: Absol|80/100',
             '|move|p2a: Absol|Swords Dance|p2a: Absol', '|-boost|p2a: Absol|atk|2', '|turn|2',
             r('Snorlax', {'bodyslam': 22}, rqid=2), '|switch|p1a: Gengar|Gengar, L80|100/100',
             '|move|p2a: Absol|Quick Attack|p1a: Gengar', '|-immune|p1a: Gengar', '|turn|3',
             r('Gengar', rqid=3), '|move|p1a: Gengar|Hidden Power|p2a: Absol', '|-damage|p2a: Absol|60/100',
             '|move|p2a: Absol|Shadow Ball|p1a: Gengar', '|-supereffective|p1a: Gengar', '|-damage|p1a: Gengar|0 fnt',
             '|faint|p1a: Gengar', r(None, rqid=4, force=True, fainted=('Gengar',)),
             '|switch|p1a: Snorlax|Snorlax, L80|100/100', '|turn|4',
             r('Snorlax', {'bodyslam': 22}, rqid=5, fainted=('Gengar',)), '|move|p1a: Snorlax|Rest|p1a: Snorlax',
             '|-status|p1a: Snorlax|slp|[from] move: Rest', '|move|p2a: Absol|Swords Dance|p2a: Absol',
             '|-boost|p2a: Absol|atk|2', '|turn|5',
             r('Snorlax', {'bodyslam': 22, 'rest': 15}, rqid=6, fainted=('Gengar',)), '|cant|p1a: Snorlax|slp',
             '|move|p1a: Snorlax|Sleep Talk|p1a: Snorlax', '|move|p1a: Snorlax|Body Slam|p2a: Absol|[from]Sleep Talk',
             '|-damage|p2a: Absol|40/100', '|turn|6',
             r('Snorlax', {'bodyslam': 22, 'rest': 15, 'sleeptalk': 15}, rqid=7, fainted=('Gengar',)),
             '|move|p1a: Snorlax|Struggle|p2a: Absol', '|-damage|p2a: Absol|30/100',
             '|move|p1a: Snorlax|Curse|p1a: Snorlax', '|-unboost|p1a: Snorlax|spe|1',
             '|move|p2a: Absol|Roar|p1a: Snorlax', '|drag|p1a: Zapdos|Zapdos, L80|100/100', '|turn|7']
    if spite:lines.insert(lines.index('|move|p1a: Snorlax|Curse|p1a: Snorlax'), '|-activate|p1a: Snorlax|move: Spite|Curse|3')
    if stop is not None:
        at = next(i for i, x in enumerate(lines) if x.startswith('|request|') and json.loads(x[9:])['rqid'] == stop)
        return tuple(lines[:at + 1]), json.loads(lines[at][9:])
    final = request('Zapdos', rqid=8, fainted=('Gengar',))
    lines.append('|request|' + json.dumps(final))
    return tuple(lines), final


# (request slot, sorted move labels, (remaining, max) per move), at the final request.
EXPECTED = (
    (0, ('bodyslam', 'curse', 'rest', 'sleeptalk'), ((22, 24), (15, 16), (15, 16), (15, 16))),
    (1, ('explosion', 'hiddenpowerfire', 'hypnosis', 'thunderbolt'), ((8, 8), (22, 24), (32, 32), (24, 24))),
    (2, ('hiddenpowerice', 'rest', 'return', 'thunderbolt'), ((24, 24), (16, 16), (32, 32), (24, 24))),
    (3, ('icebeam', 'recover', 'surf', 'thunderbolt'), ((16, 16), (32, 32), (24, 24), (24, 24))),
)

_FIXTURE = {}


def fixture():
    if _FIXTURE:return _FIXTURE
    from pokezero.category_vocab import CategoryVocabulary
    from pokezero.dex import load_showdown_dex_cached, normalize_id, resolve_move_effect
    from gpu_engine.effect_probe_device import literal_source
    from pokezero.showdown import TRACKED_VOLATILES
    dex = load_showdown_dex_cached(showdown_root_str())
    source = literal_source([[OWN, FOE]], dex)
    labels = {'field', 'stats', 'pokemon:opponent', 'pokemon:self'}
    labels |= {'request_kind:' + n for n in ('none', 'wait', 'team_preview', 'force_switch', 'move', 'unknown')}
    labels |= {'weather:' + n for n in ('raindance', 'sunnyday', 'sandstorm', 'hail')}
    labels |= {'volatile:' + n for n in TRACKED_VOLATILES}
    labels |= {'type:' + n for n in ('Normal', 'Fighting', 'Flying', 'Poison', 'Ground', 'Rock', 'Bug', 'Ghost', 'Steel',
                                     'Fire', 'Water', 'Grass', 'Electric', 'Psychic', 'Ice', 'Dragon', 'Dark', '???')}
    labels |= {'status:' + n for n in ('none', 'slp', 'par', 'brn', 'frz', 'psn', 'tox', 'fnt')}
    for name, info in dex.moves.items():
        labels |= {'move:' + name, 'move_category:' + info.gen3_category, 'move_priority:' + str(info.priority)}
        effect, _, _ = resolve_move_effect(info, ())
        if effect:labels.add('move_effect:' + effect)
    labels |= {'species:' + normalize_id(n) for n in dex.species}
    for team in (OWN, FOE):
        for mon in team:
            labels |= {'belief:possible_move:' + normalize_id(m) for m in mon['moves']}
            labels |= {'ability:' + normalize_id(mon['ability']), 'belief:possible_ability:' + normalize_id(mon['ability'])}
            labels |= {'item:' + normalize_id(mon['item']), 'belief:possible_item:' + normalize_id(mon['item'])}
    _FIXTURE.update(dex=dex, source=source, vocab=CategoryVocabulary(tuple(sorted(labels))))
    return _FIXTURE


def python_state(*, spite=False, self_moves=True, stop=None):
    from dataclasses import replace
    from pokezero import showdown as s
    from pokezero.own_moves import own_move_states
    f = fixture()
    lines, final = history(spite=spite, stop=stop)
    state = s.normalize_for_player(s.parse_showdown_replay(list(lines), battle_id='self-moves-fixture'), player_id='p1',
        configured_showdown_slot='p1', format_id='gen3randombattle', set_source=f['source'])
    if self_moves:
        state = replace(state, self_move_states=own_move_states(lines, final, maximum_pp=lambda k: f['dex'].move_info(k).max_pp))
    return state


def python_observation(self_moves, *, spite=False, stop=None):
    from dataclasses import replace
    from pokezero import showdown as s
    f = fixture()
    masks = replace(s.DEFAULT_OBSERVATION_FEATURE_MASKS, exact_state=True, opponent_tendency_stats_block=True,
        feature_pack_last_move=True, transition_token_budget=0, self_moves=self_moves)
    return s.observation_from_player_state(python_state(spite=spite, self_moves=self_moves, stop=stop), category_vocab=f['vocab'],
        spec=s.V4_REPLAY_OBSERVATION_SPEC, dex=f['dex'], feature_masks=masks)


def move_columns():
    from pokezero import showdown as s
    numeric = [s.v4_numeric_index(s.NUMERIC_REVEALED_MOVE_COUNT), s.v4_numeric_index(s.NUMERIC_POSSIBLE_MOVE_COUNT)]
    numeric += [s.v4_numeric_index(s.NUMERIC_OPP_MOVE_PP_OFFSET + i) for i in range(16)]
    numeric += [s.v4_numeric_index(s.NUMERIC_OPP_MOVE_PP_VALID_OFFSET + i) for i in range(16)]
    return numeric, list(range(s.CATEGORY_BELIEF_MOVE_OFFSET, s.CATEGORY_BELIEF_MOVE_OFFSET + 16))


class SelfMovesMaskTests(unittest.TestCase):
    """The switch: declarations win, OBS_SELF_MOVES fills only an undeclared mask, serving refuses a contradiction."""

    def test_declared_wins_and_env_fills_undeclared(self):
        from pokezero.observation import ObservationFeatureMasks, SELF_MOVES_ENV, DEFAULT_OBSERVATION_FEATURE_MASKS
        self.assertEqual(SELF_MOVES_ENV, 'OBS_SELF_MOVES')
        self.assertIs(DEFAULT_OBSERVATION_FEATURE_MASKS.self_moves, False)
        for value, declared, want in (('', None, False), ('on', None, True), ('off', None, False), ('on', False, False),
                                      ('off', True, True), ('', True, True)):
            with mock.patch.dict(os.environ, {SELF_MOVES_ENV: value}):
                self.assertIs(ObservationFeatureMasks(self_moves=declared).self_moves, want, (value, declared))
        with mock.patch.dict(os.environ, {SELF_MOVES_ENV: 'yes'}):
            with self.assertRaisesRegex(ValueError, 'OBS_SELF_MOVES'):ObservationFeatureMasks()

    def test_serving_guard_refuses_a_contradiction(self):
        from pokezero.observation import require_declared_self_moves
        with mock.patch.dict(os.environ, {'OBS_SELF_MOVES': 'on'}):
            with self.assertRaisesRegex(ValueError, 'serving refuses'):require_declared_self_moves(False, context='t')
            require_declared_self_moves(True, context='t')
        with mock.patch.dict(os.environ, {'OBS_SELF_MOVES': 'off'}):
            with self.assertRaisesRegex(ValueError, 'serving refuses'):require_declared_self_moves(True, context='t')
        with mock.patch.dict(os.environ, {'OBS_SELF_MOVES': ''}):
            require_declared_self_moves(True, context='t'); require_declared_self_moves(False, context='t')

    def test_model_config_latch_is_silent_when_off(self):
        from dataclasses import replace
        from pokezero.neural_policy import TransformerPolicyConfig, feature_masks_from_model_config
        config = TransformerPolicyConfig.compact_category(category_vocab=('species:snorlax', 'pokemon:self'))
        self.assertNotIn('self_moves', config.to_dict())
        self.assertEqual(TransformerPolicyConfig.from_dict(config.to_dict()), config)
        with mock.patch.dict(os.environ, {'OBS_SELF_MOVES': ''}):
            self.assertIs(feature_masks_from_model_config(config).self_moves, False)
        on = replace(config, self_moves=True)
        self.assertIs(on.to_dict()['self_moves'], True)
        self.assertIs(TransformerPolicyConfig.from_dict(on.to_dict()).self_moves, True)
        with mock.patch.dict(os.environ, {'OBS_SELF_MOVES': 'off'}):
            # A declared on checkpoint is on without (and despite) the measurement override.
            self.assertIs(feature_masks_from_model_config(on).self_moves, True)

    def test_cache_payload_records_only_when_on(self):
        from pokezero.dataset import _feature_masks_payload
        from pokezero.observation import ObservationFeatureMasks
        self.assertNotIn('self_moves', _feature_masks_payload(ObservationFeatureMasks(self_moves=False)))
        self.assertIs(_feature_masks_payload(ObservationFeatureMasks(self_moves=True))['self_moves'], True)


@SHOWDOWN
class PythonSelfMovesTests(unittest.TestCase):
    def test_off_leaves_self_move_columns_zero_and_on_refuses_without_states(self):
        import numpy as np
        from dataclasses import replace
        from pokezero import showdown as s
        off = python_observation(False)
        numeric, cats = move_columns()
        self.assertFalse(np.asarray(off.numeric_features)[1:7][:, numeric].any())
        self.assertFalse(np.asarray(off.categorical_ids)[1:7][:, cats].any())
        f = fixture()
        with self.assertRaisesRegex(ValueError, 'self_move_states'):
            s.observation_from_player_state(python_state(self_moves=False), category_vocab=f['vocab'],
                spec=s.V4_REPLAY_OBSERVATION_SPEC, dex=f['dex'],
                feature_masks=replace(s.DEFAULT_OBSERVATION_FEATURE_MASKS, self_moves=True, transition_token_budget=0))

    def test_on_writes_real_moves_and_exact_pp(self):
        import numpy as np
        from pokezero import showdown as s
        f = fixture()
        off, on = python_observation(False), python_observation(True)
        numeric, cats = move_columns()
        n_off, n_on = np.asarray(off.numeric_features, np.float32), np.asarray(on.numeric_features, np.float32)
        c_off, c_on = np.asarray(off.categorical_ids), np.asarray(on.categorical_ids)
        self.assertTrue(all(1 <= t <= 6 and c in numeric for t, c in np.argwhere(n_off != n_on)))
        self.assertTrue(all(1 <= t <= 6 and c in cats for t, c in np.argwhere(c_off != c_on)))
        pp = [s.v4_numeric_index(s.NUMERIC_OPP_MOVE_PP_OFFSET + i) for i in range(16)]
        valid = [s.v4_numeric_index(s.NUMERIC_OPP_MOVE_PP_VALID_OFFSET + i) for i in range(16)]
        for slot, moves, fractions in EXPECTED:
            row, cat = n_on[1 + slot], c_on[1 + slot]
            self.assertEqual([int(x) for x in cat[cats]], [f['vocab'].encode('belief:possible_move:' + m) for m in moves] + [0] * 12)
            self.assertEqual((row[s.v4_numeric_index(s.NUMERIC_REVEALED_MOVE_COUNT)],
                              row[s.v4_numeric_index(s.NUMERIC_POSSIBLE_MOVE_COUNT)]), (4.0, 4.0))
            self.assertEqual(list(row[valid]), [1.0] * 4 + [0.0] * 12)
            self.assertEqual([float(x) for x in row[pp]], [float(np.float32(a / b)) for a, b in fractions] + [0.0] * 12, (slot, moves))

    def test_active_self_pp_equals_its_action_token_pp(self):
        """At request 7 (Snorlax active after Body Slam x2 vs Pressure, Rest, Sleep Talk) the self token's exact PP is the
        request's own PP, which the action tokens show as MOVE_PP_FRACTION."""
        import numpy as np
        from pokezero import showdown as s
        on = python_observation(True, stop=7)
        num = np.asarray(on.numeric_features, np.float32); cat = np.asarray(on.categorical_ids)
        pp = [s.v4_numeric_index(s.NUMERIC_OPP_MOVE_PP_OFFSET + i) for i in range(4)]
        vocab = fixture()['vocab']
        active = 1 + NAMES.index('Snorlax')
        by_label = {int(cat[active, s.CATEGORY_BELIEF_MOVE_OFFSET + k]): float(num[active, pp[k]]) for k in range(4)}
        self.assertEqual(sorted(by_label.values()), sorted(float(np.float32(x)) for x in (22 / 24, 1.0, 15 / 16, 15 / 16)))
        for token in range(13, 17):
            primary = int(cat[token, s.CATEGORY_PRIMARY])
            name = next(n for n in OWN[0]['moves'] if vocab.encode('move:' + _key(n)) == primary)
            self.assertEqual(by_label[vocab.encode('belief:possible_move:' + name)],
                             float(num[token, s.v4_numeric_index(s.NUMERIC_MOVE_PP_FRACTION)]), name)

    def test_spite_leaves_that_mons_pp_unknown(self):
        import numpy as np
        from pokezero import showdown as s
        on = np.asarray(python_observation(True, spite=True).numeric_features, np.float32)
        valid = [s.v4_numeric_index(s.NUMERIC_OPP_MOVE_PP_VALID_OFFSET + i) for i in range(4)]
        pp = [s.v4_numeric_index(s.NUMERIC_OPP_MOVE_PP_OFFSET + i) for i in range(4)]
        self.assertFalse(on[1, valid].any() or on[1, pp].any())
        self.assertEqual(on[1, s.v4_numeric_index(s.NUMERIC_REVEALED_MOVE_COUNT)], 4.0)
        self.assertTrue(on[2, valid].all())


@SHOWDOWN
@unittest.skipUnless(importlib.util.find_spec('jax'), 'requires isolated JAX')
class B2SelfMovesTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import jax
        from gpu_engine import public_belief, v4_encoder
        from pokezero.recorded_mirror_roots import CHAMPION_CONTRACT
        f = fixture()
        cls.jax, cls.e = jax, v4_encoder
        masks = dict(CHAMPION_CONTRACT['feature_masks'])
        compiled = public_belief.compile_source(f['source'])
        with mock.patch.dict(os.environ, {'OBS_SELF_MOVES': ''}):
            cls.catalogs = {on: v4_encoder.compile_encoder(compiled, f['dex'], f['vocab'], checkpoint_sha256='a' * 64,
                feature_masks=dict(masks, self_moves=True) if on else masks) for on in (False, True)}

    def adapter(self, on, *, spite=False):
        from pokezero.recorded_encoder_runner import JaxRecordedAdapter
        lines, final = history(spite=spite)
        adapter = JaxRecordedAdapter(self.catalogs[on])
        adapter.reset(stream_id=1, battle_id='self-moves-fixture')
        adapter.feed(tuple(x for x in lines if not x.startswith('|request|')))
        return adapter, lines, final

    def b2(self, on, *, spite=False):
        import numpy as np
        adapter, lines, final = self.adapter(on, spite=spite)
        adapter.own_lines = lines
        return {k: np.asarray(v) for k, v in adapter.encode(final, owner_side=0).items()}

    def test_fold_binding_identity_and_declared_mask(self):
        import numpy as np
        off, on = self.catalogs[False], self.catalogs[True]
        self.assertIsNone(off.self_moves); self.assertFalse(off.masks.self_moves); self.assertTrue(on.masks.self_moves)
        # self_moves is an encode-only switch: the fold-state binding is the same, so one fold serves both.
        self.assertTrue(np.array_equal(np.asarray(off.binding), np.asarray(on.binding)))
        self.assertEqual(self.e.ENCODE_ONLY, frozenset({'self_moves'})); self.assertEqual(self.e.OPTIONAL_KEYS, frozenset({'self_moves'}))
        with self.assertRaisesRegex(ValueError, 'without defaults'):self.e.Masks.from_checkpoint(dict(self_moves=True))
        from pokezero.recorded_mirror_roots import CHAMPION_CONTRACT
        base = dict(CHAMPION_CONTRACT['feature_masks'])
        with mock.patch.dict(os.environ, {'OBS_SELF_MOVES': 'on'}):
            self.assertTrue(self.e.Masks.from_checkpoint(base).self_moves)        # undeclared: the override fills it
            self.assertFalse(self.e.Masks.from_checkpoint(dict(base, self_moves=False)).self_moves)   # declared wins
        with mock.patch.dict(os.environ, {'OBS_SELF_MOVES': 'off'}):
            self.assertTrue(self.e.Masks.from_checkpoint(dict(base, self_moves=True)).self_moves)

    def test_auto_log_reads_the_requests_it_encoded(self):
        """Without own_lines the adapter's own log (fed lines plus the requests it encodes) is the history: here it never
        saw the earlier requests, so members who moved since have no exact PP, and only those (never a guess)."""
        import numpy as np
        from pokezero import showdown as s
        adapter, lines, final = self.adapter(True)
        out = adapter.encode(final, owner_side=0)
        valid = [s.v4_numeric_index(s.NUMERIC_OPP_MOVE_PP_VALID_OFFSET + i) for i in range(4)]
        num = np.asarray(out['numeric_features'])
        self.assertFalse(num[1, valid].any())     # Snorlax moved with no snapshot in this log
        self.assertTrue(num[3, valid].all())      # Zapdos: the final request is its snapshot

    def test_python_and_b2_agree_off_and_on(self):
        """The Spite variant is Python-only: B2's wire refuses this authored -activate line (a fixture limit); B2's
        PP-unknown path is covered by test_auto_log_reads_the_requests_it_encoded, through the same own_move_states."""
        import numpy as np
        numeric, cats = move_columns()
        for spite in (False,):
            b2 = {on: self.b2(on, spite=spite) for on in (False, True)}
            py = {on: python_observation(on, spite=spite) for on in (False, True)}
            for name, columns in (('numeric_features', numeric), ('categorical_ids', cats)):
                p = {on: np.asarray(getattr(py[on], name), np.float32 if name == 'numeric_features' else None) for on in (False, True)}
                q = {on: b2[on][name] for on in (False, True)}
                self.assertTrue(np.array_equal(p[False] != q[False], p[True] != q[True]), (spite, name))
                self.assertTrue(np.array_equal(p[True][1:7][:, columns], q[True][1:7][:, columns]), (spite, name))
                self.assertTrue(np.array_equal(p[True] - p[False], q[True] - q[False]), (spite, name))
            for name in ('attention_mask', 'legal_action_mask'):
                for on in (False, True):
                    self.assertTrue(np.array_equal(np.asarray(getattr(py[on], name)), b2[on][name]), (spite, name, on))

    def test_on_differs_from_off_only_on_self_move_columns(self):
        import numpy as np
        off, on = self.b2(False), self.b2(True)
        numeric, cats = move_columns()
        diff = np.argwhere(off['numeric_features'] != on['numeric_features'])
        self.assertTrue(len(diff) and all(1 <= t <= 6 and c in numeric for t, c in diff), diff.tolist())
        diff = np.argwhere(off['categorical_ids'] != on['categorical_ids'])
        self.assertTrue(len(diff) and all(1 <= t <= 6 and c in cats for t, c in diff), diff.tolist())


if __name__ == '__main__':
    unittest.main()
