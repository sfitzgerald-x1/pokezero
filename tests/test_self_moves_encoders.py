"""V2 self_moves: own moves and PP on the self tokens, Python encoder and B2 (v4_encoder), flag off and on.

One authored owner-view battle (owner p1) over a literal source built from the Showdown dex:
- Snorlax (active at the request): Body Slam into a foe whose Pressure is public (2 PP), Rest (1 PP; self-targeted, never
  pressured), Sleep Talk (1 PP) calling Body Slam (the callee is not charged), a Struggle line (never charged), Curse
  unused;
- Gengar (fainted bench): Hidden Power (the protocol's generic name; the request names it hiddenpowerfire) into the
  Pressure foe (2 PP), then fainted;
- Zapdos (bench, never sent out): nothing used; its Return is spelled return102 in the request (resolved power).
Off: the self tokens' move columns stay zero (and the whole encoding equals the default masks'). On: Python and B2 agree on
every cell, and the self tokens carry exactly the representation in ObservationFeatureMasks.self_moves' docstring.
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
            item='Leftovers')]
FOE = [dict(species='Absol', level=80, moves=['swordsdance', 'shadowball', 'quickattack', 'hiddenpowerfighting'],
            ability='Pressure', item='Leftovers')]
LINES = (
    '|switch|p1a: Snorlax|Snorlax, L80|100/100',
    '|switch|p2a: Absol|Absol, L80|100/100',
    '|-ability|p2a: Absol|Pressure',
    '|turn|1',
    '|move|p1a: Snorlax|Body Slam|p2a: Absol',
    '|-damage|p2a: Absol|80/100',
    '|move|p2a: Absol|Swords Dance|p2a: Absol',
    '|-boost|p2a: Absol|atk|2',
    '|turn|2',
    '|switch|p1a: Gengar|Gengar, L80|100/100',
    '|move|p2a: Absol|Quick Attack|p1a: Gengar',
    '|-immune|p1a: Gengar',
    '|turn|3',
    '|move|p1a: Gengar|Hidden Power|p2a: Absol',
    '|-damage|p2a: Absol|60/100',
    '|move|p2a: Absol|Shadow Ball|p1a: Gengar',
    '|-supereffective|p1a: Gengar',
    '|-damage|p1a: Gengar|0 fnt',
    '|faint|p1a: Gengar',
    '|switch|p1a: Snorlax|Snorlax, L80|100/100',
    '|turn|4',
    '|move|p1a: Snorlax|Rest|p1a: Snorlax',
    '|-status|p1a: Snorlax|slp|[from] move: Rest',
    '|move|p2a: Absol|Swords Dance|p2a: Absol',
    '|-boost|p2a: Absol|atk|2',
    '|turn|5',
    '|cant|p1a: Snorlax|slp',
    '|move|p1a: Snorlax|Sleep Talk|p1a: Snorlax',
    '|move|p1a: Snorlax|Body Slam|p2a: Absol|[from]Sleep Talk',
    '|-damage|p2a: Absol|40/100',
    '|move|p2a: Absol|Swords Dance|p2a: Absol',
    '|turn|6',
    '|move|p1a: Snorlax|Struggle|p2a: Absol',
    '|-damage|p2a: Absol|30/100',
    '|turn|7',
)
# (request slot, sorted move ids, expected remaining fractions as (remaining, max)); max = floor(base PP * 8 / 5).
EXPECTED = (
    (0, ('bodyslam', 'curse', 'rest', 'sleeptalk'), ((22, 24), (16, 16), (15, 16), (15, 16))),
    (1, ('explosion', 'hiddenpowerfire', 'hypnosis', 'thunderbolt'), ((8, 8), (22, 24), (32, 32), (24, 24))),
    (2, ('hiddenpowerice', 'rest', 'return', 'thunderbolt'), ((24, 24), (16, 16), (32, 32), (24, 24))),
)


def request(owner=0):
    rows = [('Snorlax', '400/400 slp', True, OWN[0]), ('Gengar', '0 fnt', False, OWN[1]), ('Zapdos', '320/320', False, OWN[2])]
    return {'rqid': 9, 'side': {'id': f'p{owner + 1}', 'pokemon': [
        {'ident': f'p{owner + 1}: {name}', 'details': f'{name}, L80', 'condition': condition, 'active': active,
         # Showdown's request spells Return with its resolved power (getSwitchRequestData): return102.
         'moves': ['return102' if m == 'return' else m for m in mon['moves']], 'baseAbility': mon['ability'].lower(),
         'item': mon['item'].lower(),
         'stats': {'atk': 150, 'def': 160, 'spa': 170, 'spd': 180, 'spe': 190}}
        for name, condition, active, mon in rows]},
        'active': [{'moves': [{'move': m, 'id': m, 'pp': 5, 'maxpp': 16, 'target': 'normal', 'disabled': False}
                              for m in OWN[0]['moves']]}]}


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


def python_observation(self_moves, owner=0, lines=LINES, raw=None):
    from dataclasses import replace
    from pokezero import showdown as s
    f = fixture()
    raw = request(owner) if raw is None else raw
    replay = s.parse_showdown_replay([*lines, '|request|' + json.dumps(raw)], battle_id='self-moves-fixture')
    state = s.normalize_for_player(replay, player_id=f'p{owner + 1}', configured_showdown_slot=f'p{owner + 1}',
        format_id='gen3randombattle', set_source=f['source'])
    masks = replace(s.DEFAULT_OBSERVATION_FEATURE_MASKS, exact_state=True, opponent_tendency_stats_block=True,
        feature_pack_last_move=True, transition_token_budget=0, self_moves=self_moves)
    return s.observation_from_player_state(state, category_vocab=f['vocab'], spec=s.V4_REPLAY_OBSERVATION_SPEC, dex=f['dex'],
        feature_masks=masks)


def move_columns():
    from pokezero import showdown as s
    numeric = [s.v4_numeric_index(s.NUMERIC_REVEALED_MOVE_COUNT), s.v4_numeric_index(s.NUMERIC_POSSIBLE_MOVE_COUNT)]
    numeric += [s.v4_numeric_index(s.NUMERIC_OPP_MOVE_PP_OFFSET + i) for i in range(16)]
    numeric += [s.v4_numeric_index(s.NUMERIC_OPP_MOVE_PP_VALID_OFFSET + i) for i in range(16)]
    return numeric, list(range(s.CATEGORY_BELIEF_MOVE_OFFSET, s.CATEGORY_BELIEF_MOVE_OFFSET + 16))


class SelfMovesMaskTests(unittest.TestCase):
    """The switch: OBS_SELF_MOVES and the config latch (no encoder run)."""

    def test_env_override(self):
        from pokezero.observation import ObservationFeatureMasks, SELF_MOVES_ENV
        self.assertEqual(SELF_MOVES_ENV, 'OBS_SELF_MOVES')
        for value, declared, want in (('', False, False), ('', True, True), ('on', False, True), ('off', True, False)):
            with mock.patch.dict(os.environ, {SELF_MOVES_ENV: value}):
                self.assertIs(ObservationFeatureMasks(self_moves=declared).self_moves, want)
        with mock.patch.dict(os.environ, {SELF_MOVES_ENV: 'yes'}):
            with self.assertRaisesRegex(ValueError, 'OBS_SELF_MOVES'):ObservationFeatureMasks()
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop(SELF_MOVES_ENV, None)
            self.assertIs(ObservationFeatureMasks().self_moves, False)

    def test_model_config_latch_is_silent_when_off(self):
        from dataclasses import replace
        from pokezero.neural_policy import TransformerPolicyConfig, feature_masks_from_model_config
        config = TransformerPolicyConfig.compact_category(category_vocab=('species:snorlax', 'pokemon:self'))
        self.assertNotIn('self_moves', config.to_dict())
        self.assertEqual(TransformerPolicyConfig.from_dict(config.to_dict()), config)
        self.assertIs(feature_masks_from_model_config(config).self_moves, False)
        on = replace(config, self_moves=True)
        self.assertIs(on.to_dict()['self_moves'], True)
        self.assertIs(TransformerPolicyConfig.from_dict(on.to_dict()).self_moves, True)
        self.assertIs(feature_masks_from_model_config(on).self_moves, True)

    def test_cache_payload_records_only_when_on(self):
        from pokezero.dataset import _feature_masks_payload
        from pokezero.observation import ObservationFeatureMasks
        self.assertNotIn('self_moves', _feature_masks_payload(ObservationFeatureMasks()))
        self.assertIs(_feature_masks_payload(ObservationFeatureMasks(self_moves=True))['self_moves'], True)


@SHOWDOWN
class PythonSelfMovesTests(unittest.TestCase):
    def setUp(self):
        os.environ.pop('OBS_SELF_MOVES', None)

    def test_off_is_the_default_encoding_and_leaves_self_move_columns_zero(self):
        import numpy as np
        off = python_observation(False)
        numeric, cats = move_columns()
        self.assertFalse(np.asarray(off.numeric_features)[1:7][:, numeric].any())
        self.assertFalse(np.asarray(off.categorical_ids)[1:7][:, cats].any())

    def test_on_writes_own_moves_and_ledger_pp(self):
        import numpy as np
        from pokezero import showdown as s
        f = fixture()
        off, on = python_observation(False), python_observation(True)
        numeric, cats = move_columns()
        n_off, n_on = np.asarray(off.numeric_features), np.asarray(on.numeric_features)
        c_off, c_on = np.asarray(off.categorical_ids), np.asarray(on.categorical_ids)
        # Only the self tokens' move/PP cells change.
        changed_n = {(int(t), int(c)) for t, c in np.argwhere(n_off != n_on)}
        changed_c = {(int(t), int(c)) for t, c in np.argwhere(c_off != c_on)}
        self.assertTrue(all(1 <= t <= 6 and c in numeric for t, c in changed_n), sorted(changed_n))
        self.assertTrue(all(1 <= t <= 6 and c in cats for t, c in changed_c), sorted(changed_c))
        pp = [s.v4_numeric_index(s.NUMERIC_OPP_MOVE_PP_OFFSET + i) for i in range(16)]
        valid = [s.v4_numeric_index(s.NUMERIC_OPP_MOVE_PP_VALID_OFFSET + i) for i in range(16)]
        for slot, moves, fractions in EXPECTED:
            row, cat = n_on[1 + slot], c_on[1 + slot]
            self.assertEqual([int(x) for x in cat[cats]],
                [f['vocab'].encode('belief:possible_move:' + m) for m in moves] + [0] * 12)
            self.assertEqual(row[s.v4_numeric_index(s.NUMERIC_REVEALED_MOVE_COUNT)], 4.0)
            self.assertEqual(row[s.v4_numeric_index(s.NUMERIC_POSSIBLE_MOVE_COUNT)], 4.0)
            self.assertEqual(list(row[valid]), [1.0] * 4 + [0.0] * 12)
            self.assertEqual([float(x) for x in row[pp]], [np.float32(a / b) for a, b in fractions] + [0.0] * 12, (slot, moves))
        # Token 4-6: no fourth party member, nothing written.
        self.assertFalse(n_on[4:7].any())


@SHOWDOWN
@unittest.skipUnless(importlib.util.find_spec('jax'), 'requires isolated JAX')
class B2SelfMovesTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import jax
        from gpu_engine import public_belief, v4_encoder
        from pokezero.recorded_mirror_roots import CHAMPION_CONTRACT
        os.environ.pop('OBS_SELF_MOVES', None)
        f = fixture()
        cls.jax, cls.e = jax, v4_encoder
        masks = dict(CHAMPION_CONTRACT['feature_masks'])
        compiled = public_belief.compile_source(f['source'])
        cls.catalogs = {on: v4_encoder.compile_encoder(compiled, f['dex'], f['vocab'], checkpoint_sha256='a' * 64,
            feature_masks=dict(masks, self_moves=True) if on else masks) for on in (False, True)}

    def b2(self, on, owner=0, lines=LINES, raw=None):
        import numpy as np
        from pokezero.recorded_encoder_runner import JaxRecordedAdapter
        adapter = JaxRecordedAdapter(self.catalogs[on])
        adapter.reset(stream_id=1, battle_id='self-moves-fixture')
        adapter.feed(tuple(lines))
        return {k: np.asarray(v) for k, v in adapter.encode(request(owner) if raw is None else raw, owner_side=owner).items()}

    def test_fold_binding_and_off_catalog_are_unchanged(self):
        import numpy as np
        off, on = self.catalogs[False], self.catalogs[True]
        self.assertIsNone(off.self_moves)
        self.assertFalse(off.masks.self_moves)
        self.assertTrue(on.masks.self_moves)
        # self_moves is an encode-only switch: the fold-state binding is the same, so one fold serves both.
        self.assertTrue(np.array_equal(np.asarray(off.binding), np.asarray(on.binding)))
        with self.assertRaisesRegex(ValueError, 'without defaults'):
            self.e.Masks.from_checkpoint(dict(self_moves=True))

    def test_python_and_b2_agree_on_the_self_move_cells(self):
        """The fixture's lines are authored (not a recorded wire), so B2 and Python already differ on some unrelated cells
        with the flag off (HP visibility of the hand-written switch lines and the like; the real-root gate compares whole
        encodings). The switch must add no disagreement: the cells where the two encoders differ are the same off and on,
        none is a self-move cell, and the on-minus-off change is identical in both encoders."""
        import numpy as np
        numeric, cats = move_columns()
        b2 = {on: self.b2(on) for on in (False, True)}
        py = {on: python_observation(on) for on in (False, True)}
        for name, columns in (('numeric_features', numeric), ('categorical_ids', cats)):
            # Python rows are float64 lists; the model (and B2) read float32.
            p = {on: np.asarray(getattr(py[on], name), np.float32 if name == 'numeric_features' else None) for on in (False, True)}
            q = {on: b2[on][name] for on in (False, True)}
            self.assertTrue(np.array_equal(p[False] != q[False], p[True] != q[True]), name)
            self.assertTrue(np.array_equal(p[True][1:7][:, columns], q[True][1:7][:, columns]), name)
            self.assertTrue(np.array_equal(p[True] - p[False], q[True] - q[False]), name)
        for name in ('attention_mask', 'legal_action_mask'):
            for on in (False, True):
                self.assertTrue(np.array_equal(np.asarray(getattr(py[on], name)), b2[on][name]), (name, on))

    def test_on_differs_from_off_only_on_self_move_columns(self):
        import numpy as np
        off, on = self.b2(False), self.b2(True)
        numeric, cats = move_columns()
        diff = np.argwhere(off['numeric_features'] != on['numeric_features'])
        self.assertTrue(len(diff) and all(1 <= t <= 6 and c in numeric for t, c in diff), diff.tolist())
        diff = np.argwhere(off['categorical_ids'] != on['categorical_ids'])
        self.assertTrue(len(diff) and all(1 <= t <= 6 and c in cats for t, c in diff), diff.tolist())

    def test_env_switch_reaches_b2(self):
        from gpu_engine import v4_encoder
        from pokezero.recorded_mirror_roots import CHAMPION_CONTRACT
        with mock.patch.dict(os.environ, {'OBS_SELF_MOVES': 'on'}):
            self.assertTrue(v4_encoder.Masks.from_checkpoint(dict(CHAMPION_CONTRACT['feature_masks'])).self_moves)
        with mock.patch.dict(os.environ, {'OBS_SELF_MOVES': 'off'}):
            self.assertFalse(v4_encoder.Masks.from_checkpoint(dict(CHAMPION_CONTRACT['feature_masks'], self_moves=True)).self_moves)


if __name__ == '__main__':
    unittest.main()
