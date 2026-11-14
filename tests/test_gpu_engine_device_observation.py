"""Synthetic device facts only: no engine construction/turn/model/fold/game."""
from typing import NamedTuple
import unittest

import jax
import jax.numpy as j
import numpy as np

from gpu_engine import engine_device_observation as e


class State(NamedTuple):
    species: object
    level: object
    gender: object
    hp: object
    max_hp: object
    stats: object
    status: object
    types: object
    ability: object
    item: object
    active: object
    boosts: object
    moves: object
    pp: object
    disabled: object
    forced_switch: object
    done: object
    turn: object
    weather: object
    spikes: object
    choice_move: object
    encore_move: object
    encore_counter: object
    last_action_kind: object
    last_action_index: object
    charge_move: object
    base_ability: object


class DeviceObservationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Existing source-only catalog fixture; no fold/initialize/engine call.
        import test_gpu_engine_v4_encoder as f
        f.V4EncoderTests.setUpClass()
        cls.catalog = f.V4EncoderTests.catalog
        cls.metadata = dict(vocabulary=dict(species=['_none', 'gardevoir', 'snorlax', 'absol'],
            types=['_none', 'normal', 'psychic', 'ghost', 'water'],
            abilities=['_none', 'immunity', 'pressure'], items=['_none', 'leftovers'],
            moves=['_none', 'surf', 'bodyslam', 'struggle']))
        cls.tables = e.compile_tables(cls.metadata, catalog=cls.catalog)

    def facts(self, b=2):
        mon = lambda value: j.full((b, 2, 6), value, j.int32)
        species = mon(0).at[:, :, :2].set(j.asarray([1, 2], j.int32))
        state = State(species, mon(80), mon(1), mon(120), mon(240),
            j.broadcast_to(j.asarray([240, 151, 160, 170, 180, 190], j.int32), (b, 2, 6, 6)),
            mon(0), j.broadcast_to(j.asarray([1, 0], j.int32), (b, 2, 6, 2)), mon(1), mon(1),
            j.ones((b, 2), j.int32), j.zeros((b, 2, 6, 7), j.int32),
            j.broadcast_to(j.asarray([1, 2, 0, 0], j.int32), (b, 2, 6, 4)),
            j.broadcast_to(j.asarray([12, 10, 0, 0], j.int32), (b, 2, 6, 4)),
            j.zeros((b, 2, 6, 4), bool), j.zeros((b, 2), bool), j.zeros(b, bool),
            j.full(b, 7, j.int32), j.zeros(b, j.int32), j.zeros((b, 2), j.int32),
            mon(0), mon(0), mon(0), j.zeros((b, 2), j.int32), j.full((b, 2), -1, j.int32), mon(0), mon(1))
        visible = j.arange(6)[None, :] == 0
        visible = j.broadcast_to(visible, (b, 6))
        zi = j.zeros((b, 6), j.int32)
        public = e.PublicView(j.broadcast_to(j.asarray([1, 0, 2, 3, 4, 5], j.int32), (b, 6)),
            j.where(visible, 3, 0), j.where(visible, 80, 0), zi,
            j.where(visible, .69, 0).astype(j.float32), zi, visible, visible, j.zeros_like(visible),
            j.zeros((b, 6, 2), j.int32), j.zeros((b, 6, 7), j.int32), zi, zi, j.zeros(b, j.int32),
            None, j.full(b, 7, j.int32), j.zeros(b, j.int32), j.zeros((b, 2), j.int32))
        legal = j.broadcast_to(j.asarray([True, True, False, False, True, False, False, False, False]), (b, 9))
        owners = j.arange(b, dtype=j.int32) % 2
        return state, public, owners, legal

    def test_original_private_writer_columns_and_real_layout_adapters(self):
        from gpu_engine import private_team
        state, public, owners, legal = self.facts(b=1)
        out = e.encode(state, public, self.tables, observer_sides=owners, legal_action_mask=legal)
        raw = {'side': {'id': 'p1', 'pokemon': [dict(ident='p1: ' + name,
            details=name + ', L80, M', condition='120/240', active=i == 0,
            ability='Immunity', item='Leftovers',
            stats={'atk': 151, 'def': 160, 'spa': 170, 'spd': 180, 'spe': 190})
            for i, name in enumerate(('Snorlax', 'Gardevoir'))]}, 'active': [{'moves': [{'id': 'surf'}]}]}
        facts = private_team.parse_team_facts(raw, owner_side=0, vocabulary=self.catalog.vocabulary, dex=self.catalog.dex)
        expected = private_team.project_team(facts, observer_side=0,
            expected_vocabulary_binding=private_team.vocabulary_binding(self.catalog.vocabulary))
        np.testing.assert_allclose(out.numeric_features[0, 1:7, private_team.NUMERIC_COLUMNS],
            np.asarray(expected.numeric)[:, private_team.NUMERIC_COLUMNS].T, rtol=0, atol=1e-6)
        # Own public type overlays use current type, so compare the other exact categories.
        columns = [0, 1, 2, e.reference.CATEGORY_BELIEF_ABILITY_OFFSET, e.reference.CATEGORY_BELIEF_ITEM_OFFSET]
        np.testing.assert_array_equal(out.categorical_ids[0, 1:3][:, columns], np.asarray(expected.categorical)[:2, columns])
        self.assertEqual(out.numeric_features.shape, (1, 29, 132))
        self.assertEqual(out.categorical_ids.shape, (1, 29, 41))
        self.assertEqual(e.actor_inputs(out)['numeric_features'].shape, (1, 1, 29, 132))
        self.assertEqual(e.champion_inputs(out)['numeric'].shape, (1, 23, 132))
        np.testing.assert_array_equal(e.champion_inputs(out)['numeric'], out.numeric_features[:, :23])
        np.testing.assert_array_equal(out.switch_targets, [[0, 2, 3, 4, 5]])
        self.assertEqual(out.error.tolist(), [0])

    def test_jit_mixed_seats_private_opponent_noninterference_and_no_row_drop(self):
        state, public, owners, legal = self.facts()
        fn = jax.jit(lambda s, p, o, l: e.encode(s, p, self.tables, observer_sides=o, legal_action_mask=l))
        out = fn(state, public, owners, legal)
        # Mutate every private opponent field independently on each row; same
        # root-public tracker + own team must yield identical public29 tensors.
        changed = state
        rows = j.arange(2); foe = 1 - owners
        for name in ('species', 'level', 'gender', 'hp', 'max_hp', 'stats', 'status', 'types',
                'ability', 'base_ability', 'item', 'active', 'boosts', 'moves', 'pp', 'disabled', 'forced_switch'):
            a = getattr(changed, name)
            changed = changed._replace(**{name: a.at[rows, foe].set(~a[rows, foe] if a.dtype == j.bool_ else a[rows, foe] + 999)})
        # Even sampled-world public globals cannot override saved tracker fields.
        changed = changed._replace(turn=state.turn + 99, weather=state.weather + 2, spikes=state.spikes + 2)
        other = fn(changed, public, owners, legal)
        for left, right in zip(out, other):np.testing.assert_array_equal(left, right)
        bad = public._replace(species=public.species.at[0, 0].set(999), error=j.asarray([4, 0], j.int32))
        got = fn(state, bad, owners, legal)
        self.assertNotEqual(int(got.error[0]), 0); self.assertEqual(int(got.error[1]), 0)
        self.assertTrue(np.asarray(e.actor_inputs(got)['history_mask']).all())
        self.assertEqual(got.numeric_features.shape, out.numeric_features.shape)
        self.assertTrue(np.isfinite(got.numeric_features).all())

    def test_own_fainted_active_stages_come_from_the_public_carry(self):
        # OPEN BUG D: the engine zeroed the fainted active's stages (Showdown's clearVolatile); the champion's B2/Python
        # inputs keep them until the replacement arrives, so the own active token reads the carry's (readout own_boosts).
        state, public, owners, legal = self.facts(b=1)
        columns = [e._index('BOOST_' + n) for n in ('ATK', 'DEF', 'SPA', 'SPD', 'SPE')]
        kept = j.asarray([[2, 0, 0, 0, -1, 0, 0]], j.int32)
        fainted = state._replace(hp=state.hp.at[:, 0, 1].set(0))          # owner 0's active is physical 1 (token 1)
        out = e.encode(fainted, public._replace(own_boosts=kept), self.tables, observer_sides=owners,
            legal_action_mask=legal)
        np.testing.assert_allclose(out.numeric_features[0, 1, columns], [2 / 6, 0, 0, 0, -1 / 6], rtol=0, atol=1e-7)
        np.testing.assert_array_equal(out.numeric_features[0, 2:7][:, columns], 0)  # benched tokens: no stages
        # The carry is read only at hp 0: a living active reads the engine's stages; without the field, the engine's.
        alive = e.encode(state, public._replace(own_boosts=kept), self.tables, observer_sides=owners,
            legal_action_mask=legal)
        np.testing.assert_array_equal(alive.numeric_features[0, 1, columns], 0)
        bare = e.encode(fainted, public, self.tables, observer_sides=owners, legal_action_mask=legal)
        np.testing.assert_array_equal(bare.numeric_features[0, 1, columns], 0)
        same = lambda x, y: [np.testing.assert_array_equal(a, b) for a, b in zip(x, y)]
        same(bare[:1], e.encode(fainted, public._replace(own_boosts=j.zeros_like(kept)), self.tables,
            observer_sides=owners, legal_action_mask=legal)[:1])

    def test_manifest_covers_every_column_and_source_alias_mapping(self):
        state, public, owners, legal = self.facts(b=1)
        manifest = e.column_manifest(public=public)
        self.assertEqual(len(manifest), 6 * (132 + 41))
        self.assertEqual(len({(r['section'], r['kind'], r['column']) for r in manifest}), len(manifest))
        self.assertTrue(all(r['status'] in ('DIRECT', 'TRACKED', 'ZERO') and r['reason'] for r in manifest))
        out = e.encode(state, public, self.tables, observer_sides=owners, legal_action_mask=legal)
        for row in manifest:
            if row['status'] != 'ZERO':continue
            # Missing synthetic tracker hooks are conservative metadata, not
            # evidence that a direct readout must be zero. Structural and
            # unavailable columns still promise literal zero output.
            if row['reason'].startswith('deferred'):continue
            sl = {'field': slice(0, 1), 'own': slice(1, 7), 'opponent': slice(7, 13),
                'actions': slice(13, 22), 'stats': slice(22, 23), 'belief': slice(23, 29)}[row['section']]
            a = out.numeric_features if row['kind'] == 'numeric' else out.categorical_ids
            with self.subTest(section=row['section'],kind=row['kind'],column=row['column']):
                np.testing.assert_array_equal(a[:, sl, row['column']], 0)
        # With no observations beyond species, posterior equals the source
        # prior. Missing public hooks do not substitute sampled hidden sets.
        np.testing.assert_allclose(out.numeric_features[:, 23:, :64], out.numeric_features[:, 23:, 64:128], atol=1e-7, rtol=0)
        self.assertGreater(float(out.numeric_features[:, 23:, 64:128].sum()), 0)

    def test_original_action_writer_force_switch_and_struggle(self):
        from types import SimpleNamespace
        from pokezero import showdown as s
        state, public, owners, legal = self.facts(b=1)
        raw = {'side': {'id': 'p1', 'pokemon': [dict(ident='p1: ' + name,
            details=name + ', L80, M', condition='120/240', active=i == 0,
            ability='Immunity', item='Leftovers',
            stats={'atk': 151, 'def': 160, 'spa': 170, 'spd': 180, 'spe': 190})
            for i, name in enumerate(('Snorlax', 'Gardevoir'))]},
            'active': [{'moves': [dict(id=n, pp=pp, maxpp=self.catalog.dex.move_info(n).max_pp)
                for n, pp in [('surf', 12), ('bodyslam', 10)]]}]}
        out = e.encode(state, public, self.tables, observer_sides=owners, legal_action_mask=legal)
        team = s._self_team_from_request(raw, 'p1')
        source = SimpleNamespace(request=raw, self_team=team, self_active=team[0], legal_action_mask=s._legal_action_mask(raw))
        nums = [[0.] * s.V4_PRIVATE_WRITER_NUMERIC_FEATURE_COUNT for _ in range(23)]
        cats = [[''] * 60 for _ in range(23)]
        s._encode_action_tokens(cats, nums, source, dex=self.catalog.dex)
        np.testing.assert_allclose(out.numeric_features[0, 13:22], s._project_v4_numeric_rows(nums)[13:22], rtol=0, atol=1e-6)
        expected = [[self.catalog.vocabulary.encode(label) for label in row[:41]] for row in cats[13:22]]
        np.testing.assert_array_equal(out.categorical_ids[0, 13:22], expected)
        forced = e.encode(state._replace(forced_switch=j.ones((1, 2), bool)), public, self.tables,
            observer_sides=owners, legal_action_mask=legal.at[:, :4].set(False))
        np.testing.assert_array_equal(forced.numeric_features[:, 13:17], 0)
        no_pp = state._replace(pp=j.zeros_like(state.pp))
        struggle = e.encode(no_pp, public, self.tables, observer_sides=owners,
            legal_action_mask=legal.at[:, :4].set(j.asarray([True, False, False, False])))
        self.assertEqual(int(struggle.categorical_ids[0, 13, 0]), self.catalog.vocabulary.encode('move:struggle'))
        self.assertEqual(float(struggle.numeric_features[0, 13, e._index('MOVE_PP_FRACTION')]), 1.)
        np.testing.assert_array_equal(struggle.numeric_features[0, 14:17], 0)
        borrowed = e.encode(state._replace(ability=j.full_like(state.ability, 2)), public, self.tables,
            observer_sides=owners, legal_action_mask=legal)
        # getRequestData's baseAbility is unchanged by current borrowed ability.
        np.testing.assert_array_equal(borrowed.categorical_ids[:, 1:7], out.categorical_ids[:, 1:7])


if __name__ == '__main__':unittest.main()
