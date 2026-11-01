"""Pure typed/source comparisons; no engine step/model/fold/game execution."""
from types import SimpleNamespace as NS
import unittest
import jax
import jax.numpy as j
import numpy as np

from gpu_engine import engine_device_belief as b, engine_device_features as f
from gpu_engine import engine_device_observation as e
import test_gpu_engine_device_observation as fixtures


class DeviceBeliefTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        fixtures.DeviceObservationTests.setUpClass()
        cls.fixture = fixtures.DeviceObservationTests()
        cls.catalog = cls.fixture.catalog; cls.tables = cls.fixture.tables

    def reference(self, public):
        from gpu_engine import public_belief as p, public_mon_state as mon, public_moves
        from gpu_engine import public_belief_features as features
        from gpu_engine import public_expected_stats as expected
        c = self.catalog; n = public.species.shape[0]
        sid = self.tables.belief.source_species[public.species]
        source_species = j.zeros((n, 2, 6), j.int32).at[:, 0].set(sid)
        state = p.initialize(source_species, c.source.device)
        if public.belief_mask is not None:state = state._replace(hard=state.hard.at[:, 0].set(public.belief_mask), has_evidence=j.ones_like(state.has_evidence))
        state, summary = p.materialize(state, c.source.device)
        pub_ids = j.asarray([c.species.ids.get(name, 0) for name in ('', 'gardevoir', 'snorlax', 'absol')])
        mons = mon.initialize(n, species_binding=c.species.device.binding)
        mons = mons._replace(encounter=mons.encounter.at[:, 1].set(public.present.astype(j.int32)),
            species=mons.species.at[:, 1].set(pub_ids[public.species]),
            source_species=mons.source_species.at[:, 1].set(sid),
            level=mons.level.at[:, 1].set(public.level), active=mons.active.at[:, 1].set(0))
        ability = j.zeros_like(source_species).at[:, 1].set(self.tables.belief.engine_ability[public.ability])
        item = j.zeros_like(source_species).at[:, 1].set(self.tables.belief.engine_item[public.item])
        binding = lambda value: j.broadcast_to(value, (n, 8))
        moves = public_moves.initialize(n, c.moves.device)
        if public.revealed_moves is not None:
            count = public.revealed_moves.shape[-1]
            moves = moves._replace(move_ids=moves.move_ids.at[:, 1, :, :count].set(self.tables.belief.engine_move[public.revealed_moves]),
                uses=moves.uses.at[:, 1, :, :count].set(public.move_pp_used))
        abilities = NS(original=ability, binding=binding(c.features.ability_binding), species_binding=mons.species_binding,
            transformed=j.zeros_like(ability, bool), transform_species=j.zeros_like(ability))
        context = NS(abilities=abilities, items=NS(revealed=item, removed=j.zeros_like(item, bool),
            binding=binding(c.features.item_binding), species_binding=mons.species_binding), moves=moves)
        original = features.project_opponent(state, summary, mons, context, c.features,
            observer_side=j.zeros(n, j.int32), expected_binding=c.features.binding)
        stats = expected.project_opponent(state, summary, mons, abilities, c.expected.device,
            observer_side=j.zeros(n, j.int32), expected_binding=c.expected.device.binding)
        return original, stats

    def test_original_buckets_pp_fallback_weights_and_stat_bounds(self):
        _, view, _, _ = self.fixture.facts()
        # Select a publicly visible Snorlax whose source has known variants.
        view = view._replace(species=view.species.at[:, 0].set(2))
        hard = self.tables.belief.species_initial_mask[view.species]
        revealed = j.zeros((2, 6, 4), j.int32).at[:, 0, 0].set(1)
        hard = hard.at[0, 0].set(hard[0, 0] & self.tables.belief.move_mask[2, 1])
        hard = hard.at[1, 0].set(0)  # original inconsistent-reveal fallback
        view = view._replace(belief_mask=hard, revealed_moves=revealed, move_pp_used=j.where(revealed > 0, 2, 0))
        got = jax.jit(lambda v: b.project(v, self.tables.belief))(view)
        old, old_stats = self.reference(view)
        np.testing.assert_allclose(got.numeric, old.patch.numeric, atol=1e-6, rtol=0)
        np.testing.assert_array_equal(got.categorical, old.patch.categorical)
        np.testing.assert_allclose(got.weights[..., :old.posterior_weights.shape[-1]], old.posterior_weights, atol=1e-7, rtol=0)
        np.testing.assert_array_equal(got.reason, old.reason)
        stats, bad = b.expected_stats(view, got, self.tables.belief)
        np.testing.assert_allclose(stats, old_stats.numeric[..., j.asarray(b.expected.NUMERIC_COLUMNS)], atol=1e-6, rtol=0)
        self.assertFalse(np.asarray(bad).any())
        self.assertEqual(int(got.reason[1, 0]), 2)

    def test_source_hidden_power_public_alias_and_struggle_no_evidence(self):
        metadata = dict(vocabulary=dict(self.fixture.metadata['vocabulary']))
        metadata['vocabulary']['moves'] = list(metadata['vocabulary']['moves']) + ['hiddenpowerfighting', 'hiddenpowerice']
        tables = b.compile_tables(metadata, self.catalog)
        self.assertEqual(int(tables.public_move[-1]), int(tables.public_move[-2]))
        np.testing.assert_array_equal(tables.move_mask[:, -1], tables.move_mask[:, -2])
        self.assertFalse(bool(tables.source_move_evidence[3]))
        self.assertEqual(int(tables.engine_move[-1]), self.catalog.moves.ids['hiddenpower'])
        source_mask = self.catalog.source.device.predicates[self.catalog.source.predicate_id('move', 'hiddenpower')]
        np.testing.assert_array_equal(tables.move_mask[:, -1], source_mask[tables.source_species])

    def test_raw_public_clocks_categories_and_request_charge_mapping(self):
        state, view, owners, legal = self.fixture.facts(b=1)
        active = view.active; present = view.present
        values = lambda x: j.where(active, x, 0)
        facts = dict(stall_counter=values(3), confusion_turns=values(2), encore_turns=values(1),
            wrap_trap_turns=values(4), meanlook_trap=active, truant_loaf=active,
            last_damage_dealt=values(.2), last_damage_taken=values(.3))
        nums, cats, bad = f.project_mon(j.zeros((1, 6, 132), j.float32), j.zeros((1, 6, 41), j.int32),
            facts, active=active, present=present, tables=self.tables.features,
            last_moves=j.where(active, 1, 0), borrowed_ability=j.where(active, 2, 0))
        from pokezero import showdown as s
        candidate = s.ShowdownPokemon('p2: test', 'p2', 'Snorlax', '69/100', True, 'Snorlax, L80')
        source_num = [[0.] * s.V4_PRIVATE_WRITER_NUMERIC_FEATURE_COUNT for _ in range(6)]
        source_cat = [[''] * 60 for _ in range(6)]
        s._encode_pokemon_tokens(source_cat, source_num, 0, (candidate,), role='opponent', limit=6,
            active_stall_counter=3, active_confusion_elapsed=2, active_encore_elapsed=1,
            active_wrap_trap_elapsed=4, active_meanlook_trap=True, active_truant_loaf=True,
            active_last_used_move='surf', active_traced_ability='pressure',
            active_last_damage_dealt=.2, active_last_damage_taken=.3, schema_v3=True, schema_v4=True)
        columns = [e._index(f.MON_FIELDS[k][0]) for k in facts]
        np.testing.assert_allclose(nums[0, 0, columns], np.asarray(s._project_v4_numeric_rows(source_num))[0, columns], atol=1e-6, rtol=0)
        for col in (s.CATEGORY_LAST_USED_MOVE, s.CATEGORY_TRACED_ABILITY):
            self.assertEqual(int(cats[0, 0, col]), self.catalog.vocabulary.encode(source_cat[0][col]))
        self.assertFalse(np.asarray(bad).any())
        # Slot1 becomes the sole source request slot0; sampling still selects
        # the exact original engine slot1 through the explicit dispatch map.
        charge = state.charge_move.at[:, :, 1].set(2)
        locked = state._replace(charge_move=charge)
        mask = legal.at[:, :4].set(j.asarray([False, True, False, False]))
        out = e.encode(locked, view, self.tables, observer_sides=owners, legal_action_mask=mask)
        self.assertEqual(out.move_targets.tolist(), [[1, 1, 2, 3]])
        self.assertEqual(out.legal_action_mask[:, :4].tolist(), [[True, False, False, False]])
        self.assertEqual(e.map_actions(out, locked, owners, j.asarray([0], j.int32)).tolist(), [1])
        self.assertEqual(float(out.numeric_features[0, 13, e._index('MOVE_PP_FRACTION')]), 1.)

    def test_actual_encoder_public_dicts_jit_and_row_local_errors(self):
        state, view, owners, legal = self.fixture.facts()
        view = view._replace(mon_features=dict(turns_active=j.where(view.active, 8, 0),
            last_damage_dealt=j.where(view.active, .25, 0)),
            own_mon_features=dict(stall_counter=j.zeros((2, 6), j.int32)),
            field_features=dict(self_wish_pending=j.ones(2, bool), self_wish_turns=j.ones(2, j.int32)),
            stats_features=dict(opp_switch_count=j.asarray([32, 16], j.int32)),
            last_moves=j.where(view.active, 1, 0))
        fn = jax.jit(lambda s, v: e.encode(s, v, self.tables, observer_sides=owners, legal_action_mask=legal))
        out = fn(state, view)
        np.testing.assert_allclose(out.numeric_features[:, 7, e._index('TURNS_ACTIVE')], 8 / 64, atol=1e-7)
        np.testing.assert_allclose(out.numeric_features[:, 22, e._index('STAT_OPP_SWITCH_COUNT')], [.5, .25], atol=1e-7)
        self.assertTrue(np.asarray(out.attention_mask[:, 22]).all())
        self.assertFalse(np.asarray(out.error).any())
        damaged = view._replace(mon_features=dict(view.mon_features, last_damage_dealt=view.mon_features['last_damage_dealt'].at[0, 0].set(j.nan)))
        bad = fn(state, damaged)
        self.assertNotEqual(int(bad.error[0]), 0); self.assertEqual(int(bad.error[1]), 0)
        self.assertTrue(np.asarray(e.actor_inputs(bad)['history_mask']).all())
        self.assertTrue(np.isfinite(bad.numeric_features).all())


if __name__ == '__main__':unittest.main()
