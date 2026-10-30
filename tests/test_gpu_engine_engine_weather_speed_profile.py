"""Actual-array Speed profile controls; literal source fixtures, no games/model."""
from copy import deepcopy
from dataclasses import replace
import unittest

import jax
import jax.numpy as j
import numpy as np

from gpu_engine import engine_weather_speed_profile as speed
from gpu_engine import engine_public_events as public, engine_request_source as source
from gpu_engine import transition
from gpu_engine.state import empty_state
import test_gpu_engine_engine_own_facts as fixtures


class WeatherSpeedProfileTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        fixtures.DeviceOwnFactsTests.setUpClass()
        cls.f = fixtures.DeviceOwnFactsTests
        # Deliberately retain unsafe IDs in immutable metadata. Source variants
        # are unchanged Leftovers variants; injected physical values MUST fail.
        cls.metadata = deepcopy(cls.f.metadata)
        cls.metadata['vocabulary']['items'] += ['machobrace', 'quickclaw', 'starfberry']
        cls.table = transition.tables_from_metadata(cls.metadata)
        cls.engine = public.compile_catalog(cls.metadata, cls.table, public_moves=cls.f.catalog.moves,
            dex=cls.f.catalog.dex,
            exposure=public.HealthExposure(source.SHOWDOWN, 'gen3randombattle', True, False))
        cls.rules = source.load_source_rules(cls.f.f.root, cls.metadata, node=cls.f.f.node)
        cls.profile = cls.compile()

    @classmethod
    def compile(cls, metadata=None, **kwargs):
        return speed.compile_profile(cls.metadata if metadata is None else metadata,
            catalog=cls.f.catalog, engine_catalog=cls.engine, rules=cls.rules, **kwargs)

    def state(self, batch=1):
        state = empty_state(batch, seed=17)
        species = j.broadcast_to(j.arange(1, 7, dtype=j.int32), (batch, 2, 6))
        return state._replace(species=species, ability=species, base_ability=species,
            item=j.ones((batch, 2, 6), j.int32), stats=j.full((batch, 2, 6, 6), 199, j.int32),
            hp=j.full((batch, 2, 6), 199, j.int32), max_hp=j.full((batch, 2, 6), 199, j.int32),
            done=j.zeros(batch, bool))

    def streams(self, batch=1):
        return j.arange(501, 501 + batch, dtype=j.int32)

    def birth(self, state, profile=None, streams=None):
        return speed.qualify_birth(state, self.streams(state.item.shape[0]) if streams is None else streams,
            profile=self.profile if profile is None else profile)

    def retain(self, record, state, profile=None, streams=None):
        return speed.retain(record, state, self.streams(state.item.shape[0]) if streams is None else streams,
            profile=self.profile if profile is None else profile)

    def test_checks_all_twelve_actual_members_despite_unchanged_safe_source_inventory(self):
        state = self.state(4)
        source_items = {v.item for rows in self.f.catalog.source.variants for v in rows}
        self.assertEqual(source_items, {'Leftovers'})
        self.assertEqual(self.birth(state).qualified_birth.tolist(), [True] * 4)
        injected = state.item.at[0, 1, 5].set(3).at[1, 0, 4].set(4).at[2, 1, 2].set(5)
        record = self.birth(state._replace(item=injected))
        self.assertEqual(record.qualified_birth.tolist(), [False, False, False, True])
        # Same original metadata/catalog/source, not a membership assumption.
        self.assertIs(speed.check_profile(self.profile), self.profile.device)

    def test_unsafe_birth_stays_false_after_item_consumption(self):
        state = self.state(3)
        bad = state._replace(item=state.item.at[0, 1, 5].set(3).at[1, 0, 5].set(4).at[2, 1, 4].set(5))
        birth = self.birth(bad)
        consumed = bad._replace(item=j.zeros_like(bad.item))
        retained = self.retain(birth, consumed)
        self.assertEqual(retained.qualified_birth.tolist(), [False] * 3)
        self.assertEqual(self.retain(retained, state).qualified_birth.tolist(), [False] * 3)

    def test_supported_old_swap_and_zero_closure_and_new_unsafe_item_latch(self):
        state = self.state(2); birth = self.birth(state)
        # Existing production item writes consume old item to0; a lawful
        # old-item swap also cannot introduce anything outside the safe set.
        consumed = state._replace(item=state.item.at[0, 0, 5].set(0))
        swapped = consumed._replace(item=consumed.item.at[0, 1, 4].set(0).at[0, 0, 5].set(1))
        retained = self.retain(birth, swapped)
        self.assertEqual(retained.qualified_birth.tolist(), [True, True])
        unsafe = swapped._replace(item=swapped.item.at[1, 1, 5].set(4))
        refused = self.retain(retained, unsafe)
        self.assertEqual(refused.qualified_birth.tolist(), [True, False])
        self.assertEqual(self.retain(refused, consumed).qualified_birth.tolist(), [True, False])

    def test_none_at_birth_is_only_explicit_math_scope(self):
        itemless = self.state()._replace(item=j.zeros((1, 2, 6), j.int32))
        self.assertEqual(self.birth(itemless).qualified_birth.tolist(), [False])
        explicit = self.compile(allow_none_at_birth=True)
        self.assertEqual(self.birth(itemless, explicit).qualified_birth.tolist(), [True])
        self.assertFalse(bool(j.all(explicit.device.binding == self.profile.device.binding)))
        with self.assertRaises(ValueError):
            self.compile(allow_none_at_birth=1)

    def test_every_inactive_member_stat_stage_status_and_id_bounds(self):
        state = self.state(7)
        state = state._replace(stats=state.stats.at[0, 1, 5, 5].set(0).at[1, 0, 4, 5].set(2501),
            boosts=state.boosts.at[2, 1, 3, 4].set(-7).at[3, 0, 5, 4].set(7),
            status=state.status.at[4, 1, 5].set(7),
            item=state.item.at[5, 0, 5].set(-1))
        self.assertEqual(self.birth(state).qualified_birth.tolist(), [False] * 6 + [True])
        boundary = self.state()._replace(stats=self.state().stats.at[..., 5].set(2500),
            boosts=self.state().boosts.at[..., 4].set(6), status=j.full((1, 2, 6), 6, j.int32))
        self.assertEqual(self.birth(boundary).qualified_birth.tolist(), [True])
        for value in (0, 99):
            self.assertEqual(self.birth(self.state()._replace(
                species=self.state().species.at[0, 1, 5].set(value))).qualified_birth.tolist(), [False])

    def test_unowned_copied_ability_and_changed_base_stats_are_sticky_unknown(self):
        state = self.state(3); birth = self.birth(state)
        changed = state._replace(ability=state.ability.at[0, 1, 5].set(1),
            base_ability=state.base_ability.at[1, 0, 4].set(1),
            stats=state.stats.at[2, 1, 5, 5].add(1))
        retained = self.retain(birth, changed)
        self.assertEqual(retained.qualified_birth.tolist(), [False] * 3)
        self.assertEqual(self.retain(retained, state).qualified_birth.tolist(), [False] * 3)
        for value in (0, -1, 99):
            bad = state._replace(ability=state.ability.at[0, 1, 5].set(value))
            self.assertFalse(bool(self.birth(bad).qualified_birth[0]))

    def test_stream_namespace_and_batch_identity_bound_per_row(self):
        state = self.state(3); birth = self.birth(state)
        for name in ('engine_binding', 'source_binding', 'profile_binding'):
            wrong = birth._replace(**{name:getattr(birth, name).at[1, 0].add(1)})
            self.assertEqual(self.retain(wrong, state).qualified_birth.tolist(), [True, False, True])
        self.assertEqual(self.retain(birth, state, streams=j.array([501, 999, 503], j.int32)).qualified_birth.tolist(),
            [True, False, True])
        duplicates = self.birth(state, streams=j.array([501, 501, 503], j.int32))
        self.assertEqual(duplicates.qualified_birth.tolist(), [False, False, True])
        self.assertEqual(self.birth(state, streams=j.array([501, 0, 503], j.int32)).qualified_birth.tolist(),
            [True, False, True])
        with self.assertRaises(ValueError):
            self.retain(birth, self.state(2))

    def test_turn0_is_conservative_and_not_claimed_as_birth_attestation(self):
        state = self.state()._replace(turn=j.array([1], j.int32))
        self.assertEqual(self.birth(state).qualified_birth.tolist(), [False])
        # The proper carry can advance its turn; it is never reminted from a
        # later safe state or held-item scan.
        self.assertEqual(self.retain(self.birth(self.state()), state).qualified_birth.tolist(), [True])
        for changed in (self.state()._replace(done=j.ones(1, bool)),
                self.state()._replace(hp=self.state().hp.at[0, 1, 5].add(-1)),
                self.state()._replace(hp=j.zeros((1, 2, 6), j.int32))):
            self.assertEqual(self.birth(changed).qualified_birth.tolist(), [False])

    def test_fresh_batches_and_B8_record_layout_on_jitted_internal_path(self):
        data = speed.check_profile(self.profile)
        born = jax.jit(lambda state, streams:speed.qualify_birth_device(state, streams, data))
        retain = jax.jit(lambda record, state, streams:speed.retain_device(record, state, streams, data))
        for batch in (1, 8, 3):
            state = self.state(batch); streams = self.streams(batch)
            record = jax.block_until_ready(born(state, streams))
            self.assertEqual(record.qualified_birth.tolist(), [True] * batch)
            for binding in (record.engine_binding, record.source_binding, record.profile_binding):
                self.assertEqual(binding.shape, (batch, 8))
            self.assertTrue(all(v.shape[0] == batch for v in record))
            current = state._replace(item=state.item.at[0, 1, 5].set(3))
            refused = jax.block_until_ready(retain(record, current, streams))
            self.assertEqual(refused.qualified_birth.tolist(), [False] + [True] * (batch - 1))
            self.assertFalse(bool(retain(refused, state, streams).qualified_birth[0]))

    def test_exact_array_types_reject_float_wide_boolean_and_wrong_shapes(self):
        state = self.state()
        for name in ('item', 'ability', 'base_ability', 'status', 'stats', 'boosts'):
            with self.subTest(name=name):
                with self.assertRaises(ValueError):
                    self.birth(state._replace(**{name:getattr(state, name).astype(j.float32)}))
        with self.assertRaises(ValueError):
            self.birth(state, streams=np.array([501], np.int64))
        with self.assertRaises(ValueError):
            self.birth(state._replace(item=j.ones((1, 2, 5), j.int32)))
        with self.assertRaises(ValueError):
            self.retain(self.birth(state)._replace(qualified_birth=j.ones(1, j.int32)), state)

    def test_original_profile_catalog_metadata_rules_and_device_edges_required(self):
        with self.assertRaises(ValueError):
            speed.check_profile(replace(self.profile))
        changed = deepcopy(self.metadata); changed['vocabulary']['items'][1] = 'quickclaw'
        with self.assertRaises(ValueError):
            self.compile(changed)
        old = self.profile.device
        object.__setattr__(self.profile, 'device', old._replace(birth_items=j.ones_like(old.birth_items)))
        try:
            with self.assertRaises(ValueError):
                self.birth(self.state())
        finally:
            object.__setattr__(self.profile, 'device', old)
        original = self.metadata['schema']; self.metadata['schema'] = 'changed-after-compile'
        try:
            with self.assertRaises(ValueError):
                self.birth(self.state())
        finally:
            self.metadata['schema'] = original
        with self.assertRaises(ValueError):
            speed.compile_profile(self.metadata, catalog=replace(self.f.catalog), engine_catalog=self.engine, rules=self.rules)
        with self.assertRaises(ValueError):
            speed.compile_profile(self.metadata, catalog=self.f.catalog, engine_catalog=replace(self.engine), rules=self.rules)
        with self.assertRaises(ValueError):
            speed.compile_profile(self.metadata, catalog=self.f.catalog, engine_catalog=self.engine, rules=replace(self.rules))


if __name__ == '__main__':
    unittest.main()
