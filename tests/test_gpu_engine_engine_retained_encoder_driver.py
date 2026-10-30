"""Actual tiny constructor/transition driver, FAKE forward only, no models."""
from dataclasses import replace
import json
import unittest
import numpy as np

import jax
import jax.numpy as j

from gpu_engine import engine_retained_encoder_driver as driver
from gpu_engine import engine_request_source as source, engine_own_facts as own
from gpu_engine import transition
import test_gpu_engine_engine_own_facts as fixtures


class RetainedEncoderDriverTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        fixtures.DeviceOwnFactsTests.setUpClass()
        cls.f = fixtures.DeviceOwnFactsTests
        # Runtime poison and a distinctive count expose any incorrectly scored
        # refused row without Python callbacks or actual model execution.
        cls.fake = staticmethod(lambda values: driver.FakeScores(
            j.full((9,), j.nan, j.float32), j.array(7, j.int32)))
        cls.prepared = driver.prepare(cls.f.request, cls.f.compiled, fake_forward=cls.fake)
        cls.streams = j.array([501, 502], j.int32)

    def construct(self, pool=None):
        return driver.construct(self.f.pool if pool is None else pool, np.array([[0, 1], [0, 1]], np.int64),
            prepared=self.prepared, stream_ids=self.streams, seed=37)

    def chance(self):
        draws = j.zeros((2, 2, 12), j.int32).at[:, :, 4].set(1)
        return transition.ChanceInputs(draws, *[j.zeros((2, 2), j.int32) for _ in range(4)], j.zeros(2, j.int32))

    def same(self, a, b):
        for old, new in zip(jax.tree.leaves(a), jax.tree.leaves(b)):
            np.testing.assert_array_equal(old, new)

    def assert_refused_before_fake(self, result):
        for report in result.owners:
            self.assertEqual(report.policy_admitted.tolist(), [False, False])
            self.assertEqual(report.scored.tolist(), [False, False])
            self.assertEqual(report.fake_calls.tolist(), [0, 0])
            self.assertFalse(bool(report.fake_logits.any()))
            self.assertFalse(bool(report.encoded.observation.numeric_features.any()))
            self.assertFalse(bool(report.encoded.observation.categorical_ids.any()))
            self.assertFalse(bool(report.encoded.observation.attention_mask.any()))
            self.assertFalse(bool(report.encoded.observation.legal_action_mask.any()))
            self.assertTrue(bool((report.reason & driver.PUBLIC_COVERAGE).all()))
            self.assertTrue(bool((report.reason & driver.FULL_PUBLIC).all()))
            self.assertFalse(bool(report.full_encoder_admitted.any()))
            self.assertFalse(bool(report.B4_qualified.any()))

    def test_actual_constructor_reaches_ownfacts_retained173_and_real_missing_consumer_gates(self):
        session = self.construct()
        self.assertEqual(session.state.core.cursor.stream_id.tolist(), [501, 502])
        self.assertEqual(len(session.state.observers), 2)
        self.assertEqual(session.state.observers[0].owner.tolist(), [0, 0])
        self.assertEqual(session.state.observers[1].owner.tolist(), [1, 1])
        # Two separate B-row observers retain the same physical lifetime IDs;
        # no duplicate501/502 values enter an ambiguous2B event batch.
        for observer in session.state.observers:
            self.assertEqual(observer.basic.cursor.stream_id.tolist(), [501, 502])
            self.assertEqual(observer.basic.mons.encounter.shape, (2, 2, 6))
            self.assertEqual(observer.basic.cursor.sequence.tolist(), [6, 6])
            self.assertEqual(observer.basic.cursor.turn.tolist(), [1, 1])
        original = driver._SESSIONS[session]['constructed']
        for side, report in enumerate(session.result.owners):
            self.assertEqual(report.basic_error.tolist(), [0, 0])
            self.assertTrue(bool(report.own.admitted.all()))
            expected = own.produce_own_facts(original, self.f.compiled, owner_sides=j.full(2, side, j.int32))
            self.same(report.own, expected)
            self.assertEqual(report.encoded.observation.numeric_features.shape, (2, 23, 132))
            self.assertEqual(report.encoded.observation.categorical_ids.shape, (2, 23, 41))
            self.assertTrue(bool((report.retained_error != 0).all()))
            self.assertTrue(bool((report.reason & driver.FULL_PUBLIC).all()))
        self.assert_refused_before_fake(session.result)
        print('DRIVER_ACTUAL_COMPONENT_GATES', json.dumps([dict(owner=side,
            basic_error=report.basic_error.tolist(), retained_error=report.retained_error.tolist(),
            own_admitted=report.own.admitted.tolist(), reason=report.reason.tolist(),
            pending_columns=report.encoded.pending_numeric_columns.tolist(),
            unwritten_numeric=report.unwritten_numeric.tolist(), unwritten_categorical=report.unwritten_categorical.tolist(),
            policy_admitted=report.policy_admitted.tolist(), fake_calls=report.fake_calls.tolist())
            for side, report in enumerate(session.result.owners)], sort_keys=True))

    def test_same_core_one_invocation_nonphysical_bench_both_orders_and_hot_jit(self):
        arrays = {name:np.array(value, copy=True) for name, value in self.f.arrays.items()}
        # Reorder actual authored individuals, retaining all constructor fields;
        # the selected bench pair permits the subsequent real return switch.
        for value in arrays.values():
            if value.shape[:2] == (2, 6):
                value[0, [1, 5]] = value[0, [5, 1]]
        session = self.construct(self.f.make_pool(arrays))
        old = session.state.core.engine
        counts = (len(source._REGISTRATIONS), len(source._CONSTRUCTED), len(driver.actions._CONSTRUCTORS))
        actions = j.full((2, 2), 8, j.int32)  # physical5 is only public encounter1
        legacy = transition.step_with_chance(old, actions, self.f.table, self.chance())
        first = driver.advance(session, actions, chance_inputs=self.chance())
        self.same(first.state.core.engine, legacy.state)
        self.assertEqual(first.state.core.frame_number.tolist(), [1, 1])
        self.assertEqual(first.state.core.engine.active.tolist(), [[5, 5], [5, 5]])
        for side in (0, 1):
            self.assertEqual(first.state.core.orders[side].request_to_physical[:, 0].tolist(), [5, 5])
            self.assertEqual(first.state.observers[side].basic.mons.active.tolist(), [[1, 1], [1, 1]])
            self.assertEqual(first.state.observers[side].projection.cursor.generation.tolist(), [[1, 1], [1, 1]])
            self.assertEqual(first.owners[side].basic_error.tolist(), [0, 0])
            self.assertTrue(bool(first.owners[side].own.admitted.all()))
            self.assertEqual(first.state.observers[side].basic.cursor.sequence.tolist(), [16, 16])
        expected_second = transition.step_with_chance(first.state.core.engine, j.full((2, 2), 4, j.int32),
            self.f.table, self.chance())
        second = driver.advance(session, j.full((2, 2), 4, j.int32), chance_inputs=self.chance())
        self.same(second.state.core.engine, expected_second.state)
        self.assertEqual(second.state.core.frame_number.tolist(), [2, 2])
        self.assertEqual(second.state.core.engine.active.tolist(), [[0, 0], [0, 0]])
        for observer in second.state.observers:
            self.assertEqual(observer.basic.mons.active.tolist(), [[0, 0], [0, 0]])
            self.assertEqual(observer.projection.cursor.generation.tolist(), [[2, 2], [2, 2]])
            self.assertEqual(observer.basic.cursor.sequence.tolist(), [26, 26])
        self.assertEqual(counts, (len(source._REGISTRATIONS), len(source._CONSTRUCTED), len(driver.actions._CONSTRUCTORS)))
        r = driver._PREPARED[self.prepared]
        for name in ('frame', 'request_fast', 'public_fast', 'encode_fast', 'score_fast'):
            self.assertEqual(r[name]._cache_size(), 1, name)
        program = str(jax.make_jaxpr(r['frame'])(second.state.core, actions, self.chance()))
        self.assertNotIn('pure_callback', program)
        self.assertNotIn('io_callback', program)
        self.assert_refused_before_fake(first)
        self.assert_refused_before_fake(second)

    def test_actual_action_faint_then_forced_replacement_uses_source_wait_and_order(self):
        arrays = {name:np.array(value, copy=True) for name, value in self.f.arrays.items()}
        arrays['stats'][1, 0, 0] = 1
        session = self.construct(self.f.make_pool(arrays))
        first = driver.advance(session, j.zeros((2, 2), j.int32), chance_inputs=self.chance())
        self.assertEqual(first.state.core.engine.hp[:, 1, 0].tolist(), [0, 0])
        self.assertEqual(first.state.core.engine.pending_residual.tolist(), [True, True])
        # Both requests are source-produced; only the forced owner can choose.
        self.assertTrue(bool(first.state.core.requests[1].actions.facts.request.force_switch.all()))
        self.assertTrue(bool(first.state.core.requests[0].actions.facts.request.wait.all()))
        resumed = driver.advance(session, j.tile(j.array([[0, 4]], j.int32), (2, 1)), chance_inputs=self.chance())
        self.assertEqual(resumed.state.core.engine.active[:, 1].tolist(), [1, 1])
        self.assertEqual(resumed.state.core.orders[1].generation.tolist(), [1, 1])
        self.assertEqual(resumed.state.observers[0].basic.mons.active[:, 1].tolist(), [1, 1])
        self.assert_refused_before_fake(first)
        self.assert_refused_before_fake(resumed)

    def test_unknown_constructor_row_rolls_back_public_and_never_scores_or_advances(self):
        arrays = {name:np.array(value, copy=True) for name, value in self.f.arrays.items()}
        arrays['gender'][0, 0] = -1
        session = self.construct(self.f.make_pool(arrays))
        self.assertEqual(session.state.core.admitted.tolist(), [False, False])
        before = session.state.core.engine
        for observer in session.state.observers:
            self.assertEqual(observer.basic.cursor.sequence.tolist(), [0, 0])
            self.assertFalse(bool(observer.basic.mons.encounter.any()))
        result = driver.advance(session, j.full((2, 2), 8, j.int32), chance_inputs=self.chance())
        self.same(before, result.state.core.engine)
        self.assert_refused_before_fake(result)

    def test_host_guards_run_after_hot_jit_and_stale_carry_cannot_be_replayed(self):
        session = self.construct()
        driver.advance(session, j.full((2, 2), 8, j.int32), chance_inputs=self.chance())
        old = session.state
        index = self.f.catalog.vocabulary._index
        key = next(iter(index))
        value = index[key]
        index[key] = value+1
        try:
            with self.assertRaises(ValueError):
                driver.advance(session, j.full((2, 2), 4, j.int32), chance_inputs=self.chance())
            self.assertIs(session.state, old)
        finally: index[key] = value
        session.state = old._replace(core=old.core._replace(frame_number=old.core.frame_number-1))
        try:
            with self.assertRaises(ValueError):
                driver.advance(session, j.full((2, 2), 4, j.int32), chance_inputs=self.chance())
        finally: session.state = old
        with self.assertRaises(ValueError):
            jax.jit(lambda act: driver.advance(session, act, chance_inputs=self.chance()))(j.zeros((2, 2), j.int32))
        self.assert_refused_before_fake(session.result)

    def test_exact_full_catalog_input_format_gap_against_pinned_initial_wire(self):
        session = self.construct()
        r = driver._PREPARED[self.prepared]
        original = driver._SESSIONS[session]['constructed']
        catalog = r['catalog']
        before = driver.life.register(self.streams, catalog=r['request']['lifecycle'],
            expected_binding=r['request']['lifecycle'].binding,
            expected_engine_binding=r['request']['rules'].device.engine_binding)
        frame = driver.life.LifecycleResult(original.state, original.cursor, original.callbacks,
            driver.life.snapshot(before), original.registry, j.zeros(2, bool), j.zeros(2, j.int32))
        evidence = []
        for side in (0, 1):
            owners = j.full(2, side, j.int32)
            initial = driver.public.register(self.streams, owners, compiled=r['public'])
            actual = driver.public.project_frame(initial, frame, r['public'].device)
            lines = []
            for seat in (0, 1):
                sid, level, gender, hp, maximum, status = original.callbacks.switch_details[0, seat].tolist()
                self.assertEqual(status, 0)
                name = catalog.dex.species_info(self.f.metadata['vocabulary']['species'][sid]).name
                detail = name+f', L{level}'+(', M' if gender == 1 else ', F' if gender == 2 else '')
                if seat != side:
                    hp = (100*hp+maximum-1)//maximum
                    maximum = 100
                lines.append(f'|switch|p{seat+1}a: {name}|{detail}|{hp}/{maximum}')
            lines.append('|turn|1')
            _, reference = driver.v4_encoder.public_wire.project_lines(
                driver.v4_encoder.public_wire.WireState(), lines, **driver.v4_encoder.wire_kwargs(catalog))
            events = [e for p in reference for e in p.events]
            expected_words = np.array([e.words() for e in events])
            actual_words = np.array(actual.chunk.words[0])[np.array(actual.chunk.valid[0])]
            namespace_names = ('species_binding', 'ability_binding', 'move_binding', 'item_binding',
                'context_binding', 'field_binding', 'active_binding', 'damage_binding', 'type_binding',
                'sleep_binding', 'toxic_binding', 'tendency_binding')
            kwargs = {name:np.asarray(getattr(reference[0], name), np.uint32) for name in namespace_names}
            chunk = next(driver.v4_encoder.public_inputs.pack_chunks(np.array(self.streams),
                [events, events], source_binding=np.asarray(catalog.source.device.binding), **kwargs))
            full = driver.v4_encoder.initialize(catalog, stream_ids=self.streams, owner_sides=owners)
            _, reference_error = driver.v4_encoder.fold_chunk(full, chunk, catalog)
            self.assertEqual(reference_error.tolist(), [0, 0])
            mismatches = []
            for name in namespace_names:
                value = getattr(actual.chunk, name)
                if value is None or not np.array_equal(np.array(value), kwargs[name]):
                    mismatches.append(name)
            self.assertEqual(mismatches[0], 'context_binding')
            different = np.argwhere(actual_words[:min(len(actual_words), len(expected_words))] !=
                expected_words[:min(len(actual_words), len(expected_words))])
            self.assertTrue(len(different))
            row, col = map(int, different[0])
            evidence.append(dict(owner=side, mismatched_namespaces=mismatches,
                actual_event_kinds=actual_words[:, 0].tolist(), reference_event_kinds=expected_words[:, 0].tolist(),
                first_word_difference=dict(row=row, column=col, actual=int(actual_words[row,col]),
                    reference=int(expected_words[row,col])), actual_first_row=actual_words[0].tolist(),
                reference_first_row=expected_words[0].tolist(), full_reference_fold_error=reference_error.tolist()))
        print('DRIVER_EXACT_FULL_INPUT_GAP', json.dumps(evidence, sort_keys=True))


if __name__ == '__main__': unittest.main()
