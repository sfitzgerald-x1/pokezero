"""Small actual public folds/retained encoder and source-only v5 tail checks."""
from dataclasses import replace
import gc
import importlib.util
import inspect
import unittest
import weakref

import numpy as np


@unittest.skipUnless(importlib.util.find_spec('jax'), 'requires optional JAX')
class V5ObservationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import jax
        import jax.numpy as j
        import test_gpu_engine_v4_encoder as fixtures
        from gpu_engine import v5_observation as v5, public_belief as belief, v4_encoder
        from pokezero import v5_contract as contract
        from pokezero.v5_model import V5ModelConfig
        fixtures.V4EncoderTests.setUpClass()
        cls.fixture = fixtures.V4EncoderTests()
        cls.jax, cls.j, cls.v5, cls.belief, cls.encoder = jax, j, v5, belief, v4_encoder
        cls.contract = contract; cls.ModelConfig = V5ModelConfig
        cls.catalog = cls.fixture.catalog
        cls.adapter = cls.make_adapter(cls.catalog, cls.fixture.public.source)
        cls.state = cls.fixture.ingest([*cls.fixture.public.prefix, '|turn|1'])
        cls.encoded = {}
        encode = jax.jit(lambda state, own, observer: v4_encoder.encode(state, own, cls.catalog, observer_sides=observer))
        for owner in (0, 1):
            own = v4_encoder.parse_own_requests(cls.catalog, [cls.fixture.request(owner)], owner_sides=[owner])
            cls.encoded[owner] = encode(cls.state, own, j.asarray([owner], j.int32))
            if not bool(cls.encoded[owner].observation.admitted.all()):
                raise AssertionError('original retained fixture refused')

    @classmethod
    def make_adapter(cls, catalog, original_source):
        from gpu_engine.private_team import vocabulary_binding
        census = cls.contract.source_variant_census(original_source)
        vocabulary = cls.contract.LabelVocabulary(census['source_rows_sha256'], tuple(catalog.moves.names[1:]), census['species_variants'])
        category = np.asarray(vocabulary_binding(catalog.vocabulary), dtype='>u4').tobytes().hex()
        config = cls.ModelConfig(catalog.vocabulary.size, category, vocabulary, width=8, layers=1,
            attention_heads=2, expert_width=8, value_hidden=(8, 8))
        return cls.v5.compile_adapter(catalog, model_config=config)

    def append(self, owner=0, *, encoded=None, state=None, adapter=None):
        return self.v5.append_device(self.encoded[owner] if encoded is None else encoded,
            self.state if state is None else state, self.adapter if adapter is None else adapter,
            observer_sides=self.j.asarray([owner], self.j.int32))

    def same(self, left, right):
        for a, b in zip(self.jax.tree.leaves(left), self.jax.tree.leaves(right)):
            np.testing.assert_array_equal(a, b)

    def test_actual_retained_prefix_and_contract_numeric_rows_for_both_owners(self):
        for owner in (0, 1):
            out = self.append(owner); original = self.encoded[owner].observation
            self.assertEqual(out.admitted.tolist(), [True]); self.assertEqual(out.error.tolist(), [0])
            for name in ('numeric_features', 'categorical_ids', 'token_type_ids', 'attention_mask'):
                np.testing.assert_array_equal(getattr(out, name)[:, :23], getattr(original, name))
            np.testing.assert_array_equal(out.legal_action_mask, original.legal_action_mask)
            np.testing.assert_array_equal(out.switch_targets, original.switch_targets)
            np.testing.assert_array_equal(out.token_type_ids[:, 23:], np.full((1, 6), 7))
            self.assertTrue(np.asarray(out.attention_mask[:, 23:]).all())
            np.testing.assert_array_equal(out.categorical_ids[:, 23:], original.categorical_ids[:, 7:13])
            species = np.asarray(self.state.beliefs.species)[0, owner]
            for slot, sid in enumerate(species):
                name = self.adapter.model_config.label_vocabulary.species[sid - 1] if sid else None
                words = np.asarray(self.encoded[owner].posterior_mask)[0, slot]
                mask = int(words[0]) | int(words[1]) << 32
                prior = (1.,) * len(self.catalog.source.variants[sid]) if sid else ()
                posterior = self.contract.PublicPosterior(self.adapter.model_config.label_vocabulary,
                    name, mask, prior, self.contract.BELIEF_REASONS[int(out.posterior_reason[0, slot])],
                    bool(out.pin_conflict[0, slot]))
                np.testing.assert_allclose(out.numeric_features[0, 23 + slot], posterior.numeric_row(), atol=1e-7, rtol=0)
            columns = self.v5.actor_inputs(out, self.adapter)
            from gpu_engine import v5_model
            prepared = v5_model.prepare_inputs(columns, self.adapter.model_config)
            self.assertEqual(prepared['numeric_features'].shape, (1, 1, 29, 132))

    def test_unknown_slots_are_attended_zero_mass_without_refusal(self):
        out = self.append()
        self.assertTrue(bool(out.admitted[0]))
        # The only public opponent arrival is slot0; no unseen true species leaks.
        np.testing.assert_array_equal(out.posterior_weights[0, 1:], 0)
        np.testing.assert_array_equal(out.prior_weights[0, 1:], 0)
        np.testing.assert_array_equal(out.numeric_features[0, 24:, :128], 0)
        np.testing.assert_allclose(out.numeric_features[0, 24:, 130], 1 / 3, atol=1e-7)
        self.assertTrue(np.asarray(out.attention_mask[0, 24:]).all())
        # Refused original rows stay present; no refill or actor admission.
        encoded = self.encoded[0]._replace(observation=self.encoded[0].observation._replace(admitted=self.j.asarray([False])))
        refused = self.append(encoded=encoded)
        self.assertFalse(bool(refused.admitted[0])); self.assertTrue(int(refused.error[0]) & self.v5.V4_REFUSED)
        np.testing.assert_array_equal(refused.numeric_features[:, :23], encoded.observation.numeric_features)
        self.assertFalse(bool(self.v5.actor_inputs(refused, self.adapter)['history_mask'][0, 0]))

    def test_jit_exact_posterior_join_namespace_faults_and_no_private_surface(self):
        compiled = self.jax.jit(lambda encoded, state, observer:
            self.v5.append_device(encoded, state, self.adapter, observer_sides=observer))
        observer = self.j.asarray([0], self.j.int32)
        baseline = compiled(self.encoded[0], self.state, observer)
        self.same(baseline, self.append())
        for encoded in (self.encoded[0]._replace(posterior_weights=self.encoded[0].posterior_weights.at[0, 0, 0].add(.01)),
                self.encoded[0]._replace(posterior_weights=self.j.nextafter(self.encoded[0].posterior_weights,
                    self.j.full_like(self.encoded[0].posterior_weights, 2.))),
                self.encoded[0]._replace(posterior_mask=self.encoded[0].posterior_mask.at[0, 0, 1].set(1)),
                self.encoded[0]._replace(posterior_reason=self.encoded[0].posterior_reason.at[0, 0].set(3))):
            out = compiled(encoded, self.state, observer)
            self.assertFalse(bool(out.admitted[0])); self.assertTrue(int(out.error[0]) & self.v5.POSTERIOR)
        foreign = self.state._replace(binding=self.state.binding.at[0, 0].add(self.j.uint32(1)))
        self.assertFalse(bool(compiled(self.encoded[0], foreign, observer).admitted[0]))
        with self.assertRaises(ValueError):compiled(self.encoded[0], self.state, observer.astype(self.j.float32))
        self.assertEqual(set(inspect.signature(self.v5.append_device).parameters),
            {'encoded', 'public_state', 'adapter', 'observer_sides'})
        with self.assertRaises(TypeError):self.v5.append_device(self.encoded[0], self.state, self.adapter,
            observer_sides=observer, engine_private=object())
        columns = self.v5.actor_inputs(baseline, self.adapter)
        self.assertEqual(set(columns), {'schema', 'observation_schema', 'category_vocabulary_sha256',
            'label_vocabulary_sha256', 'view', 'categorical_ids', 'numeric_features', 'token_type_ids', 'attention_mask', 'history_mask'})

    def test_original_source_binding_reordering_and_copied_adapter_are_refused(self):
        model = self.adapter.model_config
        with self.assertRaisesRegex(ValueError, 'category'):
            self.v5.compile_adapter(self.catalog, model_config=replace(model, category_vocabulary_sha256='f' * 64))
        vocabulary = replace(model.label_vocabulary, source_binding_sha256='f' * 64)
        with self.assertRaisesRegex(ValueError, 'source-order'):
            self.v5.compile_adapter(self.catalog, model_config=replace(model, label_vocabulary=vocabulary))
        vocabulary = replace(model.label_vocabulary, species_variants=tuple(reversed(model.label_vocabulary.species_variants)))
        with self.assertRaisesRegex(ValueError, 'source-order'):
            self.v5.compile_adapter(self.catalog, model_config=replace(model, label_vocabulary=vocabulary))
        with self.assertRaises(ValueError):self.v5.check_adapter(replace(self.adapter))
        self.assertIs(self.v5.check_adapter(self.adapter), self.catalog)
        with self.assertRaises(ValueError):self.jax.jit(lambda x: (self.v5.check_adapter(self.adapter), x)[1])(self.j.int32(0))

    def test_mixed_B8_refusals_keep_every_row_and_global8_namespace_is_global(self):
        def repeat(value):
            return self.j.repeat(value, 8, axis=0) if value.ndim and value.shape[0] == 1 else value
        state = self.jax.tree.map(repeat, self.state)
        encoded = self.jax.tree.map(repeat, self.encoded[0])
        observer = self.j.zeros(8, self.j.int32)
        compiled = self.jax.jit(lambda encoded, state, observer:
            self.v5.append_device(encoded, state, self.adapter, observer_sides=observer))
        baseline = compiled(encoded, state, observer)
        self.assertTrue(np.asarray(baseline.admitted).all())
        encoded = encoded._replace(posterior_weights=encoded.posterior_weights.at[3, 0, 0].add(.01),
            observation=encoded.observation._replace(admitted=encoded.observation.admitted.at[5].set(False)))
        output = compiled(encoded, state, observer.at[7].set(2))
        np.testing.assert_array_equal(output.admitted, [True, True, True, False, True, False, True, False])
        np.testing.assert_array_equal(output.numeric_features[:, :23], encoded.observation.numeric_features)
        np.testing.assert_array_equal(self.v5.actor_inputs(output, self.adapter)['history_mask'][:, 0], output.admitted)
        self.assertEqual(output.numeric_features.shape[0], 8)
        visible = state.context.refresh.controller.visible
        foreign = visible._replace(source_binding=visible.source_binding.at[3].add(self.j.uint32(1)))
        controller = state.context.refresh.controller._replace(visible=foreign)
        context = state.context._replace(refresh=state.context.refresh._replace(controller=controller))
        # This eight-word digest is global, even when batch size is also eight.
        refused = compiled(self.jax.tree.map(repeat, self.encoded[0]), state._replace(context=context), observer)
        self.assertFalse(np.asarray(refused.admitted).any())

    def test_boundary_value_copy_preserves_original_pin_conflict_counters(self):
        from gpu_engine import public_wire
        catalog = self.catalog
        state = self.encoder.initialize(catalog, stream_ids=self.j.asarray([1], self.j.int32))
        wire, projections = public_wire.project_lines(public_wire.WireState(), self.fixture.public.prefix,
            **self.encoder.wire_kwargs(catalog))
        for chunk in self.fixture.chunks(projections):
            state, error = self.encoder.fold_chunk(state, chunk, catalog)
            self.assertEqual(error.tolist(), [0])
        variants = catalog.source.variants[catalog.source.species_id('Snorlax')]
        victim = next(v for v in variants if 'surf' not in tuple(m.lower() for m in v.moves))
        valid = self.j.zeros_like(state.beliefs.species, bool).at[:, 1, 0].set(True)
        pinned, _ = self.belief.narrow_pin(state.beliefs, catalog.source.pin_mask('Snorlax', [victim.to_summary()]), valid=valid)
        state = state._replace(beliefs=pinned)
        _, projections = public_wire.project_lines(wire, ['|turn|1', '|move|p1a: Snorlax|Surf|p2a: Absol'],
            **self.encoder.wire_kwargs(catalog))
        for chunk in self.fixture.chunks(projections):
            state, error = self.encoder.fold_chunk(state, chunk, catalog)
            self.assertEqual(error.tolist(), [0])
        own = self.encoder.parse_own_requests(catalog, [self.fixture.request(1)], owner_sides=[1])
        encoded = self.encoder.encode(state, own, catalog, observer_sides=self.j.asarray([1], self.j.int32))
        before = [np.asarray(a).copy() for a in self.jax.tree.leaves(state)]
        first = self.append(1, encoded=encoded, state=state)
        second = self.append(1, encoded=encoded, state=state)
        self.assertTrue(bool(first.admitted[0])); self.assertTrue(bool(first.pin_conflict[0, 0]))
        self.assertGreater(int(state.beliefs.conflicts[0, 1, 0]), 0)
        self.same(first, second)
        for old, new in zip(before, self.jax.tree.leaves(state)):np.testing.assert_array_equal(old, new)

    def test_all64_original_rows_and_high_word_order_without_truncation(self):
        from gpu_engine import public_belief, public_wire
        raw = self.fixture.public.source
        original = raw.universes['snorlax']
        variants = tuple(replace(original.variants[i % len(original.variants)], variant_id=f'ordered{i}') for i in range(64))
        raw = replace(raw, universes=dict(raw.universes, snorlax=replace(original, variants=variants)))
        source = public_belief.compile_source(raw)
        catalog = self.encoder.compile_encoder(source, self.catalog.dex, self.catalog.vocabulary,
            checkpoint_sha256='a' * 64, feature_masks=self.fixture.masks)
        adapter = self.make_adapter(catalog, raw)
        state = self.encoder.initialize(catalog, stream_ids=self.j.asarray([1], self.j.int32))
        _, projections = public_wire.project_lines(public_wire.WireState(), [*self.fixture.public.prefix, '|turn|1'],
            **self.encoder.wire_kwargs(catalog))
        # Chunk binding source belongs to this new64-row catalog, not the fixture.
        from gpu_engine import public_inputs
        p = projections[0]
        bindings = {name: np.asarray(getattr(p, name), np.uint32) for name in (
            'species_binding','ability_binding','move_binding','item_binding','context_binding','field_binding',
            'active_binding','damage_binding','type_binding','sleep_binding','toxic_binding','tendency_binding')}
        for chunk in public_inputs.pack_chunks(np.asarray([1], np.int32), [[event for p in projections for event in p.events]],
                source_binding=source.device.binding, **bindings):
            state, error = self.encoder.fold_chunk(state, chunk, catalog); self.assertEqual(error.tolist(), [0])
        own = self.encoder.parse_own_requests(catalog, [self.fixture.request(1)], owner_sides=[1])
        encoded = self.encoder.encode(state, own, catalog, observer_sides=self.j.asarray([1], self.j.int32))
        out = self.append(1, encoded=encoded, state=state, adapter=adapter)
        self.assertTrue(bool(out.admitted[0]))
        np.testing.assert_allclose(out.numeric_features[0, 23, :128], 1 / 64, rtol=0, atol=0)
        np.testing.assert_array_equal(out.numeric_features[0, 23, 128:], [1, 1, 0, 0])
        self.assertEqual(tuple(v.variant_id for v in source.variants[source.species_id('Snorlax')]),
            tuple(f'ordered{i}' for i in range(64)))
        # Existing public sampler keeps high bits and the duplicate row identities.
        mask = np.asarray(source.pin_mask('Snorlax', [variants[63].to_summary()]))
        self.assertTrue(int(mask[1]) & (1 << 31))
        bad = replace(raw, universes=dict(raw.universes, snorlax=replace(original, variants=variants + (variants[0],))))
        with self.assertRaisesRegex(ValueError, '64'):public_belief.compile_source(bad)
        reference = weakref.ref(catalog)
        del catalog; gc.collect()
        self.assertIsNone(reference())
        with self.assertRaises(ValueError):self.v5.check_adapter(adapter)


if __name__ == '__main__':unittest.main()
