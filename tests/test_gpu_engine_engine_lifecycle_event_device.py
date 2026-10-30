"""Actual-core literal lifecycle -> typed existing consumers; no games/models."""
from dataclasses import replace
import unittest
import numpy as np

import test_gpu_engine_engine_lifecycle_callbacks as fixtures


class LifecycleEventDeviceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        fixtures.LifecycleTests.setUpClass.__func__(cls)
        from gpu_engine import engine_lifecycle_event_device as d, engine_public_event_device as base
        from gpu_engine import public_team, public_items, public_field, public_damage, public_wire
        cls.d, cls.base, cls.wire = d, base, public_wire
        cls.species_catalog = public_team.compile_species(cls.host, cls.dex, cls.vocab, cls.abilities)
        cls.item_catalog = public_items.compile_items(cls.host, cls.vocab)
        cls.field_catalog = public_field.compile_field(cls.vocab)
        cls.damage_tables = public_damage.compile_damage(cls.vocab)
        cls.context_tables = cls.c.compile_context(cls.host, cls.abilities, cls.catalog,
            items=cls.item_catalog, field_catalog=cls.field_catalog, damage_tables=cls.damage_tables)
        cls.simple_context = cls.c.compile_context(cls.host, cls.abilities, cls.catalog)
        cls.projector = base.compile_projector(cls.producer, expected_binding=cls.producer.binding,
            expected_engine_binding=np.array(cls.producer.engine_binding, np.uint32))
        cls.compiled = d.compile_projector(cls.metadata, cls.projector, species_catalog=cls.species_catalog,
            source=cls.host, context_tables=cls.context_tables)

    state = fixtures.LifecycleTests.state
    chance = fixtures.LifecycleTests.chance
    kwargs = fixtures.LifecycleTests.kwargs
    initial = fixtures.LifecycleTests.initial
    step = fixtures.LifecycleTests.step
    same = fixtures.LifecycleTests.same

    def registered(self, batch=1, owner=0, compiled=None):
        return self.d.register(self.j.arange(1, batch+1, dtype=self.j.int32),
            self.j.full((batch,), owner, self.j.int32), compiled=self.compiled if compiled is None else compiled)

    def consumers(self, batch=1, context_tables=None):
        context_tables = self.context_tables if context_tables is None else context_tables
        beliefs = self.b.initialize(self.j.zeros((batch, 2, 6), self.j.int32), self.host.device)
        return (self.i.initialize(self.j.arange(1, batch+1, dtype=self.j.int32), source_binding=self.host.device.binding),
            beliefs, self.m.initialize(batch, species_binding=self.species_catalog.device.binding),
            self.c.initialize(beliefs, context_tables, species_binding=self.species_catalog.device.binding))

    def consume(self, state, frame, current, tables=None, context_tables=None):
        return self.d.project_and_fold(state, frame, self.compiled.device if tables is None else tables,
            current[0], current[1], self.host.device, mon_state=current[2], context=current[3],
            context_tables=self.context_tables if context_tables is None else context_tables)

    @staticmethod
    def carry(result):
        return result.cursor, result.beliefs, result.mon_state, result.context

    def source_words(self, lines, wire_state=None, items=True):
        kwargs = dict(species_catalog=self.species_catalog, source=self.host, moves=self.catalog,
            abilities=self.abilities, context_tables=self.context_tables if items else self.simple_context)
        if items:
            kwargs.update(items=self.item_catalog, field_catalog=self.field_catalog, damage_tables=self.damage_tables)
        state, projections = self.wire.project_lines(wire_state or self.wire.WireState(), lines, **kwargs)
        return state, np.array([e.words() for p in projections for e in p.events], np.int32)

    def words(self, projection, row=0):
        return np.array(projection.chunk.words[row])[np.array(projection.chunk.valid[row])]

    def assert_partial(self, projection):
        self.assertFalse(bool(projection.full_line_consumers_complete.any()))
        self.assertFalse(bool(projection.full_encoder_admitted.any()))
        self.assertFalse(bool(projection.B4_qualified.any()))
        self.assertTrue(bool((projection.refusal & self.e.UNINSTRUMENTED).all()))

    def test_actual_initialize_exact_source_raw_headers_and_existing_consumer_state(self):
        frame = self.initial()
        result = self.consume(self.registered(), frame, self.consumers())
        self.assertEqual(result.projection.error.tolist(), [0])
        self.assertEqual(result.fold.error.tolist(), [0])
        _, expected = self.source_words(['|switch|p1a: Snorlax|Snorlax, L100, M|100/100',
            '|switch|p2a: Absol|Absol, L100, M|100/100', '|turn|1'])
        np.testing.assert_array_equal(self.words(result.projection), expected)
        self.assertEqual(result.fold.mon_state.encounter[0, :, 0].tolist(), [1, 1])
        self.assertEqual(result.fold.mon_state.species[0, :, 0].tolist(),
            [self.species_catalog.species_id('Snorlax'), self.species_catalog.species_id('Absol')])
        self.assertEqual(result.fold.cursor.sequence.tolist(), [6])
        self.assertEqual(result.fold.cursor.turn.tolist(), [1])
        self.assertEqual(result.fold.context.damage.turn.tolist(), [1])
        self.assert_partial(result.projection)

    def test_actual_switch_then_move_resolves_causal_new_encounter_not_poststep_join(self):
        first = self.initial(self.state()._replace(species=self.state().species.at[0, 0, 5].set(3)))
        consumed = self.consume(self.registered(), first, self.consumers())
        frame = self.step(first, [[8, 0]])
        result = self.jax.jit(lambda state, f, cur, bel, mon, ctx:
            self.d.project_and_fold(state, f, self.compiled.device, cur, bel, self.host.device,
                mon_state=mon, context=ctx, context_tables=self.context_tables))(
                    consumed.projection.state, frame, *self.carry(consumed.fold))
        self.assertEqual(result.projection.error.tolist(), [0])
        self.assertEqual(result.fold.mon_state.active.tolist(), [[1, 0]])
        self.assertEqual(frame.after_registry.physical_active.tolist(), [[5, 0]])
        self.assertEqual(result.projection.state.cursor.generation.tolist(), [[1, 0]])
        words = self.words(result.projection)
        self.assertEqual(words[:, 0].tolist(), [21, 6, 21, 15, 21, 10, 21, 2, 21, 3, 21, 1])
        move = words[3]
        self.assertEqual(move[4:8].tolist(), [1, 0, 0, 1])
        self.assertEqual(words[5, 8], 2)
        self.assertEqual(result.fold.context.moves.uses[0, 1, 0, 0].item(), 1)
        self.assertGreater(float(result.fold.context.damage.last_taken[0, 0].sum()), 0)
        self.assertEqual(result.fold.context.damage.turn.tolist(), [2])
        # The new TURN's RAW_BEGIN is still turn1/residual, not turn2/action.
        self.assertEqual(words[-2, 2:4].tolist(), [1, 1])
        self.assertEqual(words[-1, 2:4].tolist(), [2, 0])
        wire_state, _ = self.source_words(['|switch|p1a: Snorlax|Snorlax, L100, M|100/100',
            '|switch|p2a: Absol|Absol, L100, M|100/100', '|turn|1'])
        hp = int(frame.callbacks.words[0, 2, 5])
        _, expected = self.source_words(['|switch|p1a: Gardevoir|Gardevoir, L100, M|100/100',
            '|move|p2a: Absol|Body Slam|p1a: Gardevoir', f'|-damage|p1a: Gardevoir|{hp}/100',
            '|', '|upkeep', '|turn|2'], wire_state)
        np.testing.assert_array_equal(words, expected)
        revisited = self.step(frame, [[4, 0]])
        returned = self.consume(result.projection.state, revisited, self.carry(result.fold))
        self.assertEqual(returned.projection.error.tolist(), [0])
        self.assertEqual(returned.fold.mon_state.active.tolist(), [[0, 0]])
        self.assertEqual(returned.projection.state.cursor.next_slot.tolist(), [[2, 1]])
        self.assertEqual(returned.projection.state.cursor.generation.tolist(), [[2, 0]])

    def test_both_owners_arrival_health_source_words_nearfull_and_unknown_details(self):
        state = self.state(2)
        state = state._replace(hp=state.hp.at[:, 1, 0].set(999), max_hp=state.max_hp.at[:, 1, 0].set(1000))
        frame = self.initial(state)
        observers = self.registered(2)._replace(owner=self.j.array([0, 1], self.j.int32))
        p = self.jax.jit(self.d.project_frame)(observers, frame, self.compiled.device)
        self.assertEqual(p.error.tolist(), [0, 0])
        self.assertEqual(p.chunk.words[:, 3, 13:15].tolist(), [[99, 100], [999, 1000]])
        self.assertEqual(p.chunk.words[:, 3, 11:13].tolist(), [[100, 1], [100, 1]])
        unknown = self.initial(self.state()._replace(gender=self.j.full_like(self.state().gender, -1)))
        refused = self.d.project_frame(self.registered(), unknown, self.compiled.device)
        self.assertFalse(bool(refused.component_admitted[0]))
        self.assertFalse(bool(refused.chunk.words.any()))
        self.assertFalse(bool(refused.state.admitted[0]))

    def test_typed_mode_has_no_arbitrary_raw_begin_and_pinned_source_matches(self):
        compiled = self.d.compile_projector(self.metadata, self.projector,
            species_catalog=self.species_catalog, source=self.host, context_tables=self.simple_context)
        frame = self.initial()
        result = self.consume(self.registered(compiled=compiled), frame, self.consumers(context_tables=self.simple_context),
            tables=compiled.device, context_tables=self.simple_context)
        self.assertEqual(result.projection.error.tolist(), [0])
        _, expected = self.source_words(['|switch|p1a: Snorlax|Snorlax, L100, M|100/100',
            '|switch|p2a: Absol|Absol, L100, M|100/100', '|turn|1'], items=False)
        np.testing.assert_array_equal(self.words(result.projection), expected)
        self.assertEqual(expected[:, 0].tolist(), [6, 6, 1])

        # Preserve the real direct-damage metadata: a consumer composition
        # lacking that consumer must refuse the whole next frame, not erase it.
        attack = self.step(frame, [[0, 0]])
        refused = self.consume(result.projection.state, attack, self.carry(result.fold),
            tables=compiled.device, context_tables=self.simple_context)
        self.assertFalse(bool(refused.projection.component_admitted[0]))
        self.assertFalse(bool(refused.projection.chunk.words.any()))
        self.assertFalse(bool(refused.projection.state.admitted[0]))
        self.assertEqual(refused.fold.cursor.sequence.tolist(), [3])

    def test_tampered_before_generation_after_and_suffix_refuse_atomically_and_latch_unknown(self):
        first = self.initial()
        consumed = self.consume(self.registered(), first, self.consumers())
        frame = self.step(first, [[4, 0]])
        for bad in (frame._replace(before_registry=frame.after_registry),
            frame._replace(callbacks=frame.callbacks._replace(event_generation=frame.callbacks.event_generation.at[0, 1, 0].set(0))),
            frame._replace(after_registry=frame.after_registry._replace(public_slot=frame.after_registry.public_slot.at[0, 0].set(4))),
            frame._replace(callbacks=frame.callbacks._replace(words=frame.callbacks.words.at[0, 4, 0].set(999)))):
            result = self.consume(consumed.projection.state, bad, self.carry(consumed.fold))
            self.assertFalse(bool(result.projection.component_admitted[0]))
            self.assertFalse(bool(result.projection.chunk.words.any()))
            self.assertEqual(result.projection.state.sequence.tolist(), [6])
            self.assertFalse(bool(result.projection.state.admitted[0]))
            for a, b in zip(self.jax.tree.leaves(self.carry(consumed.fold)), self.jax.tree.leaves(self.carry(result.fold))):
                np.testing.assert_array_equal(a, b)
            retry = self.consume(result.projection.state, frame, self.carry(result.fold))
            self.assertFalse(bool(retry.projection.component_admitted[0]))

    def test_source_and_expansion_overflow_no_partial_output_and_host_guards(self):
        overflow = self.initial(capacity=1)
        p = self.d.project_frame(self.registered(), overflow, self.compiled.device)
        self.assertFalse(bool(p.component_admitted[0]))
        self.assertFalse(bool(p.chunk.valid.any()))
        with self.assertRaises(ValueError): self.d.check_projector(replace(self.compiled))
        old = self.compiled.device
        object.__setattr__(self.compiled, 'device', old._replace(species=old.species.at[1].add(100)))
        try:
            with self.assertRaises(ValueError): self.d.check_projector(self.compiled)
        finally: object.__setattr__(self.compiled, 'device', old)
        self.assertIs(self.d.check_projector(self.compiled), old)
        forged = self.context_tables._replace(moves=self.context_tables.moves._replace(
            max_pp=self.context_tables.moves.max_pp.at[1].add(999)))
        with self.assertRaises(ValueError):
            self.d.compile_projector(self.metadata, self.projector, species_catalog=self.species_catalog,
                source=self.host, context_tables=forged)

        # Literal bounded callback primitive:33 supported blank raw lines need
        #66 typed words, so no64-event prefix may escape as a complete chunk.
        first = self.initial()
        consumed = self.consume(self.registered(), first, self.consumers())
        tape = self.life._tape(first.cursor, self.producer, 64)
        tape = self.jax.lax.fori_loop(0, 33, lambda _, t: self.life._clock(t,
            self.life.ACTION_END, self.j.ones(1, bool)), tape)
        frame = first._replace(callbacks=tape, before_registry=first.after_registry)
        result = self.consume(consumed.projection.state, frame, self.carry(consumed.fold))
        self.assertFalse(bool(result.projection.component_admitted[0]))
        self.assertFalse(bool(result.projection.chunk.valid.any()))
        self.assertFalse(bool(result.projection.chunk.words.any()))
        self.assertEqual(result.fold.cursor.sequence.tolist(), [6])
        self.assertTrue(bool(result.projection.refusal[0] & self.e.OVERFLOW))

    def test_actual_direct_faint_and_forced_resume_keep_same_turn_and_hp0fnt(self):
        state = self.state()._replace(hp=self.state().hp.at[0, 1, 0].set(1),
            status=self.state().status.at[0, 1, 0].set(4))
        first = self.initial(state)
        consumed = self.consume(self.registered(), first, self.consumers())
        knockout = self.step(first, [[0, 0]])
        result = self.consume(consumed.projection.state, knockout, self.carry(consumed.fold))
        self.assertEqual(result.projection.error.tolist(), [0])
        self.assertEqual(self.words(result.projection)[:, 0].tolist(), [21, 15, 21, 10, 21, 12])
        self.assertEqual(self.words(result.projection)[3, 8:].tolist(), [1, 0, 0, 0, 1, 1, 0, 0])
        self.assertTrue(bool(result.fold.mon_state.fainted[0, 1, 0]))
        self.assertEqual(result.fold.cursor.turn.tolist(), [1])
        resumed = self.step(knockout, [[0, 4]])
        next_result = self.consume(result.projection.state, resumed, self.carry(result.fold))
        self.assertEqual(next_result.projection.error.tolist(), [0])
        self.assertEqual(next_result.fold.mon_state.active.tolist(), [[0, 1]])
        self.assertEqual(next_result.fold.cursor.turn.tolist(), [2])
        self.assertEqual(next_result.projection.state.cursor.generation.tolist(), [[0, 1]])

    def test_actual_residual_double_replacement_preserves_phase_without_extra_upkeep(self):
        from pokezero.dex import ShowdownDex
        dex = ShowdownDex(dict(self.dex.moves, protect=replace(self.dex.moves['protect'], target='self')), {}, {})
        catalog = self.e.compile_catalog(self.metadata, self.engine_table, public_moves=self.catalog,
            dex=dex, exposure=self.exposure)
        projector = self.base.compile_projector(catalog, expected_binding=catalog.binding,
            expected_engine_binding=np.array(catalog.engine_binding, np.uint32))
        compiled = self.d.compile_projector(self.metadata, projector, species_catalog=self.species_catalog,
            source=self.host, context_tables=self.context_tables)
        state = self.state()
        state = state._replace(hp=state.hp.at[:, :, 0].set(1), status=state.status.at[:, :, 0].set(2),
            moves=self.j.full_like(state.moves, 3))
        first = self.initial(state, catalog=catalog)
        consumed = self.consume(self.registered(compiled=compiled), first, self.consumers(), tables=compiled.device)
        residual = self.step(first, [[0, 0]], catalog=catalog)
        second = self.consume(consumed.projection.state, residual, self.carry(consumed.fold), tables=compiled.device)
        self.assertEqual(second.projection.error.tolist(), [0])
        self.assertEqual(second.fold.cursor.turn.tolist(), [1])
        self.assertEqual(second.fold.cursor.phase.tolist(), [1])
        forced = self.step(residual, [[4, 4]], catalog=catalog)
        result = self.consume(second.projection.state, forced, self.carry(second.fold), tables=compiled.device)
        self.assertEqual(result.projection.error.tolist(), [0])
        self.assertEqual(self.words(result.projection)[:, 0].tolist(), [21, 6, 21, 6, 21, 1])
        self.assertEqual(self.words(result.projection)[:4, 2:4].tolist(), [[1, 1]]*4)
        self.assertEqual(result.projection.state.cursor.generation.tolist(), [[1, 1]])
        self.assertEqual(result.fold.cursor.upkeeps.tolist(), [1])
        self.assertEqual(result.fold.cursor.turn.tolist(), [2])
        self.assert_partial(result.projection)

    def test_mixed_batch_unknown_row_rollback_and_positive_hot_jit_reuse(self):
        state = self.state(2)
        unknown = self.initial(state._replace(gender=state.gender.at[0, 0, 0].set(-1)))
        registered = self.registered(2)
        initial = self.consumers(2)
        compiled = self.jax.jit(lambda state, frame, cur, bel, mon, ctx:
            self.d.project_and_fold(state, frame, self.compiled.device, cur, bel, self.host.device,
                mon_state=mon, context=ctx, context_tables=self.context_tables))
        result = compiled(registered, unknown, *initial)
        self.assertEqual(result.projection.component_admitted.tolist(), [False, True])
        self.assertFalse(bool(result.projection.chunk.words[0].any()))
        self.assertEqual(result.fold.cursor.sequence.tolist(), [0, 6])
        known = self.initial(state)
        positive = compiled(registered, known, *initial)
        self.assertEqual(positive.projection.error.tolist(), [0, 0])
        self.assertEqual(compiled._cache_size(), 1)
        program = str(self.jax.make_jaxpr(self.d.project_frame)(registered, known, self.compiled.device))
        self.assertNotIn('callback', program)


if __name__ == '__main__': unittest.main()
