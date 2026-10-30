"""Actual tiny constructor/core -> FULL writer words/consumers; no model/game."""
from dataclasses import replace
from copy import deepcopy
import json
import unittest

import jax
import jax.numpy as j
import numpy as np

from gpu_engine import engine_full_v4_event_device as d
from gpu_engine import engine_lifecycle_callbacks as life, engine_public_event_device as base, engine_own_facts as own
from gpu_engine import public_wire, public_inputs as inputs, v4_encoder as v4, transition
import test_gpu_engine_engine_own_facts as fixture


class FullV4EventDeviceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        fixture.DeviceOwnFactsTests.setUpClass()
        cls.f = fixture.DeviceOwnFactsTests
        cls.projector = base.compile_projector(cls.f.lifecycle, expected_binding=cls.f.lifecycle.binding,
            expected_engine_binding=np.array(cls.f.lifecycle.engine_binding, np.uint32))
        cls.compiled = d.compile_adapter(cls.f.metadata, cls.projector, catalog=cls.f.catalog)
        cls.fast = staticmethod(jax.jit(lambda state, frame, public:
            d.project_and_fold(state, frame, cls.compiled.device, public, cls.f.catalog)))

    def same(self, a, b):
        for left, right in zip(jax.tree.leaves(a), jax.tree.leaves(b)):
            np.testing.assert_array_equal(left, right)

    def kwargs(self):
        return dict(catalog=self.f.lifecycle, expected_binding=self.f.lifecycle.binding,
            expected_engine_binding=np.array(self.f.lifecycle.engine_binding, np.uint32))

    def initial(self, pool=None):
        constructed = self.f.construct(pool)
        before = life.register(constructed.cursor.stream_id, **self.kwargs())
        return life.LifecycleResult(constructed.state, constructed.cursor, constructed.callbacks,
            life.snapshot(before), constructed.registry, j.zeros(2, bool), j.zeros(2, j.int32))

    def observer(self, frame, owners):
        return (d.register(frame.cursor.stream_id, owners, compiled=self.compiled),
            v4.initialize(self.f.catalog, stream_ids=frame.cursor.stream_id, owner_sides=owners))

    def chance(self):
        draws = j.zeros((2, 2, 12), j.int32).at[:, :, 4].set(1)
        return transition.ChanceInputs(draws, *[j.zeros((2, 2), j.int32) for _ in range(4)], j.zeros(2, j.int32))

    def step(self, frame, actions):
        return life.trace_step(frame.state, j.array(actions, j.int32), self.f.table,
            cursor=frame.cursor, chance_inputs=self.chance(), **self.kwargs())

    @staticmethod
    def words(p, row): return np.array(p.chunk.words[row])[np.array(p.chunk.valid[row])]

    def reference(self, stream, frame, wire_states):
        # Test-only strings from the separately reviewed causal typed producer.
        # The adapter is NOT used to choose controls, metadata or ordering.
        semantic = d.causal.project_frame(stream.causal, frame, self.compiled.causal.device)
        self.assertEqual(semantic.error.tolist(), [0, 0])
        projections = []; new_states = []
        statuses = ('', ' slp', ' par', ' brn', ' frz', ' psn', ' tox')
        for row in range(2):
            lines = []
            active = dict(wire_states[row].active_idents())
            def ident(side): return active[side]
            for w in np.array(semantic.chunk.words[row])[np.array(semantic.chunk.valid[row])]:
                kind, _, turn, _, actor, _, target, _, *p = map(int, w)
                if kind == inputs.Kind.SWITCH:
                    name = self.f.catalog.species.names[p[1]]
                    label = f'p{target+1}a: '+name
                    active[target] = label
                    # Keep the same encountered ident on subsequent declarations.
                    detail = name+f', L{p[3]}'+('' if p[4] == 0 else ', M' if p[4] == 1 else ', F')
                    lines.append(f'|switch|{label}|{detail}|{p[5]}/{p[6]}'+statuses[p[7]])
                elif kind == inputs.Kind.MOVE:
                    lines.append(f'|move|{ident(actor)}|{self.f.catalog.moves.names[p[1]]}|{ident(target)}')
                elif kind == inputs.Kind.CONDITION:
                    condition = '0 fnt' if p[4] else f'{p[1]}/{p[2]}'+statuses[p[3]]
                    lines.append(f'|-damage|{ident(target)}|{condition}')
                elif kind == inputs.Kind.FAINT: lines.append('|faint|'+ident(target))
                elif kind == inputs.Kind.TURN: lines.append(f'|turn|{turn}')
                elif kind == inputs.Kind.ACTION_CHUNK_END: lines.append('|')
                elif kind == inputs.Kind.UPKEEP: lines.append('|upkeep')
                else: self.fail('unexpected causal semantic event')
            new, ps = public_wire.project_lines(wire_states[row], lines, **v4.wire_kwargs(self.f.catalog))
            projections.append([e for p in ps for e in p.events]); new_states.append(new)
        ns = {name:np.array(getattr(self.compiled.device.namespaces, name)) for name in d._NAMES if name != 'source_binding'}
        chunk = next(inputs.pack_chunks(np.array(frame.cursor.stream_id), projections,
            source_binding=np.array(self.f.catalog.source.device.binding), **ns))
        return chunk, new_states

    def compare(self, stream, frame, current, wire_states):
        reference, new_wire = self.reference(stream, frame, wire_states)
        expected, error = v4.fold_chunk(current, reference, self.f.catalog)
        self.assertEqual(error.tolist(), [0, 0])
        result = self.fast(stream, frame, current)
        self.assertEqual(result.projection.error.tolist(), [0, 0])
        self.same(result.projection.chunk, reference)
        self.same(result.public, expected)
        self.assertFalse(bool(result.projection.full_line_consumers_complete.any()))
        self.assertFalse(bool(result.projection.full_encoder_admitted.any()))
        self.assertFalse(bool(result.projection.B4_qualified.any()))
        self.assertTrue(bool((result.projection.refusal & d.engine.UNINSTRUMENTED).all()))
        return result, new_wire

    def test_real_constructor_both_independent_owners_exact_full_words_fold_and_nearfull99(self):
        arrays = {name:np.array(value, copy=True) for name, value in self.f.arrays.items()}
        arrays['stats'][1, 0, 0] = 1000
        # Actual initialize preserves this constructor HP; then tiny source
        # literal snapshot exercises owner exact vs peer percentage directly.
        frame = self.initial(self.f.make_pool(arrays))
        state = frame.state._replace(hp=frame.state.hp.at[:, 1, 0].set(999), entry_pending=j.ones((2, 2), bool))
        before = life.register(frame.cursor.stream_id, **self.kwargs())
        frame = life.initialize_with_lifecycle(state, self.f.table, cursor=before,
            tie_coin=j.zeros(2, j.int32), **self.kwargs())
        for side in (0, 1):
            stream, current = self.observer(frame, j.full(2, side, j.int32))
            result, _ = self.compare(stream, frame, current, [public_wire.WireState()]*2)
            w = self.words(result.projection, 0)
            self.assertEqual(w[:, 0].tolist(), [21, 30, 32, 6, 31, 29]*2+[21, 30, 32, 1, 31, 29])
            self.assertEqual(w[0, 8:12].tolist(), [128, 0, 0, 2])
            self.assertEqual(w[9, 13:15].tolist(), [999, 1000] if side == 1 else [99, 100])
            self.assertEqual(result.public.cursor.sequence.tolist(), [18, 18])
            self.assertEqual(result.public.mons.encounter[:, :, 0].tolist(), [[1, 1], [1, 1]])

    def test_actual_move_direct_hp_clocks_and_switch_nonphysical_encounters_hot_jit(self):
        frame = self.initial()
        owners = j.array([0, 1], j.int32)
        stream, current = self.observer(frame, owners)
        result, wire = self.compare(stream, frame, current, [public_wire.WireState()]*2)
        # One real core invocation is compared to the unchanged legacy result.
        old = frame.state
        frame = self.step(frame, [[8, 0], [8, 0]])
        legacy = transition.step_with_chance(old, j.array([[8, 0], [8, 0]], j.int32), self.f.table, self.chance())
        self.same(frame.state, legacy.state)
        result, wire = self.compare(result.projection.state, frame, result.public, wire)
        self.assertEqual(frame.after_registry.physical_active[:, 0].tolist(), [5, 5])
        self.assertEqual(result.public.mons.active[:, 0].tolist(), [1, 1])
        self.assertEqual(result.projection.state.causal.cursor.generation[:, 0].tolist(), [1, 1])
        frame = self.step(frame, [[1, 1], [1, 1]])  # actual hiddenpowerice engine alias
        result, _ = self.compare(result.projection.state, frame, result.public, wire)
        moves = self.words(result.projection, 0)
        moves = moves[moves[:, 0] == int(inputs.Kind.MOVE)]
        self.assertTrue(len(moves))
        self.assertEqual(set(moves[:, 9].tolist()), {self.f.catalog.moves.move_id('hiddenpower')})
        self.assertTrue(bool(result.public.context.moves.uses.any()))
        self.assertEqual(result.public.cursor.turn.tolist(), [3, 3])
        self.assertEqual(self.fast._cache_size(), 1)
        closed = jax.make_jaxpr(lambda a, b, c: d.project_and_fold(a, b,
            self.compiled.device, c, self.f.catalog))(stream, self.initial(), current)
        def primitive_names(program):
            names = set()
            for eq in program.jaxpr.eqns:
                names.add(eq.primitive.name)
                for value in eq.params.values():
                    for item in value if isinstance(value, (tuple, list)) else (value,):
                        if hasattr(item, 'jaxpr'): names.update(primitive_names(item))
            return names
        self.assertFalse(primitive_names(closed) & {'pure_callback', 'io_callback', 'host_callback'})

    def test_real_ko_literal_faint_and_forced_replacement_both_owners(self):
        arrays = {name:np.array(value, copy=True) for name, value in self.f.arrays.items()}
        arrays['stats'][1, 0, 0] = 1
        frame = self.initial(self.f.make_pool(arrays))
        stream, current = self.observer(frame, j.array([0, 1], j.int32))
        result, wire = self.compare(stream, frame, current, [public_wire.WireState()]*2)
        frame = self.step(frame, [[0, 0], [0, 0]])
        result, wire = self.compare(result.projection.state, frame, result.public, wire)
        for row in (0, 1):
            words = self.words(result.projection, row)
            self.assertTrue(bool(((words[:, 0] == 21) & (words[:, 12] == 3)).any()))
            self.assertTrue(bool((words[:, 0] == 12).any()))
        self.assertFalse(bool(result.projection.state.live[:, 1].any()))
        frame = self.step(frame, [[0, 4], [0, 4]])
        result, _ = self.compare(result.projection.state, frame, result.public, wire)
        self.assertEqual(result.public.mons.active[:, 1].tolist(), [1, 1])

    def test_expansion_overflow_and_unknown_details_are_whole_row_refusals(self):
        frame = self.initial(); stream, current = self.observer(frame, j.array([0, 1], j.int32))
        result, _ = self.compare(stream, frame, current, [public_wire.WireState()]*2)
        frame = self.step(frame, [[0, 0], [0, 0]])
        unknown = frame._replace(callbacks=frame.callbacks._replace(refusal=frame.callbacks.refusal | life.UNKNOWN_DETAILS))
        refused = self.fast(result.projection.state, unknown, result.public)
        self.assertFalse(bool(refused.projection.component_admitted.any()))
        self.assertFalse(bool(refused.projection.chunk.valid.any()))
        self.same(refused.public, result.public)
        self.assertFalse(bool(refused.projection.state.causal.admitted.any()))
        mixed = frame._replace(callbacks=frame.callbacks._replace(
            refusal=frame.callbacks.refusal.at[0].set(frame.callbacks.refusal[0] | life.UNKNOWN_DETAILS)))
        isolated = self.fast(result.projection.state, mixed, result.public)
        self.assertNotEqual(isolated.projection.error[0].item(), 0)
        self.assertEqual(isolated.projection.error[1].item(), 0)
        self.assertFalse(bool(isolated.projection.chunk.valid[0].any()))
        self.assertTrue(bool(isolated.projection.chunk.valid[1].any()))
        self.assertEqual(isolated.public.cursor.sequence[0].item(), result.public.cursor.sequence[0].item())
        self.assertGreater(isolated.public.cursor.sequence[1].item(), result.public.cursor.sequence[1].item())
        # Adversarial capacity fixture: repeat one actual declaration's words,
        # with the real closing clock receipts. Not claimed as a genuine core
        # trace. This isolates expansion overflow from upstream payload errors.
        tape = frame.callbacks
        move = int(np.flatnonzero(np.array(tape.words[0, :, 0]) == d.engine.MOVE)[0])
        closing = np.flatnonzero(np.isin(np.array(tape.words[0, :, 0]), [life.ACTION_END, life.UPKEEP, life.TURN]))
        selected = j.array([move]*20+closing.tolist(), j.int32)
        def repeated(array):
            return j.zeros_like(array).at[:, :len(selected)].set(array[:, selected])
        expanded = tape._replace(words=repeated(tape.words), event_turn=repeated(tape.event_turn),
            event_phase=repeated(tape.event_phase), event_generation=repeated(tape.event_generation),
            switch_details=repeated(tape.switch_details), count=j.full(2, len(selected), j.int32))
        over = d.project_frame(result.projection.state, frame._replace(callbacks=expanded), self.compiled.device)
        self.assertTrue(bool((over.refusal & d.engine.OVERFLOW).all()))
        self.assertFalse(bool(over.chunk.valid.any()))
        self.assertFalse(bool(over.chunk.words.any()))
        self.assertEqual(over.state.sequence.tolist(), result.projection.state.sequence.tolist())
        # Sequence bound refuses ALL generated rows, including a valid prefix.
        huge = result.projection.state._replace(sequence=j.full(2, 2**31-3, j.int32))
        refused = d.project_frame(huge, frame, self.compiled.device)
        self.assertFalse(bool(refused.chunk.words.any()))
        self.assertFalse(bool(refused.chunk.valid.any()))

    def test_off_owner_hidden_sets_health_and_stats_do_not_enter_admitted_public_state(self):
        baseline = self.initial()
        for owner in (0, 1):
            arrays = {name:np.array(value, copy=True) for name, value in self.f.arrays.items()}
            peer = 1-owner
            # Actual owning constructor has different hidden active exact HP,
            # stats, ability, item and bench sets; observed arrival facts stay
            # the same (full peer HP100/100, same visible species/details).
            arrays['stats'][peer] += 41
            arrays['ability'][peer] = np.roll(arrays['ability'][peer], 1)
            arrays['item'][peer] = 0
            arrays['hidden_power_type'][peer] = 9
            arrays['hidden_power_power'][peer] = 31
            arrays['moves'][peer] = arrays['moves'][peer, :, ::-1]
            arrays['pp'][peer] = arrays['pp'][peer, :, ::-1]
            changed = self.initial(self.f.make_pool(arrays))
            owners = j.full(2, owner, j.int32)
            a, public_a = self.observer(baseline, owners)
            b, public_b = self.observer(changed, owners)
            left = self.fast(a, baseline, public_a)
            right = self.fast(b, changed, public_b)
            self.assertEqual(left.projection.error.tolist(), [0, 0])
            self.assertEqual(right.projection.error.tolist(), [0, 0])
            self.assertNotEqual(baseline.state.max_hp[0, peer, 0].item(), changed.state.max_hp[0, peer, 0].item())
            self.same(left.projection.chunk, right.projection.chunk)
            self.same(left.public, right.public)

    def test_fresh_original_guards_after_hot_jit_and_namespace_drift_rollback(self):
        frame = self.initial(); stream, current = self.observer(frame, j.array([0, 1], j.int32))
        self.fast(stream, frame, current)
        original = self.compiled.device
        object.__setattr__(self.compiled, 'device', original._replace(templates=np.array(original.templates)))
        try:
            with self.assertRaises(ValueError): d.consume(stream, frame, current, compiled=self.compiled)
        finally: object.__setattr__(self.compiled, 'device', original)
        index = self.f.catalog.vocabulary._index; key = next(iter(index)); old = index[key]
        index[key] = old+1
        try:
            with self.assertRaises(ValueError): d.consume(stream, frame, current, compiled=self.compiled)
        finally: index[key] = old
        self.assertIs(d.check_adapter(self.compiled), original)
        bad = original._replace(namespaces=original.namespaces._replace(context_binding=j.zeros(8, j.uint32)))
        refused = d.project_and_fold(stream, frame, bad, current, self.f.catalog)
        self.assertEqual(refused.projection.error.tolist(), [int(inputs.Error.SOURCE)]*2)
        self.same(refused.public, current)
        self.assertFalse(bool(refused.projection.chunk.valid.any()))
        wrong_owner = current._replace(context=current.context._replace(toxic=current.context.toxic._replace(
            visibility=current.context.toxic.visibility[:, ::-1])))
        refused = self.fast(stream, frame, wrong_owner)
        self.assertEqual(refused.projection.error.tolist(), [int(inputs.Error.STREAM)]*2)
        self.same(refused.public, wrong_owner)

    def test_repeated_species_without_public_nickname_receipt_refuses_identity(self):
        arrays = {name:np.array(value, copy=True) for name, value in self.f.arrays.items()}
        # Distinct physical individual5 has the same visible species as0. The
        # source wire would reuse that exact species-form raw ident; no public
        # nickname evidence exists in LifecycleTape to invent a distinct ident.
        arrays['species'][0, 5] = arrays['species'][0, 0]
        frame = self.initial(self.f.make_pool(arrays))
        stream, public = self.observer(frame, j.array([0, 1], j.int32))
        initial = self.fast(stream, frame, public)
        self.assertEqual(initial.projection.error.tolist(), [0, 0])
        switched = self.step(frame, [[8, 0], [8, 0]])
        result = self.fast(initial.projection.state, switched, initial.public)
        self.assertEqual(result.projection.error.tolist(), [int(inputs.Error.IDENTITY)]*2)
        self.assertFalse(bool(result.projection.chunk.words.any()))
        self.same(result.public, initial.public)

    def test_actual_full_initial_writer_reports_existing_posterior_and_cell_coverage(self):
        frame = self.initial()
        owners = j.array([0, 1], j.int32)
        stream, public = self.observer(frame, owners)
        result = self.fast(stream, frame, public)
        self.assertEqual(result.projection.error.tolist(), [0, 0])
        facts = own.produce_own_facts(self.f.start, self.f.compiled, owner_sides=owners)
        self.assertEqual(facts.admitted.tolist(), [True, True])
        # Existing retained173 writer and its actual posterior outputs. No
        # replacement posterior or invented all-consumer admission is produced.
        encoded = jax.jit(lambda state, facts: own._encode(state, facts, owners,
            frame.cursor.stream_id, self.f.catalog))(result.public, facts.facts)
        a = encoded.observation
        numeric = j.sum(a.attention_mask[:, :, None] & ~a.numeric_written, axis=(1, 2))
        categorical = j.sum(a.attention_mask[:, :, None] & ~a.categorical_written, axis=(1, 2))
        report = dict(public_error=result.projection.error.tolist(), own_admitted=facts.admitted.tolist(),
            writer_admitted=a.admitted.tolist(), pending_columns=encoded.pending_numeric_columns.tolist(),
            unwritten_attended_numeric=numeric.tolist(), unwritten_attended_categorical=categorical.tolist(),
            posterior_mask_shape=list(encoded.posterior_mask.shape), posterior_weights_shape=list(encoded.posterior_weights.shape),
            full_line_consumers_complete=result.projection.full_line_consumers_complete.tolist(),
            scientific_full_encoder_admitted=encoded.full_encoder_admitted.tolist())
        print('ACTUAL_FULL_INITIAL_WRITER', json.dumps(report, sort_keys=True))
        self.assertEqual(encoded.pending_numeric_columns.tolist(), [])
        self.assertFalse(bool(encoded.full_encoder_admitted.any()))
        self.assertFalse(bool(result.projection.B4_qualified.any()))

    def test_source_mudshot_queue_bit_survives_absent_public_target(self):
        from pokezero.dex import ShowdownDex
        from pokezero.category_vocab import CategoryVocabulary
        # Add a literal source schema, not a mechanics admission roster. An
        # actual core declaration supplies the receipt. The one-event prefix
        # below is explicitly a trusted-lower boundary fixture, not a claim
        # that an ordinary core frame declares against a dead target.
        metadata = deepcopy(self.f.metadata)
        metadata['vocabulary']['moves'].append('mudshot')
        metadata['moves'].append(dict(id='mudshot', base_power=55, type=5, source_type='Ground',
            category='Physical', pp=15, accuracy=95, priority=0, flags={}))
        mud = replace(self.f.dex.moves['surf'], id='mudshot', name='mudshot', type='Ground',
            category='Physical', gen3_category='Physical', base_power=55, pp=15, accuracy=95,
            effect_label='lower_foe_spe', effect_chance=100)
        dex = ShowdownDex(dict(self.f.dex.moves, mudshot=mud), self.f.dex.species, self.f.dex.type_chart)
        vocab = CategoryVocabulary(tuple(sorted(set(self.f.catalog.vocabulary.tokens) |
            {'move:mudshot', 'belief:possible_move:mudshot', 'move_effect:lower_foe_spe'})))
        catalog = v4.compile_encoder(self.f.host, dex, vocab, checkpoint_sha256='a'*64,
            feature_masks=self.f.catalog.masks.__dict__)
        table = transition.tables_from_metadata(metadata)
        events = d.engine.compile_catalog(metadata, table, public_moves=catalog.moves, dex=dex,
            exposure=d.engine.HealthExposure(d.engine.SHOWDOWN, 'gen3randombattle', True, False))
        projector = base.compile_projector(events, expected_binding=events.binding,
            expected_engine_binding=np.array(events.engine_binding, np.uint32))
        compiled = d.compile_adapter(metadata, projector, catalog=catalog)
        streams = self.f.start.cursor.stream_id
        kwargs = dict(catalog=events, expected_binding=events.binding,
            expected_engine_binding=np.array(events.engine_binding, np.uint32))
        cursor = life.register(streams, **kwargs)
        state = self.f.start.state._replace(entry_pending=j.ones((2, 2), bool),
            moves=self.f.start.state.moves.at[:, 0, 0, 0].set(len(metadata['moves'])-1))
        frame = life.initialize_with_lifecycle(state, table, cursor=cursor, tie_coin=j.zeros(2, j.int32), **kwargs)
        stream = d.register(streams, j.zeros(2, j.int32), compiled=compiled)
        initial = d.project_frame(stream, frame, compiled.device)
        self.assertEqual(initial.error.tolist(), [0, 0])
        prefix = []
        for w in self.words(initial, 0):
            if w[0] != inputs.Kind.SWITCH: continue
            name = catalog.species.names[int(w[9])]
            prefix.append(f'|switch|p{w[6]+1}a: {name}|{name}, L{w[11]}'+
                (', M' if w[12] == 1 else ', F' if w[12] == 2 else '')+f'|{w[13]}/{w[14]}')
        wire, _ = public_wire.project_lines(public_wire.WireState(), prefix+['|turn|1'], **v4.wire_kwargs(catalog))
        actual = life.trace_step(frame.state, j.zeros((2, 2), j.int32), table, cursor=frame.cursor,
            chance_inputs=self.chance(), **kwargs)
        index = int(np.flatnonzero((np.array(actual.callbacks.words[0, :, 0]) == d.engine.MOVE) &
            (np.array(actual.callbacks.words[0, :, 1]) == 0))[0])
        tape = actual.callbacks
        def one(array): return j.zeros_like(array).at[:, 0].set(array[:, index])
        tape = tape._replace(words=one(tape.words), count=j.ones(2, j.int32),
            event_turn=one(tape.event_turn), event_phase=one(tape.event_phase),
            event_generation=one(tape.event_generation), switch_details=one(tape.switch_details), cursor=frame.cursor)
        before = life.snapshot(frame.cursor)
        literal = life.LifecycleResult(None, frame.cursor, tape, before, before, j.zeros(2, bool), j.zeros(2, j.int32))
        fast = jax.jit(d.project_frame)
        for live in (True, False):
            current = initial.state._replace(live=initial.state.live.at[:, 1].set(live))
            result = fast(current, literal, compiled.device)
            self.assertEqual(result.error.tolist(), [0, 0])
            source = replace(wire, live=(True, live))
            line = f'|move|{wire.active_idents()[0]}|mudshot|{wire.active_idents()[1]}'
            _, reference = public_wire.project_line(source, line, **v4.wire_kwargs(catalog))
            expected = np.array([e.words() for e in reference.events])
            np.testing.assert_array_equal(self.words(result, 0), expected)
            word = expected[expected[:, 0] == inputs.Kind.MUDSHOT][0]
            self.assertEqual(word[12], 1)
            self.assertEqual(word[9:12].tolist(), [1, 0, 1] if live else [-1, -1, 0])
            self.assertTrue(bool(expected[0, 8] & (1 << 16)))
        self.assertEqual(fast._cache_size(), 1)


if __name__ == '__main__': unittest.main()
