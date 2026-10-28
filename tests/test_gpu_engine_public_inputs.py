"""Lossless chunking, transactional device ordering, and public fact routing."""
import importlib.util
import unittest


@unittest.skipUnless(importlib.util.find_spec("jax"), "requires isolated JAX")
class PublicInputsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import jax
        import jax.numpy as j
        import numpy as np
        from gpu_engine import public_belief as belief, public_inputs as inputs
        from pokezero.randbat import (Gen3RandbatSource, Gen3RandbatSpeciesUniverse,
                                     Gen3RandbatVariant, RandbatSourceMetadata)
        cls.jax, cls.j, cls.np, cls.b, cls.i = jax, j, np, belief, inputs
        rows = tuple(Gen3RandbatVariant(str(k), "fixture", "Pikachu", "Fast", 80,
            ("Thunderbolt", "Surf" if k else "Toxic"), "Static", "Light Ball") for k in range(2))
        source = Gen3RandbatSource(RandbatSourceMetadata("gen3randombattle", 3, None, None, None,
            "transport-fixture"), {"pikachu": Gen3RandbatSpeciesUniverse("Pikachu", 80, rows)})
        cls.host = belief.compile_source(source)
        cls.tables = cls.host.device

    def initial(self, batch=1):
        return (self.i.initialize(self.j.arange(1, batch + 1, dtype=self.j.int32), source_binding=self.tables.binding),
                self.b.initialize(self.j.ones((batch, 2, 6), self.j.int32), self.tables))

    def chunks(self, events):
        return list(self.i.pack_chunks(self.np.arange(1, len(events) + 1, dtype=self.np.int32),
                                      events, source_binding=self.tables.binding))

    def boundary(self, kind, sequence, turn=0, phase=None):
        if phase is None:
            phase = self.i.Phase.RESIDUAL if kind == self.i.Kind.UPKEEP else self.i.Phase.ACTION
        return self.i.PublicEvent(kind, sequence, turn, phase)

    def fact(self, sequence, *, side=1, slot=3, value="Surf", turn=0, exclude=False, phase=None):
        return self.i.PublicEvent(self.i.Kind.SET_EXCLUDE if exclude else self.i.Kind.SET_REVEAL,
            sequence, turn, self.i.Phase.ACTION if phase is None else phase,
            target_side=side, target_slot=slot,
            payload=(self.host.predicate_id("move", value),) + (0,) * 7)

    def same_tree(self, a, b):
        for x, y in zip(self.jax.tree.leaves(a), self.jax.tree.leaves(b)):
            self.assertTrue(bool(self.j.array_equal(x, y)))

    def test_exact_protocol_boundaries_preserve_blank_pipe_and_reject_unhandled_semantics(self):
        blank = self.i.protocol_boundary("|", sequence=1, turn=0)
        upkeep = self.i.protocol_boundary("|upkeep", sequence=2, turn=0)
        turn = self.i.protocol_boundary("|turn|1", sequence=3, turn=0)
        self.assertEqual(blank.kind, self.i.Kind.ACTION_CHUNK_END)
        self.assertEqual(upkeep.kind, self.i.Kind.UPKEEP)
        self.assertEqual(turn.turn, 1)
        for line in ["", "|upkeep|annotation", "|move|p1a: Pikachu|Surf|p2a: Foe", "|request|{}", "|turn|1|extra"]:
            with self.subTest(line=line), self.assertRaises(ValueError):
                self.i.protocol_boundary(line, sequence=1, turn=0)

    def test_more_than_64_events_drain_without_truncation_or_replay(self):
        events = [self.boundary(self.i.Kind.ACTION_CHUNK_END, n) for n in range(1, 131)]
        chunks = self.chunks([events, events[:3]])
        self.assertEqual([int(c.valid[0].sum()) for c in chunks], [64, 64, 2])
        self.assertEqual([bool(c.more[0]) for c in chunks], [True, True, False])
        cursor, state = self.initial(2)
        fold = self.jax.jit(self.i.fold_chunk)
        for index, chunk in enumerate(chunks):
            result = fold(cursor, state, self.tables, chunk)
            self.assertEqual(result.error.tolist(), [0, 0])
            cursor, state = result.cursor, result.beliefs
            self.assertEqual(bool(cursor.needs_more[0]), index < 2)
        self.assertEqual(cursor.sequence.tolist(), [130, 3])
        self.assertEqual(cursor.action_chunks.tolist(), [130, 3])
        replay = fold(cursor, state, self.tables, chunks[-1])
        self.assertEqual(int(replay.error[0]), self.i.Error.SEQUENCE)
        self.same_tree(replay.cursor, cursor)
        self.same_tree(replay.beliefs, state)

    def test_facts_only_narrow_the_other_observers_resolved_encounter_slot(self):
        cursor, state = self.initial(2)
        chunk = self.chunks([[self.fact(1)], [self.fact(1, side=0, slot=5, exclude=True)]])[0]
        result = self.i.fold_chunk(cursor, state, self.tables, chunk)
        self.assertEqual(result.error.tolist(), [0, 0])
        _, summary = self.b.materialize(result.beliefs, self.tables)
        self.assertEqual(summary.count[0, 0, 3].item(), 1)
        self.assertEqual(summary.count[0, 1, 3].item(), 2)
        self.assertEqual(summary.count[1, 1, 5].item(), 1)
        self.assertEqual(summary.count[1, 0, 5].item(), 2)
        self.assertEqual(int(state.has_evidence.sum()), 0)
        self.assertEqual(result.beliefs.has_evidence.sum().item(), 2)

    def test_bad_suffix_rejects_entire_row_while_other_battle_commits(self):
        cursor, state = self.initial(2)
        chunk = self.chunks([[self.fact(1), self.fact(2, value="Toxic")], [self.fact(1)]])[0]
        for column, value, expected in [(0, 999, self.i.Error.UNSUPPORTED_KIND),
                                        (1, 3, self.i.Error.SEQUENCE),
                                        (6, 2, self.i.Error.HEADER),
                                        (7, 6, self.i.Error.HEADER),
                                        (8, 99999, self.i.Error.PAYLOAD),
                                        (15, 1, self.i.Error.PAYLOAD)]:
            with self.subTest(column=column):
                bad = chunk._replace(words=chunk.words.at[0, 1, column].set(value))
                result = self.i.fold_chunk(cursor, state, self.tables, bad)
                self.assertEqual(result.error.tolist(), [expected, 0])
                self.assertEqual(result.cursor.sequence.tolist(), [0, 1])
                self.assertEqual(result.beliefs.hard[0].tolist(), state.hard[0].tolist())
                self.assertEqual(int(result.beliefs.has_evidence[1].sum()), 1)

    def test_turn_residual_order_matches_window_boundaries(self):
        kinds = self.i.Kind
        events = [self.boundary(kinds.ACTION_CHUNK_END, 1), self.boundary(kinds.UPKEEP, 2),
                  self.fact(3, phase=self.i.Phase.RESIDUAL), self.boundary(kinds.TURN, 4, turn=1),
                  self.fact(5, turn=1)]
        cursor, state = self.initial()
        chunk = self.chunks([events])[0]
        result = self.jax.jit(self.i.fold_chunk)(cursor, state, self.tables, chunk)
        self.assertEqual(result.error.tolist(), [0])
        self.assertEqual(result.cursor.turn.tolist(), [1])
        self.assertEqual(result.cursor.action_chunks.tolist(), [2])
        self.assertEqual(result.cursor.upkeeps.tolist(), [1])
        # A fact cannot leave residual phase before TURN; skip-turn also fails.
        for changed in [chunk.words.at[0, 2, 3].set(0), chunk.words.at[0, 3, 2].set(2)]:
            bad = self.i.fold_chunk(cursor, state, self.tables, chunk._replace(words=changed))
            self.assertEqual(bad.error.tolist(), [self.i.Error.HEADER])
            self.same_tree(bad.cursor, cursor)

    def test_bad_stream_source_and_nonprefix_validity_are_not_accepted(self):
        cursor, state = self.initial()
        chunk = self.chunks([[self.fact(1), self.fact(2)]])[0]
        bad_chunks = [(chunk._replace(stream_id=self.j.array([2], self.j.int32)), self.i.Error.STREAM),
            (chunk._replace(source_binding=chunk.source_binding.at[0].add(1)), self.i.Error.SOURCE),
            (chunk._replace(valid=chunk.valid.at[0, 0].set(False)), self.i.Error.MASK)]
        for bad, expected in bad_chunks:
            result = self.i.fold_chunk(cursor, state, self.tables, bad)
            self.assertEqual(result.error.tolist(), [expected])
            self.same_tree(result.cursor, cursor)
            self.same_tree(result.beliefs, state)
        changed = cursor._replace(source_binding=cursor.source_binding.at[0, 0].add(1))
        result = self.i.fold_chunk(changed, state, self.tables, chunk)
        self.assertEqual(result.error.tolist(), [self.i.Error.SOURCE])
        self.same_tree(result.cursor, changed)
        waiting = cursor._replace(needs_more=self.j.array([True]))
        empty = self.chunks([[]])[0]
        result = self.i.fold_chunk(waiting, state, self.tables, empty)
        self.assertEqual(result.error.tolist(), [self.i.Error.MASK])

    def test_host_never_wraps_truncates_or_coerces_private_or_malformed_inputs(self):
        from dataclasses import replace
        event = self.fact(1)
        for bad in [replace(event, payload=(1,) * 9), replace(event, sequence=2**31),
                    replace(event, target_slot=6), replace(event, actor_slot=0),
                    replace(event, payload=(1.2,) + (0,) * 7), replace(event, kind=123)]:
            with self.subTest(bad=bad), self.assertRaises(ValueError): bad.words()
        with self.assertRaises(ValueError):
            list(self.i.pack_chunks(self.np.array([1, 1], self.np.int32), [[], []], source_binding=self.tables.binding))
        with self.assertRaises(ValueError):
            list(self.i.pack_chunks(self.np.array([1], self.np.int32), [[]], source_binding=self.np.zeros(7, self.np.uint32)))


if __name__ == "__main__":
    unittest.main()
