"""Ordered public move/ability composition, correction and atomic rollback."""
import importlib.util
import unittest


@unittest.skipUnless(importlib.util.find_spec("jax"), "requires isolated JAX")
class PublicContextTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import test_gpu_engine_public_moves as fixtures
        fixtures.PublicMovesTests.setUpClass.__func__(cls)
        from gpu_engine import public_context as c, public_abilities as a, public_inputs as i
        cls.c, cls.a, cls.i, cls.fixtures = c, a, i, fixtures
        cls.abilities = a.compile_abilities(cls.host, cls.vocab,
            ability_names=["Pressure", "Trace", "Immunity", "Levitate"])
        cls.context_tables = c.compile_context(cls.host, cls.abilities, cls.catalog)

    def initial(self, batch=1, opponent="Absol"):
        moves, beliefs, mons = self.fixtures.PublicMovesTests.initial(self, opponent=opponent, batch=batch)
        context = self.c.initialize(beliefs, self.context_tables)._replace(moves=moves)
        cursor = self.i.initialize(self.j.arange(1, batch+1, dtype=self.j.int32), source_binding=self.host.device.binding)
        return cursor, beliefs, mons, context

    def event(self, kind, sequence, payload, *, actor=-1, target=1, slot=0):
        return self.i.PublicEvent(kind, sequence, 0, self.i.Phase.ACTION,
            actor_side=actor, actor_slot=0 if actor >= 0 else -1,
            target_side=target, target_slot=slot if target >= 0 else -1,
            payload=tuple(payload)+(0,)*(8-len(payload)))

    def chunks(self, events):
        import numpy as np
        return list(self.i.pack_chunks(np.arange(1, len(events)+1, dtype=np.int32), events,
                                     source_binding=self.host.device.binding))

    def fold(self, initial, chunk):
        cursor, beliefs, mons, context = initial
        return self.i.fold_chunk(cursor, beliefs, self.host.device, chunk, mon_state=mons,
                                 context=context, context_tables=self.context_tables)

    def same(self, left, right):
        for x, y in zip(self.jax.tree.leaves(left), self.jax.tree.leaves(right)):
            self.assertTrue(bool(self.j.array_equal(x, y)))

    def test_trace_current_pressure_drives_actual_move_consumer_in_stream_order(self):
        k, aid = self.i.Kind, self.abilities.ability_id
        events = [self.event(k.ABILITY, 1, [1, aid("Pressure"), 1], target=0),
            self.event(k.ABILITY, 2, [1, aid("Trace"), 1]),
            self.event(k.TRACE, 3, [1, aid("Pressure"), 1, 1], actor=1, target=0),
            self.event(k.MOVE, 4, [1, self.catalog.move_id("Body Slam"), 0, 0], actor=0),
            self.event(k.MOVE, 5, [1, self.catalog.move_id("Protect"), 0, 0], actor=0)]
        initial = self.initial(opponent="Gardevoir")
        result = self.fold(initial, self.chunks([events])[0])
        self.assertEqual(result.error.tolist(), [0])
        ref = self.fixtures.PublicMovesTests.reference(self, ["|-ability|p1a: Snorlax|Pressure",
            "|-ability|p2a: Gardevoir|Trace",
            "|-ability|p2a: Gardevoir|Pressure|Trace|[from] ability: Trace|[of] p1a: Snorlax",
            "|move|p1a: Snorlax|Body Slam|p2a: Gardevoir", "|move|p1a: Snorlax|Protect|p1a: Snorlax"],
            opponent="Gardevoir")
        actor = ref.snapshot().side("p1")[0]
        ids = result.context.moves.move_ids[0, 0, 0].tolist()
        uses = result.context.moves.uses[0, 0, 0].tolist()
        self.assertEqual({self.catalog.names[x]: u for x, u in zip(ids, uses) if x}, dict(actor.move_uses))
        self.assertEqual(uses[:2], [2, 1])
        self.assertEqual(int(result.context.abilities.original[0, 1, 0]), aid("Trace"))

    def test_authoritative_ability_correction_recovers_variants_preserves_moves_and_pin(self):
        k, aid = self.i.Kind, self.abilities.ability_id
        events = [self.event(k.ABILITY, 1, [1, aid("Pressure"), 1], target=0),
            self.event(k.MOVE, 2, [1, self.catalog.move_id("Surf"), 0, 0], actor=0),
            self.event(k.ABILITY, 3, [1, aid("Immunity"), 1], target=0)]
        initial = self.initial()
        first = self.fold(initial, self.chunks([events[:2]])[0])
        self.assertEqual(first.error.tolist(), [0])
        self.assertFalse(bool(first.beliefs.hard[0, 1, 0].any()))
        pin = self.host.pin_mask("Snorlax", [self.host.variants[self.host.species_id("Snorlax")][1].to_summary()])
        full_pin = self.j.broadcast_to(pin, first.beliefs.hard.shape)
        beliefs, _ = self.b.narrow_pin(first.beliefs, full_pin,
            valid=self.j.zeros(first.beliefs.species.shape, bool).at[0, 1, 0].set(True))
        second = self.fold((first.cursor, beliefs, first.mon_state, first.context), self.chunks([events[2:]])[0])
        self.assertEqual(second.error.tolist(), [0])
        self.assertEqual(second.beliefs.hard[0, 1, 0].tolist(), pin.tolist())
        self.assertTrue(bool(second.beliefs.pin_active[0, 1, 0]))
        self.assertEqual(second.beliefs.pin[0, 1, 0].tolist(), pin.tolist())
        ref = self.fixtures.PublicMovesTests.reference(self, ["|-ability|p1a: Snorlax|Pressure",
            "|move|p1a: Snorlax|Surf|p2a: Absol", "|-ability|p1a: Snorlax|Immunity"])
        self.assertEqual(int(self.b.materialize(second.beliefs, self.host.device)[1].count[0, 1, 0]),
                         ref.snapshot().side("p1")[0].candidate_set_count)

    def test_transform_switch_and_bad_suffix_rollback_all_consumers_per_battle(self):
        k, aid = self.i.Kind, self.abilities.ability_id
        events = [self.event(k.ABILITY, 1, [1, aid("Pressure"), 1]),
            self.event(k.TRANSFORM, 2, [1, 1], actor=0),
            self.event(k.MOVE, 3, [1, self.catalog.move_id("Surf"), 0, 0], actor=0)]
        initial = self.initial(batch=2)
        chunk = self.chunks([events, events])[0]
        broken = chunk._replace(words=chunk.words.at[0, 2, 9].set(2**31-1))
        result = self.jax.jit(lambda cur, b, m, ctx, ch: self.fold((cur,b,m,ctx),ch))(*initial, broken)
        self.assertNotEqual(int(result.error[0]), 0); self.assertEqual(int(result.error[1]), 0)
        for before, after in zip(self.jax.tree.leaves(initial), self.jax.tree.leaves(
                (result.cursor, result.beliefs, result.mon_state, result.context))):
            self.assertTrue(bool(self.j.array_equal(before[0], after[0])))
        self.assertTrue(bool(result.context.abilities.transformed[1, 0, 0]))
        self.assertFalse(bool(result.context.moves.move_ids[1, 0, 0].any()))
        # A subsequent arrival resets Transform and last-move state, while its
        # source predicate/identity and borrowed ability lifetimes stay explicit.
        event = self.event(k.SWITCH, 4, [1, 10, self.host.species_id("Snorlax"), 80, 0, 100, 100, 0], target=0)
        next_chunk = self.chunks([[], [event]])[0]
        next_result = self.fold((result.cursor, result.beliefs, result.mon_state, result.context), next_chunk)
        self.assertEqual(int(next_result.error[1]), 0)
        self.assertFalse(bool(next_result.context.abilities.transformed[1, 0, 0]))
        self.assertEqual(int(next_result.context.moves.last_kind[1, 0]), self.p.LAST_SWITCH)

    def test_ambiguous_ability_facts_and_wrong_context_binding_refuse(self):
        k = self.i.Kind
        event = self.event(k.SET_REVEAL, 1, [self.host.predicate_id("ability", "Pressure")])
        initial = self.initial()
        result = self.fold(initial, self.chunks([[event]])[0])
        self.assertEqual(result.error.tolist(), [self.i.Error.PAYLOAD])
        self.same(initial, (result.cursor, result.beliefs, result.mon_state, result.context))
        cursor, beliefs, mons, ctx = initial
        ctx = ctx._replace(moves=ctx.moves._replace(binding=ctx.moves.binding.at[0, 0].add(1)))
        empty = self.chunks([[]])[0]
        self.assertEqual(self.fold((cursor, beliefs, mons, ctx), empty).error.tolist(), [self.i.Error.SOURCE])
        with self.assertRaises(ValueError): self.c.initialize(beliefs._replace(has_evidence=self.j.ones_like(beliefs.has_evidence)), self.context_tables)
        with self.assertRaises(ValueError): self.event(k.TRACE, 1, [1, 1, 0, 2], actor=0, target=-1).words()


if __name__ == "__main__": unittest.main()
