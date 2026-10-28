"""Public device ledger state versus actual protocol-parser consumers."""
import importlib.util
import unittest


@unittest.skipUnless(importlib.util.find_spec("jax"), "requires isolated JAX")
class PublicMonStateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import jax
        import jax.numpy as j
        from gpu_engine import public_mon_state as p
        from pokezero import showdown as s
        cls.jax, cls.j, cls.p, cls.s = jax, j, p, s

    def arrive(self, state, **kwargs):
        values = dict(side=0, slot=0, encounter=1, species=10, source_species=1,
                      level=80, gender=1, hp_numerator=100, hp_denominator=100, status=self.p.NONE)
        values.update(kwargs)
        return self.p.arrive(state, **values)

    def test_baton_pass_carries_and_drag_resets_like_actual_parser(self):
        p = self.p; state, admitted = self.arrive(p.initialize(1))
        self.assertTrue(bool(admitted[0]))
        state, _ = p.boost_update(state, side=0, stat=0, amount=2)
        lines = ["|switch|p1a: Pikachu|Pikachu, L80, M|100/100", "|-boost|p1a: Pikachu|atk|2",
                 "|move|p1a: Pikachu|Baton Pass|p1a: Pikachu",
                 "|switch|p1a: Mr. Mime|Mr. Mime, L80, F|50/100|[from] Baton Pass"]
        state, admitted = self.arrive(state, slot=1, encounter=2, species=11, source_species=2,
                                     gender=2, hp_numerator=50, kind=p.BATON_PASS)
        self.assertTrue(bool(admitted[0]))
        ref = self.s.parse_showdown_replay(lines, battle_id="fixture")
        self.assertEqual(state.active.tolist(), [[1, -1]])
        self.assertEqual(state.boosts[0, 0].tolist(), [ref.boosts["p1"].get(k, 0) for k in p.STATS])
        self.assertEqual(int(state.hp_numerator[0, 0, 1]), 50)
        self.assertTrue(bool(state.arrived_by_baton[0, 0]))
        lines.append("|drag|p1a: Pikachu|Pikachu, L80, M|90/100")
        state, _ = self.arrive(state, hp_numerator=90, kind=p.DRAG)
        ref = self.s.parse_showdown_replay(lines, battle_id="fixture")
        self.assertEqual(state.boosts[0, 0].tolist(), [ref.boosts["p1"].get(k, 0) for k in p.STATS])
        self.assertFalse(bool(state.arrived_by_baton[0, 0]))

    def test_public_hp_status_bench_cure_and_faint_keep_the_right_encounter(self):
        p = self.p; state, _ = self.arrive(p.initialize(1), status=p.PARALYSIS)
        state, _ = self.arrive(state, slot=1, encounter=2, species=11, source_species=2, status=p.TOXIC)
        state, _ = p.status_update(state, side=0, slot=0, encounter=1, status=p.NONE)
        state, _ = p.hp_update(state, side=0, slot=1, encounter=2, numerator=85, denominator=100)
        state, _ = p.faint(state, side=0, slot=1, encounter=2)
        lines = ["|switch|p1a: Pikachu|Pikachu, L80, M|100/100 par",
                 "|switch|p1a: Mr. Mime|Mr. Mime, L80, F|100/100 tox",
                 "|-curestatus|p1: Pikachu|par|[silent]", "|-damage|p1a: Mr. Mime|85/100 tox",
                 "|faint|p1a: Mr. Mime"]
        ref = self.s.parse_showdown_replay(lines, battle_id="fixture")
        self.assertEqual(ref.public_active["p1"].condition, "0 fnt")
        self.assertEqual(int(state.hp_numerator[0, 0, 1]), 0)
        self.assertEqual(int(state.hp_denominator[0, 0, 1]), 0)
        self.assertEqual(int(state.status[0, 0, 1]), p.NONE)
        self.assertTrue(bool(state.fainted[0, 0, 1]))
        self.assertEqual(int(state.status[0, 0, 0]), p.NONE)
        self.assertEqual(int(state.hp_numerator[0, 0, 0]), 100)
        self.assertEqual(state.active.tolist(), [[1, -1]])

    def test_duplicate_species_do_not_merge_or_overwrite_encounters(self):
        p = self.p; state, _ = self.arrive(p.initialize(1))
        state, admitted = self.arrive(state, slot=1, encounter=2)
        self.assertTrue(bool(admitted[0]))
        self.assertEqual(state.encounter[0, 0, :2].tolist(), [1, 2])
        for kwargs in [dict(slot=0, encounter=2), dict(slot=1, encounter=3), dict(slot=2, encounter=1)]:
            before = state
            state, admitted = self.arrive(state, **kwargs)
            self.assertFalse(bool(admitted[0]))
            for x, y in zip(state, before): self.assertTrue(bool(self.j.array_equal(x, y)))

    def test_all_boost_operations_and_integer_extremes_match_parser(self):
        p = self.p; state, _ = self.arrive(p.initialize(1))
        ref = {"p1": {}, "p2": {}}
        steps = [("-boost", "atk", 2, p.DELTA, 2), ("-unboost", "def", 3, p.DELTA, -3),
                 ("-boost", "atk", 2**31 - 1, p.DELTA, 2**31 - 1),
                 ("-setboost", "spe", -(2**31), p.SET, -(2**31)),
                 ("-clearpositiveboost", "atk", 0, p.CLEAR_POSITIVE, 0),
                 ("-clearnegativeboost", "atk", 0, p.CLEAR_NEGATIVE, 0),
                 ("-boost", "evasion", 2, p.DELTA, 2), ("-clearboost", "atk", 0, p.CLEAR, 0)]
        for event, stat, raw_amount, operation, delta in steps:
            self.s._update_boosts(["", event, "p1a: Pikachu", stat, str(raw_amount)], ref)
            state, admitted = p.boost_update(state, side=0, stat=p.STATS.index(stat), amount=delta, operation=operation)
            self.assertTrue(bool(admitted[0]))
            self.assertEqual(state.boosts[0, 0].tolist(), [ref["p1"].get(k, 0) for k in p.STATS])
        state, _ = self.arrive(state, side=1, slot=0, encounter=1)
        state, _ = p.boost_update(state, side=1, stat=1, amount=3)
        state, admitted = p.copy_boosts(state, source_side=0, target_side=1)
        self.assertTrue(bool(admitted[0]))
        self.assertEqual(state.boosts[0, 0].tolist(), state.boosts[0, 1].tolist())

    def test_jit_row_isolation_malformed_identity_and_no_input_mutation(self):
        p = self.p; initial = p.initialize(2)
        before = self.jax.tree.map(lambda x: x.copy(), initial)
        fn = self.jax.jit(lambda state: p.arrive(state, side=self.j.array([0, 1], self.j.int32),
            slot=0, encounter=1, species=10, source_species=1, level=80, gender=1,
            hp_numerator=100, hp_denominator=100, status=p.NONE))
        state, admitted = fn(initial)
        self.assertEqual(admitted.tolist(), [True, True])
        self.assertEqual(state.active.tolist(), [[0, -1], [-1, 0]])
        for x, y in zip(initial, before): self.assertTrue(bool(self.j.array_equal(x, y)))
        for method, kwargs in [(p.hp_update, dict(numerator=99, denominator=100)),
                                (p.status_update, dict(status=p.POISON)), (p.faint, {})]:
            unchanged, admitted = method(state, side=0, slot=0, encounter=99, **kwargs)
            self.assertFalse(bool(admitted.any()))
            for x, y in zip(state, unchanged): self.assertTrue(bool(self.j.array_equal(x, y)))
        for bad in [True, .1, self.j.array(.1)]:
            with self.assertRaises(ValueError): self.arrive(state, side=bad)
        _, admitted = self.arrive(state, side=self.j.array([2, -1], self.j.int32))
        self.assertFalse(bool(admitted.any()))


if __name__ == "__main__":
    unittest.main()
