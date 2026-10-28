"""Public ability context against the shipped belief engine and replay parser."""
import importlib.util
import unittest


@unittest.skipUnless(importlib.util.find_spec("jax"), "requires isolated JAX")
class PublicAbilityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import jax
        import jax.numpy as j
        from gpu_engine import public_abilities as a, public_mon_state as m, public_belief as b
        from pokezero.category_vocab import CategoryVocabulary
        from pokezero.randbat import (Gen3RandbatSource, Gen3RandbatSpeciesUniverse,
                                     Gen3RandbatVariant, RandbatSourceMetadata)
        cls.jax, cls.j, cls.a, cls.m = jax, j, a, m
        rows = {}
        for i, (species, ability) in enumerate([("Ditto", "Limber"), ("Absol", "Pressure"),
                                                ("Gardevoir", "Trace"), ("Porygon2", "Trace")]):
            v = Gen3RandbatVariant(str(i), "fixture", species, "fixture", 80,
                ("Rest", "Protect", "Body Slam", "Transform"), ability, "Leftovers")
            rows[species.lower()] = Gen3RandbatSpeciesUniverse(species, 80, (v,))
        cls.source = Gen3RandbatSource(RandbatSourceMetadata("gen3randombattle", 3, None, None, None, "ability-fixture"), rows)
        cls.source_tables = b.compile_source(cls.source)
        cls.vocab = CategoryVocabulary(("ability:pressure", "ability:trace", "ability:limber"))
        cls.catalog = a.compile_abilities(cls.source_tables, cls.vocab, ability_names=["Levitate", "Insomnia"])
        cls.tables = cls.catalog.device

    def initial(self, left="Ditto", right="Absol", batch=1):
        m = self.m; mons = m.initialize(batch)
        for side, species in enumerate((left, right)):
            mons, ok = m.arrive(mons, side=side, slot=0, encounter=1, species=10+side,
                source_species=self.source_tables.species_id(species), level=80, gender=0,
                hp_numerator=100, hp_denominator=100, status=m.NONE)
            self.assertTrue(bool(ok.all()))
        return self.a.initialize(batch, self.tables), mons

    def reveal(self, state, mons, name, side=1, authoritative=True):
        return self.a.reveal(state, mons, self.tables, side=side, slot=0, encounter=1,
            ability_id=self.catalog.ability_id(name), authoritative=authoritative)

    def reference(self, lines, left="Ditto", right="Absol"):
        from pokezero.belief import PublicBattleBeliefEngine
        from pokezero.showdown import _public_event_from_line
        prefix = [f"|switch|p1a: {left}|{left}, L80|100/100", f"|switch|p2a: {right}|{right}, L80|100/100"]
        return PublicBattleBeliefEngine.from_events([_public_event_from_line(x) for x in prefix+lines],
            format_id="gen3randombattle", set_source=self.source)

    def compare(self, state, engine):
        from pokezero.dex import normalize_id
        for side in (0, 1):
            belief = next(x for x in engine.snapshot().side(f"p{side+1}") if x.active)
            original = self.catalog.names[int(state.original[0, side, 0])]
            running = self.catalog.names[int(state.running[0, side, 0])]
            self.assertEqual(original, normalize_id(belief.revealed_ability or ""))
            self.assertEqual(running, normalize_id(engine._running_ability.get(belief.key, "")))
            self.assertEqual(bool(state.transformed[0, side, 0]), belief.transformed)

    def same(self, left, right):
        for x, y in zip(left, right): self.assertTrue(bool(self.j.array_equal(x, y)))

    def test_authoritative_replacement_and_proc_conflict_match_original_ledger(self):
        s, mons = self.initial()
        s = self.reveal(s, mons, "Pressure").state
        update = self.reveal(s, mons, "Levitate", authoritative=False)
        self.assertTrue(bool(update.conflict[0])); self.same(s, update.state)
        s = self.reveal(s, mons, "Levitate").state
        lines = ["|-ability|p2a: Absol|Pressure", "|-activate|p2a: Absol|ability: Levitate",
                 "|-ability|p2a: Absol|Levitate"]
        self.compare(s, self.reference(lines))
        self.assertFalse(bool(self.a.move_context(s, mons, self.tables, side=0, slot=0, encounter=1)[1][0]))

    def test_trace_pressure_redirects_original_and_borrows_for_pp(self):
        s, mons = self.initial(left="Absol", right="Gardevoir")
        s = self.reveal(s, mons, "Trace").state
        s, ok = self.a.trace_copy(s, mons, self.tables, side=1, slot=0, encounter=1,
            donor_side=0, donor_slot=0, donor_encounter=1, ability_id=self.catalog.ability_id("Pressure"))
        self.assertTrue(bool(ok[0]))
        line = "|-ability|p2a: Gardevoir|Pressure|Trace|[from] ability: Trace|[of] p1a: Absol"
        self.compare(s, self.reference(["|-ability|p2a: Gardevoir|Trace", line], left="Absol", right="Gardevoir"))
        transformed, pressure, admitted = self.a.move_context(s, mons, self.tables, side=0, slot=0, encounter=1)
        self.assertTrue(bool(pressure[0])); self.assertTrue(bool(admitted[0])); self.assertFalse(bool(transformed[0]))
        # A proc of the borrowed ability must not replace the tracer's original.
        later = self.reveal(s, mons, "Pressure", authoritative=False)
        self.same(s, later.state); self.assertFalse(bool(later.conflict[0]))

    def test_missing_trace_of_changes_only_the_separate_parser_feature(self):
        from pokezero.showdown import parse_showdown_replay
        s, mons = self.initial(left="Gardevoir")
        line = "|-ability|p1a: Gardevoir|Pressure|Trace|[from] ability: Trace"
        s, ok = self.a.trace_copy(s, mons, self.tables, side=0, slot=0, encounter=1,
                                 ability_id=self.catalog.ability_id("Pressure"))
        self.assertTrue(bool(ok[0]))
        self.compare(s, self.reference([line], left="Gardevoir"))
        parser = parse_showdown_replay(["|switch|p1a: Gardevoir|Gardevoir, L80|100/100", line], battle_id="fixture")
        labels, ok = self.a.traced_categories(s, self.tables)
        self.assertEqual(int(labels[0, 0]), self.vocab.encode("ability:"+parser.traced_ability["p1"]))
        self.assertEqual(s.running.tolist(), self.a.initialize(1, self.tables).running.tolist())

    def test_self_trace_installs_running_override_before_donor_guard(self):
        s, mons = self.initial(left="Gardevoir")
        line = "|-ability|p1a: Gardevoir|Pressure|Trace|[from] ability: Trace|[of] p1a: Gardevoir"
        s, ok = self.a.trace_copy(s, mons, self.tables, side=0, slot=0, encounter=1,
            donor_side=0, donor_slot=0, donor_encounter=1, ability_id=self.catalog.ability_id("Pressure"))
        self.assertTrue(bool(ok[0]))
        self.compare(s, self.reference([line], left="Gardevoir"))
        self.assertEqual(int(s.original[0, 0, 0]), 0)

    def test_transform_copies_current_borrowed_pressure_without_original_poison(self):
        s, mons = self.initial(right="Gardevoir")
        s = self.reveal(s, mons, "Limber", side=0).state
        s = self.reveal(s, mons, "Trace", side=1).state
        # Off-script but source-defined chain: a public copy can borrow an
        # ability different from its donor's original; direct reveals overwrite.
        s = self.reveal(s, mons, "Pressure", side=0).state
        s, _ = self.a.trace_copy(s, mons, self.tables, side=1, slot=0, encounter=1,
            donor_side=0, donor_slot=0, donor_encounter=1, ability_id=self.catalog.ability_id("Pressure"))
        s = self.reveal(s, mons, "Limber", side=0).state
        s, ok = self.a.transform(s, mons, self.tables, side=0, slot=0, encounter=1,
            donor_side=1, donor_slot=0, donor_encounter=1)
        self.assertTrue(bool(ok[0])); self.assertEqual(int(s.transform_species[0, 0, 0]), 11)
        lines = ["|-ability|p1a: Ditto|Limber", "|-ability|p2a: Gardevoir|Trace",
            "|-ability|p1a: Ditto|Pressure",
            "|-ability|p2a: Gardevoir|Pressure|Trace|[from] ability: Trace|[of] p1a: Ditto",
            "|-ability|p1a: Ditto|Limber", "|-transform|p1a: Ditto|p2a: Gardevoir"]
        self.compare(s, self.reference(lines, right="Gardevoir"))
        # Tracing a transformed/borrowed donor must not reveal its original.
        before = s.original
        s, _ = self.a.trace_copy(s, mons, self.tables, side=1, slot=0, encounter=1,
            donor_side=0, donor_slot=0, donor_encounter=1, ability_id=self.catalog.ability_id("Pressure"))
        self.assertTrue(bool(self.j.array_equal(before, s.original)))

    def test_unknown_transform_ability_preserves_override_faint_and_arrival_lifetimes(self):
        s, mons = self.initial()
        s, _ = self.a.trace_copy(s, mons, self.tables, side=0, slot=0, encounter=1,
            donor_side=1, donor_slot=0, donor_encounter=1, ability_id=self.catalog.ability_id("Pressure"))
        # Set up an unrevealed donor on another public encounter.
        mons, _ = self.m.arrive(mons, side=1, slot=1, encounter=2, species=12, source_species=1,
            level=80, gender=0, hp_numerator=100, hp_denominator=100, status=0)
        s, _ = self.a.transform(s, mons, self.tables, side=0, slot=0, encounter=1,
            donor_side=1, donor_slot=1, donor_encounter=2)
        self.assertEqual(int(s.running[0, 0, 0]), self.catalog.ability_id("Pressure"))
        s, _ = self.a.faint(s, mons, self.tables, side=0, slot=0, encounter=1)
        self.assertFalse(bool(s.transformed[0, 0, 0])); self.assertGreater(int(s.traced[0, 0]), 0)
        self.assertGreater(int(s.running[0, 0, 0]), 0)
        s, _ = self.a.arrive(s, mons, self.tables, side=0, slot=0, encounter=1)
        self.assertEqual(int(s.running[0, 0, 0]), 0); self.assertEqual(int(s.traced[0, 0]), 0)

    def test_batched_jit_binding_bad_address_and_fainted_pressure(self):
        s, mons = self.initial(batch=2)
        s = self.reveal(s, mons, "Pressure").state
        args = dict(side=0, slot=0, encounter=1, donor_side=1, donor_slot=0, donor_encounter=1)
        eager, ok = self.a.transform(s, mons, self.tables, **args)
        compiled, compiled_ok = self.jax.jit(lambda x, y: self.a.transform(x, y, self.tables, **args))(s, mons)
        self.same(eager, compiled); self.assertTrue(bool(self.j.array_equal(ok, compiled_ok)))
        broken = s._replace(binding=s.binding.at[0, 0].add(1))
        updated, ok = self.a.transform(broken, mons, self.tables, **args)
        self.assertEqual(ok.tolist(), [False, True])
        for x, y in zip(broken, updated): self.assertTrue(bool(self.j.array_equal(x[0], y[0])))
        updated, ok = self.a.trace_copy(s, mons, self.tables, side=0, slot=0, encounter=1,
            donor_side=1, donor_slot=0, donor_encounter=99, ability_id=self.catalog.ability_id("Pressure"))
        self.assertFalse(bool(ok.any())); self.same(s, updated)
        mons, _ = self.m.faint(mons, side=1, slot=0, encounter=1)
        self.assertFalse(bool(self.a.move_context(s, mons, self.tables, side=0, slot=0, encounter=1)[1].any()))
        with self.assertRaises(ValueError): self.catalog.ability_id("not-in-namespace")
        with self.assertRaises(ValueError): self.reveal(s, mons, "Pressure", side=0.5)


if __name__ == "__main__": unittest.main()
