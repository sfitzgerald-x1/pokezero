"""Device public-set semantics against the shipped Python source and pin fold."""
import importlib.util
import unittest


@unittest.skipUnless(importlib.util.find_spec("jax"), "requires isolated JAX")
class PublicBeliefTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import jax
        import jax.numpy as jnp
        from gpu_engine import public_belief as pb
        from pokezero.randbat import (Gen3RandbatSource, Gen3RandbatSpeciesUniverse,
                                     Gen3RandbatVariant, RandbatSourceMetadata)
        cls.jax, cls.j, cls.pb = jax, jnp, pb
        variants = tuple(Gen3RandbatVariant(str(i), "fixture", "Swalot", "Staller", 80,
            ("Protect", "Toxic", "Hidden Power Ice" if i % 2 else "Hidden Power Fire", f"move{i}"),
            "Sticky Hold" if i % 3 else "Liquid Ooze", "Leftovers" if i % 4 else "Lum Berry")
            for i in range(44))
        # Two source rows can have different IDs but identical pin identity.
        duplicate = Gen3RandbatVariant("duplicate", "fixture", "Swalot", "Staller", 80,
                                      variants[0].moves, variants[0].ability, variants[0].item)
        variants = variants[:-1] + (duplicate,)
        cls.rows = variants
        cls.source = Gen3RandbatSource(RandbatSourceMetadata(
            "gen3randombattle", 3, None, None, None, "fixture-source"),
            {"swalot": Gen3RandbatSpeciesUniverse("Swalot", 80, variants)})
        cls.host = pb.compile_source(cls.source)
        cls.tables = cls.host.device

    def state(self, shape=(1,)):
        return self.pb.initialize(self.j.ones(shape, self.j.int32), self.tables)

    def reference(self, **kwargs):
        return self.source.summarize(format_id="gen3randombattle", species="Swalot",
                                     revealed_moves=(), **kwargs)

    def decode(self, mask):
        words = mask.tolist()
        return tuple(v.variant_id for i, v in enumerate(self.rows) if words[i // 32] & (1 << (i % 32)))

    def test_reveals_exclusions_hidden_power_and_off_source_match_python(self):
        cases = [(("move", "Hidden Power", False),),
                 (("move", "Hidden Power Ice", False),),
                 (("ability", "Liquid Ooze", False), ("item", "Lum Berry", True)),
                 (("move", "Protect", False), ("ability", "Sticky Hold", True)),
                 (("move", "unknown move", False),),
                 (("item", "unknown item", True),),
                 (("move", "Hidden Power Ice", False), ("move", "Hidden Power Fire", False))]
        for facts in cases:
            with self.subTest(facts=facts):
                s = self.state(); arguments = {"revealed_moves": ()}
                for kind, value, exclude in facts:
                    s, admitted = self.pb.add_evidence(s, self.tables, self.host.predicate_id(kind, value),
                                                       exclude=exclude)
                    self.assertTrue(bool(admitted))
                    if kind == "move":
                        arguments["revealed_moves"] += (value,)
                    else:
                        name = "ruled_out_" + kind.replace("y", "ie") + "s" if exclude else "revealed_" + kind
                        arguments[name] = arguments.get(name, ()) + (value,) if exclude else value
                _, got = self.pb.materialize(s, self.tables)
                ref = self.source.summarize(format_id="gen3randombattle", species="Swalot", **arguments)
                self.assertEqual(self.decode(got.effective[0]), tuple(v["variant_id"] for v in ref.candidate_variants))
                self.assertEqual(int(got.count[0]), ref.candidate_count)
                self.assertAlmostEqual(float(got.uncertainty[0]), ref.uncertainty, places=6)

    def test_standing_pin_refusals_and_repeated_summary_conflicts_match_reference(self):
        from pokezero.belief import PublicBattleBeliefEngine
        engine = PublicBattleBeliefEngine(format_id="gen3randombattle", set_source=self.source)
        key = "p2: Swalot"
        s = self.state()
        for indices in [(0, 1, 32, 43), (), (1, 32), (5,), (1, 32)]:
            survivors = [self.rows[i].to_summary() for i in indices]
            changed_ref = engine.narrow_candidate_variants(key, survivors)
            s, changed = self.pb.narrow_pin(s, self.host.pin_mask("Swalot", survivors))
            self.assertEqual(bool(changed[0]), changed_ref)
            self.assertEqual(int(s.conflicts[0]), engine.variant_pin_conflicts.get(key, 0))
        # New public evidence is disjoint from standing pin: keep source summary.
        s, _ = self.pb.add_evidence(s, self.tables, self.host.predicate_id("move", "move5"))
        for _ in range(2):
            # Recreate the exact source summary for this reveal.
            ref = self.source.summarize(format_id="gen3randombattle", species="Swalot", revealed_moves=("move5",))
            fields = dict(candidate_variants=ref.candidate_variants, candidate_set_count=ref.candidate_count,
                          uncertainty=ref.uncertainty, possible_moves=ref.possible_moves,
                          possible_abilities=ref.possible_abilities, possible_items=ref.possible_items)
            engine._apply_variant_pin(key, fields)
            s, got = self.pb.materialize(s, self.tables)
            self.assertEqual(self.decode(got.effective[0]), tuple(v["variant_id"] for v in fields["candidate_variants"]))
            self.assertEqual(int(s.conflicts[0]), engine.variant_pin_conflicts[key])

    def test_inconsistent_source_skips_pin_but_preserves_empty_hard_and_standing_pin(self):
        s = self.state()
        pin = self.host.pin_mask("Swalot", [self.rows[0].to_summary()])
        # Duplicate variant identities must select both rows, including high bit 43.
        self.assertEqual(self.decode(pin), ("0", "duplicate"))
        s, _ = self.pb.narrow_pin(s, pin)
        s, _ = self.pb.add_evidence(s, self.tables, 0)
        s, got = self.pb.materialize(s, self.tables)
        self.assertEqual(int(got.reason[0]), self.pb.INCONSISTENT_REVEALS)
        self.assertEqual(int(got.count[0]), 44)
        self.assertEqual(s.hard.tolist(), [[0, 0]])
        self.assertEqual(s.pin[0].tolist(), pin.tolist())
        self.assertEqual(int(s.conflicts[0]), 0)

    def test_source_order_uniform_weights_and_species_unknown_are_distinct(self):
        s = self.pb.initialize(self.j.array([1, 0, -1, 999]), self.tables)
        s, got = self.pb.materialize(s, self.tables)
        weights = self.pb.variant_weights(got, self.tables)
        self.assertEqual(got.reason.tolist(), [0, 1, 1, 1])
        self.assertEqual(got.count.tolist(), [44, 0, 0, 0])
        self.assertAlmostEqual(float(weights[0].sum()), 1, places=6)
        self.assertTrue(bool(self.j.all(weights[1:] == 0)))
        self.assertTrue(bool(self.j.all(self.j.isfinite(weights))))
        no_source = self.pb.initialize(self.j.array([1, 0]), self.tables, source_available=False)
        _, absent = self.pb.materialize(no_source, self.tables)
        self.assertEqual(absent.reason.tolist(), [self.pb.NO_SOURCE] * 2)
        self.assertEqual(absent.count.tolist(), [0, 0])
        s, _ = self.pb.add_evidence(s, self.tables, self.host.predicate_id("ability", "Sticky Hold"))
        _, narrowed = self.pb.materialize(s, self.tables)
        selected = self.pb.variant_weights(narrowed, self.tables)[0] > 0
        self.assertEqual(tuple(v.variant_id for v, keep in zip(self.rows, selected.tolist()) if keep),
                         self.decode(narrowed.effective[0]))

    def test_jit_batched_updates_do_not_mutate_root_or_other_observers(self):
        s = self.state((3, 2, 6)); before = self.jax.tree.map(lambda x: x.copy(), s)
        predicates = self.j.zeros((3, 2, 6), self.j.int32).at[1, 0, 4].set(
            self.host.predicate_id("move", "move32"))
        valid = self.j.zeros((3, 2, 6), bool).at[1, 0, 4].set(True)
        def update(root, table, ids, valid):
            branch, admitted = self.pb.add_evidence(root, table, ids, valid=valid)
            branch, summary = self.pb.materialize(branch, table)
            return branch, summary, self.pb.variant_weights(summary, table), admitted
        eager = update(s, self.tables, predicates, valid)
        compiled = self.jax.jit(update)(s, self.tables, predicates, valid)
        for x, y in zip(self.jax.tree.leaves(eager), self.jax.tree.leaves(compiled)):
            self.assertTrue(bool(self.j.array_equal(x, y)))
        self.assertEqual(int(compiled[1].count[1, 0, 4]), 1)
        self.assertEqual(int(compiled[1].count[1, 1, 4]), 44)
        for x, y in zip(s, before):
            self.assertTrue(bool(self.j.array_equal(x, y)))

    def test_malformed_ids_declined_empty_fact_rejected_and_shape_guard(self):
        from dataclasses import replace
        s = self.state()
        for predicate in [-1, 1000000]:
            got, admitted = self.pb.add_evidence(s, self.tables, predicate)
            self.assertFalse(bool(admitted))
            self.assertEqual(got.hard.tolist(), s.hard.tolist())
        with self.assertRaises(ValueError): self.host.predicate_id("item", "")
        with self.assertRaises(ValueError): self.host.predicate_id("engine_move_id", "1")
        outside = dict(self.rows[0].to_summary(), ability="foreign source ability")
        for survivors in ([outside], [self.rows[0].to_summary(), outside]):
            with self.assertRaisesRegex(ValueError, "outside the bound public source"):
                self.host.pin_mask("Swalot", survivors)
        self.assertEqual(self.host.pin_mask("Swalot", []).tolist(), [0, 0])
        huge = replace(self.source, universes={"swalot": replace(
            self.source.universes["swalot"], variants=self.rows * 2)})
        with self.assertRaisesRegex(ValueError, "64"): self.pb.compile_source(huge)


if __name__ == "__main__":
    unittest.main()
