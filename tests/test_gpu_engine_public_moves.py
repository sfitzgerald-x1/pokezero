"""Move reveal/PP consumers compared with the shipped belief and token writers."""
import importlib.util
import unittest


@unittest.skipUnless(importlib.util.find_spec("jax"), "requires isolated JAX")
class PublicMovesTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import jax
        import jax.numpy as j
        from gpu_engine import public_moves as p, public_belief as b, public_mon_state as m
        from pokezero.category_vocab import CategoryVocabulary
        from pokezero.dex import MoveInfo, ShowdownDex
        from pokezero.randbat import (Gen3RandbatSource, Gen3RandbatSpeciesUniverse,
                                     Gen3RandbatVariant, RandbatSourceMetadata)
        cls.jax, cls.j, cls.p, cls.b, cls.m = jax, j, p, b, m
        def variant(species, moves, ability, number):
            return Gen3RandbatVariant(str(number), "fixture", species, "fixture", 80, moves, ability, "Leftovers")
        a = variant("Snorlax", ("Rest", "Sleep Talk", "Body Slam", "Protect"), "Immunity", 0)
        z = variant("Snorlax", ("Rest", "Sleep Talk", "Surf", "Protect"), "Immunity", 1)
        pressure = variant("Absol", ("Shadow Ball", "Rest", "Protect", "Body Slam"), "Pressure", 2)
        trace = variant("Gardevoir", ("Psychic", "Rest", "Protect", "Body Slam"), "Trace", 3)
        cls.source = Gen3RandbatSource(RandbatSourceMetadata("gen3randombattle", 3, None, None, None, "moves-fixture"), {
            "snorlax": Gen3RandbatSpeciesUniverse("Snorlax", 80, (a, z)),
            "absol": Gen3RandbatSpeciesUniverse("Absol", 80, (pressure,)),
            "gardevoir": Gen3RandbatSpeciesUniverse("Gardevoir", 80, (trace,))})
        names = ["rest", "sleeptalk", "bodyslam", "protect", "surf", "struggle", "metronome",
                 "solarbeam", "psychic", "shadowball", "snore"] + [f"move{i:02d}" for i in range(40)]
        cls.dex = ShowdownDex({name: MoveInfo(name, name, "Normal", "Physical", "Physical", 50, 100,
            0, False, False, False, None, {}, "normal", False, pp=5 if name == "rest" else 10)
            for name in names}, {}, {})
        cls.vocab = CategoryVocabulary(("move:rest", "move:sleeptalk", "move:bodyslam", "move:protect",
            "lastmove:switch", "lastmove:batonpass"), aliases={"move:body-slam": "move:bodyslam"})
        cls.host = b.compile_source(cls.source)
        cls.catalog = p.compile_moves(cls.host, cls.dex, cls.vocab)
        cls.tables = cls.catalog.device

    def initial(self, opponent="Absol", batch=1):
        j, p, b, m = self.j, self.p, self.b, self.m
        mons = m.initialize(batch)
        for side, species in [(0, "Snorlax"), (1, opponent)]:
            mons, admitted = m.arrive(mons, side=side, slot=0, encounter=1, species=10 + side,
                source_species=self.host.species_id(species), level=80, gender=0,
                hp_numerator=100, hp_denominator=100, status=m.NONE)
            self.assertTrue(bool(admitted.all()))
        species = j.zeros((batch, 2, 6), j.int32).at[:, 0, 0].set(self.host.species_id(opponent)).at[:, 1, 0].set(self.host.species_id("Snorlax"))
        beliefs, moves = b.initialize(species, self.host.device), p.initialize(batch, self.tables)
        for side in [0, 1]: moves, _ = p.arrive(moves, self.tables, side=side)
        return moves, beliefs, mons

    def reference(self, lines, opponent="Absol"):
        from pokezero.belief import PublicBattleBeliefEngine
        from pokezero.showdown import _public_event_from_line
        prefix = ["|switch|p1a: Snorlax|Snorlax, L80|100/100",
                  f"|switch|p2a: {opponent}|{opponent}, L80|100/100"]
        return PublicBattleBeliefEngine.from_events([_public_event_from_line(line) for line in prefix + lines],
            format_id="gen3randombattle", set_source=self.source)

    def apply_line(self, state, beliefs, mons, line, *, transformed=False, pressure=False, wire=-1):
        name = line.split("|")[3]
        caller, has_from = self.p.lexical_move_tags(line)
        return self.p.record_move(state, beliefs, mons, self.tables, self.host.device,
            side=0, slot=0, encounter=1, move_id=self.catalog.move_id(name), caller=caller,
            has_from=has_from, transformed=transformed, opposing_pressure=pressure, wire_target_side=wire)

    def same_tree(self, a, b):
        for x, y in zip(self.jax.tree.leaves(a), self.jax.tree.leaves(b)):
            self.assertTrue(bool(self.j.array_equal(x, y)))

    def compare_reference(self, state, beliefs, lines, opponent="Absol"):
        from pokezero.dex import normalize_id
        ref = self.reference(lines, opponent=opponent).snapshot().side("p1")[0]
        ids = state.move_ids[0, 0, 0].tolist(); uses = state.uses[0, 0, 0].tolist()
        self.assertEqual([self.catalog.names[i] for i in ids if i], [normalize_id(n) for n in ref.revealed_moves])
        self.assertEqual({self.catalog.names[i]: u for i, u in zip(ids, uses) if i and u}, dict(ref.move_uses))
        _, summary = self.b.materialize(beliefs, self.host.device)
        self.assertEqual(int(summary.count[0, 1, 0]), ref.candidate_set_count)
        self.assertAlmostEqual(float(summary.uncertainty[0, 1, 0]), ref.uncertainty, places=6)

    def test_direct_called_locked_struggle_and_unknown_from_follow_source(self):
        cases = [["|move|p1a: Snorlax|Body Slam|p2a: Absol"],
            ["|move|p1a: Snorlax|Sleep Talk|p1a: Snorlax", "|move|p1a: Snorlax|Body Slam|p2a: Absol|[from] move: Sleep Talk"],
            ["|move|p1a: Snorlax|Metronome|p1a: Snorlax", "|move|p1a: Snorlax|Surf|p2a: Absol|[from] Metronome"],
            ["|move|p1a: Snorlax|Solar Beam|p2a: Absol", "|move|p1a: Snorlax|Solar Beam|p2a: Absol|[from]lockedmove"],
            ["|move|p1a: Snorlax|Struggle|p2a: Absol"],
            ["|move|p1a: Snorlax|Surf|p2a: Absol|[from] other-source"]]
        for lines in cases:
            with self.subTest(lines=lines):
                state, beliefs, mons = self.initial()
                for line in lines:
                    result = self.apply_line(state, beliefs, mons, line)
                    self.assertTrue(bool(result.admitted[0]))
                    state, beliefs = result.state, result.beliefs
                self.compare_reference(state, beliefs, lines)

    def test_pressure_is_move_based_even_with_blank_or_self_wire_target(self):
        state, beliefs, mons = self.initial()
        prefix = ["|-ability|p2a: Absol|Pressure"]
        for move, wire, charge in [("Body Slam", -1, 2), ("Body Slam", 0, 2), ("Rest", 0, 1)]:
            line = f"|move|p1a: Snorlax|{move}|" + ("p1a: Snorlax" if wire == 0 else "")
            result = self.apply_line(state, beliefs, mons, line, pressure=True, wire=wire)
            self.assertEqual(int(result.charge[0]), charge)
            self.assertEqual(bool(result.doubled_despite_wire[0]), charge == 2)
            state, beliefs = result.state, result.beliefs; prefix.append(line)
        self.compare_reference(state, beliefs, prefix)

    def test_transform_suppresses_set_pp_and_reveal_but_not_public_lastmove(self):
        from pokezero.showdown import parse_showdown_replay
        state, beliefs, mons = self.initial()
        line = "|move|p1a: Snorlax|Body Slam|p2a: Absol"
        result = self.apply_line(state, beliefs, mons, line, transformed=True)
        self.assertFalse(bool(result.revealed[0])); self.assertEqual(int(result.charge[0]), 0)
        self.compare_reference(result.state, result.beliefs, ["|-transform|p1a: Snorlax|p2a: Absol", line])
        labels, admitted = self.p.last_move_categories(result.state, self.tables, False)
        self.assertTrue(bool(admitted[0]))
        self.assertEqual(int(labels[0, 0]), self.vocab.encode("move:bodyslam"))
        self.assertEqual(parse_showdown_replay([line], battle_id="fixture").last_used_move["p1"], "bodyslam")

    def test_borrowed_pressure_is_current_public_ability_not_original_set_ability(self):
        for copied in ["|-ability|p2a: Gardevoir|Pressure|Trace|[from] ability: Trace|[of] p1a: Absol",
                       "|-transform|p2a: Gardevoir|p1a: Absol"]:
            with self.subTest(copied=copied):
                state, beliefs, mons = self.initial(opponent="Gardevoir")
                lines = ["|switch|p1a: Absol|Absol, L80|100/100", "|-ability|p1a: Absol|Pressure",
                         copied, "|switch|p1a: Snorlax|Snorlax, L80|100/100",
                         "|move|p1a: Snorlax|Body Slam|"]
                result = self.apply_line(state, beliefs, mons, lines[-1], pressure=True)
                self.assertEqual(int(result.charge[0]), 2)
                self.compare_reference(result.state, result.beliefs, lines, opponent="Gardevoir")

    def test_switch_keeps_real_set_uses_and_resets_only_current_last_move(self):
        state, beliefs, mons = self.initial()
        line = "|move|p1a: Snorlax|Body Slam|p2a: Absol"
        result = self.apply_line(state, beliefs, mons, line)
        state, beliefs = result.state, result.beliefs
        before_uses = state.uses.copy()
        state, admitted = self.p.arrive(state, self.tables, side=0)
        self.assertTrue(bool(admitted[0]))
        self.assertTrue(bool(self.j.array_equal(state.uses, before_uses)))
        labels, _ = self.p.last_move_categories(state, self.tables, False)
        self.assertEqual(int(labels[0, 0]), self.vocab.encode("lastmove:switch"))
        result = self.apply_line(state, beliefs, mons, line)
        self.compare_reference(result.state, result.beliefs, [line,
            "|switch|p1a: Gardevoir|Gardevoir, L80|100/100",
            "|switch|p1a: Snorlax|Snorlax, L80|100/100", line])

    def test_pp_fraction_and_validity_match_actual_bucket_writer_at_zero(self):
        from pokezero import showdown as s
        state, beliefs, mons = self.initial(); lines = []
        for _ in range(8):
            line = "|move|p1a: Snorlax|Rest|p1a: Snorlax"; lines.append(line)
            result = self.apply_line(state, beliefs, mons, line)
            state, beliefs = result.state, result.beliefs
        exact = self.reference(lines).snapshot().side("p1")[0]
        bucket_names = ["rest", "surf"]
        buckets = self.j.zeros((1, 2, 6, 2), self.j.int32).at[0, 0, 0].set(
            self.j.array([self.catalog.move_id(n) for n in bucket_names], self.j.int32))
        fraction, valid, admitted = self.p.bucket_pp(state, self.tables, buckets)
        row = [0.] * 200
        s._encode_opponent_move_pp_fractions(row, exact, bucket_names, dex=self.dex, write_validity=True)
        self.assertEqual(fraction[0, 0, 0].tolist(), [row[s.NUMERIC_OPP_MOVE_PP_OFFSET+i] for i in range(2)])
        self.assertEqual(valid[0, 0, 0].tolist(), [bool(row[s.NUMERIC_OPP_MOVE_PP_VALID_OFFSET+i]) for i in range(2)])
        self.assertEqual(fraction[0, 0, 0].tolist(), [0, 0]); self.assertEqual(valid[0, 0, 0].tolist(), [True, False])
        self.assertTrue(bool(admitted[0]))

    def test_last_move_tag_distinction_switch_sentinels_alias_and_oov_rows(self):
        from pokezero.showdown import parse_showdown_replay
        state, beliefs, mons = self.initial()
        labels, _ = self.p.last_move_categories(state, self.tables, self.j.array([[False, True]]))
        self.assertEqual(labels.tolist(), [[self.vocab.encode("lastmove:switch"), self.vocab.encode("lastmove:batonpass")]])
        lines = ["|move|p1a: Snorlax|Sleep Talk|p1a: Snorlax",
                 "|move|p1a: Snorlax|Body Slam|p2a: Absol|[from] Sleep Talk",
                 "|move|p1a: Snorlax|Surf|p2a: Absol| [from] Sleep Talk"]
        for line in lines:
            result = self.apply_line(state, beliefs, mons, line); state, beliefs = result.state, result.beliefs
            expected = parse_showdown_replay(lines[:lines.index(line)+1], battle_id="fixture").last_used_move["p1"]
            labels, _ = self.p.last_move_categories(state, self.tables, False)
            self.assertEqual(int(labels[0, 0]), self.vocab.encode("move:" + expected))
        self.assertEqual(self.p._static_category(self.vocab, "move:body-slam"), self.vocab.encode("move:body-slam"))
        self.assertEqual(int(self.tables.last_category[self.catalog.move_id("surf")]), self.vocab.encode("move:surf"))

    def test_jit_batch_prefix_and_other_observer_root_are_immutable(self):
        state, beliefs, mons = self.initial(batch=2)
        before = self.jax.tree.map(lambda x: x.copy(), (state, beliefs, mons))
        def update(st, be, mo, move):
            return self.p.record_move(st, be, mo, self.tables, self.host.device,
                side=0, slot=0, encounter=1, move_id=move,
                opposing_pressure=self.j.array([False, True]), wire_target_side=-1)
        eager = update(state, beliefs, mons, self.catalog.move_id("bodyslam"))
        got = self.jax.jit(update)(state, beliefs, mons, self.j.array(self.catalog.move_id("bodyslam"), self.j.int32))
        self.same_tree(eager, got)
        self.assertEqual(got.charge.tolist(), [1, 2])
        self.assertFalse(bool(got.beliefs.has_evidence[:, 0].any()))
        self.assertTrue(bool(got.beliefs.has_evidence[:, 1, 0].all()))
        self.same_tree((state, beliefs, mons), before)

    def test_capacity_counter_identity_and_catalog_refusals_are_atomic(self):
        state, beliefs, mons = self.initial()
        for i in range(self.p.CAPACITY):
            result = self.apply_line(state, beliefs, mons, f"|move|p1a: Snorlax|move{i:02d}|p2a: Absol")
            self.assertTrue(bool(result.admitted[0])); state, beliefs = result.state, result.beliefs
        result = self.apply_line(state, beliefs, mons, "|move|p1a: Snorlax|move32|p2a: Absol")
        self.assertFalse(bool(result.admitted[0])); self.same_tree(result.state, state); self.same_tree(result.beliefs, beliefs)
        corrupted = state._replace(binding=state.binding.at[0, 0].add(1))
        result = self.apply_line(corrupted, beliefs, mons, "|move|p1a: Snorlax|move00|p2a: Absol")
        self.assertFalse(bool(result.admitted[0])); self.same_tree(result.state, corrupted)
        slot = int(self.j.argmax(state.move_ids[0, 0, 0] == self.catalog.move_id("move00")))
        full = state._replace(uses=state.uses.at[0, 0, 0, slot].set(2**31-1))
        result = self.apply_line(full, beliefs, mons, "|move|p1a: Snorlax|move00|p2a: Absol")
        self.assertFalse(bool(result.admitted[0])); self.same_tree(result.state, full)
        with self.assertRaises(ValueError): self.catalog.move_id("outside frozen namespace")


if __name__ == "__main__":
    unittest.main()
