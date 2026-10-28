"""Own-team base v4 columns compared with the actual Python token writer."""
import copy
import importlib.util
import unittest


@unittest.skipUnless(importlib.util.find_spec("jax"), "requires isolated JAX")
class PrivateTeamTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import jax
        import jax.numpy as j
        from gpu_engine import private_team as p
        from pokezero import showdown as s
        from pokezero.category_vocab import CategoryVocabulary
        from pokezero.dex import ShowdownDex, SpeciesInfo
        cls.jax, cls.j, cls.p, cls.s = jax, j, p, s
        cls.dex = ShowdownDex({}, {"pikachu": SpeciesInfo("pikachu", "Pikachu", ("Electric",),
            dict(hp=35, atk=55, **{"def": 40}, spa=50, spd=50, spe=90))}, {})
        cls.vocab = CategoryVocabulary(("pokemon:self", "species:pikachu", "status:none", "status:par",
            "status:tox", "type:electric", "belief:possible_ability:static", "belief:possible_item:lightball"))

    def request(self):
        return {"side": {"id": "p1", "pokemon": [
            {"ident": "p1: first", "details": "Pikachu, L80, M", "condition": "99/201 par",
             "active": True, "ability": "Static", "item": "Light Ball",
             "stats": {"atk": 144, "def": 110, "spa": 170, "spd": 88, "spe": 215}},
            {"ident": "p1: duplicate", "details": "Pikachu, F", "condition": "0 fnt",
             "active": False, "baseAbility": "Static", "item": ""},
            {"ident": "p1: third", "details": "Pikachu, L150", "condition": "250/200 tox",
             "stats": {"atk": 999, "def": -2}}]}, "active": [{"moves": [{"id": "surf"}]}]}

    def test_all_27_owned_numeric_and_seven_category_columns_match_actual_writer(self):
        from dataclasses import replace
        raw = self.request(); before = copy.deepcopy(raw)
        packet = self.p.parse_team_facts(raw, owner_side=0, vocabulary=self.vocab, dex=self.dex)
        got = self.p.project_team(packet, observer_side=0, expected_vocabulary_binding=self.p.vocabulary_binding(self.vocab))
        team = self.s._self_team_from_request(raw, "p1")
        numeric, categories = [[0.0] * 200 for _ in range(6)], [[""] * 60 for _ in range(6)]
        self.s._encode_pokemon_tokens(categories, numeric, 0, team, role="self", limit=6,
            dex=self.dex, schema_v3=True, schema_v4=True,
            masks=replace(self.s.DEFAULT_OBSERVATION_FEATURE_MASKS, exact_state=False))
        for row in range(6):
            for name, column in zip(self.p.CORE, self.p.NUMERIC_COLUMNS):
                writer = getattr(self.s, "NUMERIC_" + name)
                self.assertAlmostEqual(float(got.numeric[row, column]), numeric[row][writer], places=6,
                                       msg=f"row{row} {name}")
            for column in self.p.CATEGORY_COLUMNS:
                self.assertEqual(int(got.categorical[row, column]), self.vocab.encode(categories[row][column]))
        self.assertEqual(int(got.numeric_columns.sum()), 27)
        self.assertEqual(int(got.categorical_columns.sum()), 7)
        self.assertEqual(raw, before)

    def test_jit_batch_and_wrong_seat_vocabulary_refusal(self):
        p = self.p.parse_team_facts(self.request(), owner_side=0, vocabulary=self.vocab, dex=self.dex)
        batch = self.jax.tree.map(lambda x: self.j.stack([x, x]), p)
        binding = self.p.vocabulary_binding(self.vocab)
        fn = self.jax.jit(lambda x, owner, digest: self.p.project_team(x, observer_side=owner,
            expected_vocabulary_binding=digest))
        got = fn(batch, self.j.array([0, 1], self.j.int32), binding)
        self.assertEqual(got.admitted.tolist(), [True, False])
        self.assertTrue(bool(self.j.all(got.numeric[1] == 0)))
        self.assertTrue(bool(self.j.all(got.categorical[1] == 0)))
        wrong = fn(batch, self.j.array([0, 0], self.j.int32), binding.at[0].add(1))
        self.assertFalse(bool(wrong.admitted.any()))
        self.assertTrue(bool(self.j.all(wrong.numeric == 0)))
        for owner in [True, .1, self.j.array(True), self.j.array(.1)]:
            with self.assertRaises(ValueError):
                self.p.project_team(p, observer_side=owner, expected_vocabulary_binding=binding)

    def test_item_removal_unknown_details_and_absent_request_preserve_source_defaults(self):
        raw = self.request(); row = raw["side"]["pokemon"][0]
        row.update(item="", ability="", baseAbility="", details=None, condition=None, stats={})
        p = self.p.parse_team_facts(raw, owner_side=0, vocabulary=self.vocab, dex=self.dex)
        got = self.p.project_team(p, observer_side=0, expected_vocabulary_binding=self.p.vocabulary_binding(self.vocab))
        for name in ["REVEALED_ITEM", "POSSIBLE_ITEM_COUNT", "REVEALED_ABILITY", "POSSIBLE_ABILITY_COUNT", "ACTUAL_HP", "LEVEL", "HP_FRACTION"]:
            self.assertEqual(float(got.numeric[0, self.s.v4_numeric_index(getattr(self.s, "NUMERIC_" + name))]), 0)
        absent = self.p.parse_team_facts(None, owner_side=0, vocabulary=self.vocab)
        empty = self.p.project_team(absent, observer_side=0, expected_vocabulary_binding=self.p.vocabulary_binding(self.vocab))
        self.assertTrue(bool(self.j.all(empty.numeric == 0)))
        self.assertTrue(bool(self.j.all(empty.categorical == 0)))

    def test_unsupported_nonfinite_hp_and_opponent_request_fail_admission(self):
        for condition in ["nan/100", "3e-45/6e-45", "inf/100"]:
            raw = self.request(); raw["side"]["pokemon"][0]["condition"] = condition
            with self.assertRaises(ValueError):
                self.p.parse_team_facts(raw, owner_side=0, vocabulary=self.vocab, dex=self.dex)
        with self.assertRaisesRegex(ValueError, "different seat"):
            self.p.parse_team_facts(self.request(), owner_side=1, vocabulary=self.vocab, dex=self.dex)
        raw = self.request(); raw["side"]["pokemon"][0]["stats"]["atk"] = 10**1000
        with self.assertRaisesRegex(ValueError, "private stat"):
            self.p.parse_team_facts(raw, owner_side=0, vocabulary=self.vocab, dex=self.dex)


if __name__ == "__main__":
    unittest.main()
