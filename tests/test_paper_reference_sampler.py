"""Paper hidden-set sampling contracts, including the real pinned server API."""

import hashlib
import json
import random
import subprocess
from types import SimpleNamespace
import unittest

from _showdown_root import requires_showdown, showdown_root
from pokezero.local_showdown import LocalShowdownConfig, LocalShowdownEnv, LocalShowdownError
from pokezero.mcts_eval.paper_reference import ReferenceRefusal
from pokezero.mcts_eval.paper_reference_sampling import KnownSetTraits, PaperHiddenTeamSampler
from pokezero.randbat import load_gen3_randbat_source_cached


def set_row(species="Snorlax", moves=None, item="Leftovers"):
    return dict(species=species, moves=moves or ["return", "rest", "sleeptalk", "earthquake"],
        ability="Immunity", item=item, level=80, nature="", gender="M",
        evs={s: 85 for s in ("hp", "atk", "def", "spa", "spd", "spe")},
        ivs={s: 31 for s in ("hp", "atk", "def", "spa", "spd", "spe")})


class Generator:
    def __init__(self, draws=None, party=None):
        self.draws = draws or [set_row()]
        self.party = party or [set_row(s) for s in ("Pikachu", "Charizard", "Blastoise", "Venusaur", "Gengar", "Alakazam")]
        self.calls = []

    def generate_reference_set(self, *, seed, species):
        index = sum(c[0] == "set" for c in self.calls)
        self.calls.append(("set", seed, species))
        return self.draws[min(index, len(self.draws)-1)]

    def generate_scenario_team(self, *, seed):
        self.calls.append(("party", seed))
        return self.party


class ReferenceSamplingTests(unittest.TestCase):
    def sampler(self, generator):
        # Test Dex metadata only: no catalog or private team is available.
        source = SimpleNamespace(species_metadata={}, move_metadata={})
        return PaperHiddenTeamSampler(generator, set_source=source)

    def test_known_match_stops_immediately_and_receipts_bind_draws(self):
        generator = Generator()
        draw = self.sampler(generator).draw((KnownSetTraits("Snorlax", ("Return102",), "Immunity", "Leftovers"),), random.Random(1))
        self.assertEqual(len(draw.team), 6)
        self.assertEqual(draw.known[0].seeds, (generator.calls[0][1],))
        self.assertEqual(draw.forced_sets, 0)
        self.assertEqual(len(draw.unknown_party_seeds), 1)
        self.assertEqual(len(draw.packed_team_sha256), 64)

    def test_known_rejection_uses_same_species_until_matching_set(self):
        generator = Generator(draws=[set_row(item="Choice Band"), set_row(item="Leftovers")])
        draw = self.sampler(generator).draw((KnownSetTraits("Snorlax", item="Leftovers"),), random.Random(2))
        self.assertEqual([c[2] for c in generator.calls if c[0] == "set"], ["Snorlax", "Snorlax"])
        self.assertEqual(len(draw.known[0].seeds), 2)
        self.assertFalse(draw.known[0].forced)

    def test_exact_source_max_hp_conditions_draws_not_posthoc_hp_invention(self):
        low = set_row()
        low['evs']['hp'] = 0
        source = SimpleNamespace(species_metadata={'snorlax': {'baseStats': {'hp': 160}}}, move_metadata={})
        generator = Generator(draws=[low, set_row()])
        sampler = PaperHiddenTeamSampler(generator, set_source=source)
        # floor((2*160+31+floor(85/4))*80/100) + 80 + 10 = 387.
        draw = sampler.draw((KnownSetTraits('Snorlax', max_hp=387),), random.Random(1))
        self.assertEqual(len(draw.known[0].seeds), 2)
        self.assertFalse(draw.known[0].forced)

    def test_tenth_failed_draw_forces_traits_and_recomputes_hidden_power_ivs(self):
        generator = Generator()
        traits = KnownSetTraits("Snorlax", ("Hidden Power Ice", "Surf", "Thunderbolt", "Psychic"),
            "Thick Fat", "", 85, "F")
        draw = self.sampler(generator).draw((traits,), random.Random(3))
        self.assertEqual(len(draw.known[0].seeds), 10)
        self.assertEqual(draw.forced_sets, 1)
        mon = draw.team[0]
        self.assertEqual(mon.moves, ("hiddenpowerice", "surf", "thunderbolt", "psychic"))
        self.assertEqual((mon.ability, mon.item, mon.level, mon.gender), ("Thick Fat", "", 85, "F"))
        self.assertEqual(mon.ivs["atk"], 2)  # 30 minus Gen 3 confusion-damage adjustment.
        self.assertEqual(mon.ivs["def"], 30)

    def test_unknown_item_is_not_forced_to_itemless(self):
        draw = self.sampler(Generator()).draw((KnownSetTraits("Snorlax"),), random.Random(0))
        self.assertEqual(draw.team[0].item, "Leftovers")
        self.assertFalse(draw.known[0].forced)

    def test_generic_public_hidden_power_accepts_typed_generator_set_without_forcing_type(self):
        row = set_row(moves=["hiddenpowerfire", "rest", "icebeam", "sleeptalk"])
        draw = self.sampler(Generator(draws=[row])).draw((KnownSetTraits("Snorlax", ("Hidden Power",)),), random.Random(0))
        self.assertEqual(len(draw.known[0].seeds), 1)
        self.assertFalse(draw.known[0].forced)
        self.assertIn("hiddenpowerfire", draw.team[0].moves)

    def test_known_gender_is_conditioned_after_server_delegated_gender_not_ten_rejections(self):
        row = set_row()
        row["gender"] = ""
        draw = self.sampler(Generator(draws=[row])).draw((KnownSetTraits("Snorlax", gender="F"),), random.Random(0))
        self.assertEqual(len(draw.known[0].seeds), 1)
        self.assertTrue(draw.known[0].gender_assigned_after_server_draw)
        self.assertEqual(draw.team[0].gender, "F")
        self.assertFalse(draw.known[0].forced)

    def test_every_call_draws_fresh_with_reproducible_rng_receipts(self):
        generator = Generator()
        sampler = self.sampler(generator)
        rng = random.Random(10)
        first = sampler.draw((), rng)
        second = sampler.draw((), rng)
        self.assertNotEqual(first.unknown_party_seeds, second.unknown_party_seeds)
        again = self.sampler(Generator()).draw((), random.Random(10))
        self.assertEqual(first, again)
        self.assertEqual(len(generator.calls), 2)

    def test_species_clause_filters_known_species_from_unknown_party(self):
        generator = Generator(party=[set_row(s) for s in ("Snorlax", "Pikachu", "Charizard", "Blastoise", "Venusaur", "Gengar")])
        draw = self.sampler(generator).draw((KnownSetTraits("Snorlax"),), random.Random(1))
        self.assertEqual([m.species for m in draw.team].count("Snorlax"), 1)
        self.assertEqual(len({m.species for m in draw.team}), 6)

    def test_unknown_rejection_cap_refuses_instead_of_inventing_missing_species(self):
        generator = Generator(party=[set_row() for _ in range(6)])
        with self.assertRaisesRegex(ReferenceRefusal, "safety cap"):
            self.sampler(generator).draw((KnownSetTraits("Snorlax"),), random.Random(2))
        self.assertEqual(sum(c[0] == "party" for c in generator.calls), 10)

    def test_species_drift_partial_party_and_bad_known_inputs_refuse(self):
        with self.assertRaisesRegex(ReferenceRefusal, "changed species"):
            self.sampler(Generator(draws=[set_row("Gengar")])).draw((KnownSetTraits("Snorlax"),), random.Random(0))
        with self.assertRaisesRegex(ReferenceRefusal, "partial party"):
            self.sampler(Generator(party=[set_row()])).draw((), random.Random(0))
        with self.assertRaisesRegex(ReferenceRefusal, "species clause"):
            self.sampler(Generator()).draw((KnownSetTraits("Snorlax"), KnownSetTraits("snorlax")), random.Random(0))
        for kwargs in ({"moves": ("return", "return102")}, {"moves": ("",)}, {"level": True}, {"gender": "?"}):
            with self.assertRaises(ReferenceRefusal):
                KnownSetTraits("Snorlax", **kwargs)


class ReferenceSetBridgeTests(unittest.TestCase):
    def test_invalid_inputs_refuse_without_starting_a_bridge(self):
        env = LocalShowdownEnv()
        try:
            for seed in (True, -1, 2**53, 1.5, "1"):
                with self.assertRaises(ValueError):
                    env.generate_reference_set(seed=seed, species="Snorlax")
            for species in (None, "", " ", 1):
                with self.assertRaises(ValueError):
                    env.generate_reference_set(seed=1, species=species)
            self.assertIsNone(env._process)
        finally:
            env.close()

    @requires_showdown()
    def test_matches_independent_exact_species_generator_not_whole_party_rejection(self):
        seed = 1234567
        digest = hashlib.sha256(f"{seed}:reference-known-set".encode()).digest()
        parts = [int.from_bytes(digest[i:i+2], "big") for i in range(0, 8, 2)]
        direct = subprocess.run(["node", "-e", """
          const {Teams} = require(process.argv[1]);
          const generator = Teams.getGenerator('gen3randombattle', JSON.parse(process.argv[2]));
          const set = generator.randomSet(generator.dex.species.get('Snorlax'), {}, false);
          console.log(JSON.stringify(set));
        """, str(showdown_root() / "dist/sim/index.js"), json.dumps(parts)],
            capture_output=True, text=True, check=True, timeout=30)
        expected = json.loads(direct.stdout)
        env = LocalShowdownEnv(LocalShowdownConfig(showdown_root=showdown_root()))
        try:
            actual = env.generate_reference_set(seed=seed, species="Snorlax")
            for key in ("species", "moves", "ability", "item", "level", "evs", "ivs"):
                self.assertEqual(actual[key], expected[key])
            self.assertEqual(actual, env.generate_reference_set(seed=seed, species="snorlax"))
            self.assertNotEqual(actual, env.generate_reference_set(seed=seed+1, species="Snorlax"))
        finally:
            env.close()

    @requires_showdown()
    def test_unknown_species_refuses_instead_of_using_an_unconditioned_party(self):
        env = LocalShowdownEnv(LocalShowdownConfig(showdown_root=showdown_root()))
        try:
            with self.assertRaises(LocalShowdownError):
                env.generate_reference_set(seed=1, species="NotAPokemon")
        finally:
            env.close()

    @requires_showdown()
    def test_real_generator_draws_new_teams_and_conditions_revealed_traits(self):
        env = LocalShowdownEnv(LocalShowdownConfig(showdown_root=showdown_root()))
        try:
            source = load_gen3_randbat_source_cached(showdown_root())
            sampler = PaperHiddenTeamSampler(env, set_source=source)
            rng = random.Random(91)
            first = sampler.draw((KnownSetTraits("Snorlax", ("rest",)),), rng)
            second = sampler.draw((KnownSetTraits("Snorlax", ("rest",)),), rng)
            self.assertNotEqual(first.packed_team_sha256, second.packed_team_sha256)
            self.assertIn("rest", first.team[0].moves)
            self.assertIn("rest", second.team[0].moves)
            self.assertEqual(len(first.team), 6)
            self.assertEqual(len(second.team), 6)
            self.assertTrue(all(1 <= len(r.seeds) <= 10 for r in (*first.known, *second.known)))
        finally:
            env.close()


if __name__ == "__main__":
    unittest.main()
