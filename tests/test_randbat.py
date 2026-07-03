from pathlib import Path
import json
import os
import shutil
import subprocess
import tempfile
import unittest

from pokezero.belief import PublicBattleBeliefEngine
from pokezero.randbat import Gen3RandbatSource
from pokezero.showdown import parse_showdown_replay


GEN3_FIXTURE = {
    "arcanine": {
        "level": 78,
        "sets": [
            {
                "role": "Bulky Support",
                "movepool": ["flamethrower", "hiddenpowergrass", "rest", "toxic"],
                "abilities": ["Intimidate"],
            },
            {
                "role": "Wallbreaker",
                "movepool": ["fireblast", "crunch", "extremespeed", "hiddenpowergrass"],
                "abilities": ["Flash Fire"],
            },
        ],
    },
    "charizard": {
        "level": 79,
        "sets": [
            {
                "role": "Berry Sweeper",
                "movepool": ["dragonclaw", "fireblast", "hiddenpowergrass", "substitute"],
                "abilities": ["Blaze"],
            }
        ],
    },
    "xatu": {
        "level": 84,
        "sets": [
            {
                "role": "Setup Sweeper",
                "movepool": ["calmmind", "hiddenpowerfire", "psychic", "rest"],
                "abilities": ["Early Bird"],
            },
            {
                "role": "Bulky Support",
                "movepool": ["protect", "psychic", "thunderwave", "wish"],
                "abilities": ["Synchronize"],
            },
            {
                "role": "Staller",
                "movepool": ["nightshade", "protect", "toxic", "wish"],
                "abilities": ["Synchronize"],
            },
        ],
    },
    "tauros": {
        "level": 76,
        "sets": [
            {
                "role": "Wallbreaker",
                "movepool": ["doubleedge", "earthquake", "hiddenpowerghost", "return"],
                "abilities": ["Intimidate"],
            }
        ],
    },
    "unown": {
        "level": 100,
        "sets": [
            {
                "role": "Fast Attacker",
                "movepool": ["hiddenpowerpsychic"],
                "abilities": ["Levitate"],
            }
        ],
    },
}


MOVE_METADATA = {
    "calmmind": {"type": "Psychic", "category": "Status", "basePower": 0},
    "crunch": {"type": "Dark", "category": "Special", "basePower": 80},
    "dragonclaw": {"type": "Dragon", "category": "Special", "basePower": 80},
    "extremespeed": {"type": "Normal", "category": "Physical", "basePower": 80, "priority": 1},
    "fireblast": {"type": "Fire", "category": "Special", "basePower": 120, "accuracy": 85},
    "flamethrower": {"type": "Fire", "category": "Special", "basePower": 95},
    "hiddenpowerfire": {"type": "Fire", "category": "Special", "basePower": 70},
    "hiddenpowergrass": {"type": "Grass", "category": "Special", "basePower": 70},
    "hiddenpowerghost": {"type": "Ghost", "category": "Physical", "basePower": 70},
    "hiddenpowerpsychic": {"type": "Psychic", "category": "Special", "basePower": 70},
    "nightshade": {"type": "Ghost", "category": "Physical", "basePower": 0},
    "protect": {"type": "Normal", "category": "Status", "basePower": 0},
    "psychic": {"type": "Psychic", "category": "Special", "basePower": 90},
    "rest": {"type": "Psychic", "category": "Status", "basePower": 0},
    "substitute": {"type": "Normal", "category": "Status", "basePower": 0},
    "thunderwave": {"type": "Electric", "category": "Status", "basePower": 0},
    "toxic": {"type": "Poison", "category": "Status", "basePower": 0},
    "wish": {"type": "Normal", "category": "Status", "basePower": 0},
}


SPECIES_METADATA = {
    "arcanine": {"types": ["Fire"], "baseStats": {"spe": 95}},
    "charizard": {"types": ["Fire", "Flying"], "baseStats": {"spe": 100}},
    "xatu": {"types": ["Psychic", "Flying"], "baseStats": {"spe": 95}},
    "tauros": {"types": ["Normal"], "baseStats": {"spe": 110}},
    "unown": {"types": ["Psychic"], "baseStats": {"spe": 48}},
}


def source() -> Gen3RandbatSource:
    return Gen3RandbatSource.from_data(
        GEN3_FIXTURE,
        move_metadata=MOVE_METADATA,
        species_metadata=SPECIES_METADATA,
    )


class Gen3RandbatSourceTest(unittest.TestCase):
    def test_loads_source_and_builds_variant_universe(self) -> None:
        set_source = source()
        universe = set_source.universe_for("Xatu")

        self.assertIsNotNone(universe)
        self.assertEqual(universe.species, "Xatu")
        self.assertEqual(len(universe.variants), 3)
        self.assertEqual(
            sorted({variant.ability for variant in universe.variants}),
            ["Early Bird", "Synchronize"],
        )

    def test_summary_filters_by_revealed_move_and_exposes_possible_facts(self) -> None:
        summary = source().summarize(
            format_id="gen3randombattle",
            species="Xatu",
            revealed_moves=("Psychic",),
        )

        self.assertIsNotNone(summary)
        self.assertEqual(summary.candidate_count, 2)
        self.assertEqual(summary.possible_abilities, ("Early Bird", "Synchronize"))
        self.assertIn("psychic", summary.possible_moves)
        self.assertNotIn("nightshade", summary.possible_moves)

    def test_off_script_reveal_falls_back_to_full_pool(self) -> None:
        # A move no Xatu set has: Showdown randbats drift from our snapshot, or an unfiltered
        # called/copied move. Instead of returning an empty, uncertainty-0.0 state (which reads as
        # "fully certain"), degrade to the unconstrained pool at maximum uncertainty.
        summary = source().summarize(
            format_id="gen3randombattle",
            species="Xatu",
            revealed_moves=("Surf",),
        )

        self.assertIsNotNone(summary)
        self.assertTrue(summary.inconsistent)
        self.assertEqual(summary.uncertainty, 1.0)
        self.assertGreater(summary.candidate_count, 0)  # not the misleading empty/"certain" state
        self.assertIn("psychic", summary.possible_moves)  # Xatu's real moves are still offered

    def test_consistent_reveal_is_not_flagged_inconsistent(self) -> None:
        summary = source().summarize(
            format_id="gen3randombattle",
            species="Xatu",
            revealed_moves=("Psychic",),
        )
        self.assertFalse(summary.inconsistent)

    def test_generic_hidden_power_reveal_matches_typed_hidden_power_variants(self) -> None:
        summary = source().summarize(
            format_id="gen3randombattle",
            species="Tauros",
            revealed_moves=("Hidden Power",),
        )

        self.assertIsNotNone(summary)
        self.assertEqual(summary.candidate_count, 1)
        self.assertIn("hiddenpowerghost", summary.possible_moves)

    def test_unown_cosmetic_formes_use_base_universe(self) -> None:
        set_source = source()
        base = set_source.universe_for("Unown")
        forme = set_source.universe_for("Unown-Z")

        self.assertIsNotNone(base)
        self.assertIs(forme, base)
        summary = set_source.summarize(
            format_id="gen3randombattle",
            species="Unown-Z",
            revealed_moves=("Hidden Power",),
        )
        self.assertIsNotNone(summary)
        assert summary is not None
        self.assertEqual(summary.species, "Unown")
        self.assertEqual(summary.candidate_count, 1)
        self.assertEqual(summary.possible_abilities, ("Levitate",))

    def test_from_showdown_root_requires_built_dist_generator(self) -> None:
        temp_dir = Path(tempfile.mkdtemp())
        self.addCleanup(lambda: shutil.rmtree(temp_dir))
        sets_dir = temp_dir / "data" / "random-battles" / "gen3"
        sets_dir.mkdir(parents=True)
        (sets_dir / "sets.json").write_text("{}", encoding="utf-8")

        with self.assertRaisesRegex(FileNotFoundError, "Run `node build`"):
            Gen3RandbatSource.from_showdown_root(temp_dir, use_cache=False)

    @unittest.skipUnless(os.environ.get("POKEZERO_SHOWDOWN_ROOT"), "POKEZERO_SHOWDOWN_ROOT is not set")
    def test_optional_showdown_root_covers_representative_showdown_output(self) -> None:
        showdown_root = Path(os.environ["POKEZERO_SHOWDOWN_ROOT"])
        set_source = Gen3RandbatSource.from_showdown_root(
            showdown_root,
            use_cache=False,
        )
        xatu = set_source.universe_for("Xatu")
        sampled = _sample_showdown_set(showdown_root, "xatu")

        self.assertIsNotNone(xatu)
        self.assertGreater(len(xatu.variants), 0)
        self.assertTrue(
            any(
                variant.role == sampled["role"]
                and variant.ability == sampled["ability"]
                and variant.item == sampled["item"]
                and set(variant.moves) == set(sampled["moves"])
                for variant in xatu.variants
            )
        )


class Gen3RandbatBeliefPruningTest(unittest.TestCase):
    def test_revealed_move_prunes_surviving_variants(self) -> None:
        replay = parse_showdown_replay(
            [
                "|switch|p1a: Xatu|Xatu, L84|100/100",
                "|move|p1a: Xatu|Psychic|p2a: Charizard",
            ],
            battle_id="battle-gen3randombattle-test",
        )

        engine = PublicBattleBeliefEngine.from_events(
            replay.public_events,
            format_id="gen3randombattle",
            set_source=source(),
        )
        xatu = engine.snapshot().side("p1")[0]

        self.assertEqual(xatu.revealed_moves, ("Psychic",))
        self.assertEqual(xatu.candidate_set_count, 2)
        self.assertTrue(any(item.kind == "revealed-move" for item in xatu.evidence))

    def test_public_ability_reveal_prunes_surviving_variants(self) -> None:
        replay = parse_showdown_replay(
            [
                "|switch|p1a: Arcanine|Arcanine, L78|100/100",
                "|-ability|p1a: Arcanine|Flash Fire",
            ],
            battle_id="battle-gen3randombattle-test",
        )

        engine = PublicBattleBeliefEngine.from_events(
            replay.public_events,
            format_id="gen3randombattle",
            set_source=source(),
        )
        arcanine = engine.snapshot().side("p1")[0]

        self.assertEqual(arcanine.revealed_ability, "Flash Fire")
        self.assertEqual(arcanine.possible_abilities, ("Flash Fire",))
        self.assertEqual(arcanine.candidate_set_count, 1)

    def test_raw_ability_effect_reveal_prunes_surviving_variants(self) -> None:
        replay = parse_showdown_replay(
            [
                "|switch|p1a: Arcanine|Arcanine, L78|100/100",
                "|-immune|p1a: Arcanine|[from] ability: Flash Fire|[of] p1a: Arcanine",
            ],
            battle_id="battle-gen3randombattle-test",
        )

        engine = PublicBattleBeliefEngine.from_events(
            replay.public_events,
            format_id="gen3randombattle",
            set_source=source(),
        )
        arcanine = engine.snapshot().side("p1")[0]

        self.assertEqual(arcanine.revealed_ability, "Flash Fire")
        self.assertEqual(arcanine.possible_abilities, ("Flash Fire",))

    def test_intimidate_trigger_confirms_intimidate(self) -> None:
        replay = parse_showdown_replay(
            [
                "|switch|p2a: Charizard|Charizard, L79|100/100",
                "|switch|p1a: Arcanine|Arcanine, L78|100/100",
                "|-ability|p1a: Arcanine|Intimidate",
            ],
            battle_id="battle-gen3randombattle-test",
        )

        engine = PublicBattleBeliefEngine.from_events(
            replay.public_events,
            format_id="gen3randombattle",
            set_source=source(),
        )
        arcanine = engine.snapshot().side("p1")[0]

        self.assertEqual(arcanine.revealed_ability, "Intimidate")
        self.assertEqual(arcanine.possible_abilities, ("Intimidate",))

    def test_safe_intimidate_non_trigger_rules_out_intimidate(self) -> None:
        replay = parse_showdown_replay(
            [
                "|switch|p2a: Charizard|Charizard, L79|100/100",
                "|switch|p1a: Arcanine|Arcanine, L78|100/100",
                "|turn|1",
            ],
            battle_id="battle-gen3randombattle-test",
        )

        engine = PublicBattleBeliefEngine.from_events(
            replay.public_events,
            format_id="gen3randombattle",
            set_source=source(),
        )
        engine.resolve_pending_switches_at_boundary()
        arcanine = engine.snapshot().side("p1")[0]

        self.assertEqual(arcanine.ruled_out_abilities, ("Intimidate",))
        self.assertEqual(arcanine.possible_abilities, ("Flash Fire",))
        self.assertEqual(arcanine.candidate_set_count, 1)


if __name__ == "__main__":
    unittest.main()


def _sample_showdown_set(showdown_root: Path, species: str) -> dict[str, object]:
    script = """
const root = process.argv[1];
const species = process.argv[2];
const {RandomGen3Teams} = require(root + '/dist/data/random-battles/gen3/teams.js');
const generator = new RandomGen3Teams('gen3randombattle', [1, 2, 3, 4]);
const set = generator.randomSet(species);
console.log(JSON.stringify({role: set.role, ability: set.ability, item: set.item, moves: [...set.moves].sort()}));
"""
    result = subprocess.run(
        ["node", "-e", script, str(showdown_root), species],
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(result.stdout)
