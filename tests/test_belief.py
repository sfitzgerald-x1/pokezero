from pathlib import Path
import unittest

from pokezero.belief import (
    CandidateSetSummary,
    PlayerBeliefView,
    PublicBattleBeliefEngine,
    RevealedPokemonBelief,
    sample_opponent_determinizations,
)
from pokezero.showdown import parse_showdown_replay


FIXTURE_ROOT = Path(__file__).parent / "fixtures" / "showdown"


def fixture_lines(name: str) -> list[str]:
    return (FIXTURE_ROOT / name).read_text(encoding="utf-8").splitlines()


class FakeSetSource:
    def summarize(self, *, format_id, species, revealed_moves):
        candidate_count = max(1, 4 - len(revealed_moves))
        uncertainty = candidate_count / 4.0
        return CandidateSetSummary(
            species=species,
            candidate_count=candidate_count,
            uncertainty=uncertainty,
        )


class PublicBattleBeliefEngineTest(unittest.TestCase):
    def test_tracks_public_reveals_moves_and_conditions(self) -> None:
        replay = parse_showdown_replay(fixture_lines("p2_seat_replay.txt"), battle_id="battle-gen3randombattle-1")

        snapshot = PublicBattleBeliefEngine.from_events(replay.public_events).snapshot()
        p1_species = [pokemon.species for pokemon in snapshot.side("p1")]
        xatu = next(pokemon for pokemon in snapshot.side("p1") if pokemon.species == "Xatu")

        self.assertEqual(p1_species, ["Arcanine", "Xatu"])
        self.assertFalse(snapshot.side("p1")[0].active)
        self.assertTrue(xatu.active)
        self.assertEqual(xatu.condition, "70/100")
        self.assertEqual(xatu.revealed_moves, ("Psychic",))
        self.assertNotIn("Thunder Wave", xatu.revealed_moves)

    def test_tracks_item_reveals_from_passive_tags_and_enditem(self) -> None:
        # The explicit `-item` event (Frisk/Trick/Trace) is the rare case. The common items surface
        # as an inline `[from] item:` tag on -heal/-damage (Leftovers, Life Orb) or via `-enditem`
        # (berries, Knock Off). Neither was tracked before, so Leftovers etc. never registered.
        lines = [
            "|start",
            "|switch|p1a: Blissey|Blissey, F|352/352",
            "|switch|p2a: Skarmory|Skarmory, M|271/271",
            "|turn|1",
            "|-damage|p1a: Blissey|300/352",
            "|-heal|p1a: Blissey|322/352|[from] item: Leftovers",
            "|-enditem|p2a: Skarmory|Salac Berry|[eat]",
            "|turn|2",
        ]
        replay = parse_showdown_replay(lines, battle_id="battle-gen3randombattle-1")

        snapshot = PublicBattleBeliefEngine.from_events(replay.public_events).snapshot()
        blissey = next(pokemon for pokemon in snapshot.side("p1") if pokemon.species == "Blissey")
        skarmory = next(pokemon for pokemon in snapshot.side("p2") if pokemon.species == "Skarmory")

        self.assertEqual(blissey.revealed_item, "Leftovers")
        self.assertEqual(skarmory.revealed_item, "Salac Berry")

    def test_player_view_is_overlay_ready_and_player_relative(self) -> None:
        replay = parse_showdown_replay(fixture_lines("p2_seat_replay.txt"), battle_id="battle-gen3randombattle-1")

        view = PublicBattleBeliefEngine.from_events(replay.public_events).snapshot().for_player("p2")
        payload = view.to_overlay_payload()

        self.assertEqual(payload["self_slot"], "p2")
        self.assertEqual(payload["opponent_slot"], "p1")
        self.assertEqual(
            [pokemon["species"] for pokemon in payload["opponent_pokemon"]],
            ["Arcanine", "Xatu"],
        )
        self.assertEqual(payload["opponent_pokemon"][1]["revealed_moves"], ["Psychic"])

    def test_set_source_can_attach_candidate_summary_without_engine_coupling(self) -> None:
        replay = parse_showdown_replay(fixture_lines("p2_seat_replay.txt"), battle_id="battle-gen3randombattle-1")

        snapshot = PublicBattleBeliefEngine.from_events(
            replay.public_events,
            format_id="gen3randombattle",
            set_source=FakeSetSource(),
        ).snapshot()
        xatu = next(pokemon for pokemon in snapshot.side("p1") if pokemon.species == "Xatu")

        self.assertEqual(xatu.candidate_set_count, 3)
        self.assertEqual(xatu.uncertainty, 0.75)


class OpponentBeliefDeterminizationTest(unittest.TestCase):
    def test_samples_player_relative_opponent_variants_in_source_order(self) -> None:
        view = PlayerBeliefView(
            self_slot="p2",
            opponent_slot="p1",
            self_pokemon=(_belief("p2", "Charizard", variants=("self-charizard",)),),
            opponent_pokemon=(
                _belief("p1", "Arcanine", variants=("arcanine-a", "arcanine-b"), active=True),
                _belief("p1", "Xatu", variants=("xatu-a", "xatu-b")),
            ),
        )

        samples = sample_opponent_determinizations(view, sample_count=4)

        self.assertEqual(len(samples), 4)
        self.assertEqual(samples[0].combination_count, 4)
        self.assertEqual(
            [
                tuple(pokemon.variant_id for pokemon in sample.opponent_pokemon)
                for sample in samples
            ],
            [
                ("arcanine-a", "xatu-a"),
                ("arcanine-b", "xatu-a"),
                ("arcanine-a", "xatu-b"),
                ("arcanine-b", "xatu-b"),
            ],
        )
        self.assertEqual(samples[0].self_slot, "p2")
        self.assertEqual(samples[0].opponent_slot, "p1")
        self.assertTrue(samples[0].opponent_pokemon[0].resolved)
        self.assertEqual(samples[0].to_payload()["unresolved_count"], 0)
        self.assertEqual([pokemon.species for pokemon in samples[0].opponent_pokemon], ["Arcanine", "Xatu"])
        self.assertNotIn("Charizard", [pokemon.species for pokemon in samples[0].opponent_pokemon])

    def test_determinization_cap_does_not_repeat_source_order_combinations(self) -> None:
        view = PlayerBeliefView(
            self_slot="p1",
            opponent_slot="p2",
            self_pokemon=(),
            opponent_pokemon=(_belief("p2", "Xatu", variants=("xatu-a", "xatu-b")),),
        )

        samples = sample_opponent_determinizations(view, sample_count=5)

        self.assertEqual(len(samples), 2)
        self.assertEqual([sample.opponent_pokemon[0].variant_id for sample in samples], ["xatu-a", "xatu-b"])

    def test_unsourced_opponent_stays_unresolved_instead_of_inventing_hidden_facts(self) -> None:
        view = PlayerBeliefView(
            self_slot="p1",
            opponent_slot="p2",
            self_pokemon=(),
            opponent_pokemon=(
                RevealedPokemonBelief(
                    showdown_slot="p2",
                    species="Tauros",
                    active=True,
                    revealed_moves=("Hidden Power",),
                    candidate_set_count=0,
                    possible_moves=("hiddenpowerghost",),
                ),
            ),
        )

        samples = sample_opponent_determinizations(view, sample_count=3)

        self.assertEqual(len(samples), 1)
        self.assertEqual(samples[0].combination_count, 1)
        self.assertEqual(samples[0].unresolved_count, 1)
        tauros = samples[0].opponent_pokemon[0]
        self.assertFalse(tauros.resolved)
        self.assertIsNone(tauros.variant_id)
        self.assertEqual(tauros.revealed_moves, ("Hidden Power",))
        self.assertEqual(tauros.possible_moves, ("hiddenpowerghost",))
        self.assertEqual(tauros.moves, ())

    def test_rejects_non_positive_sample_count(self) -> None:
        view = PlayerBeliefView(self_slot="p1", opponent_slot="p2", self_pokemon=(), opponent_pokemon=())

        with self.assertRaisesRegex(ValueError, "sample_count"):
            sample_opponent_determinizations(view, sample_count=0)


def _belief(
    slot: str,
    species: str,
    *,
    variants: tuple[str, ...],
    active: bool = False,
) -> RevealedPokemonBelief:
    return RevealedPokemonBelief(
        showdown_slot=slot,
        species=species,
        active=active,
        candidate_set_count=len(variants),
        uncertainty=1.0,
        possible_abilities=tuple(f"ability-{variant}" for variant in variants),
        possible_items=tuple(f"item-{variant}" for variant in variants),
        possible_moves=tuple(f"move-{variant}" for variant in variants),
        candidate_variants=tuple(
            {
                "variant_id": variant,
                "source_set_id": f"{variant}-source",
                "role": "fixture",
                "level": 80,
                "moves": [f"move-{variant}"],
                "ability": f"ability-{variant}",
                "item": f"item-{variant}",
            }
            for variant in variants
        ),
    )


if __name__ == "__main__":
    unittest.main()
