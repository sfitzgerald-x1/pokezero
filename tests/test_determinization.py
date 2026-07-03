from __future__ import annotations

import random
import unittest

from pokezero.actions import ACTION_COUNT
from pokezero.determinization import (
    gen3_randbat_belief_start_override,
    gen3_randbat_belief_start_override_planner,
    player_belief_view_from_payload,
)
from pokezero.observation import PokeZeroObservationV0
from pokezero.policy import PolicyContext
from pokezero.randbat import Gen3RandbatSource
from pokezero.search_policy import OpponentActionScenario
from pokezero.trajectory import BattleTrajectory


def _source() -> Gen3RandbatSource:
    return Gen3RandbatSource.from_data(
        {
            "xatu": {
                "level": 84,
                "sets": [
                    {
                        "role": "Support",
                        "movepool": ["psychic", "thunderwave", "wish", "protect"],
                        "abilities": ["Synchronize"],
                    },
                    {
                        "role": "Setup",
                        "movepool": ["psychic", "calmmind", "rest", "hiddenpowerfire"],
                        "abilities": ["Early Bird"],
                    },
                ],
            },
            "arcanine": {
                "level": 78,
                "sets": [
                    {
                        "role": "Breaker",
                        "movepool": ["fireblast", "crunch", "extremespeed", "hiddenpowergrass"],
                        "abilities": ["Flash Fire"],
                    }
                ],
            },
            "tauros": {
                "level": 76,
                "sets": [
                    {
                        "role": "Breaker",
                        "movepool": ["return", "earthquake", "doubleedge", "hiddenpowerghost"],
                        "abilities": ["Intimidate"],
                    }
                ],
            },
        }
    )


def _context(metadata: dict[str, object], *, format_id: str = "gen3randombattle") -> PolicyContext:
    observation = PokeZeroObservationV0(
        categorical_ids=(),
        numeric_features=(),
        token_type_ids=(),
        attention_mask=(),
        legal_action_mask=tuple(index == 0 for index in range(ACTION_COUNT)),
        metadata=metadata,
    )
    return PolicyContext(
        player_id="p2",
        decision_round_index=0,
        battle_id="belief-start",
        format_id=format_id,
        seed=11,
        observation=observation,
        requested_players=("p1", "p2"),
        trajectory=BattleTrajectory(battle_id="belief-start", format_id=format_id, seed=11),
        requested_legal_action_masks={"p2": observation.legal_action_mask},
    )


def _metadata() -> dict[str, object]:
    return {
        "self_team": [
            {
                "showdown_slot": "p2",
                "species": "Charizard",
                "details": "Charizard, L79",
                "moves": ["fireblast", "dragonclaw", "hiddenpowergrass", "substitute"],
                "ability": "Blaze",
                "item": "Petaya Berry",
            },
            {
                "showdown_slot": "p2",
                "species": "Blissey",
                "details": "Blissey, L75",
                "moves": ["seismictoss", "softboiled", "toxic", "thunderwave"],
                "ability": "Natural Cure",
                "item": "Leftovers",
            },
            {
                "showdown_slot": "p2",
                "species": "Snorlax",
                "details": "Snorlax, L75",
                "moves": ["return", "earthquake", "rest", "curse"],
                "ability": "Immunity",
                "item": "Leftovers",
            },
        ],
        "belief_view": {
            "self_slot": "p2",
            "opponent_slot": "p1",
            "self_pokemon": [],
            "opponent_pokemon": [
                {
                    "showdown_slot": "p1",
                    "species": "Xatu",
                    "active": True,
                    "revealed_moves": ["Psychic"],
                    "candidate_variants": [
                        {
                            "variant_id": "xatu-support",
                            "source_set_id": "xatu-1",
                            "role": "Support",
                            "level": 84,
                            "moves": ["psychic", "thunderwave", "wish", "protect"],
                            "ability": "Synchronize",
                            "item": "Leftovers",
                        }
                    ],
                }
            ],
        },
    }


class Gen3RandbatBeliefStartOverrideTest(unittest.TestCase):
    def test_parses_player_belief_view_payload(self) -> None:
        view = player_belief_view_from_payload(_metadata()["belief_view"])

        self.assertIsNotNone(view)
        assert view is not None
        self.assertEqual(view.self_slot, "p2")
        self.assertEqual(view.opponent_slot, "p1")
        self.assertEqual(view.opponent_pokemon[0].species, "Xatu")
        self.assertEqual(view.opponent_pokemon[0].revealed_moves, ("Psychic",))

    def test_builds_player_relative_custom_start_override_from_public_belief(self) -> None:
        context = _context(_metadata())

        override = gen3_randbat_belief_start_override(
            context=context,
            set_source=_source(),
            rng=random.Random(7),
            team_size=3,
        )

        self.assertIsNotNone(override)
        assert override is not None
        self.assertEqual(set(override.player_teams), {"p1", "p2"})
        self.assertIn("Charizard", override.player_teams["p2"])
        self.assertIn("Blaze", override.player_teams["p2"])
        self.assertIn("Xatu", override.player_teams["p1"])
        self.assertIn("Synchronize", override.player_teams["p1"])
        self.assertEqual(len(override.player_teams["p1"].split("]")), 3)
        self.assertEqual(override.player_teams["p1"].count("Xatu"), 1)

    def test_planner_returns_resampling_source_for_supported_format(self) -> None:
        planner = gen3_randbat_belief_start_override_planner(_source(), team_size=3)
        context = _context(_metadata())

        source = planner(
            context,
            OpponentActionScenario(actions={"p1": 0}),
            0,
            random.Random(3),
        )

        self.assertTrue(callable(source))
        assert callable(source)
        self.assertIsNotNone(source())

    def test_unsupported_format_disables_planner(self) -> None:
        planner = gen3_randbat_belief_start_override_planner(_source(), team_size=3)

        source = planner(
            _context(_metadata(), format_id="gen3ou"),
            OpponentActionScenario(actions={"p1": 0}),
            0,
            random.Random(3),
        )

        self.assertIsNone(source)

    def test_incomplete_self_team_returns_none_instead_of_inventing_own_moves(self) -> None:
        metadata = _metadata()
        metadata["self_team"] = [
            {
                "showdown_slot": "p2",
                "species": "Charizard",
                "details": "Charizard, L79",
                "moves": [],
                "ability": "Blaze",
            }
        ]

        override = gen3_randbat_belief_start_override(
            context=_context(metadata),
            set_source=_source(),
            rng=random.Random(7),
            team_size=3,
        )

        self.assertIsNone(override)


if __name__ == "__main__":
    unittest.main()
