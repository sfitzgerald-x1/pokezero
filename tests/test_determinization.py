from __future__ import annotations

from dataclasses import replace
import os
from pathlib import Path
import random
import shutil
import unittest

from pokezero.actions import ACTION_COUNT
from pokezero.determinization import (
    _gen3_randbat_fixture_spread,
    belief_world_sampling_profile,
    gen3_randbat_belief_start_override,
    gen3_randbat_belief_start_override_planner,
    player_belief_view_from_payload,
)
from pokezero.local_showdown import DEFAULT_SHOWDOWN_ROOT, LocalShowdownConfig, LocalShowdownEnv
from pokezero.observation import PokeZeroObservationV0
from pokezero.policy import PolicyContext
from pokezero.public_decision_corpus import PublicActionIdentifier, PublicResolvedActionRound
from pokezero.randbat import Gen3RandbatSource
from pokezero.replay_branching import replay_trajectory_branch
from pokezero.search_policy import OpponentActionScenario
from pokezero.trajectory import BattleTrajectory, TrajectoryStep


MOVE_METADATA = {
    "calmmind": {"type": "Psychic", "category": "Status", "basePower": 0},
    "crunch": {"type": "Dark", "category": "Special", "basePower": 80},
    "curse": {"type": "Ghost", "category": "Status", "basePower": 0},
    "doubleedge": {"type": "Normal", "category": "Physical", "basePower": 120},
    "dragonclaw": {"type": "Dragon", "category": "Special", "basePower": 80},
    "earthquake": {"type": "Ground", "category": "Physical", "basePower": 100},
    "extremespeed": {"type": "Normal", "category": "Physical", "basePower": 80},
    "fireblast": {"type": "Fire", "category": "Special", "basePower": 120},
    "hiddenpowerfire": {"type": "Fire", "category": "Special", "basePower": 70},
    "hiddenpowerghost": {"type": "Ghost", "category": "Physical", "basePower": 70},
    "hiddenpowerground": {"type": "Ground", "category": "Physical", "basePower": 70},
    "hiddenpowergrass": {"type": "Grass", "category": "Special", "basePower": 70},
    "hiddenpowerpsychic": {"type": "Psychic", "category": "Special", "basePower": 70},
    "protect": {"type": "Normal", "category": "Status", "basePower": 0},
    "psychic": {"type": "Psychic", "category": "Special", "basePower": 90},
    "rest": {"type": "Psychic", "category": "Status", "basePower": 0},
    "return": {"type": "Normal", "category": "Physical", "basePower": 0},
    "seismictoss": {"type": "Fighting", "category": "Physical", "basePower": 0, "damage": "level"},
    "softboiled": {"type": "Normal", "category": "Status", "basePower": 0},
    "substitute": {"type": "Normal", "category": "Status", "basePower": 0},
    "thunderwave": {"type": "Electric", "category": "Status", "basePower": 0},
    "toxic": {"type": "Poison", "category": "Status", "basePower": 0},
    "wish": {"type": "Normal", "category": "Status", "basePower": 0},
}
SPECIES_METADATA = {
    "arcanine": {"baseStats": {"hp": 90, "atk": 110, "def": 80, "spa": 100, "spd": 80, "spe": 95}},
    "blissey": {"baseStats": {"hp": 255, "atk": 10, "def": 10, "spa": 75, "spd": 135, "spe": 55}},
    "charizard": {"baseStats": {"hp": 78, "atk": 84, "def": 78, "spa": 109, "spd": 85, "spe": 100}},
    "registeel": {"baseStats": {"hp": 80, "atk": 75, "def": 150, "spa": 75, "spd": 150, "spe": 50}},
    "snorlax": {"baseStats": {"hp": 160, "atk": 110, "def": 65, "spa": 65, "spd": 110, "spe": 30}},
    "tauros": {"baseStats": {"hp": 75, "atk": 100, "def": 95, "spa": 40, "spd": 70, "spe": 110}},
    "unown": {"baseStats": {"hp": 48, "atk": 72, "def": 48, "spa": 72, "spd": 48, "spe": 48}},
    "umbreon": {"baseStats": {"hp": 95, "atk": 65, "def": 110, "spa": 60, "spd": 130, "spe": 65}},
    "xatu": {"baseStats": {"hp": 65, "atk": 75, "def": 70, "spa": 95, "spd": 70, "spe": 95}},
}


def integration_config() -> LocalShowdownConfig | None:
    root = Path(os.environ.get("POKEZERO_SHOWDOWN_ROOT") or DEFAULT_SHOWDOWN_ROOT)
    if not (root / "dist" / "sim" / "index.js").exists():
        return None
    if shutil.which("node") is None:
        return None
    return LocalShowdownConfig(showdown_root=root, read_timeout_seconds=10.0)


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
        },
        move_metadata=MOVE_METADATA,
        species_metadata=SPECIES_METADATA,
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
                "stats": {"hp": 252, "atk": 139, "def": 169, "spa": 217, "spd": 180, "spe": 204},
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

    def test_belief_world_sampling_profile_is_bounded_by_public_variant_combinations(self) -> None:
        metadata = _metadata()
        opponent = metadata["belief_view"]["opponent_pokemon"][0]  # type: ignore[index]
        opponent["candidate_variants"].append(  # type: ignore[index]
            {
                "variant_id": "xatu-setup",
                "source_set_id": "xatu-2",
                "role": "Setup",
                "level": 84,
                "moves": ["psychic", "calmmind", "rest", "hiddenpowerfire"],
                "ability": "Early Bird",
                "item": "Leftovers",
            }
        )

        profile = belief_world_sampling_profile(
            _context(metadata),
            sample_cap=8,
            set_source=_source(),
            team_size=3,
        )

        self.assertIsNotNone(profile)
        assert profile is not None
        self.assertEqual(profile.combination_count, 12)
        self.assertEqual(profile.sample_count, 8)
        self.assertGreater(profile.uncertainty_bits, 3.0)
        self.assertEqual(profile.uncertain_slot_count, 3)

    def test_belief_world_sampling_checksum_ignores_true_hidden_team_payload(self) -> None:
        public_metadata = _metadata()
        private_support = PokeZeroObservationV0(
            categorical_ids=(),
            numeric_features=(),
            token_type_ids=(),
            attention_mask=(),
            legal_action_mask=tuple(index == 0 for index in range(ACTION_COUNT)),
            metadata={"self_team": [{"species": "Xatu", "moves": ["Psychic", "Wish"]}]},
        )
        private_setup = replace(
            private_support,
            metadata={"self_team": [{"species": "Xatu", "moves": ["Psychic", "Calm Mind"]}]},
        )
        first_context = replace(_context(public_metadata), requested_observations={"p1": private_support})
        second_context = replace(_context(public_metadata), requested_observations={"p1": private_setup})

        first_profile = belief_world_sampling_profile(
            first_context,
            sample_cap=4,
            set_source=_source(),
            team_size=3,
        )
        second_profile = belief_world_sampling_profile(
            second_context,
            sample_cap=4,
            set_source=_source(),
            team_size=3,
        )

        self.assertIsNotNone(first_profile)
        self.assertIsNotNone(second_profile)
        assert first_profile is not None and second_profile is not None
        self.assertEqual(first_profile.public_checksum, second_profile.public_checksum)
        self.assertEqual(first_profile.sample_count, second_profile.sample_count)
        planner = gen3_randbat_belief_start_override_planner(_source(), team_size=3, world_sample_cap=4)
        first_source = planner(first_context, OpponentActionScenario(actions={"p1": 0}), 0, random.Random(7))
        second_source = planner(second_context, OpponentActionScenario(actions={"p1": 0}), 0, random.Random(7))
        self.assertEqual(first_source(), second_source())

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
        self.assertEqual(override.observation_format_id, "gen3randombattle")
        self.assertEqual(set(override.player_teams), {"p1", "p2"})
        self.assertIn("Charizard", override.player_teams["p2"])
        self.assertIn("Blaze", override.player_teams["p2"])
        self.assertIn("Xatu", override.player_teams["p1"])
        self.assertIn("Synchronize", override.player_teams["p1"])
        self.assertEqual(len(override.player_teams["p1"].split("]")), 3)
        self.assertEqual(override.player_teams["p1"].count("Xatu"), 1)
        self.assertIn("81,,85,85,85,85", override.player_teams["p2"])
        self.assertIn(",2,,30,,", override.player_teams["p2"])

    def test_observed_self_stat_mismatch_disables_override(self) -> None:
        metadata = _metadata()
        metadata["self_team"][0]["stats"] = {
            "hp": 999,
            "atk": 139,
            "def": 169,
            "spa": 217,
            "spd": 180,
            "spe": 204,
        }

        override = gen3_randbat_belief_start_override(
            context=_context(metadata),
            set_source=_source(),
            rng=random.Random(7),
            team_size=3,
        )

        self.assertIsNone(override)

    def test_planner_returns_memoized_source_for_supported_format(self) -> None:
        planner = gen3_randbat_belief_start_override_planner(_source(), team_size=3)
        context = _context(_metadata())

        self.assertTrue(getattr(planner, "scenario_independent"))
        source = planner(
            context,
            OpponentActionScenario(actions={"p1": 0}),
            0,
            random.Random(3),
        )

        self.assertTrue(callable(source))
        assert callable(source)
        first = source()
        second = source()
        self.assertIsNotNone(first)
        self.assertIs(first, second)

    def test_planner_exposes_public_context_world_count_and_checksum(self) -> None:
        metadata = _metadata()
        opponent = metadata["belief_view"]["opponent_pokemon"][0]  # type: ignore[index]
        opponent["candidate_variants"].append(  # type: ignore[index]
            {
                "variant_id": "xatu-setup",
                "source_set_id": "xatu-2",
                "role": "Setup",
                "level": 84,
                "moves": ["psychic", "calmmind", "rest", "hiddenpowerfire"],
                "ability": "Early Bird",
                "item": "Leftovers",
            }
        )
        planner = gen3_randbat_belief_start_override_planner(_source(), team_size=3, world_sample_cap=4)
        context = _context(metadata)

        self.assertEqual(planner.sample_count_for_context(context), 4)  # type: ignore[attr-defined]
        sampling_metadata = planner.sampling_metadata_for_context(context)  # type: ignore[attr-defined]
        self.assertEqual(sampling_metadata["root_puct_belief_world_sample_cap"], 4)
        self.assertEqual(sampling_metadata["root_puct_belief_world_sample_count"], 4)
        self.assertTrue(sampling_metadata["root_puct_belief_public_checksum"])

    def test_planner_returns_reason_bearing_missing_source_for_supported_failure(self) -> None:
        planner = gen3_randbat_belief_start_override_planner(_source(), team_size=3)
        metadata = _metadata()
        metadata.pop("self_team")

        source = planner(
            _context(metadata),
            OpponentActionScenario(actions={"p1": 0}),
            0,
            random.Random(3),
        )

        self.assertTrue(callable(source))
        assert callable(source)
        with self.assertRaisesRegex(ValueError, "self_team"):
            source()

    def test_unown_cosmetic_form_belief_materializes_from_base_universe(self) -> None:
        metadata = _metadata()
        metadata["belief_view"] = {
            "self_slot": "p2",
            "opponent_slot": "p1",
            "self_pokemon": [],
            "opponent_pokemon": [
                {
                    "showdown_slot": "p1",
                    "species": "Unown-Z",
                    "condition": "100/100",
                    "active": True,
                    "revealed_moves": ["Hidden Power"],
                },
            ],
        }

        override = gen3_randbat_belief_start_override(
            context=_context(metadata),
            set_source=_source(),
            rng=random.Random(7),
            team_size=3,
        )

        self.assertIsNotNone(override)
        assert override is not None
        self.assertIn("Unown-Z", override.player_teams["p1"])
        self.assertIn("Levitate", override.player_teams["p1"])
        self.assertEqual(override.player_teams["p1"].count("Unown-Z"), 1)

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

    def test_gen3_spread_reconstructs_fixed_damage_and_typed_hidden_power_stats(self) -> None:
        set_source = Gen3RandbatSource.from_data(
            {},
            move_metadata=MOVE_METADATA,
            species_metadata=SPECIES_METADATA,
        )

        registeel_spread = _gen3_randbat_fixture_spread(
            {
                "stats": {"atk": 122, "def": 279, "spa": 162, "spd": 279, "spe": 123, "hp": 253},
            },
            species="Registeel",
            moves=("sleeptalk", "rest", "seismictoss", "toxic"),
            item="Leftovers",
            level=78,
            set_source=set_source,
        )
        umbreon_spread = _gen3_randbat_fixture_spread(
            {
                "stats": {"atk": 165, "def": 244, "spa": 155, "spd": 278, "spe": 165, "hp": 310},
            },
            species="Umbreon",
            moves=("wish", "protect", "toxic", "hiddenpowerground"),
            item="Leftovers",
            level=88,
            set_source=set_source,
        )

        self.assertIsNotNone(registeel_spread)
        self.assertIsNotNone(umbreon_spread)
        assert registeel_spread is not None
        assert umbreon_spread is not None
        self.assertEqual(registeel_spread["evs"]["atk"], 0)
        self.assertEqual(registeel_spread["ivs"]["atk"], 0)
        self.assertEqual(umbreon_spread["ivs"]["spa"], 30)
        self.assertEqual(umbreon_spread["ivs"]["spd"], 30)
        self.assertEqual(umbreon_spread["ivs"]["atk"], 31)

    def test_revealed_opponent_absolute_hp_filters_sampled_variants(self) -> None:
        metadata = {
            "self_team": [
                {
                    "showdown_slot": "p2",
                    "species": "Blissey",
                    "details": "Blissey, L75",
                    "moves": ["seismictoss", "softboiled", "toxic", "thunderwave"],
                    "ability": "Natural Cure",
                    "item": "Leftovers",
                }
            ],
            "belief_view": {
                "self_slot": "p2",
                "opponent_slot": "p1",
                "self_pokemon": [],
                "opponent_pokemon": [
                    {
                        "showdown_slot": "p1",
                        "species": "Charizard",
                        "active": True,
                        "condition": "252/252",
                        "candidate_variants": [
                            {
                                "variant_id": "charizard-matching-hp",
                                "source_set_id": "charizard-1",
                                "role": "Berry Sweeper",
                                "level": 79,
                                "moves": ["fireblast", "dragonclaw", "hiddenpowergrass", "substitute"],
                                "ability": "Blaze",
                                "item": "Petaya Berry",
                            },
                            {
                                "variant_id": "charizard-wrong-hp",
                                "source_set_id": "charizard-2",
                                "role": "Support",
                                "level": 79,
                                "moves": ["fireblast", "dragonclaw", "hiddenpowergrass", "rest"],
                                "ability": "Blaze",
                                "item": "Leftovers",
                            },
                        ],
                    }
                ],
            },
        }

        context = _context(metadata)

        override = gen3_randbat_belief_start_override(
            context=context,
            set_source=_source(),
            rng=random.Random(1),
            team_size=1,
        )

        self.assertIsNotNone(override)
        assert override is not None
        self.assertIn("PetayaBerry", override.player_teams["p1"])
        self.assertIn("substitute", override.player_teams["p1"].lower())
        self.assertNotIn("Leftovers", override.player_teams["p1"])

    def test_revealed_opponent_percentage_condition_does_not_filter_by_absolute_hp(self) -> None:
        metadata = _metadata()
        opponent = metadata["belief_view"]["opponent_pokemon"][0]
        opponent["condition"] = "70/100"

        override = gen3_randbat_belief_start_override(
            context=_context(metadata),
            set_source=_source(),
            rng=random.Random(7),
            team_size=3,
        )

        self.assertIsNotNone(override)

    def test_opponent_private_trajectory_moves_do_not_constrain_sampled_variants(self) -> None:
        metadata = _metadata()
        context = _context(metadata)
        opponent_observation = replace(
            context.observation,
            metadata={
                "self_active": {
                    "species": "Xatu",
                    "moves": ["impossiblemove", "psychic", "wish", "protect"],
                }
            },
        )
        context.trajectory.append(
            TrajectoryStep(
                player_id="p1",
                turn_index=0,
                observation=opponent_observation,
                legal_action_mask=tuple(context.observation.legal_action_mask),
                action_index=0,
            )
        )

        override = gen3_randbat_belief_start_override(
            context=context,
            set_source=_source(),
            rng=random.Random(7),
            team_size=3,
        )

        self.assertIsNotNone(override)
        assert override is not None
        self.assertIn("Xatu", override.player_teams["p1"])

    def test_persisted_public_moves_constrain_replay_move_slots_without_private_moves(self) -> None:
        metadata = {
            "self_team": [
                {
                    "showdown_slot": "p2",
                    "species": "Blissey",
                    "details": "Blissey, L75",
                    "moves": ["seismictoss", "softboiled", "toxic", "thunderwave"],
                    "ability": "Natural Cure",
                    "item": "Leftovers",
                }
            ],
            "belief_view": {
                "self_slot": "p2",
                "opponent_slot": "p1",
                "self_pokemon": [],
                "opponent_pokemon": [
                    {
                        "showdown_slot": "p1",
                        "species": "Charizard",
                        "active": True,
                        "candidate_variants": [
                            {
                                "variant_id": "charizard-public-move",
                                "source_set_id": "charizard-1",
                                "role": "Berry Sweeper",
                                "level": 79,
                                "moves": ["fireblast", "dragonclaw", "hiddenpowergrass", "substitute"],
                                "ability": "Blaze",
                                "item": "Petaya Berry",
                            },
                        ],
                    }
                ],
            },
        }
        context = _context(metadata)
        before_observation = replace(
            context.observation,
            metadata={
                **metadata,
                "opponent_active": {"species": "Charizard"},
                "recent_public_events": [],
            },
        )
        context.trajectory.append(
            TrajectoryStep(
                player_id="p2",
                turn_index=0,
                observation=before_observation,
                legal_action_mask=tuple(before_observation.legal_action_mask),
                action_index=0,
            )
        )
        context.trajectory.append(
            TrajectoryStep(
                player_id="p1",
                turn_index=0,
                observation=before_observation,
                legal_action_mask=tuple(before_observation.legal_action_mask),
                action_index=0,
            )
        )
        current_observation = replace(
            context.observation,
            metadata={
                **metadata,
                "opponent_active": {"species": "Charizard"},
                "recent_public_events": [],
            },
        )
        context = replace(context, decision_round_index=1, observation=current_observation)
        context.trajectory.metadata = {
            "public_resolved_action_rounds": [
                PublicResolvedActionRound(
                    turn_index=0,
                    actions={
                        "p1": PublicActionIdentifier(kind="move", move_id="hiddenpowergrass"),
                        "p2": PublicActionIdentifier(
                            kind="event",
                            event_id="unresolved-public-event",
                        ),
                    },
                ).to_dict()
            ]
        }

        override = gen3_randbat_belief_start_override(
            context=context,
            set_source=_source(),
            rng=random.Random(7),
            team_size=1,
        )

        self.assertIsNotNone(override)
        assert override is not None
        moves_field = override.player_teams["p1"].split("|")[4]
        self.assertTrue(moves_field.startswith("hiddenpowergrass,"))

    def test_bare_public_hidden_power_constrains_typed_replay_move_slot(self) -> None:
        metadata = {
            "self_team": [
                {
                    "showdown_slot": "p2",
                    "species": "Blissey",
                    "details": "Blissey, L75",
                    "moves": ["seismictoss", "softboiled", "toxic", "thunderwave"],
                    "ability": "Natural Cure",
                    "item": "Leftovers",
                }
            ],
            "belief_view": {
                "self_slot": "p2",
                "opponent_slot": "p1",
                "self_pokemon": [],
                "opponent_pokemon": [
                    {
                        "showdown_slot": "p1",
                        "species": "Charizard",
                        "active": True,
                        "candidate_variants": [
                            {
                                "variant_id": "charizard-public-hidden-power",
                                "source_set_id": "charizard-1",
                                "role": "Berry Sweeper",
                                "level": 79,
                                "moves": ["fireblast", "dragonclaw", "hiddenpowergrass", "substitute"],
                                "ability": "Blaze",
                                "item": "Petaya Berry",
                            },
                        ],
                    }
                ],
            },
        }
        context = _context(metadata)
        before_observation = replace(
            context.observation,
            metadata={
                **metadata,
                "opponent_active": {"species": "Charizard"},
                "recent_public_events": [],
            },
        )
        context.trajectory.append(
            TrajectoryStep(
                player_id="p2",
                turn_index=0,
                observation=before_observation,
                legal_action_mask=tuple(before_observation.legal_action_mask),
                action_index=0,
            )
        )
        context.trajectory.append(
            TrajectoryStep(
                player_id="p1",
                turn_index=0,
                observation=before_observation,
                legal_action_mask=tuple(before_observation.legal_action_mask),
                action_index=0,
            )
        )
        current_observation = replace(
            context.observation,
            metadata={
                **metadata,
                "opponent_active": {"species": "Charizard"},
                "recent_public_events": [
                    "|move|opponenta: Charizard|Hidden Power|selfa: Blissey",
                ],
            },
        )
        context = replace(context, decision_round_index=1, observation=current_observation)

        override = gen3_randbat_belief_start_override(
            context=context,
            set_source=_source(),
            rng=random.Random(7),
            team_size=1,
        )

        self.assertIsNotNone(override)
        assert override is not None
        moves_field = override.player_teams["p1"].split("|")[4]
        self.assertTrue(moves_field.startswith("hiddenpowergrass,"))

    def test_consecutive_public_moves_use_next_round_not_oldest_rolling_event(self) -> None:
        metadata = {
            "self_team": [
                {
                    "showdown_slot": "p2",
                    "species": "Blissey",
                    "details": "Blissey, L75",
                    "moves": ["seismictoss", "softboiled", "toxic", "thunderwave"],
                    "ability": "Natural Cure",
                    "item": "Leftovers",
                }
            ],
            "belief_view": {
                "self_slot": "p2",
                "opponent_slot": "p1",
                "self_pokemon": [],
                "opponent_pokemon": [
                    {
                        "showdown_slot": "p1",
                        "species": "Charizard",
                        "active": True,
                        "candidate_variants": [
                            {
                                "variant_id": "charizard-two-public-moves",
                                "source_set_id": "charizard-1",
                                "role": "Berry Sweeper",
                                "level": 79,
                                "moves": ["substitute", "fireblast", "hiddenpowergrass", "dragonclaw"],
                                "ability": "Blaze",
                                "item": "Petaya Berry",
                            },
                        ],
                    }
                ],
            },
        }
        context = _context(metadata)
        move_mask = tuple(index < 4 for index in range(ACTION_COUNT))
        round_0_observation = replace(
            context.observation,
            legal_action_mask=move_mask,
            metadata={
                **metadata,
                "opponent_active": {"species": "Charizard"},
                "recent_public_events": [],
            },
        )
        round_1_observation = replace(
            context.observation,
            legal_action_mask=move_mask,
            metadata={
                **metadata,
                "opponent_active": {"species": "Charizard"},
                "recent_public_events": [
                    "|move|opponenta: Charizard|Fire Blast|selfa: Blissey",
                ],
            },
        )
        round_2_observation = replace(
            context.observation,
            legal_action_mask=move_mask,
            metadata={
                **metadata,
                "opponent_active": {"species": "Charizard"},
                "recent_public_events": [
                    "|move|opponenta: Charizard|Fire Blast|selfa: Blissey",
                    "|-damage|selfa: Blissey|70/100",
                    "|move|opponenta: Charizard|Dragon Claw|selfa: Blissey",
                ],
            },
        )
        for turn_index, observation, opponent_action in (
            (0, round_0_observation, 1),
            (1, round_1_observation, 3),
        ):
            context.trajectory.append(
                TrajectoryStep(
                    player_id="p2",
                    turn_index=turn_index,
                    observation=observation,
                    legal_action_mask=tuple(observation.legal_action_mask),
                    action_index=0,
                )
            )
            context.trajectory.append(
                TrajectoryStep(
                    player_id="p1",
                    turn_index=turn_index,
                    observation=observation,
                    legal_action_mask=tuple(observation.legal_action_mask),
                    action_index=opponent_action,
                )
            )
        context = replace(context, decision_round_index=2, observation=round_2_observation)

        override = gen3_randbat_belief_start_override(
            context=context,
            set_source=_source(),
            rng=random.Random(7),
            team_size=1,
        )

        self.assertIsNotNone(override)
        assert override is not None
        moves_field = override.player_teams["p1"].split("|")[4]
        self.assertTrue(moves_field.startswith("substitute,fireblast,hiddenpowergrass,dragonclaw"))

    def test_public_switch_events_constrain_replay_switch_team_slots(self) -> None:
        metadata = {
            "self_team": [
                {
                    "showdown_slot": "p2",
                    "species": "Charizard",
                    "details": "Charizard, L79",
                    "moves": ["fireblast", "dragonclaw", "hiddenpowergrass", "substitute"],
                    "ability": "Blaze",
                    "item": "Petaya Berry",
                    "stats": {"hp": 252, "atk": 139, "def": 169, "spa": 217, "spd": 180, "spe": 204},
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
                        "active": False,
                        "candidate_variants": [
                            {
                                "variant_id": "xatu-support",
                                "source_set_id": "xatu-1",
                                "role": "Support",
                                "level": 84,
                                "moves": ["psychic", "thunderwave", "wish", "protect"],
                                "ability": "Synchronize",
                                "item": "Leftovers",
                            },
                        ],
                    },
                    {
                        "showdown_slot": "p1",
                        "species": "Tauros",
                        "active": False,
                        "candidate_variants": [
                            {
                                "variant_id": "tauros-breaker",
                                "source_set_id": "tauros-1",
                                "role": "Breaker",
                                "level": 76,
                                "moves": ["return", "earthquake", "doubleedge", "hiddenpowerghost"],
                                "ability": "Intimidate",
                                "item": "Choice Band",
                            },
                        ],
                    },
                    {
                        "showdown_slot": "p1",
                        "species": "Arcanine",
                        "active": True,
                        "candidate_variants": [
                            {
                                "variant_id": "arcanine-breaker",
                                "source_set_id": "arcanine-1",
                                "role": "Breaker",
                                "level": 78,
                                "moves": ["fireblast", "crunch", "extremespeed", "hiddenpowergrass"],
                                "ability": "Flash Fire",
                                "item": "Leftovers",
                            },
                        ],
                    },
                ],
            },
        }
        context = _context(metadata)
        switch_mask = tuple(index in {0, 4} for index in range(ACTION_COUNT))
        round_0_observation = replace(
            context.observation,
            legal_action_mask=switch_mask,
            metadata={
                **metadata,
                "opponent_active": {"species": "Xatu"},
                "recent_public_events": [],
            },
        )
        round_1_observation = replace(
            context.observation,
            metadata={
                **metadata,
                "opponent_active": {"species": "Arcanine"},
                "recent_public_events": [
                    "|switch|opponenta: Arcanine|Arcanine, L78|100/100",
                ],
            },
        )
        context.trajectory.append(
            TrajectoryStep(
                player_id="p2",
                turn_index=0,
                observation=round_0_observation,
                legal_action_mask=tuple(round_0_observation.legal_action_mask),
                action_index=0,
            )
        )
        context.trajectory.append(
            TrajectoryStep(
                player_id="p1",
                turn_index=0,
                observation=round_0_observation,
                legal_action_mask=tuple(round_0_observation.legal_action_mask),
                # With Xatu active at team index 0, action 4 targets team index 1.
                action_index=4,
            )
        )
        context = replace(context, decision_round_index=1, observation=round_1_observation)

        override = gen3_randbat_belief_start_override(
            context=context,
            set_source=_source(),
            rng=random.Random(7),
            team_size=3,
        )

        self.assertIsNotNone(override)
        assert override is not None
        opponent_species_order = [packed.split("|", 1)[0] for packed in override.player_teams["p1"].split("]")]
        self.assertEqual(opponent_species_order, ["Xatu", "Arcanine", "Tauros"])

    def test_public_switch_constraints_force_missing_revealed_species_into_hidden_backline(self) -> None:
        metadata = _metadata()
        metadata["self_team"] = metadata["self_team"][:2]
        metadata["belief_view"] = {
            "self_slot": "p2",
            "opponent_slot": "p1",
            "self_pokemon": [],
            "opponent_pokemon": [
                {
                    "showdown_slot": "p1",
                    "species": "Arcanine",
                    "active": True,
                    "candidate_variants": [
                        {
                            "variant_id": "arcanine-breaker",
                            "source_set_id": "arcanine-1",
                            "role": "Breaker",
                            "level": 78,
                            "moves": ["fireblast", "crunch", "extremespeed", "hiddenpowergrass"],
                            "ability": "Flash Fire",
                            "item": "Leftovers",
                        },
                    ],
                },
            ],
        }
        context = _context(metadata)
        switch_mask = tuple(index in {0, 4} for index in range(ACTION_COUNT))
        round_0_observation = replace(
            context.observation,
            legal_action_mask=switch_mask,
            metadata={
                **metadata,
                "opponent_active": {"species": "Xatu"},
                "recent_public_events": [],
            },
        )
        round_1_observation = replace(
            context.observation,
            legal_action_mask=switch_mask,
            metadata={
                **metadata,
                "opponent_active": {"species": "Arcanine"},
                "recent_public_events": [
                    "|switch|opponenta: Arcanine|Arcanine, L78|100/100",
                ],
            },
        )
        context.trajectory.append(
            TrajectoryStep(
                player_id="p2",
                turn_index=0,
                observation=round_0_observation,
                legal_action_mask=tuple(round_0_observation.legal_action_mask),
                action_index=0,
            )
        )
        context.trajectory.append(
            TrajectoryStep(
                player_id="p1",
                turn_index=0,
                observation=round_0_observation,
                legal_action_mask=tuple(round_0_observation.legal_action_mask),
                action_index=4,
            )
        )
        context = replace(context, decision_round_index=1, observation=round_1_observation)

        override = gen3_randbat_belief_start_override(
            context=context,
            set_source=_source(),
            rng=random.Random(7),
            team_size=2,
        )

        self.assertIsNotNone(override)
        assert override is not None
        opponent_species_order = [packed.split("|", 1)[0] for packed in override.player_teams["p1"].split("]")]
        self.assertEqual(opponent_species_order, ["Xatu", "Arcanine"])

    def test_forced_hidden_backline_species_respects_public_move_slot_constraints(self) -> None:
        metadata = _metadata()
        metadata["self_team"] = metadata["self_team"][:2]
        metadata["belief_view"] = {
            "self_slot": "p2",
            "opponent_slot": "p1",
            "self_pokemon": [],
            "opponent_pokemon": [
                {
                    "showdown_slot": "p1",
                    "species": "Arcanine",
                    "active": True,
                    "candidate_variants": [
                        {
                            "variant_id": "arcanine-breaker",
                            "source_set_id": "arcanine-1",
                            "role": "Breaker",
                            "level": 78,
                            "moves": ["fireblast", "crunch", "extremespeed", "hiddenpowergrass"],
                            "ability": "Flash Fire",
                            "item": "Leftovers",
                        },
                    ],
                },
            ],
        }
        context = _context(metadata)
        action_mask = tuple(index in {0, 1, 4} for index in range(ACTION_COUNT))
        round_0_observation = replace(
            context.observation,
            legal_action_mask=action_mask,
            metadata={
                **metadata,
                "opponent_active": {"species": "Xatu"},
                "recent_public_events": [],
            },
        )
        round_1_observation = replace(
            context.observation,
            legal_action_mask=action_mask,
            metadata={
                **metadata,
                "opponent_active": {"species": "Xatu"},
                "recent_public_events": [
                    "|move|opponenta: Xatu|Thunder Wave|p2a: Charizard",
                ],
            },
        )
        round_2_observation = replace(
            context.observation,
            legal_action_mask=action_mask,
            metadata={
                **metadata,
                "opponent_active": {"species": "Arcanine"},
                "recent_public_events": [
                    "|switch|opponenta: Arcanine|Arcanine, L78|100/100",
                ],
            },
        )
        for player_id, turn_index, action_index, observation in (
            ("p2", 0, 0, round_0_observation),
            ("p1", 0, 1, round_0_observation),
            ("p2", 1, 0, round_1_observation),
            ("p1", 1, 4, round_1_observation),
        ):
            context.trajectory.append(
                TrajectoryStep(
                    player_id=player_id,
                    turn_index=turn_index,
                    observation=observation,
                    legal_action_mask=tuple(observation.legal_action_mask),
                    action_index=action_index,
                )
            )
        context = replace(context, decision_round_index=2, observation=round_2_observation)

        override = gen3_randbat_belief_start_override(
            context=context,
            set_source=_source(),
            rng=random.Random(7),
            team_size=2,
        )

        self.assertIsNotNone(override)
        assert override is not None
        xatu = override.player_teams["p1"].split("]")[0]
        packed_parts = xatu.split("|")
        self.assertEqual(packed_parts[0], "Xatu")
        self.assertEqual(packed_parts[3], "Synchronize")
        self.assertEqual(packed_parts[4].split(",")[1], "thunderwave")

    def test_persisted_public_switch_constraints_track_showdown_party_swaps(self) -> None:
        metadata = {
            "self_team": [
                {
                    "showdown_slot": "p2",
                    "species": "Charizard",
                    "details": "Charizard, L79",
                    "moves": ["fireblast", "dragonclaw", "hiddenpowergrass", "substitute"],
                    "ability": "Blaze",
                    "item": "Petaya Berry",
                    "stats": {"hp": 252, "atk": 139, "def": 169, "spa": 217, "spd": 180, "spe": 204},
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
                        "active": False,
                        "candidate_variants": [
                            {
                                "variant_id": "xatu-support",
                                "source_set_id": "xatu-1",
                                "role": "Support",
                                "level": 84,
                                "moves": ["psychic", "thunderwave", "wish", "protect"],
                                "ability": "Synchronize",
                                "item": "Leftovers",
                            },
                        ],
                    },
                    {
                        "showdown_slot": "p1",
                        "species": "Arcanine",
                        "active": False,
                        "candidate_variants": [
                            {
                                "variant_id": "arcanine-breaker",
                                "source_set_id": "arcanine-1",
                                "role": "Breaker",
                                "level": 78,
                                "moves": ["fireblast", "crunch", "extremespeed", "hiddenpowergrass"],
                                "ability": "Flash Fire",
                                "item": "Leftovers",
                            },
                        ],
                    },
                    {
                        "showdown_slot": "p1",
                        "species": "Tauros",
                        "active": True,
                        "candidate_variants": [
                            {
                                "variant_id": "tauros-breaker",
                                "source_set_id": "tauros-1",
                                "role": "Breaker",
                                "level": 76,
                                "moves": ["return", "earthquake", "doubleedge", "hiddenpowerghost"],
                                "ability": "Intimidate",
                                "item": "Choice Band",
                            },
                        ],
                    },
                ],
            },
        }
        context = _context(metadata)
        switch_mask = tuple(index in {0, 4, 5} for index in range(ACTION_COUNT))
        round_0_observation = replace(
            context.observation,
            legal_action_mask=switch_mask,
            metadata={
                **metadata,
                "opponent_active": {"species": "Xatu"},
                "recent_public_events": [],
            },
        )
        round_1_observation = replace(
            context.observation,
            legal_action_mask=switch_mask,
            metadata={
                **metadata,
                "opponent_active": {"species": "Tauros"},
                "recent_public_events": [],
            },
        )
        round_2_observation = replace(
            context.observation,
            legal_action_mask=switch_mask,
            metadata={
                **metadata,
                "opponent_active": {"species": "Arcanine"},
                "recent_public_events": [],
            },
        )
        for turn_index, observation, opponent_action in (
            # With Xatu active at current position 0, action 5 targets current position 2.
            (0, round_0_observation, 5),
            # Showdown swaps Tauros into current position 0, so action 4 now targets current
            # position 1, which corresponds to Arcanine's initial party index.
            (1, round_1_observation, 4),
        ):
            context.trajectory.append(
                TrajectoryStep(
                    player_id="p2",
                    turn_index=turn_index,
                    observation=observation,
                    legal_action_mask=tuple(observation.legal_action_mask),
                    action_index=0,
                )
            )
            context.trajectory.append(
                TrajectoryStep(
                    player_id="p1",
                    turn_index=turn_index,
                    observation=observation,
                    legal_action_mask=tuple(observation.legal_action_mask),
                    action_index=opponent_action,
                )
            )
        context.trajectory.metadata = {
            "public_resolved_action_rounds": [
                PublicResolvedActionRound(
                    turn_index=0,
                    actions={
                        "p1": PublicActionIdentifier(kind="switch", switched_species="tauros"),
                        "p2": PublicActionIdentifier(
                            kind="event",
                            event_id="unresolved-public-event",
                        ),
                    },
                ).to_dict(),
                PublicResolvedActionRound(
                    turn_index=1,
                    actions={
                        "p1": PublicActionIdentifier(kind="switch", switched_species="arcanine"),
                        "p2": PublicActionIdentifier(
                            kind="event",
                            event_id="unresolved-public-event",
                        ),
                    },
                ).to_dict(),
            ]
        }
        context = replace(context, decision_round_index=2, observation=round_2_observation)

        override = gen3_randbat_belief_start_override(
            context=context,
            set_source=_source(),
            rng=random.Random(7),
            team_size=3,
        )

        self.assertIsNotNone(override)
        assert override is not None
        opponent_species_order = [packed.split("|", 1)[0] for packed in override.player_teams["p1"].split("]")]
        self.assertEqual(opponent_species_order, ["Xatu", "Arcanine", "Tauros"])

    def test_conflicting_public_switch_team_slot_constraints_disable_override(self) -> None:
        metadata = _metadata()
        context = _context(metadata)
        switch_mask = tuple(index in {0, 4} for index in range(ACTION_COUNT))
        round_0_observation = replace(
            context.observation,
            legal_action_mask=switch_mask,
            metadata={
                **metadata,
                "opponent_active": {"species": "Xatu"},
                "recent_public_events": [],
            },
        )
        round_1_observation = replace(
            context.observation,
            legal_action_mask=switch_mask,
            metadata={
                **metadata,
                "opponent_active": {"species": "Xatu"},
                "recent_public_events": [
                    "|switch|opponenta: Xatu|Xatu, L84|100/100",
                ],
            },
        )
        context.trajectory.append(
            TrajectoryStep(
                player_id="p2",
                turn_index=0,
                observation=round_0_observation,
                legal_action_mask=tuple(round_0_observation.legal_action_mask),
                action_index=0,
            )
        )
        context.trajectory.append(
            TrajectoryStep(
                player_id="p1",
                turn_index=0,
                observation=round_0_observation,
                legal_action_mask=tuple(round_0_observation.legal_action_mask),
                # This switch action targets team index 1, conflicting with Xatu's initial index 0.
                action_index=4,
            )
        )
        context = replace(context, decision_round_index=1, observation=round_1_observation)

        override = gen3_randbat_belief_start_override(
            context=context,
            set_source=_source(),
            rng=random.Random(7),
            team_size=3,
        )

        self.assertIsNone(override)

    def test_called_public_move_events_do_not_constrain_replay_move_slots(self) -> None:
        metadata = _metadata()
        context = _context(metadata)
        before_observation = replace(
            context.observation,
            metadata={
                **metadata,
                "opponent_active": {"species": "Xatu"},
                "recent_public_events": [],
            },
        )
        context.trajectory.append(
            TrajectoryStep(
                player_id="p2",
                turn_index=0,
                observation=before_observation,
                legal_action_mask=tuple(before_observation.legal_action_mask),
                action_index=0,
            )
        )
        context.trajectory.append(
            TrajectoryStep(
                player_id="p1",
                turn_index=0,
                observation=before_observation,
                legal_action_mask=tuple(before_observation.legal_action_mask),
                action_index=0,
            )
        )
        current_observation = replace(
            context.observation,
            metadata={
                **metadata,
                "opponent_active": {"species": "Xatu"},
                "recent_public_events": [
                    "|move|opponenta: Xatu|Ice Beam|selfa: Charizard|[from] Sleep Talk",
                ],
            },
        )
        context = replace(context, decision_round_index=1, observation=current_observation)

        override = gen3_randbat_belief_start_override(
            context=context,
            set_source=_source(),
            rng=random.Random(7),
            team_size=3,
        )

        self.assertIsNotNone(override)

    def test_replay_root_uses_initial_self_team_snapshot(self) -> None:
        initial_metadata = _metadata()
        current_metadata = _metadata()
        current_metadata["self_team"][0]["item"] = "Leftovers"
        context = _context(current_metadata)
        initial_context = _context(initial_metadata)
        context.trajectory.append(
            TrajectoryStep(
                player_id="p2",
                turn_index=0,
                observation=initial_context.observation,
                legal_action_mask=tuple(initial_context.observation.legal_action_mask),
                action_index=0,
            )
        )

        override = gen3_randbat_belief_start_override(
            context=context,
            set_source=_source(),
            rng=random.Random(7),
            team_size=3,
        )

        self.assertIsNotNone(override)
        assert override is not None
        self.assertIn("PetayaBerry", override.player_teams["p2"])


@unittest.skipIf(integration_config() is None, "requires node and built Pokemon Showdown checkout")
class Gen3RandbatBeliefStartOverrideIntegrationTest(unittest.TestCase):
    def test_turn_zero_belief_override_handles_real_stat_regression_seeds(self) -> None:
        config = integration_config()
        assert config is not None
        set_source = Gen3RandbatSource.from_showdown_root(config.showdown_root, use_cache=False)
        for seed in (910000, 910001):
            with self.subTest(seed=seed):
                with LocalShowdownEnv(config) as env:
                    env.reset(seed=seed, format_id="gen3randombattle")
                    observation = env.observe("p1")
                opponent_active = observation.metadata.get("opponent_active")
                self.assertIsInstance(opponent_active, dict)
                metadata = {
                    **dict(observation.metadata),
                    "belief_view": {
                        "self_slot": "p1",
                        "opponent_slot": "p2",
                        "self_pokemon": [],
                        "opponent_pokemon": [
                            {
                                "showdown_slot": "p2",
                                "species": str(opponent_active["species"]),
                                "active": True,
                                "revealed_moves": [],
                            }
                        ],
                    },
                }
                context = PolicyContext(
                    player_id="p1",
                    decision_round_index=0,
                    battle_id=f"belief-start-real-{seed}",
                    format_id="gen3randombattle",
                    seed=seed,
                    observation=replace(observation, metadata=metadata),
                    requested_players=("p1", "p2"),
                    trajectory=BattleTrajectory(
                        battle_id=f"belief-start-real-{seed}",
                        format_id="gen3randombattle",
                        seed=seed,
                    ),
                    requested_legal_action_masks={"p1": observation.legal_action_mask},
                )

                override = gen3_randbat_belief_start_override(
                    context=context,
                    set_source=set_source,
                    rng=random.Random(7),
                )

                self.assertIsNotNone(override)

    def test_turn_zero_belief_override_reproduces_real_self_observation(self) -> None:
        config = integration_config()
        assert config is not None
        set_source = Gen3RandbatSource.from_showdown_root(config.showdown_root)
        seed = 731
        with LocalShowdownEnv(config) as env:
            env.reset(seed=seed, format_id="gen3randombattle")
            observation = env.observe("p1")
            requested_players = env.requested_players()
            requested_legal_action_masks = {
                player: env.legal_actions(player)
                for player in requested_players
            }
        opponent_active = observation.metadata.get("opponent_active")
        self.assertIsInstance(opponent_active, dict)
        opponent_species = str(opponent_active["species"])
        metadata = {
            **dict(observation.metadata),
            "belief_view": {
                "self_slot": "p1",
                "opponent_slot": "p2",
                "self_pokemon": [],
                "opponent_pokemon": [
                    {
                        "showdown_slot": "p2",
                        "species": opponent_species,
                        "active": True,
                        "revealed_moves": [],
                    }
                ],
            },
        }
        search_observation = replace(observation, metadata=metadata)
        context = PolicyContext(
            player_id="p1",
            decision_round_index=0,
            battle_id="belief-start-real",
            format_id="gen3randombattle",
            seed=seed,
            observation=search_observation,
            requested_players=requested_players,
            trajectory=BattleTrajectory(battle_id="belief-start-real", format_id="gen3randombattle", seed=seed),
            requested_legal_action_masks=requested_legal_action_masks,
        )

        override = gen3_randbat_belief_start_override(
            context=context,
            set_source=set_source,
            rng=random.Random(7),
        )

        self.assertIsNotNone(override)
        assert override is not None
        branch_actions = {
            player: next(index for index, legal in enumerate(mask) if legal)
            for player, mask in requested_legal_action_masks.items()
        }
        with LocalShowdownEnv(config) as branch_env:
            result = replay_trajectory_branch(
                branch_env,
                context.trajectory,
                prefix_decision_round_count=0,
                branch_actions=branch_actions,
                start_override=override,
                consistency_player_id="p1",
                expected_current_observation=observation,
            )

        self.assertEqual(result.prefix.replayed_round_count, 0)


if __name__ == "__main__":
    unittest.main()
