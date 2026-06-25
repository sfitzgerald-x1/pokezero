from pathlib import Path
import unittest

from pokezero.actions import ACTION_COUNT
from pokezero.belief import CandidateSetSummary
from pokezero.observation import (
    ACTION_CANDIDATE_TOKEN_COUNT,
    FIELD_TOKEN_COUNT,
    OPPONENT_POKEMON_TOKEN_COUNT,
    SELF_POKEMON_TOKEN_COUNT,
)
from pokezero.showdown import (
    DEFAULT_REPLAY_OBSERVATION_SPEC,
    PlayerRelativePublicEvent,
    _event_detail_category,
    detect_showdown_slot,
    normalize_for_player,
    observation_from_player_state,
    parse_showdown_replay,
    showdown_choice_for_action,
    showdown_submission_for_action,
    stable_category_id,
)


class EventDetailCategoryTest(unittest.TestCase):
    def _event(self, event_type: str, primary: str) -> PlayerRelativePublicEvent:
        return PlayerRelativePublicEvent(event_type=event_type, raw_line="", primary=primary)

    def test_enumerable_details_emit_in_vocab_tokens(self) -> None:
        self.assertEqual(_event_detail_category(self._event("move", "Flamethrower")), "move:Flamethrower")
        self.assertEqual(_event_detail_category(self._event("switch", "Snorlax")), "species:Snorlax")
        self.assertEqual(_event_detail_category(self._event("-status", "par")), "status:par")

    def test_unactionable_details_are_dropped(self) -> None:
        # HP strings, usernames, winner identity, and free-form payloads -> None (padding slot).
        self.assertIsNone(_event_detail_category(self._event("-damage", "70/100")))
        self.assertIsNone(_event_detail_category(self._event("-heal", "200/267 tox")))
        self.assertIsNone(_event_detail_category(self._event("player", "SomeUsername")))
        self.assertIsNone(_event_detail_category(self._event("win", "SomeUsername")))
        self.assertIsNone(_event_detail_category(self._event("-weather", "Sandstorm")))


FIXTURE_ROOT = Path(__file__).parent / "fixtures" / "showdown"


def fixture_lines(name: str) -> list[str]:
    return (FIXTURE_ROOT / name).read_text(encoding="utf-8").splitlines()


class FakeSetSource:
    def summarize(self, *, format_id, species, revealed_moves, **kwargs):
        if species == "Xatu":
            return CandidateSetSummary(
                species=species,
                candidate_count=2,
                uncertainty=0.5,
                possible_abilities=("Early Bird", "Synchronize"),
                possible_items=("Leftovers",),
                possible_moves=("psychic", "thunderwave", "wish"),
            )
        return CandidateSetSummary(
            species=species,
            candidate_count=1,
            uncertainty=0.25,
            possible_abilities=("Intimidate",),
            possible_items=("Lum Berry", "Leftovers"),
            possible_moves=tuple(revealed_moves),
        )


class ReorderedFakeSetSource(FakeSetSource):
    def summarize(self, *, format_id, species, revealed_moves, **kwargs):
        if species == "Xatu":
            return CandidateSetSummary(
                species=species,
                candidate_count=2,
                uncertainty=0.5,
                possible_abilities=("Synchronize", "Early Bird"),
                possible_items=("Leftovers",),
                possible_moves=("wish", "thunderwave", "psychic"),
            )
        return super().summarize(
            format_id=format_id,
            species=species,
            revealed_moves=revealed_moves,
            **kwargs,
        )


class ShowdownReplayNormalizationTest(unittest.TestCase):
    def test_detected_player_name_overrides_stale_configured_slot(self) -> None:
        replay = parse_showdown_replay(fixture_lines("p2_seat_replay.txt"), battle_id="battle-gen3randombattle-1")

        detected_slot = detect_showdown_slot(
            replay,
            player_name="PokeZeroBot",
            configured_showdown_slot="p1",
        )

        self.assertEqual(detected_slot, "p2")

    def test_p2_observation_is_player_relative_not_protocol_relative(self) -> None:
        replay = parse_showdown_replay(fixture_lines("p2_seat_replay.txt"), battle_id="battle-gen3randombattle-1")

        state = normalize_for_player(
            replay,
            player_id="agent",
            player_name="PokeZeroBot",
            configured_showdown_slot="p1",
        )

        self.assertEqual(state.perspective.showdown_slot, "p2")
        self.assertEqual(state.perspective.opponent_showdown_slot, "p1")
        self.assertEqual(state.self_active.species, "Charizard")
        self.assertEqual(state.opponent_active.species, "Xatu")
        self.assertEqual(state.request_kind, "move")
        self.assertEqual(len(state.legal_action_mask), ACTION_COUNT)
        self.assertTrue(state.legal_action_mask[0])
        self.assertTrue(state.legal_action_mask[1])
        self.assertFalse(state.legal_action_mask[2])
        self.assertTrue(state.legal_action_mask[3])
        self.assertTrue(state.legal_action_mask[4])
        self.assertFalse(state.legal_action_mask[5])

    def test_two_players_receive_mirrored_self_and_opponent_views(self) -> None:
        replay = parse_showdown_replay(fixture_lines("p2_seat_replay.txt"), battle_id="battle-gen3randombattle-1")

        p1_state = normalize_for_player(replay, player_id="human", player_name="HumanFriend")
        p2_state = normalize_for_player(replay, player_id="agent", player_name="PokeZeroBot")

        self.assertEqual(p1_state.self_active.species, "Xatu")
        self.assertEqual(p1_state.opponent_active.species, "Charizard")
        self.assertEqual(p2_state.self_active.species, "Charizard")
        self.assertEqual(p2_state.opponent_active.species, "Xatu")
        self.assertEqual(p1_state.perspective.showdown_slot, p2_state.perspective.opponent_showdown_slot)
        self.assertEqual(p2_state.perspective.showdown_slot, p1_state.perspective.opponent_showdown_slot)

    def test_hidden_opponent_request_state_is_not_exposed_in_opponent_team(self) -> None:
        replay = parse_showdown_replay(fixture_lines("p2_seat_replay.txt"), battle_id="battle-gen3randombattle-1")

        state = normalize_for_player(replay, player_id="agent", player_name="PokeZeroBot")

        opponent_species = {pokemon.species for pokemon in state.opponent_team}
        self.assertEqual(opponent_species, {"Arcanine", "Xatu"})
        self.assertNotIn("Alakazam", opponent_species)

    def test_previously_revealed_opponent_pokemon_remain_in_public_memory(self) -> None:
        replay = parse_showdown_replay(fixture_lines("p2_seat_replay.txt"), battle_id="battle-gen3randombattle-1")

        state = normalize_for_player(replay, player_id="agent", player_name="PokeZeroBot")

        self.assertEqual([pokemon.species for pokemon in state.opponent_team], ["Arcanine", "Xatu"])
        self.assertFalse(state.opponent_team[0].active)
        self.assertTrue(state.opponent_team[1].active)
        self.assertEqual(state.opponent_active.species, "Xatu")
        self.assertEqual(state.belief_view.opponent_pokemon[1].revealed_moves, ("Psychic",))
        self.assertEqual(state.belief_view.opponent_pokemon[1].condition, "70/100")

    def test_observation_shell_carries_detected_perspective_and_legal_mask(self) -> None:
        replay = parse_showdown_replay(fixture_lines("p2_seat_replay.txt"), battle_id="battle-gen3randombattle-1")
        state = normalize_for_player(replay, player_id="agent", player_name="PokeZeroBot")

        observation = observation_from_player_state(state)

        observation.validate(DEFAULT_REPLAY_OBSERVATION_SPEC)
        self.assertEqual(observation.perspective.showdown_slot, "p2")
        self.assertEqual(observation.perspective.opponent_showdown_slot, "p1")
        self.assertEqual(observation.legal_action_mask, state.legal_action_mask)
        self.assertEqual(observation.metadata["self_active"]["species"], "Charizard")
        self.assertEqual(observation.metadata["opponent_active"]["species"], "Xatu")
        self.assertEqual(observation.metadata["action_candidates"][0]["move_name"], "flamethrower")
        self.assertEqual(observation.metadata["action_candidates"][4]["pokemon"]["species"], "Snorlax")

    def test_side_conditions_are_player_relative_in_metadata(self) -> None:
        lines = [
            *fixture_lines("p2_seat_replay.txt")[:5],
            "|-sidestart|p1: HumanFriend|Spikes",
            "|-sidestart|p2: PokeZeroBot|Spikes",
            "|-sidestart|p2: PokeZeroBot|move: Stealth Rock",
            "|-sideend|p1: HumanFriend|Spikes",
            *fixture_lines("p2_seat_replay.txt")[5:],
        ]
        replay = parse_showdown_replay(lines, battle_id="battle-gen3randombattle-1")

        state = normalize_for_player(replay, player_id="agent", player_name="PokeZeroBot")
        observation = observation_from_player_state(state)

        self.assertEqual(state.self_side_conditions, ("spikes", "stealthrock"))
        self.assertEqual(state.opponent_side_conditions, ())
        self.assertEqual(observation.metadata["self_side_conditions"], ["spikes", "stealthrock"])
        self.assertEqual(observation.metadata["opponent_side_conditions"], [])

    def test_side_condition_layer_counts_are_player_relative_in_metadata(self) -> None:
        lines = [
            *fixture_lines("p2_seat_replay.txt")[:5],
            "|-sidestart|p1: HumanFriend|Spikes",
            "|-sidestart|p1: HumanFriend|Spikes",
            "|-sidestart|p1: HumanFriend|Toxic Spikes",
            "|-sidestart|p1: HumanFriend|Toxic Spikes",
            "|-sidestart|p1: HumanFriend|Toxic Spikes",
            "|-sidestart|p2: PokeZeroBot|Spikes",
            "|-sidestart|p2: PokeZeroBot|move: Stealth Rock",
            *fixture_lines("p2_seat_replay.txt")[5:],
        ]
        replay = parse_showdown_replay(lines, battle_id="battle-gen3randombattle-1")

        state = normalize_for_player(replay, player_id="agent", player_name="PokeZeroBot")
        observation = observation_from_player_state(state)

        self.assertEqual(state.self_side_conditions, ("spikes", "stealthrock"))
        self.assertEqual(state.opponent_side_conditions, ("spikes", "toxicspikes"))
        self.assertEqual(state.self_side_condition_counts, {"spikes": 1, "stealthrock": 1})
        self.assertEqual(state.opponent_side_condition_counts, {"spikes": 2, "toxicspikes": 2})
        self.assertEqual(observation.metadata["self_side_condition_counts"], {"spikes": 1, "stealthrock": 1})
        self.assertEqual(observation.metadata["opponent_side_condition_counts"], {"spikes": 2, "toxicspikes": 2})

    def test_observation_encodes_player_relative_content(self) -> None:
        replay = parse_showdown_replay(fixture_lines("p2_seat_replay.txt"), battle_id="battle-gen3randombattle-1")
        state = normalize_for_player(replay, player_id="agent", player_name="PokeZeroBot")

        observation = observation_from_player_state(state)
        self_offset = FIELD_TOKEN_COUNT
        opponent_offset = self_offset + SELF_POKEMON_TOKEN_COUNT
        action_offset = opponent_offset + OPPONENT_POKEMON_TOKEN_COUNT
        event_offset = action_offset + ACTION_CANDIDATE_TOKEN_COUNT

        self.assertEqual(observation.categorical_ids[0][0], stable_category_id("request_kind:move"))
        self.assertEqual(observation.categorical_ids[self_offset][0], stable_category_id("species:Charizard"))
        self.assertEqual(observation.numeric_features[self_offset][0], 1.0)
        self.assertEqual(observation.numeric_features[self_offset][1], 1.0)
        self.assertEqual(observation.categorical_ids[opponent_offset][0], stable_category_id("species:Arcanine"))
        self.assertEqual(observation.numeric_features[opponent_offset][1], 0.0)
        self.assertEqual(observation.categorical_ids[opponent_offset + 1][0], stable_category_id("species:Xatu"))
        self.assertEqual(observation.numeric_features[opponent_offset + 1][0], 0.7)
        self.assertEqual(observation.numeric_features[opponent_offset + 1][1], 1.0)
        self.assertEqual(observation.numeric_features[opponent_offset + 1][4], 1.0)
        self.assertEqual(observation.numeric_features[opponent_offset + 1][5], 0.0)
        self.assertEqual(observation.numeric_features[opponent_offset + 1][6], 1.0)
        self.assertEqual(observation.categorical_ids[action_offset][0], stable_category_id("move:flamethrower"))
        self.assertEqual(observation.numeric_features[action_offset][2], 1.0)
        self.assertEqual(observation.categorical_ids[action_offset + 2][0], stable_category_id("move:dragonclaw"))
        self.assertEqual(observation.numeric_features[action_offset + 2][1], 0.0)
        self.assertEqual(observation.numeric_features[action_offset + 2][2], 0.0)
        self.assertEqual(observation.categorical_ids[action_offset + 4][0], stable_category_id("species:Snorlax"))
        self.assertEqual(observation.numeric_features[action_offset + 4][2], 1.0)
        self.assertEqual(observation.categorical_ids[action_offset + 5][0], stable_category_id("species:Blissey"))
        self.assertEqual(observation.numeric_features[action_offset + 5][0], 0.0)
        self.assertEqual(observation.numeric_features[action_offset + 5][2], 0.0)
        self.assertEqual(observation.categorical_ids[event_offset][0], stable_category_id("event:player"))
        move_event_index = next(
            index for index, event in enumerate(state.recent_events) if event.event_type == "move"
        )
        damage_event_index = next(
            index for index, event in enumerate(state.recent_events) if event.event_type == "-damage"
        )
        self.assertEqual(
            observation.categorical_ids[event_offset + move_event_index][0],
            stable_category_id("event:move"),
        )
        self.assertEqual(
            observation.categorical_ids[event_offset + move_event_index][1],
            stable_category_id("move:Flamethrower"),
        )
        self.assertEqual(
            observation.categorical_ids[event_offset + move_event_index][2],
            stable_category_id("event_actor:self"),
        )
        self.assertEqual(
            observation.categorical_ids[event_offset + move_event_index][3],
            stable_category_id("event_target:opponent"),
        )
        self.assertEqual(
            observation.categorical_ids[event_offset + damage_event_index][0],
            stable_category_id("event:-damage"),
        )
        # Lean encoding: the -damage detail (raw HP string "70/100") is unactionable — HP is
        # captured numerically and status via -status events — so the SECONDARY slot is padding.
        self.assertEqual(observation.categorical_ids[event_offset + damage_event_index][1], 0)
        self.assertEqual(
            observation.categorical_ids[event_offset + damage_event_index][3],
            stable_category_id("event_target:opponent"),
        )

    def test_observation_encodes_public_belief_summary_features(self) -> None:
        replay = parse_showdown_replay(fixture_lines("p2_seat_replay.txt"), battle_id="battle-gen3randombattle-1")
        state = normalize_for_player(
            replay,
            player_id="agent",
            player_name="PokeZeroBot",
            format_id="gen3randombattle",
            set_source=FakeSetSource(),
        )

        observation = observation_from_player_state(state)
        opponent_offset = FIELD_TOKEN_COUNT + SELF_POKEMON_TOKEN_COUNT
        xatu_offset = opponent_offset + 1

        self.assertEqual(observation.numeric_features[xatu_offset][5], 2.0)
        self.assertEqual(observation.numeric_features[xatu_offset][6], 0.5)
        self.assertEqual(observation.numeric_features[xatu_offset][7], 2.0)
        self.assertEqual(observation.numeric_features[xatu_offset][8], 1.0)
        self.assertEqual(observation.numeric_features[xatu_offset][9], 3.0)
        self.assertEqual(observation.numeric_features[xatu_offset][10], 0.0)
        self.assertEqual(observation.numeric_features[xatu_offset][11], 0.0)
        belief_fact_ids = observation.categorical_ids[xatu_offset][4:]
        self.assertIn(stable_category_id("belief:possible_ability:earlybird"), belief_fact_ids)
        self.assertIn(stable_category_id("belief:possible_ability:synchronize"), belief_fact_ids)
        self.assertIn(stable_category_id("belief:possible_item:leftovers"), belief_fact_ids)
        self.assertIn(stable_category_id("belief:possible_move:psychic"), belief_fact_ids)
        self.assertIn(stable_category_id("belief:possible_move:thunderwave"), belief_fact_ids)
        self.assertIn(stable_category_id("belief:possible_move:wish"), belief_fact_ids)
        self.assertNotIn(stable_category_id("belief:possible_moves:psychic|thunderwave|wish"), belief_fact_ids)
        self.assertNotIn("belief", observation.metadata)
        self.assertNotIn("belief", observation.metadata["opponent_active"])

        reordered_state = normalize_for_player(
            replay,
            player_id="agent",
            player_name="PokeZeroBot",
            format_id="gen3randombattle",
            set_source=ReorderedFakeSetSource(),
        )
        reordered = observation_from_player_state(reordered_state)
        self.assertEqual(
            observation.categorical_ids[xatu_offset],
            reordered.categorical_ids[xatu_offset],
        )

    def test_policy_action_translates_back_to_showdown_choice_for_detected_side(self) -> None:
        replay = parse_showdown_replay(fixture_lines("p2_seat_replay.txt"), battle_id="battle-gen3randombattle-1")
        state = normalize_for_player(replay, player_id="agent", player_name="PokeZeroBot")

        self.assertEqual(showdown_choice_for_action(state, 1), "move 2")
        self.assertEqual(showdown_choice_for_action(state, 4), "switch 2")
        self.assertEqual(showdown_submission_for_action(state, 4).showdown_slot, "p2")
        self.assertEqual(showdown_submission_for_action(state, 4).choice, "switch 2")

        with self.assertRaisesRegex(ValueError, "not legal"):
            showdown_choice_for_action(state, 2)

    def test_recent_events_are_normalized_to_self_and_opponent_roles(self) -> None:
        replay = parse_showdown_replay(fixture_lines("p2_seat_replay.txt"), battle_id="battle-gen3randombattle-1")

        state = normalize_for_player(replay, player_id="agent", player_name="PokeZeroBot")

        joined_events = "\n".join(state.recent_public_events)
        self.assertIn("|player|p1|HumanFriend|", joined_events)
        self.assertIn("|player|p2|PokeZeroBot|", joined_events)
        self.assertIn("|move|selfa: Charizard|Flamethrower|opponenta: Xatu", joined_events)
        self.assertIn("|-damage|opponenta: Xatu|70/100", joined_events)
        self.assertIn("opponenta: Xatu", joined_events)
        self.assertIn("selfa: Charizard", joined_events)
        self.assertNotIn("|player|opponent|", joined_events)
        self.assertNotIn("|player|self|", joined_events)
        self.assertNotIn("p1a: Xatu", joined_events)
        self.assertNotIn("p2a: Charizard", joined_events)

    def test_recent_events_are_structured_before_rendering(self) -> None:
        replay = parse_showdown_replay(fixture_lines("p2_seat_replay.txt"), battle_id="battle-gen3randombattle-1")

        state = normalize_for_player(replay, player_id="agent", player_name="PokeZeroBot")
        move_event = next(event for event in state.recent_events if event.event_type == "move")
        damage_event = next(event for event in state.recent_events if event.event_type == "-damage")

        self.assertEqual(move_event.actor_role, "self")
        self.assertEqual(move_event.target_role, "opponent")
        self.assertEqual(move_event.primary, "Flamethrower")
        self.assertEqual(damage_event.actor_role, "none")
        self.assertEqual(damage_event.target_role, "opponent")
        self.assertEqual(damage_event.primary, "70/100")


if __name__ == "__main__":
    unittest.main()
