from pathlib import Path
import unittest

from pokezero.actions import ACTION_COUNT
from pokezero.observation import (
    ACTION_CANDIDATE_TOKEN_COUNT,
    FIELD_TOKEN_COUNT,
    OPPONENT_POKEMON_TOKEN_COUNT,
    SELF_POKEMON_TOKEN_COUNT,
)
from pokezero.showdown import (
    DEFAULT_REPLAY_OBSERVATION_SPEC,
    detect_showdown_slot,
    normalize_for_player,
    observation_from_player_state,
    parse_showdown_replay,
    showdown_choice_for_action,
    showdown_submission_for_action,
    stable_category_id,
)


FIXTURE_ROOT = Path(__file__).parent / "fixtures" / "showdown"


def fixture_lines(name: str) -> list[str]:
    return (FIXTURE_ROOT / name).read_text(encoding="utf-8").splitlines()


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

    def test_observation_shell_carries_detected_perspective_and_legal_mask(self) -> None:
        replay = parse_showdown_replay(fixture_lines("p2_seat_replay.txt"), battle_id="battle-gen3randombattle-1")
        state = normalize_for_player(replay, player_id="agent", player_name="PokeZeroBot")

        observation = observation_from_player_state(state)

        observation.validate(DEFAULT_REPLAY_OBSERVATION_SPEC)
        self.assertEqual(observation.perspective.showdown_slot, "p2")
        self.assertEqual(observation.perspective.opponent_showdown_slot, "p1")
        self.assertEqual(observation.legal_action_mask, state.legal_action_mask)

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
        self.assertEqual(observation.numeric_features[opponent_offset + 1][1], 1.0)
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
        self.assertEqual(
            observation.categorical_ids[event_offset + damage_event_index][1],
            stable_category_id("condition:70/100"),
        )
        self.assertEqual(
            observation.categorical_ids[event_offset + damage_event_index][3],
            stable_category_id("event_target:opponent"),
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
