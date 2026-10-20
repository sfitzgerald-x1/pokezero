import unittest

from pokezero.public_action_capture import (
    append_public_action_round,
    public_action_round_from_protocol_lines,
    public_action_rounds_from_trajectory_metadata,
)
from pokezero.trajectory import BattleTrajectory
from pokezero.public_decision_corpus import PublicResolvedActionRound


class PublicActionCaptureTest(unittest.TestCase):
    def test_captures_protocol_actions_and_marks_unresolved_requested_players(self) -> None:
        action_round = public_action_round_from_protocol_lines(
            (
                "|move|p1a: Lead|Hidden Power Grass|p2a: Rival",
                "|switch|p2a: Rival|Donphan, L74|100/100",
            ),
            turn_index=3,
            requested_players=("p1", "p2"),
        )

        self.assertEqual(
            action_round.to_dict(),
            {
                "turn_index": 3,
                "actions": {
                    "p1": {"kind": "move", "move_id": "hiddenpowergrass"},
                    "p2": {"kind": "switch", "switched_species": "donphan"},
                },
                "team_transitions": [{"player_id": "p2", "kind": "switch", "species": "donphan"}],
            },
        )

    def test_persists_rounds_without_request_local_action_slots(self) -> None:
        trajectory = BattleTrajectory(battle_id="public-actions", format_id="gen3randombattle", seed=7)
        action_round = public_action_round_from_protocol_lines(
            ("|cant|p2a: Rival|slp",),
            turn_index=0,
            requested_players=("p1", "p2"),
        )

        append_public_action_round(trajectory, action_round)

        self.assertEqual(
            trajectory.metadata["public_resolved_action_rounds"],
            [
                {
                    "turn_index": 0,
                    "actions": {
                        "p1": {"kind": "event", "event_id": "unresolved-public-event"},
                        "p2": {"kind": "event", "event_id": "cant:slp"},
                    },
                    "team_transitions": [],
                }
            ],
        )
        decoded = public_action_rounds_from_trajectory_metadata(trajectory)
        self.assertEqual(decoded[0].actions["p2"].event_id, "cant:slp")

        with self.assertRaisesRegex(ValueError, "captured more than once"):
            append_public_action_round(trajectory, action_round)

    def test_sleep_talk_move_replaces_a_provisional_sleep_event(self) -> None:
        action_round = public_action_round_from_protocol_lines(
            (
                "|cant|p1a: Lead|slp",
                "|move|p1a: Lead|Sleep Talk|p2a: Rival",
                "|move|p1a: Lead|Surf|p2a: Rival|[from]move: Sleep Talk",
            ),
            turn_index=4,
            requested_players=("p1",),
        )

        self.assertEqual(
            action_round.to_dict()["actions"]["p1"],
            {"kind": "move", "move_id": "sleeptalk"},
        )

    def test_ordered_public_transitions_round_trip_without_private_fields(self) -> None:
        action_round = public_action_round_from_protocol_lines(
            ("|switch|p2a: private nickname|Smeargle, L80|100/100",
             "|drag|p1a: own nickname|Absol, L80|100/100",
             "|drag|p2a: private nickname|Typhlosion, L80|100/100"),
            turn_index=0, requested_players=("p1", "p2"),
        )
        expected = [dict(player_id="p2", kind="switch", species="smeargle"),
                    dict(player_id="p1", kind="drag", species="absol"),
                    dict(player_id="p2", kind="drag", species="typhlosion")]
        payload = action_round.to_dict()
        self.assertEqual(payload["team_transitions"], expected)
        self.assertNotIn("nickname", str(payload))
        self.assertEqual(PublicResolvedActionRound.from_dict(payload).to_dict(), payload)

    def test_legacy_absent_and_exact_empty_boundaries_remain_distinct(self) -> None:
        legacy = {"turn_index": 0, "actions": {"p1": {"kind": "move", "move_id": "surf"}}}
        self.assertIsNone(PublicResolvedActionRound.from_dict(legacy).team_transitions)
        self.assertEqual(PublicResolvedActionRound.from_dict(legacy).to_dict(), legacy)
        empty = {**legacy, "team_transitions": []}
        self.assertEqual(PublicResolvedActionRound.from_dict(empty).team_transitions, ())
        self.assertEqual(PublicResolvedActionRound.from_dict(empty).to_dict(), empty)
        for value in (None, "switch", {}, [None], [dict(player_id=[], kind="drag", species="absol")],
                      [dict(player_id="p2", kind="drag", species="absol", private_request={})]):
            with self.subTest(value=value), self.assertRaises(ValueError):
                PublicResolvedActionRound.from_dict({**legacy, "team_transitions": value})
