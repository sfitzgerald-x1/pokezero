import unittest

from pokezero.public_action_capture import (
    append_public_action_round,
    public_action_round_from_protocol_lines,
    public_action_rounds_from_trajectory_metadata,
)
from pokezero.trajectory import BattleTrajectory


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
                }
            ],
        )
        decoded = public_action_rounds_from_trajectory_metadata(trajectory)
        self.assertEqual(decoded[0].actions["p2"].event_id, "cant:slp")

        with self.assertRaisesRegex(ValueError, "captured more than once"):
            append_public_action_round(trajectory, action_round)
