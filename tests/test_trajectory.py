import unittest

from pokezero.env import TerminalState
from pokezero.observation import ObservationSpec, PokeZeroObservationV0
from pokezero.trajectory import BattleTrajectory, TrajectoryStep


def observation(mask: tuple[bool, ...]) -> PokeZeroObservationV0:
    spec = ObservationSpec(categorical_feature_count=1, numeric_feature_count=1)
    return PokeZeroObservationV0(
        categorical_ids=tuple((0,) for _ in range(spec.token_count)),
        numeric_features=tuple((0.0,) for _ in range(spec.token_count)),
        token_type_ids=tuple(0 for _ in range(spec.token_count)),
        attention_mask=tuple(True for _ in range(spec.token_count)),
        legal_action_mask=mask,
    )


class TrajectoryTest(unittest.TestCase):
    def test_trajectory_records_steps_rewards_and_terminal_state(self) -> None:
        mask = (True, False, False, False, False, False, False, False, False)
        trajectory = BattleTrajectory(battle_id="battle-1", format_id="gen3randombattle", seed=123)

        trajectory.append(
            TrajectoryStep(
                player_id="p1",
                turn_index=0,
                observation=observation(mask),
                legal_action_mask=mask,
                action_index=0,
                reward=1.0,
                opponent_action_index=1,
                action_probability=1.0,
            )
        )
        trajectory.append(
            TrajectoryStep(
                player_id="p2",
                turn_index=0,
                observation=observation(mask),
                legal_action_mask=mask,
                action_index=0,
                reward=-1.0,
            )
        )
        trajectory.record_terminal(TerminalState(winner="p1", turn_count=12, capped=False))

        self.assertEqual(trajectory.players(), ("p1", "p2"))
        self.assertEqual([step.player_id for step in trajectory.steps_for_turn(0)], ["p1", "p2"])
        self.assertEqual(len(trajectory.steps_for_player("p1")), 1)
        self.assertEqual(trajectory.total_reward("p1"), 1.0)
        self.assertEqual(trajectory.total_reward("p2"), -1.0)
        self.assertFalse(trajectory.capped)

    def test_trajectory_rejects_append_after_terminal(self) -> None:
        mask = (True, False, False, False, False, False, False, False, False)
        trajectory = BattleTrajectory(battle_id="battle-1", format_id="gen3randombattle", seed=123)
        trajectory.record_terminal(TerminalState(winner=None, turn_count=250, capped=True))

        with self.assertRaisesRegex(ValueError, "terminal"):
            trajectory.append(
                TrajectoryStep(
                    player_id="p1",
                    turn_index=0,
                    observation=observation(mask),
                    legal_action_mask=mask,
                    action_index=0,
                )
            )

    def test_trajectory_step_requires_legal_action(self) -> None:
        mask = (False, True, False, False, False, False, False, False, False)

        with self.assertRaisesRegex(ValueError, "must be legal"):
            TrajectoryStep(
                player_id="p1",
                turn_index=0,
                observation=observation(mask),
                legal_action_mask=mask,
                action_index=0,
            )

    def test_trajectory_step_requires_non_negative_turn_index(self) -> None:
        mask = (True, False, False, False, False, False, False, False, False)

        with self.assertRaisesRegex(ValueError, "turn_index"):
            TrajectoryStep(
                player_id="p1",
                turn_index=-1,
                observation=observation(mask),
                legal_action_mask=mask,
                action_index=0,
            )

    def test_trajectory_step_requires_mask_to_match_observation(self) -> None:
        observation_mask = (True, False, False, False, False, False, False, False, False)
        recorded_mask = (False, True, False, False, False, False, False, False, False)

        with self.assertRaisesRegex(ValueError, "must match"):
            TrajectoryStep(
                player_id="p1",
                turn_index=0,
                observation=observation(observation_mask),
                legal_action_mask=recorded_mask,
                action_index=1,
            )


if __name__ == "__main__":
    unittest.main()
