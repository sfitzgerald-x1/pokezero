import unittest

from pokezero.env import StepResult, TerminalState
from pokezero.observation import ObservationSpec, PokeZeroObservationV0
from pokezero.policy import RandomLegalPolicy, SimpleLegalPolicy
from pokezero.rollout import RolloutConfig, RolloutDriver


def observation(mask: tuple[bool, ...]) -> PokeZeroObservationV0:
    spec = ObservationSpec(categorical_feature_count=1, numeric_feature_count=1)
    return PokeZeroObservationV0(
        categorical_ids=tuple((0,) for _ in range(spec.token_count)),
        numeric_features=tuple((0.0,) for _ in range(spec.token_count)),
        token_type_ids=tuple(0 for _ in range(spec.token_count)),
        attention_mask=tuple(True for _ in range(spec.token_count)),
        legal_action_mask=mask,
    )


class ScriptedEnv:
    def __init__(
        self,
        *,
        requested_sequence: list[tuple[str, ...]],
        terminal_after_steps: int | None = None,
    ) -> None:
        self.requested_sequence = requested_sequence
        self.terminal_after_steps = terminal_after_steps
        self.step_calls: list[dict[str, int]] = []
        self.reset_calls: list[tuple[int, str]] = []
        self.default_observation = observation((True, False, False, False, True, False, False, False, False))

    def reset(self, *, seed: int, format_id: str = "gen3randombattle") -> None:
        self.reset_calls.append((seed, format_id))
        self.step_calls.clear()

    def observe(self, player: str) -> PokeZeroObservationV0:
        return self.default_observation

    def legal_actions(self, player: str) -> tuple[bool, ...]:
        return self.default_observation.legal_action_mask

    def requested_players(self) -> tuple[str, ...]:
        if len(self.step_calls) < len(self.requested_sequence):
            return self.requested_sequence[len(self.step_calls)]
        return ()

    def step(self, actions: dict[str, int]) -> StepResult:
        self.step_calls.append(dict(actions))
        terminal = self.terminal()
        return StepResult(
            observations={player: self.default_observation for player in ("p1", "p2")},
            rewards={"p1": 1.0 if terminal else 0.25, "p2": -1.0 if terminal else -0.25},
            terminal=terminal,
            requested_players=self.requested_players(),
        )

    def terminal(self) -> TerminalState | None:
        if self.terminal_after_steps is not None and len(self.step_calls) >= self.terminal_after_steps:
            return TerminalState(winner="p1", turn_count=len(self.step_calls))
        return None


class RolloutDriverTest(unittest.TestCase):
    def test_rollout_records_simultaneous_turn_steps_and_terminal(self) -> None:
        env = ScriptedEnv(requested_sequence=[("p1", "p2")], terminal_after_steps=1)
        driver = RolloutDriver(
            env=env,
            policies={
                "p1": RandomLegalPolicy(),
                "p2": SimpleLegalPolicy(switch_probability=0.0),
            },
        )

        result = driver.run(seed=123, battle_id="battle-test")

        self.assertEqual(env.reset_calls, [(123, "gen3randombattle")])
        self.assertEqual(result.terminal, TerminalState(winner="p1", turn_count=1))
        self.assertEqual(result.step_count, 1)
        self.assertEqual(result.trajectory.players(), ("p1", "p2"))
        self.assertEqual([step.turn_index for step in result.trajectory.steps], [0, 0])
        self.assertEqual(result.trajectory.steps[0].opponent_action_index, result.trajectory.steps[1].action_index)
        self.assertEqual(result.trajectory.steps[1].opponent_action_index, result.trajectory.steps[0].action_index)
        self.assertEqual(result.trajectory.steps[0].metadata["policy_id"], "random-legal")
        self.assertEqual(result.trajectory.steps[1].metadata["policy_id"], "simple-legal")

    def test_rollout_records_asymmetric_requested_player_without_opponent_action(self) -> None:
        env = ScriptedEnv(requested_sequence=[("p1",)], terminal_after_steps=1)
        driver = RolloutDriver(
            env=env,
            policies={"p1": RandomLegalPolicy()},
        )

        result = driver.run(seed=1)

        self.assertEqual(len(result.trajectory.steps), 1)
        self.assertEqual(result.trajectory.steps[0].player_id, "p1")
        self.assertIsNone(result.trajectory.steps[0].opponent_action_index)

    def test_rollout_caps_when_environment_does_not_terminal(self) -> None:
        env = ScriptedEnv(requested_sequence=[("p1", "p2")] * 5, terminal_after_steps=None)
        driver = RolloutDriver(
            env=env,
            policies={"p1": RandomLegalPolicy(), "p2": RandomLegalPolicy()},
            config=RolloutConfig(max_steps=3),
        )

        result = driver.run(seed=5)

        self.assertEqual(result.terminal, TerminalState(winner=None, turn_count=3, capped=True))
        self.assertEqual(result.step_count, 3)
        self.assertTrue(result.trajectory.capped)
        self.assertEqual(len(result.trajectory.steps), 6)

    def test_rollout_rejects_missing_policy_for_requested_player(self) -> None:
        env = ScriptedEnv(requested_sequence=[("p1", "p2")], terminal_after_steps=1)
        driver = RolloutDriver(env=env, policies={"p1": RandomLegalPolicy()})

        with self.assertRaisesRegex(ValueError, "no policy configured"):
            driver.run(seed=1)

    def test_rollout_rejects_non_terminal_empty_request(self) -> None:
        env = ScriptedEnv(requested_sequence=[], terminal_after_steps=None)
        driver = RolloutDriver(env=env, policies={})

        with self.assertRaisesRegex(ValueError, "requested no players"):
            driver.run(seed=1)


if __name__ == "__main__":
    unittest.main()
