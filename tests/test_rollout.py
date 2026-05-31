import unittest

from pokezero.env import StepResult, TerminalState
from pokezero.observation import ObservationSpec, PokeZeroObservationV0
from pokezero.policy import PolicyDecision, RandomLegalPolicy, SimpleLegalPolicy
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
        self.observe_calls: list[str] = []
        self.default_observation = observation((True, False, False, False, True, False, False, False, False))

    def reset(self, *, seed: int, format_id: str = "gen3randombattle") -> None:
        self.reset_calls.append((seed, format_id))
        self.step_calls.clear()
        self.observe_calls.clear()

    def observe(self, player: str) -> PokeZeroObservationV0:
        self.observe_calls.append(player)
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
        self.assertEqual(result.decision_round_count, 1)
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
            config=RolloutConfig(max_decision_rounds=3),
        )

        result = driver.run(seed=5)

        self.assertEqual(result.terminal, TerminalState(winner=None, turn_count=3, capped=True))
        self.assertEqual(result.decision_round_count, 3)
        self.assertTrue(result.trajectory.capped)
        self.assertEqual(len(result.trajectory.steps), 6)

    def test_rollout_uses_per_player_rng_streams(self) -> None:
        env_a = ScriptedEnv(requested_sequence=[("p1", "p2")], terminal_after_steps=1)
        env_b = ScriptedEnv(requested_sequence=[("p1", "p2")], terminal_after_steps=1)
        p2_policy = RandomLegalPolicy()

        result_a = RolloutDriver(
            env=env_a,
            policies={"p1": DrawBurningPolicy(draw_count=0), "p2": p2_policy},
        ).run(seed=55)
        result_b = RolloutDriver(
            env=env_b,
            policies={"p1": DrawBurningPolicy(draw_count=20), "p2": p2_policy},
        ).run(seed=55)

        p2_action_a = result_a.trajectory.steps_for_player("p2")[0].action_index
        p2_action_b = result_b.trajectory.steps_for_player("p2")[0].action_index
        self.assertEqual(p2_action_a, p2_action_b)

    def test_rollout_consumes_step_result_requested_players_and_observations(self) -> None:
        first_observation = observation((True, False, False, False, False, False, False, False, False))
        second_observation = observation((False, False, False, False, True, False, False, False, False))
        env = StepResultDrivenEnv(first_observation=first_observation, second_observation=second_observation)
        driver = RolloutDriver(env=env, policies={"p1": RandomLegalPolicy()}, config=RolloutConfig(max_decision_rounds=2))

        result = driver.run(seed=8)

        self.assertEqual(env.requested_players_calls, 1)
        self.assertEqual(env.observe_calls, ["p1"])
        self.assertEqual([step.action_index for step in result.trajectory.steps], [0, 4])

    def test_rollout_preserves_policy_action_probability(self) -> None:
        env = ScriptedEnv(requested_sequence=[("p1",)], terminal_after_steps=1)
        driver = RolloutDriver(
            env=env,
            policies={"p1": SimpleLegalPolicy(switch_probability=0.25)},
        )

        result = driver.run(seed=1)
        step = result.trajectory.steps[0]

        if step.action_index < 4:
            self.assertEqual(step.action_probability, 0.75)
        else:
            self.assertEqual(step.action_probability, 0.25)

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

class DrawBurningPolicy:
    policy_id = "draw-burning"

    def __init__(self, *, draw_count: int) -> None:
        self.draw_count = draw_count

    def select_action(self, observation: PokeZeroObservationV0, *, rng) -> PolicyDecision:
        for _ in range(self.draw_count):
            rng.random()
        return RandomLegalPolicy(policy_id=self.policy_id).select_action(observation, rng=rng)


class StepResultDrivenEnv:
    def __init__(
        self,
        *,
        first_observation: PokeZeroObservationV0,
        second_observation: PokeZeroObservationV0,
    ) -> None:
        self.first_observation = first_observation
        self.second_observation = second_observation
        self.observe_calls: list[str] = []
        self.requested_players_calls = 0
        self.step_calls: list[dict[str, int]] = []

    def reset(self, *, seed: int, format_id: str = "gen3randombattle") -> None:
        self.observe_calls.clear()
        self.requested_players_calls = 0
        self.step_calls.clear()

    def observe(self, player: str) -> PokeZeroObservationV0:
        self.observe_calls.append(player)
        return self.first_observation

    def legal_actions(self, player: str) -> tuple[bool, ...]:
        return self.first_observation.legal_action_mask

    def requested_players(self) -> tuple[str, ...]:
        self.requested_players_calls += 1
        return ("p1",)

    def step(self, actions: dict[str, int]) -> StepResult:
        self.step_calls.append(dict(actions))
        terminal = TerminalState(winner="p1", turn_count=2) if len(self.step_calls) == 2 else None
        return StepResult(
            observations={"p1": self.second_observation},
            rewards={"p1": 0.0},
            terminal=terminal,
            requested_players=() if terminal else ("p1",),
        )

    def terminal(self) -> TerminalState | None:
        return None


if __name__ == "__main__":
    unittest.main()
