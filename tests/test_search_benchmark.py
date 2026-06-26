import unittest

from pokezero.actions import ACTION_COUNT
from pokezero.env import StepResult, TerminalState
from pokezero.observation import PokeZeroObservationV0
from pokezero.policy import PolicyDecision
from pokezero.rollout import RolloutConfig
from pokezero.search_benchmark import benchmark_root_puct_search


def _mask(*legal_indices: int) -> tuple[bool, ...]:
    return tuple(index in set(legal_indices) for index in range(ACTION_COUNT))


def _observation(*legal_indices: int, branch_action: int | None = None) -> PokeZeroObservationV0:
    return PokeZeroObservationV0(
        categorical_ids=(),
        numeric_features=(),
        token_type_ids=(),
        attention_mask=(),
        legal_action_mask=_mask(*legal_indices),
        metadata={} if branch_action is None else {"branch_action": branch_action},
    )


class FixedPolicy:
    def __init__(self, action_index: int, *, policy_id: str) -> None:
        self.action_index = action_index
        self.policy_id = policy_id

    def select_action(self, observation: PokeZeroObservationV0, *, rng) -> PolicyDecision:
        return PolicyDecision(action_index=self.action_index, policy_id=self.policy_id, action_probability=1.0)


class TwoRoundBranchEnv:
    def __init__(self) -> None:
        self.reset_calls: list[tuple[int, str]] = []
        self.step_calls: list[dict[str, int]] = []
        self._round_index = 0
        self._terminal: TerminalState | None = None
        self.closed = False

    def reset(self, *, seed: int, format_id: str = "gen3randombattle") -> None:
        self.reset_calls.append((seed, format_id))
        self.step_calls.clear()
        self._round_index = 0
        self._terminal = None

    def observe(self, player: str) -> PokeZeroObservationV0:
        return _observation(0, 1) if player == "p1" else _observation(0)

    def legal_actions(self, player: str) -> tuple[bool, ...]:
        return self.observe(player).legal_action_mask

    def requested_players(self) -> tuple[str, ...]:
        return () if self._terminal is not None else ("p1", "p2")

    def step(self, actions: dict[str, int]) -> StepResult:
        self.step_calls.append(dict(actions))
        p1_action = int(actions["p1"])
        if self._round_index >= 1:
            winner = "p1" if p1_action == 0 else "p2"
            self._terminal = TerminalState(winner=winner, turn_count=self._round_index + 1)
            return StepResult(
                observations={},
                rewards={"p1": 1.0 if winner == "p1" else -1.0, "p2": -1.0 if winner == "p1" else 1.0},
                terminal=self._terminal,
                requested_players=(),
            )
        self._round_index += 1
        return StepResult(
            observations={"p1": _observation(0, 1, branch_action=p1_action), "p2": _observation(0)},
            rewards={"p1": 0.0, "p2": 0.0},
            terminal=None,
            requested_players=("p1", "p2"),
        )

    def terminal(self) -> TerminalState | None:
        return self._terminal

    def close(self) -> None:
        self.closed = True


class RootPUCTSearchBenchmarkTest(unittest.TestCase):
    def test_benchmark_root_puct_search_reports_prefix_decision_deltas(self) -> None:
        envs: list[TwoRoundBranchEnv] = []

        def env_factory() -> TwoRoundBranchEnv:
            env = TwoRoundBranchEnv()
            envs.append(env)
            return env

        def value_fn(history: tuple[PokeZeroObservationV0, ...]) -> float:
            return {0: 0.1, 1: 0.8}.get(int(history[-1].metadata.get("branch_action", 0)), 0.0)

        def prior_fn(history: tuple[PokeZeroObservationV0, ...]) -> tuple[float, ...]:
            return (0.5, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

        report = benchmark_root_puct_search(
            env_factory=env_factory,
            policies={"p1": FixedPolicy(0, policy_id="fixed-p1"), "p2": FixedPolicy(0, policy_id="fixed-p2")},
            rollout_config=RolloutConfig(max_decision_rounds=3),
            games=1,
            prefixes_per_game=2,
            value_fn=value_fn,
            prior_fn=prior_fn,
            cpuct=0.0,
        )

        self.assertEqual(report.source_decision_rounds, (2,))
        self.assertEqual(report.evaluated_prefixes, 2)
        self.assertEqual(report.skipped_prefixes, 0)
        self.assertEqual(report.changed_actions, 1)
        self.assertEqual(report.action_change_rate, 0.5)
        self.assertEqual([decision.prefix_decision_round_count for decision in report.decisions], [0, 1])
        self.assertEqual([decision.recorded_action_index for decision in report.decisions], [0, 0])
        self.assertEqual([decision.selected_action_index for decision in report.decisions], [1, 0])
        self.assertEqual([decision.candidate_count for decision in report.decisions], [2, 2])
        self.assertEqual(report.to_dict()["evaluated_prefixes"], 2)
        self.assertTrue(envs[0].closed)

    def test_benchmark_root_puct_search_requires_search_player_policy(self) -> None:
        with self.assertRaisesRegex(ValueError, "search_player"):
            benchmark_root_puct_search(
                env_factory=TwoRoundBranchEnv,
                policies={"p1": FixedPolicy(0, policy_id="fixed-p1")},
                rollout_config=RolloutConfig(max_decision_rounds=3),
                games=1,
                value_fn=lambda history: 0.0,
                prior_fn=lambda history: (1.0,) + (0.0,) * (ACTION_COUNT - 1),
                search_player="p2",
            )


if __name__ == "__main__":
    unittest.main()
