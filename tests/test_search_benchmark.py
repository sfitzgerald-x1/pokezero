import unittest
from types import SimpleNamespace
from unittest.mock import patch

from pokezero.actions import ACTION_COUNT
from pokezero.env import StepResult, TerminalState
from pokezero.observation import PokeZeroObservationV0
from pokezero.policy import PolicyDecision
from pokezero.rollout import RolloutConfig
from pokezero.search_benchmark import benchmark_root_puct_counterfactual_rollouts, benchmark_root_puct_search
from pokezero.search import RootPUCTSearchTiming


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


class ImmediateOutcomeEnv:
    def __init__(self) -> None:
        self.reset_calls: list[tuple[int, str]] = []
        self.step_calls: list[dict[str, int]] = []
        self._terminal: TerminalState | None = None
        self.closed = False

    def reset(self, *, seed: int, format_id: str = "gen3randombattle") -> None:
        self.reset_calls.append((seed, format_id))
        self.step_calls.clear()
        self._terminal = None

    def observe(self, player: str) -> PokeZeroObservationV0:
        return _observation(0, 1) if player == "p1" else _observation(0)

    def legal_actions(self, player: str) -> tuple[bool, ...]:
        return self.observe(player).legal_action_mask

    def requested_players(self) -> tuple[str, ...]:
        return () if self._terminal is not None else ("p1", "p2")

    def step(self, actions: dict[str, int]) -> StepResult:
        self.step_calls.append(dict(actions))
        winner = "p1" if int(actions["p1"]) == 1 else "p2"
        self._terminal = TerminalState(winner=winner, turn_count=1)
        return StepResult(
            observations={},
            rewards={"p1": 1.0 if winner == "p1" else -1.0, "p2": -1.0 if winner == "p1" else 1.0},
            terminal=self._terminal,
            requested_players=(),
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
        self.assertEqual([decision.total_visits for decision in report.decisions], [2, 2])
        payload = report.to_dict()
        self.assertEqual(payload["schema_version"], "pokezero.root-puct-search-benchmark.v2")
        self.assertEqual(payload["evaluated_prefixes"], 2)
        self.assertIsNone(payload["root_extra_visits"])
        timing = payload["timing"]
        decision_timings = [decision["timing"] for decision in payload["decisions"]]
        for key in (
            "prefix_replay_count",
            "branch_simulator_step_count",
            "state_snapshot_count",
            "state_restore_count",
            "root_initial_sweep_orchestration_count",
            "root_search_setup_count",
            "root_adaptive_visit_orchestration_count",
            "root_search_finalization_count",
            "branch_action_validation_count",
            "post_branch_history_count",
            "belief_world_materialization_count",
            "opponent_scenario_planning_count",
            "root_policy_setup_count",
            "direct_prefix_construction_count",
            "scenario_dispatch_orchestration_count",
            "policy_evaluation_count",
            "observation_encoding_count",
            "neural_forward_count",
            "action_prior_neural_forward_count",
            "opponent_action_prior_neural_forward_count",
            "policy_neural_forward_count",
            "value_neural_forward_count",
            "value_evaluation_count",
            "policy_value_evaluation_count",
            "rollout_tail_count",
        ):
            self.assertEqual(timing[key], sum(decision_timing[key] for decision_timing in decision_timings))
        for key in (
            "prefix_replay_seconds",
            "branch_simulator_step_seconds",
            "state_snapshot_seconds",
            "state_restore_seconds",
            "root_initial_sweep_orchestration_seconds",
            "root_search_setup_seconds",
            "root_adaptive_visit_orchestration_seconds",
            "root_search_finalization_seconds",
            "branch_action_validation_seconds",
            "post_branch_history_seconds",
            "belief_world_materialization_seconds",
            "opponent_scenario_planning_seconds",
            "root_policy_setup_seconds",
            "direct_prefix_construction_seconds",
            "scenario_dispatch_orchestration_seconds",
            "policy_evaluation_seconds",
            "observation_encoding_seconds",
            "neural_forward_seconds",
            "action_prior_neural_forward_seconds",
            "opponent_action_prior_neural_forward_seconds",
            "policy_neural_forward_seconds",
            "value_neural_forward_seconds",
            "value_evaluation_seconds",
            "policy_value_evaluation_seconds",
            "rollout_tail_seconds",
            "raw_residual_seconds",
            "total_seconds",
        ):
            self.assertAlmostEqual(timing[key], sum(decision_timing[key] for decision_timing in decision_timings))
        for timing_payload in (*decision_timings, timing):
            self.assertGreaterEqual(timing_payload["raw_residual_seconds"], -1e-9)
            self.assertAlmostEqual(
                timing_payload["total_seconds"],
                timing_payload["prefix_replay_seconds"]
                + timing_payload["branch_simulator_step_seconds"]
                + timing_payload["state_snapshot_seconds"]
                + timing_payload["state_restore_seconds"]
                + timing_payload["root_initial_sweep_orchestration_seconds"]
                + timing_payload["root_search_setup_seconds"]
                + timing_payload["root_adaptive_visit_orchestration_seconds"]
                + timing_payload["root_search_finalization_seconds"]
                + timing_payload["branch_action_validation_seconds"]
                + timing_payload["post_branch_history_seconds"]
                + timing_payload["belief_world_materialization_seconds"]
                + timing_payload["opponent_scenario_planning_seconds"]
                + timing_payload["root_policy_setup_seconds"]
                + timing_payload["direct_prefix_construction_seconds"]
                + timing_payload["scenario_dispatch_orchestration_seconds"]
                + timing_payload["policy_value_evaluation_seconds"]
                + timing_payload["rollout_tail_seconds"]
                + timing_payload["raw_residual_seconds"],
            )
        self.assertEqual(timing["rollout_tail_count"], 0)
        self.assertEqual(timing["rollout_tail_seconds"], 0.0)
        self.assertEqual(timing["opponent_scenario_planning_count"], 0)
        self.assertEqual(timing["opponent_scenario_planning_seconds"], 0.0)
        self.assertTrue(envs[0].closed)

    def test_benchmark_root_puct_search_selects_the_most_visited_candidate(self) -> None:
        score_winner = SimpleNamespace(action_index=0, visits=1, prior=0.8, value=0.9, score=1.5)
        visit_winner = SimpleNamespace(action_index=1, visits=3, prior=0.2, value=0.1, score=0.2)
        search = SimpleNamespace(
            candidates=(score_winner, visit_winner),
            most_visited_candidate=visit_winner,
            total_visits=4,
            timing=RootPUCTSearchTiming(),
        )

        with patch("pokezero.search_benchmark.puct_branch_search", return_value=search):
            report = benchmark_root_puct_search(
                env_factory=TwoRoundBranchEnv,
                policies={"p1": FixedPolicy(0, policy_id="fixed-p1"), "p2": FixedPolicy(0, policy_id="fixed-p2")},
                rollout_config=RolloutConfig(max_decision_rounds=3),
                games=1,
                prefixes_per_game=2,
                value_fn=lambda _history: 0.0,
                prior_fn=lambda _history: (0.5, 0.5) + (0.0,) * 7,
            )

        self.assertEqual([decision.selected_action_index for decision in report.decisions], [1, 1])
        self.assertEqual([decision.selected_value for decision in report.decisions], [0.1, 0.1])
        self.assertEqual([decision.selected_score for decision in report.decisions], [0.2, 0.2])

    def test_benchmark_root_puct_search_supports_fixed_extra_visits(self) -> None:
        def value_fn(history: tuple[PokeZeroObservationV0, ...]) -> float:
            return {0: 0.1, 1: 0.8}.get(int(history[-1].metadata.get("branch_action", 0)), 0.0)

        report = benchmark_root_puct_search(
            env_factory=TwoRoundBranchEnv,
            policies={"p1": FixedPolicy(0, policy_id="fixed-p1"), "p2": FixedPolicy(0, policy_id="fixed-p2")},
            rollout_config=RolloutConfig(max_decision_rounds=3),
            games=1,
            prefixes_per_game=2,
            value_fn=value_fn,
            prior_fn=lambda _history: (0.5, 0.5) + (0.0,) * 7,
            cpuct=0.0,
            root_extra_visits=3,
        )

        self.assertEqual(report.root_extra_visits, 3)
        self.assertEqual(
            [decision.total_visits for decision in report.decisions],
            [decision.candidate_count + 3 for decision in report.decisions],
        )
        self.assertEqual(report.to_dict()["root_extra_visits"], 3)

        with self.assertRaisesRegex(ValueError, "non-negative"):
            benchmark_root_puct_search(
                env_factory=TwoRoundBranchEnv,
                policies={"p1": FixedPolicy(0, policy_id="fixed-p1"), "p2": FixedPolicy(0, policy_id="fixed-p2")},
                rollout_config=RolloutConfig(max_decision_rounds=3),
                games=1,
                value_fn=value_fn,
                prior_fn=lambda _history: (0.5, 0.5) + (0.0,) * 7,
                root_extra_visits=-1,
            )

    def test_benchmark_root_puct_counterfactual_rollouts_compares_recorded_and_selected_outcomes(self) -> None:
        envs: list[ImmediateOutcomeEnv] = []

        def env_factory() -> ImmediateOutcomeEnv:
            env = ImmediateOutcomeEnv()
            envs.append(env)
            return env

        report = benchmark_root_puct_counterfactual_rollouts(
            env_factory=env_factory,
            policies={"p1": FixedPolicy(0, policy_id="source-p1"), "p2": FixedPolicy(0, policy_id="source-p2")},
            continuation_policies={
                "p1": FixedPolicy(0, policy_id="continue-p1"),
                "p2": FixedPolicy(0, policy_id="continue-p2"),
            },
            rollout_config=RolloutConfig(max_decision_rounds=3),
            games=2,
            prefixes_per_game=1,
            value_fn=lambda history: 0.0,
            prior_fn=lambda history: (0.5, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
            cpuct=0.0,
        )

        self.assertEqual(report.source_decision_rounds, (1, 1))
        self.assertEqual(report.evaluated_prefixes, 2)
        self.assertEqual(report.changed_actions, 2)
        self.assertEqual(report.improved_actions, 2)
        self.assertEqual(report.worsened_actions, 0)
        self.assertEqual(report.average_recorded_rollout_value, -1.0)
        self.assertEqual(report.average_selected_rollout_value, 1.0)
        self.assertEqual(report.average_rollout_value_delta, 2.0)
        self.assertEqual(report.continuation_policy_ids, {"p1": "continue-p1", "p2": "continue-p2"})
        decision = report.decisions[0]
        self.assertEqual(decision.recorded_action_index, 0)
        self.assertEqual(decision.selected_action_index, 1)
        self.assertEqual(decision.recorded_winner, "p2")
        self.assertEqual(decision.selected_winner, "p1")
        self.assertEqual(decision.rollout_value_delta, 2.0)
        payload = report.to_dict()
        self.assertEqual(payload["improved_actions"], 2)
        self.assertIn("search_timing", payload)
        search_timing = payload["search_timing"]
        decision_timings = [decision_payload["search_timing"] for decision_payload in payload["decisions"]]
        self.assertEqual(len(decision_timings), 2)
        for key in (
            "prefix_replay_count",
            "branch_simulator_step_count",
            "state_snapshot_count",
            "state_restore_count",
            "belief_world_materialization_count",
            "opponent_scenario_planning_count",
            "policy_evaluation_count",
            "value_evaluation_count",
            "policy_value_evaluation_count",
            "rollout_tail_count",
        ):
            self.assertEqual(search_timing[key], sum(timing[key] for timing in decision_timings))
        for key in (
            "prefix_replay_seconds",
            "branch_simulator_step_seconds",
            "state_snapshot_seconds",
            "state_restore_seconds",
            "root_initial_sweep_orchestration_seconds",
            "root_search_setup_seconds",
            "root_adaptive_visit_orchestration_seconds",
            "root_search_finalization_seconds",
            "branch_action_validation_seconds",
            "post_branch_history_seconds",
            "belief_world_materialization_seconds",
            "opponent_scenario_planning_seconds",
            "root_policy_setup_seconds",
            "direct_prefix_construction_seconds",
            "scenario_dispatch_orchestration_seconds",
            "policy_evaluation_seconds",
            "value_evaluation_seconds",
            "policy_value_evaluation_seconds",
            "rollout_tail_seconds",
            "raw_residual_seconds",
            "total_seconds",
        ):
            self.assertAlmostEqual(search_timing[key], sum(timing[key] for timing in decision_timings))
        for timing in (*decision_timings, search_timing):
            self.assertGreaterEqual(timing["raw_residual_seconds"], -1e-9)
            self.assertAlmostEqual(
                timing["total_seconds"],
                timing["prefix_replay_seconds"]
                + timing["branch_simulator_step_seconds"]
                + timing["state_snapshot_seconds"]
                + timing["state_restore_seconds"]
                + timing["root_initial_sweep_orchestration_seconds"]
                + timing["root_search_setup_seconds"]
                + timing["root_adaptive_visit_orchestration_seconds"]
                + timing["root_search_finalization_seconds"]
                + timing["branch_action_validation_seconds"]
                + timing["post_branch_history_seconds"]
                + timing["belief_world_materialization_seconds"]
                + timing["opponent_scenario_planning_seconds"]
                + timing["root_policy_setup_seconds"]
                + timing["direct_prefix_construction_seconds"]
                + timing["scenario_dispatch_orchestration_seconds"]
                + timing["policy_value_evaluation_seconds"]
                + timing["rollout_tail_seconds"]
                + timing["raw_residual_seconds"],
            )
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

    def test_benchmark_root_puct_counterfactual_requires_search_player_policy(self) -> None:
        with self.assertRaisesRegex(ValueError, "search_player"):
            benchmark_root_puct_counterfactual_rollouts(
                env_factory=ImmediateOutcomeEnv,
                policies={"p1": FixedPolicy(0, policy_id="fixed-p1")},
                rollout_config=RolloutConfig(max_decision_rounds=3),
                games=1,
                value_fn=lambda history: 0.0,
                prior_fn=lambda history: (1.0,) + (0.0,) * (ACTION_COUNT - 1),
                search_player="p2",
            )


if __name__ == "__main__":
    unittest.main()
