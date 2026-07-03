import random
import unittest

from pokezero.actions import ACTION_COUNT
from pokezero.env import BattleStartOverride, StepResult, TerminalState
from pokezero.observation import PokeZeroObservationV0
from pokezero.policy import PolicyContext, PolicyDecision, RandomLegalPolicy
from pokezero.rollout import RolloutConfig, RolloutDriver
from pokezero.search import puct_branch_search
from pokezero.search_policy import (
    OpponentActionScenario,
    RootPUCTSearchPolicy,
    _aggregate_scenario_searches,
    _opponent_scenario_replay_legality_error,
    greedy_opponent_action_planner,
    policy_opponent_action_planner,
    prior_top_k_opponent_action_scenario_planner,
)
from pokezero.trajectory import BattleTrajectory


def _mask(*legal_indices: int) -> tuple[bool, ...]:
    return tuple(index in set(legal_indices) for index in range(ACTION_COUNT))


def _observation(*legal_indices: int) -> PokeZeroObservationV0:
    return PokeZeroObservationV0(
        categorical_ids=(),
        numeric_features=(),
        token_type_ids=(),
        attention_mask=(),
        legal_action_mask=_mask(*legal_indices),
    )


class FixedPolicy:
    def __init__(
        self,
        action_index: int,
        *,
        policy_id: str = "fixed",
        value_estimate: float | None = None,
    ) -> None:
        self.action_index = action_index
        self.policy_id = policy_id
        self.value_estimate = value_estimate

    def select_action(self, observation: PokeZeroObservationV0, *, rng) -> PolicyDecision:
        return PolicyDecision(
            action_index=self.action_index,
            policy_id=self.policy_id,
            action_probability=1.0,
            value_estimate=self.value_estimate,
        )


class ResettableFixedPolicy(FixedPolicy):
    def __init__(self, action_index: int, *, policy_id: str = "fixed") -> None:
        super().__init__(action_index, policy_id=policy_id)
        self.observations: list[PokeZeroObservationV0] = []
        self.reset_calls = 0

    def select_action(self, observation: PokeZeroObservationV0, *, rng) -> PolicyDecision:
        self.observations.append(observation)
        return super().select_action(observation, rng=rng)

    def reset(self) -> None:
        self.reset_calls += 1


class ContextRecordingPolicy:
    policy_id = "context-recorder"

    def __init__(self, action_index: int) -> None:
        self.action_index = action_index
        self.contexts: list[PolicyContext] = []

    def select_action(self, observation: PokeZeroObservationV0, *, rng) -> PolicyDecision:
        raise AssertionError("context-aware planner should call select_action_with_context")

    def select_action_with_context(self, context: PolicyContext, *, rng) -> PolicyDecision:
        self.contexts.append(context)
        return PolicyDecision(action_index=self.action_index, policy_id=self.policy_id, action_probability=1.0)


class ImmediateOutcomeEnv:
    def __init__(self, *, label: str) -> None:
        self.label = label
        self.reset_calls: list[tuple[int, str]] = []
        self.step_calls: list[dict[str, int]] = []
        self.all_step_calls: list[dict[str, int]] = []
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
        self.all_step_calls.append(dict(actions))
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


class StartOverrideOutcomeEnv(ImmediateOutcomeEnv):
    def __init__(self, *, label: str) -> None:
        super().__init__(label=label)
        self.start_overrides: list[BattleStartOverride] = []

    def reset_with_start_override(
        self,
        *,
        seed: int,
        format_id: str | None = None,
        start_override: BattleStartOverride,
    ) -> None:
        self.start_overrides.append(start_override)
        self.reset(seed=seed, format_id=format_id or start_override.format_id)


class RejectingStartOverrideOutcomeEnv(StartOverrideOutcomeEnv):
    def __init__(self, *, label: str) -> None:
        super().__init__(label=label)
        self.rejected_start_overrides = 0

    def reset_with_start_override(
        self,
        *,
        seed: int,
        format_id: str | None = None,
        start_override: BattleStartOverride,
    ) -> None:
        packed_teams = tuple(start_override.player_teams.values())
        if any("Badmon" in packed_team for packed_team in packed_teams):
            self.rejected_start_overrides += 1
            raise ValueError(
                "start override does not reproduce recorded replay prefix observations "
                "for decision round 0: p1."
            )
        super().reset_with_start_override(
            seed=seed,
            format_id=format_id,
            start_override=start_override,
        )


class TwoOpponentActionEnv(ImmediateOutcomeEnv):
    def observe(self, player: str) -> PokeZeroObservationV0:
        return _observation(0, 1)


class StrictOpponentActionEnv(ImmediateOutcomeEnv):
    def step(self, actions: dict[str, int]) -> StepResult:
        p2_action = int(actions["p2"])
        if p2_action != 0:
            raise ValueError(f"p2: action_index {p2_action} is not legal for the current request.")
        return super().step(actions)


class DelayedOutcomeEnv:
    def __init__(self, winners_after_branch: dict[int, str | None]) -> None:
        self.winners_after_branch = winners_after_branch
        self.all_step_calls: list[dict[str, int]] = []
        self.reset_calls: list[tuple[int, str]] = []
        self._terminal: TerminalState | None = None
        self._requested = ("p1", "p2")
        self._branch_action: int | None = None
        self.closed = False

    def reset(self, *, seed: int, format_id: str = "gen3randombattle") -> None:
        self.reset_calls.append((seed, format_id))
        self._terminal = None
        self._requested = ("p1", "p2")
        self._branch_action = None

    def observe(self, player: str) -> PokeZeroObservationV0:
        return _observation(0, 1) if player == "p1" else _observation(0)

    def legal_actions(self, player: str) -> tuple[bool, ...]:
        return self.observe(player).legal_action_mask

    def requested_players(self) -> tuple[str, ...]:
        return self._requested

    def step(self, actions: dict[str, int]) -> StepResult:
        self.all_step_calls.append(dict(actions))
        if self._branch_action is None:
            self._branch_action = int(actions["p1"])
            self._requested = ("p1", "p2")
            return StepResult(
                observations={"p1": _observation(0, 1), "p2": _observation(0)},
                rewards={"p1": 0.0, "p2": 0.0},
                terminal=None,
                requested_players=self._requested,
            )
        winner = self.winners_after_branch[self._branch_action]
        self._terminal = TerminalState(winner=winner, turn_count=len(self.all_step_calls))
        self._requested = ()
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


class RootPUCTSearchPolicyTest(unittest.TestCase):
    def test_greedy_opponent_action_planner_uses_player_local_history(self) -> None:
        observed_history_lengths: list[int] = []

        def prior_fn(history: tuple[PokeZeroObservationV0, ...]) -> tuple[float, ...]:
            observed_history_lengths.append(len(history))
            return (0.1, 0.7, 0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

        planner = greedy_opponent_action_planner(prior_fn)
        context = PolicyContext(
            player_id="p1",
            decision_round_index=0,
            battle_id="planner",
            format_id="gen3randombattle",
            seed=7,
            observation=_observation(0, 1),
            requested_players=("p1", "p2"),
            trajectory=BattleTrajectory(battle_id="planner", format_id="gen3randombattle", seed=7),
        )

        self.assertEqual(planner(context, random.Random(1)), {"p2": 1})
        self.assertEqual(observed_history_lengths, [1])

    def test_greedy_opponent_action_planner_masks_requested_opponent_legal_actions(self) -> None:
        planner = greedy_opponent_action_planner(lambda history: (0.1, 0.4, 0.9) + (0.0,) * 6)
        context = PolicyContext(
            player_id="p1",
            decision_round_index=0,
            battle_id="planner",
            format_id="gen3randombattle",
            seed=7,
            observation=_observation(0, 1),
            requested_players=("p1", "p2"),
            trajectory=BattleTrajectory(battle_id="planner", format_id="gen3randombattle", seed=7),
            requested_legal_action_masks={"p1": _mask(0, 1), "p2": _mask(1)},
        )

        self.assertEqual(planner(context, random.Random(1)), {"p2": 1})

    def test_greedy_opponent_action_planner_rejects_bad_prior_width(self) -> None:
        planner = greedy_opponent_action_planner(lambda history: (1.0,))
        context = PolicyContext(
            player_id="p1",
            decision_round_index=0,
            battle_id="planner",
            format_id="gen3randombattle",
            seed=7,
            observation=_observation(0, 1),
            requested_players=("p1", "p2"),
            trajectory=BattleTrajectory(battle_id="planner", format_id="gen3randombattle", seed=7),
        )

        with self.assertRaisesRegex(ValueError, "opponent action priors"):
            planner(context, random.Random(1))

    def test_prior_top_k_opponent_action_scenario_planner_uses_player_local_priors(self) -> None:
        planner = prior_top_k_opponent_action_scenario_planner(
            lambda history: (0.1, 0.7, 0.2) + (0.0,) * (ACTION_COUNT - 3),
            scenario_count=2,
        )
        context = PolicyContext(
            player_id="p1",
            decision_round_index=0,
            battle_id="planner",
            format_id="gen3randombattle",
            seed=7,
            observation=_observation(0, 1),
            requested_players=("p1", "p2"),
            trajectory=BattleTrajectory(battle_id="planner", format_id="gen3randombattle", seed=7),
            requested_legal_action_masks={"p1": _mask(0, 1), "p2": _mask(1, 2)},
        )

        scenarios = planner(context, random.Random(1))

        self.assertEqual(getattr(planner, "planner_id"), "checkpoint-top2")
        self.assertEqual([dict(scenario.actions) for scenario in scenarios], [{"p2": 1}, {"p2": 2}])
        self.assertAlmostEqual(scenarios[0].weight, 0.7 / 0.9)
        self.assertAlmostEqual(scenarios[1].weight, 0.2 / 0.9)
        self.assertEqual([scenario.label for scenario in scenarios], ["p2:1", "p2:2"])

    def test_policy_opponent_action_planner_uses_requested_opponent_observation(self) -> None:
        opponent_policy = ResettableFixedPolicy(1, policy_id="benchmark-opponent")
        planner = policy_opponent_action_planner({"p2": opponent_policy}, planner_id="benchmark")
        opponent_observation = _observation(1)
        context = PolicyContext(
            player_id="p1",
            decision_round_index=0,
            battle_id="planner",
            format_id="gen3randombattle",
            seed=7,
            observation=_observation(0),
            requested_players=("p1", "p2"),
            trajectory=BattleTrajectory(battle_id="planner", format_id="gen3randombattle", seed=7),
            requested_legal_action_masks={"p1": _mask(0), "p2": _mask(1)},
            requested_observations={"p1": _observation(0), "p2": opponent_observation},
        )

        self.assertEqual(planner(context, random.Random(1)), {"p2": 1})
        self.assertEqual(opponent_policy.observations, [opponent_observation])
        planner.reset()
        self.assertEqual(opponent_policy.reset_calls, 1)

    def test_policy_opponent_action_planner_supports_context_aware_policy(self) -> None:
        opponent_policy = ContextRecordingPolicy(1)
        planner = policy_opponent_action_planner({"p2": opponent_policy}, planner_id="benchmark")
        trajectory = BattleTrajectory(battle_id="planner", format_id="gen3randombattle", seed=7)
        opponent_observation = _observation(1)
        context = PolicyContext(
            player_id="p1",
            decision_round_index=3,
            battle_id="planner",
            format_id="gen3randombattle",
            seed=7,
            observation=_observation(0),
            requested_players=("p1", "p2"),
            trajectory=trajectory,
            requested_legal_action_masks={"p1": _mask(0), "p2": _mask(1)},
            requested_observations={"p1": _observation(0), "p2": opponent_observation},
        )

        self.assertEqual(planner(context, random.Random(1)), {"p2": 1})
        self.assertEqual(len(opponent_policy.contexts), 1)
        opponent_context = opponent_policy.contexts[0]
        self.assertEqual(opponent_context.player_id, "p2")
        self.assertEqual(opponent_context.decision_round_index, 3)
        self.assertIs(opponent_context.observation, opponent_observation)
        self.assertIs(opponent_context.trajectory, trajectory)
        self.assertEqual(opponent_context.requested_players, ("p1", "p2"))

    def test_root_puct_policy_selects_search_action_using_separate_branch_env(self) -> None:
        branch_envs: list[ImmediateOutcomeEnv] = []

        def branch_env_factory() -> ImmediateOutcomeEnv:
            env = ImmediateOutcomeEnv(label=f"branch-{len(branch_envs)}")
            branch_envs.append(env)
            return env

        policy = RootPUCTSearchPolicy(
            env_factory=branch_env_factory,
            rollout_config=RolloutConfig(max_decision_rounds=3),
            value_fn=lambda history: 0.0,
            prior_fn=lambda history: (0.5, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
            opponent_action_planner=lambda context, rng: {"p2": 0},
            cpuct=0.0,
            root_visit_budget=None,
        )
        live_env = ImmediateOutcomeEnv(label="live")

        result = RolloutDriver(
            env=live_env,
            policies={"p1": policy, "p2": FixedPolicy(0, policy_id="fixed-p2")},
            config=RolloutConfig(max_decision_rounds=3),
        ).run(seed=91, battle_id="search-policy")

        self.assertEqual(result.terminal.winner, "p1")
        self.assertEqual(result.trajectory.steps_for_player("p1")[0].action_index, 1)
        metadata = result.trajectory.steps_for_player("p1")[0].metadata
        self.assertEqual(metadata["policy_id"], "root-puct-search")
        self.assertFalse(metadata["root_puct_fallback"])
        self.assertEqual(metadata["root_puct_candidate_count"], 2)
        self.assertEqual(metadata["root_puct_search_action"], 1)
        self.assertEqual(metadata["root_puct_prior_action"], 0)
        self.assertTrue(metadata["root_puct_selected_changed_prior_action"])
        self.assertTrue(metadata["root_puct_pre_gate_changed_prior_action"])
        self.assertEqual(metadata["root_puct_selected_action_visits"], 1)
        self.assertEqual(metadata["root_puct_prior_action_visits"], 1)
        self.assertEqual(metadata["root_puct_opponent_actions"], {"p2": 0})
        self.assertTrue(metadata["root_puct_opponent_actions_legality_checked"])
        self.assertEqual(live_env.step_calls, [{"p1": 1, "p2": 0}])
        self.assertEqual(len(branch_envs), 1)
        self.assertTrue(branch_envs[0].closed)
        self.assertEqual(branch_envs[0].all_step_calls, [{"p1": 0, "p2": 0}, {"p1": 1, "p2": 0}])

    def test_root_puct_policy_can_plan_root_opponent_action_from_policy(self) -> None:
        planner_policy = ResettableFixedPolicy(0, policy_id="benchmark-opponent")
        policy = RootPUCTSearchPolicy(
            env_factory=lambda: ImmediateOutcomeEnv(label="branch"),
            rollout_config=RolloutConfig(max_decision_rounds=3),
            value_fn=lambda history: 0.0,
            prior_fn=lambda history: (0.5, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
            opponent_action_planner=policy_opponent_action_planner({"p2": planner_policy}, planner_id="benchmark"),
            cpuct=0.0,
        )

        result = RolloutDriver(
            env=ImmediateOutcomeEnv(label="live"),
            policies={"p1": policy, "p2": FixedPolicy(0, policy_id="fixed-p2")},
            config=RolloutConfig(max_decision_rounds=3),
        ).run(seed=91, battle_id="search-policy")

        self.assertEqual(result.terminal.winner, "p1")
        metadata = result.trajectory.steps_for_player("p1")[0].metadata
        self.assertEqual(metadata["root_puct_opponent_actions"], {"p2": 0})
        self.assertEqual(metadata["root_puct_opponent_action_policy"], "benchmark")
        self.assertEqual(len(planner_policy.observations), 1)

    def test_root_puct_policy_can_average_checkpoint_prior_opponent_action_scenarios(self) -> None:
        branch_envs: list[TwoOpponentActionEnv] = []

        def branch_env_factory() -> TwoOpponentActionEnv:
            env = TwoOpponentActionEnv(label=f"branch-{len(branch_envs)}")
            branch_envs.append(env)
            return env

        policy = RootPUCTSearchPolicy(
            env_factory=branch_env_factory,
            rollout_config=RolloutConfig(max_decision_rounds=3),
            value_fn=lambda history: 0.0,
            prior_fn=lambda history: (0.5, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
            opponent_action_planner=greedy_opponent_action_planner(
                lambda history: (0.6, 0.4, 0.0) + (0.0,) * (ACTION_COUNT - 3)
            ),
            opponent_action_scenario_planner=prior_top_k_opponent_action_scenario_planner(
                lambda history: (0.6, 0.4, 0.0) + (0.0,) * (ACTION_COUNT - 3),
                scenario_count=2,
            ),
            cpuct=0.0,
            root_visit_budget=2,
            root_time_budget_seconds=8.0,
        )
        live_env = TwoOpponentActionEnv(label="live")

        result = RolloutDriver(
            env=live_env,
            policies={"p1": policy, "p2": FixedPolicy(0, policy_id="fixed-p2")},
            config=RolloutConfig(max_decision_rounds=3),
        ).run(seed=91, battle_id="search-policy")

        self.assertEqual(result.terminal.winner, "p1")
        step = result.trajectory.steps_for_player("p1")[0]
        self.assertEqual(step.action_index, 1)
        metadata = step.metadata
        self.assertFalse(metadata["root_puct_fallback"])
        self.assertEqual(metadata["root_puct_opponent_action_policy"], "checkpoint-top2")
        self.assertEqual(metadata["root_puct_opponent_action_scenario_count"], 2)
        self.assertEqual(metadata["root_puct_root_time_budget_seconds"], 8.0)
        self.assertEqual(metadata["root_puct_root_scenario_time_budget_seconds"], 4.0)
        self.assertEqual(
            metadata["root_puct_opponent_action_scenarios"],
            [
                {"label": "p2:0", "weight": 0.6, "actions": {"p2": 0}},
                {"label": "p2:1", "weight": 0.4, "actions": {"p2": 1}},
            ],
        )
        self.assertEqual(branch_envs[0].all_step_calls, [
            {"p1": 0, "p2": 0},
            {"p1": 1, "p2": 0},
            {"p1": 0, "p2": 1},
            {"p1": 1, "p2": 1},
        ])

    def test_root_puct_policy_passes_start_override_to_branch_search(self) -> None:
        branch_envs: list[StartOverrideOutcomeEnv] = []
        sampled_overrides: list[BattleStartOverride] = []

        def branch_env_factory() -> StartOverrideOutcomeEnv:
            env = StartOverrideOutcomeEnv(label=f"branch-{len(branch_envs)}")
            branch_envs.append(env)
            return env

        def start_override_planner(
            context: PolicyContext,
            scenario: OpponentActionScenario,
            scenario_index: int,
            rng: random.Random,
        ):
            del context, rng
            self.assertEqual(scenario.actions, {"p2": 0})
            self.assertEqual(scenario_index, 0)

            def sample_override() -> BattleStartOverride:
                override = BattleStartOverride(
                    player_teams={
                        "p1": "Charizard||||Tackle|||||||",
                        "p2": f"Xatu||||Psychic|||||||{len(sampled_overrides)}",
                    }
                )
                sampled_overrides.append(override)
                return override

            return sample_override

        policy = RootPUCTSearchPolicy(
            env_factory=branch_env_factory,
            rollout_config=RolloutConfig(max_decision_rounds=3),
            value_fn=lambda history: 0.0,
            prior_fn=lambda history: (0.5, 0.5) + (0.0,) * (ACTION_COUNT - 2),
            opponent_action_planner=lambda context, rng: {"p2": 0},
            cpuct=0.0,
            root_visit_budget=3,
            start_override_planner=start_override_planner,
        )

        result = RolloutDriver(
            env=ImmediateOutcomeEnv(label="live"),
            policies={"p1": policy, "p2": FixedPolicy(0, policy_id="fixed-p2")},
            config=RolloutConfig(max_decision_rounds=3),
        ).run(seed=91, battle_id="search-policy")

        step = result.trajectory.steps_for_player("p1")[0]
        self.assertEqual(step.metadata["root_puct_start_override_sources_used"], 1)
        self.assertEqual(branch_envs[0].start_overrides, sampled_overrides)
        self.assertEqual(len(sampled_overrides), 3)

    def test_root_puct_policy_skips_hidden_opponent_scenarios_replay_rejects(self) -> None:
        branch_envs: list[StrictOpponentActionEnv] = []

        def branch_env_factory() -> StrictOpponentActionEnv:
            env = StrictOpponentActionEnv(label=f"branch-{len(branch_envs)}")
            branch_envs.append(env)
            return env

        def scenario_planner(context: PolicyContext, rng: random.Random) -> tuple[OpponentActionScenario, ...]:
            del context, rng
            return (
                OpponentActionScenario(actions={"p2": 2}, weight=0.75, label="illegal-hidden"),
                OpponentActionScenario(actions={"p2": 0}, weight=0.25, label="legal-hidden"),
            )

        scenario_planner.planner_id = "test-hidden-top2"  # type: ignore[attr-defined]
        policy = RootPUCTSearchPolicy(
            env_factory=branch_env_factory,
            rollout_config=RolloutConfig(max_decision_rounds=3),
            value_fn=lambda history: 0.0,
            prior_fn=lambda history: (0.5, 0.5) + (0.0,) * (ACTION_COUNT - 2),
            opponent_action_scenario_planner=scenario_planner,
            cpuct=0.0,
            root_visit_budget=None,
        )
        context = PolicyContext(
            player_id="p1",
            decision_round_index=0,
            battle_id="search-policy",
            format_id="gen3randombattle",
            seed=91,
            observation=_observation(0, 1),
            requested_players=("p1", "p2"),
            trajectory=BattleTrajectory(battle_id="search-policy", format_id="gen3randombattle", seed=91),
            requested_legal_action_masks={"p1": _mask(0, 1)},
        )

        decision = policy.select_action_with_context(context, rng=random.Random(1))

        self.assertEqual(decision.action_index, 1)
        metadata = decision.metadata
        self.assertFalse(metadata["root_puct_fallback"])
        self.assertEqual(metadata["root_puct_opponent_action_policy"], "test-hidden-top2")
        self.assertFalse(metadata["root_puct_opponent_actions_legality_checked"])
        self.assertEqual(metadata["root_puct_opponent_action_scenarios_generated"], 2)
        self.assertEqual(metadata["root_puct_opponent_action_scenarios_skipped"], 1)
        self.assertEqual(metadata["root_puct_opponent_action_scenario_count"], 1)
        self.assertEqual(
            metadata["root_puct_opponent_action_scenarios"],
            [{"label": "legal-hidden", "weight": 1.0, "actions": {"p2": 0}}],
        )
        self.assertEqual(
            metadata["root_puct_opponent_action_skipped_scenarios"],
            [
                {
                    "label": "illegal-hidden",
                    "weight": 0.75,
                    "actions": {"p2": 2},
                    "reason": "p2: action_index 2 is not legal for the current request.",
                }
            ],
        )
        self.assertEqual(branch_envs[0].all_step_calls, [
            {"p1": 0, "p2": 0},
            {"p1": 1, "p2": 0},
        ])

    def test_root_puct_policy_skips_start_override_consistency_mismatch_scenario(self) -> None:
        branch_envs: list[StartOverrideOutcomeEnv] = []

        def branch_env_factory() -> StartOverrideOutcomeEnv:
            env = StartOverrideOutcomeEnv(label=f"branch-{len(branch_envs)}")
            branch_envs.append(env)
            return env

        def scenario_planner(context: PolicyContext, rng: random.Random) -> tuple[OpponentActionScenario, ...]:
            del context, rng
            return (
                OpponentActionScenario(actions={"p2": 0}, weight=0.5, label="bad-override"),
                OpponentActionScenario(actions={"p2": 0}, weight=0.5, label="plain"),
            )

        def start_override_planner(
            context: PolicyContext,
            scenario: OpponentActionScenario,
            scenario_index: int,
            rng: random.Random,
        ):
            del context, rng
            self.assertIn(scenario_index, {0, 1})
            if scenario.label != "bad-override":
                return None
            return BattleStartOverride(
                player_teams={"p1": "Charizard||||Tackle|||||||", "p2": "Xatu||||Psychic|||||||"}
            )

        policy = RootPUCTSearchPolicy(
            env_factory=branch_env_factory,
            rollout_config=RolloutConfig(max_decision_rounds=3),
            value_fn=lambda history: 0.0,
            prior_fn=lambda history: (0.0, 1.0) + (0.0,) * (ACTION_COUNT - 2),
            opponent_action_scenario_planner=scenario_planner,
            cpuct=0.0,
            start_override_planner=start_override_planner,
        )
        context = PolicyContext(
            player_id="p1",
            decision_round_index=0,
            battle_id="search-policy",
            format_id="gen3randombattle",
            seed=91,
            observation=_observation(1),
            requested_players=("p1", "p2"),
            trajectory=BattleTrajectory(battle_id="search-policy", format_id="gen3randombattle", seed=91),
            requested_legal_action_masks={"p1": _mask(1)},
        )

        decision = policy.select_action_with_context(context, rng=random.Random(1))

        self.assertEqual(decision.action_index, 1)
        metadata = decision.metadata
        self.assertFalse(metadata["root_puct_fallback"])
        self.assertEqual(metadata["root_puct_opponent_action_scenarios_generated"], 2)
        self.assertEqual(metadata["root_puct_opponent_action_scenarios_skipped"], 1)
        self.assertEqual(metadata["root_puct_start_override_sources_used"], 0)
        skip_reason = metadata["root_puct_opponent_action_skipped_scenarios"][0]["reason"]
        self.assertIn(
            "start override does not reproduce recorded replay prefix observations "
            "for decision round 0: p1.",
            skip_reason,
        )
        self.assertIn("legal_action_mask", skip_reason)

    def test_root_puct_policy_retries_start_override_replay_rejections(self) -> None:
        branch_envs: list[RejectingStartOverrideOutcomeEnv] = []
        planner_calls = 0

        def branch_env_factory() -> RejectingStartOverrideOutcomeEnv:
            env = RejectingStartOverrideOutcomeEnv(label=f"branch-{len(branch_envs)}")
            branch_envs.append(env)
            return env

        def start_override_planner(
            context: PolicyContext,
            scenario: OpponentActionScenario,
            scenario_index: int,
            rng: random.Random,
        ):
            nonlocal planner_calls
            del context, scenario, scenario_index, rng
            planner_calls += 1
            species = "Badmon" if planner_calls == 1 else "Xatu"
            return BattleStartOverride(
                player_teams={
                    "p1": "Charizard||||Tackle|||||||",
                    "p2": f"{species}||||Psychic|||||||",
                }
            )

        policy = RootPUCTSearchPolicy(
            env_factory=branch_env_factory,
            rollout_config=RolloutConfig(max_decision_rounds=3),
            value_fn=lambda history: 0.0,
            prior_fn=lambda history: (0.0, 1.0) + (0.0,) * (ACTION_COUNT - 2),
            opponent_action_planner=lambda context, rng: {"p2": 0},
            cpuct=0.0,
            start_override_planner=start_override_planner,
            start_override_attempts=2,
        )
        context = PolicyContext(
            player_id="p1",
            decision_round_index=0,
            battle_id="search-policy",
            format_id="gen3randombattle",
            seed=91,
            observation=_observation(0, 1),
            requested_players=("p1", "p2"),
            trajectory=BattleTrajectory(battle_id="search-policy", format_id="gen3randombattle", seed=91),
            requested_legal_action_masks={"p1": _mask(0, 1)},
        )

        decision = policy.select_action_with_context(context, rng=random.Random(1))

        self.assertEqual(decision.action_index, 1)
        self.assertFalse(decision.metadata["root_puct_fallback"])
        self.assertEqual(planner_calls, 2)
        self.assertEqual(decision.metadata["root_puct_start_override_attempts"], 2)
        self.assertEqual(decision.metadata["root_puct_start_override_attempts_used"], 2)
        self.assertEqual(decision.metadata["root_puct_start_override_sources_used"], 1)
        self.assertEqual(branch_envs[0].rejected_start_overrides, 1)

    def test_root_puct_policy_reports_start_override_attempts_on_rejected_fallback(self) -> None:
        def start_override_planner(
            context: PolicyContext,
            scenario: OpponentActionScenario,
            scenario_index: int,
            rng: random.Random,
        ):
            del context, scenario, scenario_index, rng
            return BattleStartOverride(
                player_teams={
                    "p1": "Charizard||||Tackle|||||||",
                    "p2": "Badmon||||Psychic|||||||",
                }
            )

        policy = RootPUCTSearchPolicy(
            env_factory=lambda: RejectingStartOverrideOutcomeEnv(label="branch"),
            rollout_config=RolloutConfig(max_decision_rounds=3),
            value_fn=lambda history: 0.0,
            prior_fn=lambda history: (1.0,) + (0.0,) * (ACTION_COUNT - 1),
            opponent_action_planner=lambda context, rng: {"p2": 0},
            fallback_policy=FixedPolicy(1, policy_id="fallback-fixed"),
            allow_fallback=True,
            cpuct=0.0,
            start_override_planner=start_override_planner,
            start_override_attempts=3,
        )
        context = PolicyContext(
            player_id="p1",
            decision_round_index=0,
            battle_id="search-policy",
            format_id="gen3randombattle",
            seed=91,
            observation=_observation(0, 1),
            requested_players=("p1", "p2"),
            trajectory=BattleTrajectory(battle_id="search-policy", format_id="gen3randombattle", seed=91),
            requested_legal_action_masks={"p1": _mask(0, 1)},
        )

        decision = policy.select_action_with_context(context, rng=random.Random(1))

        self.assertTrue(decision.metadata["root_puct_fallback"])
        self.assertEqual(decision.metadata["root_puct_start_override_attempts"], 3)
        self.assertEqual(decision.metadata["root_puct_start_override_attempts_used"], 3)
        self.assertEqual(decision.metadata["root_puct_start_override_sources_used"], 0)

    def test_opponent_scenario_replay_legality_classifies_request_drift(self) -> None:
        scenario = OpponentActionScenario(actions={"p2": 0})
        message = (
            "replay actions for decision round 4 do not match environment request "
            "(unexpected players: p1)."
        )

        self.assertEqual(
            _opponent_scenario_replay_legality_error(ValueError(message), scenario),
            message,
        )

    def test_opponent_scenario_replay_legality_classifies_illegal_prefix_action(self) -> None:
        scenario = OpponentActionScenario(actions={"p2": 0})
        message = "p1: action_index 3 is not legal for the current request."

        self.assertEqual(
            _opponent_scenario_replay_legality_error(ValueError(message), scenario),
            message,
        )

    def test_root_puct_policy_value_gate_keeps_prior_action_without_sufficient_value_lift(self) -> None:
        policy = RootPUCTSearchPolicy(
            env_factory=lambda: ImmediateOutcomeEnv(label="branch"),
            rollout_config=RolloutConfig(max_decision_rounds=3),
            value_fn=lambda history: 0.0,
            prior_fn=lambda history: (0.9, 0.1) + (0.0,) * (ACTION_COUNT - 2),
            opponent_action_planner=lambda context, rng: {"p2": 0},
            cpuct=0.0,
            minimum_value_improvement=3.0,
        )

        result = RolloutDriver(
            env=ImmediateOutcomeEnv(label="live"),
            policies={"p1": policy, "p2": FixedPolicy(0, policy_id="fixed-p2")},
            config=RolloutConfig(max_decision_rounds=3),
        ).run(seed=94, battle_id="search-policy")

        step = result.trajectory.steps_for_player("p1")[0]
        self.assertEqual(step.action_index, 0)
        self.assertTrue(step.metadata["root_puct_value_gate_used"])
        self.assertEqual(step.metadata["root_puct_pre_gate_action"], 1)
        self.assertEqual(step.metadata["root_puct_search_action"], 1)
        self.assertEqual(step.metadata["root_puct_prior_action"], 0)
        self.assertFalse(step.metadata["root_puct_selected_changed_prior_action"])
        self.assertTrue(step.metadata["root_puct_pre_gate_changed_prior_action"])
        self.assertEqual(step.metadata["root_puct_minimum_value_improvement"], 3.0)

    def test_root_puct_policy_default_selection_mode_uses_most_visited_branch(self) -> None:
        policy = RootPUCTSearchPolicy(
            env_factory=lambda: ImmediateOutcomeEnv(label="branch"),
            rollout_config=RolloutConfig(max_decision_rounds=3),
            value_fn=lambda history: 0.0,
            prior_fn=lambda history: (0.9, 0.1) + (0.0,) * (ACTION_COUNT - 2),
            opponent_action_planner=lambda context, rng: {"p2": 0},
            cpuct=4.0,
        )

        result = RolloutDriver(
            env=ImmediateOutcomeEnv(label="live"),
            policies={"p1": policy, "p2": FixedPolicy(0, policy_id="fixed-p2")},
            config=RolloutConfig(max_decision_rounds=3),
        ).run(seed=96, battle_id="search-policy")

        step = result.trajectory.steps_for_player("p1")[0]
        self.assertEqual(step.action_index, 1)
        self.assertEqual(step.metadata["root_puct_selection_mode"], "visits")
        self.assertEqual(step.metadata["root_puct_total_visits"], 16)
        self.assertEqual(step.metadata["root_puct_prior_action"], 0)
        self.assertGreater(
            step.metadata["root_puct_selected_action_visits"],
            step.metadata["root_puct_prior_action_visits"],
        )
        self.assertTrue(step.metadata["root_puct_selected_changed_prior_action"])

    def test_root_puct_policy_puct_selection_mode_uses_final_exploration_score(self) -> None:
        policy = RootPUCTSearchPolicy(
            env_factory=lambda: ImmediateOutcomeEnv(label="branch"),
            rollout_config=RolloutConfig(max_decision_rounds=3),
            value_fn=lambda history: 0.0,
            prior_fn=lambda history: (0.9, 0.1) + (0.0,) * (ACTION_COUNT - 2),
            opponent_action_planner=lambda context, rng: {"p2": 0},
            cpuct=4.0,
            selection_mode="puct",
            root_visit_budget=None,
        )

        result = RolloutDriver(
            env=ImmediateOutcomeEnv(label="live"),
            policies={"p1": policy, "p2": FixedPolicy(0, policy_id="fixed-p2")},
            config=RolloutConfig(max_decision_rounds=3),
        ).run(seed=96, battle_id="search-policy")

        step = result.trajectory.steps_for_player("p1")[0]
        self.assertEqual(step.action_index, 0)
        self.assertEqual(step.metadata["root_puct_selection_mode"], "puct")
        self.assertGreater(step.metadata["root_puct_selected_score"], 1.0)

    def test_root_puct_policy_value_selection_mode_uses_highest_value_branch(self) -> None:
        policy = RootPUCTSearchPolicy(
            env_factory=lambda: ImmediateOutcomeEnv(label="branch"),
            rollout_config=RolloutConfig(max_decision_rounds=3),
            value_fn=lambda history: 0.0,
            prior_fn=lambda history: (0.9, 0.1) + (0.0,) * (ACTION_COUNT - 2),
            opponent_action_planner=lambda context, rng: {"p2": 0},
            cpuct=4.0,
            selection_mode="value",
        )

        result = RolloutDriver(
            env=ImmediateOutcomeEnv(label="live"),
            policies={"p1": policy, "p2": FixedPolicy(0, policy_id="fixed-p2")},
            config=RolloutConfig(max_decision_rounds=3),
        ).run(seed=96, battle_id="search-policy")

        step = result.trajectory.steps_for_player("p1")[0]
        self.assertEqual(step.action_index, 1)
        self.assertEqual(step.metadata["root_puct_selection_mode"], "value")
        self.assertEqual(step.metadata["root_puct_selected_value"], 1.0)
        self.assertNotIn("root_puct_leaf_rollout_rounds", step.metadata)

    def test_root_puct_policy_visits_selection_mode_uses_most_visited_branch(self) -> None:
        policy = RootPUCTSearchPolicy(
            env_factory=lambda: ImmediateOutcomeEnv(label="branch"),
            rollout_config=RolloutConfig(max_decision_rounds=3),
            value_fn=lambda history: 0.0,
            prior_fn=lambda history: (0.9, 0.1) + (0.0,) * (ACTION_COUNT - 2),
            opponent_action_planner=lambda context, rng: {"p2": 0},
            cpuct=8.0,
            selection_mode="visits",
            root_visit_budget=5,
        )

        result = RolloutDriver(
            env=ImmediateOutcomeEnv(label="live"),
            policies={"p1": policy, "p2": FixedPolicy(0, policy_id="fixed-p2")},
            config=RolloutConfig(max_decision_rounds=3),
        ).run(seed=96, battle_id="search-policy")

        step = result.trajectory.steps_for_player("p1")[0]
        self.assertEqual(step.action_index, 0)
        self.assertEqual(step.metadata["root_puct_selection_mode"], "visits")
        self.assertEqual(step.metadata["root_puct_total_visits"], 5)

    def test_root_puct_policy_visits_selection_tie_prefers_prior(self) -> None:
        policy = RootPUCTSearchPolicy(
            env_factory=lambda: ImmediateOutcomeEnv(label="branch"),
            rollout_config=RolloutConfig(max_decision_rounds=3),
            value_fn=lambda history: 0.0,
            prior_fn=lambda history: (0.9, 0.1) + (0.0,) * (ACTION_COUNT - 2),
            opponent_action_planner=lambda context, rng: {"p2": 0},
            cpuct=0.0,
            selection_mode="visits",
            root_visit_budget=2,
        )

        result = RolloutDriver(
            env=ImmediateOutcomeEnv(label="live"),
            policies={"p1": policy, "p2": FixedPolicy(0, policy_id="fixed-p2")},
            config=RolloutConfig(max_decision_rounds=3),
        ).run(seed=96, battle_id="search-policy")

        step = result.trajectory.steps_for_player("p1")[0]
        self.assertEqual(step.action_index, 0)
        self.assertEqual(step.metadata["root_puct_selection_mode"], "visits")
        self.assertEqual(step.metadata["root_puct_search_action"], 0)
        self.assertEqual(step.metadata["root_puct_prior_action"], 0)
        self.assertEqual(step.metadata["root_puct_selected_action_visits"], 1)
        self.assertEqual(step.metadata["root_puct_prior_action_visits"], 1)
        self.assertFalse(step.metadata["root_puct_selected_changed_prior_action"])

    def test_root_puct_policy_applies_root_prior_temperature(self) -> None:
        policy = RootPUCTSearchPolicy(
            env_factory=lambda: ImmediateOutcomeEnv(label="branch"),
            rollout_config=RolloutConfig(max_decision_rounds=3),
            value_fn=lambda history: 0.0,
            prior_fn=lambda history: (0.9, 0.1) + (0.0,) * (ACTION_COUNT - 2),
            opponent_action_planner=lambda context, rng: {"p2": 0},
            cpuct=0.0,
            selection_mode="visits",
            root_visit_budget=2,
            root_prior_temperature=2.0,
        )

        result = RolloutDriver(
            env=ImmediateOutcomeEnv(label="live"),
            policies={"p1": policy, "p2": FixedPolicy(0, policy_id="fixed-p2")},
            config=RolloutConfig(max_decision_rounds=3),
        ).run(seed=96, battle_id="search-policy")

        step = result.trajectory.steps_for_player("p1")[0]
        self.assertEqual(step.metadata["root_puct_root_prior_temperature"], 2.0)
        self.assertEqual(step.metadata["root_puct_prior_action"], 0)
        self.assertAlmostEqual(step.metadata["root_puct_prior_action_prior"], 0.75)

    def test_root_puct_multi_scenario_no_budget_preserves_synthetic_one_visit_per_action(self) -> None:
        search_1 = puct_branch_search(
            env=ImmediateOutcomeEnv(label="scenario-1"),
            trajectory=BattleTrajectory(battle_id="search-policy", format_id="gen3randombattle", seed=7),
            player_id="p1",
            prefix_decision_round_count=0,
            legal_action_mask=_mask(0, 1),
            opponent_actions={"p2": 0},
            value_fn=lambda history: 0.0,
            action_priors=(0.9, 0.1) + (0.0,) * (ACTION_COUNT - 2),
            cpuct=2.0,
        )
        search_2 = puct_branch_search(
            env=ImmediateOutcomeEnv(label="scenario-2"),
            trajectory=BattleTrajectory(battle_id="search-policy", format_id="gen3randombattle", seed=7),
            player_id="p1",
            prefix_decision_round_count=0,
            legal_action_mask=_mask(0, 1),
            opponent_actions={"p2": 1},
            value_fn=lambda history: 0.0,
            action_priors=(0.9, 0.1) + (0.0,) * (ACTION_COUNT - 2),
            cpuct=2.0,
        )

        aggregate = _aggregate_scenario_searches(
            (search_1, search_2),
            opponent_scenarios=(
                OpponentActionScenario(actions={"p2": 0}, weight=0.25, label="low"),
                OpponentActionScenario(actions={"p2": 1}, weight=0.75, label="high"),
            ),
            cpuct=2.0,
        )

        self.assertEqual(aggregate.total_visits, 2)
        self.assertEqual([candidate.visits for candidate in aggregate.candidates], [1, 1])

    def test_root_puct_multi_scenario_visit_counts_are_weighted(self) -> None:
        search_prior_zero = puct_branch_search(
            env=ImmediateOutcomeEnv(label="scenario-1"),
            trajectory=BattleTrajectory(battle_id="search-policy", format_id="gen3randombattle", seed=7),
            player_id="p1",
            prefix_decision_round_count=0,
            legal_action_mask=_mask(0, 1),
            opponent_actions={"p2": 0},
            value_fn=lambda history: 0.0,
            action_priors=(0.9, 0.1) + (0.0,) * (ACTION_COUNT - 2),
            cpuct=8.0,
            root_visit_budget=5,
        )
        search_prior_one = puct_branch_search(
            env=ImmediateOutcomeEnv(label="scenario-2"),
            trajectory=BattleTrajectory(battle_id="search-policy", format_id="gen3randombattle", seed=7),
            player_id="p1",
            prefix_decision_round_count=0,
            legal_action_mask=_mask(0, 1),
            opponent_actions={"p2": 1},
            value_fn=lambda history: 0.0,
            action_priors=(0.1, 0.9) + (0.0,) * (ACTION_COUNT - 2),
            cpuct=8.0,
            root_visit_budget=5,
        )

        aggregate = _aggregate_scenario_searches(
            (search_prior_zero, search_prior_one),
            opponent_scenarios=(
                OpponentActionScenario(actions={"p2": 0}, weight=0.8, label="likely"),
                OpponentActionScenario(actions={"p2": 1}, weight=0.2, label="unlikely"),
            ),
            cpuct=8.0,
        )

        visits_by_action = {candidate.action_index: candidate.visits for candidate in aggregate.candidates}
        self.assertEqual(visits_by_action, {0: 3, 1: 2})
        self.assertEqual(aggregate.total_visits, 5)

    def test_root_puct_policy_value_selection_mode_can_be_gated_back_to_prior(self) -> None:
        policy = RootPUCTSearchPolicy(
            env_factory=lambda: ImmediateOutcomeEnv(label="branch"),
            rollout_config=RolloutConfig(max_decision_rounds=3),
            value_fn=lambda history: 0.0,
            prior_fn=lambda history: (0.9, 0.1) + (0.0,) * (ACTION_COUNT - 2),
            opponent_action_planner=lambda context, rng: {"p2": 0},
            cpuct=4.0,
            selection_mode="value",
            minimum_value_improvement=3.0,
        )

        result = RolloutDriver(
            env=ImmediateOutcomeEnv(label="live"),
            policies={"p1": policy, "p2": FixedPolicy(0, policy_id="fixed-p2")},
            config=RolloutConfig(max_decision_rounds=3),
        ).run(seed=97, battle_id="search-policy")

        step = result.trajectory.steps_for_player("p1")[0]
        self.assertEqual(step.action_index, 0)
        self.assertEqual(step.metadata["root_puct_selection_mode"], "value")
        self.assertTrue(step.metadata["root_puct_value_gate_used"])
        self.assertEqual(step.metadata["root_puct_pre_gate_action"], 1)
        self.assertEqual(step.metadata["root_puct_prior_action"], 0)

    def test_root_puct_policy_prior_ratio_gate_keeps_low_prior_override_at_prior_action(self) -> None:
        policy = RootPUCTSearchPolicy(
            env_factory=lambda: ImmediateOutcomeEnv(label="branch"),
            rollout_config=RolloutConfig(max_decision_rounds=3),
            value_fn=lambda history: 0.0,
            prior_fn=lambda history: (0.9, 0.1) + (0.0,) * (ACTION_COUNT - 2),
            opponent_action_planner=lambda context, rng: {"p2": 0},
            cpuct=0.0,
            selection_mode="value",
            minimum_override_prior_ratio=0.5,
        )

        result = RolloutDriver(
            env=ImmediateOutcomeEnv(label="live"),
            policies={"p1": policy, "p2": FixedPolicy(0, policy_id="fixed-p2")},
            config=RolloutConfig(max_decision_rounds=3),
        ).run(seed=97, battle_id="search-policy")

        step = result.trajectory.steps_for_player("p1")[0]
        self.assertEqual(step.action_index, 0)
        self.assertEqual(step.metadata["root_puct_search_action"], 1)
        self.assertEqual(step.metadata["root_puct_prior_action"], 0)
        self.assertTrue(step.metadata["root_puct_prior_ratio_gate_used"])
        self.assertEqual(step.metadata["root_puct_minimum_override_prior_ratio"], 0.5)
        self.assertAlmostEqual(step.metadata["root_puct_prior_ratio_gate_required_prior"], 0.45)
        self.assertFalse(step.metadata["root_puct_selected_changed_prior_action"])
        self.assertTrue(step.metadata["root_puct_pre_gate_changed_prior_action"])

    def test_root_puct_policy_prior_ratio_gate_allows_supported_override(self) -> None:
        policy = RootPUCTSearchPolicy(
            env_factory=lambda: ImmediateOutcomeEnv(label="branch"),
            rollout_config=RolloutConfig(max_decision_rounds=3),
            value_fn=lambda history: 0.0,
            prior_fn=lambda history: (0.6, 0.4) + (0.0,) * (ACTION_COUNT - 2),
            opponent_action_planner=lambda context, rng: {"p2": 0},
            cpuct=0.0,
            selection_mode="value",
            minimum_override_prior_ratio=0.5,
        )

        result = RolloutDriver(
            env=ImmediateOutcomeEnv(label="live"),
            policies={"p1": policy, "p2": FixedPolicy(0, policy_id="fixed-p2")},
            config=RolloutConfig(max_decision_rounds=3),
        ).run(seed=97, battle_id="search-policy")

        step = result.trajectory.steps_for_player("p1")[0]
        self.assertEqual(step.action_index, 1)
        self.assertEqual(step.metadata["root_puct_search_action"], 1)
        self.assertEqual(step.metadata["root_puct_prior_action"], 0)
        self.assertFalse(step.metadata["root_puct_prior_ratio_gate_used"])
        self.assertAlmostEqual(step.metadata["root_puct_prior_ratio_gate_required_prior"], 0.3)
        self.assertTrue(step.metadata["root_puct_selected_changed_prior_action"])
        self.assertTrue(step.metadata["root_puct_pre_gate_changed_prior_action"])

    def test_root_puct_policy_prior_ratio_gate_noops_after_value_gate_revert(self) -> None:
        policy = RootPUCTSearchPolicy(
            env_factory=lambda: ImmediateOutcomeEnv(label="branch"),
            rollout_config=RolloutConfig(max_decision_rounds=3),
            value_fn=lambda history: 0.0,
            prior_fn=lambda history: (0.9, 0.1) + (0.0,) * (ACTION_COUNT - 2),
            opponent_action_planner=lambda context, rng: {"p2": 0},
            cpuct=0.0,
            selection_mode="value",
            minimum_value_improvement=3.0,
            minimum_override_prior_ratio=0.5,
        )

        result = RolloutDriver(
            env=ImmediateOutcomeEnv(label="live"),
            policies={"p1": policy, "p2": FixedPolicy(0, policy_id="fixed-p2")},
            config=RolloutConfig(max_decision_rounds=3),
        ).run(seed=97, battle_id="search-policy")

        step = result.trajectory.steps_for_player("p1")[0]
        self.assertEqual(step.action_index, 0)
        self.assertEqual(step.metadata["root_puct_search_action"], 1)
        self.assertEqual(step.metadata["root_puct_prior_action"], 0)
        self.assertTrue(step.metadata["root_puct_value_gate_used"])
        self.assertFalse(step.metadata["root_puct_prior_ratio_gate_used"])
        self.assertTrue(step.metadata["root_puct_pre_gate_changed_prior_action"])
        self.assertFalse(step.metadata["root_puct_selected_changed_prior_action"])

    def test_root_puct_policy_score_gate_keeps_lower_score_override_at_prior_action(self) -> None:
        policy = RootPUCTSearchPolicy(
            env_factory=lambda: ImmediateOutcomeEnv(label="branch"),
            rollout_config=RolloutConfig(max_decision_rounds=3),
            value_fn=lambda history: 0.0,
            prior_fn=lambda history: (0.9, 0.1) + (0.0,) * (ACTION_COUNT - 2),
            opponent_action_planner=lambda context, rng: {"p2": 0},
            cpuct=4.0,
            selection_mode="value",
            minimum_score_improvement=0.0,
            root_visit_budget=None,
        )

        result = RolloutDriver(
            env=ImmediateOutcomeEnv(label="live"),
            policies={"p1": policy, "p2": FixedPolicy(0, policy_id="fixed-p2")},
            config=RolloutConfig(max_decision_rounds=3),
        ).run(seed=97, battle_id="search-policy")

        step = result.trajectory.steps_for_player("p1")[0]
        self.assertEqual(step.action_index, 0)
        self.assertEqual(step.metadata["root_puct_search_action"], 1)
        self.assertEqual(step.metadata["root_puct_prior_action"], 0)
        self.assertTrue(step.metadata["root_puct_score_gate_used"])
        self.assertEqual(
            step.metadata["root_puct_score_gate_required_score"],
            step.metadata["root_puct_prior_score"],
        )
        self.assertLess(
            step.metadata["root_puct_search_action_score"],
            step.metadata["root_puct_score_gate_required_score"],
        )
        self.assertTrue(step.metadata["root_puct_pre_gate_changed_prior_action"])
        self.assertFalse(step.metadata["root_puct_selected_changed_prior_action"])

    def test_root_puct_policy_score_gate_suppresses_lower_score_visits_override(self) -> None:
        policy = RootPUCTSearchPolicy(
            env_factory=lambda: ImmediateOutcomeEnv(label="branch"),
            rollout_config=RolloutConfig(max_decision_rounds=3),
            value_fn=lambda history: 0.0,
            prior_fn=lambda history: (0.51, 0.49) + (0.0,) * (ACTION_COUNT - 2),
            opponent_action_planner=lambda context, rng: {"p2": 0},
            cpuct=2.0,
            selection_mode="visits",
            root_visit_budget=20,
            minimum_score_improvement=0.0,
        )

        result = RolloutDriver(
            env=ImmediateOutcomeEnv(label="live"),
            policies={"p1": policy, "p2": FixedPolicy(0, policy_id="fixed-p2")},
            config=RolloutConfig(max_decision_rounds=3),
        ).run(seed=97, battle_id="search-policy")

        step = result.trajectory.steps_for_player("p1")[0]
        self.assertEqual(step.action_index, 0)
        self.assertEqual(step.metadata["root_puct_selection_mode"], "visits")
        self.assertEqual(step.metadata["root_puct_search_action"], 1)
        self.assertEqual(step.metadata["root_puct_prior_action"], 0)
        self.assertGreater(
            step.metadata["root_puct_search_action_visits"],
            step.metadata["root_puct_prior_action_visits"],
        )
        self.assertTrue(step.metadata["root_puct_score_gate_used"])
        self.assertLess(
            step.metadata["root_puct_search_action_score"],
            step.metadata["root_puct_score_gate_required_score"],
        )
        self.assertTrue(step.metadata["root_puct_pre_gate_changed_prior_action"])
        self.assertFalse(step.metadata["root_puct_selected_changed_prior_action"])

    def test_root_puct_policy_score_gate_allows_score_improving_override_with_positive_margin(self) -> None:
        policy = RootPUCTSearchPolicy(
            env_factory=lambda: ImmediateOutcomeEnv(label="branch"),
            rollout_config=RolloutConfig(max_decision_rounds=3),
            value_fn=lambda history: 0.0,
            prior_fn=lambda history: (0.9, 0.1) + (0.0,) * (ACTION_COUNT - 2),
            opponent_action_planner=lambda context, rng: {"p2": 0},
            cpuct=0.0,
            selection_mode="value",
            minimum_score_improvement=1.5,
        )

        result = RolloutDriver(
            env=ImmediateOutcomeEnv(label="live"),
            policies={"p1": policy, "p2": FixedPolicy(0, policy_id="fixed-p2")},
            config=RolloutConfig(max_decision_rounds=3),
        ).run(seed=97, battle_id="search-policy")

        step = result.trajectory.steps_for_player("p1")[0]
        self.assertEqual(step.action_index, 1)
        self.assertEqual(step.metadata["root_puct_search_action"], 1)
        self.assertEqual(step.metadata["root_puct_prior_action"], 0)
        self.assertFalse(step.metadata["root_puct_score_gate_used"])
        self.assertEqual(
            step.metadata["root_puct_score_gate_required_score"],
            step.metadata["root_puct_prior_score"] + 1.5,
        )
        self.assertGreaterEqual(
            step.metadata["root_puct_search_action_score"],
            step.metadata["root_puct_score_gate_required_score"],
        )
        self.assertTrue(step.metadata["root_puct_selected_changed_prior_action"])
        self.assertTrue(step.metadata["root_puct_pre_gate_changed_prior_action"])

    def test_root_puct_policy_can_use_bounded_leaf_rollouts_for_branch_values(self) -> None:
        branch_envs: list[DelayedOutcomeEnv] = []

        def branch_env_factory() -> DelayedOutcomeEnv:
            env = DelayedOutcomeEnv({0: "p2", 1: "p1"})
            branch_envs.append(env)
            return env

        policy = RootPUCTSearchPolicy(
            env_factory=branch_env_factory,
            rollout_config=RolloutConfig(max_decision_rounds=3),
            value_fn=lambda history: 0.0,
            prior_fn=lambda history: (0.5, 0.5) + (0.0,) * (ACTION_COUNT - 2),
            opponent_action_planner=lambda context, rng: {"p2": 0},
            cpuct=0.0,
            leaf_rollout_decision_rounds=1,
            leaf_rollout_policy_factory=lambda player_id: FixedPolicy(0, policy_id=f"leaf-{player_id}"),
            root_visit_budget=None,
        )
        context = PolicyContext(
            player_id="p1",
            decision_round_index=0,
            battle_id="search-policy",
            format_id="gen3randombattle",
            seed=98,
            observation=_observation(0, 1),
            requested_players=("p1", "p2"),
            trajectory=BattleTrajectory(battle_id="search-policy", format_id="gen3randombattle", seed=98),
            requested_legal_action_masks={"p1": _mask(0, 1), "p2": _mask(0)},
        )

        decision = policy.select_action_with_context(context, rng=random.Random(1))

        self.assertEqual(decision.action_index, 1)
        self.assertEqual(decision.metadata["root_puct_leaf_rollout_rounds"], 1)
        self.assertEqual(decision.metadata["root_puct_leaf_actual_rollout_rounds"], {"1": 2})
        self.assertEqual(decision.metadata["root_puct_leaf_evaluations"], {"rollout_terminal": 2})
        self.assertEqual(decision.metadata["root_puct_candidate_count"], 2)
        self.assertEqual(len(branch_envs), 1)
        self.assertTrue(branch_envs[0].closed)
        self.assertEqual(
            branch_envs[0].all_step_calls,
            [
                {"p1": 0, "p2": 0},
                {"p1": 0, "p2": 0},
                {"p1": 1, "p2": 0},
                {"p1": 0, "p2": 0},
            ],
        )

    def test_root_puct_policy_value_gate_keeps_search_action_with_sufficient_value_lift(self) -> None:
        policy = RootPUCTSearchPolicy(
            env_factory=lambda: ImmediateOutcomeEnv(label="branch"),
            rollout_config=RolloutConfig(max_decision_rounds=3),
            value_fn=lambda history: 0.0,
            prior_fn=lambda history: (0.9, 0.1) + (0.0,) * (ACTION_COUNT - 2),
            opponent_action_planner=lambda context, rng: {"p2": 0},
            cpuct=0.0,
            minimum_value_improvement=0.0,
        )

        result = RolloutDriver(
            env=ImmediateOutcomeEnv(label="live"),
            policies={"p1": policy, "p2": FixedPolicy(0, policy_id="fixed-p2")},
            config=RolloutConfig(max_decision_rounds=3),
        ).run(seed=95, battle_id="search-policy")

        step = result.trajectory.steps_for_player("p1")[0]
        self.assertEqual(step.action_index, 1)
        self.assertFalse(step.metadata["root_puct_value_gate_used"])
        self.assertEqual(step.metadata["root_puct_pre_gate_action"], 1)
        self.assertEqual(step.metadata["root_puct_prior_action"], 0)

    def test_root_puct_policy_rejects_invalid_value_gate_margin(self) -> None:
        with self.assertRaisesRegex(ValueError, "minimum_value_improvement"):
            RootPUCTSearchPolicy(
                env_factory=lambda: ImmediateOutcomeEnv(label="branch"),
                rollout_config=RolloutConfig(max_decision_rounds=3),
                value_fn=lambda history: 0.0,
                prior_fn=lambda history: (1.0,) + (0.0,) * (ACTION_COUNT - 1),
                minimum_value_improvement=-0.1,
            )

    def test_root_puct_policy_rejects_invalid_prior_ratio_gate(self) -> None:
        with self.assertRaisesRegex(ValueError, "minimum_override_prior_ratio"):
            RootPUCTSearchPolicy(
                env_factory=lambda: ImmediateOutcomeEnv(label="branch"),
                rollout_config=RolloutConfig(max_decision_rounds=3),
                value_fn=lambda history: 0.0,
                prior_fn=lambda history: (1.0,) + (0.0,) * (ACTION_COUNT - 1),
                minimum_override_prior_ratio=-0.1,
            )

    def test_root_puct_policy_rejects_invalid_score_gate_margin(self) -> None:
        with self.assertRaisesRegex(ValueError, "minimum_score_improvement"):
            RootPUCTSearchPolicy(
                env_factory=lambda: ImmediateOutcomeEnv(label="branch"),
                rollout_config=RolloutConfig(max_decision_rounds=3),
                value_fn=lambda history: 0.0,
                prior_fn=lambda history: (1.0,) + (0.0,) * (ACTION_COUNT - 1),
                minimum_score_improvement=-0.1,
            )

    def test_root_puct_policy_rejects_invalid_selection_mode(self) -> None:
        with self.assertRaisesRegex(ValueError, "selection_mode"):
            RootPUCTSearchPolicy(
                env_factory=lambda: ImmediateOutcomeEnv(label="branch"),
                rollout_config=RolloutConfig(max_decision_rounds=3),
                value_fn=lambda history: 0.0,
                prior_fn=lambda history: (1.0,) + (0.0,) * (ACTION_COUNT - 1),
                selection_mode="unknown",
            )

    def test_root_puct_policy_rejects_invalid_root_visit_budget(self) -> None:
        with self.assertRaisesRegex(ValueError, "root_visit_budget"):
            RootPUCTSearchPolicy(
                env_factory=lambda: ImmediateOutcomeEnv(label="branch"),
                rollout_config=RolloutConfig(max_decision_rounds=3),
                value_fn=lambda history: 0.0,
                prior_fn=lambda history: (0.9, 0.1) + (0.0,) * (ACTION_COUNT - 2),
                root_visit_budget=0,
            )

    def test_root_puct_policy_rejects_invalid_root_time_budget(self) -> None:
        with self.assertRaisesRegex(ValueError, "root_time_budget_seconds"):
            RootPUCTSearchPolicy(
                env_factory=lambda: ImmediateOutcomeEnv(label="branch"),
                rollout_config=RolloutConfig(max_decision_rounds=3),
                value_fn=lambda history: 0.0,
                prior_fn=lambda history: (0.9, 0.1) + (0.0,) * (ACTION_COUNT - 2),
                root_time_budget_seconds=0.0,
            )

    def test_root_puct_policy_rejects_invalid_root_prior_temperature(self) -> None:
        with self.assertRaisesRegex(ValueError, "root_prior_temperature"):
            RootPUCTSearchPolicy(
                env_factory=lambda: ImmediateOutcomeEnv(label="branch"),
                rollout_config=RolloutConfig(max_decision_rounds=3),
                value_fn=lambda history: 0.0,
                prior_fn=lambda history: (0.9, 0.1) + (0.0,) * (ACTION_COUNT - 2),
                root_prior_temperature=0.0,
            )

    def test_root_puct_policy_rejects_leaf_rollouts_without_policy_factory(self) -> None:
        with self.assertRaisesRegex(ValueError, "leaf_rollout_policy_factory"):
            RootPUCTSearchPolicy(
                env_factory=lambda: ImmediateOutcomeEnv(label="branch"),
                rollout_config=RolloutConfig(max_decision_rounds=3),
                value_fn=lambda history: 0.0,
                prior_fn=lambda history: (1.0,) + (0.0,) * (ACTION_COUNT - 1),
                leaf_rollout_decision_rounds=1,
            )

    def test_root_puct_policy_rejects_invalid_start_override_attempts(self) -> None:
        with self.assertRaisesRegex(ValueError, "start_override_attempts"):
            RootPUCTSearchPolicy(
                env_factory=lambda: ImmediateOutcomeEnv(label="branch"),
                rollout_config=RolloutConfig(max_decision_rounds=3),
                value_fn=lambda history: 0.0,
                prior_fn=lambda history: (1.0,) + (0.0,) * (ACTION_COUNT - 1),
                start_override_attempts=0,
            )

    def test_root_puct_policy_rejects_missing_opponent_action_planner_for_simultaneous_turn(self) -> None:
        policy = RootPUCTSearchPolicy(
            env_factory=lambda: ImmediateOutcomeEnv(label="branch"),
            rollout_config=RolloutConfig(max_decision_rounds=3),
            value_fn=lambda history: 0.0,
            prior_fn=lambda history: (1.0,) + (0.0,) * (ACTION_COUNT - 1),
            cpuct=0.0,
        )

        with self.assertRaisesRegex(ValueError, "missing opponent actions"):
            RolloutDriver(
                env=ImmediateOutcomeEnv(label="live"),
                policies={"p1": policy, "p2": FixedPolicy(0, policy_id="fixed-p2")},
                config=RolloutConfig(max_decision_rounds=3),
            ).run(seed=92, battle_id="search-policy")

    def test_root_puct_policy_falls_back_when_opponent_planner_returns_illegal_action(self) -> None:
        policy = RootPUCTSearchPolicy(
            env_factory=lambda: ImmediateOutcomeEnv(label="branch"),
            rollout_config=RolloutConfig(max_decision_rounds=3),
            value_fn=lambda history: 0.0,
            prior_fn=lambda history: (1.0,) + (0.0,) * (ACTION_COUNT - 1),
            opponent_action_planner=lambda context, rng: {"p2": 99},
            fallback_policy=FixedPolicy(1, policy_id="fallback-fixed", value_estimate=0.25),
            allow_fallback=True,
            cpuct=0.0,
        )

        result = RolloutDriver(
            env=ImmediateOutcomeEnv(label="live"),
            policies={"p1": policy, "p2": FixedPolicy(0, policy_id="fixed-p2")},
            config=RolloutConfig(max_decision_rounds=3),
        ).run(seed=93, battle_id="search-policy")

        step = result.trajectory.steps_for_player("p1")[0]
        self.assertEqual(step.action_index, 1)
        self.assertTrue(step.metadata["root_puct_fallback"])
        self.assertIn("illegal action for p2", step.metadata["root_puct_fallback_reason"])
        self.assertEqual(step.metadata["fallback_policy_id"], "fallback-fixed")
        self.assertEqual(step.value_estimate, 0.25)

    def test_root_puct_policy_falls_back_when_all_hidden_opponent_scenarios_are_replay_rejected(self) -> None:
        def scenario_planner(context: PolicyContext, rng: random.Random) -> tuple[OpponentActionScenario, ...]:
            del context, rng
            return (
                OpponentActionScenario(actions={"p2": 2}, weight=1.0, label="illegal-hidden"),
            )

        policy = RootPUCTSearchPolicy(
            env_factory=lambda: StrictOpponentActionEnv(label="branch"),
            rollout_config=RolloutConfig(max_decision_rounds=3),
            value_fn=lambda history: 0.0,
            prior_fn=lambda history: (1.0,) + (0.0,) * (ACTION_COUNT - 1),
            opponent_action_scenario_planner=scenario_planner,
            fallback_policy=FixedPolicy(1, policy_id="fallback-fixed", value_estimate=0.25),
            allow_fallback=True,
            cpuct=0.0,
        )
        context = PolicyContext(
            player_id="p1",
            decision_round_index=0,
            battle_id="search-policy",
            format_id="gen3randombattle",
            seed=93,
            observation=_observation(0),
            requested_players=("p1", "p2"),
            trajectory=BattleTrajectory(battle_id="search-policy", format_id="gen3randombattle", seed=93),
            requested_legal_action_masks={"p1": _mask(0)},
        )

        decision = policy.select_action_with_context(context, rng=random.Random(1))

        self.assertEqual(decision.action_index, 1)
        self.assertTrue(decision.metadata["root_puct_fallback"])
        self.assertIn(
            "all opponent action scenarios were replay-illegal",
            decision.metadata["root_puct_fallback_reason"],
        )
        self.assertEqual(decision.metadata["root_puct_opponent_action_scenarios_generated"], 1)
        self.assertEqual(decision.metadata["root_puct_opponent_action_scenarios_skipped"], 1)
        self.assertEqual(
            decision.metadata["root_puct_opponent_action_skipped_scenarios"],
            [
                {
                    "label": "illegal-hidden",
                    "weight": 1.0,
                    "actions": {"p2": 2},
                    "reason": "p2: action_index 2 is not legal for the current request.",
                }
            ],
        )
        self.assertEqual(decision.metadata["fallback_policy_id"], "fallback-fixed")

    def test_root_puct_policy_no_fallback_reports_clean_all_hidden_scenarios_rejected_error(self) -> None:
        def scenario_planner(context: PolicyContext, rng: random.Random) -> tuple[OpponentActionScenario, ...]:
            del context, rng
            return (
                OpponentActionScenario(actions={"p2": 2}, weight=1.0, label="illegal-hidden"),
            )

        policy = RootPUCTSearchPolicy(
            env_factory=lambda: StrictOpponentActionEnv(label="branch"),
            rollout_config=RolloutConfig(max_decision_rounds=3),
            value_fn=lambda history: 0.0,
            prior_fn=lambda history: (1.0,) + (0.0,) * (ACTION_COUNT - 1),
            opponent_action_scenario_planner=scenario_planner,
            allow_fallback=False,
            cpuct=0.0,
        )
        context = PolicyContext(
            player_id="p1",
            decision_round_index=0,
            battle_id="search-policy",
            format_id="gen3randombattle",
            seed=93,
            observation=_observation(0),
            requested_players=("p1", "p2"),
            trajectory=BattleTrajectory(battle_id="search-policy", format_id="gen3randombattle", seed=93),
            requested_legal_action_masks={"p1": _mask(0)},
        )

        with self.assertRaisesRegex(
            ValueError,
            (
                "root PUCT search cannot select an action: all opponent action scenarios "
                "were replay-illegal"
            ),
        ) as raised:
            policy.select_action_with_context(context, rng=random.Random(1))

        self.assertNotIn("search failed: root PUCT search cannot select", str(raised.exception))

    def test_root_puct_policy_can_fallback_when_context_is_missing(self) -> None:
        policy = RootPUCTSearchPolicy(
            env_factory=lambda: ImmediateOutcomeEnv(label="branch"),
            rollout_config=RolloutConfig(max_decision_rounds=3),
            value_fn=lambda history: 0.0,
            prior_fn=lambda history: (1.0,) + (0.0,) * (ACTION_COUNT - 1),
            fallback_policy=RandomLegalPolicy(policy_id="fallback-random"),
            allow_fallback=True,
        )

        decision = policy.select_action(_observation(0), rng=random.Random(1))

        self.assertEqual(decision.action_index, 0)
        self.assertEqual(decision.policy_id, "root-puct-search")
        self.assertTrue(decision.metadata["root_puct_fallback"])
        self.assertEqual(decision.metadata["fallback_policy_id"], "fallback-random")


if __name__ == "__main__":
    unittest.main()
