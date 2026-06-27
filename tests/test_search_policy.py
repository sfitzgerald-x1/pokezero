import random
import unittest

from pokezero.actions import ACTION_COUNT
from pokezero.env import StepResult, TerminalState
from pokezero.observation import PokeZeroObservationV0
from pokezero.policy import PolicyContext, PolicyDecision, RandomLegalPolicy
from pokezero.rollout import RolloutConfig, RolloutDriver
from pokezero.search_policy import RootPUCTSearchPolicy, greedy_opponent_action_planner, policy_opponent_action_planner
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
    def __init__(self, action_index: int, *, policy_id: str = "fixed") -> None:
        self.action_index = action_index
        self.policy_id = policy_id

    def select_action(self, observation: PokeZeroObservationV0, *, rng) -> PolicyDecision:
        return PolicyDecision(action_index=self.action_index, policy_id=self.policy_id, action_probability=1.0)


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
        self.assertEqual(step.metadata["root_puct_prior_action"], 0)
        self.assertEqual(step.metadata["root_puct_minimum_value_improvement"], 3.0)

    def test_root_puct_policy_default_selection_mode_uses_puct_score(self) -> None:
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

    def test_root_puct_policy_rejects_invalid_selection_mode(self) -> None:
        with self.assertRaisesRegex(ValueError, "selection_mode"):
            RootPUCTSearchPolicy(
                env_factory=lambda: ImmediateOutcomeEnv(label="branch"),
                rollout_config=RolloutConfig(max_decision_rounds=3),
                value_fn=lambda history: 0.0,
                prior_fn=lambda history: (1.0,) + (0.0,) * (ACTION_COUNT - 1),
                selection_mode="unknown",
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
            fallback_policy=FixedPolicy(1, policy_id="fallback-fixed"),
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
