import unittest

from pokezero.actions import ACTION_COUNT
from pokezero.env import StepResult, TerminalState
from pokezero.observation import PokeZeroObservationV0
from pokezero.policy import RandomLegalPolicy
from pokezero.policy import PolicyDecision
from pokezero.rollout import RolloutConfig
from pokezero.search import flat_branch_search, puct_branch_search, terminal_value_for_player, value_branch_search
from pokezero.trajectory import BattleTrajectory, TrajectoryStep


def _observation(action_index: int) -> PokeZeroObservationV0:
    return PokeZeroObservationV0(
        categorical_ids=(),
        numeric_features=(),
        token_type_ids=(),
        attention_mask=(),
        legal_action_mask=tuple(index == action_index for index in range(ACTION_COUNT)),
    )


class BranchOutcomeEnv:
    def __init__(self, winners_by_action: dict[int, str | None]) -> None:
        self.winners_by_action = winners_by_action
        self.step_calls: list[dict[str, int]] = []
        self.all_step_calls: list[dict[str, int]] = []
        self.reset_calls: list[tuple[int, str]] = []
        self._requested = ("p1", "p2")
        self._terminal: TerminalState | None = None

    def reset(self, *, seed: int, format_id: str = "gen3randombattle") -> None:
        self.reset_calls.append((seed, format_id))
        self.step_calls.clear()
        self._requested = ("p1", "p2")
        self._terminal = None

    def observe(self, player: str) -> PokeZeroObservationV0:
        return _observation(0)

    def legal_actions(self, player: str) -> tuple[bool, ...]:
        return self.observe(player).legal_action_mask

    def requested_players(self) -> tuple[str, ...]:
        return self._requested

    def step(self, actions: dict[str, int]) -> StepResult:
        self.step_calls.append(dict(actions))
        self.all_step_calls.append(dict(actions))
        p1_action = int(actions["p1"])
        winner = self.winners_by_action[p1_action]
        self._terminal = TerminalState(winner=winner, turn_count=1)
        self._requested = ()
        return StepResult(
            observations={},
            rewards={"p1": 1.0 if winner == "p1" else -1.0 if winner == "p2" else 0.0},
            terminal=self._terminal,
            requested_players=(),
        )

    def terminal(self) -> TerminalState | None:
        return self._terminal


class ValueBranchEnv:
    def __init__(self, terminal_winners_by_action: dict[int, str | None] | None = None) -> None:
        self.terminal_winners_by_action = terminal_winners_by_action or {}
        self.all_step_calls: list[dict[str, int]] = []
        self.reset_calls: list[tuple[int, str]] = []
        self._requested = ("p1", "p2")
        self._terminal: TerminalState | None = None

    def reset(self, *, seed: int, format_id: str = "gen3randombattle") -> None:
        self.reset_calls.append((seed, format_id))
        self._requested = ("p1", "p2")
        self._terminal = None

    def observe(self, player: str) -> PokeZeroObservationV0:
        return _observation(0)

    def legal_actions(self, player: str) -> tuple[bool, ...]:
        return self.observe(player).legal_action_mask

    def requested_players(self) -> tuple[str, ...]:
        return self._requested

    def step(self, actions: dict[str, int]) -> StepResult:
        self.all_step_calls.append(dict(actions))
        p1_action = int(actions["p1"])
        if p1_action in self.terminal_winners_by_action:
            winner = self.terminal_winners_by_action[p1_action]
            self._terminal = TerminalState(winner=winner, turn_count=len(self.all_step_calls))
            self._requested = ()
            return StepResult(
                observations={},
                rewards={"p1": 1.0 if winner == "p1" else -1.0 if winner == "p2" else 0.0},
                terminal=self._terminal,
                requested_players=(),
            )
        self._requested = ("p1", "p2")
        return StepResult(
            observations={"p1": _observation(p1_action), "p2": _observation(0)},
            rewards={"p1": 0.0, "p2": 0.0},
            terminal=None,
            requested_players=self._requested,
        )

    def terminal(self) -> TerminalState | None:
        return self._terminal


class StrictLegalValueBranchEnv(ValueBranchEnv):
    def __init__(self, legal_actions: set[int]) -> None:
        super().__init__()
        self.strict_legal_actions = set(legal_actions)

    def step(self, actions: dict[str, int]) -> StepResult:
        p1_action = int(actions["p1"])
        if p1_action not in self.strict_legal_actions:
            raise ValueError(f"action_index {p1_action} is not legal for the current request.")
        return super().step(actions)


class FixedPolicy:
    def __init__(self, action_index: int, *, policy_id: str = "fixed") -> None:
        self.action_index = action_index
        self.policy_id = policy_id

    def select_action(self, observation: PokeZeroObservationV0, *, rng) -> PolicyDecision:
        return PolicyDecision(action_index=self.action_index, policy_id=self.policy_id)


class FirstLegalPolicy:
    policy_id = "first-legal"

    def select_action(self, observation: PokeZeroObservationV0, *, rng) -> PolicyDecision:
        del rng
        return PolicyDecision(
            action_index=tuple(observation.legal_action_mask).index(True),
            policy_id=self.policy_id,
        )


class ContinuationOutcomeEnv:
    def __init__(self, winners_after_branch: dict[int, str | None]) -> None:
        self.winners_after_branch = winners_after_branch
        self.all_step_calls: list[dict[str, int]] = []
        self.reset_calls: list[tuple[int, str]] = []
        self._requested = ("p1", "p2")
        self._terminal: TerminalState | None = None
        self._branch_action: int | None = None

    def reset(self, *, seed: int, format_id: str = "gen3randombattle") -> None:
        self.reset_calls.append((seed, format_id))
        self._requested = ("p1", "p2")
        self._terminal = None
        self._branch_action = None

    def observe(self, player: str) -> PokeZeroObservationV0:
        del player
        return _observation(0)

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
                observations={"p1": _observation(self._branch_action), "p2": _observation(0)},
                rewards={"p1": 0.0, "p2": 0.0},
                terminal=None,
                requested_players=self._requested,
            )
        winner = self.winners_after_branch[self._branch_action]
        if winner is None:
            self._requested = ("p1", "p2")
            return StepResult(
                observations={"p1": _observation(self._branch_action), "p2": _observation(0)},
                rewards={"p1": 0.0, "p2": 0.0},
                terminal=None,
                requested_players=self._requested,
            )
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


class FlatBranchSearchTest(unittest.TestCase):
    def test_terminal_value_for_player_scores_win_loss_and_tie(self) -> None:
        self.assertEqual(terminal_value_for_player(TerminalState(winner="p1", turn_count=1), player_id="p1"), 1.0)
        self.assertEqual(terminal_value_for_player(TerminalState(winner="p2", turn_count=1), player_id="p1"), -1.0)
        self.assertEqual(terminal_value_for_player(TerminalState(winner=None, turn_count=1), player_id="p1"), 0.0)

    def test_value_branch_search_scores_post_branch_observation_histories(self) -> None:
        env = ValueBranchEnv()
        trajectory = BattleTrajectory(battle_id="battle", format_id="gen3randombattle", seed=77)
        trajectory.append(
            TrajectoryStep(
                player_id="p1",
                turn_index=0,
                observation=_observation(0),
                legal_action_mask=_observation(0).legal_action_mask,
                action_index=0,
            )
        )
        trajectory.append(
            TrajectoryStep(
                player_id="p2",
                turn_index=0,
                observation=_observation(0),
                legal_action_mask=_observation(0).legal_action_mask,
                action_index=0,
            )
        )
        trajectory.append(
            TrajectoryStep(
                player_id="p1",
                turn_index=1,
                observation=_observation(1),
                legal_action_mask=_observation(1).legal_action_mask,
                action_index=1,
            )
        )
        histories: list[tuple[int, ...]] = []

        def value_fn(history: tuple[PokeZeroObservationV0, ...]) -> float:
            action_history = tuple(_only_legal_action(observation) for observation in history)
            histories.append(action_history)
            return {0: 0.1, 1: 0.7, 4: 0.2}[action_history[-1]]

        result = value_branch_search(
            env=env,
            trajectory=trajectory,
            player_id="p1",
            prefix_decision_round_count=1,
            legal_action_mask=(True, True, False, False, True, False, False, False, False),
            opponent_actions={"p2": 0},
            value_fn=value_fn,
        )

        self.assertEqual(result.action_index, 1)
        self.assertEqual([candidate.action_index for candidate in result.candidates], [0, 1, 4])
        self.assertEqual([candidate.value for candidate in result.candidates], [0.1, 0.7, 0.2])
        self.assertEqual(histories, [(0, 1, 0), (0, 1, 1), (0, 1, 4)])
        self.assertEqual([candidate.evaluated_history_length for candidate in result.candidates], [3, 3, 3])
        self.assertEqual(result.to_dict()["selected_action_index"], 1)

    def test_value_branch_search_uses_terminal_value_without_calling_value_fn(self) -> None:
        env = ValueBranchEnv({1: "p1"})
        trajectory = BattleTrajectory(battle_id="battle", format_id="gen3randombattle", seed=77)

        def value_fn(history: tuple[PokeZeroObservationV0, ...]) -> float:
            raise AssertionError("terminal branches should not call value_fn")

        result = value_branch_search(
            env=env,
            trajectory=trajectory,
            player_id="p1",
            prefix_decision_round_count=0,
            legal_action_mask=(False, True, False, False, False, False, False, False, False),
            opponent_actions={"p2": 0},
            value_fn=value_fn,
        )

        self.assertEqual(result.action_index, 1)
        self.assertEqual(result.best_candidate.value, 1.0)
        self.assertEqual(result.best_candidate.evaluated_history_length, 0)

    def test_value_branch_search_skips_candidate_actions_rejected_by_replay(self) -> None:
        env = StrictLegalValueBranchEnv({1})
        trajectory = BattleTrajectory(battle_id="battle", format_id="gen3randombattle", seed=77)

        result = value_branch_search(
            env=env,
            trajectory=trajectory,
            player_id="p1",
            prefix_decision_round_count=0,
            legal_action_mask=(True, True, False, False, False, False, False, False, False),
            opponent_actions={"p2": 0},
            value_fn=lambda history: 0.25,
        )

        self.assertEqual([candidate.action_index for candidate in result.candidates], [1])
        self.assertEqual(result.action_index, 1)

    def test_value_branch_search_can_score_bounded_leaf_rollout_terminals(self) -> None:
        env = ContinuationOutcomeEnv({0: "p2", 1: "p1"})
        trajectory = BattleTrajectory(battle_id="battle", format_id="gen3randombattle", seed=77)

        result = value_branch_search(
            env=env,
            trajectory=trajectory,
            player_id="p1",
            prefix_decision_round_count=0,
            legal_action_mask=(True, True, False, False, False, False, False, False, False),
            opponent_actions={"p2": 0},
            value_fn=lambda history: 0.0,
            leaf_rollout_policies={"p1": FirstLegalPolicy(), "p2": FixedPolicy(0)},
            leaf_rollout_config=RolloutConfig(max_decision_rounds=5),
            leaf_rollout_decision_rounds=1,
        )

        self.assertEqual(result.action_index, 1)
        self.assertEqual([candidate.value for candidate in result.candidates], [-1.0, 1.0])
        self.assertEqual(
            [candidate.leaf_evaluation for candidate in result.candidates],
            ["rollout_terminal", "rollout_terminal"],
        )
        self.assertEqual(
            [candidate.leaf_rollout_decision_round_count for candidate in result.candidates],
            [1, 1],
        )

    def test_value_branch_search_uses_value_fn_at_truncated_leaf_rollout(self) -> None:
        env = ContinuationOutcomeEnv({0: None, 1: None})
        trajectory = BattleTrajectory(battle_id="battle", format_id="gen3randombattle", seed=77)
        histories: list[tuple[int, ...]] = []

        def value_fn(history: tuple[PokeZeroObservationV0, ...]) -> float:
            action_history = tuple(_only_legal_action(observation) for observation in history)
            histories.append(action_history)
            return {0: 0.2, 1: 0.8}[action_history[-1]]

        result = value_branch_search(
            env=env,
            trajectory=trajectory,
            player_id="p1",
            prefix_decision_round_count=0,
            legal_action_mask=(True, True, False, False, False, False, False, False, False),
            opponent_actions={"p2": 0},
            value_fn=value_fn,
            leaf_rollout_policies={"p1": FirstLegalPolicy(), "p2": FixedPolicy(0)},
            leaf_rollout_config=RolloutConfig(max_decision_rounds=2),
            leaf_rollout_decision_rounds=1,
        )

        self.assertEqual(result.action_index, 1)
        self.assertEqual([candidate.value for candidate in result.candidates], [0.2, 0.8])
        self.assertEqual(
            [candidate.leaf_evaluation for candidate in result.candidates],
            ["rollout_value_fn", "rollout_value_fn"],
        )
        self.assertEqual(histories, [(0,), (1,)])

    def test_puct_branch_search_combines_value_and_policy_prior(self) -> None:
        env = ValueBranchEnv()
        trajectory = BattleTrajectory(battle_id="battle", format_id="gen3randombattle", seed=77)

        def value_fn(history: tuple[PokeZeroObservationV0, ...]) -> float:
            return {0: 0.1, 1: 0.2}[_only_legal_action(history[-1])]

        result = puct_branch_search(
            env=env,
            trajectory=trajectory,
            player_id="p1",
            prefix_decision_round_count=0,
            legal_action_mask=(True, True, False, False, False, False, False, False, False),
            opponent_actions={"p2": 0},
            value_fn=value_fn,
            action_priors=(0.9, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
            cpuct=2.0,
        )

        self.assertEqual(result.action_index, 0)
        self.assertEqual([candidate.action_index for candidate in result.candidates], [0, 1])
        self.assertAlmostEqual(result.candidates[0].prior, 0.9)
        self.assertGreater(result.candidates[0].score, result.candidates[1].score)
        self.assertEqual(result.to_dict()["selected_action_index"], 0)

    def test_puct_branch_search_accumulates_root_visit_budget(self) -> None:
        env = ValueBranchEnv()
        trajectory = BattleTrajectory(battle_id="battle", format_id="gen3randombattle", seed=77)

        def value_fn(history: tuple[PokeZeroObservationV0, ...]) -> float:
            return {0: 0.1, 1: 0.2}[_only_legal_action(history[-1])]

        result = puct_branch_search(
            env=env,
            trajectory=trajectory,
            player_id="p1",
            prefix_decision_round_count=0,
            legal_action_mask=(True, True, False, False, False, False, False, False, False),
            opponent_actions={"p2": 0},
            value_fn=value_fn,
            action_priors=(0.9, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
            cpuct=2.0,
            root_visit_budget=5,
        )

        self.assertEqual(result.total_visits, 5)
        self.assertEqual(sum(candidate.visits for candidate in result.candidates), 5)
        self.assertEqual([candidate.visits for candidate in result.candidates], [4, 1])
        self.assertEqual(len(env.all_step_calls), 5)
        self.assertEqual(result.action_index, 0)

    def test_puct_branch_search_rejects_budget_below_legal_action_count(self) -> None:
        with self.assertRaisesRegex(ValueError, "root_visit_budget"):
            puct_branch_search(
                env=ValueBranchEnv(),
                trajectory=BattleTrajectory(battle_id="battle", format_id="gen3randombattle", seed=77),
                player_id="p1",
                prefix_decision_round_count=0,
                legal_action_mask=(True, True, False, False, False, False, False, False, False),
                opponent_actions={"p2": 0},
                value_fn=lambda history: 0.0,
                action_priors=(0.5, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
                root_visit_budget=1,
            )

    def test_puct_branch_search_falls_back_to_uniform_when_legal_prior_mass_is_zero(self) -> None:
        env = ValueBranchEnv()
        trajectory = BattleTrajectory(battle_id="battle", format_id="gen3randombattle", seed=77)

        def value_fn(history: tuple[PokeZeroObservationV0, ...]) -> float:
            return {0: 0.1, 1: 0.4}[_only_legal_action(history[-1])]

        result = puct_branch_search(
            env=env,
            trajectory=trajectory,
            player_id="p1",
            prefix_decision_round_count=0,
            legal_action_mask=(True, True, False, False, False, False, False, False, False),
            opponent_actions={"p2": 0},
            value_fn=value_fn,
            action_priors=(0.0,) * ACTION_COUNT,
            cpuct=0.5,
        )

        self.assertEqual(result.action_index, 1)
        self.assertEqual([candidate.prior for candidate in result.candidates], [0.5, 0.5])

    def test_puct_branch_search_rejects_invalid_priors(self) -> None:
        with self.assertRaisesRegex(ValueError, "action_priors"):
            puct_branch_search(
                env=ValueBranchEnv(),
                trajectory=BattleTrajectory(battle_id="battle", format_id="gen3randombattle", seed=77),
                player_id="p1",
                prefix_decision_round_count=0,
                legal_action_mask=(True, False, False, False, False, False, False, False, False),
                opponent_actions={"p2": 0},
                value_fn=lambda history: 0.0,
                action_priors=(1.0,),
            )

    def test_flat_branch_search_selects_best_legal_action_from_rollout_outcomes(self) -> None:
        env = BranchOutcomeEnv({0: "p2", 1: "p1", 4: None})
        trajectory = BattleTrajectory(battle_id="battle", format_id="gen3randombattle", seed=77)

        result = flat_branch_search(
            env=env,
            trajectory=trajectory,
            player_id="p1",
            prefix_decision_round_count=0,
            legal_action_mask=(True, True, False, False, True, False, False, False, False),
            opponent_actions={"p2": 0},
            rollout_policies={"p1": RandomLegalPolicy(), "p2": RandomLegalPolicy()},
            rollout_config=RolloutConfig(max_decision_rounds=5),
        )

        self.assertEqual(result.action_index, 1)
        self.assertEqual([candidate.action_index for candidate in result.candidates], [0, 1, 4])
        self.assertEqual([candidate.value for candidate in result.candidates], [-1.0, 1.0, 0.0])
        self.assertEqual(
            env.all_step_calls,
            [
                {"p1": 0, "p2": 0},
                {"p1": 1, "p2": 0},
                {"p1": 4, "p2": 0},
            ],
        )
        self.assertEqual(result.to_dict()["selected_action_index"], 1)

    def test_flat_branch_search_rejects_bad_legal_mask_width(self) -> None:
        with self.assertRaisesRegex(ValueError, "legal_action_mask"):
            flat_branch_search(
                env=BranchOutcomeEnv({}),
                trajectory=BattleTrajectory(battle_id="battle", format_id="gen3randombattle", seed=77),
                player_id="p1",
                prefix_decision_round_count=0,
                legal_action_mask=(True,),
                opponent_actions={"p2": 0},
                rollout_policies={"p1": RandomLegalPolicy(), "p2": RandomLegalPolicy()},
                rollout_config=RolloutConfig(max_decision_rounds=5),
            )

    def test_flat_branch_search_rejects_player_in_opponent_actions(self) -> None:
        with self.assertRaisesRegex(ValueError, "must not include"):
            flat_branch_search(
                env=BranchOutcomeEnv({0: "p1"}),
                trajectory=BattleTrajectory(battle_id="battle", format_id="gen3randombattle", seed=77),
                player_id="p1",
                prefix_decision_round_count=0,
                legal_action_mask=(True, False, False, False, False, False, False, False, False),
                opponent_actions={"p1": 0, "p2": 0},
                rollout_policies={"p1": RandomLegalPolicy(), "p2": RandomLegalPolicy()},
                rollout_config=RolloutConfig(max_decision_rounds=5),
            )


def _only_legal_action(observation: PokeZeroObservationV0) -> int:
    return tuple(observation.legal_action_mask).index(True)


if __name__ == "__main__":
    unittest.main()
