import unittest

from pokezero.actions import ACTION_COUNT
from pokezero.env import StepResult, TerminalState
from pokezero.observation import PokeZeroObservationV0
from pokezero.policy import RandomLegalPolicy
from pokezero.rollout import RolloutConfig
from pokezero.search import flat_branch_search, terminal_value_for_player, value_branch_search
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
