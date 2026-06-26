import unittest
import os
from pathlib import Path
import shutil

from pokezero.actions import ACTION_COUNT
from pokezero.env import StepResult
from pokezero.local_showdown import DEFAULT_SHOWDOWN_ROOT, LocalShowdownConfig, LocalShowdownEnv
from pokezero.observation import PokeZeroObservationV0
from pokezero.policy import RandomLegalPolicy
from pokezero.replay_branching import (
    ReplayActionRound,
    action_rounds_from_trajectory,
    replay_action_rounds,
    replay_trajectory_prefix,
)
from pokezero.rollout import RolloutConfig, RolloutDriver
from pokezero.trajectory import BattleTrajectory, TrajectoryStep


def integration_config() -> LocalShowdownConfig | None:
    root = Path(os.environ.get("POKEZERO_SHOWDOWN_ROOT") or DEFAULT_SHOWDOWN_ROOT)
    if not (root / "dist" / "sim" / "index.js").exists():
        return None
    if shutil.which("node") is None:
        return None
    return LocalShowdownConfig(showdown_root=root, read_timeout_seconds=10.0)


def _without_timestamp_lines(lines: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(line for line in lines if not line.startswith("|t:|"))


def _observation(*, legal_action: int = 0) -> PokeZeroObservationV0:
    legal_action_mask = tuple(index == legal_action for index in range(ACTION_COUNT))
    return PokeZeroObservationV0(
        categorical_ids=(),
        numeric_features=(),
        token_type_ids=(),
        attention_mask=(),
        legal_action_mask=legal_action_mask,
    )


def _step(player_id: str, turn_index: int, action_index: int) -> TrajectoryStep:
    return TrajectoryStep(
        player_id=player_id,
        turn_index=turn_index,
        observation=_observation(legal_action=action_index),
        legal_action_mask=tuple(index == action_index for index in range(ACTION_COUNT)),
        action_index=action_index,
    )


class ScriptedReplayEnv:
    def __init__(self, requested_by_round: tuple[tuple[str, ...], ...]) -> None:
        self.requested_by_round = requested_by_round
        self.reset_calls: list[tuple[int, str]] = []
        self.submitted_actions: list[dict[str, int]] = []
        self.round_index = 0

    def reset(self, *, seed: int, format_id: str = "gen3randombattle") -> None:
        self.reset_calls.append((seed, format_id))
        self.submitted_actions = []
        self.round_index = 0

    def observe(self, player: str) -> PokeZeroObservationV0:
        return _observation(legal_action=0)

    def legal_actions(self, player: str) -> tuple[bool, ...]:
        return self.observe(player).legal_action_mask

    def requested_players(self) -> tuple[str, ...]:
        if self.round_index >= len(self.requested_by_round):
            return ()
        return self.requested_by_round[self.round_index]

    def step(self, actions):
        self.submitted_actions.append(dict(actions))
        self.round_index += 1
        return StepResult(
            observations={},
            rewards={"p1": 0.0, "p2": 0.0},
            terminal=None,
            requested_players=self.requested_players(),
        )

    def terminal(self):
        return None


class ReplayBranchingUnitTest(unittest.TestCase):
    def test_action_rounds_from_trajectory_groups_steps_by_decision_round(self) -> None:
        trajectory = BattleTrajectory(battle_id="battle", format_id="gen3randombattle", seed=123)
        trajectory.append(_step("p1", 0, 2))
        trajectory.append(_step("p2", 0, 3))
        trajectory.append(_step("p1", 1, 4))

        rounds = action_rounds_from_trajectory(trajectory)

        self.assertEqual(
            rounds,
            (
                ReplayActionRound(turn_index=0, actions={"p1": 2, "p2": 3}),
                ReplayActionRound(turn_index=1, actions={"p1": 4}),
            ),
        )

    def test_action_rounds_from_trajectory_can_take_prefix_only(self) -> None:
        trajectory = BattleTrajectory(battle_id="battle", format_id="gen3randombattle", seed=123)
        trajectory.append(_step("p1", 0, 2))
        trajectory.append(_step("p2", 0, 3))
        trajectory.append(_step("p1", 1, 4))

        rounds = action_rounds_from_trajectory(trajectory, decision_round_count=1)

        self.assertEqual(rounds, (ReplayActionRound(turn_index=0, actions={"p1": 2, "p2": 3}),))

    def test_action_rounds_from_trajectory_rejects_duplicate_player_rounds(self) -> None:
        trajectory = BattleTrajectory(battle_id="battle", format_id="gen3randombattle", seed=123)
        trajectory.append(_step("p1", 0, 2))
        trajectory.append(_step("p1", 0, 3))

        with self.assertRaisesRegex(ValueError, "duplicate action"):
            action_rounds_from_trajectory(trajectory)

    def test_action_rounds_from_trajectory_rejects_gaps(self) -> None:
        trajectory = BattleTrajectory(battle_id="battle", format_id="gen3randombattle", seed=123)
        trajectory.append(_step("p1", 1, 2))

        with self.assertRaisesRegex(ValueError, "contiguous"):
            action_rounds_from_trajectory(trajectory)

    def test_action_rounds_from_trajectory_rejects_missing_requested_prefix(self) -> None:
        trajectory = BattleTrajectory(battle_id="battle", format_id="gen3randombattle", seed=123)
        trajectory.append(_step("p1", 0, 2))

        with self.assertRaisesRegex(ValueError, "2 were requested"):
            action_rounds_from_trajectory(trajectory, decision_round_count=2)

    def test_replay_action_rounds_resets_and_submits_exact_requested_actions(self) -> None:
        env = ScriptedReplayEnv((("p1", "p2"), ("p1",)))

        result = replay_action_rounds(
            env,
            seed=17,
            format_id="gen3randombattle",
            action_rounds=(
                ReplayActionRound(turn_index=0, actions={"p1": 2, "p2": 3}),
                ReplayActionRound(turn_index=1, actions={"p1": 4}),
            ),
        )

        self.assertEqual(env.reset_calls, [(17, "gen3randombattle")])
        self.assertEqual(env.submitted_actions, [{"p1": 2, "p2": 3}, {"p1": 4}])
        self.assertEqual(result.replayed_round_count, 2)
        self.assertEqual(result.requested_players, ())

    def test_replay_action_rounds_rejects_request_mismatch(self) -> None:
        env = ScriptedReplayEnv((("p1",),))

        with self.assertRaisesRegex(ValueError, "unexpected players"):
            replay_action_rounds(
                env,
                seed=17,
                action_rounds=(ReplayActionRound(turn_index=0, actions={"p1": 2, "p2": 3}),),
            )


@unittest.skipIf(integration_config() is None, "requires node and built Pokemon Showdown checkout")
class ReplayBranchingIntegrationTest(unittest.TestCase):
    def test_replaying_prefix_recreates_original_branch_point_observations(self) -> None:
        trajectory = self._random_rollout_trajectory(seed=29, max_decision_rounds=40)
        prefix_round_count = 4
        self.assertGreaterEqual(len(trajectory.steps_for_turn(prefix_round_count)), 1)
        expected_observations = {
            step.player_id: step.observation for step in trajectory.steps_for_turn(prefix_round_count)
        }
        config = integration_config()
        assert config is not None

        with LocalShowdownEnv(config) as env:
            result = replay_trajectory_prefix(
                env,
                trajectory,
                decision_round_count=prefix_round_count,
            )
            actual_observations = {
                player: env.observe(player) for player in result.requested_players
            }

        self.assertEqual(set(actual_observations), set(expected_observations))
        self.assertEqual(actual_observations, expected_observations)

    def test_replaying_prefix_then_action_matches_full_replay_from_root(self) -> None:
        trajectory = self._random_rollout_trajectory(seed=31, max_decision_rounds=40)
        prefix_round_count = 5
        action_rounds = action_rounds_from_trajectory(
            trajectory,
            decision_round_count=prefix_round_count + 1,
        )
        branch_action = action_rounds[-1]
        config = integration_config()
        assert config is not None

        with LocalShowdownEnv(config) as full_replay:
            replay_trajectory_prefix(
                full_replay,
                trajectory,
                decision_round_count=prefix_round_count + 1,
            )
            full_replay_lines = full_replay.protocol_lines

        with LocalShowdownEnv(config) as branch_replay:
            replay_trajectory_prefix(
                branch_replay,
                trajectory,
                decision_round_count=prefix_round_count,
            )
            branch_replay.step(branch_action.actions)
            branch_replay_lines = branch_replay.protocol_lines

        self.assertEqual(
            _without_timestamp_lines(branch_replay_lines),
            _without_timestamp_lines(full_replay_lines),
        )

    def _random_rollout_trajectory(self, *, seed: int, max_decision_rounds: int) -> BattleTrajectory:
        config = integration_config()
        assert config is not None
        with LocalShowdownEnv(config) as env:
            result = RolloutDriver(
                env=env,
                policies={"p1": RandomLegalPolicy(), "p2": RandomLegalPolicy()},
                config=RolloutConfig(max_decision_rounds=max_decision_rounds),
            ).run(seed=seed)
        self.assertGreaterEqual(result.decision_round_count, 8)
        return result.trajectory


if __name__ == "__main__":
    unittest.main()
