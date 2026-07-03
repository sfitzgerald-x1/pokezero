import unittest
import os
from dataclasses import replace
from pathlib import Path
import shutil

from pokezero.actions import ACTION_COUNT
from pokezero.env import BattleStartOverride, DEFAULT_BATTLE_START_OVERRIDE_FORMAT, StepResult
from pokezero.local_showdown import DEFAULT_SHOWDOWN_ROOT, LocalShowdownConfig, LocalShowdownEnv
from pokezero.observation import (
    ACTION_CANDIDATE_TOKEN_COUNT,
    FIELD_TOKEN_COUNT,
    OPPONENT_POKEMON_TOKEN_COUNT,
    PokeZeroObservationV0,
    SELF_POKEMON_TOKEN_COUNT,
)
from pokezero.policy import RandomLegalPolicy
from pokezero.replay_branching import (
    ReplayActionRound,
    action_rounds_from_trajectory,
    replay_action_rounds,
    replay_trajectory_branch,
    replay_trajectory_branch_rollout,
    replay_trajectory_prefix,
)
from pokezero.rollout import RolloutConfig, RolloutDriver
from pokezero.showdown import DEFAULT_REPLAY_OBSERVATION_SPEC
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


def _observation_with_recent_event_token(*, legal_action: int = 0, recent_event_token: int = 0) -> PokeZeroObservationV0:
    spec = DEFAULT_REPLAY_OBSERVATION_SPEC
    recent_event_offset = (
        FIELD_TOKEN_COUNT
        + SELF_POKEMON_TOKEN_COUNT
        + OPPONENT_POKEMON_TOKEN_COUNT
        + ACTION_CANDIDATE_TOKEN_COUNT
    )
    categorical_ids = [[0] * spec.categorical_feature_count for _ in range(spec.token_count)]
    numeric_features = [[0.0] * spec.numeric_feature_count for _ in range(spec.token_count)]
    token_type_ids = [0] * spec.token_count
    attention_mask = [False] * spec.token_count
    categorical_ids[0][0] = 1
    numeric_features[0][0] = 1.0
    attention_mask[0] = True
    categorical_ids[recent_event_offset][0] = recent_event_token
    numeric_features[recent_event_offset][0] = 1.0
    attention_mask[recent_event_offset] = True
    legal_action_mask = tuple(index == legal_action for index in range(ACTION_COUNT))
    return PokeZeroObservationV0(
        categorical_ids=tuple(tuple(row) for row in categorical_ids),
        numeric_features=tuple(tuple(row) for row in numeric_features),
        token_type_ids=tuple(token_type_ids),
        attention_mask=tuple(attention_mask),
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


class StartOverrideReplayEnv(ScriptedReplayEnv):
    def __init__(self, requested_by_round: tuple[tuple[str, ...], ...]) -> None:
        super().__init__(requested_by_round)
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


class ObservationReplayEnv(StartOverrideReplayEnv):
    def __init__(self, requested_by_round: tuple[tuple[str, ...], ...], observation: PokeZeroObservationV0) -> None:
        super().__init__(requested_by_round)
        self.observation = observation

    def observe(self, player: str) -> PokeZeroObservationV0:
        return self.observation


class ReplayBranchingUnitTest(unittest.TestCase):
    def test_action_rounds_from_trajectory_groups_steps_by_decision_round(self) -> None:
        trajectory = BattleTrajectory(battle_id="battle", format_id="gen3randombattle", seed=123)
        trajectory.append(_step("p1", 0, 2))
        trajectory.append(_step("p2", 0, 3))
        trajectory.append(_step("p1", 1, 4))

        rounds = action_rounds_from_trajectory(trajectory)

        self.assertEqual([round.turn_index for round in rounds], [0, 1])
        self.assertEqual([round.actions for round in rounds], [{"p1": 2, "p2": 3}, {"p1": 4}])
        self.assertEqual(set(rounds[0].expected_observations or ()), {"p1", "p2"})
        self.assertEqual(set(rounds[1].expected_observations or ()), {"p1"})

    def test_action_rounds_from_trajectory_can_take_prefix_only(self) -> None:
        trajectory = BattleTrajectory(battle_id="battle", format_id="gen3randombattle", seed=123)
        trajectory.append(_step("p1", 0, 2))
        trajectory.append(_step("p2", 0, 3))
        trajectory.append(_step("p1", 1, 4))

        rounds = action_rounds_from_trajectory(trajectory, decision_round_count=1)

        self.assertEqual(len(rounds), 1)
        self.assertEqual(rounds[0].turn_index, 0)
        self.assertEqual(rounds[0].actions, {"p1": 2, "p2": 3})
        self.assertEqual(set(rounds[0].expected_observations or ()), {"p1", "p2"})

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

    def test_replay_action_rounds_passes_start_override_before_prefix_actions(self) -> None:
        env = StartOverrideReplayEnv((("p1", "p2"),))
        start_override = BattleStartOverride(
            player_teams={"p1": "Charizard||||Tackle|||||||", "p2": "Xatu||||Psychic|||||||"}
        )

        replay_action_rounds(
            env,
            seed=17,
            format_id="gen3randombattle",
            action_rounds=(ReplayActionRound(turn_index=0, actions={"p1": 2, "p2": 3}),),
            start_override=start_override,
        )

        self.assertEqual(env.start_overrides, [start_override])
        self.assertEqual(env.reset_calls, [(17, DEFAULT_BATTLE_START_OVERRIDE_FORMAT)])
        self.assertEqual(env.submitted_actions, [{"p1": 2, "p2": 3}])

    def test_replay_action_rounds_rejects_start_override_without_env_support(self) -> None:
        with self.assertRaisesRegex(ValueError, "start overrides"):
            replay_action_rounds(
                ScriptedReplayEnv((("p1",),)),
                seed=17,
                action_rounds=(ReplayActionRound(turn_index=0, actions={"p1": 2}),),
                start_override=BattleStartOverride(
                    player_teams={"p1": "Charizard||||Tackle|||||||", "p2": "Xatu||||Psychic|||||||"}
                ),
            )

    def test_replay_action_rounds_rejects_start_override_prefix_observation_mismatch(self) -> None:
        env = StartOverrideReplayEnv((("p1",),))
        start_override = BattleStartOverride(
            player_teams={"p1": "Charizard||||Tackle|||||||", "p2": "Xatu||||Psychic|||||||"}
        )

        with self.assertRaisesRegex(ValueError, "does not reproduce recorded replay prefix observations"):
            replay_action_rounds(
                env,
                seed=17,
                action_rounds=(
                    ReplayActionRound(
                        turn_index=0,
                        actions={"p1": 2},
                        expected_observations={"p1": _observation(legal_action=2)},
                    ),
                ),
                start_override=start_override,
                consistency_player_id="p1",
            )

    def test_replay_action_rounds_start_override_consistency_ignores_metadata_only_differences(self) -> None:
        env = StartOverrideReplayEnv((("p1",),))
        start_override = BattleStartOverride(
            player_teams={"p1": "Charizard||||Tackle|||||||", "p2": "Xatu||||Psychic|||||||"}
        )
        expected = replace(_observation(legal_action=0), metadata={"battle_id": "recorded"})

        replay_action_rounds(
            env,
            seed=17,
            action_rounds=(
                ReplayActionRound(
                    turn_index=0,
                    actions={"p1": 0},
                    expected_observations={"p1": expected},
                ),
            ),
            start_override=start_override,
            consistency_player_id="p1",
        )

        self.assertEqual(env.submitted_actions, [{"p1": 0}])

    def test_replay_action_rounds_start_override_consistency_checks_only_selected_player(self) -> None:
        env = StartOverrideReplayEnv((("p1", "p2"),))
        start_override = BattleStartOverride(
            player_teams={"p1": "Charizard||||Tackle|||||||", "p2": "Xatu||||Psychic|||||||"}
        )

        replay_action_rounds(
            env,
            seed=17,
            action_rounds=(
                ReplayActionRound(
                    turn_index=0,
                    actions={"p1": 0, "p2": 0},
                    expected_observations={
                        "p1": _observation(legal_action=0),
                        "p2": _observation(legal_action=2),
                    },
                ),
            ),
            start_override=start_override,
            consistency_player_id="p1",
        )

        self.assertEqual(env.submitted_actions, [{"p1": 0, "p2": 0}])

    def test_start_override_consistency_ignores_recent_event_token_drift(self) -> None:
        env = ObservationReplayEnv((("p1",),), _observation_with_recent_event_token(recent_event_token=2))
        start_override = BattleStartOverride(
            player_teams={"p1": "Charizard||||Tackle|||||||", "p2": "Xatu||||Psychic|||||||"}
        )

        replay_action_rounds(
            env,
            seed=17,
            action_rounds=(
                ReplayActionRound(
                    turn_index=0,
                    actions={"p1": 0},
                    expected_observations={
                        "p1": _observation_with_recent_event_token(recent_event_token=1),
                    },
                ),
            ),
            start_override=start_override,
            consistency_player_id="p1",
        )

        self.assertEqual(env.submitted_actions, [{"p1": 0}])

    def test_replay_action_rounds_rejects_request_mismatch(self) -> None:
        env = ScriptedReplayEnv((("p1",),))

        with self.assertRaisesRegex(ValueError, "unexpected players"):
            replay_action_rounds(
                env,
                seed=17,
                action_rounds=(ReplayActionRound(turn_index=0, actions={"p1": 2, "p2": 3}),),
            )

    def test_replay_trajectory_branch_submits_action_after_prefix(self) -> None:
        trajectory = BattleTrajectory(battle_id="battle", format_id="gen3randombattle", seed=123)
        trajectory.append(_step("p1", 0, 2))
        trajectory.append(_step("p2", 0, 3))
        env = ScriptedReplayEnv((("p1", "p2"), ("p1",)))

        result = replay_trajectory_branch(
            env,
            trajectory,
            prefix_decision_round_count=1,
            branch_actions={"p1": 4},
        )

        self.assertEqual(env.submitted_actions, [{"p1": 2, "p2": 3}, {"p1": 4}])
        self.assertEqual(result.prefix.replayed_round_count, 1)
        self.assertEqual(result.branch_round, ReplayActionRound(turn_index=1, actions={"p1": 4}))
        self.assertEqual(result.step_result.requested_players, ())

    def test_replay_trajectory_branch_rejects_request_mismatch(self) -> None:
        trajectory = BattleTrajectory(battle_id="battle", format_id="gen3randombattle", seed=123)
        env = ScriptedReplayEnv((("p1",),))

        with self.assertRaisesRegex(ValueError, "unexpected players"):
            replay_trajectory_branch(
                env,
                trajectory,
                prefix_decision_round_count=0,
                branch_actions={"p1": 4, "p2": 3},
            )

    def test_replay_trajectory_branch_checks_start_override_current_observation_at_zero_prefix(self) -> None:
        trajectory = BattleTrajectory(battle_id="battle", format_id="gen3randombattle", seed=123)
        env = StartOverrideReplayEnv((("p1",),))
        start_override = BattleStartOverride(
            player_teams={"p1": "Charizard||||Tackle|||||||", "p2": "Xatu||||Psychic|||||||"}
        )

        with self.assertRaisesRegex(ValueError, "decision round 0: p1"):
            replay_trajectory_branch(
                env,
                trajectory,
                prefix_decision_round_count=0,
                branch_actions={"p1": 0},
                start_override=start_override,
                consistency_player_id="p1",
                expected_current_observation=_observation(legal_action=2),
            )

    def test_replay_trajectory_branch_rollout_continues_after_branch_action(self) -> None:
        trajectory = BattleTrajectory(battle_id="battle", format_id="gen3randombattle", seed=123)
        trajectory.append(_step("p1", 0, 2))
        trajectory.append(_step("p2", 0, 3))
        env = ScriptedReplayEnv((("p1", "p2"), ("p1",), ("p1",)))

        result = replay_trajectory_branch_rollout(
            env,
            trajectory,
            prefix_decision_round_count=1,
            branch_actions={"p1": 4},
            policies={"p1": RandomLegalPolicy()},
            rollout_config=RolloutConfig(max_decision_rounds=3),
        )

        self.assertEqual(env.submitted_actions, [{"p1": 2, "p2": 3}, {"p1": 4}, {"p1": 0}])
        self.assertEqual(result.branch.branch_round, ReplayActionRound(turn_index=1, actions={"p1": 4}))
        self.assertTrue(result.continuation.terminal.capped)
        self.assertEqual(result.continuation.decision_round_count, 1)
        self.assertEqual([step.turn_index for step in result.continuation.trajectory.steps], [2])


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
            replay_trajectory_branch(
                branch_replay,
                trajectory,
                prefix_decision_round_count=prefix_round_count,
                branch_actions=branch_action.actions,
            )
            branch_replay_lines = branch_replay.protocol_lines

        self.assertEqual(
            _without_timestamp_lines(branch_replay_lines),
            _without_timestamp_lines(full_replay_lines),
        )

    def test_replaying_prefix_with_divergent_action_explores_different_line(self) -> None:
        trajectory = self._random_rollout_trajectory(seed=37, max_decision_rounds=40)
        original_round = action_rounds_from_trajectory(
            trajectory,
            decision_round_count=1,
        )[0]
        p1_step = next(step for step in trajectory.steps_for_turn(0) if step.player_id == "p1")
        alternate = next(
            (
                action_index
                for action_index, legal in enumerate(p1_step.legal_action_mask)
                if legal and action_index != p1_step.action_index
            ),
            None,
        )
        if alternate is None:
            self.skipTest("source battle did not expose an alternate legal p1 action at turn 0")
        divergent_actions = dict(original_round.actions)
        divergent_actions["p1"] = alternate
        config = integration_config()
        assert config is not None

        with LocalShowdownEnv(config) as original_replay:
            replay_trajectory_branch(
                original_replay,
                trajectory,
                prefix_decision_round_count=0,
                branch_actions=original_round.actions,
            )
            original_lines = original_replay.protocol_lines

        with LocalShowdownEnv(config) as divergent_replay:
            result = replay_trajectory_branch(
                divergent_replay,
                trajectory,
                prefix_decision_round_count=0,
                branch_actions=divergent_actions,
            )
            divergent_lines = divergent_replay.protocol_lines

        self.assertEqual(result.prefix.replayed_round_count, 0)
        self.assertEqual(result.branch_round.actions["p1"], alternate)
        self.assertNotEqual(
            _without_timestamp_lines(divergent_lines),
            _without_timestamp_lines(original_lines),
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
