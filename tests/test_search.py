import time
import unittest
from unittest.mock import patch

from pokezero.actions import ACTION_COUNT
from pokezero.env import BattleStartOverride, StepResult, TerminalState
from pokezero.observation import PokeZeroObservationV0
from pokezero.policy import RandomLegalPolicy
from pokezero.policy import PolicyDecision
from pokezero.rollout import RolloutConfig
from pokezero.search import (
    PUCTBranchSearchRequest,
    RootPUCTSearchTiming,
    _branch_step_timing_snapshot,
    _is_candidate_illegal_action_error,
    flat_branch_search,
    puct_branch_search,
    puct_branch_search_group,
    prepare_direct_materialization_prefix,
    prepare_replay_prefix,
    release_prepared_replay_prefix,
    terminal_value_for_player,
    value_branch_search,
)
from pokezero.trajectory import BattleTrajectory, TrajectoryStep


def _observation(action_index: int) -> PokeZeroObservationV0:
    return PokeZeroObservationV0(
        categorical_ids=(),
        numeric_features=(),
        token_type_ids=(),
        attention_mask=(),
        legal_action_mask=tuple(index == action_index for index in range(ACTION_COUNT)),
    )


def _start_override() -> BattleStartOverride:
    return BattleStartOverride(
        player_teams={"p1": "Charizard||||Tackle|||||||", "p2": "Xatu||||Psychic|||||||"}
    )


class CandidateIllegalActionErrorTest(unittest.TestCase):
    def test_accepts_legacy_and_request_kind_qualified_errors_for_the_candidate(self) -> None:
        for message in (
            "action_index 4 is not legal for the current request.",
            "p1: action_index 4 is not legal for the current request.",
            "action_index 4 is not legal for the current request (request_kind=force_switch).",
            "p1: action_index 4 is not legal for the current request (request_kind=force_switch).",
        ):
            with self.subTest(message=message):
                self.assertTrue(
                    _is_candidate_illegal_action_error(
                        ValueError(message), player_id="p1", action_index=4
                    )
                )

    def test_rejects_qualified_errors_for_a_different_action_or_player(self) -> None:
        self.assertFalse(
            _is_candidate_illegal_action_error(
                ValueError("p2: action_index 4 is not legal for the current request (request_kind=move)."),
                player_id="p1",
                action_index=4,
            )
        )
        self.assertFalse(
            _is_candidate_illegal_action_error(
                ValueError("p1: action_index 3 is not legal for the current request (request_kind=move)."),
                player_id="p1",
                action_index=4,
            )
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

    def reset_with_start_override(
        self,
        *,
        seed: int,
        format_id: str | None = None,
        start_override: BattleStartOverride,
    ) -> None:
        del start_override
        self.reset(seed=seed, format_id=format_id or "gen3customgame")

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


class SnapshotValueBranchEnv(ValueBranchEnv):
    def __init__(self, terminal_winners_by_action: dict[int, str | None] | None = None) -> None:
        super().__init__(terminal_winners_by_action)
        self.snapshot_calls = 0
        self.restore_calls = 0

    def snapshot(self):
        self.snapshot_calls += 1
        return self._requested, self._terminal

    def restore(self, snapshot) -> None:
        self.restore_calls += 1
        self._requested, self._terminal = snapshot


class BridgeSnapshotValueBranchEnv(SnapshotValueBranchEnv):
    """Test double for the search-only bridge-resident snapshot API."""

    def __init__(self, terminal_winners_by_action: dict[int, str | None] | None = None) -> None:
        super().__init__(terminal_winners_by_action)
        self.search_snapshot_calls = 0
        self.search_restore_calls = 0
        self.released_search_snapshots: list[tuple[tuple[str, ...], TerminalState | None]] = []

    def snapshot_for_search(self):
        self.search_snapshot_calls += 1
        return self._requested, self._terminal

    def restore_search_snapshot(self, snapshot) -> None:
        self.search_restore_calls += 1
        self._requested, self._terminal = snapshot

    def release_search_snapshot(self, snapshot) -> bool:
        self.released_search_snapshots.append(snapshot)
        return True


class BridgeTimedSnapshotValueBranchEnv(BridgeSnapshotValueBranchEnv):
    """Bridge-handle double exposing cumulative W5 transport counters."""

    def __init__(self, terminal_winners_by_action: dict[int, str | None] | None = None) -> None:
        super().__init__(terminal_winners_by_action)
        self._bridge_round_trip_seconds = 0.0
        self._bridge_round_trip_count = 0
        self._bridge_node_processing_seconds = 0.0
        self._bridge_node_processing_count = 0

    def root_puct_bridge_timing_snapshot(self) -> dict[str, float | int]:
        return {
            "bridge_round_trip_seconds": self._bridge_round_trip_seconds,
            "bridge_round_trip_count": self._bridge_round_trip_count,
            "bridge_node_processing_seconds": self._bridge_node_processing_seconds,
            "bridge_node_processing_count": self._bridge_node_processing_count,
        }

    def _record_bridge_round_trip(self, *, elapsed_seconds: float, node_seconds: float) -> None:
        self._bridge_round_trip_seconds += elapsed_seconds
        self._bridge_round_trip_count += 1
        self._bridge_node_processing_seconds += node_seconds
        self._bridge_node_processing_count += 1

    def restore_search_snapshot(self, snapshot) -> None:
        self._record_bridge_round_trip(elapsed_seconds=0.010, node_seconds=0.006)
        super().restore_search_snapshot(snapshot)

    def step(self, actions: dict[str, int]) -> StepResult:
        self._record_bridge_round_trip(elapsed_seconds=0.020, node_seconds=0.012)
        return super().step(actions)


class FusedBridgeTimedSnapshotValueBranchEnv(BridgeTimedSnapshotValueBranchEnv):
    """Bridge-handle double that restores and branches in one transport exchange."""

    def __init__(self, terminal_winners_by_action: dict[int, str | None] | None = None) -> None:
        super().__init__(terminal_winners_by_action)
        self.fused_search_step_calls = 0
        self._branch_local_state_restore_seconds = 0.0
        self._branch_local_state_restore_count = 0
        self._branch_choice_encoding_seconds = 0.0
        self._branch_choice_encoding_count = 0
        self._branch_bridge_round_trip_seconds = 0.0
        self._branch_bridge_round_trip_count = 0
        self._branch_bridge_node_processing_seconds = 0.0
        self._branch_bridge_node_processing_count = 0
        self._branch_result_projection_seconds = 0.0
        self._branch_result_projection_count = 0
        self._branch_observation_projection_seconds = 0.0
        self._branch_observation_projection_count = 0
        self._branch_observation_state_normalization_seconds = 0.0
        self._branch_observation_state_normalization_count = 0
        self._branch_observation_incremental_sync_seconds = 0.0
        self._branch_observation_incremental_sync_count = 0
        self._branch_observation_replay_snapshot_seconds = 0.0
        self._branch_observation_replay_snapshot_count = 0
        self._branch_observation_player_state_normalization_seconds = 0.0
        self._branch_observation_player_state_normalization_count = 0
        self._branch_observation_state_annotation_seconds = 0.0
        self._branch_observation_state_annotation_count = 0
        self._branch_observation_encoding_seconds = 0.0
        self._branch_observation_encoding_count = 0
        self._branch_belief_overlay_projection_seconds = 0.0
        self._branch_belief_overlay_projection_count = 0

    def root_puct_branch_step_timing_snapshot(self) -> dict[str, float | int]:
        return {
            "branch_local_state_restore_seconds": self._branch_local_state_restore_seconds,
            "branch_local_state_restore_count": self._branch_local_state_restore_count,
            "branch_choice_encoding_seconds": self._branch_choice_encoding_seconds,
            "branch_choice_encoding_count": self._branch_choice_encoding_count,
            "branch_bridge_round_trip_seconds": self._branch_bridge_round_trip_seconds,
            "branch_bridge_round_trip_count": self._branch_bridge_round_trip_count,
            "branch_bridge_node_processing_seconds": self._branch_bridge_node_processing_seconds,
            "branch_bridge_node_processing_count": self._branch_bridge_node_processing_count,
            "branch_result_projection_seconds": self._branch_result_projection_seconds,
            "branch_result_projection_count": self._branch_result_projection_count,
            "branch_observation_projection_seconds": self._branch_observation_projection_seconds,
            "branch_observation_projection_count": self._branch_observation_projection_count,
            "branch_observation_state_normalization_seconds": (
                self._branch_observation_state_normalization_seconds
            ),
            "branch_observation_state_normalization_count": (
                self._branch_observation_state_normalization_count
            ),
            "branch_observation_incremental_sync_seconds": (
                self._branch_observation_incremental_sync_seconds
            ),
            "branch_observation_incremental_sync_count": (
                self._branch_observation_incremental_sync_count
            ),
            "branch_observation_replay_snapshot_seconds": (
                self._branch_observation_replay_snapshot_seconds
            ),
            "branch_observation_replay_snapshot_count": (
                self._branch_observation_replay_snapshot_count
            ),
            "branch_observation_player_state_normalization_seconds": (
                self._branch_observation_player_state_normalization_seconds
            ),
            "branch_observation_player_state_normalization_count": (
                self._branch_observation_player_state_normalization_count
            ),
            "branch_observation_state_annotation_seconds": (
                self._branch_observation_state_annotation_seconds
            ),
            "branch_observation_state_annotation_count": (
                self._branch_observation_state_annotation_count
            ),
            "branch_observation_encoding_seconds": self._branch_observation_encoding_seconds,
            "branch_observation_encoding_count": self._branch_observation_encoding_count,
            "branch_belief_overlay_projection_seconds": (
                self._branch_belief_overlay_projection_seconds
            ),
            "branch_belief_overlay_projection_count": (
                self._branch_belief_overlay_projection_count
            ),
        }

    def step_from_search_snapshot(self, snapshot, actions: dict[str, int]) -> StepResult:
        self.fused_search_step_calls += 1
        self._branch_local_state_restore_seconds += 0.003
        self._branch_local_state_restore_count += 1
        self._branch_choice_encoding_seconds += 0.002
        self._branch_choice_encoding_count += 1
        self._record_bridge_round_trip(elapsed_seconds=0.020, node_seconds=0.012)
        self._branch_bridge_round_trip_seconds += 0.020
        self._branch_bridge_round_trip_count += 1
        self._branch_bridge_node_processing_seconds += 0.012
        self._branch_bridge_node_processing_count += 1
        self._requested, self._terminal = snapshot
        result = ValueBranchEnv.step(self, actions)
        self._branch_result_projection_seconds += 0.004
        self._branch_result_projection_count += 1
        self._branch_observation_projection_seconds += 0.003
        self._branch_observation_projection_count += 1
        self._branch_observation_state_normalization_seconds += 0.001
        self._branch_observation_state_normalization_count += 1
        self._branch_observation_incremental_sync_seconds += 0.0001
        self._branch_observation_incremental_sync_count += 1
        self._branch_observation_replay_snapshot_seconds += 0.0002
        self._branch_observation_replay_snapshot_count += 1
        self._branch_observation_player_state_normalization_seconds += 0.0006
        self._branch_observation_player_state_normalization_count += 1
        self._branch_observation_state_annotation_seconds += 0.0001
        self._branch_observation_state_annotation_count += 1
        self._branch_observation_encoding_seconds += 0.0015
        self._branch_observation_encoding_count += 1
        self._branch_belief_overlay_projection_seconds += 0.0005
        self._branch_belief_overlay_projection_count += 1
        return result


class PlayerObservationFusedBridgeEnv(FusedBridgeTimedSnapshotValueBranchEnv):
    """Bridge-handle double exposing the zero-rollout single-view fast path."""

    def __init__(self, terminal_winners_by_action: dict[int, str | None] | None = None) -> None:
        super().__init__(terminal_winners_by_action)
        self.player_observation_calls: list[str] = []

    def step_from_search_snapshot_for_player(
        self,
        snapshot,
        actions: dict[str, int],
        *,
        observation_player: str,
    ) -> StepResult:
        self.player_observation_calls.append(observation_player)
        return self.step_from_search_snapshot(snapshot, actions)


class BranchStepTimingCompatibilityTest(unittest.TestCase):
    def test_missing_nested_observation_slices_default_to_zero(self) -> None:
        env = FusedBridgeTimedSnapshotValueBranchEnv()
        legacy_payload = env.root_puct_branch_step_timing_snapshot()
        for field in (
            "branch_observation_state_normalization_seconds",
            "branch_observation_state_normalization_count",
            "branch_observation_incremental_sync_seconds",
            "branch_observation_incremental_sync_count",
            "branch_observation_replay_snapshot_seconds",
            "branch_observation_replay_snapshot_count",
            "branch_observation_player_state_normalization_seconds",
            "branch_observation_player_state_normalization_count",
            "branch_observation_state_annotation_seconds",
            "branch_observation_state_annotation_count",
            "branch_observation_encoding_seconds",
            "branch_observation_encoding_count",
            "branch_belief_overlay_projection_seconds",
            "branch_belief_overlay_projection_count",
        ):
            legacy_payload.pop(field)
        env.root_puct_branch_step_timing_snapshot = lambda: legacy_payload  # type: ignore[method-assign]

        timing = _branch_step_timing_snapshot(env)

        self.assertIsNotNone(timing)
        assert timing is not None
        self.assertEqual(timing["branch_observation_state_normalization_seconds"], 0.0)
        self.assertEqual(timing["branch_observation_incremental_sync_seconds"], 0.0)
        self.assertEqual(timing["branch_observation_replay_snapshot_seconds"], 0.0)
        self.assertEqual(timing["branch_observation_player_state_normalization_seconds"], 0.0)
        self.assertEqual(timing["branch_observation_state_annotation_seconds"], 0.0)
        self.assertEqual(timing["branch_observation_encoding_seconds"], 0.0)
        self.assertEqual(timing["branch_belief_overlay_projection_seconds"], 0.0)
        self.assertEqual(timing["branch_observation_state_normalization_count"], 0)
        self.assertEqual(timing["branch_observation_incremental_sync_count"], 0)
        self.assertEqual(timing["branch_observation_replay_snapshot_count"], 0)
        self.assertEqual(timing["branch_observation_player_state_normalization_count"], 0)
        self.assertEqual(timing["branch_observation_state_annotation_count"], 0)
        self.assertEqual(timing["branch_observation_encoding_seconds"], 0.0)
        self.assertEqual(timing["branch_belief_overlay_projection_seconds"], 0.0)
        self.assertEqual(timing["branch_observation_state_normalization_count"], 0)
        self.assertEqual(timing["branch_observation_encoding_count"], 0)
        self.assertEqual(timing["branch_belief_overlay_projection_count"], 0)


class TimedSnapshotValueBranchEnv(SnapshotValueBranchEnv):
    def snapshot(self):
        time.sleep(0.001)
        return super().snapshot()

    def restore(self, snapshot) -> None:
        time.sleep(0.001)
        super().restore(snapshot)


class OpponentConditionalTerminalValueBranchEnv(TimedSnapshotValueBranchEnv):
    """Make one sampled opponent world terminate while another needs value leaves."""

    def __init__(self, terminal_winners: dict[tuple[int, int], str | None]) -> None:
        super().__init__()
        self.terminal_winners = terminal_winners

    def step(self, actions: dict[str, int]) -> StepResult:
        self.all_step_calls.append(dict(actions))
        p1_action = int(actions["p1"])
        p2_action = int(actions["p2"])
        winner = self.terminal_winners.get((p2_action, p1_action))
        if (p2_action, p1_action) in self.terminal_winners:
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


class DirectMaterializationValueBranchEnv(SnapshotValueBranchEnv):
    def __init__(
        self,
        *,
        reject_materialization: bool = False,
        rejection_message: str = "unsupported public state",
    ) -> None:
        super().__init__()
        self.reject_materialization = reject_materialization
        self.rejection_message = rejection_message
        self.materialize_calls: list[tuple[object, BattleStartOverride, int]] = []

    def materialize_public_world(self, *, state: object, start_override: BattleStartOverride, seed: int) -> None:
        self.materialize_calls.append((state, start_override, seed))
        if self.reject_materialization:
            raise RuntimeError(self.rejection_message)
        self._terminal = None
        self._requested = ("p1", "p2")


class StrictLegalValueBranchEnv(ValueBranchEnv):
    def __init__(self, legal_actions: set[int]) -> None:
        super().__init__()
        self.strict_legal_actions = set(legal_actions)

    def step(self, actions: dict[str, int]) -> StepResult:
        p1_action = int(actions["p1"])
        if p1_action not in self.strict_legal_actions:
            raise ValueError(f"p1: action_index {p1_action} is not legal for the current request.")
        return super().step(actions)


class OpponentIllegalActionEnv(ValueBranchEnv):
    def step(self, actions: dict[str, int]) -> StepResult:
        p2_action = int(actions["p2"])
        raise ValueError(f"p2: action_index {p2_action} is not legal for the current request.")


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


class RngRecordingPolicy:
    policy_id = "rng-recording"

    def __init__(self) -> None:
        self.samples_by_branch_action: dict[int, list[float]] = {}

    def select_action(self, observation: PokeZeroObservationV0, *, rng) -> PolicyDecision:
        branch_action = _only_legal_action(observation)
        self.samples_by_branch_action.setdefault(branch_action, []).append(rng.random())
        return PolicyDecision(action_index=branch_action, policy_id=self.policy_id)


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

    def test_value_branch_search_requires_current_observation_for_start_override(self) -> None:
        with self.assertRaisesRegex(ValueError, "expected_current_observation"):
            value_branch_search(
                env=ValueBranchEnv({0: "p1"}),
                trajectory=BattleTrajectory(battle_id="battle", format_id="gen3randombattle", seed=77),
                player_id="p1",
                prefix_decision_round_count=0,
                legal_action_mask=(True, False, False, False, False, False, False, False, False),
                opponent_actions={"p2": 0},
                value_fn=lambda history: 0.0,
                start_override=_start_override(),
            )

    def test_value_branch_search_rejects_callable_start_override_that_returns_none(self) -> None:
        with self.assertRaisesRegex(ValueError, "start override source did not produce a sampled world"):
            value_branch_search(
                env=ValueBranchEnv({0: "p1"}),
                trajectory=BattleTrajectory(battle_id="battle", format_id="gen3randombattle", seed=77),
                player_id="p1",
                prefix_decision_round_count=0,
                legal_action_mask=(True, False, False, False, False, False, False, False, False),
                opponent_actions={"p2": 0},
                value_fn=lambda history: 0.0,
                start_override=lambda: None,
                expected_current_observation=_observation(0),
            )

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

    def test_value_branch_search_restores_replayed_prefix_when_env_supports_snapshots(self) -> None:
        env = SnapshotValueBranchEnv()
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

        result = value_branch_search(
            env=env,
            trajectory=trajectory,
            player_id="p1",
            prefix_decision_round_count=1,
            legal_action_mask=(True, True, False, False, True, False, False, False, False),
            opponent_actions={"p2": 0},
            value_fn=lambda history: float(_only_legal_action(history[-1])),
        )

        self.assertEqual([candidate.action_index for candidate in result.candidates], [0, 1, 4])
        self.assertEqual(len(env.reset_calls), 1)
        self.assertEqual(env.snapshot_calls, 1)
        self.assertEqual(env.restore_calls, 3)
        self.assertEqual(len(env.all_step_calls), 4)
        self.assertEqual(env.all_step_calls[0], {"p1": 0, "p2": 0})

    def test_direct_materialization_prepares_a_sampled_world_without_prefix_replay(self) -> None:
        env = DirectMaterializationValueBranchEnv()
        trajectory = BattleTrajectory(battle_id="battle", format_id="gen3randombattle", seed=77)
        public_state = object()

        prepared = prepare_direct_materialization_prefix(
            env=env,
            trajectory=trajectory,
            player_id="p1",
            prefix_decision_round_count=0,
            start_override=_start_override(),
            public_materialization_state=public_state,
            expected_current_observation=_observation(0),
        )

        self.assertIsNotNone(prepared)
        assert prepared is not None
        self.assertEqual(prepared.materialization_mode, "direct")
        self.assertEqual(
            prepared.world_legal_action_masks,
            {"p1": _observation(0).legal_action_mask, "p2": _observation(0).legal_action_mask},
        )
        self.assertEqual(env.materialize_calls, [(public_state, _start_override(), 77)])
        self.assertEqual(env.reset_calls, [])
        self.assertEqual(env.snapshot_calls, 1)

    def test_direct_materialization_fails_closed_to_tier_one(self) -> None:
        env = DirectMaterializationValueBranchEnv(reject_materialization=True)
        trajectory = BattleTrajectory(battle_id="battle", format_id="gen3randombattle", seed=77)
        rejection_categories: list[str] = []
        mismatch_paths: list[str] = []

        prepared = prepare_direct_materialization_prefix(
            env=env,
            trajectory=trajectory,
            player_id="p1",
            prefix_decision_round_count=0,
            start_override=_start_override(),
            public_materialization_state=object(),
            expected_current_observation=_observation(0),
            on_unavailable=rejection_categories.append,
            on_observation_mismatch_path=mismatch_paths.append,
        )

        self.assertIsNone(prepared)
        self.assertEqual(len(env.materialize_calls), 1)
        self.assertEqual(rejection_categories, ["materializer_error"])
        self.assertEqual(mismatch_paths, [])

    def test_direct_materialization_records_only_the_first_safe_observation_path(self) -> None:
        env = DirectMaterializationValueBranchEnv(
            reject_materialization=True,
            rejection_message=(
                "start override does not reproduce recorded replay prefix observations "
                "for decision round 3: p1. "
                "(categorical_ids/opponent_pokemon[8][11]: actual=76 expected=0)"
            ),
        )
        trajectory = BattleTrajectory(battle_id="battle", format_id="gen3randombattle", seed=77)
        rejection_categories: list[str] = []
        mismatch_paths: list[str] = []

        prepared = prepare_direct_materialization_prefix(
            env=env,
            trajectory=trajectory,
            player_id="p1",
            prefix_decision_round_count=3,
            start_override=_start_override(),
            public_materialization_state=object(),
            expected_current_observation=_observation(0),
            on_unavailable=rejection_categories.append,
            on_observation_mismatch_path=mismatch_paths.append,
        )

        self.assertIsNone(prepared)
        self.assertEqual(rejection_categories, ["observation_mismatch"])
        self.assertEqual(mismatch_paths, ["categorical_ids/opponent_pokemon[8][11]"])

    def test_puct_branch_search_reuses_prepared_sampled_world_prefix_without_replay(self) -> None:
        env = TimedSnapshotValueBranchEnv()
        trajectory = BattleTrajectory(battle_id="battle", format_id="gen3randombattle", seed=77)
        prepared_prefix = prepare_replay_prefix(
            env=env,
            trajectory=trajectory,
            player_id="p1",
            prefix_decision_round_count=0,
            start_override=_start_override(),
            expected_current_observation=_observation(0),
        )
        self.assertIsNotNone(prepared_prefix)

        result = puct_branch_search(
            env=env,
            trajectory=trajectory,
            player_id="p1",
            prefix_decision_round_count=0,
            legal_action_mask=(True, True, False, False, False, False, False, False, False),
            opponent_actions={"p2": 0},
            value_fn=lambda history: float(_only_legal_action(history[-1])),
            action_priors=(0.9, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
            cpuct=2.0,
            root_visit_budget=5,
            start_override=_start_override(),
            expected_current_observation=_observation(0),
            prepared_prefix=prepared_prefix,
        )

        self.assertEqual(result.total_visits, 5)
        self.assertEqual(len(env.reset_calls), 1)
        self.assertEqual(env.snapshot_calls, 1)
        self.assertEqual(env.restore_calls, 5)
        timing = result.timing.to_dict()
        self.assertEqual(timing["prefix_replay_count"], 0)
        self.assertEqual(timing["state_snapshot_count"], 0)
        self.assertEqual(timing["state_restore_count"], 5)

    def test_puct_branch_search_batches_only_the_independent_initial_leaves(self) -> None:
        trajectory = BattleTrajectory(battle_id="batch", format_id="gen3randombattle", seed=77)
        baseline_scalar_histories = []
        scalar_histories = []
        batch_histories = []

        def scalar_baseline_value(history):
            baseline_scalar_histories.append(history)
            return float(_only_legal_action(history[-1]))

        def scalar_value(history):
            scalar_histories.append(history)
            return float(_only_legal_action(history[-1]))

        def batch_values(histories):
            batch_histories.append(histories)
            return tuple(float(_only_legal_action(history[-1])) for history in histories)

        common_kwargs = {
            "trajectory": trajectory,
            "player_id": "p1",
            "prefix_decision_round_count": 0,
            "legal_action_mask": (True, True, False, False, False, False, False, False, False),
            "opponent_actions": {"p2": 0},
            "action_priors": (0.9, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
            "cpuct": 2.0,
            "root_visit_budget": 5,
        }
        baseline = puct_branch_search(
            env=TimedSnapshotValueBranchEnv(),
            value_fn=scalar_baseline_value,
            **common_kwargs,
        )
        result = puct_branch_search(
            env=TimedSnapshotValueBranchEnv(),
            value_fn=scalar_value,
            value_batch_fn=batch_values,
            **common_kwargs,
        )

        self.assertEqual(len(baseline_scalar_histories), 5)
        self.assertEqual(
            [(candidate.action_index, candidate.value, candidate.visits, candidate.total_value) for candidate in result.candidates],
            [(candidate.action_index, candidate.value, candidate.visits, candidate.total_value) for candidate in baseline.candidates],
        )
        self.assertEqual(result.action_index, baseline.action_index)
        self.assertEqual(result.most_visited_candidate.action_index, baseline.most_visited_candidate.action_index)
        self.assertEqual(len(batch_histories), 1)
        self.assertEqual(len(batch_histories[0]), 2)
        # The three adaptive PUCT visits depend on preceding backups and must
        # remain scalar so batching cannot change the search trajectory.
        self.assertEqual(len(scalar_histories), 3)
        self.assertEqual(result.total_visits, 5)
        self.assertEqual(result.timing.value_evaluation_count, 5)

    def test_puct_branch_search_group_batches_initial_leaves_across_worlds(self) -> None:
        trajectory = BattleTrajectory(battle_id="batch-worlds", format_id="gen3randombattle", seed=77)
        batch_histories = []
        group_scalar_histories = []

        def scalar_value(history):
            group_scalar_histories.append(history)
            return float(_only_legal_action(history[-1]))

        def batch_values(histories):
            batch_histories.append(histories)
            return tuple(float(_only_legal_action(history[-1])) for history in histories)

        common_kwargs = {
            "trajectory": trajectory,
            "player_id": "p1",
            "prefix_decision_round_count": 0,
            "legal_action_mask": (True, True, False, False, False, False, False, False, False),
            "action_priors": (0.9, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
            "cpuct": 2.0,
            "root_visit_budget": 5,
        }
        reference = tuple(
            puct_branch_search(
                env=TimedSnapshotValueBranchEnv(),
                opponent_actions=opponent_actions,
                value_fn=lambda history: float(_only_legal_action(history[-1])),
                **common_kwargs,
            )
            for opponent_actions in ({"p2": 0}, {"p2": 1})
        )

        results = puct_branch_search_group(
            env=TimedSnapshotValueBranchEnv(),
            requests=(
                PUCTBranchSearchRequest(opponent_actions={"p2": 0}),
                PUCTBranchSearchRequest(opponent_actions={"p2": 1}),
            ),
            value_fn=scalar_value,
            value_batch_fn=batch_values,
            **common_kwargs,
        )

        self.assertEqual(len(batch_histories), 1)
        self.assertEqual(len(batch_histories[0]), 4)
        # Each world's adaptive visits remain scalar because their choices
        # depend on its own preceding backups.
        self.assertEqual(len(group_scalar_histories), 6)
        self.assertEqual(
            [
                (
                    result.action_index,
                    result.most_visited_candidate.action_index,
                    [(candidate.action_index, candidate.value, candidate.visits, candidate.total_value) for candidate in result.candidates],
                )
                for result in results
            ],
            [
                (
                    result.action_index,
                    result.most_visited_candidate.action_index,
                    [(candidate.action_index, candidate.value, candidate.visits, candidate.total_value) for candidate in result.candidates],
                )
                for result in reference
            ],
        )
        self.assertEqual(
            sum(result.timing.value_evaluation_count for result in results),
            10,
        )

    def test_puct_branch_search_group_batches_adaptive_leaves_without_changing_world_results(self) -> None:
        trajectory = BattleTrajectory(battle_id="batch-adaptive-worlds", format_id="gen3randombattle", seed=78)
        batch_histories = []

        def batch_values(histories):
            batch_histories.append(histories)
            return tuple(float(_only_legal_action(history[-1])) for history in histories)

        common_kwargs = {
            "trajectory": trajectory,
            "player_id": "p1",
            "prefix_decision_round_count": 0,
            "legal_action_mask": (True, True, False, False, False, False, False, False, False),
            "action_priors": (0.9, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
            "cpuct": 2.0,
            "root_visit_budget": 5,
        }
        reference = tuple(
            puct_branch_search(
                env=TimedSnapshotValueBranchEnv(),
                opponent_actions=opponent_actions,
                value_fn=lambda history: float(_only_legal_action(history[-1])),
                **common_kwargs,
            )
            for opponent_actions in ({"p2": 0}, {"p2": 1})
        )

        results = puct_branch_search_group(
            env=TimedSnapshotValueBranchEnv(),
            requests=(
                PUCTBranchSearchRequest(opponent_actions={"p2": 0}),
                PUCTBranchSearchRequest(opponent_actions={"p2": 1}),
            ),
            value_fn=lambda _history: (_ for _ in ()).throw(AssertionError("all leaves must batch")),
            value_batch_fn=batch_values,
            batch_adaptive_values=True,
            **common_kwargs,
        )

        # Two mandatory leaves per world, then one adaptive leaf per world for
        # each of the three PUCT waves. Each root is still updated before its
        # next selection, so the scalar and batched candidates must agree.
        self.assertEqual([len(histories) for histories in batch_histories], [4, 2, 2, 2])
        self.assertEqual(
            [
                (
                    result.action_index,
                    result.most_visited_candidate.action_index,
                    [(candidate.action_index, candidate.value, candidate.visits, candidate.total_value) for candidate in result.candidates],
                )
                for result in results
            ],
            [
                (
                    result.action_index,
                    result.most_visited_candidate.action_index,
                    [(candidate.action_index, candidate.value, candidate.visits, candidate.total_value) for candidate in result.candidates],
                )
                for result in reference
            ],
        )
        self.assertEqual(sum(result.timing.value_evaluation_count for result in results), 10)
        self.assertEqual(
            [result.timing.adaptive_value_evaluation_count for result in results],
            [3, 3],
        )
        self.assertEqual(
            [result.timing.adaptive_cross_world_batched_leaf_count for result in results],
            [3, 3],
        )
        aggregate_timing = RootPUCTSearchTiming.aggregate(tuple(result.timing for result in results))
        self.assertEqual(aggregate_timing.adaptive_value_evaluation_count, 6)
        self.assertEqual(aggregate_timing.adaptive_cross_world_batched_leaf_count, 6)

    def test_puct_branch_search_group_reuses_direct_root_branches_without_changing_values(self) -> None:
        trajectory = BattleTrajectory(battle_id="cached-adaptive-worlds", format_id="gen3randombattle", seed=80)
        common_kwargs = {
            "trajectory": trajectory,
            "player_id": "p1",
            "prefix_decision_round_count": 0,
            "legal_action_mask": (True, True, False, False, False, False, False, False, False),
            "action_priors": (0.9, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
            "cpuct": 2.0,
            "root_visit_budget": 5,
            "value_fn": lambda _history: (_ for _ in ()).throw(
                AssertionError("all leaves must batch")
            ),
            "value_batch_fn": lambda histories: tuple(
                float(_only_legal_action(history[-1])) for history in histories
            ),
            "batch_adaptive_values": True,
        }

        def requests_for(env):
            return tuple(
                PUCTBranchSearchRequest(
                    opponent_actions=opponent_actions,
                    start_override=_start_override(),
                    prepared_prefix=prepare_replay_prefix(
                        env=env,
                        trajectory=trajectory,
                        player_id="p1",
                        prefix_decision_round_count=0,
                        start_override=_start_override(),
                        expected_current_observation=_observation(0),
                    ),
                )
                for opponent_actions in ({"p2": 0}, {"p2": 1})
            )

        reference_env = PlayerObservationFusedBridgeEnv()
        reference = puct_branch_search_group(
            env=reference_env,
            requests=requests_for(reference_env),
            expected_current_observation=_observation(0),
            **common_kwargs,
        )
        cached_env = PlayerObservationFusedBridgeEnv()
        cached = puct_branch_search_group(
            env=cached_env,
            requests=requests_for(cached_env),
            expected_current_observation=_observation(0),
            reuse_adaptive_root_branches=True,
            **common_kwargs,
        )

        self.assertEqual(
            [
                (
                    result.action_index,
                    result.most_visited_candidate.action_index,
                    [
                        (candidate.action_index, candidate.value, candidate.visits, candidate.total_value)
                        for candidate in result.candidates
                    ],
                )
                for result in cached
            ],
            [
                (
                    result.action_index,
                    result.most_visited_candidate.action_index,
                    [
                        (candidate.action_index, candidate.value, candidate.visits, candidate.total_value)
                        for candidate in result.candidates
                    ],
                )
                for result in reference
            ],
        )
        self.assertEqual([result.timing.value_evaluation_count for result in cached], [5, 5])
        self.assertEqual(
            [result.timing.adaptive_value_evaluation_count for result in cached],
            [3, 3],
        )
        self.assertEqual(
            [result.timing.adaptive_cross_world_batched_leaf_count for result in cached],
            [3, 3],
        )
        self.assertEqual(
            [result.timing.adaptive_reused_root_branch_count for result in cached],
            [3, 3],
        )
        self.assertEqual(reference_env.fused_search_step_calls, 10)
        self.assertEqual(cached_env.fused_search_step_calls, 4)
        self.assertEqual(cached_env.player_observation_calls, ["p1"] * 4)

    def test_puct_branch_search_group_rejects_branch_reuse_without_adaptive_batching(self) -> None:
        with self.assertRaisesRegex(ValueError, "requires batched adaptive values"):
            puct_branch_search_group(
                env=ValueBranchEnv(),
                trajectory=BattleTrajectory(
                    battle_id="invalid-branch-cache",
                    format_id="gen3randombattle",
                    seed=81,
                ),
                player_id="p1",
                prefix_decision_round_count=0,
                legal_action_mask=(True,) + (False,) * (ACTION_COUNT - 1),
                requests=(PUCTBranchSearchRequest(opponent_actions={"p2": 0}),),
                value_fn=lambda _history: 0.0,
                value_batch_fn=lambda histories: tuple(0.0 for _history in histories),
                action_priors=(1.0,) + (0.0,) * (ACTION_COUNT - 1),
                reuse_adaptive_root_branches=True,
            )

    def test_puct_branch_search_group_batches_only_nonterminal_adaptive_leaves(self) -> None:
        trajectory = BattleTrajectory(battle_id="batch-adaptive-terminals", format_id="gen3randombattle", seed=79)
        batch_histories = []

        def batch_values(histories):
            batch_histories.append(histories)
            return tuple(float(_only_legal_action(history[-1])) for history in histories)

        common_kwargs = {
            "trajectory": trajectory,
            "player_id": "p1",
            "prefix_decision_round_count": 0,
            "legal_action_mask": (True, True, False, False, False, False, False, False, False),
            "action_priors": (0.9, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
            "cpuct": 2.0,
            "root_visit_budget": 5,
        }
        requests = (
            PUCTBranchSearchRequest(opponent_actions={"p2": 0}),
            PUCTBranchSearchRequest(opponent_actions={"p2": 1}),
        )
        terminal_winners = {(0, 0): "p1", (0, 1): "p2"}
        reference = tuple(
            puct_branch_search(
                env=OpponentConditionalTerminalValueBranchEnv(terminal_winners),
                opponent_actions=request.opponent_actions,
                value_fn=lambda history: float(_only_legal_action(history[-1])),
                **common_kwargs,
            )
            for request in requests
        )

        results = puct_branch_search_group(
            env=OpponentConditionalTerminalValueBranchEnv(terminal_winners),
            requests=requests,
            value_fn=lambda _history: (_ for _ in ()).throw(AssertionError("all nonterminal leaves must batch")),
            value_batch_fn=batch_values,
            batch_adaptive_values=True,
            **common_kwargs,
        )

        # The p2=0 world backs up terminal leaves directly. Only p2=1 leaves
        # enter the initial or adaptive evaluator batches.
        self.assertEqual([len(histories) for histories in batch_histories], [2, 1, 1, 1])
        self.assertEqual(
            [
                (
                    result.action_index,
                    result.most_visited_candidate.action_index,
                    [(candidate.action_index, candidate.value, candidate.visits, candidate.total_value) for candidate in result.candidates],
                )
                for result in results
            ],
            [
                (
                    result.action_index,
                    result.most_visited_candidate.action_index,
                    [(candidate.action_index, candidate.value, candidate.visits, candidate.total_value) for candidate in result.candidates],
                )
                for result in reference
            ],
        )
        self.assertEqual([result.total_visits for result in results], [5, 5])
        self.assertEqual([result.timing.value_evaluation_count for result in results], [0, 5])
        self.assertEqual(
            [result.timing.adaptive_value_evaluation_count for result in results],
            [0, 3],
        )
        # Only one world had nonterminal leaves, so adaptive dispatch ran but
        # did not create a cross-world batch in this fixture.
        self.assertEqual(
            [result.timing.adaptive_cross_world_batched_leaf_count for result in results],
            [0, 0],
        )

    def test_puct_branch_search_group_batches_adaptive_leaves_with_per_world_budgets(self) -> None:
        trajectory = BattleTrajectory(battle_id="batch-adaptive-budgets", format_id="gen3randombattle", seed=79)
        batch_histories = []

        def batch_values(histories):
            batch_histories.append(histories)
            return tuple(float(_only_legal_action(history[-1])) for history in histories)

        common_kwargs = {
            "trajectory": trajectory,
            "player_id": "p1",
            "prefix_decision_round_count": 0,
            "legal_action_mask": (True, True, False, False, False, False, False, False, False),
            "action_priors": (0.9, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
            "cpuct": 2.0,
            "root_visit_budget": 5,
        }
        requests = (
            PUCTBranchSearchRequest(
                opponent_actions={"p2": 0},
                root_visit_budget_resolver=lambda _context: 4,
            ),
            PUCTBranchSearchRequest(
                opponent_actions={"p2": 1},
                root_visit_budget_resolver=lambda _context: 5,
            ),
        )
        reference = tuple(
            puct_branch_search(
                env=TimedSnapshotValueBranchEnv(),
                opponent_actions=request.opponent_actions,
                root_visit_budget_resolver=request.root_visit_budget_resolver,
                value_fn=lambda history: float(_only_legal_action(history[-1])),
                **common_kwargs,
            )
            for request in requests
        )

        results = puct_branch_search_group(
            env=TimedSnapshotValueBranchEnv(),
            requests=requests,
            value_fn=lambda _history: (_ for _ in ()).throw(AssertionError("all leaves must batch")),
            value_batch_fn=batch_values,
            batch_adaptive_values=True,
            **common_kwargs,
        )

        # The four-visit world retires after its second adaptive wave; the
        # five-visit world then contributes the final one-leaf batch alone.
        self.assertEqual([len(histories) for histories in batch_histories], [4, 2, 2, 1])
        self.assertEqual([result.total_visits for result in results], [4, 5])
        self.assertEqual(
            [
                [(candidate.action_index, candidate.value, candidate.visits, candidate.total_value) for candidate in result.candidates]
                for result in results
            ],
            [
                [(candidate.action_index, candidate.value, candidate.visits, candidate.total_value) for candidate in result.candidates]
                for result in reference
            ],
        )

    def test_puct_branch_search_group_rejects_leaf_rollouts(self) -> None:
        with self.assertRaisesRegex(ValueError, "zero leaf rollout"):
            puct_branch_search_group(
                env=TimedSnapshotValueBranchEnv(),
                trajectory=BattleTrajectory(battle_id="batch-worlds", format_id="gen3randombattle", seed=77),
                player_id="p1",
                prefix_decision_round_count=0,
                legal_action_mask=(True, False, False, False, False, False, False, False, False),
                requests=(PUCTBranchSearchRequest(opponent_actions={"p2": 0}),),
                value_fn=lambda _history: 0.0,
                value_batch_fn=lambda histories: tuple(0.0 for _history in histories),
                action_priors=(1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
                leaf_rollout_decision_rounds=1,
            )

    def test_puct_branch_search_rejects_mismatched_batched_value_count(self) -> None:
        with self.assertRaisesRegex(ValueError, "different number of values"):
            puct_branch_search(
                env=TimedSnapshotValueBranchEnv(),
                trajectory=BattleTrajectory(battle_id="batch", format_id="gen3randombattle", seed=77),
                player_id="p1",
                prefix_decision_round_count=0,
                legal_action_mask=(True, True, False, False, False, False, False, False, False),
                opponent_actions={"p2": 0},
                value_fn=lambda _history: 0.0,
                value_batch_fn=lambda _histories: (0.0,),
                action_priors=(0.9, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
                root_visit_budget=2,
            )

    def test_puct_branch_search_prefers_bridge_resident_snapshot_handles(self) -> None:
        env = BridgeSnapshotValueBranchEnv()
        trajectory = BattleTrajectory(battle_id="battle", format_id="gen3randombattle", seed=77)
        prepared_prefix = prepare_replay_prefix(
            env=env,
            trajectory=trajectory,
            player_id="p1",
            prefix_decision_round_count=0,
            start_override=_start_override(),
            expected_current_observation=_observation(0),
        )
        self.assertIsNotNone(prepared_prefix)
        assert prepared_prefix is not None
        self.assertEqual(prepared_prefix.snapshot_restore_mode, "bridge-handle")

        result = puct_branch_search(
            env=env,
            trajectory=trajectory,
            player_id="p1",
            prefix_decision_round_count=0,
            legal_action_mask=(True, True, False, False, False, False, False, False, False),
            opponent_actions={"p2": 0},
            value_fn=lambda history: float(_only_legal_action(history[-1])),
            action_priors=(0.9, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
            cpuct=2.0,
            root_visit_budget=5,
            start_override=_start_override(),
            expected_current_observation=_observation(0),
            prepared_prefix=prepared_prefix,
        )

        self.assertEqual(result.total_visits, 5)
        self.assertEqual(env.search_snapshot_calls, 1)
        self.assertEqual(env.search_restore_calls, 5)
        self.assertEqual(env.snapshot_calls, 0)
        self.assertEqual(env.restore_calls, 0)

    def test_puct_branch_search_reports_nested_bridge_timing_for_each_visit(self) -> None:
        env = BridgeTimedSnapshotValueBranchEnv()
        trajectory = BattleTrajectory(battle_id="battle", format_id="gen3randombattle", seed=77)
        prepared_prefix = prepare_replay_prefix(
            env=env,
            trajectory=trajectory,
            player_id="p1",
            prefix_decision_round_count=0,
            start_override=_start_override(),
            expected_current_observation=_observation(0),
        )
        self.assertIsNotNone(prepared_prefix)

        result = puct_branch_search(
            env=env,
            trajectory=trajectory,
            player_id="p1",
            prefix_decision_round_count=0,
            legal_action_mask=(True, True, False, False, False, False, False, False, False),
            opponent_actions={"p2": 0},
            value_fn=lambda history: float(_only_legal_action(history[-1])),
            action_priors=(0.9, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
            cpuct=2.0,
            root_visit_budget=5,
            start_override=_start_override(),
            expected_current_observation=_observation(0),
            prepared_prefix=prepared_prefix,
        )

        timing = result.timing.to_dict()
        # Two mandatory legal-action visits plus three PUCT revisits. Each
        # visit restores the Node-held sampled world, then submits one branch.
        self.assertEqual(timing["bridge_round_trip_count"], 10)
        self.assertEqual(timing["bridge_node_processing_count"], 10)
        self.assertAlmostEqual(timing["bridge_round_trip_seconds"], 0.150)
        self.assertAlmostEqual(timing["bridge_node_processing_seconds"], 0.090)
        self.assertAlmostEqual(timing["bridge_python_orchestration_seconds"], 0.060)
        self.assertEqual(timing["bridge_python_orchestration_count"], 10)
        # Nested diagnostics are deliberately excluded from the additive wall
        # decomposition; adding them again would double-count restore/step.
        self.assertAlmostEqual(
            timing["raw_residual_seconds"],
            timing["total_seconds"]
            - timing["branch_simulator_step_seconds"]
            - timing["state_restore_seconds"]
            - timing["root_initial_sweep_orchestration_seconds"]
            - timing["root_search_setup_seconds"]
            - timing["root_adaptive_visit_orchestration_seconds"]
            - timing["root_search_finalization_seconds"]
            - timing["branch_action_validation_seconds"]
            - timing["post_branch_history_seconds"]
            - timing["value_evaluation_seconds"],
        )

    def test_puct_branch_search_fuses_bridge_restore_and_step_per_visit(self) -> None:
        env = FusedBridgeTimedSnapshotValueBranchEnv()
        trajectory = BattleTrajectory(battle_id="battle", format_id="gen3randombattle", seed=77)
        prepared_prefix = prepare_replay_prefix(
            env=env,
            trajectory=trajectory,
            player_id="p1",
            prefix_decision_round_count=0,
            start_override=_start_override(),
            expected_current_observation=_observation(0),
        )
        self.assertIsNotNone(prepared_prefix)

        result = puct_branch_search(
            env=env,
            trajectory=trajectory,
            player_id="p1",
            prefix_decision_round_count=0,
            legal_action_mask=(True, True, False, False, False, False, False, False, False),
            opponent_actions={"p2": 0},
            value_fn=lambda history: float(_only_legal_action(history[-1])),
            action_priors=(0.9, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
            cpuct=2.0,
            root_visit_budget=5,
            start_override=_start_override(),
            expected_current_observation=_observation(0),
            prepared_prefix=prepared_prefix,
        )

        timing = result.timing.to_dict()
        self.assertEqual(env.fused_search_step_calls, 5)
        self.assertEqual(env.search_restore_calls, 0)
        self.assertEqual(timing["state_restore_count"], 0)
        self.assertEqual(timing["branch_simulator_step_count"], 5)
        self.assertEqual(timing["bridge_round_trip_count"], 5)
        self.assertEqual(timing["bridge_node_processing_count"], 5)
        self.assertAlmostEqual(timing["bridge_round_trip_seconds"], 0.100)
        self.assertAlmostEqual(timing["bridge_node_processing_seconds"], 0.060)
        self.assertAlmostEqual(timing["bridge_python_orchestration_seconds"], 0.040)
        self.assertEqual(timing["branch_local_state_restore_count"], 5)
        self.assertAlmostEqual(timing["branch_local_state_restore_seconds"], 0.015)
        self.assertEqual(timing["branch_choice_encoding_count"], 5)
        self.assertAlmostEqual(timing["branch_choice_encoding_seconds"], 0.010)
        self.assertEqual(timing["branch_bridge_round_trip_count"], 5)
        self.assertAlmostEqual(timing["branch_bridge_round_trip_seconds"], 0.100)
        self.assertEqual(timing["branch_bridge_node_processing_count"], 5)
        self.assertAlmostEqual(timing["branch_bridge_node_processing_seconds"], 0.060)
        self.assertAlmostEqual(timing["branch_bridge_python_orchestration_seconds"], 0.040)
        self.assertEqual(timing["branch_result_projection_count"], 5)
        self.assertAlmostEqual(timing["branch_result_projection_seconds"], 0.020)
        self.assertEqual(timing["branch_observation_projection_count"], 5)
        self.assertAlmostEqual(timing["branch_observation_projection_seconds"], 0.015)
        self.assertEqual(timing["branch_observation_state_normalization_count"], 5)
        self.assertAlmostEqual(timing["branch_observation_state_normalization_seconds"], 0.005)
        self.assertEqual(timing["branch_observation_incremental_sync_count"], 5)
        self.assertAlmostEqual(timing["branch_observation_incremental_sync_seconds"], 0.0005)
        self.assertEqual(timing["branch_observation_replay_snapshot_count"], 5)
        self.assertAlmostEqual(timing["branch_observation_replay_snapshot_seconds"], 0.001)
        self.assertEqual(timing["branch_observation_player_state_normalization_count"], 5)
        self.assertAlmostEqual(
            timing["branch_observation_player_state_normalization_seconds"], 0.003
        )
        self.assertEqual(timing["branch_observation_state_annotation_count"], 5)
        self.assertAlmostEqual(timing["branch_observation_state_annotation_seconds"], 0.0005)
        self.assertAlmostEqual(
            timing["branch_observation_state_normalization_raw_unattributed_seconds"], 0.0
        )
        self.assertAlmostEqual(
            timing["branch_observation_state_normalization_unattributed_seconds"], 0.0
        )
        self.assertEqual(timing["branch_observation_encoding_count"], 5)
        self.assertAlmostEqual(timing["branch_observation_encoding_seconds"], 0.0075)
        self.assertEqual(timing["branch_belief_overlay_projection_count"], 5)
        self.assertAlmostEqual(timing["branch_belief_overlay_projection_seconds"], 0.0025)
        self.assertAlmostEqual(timing["branch_observation_unattributed_seconds"], 0.0)
        self.assertAlmostEqual(timing["branch_projection_raw_unattributed_seconds"], 0.005)
        self.assertAlmostEqual(timing["branch_projection_unattributed_seconds"], 0.005)
        self.assertGreaterEqual(timing["raw_residual_seconds"], -1e-9)

    def test_puct_branch_search_uses_single_view_fast_path_without_rollout_tails(self) -> None:
        env = PlayerObservationFusedBridgeEnv(terminal_winners_by_action={0: "p1"})
        trajectory = BattleTrajectory(battle_id="battle", format_id="gen3randombattle", seed=77)
        prepared_prefix = prepare_replay_prefix(
            env=env,
            trajectory=trajectory,
            player_id="p1",
            prefix_decision_round_count=0,
            start_override=_start_override(),
            expected_current_observation=_observation(0),
        )
        self.assertIsNotNone(prepared_prefix)

        result = puct_branch_search(
            env=env,
            trajectory=trajectory,
            player_id="p1",
            prefix_decision_round_count=0,
            legal_action_mask=(True, True, False, False, False, False, False, False, False),
            opponent_actions={"p2": 0},
            value_fn=lambda history: float(_only_legal_action(history[-1])),
            action_priors=(0.9, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
            cpuct=2.0,
            root_visit_budget=5,
            start_override=_start_override(),
            expected_current_observation=_observation(0),
            prepared_prefix=prepared_prefix,
        )

        self.assertEqual(result.action_index, 0)
        self.assertEqual(env.player_observation_calls, ["p1"] * 5)

    def test_puct_branch_search_keeps_full_step_path_for_rollout_tails(self) -> None:
        env = PlayerObservationFusedBridgeEnv()
        trajectory = BattleTrajectory(battle_id="battle", format_id="gen3randombattle", seed=77)
        prepared_prefix = prepare_replay_prefix(
            env=env,
            trajectory=trajectory,
            player_id="p1",
            prefix_decision_round_count=0,
            start_override=_start_override(),
            expected_current_observation=_observation(0),
        )
        self.assertIsNotNone(prepared_prefix)

        puct_branch_search(
            env=env,
            trajectory=trajectory,
            player_id="p1",
            prefix_decision_round_count=0,
            legal_action_mask=(True, False, False, False, False, False, False, False, False),
            opponent_actions={"p2": 0},
            value_fn=lambda _history: 0.0,
            action_priors=(1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
            leaf_rollout_policies={"p1": FirstLegalPolicy(), "p2": FixedPolicy(0)},
            leaf_rollout_config=RolloutConfig(max_decision_rounds=3),
            leaf_rollout_decision_rounds=1,
            start_override=_start_override(),
            expected_current_observation=_observation(0),
            prepared_prefix=prepared_prefix,
        )
        self.assertEqual(env.player_observation_calls, [])
        self.assertEqual(env.fused_search_step_calls, 1)

    def test_prepared_bridge_snapshot_can_be_released_after_search(self) -> None:
        env = BridgeSnapshotValueBranchEnv()
        trajectory = BattleTrajectory(battle_id="battle", format_id="gen3randombattle", seed=77)
        prepared_prefix = prepare_replay_prefix(
            env=env,
            trajectory=trajectory,
            player_id="p1",
            prefix_decision_round_count=0,
            start_override=_start_override(),
            expected_current_observation=_observation(0),
        )

        self.assertIsNotNone(prepared_prefix)
        assert prepared_prefix is not None
        self.assertEqual(prepared_prefix.snapshot_restore_mode, "bridge-handle")
        self.assertTrue(release_prepared_replay_prefix(env, prepared_prefix))
        self.assertEqual(env.released_search_snapshots, [(('p1', 'p2'), None)])

    def test_puct_branch_search_uses_generic_snapshot_for_an_oracle_world(self) -> None:
        env = BridgeSnapshotValueBranchEnv()
        trajectory = BattleTrajectory(battle_id="battle", format_id="gen3randombattle", seed=77)

        result = puct_branch_search(
            env=env,
            trajectory=trajectory,
            player_id="p1",
            prefix_decision_round_count=0,
            legal_action_mask=(True, True, False, False, False, False, False, False, False),
            opponent_actions={"p2": 0},
            value_fn=lambda history: float(_only_legal_action(history[-1])),
            action_priors=(0.9, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
            cpuct=2.0,
            root_visit_budget=5,
        )

        self.assertEqual(result.total_visits, 5)
        self.assertEqual(env.snapshot_calls, 1)
        self.assertEqual(env.restore_calls, 5)
        self.assertEqual(env.search_snapshot_calls, 0)
        self.assertEqual(env.search_restore_calls, 0)

    def test_puct_branch_search_rejects_prepared_prefix_from_a_different_world(self) -> None:
        env = TimedSnapshotValueBranchEnv()
        trajectory = BattleTrajectory(battle_id="battle", format_id="gen3randombattle", seed=77)
        prepared_prefix = prepare_replay_prefix(
            env=env,
            trajectory=trajectory,
            player_id="p1",
            prefix_decision_round_count=0,
            start_override=_start_override(),
            expected_current_observation=_observation(0),
        )
        self.assertIsNotNone(prepared_prefix)

        with self.assertRaisesRegex(ValueError, "different sampled world"):
            puct_branch_search(
                env=env,
                trajectory=trajectory,
                player_id="p1",
                prefix_decision_round_count=0,
                legal_action_mask=(True, True, False, False, False, False, False, False, False),
                opponent_actions={"p2": 0},
                value_fn=lambda history: 0.0,
                action_priors=(0.9, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
                cpuct=2.0,
                root_visit_budget=2,
                start_override=BattleStartOverride(
                    player_teams={
                        "p1": "Charizard||||Tackle|||||||",
                        "p2": "Tauros||||Body Slam|||||||",
                    }
                ),
                expected_current_observation=_observation(0),
                prepared_prefix=prepared_prefix,
            )

    def test_puct_branch_search_rejects_prepared_prefix_from_a_different_public_state(self) -> None:
        env = TimedSnapshotValueBranchEnv()
        trajectory = BattleTrajectory(battle_id="battle", format_id="gen3randombattle", seed=77)
        prepared_prefix = prepare_replay_prefix(
            env=env,
            trajectory=trajectory,
            player_id="p1",
            prefix_decision_round_count=0,
            start_override=_start_override(),
            expected_current_observation=_observation(0),
        )
        self.assertIsNotNone(prepared_prefix)

        with self.assertRaisesRegex(ValueError, "different public decision state"):
            puct_branch_search(
                env=env,
                trajectory=trajectory,
                player_id="p1",
                prefix_decision_round_count=0,
                legal_action_mask=(True, True, False, False, False, False, False, False, False),
                opponent_actions={"p2": 0},
                value_fn=lambda history: 0.0,
                action_priors=(0.9, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
                cpuct=2.0,
                root_visit_budget=2,
                start_override=_start_override(),
                expected_current_observation=_observation(1),
                prepared_prefix=prepared_prefix,
            )

    def test_puct_branch_search_rejects_prepared_prefix_from_a_different_trajectory_prefix(self) -> None:
        def trajectory_with_first_action(action_index: int) -> BattleTrajectory:
            trajectory = BattleTrajectory(battle_id="battle", format_id="gen3randombattle", seed=77)
            trajectory.append(
                TrajectoryStep(
                    player_id="p1",
                    turn_index=0,
                    observation=_observation(action_index),
                    legal_action_mask=_observation(action_index).legal_action_mask,
                    action_index=action_index,
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
            return trajectory

        env = TimedSnapshotValueBranchEnv()
        prepared_prefix = prepare_replay_prefix(
            env=env,
            trajectory=trajectory_with_first_action(0),
            player_id="p1",
            prefix_decision_round_count=1,
            start_override=_start_override(),
            expected_current_observation=_observation(0),
        )
        self.assertIsNotNone(prepared_prefix)

        with self.assertRaisesRegex(ValueError, "different trajectory prefix"):
            puct_branch_search(
                env=env,
                trajectory=trajectory_with_first_action(1),
                player_id="p1",
                prefix_decision_round_count=1,
                legal_action_mask=(True, True, False, False, False, False, False, False, False),
                opponent_actions={"p2": 0},
                value_fn=lambda history: 0.0,
                action_priors=(0.9, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
                cpuct=2.0,
                root_visit_budget=2,
                start_override=_start_override(),
                expected_current_observation=_observation(0),
                prepared_prefix=prepared_prefix,
            )

    def test_value_branch_search_does_not_skip_opponent_illegal_action_errors(self) -> None:
        with self.assertRaisesRegex(ValueError, "p2: action_index 0"):
            value_branch_search(
                env=OpponentIllegalActionEnv(),
                trajectory=BattleTrajectory(battle_id="battle", format_id="gen3randombattle", seed=77),
                player_id="p1",
                prefix_decision_round_count=0,
                legal_action_mask=(True, False, False, False, False, False, False, False, False),
                opponent_actions={"p2": 0},
                value_fn=lambda history: 0.25,
            )

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
        timing = result.timing.to_dict()
        self.assertEqual(timing["prefix_replay_count"], 2)
        self.assertEqual(timing["branch_simulator_step_count"], 2)
        self.assertEqual(timing["state_snapshot_count"], 0)
        self.assertEqual(timing["state_restore_count"], 0)
        self.assertEqual(timing["opponent_scenario_planning_count"], 0)
        self.assertEqual(timing["policy_evaluation_count"], 0)
        self.assertEqual(timing["root_initial_sweep_orchestration_count"], 1)
        self.assertEqual(timing["root_search_setup_count"], 1)
        self.assertEqual(timing["root_adaptive_visit_orchestration_count"], 0)
        self.assertEqual(timing["root_search_finalization_count"], 1)
        self.assertEqual(timing["branch_action_validation_count"], 2)
        self.assertEqual(timing["post_branch_history_count"], 2)
        self.assertEqual(timing["value_evaluation_count"], 2)
        self.assertEqual(timing["policy_value_evaluation_count"], 2)
        self.assertEqual(timing["rollout_tail_count"], 0)
        self.assertEqual(timing["policy_evaluation_seconds"], 0.0)
        self.assertEqual(timing["rollout_tail_seconds"], 0.0)
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
            + timing["policy_value_evaluation_seconds"]
            + timing["rollout_tail_seconds"]
            + timing["raw_residual_seconds"],
        )

    def test_root_puct_timing_exposes_raw_residual_before_clamping(self) -> None:
        timing = RootPUCTSearchTiming(branch_simulator_step_seconds=2.0, total_seconds=1.0).to_dict()

        self.assertEqual(timing["raw_residual_seconds"], -1.0)
        self.assertEqual(timing["residual_seconds"], 0.0)

    def test_branch_step_subtiming_accepts_pre_observation_timing_callers(self) -> None:
        timing = RootPUCTSearchTiming().with_branch_step_subtiming(
            branch_local_state_restore_seconds=0.01,
            branch_local_state_restore_count=1,
            branch_choice_encoding_seconds=0.02,
            branch_choice_encoding_count=1,
            branch_bridge_round_trip_seconds=0.03,
            branch_bridge_round_trip_count=1,
            branch_bridge_node_processing_seconds=0.02,
            branch_bridge_node_processing_count=1,
            branch_result_projection_seconds=0.04,
            branch_result_projection_count=1,
        )

        self.assertEqual(timing.branch_observation_projection_count, 0)
        self.assertEqual(timing.branch_observation_projection_seconds, 0.0)
        self.assertEqual(timing.branch_projection_unattributed_seconds, 0.04)

    def test_root_puct_timing_partitions_residual_outside_branch_search_results(self) -> None:
        timing = (
            RootPUCTSearchTiming(branch_simulator_step_seconds=2.0, total_seconds=11.0)
            .with_puct_search_residual_partition(
                result_residual_seconds=4.0,
                result_count=2,
                unrecorded_call_seconds=3.0,
                call_count=3,
            )
            .to_dict()
        )

        self.assertEqual(timing["raw_residual_seconds"], 9.0)
        self.assertEqual(timing["puct_search_result_residual_seconds"], 4.0)
        self.assertEqual(timing["puct_search_result_residual_count"], 2)
        self.assertEqual(timing["puct_search_unrecorded_call_seconds"], 3.0)
        self.assertEqual(timing["puct_search_call_count"], 3)
        self.assertEqual(timing["raw_outer_policy_residual_seconds"], 2.0)
        self.assertEqual(timing["outer_policy_residual_seconds"], 2.0)

    def test_root_puct_timing_splits_completed_and_rejected_call_wall(self) -> None:
        timing = (
            RootPUCTSearchTiming(branch_simulator_step_seconds=2.0, total_seconds=11.0)
            .with_puct_search_residual_partition(
                result_residual_seconds=4.0,
                result_count=2,
                unrecorded_call_seconds=5.0,
                call_count=3,
            )
            .with_puct_search_call_outcomes(
                completed_call_seconds=7.0,
                completed_call_count=2,
                retained_completed_call_seconds=7.0,
                retained_completed_call_count=2,
                completed_result_seconds=5.0,
                completed_result_count=2,
                rejected_call_seconds=3.0,
                rejected_call_count=1,
            )
            .to_dict()
        )

        self.assertEqual(timing["puct_search_completed_call_seconds"], 7.0)
        self.assertEqual(timing["puct_search_completed_call_count"], 2)
        self.assertEqual(timing["puct_search_retained_completed_call_seconds"], 7.0)
        self.assertEqual(timing["puct_search_retained_completed_call_count"], 2)
        self.assertEqual(timing["puct_search_completed_result_seconds"], 5.0)
        self.assertEqual(timing["puct_search_completed_result_count"], 2)
        self.assertEqual(timing["puct_search_completed_call_overhead_seconds"], 2.0)
        self.assertEqual(timing["puct_search_discarded_completed_call_seconds"], 0.0)
        self.assertEqual(timing["puct_search_discarded_completed_call_count"], 0)
        self.assertEqual(timing["puct_search_rejected_call_seconds"], 3.0)
        self.assertEqual(timing["puct_search_rejected_call_count"], 1)
        self.assertEqual(timing["puct_search_unrecorded_call_seconds"], 5.0)
        self.assertEqual(timing["raw_outer_policy_residual_seconds"], 0.0)

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

    def test_puct_branch_search_reuses_initial_value_sweep_prefix_snapshot(self) -> None:
        env = TimedSnapshotValueBranchEnv()
        trajectory = BattleTrajectory(battle_id="battle", format_id="gen3randombattle", seed=77)

        result = puct_branch_search(
            env=env,
            trajectory=trajectory,
            player_id="p1",
            prefix_decision_round_count=0,
            legal_action_mask=(True, True, False, False, False, False, False, False, False),
            opponent_actions={"p2": 0},
            value_fn=lambda history: float(_only_legal_action(history[-1])),
            action_priors=(0.9, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
            cpuct=2.0,
            root_visit_budget=5,
        )

        self.assertEqual(result.total_visits, 5)
        self.assertEqual(len(env.reset_calls), 1)
        self.assertEqual(env.snapshot_calls, 1)
        self.assertEqual(env.restore_calls, 5)
        self.assertEqual(len(env.all_step_calls), 5)
        timing = result.timing.to_dict()
        self.assertEqual(timing["prefix_replay_count"], 1)
        self.assertEqual(timing["state_snapshot_count"], 1)
        self.assertEqual(timing["state_restore_count"], 5)
        self.assertEqual(timing["branch_simulator_step_count"], 5)
        self.assertGreater(timing["state_snapshot_seconds"], 0.0)
        self.assertGreater(timing["state_restore_seconds"], 0.0)
        self.assertLessEqual(timing["state_snapshot_seconds"], timing["total_seconds"])
        self.assertLessEqual(timing["state_restore_seconds"], timing["total_seconds"])
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
            + timing["policy_value_evaluation_seconds"]
            + timing["rollout_tail_seconds"]
            + timing["raw_residual_seconds"],
        )

    def test_puct_branch_search_accumulates_until_root_time_budget_expires(self) -> None:
        env = ValueBranchEnv()
        trajectory = BattleTrajectory(battle_id="battle", format_id="gen3randombattle", seed=77)

        def value_fn(history: tuple[PokeZeroObservationV0, ...]) -> float:
            return {0: 0.1, 1: 0.2}[_only_legal_action(history[-1])]

        times = iter((0.0, 0.1, 0.2, 0.6))
        with patch("pokezero.search.perf_counter", side_effect=lambda: next(times)):
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
                root_time_budget_seconds=0.5,
            )

        self.assertEqual(result.total_visits, 4)
        self.assertEqual(len(env.all_step_calls), 4)
        self.assertTrue(result.time_budget_exhausted)
        self.assertEqual(result.root_time_budget_seconds, 0.5)
        self.assertIsNone(result.root_visit_budget)

    def test_puct_branch_search_suppresses_extra_visits_when_initial_sweep_exceeds_time_budget(self) -> None:
        env = ValueBranchEnv()
        trajectory = BattleTrajectory(battle_id="battle", format_id="gen3randombattle", seed=77)

        times = iter((0.0, 1.0))
        with patch("pokezero.search.perf_counter", side_effect=lambda: next(times)):
            result = puct_branch_search(
                env=env,
                trajectory=trajectory,
                player_id="p1",
                prefix_decision_round_count=0,
                legal_action_mask=(True, True, False, False, False, False, False, False, False),
                opponent_actions={"p2": 0},
                value_fn=lambda history: 0.0,
                action_priors=(0.9, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
                cpuct=2.0,
                root_time_budget_seconds=0.5,
            )

        self.assertEqual(result.total_visits, 2)
        self.assertEqual(len(env.all_step_calls), 2)
        self.assertTrue(result.time_budget_exhausted)

    def test_puct_branch_search_visit_budget_caps_root_time_budget(self) -> None:
        env = ValueBranchEnv()
        trajectory = BattleTrajectory(battle_id="battle", format_id="gen3randombattle", seed=77)

        times = iter((0.0, 0.1))
        with patch("pokezero.search.perf_counter", side_effect=lambda: next(times)):
            result = puct_branch_search(
                env=env,
                trajectory=trajectory,
                player_id="p1",
                prefix_decision_round_count=0,
                legal_action_mask=(True, True, False, False, False, False, False, False, False),
                opponent_actions={"p2": 0},
                value_fn=lambda history: 0.0,
                action_priors=(0.9, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
                cpuct=2.0,
                root_visit_budget=3,
                root_time_budget_seconds=10.0,
            )

        self.assertEqual(result.total_visits, 3)
        self.assertEqual(len(env.all_step_calls), 3)
        self.assertFalse(result.time_budget_exhausted)

    def test_puct_branch_search_varies_leaf_rollout_rng_across_repeated_visits(self) -> None:
        env = ContinuationOutcomeEnv({0: None, 1: None})
        trajectory = BattleTrajectory(battle_id="battle", format_id="gen3randombattle", seed=77)
        recording_policy = RngRecordingPolicy()

        result = puct_branch_search(
            env=env,
            trajectory=trajectory,
            player_id="p1",
            prefix_decision_round_count=0,
            legal_action_mask=(True, True, False, False, False, False, False, False, False),
            opponent_actions={"p2": 0},
            value_fn=lambda history: 0.0,
            action_priors=(0.9, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
            cpuct=2.0,
            root_visit_budget=5,
            leaf_rollout_policies={"p1": recording_policy, "p2": FixedPolicy(0)},
            leaf_rollout_config=RolloutConfig(max_decision_rounds=2),
            leaf_rollout_decision_rounds=1,
        )

        self.assertEqual(result.total_visits, 5)
        action_zero_samples = recording_policy.samples_by_branch_action[0]
        self.assertGreater(len(action_zero_samples), 1)
        self.assertEqual(len(action_zero_samples), len(set(action_zero_samples)))

    def test_puct_branch_search_leaf_rollout_rng_is_reproducible_for_same_inputs(self) -> None:
        first = RngRecordingPolicy()
        second = RngRecordingPolicy()

        for policy in (first, second):
            puct_branch_search(
                env=ContinuationOutcomeEnv({0: None}),
                trajectory=BattleTrajectory(battle_id="battle", format_id="gen3randombattle", seed=77),
                player_id="p1",
                prefix_decision_round_count=0,
                legal_action_mask=(True, False, False, False, False, False, False, False, False),
                opponent_actions={"p2": 0},
                value_fn=lambda history: 0.0,
                action_priors=(1.0,) + (0.0,) * (ACTION_COUNT - 1),
                cpuct=0.0,
                leaf_rollout_policies={"p1": policy, "p2": FixedPolicy(0)},
                leaf_rollout_config=RolloutConfig(max_decision_rounds=2),
                leaf_rollout_decision_rounds=1,
            )

        self.assertEqual(first.samples_by_branch_action, second.samples_by_branch_action)

    def test_puct_branch_search_leaf_rollout_rng_varies_by_opponent_scenario(self) -> None:
        first = RngRecordingPolicy()
        second = RngRecordingPolicy()

        for opponent_action, policy in ((0, first), (1, second)):
            puct_branch_search(
                env=ContinuationOutcomeEnv({0: None}),
                trajectory=BattleTrajectory(battle_id="battle", format_id="gen3randombattle", seed=77),
                player_id="p1",
                prefix_decision_round_count=0,
                legal_action_mask=(True, False, False, False, False, False, False, False, False),
                opponent_actions={"p2": opponent_action},
                value_fn=lambda history: 0.0,
                action_priors=(1.0,) + (0.0,) * (ACTION_COUNT - 1),
                cpuct=0.0,
                leaf_rollout_policies={"p1": policy, "p2": FixedPolicy(0)},
                leaf_rollout_config=RolloutConfig(max_decision_rounds=2),
                leaf_rollout_decision_rounds=1,
            )

        self.assertNotEqual(first.samples_by_branch_action, second.samples_by_branch_action)

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

    def test_puct_branch_search_rejects_invalid_time_budget(self) -> None:
        with self.assertRaisesRegex(ValueError, "root_time_budget_seconds"):
            puct_branch_search(
                env=ValueBranchEnv(),
                trajectory=BattleTrajectory(battle_id="battle", format_id="gen3randombattle", seed=77),
                player_id="p1",
                prefix_decision_round_count=0,
                legal_action_mask=(True, False, False, False, False, False, False, False, False),
                opponent_actions={"p2": 0},
                value_fn=lambda history: 0.0,
                action_priors=(1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
                root_time_budget_seconds=0.0,
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

    def test_flat_branch_search_requires_current_observation_for_start_override(self) -> None:
        with self.assertRaisesRegex(ValueError, "expected_current_observation"):
            flat_branch_search(
                env=BranchOutcomeEnv({0: "p1"}),
                trajectory=BattleTrajectory(battle_id="battle", format_id="gen3randombattle", seed=77),
                player_id="p1",
                prefix_decision_round_count=0,
                legal_action_mask=(True, False, False, False, False, False, False, False, False),
                opponent_actions={"p2": 0},
                rollout_policies={"p1": RandomLegalPolicy(), "p2": RandomLegalPolicy()},
                rollout_config=RolloutConfig(max_decision_rounds=5),
                start_override=_start_override(),
            )

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
