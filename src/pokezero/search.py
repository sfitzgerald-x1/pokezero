"""Search helpers built on replay-from-root branch rollouts."""

from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import math
import re
from time import perf_counter
from time import perf_counter as _timing_perf_counter
from typing import Any, Callable, Mapping, Sequence

from .actions import ACTION_COUNT
from .env import BattleStartOverride, PlayerId, PokeZeroEnv, TerminalState
from .mcts_diagnostics import root_puct_first_observation_mismatch_path_counts
from .observation import PokeZeroObservationV0
from .policy import Policy
from .replay_branching import (
    ReplayActionRound,
    ReplayBranchResult,
    ReplayBranchRolloutResult,
    ReplayPrefixResult,
    action_rounds_from_trajectory,
    require_current_observation_match,
    replay_trajectory_branch_rollout,
    replay_trajectory_prefix,
)
from .rollout import RolloutConfig, continue_rollout_from_current_state
from .trajectory import BattleTrajectory

ObservationValueFunction = Callable[[tuple[PokeZeroObservationV0, ...]], float]
ActionPriorVector = tuple[float, ...]
RootVisitBudgetResolver = Callable[["RootPUCTVisitBudgetContext"], int | None]
StartOverrideSource = BattleStartOverride | Callable[[], BattleStartOverride] | None
START_OVERRIDE_MISSING_WORLD_MESSAGE = "start override source did not produce a sampled world."
_ILLEGAL_ACTION_FOR_REQUEST_RE = re.compile(
    r"^(?:(?P<player_id>[^:]+): )?action_index (?P<action_index>\d+) "
    r"is not legal for the current request(?: \(request_kind=[^)]+\))?\.$"
)


@dataclass(frozen=True)
class RootPUCTVisitBudgetContext:
    """Public inputs available after root's mandatory one-visit-per-action sweep.

    Adaptive callers can use policy entropy and the initial leaf-value margin
    without changing the sweep that makes every legal action searchable.
    """

    player_id: PlayerId
    prefix_decision_round_count: int
    opponent_actions: Mapping[PlayerId, int]
    configured_root_visit_budget: int | None
    action_priors: tuple[tuple[int, float], ...]
    initial_values: tuple[tuple[int, float], ...]

    @property
    def policy_entropy(self) -> float:
        action_count = len(self.action_priors)
        if action_count < 2:
            return 0.0
        entropy = -sum(prior * math.log(prior) for _action, prior in self.action_priors if prior > 0.0)
        return entropy / math.log(action_count)

    @property
    def value_margin(self) -> float | None:
        values = sorted((value for _action, value in self.initial_values), reverse=True)
        return values[0] - values[1] if len(values) >= 2 else None

    def to_dict(self) -> dict[str, object]:
        return {
            "player_id": self.player_id,
            "prefix_decision_round_count": self.prefix_decision_round_count,
            "opponent_actions": dict(self.opponent_actions),
            "configured_root_visit_budget": self.configured_root_visit_budget,
            "action_priors": {str(action): prior for action, prior in self.action_priors},
            "initial_values": {str(action): value for action, value in self.initial_values},
            "policy_entropy": self.policy_entropy,
            "value_margin": self.value_margin,
        }


@dataclass(frozen=True)
class RootPUCTSearchTiming:
    """Non-overlapping wall-clock components for one root-PUCT decision.

    ``total_seconds`` covers root-decision preparation through branch search.
    ``opponent_scenario_planning`` includes any neural policy work performed by
    an opponent-action scenario planner; recorded-prefix benchmarks have no
    such planner and report this bucket as zero.

    ``observation_encoding`` and ``neural_forward`` are intentionally
    overlapping diagnostic sub-slices of policy/value/scenario work. They are
    emitted for W2's stage profile, but excluded from residual accounting so
    they do not double-count the end-to-end decision wall time.
    """

    prefix_replay_seconds: float = 0.0
    prefix_replay_count: int = 0
    branch_simulator_step_seconds: float = 0.0
    branch_simulator_step_count: int = 0
    state_snapshot_seconds: float = 0.0
    state_snapshot_count: int = 0
    state_restore_seconds: float = 0.0
    state_restore_count: int = 0
    belief_world_materialization_seconds: float = 0.0
    belief_world_materialization_count: int = 0
    opponent_scenario_planning_seconds: float = 0.0
    opponent_scenario_planning_count: int = 0
    policy_evaluation_seconds: float = 0.0
    policy_evaluation_count: int = 0
    observation_encoding_seconds: float = 0.0
    observation_encoding_count: int = 0
    neural_forward_seconds: float = 0.0
    neural_forward_count: int = 0
    value_evaluation_seconds: float = 0.0
    value_evaluation_count: int = 0
    rollout_tail_seconds: float = 0.0
    rollout_tail_count: int = 0
    total_seconds: float = 0.0

    @property
    def policy_value_evaluation_seconds(self) -> float:
        return self.policy_evaluation_seconds + self.value_evaluation_seconds

    @property
    def policy_value_evaluation_count(self) -> int:
        return self.policy_evaluation_count + self.value_evaluation_count

    @property
    def raw_residual_seconds(self) -> float:
        accounted_components = (
            self.prefix_replay_seconds
            + self.branch_simulator_step_seconds
            + self.state_snapshot_seconds
            + self.state_restore_seconds
            + self.belief_world_materialization_seconds
            + self.opponent_scenario_planning_seconds
            + self.policy_value_evaluation_seconds
            + self.rollout_tail_seconds
        )
        return self.total_seconds - accounted_components

    @property
    def residual_seconds(self) -> float:
        return max(0.0, self.raw_residual_seconds)

    def with_opponent_scenario_planning(self, elapsed_seconds: float) -> "RootPUCTSearchTiming":
        return replace(
            self,
            opponent_scenario_planning_seconds=self.opponent_scenario_planning_seconds + elapsed_seconds,
            opponent_scenario_planning_count=self.opponent_scenario_planning_count + 1,
        )

    def with_belief_world_materialization(
        self,
        elapsed_seconds: float,
        *,
        attempt_count: int,
    ) -> "RootPUCTSearchTiming":
        """Record public belief-world sampling plus replay validation work.

        This stage runs before ``puct_branch_search`` and is therefore not
        included in the branch-level replay timings it later returns.
        """

        return replace(
            self,
            belief_world_materialization_seconds=(
                self.belief_world_materialization_seconds + elapsed_seconds
            ),
            belief_world_materialization_count=(
                self.belief_world_materialization_count + attempt_count
            ),
        )

    def with_policy_evaluation(self, elapsed_seconds: float) -> "RootPUCTSearchTiming":
        return replace(
            self,
            policy_evaluation_seconds=self.policy_evaluation_seconds + elapsed_seconds,
            policy_evaluation_count=self.policy_evaluation_count + 1,
        )

    def with_neural_subtiming(
        self,
        *,
        observation_encoding_seconds: float,
        observation_encoding_count: int,
        neural_forward_seconds: float,
        neural_forward_count: int,
    ) -> "RootPUCTSearchTiming":
        """Attach non-additive transformer sub-timings to this decision."""

        return replace(
            self,
            observation_encoding_seconds=(
                self.observation_encoding_seconds + observation_encoding_seconds
            ),
            observation_encoding_count=(
                self.observation_encoding_count + observation_encoding_count
            ),
            neural_forward_seconds=self.neural_forward_seconds + neural_forward_seconds,
            neural_forward_count=self.neural_forward_count + neural_forward_count,
        )

    def with_total(self, elapsed_seconds: float) -> "RootPUCTSearchTiming":
        return replace(self, total_seconds=elapsed_seconds)

    @classmethod
    def aggregate(cls, timings: Sequence["RootPUCTSearchTiming"]) -> "RootPUCTSearchTiming":
        return cls(
            prefix_replay_seconds=sum(timing.prefix_replay_seconds for timing in timings),
            prefix_replay_count=sum(timing.prefix_replay_count for timing in timings),
            branch_simulator_step_seconds=sum(
                timing.branch_simulator_step_seconds for timing in timings
            ),
            branch_simulator_step_count=sum(timing.branch_simulator_step_count for timing in timings),
            state_snapshot_seconds=sum(timing.state_snapshot_seconds for timing in timings),
            state_snapshot_count=sum(timing.state_snapshot_count for timing in timings),
            state_restore_seconds=sum(timing.state_restore_seconds for timing in timings),
            state_restore_count=sum(timing.state_restore_count for timing in timings),
            belief_world_materialization_seconds=sum(
                timing.belief_world_materialization_seconds for timing in timings
            ),
            belief_world_materialization_count=sum(
                timing.belief_world_materialization_count for timing in timings
            ),
            opponent_scenario_planning_seconds=sum(
                timing.opponent_scenario_planning_seconds for timing in timings
            ),
            opponent_scenario_planning_count=sum(
                timing.opponent_scenario_planning_count for timing in timings
            ),
            policy_evaluation_seconds=sum(timing.policy_evaluation_seconds for timing in timings),
            policy_evaluation_count=sum(timing.policy_evaluation_count for timing in timings),
            observation_encoding_seconds=sum(
                timing.observation_encoding_seconds for timing in timings
            ),
            observation_encoding_count=sum(
                timing.observation_encoding_count for timing in timings
            ),
            neural_forward_seconds=sum(timing.neural_forward_seconds for timing in timings),
            neural_forward_count=sum(timing.neural_forward_count for timing in timings),
            value_evaluation_seconds=sum(timing.value_evaluation_seconds for timing in timings),
            value_evaluation_count=sum(timing.value_evaluation_count for timing in timings),
            rollout_tail_seconds=sum(timing.rollout_tail_seconds for timing in timings),
            rollout_tail_count=sum(timing.rollout_tail_count for timing in timings),
            total_seconds=sum(timing.total_seconds for timing in timings),
        )

    def to_dict(self) -> dict[str, float | int]:
        return {
            "prefix_replay_seconds": self.prefix_replay_seconds,
            "prefix_replay_count": self.prefix_replay_count,
            "branch_simulator_step_seconds": self.branch_simulator_step_seconds,
            "branch_simulator_step_count": self.branch_simulator_step_count,
            "state_snapshot_seconds": self.state_snapshot_seconds,
            "state_snapshot_count": self.state_snapshot_count,
            "state_restore_seconds": self.state_restore_seconds,
            "state_restore_count": self.state_restore_count,
            "belief_world_materialization_seconds": self.belief_world_materialization_seconds,
            "belief_world_materialization_count": self.belief_world_materialization_count,
            "opponent_scenario_planning_seconds": self.opponent_scenario_planning_seconds,
            "opponent_scenario_planning_count": self.opponent_scenario_planning_count,
            "policy_evaluation_seconds": self.policy_evaluation_seconds,
            "policy_evaluation_count": self.policy_evaluation_count,
            "observation_encoding_seconds": self.observation_encoding_seconds,
            "observation_encoding_count": self.observation_encoding_count,
            "neural_forward_seconds": self.neural_forward_seconds,
            "neural_forward_count": self.neural_forward_count,
            "value_evaluation_seconds": self.value_evaluation_seconds,
            "value_evaluation_count": self.value_evaluation_count,
            "policy_value_evaluation_seconds": self.policy_value_evaluation_seconds,
            "policy_value_evaluation_count": self.policy_value_evaluation_count,
            "rollout_tail_seconds": self.rollout_tail_seconds,
            "rollout_tail_count": self.rollout_tail_count,
            "raw_residual_seconds": self.raw_residual_seconds,
            "residual_seconds": self.residual_seconds,
            "total_seconds": self.total_seconds,
        }


@dataclass
class _RootPUCTSearchTimingAccumulator:
    prefix_replay_seconds: float = 0.0
    prefix_replay_count: int = 0
    branch_simulator_step_seconds: float = 0.0
    branch_simulator_step_count: int = 0
    state_snapshot_seconds: float = 0.0
    state_snapshot_count: int = 0
    state_restore_seconds: float = 0.0
    state_restore_count: int = 0
    value_evaluation_seconds: float = 0.0
    value_evaluation_count: int = 0
    rollout_tail_seconds: float = 0.0
    rollout_tail_count: int = 0

    def add_prefix_replay(self, elapsed_seconds: float) -> None:
        self.prefix_replay_seconds += elapsed_seconds
        self.prefix_replay_count += 1

    def add_branch_simulator_step(self, elapsed_seconds: float) -> None:
        self.branch_simulator_step_seconds += elapsed_seconds
        self.branch_simulator_step_count += 1

    def add_state_snapshot(self, elapsed_seconds: float) -> None:
        self.state_snapshot_seconds += elapsed_seconds
        self.state_snapshot_count += 1

    def add_state_restore(self, elapsed_seconds: float) -> None:
        self.state_restore_seconds += elapsed_seconds
        self.state_restore_count += 1

    def add_value_evaluation(self, elapsed_seconds: float) -> None:
        self.value_evaluation_seconds += elapsed_seconds
        self.value_evaluation_count += 1

    def add_rollout_tail(self, elapsed_seconds: float) -> None:
        self.rollout_tail_seconds += elapsed_seconds
        self.rollout_tail_count += 1

    def finish(self, total_seconds: float) -> RootPUCTSearchTiming:
        return RootPUCTSearchTiming(
            prefix_replay_seconds=self.prefix_replay_seconds,
            prefix_replay_count=self.prefix_replay_count,
            branch_simulator_step_seconds=self.branch_simulator_step_seconds,
            branch_simulator_step_count=self.branch_simulator_step_count,
            state_snapshot_seconds=self.state_snapshot_seconds,
            state_snapshot_count=self.state_snapshot_count,
            state_restore_seconds=self.state_restore_seconds,
            state_restore_count=self.state_restore_count,
            value_evaluation_seconds=self.value_evaluation_seconds,
            value_evaluation_count=self.value_evaluation_count,
            rollout_tail_seconds=self.rollout_tail_seconds,
            rollout_tail_count=self.rollout_tail_count,
            total_seconds=total_seconds,
        )


@dataclass(frozen=True)
class BranchSearchCandidate:
    action_index: int
    value: float
    terminal: TerminalState
    rollout: ReplayBranchRolloutResult

    def to_dict(self) -> dict[str, object]:
        return {
            "action_index": self.action_index,
            "value": self.value,
            "terminal": {
                "winner": self.terminal.winner,
                "turn_count": self.terminal.turn_count,
                "capped": self.terminal.capped,
            },
            "continuation_decision_round_count": self.rollout.continuation.decision_round_count,
        }


@dataclass(frozen=True)
class ValueBranchSearchCandidate:
    action_index: int
    value: float
    terminal: TerminalState | None
    branch: ReplayBranchResult
    evaluated_history_length: int
    leaf_evaluation: str = "value_fn"
    leaf_rollout_decision_round_count: int = 0

    def to_dict(self) -> dict[str, object]:
        return {
            "action_index": self.action_index,
            "value": self.value,
            "terminal": None
            if self.terminal is None
            else {
                "winner": self.terminal.winner,
                "turn_count": self.terminal.turn_count,
                "capped": self.terminal.capped,
            },
            "post_branch_requested_players": list(self.branch.step_result.requested_players),
            "evaluated_history_length": self.evaluated_history_length,
            "leaf_evaluation": self.leaf_evaluation,
            "leaf_rollout_decision_round_count": self.leaf_rollout_decision_round_count,
        }


@dataclass(frozen=True)
class ValueBranchSearchResult:
    player_id: PlayerId
    prefix_decision_round_count: int
    opponent_actions: Mapping[PlayerId, int]
    candidates: tuple[ValueBranchSearchCandidate, ...]

    @property
    def best_candidate(self) -> ValueBranchSearchCandidate:
        if not self.candidates:
            raise ValueError("value branch search produced no candidates.")
        return max(self.candidates, key=lambda candidate: (candidate.value, -candidate.action_index))

    @property
    def action_index(self) -> int:
        return self.best_candidate.action_index

    def to_dict(self) -> dict[str, object]:
        best = self.best_candidate
        return {
            "player_id": self.player_id,
            "prefix_decision_round_count": self.prefix_decision_round_count,
            "opponent_actions": dict(self.opponent_actions),
            "selected_action_index": best.action_index,
            "selected_value": best.value,
            "candidates": [candidate.to_dict() for candidate in self.candidates],
        }


@dataclass(frozen=True)
class _RestorablePrefix:
    prefix: ReplayPrefixResult
    snapshot: Any


@dataclass(frozen=True)
class PreparedReplayPrefix:
    """A sampled-world branch point captured after public-state validation.

    This is intentionally created only from a caller-provided start override,
    never from a live hidden-information battle. The concrete override identity
    and validated public observation are retained so callers cannot accidentally
    reuse this snapshot with a different sampled world or decision state. The
    snapshot remains tied to the environment that materialized it and can be
    restored for every root visit without replaying the recorded prefix again.
    """

    trajectory_seed: int
    format_id: str
    player_id: PlayerId
    prefix_decision_round_count: int
    trajectory_prefix_key: tuple[tuple[int, tuple[tuple[PlayerId, int], ...]], ...]
    start_override_key: tuple[object, ...]
    expected_current_observation: PokeZeroObservationV0
    prefix: ReplayPrefixResult
    snapshot: Any
    materialization_mode: str = "replay"


@dataclass(frozen=True)
class PUCTBranchSearchCandidate:
    action_index: int
    prior: float
    value: float
    visits: int
    total_value: float
    exploration_score: float
    score: float
    value_candidate: ValueBranchSearchCandidate

    @property
    def mean_value(self) -> float:
        return self.total_value / self.visits

    def to_dict(self) -> dict[str, object]:
        return {
            "action_index": self.action_index,
            "prior": self.prior,
            "value": self.value,
            "visits": self.visits,
            "total_value": self.total_value,
            "mean_value": self.mean_value,
            "exploration_score": self.exploration_score,
            "score": self.score,
            "value_candidate": self.value_candidate.to_dict(),
        }


@dataclass(frozen=True)
class PUCTBranchSearchResult:
    player_id: PlayerId
    prefix_decision_round_count: int
    opponent_actions: Mapping[PlayerId, int]
    cpuct: float
    total_visits: int
    candidates: tuple[PUCTBranchSearchCandidate, ...]
    value_search: ValueBranchSearchResult
    root_visit_budget: int | None = None
    configured_root_visit_budget: int | None = None
    visit_budget_context: RootPUCTVisitBudgetContext | None = None
    root_time_budget_seconds: float | None = None
    time_budget_exhausted: bool = False
    timing: RootPUCTSearchTiming = RootPUCTSearchTiming()

    @property
    def best_candidate(self) -> PUCTBranchSearchCandidate:
        if not self.candidates:
            raise ValueError("PUCT branch search produced no candidates.")
        return max(self.candidates, key=lambda candidate: (candidate.score, candidate.value, -candidate.action_index))

    @property
    def most_visited_candidate(self) -> PUCTBranchSearchCandidate:
        if not self.candidates:
            raise ValueError("PUCT branch search produced no candidates.")
        return max(
            self.candidates,
            key=lambda candidate: (
                candidate.visits,
                candidate.prior,
                candidate.value,
                candidate.score,
                -candidate.action_index,
            ),
        )

    @property
    def action_index(self) -> int:
        return self.best_candidate.action_index

    def to_dict(self) -> dict[str, object]:
        best = self.best_candidate
        return {
            "player_id": self.player_id,
            "prefix_decision_round_count": self.prefix_decision_round_count,
            "opponent_actions": dict(self.opponent_actions),
            "cpuct": self.cpuct,
            "total_visits": self.total_visits,
            "root_visit_budget": self.root_visit_budget,
            "configured_root_visit_budget": self.configured_root_visit_budget,
            "visit_budget_context": (
                self.visit_budget_context.to_dict() if self.visit_budget_context is not None else None
            ),
            "root_time_budget_seconds": self.root_time_budget_seconds,
            "time_budget_exhausted": self.time_budget_exhausted,
            "timing": self.timing.to_dict(),
            "selected_action_index": best.action_index,
            "selected_value": best.value,
            "selected_score": best.score,
            "candidates": [candidate.to_dict() for candidate in self.candidates],
        }


@dataclass(frozen=True)
class FlatBranchSearchResult:
    player_id: PlayerId
    prefix_decision_round_count: int
    opponent_actions: Mapping[PlayerId, int]
    candidates: tuple[BranchSearchCandidate, ...]

    @property
    def best_candidate(self) -> BranchSearchCandidate:
        if not self.candidates:
            raise ValueError("flat branch search produced no candidates.")
        return max(self.candidates, key=lambda candidate: (candidate.value, -candidate.action_index))

    @property
    def action_index(self) -> int:
        return self.best_candidate.action_index

    def to_dict(self) -> dict[str, object]:
        best = self.best_candidate
        return {
            "player_id": self.player_id,
            "prefix_decision_round_count": self.prefix_decision_round_count,
            "opponent_actions": dict(self.opponent_actions),
            "selected_action_index": best.action_index,
            "selected_value": best.value,
            "candidates": [candidate.to_dict() for candidate in self.candidates],
        }


def value_branch_search(
    *,
    env: PokeZeroEnv,
    trajectory: BattleTrajectory,
    player_id: PlayerId,
    prefix_decision_round_count: int,
    legal_action_mask: tuple[bool, ...],
    opponent_actions: Mapping[PlayerId, int],
    value_fn: ObservationValueFunction,
    leaf_rollout_policies: Mapping[PlayerId, Policy] | None = None,
    leaf_rollout_config: RolloutConfig | None = None,
    leaf_rollout_decision_rounds: int = 0,
    start_override: StartOverrideSource = None,
    expected_current_observation: PokeZeroObservationV0 | None = None,
    replay_hp_fraction_tolerance: float = 0.0,
    prepared_prefix: PreparedReplayPrefix | None = None,
) -> ValueBranchSearchResult:
    result, _restorable_prefix = _value_branch_search_with_prefix(
        env=env,
        trajectory=trajectory,
        player_id=player_id,
        prefix_decision_round_count=prefix_decision_round_count,
        legal_action_mask=legal_action_mask,
        opponent_actions=opponent_actions,
        value_fn=value_fn,
        leaf_rollout_policies=leaf_rollout_policies,
        leaf_rollout_config=leaf_rollout_config,
        leaf_rollout_decision_rounds=leaf_rollout_decision_rounds,
        start_override=start_override,
        expected_current_observation=expected_current_observation,
        replay_hp_fraction_tolerance=replay_hp_fraction_tolerance,
        prepared_prefix=prepared_prefix,
    )
    return result


def _value_branch_search_with_prefix(
    *,
    env: PokeZeroEnv,
    trajectory: BattleTrajectory,
    player_id: PlayerId,
    prefix_decision_round_count: int,
    legal_action_mask: tuple[bool, ...],
    opponent_actions: Mapping[PlayerId, int],
    value_fn: ObservationValueFunction,
    leaf_rollout_policies: Mapping[PlayerId, Policy] | None = None,
    leaf_rollout_config: RolloutConfig | None = None,
    leaf_rollout_decision_rounds: int = 0,
    start_override: StartOverrideSource = None,
    expected_current_observation: PokeZeroObservationV0 | None = None,
    replay_hp_fraction_tolerance: float = 0.0,
    timing: _RootPUCTSearchTimingAccumulator | None = None,
    prepared_prefix: PreparedReplayPrefix | None = None,
) -> tuple[ValueBranchSearchResult, _RestorablePrefix | None]:
    """Enumerate legal root actions and score branch leaves.

    The default path is the original one-ply evaluator: branch once and score the
    immediate post-branch observation with ``value_fn``. When leaf rollouts are
    enabled, each non-terminal branch is continued through the simulator for a
    bounded number of decision rounds before scoring the terminal or truncated leaf.
    """

    if len(legal_action_mask) != ACTION_COUNT:
        raise ValueError(f"legal_action_mask must contain {ACTION_COUNT} values.")
    candidate_indices = tuple(index for index, legal in enumerate(legal_action_mask) if legal)
    if not candidate_indices:
        raise ValueError("value branch search requires at least one legal action.")
    if player_id in opponent_actions:
        raise ValueError("opponent_actions must not include the searched player.")
    if leaf_rollout_decision_rounds < 0:
        raise ValueError("leaf_rollout_decision_rounds must be non-negative.")
    if leaf_rollout_decision_rounds and leaf_rollout_policies is None:
        raise ValueError("leaf_rollout_policies are required when leaf rollouts are enabled.")
    if leaf_rollout_decision_rounds and leaf_rollout_config is None:
        raise ValueError("leaf_rollout_config is required when leaf rollouts are enabled.")
    if replay_hp_fraction_tolerance < 0.0 or not math.isfinite(replay_hp_fraction_tolerance):
        raise ValueError("replay_hp_fraction_tolerance must be a finite non-negative value.")
    _require_current_observation_for_start_override(
        start_override=start_override,
        expected_current_observation=expected_current_observation,
    )

    prefix_history = player_observation_history(
        trajectory,
        player_id=player_id,
        through_decision_round=prefix_decision_round_count,
    )
    restorable_prefix = _restorable_prefix_from_prepared(
        prepared_prefix,
        trajectory=trajectory,
        player_id=player_id,
        prefix_decision_round_count=prefix_decision_round_count,
        start_override=start_override,
        expected_current_observation=expected_current_observation,
    )
    if restorable_prefix is None:
        restorable_prefix = _restorable_prefix_snapshot(
            env=env,
            trajectory=trajectory,
            player_id=player_id,
            prefix_decision_round_count=prefix_decision_round_count,
            start_override=start_override,
            expected_current_observation=expected_current_observation,
            replay_hp_fraction_tolerance=replay_hp_fraction_tolerance,
            timing=timing,
        )
    candidates: list[ValueBranchSearchCandidate] = []
    for action_index in candidate_indices:
        branch_actions = {
            **dict(opponent_actions),
            player_id: action_index,
        }
        try:
            branch = _branch_from_replay_prefix(
                env=env,
                trajectory=trajectory,
                player_id=player_id,
                prefix_decision_round_count=prefix_decision_round_count,
                branch_actions=branch_actions,
                start_override=start_override,
                expected_current_observation=expected_current_observation,
                restorable_prefix=restorable_prefix,
                replay_hp_fraction_tolerance=replay_hp_fraction_tolerance,
                timing=timing,
            )
        except ValueError as exc:
            if _is_candidate_illegal_action_error(exc, player_id=player_id, action_index=action_index):
                continue
            raise
        candidates.append(
            _value_branch_candidate(
                env=env,
                trajectory=trajectory,
                player_id=player_id,
                prefix_decision_round_count=prefix_decision_round_count,
                prefix_history=prefix_history,
                branch=branch,
                action_index=action_index,
                value_fn=value_fn,
                leaf_rollout_policies=leaf_rollout_policies,
                leaf_rollout_config=leaf_rollout_config,
                leaf_rollout_decision_rounds=leaf_rollout_decision_rounds,
                leaf_rollout_seed=_branch_rollout_seed(
                    trajectory.seed,
                    player_id=player_id,
                    prefix_decision_round_count=prefix_decision_round_count,
                    opponent_actions=opponent_actions,
                    action_index=action_index,
                    visit_index=0,
                ),
                timing=timing,
            )
        )

    if not candidates:
        raise ValueError("value branch search found no replay-legal root actions.")

    result = ValueBranchSearchResult(
        player_id=player_id,
        prefix_decision_round_count=prefix_decision_round_count,
        opponent_actions=dict(opponent_actions),
        candidates=tuple(candidates),
    )
    return result, restorable_prefix


def puct_branch_search(
    *,
    env: PokeZeroEnv,
    trajectory: BattleTrajectory,
    player_id: PlayerId,
    prefix_decision_round_count: int,
    legal_action_mask: tuple[bool, ...],
    opponent_actions: Mapping[PlayerId, int],
    value_fn: ObservationValueFunction,
    action_priors: ActionPriorVector,
    cpuct: float = 1.25,
    leaf_rollout_policies: Mapping[PlayerId, Policy] | None = None,
    leaf_rollout_config: RolloutConfig | None = None,
    leaf_rollout_decision_rounds: int = 0,
    root_visit_budget: int | None = None,
    root_visit_budget_resolver: RootVisitBudgetResolver | None = None,
    budget_action_priors: ActionPriorVector | None = None,
    root_time_budget_seconds: float | None = None,
    start_override: StartOverrideSource = None,
    expected_current_observation: PokeZeroObservationV0 | None = None,
    replay_hp_fraction_tolerance: float = 0.0,
    prepared_prefix: PreparedReplayPrefix | None = None,
) -> PUCTBranchSearchResult:
    """Score root replay branches with PUCT-style policy-prior exploration.

    By default this preserves the original one-visit-per-legal-action behavior.
    When ``root_visit_budget`` or ``root_time_budget_seconds`` permit more than
    the initial one visit per legal root action, the search repeatedly selects
    the current highest-PUCT action, re-evaluates that branch, and backs the
    value up into root visit statistics. The mandatory initial sweep is always
    completed; the time budget controls only additional post-sweep visits.
    """

    if cpuct < 0.0 or not math.isfinite(cpuct):
        raise ValueError("cpuct must be a finite non-negative value.")
    if root_visit_budget is not None and root_visit_budget <= 0:
        raise ValueError("root_visit_budget must be positive when set.")
    if root_time_budget_seconds is not None and (
        root_time_budget_seconds <= 0.0 or not math.isfinite(root_time_budget_seconds)
    ):
        raise ValueError("root_time_budget_seconds must be a finite positive value when set.")
    if replay_hp_fraction_tolerance < 0.0 or not math.isfinite(replay_hp_fraction_tolerance):
        raise ValueError("replay_hp_fraction_tolerance must be a finite non-negative value.")
    time_budget_start = perf_counter() if root_time_budget_seconds is not None else None
    timing_started_at = _timing_perf_counter()
    timing = _RootPUCTSearchTimingAccumulator()
    value_search, restorable_prefix = _value_branch_search_with_prefix(
        env=env,
        trajectory=trajectory,
        player_id=player_id,
        prefix_decision_round_count=prefix_decision_round_count,
        legal_action_mask=legal_action_mask,
        opponent_actions=opponent_actions,
        value_fn=value_fn,
        leaf_rollout_policies=leaf_rollout_policies,
        leaf_rollout_config=leaf_rollout_config,
        leaf_rollout_decision_rounds=leaf_rollout_decision_rounds,
        start_override=start_override,
        expected_current_observation=expected_current_observation,
        replay_hp_fraction_tolerance=replay_hp_fraction_tolerance,
        timing=timing,
        prepared_prefix=prepared_prefix,
    )
    legal_action_indices = tuple(candidate.action_index for candidate in value_search.candidates)
    if (
        root_visit_budget_resolver is None
        and root_visit_budget is not None
        and root_visit_budget < len(legal_action_indices)
    ):
        raise ValueError("root_visit_budget must be at least the number of legal root actions.")
    normalized_priors = _normalized_legal_priors(
        action_priors,
        legal_action_indices=legal_action_indices,
    )
    budget_priors = (
        normalized_priors
        if budget_action_priors is None
        else _normalized_legal_priors(budget_action_priors, legal_action_indices=legal_action_indices)
    )
    visit_budget_context = (
        RootPUCTVisitBudgetContext(
            player_id=player_id,
            prefix_decision_round_count=prefix_decision_round_count,
            opponent_actions=dict(opponent_actions),
            configured_root_visit_budget=root_visit_budget,
            action_priors=tuple((action, budget_priors[action]) for action in legal_action_indices),
            initial_values=tuple((candidate.action_index, candidate.value) for candidate in value_search.candidates),
        )
        if root_visit_budget_resolver is not None
        else None
    )
    visit_budget = root_visit_budget
    if root_visit_budget_resolver is not None:
        assert visit_budget_context is not None
        visit_budget = _resolve_root_visit_budget(
            root_visit_budget_resolver(visit_budget_context),
            legal_action_count=len(legal_action_indices),
        )
    accumulators = {
        candidate.action_index: _PUCTRootAccumulator(
            value_candidate=candidate,
            prior=normalized_priors[candidate.action_index],
            visits=1,
            total_value=candidate.value,
        )
        for candidate in value_search.candidates
    }
    prefix_history = player_observation_history(
        trajectory,
        player_id=player_id,
        through_decision_round=prefix_decision_round_count,
    )
    time_budget_exhausted = False
    while True:
        current_visits = sum(accumulator.visits for accumulator in accumulators.values())
        if visit_budget is not None and current_visits >= visit_budget:
            break
        if root_time_budget_seconds is None and visit_budget is None:
            break
        if time_budget_start is not None and perf_counter() - time_budget_start >= root_time_budget_seconds:
            time_budget_exhausted = True
            break
        action_index = _select_root_accumulator(
            tuple(accumulators.values()),
            cpuct=cpuct,
        ).action_index
        branch_actions = {
            **dict(opponent_actions),
            player_id: action_index,
        }
        branch = _branch_from_replay_prefix(
            env=env,
            trajectory=trajectory,
            player_id=player_id,
            prefix_decision_round_count=prefix_decision_round_count,
            branch_actions=branch_actions,
            start_override=start_override,
            expected_current_observation=expected_current_observation,
            restorable_prefix=restorable_prefix,
            replay_hp_fraction_tolerance=replay_hp_fraction_tolerance,
            timing=timing,
        )
        value_candidate = _value_branch_candidate(
            env=env,
            trajectory=trajectory,
            player_id=player_id,
            prefix_decision_round_count=prefix_decision_round_count,
            prefix_history=prefix_history,
            branch=branch,
            action_index=action_index,
            value_fn=value_fn,
            leaf_rollout_policies=leaf_rollout_policies,
            leaf_rollout_config=leaf_rollout_config,
            leaf_rollout_decision_rounds=leaf_rollout_decision_rounds,
            leaf_rollout_seed=_branch_rollout_seed(
                trajectory.seed,
                player_id=player_id,
                prefix_decision_round_count=prefix_decision_round_count,
                opponent_actions=opponent_actions,
                action_index=action_index,
                visit_index=accumulators[action_index].visits,
            ),
            timing=timing,
        )
        accumulator = accumulators[action_index]
        accumulators[action_index] = replace(
            accumulator,
            value_candidate=value_candidate,
            visits=accumulator.visits + 1,
            total_value=accumulator.total_value + value_candidate.value,
        )
    total_visits = sum(accumulator.visits for accumulator in accumulators.values())
    sqrt_total = math.sqrt(total_visits)
    candidates = tuple(
        _puct_candidate(
            value_candidate=accumulator.value_candidate,
            prior=accumulator.prior,
            cpuct=cpuct,
            sqrt_total_visits=sqrt_total,
            visits=accumulator.visits,
            total_value=accumulator.total_value,
        )
        for accumulator in accumulators.values()
    )
    return PUCTBranchSearchResult(
        player_id=player_id,
        prefix_decision_round_count=prefix_decision_round_count,
        opponent_actions=dict(opponent_actions),
        cpuct=cpuct,
        total_visits=total_visits,
        candidates=candidates,
        value_search=value_search,
        root_visit_budget=visit_budget,
        configured_root_visit_budget=root_visit_budget,
        visit_budget_context=visit_budget_context,
        root_time_budget_seconds=root_time_budget_seconds,
        time_budget_exhausted=time_budget_exhausted,
        timing=timing.finish(_timing_perf_counter() - timing_started_at),
    )


def _resolve_root_visit_budget(
    budget: int | None,
    *,
    legal_action_count: int,
) -> int | None:
    if budget is None:
        return None
    if isinstance(budget, bool) or not isinstance(budget, int) or budget <= 0:
        raise ValueError("root_visit_budget_resolver must return a positive integer or None.")
    if budget < legal_action_count:
        raise ValueError("root_visit_budget_resolver must return at least the number of legal root actions.")
    return budget


def flat_branch_search(
    *,
    env: PokeZeroEnv,
    trajectory: BattleTrajectory,
    player_id: PlayerId,
    prefix_decision_round_count: int,
    legal_action_mask: tuple[bool, ...],
    opponent_actions: Mapping[PlayerId, int],
    rollout_policies: Mapping[PlayerId, Policy],
    rollout_config: RolloutConfig,
    start_override: StartOverrideSource = None,
    expected_current_observation: PokeZeroObservationV0 | None = None,
    replay_hp_fraction_tolerance: float = 0.0,
) -> FlatBranchSearchResult:
    """Enumerate legal root actions, roll each branch out, and score terminal outcomes."""

    if len(legal_action_mask) != ACTION_COUNT:
        raise ValueError(f"legal_action_mask must contain {ACTION_COUNT} values.")
    candidate_indices = tuple(index for index, legal in enumerate(legal_action_mask) if legal)
    if not candidate_indices:
        raise ValueError("flat branch search requires at least one legal action.")
    if player_id in opponent_actions:
        raise ValueError("opponent_actions must not include the searched player.")
    if replay_hp_fraction_tolerance < 0.0 or not math.isfinite(replay_hp_fraction_tolerance):
        raise ValueError("replay_hp_fraction_tolerance must be a finite non-negative value.")
    _require_current_observation_for_start_override(
        start_override=start_override,
        expected_current_observation=expected_current_observation,
    )

    candidates: list[BranchSearchCandidate] = []
    for action_index in candidate_indices:
        branch_actions = {
            **dict(opponent_actions),
            player_id: action_index,
        }
        rollout = replay_trajectory_branch_rollout(
            env,
            trajectory,
            prefix_decision_round_count=prefix_decision_round_count,
            branch_actions=branch_actions,
            policies=rollout_policies,
            rollout_config=rollout_config,
            battle_id=f"flat-branch-search-{player_id}-{prefix_decision_round_count}-{action_index}",
            start_override=_materialize_start_override(start_override),
            consistency_player_id=player_id,
            expected_current_observation=expected_current_observation,
            # Flat search has the same branch-point consistency contract as value/PUCT search.
            check_prefix_observations=False,
            hp_fraction_tolerance=replay_hp_fraction_tolerance,
        )
        terminal = rollout.continuation.terminal
        candidates.append(
            BranchSearchCandidate(
                action_index=action_index,
                value=terminal_value_for_player(terminal, player_id=player_id),
                terminal=terminal,
                rollout=rollout,
            )
        )

    return FlatBranchSearchResult(
        player_id=player_id,
        prefix_decision_round_count=prefix_decision_round_count,
        opponent_actions=dict(opponent_actions),
        candidates=tuple(candidates),
    )


def terminal_value_for_player(terminal: TerminalState, *, player_id: PlayerId) -> float:
    if terminal.winner == player_id:
        return 1.0
    if terminal.winner is None:
        return 0.0
    return -1.0


def _value_branch_candidate(
    *,
    env: PokeZeroEnv,
    trajectory: BattleTrajectory,
    player_id: PlayerId,
    prefix_decision_round_count: int,
    prefix_history: tuple[PokeZeroObservationV0, ...],
    branch: ReplayBranchResult,
    action_index: int,
    value_fn: ObservationValueFunction,
    leaf_rollout_policies: Mapping[PlayerId, Policy] | None,
    leaf_rollout_config: RolloutConfig | None,
    leaf_rollout_decision_rounds: int,
    leaf_rollout_seed: int,
    timing: _RootPUCTSearchTimingAccumulator | None = None,
) -> ValueBranchSearchCandidate:
    terminal = branch.step_result.terminal
    if terminal is not None:
        return ValueBranchSearchCandidate(
            action_index=action_index,
            value=terminal_value_for_player(terminal, player_id=player_id),
            terminal=terminal,
            branch=branch,
            evaluated_history_length=len(prefix_history),
            leaf_evaluation="terminal",
            leaf_rollout_decision_round_count=0,
        )

    post_branch_history = _post_branch_history(
        env=env,
        player_id=player_id,
        prefix_history=prefix_history,
        branch=branch,
    )

    if leaf_rollout_decision_rounds <= 0:
        return ValueBranchSearchCandidate(
            action_index=action_index,
            value=_timed_value_evaluation(value_fn, post_branch_history, timing=timing),
            terminal=None,
            branch=branch,
            evaluated_history_length=len(post_branch_history),
            leaf_evaluation="value_fn",
            leaf_rollout_decision_round_count=0,
        )

    if leaf_rollout_policies is None or leaf_rollout_config is None:
        raise ValueError("leaf rollout policy/config missing.")
    leaf_max_decision_rounds = min(
        leaf_rollout_config.max_decision_rounds,
        prefix_decision_round_count + 1 + leaf_rollout_decision_rounds,
    )
    if leaf_max_decision_rounds <= prefix_decision_round_count + 1:
        return ValueBranchSearchCandidate(
            action_index=action_index,
            value=_timed_value_evaluation(value_fn, post_branch_history, timing=timing),
            terminal=None,
            branch=branch,
            evaluated_history_length=len(post_branch_history),
            leaf_evaluation="value_fn",
            leaf_rollout_decision_round_count=0,
        )

    rollout_tail_started_at = _timing_perf_counter() if timing is not None else None
    continuation = continue_rollout_from_current_state(
        env=env,
        policies=leaf_rollout_policies,
        config=RolloutConfig(
            max_decision_rounds=leaf_max_decision_rounds,
            format_id=leaf_rollout_config.format_id,
        ),
        seed=leaf_rollout_seed,
        battle_id=f"value-branch-leaf-{player_id}-{prefix_decision_round_count}-{action_index}",
        starting_decision_round_index=prefix_decision_round_count + 1,
        available_observations=branch.step_result.observations,
        reset_policies=True,
    )
    if timing is not None:
        assert rollout_tail_started_at is not None
        timing.add_rollout_tail(_timing_perf_counter() - rollout_tail_started_at)
    continuation_observations = tuple(
        step.observation
        for step in continuation.trajectory.steps
        if step.player_id == player_id
    )
    evaluated_history = _rollout_leaf_history(
        post_branch_history=post_branch_history,
        continuation_observations=continuation_observations,
    )
    terminal = continuation.terminal
    if terminal.winner is not None or not terminal.capped:
        value = terminal_value_for_player(terminal, player_id=player_id)
        leaf_evaluation = "rollout_terminal"
    else:
        value = _timed_value_evaluation(value_fn, evaluated_history, timing=timing)
        leaf_evaluation = "rollout_value_fn"
    return ValueBranchSearchCandidate(
        action_index=action_index,
        value=value,
        terminal=terminal,
        branch=branch,
        evaluated_history_length=len(evaluated_history),
        leaf_evaluation=leaf_evaluation,
        leaf_rollout_decision_round_count=continuation.decision_round_count,
    )


def _materialize_start_override(start_override: StartOverrideSource) -> BattleStartOverride | None:
    if callable(start_override):
        sampled_override = start_override()
        if sampled_override is None:
            raise ValueError(START_OVERRIDE_MISSING_WORLD_MESSAGE)
        return sampled_override
    return start_override


def _prepared_start_override_key(start_override: BattleStartOverride) -> tuple[object, ...]:
    """Return an immutable identity for one concrete determinized world."""

    return (
        start_override.format_id,
        start_override.observation_format_id,
        tuple(sorted((str(player), str(team)) for player, team in start_override.player_teams.items())),
    )


def _prepared_trajectory_prefix_key(
    trajectory: BattleTrajectory,
    *,
    prefix_decision_round_count: int,
) -> tuple[tuple[int, tuple[tuple[PlayerId, int], ...]], ...]:
    """Return the replay-relevant action identity for a prepared branch point."""

    return tuple(
        (round_.turn_index, tuple(sorted(round_.actions.items())))
        for round_ in action_rounds_from_trajectory(
            trajectory,
            decision_round_count=prefix_decision_round_count,
        )
    )


def prepare_replay_prefix(
    *,
    env: PokeZeroEnv,
    trajectory: BattleTrajectory,
    player_id: PlayerId,
    prefix_decision_round_count: int,
    start_override: BattleStartOverride | None,
    expected_current_observation: PokeZeroObservationV0 | None = None,
    replay_hp_fraction_tolerance: float = 0.0,
) -> PreparedReplayPrefix | None:
    """Replay a determinized world once and retain its branch-point snapshot.

    ``start_override`` must already be a concrete sampled world. Callers use
    this before search only after the override has been generated from public
    information and the branch point has passed the existing observation check.
    Environments without snapshot support still perform the replay validation
    and return ``None`` so search retains its replay-from-root fallback.
    """

    if start_override is None:
        raise ValueError("prepared replay prefix requires a concrete sampled start override.")
    if replay_hp_fraction_tolerance < 0.0 or not math.isfinite(replay_hp_fraction_tolerance):
        raise ValueError("replay_hp_fraction_tolerance must be a finite non-negative value.")
    _require_current_observation_for_start_override(
        start_override=start_override,
        expected_current_observation=expected_current_observation,
    )
    prefix = replay_trajectory_prefix(
        env,
        trajectory,
        decision_round_count=prefix_decision_round_count,
        start_override=start_override,
        consistency_player_id=player_id,
        expected_current_observation=expected_current_observation,
        check_prefix_observations=False,
        hp_fraction_tolerance=replay_hp_fraction_tolerance,
    )
    if prefix.terminal is not None:
        raise ValueError("cannot branch from a terminal replay prefix.")
    snapshotter = getattr(env, "snapshot", None)
    restorer = getattr(env, "restore", None)
    if not callable(snapshotter) or not callable(restorer):
        return None
    return PreparedReplayPrefix(
        trajectory_seed=trajectory.seed,
        format_id=trajectory.format_id,
        player_id=player_id,
        prefix_decision_round_count=prefix_decision_round_count,
        trajectory_prefix_key=_prepared_trajectory_prefix_key(
            trajectory,
            prefix_decision_round_count=prefix_decision_round_count,
        ),
        start_override_key=_prepared_start_override_key(start_override),
        expected_current_observation=expected_current_observation,
        prefix=prefix,
        snapshot=snapshotter(),
    )


def prepare_direct_materialization_prefix(
    *,
    env: PokeZeroEnv,
    trajectory: BattleTrajectory,
    player_id: PlayerId,
    prefix_decision_round_count: int,
    start_override: BattleStartOverride | None,
    public_materialization_state: object | None,
    expected_current_observation: PokeZeroObservationV0 | None = None,
    replay_hp_fraction_tolerance: float = 0.0,
    on_unavailable: Callable[[str], None] | None = None,
    on_observation_mismatch_path: Callable[[str], None] | None = None,
) -> PreparedReplayPrefix | None:
    """Construct a sampled branch point from public state without replaying its prefix.

    Environments opt in through ``materialize_public_world``. Unsupported public effects are
    represented by ``None`` so callers retain the verified Tier 1 replay path rather than filling
    unknown simulator state with an approximation.
    """

    if start_override is None:
        _record_direct_materialization_unavailable(on_unavailable, "missing_start_override")
        return None
    if public_materialization_state is None:
        _record_direct_materialization_unavailable(on_unavailable, "missing_public_state")
        return None
    if replay_hp_fraction_tolerance < 0.0 or not math.isfinite(replay_hp_fraction_tolerance):
        raise ValueError("replay_hp_fraction_tolerance must be a finite non-negative value.")
    _require_current_observation_for_start_override(
        start_override=start_override,
        expected_current_observation=expected_current_observation,
    )
    materializer = getattr(env, "materialize_public_world", None)
    snapshotter = getattr(env, "snapshot", None)
    restorer = getattr(env, "restore", None)
    if not callable(materializer) or not callable(snapshotter) or not callable(restorer):
        _record_direct_materialization_unavailable(on_unavailable, "environment_unavailable")
        return None
    try:
        materializer(
            state=public_materialization_state,
            start_override=start_override,
            seed=trajectory.seed,
        )
        if expected_current_observation is not None:
            require_current_observation_match(
                env,
                expected=expected_current_observation,
                player_id=player_id,
                turn_index=prefix_decision_round_count,
                hp_fraction_tolerance=replay_hp_fraction_tolerance,
            )
    except (RuntimeError, ValueError) as error:
        category = _direct_materialization_rejection_category(error)
        _record_direct_materialization_unavailable(
            on_unavailable,
            category,
        )
        if category == "observation_mismatch" and on_observation_mismatch_path is not None:
            for path, count in root_puct_first_observation_mismatch_path_counts(error).items():
                for _ in range(count):
                    on_observation_mismatch_path(path)
        return None
    prefix = ReplayPrefixResult(
        replayed_round_count=prefix_decision_round_count,
        requested_players=env.requested_players(),
        terminal=env.terminal(),
    )
    if prefix.terminal is not None:
        _record_direct_materialization_unavailable(on_unavailable, "terminal_state")
        return None
    return PreparedReplayPrefix(
        trajectory_seed=trajectory.seed,
        format_id=trajectory.format_id,
        player_id=player_id,
        prefix_decision_round_count=prefix_decision_round_count,
        trajectory_prefix_key=_prepared_trajectory_prefix_key(
            trajectory,
            prefix_decision_round_count=prefix_decision_round_count,
        ),
        start_override_key=_prepared_start_override_key(start_override),
        expected_current_observation=expected_current_observation,
        prefix=prefix,
        snapshot=snapshotter(),
        materialization_mode="direct",
    )


def _record_direct_materialization_unavailable(
    callback: Callable[[str], None] | None,
    category: str,
) -> None:
    if callback is not None:
        callback(category)


def _direct_materialization_rejection_category(error: Exception) -> str:
    """Map direct-construction failures to public-safe, stable telemetry categories."""

    message = str(error).lower()
    if "does not reproduce recorded replay prefix observations" in message:
        return "observation_mismatch"
    if "spent pp for a benched acting pokemon" in message:
        return "self_benched_move_history"
    if "future sight" in message:
        return "future_sight"
    if "volatile effects" in message:
        return "volatile_effects"
    if "side condition" in message:
        return "unsupported_side_condition"
    if "cannot uniquely match" in message:
        return "ambiguous_species"
    if "positive integer turn" in message:
        return "invalid_turn"
    if "requires one active" in message:
        return "missing_active"
    if "no actionable request boundary" in message:
        return "no_actionable_boundary"
    return "materializer_error"


def _restorable_prefix_from_prepared(
    prepared_prefix: PreparedReplayPrefix | None,
    *,
    trajectory: BattleTrajectory,
    player_id: PlayerId,
    prefix_decision_round_count: int,
    start_override: StartOverrideSource,
    expected_current_observation: PokeZeroObservationV0 | None,
) -> _RestorablePrefix | None:
    if prepared_prefix is None:
        return None
    if prepared_prefix.trajectory_seed != trajectory.seed or prepared_prefix.format_id != trajectory.format_id:
        raise ValueError("prepared replay prefix belongs to a different trajectory.")
    if prepared_prefix.player_id != player_id:
        raise ValueError("prepared replay prefix belongs to a different player.")
    if prepared_prefix.prefix_decision_round_count != prefix_decision_round_count:
        raise ValueError("prepared replay prefix belongs to a different decision round.")
    if prepared_prefix.trajectory_prefix_key != _prepared_trajectory_prefix_key(
        trajectory,
        prefix_decision_round_count=prefix_decision_round_count,
    ):
        raise ValueError("prepared replay prefix belongs to a different trajectory prefix.")
    if expected_current_observation != prepared_prefix.expected_current_observation:
        raise ValueError("prepared replay prefix belongs to a different public decision state.")
    if (
        callable(start_override)
        or start_override is None
        or _prepared_start_override_key(start_override) != prepared_prefix.start_override_key
    ):
        raise ValueError("prepared replay prefix belongs to a different sampled world.")
    if prepared_prefix.prefix.terminal is not None:
        raise ValueError("cannot branch from a terminal prepared replay prefix.")
    return _RestorablePrefix(prefix=prepared_prefix.prefix, snapshot=prepared_prefix.snapshot)


def _restorable_prefix_snapshot(
    *,
    env: PokeZeroEnv,
    trajectory: BattleTrajectory,
    player_id: PlayerId,
    prefix_decision_round_count: int,
    start_override: StartOverrideSource,
    expected_current_observation: PokeZeroObservationV0 | None,
    replay_hp_fraction_tolerance: float,
    timing: _RootPUCTSearchTimingAccumulator | None = None,
) -> _RestorablePrefix | None:
    snapshotter = getattr(env, "snapshot", None)
    restorer = getattr(env, "restore", None)
    if not callable(snapshotter) or not callable(restorer):
        return None
    if callable(start_override):
        return None
    prefix_replay_started_at = _timing_perf_counter() if timing is not None else None
    prefix = replay_trajectory_prefix(
        env,
        trajectory,
        decision_round_count=prefix_decision_round_count,
        start_override=start_override,
        consistency_player_id=player_id,
        expected_current_observation=expected_current_observation,
        # Root search validates the branch point. Earlier custom-game replay observations can
        # drift while sampled hidden worlds are rejected by the current observation check.
        check_prefix_observations=False,
        hp_fraction_tolerance=replay_hp_fraction_tolerance,
    )
    if timing is not None:
        assert prefix_replay_started_at is not None
        timing.add_prefix_replay(_timing_perf_counter() - prefix_replay_started_at)
    if prefix.terminal is not None:
        raise ValueError("cannot branch from a terminal replay prefix.")
    snapshot_started_at = _timing_perf_counter() if timing is not None else None
    snapshot = snapshotter()
    if timing is not None:
        assert snapshot_started_at is not None
        timing.add_state_snapshot(_timing_perf_counter() - snapshot_started_at)
    return _RestorablePrefix(prefix=prefix, snapshot=snapshot)


def _branch_from_replay_prefix(
    *,
    env: PokeZeroEnv,
    trajectory: BattleTrajectory,
    player_id: PlayerId,
    prefix_decision_round_count: int,
    branch_actions: Mapping[PlayerId, int],
    start_override: StartOverrideSource,
    expected_current_observation: PokeZeroObservationV0 | None,
    restorable_prefix: _RestorablePrefix | None,
    replay_hp_fraction_tolerance: float,
    timing: _RootPUCTSearchTimingAccumulator | None = None,
) -> ReplayBranchResult:
    if restorable_prefix is None:
        prefix_replay_started_at = _timing_perf_counter() if timing is not None else None
        prefix = replay_trajectory_prefix(
            env,
            trajectory,
            decision_round_count=prefix_decision_round_count,
            start_override=_materialize_start_override(start_override),
            consistency_player_id=player_id,
            expected_current_observation=expected_current_observation,
            # Root search scores the current decision point. Earlier custom-game replay
            # observations can drift while sampled hidden worlds are rejected by the
            # branch-point observation check below.
            check_prefix_observations=False,
            hp_fraction_tolerance=replay_hp_fraction_tolerance,
        )
        if timing is not None:
            assert prefix_replay_started_at is not None
            timing.add_prefix_replay(_timing_perf_counter() - prefix_replay_started_at)
        if prefix.terminal is not None:
            raise ValueError("cannot branch from a terminal replay prefix.")
        branch_round = ReplayActionRound(
            turn_index=prefix_decision_round_count,
            actions=branch_actions,
        )
        _require_exact_requested_players(
            branch_actions=branch_round.actions,
            requested_players=prefix.requested_players,
            turn_index=prefix_decision_round_count,
        )
        branch_step_started_at = _timing_perf_counter() if timing is not None else None
        step_result = env.step(branch_round.actions)
        if timing is not None:
            assert branch_step_started_at is not None
            timing.add_branch_simulator_step(_timing_perf_counter() - branch_step_started_at)
        return ReplayBranchResult(
            prefix=prefix,
            branch_round=branch_round,
            step_result=step_result,
        )
    restorer = getattr(env, "restore", None)
    if not callable(restorer):
        raise ValueError("environment snapshot restore became unavailable.")
    restore_started_at = _timing_perf_counter() if timing is not None else None
    restorer(restorable_prefix.snapshot)
    if timing is not None:
        assert restore_started_at is not None
        timing.add_state_restore(_timing_perf_counter() - restore_started_at)
    branch_round = ReplayActionRound(
        turn_index=prefix_decision_round_count,
        actions=branch_actions,
    )
    _require_exact_requested_players(
        branch_actions=branch_actions,
        requested_players=restorable_prefix.prefix.requested_players,
        turn_index=prefix_decision_round_count,
    )
    branch_step_started_at = _timing_perf_counter() if timing is not None else None
    step_result = env.step(branch_round.actions)
    if timing is not None:
        assert branch_step_started_at is not None
        timing.add_branch_simulator_step(_timing_perf_counter() - branch_step_started_at)
    return ReplayBranchResult(
        prefix=restorable_prefix.prefix,
        branch_round=branch_round,
        step_result=step_result,
    )


def _require_exact_requested_players(
    *,
    branch_actions: Mapping[PlayerId, int],
    requested_players: tuple[PlayerId, ...],
    turn_index: int,
) -> None:
    requested_set = set(requested_players)
    action_players = set(branch_actions)
    if action_players == requested_set:
        return
    missing = sorted(requested_set - action_players)
    extra = sorted(action_players - requested_set)
    details: list[str] = [
        f"requested players: {_format_player_set(requested_set)}",
        f"action players: {_format_player_set(action_players)}",
    ]
    if missing:
        details.append(f"missing requested players: {', '.join(missing)}")
    if extra:
        details.append(f"unexpected players: {', '.join(extra)}")
    raise ValueError(
        f"replay actions for decision round {turn_index} "
        f"do not match environment request ({'; '.join(details)})."
    )


def _format_player_set(players: set[PlayerId]) -> str:
    if not players:
        return "none"
    return ", ".join(sorted(players))


def _require_current_observation_for_start_override(
    *,
    start_override: StartOverrideSource,
    expected_current_observation: PokeZeroObservationV0 | None,
) -> None:
    if start_override is not None and expected_current_observation is None:
        raise ValueError("expected_current_observation is required when start_override is provided.")


def _post_branch_history(
    *,
    env: PokeZeroEnv,
    player_id: PlayerId,
    prefix_history: tuple[PokeZeroObservationV0, ...],
    branch: ReplayBranchResult,
) -> tuple[PokeZeroObservationV0, ...]:
    post_branch_observation = branch.step_result.observations.get(player_id)
    if post_branch_observation is None:
        post_branch_observation = env.observe(player_id)
    return (*prefix_history, post_branch_observation)


def _rollout_leaf_history(
    *,
    post_branch_history: tuple[PokeZeroObservationV0, ...],
    continuation_observations: tuple[PokeZeroObservationV0, ...],
) -> tuple[PokeZeroObservationV0, ...]:
    if not continuation_observations:
        return post_branch_history
    if post_branch_history and continuation_observations[0] == post_branch_history[-1]:
        return (*post_branch_history[:-1], *continuation_observations)
    return (*post_branch_history, *continuation_observations)


def _branch_rollout_seed(
    seed: int,
    *,
    player_id: PlayerId,
    prefix_decision_round_count: int,
    opponent_actions: Mapping[PlayerId, int],
    action_index: int,
    visit_index: int,
) -> int:
    opponent_key = ",".join(
        f"{player}:{action}"
        for player, action in sorted(opponent_actions.items())
    )
    digest = hashlib.sha256(
        (
            f"{seed}:{player_id}:{prefix_decision_round_count}:"
            f"{opponent_key}:{action_index}:{visit_index}"
        ).encode("utf-8")
    ).digest()
    return int.from_bytes(digest[:8], "big")


def _finite_value(value: float) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError("value_fn returned a non-finite branch value.")
    return result


def _timed_value_evaluation(
    value_fn: ObservationValueFunction,
    history: tuple[PokeZeroObservationV0, ...],
    *,
    timing: _RootPUCTSearchTimingAccumulator | None,
) -> float:
    if timing is None:
        return _finite_value(value_fn(history))
    started_at = _timing_perf_counter()
    value = _finite_value(value_fn(history))
    timing.add_value_evaluation(_timing_perf_counter() - started_at)
    return value


def _is_candidate_illegal_action_error(exc: ValueError, *, player_id: PlayerId, action_index: int) -> bool:
    match = _ILLEGAL_ACTION_FOR_REQUEST_RE.fullmatch(str(exc))
    if match is None or int(match.group("action_index")) != action_index:
        return False
    reported_player_id = match.group("player_id")
    return reported_player_id is None or reported_player_id == player_id


def player_observation_history(
    trajectory: BattleTrajectory,
    *,
    player_id: PlayerId,
    through_decision_round: int,
) -> tuple[PokeZeroObservationV0, ...]:
    return tuple(
        step.observation
        for step in trajectory.steps
        if step.player_id == player_id and step.turn_index <= through_decision_round
    )


def _normalized_legal_priors(
    action_priors: ActionPriorVector,
    *,
    legal_action_indices: tuple[int, ...],
) -> tuple[float, ...]:
    if len(action_priors) != ACTION_COUNT:
        raise ValueError(f"action_priors must contain {ACTION_COUNT} values.")
    if not legal_action_indices:
        raise ValueError("legal_action_indices must not be empty.")
    cleaned = []
    for prior in action_priors:
        value = float(prior)
        if value < 0.0 or not math.isfinite(value):
            raise ValueError("action_priors must contain finite non-negative values.")
        cleaned.append(value)
    legal_sum = sum(cleaned[index] for index in legal_action_indices)
    if legal_sum <= 0.0:
        uniform = 1.0 / len(legal_action_indices)
        return tuple(uniform if index in legal_action_indices else 0.0 for index in range(ACTION_COUNT))
    return tuple(
        cleaned[index] / legal_sum if index in legal_action_indices else 0.0
        for index in range(ACTION_COUNT)
    )


def _puct_candidate(
    *,
    value_candidate: ValueBranchSearchCandidate,
    prior: float,
    cpuct: float,
    sqrt_total_visits: float,
    visits: int = 1,
    total_value: float | None = None,
) -> PUCTBranchSearchCandidate:
    if visits <= 0:
        raise ValueError("PUCT candidate visits must be positive.")
    if total_value is None:
        total_value = value_candidate.value
    mean_value = total_value / visits
    exploration_score = cpuct * prior * sqrt_total_visits / (1 + visits)
    score = mean_value + exploration_score
    return PUCTBranchSearchCandidate(
        action_index=value_candidate.action_index,
        prior=prior,
        value=mean_value,
        visits=visits,
        total_value=total_value,
        exploration_score=exploration_score,
        score=score,
        value_candidate=replace(value_candidate, value=mean_value),
    )


@dataclass(frozen=True)
class _PUCTRootAccumulator:
    value_candidate: ValueBranchSearchCandidate
    prior: float
    visits: int
    total_value: float

    @property
    def action_index(self) -> int:
        return self.value_candidate.action_index

    @property
    def mean_value(self) -> float:
        return self.total_value / self.visits


def _select_root_accumulator(
    accumulators: tuple[_PUCTRootAccumulator, ...],
    *,
    cpuct: float,
) -> _PUCTRootAccumulator:
    if not accumulators:
        raise ValueError("root PUCT search produced no accumulators.")
    total_visits = sum(accumulator.visits for accumulator in accumulators)
    sqrt_total = math.sqrt(total_visits)
    return max(
        accumulators,
        key=lambda accumulator: (
            accumulator.mean_value
            + cpuct * accumulator.prior * sqrt_total / (1 + accumulator.visits),
            accumulator.mean_value,
            -accumulator.action_index,
        ),
    )
