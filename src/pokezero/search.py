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
from .mcts_diagnostics import (
    root_puct_direct_materialization_rejection_category,
    root_puct_first_observation_mismatch_path_counts,
)
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
ObservationValueBatchFunction = Callable[
    [tuple[tuple[PokeZeroObservationV0, ...], ...]], tuple[float, ...]
]
ActionPriorVector = tuple[float, ...]
RootVisitBudgetResolver = Callable[["RootPUCTVisitBudgetContext"], int | None]
StartOverrideSource = BattleStartOverride | Callable[[], BattleStartOverride] | None
START_OVERRIDE_MISSING_WORLD_MESSAGE = "start override source did not produce a sampled world."
_BRIDGE_TIMING_SECONDS = (
    "bridge_round_trip_seconds",
    "bridge_node_processing_seconds",
)
_BRIDGE_TIMING_COUNTS = (
    "bridge_round_trip_count",
    "bridge_node_processing_count",
)
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

    ``total_seconds`` covers root-decision preparation through final action
    selection and branch-environment cleanup.
    ``opponent_scenario_planning`` includes any neural policy work performed by
    an opponent-action scenario planner; recorded-prefix benchmarks have no
    such planner and report this bucket as zero.

    ``observation_encoding``, ``neural_forward``, the ``bridge_*`` fields,
    and the ``puct_search_*`` fields are intentionally overlapping diagnostic
    sub-slices. Bridge time is already contained in snapshot/restore/step
    stages, while neural timings are contained in policy/value/scenario work.
    The PUCT fields partition residual time after the additive stages have
    been measured. They are excluded from residual accounting so they do not
    double-count decision wall time.
    """

    prefix_replay_seconds: float = 0.0
    prefix_replay_count: int = 0
    branch_simulator_step_seconds: float = 0.0
    branch_simulator_step_count: int = 0
    state_snapshot_seconds: float = 0.0
    state_snapshot_count: int = 0
    state_restore_seconds: float = 0.0
    state_restore_count: int = 0
    # The following Root-PUCT stages are additive Python orchestration. They
    # deliberately exclude replay, bridge steps, and leaf evaluation so the
    # remaining residual names work we have not yet attributed.
    root_initial_sweep_orchestration_seconds: float = 0.0
    root_initial_sweep_orchestration_count: int = 0
    root_search_setup_seconds: float = 0.0
    root_search_setup_count: int = 0
    root_adaptive_visit_orchestration_seconds: float = 0.0
    root_adaptive_visit_orchestration_count: int = 0
    root_search_finalization_seconds: float = 0.0
    root_search_finalization_count: int = 0
    branch_action_validation_seconds: float = 0.0
    branch_action_validation_count: int = 0
    post_branch_history_seconds: float = 0.0
    post_branch_history_count: int = 0
    bridge_round_trip_seconds: float = 0.0
    bridge_round_trip_count: int = 0
    bridge_node_processing_seconds: float = 0.0
    bridge_node_processing_count: int = 0
    belief_world_materialization_seconds: float = 0.0
    belief_world_materialization_count: int = 0
    opponent_scenario_planning_seconds: float = 0.0
    opponent_scenario_planning_count: int = 0
    root_policy_setup_seconds: float = 0.0
    root_policy_setup_count: int = 0
    direct_prefix_construction_seconds: float = 0.0
    direct_prefix_construction_count: int = 0
    scenario_dispatch_orchestration_seconds: float = 0.0
    scenario_dispatch_orchestration_count: int = 0
    policy_evaluation_seconds: float = 0.0
    policy_evaluation_count: int = 0
    observation_encoding_seconds: float = 0.0
    observation_encoding_count: int = 0
    neural_forward_seconds: float = 0.0
    neural_forward_count: int = 0
    action_prior_neural_forward_seconds: float = 0.0
    action_prior_neural_forward_count: int = 0
    opponent_action_prior_neural_forward_seconds: float = 0.0
    opponent_action_prior_neural_forward_count: int = 0
    policy_neural_forward_seconds: float = 0.0
    policy_neural_forward_count: int = 0
    value_neural_forward_seconds: float = 0.0
    value_neural_forward_count: int = 0
    value_evaluation_seconds: float = 0.0
    value_evaluation_count: int = 0
    rollout_tail_seconds: float = 0.0
    rollout_tail_count: int = 0
    # These diagnostic fields partition residual wall time without changing the
    # additive accounting above. A policy wrapper attaches them after it has
    # measured its puct_branch_search calls.
    puct_search_result_residual_seconds: float = 0.0
    puct_search_result_residual_count: int = 0
    # The policy wrapper records every returned puct_branch_search call, then
    # distinguishes results retained in the capped scenario aggregate from
    # completed calls discarded by that cap. These remain diagnostic
    # subdivisions of unrecorded call wall time.
    puct_search_completed_call_seconds: float = 0.0
    puct_search_completed_call_count: int = 0
    puct_search_retained_completed_call_seconds: float = 0.0
    puct_search_retained_completed_call_count: int = 0
    puct_search_completed_result_seconds: float = 0.0
    puct_search_completed_result_count: int = 0
    puct_search_discarded_completed_call_seconds: float = 0.0
    puct_search_discarded_completed_call_count: int = 0
    puct_search_rejected_call_seconds: float = 0.0
    puct_search_rejected_call_count: int = 0
    puct_search_unrecorded_call_seconds: float = 0.0
    puct_search_call_count: int = 0
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
            + self.root_initial_sweep_orchestration_seconds
            + self.root_search_setup_seconds
            + self.root_adaptive_visit_orchestration_seconds
            + self.root_search_finalization_seconds
            + self.branch_action_validation_seconds
            + self.post_branch_history_seconds
            + self.belief_world_materialization_seconds
            + self.opponent_scenario_planning_seconds
            + self.root_policy_setup_seconds
            + self.direct_prefix_construction_seconds
            + self.scenario_dispatch_orchestration_seconds
            + self.policy_value_evaluation_seconds
            + self.rollout_tail_seconds
        )
        return self.total_seconds - accounted_components

    @property
    def residual_seconds(self) -> float:
        return max(0.0, self.raw_residual_seconds)

    @property
    def raw_outer_policy_residual_seconds(self) -> float:
        """Return residual wall time outside recorded branch-search results.

        ``puct_search_unrecorded_call_seconds`` includes the portion of a
        puct_branch_search invocation that is absent from the result timing,
        such as a rejected call or Python call-boundary work. The remainder is
        policy-level orchestration not otherwise attributed by this object.
        """

        return (
            self.raw_residual_seconds
            - self.puct_search_result_residual_seconds
            - self.puct_search_unrecorded_call_seconds
        )

    @property
    def outer_policy_residual_seconds(self) -> float:
        return max(0.0, self.raw_outer_policy_residual_seconds)

    @property
    def puct_search_completed_call_overhead_seconds(self) -> float:
        """Return retained-call wall absent from retained result timing."""

        return max(
            0.0,
            self.puct_search_retained_completed_call_seconds
            - self.puct_search_completed_result_seconds,
        )

    @property
    def bridge_python_orchestration_seconds(self) -> float:
        """Bridge wall not spent in Node simulator work.

        This is a diagnostic subdivision of the already-accounted bridge
        round-trip time: IPC, JSON work, queue routing, and Python-side bridge
        processing. It must not be added to ``total_seconds`` again.
        """

        return max(0.0, self.bridge_round_trip_seconds - self.bridge_node_processing_seconds)

    @property
    def bridge_python_orchestration_count(self) -> int:
        return self.bridge_round_trip_count

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

    def with_root_policy_setup(self, elapsed_seconds: float) -> "RootPUCTSearchTiming":
        return replace(
            self,
            root_policy_setup_seconds=self.root_policy_setup_seconds + elapsed_seconds,
            root_policy_setup_count=self.root_policy_setup_count + 1,
        )

    def with_direct_prefix_construction(
        self,
        elapsed_seconds: float,
        *,
        attempt_count: int,
    ) -> "RootPUCTSearchTiming":
        return replace(
            self,
            direct_prefix_construction_seconds=(
                self.direct_prefix_construction_seconds + elapsed_seconds
            ),
            direct_prefix_construction_count=(
                self.direct_prefix_construction_count + attempt_count
            ),
        )

    def with_scenario_dispatch_orchestration(
        self,
        elapsed_seconds: float,
        *,
        attempt_count: int,
    ) -> "RootPUCTSearchTiming":
        return replace(
            self,
            scenario_dispatch_orchestration_seconds=(
                self.scenario_dispatch_orchestration_seconds + elapsed_seconds
            ),
            scenario_dispatch_orchestration_count=(
                self.scenario_dispatch_orchestration_count + attempt_count
            ),
        )

    def with_puct_search_residual_partition(
        self,
        *,
        result_residual_seconds: float,
        result_count: int,
        unrecorded_call_seconds: float,
        call_count: int,
    ) -> "RootPUCTSearchTiming":
        """Attach a non-additive partition of residual branch-search wall time."""

        return replace(
            self,
            puct_search_result_residual_seconds=(
                self.puct_search_result_residual_seconds + result_residual_seconds
            ),
            puct_search_result_residual_count=(
                self.puct_search_result_residual_count + result_count
            ),
            puct_search_unrecorded_call_seconds=(
                self.puct_search_unrecorded_call_seconds + unrecorded_call_seconds
            ),
            puct_search_call_count=self.puct_search_call_count + call_count,
        )

    def with_puct_search_call_outcomes(
        self,
        *,
        completed_call_seconds: float,
        completed_call_count: int,
        retained_completed_call_seconds: float,
        retained_completed_call_count: int,
        completed_result_seconds: float,
        completed_result_count: int,
        rejected_call_seconds: float,
        rejected_call_count: int,
    ) -> "RootPUCTSearchTiming":
        """Split opaque Root-PUCT call wall by returned versus rejected calls.

        A rejected call cannot return a ``RootPUCTSearchTiming``. A completed
        call can also be discarded by the opponent-action cap, so retain that
        full wall separately from a kept result's wrapper overhead. The
        existing unrecorded-call value remains the legacy total partition.
        """

        discarded_completed_call_seconds = max(
            0.0,
            completed_call_seconds - retained_completed_call_seconds,
        )
        discarded_completed_call_count = max(0, completed_call_count - retained_completed_call_count)
        return replace(
            self,
            puct_search_completed_call_seconds=(
                self.puct_search_completed_call_seconds + completed_call_seconds
            ),
            puct_search_completed_call_count=(
                self.puct_search_completed_call_count + completed_call_count
            ),
            puct_search_retained_completed_call_seconds=(
                self.puct_search_retained_completed_call_seconds + retained_completed_call_seconds
            ),
            puct_search_retained_completed_call_count=(
                self.puct_search_retained_completed_call_count + retained_completed_call_count
            ),
            puct_search_completed_result_seconds=(
                self.puct_search_completed_result_seconds + completed_result_seconds
            ),
            puct_search_completed_result_count=(
                self.puct_search_completed_result_count + completed_result_count
            ),
            puct_search_discarded_completed_call_seconds=(
                self.puct_search_discarded_completed_call_seconds + discarded_completed_call_seconds
            ),
            puct_search_discarded_completed_call_count=(
                self.puct_search_discarded_completed_call_count + discarded_completed_call_count
            ),
            puct_search_rejected_call_seconds=(
                self.puct_search_rejected_call_seconds + rejected_call_seconds
            ),
            puct_search_rejected_call_count=(
                self.puct_search_rejected_call_count + rejected_call_count
            ),
        )

    def with_neural_subtiming(
        self,
        *,
        observation_encoding_seconds: float,
        observation_encoding_count: int,
        neural_forward_seconds: float,
        neural_forward_count: int,
        action_prior_neural_forward_seconds: float = 0.0,
        action_prior_neural_forward_count: int = 0,
        opponent_action_prior_neural_forward_seconds: float = 0.0,
        opponent_action_prior_neural_forward_count: int = 0,
        policy_neural_forward_seconds: float = 0.0,
        policy_neural_forward_count: int = 0,
        value_neural_forward_seconds: float = 0.0,
        value_neural_forward_count: int = 0,
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
            action_prior_neural_forward_seconds=(
                self.action_prior_neural_forward_seconds + action_prior_neural_forward_seconds
            ),
            action_prior_neural_forward_count=(
                self.action_prior_neural_forward_count + action_prior_neural_forward_count
            ),
            opponent_action_prior_neural_forward_seconds=(
                self.opponent_action_prior_neural_forward_seconds
                + opponent_action_prior_neural_forward_seconds
            ),
            opponent_action_prior_neural_forward_count=(
                self.opponent_action_prior_neural_forward_count
                + opponent_action_prior_neural_forward_count
            ),
            policy_neural_forward_seconds=(
                self.policy_neural_forward_seconds + policy_neural_forward_seconds
            ),
            policy_neural_forward_count=(
                self.policy_neural_forward_count + policy_neural_forward_count
            ),
            value_neural_forward_seconds=(
                self.value_neural_forward_seconds + value_neural_forward_seconds
            ),
            value_neural_forward_count=(
                self.value_neural_forward_count + value_neural_forward_count
            ),
        )

    def with_bridge_subtiming(
        self,
        *,
        bridge_round_trip_seconds: float,
        bridge_round_trip_count: int,
        bridge_node_processing_seconds: float,
        bridge_node_processing_count: int,
    ) -> "RootPUCTSearchTiming":
        """Attach non-additive bridge transport and simulator timing deltas."""

        return replace(
            self,
            bridge_round_trip_seconds=(
                self.bridge_round_trip_seconds + bridge_round_trip_seconds
            ),
            bridge_round_trip_count=self.bridge_round_trip_count + bridge_round_trip_count,
            bridge_node_processing_seconds=(
                self.bridge_node_processing_seconds + bridge_node_processing_seconds
            ),
            bridge_node_processing_count=(
                self.bridge_node_processing_count + bridge_node_processing_count
            ),
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
            root_initial_sweep_orchestration_seconds=sum(
                timing.root_initial_sweep_orchestration_seconds for timing in timings
            ),
            root_initial_sweep_orchestration_count=sum(
                timing.root_initial_sweep_orchestration_count for timing in timings
            ),
            root_search_setup_seconds=sum(timing.root_search_setup_seconds for timing in timings),
            root_search_setup_count=sum(timing.root_search_setup_count for timing in timings),
            root_adaptive_visit_orchestration_seconds=sum(
                timing.root_adaptive_visit_orchestration_seconds for timing in timings
            ),
            root_adaptive_visit_orchestration_count=sum(
                timing.root_adaptive_visit_orchestration_count for timing in timings
            ),
            root_search_finalization_seconds=sum(
                timing.root_search_finalization_seconds for timing in timings
            ),
            root_search_finalization_count=sum(
                timing.root_search_finalization_count for timing in timings
            ),
            branch_action_validation_seconds=sum(
                timing.branch_action_validation_seconds for timing in timings
            ),
            branch_action_validation_count=sum(
                timing.branch_action_validation_count for timing in timings
            ),
            post_branch_history_seconds=sum(
                timing.post_branch_history_seconds for timing in timings
            ),
            post_branch_history_count=sum(
                timing.post_branch_history_count for timing in timings
            ),
            bridge_round_trip_seconds=sum(timing.bridge_round_trip_seconds for timing in timings),
            bridge_round_trip_count=sum(timing.bridge_round_trip_count for timing in timings),
            bridge_node_processing_seconds=sum(
                timing.bridge_node_processing_seconds for timing in timings
            ),
            bridge_node_processing_count=sum(
                timing.bridge_node_processing_count for timing in timings
            ),
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
            root_policy_setup_seconds=sum(timing.root_policy_setup_seconds for timing in timings),
            root_policy_setup_count=sum(timing.root_policy_setup_count for timing in timings),
            direct_prefix_construction_seconds=sum(
                timing.direct_prefix_construction_seconds for timing in timings
            ),
            direct_prefix_construction_count=sum(
                timing.direct_prefix_construction_count for timing in timings
            ),
            scenario_dispatch_orchestration_seconds=sum(
                timing.scenario_dispatch_orchestration_seconds for timing in timings
            ),
            scenario_dispatch_orchestration_count=sum(
                timing.scenario_dispatch_orchestration_count for timing in timings
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
            action_prior_neural_forward_seconds=sum(
                timing.action_prior_neural_forward_seconds for timing in timings
            ),
            action_prior_neural_forward_count=sum(
                timing.action_prior_neural_forward_count for timing in timings
            ),
            opponent_action_prior_neural_forward_seconds=sum(
                timing.opponent_action_prior_neural_forward_seconds for timing in timings
            ),
            opponent_action_prior_neural_forward_count=sum(
                timing.opponent_action_prior_neural_forward_count for timing in timings
            ),
            policy_neural_forward_seconds=sum(
                timing.policy_neural_forward_seconds for timing in timings
            ),
            policy_neural_forward_count=sum(
                timing.policy_neural_forward_count for timing in timings
            ),
            value_neural_forward_seconds=sum(
                timing.value_neural_forward_seconds for timing in timings
            ),
            value_neural_forward_count=sum(
                timing.value_neural_forward_count for timing in timings
            ),
            value_evaluation_seconds=sum(timing.value_evaluation_seconds for timing in timings),
            value_evaluation_count=sum(timing.value_evaluation_count for timing in timings),
            rollout_tail_seconds=sum(timing.rollout_tail_seconds for timing in timings),
            rollout_tail_count=sum(timing.rollout_tail_count for timing in timings),
            puct_search_result_residual_seconds=sum(
                timing.puct_search_result_residual_seconds for timing in timings
            ),
            puct_search_result_residual_count=sum(
                timing.puct_search_result_residual_count for timing in timings
            ),
            puct_search_completed_call_seconds=sum(
                timing.puct_search_completed_call_seconds for timing in timings
            ),
            puct_search_completed_call_count=sum(
                timing.puct_search_completed_call_count for timing in timings
            ),
            puct_search_retained_completed_call_seconds=sum(
                timing.puct_search_retained_completed_call_seconds for timing in timings
            ),
            puct_search_retained_completed_call_count=sum(
                timing.puct_search_retained_completed_call_count for timing in timings
            ),
            puct_search_completed_result_seconds=sum(
                timing.puct_search_completed_result_seconds for timing in timings
            ),
            puct_search_completed_result_count=sum(
                timing.puct_search_completed_result_count for timing in timings
            ),
            puct_search_discarded_completed_call_seconds=sum(
                timing.puct_search_discarded_completed_call_seconds for timing in timings
            ),
            puct_search_discarded_completed_call_count=sum(
                timing.puct_search_discarded_completed_call_count for timing in timings
            ),
            puct_search_rejected_call_seconds=sum(
                timing.puct_search_rejected_call_seconds for timing in timings
            ),
            puct_search_rejected_call_count=sum(
                timing.puct_search_rejected_call_count for timing in timings
            ),
            puct_search_unrecorded_call_seconds=sum(
                timing.puct_search_unrecorded_call_seconds for timing in timings
            ),
            puct_search_call_count=sum(timing.puct_search_call_count for timing in timings),
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
            "root_initial_sweep_orchestration_seconds": self.root_initial_sweep_orchestration_seconds,
            "root_initial_sweep_orchestration_count": self.root_initial_sweep_orchestration_count,
            "root_search_setup_seconds": self.root_search_setup_seconds,
            "root_search_setup_count": self.root_search_setup_count,
            "root_adaptive_visit_orchestration_seconds": self.root_adaptive_visit_orchestration_seconds,
            "root_adaptive_visit_orchestration_count": self.root_adaptive_visit_orchestration_count,
            "root_search_finalization_seconds": self.root_search_finalization_seconds,
            "root_search_finalization_count": self.root_search_finalization_count,
            "branch_action_validation_seconds": self.branch_action_validation_seconds,
            "branch_action_validation_count": self.branch_action_validation_count,
            "post_branch_history_seconds": self.post_branch_history_seconds,
            "post_branch_history_count": self.post_branch_history_count,
            "bridge_round_trip_seconds": self.bridge_round_trip_seconds,
            "bridge_round_trip_count": self.bridge_round_trip_count,
            "bridge_node_processing_seconds": self.bridge_node_processing_seconds,
            "bridge_node_processing_count": self.bridge_node_processing_count,
            "bridge_python_orchestration_seconds": self.bridge_python_orchestration_seconds,
            "bridge_python_orchestration_count": self.bridge_python_orchestration_count,
            "belief_world_materialization_seconds": self.belief_world_materialization_seconds,
            "belief_world_materialization_count": self.belief_world_materialization_count,
            "opponent_scenario_planning_seconds": self.opponent_scenario_planning_seconds,
            "opponent_scenario_planning_count": self.opponent_scenario_planning_count,
            "root_policy_setup_seconds": self.root_policy_setup_seconds,
            "root_policy_setup_count": self.root_policy_setup_count,
            "direct_prefix_construction_seconds": self.direct_prefix_construction_seconds,
            "direct_prefix_construction_count": self.direct_prefix_construction_count,
            "scenario_dispatch_orchestration_seconds": self.scenario_dispatch_orchestration_seconds,
            "scenario_dispatch_orchestration_count": self.scenario_dispatch_orchestration_count,
            "policy_evaluation_seconds": self.policy_evaluation_seconds,
            "policy_evaluation_count": self.policy_evaluation_count,
            "observation_encoding_seconds": self.observation_encoding_seconds,
            "observation_encoding_count": self.observation_encoding_count,
            "neural_forward_seconds": self.neural_forward_seconds,
            "neural_forward_count": self.neural_forward_count,
            "action_prior_neural_forward_seconds": self.action_prior_neural_forward_seconds,
            "action_prior_neural_forward_count": self.action_prior_neural_forward_count,
            "opponent_action_prior_neural_forward_seconds": (
                self.opponent_action_prior_neural_forward_seconds
            ),
            "opponent_action_prior_neural_forward_count": (
                self.opponent_action_prior_neural_forward_count
            ),
            "policy_neural_forward_seconds": self.policy_neural_forward_seconds,
            "policy_neural_forward_count": self.policy_neural_forward_count,
            "value_neural_forward_seconds": self.value_neural_forward_seconds,
            "value_neural_forward_count": self.value_neural_forward_count,
            "value_evaluation_seconds": self.value_evaluation_seconds,
            "value_evaluation_count": self.value_evaluation_count,
            "policy_value_evaluation_seconds": self.policy_value_evaluation_seconds,
            "policy_value_evaluation_count": self.policy_value_evaluation_count,
            "rollout_tail_seconds": self.rollout_tail_seconds,
            "rollout_tail_count": self.rollout_tail_count,
            "puct_search_result_residual_seconds": self.puct_search_result_residual_seconds,
            "puct_search_result_residual_count": self.puct_search_result_residual_count,
            "puct_search_completed_call_seconds": self.puct_search_completed_call_seconds,
            "puct_search_completed_call_count": self.puct_search_completed_call_count,
            "puct_search_retained_completed_call_seconds": (
                self.puct_search_retained_completed_call_seconds
            ),
            "puct_search_retained_completed_call_count": (
                self.puct_search_retained_completed_call_count
            ),
            "puct_search_completed_result_seconds": self.puct_search_completed_result_seconds,
            "puct_search_completed_result_count": self.puct_search_completed_result_count,
            "puct_search_completed_call_overhead_seconds": (
                self.puct_search_completed_call_overhead_seconds
            ),
            "puct_search_discarded_completed_call_seconds": (
                self.puct_search_discarded_completed_call_seconds
            ),
            "puct_search_discarded_completed_call_count": (
                self.puct_search_discarded_completed_call_count
            ),
            "puct_search_rejected_call_seconds": self.puct_search_rejected_call_seconds,
            "puct_search_rejected_call_count": self.puct_search_rejected_call_count,
            "puct_search_unrecorded_call_seconds": self.puct_search_unrecorded_call_seconds,
            "puct_search_call_count": self.puct_search_call_count,
            "raw_residual_seconds": self.raw_residual_seconds,
            "residual_seconds": self.residual_seconds,
            "raw_outer_policy_residual_seconds": self.raw_outer_policy_residual_seconds,
            "outer_policy_residual_seconds": self.outer_policy_residual_seconds,
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
    root_initial_sweep_orchestration_seconds: float = 0.0
    root_initial_sweep_orchestration_count: int = 0
    root_search_setup_seconds: float = 0.0
    root_search_setup_count: int = 0
    root_adaptive_visit_orchestration_seconds: float = 0.0
    root_adaptive_visit_orchestration_count: int = 0
    root_search_finalization_seconds: float = 0.0
    root_search_finalization_count: int = 0
    branch_action_validation_seconds: float = 0.0
    branch_action_validation_count: int = 0
    post_branch_history_seconds: float = 0.0
    post_branch_history_count: int = 0
    bridge_round_trip_seconds: float = 0.0
    bridge_round_trip_count: int = 0
    bridge_node_processing_seconds: float = 0.0
    bridge_node_processing_count: int = 0
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

    @property
    def branch_evaluation_seconds(self) -> float:
        """Return additive branch work used to isolate orchestration slices."""

        return (
            self.prefix_replay_seconds
            + self.branch_simulator_step_seconds
            + self.state_snapshot_seconds
            + self.state_restore_seconds
            + self.branch_action_validation_seconds
            + self.post_branch_history_seconds
            + self.value_evaluation_seconds
            + self.rollout_tail_seconds
        )

    def add_root_initial_sweep_orchestration(self, elapsed_seconds: float) -> None:
        self.root_initial_sweep_orchestration_seconds += elapsed_seconds
        self.root_initial_sweep_orchestration_count += 1

    def add_root_search_setup(self, elapsed_seconds: float) -> None:
        self.root_search_setup_seconds += elapsed_seconds
        self.root_search_setup_count += 1

    def add_root_adaptive_visit_orchestration(self, elapsed_seconds: float) -> None:
        self.root_adaptive_visit_orchestration_seconds += elapsed_seconds
        self.root_adaptive_visit_orchestration_count += 1

    def add_root_search_finalization(self, elapsed_seconds: float) -> None:
        self.root_search_finalization_seconds += elapsed_seconds
        self.root_search_finalization_count += 1

    def add_branch_action_validation(self, elapsed_seconds: float) -> None:
        self.branch_action_validation_seconds += elapsed_seconds
        self.branch_action_validation_count += 1

    def add_post_branch_history(self, elapsed_seconds: float) -> None:
        self.post_branch_history_seconds += elapsed_seconds
        self.post_branch_history_count += 1

    def add_bridge_subtiming(self, timing: Mapping[str, float | int]) -> None:
        self.bridge_round_trip_seconds += float(timing["bridge_round_trip_seconds"])
        self.bridge_round_trip_count += int(timing["bridge_round_trip_count"])
        self.bridge_node_processing_seconds += float(timing["bridge_node_processing_seconds"])
        self.bridge_node_processing_count += int(timing["bridge_node_processing_count"])

    def add_value_evaluation(self, elapsed_seconds: float, *, count: int = 1) -> None:
        if count <= 0:
            raise ValueError("value evaluation count must be positive.")
        self.value_evaluation_seconds += elapsed_seconds
        self.value_evaluation_count += count

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
            root_initial_sweep_orchestration_seconds=self.root_initial_sweep_orchestration_seconds,
            root_initial_sweep_orchestration_count=self.root_initial_sweep_orchestration_count,
            root_search_setup_seconds=self.root_search_setup_seconds,
            root_search_setup_count=self.root_search_setup_count,
            root_adaptive_visit_orchestration_seconds=(
                self.root_adaptive_visit_orchestration_seconds
            ),
            root_adaptive_visit_orchestration_count=self.root_adaptive_visit_orchestration_count,
            root_search_finalization_seconds=self.root_search_finalization_seconds,
            root_search_finalization_count=self.root_search_finalization_count,
            branch_action_validation_seconds=self.branch_action_validation_seconds,
            branch_action_validation_count=self.branch_action_validation_count,
            post_branch_history_seconds=self.post_branch_history_seconds,
            post_branch_history_count=self.post_branch_history_count,
            bridge_round_trip_seconds=self.bridge_round_trip_seconds,
            bridge_round_trip_count=self.bridge_round_trip_count,
            bridge_node_processing_seconds=self.bridge_node_processing_seconds,
            bridge_node_processing_count=self.bridge_node_processing_count,
            value_evaluation_seconds=self.value_evaluation_seconds,
            value_evaluation_count=self.value_evaluation_count,
            rollout_tail_seconds=self.rollout_tail_seconds,
            rollout_tail_count=self.rollout_tail_count,
            total_seconds=total_seconds,
        )


def _bridge_timing_snapshot(env: PokeZeroEnv) -> dict[str, float | int] | None:
    """Read optional cumulative bridge counters without changing search behavior.

    Only the local Showdown environment exposes these counters. Other search
    environments retain their existing timing behavior, and a malformed
    diagnostic source is ignored rather than becoming a policy failure.
    """

    source = getattr(env, "root_puct_bridge_timing_snapshot", None)
    if not callable(source):
        return None
    try:
        payload = source()
    except Exception:
        return None
    if not isinstance(payload, Mapping):
        return None
    result: dict[str, float | int] = {}
    for field in _BRIDGE_TIMING_SECONDS:
        value = payload.get(field)
        if isinstance(value, bool) or not isinstance(value, (float, int)):
            return None
        parsed = float(value)
        if not math.isfinite(parsed) or parsed < 0.0:
            return None
        result[field] = parsed
    for field in _BRIDGE_TIMING_COUNTS:
        value = payload.get(field)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            return None
        result[field] = value
    return result


def _bridge_timing_delta(
    before: Mapping[str, float | int] | None,
    after: Mapping[str, float | int] | None,
) -> dict[str, float | int] | None:
    """Return one PUCT call's monotonic bridge-counter delta, if available."""

    if before is None or after is None:
        return None
    result: dict[str, float | int] = {}
    for field in _BRIDGE_TIMING_SECONDS:
        delta = float(after[field]) - float(before[field])
        if delta < 0.0:
            return None
        result[field] = delta
    for field in _BRIDGE_TIMING_COUNTS:
        delta = int(after[field]) - int(before[field])
        if delta < 0:
            return None
        result[field] = delta
    return result


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
    restore_snapshot: Callable[[Any], None]
    snapshot_restore_mode: str = "generic"


@dataclass(frozen=True)
class _PendingValueBranch:
    """One non-terminal mandatory-sweep leaf awaiting a value evaluation."""

    action_index: int
    branch: ReplayBranchResult
    post_branch_history: tuple[PokeZeroObservationV0, ...]


@dataclass
class _PreparedValueBranchSearch:
    """Branch-point work that can be completed by one or more value batches."""

    player_id: PlayerId
    prefix_decision_round_count: int
    opponent_actions: Mapping[PlayerId, int]
    candidate_indices: tuple[int, ...]
    candidates_by_action: dict[int, ValueBranchSearchCandidate]
    pending_value_branches: tuple[_PendingValueBranch, ...]
    restorable_prefix: _RestorablePrefix | None


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
    snapshot_restore_mode: str = "generic"


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
class PUCTBranchSearchRequest:
    """Scenario-specific inputs for a shared initial Root-PUCT value batch."""

    opponent_actions: Mapping[PlayerId, int]
    start_override: StartOverrideSource = None
    prepared_prefix: PreparedReplayPrefix | None = None
    root_visit_budget_resolver: RootVisitBudgetResolver | None = None


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
    value_batch_fn: ObservationValueBatchFunction | None = None,
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
        value_batch_fn=value_batch_fn,
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
    value_batch_fn: ObservationValueBatchFunction | None = None,
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

    prepared = _prepare_value_branch_search(
        env=env,
        trajectory=trajectory,
        player_id=player_id,
        prefix_decision_round_count=prefix_decision_round_count,
        legal_action_mask=legal_action_mask,
        opponent_actions=opponent_actions,
        value_fn=value_fn,
        value_batch_fn=value_batch_fn,
        leaf_rollout_policies=leaf_rollout_policies,
        leaf_rollout_config=leaf_rollout_config,
        leaf_rollout_decision_rounds=leaf_rollout_decision_rounds,
        start_override=start_override,
        expected_current_observation=expected_current_observation,
        replay_hp_fraction_tolerance=replay_hp_fraction_tolerance,
        timing=timing,
        prepared_prefix=prepared_prefix,
    )
    batch_values = (
        _timed_value_batch_evaluation(
            value_batch_fn,
            histories=tuple(pending.post_branch_history for pending in prepared.pending_value_branches),
            timing=timing,
        )
        if prepared.pending_value_branches
        else ()
    )
    return (
        _complete_prepared_value_branch_search(prepared, batch_values=batch_values),
        prepared.restorable_prefix,
    )


def _prepare_value_branch_search(
    *,
    env: PokeZeroEnv,
    trajectory: BattleTrajectory,
    player_id: PlayerId,
    prefix_decision_round_count: int,
    legal_action_mask: tuple[bool, ...],
    opponent_actions: Mapping[PlayerId, int],
    value_fn: ObservationValueFunction,
    value_batch_fn: ObservationValueBatchFunction | None,
    leaf_rollout_policies: Mapping[PlayerId, Policy] | None,
    leaf_rollout_config: RolloutConfig | None,
    leaf_rollout_decision_rounds: int,
    start_override: StartOverrideSource,
    expected_current_observation: PokeZeroObservationV0 | None,
    replay_hp_fraction_tolerance: float,
    timing: _RootPUCTSearchTimingAccumulator | None,
    prepared_prefix: PreparedReplayPrefix | None,
) -> _PreparedValueBranchSearch:
    """Build mandatory root leaves, deferring only independent batched values."""

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
        env=env,
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
    candidates_by_action: dict[int, ValueBranchSearchCandidate] = {}
    pending_value_branches: list[_PendingValueBranch] = []
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
        if value_batch_fn is not None and leaf_rollout_decision_rounds == 0 and branch.step_result.terminal is None:
            pending_value_branches.append(
                _PendingValueBranch(
                    action_index=action_index,
                    branch=branch,
                    post_branch_history=_post_branch_history(
                        env=env,
                        player_id=player_id,
                        prefix_history=prefix_history,
                        branch=branch,
                        timing=timing,
                    ),
                )
            )
            continue
        candidates_by_action[action_index] = _value_branch_candidate(
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

    return _PreparedValueBranchSearch(
        player_id=player_id,
        prefix_decision_round_count=prefix_decision_round_count,
        opponent_actions=dict(opponent_actions),
        candidate_indices=candidate_indices,
        candidates_by_action=candidates_by_action,
        pending_value_branches=tuple(pending_value_branches),
        restorable_prefix=restorable_prefix,
    )


def _complete_prepared_value_branch_search(
    prepared: _PreparedValueBranchSearch,
    *,
    batch_values: Sequence[float],
) -> ValueBranchSearchResult:
    """Attach batch values to a prepared mandatory root sweep."""

    if len(batch_values) != len(prepared.pending_value_branches):
        raise ValueError("value batch evaluator returned a different number of values than histories.")
    for pending, value in zip(prepared.pending_value_branches, batch_values, strict=True):
        prepared.candidates_by_action[pending.action_index] = ValueBranchSearchCandidate(
            action_index=pending.action_index,
            value=_finite_value(value),
            terminal=None,
            branch=pending.branch,
            evaluated_history_length=len(pending.post_branch_history),
            leaf_evaluation="value_fn",
            leaf_rollout_decision_round_count=0,
        )

    candidates = [
        prepared.candidates_by_action[action_index]
        for action_index in prepared.candidate_indices
        if action_index in prepared.candidates_by_action
    ]

    if not candidates:
        raise ValueError("value branch search found no replay-legal root actions.")

    result = ValueBranchSearchResult(
        player_id=prepared.player_id,
        prefix_decision_round_count=prepared.prefix_decision_round_count,
        opponent_actions=dict(prepared.opponent_actions),
        candidates=tuple(candidates),
    )
    return result


def puct_branch_search(
    *,
    env: PokeZeroEnv,
    trajectory: BattleTrajectory,
    player_id: PlayerId,
    prefix_decision_round_count: int,
    legal_action_mask: tuple[bool, ...],
    opponent_actions: Mapping[PlayerId, int],
    value_fn: ObservationValueFunction,
    value_batch_fn: ObservationValueBatchFunction | None = None,
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
    bridge_timing_before = _bridge_timing_snapshot(env)
    initial_sweep_started_at = _timing_perf_counter()
    branch_evaluation_before = timing.branch_evaluation_seconds
    value_search, restorable_prefix = _value_branch_search_with_prefix(
        env=env,
        trajectory=trajectory,
        player_id=player_id,
        prefix_decision_round_count=prefix_decision_round_count,
        legal_action_mask=legal_action_mask,
        opponent_actions=opponent_actions,
        value_fn=value_fn,
        value_batch_fn=value_batch_fn,
        leaf_rollout_policies=leaf_rollout_policies,
        leaf_rollout_config=leaf_rollout_config,
        leaf_rollout_decision_rounds=leaf_rollout_decision_rounds,
        start_override=start_override,
        expected_current_observation=expected_current_observation,
        replay_hp_fraction_tolerance=replay_hp_fraction_tolerance,
        timing=timing,
        prepared_prefix=prepared_prefix,
    )
    timing.add_root_initial_sweep_orchestration(
        max(
            0.0,
            _timing_perf_counter()
            - initial_sweep_started_at
            - (timing.branch_evaluation_seconds - branch_evaluation_before),
        )
    )
    return _finish_puct_branch_search(
        env=env,
        trajectory=trajectory,
        player_id=player_id,
        prefix_decision_round_count=prefix_decision_round_count,
        opponent_actions=opponent_actions,
        value_fn=value_fn,
        action_priors=action_priors,
        cpuct=cpuct,
        leaf_rollout_policies=leaf_rollout_policies,
        leaf_rollout_config=leaf_rollout_config,
        leaf_rollout_decision_rounds=leaf_rollout_decision_rounds,
        root_visit_budget=root_visit_budget,
        root_visit_budget_resolver=root_visit_budget_resolver,
        budget_action_priors=budget_action_priors,
        root_time_budget_seconds=root_time_budget_seconds,
        start_override=start_override,
        expected_current_observation=expected_current_observation,
        replay_hp_fraction_tolerance=replay_hp_fraction_tolerance,
        value_search=value_search,
        restorable_prefix=restorable_prefix,
        timing=timing,
        time_budget_start=time_budget_start,
        bridge_timing_before=bridge_timing_before,
        timing_started_at=timing_started_at,
    )


def puct_branch_search_group(
    *,
    env: PokeZeroEnv,
    trajectory: BattleTrajectory,
    player_id: PlayerId,
    prefix_decision_round_count: int,
    legal_action_mask: tuple[bool, ...],
    requests: Sequence[PUCTBranchSearchRequest],
    value_fn: ObservationValueFunction,
    value_batch_fn: ObservationValueBatchFunction,
    action_priors: ActionPriorVector,
    cpuct: float = 1.25,
    leaf_rollout_policies: Mapping[PlayerId, Policy] | None = None,
    leaf_rollout_config: RolloutConfig | None = None,
    leaf_rollout_decision_rounds: int = 0,
    root_visit_budget: int | None = None,
    budget_action_priors: ActionPriorVector | None = None,
    expected_current_observation: PokeZeroObservationV0 | None = None,
    replay_hp_fraction_tolerance: float = 0.0,
) -> tuple[PUCTBranchSearchResult, ...]:
    """Batch mandatory root leaves across independent sampled worlds.

    Each request still constructs, validates, and searches its own determinized
    world. Only the initial, independent value leaves share one evaluator call.
    Adaptive visits remain sequential after their own backups, so this cannot
    alter PUCT selection semantics. Time-budgeted and leaf-rollout searches are
    intentionally excluded because grouping either would change their budget or
    continuation semantics.
    """

    if not requests:
        raise ValueError("puct branch search group requires at least one request.")
    if cpuct < 0.0 or not math.isfinite(cpuct):
        raise ValueError("cpuct must be a finite non-negative value.")
    if root_visit_budget is not None and root_visit_budget <= 0:
        raise ValueError("root_visit_budget must be positive when set.")
    if leaf_rollout_decision_rounds != 0:
        raise ValueError("puct branch search group supports only zero leaf rollout decision rounds.")
    if replay_hp_fraction_tolerance < 0.0 or not math.isfinite(replay_hp_fraction_tolerance):
        raise ValueError("replay_hp_fraction_tolerance must be a finite non-negative value.")

    prepared_requests: list[
        tuple[
            PUCTBranchSearchRequest,
            _PreparedValueBranchSearch,
            _RootPUCTSearchTimingAccumulator,
            float,
        ]
    ] = []
    for request in requests:
        timing = _RootPUCTSearchTimingAccumulator()
        bridge_timing_before = _bridge_timing_snapshot(env)
        initial_sweep_started_at = _timing_perf_counter()
        branch_evaluation_before = timing.branch_evaluation_seconds
        prepared = _prepare_value_branch_search(
            env=env,
            trajectory=trajectory,
            player_id=player_id,
            prefix_decision_round_count=prefix_decision_round_count,
            legal_action_mask=legal_action_mask,
            opponent_actions=request.opponent_actions,
            value_fn=value_fn,
            value_batch_fn=value_batch_fn,
            leaf_rollout_policies=leaf_rollout_policies,
            leaf_rollout_config=leaf_rollout_config,
            leaf_rollout_decision_rounds=leaf_rollout_decision_rounds,
            start_override=request.start_override,
            expected_current_observation=expected_current_observation,
            replay_hp_fraction_tolerance=replay_hp_fraction_tolerance,
            timing=timing,
            prepared_prefix=request.prepared_prefix,
        )
        initial_elapsed_seconds = _timing_perf_counter() - initial_sweep_started_at
        timing.add_root_initial_sweep_orchestration(
            max(
                0.0,
                initial_elapsed_seconds - (timing.branch_evaluation_seconds - branch_evaluation_before),
            )
        )
        bridge_timing = _bridge_timing_delta(
            bridge_timing_before,
            _bridge_timing_snapshot(env),
        )
        if bridge_timing is not None:
            timing.add_bridge_subtiming(bridge_timing)
        prepared_requests.append((request, prepared, timing, initial_elapsed_seconds))

    pending_histories = tuple(
        pending.post_branch_history
        for _request, prepared, _timing, _elapsed in prepared_requests
        for pending in prepared.pending_value_branches
    )
    batch_values: tuple[float, ...] = ()
    batch_elapsed_seconds = 0.0
    if pending_histories:
        batch_started_at = _timing_perf_counter()
        batch_values = tuple(_finite_value(value) for value in value_batch_fn(pending_histories))
        batch_elapsed_seconds = _timing_perf_counter() - batch_started_at
        if len(batch_values) != len(pending_histories):
            raise ValueError(
                "value batch evaluation returned a different number of values than input histories."
            )

    completed_value_searches: list[
        tuple[
            PUCTBranchSearchRequest,
            ValueBranchSearchResult,
            _RestorablePrefix | None,
            _RootPUCTSearchTimingAccumulator,
            float,
        ]
    ] = []
    value_offset = 0
    total_pending_count = len(pending_histories)
    remaining_pending_count = total_pending_count
    remaining_batch_seconds = batch_elapsed_seconds
    for request, prepared, timing, initial_elapsed_seconds in prepared_requests:
        pending_count = len(prepared.pending_value_branches)
        request_values = batch_values[value_offset : value_offset + pending_count]
        value_offset += pending_count
        if pending_count:
            if pending_count == remaining_pending_count:
                request_batch_seconds = remaining_batch_seconds
            else:
                request_batch_seconds = batch_elapsed_seconds * pending_count / total_pending_count
            remaining_batch_seconds -= request_batch_seconds
            remaining_pending_count -= pending_count
            timing.add_value_evaluation(request_batch_seconds, count=pending_count)
        else:
            request_batch_seconds = 0.0
        completed_value_searches.append(
            (
                request,
                _complete_prepared_value_branch_search(prepared, batch_values=request_values),
                prepared.restorable_prefix,
                timing,
                initial_elapsed_seconds + request_batch_seconds,
            )
        )
    assert value_offset == len(batch_values)

    results: list[PUCTBranchSearchResult] = []
    for request, value_search, restorable_prefix, timing, initial_elapsed_seconds in completed_value_searches:
        adaptive_bridge_timing_before = _bridge_timing_snapshot(env)
        adaptive_started_at = _timing_perf_counter()
        results.append(
            _finish_puct_branch_search(
                env=env,
                trajectory=trajectory,
                player_id=player_id,
                prefix_decision_round_count=prefix_decision_round_count,
                opponent_actions=request.opponent_actions,
                value_fn=value_fn,
                action_priors=action_priors,
                cpuct=cpuct,
                leaf_rollout_policies=leaf_rollout_policies,
                leaf_rollout_config=leaf_rollout_config,
                leaf_rollout_decision_rounds=leaf_rollout_decision_rounds,
                root_visit_budget=root_visit_budget,
                root_visit_budget_resolver=request.root_visit_budget_resolver,
                budget_action_priors=budget_action_priors,
                root_time_budget_seconds=None,
                start_override=request.start_override,
                expected_current_observation=expected_current_observation,
                replay_hp_fraction_tolerance=replay_hp_fraction_tolerance,
                value_search=value_search,
                restorable_prefix=restorable_prefix,
                timing=timing,
                time_budget_start=None,
                bridge_timing_before=adaptive_bridge_timing_before,
                timing_started_at=adaptive_started_at,
                initial_elapsed_seconds=initial_elapsed_seconds,
            )
        )
    return tuple(results)


def _finish_puct_branch_search(
    *,
    env: PokeZeroEnv,
    trajectory: BattleTrajectory,
    player_id: PlayerId,
    prefix_decision_round_count: int,
    opponent_actions: Mapping[PlayerId, int],
    value_fn: ObservationValueFunction,
    action_priors: ActionPriorVector,
    cpuct: float,
    leaf_rollout_policies: Mapping[PlayerId, Policy] | None,
    leaf_rollout_config: RolloutConfig | None,
    leaf_rollout_decision_rounds: int,
    root_visit_budget: int | None,
    root_visit_budget_resolver: RootVisitBudgetResolver | None,
    budget_action_priors: ActionPriorVector | None,
    root_time_budget_seconds: float | None,
    start_override: StartOverrideSource,
    expected_current_observation: PokeZeroObservationV0 | None,
    replay_hp_fraction_tolerance: float,
    value_search: ValueBranchSearchResult,
    restorable_prefix: _RestorablePrefix | None,
    timing: _RootPUCTSearchTimingAccumulator,
    time_budget_start: float | None,
    bridge_timing_before: Mapping[str, float | int] | None,
    timing_started_at: float,
    initial_elapsed_seconds: float | None = None,
) -> PUCTBranchSearchResult:
    """Run adaptive root visits after a completed mandatory value sweep."""

    adaptive_started_at = _timing_perf_counter()
    root_setup_started_at = _timing_perf_counter()
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
    timing.add_root_search_setup(_timing_perf_counter() - root_setup_started_at)
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
        adaptive_visit_started_at = _timing_perf_counter()
        branch_evaluation_before = timing.branch_evaluation_seconds
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
        timing.add_root_adaptive_visit_orchestration(
            max(
                0.0,
                _timing_perf_counter()
                - adaptive_visit_started_at
                - (timing.branch_evaluation_seconds - branch_evaluation_before),
            )
        )
    finalization_started_at = _timing_perf_counter()
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
    bridge_timing = _bridge_timing_delta(
        bridge_timing_before,
        _bridge_timing_snapshot(env),
    )
    if bridge_timing is not None:
        timing.add_bridge_subtiming(bridge_timing)
    timing.add_root_search_finalization(_timing_perf_counter() - finalization_started_at)
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
        timing=timing.finish(
            _timing_perf_counter() - timing_started_at
            if initial_elapsed_seconds is None
            else initial_elapsed_seconds + (_timing_perf_counter() - adaptive_started_at)
        ),
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
        timing=timing,
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
    snapshot_hooks = _snapshot_hooks(env, prefer_search_snapshots=True)
    if snapshot_hooks is None:
        return None
    snapshotter, _restorer, snapshot_restore_mode = snapshot_hooks
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
        snapshot_restore_mode=snapshot_restore_mode,
    )


def prepare_direct_materialization_prefix(
    *,
    env: PokeZeroEnv,
    trajectory: BattleTrajectory,
    player_id: PlayerId,
    prefix_decision_round_count: int,
    start_override: BattleStartOverride | None,
    public_materialization_state: object | None,
    deferred_opponent_actions: Mapping[PlayerId, int] | None = None,
    deferred_opponent_action_priors: Mapping[PlayerId, Sequence[float]] | None = None,
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
    snapshot_hooks = _snapshot_hooks(env, prefer_search_snapshots=True)
    if not callable(materializer) or snapshot_hooks is None:
        _record_direct_materialization_unavailable(on_unavailable, "environment_unavailable")
        return None
    snapshotter, _restorer, snapshot_restore_mode = snapshot_hooks
    try:
        materialization_kwargs: dict[str, object] = {
            "state": public_materialization_state,
            "start_override": start_override,
            "seed": trajectory.seed,
        }
        if deferred_opponent_actions:
            materialization_kwargs["deferred_opponent_actions"] = dict(deferred_opponent_actions)
        if deferred_opponent_action_priors:
            materialization_kwargs["deferred_opponent_action_priors"] = dict(
                deferred_opponent_action_priors
            )
        materializer(**materialization_kwargs)
        if expected_current_observation is not None:
            require_current_observation_match(
                env,
                expected=expected_current_observation,
                player_id=player_id,
                turn_index=prefix_decision_round_count,
                hp_fraction_tolerance=replay_hp_fraction_tolerance,
            )
    except (RuntimeError, ValueError) as error:
        category = root_puct_direct_materialization_rejection_category(error)
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
        snapshot_restore_mode=snapshot_restore_mode,
    )


def release_prepared_replay_prefix(env: PokeZeroEnv, prepared_prefix: PreparedReplayPrefix) -> bool:
    """Release a bridge handle after its final scenario without affecting generic snapshots."""

    if prepared_prefix.snapshot_restore_mode != "bridge-handle":
        return False
    releaser = getattr(env, "release_search_snapshot", None)
    if not callable(releaser):
        return False
    result = releaser(prepared_prefix.snapshot)
    return bool(result)


def _record_direct_materialization_unavailable(
    callback: Callable[[str], None] | None,
    category: str,
) -> None:
    if callback is not None:
        callback(category)


def _restorable_prefix_from_prepared(
    prepared_prefix: PreparedReplayPrefix | None,
    *,
    env: PokeZeroEnv,
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
    restorer = _snapshot_restorer_for_mode(env, prepared_prefix.snapshot_restore_mode)
    if restorer is None:
        raise ValueError("prepared replay prefix restore path became unavailable.")
    return _RestorablePrefix(
        prefix=prepared_prefix.prefix,
        snapshot=prepared_prefix.snapshot,
        restore_snapshot=restorer,
        snapshot_restore_mode=prepared_prefix.snapshot_restore_mode,
    )


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
    snapshot_hooks = _snapshot_hooks(
        env,
        prefer_search_snapshots=start_override is not None and not callable(start_override),
    )
    if snapshot_hooks is None:
        return None
    snapshotter, restorer, snapshot_restore_mode = snapshot_hooks
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
    return _RestorablePrefix(
        prefix=prefix,
        snapshot=snapshot,
        restore_snapshot=restorer,
        snapshot_restore_mode=snapshot_restore_mode,
    )


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
        branch_round = _validated_branch_round(
            branch_actions=branch_actions,
            requested_players=prefix.requested_players,
            turn_index=prefix_decision_round_count,
            timing=timing,
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
    step_from_search_snapshot = (
        getattr(env, "step_from_search_snapshot", None)
        if restorable_prefix.snapshot_restore_mode == "bridge-handle"
        else None
    )
    if not callable(step_from_search_snapshot):
        # Preserve the generic/oracle path's original restore-before-validation ordering. It is
        # deliberately separate from the bridge-handle fusion below.
        restore_started_at = _timing_perf_counter() if timing is not None else None
        restorable_prefix.restore_snapshot(restorable_prefix.snapshot)
        if timing is not None:
            assert restore_started_at is not None
            timing.add_state_restore(_timing_perf_counter() - restore_started_at)
        branch_round = _validated_branch_round(
            branch_actions=branch_actions,
            requested_players=restorable_prefix.prefix.requested_players,
            turn_index=prefix_decision_round_count,
            timing=timing,
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

    branch_round = _validated_branch_round(
        branch_actions=branch_actions,
        requested_players=restorable_prefix.prefix.requested_players,
        turn_index=prefix_decision_round_count,
        timing=timing,
    )
    # This optional fast path is restricted to bridge-resident snapshots created from a
    # belief-sampled world. The fused bridge command includes both restore and branch
    # execution, so it is intentionally recorded as one non-overlapping branch stage.
    branch_step_started_at = _timing_perf_counter() if timing is not None else None
    step_result = step_from_search_snapshot(restorable_prefix.snapshot, branch_round.actions)
    if timing is not None:
        assert branch_step_started_at is not None
        timing.add_branch_simulator_step(_timing_perf_counter() - branch_step_started_at)
    return ReplayBranchResult(
        prefix=restorable_prefix.prefix,
        branch_round=branch_round,
        step_result=step_result,
    )


def _validated_branch_round(
    *,
    branch_actions: Mapping[PlayerId, int],
    requested_players: tuple[PlayerId, ...],
    turn_index: int,
    timing: _RootPUCTSearchTimingAccumulator | None,
) -> ReplayActionRound:
    """Build and validate one branch action map outside simulator timing."""

    started_at = _timing_perf_counter() if timing is not None else None
    branch_round = ReplayActionRound(turn_index=turn_index, actions=branch_actions)
    _require_exact_requested_players(
        branch_actions=branch_round.actions,
        requested_players=requested_players,
        turn_index=turn_index,
    )
    if timing is not None:
        assert started_at is not None
        timing.add_branch_action_validation(_timing_perf_counter() - started_at)
    return branch_round


def _snapshot_hooks(
    env: PokeZeroEnv,
    *,
    prefer_search_snapshots: bool = False,
) -> tuple[Callable[[], Any], Callable[[Any], None], str] | None:
    """Return snapshot hooks without using search handles for live/oracle rollouts."""

    if prefer_search_snapshots:
        snapshotter = getattr(env, "snapshot_for_search", None)
        restorer = getattr(env, "restore_search_snapshot", None)
        if callable(snapshotter) and callable(restorer):
            return snapshotter, restorer, "bridge-handle"
    snapshotter = getattr(env, "snapshot", None)
    restorer = getattr(env, "restore", None)
    if callable(snapshotter) and callable(restorer):
        return snapshotter, restorer, "generic"
    return None


def _snapshot_restorer_for_mode(
    env: PokeZeroEnv,
    mode: str,
) -> Callable[[Any], None] | None:
    if mode == "bridge-handle":
        restorer = getattr(env, "restore_search_snapshot", None)
    elif mode == "generic":
        restorer = getattr(env, "restore", None)
    else:
        raise ValueError(f"prepared replay prefix has unsupported snapshot restore mode {mode!r}.")
    return restorer if callable(restorer) else None


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
    timing: _RootPUCTSearchTimingAccumulator | None = None,
) -> tuple[PokeZeroObservationV0, ...]:
    started_at = _timing_perf_counter() if timing is not None else None
    post_branch_observation = branch.step_result.observations.get(player_id)
    if post_branch_observation is None:
        post_branch_observation = env.observe(player_id)
    result = (*prefix_history, post_branch_observation)
    if timing is not None:
        assert started_at is not None
        timing.add_post_branch_history(_timing_perf_counter() - started_at)
    return result


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


def _timed_value_batch_evaluation(
    value_batch_fn: ObservationValueBatchFunction,
    *,
    histories: tuple[tuple[PokeZeroObservationV0, ...], ...],
    timing: _RootPUCTSearchTimingAccumulator | None,
) -> tuple[float, ...]:
    """Evaluate independent mandatory root leaves in one model call.

    This applies only to the initial one-visit-per-action sweep. Later PUCT
    visits remain sequential because each selection depends on the preceding
    backup, so batching cannot silently change the search policy.
    """

    if not histories:
        raise ValueError("value batch evaluation requires at least one history.")
    started_at = _timing_perf_counter() if timing is not None else None
    values = tuple(_finite_value(value) for value in value_batch_fn(histories))
    if len(values) != len(histories):
        raise ValueError(
            "value batch evaluation returned a different number of values than input histories."
        )
    if timing is not None:
        assert started_at is not None
        timing.add_value_evaluation(
            _timing_perf_counter() - started_at,
            count=len(histories),
        )
    return values


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
