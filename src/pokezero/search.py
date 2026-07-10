"""Search helpers built on replay-from-root branch rollouts."""

from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import math
from time import perf_counter
from typing import Any, Callable, Mapping

from .actions import ACTION_COUNT
from .env import BattleStartOverride, PlayerId, PokeZeroEnv, TerminalState
from .observation import PokeZeroObservationV0
from .policy import Policy
from .replay_branching import (
    ReplayActionRound,
    ReplayBranchResult,
    ReplayBranchRolloutResult,
    ReplayPrefixResult,
    replay_trajectory_branch,
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
        return -sum(prior * math.log(prior) for _action, prior in self.action_priors if prior > 0.0)

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

    @property
    def best_candidate(self) -> PUCTBranchSearchCandidate:
        if not self.candidates:
            raise ValueError("PUCT branch search produced no candidates.")
        return max(self.candidates, key=lambda candidate: (candidate.score, candidate.value, -candidate.action_index))

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
    restorable_prefix = _restorable_prefix_snapshot(
        env=env,
        trajectory=trajectory,
        player_id=player_id,
        prefix_decision_round_count=prefix_decision_round_count,
        start_override=start_override,
        expected_current_observation=expected_current_observation,
        replay_hp_fraction_tolerance=replay_hp_fraction_tolerance,
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
            value=_finite_value(value_fn(post_branch_history)),
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
            value=_finite_value(value_fn(post_branch_history)),
            terminal=None,
            branch=branch,
            evaluated_history_length=len(post_branch_history),
            leaf_evaluation="value_fn",
            leaf_rollout_decision_round_count=0,
        )

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
        value = _finite_value(value_fn(evaluated_history))
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


def _restorable_prefix_snapshot(
    *,
    env: PokeZeroEnv,
    trajectory: BattleTrajectory,
    player_id: PlayerId,
    prefix_decision_round_count: int,
    start_override: StartOverrideSource,
    expected_current_observation: PokeZeroObservationV0 | None,
    replay_hp_fraction_tolerance: float,
) -> _RestorablePrefix | None:
    snapshotter = getattr(env, "snapshot", None)
    restorer = getattr(env, "restore", None)
    if not callable(snapshotter) or not callable(restorer):
        return None
    if callable(start_override):
        return None
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
    if prefix.terminal is not None:
        raise ValueError("cannot branch from a terminal replay prefix.")
    return _RestorablePrefix(prefix=prefix, snapshot=snapshotter())


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
) -> ReplayBranchResult:
    if restorable_prefix is None:
        return replay_trajectory_branch(
            env,
            trajectory,
            prefix_decision_round_count=prefix_decision_round_count,
            branch_actions=branch_actions,
            start_override=_materialize_start_override(start_override),
            consistency_player_id=player_id,
            expected_current_observation=expected_current_observation,
            # Root search scores the current decision point. Earlier custom-game replay
            # observations can drift while sampled hidden worlds are rejected by the
            # branch-point observation check below.
            check_prefix_observations=False,
            hp_fraction_tolerance=replay_hp_fraction_tolerance,
        )
    restorer = getattr(env, "restore", None)
    if not callable(restorer):
        raise ValueError("environment snapshot restore became unavailable.")
    restorer(restorable_prefix.snapshot)
    branch_round = ReplayActionRound(
        turn_index=prefix_decision_round_count,
        actions=branch_actions,
    )
    _require_exact_requested_players(
        branch_actions=branch_actions,
        requested_players=restorable_prefix.prefix.requested_players,
        turn_index=prefix_decision_round_count,
    )
    step_result = env.step(branch_round.actions)
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


def _is_candidate_illegal_action_error(exc: ValueError, *, player_id: PlayerId, action_index: int) -> bool:
    message = str(exc)
    unqualified = f"action_index {action_index} is not legal for the current request."
    return message == unqualified or message == f"{player_id}: {unqualified}"


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
