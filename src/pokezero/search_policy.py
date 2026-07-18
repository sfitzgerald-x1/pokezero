"""Policy adapters backed by replay-from-root search."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field, fields, replace
import hashlib
from itertools import product
from time import perf_counter
from time import perf_counter as _timing_perf_counter
import math
import random
import re
from typing import Callable, Mapping, Sequence

from .actions import ACTION_COUNT, MOVE_ACTION_COUNT
from .env import BattleStartOverride, PlayerId, PokeZeroEnv
from .mcts_diagnostics import (
    root_puct_first_observation_mismatch_path_counts,
    root_puct_fallback_category,
    root_puct_missing_sampled_world_reason_counts,
    root_puct_replay_rejection_decision_round_counts,
    root_puct_replay_request_mismatch_decision_round_counts,
    root_puct_replay_request_mismatch_player_counts,
    root_puct_replay_request_mismatch_shape_counts,
    root_puct_start_override_mismatch_decision_round_counts,
)
from .observation import PokeZeroObservationV0
from .policy import Policy, PolicyContext, PolicyDecision, RandomLegalPolicy, legal_action_indices
from .rollout import RolloutConfig, _reset_unique_policies
from .search import (
    ActionPriorVector,
    ObservationValueBatchFunction,
    ObservationValueFunction,
    PUCTBranchSearchCandidate,
    PUCTBranchSearchResult,
    PreparedReplayPrefix,
    RootPUCTSearchTiming,
    RootPUCTVisitBudgetContext,
    RootVisitBudgetResolver,
    START_OVERRIDE_MISSING_WORLD_MESSAGE,
    StartOverrideSource,
    _materialize_start_override,
    _puct_candidate,
    prepare_direct_materialization_prefix,
    prepare_replay_prefix,
    player_observation_history,
    puct_branch_search,
    release_prepared_replay_prefix,
)
from .trajectory import BattleTrajectory, TrajectoryStep

OpponentActionPlanner = Callable[[PolicyContext, random.Random], Mapping[PlayerId, int]]
OpponentActionScenarioPlanner = Callable[[PolicyContext, random.Random], Sequence["OpponentActionScenario"]]
ActionPriorFunction = Callable[[tuple[PokeZeroObservationV0, ...]], ActionPriorVector]
OpponentActionPriorFunction = Callable[[tuple[PokeZeroObservationV0, ...]], ActionPriorVector]
LeafRolloutPolicyFactory = Callable[[PlayerId], Policy]
RootVisitBudgetSelector = Callable[[PolicyContext, RootPUCTVisitBudgetContext], int | None]
NeuralTimingSnapshot = Callable[[], Mapping[str, float | int]]

_NEURAL_TIMING_SECONDS = (
    "observation_encoding_seconds",
    "neural_forward_seconds",
    "action_prior_neural_forward_seconds",
    "opponent_action_prior_neural_forward_seconds",
    "policy_neural_forward_seconds",
    "value_neural_forward_seconds",
)
_NEURAL_TIMING_COUNTS = (
    "observation_encoding_count",
    "neural_forward_count",
    "action_prior_neural_forward_count",
    "opponent_action_prior_neural_forward_count",
    "policy_neural_forward_count",
    "value_neural_forward_count",
)


def _neural_timing_snapshot(source: NeuralTimingSnapshot | None) -> dict[str, float | int] | None:
    """Normalize a cumulative evaluator-timing snapshot for one root decision."""

    if source is None:
        return None
    payload = source()
    if not isinstance(payload, Mapping):
        raise ValueError("neural_timing_snapshot must return a mapping.")
    result: dict[str, float | int] = {}
    for field in _NEURAL_TIMING_SECONDS:
        value = payload.get(field, 0.0)
        if isinstance(value, bool) or not isinstance(value, (float, int)) or value < 0.0:
            raise ValueError(f"neural_timing_snapshot has invalid {field}.")
        result[field] = float(value)
    for field in _NEURAL_TIMING_COUNTS:
        value = payload.get(field, 0)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"neural_timing_snapshot has invalid {field}.")
        result[field] = value
    return result


def _neural_timing_delta(
    before: Mapping[str, float | int] | None,
    after: Mapping[str, float | int] | None,
) -> dict[str, float | int]:
    """Return a validated per-decision delta from cumulative evaluator counters."""

    if before is None or after is None:
        return {
            **{field: 0.0 for field in _NEURAL_TIMING_SECONDS},
            **{field: 0 for field in _NEURAL_TIMING_COUNTS},
        }
    result: dict[str, float | int] = {}
    for field in _NEURAL_TIMING_SECONDS:
        delta = float(after[field]) - float(before[field])
        if delta < 0.0:
            raise ValueError(f"neural_timing_snapshot moved backwards for {field}.")
        result[field] = delta
    for field in _NEURAL_TIMING_COUNTS:
        delta = int(after[field]) - int(before[field])
        if delta < 0:
            raise ValueError(f"neural_timing_snapshot moved backwards for {field}.")
        result[field] = delta
    return result


@dataclass(frozen=True)
class FixedExtraVisitBudgetSelector:
    """Spend a fixed number of visits after the mandatory legal-action sweep.

    Root-PUCT always evaluates each legal action once.  Fixed experiment budgets
    therefore need to be expressed relative to that sweep rather than as an
    absolute visit cap whose effective extra work changes with legal-action
    count.
    """

    extra_visits: int
    selector_id: str = "fixed-extra-visits"

    def __post_init__(self) -> None:
        if (
            isinstance(self.extra_visits, bool)
            or not isinstance(self.extra_visits, int)
            or self.extra_visits < 0
        ):
            raise ValueError("extra_visits must be a non-negative integer.")

    def __call__(self, context: PolicyContext, budget_context: RootPUCTVisitBudgetContext) -> int:
        del context
        return len(budget_context.action_priors) + self.extra_visits

    def to_dict(self) -> dict[str, object]:
        return {
            "selector_id": self.selector_id,
            "extra_visits": self.extra_visits,
        }


@dataclass(frozen=True)
class EntropyMarginVisitBudgetSelector:
    """Spend additional root visits only at policy- or value-contested decisions."""

    contested_extra_visits: int
    uncontested_extra_visits: int = 0
    minimum_policy_entropy: float | None = None
    maximum_value_margin: float | None = None
    selector_id: str = "entropy-or-value-margin"

    def __post_init__(self) -> None:
        for name, value in (
            ("contested_extra_visits", self.contested_extra_visits),
            ("uncontested_extra_visits", self.uncontested_extra_visits),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer.")
        if self.minimum_policy_entropy is None and self.maximum_value_margin is None:
            raise ValueError("an entropy or value-margin threshold is required for adaptive visit budgeting.")
        for name, value in (
            ("minimum_policy_entropy", self.minimum_policy_entropy),
            ("maximum_value_margin", self.maximum_value_margin),
        ):
            if value is not None and (value < 0.0 or not math.isfinite(value)):
                raise ValueError(f"{name} must be a finite non-negative value when set.")

    def __call__(self, context: PolicyContext, budget_context: RootPUCTVisitBudgetContext) -> int:
        del context
        entropy_contested = (
            self.minimum_policy_entropy is not None
            and budget_context.policy_entropy >= self.minimum_policy_entropy
        )
        margin_contested = (
            self.maximum_value_margin is not None
            and budget_context.value_margin is not None
            and budget_context.value_margin <= self.maximum_value_margin
        )
        extra_visits = (
            self.contested_extra_visits
            if entropy_contested or margin_contested
            else self.uncontested_extra_visits
        )
        return len(budget_context.action_priors) + extra_visits

    def to_dict(self) -> dict[str, object]:
        return {
            "selector_id": self.selector_id,
            "contested_extra_visits": self.contested_extra_visits,
            "uncontested_extra_visits": self.uncontested_extra_visits,
            "minimum_policy_entropy": self.minimum_policy_entropy,
            "maximum_value_margin": self.maximum_value_margin,
        }


def no_opponent_action_planner(context: PolicyContext, rng: random.Random) -> Mapping[PlayerId, int]:
    del context, rng
    return {}


@dataclass(frozen=True)
class OpponentActionScenario:
    actions: Mapping[PlayerId, int]
    weight: float = 1.0
    label: str = "single"
    # An opponent action that was committed before a player-only forced-switch request. It is
    # reconstructed into the direct world's queue, not submitted alongside the root action.
    deferred_actions: Mapping[PlayerId, int] = field(default_factory=dict)
    # The opponent has no request at a forced-switch boundary. These player-local move-slot
    # priors are resolved against the sampled world's legal slots by direct materialization.
    deferred_action_priors: Mapping[PlayerId, tuple[float, ...]] = field(default_factory=dict)

    def normalized(self, *, total_weight: float) -> "OpponentActionScenario":
        return OpponentActionScenario(
            actions=dict(self.actions),
            deferred_actions=dict(self.deferred_actions),
            deferred_action_priors=dict(self.deferred_action_priors),
            weight=self.weight / total_weight,
            label=self.label,
        )


@dataclass(frozen=True)
class _OpponentActionScenarioGroup:
    root: OpponentActionScenario
    samples: tuple[OpponentActionScenario, ...]


@dataclass(frozen=True)
class _SharedStartOverrideSamples:
    overrides: tuple[BattleStartOverride | None, ...]
    prepared_prefixes: tuple[PreparedReplayPrefix | None, ...]
    materialization_modes: tuple[str | None, ...]
    rejection_reasons: tuple[str | None, ...]
    attempts_used: int
    duplicate_attempts: int = 0


StartOverridePlanner = Callable[
    [PolicyContext, OpponentActionScenario, int, random.Random],
    StartOverrideSource,
]


def greedy_opponent_action_planner(
    prior_fn: OpponentActionPriorFunction,
) -> OpponentActionPlanner:
    """Create a planner that predicts each requested opponent action from player-local history.

    The prior function sees only the acting player's observation history. It should model the
    opponent-action auxiliary head, not the acting player's legal-action policy head. When the
    rollout harness has already observed requested-player legal masks, this planner uses those
    masks only to keep the planned opponent action submit-valid for replay search.
    """

    def planner(context: PolicyContext, rng: random.Random) -> Mapping[PlayerId, int]:
        del rng
        requested_opponents = tuple(player for player in context.requested_players if player != context.player_id)
        if not requested_opponents:
            return {}
        trajectory = _trajectory_with_current_observation(context)
        history = player_observation_history(
            trajectory,
            player_id=context.player_id,
            through_decision_round=context.decision_round_index,
        )
        priors = tuple(float(value) for value in prior_fn(history))
        _validate_action_prior_vector(priors, name="opponent action priors")
        opponent_actions = {}
        for player in requested_opponents:
            legal = _requested_legal_action_indices_for_player(context, player)
            if legal:
                opponent_actions[player] = max(legal, key=lambda index: (priors[index], -index))
            else:
                opponent_actions[player] = max(range(ACTION_COUNT), key=lambda index: (priors[index], -index))
        return opponent_actions

    planner.planner_id = "checkpoint"  # type: ignore[attr-defined]
    planner.opponent_prior_fn = prior_fn  # type: ignore[attr-defined]
    return planner


def prior_top_k_opponent_action_scenario_planner(
    prior_fn: OpponentActionPriorFunction,
    *,
    scenario_count: int,
) -> OpponentActionScenarioPlanner:
    """Enumerate likely requested-opponent scenarios from player-local opponent priors.

    The prior function only sees the acting player's observation history. Requested-opponent legal
    masks are still a privileged benchmark safety guard, and they affect scenario support and
    weights so replay branches stay submit-valid. For multi-opponent turns, the final joint scenario
    set is capped to ``scenario_count`` after combining per-opponent choices. A move committed
    before a player-only forced switch has no live request; its four move-slot priors are instead
    conditioned on the sampled world's legal slots during direct materialization.
    """

    if scenario_count <= 0:
        raise ValueError("scenario_count must be positive.")

    def planner(context: PolicyContext, rng: random.Random) -> tuple[OpponentActionScenario, ...]:
        del rng
        requested_opponents = tuple(player for player in context.requested_players if player != context.player_id)
        deferred_opponents = _deferred_opponent_action_players(context)
        scenario_players = requested_opponents + deferred_opponents
        if not scenario_players:
            return (OpponentActionScenario(actions={}, weight=1.0, label="no-opponent"),)
        trajectory = _trajectory_with_current_observation(context)
        history = player_observation_history(
            trajectory,
            player_id=context.player_id,
            through_decision_round=context.decision_round_index,
        )
        priors = tuple(float(value) for value in prior_fn(history))
        _validate_action_prior_vector(priors, name="opponent action priors")
        choices_by_player = tuple(
            (
                player,
                _top_prior_action_choices(
                    context,
                    player,
                    priors,
                    limit=scenario_count,
                    rng=_opponent_action_choice_rng(context, player),
                ),
            )
            for player in requested_opponents
        )
        scenarios: list[OpponentActionScenario] = []
        combinations = product(*(choices for _player, choices in choices_by_player))
        deferred_action_priors = {
            player: tuple(priors[:MOVE_ACTION_COUNT])
            for player in deferred_opponents
        }
        for combination in combinations:
            actions = {
                player: action
                for (player, _choices), (action, _weight) in zip(choices_by_player, combination, strict=True)
                if player in requested_opponents
            }
            weight = math.prod(weight for _action, weight in combination)
            label_parts = [
                f"{player}:{action}"
                for (player, _choices), (action, _weight) in zip(choices_by_player, combination, strict=True)
            ]
            label_parts.extend(f"{player}:sampled-move" for player in deferred_opponents)
            scenarios.append(
                OpponentActionScenario(
                    actions=actions,
                    deferred_action_priors=dict(deferred_action_priors),
                    weight=weight,
                    label=",".join(label_parts),
                )
            )
        scenarios.sort(key=lambda scenario: (-scenario.weight, tuple(sorted(scenario.actions.items()))))
        return _normalize_scenarios(tuple(scenarios[:scenario_count]))

    planner.planner_id = f"checkpoint-top{scenario_count}"  # type: ignore[attr-defined]
    planner.scenario_count = scenario_count  # type: ignore[attr-defined]
    return planner


@dataclass
class PolicyOpponentActionPlanner:
    policies: Mapping[PlayerId, Policy]
    planner_id: str = "policy"

    def __call__(self, context: PolicyContext, rng: random.Random) -> Mapping[PlayerId, int]:
        requested_opponents = tuple(player for player in context.requested_players if player != context.player_id)
        opponent_actions: dict[PlayerId, int] = {}
        for player in requested_opponents:
            policy = self.policies.get(player)
            observation = context.requested_observations.get(player)
            if policy is None or observation is None:
                continue
            opponent_context = PolicyContext(
                player_id=player,
                decision_round_index=context.decision_round_index,
                battle_id=context.battle_id,
                format_id=context.format_id,
                seed=context.seed,
                observation=observation,
                requested_players=context.requested_players,
                trajectory=context.trajectory,
                requested_legal_action_masks=context.requested_legal_action_masks,
                requested_observations=context.requested_observations,
            )
            selector = getattr(policy, "select_action_with_context", None)
            if callable(selector):
                decision = selector(opponent_context, rng=rng)
            else:
                decision = policy.select_action(observation, rng=rng)
            opponent_actions[player] = decision.action_index
        return opponent_actions

    def reset(self) -> None:
        _reset_unique_policies(self.policies)


def policy_opponent_action_planner(
    policies: Mapping[PlayerId, Policy],
    *,
    planner_id: str = "policy",
) -> PolicyOpponentActionPlanner:
    return PolicyOpponentActionPlanner(policies=dict(policies), planner_id=planner_id)


@dataclass
class RootPUCTSearchPolicy:
    """Context-aware policy adapter that selects actions with root-level PUCT.

    The policy's own action search is rooted in the acting player's observation through
    ``PolicyContext``. Simultaneous-turn opponent actions must come from the explicit
    ``opponent_action_planner`` hook. Some planners are hidden-info-safe predictors, while
    benchmark/evaluation planners may intentionally consume the opponent's private observation;
    keep that assumption visible in planner metadata. Branch search runs in a separate env from
    ``env_factory`` so it cannot mutate the live rollout.
    """

    env_factory: Callable[[], PokeZeroEnv]
    rollout_config: RolloutConfig
    value_fn: ObservationValueFunction
    prior_fn: ActionPriorFunction
    policy_id: str = "root-puct-search"
    cpuct: float = 1.25
    opponent_action_planner: OpponentActionPlanner = no_opponent_action_planner
    opponent_action_scenario_planner: OpponentActionScenarioPlanner | None = None
    fallback_policy: Policy = field(default_factory=RandomLegalPolicy)
    allow_fallback: bool = False
    minimum_value_improvement: float | None = None
    minimum_override_prior_ratio: float | None = None
    minimum_score_improvement: float | None = None
    selection_mode: str = "visits"
    root_visit_budget: int | None = ACTION_COUNT + 7
    root_visit_budget_selector: RootVisitBudgetSelector | None = None
    root_time_budget_seconds: float | None = None
    root_prior_temperature: float = 1.0
    # Disabled unless alpha is set. Search evaluation defaults to deterministic priors;
    # Dirichlet noise is an explicitly labeled blind-spot audit arm.
    root_dirichlet_alpha: float | None = None
    root_dirichlet_mix: float = 0.25
    root_dirichlet_seed: int = 0
    max_opponent_action_scenarios: int | None = None
    leaf_rollout_decision_rounds: int = 0
    leaf_rollout_policy_factory: LeafRolloutPolicyFactory | None = None
    start_override_planner: StartOverridePlanner | None = None
    start_override_attempts: int = 1
    # ``None`` delegates the count to a belief-world planner's public-context
    # sampler. Static integer counts preserve the original behavior.
    start_override_samples_per_scenario: int | None = 1
    start_override_hp_fraction_tolerance: float = 0.02
    leaf_rollout_metadata: Mapping[str, object] = field(default_factory=dict)
    # Search is a policy wrapper rather than an alias wrapper. Preserve the
    # underlying raw-policy checkpoint explicitly so benchmark provenance can
    # audit the concrete prior used by every search decision. Keep these at the
    # end to preserve the existing positional constructor contract.
    checkpoint_path: str | None = None
    weights_sha256: str | None = None
    # Optional snapshot source for transformer encode/forward sub-timings.
    # Its counters are cumulative across decisions; ``select_action_with_context``
    # records only the local delta in RootPUCTSearchTiming.
    neural_timing_snapshot: NeuralTimingSnapshot | None = None
    # Optional exact batch evaluator for the mandatory independent root sweep.
    # Keep this new field last so existing positional construction keeps its
    # historical argument layout. Adaptive PUCT revisits remain scalar.
    value_batch_fn: ObservationValueBatchFunction | None = None

    def __post_init__(self) -> None:
        if self.selection_mode not in {"puct", "value", "visits"}:
            raise ValueError("selection_mode must be 'puct', 'value', or 'visits'.")
        if self.minimum_value_improvement is not None and (
            self.minimum_value_improvement < 0.0 or not math.isfinite(self.minimum_value_improvement)
        ):
            raise ValueError("minimum_value_improvement must be a finite non-negative value when set.")
        if self.minimum_override_prior_ratio is not None and (
            self.minimum_override_prior_ratio < 0.0 or not math.isfinite(self.minimum_override_prior_ratio)
        ):
            raise ValueError("minimum_override_prior_ratio must be a finite non-negative value when set.")
        if self.minimum_score_improvement is not None and (
            self.minimum_score_improvement < 0.0 or not math.isfinite(self.minimum_score_improvement)
        ):
            raise ValueError("minimum_score_improvement must be a finite non-negative value when set.")
        if self.root_visit_budget is not None and self.root_visit_budget <= 0:
            raise ValueError("root_visit_budget must be positive when set.")
        if self.root_visit_budget_selector is not None and not callable(self.root_visit_budget_selector):
            raise ValueError("root_visit_budget_selector must be callable when set.")
        if self.value_batch_fn is not None and not callable(self.value_batch_fn):
            raise ValueError("value_batch_fn must be callable when set.")
        if self.neural_timing_snapshot is not None and not callable(self.neural_timing_snapshot):
            raise ValueError("neural_timing_snapshot must be callable when set.")
        if self.root_time_budget_seconds is not None and (
            self.root_time_budget_seconds <= 0.0 or not math.isfinite(self.root_time_budget_seconds)
        ):
            raise ValueError("root_time_budget_seconds must be a finite positive value when set.")
        if self.root_prior_temperature <= 0.0 or not math.isfinite(self.root_prior_temperature):
            raise ValueError("root_prior_temperature must be a finite positive value.")
        if self.root_dirichlet_alpha is not None and (
            self.root_dirichlet_alpha <= 0.0 or not math.isfinite(self.root_dirichlet_alpha)
        ):
            raise ValueError("root_dirichlet_alpha must be a finite positive value when set.")
        if not math.isfinite(self.root_dirichlet_mix) or not 0.0 <= self.root_dirichlet_mix <= 1.0:
            raise ValueError("root_dirichlet_mix must be finite and between 0 and 1.")
        if self.root_dirichlet_alpha is not None and self.root_dirichlet_mix == 0.0:
            raise ValueError("root_dirichlet_mix must be positive when root_dirichlet_alpha is set.")
        if isinstance(self.root_dirichlet_seed, bool) or not isinstance(self.root_dirichlet_seed, int):
            raise ValueError("root_dirichlet_seed must be an integer.")
        if self.root_dirichlet_alpha is not None and not self.policy_id.endswith("+dirichlet"):
            self.policy_id = f"{self.policy_id}+dirichlet"
        if self.max_opponent_action_scenarios is not None and self.max_opponent_action_scenarios <= 0:
            raise ValueError("max_opponent_action_scenarios must be positive when set.")
        if self.leaf_rollout_decision_rounds < 0:
            raise ValueError("leaf_rollout_decision_rounds must be non-negative.")
        if self.leaf_rollout_decision_rounds and self.leaf_rollout_policy_factory is None:
            raise ValueError("leaf_rollout_policy_factory is required when leaf rollouts are enabled.")
        if self.start_override_attempts <= 0:
            raise ValueError("start_override_attempts must be positive.")
        if self.start_override_samples_per_scenario is not None:
            if self.start_override_samples_per_scenario <= 0:
                raise ValueError("start_override_samples_per_scenario must be positive when set.")
            if self.start_override_samples_per_scenario > 1 and self.start_override_planner is None:
                raise ValueError("start_override_samples_per_scenario requires start_override_planner.")
        elif self.start_override_planner is None or not callable(
            getattr(self.start_override_planner, "sample_count_for_context", None)
        ):
            raise ValueError(
                "dynamic start_override_samples_per_scenario requires a planner with sample_count_for_context."
            )
        if self.start_override_hp_fraction_tolerance < 0.0 or not math.isfinite(
            self.start_override_hp_fraction_tolerance
        ):
            raise ValueError("start_override_hp_fraction_tolerance must be a finite non-negative value.")

    def reset(self) -> None:
        reset = getattr(self.fallback_policy, "reset", None)
        if callable(reset):
            reset()
        planner_reset = getattr(self.opponent_action_planner, "reset", None)
        if callable(planner_reset):
            planner_reset()
        scenario_planner_reset = getattr(self.opponent_action_scenario_planner, "reset", None)
        if callable(scenario_planner_reset):
            scenario_planner_reset()

    def select_action(
        self,
        observation: PokeZeroObservationV0,
        *,
        rng: random.Random,
    ) -> PolicyDecision:
        timing_started_at = _timing_perf_counter()
        neural_timing_before = _neural_timing_snapshot(self.neural_timing_snapshot)
        decision = self.fallback_policy.select_action(observation, rng=rng)
        return PolicyDecision(
            action_index=decision.action_index,
            policy_id=self.policy_id,
            action_probability=decision.action_probability,
            value_estimate=decision.value_estimate,
            metadata={
                **dict(decision.metadata),
                "policy_family": "root-puct-search",
                "root_puct_fallback": True,
                "root_puct_fallback_reason": "missing policy context",
                "root_puct_fallback_category": root_puct_fallback_category("missing policy context"),
                "fallback_policy_id": decision.policy_id,
                **self._fallback_timing_metadata(
                    timing_started_at=timing_started_at,
                    neural_timing_before=neural_timing_before,
                ),
            },
        )

    def select_action_with_context(
        self,
        context: PolicyContext,
        *,
        rng: random.Random,
    ) -> PolicyDecision:
        timing_started_at = _timing_perf_counter()
        neural_timing_before = _neural_timing_snapshot(self.neural_timing_snapshot)
        if context.player_id not in context.requested_players:
            return self._fallback(
                context,
                rng=rng,
                reason="player is not requested",
                timing_started_at=timing_started_at,
                neural_timing_before=neural_timing_before,
            )
        opponent_scenario_planning_started_at = _timing_perf_counter()
        try:
            opponent_scenarios = _opponent_action_scenarios(self, context, rng)
        except ValueError as exc:
            return self._fallback(
                context,
                rng=rng,
                reason=str(exc),
                timing_started_at=timing_started_at,
                neural_timing_before=neural_timing_before,
                opponent_scenario_planning_seconds=(
                    _timing_perf_counter() - opponent_scenario_planning_started_at
                ),
            )
        opponent_scenario_planning_seconds = _timing_perf_counter() - opponent_scenario_planning_started_at
        root_policy_setup_seconds = 0.0
        root_policy_setup_started_at = _timing_perf_counter()

        def current_root_policy_setup_seconds() -> float:
            return root_policy_setup_seconds + (
                _timing_perf_counter() - root_policy_setup_started_at
            )

        legality_checked = False
        for scenario in opponent_scenarios:
            planner_error = _opponent_action_planner_error(
                player_id=context.player_id,
                requested_players=context.requested_players,
                opponent_actions=scenario.actions,
            )
            if planner_error is not None:
                return self._fallback(
                    context,
                    rng=rng,
                    reason=planner_error,
                    timing_started_at=timing_started_at,
                    neural_timing_before=neural_timing_before,
                    opponent_scenario_planning_seconds=opponent_scenario_planning_seconds,
                    root_policy_setup_seconds=current_root_policy_setup_seconds(),
                )
            deferred_error = _deferred_opponent_action_planner_error(
                context=context,
                deferred_actions=scenario.deferred_actions,
                deferred_action_priors=scenario.deferred_action_priors,
            )
            if deferred_error is not None:
                return self._fallback(
                    context,
                    rng=rng,
                    reason=deferred_error,
                    timing_started_at=timing_started_at,
                    neural_timing_before=neural_timing_before,
                    opponent_scenario_planning_seconds=opponent_scenario_planning_seconds,
                    root_policy_setup_seconds=current_root_policy_setup_seconds(),
                )
            legality_report = _opponent_action_legality_report(context, scenario.actions)
            legality_checked = legality_checked or legality_report.checked
            if legality_report.error is not None:
                return self._fallback(
                    context,
                    rng=rng,
                    reason=legality_report.error,
                    timing_started_at=timing_started_at,
                    neural_timing_before=neural_timing_before,
                    opponent_scenario_planning_seconds=opponent_scenario_planning_seconds,
                    root_policy_setup_seconds=current_root_policy_setup_seconds(),
                )
        try:
            start_override_samples_per_scenario = _start_override_samples_per_scenario(self, context)
            start_override_sampling_metadata = _start_override_sampling_metadata(self, context)
        except ValueError as exc:
            return self._fallback(
                context,
                rng=rng,
                reason=str(exc),
                timing_started_at=timing_started_at,
                neural_timing_before=neural_timing_before,
                opponent_scenario_planning_seconds=opponent_scenario_planning_seconds,
                root_policy_setup_seconds=current_root_policy_setup_seconds(),
            )
        search_scenario_groups = _start_override_sampled_scenario_groups(
            opponent_scenarios,
            samples_per_scenario=(
                start_override_samples_per_scenario
                if self.start_override_planner is not None
                else 1
            ),
        )
        search_scenarios = _flatten_scenario_groups(search_scenario_groups)

        search_trajectory = _trajectory_with_current_observation(context)
        history = player_observation_history(
            search_trajectory,
            player_id=context.player_id,
            through_decision_round=context.decision_round_index,
        )
        root_policy_setup_seconds += _timing_perf_counter() - root_policy_setup_started_at
        policy_evaluation_started_at = _timing_perf_counter()
        base_priors = _temperature_scale_action_priors(
            self.prior_fn(history),
            temperature=self.root_prior_temperature,
        )
        policy_evaluation_seconds = _timing_perf_counter() - policy_evaluation_started_at
        root_policy_setup_started_at = _timing_perf_counter()
        # Successful scenario branches are available even if a later branch triggers
        # a graceful fallback. Preserve their measured stage timing in that result.
        completed_search_timings: list[RootPUCTSearchTiming] = []
        priors, root_dirichlet_metadata = _root_dirichlet_action_priors(
            base_priors,
            context=context,
            legal_action_mask=context.observation.legal_action_mask,
            alpha=self.root_dirichlet_alpha,
            mix=self.root_dirichlet_mix,
            base_seed=self.root_dirichlet_seed,
        )
        leaf_rollout_policies = (
            _leaf_rollout_policies(context, self.leaf_rollout_policy_factory)
            if self.leaf_rollout_decision_rounds
            else None
        )
        # The capstone's wall budget is defined at policy dispatch. Count opponent
        # planning and prior evaluation before replay work spends its remaining budget.
        search_started_at = perf_counter()
        env = self.env_factory()
        root_policy_setup_seconds += _timing_perf_counter() - root_policy_setup_started_at
        prepared_prefixes_to_release: list[PreparedReplayPrefix] = []
        skipped_scenarios: list[tuple[OpponentActionScenario, str]] = []
        inner_fallback_decision: PolicyDecision | None = None
        try:
            try:
                scenario_search_pairs: list[tuple[OpponentActionScenario, PUCTBranchSearchResult]] = []
                start_override_sources_used = 0
                start_override_attempts_used = 0
                unsearched_scenario_count = 0
                unsearched_action_group_count = 0
                skipped_action_group_count = 0
                searched_action_group_count = 0
                flat_scenario_index = 0
                belief_world_materialization_seconds = 0.0
                belief_world_materialization_count = 0
                shared_start_override_samples = None
                direct_materialization_count = 0
                replay_materialization_count = 0
                direct_materialization_rejection_categories: Counter[str] = Counter()
                direct_materialization_observation_mismatch_paths: Counter[str] = Counter()
                direct_prefix_construction_seconds = 0.0
                direct_prefix_construction_count = 0
                puct_search_call_seconds = 0.0
                puct_search_call_count = 0
                scenario_dispatch_started_at: float | None = None

                def current_scenario_dispatch_orchestration_seconds() -> float:
                    if scenario_dispatch_started_at is None:
                        return 0.0
                    return max(
                        0.0,
                        _timing_perf_counter()
                        - scenario_dispatch_started_at
                        - puct_search_call_seconds
                        - direct_prefix_construction_seconds,
                    )

                def record_direct_materialization_rejection(category: str) -> None:
                    direct_materialization_rejection_categories[category] += 1

                def record_direct_materialization_observation_mismatch_path(path: str) -> None:
                    direct_materialization_observation_mismatch_paths[path] += 1
                # Count only worlds that supplied a completed branch search. A
                # prepared shared world can serve several action scenarios, so
                # its identity also prevents duplicate credits.
                used_shared_sample_indices: set[int] = set()

                def record_materialization_usage(
                    prepared_prefix: PreparedReplayPrefix | None,
                    search: PUCTBranchSearchResult,
                    *,
                    shared_sample_index: int | None,
                ) -> None:
                    nonlocal direct_materialization_count, replay_materialization_count
                    if prepared_prefix is not None:
                        if shared_sample_index is not None:
                            if shared_sample_index in used_shared_sample_indices:
                                return
                            used_shared_sample_indices.add(shared_sample_index)
                        if prepared_prefix.materialization_mode == "direct":
                            direct_materialization_count += 1
                        elif prepared_prefix.materialization_mode == "replay":
                            replay_materialization_count += 1
                        else:
                            raise ValueError(
                                "prepared start override has an unsupported materialization mode."
                            )
                    elif search.timing.prefix_replay_count:
                        replay_materialization_count += 1

                if _uses_scenario_independent_start_overrides(self, context):
                    def record_belief_world_materialization_attempt() -> None:
                        nonlocal belief_world_materialization_count
                        belief_world_materialization_count += 1

                    belief_world_materialization_started_at = _timing_perf_counter()
                    try:
                        shared_start_override_samples = _shared_start_override_samples(
                            env=env,
                            policy=self,
                            context=context,
                            rng=rng,
                            sample_scenarios=search_scenario_groups[0].samples,
                            search_trajectory=search_trajectory,
                            on_attempt=record_belief_world_materialization_attempt,
                            on_direct_materialization_unavailable=record_direct_materialization_rejection,
                            on_direct_materialization_observation_mismatch_path=(
                                record_direct_materialization_observation_mismatch_path
                            ),
                        )
                        prepared_prefixes_to_release.extend(
                            prepared_prefix
                            for prepared_prefix in shared_start_override_samples.prepared_prefixes
                            if prepared_prefix is not None
                        )
                    finally:
                        belief_world_materialization_seconds = (
                            _timing_perf_counter() - belief_world_materialization_started_at
                        )
                if shared_start_override_samples is not None:
                    start_override_attempts_used += shared_start_override_samples.attempts_used
                scenario_dispatch_started_at = _timing_perf_counter()

                for group_index, scenario_group in enumerate(search_scenario_groups):
                    group_search_pairs: list[tuple[OpponentActionScenario, PUCTBranchSearchResult]] = []
                    for sample_index, scenario in enumerate(scenario_group.samples):
                        scenario_index = flat_scenario_index
                        flat_scenario_index += 1
                        scenario_search: PUCTBranchSearchResult | None = None
                        scenario_start_override: StartOverrideSource = None
                        replay_rejection_reasons: list[str] = []
                        if shared_start_override_samples is not None:
                            start_override = shared_start_override_samples.overrides[sample_index]
                            prepared_prefix = shared_start_override_samples.prepared_prefixes[sample_index]
                            if start_override is None:
                                replay_rejection_reasons.append(
                                    shared_start_override_samples.rejection_reasons[sample_index]
                                    or "start override planner did not produce a sampled world"
                                )
                                skipped_scenarios.append(
                                    (
                                        scenario,
                                        _format_replay_rejection_reasons(replay_rejection_reasons),
                                    )
                                )
                                continue
                            attempts = 1
                        else:
                            attempts = self.start_override_attempts if self.start_override_planner is not None else 1
                            start_override = None
                            prepared_prefix = None
                        for _attempt_index in range(attempts):
                            if shared_start_override_samples is None:
                                start_override_attempts_used += 1
                                # A retry receives a new belief world. Never carry a
                                # snapshot prepared for a prior sampled world into it.
                                prepared_prefix = None
                                start_override = (
                                    None
                                    if self.start_override_planner is None
                                    else self.start_override_planner(
                                        context,
                                        scenario,
                                        scenario_index,
                                        rng,
                                    )
                                )
                            if (
                                shared_start_override_samples is None
                                and self.start_override_planner is not None
                                and start_override is None
                            ):
                                replay_rejection_reasons.append(
                                    "start override planner did not produce a sampled world"
                                )
                                continue
                            scenario_root_time_budget_seconds = _remaining_root_time_budget_seconds(
                                total_budget_seconds=self.root_time_budget_seconds,
                                started_at=timing_started_at,
                            )
                            visit_budget_resolver: RootVisitBudgetResolver | None = None
                            if self.root_visit_budget_selector is not None:

                                def visit_budget_resolver(
                                    budget_context: RootPUCTVisitBudgetContext,
                                ) -> int | None:
                                    return self.root_visit_budget_selector(context, budget_context)

                            try:
                                if start_override is not None and prepared_prefix is None:
                                    direct_prefix_started_at = _timing_perf_counter()
                                    try:
                                        prepared_prefix = prepare_direct_materialization_prefix(
                                            env=env,
                                            trajectory=search_trajectory,
                                            player_id=context.player_id,
                                            prefix_decision_round_count=context.decision_round_index,
                                            start_override=start_override,
                                            public_materialization_state=context.public_materialization_state,
                                            deferred_opponent_actions=scenario.deferred_actions,
                                            deferred_opponent_action_priors=(
                                                scenario.deferred_action_priors
                                            ),
                                            expected_current_observation=context.observation,
                                            replay_hp_fraction_tolerance=(
                                                self.start_override_hp_fraction_tolerance
                                            ),
                                            on_unavailable=record_direct_materialization_rejection,
                                            on_observation_mismatch_path=(
                                                record_direct_materialization_observation_mismatch_path
                                            ),
                                        )
                                    finally:
                                        direct_prefix_construction_seconds += (
                                            _timing_perf_counter() - direct_prefix_started_at
                                        )
                                        direct_prefix_construction_count += 1
                                    if prepared_prefix is not None:
                                        prepared_prefixes_to_release.append(prepared_prefix)
                                puct_search_call_count += 1
                                puct_search_started_at = _timing_perf_counter()
                                try:
                                    search = puct_branch_search(
                                        env=env,
                                        trajectory=search_trajectory,
                                        player_id=context.player_id,
                                        prefix_decision_round_count=context.decision_round_index,
                                        legal_action_mask=context.observation.legal_action_mask,
                                        opponent_actions=scenario.actions,
                                        value_fn=self.value_fn,
                                        value_batch_fn=self.value_batch_fn,
                                        action_priors=priors,
                                        cpuct=self.cpuct,
                                        leaf_rollout_policies=leaf_rollout_policies,
                                        leaf_rollout_config=self.rollout_config,
                                        leaf_rollout_decision_rounds=self.leaf_rollout_decision_rounds,
                                        root_visit_budget=self.root_visit_budget,
                                        root_visit_budget_resolver=visit_budget_resolver,
                                        budget_action_priors=base_priors,
                                        root_time_budget_seconds=scenario_root_time_budget_seconds,
                                        start_override=start_override,
                                        expected_current_observation=context.observation,
                                        replay_hp_fraction_tolerance=(
                                            self.start_override_hp_fraction_tolerance
                                            if start_override is not None
                                            else 0.0
                                        ),
                                        prepared_prefix=prepared_prefix,
                                    )
                                finally:
                                    puct_search_call_seconds += (
                                        _timing_perf_counter() - puct_search_started_at
                                    )
                            except ValueError as exc:
                                reason = _opponent_scenario_replay_legality_error(exc, scenario)
                                if reason is None:
                                    raise
                                replay_rejection_reasons.append(reason)
                                if start_override is None:
                                    break
                            else:
                                record_materialization_usage(
                                    prepared_prefix,
                                    search,
                                    shared_sample_index=(
                                        sample_index
                                        if shared_start_override_samples is not None
                                        else None
                                    ),
                                )
                                scenario_search = search
                                scenario_start_override = start_override
                                completed_search_timings.append(search.timing)
                                break
                        if scenario_search is None:
                            skipped_scenarios.append(
                                (
                                    scenario,
                                    _format_replay_rejection_reasons(replay_rejection_reasons),
                                )
                            )
                        else:
                            group_search_pairs.append((scenario, scenario_search))
                            if scenario_start_override is not None:
                                start_override_sources_used += 1
                    if not group_search_pairs:
                        skipped_action_group_count += 1
                        continue
                    searched_action_group_count += 1
                    group_sample_weight = scenario_group.root.weight / len(group_search_pairs)
                    action_group_cap_reached = False
                    for scenario, scenario_search in group_search_pairs:
                        scenario_search_pairs.append(
                            (
                                replace(scenario, weight=group_sample_weight),
                                scenario_search,
                            )
                        )
                        if (
                            self.max_opponent_action_scenarios is not None
                            and searched_action_group_count >= self.max_opponent_action_scenarios
                        ):
                            remaining_groups = search_scenario_groups[group_index + 1 :]
                            unsearched_scenario_count = sum(
                                len(group.samples) for group in remaining_groups
                            )
                            unsearched_action_group_count = len(remaining_groups)
                            action_group_cap_reached = True
                            break
                    if action_group_cap_reached:
                        break
                if not scenario_search_pairs:
                    details = "; ".join(reason for _scenario, reason in skipped_scenarios) or "none"
                    metadata = _opponent_scenario_skip_metadata(
                        opponent_scenarios=search_scenarios,
                        used_scenarios=(),
                        skipped_scenarios=tuple(skipped_scenarios),
                        unsearched_scenario_count=0,
                        opponent_action_group_count=len(search_scenario_groups),
                        used_action_group_count=0,
                        skipped_action_group_count=skipped_action_group_count,
                        unsearched_action_group_count=0,
                    )
                    if self.start_override_planner is not None:
                        metadata.update(
                            {
                                "root_puct_start_override_sources_used": 0,
                                "root_puct_start_override_attempts": self.start_override_attempts,
                                "root_puct_start_override_attempts_used": start_override_attempts_used,
                                "root_puct_start_override_samples_per_scenario": start_override_samples_per_scenario,
                                "root_puct_start_override_hp_fraction_tolerance": (
                                    self.start_override_hp_fraction_tolerance
                                ),
                                "root_puct_start_override_direct_materializations": (
                                    direct_materialization_count
                                ),
                                "root_puct_start_override_replay_materializations": (
                                    replay_materialization_count
                                ),
                                **_direct_materialization_rejection_metadata(
                                    direct_materialization_rejection_categories,
                                    direct_materialization_observation_mismatch_paths,
                                ),
                                **_shared_start_override_metadata(shared_start_override_samples),
                                **start_override_sampling_metadata,
                            }
                        )
                    raise _AllOpponentScenariosReplayIllegal(
                        details=details,
                        metadata=metadata,
                    )
                used_scenarios = _normalize_scenarios(tuple(scenario for scenario, _search in scenario_search_pairs))
                scenario_searches = tuple(search for _scenario, search in scenario_search_pairs)
            except _AllOpponentScenariosReplayIllegal as exc:
                inner_fallback_decision = self._fallback(
                    context,
                    rng=rng,
                    reason=str(exc),
                    extra_metadata=exc.metadata,
                    timing_started_at=timing_started_at,
                    neural_timing_before=neural_timing_before,
                    opponent_scenario_planning_seconds=opponent_scenario_planning_seconds,
                    policy_evaluation_seconds=policy_evaluation_seconds,
                    belief_world_materialization_seconds=belief_world_materialization_seconds,
                    belief_world_materialization_count=belief_world_materialization_count,
                    root_policy_setup_seconds=root_policy_setup_seconds,
                    direct_prefix_construction_seconds=direct_prefix_construction_seconds,
                    direct_prefix_construction_count=direct_prefix_construction_count,
                    scenario_dispatch_orchestration_seconds=(
                        current_scenario_dispatch_orchestration_seconds()
                    ),
                    scenario_dispatch_orchestration_count=flat_scenario_index,
                    completed_search_timing=RootPUCTSearchTiming.aggregate(completed_search_timings),
                    puct_search_call_seconds=puct_search_call_seconds,
                    puct_search_call_count=puct_search_call_count,
                    puct_search_result_count=len(completed_search_timings),
                )
                return inner_fallback_decision
            except Exception as exc:
                materialization_metadata = (
                    {
                        "root_puct_start_override_direct_materializations": (
                            direct_materialization_count
                        ),
                        "root_puct_start_override_replay_materializations": (
                            replay_materialization_count
                        ),
                        **_direct_materialization_rejection_metadata(
                            direct_materialization_rejection_categories,
                            direct_materialization_observation_mismatch_paths,
                        ),
                    }
                    if self.start_override_planner is not None
                    else None
                )
                inner_fallback_decision = self._fallback(
                    context,
                    rng=rng,
                    reason=f"search failed: {exc}",
                    extra_metadata=materialization_metadata,
                    timing_started_at=timing_started_at,
                    neural_timing_before=neural_timing_before,
                    opponent_scenario_planning_seconds=opponent_scenario_planning_seconds,
                    policy_evaluation_seconds=policy_evaluation_seconds,
                    belief_world_materialization_seconds=belief_world_materialization_seconds,
                    belief_world_materialization_count=belief_world_materialization_count,
                    root_policy_setup_seconds=root_policy_setup_seconds,
                    direct_prefix_construction_seconds=direct_prefix_construction_seconds,
                    direct_prefix_construction_count=direct_prefix_construction_count,
                    scenario_dispatch_orchestration_seconds=(
                        current_scenario_dispatch_orchestration_seconds()
                    ),
                    scenario_dispatch_orchestration_count=flat_scenario_index,
                    completed_search_timing=RootPUCTSearchTiming.aggregate(completed_search_timings),
                    puct_search_call_seconds=puct_search_call_seconds,
                    puct_search_call_count=puct_search_call_count,
                    puct_search_result_count=len(completed_search_timings),
                )
                return inner_fallback_decision
            elapsed_seconds = perf_counter() - search_started_at
            scenario_dispatch_orchestration_seconds = current_scenario_dispatch_orchestration_seconds()
            completed_search_timing = RootPUCTSearchTiming.aggregate(
                tuple(scenario_search.timing for scenario_search in scenario_searches)
            )
        finally:
            try:
                for prepared_prefix in prepared_prefixes_to_release:
                    try:
                        release_prepared_replay_prefix(env, prepared_prefix)
                    except Exception:
                        # Close below still clears the bridge cache; cleanup must not mask a
                        # completed search result or a more relevant search failure.
                        pass
            finally:
                close = getattr(env, "close", None)
                if callable(close):
                    close()
            if inner_fallback_decision is not None:
                _refresh_root_puct_timing_total(
                    inner_fallback_decision.metadata,
                    total_seconds=_timing_perf_counter() - timing_started_at,
                )

        search = _aggregate_scenario_searches(
            scenario_searches,
            opponent_scenarios=used_scenarios,
            cpuct=self.cpuct,
        )
        raw_total_visits = sum(scenario_search.total_visits for scenario_search in scenario_searches)
        search_best = _selected_candidate(search, mode=self.selection_mode)
        prior_best = _best_prior_candidate(search.candidates)
        best = search_best
        gate_metadata = {}
        if self.minimum_value_improvement is not None:
            value_gate_used = False
            if (
                search_best.action_index != prior_best.action_index
                and search_best.value < prior_best.value + self.minimum_value_improvement
            ):
                best = prior_best
                value_gate_used = True
            gate_metadata = {
                "root_puct_minimum_value_improvement": self.minimum_value_improvement,
                "root_puct_value_gate_used": value_gate_used,
                "root_puct_pre_gate_action": search_best.action_index,
            }
        prior_ratio_metadata = {}
        if self.minimum_override_prior_ratio is not None:
            prior_ratio_gate_used = False
            required_prior = prior_best.prior * self.minimum_override_prior_ratio
            if best.action_index != prior_best.action_index and best.prior < required_prior:
                best = prior_best
                prior_ratio_gate_used = True
            prior_ratio_metadata = {
                "root_puct_minimum_override_prior_ratio": self.minimum_override_prior_ratio,
                "root_puct_prior_ratio_gate_used": prior_ratio_gate_used,
                "root_puct_prior_ratio_gate_required_prior": required_prior,
            }
        score_gate_metadata = {}
        if self.minimum_score_improvement is not None:
            score_gate_used = False
            # This compares root-PUCT score (Q + exploration bonus), not pure leaf value.
            required_score = prior_best.score + self.minimum_score_improvement
            if best.action_index != prior_best.action_index and best.score < required_score:
                best = prior_best
                score_gate_used = True
            score_gate_metadata = {
                "root_puct_minimum_score_improvement": self.minimum_score_improvement,
                "root_puct_score_gate_used": score_gate_used,
                "root_puct_score_gate_required_score": required_score,
            }
        leaf_metadata = _leaf_rollout_metadata(
            scenario_searches,
            configured_rounds=self.leaf_rollout_decision_rounds,
        )
        planner = self.opponent_action_scenario_planner or self.opponent_action_planner
        planner_id = getattr(planner, "planner_id", None)
        planner_metadata = (
            {"root_puct_opponent_action_policy": str(planner_id)}
            if planner_id is not None
            else {}
        )
        visit_metadata: dict[str, int] = {"root_puct_total_visits": raw_total_visits}
        if search.total_visits != raw_total_visits:
            visit_metadata["root_puct_effective_total_visits"] = search.total_visits
        budget_metadata: dict[str, object] = {}
        if self.root_time_budget_seconds is not None:
            budget_metadata.update(
                {
                    "root_puct_root_time_budget_seconds": self.root_time_budget_seconds,
                    "root_puct_root_scenario_time_budget_seconds": scenario_searches[0].root_time_budget_seconds,
                    "root_puct_time_budget_exhausted": any(
                        search.time_budget_exhausted for search in scenario_searches
                    ),
                }
            )
        if self.root_visit_budget_selector is not None:
            budget_contexts = tuple(
                search.visit_budget_context
                for search in scenario_searches
                if search.visit_budget_context is not None
            )
            budget_metadata.update(
                {
                    "root_puct_root_visit_budget_mode": "adaptive",
                    "root_puct_root_visit_budget_selector": getattr(
                        self.root_visit_budget_selector,
                        "selector_id",
                        "custom",
                    ),
                    "root_puct_configured_root_visit_budget": self.root_visit_budget,
                    "root_puct_effective_root_visit_budgets": tuple(
                        search.root_visit_budget for search in scenario_searches
                    ),
                    "root_puct_visit_budget_contexts": tuple(
                        budget_context.to_dict() for budget_context in budget_contexts
                    ),
                    "root_puct_root_visit_budget_selector_config": _root_visit_budget_selector_config(
                        self.root_visit_budget_selector
                    ),
                }
            )
        start_override_metadata = (
            {
                "root_puct_start_override_sources_used": start_override_sources_used,
                "root_puct_start_override_attempts": self.start_override_attempts,
                "root_puct_start_override_attempts_used": start_override_attempts_used,
                "root_puct_start_override_samples_per_scenario": start_override_samples_per_scenario,
                "root_puct_start_override_hp_fraction_tolerance": (
                    self.start_override_hp_fraction_tolerance
                ),
                "root_puct_start_override_direct_materializations": direct_materialization_count,
                "root_puct_start_override_replay_materializations": replay_materialization_count,
                **_direct_materialization_rejection_metadata(
                    direct_materialization_rejection_categories,
                    direct_materialization_observation_mismatch_paths,
                ),
                **_shared_start_override_metadata(shared_start_override_samples),
                **start_override_sampling_metadata,
            }
            if self.start_override_planner is not None
            else {}
        )
        timing = _finalize_root_puct_timing(
            completed_search_timing=completed_search_timing,
            puct_search_call_seconds=puct_search_call_seconds,
            puct_search_call_count=puct_search_call_count,
            puct_search_result_count=len(scenario_searches),
            belief_world_materialization_seconds=belief_world_materialization_seconds,
            belief_world_materialization_count=belief_world_materialization_count,
            opponent_scenario_planning_seconds=opponent_scenario_planning_seconds,
            root_policy_setup_seconds=root_policy_setup_seconds,
            direct_prefix_construction_seconds=direct_prefix_construction_seconds,
            direct_prefix_construction_count=direct_prefix_construction_count,
            scenario_dispatch_orchestration_seconds=scenario_dispatch_orchestration_seconds,
            scenario_dispatch_orchestration_count=flat_scenario_index,
            policy_evaluation_seconds=policy_evaluation_seconds,
            timing_started_at=timing_started_at,
            neural_timing_before=neural_timing_before,
            neural_timing_snapshot=self.neural_timing_snapshot,
        )
        return PolicyDecision(
            action_index=best.action_index,
            policy_id=self.policy_id,
            action_probability=None,
            metadata={
                "policy_family": "root-puct-search",
                "root_puct_fallback": False,
                "root_puct_cpuct": self.cpuct,
                "root_puct_selection_mode": self.selection_mode,
                "root_puct_root_prior_temperature": self.root_prior_temperature,
                **root_dirichlet_metadata,
                "root_puct_selected_value": best.value,
                "root_puct_selected_score": best.score,
                "root_puct_selected_action_prior": best.prior,
                "root_puct_selected_action_visits": best.visits,
                "root_puct_search_action": search_best.action_index,
                "root_puct_search_action_value": search_best.value,
                "root_puct_search_action_score": search_best.score,
                "root_puct_search_action_prior": search_best.prior,
                "root_puct_search_action_visits": search_best.visits,
                "root_puct_prior_action": prior_best.action_index,
                "root_puct_prior_value": prior_best.value,
                "root_puct_prior_score": prior_best.score,
                "root_puct_prior_action_prior": prior_best.prior,
                "root_puct_prior_action_visits": prior_best.visits,
                "root_puct_selected_changed_prior_action": best.action_index != prior_best.action_index,
                "root_puct_pre_gate_changed_prior_action": search_best.action_index != prior_best.action_index,
                "root_puct_candidate_count": len(search.candidates),
                "root_puct_elapsed_seconds": elapsed_seconds,
                "root_puct_timing": timing.to_dict(),
                "root_puct_opponent_actions": dict(used_scenarios[0].actions),
                "root_puct_opponent_action_scenario_count": len(used_scenarios),
                **_opponent_scenario_skip_metadata(
                    opponent_scenarios=search_scenarios,
                    used_scenarios=used_scenarios,
                    skipped_scenarios=tuple(skipped_scenarios),
                    unsearched_scenario_count=unsearched_scenario_count,
                    opponent_action_group_count=len(search_scenario_groups),
                    used_action_group_count=searched_action_group_count,
                    skipped_action_group_count=skipped_action_group_count,
                    unsearched_action_group_count=unsearched_action_group_count,
                ),
                "root_puct_max_opponent_action_scenarios": self.max_opponent_action_scenarios,
                "root_puct_opponent_actions_legality_checked": legality_checked,
                **planner_metadata,
                **gate_metadata,
                **prior_ratio_metadata,
                **score_gate_metadata,
                **visit_metadata,
                **budget_metadata,
                **start_override_metadata,
                **dict(self.leaf_rollout_metadata),
                **leaf_metadata,
            },
        )

    def _fallback(
        self,
        context: PolicyContext,
        *,
        rng: random.Random,
        reason: str,
        extra_metadata: Mapping[str, object] | None = None,
        timing_started_at: float | None = None,
        neural_timing_before: Mapping[str, float | int] | None = None,
        opponent_scenario_planning_seconds: float | None = None,
        policy_evaluation_seconds: float | None = None,
        belief_world_materialization_seconds: float | None = None,
        belief_world_materialization_count: int | None = None,
        root_policy_setup_seconds: float | None = None,
        direct_prefix_construction_seconds: float | None = None,
        direct_prefix_construction_count: int | None = None,
        scenario_dispatch_orchestration_seconds: float | None = None,
        scenario_dispatch_orchestration_count: int | None = None,
        completed_search_timing: RootPUCTSearchTiming | None = None,
        puct_search_call_seconds: float | None = None,
        puct_search_call_count: int | None = None,
        puct_search_result_count: int | None = None,
    ) -> PolicyDecision:
        if not self.allow_fallback:
            raise ValueError(f"root PUCT search cannot select an action: {reason}")
        decision = self.fallback_policy.select_action(context.observation, rng=rng)
        category = root_puct_fallback_category(reason)
        return PolicyDecision(
            action_index=decision.action_index,
            policy_id=self.policy_id,
            action_probability=decision.action_probability,
            value_estimate=decision.value_estimate,
            metadata={
                **dict(decision.metadata),
                "policy_family": "root-puct-search",
                "root_puct_fallback": True,
                "root_puct_fallback_reason": reason,
                "root_puct_fallback_category": category,
                "fallback_policy_id": decision.policy_id,
                **self._fallback_timing_metadata(
                    timing_started_at=timing_started_at,
                    neural_timing_before=neural_timing_before,
                    opponent_scenario_planning_seconds=opponent_scenario_planning_seconds,
                    policy_evaluation_seconds=policy_evaluation_seconds,
                    belief_world_materialization_seconds=belief_world_materialization_seconds,
                    belief_world_materialization_count=belief_world_materialization_count,
                    root_policy_setup_seconds=root_policy_setup_seconds,
                    direct_prefix_construction_seconds=direct_prefix_construction_seconds,
                    direct_prefix_construction_count=direct_prefix_construction_count,
                    scenario_dispatch_orchestration_seconds=(
                        scenario_dispatch_orchestration_seconds
                    ),
                    scenario_dispatch_orchestration_count=(
                        scenario_dispatch_orchestration_count
                    ),
                    completed_search_timing=completed_search_timing,
                    puct_search_call_seconds=puct_search_call_seconds,
                    puct_search_call_count=puct_search_call_count,
                    puct_search_result_count=puct_search_result_count,
                ),
                **dict(extra_metadata or {}),
            },
        )

    def _fallback_timing_metadata(
        self,
        *,
        timing_started_at: float | None,
        neural_timing_before: Mapping[str, float | int] | None,
        opponent_scenario_planning_seconds: float | None = None,
        policy_evaluation_seconds: float | None = None,
        belief_world_materialization_seconds: float | None = None,
        belief_world_materialization_count: int | None = None,
        root_policy_setup_seconds: float | None = None,
        direct_prefix_construction_seconds: float | None = None,
        direct_prefix_construction_count: int | None = None,
        scenario_dispatch_orchestration_seconds: float | None = None,
        scenario_dispatch_orchestration_count: int | None = None,
        completed_search_timing: RootPUCTSearchTiming | None = None,
        puct_search_call_seconds: float | None = None,
        puct_search_call_count: int | None = None,
        puct_search_result_count: int | None = None,
    ) -> dict[str, object]:
        """Attach all work done before a graceful fallback to the decision artifact."""

        if timing_started_at is None:
            return {}
        timing = _finalize_root_puct_timing(
            completed_search_timing=completed_search_timing,
            puct_search_call_seconds=puct_search_call_seconds,
            puct_search_call_count=puct_search_call_count,
            puct_search_result_count=puct_search_result_count,
            belief_world_materialization_seconds=belief_world_materialization_seconds,
            belief_world_materialization_count=belief_world_materialization_count,
            opponent_scenario_planning_seconds=opponent_scenario_planning_seconds,
            root_policy_setup_seconds=root_policy_setup_seconds,
            direct_prefix_construction_seconds=direct_prefix_construction_seconds,
            direct_prefix_construction_count=direct_prefix_construction_count,
            scenario_dispatch_orchestration_seconds=scenario_dispatch_orchestration_seconds,
            scenario_dispatch_orchestration_count=scenario_dispatch_orchestration_count,
            policy_evaluation_seconds=policy_evaluation_seconds,
            timing_started_at=timing_started_at,
            neural_timing_before=neural_timing_before,
            neural_timing_snapshot=self.neural_timing_snapshot,
        )
        return {
            "root_puct_elapsed_seconds": timing.total_seconds,
            "root_puct_timing": timing.to_dict(),
        }


def _finalize_root_puct_timing(
    *,
    completed_search_timing: RootPUCTSearchTiming | None,
    puct_search_call_seconds: float | None,
    puct_search_call_count: int | None,
    puct_search_result_count: int | None,
    belief_world_materialization_seconds: float | None,
    belief_world_materialization_count: int | None,
    opponent_scenario_planning_seconds: float | None,
    root_policy_setup_seconds: float | None,
    direct_prefix_construction_seconds: float | None,
    direct_prefix_construction_count: int | None,
    scenario_dispatch_orchestration_seconds: float | None,
    scenario_dispatch_orchestration_count: int | None,
    policy_evaluation_seconds: float | None,
    timing_started_at: float,
    neural_timing_before: Mapping[str, float | int] | None,
    neural_timing_snapshot: NeuralTimingSnapshot | None,
) -> RootPUCTSearchTiming:
    """Build a full-decision timing after cleanup and action selection work."""

    timing = completed_search_timing or RootPUCTSearchTiming()
    if puct_search_call_seconds is not None:
        timing = timing.with_puct_search_residual_partition(
            result_residual_seconds=timing.residual_seconds,
            result_count=puct_search_result_count or 0,
            unrecorded_call_seconds=max(
                0.0,
                puct_search_call_seconds - timing.total_seconds,
            ),
            call_count=puct_search_call_count or 0,
        )
    if opponent_scenario_planning_seconds is not None:
        timing = timing.with_opponent_scenario_planning(opponent_scenario_planning_seconds)
    if policy_evaluation_seconds is not None:
        timing = timing.with_policy_evaluation(policy_evaluation_seconds)
    if belief_world_materialization_seconds is not None:
        timing = timing.with_belief_world_materialization(
            belief_world_materialization_seconds,
            attempt_count=belief_world_materialization_count or 0,
        )
    if root_policy_setup_seconds is not None:
        timing = timing.with_root_policy_setup(root_policy_setup_seconds)
    if direct_prefix_construction_seconds is not None:
        timing = timing.with_direct_prefix_construction(
            direct_prefix_construction_seconds,
            attempt_count=direct_prefix_construction_count or 0,
        )
    if scenario_dispatch_orchestration_seconds is not None:
        timing = timing.with_scenario_dispatch_orchestration(
            scenario_dispatch_orchestration_seconds,
            attempt_count=scenario_dispatch_orchestration_count or 0,
        )
    neural_timing = _neural_timing_delta(
        neural_timing_before,
        _neural_timing_snapshot(neural_timing_snapshot),
    )
    return timing.with_neural_subtiming(
        observation_encoding_seconds=float(neural_timing["observation_encoding_seconds"]),
        observation_encoding_count=int(neural_timing["observation_encoding_count"]),
        neural_forward_seconds=float(neural_timing["neural_forward_seconds"]),
        neural_forward_count=int(neural_timing["neural_forward_count"]),
        action_prior_neural_forward_seconds=float(neural_timing["action_prior_neural_forward_seconds"]),
        action_prior_neural_forward_count=int(neural_timing["action_prior_neural_forward_count"]),
        opponent_action_prior_neural_forward_seconds=float(
            neural_timing["opponent_action_prior_neural_forward_seconds"]
        ),
        opponent_action_prior_neural_forward_count=int(
            neural_timing["opponent_action_prior_neural_forward_count"]
        ),
        policy_neural_forward_seconds=float(neural_timing["policy_neural_forward_seconds"]),
        policy_neural_forward_count=int(neural_timing["policy_neural_forward_count"]),
        value_neural_forward_seconds=float(neural_timing["value_neural_forward_seconds"]),
        value_neural_forward_count=int(neural_timing["value_neural_forward_count"]),
    ).with_total(_timing_perf_counter() - timing_started_at)


def _refresh_root_puct_timing_total(
    metadata: Mapping[str, object],
    *,
    total_seconds: float,
) -> None:
    """Refresh a fallback timing after its branch environment has closed."""

    if not isinstance(metadata, dict):
        return
    payload = metadata.get("root_puct_timing")
    if not isinstance(payload, Mapping):
        return
    values = {
        timing_field.name: payload[timing_field.name]
        for timing_field in fields(RootPUCTSearchTiming)
        if timing_field.name in payload
    }
    metadata["root_puct_timing"] = RootPUCTSearchTiming(**values).with_total(total_seconds).to_dict()
    metadata["root_puct_elapsed_seconds"] = total_seconds


def _opponent_action_planner_error(
    *,
    player_id: PlayerId,
    requested_players: tuple[PlayerId, ...],
    opponent_actions: Mapping[PlayerId, int],
) -> str | None:
    if player_id in opponent_actions:
        return "opponent_action_planner returned the acting player's action"
    requested_opponents = set(requested_players) - {player_id}
    action_players = set(opponent_actions)
    missing = sorted(requested_opponents - action_players)
    extra = sorted(action_players - requested_opponents)
    if missing or extra:
        details = []
        if missing:
            details.append(f"missing opponent actions for {', '.join(missing)}")
        if extra:
            details.append(f"unexpected opponent actions for {', '.join(extra)}")
        return "; ".join(details)
    return None


def _deferred_opponent_action_players(context: PolicyContext) -> tuple[PlayerId, ...]:
    """Return public-timing opponent actions that must resolve after this root choice."""

    state = context.public_materialization_state
    player = getattr(state, "deferred_opponent_action_player", None)
    if player in {"p1", "p2"} and player != context.player_id:
        return (player,)
    return ()


def _deferred_opponent_action_planner_error(
    *,
    context: PolicyContext,
    deferred_actions: Mapping[PlayerId, int],
    deferred_action_priors: Mapping[PlayerId, tuple[float, ...]],
) -> str | None:
    expected = set(_deferred_opponent_action_players(context))
    action_players = set(deferred_actions)
    prior_players = set(deferred_action_priors)
    if action_players & prior_players or action_players | prior_players != expected:
        return "deferred opponent action planner returned an unexpected action set"
    for player, action_index in deferred_actions.items():
        if (
            isinstance(action_index, bool)
            or not isinstance(action_index, int)
            or not 0 <= action_index < MOVE_ACTION_COUNT
        ):
            return f"deferred opponent action planner returned an invalid action for {player}: {action_index!r}"
    for player, priors in deferred_action_priors.items():
        if len(priors) != MOVE_ACTION_COUNT:
            return f"deferred opponent action planner returned invalid move priors for {player}"
        if any(
            isinstance(value, bool)
            or not isinstance(value, (float, int))
            or not math.isfinite(value)
            or value < 0.0
            for value in priors
        ) or sum(priors) <= 0.0:
            return f"deferred opponent action planner returned invalid move priors for {player}"
    return None


def _best_prior_candidate(
    candidates: tuple[PUCTBranchSearchCandidate, ...],
) -> PUCTBranchSearchCandidate:
    if not candidates:
        raise ValueError("root PUCT search produced no candidates.")
    return max(candidates, key=lambda candidate: (candidate.prior, -candidate.action_index))


def _best_value_candidate(
    candidates: tuple[PUCTBranchSearchCandidate, ...],
) -> PUCTBranchSearchCandidate:
    if not candidates:
        raise ValueError("root PUCT search produced no candidates.")
    return max(candidates, key=lambda candidate: (candidate.value, -candidate.action_index))


def _most_visited_candidate(
    candidates: tuple[PUCTBranchSearchCandidate, ...],
) -> PUCTBranchSearchCandidate:
    if not candidates:
        raise ValueError("root PUCT search produced no candidates.")
    return max(
        candidates,
        key=lambda candidate: (
            candidate.visits,
            candidate.prior,
            candidate.value,
            candidate.score,
            -candidate.action_index,
        ),
    )


def _selected_candidate(
    search: PUCTBranchSearchResult,
    *,
    mode: str,
) -> PUCTBranchSearchCandidate:
    if mode == "puct":
        return search.best_candidate
    if mode == "value":
        return _best_value_candidate(search.candidates)
    if mode == "visits":
        return _most_visited_candidate(search.candidates)
    raise ValueError("selection mode must be 'puct', 'value', or 'visits'.")


def _opponent_action_scenarios(
    policy: RootPUCTSearchPolicy,
    context: PolicyContext,
    rng: random.Random,
) -> tuple[OpponentActionScenario, ...]:
    if policy.opponent_action_scenario_planner is not None:
        return _normalize_scenarios(tuple(policy.opponent_action_scenario_planner(context, rng)))
    if _deferred_opponent_action_players(context):
        prior_fn = getattr(policy.opponent_action_planner, "opponent_prior_fn", None)
        if callable(prior_fn):
            return _normalize_scenarios(
                tuple(prior_top_k_opponent_action_scenario_planner(prior_fn, scenario_count=1)(context, rng))
            )
    return _normalize_scenarios(
        (
            OpponentActionScenario(
                actions=dict(policy.opponent_action_planner(context, rng)),
                weight=1.0,
                label="single",
            ),
        )
    )


def _start_override_sampled_scenario_groups(
    scenarios: Sequence[OpponentActionScenario],
    *,
    samples_per_scenario: int,
) -> tuple[_OpponentActionScenarioGroup, ...]:
    if samples_per_scenario <= 0:
        raise ValueError("samples_per_scenario must be positive.")
    scenario_tuple = tuple(scenarios)
    if samples_per_scenario == 1:
        return tuple(
            _OpponentActionScenarioGroup(root=scenario, samples=(scenario,))
            for scenario in scenario_tuple
        )
    groups: list[_OpponentActionScenarioGroup] = []
    for scenario in scenario_tuple:
        sample_weight = scenario.weight / samples_per_scenario
        samples: list[OpponentActionScenario] = []
        for sample_index in range(samples_per_scenario):
            samples.append(
                OpponentActionScenario(
                    actions=dict(scenario.actions),
                    deferred_actions=dict(scenario.deferred_actions),
                    deferred_action_priors=dict(scenario.deferred_action_priors),
                    weight=sample_weight,
                    label=f"{scenario.label}/belief-sample-{sample_index + 1}",
                )
            )
        groups.append(_OpponentActionScenarioGroup(root=scenario, samples=tuple(samples)))
    return tuple(groups)


def _start_override_samples_per_scenario(
    policy: RootPUCTSearchPolicy,
    context: PolicyContext,
) -> int:
    configured = policy.start_override_samples_per_scenario
    if configured is not None:
        return configured
    if policy.start_override_planner is None:
        raise ValueError("dynamic start-override sampling requires start_override_planner.")
    resolver = getattr(policy.start_override_planner, "sample_count_for_context", None)
    if not callable(resolver):
        raise ValueError("dynamic start-override sampling planner lacks sample_count_for_context.")
    resolved = resolver(context)
    if isinstance(resolved, bool) or not isinstance(resolved, int) or resolved <= 0:
        raise ValueError("sample_count_for_context must return a positive integer.")
    return resolved


def _start_override_sampling_metadata(
    policy: RootPUCTSearchPolicy,
    context: PolicyContext,
) -> Mapping[str, object]:
    if policy.start_override_planner is None:
        return {}
    provider = getattr(policy.start_override_planner, "sampling_metadata_for_context", None)
    if not callable(provider):
        return {}
    metadata = provider(context)
    if not isinstance(metadata, Mapping):
        raise ValueError("sampling_metadata_for_context must return a mapping.")
    return dict(metadata)


def _flatten_scenario_groups(
    scenario_groups: Sequence[_OpponentActionScenarioGroup],
) -> tuple[OpponentActionScenario, ...]:
    return tuple(scenario for group in scenario_groups for scenario in group.samples)


def _uses_scenario_independent_start_overrides(
    policy: RootPUCTSearchPolicy,
    context: PolicyContext,
) -> bool:
    return policy.start_override_planner is not None and bool(
        getattr(policy.start_override_planner, "scenario_independent", False)
    ) and not _deferred_opponent_action_players(context)


def _shared_start_override_samples(
    *,
    env: PokeZeroEnv,
    policy: RootPUCTSearchPolicy,
    context: PolicyContext,
    rng: random.Random,
    sample_scenarios: Sequence[OpponentActionScenario],
    search_trajectory: BattleTrajectory,
    on_attempt: Callable[[], None] | None = None,
    on_direct_materialization_unavailable: Callable[[str], None] | None = None,
    on_direct_materialization_observation_mismatch_path: Callable[[str], None] | None = None,
) -> _SharedStartOverrideSamples:
    if policy.start_override_planner is None:
        raise ValueError("start_override_planner is required for shared start overrides.")

    overrides: list[BattleStartOverride | None] = []
    prepared_prefixes: list[PreparedReplayPrefix | None] = []
    materialization_modes: list[str | None] = []
    rejection_reasons: list[str | None] = []
    attempts_used = 0
    duplicate_attempts = 0
    seen_override_keys: set[tuple[object, ...]] = set()
    for sample_index, sample_scenario in enumerate(sample_scenarios):
        sampled_override: BattleStartOverride | None = None
        prepared_prefix: PreparedReplayPrefix | None = None
        sample_rejections: list[str] = []
        for _attempt_index in range(policy.start_override_attempts):
            attempts_used += 1
            if on_attempt is not None:
                on_attempt()
            start_override = policy.start_override_planner(
                context,
                sample_scenario,
                sample_index,
                rng,
            )
            if start_override is None:
                sample_rejections.append("start override planner did not produce a sampled world")
                continue
            try:
                materialized_override = _materialize_start_override(start_override)
            except ValueError as exc:
                reason = _opponent_scenario_replay_legality_error(exc, sample_scenario)
                if reason is None:
                    raise
                sample_rejections.append(reason)
                continue
            override_key = _start_override_key(materialized_override)
            if override_key in seen_override_keys:
                duplicate_attempts += 1
                sample_rejections.append("sampled start override duplicated an earlier materialized world")
                continue
            seen_override_keys.add(override_key)
            try:
                prepared_prefix = prepare_direct_materialization_prefix(
                    env=env,
                    trajectory=search_trajectory,
                    player_id=context.player_id,
                    prefix_decision_round_count=context.decision_round_index,
                    start_override=materialized_override,
                    public_materialization_state=context.public_materialization_state,
                    deferred_opponent_actions=sample_scenario.deferred_actions,
                    expected_current_observation=context.observation,
                    # Shared sampled worlds need only prove the branch-point state. Earlier
                    # custom-game replay observations can drift for metadata-only reasons.
                    replay_hp_fraction_tolerance=policy.start_override_hp_fraction_tolerance,
                    on_unavailable=on_direct_materialization_unavailable,
                    on_observation_mismatch_path=on_direct_materialization_observation_mismatch_path,
                )
                if prepared_prefix is None:
                    prepared_prefix = prepare_replay_prefix(
                        env=env,
                        trajectory=search_trajectory,
                        player_id=context.player_id,
                        prefix_decision_round_count=context.decision_round_index,
                        start_override=materialized_override,
                        expected_current_observation=context.observation,
                        replay_hp_fraction_tolerance=policy.start_override_hp_fraction_tolerance,
                    )
            except ValueError as exc:
                reason = _opponent_scenario_replay_legality_error(exc, sample_scenario)
                if reason is None:
                    raise
                sample_rejections.append(reason)
                continue
            sampled_override = materialized_override
            break
        overrides.append(sampled_override)
        prepared_prefixes.append(prepared_prefix)
        materialization_modes.append(
            prepared_prefix.materialization_mode if prepared_prefix is not None else None
        )
        rejection_reasons.append(
            None if sampled_override is not None else _format_replay_rejection_reasons(sample_rejections)
        )
    return _SharedStartOverrideSamples(
        overrides=tuple(overrides),
        prepared_prefixes=tuple(prepared_prefixes),
        materialization_modes=tuple(materialization_modes),
        rejection_reasons=tuple(rejection_reasons),
        attempts_used=attempts_used,
        duplicate_attempts=duplicate_attempts,
    )


def _shared_start_override_metadata(samples: _SharedStartOverrideSamples | None) -> dict[str, int]:
    if samples is None:
        return {}
    accepted = sum(1 for override in samples.overrides if override is not None)
    metadata = {
        "root_puct_start_override_shared_samples": len(samples.overrides),
        "root_puct_start_override_shared_samples_accepted": accepted,
        "root_puct_start_override_shared_samples_rejected": len(samples.overrides) - accepted,
    }
    if samples.duplicate_attempts:
        metadata["root_puct_start_override_duplicate_attempts"] = samples.duplicate_attempts
    return metadata


def _direct_materialization_rejection_metadata(
    categories: Mapping[str, int],
    mismatch_paths: Mapping[str, int],
) -> dict[str, Mapping[str, int]]:
    cleaned = {
        str(category): int(count)
        for category, count in sorted(categories.items())
        if isinstance(count, int) and not isinstance(count, bool) and count > 0
    }
    cleaned_paths = {
        str(path): int(count)
        for path, count in sorted(mismatch_paths.items())
        if isinstance(count, int) and not isinstance(count, bool) and count > 0
    }
    result: dict[str, Mapping[str, int]] = {}
    if cleaned:
        result["root_puct_direct_materialization_rejection_categories"] = cleaned
    if cleaned_paths:
        result["root_puct_direct_materialization_observation_mismatch_paths"] = cleaned_paths
    return result


def _start_override_key(start_override: BattleStartOverride) -> tuple[object, ...]:
    return (
        start_override.format_id,
        start_override.observation_format_id,
        tuple(sorted(start_override.player_teams.items())),
    )


def _remaining_root_time_budget_seconds(
    *,
    total_budget_seconds: float | None,
    started_at: float,
) -> float | None:
    if total_budget_seconds is None:
        return None
    elapsed = perf_counter() - started_at
    return max(1e-9, total_budget_seconds - elapsed)



def _top_prior_action_choices(
    context: PolicyContext,
    player: PlayerId,
    priors: tuple[float, ...],
    *,
    limit: int,
    rng: random.Random,
    allowed_action_indices: tuple[int, ...] | None = None,
) -> tuple[tuple[int, float], ...]:
    if limit <= 0:
        raise ValueError("opponent action scenario limit must be positive.")
    if allowed_action_indices is not None:
        # A deferred move was committed before the acting side's forced replacement, so the
        # current boundary deliberately has no opponent request mask. Filtering on a stale or
        # synthetic mask would either leak private request state or silently do nothing. The
        # direct sampled world validates availability when it restores the queued action.
        candidates = tuple((index, priors[index]) for index in allowed_action_indices)
    else:
        legal = _requested_legal_action_indices_for_player(context, player)
        candidates = (
            tuple((index, priors[index]) for index in legal)
            if legal
            else _hidden_mask_prior_action_choices(context, player, priors, rng=rng)
        )
    ranked = sorted(candidates, key=lambda item: (-item[1], item[0]))[:limit]
    if not ranked:
        raise ValueError(f"no opponent action candidates available for {player}.")
    total = sum(weight for _index, weight in ranked)
    if total <= 0.0:
        uniform = 1.0 / len(ranked)
        return tuple((index, uniform) for index, _weight in ranked)
    return tuple((index, weight / total) for index, weight in ranked)


def _opponent_action_choice_rng(context: PolicyContext, player: PlayerId) -> random.Random:
    """Return an independent RNG for hidden-mask replay handles.

    Opponent-action planning runs before belief start-override sampling. Keep this stream separate
    so choosing a concrete replay handle for an abstract switch bucket does not shift downstream
    determinization samples.
    """

    return random.Random(
        f"{context.seed}|{context.battle_id}|{context.decision_round_index}|"
        f"{context.player_id}|{player}|hidden-switch-handle"
    )


def _hidden_mask_prior_action_choices(
    context: PolicyContext,
    player: PlayerId,
    priors: tuple[float, ...],
    *,
    rng: random.Random,
) -> tuple[tuple[int, float], ...]:
    """Return public-information-compatible choices without an opponent request mask.

    A fainted opposing active is protocol-visible, so its next request can only be a
    replacement switch. Other request details (move disablement, trapping, and the
    private party order behind a switch slot) remain hidden and therefore stay in
    the replay-time legality path.
    """

    move_choices = (
        ()
        if _public_opponent_force_switch_is_required(context, player)
        else tuple((index, priors[index]) for index in range(MOVE_ACTION_COUNT))
    )
    switch_indices = tuple(range(MOVE_ACTION_COUNT, ACTION_COUNT))
    switch_weight = sum(priors[index] for index in switch_indices)
    # Hidden switch slots are exchangeable at the information-set level, but replay still needs a
    # concrete handle. Sample that handle from the slot-prior mass so repeated determinizations can
    # cover different hidden backline positions without splitting the abstract switch bucket.
    representative_switch = _sample_representative_switch_action(priors, switch_indices, rng)
    return (*move_choices, (representative_switch, switch_weight))


def _public_opponent_force_switch_is_required(
    context: PolicyContext,
    player: PlayerId,
) -> bool:
    """Recognize the one opponent request-family constraint visible to both seats.

    ``opponent_active`` is normalized from public Showdown protocol state for the
    acting player's perspective. Deliberately require the literal boolean value:
    malformed or partial metadata must preserve the conservative hidden-mask
    support instead of suppressing potentially legal move hypotheses.
    """

    if player == context.player_id:
        return False
    opponent_active = context.observation.metadata.get("opponent_active")
    return isinstance(opponent_active, Mapping) and opponent_active.get("fainted") is True


def _sample_representative_switch_action(
    priors: tuple[float, ...],
    switch_indices: tuple[int, ...],
    rng: random.Random,
) -> int:
    switch_weight = sum(max(0.0, priors[index]) for index in switch_indices)
    if switch_weight <= 0.0:
        return switch_indices[rng.randrange(len(switch_indices))]
    threshold = rng.random() * switch_weight
    cumulative = 0.0
    for index in switch_indices:
        cumulative += max(0.0, priors[index])
        if threshold <= cumulative:
            return index
    return switch_indices[-1]


def _normalize_scenarios(
    scenarios: tuple[OpponentActionScenario, ...],
) -> tuple[OpponentActionScenario, ...]:
    if not scenarios:
        raise ValueError("opponent action scenario planner produced no scenarios.")
    total_weight = 0.0
    for scenario in scenarios:
        if scenario.weight <= 0.0 or not math.isfinite(scenario.weight):
            raise ValueError("opponent action scenarios must have finite positive weights.")
        total_weight += scenario.weight
    if total_weight <= 0.0 or not math.isfinite(total_weight):
        raise ValueError("opponent action scenario weights must sum to a finite positive value.")
    return tuple(scenario.normalized(total_weight=total_weight) for scenario in scenarios)


def _aggregate_scenario_searches(
    searches: Sequence[PUCTBranchSearchResult],
    *,
    opponent_scenarios: Sequence[OpponentActionScenario],
    cpuct: float,
) -> PUCTBranchSearchResult:
    scenario_searches = tuple(searches)
    scenarios = tuple(opponent_scenarios)
    if not scenario_searches:
        raise ValueError("root PUCT search produced no scenario searches.")
    if len(scenario_searches) != len(scenarios):
        raise ValueError("scenario search count must match opponent scenario count.")
    if len(scenario_searches) == 1:
        return scenario_searches[0]

    first = scenario_searches[0]
    action_order = tuple(candidate.action_index for candidate in first.candidates)
    if not action_order:
        raise ValueError("root PUCT search produced no candidates.")
    aggregate_inputs: list[tuple[PUCTBranchSearchCandidate, int, float]] = []
    total_visits = 0
    for action_index in action_order:
        weighted_value = 0.0
        weighted_visits = 0.0
        first_candidate: PUCTBranchSearchCandidate | None = None
        for scenario, search in zip(scenarios, scenario_searches, strict=True):
            by_action = {candidate.action_index: candidate for candidate in search.candidates}
            candidate = by_action.get(action_index)
            if candidate is None:
                raise ValueError("scenario searches produced mismatched root action candidates.")
            if first_candidate is None:
                first_candidate = candidate
            weighted_visits += scenario.weight * candidate.visits
            weighted_value += scenario.weight * candidate.value
        if first_candidate is None:
            raise ValueError("root PUCT search produced no candidates.")
        visits = max(1, int(round(weighted_visits)))
        total_visits += visits
        aggregate_inputs.append((first_candidate, visits, weighted_value))

    sqrt_total = math.sqrt(total_visits)
    aggregated_candidates: list[PUCTBranchSearchCandidate] = []
    for first_candidate, visits, weighted_value in aggregate_inputs:
        value_candidate = replace(first_candidate.value_candidate, value=weighted_value)
        aggregated_candidates.append(
            _puct_candidate(
                value_candidate=value_candidate,
                prior=first_candidate.prior,
                cpuct=cpuct,
                sqrt_total_visits=sqrt_total,
                visits=visits,
                total_value=weighted_value * visits,
            )
        )

    return PUCTBranchSearchResult(
        player_id=first.player_id,
        prefix_decision_round_count=first.prefix_decision_round_count,
        opponent_actions=dict(scenarios[0].actions),
        cpuct=cpuct,
        total_visits=total_visits,
        candidates=tuple(aggregated_candidates),
        value_search=first.value_search,
        root_visit_budget=first.root_visit_budget,
        configured_root_visit_budget=first.configured_root_visit_budget,
        visit_budget_context=first.visit_budget_context,
        root_time_budget_seconds=first.root_time_budget_seconds,
        time_budget_exhausted=any(search.time_budget_exhausted for search in scenario_searches),
        timing=RootPUCTSearchTiming.aggregate(tuple(search.timing for search in scenario_searches)),
    )


def _root_visit_budget_selector_config(selector: RootVisitBudgetSelector) -> Mapping[str, object]:
    to_dict = getattr(selector, "to_dict", None)
    if callable(to_dict):
        payload = to_dict()
        if isinstance(payload, Mapping):
            return dict(payload)
    return {"selector_id": str(getattr(selector, "selector_id", "custom"))}


def _opponent_action_scenario_payload(scenario: OpponentActionScenario) -> dict[str, object]:
    payload: dict[str, object] = {
        "label": scenario.label,
        "weight": scenario.weight,
        "actions": dict(scenario.actions),
    }
    if scenario.deferred_actions:
        payload["deferred_actions"] = dict(scenario.deferred_actions)
    if scenario.deferred_action_priors:
        payload["deferred_action_priors"] = {
            player: list(priors)
            for player, priors in scenario.deferred_action_priors.items()
        }
    return payload


def _opponent_scenario_skip_metadata(
    *,
    opponent_scenarios: Sequence[OpponentActionScenario],
    used_scenarios: Sequence[OpponentActionScenario],
    skipped_scenarios: Sequence[tuple[OpponentActionScenario, str]],
    unsearched_scenario_count: int = 0,
    opponent_action_group_count: int | None = None,
    used_action_group_count: int | None = None,
    skipped_action_group_count: int | None = None,
    unsearched_action_group_count: int | None = None,
) -> dict[str, object]:
    skip_categories: dict[str, int] = {}
    replay_rejection_decision_rounds: dict[str, int] = {}
    replay_request_mismatch_decision_rounds: dict[str, int] = {}
    replay_request_mismatch_players: dict[str, int] = {}
    replay_request_mismatch_shapes: dict[str, int] = {}
    start_override_mismatch_decision_rounds: dict[str, int] = {}
    first_observation_mismatch_paths: dict[str, int] = {}
    missing_sampled_world_reason_categories: dict[str, int] = {}
    for _scenario, reason in skipped_scenarios:
        category = root_puct_fallback_category(reason)
        skip_categories[category] = skip_categories.get(category, 0) + 1
        _merge_counts(
            replay_rejection_decision_rounds,
            root_puct_replay_rejection_decision_round_counts(reason),
        )
        _merge_counts(
            replay_request_mismatch_decision_rounds,
            root_puct_replay_request_mismatch_decision_round_counts(reason),
        )
        _merge_counts(
            replay_request_mismatch_players,
            root_puct_replay_request_mismatch_player_counts(reason),
        )
        _merge_counts(
            replay_request_mismatch_shapes,
            root_puct_replay_request_mismatch_shape_counts(reason),
        )
        _merge_counts(
            start_override_mismatch_decision_rounds,
            root_puct_start_override_mismatch_decision_round_counts(reason),
        )
        _merge_counts(
            first_observation_mismatch_paths,
            root_puct_first_observation_mismatch_path_counts(reason),
        )
        _merge_counts(
            missing_sampled_world_reason_categories,
            root_puct_missing_sampled_world_reason_counts(reason),
        )
    metadata: dict[str, object] = {
        "root_puct_opponent_action_scenarios_generated": len(opponent_scenarios),
        "root_puct_opponent_action_scenarios_skipped": len(skipped_scenarios),
        "root_puct_opponent_action_scenarios_unsearched": unsearched_scenario_count,
        "root_puct_opponent_action_scenarios": [
            _opponent_action_scenario_payload(scenario)
            for scenario in used_scenarios
        ],
        "root_puct_opponent_action_skipped_scenarios": [
            {
                **_opponent_action_scenario_payload(scenario),
                "reason": reason,
                "category": root_puct_fallback_category(reason),
            }
            for scenario, reason in skipped_scenarios
        ],
    }
    if skip_categories:
        metadata["root_puct_opponent_action_skip_categories"] = dict(sorted(skip_categories.items()))
    if replay_rejection_decision_rounds:
        metadata["root_puct_opponent_action_replay_rejection_decision_rounds"] = dict(
            sorted(replay_rejection_decision_rounds.items(), key=lambda item: int(item[0]))
        )
    if replay_request_mismatch_decision_rounds:
        metadata["root_puct_opponent_action_replay_request_mismatch_decision_rounds"] = dict(
            sorted(replay_request_mismatch_decision_rounds.items(), key=lambda item: int(item[0]))
        )
    if replay_request_mismatch_players:
        metadata["root_puct_opponent_action_replay_request_mismatch_players"] = dict(
            sorted(replay_request_mismatch_players.items())
        )
    if replay_request_mismatch_shapes:
        metadata["root_puct_opponent_action_replay_request_mismatch_shapes"] = dict(
            sorted(replay_request_mismatch_shapes.items())
        )
    if start_override_mismatch_decision_rounds:
        metadata["root_puct_opponent_action_start_override_mismatch_decision_rounds"] = dict(
            sorted(start_override_mismatch_decision_rounds.items(), key=lambda item: int(item[0]))
        )
    if first_observation_mismatch_paths:
        metadata["root_puct_opponent_action_first_observation_mismatch_paths"] = dict(
            sorted(first_observation_mismatch_paths.items())
        )
    if missing_sampled_world_reason_categories:
        metadata["root_puct_opponent_action_missing_sampled_world_reason_categories"] = dict(
            sorted(missing_sampled_world_reason_categories.items())
        )
    if opponent_action_group_count is not None:
        metadata.update(
            {
                "root_puct_opponent_action_groups_generated": opponent_action_group_count,
                "root_puct_opponent_action_groups_used": used_action_group_count,
                "root_puct_opponent_action_groups_skipped": skipped_action_group_count,
                "root_puct_opponent_action_groups_unsearched": unsearched_action_group_count,
            }
        )
    return metadata


def _merge_counts(target: dict[str, int], source: Mapping[str, int]) -> None:
    for key, count in source.items():
        target[str(key)] = target.get(str(key), 0) + int(count)


def _leaf_rollout_metadata(
    search: PUCTBranchSearchResult | Sequence[PUCTBranchSearchResult],
    *,
    configured_rounds: int,
) -> Mapping[str, object]:
    if configured_rounds <= 0:
        return {}
    searches = (search,) if isinstance(search, PUCTBranchSearchResult) else tuple(search)
    leaf_evaluations: dict[str, int] = {}
    actual_rounds: dict[str, int] = {}
    for scenario_search in searches:
        for candidate in scenario_search.candidates:
            value_candidate = candidate.value_candidate
            leaf_evaluations[value_candidate.leaf_evaluation] = (
                leaf_evaluations.get(value_candidate.leaf_evaluation, 0) + 1
            )
            round_key = str(value_candidate.leaf_rollout_decision_round_count)
            actual_rounds[round_key] = actual_rounds.get(round_key, 0) + 1
    return {
        "root_puct_leaf_rollout_rounds": configured_rounds,
        "root_puct_leaf_actual_rollout_rounds": dict(sorted(actual_rounds.items())),
        "root_puct_leaf_evaluations": dict(sorted(leaf_evaluations.items())),
    }


@dataclass(frozen=True)
class _OpponentActionLegalityReport:
    checked: bool
    error: str | None = None


class _AllOpponentScenariosReplayIllegal(ValueError):
    def __init__(self, *, details: str, metadata: Mapping[str, object]) -> None:
        super().__init__(f"all opponent action scenarios were replay-illegal: {details}")
        self.metadata = dict(metadata)


def _opponent_action_legality_report(
    context: PolicyContext,
    opponent_actions: Mapping[PlayerId, int],
) -> _OpponentActionLegalityReport:
    checked = False
    for player, action_index in opponent_actions.items():
        legal = _requested_legal_action_indices_for_player(context, player)
        if not legal:
            continue
        checked = True
        if action_index not in legal:
            return _OpponentActionLegalityReport(
                checked=True,
                error=(
                    "opponent_action_planner returned an illegal action "
                    f"for {player}: {action_index}"
                ),
            )
    return _OpponentActionLegalityReport(checked=checked)


def _opponent_scenario_replay_legality_error(
    exc: ValueError,
    scenario: OpponentActionScenario,
) -> str | None:
    message = str(exc)
    if message.startswith("start override does not reproduce recorded replay prefix observations"):
        return message
    if message.startswith("start override planner did not produce a sampled world:"):
        return message
    if message == START_OVERRIDE_MISSING_WORLD_MESSAGE:
        return message
    if message.startswith("replay actions for decision round "):
        return message
    if message.startswith("cannot replay decision round "):
        return message
    if message == "cannot branch from a terminal replay prefix.":
        return message
    if "action_index " in message and re.search(
        r" is not legal for the current request(?: \(request_kind=[^)]+\))?\.$", message
    ):
        return message
    for player, action_index in scenario.actions.items():
        unqualified = f"action_index {action_index} is not legal for the current request."
        if message == unqualified or message == f"{player}: {unqualified}":
            return message
    return None


def _format_replay_rejection_reasons(reasons: Sequence[str]) -> str:
    if not reasons:
        return "unknown replay rejection"
    counts: dict[str, int] = {}
    ordered: list[str] = []
    for reason in reasons:
        if reason not in counts:
            ordered.append(reason)
            counts[reason] = 0
        counts[reason] += 1
    return "; ".join(
        reason if counts[reason] == 1 else f"{reason} ({counts[reason]} attempts)"
        for reason in ordered
    )


def _requested_legal_action_indices_for_player(
    context: PolicyContext,
    player: PlayerId,
) -> tuple[int, ...]:
    legal_action_mask = context.requested_legal_action_masks.get(player)
    if legal_action_mask is None:
        return ()
    if len(legal_action_mask) != ACTION_COUNT:
        return ()
    return tuple(index for index, legal in enumerate(legal_action_mask) if legal)


def _leaf_rollout_policies(
    context: PolicyContext,
    factory: LeafRolloutPolicyFactory | None,
) -> Mapping[PlayerId, Policy]:
    if factory is None:
        raise ValueError("leaf_rollout_policy_factory is required when leaf rollouts are enabled.")
    players = {
        "p1",
        "p2",
        context.player_id,
        *context.requested_players,
    }
    for step in context.trajectory.steps:
        players.add(step.player_id)
    return {player: factory(player) for player in sorted(players)}


def _validate_action_prior_vector(priors: tuple[float, ...], *, name: str) -> None:
    if len(priors) != ACTION_COUNT:
        raise ValueError(f"{name} must contain {ACTION_COUNT} values.")
    if any(value < 0.0 or not math.isfinite(value) for value in priors):
        raise ValueError(f"{name} must contain finite non-negative values.")


def _temperature_scale_action_priors(
    priors: Sequence[float],
    *,
    temperature: float,
) -> tuple[float, ...]:
    if temperature <= 0.0 or not math.isfinite(temperature):
        raise ValueError("root_prior_temperature must be a finite positive value.")
    normalized = tuple(float(value) for value in priors)
    _validate_action_prior_vector(normalized, name="action priors")
    if temperature == 1.0:
        return normalized
    exponent = 1.0 / temperature
    scaled = tuple(0.0 if value <= 0.0 else value**exponent for value in normalized)
    total = sum(scaled)
    if total <= 0.0:
        return scaled
    return tuple(value / total for value in scaled)


def _root_dirichlet_action_priors(
    priors: Sequence[float],
    *,
    context: PolicyContext,
    legal_action_mask: Sequence[bool],
    alpha: float | None,
    mix: float,
    base_seed: int,
) -> tuple[tuple[float, ...], dict[str, object]]:
    """Mix a reproducible Dirichlet draw into legal root priors when explicitly enabled."""

    normalized = tuple(float(value) for value in priors)
    _validate_action_prior_vector(normalized, name="action priors")
    if alpha is None:
        return normalized, {"root_puct_root_dirichlet_enabled": False}

    legal = legal_action_indices(tuple(legal_action_mask))
    if not legal:
        raise ValueError("root Dirichlet noise requires at least one legal action.")
    decision_seed = _root_dirichlet_decision_seed(context, base_seed=base_seed)
    noise_rng = random.Random(decision_seed)
    samples = [noise_rng.gammavariate(alpha, 1.0) for _index in legal]
    sample_total = sum(samples)
    underflow_fallback = False
    if sample_total <= 0.0 or not math.isfinite(sample_total):
        # Extremely small alpha values can underflow every gamma draw. Preserve the
        # audit run by falling back to a valid symmetric draw instead of aborting a game.
        noise = (1.0 / len(legal),) * len(legal)
        underflow_fallback = True
    else:
        noise = tuple(sample / sample_total for sample in samples)
    legal_prior_total = sum(normalized[index] for index in legal)
    if legal_prior_total > 0.0:
        legal_priors = tuple(normalized[index] / legal_prior_total for index in legal)
    else:
        legal_priors = (1.0 / len(legal),) * len(legal)
    mixed = tuple((1.0 - mix) * prior + mix * noise_value for prior, noise_value in zip(legal_priors, noise, strict=True))
    output = [0.0] * ACTION_COUNT
    for index, value in zip(legal, mixed, strict=True):
        output[index] = value
    return tuple(output), {
        "root_puct_root_dirichlet_enabled": True,
        "root_puct_root_dirichlet_alpha": alpha,
        "root_puct_root_dirichlet_mix": mix,
        "root_puct_root_dirichlet_base_seed": base_seed,
        "root_puct_root_dirichlet_decision_seed": decision_seed,
        "root_puct_root_dirichlet_underflow_fallback": underflow_fallback,
        "root_puct_root_dirichlet_noise": {str(index): value for index, value in zip(legal, noise, strict=True)},
        "root_puct_root_dirichlet_mixed_priors": {
            str(index): value for index, value in zip(legal, mixed, strict=True)
        },
    }


def _root_dirichlet_decision_seed(context: PolicyContext, *, base_seed: int) -> int:
    """Derive an independent reproducible noise seed for one player decision."""

    digest = hashlib.sha256(
        f"{base_seed}:{context.seed}:{context.player_id}:{context.decision_round_index}:root-dirichlet".encode(
            "utf-8"
        )
    ).digest()
    return int.from_bytes(digest[:8], "big")


def _trajectory_with_current_observation(context: PolicyContext) -> BattleTrajectory:
    trajectory = BattleTrajectory(
        battle_id=context.trajectory.battle_id,
        format_id=context.trajectory.format_id,
        seed=context.trajectory.seed,
        steps=list(context.trajectory.steps),
        terminal=context.trajectory.terminal,
        metadata=dict(context.trajectory.metadata),
    )
    if any(
        step.player_id == context.player_id and step.turn_index == context.decision_round_index
        for step in trajectory.steps
    ):
        return trajectory
    legal = legal_action_indices(context.observation.legal_action_mask)
    trajectory.append(
        TrajectoryStep(
            player_id=context.player_id,
            turn_index=context.decision_round_index,
            observation=context.observation,
            legal_action_mask=tuple(context.observation.legal_action_mask),
            action_index=legal[0],
            metadata={"synthetic_current_observation": True},
        )
    )
    return trajectory
