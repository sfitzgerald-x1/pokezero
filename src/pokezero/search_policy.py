"""Policy adapters backed by replay-from-root search."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from itertools import product
from time import perf_counter
import math
import random
from typing import Callable, Mapping, Sequence

from .actions import ACTION_COUNT
from .env import PlayerId, PokeZeroEnv
from .observation import PokeZeroObservationV0
from .policy import Policy, PolicyContext, PolicyDecision, RandomLegalPolicy, legal_action_indices
from .rollout import RolloutConfig, _reset_unique_policies
from .search import (
    ActionPriorVector,
    ObservationValueFunction,
    PUCTBranchSearchCandidate,
    PUCTBranchSearchResult,
    StartOverrideSource,
    _puct_candidate,
    player_observation_history,
    puct_branch_search,
)
from .trajectory import BattleTrajectory, TrajectoryStep

OpponentActionPlanner = Callable[[PolicyContext, random.Random], Mapping[PlayerId, int]]
OpponentActionScenarioPlanner = Callable[[PolicyContext, random.Random], Sequence["OpponentActionScenario"]]
ActionPriorFunction = Callable[[tuple[PokeZeroObservationV0, ...]], ActionPriorVector]
OpponentActionPriorFunction = Callable[[tuple[PokeZeroObservationV0, ...]], ActionPriorVector]
LeafRolloutPolicyFactory = Callable[[PlayerId], Policy]


def no_opponent_action_planner(context: PolicyContext, rng: random.Random) -> Mapping[PlayerId, int]:
    del context, rng
    return {}


@dataclass(frozen=True)
class OpponentActionScenario:
    actions: Mapping[PlayerId, int]
    weight: float = 1.0
    label: str = "single"

    def normalized(self, *, total_weight: float) -> "OpponentActionScenario":
        return OpponentActionScenario(
            actions=dict(self.actions),
            weight=self.weight / total_weight,
            label=self.label,
        )


@dataclass(frozen=True)
class _OpponentActionScenarioGroup:
    root: OpponentActionScenario
    samples: tuple[OpponentActionScenario, ...]


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
    return planner


def prior_top_k_opponent_action_scenario_planner(
    prior_fn: OpponentActionPriorFunction,
    *,
    scenario_count: int,
) -> OpponentActionScenarioPlanner:
    """Enumerate likely opponent root-action scenarios from player-local opponent priors.

    The prior function only sees the acting player's observation history. Requested-opponent legal
    masks are still a privileged benchmark safety guard, and they affect scenario support and
    weights so replay branches stay submit-valid. For multi-opponent turns, the final joint scenario
    set is capped to ``scenario_count`` after combining per-opponent choices.
    """

    if scenario_count <= 0:
        raise ValueError("scenario_count must be positive.")

    def planner(context: PolicyContext, rng: random.Random) -> tuple[OpponentActionScenario, ...]:
        del rng
        requested_opponents = tuple(player for player in context.requested_players if player != context.player_id)
        if not requested_opponents:
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
            (player, _top_prior_action_choices(context, player, priors, limit=scenario_count))
            for player in requested_opponents
        )
        scenarios: list[OpponentActionScenario] = []
        for combination in product(*(choices for _player, choices in choices_by_player)):
            actions = {
                player: action
                for (player, _choices), (action, _weight) in zip(choices_by_player, combination, strict=True)
            }
            weight = math.prod(weight for _action, weight in combination)
            label = ",".join(
                f"{player}:{action}"
                for (player, _choices), (action, _weight) in zip(choices_by_player, combination, strict=True)
            )
            scenarios.append(OpponentActionScenario(actions=actions, weight=weight, label=label))
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
    root_time_budget_seconds: float | None = None
    root_prior_temperature: float = 1.0
    max_opponent_action_scenarios: int | None = None
    leaf_rollout_decision_rounds: int = 0
    leaf_rollout_policy_factory: LeafRolloutPolicyFactory | None = None
    start_override_planner: StartOverridePlanner | None = None
    start_override_attempts: int = 1
    start_override_samples_per_scenario: int = 1
    leaf_rollout_metadata: Mapping[str, object] = field(default_factory=dict)

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
        if self.root_time_budget_seconds is not None and (
            self.root_time_budget_seconds <= 0.0 or not math.isfinite(self.root_time_budget_seconds)
        ):
            raise ValueError("root_time_budget_seconds must be a finite positive value when set.")
        if self.root_prior_temperature <= 0.0 or not math.isfinite(self.root_prior_temperature):
            raise ValueError("root_prior_temperature must be a finite positive value.")
        if self.max_opponent_action_scenarios is not None and self.max_opponent_action_scenarios <= 0:
            raise ValueError("max_opponent_action_scenarios must be positive when set.")
        if self.leaf_rollout_decision_rounds < 0:
            raise ValueError("leaf_rollout_decision_rounds must be non-negative.")
        if self.leaf_rollout_decision_rounds and self.leaf_rollout_policy_factory is None:
            raise ValueError("leaf_rollout_policy_factory is required when leaf rollouts are enabled.")
        if self.start_override_attempts <= 0:
            raise ValueError("start_override_attempts must be positive.")
        if self.start_override_samples_per_scenario <= 0:
            raise ValueError("start_override_samples_per_scenario must be positive.")
        if self.start_override_samples_per_scenario > 1 and self.start_override_planner is None:
            raise ValueError("start_override_samples_per_scenario requires start_override_planner.")

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
                "fallback_policy_id": decision.policy_id,
            },
        )

    def select_action_with_context(
        self,
        context: PolicyContext,
        *,
        rng: random.Random,
    ) -> PolicyDecision:
        if context.player_id not in context.requested_players:
            return self._fallback(context, rng=rng, reason="player is not requested")
        try:
            opponent_scenarios = _opponent_action_scenarios(self, context, rng)
        except ValueError as exc:
            return self._fallback(context, rng=rng, reason=str(exc))
        legality_checked = False
        for scenario in opponent_scenarios:
            planner_error = _opponent_action_planner_error(
                player_id=context.player_id,
                requested_players=context.requested_players,
                opponent_actions=scenario.actions,
            )
            if planner_error is not None:
                return self._fallback(context, rng=rng, reason=planner_error)
            legality_report = _opponent_action_legality_report(context, scenario.actions)
            legality_checked = legality_checked or legality_report.checked
            if legality_report.error is not None:
                return self._fallback(context, rng=rng, reason=legality_report.error)
        search_scenario_groups = _start_override_sampled_scenario_groups(
            opponent_scenarios,
            samples_per_scenario=(
                self.start_override_samples_per_scenario
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
        priors = _temperature_scale_action_priors(
            self.prior_fn(history),
            temperature=self.root_prior_temperature,
        )
        leaf_rollout_policies = (
            _leaf_rollout_policies(context, self.leaf_rollout_policy_factory)
            if self.leaf_rollout_decision_rounds
            else None
        )
        start = perf_counter()
        env = self.env_factory()
        skipped_scenarios: list[tuple[OpponentActionScenario, str]] = []
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
                for group_index, scenario_group in enumerate(search_scenario_groups):
                    group_search_pairs: list[tuple[OpponentActionScenario, PUCTBranchSearchResult]] = []
                    for scenario in scenario_group.samples:
                        scenario_index = flat_scenario_index
                        flat_scenario_index += 1
                        scenario_search: PUCTBranchSearchResult | None = None
                        scenario_start_override: StartOverrideSource = None
                        replay_rejection_reasons: list[str] = []
                        attempts = self.start_override_attempts if self.start_override_planner is not None else 1
                        for _attempt_index in range(attempts):
                            start_override_attempts_used += 1
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
                            scenario_root_time_budget_seconds = _remaining_root_time_budget_seconds(
                                total_budget_seconds=self.root_time_budget_seconds,
                                started_at=start,
                            )
                            try:
                                search = puct_branch_search(
                                    env=env,
                                    trajectory=search_trajectory,
                                    player_id=context.player_id,
                                    prefix_decision_round_count=context.decision_round_index,
                                    legal_action_mask=context.observation.legal_action_mask,
                                    opponent_actions=scenario.actions,
                                    value_fn=self.value_fn,
                                    action_priors=priors,
                                    cpuct=self.cpuct,
                                    leaf_rollout_policies=leaf_rollout_policies,
                                    leaf_rollout_config=self.rollout_config,
                                    leaf_rollout_decision_rounds=self.leaf_rollout_decision_rounds,
                                    root_visit_budget=self.root_visit_budget,
                                    root_time_budget_seconds=scenario_root_time_budget_seconds,
                                    start_override=start_override,
                                    expected_current_observation=context.observation,
                                )
                            except ValueError as exc:
                                reason = _opponent_scenario_replay_legality_error(exc, scenario)
                                if reason is None:
                                    raise
                                replay_rejection_reasons.append(reason)
                                if start_override is None:
                                    break
                            else:
                                scenario_search = search
                                scenario_start_override = start_override
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
                                "root_puct_start_override_samples_per_scenario": (
                                    self.start_override_samples_per_scenario
                                ),
                            }
                        )
                    raise _AllOpponentScenariosReplayIllegal(
                        details=details,
                        metadata=metadata,
                    )
                used_scenarios = _normalize_scenarios(tuple(scenario for scenario, _search in scenario_search_pairs))
                scenario_searches = tuple(search for _scenario, search in scenario_search_pairs)
            except _AllOpponentScenariosReplayIllegal as exc:
                return self._fallback(
                    context,
                    rng=rng,
                    reason=str(exc),
                    extra_metadata=exc.metadata,
                )
            except Exception as exc:
                return self._fallback(context, rng=rng, reason=f"search failed: {exc}")
            elapsed_seconds = perf_counter() - start
        finally:
            close = getattr(env, "close", None)
            if callable(close):
                close()

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
            budget_metadata = {
                "root_puct_root_time_budget_seconds": self.root_time_budget_seconds,
                "root_puct_root_scenario_time_budget_seconds": scenario_searches[0].root_time_budget_seconds,
                "root_puct_time_budget_exhausted": any(search.time_budget_exhausted for search in scenario_searches),
            }
        start_override_metadata = (
            {
                "root_puct_start_override_sources_used": start_override_sources_used,
                "root_puct_start_override_attempts": self.start_override_attempts,
                "root_puct_start_override_attempts_used": start_override_attempts_used,
                "root_puct_start_override_samples_per_scenario": self.start_override_samples_per_scenario,
            }
            if self.start_override_planner is not None
            else {}
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
    ) -> PolicyDecision:
        if not self.allow_fallback:
            raise ValueError(f"root PUCT search cannot select an action: {reason}")
        decision = self.fallback_policy.select_action(context.observation, rng=rng)
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
                "fallback_policy_id": decision.policy_id,
                **dict(extra_metadata or {}),
            },
        )


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
                    weight=sample_weight,
                    label=f"{scenario.label}/belief-sample-{sample_index + 1}",
                )
            )
        groups.append(_OpponentActionScenarioGroup(root=scenario, samples=tuple(samples)))
    return tuple(groups)


def _flatten_scenario_groups(
    scenario_groups: Sequence[_OpponentActionScenarioGroup],
) -> tuple[OpponentActionScenario, ...]:
    return tuple(scenario for group in scenario_groups for scenario in group.samples)


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
) -> tuple[tuple[int, float], ...]:
    if limit <= 0:
        raise ValueError("opponent action scenario limit must be positive.")
    legal = _requested_legal_action_indices_for_player(context, player)
    candidate_indices = legal if legal else tuple(range(ACTION_COUNT))
    ranked = sorted(candidate_indices, key=lambda index: (-priors[index], index))[:limit]
    if not ranked:
        raise ValueError(f"no opponent action candidates available for {player}.")
    total = sum(priors[index] for index in ranked)
    if total <= 0.0:
        uniform = 1.0 / len(ranked)
        return tuple((index, uniform) for index in ranked)
    return tuple((index, priors[index] / total) for index in ranked)


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
        root_time_budget_seconds=first.root_time_budget_seconds,
        time_budget_exhausted=any(search.time_budget_exhausted for search in scenario_searches),
    )


def _opponent_action_scenario_payload(scenario: OpponentActionScenario) -> dict[str, object]:
    return {
        "label": scenario.label,
        "weight": scenario.weight,
        "actions": dict(scenario.actions),
    }


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
            }
            for scenario, reason in skipped_scenarios
        ],
    }
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
    if message.startswith("replay actions for decision round "):
        return message
    if message.startswith("cannot replay decision round "):
        return message
    if message == "cannot branch from a terminal replay prefix.":
        return message
    if "action_index " in message and message.endswith(" is not legal for the current request."):
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
