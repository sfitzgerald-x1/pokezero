"""Benchmark helpers for root-level PUCT search on recorded rollout prefixes."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from time import perf_counter as _timing_perf_counter
from typing import Callable, Mapping

from .env import PlayerId, PokeZeroEnv
from .observation import PokeZeroObservationV0
from .policy import Policy
from .replay_branching import replay_trajectory_branch_rollout
from .replay_benchmark import replay_prefix_counts
from .rollout import RolloutConfig, RolloutDriver
from .search import (
    ActionPriorVector,
    ObservationValueFunction,
    RootPUCTSearchTiming,
    player_observation_history,
    puct_branch_search,
    terminal_value_for_player,
)

ActionPriorFunction = Callable[[tuple[PokeZeroObservationV0, ...]], ActionPriorVector]
ROOT_PUCT_SEARCH_BENCHMARK_SCHEMA_VERSION = "pokezero.root-puct-search-benchmark.v2"


@dataclass(frozen=True)
class _BranchRolloutEvaluation:
    value: float
    decision_rounds: int
    winner: PlayerId | None
    capped: bool


@dataclass(frozen=True)
class RootPUCTSearchDecision:
    seed: int
    battle_id: str
    player_id: PlayerId
    prefix_decision_round_count: int
    recorded_action_index: int
    selected_action_index: int
    selected_value: float
    selected_score: float
    candidate_count: int
    total_visits: int
    elapsed_seconds: float
    timing: RootPUCTSearchTiming

    @property
    def changed_action(self) -> bool:
        return self.selected_action_index != self.recorded_action_index

    def to_dict(self) -> dict[str, object]:
        return {
            "seed": self.seed,
            "battle_id": self.battle_id,
            "player_id": self.player_id,
            "prefix_decision_round_count": self.prefix_decision_round_count,
            "recorded_action_index": self.recorded_action_index,
            "selected_action_index": self.selected_action_index,
            "changed_action": self.changed_action,
            "selected_value": self.selected_value,
            "selected_score": self.selected_score,
            "candidate_count": self.candidate_count,
            "total_visits": self.total_visits,
            "elapsed_seconds": self.elapsed_seconds,
            "timing": self.timing.to_dict(),
        }


@dataclass(frozen=True)
class RootPUCTSearchBenchmarkReport:
    format_id: str
    max_decision_rounds: int
    games: int
    prefixes_per_game: int
    search_player: PlayerId
    cpuct: float
    source_policy_ids: Mapping[PlayerId, str]
    source_decision_rounds: tuple[int, ...]
    decisions: tuple[RootPUCTSearchDecision, ...]
    skipped_prefixes: int = 0
    root_extra_visits: int | None = None

    @property
    def evaluated_prefixes(self) -> int:
        return len(self.decisions)

    @property
    def changed_actions(self) -> int:
        return sum(1 for decision in self.decisions if decision.changed_action)

    @property
    def action_change_rate(self) -> float:
        return self.changed_actions / self.evaluated_prefixes if self.evaluated_prefixes else 0.0

    @property
    def total_elapsed_seconds(self) -> float:
        return sum(decision.elapsed_seconds for decision in self.decisions)

    @property
    def average_elapsed_seconds(self) -> float:
        return self.total_elapsed_seconds / self.evaluated_prefixes if self.evaluated_prefixes else 0.0

    @property
    def average_candidate_count(self) -> float:
        return sum(decision.candidate_count for decision in self.decisions) / self.evaluated_prefixes if self.evaluated_prefixes else 0.0

    @property
    def average_source_decision_rounds(self) -> float:
        return sum(self.source_decision_rounds) / len(self.source_decision_rounds) if self.source_decision_rounds else 0.0

    @property
    def timing(self) -> RootPUCTSearchTiming:
        return RootPUCTSearchTiming.aggregate(tuple(decision.timing for decision in self.decisions))

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": ROOT_PUCT_SEARCH_BENCHMARK_SCHEMA_VERSION,
            "format_id": self.format_id,
            "max_decision_rounds": self.max_decision_rounds,
            "games": self.games,
            "prefixes_per_game": self.prefixes_per_game,
            "search_player": self.search_player,
            "cpuct": self.cpuct,
            "root_extra_visits": self.root_extra_visits,
            "source_policy_ids": dict(self.source_policy_ids),
            "source_decision_rounds": list(self.source_decision_rounds),
            "average_source_decision_rounds": self.average_source_decision_rounds,
            "evaluated_prefixes": self.evaluated_prefixes,
            "skipped_prefixes": self.skipped_prefixes,
            "changed_actions": self.changed_actions,
            "action_change_rate": self.action_change_rate,
            "total_elapsed_seconds": self.total_elapsed_seconds,
            "average_elapsed_seconds": self.average_elapsed_seconds,
            "average_candidate_count": self.average_candidate_count,
            "timing": self.timing.to_dict(),
            "decisions": [decision.to_dict() for decision in self.decisions],
        }


@dataclass(frozen=True)
class RootPUCTCounterfactualDecision:
    seed: int
    battle_id: str
    player_id: PlayerId
    prefix_decision_round_count: int
    recorded_action_index: int
    selected_action_index: int
    selected_search_value: float
    selected_search_score: float
    candidate_count: int
    search_elapsed_seconds: float
    recorded_rollout_value: float
    selected_rollout_value: float
    recorded_rollout_decision_rounds: int
    selected_rollout_decision_rounds: int
    recorded_winner: PlayerId | None
    selected_winner: PlayerId | None
    recorded_capped: bool
    selected_capped: bool
    rollout_elapsed_seconds: float
    search_timing: RootPUCTSearchTiming

    @property
    def changed_action(self) -> bool:
        return self.selected_action_index != self.recorded_action_index

    @property
    def rollout_value_delta(self) -> float:
        return self.selected_rollout_value - self.recorded_rollout_value

    def to_dict(self) -> dict[str, object]:
        return {
            "seed": self.seed,
            "battle_id": self.battle_id,
            "player_id": self.player_id,
            "prefix_decision_round_count": self.prefix_decision_round_count,
            "recorded_action_index": self.recorded_action_index,
            "selected_action_index": self.selected_action_index,
            "changed_action": self.changed_action,
            "selected_search_value": self.selected_search_value,
            "selected_search_score": self.selected_search_score,
            "candidate_count": self.candidate_count,
            "search_elapsed_seconds": self.search_elapsed_seconds,
            "recorded_rollout_value": self.recorded_rollout_value,
            "selected_rollout_value": self.selected_rollout_value,
            "rollout_value_delta": self.rollout_value_delta,
            "recorded_rollout_decision_rounds": self.recorded_rollout_decision_rounds,
            "selected_rollout_decision_rounds": self.selected_rollout_decision_rounds,
            "recorded_winner": self.recorded_winner,
            "selected_winner": self.selected_winner,
            "recorded_capped": self.recorded_capped,
            "selected_capped": self.selected_capped,
            "rollout_elapsed_seconds": self.rollout_elapsed_seconds,
            "search_timing": self.search_timing.to_dict(),
        }


@dataclass(frozen=True)
class RootPUCTCounterfactualBenchmarkReport:
    format_id: str
    max_decision_rounds: int
    games: int
    prefixes_per_game: int
    search_player: PlayerId
    cpuct: float
    source_policy_ids: Mapping[PlayerId, str]
    continuation_policy_ids: Mapping[PlayerId, str]
    source_decision_rounds: tuple[int, ...]
    decisions: tuple[RootPUCTCounterfactualDecision, ...]
    skipped_prefixes: int = 0

    @property
    def evaluated_prefixes(self) -> int:
        return len(self.decisions)

    @property
    def changed_actions(self) -> int:
        return sum(1 for decision in self.decisions if decision.changed_action)

    @property
    def action_change_rate(self) -> float:
        return self.changed_actions / self.evaluated_prefixes if self.evaluated_prefixes else 0.0

    @property
    def improved_actions(self) -> int:
        return sum(1 for decision in self.decisions if decision.rollout_value_delta > 0.0)

    @property
    def worsened_actions(self) -> int:
        return sum(1 for decision in self.decisions if decision.rollout_value_delta < 0.0)

    @property
    def tied_actions(self) -> int:
        return sum(1 for decision in self.decisions if decision.rollout_value_delta == 0.0)

    @property
    def average_rollout_value_delta(self) -> float:
        return (
            sum(decision.rollout_value_delta for decision in self.decisions) / self.evaluated_prefixes
            if self.evaluated_prefixes
            else 0.0
        )

    @property
    def average_recorded_rollout_value(self) -> float:
        return (
            sum(decision.recorded_rollout_value for decision in self.decisions) / self.evaluated_prefixes
            if self.evaluated_prefixes
            else 0.0
        )

    @property
    def average_selected_rollout_value(self) -> float:
        return (
            sum(decision.selected_rollout_value for decision in self.decisions) / self.evaluated_prefixes
            if self.evaluated_prefixes
            else 0.0
        )

    @property
    def total_search_elapsed_seconds(self) -> float:
        return sum(decision.search_elapsed_seconds for decision in self.decisions)

    @property
    def average_search_elapsed_seconds(self) -> float:
        return self.total_search_elapsed_seconds / self.evaluated_prefixes if self.evaluated_prefixes else 0.0

    @property
    def total_rollout_elapsed_seconds(self) -> float:
        return sum(decision.rollout_elapsed_seconds for decision in self.decisions)

    @property
    def average_rollout_elapsed_seconds(self) -> float:
        return self.total_rollout_elapsed_seconds / self.evaluated_prefixes if self.evaluated_prefixes else 0.0

    @property
    def average_candidate_count(self) -> float:
        return sum(decision.candidate_count for decision in self.decisions) / self.evaluated_prefixes if self.evaluated_prefixes else 0.0

    @property
    def average_source_decision_rounds(self) -> float:
        return sum(self.source_decision_rounds) / len(self.source_decision_rounds) if self.source_decision_rounds else 0.0

    @property
    def search_timing(self) -> RootPUCTSearchTiming:
        return RootPUCTSearchTiming.aggregate(tuple(decision.search_timing for decision in self.decisions))

    def to_dict(self) -> dict[str, object]:
        return {
            "format_id": self.format_id,
            "max_decision_rounds": self.max_decision_rounds,
            "games": self.games,
            "prefixes_per_game": self.prefixes_per_game,
            "search_player": self.search_player,
            "cpuct": self.cpuct,
            "source_policy_ids": dict(self.source_policy_ids),
            "continuation_policy_ids": dict(self.continuation_policy_ids),
            "source_decision_rounds": list(self.source_decision_rounds),
            "average_source_decision_rounds": self.average_source_decision_rounds,
            "evaluated_prefixes": self.evaluated_prefixes,
            "skipped_prefixes": self.skipped_prefixes,
            "changed_actions": self.changed_actions,
            "action_change_rate": self.action_change_rate,
            "improved_actions": self.improved_actions,
            "worsened_actions": self.worsened_actions,
            "tied_actions": self.tied_actions,
            "average_rollout_value_delta": self.average_rollout_value_delta,
            "average_recorded_rollout_value": self.average_recorded_rollout_value,
            "average_selected_rollout_value": self.average_selected_rollout_value,
            "total_search_elapsed_seconds": self.total_search_elapsed_seconds,
            "average_search_elapsed_seconds": self.average_search_elapsed_seconds,
            "total_rollout_elapsed_seconds": self.total_rollout_elapsed_seconds,
            "average_rollout_elapsed_seconds": self.average_rollout_elapsed_seconds,
            "average_candidate_count": self.average_candidate_count,
            "search_timing": self.search_timing.to_dict(),
            "decisions": [decision.to_dict() for decision in self.decisions],
        }


def benchmark_root_puct_search(
    *,
    env_factory: Callable[[], PokeZeroEnv],
    policies: Mapping[PlayerId, Policy],
    rollout_config: RolloutConfig,
    games: int,
    value_fn: ObservationValueFunction,
    prior_fn: ActionPriorFunction,
    prefixes_per_game: int = 5,
    seed_start: int = 1,
    search_player: PlayerId = "p1",
    cpuct: float = 1.25,
    root_extra_visits: int | None = None,
) -> RootPUCTSearchBenchmarkReport:
    if games <= 0:
        raise ValueError("games must be positive.")
    if prefixes_per_game <= 0:
        raise ValueError("prefixes_per_game must be positive.")
    if search_player not in policies:
        raise ValueError(f"search_player {search_player!r} is not present in source policies.")
    if root_extra_visits is not None and root_extra_visits < 0:
        raise ValueError("root_extra_visits must be non-negative when set.")

    env = env_factory()
    source_decision_rounds: list[int] = []
    decisions: list[RootPUCTSearchDecision] = []
    skipped_prefixes = 0
    try:
        for game_index in range(games):
            seed = seed_start + game_index
            source = RolloutDriver(
                env=env,
                policies=policies,
                config=rollout_config,
            ).run(seed=seed, battle_id=f"root-puct-source-{seed}")
            source_decision_rounds.append(source.decision_round_count)
            for prefix_decision_round_count in replay_prefix_counts(
                source.decision_round_count,
                prefixes_per_game=prefixes_per_game,
            ):
                steps = source.trajectory.steps_for_turn(prefix_decision_round_count)
                player_step = next((step for step in steps if step.player_id == search_player), None)
                if player_step is None:
                    skipped_prefixes += 1
                    continue
                timing_started_at = _timing_perf_counter()
                opponent_actions = {
                    step.player_id: step.action_index
                    for step in steps
                    if step.player_id != search_player
                }
                history = player_observation_history(
                    source.trajectory,
                    player_id=search_player,
                    through_decision_round=prefix_decision_round_count,
                )
                policy_evaluation_started_at = _timing_perf_counter()
                priors = prior_fn(history)
                policy_evaluation_seconds = _timing_perf_counter() - policy_evaluation_started_at
                start = perf_counter()
                search = puct_branch_search(
                    env=env,
                    trajectory=source.trajectory,
                    player_id=search_player,
                    prefix_decision_round_count=prefix_decision_round_count,
                    legal_action_mask=player_step.legal_action_mask,
                    opponent_actions=opponent_actions,
                    value_fn=value_fn,
                    action_priors=priors,
                    cpuct=cpuct,
                    root_visit_budget_resolver=(
                        None
                        if root_extra_visits is None
                        else lambda budget_context: len(budget_context.action_priors) + root_extra_visits
                    ),
                )
                elapsed = perf_counter() - start
                timing = search.timing.with_policy_evaluation(policy_evaluation_seconds).with_total(
                    _timing_perf_counter() - timing_started_at
                )
                decisions.append(
                    RootPUCTSearchDecision(
                        seed=seed,
                        battle_id=source.trajectory.battle_id,
                        player_id=search_player,
                        prefix_decision_round_count=prefix_decision_round_count,
                        recorded_action_index=player_step.action_index,
                        selected_action_index=search.action_index,
                        selected_value=search.best_candidate.value,
                        selected_score=search.best_candidate.score,
                        candidate_count=len(search.candidates),
                        total_visits=search.total_visits,
                        elapsed_seconds=elapsed,
                        timing=timing,
                    )
                )
    finally:
        close = getattr(env, "close", None)
        if callable(close):
            close()

    return RootPUCTSearchBenchmarkReport(
        format_id=rollout_config.format_id,
        max_decision_rounds=rollout_config.max_decision_rounds,
        games=games,
        prefixes_per_game=prefixes_per_game,
        search_player=search_player,
        cpuct=cpuct,
        source_policy_ids={player: policy.policy_id for player, policy in policies.items()},
        source_decision_rounds=tuple(source_decision_rounds),
        decisions=tuple(decisions),
        skipped_prefixes=skipped_prefixes,
        root_extra_visits=root_extra_visits,
    )


def benchmark_root_puct_counterfactual_rollouts(
    *,
    env_factory: Callable[[], PokeZeroEnv],
    policies: Mapping[PlayerId, Policy],
    rollout_config: RolloutConfig,
    games: int,
    value_fn: ObservationValueFunction,
    prior_fn: ActionPriorFunction,
    continuation_policies: Mapping[PlayerId, Policy] | None = None,
    prefixes_per_game: int = 5,
    seed_start: int = 1,
    search_player: PlayerId = "p1",
    cpuct: float = 1.25,
) -> RootPUCTCounterfactualBenchmarkReport:
    """Compare recorded vs root-PUCT-selected branch outcomes on sampled source prefixes."""

    if games <= 0:
        raise ValueError("games must be positive.")
    if prefixes_per_game <= 0:
        raise ValueError("prefixes_per_game must be positive.")
    if search_player not in policies:
        raise ValueError(f"search_player {search_player!r} is not present in source policies.")
    continuation = continuation_policies or policies
    if search_player not in continuation:
        raise ValueError(f"search_player {search_player!r} is not present in continuation policies.")

    env = env_factory()
    source_decision_rounds: list[int] = []
    decisions: list[RootPUCTCounterfactualDecision] = []
    skipped_prefixes = 0
    try:
        for game_index in range(games):
            seed = seed_start + game_index
            source = RolloutDriver(
                env=env,
                policies=policies,
                config=rollout_config,
            ).run(seed=seed, battle_id=f"root-puct-counterfactual-source-{seed}")
            source_decision_rounds.append(source.decision_round_count)
            for prefix_decision_round_count in replay_prefix_counts(
                source.decision_round_count,
                prefixes_per_game=prefixes_per_game,
            ):
                steps = source.trajectory.steps_for_turn(prefix_decision_round_count)
                player_step = next((step for step in steps if step.player_id == search_player), None)
                if player_step is None:
                    skipped_prefixes += 1
                    continue
                search_timing_started_at = _timing_perf_counter()
                opponent_actions = {
                    step.player_id: step.action_index
                    for step in steps
                    if step.player_id != search_player
                }
                history = player_observation_history(
                    source.trajectory,
                    player_id=search_player,
                    through_decision_round=prefix_decision_round_count,
                )
                policy_evaluation_started_at = _timing_perf_counter()
                priors = prior_fn(history)
                policy_evaluation_seconds = _timing_perf_counter() - policy_evaluation_started_at
                search_start = perf_counter()
                search = puct_branch_search(
                    env=env,
                    trajectory=source.trajectory,
                    player_id=search_player,
                    prefix_decision_round_count=prefix_decision_round_count,
                    legal_action_mask=player_step.legal_action_mask,
                    opponent_actions=opponent_actions,
                    value_fn=value_fn,
                    action_priors=priors,
                    cpuct=cpuct,
                )
                search_elapsed = perf_counter() - search_start
                search_timing = search.timing.with_policy_evaluation(policy_evaluation_seconds).with_total(
                    _timing_perf_counter() - search_timing_started_at
                )
                rollout_start = perf_counter()
                recorded = _branch_rollout_value(
                    env=env,
                    source_trajectory=source.trajectory,
                    player_id=search_player,
                    prefix_decision_round_count=prefix_decision_round_count,
                    action_index=player_step.action_index,
                    opponent_actions=opponent_actions,
                    continuation_policies=continuation,
                    rollout_config=rollout_config,
                    battle_id_suffix="recorded",
                )
                if search.action_index == player_step.action_index:
                    selected = recorded
                else:
                    selected = _branch_rollout_value(
                        env=env,
                        source_trajectory=source.trajectory,
                        player_id=search_player,
                        prefix_decision_round_count=prefix_decision_round_count,
                        action_index=search.action_index,
                        opponent_actions=opponent_actions,
                        continuation_policies=continuation,
                        rollout_config=rollout_config,
                        battle_id_suffix="selected",
                    )
                rollout_elapsed = perf_counter() - rollout_start
                decisions.append(
                    RootPUCTCounterfactualDecision(
                        seed=seed,
                        battle_id=source.trajectory.battle_id,
                        player_id=search_player,
                        prefix_decision_round_count=prefix_decision_round_count,
                        recorded_action_index=player_step.action_index,
                        selected_action_index=search.action_index,
                        selected_search_value=search.best_candidate.value,
                        selected_search_score=search.best_candidate.score,
                        candidate_count=len(search.candidates),
                        search_elapsed_seconds=search_elapsed,
                        recorded_rollout_value=recorded.value,
                        selected_rollout_value=selected.value,
                        recorded_rollout_decision_rounds=recorded.decision_rounds,
                        selected_rollout_decision_rounds=selected.decision_rounds,
                        recorded_winner=recorded.winner,
                        selected_winner=selected.winner,
                        recorded_capped=recorded.capped,
                        selected_capped=selected.capped,
                        rollout_elapsed_seconds=rollout_elapsed,
                        search_timing=search_timing,
                    )
                )
    finally:
        close = getattr(env, "close", None)
        if callable(close):
            close()

    return RootPUCTCounterfactualBenchmarkReport(
        format_id=rollout_config.format_id,
        max_decision_rounds=rollout_config.max_decision_rounds,
        games=games,
        prefixes_per_game=prefixes_per_game,
        search_player=search_player,
        cpuct=cpuct,
        source_policy_ids={player: policy.policy_id for player, policy in policies.items()},
        continuation_policy_ids={player: policy.policy_id for player, policy in continuation.items()},
        source_decision_rounds=tuple(source_decision_rounds),
        decisions=tuple(decisions),
        skipped_prefixes=skipped_prefixes,
    )


def _branch_rollout_value(
    *,
    env: PokeZeroEnv,
    source_trajectory,
    player_id: PlayerId,
    prefix_decision_round_count: int,
    action_index: int,
    opponent_actions: Mapping[PlayerId, int],
    continuation_policies: Mapping[PlayerId, Policy],
    rollout_config: RolloutConfig,
    battle_id_suffix: str,
) -> _BranchRolloutEvaluation:
    branch_actions = {
        **dict(opponent_actions),
        player_id: action_index,
    }
    rollout = replay_trajectory_branch_rollout(
        env,
        source_trajectory,
        prefix_decision_round_count=prefix_decision_round_count,
        branch_actions=branch_actions,
        policies=continuation_policies,
        rollout_config=rollout_config,
        battle_id=f"root-puct-counterfactual-{battle_id_suffix}-{player_id}-{prefix_decision_round_count}-{action_index}",
    )
    terminal = rollout.continuation.terminal
    return _BranchRolloutEvaluation(
        value=terminal_value_for_player(terminal, player_id=player_id),
        decision_rounds=rollout.continuation.decision_round_count,
        winner=terminal.winner,
        capped=terminal.capped,
    )
