"""Benchmark helpers for root-level PUCT search on recorded rollout prefixes."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Callable, Mapping

from .env import PlayerId, PokeZeroEnv
from .observation import PokeZeroObservationV0
from .policy import Policy
from .replay_benchmark import replay_prefix_counts
from .rollout import RolloutConfig, RolloutDriver
from .search import ActionPriorVector, ObservationValueFunction, player_observation_history, puct_branch_search

ActionPriorFunction = Callable[[tuple[PokeZeroObservationV0, ...]], ActionPriorVector]


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
    elapsed_seconds: float

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
            "elapsed_seconds": self.elapsed_seconds,
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

    def to_dict(self) -> dict[str, object]:
        return {
            "format_id": self.format_id,
            "max_decision_rounds": self.max_decision_rounds,
            "games": self.games,
            "prefixes_per_game": self.prefixes_per_game,
            "search_player": self.search_player,
            "cpuct": self.cpuct,
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
) -> RootPUCTSearchBenchmarkReport:
    if games <= 0:
        raise ValueError("games must be positive.")
    if prefixes_per_game <= 0:
        raise ValueError("prefixes_per_game must be positive.")
    if search_player not in policies:
        raise ValueError(f"search_player {search_player!r} is not present in source policies.")

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
                priors = prior_fn(history)
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
                )
                elapsed = perf_counter() - start
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
                        elapsed_seconds=elapsed,
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
    )
