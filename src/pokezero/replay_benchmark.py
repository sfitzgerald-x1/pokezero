"""Replay-from-root timing helpers for search planning."""

from __future__ import annotations

from dataclasses import dataclass
import math
from time import perf_counter
from typing import Callable, Mapping

from .env import PlayerId, PokeZeroEnv
from .policy import Policy
from .replay_branching import replay_trajectory_prefix
from .rollout import RolloutConfig, RolloutDriver


@dataclass(frozen=True)
class ReplayPrefixTiming:
    seed: int
    decision_round_count: int
    elapsed_seconds: float
    requested_players: tuple[PlayerId, ...]
    terminal: bool

    def to_dict(self) -> dict[str, object]:
        return {
            "seed": self.seed,
            "decision_round_count": self.decision_round_count,
            "elapsed_seconds": self.elapsed_seconds,
            "requested_players": list(self.requested_players),
            "terminal": self.terminal,
        }


@dataclass(frozen=True)
class ReplayPrefixBenchmarkReport:
    format_id: str
    max_decision_rounds: int
    games: int
    prefixes_per_game: int
    source_policy_ids: Mapping[PlayerId, str]
    source_decision_rounds: tuple[int, ...]
    timings: tuple[ReplayPrefixTiming, ...]

    @property
    def total_prefixes(self) -> int:
        return len(self.timings)

    @property
    def total_replay_elapsed_seconds(self) -> float:
        return sum(timing.elapsed_seconds for timing in self.timings)

    @property
    def average_replay_seconds(self) -> float:
        return self.total_replay_elapsed_seconds / self.total_prefixes if self.total_prefixes else 0.0

    @property
    def median_replay_seconds(self) -> float:
        return _percentile(_elapsed_values(self.timings), 0.50)

    @property
    def p95_replay_seconds(self) -> float:
        return _percentile(_elapsed_values(self.timings), 0.95)

    @property
    def max_replay_seconds(self) -> float:
        values = _elapsed_values(self.timings)
        return max(values) if values else 0.0

    @property
    def average_source_decision_rounds(self) -> float:
        return sum(self.source_decision_rounds) / len(self.source_decision_rounds) if self.source_decision_rounds else 0.0

    @property
    def average_replayed_decision_rounds(self) -> float:
        return (
            sum(timing.decision_round_count for timing in self.timings) / self.total_prefixes
            if self.total_prefixes
            else 0.0
        )

    @property
    def replayed_decision_rounds_per_second(self) -> float:
        elapsed = self.total_replay_elapsed_seconds
        return sum(timing.decision_round_count for timing in self.timings) / elapsed if elapsed > 0 else 0.0

    def to_dict(self) -> dict[str, object]:
        return {
            "format_id": self.format_id,
            "max_decision_rounds": self.max_decision_rounds,
            "games": self.games,
            "prefixes_per_game": self.prefixes_per_game,
            "source_policy_ids": dict(self.source_policy_ids),
            "source_decision_rounds": list(self.source_decision_rounds),
            "average_source_decision_rounds": self.average_source_decision_rounds,
            "total_prefixes": self.total_prefixes,
            "total_replay_elapsed_seconds": self.total_replay_elapsed_seconds,
            "average_replay_seconds": self.average_replay_seconds,
            "median_replay_seconds": self.median_replay_seconds,
            "p95_replay_seconds": self.p95_replay_seconds,
            "max_replay_seconds": self.max_replay_seconds,
            "average_replayed_decision_rounds": self.average_replayed_decision_rounds,
            "replayed_decision_rounds_per_second": self.replayed_decision_rounds_per_second,
            "timings": [timing.to_dict() for timing in self.timings],
        }


def benchmark_replay_prefixes(
    *,
    env_factory: Callable[[], PokeZeroEnv],
    policies: Mapping[PlayerId, Policy],
    rollout_config: RolloutConfig,
    games: int,
    prefixes_per_game: int = 5,
    seed_start: int = 1,
) -> ReplayPrefixBenchmarkReport:
    if games <= 0:
        raise ValueError("games must be positive.")
    if prefixes_per_game <= 0:
        raise ValueError("prefixes_per_game must be positive.")
    if not policies:
        raise ValueError("at least one source policy is required.")

    env = env_factory()
    source_decision_rounds: list[int] = []
    timings: list[ReplayPrefixTiming] = []
    try:
        for game_index in range(games):
            seed = seed_start + game_index
            result = RolloutDriver(
                env=env,
                policies=policies,
                config=rollout_config,
            ).run(seed=seed, battle_id=f"replay-benchmark-{seed}")
            source_decision_rounds.append(result.decision_round_count)
            for decision_round_count in replay_prefix_counts(
                result.decision_round_count,
                prefixes_per_game=prefixes_per_game,
            ):
                start = perf_counter()
                prefix_result = replay_trajectory_prefix(
                    env,
                    result.trajectory,
                    decision_round_count=decision_round_count,
                )
                elapsed = perf_counter() - start
                timings.append(
                    ReplayPrefixTiming(
                        seed=seed,
                        decision_round_count=decision_round_count,
                        elapsed_seconds=elapsed,
                        requested_players=prefix_result.requested_players,
                        terminal=prefix_result.terminal is not None,
                    )
                )
    finally:
        close = getattr(env, "close", None)
        if callable(close):
            close()

    return ReplayPrefixBenchmarkReport(
        format_id=rollout_config.format_id,
        max_decision_rounds=rollout_config.max_decision_rounds,
        games=games,
        prefixes_per_game=prefixes_per_game,
        source_policy_ids={player: policy.policy_id for player, policy in policies.items()},
        source_decision_rounds=tuple(source_decision_rounds),
        timings=tuple(timings),
    )


def replay_prefix_counts(decision_round_count: int, *, prefixes_per_game: int) -> tuple[int, ...]:
    """Evenly sample valid branch-prefix lengths for one completed source trajectory."""

    if decision_round_count < 0:
        raise ValueError("decision_round_count must be non-negative.")
    if prefixes_per_game <= 0:
        raise ValueError("prefixes_per_game must be positive.")
    if decision_round_count == 0:
        return ()
    selected_count = min(prefixes_per_game, decision_round_count)
    if selected_count == 1:
        return (decision_round_count // 2,)
    max_prefix = decision_round_count - 1
    return tuple(
        sorted(
            {
                round(index * max_prefix / (selected_count - 1))
                for index in range(selected_count)
            }
        )
    )


def _elapsed_values(timings: tuple[ReplayPrefixTiming, ...]) -> tuple[float, ...]:
    return tuple(sorted(timing.elapsed_seconds for timing in timings))


def _percentile(values: tuple[float, ...], quantile: float) -> float:
    if not values:
        return 0.0
    index = min(len(values) - 1, max(0, math.ceil(quantile * len(values)) - 1))
    return values[index]
