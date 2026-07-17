"""Rollout collection and JSONL persistence helpers."""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
from pathlib import Path
import sys
import threading
from time import perf_counter
from typing import TYPE_CHECKING, Any, Callable, Iterable, Iterator, Mapping, TextIO
from urllib.parse import parse_qsl, urlencode

from .env import PokeZeroEnv, TerminalState
from .mcts_diagnostics import root_puct_fallback_category
from .root_puct_telemetry import root_puct_decision_telemetry
from .policy import MaxDamagePolicy, Policy, RandomLegalPolicy, ScriptedTeacherPolicy, SimpleLegalPolicy
from .rollout import RolloutConfig, RolloutDriver, RolloutResult
from .trajectory import BattleTrajectory, trajectory_from_dict, trajectory_to_dict

if TYPE_CHECKING:
    from .dataset import TrajectoryDatasetConfig, TrainingCacheSummary
    from .linear_policy import LinearPolicyModel

# v2: records carry observation-spec-v2 tensors (window-1 + transition tokens). Bumped with the
# observation break so pre-break JSONL refuses at the record guard instead of failing shape-wise.
ROLLOUT_RECORD_SCHEMA_VERSION = "pokezero.rollout_record.v2"
LINEAR_POLICY_SPEC_PREFIX = "linear:"
NEURAL_POLICY_SPEC_PREFIX = "neural:"
REMOTE_POLICY_SPEC_PREFIX = "remote:"


@dataclass(frozen=True)
class BenchmarkMatchup:
    label: str
    p1_policy: Policy
    p2_policy: Policy

    def __post_init__(self) -> None:
        if not self.label.strip():
            raise ValueError("benchmark matchup label must be non-empty.")


@dataclass(frozen=True)
class BenchmarkProgress:
    """One completed game for opt-in benchmark progress reporting.

    Progress stays out-of-band from the persisted report so job liveness can be
    observed without changing the reproducible result artifact.
    """

    matchup_label: str
    matchup_index: int
    matchup_count: int
    games_completed: int
    games_total: int
    seed: int
    matchup_elapsed_seconds: float
    # Compact current-game diagnostics let long-running search benchmarks
    # report a cumulative fallback taxonomy without changing result artifacts.
    root_puct_by_player: Mapping[str, Mapping[str, Any]] = field(default_factory=dict)


@dataclass(frozen=True)
class BenchmarkGameResult:
    """Compact per-seed benchmark evidence for paired evaluation.

    Full trajectories remain in-memory rollout artifacts; this deliberately retains only the
    outcome, elapsed time, and root-search diagnostics needed to compare independently run arms
    on the same seed without serializing private observations or action histories.
    """

    seed: int
    battle_id: str
    winner: str | None
    capped: bool
    decision_rounds: int
    elapsed_seconds: float
    root_puct_by_player: Mapping[str, Mapping[str, Any]] = field(default_factory=dict)
    # Compact, per-decision Root-PUCT telemetry.  This retains stable search
    # diagnostics without serializing observations, actions, or raw errors.
    root_puct_decision_telemetry_by_player: Mapping[str, tuple[Mapping[str, Any], ...]] = field(
        default_factory=dict
    )
    opponent_legal_mask_mode: str = "privileged"
    # Full policy-dispatch wall samples, present only when the rollout config
    # explicitly opts in. These are distinct from root-PUCT's internal timer.
    policy_elapsed_seconds_by_player: Mapping[str, tuple[float, ...]] = field(default_factory=dict)

    @property
    def tied(self) -> bool:
        return self.winner is None and not self.capped

    def score_for(self, player_id: str) -> float:
        if self.winner == player_id:
            return 1.0
        if self.tied or self.capped:
            return 0.5
        return 0.0

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "seed": self.seed,
            "battle_id": self.battle_id,
            "winner": self.winner,
            "tied": self.tied,
            "capped": self.capped,
            "decision_rounds": self.decision_rounds,
            "elapsed_seconds": self.elapsed_seconds,
            "p1_score": self.score_for("p1"),
            "p2_score": self.score_for("p2"),
            "opponent_legal_mask_mode": self.opponent_legal_mask_mode,
        }
        if self.root_puct_by_player:
            payload["root_puct_by_player"] = {
                player: dict(diagnostics)
                for player, diagnostics in sorted(self.root_puct_by_player.items())
            }
        if self.root_puct_decision_telemetry_by_player:
            payload["root_puct_decision_telemetry_by_player"] = {
                player: [dict(item) for item in diagnostics]
                for player, diagnostics in sorted(self.root_puct_decision_telemetry_by_player.items())
            }
        if self.policy_elapsed_seconds_by_player:
            payload["policy_elapsed_seconds_by_player"] = {
                player: list(samples)
                for player, samples in sorted(self.policy_elapsed_seconds_by_player.items())
            }
        return payload


@dataclass(frozen=True)
class BenchmarkMatchupResult:
    label: str
    p1_policy_id: str
    p2_policy_id: str
    seed_start: int
    metrics: "CollectionMetrics"
    p1_policy_provenance: Mapping[str, Any] | None = None
    p2_policy_provenance: Mapping[str, Any] | None = None
    root_puct_belief_public_checksums_by_seed: Mapping[int, tuple[str, ...]] | None = None
    game_results: tuple[BenchmarkGameResult, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        result = {
            "label": self.label,
            "p1_policy_id": self.p1_policy_id,
            "p2_policy_id": self.p2_policy_id,
            "p1_policy_provenance": dict(self.p1_policy_provenance or {}),
            "p2_policy_provenance": dict(self.p2_policy_provenance or {}),
            "seed_start": self.seed_start,
            "metrics": self.metrics.to_dict(),
        }
        if self.root_puct_belief_public_checksums_by_seed:
            result["root_puct_belief_public_checksums_by_seed"] = {
                str(seed): list(checksums)
                for seed, checksums in sorted(self.root_puct_belief_public_checksums_by_seed.items())
            }
        if self.game_results:
            result["game_results"] = [game.to_dict() for game in self.game_results]
        return result


@dataclass(frozen=True)
class BenchmarkHeadToHeadResult:
    label: str
    first_policy_id: str
    second_policy_id: str
    games: int
    first_policy_wins: int
    second_policy_wins: int
    ties: int
    capped_games: int

    @property
    def first_policy_win_rate(self) -> float:
        return self.first_policy_wins / self.games if self.games else 0.0

    @property
    def second_policy_win_rate(self) -> float:
        return self.second_policy_wins / self.games if self.games else 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "first_policy_id": self.first_policy_id,
            "second_policy_id": self.second_policy_id,
            "games": self.games,
            "first_policy_wins": self.first_policy_wins,
            "second_policy_wins": self.second_policy_wins,
            "ties": self.ties,
            "capped_games": self.capped_games,
            "first_policy_win_rate": self.first_policy_win_rate,
            "second_policy_win_rate": self.second_policy_win_rate,
        }


@dataclass(frozen=True)
class BenchmarkReport:
    format_id: str
    max_decision_rounds: int
    games_per_matchup: int
    matchups: tuple[BenchmarkMatchupResult, ...]
    policy_provenance: Mapping[str, Mapping[str, Any]] | None = None

    @property
    def total_games(self) -> int:
        return sum(result.metrics.games for result in self.matchups)

    @property
    def elapsed_seconds(self) -> float:
        return sum(result.metrics.elapsed_seconds for result in self.matchups)

    @property
    def total_decision_rounds(self) -> int:
        return sum(result.metrics.total_decision_rounds for result in self.matchups)

    @property
    def games_per_second(self) -> float:
        return self.total_games / self.elapsed_seconds if self.elapsed_seconds > 0 else 0.0

    @property
    def decisions_per_second(self) -> float:
        return self.total_decision_rounds / self.elapsed_seconds if self.elapsed_seconds > 0 else 0.0

    @property
    def average_decision_rounds(self) -> float:
        return self.total_decision_rounds / self.total_games if self.total_games else 0.0

    @property
    def peak_rss_mb(self) -> float | None:
        values = tuple(
            result.metrics.peak_rss_mb
            for result in self.matchups
            if result.metrics.peak_rss_mb is not None
        )
        return max(values) if values else None

    @property
    def head_to_head_results(self) -> tuple[BenchmarkHeadToHeadResult, ...]:
        return aggregate_benchmark_head_to_heads(self.matchups)

    def to_dict(self) -> dict[str, Any]:
        return {
            "format_id": self.format_id,
            "max_decision_rounds": self.max_decision_rounds,
            "games_per_matchup": self.games_per_matchup,
            "total_games": self.total_games,
            "elapsed_seconds": self.elapsed_seconds,
            "games_per_second": self.games_per_second,
            "decisions_per_second": self.decisions_per_second,
            "average_decision_rounds": self.average_decision_rounds,
            **({"peak_rss_mb": self.peak_rss_mb} if self.peak_rss_mb is not None else {}),
            "policy_provenance": {
                policy_id: dict(provenance)
                for policy_id, provenance in sorted((self.policy_provenance or {}).items())
            },
            "matchups": [result.to_dict() for result in self.matchups],
            "head_to_heads": [result.to_dict() for result in self.head_to_head_results],
        }


@dataclass(frozen=True)
class RolloutRecord:
    battle_id: str
    seed: int
    format_id: str
    policy_ids: Mapping[str, str]
    decision_round_count: int
    elapsed_seconds: float
    terminal: TerminalState
    trajectory: BattleTrajectory
    # Belief-system provenance: the candidate-set source_hash the collecting env encoded
    # observations with (None = source disabled or pre-provenance record). Flows into checkpoint
    # metadata at train time so eval can match observation conditions to training.
    belief_set_source_hash: str | None = None
    # In-memory-only per-phase collection timing sidecar (see rollout.RolloutResult.timing).
    # Deliberately NOT serialized by rollout_record_to_dict — record payloads must stay
    # deterministic (twin-replay byte-exactness). None on records reloaded from JSONL.
    rollout_timing: Mapping[str, Any] | None = None


@dataclass(frozen=True)
class CollectionMetrics:
    games: int
    elapsed_seconds: float
    total_decision_rounds: int
    total_simulator_turns: int
    p1_wins: int
    p2_wins: int
    ties: int
    capped_games: int
    peak_rss_mb: float | None = None
    peak_rss_mb_by_phase: Mapping[str, float | None] | None = None
    policy_decision_summary: Mapping[str, Mapping[str, Any]] | None = None
    collection_timing: Mapping[str, Any] | None = None

    @property
    def games_per_second(self) -> float:
        return self.games / self.elapsed_seconds if self.elapsed_seconds > 0 else 0.0

    @property
    def decisions_per_second(self) -> float:
        return self.total_decision_rounds / self.elapsed_seconds if self.elapsed_seconds > 0 else 0.0

    @property
    def average_decision_rounds(self) -> float:
        return self.total_decision_rounds / self.games if self.games else 0.0

    @property
    def average_simulator_turns(self) -> float:
        return self.total_simulator_turns / self.games if self.games else 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "games": self.games,
            "elapsed_seconds": self.elapsed_seconds,
            "total_decision_rounds": self.total_decision_rounds,
            "total_simulator_turns": self.total_simulator_turns,
            "p1_wins": self.p1_wins,
            "p2_wins": self.p2_wins,
            "ties": self.ties,
            "capped_games": self.capped_games,
            "games_per_second": self.games_per_second,
            "decisions_per_second": self.decisions_per_second,
            "average_decision_rounds": self.average_decision_rounds,
            "average_simulator_turns": self.average_simulator_turns,
            **({"peak_rss_mb": self.peak_rss_mb} if self.peak_rss_mb is not None else {}),
            **(
                {"peak_rss_mb_by_phase": dict(self.peak_rss_mb_by_phase)}
                if self.peak_rss_mb_by_phase
                else {}
            ),
            **(
                {
                    "policy_decision_summary": {
                        key: dict(value)
                        for key, value in self.policy_decision_summary.items()
                    }
                }
                if self.policy_decision_summary
                else {}
            ),
            **({"collection_timing": dict(self.collection_timing)} if self.collection_timing else {}),
        }


def collect_rollouts(
    *,
    output_path: Path,
    games: int,
    env_factory: Callable[[], PokeZeroEnv],
    policies: Mapping[str, Policy],
    rollout_config: RolloutConfig,
    seed_start: int = 1,
    append: bool = False,
) -> CollectionMetrics:
    if games <= 0:
        raise ValueError("games must be positive.")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    accumulator = _MetricsAccumulator()
    collection_start = perf_counter()
    write_path = output_path if append else _temporary_output_path(output_path)
    # One env reused across all games (warm bridge process), instead of spawning a node per game.
    env = env_factory()
    try:
        with write_path.open("a" if append else "w", encoding="utf-8") as handle:
            for game_index in range(games):
                seed = seed_start + game_index
                record = run_rollout_record_on_env(
                    env=env,
                    policies=policies,
                    rollout_config=rollout_config,
                    seed=seed,
                    battle_id=f"rollout-{seed}",
                )
                accumulator.add(record)
                write_started = perf_counter()
                write_rollout_record(handle, record)
                accumulator.add_timing("raw_rollout_jsonl_write", perf_counter() - write_started)
        if not append:
            write_path.replace(output_path)
    except Exception:
        if not append:
            write_path.unlink(missing_ok=True)
        raise
    finally:
        close = getattr(env, "close", None)
        if callable(close):
            close()
    elapsed = perf_counter() - collection_start
    return accumulator.to_metrics(elapsed_seconds=elapsed, peak_rss_mb=current_peak_rss_mb())


def collect_training_cache(
    *,
    output_path: Path,
    games: int,
    env_factory: Callable[[], PokeZeroEnv],
    policies: Mapping[str, Policy],
    rollout_config: RolloutConfig,
    dataset_config: "TrajectoryDatasetConfig",
    seed_start: int = 1,
    overwrite: bool = False,
    max_cache_root_bytes: int | None = None,
    cache_root: Path | None = None,
) -> tuple[CollectionMetrics, "TrainingCacheSummary"]:
    """Collect rollouts and persist compact training examples instead of raw JSONL."""

    if games <= 0:
        raise ValueError("games must be positive.")
    from .dataset import TrainingCacheBuilder

    accumulator = _MetricsAccumulator()
    builder = TrainingCacheBuilder(config=dataset_config)
    collection_start = perf_counter()
    env = env_factory()
    try:
        for game_index in range(games):
            seed = seed_start + game_index
            record = run_rollout_record_on_env(
                env=env,
                policies=policies,
                rollout_config=rollout_config,
                seed=seed,
                battle_id=f"rollout-{seed}",
            )
            accumulator.add(record)
            write_started = perf_counter()
            builder.add_record(record)
            accumulator.add_timing("training_cache_add_record", perf_counter() - write_started)
        write_kwargs: dict[str, object] = {"overwrite": overwrite}
        if max_cache_root_bytes is not None:
            write_kwargs["max_cache_root_bytes"] = max_cache_root_bytes
        if cache_root is not None:
            write_kwargs["cache_root"] = cache_root
        write_started = perf_counter()
        summary = builder.write(output_path, **write_kwargs)
        accumulator.add_timing("training_cache_flush_write", perf_counter() - write_started)
    finally:
        close = getattr(env, "close", None)
        if callable(close):
            close()
    elapsed = perf_counter() - collection_start
    return accumulator.to_metrics(elapsed_seconds=elapsed, peak_rss_mb=current_peak_rss_mb()), summary


def benchmark_rollouts(
    *,
    games: int,
    env_factory: Callable[[], PokeZeroEnv],
    rollout_config: RolloutConfig,
    seed_start: int = 1,
    matchups: Iterable[BenchmarkMatchup] | None = None,
    progress_callback: Callable[[BenchmarkProgress], None] | None = None,
) -> BenchmarkReport:
    if games <= 0:
        raise ValueError("games must be positive.")
    selected_matchups = tuple(matchups) if matchups is not None else default_benchmark_matchups()
    if not selected_matchups:
        raise ValueError("at least one benchmark matchup is required.")

    results: list[BenchmarkMatchupResult] = []
    # One env reused across every matchup and game (warm bridge process).
    env = env_factory()
    try:
        for matchup_index, matchup in enumerate(selected_matchups):
            policies = {
                "p1": matchup.p1_policy,
                "p2": matchup.p2_policy,
            }
            accumulator = _MetricsAccumulator()
            belief_public_checksums_by_seed: dict[int, tuple[str, ...]] = {}
            game_results: list[BenchmarkGameResult] = []
            matchup_start = perf_counter()
            progress_callback_seconds = 0.0
            for game_index in range(games):
                seed = seed_start + game_index
                record = run_rollout_record_on_env(
                    env=env,
                    policies=policies,
                    rollout_config=rollout_config,
                    seed=seed,
                    battle_id=f"benchmark-{_slugify_label(matchup.label)}-{seed}",
                )
                accumulator.add(record)
                game_result = _benchmark_game_result(record)
                game_results.append(game_result)
                checksums = tuple(
                    sorted(
                        {
                            str(step.metadata["root_puct_belief_public_checksum"])
                            for step in record.trajectory.steps
                            if step.metadata.get("root_puct_belief_public_checksum")
                        }
                    )
                )
                if checksums:
                    belief_public_checksums_by_seed[seed] = checksums
                if progress_callback is not None:
                    matchup_elapsed_seconds = (
                        perf_counter() - matchup_start - progress_callback_seconds
                    )
                    callback_started = perf_counter()
                    progress_callback(
                        BenchmarkProgress(
                            matchup_label=matchup.label,
                            matchup_index=matchup_index,
                            matchup_count=len(selected_matchups),
                            games_completed=game_index + 1,
                            games_total=games,
                            seed=seed,
                            matchup_elapsed_seconds=matchup_elapsed_seconds,
                            root_puct_by_player=game_result.root_puct_by_player,
                        )
                    )
                    progress_callback_seconds += perf_counter() - callback_started
            elapsed = perf_counter() - matchup_start - progress_callback_seconds
            results.append(
                BenchmarkMatchupResult(
                    label=matchup.label,
                    p1_policy_id=matchup.p1_policy.policy_id,
                    p2_policy_id=matchup.p2_policy.policy_id,
                    seed_start=seed_start,
                    metrics=accumulator.to_metrics(elapsed_seconds=elapsed, peak_rss_mb=current_peak_rss_mb()),
                    p1_policy_provenance=benchmark_policy_provenance(matchup.p1_policy),
                    p2_policy_provenance=benchmark_policy_provenance(matchup.p2_policy),
                    root_puct_belief_public_checksums_by_seed=belief_public_checksums_by_seed or None,
                    game_results=tuple(game_results),
                )
            )
    finally:
        close = getattr(env, "close", None)
        if callable(close):
            close()

    return BenchmarkReport(
        format_id=rollout_config.format_id,
        max_decision_rounds=rollout_config.max_decision_rounds,
        games_per_matchup=games,
        matchups=tuple(results),
        policy_provenance=_benchmark_report_policy_provenance(results),
    )


def _benchmark_game_result(record: RolloutRecord) -> BenchmarkGameResult:
    root_puct_by_player: dict[str, dict[str, Any]] = {}
    root_puct_decision_telemetry_by_player: dict[str, tuple[Mapping[str, Any], ...]] = {}
    policy_elapsed_seconds_by_player: dict[str, tuple[float, ...]] = {}
    for player_id in ("p1", "p2"):
        accumulator = _PolicyDecisionAccumulator()
        elapsed_samples: list[float] = []
        policy_elapsed_samples: list[float] = []
        root_puct_decisions: list[Mapping[str, Any]] = []
        decision_index = 0
        for step in record.trajectory.steps:
            if step.player_id != player_id:
                continue
            telemetry = root_puct_decision_telemetry(
                step.metadata,
                decision_index=decision_index,
                turn_index=step.turn_index,
            )
            decision_index += 1
            if telemetry is not None:
                root_puct_decisions.append(telemetry)
            accumulator.add(step.metadata)
            elapsed = _metadata_optional_float(step.metadata.get("root_puct_elapsed_seconds"))
            if elapsed is not None:
                elapsed_samples.append(elapsed)
            policy_elapsed = _metadata_optional_float(step.metadata.get("policy_elapsed_seconds"))
            if policy_elapsed is not None:
                policy_elapsed_samples.append(policy_elapsed)
        diagnostics = accumulator.to_dict()
        if policy_elapsed_samples:
            policy_elapsed_seconds_by_player[player_id] = tuple(policy_elapsed_samples)
        if "root_puct_searches" not in diagnostics:
            continue
        # Reasons can embed raw replay-observation mismatch values. Seed-level artifacts are
        # intended for paired strength analysis, so retain only the stable category histogram.
        diagnostics.pop("root_puct_fallback_reasons", None)
        if elapsed_samples:
            diagnostics["root_puct_elapsed_seconds"] = elapsed_samples
        root_puct_by_player[player_id] = diagnostics
        if root_puct_decisions:
            root_puct_decision_telemetry_by_player[player_id] = tuple(root_puct_decisions)
    return BenchmarkGameResult(
        seed=record.seed,
        battle_id=record.battle_id,
        winner=record.terminal.winner,
        capped=record.terminal.capped,
        decision_rounds=record.decision_round_count,
        elapsed_seconds=record.elapsed_seconds,
        root_puct_by_player=root_puct_by_player,
        root_puct_decision_telemetry_by_player=root_puct_decision_telemetry_by_player,
        opponent_legal_mask_mode=(
            "hidden" if bool(record.trajectory.metadata.get("opponent_legal_action_masks_hidden")) else "privileged"
        ),
        policy_elapsed_seconds_by_player=policy_elapsed_seconds_by_player,
    )


def benchmark_policy_provenance(policy: Policy) -> dict[str, Any]:
    """Audit payload for the concrete policy object used in a benchmark seat.

    Policy IDs are labels and may be aliases. The checkpoint path/hash fields are the guardrail
    that catches alias plumbing bugs where a benchmark row looks right but loaded the wrong
    weights.
    """

    policy_id = str(getattr(policy, "policy_id", "unknown"))
    base_policy = _unwrap_policy_alias(policy)
    base_policy_id = str(getattr(base_policy, "policy_id", policy_id))
    checkpoint_path = _optional_str(getattr(base_policy, "checkpoint_path", None))
    weights_sha256 = _optional_str(getattr(base_policy, "weights_sha256", None))
    return {
        "policy_id": policy_id,
        "base_policy_id": base_policy_id,
        "policy_class": type(base_policy).__name__,
        "checkpoint_path": checkpoint_path,
        "weights_sha256": weights_sha256,
        "aliased": policy is not base_policy or policy_id != base_policy_id,
    }


def _unwrap_policy_alias(policy: Policy) -> Policy:
    current = policy
    seen: set[int] = set()
    while id(current) not in seen:
        seen.add(id(current))
        wrapped = getattr(current, "policy", None)
        if wrapped is None:
            break
        current = wrapped
    return current


def _benchmark_report_policy_provenance(
    results: Iterable[BenchmarkMatchupResult],
) -> dict[str, Mapping[str, Any]]:
    provenance: dict[str, Mapping[str, Any]] = {}
    for result in results:
        if result.p1_policy_provenance is not None:
            provenance[result.p1_policy_id] = result.p1_policy_provenance
        if result.p2_policy_provenance is not None:
            provenance[result.p2_policy_id] = result.p2_policy_provenance
    return provenance


def _optional_str(value: object) -> str | None:
    if value is None:
        return None
    text = str(value)
    return text if text else None


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def default_benchmark_matchups() -> tuple[BenchmarkMatchup, ...]:
    return (
        BenchmarkMatchup("random-legal vs random-legal", RandomLegalPolicy(), RandomLegalPolicy()),
        BenchmarkMatchup("simple-legal vs random-legal", SimpleLegalPolicy(), RandomLegalPolicy()),
        BenchmarkMatchup("random-legal vs simple-legal", RandomLegalPolicy(), SimpleLegalPolicy()),
        BenchmarkMatchup("simple-legal vs simple-legal", SimpleLegalPolicy(), SimpleLegalPolicy()),
    )


def policy_benchmark_matchups(
    *,
    policy_specs: Iterable[str],
    opponent_policy_specs: Iterable[str] = ("random-legal", "simple-legal"),
    showdown_root: Path | str | None = None,
    include_policy_head_to_head: bool = False,
) -> tuple[BenchmarkMatchup, ...]:
    candidates = _policy_factories(policy_specs, showdown_root=showdown_root, label="candidate policy")
    opponents = _policy_factories(opponent_policy_specs, showdown_root=showdown_root, label="opponent policy")
    if include_policy_head_to_head and len(candidates) < 2:
        raise ValueError("--include-policy-head-to-head requires at least two distinct candidate policies.")
    overlapping_policy_ids = sorted(
        {candidate_id for candidate_id, _ in candidates}
        & {opponent_id for opponent_id, _ in opponents}
    )
    if overlapping_policy_ids:
        raise ValueError(
            "candidate and opponent policy ids must be distinct for shared-opponent benchmarks: "
            f"{', '.join(overlapping_policy_ids)}. Remove the duplicated opponent or retrain with a distinct --policy-id."
        )
    matchups: list[BenchmarkMatchup] = []

    for candidate_id, candidate_factory in candidates:
        for opponent_id, opponent_factory in opponents:
            matchups.extend(
                (
                    BenchmarkMatchup(
                        f"{candidate_id} vs {opponent_id}",
                        candidate_factory(),
                        opponent_factory(),
                    ),
                    BenchmarkMatchup(
                        f"{opponent_id} vs {candidate_id}",
                        opponent_factory(),
                        candidate_factory(),
                    ),
                )
            )

    if include_policy_head_to_head:
        for index, (first_id, first_factory) in enumerate(candidates):
            for second_id, second_factory in candidates[index + 1 :]:
                matchups.extend(
                    (
                        BenchmarkMatchup(
                            f"{first_id} vs {second_id}",
                            first_factory(),
                            second_factory(),
                        ),
                        BenchmarkMatchup(
                            f"{second_id} vs {first_id}",
                            second_factory(),
                            first_factory(),
                        ),
                    )
                )

    if not matchups:
        raise ValueError("custom policy benchmark produced no matchups; choose distinct policy ids.")
    return tuple(matchups)


def aggregate_benchmark_head_to_heads(
    matchup_results: Iterable[BenchmarkMatchupResult],
) -> tuple[BenchmarkHeadToHeadResult, ...]:
    accumulators: dict[tuple[str, str], _HeadToHeadAccumulator] = {}
    ordered_keys: list[tuple[str, str]] = []

    for result in matchup_results:
        p1_policy_id = result.p1_policy_id
        p2_policy_id = result.p2_policy_id
        if p1_policy_id == p2_policy_id:
            continue
        unordered_key = tuple(sorted((p1_policy_id, p2_policy_id)))
        accumulator = accumulators.get(unordered_key)
        if accumulator is None:
            accumulator = _HeadToHeadAccumulator(
                first_policy_id=p1_policy_id,
                second_policy_id=p2_policy_id,
            )
            accumulators[unordered_key] = accumulator
            ordered_keys.append(unordered_key)
        accumulator.add(result)

    return tuple(accumulators[key].to_result() for key in ordered_keys)


def run_rollout_record_on_env(
    *,
    env: PokeZeroEnv,
    policies: Mapping[str, Policy],
    rollout_config: RolloutConfig,
    seed: int,
    battle_id: str,
) -> RolloutRecord:
    """Play one game on an already-created env (RolloutDriver.run resets it per game).

    Reusing one env across games keeps the bridge process warm — a fresh battle on a live process
    costs ~3 ms vs ~240 ms to spawn+load a new one. Callers own the env's lifetime (close it).
    """
    start = perf_counter()
    result = RolloutDriver(env=env, policies=policies, config=rollout_config).run(seed=seed, battle_id=battle_id)
    elapsed = perf_counter() - start
    return record_from_result(
        result,
        policies=policies,
        elapsed_seconds=elapsed,
        belief_set_source_hash=getattr(env, "belief_set_source_hash", None),
    )


def run_rollout_record(
    *,
    env_factory: Callable[[], PokeZeroEnv],
    policies: Mapping[str, Policy],
    rollout_config: RolloutConfig,
    seed: int,
    battle_id: str,
) -> RolloutRecord:
    """One-shot: create an env, play a single game, close it. Prefer reusing an env across games
    (run_rollout_record_on_env / ReusableEnvPool) so the bridge process stays warm."""
    env = env_factory()
    try:
        return run_rollout_record_on_env(
            env=env, policies=policies, rollout_config=rollout_config, seed=seed, battle_id=battle_id
        )
    finally:
        close = getattr(env, "close", None)
        if callable(close):
            close()


class ReusableEnvPool:
    """Hands each worker thread its own env, reused across that thread's games (warm bridge process).

    LocalShowdownEnv reuses its live node process across reset(), so one env per thread amortizes the
    ~240 ms spawn+data-load over all of that thread's games. Call close_all() when collection ends.
    """

    def __init__(self, env_factory: Callable[[], PokeZeroEnv]) -> None:
        self._env_factory = env_factory
        self._local = threading.local()
        self._envs: list[PokeZeroEnv] = []
        self._lock = threading.Lock()

    def get(self) -> PokeZeroEnv:
        env = getattr(self._local, "env", None)
        if env is None:
            env = self._env_factory()
            self._local.env = env
            with self._lock:
                self._envs.append(env)
        return env

    def close_all(self) -> None:
        with self._lock:
            envs = list(self._envs)
            self._envs.clear()
        for env in envs:
            close = getattr(env, "close", None)
            if callable(close):
                try:
                    close()
                except Exception:
                    pass


def current_peak_rss_mb() -> float | None:
    # ru_maxrss is a process-lifetime high-water mark. It is useful for
    # coarse run health, not per-game or per-matchup memory attribution.
    try:
        import resource
    except ImportError:
        return None
    usage = resource.getrusage(resource.RUSAGE_SELF)
    rss = float(getattr(usage, "ru_maxrss", 0.0))
    if rss <= 0.0:
        return None
    if sys.platform == "darwin":
        return rss / (1024.0 * 1024.0)
    return rss / 1024.0


def record_from_result(
    result: RolloutResult,
    *,
    policies: Mapping[str, Policy],
    elapsed_seconds: float,
    belief_set_source_hash: str | None = None,
) -> RolloutRecord:
    return RolloutRecord(
        battle_id=result.trajectory.battle_id,
        seed=result.trajectory.seed,
        format_id=result.trajectory.format_id,
        policy_ids={player: policy.policy_id for player, policy in policies.items()},
        decision_round_count=result.decision_round_count,
        elapsed_seconds=elapsed_seconds,
        terminal=result.terminal,
        trajectory=result.trajectory,
        belief_set_source_hash=belief_set_source_hash,
        rollout_timing=result.timing.to_dict() if result.timing is not None else None,
    )


_BELIEF_HASH_KEY = "belief_set_source_hash"
# Sentinel distinct-hash entry for cache directories whose builder recorded mixed provenance;
# guarantees the caller's single-hash gate fails and the mixed warning names the cause.
BELIEF_PROVENANCE_MIXED = "<mixed-provenance-cache>"


def cache_feature_masks_by_path(paths: Iterable[Path | str]) -> tuple[tuple[Path, dict | None], ...]:
    """Per training-cache directory: the encode-time feature masks its metadata records.

    The mask-axis twin of ``distinct_belief_set_source_hashes``: collection stamps the
    resolved ``ObservationFeatureMasks`` into ``metadata.json`` so the trainer can
    hard-fail on a cache-vs-model mask mismatch. None marks non-cache inputs (JSONL) and
    pre-mask legacy caches, which cannot be checked.
    """
    results: list[tuple[Path, dict | None]] = []
    for path in paths:
        resolved = Path(path)
        metadata_path = resolved / "metadata.json"
        if not metadata_path.is_file():
            continue
        try:
            payload = json.loads(metadata_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            results.append((resolved, None))
            continue
        masks = payload.get("feature_masks")
        results.append((resolved, dict(masks) if isinstance(masks, dict) else None))
    return tuple(results)


def cache_observation_schemas_by_path(
    paths: Iterable[Path | str],
) -> tuple[tuple[Path, str | None], ...]:
    """(cache path, recorded observation schema or None) per training-cache input.

    The schema-axis twin of ``cache_feature_masks_by_path``: collection stamps the
    encoding env's observation schema into cache metadata so the trainer can refuse a
    cross-schema train. ``None`` = legacy cache (predates recording — by definition NOT
    v2.2, since recording shipped with the v2.2 fresh-selection latch) or non-cache
    input (JSONL).
    """
    results: list[tuple[Path, str | None]] = []
    for raw_path in paths:
        path = Path(raw_path)
        metadata_path = path / "metadata.json"
        schema: str | None = None
        if metadata_path.is_file():
            try:
                payload = json.loads(metadata_path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                payload = None
            if isinstance(payload, Mapping):
                recorded = payload.get("observation_schema")
                schema = str(recorded) if recorded else None
        results.append((path, schema))
    return tuple(results)


def cache_shaping_configs_by_path(
    paths: Iterable[Path | str],
) -> tuple[tuple[Path, dict | None, bool], ...]:
    """Per training-cache directory: the dense-shaping config its dataset_config records.

    The shaping-axis twin of ``cache_feature_masks_by_path``: collection bakes shaped
    returns into the cache and stamps ``dataset_config.potential_shaping`` into
    ``metadata.json``, so the trainer can hard-fail (both directions) on a cache-vs-flag
    shaping mismatch. Each row is ``(path, shaping_config_dict | None, checkable)``:
    shaping None with ``checkable=True`` means the cache is definitively UNSHAPED (all
    pre-shaping caches — the field did not exist, so their returns were never shaped);
    ``checkable=False`` marks unreadable metadata, which cannot be checked and is
    skipped. Non-cache inputs (JSONL) are omitted, matching the mask helper.
    """
    results: list[tuple[Path, dict | None, bool]] = []
    for path in paths:
        resolved = Path(path)
        metadata_path = resolved / "metadata.json"
        if not metadata_path.is_file():
            continue
        try:
            payload = json.loads(metadata_path.read_text(encoding="utf-8"))
            dataset_config = payload.get("dataset_config") if isinstance(payload, Mapping) else None
        except (OSError, json.JSONDecodeError):
            results.append((resolved, None, False))
            continue
        if not isinstance(dataset_config, Mapping):
            results.append((resolved, None, False))
            continue
        shaping = dataset_config.get("potential_shaping")
        results.append((resolved, dict(shaping) if isinstance(shaping, Mapping) else None, True))
    return tuple(results)


def distinct_belief_set_source_hashes(paths: Iterable[Path | str]) -> tuple[str | None, ...]:
    """Distinct belief provenance across training inputs (rollout jsonl or cache directories).

    Jsonl files are scanned in full (append flows can mix provenance within one file); the scan
    is a cheap substring test per line, parsing only lines that carry the key, and stops as soon
    as the outcome is decided (mixed). Cache directories read the hash their builder recorded in
    ``metadata.json``. Returns a sorted tuple; None marks source-off, pre-provenance, or
    unreadable inputs. Best-effort by design: provenance must never fail training.
    """
    seen: set[str | None] = set()
    for path in paths:
        resolved = Path(path)
        try:
            metadata_path = resolved / "metadata.json"
            if resolved.is_dir():
                payload = json.loads(metadata_path.read_text(encoding="utf-8"))
                if isinstance(payload, Mapping) and payload.get("belief_set_source_mixed"):
                    seen.update({None, BELIEF_PROVENANCE_MIXED})
                elif isinstance(payload, Mapping):
                    seen.add(payload.get(_BELIEF_HASH_KEY) or None)
                else:
                    seen.add(None)
                continue
            file_hashes: set[str | None] = set()
            with resolved.open("r", encoding="utf-8") as handle:
                for line in handle:
                    if not line.strip():
                        continue
                    if _BELIEF_HASH_KEY not in line:
                        file_hashes.add(None)
                    else:
                        payload = json.loads(line)
                        value = payload.get(_BELIEF_HASH_KEY) if isinstance(payload, Mapping) else None
                        file_hashes.add(str(value) if value else None)
                    if len(file_hashes) > 1:
                        break
            seen.update(file_hashes or {None})
        except (OSError, ValueError, AttributeError, TypeError):
            seen.add(None)
    return tuple(sorted(seen, key=lambda value: (value is None, value or "")))


def write_rollout_record(handle: TextIO, record: RolloutRecord) -> None:
    handle.write(json.dumps(rollout_record_to_dict(record), separators=(",", ":"), sort_keys=True))
    handle.write("\n")
    handle.flush()


def read_rollout_records(path: Path) -> tuple[RolloutRecord, ...]:
    return tuple(iter_rollout_records(path))


def iter_rollout_records(path: Path) -> Iterator[RolloutRecord]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            yield rollout_record_from_dict(json.loads(line))


def rollout_record_to_dict(record: RolloutRecord) -> dict[str, Any]:
    return {
        "schema_version": ROLLOUT_RECORD_SCHEMA_VERSION,
        "battle_id": record.battle_id,
        "seed": record.seed,
        "format_id": record.format_id,
        "policy_ids": dict(record.policy_ids),
        "decision_round_count": record.decision_round_count,
        "elapsed_seconds": record.elapsed_seconds,
        "terminal": _terminal_to_dict(record.terminal),
        "trajectory": trajectory_to_dict(record.trajectory),
        **(
            {"belief_set_source_hash": record.belief_set_source_hash}
            if record.belief_set_source_hash is not None
            else {}
        ),
    }


def rollout_record_from_dict(payload: Mapping[str, Any]) -> RolloutRecord:
    if payload.get("schema_version") != ROLLOUT_RECORD_SCHEMA_VERSION:
        raise ValueError(f"Unsupported rollout record schema: {payload.get('schema_version')!r}.")
    return RolloutRecord(
        battle_id=str(payload["battle_id"]),
        seed=int(payload["seed"]),
        format_id=str(payload["format_id"]),
        policy_ids={str(player): str(policy) for player, policy in _mapping(payload["policy_ids"]).items()},
        decision_round_count=int(payload["decision_round_count"]),
        elapsed_seconds=float(payload["elapsed_seconds"]),
        terminal=_terminal_from_dict(_mapping(payload["terminal"])),
        trajectory=trajectory_from_dict(_mapping(payload["trajectory"])),
        belief_set_source_hash=(
            str(payload["belief_set_source_hash"]) if payload.get("belief_set_source_hash") else None
        ),
    )


def summarize_records(records: Iterable[RolloutRecord], *, elapsed_seconds: float) -> CollectionMetrics:
    accumulator = _MetricsAccumulator()
    for record in records:
        accumulator.add(record)
    return accumulator.to_metrics(elapsed_seconds=elapsed_seconds, peak_rss_mb=current_peak_rss_mb())


def policy_from_spec(spec: str) -> Policy:
    return policy_factory_from_spec(spec)()


def linear_policy_factory_from_model_spec(spec: str, model: "LinearPolicyModel") -> Callable[[], Policy]:
    """Create a linear policy factory from an already-loaded model and a policy spec's options."""

    policy_body, options = _split_policy_spec_options(spec.strip())
    if not policy_body.lower().startswith(LINEAR_POLICY_SPEC_PREFIX):
        raise ValueError("linear model factory override requires a linear: policy spec.")
    checkpoint = policy_body[len(LINEAR_POLICY_SPEC_PREFIX) :].strip()
    if not checkpoint:
        raise ValueError("linear policy spec must include a checkpoint path after 'linear:'.")
    from .linear_policy import LinearSoftmaxPolicy

    linear_options = _linear_policy_options(options)
    return lambda: LinearSoftmaxPolicy(model=model, **linear_options)


def policy_factory_from_spec(spec: str) -> Callable[[], Policy]:
    normalized = spec.strip()
    policy_body, options = _split_policy_spec_options(normalized)
    lowered = policy_body.lower()
    if lowered == "random-legal":
        if options:
            raise ValueError("random-legal does not support policy spec options.")
        return RandomLegalPolicy
    if lowered == "simple-legal":
        if options:
            raise ValueError("simple-legal does not support policy spec options.")
        return SimpleLegalPolicy
    if lowered == "scripted-teacher":
        teacher_options = _scripted_teacher_options(options)
        return lambda: ScriptedTeacherPolicy(**teacher_options)
    if lowered in {"max-damage", "aggressive-damage"}:
        max_damage_options = _max_damage_options(options)
        if lowered == "aggressive-damage":
            max_damage_options["policy_id"] = "aggressive-damage"
        return lambda: MaxDamagePolicy(**max_damage_options)
    if lowered.startswith(LINEAR_POLICY_SPEC_PREFIX):
        from .linear_policy import LinearSoftmaxPolicy, load_linear_model

        checkpoint = policy_body[len(LINEAR_POLICY_SPEC_PREFIX) :].strip()
        if not checkpoint:
            raise ValueError("linear policy spec must include a checkpoint path after 'linear:'.")
        linear_options = _linear_policy_options(options)
        checkpoint_path = Path(checkpoint)
        model = load_linear_model(checkpoint_path)
        checkpoint_provenance = {
            "checkpoint_path": str(checkpoint_path.resolve(strict=False)),
            "weights_sha256": _file_sha256(checkpoint_path),
        }

        def factory() -> Policy:
            policy = LinearSoftmaxPolicy(model=model, **linear_options)
            policy.checkpoint_path = checkpoint_provenance["checkpoint_path"]
            policy.weights_sha256 = checkpoint_provenance["weights_sha256"]
            return policy

        return factory
    if lowered.startswith(NEURAL_POLICY_SPEC_PREFIX):
        from .neural_policy import load_transformer_policy

        checkpoint = policy_body[len(NEURAL_POLICY_SPEC_PREFIX) :].strip()
        if not checkpoint:
            raise ValueError("neural policy spec must include a checkpoint path after 'neural:'.")
        neural_options = _neural_policy_options(options)
        return lambda: load_transformer_policy(Path(checkpoint), **neural_options)
    if lowered.startswith(REMOTE_POLICY_SPEC_PREFIX):
        from .inference_service import remote_inference_policy

        base_url = policy_body[len(REMOTE_POLICY_SPEC_PREFIX) :].strip()
        if not base_url:
            raise ValueError("remote policy spec must include a server URL after 'remote:'.")
        if "://" not in base_url:
            base_url = "http://" + base_url
        remote_options = _neural_policy_options(options)
        return lambda: remote_inference_policy(base_url, **remote_options)
    raise ValueError(
        f"Unsupported policy spec: {spec!r}. Expected random-legal, simple-legal, max-damage, "
        "aggressive-damage, "
        "scripted-teacher, linear:/path/to/checkpoint.json, neural:/path/to/checkpoint.pt, "
        "or remote:host:port."
    )


def policy_from_name(name: str) -> Policy:
    return policy_from_spec(name)


def env_config_with_policy_spec_masks(env_config, specs: Iterable[str | None], *, context: str):
    """Adopt encode-time feature masks + observation spec from ``neural:`` checkpoint specs.

    The HIGH-1 train/eval latch for spec-driven CLI harnesses: every checkpoint that will
    observe through the env contributes its stamped masks AND its stamped observation
    schema/width (the dual-schema resolution: a v2 checkpoint keeps the v2 encode, a v2.1
    checkpoint the v2.1 encode); conflicts (between checkpoints, or with an explicit env
    override) hard-fail in ``env_config_with_checkpoint_masks``. Torch-free when no neural
    specs are present.
    """
    paths = neural_checkpoint_paths_from_policy_specs(specs)
    if not paths:
        return env_config
    from .local_showdown import env_config_with_checkpoint_masks
    from .neural_policy import (
        feature_masks_from_model_config,
        load_transformer_model_config,
        observation_spec_from_model_config,
    )

    configs = [load_transformer_model_config(path) for path in paths]
    return env_config_with_checkpoint_masks(
        env_config,
        [feature_masks_from_model_config(config) for config in configs],
        context=context,
        required_specs=[observation_spec_from_model_config(config) for config in configs],
    )


def neural_checkpoint_paths_from_policy_specs(specs: Iterable[str | None]) -> tuple[Path, ...]:
    """Checkpoint paths of every ``neural:`` policy spec (string parsing only, torch-free).

    Used by CLI harnesses to derive env encode-time feature masks from the checkpoints that
    will observe through the env (see ``local_showdown.env_config_with_checkpoint_masks``).
    """
    paths: list[Path] = []
    for spec in specs:
        if spec is None:
            continue
        body, _ = _split_policy_spec_options(str(spec).strip())
        if body.lower().startswith(NEURAL_POLICY_SPEC_PREFIX):
            checkpoint = body[len(NEURAL_POLICY_SPEC_PREFIX) :].strip()
            if checkpoint:
                paths.append(Path(checkpoint))
    return tuple(paths)


# Baselines that exist only to evaluate candidates and must never seed training data.
EVAL_ONLY_POLICY_NAMES = frozenset({"max-damage"})


def reject_eval_only_specs(specs: Iterable[str], *, role: str) -> None:
    """Raise if any spec names an evaluation-only baseline (e.g. max-damage) used for training."""
    for spec in specs:
        if spec is None:
            continue
        body, _ = _split_policy_spec_options(str(spec).strip())
        if body.lower() in EVAL_ONLY_POLICY_NAMES:
            raise ValueError(
                f"'{body}' is an evaluation-only baseline and cannot be used as a {role}; "
                "use it as a benchmark opponent (e.g. rollout_cli benchmark --opponent-policy max-damage) instead."
            )


def policy_spec_with_showdown_root(spec: str, showdown_root: Path | str | None) -> str:
    if showdown_root is None:
        return spec
    policy_body, options = _split_policy_spec_options(spec.strip())
    if policy_body.lower() not in ("scripted-teacher", "max-damage", "aggressive-damage") or "showdown_root" in options:
        return spec
    options = {**options, "showdown_root": str(showdown_root)}
    return f"{policy_body}?{urlencode(options)}"


def _policy_factories(
    specs: Iterable[str],
    *,
    showdown_root: Path | str | None,
    label: str,
) -> tuple[tuple[str, Callable[[], Policy]], ...]:
    deduped_specs = tuple(dict.fromkeys(str(spec) for spec in specs))
    if not deduped_specs:
        raise ValueError(f"at least one {label} spec is required.")
    factories: list[tuple[str, Callable[[], Policy]]] = []
    seen_policy_ids: set[str] = set()
    for spec in deduped_specs:
        rooted_spec = policy_spec_with_showdown_root(spec, showdown_root)
        factory = policy_factory_from_spec(rooted_spec)
        policy_id = str(factory().policy_id)
        if policy_id in seen_policy_ids:
            raise ValueError(
                f"duplicate {label} id: {policy_id}. Retrain with a distinct --policy-id "
                "so benchmark labels and head-to-head aggregation can distinguish checkpoints."
            )
        seen_policy_ids.add(policy_id)
        factories.append((policy_id, factory))
    return tuple(factories)


def _split_policy_spec_options(spec: str) -> tuple[str, dict[str, str]]:
    body, separator, query = spec.partition("?")
    if not separator:
        return body, {}
    options: dict[str, str] = {}
    for key, value in parse_qsl(query, keep_blank_values=True):
        normalized_key = key.strip().lower()
        if not normalized_key:
            raise ValueError("policy spec option names must be non-empty.")
        if normalized_key in options:
            raise ValueError(f"duplicate policy spec option: {normalized_key}.")
        options[normalized_key] = value.strip()
    return body, options


def _linear_policy_options(options: Mapping[str, str]) -> dict[str, object]:
    supported = {"sample", "deterministic", "epsilon", "temperature"}
    unknown = sorted(set(options) - supported)
    if unknown:
        raise ValueError(f"Unsupported linear policy option(s): {', '.join(unknown)}.")

    sample = _optional_bool(options, "sample")
    deterministic = _optional_bool(options, "deterministic")
    if sample is not None and deterministic is not None and sample == deterministic:
        raise ValueError("linear policy options 'sample' and 'deterministic' conflict.")
    deterministic_policy = deterministic if deterministic is not None else False
    if sample is not None:
        deterministic_policy = not sample

    exploration_epsilon = _optional_float(options, "epsilon", default=0.0)
    sampling_temperature = _optional_float(options, "temperature", default=1.0)
    if not 0.0 <= exploration_epsilon <= 1.0:
        raise ValueError("linear policy epsilon must be between 0 and 1.")
    if sampling_temperature <= 0.0:
        raise ValueError("linear policy temperature must be positive.")
    return {
        "deterministic": deterministic_policy,
        "exploration_epsilon": exploration_epsilon,
        "sampling_temperature": sampling_temperature,
    }


def _neural_policy_options(options: Mapping[str, str]) -> dict[str, object]:
    supported = {"sample", "deterministic", "epsilon", "temperature", "device", "family_gated"}
    unknown = sorted(set(options) - supported)
    if unknown:
        raise ValueError(f"Unsupported neural policy option(s): {', '.join(unknown)}.")
    policy_options = _linear_policy_options({key: value for key, value in options.items() if key not in {"device", "family_gated"}})
    if options.get("device"):
        policy_options["device"] = options["device"]
    policy_options["family_gated_selection"] = _optional_bool(options, "family_gated") or False
    return policy_options


def _scripted_teacher_options(options: Mapping[str, str]) -> dict[str, object]:
    supported = {
        "showdown_root",
        "switch_margin",
        "poor_move_threshold",
        "team_status_cure_score",
        "status_pressure_score",
        "statused_switch_penalty",
        "low_hp_switch_bonus",
        "active_danger_switch_bonus",
        "tie_breaker",
        "allow_fallback",
        "allow_unknown_moves",
    }
    unknown = sorted(set(options) - supported)
    if unknown:
        raise ValueError(f"Unsupported scripted-teacher option(s): {', '.join(unknown)}.")
    teacher_options: dict[str, object] = {}
    if options.get("showdown_root"):
        teacher_options["showdown_root"] = Path(options["showdown_root"])
    if "switch_margin" in options:
        teacher_options["switch_margin"] = _optional_float(options, "switch_margin", default=8.0)
    if "poor_move_threshold" in options:
        teacher_options["poor_move_threshold"] = _optional_float(options, "poor_move_threshold", default=35.0)
    if "team_status_cure_score" in options:
        teacher_options["team_status_cure_score"] = _optional_float(options, "team_status_cure_score", default=64.0)
    if "status_pressure_score" in options:
        teacher_options["status_pressure_score"] = _optional_float(options, "status_pressure_score", default=55.0)
    if "statused_switch_penalty" in options:
        teacher_options["statused_switch_penalty"] = _optional_float(options, "statused_switch_penalty", default=10.0)
    if "low_hp_switch_bonus" in options:
        teacher_options["low_hp_switch_bonus"] = _optional_float(options, "low_hp_switch_bonus", default=35.0)
    if "active_danger_switch_bonus" in options:
        teacher_options["active_danger_switch_bonus"] = _optional_float(
            options, "active_danger_switch_bonus", default=45.0
        )
    if "tie_breaker" in options:
        teacher_options["tie_breaker"] = options["tie_breaker"]
    allow_fallback = _optional_bool(options, "allow_fallback")
    if allow_fallback is not None:
        teacher_options["allow_fallback"] = allow_fallback
    allow_unknown_moves = _optional_bool(options, "allow_unknown_moves")
    if allow_unknown_moves is not None:
        teacher_options["allow_unknown_moves"] = allow_unknown_moves
    return teacher_options


def _max_damage_options(options: Mapping[str, str]) -> dict[str, object]:
    unknown = sorted(set(options) - {"showdown_root"})
    if unknown:
        raise ValueError(f"Unsupported max-damage option(s): {', '.join(unknown)}.")
    max_damage_options: dict[str, object] = {}
    if options.get("showdown_root"):
        max_damage_options["showdown_root"] = Path(options["showdown_root"])
    return max_damage_options


def _optional_bool(options: Mapping[str, str], key: str) -> bool | None:
    if key not in options:
        return None
    value = options[key].strip().lower()
    if value == "":
        return True
    if value in {"1", "true", "yes", "on"}:
        return True
    if value in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"policy spec option {key!r} must be a boolean value.")


def _optional_float(options: Mapping[str, str], key: str, *, default: float) -> float:
    if key not in options:
        return default
    try:
        return float(options[key])
    except ValueError as exc:
        raise ValueError(f"policy spec option {key!r} must be numeric.") from exc


def _terminal_to_dict(terminal: TerminalState) -> dict[str, Any]:
    return {
        "winner": terminal.winner,
        "turn_count": terminal.turn_count,
        "capped": terminal.capped,
    }


def _terminal_from_dict(payload: Mapping[str, Any]) -> TerminalState:
    winner = payload.get("winner")
    return TerminalState(
        winner=str(winner) if winner is not None else None,
        turn_count=int(payload["turn_count"]),
        capped=bool(payload.get("capped", False)),
    )


def _mapping(value: Any) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError("expected JSON object payload.")
    return value


def _temporary_output_path(output_path: Path) -> Path:
    return output_path.with_name(f".{output_path.name}.tmp")


def _slugify_label(label: str) -> str:
    slug = "".join(character.lower() if character.isalnum() else "-" for character in label.strip())
    return "-".join(part for part in slug.split("-") if part)


@dataclass
class _CollectionTimingAccumulator:
    elapsed_by_phase: dict[str, float] = field(default_factory=dict)
    calls_by_phase: dict[str, int] = field(default_factory=dict)

    def add(self, phase: str, elapsed_seconds: float, *, calls: int = 1) -> None:
        if elapsed_seconds < 0.0:
            return
        self.elapsed_by_phase[phase] = self.elapsed_by_phase.get(phase, 0.0) + elapsed_seconds
        self.calls_by_phase[phase] = self.calls_by_phase.get(phase, 0) + max(0, calls)

    def add_rollout_timing(self, value: object) -> None:
        if not isinstance(value, Mapping):
            return
        for phase in ("env_reset", "env_observe", "policy_select", "env_step"):
            elapsed = _metadata_optional_float(value.get(f"{phase}_elapsed_seconds"))
            if elapsed is None:
                continue
            calls = _metadata_optional_int(value.get(f"{phase}_calls"))
            if calls is None:
                calls = 1 if phase == "env_reset" else 0
            self.add(phase, elapsed, calls=calls)

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {}
        total_profiled = 0.0
        for phase in sorted(self.elapsed_by_phase):
            elapsed = self.elapsed_by_phase[phase]
            calls = self.calls_by_phase.get(phase, 0)
            payload[f"{phase}_elapsed_seconds"] = elapsed
            payload[f"{phase}_calls"] = calls
            if calls:
                payload[f"{phase}_average_elapsed_seconds"] = elapsed / calls
            total_profiled += elapsed
        if payload:
            payload["total_profiled_elapsed_seconds"] = total_profiled
        return payload


@dataclass
class _MetricsAccumulator:
    games: int = 0
    total_decision_rounds: int = 0
    total_simulator_turns: int = 0
    p1_wins: int = 0
    p2_wins: int = 0
    ties: int = 0
    capped_games: int = 0
    policy_summaries: dict[str, "_PolicyDecisionAccumulator"] = field(default_factory=dict)
    timing: _CollectionTimingAccumulator = field(default_factory=_CollectionTimingAccumulator)

    def add(self, record: RolloutRecord) -> None:
        self.games += 1
        self.total_decision_rounds += record.decision_round_count
        self.total_simulator_turns += record.terminal.turn_count
        if record.terminal.winner == "p1":
            self.p1_wins += 1
        elif record.terminal.winner == "p2":
            self.p2_wins += 1
        elif not record.terminal.capped:
            self.ties += 1
        if record.terminal.capped:
            self.capped_games += 1
        self.timing.add_rollout_timing(record.rollout_timing)
        for step in record.trajectory.steps:
            metadata = step.metadata
            policy_id = str(metadata.get("policy_id") or "unknown")
            summary = self.policy_summaries.get(policy_id)
            if summary is None:
                summary = _PolicyDecisionAccumulator()
                self.policy_summaries[policy_id] = summary
            summary.add(metadata)

    def add_timing(self, phase: str, elapsed_seconds: float, *, calls: int = 1) -> None:
        self.timing.add(phase, elapsed_seconds, calls=calls)

    def to_metrics(self, *, elapsed_seconds: float, peak_rss_mb: float | None = None) -> CollectionMetrics:
        return CollectionMetrics(
            games=self.games,
            elapsed_seconds=elapsed_seconds,
            total_decision_rounds=self.total_decision_rounds,
            total_simulator_turns=self.total_simulator_turns,
            p1_wins=self.p1_wins,
            p2_wins=self.p2_wins,
            ties=self.ties,
            capped_games=self.capped_games,
            peak_rss_mb=peak_rss_mb,
            policy_decision_summary={
                policy_id: summary.to_dict()
                for policy_id, summary in sorted(self.policy_summaries.items())
            },
            collection_timing=self.timing.to_dict(),
        )


@dataclass
class _PolicyDecisionAccumulator:
    decisions: int = 0
    root_puct_searches: int = 0
    root_puct_fallbacks: int = 0
    root_puct_total_visits: int = 0
    root_puct_effective_total_visits: int = 0
    root_puct_elapsed_seconds_total: float = 0.0
    root_puct_elapsed_seconds_samples: int = 0
    root_puct_candidate_count_total: int = 0
    root_puct_candidate_count_samples: int = 0
    root_puct_selected_value_total: float = 0.0
    root_puct_selected_value_samples: int = 0
    root_puct_selected_score_total: float = 0.0
    root_puct_selected_score_samples: int = 0
    root_puct_value_gate_checks: int = 0
    root_puct_value_gate_uses: int = 0
    root_puct_time_budget_checks: int = 0
    root_puct_time_budget_exhaustions: int = 0
    root_puct_fallback_reasons: dict[str, int] = field(default_factory=dict)
    root_puct_fallback_categories: dict[str, int] = field(default_factory=dict)
    root_puct_opponent_action_missing_sampled_world_reason_categories: dict[str, int] = field(
        default_factory=dict
    )
    root_puct_selection_modes: dict[str, int] = field(default_factory=dict)
    root_puct_opponent_action_policies: dict[str, int] = field(default_factory=dict)
    root_puct_opponent_action_scenario_counts: dict[str, int] = field(default_factory=dict)
    root_puct_leaf_rollout_rounds: dict[str, int] = field(default_factory=dict)
    root_puct_leaf_rollout_opponent_policies: dict[str, int] = field(default_factory=dict)
    root_puct_leaf_actual_rollout_rounds: dict[str, int] = field(default_factory=dict)
    root_puct_leaf_evaluations: dict[str, int] = field(default_factory=dict)

    def add(self, metadata: Mapping[str, Any]) -> None:
        self.decisions += 1
        if metadata.get("policy_family") != "root-puct-search":
            return
        _merge_count_mapping(
            self.root_puct_opponent_action_missing_sampled_world_reason_categories,
            metadata.get("root_puct_opponent_action_missing_sampled_world_reason_categories"),
        )
        if bool(metadata.get("root_puct_fallback")):
            self.root_puct_fallbacks += 1
            reason = str(metadata.get("root_puct_fallback_reason") or "unknown")
            self.root_puct_fallback_reasons[reason] = (
                self.root_puct_fallback_reasons.get(reason, 0) + 1
            )
            category = str(
                metadata.get("root_puct_fallback_category")
                or root_puct_fallback_category(reason)
            )
            self.root_puct_fallback_categories[category] = (
                self.root_puct_fallback_categories.get(category, 0) + 1
            )
            return
        self.root_puct_searches += 1
        total_visits = _metadata_optional_int(metadata.get("root_puct_total_visits"))
        if total_visits is not None:
            self.root_puct_total_visits += total_visits
        effective_total_visits = _metadata_optional_int(metadata.get("root_puct_effective_total_visits"))
        if effective_total_visits is not None:
            self.root_puct_effective_total_visits += effective_total_visits
        elapsed_seconds = _metadata_optional_float(metadata.get("root_puct_elapsed_seconds"))
        if elapsed_seconds is not None:
            self.root_puct_elapsed_seconds_total += elapsed_seconds
            self.root_puct_elapsed_seconds_samples += 1
        candidate_count = _metadata_optional_int(metadata.get("root_puct_candidate_count"))
        if candidate_count is not None:
            self.root_puct_candidate_count_total += candidate_count
            self.root_puct_candidate_count_samples += 1
        selected_value = _metadata_optional_float(metadata.get("root_puct_selected_value"))
        if selected_value is not None:
            self.root_puct_selected_value_total += selected_value
            self.root_puct_selected_value_samples += 1
        selected_score = _metadata_optional_float(metadata.get("root_puct_selected_score"))
        if selected_score is not None:
            self.root_puct_selected_score_total += selected_score
            self.root_puct_selected_score_samples += 1
        if "root_puct_value_gate_used" in metadata:
            self.root_puct_value_gate_checks += 1
            if bool(metadata.get("root_puct_value_gate_used")):
                self.root_puct_value_gate_uses += 1
        if "root_puct_time_budget_exhausted" in metadata:
            self.root_puct_time_budget_checks += 1
            if bool(metadata.get("root_puct_time_budget_exhausted")):
                self.root_puct_time_budget_exhaustions += 1
        selection_mode = metadata.get("root_puct_selection_mode")
        if selection_mode is not None:
            key = str(selection_mode)
            self.root_puct_selection_modes[key] = self.root_puct_selection_modes.get(key, 0) + 1
        opponent_action_policy = metadata.get("root_puct_opponent_action_policy")
        if opponent_action_policy is not None:
            key = str(opponent_action_policy)
            self.root_puct_opponent_action_policies[key] = (
                self.root_puct_opponent_action_policies.get(key, 0) + 1
            )
        opponent_action_scenario_count = metadata.get("root_puct_opponent_action_scenario_count")
        if opponent_action_scenario_count is not None:
            key = str(opponent_action_scenario_count)
            self.root_puct_opponent_action_scenario_counts[key] = (
                self.root_puct_opponent_action_scenario_counts.get(key, 0) + 1
            )
        leaf_rollout_rounds = metadata.get("root_puct_leaf_rollout_rounds")
        if leaf_rollout_rounds is not None:
            key = str(leaf_rollout_rounds)
            self.root_puct_leaf_rollout_rounds[key] = self.root_puct_leaf_rollout_rounds.get(key, 0) + 1
        leaf_opponent_policy = metadata.get("root_puct_leaf_rollout_opponent_policy")
        if leaf_opponent_policy is not None:
            key = str(leaf_opponent_policy)
            self.root_puct_leaf_rollout_opponent_policies[key] = (
                self.root_puct_leaf_rollout_opponent_policies.get(key, 0) + 1
            )
        _merge_count_mapping(
            self.root_puct_leaf_actual_rollout_rounds,
            metadata.get("root_puct_leaf_actual_rollout_rounds"),
        )
        _merge_count_mapping(
            self.root_puct_leaf_evaluations,
            metadata.get("root_puct_leaf_evaluations"),
        )

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {"decisions": self.decisions}
        if self.root_puct_searches or self.root_puct_fallbacks:
            result.update(
                {
                    "root_puct_searches": self.root_puct_searches,
                    "root_puct_fallbacks": self.root_puct_fallbacks,
                    "root_puct_total_visits": self.root_puct_total_visits,
                }
            )
            if self.root_puct_effective_total_visits:
                result["root_puct_effective_total_visits"] = self.root_puct_effective_total_visits
            if self.root_puct_elapsed_seconds_samples:
                result["root_puct_average_elapsed_seconds"] = (
                    self.root_puct_elapsed_seconds_total / self.root_puct_elapsed_seconds_samples
                )
            if self.root_puct_candidate_count_samples:
                result["root_puct_average_candidate_count"] = (
                    self.root_puct_candidate_count_total / self.root_puct_candidate_count_samples
                )
            if self.root_puct_selected_value_samples:
                result["root_puct_average_selected_value"] = (
                    self.root_puct_selected_value_total / self.root_puct_selected_value_samples
                )
            if self.root_puct_selected_score_samples:
                result["root_puct_average_selected_score"] = (
                    self.root_puct_selected_score_total / self.root_puct_selected_score_samples
                )
            if self.root_puct_value_gate_checks:
                result["root_puct_value_gate_checks"] = self.root_puct_value_gate_checks
                result["root_puct_value_gate_uses"] = self.root_puct_value_gate_uses
            if self.root_puct_time_budget_checks:
                result["root_puct_time_budget_checks"] = self.root_puct_time_budget_checks
                result["root_puct_time_budget_exhaustions"] = self.root_puct_time_budget_exhaustions
            if self.root_puct_selection_modes:
                result["root_puct_selection_modes"] = dict(sorted(self.root_puct_selection_modes.items()))
            if self.root_puct_opponent_action_policies:
                result["root_puct_opponent_action_policies"] = dict(
                    sorted(self.root_puct_opponent_action_policies.items())
                )
            if self.root_puct_opponent_action_scenario_counts:
                result["root_puct_opponent_action_scenario_counts"] = dict(
                    sorted(self.root_puct_opponent_action_scenario_counts.items())
                )
            if self.root_puct_leaf_rollout_rounds:
                result["root_puct_leaf_rollout_rounds"] = dict(
                    sorted(self.root_puct_leaf_rollout_rounds.items())
                )
            if self.root_puct_leaf_rollout_opponent_policies:
                result["root_puct_leaf_rollout_opponent_policies"] = dict(
                    sorted(self.root_puct_leaf_rollout_opponent_policies.items())
                )
            if self.root_puct_leaf_actual_rollout_rounds:
                result["root_puct_leaf_actual_rollout_rounds"] = dict(
                    sorted(self.root_puct_leaf_actual_rollout_rounds.items())
                )
            if self.root_puct_leaf_evaluations:
                result["root_puct_leaf_evaluations"] = dict(
                    sorted(self.root_puct_leaf_evaluations.items())
                )
            if self.root_puct_fallback_reasons:
                result["root_puct_fallback_reasons"] = dict(
                    sorted(self.root_puct_fallback_reasons.items())
                )
            if self.root_puct_fallback_categories:
                result["root_puct_fallback_categories"] = dict(
                    sorted(self.root_puct_fallback_categories.items())
                )
            if self.root_puct_opponent_action_missing_sampled_world_reason_categories:
                result["root_puct_opponent_action_missing_sampled_world_reason_categories"] = dict(
                    sorted(self.root_puct_opponent_action_missing_sampled_world_reason_categories.items())
                )
        return result


def _metadata_optional_float(value: object) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _metadata_optional_int(value: object) -> int | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _merge_count_mapping(target: dict[str, int], value: object) -> None:
    if not isinstance(value, Mapping):
        return
    for key, count in value.items():
        parsed_count = _metadata_optional_int(count)
        if parsed_count is None:
            continue
        parsed_key = str(key)
        target[parsed_key] = target.get(parsed_key, 0) + parsed_count


@dataclass
class _HeadToHeadAccumulator:
    first_policy_id: str
    second_policy_id: str
    games: int = 0
    first_policy_wins: int = 0
    second_policy_wins: int = 0
    ties: int = 0
    capped_games: int = 0

    def add(self, result: BenchmarkMatchupResult) -> None:
        metrics = result.metrics
        self.games += metrics.games
        self.ties += metrics.ties
        self.capped_games += metrics.capped_games
        if result.p1_policy_id == self.first_policy_id and result.p2_policy_id == self.second_policy_id:
            self.first_policy_wins += metrics.p1_wins
            self.second_policy_wins += metrics.p2_wins
        elif result.p1_policy_id == self.second_policy_id and result.p2_policy_id == self.first_policy_id:
            self.first_policy_wins += metrics.p2_wins
            self.second_policy_wins += metrics.p1_wins
        else:
            raise ValueError("matchup result does not match head-to-head policies.")

    def to_result(self) -> BenchmarkHeadToHeadResult:
        return BenchmarkHeadToHeadResult(
            label=f"{self.first_policy_id} vs {self.second_policy_id}",
            first_policy_id=self.first_policy_id,
            second_policy_id=self.second_policy_id,
            games=self.games,
            first_policy_wins=self.first_policy_wins,
            second_policy_wins=self.second_policy_wins,
            ties=self.ties,
            capped_games=self.capped_games,
        )
