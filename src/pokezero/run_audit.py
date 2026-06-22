"""Run-level health audits for self-play experiment manifests."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from .evaluation import (
    DEFAULT_MAX_BENCHMARK_CAPPED_RATE,
    DEFAULT_MAX_COLLECTION_CAPPED_RATE,
    DEFAULT_MIN_BENCHMARK_GAMES,
    DEFAULT_MIN_BENCHMARK_WIN_RATE,
    NEURAL_SELFPLAY_RUN_SCHEMA_VERSION,
    _benchmark_summary,
    _capped_rate,
    _load_manifest,
    _manifest_path,
    _mapping,
    _sequence,
)
from .selfplay import SELFPLAY_RUN_SCHEMA_VERSION


DEFAULT_MAX_BENCHMARK_WIN_RATE_DROP = 0.05
DEFAULT_MAX_CONSECUTIVE_PROMOTION_FAILURES = 1


@dataclass(frozen=True)
class RunAuditConfig:
    min_latest_benchmark_win_rate: float = DEFAULT_MIN_BENCHMARK_WIN_RATE
    min_latest_benchmark_games: int = DEFAULT_MIN_BENCHMARK_GAMES
    max_latest_collection_capped_rate: float = DEFAULT_MAX_COLLECTION_CAPPED_RATE
    max_latest_benchmark_capped_rate: float = DEFAULT_MAX_BENCHMARK_CAPPED_RATE
    max_latest_average_decision_rounds: float | None = None
    max_latest_benchmark_average_decision_rounds: float | None = None
    max_benchmark_win_rate_drop: float = DEFAULT_MAX_BENCHMARK_WIN_RATE_DROP
    max_consecutive_promotion_failures: int = DEFAULT_MAX_CONSECUTIVE_PROMOTION_FAILURES
    require_benchmark: bool = True
    require_latest_promotion: bool = False

    def __post_init__(self) -> None:
        if self.min_latest_benchmark_games < 0:
            raise ValueError("min_latest_benchmark_games must be non-negative.")
        if self.max_consecutive_promotion_failures < 0:
            raise ValueError("max_consecutive_promotion_failures must be non-negative.")
        if self.max_latest_average_decision_rounds is not None and self.max_latest_average_decision_rounds < 0.0:
            raise ValueError("max_latest_average_decision_rounds must be non-negative.")
        if (
            self.max_latest_benchmark_average_decision_rounds is not None
            and self.max_latest_benchmark_average_decision_rounds < 0.0
        ):
            raise ValueError("max_latest_benchmark_average_decision_rounds must be non-negative.")
        for field_name in (
            "min_latest_benchmark_win_rate",
            "max_latest_collection_capped_rate",
            "max_latest_benchmark_capped_rate",
            "max_benchmark_win_rate_drop",
        ):
            value = float(getattr(self, field_name))
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{field_name} must be between 0 and 1.")


@dataclass(frozen=True)
class RunAuditCheck:
    name: str
    passed: bool
    observed: float | int | str | None
    threshold: float | int | str | None
    message: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "passed": self.passed,
            "observed": self.observed,
            "threshold": self.threshold,
            "message": self.message,
        }


@dataclass(frozen=True)
class RunAuditBenchmarkOpponent:
    opponent_policy_id: str
    wins: int
    games: int
    capped_games: int

    @property
    def win_rate(self) -> float:
        return self.wins / self.games if self.games else 0.0

    @property
    def capped_rate(self) -> float:
        return self.capped_games / self.games if self.games else 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "opponent_policy_id": self.opponent_policy_id,
            "wins": self.wins,
            "games": self.games,
            "capped_games": self.capped_games,
            "win_rate": self.win_rate,
            "capped_rate": self.capped_rate,
        }


@dataclass(frozen=True)
class RunAuditIterationSummary:
    iteration: int
    policy_id: str | None
    checkpoint_path: str | None
    collection_games: int
    collection_capped_rate: float | None
    average_decision_rounds: float | None
    benchmark_win_rate: float | None
    benchmark_games: int
    benchmark_capped_rate: float | None
    benchmark_average_decision_rounds: float | None
    benchmark_opponents: tuple[RunAuditBenchmarkOpponent, ...]
    promotion_recorded: bool | None
    advancement_recorded: bool | None
    advancement_reason: str | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "iteration": self.iteration,
            "policy_id": self.policy_id,
            "checkpoint_path": self.checkpoint_path,
            "collection_games": self.collection_games,
            "collection_capped_rate": self.collection_capped_rate,
            "average_decision_rounds": self.average_decision_rounds,
            "benchmark_win_rate": self.benchmark_win_rate,
            "benchmark_games": self.benchmark_games,
            "benchmark_capped_rate": self.benchmark_capped_rate,
            "benchmark_average_decision_rounds": self.benchmark_average_decision_rounds,
            "benchmark_opponents": [opponent.to_dict() for opponent in self.benchmark_opponents],
            "promotion_recorded": self.promotion_recorded,
            "advancement_recorded": self.advancement_recorded,
            "advancement_reason": self.advancement_reason,
        }


@dataclass(frozen=True)
class RunAuditOpponentRegression:
    opponent_policy_id: str
    latest_win_rate: float
    best_previous_win_rate: float
    drop: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "opponent_policy_id": self.opponent_policy_id,
            "latest_win_rate": self.latest_win_rate,
            "best_previous_win_rate": self.best_previous_win_rate,
            "drop": self.drop,
        }


@dataclass(frozen=True)
class RunAuditResult:
    manifest_path: Path
    schema_version: str
    source_type: str
    iterations: tuple[RunAuditIterationSummary, ...]
    best_benchmark_win_rate: float | None
    latest_benchmark_win_rate: float | None
    latest_collection_capped_rate: float | None
    latest_average_decision_rounds: float | None
    latest_benchmark_capped_rate: float | None
    latest_benchmark_average_decision_rounds: float | None
    benchmark_regressions: tuple[RunAuditOpponentRegression, ...]
    consecutive_promotion_failures: int
    checks: tuple[RunAuditCheck, ...]

    @property
    def latest_iteration(self) -> int | None:
        return self.iterations[-1].iteration if self.iterations else None

    @property
    def passed(self) -> bool:
        return all(check.passed for check in self.checks)

    def to_dict(self) -> dict[str, Any]:
        return {
            "manifest_path": str(self.manifest_path),
            "schema_version": self.schema_version,
            "source_type": self.source_type,
            "iterations": [iteration.to_dict() for iteration in self.iterations],
            "latest_iteration": self.latest_iteration,
            "best_benchmark_win_rate": self.best_benchmark_win_rate,
            "latest_benchmark_win_rate": self.latest_benchmark_win_rate,
            "latest_collection_capped_rate": self.latest_collection_capped_rate,
            "latest_average_decision_rounds": self.latest_average_decision_rounds,
            "latest_benchmark_capped_rate": self.latest_benchmark_capped_rate,
            "latest_benchmark_average_decision_rounds": self.latest_benchmark_average_decision_rounds,
            "benchmark_regressions": [regression.to_dict() for regression in self.benchmark_regressions],
            "consecutive_promotion_failures": self.consecutive_promotion_failures,
            "passed": self.passed,
            "checks": [check.to_dict() for check in self.checks],
        }


class RunAuditFailure(RuntimeError):
    def __init__(self, result: RunAuditResult) -> None:
        self.result = result
        failed = tuple(check.name for check in result.checks if not check.passed)
        failed_summary = ", ".join(failed) if failed else "unknown"
        super().__init__(
            f"run audit failed for {result.manifest_path}: {failed_summary}"
        )


def audit_run(
    path: Path,
    *,
    config: RunAuditConfig = RunAuditConfig(),
) -> RunAuditResult:
    manifest_path = _manifest_path(path)
    manifest = _load_manifest(manifest_path)
    schema_version = str(manifest.get("schema_version") or "")
    source_type = _source_type(schema_version)
    iterations = tuple(
        _iteration_summary(iteration, schema_version=schema_version)
        for iteration in tuple(_mapping(iteration) for iteration in _sequence(manifest.get("iterations", ())))
    )
    if not iterations:
        raise ValueError("run manifest contains no iterations.")

    latest = iterations[-1]
    benchmark_values = tuple(
        iteration.benchmark_win_rate
        for iteration in iterations
        if iteration.benchmark_win_rate is not None
    )
    best_benchmark_win_rate = max(benchmark_values) if benchmark_values else None
    benchmark_regressions = _opponent_regressions(iterations)
    consecutive_promotion_failures = _consecutive_promotion_failures(iterations)
    checks = (
        _latest_collection_capped_check(latest, config),
        *_latest_average_decision_rounds_checks(latest, config),
        *_latest_benchmark_checks(latest, config),
        *_latest_benchmark_average_decision_rounds_checks(latest, config),
        _benchmark_regression_check(iterations, benchmark_regressions, config),
        _promotion_failure_check(consecutive_promotion_failures, config),
        _latest_promotion_check(latest, config),
    )
    return RunAuditResult(
        manifest_path=manifest_path,
        schema_version=schema_version,
        source_type=source_type,
        iterations=iterations,
        best_benchmark_win_rate=best_benchmark_win_rate,
        latest_benchmark_win_rate=latest.benchmark_win_rate,
        latest_collection_capped_rate=latest.collection_capped_rate,
        latest_average_decision_rounds=latest.average_decision_rounds,
        latest_benchmark_capped_rate=latest.benchmark_capped_rate,
        latest_benchmark_average_decision_rounds=latest.benchmark_average_decision_rounds,
        benchmark_regressions=benchmark_regressions,
        consecutive_promotion_failures=consecutive_promotion_failures,
        checks=checks,
    )


def enforce_run_audit(
    path: Path,
    *,
    config: RunAuditConfig = RunAuditConfig(),
) -> RunAuditResult:
    result = audit_run(path, config=config)
    if not result.passed:
        raise RunAuditFailure(result)
    return result


def _source_type(schema_version: str) -> str:
    if schema_version == SELFPLAY_RUN_SCHEMA_VERSION:
        return "linear_selfplay"
    if schema_version == NEURAL_SELFPLAY_RUN_SCHEMA_VERSION:
        return "neural_selfplay"
    raise ValueError(f"Unsupported run manifest schema: {schema_version!r}.")


def _iteration_summary(
    iteration: Mapping[str, Any],
    *,
    schema_version: str,
) -> RunAuditIterationSummary:
    policy_id = _policy_id(iteration, schema_version=schema_version)
    collection_metrics = _mapping(iteration.get("collection_metrics", {}))
    collection_games = int(collection_metrics.get("games", 0))
    benchmark = _optional_mapping(iteration.get("benchmark"))
    benchmark_summary = _benchmark_summary(benchmark, policy_id)
    promotion = _optional_mapping(iteration.get("promotion"))
    advancement = _optional_mapping(iteration.get("advancement"))
    return RunAuditIterationSummary(
        iteration=int(iteration.get("iteration", 0)),
        policy_id=policy_id,
        checkpoint_path=_optional_str(iteration.get("checkpoint_path")),
        collection_games=collection_games,
        collection_capped_rate=_capped_rate(collection_metrics),
        average_decision_rounds=_optional_float(collection_metrics.get("average_decision_rounds")),
        benchmark_win_rate=benchmark_summary.win_rate if benchmark_summary.games else None,
        benchmark_games=benchmark_summary.games,
        benchmark_capped_rate=benchmark_summary.capped_rate if benchmark_summary.games else None,
        benchmark_average_decision_rounds=_benchmark_average_decision_rounds(benchmark),
        benchmark_opponents=tuple(
            RunAuditBenchmarkOpponent(
                opponent_policy_id=opponent.opponent_policy_id,
                wins=opponent.wins,
                games=opponent.games,
                capped_games=opponent.capped_games,
            )
            for opponent in benchmark_summary.opponents
        ),
        promotion_recorded=(
            None
            if promotion is None
            else bool(promotion.get("recorded"))
        ),
        advancement_recorded=(
            None
            if advancement is None
            else bool(advancement.get("advance_collector"))
        ),
        advancement_reason=(
            None
            if advancement is None or advancement.get("reason") is None
            else str(advancement.get("reason"))
        ),
    )


def _policy_id(iteration: Mapping[str, Any], *, schema_version: str) -> str | None:
    training = _mapping(iteration.get("training", {}))
    if schema_version == SELFPLAY_RUN_SCHEMA_VERSION:
        model = _mapping(training.get("model", {}))
        return _optional_str(model.get("policy_id"))
    if schema_version == NEURAL_SELFPLAY_RUN_SCHEMA_VERSION:
        model_config = _mapping(training.get("model_config", {}))
        return _optional_str(model_config.get("policy_id"))
    raise ValueError(f"Unsupported run manifest schema: {schema_version!r}.")


def _latest_collection_capped_check(
    latest: RunAuditIterationSummary,
    config: RunAuditConfig,
) -> RunAuditCheck:
    observed = latest.collection_capped_rate
    return RunAuditCheck(
        name="latest_collection_capped_rate",
        passed=observed is not None and observed <= config.max_latest_collection_capped_rate,
        observed=observed,
        threshold=config.max_latest_collection_capped_rate,
        message=(
            "latest collection capped-game rate is within limit"
            if observed is not None
            else "latest collection capped-game rate is unavailable"
        ),
    )


def _latest_average_decision_rounds_checks(
    latest: RunAuditIterationSummary,
    config: RunAuditConfig,
) -> tuple[RunAuditCheck, ...]:
    if config.max_latest_average_decision_rounds is None:
        return ()
    observed = latest.average_decision_rounds
    if observed is None:
        message = "latest collection average decision rounds are unavailable"
    elif observed <= config.max_latest_average_decision_rounds:
        message = "latest collection average decision rounds are within limit"
    else:
        message = "latest collection average decision rounds exceed limit"
    return (
        RunAuditCheck(
            name="latest_average_decision_rounds",
            passed=observed is not None and observed <= config.max_latest_average_decision_rounds,
            observed=observed,
            threshold=config.max_latest_average_decision_rounds,
            message=message,
        ),
    )


def _latest_benchmark_checks(
    latest: RunAuditIterationSummary,
    config: RunAuditConfig,
) -> tuple[RunAuditCheck, ...]:
    if latest.benchmark_games <= 0:
        return (
            RunAuditCheck(
                name="latest_benchmark_available",
                passed=not config.require_benchmark,
                observed=None,
                threshold="required" if config.require_benchmark else "optional",
                message="latest benchmark evidence is unavailable",
            ),
        )
    return (
        RunAuditCheck(
            name="latest_benchmark_games",
            passed=latest.benchmark_games >= config.min_latest_benchmark_games,
            observed=latest.benchmark_games,
            threshold=config.min_latest_benchmark_games,
            message="latest benchmark has enough games",
        ),
        RunAuditCheck(
            name="latest_benchmark_win_rate",
            passed=(
                latest.benchmark_win_rate is not None
                and latest.benchmark_win_rate >= config.min_latest_benchmark_win_rate
            ),
            observed=latest.benchmark_win_rate,
            threshold=config.min_latest_benchmark_win_rate,
            message="latest benchmark win rate meets the run-health floor",
        ),
        RunAuditCheck(
            name="latest_benchmark_capped_rate",
            passed=(
                latest.benchmark_capped_rate is not None
                and latest.benchmark_capped_rate <= config.max_latest_benchmark_capped_rate
            ),
            observed=latest.benchmark_capped_rate,
            threshold=config.max_latest_benchmark_capped_rate,
            message="latest benchmark capped-game rate is within limit",
        ),
    )


def _latest_benchmark_average_decision_rounds_checks(
    latest: RunAuditIterationSummary,
    config: RunAuditConfig,
) -> tuple[RunAuditCheck, ...]:
    if config.max_latest_benchmark_average_decision_rounds is None:
        return ()
    observed = latest.benchmark_average_decision_rounds
    if observed is None:
        passed = latest.benchmark_games <= 0 and not config.require_benchmark
        message = (
            "latest benchmark average decision rounds are unavailable because latest benchmark is optional"
            if passed
            else "latest benchmark average decision rounds are unavailable"
        )
    elif observed <= config.max_latest_benchmark_average_decision_rounds:
        passed = True
        message = "latest benchmark average decision rounds are within limit"
    else:
        passed = False
        message = "latest benchmark average decision rounds exceed limit"
    return (
        RunAuditCheck(
            name="latest_benchmark_average_decision_rounds",
            passed=passed,
            observed=observed,
            threshold=config.max_latest_benchmark_average_decision_rounds,
            message=message,
        ),
    )


def _benchmark_regression_check(
    iterations: tuple[RunAuditIterationSummary, ...],
    regressions: tuple[RunAuditOpponentRegression, ...],
    config: RunAuditConfig,
) -> RunAuditCheck:
    previous_benchmark_count = sum(1 for iteration in iterations[:-1] if iteration.benchmark_games > 0)
    latest = iterations[-1]
    if not previous_benchmark_count:
        return RunAuditCheck(
            name="benchmark_win_rate_drop_by_opponent",
            passed=True,
            observed=None,
            threshold=config.max_benchmark_win_rate_drop,
            message="no prior benchmark exists for same-opponent regression comparison",
        )
    if latest.benchmark_games <= 0:
        return RunAuditCheck(
            name="benchmark_win_rate_drop_by_opponent",
            passed=False,
            observed="missing_latest_benchmark",
            threshold=config.max_benchmark_win_rate_drop,
            message="latest benchmark is required for same-opponent regression comparison",
        )
    if not regressions:
        return RunAuditCheck(
            name="benchmark_win_rate_drop_by_opponent",
            passed=False,
            observed="no_shared_opponent",
            threshold=config.max_benchmark_win_rate_drop,
            message="latest benchmark has no opponent overlap with prior benchmark evidence",
        )
    max_drop = max(regression.drop for regression in regressions)
    return RunAuditCheck(
        name="benchmark_win_rate_drop_by_opponent",
        passed=max_drop <= config.max_benchmark_win_rate_drop,
        observed=max_drop,
        threshold=config.max_benchmark_win_rate_drop,
        message="latest same-opponent benchmark win rates have not regressed too far from previous best",
    )


def _promotion_failure_check(
    consecutive_promotion_failures: int,
    config: RunAuditConfig,
) -> RunAuditCheck:
    return RunAuditCheck(
        name="consecutive_promotion_failures",
        passed=consecutive_promotion_failures <= config.max_consecutive_promotion_failures,
        observed=consecutive_promotion_failures,
        threshold=config.max_consecutive_promotion_failures,
        message="trailing promotion failures are within limit",
    )


def _latest_promotion_check(
    latest: RunAuditIterationSummary,
    config: RunAuditConfig,
) -> RunAuditCheck:
    if not config.require_latest_promotion:
        return RunAuditCheck(
            name="latest_promotion_recorded",
            passed=True,
            observed=latest.promotion_recorded,
            threshold="optional",
            message="latest promotion is optional for this audit",
        )
    return RunAuditCheck(
        name="latest_promotion_recorded",
        passed=latest.promotion_recorded is True,
        observed=latest.promotion_recorded,
        threshold=True,
        message="latest iteration must have a recorded promotion",
    )


def _consecutive_promotion_failures(iterations: tuple[RunAuditIterationSummary, ...]) -> int:
    failures = 0
    for iteration in reversed(iterations):
        if iteration.promotion_recorded is False:
            failures += 1
            continue
        break
    return failures


def _opponent_regressions(
    iterations: tuple[RunAuditIterationSummary, ...],
) -> tuple[RunAuditOpponentRegression, ...]:
    if len(iterations) < 2:
        return ()
    latest = iterations[-1]
    previous_best: dict[str, float] = {}
    for iteration in iterations[:-1]:
        for opponent in iteration.benchmark_opponents:
            previous_best[opponent.opponent_policy_id] = max(
                opponent.win_rate,
                previous_best.get(opponent.opponent_policy_id, 0.0),
            )
    regressions: list[RunAuditOpponentRegression] = []
    for opponent in latest.benchmark_opponents:
        best_previous = previous_best.get(opponent.opponent_policy_id)
        if best_previous is None:
            continue
        regressions.append(
            RunAuditOpponentRegression(
                opponent_policy_id=opponent.opponent_policy_id,
                latest_win_rate=opponent.win_rate,
                best_previous_win_rate=best_previous,
                drop=max(0.0, best_previous - opponent.win_rate),
            )
        )
    return tuple(regressions)


def _benchmark_average_decision_rounds(benchmark: Mapping[str, Any] | None) -> float | None:
    if benchmark is None:
        return None
    direct = benchmark.get("average_decision_rounds")
    if direct is not None:
        return float(direct)
    total_games = 0
    total_decision_rounds = 0.0
    for result in tuple(_mapping(result) for result in _sequence(benchmark.get("matchups", ()))):
        metrics = _mapping(result.get("metrics", {}))
        games = int(metrics.get("games", 0))
        average = metrics.get("average_decision_rounds")
        if games <= 0 or average is None:
            continue
        total_games += games
        total_decision_rounds += float(average) * games
    if total_games <= 0:
        return None
    return total_decision_rounds / total_games


def _optional_mapping(value: Any) -> Mapping[str, Any] | None:
    if value is None:
        return None
    return _mapping(value)


def _optional_str(value: Any) -> str | None:
    if value is None:
        return None
    return str(value)


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    return float(value)
