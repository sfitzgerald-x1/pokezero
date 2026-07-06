"""Run-level health audits for self-play experiment manifests."""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping

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
from .opponents import HISTORICAL_OPPONENT_SELECTION_MODES, historical_opponent_policy_specs
from .selfplay import SELFPLAY_RUN_SCHEMA_VERSION


DEFAULT_MAX_BENCHMARK_WIN_RATE_DROP = 0.05
DEFAULT_MAX_CONSECUTIVE_PROMOTION_FAILURES = 1
DEFAULT_AUDIT_CALIBRATION_MARGIN = 0.10
DEFAULT_REQUIRED_BENCHMARK_OPPONENTS = ("random-legal", "simple-legal")
AUDIT_CALIBRATION_AGGREGATE_MODES = ("median", "envelope")
RUN_AUDIT_CONFIG_SCHEMA_VERSION = "pokezero.run_audit_config.v1"
_THRESHOLD_EPSILON = 1e-12
RUN_AUDIT_CHECK_SEVERITIES = ("error", "warning")
RUN_AUDIT_CHECK_NAMES = (
    "latest_collection_capped_rate",
    "promoted_opponent_pool_requirement",
    "latest_average_decision_rounds",
    "latest_benchmark_available",
    "latest_benchmark_games",
    "latest_benchmark_win_rate",
    "latest_benchmark_capped_rate",
    "latest_benchmark_average_decision_rounds",
    "latest_process_peak_rss_mb",
    "latest_benchmark_opponent_coverage",
    "benchmark_win_rate_drop_by_opponent",
    "consecutive_promotion_failures",
    "latest_promotion_recorded",
)
RUN_AUDIT_RUNTIME_HEALTH_CHECK_NAMES = frozenset(
    (
        "latest_collection_capped_rate",
        "promoted_opponent_pool_requirement",
        "latest_average_decision_rounds",
        "latest_benchmark_available",
        "latest_benchmark_games",
        "latest_benchmark_win_rate",
        "latest_benchmark_capped_rate",
        "latest_benchmark_average_decision_rounds",
        "latest_process_peak_rss_mb",
        "latest_benchmark_opponent_coverage",
    )
)
RUN_AUDIT_PROMOTION_STRENGTH_CHECK_NAMES = frozenset(
    (
        "benchmark_win_rate_drop_by_opponent",
        "consecutive_promotion_failures",
        "latest_promotion_recorded",
    )
)
RUN_AUDIT_CLASSIFIED_CHECK_NAMES = RUN_AUDIT_RUNTIME_HEALTH_CHECK_NAMES | RUN_AUDIT_PROMOTION_STRENGTH_CHECK_NAMES
if RUN_AUDIT_CLASSIFIED_CHECK_NAMES != frozenset(RUN_AUDIT_CHECK_NAMES):
    _missing_classified_checks = sorted(set(RUN_AUDIT_CHECK_NAMES) - RUN_AUDIT_CLASSIFIED_CHECK_NAMES)
    _extra_classified_checks = sorted(RUN_AUDIT_CLASSIFIED_CHECK_NAMES - set(RUN_AUDIT_CHECK_NAMES))
    raise RuntimeError(
        "Run-audit check classification is out of sync: "
        f"missing={_missing_classified_checks}, extra={_extra_classified_checks}"
    )


def runtime_health_failed_check_names(check_names: Iterable[str]) -> tuple[str, ...]:
    return tuple(
        check
        for check in check_names
        if check in RUN_AUDIT_RUNTIME_HEALTH_CHECK_NAMES or check not in RUN_AUDIT_CLASSIFIED_CHECK_NAMES
    )


def promotion_strength_failed_check_names(check_names: Iterable[str]) -> tuple[str, ...]:
    return tuple(check for check in check_names if check in RUN_AUDIT_PROMOTION_STRENGTH_CHECK_NAMES)


def _string_tuple(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        raise ValueError("string sequence value must be an array, not a string.")
    return tuple(str(item) for item in _sequence(value))


def _normalized_warning_check_names(value: Any) -> tuple[str, ...]:
    names: list[str] = []
    for raw_name in _string_tuple(value):
        name = raw_name.strip()
        if not name:
            raise ValueError("warning_check_names entries must be non-empty.")
        if name not in RUN_AUDIT_CHECK_NAMES:
            choices = ", ".join(RUN_AUDIT_CHECK_NAMES)
            raise ValueError(f"unknown warning_check_names entry {name!r}; choose one of: {choices}")
        if name not in names:
            names.append(name)
    return tuple(names)


@dataclass(frozen=True)
class RunAuditConfig:
    min_latest_benchmark_win_rate: float = DEFAULT_MIN_BENCHMARK_WIN_RATE
    min_latest_benchmark_games: int = DEFAULT_MIN_BENCHMARK_GAMES
    max_latest_collection_capped_rate: float = DEFAULT_MAX_COLLECTION_CAPPED_RATE
    max_latest_benchmark_capped_rate: float = DEFAULT_MAX_BENCHMARK_CAPPED_RATE
    max_latest_average_decision_rounds: float | None = None
    max_latest_benchmark_average_decision_rounds: float | None = None
    max_latest_process_peak_rss_mb: float | None = None
    max_benchmark_win_rate_drop: float = DEFAULT_MAX_BENCHMARK_WIN_RATE_DROP
    max_consecutive_promotion_failures: int = DEFAULT_MAX_CONSECUTIVE_PROMOTION_FAILURES
    require_benchmark: bool = True
    require_latest_promotion: bool = False
    require_benchmark_opponent_coverage: bool = True
    warning_check_names: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "warning_check_names", _normalized_warning_check_names(self.warning_check_names))
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
        if self.max_latest_process_peak_rss_mb is not None and self.max_latest_process_peak_rss_mb < 0.0:
            raise ValueError("max_latest_process_peak_rss_mb must be non-negative.")
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
    severity: str = "error"

    def __post_init__(self) -> None:
        if self.severity not in RUN_AUDIT_CHECK_SEVERITIES:
            choices = ", ".join(RUN_AUDIT_CHECK_SEVERITIES)
            raise ValueError(f"unknown run-audit check severity {self.severity!r}; choose one of: {choices}")

    @property
    def blocking_failed(self) -> bool:
        return not self.passed and self.severity != "warning"

    @property
    def warning_failed(self) -> bool:
        return not self.passed and self.severity == "warning"

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "passed": self.passed,
            "observed": self.observed,
            "threshold": self.threshold,
            "message": self.message,
            "severity": self.severity,
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
    collection_games_per_hour: float | None
    collection_capped_rate: float | None
    collection_peak_rss_mb: float | None
    collection_peak_rss_mb_by_phase: Mapping[str, float | None]
    average_decision_rounds: float | None
    benchmark_win_rate: float | None
    benchmark_games: int
    benchmark_games_per_hour: float | None
    benchmark_capped_rate: float | None
    benchmark_peak_rss_mb: float | None
    benchmark_average_decision_rounds: float | None
    benchmark_opponents: tuple[RunAuditBenchmarkOpponent, ...]
    promotion_recorded: bool | None
    advancement_recorded: bool | None
    advancement_reason: str | None
    process_peak_rss_mb_by_phase: Mapping[str, float | None]

    def to_dict(self) -> dict[str, Any]:
        return {
            "iteration": self.iteration,
            "policy_id": self.policy_id,
            "checkpoint_path": self.checkpoint_path,
            "collection_games": self.collection_games,
            "collection_games_per_hour": self.collection_games_per_hour,
            "collection_capped_rate": self.collection_capped_rate,
            "collection_peak_rss_mb": self.collection_peak_rss_mb,
            "collection_peak_rss_mb_by_phase": dict(self.collection_peak_rss_mb_by_phase),
            "average_decision_rounds": self.average_decision_rounds,
            "benchmark_win_rate": self.benchmark_win_rate,
            "benchmark_games": self.benchmark_games,
            "benchmark_games_per_hour": self.benchmark_games_per_hour,
            "benchmark_capped_rate": self.benchmark_capped_rate,
            "benchmark_peak_rss_mb": self.benchmark_peak_rss_mb,
            "benchmark_average_decision_rounds": self.benchmark_average_decision_rounds,
            "benchmark_opponents": [opponent.to_dict() for opponent in self.benchmark_opponents],
            "promotion_recorded": self.promotion_recorded,
            "advancement_recorded": self.advancement_recorded,
            "advancement_reason": self.advancement_reason,
            "process_peak_rss_mb_by_phase": dict(self.process_peak_rss_mb_by_phase),
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
    source_metadata: Mapping[str, Any]
    iterations: tuple[RunAuditIterationSummary, ...]
    best_benchmark_win_rate: float | None
    latest_benchmark_win_rate: float | None
    latest_collection_capped_rate: float | None
    latest_average_decision_rounds: float | None
    latest_benchmark_capped_rate: float | None
    latest_benchmark_average_decision_rounds: float | None
    latest_process_peak_rss_mb: float | None
    missing_latest_benchmark_opponents: tuple[str, ...]
    benchmark_regressions: tuple[RunAuditOpponentRegression, ...]
    consecutive_promotion_failures: int
    checks: tuple[RunAuditCheck, ...]

    @property
    def latest_iteration(self) -> int | None:
        return self.iterations[-1].iteration if self.iterations else None

    @property
    def passed(self) -> bool:
        return not self.blocking_failed_checks

    @property
    def blocking_failed_checks(self) -> tuple[RunAuditCheck, ...]:
        return tuple(check for check in self.checks if check.blocking_failed)

    @property
    def warning_failed_checks(self) -> tuple[RunAuditCheck, ...]:
        return tuple(check for check in self.checks if check.warning_failed)

    def to_dict(self) -> dict[str, Any]:
        return {
            "manifest_path": str(self.manifest_path),
            "schema_version": self.schema_version,
            "source_type": self.source_type,
            "source_metadata": dict(self.source_metadata),
            "iterations": [iteration.to_dict() for iteration in self.iterations],
            "latest_iteration": self.latest_iteration,
            "best_benchmark_win_rate": self.best_benchmark_win_rate,
            "latest_benchmark_win_rate": self.latest_benchmark_win_rate,
            "latest_collection_capped_rate": self.latest_collection_capped_rate,
            "latest_average_decision_rounds": self.latest_average_decision_rounds,
            "latest_benchmark_capped_rate": self.latest_benchmark_capped_rate,
            "latest_benchmark_average_decision_rounds": self.latest_benchmark_average_decision_rounds,
            "latest_process_peak_rss_mb": self.latest_process_peak_rss_mb,
            "missing_latest_benchmark_opponents": list(self.missing_latest_benchmark_opponents),
            "benchmark_regressions": [regression.to_dict() for regression in self.benchmark_regressions],
            "consecutive_promotion_failures": self.consecutive_promotion_failures,
            "passed": self.passed,
            "failed_checks": [check.name for check in self.blocking_failed_checks],
            "warning_checks": [check.name for check in self.warning_failed_checks],
            "checks": [check.to_dict() for check in self.checks],
        }


@dataclass(frozen=True)
class RunComparisonEntry:
    label: str
    manifest_path: Path
    source_type: str
    source_metadata: Mapping[str, Any]
    iteration_count: int
    latest_iteration: int | None
    latest_policy_id: str | None
    latest_checkpoint_path: str | None
    latest_benchmark_win_rate: float | None
    best_benchmark_win_rate: float | None
    best_benchmark_games: int
    latest_benchmark_games: int
    latest_collection_games_per_hour: float | None
    latest_benchmark_games_per_hour: float | None
    latest_collection_capped_rate: float | None
    latest_benchmark_capped_rate: float | None
    latest_collection_peak_rss_mb: float | None
    latest_benchmark_peak_rss_mb: float | None
    latest_process_peak_rss_mb: float | None
    latest_average_decision_rounds: float | None
    latest_benchmark_average_decision_rounds: float | None
    latest_promotion_recorded: bool | None
    latest_advancement_recorded: bool | None
    latest_advancement_reason: str | None
    audit_profile: str | None = None
    audit_passed: bool | None = None
    audit_failed_checks: tuple[str, ...] = ()
    audit_warning_checks: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "manifest_path": str(self.manifest_path),
            "source_type": self.source_type,
            "source_metadata": dict(self.source_metadata),
            "iteration_count": self.iteration_count,
            "latest_iteration": self.latest_iteration,
            "latest_policy_id": self.latest_policy_id,
            "latest_checkpoint_path": self.latest_checkpoint_path,
            "latest_benchmark_win_rate": self.latest_benchmark_win_rate,
            "best_benchmark_win_rate": self.best_benchmark_win_rate,
            "best_benchmark_games": self.best_benchmark_games,
            "latest_benchmark_games": self.latest_benchmark_games,
            "latest_collection_games_per_hour": self.latest_collection_games_per_hour,
            "latest_benchmark_games_per_hour": self.latest_benchmark_games_per_hour,
            "latest_collection_capped_rate": self.latest_collection_capped_rate,
            "latest_benchmark_capped_rate": self.latest_benchmark_capped_rate,
            "latest_collection_peak_rss_mb": self.latest_collection_peak_rss_mb,
            "latest_benchmark_peak_rss_mb": self.latest_benchmark_peak_rss_mb,
            "latest_process_peak_rss_mb": self.latest_process_peak_rss_mb,
            "latest_average_decision_rounds": self.latest_average_decision_rounds,
            "latest_benchmark_average_decision_rounds": self.latest_benchmark_average_decision_rounds,
            "latest_promotion_recorded": self.latest_promotion_recorded,
            "latest_advancement_recorded": self.latest_advancement_recorded,
            "latest_advancement_reason": self.latest_advancement_reason,
            "audit_profile": self.audit_profile,
            "audit_passed": self.audit_passed,
            "audit_failed_checks": list(self.audit_failed_checks),
            "audit_warning_checks": list(self.audit_warning_checks),
        }


@dataclass(frozen=True)
class RunComparisonError:
    label: str
    path: str
    error: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "path": self.path,
            "error": self.error,
        }


@dataclass(frozen=True)
class RunComparisonResult:
    entries: tuple[RunComparisonEntry, ...]
    errors: tuple[RunComparisonError, ...] = ()
    min_benchmark_games: int = DEFAULT_MIN_BENCHMARK_GAMES
    audit_profile: str | None = None

    @property
    def best_latest_benchmark_entry(self) -> RunComparisonEntry | None:
        return _best_entry_by_optional_value(
            tuple(
                entry
                for entry in self.entries
                if entry.latest_benchmark_games >= self.min_benchmark_games and entry.audit_passed is not False
            ),
            "latest_benchmark_win_rate",
        )

    @property
    def best_historical_benchmark_entry(self) -> RunComparisonEntry | None:
        return _best_entry_by_optional_value(
            tuple(
                entry
                for entry in self.entries
                if entry.best_benchmark_games >= self.min_benchmark_games and entry.audit_passed is not False
            ),
            "best_benchmark_win_rate",
        )

    @property
    def audit_failed(self) -> bool:
        return any(entry.audit_passed is False for entry in self.entries)

    def to_dict(self) -> dict[str, Any]:
        return {
            "entries": [entry.to_dict() for entry in self.entries],
            "errors": [error.to_dict() for error in self.errors],
            "min_benchmark_games": self.min_benchmark_games,
            "audit_profile": self.audit_profile,
            "audit_failed": self.audit_failed,
            "best_latest_benchmark_label": (
                self.best_latest_benchmark_entry.label
                if self.best_latest_benchmark_entry is not None
                else None
            ),
            "best_historical_benchmark_label": (
                self.best_historical_benchmark_entry.label
                if self.best_historical_benchmark_entry is not None
                else None
            ),
        }


@dataclass(frozen=True)
class RunAuditCalibrationResult:
    manifest_path: Path
    schema_version: str
    source_type: str
    iteration_count: int
    benchmark_iteration_count: int
    margin: float
    require_benchmark: bool
    min_latest_benchmark_win_rate: float | None
    min_latest_benchmark_games: int
    max_latest_collection_capped_rate: float | None
    max_latest_benchmark_capped_rate: float | None
    max_latest_average_decision_rounds: float | None
    max_latest_benchmark_average_decision_rounds: float | None
    max_latest_process_peak_rss_mb: float | None
    max_benchmark_win_rate_drop: float | None
    max_consecutive_promotion_failures: int
    require_benchmark_opponent_coverage: bool
    notes: tuple[str, ...] = ()

    def suggested_config(self) -> dict[str, float | int | bool | None]:
        return _suggested_audit_config(self)

    def suggested_cli_flags(self) -> tuple[str, ...]:
        return _suggested_audit_cli_flags(self)

    def suggested_post_iteration_cli_flags(self) -> tuple[str, ...]:
        return _suggested_post_iteration_audit_cli_flags(self)

    def to_dict(self) -> dict[str, Any]:
        return {
            "manifest_path": str(self.manifest_path),
            "schema_version": self.schema_version,
            "source_type": self.source_type,
            "iteration_count": self.iteration_count,
            "benchmark_iteration_count": self.benchmark_iteration_count,
            "margin": self.margin,
            "suggested_config": self.suggested_config(),
            "suggested_cli_flags": list(self.suggested_cli_flags()),
            "suggested_post_iteration_cli_flags": list(self.suggested_post_iteration_cli_flags()),
            "notes": list(self.notes),
        }


@dataclass(frozen=True)
class MultiRunAuditCalibrationResult:
    paths: tuple[Path, ...]
    source_type: str
    aggregate_mode: str
    run_count: int
    iteration_count: int
    benchmark_iteration_count: int
    margin: float
    calibrations: tuple[RunAuditCalibrationResult, ...]
    require_benchmark: bool
    min_latest_benchmark_win_rate: float | None
    min_latest_benchmark_games: int
    max_latest_collection_capped_rate: float | None
    max_latest_benchmark_capped_rate: float | None
    max_latest_average_decision_rounds: float | None
    max_latest_benchmark_average_decision_rounds: float | None
    max_latest_process_peak_rss_mb: float | None
    max_benchmark_win_rate_drop: float | None
    max_consecutive_promotion_failures: int
    require_benchmark_opponent_coverage: bool
    notes: tuple[str, ...] = ()

    def suggested_config(self) -> dict[str, float | int | bool | None]:
        return _suggested_audit_config(self)

    def suggested_cli_flags(self) -> tuple[str, ...]:
        return _suggested_audit_cli_flags(self)

    def suggested_post_iteration_cli_flags(self) -> tuple[str, ...]:
        return _suggested_post_iteration_audit_cli_flags(self)

    def to_dict(self) -> dict[str, Any]:
        return {
            "paths": [str(path) for path in self.paths],
            "source_type": self.source_type,
            "aggregate_mode": self.aggregate_mode,
            "run_count": self.run_count,
            "iteration_count": self.iteration_count,
            "benchmark_iteration_count": self.benchmark_iteration_count,
            "margin": self.margin,
            "suggested_config": self.suggested_config(),
            "suggested_cli_flags": list(self.suggested_cli_flags()),
            "suggested_post_iteration_cli_flags": list(self.suggested_post_iteration_cli_flags()),
            "notes": list(self.notes),
            "sources": [calibration.to_dict() for calibration in self.calibrations],
        }


class RunAuditFailure(RuntimeError):
    def __init__(self, result: RunAuditResult) -> None:
        self.result = result
        failed = tuple(check.name for check in result.blocking_failed_checks)
        failed_summary = ", ".join(failed) if failed else "unknown"
        super().__init__(
            f"run audit failed for {result.manifest_path}: {failed_summary}"
        )


def run_audit_config_to_dict(config: RunAuditConfig) -> dict[str, float | int | bool | None | list[str]]:
    return {
        "min_latest_benchmark_win_rate": config.min_latest_benchmark_win_rate,
        "min_latest_benchmark_games": config.min_latest_benchmark_games,
        "max_latest_collection_capped_rate": config.max_latest_collection_capped_rate,
        "max_latest_benchmark_capped_rate": config.max_latest_benchmark_capped_rate,
        "max_latest_average_decision_rounds": config.max_latest_average_decision_rounds,
        "max_latest_benchmark_average_decision_rounds": config.max_latest_benchmark_average_decision_rounds,
        "max_latest_process_peak_rss_mb": config.max_latest_process_peak_rss_mb,
        "max_benchmark_win_rate_drop": config.max_benchmark_win_rate_drop,
        "max_consecutive_promotion_failures": config.max_consecutive_promotion_failures,
        "require_benchmark": config.require_benchmark,
        "require_latest_promotion": config.require_latest_promotion,
        "require_benchmark_opponent_coverage": config.require_benchmark_opponent_coverage,
        "warning_check_names": list(config.warning_check_names),
    }


def run_audit_config_from_dict(payload: Mapping[str, Any]) -> RunAuditConfig:
    return RunAuditConfig(
        min_latest_benchmark_win_rate=float(payload["min_latest_benchmark_win_rate"]),
        min_latest_benchmark_games=int(payload["min_latest_benchmark_games"]),
        max_latest_collection_capped_rate=float(payload["max_latest_collection_capped_rate"]),
        max_latest_benchmark_capped_rate=float(payload["max_latest_benchmark_capped_rate"]),
        max_latest_average_decision_rounds=_optional_float(payload.get("max_latest_average_decision_rounds")),
        max_latest_benchmark_average_decision_rounds=_optional_float(
            payload.get("max_latest_benchmark_average_decision_rounds")
        ),
        max_latest_process_peak_rss_mb=_optional_float(payload.get("max_latest_process_peak_rss_mb")),
        max_benchmark_win_rate_drop=float(payload["max_benchmark_win_rate_drop"]),
        max_consecutive_promotion_failures=int(payload["max_consecutive_promotion_failures"]),
        require_benchmark=_required_bool(payload, "require_benchmark"),
        require_latest_promotion=_required_bool(payload, "require_latest_promotion"),
        require_benchmark_opponent_coverage=_required_bool(payload, "require_benchmark_opponent_coverage"),
        warning_check_names=_string_tuple(payload.get("warning_check_names", ())),
    )


def load_run_audit_config(path: Path) -> RunAuditConfig:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"run audit config must be a JSON object: {path}")
    if payload.get("schema_version") != RUN_AUDIT_CONFIG_SCHEMA_VERSION:
        raise ValueError(
            "Unsupported run audit config schema: "
            f"{payload.get('schema_version')!r}; expected {RUN_AUDIT_CONFIG_SCHEMA_VERSION!r}."
        )
    config = payload.get("config")
    if not isinstance(config, Mapping):
        raise ValueError(f"run audit config payload must include a config object: {path}")
    return run_audit_config_from_dict(config)


def run_audit_config_payload(
    config: RunAuditConfig,
    *,
    source: Mapping[str, Any] | None = None,
    calibration: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "schema_version": RUN_AUDIT_CONFIG_SCHEMA_VERSION,
        "config": run_audit_config_to_dict(config),
    }
    if source is not None:
        payload["source"] = dict(source)
    if calibration is not None:
        payload["calibration"] = dict(calibration)
    return payload


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
    missing_latest_benchmark_opponents = _missing_latest_benchmark_opponents(iterations)
    consecutive_promotion_failures = _consecutive_promotion_failures(iterations)
    checks = _apply_warning_check_severity(
        (
            _latest_collection_capped_check(latest, config),
            _promoted_opponent_pool_requirement_check(manifest),
            *_latest_average_decision_rounds_checks(latest, config),
            *_latest_benchmark_checks(latest, config),
            *_latest_benchmark_average_decision_rounds_checks(latest, config),
            *_latest_process_peak_rss_checks(latest, config),
            _benchmark_opponent_coverage_check(latest, missing_latest_benchmark_opponents, config),
            _benchmark_regression_check(iterations, benchmark_regressions, config),
            _promotion_failure_check(consecutive_promotion_failures, config),
            _latest_promotion_check(latest, config),
        ),
        warning_check_names=config.warning_check_names,
    )
    return RunAuditResult(
        manifest_path=manifest_path,
        schema_version=schema_version,
        source_type=source_type,
        source_metadata=_manifest_source_metadata(manifest),
        iterations=iterations,
        best_benchmark_win_rate=best_benchmark_win_rate,
        latest_benchmark_win_rate=latest.benchmark_win_rate,
        latest_collection_capped_rate=latest.collection_capped_rate,
        latest_average_decision_rounds=latest.average_decision_rounds,
        latest_benchmark_capped_rate=latest.benchmark_capped_rate,
        latest_benchmark_average_decision_rounds=latest.benchmark_average_decision_rounds,
        latest_process_peak_rss_mb=_latest_process_peak_rss_mb(latest),
        missing_latest_benchmark_opponents=missing_latest_benchmark_opponents,
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


def compare_run_manifests(paths: Iterable[Path]) -> RunComparisonResult:
    return compare_run_manifests_with_threshold(paths, min_benchmark_games=DEFAULT_MIN_BENCHMARK_GAMES)


def compare_run_manifests_with_threshold(
    paths: Iterable[Path],
    *,
    min_benchmark_games: int,
    audit_config: RunAuditConfig | None = None,
    audit_profile: str | None = None,
) -> RunComparisonResult:
    if min_benchmark_games < 0:
        raise ValueError("min_benchmark_games must be non-negative.")
    selected_paths = tuple(paths)
    if not selected_paths:
        raise ValueError("at least one run path is required.")
    entries: list[RunComparisonEntry] = []
    errors: list[RunComparisonError] = []
    for path in selected_paths:
        try:
            entries.append(
                _comparison_entry(
                    path,
                    audit_config=audit_config,
                    audit_profile=audit_profile,
                )
            )
        except Exception as exc:
            errors.append(
                RunComparisonError(
                    label=_comparison_error_label(path),
                    path=str(path),
                    error=str(exc),
                )
            )
    return RunComparisonResult(
        entries=tuple(entries),
        errors=tuple(errors),
        min_benchmark_games=min_benchmark_games,
        audit_profile=audit_profile,
    )


def calibrate_run_audit(
    path: Path,
    *,
    margin: float = DEFAULT_AUDIT_CALIBRATION_MARGIN,
) -> RunAuditCalibrationResult:
    if margin < 0.0:
        raise ValueError("margin must be non-negative.")
    result = audit_run(path, config=_permissive_audit_config())
    benchmark_iterations = tuple(iteration for iteration in result.iterations if iteration.benchmark_games > 0)
    notes: list[str] = []
    if not benchmark_iterations:
        notes.append("No benchmark iterations were present; suggested audit config allows missing benchmarks.")
    observed_drop = _max_observed_same_opponent_drop(result.iterations)
    if benchmark_iterations and observed_drop is None:
        notes.append("No same-opponent regression history was present; using the default benchmark drop threshold.")
    min_benchmark_games = (
        min(iteration.benchmark_games for iteration in benchmark_iterations)
        if benchmark_iterations
        else 0
    )
    if benchmark_iterations and min_benchmark_games < DEFAULT_MIN_BENCHMARK_GAMES:
        notes.append(
            "Observed benchmark game counts are below the default audit minimum; "
            "calibration keeps the observed minimum so the pilot remains reproducible."
        )

    return RunAuditCalibrationResult(
        manifest_path=result.manifest_path,
        schema_version=result.schema_version,
        source_type=result.source_type,
        iteration_count=len(result.iterations),
        benchmark_iteration_count=len(benchmark_iterations),
        margin=margin,
        require_benchmark=bool(benchmark_iterations),
        min_latest_benchmark_win_rate=_floor_observed_rate(
            _min_optional(iteration.benchmark_win_rate for iteration in benchmark_iterations),
            margin=margin,
        ),
        min_latest_benchmark_games=min_benchmark_games,
        max_latest_collection_capped_rate=_ceiling_observed_rate(
            _max_optional(iteration.collection_capped_rate for iteration in result.iterations),
            margin=margin,
            minimum=DEFAULT_MAX_COLLECTION_CAPPED_RATE,
        ),
        max_latest_benchmark_capped_rate=_ceiling_observed_rate(
            _max_optional(iteration.benchmark_capped_rate for iteration in benchmark_iterations),
            margin=margin,
            minimum=DEFAULT_MAX_BENCHMARK_CAPPED_RATE,
        ),
        max_latest_average_decision_rounds=_ceiling_observed_float(
            _max_optional(iteration.average_decision_rounds for iteration in result.iterations),
            margin=margin,
        ),
        max_latest_benchmark_average_decision_rounds=_ceiling_observed_float(
            _max_optional(iteration.benchmark_average_decision_rounds for iteration in benchmark_iterations),
            margin=margin,
        ),
        max_latest_process_peak_rss_mb=_ceiling_observed_float(
            _max_optional(_latest_process_peak_rss_mb(iteration) for iteration in result.iterations),
            margin=margin,
        ),
        max_benchmark_win_rate_drop=(
            (
                DEFAULT_MAX_BENCHMARK_WIN_RATE_DROP
                if observed_drop is None
                else _ceiling_observed_rate(
                    observed_drop,
                    margin=margin,
                    minimum=DEFAULT_MAX_BENCHMARK_WIN_RATE_DROP,
                )
            )
            if benchmark_iterations
            else None
        ),
        max_consecutive_promotion_failures=max(
            DEFAULT_MAX_CONSECUTIVE_PROMOTION_FAILURES,
            _max_consecutive_promotion_failures(result.iterations),
        ),
        require_benchmark_opponent_coverage=bool(benchmark_iterations),
        notes=tuple(notes),
    )


def calibrate_run_audits(
    paths: Iterable[Path],
    *,
    margin: float = DEFAULT_AUDIT_CALIBRATION_MARGIN,
    aggregate_mode: str = "median",
) -> MultiRunAuditCalibrationResult:
    if aggregate_mode not in AUDIT_CALIBRATION_AGGREGATE_MODES:
        choices = ", ".join(AUDIT_CALIBRATION_AGGREGATE_MODES)
        raise ValueError(f"unknown aggregate mode {aggregate_mode!r}; choose one of: {choices}")
    selected_paths = tuple(paths)
    if not selected_paths:
        raise ValueError("at least one run path is required.")
    calibrations = tuple(calibrate_run_audit(path, margin=margin) for path in selected_paths)
    source_types = tuple(dict.fromkeys(calibration.source_type for calibration in calibrations))
    notes = [
        (
            "Aggregated from multiple audit calibrations; thresholds are chosen "
            "to keep every supplied pilot run passable with the requested margin."
        )
    ]
    if not all(calibration.require_benchmark for calibration in calibrations):
        notes.append("At least one supplied run has no benchmark iterations; aggregate allows missing benchmarks.")
    if not all(calibration.require_benchmark_opponent_coverage for calibration in calibrations):
        notes.append(
            "At least one supplied run has no benchmark opponent coverage; "
            "aggregate allows missing benchmark opponents."
        )
    return MultiRunAuditCalibrationResult(
        paths=tuple(calibration.manifest_path for calibration in calibrations),
        source_type=source_types[0] if len(source_types) == 1 else "mixed",
        aggregate_mode=aggregate_mode,
        run_count=len(calibrations),
        iteration_count=sum(calibration.iteration_count for calibration in calibrations),
        benchmark_iteration_count=sum(calibration.benchmark_iteration_count for calibration in calibrations),
        margin=margin,
        calibrations=calibrations,
        require_benchmark=all(calibration.require_benchmark for calibration in calibrations),
        min_latest_benchmark_win_rate=_aggregate_floor(
            (calibration.min_latest_benchmark_win_rate for calibration in calibrations),
            aggregate_mode=aggregate_mode,
        ),
        min_latest_benchmark_games=_aggregate_minimum_count(
            (calibration.min_latest_benchmark_games for calibration in calibrations),
            aggregate_mode=aggregate_mode,
        ),
        max_latest_collection_capped_rate=_aggregate_ceiling(
            (calibration.max_latest_collection_capped_rate for calibration in calibrations),
            aggregate_mode=aggregate_mode,
        ),
        max_latest_benchmark_capped_rate=_aggregate_ceiling(
            (calibration.max_latest_benchmark_capped_rate for calibration in calibrations),
            aggregate_mode=aggregate_mode,
        ),
        max_latest_average_decision_rounds=_aggregate_ceiling(
            (calibration.max_latest_average_decision_rounds for calibration in calibrations),
            aggregate_mode=aggregate_mode,
        ),
        max_latest_benchmark_average_decision_rounds=_aggregate_ceiling(
            (calibration.max_latest_benchmark_average_decision_rounds for calibration in calibrations),
            aggregate_mode=aggregate_mode,
        ),
        max_latest_process_peak_rss_mb=_aggregate_ceiling(
            (calibration.max_latest_process_peak_rss_mb for calibration in calibrations),
            aggregate_mode=aggregate_mode,
        ),
        max_benchmark_win_rate_drop=_aggregate_ceiling(
            (calibration.max_benchmark_win_rate_drop for calibration in calibrations),
            aggregate_mode=aggregate_mode,
        ),
        max_consecutive_promotion_failures=_aggregate_maximum_count(
            (calibration.max_consecutive_promotion_failures for calibration in calibrations),
            aggregate_mode=aggregate_mode,
        ),
        require_benchmark_opponent_coverage=all(
            calibration.require_benchmark_opponent_coverage for calibration in calibrations
        ),
        notes=tuple(notes),
    )


def _comparison_entry(
    path: Path,
    *,
    audit_config: RunAuditConfig | None = None,
    audit_profile: str | None = None,
) -> RunComparisonEntry:
    strict_audit = audit_run(path, config=audit_config) if audit_config is not None else None
    audit = strict_audit if strict_audit is not None else audit_run(path, config=_permissive_audit_config())
    latest = audit.iterations[-1]
    process_peak_rss_mb = _latest_process_peak_rss_mb(latest)
    return RunComparisonEntry(
        label=_comparison_label(audit.manifest_path),
        manifest_path=audit.manifest_path,
        source_type=audit.source_type,
        source_metadata=audit.source_metadata,
        iteration_count=len(audit.iterations),
        latest_iteration=audit.latest_iteration,
        latest_policy_id=latest.policy_id,
        latest_checkpoint_path=latest.checkpoint_path,
        latest_benchmark_win_rate=audit.latest_benchmark_win_rate,
        best_benchmark_win_rate=audit.best_benchmark_win_rate,
        best_benchmark_games=_best_benchmark_games(audit.iterations),
        latest_benchmark_games=latest.benchmark_games,
        latest_collection_games_per_hour=latest.collection_games_per_hour,
        latest_benchmark_games_per_hour=latest.benchmark_games_per_hour,
        latest_collection_capped_rate=audit.latest_collection_capped_rate,
        latest_benchmark_capped_rate=audit.latest_benchmark_capped_rate,
        latest_collection_peak_rss_mb=latest.collection_peak_rss_mb,
        latest_benchmark_peak_rss_mb=latest.benchmark_peak_rss_mb,
        latest_process_peak_rss_mb=process_peak_rss_mb,
        latest_average_decision_rounds=audit.latest_average_decision_rounds,
        latest_benchmark_average_decision_rounds=audit.latest_benchmark_average_decision_rounds,
        latest_promotion_recorded=latest.promotion_recorded,
        latest_advancement_recorded=latest.advancement_recorded,
        latest_advancement_reason=latest.advancement_reason,
        audit_profile=audit_profile,
        audit_passed=strict_audit.passed if strict_audit is not None else None,
        audit_failed_checks=tuple(check.name for check in strict_audit.blocking_failed_checks)
        if strict_audit is not None
        else (),
        audit_warning_checks=tuple(check.name for check in strict_audit.warning_failed_checks)
        if strict_audit is not None
        else (),
    )


def _comparison_label(manifest_path: Path) -> str:
    parent = manifest_path.parent
    if parent.name:
        return parent.name
    return manifest_path.name


def _comparison_error_label(path: Path) -> str:
    candidate = path.expanduser()
    if candidate.name == "manifest.json" and candidate.parent.name:
        return candidate.parent.name
    return candidate.name or str(candidate)


def _best_benchmark_games(iterations: tuple[RunAuditIterationSummary, ...]) -> int:
    benchmark_iterations = tuple(
        iteration
        for iteration in iterations
        if iteration.benchmark_win_rate is not None and iteration.benchmark_games > 0
    )
    if not benchmark_iterations:
        return 0
    best = max(
        benchmark_iterations,
        key=lambda iteration: (float(iteration.benchmark_win_rate or 0.0), iteration.benchmark_games),
    )
    return best.benchmark_games


def _best_entry_by_optional_value(
    entries: tuple[RunComparisonEntry, ...],
    field_name: str,
) -> RunComparisonEntry | None:
    present = tuple(entry for entry in entries if getattr(entry, field_name) is not None)
    if not present:
        return None
    return max(present, key=lambda entry: float(getattr(entry, field_name)))


def _source_type(schema_version: str) -> str:
    if schema_version == SELFPLAY_RUN_SCHEMA_VERSION:
        return "linear_selfplay"
    if schema_version == NEURAL_SELFPLAY_RUN_SCHEMA_VERSION:
        return "neural_selfplay"
    raise ValueError(f"Unsupported run manifest schema: {schema_version!r}.")


def _manifest_source_metadata(manifest: Mapping[str, Any]) -> Mapping[str, Any]:
    source = manifest.get("source")
    return dict(source) if isinstance(source, Mapping) else {}


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
        collection_games_per_hour=_games_per_hour(collection_metrics),
        collection_capped_rate=_capped_rate(collection_metrics),
        collection_peak_rss_mb=_optional_float(collection_metrics.get("peak_rss_mb")),
        collection_peak_rss_mb_by_phase=_collection_peak_rss_by_phase(collection_metrics),
        average_decision_rounds=_optional_float(collection_metrics.get("average_decision_rounds")),
        benchmark_win_rate=benchmark_summary.win_rate if benchmark_summary.games else None,
        benchmark_games=benchmark_summary.games,
        benchmark_games_per_hour=_games_per_hour(benchmark),
        benchmark_capped_rate=benchmark_summary.capped_rate if benchmark_summary.games else None,
        benchmark_peak_rss_mb=(
            None
            if benchmark is None
            else _optional_float(benchmark.get("peak_rss_mb"))
        ),
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
        process_peak_rss_mb_by_phase=_process_peak_rss_by_phase(iteration),
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


def _promoted_opponent_pool_requirement_check(manifest: Mapping[str, Any]) -> RunAuditCheck:
    requirements: list[tuple[int, int]] = []
    failures: list[str] = []
    iterations = tuple(_mapping(iteration) for iteration in _sequence(manifest.get("iterations", ())))
    iterations_by_number = {int(iteration.get("iteration", 0)): iteration for iteration in iterations}
    for invocation_index, invocation_config in enumerate(_manifest_invocation_configs(manifest), start=1):
        opponent_pool = _optional_mapping(invocation_config.get("opponent_pool"))
        if opponent_pool is None:
            continue
        raw_required_size = opponent_pool.get("required_promoted_opponent_pool_size")
        if raw_required_size is None:
            continue
        required_size = _optional_int(raw_required_size)
        if required_size is None:
            failures.append(f"invocation_{invocation_index}:invalid_required_promoted_opponent_pool_size")
            continue
        if required_size < 0:
            failures.append(f"invocation_{invocation_index}:invalid_required_promoted_opponent_pool_size")
            continue
        if required_size == 0:
            continue
        pool_registry_path = _optional_str(opponent_pool.get("promotion_pool_registry_path"))
        if not pool_registry_path:
            failures.append(f"invocation_{invocation_index}:missing_promotion_pool_registry_path")
        raw_max_historical_opponents = opponent_pool.get("max_historical_opponents")
        max_historical_opponents = _optional_int(raw_max_historical_opponents)
        if max_historical_opponents is None:
            failures.append(f"invocation_{invocation_index}:missing_or_invalid_max_historical_opponents")
        elif required_size > max_historical_opponents:
            failures.append(
                f"invocation_{invocation_index}:required={required_size},max_historical={max_historical_opponents}"
            )
        selection_mode = str(opponent_pool.get("historical_opponent_selection") or "recent")
        if selection_mode not in HISTORICAL_OPPONENT_SELECTION_MODES:
            failures.append(f"invocation_{invocation_index}:invalid_historical_opponent_selection")
            selection_mode = "recent"
        promoted_specs = _string_sequence(opponent_pool.get("promoted_checkpoint_policy_specs", ()))
        fixed_specs = set(_string_sequence(opponent_pool.get("fixed_opponent_policy_specs", ())))
        fixed_spec_bodies = {_policy_spec_body(spec) for spec in fixed_specs}
        first_iteration = _optional_int(invocation_config.get("first_iteration"))
        first_iteration_payload = iterations_by_number.get(first_iteration or 0)
        current_policy_spec = (
            _optional_str(first_iteration_payload.get("current_policy_spec"))
            if first_iteration_payload is not None
            else _optional_str(invocation_config.get("initial_policy_spec"))
        )
        if max_historical_opponents is not None:
            launch_selectable_specs = _historical_specs(
                promoted_specs,
                current_policy_spec=current_policy_spec,
                max_historical_opponents=max_historical_opponents,
                selection_mode=selection_mode,
            )
            requirements.append((len(launch_selectable_specs), required_size))
            if len(launch_selectable_specs) < required_size:
                failures.append(
                    f"invocation_{invocation_index}:launch_selectable={len(launch_selectable_specs)},required={required_size}"
                )
        covered_iterations = _covered_invocation_iterations(
            invocation_config,
            iterations=iterations,
        )
        if not covered_iterations:
            failures.append(f"invocation_{invocation_index}:no_covered_iterations")
            continue
        for iteration in covered_iterations:
            iteration_number = int(iteration.get("iteration", 0))
            opponent_specs = _string_sequence(iteration.get("opponent_policy_specs", ()))
            if not opponent_specs:
                failures.append(f"invocation_{invocation_index}:iteration_{iteration_number}:missing_opponent_policy_specs")
                requirements.append((0, required_size))
                continue
            selected_promoted_specs = tuple(
                spec for spec in opponent_specs if _policy_spec_body(spec) not in fixed_spec_bodies
            )
            requirements.append((len(selected_promoted_specs), required_size))
            if len(selected_promoted_specs) < required_size:
                failures.append(
                    f"invocation_{invocation_index}:iteration_{iteration_number}:selected={len(selected_promoted_specs)},required={required_size}"
                )
    if not requirements and not failures:
        return RunAuditCheck(
            name="promoted_opponent_pool_requirement",
            passed=True,
            observed=None,
            threshold="not_recorded",
            message="no promoted opponent pool requirement was recorded",
        )
    observed = min((available / required for available, required in requirements), default=None)
    return RunAuditCheck(
        name="promoted_opponent_pool_requirement",
        passed=not failures,
        observed=observed,
        threshold=1.0,
        message=(
            "recorded promoted opponent pools satisfied their required sizes"
            if not failures
            else "recorded promoted opponent pool requirement was undersized: " + "; ".join(failures)
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


def _latest_process_peak_rss_checks(
    latest: RunAuditIterationSummary,
    config: RunAuditConfig,
) -> tuple[RunAuditCheck, ...]:
    if config.max_latest_process_peak_rss_mb is None:
        return ()
    observed = _latest_process_peak_rss_mb(latest)
    if observed is None:
        passed = True
        message = "latest process peak RSS is unavailable; RSS ceiling is skipped"
    elif observed <= config.max_latest_process_peak_rss_mb:
        passed = True
        message = "latest process peak RSS is within limit"
    else:
        passed = False
        message = "latest process peak RSS exceeds limit"
    return (
        RunAuditCheck(
            name="latest_process_peak_rss_mb",
            passed=passed,
            observed=observed,
            threshold=config.max_latest_process_peak_rss_mb,
            message=message,
        ),
    )


def _latest_process_peak_rss_mb(latest: RunAuditIterationSummary) -> float | None:
    return _max_optional(
        (
            latest.collection_peak_rss_mb,
            latest.benchmark_peak_rss_mb,
            *latest.process_peak_rss_mb_by_phase.values(),
        )
    )


def _collection_peak_rss_by_phase(collection_metrics: Mapping[str, Any]) -> dict[str, float | None]:
    payload = collection_metrics.get("peak_rss_mb_by_phase")
    if not isinstance(payload, Mapping):
        return {}
    return {
        str(phase): _optional_float(value)
        for phase, value in payload.items()
    }


def _process_peak_rss_by_phase(iteration: Mapping[str, Any]) -> dict[str, float | None]:
    payload = iteration.get("process_peak_rss_mb_by_phase")
    if not isinstance(payload, Mapping):
        return {}
    return {
        str(phase): _optional_float(value)
        for phase, value in payload.items()
    }


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
        passed=_threshold_lte(max_drop, config.max_benchmark_win_rate_drop),
        observed=max_drop,
        threshold=config.max_benchmark_win_rate_drop,
        message="latest same-opponent benchmark win rates have not regressed too far from previous best",
    )


def _benchmark_opponent_coverage_check(
    latest: RunAuditIterationSummary,
    missing_opponents: tuple[str, ...],
    config: RunAuditConfig,
) -> RunAuditCheck:
    if latest.benchmark_games <= 0:
        return RunAuditCheck(
            name="latest_benchmark_opponent_coverage",
            passed=not config.require_benchmark,
            observed=None,
            threshold="required" if config.require_benchmark else "optional",
            message="latest benchmark evidence is unavailable for opponent coverage",
        )
    if not config.require_benchmark_opponent_coverage:
        return RunAuditCheck(
            name="latest_benchmark_opponent_coverage",
            passed=True,
            observed="optional",
            threshold="required_baseline_opponents",
            message="latest fixed-baseline benchmark opponent coverage is optional for this audit",
        )
    if missing_opponents:
        return RunAuditCheck(
            name="latest_benchmark_opponent_coverage",
            passed=False,
            observed=", ".join(missing_opponents),
            threshold="required_baseline_opponents",
            message="latest benchmark is missing fixed baseline opponents that appeared in prior benchmark evidence",
        )
    return RunAuditCheck(
        name="latest_benchmark_opponent_coverage",
        passed=True,
        observed=None,
        threshold="required_baseline_opponents",
        message="latest benchmark includes required fixed baseline opponents",
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


def _missing_latest_benchmark_opponents(
    iterations: tuple[RunAuditIterationSummary, ...],
) -> tuple[str, ...]:
    if len(iterations) < 2:
        return ()
    if iterations[-1].benchmark_games <= 0:
        return ()
    prior_opponents = {
        opponent.opponent_policy_id
        for iteration in iterations[:-1]
        if iteration.benchmark_games > 0
        for opponent in iteration.benchmark_opponents
    }
    required_prior_opponents = set(DEFAULT_REQUIRED_BENCHMARK_OPPONENTS) & prior_opponents
    latest_opponents = {
        opponent.opponent_policy_id
        for opponent in iterations[-1].benchmark_opponents
    }
    return tuple(sorted(required_prior_opponents - latest_opponents))


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


def _policy_spec_body(value: str) -> str:
    return str(value).partition("?")[0]


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    return float(value)


def _apply_warning_check_severity(
    checks: tuple[RunAuditCheck, ...],
    *,
    warning_check_names: tuple[str, ...],
) -> tuple[RunAuditCheck, ...]:
    if not warning_check_names:
        return checks
    warning_names = set(warning_check_names)
    return tuple(
        RunAuditCheck(
            name=check.name,
            passed=check.passed,
            observed=check.observed,
            threshold=check.threshold,
            message=check.message,
            severity="warning" if check.name in warning_names else check.severity,
        )
        for check in checks
    )


def _required_bool(payload: Mapping[str, Any], field_name: str) -> bool:
    value = payload[field_name]
    if not isinstance(value, bool):
        raise ValueError(f"{field_name} must be a boolean.")
    return value


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _manifest_invocation_configs(manifest: Mapping[str, Any]) -> tuple[Mapping[str, Any], ...]:
    configs = manifest.get("invocation_configs")
    if configs is not None:
        return tuple(_mapping(config) for config in _sequence(configs))
    configs_by_fingerprint: dict[str, Mapping[str, Any]] = {}
    for iteration in tuple(_mapping(iteration) for iteration in _sequence(manifest.get("iterations", ()))):
        config = iteration.get("invocation_config")
        if config is None:
            continue
        mapped = _mapping(config)
        configs_by_fingerprint.setdefault(json.dumps(mapped, sort_keys=True), mapped)
    return tuple(configs_by_fingerprint.values())


def _covered_invocation_iterations(
    invocation_config: Mapping[str, Any],
    *,
    iterations: tuple[Mapping[str, Any], ...],
) -> tuple[Mapping[str, Any], ...]:
    first_iteration = _optional_int(invocation_config.get("first_iteration"))
    requested = _optional_int(invocation_config.get("iterations_requested"))
    if first_iteration is None or requested is None or requested <= 0:
        return ()
    last_iteration = first_iteration + requested - 1
    return tuple(
        iteration
        for iteration in iterations
        if first_iteration <= int(iteration.get("iteration", 0)) <= last_iteration
    )


def _historical_specs(
    specs: tuple[str, ...],
    *,
    current_policy_spec: str | None,
    max_historical_opponents: int,
    selection_mode: str = "recent",
) -> tuple[str, ...]:
    return historical_opponent_policy_specs(
        specs,
        current_policy_spec=current_policy_spec,
        max_historical_opponents=max(0, max_historical_opponents),
        selection_mode=selection_mode,
    )


def _string_sequence(value: Any) -> tuple[str, ...]:
    try:
        return tuple(str(item) for item in _sequence(value))
    except ValueError:
        return ()


def _games_per_hour(metrics: Mapping[str, Any] | None) -> float | None:
    if metrics is None:
        return None
    games_per_second = _optional_float(metrics.get("games_per_second"))
    if games_per_second is not None:
        return games_per_second * 3600.0
    raw_games = metrics.get("games", metrics.get("total_games"))
    raw_elapsed = metrics.get("elapsed_seconds")
    if raw_games is None or raw_elapsed is None:
        return None
    elapsed_seconds = float(raw_elapsed)
    if elapsed_seconds <= 0.0:
        return 0.0
    return (float(raw_games) / elapsed_seconds) * 3600.0


def _permissive_audit_config() -> RunAuditConfig:
    return RunAuditConfig(
        min_latest_benchmark_win_rate=0.0,
        min_latest_benchmark_games=0,
        max_latest_collection_capped_rate=1.0,
        max_latest_benchmark_capped_rate=1.0,
        max_benchmark_win_rate_drop=1.0,
        max_consecutive_promotion_failures=1_000_000,
        require_benchmark=False,
    )


def _suggested_audit_config(result: Any) -> dict[str, float | int | bool | None | list[str]]:
    require_benchmark = bool(result.require_benchmark)
    return {
        "min_latest_benchmark_win_rate": (
            result.min_latest_benchmark_win_rate
            if require_benchmark and result.min_latest_benchmark_win_rate is not None
            else 0.0
        ),
        "min_latest_benchmark_games": result.min_latest_benchmark_games if require_benchmark else 0,
        "max_latest_collection_capped_rate": (
            result.max_latest_collection_capped_rate
            if result.max_latest_collection_capped_rate is not None
            else 1.0
        ),
        "max_latest_benchmark_capped_rate": (
            result.max_latest_benchmark_capped_rate
            if require_benchmark and result.max_latest_benchmark_capped_rate is not None
            else 1.0
        ),
        "max_latest_average_decision_rounds": result.max_latest_average_decision_rounds,
        "max_latest_benchmark_average_decision_rounds": (
            result.max_latest_benchmark_average_decision_rounds if require_benchmark else None
        ),
        "max_latest_process_peak_rss_mb": result.max_latest_process_peak_rss_mb,
        "max_benchmark_win_rate_drop": (
            result.max_benchmark_win_rate_drop
            if require_benchmark and result.max_benchmark_win_rate_drop is not None
            else 1.0
        ),
        "max_consecutive_promotion_failures": result.max_consecutive_promotion_failures,
        "require_benchmark": require_benchmark,
        "require_latest_promotion": False,
        "require_benchmark_opponent_coverage": bool(result.require_benchmark_opponent_coverage),
        "warning_check_names": [],
    }


def _suggested_audit_cli_flags(result: Any) -> tuple[str, ...]:
    config = _suggested_audit_config(result)
    require_benchmark = bool(config["require_benchmark"])
    benchmark_only_fields = {
        "min_latest_benchmark_win_rate",
        "min_latest_benchmark_games",
        "max_latest_benchmark_capped_rate",
        "max_latest_benchmark_average_decision_rounds",
        "max_benchmark_win_rate_drop",
    }
    flags: list[str] = []
    for field_name, flag_name in (
        ("min_latest_benchmark_win_rate", "--min-latest-benchmark-win-rate"),
        ("min_latest_benchmark_games", "--min-latest-benchmark-games"),
        ("max_latest_collection_capped_rate", "--max-latest-collection-capped-rate"),
        ("max_latest_benchmark_capped_rate", "--max-latest-benchmark-capped-rate"),
        ("max_latest_average_decision_rounds", "--max-latest-average-decision-rounds"),
        ("max_latest_benchmark_average_decision_rounds", "--max-latest-benchmark-average-decision-rounds"),
        ("max_latest_process_peak_rss_mb", "--max-latest-process-peak-rss-mb"),
        ("max_benchmark_win_rate_drop", "--max-benchmark-win-rate-drop"),
        ("max_consecutive_promotion_failures", "--max-consecutive-promotion-failures"),
    ):
        if not require_benchmark and field_name in benchmark_only_fields:
            continue
        value = config[field_name]
        if value is not None:
            flags.extend((flag_name, str(value)))
    if not require_benchmark:
        flags.append("--allow-missing-benchmark")
    if not config["require_benchmark_opponent_coverage"]:
        flags.append("--allow-missing-benchmark-opponents")
    for check_name in config["warning_check_names"]:
        flags.extend(("--warning-check", str(check_name)))
    return tuple(flags)


def _suggested_post_iteration_audit_cli_flags(result: Any) -> tuple[str, ...]:
    config = _suggested_audit_config(result)
    nullable_fields = {
        "max_latest_average_decision_rounds",
        "max_latest_benchmark_average_decision_rounds",
        "max_latest_process_peak_rss_mb",
    }
    flags: list[str] = ["--audit-after-iteration"]
    for field_name, flag_name in (
        ("min_latest_benchmark_win_rate", "--audit-min-latest-benchmark-win-rate"),
        ("min_latest_benchmark_games", "--audit-min-latest-benchmark-games"),
        ("max_latest_collection_capped_rate", "--audit-max-latest-collection-capped-rate"),
        ("max_latest_benchmark_capped_rate", "--audit-max-latest-benchmark-capped-rate"),
        ("max_latest_average_decision_rounds", "--audit-max-latest-average-decision-rounds"),
        ("max_latest_benchmark_average_decision_rounds", "--audit-max-latest-benchmark-average-decision-rounds"),
        ("max_latest_process_peak_rss_mb", "--audit-max-latest-process-peak-rss-mb"),
        ("max_benchmark_win_rate_drop", "--audit-max-benchmark-win-rate-drop"),
        ("max_consecutive_promotion_failures", "--audit-max-consecutive-promotion-failures"),
    ):
        value = config[field_name]
        if value is None and field_name not in nullable_fields:
            continue
        flags.extend((flag_name, "none" if value is None else str(value)))
    flags.append("--audit-require-benchmark" if config["require_benchmark"] else "--audit-allow-missing-benchmark")
    flags.append(
        "--audit-require-benchmark-opponents"
        if config["require_benchmark_opponent_coverage"]
        else "--audit-allow-missing-benchmark-opponents"
    )
    flags.append(
        "--audit-require-latest-promotion"
        if config["require_latest_promotion"]
        else "--audit-allow-missing-latest-promotion"
    )
    for check_name in config["warning_check_names"]:
        flags.extend(("--audit-warning-check", str(check_name)))
    return tuple(flags)


def _aggregate_floor(values: Any, *, aggregate_mode: str) -> float | None:
    return _min_optional(values) if aggregate_mode == "envelope" else _median_optional(values)


def _aggregate_ceiling(values: Any, *, aggregate_mode: str) -> float | None:
    return _max_optional(values) if aggregate_mode == "envelope" else _median_optional(values)


def _aggregate_minimum_count(values: Any, *, aggregate_mode: str) -> int:
    present = tuple(sorted(int(value) for value in values))
    if aggregate_mode == "envelope":
        return min(present)
    return math.floor(_median(present))


def _aggregate_maximum_count(values: Any, *, aggregate_mode: str) -> int:
    present = tuple(sorted(int(value) for value in values))
    if aggregate_mode == "envelope":
        return max(present)
    return math.ceil(_median(present))


def _min_optional(values: Any) -> float | None:
    present = tuple(float(value) for value in values if value is not None)
    return min(present) if present else None


def _max_optional(values: Any) -> float | None:
    present = tuple(float(value) for value in values if value is not None)
    return max(present) if present else None


def _median_optional(values: Any) -> float | None:
    present = tuple(sorted(float(value) for value in values if value is not None))
    return _round_threshold(_median(present)) if present else None


def _median(values: tuple[float, ...] | tuple[int, ...]) -> float:
    if not values:
        raise ValueError("cannot calculate median of empty values.")
    middle = len(values) // 2
    if len(values) % 2:
        return float(values[middle])
    return (float(values[middle - 1]) + float(values[middle])) / 2.0


def _floor_observed_rate(value: float | None, *, margin: float) -> float | None:
    if value is None:
        return None
    return _round_threshold(max(0.0, value * (1.0 - margin)))


def _ceiling_observed_rate(
    value: float | None,
    *,
    margin: float,
    minimum: float = 0.0,
) -> float | None:
    if value is None:
        return None
    return _round_threshold(min(1.0, max(minimum, value * (1.0 + margin))))


def _ceiling_observed_float(value: float | None, *, margin: float) -> float | None:
    if value is None:
        return None
    return _round_threshold(value * (1.0 + margin))


def _round_threshold(value: float) -> float:
    return round(value, 6)


def _threshold_lte(observed: float, threshold: float) -> bool:
    return observed <= threshold + _THRESHOLD_EPSILON


def _max_observed_same_opponent_drop(iterations: tuple[RunAuditIterationSummary, ...]) -> float | None:
    previous_best: dict[str, float] = {}
    drops: list[float] = []
    for iteration in iterations:
        if iteration.benchmark_games <= 0:
            continue
        for opponent in iteration.benchmark_opponents:
            best_previous = previous_best.get(opponent.opponent_policy_id)
            if best_previous is not None:
                drops.append(max(0.0, best_previous - opponent.win_rate))
                previous_best[opponent.opponent_policy_id] = max(best_previous, opponent.win_rate)
            else:
                previous_best[opponent.opponent_policy_id] = opponent.win_rate
    return max(drops) if drops else None


def _max_consecutive_promotion_failures(iterations: tuple[RunAuditIterationSummary, ...]) -> int:
    longest = 0
    current = 0
    for iteration in iterations:
        if iteration.promotion_recorded is False:
            current += 1
            longest = max(longest, current)
            continue
        current = 0
    return longest
