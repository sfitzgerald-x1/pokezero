"""Promotion gate helpers for experiment manifests."""

from __future__ import annotations

from collections.abc import Iterable as IterableABC
from dataclasses import dataclass
from dataclasses import field
import json
import math
from pathlib import Path
from typing import Any, Mapping

from .bootstrap import TEACHER_BOOTSTRAP_SCHEMA_VERSION
from .selfplay import SELFPLAY_RUN_SCHEMA_VERSION


NEURAL_SELFPLAY_RUN_SCHEMA_VERSION = "pokezero.neural_selfplay_run.v1"
DEFAULT_MIN_BENCHMARK_WIN_RATE = 0.55
DEFAULT_MIN_INCUMBENT_WIN_RATE = 0.55
DEFAULT_MIN_BENCHMARK_GAMES = 50
DEFAULT_MIN_INCUMBENT_GAMES = 200
DEFAULT_MAX_COLLECTION_CAPPED_RATE = 0.10
DEFAULT_MAX_BENCHMARK_CAPPED_RATE = 0.10
DEFAULT_MAX_INCUMBENT_CAPPED_RATE = 0.10
DEFAULT_MAX_TEACHER_DEGRADATION_RATE = 0.0
DEFAULT_INCUMBENT_CONFIDENCE_Z = 1.645
DEFAULT_MIN_INCUMBENT_WIN_RATE_LOWER_BOUND = 0.50


@dataclass(frozen=True)
class PromotionGateConfig:
    min_benchmark_win_rate: float = DEFAULT_MIN_BENCHMARK_WIN_RATE
    min_incumbent_win_rate: float = DEFAULT_MIN_INCUMBENT_WIN_RATE
    min_benchmark_games: int = DEFAULT_MIN_BENCHMARK_GAMES
    min_incumbent_games: int = DEFAULT_MIN_INCUMBENT_GAMES
    max_collection_capped_rate: float = DEFAULT_MAX_COLLECTION_CAPPED_RATE
    max_benchmark_capped_rate: float = DEFAULT_MAX_BENCHMARK_CAPPED_RATE
    max_incumbent_capped_rate: float = DEFAULT_MAX_INCUMBENT_CAPPED_RATE
    max_teacher_degradation_rate: float = DEFAULT_MAX_TEACHER_DEGRADATION_RATE
    min_incumbent_win_rate_lower_bound: float = DEFAULT_MIN_INCUMBENT_WIN_RATE_LOWER_BOUND
    incumbent_confidence_z: float = DEFAULT_INCUMBENT_CONFIDENCE_Z
    require_benchmark: bool = True
    required_benchmark_opponents: tuple[str, ...] = ()
    opponent_min_win_rates: Mapping[str, float] = field(default_factory=dict)
    incumbent_policy_id: str | None = None

    def __post_init__(self) -> None:
        if self.min_benchmark_games < 0:
            raise ValueError("min_benchmark_games must be non-negative.")
        if self.min_incumbent_games < 0:
            raise ValueError("min_incumbent_games must be non-negative.")
        for field_name in (
            "min_benchmark_win_rate",
            "min_incumbent_win_rate",
            "min_incumbent_win_rate_lower_bound",
            "max_collection_capped_rate",
            "max_benchmark_capped_rate",
            "max_incumbent_capped_rate",
            "max_teacher_degradation_rate",
        ):
            value = float(getattr(self, field_name))
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{field_name} must be between 0 and 1.")
        if self.incumbent_confidence_z < 0.0:
            raise ValueError("incumbent_confidence_z must be non-negative.")
        for opponent_id, threshold in self.opponent_min_win_rates.items():
            if not str(opponent_id):
                raise ValueError("opponent_min_win_rates keys must be non-empty.")
            if not 0.0 <= float(threshold) <= 1.0:
                raise ValueError("opponent-specific win-rate thresholds must be between 0 and 1.")
        if self.incumbent_policy_id is not None and not self.incumbent_policy_id.strip():
            raise ValueError("incumbent_policy_id must be non-empty when set.")


@dataclass(frozen=True)
class PromotionGateCheck:
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
class PromotionGateResult:
    gate_mode: str
    source_type: str
    manifest_path: Path
    candidate_policy_id: str | None
    checkpoint_path: str | None
    source_iteration: int | None
    collection_capped_rate: float | None
    benchmark_win_rate: float | None
    benchmark_capped_rate: float | None
    benchmark_games: int
    benchmark_opponents: tuple["PromotionBenchmarkOpponentResult", ...]
    incumbent_policy_id: str | None
    incumbent_win_rate: float | None
    incumbent_win_rate_lower_bound: float | None
    incumbent_games: int
    incumbent_capped_rate: float | None
    teacher_degradation_rate: float | None
    checks: tuple[PromotionGateCheck, ...]

    @property
    def passed(self) -> bool:
        return all(check.passed for check in self.checks)

    def to_dict(self) -> dict[str, Any]:
        return {
            "gate_mode": self.gate_mode,
            "source_type": self.source_type,
            "manifest_path": str(self.manifest_path),
            "candidate_policy_id": self.candidate_policy_id,
            "checkpoint_path": self.checkpoint_path,
            "source_iteration": self.source_iteration,
            "collection_capped_rate": self.collection_capped_rate,
            "benchmark_win_rate": self.benchmark_win_rate,
            "benchmark_capped_rate": self.benchmark_capped_rate,
            "benchmark_games": self.benchmark_games,
            "benchmark_opponents": [opponent.to_dict() for opponent in self.benchmark_opponents],
            "incumbent_policy_id": self.incumbent_policy_id,
            "incumbent_win_rate": self.incumbent_win_rate,
            "incumbent_win_rate_lower_bound": self.incumbent_win_rate_lower_bound,
            "incumbent_games": self.incumbent_games,
            "incumbent_capped_rate": self.incumbent_capped_rate,
            "teacher_degradation_rate": self.teacher_degradation_rate,
            "passed": self.passed,
            "checks": [check.to_dict() for check in self.checks],
        }


@dataclass(frozen=True)
class PromotionBenchmarkOpponentResult:
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


def evaluate_promotion_gate(
    path: Path,
    *,
    config: PromotionGateConfig = PromotionGateConfig(),
) -> PromotionGateResult:
    manifest_path = _manifest_path(path)
    manifest = _load_manifest(manifest_path)
    source_type = str(manifest.get("schema_version") or "")
    if source_type == SELFPLAY_RUN_SCHEMA_VERSION:
        candidate = _candidate_from_selfplay_manifest(manifest, manifest_path=manifest_path)
    elif source_type == TEACHER_BOOTSTRAP_SCHEMA_VERSION:
        candidate = _candidate_from_bootstrap_manifest(manifest)
    elif source_type == NEURAL_SELFPLAY_RUN_SCHEMA_VERSION:
        candidate = _candidate_from_neural_selfplay_manifest(manifest, manifest_path=manifest_path)
    else:
        raise ValueError(f"Unsupported experiment manifest schema: {source_type!r}.")

    all_benchmark = _benchmark_summary(candidate.benchmark, candidate.policy_id)
    incumbent_policy_id = (
        config.incumbent_policy_id.strip()
        if config.incumbent_policy_id is not None
        else candidate.derived_incumbent_policy_id
    )
    incumbent_opponent = _benchmark_opponent_for_policy_id(all_benchmark.opponents, incumbent_policy_id)
    incumbent_win_rate_lower_bound = (
        _wilson_lower_bound(incumbent_opponent.wins, incumbent_opponent.games, z=config.incumbent_confidence_z)
        if incumbent_opponent is not None
        else None
    )
    benchmark = _benchmark_without_opponent(all_benchmark, incumbent_policy_id)
    excluded_gate_opponents = (
        ()
        if config.required_benchmark_opponents
        else candidate.benchmark_reference_policy_ids
    )
    benchmark_for_gate = _benchmark_without_opponents(benchmark, excluded_gate_opponents)
    checks: list[PromotionGateCheck] = [
        _threshold_check(
            name="collection_capped_rate",
            observed=candidate.collection_capped_rate,
            threshold=config.max_collection_capped_rate,
            passed=(
                candidate.collection_capped_rate is not None
                and candidate.collection_capped_rate <= config.max_collection_capped_rate
            ),
            message=(
                "collection capped-game rate is within limit"
                if candidate.collection_capped_rate is not None
                else "collection capped-game rate is unavailable"
            ),
        )
    ]
    if benchmark.games or config.required_benchmark_opponents:
        gated_opponents = _gated_benchmark_opponents(
            benchmark.opponents,
            required_opponents=config.required_benchmark_opponents,
            excluded_opponents=excluded_gate_opponents,
        )
        for missing_opponent in _missing_benchmark_opponents(
            benchmark.opponents,
            required_opponents=config.required_benchmark_opponents,
        ):
            checks.append(
                PromotionGateCheck(
                    name=f"benchmark_opponent:{missing_opponent}",
                    passed=False,
                    observed=None,
                    threshold="required",
                    message=f"required benchmark opponent is missing: {missing_opponent}",
                )
            )
        for opponent in gated_opponents:
            checks.append(
                PromotionGateCheck(
                    name=f"benchmark_games:{opponent.opponent_policy_id}",
                    passed=opponent.games >= config.min_benchmark_games,
                    observed=opponent.games,
                    threshold=config.min_benchmark_games,
                    message="benchmark opponent has enough games",
                )
            )
            threshold = float(
                config.opponent_min_win_rates.get(
                    opponent.opponent_policy_id,
                    config.min_benchmark_win_rate,
                )
            )
            checks.append(
                _threshold_check(
                    name=f"benchmark_win_rate:{opponent.opponent_policy_id}",
                    observed=opponent.win_rate,
                    threshold=threshold,
                    passed=opponent.win_rate >= threshold,
                    message="benchmark win rate meets opponent-specific promotion floor",
                )
            )
        if benchmark_for_gate.games:
            checks.extend(
                (
                    _threshold_check(
                        name="benchmark_capped_rate",
                        observed=benchmark_for_gate.capped_rate,
                        threshold=config.max_benchmark_capped_rate,
                        passed=benchmark_for_gate.capped_rate <= config.max_benchmark_capped_rate,
                        message="benchmark capped-game rate is within limit",
                    ),
                )
            )
    elif config.require_benchmark and not all_benchmark.games:
        checks.append(
            PromotionGateCheck(
                name="benchmark_available",
                passed=False,
                observed=None,
                threshold="required",
                message="benchmark evidence is required for promotion",
            )
        )

    if incumbent_policy_id is not None:
        if incumbent_opponent is None:
            checks.append(
                PromotionGateCheck(
                    name=f"incumbent_benchmark_opponent:{incumbent_policy_id}",
                    passed=False,
                    observed=None,
                    threshold="required",
                    message=f"incumbent benchmark opponent is missing: {incumbent_policy_id}",
                )
            )
        else:
            checks.append(
                PromotionGateCheck(
                    name=f"incumbent_benchmark_games:{incumbent_policy_id}",
                    passed=incumbent_opponent.games >= config.min_incumbent_games,
                    observed=incumbent_opponent.games,
                    threshold=config.min_incumbent_games,
                    message="incumbent benchmark has enough games",
                )
            )
            checks.append(
                _threshold_check(
                    name=f"incumbent_win_rate:{incumbent_policy_id}",
                    observed=incumbent_opponent.win_rate,
                    threshold=config.min_incumbent_win_rate,
                    passed=incumbent_opponent.win_rate >= config.min_incumbent_win_rate,
                    message="candidate win rate clears incumbent promotion floor",
                )
            )
            checks.append(
                _threshold_check(
                    name=f"incumbent_win_rate_lower_bound:{incumbent_policy_id}",
                    observed=incumbent_win_rate_lower_bound,
                    threshold=config.min_incumbent_win_rate_lower_bound,
                    passed=(
                        incumbent_win_rate_lower_bound is not None
                        and incumbent_win_rate_lower_bound >= config.min_incumbent_win_rate_lower_bound
                    ),
                    message="candidate incumbent win-rate lower bound clears no-regression floor",
                )
            )
            checks.append(
                _threshold_check(
                    name=f"incumbent_capped_rate:{incumbent_policy_id}",
                    observed=incumbent_opponent.capped_rate,
                    threshold=config.max_incumbent_capped_rate,
                    passed=incumbent_opponent.capped_rate <= config.max_incumbent_capped_rate,
                    message="incumbent benchmark capped-game rate is within limit",
                )
            )

    if candidate.teacher_degradation_rate is not None:
        checks.append(
            _threshold_check(
                name="teacher_degradation_rate",
                observed=candidate.teacher_degradation_rate,
                threshold=config.max_teacher_degradation_rate,
                passed=candidate.teacher_degradation_rate <= config.max_teacher_degradation_rate,
                message="teacher degradation rate is within limit",
            )
        )

    return PromotionGateResult(
        gate_mode="absolute_floor+incumbent_delta" if incumbent_policy_id is not None else "absolute_floor",
        source_type=source_type,
        manifest_path=manifest_path,
        candidate_policy_id=candidate.policy_id,
        checkpoint_path=candidate.checkpoint_path,
        source_iteration=candidate.iteration,
        collection_capped_rate=candidate.collection_capped_rate,
        benchmark_win_rate=benchmark.win_rate if benchmark.games else None,
        benchmark_capped_rate=benchmark.capped_rate if benchmark.games else None,
        benchmark_games=benchmark.games,
        benchmark_opponents=tuple(
            PromotionBenchmarkOpponentResult(
                opponent_policy_id=opponent.opponent_policy_id,
                wins=opponent.wins,
                games=opponent.games,
                capped_games=opponent.capped_games,
            )
            for opponent in benchmark.opponents
        ),
        incumbent_policy_id=incumbent_policy_id,
        incumbent_win_rate=incumbent_opponent.win_rate if incumbent_opponent is not None else None,
        incumbent_win_rate_lower_bound=incumbent_win_rate_lower_bound,
        incumbent_games=incumbent_opponent.games if incumbent_opponent is not None else 0,
        incumbent_capped_rate=incumbent_opponent.capped_rate if incumbent_opponent is not None else None,
        teacher_degradation_rate=candidate.teacher_degradation_rate,
        checks=tuple(checks),
    )


@dataclass(frozen=True)
class _CandidateManifest:
    policy_id: str | None
    checkpoint_path: str | None
    iteration: int | None
    collection_capped_rate: float | None
    benchmark: Mapping[str, Any] | None
    teacher_degradation_rate: float | None
    derived_incumbent_policy_id: str | None = None
    benchmark_reference_policy_ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class _BenchmarkSummary:
    wins: int
    games: int
    capped_games: int
    opponents: tuple["_BenchmarkOpponentSummary", ...] = ()

    @property
    def win_rate(self) -> float:
        return self.wins / self.games if self.games else 0.0

    @property
    def capped_rate(self) -> float:
        return self.capped_games / self.games if self.games else 0.0


@dataclass(frozen=True)
class _BenchmarkOpponentSummary:
    opponent_policy_id: str
    wins: int = 0
    games: int = 0
    capped_games: int = 0

    @property
    def win_rate(self) -> float:
        return self.wins / self.games if self.games else 0.0

    @property
    def capped_rate(self) -> float:
        return self.capped_games / self.games if self.games else 0.0


def _candidate_from_selfplay_manifest(
    manifest: Mapping[str, Any],
    *,
    manifest_path: Path,
) -> _CandidateManifest:
    iterations = tuple(_mapping(iteration) for iteration in _sequence(manifest.get("iterations", ())))
    if not iterations:
        raise ValueError("self-play run manifest contains no iterations.")
    latest = iterations[-1]
    training = _mapping(latest.get("training", {}))
    model = _mapping(training.get("model", {}))
    return _CandidateManifest(
        policy_id=_optional_str(model.get("policy_id")),
        checkpoint_path=_optional_str(latest.get("checkpoint_path")),
        iteration=int(latest.get("iteration", 0)),
        collection_capped_rate=_capped_rate(_mapping(latest.get("collection_metrics", {}))),
        benchmark=_optional_mapping(latest.get("benchmark")),
        teacher_degradation_rate=None,
        derived_incumbent_policy_id=_derived_selfplay_incumbent_policy_id(iterations, manifest_path=manifest_path),
        benchmark_reference_policy_ids=_benchmark_reference_policy_ids(latest, manifest_path=manifest_path),
    )


def _candidate_from_bootstrap_manifest(manifest: Mapping[str, Any]) -> _CandidateManifest:
    training = _mapping(manifest.get("training", {}))
    model = _mapping(training.get("model", {}))
    return _CandidateManifest(
        policy_id=_optional_str(model.get("policy_id")),
        checkpoint_path=_optional_str(manifest.get("checkpoint_path")),
        iteration=None,
        collection_capped_rate=_capped_rate(_mapping(manifest.get("train_collection_metrics", {}))),
        benchmark=_optional_mapping(manifest.get("benchmark")),
        teacher_degradation_rate=_teacher_degradation_rate(manifest.get("teacher_decision_summary")),
    )


def _candidate_from_neural_selfplay_manifest(
    manifest: Mapping[str, Any],
    *,
    manifest_path: Path,
) -> _CandidateManifest:
    iterations = tuple(_mapping(iteration) for iteration in _sequence(manifest.get("iterations", ())))
    if not iterations:
        raise ValueError("neural self-play run manifest contains no iterations.")
    latest = iterations[-1]
    training = _mapping(latest.get("training", {}))
    model_config = _mapping(training.get("model_config", {}))
    return _CandidateManifest(
        policy_id=_optional_str(model_config.get("policy_id")),
        checkpoint_path=_optional_str(latest.get("checkpoint_path")),
        iteration=int(latest.get("iteration", 0)),
        collection_capped_rate=_capped_rate(_mapping(latest.get("collection_metrics", {}))),
        benchmark=_optional_mapping(latest.get("benchmark")),
        teacher_degradation_rate=None,
        derived_incumbent_policy_id=_derived_neural_selfplay_incumbent_policy_id(
            iterations,
            manifest_path=manifest_path,
        ),
    )


def _derived_selfplay_incumbent_policy_id(
    iterations: tuple[Mapping[str, Any], ...],
    *,
    manifest_path: Path,
) -> str | None:
    if len(iterations) >= 2:
        previous_training = _mapping(iterations[-2].get("training", {}))
        previous_model = _mapping(previous_training.get("model", {}))
        previous_policy_id = _optional_str(previous_model.get("policy_id"))
        if previous_policy_id:
            return previous_policy_id
    latest = iterations[-1]
    return _policy_id_from_policy_spec(latest.get("current_policy_spec"), manifest_path=manifest_path)


def _derived_neural_selfplay_incumbent_policy_id(
    iterations: tuple[Mapping[str, Any], ...],
    *,
    manifest_path: Path,
) -> str | None:
    latest = iterations[-1]
    current_policy_id = _neural_current_policy_id(
        iterations,
        latest.get("current_policy_spec"),
        manifest_path=manifest_path,
    )
    if current_policy_id:
        return current_policy_id
    if _is_fixed_no_incumbent_policy_spec(latest.get("current_policy_spec")):
        return None
    if len(iterations) >= 2:
        previous_training = _mapping(iterations[-2].get("training", {}))
        previous_model_config = _mapping(previous_training.get("model_config", {}))
        previous_policy_id = _optional_str(previous_model_config.get("policy_id"))
        if previous_policy_id:
            return previous_policy_id
    return _policy_id_from_policy_spec(latest.get("current_policy_spec"), manifest_path=manifest_path)


def _neural_current_policy_id(
    iterations: tuple[Mapping[str, Any], ...],
    value: Any,
    *,
    manifest_path: Path,
) -> str | None:
    direct = _policy_id_from_policy_spec(value, manifest_path=manifest_path)
    if direct is not None:
        return direct
    if value is None:
        return None
    current = str(value).strip()
    if not current.lower().startswith("neural:"):
        return None
    for iteration in reversed(iterations[:-1]):
        candidates = (
            iteration.get("next_current_policy_spec"),
            iteration.get("checkpoint_policy_spec"),
        )
        if current not in {str(candidate) for candidate in candidates if candidate is not None}:
            continue
        training = _mapping(iteration.get("training", {}))
        model_config = _mapping(training.get("model_config", {}))
        policy_id = _optional_str(model_config.get("policy_id"))
        if policy_id:
            return policy_id
    return None


def _is_fixed_no_incumbent_policy_spec(value: Any) -> bool:
    if value is None:
        return False
    return str(value).partition("?")[0].strip().lower() in {"random-legal", "simple-legal"}


def _policy_id_from_policy_spec(value: Any, *, manifest_path: Path) -> str | None:
    if value is None:
        return None
    body = str(value).partition("?")[0].strip()
    lowered = body.lower()
    if lowered in {"random-legal", "simple-legal"}:
        return None
    if lowered == "scripted-teacher":
        return "scripted-teacher"
    linear_prefix = "linear:"
    if not lowered.startswith(linear_prefix):
        return None
    checkpoint_text = body[len(linear_prefix) :].strip()
    if not checkpoint_text:
        return None
    for checkpoint_path in _candidate_checkpoint_paths(checkpoint_text, manifest_path=manifest_path):
        if not checkpoint_path.exists() or not checkpoint_path.is_file():
            continue
        try:
            payload = _mapping(json.loads(checkpoint_path.read_text(encoding="utf-8")))
        except (OSError, ValueError, json.JSONDecodeError):
            continue
        policy_id = _optional_str(payload.get("policy_id"))
        if policy_id:
            return policy_id
    return None


def _benchmark_reference_policy_ids(
    manifest: Mapping[str, Any],
    *,
    manifest_path: Path,
) -> tuple[str, ...]:
    policy_ids: list[str] = []
    for policy_spec in _sequence(manifest.get("benchmark_reference_policy_specs", ())):
        policy_id = _policy_id_from_policy_spec(policy_spec, manifest_path=manifest_path)
        if policy_id is not None and policy_id not in policy_ids:
            policy_ids.append(policy_id)
    return tuple(policy_ids)


def _candidate_checkpoint_paths(checkpoint_text: str, *, manifest_path: Path) -> tuple[Path, ...]:
    checkpoint_path = Path(checkpoint_text).expanduser()
    if checkpoint_path.is_absolute():
        return (checkpoint_path,)
    return (
        manifest_path.parent / checkpoint_path,
        Path.cwd() / checkpoint_path,
        checkpoint_path,
    )


def _benchmark_summary(benchmark: Mapping[str, Any] | None, policy_id: str | None) -> _BenchmarkSummary:
    if benchmark is None or not policy_id:
        return _BenchmarkSummary(wins=0, games=0, capped_games=0, opponents=())
    accumulators: dict[str, _BenchmarkOpponentAccumulator] = {}
    ordered_opponents: list[str] = []
    for result in tuple(_mapping(result) for result in _sequence(benchmark.get("head_to_heads", ()))):
        result_games = int(result.get("games", 0))
        if result.get("first_policy_id") == policy_id:
            _add_opponent_result(
                accumulators=accumulators,
                ordered_opponents=ordered_opponents,
                opponent_policy_id=str(result.get("second_policy_id")),
                wins=int(result.get("first_policy_wins", 0)),
                games=result_games,
                capped_games=int(result.get("capped_games", 0)),
            )
        elif result.get("second_policy_id") == policy_id:
            _add_opponent_result(
                accumulators=accumulators,
                ordered_opponents=ordered_opponents,
                opponent_policy_id=str(result.get("first_policy_id")),
                wins=int(result.get("second_policy_wins", 0)),
                games=result_games,
                capped_games=int(result.get("capped_games", 0)),
            )
    if accumulators:
        return _summary_from_opponents(accumulators, ordered_opponents)

    for result in tuple(_mapping(result) for result in _sequence(benchmark.get("matchups", ()))):
        metrics = _mapping(result.get("metrics", {}))
        result_games = int(metrics.get("games", 0))
        if result.get("p1_policy_id") == policy_id:
            _add_opponent_result(
                accumulators=accumulators,
                ordered_opponents=ordered_opponents,
                opponent_policy_id=str(result.get("p2_policy_id")),
                wins=int(metrics.get("p1_wins", 0)),
                games=result_games,
                capped_games=int(metrics.get("capped_games", 0)),
            )
        elif result.get("p2_policy_id") == policy_id:
            _add_opponent_result(
                accumulators=accumulators,
                ordered_opponents=ordered_opponents,
                opponent_policy_id=str(result.get("p1_policy_id")),
                wins=int(metrics.get("p2_wins", 0)),
                games=result_games,
                capped_games=int(metrics.get("capped_games", 0)),
            )
    return _summary_from_opponents(accumulators, ordered_opponents)


@dataclass
class _BenchmarkOpponentAccumulator:
    opponent_policy_id: str
    wins: int = 0
    games: int = 0
    capped_games: int = 0

    def add(self, *, wins: int, games: int, capped_games: int) -> None:
        self.wins += wins
        self.games += games
        self.capped_games += capped_games

    def to_summary(self) -> _BenchmarkOpponentSummary:
        return _BenchmarkOpponentSummary(
            opponent_policy_id=self.opponent_policy_id,
            wins=self.wins,
            games=self.games,
            capped_games=self.capped_games,
        )


def _add_opponent_result(
    *,
    accumulators: dict[str, _BenchmarkOpponentAccumulator],
    ordered_opponents: list[str],
    opponent_policy_id: str,
    wins: int,
    games: int,
    capped_games: int,
) -> None:
    accumulator = accumulators.get(opponent_policy_id)
    if accumulator is None:
        accumulator = _BenchmarkOpponentAccumulator(opponent_policy_id=opponent_policy_id)
        accumulators[opponent_policy_id] = accumulator
        ordered_opponents.append(opponent_policy_id)
    accumulator.add(wins=wins, games=games, capped_games=capped_games)


def _summary_from_opponents(
    accumulators: Mapping[str, _BenchmarkOpponentAccumulator],
    ordered_opponents: list[str],
) -> _BenchmarkSummary:
    opponents = tuple(accumulators[opponent].to_summary() for opponent in ordered_opponents)
    return _BenchmarkSummary(
        wins=sum(opponent.wins for opponent in opponents),
        games=sum(opponent.games for opponent in opponents),
        capped_games=sum(opponent.capped_games for opponent in opponents),
        opponents=opponents,
    )


def _benchmark_without_opponent(
    summary: _BenchmarkSummary,
    opponent_policy_id: str | None,
) -> _BenchmarkSummary:
    if opponent_policy_id is None:
        return summary
    opponents = tuple(
        opponent for opponent in summary.opponents if opponent.opponent_policy_id != opponent_policy_id
    )
    return _BenchmarkSummary(
        wins=sum(opponent.wins for opponent in opponents),
        games=sum(opponent.games for opponent in opponents),
        capped_games=sum(opponent.capped_games for opponent in opponents),
        opponents=opponents,
    )


def _benchmark_without_opponents(
    summary: _BenchmarkSummary,
    opponent_policy_ids: IterableABC[str],
) -> _BenchmarkSummary:
    excluded = set(opponent_policy_ids)
    if not excluded:
        return summary
    opponents = tuple(
        opponent for opponent in summary.opponents if opponent.opponent_policy_id not in excluded
    )
    return _BenchmarkSummary(
        wins=sum(opponent.wins for opponent in opponents),
        games=sum(opponent.games for opponent in opponents),
        capped_games=sum(opponent.capped_games for opponent in opponents),
        opponents=opponents,
    )


def _gated_benchmark_opponents(
    opponents: tuple[_BenchmarkOpponentSummary, ...],
    *,
    required_opponents: tuple[str, ...],
    excluded_opponents: tuple[str, ...] = (),
) -> tuple[_BenchmarkOpponentSummary, ...]:
    excluded = set(excluded_opponents)
    if not required_opponents:
        return tuple(opponent for opponent in opponents if opponent.opponent_policy_id not in excluded)
    required = set(required_opponents)
    return tuple(
        opponent
        for opponent in opponents
        if opponent.opponent_policy_id in required and opponent.opponent_policy_id not in excluded
    )


def _missing_benchmark_opponents(
    opponents: tuple[_BenchmarkOpponentSummary, ...],
    *,
    required_opponents: tuple[str, ...],
) -> tuple[str, ...]:
    if not required_opponents:
        return ()
    seen = {opponent.opponent_policy_id for opponent in opponents}
    return tuple(opponent for opponent in required_opponents if opponent not in seen)


def _benchmark_opponent_for_policy_id(
    opponents: tuple[_BenchmarkOpponentSummary, ...],
    policy_id: str | None,
) -> _BenchmarkOpponentSummary | None:
    if policy_id is None:
        return None
    for opponent in opponents:
        if opponent.opponent_policy_id == policy_id:
            return opponent
    return None


def _wilson_lower_bound(wins: int, games: int, *, z: float) -> float:
    if games <= 0:
        return 0.0
    if z == 0.0:
        return wins / games
    p_hat = wins / games
    z_squared = z * z
    denominator = 1.0 + (z_squared / games)
    center = p_hat + (z_squared / (2.0 * games))
    adjustment = z * math.sqrt(((p_hat * (1.0 - p_hat)) + (z_squared / (4.0 * games))) / games)
    return max(0.0, (center - adjustment) / denominator)


def _threshold_check(
    *,
    name: str,
    observed: float | None,
    threshold: float,
    passed: bool,
    message: str,
) -> PromotionGateCheck:
    return PromotionGateCheck(
        name=name,
        passed=passed,
        observed=observed,
        threshold=threshold,
        message=message,
    )


def _manifest_path(path: Path) -> Path:
    resolved = path.expanduser()
    if resolved.is_dir():
        resolved = resolved / "manifest.json"
    if not resolved.exists():
        raise FileNotFoundError(f"Experiment manifest does not exist: {resolved}")
    if not resolved.is_file():
        raise ValueError(f"Experiment manifest path must be a file: {resolved}")
    return resolved


def _load_manifest(path: Path) -> Mapping[str, Any]:
    return _mapping(json.loads(path.read_text(encoding="utf-8")))


def _capped_rate(metrics: Mapping[str, Any]) -> float | None:
    games = int(metrics.get("games", 0))
    if games <= 0:
        return None
    return int(metrics.get("capped_games", 0)) / games


def _teacher_degradation_rate(value: Any) -> float | None:
    if value is None:
        return None
    summary = _mapping(value)
    decisions = int(summary.get("scripted_teacher_decisions", 0))
    if decisions <= 0:
        decisions = int(summary.get("total_decisions", 0))
    if decisions <= 0:
        return None
    degraded = int(summary.get("unknown_move_decisions", 0)) + int(summary.get("fallback_decisions", 0))
    return degraded / decisions


def _mapping(value: Any) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError("expected JSON object payload.")
    return value


def _optional_mapping(value: Any) -> Mapping[str, Any] | None:
    if value is None:
        return None
    return _mapping(value)


def _sequence(value: Any) -> tuple[Any, ...]:
    if isinstance(value, (str, bytes, Mapping)) or not isinstance(value, IterableABC):
        raise ValueError("expected JSON array payload.")
    return tuple(value)


def _optional_str(value: Any) -> str | None:
    if value is None:
        return None
    return str(value)
