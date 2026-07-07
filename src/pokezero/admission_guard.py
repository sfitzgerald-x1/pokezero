"""Guardrails for non-vacuous population/admission artifacts.

The deployment-side admission runner is intentionally outside the public repo,
but the invariants are public and testable: an admission must have a real
strength floor and active novelty evidence. This module validates generic JSON
artifacts without depending on private cluster/run schemas.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping, Sequence


ADMISSION_GUARD_SCHEMA_VERSION = "pokezero.admission_guard.v1"


@dataclass(frozen=True)
class AdmissionGuardConfig:
    """Minimum evidence required before an artifact can be treated as admission-ready."""

    min_win_rate_floor: float = 0.0
    min_comparison_vectors: int = 1
    require_vector_distance: bool = True

    def __post_init__(self) -> None:
        if not math.isfinite(self.min_win_rate_floor) or self.min_win_rate_floor < 0.0:
            raise ValueError("min_win_rate_floor must be finite and non-negative.")
        if self.min_comparison_vectors < 0:
            raise ValueError("min_comparison_vectors must be non-negative.")


@dataclass(frozen=True)
class AdmissionGuardCheck:
    name: str
    passed: bool
    observed: float | int | None
    threshold: float | int
    message: str
    source: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "passed": self.passed,
            "observed": self.observed,
            "threshold": self.threshold,
            "message": self.message,
            "source": self.source,
        }


@dataclass(frozen=True)
class AdmissionGuardResult:
    config: AdmissionGuardConfig
    checks: tuple[AdmissionGuardCheck, ...]

    @property
    def passed(self) -> bool:
        return all(check.passed for check in self.checks)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": ADMISSION_GUARD_SCHEMA_VERSION,
            "passed": self.passed,
            "config": {
                "min_win_rate_floor": self.config.min_win_rate_floor,
                "min_comparison_vectors": self.config.min_comparison_vectors,
                "require_vector_distance": self.config.require_vector_distance,
            },
            "checks": [check.to_dict() for check in self.checks],
        }


def validate_admission_guard(
    payload: Mapping[str, Any],
    *,
    config: AdmissionGuardConfig | None = None,
) -> AdmissionGuardResult:
    """Validate that an admission artifact has non-vacuous strength/novelty evidence.

    The validator is deliberately schema-tolerant. It accepts the public dashboard
    shapes plus common deployment-summary names so private admission runners can
    call one stable guard without leaking their full schema into this repo.
    """

    resolved_config = config or AdmissionGuardConfig()
    strength_floor = _minimum_win_rate_floor(payload)
    comparison_vectors = _comparison_vector_count(payload)
    vector_threshold = _minimum_vector_distance_threshold(payload)
    observed_vector_distance = _largest_observed_vector_distance(payload)

    checks = [
        AdmissionGuardCheck(
            name="strength_floor_positive",
            passed=strength_floor.value is not None and strength_floor.value > resolved_config.min_win_rate_floor,
            observed=strength_floor.value,
            threshold=resolved_config.min_win_rate_floor,
            message="admission requires at least one positive win-rate floor",
            source=strength_floor.path,
        ),
        AdmissionGuardCheck(
            name="comparison_vectors_present",
            passed=comparison_vectors.value >= resolved_config.min_comparison_vectors,
            observed=comparison_vectors.value,
            threshold=resolved_config.min_comparison_vectors,
            message="admission requires comparison vectors or pairwise novelty evidence",
            source=comparison_vectors.path,
        ),
    ]
    if resolved_config.require_vector_distance:
        checks.append(
            AdmissionGuardCheck(
                name="vector_distance_threshold_positive",
                passed=vector_threshold.value is not None and vector_threshold.value > 0.0,
                observed=vector_threshold.value,
                threshold=0.0,
                message="admission requires an active vector-distance novelty threshold",
                source=vector_threshold.path,
            )
        )
        threshold = vector_threshold.value if vector_threshold.value is not None else 0.0
        checks.append(
            AdmissionGuardCheck(
                name="observed_vector_distance_meets_threshold",
                passed=observed_vector_distance.value is not None and observed_vector_distance.value >= threshold > 0.0,
                observed=observed_vector_distance.value,
                threshold=threshold,
                message="admission requires observed novelty distance to meet the active threshold",
                source=observed_vector_distance.path,
            )
        )
    return AdmissionGuardResult(config=resolved_config, checks=tuple(checks))


@dataclass(frozen=True)
class _ObservedNumber:
    value: float | None
    path: str | None


@dataclass(frozen=True)
class _ObservedCount:
    value: int
    path: str | None


def _minimum_win_rate_floor(payload: Mapping[str, Any]) -> _ObservedNumber:
    candidates: list[_ObservedNumber] = []
    for path in (
        ("min_win_rate",),
        ("min_benchmark_win_rate",),
        ("minimum_win_rate",),
        ("admission", "min_win_rate"),
        ("admission", "min_benchmark_win_rate"),
        ("config", "min_win_rate"),
        ("config", "min_benchmark_win_rate"),
        ("gate", "min_win_rate"),
        ("gate", "min_benchmark_win_rate"),
        ("quality_gate", "min_win_rate"),
        ("quality_gate", "min_benchmark_win_rate"),
    ):
        maybe = _number_at_path(payload, path)
        if maybe.value is not None:
            candidates.append(maybe)
    for path in (
        ("opponent_min_win_rates",),
        ("opponent_win_rate_thresholds",),
        ("config", "opponent_min_win_rates"),
        ("gate", "opponent_min_win_rates"),
        ("quality_gate", "opponent_min_win_rates"),
    ):
        mapping = _mapping_at_path(payload, path)
        if mapping is None:
            continue
        for key, value in mapping.items():
            parsed = _number(value)
            if parsed is not None:
                candidates.append(_ObservedNumber(parsed, ".".join(path + (str(key),))))
    return _min_observed_number(candidates)


def _minimum_vector_distance_threshold(payload: Mapping[str, Any]) -> _ObservedNumber:
    candidates: list[_ObservedNumber] = []
    for path in (
        ("min_vector_distance",),
        ("vector_distance_threshold",),
        ("behavior_cluster_distance",),
        ("admission", "min_vector_distance"),
        ("admission", "vector_distance_threshold"),
        ("config", "min_vector_distance"),
        ("config", "vector_distance_threshold"),
        ("config", "behavior_cluster_distance"),
        ("thresholds", "behavior_cluster_distance"),
        ("behavior_embedding", "distance_threshold"),
        ("diversity", "min_vector_distance"),
    ):
        maybe = _number_at_path(payload, path)
        if maybe.value is not None:
            candidates.append(maybe)
    return _min_observed_number(candidates)


def _largest_observed_vector_distance(payload: Mapping[str, Any]) -> _ObservedNumber:
    candidates: list[_ObservedNumber] = []
    for path in (
        ("vector_distance",),
        ("observed_vector_distance",),
        ("admission", "vector_distance"),
        ("admission", "observed_vector_distance"),
        ("diversity", "vector_distance"),
    ):
        maybe = _number_at_path(payload, path)
        if maybe.value is not None:
            candidates.append(maybe)

    for path in (
        ("behavior_embedding", "pairwise_distances"),
        ("dashboard", "behavior_embedding", "pairwise_distances"),
    ):
        rows = _value_at_path(payload, path)
        if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes)):
            continue
        for index, row in enumerate(rows):
            if not isinstance(row, Mapping):
                continue
            maybe = _number(row.get("distance"))
            if maybe is not None:
                candidates.append(_ObservedNumber(maybe, ".".join(path + (str(index), "distance"))))

    for path in (
        ("policy_js_divergence", "pairwise"),
        ("dashboard", "policy_js_divergence", "pairwise"),
    ):
        rows = _value_at_path(payload, path)
        if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes)):
            continue
        for index, row in enumerate(rows):
            if not isinstance(row, Mapping):
                continue
            for key in ("js_divergence_mean", "js_divergence_max", "distance"):
                maybe = _number(row.get(key))
                if maybe is not None:
                    candidates.append(_ObservedNumber(maybe, ".".join(path + (str(index), key))))
    return _max_observed_number(candidates)


def _comparison_vector_count(payload: Mapping[str, Any]) -> _ObservedCount:
    counts: list[_ObservedCount] = []
    for path in (
        ("comparison_vectors",),
        ("comparison_vector_ids",),
        ("comparisons",),
        ("reference_vectors",),
        ("anchor_vectors",),
        ("anchors",),
        ("admission", "comparison_vectors"),
        ("admission", "reference_vectors"),
        ("config", "comparison_vectors"),
    ):
        count = _sequence_or_mapping_count_at_path(payload, path)
        if count is not None:
            counts.append(_ObservedCount(count, ".".join(path)))

    payoff_vectors = _mapping_at_path(payload, ("payoff_vectors",))
    if payoff_vectors is None:
        payoff_vectors = _mapping_at_path(payload, ("dashboard", "payoff_vectors"))
    if payoff_vectors is not None:
        counts.append(_ObservedCount(_payoff_comparison_count(payoff_vectors), "payoff_vectors"))
    payoff_rank = _mapping_at_path(payload, ("payoff_rank",))
    if payoff_rank is None:
        payoff_rank = _mapping_at_path(payload, ("dashboard", "payoff_rank"))
    if payoff_rank is not None:
        member_count = _number(payoff_rank.get("member_count"))
        opponent_count = _number(payoff_rank.get("opponent_count"))
        if member_count is not None and opponent_count is not None:
            counts.append(
                _ObservedCount(
                    max(0, int(member_count) * int(opponent_count) - 1),
                    "payoff_rank.member_count/opponent_count",
                )
            )

    for path in (
        ("behavior_embedding", "pairwise_distances"),
        ("policy_js_divergence", "pairwise"),
        ("dashboard", "behavior_embedding", "pairwise_distances"),
        ("dashboard", "policy_js_divergence", "pairwise"),
    ):
        count = _sequence_or_mapping_count_at_path(payload, path)
        if count is not None:
            counts.append(_ObservedCount(count, ".".join(path)))

    embedded_count = _number_at_path(payload, ("behavior_embedding", "embedded_count"))
    if embedded_count.value is not None:
        counts.append(
            _ObservedCount(
                max(0, int(embedded_count.value) - 1),
                "behavior_embedding.embedded_count",
            )
        )

    if not counts:
        return _ObservedCount(0, None)
    return max(counts, key=lambda item: item.value)


def _payoff_comparison_count(payoff_vectors: Mapping[str, Any]) -> int:
    count = 0
    for vector in payoff_vectors.values():
        if isinstance(vector, Mapping):
            count += sum(1 for value in vector.values() if _number(value) is not None)
    return count


def _sequence_or_mapping_count_at_path(payload: Mapping[str, Any], path: tuple[str, ...]) -> int | None:
    value = _value_at_path(payload, path)
    if isinstance(value, Mapping):
        return len(value)
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return len(value)
    return None


def _mapping_at_path(payload: Mapping[str, Any], path: tuple[str, ...]) -> Mapping[str, Any] | None:
    value = _value_at_path(payload, path)
    return value if isinstance(value, Mapping) else None


def _number_at_path(payload: Mapping[str, Any], path: tuple[str, ...]) -> _ObservedNumber:
    return _ObservedNumber(_number(_value_at_path(payload, path)), ".".join(path))


def _value_at_path(payload: Mapping[str, Any], path: tuple[str, ...]) -> Any:
    current: Any = payload
    for part in path:
        if not isinstance(current, Mapping):
            return None
        current = current.get(part)
    return current


def _number(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def _max_observed_number(candidates: Sequence[_ObservedNumber]) -> _ObservedNumber:
    if not candidates:
        return _ObservedNumber(None, None)
    return max(candidates, key=lambda item: item.value if item.value is not None else float("-inf"))


def _min_observed_number(candidates: Sequence[_ObservedNumber]) -> _ObservedNumber:
    if not candidates:
        return _ObservedNumber(None, None)
    return min(candidates, key=lambda item: item.value if item.value is not None else float("inf"))
