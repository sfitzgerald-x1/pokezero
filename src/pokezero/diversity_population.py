"""Read-only population coverage metrics for diversity-tier dashboards."""

from __future__ import annotations

import math
from typing import Any, Iterable, Mapping

SCHEMA_VERSION = "pokezero.diversity_population_dashboard.v1"


DEFAULT_THRESHOLDS: dict[str, float] = {
    "move_rate": 0.05,
    "pivot_rate": 0.05,
    "avg_turns": 5.0,
    "distinct_moves": 3.0,
}

AXIS_METRICS: dict[str, tuple[str, ...]] = {
    "hazard_cycle": ("move_class_rate:hazard", "move_class_rate:clear"),
    "tempo": ("move_class_rate:attack", "move_class_rate:status", "move_class_rate:heal", "avg_turns"),
    "aggression_structure": ("move_class_rate:setup", "move_class_rate:phaze"),
    "interaction": ("pivot_rate",),
    "generic_behavior": ("distinct_moves",),
}


def number_or_none(value: Any) -> float | None:
    if value is None:
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def _metric_value(row: Mapping[str, Any], metric: str) -> float | None:
    if metric.startswith("move_class_rate:"):
        move_class = metric.split(":", 1)[1]
        usage = row.get("move_class_usage")
        if not isinstance(usage, Mapping):
            return None
        entry = usage.get(move_class)
        if not isinstance(entry, Mapping):
            return None
        return number_or_none(entry.get("rate"))
    return number_or_none(row.get(metric))


def _metric_threshold(metric: str, thresholds: Mapping[str, float]) -> float:
    if metric.startswith("move_class_rate:"):
        return float(thresholds.get("move_rate", DEFAULT_THRESHOLDS["move_rate"]))
    if metric == "pivot_rate":
        return float(thresholds.get("pivot_rate", DEFAULT_THRESHOLDS["pivot_rate"]))
    if metric == "avg_turns":
        return float(thresholds.get("avg_turns", DEFAULT_THRESHOLDS["avg_turns"]))
    if metric == "distinct_moves":
        return float(thresholds.get("distinct_moves", DEFAULT_THRESHOLDS["distinct_moves"]))
    return 0.0


def _label(row: Mapping[str, Any], index: int) -> str:
    for key in ("label", "member_id", "policy_id", "checkpoint"):
        value = row.get(key)
        if value not in (None, ""):
            return str(value)
    return f"row-{index}"


def summarize_behavior_spread(
    rows: Iterable[Mapping[str, Any]],
    *,
    thresholds: Mapping[str, float] | None = None,
) -> dict[str, Any]:
    """Summarize read-only behavioral spread across a population.

    The returned metrics are dashboard signals only. They are deliberately shaped
    as observations about population coverage, not training rewards or admission
    criteria.
    """
    threshold_values = {**DEFAULT_THRESHOLDS, **dict(thresholds or {})}
    materialized = [dict(row) for row in rows]
    axes: dict[str, Any] = {}
    live_axis_count = 0
    live_metrics: list[str] = []
    labels = [_label(row, index) for index, row in enumerate(materialized)]
    for axis, metrics in AXIS_METRICS.items():
        metric_payloads: dict[str, Any] = {}
        axis_live = False
        for metric in metrics:
            values = []
            for label, row in zip(labels, materialized):
                value = _metric_value(row, metric)
                if value is not None:
                    values.append((label, value))
            threshold = _metric_threshold(metric, threshold_values)
            if values:
                numeric_values = [value for _, value in values]
                min_label, min_value = min(values, key=lambda item: item[1])
                max_label, max_value = max(values, key=lambda item: item[1])
                spread = max_value - min_value
                live = spread >= threshold
                if live:
                    axis_live = True
                    live_metrics.append(metric)
                metric_payloads[metric] = {
                    "observed_count": len(values),
                    "threshold": threshold,
                    "min": round(min_value, 6),
                    "max": round(max_value, 6),
                    "spread": round(spread, 6),
                    "live_spread": live,
                    "min_label": min_label,
                    "max_label": max_label,
                }
            else:
                metric_payloads[metric] = {
                    "observed_count": 0,
                    "threshold": threshold,
                    "min": None,
                    "max": None,
                    "spread": None,
                    "live_spread": False,
                    "min_label": None,
                    "max_label": None,
                }
        if axis_live:
            live_axis_count += 1
        axes[axis] = {
            "live_spread": axis_live,
            "metrics": metric_payloads,
        }
    return {
        "population_size": len(materialized),
        "thresholds": threshold_values,
        "live_axis_count": live_axis_count,
        "live_metrics": live_metrics,
        "axes": axes,
    }


def _jacobi_eigenvalues_symmetric(matrix: list[list[float]], *, max_sweeps: int = 80) -> list[float]:
    n = len(matrix)
    if n == 0:
        return []
    work = [row[:] for row in matrix]
    for _ in range(max_sweeps):
        pivot_i = 0
        pivot_j = 1 if n > 1 else 0
        max_offdiag = 0.0
        for i in range(n):
            for j in range(i + 1, n):
                value = abs(work[i][j])
                if value > max_offdiag:
                    max_offdiag = value
                    pivot_i = i
                    pivot_j = j
        if max_offdiag < 1e-12:
            break
        app = work[pivot_i][pivot_i]
        aqq = work[pivot_j][pivot_j]
        apq = work[pivot_i][pivot_j]
        angle = 0.5 * math.atan2(2.0 * apq, aqq - app)
        c = math.cos(angle)
        s = math.sin(angle)
        for k in range(n):
            if k in (pivot_i, pivot_j):
                continue
            aik = work[pivot_i][k]
            ajk = work[pivot_j][k]
            work[pivot_i][k] = work[k][pivot_i] = c * aik - s * ajk
            work[pivot_j][k] = work[k][pivot_j] = s * aik + c * ajk
        work[pivot_i][pivot_i] = c * c * app - 2.0 * s * c * apq + s * s * aqq
        work[pivot_j][pivot_j] = s * s * app + 2.0 * s * c * apq + c * c * aqq
        work[pivot_i][pivot_j] = work[pivot_j][pivot_i] = 0.0
    return [work[i][i] for i in range(n)]


def payoff_effective_rank(payoff_vectors: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    """Compute a neutral-centered payoff-vector effective-rank summary.

    Missing pair entries are treated as neutral 0.5 win rate. The rank is used as
    a read-only population diversity signal, not as a pool admission rule.
    """
    members = sorted(str(member_id) for member_id in payoff_vectors)
    opponents = sorted({str(opponent) for vector in payoff_vectors.values() for opponent in vector})
    rows: list[list[float]] = []
    for member in members:
        vector = payoff_vectors.get(member, {})
        row: list[float] = []
        for opponent in opponents:
            value = number_or_none(vector.get(opponent)) if opponent in vector else None
            row.append((value if value is not None else 0.5) - 0.5)
        rows.append(row)
    if not rows or not opponents:
        return {
            "member_count": len(members),
            "opponent_count": len(opponents),
            "linear_rank": 0,
            "effective_rank": 0.0,
            "eigenvalues": [],
            "members": members,
            "opponents": opponents,
        }
    gram: list[list[float]] = []
    for left in rows:
        gram.append([sum(a * b for a, b in zip(left, right)) for right in rows])
    eigenvalues = sorted((max(0.0, value) for value in _jacobi_eigenvalues_symmetric(gram)), reverse=True)
    positive = [value for value in eigenvalues if value > 1e-10]
    total = sum(positive)
    if total > 0.0:
        probabilities = [value / total for value in positive]
        entropy = -sum(prob * math.log(prob) for prob in probabilities if prob > 0.0)
        effective_rank = math.exp(entropy)
    else:
        effective_rank = 0.0
    return {
        "member_count": len(members),
        "opponent_count": len(opponents),
        "linear_rank": len(positive),
        "effective_rank": round(effective_rank, 6),
        "eigenvalues": [round(value, 8) for value in positive],
        "members": members,
        "opponents": opponents,
    }


def diversity_population_dashboard(
    behavior_rows: Iterable[Mapping[str, Any]],
    *,
    payoff_vectors: Mapping[str, Mapping[str, Any]] | None = None,
    thresholds: Mapping[str, float] | None = None,
) -> dict[str, Any]:
    behavior = summarize_behavior_spread(behavior_rows, thresholds=thresholds)
    payoff_rank = payoff_effective_rank(payoff_vectors or {})
    return {
        "schema_version": SCHEMA_VERSION,
        "behavior": behavior,
        "payoff_rank": payoff_rank,
    }
