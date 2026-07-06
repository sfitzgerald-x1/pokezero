"""Read-only population coverage metrics for diversity-tier dashboards."""

from __future__ import annotations

import math
from typing import Any, Iterable, Mapping, Sequence

SCHEMA_VERSION = "pokezero.diversity_population_dashboard.v1"
COVERAGE_RATE_SCHEMA_VERSION = "pokezero.diversity_coverage_rate.v1"


DEFAULT_THRESHOLDS: dict[str, float] = {
    "move_rate": 0.05,
    "pivot_rate": 0.05,
    "avg_turns": 5.0,
    "distinct_moves": 3.0,
    "behavior_cluster_distance": 0.20,
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


def _euclidean_distance(left: Sequence[float], right: Sequence[float]) -> float:
    if len(left) != len(right):
        raise ValueError("embedding vectors must have equal length")
    if not left:
        return 0.0
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(left, right)) / len(left))


def _probability_vector(value: Any) -> tuple[float, ...] | None:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return None
    vector: list[float] = []
    for item in value:
        parsed = number_or_none(item)
        if parsed is None or parsed < 0.0:
            return None
        vector.append(parsed)
    total = sum(vector)
    if not vector or total <= 0.0:
        return None
    return tuple(item / total for item in vector)


def _jensen_shannon_divergence(left: Sequence[float], right: Sequence[float]) -> float:
    if len(left) != len(right):
        raise ValueError("probability vectors must have equal length")
    if not left:
        return 0.0

    def kl_divergence(probs: Sequence[float], baseline: Sequence[float]) -> float:
        total = 0.0
        for prob, base in zip(probs, baseline):
            if prob <= 0.0:
                continue
            if base <= 0.0:
                raise ValueError("baseline probability must be positive when compared value is positive")
            total += prob * math.log(prob / base)
        return total

    midpoint = tuple((a + b) * 0.5 for a, b in zip(left, right))
    return 0.5 * kl_divergence(left, midpoint) + 0.5 * kl_divergence(right, midpoint)


def _policy_state_vectors(row: Mapping[str, Any]) -> dict[str, tuple[float, ...]]:
    states = row.get("states")
    if not isinstance(states, Sequence) or isinstance(states, (str, bytes)):
        return {}
    vectors: dict[str, tuple[float, ...]] = {}
    for index, state in enumerate(states):
        if not isinstance(state, Mapping):
            continue
        state_id = state.get("state_id", index)
        vector = _probability_vector(state.get("action_probabilities"))
        if vector is None:
            continue
        vectors[str(state_id)] = vector
    return vectors


def policy_js_divergence_summary(rows: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """Summarize pairwise policy JS-divergence over a fixed state corpus.

    Rows are read-only probe artifacts. Each row should contain a unique label
    and a ``states`` list with stable ``state_id`` plus action probabilities.
    Pairwise divergences are computed only over shared state ids so incomplete
    probes do not invent zero-probability states.
    """
    materialized = [dict(row) for row in rows]
    labels = [_label(row, index) for index, row in enumerate(materialized)]
    policies: list[dict[str, Any]] = []
    skipped_labels: list[str] = []
    for label, row in zip(labels, materialized):
        vectors = _policy_state_vectors(row)
        if not vectors:
            skipped_labels.append(label)
            continue
        policies.append({"label": label, "vectors": vectors})
    policies.sort(key=lambda item: item["label"])

    pairwise: list[dict[str, Any]] = []
    all_distances: list[float] = []
    state_ids: set[str] = set()
    for policy in policies:
        state_ids.update(policy["vectors"])

    for left_index, left in enumerate(policies):
        for right in policies[left_index + 1:]:
            shared = sorted(set(left["vectors"]) & set(right["vectors"]))
            distances: list[float] = []
            for state_id in shared:
                left_vector = left["vectors"][state_id]
                right_vector = right["vectors"][state_id]
                if len(left_vector) != len(right_vector):
                    continue
                distances.append(_jensen_shannon_divergence(left_vector, right_vector))
            all_distances.extend(distances)
            mean_distance = sum(distances) / len(distances) if distances else None
            max_distance = max(distances) if distances else None
            pairwise.append(
                {
                    "left": left["label"],
                    "right": right["label"],
                    "shared_state_count": len(distances),
                    "js_divergence_mean": round(mean_distance, 6) if mean_distance is not None else None,
                    "js_divergence_max": round(max_distance, 6) if max_distance is not None else None,
                }
            )

    mean_pairwise = sum(all_distances) / len(all_distances) if all_distances else None
    max_pairwise = max(all_distances) if all_distances else None
    return {
        "divergence": "jensen_shannon_nats",
        "policy_count": len(policies),
        "skipped_count": len(skipped_labels),
        "skipped_labels": sorted(skipped_labels),
        "state_count": len(state_ids),
        "pair_count": len(pairwise),
        "mean_pairwise_js_divergence": round(mean_pairwise, 6) if mean_pairwise is not None else None,
        "max_pairwise_js_divergence": round(max_pairwise, 6) if max_pairwise is not None else None,
        "pairwise": pairwise,
        "rows": [
            {
                "label": policy["label"],
                "state_count": len(policy["vectors"]),
            }
            for policy in policies
        ],
    }


def _move_usage_features(rows: Sequence[Mapping[str, Any]]) -> tuple[str, ...]:
    moves: set[str] = set()
    for row in rows:
        usage = row.get("move_usage")
        if not isinstance(usage, Mapping):
            continue
        for move, value in usage.items():
            if number_or_none(value) is not None:
                moves.add(str(move))
    return tuple(f"move_usage:{move}" for move in sorted(moves))


def _move_usage_embedding(row: Mapping[str, Any], moves: Sequence[str]) -> tuple[tuple[float, ...], int] | None:
    usage = row.get("move_usage")
    if not isinstance(usage, Mapping):
        return None
    values: list[float] = []
    active = 0
    for move in moves:
        value = number_or_none(usage.get(move))
        if value is None:
            value = 0.0
        elif value != 0.0:
            active += 1
        values.append(value)
    if active == 0:
        return None
    return tuple(values), active


def _connected_components_from_distances(
    labels: Sequence[str],
    distances: Mapping[tuple[str, str], float],
    *,
    threshold: float,
) -> list[list[str]]:
    adjacency: dict[str, set[str]] = {label: set() for label in labels}
    for left in labels:
        for right in labels:
            if left >= right:
                continue
            distance = distances.get((left, right), distances.get((right, left)))
            if distance is not None and distance <= threshold:
                adjacency[left].add(right)
                adjacency[right].add(left)

    components: list[list[str]] = []
    visited: set[str] = set()
    for label in sorted(labels):
        if label in visited:
            continue
        stack = [label]
        component: list[str] = []
        visited.add(label)
        while stack:
            current = stack.pop()
            component.append(current)
            for neighbor in sorted(adjacency[current], reverse=True):
                if neighbor in visited:
                    continue
                visited.add(neighbor)
                stack.append(neighbor)
        components.append(sorted(component))
    return components


def behavior_embedding_summary(
    rows: Iterable[Mapping[str, Any]],
    *,
    distance_threshold: float | None = None,
) -> dict[str, Any]:
    """Cluster behavior-probe rows by a stable read-only behavior embedding.

    This is a dashboard cross-check only. It is intentionally deterministic and
    uses raw move-usage probe outputs rather than the curated strategy-axis
    buckets, reward, admission, or matchmaking signals.
    """
    threshold = (
        DEFAULT_THRESHOLDS["behavior_cluster_distance"]
        if distance_threshold is None
        else float(distance_threshold)
    )
    if not math.isfinite(threshold) or threshold < 0.0:
        raise ValueError("distance_threshold must be finite and non-negative")
    materialized = [dict(row) for row in rows]
    labels = [_label(row, index) for index, row in enumerate(materialized)]
    feature_names = _move_usage_features(materialized)
    moves = tuple(feature.split(":", 1)[1] for feature in feature_names)
    embedded_rows: list[dict[str, Any]] = []
    skipped_labels: list[str] = []
    for label, row in zip(labels, materialized):
        embedding = _move_usage_embedding(row, moves)
        if embedding is None:
            skipped_labels.append(label)
            continue
        vector, active = embedding
        embedded_rows.append(
            {
                "label": label,
                "vector": vector,
                "active_feature_count": active,
                "zero_feature_count": len(feature_names) - active,
            }
        )
    embedded_rows.sort(key=lambda item: item["label"])

    pairwise: list[dict[str, Any]] = []
    raw_distances: dict[tuple[str, str], float] = {}
    for left_index, left in enumerate(embedded_rows):
        for right in embedded_rows[left_index + 1:]:
            distance = _euclidean_distance(left["vector"], right["vector"])
            raw_distances[(left["label"], right["label"])] = distance
            pairwise.append(
                {
                    "left": left["label"],
                    "right": right["label"],
                    "distance": round(distance, 6),
                }
            )

    components = _connected_components_from_distances(
        [row["label"] for row in embedded_rows],
        raw_distances,
        threshold=threshold,
    )

    public_clusters = [
        {
            "representative_label": component[0],
            "members": component,
            "size": len(component),
        }
        for component in components
    ]
    return {
        "embedding_kind": "move_usage_distribution",
        "feature_names": list(feature_names),
        "distance_threshold": threshold,
        "embedded_count": len(embedded_rows),
        "skipped_count": len(skipped_labels),
        "skipped_labels": skipped_labels,
        "cluster_count": len(public_clusters),
        "clusters": public_clusters,
        "pairwise_distances": pairwise,
        "rows": [
            {
                "label": row["label"],
                "active_feature_count": row["active_feature_count"],
                "zero_feature_count": row["zero_feature_count"],
            }
            for row in embedded_rows
        ],
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
    policy_prior_rows: Iterable[Mapping[str, Any]] | None = None,
    thresholds: Mapping[str, float] | None = None,
) -> dict[str, Any]:
    threshold_values = {**DEFAULT_THRESHOLDS, **dict(thresholds or {})}
    rows = [dict(row) for row in behavior_rows]
    behavior = summarize_behavior_spread(rows, thresholds=threshold_values)
    behavior_embedding = behavior_embedding_summary(
        rows,
        distance_threshold=threshold_values["behavior_cluster_distance"],
    )
    payoff_rank = payoff_effective_rank(payoff_vectors or {})
    return {
        "schema_version": SCHEMA_VERSION,
        "behavior": behavior,
        "behavior_embedding": behavior_embedding,
        "policy_js_divergence": policy_js_divergence_summary(policy_prior_rows or ()),
        "payoff_rank": payoff_rank,
    }


def diversity_coverage_rate_report(
    milestones: Iterable[Mapping[str, Any]],
) -> dict[str, Any]:
    """Build per-100k-game coverage-rate metrics from dashboard snapshots.

    This is a read-only trend artifact for the diversity-tier dashboard.  It
    does not feed admission, training, rewards, or matchmaking.
    """
    points = [_coverage_point(row, index) for index, row in enumerate(milestones)]
    points.sort(key=lambda point: point["games"])
    _validate_coverage_points(points)
    intervals = [
        _coverage_interval(previous, current)
        for previous, current in zip(points, points[1:])
    ]
    return {
        "schema_version": COVERAGE_RATE_SCHEMA_VERSION,
        "units": "delta_per_100k_games",
        "point_count": len(points),
        "interval_count": len(intervals),
        "metrics": [
            "payoff_effective_rank",
            "behavior_live_axis_count",
            "behavior_cluster_count",
            "policy_mean_pairwise_js_divergence",
            "policy_max_pairwise_js_divergence",
        ],
        "points": points,
        "intervals": intervals,
        "latest_interval": intervals[-1] if intervals else None,
    }


def _coverage_point(row: Mapping[str, Any], index: int) -> dict[str, Any]:
    games = _coverage_games(row)
    dashboard = row.get("dashboard", row)
    if not isinstance(dashboard, Mapping):
        raise ValueError("coverage milestone dashboard must be a JSON object")
    return {
        "label": str(row.get("label") or dashboard.get("label") or f"milestone-{index}"),
        "games": games,
        "metrics": _coverage_metrics(dashboard),
    }


def _coverage_games(row: Mapping[str, Any]) -> int:
    for key in ("games", "completed_games", "milestone_games"):
        value = row.get(key)
        if value is None:
            continue
        parsed = number_or_none(value)
        if parsed is None or parsed < 0 or int(parsed) != parsed:
            raise ValueError(f"coverage milestone {key} must be a non-negative integer")
        return int(parsed)
    raise ValueError("coverage milestone is missing games/completed_games/milestone_games")


def _coverage_metrics(dashboard: Mapping[str, Any]) -> dict[str, float | int | None]:
    payoff_rank = dashboard.get("payoff_rank")
    behavior = dashboard.get("behavior")
    behavior_embedding = dashboard.get("behavior_embedding")
    policy_js = dashboard.get("policy_js_divergence")
    if not isinstance(payoff_rank, Mapping):
        payoff_rank = {}
    if not isinstance(behavior, Mapping):
        behavior = {}
    if not isinstance(behavior_embedding, Mapping):
        behavior_embedding = {}
    if not isinstance(policy_js, Mapping):
        policy_js = {}
    return {
        "payoff_effective_rank": _round_optional(number_or_none(payoff_rank.get("effective_rank"))),
        "behavior_live_axis_count": _int_or_none(behavior.get("live_axis_count")),
        "behavior_cluster_count": _int_or_none(behavior_embedding.get("cluster_count")),
        "policy_mean_pairwise_js_divergence": _round_optional(
            number_or_none(policy_js.get("mean_pairwise_js_divergence"))
        ),
        "policy_max_pairwise_js_divergence": _round_optional(
            number_or_none(policy_js.get("max_pairwise_js_divergence"))
        ),
    }


def _validate_coverage_points(points: Sequence[Mapping[str, Any]]) -> None:
    seen: set[int] = set()
    for point in points:
        games = int(point["games"])
        if games in seen:
            raise ValueError(f"duplicate coverage milestone games: {games}")
        seen.add(games)


def _coverage_interval(previous: Mapping[str, Any], current: Mapping[str, Any]) -> dict[str, Any]:
    previous_games = int(previous["games"])
    current_games = int(current["games"])
    game_delta = current_games - previous_games
    if game_delta <= 0:
        raise ValueError("coverage milestones must be strictly increasing")
    scale = 100_000.0 / game_delta
    previous_metrics = previous["metrics"]
    current_metrics = current["metrics"]
    if not isinstance(previous_metrics, Mapping) or not isinstance(current_metrics, Mapping):
        raise ValueError("coverage point metrics must be JSON objects")
    rates: dict[str, float | None] = {}
    deltas: dict[str, float | None] = {}
    for metric in (
        "payoff_effective_rank",
        "behavior_live_axis_count",
        "behavior_cluster_count",
        "policy_mean_pairwise_js_divergence",
        "policy_max_pairwise_js_divergence",
    ):
        previous_value = number_or_none(previous_metrics.get(metric))
        current_value = number_or_none(current_metrics.get(metric))
        if previous_value is None or current_value is None:
            deltas[metric] = None
            rates[metric] = None
            continue
        delta = current_value - previous_value
        deltas[metric] = round(delta, 6)
        rates[metric] = round(delta * scale, 6)
    return {
        "from_games": previous_games,
        "to_games": current_games,
        "game_delta": game_delta,
        "from_label": previous.get("label"),
        "to_label": current.get("label"),
        "deltas": deltas,
        "rates_per_100k_games": rates,
    }


def _int_or_none(value: Any) -> int | None:
    parsed = number_or_none(value)
    if parsed is None or int(parsed) != parsed:
        return None
    return int(parsed)


def _round_optional(value: float | None) -> float | None:
    return round(value, 6) if value is not None else None
