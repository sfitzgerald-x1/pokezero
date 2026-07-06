"""Read-only hazard-pricing trajectory metrics for diversity-tier probes."""

from __future__ import annotations

import math
import re
from typing import Any, Iterable, Mapping


def number_or_none(value: Any) -> float | None:
    if value is None:
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def parse_milestone_games(row: Mapping[str, Any]) -> int | None:
    for key in ("milestone_games", "games_at", "games"):
        value = row.get(key)
        if isinstance(value, int) and value >= 0:
            return value
    label = str(row.get("label") or "")
    parsed = parse_milestone_games_text(label)
    if parsed is not None:
        return parsed
    return parse_milestone_games_text(str(row.get("checkpoint") or ""))


def parse_milestone_games_text(text: str) -> int | None:
    normalized = str(text or "").strip()
    if re.fullmatch(r"\d{1,7}", normalized):
        return int(normalized)
    decimal_matches = re.findall(
        r"(?<![a-z0-9])(\d+)[._-](\d+)\s*([kKmM])(?![a-z0-9])",
        normalized,
    )
    if decimal_matches:
        whole, fraction, suffix = decimal_matches[-1]
        value = float(f"{whole}.{fraction}")
        return int(value * (1_000_000 if suffix.lower() == "m" else 1_000))
    suffixed_matches = re.findall(
        r"(?<![a-z0-9])(\d+(?:\.\d+)?)\s*([kKmM])(?![a-z0-9])",
        normalized,
    )
    if suffixed_matches:
        value, suffix = suffixed_matches[-1]
        return int(float(value) * (1_000_000 if suffix.lower() == "m" else 1_000))
    return None


def correct_pricing(row: Mapping[str, Any]) -> float | None:
    spread = number_or_none(row.get("value_spread"))
    self_response = number_or_none(row.get("value_self_hazard_response"))
    opp_response = number_or_none(row.get("value_opp_hazard_response"))
    if spread is None or spread <= 0 or self_response is None or opp_response is None:
        return None
    return round((opp_response - self_response) / spread, 6)


def _linear_slope(values: list[float]) -> float | None:
    if len(values) < 2:
        return None
    xs = list(range(len(values)))
    x_mean = sum(xs) / len(xs)
    y_mean = sum(values) / len(values)
    denominator = sum((x - x_mean) ** 2 for x in xs)
    if denominator <= 0:
        return None
    return sum((x - x_mean) * (y - y_mean) for x, y in zip(xs, values)) / denominator


def aggregate_hazard_rows(
    rows: Iterable[Mapping[str, Any]],
    *,
    threshold: float = 0.10,
    trend_epsilon: float = 0.0,
) -> dict[str, Any]:
    points = []
    for index, row in enumerate(rows):
        pricing = correct_pricing(row)
        milestone = parse_milestone_games(row)
        value_spread = number_or_none(row.get("value_spread"))
        self_response = number_or_none(row.get("value_self_hazard_response"))
        opp_response = number_or_none(row.get("value_opp_hazard_response"))
        spin_response = number_or_none(row.get("spin_hazard_response"))
        point = {
            "index": index,
            "label": row.get("label"),
            "checkpoint": row.get("checkpoint"),
            "milestone_games": milestone,
            "value_spread": value_spread,
            "value_self_hazard_response": self_response,
            "value_opp_hazard_response": opp_response,
            "spin_hazard_response": spin_response,
            "correct_pricing": pricing,
            "correctly_signed": bool(self_response is not None and opp_response is not None and self_response < 0 < opp_response),
            "level_pass": bool(pricing is not None and pricing >= threshold),
            "spin_corrob": bool(spin_response is not None and abs(spin_response) > 0),
        }
        points.append(point)
    points.sort(key=lambda point: (point["milestone_games"] is None, point["milestone_games"] or point["index"], point["index"]))
    valid = [point for point in points if point["correct_pricing"] is not None]
    pricing_values = [float(point["correct_pricing"]) for point in valid]
    slope = _linear_slope(pricing_values)
    strict_monotone = (
        len(pricing_values) >= 2
        and all(curr + trend_epsilon >= prev for prev, curr in zip(pricing_values, pricing_values[1:]))
    )
    trend_pass = (
        len(pricing_values) >= 5
        and slope is not None
        and slope > trend_epsilon
        and pricing_values[-1] >= pricing_values[0]
    )
    last_two = valid[-2:] if len(valid) >= 2 else []
    last_two_level_pass = len(last_two) == 2 and all(point["level_pass"] for point in last_two)
    last_two_signed = len(last_two) == 2 and all(point["correctly_signed"] for point in last_two)
    last_two_spin_corrob = len(last_two) == 2 and all(point["spin_corrob"] for point in last_two)
    missing_milestone_points = [
        int(point["index"])
        for point in valid
        if point["milestone_games"] is None
    ]
    ordering_complete = not missing_milestone_points
    gate_pass = (
        len(valid) >= 5
        and ordering_complete
        and trend_pass
        and last_two_level_pass
        and last_two_signed
        and last_two_spin_corrob
    )
    return {
        "schema_version": "pokezero.hazard_trajectory.v1",
        "threshold": threshold,
        "trend_epsilon": trend_epsilon,
        "valid_points": len(valid),
        "latest_correct_pricing": pricing_values[-1] if pricing_values else None,
        "trend_slope": round(slope, 6) if slope is not None else None,
        "trend_pass": trend_pass,
        "monotone_non_decreasing": strict_monotone,
        "ordering_complete": ordering_complete,
        "missing_milestone_point_indexes": missing_milestone_points,
        "last_two_level_pass": last_two_level_pass,
        "last_two_correctly_signed": last_two_signed,
        "last_two_spin_corrob": last_two_spin_corrob,
        "spin_corrob": last_two_spin_corrob,
        "gate_pass": gate_pass,
        "points": points,
    }
