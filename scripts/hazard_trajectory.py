"""Aggregate hazard_probe.py milestone outputs into the diversity-tier gate metrics.

This is a read-only measurement tool. It computes the `correct_pricing` scalar
from docs/diversity_tier_design.md:

    (value_opp_hazard_response - value_self_hazard_response) / value_spread

and reports whether the current trajectory satisfies the measurement-only gate:
positive pricing at the configured level for the last two valid milestones,
non-decreasing trend across at least five valid milestones, and non-zero Rapid
Spin response as behavioral corroboration when that signal is available.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


def _number(value: Any) -> float | None:
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
        if isinstance(value, int):
            return value
    for key in ("label", "checkpoint"):
        parsed = _parse_games_text(str(row.get(key) or ""))
        if parsed is not None:
            return parsed
    return None


def _parse_games_text(text: str) -> int | None:
    matches = re.findall(r"(?<![a-z0-9])(\d+(?:\.\d+)?)\s*([kKmM])?(?![a-z0-9])", text)
    if not matches:
        return None
    value, suffix = matches[-1]
    multiplier = 1
    if suffix.lower() == "k":
        multiplier = 1_000
    elif suffix.lower() == "m":
        multiplier = 1_000_000
    return int(float(value) * multiplier)


def correct_pricing(row: Mapping[str, Any]) -> float | None:
    spread = _number(row.get("value_spread"))
    self_response = _number(row.get("value_self_hazard_response"))
    opp_response = _number(row.get("value_opp_hazard_response"))
    if spread is None or spread <= 0 or self_response is None or opp_response is None:
        return None
    return round((opp_response - self_response) / spread, 6)


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
        value_spread = _number(row.get("value_spread"))
        self_response = _number(row.get("value_self_hazard_response"))
        opp_response = _number(row.get("value_opp_hazard_response"))
        spin_response = _number(row.get("spin_hazard_response"))
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
    monotone_non_decreasing = (
        len(pricing_values) >= 2
        and all(curr + trend_epsilon >= prev for prev, curr in zip(pricing_values, pricing_values[1:]))
    )
    last_two = valid[-2:] if len(valid) >= 2 else []
    last_two_level_pass = len(last_two) == 2 and all(point["level_pass"] for point in last_two)
    last_two_signed = len(last_two) == 2 and all(point["correctly_signed"] for point in last_two)
    spin_corrob = any(point["spin_corrob"] for point in valid)
    gate_pass = len(valid) >= 5 and monotone_non_decreasing and last_two_level_pass and last_two_signed and spin_corrob
    return {
        "schema_version": "pokezero.hazard_trajectory.v1",
        "threshold": threshold,
        "trend_epsilon": trend_epsilon,
        "valid_points": len(valid),
        "latest_correct_pricing": pricing_values[-1] if pricing_values else None,
        "monotone_non_decreasing": monotone_non_decreasing,
        "last_two_level_pass": last_two_level_pass,
        "last_two_correctly_signed": last_two_signed,
        "spin_corrob": spin_corrob,
        "gate_pass": gate_pass,
        "points": points,
    }


def _load_rows(path: Path) -> list[Mapping[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, Mapping) and isinstance(payload.get("checkpoints"), list):
        rows = payload["checkpoints"]
    elif isinstance(payload, list):
        rows = payload
    else:
        raise ValueError(f"{path} must be a hazard_probe JSON object with checkpoints or a list of rows")
    return [row for row in rows if isinstance(row, Mapping)]


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hazard", action="append", type=Path, required=True, help="hazard_probe.py JSON; repeatable")
    parser.add_argument("--threshold", type=float, default=0.10)
    parser.add_argument("--trend-epsilon", type=float, default=0.0)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args(argv)

    rows: list[Mapping[str, Any]] = []
    for path in args.hazard:
        rows.extend(_load_rows(path))
    payload = aggregate_hazard_rows(rows, threshold=args.threshold, trend_epsilon=args.trend_epsilon)
    text = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text, encoding="utf-8")
        print(f"[hazard-trajectory] wrote {args.out}", file=sys.stderr)
    else:
        print(text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
