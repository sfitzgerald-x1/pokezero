"""Aggregate G4 refutation reports into cycle-level progress artifacts."""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
import re
from typing import Any, Iterable, Mapping

from .refutation_mining import REFUTATION_REPORT_SCHEMA_VERSION


REFUTATION_CYCLE_REPORT_SCHEMA_VERSION = "pokezero.refutation_cycle_report.v1"


@dataclass(frozen=True)
class RefutationCycleReportInput:
    cycle_id: str
    report_path: Path
    report: Mapping[str, Any]


@dataclass(frozen=True)
class RefutationCycleReport:
    inputs: tuple[RefutationCycleReportInput, ...]
    rows: tuple[Mapping[str, Any], ...]
    mode_trends: Mapping[str, Any]
    oracle_fair_gaps: tuple[Mapping[str, Any], ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": REFUTATION_CYCLE_REPORT_SCHEMA_VERSION,
            "report_count": len(self.rows),
            "cycle_count": len({str(row["cycle_id"]) for row in self.rows}),
            "rows": [dict(row) for row in self.rows],
            "mode_trends": dict(self.mode_trends),
            "oracle_fair_gaps": [dict(gap) for gap in self.oracle_fair_gaps],
        }


def load_refutation_cycle_report_input(spec: str, *, default_index: int) -> RefutationCycleReportInput:
    """Load one cycle-report input from ``[cycle_id=]path`` syntax."""

    if "=" in spec:
        raw_cycle_id, raw_path = spec.split("=", 1)
        cycle_id = raw_cycle_id.strip()
        if not cycle_id:
            raise ValueError("cycle id must not be empty")
    else:
        cycle_id = f"cycle-{default_index + 1:04d}"
        raw_path = spec
    path = Path(raw_path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"refutation report must be a JSON object: {path}")
    return RefutationCycleReportInput(cycle_id=cycle_id, report_path=path, report=payload)


def build_refutation_cycle_report(inputs: Iterable[RefutationCycleReportInput]) -> RefutationCycleReport:
    materialized = tuple(inputs)
    rows = tuple(sorted(
        (_row_from_input(item, index=index) for index, item in enumerate(materialized)),
        key=lambda row: (_natural_sort_key(str(row["cycle_id"])), int(row["input_index"])),
    ))
    _validate_unique_cycle_modes(rows)
    return RefutationCycleReport(
        inputs=materialized,
        rows=rows,
        mode_trends=_mode_trends(rows),
        oracle_fair_gaps=_oracle_fair_gaps(rows),
    )


def write_refutation_cycle_report(path: Path, report: RefutationCycleReport) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _row_from_input(item: RefutationCycleReportInput, *, index: int) -> dict[str, Any]:
    report = item.report
    if report.get("schema_version") != REFUTATION_REPORT_SCHEMA_VERSION:
        raise ValueError(
            f"unsupported refutation report schema in {item.report_path}: "
            f"{report.get('schema_version')!r}"
        )
    config = _mapping(report.get("config"), label="config")
    sampled_win_count = _int(report.get("sampled_win_count"), label="sampled_win_count")
    refuted_game_count = _int(report.get("refuted_game_count"), label="refuted_game_count")
    certified_refutation_count = _int(
        report.get("certified_refutation_count"),
        label="certified_refutation_count",
    )
    if sampled_win_count <= 0:
        raise ValueError(f"sampled_win_count must be positive in {item.report_path}")
    if refuted_game_count < 0 or refuted_game_count > sampled_win_count:
        raise ValueError(f"refuted_game_count must be between 0 and sampled_win_count in {item.report_path}")
    if certified_refutation_count < 0:
        raise ValueError(f"certified_refutation_count must be non-negative in {item.report_path}")
    refutation_rate = _float(report.get("refutation_rate"), label="refutation_rate")
    expected_rate = refuted_game_count / sampled_win_count
    if abs(refutation_rate - expected_rate) > 1e-9:
        raise ValueError(f"refutation_rate does not match counts in {item.report_path}")
    return {
        "input_index": index,
        "cycle_id": item.cycle_id,
        "report_path": str(item.report_path),
        "mode": _str(config.get("mode"), label="config.mode"),
        "champion_policy_id": config.get("champion_policy_id"),
        "champion_player_id": config.get("champion_player_id"),
        "sampled_win_count": sampled_win_count,
        "refuted_game_count": refuted_game_count,
        "certified_refutation_count": certified_refutation_count,
        "certified_refutations_per_sampled_win": _float(
            report.get("certified_refutations_per_sampled_win"),
            label="certified_refutations_per_sampled_win",
        ),
        "refutation_rate": refutation_rate,
    }


def _mode_trends(rows: tuple[Mapping[str, Any], ...]) -> dict[str, Any]:
    trends: dict[str, Any] = {}
    modes = sorted({str(row["mode"]) for row in rows})
    for mode in modes:
        mode_rows = [row for row in rows if row["mode"] == mode]
        values = [float(row["refutation_rate"]) for row in mode_rows]
        slope = _linear_slope(values)
        non_increasing = all(curr <= prev for prev, curr in zip(values, values[1:]))
        trends[mode] = {
            "point_count": len(mode_rows),
            "first_refutation_rate": values[0] if values else None,
            "latest_refutation_rate": values[-1] if values else None,
            "delta": (values[-1] - values[0]) if len(values) >= 2 else None,
            "slope_per_cycle": slope,
            "non_increasing": non_increasing if len(values) >= 2 else None,
            "declining": bool(len(values) >= 2 and values[-1] < values[0] and slope is not None and slope < 0.0),
            "rows": [
                {
                    "cycle_id": row["cycle_id"],
                    "report_path": row["report_path"],
                    "refutation_rate": row["refutation_rate"],
                    "sampled_win_count": row["sampled_win_count"],
                }
                for row in mode_rows
            ],
        }
    return trends


def _oracle_fair_gaps(rows: tuple[Mapping[str, Any], ...]) -> tuple[dict[str, Any], ...]:
    by_cycle: dict[str, dict[str, Mapping[str, Any]]] = {}
    for row in rows:
        by_cycle.setdefault(str(row["cycle_id"]), {})[str(row["mode"])] = row
    gaps = []
    for cycle_id in sorted(by_cycle, key=_natural_sort_key):
        modes = by_cycle[cycle_id]
        oracle = modes.get("oracle")
        fair = modes.get("fair")
        if oracle is None or fair is None:
            continue
        oracle_rate = float(oracle["refutation_rate"])
        fair_rate = float(fair["refutation_rate"])
        gaps.append(
            {
                "cycle_id": cycle_id,
                "oracle_refutation_rate": oracle_rate,
                "fair_refutation_rate": fair_rate,
                "oracle_minus_fair_refutation_rate": oracle_rate - fair_rate,
                "oracle_sampled_win_count": oracle["sampled_win_count"],
                "fair_sampled_win_count": fair["sampled_win_count"],
            }
        )
    return tuple(gaps)


def _validate_unique_cycle_modes(rows: tuple[Mapping[str, Any], ...]) -> None:
    seen: set[tuple[str, str]] = set()
    for row in rows:
        key = (str(row["cycle_id"]), str(row["mode"]))
        if key in seen:
            raise ValueError(f"duplicate refutation report for cycle_id={key[0]!r}, mode={key[1]!r}")
        seen.add(key)


def _natural_sort_key(value: str) -> tuple[tuple[int, Any], ...]:
    parts = re.split(r"([0-9]+)", value)
    return tuple(
        (0, int(part)) if part.isdigit() else (1, part)
        for part in parts
        if part != ""
    )


def _linear_slope(values: list[float]) -> float | None:
    if len(values) < 2:
        return None
    n = len(values)
    xs = list(range(n))
    x_mean = sum(xs) / n
    y_mean = sum(values) / n
    denominator = sum((x - x_mean) ** 2 for x in xs)
    if denominator <= 0.0:
        return None
    return sum((x - x_mean) * (y - y_mean) for x, y in zip(xs, values)) / denominator


def _mapping(value: Any, *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be an object")
    return value


def _int(value: Any, *, label: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be an integer")
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be an integer") from exc


def _float(value: Any, *, label: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be a number") from exc
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite")
    return result


def _str(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} must be a non-empty string")
    return value
