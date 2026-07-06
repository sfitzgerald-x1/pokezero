"""Aggregate hazard_probe.py milestone outputs into the diversity-tier gate metrics.

This is a read-only measurement tool. It computes the `correct_pricing` scalar
from docs/diversity_tier_design.md:

    (value_opp_hazard_response - value_self_hazard_response) / value_spread

and reports whether the current trajectory satisfies the measurement-only gate:
positive pricing at the configured level for the last two valid milestones,
a positive slope trend across at least five valid milestones, and non-zero Rapid
Spin response in the same last-two milestone window as behavioral corroboration.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

from pokezero.hazard_metrics import aggregate_hazard_rows


_HAZARD_MILESTONE_RE = re.compile(r"^hazard-(\d+)\.json$")


def _milestone_games_from_path(path: Path) -> int | None:
    # Keep this in sync with scripts/milestone_probes.sh hazard_out.
    match = _HAZARD_MILESTONE_RE.match(path.name)
    if match is None:
        return None
    return int(match.group(1))


def _has_explicit_milestone_games(row: Mapping[str, Any]) -> bool:
    for key in ("milestone_games", "games_at", "games"):
        value = row.get(key)
        if isinstance(value, int) and value >= 0:
            return True
    return False


def _load_rows(path: Path) -> list[Mapping[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, Mapping) and isinstance(payload.get("checkpoints"), list):
        rows = payload["checkpoints"]
    elif isinstance(payload, list):
        rows = payload
    else:
        raise ValueError(f"{path} must be a hazard_probe JSON object with checkpoints or a list of rows")
    milestone_games = _milestone_games_from_path(path)
    loaded: list[Mapping[str, Any]] = []
    for row in rows:
        if not isinstance(row, Mapping):
            continue
        if milestone_games is not None and not _has_explicit_milestone_games(row):
            row = {**row, "milestone_games": milestone_games}
        loaded.append(row)
    return loaded


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
