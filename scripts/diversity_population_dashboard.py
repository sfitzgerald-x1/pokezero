"""Build a read-only diversity population dashboard from probe artifacts.

This script consumes existing behavior-probe JSON plus optional diversity-pool
ledger payoff vectors. It does not launch games and does not feed any signal
back into training, admission, eviction, or matchmaking.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

from pokezero.diversity_population import diversity_population_dashboard


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _behavior_rows_from_payload(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        return [dict(row) for row in payload if isinstance(row, dict)]
    if isinstance(payload, dict):
        for key in ("checkpoints", "rows", "agents"):
            rows = payload.get(key)
            if isinstance(rows, list):
                return [dict(row) for row in rows if isinstance(row, dict)]
        return [dict(payload)]
    raise ValueError("behavior input must be a JSON object or list")


def _load_behavior_rows(paths: list[Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in paths:
        rows.extend(_behavior_rows_from_payload(_read_json(path)))
    return rows


def _policy_rows_from_payload(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        return [dict(row) for row in payload if isinstance(row, dict)]
    if isinstance(payload, dict):
        for key in ("policies", "checkpoints", "rows", "agents"):
            rows = payload.get(key)
            if isinstance(rows, list):
                return [dict(row) for row in rows if isinstance(row, dict)]
        return [dict(payload)]
    raise ValueError("policy-priors input must be a JSON object or list")


def _load_policy_rows(paths: list[Path] | None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in paths or ():
        rows.extend(_policy_rows_from_payload(_read_json(path)))
    return rows


def _load_payoff_vectors(path: Path | None) -> dict[str, dict[str, Any]]:
    if path is None:
        return {}
    payload = _read_json(path)
    if not isinstance(payload, dict):
        raise ValueError("--pool-ledger must point to a JSON object")
    raw_vectors = payload.get("payoff_vectors", payload)
    if not isinstance(raw_vectors, dict):
        raise ValueError("--pool-ledger must contain a payoff_vectors object")
    return {
        str(member_id): dict(vector)
        for member_id, vector in raw_vectors.items()
        if isinstance(vector, dict)
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--behavior", type=Path, action="append", required=True, help="behavior_probe JSON output; repeatable")
    parser.add_argument(
        "--policy-priors",
        type=Path,
        action="append",
        help="optional policy_js_divergence_probe JSON output; repeatable",
    )
    parser.add_argument("--pool-ledger", type=Path, help="optional diversity_pool.py ledger containing payoff_vectors")
    parser.add_argument("--out", type=Path, help="write dashboard JSON here instead of stdout")
    parser.add_argument("--threshold-move-rate", type=float, default=0.05)
    parser.add_argument("--threshold-pivot-rate", type=float, default=0.05)
    parser.add_argument("--threshold-avg-turns", type=float, default=5.0)
    parser.add_argument("--threshold-distinct-moves", type=float, default=3.0)
    parser.add_argument("--threshold-behavior-cluster-distance", type=float, default=0.20)
    args = parser.parse_args(argv)

    thresholds = {
        "move_rate": args.threshold_move_rate,
        "pivot_rate": args.threshold_pivot_rate,
        "avg_turns": args.threshold_avg_turns,
        "distinct_moves": args.threshold_distinct_moves,
        "behavior_cluster_distance": args.threshold_behavior_cluster_distance,
    }
    payload = diversity_population_dashboard(
        _load_behavior_rows(args.behavior),
        payoff_vectors=_load_payoff_vectors(args.pool_ledger),
        policy_prior_rows=_load_policy_rows(args.policy_priors),
        thresholds=thresholds,
    )
    text = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text, encoding="utf-8")
        print(f"[diversity-population] wrote {args.out}", file=sys.stderr)
    else:
        print(text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
