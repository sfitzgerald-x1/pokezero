"""Build a read-only diversity coverage-rate report from dashboard snapshots.

Inputs are explicit ``GAMES=dashboard.json`` pairs so milestone ordering is
durable and does not depend on filename parsing. The output reports dashboard
coverage signal deltas normalized per 100k games.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

from pokezero.diversity_population import diversity_coverage_rate_report


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _parse_dashboard_arg(value: str) -> dict[str, Any]:
    raw_games, separator, raw_path = value.partition("=")
    if not separator or not raw_games.strip() or not raw_path.strip():
        raise ValueError("--dashboard must use GAMES=/path/to/dashboard.json")
    try:
        games = int(raw_games.replace("_", ""))
    except ValueError as exc:
        raise ValueError("--dashboard GAMES must be an integer") from exc
    if games < 0:
        raise ValueError("--dashboard GAMES must be non-negative")
    path = Path(raw_path).expanduser()
    dashboard = _read_json(path)
    if not isinstance(dashboard, dict):
        raise ValueError(f"{path} must contain a diversity dashboard JSON object")
    return {
        "games": games,
        "label": path.stem,
        "dashboard": dashboard,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dashboard",
        action="append",
        required=True,
        help="Milestone dashboard as GAMES=/path/to/dashboard.json; repeatable.",
    )
    parser.add_argument("--out", type=Path, help="write coverage-rate JSON here instead of stdout")
    args = parser.parse_args(argv)

    try:
        payload = diversity_coverage_rate_report(
            [_parse_dashboard_arg(value) for value in args.dashboard]
        )
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    text = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text, encoding="utf-8")
        print(f"[diversity-coverage-rate] wrote {args.out}", file=sys.stderr)
    else:
        print(text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
