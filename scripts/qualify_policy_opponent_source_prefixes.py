#!/usr/bin/env python3
"""Read-only frozen-source preflight. JSON stdout is not a strength receipt."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pokezero.mcts_eval.policy_opponent_roster import RosterError
from pokezero.mcts_eval.policy_opponent_source_prefixes import qualify_source_prefixes


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--roster", required=True)
    parser.add_argument("--roster-sha256", required=True)
    parser.add_argument("--source-root", required=True)
    args = parser.parse_args()
    try:
        result = qualify_source_prefixes(args.roster, expected_roster_sha256=args.roster_sha256,
                                        source_root=args.source_root)
    except (RosterError, OSError, ValueError) as error:
        print(f"SOURCE PREFLIGHT REFUSED: {error}", file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
