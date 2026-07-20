"""Merge deterministic coverage-audit shard ledgers into one completion verdict."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Iterable

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pokezero.coverage_enumeration_audit import merge_coverage_ledgers  # noqa: E402


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, nargs="+", required=True, help="Per-shard coverage ledgers.")
    parser.add_argument("--output", type=Path, required=True, help="Write the merged ledger here.")
    args = parser.parse_args(list(argv) if argv is not None else None)

    ledgers = [json.loads(path.read_text(encoding="utf-8")) for path in args.input]
    merged = merge_coverage_ledgers(ledgers)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(merged, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        "coverage ledger merge: "
        f"shards={len(ledgers)} games={merged['games_selected']}/{merged['games_total']} "
        f"coverage_complete={merged['complete']}"
    )
    return 0 if merged["complete"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
