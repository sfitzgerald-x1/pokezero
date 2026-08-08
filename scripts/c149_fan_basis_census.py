#!/usr/bin/env python
"""How often do the engine's TWO damage-fan bases disagree about a band's size?

Ledger G8 / C149. The gen 3 partition has two different models of the same
16-roll fan:

  * ``compare_health_with_damage_multiples`` walks an f32 accumulator --
    ``max * 0.85`` stepped by ``max * 0.01`` -- and counts how many of the sixteen
    land at or above a threshold. That count is what PRICES a residual band.
  * ``push_enumerated_rolls`` (and, since this change,
    ``residual_band_roll_fan``) uses exact integer arithmetic,
    ``floor(max * r / 100)`` for ``r`` in ``85..=100``. That is what Showdown
    actually rolls.

The f32 accumulator drifts (C116 M5), so the two can disagree about how many rolls
a band holds. The band split declines to split whenever they do, keeping the single
collapsed arm -- which is what makes the split a strict improvement or a no-op on
any given band rather than a trade. This script measures how large that fallback
region is, so the claim in the patch comment is reproducible rather than asserted.

Scope of the census, stated because it is not "all": ``max_damage`` in
``10..=600`` and, for each, every integer threshold strictly above the f32 fan's
floor and at most ``max_damage``. That rectangle and nothing wider.

Usage::

    python scripts/c149_fan_basis_census.py --json out.json
"""

from __future__ import annotations

import argparse
import json
import struct
import sys
from pathlib import Path

MAX_DAMAGE_LOW = 10
MAX_DAMAGE_HIGH = 600


def f32(value: float) -> float:
    """Round to the nearest binary32, so Python reproduces Rust's `f32`."""

    return struct.unpack("f", struct.pack("f", value))[0]


def f32_rolls_at_or_above(max_damage: int, threshold: int) -> int:
    """`compare_health_with_damage_multiples`'s `num_greater_than`, transcribed."""

    increment = f32(f32(max_damage) * f32(0.01))
    damage = f32(f32(max_damage) * f32(0.85))
    count = 0
    for _ in range(16):
        if not damage < f32(threshold):
            count += 1
        damage = f32(damage + increment)
    return count


def integer_rolls_at_or_above(max_damage: int, threshold: int) -> int:
    """`residual_band_roll_fan`'s `floor(max * r / 100)`, transcribed."""

    return sum(1 for r in range(85, 101) if (max_damage * r) // 100 >= threshold)


def census() -> dict:
    windows = 0
    disagreements = 0
    affected: set[int] = set()
    examples: list[dict] = []
    for max_damage in range(MAX_DAMAGE_LOW, MAX_DAMAGE_HIGH + 1):
        floor = int(f32(f32(max_damage) * f32(0.85)))
        for threshold in range(floor + 1, max_damage + 1):
            windows += 1
            f32_count = f32_rolls_at_or_above(max_damage, threshold)
            int_count = integer_rolls_at_or_above(max_damage, threshold)
            if f32_count != int_count:
                disagreements += 1
                affected.add(max_damage)
                if len(examples) < 10:
                    examples.append(
                        {
                            "max_damage": max_damage,
                            "threshold": threshold,
                            "f32_count": f32_count,
                            "integer_count": int_count,
                        }
                    )
    return {
        "scope": {
            "max_damage_range": [MAX_DAMAGE_LOW, MAX_DAMAGE_HIGH],
            "thresholds": "every integer strictly above the f32 fan floor, up to max_damage",
        },
        "windows_examined": windows,
        "count_disagreements": disagreements,
        "disagreement_fraction_percent": round(100.0 * disagreements / windows, 3),
        "distinct_max_damage_values_examined": MAX_DAMAGE_HIGH - MAX_DAMAGE_LOW + 1,
        "distinct_max_damage_values_affected": len(affected),
        "examples": examples,
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", help="write the census here")
    args = parser.parse_args(argv)
    report = census()
    print(json.dumps(report, indent=2))
    if args.json:
        Path(args.json).write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"-> {args.json}", file=sys.stderr)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
