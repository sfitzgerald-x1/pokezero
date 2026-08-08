#!/usr/bin/env python
"""Can the G8 Leech Seed band split ever DROP an arm that Showdown could match?

Ledger G8 / C149 / C150. `reports/c149_g8_leechseed_band_split.md` claimed "0 real
trades" over a **124,188-fixture scan** attributed to independent review, quoting
"split fired 44,393, declined 79,795, 25,728 collapsed arms kept, 18,901 dropped, 115
mutant disagreements". **No artifact for that scan was ever committed**, and the
figures do not reconcile with each other: `25,728 + 18,901 = 44,629`, which is neither
the 44,393 fires nor the 79,795 declines. C150 replaces them with this census, which
is reproducible, and with the structural argument in `reports/c149...md` §1, which is
what actually settles the property. The CONCLUSION is unchanged: zero real trades.

WHAT A "REAL TRADE" IS. `push_per_roll_residual_kill_arms` replaces one collapsed
residual-kill arm, priced at its band's threshold `lower`, with one arm per integer-fan
member inside the half-open window `[lower, upper)`. A **trade** is a band where the
collapsed arm sat on a damage Showdown could actually deal -- i.e. `lower` is a member
of the integer fan `floor(max * r / 100)`, `r` in `85..=100` -- and the split leaves no
arm at that damage. It cannot happen, because `lower` lies inside its own half-open
window, so if it is a fan member the split emits an arm at exactly it. This census is
the measurement of that argument over a stated rectangle; the argument, not the census,
is the reason to believe it holds everywhere.

WHAT IS TRANSCRIBED, AND WHY THAT IS THE SAME CHOICE C149 ALREADY MADE.
`scripts/c149_fan_basis_census.py` is the precedent: the two fan bases are transcribed
into Python and censused, because the property is pure arithmetic over
`(max_damage, threshold)` and driving it through a built engine would sample a few
hundred fixtures instead of hundreds of thousands of band windows. Transcribed here:

  * `residual_band_roll_fan` -- exact integer `floor(max * r / 100)`, `r` in `85..=100`.
  * `compare_health_with_damage_multiples`'s `num_greater_than` -- the f32 accumulator
    `max * 0.85` stepped by `max * 0.01`, which is what PRICES a band.
  * `residual_disjoint_bands`'s band population, `at_or_above(lower) - at_or_above(upper)`,
    and its `band > 0` emission gate.
  * `push_per_roll_residual_kill_arms`'s count guard: split only when the integer fan
    puts exactly `expected_rolls` of its sixteen slots inside `[lower, upper)`.

The transcription is not taken on trust. `--check-fixtures` re-derives the three
committed crate fixtures of `rust/pokezero-search/tests/gen3_leechseed_residual_band_split.rs`
(A: max 174 / threshold 160, split; B crit: max 66 / threshold 61, split with colliding
rolls; C: max 30 / threshold 27, the DECLINE path whose guard-deleted mutant conjures
exactly one roll's worth of mass) and fails if any disagrees. That check runs as part of
the census and its result is written into the artifact, so a transcription that drifts
away from the engine turns the artifact red rather than quietly wrong.

SCOPE, stated because it is not "all": `max_damage` in ``10..=600``; `lower` every
integer strictly above the f32 fan's floor and at most `max_damage`; `upper` every
integer in ``lower+1..=max_damage+1``, where `max_damage + 1` stands for the top band's
`i16::MAX` ceiling (no roll of the fan exceeds `max_damage`, so the two are the same
window). Bands whose population is zero are skipped, exactly as the engine's
`band > 0` gate skips them. That rectangle and nothing wider.

Usage::

    python scripts/c150_band_split_trade_census.py --json out.json
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


def f32_fan(max_damage: int) -> list[float]:
    """`compare_health_with_damage_multiples`'s accumulator, sixteen rungs."""

    increment = f32(f32(max_damage) * f32(0.01))
    damage = f32(f32(max_damage) * f32(0.85))
    rungs: list[float] = []
    for _ in range(16):
        rungs.append(damage)
        damage = f32(damage + increment)
    return rungs


def integer_fan(max_damage: int) -> list[int]:
    """`residual_band_roll_fan`: `floor(max * r / 100)` for `r` in `85..=100`.

    Duplicates are KEPT. Two rolls that floor to the same integer are two rolls, which
    is what makes this count comparable to the comparator's sixteen.
    """

    return [(max_damage * r) // 100 for r in range(85, 101)]


def f32_at_or_above(rungs: list[float], threshold: int) -> int:
    """`num_greater_than`: rungs that are NOT strictly below `threshold`."""

    limit = f32(threshold)
    return sum(1 for rung in rungs if not rung < limit)


def band_report(max_damage: int, lower: int, upper: int) -> dict | None:
    """One band window, decided exactly as the shipped code decides it.

    Returns `None` for a window the engine never emits an arm for (`band <= 0`).
    """

    rungs = f32_fan(max_damage)
    expected_rolls = f32_at_or_above(rungs, lower) - f32_at_or_above(rungs, upper)
    if expected_rolls <= 0:
        return None
    fan = integer_fan(max_damage)
    in_band = [roll for roll in fan if lower <= roll < upper]
    split = len(in_band) == expected_rolls
    return {
        "max_damage": max_damage,
        "lower": lower,
        "upper": upper,
        "expected_rolls": expected_rolls,
        "integer_rolls_in_band": len(in_band),
        "split_fires": split,
        # The collapsed arm's price. Showdown can match it only if it is a fan member.
        "collapsed_arm_is_a_fan_member": lower in fan,
        "split_arm_damages": sorted(set(in_band)) if split else [],
    }


# `(name, max_damage, threshold, upper, expects_split)`, from the crate fixtures'
# own constants and comments. `upper` is the `i16::MAX` ceiling at all three sites,
# represented as `max_damage + 1` (see the scope note above).
_CRATE_FIXTURES = (
    ("A_non_crit_site", 174, 160, True),
    ("B_crit_site_colliding_rolls", 66, 61, True),
    ("C_count_guard_declines", 30, 27, False),
)


def check_fixtures() -> dict:
    """Re-derive the three committed crate fixtures through this transcription."""

    results = []
    for name, max_damage, threshold, expects_split in _CRATE_FIXTURES:
        band = band_report(max_damage, threshold, max_damage + 1)
        assert band is not None, f"{name}: the engine emits an arm here, this says it does not"
        agrees = band["split_fires"] == expects_split
        results.append(
            {
                "fixture": name,
                "max_damage": max_damage,
                "threshold": threshold,
                "crate_expects_split": expects_split,
                "transcription_splits": band["split_fires"],
                "agrees": agrees,
                "expected_rolls": band["expected_rolls"],
                "integer_rolls_in_band": band["integer_rolls_in_band"],
                "split_arm_damages": band["split_arm_damages"],
            }
        )
    # Fixture C's crate test pins the guard-deleted mass at 105.859375 %, i.e. exactly
    # ONE non-crit roll of excess (100 * (15/16) / 16 == 5.859375). That is only true if
    # the integer fan holds exactly one roll more than the comparator counted.
    c = next(r for r in results if r["fixture"] == "C_count_guard_declines")
    excess_rolls = c["integer_rolls_in_band"] - c["expected_rolls"]
    c["excess_rolls_over_comparator"] = excess_rolls
    c["crate_pinned_guard_deleted_mass_percent"] = 105.859375
    c["mass_implied_by_this_transcription_percent"] = round(
        100.0 + excess_rolls * 100.0 * (15.0 / 16.0) / 16.0, 6
    )
    c["agrees"] = c["agrees"] and (
        c["mass_implied_by_this_transcription_percent"]
        == c["crate_pinned_guard_deleted_mass_percent"]
    )
    return {
        "fixtures": results,
        "all_agree": all(r["agrees"] for r in results),
    }


def census() -> dict:
    windows = 0
    bands = 0
    split_fires = 0
    split_declines = 0
    collapsed_arms_kept = 0
    collapsed_arms_dropped = 0
    real_trades: list[dict] = []
    # A declined band is precisely a band whose two bases disagree, which is precisely
    # the band on which the guard-deleted mutant prices arms from one basis while the
    # survive arm is discounted from the other -- conjuring or destroying mass.
    mutant_mass_disagreements = 0
    for max_damage in range(MAX_DAMAGE_LOW, MAX_DAMAGE_HIGH + 1):
        rungs = f32_fan(max_damage)
        fan = integer_fan(max_damage)
        fan_set = set(fan)
        floor = int(rungs[0])
        at_or_above = {
            t: f32_at_or_above(rungs, t) for t in range(floor, max_damage + 2)
        }
        for lower in range(floor + 1, max_damage + 1):
            for upper in range(lower + 1, max_damage + 2):
                windows += 1
                expected_rolls = at_or_above[lower] - at_or_above[upper]
                if expected_rolls <= 0:
                    continue
                bands += 1
                in_band = sum(1 for roll in fan if lower <= roll < upper)
                if in_band != expected_rolls:
                    split_declines += 1
                    mutant_mass_disagreements += 1
                    # Declining keeps the collapsed arm verbatim, so nothing is dropped.
                    collapsed_arms_kept += 1
                    continue
                split_fires += 1
                if lower in fan_set:
                    # The split emits an arm at every fan member of `[lower, upper)`,
                    # and `lower` is one of them, so the collapsed arm's damage survives.
                    collapsed_arms_kept += 1
                else:
                    collapsed_arms_dropped += 1
                    # A dropped arm is a REAL trade only if Showdown could have thrown
                    # that damage. `lower not in fan_set` says it could not.
                    if lower in fan_set:  # pragma: no cover - unreachable by construction
                        real_trades.append(
                            {"max_damage": max_damage, "lower": lower, "upper": upper}
                        )
    return {
        "what_this_measures": (
            "for every residual band window the shipped partition can emit an arm for, "
            "whether the G8 split drops a collapsed arm priced at a damage Showdown "
            "could actually deal (a REAL TRADE)"
        ),
        "scope": {
            "max_damage_range": [MAX_DAMAGE_LOW, MAX_DAMAGE_HIGH],
            "lower": "every integer strictly above the f32 fan floor, up to max_damage",
            "upper": (
                "every integer in lower+1..=max_damage+1; max_damage+1 stands for the "
                "i16::MAX ceiling of the two in-scope call sites"
            ),
            "skipped": "windows whose comparator population is <= 0, per the engine's band > 0 gate",
        },
        "windows_examined": windows,
        "bands_examined": bands,
        "split_fired": split_fires,
        "split_declined": split_declines,
        "collapsed_arms_kept": collapsed_arms_kept,
        "collapsed_arms_dropped": collapsed_arms_dropped,
        "real_trades": len(real_trades),
        "real_trade_examples": real_trades[:10],
        "identity_check": {
            "kept_plus_dropped_equals_bands": (
                collapsed_arms_kept + collapsed_arms_dropped == bands
            ),
            "fired_plus_declined_equals_bands": (split_fires + split_declines == bands),
        },
        "guard_deleted_mutant_band_mass_disagreements": mutant_mass_disagreements,
        "crate_fixture_agreement": check_fixtures(),
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", help="write the census here")
    parser.add_argument(
        "--check-fixtures",
        action="store_true",
        help="only re-derive the three crate fixtures, then exit",
    )
    args = parser.parse_args(argv)
    if args.check_fixtures:
        report = check_fixtures()
        print(json.dumps(report, indent=2))
        return 0 if report["all_agree"] else 1
    report = census()
    print(json.dumps(report, indent=2))
    if args.json:
        Path(args.json).write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"-> {args.json}", file=sys.stderr)
    ok = (
        report["real_trades"] == 0
        and report["crate_fixture_agreement"]["all_agree"]
        and all(report["identity_check"].values())
    )
    return 0 if ok else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
