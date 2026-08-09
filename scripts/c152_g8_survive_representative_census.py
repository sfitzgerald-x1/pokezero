#!/usr/bin/env python
"""C152: size G8's remaining arm -- the collapsed SURVIVE representative, off-fan.

Ledger row G8 ends: *"Both remainders are unmeasured, and the final holdout was
not swept."* This census measures the first remainder. It is the one the second
instance `19200244/115` exhibits and the one C149's band split cannot reach,
because the split lives inside the `Some(residual_bands)` arm and this value is
what the partition uses when that arm is absent.

**The two quantities, and why they are different.**

The engine prices the non-KO arm of a damage fan with
``compare_health_with_damage_multiples(max_damage, health).0``, an average over
an **f32 accumulator**::

    damage = max * 0.85;  repeated 16 times:  take `damage as i16`, damage += max * 0.01

Showdown throws from the **exact integer fan** ``floor(max * r / 100)`` for
``r`` in 85..=100. The engine's own C149 patch says so in as many words -- it
takes its per-roll arm values from the integer fan precisely so that "no arm is
ever priced at a damage Showdown cannot deal". The survive representative is an
**average**, so it is under no obligation to be a fan member at all, and when it
is not, the arm prices **zero** of the rolls in its own band rather than one.

That is the whole of G8's remaining mechanism, stated without reference to any
particular boundary. This script measures how much of the ``(max_damage,
health)`` plane it covers.

**Validation before use, not trust.** The census re-derives the one boundary
where this shape is exhibited in the record, `19200244/115`
(`reports/artifacts/c141_final_holdout_sweep.json`, replayed under C152's
instrumented build): ``max_damage = 159``, ``health = 157`` must give a 14-roll
survive band, a survive representative of **145**, and 145 **not** a member of
the integer fan. If that check fails the script exits non-zero and writes
nothing, because a census whose model does not reproduce the one case it is
about is measuring something else.

Usage::

    python scripts/c152_g8_survive_representative_census.py \\
        --json reports/artifacts/c152_g8_survive_representative_census.json
"""

from __future__ import annotations

import argparse
import json
import struct
from pathlib import Path
from typing import Any

MIN_MAX_DAMAGE = 10
MAX_MAX_DAMAGE = 600


def _f32(value: float) -> float:
    """Round a Python f64 to the nearest f32, the way Rust `f32` arithmetic does.

    Standard library only -- `struct` pack/unpack through `'f'` IS the f32
    rounding. Doing this at every step matters: the engine accumulates in `f32`,
    and an f64 accumulation drifts off it for large `max_damage`.
    """
    return struct.unpack("f", struct.pack("f", value))[0]


def f32_accumulator_rolls(max_damage: int) -> list[int]:
    """The engine's 16 f32-accumulator roll values, `damage as i16` each step.

    Mirrors `compare_health_with_damage_multiples`
    (`gen3/generate_instructions.rs`) exactly, including that the truncation to
    `i16` happens on the accumulator's CURRENT value and does not feed back.
    """
    increment = _f32(_f32(max_damage) * _f32(0.01))
    damage = _f32(_f32(max_damage) * _f32(0.85))
    out: list[int] = []
    for _ in range(16):
        out.append(int(damage))  # `damage as i16` truncates toward zero
        damage = _f32(damage + increment)
    return out


def integer_fan(max_damage: int) -> list[int]:
    """What Showdown can actually throw: `floor(max * r / 100)`, r in 85..=100."""
    return [(max_damage * r) // 100 for r in range(85, 101)]


def survive_representative(max_damage: int, health: int) -> tuple[int, int] | None:
    """`(representative, band_size)`, or None when no roll survives.

    `total_less_than / num_less_than` with integer division, exactly as the Rust
    does -- the accumulator values are already truncated to i16 before summing.
    """
    rolls = f32_accumulator_rolls(max_damage)
    band = [d for d in rolls if d < health]
    if not band:
        return None
    return sum(band) // len(band), len(band)


def census() -> dict[str, Any]:
    windows = 0
    with_band = 0
    off_fan = 0
    off_fan_and_in_band_range = 0
    on_fan = 0
    prices_zero_rolls = 0
    worst: list[dict[str, int]] = []

    for max_damage in range(MIN_MAX_DAMAGE, MAX_MAX_DAMAGE + 1):
        fan = set(integer_fan(max_damage))
        rolls = f32_accumulator_rolls(max_damage)
        lo, hi = min(rolls), max(rolls)
        for health in range(1, hi + 2):
            windows += 1
            result = survive_representative(max_damage, health)
            if result is None:
                continue
            with_band += 1
            representative, band_size = result
            band = [d for d in rolls if d < health]
            if representative in fan:
                on_fan += 1
                # It is a fan member; it prices one roll iff that roll is also in
                # the band it represents (it always is -- the representative of a
                # set of values all below `health` is itself below `health`).
                continue
            off_fan += 1
            if min(band) <= representative <= max(band):
                off_fan_and_in_band_range += 1
            # An off-fan representative is a damage Showdown cannot deal, so the
            # arm reproduces NO roll of its own band.
            prices_zero_rolls += 1
            if band_size >= 8 and len(worst) < 40:
                worst.append(
                    {
                        "max_damage": max_damage,
                        "health": health,
                        "band_size": band_size,
                        "representative": representative,
                    }
                )
        del lo

    return {
        "max_damage_range": [MIN_MAX_DAMAGE, MAX_MAX_DAMAGE],
        "windows_examined": windows,
        "windows_with_a_survive_band": with_band,
        "representative_is_a_fan_member": on_fan,
        "representative_is_OFF_fan": off_fan,
        "off_fan_but_inside_the_band_range": off_fan_and_in_band_range,
        "arms_pricing_zero_achievable_rolls": prices_zero_rolls,
        "off_fan_fraction_of_bands": round(off_fan / with_band, 6) if with_band else None,
        "closure_identity_on_fan_plus_off_fan_equals_bands": on_fan + off_fan == with_band,
        "sample_off_fan_windows": worst,
    }


def validate() -> dict[str, Any]:
    """Reproduce `19200244/115` before measuring anything else."""
    rolls = f32_accumulator_rolls(159)
    band = [d for d in rolls if d < 157]
    representative, band_size = survive_representative(159, 157)  # type: ignore[misc]
    fan = integer_fan(159)
    checks = {
        "f32_accumulator_rolls": rolls,
        "integer_fan": fan,
        "band_size": band_size,
        "band_size_is_14": band_size == 14,
        "survive_representative": representative,
        "survive_representative_is_145": representative == 145,
        "representative_is_off_fan": representative not in set(fan),
        "band_minimum_is_135": min(band) == 135,
    }
    checks["ALL_PASS"] = bool(
        checks["band_size_is_14"]
        and checks["survive_representative_is_145"]
        and checks["representative_is_off_fan"]
        and checks["band_minimum_is_135"]
    )
    return checks


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--json", required=True)
    args = parser.parse_args(argv)

    validation = validate()
    if not validation["ALL_PASS"]:
        print("VALIDATION FAILED -- the model does not reproduce 19200244/115:")
        print(json.dumps(validation, indent=2))
        return 2

    out = {
        "schema": "c152-g8-survive-representative-census/1",
        "what": (
            "How often the collapsed partition's SURVIVE representative -- "
            "compare_health_with_damage_multiples(max, health).0, an average over the "
            "f32 accumulator -- is not a member of the exact integer fan "
            "floor(max * r / 100) that Showdown throws from. An off-fan representative "
            "prices ZERO rolls of its own band rather than one. This is G8's first "
            "unmeasured remainder and the shape 19200244/115 exhibits."
        ),
        "why_c149_cannot_reach_it": (
            "The C149 per-roll split lives inside the `Some(residual_bands)` arm of "
            "`residual_disjoint_bands`. The survive representative is what the "
            "partition uses when that arm is ABSENT, so no change confined to the "
            "split's two `i16::MAX`-ceiling call sites can touch it."
        ),
        "validation_against_19200244_115": validation,
        "census": census(),
    }
    Path(args.json).write_text(json.dumps(out, indent=2, sort_keys=True) + "\n")
    print(json.dumps(out["census"], indent=2, sort_keys=True))
    print(f"-> {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
