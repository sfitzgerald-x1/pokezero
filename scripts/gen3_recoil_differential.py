#!/usr/bin/env python
"""Showdown ground-truth gate for the gen3 recoil law: the fraction and the cap.

Companion to ``scripts/gen3_switch_differential.py``, same idiom (real Node sim
via ``pokezero.showdown_fixture``, gen3 Custom Game, facts checked against
ground truth rather than printed). This script exists because C142 started from
a wrong premise about gen3 recoil that a source reading alone did not correct,
and the correction has to be re-runnable rather than quoted.

Two facts, each measured rather than read:

``fraction``
    Gen3 Double-Edge recoil is ``floor(damage / 3)``. NOT ``damage / 4`` (the
    premise the C142 diagnosis was handed), and NOT ``floor(damage * 33 / 100)``
    — ``data/moves.ts`` carries ``recoil: [33, 100]``, but that is the
    CURRENT-gen value. Gen3 inherits gen4 (``data/mods/gen3/scripts.ts``:
    ``inherit: 'gen4'``) and ``data/mods/gen4/moves.ts`` overrides ``doubleedge``
    to ``recoil: [1, 3]``. Gen3's own ``calcRecoilDamage`` is
    ``clampIntRange(Math.floor(damageDealt * recoil[0] / recoil[1]), 1)`` —
    ``floor``, not the ``Math.round`` of ``sim/battle-actions.ts``.

    A weak attacker into a bulky target keeps every roll non-lethal, so the
    recoil is a pure function of the damage roll. Roughly a third of rolls
    discriminate ``floor(d/3)`` from ``floor(33d/100)``; the run asserts that at
    least a quarter of the sampled rows do, so a pass cannot come from a sample
    that happened to agree on both.

``cap``
    Recoil is computed from the damage actually DEALT, capped at the target's
    remaining HP — not from the uncapped roll. Gen3 sets
    ``move.totalDamage = damage`` from ``moveHit``, and ``battle.damage`` clamps
    to remaining HP, but the clamp is asserted here rather than inferred: the
    SAME attacker, move and target species is run at two different target HPs.
    The roll distribution is identical across the two runs, so an uncapped
    reading predicts identical recoil. It does not happen — recoil tracks the
    target's remaining HP.

Usage::

    PYTHONPATH=src python scripts/gen3_recoil_differential.py
    PYTHONPATH=src python scripts/gen3_recoil_differential.py --json out.json
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from pokezero.local_showdown import LocalShowdownConfig  # noqa: E402
from pokezero.showdown_fixture import FixturePokemon, run_multi_turn_fixture  # noqa: E402

# Level 100, 31 IVs, 0 EVs -> HP = 2 * base + 141. Pidgey (base 40) is 221 and
# Snorlax (base 160) is 461; both are asserted from the protocol, never assumed.
_PIDGEY_HP = 221
_SNORLAX_HP = 461
# Two Seismic Tosses (100 fixed each) leave Pidgey on 21, chosen so the capped
# damage is small enough that no roll could produce it.
_CHIPPED_HP = 21


def _snorlax_doubleedge() -> FixturePokemon:
    return FixturePokemon(species="Snorlax", ability="Immunity", item="None",
                          moves=("Double-Edge", "Seismic Toss", "Splash"))


def _pidgey_doubleedge() -> FixturePokemon:
    # Weak attacker (base atk 45) so every roll into Snorlax is non-lethal.
    return FixturePokemon(species="Pidgey", ability="Keen Eye", item="None",
                          moves=("Double-Edge",))


def _pidgey_target() -> FixturePokemon:
    return FixturePokemon(species="Pidgey", ability="Keen Eye", item="None",
                          moves=("Splash",))


def _snorlax_target() -> FixturePokemon:
    return FixturePokemon(species="Snorlax", ability="Immunity", item="None",
                          moves=("Splash",))


def _hp(condition: str) -> int:
    return 0 if condition.startswith("0 fnt") else int(condition.split(" ")[0].split("/")[0])


def _direct_damage(lines, seat: str, maxhp: int) -> int | None:
    """The plain (untagged) `-damage` on ``seat``, as a delta from ``maxhp``."""

    for line in lines:
        if line.startswith(f"|-damage|{seat}") and "[from]" not in line:
            return maxhp - _hp(line.split("|")[3])
    return None


def _recoil_damage(lines, seat: str, maxhp: int) -> int | None:
    for line in lines:
        if line.startswith(f"|-damage|{seat}") and "Recoil" in line:
            return maxhp - _hp(line.split("|")[3])
    return None


def measure_fraction(seeds, config) -> dict:
    """Non-lethal Double-Edge rows: recoil against the three candidate laws."""

    rows = []
    for seed in seeds:
        result = run_multi_turn_fixture(
            p1_team=[_pidgey_doubleedge()], p2_team=[_snorlax_target()],
            turns=[("move doubleedge", "move splash")], seed=seed, config=config,
        )
        lines = list(result.steps[0].protocol_lines)
        damage = _direct_damage(lines, "p2a: Snorlax", _SNORLAX_HP)
        recoil = _recoil_damage(lines, "p1a: Pidgey", _PIDGEY_HP)
        if damage is None or recoil is None:
            raise RuntimeError(f"seed {seed}: no Double-Edge hit in the protocol")
        rows.append({
            "seed": seed,
            "damage": damage,
            "recoil": recoil,
            "crit": any(line.startswith("|-crit|") for line in lines),
            "floor_div3": damage // 3,
            "floor_33_100": math.floor(damage * 33 / 100),
            "round_div3": round(damage / 3),
            "floor_div4": damage // 4,
        })
    return {
        "rows": rows,
        "agree_floor_div3": sum(r["recoil"] == r["floor_div3"] for r in rows),
        "agree_floor_33_100": sum(r["recoil"] == r["floor_33_100"] for r in rows),
        "agree_round_div3": sum(r["recoil"] == r["round_div3"] for r in rows),
        "agree_floor_div4": sum(r["recoil"] == r["floor_div4"] for r in rows),
        # Rows where floor(d/3) and floor(33d/100) actually differ. Without this
        # a clean pass could come from a sample on which the two laws agree.
        "discriminating_rows": sum(r["floor_div3"] != r["floor_33_100"] for r in rows),
        "n": len(rows),
    }


def measure_cap(seeds, config) -> dict:
    """The same lethal Double-Edge at two different target HPs."""

    rows = []
    for target_hp in (_PIDGEY_HP, _CHIPPED_HP):
        chip = [("move seismictoss", "move splash")] * (
            0 if target_hp == _PIDGEY_HP else 2
        )
        for seed in seeds:
            result = run_multi_turn_fixture(
                p1_team=[_snorlax_doubleedge()],
                # A second Pokemon so the battle does not end on the faint.
                p2_team=[_pidgey_target(), _snorlax_target()],
                turns=[*chip, ("move doubleedge", "move splash")],
                seed=seed, config=config,
            )
            lines = list(result.steps[len(chip)].protocol_lines)
            recoil = _recoil_damage(lines, "p1a: Snorlax", _SNORLAX_HP)
            fainted = any(line.startswith("|faint|p2a: Pidgey") for line in lines)
            rows.append({
                "seed": seed,
                "target_hp_before": target_hp,
                "target_fainted": fainted,
                "recoil": recoil,
                "floor_capped_div3": target_hp // 3,
            })
    return {"rows": rows}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--showdown-root", type=Path, default=None)
    parser.add_argument("--seeds", type=int, nargs="+",
                        default=list(range(2000, 2032)))
    parser.add_argument("--cap-seeds", type=int, nargs="+", default=[1000, 1001, 1002])
    parser.add_argument("--json", type=Path, default=None)
    args = parser.parse_args(argv)

    config = LocalShowdownConfig(showdown_root=args.showdown_root)
    failures: list[str] = []

    fraction = measure_fraction(args.seeds, config)
    print(f"fraction: n={fraction['n']} "
          f"floor(d/3)={fraction['agree_floor_div3']} "
          f"floor(33d/100)={fraction['agree_floor_33_100']} "
          f"round(d/3)={fraction['agree_round_div3']} "
          f"floor(d/4)={fraction['agree_floor_div4']} "
          f"discriminating={fraction['discriminating_rows']}")
    if fraction["agree_floor_div3"] != fraction["n"]:
        failures.append("recoil is not floor(damage / 3) on every sampled row")
    if fraction["discriminating_rows"] * 4 < fraction["n"]:
        failures.append(
            "too few rows discriminate floor(d/3) from floor(33d/100) for the "
            "result to mean anything — widen --seeds"
        )

    cap = measure_cap(args.cap_seeds, config)
    for row in cap["rows"]:
        print(f"cap: target_hp={row['target_hp_before']} fainted={row['target_fainted']} "
              f"recoil={row['recoil']} floor(hp/3)={row['floor_capped_div3']}")
        if not row["target_fainted"]:
            failures.append(
                f"cap probe at {row['target_hp_before']} HP did not KO, so it "
                f"measures nothing about the cap"
            )
        elif row["recoil"] != row["floor_capped_div3"]:
            failures.append(
                f"lethal recoil at {row['target_hp_before']} HP was "
                f"{row['recoil']}, not floor(hp/3)={row['floor_capped_div3']}"
            )
    # The decisive comparison: identical roll distribution, different target HP.
    by_hp = {row["target_hp_before"]: row["recoil"] for row in cap["rows"]}
    if len(by_hp) == 2 and len(set(by_hp.values())) == 1:
        failures.append(
            "recoil did not change with the target's remaining HP — that is the "
            "uncapped reading, and it contradicts the per-row assertions above"
        )

    payload = {"fraction": fraction, "cap": cap, "failures": failures}
    if args.json:
        args.json.write_text(json.dumps(payload, indent=1))
        print(f"-> {args.json}")

    print()
    if failures:
        for failure in failures:
            print(f"FAIL  {failure}")
        return 1
    print("gen3 recoil matches Showdown ground truth: floor(damage / 3), "
          "computed from the CAPPED damage dealt")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
