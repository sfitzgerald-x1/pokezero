#!/usr/bin/env python
"""Pre-measure the crit-arm kill-split repair, offline, before any engine edit.

The gen3 damage brancher already partitions the KO threshold -- but only on one
of its two paths. In ``gen3/generate_instructions.rs``:

    if branch_on_damage
        && max_damage_dealt >= defender_active.hp
        && min_damage_dealt < defender_active.hp
    {
        let (average_non_kill_damage, num_kill_rolls) =
            compare_health_with_damage_multiples(max_damage_dealt, defender_active.hp);
        ...
        let branch_chance = ((1.0 - crit_rate) * (num_kill_rolls as f32 / 16.0)) + crit_rate;

That is CASE A: the non-crit roll range straddles the defender's HP. It splits
kill from non-kill with exact masses and an exact non-kill representative, and
it is correct. But it folds crit wholesale into ``branch_chance`` -- asserting
P(kill | crit) = 1 -- and the sibling path is:

    } else if branch_on_damage && fixed_damage.is_none()
        && max_damage_dealt < defender_active.hp
    {
        ...
        branch_damage = (max_crit_damage as f32 * 0.925) as i16;

CASE B: the non-crit roll can never kill. It splits crit from non-crit but
applies NO kill split to the crit arm -- the arm carries the AVERAGE crit roll.
When the crit roll range straddles the defender's HP, that average sits on one
side of the threshold and the other side is simply absent from the branch
support. This is the shape behind the Alakazam boundary (seed 17000001 step 1):
a 6.25% crit arm asserting a KO, while Showdown's crit rolled 205 of 209 and the
mon survived to take its Leftovers tick.

So the repair is to apply the EXISTING identity to the crit arm, not to build a
second partitioner beside it.

This script measures the repair before it is written, from retained rows the
patch cannot see. It reports:

  (a) how often each damage-branch case occurs;
  (b) which currently-divergent rows the repair could convert, split by whether
      the threshold is same-transition (convertible) or cross-turn (not);
  (c) how many branches the repair would add.

The clearance figure is an explicit UPPER BOUND. A row is counted convertible
when the repair would place a reachable arm on the observed side of the
threshold; whether the rest of that arm's transition then matches can only be
settled by the patched engine. Registering the bound before the patch exists is
the point -- a number derived afterwards proves nothing.

Usage::

    PYTHONPATH=src python scripts/c27_kill_split_premeasure.py \\
        --shards 'path/cert_shard_*.json' [--report fresh-report.json] --json out.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

GEN3_BRANCHER = REPO_ROOT / "third_party" / "poke-engine-src" / "src" / "gen3" / "generate_instructions.rs"

# The engine's own roll model: floor(base * random(85, 100) / 100), sixteen
# rolls, and 0.925 as the average multiplier for a collapsed arm.
ROLLS = range(85, 101)
AVERAGE_MULTIPLIER = 0.925


def _roll(base: int, percent: int) -> int:
    return int(base) * percent // 100


def _average_arm(base: int) -> int:
    return int(base * AVERAGE_MULTIPLIER)


def classify_damage_branch(non_crit_base: int, crit_base: int, defender_hp: int) -> str:
    """Which damage-branch case the engine takes for this hit.

    Mirrors the gen3 brancher's own guards so the measurement describes the
    code that exists, not the code being proposed.
    """

    if non_crit_base <= 0 or defender_hp <= 0:
        return "no_damage"
    min_non_crit = int(non_crit_base * 0.85)
    if non_crit_base >= defender_hp and min_non_crit < defender_hp:
        return "case_a_non_crit_straddle_split"
    if non_crit_base >= defender_hp:
        return "non_crit_always_kills"
    # Non-crit can never kill: the engine takes the crit/non-crit split only.
    if crit_base <= 0:
        return "case_b_no_crit_damage"
    min_crit = int(crit_base * 0.85)
    if crit_base < defender_hp:
        return "case_b_crit_never_kills"
    if min_crit >= defender_hp:
        return "case_b_crit_always_kills"
    return "case_b_crit_straddle_unsplit"


def crit_arm_gap(non_crit_base: int, crit_base: int, defender_hp: int) -> dict[str, Any] | None:
    """The missing side of the crit arm, when the repair would add one."""

    if classify_damage_branch(non_crit_base, crit_base, defender_hp) != "case_b_crit_straddle_unsplit":
        return None
    kill_rolls = [p for p in ROLLS if _roll(crit_base, p) >= defender_hp]
    survive_rolls = [p for p in ROLLS if _roll(crit_base, p) < defender_hp]
    emitted = _average_arm(crit_base)
    return {
        "crit_base": int(crit_base),
        "defender_hp": int(defender_hp),
        "emitted_crit_damage": emitted,
        "emitted_arm_kills": emitted >= defender_hp,
        "kill_rolls": len(kill_rolls),
        "survive_rolls": len(survive_rolls),
        # The arm the current branch set cannot reach.
        "absent_side": "survive" if emitted >= defender_hp else "kill",
    }


def _damage_bases(state, choices: Mapping[str, str]) -> dict[str, tuple[int, int]] | None:
    import poke_engine  # noqa: PLC0415

    try:
        s1, s2 = poke_engine.calculate_damage(
            state, choices.get("p1", ""), choices.get("p2", ""), True
        )
    except (KeyboardInterrupt, SystemExit):
        raise
    except BaseException:  # noqa: BLE001 - an unusable pair is a skip, not a crash
        return None
    s1, s2 = list(s1), list(s2)
    if len(s1) < 2 or len(s2) < 2:
        return None
    # side_one's rolls are the damage it DEALS, so they land on side_two.
    return {"p2": (s1[0], s1[1]), "p1": (s2[0], s2[1])}


def measure(rows: Sequence[Mapping[str, Any]], *, label: str) -> dict[str, Any]:
    import poke_engine  # noqa: PLC0415

    import cert_sweep_readout as readout  # noqa: PLC0415

    cases: Counter = Counter()
    added_branches: Counter = Counter()
    convertible: list[dict[str, Any]] = []
    by_family: Counter = Counter()
    skipped = 0

    for row in rows:
        choices = row.get("choices") or {}
        states = row.get("engine_states") or []
        pre = row.get("pre_features") or {}
        if not states or not choices:
            skipped += 1
            continue
        try:
            state = poke_engine.State.from_string(states[0])
        except BaseException:  # noqa: BLE001
            skipped += 1
            continue
        bases = _damage_bases(state, choices)
        if bases is None:
            skipped += 1
            continue

        family, _basis, _counter = readout.classify_row(row)
        row_added = 0
        row_gaps = []
        for slot, (non_crit, crit) in bases.items():
            defender_hp = pre.get(f"{slot}_hp")
            if not isinstance(defender_hp, int):
                continue
            case = classify_damage_branch(non_crit, crit, defender_hp)
            cases[case] += 1
            gap = crit_arm_gap(non_crit, crit, defender_hp)
            if gap is not None:
                row_added += 1
                row_gaps.append({"slot": slot, **gap})
        added_branches[row_added] += 1
        if row_gaps:
            by_family[family] += 1
            convertible.append({
                "seed": row.get("seed"), "step": row.get("step"),
                "current_family": family,
                "current_class": row.get("divergence_class"),
                "gaps": row_gaps,
            })

    return {
        "label": label,
        "rows": len(rows),
        "rows_skipped": skipped,
        "damage_branch_cases": dict(cases.most_common()),
        "rows_by_added_branches": {str(k): v for k, v in sorted(added_branches.items())},
        "rows_with_an_unreachable_crit_arm": len(convertible),
        "unreachable_crit_arm_by_current_family": dict(by_family.most_common()),
        "convertible_upper_bound": convertible,
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--shards", default=None, help="retained archive glob")
    parser.add_argument("--report", default=None, help="a fresh differential report")
    parser.add_argument("--json", required=True)
    args = parser.parse_args(argv)
    if not args.shards and not args.report:
        parser.error("need --shards or --report")

    populations: dict[str, list[Mapping[str, Any]]] = {}
    if args.report:
        report = json.loads(Path(args.report).read_text(encoding="utf-8"))
        retention = report.get("repro_retention") or {}
        if retention.get("repros_complete") is not True:
            print("REFUSED: fresh report retention is incomplete", file=sys.stderr)
            return 2
        populations["fresh_sample"] = [
            r for r in report.get("repros") or []
            if isinstance(r, Mapping) and r.get("kind") == "transition_diverged"
        ]
    if args.shards:
        from c26_archival_recalibration import verify_archive  # noqa: PLC0415

        try:
            _paths, retained = verify_archive(args.shards)
        except (OSError, ValueError, json.JSONDecodeError) as error:
            print(f"ARCHIVE INPUT REFUSED: {error}", file=sys.stderr)
            return 2
        populations["retained_archive"] = list(retained)

    out = {
        "schema": "c27-kill-split-premeasure/1",
        "purpose": (
            "Pre-registered measurement of the crit-arm kill-split repair, "
            "derived offline from retained rows before any engine change. The "
            "gen3 brancher already partitions the KO threshold when the "
            "NON-CRIT roll range straddles it (case A) and applies no kill "
            "split at all when only the CRIT range straddles it (case B); the "
            "repair applies the existing "
            "compare_health_with_damage_multiples identity to the crit arm."
        ),
        "prediction_discipline": (
            "convertible_upper_bound is an UPPER BOUND on clearance: it counts "
            "boundaries where the repair would place a reachable arm on the "
            "observed side of the threshold. Whether the rest of that arm's "
            "transition matches can only be settled by the patched engine. "
            "Rows whose threshold matters only through a LATER choice -- a "
            "Substitute that fails next turn because this hit left the mon "
            "below a quarter -- are not convertible by any same-transition "
            "partition and remain comparison limits."
        ),
        "gen3_brancher_sha256": hashlib.sha256(GEN3_BRANCHER.read_bytes()).hexdigest(),
        "readout_sha256": hashlib.sha256(
            (REPO_ROOT / "scripts" / "cert_sweep_readout.py").read_bytes()
        ).hexdigest(),
        "populations": {},
    }
    for label, rows in populations.items():
        out["populations"][label] = measure(rows, label=label)

    Path(args.json).write_text(json.dumps(out, indent=1) + "\n")
    for label, entry in out["populations"].items():
        print(f"== {label}: {entry['rows']} rows ({entry['rows_skipped']} skipped)")
        for case, count in entry["damage_branch_cases"].items():
            print(f"     {count:6d}  {case}")
        print(f"     rows with an unreachable crit arm: "
              f"{entry['rows_with_an_unreachable_crit_arm']}")
        print(f"     by current family: {entry['unreachable_crit_arm_by_current_family']}")
    print(f"-> {args.json}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
