#!/usr/bin/env python
"""WRONG-FAN CONTROL: does enumeration close those rows for its VALUES, or its COUNT?

``evaluate_boundary_strict`` accepts a boundary as soon as ANY rendered branch matches,
and enumeration multiplies the branch count by 8.5x-72.5x on the rows it closes. So
"nothing opened, four rows closed" is consistent with a real fix AND with a lottery: more
branches is more chances to match, whatever the branches say.

This separates the two, using the same negative-control pattern the rest of C134 uses.
For each row enumeration closes, the boundary is adjudicated three ways with the SAME
matcher and the same recorded inputs:

  collapsed   the shipped cascade                       -- must stay DIVERGED (reproduces
                                                           the committed artifact)
  enumerated  the oracle                                -- must MATCH (reproduces closure)
  wrong-fan   the oracle's branch COUNT, with the move  -- must stay DIVERGED
              damage shifted OFF the legal roll set

If the wrong fan also closes them, cardinality alone buys acceptance and the closure is a
lottery. If it does not, acceptance is discriminating on the values, and the closure is
attributable to enumeration being RIGHT rather than merely being BIG.

The shift is applied to the rendered branch events, not to the engine: enumeration itself
is untouched, so the control differs from the real thing in exactly one property.

Usage::

    PYTHONPATH=src python scripts/c134_wrong_fan_control.py --json <out.json>
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import re
import sys
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import pokezero_search  # noqa: E402

_spec = importlib.util.spec_from_file_location(
    "engine_transition_differential_control",
    REPO_ROOT / "scripts" / "engine_transition_differential.py",
)
assert _spec is not None and _spec.loader is not None
runner = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = runner
_spec.loader.exec_module(runner)

ARTIFACTS = REPO_ROOT / "reports" / "artifacts"

# ``|-damage|p2a: Name|113/244|[from] Sandstorm`` / ``|-damage|p2a: Name|0 fnt``
_HP_FIELD = re.compile(r"\A(\d+)/(\d+)\Z")


class _StateShim:
    """``evaluate_boundary_strict`` only ever calls ``.to_string()`` on a state."""

    def __init__(self, text: str) -> None:
        self._text = text

    def to_string(self) -> str:
        return self._text


class _Features:
    def __init__(self, payload: dict) -> None:
        self.p1_hp = payload["p1_hp"]
        self.p2_hp = payload["p2_hp"]


def _move_damage_values(rendered: dict, target: str) -> list[int]:
    """Distinct MOVE damage amounts across a rendered branch set, per branch.

    Move damage is an untagged ``-damage``; residuals carry ``[from] ...``.
    """

    values: set[int] = set()
    for branch in rendered.get("branches") or []:
        for line in branch.get("events") or []:
            parts = line.split("|")
            if len(parts) < 4 or parts[1] != "-damage" or not parts[2].startswith(target):
                continue
            if len(parts) > 4 and parts[4].startswith("[from]"):
                continue
            hp = _HP_FIELD.match(parts[3])
            if hp:
                values.add(int(hp.group(2)) - int(hp.group(1)))
            elif parts[3].endswith("fnt"):
                values.add(-1)  # a faint hides the amount; excluded from the shift basis
            break
    return sorted(v for v in values if v >= 0)


def _branch_has_move_damage(branch: dict, target: str) -> bool:
    return _branch_move_damage(branch, target) is not None


def _branch_move_damage(branch: dict, target: str) -> int | None:
    """This branch's MOVE damage on ``target``, or ``None``.

    Move damage is an untagged ``-damage``; residuals carry ``[from] ...``. A ``0 fnt``
    reading hides the amount and returns ``None``, which is why fainting branches cannot
    be remapped coherently and are dropped by the caller.
    """

    for line in branch.get("events") or []:
        parts = line.split("|")
        if len(parts) < 4 or parts[1] != "-damage" or not parts[2].startswith(target):
            continue
        if len(parts) > 4 and parts[4].startswith("[from]"):
            continue
        hp = _HP_FIELD.match(parts[3])
        if not hp:
            return None
        return int(hp.group(2)) - int(hp.group(1))
    return None


def wrong_fan_map(legal: list[int]) -> dict[int, int]:
    """An INJECTIVE remap of a legal roll fan onto values that are not legal rolls.

    Same cardinality, nearest available magnitude, disjoint from the input by
    construction. That is the control this script needs: "as many branches, none of them
    a legal roll".

    Nearest rather than a constant offset, because a constant offset does not survive
    contact with real fixtures. Shifting every value down by the fan's width overflows to
    negative damage on a low-HP defender -- on 19000074/27 it dropped every branch and
    left the control with nothing to say -- and clamping the offset instead leaves the
    shifted fan OVERLAPPING the legal one, which is a generous control masquerading as a
    strict one. A per-value nearest-free-integer map has neither failure: it stays in the
    same magnitude range, so no branch gains or loses a faint, and it is disjoint.
    """

    forbidden = set(legal)
    used: set[int] = set()
    mapping: dict[int, int] = {}
    for value in sorted(legal):
        for offset in range(1, 4 * max(2, len(legal))):
            for candidate in (value + offset, value - offset):
                if candidate <= 0 or candidate in forbidden or candidate in used:
                    continue
                mapping[value] = candidate
                used.add(candidate)
                break
            if value in mapping:
                break
    return mapping


def _remap_branch(branch: dict, target: str, mapping: dict[int, int]) -> bool:
    """Rewrite this branch's move damage to its remapped value.

    Every subsequent HP reading for the same target moves by the same delta, so only the
    MOVE component changes: the harness reconstructs components from consecutive HP
    readings, and rewriting one line alone would corrupt the next component too.
    """

    damage = _branch_move_damage(branch, target)
    if damage is None or damage not in mapping:
        return False
    delta = damage - mapping[damage]  # added to HP
    if delta == 0:
        return False

    events = list(branch.get("events") or [])
    shifting = False
    changed = False
    for index, line in enumerate(events):
        parts = line.split("|")
        if len(parts) < 4 or not parts[2].startswith(target):
            continue
        if parts[1] not in {"-damage", "-heal", "-sethp"}:
            continue
        tagged = len(parts) > 4 and parts[4].startswith("[from]")
        if not shifting:
            if parts[1] != "-damage" or tagged:
                continue
            shifting = True
        hp = _HP_FIELD.match(parts[3])
        if not hp:
            return False
        current, maximum = int(hp.group(1)), int(hp.group(2))
        moved = current + delta
        if not 0 < moved <= maximum:
            return False
        parts[3] = f"{moved}/{maximum}"
        events[index] = "|".join(parts)
        changed = True
    if changed:
        branch["events"] = events
    return changed


def _adjudicate(repro: dict, mode: str, shift_report: dict) -> tuple[str, int, list[str]]:
    slot_sides = repro["slot_sides"]
    # BOTH sides, always. The first version shifted only ``p2a``, on the assumption that
    # the defender is the second seat. It is not: 19100107/135 diverges on p1 (Roselia
    # taking Sacred Fire), so that run shifted a side the adjudication was not looking at
    # and the row "matched a wrong fan" while its wrong fan was untouched where it
    # mattered. Shifting both sides removes the guess.
    targets = ("p1a", "p2a")

    real_branch_events = pokezero_search.branch_events

    def patched(state, s1, s2, ctx, a, b):  # noqa: ANN001
        payload = json.loads(real_branch_events(state, s1, s2, ctx, a, b))
        if mode != "wrong_fan":
            return json.dumps(payload)
        shifts: dict[str, dict[int, int]] = {}
        for target in targets:
            legal = _move_damage_values(payload, target)
            if not legal:
                continue
            mapping = wrong_fan_map(legal)
            shifts[target] = mapping
            wrong = sorted(mapping.values())
            shift_report.setdefault("shifts", []).append(
                {
                    "target": target,
                    "legal_fan": legal,
                    "wrong_fan": wrong,
                    "same_cardinality": len(wrong) == len(legal),
                    "disjoint_from_legal": not (set(wrong) & set(legal)),
                }
            )
        if not shifts:
            shift_report["no_move_damage"] = shift_report.get("no_move_damage", 0) + 1
            return json.dumps(payload)
        # EVERY branch must move. A branch left unshifted keeps a CORRECT roll in the
        # fan, and the row then matches for the right reason -- which makes the control
        # prove nothing. The first version of this script had exactly that bug: 7 of 34
        # branches on 19100191/5 carried a ``0 fnt`` reading, which cannot be shifted
        # coherently because the faint hides the amount, so the true roll survived and
        # the "wrong" fan matched.
        #
        # Unshiftable branches are DROPPED and counted. That lowers the cardinality, so
        # the count is reported alongside the enumerated count and the comparison states
        # what it actually had.
        kept, dropped = [], 0
        for branch in payload.get("branches") or []:
            moved = True
            for target, mapping in shifts.items():
                if _branch_has_move_damage(branch, target) and not _remap_branch(
                    branch, target, mapping
                ):
                    moved = False
                    break
            if moved:
                kept.append(branch)
            else:
                dropped += 1
        payload["branches"] = kept
        shift_report["dropped_unshiftable"] = shift_report.get("dropped_unshiftable", 0) + dropped
        shift_report["kept_shifted"] = shift_report.get("kept_shifted", 0) + len(kept)
        return json.dumps(payload)

    pokezero_search.branch_events = patched
    try:
        counts: Counter = Counter()
        verdict, misses, branch_count = runner.evaluate_boundary_strict(
            states=[_StateShim(text) for text in repro["engine_states"]],
            slot_sides=slot_sides,
            choices=repro["choices"],
            party_display=repro["party_display"],
            turn=repro["turn"],
            pre_features=_Features(repro["pre_features"]),
            observed=_Features(repro["observed"]),
            step_lines=repro["protocol"],
            observed_boosts=repro.get("observed_boost_deltas") or {},
            active_changed=repro.get("active_changed") or {},
            counts=counts,
        )
    finally:
        pokezero_search.branch_events = real_branch_events
    return verdict, branch_count, misses[:3]


def _run_mode(mode: str) -> list[dict]:
    rows = []
    for window in ("dev", "holdout"):
        collapsed = json.loads(
            (ARTIFACTS / f"c134_collapsed_{window}_sweep.json").read_text(encoding="utf-8")
        )
        enumerated = json.loads(
            (ARTIFACTS / f"c134_enumerated_{window}_sweep.json").read_text(encoding="utf-8")
        )
        survived = {(r["seed"], r["step"]) for r in enumerated["repros"]}
        for repro in collapsed["repros"]:
            key = (repro["seed"], repro["step"])
            shift_report: dict = {}
            verdict, branch_count, misses = _adjudicate(repro, mode, shift_report)
            rows.append(
                {
                    "window": window,
                    "seed": repro["seed"],
                    "step": repro["step"],
                    "divergence_class": repro["divergence_class"],
                    "closed_by_enumeration": key not in survived,
                    "verdict": verdict,
                    "branches": branch_count,
                    "first_misses": misses,
                    "shift": shift_report,
                }
            )
    return rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("collapsed", "enumerated", "wrong_fan"), default=None)
    parser.add_argument("--json", default=None)
    args = parser.parse_args(argv)

    if args.mode:
        # One MODE per process. The engine latches POKEZERO_ENUMERATE_ROLLS in a
        # OnceLock on its first call, so a parent cannot switch paths mid-run; the
        # driver below spawns one child per mode with the right environment.
        print(json.dumps({"mode": args.mode, "rows": _run_mode(args.mode)}))
        return 0

    import os
    import subprocess

    combined: dict[tuple[str, int, int], dict] = {}
    for mode, flag in (("collapsed", "0"), ("enumerated", "1"), ("wrong_fan", "1")):
        environment = dict(os.environ)
        environment["POKEZERO_ENUMERATE_ROLLS"] = flag
        environment["PYTHONPATH"] = os.pathsep.join(
            [str(REPO_ROOT / "src"), environment.get("PYTHONPATH", "")]
        ).rstrip(os.pathsep)
        result = subprocess.run(
            [sys.executable, str(Path(__file__).resolve()), "--mode", mode],
            cwd=REPO_ROOT, env=environment, capture_output=True, text=True,
        )
        if result.returncode:
            print(result.stdout, file=sys.stderr)
            print(result.stderr, file=sys.stderr)
            raise SystemExit(f"{mode} child exited {result.returncode}")
        for row in json.loads(result.stdout)["rows"]:
            key = (row["window"], row["seed"], row["step"])
            entry = combined.setdefault(
                key,
                {
                    "window": row["window"], "seed": row["seed"], "step": row["step"],
                    "divergence_class": row["divergence_class"],
                    "closed_by_enumeration": row["closed_by_enumeration"],
                },
            )
            entry[mode] = {
                "verdict": row["verdict"],
                "branches": row["branches"],
                "first_misses": row["first_misses"],
            }
            if mode == "wrong_fan":
                entry["shift"] = row["shift"]

    rows = [combined[key] for key in sorted(combined)]

    closed = [r for r in rows if r["closed_by_enumeration"]]
    for row in rows:
        collapsed_branches = row["collapsed"]["branches"] or 1
        row["wrong_fan"]["inflation_vs_collapsed"] = round(
            row["wrong_fan"]["branches"] / collapsed_branches, 1
        )
        row["enumerated"]["inflation_vs_collapsed"] = round(
            row["enumerated"]["branches"] / collapsed_branches, 1
        )
        shifts = row.get("shift", {}).get("shifts", [])
        # A shift that OVERLAPS the legal fan leaves some correct values in place, so the
        # control is GENEROUS there -- easier to match, not harder. Reported per row
        # because a row that stays divergent under a generous control is stronger
        # evidence, and a row that matches under one proves nothing.
        row["control_strength"] = (
            "strict"
            if shifts
            and all(s["disjoint_from_legal"] and s["same_cardinality"] for s in shifts)
            else "WEAK -- inspect the shift block" if shifts
            else "n/a (no move damage to remap)"
        )

    verdicts = {
        # Sanity: the control reproduces the committed artifact before it perturbs it.
        "reproduced_collapsed_divergence": all(
            r["collapsed"]["verdict"] == "diverged" for r in rows
        ),
        "reproduced_enumerated_closure": all(
            r["enumerated"]["verdict"] == "matched" for r in closed
        ),
        # THE DISCRIMINATING RESULT. A wrong fan of comparable cardinality must not
        # close what the right fan closed.
        "wrong_fan_kept_every_closed_row_divergent": all(
            r["wrong_fan"]["verdict"] == "diverged" for r in closed
        ),
        # The control is only interesting if the wrong fan is still BIG. Cardinality is
        # not identical to the enumerated fan -- fainting branches cannot be shifted
        # coherently, because the ``0 fnt`` reading hides the amount, so they are dropped
        # and counted -- but it must retain most of the inflation the lottery hypothesis
        # is about.
        "wrong_fan_kept_at_least_5x_the_collapsed_cardinality": all(
            r["wrong_fan"]["inflation_vs_collapsed"] >= 5.0 for r in closed
        ),
        "every_closed_row_had_at_least_one_strict_shift": all(
            any(
                s["disjoint_from_legal"]
                for s in r.get("shift", {}).get("shifts", [])
            )
            for r in closed
        ),
    }
    payload = {"rows": rows, "verdicts": verdicts}
    print(json.dumps(payload, indent=2))
    if args.json:
        Path(args.json).write_text(json.dumps(payload, indent=2))
    return 0 if all(verdicts.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
