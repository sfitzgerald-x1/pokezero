#!/usr/bin/env python
"""WRONG-FAN CONTROL: does enumeration close those rows for its VALUES, or its COUNT?

**ANSWER: neither. Acceptance does not read the branch's damage value at all.** A fan
carrying no legal roll value earns acceptance on all four rows, and on two of them even
when every value is 12x-106x away from any legal roll. The script still exits non-zero,
because ``only_visible_wrong_fan_kept_every_closed_row_divergent`` is the property it was
built to demonstrate and that property is FALSE.

THE QUESTION. ``evaluate_boundary_strict`` accepts as soon as ANY rendered branch matches,
and enumeration multiplies the branch count 8.5x-72.5x on the rows it closes. So "nothing
opened, four rows closed" is consistent with a real fix AND with a lottery.

THE ARMS. Each row is adjudicated by the SAME matcher on the SAME recorded inputs:

  collapsed                    the shipped cascade            -- reproduces the artifact
  enumerated                   the oracle, untouched          -- reproduces the closure
  wrong_fan                    whole fan, set HELD FIXED,
                               values remapped off the legal
                               roll set
  drop_only                    drop only hidden-amount
                               branches; values left LEGAL
  drop_only_legacy             drop the set a SUPERSEDED
                               version dropped; values LEGAL
  only_lethal                  keep only hidden-amount branches
  only_visible                 keep only visible-amount branches
  only_visible_wrong_fan       only_visible, then remap every
                               value off the legal set
  only_visible_wrong_fan_far   the same, pushed to the far end
                               of each branch's feasible window

WHY only_visible IS THE WELL-POSED SUBSET. A branch whose target faints ON THE MOVE renders
as ``0 fnt``; the amount is not in the protocol, so that branch is compatible with every
lethal roll including the legal ones and cannot be made "wrong". Restricting to
visible-amount branches removes that remainder entirely -- and it is where the closing
branch lives, since ``only_visible`` matches on every row while ``only_lethal`` diverges on
every row.

A branch that survives the move and faints LATER from a residual is NOT irreducible, though
an earlier version of this script claimed it was. Remap it UPWARD: if it reached the
residual at HP h and died, at h-delta it still dies, because the residual removed at least
h. The ``0 fnt`` line is untouched and the outcome class is preserved by monotonicity.

WHAT THE RESULT MEANS. ``roll_components_agree`` accepts a roll-scaled component when the
OBSERVED magnitude is in a legal roll set derived from the STATE -- "an ADDITIONAL accept
path, never a veto" -- so the branch's own value is consulted only through a +/-9%
proportional fallback. A value-based wrong fan is therefore the wrong instrument. What
enumeration multiplies is branch STRUCTURE, and that is what a discriminating control would
have to perturb.

FIVE CONFOUNDS were found in this one script: a shift too large for a low-HP defender, a
clamped shift that overlapped the legal fan, a run that perturbed only ``p2a`` when the row
diverged on ``p1``, a drop of unremappable branches that produced the entire result, and a
helper deleted by an edit that turned a whole arm into ``skip_lossy(0)`` while still
emitting a tidy artifact. The last is why ``_adjudicate`` now RAISES on any
``branch_events`` error rather than inheriting the differential's fail-soft behaviour.

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


def _shiftable_readings(
    branch: dict, target: str
) -> tuple[list[tuple[int, int, int]], bool] | None:
    """``([(event_index, current_hp, max_hp)], trailing_faint)`` for a remap.

    The move-damage line and every later HP reading for the same target: the harness
    reconstructs components from CONSECUTIVE readings, so moving one line alone would
    corrupt the next component as well as the one intended.

    A trailing ``0 fnt`` is TOLERATED and reported, not treated as a failure. It marks a
    branch where the target survived the move and died to a residual. Such a branch is
    still remappable -- upward only -- see ``_remap_branch``. An earlier version returned
    ``None`` here, which dropped 6/406/112/5 branches across the four rows and led this
    document to claim, wrongly, that those branches were irreducible.
    """

    readings: list[tuple[int, int, int]] = []
    shifting = False
    trailing_faint = False
    for index, line in enumerate(branch.get("events") or []):
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
            if parts[3].endswith("fnt") and readings:
                # The target dies here. The line carries no amount, so there is nothing
                # to shift, and nothing after it can move either.
                trailing_faint = True
                break
            return None
        readings.append((index, int(hp.group(1)), int(hp.group(2))))
    if not readings:
        return None
    return readings, trailing_faint


def _remap_branch(
    branch: dict, target: str, legal: set[int], preferred: int | None, *, far: bool = False
) -> bool:
    """Rewrite this branch's move damage to a value that is NOT a legal roll.

    ``preferred`` is the fan-wide injective choice; it is used when it FITS, otherwise the
    nearest feasible non-legal value is used, and the branch fails only when no such value
    exists.

    TRAILING-FAINT BRANCHES ARE REMAPPED UPWARD ONLY, and that is what makes them
    tractable. If the target reached the residual at HP ``h`` and died, the residual
    removed at least ``h``; dealing MORE move damage brings it to the residual at
    ``h - delta < h``, where the same residual still kills. The faint is preserved by
    monotonicity, the ``0 fnt`` line is never touched, and every other reading on that
    target moves by the same delta -- so the outcome class is identical and only the value
    changed. Downward would be unsound: less damage can leave the target above the
    residual and turn a faint into a survival, which is a different branch rather than a
    different value.

    This is the construction the report previously called impossible. It is not, and the
    measurement is in ``reports/artifacts/c134_wrong_fan_control.json``.
    """

    found = _shiftable_readings(branch, target)
    if found is None:
        return False
    readings, trailing_faint = found
    _, first_hp, first_max = readings[0]
    damage = first_max - first_hp
    # Feasible shift window: every moved reading must stay strictly above 0 (no branch may
    # gain a faint it did not have) and at or below max HP.
    lowest = min(hp for _, hp, _ in readings)
    headroom = min(maximum - hp for _, hp, maximum in readings)
    min_delta, max_delta = 1 - lowest, headroom

    candidates: list[int] = []
    if far:
        # FAR variant. The nearest non-legal integer sits within a point or two of a legal
        # roll, i.e. comfortably inside the matcher's proportional fallback window
        # (0.92x-1.09x of the engine's value). A fan that is "wrong but adjacent" therefore
        # cannot separate the two accept paths. This walks the feasible window from its
        # extremes INWARD, so the value lands as far from any legal roll as the branch
        # physically allows.
        for delta in sorted(range(min_delta, max_delta + 1), key=lambda d: -abs(d)):
            candidates.append(damage - delta)
    else:
        if preferred is not None:
            candidates.append(preferred)
        for offset in range(1, 4 * max(4, len(legal)) + 1):
            candidates.extend((damage + offset, damage - offset))

    for candidate in candidates:
        if candidate <= 0 or candidate in legal:
            continue
        if trailing_faint and candidate <= damage:
            # Upward only. See the docstring: downward can undo the faint.
            continue
        delta = damage - candidate
        if not (min_delta <= delta <= max_delta) or delta == 0:
            continue
        events = list(branch["events"])
        for index, hp, maximum in readings:
            parts = events[index].split("|")
            parts[3] = f"{hp + delta}/{maximum}"
            events[index] = "|".join(parts)
        branch["events"] = events
        return True
    return False


def _legacy_remap_feasible(branch: dict, target: str, mapping: dict[int, int]) -> bool:
    """Would the SUPERSEDED single-candidate remap have worked on this branch?

    The superseded control took one fan-wide value per legal roll, could not see past a
    trailing faint at all, and DROPPED the branch whenever either failed. Reproduced
    verbatim so ``drop_only_legacy`` deletes exactly the set that control deleted.
    """

    found = _shiftable_readings(branch, target)
    if found is None:
        return False
    readings, trailing_faint = found
    if trailing_faint:
        return False
    _, first_hp, first_max = readings[0]
    damage = first_max - first_hp
    candidate = mapping.get(damage)
    if candidate is None:
        return False
    delta = damage - candidate
    if delta == 0:
        return False
    return all(0 < hp + delta <= maximum for _, hp, maximum in readings)


MODES = (
    "collapsed", "enumerated", "wrong_fan",
    "drop_only", "drop_only_legacy", "only_lethal", "only_visible",
    "only_visible_wrong_fan", "only_visible_wrong_fan_far",
)


def _branch_damage_class(branch: dict, target: str) -> str:
    """``"visible"``, ``"hidden"`` or ``"none"`` for this branch's move damage.

    HIDDEN means the target faints on the move: the renderer emits ``0 fnt`` and the
    amount is simply not in the protocol. That distinction is the whole design of this
    control, so it is computed once and named rather than inferred from a failure.
    """

    for line in branch.get("events") or []:
        parts = line.split("|")
        if len(parts) < 4 or parts[1] != "-damage" or not parts[2].startswith(target):
            continue
        if len(parts) > 4 and parts[4].startswith("[from]"):
            continue
        return "visible" if _HP_FIELD.match(parts[3]) else "hidden"
    return "none"


def _adjudicate(repro: dict, mode: str, shift_report: dict) -> tuple[str, int, list[str]]:
    slot_sides = repro["slot_sides"]
    # BOTH sides, always. An early version shifted only ``p2a``, on the assumption that
    # the defender is the second seat. It is not: 19100107/135 diverges on p1 (Roselia
    # taking Sacred Fire), so that run perturbed a side the adjudication was not looking
    # at and reported "a wrong fan also matches" while its wrong fan was untouched where
    # it mattered.
    targets = ("p1a", "p2a")

    real_branch_events = pokezero_search.branch_events

    def patched(state, s1, s2, ctx, a, b):  # noqa: ANN001
        payload = json.loads(real_branch_events(state, s1, s2, ctx, a, b))
        if mode in {"collapsed", "enumerated"}:
            return json.dumps(payload)

        branches = list(payload.get("branches") or [])
        classes = {
            target: [_branch_damage_class(branch, target) for branch in branches]
            for target in targets
        }
        # A branch is HIDDEN-VALUED if any side's move damage is concealed by a faint.
        hidden = [
            any(classes[target][index] == "hidden" for target in targets)
            for index in range(len(branches))
        ]

        if mode == "only_lethal":
            payload["branches"] = [b for b, h in zip(branches, hidden) if h]
            return json.dumps(payload)
        if mode == "only_visible":
            payload["branches"] = [b for b, h in zip(branches, hidden) if not h]
            return json.dumps(payload)
        if mode == "drop_only_legacy":
            # REPRODUCES REVIEW'S ISOLATING DIAGNOSTIC, re-derived here rather than
            # quoted. It drops exactly the branch set the SUPERSEDED control dropped --
            # every branch where a single fan-wide remap attempt was infeasible, whether
            # because the amount was hidden by a faint or because the shift left the HP
            # range -- and leaves the survivors' values LEGAL and untouched.
            #
            # If that reproduces the superseded control's "diverged" verdicts, then those
            # verdicts came from the DROP and said nothing about values.
            legacy_mapping: dict[str, dict[int, int]] = {}
            for target in targets:
                legal = _move_damage_values(payload, target)
                if legal:
                    legacy_mapping[target] = wrong_fan_map(legal)
            kept = []
            for index, branch in enumerate(branches):
                droppable = hidden[index]
                if not droppable:
                    for target, mapping in legacy_mapping.items():
                        if classes[target][index] != "visible":
                            continue
                        if not _legacy_remap_feasible(branch, target, mapping):
                            droppable = True
                            break
                if not droppable:
                    kept.append(branch)
            payload["branches"] = kept
            shift_report["legacy_dropped"] = shift_report.get("legacy_dropped", 0) + (
                len(branches) - len(kept)
            )
            return json.dumps(payload)
        if mode == "drop_only":
            # THE ISOLATING DIAGNOSTIC. Apply the same branch-set reduction the earlier
            # version of this control applied -- drop every hidden-valued branch -- and
            # leave the survivors' values LEGAL and untouched. If the row goes divergent
            # here too, the earlier control's verdict came from the DROP and not from the
            # remap, and it measured nothing about values. It did, on all four rows.
            payload["branches"] = [b for b, h in zip(branches, hidden) if not h]
            shift_report["drop_only_dropped"] = shift_report.get("drop_only_dropped", 0) + sum(hidden)
            return json.dumps(payload)

        # mode in the wrong-fan family: VARY VALUES ONLY.
        #
        # ``only_visible_wrong_fan`` first restricts to visible-amount branches. That
        # subset carries ZERO irreducible contamination -- there is no ``0 fnt``-on-the-
        # move branch left to be compatible with every lethal roll -- so it is the branch
        # set on which a value-only control is actually well posed. The premise is this
        # branch's own measurement: ``only_visible`` matches on every row and
        # ``only_lethal`` diverges on every row, so the closing branch lives here.
        if mode in {"only_visible_wrong_fan", "only_visible_wrong_fan_far"}:
            branches = [b for b, h in zip(branches, hidden) if not h]
            payload["branches"] = branches
            classes = {
                target: [_branch_damage_class(branch, target) for branch in branches]
                for target in targets
            }
            hidden = [False] * len(branches)
        #
        # Nothing is dropped. A hidden-valued branch is kept UNCHANGED, and that is not a
        # concession: the renderer emits ``0 fnt``, so every lethal roll produces a
        # byte-identical branch. Remapping such a branch onto another lethal value is a
        # no-op at the only layer the matcher can see, so there is no value to vary and
        # nothing to hold out. Dropping it instead -- which is what the first three
        # versions of this control did -- changes the branch SET, and review proved with
        # a ``drop_only`` arm that the drop, not the remap, was producing the whole
        # result.
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

        remapped = failed = kept_lethal = 0
        for index, branch in enumerate(branches):
            if hidden[index]:
                kept_lethal += 1
                continue
            moved = False
            for target, mapping in shifts.items():
                if classes[target][index] != "visible":
                    continue
                legal_set = set(mapping)
                damage = _branch_move_damage(branch, target)
                if _remap_branch(
                    branch, target, legal_set, mapping.get(damage),
                    far=mode.endswith("_far"),
                ):
                    moved = True
                else:
                    failed += 1
            remapped += 1 if moved else 0
        payload["branches"] = branches  # unchanged length, by construction

        # CONTAMINATION, measured rather than assumed. Any branch still carrying a legal
        # move-damage value is a branch that could close the row for the RIGHT reason,
        # which makes a "wrong fan matched" verdict uninterpretable. Counting it is what
        # turns this script from an argument into a measurement -- three earlier versions
        # asserted a conclusion this number would have refuted.
        still_legal = 0
        for index, branch in enumerate(branches):
            if hidden[index]:
                # A lethal branch renders as ``0 fnt``. That rendering is compatible with
                # EVERY lethal roll, the legal ones included, so it can close the row for
                # a correct-value reason no matter what this script does to it. It counts
                # as contamination.
                still_legal += 1
                continue
            for target, mapping in shifts.items():
                if classes[target][index] != "visible":
                    continue
                value = _branch_move_damage(branch, target)
                if value is not None and value in set(mapping):
                    still_legal += 1
                    break
        shift_report["remapped"] = shift_report.get("remapped", 0) + remapped
        shift_report["kept_lethal_unchanged"] = shift_report.get("kept_lethal_unchanged", 0) + kept_lethal
        shift_report["remap_failed"] = shift_report.get("remap_failed", 0) + failed
        shift_report["branches_compatible_with_a_legal_roll"] = (
            shift_report.get("branches_compatible_with_a_legal_roll", 0) + still_legal
        )
        shift_report["branches_total"] = shift_report.get("branches_total", 0) + len(branches)

        # INDEPENDENT RE-EXTRACTION. The counter above is derived from the same bookkeeping
        # that did the remapping, so it would not notice a remap that silently no-opped.
        # This re-reads the EMITTED values straight out of the payload the matcher will
        # receive and intersects them with the legal fan -- the same check a reviewer would
        # run from outside, computed here so the artifact carries it.
        emitted_legal: dict[str, list[int]] = {}
        displacement = 0.0
        for target in targets:
            legal_fan = set(shifts.get(target) or {})
            if not legal_fan:
                continue
            emitted = {
                value
                for branch in branches
                if (value := _branch_move_damage(branch, target)) is not None
            }
            overlap = sorted(emitted & legal_fan)
            if overlap:
                emitted_legal[target] = overlap
            # How far the wrong values actually sit from the nearest legal roll, as a
            # fraction. The matcher's fallback accept path is a proportional window of
            # about +/-9% around the engine's value, so a displacement well inside that
            # cannot separate "value matters" from "value is ignored".
            for value in emitted:
                nearest = min(abs(value - legal) for legal in legal_fan)
                displacement = max(displacement, nearest / max(value, 1))
        shift_report["max_relative_displacement"] = round(
            max(displacement, shift_report.get("max_relative_displacement", 0.0)), 4
        )
        shift_report["emitted_values_still_legal"] = {
            **shift_report.get("emitted_values_still_legal", {}),
            **emitted_legal,
        }
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
    # FAIL LOUDLY. ``evaluate_boundary_strict`` catches every exception out of
    # ``branch_events`` and counts it, then falls through to ``skip_lossy`` when no branch
    # survives. That is right for the differential and wrong for this script: a bug in the
    # patched callback would look like a clean "this boundary is unmeasurable" verdict.
    # It already did once -- a helper deleted by an edit turned the whole drop_only_legacy
    # arm into skip_lossy(0) and the run still produced a tidy artifact.
    errors = {k: v for k, v in counts.items() if k.startswith("strict:branch_events_error")}
    if errors:
        raise RuntimeError(f"{mode}: the patched branch_events raised {errors}")
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
    parser.add_argument("--mode", choices=MODES, default=None)
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
    for mode, flag in (
        ("collapsed", "0"), ("enumerated", "1"), ("wrong_fan", "1"),
        ("drop_only", "1"), ("drop_only_legacy", "1"),
        ("only_lethal", "1"), ("only_visible", "1"),
        ("only_visible_wrong_fan", "1"), ("only_visible_wrong_fan_far", "1"),
    ):
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
            if row.get("shift"):
                entry.setdefault("shift", {})[mode] = row["shift"]

    rows = [combined[key] for key in sorted(combined)]

    closed = [r for r in rows if r["closed_by_enumeration"]]
    for row in rows:
        collapsed_branches = row["collapsed"]["branches"] or 1
        for mode in MODES:
            if mode in row:
                row[mode]["inflation_vs_collapsed"] = round(
                    row[mode]["branches"] / collapsed_branches, 1
                )
        shifts = row.get("shift", {}).get("wrong_fan", {}).get("shifts", [])
        row["control_strength"] = (
            "strict"
            if shifts
            and all(s["disjoint_from_legal"] and s["same_cardinality"] for s in shifts)
            else "WEAK -- inspect the shift block" if shifts
            else "n/a (no move damage to remap)"
        )

    def shift_of(row: dict, mode: str) -> dict:
        return row.get("shift", {}).get(mode, {})

    verdicts = {
        # Sanity: reproduce the committed artifact before perturbing anything.
        "reproduced_collapsed_divergence": all(
            r["collapsed"]["verdict"] == "diverged" for r in rows
        ),
        "reproduced_enumerated_closure": all(
            r["enumerated"]["verdict"] == "matched" for r in closed
        ),
        # The branch set must be HELD FIXED in both value-only arms. This is the property
        # whose absence confounded the first versions of this control: they dropped every
        # branch they could not remap, and the DROP produced the result.
        "wrong_fan_held_the_branch_set_fixed": all(
            r["wrong_fan"]["branches"] == r["enumerated"]["branches"] for r in rows
        ),
        "only_visible_wrong_fan_held_its_branch_set_fixed": all(
            r["only_visible_wrong_fan"]["branches"] == r["only_visible"]["branches"]
            for r in rows
        ),
        # PRECONDITION, and on only_visible it now PASSES. Independently re-extracted from
        # the emitted payload, not read off the remapper's own bookkeeping.
        "only_visible_wrong_fan_emits_no_legal_value": all(
            shift_of(r, "only_visible_wrong_fan").get("emitted_values_still_legal", {}) == {}
            for r in closed
        ),
        "only_visible_wrong_fan_remapped_everything": all(
            shift_of(r, "only_visible_wrong_fan").get("remap_failed", 1) == 0
            for r in closed
        ),
        # THE ISOLATING DIAGNOSTIC that condemned the superseded control: drop the same
        # set with LEGAL values and the rows still reopen, so the drop did the work.
        "drop_only_legacy_reproduces_the_superseded_verdict": all(
            r["drop_only_legacy"]["verdict"] == "diverged" for r in closed
        ),
        # THE ANSWER. On a zero-contamination fan carrying no legal roll value at all,
        # does acceptance survive? Where it does, the closure is NOT value-driven.
        "only_visible_wrong_fan_kept_every_closed_row_divergent": all(
            r["only_visible_wrong_fan"]["verdict"] == "diverged" for r in closed
        ),
    }
    payload = {"rows": rows, "verdicts": verdicts}
    print(json.dumps(payload, indent=2))
    if args.json:
        Path(args.json).write_text(json.dumps(payload, indent=2))
    return 0 if all(verdicts.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
