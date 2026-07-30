#!/usr/bin/env python
"""Certification-sweep readout: aggregate shards, score class rates against the
pre-registered table, and attribute EVERY divergent row to a documented family.

The binding criterion (reports/c14_sweep_prediction.json): zero unattributed
rows. A row that matches no documented family signature lands in the
`unattributed` list and the sweep FAILS — the list is the deliverable then,
not a rug.

Attribution is by MECHANISM SIGNATURE, not row identity (fresh seeds):

  limit:* classes                      -> documented limit classes (self-attributing)
  Sleep Talk callee union              -> I6 (LOSSY flag / [from] Sleep Talk + fan)
  cap-state / _to_full shape           -> I1 (avg-roll world evolution; Z7.3/Z10.1)
  capped-lethal roll-divergence shape  -> documented limit-shape family (X-walk class)
  majority-arm equal-magnitude tie     -> I4 (mapper attribution, #908/I.2 lineage)
  slice ends pre-upkeep / battle end   -> I5 (measurement-boundary truncation, #876)
  Pain Split / drain-cap inheritance   -> I3 (roll-inherited exact components)
  observed roll in engine's legal set  -> I2 (matcher accounting / legal-set gap)
  observed within [0.919, 1.09] window -> I2 (overreport window)
  best branch matches observed         -> I2 (pairing/branch-selection echo)

Anything else is UNATTRIBUTED pending replay (`scripts/replay_residue.py`).

Usage::

    PYTHONPATH=src python scripts/cert_sweep_readout.py \\
        --shards cert_shard_*.json --prediction reports/c14_sweep_prediction.json \\
        --json cert_readout.json
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import poke_engine  # noqa: E402

from engine_transition_differential import (  # noqa: E402
    _ROLL_SCALED_SOURCES,
    damage_components,
    legal_roll_damages,
    _split_components,
)

_PCT_RE = re.compile(r"pct=([\d.]+)")
_PAIR_RE = re.compile(r"\('([^']*)',\s*(-?\d+)\)")
_MISS_SIDE_RE = re.compile(r"\b(p[12])\s+(?:attributed|roll-scaled) components differ")


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return 0.0, 0.0
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - h), c + h


def _majority_miss(misses: Sequence[str]) -> str:
    if not misses:
        return ""
    return max(misses, key=lambda m: float(_PCT_RE.search(m).group(1)) if _PCT_RE.search(m) else 0.0)


def _miss_pairs(miss: str) -> tuple[list[tuple[str, int]], list[tuple[str, int]]]:
    """(observed, engine) component pairs out of a miss string."""
    obs_part, _, eng_part = miss.partition("engine")
    obs = [(s, int(v)) for s, v in _PAIR_RE.findall(obs_part)]
    eng = [(s, int(v)) for s, v in _PAIR_RE.findall(eng_part)]
    return obs, eng


def _sibling_arm_carries_observed_components(
    misses: Sequence[str], majority: str, observed: Sequence[tuple[str, int]]
) -> bool:
    """Whether a non-majority engine arm exactly reproduces the observed components.

    A count mismatch alone says nothing about why the majority arm differs.  The
    structural-echo family is valid only when another engine arm actually
    carries the full observed component multiset.  Empty observations are not
    evidence of a carried shape.
    """

    observed_components = Counter(observed)
    majority_side = _MISS_SIDE_RE.search(majority)
    if not observed_components or majority_side is None:
        return False
    for miss in misses:
        if miss == majority:
            continue
        sibling_side = _MISS_SIDE_RE.search(miss)
        if sibling_side is None or sibling_side.group(1) != majority_side.group(1):
            continue
        _, sibling_engine_components = _miss_pairs(miss)
        if Counter(sibling_engine_components) == observed_components:
            return True
    return False


def attribute_row(row: Mapping[str, Any]) -> tuple[str, str]:
    """Return (family, basis) for a divergent row, or ("UNATTRIBUTED", why)."""

    cls = row.get("divergence_class") or ""
    if cls.startswith("limit:"):
        return cls, "documented limit class (self-attributing)"

    protocol = [l for l in (row.get("protocol") or []) if not l.startswith("|request|")]
    proto = "\n".join(protocol)
    misses = row.get("branch_misses") or []
    majority = _majority_miss(misses)
    obs_c, eng_c = _miss_pairs(majority)

    # ------------------------------------------------------------------
    # NEW-MECHANISM AND CANDIDATE EXCLUSIONS, ordered AHEAD of every
    # broader family rule. The collision discipline is the absorb lesson
    # (Z12.6): a narrow mechanism shape must be tested before any rule wide
    # enough to swallow it, and each rule cites a validation row FROM THE
    # SWEEP data — the c13 validation set cannot contain these shapes, so
    # a rule only proven there is unproven where it matters.
    # These return UNATTRIBUTED deliberately: post-fix they predict ZERO,
    # and any appearance in a future sweep must surface, not be absorbed.
    # ------------------------------------------------------------------
    choices = row.get("choices") or {}
    # Recharge-turn residual gap (validation: s2000583/23 — |cant|recharge,
    # engine branch emits no EOT residuals). Wide count-mismatch rules below
    # would eat this shape.
    if ("|recharge" in proto or "none" in (choices.get("p1", ""), choices.get("p2", ""))):
        return ("UNATTRIBUTED",
                "recharge-turn boundary: engine branch drops end-of-turn "
                "residuals (sweep NEW mechanism; predicts zero post-fix)")
    # Truant loaf-phase drift (validation: s2000059/11 — Slaking attacks in
    # the sim, engine's branch loafed; also the inverse s2000054/49). The
    # structural-arm echo rule below would eat both directions.
    if ("Slaking" in proto or "Slakoth" in proto or "Truant" in proto) and (
            (not obs_c and eng_c) or (obs_c and not eng_c)):
        return ("UNATTRIBUTED",
                "Truant boundary with one-sided damage components: loaf-phase "
                "drift (sweep NEW mechanism; predicts zero post-#970)")
    # Recoil basis on a Substitute-breaking hit (validation: s2000031/60 —
    # obs recoil -10 vs engine -21 after |-end|Substitute). WHAT-level
    # candidate; the magnitude rules below would misfile it as accounting.
    if "Substitute" in proto and any(
            s == "recoil" for s, _ in obs_c + eng_c):
        return ("UNATTRIBUTED",
                "recoil magnitude on a Substitute-breaking hit (WHAT-level "
                "candidate: damage-basis question, WHY open)")
    # Incapacitated-arm pricing (validation: s2000131/47 — observed
    # |cant|frz outcome, engine majority arm attacks). The crit/structural
    # echo rules below would swallow the engine-only-damage shape.
    if cls == "roll_scaled_component" and not obs_c and eng_c \
            and "|cant|" in proto and (
            "|frz" in proto or ("slp" in proto and "|-status|" in proto)):
        return ("UNATTRIBUTED",
                "observed |cant| frz/fresh-slp outcome is not the engine "
                "majority arm (WHAT-level candidate: arm pricing, WHY open)")

    # Best-branch echo: when the missing MASS is a small minority, the
    # engine's majority branch reproduced the observed transition and the
    # misses are sibling branches that are CORRECTLY different (a crit arm
    # that did not happen, a fainted arm whose residuals correctly differ).
    # The divergence is branch-set accounting, not disagreement — c12's
    # matcher_accounting_best_branch family, and the minority-labelled
    # X-walk shapes whose majority matched.
    total_miss_mass = 0.0
    for m in misses:
        found = _PCT_RE.search(m)
        if found:
            total_miss_mass += float(found.group(1))
    if misses and total_miss_mass < 50.0:
        return ("I2_matcher_accounting",
                f"best branch reproduces the observed transition; misses carry only "
                f"{total_miss_mass:.1f}% minority mass (branch-set accounting echo)")

    # Confusion fan: self-hit roll + hidden-counter fan over confusion arms.
    if "confusion" in proto and cls == "roll_scaled_component":
        return ("LS_confusion_fan",
                "confusion self-hit fan (documented comparison-limit shape)")

    # I6: Sleep Talk callee union
    if "[from] Sleep Talk" in proto or any("sleeptalk" in str(m) for m in misses) \
            or cls in ("mapper_lossy", "no_usable_branch"):
        return "I6_sleeptalk_callee_union", "Sleep Talk callee boundary / lossy callee-union rendering"

    # ABSORB-SHAPE EXCLUSION — ordered AHEAD of the I1 _to_full rule
    # deliberately (#969 review): an absorb-ability heal that happens to CAP
    # renders as `abilitywaterabsorb_to_full` and satisfied the I1 signature,
    # hiding ~38 sweep rows of the NEW absorb mechanisms inside a documented
    # family. The 103/103 validation could not catch this: c13 contains zero
    # absorb rows, so the signature was never exercised there.
    joined = " ".join(misses)
    if re.search(r"ability(?:water|volt)absorb", joined):
        engine_side_absorb = any(
            re.search(r"ability(?:water|volt)absorb", miss.partition("engine")[2])
            for miss in misses)
        blocked_or_missed = ("Protect" in proto and "-activate" in proto) or \
            "|-miss|" in proto or "[miss]" in proto
        if engine_side_absorb and blocked_or_missed:
            return ("UNATTRIBUTED",
                    "engine-only absorb heal on a Protect-blocked or missed "
                    "move (sweep NEW absorb mechanism; capped variant)")

    # I1: cap-state shape (_to_full on either side of the majority miss)
    if "_to_full" in majority:
        return "I1_cap_state_shape", "capped-heal component shape in the majority miss (avg-roll world evolution)"

    # I4: equal-magnitude label tie in the majority arm only. A tie confined
    # to a minority arm cannot explain the majority-arm complaint.
    if obs_c and eng_c and len(obs_c) == len(eng_c) \
            and sorted(abs(v) for _, v in obs_c) == sorted(abs(v) for _, v in eng_c) \
            and sorted(s for s, _ in obs_c) != sorted(s for s, _ in eng_c):
        return "I4_attribution_tie", "identical magnitudes, different source labels in the majority arm (mapper attribution)"

    # I5: measurement boundary — slice ends before upkeep, or battle ends
    engine_only_residuals = (not obs_c) and eng_c and all(
        s and s not in ("",) for s, _ in eng_c)
    if ("|upkeep" not in proto or "|win|" in proto) and (
            engine_only_residuals or cls.startswith("component_extra_in_engine:")):
        return "I5_boundary_truncation", "Showdown slice ends before residuals (faint/switch/battle-end); engine completes the turn"

    # capped-lethal roll-divergence shape: majority miss roll-scaled with a
    # capped component on either side; residual-named or compound class
    if "roll-scaled" in majority and "capped_lethal" in majority:
        return "LS_capped_lethal_shape", "faint-divergent roll shape (documented X-walk comparison-limit family)"
    if cls.startswith(("component_missing_in_engine:", "component_extra_in_engine:",
                       "component_mismatch:", "component_magnitude:")):
        joined = " ".join(misses)
        if "capped_lethal" in joined:
            return "LS_capped_lethal_shape", "faint-divergent roll shape in a minority/compound branch (X-walk family)"

    # I3: roll-inherited deterministic components
    if "movepainsplit" in cls or "sethp" in majority:
        return "I3_roll_inherited", "Pain Split value inherits the same-turn roll (B.4/C.2 family)"

    # Minority-residual echo (the #946/W.6 signature): a small-arm miss that
    # complains ONLY about observed-side residuals — the arm reproduced the
    # damage, killed, and correctly deferred residuals — while the majority
    # miss is a roll-magnitude complaint. Faint-divergent comparison shape.
    minority_residual = any(
        (m := _PCT_RE.search(miss)) and float(m.group(1)) <= 25.0
        and "attributed components differ" in miss
        and "engine_only=[]" in miss
        for miss in misses)
    if minority_residual and cls.startswith(("component_missing_in_engine:",
                                             "component_extra_in_engine:")):
        return ("LS_capped_lethal_shape",
                "minority arm reproduces modulo faint-deferred residuals; majority "
                "differs only on roll magnitude (X-walk comparison-limit family)")

    # roll_scaled_component: replay through the committed triage instrument
    if cls == "roll_scaled_component":
        verdict = _attribute_roll_row(row, majority, obs_c, eng_c)
        if verdict:
            return verdict
        bucket = _triage_bucket(row)
        if bucket == "matches_best_branch":
            return ("I2_matcher_accounting",
                    "triage: best branch matches observed per-component "
                    "(capped inequalities + legal-set membership)")
        if bucket == "legitimate_roll_in_legal_set" or (
                bucket and bucket.startswith("damage_calc:within_roll_window")):
            return "I2_matcher_accounting", f"triage: {bucket}"
        if bucket == "no_usable_branch":
            return "I6_sleeptalk_callee_union", "triage: no usable branch (callee-union path)"
        # other buckets (structural counts, ratio findings) fall THROUGH to
        # the echo rules below instead of dead-ending — the v3 coverage run
        # showed a catch-all here starved the sweep-scale rules entirely.

    # magnitude-only heal differences with drain/leech context -> I3
    if cls.startswith("component_magnitude:heal") and ("Leech Seed" in proto or "[silent]" in proto):
        return "I3_roll_inherited", "drain/leech heal capped by a roll-dependent HP (leech-cap family)"

    # Crit-arm pairing echo (sweep-scale rule; validation: s2000705/102 —
    # observed crit recoil -81 paired against the non-crit arm's -44; and
    # s2001179/136, crit KO ends the turn while survive-arms complain).
    # c13 could not validate this: its crit shapes were all majority-arm.
    ratio = None
    if len(obs_c) == 1 and len(eng_c) == 1 and eng_c[0][1] != 0:
        ratio = abs(obs_c[0][1]) / max(1, abs(eng_c[0][1]))
    if "|-crit|" in proto and (
            (ratio is not None and 1.5 <= ratio <= 2.2)
            or (not obs_c and eng_c and "faint" in proto)):
        return ("LS_crit_arm_pairing_echo",
                "observed crit outcome paired against the non-crit majority "
                "arm (branch-set accounting; the crit arm carries the shape)")

    # Same-turn stat/status boundary with a sub-window magnitude ratio —
    # WHAT-level candidate, surfaced not absorbed (validation: s2000261/31,
    # ratio 0.87 after a same-turn Calm Mind).
    if ratio is not None and 0.70 <= ratio <= 0.96 and (
            "|-boost|" in proto or "|-unboost|" in proto or "|-status|" in proto):
        return ("UNATTRIBUTED",
                f"majority magnitude ratio {ratio:.2f} on a same-turn "
                "boost/status boundary (WHAT-level candidate, WHY open)")

    # Structural-arm echo — deliberately LAST before the fallback: the
    # broadest rule, safe only because every narrower mechanism above has
    # already had its chance (the absorb-ordering lesson). A component-count
    # mismatch is only an echo when an actual sibling engine arm carries the
    # full observed component multiset. The prior s2000561/67 citation was
    # stale: its sibling arms do not carry the observed hit. The c14 archive
    # re-run therefore treats unsupported structural shapes as named WHAT
    # candidates rather than allowing this rule to absorb them.
    if obs_c is not None and eng_c is not None and len(obs_c) != len(eng_c) \
            and cls == "roll_scaled_component":
        if _sibling_arm_carries_observed_components(misses, majority, obs_c):
            return ("LS_structural_arm_echo",
                    "component count differs against the majority arm, but a sibling "
                    "engine arm exactly carries the observed components (branch-set accounting)")
        return ("UNATTRIBUTED",
                "structural component-count mismatch without a sibling engine arm "
                "carrying the observed components (WHAT-level candidate, WHY open)")

    return "UNATTRIBUTED", f"no documented signature matched (class {cls}; majority: {majority[:120]})"


def _triage_bucket(row) -> str | None:
    """Run the committed roll-component triage on one row (c9's instrument)."""
    try:
        from triage_roll_components import triage_row  # noqa: PLC0415
        result = triage_row(row, verbose=False)
        return result.get("bucket")
    except (KeyboardInterrupt, SystemExit):
        raise
    except BaseException:  # noqa: BLE001
        return None


def _attribute_roll_row(row, majority, obs_c, eng_c):
    """Roll-scaled rows: legal-set / window / best-branch accounting -> I2."""

    states = row.get("engine_states") or []
    choices = row.get("choices") or {}
    pre = row.get("pre_features") or {}
    if not states or not choices:
        return None
    try:
        state = poke_engine.State.from_string(states[0])
        s1r, s2r = poke_engine.calculate_damage(
            state, choices.get("p1", ""), choices.get("p2", ""), True)
        legal = legal_roll_damages(list(s1r) + list(s2r))
    except (KeyboardInterrupt, SystemExit):
        raise
    except BaseException:  # noqa: BLE001
        legal = set()

    obs_rolled = [(s, v) for s, v in obs_c if s in _ROLL_SCALED_SOURCES or s.endswith("_to_full")]
    eng_rolled = [(s, v) for s, v in eng_c if s in _ROLL_SCALED_SOURCES or s.endswith("_to_full")]
    if obs_rolled and len(obs_rolled) == len(eng_rolled):
        all_ok, any_window = True, False
        for (os_, ov), (es_, ev) in zip(sorted(obs_rolled, key=lambda x: x[1]),
                                        sorted(eng_rolled, key=lambda x: x[1])):
            o, e = abs(ov), abs(ev)
            if o == e or o in legal:
                continue
            if e and 0.919 * e - 1 <= o <= 1.09 * e + 1:
                any_window = True
                continue
            all_ok = False
        if all_ok:
            basis = "observed roll(s) in the engine's legal set" + (
                " / within the [0.919,1.09] window" if any_window else "")
            return "I2_matcher_accounting", basis
    if len(obs_rolled) != len(eng_rolled):
        return None  # structural: fall through to UNATTRIBUTED for replay
    return None


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--shards", nargs="+", required=True)
    ap.add_argument("--prediction", type=Path, required=True)
    ap.add_argument("--json", type=Path, default=None)
    args = ap.parse_args(argv)

    paths = []
    for pattern in args.shards:
        paths.extend(sorted(glob.glob(pattern)))
    agg = {"boundaries_measured": 0, "boundaries_full_round": 0,
           "transitions_matched": 0, "transitions_diverged": 0,
           "engine_errors": 0, "games": 0}
    classes: Counter = Counter()
    rows = []
    retention_ok = True
    for p in paths:
        shard = json.loads(Path(p).read_text())
        for k in agg:
            agg[k] += int(shard.get(k, 0))
        for c, n in (shard.get("divergence_classes") or {}).items():
            classes[c] += n
        rows.extend(shard.get("repros") or [])
        ret = shard.get("repro_retention") or {}
        if not ret.get("repros_complete"):
            retention_ok = False
    coverage = agg["boundaries_measured"] / max(1, agg["boundaries_full_round"])

    fam_counts: Counter = Counter()
    unattributed = []
    attributed_rows = []
    for row in rows:
        fam, basis = attribute_row(row)
        fam_counts[fam] += 1
        entry = {"seed": row.get("seed"), "step": row.get("step"),
                 "class": row.get("divergence_class"), "family": fam, "basis": basis}
        attributed_rows.append(entry)
        if fam == "UNATTRIBUTED":
            unattributed.append(entry)

    pred = json.loads(args.prediction.read_text())
    pred_classes = pred.get("predicted_class_rates_10k") or {}
    per_class = {}
    n = agg["boundaries_measured"]
    for cls in sorted(set(classes) | set(pred_classes)):
        k = classes.get(cls, 0)
        lo, hi = wilson(k, n)
        per_class[cls] = {
            "observed": k,
            "observed_wilson95_rate": [lo, hi],
            "predicted": (pred_classes.get(cls) or {}).get("expected_10k"),
            "predicted_wilson95_count": (pred_classes.get(cls) or {}).get("wilson95_count_10k"),
        }

    verdict = "PASS" if (not unattributed and agg["engine_errors"] == 0
                         and retention_ok and len(rows) == agg["transitions_diverged"]) else "FAIL"
    out = {
        "verdict": verdict,
        "aggregate": agg,
        "coverage_measured_fraction": round(coverage, 4),
        "repros_complete_all_shards": retention_ok,
        "rows_retained": len(rows),
        "family_attribution": dict(fam_counts.most_common()),
        "unattributed_rows": unattributed,
        "per_class_observed_vs_predicted": per_class,
    }
    print(f"VERDICT: {verdict}")
    print(f"games={agg['games']} boundaries={agg['boundaries_measured']} "
          f"diverged={agg['transitions_diverged']} engine_errors={agg['engine_errors']} "
          f"coverage={coverage:.4f} retained={len(rows)}")
    for fam, k in fam_counts.most_common():
        print(f"  {k:5d}  {fam}")
    if unattributed:
        print(f"UNATTRIBUTED: {len(unattributed)} rows (sweep FAILURE pending replay-first triage)")
    if args.json:
        args.json.write_text(json.dumps(out, indent=1))
        print(f"wrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
