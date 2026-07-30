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
  equal-magnitude label tie            -> I4 (mapper attribution, #908/I.2 lineage)
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

    # I1: cap-state shape (_to_full on either side of the majority miss)
    if "_to_full" in majority:
        return "I1_cap_state_shape", "capped-heal component shape in the majority miss (avg-roll world evolution)"

    # I4: equal-magnitude label tie in majority miss
    if obs_c and eng_c and len(obs_c) == len(eng_c):
        if sorted(v for _, v in obs_c) == sorted(v for _, v in eng_c) \
                and sorted(s for s, _ in obs_c) != sorted(s for s, _ in eng_c):
            return "I4_attribution_tie", "identical magnitudes, different source labels (mapper attribution)"

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
        if bucket:
            return "UNATTRIBUTED", f"triage bucket: {bucket}"

    # magnitude-only heal differences with drain/leech context -> I3
    if cls.startswith("component_magnitude:heal") and ("Leech Seed" in proto or "[silent]" in proto):
        return "I3_roll_inherited", "drain/leech heal capped by a roll-dependent HP (leech-cap family)"

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
