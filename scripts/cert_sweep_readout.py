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

Final certification contracts set ``requires_execution_contract=true`` and
also require ``--execution-manifest``. The manifest binds each shard's durable
completion marker and report hash to the registered source, engine, contract,
readout, and behavioral-probe provenance. Legacy evidence regeneration remains
available only for older prediction artifacts that do not require that
execution contract.
"""

from __future__ import annotations

import argparse
import glob
import hashlib
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
from pokezero.audit_provenance import public_repo_commit  # noqa: E402

from engine_transition_differential import (  # noqa: E402
    CHECKPOINT_SCHEMA,
    checkpoint_report_binding_failures,
    _ROLL_SCALED_SOURCES,
    damage_components,
    legal_roll_damages,
    _split_components,
)
from cert_execution_manifest import (  # noqa: E402
    EMITTABLE_DOCUMENTED_FAMILIES,
    EMITTABLE_EXCLUSION_COUNTERS,
    validate_execution_manifest_schema,
    validate_final_contract_schema,
    validate_predicted_class_rates,
)

_PCT_RE = re.compile(r"pct=([\d.]+)")
_PAIR_RE = re.compile(r"\('([^']*)',\s*(-?\d+)\)")
_MISS_SIDE_RE = re.compile(r"\b(p[12])\s+(?:attributed|roll-scaled) components differ")
_PROBE_PASS_RE = re.compile(r"^\[[^]]+\] PASS\b", re.MULTILINE)
_PROBE_FAIL_RE = re.compile(r"^\[[^]]+\] FAIL\b", re.MULTILINE)

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
    misses: Sequence[str], anchor: str, observed: Sequence[tuple[str, int]]
) -> bool:
    """Whether a non-majority engine arm exactly reproduces the observed components.

    A count mismatch alone says nothing about why the majority arm differs.  The
    structural-echo family is valid only when another engine arm actually
    carries the full observed component multiset.  Empty observations are not
    evidence of a carried shape.
    """

    observed_components = Counter(observed)
    anchor_side = _MISS_SIDE_RE.search(anchor)
    if not observed_components or anchor_side is None:
        return False
    for miss in misses:
        if miss == anchor:
            continue
        sibling_side = _MISS_SIDE_RE.search(miss)
        if sibling_side is None or sibling_side.group(1) != anchor_side.group(1):
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

    # These candidates must outrank I2 even when the matching branch has only
    # minority mass. I2 explains branch-set accounting; it cannot explain a
    # concrete engine behavior that a small branch exposes.
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

    if "|-boost|" in proto or "|-unboost|" in proto or "|-status|" in proto:
        for miss in misses:
            miss_obs, miss_eng = _miss_pairs(miss)
            if len(miss_obs) == 1 and len(miss_eng) == 1 and miss_eng[0][1] != 0:
                miss_ratio = abs(miss_obs[0][1]) / max(1, abs(miss_eng[0][1]))
                if 0.70 <= miss_ratio <= 0.96:
                    return ("UNATTRIBUTED",
                            f"majority magnitude ratio {miss_ratio:.2f} on a same-turn "
                            "boost/status boundary (WHAT-level candidate, WHY open)")

    if cls == "roll_scaled_component":
        for miss in misses:
            miss_obs, miss_eng = _miss_pairs(miss)
            if len(miss_obs) != len(miss_eng):
                if _sibling_arm_carries_observed_components(misses, miss, miss_obs):
                    return ("LS_structural_arm_echo",
                            "component count differs against an engine arm, but a sibling "
                            "engine arm exactly carries the observed components (branch-set accounting)")
                return ("UNATTRIBUTED",
                        "structural component-count mismatch without a sibling engine arm "
                        "carrying the observed components (WHAT-level candidate, WHY open)")

    known_class = cls in ("roll_scaled_component", "mapper_lossy", "no_usable_branch") or cls.startswith(
        ("limit:", "component_missing_in_engine:", "component_extra_in_engine:",
         "component_mismatch:", "component_magnitude:", "movepainsplit")
    )
    if not known_class:
        return "UNATTRIBUTED", f"no documented signature matched (unknown class {cls})"

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

    return "UNATTRIBUTED", f"no documented signature matched (class {cls}; majority: {majority[:120]})"


def _unattributed_counter(basis: str) -> str:
    """Return the classifier-emitted exclusion counter for an unattributed row.

    The rate contract names these outcomes rather than historical mechanism
    labels.  This keeps a predicted-zero gate connected to a branch that can
    actually fire in ``attribute_row``.
    """

    if basis.startswith("recharge-turn boundary"):
        return "recharge_turn_residual_gap"
    if basis.startswith("Truant boundary"):
        return "truant_loaf_phase_drift"
    if basis.startswith("engine-only absorb heal"):
        return "absorb_through_protect_or_miss"
    if basis.startswith("recoil magnitude on a Substitute-breaking"):
        return "recoil_vs_substitute_basis"
    if basis.startswith("observed |cant|"):
        return "incapacitated_arm_pricing"
    if "same-turn boost/status boundary" in basis:
        return "same_turn_stat_event_gap"
    if basis.startswith("structural component-count mismatch"):
        return "structural_component_count_without_supported_sibling"
    return "unattributed_generic"


def classify_row(row: Mapping[str, Any]) -> tuple[str, str, str | None]:
    """Classify a row and expose the fail-closed exclusion it reached."""

    try:
        family, basis = attribute_row(row)
    except (AttributeError, TypeError, ValueError):
        family, basis = "UNATTRIBUTED", "malformed divergent-row classifier input"
    return family, basis, _unattributed_counter(basis) if family == "UNATTRIBUTED" else None


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


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _is_lower_hex(value: object, length: int) -> bool:
    return (
        isinstance(value, str)
        and len(value) == length
        and all(character in "0123456789abcdef" for character in value)
    )


def _strict_int(value: object) -> int | None:
    """Accept JSON integers, but never silently accept ``True`` as seed 1."""

    return value if type(value) is int else None


def _finite_number(value: object) -> float | None:
    if type(value) not in (int, float):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _non_overlapping_seed_total(
    blocks: Sequence[Mapping[str, int]], *, label: str, failures: list[str]
) -> int:
    """Reject overlap before treating per-block counts as a global seed total."""

    total = 0
    previous_max: int | None = None
    for block in sorted(blocks, key=lambda item: item["start"]):
        start, games = block["start"], block["games"]
        maximum = start + games - 1
        if previous_max is not None and start <= previous_max:
            failures.append(
                f"{label} seed blocks overlap at {start}..{maximum} after {previous_max}"
            )
        total += games
        previous_max = max(previous_max, maximum) if previous_max is not None else maximum
    return total


def _current_runtime_provenance() -> dict[str, Any]:
    """Evidence from the checkout executing this readout, not a contract field."""

    source_commit = public_repo_commit(REPO_ROOT)
    return {
        "source_commit": source_commit if _is_lower_hex(source_commit, 40) else None,
        "checkout": str(REPO_ROOT.resolve()),
        "readout_path": str(Path(__file__).resolve()),
        "readout_sha256": _sha256(Path(__file__).resolve()),
    }


def _checkout_commit(path: Path) -> str | None:
    return public_repo_commit(path)


def _file_evidence(
    value: object,
    *,
    label: str,
    failures: list[str],
    required_sha256: str | None = None,
) -> Mapping[str, Any] | None:
    """Validate a path-backed SHA-256 record, rather than trusting a string."""

    if not isinstance(value, Mapping):
        failures.append(f"{label} is absent")
        return None
    raw_path = value.get("path")
    digest = value.get("sha256")
    if not isinstance(raw_path, str) or not raw_path:
        failures.append(f"{label} has no artifact path")
        return None
    if not _is_lower_hex(digest, 64):
        failures.append(f"{label} has no valid SHA-256")
        return None
    path = Path(raw_path)
    if not path.is_file():
        failures.append(f"{label} artifact is missing: {path}")
        return None
    actual = _sha256(path)
    if actual != digest:
        failures.append(f"{label} SHA-256 does not match its artifact")
    if required_sha256 is not None and actual != required_sha256:
        failures.append(f"{label} does not match the registered SHA-256")
    return value


def _probe_evidence(
    value: object,
    *,
    label: str,
    failures: list[str],
    required_passes: int | None,
    branch_probe: bool = False,
) -> None:
    evidence = _file_evidence(value, label=label, failures=failures)
    if evidence is None:
        return
    if branch_probe:
        text = Path(str(evidence["path"])).read_text(encoding="utf-8", errors="replace")
        if evidence.get("passed") is not True or "[search-crate-branch-events] PASS" not in text \
                or _PROBE_FAIL_RE.search(text):
            failures.append(f"{label} did not pass")
        return
    passed, total = _strict_int(evidence.get("passed")), _strict_int(evidence.get("total"))
    text = Path(str(evidence["path"])).read_text(encoding="utf-8", errors="replace")
    actual_passed = len(_PROBE_PASS_RE.findall(text))
    actual_total = actual_passed + len(_PROBE_FAIL_RE.findall(text))
    if passed is None or total is None or required_passes is None:
        failures.append(f"{label} has malformed pass counts")
    elif passed != actual_passed or total != actual_total:
        failures.append(f"{label} pass counts do not match its log")
    elif passed != required_passes or total != required_passes:
        failures.append(f"{label} is not {required_passes}/{required_passes}")


def _checkpoint_provenance(
    value: object,
    *,
    label: str,
    failures: list[str],
    required_source: str,
    required_fingerprint: str,
    required_image: str,
    expected_seed_range: tuple[int, int],
    expected_records: int,
    expected_distinct_seeds: int,
    report: Mapping[str, Any],
) -> None:
    evidence = _file_evidence(value, label=label, failures=failures)
    if evidence is None:
        return
    path = Path(str(evidence["path"]))
    records = 0
    seeds: set[int] = set()
    checkpoint_rows: list[Mapping[str, Any]] = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as error:
        failures.append(f"{label} cannot be read: {error}")
        return
    for number, raw in enumerate(lines, start=1):
        if not raw.strip():
            continue
        try:
            record = json.loads(raw)
        except json.JSONDecodeError:
            failures.append(f"{label} has invalid JSON at record {number}")
            continue
        if not isinstance(record, Mapping):
            failures.append(f"{label} has a non-object record {number}")
            continue
        if record.get("schema") != CHECKPOINT_SCHEMA:
            failures.append(
                f"{label} record {number} does not use checkpoint schema {CHECKPOINT_SCHEMA}"
            )
            continue
        checkpoint_rows.append(record)
        seed = _strict_int(record.get("seed"))
        if seed is None or not expected_seed_range[0] <= seed <= expected_seed_range[1]:
            failures.append(f"{label} record {number} has a seed outside its shard band")
        elif seed in seeds:
            failures.append(f"{label} contains duplicate checkpoint seed {seed}")
        else:
            seeds.add(seed)
        provenance = record.get("provenance")
        if not isinstance(provenance, Mapping):
            failures.append(f"{label} record {number} has no resume provenance")
        elif (
            provenance.get("source_commit") != required_source
            or provenance.get("engine_fingerprint") != required_fingerprint
            or provenance.get("image_commit") != required_image
        ):
            failures.append(f"{label} record {number} resume provenance does not match contract")
        records += 1
    if records == 0:
        failures.append(f"{label} contains no completed-game records")
    if records != expected_records:
        failures.append(
            f"{label} has {records} checkpoint records, expected {expected_records} for its shard"
        )
    if len(seeds) != expected_distinct_seeds:
        failures.append(
            f"{label} has {len(seeds)} distinct checkpoint seeds, expected "
            f"{expected_distinct_seeds} for its shard"
        )
    for failure in checkpoint_report_binding_failures(checkpoint_rows, report):
        failures.append(f"{label}: {failure}")


def _contract_gates(
    *,
    paths: Sequence[Path],
    shards: Sequence[Mapping[str, Any]],
    contract: Mapping[str, Any],
    contract_path: Path,
    execution_manifest: Mapping[str, Any] | None,
    coverage: float,
    aggregate: Mapping[str, int],
    legacy_opt_out: bool = False,
    runtime_provenance: Mapping[str, Any] | None = None,
) -> tuple[list[str], dict[str, Any]]:
    """Validate a registered certification contract against file-backed evidence."""

    gates = contract.get("certification_gates")
    registered = contract.get("registered_before_launch") is True
    final = (
        registered
        or contract.get("requires_execution_contract") is True
        or "certification_gates" in contract
    )
    explicit_legacy = (
        legacy_opt_out
        and contract.get("legacy_contract_opt_out") is True
        and not registered
        and contract.get("requires_execution_contract") is not True
        and "certification_gates" not in contract
    )
    if not isinstance(gates, Mapping):
        if explicit_legacy:
            return [], {
                "enforced": False,
                "enforcement_status": "legacy-opt-out",
                "legacy_opt_out": True,
            }
        status = "refused-final-contract" if final else "refused-implicit-legacy"
        return [
            "certification gates are absent; pass --legacy-opt-out only with "
            "legacy_contract_opt_out=true on an unregistered legacy artifact"
        ], {"enforced": False, "enforcement_status": status, "legacy_opt_out": False}

    failures: list[str] = []
    evidence: dict[str, Any] = {"enforced": True, "enforcement_status": "enforced"}
    for error in validate_final_contract_schema(contract):
        failures.append(f"final contract schema violation: {error}")
    if not registered:
        failures.append("contract was not registered before launch")
    if contract.get("requires_execution_contract") is not True:
        failures.append("registered contract must set requires_execution_contract=true")

    expected_shards = _strict_int(gates.get("expected_shards"))
    expected_games = _strict_int(gates.get("expected_games"))
    minimum_coverage = _finite_number(gates.get("minimum_coverage_measured_fraction"))
    if expected_shards is None or expected_shards <= 0:
        failures.append("expected_shards must be a positive integer")
        expected_shards = 0
    if expected_games is None or expected_games <= 0:
        failures.append("expected_games must be a positive integer")
        expected_games = 0
    if minimum_coverage is None or not 0.0 < minimum_coverage <= 1.0:
        failures.append("minimum_coverage_measured_fraction must be in (0, 1]")
        minimum_coverage = 1.0
    if not math.isfinite(coverage) or coverage < 0.0 or coverage > 1.0:
        failures.append(f"coverage {coverage!r} is outside [0, 1]")
    elif coverage < minimum_coverage:
        failures.append(
            f"coverage {coverage:.6f} is below registered floor {minimum_coverage:.6f}"
        )

    expected_blocks_raw = gates.get("seed_blocks")
    expected_blocks: list[dict[str, int]] = []
    if not isinstance(expected_blocks_raw, list) or not expected_blocks_raw:
        failures.append("contract has no explicit seed_blocks")
    else:
        for index, block in enumerate(expected_blocks_raw):
            if not isinstance(block, Mapping):
                failures.append(f"seed_blocks[{index}] is not an object")
                continue
            start, games = _strict_int(block.get("start")), _strict_int(block.get("games"))
            if start is None or games is None or start < 0 or games <= 0:
                failures.append(f"seed_blocks[{index}] has malformed start/games")
                continue
            expected_blocks.append({"start": start, "games": games})
    expected_blocks.sort(key=lambda block: block["start"])
    expected_seed_total = _non_overlapping_seed_total(
        expected_blocks, label="registered", failures=failures
    )
    if len(expected_blocks) != expected_shards:
        failures.append(
            f"registered seed blocks contain {len(expected_blocks)} blocks, expected {expected_shards}"
        )
    if expected_seed_total != expected_games:
        failures.append(
            f"registered seed blocks contain {expected_seed_total} games, expected {expected_games}"
        )
    if len(paths) != expected_shards:
        failures.append(f"expected {expected_shards} shard reports, found {len(paths)}")
    if aggregate.get("games") != expected_games:
        failures.append(
            f"expected {expected_games} aggregate games, found {aggregate.get('games')}"
        )

    required_build = gates.get("required_build_check")
    required_matcher = gates.get("required_matcher")
    required_repros_per_game = _strict_int(gates.get("required_repros_per_game"))
    required_keep_repro = _strict_int(gates.get("required_keep_repro"))
    if required_repros_per_game is None or required_repros_per_game < 0:
        failures.append("required_repros_per_game must be a non-negative integer")
    if required_keep_repro is None or required_keep_repro < 0:
        failures.append("required_keep_repro must be a non-negative integer")

    actual_blocks: list[dict[str, int]] = []
    actual_shards: dict[int, tuple[Path, Mapping[str, Any], dict[str, int]]] = {}
    distinct_seed_total = 0
    for path, shard in zip(paths, shards):
        seeds = shard.get("seeds") if isinstance(shard, Mapping) else None
        if not isinstance(seeds, Mapping):
            failures.append(f"{path.name}: missing seed summary")
            continue
        start = _strict_int(seeds.get("min"))
        games = _strict_int(shard.get("games"))
        maximum = _strict_int(seeds.get("max"))
        distinct = _strict_int(seeds.get("distinct"))
        if None in (start, games, maximum, distinct):
            failures.append(f"{path.name}: malformed seed summary or games scalar")
            continue
        assert start is not None and games is not None and maximum is not None and distinct is not None
        block = {"start": start, "games": games, "max": maximum, "distinct": distinct}
        actual_blocks.append(block)
        if start in actual_shards:
            failures.append(f"{path.name}: repeats shard seed start {start}")
        else:
            actual_shards[start] = (path, shard, block)
        if games <= 0:
            failures.append(f"{path.name}: non-positive game count")
        expected_max = start + games - 1
        if maximum != expected_max or distinct != games:
            failures.append(
                f"{path.name}: seed population is not the complete contiguous range "
                f"{start}..{expected_max}"
            )
        if shard.get("build_check") != required_build:
            failures.append(f"{path.name}: build_check does not match contract")
        if shard.get("matcher") != required_matcher:
            failures.append(f"{path.name}: matcher does not match contract")
        if shard.get("acceptance_eligible") is not True:
            failures.append(f"{path.name}: shard is not acceptance_eligible")
        retention = shard.get("repro_retention")
        if not isinstance(retention, Mapping):
            failures.append(f"{path.name}: missing repro_retention")
        else:
            if _strict_int(retention.get("repros_per_game")) != required_repros_per_game:
                failures.append(f"{path.name}: repros_per_game does not match contract")
            if _strict_int(retention.get("keep_repro")) != required_keep_repro:
                failures.append(f"{path.name}: keep_repro does not match contract")
            if retention.get("repros_complete") is not True:
                failures.append(f"{path.name}: repro population is incomplete")
    normalized_actual_blocks = sorted(
        ({"start": block["start"], "games": block["games"]} for block in actual_blocks),
        key=lambda block: block["start"],
    )
    distinct_seed_total = _non_overlapping_seed_total(
        normalized_actual_blocks, label="observed", failures=failures
    )
    if normalized_actual_blocks != expected_blocks:
        failures.append("observed seed blocks do not exactly match the registered blocks")
    if distinct_seed_total != expected_games:
        failures.append(f"expected {expected_games} distinct seeds, found {distinct_seed_total}")

    required_source = gates.get("required_source_commit")
    required_fingerprint = gates.get("required_engine_fingerprint")
    required_readout_sha = gates.get("required_readout_sha256")
    required_image = gates.get("required_image_commit")
    required_producer_sha = gates.get("required_execution_manifest_producer_sha256")
    required_probe_passes = _strict_int(gates.get("required_behavioral_probe_passes"))
    for field, value, length in (
        ("required_source_commit", required_source, 40),
        ("required_engine_fingerprint", required_fingerprint, 64),
        ("required_readout_sha256", required_readout_sha, 64),
        ("required_image_commit", required_image, 40),
        ("required_execution_manifest_producer_sha256", required_producer_sha, 64),
    ):
        if not _is_lower_hex(value, length):
            failures.append(f"{field} is not a valid lowercase hash")
    if required_probe_passes is None or required_probe_passes <= 0:
        failures.append("required_behavioral_probe_passes must be a positive integer")
        required_probe_passes = None
    runtime = dict(runtime_provenance or _current_runtime_provenance())
    actual_contract_sha = _sha256(contract_path)
    if runtime.get("source_commit") != required_source:
        failures.append("current readout checkout commit does not match contract")
    if runtime.get("readout_sha256") != required_readout_sha:
        failures.append("executed readout hash does not match the registered hash")

    launch_registration = contract.get("launch_registration")
    if not isinstance(launch_registration, Mapping):
        failures.append("contract has no launch_registration provenance")
    else:
        fresh_measurements = launch_registration.get(
            "fresh_measurements_inspected_before_registration"
        )
        if type(fresh_measurements) is not int or fresh_measurements != 0:
            failures.append("contract does not prove zero fresh measurements inspected before registration")
        if launch_registration.get("coordinator_go") is not True:
            failures.append("contract has no explicit coordinator go")
        patch_count = _strict_int(launch_registration.get("engine_patch_count"))
        if patch_count is None or patch_count <= 0:
            failures.append("launch_registration engine_patch_count must be positive")

    evidence.update({
        "expected_shards": expected_shards,
        "expected_games": expected_games,
        "distinct_seed_total": distinct_seed_total,
        "seed_blocks": normalized_actual_blocks,
        "minimum_coverage_measured_fraction": minimum_coverage,
        "runtime_source_commit": runtime.get("source_commit"),
        "runtime_readout_sha256": runtime.get("readout_sha256"),
        "contract_sha256": actual_contract_sha,
    })
    if not isinstance(execution_manifest, Mapping):
        failures.append("certification contract requires an execution manifest")
        return failures, evidence
    if execution_manifest.get("schema") != "engine-cert-execution-manifest/2":
        failures.append("execution manifest schema is not engine-cert-execution-manifest/2")
        return failures, evidence
    for error in validate_execution_manifest_schema(execution_manifest):
        failures.append(f"execution manifest schema violation: {error}")

    producer = _file_evidence(
        execution_manifest.get("producer"), label="execution manifest producer",
        failures=failures,
        required_sha256=required_producer_sha if isinstance(required_producer_sha, str) else None,
    )
    if producer is not None and Path(str(producer["path"])).resolve() != (
            REPO_ROOT / "scripts" / "cert_execution_manifest.py").resolve():
        failures.append("execution manifest was not produced by this checkout's manifest producer")
    source = execution_manifest.get("source")
    if not isinstance(source, Mapping) or source.get("commit") != required_source:
        failures.append("execution manifest source checkout does not match contract")
    elif not isinstance(source.get("checkout"), str) or _checkout_commit(Path(source["checkout"])) != required_source:
        failures.append("execution manifest source checkout path does not resolve to its recorded commit")
    contract_blob = _file_evidence(
        execution_manifest.get("contract_blob"), label="execution manifest contract blob",
        failures=failures, required_sha256=actual_contract_sha,
    )
    readout_blob = _file_evidence(
        execution_manifest.get("readout_blob"), label="execution manifest readout blob",
        failures=failures, required_sha256=required_readout_sha if isinstance(required_readout_sha, str) else None,
    )
    if readout_blob is not None and runtime.get("readout_sha256") != _sha256(Path(str(readout_blob["path"]))):
        failures.append("execution manifest readout blob is not the executing readout")
    engine = execution_manifest.get("engine_provenance")
    if not isinstance(engine, Mapping):
        failures.append("execution manifest has no engine provenance")
    else:
        stamp = _file_evidence(engine.get("stamp"), label="engine build stamp", failures=failures)
        if engine.get("fingerprint") != required_fingerprint:
            failures.append("execution manifest engine fingerprint does not match contract")
        if stamp is not None:
            try:
                stamp_payload = json.loads(Path(str(stamp["path"])).read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                failures.append("engine build stamp is not readable JSON")
            else:
                if not isinstance(stamp_payload, Mapping) or stamp_payload.get("fingerprint") != required_fingerprint:
                    failures.append("engine build stamp fingerprint does not match contract")
                if stamp_payload.get("schema") != "pokezero-engine-build/2":
                    failures.append("engine build stamp does not attest both consumers")
                artifacts = stamp_payload.get("artifacts") if isinstance(stamp_payload, Mapping) else None
                if not isinstance(artifacts, Mapping) or set(artifacts) != {"poke_engine", "pokezero_search"}:
                    failures.append("engine build stamp has no two-consumer artifact identities")
                else:
                    for name, artifact in artifacts.items():
                        if not isinstance(artifact, Mapping):
                            failures.append(f"engine build stamp has malformed {name} artifact")
                            continue
                        module_path, module_sha = artifact.get("module_path"), artifact.get("module_sha256")
                        module = Path(module_path) if isinstance(module_path, str) else None
                        if module is None or not module.is_file() or not _is_lower_hex(module_sha, 64):
                            failures.append(f"engine build stamp {name} module artifact is absent")
                        elif _sha256(module) != module_sha:
                            failures.append(f"engine build stamp {name} module hash does not match artifact")

    aggregate = execution_manifest.get("aggregate_provenance")
    if not isinstance(aggregate, Mapping):
        failures.append("execution manifest has no aggregate provenance")
    else:
        _probe_evidence(
            aggregate.get("behavioral_probes"), label="aggregate behavioral probe log",
            failures=failures, required_passes=required_probe_passes,
        )
        _probe_evidence(
            aggregate.get("branch_events_probe"), label="aggregate branch-events probe log",
            failures=failures, required_passes=None, branch_probe=True,
        )

    manifest_shards = execution_manifest.get("shards")
    manifest_by_seed_start: dict[int, Mapping[str, Any]] = {}
    if not isinstance(manifest_shards, list):
        failures.append("execution manifest has no shard list")
        manifest_shards = []
    for entry in manifest_shards:
        seed_start = entry.get("seed_start") if isinstance(entry, Mapping) else None
        if not isinstance(entry, Mapping) or _strict_int(seed_start) is None:
            failures.append("execution manifest contains an invalid shard entry")
            continue
        assert isinstance(seed_start, int)
        if seed_start in manifest_by_seed_start:
            failures.append(f"execution manifest repeats shard seed_start {seed_start}")
        manifest_by_seed_start[seed_start] = entry
    if len(manifest_by_seed_start) != expected_shards:
        failures.append(
            f"execution manifest contains {len(manifest_by_seed_start)} unique shard entries, "
            f"expected {expected_shards}"
        )
    actual_seed_starts = set(actual_shards)
    if set(manifest_by_seed_start) != actual_seed_starts:
        failures.append("execution manifest seed-start set does not match supplied shard reports")
    for seed_start in sorted(actual_shards):
        path, shard, block = actual_shards[seed_start]
        entry = manifest_by_seed_start.get(block["start"])
        if entry is None:
            continue
        report = _file_evidence(entry.get("report"), label=f"{path.name} report", failures=failures)
        if report is not None and _sha256(path) != _sha256(Path(str(report["path"]))):
            failures.append(f"{path.name}: supplied report differs from manifest artifact")
        _file_evidence(entry.get("completion_marker"), label=f"{path.name} completion marker", failures=failures)
        if entry.get("image_commit") != required_image:
            failures.append(f"{path.name}: per-shard image commit does not match contract")
        _probe_evidence(
            entry.get("behavioral_probes"), label=f"{path.name} behavioral probe log",
            failures=failures, required_passes=required_probe_passes,
        )
        _probe_evidence(
            entry.get("branch_events_probe"), label=f"{path.name} branch-events probe log",
            failures=failures, required_passes=None, branch_probe=True,
        )
        _checkpoint_provenance(
            entry.get("checkpoint"), label=f"{path.name} checkpoint", failures=failures,
            required_source=str(required_source), required_fingerprint=str(required_fingerprint),
            required_image=str(required_image),
            expected_seed_range=(block["start"], block["max"]),
            expected_records=block["games"], expected_distinct_seeds=block["distinct"],
            report=shard,
        )
        report_provenance = shard.get("checkpoint_provenance")
        if not isinstance(report_provenance, Mapping):
            failures.append(f"{path.name}: missing checkpoint_provenance")
        elif report_provenance.get("complete") is not True:
            failures.append(f"{path.name}: checkpoint_provenance is not complete")
        elif _strict_int(report_provenance.get("records_with_provenance")) != block["games"]:
            failures.append(
                f"{path.name}: checkpoint_provenance records_with_provenance "
                "does not match games"
            )
    return failures, evidence


def _family_rate_gates(
    family_counts: Mapping[str, int],
    contract: Mapping[str, Any],
    *,
    boundaries_measured: int,
    exclusion_counts: Mapping[str, int] | None = None,
) -> tuple[list[str], dict[str, Any]]:
    """Bind registered upper rates and predicted-zero counters.

    Wilson lower bounds estimated from the calibration corpus are descriptive,
    not a prediction that a fresh corpus must reproduce at the same minimum
    rate. A separately declared ``prediction_interval_rate`` is the opt-in
    route for a genuinely pre-registered two-sided prediction interval.
    """

    if not isinstance(contract.get("certification_gates"), Mapping):
        status = (
            "refused-final-contract"
            if "certification_gates" in contract
            else "legacy-opt-out"
        )
        return [], {"enforced": False, "enforcement_status": status}
    table = contract.get("pre_registered_family_rate_table")
    failures: list[str] = []
    evidence: dict[str, Any] = {"enforced": True, "families": {}}
    if not isinstance(table, Mapping):
        return ["missing pre_registered_family_rate_table"], evidence
    registered = table.get("documented_families")
    if not isinstance(registered, Mapping):
        return ["pre_registered_family_rate_table has no documented_families"], evidence
    if type(boundaries_measured) is not int or boundaries_measured <= 0:
        failures.append("family-rate gate requires positive boundaries_measured")
    calibration_boundaries = _strict_int(table.get("calibration_boundaries"))
    if calibration_boundaries is None:
        calibration_boundaries = -1

    observed_families = set(family_counts) - {"UNATTRIBUTED"}
    for family in observed_families - set(registered):
        failures.append(f"attributed family {family!r} was not pre-registered")
    for family, prediction in registered.items():
        if not isinstance(family, str) or not (
            family in EMITTABLE_DOCUMENTED_FAMILIES
            or (family.startswith("limit:") and len(family) > len("limit:"))
        ):
            failures.append(f"registered family {family!r} cannot be emitted by the classifier")
            continue
        if not isinstance(prediction, Mapping):
            failures.append(f"registered family {family!r} has no prediction object")
            continue
        rate_interval = prediction.get("wilson95_rate")
        if rate_interval is None:
            count_interval = prediction.get("wilson95")
            if (
                isinstance(count_interval, list)
                and len(count_interval) == 2
                and all(_finite_number(value) is not None for value in count_interval)
                and calibration_boundaries > 0
            ):
                rate_interval = [
                    float(count_interval[0]) / calibration_boundaries,
                    float(count_interval[1]) / calibration_boundaries,
                ]
        elif not (
            isinstance(rate_interval, list)
            and len(rate_interval) == 2
            and all(_finite_number(value) is not None for value in rate_interval)
        ):
            failures.append(
                f"registered family {family!r} has an invalid wilson95_rate interval"
            )
            continue
        if not (
            isinstance(rate_interval, list)
            and len(rate_interval) == 2
            and all(_finite_number(value) is not None for value in rate_interval)
        ):
            failures.append(
                f"registered family {family!r} has no registered Wilson rate interval"
            )
            continue
        if not (
            0.0 <= float(rate_interval[0]) <= float(rate_interval[1]) <= 1.0
        ):
            failures.append(
                f"registered family {family!r} has an invalid Wilson rate interval"
            )
            continue
        lower_rate, upper_rate = float(rate_interval[0]), float(rate_interval[1])
        prediction_interval = prediction.get("prediction_interval_rate")
        has_prediction_interval = (
            isinstance(prediction_interval, list)
            and len(prediction_interval) == 2
            and all(_finite_number(value) is not None for value in prediction_interval)
            and 0.0 <= float(prediction_interval[0]) <= float(prediction_interval[1]) <= 1.0
        )
        if prediction_interval is not None and not has_prediction_interval:
            failures.append(f"registered family {family!r} has an invalid prediction interval")
            continue
        count = _strict_int(family_counts.get(family, 0))
        if count is None or count < 0:
            failures.append(f"family {family!r} has a malformed observed count")
            count = 0
        observed_rate = count / max(1, boundaries_measured)
        evidence["families"][family] = {
            "observed": int(count),
            "observed_rate_per_measured_boundary": observed_rate,
            "registered_wilson95_rate": rate_interval,
            "lower_rate_advisory": lower_rate,
            "upper_rate_enforced": upper_rate,
            "prediction_interval_rate": prediction_interval if has_prediction_interval else None,
        }
        if has_prediction_interval and observed_rate < float(prediction_interval[0]):
            failures.append(
                f"registered family {family!r} rate {observed_rate:.8g} is below "
                f"pre-registered prediction lower rate {float(prediction_interval[0]):.8g}"
            )
        if has_prediction_interval and observed_rate > float(prediction_interval[1]):
            failures.append(
                f"registered family {family!r} rate {observed_rate:.8g} exceeds "
                f"pre-registered prediction upper rate {float(prediction_interval[1]):.8g}"
            )
        if observed_rate > upper_rate:
            failures.append(
                f"registered family {family!r} rate {observed_rate:.8g} exceeds "
                f"registered upper rate {upper_rate:.8g}"
            )

    predicted_zero = table.get("new_mechanisms_post_fix")
    if not isinstance(predicted_zero, Mapping):
        failures.append("pre_registered_family_rate_table has no new_mechanisms_post_fix")
    else:
        counters = exclusion_counts or {}
        for mechanism, prediction in predicted_zero.items():
            if not isinstance(prediction, Mapping) or prediction.get("predicted_next") != 0:
                failures.append(f"post-fix mechanism {mechanism!r} is not registered at zero")
                continue
            counter = prediction.get("exclusion_counter")
            if prediction.get("classifier_outcome") != "UNATTRIBUTED" or counter not in EMITTABLE_EXCLUSION_COUNTERS:
                failures.append(
                    f"post-fix mechanism {mechanism!r} has no emittable classifier exclusion counter"
                )
                continue
            observed = _strict_int(counters.get(counter, 0))
            if observed is None or observed < 0:
                failures.append(f"exclusion counter {counter!r} has a malformed observed count")
                observed = 0
            evidence["families"][mechanism] = {
                "observed": observed,
                "registered_expected": 0,
                "classifier_outcome": "UNATTRIBUTED",
                "exclusion_counter": counter,
            }
            if observed:
                failures.append(
                    f"predicted-zero post-fix mechanism {mechanism!r} observed {observed} times "
                    f"through exclusion counter {counter!r}"
                )
    return failures, evidence


def _repro_integrity_gates(
    rows: Sequence[Mapping[str, Any]],
    shards: Sequence[Mapping[str, Any]],
) -> list[str]:
    failures: list[str] = []
    identities: list[tuple[int, int]] = []
    ranges: list[tuple[int, int]] = []
    for shard in shards:
        seeds = shard.get("seeds")
        if isinstance(seeds, Mapping):
            start, end = _strict_int(seeds.get("min")), _strict_int(seeds.get("max"))
            if start is None or end is None:
                failures.append("shard has a malformed seed band")
                continue
            ranges.append((start, end))
    for index, row in enumerate(rows):
        seed, step = row.get("seed"), row.get("step")
        if type(seed) is not int or type(step) is not int:
            failures.append(f"retained repro {index} has a malformed seed/step identity")
            continue
        identities.append((seed, step))
        shard_start = _strict_int(row.get("_cert_shard_seed_start"))
        own_ranges = [band for band in ranges if band[0] == shard_start] if shard_start is not None else ranges
        if not any(start <= seed <= end for start, end in own_ranges):
            failures.append(
                f"retained repro identity ({seed}, {step}) is outside its shard seed band"
            )
    duplicate_count = len(identities) - len(set(identities))
    if duplicate_count:
        failures.append(
            f"retained repro population contains {duplicate_count} duplicate seed/step identities"
        )
    return failures


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--shards", nargs="+", required=True)
    ap.add_argument("--prediction", type=Path, required=True)
    ap.add_argument(
        "--execution-manifest",
        type=Path,
        default=None,
        help=(
            "file-backed engine-cert-execution-manifest/2 provenance; required by "
            "final certification contracts"
        ),
    )
    ap.add_argument(
        "--legacy-opt-out",
        action="store_true",
        help=(
            "allow an unregistered historical artifact only when it sets "
            "legacy_contract_opt_out=true; never bypasses final contract gates"
        ),
    )
    ap.add_argument("--json", type=Path, default=None)
    args = ap.parse_args(argv)

    paths = []
    for pattern in args.shards:
        paths.extend(sorted(glob.glob(pattern)))
    if not paths:
        ap.error("--shards matched no files")
    resolved_paths = [Path(path).resolve() for path in paths]
    if len(set(resolved_paths)) != len(resolved_paths):
        ap.error("--shards contains duplicate report paths")

    agg = {"boundaries_measured": 0, "boundaries_full_round": 0,
           "transitions_matched": 0, "transitions_diverged": 0,
           "engine_errors": 0, "games": 0}
    aggregate_counters: Counter = Counter()
    classes: Counter = Counter()
    rows: list[Mapping[str, Any]] = []
    shards: list[Mapping[str, Any]] = []
    input_failures: list[str] = []
    retention_ok = True
    for p in paths:
        try:
            loaded = json.loads(Path(p).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            input_failures.append(f"{Path(p).name}: unreadable shard JSON ({error})")
            loaded = {}
        if not isinstance(loaded, Mapping):
            input_failures.append(f"{Path(p).name}: shard root is not an object")
            loaded = {}
        shard = loaded
        shards.append(shard)
        for k in agg:
            value = _strict_int(shard.get(k))
            if value is None or value < 0:
                input_failures.append(
                    f"{Path(p).name}: missing or malformed aggregate scalar {k!r}"
                )
                continue
            agg[k] += value
        counters = shard.get("counters")
        if not isinstance(counters, Mapping):
            input_failures.append(f"{Path(p).name}: counters is not an object")
        else:
            for key, value in counters.items():
                number = _strict_int(value)
                if not isinstance(key, str) or number is None or number < 0:
                    input_failures.append(f"{Path(p).name}: malformed counter {key!r}")
                    continue
                aggregate_counters[key] += number
        divergence_classes = shard.get("divergence_classes")
        if not isinstance(divergence_classes, Mapping):
            input_failures.append(f"{Path(p).name}: divergence_classes is not an object")
        else:
            for cls, value in divergence_classes.items():
                number = _strict_int(value)
                if not isinstance(cls, str) or number is None or number < 0:
                    input_failures.append(f"{Path(p).name}: malformed divergence class {cls!r}")
                    continue
                classes[cls] += number
        repros = shard.get("repros")
        if not isinstance(repros, list) or not all(isinstance(row, Mapping) for row in repros):
            input_failures.append(f"{Path(p).name}: repros is not a list of objects")
        else:
            shard_start = _strict_int((shard.get("seeds") or {}).get("min"))
            for row in repros:
                tagged = dict(row)
                if shard_start is not None:
                    tagged["_cert_shard_seed_start"] = shard_start
                rows.append(tagged)
        ret = shard.get("repro_retention")
        if not isinstance(ret, Mapping) or ret.get("repros_complete") is not True:
            retention_ok = False
    if agg["boundaries_full_round"] == 0:
        coverage = 0.0 if agg["boundaries_measured"] == 0 else math.inf
    else:
        coverage = agg["boundaries_measured"] / agg["boundaries_full_round"]

    fam_counts: Counter = Counter()
    exclusion_counts: Counter = Counter()
    unattributed = []
    for row in rows:
        fam, basis, exclusion_counter = classify_row(row)
        fam_counts[fam] += 1
        entry = {"seed": row.get("seed"), "step": row.get("step"),
                 "class": row.get("divergence_class"), "family": fam, "basis": basis}
        if fam == "UNATTRIBUTED":
            entry["exclusion_counter"] = exclusion_counter
            unattributed.append(entry)
            assert exclusion_counter is not None
            exclusion_counts[exclusion_counter] += 1

    try:
        pred = json.loads(args.prediction.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        input_failures.append(f"prediction is unreadable JSON ({error})")
        pred = {}
    if not isinstance(pred, Mapping):
        input_failures.append("prediction root is not an object")
        pred = {}
    execution_manifest = None
    if args.execution_manifest is not None:
        try:
            candidate = json.loads(args.execution_manifest.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            input_failures.append(f"execution manifest is unreadable JSON ({error})")
        else:
            if isinstance(candidate, Mapping):
                execution_manifest = candidate
            else:
                input_failures.append("execution manifest root is not an object")
    predicted_classes_value = pred.get("predicted_class_rates_10k")
    for error in validate_predicted_class_rates(predicted_classes_value):
        input_failures.append(error)
    pred_classes = predicted_classes_value or {}
    if not isinstance(pred_classes, Mapping):
        input_failures.append("predicted_class_rates_10k is not an object")
        pred_classes = {}
    else:
        invalid_prediction_classes = [key for key in pred_classes if not isinstance(key, str)]
        for key in invalid_prediction_classes:
            input_failures.append(f"predicted class key {key!r} is not a string")
        pred_classes = {
            key: value for key, value in pred_classes.items() if isinstance(key, str)
        }
    per_class = {}
    n = agg["boundaries_measured"]
    for cls in sorted(set(classes) | set(pred_classes)):
        k = classes.get(cls, 0)
        lo, hi = wilson(k, n)
        prediction = pred_classes.get(cls)
        if not isinstance(prediction, Mapping):
            input_failures.append(f"predicted class {cls!r} is not an object")
            prediction = {}
        per_class[cls] = {
            "observed": k,
            "observed_wilson95_rate": [lo, hi],
            "predicted": prediction.get("expected_10k"),
            "predicted_wilson95_count": prediction.get("wilson95_count_10k"),
        }

    contract_failures, contract_evidence = _contract_gates(
        paths=[Path(path) for path in paths],
        shards=shards,
        contract=pred,
        contract_path=args.prediction.resolve(),
        execution_manifest=execution_manifest,
        coverage=coverage,
        aggregate=agg,
        legacy_opt_out=args.legacy_opt_out,
    )
    family_failures, family_evidence = _family_rate_gates(
        fam_counts,
        pred,
        boundaries_measured=agg["boundaries_measured"],
        exclusion_counts=exclusion_counts,
    )
    repro_failures = _repro_integrity_gates(rows, shards)
    gate_failures = input_failures + contract_failures + family_failures + repro_failures
    verdict = "PASS" if (
        not unattributed
        and agg["engine_errors"] == 0
        and retention_ok
        and len(rows) == agg["transitions_diverged"]
        and not gate_failures
    ) else "FAIL"
    skip_counters = {
        key: value for key, value in sorted(aggregate_counters.items())
        if key.startswith("skip:")
    }
    full_rounds = max(1, agg["boundaries_full_round"])
    skip_counter_rates = {
        key: value / full_rounds for key, value in skip_counters.items()
    }
    out = {
        "verdict": verdict,
        "aggregate": agg,
        "coverage_measured_fraction": round(coverage, 4),
        "unmeasured_full_round_fraction": round(max(0.0, 1.0 - coverage), 4),
        "skip_counters": skip_counters,
        "skip_counter_rates_per_full_round": skip_counter_rates,
        "repros_complete_all_shards": retention_ok,
        "rows_retained": len(rows),
        "repro_integrity_failures": repro_failures,
        "family_attribution": dict(fam_counts.most_common()),
        "exclusion_counters": dict(exclusion_counts.most_common()),
        "unattributed_rows": unattributed,
        "per_class_observed_vs_predicted": per_class,
        "contract_evidence": contract_evidence,
        "family_rate_evidence": family_evidence,
        "enforcement_status": contract_evidence.get("enforcement_status"),
        "gate_failures": gate_failures,
    }
    print(f"VERDICT: {verdict}")
    print(f"games={agg['games']} boundaries={agg['boundaries_measured']} "
          f"diverged={agg['transitions_diverged']} engine_errors={agg['engine_errors']} "
          f"coverage={coverage:.4f} retained={len(rows)}")
    for fam, k in fam_counts.most_common():
        print(f"  {k:5d}  {fam}")
    if unattributed:
        print(f"UNATTRIBUTED: {len(unattributed)} rows (sweep FAILURE pending replay-first triage)")
    for failure in gate_failures:
        print(f"GATE FAILURE: {failure}")
    if args.json:
        args.json.write_text(json.dumps(out, indent=1))
        print(f"wrote {args.json}")
    return 0 if verdict == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
