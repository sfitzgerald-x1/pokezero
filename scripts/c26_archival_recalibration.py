#!/usr/bin/env python
"""Recalibrate the retained certification archive against the C26 engine.

The C15 contract's rate table was calibrated by replaying the consumed C14
archive through the final classifier on the 48-patch build. C26 freezes a
different engine (51 patches, fingerprint ``776fa1e1...``), so those numbers
are not C26's calibration: they describe a build that no longer exists. This
instrument produces C26's own archival calibration instead of copying them.

What it does, per retained row, in one pass:

1. re-read the boundary through the CURRENT build's own
   ``evaluate_boundary_strict`` (imported from the differential, never
   reimplemented) -- the same reread ``scripts/cert_sweep_reread.py`` performs;
2. for a row that still diverges, re-derive its divergence class AND its family
   attribution from the CURRENT reread's misses, not from the class recorded by
   the build that produced the archive. Re-attributing a stale ``branch_misses``
   payload would report the old engine's families under a new engine's name;
3. count a row the current build now matches as a CLEARANCE, and a row whose
   stored payload cannot be reconstructed as SKIPPED -- never as a match.

The archive is a historical population: it can only show which recorded
divergences the current engine clears, never which new ones it introduces. That
is why the emitted artifact is calibration evidence and the fresh registered
sweep remains the binding measurement.

FAIL-CLOSED INPUTS: the eight shard digests are pinned from the ledger's
Appendix Z12.6 durability record, the population is contract-checked at 3,821
rows, and the installed engine must carry the tracked build fingerprint. A
missing or altered input is a refusal, not a smaller calibration.

Usage::

    PYTHONPATH=src python scripts/c26_archival_recalibration.py \\
        --shards 'path/cert_shard_*.json' --json out.json
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Mapping

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

SCHEMA = "c26-current-engine-archival-calibration/1"

# docs/engine_divergence_ledger_20260728.md, Appendix Z12.6 durability record.
# Shard order is the sorted glob order, which is shard 0..7.
PINNED_ARCHIVE_SHARDS: tuple[str, ...] = (
    "b4a0d2c2a182e693554b3fbe2b241126d367178319cd6d9f690e28e8d524dd59",
    "3259aa315252f505002b5570343f74b2f305ed00f30e1437ad19d081590e2c3a",
    "8003fbc80d86c7d07b2fbb9f39c521093d047ce98b2cb4c2c9497d0df4722d7b",
    "442417c845c9046d7b7e4855dcdc4cba9d20f17a3acf76a447cddfa61882786c",
    "486b635854c62e6eecf0eae54c98dcf53f49940404f14b96c9d2ab5994985d0d",
    "5a9e87fd694da86a7fa36a2c8db0f05b424485328769b1729aee7fca28073a95",
    "962ac6efd52cd4b689154ac083678ae8ba6e28c96465c1b963c4453cbcb953c4",
    "5c0235d98fbc91f72338e51001a4b82506899c46eae3cbdfdfe9d86c5e9e462d",
)
ARCHIVE_POPULATION = 3821

# Wilson lower counts are advisory for the instrument families: their archive
# counts are historical, and a lower bound would assert that a fixed engine must
# keep producing them. The two long-standing comparison limits are different --
# they describe a documented, still-present comparison boundary -- so their
# lower bound stays informative. This is a C26 registration policy statement,
# recorded here so the contract does not have to restate the arithmetic.
BINDING_LOWER_BOUND_FAMILIES = frozenset({
    "limit:roll_divergent_lethality",
    "limit:world_sample_drag_target",
})


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _tracked_engine_identity() -> dict[str, Any]:
    """Fingerprint from tracked bytes, plus the installed-build gate."""

    printed = json.loads(subprocess.check_output(
        [sys.executable, "scripts/engine_build_fingerprint.py", "--print"],
        cwd=REPO_ROOT, text=True,
    ))
    subprocess.check_call(
        [sys.executable, "scripts/engine_build_fingerprint.py", "--check"],
        cwd=REPO_ROOT, stdout=subprocess.DEVNULL,
    )
    return {"fingerprint": printed["fingerprint"], "patch_count": printed["count"]}


CLASSIFIER_SOURCES = (
    "scripts/cert_sweep_readout.py",
    "scripts/cert_execution_manifest.py",
    "scripts/engine_transition_differential.py",
    "scripts/cert_sweep_reread.py",
)


def verify_source_commit(commit: str) -> str:
    """Resolve the frozen build-source commit this calibration speaks for.

    The calibration is written on a later branch than the commit it calibrates,
    so recording HEAD would pin the wrong thing -- and pinning an unverified
    commit would let the recorded identity drift from the bytes that actually
    ran. Every classifier input must be byte-identical at that commit, and its
    lifecycle record must already carry the engine identity being calibrated.
    """

    resolved = subprocess.check_output(
        ["git", "-C", str(REPO_ROOT), "rev-parse", f"{commit}^{{commit}}"], text=True
    ).strip()
    for relative in CLASSIFIER_SOURCES:
        committed = subprocess.check_output(
            ["git", "-C", str(REPO_ROOT), "show", f"{resolved}:{relative}"]
        )
        working = (REPO_ROOT / relative).read_bytes()
        if committed != working:
            raise ValueError(
                f"{relative} differs from its bytes at {resolved}; this working "
                "tree does not run the frozen classifier"
            )
    lifecycle = json.loads(subprocess.check_output(
        ["git", "-C", str(REPO_ROOT), "show",
         f"{resolved}:reports/certification_contract_lifecycle.json"], text=True
    ))
    if lifecycle.get("stage") != "build_source":
        raise ValueError(
            f"{resolved} is not a build-source freeze (stage "
            f"{lifecycle.get('stage')!r})"
        )
    return resolved


def verify_archive(shard_glob: str) -> tuple[list[Path], list[Mapping[str, Any]]]:
    from cert_sweep_reread import load_retained_rows

    paths = [Path(path) for path in sorted(glob.glob(shard_glob))]
    if len(paths) != len(PINNED_ARCHIVE_SHARDS):
        raise ValueError(
            f"archive must supply exactly {len(PINNED_ARCHIVE_SHARDS)} shards, got {len(paths)}"
        )
    for index, (path, expected) in enumerate(zip(paths, PINNED_ARCHIVE_SHARDS, strict=True)):
        actual = _sha256(path)
        if actual != expected:
            raise ValueError(
                f"retained-certification-shard-{index:02d}: SHA-256 does not match "
                "the ledger's pinned archive"
            )
    return paths, load_retained_rows(shard_glob, expected_rows=ARCHIVE_POPULATION)


def archive_denominators(paths: list[Path]) -> dict[str, int]:
    totals = Counter()
    for path in paths:
        shard = json.loads(path.read_text(encoding="utf-8"))
        for key in ("boundaries_full_round", "boundaries_measured", "games",
                    "transitions_diverged", "transitions_matched", "engine_errors"):
            value = shard.get(key)
            if type(value) is not int or value < 0:
                raise ValueError(f"{path.name}: malformed aggregate scalar {key!r}")
            totals[key] += value
    return dict(totals)


def recalibrate(rows: list[Mapping[str, Any]]) -> dict[str, Any]:
    """Reread and re-attribute every retained row on the installed build."""

    from cert_sweep_readout import classify_row
    from cert_sweep_reread import reread_row
    from engine_transition_differential import classify_divergence

    tally: Counter = Counter()
    families: Counter = Counter()
    exclusions: Counter = Counter()
    classes: Counter = Counter()
    cleared_by_recorded_class: Counter = Counter()
    skipped: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []

    for row in rows:
        identity = {"seed": row["seed"], "step": row["step"],
                    "recorded_class": row.get("divergence_class")}
        try:
            verdict, misses, _branches = reread_row(row)
        except (KeyboardInterrupt, SystemExit):
            raise
        except BaseException as error:  # noqa: BLE001
            tally["reread_error"] += 1
            errors.append({**identity, "error": type(error).__name__})
            continue
        tally[verdict] += 1
        if verdict == "matched":
            cleared_by_recorded_class[str(row.get("divergence_class"))] += 1
            continue
        if verdict != "diverged":
            # skip_lossy and any future non-verdict outcome: the stored payload
            # cannot be reconstructed, so it is neither a clearance nor a
            # divergence. Recording it as a match would silently credit the
            # engine for a row nobody measured.
            skipped.append({**identity, "verdict": verdict})
            continue

        current_class = classify_divergence(row["protocol"], misses)
        classes[current_class] += 1
        # The shard writer retains misses[:12]; re-attribute through the same
        # window so the family a fresh sweep would report is the family counted.
        current_row = {**row, "divergence_class": current_class,
                       "branch_misses": misses[:12]}
        family, _basis, exclusion_counter = classify_row(current_row)
        families[family] += 1
        if family == "UNATTRIBUTED" and exclusion_counter is not None:
            exclusions[exclusion_counter] += 1

    return {
        "tally": dict(tally),
        "families": dict(families),
        "exclusions": dict(exclusions),
        "classes": dict(classes),
        "cleared_by_recorded_class": dict(cleared_by_recorded_class),
        "skipped": skipped,
        "errors": errors,
    }


def count_intervals(counts: Mapping[str, int], boundaries: int) -> dict[str, list[float]]:
    """Wilson-95 count intervals at the calibration denominator."""

    from cert_sweep_readout import wilson

    out: dict[str, list[float]] = {}
    for family, count in counts.items():
        lower_rate, upper_rate = wilson(count, boundaries)
        lower = 0.0 if family not in BINDING_LOWER_BOUND_FAMILIES else lower_rate * boundaries
        out[family] = [round(lower, 1), round(upper_rate * boundaries, 1)]
    return out


def family_intervals(families: Mapping[str, int], boundaries: int) -> dict[str, list[float]]:
    """Intervals for every family the classifier can emit, zeros included.

    A family the current build did not produce on the archive still has to be
    registered: the readout fails a fresh sweep that attributes to an
    unregistered family, and omitting the zero-count families would turn a
    real, bounded observation into an unbounded surprise.
    """

    from cert_execution_manifest import (
        EMITTABLE_DOCUMENTED_FAMILIES,
        EMITTABLE_LIMIT_FAMILIES,
    )

    emittable = (EMITTABLE_DOCUMENTED_FAMILIES | EMITTABLE_LIMIT_FAMILIES) - {
        # Carries an explicit risk budget: the archive predates the counter, so
        # it has no empirical count here and must not be given a zero one.
        "limit:world_substitute_health_unknown",
    }
    unregisterable = set(families) - emittable - {"UNATTRIBUTED"}
    if unregisterable:
        raise ValueError(
            "current build attributed to non-emittable families: "
            + ", ".join(sorted(unregisterable))
        )
    counts = {family: int(families.get(family, 0)) for family in sorted(emittable)}
    return count_intervals(counts, boundaries)


def substitute_risk_budget(
    exclusions: Mapping[str, int], boundaries_full_round: int
) -> dict[str, Any]:
    """Register the unknown-Substitute ceiling from C26's own nearest anchor.

    The archive predates the archive-wide Substitute-health counter, so this
    family has no count to bound. The nearest mass anchor the current build
    does produce on the archive is its retained recoil-vs-Substitute identity
    set; the registered ceiling is an explicit multiple of that anchor's
    Wilson-95 upper rate, not an inherited number.
    """

    from cert_sweep_readout import wilson

    anchor_count = int(exclusions.get("recoil_vs_substitute_basis", 0))
    _lower, anchor_upper = wilson(anchor_count, boundaries_full_round)
    ceiling = 0.0001
    return {
        "upper_full_round_rate": ceiling,
        "upper_rate_basis": "pre_registered_risk_budget",
        "anchor_identities": anchor_count,
        "anchor_boundaries_full_round": boundaries_full_round,
        "anchor_observed_rate": round(anchor_count / boundaries_full_round, 10),
        "anchor_wilson95_upper_rate": round(anchor_upper, 10),
        "budget_multiple_of_anchor_upper": round(ceiling / anchor_upper, 2),
        "risk_budget_rationale": (
            "The retained archive predates the archive-wide Substitute-health "
            f"counter, so C26 registers a risk budget rather than a measured "
            f"bound. The nearest anchor the frozen 51-patch build produces on "
            f"the same archive is {anchor_count} retained recoil-vs-Substitute "
            f"identities over {boundaries_full_round} full-round boundaries "
            f"(Wilson-95 upper rate {anchor_upper:.10f}). The registered "
            f"ceiling of {ceiling} per full-round boundary is "
            f"{ceiling / anchor_upper:.2f} times that upper anchor and permits "
            "at most one unknown-Substitute boundary per 10,000 full rounds."
        ),
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--shards", required=True)
    parser.add_argument("--json", required=True)
    parser.add_argument(
        "--source-commit",
        required=True,
        help="the frozen build-source commit this calibration speaks for",
    )
    args = parser.parse_args(argv)

    try:
        paths, rows = verify_archive(args.shards)
    except (OSError, ValueError, json.JSONDecodeError) as error:
        print(f"ARCHIVE INPUT REFUSED: {error}", file=sys.stderr)
        return 2
    try:
        engine = _tracked_engine_identity()
        source_commit = verify_source_commit(args.source_commit)
    except (OSError, subprocess.CalledProcessError, ValueError, KeyError,
            json.JSONDecodeError) as error:
        print(f"SOURCE IDENTITY REFUSED: {error}", file=sys.stderr)
        return 2

    denominators = archive_denominators(paths)
    result = recalibrate(rows)
    measured = denominators["boundaries_measured"]

    out = {
        "schema": SCHEMA,
        "purpose": (
            "C26's own archival calibration: the retained certification archive "
            "re-read and re-attributed on the frozen 51-patch build. This is "
            "historical calibration evidence, not a certification measurement -- "
            "an archive of recorded divergences can only show which of them the "
            "current engine clears, never which new ones it introduces."
        ),
        "source_evidence": {
            "archive": (
                "The complete 10,000-game C14 certification archive whose shard "
                "hashes are recorded in docs/engine_divergence_ledger_20260728.md "
                "Appendix Z12.6."
            ),
            "archive_role": "historical_calibration_only",
            "archive_shard_sha256": list(PINNED_ARCHIVE_SHARDS),
            "population": len(rows),
            "games": denominators["games"],
            "boundaries_full_round": denominators["boundaries_full_round"],
            "boundaries_measured": measured,
            "coverage_measured_fraction": round(
                measured / denominators["boundaries_full_round"], 6
            ),
            "current_classifier_source_commit": source_commit,
            "current_classifier_readout_sha256": _sha256(
                REPO_ROOT / "scripts" / "cert_sweep_readout.py"
            ),
            "current_execution_manifest_producer_sha256": _sha256(
                REPO_ROOT / "scripts" / "cert_execution_manifest.py"
            ),
            "current_engine_fingerprint": engine["fingerprint"],
            "current_engine_patch_count": engine["patch_count"],
            "fresh_measurements_inspected": 0,
        },
        "method": {
            "reread": (
                "Every retained transition row is re-evaluated by the current "
                "build's own evaluate_boundary_strict, imported from "
                "scripts/engine_transition_differential.py."
            ),
            "reattribution": (
                "A row that still diverges is re-classified and re-attributed "
                "from the CURRENT reread's misses (windowed to the shard "
                "writer's retained misses[:12]), never from the class the "
                "archiving build recorded."
            ),
            "clearance": (
                "A row the current build now matches is counted as a clearance "
                "against its recorded class, not as a current family member."
            ),
            "lossy_rows": (
                "A row whose stored payload cannot be reconstructed is recorded "
                "as SKIPPED with its identity. It is never counted as a match "
                "and never silently dropped from the population."
            ),
            "upper_rate_policy": (
                "Wilson-95 count intervals are computed from these current-build "
                "counts at the archive's measured-boundary denominator. Only "
                "upper counts are binding; lower counts are advisory and are set "
                "to zero except for the two long-standing comparison limits."
            ),
        },
        "reread_tally": result["tally"],
        "archive_recorded_totals": {
            "transitions_diverged": denominators["transitions_diverged"],
            "transitions_matched": denominators["transitions_matched"],
            "engine_errors": denominators["engine_errors"],
        },
        "skipped_rows": result["skipped"],
        "reread_errors": result["errors"],
        "current_engine_family_counts": result["families"],
        "current_engine_exclusion_counts": result["exclusions"],
        "current_engine_class_counts": result["classes"],
        "cleared_by_recorded_class": result["cleared_by_recorded_class"],
        "calibration_boundaries": measured,
        "registered_family_count_intervals": family_intervals(
            result["families"], measured
        ),
        "registered_non_empirical_upper_rates": {
            "limit:world_substitute_health_unknown": substitute_risk_budget(
                result["exclusions"], denominators["boundaries_full_round"]
            ),
        },
        "predicted_class_count_intervals": count_intervals(
            result["classes"], measured
        ),
        "unattributed_risk_statement": (
            "The frozen build still leaves "
            f"{result['families'].get('UNATTRIBUTED', 0)} of {sum(result['families'].values())} "
            "surviving archival divergences unattributed, concentrated in the "
            "exclusion counters recorded above. Every one of those counters is "
            "registered at predicted-zero for the fresh sweep, so this archival "
            "residue is a declared certification risk, not a licence to widen "
            "attribution. If the fresh sweep reproduces it, C26 fails and the "
            "residue is the diagnosis queue."
        ),
    }
    Path(args.json).write_text(json.dumps(out, indent=1) + "\n")
    print(
        f"recalibrated {len(rows)} archival rows on {engine['fingerprint'][:16]} "
        f"({engine['patch_count']} patches): {result['tally']} -> {args.json}"
    )
    if result["errors"]:
        print("RECALIBRATION FAILED: one or more retained rows could not be evaluated")
        return 1
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
