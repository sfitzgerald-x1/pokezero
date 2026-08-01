#!/usr/bin/env python
"""Diagnose the structural component-count residue against the C26 classifier.

``structural_component_count_without_supported_sibling`` is registered at
predicted-zero for the C26 sweep, and it is not zero: a fresh sample of the
frozen build produces it at roughly half a row per game. Before anyone decides
whether that is an engine gap or a classifier gap, the residue needs a
breakdown that separates the two.

This tool answers three questions over a divergent-row population:

1. WHERE does the count mismatch sit? The rule scans every branch arm, so a
   6%-mass arm can decide the row while the majority arm complains about
   something else entirely. Every neighbouring rule in the classifier is
   majority-scoped, and I4 states the reason in its own comment: "A tie
   confined to a minority arm cannot explain the majority-arm complaint."
2. WHICH documented family would have claimed the row? The rule precedes
   I1/I2/I5, so a row whose majority miss is a ``_to_full`` cap shape -- I1's
   literal definition -- never reaches I1.
3. Does a SIBLING arm actually carry the observed shape? The carry check
   compares roll-scaled components by exact magnitude, while the matcher that
   produced the miss accepts them at any legal roll. Those two tolerances
   disagree, so a supporting sibling at a different legal roll reads as absent.

The counterfactual scopings are compiled IN MEMORY from the pinned classifier
source. This tool never edits ``scripts/cert_sweep_readout.py``: that file's
SHA-256 is part of the frozen C26 source identity, and changing it would
silently unpin the registered contract. A counterfactual is evidence for a
decision, not the decision.

READ THIS BEFORE ACTING ON THE NUMBERS: moving a rule that emits UNATTRIBUTED
to sit after rules that attribute will always reduce the unattributed count.
That is the exact shape of widening attribution to make a gate pass. Each
scoping below has to stand on its own stated semantics -- and the residue that
survives every one of them is the real diagnosis queue.

Usage::

    PYTHONPATH=src python scripts/c26_structural_residue_diagnosis.py \\
        --report fresh-differential-report.json [--archive-shards GLOB] \\
        --json out.json
"""

from __future__ import annotations

import argparse
import json
import sys
import types
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

CLASSIFIER = REPO_ROOT / "scripts" / "cert_sweep_readout.py"
STRUCTURAL_COUNTER = "structural_component_count_without_supported_sibling"

# Anchors into the pinned classifier. If one stops matching, the classifier
# moved and every counterfactual below is describing code that no longer
# exists -- so this fails loudly instead of reporting stale arithmetic.
_ALL_ARMS = '''    if cls == "roll_scaled_component":
        for miss in misses:
            miss_obs, miss_eng = _miss_pairs(miss)'''
_MAJORITY_ARM = '''    if cls == "roll_scaled_component":
        for miss in [majority]:
            miss_obs, miss_eng = _miss_pairs(miss)'''
_EXACT_CARRY = '''        if Counter(sibling_engine_components) == observed_components:
            return True'''
_TOLERANT_CARRY = '''        if Counter(sibling_engine_components) == observed_components:
            return True
        if _same_component_labels(sibling_engine_components, observed):
            return True'''
_CARRY_DEF = "\ndef _sibling_arm_carries_observed_components("
_CARRY_HELPER = '''
def _same_component_labels(engine, observed) -> bool:
    """Label-level carry, at the tolerance the matcher itself already uses."""

    if not engine or not observed:
        return False
    return sorted(source for source, _value in engine) == sorted(
        source for source, _value in observed
    )

''' + _CARRY_DEF


def _variant_sources() -> dict[str, str]:
    source = CLASSIFIER.read_text(encoding="utf-8")
    for anchor in (_ALL_ARMS, _EXACT_CARRY, _CARRY_DEF):
        if source.count(anchor) != 1:
            raise ValueError(
                "the pinned classifier no longer matches this diagnosis; "
                "re-derive the counterfactuals before trusting any number"
            )
    majority = source.replace(_ALL_ARMS, _MAJORITY_ARM)
    return {
        "registered": source,
        "majority_scoped": majority,
        "majority_scoped_and_roll_tolerant_sibling": (
            majority.replace(_EXACT_CARRY, _TOLERANT_CARRY)
            .replace(_CARRY_DEF, _CARRY_HELPER)
        ),
    }


def _load(source: str, name: str) -> types.ModuleType:
    module = types.ModuleType(name)
    module.__file__ = f"{name}.py"
    sys.modules[name] = module
    exec(compile(source, module.__file__, "exec"), module.__dict__)
    return module


def _tally(module: types.ModuleType, rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    families: Counter = Counter()
    counters: Counter = Counter()
    for row in rows:
        family, _basis, counter = module.classify_row(row)
        families[family] += 1
        if family == "UNATTRIBUTED":
            counters[counter] += 1
    return {
        "rows": len(rows),
        "unattributed": families["UNATTRIBUTED"],
        "structural": counters.get(STRUCTURAL_COUNTER, 0),
        "families": dict(families.most_common()),
        "exclusion_counters": dict(counters.most_common()),
    }


def shape_breakdown(
    registered: types.ModuleType, rows: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    """Where the deciding count mismatch sits, and what shape it carries."""

    location: Counter = Counter()
    cap_shape: Counter = Counter()
    majority_mass: list[float] = []
    for row in rows:
        family, _basis, counter = registered.classify_row(row)
        if family != "UNATTRIBUTED" or counter != STRUCTURAL_COUNTER:
            continue
        misses = row.get("branch_misses") or []
        majority = registered._majority_miss(misses)
        observed, engine = registered._miss_pairs(majority)
        on_majority = len(observed) != len(engine)
        location["majority_arm" if on_majority else "minority_arm_only"] += 1
        cap_shape["to_full_in_majority" if "_to_full" in majority else "no_cap_shape"] += 1
        found = registered._PCT_RE.search(majority)
        if found:
            majority_mass.append(float(found.group(1)))
    ordered = sorted(majority_mass)
    return {
        "count_mismatch_location": dict(location),
        "majority_miss_cap_shape": dict(cap_shape),
        "majority_arm_mass_percent": {
            "min": ordered[0] if ordered else None,
            "median": ordered[len(ordered) // 2] if ordered else None,
            "max": ordered[-1] if ordered else None,
        },
    }


def divergent_rows(report: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    retention = report.get("repro_retention")
    if not isinstance(retention, Mapping) or retention.get("repros_complete") is not True:
        raise ValueError(
            "report retention is incomplete; a truncated row population would "
            "understate the residue"
        )
    return [
        row for row in (report.get("repros") or [])
        if isinstance(row, Mapping) and row.get("kind") == "transition_diverged"
    ]


def archive_rows(shard_glob: str) -> list[Mapping[str, Any]]:
    """Re-read the retained archive so its rows carry CURRENT-build misses."""

    from c26_archival_recalibration import verify_archive
    from cert_sweep_reread import reread_row
    from engine_transition_differential import classify_divergence

    _paths, retained = verify_archive(shard_glob)
    rows = []
    for row in retained:
        verdict, misses, _branches = reread_row(row)
        if verdict != "diverged":
            continue
        rows.append({
            **row,
            "divergence_class": classify_divergence(row["protocol"], misses),
            "branch_misses": misses[:12],
        })
    return rows


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--report", required=True,
                        help="a complete-retention engine_transition_differential report")
    parser.add_argument("--report-games", type=int, default=None,
                        help="games behind --report (default: the report's own count)")
    parser.add_argument("--archive-shards", default=None,
                        help="optional retained-archive glob for a second population")
    parser.add_argument("--json", required=True)
    args = parser.parse_args(argv)

    try:
        report = json.loads(Path(args.report).read_text(encoding="utf-8"))
        fresh = divergent_rows(report)
        variants = _variant_sources()
    except (OSError, ValueError, json.JSONDecodeError) as error:
        print(f"DIAGNOSIS INPUT REFUSED: {error}", file=sys.stderr)
        return 2

    games = args.report_games or report.get("games")
    populations: dict[str, list[Mapping[str, Any]]] = {"fresh_sample": fresh}
    if args.archive_shards:
        try:
            populations["retained_archive"] = archive_rows(args.archive_shards)
        except (OSError, ValueError, json.JSONDecodeError) as error:
            print(f"ARCHIVE INPUT REFUSED: {error}", file=sys.stderr)
            return 2

    modules = {name: _load(src, f"c26_variant_{name}") for name, src in variants.items()}
    out: dict[str, Any] = {
        "schema": "c26-structural-residue-diagnosis/1",
        "purpose": (
            "Separate the classifier-scoping share of the structural residue "
            "from the share that survives every defensible scoping. The "
            "surviving share is the diagnosis queue; the rest is a rule that "
            "fires wider than its own stated semantics."
        ),
        "warning": (
            "Reordering or rescoping a rule that emits UNATTRIBUTED will always "
            "reduce the unattributed count. These counterfactuals are evidence "
            "for a decision that must be pre-registered against a new source "
            "freeze, never a repair applied to a registered contract in place."
        ),
        "classifier_sha256": __import__("hashlib").sha256(
            CLASSIFIER.read_bytes()
        ).hexdigest(),
        "populations": {},
    }
    for population, rows in populations.items():
        entry: dict[str, Any] = {
            "shape": shape_breakdown(modules["registered"], rows),
            "scopings": {name: _tally(module, rows) for name, module in modules.items()},
        }
        if population == "fresh_sample" and games:
            entry["games"] = games
            for name, tally in entry["scopings"].items():
                tally["unattributed_per_game"] = round(tally["unattributed"] / games, 4)
                tally["projected_unattributed_per_10k_games"] = round(
                    tally["unattributed"] / games * 10_000
                )
        out["populations"][population] = entry

    Path(args.json).write_text(json.dumps(out, indent=1) + "\n")
    for population, entry in out["populations"].items():
        for name, tally in entry["scopings"].items():
            print(f"{population:16s} {name:42s} unattributed={tally['unattributed']:5d}"
                  f"/{tally['rows']:<5d} structural={tally['structural']}")
    print(f"-> {args.json}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
