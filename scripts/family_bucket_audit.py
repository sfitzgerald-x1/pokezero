#!/usr/bin/env python
"""Bucket every registered family: engine-gap, instrument-artifact, or limit.

The registered family table does not record WHY a family diverges, and that gap
is what let `limit:roll_divergent_lethality` sit under the comment "a limit of
the comparison, not an engine fault" while C27 moved 285 rows out of it. Three
prior adjudications turned out to rest on incomplete branch support. So each
family needs evidence for its bucket, not a label.

The three buckets and what discriminates them, mechanically:

* **engine-gap** — the engine's branch support does not contain a transition the
  simulator reached, and enumerating more of the roll space would. The signature
  is a legal roll range that STRADDLES a discrete threshold while the arm the
  engine emitted sits on one side of it. C27 (KO threshold) and C31 (Substitute's
  quarter-HP gate) were both found this way.
* **instrument-artifact** — the comparison rejects something the engine's own
  tolerance accepts. The signature is a majority arm whose NET HP equals the
  observed transition's: the end states agree and only the decomposition differs.
* **comparison-limit** — neither signature, and no same-transition enumeration
  reaches it.

Anything that shows no signature inside its budget lands as
**candidate-not-finding**, with what was measured recorded. That is a complete
outcome under the fidelity charter, not a failure, and it is what stops this
becoming an unbounded re-litigation of every family.

Ranking for the engine-gap bucket is rows x search impact, not rows alone: C27's
justification was mispriced Q at KO margins rather than its row count. A family
whose threshold decides a discrete outcome inside a searched world is weighted
above one that only changes how a difference is described.

Usage::

    PYTHONPATH=src python scripts/family_bucket_audit.py \\
        --shards 'path/cert_shard_*.json' --json out.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

# Families whose bucket is already established by a shipped, measured change.
# Recording them here keeps the table honest about what was earned rather than
# inferred, and names the artifact that earned it.
ESTABLISHED: dict[str, dict[str, str]] = {
    "limit:roll_divergent_lethality": {
        "bucket": "engine-gap (partially resolved)",
        "evidence": "reports/c27_kill_split_postmeasure.json",
        "finding": (
            "C27 moved 285 archive rows out of this family by applying the existing "
            "kill-split identity to the crit arm. The family's own code comment claimed "
            "it was a limit of the comparison and not an engine fault; that was wrong "
            "for at least 14% of it."
        ),
        "remainder_measured": (
            "1,082 rows survive on the C32 build. 73 of them (6.7%) have their observed "
            "damage inside the engine's own enumerated legal roll set, so the engine "
            "prices that roll and simply did not emit it: a confirmed support gap of the "
            "C27/C31 class, at the residual-lethality threshold. The remaining 1,009 are "
            "NOT thereby disagreements, and claiming so would repeat the error the matcher "
            "warns against in its own comment: the legal set is computed from the pre-state "
            "with an assumed move order, so it is 'an ADDITIONAL accept path, never a veto'. "
            "758 rows fall outside it, 134 carry no observed damage component and 117 have "
            "no recoverable legal set -- all three are undiscriminated by this measurement, "
            "not resolved by it."
        ),
        "next": (
            "The 73 confirmed rows are objective 4's next engine-gap work: extend the "
            "threshold partition to residual lethality, the third threshold after KO (C27) "
            "and Substitute viability (C31). The other 1,009 need a discriminator that does "
            "not rest on the legal set."
        ),
    },
    "structural_component_count_without_supported_sibling": {
        "bucket": "instrument-artifact (partially resolved)",
        "evidence": "reports/c28_structural_rule_postmeasure.json, "
                    "reports/c30_capped_heal_scale_postmeasure.json, "
                    "reports/c34_aliasing_fix_postmeasure.json, "
                    "reports/c32_current_engine_attestation.json",
        "finding": (
            "Four instrument defects found so far: the rule fired from minority arms (C28), the "
            "capped-heal bound was denied its scale input (C30), and C29's own fix aliased the "
            "shared observed split (C34). On the FRESH C32 sweep the counter reached 3,779 and "
            "C34 brought it to 433."
        ),
        "corrected": (
            "This entry previously read 'resolved', on the strength of a 60-game probe reaching "
            "zero. The C32 sweep produced 3,779 -- the probe was a 167th of the population and "
            "sampled the opposite shape. 433 rows survive C34, and their dominant sample inverts "
            "C34's: the ENGINE side is the longer one, splitting into damage plus a capped Leech "
            "Seed. Not resolved."
        ),
    },
    "limit:world_sample_drag_target": {
        "bucket": "comparison-limit",
        "evidence": "docs/engine_divergence_ledger_20260728.md (drag-target determinization)",
        "finding": (
            "Documented, not inferred: when Showdown's realized drag target is not the one a "
            "branch dragged, the entry-hazard arithmetic lands on a different mon with a "
            "different max HP, so the component diff reads as a hazard/residual disagreement "
            "that it is not. The ledger records that the engine lane verified these as "
            "determinization limits and not engine bugs, by feeding the exact repro state back "
            "and getting correct fan-out including the observed tick. The harness samples one "
            "target; no same-transition enumeration changes which one Showdown picked."
        ),
    },
    "truant_loaf_phase_drift": {
        "bucket": "instrument-artifact",
        "evidence": "reports/c45_truant_counter_adjudication.json",
        "finding": (
            "The C32 sweep's 9 rows are NOT loaf boundaries. 8 of 9 are switch boundaries "
            "and 8 of 9 have the Slaking on a different slot than the one being compared; "
            "the 3 that carry a TRUANT volatile carry it only because "
            "RemoveVolatileStatus TRUANT fires as the Slaking leaves the field, and the "
            "complaining slot's HP delta belongs to the incoming mon. The rule gated on "
            "the protocol MENTIONING a Slaking, so any switch boundary in a game "
            "containing one qualified."
        ),
        "consequence": (
            "The previous entry read this as an engine-side loaf-phase drift surviving "
            "#970 and a certification failure in its own right. That was a false alarm "
            "produced by the counter's own breadth. The rule is narrowed to the loaf "
            "signature -- |cant| truant with the Slaking active on the compared slot -- "
            "which preserves sensitivity to a genuine drift while ending the false "
            "positives. The 9 rows move to the structural counter; total unattributed is "
            "unchanged at 485, so this is a relabel and not a residue reduction."
        ),
    },
    "I1_cap_state_shape": {
        "bucket": "instrument-artifact (largely resolved)",
        "evidence": "reports/c30_capped_heal_scale_postmeasure.json",
        "finding": (
            "Fell 39 -> 1 on the fresh sample and 167 -> 42 on the archive when the "
            "capped-heal bound received the step's damage scale. This family IS the "
            "capped-heal shape by its own definition, so a silently-zero bound is "
            "precisely what manufactured it."
        ),
    },
}


def _norm(source: str) -> str:
    return source[: -len("_to_full")] if source.endswith("_to_full") else source


def _in_window(observed: int, engine: int) -> bool:
    observed, engine = abs(observed), abs(engine)
    return observed == engine or (
        engine > 0 and 0.919 * engine - 1 <= observed <= 1.09 * engine + 1
    )


def signatures(readout, row: Mapping[str, Any]) -> dict[str, Any]:
    """The two mechanical signatures, measured on the majority arm."""

    misses = row.get("branch_misses") or []
    if not misses:
        return {"measurable": False}
    majority = readout._majority_miss(misses)
    observed, engine = readout._miss_pairs(majority)
    if not observed and not engine:
        return {"measurable": False}
    # net_identical requires BOTH sides to carry components. sum([]) == 0, so
    # an empty observation against engine components that happen to cancel --
    # sandstorm -6 with a Leftovers +6, both max_hp/16 in gen3, so exact
    # cancellation is common rather than contrived -- scored as "the end states
    # agree". They do not; that is an I5_boundary_truncation shape. This is the
    # ONLY signature bucket_from_signatures treats as decisive, so a false
    # positive here becomes a published adjudication.
    net_identical = (
        bool(observed)
        and bool(engine)
        and sum(v for _s, v in observed) == sum(v for _s, v in engine)
    )
    labels_align = [_norm(s) for s, _v in observed] == [_norm(s) for s, _v in engine]
    # Gate the tolerance check on label agreement. zip() pairs by POSITION, so a
    # transposed pair -- observed [recoil -20, damage -100] against engine
    # [damage -20, recoil -100] -- scored as tolerance-passing. That is an
    # I4_attribution_tie shape, not a roll difference.
    rolls_inside = labels_align and all(
        _in_window(o, e)
        for (_os, o), (_es, e) in zip(observed, engine)
    ) if len(observed) == len(engine) else False
    return {
        "measurable": True,
        "net_identical": net_identical,
        "labels_align_cap_normalised": labels_align,
        "rolls_inside_tolerance": rolls_inside,
        "count_mismatch": len(observed) != len(engine),
        "carries_cap": any(s.endswith("_to_full") for s, _v in observed + engine),
    }


def signature_profile(tally: Counter, rows: int) -> str:
    """A named shape for the residue, so a non-verdict is still actionable.

    These are pointers for the next pass, not buckets. The distinction matters:
    a profile says where to look, a bucket says what it is.
    """

    if not rows:
        return "no-rows"
    net = tally["net_identical"] / rows
    inside = tally["rolls_inside_tolerance"] / rows
    mismatch = tally["count_mismatch"] / rows
    if net >= 0.5:
        return "net-identical-dominated"
    if inside >= 0.5 and mismatch < 0.25:
        return "tolerance-passing-majority-arm"
    if mismatch >= 0.75:
        return "count-mismatch-dominated"
    if net >= 0.25 or inside >= 0.25:
        return "mixed-with-a-partial-signature"
    return "no-signature"


_MIN_MEASURED_FOR_A_VERDICT = 5


def bucket_from_signatures(tally: Counter, rows: int) -> tuple[str, str]:
    """Assign a bucket only where a signature is unambiguous.

    LIMITS OF THESE DISCRIMINATORS, stated so the table is not over-read. They
    measure the MAJORITY arm only, and they measure two things: whether the net
    HP agrees, and whether each paired magnitude sits inside the +/-9% window.
    They do NOT identify the reject reason. A row can show a
    tolerance-passing majority arm and still be divergent because the complaint
    lies in the exact-component bucket, on the other slot, or in the
    selected-direct-event source/crit check -- none of which these signatures
    see. So a high tolerance-passing rate is a POINTER to look at the other
    comparison paths, not evidence of an instrument artifact.

    Only net-identical is treated as decisive, because it is a statement about
    the outcome rather than about one comparison path: if the end states agree,
    the difference is in how the transition was decomposed.
    """

    if not rows:
        return "no-rows", "no rows on this population to measure"
    # A bucket verdict needs enough measured rows to mean anything. Without a
    # floor, one measurable row in a thousand-row family published a confident
    # instrument-artifact verdict off a 1/1 majority -- and this audit exists
    # because three prior adjudications turned out to rest on incomplete branch
    # support.
    if rows < _MIN_MEASURED_FOR_A_VERDICT:
        return "candidate-not-finding", (
            f"only {rows} measurable row(s), below the {_MIN_MEASURED_FOR_A_VERDICT}-row "
            "floor for a bucket verdict"
        )
    if tally["net_identical"] / rows >= 0.5:
        return "instrument-artifact", (
            f"{tally['net_identical']}/{rows} majority arms reach an IDENTICAL net HP: "
            "the end states agree and only the decomposition differs"
        )
    return "candidate-not-finding", (
        f"no decisive signature: net-identical {tally['net_identical']}/{rows}, "
        f"tolerance-passing {tally['rolls_inside_tolerance']}/{rows}, "
        f"count-mismatch {tally['count_mismatch']}/{rows}. Profile recorded for the "
        "next pass; resolving it needs the reject reason, which these signatures do "
        "not measure."
    )


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--shards", default=None, help="retained-archive glob")
    parser.add_argument("--rows", default=None,
                        help="a JSON list of retained divergent rows from a FRESH sweep; "
                             "preferred over --shards, because the archive is a pre-fix, "
                             "divergence-conditioned population and a fresh sweep is not")
    parser.add_argument("--json", required=True)
    args = parser.parse_args(argv)

    import cert_sweep_readout as readout
    from c26_archival_recalibration import verify_archive
    from cert_execution_manifest import (
        EMITTABLE_DOCUMENTED_FAMILIES,
        EMITTABLE_EXCLUSION_COUNTERS,
        EMITTABLE_LIMIT_FAMILIES,
    )
    from cert_sweep_reread import reread_row
    from engine_transition_differential import classify_divergence

    if args.rows:
        retained = json.loads(Path(args.rows).read_text(encoding="utf-8"))
        population_kind = "fresh_sweep"
    elif args.shards:
        try:
            _paths, retained = verify_archive(args.shards)
        except (OSError, ValueError, json.JSONDecodeError) as error:
            print(f"ARCHIVE INPUT REFUSED: {error}", file=sys.stderr)
            return 2
        population_kind = "retained_archive"
    else:
        parser.error("need --rows or --shards")

    per_family: dict[str, Counter] = defaultdict(Counter)
    counts: Counter = Counter()
    skipped = 0
    for row in retained:
        verdict, misses, _branches = reread_row(row)
        if verdict != "diverged":
            skipped += 1
            continue
        current = {
            **row,
            "divergence_class": classify_divergence(row["protocol"], misses),
            "branch_misses": misses[:12],
        }
        family, _basis, counter = readout.classify_row(current)
        key = counter if family == "UNATTRIBUTED" and counter else family
        counts[key] += 1
        sig = signatures(readout, current)
        if not sig["measurable"]:
            per_family[key]["unmeasurable"] += 1
            continue
        per_family[key]["measured"] += 1
        for name in ("net_identical", "labels_align_cap_normalised",
                     "rolls_inside_tolerance", "count_mismatch", "carries_cap"):
            if sig[name]:
                per_family[key][name] += 1

    registered = (
        set(EMITTABLE_DOCUMENTED_FAMILIES)
        | set(EMITTABLE_LIMIT_FAMILIES)
        | set(EMITTABLE_EXCLUSION_COUNTERS)
    )
    table: dict[str, Any] = {}
    for family in sorted(registered | set(counts)):
        rows = counts.get(family, 0)
        tally = per_family.get(family, Counter())
        measured = tally.get("measured", 0)
        if family in ESTABLISHED:
            entry = dict(ESTABLISHED[family])
            # These are ASSERTED, not derived on this run. Labelling them
            # "measured change" put a hardcoded verdict beside measured
            # signatures that can contradict it, under an era_discipline header
            # promising mechanical re-derivation -- the "label, not evidence"
            # this module's own docstring condemns. Say what they are, and say
            # whether the evidence each cites is actually present.
            entry["source"] = "asserted from a prior measurement, not re-derived here"
            evidence = entry.get("evidence")
            if evidence:
                entry["evidence_present"] = (ROOT / evidence).is_file()
            entry["measured_signatures_may_disagree"] = True
        else:
            bucket, why = bucket_from_signatures(tally, measured)
            entry = {"bucket": bucket, "finding": why, "source": "mechanical signatures",
                     "signature_profile": signature_profile(tally, measured)}
        entry["archive_rows"] = rows
        entry["signatures"] = dict(tally)
        table[family] = entry

    out = {
        "schema": "family-bucket-audit/1",
        "purpose": __doc__.split("\n\n")[1].strip(),
        "era": {
            "engine_fingerprint": json.loads(
                (REPO_ROOT / "reports" / "certification_contract_lifecycle.json").read_text(
                    encoding="utf-8"
                )
            )["source_code_identity"]["engine_fingerprint"],
            "readout_sha256": hashlib.sha256(
                (REPO_ROOT / "scripts" / "cert_sweep_readout.py").read_bytes()
            ).hexdigest(),
            "differential_sha256": hashlib.sha256(
                (REPO_ROOT / "scripts" / "engine_transition_differential.py").read_bytes()
            ).hexdigest(),
        },
        "population": {
            "kind": population_kind,
            "rows": len(retained),
            "cleared_or_skipped_on_this_build": skipped,
            "divergent_rows_measured": sum(counts.values()),
        },
        "buckets": Counter(entry["bucket"].split(" ")[0] for entry in table.values()),
        "families": table,
        "era_discipline": (
            "Every adjudication here is stamped with the engine fingerprint, readout hash and "
            "differential hash above, and must be mechanically re-derived when any of the three "
            "changes. An adjudication without a current-era stamp is stale by construction."
        ),
    }
    out["buckets"] = dict(out["buckets"])
    Path(args.json).write_text(json.dumps(out, indent=1) + "\n")
    for family, entry in sorted(table.items(), key=lambda kv: -kv[1]["archive_rows"]):
        print(f"  {entry['archive_rows']:5d}  {entry['bucket']:34s}  {family}")
    print(f"\nbuckets: {out['buckets']}")
    print(f"-> {args.json}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
