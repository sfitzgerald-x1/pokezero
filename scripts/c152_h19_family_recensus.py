#!/usr/bin/env python
"""H19: do the four never-adjudicated families survive into the current era?

Ledger row H19 (`reports/c138_known_gaps_ledger.md`) names four families from
`reports/c86_current_era_family_adjudication.json` -- `LS_capped_lethal_shape`,
`I2_matcher_accounting`, `I3_roll_inherited`, `I5_boundary_truncation` -- and
records **UNKNOWN whether they survive into the current era**, on the evidence
that *"none of their labels appears in the c136 counters"*.

**That evidence is vacuous, and this script exists partly to say so.** No sweep
has ever emitted a family label into a counter. `scripts/engine_transition_differential.py`
does not import `scripts/cert_sweep_readout.py` at all -- verified by
`--check-not-wired` below -- so a divergent row gets a `divergence_class` and no
family. The families are a SECOND classification pass, applied by
`cert_sweep_readout.classify_row` to a row's recorded `divergence_class`,
`protocol` and `branch_misses`. Absence from a counter vocabulary the layer never
writes to says nothing at all.

H19's named settling measurement -- *"re-run `scripts/family_bucket_audit.py`
against the c136 artifacts"* -- also could not be run as written until C152:
`family_bucket_audit.py:355` referenced an undefined `ROOT`, and that line is
reached unconditionally, so the script raised `NameError` on every input. C152
fixes it. This module is the surrounding measurement.

Three passes, and they answer three different questions:

``as_recorded``
    Apply `classify_row` to every committed divergent repro, using each
    artifact's OWN recorded `divergence_class` and `branch_misses`. This is the
    historical question: did the family ever have rows, and where was its last
    one? The glob is stated in the output and is the widest one available --
    `reports/**/*.json` plus `docs/**/*.json`.

``rereading``
    Re-read a named subset of rows through the CURRENT build with
    `cert_sweep_reread.reread_row`, which calls the shipped
    `evaluate_boundary_strict`. A row that no longer diverges has no family. This
    is a replay of committed states, not a sweep.

    ⚠ **A re-read is not a re-sweep, and the difference is load-bearing for two
    rows.** `reread_row` replays the state RECORDED in the artifact. Where a
    later fix changed WORLD CONSTRUCTION rather than the engine -- `19100170/71`
    and `/72`, closed by `d27316b6` (#1148) and bisected in
    `reports/c145_itemleftovers_row_adjudication.md` -- the recorded state still
    encodes the pre-fix world, so it still diverges on re-read while a fresh
    sweep of the same seed produces no row at all. Both facts are reported.

``live``
    The family counts implied by a fresh sweep: with zero divergent rows there
    are zero rows in every registered family, by construction.

Usage::

    PYTHONPATH=src python scripts/c152_h19_family_recensus.py \\
        --reread reports/artifacts/c136_faintcancels_fix_dev_sweep.json \\
        --reread reports/artifacts/c141_final_holdout_sweep.json \\
        --live-sweep reports/artifacts/c152_head_dev_sweep.json \\
        --json reports/artifacts/c152_h19_family_recensus.json
"""

from __future__ import annotations

import argparse
import glob
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

FOUR = (
    "LS_capped_lethal_shape",
    "I2_matcher_accounting",
    "I3_roll_inherited",
    "I5_boundary_truncation",
)

GLOB = ("reports/**/*.json", "docs/**/*.json")

# Excluded from `as_recorded`, with the reason, because a family history is a
# statement about what the SHIPPING matcher produced. C152's window-disabled arm
# is a mutant comparator run for H8's settling measurement; its divergent rows
# are an artefact of removing the +/-9 % accept, not history. Counting them
# inflated `I2_matcher_accounting` from 85 to 113 on the first run of this
# script, which is exactly the kind of self-reference a census must not absorb
# silently.
EXCLUDED = (
    "reports/artifacts/c152_h8_nowindow_dev_sweep.json",
    "reports/artifacts/c152_h8_nowindow_holdout_sweep.json",
)


def _c_number(path: str) -> int:
    match = re.search(r"/c(\d+)[_.]", path)
    return int(match.group(1)) if match else -1


def _relative(path: str) -> str:
    """Repo-relative, whether the caller passed an absolute or a relative path."""
    resolved = Path(path).resolve()
    try:
        return str(resolved.relative_to(REPO_ROOT))
    except ValueError:
        return str(resolved)


def _family_of(readout: Any, row: dict[str, Any]) -> str:
    family, _basis, counter = readout.classify_row(row)
    return counter if family == "UNATTRIBUTED" and counter else family


def as_recorded() -> dict[str, Any]:
    import cert_sweep_readout as readout

    totals: Counter = Counter()
    per_family: dict[str, Counter] = defaultdict(Counter)
    artifacts = 0
    rows = 0
    for pattern in GLOB:
        for path in sorted(glob.glob(str(REPO_ROOT / pattern), recursive=True)):
            try:
                loaded = json.loads(Path(path).read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError, UnicodeDecodeError):
                continue
            if not isinstance(loaded, dict) or not loaded.get("repros"):
                continue
            relative = str(Path(path).relative_to(REPO_ROOT))
            if relative in EXCLUDED:
                continue
            counted = False
            for row in loaded["repros"]:
                if not isinstance(row, dict) or row.get("kind") != "transition_diverged":
                    continue
                counted = True
                rows += 1
                family = _family_of(readout, row)
                totals[family] += 1
                per_family[family][relative] += 1
            artifacts += int(counted)

    def latest(family: str) -> list[dict[str, Any]]:
        items = sorted(per_family[family].items(), key=lambda t: (_c_number(t[0]), t[0]))
        return [
            {"artifact": path, "c_number": _c_number(path), "rows": n}
            for path, n in items[-4:]
        ]

    return {
        "glob": list(GLOB),
        "excluded_and_why": {
            "paths": list(EXCLUDED),
            "reason": (
                "C152's window-disabled arm is a MUTANT comparator, run only to "
                "measure H8. Its divergent rows are produced by removing the "
                "+/-9 % accept and are not evidence about the shipping matcher. "
                "Leaving them in inflated I2_matcher_accounting from 85 to 113."
            ),
        },
        "artifacts_carrying_repros": artifacts,
        "divergent_rows_classified": rows,
        "rows_per_family": dict(totals),
        "the_four": {
            family: {
                "rows_ever": totals.get(family, 0),
                "artifacts": len(per_family[family]),
                "highest_c_number_artifacts": latest(family),
            }
            for family in FOUR
        },
    }


def rereading(paths: list[str]) -> dict[str, Any]:
    import cert_sweep_readout as readout
    from cert_sweep_reread import reread_row
    from engine_transition_differential import classify_divergence

    out: list[dict[str, Any]] = []
    for path in paths:
        loaded = json.loads(Path(path).read_text(encoding="utf-8"))
        for row in loaded.get("repros") or []:
            if row.get("kind") != "transition_diverged":
                continue
            recorded = _family_of(readout, row)
            verdict, misses, branches = reread_row(row)
            if verdict == "diverged":
                current = {
                    **row,
                    "divergence_class": classify_divergence(row["protocol"], misses),
                    "branch_misses": misses[:12],
                }
                now = _family_of(readout, current)
            else:
                now = None
            out.append(
                {
                    "artifact": _relative(path),
                    "seed": row["seed"],
                    "step": row["step"],
                    "family_as_recorded": recorded,
                    "verdict_on_this_build": verdict,
                    "family_on_this_build": now,
                    "branches": branches,
                }
            )
    return {"rows": out, "still_diverged": sum(1 for r in out if r["family_on_this_build"])}


def live(paths: list[str]) -> dict[str, Any]:
    out = {}
    for path in paths:
        loaded = json.loads(Path(path).read_text(encoding="utf-8"))
        out[_relative(path)] = {
            "boundaries_measured": loaded.get("boundaries_measured"),
            "transitions_diverged": loaded.get("transitions_diverged"),
            "divergence_classes": loaded.get("divergence_classes"),
            "rows_in_every_registered_family": 0
            if loaded.get("transitions_diverged") == 0
            else "NONZERO -- classify the repros",
        }
    return out


def check_not_wired() -> dict[str, Any]:
    """Is the family layer reachable from the sweep at all? Measured, not argued."""
    source = (REPO_ROOT / "scripts" / "engine_transition_differential.py").read_text(
        encoding="utf-8"
    )
    return {
        "differential_imports_cert_sweep_readout": "cert_sweep_readout" in source,
        "differential_calls_classify_row": "classify_row" in source,
        "reading": (
            "Both false means a divergent row today carries a `divergence_class` and "
            "no family label, so H19's 'none of their labels appears in the c136 "
            "counters' is vacuous rather than evidential."
        ),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--reread", action="append", default=[])
    parser.add_argument("--live-sweep", action="append", default=[])
    parser.add_argument("--json", required=True)
    args = parser.parse_args(argv)

    out = {
        "schema": "c152-h19-family-recensus/1",
        "what": (
            "Whether H19's four never-adjudicated families survive into the current "
            "era, measured three ways: as recorded across every committed artifact, "
            "re-read through the current build, and against fresh head sweeps."
        ),
        "is_the_family_layer_wired_into_the_sweep": check_not_wired(),
        "as_recorded": as_recorded(),
        "rereading": rereading(args.reread) if args.reread else None,
        "live": live(args.live_sweep) if args.live_sweep else None,
    }
    Path(args.json).write_text(json.dumps(out, indent=2, sort_keys=True) + "\n")
    print(json.dumps(out["as_recorded"]["the_four"], indent=2, sort_keys=True))
    print(f"-> {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
