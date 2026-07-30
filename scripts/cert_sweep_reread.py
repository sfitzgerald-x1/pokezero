#!/usr/bin/env python
"""Re-read retained certification-sweep rows through the CURRENT engine build.

The certification sweep (ledger Appendix Z12) retained every divergent row with
full repro payloads (``repros_complete`` on all shards). Because a retained row
carries the exact engine states, both choices, the observed features and the
step protocol, the strict per-boundary comparison can be re-executed OFFLINE
against a rebuilt engine — no Showdown re-run — which makes "does the fix clear
the row?" a direct per-row measurement over the full sweep population instead
of a fresh-sample estimate.

Method: for every retained ``transition_diverged`` row, rebuild the matcher
inputs from the recorded payloads and call the differential's own
``evaluate_boundary_strict`` (imported, not reimplemented) on the current
build. Two caveats, both verified by the validation gate below:

* ``slot_sides`` is not recorded on rows; the identity mapping (p1=side_one)
  is used, exactly as ``scripts/replay_residue.py`` does.
* observed side conditions are recorded as PRESENCE (the verdict compares
  presence only, ``_transition_mismatch``), so reconstruction is lossless for
  the verdict.

VALIDATION GATE: run with ``--expect diverged`` on the build the sweep itself
used (same fingerprint) — every row must re-read as diverged. A row that
re-reads as matched on the sweep's own build marks reconstruction infidelity
and is counted separately (``reread_infidelity``), never as a clearance.

Usage::

    PYTHONPATH=src python scripts/cert_sweep_reread.py \\
        --shards 'path/cert_shard_*.json' --json out.json [--expect diverged]
"""

from __future__ import annotations

import argparse
import glob
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Mapping

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

_MON_NAME = re.compile(r"(?:^|=)([A-Z0-9]+),\d+,[A-Z]+,[A-Z]+,")
DEFAULT_EXPECTED_ROWS = 3821


def _features(payload: Mapping[str, Any]) -> TurnFeatures:
    from pokezero.engine_fidelity import TurnFeatures

    return TurnFeatures(
        p1_hp=int(payload["p1_hp"]),
        p2_hp=int(payload["p2_hp"]),
        p1_status=str(payload["p1_status"]),
        p2_status=str(payload["p2_status"]),
        fainted=frozenset(payload.get("fainted") or []),
        weather=str(payload["weather"]),
        side_conditions={
            side: {name: 1 for name in names}
            for side, names in (payload.get("side_conditions") or {}).items()
        },
    )


def _party_display(state_str: str) -> dict[str, list[str]]:
    names = [m.group(1).title() for m in _MON_NAME.finditer(state_str)]
    # The serialized state lists side_one's six mons first, then side_two's.
    if len(names) >= 12:
        return {"p1": names[:6], "p2": names[6:12]}
    half = max(1, len(names) // 2)
    return {"p1": names[:half], "p2": names[half:]}


def reread_row(row: Mapping[str, Any]) -> tuple[str, list[str], int]:
    import poke_engine

    from engine_transition_differential import evaluate_boundary_strict

    states = [poke_engine.State.from_string(s) for s in row["engine_states"]]
    slot_sides = {"p1": "side_one", "p2": "side_two"}
    counts: Counter = Counter()
    return evaluate_boundary_strict(
        states=states,
        slot_sides=slot_sides,
        choices=row["choices"],
        party_display=_party_display(row["engine_states"][0]),
        turn=0,
        pre_features=_features(row["pre_features"]),
        observed=_features(row["observed"]),
        step_lines=list(row["protocol"]),
        observed_boosts=row.get("observed_boost_deltas") or {},
        active_changed=row.get("active_changed") or {"p1": False, "p2": False},
        counts=counts,
    )


def load_retained_rows(shard_glob: str, *, expected_rows: int) -> list[Mapping[str, Any]]:
    """Load one complete retained population, refusing partial shard sets.

    The certification archive has exactly 3,821 transition-divergence rows. A
    smaller input could make a clearance count look better merely because rows
    were omitted, so both each shard's retention declaration and the aggregate
    row count are part of the reread contract.
    """
    shard_paths = sorted(glob.glob(shard_glob))
    if not shard_paths:
        raise ValueError(f"no shard files matched {shard_glob!r}")

    rows: list[Mapping[str, Any]] = []
    for shard_path in shard_paths:
        data = json.loads(Path(shard_path).read_text())
        if not isinstance(data, Mapping):
            raise ValueError(f"{shard_path}: expected a JSON object")
        retention = data.get("repro_retention")
        if not isinstance(retention, Mapping) or retention.get("repros_complete") is not True:
            raise ValueError(f"{shard_path}: retained input is incomplete (repros_complete != true)")

        retained = retention.get("repros_retained")
        diverged = retention.get("transitions_diverged")
        repros = data.get("repros")
        if not isinstance(retained, int) or not isinstance(diverged, int) or not isinstance(repros, list):
            raise ValueError(f"{shard_path}: incomplete retained-input metadata")
        if retained != diverged:
            raise ValueError(
                f"{shard_path}: retained input is incomplete "
                f"({retained} retained != {diverged} divergent)"
            )

        transition_rows = [
            row for row in repros
            if isinstance(row, Mapping) and row.get("kind") == "transition_diverged"
        ]
        if len(transition_rows) != diverged:
            raise ValueError(
                f"{shard_path}: retained input is incomplete "
                f"({len(transition_rows)} transition rows != {diverged} declared)"
            )
        rows.extend(transition_rows)

    identities: set[tuple[int, int]] = set()
    for row in rows:
        seed = row.get("seed")
        step = row.get("step")
        if (
            isinstance(seed, bool)
            or not isinstance(seed, int)
            or isinstance(step, bool)
            or not isinstance(step, int)
        ):
            raise ValueError(
                "retained input has a transition row without an integer seed/step identity"
            )
        identity = (seed, step)
        if identity in identities:
            raise ValueError(f"retained input has duplicate transition identity {identity}")
        identities.add(identity)

    if len(rows) != expected_rows:
        raise ValueError(
            "retained input does not satisfy the expected-row contract "
            f"({len(rows)} rows != {expected_rows})"
        )
    return rows


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--shards", required=True)
    parser.add_argument("--json", required=True)
    parser.add_argument(
        "--expected-rows",
        type=int,
        default=DEFAULT_EXPECTED_ROWS,
        help=("required retained transition-row population "
              f"(default: {DEFAULT_EXPECTED_ROWS})"),
    )
    parser.add_argument("--expect", choices=["diverged"], default=None,
                        help="validation gate: every row must re-read diverged")
    args = parser.parse_args(argv)

    if args.expected_rows < 1:
        parser.error("--expected-rows must be positive")
    try:
        retained_rows = load_retained_rows(args.shards, expected_rows=args.expected_rows)
    except (OSError, ValueError, json.JSONDecodeError) as error:
        print(f"RETAINED INPUT FAILED: {error}", file=sys.stderr)
        return 2

    from engine_transition_differential import classify_divergence

    results = []
    tally: Counter = Counter()
    for row in retained_rows:
        key = {"seed": row["seed"], "step": row["step"],
               "recorded_class": row.get("divergence_class")}
        try:
            verdict, misses, _branches = reread_row(row)
        except (KeyboardInterrupt, SystemExit):
            raise
        except BaseException as error:  # noqa: BLE001
            tally["reread_error"] += 1
            results.append({**key, "verdict": f"error:{type(error).__name__}"})
            continue
        entry = {**key, "verdict": verdict}
        if verdict == "diverged":
            entry["class"] = classify_divergence(row["protocol"], misses)
            entry["misses"] = misses[:4]
        results.append(entry)
        tally[verdict] += 1

    out = {
        "engine_note": "verdicts computed against the CURRENTLY INSTALLED build; "
                       "pair this file with the build fingerprint recorded beside it",
        "rows": len(results),
        "tally": dict(tally),
        "results": results,
    }
    Path(args.json).write_text(json.dumps(out, indent=1))
    print(f"re-read {len(results)} rows: {dict(tally)} -> {args.json}")

    if tally.get("reread_error"):
        print("REREAD FAILED: one or more retained rows could not be evaluated")
        return 1
    if args.expect == "diverged" and tally.get("matched"):
        print("VALIDATION GATE FAILED: matched/error rows on the reference build")
        return 1
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
