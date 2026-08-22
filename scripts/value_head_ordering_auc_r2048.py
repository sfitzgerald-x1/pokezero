#!/usr/bin/env python3
"""Score Phase-3 heads only on the registered OI-1 R=2048 confirmation bank.

This is a separate configuration from ``value_head_ordering_auc.py``.  That
instrument remains intentionally pinned to the historical R=64 VHProbe bank;
using it for this corpus would mix label-noise regimes and silently turn the
screen into a gate.  This wrapper accepts only rescore cells that bind the
registered targeted-gap contract and confirmation-shard inventory below.

The statistic, tie treatment, paired sign-flip test, clustered bootstrap, and
advance threshold are imported from the reviewed OI-1 implementation.  Only
the corpus identity and R=2048 precision configuration differ.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
from pathlib import Path
from typing import Any, Mapping


HERE = Path(__file__).resolve().parent
_SPEC = importlib.util.spec_from_file_location(
    "value_head_ordering_auc_base", HERE / "value_head_ordering_auc.py")
if _SPEC is None or _SPEC.loader is None:  # pragma: no cover - installation failure
    raise RuntimeError("cannot load the reviewed OI-1 scorer")
base = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(base)


SCHEMA = "pokezero.oi1-targeted-gap-r2048-cell.v1"
CONTRACT_SHA256 = "b431047c20d98d8464037939847deec808ecd576f77224d21480424b531421b8"
ROLLOUTS = 2048
TAU_PRIMARY = 0.10
MIN_COMPLETE_PAIRS = 1800
MIN_PRIMARY_ELIGIBLE_PAIRS = 1000
WORST_CASE_GAP_SE = 0.5 * math.sqrt(2.0 / ROLLOUTS)
SE_OVER_TAU = WORST_CASE_GAP_SE / TAU_PRIMARY


class Refusal(RuntimeError):
    """The requested score lacks the preregistered evidence needed for a gate."""


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _json(path: Path) -> Mapping[str, Any]:
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise Refusal(f"REFUSING: cannot read {path.name}: {exc}") from exc
    if not isinstance(value, Mapping):
        raise Refusal(f"REFUSING: {path.name} must contain a JSON object")
    return value


def pair_set_sha256(rows: Mapping[tuple, Mapping[str, Any]]) -> str:
    """Identity of the exact paired state set, independent of head values."""
    payload = [
        {"seed": key[0], "prefix": key[1], "seat": key[2],
         "true_gap": float(rows[key]["true_gap"]),
         "arm_a": rows[key].get("arm_a"), "arm_b": rows[key].get("arm_b")}
        for key in sorted(rows)
    ]
    return hashlib.sha256(json.dumps(payload, allow_nan=False, sort_keys=True,
                                     separators=(",", ":")).encode()).hexdigest()


def _metadata(doc: Mapping[str, Any], name: str, path: Path) -> Mapping[str, Any]:
    value = doc.get("oi1_targeted_gap_r2048")
    if not isinstance(value, Mapping):
        raise Refusal(
            f"REFUSING: {name} ({path.name}) has no R=2048 targeted-gap provenance. "
            "A generic R=64 or screen cell cannot be promoted into this gate.")
    required = {
        "schema": SCHEMA,
        "contract_sha256": CONTRACT_SHA256,
        "stage": "confirmation",
        "rollouts_per_arm": ROLLOUTS,
        "screen_rows_included": False,
    }
    for field, expected in required.items():
        if value.get(field) != expected:
            raise Refusal(
                f"REFUSING: {name} ({path.name}) has {field}={value.get(field)!r}; "
                f"expected {expected!r} for the registered R=2048 confirmation gate.")
    for field in ("corpus_sha256", "confirmation_shard_sha256", "pair_set_sha256",
                  "source_checkpoint_sha256"):
        if not value.get(field):
            raise Refusal(f"REFUSING: {name} ({path.name}) lacks provenance field {field!r}")
    if not isinstance(value["confirmation_shard_sha256"], Mapping):
        raise Refusal(f"REFUSING: {name} ({path.name}) has no confirmation-shard digest map")
    return value


def load_cell(path: Path, name: str) -> tuple[dict[tuple, dict], Mapping[str, Any]]:
    """Load a rescore cell and reject any budget/stage/provenance substitution."""
    doc = _json(path)
    meta = _metadata(doc, name, path)
    try:
        rows = base.load_cell(path, name, rollouts=ROLLOUTS)
    except SystemExit as exc:
        raise Refusal(str(exc)) from exc
    for key, row in rows.items():
        if row.get("rollouts_a") != ROLLOUTS or row.get("rollouts_b") != ROLLOUTS:
            raise Refusal(f"REFUSING: {name} lacks complete R=2048 truth at {key}")
        # The registered producer has a deliberate sparse encoding for cap
        # counts: it omits a cap field iff no continuation capped, and emits a
        # nonzero integer otherwise. Keep that exact canonical zero encoding,
        # but refuse every present malformed/nonzero count. Failure telemetry
        # is different: the producer always writes an explicit empty list.
        for field in ("capped_a", "capped_b"):
            if field in row and (type(row[field]) is not int or row[field] != 0):
                raise Refusal(f"REFUSING: {name} lacks verified uncapped R=2048 truth at {key}")
        for field in ("failed_a", "failed_b"):
            if not isinstance(row.get(field), list) or row[field] != []:
                raise Refusal(f"REFUSING: {name} lacks verified complete R=2048 truth at {key}")
        if row.get("pairing_intact") is not True:
            raise Refusal(f"REFUSING: {name} has incomplete paired R=2048 truth at {key}")
    actual = pair_set_sha256(rows)
    if actual != meta["pair_set_sha256"]:
        raise Refusal(
            f"REFUSING: {name} ({path.name}) pair-set digest disagrees with its content; "
            "the claimed identical-key comparison is not reproducible.")
    return rows, meta


def align_metadata(metadata: Mapping[str, Mapping[str, Any]]) -> Mapping[str, Any]:
    """Every arm must be a rescore of the same terminal corpus and confirmation rows."""
    names = sorted(metadata)
    first = metadata[names[0]]
    for name in names[1:]:
        current = metadata[name]
        for field in ("corpus_sha256", "confirmation_shard_sha256", "pair_set_sha256",
                      "source_checkpoint_sha256"):
            if current[field] != first[field]:
                raise Refusal(
                    f"REFUSING: {name} and {names[0]} disagree on {field}; paired dC "
                    "requires the same R=2048 confirmation evidence.")
    return first


def validate_bank(rows: Mapping[tuple, Mapping[str, Any]]) -> None:
    if len(rows) < MIN_COMPLETE_PAIRS:
        raise Refusal(
            f"REFUSING: only {len(rows)} complete confirmation pairs; the registered "
            f"minimum is {MIN_COMPLETE_PAIRS}.")
    eligible = len(base.eligible_keys(rows, sorted(rows), TAU_PRIMARY))
    if eligible < MIN_PRIMARY_ELIGIBLE_PAIRS:
        raise Refusal(
            f"REFUSING: only {eligible} tau={TAU_PRIMARY:.2f} eligible confirmation pairs; "
            f"the registered minimum is {MIN_PRIMARY_ELIGIBLE_PAIRS}.")


def score(ref_name: str, cells: Mapping[str, Mapping[tuple, dict]],
          metadata: Mapping[str, Mapping[str, Any]], bootstrap_reps: int) -> dict[str, Any]:
    """Produce the reviewed OI statistic after enforcing the R=2048 contract."""
    meta = align_metadata(metadata)
    keys = base.align(cells, ref_name)
    validate_bank(cells[ref_name])
    if not math.isclose(SE_OVER_TAU, 0.15625, rel_tol=0.0, abs_tol=1e-12):
        raise AssertionError("the fixed R=2048 precision calculation changed")

    # The underlying helpers read these module globals at call time.  This is a
    # separate executable configuration, not a caller-supplied knob.
    base.PINNED_ROLLOUTS = ROLLOUTS
    base.LABEL_GAP_SE_WORST_CASE = WORST_CASE_GAP_SE
    base.PAIR_SAMPLING_DISTRIBUTION = (
        "top-2 sibling action pairs at six prefixes from 320 selected fresh p1 games; "
        "R=2048 policy-continuation terminal labels on the independent confirmation stream; "
        "seat p1 only, common random numbers within a sibling pair")
    note = base.label_se_note(cells[ref_name], keys, TAU_PRIMARY)
    board = base.run_scoreboard(
        ref_name, cells[ref_name], cells, keys,
        [base.TAU_PRIMARY, *base.TAU_CHECKS], bootstrap_reps)
    return {
        "schema": "pokezero.phase3.ordering-instrument-r2048.v1",
        "instrument": "OI-1 targeted-gap R=2048",
        "baseline_cell": ref_name,
        "n_pairs_aligned": len(keys),
        "r2048_contract": {
            "contract_sha256": CONTRACT_SHA256,
            "rollouts_per_arm": ROLLOUTS,
            "tau_primary": TAU_PRIMARY,
            "minimum_complete_confirmation_pairs": MIN_COMPLETE_PAIRS,
            "minimum_primary_tau_eligible_pairs": MIN_PRIMARY_ELIGIBLE_PAIRS,
            "worst_case_gap_se": WORST_CASE_GAP_SE,
            "se_over_tau_primary": SE_OVER_TAU,
            "screen_rows_are_forbidden": True,
        },
        "confirmation_provenance": dict(meta),
        "label_se_vs_tau": note,
        "cells": board["cells"],
    }


def _parse(spec: str) -> tuple[str, Path]:
    name, marker, text = spec.partition("=")
    if not name or marker != "=" or not text:
        raise Refusal(f"REFUSING: --ref/--cell wants NAME=FILE, got {spec!r}")
    return name, Path(text)


def _write_new(path: Path, payload: Mapping[str, Any]) -> None:
    try:
        with path.open("x", encoding="utf-8") as handle:
            json.dump(payload, handle, allow_nan=False, indent=1, sort_keys=True)
            handle.write("\n")
    except FileExistsError as exc:
        raise Refusal(f"REFUSING: output {path.name} already exists; evidence is create-only") from exc


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--ref", required=True, metavar="NAME=FILE")
    parser.add_argument("--cell", action="append", default=[], metavar="NAME=FILE")
    parser.add_argument("--bootstrap-reps", type=int, default=2000)
    parser.add_argument("--json", type=Path, required=True,
                        help="new create-only scoreboard path")
    args = parser.parse_args()
    try:
        ref_name, ref_path = _parse(args.ref)
        cells: dict[str, dict[tuple, dict]] = {}
        metadata: dict[str, Mapping[str, Any]] = {}
        cells[ref_name], metadata[ref_name] = load_cell(ref_path, ref_name)
        for raw in args.cell:
            name, path = _parse(raw)
            if name in cells:
                raise Refusal(f"REFUSING: duplicate cell name {name!r}")
            cells[name], metadata[name] = load_cell(path, name)
        if not args.cell:
            raise Refusal("REFUSING: a baseline without at least one Phase-3 arm has no dC")
        result = score(ref_name, cells, metadata, args.bootstrap_reps)
        _write_new(args.json, result)
    except Refusal as exc:
        print(exc)
        return 2
    print("OI-1 R=2048 TARGETED-GAP SCOREBOARD")
    print(f"  aligned confirmation pairs: {result['n_pairs_aligned']}")
    for name, by_tau in result["cells"].items():
        primary = by_tau[str(TAU_PRIMARY)]
        print(f"  {name}: dC={primary['delta_c_gate']:+.4f} "
              f"p={primary['p_gate']:.4f} {primary['verdict']}")
    print(f"WROTE OI1 R2048 SCOREBOARD: {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
