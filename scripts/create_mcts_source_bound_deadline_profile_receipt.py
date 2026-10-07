#!/usr/bin/env python3
"""Create a receipt for a current-source MCTS deadline profile.

Unlike the historical deadline qualification receipt, this binds the exact
execution image without asserting that it contains the older reviewed deadline
mechanism.  The companion runner records that distinction in its terminal
evidence and still verifies every source and native-runtime hash at execution.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import create_mcts_deadline_source_receipt as deadline_receipt


PROFILE_RUNNER = "scripts/run_mcts_source_bound_deadline_profile.py"
FIXED_WORK_PARITY_RUNNER = "scripts/run_mcts_source_bound_fixed_work_parity.py"
LEAF_ABLATION_RUNNER = "scripts/run_source_root_leaf_ablation.py"
LEAF_CONTINUATION_RUNNER = "scripts/run_source_root_leaf_continuations.py"
LEGACY_RUNNER = "scripts/run_mcts_deadline_qualification.py"


def build_receipt(*, source_root: Path, b2_receipt_path: Path) -> dict[str, Any]:
    """Bind the profile and delegated verifier to one clean source/image pair."""

    commit = deadline_receipt._clean_detached_commit(source_root)
    b2_receipt = deadline_receipt._read_canonical_json(b2_receipt_path)
    runtime = b2_receipt.get("model_runtime")
    if not isinstance(runtime, Mapping):
        raise deadline_receipt.ReceiptError("source-image receipt omits model runtime evidence")
    fingerprint = runtime.get("engine_fingerprint")
    if not isinstance(fingerprint, str) or not deadline_receipt.SHA256.fullmatch(fingerprint):
        raise deadline_receipt.ReceiptError("source-image receipt has an invalid native fingerprint")
    deadline_receipt._validate_b2_receipt(
        b2_receipt,
        commit=commit,
        fingerprint=fingerprint,
    )

    legacy_pins = deadline_receipt._assignment_literals(source_root / LEGACY_RUNNER)
    required_files = (
        *legacy_pins["REQUIRED_RECEIPT_FILES"],
        PROFILE_RUNNER,
        FIXED_WORK_PARITY_RUNNER,
        LEAF_ABLATION_RUNNER,
        LEAF_CONTINUATION_RUNNER,
    )
    source_files: dict[str, str] = {}
    for relative in required_files:
        candidate = (source_root / relative).resolve()
        try:
            candidate.relative_to(source_root.resolve())
        except ValueError as error:
            raise deadline_receipt.ReceiptError("profile source file escapes source root") from error
        if not candidate.is_file() or candidate.is_symlink():
            raise deadline_receipt.ReceiptError(f"profile source file is missing: {relative}")
        source_files[relative] = deadline_receipt._sha256(candidate)

    engine = source_root / "src" / "pokezero" / "engine_search.py"
    if source_files.get("src/pokezero/engine_search.py") != deadline_receipt._sha256(engine):
        raise deadline_receipt.ReceiptError("profile receipt omitted engine_search.py")
    inputs = deadline_receipt._execution_source_files(source_root)
    return {
        "schema_version": deadline_receipt.SOURCE_RECEIPT_SCHEMA_VERSION,
        "complete": True,
        "immutable_image": b2_receipt["immutable_image"],
        "source_commit": commit,
        "execution_tree_sha256": deadline_receipt._execution_tree_sha256(source_root, inputs),
        "engine_fingerprint": fingerprint,
        "source_files_sha256": source_files,
    }


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--create", action="store_true", help="Create one new receipt; never replace.")
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--b2-receipt", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args(argv)
    if not args.create:
        parser.error("--create is required")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        receipt = build_receipt(
            source_root=args.source_root.resolve(),
            b2_receipt_path=args.b2_receipt.resolve(),
        )
        deadline_receipt._create_only_json(args.out.resolve(), receipt)
    except deadline_receipt.ReceiptError as error:
        print(f"CANNOT CREATE SOURCE-BOUND DEADLINE PROFILE RECEIPT: {error}", file=sys.stderr)
        return 2
    print(f"WROTE SOURCE-BOUND MCTS DEADLINE PROFILE RECEIPT {args.out.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
