#!/usr/bin/env python3
"""Run a source-bound deadline profile without relabelling a historical gate.

The legacy qualification runner deliberately pins one reviewed deadline
mechanism.  A current-source profile needs the same durable decision evidence,
but must bind to the image receipt that actually executes rather than pretending
that it ran that historical mechanism.  This wrapper changes provenance only;
it delegates all decision, resume, and terminal validation to the legacy
runner.
"""

from __future__ import annotations

import argparse
import sys
from typing import Any, Mapping, Sequence

import run_mcts_deadline_qualification as legacy


SELF_RELATIVE_PATH = "scripts/run_mcts_source_bound_deadline_profile.py"


def _source_bound_receipt(path: str) -> dict[str, Any]:
    """Read the receipt before installing its identity into the legacy gate."""

    receipt = legacy._source_receipt(path)
    hashes = receipt["source_files_sha256"]
    if SELF_RELATIVE_PATH not in hashes:
        raise legacy.DeadlineQualificationError(
            "source-bound profile receipt omits its executing wrapper"
        )
    return receipt


def _configure_legacy(receipt: Mapping[str, Any]) -> None:
    """Bind the delegated verifier to this exact execution receipt.

    The legacy verifier still independently checks the active source tree,
    native fingerprint, and receipt hashes.  Replacing its historical pin only
    after this receipt is parsed prevents a profile from claiming that an older
    reviewed deadline implementation ran.
    """

    declared = receipt["source_files_sha256"]
    legacy.REVIEWED_DEADLINE_SOURCE_COMMIT = receipt["source_commit"]
    legacy.REVIEWED_ENGINE_SEARCH_SHA256 = declared["src/pokezero/engine_search.py"]
    legacy.REVIEWED_ENGINE_FINGERPRINT = receipt["engine_fingerprint"]
    if SELF_RELATIVE_PATH not in legacy.REQUIRED_RECEIPT_FILES:
        legacy.REQUIRED_RECEIPT_FILES = (*legacy.REQUIRED_RECEIPT_FILES, SELF_RELATIVE_PATH)

    historical_evidence = legacy._deadline_mechanics_evidence

    def source_bound_evidence(source_receipt: Mapping[str, Any]) -> dict[str, Any]:
        evidence = dict(historical_evidence(source_receipt))
        evidence.pop("reviewed_source_commit", None)
        evidence["source_bound_execution_commit"] = source_receipt["source_commit"]
        evidence["qualification_kind"] = "source_bound_deadline_profile"
        return evidence

    legacy._deadline_mechanics_evidence = source_bound_evidence


def _parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--source-receipt", required=True)
    args, _unknown = parser.parse_known_args(argv)
    return args


def main(argv: Sequence[str] | None = None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    try:
        receipt = _source_bound_receipt(_parse_args(arguments).source_receipt)
        _configure_legacy(receipt)
    except legacy.DeadlineQualificationError as error:
        print(f"CANNOT RUN SOURCE-BOUND DEADLINE PROFILE: {error}", file=sys.stderr)
        return 2
    return legacy.main(arguments)


if __name__ == "__main__":
    raise SystemExit(main())
