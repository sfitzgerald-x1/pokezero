"""One-game, source-bound live-continuation value-leaf collection command."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

from .foulplay_bridge import _config_from_args, _remove_optional_argument, _write_json, build_arg_parser
from .value_leaf_training import capture_live_foulplay_value_leaf_training_cache


def build_value_leaf_capture_arg_parser() -> argparse.ArgumentParser:
    parser = build_arg_parser()
    parser.prog = "pokezero-value-leaf-capture"
    _remove_optional_argument(parser, "--policy-mode")
    parser.set_defaults(policy_mode="raw")
    parser.description = (
        "Collect one source-bound, value-only cache shard from a completed live "
        "FoulPlay continuation-oracle game."
    )
    parser.add_argument("--out", type=Path, required=True, help="New cache directory to create.")
    parser.add_argument(
        "--source-binding-manifest",
        type=Path,
        required=True,
        help=(
            "Immutable JSON receipt containing a source_binding object. Its digest must match "
            "--source-binding-receipt-sha256."
        ),
    )
    parser.add_argument("--source-binding-receipt-sha256", required=True)
    return parser


async def async_main(argv: Sequence[str] | None = None) -> int:
    parser = build_value_leaf_capture_arg_parser()
    args = parser.parse_args(argv)
    if args.showdown_root is None:
        parser.error("--showdown-root is required unless POKEZERO_SHOWDOWN_ROOT is set.")
    if args.games != 1:
        parser.error("value-leaf collection requires --games 1 so each cache shard is durable.")
    if not args.live_continuation_oracle:
        parser.error("value-leaf collection requires --live-continuation-oracle.")
    if args.summary_out is not None and args.summary_out.resolve() == args.out.resolve():
        parser.error("--summary-out must not be the same path as --out.")
    config = _config_from_args(args, policy_mode="raw")
    try:
        source_binding = _source_binding_from_manifest(
            args.source_binding_manifest,
            expected_receipt_sha256=args.source_binding_receipt_sha256,
        )
    except ValueError as error:
        parser.error(str(error))
    result = await capture_live_foulplay_value_leaf_training_cache(
        config=config,
        output_path=args.out,
        source_binding=source_binding,
    )
    payload = result.to_dict()
    if args.summary_out is not None:
        try:
            _write_json(args.summary_out, payload)
        except Exception as error:
            # A cache is already atomically complete at this point.  Reporting
            # a sidecar failure as a failed collection would make safe retries
            # reject the completed shard, so leave a recoverable warning.
            print(
                f"value_leaf_capture_summary_write_failed: {args.summary_out}: {error}; "
                f"cache remains complete at {args.out}",
                file=sys.stderr,
            )
        else:
            print(f"value_leaf_capture_summary: {args.summary_out}", file=sys.stderr)
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print(f"captured {result.cache.capture_count} value leaves to {args.out}")
    return 0


def _source_binding_from_manifest(
    path: Path, *, expected_receipt_sha256: str
) -> Mapping[str, Any]:
    try:
        raw = path.read_bytes()
    except OSError as error:
        raise ValueError(f"cannot read source-binding manifest {path}: {error}") from error
    actual_sha256 = hashlib.sha256(raw).hexdigest()
    if actual_sha256.lower() != expected_receipt_sha256.lower():
        raise ValueError(
            "source-binding manifest SHA-256 does not match --source-binding-receipt-sha256"
        )
    try:
        document = json.loads(raw)
    except json.JSONDecodeError as error:
        raise ValueError(f"source-binding manifest is not valid JSON: {error}") from error
    if not isinstance(document, dict) or not isinstance(document.get("source_binding"), dict):
        raise ValueError("source-binding manifest must contain an object field source_binding")
    return dict(document["source_binding"])


def main(argv: Sequence[str] | None = None) -> int:
    return asyncio.run(async_main(argv))


if __name__ == "__main__":
    raise SystemExit(main())
