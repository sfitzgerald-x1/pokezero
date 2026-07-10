"""Capture schema-matched external foul-play rollouts for value evaluation."""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
from typing import Sequence

from .foulplay_bridge import (
    _config_from_args,
    _remove_optional_argument,
    _write_json,
    build_arg_parser,
    capture_controlled_foulplay_rollouts,
)


def build_capture_arg_parser() -> argparse.ArgumentParser:
    parser = build_arg_parser()
    parser.prog = "pokezero-foulplay-capture"
    _remove_optional_argument(parser, "--policy-mode")
    parser.set_defaults(policy_mode="raw")
    parser.description = (
        "Capture raw PokeZero p1 trajectories against external foul-play. "
        "The output is a schema-matched rollout JSONL suitable for held-out value evaluation."
    )
    parser.add_argument("--out", type=Path, required=True, help="New rollout JSONL path to create.")
    parser.add_argument("--pool-id", default="controlled-foulplay", help="Capture-pool provenance label.")
    return parser


async def async_main(argv: Sequence[str] | None = None) -> int:
    parser = build_capture_arg_parser()
    args = parser.parse_args(argv)
    if args.showdown_root is None:
        parser.error("--showdown-root is required unless POKEZERO_SHOWDOWN_ROOT is set.")
    config = _config_from_args(args, policy_mode="raw")
    def capture_progress(payload: dict) -> None:
        if args.summary_out is not None:
            _write_json(args.summary_out, payload)

    result = await capture_controlled_foulplay_rollouts(
        config,
        out_path=args.out,
        pool_id=args.pool_id,
        capture_progress_callback=capture_progress,
    )
    payload = result.to_dict()
    if args.summary_out is not None:
        _write_json(args.summary_out, payload)
        print(f"controlled_foulplay_capture_summary: {args.summary_out}")
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print(
            f"captured {result.captured_games}/{config.games} labeled games to {args.out} "
            f"(pool={args.pool_id}, policy=raw)"
        )
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    return asyncio.run(async_main(argv))


if __name__ == "__main__":
    raise SystemExit(main())
