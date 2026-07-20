"""Capture compact public collision sketches from controlled FoulPlay games."""

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
    capture_controlled_foulplay_collision_sketch,
)


def build_collision_capture_arg_parser() -> argparse.ArgumentParser:
    parser = build_arg_parser()
    parser.prog = "pokezero-foulplay-collision-capture"
    parser.description = (
        "Capture compact public collision sketches from raw PokeZero p1 games against external FoulPlay. "
        "The output retains only deterministic hashes and replay locators, never model tensors or private requests."
    )
    _remove_optional_argument(parser, "--policy-mode")
    parser.set_defaults(policy_mode="raw")
    parser.add_argument("--out", type=Path, required=True, help="New compact collision sketch JSONL path to create.")
    parser.add_argument("--pool-id", default="controlled-foulplay-collision", help="Capture-pool provenance label.")
    return parser


async def async_main(argv: Sequence[str] | None = None) -> int:
    parser = build_collision_capture_arg_parser()
    args = parser.parse_args(argv)
    if args.summary_out is not None and args.summary_out.expanduser().resolve() == args.out.expanduser().resolve():
        parser.error("--summary-out must differ from --out so progress cannot replace the collision sketch.")
    if args.showdown_root is None:
        parser.error("--showdown-root is required unless POKEZERO_SHOWDOWN_ROOT is set.")
    if args.pokezero_player != "p1":
        parser.error("collision sketch capture supports only --pokezero-player p1.")
    if args.opponent_legal_mask_mode != "hidden":
        parser.error("collision sketch capture refuses --opponent-legal-mask-mode privileged.")
    config = _config_from_args(args, policy_mode="raw")

    def capture_progress(payload: dict) -> None:
        if args.summary_out is not None:
            _write_json(args.summary_out, payload)

    result = await capture_controlled_foulplay_collision_sketch(
        config,
        out_path=args.out,
        pool_id=args.pool_id,
        capture_progress_callback=capture_progress,
    )
    payload = result.to_dict()
    if args.summary_out is not None:
        _write_json(args.summary_out, payload)
        print(f"controlled_foulplay_collision_capture_summary: {args.summary_out}")
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        capture = payload["collision_sketch_capture"]
        print(
            f"captured {capture['captured_games']}/{config.games} labeled games and "
            f"{capture['captured_decisions']} public sketches to {args.out} (pool={args.pool_id}, policy=raw)"
        )
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    return asyncio.run(async_main(argv))


if __name__ == "__main__":
    raise SystemExit(main())
