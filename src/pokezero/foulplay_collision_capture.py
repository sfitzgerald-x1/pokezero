"""Capture compact public collision sketches from controlled FoulPlay games."""

from __future__ import annotations

import argparse
import asyncio
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
from typing import Mapping, Sequence

from .audit_provenance import public_repo_commit
from .deep_line_audit import PROTOCOL_SIGNATURE_SCHEMA_VERSION
from .foulplay_bridge import (
    _config_from_args,
    _remove_optional_argument,
    _write_json,
    build_arg_parser,
    capture_controlled_foulplay_collision_sketch,
)
from .randbat import load_gen3_randbat_source_cached


ROOT = Path(__file__).resolve().parents[2]


def build_collision_capture_arg_parser() -> argparse.ArgumentParser:
    parser = build_arg_parser()
    parser.prog = "pokezero-foulplay-collision-capture"
    parser.description = (
        "Capture compact public collision sketches from raw PokeZero p1 games against external FoulPlay. "
        "The output retains only deterministic hashes and replay locators, never model tensors or private requests."
    )
    _remove_optional_argument(parser, "--policy-mode")
    _remove_optional_argument(parser, "--checkpoint")
    parser.set_defaults(policy_mode="raw")
    parser.add_argument(
        "--capture-driver",
        choices=("checkpoint", "random-legal"),
        default="checkpoint",
        help=(
            "Decision policy used only to explore FoulPlay states. 'random-legal' is an explicitly "
            "untrained v3 audit driver and never represents a strength measurement."
        ),
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Transformer checkpoint for --capture-driver checkpoint; forbidden for random-legal.",
    )
    parser.add_argument(
        "--observation-schema",
        choices=("v3",),
        default=None,
        help="Required for --capture-driver random-legal so its captured inputs are schema-v3.",
    )
    parser.add_argument("--out", type=Path, required=True, help="New compact collision sketch JSONL path to create.")
    parser.add_argument("--pool-id", default="controlled-foulplay-collision", help="Capture-pool provenance label.")
    return parser


def _resume_protocol_signature_census(
    *,
    sketch_path: Path,
    summary_path: Path | None,
    expected_provenance: Mapping[str, object],
    expected_pool_id: str,
) -> tuple[dict[str, int], tuple[str, ...]]:
    """Load retry-safe count-only census state from the atomic progress summary.

    A nonempty sketch without a compatible summary cannot be safely resumed:
    the sketch intentionally omits protocol lines, so recreating its aggregate
    counts would otherwise silently under- or over-count prior games.
    """

    if summary_path is None or not summary_path.exists():
        if sketch_path.exists():
            raise ValueError(
                "collision sketch exists without a resumable protocol census summary; "
                "use a new --out/--summary-out pair rather than publishing incomplete counts"
            )
        return {}, ()
    if not sketch_path.is_file():
        raise ValueError(
            "resumable protocol census summary has no matching collision sketch; "
            "use a new --out/--summary-out pair"
        )
    try:
        payload = json.loads(summary_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read resumable protocol census summary {summary_path}: {exc}") from exc
    if payload.get("protocol_signature_schema_version") != PROTOCOL_SIGNATURE_SCHEMA_VERSION:
        raise ValueError(
            "resumable protocol census summary has an incompatible signature schema; "
            "use a new --out/--summary-out pair"
        )
    provenance = payload.get("audit_provenance")
    immutable_provenance_fields = (
        "schema_version",
        "public_repo_commit",
        "showdown_source_hash",
        "observation_schema",
        "image_digest",
        "execution_scope",
    )
    if not isinstance(provenance, Mapping) or any(
        provenance.get(field) != expected_provenance.get(field) for field in immutable_provenance_fields
    ):
        raise ValueError(
            "resumable protocol census summary has incompatible source, schema, or image provenance; "
            "use a new --out/--summary-out pair"
        )
    counts = payload.get("protocol_signatures")
    game_ids = payload.get("protocol_signature_game_ids")
    capture = payload.get("collision_sketch_capture")
    recorded_sketch_path = capture.get("out") if isinstance(capture, Mapping) else None
    recorded_pool_id = capture.get("pool_id") if isinstance(capture, Mapping) else None
    if (
        not isinstance(counts, Mapping)
        or not isinstance(game_ids, list)
        or not isinstance(recorded_sketch_path, str)
        or Path(recorded_sketch_path).expanduser().resolve() != sketch_path.expanduser().resolve()
        or recorded_pool_id != expected_pool_id
    ):
        raise ValueError(
            "resumable protocol census summary does not match this sketch or is missing count-only state; "
            "use a new --out/--summary-out pair"
        )
    return dict(counts), tuple(game_ids)


def _protocol_census_provenance(
    *,
    source_hash: str,
    command_arguments: Sequence[str],
    seed_start: int,
    games: int,
    capture_driver: str,
    max_decision_rounds: int,
) -> dict[str, object]:
    """Return the immutable provenance carried by progress and complete summaries."""

    return {
        "schema_version": "pokezero.protocol-capture-provenance.v1",
        "recorded_at": datetime.now(timezone.utc).isoformat(),
        "public_repo_commit": public_repo_commit(ROOT),
        "showdown_source_hash": source_hash,
        "observation_schema": "pokezero.observation.v3",
        "image_digest": os.environ.get("POKEZERO_AUDIT_IMAGE_DIGEST", "local-uncontainerized"),
        "command": [str(Path(__file__).relative_to(ROOT)), *command_arguments],
        "execution_scope": {
            "seed_range": {"start": seed_start, "end": seed_start + games - 1, "count": games},
            "capture_driver": capture_driver,
            "max_decision_rounds": max_decision_rounds,
        },
    }


async def async_main(argv: Sequence[str] | None = None) -> int:
    command_arguments = list(argv) if argv is not None else list(sys.argv[1:])
    parser = build_collision_capture_arg_parser()
    args = parser.parse_args(command_arguments)
    if args.summary_out is not None and args.summary_out.expanduser().resolve() == args.out.expanduser().resolve():
        parser.error("--summary-out must differ from --out so progress cannot replace the collision sketch.")
    if args.showdown_root is None:
        parser.error("--showdown-root is required unless POKEZERO_SHOWDOWN_ROOT is set.")
    if args.pokezero_player != "p1":
        parser.error("collision sketch capture supports only --pokezero-player p1.")
    if args.opponent_legal_mask_mode != "hidden":
        parser.error("collision sketch capture refuses --opponent-legal-mask-mode privileged.")
    if args.capture_driver == "checkpoint":
        if args.checkpoint is None:
            parser.error("--checkpoint is required for --capture-driver checkpoint.")
        if args.observation_schema is not None:
            parser.error("--observation-schema is only valid for --capture-driver random-legal.")
    else:
        if args.checkpoint is not None:
            parser.error("--checkpoint is forbidden for --capture-driver random-legal.")
        if args.observation_schema != "v3":
            parser.error("--capture-driver random-legal requires --observation-schema v3.")
    config = _config_from_args(args, policy_mode="raw")
    source = load_gen3_randbat_source_cached(config.showdown_root)
    provenance = _protocol_census_provenance(
        source_hash=source.metadata.source_hash,
        command_arguments=command_arguments,
        seed_start=args.seed_start,
        games=args.games,
        capture_driver=args.capture_driver,
        max_decision_rounds=args.max_decision_rounds,
    )
    try:
        initial_protocol_signatures, initial_protocol_signature_game_ids = _resume_protocol_signature_census(
            sketch_path=args.out,
            summary_path=args.summary_out,
            expected_provenance=provenance,
            expected_pool_id=args.pool_id,
        )
    except ValueError as exc:
        parser.error(str(exc))

    def capture_progress(payload: dict) -> None:
        if args.summary_out is not None:
            payload["audit_provenance"] = provenance
            _write_json(args.summary_out, payload)

    result = await capture_controlled_foulplay_collision_sketch(
        config,
        out_path=args.out,
        pool_id=args.pool_id,
        capture_progress_callback=capture_progress,
        initial_protocol_signatures=initial_protocol_signatures,
        initial_protocol_signature_game_ids=initial_protocol_signature_game_ids,
    )
    payload = result.to_dict()
    if (
        result.observation_schema_version is not None
        and result.observation_schema_version != provenance["observation_schema"]
    ):
        raise RuntimeError("collision sketch capture observation schema disagrees with its provenance")
    payload["audit_provenance"] = provenance
    if args.summary_out is not None:
        _write_json(args.summary_out, payload)
        print(f"controlled_foulplay_collision_capture_summary: {args.summary_out}")
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        capture = payload["collision_sketch_capture"]
        print(
            f"captured {capture['captured_games']}/{config.games} labeled games and "
            f"{capture['captured_decisions']} public sketches to {args.out} "
            f"(pool={args.pool_id}, driver={config.capture_driver})"
        )
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    return asyncio.run(async_main(argv))


if __name__ == "__main__":
    raise SystemExit(main())
