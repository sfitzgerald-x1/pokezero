#!/usr/bin/env python3
"""Run the non-bankable B2-pre live FoulPlay continuation capability smoke."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from pokezero.foulplay_bridge import ControlledFoulPlayConfig, run_controlled_foulplay_benchmark  # noqa: E402
from pokezero.trajectory import BattleTrajectory  # noqa: E402


SCHEMA_VERSION = "pokezero.live-foulplay-continuation-smoke.v2"
SUCCESS_MARKER = "WROTE B2-PRE LIVE FOULPLAY CONTINUATION SMOKE RECEIPT"


class ContinuationSmokeError(RuntimeError):
    """The B2-pre capability proof did not complete safely."""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_new_json(path: Path, payload: Mapping[str, object]) -> None:
    """Atomically create one receipt without replacing an existing artifact."""

    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"refusing to replace existing smoke receipt: {path}")
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary_path, path)
        except FileExistsError as error:
            raise FileExistsError(f"refusing to replace existing smoke receipt: {path}") from error
    finally:
        temporary_path.unlink(missing_ok=True)


def proof_from_trajectory(trajectory: BattleTrajectory) -> dict[str, object]:
    """Extract and validate the scorer-only proof written by the live bridge."""

    if trajectory.terminal is None:
        raise ContinuationSmokeError("live FoulPlay source game was not terminal")
    if trajectory.terminal.capped:
        raise ContinuationSmokeError("live FoulPlay source game capped before the smoke completed")
    metadata = dict(trajectory.metadata)
    if metadata.get("controlled_foulplay_bridge") is not True:
        raise ContinuationSmokeError("trajectory lacks controlled live-FoulPlay provenance")
    proof = metadata.get("live_foulplay_continuation_smoke")
    if not isinstance(proof, Mapping):
        raise ContinuationSmokeError("live source trajectory lacks a continuation proof")
    continuation = proof.get("continuation")
    if not isinstance(continuation, Mapping):
        raise ContinuationSmokeError("continuation proof lacks its terminal readout")
    if int(continuation.get("decision_round_count") or 0) <= 0:
        raise ContinuationSmokeError("continuation proof has no post-joint-action decision")
    terminal = continuation.get("terminal")
    if not isinstance(terminal, Mapping) or terminal.get("capped") is not False:
        raise ContinuationSmokeError("continuation proof is capped or lacks a terminal result")
    expected = {"p1", "p2"}
    if set(proof.get("source_request_sha256") or ()) != expected:
        raise ContinuationSmokeError("continuation proof does not bind both source requests")
    if set(proof.get("snapshot_request_sha256") or ()) != expected:
        raise ContinuationSmokeError("continuation proof does not bind both snapshot requests")
    joint_step = proof.get("first_restored_joint_step")
    if not isinstance(joint_step, Mapping) or set(joint_step) != expected:
        raise ContinuationSmokeError("continuation proof lacks the fixed restored joint step")
    if proof.get("continuation_policy_mode") != "raw":
        raise ContinuationSmokeError("continuation proof did not use fresh raw policies")
    if proof.get("full_state_snapshot_scope") != "scorer-only":
        raise ContinuationSmokeError("full-state snapshot reached a path other than the scorer")
    if not isinstance(proof.get("actual_foulplay_choice"), str) or not proof["actual_foulplay_choice"].strip():
        raise ContinuationSmokeError("continuation proof lacks the actual FoulPlay choice")
    if not isinstance(proof.get("decoded_actual_foulplay_action"), int):
        raise ContinuationSmokeError("continuation proof lacks the decoded FoulPlay action")
    if int(proof.get("source_decision_round") or 0) < 1:
        raise ContinuationSmokeError("continuation proof is not from a mid-game source boundary")
    return dict(proof)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--showdown-root", type=Path, required=True)
    parser.add_argument("--foulplay-root", type=Path, required=True)
    parser.add_argument("--foulplay-python", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--node-binary", default="node")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--foulplay-search-time-ms", type=int, default=1000)
    parser.add_argument("--max-decision-rounds", type=int, default=64)
    parser.add_argument("--minimum-live-decision-round", type=int, default=1)
    return parser


async def _run(args: argparse.Namespace) -> dict[str, object]:
    observed: list[BattleTrajectory] = []
    config = ControlledFoulPlayConfig(
        checkpoint=args.checkpoint,
        showdown_root=args.showdown_root,
        foulplay_root=args.foulplay_root,
        foulplay_python=args.foulplay_python,
        games=1,
        seed_start=args.seed,
        foulplay_random_seed=args.seed,
        search_time_ms=args.foulplay_search_time_ms,
        max_decision_rounds=args.max_decision_rounds,
        policy_mode="raw",
        device=args.device,
        opponent_legal_mask_mode="hidden",
        allow_search_fallback=False,
        node_binary=args.node_binary,
        pokezero_player="p1",
        record_refusals=False,
        live_continuation_smoke=True,
        live_continuation_minimum_decision_round=args.minimum_live_decision_round,
    )
    result = await run_controlled_foulplay_benchmark(
        config,
        trajectory_callback=observed.append,
    )
    if len(result.games) != 1 or len(observed) != 1:
        raise ContinuationSmokeError(
            f"expected one completed source game and one callback, got games={len(result.games)} callbacks={len(observed)}"
        )
    proof = proof_from_trajectory(observed[0])
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "PASS",
        "purpose": "B2-pre live-FoulPlay continuation capability smoke",
        "non_bankable": True,
        "does_not_license": ["B2", "B3", "B4"],
        "success_marker": SUCCESS_MARKER,
        "runtime": {
            "checkpoint": str(args.checkpoint),
            "checkpoint_sha256": _sha256_file(args.checkpoint),
            "showdown_root": str(args.showdown_root),
            "foulplay_root": str(args.foulplay_root),
            "foulplay_search_time_ms": args.foulplay_search_time_ms,
            "seed": args.seed,
            "device": args.device,
            "minimum_live_decision_round": args.minimum_live_decision_round,
        },
        "proof": proof,
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    payload = asyncio.run(_run(args))
    _write_new_json(args.out, payload)
    print(SUCCESS_MARKER)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
