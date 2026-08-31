#!/usr/bin/env python3
"""Fail-closed B2-pre proof for a continuation rollout in a live FoulPlay game.

The ordinary controlled-FoulPlay bridge is an online match driver.  Root-PUCT
search reconstructs branch environments from the live bridge trajectory, and a
positive ``root_puct_leaf_actual_rollout_rounds`` entry is the execution-side
evidence that a branch was continued after its root action.  This small driver
makes that evidence an explicit, durable capability receipt instead of asking a
reviewer to infer it from a general benchmark summary.

It is deliberately a smoke, not a strength evaluation: one FoulPlay game, one
mid-game PokeZero decision, and at least one realised continuation decision in
that decision's branch search.  The runner refuses fallbacks, an opening-only
decision, capped games, and zero-length continuations.
"""

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


SCHEMA_VERSION = "pokezero.live-foulplay-continuation-smoke.v1"
SUCCESS_MARKER = "WROTE LIVE FOULPLAY CONTINUATION SMOKE RECEIPT"


class ContinuationSmokeError(RuntimeError):
    """The requested live continuation proof was not observed."""


def _positive_actual_continuation_rounds(value: object) -> int:
    """Return a fail-closed total from Root-PUCT's per-leaf continuation metadata."""

    if not isinstance(value, Mapping):
        return 0
    total = 0
    for rounds, count in value.items():
        try:
            rounds_int = int(rounds)
            count_int = int(count)
        except (TypeError, ValueError):
            return 0
        if rounds_int < 0 or count_int < 0:
            return 0
        total += rounds_int * count_int
    return total


def continuation_proof_from_trajectory(
    trajectory: BattleTrajectory,
    *,
    pokezero_player: str,
    minimum_live_decision_round: int,
) -> dict[str, object]:
    """Extract one validated mid-game continuation proof from a bridge trajectory.

    ``trajectory`` is the callback payload produced only after the controlled
    bridge reaches a terminal result.  A static or self-play trajectory cannot
    satisfy the controlled-bridge provenance check, and an ordinary Root-PUCT
    choice with no realised tail cannot satisfy the positive-round check.
    """

    if minimum_live_decision_round < 1:
        raise ValueError("minimum_live_decision_round must be at least 1.")
    if trajectory.terminal is None:
        raise ContinuationSmokeError("live FoulPlay trajectory was not terminal")
    if trajectory.terminal.capped:
        raise ContinuationSmokeError("live FoulPlay trajectory capped before a smoke proof")
    if dict(trajectory.metadata).get("controlled_foulplay_bridge") is not True:
        raise ContinuationSmokeError("trajectory lacks controlled live-FoulPlay bridge provenance")

    for step in trajectory.steps:
        if step.player_id != pokezero_player or step.turn_index < minimum_live_decision_round:
            continue
        metadata = dict(step.metadata)
        if metadata.get("policy_family") != "root-puct-search":
            continue
        if metadata.get("root_puct_fallback"):
            raise ContinuationSmokeError(
                f"mid-game decision {step.turn_index} fell back instead of running continuation search"
            )
        configured = metadata.get("root_puct_leaf_rollout_rounds")
        try:
            configured_rounds = int(configured)
        except (TypeError, ValueError):
            configured_rounds = 0
        actual_rounds = _positive_actual_continuation_rounds(
            metadata.get("root_puct_leaf_actual_rollout_rounds")
        )
        total_visits = int(metadata.get("root_puct_total_visits") or 0)
        if configured_rounds <= 0 or total_visits <= 0 or actual_rounds <= 0:
            continue
        return {
            "battle_id": trajectory.battle_id,
            "seed": trajectory.seed,
            "pokezero_player": pokezero_player,
            "live_decision_round": step.turn_index,
            "policy_family": metadata["policy_family"],
            "root_puct_total_visits": total_visits,
            "configured_leaf_rollout_rounds": configured_rounds,
            "actual_leaf_continuation_decision_rounds": actual_rounds,
            "actual_leaf_rollout_rounds": dict(
                metadata["root_puct_leaf_actual_rollout_rounds"]
            ),
        }
    raise ContinuationSmokeError(
        "no non-fallback Root-PUCT continuation rollout was observed from a live mid-game FoulPlay state"
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_new_json(path: Path, payload: Mapping[str, object]) -> None:
    """Atomically create a receipt without replacing an existing artifact."""

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
    parser.add_argument("--root-visit-budget", type=int, default=1)
    return parser


async def _run(args: argparse.Namespace) -> dict[str, object]:
    if args.minimum_live_decision_round < 1:
        raise ValueError("--minimum-live-decision-round must be at least 1")
    if args.root_visit_budget <= 0:
        raise ValueError("--root-visit-budget must be positive")
    if args.foulplay_search_time_ms <= 0:
        raise ValueError("--foulplay-search-time-ms must be positive")
    if args.max_decision_rounds <= args.minimum_live_decision_round:
        raise ValueError("--max-decision-rounds must leave room for the required mid-game decision")

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
        policy_mode="root-puct",
        device=args.device,
        root_visit_budget=args.root_visit_budget,
        root_opponent_action_scenarios=1,
        root_opponent_action_candidate_scenarios=1,
        leaf_rollout_rounds=1,
        opponent_legal_mask_mode="hidden",
        allow_search_fallback=False,
        node_binary=args.node_binary,
        pokezero_player="p1",
        record_refusals=False,
    )
    result = await run_controlled_foulplay_benchmark(
        config,
        trajectory_callback=observed.append,
    )
    if len(result.games) != 1 or len(observed) != 1:
        raise ContinuationSmokeError(
            f"expected one completed live game and one callback, got games={len(result.games)} callbacks={len(observed)}"
        )
    proof = continuation_proof_from_trajectory(
        observed[0],
        pokezero_player=config.pokezero_player,
        minimum_live_decision_round=args.minimum_live_decision_round,
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "PASS",
        "purpose": "B2-pre live-FoulPlay continuation capability smoke",
        "success_marker": SUCCESS_MARKER,
        "runtime": {
            "checkpoint": str(args.checkpoint),
            "checkpoint_sha256": _sha256_file(args.checkpoint),
            "showdown_root": str(args.showdown_root),
            "foulplay_root": str(args.foulplay_root),
            "foulplay_search_time_ms": args.foulplay_search_time_ms,
            "seed": args.seed,
            "device": args.device,
            "root_visit_budget": args.root_visit_budget,
            "leaf_rollout_rounds": 1,
            "allow_search_fallback": False,
        },
        "proof": proof,
        "result": result.to_dict(),
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    payload = asyncio.run(_run(args))
    _write_new_json(args.out, payload)
    print(SUCCESS_MARKER)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
