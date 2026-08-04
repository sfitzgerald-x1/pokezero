#!/usr/bin/env python3
"""Depth grid for engine search with the HANDCRAFTED leaf evaluation.

The great divider for docs/mcts_degradation_findings.md. That study measured a
monotone decay of engine MCTS against its own raw policy as rollout depth grows
(d1 0.53 -> d6 0.36), reproduced on two builds ~115 commits apart, with the two
leading mechanisms (cross-world strategy fusion, dynamics divergence) both
falsified. What remains untested is whether the decay lives in the SEARCH
STRUCTURE or in the LEARNED VALUE at the leaves.

This harness runs the same depth grid with the crate's own PUCT tree but the
handcrafted HP-fraction leaf evaluator instead of the network
(``EngineMctsConfig.leaf_eval='hp_fraction_crate'``). Everything about the tree
is held fixed - identical ``traverse``/``expand``/``finalize``, identical
decision/chance node shape, identical depth and c_puct semantics - so the only
thing that changed is what prices a leaf.

Read the SLOPE, not the level. Handcrafted-search-vs-raw-NN-policy is a
different opponent pairing than NN-search-vs-raw-NN-policy, so the absolute
scores are not comparable to the findings doc:

  (a) hc decays with depth like the NN arm  -> the defect is structural (backup
      at simultaneous/chance nodes, decoupled selection), independent of the
      value function;
  (b) hc is flat or rising with depth       -> the defect is NN-value-specific
      (orientation, or off-distribution leaves at depth).

Seats are mirrored by seed parity (``seed % 2``), matching the findings doc, so
seat advantage cancels within a cell and cells are paired on seeds.

Also reports the depth ACTUALLY reached per decision (the crate's
``max_depth_reached``), which the findings doc flags as untested after d6 and d8
produced identical per-seed outcomes. Note the off-by-one: the cap bounds child
CREATION (``depth + 1 >= max_depth``), and the counter is a node depth with the
root at 0, so a binding cap ``d`` shows ``max_depth_reached == d - 1``.

Usage:

    PYTHONPATH=src python scripts/hc_depth_grid.py \
        --checkpoint checkpoints/pz-v2-2-1m.pt \
        --showdown-root ~/workspace/pokerena/vendor/pokemon-showdown \
        --cells control,hc-d1,hc-d2,hc-d4,hc-d6 \
        --seed-start 600000 --games 100 --sims 1024 --worlds 4 \
        --out runs/hc-depth-grid
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import sys
import time
from typing import Any

_REPO = Path(__file__).resolve().parents[1]
_SRC = _REPO / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))


def wilson_interval(wins: float, n: int, z: float = 1.959963984540054) -> tuple[float, float]:
    """Wilson score interval for a proportion.

    ``wins`` may be fractional (draws score 0.5); the interval is then an
    approximation that treats the score as if it came from ``n`` Bernoulli
    trials, which is what the findings doc's table does.
    """

    if n <= 0:
        return (0.0, 0.0)
    p = wins / n
    denom = 1.0 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = (z / denom) * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (max(0.0, center - half), min(1.0, center + half))


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--checkpoint", required=True, help="Raw-policy checkpoint (the opponent).")
    parser.add_argument("--showdown-root", default=os.environ.get("POKEZERO_SHOWDOWN_ROOT"))
    parser.add_argument("--out", required=True, help="Output directory for per-cell JSON.")
    parser.add_argument(
        "--cells",
        default="control,hc-d1,hc-d2,hc-d4,hc-d6",
        help=(
            "Comma-separated cells: 'control' (raw v raw), 'hc-d<N>' (handcrafted-leaf "
            "crate search at depth N), or 'vs:<policy spec>' to seat an arbitrary "
            "baseline in the candidate slot (e.g. 'vs:max-damage') for calibration."
        ),
    )
    parser.add_argument("--seed-start", type=int, default=600_000)
    parser.add_argument("--games", type=int, default=100)
    parser.add_argument("--sims", type=int, default=1024)
    parser.add_argument("--worlds", type=int, default=4)
    parser.add_argument("--c-puct", type=float, default=1.4)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--format", dest="format_id", default="gen3randombattle")
    parser.add_argument("--max-decision-rounds", type=int, default=250)
    parser.add_argument("--node-binary", default="node")
    parser.add_argument("--shard-id", type=int, default=0)
    parser.add_argument("--shards", type=int, default=1)
    parser.add_argument(
        "--deep-ko-split", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument(
        "--record-depths",
        action="store_true",
        help="Persist the per-decision max_depth_reached series for every game.",
    )
    return parser


_BASELINE_CELL_PREFIX = "vs:"


def _cell_depth(cell: str) -> int | None:
    """Search depth for a cell, or None when the cell seats a plain policy."""

    if cell == "control" or cell.startswith(_BASELINE_CELL_PREFIX):
        return None
    if not cell.startswith("hc-d"):
        raise ValueError(
            f"unknown cell {cell!r}; expected 'control', 'hc-d<N>' or 'vs:<spec>'."
        )
    return int(cell[len("hc-d") :])


def _provenance(args: argparse.Namespace) -> dict[str, Any]:
    import poke_engine
    import pokezero_search
    import pokezero
    from pokezero.paths import portable_path

    payload: dict[str, Any] = {
        "python": sys.version.split()[0],
        # portable_path, not str(...resolve()): this payload is COMMITTED as an audit
        # artifact, and an absolute module path puts a username in a public repo.
        "pokezero_module": portable_path(pokezero.__file__),
        "pokezero_search_module": portable_path(pokezero_search.__file__),
        "pokezero_search_engine_features": pokezero_search.ENGINE_FEATURES,
        "pokezero_search_model_feature": bool(pokezero_search.MODEL_FEATURE_ENABLED),
        "poke_engine_module": portable_path(poke_engine.__file__),
    }
    fingerprint = Path(sys.prefix) / ".engine-build-fingerprint.json"
    if fingerprint.is_file():
        payload["engine_build_fingerprint"] = json.loads(fingerprint.read_text())
    return payload


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    if not args.showdown_root:
        raise SystemExit("--showdown-root (or POKEZERO_SHOWDOWN_ROOT) is required.")
    os.environ.setdefault("POKEZERO_SHOWDOWN_ROOT", str(args.showdown_root))

    from pokezero.collection import (
        env_config_with_policy_spec_masks,
        policy_from_spec,
        run_rollout_record_on_env,
    )
    from pokezero.dex import load_showdown_dex_cached
    from pokezero.engine_search import EngineMctsConfig, EngineMctsPolicy
    from pokezero.local_showdown import (
        LocalShowdownConfig,
        LocalShowdownEnv,
        belief_set_source_env_enabled,
    )
    from pokezero.randbat import load_gen3_randbat_source_cached
    from pokezero.rollout import RolloutConfig

    raw_spec = f"neural:{args.checkpoint}?deterministic=true&device={args.device}"
    env_config = LocalShowdownConfig(
        showdown_root=Path(args.showdown_root), node_binary=args.node_binary
    )
    env_config = env_config_with_policy_spec_masks(
        env_config, (raw_spec,), context="hc depth grid"
    )
    rollout_config = RolloutConfig(
        max_decision_rounds=args.max_decision_rounds, format_id=args.format_id
    )

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    seeds = [
        args.seed_start + index
        for index in range(args.games)
        if index % args.shards == args.shard_id
    ]

    dex = load_showdown_dex_cached(str(args.showdown_root))
    set_source = (
        load_gen3_randbat_source_cached(str(args.showdown_root))
        if belief_set_source_env_enabled()
        else None
    )

    provenance = _provenance(args)
    print(json.dumps(provenance, indent=2, sort_keys=True), flush=True)

    for cell in [item.strip() for item in args.cells.split(",") if item.strip()]:
        depth = _cell_depth(cell)
        # One policy instance per cell, reused across games (matches
        # benchmark_rollouts): the engine policy keys its per-battle state by
        # (battle_id, seat), so a seat that flips with seed parity is safe.
        raw_policy = policy_from_spec(raw_spec)
        if cell.startswith(_BASELINE_CELL_PREFIX):
            candidate: Any = policy_from_spec(cell[len(_BASELINE_CELL_PREFIX) :])
        elif depth is None:
            candidate = policy_from_spec(raw_spec)
        else:
            candidate = EngineMctsPolicy(
                dex=dex,
                set_source=set_source,
                config=EngineMctsConfig(
                    leaf_eval="hp_fraction_crate",
                    worlds=args.worlds,
                    search_sims=args.sims,
                    search_depth=depth,
                    c_puct=args.c_puct,
                    deep_ko_split=args.deep_ko_split,
                ),
                policy_id=f"engine-mcts-hc-d{depth}-s{args.sims}",
            )

        env = LocalShowdownEnv(env_config)
        results: list[dict[str, Any]] = []
        cell_started = time.perf_counter()
        try:
            for seed in seeds:
                # Mirrored seats by seed parity, as in the findings doc.
                candidate_seat = "p1" if seed % 2 == 0 else "p2"
                opponent_seat = "p2" if candidate_seat == "p1" else "p1"
                policies = {candidate_seat: candidate, opponent_seat: raw_policy}
                record = run_rollout_record_on_env(
                    env=env,
                    policies=policies,
                    rollout_config=rollout_config,
                    seed=seed,
                    battle_id=f"hcgrid-{cell}-{seed}",
                )
                winner = record.terminal.winner
                score = 1.0 if winner == candidate_seat else 0.0 if winner else 0.5
                depths = [
                    int(step.metadata["engine_mcts"]["max_depth_reached"])
                    for step in record.trajectory.steps
                    if step.player_id == candidate_seat
                    and isinstance(step.metadata.get("engine_mcts"), dict)
                    and "max_depth_reached" in step.metadata["engine_mcts"]
                ]
                row: dict[str, Any] = {
                    "seed": seed,
                    "candidate_seat": candidate_seat,
                    "winner": winner,
                    "score": score,
                    "decisions": len(record.trajectory.steps),
                    "elapsed_seconds": record.elapsed_seconds,
                }
                if depths:
                    row["depth_reached_max"] = max(depths)
                    row["depth_reached_mean"] = sum(depths) / len(depths)
                    row["depth_reached_decisions"] = len(depths)
                    if args.record_depths:
                        row["depth_reached_series"] = depths
                results.append(row)
                wins = sum(item["score"] for item in results)
                print(
                    f"[{cell}] seed {seed} seat {candidate_seat} winner {winner} "
                    f"running {wins:.1f}/{len(results)}",
                    flush=True,
                )
        finally:
            close = getattr(env, "close", None)
            if callable(close):
                close()

        wins = sum(item["score"] for item in results)
        low, high = wilson_interval(wins, len(results))
        payload = {
            "cell": cell,
            "depth": depth,
            "sims": args.sims if depth is not None else None,
            "worlds": args.worlds if depth is not None else None,
            "c_puct": args.c_puct if depth is not None else None,
            "deep_ko_split": args.deep_ko_split if depth is not None else None,
            "checkpoint": str(args.checkpoint),
            "raw_spec": raw_spec,
            "seed_start": args.seed_start,
            "games": len(results),
            "score": wins / len(results) if results else 0.0,
            "wilson95": [low, high],
            "wall_seconds": time.perf_counter() - cell_started,
            "provenance": provenance,
            "results": results,
        }
        if depth is not None and hasattr(candidate, "stats"):
            payload["engine_stats"] = candidate.stats.to_dict()
        target = out_dir / f"{cell.replace(':', '-').replace('/', '_')}.json"
        target.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        print(
            f"[{cell}] score {payload['score']:.3f} "
            f"[{low:.3f}, {high:.3f}] n={len(results)} -> {target}",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
