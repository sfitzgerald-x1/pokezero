"""Command-line rollout collection utilities."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

from .collection import collect_rollouts, policy_from_name
from .local_showdown import LocalShowdownConfig, LocalShowdownEnv
from .rollout import RolloutConfig


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m pokezero.rollout_cli")
    subparsers = parser.add_subparsers(dest="command", required=True)

    collect = subparsers.add_parser("collect", help="Collect self-play rollout trajectories.")
    collect.add_argument("--games", type=int, required=True, help="Number of games to collect.")
    collect.add_argument("--out", type=Path, required=True, help="JSONL output path.")
    collect.add_argument("--showdown-root", type=Path, default=None, help="Built Pokemon Showdown checkout root.")
    collect.add_argument("--format", dest="format_id", default="gen3randombattle", help="Showdown format id.")
    collect.add_argument("--seed-start", type=int, default=1, help="First deterministic rollout seed.")
    collect.add_argument("--max-decision-rounds", type=int, default=250, help="Rollout decision-round cap.")
    collect.add_argument("--p1-policy", default="random-legal", help="Policy for p1.")
    collect.add_argument("--p2-policy", default="random-legal", help="Policy for p2.")
    collect.add_argument("--append", action="store_true", help="Append to the output JSONL instead of replacing it.")
    collect.add_argument("--node-binary", default="node", help="Node executable used for the BattleStream bridge.")
    collect.set_defaults(func=_collect)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


def _collect(args: argparse.Namespace) -> int:
    policies = {
        "p1": policy_from_name(args.p1_policy),
        "p2": policy_from_name(args.p2_policy),
    }
    env_config = LocalShowdownConfig(
        showdown_root=args.showdown_root,
        node_binary=args.node_binary,
    )
    rollout_config = RolloutConfig(
        max_decision_rounds=args.max_decision_rounds,
        format_id=args.format_id,
    )
    metrics = collect_rollouts(
        output_path=args.out,
        games=args.games,
        env_factory=lambda: LocalShowdownEnv(env_config),
        policies=policies,
        rollout_config=rollout_config,
        seed_start=args.seed_start,
        append=args.append,
    )
    _print_metrics(metrics.to_dict())
    return 0


def _print_metrics(metrics: dict[str, object]) -> None:
    print(f"games: {metrics['games']}")
    print(f"elapsed_seconds: {float(metrics['elapsed_seconds']):.3f}")
    print(f"games_per_second: {float(metrics['games_per_second']):.3f}")
    print(f"decisions_per_second: {float(metrics['decisions_per_second']):.3f}")
    print(f"p1_wins: {metrics['p1_wins']}")
    print(f"p2_wins: {metrics['p2_wins']}")
    print(f"ties: {metrics['ties']}")
    print(f"capped_games: {metrics['capped_games']}")
    print(f"average_decision_rounds: {float(metrics['average_decision_rounds']):.2f}")
    print(f"average_simulator_turns: {float(metrics['average_simulator_turns']):.2f}")


if __name__ == "__main__":
    raise SystemExit(main())
