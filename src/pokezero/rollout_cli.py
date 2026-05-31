"""Command-line rollout collection utilities."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from .collection import BenchmarkReport, benchmark_rollouts, collect_rollouts, policy_from_name
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

    benchmark = subparsers.add_parser("benchmark", help="Run baseline rollout throughput benchmarks without writing trajectories.")
    benchmark.add_argument(
        "--games",
        type=int,
        default=20,
        help="Number of games to run per baseline matchup. Default is a smoke size; use hundreds for quality comparisons.",
    )
    benchmark.add_argument("--showdown-root", type=Path, default=None, help="Built Pokemon Showdown checkout root.")
    benchmark.add_argument("--format", dest="format_id", default="gen3randombattle", help="Showdown format id.")
    benchmark.add_argument("--seed-start", type=int, default=1, help="First deterministic rollout seed for every matchup.")
    benchmark.add_argument("--max-decision-rounds", type=int, default=250, help="Rollout decision-round cap.")
    benchmark.add_argument("--node-binary", default="node", help="Node executable used for the BattleStream bridge.")
    benchmark.add_argument("--json", action="store_true", help="Print benchmark results as JSON.")
    benchmark.set_defaults(func=_benchmark)
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


def _benchmark(args: argparse.Namespace) -> int:
    env_config = LocalShowdownConfig(
        showdown_root=args.showdown_root,
        node_binary=args.node_binary,
    )
    rollout_config = RolloutConfig(
        max_decision_rounds=args.max_decision_rounds,
        format_id=args.format_id,
    )
    report = benchmark_rollouts(
        games=args.games,
        env_factory=lambda: LocalShowdownEnv(env_config),
        rollout_config=rollout_config,
        seed_start=args.seed_start,
    )
    if args.json:
        print(json.dumps(report.to_dict(), indent=2, sort_keys=True))
    else:
        _print_benchmark_report(report)
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


def _print_benchmark_report(report: BenchmarkReport) -> None:
    print(f"format: {report.format_id}")
    print(f"games_per_matchup: {report.games_per_matchup}")
    print(f"max_decision_rounds: {report.max_decision_rounds}")
    print(f"total_games: {report.total_games}")
    print(f"elapsed_seconds: {report.elapsed_seconds:.3f}")
    print(f"games_per_second: {report.games_per_second:.3f}")
    print(f"decisions_per_second: {report.decisions_per_second:.3f}")
    print("note: default --games is a throughput smoke; use larger N for policy-quality claims.")
    print("")
    header = (
        f"{'matchup':32} {'games':>5} {'p1_wins':>7} {'p2_wins':>7} {'ties':>4} {'capped':>6} "
        f"{'avg_dec':>8} {'avg_turns':>9} {'games/s':>8} {'dec/s':>8}"
    )
    print(header)
    print("-" * len(header))
    for result in report.matchups:
        metrics = result.metrics
        print(
            f"{result.label[:32]:32} "
            f"{metrics.games:5d} "
            f"{metrics.p1_wins:7d} "
            f"{metrics.p2_wins:7d} "
            f"{metrics.ties:4d} "
            f"{metrics.capped_games:6d} "
            f"{metrics.average_decision_rounds:8.2f} "
            f"{metrics.average_simulator_turns:9.2f} "
            f"{metrics.games_per_second:8.3f} "
            f"{metrics.decisions_per_second:8.3f}"
        )
    if not report.head_to_head_results:
        return
    print("")
    head_to_head_header = (
        f"{'mirror head-to-head':32} {'games':>5} {'first_w':>7} {'second_w':>8} {'ties':>4} {'capped':>6} "
        f"{'first_wr':>8} {'second_wr':>9}"
    )
    print(head_to_head_header)
    print("-" * len(head_to_head_header))
    for result in report.head_to_head_results:
        print(
            f"{result.label[:32]:32} "
            f"{result.games:5d} "
            f"{result.first_policy_wins:7d} "
            f"{result.second_policy_wins:8d} "
            f"{result.ties:4d} "
            f"{result.capped_games:6d} "
            f"{result.first_policy_win_rate:8.3f} "
            f"{result.second_policy_win_rate:9.3f}"
        )


if __name__ == "__main__":
    raise SystemExit(main())
