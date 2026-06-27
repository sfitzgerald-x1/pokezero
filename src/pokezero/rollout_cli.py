"""Command-line rollout collection utilities."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from .collection import (
    BenchmarkReport,
    benchmark_rollouts,
    collect_rollouts,
    policy_benchmark_matchups,
    policy_from_spec,
    policy_spec_with_showdown_root,
)
from .local_showdown import LocalShowdownConfig, LocalShowdownEnv
from .replay_benchmark import ReplayPrefixBenchmarkReport, benchmark_replay_prefixes
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
    policy_help = (
        "Policy spec. Supports random-legal, simple-legal, scripted-teacher, aggressive-damage, "
        "linear:/path/to/checkpoint.json, or neural:/path/to/checkpoint.pt."
    )
    collect.add_argument("--p1-policy", default="random-legal", help=f"Policy for p1. {policy_help}")
    collect.add_argument("--p2-policy", default="random-legal", help=f"Policy for p2. {policy_help}")
    collect.add_argument("--append", action="store_true", help="Append to the output JSONL instead of replacing it.")
    collect.add_argument("--node-binary", default="node", help="Node executable used for the BattleStream bridge.")
    collect.set_defaults(func=_collect)

    benchmark = subparsers.add_parser("benchmark", help="Run rollout benchmarks without writing trajectories.")
    benchmark.add_argument(
        "--games",
        type=int,
        default=20,
        help="Number of games to run per matchup. Default is a smoke size; use hundreds for quality comparisons.",
    )
    benchmark.add_argument("--showdown-root", type=Path, default=None, help="Built Pokemon Showdown checkout root.")
    benchmark.add_argument("--format", dest="format_id", default="gen3randombattle", help="Showdown format id.")
    benchmark.add_argument("--seed-start", type=int, default=1, help="First deterministic rollout seed for every matchup.")
    benchmark.add_argument("--max-decision-rounds", type=int, default=250, help="Rollout decision-round cap.")
    benchmark.add_argument("--node-binary", default="node", help="Node executable used for the BattleStream bridge.")
    benchmark.add_argument(
        "--policy",
        action="append",
        default=None,
        help=(
            "Candidate policy spec to benchmark. May be repeated. When omitted, the command runs "
            "the default baseline throughput matchups."
        ),
    )
    benchmark.add_argument(
        "--opponent-policy",
        action="append",
        default=None,
        help=(
            "Opponent policy spec for custom --policy benchmarks. May be repeated. "
            "Defaults to random-legal and simple-legal."
        ),
    )
    benchmark.add_argument(
        "--include-policy-head-to-head",
        action="store_true",
        help="Also run mirrored candidate-vs-candidate matchups when multiple --policy values are provided.",
    )
    benchmark.add_argument("--json", action="store_true", help="Print benchmark results as JSON.")
    benchmark.set_defaults(func=_benchmark)

    replay_benchmark = subparsers.add_parser(
        "replay-benchmark",
        help="Measure replay-from-root prefix latency for future search/MCTS planning.",
    )
    replay_benchmark.add_argument("--games", type=int, default=3, help="Number of source games to generate.")
    replay_benchmark.add_argument(
        "--prefixes-per-game",
        type=int,
        default=5,
        help="Evenly sampled branch-prefix lengths to replay per source game.",
    )
    replay_benchmark.add_argument("--showdown-root", type=Path, default=None, help="Built Pokemon Showdown checkout root.")
    replay_benchmark.add_argument("--format", dest="format_id", default="gen3randombattle", help="Showdown format id.")
    replay_benchmark.add_argument("--seed-start", type=int, default=1, help="First deterministic rollout seed.")
    replay_benchmark.add_argument("--max-decision-rounds", type=int, default=250, help="Source rollout decision-round cap.")
    replay_benchmark.add_argument("--node-binary", default="node", help="Node executable used for the BattleStream bridge.")
    replay_benchmark.add_argument("--p1-policy", default="random-legal", help=f"Source policy for p1. {policy_help}")
    replay_benchmark.add_argument("--p2-policy", default="random-legal", help=f"Source policy for p2. {policy_help}")
    replay_benchmark.add_argument("--json", action="store_true", help="Print replay benchmark results as JSON.")
    replay_benchmark.set_defaults(func=_replay_benchmark)
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
    env_config = LocalShowdownConfig(
        showdown_root=args.showdown_root,
        node_binary=args.node_binary,
    )
    policy_showdown_root = env_config.resolved_showdown_root()
    policies = {
        "p1": policy_from_spec(policy_spec_with_showdown_root(args.p1_policy, policy_showdown_root)),
        "p2": policy_from_spec(policy_spec_with_showdown_root(args.p2_policy, policy_showdown_root)),
    }
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
    policy_showdown_root = env_config.resolved_showdown_root()
    rollout_config = RolloutConfig(
        max_decision_rounds=args.max_decision_rounds,
        format_id=args.format_id,
    )
    matchups = None
    if args.policy:
        matchups = policy_benchmark_matchups(
            policy_specs=args.policy,
            opponent_policy_specs=args.opponent_policy or ("random-legal", "simple-legal"),
            showdown_root=policy_showdown_root,
            include_policy_head_to_head=args.include_policy_head_to_head,
        )
    elif args.opponent_policy:
        raise ValueError("--opponent-policy requires at least one --policy.")
    elif args.include_policy_head_to_head:
        raise ValueError("--include-policy-head-to-head requires at least two --policy values.")

    report = benchmark_rollouts(
        games=args.games,
        env_factory=lambda: LocalShowdownEnv(env_config),
        rollout_config=rollout_config,
        seed_start=args.seed_start,
        matchups=matchups,
    )
    if args.json:
        print(json.dumps(report.to_dict(), indent=2, sort_keys=True))
    else:
        print_benchmark_report(report)
    return 0


def _replay_benchmark(args: argparse.Namespace) -> int:
    env_config = LocalShowdownConfig(
        showdown_root=args.showdown_root,
        node_binary=args.node_binary,
    )
    policy_showdown_root = env_config.resolved_showdown_root()
    policies = {
        "p1": policy_from_spec(policy_spec_with_showdown_root(args.p1_policy, policy_showdown_root)),
        "p2": policy_from_spec(policy_spec_with_showdown_root(args.p2_policy, policy_showdown_root)),
    }
    rollout_config = RolloutConfig(
        max_decision_rounds=args.max_decision_rounds,
        format_id=args.format_id,
    )
    report = benchmark_replay_prefixes(
        env_factory=lambda: LocalShowdownEnv(env_config),
        policies=policies,
        rollout_config=rollout_config,
        games=args.games,
        prefixes_per_game=args.prefixes_per_game,
        seed_start=args.seed_start,
    )
    if args.json:
        print(json.dumps(report.to_dict(), indent=2, sort_keys=True))
    else:
        print_replay_benchmark_report(report)
    return 0


def _print_metrics(metrics: dict[str, object]) -> None:
    print(f"games: {metrics['games']}")
    print(f"elapsed_seconds: {float(metrics['elapsed_seconds']):.3f}")
    print(f"games_per_second: {float(metrics['games_per_second']):.3f}")
    print(f"decisions_per_second: {float(metrics['decisions_per_second']):.3f}")
    if metrics.get("peak_rss_mb") is not None:
        print(f"peak_rss_mb: {float(metrics['peak_rss_mb']):.2f}")
    print(f"p1_wins: {metrics['p1_wins']}")
    print(f"p2_wins: {metrics['p2_wins']}")
    print(f"ties: {metrics['ties']}")
    print(f"capped_games: {metrics['capped_games']}")
    print(f"average_decision_rounds: {float(metrics['average_decision_rounds']):.2f}")
    print(f"average_simulator_turns: {float(metrics['average_simulator_turns']):.2f}")


def print_benchmark_report(report: BenchmarkReport) -> None:
    print(f"format: {report.format_id}")
    print(f"games_per_matchup: {report.games_per_matchup}")
    print(f"max_decision_rounds: {report.max_decision_rounds}")
    print(f"total_games: {report.total_games}")
    print(f"elapsed_seconds: {report.elapsed_seconds:.3f}")
    print(f"games_per_second: {report.games_per_second:.3f}")
    print(f"decisions_per_second: {report.decisions_per_second:.3f}")
    if report.peak_rss_mb is not None:
        print(f"peak_rss_mb: {report.peak_rss_mb:.2f}")
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
    _print_policy_decision_diagnostics(report)
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


def _print_policy_decision_diagnostics(report: BenchmarkReport) -> None:
    rows = []
    for result in report.matchups:
        summary = result.metrics.policy_decision_summary or {}
        for policy_id, metrics in summary.items():
            if "root_puct_searches" not in metrics and "root_puct_fallbacks" not in metrics:
                continue
            rows.append((result.label, policy_id, metrics))
    if not rows:
        return

    print("")
    print("root-puct diagnostics:")
    header = (
        f"{'matchup':32} {'policy':32} {'dec':>5} {'search':>6} {'fallback':>8} "
        f"{'gate':>8} {'cand':>7} {'ms/dec':>8} {'value':>8} {'score':>8}"
    )
    print(header)
    print("-" * len(header))
    fallback_reasons = []
    for matchup, policy_id, metrics in rows:
        average_candidate_count = metrics.get("root_puct_average_candidate_count")
        average_elapsed_seconds = metrics.get("root_puct_average_elapsed_seconds")
        average_selected_value = metrics.get("root_puct_average_selected_value")
        average_selected_score = metrics.get("root_puct_average_selected_score")
        value_gate_checks = metrics.get("root_puct_value_gate_checks")
        value_gate_uses = metrics.get("root_puct_value_gate_uses")
        print(
            f"{matchup[:32]:32} "
            f"{policy_id[:32]:32} "
            f"{int(metrics.get('decisions', 0)):5d} "
            f"{int(metrics.get('root_puct_searches', 0)):6d} "
            f"{int(metrics.get('root_puct_fallbacks', 0)):8d} "
            f"{_optional_report_ratio(value_gate_uses, value_gate_checks):>8} "
            f"{_optional_report_float(average_candidate_count):>7} "
            f"{_optional_report_millis(average_elapsed_seconds):>8} "
            f"{_optional_report_float(average_selected_value):>8} "
            f"{_optional_report_float(average_selected_score):>8}"
        )
        reasons = metrics.get("root_puct_fallback_reasons")
        if isinstance(reasons, dict) and reasons:
            formatted = ", ".join(f"{reason}={count}" for reason, count in reasons.items())
            fallback_reasons.append(f"{policy_id}: {formatted}")
    if fallback_reasons:
        print("fallback_reasons:")
        for reason in fallback_reasons:
            print(f"  {reason}")


def _optional_report_float(value: object) -> str:
    if value is None:
        return "-"
    try:
        return f"{float(value):.3f}"
    except (TypeError, ValueError):
        return "-"


def _optional_report_ratio(numerator: object, denominator: object) -> str:
    if numerator is None or denominator is None:
        return "-"
    try:
        return f"{int(numerator)}/{int(denominator)}"
    except (TypeError, ValueError):
        return "-"


def _optional_report_millis(value: object) -> str:
    if value is None:
        return "-"
    try:
        return f"{float(value) * 1000.0:.2f}"
    except (TypeError, ValueError):
        return "-"


def print_replay_benchmark_report(report: ReplayPrefixBenchmarkReport) -> None:
    print(f"format: {report.format_id}")
    print(f"games: {report.games}")
    print(f"prefixes_per_game: {report.prefixes_per_game}")
    print(f"max_decision_rounds: {report.max_decision_rounds}")
    print(f"source_policy_ids: {dict(report.source_policy_ids)}")
    print(f"source_average_decision_rounds: {report.average_source_decision_rounds:.2f}")
    print(f"total_prefixes: {report.total_prefixes}")
    print(f"replay_elapsed_seconds: {report.total_replay_elapsed_seconds:.3f}")
    print(f"avg_replayed_decision_rounds: {report.average_replayed_decision_rounds:.2f}")
    print(f"replayed_decision_rounds_per_second: {report.replayed_decision_rounds_per_second:.1f}")
    print(f"avg_replay_ms: {report.average_replay_seconds * 1000.0:.2f}")
    print(f"median_replay_ms: {report.median_replay_seconds * 1000.0:.2f}")
    print(f"p95_replay_ms: {report.p95_replay_seconds * 1000.0:.2f}")
    print(f"max_replay_ms: {report.max_replay_seconds * 1000.0:.2f}")
    print("note: replay timing excludes source-game generation and includes env.reset plus prefix replay.")


if __name__ == "__main__":
    raise SystemExit(main())
