"""Command-line bootstrap workflows."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from .bootstrap import (
    DEFAULT_BENCHMARK_GAMES,
    DEFAULT_PREFLIGHT_GAMES,
    DEFAULT_PREFLIGHT_SEED_START,
    run_teacher_bootstrap,
)
from .collection import policy_spec_with_showdown_root
from .linear_policy import LinearTrainingConfig
from .local_showdown import LocalShowdownConfig, LocalShowdownEnv
from .rollout import RolloutConfig


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m pokezero.bootstrap_cli")
    subparsers = parser.add_subparsers(dest="command", required=True)

    teacher = subparsers.add_parser("teacher", help="Collect scripted-teacher data and train a bootstrap checkpoint.")
    teacher.add_argument("--run-dir", type=Path, required=True, help="Output directory for data, checkpoint, and manifest.")
    teacher.add_argument("--train-games", type=int, required=True, help="Teacher-current games for training data.")
    teacher.add_argument("--validation-games", type=int, required=True, help="Teacher-current games for held-out validation data.")
    teacher.add_argument("--workers", type=int, default=1, help="Parallel rollout collection workers.")
    teacher.add_argument("--showdown-root", type=Path, default=None, help="Built Pokemon Showdown checkout root.")
    teacher.add_argument("--format", dest="format_id", default="gen3randombattle", help="Showdown format id.")
    teacher.add_argument("--seed-start", type=int, default=1, help="First deterministic training-data seed.")
    teacher.add_argument("--validation-seed-start", type=int, default=1_000_000, help="First deterministic validation-data seed.")
    teacher.add_argument("--benchmark-games", type=int, default=DEFAULT_BENCHMARK_GAMES, help="Benchmark games per matchup after training. Set 0 to disable.")
    teacher.add_argument("--benchmark-seed-start", type=int, default=2_000_000, help="First deterministic benchmark seed.")
    teacher.add_argument("--preflight-games", type=int, default=DEFAULT_PREFLIGHT_GAMES, help="Strict teacher warmup games before the full run. Set 0 to disable.")
    teacher.add_argument("--preflight-seed-start", type=int, default=DEFAULT_PREFLIGHT_SEED_START, help="First deterministic preflight seed.")
    teacher.add_argument("--max-decision-rounds", type=int, default=250, help="Rollout decision-round cap.")
    teacher.add_argument("--node-binary", default="node", help="Node executable used for the BattleStream bridge.")
    teacher.add_argument("--teacher-policy", default="scripted-teacher", help="Teacher policy spec.")
    teacher.add_argument(
        "--opponent-policy",
        action="append",
        default=None,
        help="Opponent policy spec for teacher collection. May be repeated. Defaults to teacher mirror, simple-legal, and random-legal.",
    )
    teacher.add_argument("--epochs", type=int, default=1, help="Training epochs for the bootstrap checkpoint.")
    teacher.add_argument("--learning-rate", type=float, default=0.05, help="SGD learning rate.")
    teacher.add_argument(
        "--opponent-action-loss-weight",
        type=float,
        default=0.0,
        help=(
            "Auxiliary opponent-action prediction loss weight. The linear policy's "
            "action weights are independent of this head, so it does not affect play; "
            "it is opt-in scaffolding for future shared-representation models. "
            "Off by default."
        ),
    )
    teacher.add_argument("--l2", type=float, default=0.0, help="L2 penalty applied on active features.")
    teacher.add_argument("--feature-count", type=int, default=131_072, help="Hashed feature bucket count.")
    teacher.add_argument("--window-size", type=int, default=1, help="Per-player observation history window.")
    teacher.add_argument("--shuffle-buffer-size", type=int, default=1024, help="Streaming shuffle buffer size; 0 disables shuffling.")
    teacher.add_argument("--shuffle-seed", type=int, default=1, help="Deterministic shuffle seed.")
    teacher.add_argument("--max-examples", type=int, default=None, help="Optional max examples per epoch.")
    teacher.add_argument("--policy-id", default="linear-bootstrap", help="Policy id stored in the bootstrap checkpoint.")
    teacher.add_argument("--json", action="store_true", help="Print the bootstrap manifest as JSON.")
    teacher.set_defaults(func=_teacher)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


def _teacher(args: argparse.Namespace) -> int:
    env_config = LocalShowdownConfig(
        showdown_root=args.showdown_root,
        node_binary=args.node_binary,
    )
    policy_showdown_root = env_config.resolved_showdown_root()
    teacher_policy = policy_spec_with_showdown_root(args.teacher_policy, policy_showdown_root)
    opponent_policies = (
        tuple(policy_spec_with_showdown_root(spec, policy_showdown_root) for spec in args.opponent_policy)
        if args.opponent_policy is not None
        else None
    )
    result = run_teacher_bootstrap(
        run_dir=args.run_dir,
        env_factory=lambda: LocalShowdownEnv(env_config),
        rollout_config=RolloutConfig(
            max_decision_rounds=args.max_decision_rounds,
            format_id=args.format_id,
        ),
        training_config=LinearTrainingConfig(
            feature_count=args.feature_count,
            window_size=args.window_size,
            objective="behavior-cloning",
            epochs=args.epochs,
            learning_rate=args.learning_rate,
            opponent_action_loss_weight=args.opponent_action_loss_weight,
            l2=args.l2,
            shuffle_buffer_size=args.shuffle_buffer_size,
            shuffle_seed=args.shuffle_seed,
            max_examples=args.max_examples,
            policy_id=args.policy_id,
        ),
        train_games=args.train_games,
        validation_games=args.validation_games,
        teacher_policy_spec=teacher_policy,
        opponent_policy_specs=opponent_policies,
        seed_start=args.seed_start,
        validation_seed_start=args.validation_seed_start,
        benchmark_games=args.benchmark_games,
        benchmark_seed_start=args.benchmark_seed_start,
        preflight_games=args.preflight_games,
        preflight_seed_start=args.preflight_seed_start,
        worker_count=args.workers,
    )
    if args.json:
        print(json.dumps(result.to_dict(), indent=2, sort_keys=True))
    else:
        _print_teacher_summary(result)
    return 0


def _print_teacher_summary(result) -> None:
    final_epoch = result.training.final_metrics
    print(f"run_dir: {result.run_dir}")
    print(f"train_rollouts: {result.train_rollout_path}")
    print(f"validation_rollouts: {result.validation_rollout_path}")
    print(f"checkpoint: {result.checkpoint_path}")
    print(f"train_games: {result.train_metrics.games}")
    print(f"validation_games: {result.validation_metrics.games}")
    if result.preflight_metrics is not None:
        print(f"preflight_games: {result.preflight_metrics.games}")
    print(
        f"training examples={final_epoch.examples} "
        f"loss={final_epoch.loss:.6f} "
        f"accuracy={final_epoch.accuracy:.4f}"
    )
    if getattr(final_epoch, "opponent_examples", 0):
        print(
            f"training opponent_examples={final_epoch.opponent_examples} "
            f"opponent_loss={final_epoch.opponent_loss:.6f} "
            f"opponent_accuracy={final_epoch.opponent_accuracy:.4f}"
        )
    if result.training.validation_metrics is not None:
        metrics = result.training.validation_metrics
        print(
            f"validation examples={metrics.examples} "
            f"loss={metrics.loss:.6f} "
            f"accuracy={metrics.accuracy:.4f}"
        )
        if getattr(metrics, "opponent_examples", 0):
            print(
                f"validation opponent_examples={metrics.opponent_examples} "
                f"opponent_loss={metrics.opponent_loss:.6f} "
                f"opponent_accuracy={metrics.opponent_accuracy:.4f}"
            )
    if result.benchmark is not None:
        print(f"benchmark_total_games: {result.benchmark.total_games}")
        for row in result.benchmark.head_to_head_results:
            print(
                f"benchmark {row.label}: "
                f"{row.first_policy_id}_wr={row.first_policy_win_rate:.3f} "
                f"{row.second_policy_id}_wr={row.second_policy_win_rate:.3f} "
                f"capped={row.capped_games}"
            )
    summary = result.teacher_decision_summary
    if summary.get("unknown_move_decisions") or summary.get("fallback_decisions"):
        print(
            "teacher_degradation: "
            f"unknown_moves={summary.get('unknown_move_decisions', 0)} "
            f"fallbacks={summary.get('fallback_decisions', 0)}"
        )
    print(f"manifest: {result.manifest_path}")


if __name__ == "__main__":
    raise SystemExit(main())
