"""Command-line utilities for the linear policy baseline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from .collection import BenchmarkMatchup, benchmark_rollouts
from .linear_policy import (
    LinearSoftmaxPolicy,
    LinearTrainingConfig,
    evaluate_linear_policy,
    load_linear_model,
    save_linear_model,
    train_linear_policy,
)
from .local_showdown import LocalShowdownConfig, LocalShowdownEnv
from .policy import RandomLegalPolicy, SimpleLegalPolicy
from .rollout import RolloutConfig
from .rollout_cli import print_benchmark_report


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m pokezero.linear_cli")
    subparsers = parser.add_subparsers(dest="command", required=True)

    train = subparsers.add_parser("train", help="Train a masked linear policy from rollout JSONL.")
    train.add_argument("--data", type=Path, nargs="+", required=True, help="One or more rollout JSONL files.")
    train.add_argument("--validation-data", type=Path, nargs="+", default=None, help="Optional held-out rollout JSONL for validation metrics.")
    train.add_argument("--out", type=Path, required=True, help="Checkpoint output path.")
    train.add_argument("--epochs", type=int, default=1, help="Number of streaming training passes.")
    train.add_argument("--learning-rate", type=float, default=0.05, help="SGD learning rate.")
    train.add_argument("--l2", type=float, default=0.0, help="L2 penalty applied on active features.")
    train.add_argument("--feature-count", type=int, default=131_072, help="Hashed feature bucket count.")
    train.add_argument("--window-size", type=int, default=1, help="Per-player observation history window.")
    train.add_argument("--discount", type=float, default=1.0, help="Terminal return discount per player decision.")
    train.add_argument("--capped-terminal-value", type=float, default=0.0, help="Return assigned to each player in capped games.")
    train.add_argument(
        "--objective",
        choices=("behavior-cloning", "reward-weighted"),
        default="behavior-cloning",
        help="Training objective. reward-weighted reinforces positive-return actions and ignores non-positive returns.",
    )
    train.add_argument("--shuffle-buffer-size", type=int, default=1024, help="Streaming shuffle buffer size; 0 disables shuffling.")
    train.add_argument("--shuffle-seed", type=int, default=1, help="Deterministic shuffle seed.")
    train.add_argument("--max-examples", type=int, default=None, help="Optional max examples per epoch.")
    train.add_argument("--policy-id", default="linear-softmax", help="Policy id stored in the checkpoint.")
    train.set_defaults(func=_train)

    evaluate = subparsers.add_parser("evaluate", help="Evaluate a linear checkpoint against rollout JSONL.")
    evaluate.add_argument("--data", type=Path, nargs="+", required=True, help="One or more rollout JSONL files.")
    evaluate.add_argument("--checkpoint", type=Path, required=True, help="Linear checkpoint path.")
    evaluate.add_argument("--discount", type=float, default=1.0, help="Terminal return discount per player decision.")
    evaluate.add_argument("--capped-terminal-value", type=float, default=0.0, help="Return assigned to each player in capped games.")
    evaluate.add_argument("--max-examples", type=int, default=None, help="Optional max examples to evaluate.")
    evaluate.add_argument("--json", action="store_true", help="Print metrics as JSON.")
    evaluate.set_defaults(func=_evaluate)

    benchmark = subparsers.add_parser("benchmark", help="Benchmark a linear checkpoint against fixed baselines.")
    benchmark.add_argument("--checkpoint", type=Path, required=True, help="Linear checkpoint path.")
    benchmark.add_argument("--games", type=int, default=20, help="Number of games per matchup.")
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


def _train(args: argparse.Namespace) -> int:
    config = LinearTrainingConfig(
        feature_count=args.feature_count,
        window_size=args.window_size,
        discount=args.discount,
        capped_terminal_value=args.capped_terminal_value,
        objective=args.objective,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        l2=args.l2,
        shuffle_buffer_size=args.shuffle_buffer_size,
        shuffle_seed=args.shuffle_seed,
        max_examples=args.max_examples,
        policy_id=args.policy_id,
    )
    result = train_linear_policy(args.data, config=config, validation_paths=args.validation_data)
    save_linear_model(args.out, result.model)
    for metrics in result.epochs:
        print(
            f"epoch={metrics.epoch} examples={metrics.examples} "
            f"loss={metrics.loss:.6f} accuracy={metrics.accuracy:.4f} "
            f"elapsed_seconds={metrics.elapsed_seconds:.3f}"
        )
    if result.validation_metrics is not None:
        metrics = result.validation_metrics
        print(
            f"validation examples={metrics.examples} "
            f"loss={metrics.loss:.6f} accuracy={metrics.accuracy:.4f} "
            f"elapsed_seconds={metrics.elapsed_seconds:.3f}"
        )
    print(f"checkpoint: {args.out}")
    return 0


def _evaluate(args: argparse.Namespace) -> int:
    model = load_linear_model(args.checkpoint)
    metrics = evaluate_linear_policy(
        args.data,
        model,
        discount=args.discount,
        capped_terminal_value=args.capped_terminal_value,
        max_examples=args.max_examples,
    )
    if args.json:
        print(json.dumps(metrics.to_dict(), indent=2, sort_keys=True))
    else:
        print(f"examples: {metrics.examples}")
        print(f"loss: {metrics.loss:.6f}")
        print(f"accuracy: {metrics.accuracy:.4f}")
        print(f"elapsed_seconds: {metrics.elapsed_seconds:.3f}")
    return 0


def _benchmark(args: argparse.Namespace) -> int:
    model = load_linear_model(args.checkpoint)
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
        matchups=(
            BenchmarkMatchup(f"{model.policy_id} vs random-legal", _policy_from_model(model), RandomLegalPolicy()),
            BenchmarkMatchup(f"random-legal vs {model.policy_id}", RandomLegalPolicy(), _policy_from_model(model)),
            BenchmarkMatchup(f"{model.policy_id} vs simple-legal", _policy_from_model(model), SimpleLegalPolicy()),
            BenchmarkMatchup(f"simple-legal vs {model.policy_id}", SimpleLegalPolicy(), _policy_from_model(model)),
        ),
    )
    if args.json:
        print(json.dumps(report.to_dict(), indent=2, sort_keys=True))
    else:
        print_benchmark_report(report)
    return 0


def _policy_from_model(model):
    return LinearSoftmaxPolicy(model=model)


if __name__ == "__main__":
    raise SystemExit(main())
