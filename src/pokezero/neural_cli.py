"""Command-line utilities for optional neural policy experiments."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from .cli_audit import (
    add_post_iteration_audit_arguments,
    post_iteration_audit_config_from_args,
    validate_post_iteration_audit_evaluation_games,
)
from .collection import BenchmarkMatchup, benchmark_rollouts, policy_spec_with_showdown_root
from .local_showdown import LocalShowdownConfig, LocalShowdownEnv
from .neural_policy import (
    TransformerPolicyConfig,
    TransformerTrainingConfig,
    load_transformer_policy,
    save_transformer_checkpoint,
    torch_available,
    train_transformer_policy,
)
from .neural_selfplay import NeuralSelfPlayPromotionConfig, run_neural_selfplay_iterations
from .policy import RandomLegalPolicy, SimpleLegalPolicy
from .run_audit import RunAuditFailure
from .rollout import RolloutConfig
from .rollout_cli import print_benchmark_report
from .eval_cli import _add_gate_arguments, _gate_config_from_args


MIN_NEURAL_POST_ITERATION_BENCHMARK_MATCHUPS = 4


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m pokezero.neural_cli")
    subparsers = parser.add_subparsers(dest="command", required=True)

    describe = subparsers.add_parser("describe", help="Print the default neural policy config and torch availability.")
    describe.add_argument("--json", action="store_true", help="Print machine-readable JSON.")
    describe.set_defaults(func=_describe)

    train = subparsers.add_parser("train", help="Train an entity-token transformer policy from rollout JSONL.")
    train.add_argument("--data", type=Path, nargs="+", required=True, help="One or more rollout JSONL files.")
    train.add_argument("--out", type=Path, required=True, help="Checkpoint output path.")
    train.add_argument("--epochs", type=int, default=1, help="Number of training epochs.")
    train.add_argument("--batch-size", type=int, default=64, help="Training batch size.")
    train.add_argument("--learning-rate", type=float, default=3e-4, help="AdamW learning rate.")
    train.add_argument("--weight-decay", type=float, default=0.0, help="AdamW weight decay.")
    train.add_argument("--window-size", type=int, default=4, help="Per-player observation history window.")
    train.add_argument("--discount", type=float, default=1.0, help="Terminal return discount per player decision.")
    train.add_argument("--capped-terminal-value", type=float, default=0.0, help="Return assigned to each player in capped games.")
    train.add_argument("--value-loss-weight", type=float, default=0.25, help="Scalar value-head MSE loss weight.")
    train.add_argument("--opponent-action-loss-weight", type=float, default=0.1, help="Opponent-action auxiliary loss weight.")
    train.add_argument("--max-batches", type=int, default=None, help="Optional max batches per epoch for smoke runs.")
    train.add_argument("--device", default=None, help="Torch device, e.g. cpu, cuda, or mps. Defaults to cuda when available, else cpu.")
    train.add_argument("--embedding-dim", type=int, default=128, help="Transformer embedding width.")
    train.add_argument("--layers", type=int, default=2, help="Transformer encoder layer count.")
    train.add_argument("--attention-heads", type=int, default=4, help="Transformer attention head count.")
    train.add_argument("--feedforward-dim", type=int, default=256, help="Transformer feedforward width.")
    train.add_argument("--dropout", type=float, default=0.1, help="Transformer dropout.")
    train.add_argument("--policy-id", default="entity-transformer", help="Policy id stored in the checkpoint config.")
    train.set_defaults(func=_train)

    benchmark = subparsers.add_parser("benchmark", help="Benchmark a neural checkpoint against fixed baselines.")
    benchmark.add_argument("--checkpoint", type=Path, required=True, help="Neural checkpoint path.")
    benchmark.add_argument("--games", type=int, default=20, help="Number of games per matchup.")
    benchmark.add_argument("--showdown-root", type=Path, default=None, help="Built Pokemon Showdown checkout root.")
    benchmark.add_argument("--format", dest="format_id", default="gen3randombattle", help="Showdown format id.")
    benchmark.add_argument("--seed-start", type=int, default=1, help="First deterministic rollout seed for every matchup.")
    benchmark.add_argument("--max-decision-rounds", type=int, default=250, help="Rollout decision-round cap.")
    benchmark.add_argument("--node-binary", default="node", help="Node executable used for the BattleStream bridge.")
    benchmark.add_argument("--device", default=None, help="Torch device, e.g. cpu, cuda, or mps. Defaults to checkpoint load behavior.")
    benchmark.add_argument("--sample", action="store_true", help="Sample from the checkpoint policy distribution instead of greedy selection.")
    benchmark.add_argument("--epsilon", type=float, default=0.0, help="Random legal exploration rate during benchmark.")
    benchmark.add_argument("--temperature", type=float, default=1.0, help="Softmax sampling temperature.")
    benchmark.add_argument("--json", action="store_true", help="Print benchmark results as JSON.")
    benchmark.set_defaults(func=_benchmark)

    iterate = subparsers.add_parser("iterate", help="Run neural-policy self-play training iterations.")
    iterate.add_argument("--run-dir", type=Path, required=True, help="Directory for rollouts, checkpoints, and manifests.")
    iterate.add_argument("--iterations", type=int, required=True, help="Number of collect/train/evaluate iterations.")
    iterate.add_argument("--resume", action="store_true", help="Continue an existing neural self-play run directory from its latest manifest.")
    iterate.add_argument("--games-per-iteration", type=int, required=True, help="Rollout games collected before each train step.")
    iterate.add_argument("--workers", type=int, default=1, help="Parallel rollout collection workers per iteration.")
    iterate.add_argument("--showdown-root", type=Path, default=None, help="Built Pokemon Showdown checkout root.")
    iterate.add_argument("--format", dest="format_id", default="gen3randombattle", help="Showdown format id.")
    iterate.add_argument("--seed-start", type=int, default=1, help="First deterministic self-play seed.")
    iterate.add_argument("--max-decision-rounds", type=int, default=250, help="Rollout decision-round cap.")
    iterate.add_argument("--node-binary", default="node", help="Node executable used for the BattleStream bridge.")
    iterate.add_argument("--initial-policy", required=True, help="Policy spec used before the first checkpoint exists.")
    iterate.add_argument(
        "--opponent-policy",
        action="append",
        default=None,
        help="Fixed opponent policy spec. May be repeated. Defaults to random-legal and simple-legal.",
    )
    iterate.add_argument("--max-historical-opponents", type=int, default=3, help="Number of older checkpoints kept in the opponent pool.")
    iterate.add_argument(
        "--promotion-registry",
        type=Path,
        default=None,
        help="Optional promotion registry. When set, historical opponents come from promoted checkpoints instead of raw accepted neural checkpoints.",
    )
    iterate.add_argument(
        "--require-promoted-opponent-pool-size",
        type=int,
        default=None,
        help=(
            "Fail before rollout collection unless at least this many promoted historical opponents "
            "are selectable from the promotion registry after current-policy exclusion. "
            "Cannot exceed --max-historical-opponents."
        ),
    )
    iterate.add_argument(
        "--auto-promote",
        action="store_true",
        help="After each iteration, evaluate the promotion gate and record passing checkpoints in --promotion-registry.",
    )
    iterate.add_argument(
        "--promotion-artifact-dir",
        type=Path,
        default=None,
        help="Optional artifact directory for auto-promoted neural checkpoint copies.",
    )
    iterate.add_argument(
        "--promotion-label-prefix",
        default="neural-selfplay",
        help="Label prefix for auto-promotion entries. Use an empty string to omit labels.",
    )
    iterate.add_argument("--promotion-notes", default=None, help="Optional notes stored on each auto-promotion entry.")
    iterate.add_argument(
        "--allow-duplicate-promotion",
        action="store_true",
        help="Allow auto-promotion to record a checkpoint already present in the registry.",
    )
    _add_gate_arguments(iterate)
    iterate.add_argument(
        "--evaluation-games",
        type=int,
        default=0,
        help="Benchmark games per matchup after each train step. Required to be positive for multi-iteration runs.",
    )
    iterate.add_argument("--evaluation-seed-start", type=int, default=1_000_000, help="First deterministic benchmark seed.")
    iterate.add_argument("--epochs", type=int, default=1, help="Training epochs per iteration.")
    iterate.add_argument("--batch-size", type=int, default=64, help="Training batch size.")
    iterate.add_argument("--learning-rate", type=float, default=3e-4, help="AdamW learning rate.")
    iterate.add_argument("--weight-decay", type=float, default=0.0, help="AdamW weight decay.")
    iterate.add_argument("--window-size", type=int, default=4, help="Per-player observation history window.")
    iterate.add_argument("--discount", type=float, default=1.0, help="Terminal return discount per player decision.")
    iterate.add_argument("--capped-terminal-value", type=float, default=-0.25, help="Return assigned to each player in capped games.")
    iterate.add_argument("--value-loss-weight", type=float, default=0.25, help="Scalar value-head MSE loss weight.")
    iterate.add_argument("--opponent-action-loss-weight", type=float, default=0.1, help="Opponent-action auxiliary loss weight.")
    iterate.add_argument("--max-batches", type=int, default=None, help="Optional max batches per epoch for smoke runs.")
    iterate.add_argument("--device", default=None, help="Torch device, e.g. cpu, cuda, or mps. Defaults to cuda when available, else cpu.")
    iterate.add_argument("--embedding-dim", type=int, default=128, help="Transformer embedding width.")
    iterate.add_argument("--layers", type=int, default=2, help="Transformer encoder layer count.")
    iterate.add_argument("--attention-heads", type=int, default=4, help="Transformer attention head count.")
    iterate.add_argument("--feedforward-dim", type=int, default=256, help="Transformer feedforward width.")
    iterate.add_argument("--dropout", type=float, default=0.1, help="Transformer dropout.")
    iterate.add_argument("--policy-id", default="entity-transformer-selfplay", help="Base policy id for generated checkpoints.")
    add_post_iteration_audit_arguments(iterate)
    iterate.add_argument("--json", action="store_true", help="Print the run manifest as JSON.")
    iterate.set_defaults(func=_iterate)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except RunAuditFailure as exc:
        _print_run_audit_failure(exc)
        return 3
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


def _describe(args: argparse.Namespace) -> int:
    config = TransformerPolicyConfig()
    payload = {
        "torch_available": torch_available(),
        "model_config": config.to_dict(),
    }
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print(f"torch_available: {payload['torch_available']}")
        print(f"policy_id: {config.policy_id}")
        print(f"window_size: {config.window_size}")
        print(f"token_count: {config.token_count}")
        print(f"categorical_feature_count: {config.categorical_feature_count}")
        print(f"numeric_feature_count: {config.numeric_feature_count}")
        print(f"embedding_dim: {config.embedding_dim}")
        print(f"layers: {config.transformer_layers}")
        print(f"attention_heads: {config.attention_heads}")
    return 0


def _train(args: argparse.Namespace) -> int:
    training_config = TransformerTrainingConfig(
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        window_size=args.window_size,
        discount=args.discount,
        capped_terminal_value=args.capped_terminal_value,
        value_loss_weight=args.value_loss_weight,
        opponent_action_loss_weight=args.opponent_action_loss_weight,
        max_batches=args.max_batches,
        device=args.device,
    )
    model_config = TransformerPolicyConfig(
        policy_id=args.policy_id,
        window_size=args.window_size,
        embedding_dim=args.embedding_dim,
        transformer_layers=args.layers,
        attention_heads=args.attention_heads,
        feedforward_dim=args.feedforward_dim,
        dropout=args.dropout,
    )
    model, result = train_transformer_policy(args.data, model_config=model_config, training_config=training_config)
    save_transformer_checkpoint(args.out, model, result=result)
    for metrics in result.epochs:
        line = (
            f"epoch={metrics.epoch} examples={metrics.examples} "
            f"loss={metrics.loss:.6f} policy_loss={metrics.policy_loss:.6f} "
            f"policy_accuracy={metrics.policy_accuracy:.4f}"
        )
        if metrics.value_loss is not None:
            line += f" value_loss={metrics.value_loss:.6f}"
        if metrics.opponent_loss is not None:
            line += (
                f" opponent_loss={metrics.opponent_loss:.6f} "
                f"opponent_accuracy={metrics.opponent_accuracy:.4f}"
            )
        print(line)
    print(f"checkpoint: {args.out}")
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
    deterministic = not bool(args.sample)
    checkpoint_policy = _policy_from_checkpoint(
        args.checkpoint,
        deterministic=deterministic,
        exploration_epsilon=args.epsilon,
        sampling_temperature=args.temperature,
        device=args.device,
    )
    policy_id = checkpoint_policy.policy_id
    report = benchmark_rollouts(
        games=args.games,
        env_factory=lambda: LocalShowdownEnv(env_config),
        rollout_config=rollout_config,
        seed_start=args.seed_start,
        matchups=(
            BenchmarkMatchup(
                f"{policy_id} vs random-legal",
                checkpoint_policy,
                RandomLegalPolicy(),
            ),
            BenchmarkMatchup(
                f"random-legal vs {policy_id}",
                RandomLegalPolicy(),
                checkpoint_policy,
            ),
            BenchmarkMatchup(
                f"{policy_id} vs simple-legal",
                checkpoint_policy,
                SimpleLegalPolicy(),
            ),
            BenchmarkMatchup(
                f"simple-legal vs {policy_id}",
                SimpleLegalPolicy(),
                checkpoint_policy,
            ),
        ),
    )
    if args.json:
        print(json.dumps(report.to_dict(), indent=2, sort_keys=True))
    else:
        print_benchmark_report(report)
    return 0


def _iterate(args: argparse.Namespace) -> int:
    if args.auto_promote and args.promotion_registry is None:
        raise ValueError("--auto-promote requires --promotion-registry.")
    if args.auto_promote and args.evaluation_games <= 0 and args.require_benchmark is not False:
        raise ValueError("--auto-promote requires --evaluation-games > 0 unless --allow-missing-benchmark is set.")
    post_iteration_audit_config = post_iteration_audit_config_from_args(args)
    validate_post_iteration_audit_evaluation_games(
        post_iteration_audit_config,
        evaluation_games=args.evaluation_games,
        minimum_benchmark_matchups=MIN_NEURAL_POST_ITERATION_BENCHMARK_MATCHUPS,
    )
    env_config = LocalShowdownConfig(
        showdown_root=args.showdown_root,
        node_binary=args.node_binary,
    )
    rollout_config = RolloutConfig(
        max_decision_rounds=args.max_decision_rounds,
        format_id=args.format_id,
    )
    training_config = TransformerTrainingConfig(
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        window_size=args.window_size,
        discount=args.discount,
        capped_terminal_value=args.capped_terminal_value,
        value_loss_weight=args.value_loss_weight,
        opponent_action_loss_weight=args.opponent_action_loss_weight,
        max_batches=args.max_batches,
        device=args.device,
    )
    model_config = TransformerPolicyConfig(
        policy_id=args.policy_id,
        window_size=args.window_size,
        embedding_dim=args.embedding_dim,
        transformer_layers=args.layers,
        attention_heads=args.attention_heads,
        feedforward_dim=args.feedforward_dim,
        dropout=args.dropout,
    )
    initial_policy = policy_spec_with_showdown_root(args.initial_policy, args.showdown_root)
    opponent_policies = tuple(
        policy_spec_with_showdown_root(spec, args.showdown_root)
        for spec in (args.opponent_policy or ("random-legal", "simple-legal"))
    )
    auto_promotion_config = _auto_promotion_config_from_args(args)
    result = run_neural_selfplay_iterations(
        run_dir=args.run_dir,
        iterations=args.iterations,
        games_per_iteration=args.games_per_iteration,
        env_factory=lambda: LocalShowdownEnv(env_config),
        rollout_config=rollout_config,
        model_config=model_config,
        training_config=training_config,
        seed_start=args.seed_start,
        initial_policy_spec=initial_policy,
        fixed_opponent_policy_specs=opponent_policies,
        max_historical_opponents=args.max_historical_opponents,
        evaluation_games=args.evaluation_games,
        evaluation_seed_start=args.evaluation_seed_start,
        worker_count=args.workers,
        promotion_registry_path=args.promotion_registry,
        required_promoted_opponent_pool_size=args.require_promoted_opponent_pool_size,
        auto_promotion_config=auto_promotion_config,
        post_iteration_audit_config=post_iteration_audit_config,
        resume=args.resume,
    )
    if args.json:
        print(json.dumps(result.to_dict(), indent=2, sort_keys=True))
    else:
        _print_iterate_summary(result)
    return 0


def _print_run_audit_failure(exc: RunAuditFailure) -> None:
    failed = [check.name for check in exc.result.checks if not check.passed]
    print(f"audit_failed: {exc.result.manifest_path}", file=sys.stderr)
    print(f"failed_checks: {', '.join(failed) if failed else 'unknown'}", file=sys.stderr)


def _auto_promotion_config_from_args(args: argparse.Namespace) -> NeuralSelfPlayPromotionConfig | None:
    if not args.auto_promote:
        return None
    gate_args = argparse.Namespace(**vars(args))
    gate_args.registry = None
    label_prefix = args.promotion_label_prefix if args.promotion_label_prefix else None
    return NeuralSelfPlayPromotionConfig(
        registry_path=args.promotion_registry,
        gate_config=_gate_config_from_args(gate_args),
        artifact_dir=args.promotion_artifact_dir,
        label_prefix=label_prefix,
        notes=args.promotion_notes,
        allow_duplicate=args.allow_duplicate_promotion,
    )


def _print_iterate_summary(result) -> None:
    print(f"run_dir: {result.run_dir}")
    for iteration in result.iterations:
        final_epoch = iteration.training.final_metrics
        print(
            f"iteration={iteration.iteration} games={iteration.metrics.games} "
            f"checkpoint={iteration.checkpoint_path} "
            f"loss={final_epoch.loss:.6f} "
            f"policy_accuracy={final_epoch.policy_accuracy:.4f} "
            f"promotion={_promotion_status(getattr(iteration, 'promotion', None))}"
        )
        if iteration.benchmark is not None:
            print(f"benchmark_total_games={iteration.benchmark.total_games}")
    if result.latest_checkpoint_path is not None:
        print(f"latest_checkpoint: {result.latest_checkpoint_path}")
    print(f"manifest: {result.run_dir / 'manifest.json'}")


def _promotion_status(promotion) -> str:
    if promotion is None:
        return "-"
    return "recorded" if promotion.recorded else "failed"


def _policy_from_checkpoint(
    checkpoint: Path,
    *,
    deterministic: bool,
    exploration_epsilon: float,
    sampling_temperature: float,
    device: str | None,
):
    return load_transformer_policy(
        checkpoint,
        deterministic=deterministic,
        exploration_epsilon=exploration_epsilon,
        sampling_temperature=sampling_temperature,
        device=device,
    )


if __name__ == "__main__":
    raise SystemExit(main())
