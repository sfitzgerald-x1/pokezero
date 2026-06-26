"""Command-line utilities for optional neural policy experiments."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Mapping

from .cli_audit import (
    add_post_iteration_audit_arguments,
    post_iteration_audit_config_from_args,
    validate_post_iteration_audit_evaluation_games,
)
from .collection import BenchmarkMatchup, benchmark_rollouts, policy_spec_with_showdown_root, reject_eval_only_specs
from .local_showdown import LocalShowdownConfig, LocalShowdownEnv
from .neural_policy import (
    DEFAULT_CATEGORY_OOV_BUCKETS,
    TransformerPolicyConfig,
    TransformerTrainingConfig,
    collect_categorical_ids,
    load_transformer_checkpoint,
    load_transformer_policy,
    require_torch,
    save_transformer_checkpoint,
    torch_available,
    train_transformer_policy,
)
from .value_calibration import ValueCalibrationReport, evaluate_value_calibration
from .neural_selfplay import (
    NeuralSelfPlayPromotionConfig,
    _mapping,
    _sequence,
    load_neural_selfplay_run_manifest,
    run_neural_selfplay_iterations,
)
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
    train.add_argument(
        "--switch-action-loss-weight",
        type=float,
        default=1.0,
        help="Multiplier for switch-action policy CE examples under behavior-cloning / reward-weighted objectives.",
    )
    train.add_argument(
        "--action-family-loss-weight",
        type=float,
        default=0.0,
        help="Auxiliary move-vs-switch classification loss weight derived from legal action logits.",
    )
    train.add_argument(
        "--switch-target-loss-weight",
        type=float,
        default=0.0,
        help="Auxiliary conditional switch-target classification loss weight over switch-labeled examples.",
    )
    train.add_argument(
        "--objective",
        choices=("behavior-cloning", "reward-weighted", "ppo"),
        default="behavior-cloning",
        help=(
            "Training objective: supervised behavior cloning (default), reward-weighted "
            "behavior cloning, or PPO self-play RL."
        ),
    )
    train.add_argument("--clip-epsilon", type=float, default=0.2, help="PPO clipped-surrogate epsilon (objective=ppo).")
    train.add_argument("--entropy-coef", type=float, default=0.0, help="PPO entropy bonus coefficient (objective=ppo).")
    train.add_argument("--no-normalize-advantage", action="store_true", help="Disable PPO advantage normalization (objective=ppo).")
    train.add_argument("--max-batches", type=int, default=None, help="Optional max batches per epoch for smoke runs.")
    train.add_argument("--device", default=None, help="Torch device, e.g. cpu, cuda, or mps. Defaults to cuda when available, else cpu.")
    train.add_argument("--embedding-dim", type=int, default=128, help="Transformer embedding width.")
    train.add_argument("--layers", type=int, default=2, help="Transformer encoder layer count.")
    train.add_argument("--attention-heads", type=int, default=4, help="Transformer attention head count.")
    train.add_argument("--feedforward-dim", type=int, default=256, help="Transformer feedforward width.")
    train.add_argument("--dropout", type=float, default=0.1, help="Transformer dropout.")
    train.add_argument("--policy-id", default="entity-transformer", help="Policy id stored in the checkpoint config.")
    train.add_argument(
        "--category-oov-buckets",
        type=int,
        default=DEFAULT_CATEGORY_OOV_BUCKETS,
        help="Reserved out-of-vocabulary rows in the compact category embedding.",
    )
    train.add_argument(
        "--showdown-root",
        type=Path,
        default=None,
        help="Built Pokemon Showdown checkout root (required: the category vocabulary is the closed Gen 3 randbat universe).",
    )
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

    value_calibration = subparsers.add_parser(
        "value-calibration",
        help="Evaluate a neural checkpoint value head against rollout return targets.",
    )
    value_calibration.add_argument("--checkpoint", type=Path, required=True, help="Neural checkpoint path.")
    value_calibration.add_argument("--data", type=Path, nargs="+", required=True, help="One or more rollout JSONL files.")
    value_calibration.add_argument("--batch-size", type=int, default=128, help="Evaluation batch size.")
    value_calibration.add_argument("--bins", type=int, default=10, help="Number of prediction bins across [-1, 1].")
    value_calibration.add_argument("--device", default=None, help="Torch device, e.g. cpu, cuda, or mps.")
    value_calibration.add_argument("--json", action="store_true", help="Print calibration results as JSON.")
    value_calibration.set_defaults(func=_value_calibration)

    iterate = subparsers.add_parser("iterate", help="Run neural-policy self-play training iterations.")
    iterate.add_argument("--run-dir", type=Path, required=True, help="Directory for rollouts, checkpoints, and manifests.")
    iterate.add_argument("--iterations", type=int, required=True, help="Number of collect/train/evaluate iterations.")
    iterate.add_argument("--resume", action="store_true", help="Continue an existing neural self-play run directory from its latest manifest.")
    iterate.add_argument("--games-per-iteration", type=int, required=True, help="Rollout games collected before each train step.")
    iterate.add_argument("--workers", type=int, default=16, help="Parallel rollout collection workers per iteration (capped at the game count).")
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
    iterate.add_argument(
        "--benchmark-reference-policy",
        action="append",
        default=None,
        help=(
            "Eval-only policy spec (e.g. max-damage) benchmarked against the candidate each "
            "iteration. May be repeated. Never used for rollout collection or training opponents."
        ),
    )
    iterate.add_argument(
        "--mirror-match",
        action="store_true",
        help=(
            "Add the current policy to the collection opponent pool so it plays copies of "
            "itself (current-vs-current self-play) from iteration 1, rather than self-play only "
            "starting once a checkpoint is promoted into the history pool."
        ),
    )
    iterate.add_argument(
        "--collection-temperature",
        type=float,
        default=1.0,
        help=(
            "Softmax sampling temperature for the self-play collector (>1 explores more). Applies "
            "only to rollout collection; benchmark/advancement use the deterministic policy. "
            "Default 1.0 (unchanged)."
        ),
    )
    iterate.add_argument(
        "--tensorboard-logdir",
        type=Path,
        default=None,
        help="Write per-iteration TensorBoard scalars (loss, accuracy, win rate vs each benchmarked opponent, advancement) to this directory. Requires the tensorboard package (in the neural extra).",
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
    iterate.add_argument(
        "--switch-action-loss-weight",
        type=float,
        default=1.0,
        help="Multiplier for switch-action policy CE examples under behavior-cloning / reward-weighted objectives.",
    )
    iterate.add_argument(
        "--action-family-loss-weight",
        type=float,
        default=0.0,
        help="Auxiliary move-vs-switch classification loss weight derived from legal action logits.",
    )
    iterate.add_argument(
        "--switch-target-loss-weight",
        type=float,
        default=0.0,
        help="Auxiliary conditional switch-target classification loss weight over switch-labeled examples.",
    )
    iterate.add_argument(
        "--objective",
        choices=("behavior-cloning", "reward-weighted", "ppo"),
        default="behavior-cloning",
        help=(
            "Training objective: supervised behavior cloning (default), reward-weighted "
            "behavior cloning, or PPO self-play RL."
        ),
    )
    iterate.add_argument("--clip-epsilon", type=float, default=0.2, help="PPO clipped-surrogate epsilon (objective=ppo).")
    iterate.add_argument("--entropy-coef", type=float, default=0.0, help="PPO entropy bonus coefficient (objective=ppo).")
    iterate.add_argument("--no-normalize-advantage", action="store_true", help="Disable PPO advantage normalization (objective=ppo).")
    iterate.add_argument("--max-batches", type=int, default=None, help="Optional max batches per epoch for smoke runs.")
    iterate.add_argument("--device", default=None, help="Torch device, e.g. cpu, cuda, or mps. Defaults to cuda when available, else cpu.")
    iterate.add_argument("--embedding-dim", type=int, default=128, help="Transformer embedding width.")
    iterate.add_argument("--layers", type=int, default=2, help="Transformer encoder layer count.")
    iterate.add_argument("--attention-heads", type=int, default=4, help="Transformer attention head count.")
    iterate.add_argument("--feedforward-dim", type=int, default=256, help="Transformer feedforward width.")
    iterate.add_argument("--dropout", type=float, default=0.1, help="Transformer dropout.")
    iterate.add_argument("--policy-id", default="entity-transformer-selfplay", help="Base policy id for generated checkpoints.")
    iterate.add_argument(
        "--category-oov-buckets",
        type=int,
        default=DEFAULT_CATEGORY_OOV_BUCKETS,
        help="Reserved out-of-vocabulary rows in the compact category embedding.",
    )
    add_post_iteration_audit_arguments(iterate)
    iterate.add_argument("--json", action="store_true", help="Print the run manifest as JSON.")
    iterate.set_defaults(func=_iterate)

    report = subparsers.add_parser("report", help="Print a summary of a neural self-play run manifest.")
    report.add_argument("--run-dir", type=Path, required=True, help="Neural self-play run directory containing manifest.json.")
    report.add_argument("--json", action="store_true", help="Print the raw run manifest as formatted JSON.")
    report.set_defaults(func=_report)

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
    # The category embedding is a compact vocabulary built at train time; use a minimal
    # placeholder here just to surface the architecture defaults.
    config = TransformerPolicyConfig.compact_category(category_vocab=("placeholder",), category_oov_buckets=1)
    model_config = config.to_dict()
    for key in ("category_vocab", "categorical_vocab_size", "category_oov_buckets"):
        model_config.pop(key, None)
    model_config["category_embedding"] = "compact vocabulary built at train time (legacy hash embedding retired)"
    payload = {
        "torch_available": torch_available(),
        "model_config": model_config,
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
    # Surface the missing-neural-extra message before any file I/O (vocab building reads data).
    require_torch()
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
        switch_action_loss_weight=args.switch_action_loss_weight,
        action_family_loss_weight=args.action_family_loss_weight,
        switch_target_loss_weight=args.switch_target_loss_weight,
        max_batches=args.max_batches,
        device=args.device,
        objective=args.objective,
        clip_epsilon=args.clip_epsilon,
        entropy_coef=args.entropy_coef,
        normalize_advantage=not args.no_normalize_advantage,
    )
    model_config_kwargs = dict(
        policy_id=args.policy_id,
        window_size=args.window_size,
        embedding_dim=args.embedding_dim,
        transformer_layers=args.layers,
        attention_heads=args.attention_heads,
        feedforward_dim=args.feedforward_dim,
        dropout=args.dropout,
    )
    # The category vocabulary is the closed Gen 3 randbat universe (string->row), the same one
    # the env builds at encode time, so rows align deterministically. (The legacy training-data
    # vocab source is retired: observations now store rows, not collectible hash ids.)
    if args.showdown_root is None:
        raise ValueError("neural training requires --showdown-root for the Gen 3 randbat category vocabulary.")
    from .randbat_vocab import gen3_category_vocabulary

    category_vocab = gen3_category_vocabulary(args.showdown_root, oov_buckets=args.category_oov_buckets)
    model_config = TransformerPolicyConfig.compact_category(
        category_vocab=category_vocab.tokens,
        category_oov_buckets=category_vocab.oov_buckets,
        **model_config_kwargs,
    )
    print(
        f"category vocab (randbat-dex universe): {len(category_vocab.tokens):,} tokens + {args.category_oov_buckets:,} oov "
        f"-> embedding rows {model_config.categorical_vocab_size:,}",
        file=sys.stderr,
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
    # Benchmark loads arbitrary checkpoints; the env builds the vocabulary from showdown_root
    # (the closed-universe default), which matches any checkpoint trained on the same root.
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


def _value_calibration(args: argparse.Namespace) -> int:
    require_torch()
    model, training_result = load_transformer_checkpoint(args.checkpoint, map_location=args.device)
    report = evaluate_value_calibration(
        model=model,
        training_result=training_result,
        paths=args.data,
        batch_size=args.batch_size,
        bins=args.bins,
        device=args.device,
    )
    if args.json:
        print(json.dumps(report.to_dict(), indent=2, sort_keys=True))
    else:
        print_value_calibration_report(report)
    return 0


def _iterate(args: argparse.Namespace) -> int:
    # Surface the missing-neural-extra message before any Showdown file I/O (vocab build).
    require_torch()
    # Fail fast: eval-only baselines (max-damage) cannot seed self-play training.
    reject_eval_only_specs([args.initial_policy], role="self-play initial policy")
    reject_eval_only_specs(args.opponent_policy or (), role="self-play training opponent")
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
    # Self-play always uses the compact full Gen 3 randbat dex universe embedding. Build the
    # vocabulary ONCE and share it between the env (encode-time rows) and the model config
    # (embedding) so rows can never drift.
    if args.showdown_root is None:
        raise ValueError("neural self-play requires --showdown-root (used for the category vocabulary and the env).")
    from .randbat_vocab import gen3_category_vocabulary

    category_vocab = gen3_category_vocabulary(args.showdown_root, oov_buckets=args.category_oov_buckets)
    env_config = LocalShowdownConfig(
        showdown_root=args.showdown_root,
        node_binary=args.node_binary,
        category_vocab=category_vocab,
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
        switch_action_loss_weight=args.switch_action_loss_weight,
        action_family_loss_weight=args.action_family_loss_weight,
        switch_target_loss_weight=args.switch_target_loss_weight,
        max_batches=args.max_batches,
        device=args.device,
        objective=args.objective,
        clip_epsilon=args.clip_epsilon,
        entropy_coef=args.entropy_coef,
        normalize_advantage=not args.no_normalize_advantage,
    )
    iterate_model_config_kwargs = dict(
        policy_id=args.policy_id,
        window_size=args.window_size,
        embedding_dim=args.embedding_dim,
        transformer_layers=args.layers,
        attention_heads=args.attention_heads,
        feedforward_dim=args.feedforward_dim,
        dropout=args.dropout,
    )
    # Reuse the single vocabulary built above (shared with the env), so the embedding rows the
    # model learns are exactly the rows the env encodes.
    model_config = TransformerPolicyConfig.compact_category(
        category_vocab=category_vocab.tokens,
        category_oov_buckets=category_vocab.oov_buckets,
        **iterate_model_config_kwargs,
    )
    print(
        f"category vocab (randbat-dex universe): {len(category_vocab.tokens):,} tokens + "
        f"{args.category_oov_buckets:,} oov -> embedding rows {model_config.categorical_vocab_size:,}",
        file=sys.stderr,
    )
    initial_policy = policy_spec_with_showdown_root(args.initial_policy, args.showdown_root)
    opponent_policies = tuple(
        policy_spec_with_showdown_root(spec, args.showdown_root)
        for spec in (args.opponent_policy or ("random-legal", "simple-legal"))
    )
    # Eval-only references (e.g. max-damage) are allowed here but never seed training above.
    benchmark_references = tuple(
        policy_spec_with_showdown_root(spec, args.showdown_root)
        for spec in (args.benchmark_reference_policy or ())
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
        benchmark_reference_policy_specs=benchmark_references,
        mirror_match=args.mirror_match,
        collection_temperature=args.collection_temperature,
        tensorboard_log_dir=args.tensorboard_logdir,
        max_historical_opponents=args.max_historical_opponents,
        evaluation_games=args.evaluation_games,
        evaluation_seed_start=args.evaluation_seed_start,
        worker_count=args.workers,
        promotion_registry_path=args.promotion_registry,
        required_promoted_opponent_pool_size=args.require_promoted_opponent_pool_size,
        auto_promotion_config=auto_promotion_config,
        post_iteration_audit_config=post_iteration_audit_config,
        post_iteration_audit_failure_mode=args.audit_failure_mode,
        resume=args.resume,
    )
    if args.json:
        print(json.dumps(result.to_dict(), indent=2, sort_keys=True))
    else:
        _print_iterate_summary(result)
    return 0


def _print_run_audit_failure(exc: RunAuditFailure) -> None:
    failed = [check.name for check in exc.result.blocking_failed_checks]
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


def _report(args: argparse.Namespace) -> int:
    manifest = load_neural_selfplay_run_manifest(args.run_dir)
    if args.json:
        print(json.dumps(manifest, indent=2, sort_keys=True))
        return 0
    _print_manifest_report(manifest)
    return 0


def _print_manifest_report(manifest: Mapping[str, Any]) -> None:
    iterations = tuple(_mapping(iteration) for iteration in _sequence(manifest.get("iterations", ())))
    print(f"run_dir: {manifest.get('run_dir')}")
    print(f"current_policy: {_format_manifest_value(manifest.get('current_policy_spec'))}")
    print(f"latest_checkpoint: {_format_manifest_value(manifest.get('latest_checkpoint_path'))}")
    print(f"latest_accepted_checkpoint: {_format_manifest_value(manifest.get('latest_accepted_checkpoint_path'))}")
    _print_source_metadata(_manifest_source_metadata(manifest))
    print(f"iterations: {len(iterations)}")
    if not iterations:
        return
    print("note: incumbent win rate drives advancement; blended benchmark win rate is broad health.")
    print("")
    header = (
        f"{'iter':>4} {'games':>5} {'cap':>4} {'bench_wr':>8} {'inc_wr':>8} {'advance':>7} {'promo':>8} "
        f"{'loss':>10} {'pol_acc':>8} {'value':>10} {'opp_acc':>8} checkpoint"
    )
    print(header)
    print("-" * len(header))
    for iteration in iterations:
        metrics = _mapping(iteration.get("collection_metrics", {}))
        final_epoch = _final_epoch_metrics(iteration)
        advancement = _optional_mapping(iteration.get("advancement"))
        print(
            f"{int(iteration.get('iteration', 0)):4d} "
            f"{int(metrics.get('games', 0)):5d} "
            f"{int(metrics.get('capped_games', 0)):4d} "
            f"{_format_optional_float(_benchmark_win_rate(iteration)):>8} "
            f"{_format_optional_float(_incumbent_win_rate(iteration)):>8} "
            f"{_format_bool(advancement.get('advance_collector')):>7} "
            f"{_manifest_promotion_status(iteration):>8} "
            f"{_format_optional_float(final_epoch.get('loss') if final_epoch else None, digits=6):>10} "
            f"{_format_optional_float(final_epoch.get('policy_accuracy') if final_epoch else None, digits=4):>8} "
            f"{_format_optional_float(final_epoch.get('value_loss') if final_epoch else None, digits=6):>10} "
            f"{_format_optional_float(final_epoch.get('opponent_accuracy') if final_epoch else None, digits=4):>8} "
            f"{iteration.get('checkpoint_path')}"
        )


def _final_epoch_metrics(iteration: Mapping[str, Any]) -> Mapping[str, Any] | None:
    training = _mapping(iteration.get("training", {}))
    epochs = tuple(_mapping(epoch) for epoch in _sequence(training.get("epochs", ())))
    return epochs[-1] if epochs else None


def _benchmark_win_rate(iteration: Mapping[str, Any]) -> float | None:
    benchmark = iteration.get("benchmark")
    if benchmark is None:
        return None
    benchmark_payload = _mapping(benchmark)
    policy_id = _iteration_policy_id(iteration)
    if policy_id is None:
        return None
    wins = 0
    games = 0
    for result in tuple(_mapping(result) for result in _sequence(benchmark_payload.get("head_to_heads", ()))):
        result_games = int(result.get("games", 0))
        if result.get("first_policy_id") == policy_id:
            wins += int(result.get("first_policy_wins", 0))
            games += result_games
        elif result.get("second_policy_id") == policy_id:
            wins += int(result.get("second_policy_wins", 0))
            games += result_games
    if games:
        return wins / games
    for result in tuple(_mapping(result) for result in _sequence(benchmark_payload.get("matchups", ()))):
        metrics = _mapping(result.get("metrics", {}))
        result_games = int(metrics.get("games", 0))
        if result.get("p1_policy_id") == policy_id:
            wins += int(metrics.get("p1_wins", 0))
            games += result_games
        elif result.get("p2_policy_id") == policy_id:
            wins += int(metrics.get("p2_wins", 0))
            games += result_games
    return (wins / games) if games else None


def _incumbent_win_rate(iteration: Mapping[str, Any]) -> float | None:
    advancement = _optional_mapping(iteration.get("advancement"))
    candidate_win_rate = advancement.get("candidate_win_rate")
    if candidate_win_rate is not None:
        return float(candidate_win_rate)
    candidate_policy_id = advancement.get("candidate_policy_id")
    incumbent_policy_id = advancement.get("incumbent_policy_id")
    if not isinstance(candidate_policy_id, str) or not isinstance(incumbent_policy_id, str):
        return None
    benchmark = iteration.get("benchmark")
    if benchmark is None:
        return None
    benchmark_payload = _mapping(benchmark)
    for result in tuple(_mapping(result) for result in _sequence(benchmark_payload.get("head_to_heads", ()))):
        ids = {result.get("first_policy_id"), result.get("second_policy_id")}
        if ids != {candidate_policy_id, incumbent_policy_id}:
            continue
        games = int(result.get("games", 0))
        if not games:
            return None
        if result.get("first_policy_id") == candidate_policy_id:
            return int(result.get("first_policy_wins", 0)) / games
        return int(result.get("second_policy_wins", 0)) / games
    return None


def _iteration_policy_id(iteration: Mapping[str, Any]) -> str | None:
    training = _mapping(iteration.get("training", {}))
    model_config = _mapping(training.get("model_config", {}))
    policy_id = model_config.get("policy_id")
    return policy_id if isinstance(policy_id, str) and policy_id else None


def _manifest_promotion_status(iteration: Mapping[str, Any]) -> str:
    promotion = iteration.get("promotion")
    if promotion is None:
        return "-"
    promotion_payload = _optional_mapping(promotion)
    return "yes" if promotion_payload.get("recorded") else "no"


def _manifest_source_metadata(manifest: Mapping[str, Any]) -> Mapping[str, Any]:
    source = manifest.get("source")
    return dict(source) if isinstance(source, Mapping) else {}


def _optional_mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _print_source_metadata(metadata: Mapping[str, Any]) -> None:
    if not metadata:
        print("source_metadata: -")
        return
    print("source_metadata:")
    print(f"  available: {_format_bool(metadata.get('available'))}")
    print(f"  branch: {_format_manifest_value(metadata.get('branch'))}")
    print(f"  head: {_format_manifest_value(metadata.get('head'))}")
    print(f"  dirty: {_format_bool(metadata.get('dirty'))}")
    print(f"  repo_root: {_format_manifest_value(metadata.get('repo_root'))}")
    if metadata.get("error") is not None:
        print(f"  error: {_format_manifest_value(metadata.get('error'))}")


def _promotion_status(promotion) -> str:
    if promotion is None:
        return "-"
    return "recorded" if promotion.recorded else "failed"


def _format_bool(value: Any) -> str:
    if value is True:
        return "yes"
    if value is False:
        return "no"
    return "-"


def _format_manifest_value(value: Any) -> str:
    if value is None:
        return "-"
    return str(value)


def _format_optional_float(value: object, *, digits: int = 3) -> str:
    if value is None:
        return "-"
    return f"{float(value):.{digits}f}"


def print_value_calibration_report(report: ValueCalibrationReport) -> None:
    print(f"examples: {report.examples}")
    print(f"mse: {report.mse:.6f}")
    print(f"mae: {report.mae:.6f}")
    print(f"bias: {report.bias:.6f}")
    print(f"sign_accuracy: {report.sign_accuracy:.4f}")
    print(f"expected_calibration_error: {report.expected_calibration_error:.6f}")
    print("")
    header = f"{'bin':>13} {'count':>6} {'pred':>9} {'return':>9} {'cal_err':>9}"
    print(header)
    print("-" * len(header))
    for bin_result in report.bins:
        print(
            f"[{bin_result.lower:5.2f},{bin_result.upper:5.2f}) "
            f"{bin_result.count:6d} "
            f"{bin_result.mean_prediction:9.4f} "
            f"{bin_result.mean_return:9.4f} "
            f"{bin_result.calibration_error:9.4f}"
        )


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
