"""Command-line utilities for optional neural policy experiments."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from .neural_policy import (
    TransformerPolicyConfig,
    TransformerTrainingConfig,
    save_transformer_checkpoint,
    torch_available,
    train_transformer_policy,
)


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

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
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


if __name__ == "__main__":
    raise SystemExit(main())
