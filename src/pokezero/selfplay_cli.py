"""Command-line self-play iteration harness."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

from .linear_policy import LinearTrainingConfig
from .local_showdown import LocalShowdownConfig, LocalShowdownEnv
from .rollout import RolloutConfig
from .selfplay import run_selfplay_iterations


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m pokezero.selfplay_cli")
    subparsers = parser.add_subparsers(dest="command", required=True)

    iterate = subparsers.add_parser("iterate", help="Run linear-policy self-play training iterations.")
    iterate.add_argument("--run-dir", type=Path, required=True, help="Directory for rollouts, checkpoints, and manifests.")
    iterate.add_argument("--iterations", type=int, required=True, help="Number of collect/train/evaluate iterations.")
    iterate.add_argument("--resume", action="store_true", help="Continue an existing run directory from its latest manifest.")
    iterate.add_argument("--games-per-iteration", type=int, required=True, help="Rollout games collected before each train step.")
    iterate.add_argument("--workers", type=int, default=1, help="Parallel rollout collection workers per iteration.")
    iterate.add_argument("--showdown-root", type=Path, default=None, help="Built Pokemon Showdown checkout root.")
    iterate.add_argument("--format", dest="format_id", default="gen3randombattle", help="Showdown format id.")
    iterate.add_argument("--seed-start", type=int, default=1, help="First deterministic self-play seed.")
    iterate.add_argument("--max-decision-rounds", type=int, default=250, help="Rollout decision-round cap.")
    iterate.add_argument("--node-binary", default="node", help="Node executable used for the BattleStream bridge.")
    iterate.add_argument("--initial-policy", default="random-legal", help="Policy spec used before the first checkpoint exists.")
    iterate.add_argument(
        "--opponent-policy",
        action="append",
        default=None,
        help="Fixed opponent policy spec. May be repeated. Defaults to random-legal and simple-legal.",
    )
    iterate.add_argument("--max-historical-opponents", type=int, default=3, help="Number of older checkpoints kept in the opponent pool.")
    iterate.add_argument("--evaluation-games", type=int, default=0, help="Optional benchmark games per baseline matchup after each iteration.")
    iterate.add_argument("--evaluation-seed-start", type=int, default=1_000_000, help="First deterministic evaluation seed.")
    iterate.add_argument("--epochs", type=int, default=1, help="Training epochs per iteration.")
    iterate.add_argument("--learning-rate", type=float, default=0.05, help="SGD learning rate.")
    iterate.add_argument("--l2", type=float, default=0.0, help="L2 penalty applied on active features.")
    iterate.add_argument("--feature-count", type=int, default=131_072, help="Hashed feature bucket count.")
    iterate.add_argument("--window-size", type=int, default=1, help="Per-player observation history window.")
    iterate.add_argument("--discount", type=float, default=1.0, help="Terminal return discount per player decision.")
    iterate.add_argument(
        "--objective",
        choices=("behavior-cloning", "reward-weighted"),
        default="reward-weighted",
        help="Training objective for each iteration.",
    )
    iterate.add_argument("--shuffle-buffer-size", type=int, default=1024, help="Streaming shuffle buffer size; 0 disables shuffling.")
    iterate.add_argument("--shuffle-seed", type=int, default=1, help="Base deterministic shuffle seed.")
    iterate.add_argument("--max-examples", type=int, default=None, help="Optional max examples per epoch.")
    iterate.add_argument("--policy-id", default="linear-selfplay", help="Policy id prefix stored in checkpoints.")
    iterate.set_defaults(func=_iterate)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


def _iterate(args: argparse.Namespace) -> int:
    env_config = LocalShowdownConfig(
        showdown_root=args.showdown_root,
        node_binary=args.node_binary,
    )
    rollout_config = RolloutConfig(
        max_decision_rounds=args.max_decision_rounds,
        format_id=args.format_id,
    )
    training_config = LinearTrainingConfig(
        feature_count=args.feature_count,
        window_size=args.window_size,
        discount=args.discount,
        objective=args.objective,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        l2=args.l2,
        shuffle_buffer_size=args.shuffle_buffer_size,
        shuffle_seed=args.shuffle_seed,
        max_examples=args.max_examples,
        policy_id=args.policy_id,
    )
    result = run_selfplay_iterations(
        run_dir=args.run_dir,
        iterations=args.iterations,
        games_per_iteration=args.games_per_iteration,
        env_factory=lambda: LocalShowdownEnv(env_config),
        rollout_config=rollout_config,
        training_config=training_config,
        seed_start=args.seed_start,
        initial_policy_spec=args.initial_policy,
        fixed_opponent_policy_specs=tuple(args.opponent_policy or ("random-legal", "simple-legal")),
        max_historical_opponents=args.max_historical_opponents,
        evaluation_games=args.evaluation_games,
        evaluation_seed_start=args.evaluation_seed_start,
        resume=args.resume,
        worker_count=args.workers,
    )
    _print_run_summary(result)
    return 0


def _print_run_summary(result) -> None:
    print(f"run_dir: {result.run_dir}")
    for iteration in result.iterations:
        final_epoch = iteration.training.final_metrics
        print(
            f"iteration={iteration.iteration} "
            f"games={iteration.metrics.games} "
            f"decisions_per_second={iteration.metrics.decisions_per_second:.3f} "
            f"loss={final_epoch.loss:.6f} "
            f"accuracy={final_epoch.accuracy:.4f} "
            f"checkpoint={iteration.checkpoint_path}"
        )
    if result.latest_checkpoint_path is not None:
        print(f"latest_checkpoint: {result.latest_checkpoint_path}")


if __name__ == "__main__":
    raise SystemExit(main())
