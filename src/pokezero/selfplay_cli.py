"""Command-line self-play iteration harness."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Mapping

from .cli_audit import add_post_iteration_audit_arguments, post_iteration_audit_config_from_args
from .collection import policy_spec_with_showdown_root
from .linear_policy import LinearTrainingConfig
from .local_showdown import LocalShowdownConfig, LocalShowdownEnv
from .run_audit import RunAuditFailure
from .rollout import RolloutConfig
from .selfplay import (
    SelfPlayPromotionConfig,
    _mapping,
    _sequence,
    load_selfplay_run_manifest,
    run_selfplay_iterations,
)
from .eval_cli import _add_gate_arguments, _gate_config_from_args


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
        "--validation-data",
        type=Path,
        action="append",
        default=None,
        help="Held-out rollout JSONL used for validation metrics after each train step. May be repeated.",
    )
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
            "Additional policy spec retained as a benchmark reference across iterations. "
            "May be repeated. A linear --initial-policy checkpoint is retained by default."
        ),
    )
    iterate.add_argument("--max-historical-opponents", type=int, default=3, help="Number of older checkpoints kept in the opponent pool.")
    iterate.add_argument(
        "--promotion-registry",
        type=Path,
        default=None,
        help="Optional promotion registry. When set, historical opponents come from promoted checkpoints instead of raw iteration history.",
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
        help="Optional artifact directory for auto-promoted checkpoint copies.",
    )
    iterate.add_argument(
        "--promotion-label-prefix",
        default="selfplay",
        help="Label prefix for auto-promotion entries. Use an empty string to omit labels.",
    )
    iterate.add_argument("--promotion-notes", default=None, help="Optional notes stored on each auto-promotion entry.")
    iterate.add_argument(
        "--allow-duplicate-promotion",
        action="store_true",
        help="Allow auto-promotion to record a checkpoint already present in the registry.",
    )
    _add_gate_arguments(iterate)
    iterate.add_argument("--evaluation-games", type=int, default=0, help="Optional benchmark games per baseline matchup after each iteration.")
    iterate.add_argument("--evaluation-seed-start", type=int, default=1_000_000, help="First deterministic evaluation seed.")
    iterate.add_argument("--epochs", type=int, default=1, help="Training epochs per iteration.")
    iterate.add_argument("--learning-rate", type=float, default=0.05, help="SGD learning rate.")
    iterate.add_argument(
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
    iterate.add_argument("--l2", type=float, default=0.0, help="L2 penalty applied on active features.")
    iterate.add_argument("--feature-count", type=int, default=131_072, help="Hashed feature bucket count.")
    iterate.add_argument("--window-size", type=int, default=1, help="Per-player observation history window.")
    iterate.add_argument("--discount", type=float, default=1.0, help="Terminal return discount per player decision.")
    iterate.add_argument(
        "--capped-terminal-value",
        type=float,
        default=-0.25,
        help="Return assigned to each player in capped self-play games. Default is a mild double-loss penalty.",
    )
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
    add_post_iteration_audit_arguments(iterate)
    iterate.set_defaults(func=_iterate)

    report = subparsers.add_parser("report", help="Print a summary of a self-play run manifest.")
    report.add_argument("--run-dir", type=Path, required=True, help="Self-play run directory containing manifest.json.")
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


def _iterate(args: argparse.Namespace) -> int:
    if args.auto_promote and args.promotion_registry is None:
        raise ValueError("--auto-promote requires --promotion-registry.")
    if args.auto_promote and args.evaluation_games <= 0 and args.require_benchmark is not False:
        raise ValueError("--auto-promote requires --evaluation-games > 0 unless --allow-missing-benchmark is set.")
    post_iteration_audit_config = post_iteration_audit_config_from_args(args)
    if (
        post_iteration_audit_config is not None
        and post_iteration_audit_config.require_benchmark
        and args.evaluation_games <= 0
    ):
        raise ValueError(
            "--audit-after-iteration requires --evaluation-games > 0 unless "
            "--audit-allow-missing-benchmark is set."
        )
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
        capped_terminal_value=args.capped_terminal_value,
        objective=args.objective,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        opponent_action_loss_weight=args.opponent_action_loss_weight,
        l2=args.l2,
        shuffle_buffer_size=args.shuffle_buffer_size,
        shuffle_seed=args.shuffle_seed,
        max_examples=args.max_examples,
        policy_id=args.policy_id,
    )
    policy_showdown_root = env_config.resolved_showdown_root()
    initial_policy = policy_spec_with_showdown_root(args.initial_policy, policy_showdown_root)
    fixed_opponents = tuple(
        policy_spec_with_showdown_root(spec, policy_showdown_root)
        for spec in (args.opponent_policy or ("random-legal", "simple-legal"))
    )
    benchmark_references = tuple(
        policy_spec_with_showdown_root(spec, policy_showdown_root)
        for spec in (args.benchmark_reference_policy or ())
    )
    auto_promotion_config = _auto_promotion_config_from_args(args)
    result = run_selfplay_iterations(
        run_dir=args.run_dir,
        iterations=args.iterations,
        games_per_iteration=args.games_per_iteration,
        env_factory=lambda: LocalShowdownEnv(env_config),
        rollout_config=rollout_config,
        training_config=training_config,
        seed_start=args.seed_start,
        initial_policy_spec=initial_policy,
        fixed_opponent_policy_specs=fixed_opponents,
        benchmark_reference_policy_specs=benchmark_references,
        max_historical_opponents=args.max_historical_opponents,
        evaluation_games=args.evaluation_games,
        evaluation_seed_start=args.evaluation_seed_start,
        validation_rollout_paths=tuple(args.validation_data or ()),
        promotion_registry_path=args.promotion_registry,
        auto_promotion_config=auto_promotion_config,
        post_iteration_audit_config=post_iteration_audit_config,
        resume=args.resume,
        worker_count=args.workers,
    )
    _print_run_summary(result)
    return 0


def _print_run_audit_failure(exc: RunAuditFailure) -> None:
    failed = [check.name for check in exc.result.checks if not check.passed]
    print(f"audit_failed: {exc.result.manifest_path}", file=sys.stderr)
    print(f"failed_checks: {', '.join(failed) if failed else 'unknown'}", file=sys.stderr)


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
            f"opponent_accuracy={_format_optional_float(getattr(final_epoch, 'opponent_accuracy', None), digits=4)} "
            f"promotion={_promotion_status(getattr(iteration, 'promotion', None))} "
            f"checkpoint={iteration.checkpoint_path}"
        )
    if result.latest_checkpoint_path is not None:
        print(f"latest_checkpoint: {result.latest_checkpoint_path}")


def _auto_promotion_config_from_args(args: argparse.Namespace) -> SelfPlayPromotionConfig | None:
    if not args.auto_promote:
        return None
    gate_args = argparse.Namespace(**vars(args))
    gate_args.registry = None
    label_prefix = args.promotion_label_prefix if args.promotion_label_prefix else None
    return SelfPlayPromotionConfig(
        registry_path=args.promotion_registry,
        gate_config=_gate_config_from_args(gate_args),
        artifact_dir=args.promotion_artifact_dir,
        label_prefix=label_prefix,
        notes=args.promotion_notes,
        allow_duplicate=args.allow_duplicate_promotion,
    )


def _promotion_status(promotion) -> str:
    if promotion is None:
        return "-"
    return "recorded" if promotion.recorded else "failed"


def _report(args: argparse.Namespace) -> int:
    manifest = load_selfplay_run_manifest(args.run_dir)
    if args.json:
        print(json.dumps(manifest, indent=2, sort_keys=True))
        return 0
    _print_manifest_report(manifest)
    return 0


def _print_manifest_report(manifest: Mapping[str, Any]) -> None:
    iterations = tuple(_mapping(iteration) for iteration in _sequence(manifest.get("iterations", ())))
    print(f"run_dir: {manifest.get('run_dir')}")
    print(f"latest_checkpoint: {manifest.get('latest_checkpoint_path')}")
    print(f"iterations: {len(iterations)}")
    if not iterations:
        return
    print("note: benchmark win rate is the strength signal; fit metrics measure imitation of rollout labels.")
    if _validation_paths_changed(iterations):
        print("warning: validation rollout paths changed across iterations; fit metrics are not directly comparable.")
    print("")
    header = (
        f"{'iter':>4} {'games':>5} {'cap':>4} {'p1w':>4} {'p2w':>4} {'ties':>4} "
        f"{'bench_wr':>8} {'promo':>8} {'dec/s':>8} {'avg_dec':>8} {'peak_mb':>8} "
        f"{'fit':>5} {'fit_loss':>10} {'fit_acc':>8} {'opp_acc':>8} checkpoint"
    )
    print(header)
    print("-" * len(header))
    for iteration in iterations:
        metrics = _mapping(iteration.get("collection_metrics", {}))
        training = _mapping(iteration.get("training", {}))
        fit_source, fit_metrics = _fit_metrics(training)
        print(
            f"{int(iteration.get('iteration', 0)):4d} "
            f"{int(metrics.get('games', 0)):5d} "
            f"{int(metrics.get('capped_games', 0)):4d} "
            f"{int(metrics.get('p1_wins', 0)):4d} "
            f"{int(metrics.get('p2_wins', 0)):4d} "
            f"{int(metrics.get('ties', 0)):4d} "
            f"{_format_optional_float(_benchmark_win_rate(iteration)):>8} "
            f"{_manifest_promotion_status(iteration):>8} "
            f"{float(metrics.get('decisions_per_second', 0.0)):8.3f} "
            f"{_format_optional_float(metrics.get('average_decision_rounds')):>8} "
            f"{_format_optional_float(metrics.get('peak_rss_mb')):>8} "
            f"{fit_source:>5} "
            f"{_format_optional_float(fit_metrics.get('loss') if fit_metrics else None, digits=6):>10} "
            f"{_format_optional_float(fit_metrics.get('accuracy') if fit_metrics else None, digits=4):>8} "
            f"{_format_optional_float(fit_metrics.get('opponent_accuracy') if fit_metrics else None, digits=4):>8} "
            f"{iteration.get('checkpoint_path')}"
        )


def _manifest_promotion_status(iteration: Mapping[str, Any]) -> str:
    promotion = iteration.get("promotion")
    if promotion is None:
        return "-"
    return "yes" if _mapping(promotion).get("recorded") else "no"


def _fit_metrics(training: Mapping[str, Any]) -> tuple[str, Mapping[str, Any] | None]:
    validation = training.get("validation_metrics")
    if validation is not None:
        return "val", _mapping(validation)
    epochs = tuple(_mapping(epoch) for epoch in _sequence(training.get("epochs", ())))
    if epochs:
        return "train", epochs[-1]
    return "-", None


def _validation_paths_changed(iterations: tuple[Mapping[str, Any], ...]) -> bool:
    seen = {
        tuple(str(path) for path in _sequence(iteration.get("validation_rollout_paths", ())))
        for iteration in iterations
    }
    return len(seen) > 1


def _benchmark_win_rate(iteration: Mapping[str, Any]) -> float | None:
    benchmark = iteration.get("benchmark")
    if benchmark is None:
        return None
    benchmark_payload = _mapping(benchmark)
    training = _mapping(iteration.get("training", {}))
    model = _mapping(training.get("model", {}))
    policy_id = model.get("policy_id")
    if not isinstance(policy_id, str) or not policy_id:
        return None
    head_to_heads = tuple(_mapping(result) for result in _sequence(benchmark_payload.get("head_to_heads", ())))
    wins = 0
    games = 0
    for result in head_to_heads:
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
    if not games:
        return None
    return wins / games


def _format_optional_float(value: object, *, digits: int = 3) -> str:
    if value is None:
        return "-"
    return f"{float(value):.{digits}f}"


if __name__ == "__main__":
    raise SystemExit(main())
