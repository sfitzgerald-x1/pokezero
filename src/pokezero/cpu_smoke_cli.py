"""CLI for the local CPU self-play smoke workflow."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from .collection import policy_spec_with_showdown_root
from .cpu_smoke import run_cpu_smoke_experiment
from .evaluation_profiles import EVALUATION_PROFILES
from .local_showdown import LocalShowdownConfig, LocalShowdownEnv
from .rollout import RolloutConfig


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m pokezero.cpu_smoke_cli")
    parser.add_argument("--run-dir", type=Path, required=True, help="Output directory for the smoke run.")
    parser.add_argument("--showdown-root", type=Path, default=None, help="Built Pokemon Showdown checkout root.")
    parser.add_argument("--format", dest="format_id", default="gen3randombattle", help="Showdown format id.")
    parser.add_argument("--node-binary", default="node", help="Node executable used for the BattleStream bridge.")
    parser.add_argument("--max-decision-rounds", type=int, default=250, help="Rollout decision-round cap.")
    parser.add_argument("--workers", type=int, default=1, help="Parallel rollout collection workers.")
    parser.add_argument("--audit-profile", choices=tuple(sorted(EVALUATION_PROFILES)), default="smoke")
    parser.add_argument("--teacher-policy", default="scripted-teacher", help="Teacher policy spec.")
    parser.add_argument(
        "--bootstrap-opponent-policy",
        action="append",
        default=None,
        help="Opponent policy for teacher bootstrap collection. May be repeated.",
    )
    parser.add_argument(
        "--selfplay-opponent-policy",
        action="append",
        default=None,
        help="Fixed self-play opponent policy. May be repeated. Defaults to random-legal and simple-legal.",
    )
    parser.add_argument("--train-games", type=int, default=4)
    parser.add_argument("--validation-games", type=int, default=2)
    parser.add_argument("--bootstrap-benchmark-games", type=int, default=2)
    parser.add_argument("--preflight-games", type=int, default=1)
    parser.add_argument("--selfplay-iterations", type=int, default=1)
    parser.add_argument("--games-per-iteration", type=int, default=4)
    parser.add_argument("--evaluation-games", type=int, default=2)
    parser.add_argument("--feature-count", type=int, default=8192)
    parser.add_argument("--window-size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--json", action="store_true", help="Print summary JSON.")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    try:
        result = _run(args)
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    if args.json:
        print(json.dumps(result.to_dict(), indent=2, sort_keys=True))
    else:
        _print_summary(result)
    return 0 if result.passed else 2


def _run(args: argparse.Namespace):
    env_config = LocalShowdownConfig(
        showdown_root=args.showdown_root,
        node_binary=args.node_binary,
    )
    showdown_root = env_config.resolved_showdown_root()
    teacher_policy = policy_spec_with_showdown_root(args.teacher_policy, showdown_root)
    bootstrap_opponents = (
        tuple(policy_spec_with_showdown_root(spec, showdown_root) for spec in args.bootstrap_opponent_policy)
        if args.bootstrap_opponent_policy is not None
        else None
    )
    fixed_opponents = tuple(
        policy_spec_with_showdown_root(spec, showdown_root)
        for spec in (args.selfplay_opponent_policy or ("random-legal", "simple-legal"))
    )
    return run_cpu_smoke_experiment(
        run_dir=args.run_dir,
        env_factory=lambda: LocalShowdownEnv(env_config),
        rollout_config=RolloutConfig(
            max_decision_rounds=args.max_decision_rounds,
            format_id=args.format_id,
        ),
        audit_profile=args.audit_profile,
        train_games=args.train_games,
        validation_games=args.validation_games,
        bootstrap_benchmark_games=args.bootstrap_benchmark_games,
        preflight_games=args.preflight_games,
        selfplay_iterations=args.selfplay_iterations,
        games_per_iteration=args.games_per_iteration,
        evaluation_games=args.evaluation_games,
        worker_count=args.workers,
        teacher_policy_spec=teacher_policy,
        bootstrap_opponent_policy_specs=bootstrap_opponents,
        fixed_opponent_policy_specs=fixed_opponents,
        feature_count=args.feature_count,
        window_size=args.window_size,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
    )


def _print_summary(result) -> None:
    latest_checkpoint = result.selfplay.latest_checkpoint_path
    print(f"status: {'PASS' if result.passed else 'FAIL'}")
    print(f"run_dir: {result.run_dir}")
    print(f"summary: {result.summary_path}")
    print(f"audit_profile: {result.audit_profile}")
    print(f"bootstrap_manifest: {result.bootstrap.manifest_path}")
    print(f"bootstrap_checkpoint: {result.bootstrap.checkpoint_path}")
    print(f"selfplay_manifest: {result.selfplay.run_dir / 'manifest.json'}")
    print(f"latest_checkpoint: {latest_checkpoint if latest_checkpoint is not None else '-'}")
    print(f"promotion_registry: {result.promotion_registry_path}")
    print(f"audit_passed: {result.audit.passed}")
    failed = [check.name for check in result.audit.checks if not check.passed]
    if failed:
        print(f"audit_failed_checks: {', '.join(failed)}")
    if result.calibration.notes:
        print("calibration_notes:")
        for note in result.calibration.notes:
            print(f"- {note}")
    if result.calibration.min_latest_benchmark_games < 20:
        print(
            "calibration_warning: smoke-run suggested flags are only a plumbing sanity check; "
            "do not use them as quality thresholds for larger experiments without more benchmark games."
        )
    print("suggested_audit_flags:")
    flags = result.calibration.suggested_cli_flags()
    print(" ".join(flags) if flags else "-")


if __name__ == "__main__":
    raise SystemExit(main())
