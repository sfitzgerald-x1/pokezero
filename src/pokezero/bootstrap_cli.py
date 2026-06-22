"""Command-line bootstrap workflows."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys

from .bootstrap import (
    DEFAULT_BENCHMARK_GAMES,
    DEFAULT_PREFLIGHT_GAMES,
    DEFAULT_PREFLIGHT_SEED_START,
    benchmark_teacher_policy,
    run_teacher_bootstrap,
)
from .collection import policy_spec_with_showdown_root
from .linear_policy import LinearTrainingConfig
from .local_showdown import LocalShowdownConfig, LocalShowdownEnv
from .rollout import RolloutConfig

TEACHER_BENCHMARK_PREFLIGHT_SCHEMA_VERSION = "pokezero.teacher_benchmark_preflight.v1"


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

    teacher_benchmark = subparsers.add_parser("teacher-benchmark", help="Benchmark a teacher policy against fixed baselines.")
    teacher_benchmark.add_argument("--games", type=int, default=DEFAULT_BENCHMARK_GAMES, help="Benchmark games per matchup.")
    teacher_benchmark.add_argument("--showdown-root", type=Path, default=None, help="Built Pokemon Showdown checkout root.")
    teacher_benchmark.add_argument("--format", dest="format_id", default="gen3randombattle", help="Showdown format id.")
    teacher_benchmark.add_argument("--seed-start", type=int, default=1, help="First deterministic benchmark seed.")
    teacher_benchmark.add_argument("--max-decision-rounds", type=int, default=250, help="Rollout decision-round cap.")
    teacher_benchmark.add_argument("--node-binary", default="node", help="Node executable used for the BattleStream bridge.")
    teacher_benchmark.add_argument("--teacher-policy", default="scripted-teacher", help="Teacher policy spec.")
    teacher_benchmark.add_argument(
        "--baseline-policy",
        action="append",
        default=None,
        help="Baseline policy spec. May be repeated. Defaults to simple-legal and random-legal.",
    )
    teacher_benchmark.add_argument("--out", type=Path, default=None, help="Optional JSON report path for the benchmark preflight payload.")
    teacher_benchmark.add_argument(
        "--min-teacher-win-rate",
        type=float,
        default=None,
        help="Fail unless the teacher reaches this win rate against every baseline head-to-head.",
    )
    teacher_benchmark.add_argument(
        "--max-capped-rate",
        type=float,
        default=None,
        help="Fail unless every teacher benchmark head-to-head has capped-game rate at or below this value.",
    )
    teacher_benchmark.add_argument(
        "--fail-on-degraded-decisions",
        action="store_true",
        help="Fail if the teacher used unknown-move or fallback decisions during the benchmark.",
    )
    teacher_benchmark.add_argument("--json", action="store_true", help="Print the benchmark report as JSON.")
    teacher_benchmark.set_defaults(func=_teacher_benchmark)
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


def _teacher_benchmark(args: argparse.Namespace) -> int:
    _validate_optional_rate(args.min_teacher_win_rate, "--min-teacher-win-rate")
    _validate_optional_rate(args.max_capped_rate, "--max-capped-rate")
    env_config = LocalShowdownConfig(
        showdown_root=args.showdown_root,
        node_binary=args.node_binary,
    )
    policy_showdown_root = env_config.resolved_showdown_root()
    teacher_policy = policy_spec_with_showdown_root(args.teacher_policy, policy_showdown_root)
    baseline_policies = (
        tuple(policy_spec_with_showdown_root(spec, policy_showdown_root) for spec in args.baseline_policy)
        if args.baseline_policy is not None
        else None
    )
    result = benchmark_teacher_policy(
        env_factory=lambda: LocalShowdownEnv(env_config),
        rollout_config=RolloutConfig(
            max_decision_rounds=args.max_decision_rounds,
            format_id=args.format_id,
        ),
        teacher_policy_spec=teacher_policy,
        baseline_policy_specs=baseline_policies,
        games=args.games,
        seed_start=args.seed_start,
    )
    teacher_policy_id = _teacher_policy_id_from_spec(args.teacher_policy) or _teacher_policy_id_from_benchmark(result)
    checks = _teacher_benchmark_checks(
        result,
        teacher_policy_id=teacher_policy_id,
        min_teacher_win_rate=args.min_teacher_win_rate,
        max_capped_rate=args.max_capped_rate,
        fail_on_degraded_decisions=args.fail_on_degraded_decisions,
    )
    passed = all(bool(check["passed"]) for check in checks)
    payload = _teacher_benchmark_payload(
        result,
        teacher_policy_id=teacher_policy_id,
        checks=checks,
        passed=passed,
    )
    if args.out is not None:
        _write_json(args.out, payload)
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        _print_teacher_benchmark_result(result, checks=checks, passed=passed)
        if args.out is not None:
            print(f"report: {args.out}")
    return 0 if passed else 2


def _validate_optional_rate(value: float | None, flag_name: str) -> None:
    if value is None:
        return
    if not math.isfinite(value) or value < 0.0 or value > 1.0:
        raise ValueError(f"{flag_name} must be between 0 and 1.")


def _teacher_policy_id_from_spec(spec: str) -> str | None:
    policy_body = spec.strip().partition("?")[0].strip()
    lowered = policy_body.lower()
    if lowered in {"random-legal", "simple-legal", "scripted-teacher"}:
        return lowered
    if lowered.startswith("linear:"):
        checkpoint = policy_body[len("linear:") :].strip()
        if not checkpoint:
            return None
        try:
            payload = json.loads(Path(checkpoint).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        policy_id = payload.get("policy_id")
        if isinstance(policy_id, str) and policy_id.strip():
            return policy_id
    return None


def _teacher_policy_id_from_benchmark(result) -> str:
    if result.benchmark.matchups:
        return result.benchmark.matchups[0].p1_policy_id
    if result.benchmark.head_to_head_results:
        return result.benchmark.head_to_head_results[0].first_policy_id
    return "unknown-teacher"


def _teacher_benchmark_checks(
    result,
    *,
    teacher_policy_id: str,
    min_teacher_win_rate: float | None,
    max_capped_rate: float | None,
    fail_on_degraded_decisions: bool,
) -> list[dict[str, object]]:
    checks: list[dict[str, object]] = []
    matched_head_to_heads = 0
    for row in result.benchmark.head_to_head_results:
        if row.first_policy_id == teacher_policy_id:
            teacher_win_rate = row.first_policy_win_rate
            opponent_policy_id = row.second_policy_id
        elif row.second_policy_id == teacher_policy_id:
            teacher_win_rate = row.second_policy_win_rate
            opponent_policy_id = row.first_policy_id
        else:
            continue
        matched_head_to_heads += 1
        capped_rate = row.capped_games / row.games if row.games else 0.0
        if min_teacher_win_rate is not None:
            checks.append(
                _preflight_check(
                    name=f"teacher_win_rate:{opponent_policy_id}",
                    passed=teacher_win_rate >= min_teacher_win_rate,
                    observed=teacher_win_rate,
                    threshold=min_teacher_win_rate,
                    message=(
                        f"teacher win rate vs {opponent_policy_id} "
                        f"observed={teacher_win_rate:.3f} required>={min_teacher_win_rate:.3f}"
                    ),
                )
            )
        if max_capped_rate is not None:
            checks.append(
                _preflight_check(
                    name=f"capped_rate:{opponent_policy_id}",
                    passed=capped_rate <= max_capped_rate,
                    observed=capped_rate,
                    threshold=max_capped_rate,
                    message=(
                        f"capped rate vs {opponent_policy_id} "
                        f"observed={capped_rate:.3f} required<={max_capped_rate:.3f}"
                    ),
                )
            )
    if (min_teacher_win_rate is not None or max_capped_rate is not None) and matched_head_to_heads == 0:
        checks.append(
            _preflight_check(
                name="teacher_head_to_head_present",
                passed=False,
                observed=0,
                threshold=1,
                message=f"teacher policy {teacher_policy_id} did not appear in any benchmark head-to-head row",
            )
        )
    if fail_on_degraded_decisions:
        unknown_moves = int(result.teacher_decision_summary.get("unknown_move_decisions", 0))
        fallbacks = int(result.teacher_decision_summary.get("fallback_decisions", 0))
        degraded = unknown_moves + fallbacks
        checks.append(
            _preflight_check(
                name="teacher_degraded_decisions",
                passed=degraded == 0,
                observed=degraded,
                threshold=0,
                message=(
                    "teacher degraded decisions "
                    f"{degraded} == 0 (unknown_moves={unknown_moves}, fallbacks={fallbacks})"
                ),
            )
        )
    return checks


def _preflight_check(
    *,
    name: str,
    passed: bool,
    observed: float | int,
    threshold: float | int,
    message: str,
) -> dict[str, object]:
    return {
        "name": name,
        "passed": passed,
        "observed": observed,
        "threshold": threshold,
        "message": message,
    }


def _teacher_benchmark_payload(
    result,
    *,
    teacher_policy_id: str,
    checks: list[dict[str, object]],
    passed: bool,
) -> dict[str, object]:
    payload = result.to_dict()
    payload["schema_version"] = TEACHER_BENCHMARK_PREFLIGHT_SCHEMA_VERSION
    payload["teacher_policy_id"] = teacher_policy_id
    payload["passed"] = passed
    payload["checks"] = checks
    return payload


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(f".{path.name}.tmp")
    temporary_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary_path.replace(path)


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


def _print_teacher_benchmark_result(result, *, checks: list[dict[str, object]], passed: bool) -> None:
    report = result.benchmark
    print(f"format: {report.format_id}")
    print(f"games_per_matchup: {report.games_per_matchup}")
    print(f"total_games: {report.total_games}")
    print(f"average_decision_rounds: {report.average_decision_rounds:.3f}")
    for row in report.head_to_head_results:
        print(
            f"benchmark {row.label}: "
            f"{row.first_policy_id}_wr={row.first_policy_win_rate:.3f} "
            f"{row.second_policy_id}_wr={row.second_policy_win_rate:.3f} "
            f"capped={row.capped_games}"
        )
    if any(row.capped_games for row in report.head_to_head_results):
        print("note: benchmark win rates include capped games in the denominator.")
    summary = result.teacher_decision_summary
    print(
        "teacher_decisions: "
        f"scripted={summary.get('scripted_teacher_decisions', 0)} "
        f"unknown_moves={summary.get('unknown_move_decisions', 0)} "
        f"fallbacks={summary.get('fallback_decisions', 0)}"
    )
    if checks:
        print(f"preflight: {'PASS' if passed else 'FAIL'}")
        for check in checks:
            print(
                f"- {'PASS' if check['passed'] else 'FAIL'} {check['name']}: "
                f"{check['message']}"
            )
    fallback_reasons = summary.get("fallback_reasons") or {}
    if fallback_reasons:
        print("teacher_fallback_reasons:")
        for reason, count in sorted(fallback_reasons.items()):
            print(f"- {reason}: {count}")


if __name__ == "__main__":
    raise SystemExit(main())
