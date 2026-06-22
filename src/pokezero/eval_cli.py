"""Command-line promotion gates for experiment manifests."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from .evaluation import (
    DEFAULT_MAX_BENCHMARK_CAPPED_RATE,
    DEFAULT_MAX_COLLECTION_CAPPED_RATE,
    DEFAULT_MAX_INCUMBENT_CAPPED_RATE,
    DEFAULT_MAX_TEACHER_DEGRADATION_RATE,
    DEFAULT_MIN_INCUMBENT_GAMES,
    DEFAULT_MIN_INCUMBENT_WIN_RATE_LOWER_BOUND,
    DEFAULT_MIN_BENCHMARK_GAMES,
    DEFAULT_MIN_BENCHMARK_WIN_RATE,
    DEFAULT_MIN_INCUMBENT_WIN_RATE,
    DEFAULT_INCUMBENT_CONFIDENCE_Z,
    PromotionGateConfig,
    evaluate_promotion_gate,
)
from .promotion import load_promotion_registry, record_promotion, verify_promotion_registry
from .run_audit import (
    DEFAULT_MAX_BENCHMARK_WIN_RATE_DROP,
    DEFAULT_MAX_CONSECUTIVE_PROMOTION_FAILURES,
    DEFAULT_AUDIT_CALIBRATION_MARGIN,
    RunAuditConfig,
    audit_run,
    calibrate_run_audit,
)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m pokezero.eval_cli")
    subparsers = parser.add_subparsers(dest="command", required=True)

    gate = subparsers.add_parser("gate", help="Evaluate whether a manifest clears promotion thresholds.")
    gate.add_argument("path", type=Path, help="Experiment run directory or manifest.json path.")
    gate.add_argument("--registry", type=Path, default=None, help="Optional promotion registry used as the default incumbent source.")
    _add_gate_arguments(gate)
    gate.add_argument("--json", action="store_true", help="Print the gate result as JSON.")
    gate.set_defaults(func=_gate)

    promote = subparsers.add_parser("promote", help="Evaluate a candidate and append it to a promotion registry if it passes.")
    promote.add_argument("path", type=Path, help="Experiment run directory or manifest.json path.")
    promote.add_argument("--registry", type=Path, required=True, help="Promotion registry JSON path. Also defaults the incumbent to the latest registry entry.")
    promote.add_argument("--label", default=None, help="Optional short label for the promotion entry.")
    promote.add_argument("--notes", default=None, help="Optional notes stored with the promotion entry.")
    promote.add_argument(
        "--artifact-dir",
        type=Path,
        default=None,
        help="Optional directory that receives a stable copy of the promoted checkpoint.",
    )
    promote.add_argument("--allow-duplicate", action="store_true", help="Allow recording a checkpoint already present in the registry.")
    _add_gate_arguments(promote)
    promote.add_argument("--json", action="store_true", help="Print the promotion result as JSON.")
    promote.set_defaults(func=_promote)

    promotions = subparsers.add_parser("promotions", help="Print a promotion registry summary.")
    promotions.add_argument("--registry", type=Path, required=True, help="Promotion registry JSON path.")
    promotions.add_argument("--verify", action="store_true", help="Verify promoted checkpoint paths and stored checksums.")
    promotions.add_argument("--skip-checksum", action="store_true", help="With --verify, skip checksum validation even when metadata exists.")
    promotions.add_argument("--require-checksum", action="store_true", help="With --verify, fail entries that do not include checksum metadata.")
    promotions.add_argument("--verify-loadable", action="store_true", help="With --verify, load each promoted policy spec through the normal policy selection path.")
    promotions.add_argument(
        "--opponent-pool-size",
        type=int,
        default=None,
        help="Preview the latest N promoted policy specs that self-play would use as historical opponents.",
    )
    promotions.add_argument(
        "--current-policy-spec",
        default=None,
        help="Policy spec to exclude from --opponent-pool-size, matching self-play current-policy filtering.",
    )
    promotions.add_argument("--json", action="store_true", help="Print the registry as formatted JSON.")
    promotions.set_defaults(func=_promotions)

    audit = subparsers.add_parser("audit", help="Audit a self-play run manifest for regression health.")
    audit.add_argument("path", type=Path, help="Self-play or neural self-play run directory or manifest.json path.")
    audit.add_argument("--min-latest-benchmark-win-rate", type=float, default=DEFAULT_MIN_BENCHMARK_WIN_RATE)
    audit.add_argument("--min-latest-benchmark-games", type=int, default=DEFAULT_MIN_BENCHMARK_GAMES)
    audit.add_argument("--max-latest-collection-capped-rate", type=float, default=DEFAULT_MAX_COLLECTION_CAPPED_RATE)
    audit.add_argument("--max-latest-benchmark-capped-rate", type=float, default=DEFAULT_MAX_BENCHMARK_CAPPED_RATE)
    audit.add_argument("--max-latest-average-decision-rounds", type=float, default=None)
    audit.add_argument("--max-latest-benchmark-average-decision-rounds", type=float, default=None)
    audit.add_argument("--max-benchmark-win-rate-drop", type=float, default=DEFAULT_MAX_BENCHMARK_WIN_RATE_DROP)
    audit.add_argument(
        "--max-consecutive-promotion-failures",
        type=int,
        default=DEFAULT_MAX_CONSECUTIVE_PROMOTION_FAILURES,
    )
    audit.add_argument("--allow-missing-benchmark", action="store_true", help="Do not fail solely because the latest benchmark is missing.")
    audit.add_argument("--require-latest-promotion", action="store_true", help="Fail unless the latest iteration recorded a promotion.")
    audit.add_argument("--json", action="store_true", help="Print the audit result as JSON.")
    audit.set_defaults(func=_audit)

    audit_calibrate = subparsers.add_parser("audit-calibrate", help="Suggest audit thresholds from an observed self-play run.")
    audit_calibrate.add_argument("path", type=Path, help="Self-play or neural self-play run directory or manifest.json path.")
    audit_calibrate.add_argument(
        "--margin",
        type=float,
        default=DEFAULT_AUDIT_CALIBRATION_MARGIN,
        help="Fractional safety margin applied to observed threshold suggestions.",
    )
    audit_calibrate.add_argument("--json", action="store_true", help="Print the calibration result as JSON.")
    audit_calibrate.set_defaults(func=_audit_calibrate)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


def _gate(args: argparse.Namespace) -> int:
    result = evaluate_promotion_gate(
        args.path,
        config=_gate_config_from_args(args),
    )
    if args.json:
        print(json.dumps(result.to_dict(), indent=2, sort_keys=True))
    else:
        _print_gate_result(result)
    return 0 if result.passed else 2


def _promote(args: argparse.Namespace) -> int:
    result = record_promotion(
        args.path,
        registry_path=args.registry,
        config=_gate_config_from_args(args),
        label=args.label,
        notes=args.notes,
        artifact_dir=args.artifact_dir,
        allow_duplicate=args.allow_duplicate,
    )
    if args.json:
        print(json.dumps(result.to_dict(), indent=2, sort_keys=True))
    else:
        _print_gate_result(result.gate_result)
        print(f"promotion_recorded: {'yes' if result.recorded else 'no'}")
        print(f"registry: {result.registry_path}")
        if result.entry is not None:
            print(f"promotion_sequence: {result.entry.sequence}")
            print(f"promoted_policy: {result.entry.policy_id or '-'}")
            print(f"promoted_checkpoint: {result.entry.checkpoint_path or '-'}")
            if result.entry.source_checkpoint_path is not None:
                print(f"source_checkpoint: {result.entry.source_checkpoint_path}")
    return 0 if result.recorded else 2


def _promotions(args: argparse.Namespace) -> int:
    if args.skip_checksum and args.require_checksum:
        raise ValueError("--skip-checksum cannot be combined with --require-checksum.")
    if args.verify_loadable and not args.verify:
        raise ValueError("--verify-loadable requires --verify.")
    if args.current_policy_spec is not None and args.opponent_pool_size is None:
        raise ValueError("--current-policy-spec requires --opponent-pool-size.")
    registry = load_promotion_registry(args.registry)
    opponent_pool = (
        registry.opponent_pool_policy_specs(
            max_historical_opponents=args.opponent_pool_size,
            current_policy_spec=args.current_policy_spec,
        )
        if args.opponent_pool_size is not None
        else None
    )
    verification = (
        verify_promotion_registry(
            args.registry,
            verify_checksums=not args.skip_checksum,
            require_checksums=args.require_checksum,
            verify_loadable=args.verify_loadable,
        )
        if args.verify
        else None
    )
    if args.json:
        payload = registry.to_dict()
        if opponent_pool is not None:
            payload["opponent_pool_policy_specs"] = list(opponent_pool)
        if verification is not None:
            payload["verification"] = verification.to_dict()
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0 if verification is None or verification.passed else 2
    print(f"registry: {registry.path}")
    print(f"promotions: {len(registry.entries)}")
    if registry.latest is not None:
        print(f"latest_policy: {registry.latest.policy_id or '-'}")
        print(f"latest_checkpoint: {registry.latest.checkpoint_path or '-'}")
    if registry.entries:
        print("entries:")
        for entry in registry.entries:
            label = f" label={entry.label}" if entry.label else ""
            source = f" source={entry.source_checkpoint_path}" if entry.source_checkpoint_path else ""
            print(
                f"- {entry.sequence}: policy={entry.policy_id or '-'} "
                f"checkpoint={entry.checkpoint_path or '-'} promoted_at={entry.promoted_at}{label}{source}"
            )
    if opponent_pool is not None:
        print("opponent_pool_policy_specs:")
        for spec in opponent_pool:
            print(f"- {spec}")
    if verification is not None:
        _print_registry_verification(verification)
    return 0 if verification is None or verification.passed else 2


def _audit(args: argparse.Namespace) -> int:
    result = audit_run(
        args.path,
        config=RunAuditConfig(
            min_latest_benchmark_win_rate=args.min_latest_benchmark_win_rate,
            min_latest_benchmark_games=args.min_latest_benchmark_games,
            max_latest_collection_capped_rate=args.max_latest_collection_capped_rate,
            max_latest_benchmark_capped_rate=args.max_latest_benchmark_capped_rate,
            max_latest_average_decision_rounds=args.max_latest_average_decision_rounds,
            max_latest_benchmark_average_decision_rounds=args.max_latest_benchmark_average_decision_rounds,
            max_benchmark_win_rate_drop=args.max_benchmark_win_rate_drop,
            max_consecutive_promotion_failures=args.max_consecutive_promotion_failures,
            require_benchmark=not args.allow_missing_benchmark,
            require_latest_promotion=args.require_latest_promotion,
        ),
    )
    if args.json:
        print(json.dumps(result.to_dict(), indent=2, sort_keys=True))
    else:
        _print_audit_result(result)
    return 0 if result.passed else 2


def _audit_calibrate(args: argparse.Namespace) -> int:
    result = calibrate_run_audit(args.path, margin=args.margin)
    if args.json:
        print(json.dumps(result.to_dict(), indent=2, sort_keys=True))
    else:
        _print_audit_calibration(result)
    return 0


def _add_gate_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--min-benchmark-win-rate", type=float, default=DEFAULT_MIN_BENCHMARK_WIN_RATE)
    parser.add_argument("--min-incumbent-win-rate", type=float, default=DEFAULT_MIN_INCUMBENT_WIN_RATE)
    parser.add_argument("--min-benchmark-games", type=int, default=DEFAULT_MIN_BENCHMARK_GAMES)
    parser.add_argument("--min-incumbent-games", type=int, default=DEFAULT_MIN_INCUMBENT_GAMES)
    parser.add_argument("--max-collection-capped-rate", type=float, default=DEFAULT_MAX_COLLECTION_CAPPED_RATE)
    parser.add_argument("--max-benchmark-capped-rate", type=float, default=DEFAULT_MAX_BENCHMARK_CAPPED_RATE)
    parser.add_argument("--max-incumbent-capped-rate", type=float, default=DEFAULT_MAX_INCUMBENT_CAPPED_RATE)
    parser.add_argument("--max-teacher-degradation-rate", type=float, default=DEFAULT_MAX_TEACHER_DEGRADATION_RATE)
    parser.add_argument(
        "--min-incumbent-win-rate-lower-bound",
        type=float,
        default=DEFAULT_MIN_INCUMBENT_WIN_RATE_LOWER_BOUND,
        help="Minimum Wilson lower confidence bound for candidate win rate against the incumbent.",
    )
    parser.add_argument(
        "--incumbent-confidence-z",
        type=float,
        default=DEFAULT_INCUMBENT_CONFIDENCE_Z,
        help="Z-score used for the incumbent Wilson lower-bound check. Default is one-sided 95%%.",
    )
    parser.add_argument(
        "--benchmark-opponent",
        action="append",
        default=None,
        help="Require and gate a specific benchmark opponent policy id. May be repeated. Defaults to every candidate benchmark opponent.",
    )
    parser.add_argument(
        "--opponent-win-rate",
        action="append",
        default=None,
        metavar="POLICY_ID=RATE",
        help="Override the win-rate floor for a specific benchmark opponent. May be repeated.",
    )
    parser.add_argument(
        "--incumbent-policy",
        default=None,
        help="Require direct benchmark evidence against this incumbent policy id and gate the candidate win rate against it.",
    )
    parser.add_argument("--allow-missing-benchmark", action="store_true", help="Do not fail solely because benchmark evidence is missing.")


def _gate_config_from_args(args: argparse.Namespace) -> PromotionGateConfig:
    incumbent_policy_id = args.incumbent_policy
    registry_path = getattr(args, "registry", None)
    if incumbent_policy_id is None and registry_path is not None:
        latest = load_promotion_registry(registry_path).latest
        incumbent_policy_id = latest.policy_id if latest is not None else None
    return PromotionGateConfig(
        min_benchmark_win_rate=args.min_benchmark_win_rate,
        min_incumbent_win_rate=args.min_incumbent_win_rate,
        min_benchmark_games=args.min_benchmark_games,
        min_incumbent_games=args.min_incumbent_games,
        max_collection_capped_rate=args.max_collection_capped_rate,
        max_benchmark_capped_rate=args.max_benchmark_capped_rate,
        max_incumbent_capped_rate=args.max_incumbent_capped_rate,
        max_teacher_degradation_rate=args.max_teacher_degradation_rate,
        min_incumbent_win_rate_lower_bound=args.min_incumbent_win_rate_lower_bound,
        incumbent_confidence_z=args.incumbent_confidence_z,
        require_benchmark=not args.allow_missing_benchmark,
        required_benchmark_opponents=tuple(args.benchmark_opponent or ()),
        opponent_min_win_rates=_parse_opponent_win_rates(tuple(args.opponent_win_rate or ())),
        incumbent_policy_id=incumbent_policy_id,
    )


def _print_audit_result(result) -> None:
    status = "PASS" if result.passed else "FAIL"
    print(f"status: {status}")
    print(f"source: {result.source_type}")
    print(f"manifest: {result.manifest_path}")
    print(f"iterations: {len(result.iterations)}")
    print(f"latest_iteration: {result.latest_iteration if result.latest_iteration is not None else '-'}")
    print(f"latest_benchmark_win_rate: {_format_optional_float(result.latest_benchmark_win_rate)}")
    print(f"best_benchmark_win_rate: {_format_optional_float(result.best_benchmark_win_rate)}")
    print(f"latest_collection_capped_rate: {_format_optional_float(result.latest_collection_capped_rate)}")
    print(f"latest_average_decision_rounds: {_format_optional_float(result.latest_average_decision_rounds)}")
    print(f"latest_benchmark_capped_rate: {_format_optional_float(result.latest_benchmark_capped_rate)}")
    print(
        "latest_benchmark_average_decision_rounds: "
        f"{_format_optional_float(result.latest_benchmark_average_decision_rounds)}"
    )
    print(f"consecutive_promotion_failures: {result.consecutive_promotion_failures}")
    if result.benchmark_regressions:
        print("benchmark_regressions:")
        for regression in result.benchmark_regressions:
            print(
                f"- {regression.opponent_policy_id}: "
                f"latest={regression.latest_win_rate:.3f} "
                f"previous_best={regression.best_previous_win_rate:.3f} "
                f"drop={regression.drop:.3f}"
            )
    print("checks:")
    for check in result.checks:
        check_status = "pass" if check.passed else "fail"
        print(f"- {check_status} {check.name}: observed={check.observed} threshold={check.threshold}")


def _print_audit_calibration(result) -> None:
    print(f"source: {result.source_type}")
    print(f"manifest: {result.manifest_path}")
    print(f"iterations: {result.iteration_count}")
    print(f"benchmark_iterations: {result.benchmark_iteration_count}")
    print(f"margin: {result.margin:.3f}")
    print("suggested_config:")
    for key, value in result.suggested_config().items():
        if isinstance(value, float):
            rendered = _format_optional_float(value)
        elif value is None:
            rendered = "-"
        else:
            rendered = str(value)
        print(f"- {key}: {rendered}")
    print("suggested_audit_flags:")
    flags = result.suggested_cli_flags()
    print(" ".join(flags) if flags else "-")
    if result.notes:
        print("notes:")
        for note in result.notes:
            print(f"- {note}")


def _print_registry_verification(result) -> None:
    status = "PASS" if result.passed else "FAIL"
    print(f"verification_status: {status}")
    print(f"checked_checkpoints: {result.checked_checkpoint_count}")
    print(f"verified_checksums: {result.verified_checksum_count}")
    print(f"verified_loadable: {result.verified_loadable_count}")
    print("verification_checks:")
    for check in result.checks:
        check_status = "pass" if check.passed else "fail"
        entry = "-" if check.entry_sequence is None else str(check.entry_sequence)
        print(
            f"- {check_status} {check.name}: "
            f"entry={entry} observed={check.observed} expected={check.expected}"
        )


def _print_gate_result(result) -> None:
    status = "PASS" if result.passed else "FAIL"
    print(f"status: {status}")
    print(f"mode: {result.gate_mode}")
    print(f"source: {result.source_type}")
    print(f"manifest: {result.manifest_path}")
    print(f"candidate_policy: {result.candidate_policy_id or '-'}")
    print(f"checkpoint: {result.checkpoint_path or '-'}")
    if result.source_iteration is not None:
        print(f"iteration: {result.source_iteration}")
    print(f"pooled_benchmark_win_rate: {_format_optional_float(result.benchmark_win_rate)}")
    print(f"collection_capped_rate: {_format_optional_float(result.collection_capped_rate)}")
    print(f"benchmark_capped_rate: {_format_optional_float(result.benchmark_capped_rate)}")
    if result.incumbent_policy_id is not None:
        print(f"incumbent_policy: {result.incumbent_policy_id}")
        print(f"incumbent_win_rate: {_format_optional_float(result.incumbent_win_rate)}")
        print(f"incumbent_win_rate_lower_bound: {_format_optional_float(result.incumbent_win_rate_lower_bound)}")
        print(f"incumbent_games: {result.incumbent_games}")
        print(f"incumbent_capped_rate: {_format_optional_float(result.incumbent_capped_rate)}")
    if result.teacher_degradation_rate is not None:
        print(f"teacher_degradation_rate: {_format_optional_float(result.teacher_degradation_rate)}")
    if result.benchmark_opponents:
        print("benchmark_opponents:")
        for opponent in result.benchmark_opponents:
            print(
                f"- {opponent.opponent_policy_id}: "
                f"win_rate={opponent.win_rate:.3f} "
                f"games={opponent.games} "
                f"capped_rate={opponent.capped_rate:.3f}"
            )
    print("checks:")
    for check in result.checks:
        check_status = "pass" if check.passed else "fail"
        print(f"- {check_status} {check.name}: observed={check.observed} threshold={check.threshold}")


def _format_optional_float(value: object) -> str:
    if value is None:
        return "-"
    return f"{float(value):.3f}"


def _parse_opponent_win_rates(values: tuple[str, ...]) -> dict[str, float]:
    parsed: dict[str, float] = {}
    for raw_value in values:
        opponent, separator, raw_threshold = raw_value.partition("=")
        opponent = opponent.strip()
        if not separator or not opponent:
            raise ValueError("--opponent-win-rate must use POLICY_ID=RATE.")
        try:
            threshold = float(raw_threshold)
        except ValueError as exc:
            raise ValueError("--opponent-win-rate RATE must be numeric.") from exc
        parsed[opponent] = threshold
    return parsed


if __name__ == "__main__":
    raise SystemExit(main())
