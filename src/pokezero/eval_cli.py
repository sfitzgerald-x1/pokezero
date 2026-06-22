"""Command-line promotion gates for experiment manifests."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from .evaluation import (
    DEFAULT_MIN_BENCHMARK_GAMES,
    PromotionGateConfig,
    evaluate_promotion_gate,
)
from .evaluation_profiles import EVALUATION_PROFILES, evaluation_profile
from .promotion import load_promotion_registry, record_promotion, verify_promotion_registry
from .run_audit import (
    DEFAULT_AUDIT_CALIBRATION_MARGIN,
    RunAuditConfig,
    audit_run,
    calibrate_run_audit,
    calibrate_run_audits,
    compare_run_manifests_with_threshold,
)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m pokezero.eval_cli")
    subparsers = parser.add_subparsers(dest="command", required=True)
    profile_choices = tuple(sorted(EVALUATION_PROFILES))

    gate = subparsers.add_parser("gate", help="Evaluate whether a manifest clears promotion thresholds.")
    gate.add_argument("path", type=Path, help="Experiment run directory or manifest.json path.")
    gate.add_argument("--registry", type=Path, default=None, help="Optional promotion registry used as the default incumbent source.")
    gate.add_argument("--profile", choices=profile_choices, default="default", help="Named threshold profile used as defaults for gate checks.")
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
    promote.add_argument("--profile", choices=profile_choices, default="default", help="Named threshold profile used as defaults for gate checks.")
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
        help=(
            "Preview the latest N promoted policy specs that self-play would use as historical opponents. "
            "By default, excludes the latest promoted policy as the assumed current collector."
        ),
    )
    promotions.add_argument(
        "--require-opponent-pool-size",
        type=int,
        default=None,
        help=(
            "With --opponent-pool-size, return non-zero unless at least this many promoted "
            "historical opponents are selected."
        ),
    )
    promotions.add_argument(
        "--current-policy-spec",
        default=None,
        help="Policy spec to exclude from --opponent-pool-size instead of the latest promoted policy.",
    )
    promotions.add_argument("--json", action="store_true", help="Print the registry as formatted JSON.")
    promotions.set_defaults(func=_promotions)

    profiles = subparsers.add_parser("profiles", help="Print named gate/audit threshold profiles.")
    profiles.add_argument("--json", action="store_true", help="Print profiles as formatted JSON.")
    profiles.set_defaults(func=_profiles)

    audit = subparsers.add_parser("audit", help="Audit a self-play run manifest for regression health.")
    audit.add_argument("path", type=Path, help="Self-play or neural self-play run directory or manifest.json path.")
    audit.add_argument("--profile", choices=profile_choices, default="default", help="Named threshold profile used as defaults for audit checks.")
    audit.add_argument("--min-latest-benchmark-win-rate", type=float, default=None)
    audit.add_argument("--min-latest-benchmark-games", type=int, default=None)
    audit.add_argument("--max-latest-collection-capped-rate", type=float, default=None)
    audit.add_argument("--max-latest-benchmark-capped-rate", type=float, default=None)
    audit.add_argument("--max-latest-average-decision-rounds", type=float, default=None)
    audit.add_argument("--max-latest-benchmark-average-decision-rounds", type=float, default=None)
    audit.add_argument("--max-latest-process-peak-rss-mb", type=float, default=None)
    audit.add_argument("--max-benchmark-win-rate-drop", type=float, default=None)
    audit.add_argument(
        "--max-consecutive-promotion-failures",
        type=int,
        default=None,
    )
    _add_benchmark_requirement_arguments(
        audit,
        missing_help="Do not fail solely because the latest benchmark is missing.",
    )
    latest_promotion_group = audit.add_mutually_exclusive_group()
    latest_promotion_group.add_argument(
        "--require-latest-promotion",
        dest="require_latest_promotion",
        action="store_true",
        default=None,
        help="Fail unless the latest iteration recorded a promotion.",
    )
    latest_promotion_group.add_argument(
        "--allow-missing-latest-promotion",
        dest="require_latest_promotion",
        action="store_false",
        default=None,
        help="Do not fail solely because the latest iteration did not record a promotion.",
    )
    benchmark_opponent_group = audit.add_mutually_exclusive_group()
    benchmark_opponent_group.add_argument(
        "--require-benchmark-opponents",
        dest="require_benchmark_opponent_coverage",
        action="store_true",
        default=None,
        help="Fail when the latest benchmark omits fixed baseline opponents seen in prior benchmark evidence.",
    )
    benchmark_opponent_group.add_argument(
        "--allow-missing-benchmark-opponents",
        dest="require_benchmark_opponent_coverage",
        action="store_false",
        default=None,
        help="Do not fail when the latest benchmark omits fixed baseline opponents seen in prior benchmark evidence.",
    )
    audit.add_argument("--json", action="store_true", help="Print the audit result as JSON.")
    audit.set_defaults(func=_audit)

    audit_calibrate = subparsers.add_parser("audit-calibrate", help="Suggest audit thresholds from observed self-play runs.")
    audit_calibrate.add_argument(
        "paths",
        type=Path,
        nargs="+",
        help="Self-play or neural self-play run directories or manifest.json paths.",
    )
    audit_calibrate.add_argument(
        "--margin",
        type=float,
        default=DEFAULT_AUDIT_CALIBRATION_MARGIN,
        help="Fractional safety margin applied to observed threshold suggestions.",
    )
    audit_calibrate.add_argument(
        "--aggregate-mode",
        choices=("median", "envelope"),
        default="median",
        help=(
            "How multiple run calibrations are combined. median resists noisy pilots; "
            "envelope keeps every supplied pilot passable."
        ),
    )
    audit_calibrate.add_argument("--json", action="store_true", help="Print the calibration result as JSON.")
    audit_calibrate.set_defaults(func=_audit_calibrate)

    compare = subparsers.add_parser("compare", help="Compare self-play run manifests side by side.")
    compare.add_argument("paths", type=Path, nargs="+", help="Self-play or neural self-play run directories or manifest.json paths.")
    compare.add_argument(
        "--min-benchmark-games",
        type=int,
        default=DEFAULT_MIN_BENCHMARK_GAMES,
        help="Minimum benchmark games required for a run to be eligible for best-run labels.",
    )
    compare.add_argument(
        "--audit-profile",
        choices=profile_choices,
        default=None,
        help="Also evaluate each compared run against this named audit profile and include pass/fail status.",
    )
    compare.add_argument(
        "--fail-on-audit",
        action="store_true",
        help="With --audit-profile, return non-zero when any compared run fails the selected audit profile.",
    )
    compare.add_argument("--json", action="store_true", help="Print the comparison result as JSON.")
    compare.set_defaults(func=_compare)
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
    if args.require_opponent_pool_size is not None and args.opponent_pool_size is None:
        raise ValueError("--require-opponent-pool-size requires --opponent-pool-size.")
    if args.require_opponent_pool_size is not None and args.require_opponent_pool_size < 0:
        raise ValueError("--require-opponent-pool-size must be non-negative.")
    if (
        args.require_opponent_pool_size is not None
        and args.opponent_pool_size is not None
        and args.require_opponent_pool_size > args.opponent_pool_size
    ):
        raise ValueError("--require-opponent-pool-size cannot exceed --opponent-pool-size.")
    registry = load_promotion_registry(args.registry)
    preview_current_policy_spec = args.current_policy_spec
    if preview_current_policy_spec is None and args.opponent_pool_size is not None and registry.latest is not None:
        preview_current_policy_spec = registry.latest_selection_checkpoint_policy_spec()
    available_opponent_pool = (
        registry.opponent_pool_policy_specs(
            max_historical_opponents=len(registry.entries),
            current_policy_spec=preview_current_policy_spec,
        )
        if args.opponent_pool_size is not None
        else None
    )
    opponent_pool = (
        registry.opponent_pool_policy_specs(
            max_historical_opponents=args.opponent_pool_size,
            current_policy_spec=preview_current_policy_spec,
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
    entry_statuses = _promotion_entry_statuses(
        registry,
        verification=verification,
        opponent_pool=opponent_pool,
    )
    if args.json:
        payload = registry.to_dict()
        payload["entry_statuses"] = entry_statuses
        if opponent_pool is not None:
            payload["opponent_pool_policy_specs"] = list(opponent_pool)
            payload["opponent_pool_excluded_current_policy_spec"] = preview_current_policy_spec
            payload["opponent_pool_verified"] = verification.passed if verification is not None else None
            payload["opponent_pool_requested_size"] = args.opponent_pool_size
            payload["opponent_pool_selected_size"] = len(opponent_pool)
            payload["opponent_pool_available_size"] = (
                len(available_opponent_pool) if available_opponent_pool is not None else None
            )
            payload["opponent_pool_required_size"] = args.require_opponent_pool_size
            payload["opponent_pool_requirement_passed"] = _opponent_pool_requirement_passed(
                opponent_pool,
                required_size=args.require_opponent_pool_size,
            )
        if verification is not None:
            payload["verification"] = verification.to_dict()
        print(json.dumps(payload, indent=2, sort_keys=True))
        return _promotions_exit_code(
            verification=verification,
            opponent_pool=opponent_pool,
            required_opponent_pool_size=args.require_opponent_pool_size,
        )
    print(f"registry: {registry.path}")
    print(f"promotions: {len(registry.entries)}")
    if registry.latest is not None:
        print(f"latest_policy: {registry.latest.policy_id or '-'}")
        print(f"latest_checkpoint: {registry.latest.checkpoint_path or '-'}")
    if registry.entries:
        print("entries:")
        status_by_sequence = {status["sequence"]: status for status in entry_statuses}
        for entry in registry.entries:
            label = f" label={entry.label}" if entry.label else ""
            source = f" source={entry.source_checkpoint_path}" if entry.source_checkpoint_path else ""
            status = status_by_sequence[entry.sequence]
            selected_as = ",".join(status["selected_as"]) if status["selected_as"] else "-"
            print(
                f"- {entry.sequence}: policy={entry.policy_id or '-'} "
                f"checkpoint={entry.checkpoint_path or '-'} promoted_at={entry.promoted_at} "
                f"status={status['verification_status']} selected={selected_as} "
                f"path={status['checkpoint_path_present']} exists={status['checkpoint_exists']} checksum={status['checksum']} "
                f"loadable={status['loadable']}{label}{source}"
            )
    if opponent_pool is not None:
        print(f"opponent_pool_excluded_current_policy_spec: {preview_current_policy_spec or '-'}")
        print(f"opponent_pool_requested_size: {args.opponent_pool_size}")
        print(f"opponent_pool_selected_size: {len(opponent_pool)}")
        if available_opponent_pool is not None:
            print(f"opponent_pool_available_size: {len(available_opponent_pool)}")
        if args.require_opponent_pool_size is not None:
            pool_status = (
                "PASS"
                if _opponent_pool_requirement_passed(
                    opponent_pool,
                    required_size=args.require_opponent_pool_size,
                )
                else "FAIL"
            )
            print(f"opponent_pool_required_size: {args.require_opponent_pool_size}")
            print(f"opponent_pool_requirement: {pool_status}")
        print("opponent_pool_policy_specs:")
        for spec in opponent_pool:
            print(f"- {spec}")
        if verification is None:
            print("note: pass --verify to confirm the previewed registry is selectable by runtime.")
    if verification is not None:
        _print_registry_verification(verification)
    return _promotions_exit_code(
        verification=verification,
        opponent_pool=opponent_pool,
        required_opponent_pool_size=args.require_opponent_pool_size,
    )


def _promotions_exit_code(
    *,
    verification,
    opponent_pool,
    required_opponent_pool_size: int | None,
) -> int:
    if verification is not None and not verification.passed:
        return 2
    if not _opponent_pool_requirement_passed(opponent_pool, required_size=required_opponent_pool_size):
        return 2
    return 0


def _opponent_pool_requirement_passed(
    opponent_pool,
    *,
    required_size: int | None,
) -> bool:
    return required_size is None or (opponent_pool is not None and len(opponent_pool) >= required_size)


def _promotion_entry_statuses(
    registry,
    *,
    verification,
    opponent_pool,
) -> list[dict[str, object]]:
    checks_by_sequence: dict[int, list[object]] = {}
    if verification is not None:
        for check in verification.checks:
            if check.entry_sequence is None:
                continue
            checks_by_sequence.setdefault(check.entry_sequence, []).append(check)
    opponent_pool_sequences = _opponent_pool_entry_sequences(registry, opponent_pool)
    latest_sequence = registry.latest.sequence if registry.latest is not None else None
    statuses: list[dict[str, object]] = []
    for entry in registry.entries:
        checks = tuple(checks_by_sequence.get(entry.sequence, ()))
        selected_as = []
        selection_checkpoint_policy_spec = registry.selection_checkpoint_policy_spec_for_entry(entry)
        if entry.sequence == latest_sequence:
            selected_as.append("latest")
        if entry.sequence in opponent_pool_sequences:
            selected_as.append("opponent_pool")
        failed_checks = [check.name for check in checks if not check.passed]
        checkpoint_path_present = _checkpoint_path_present_status(
            checks,
            verification_enabled=verification is not None,
        )
        checkpoint_exists = _checkpoint_exists_status(
            checks,
            verification_enabled=verification is not None,
        )
        checksum = _verification_check_status(
            checks,
            ("checkpoint_sha256", "checkpoint_sha256_present"),
            verification_enabled=verification is not None,
        )
        loadable = _verification_check_status(
            checks,
            ("checkpoint_policy_loadable",),
            verification_enabled=verification is not None,
        )
        policy_id_matches = _verification_check_status(
            checks,
            ("checkpoint_policy_id",),
            verification_enabled=verification is not None,
        )
        statuses.append(
            {
                "sequence": entry.sequence,
                "policy_id": entry.policy_id,
                "label": entry.label,
                "checkpoint_path": entry.checkpoint_path,
                "checkpoint_policy_spec": entry.checkpoint_policy_spec,
                "selection_checkpoint_policy_spec": selection_checkpoint_policy_spec,
                "source_checkpoint_path": entry.source_checkpoint_path,
                "source_type": entry.source_type,
                "source_iteration": entry.source_iteration,
                "promoted_at": entry.promoted_at,
                "selected_as": selected_as,
                "verification_status": _entry_verification_status(
                    checks,
                    verification_enabled=verification is not None,
                    detail_statuses=(
                        checkpoint_path_present,
                        checkpoint_exists,
                        checksum,
                        loadable,
                        policy_id_matches,
                    ),
                ),
                "checkpoint_path_present": checkpoint_path_present,
                "checkpoint_exists": checkpoint_exists,
                "checksum": checksum,
                "loadable": loadable,
                "policy_id_matches": policy_id_matches,
                "failed_checks": failed_checks,
            }
        )
    return statuses


def _opponent_pool_entry_sequences(registry, opponent_pool) -> set[int]:
    if opponent_pool is None:
        return set()
    wanted_specs = list(reversed(opponent_pool))
    selected_sequences: set[int] = set()
    wanted_index = 0
    for entry in reversed(registry.entries):
        if wanted_index >= len(wanted_specs):
            break
        if registry.selection_checkpoint_policy_spec_for_entry(entry) == wanted_specs[wanted_index]:
            selected_sequences.add(entry.sequence)
            wanted_index += 1
    return selected_sequences


def _entry_verification_status(
    checks: tuple[object, ...],
    *,
    verification_enabled: bool,
    detail_statuses: tuple[str, ...],
) -> str:
    if not verification_enabled:
        return "not_verified"
    if any(not check.passed for check in checks):
        return "fail"
    if any(status == "not_checked" for status in detail_statuses):
        return "partial"
    return "pass"


def _verification_check_status(
    checks: tuple[object, ...],
    names: tuple[str, ...],
    *,
    verification_enabled: bool,
) -> str:
    if not verification_enabled:
        return "not_verified"
    for check in checks:
        if check.name in names:
            return "pass" if check.passed else "fail"
    return "not_checked"


def _checkpoint_path_present_status(checks: tuple[object, ...], *, verification_enabled: bool) -> str:
    if not verification_enabled:
        return "not_verified"
    for check in checks:
        if check.name == "checkpoint_path_present":
            return "pass" if check.passed else "fail"
    for check in checks:
        if check.name == "checkpoint_exists":
            return "pass"
    return "not_checked"


def _checkpoint_exists_status(checks: tuple[object, ...], *, verification_enabled: bool) -> str:
    if not verification_enabled:
        return "not_verified"
    checkpoint_path_present = _checkpoint_path_present_status(
        checks,
        verification_enabled=verification_enabled,
    )
    if checkpoint_path_present == "fail":
        return "fail"
    return _verification_check_status(
        checks,
        ("checkpoint_exists",),
        verification_enabled=verification_enabled,
    )


def _audit(args: argparse.Namespace) -> int:
    result = audit_run(
        args.path,
        config=_audit_config_from_args(args),
    )
    if args.json:
        print(json.dumps(result.to_dict(), indent=2, sort_keys=True))
    else:
        _print_audit_result(result)
    return 0 if result.passed else 2


def _profiles(args: argparse.Namespace) -> int:
    profiles = tuple(EVALUATION_PROFILES[name] for name in sorted(EVALUATION_PROFILES))
    if args.json:
        print(json.dumps({"profiles": [profile.to_dict() for profile in profiles]}, indent=2, sort_keys=True))
        return 0
    print("profiles:")
    for profile in profiles:
        print(f"- {profile.name}: {profile.description}")
        print(
            "  gate: "
            f"min_win_rate={profile.gate_config.min_benchmark_win_rate:.3f} "
            f"min_games={profile.gate_config.min_benchmark_games} "
            f"max_collection_capped={profile.gate_config.max_collection_capped_rate:.3f} "
            f"max_benchmark_capped={profile.gate_config.max_benchmark_capped_rate:.3f} "
            f"require_benchmark={profile.gate_config.require_benchmark}"
        )
        print(
            "  audit: "
            f"min_latest_win_rate={profile.audit_config.min_latest_benchmark_win_rate:.3f} "
            f"min_latest_games={profile.audit_config.min_latest_benchmark_games} "
            f"max_latest_collection_capped={profile.audit_config.max_latest_collection_capped_rate:.3f} "
            f"max_latest_benchmark_capped={profile.audit_config.max_latest_benchmark_capped_rate:.3f} "
            f"max_latest_avg_dec={_format_optional_float(profile.audit_config.max_latest_average_decision_rounds)} "
            "max_latest_benchmark_avg_dec="
            f"{_format_optional_float(profile.audit_config.max_latest_benchmark_average_decision_rounds)} "
            f"max_latest_rss_mb={_format_optional_one_decimal(profile.audit_config.max_latest_process_peak_rss_mb)} "
            f"max_win_rate_drop={profile.audit_config.max_benchmark_win_rate_drop:.3f} "
            f"max_promotion_failures={profile.audit_config.max_consecutive_promotion_failures} "
            f"require_benchmark={profile.audit_config.require_benchmark} "
            f"require_benchmark_opponents={profile.audit_config.require_benchmark_opponent_coverage} "
            f"require_latest_promotion={profile.audit_config.require_latest_promotion}"
        )
    return 0


def _audit_calibrate(args: argparse.Namespace) -> int:
    result = (
        calibrate_run_audit(args.paths[0], margin=args.margin)
        if len(args.paths) == 1
        else calibrate_run_audits(args.paths, margin=args.margin, aggregate_mode=args.aggregate_mode)
    )
    if args.json:
        print(json.dumps(result.to_dict(), indent=2, sort_keys=True))
    else:
        _print_audit_calibration(result)
    return 0


def _compare(args: argparse.Namespace) -> int:
    if args.fail_on_audit and args.audit_profile is None:
        raise ValueError("--fail-on-audit requires --audit-profile.")
    audit_profile = evaluation_profile(args.audit_profile) if args.audit_profile is not None else None
    result = compare_run_manifests_with_threshold(
        args.paths,
        min_benchmark_games=args.min_benchmark_games,
        audit_config=audit_profile.audit_config if audit_profile is not None else None,
        audit_profile=audit_profile.name if audit_profile is not None else None,
    )
    if args.json:
        print(json.dumps(result.to_dict(), indent=2, sort_keys=True))
    else:
        _print_run_comparison(result)
    return 2 if result.errors or (args.fail_on_audit and result.audit_failed) else 0


def _add_gate_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--min-benchmark-win-rate", type=float, default=None)
    parser.add_argument("--min-incumbent-win-rate", type=float, default=None)
    parser.add_argument("--min-benchmark-games", type=int, default=None)
    parser.add_argument("--min-incumbent-games", type=int, default=None)
    parser.add_argument("--max-collection-capped-rate", type=float, default=None)
    parser.add_argument("--max-benchmark-capped-rate", type=float, default=None)
    parser.add_argument("--max-incumbent-capped-rate", type=float, default=None)
    parser.add_argument("--max-teacher-degradation-rate", type=float, default=None)
    parser.add_argument(
        "--min-incumbent-win-rate-lower-bound",
        type=float,
        default=None,
        help="Minimum Wilson lower confidence bound for candidate win rate against the incumbent.",
    )
    parser.add_argument(
        "--incumbent-confidence-z",
        type=float,
        default=None,
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
    _add_benchmark_requirement_arguments(
        parser,
        missing_help="Do not fail solely because benchmark evidence is missing.",
    )


def _add_benchmark_requirement_arguments(parser: argparse.ArgumentParser, *, missing_help: str) -> None:
    benchmark_group = parser.add_mutually_exclusive_group()
    benchmark_group.add_argument(
        "--require-benchmark",
        dest="require_benchmark",
        action="store_true",
        default=None,
        help="Fail when required benchmark evidence is missing.",
    )
    benchmark_group.add_argument(
        "--allow-missing-benchmark",
        dest="require_benchmark",
        action="store_false",
        default=None,
        help=missing_help,
    )


def _gate_config_from_args(args: argparse.Namespace) -> PromotionGateConfig:
    profile_config = evaluation_profile(getattr(args, "profile", None)).gate_config
    incumbent_policy_id = args.incumbent_policy
    registry_path = getattr(args, "registry", None)
    if incumbent_policy_id is None and registry_path is not None:
        latest = load_promotion_registry(registry_path).latest
        incumbent_policy_id = latest.policy_id if latest is not None else None
    return PromotionGateConfig(
        min_benchmark_win_rate=_arg_or_default(args.min_benchmark_win_rate, profile_config.min_benchmark_win_rate),
        min_incumbent_win_rate=_arg_or_default(args.min_incumbent_win_rate, profile_config.min_incumbent_win_rate),
        min_benchmark_games=_arg_or_default(args.min_benchmark_games, profile_config.min_benchmark_games),
        min_incumbent_games=_arg_or_default(args.min_incumbent_games, profile_config.min_incumbent_games),
        max_collection_capped_rate=_arg_or_default(args.max_collection_capped_rate, profile_config.max_collection_capped_rate),
        max_benchmark_capped_rate=_arg_or_default(args.max_benchmark_capped_rate, profile_config.max_benchmark_capped_rate),
        max_incumbent_capped_rate=_arg_or_default(args.max_incumbent_capped_rate, profile_config.max_incumbent_capped_rate),
        max_teacher_degradation_rate=_arg_or_default(args.max_teacher_degradation_rate, profile_config.max_teacher_degradation_rate),
        min_incumbent_win_rate_lower_bound=_arg_or_default(
            args.min_incumbent_win_rate_lower_bound,
            profile_config.min_incumbent_win_rate_lower_bound,
        ),
        incumbent_confidence_z=_arg_or_default(args.incumbent_confidence_z, profile_config.incumbent_confidence_z),
        require_benchmark=_arg_or_default(args.require_benchmark, profile_config.require_benchmark),
        required_benchmark_opponents=tuple(args.benchmark_opponent or ()),
        opponent_min_win_rates=_parse_opponent_win_rates(tuple(args.opponent_win_rate or ())),
        incumbent_policy_id=incumbent_policy_id,
    )


def _audit_config_from_args(args: argparse.Namespace) -> RunAuditConfig:
    profile_config = evaluation_profile(args.profile).audit_config
    return RunAuditConfig(
        min_latest_benchmark_win_rate=_arg_or_default(
            args.min_latest_benchmark_win_rate,
            profile_config.min_latest_benchmark_win_rate,
        ),
        min_latest_benchmark_games=_arg_or_default(
            args.min_latest_benchmark_games,
            profile_config.min_latest_benchmark_games,
        ),
        max_latest_collection_capped_rate=_arg_or_default(
            args.max_latest_collection_capped_rate,
            profile_config.max_latest_collection_capped_rate,
        ),
        max_latest_benchmark_capped_rate=_arg_or_default(
            args.max_latest_benchmark_capped_rate,
            profile_config.max_latest_benchmark_capped_rate,
        ),
        max_latest_average_decision_rounds=_arg_or_default(
            args.max_latest_average_decision_rounds,
            profile_config.max_latest_average_decision_rounds,
        ),
        max_latest_benchmark_average_decision_rounds=_arg_or_default(
            args.max_latest_benchmark_average_decision_rounds,
            profile_config.max_latest_benchmark_average_decision_rounds,
        ),
        max_latest_process_peak_rss_mb=_arg_or_default(
            args.max_latest_process_peak_rss_mb,
            profile_config.max_latest_process_peak_rss_mb,
        ),
        max_benchmark_win_rate_drop=_arg_or_default(
            args.max_benchmark_win_rate_drop,
            profile_config.max_benchmark_win_rate_drop,
        ),
        max_consecutive_promotion_failures=_arg_or_default(
            args.max_consecutive_promotion_failures,
            profile_config.max_consecutive_promotion_failures,
        ),
        require_benchmark=_arg_or_default(args.require_benchmark, profile_config.require_benchmark),
        require_latest_promotion=_arg_or_default(
            args.require_latest_promotion,
            profile_config.require_latest_promotion,
        ),
        require_benchmark_opponent_coverage=_arg_or_default(
            args.require_benchmark_opponent_coverage,
            profile_config.require_benchmark_opponent_coverage,
        ),
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
    print(f"latest_process_peak_rss_mb: {_format_optional_one_decimal(result.latest_process_peak_rss_mb)}")
    if result.missing_latest_benchmark_opponents:
        print("missing_latest_benchmark_opponents:")
        for opponent in result.missing_latest_benchmark_opponents:
            print(f"- {opponent}")
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
    if hasattr(result, "manifest_path"):
        print(f"manifest: {result.manifest_path}")
    else:
        print(f"runs: {result.run_count}")
        print(f"aggregate_mode: {result.aggregate_mode}")
        print("manifests:")
        for path in result.paths:
            print(f"- {path}")
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


def _print_run_comparison(result) -> None:
    print(f"runs: {len(result.entries)}")
    print(f"errors: {len(result.errors)}")
    print(f"min_benchmark_games_for_best: {result.min_benchmark_games}")
    if result.audit_profile is not None:
        print(f"audit_profile: {result.audit_profile}")
    latest = result.best_latest_benchmark_entry
    historical = result.best_historical_benchmark_entry
    print(f"best_latest_benchmark: {latest.label if latest is not None else '-'}")
    print(f"best_historical_benchmark: {historical.label if historical is not None else '-'}")
    print("")
    audit_header = f"{'audit':>6} " if result.audit_profile is not None else ""
    header = (
        f"{'run':<24} {'src':<15} {'iter':>4} {audit_header}{'bench_wr':>8} {'best_wr':>8} {'bench_g':>7} "
        f"{'coll_cap':>8} {'bench_cap':>9} {'coll_gph':>8} {'bench_gph':>9} {'rss_hi_mb':>9} "
        f"{'avg_dec':>8} {'bench_dec':>9} {'promo':>6} {'adv':>6} checkpoint"
    )
    print(header)
    print("-" * len(header))
    for entry in result.entries:
        print(
            f"{entry.label:<24.24} "
            f"{entry.source_type:<15.15} "
            f"{(entry.latest_iteration if entry.latest_iteration is not None else 0):4d} "
            f"{(f'{_format_optional_bool(entry.audit_passed):>6} ') if result.audit_profile is not None else ''}"
            f"{_format_optional_float(entry.latest_benchmark_win_rate):>8} "
            f"{_format_optional_float(entry.best_benchmark_win_rate):>8} "
            f"{entry.latest_benchmark_games:7d} "
            f"{_format_optional_float(entry.latest_collection_capped_rate):>8} "
            f"{_format_optional_float(entry.latest_benchmark_capped_rate):>9} "
            f"{_format_optional_whole_number(entry.latest_collection_games_per_hour):>8} "
            f"{_format_optional_whole_number(entry.latest_benchmark_games_per_hour):>9} "
            f"{_format_optional_one_decimal(entry.latest_process_peak_rss_mb):>9} "
            f"{_format_optional_float(entry.latest_average_decision_rounds):>8} "
            f"{_format_optional_float(entry.latest_benchmark_average_decision_rounds):>9} "
            f"{_format_optional_bool(entry.latest_promotion_recorded):>6} "
            f"{_format_optional_bool(entry.latest_advancement_recorded):>6} "
            f"{entry.latest_checkpoint_path or '-'}"
        )
    if result.errors:
        print("")
        print("errors:")
        for error in result.errors:
            print(f"- {error.label}: {error.error}")
    if result.audit_profile is not None:
        failed_entries = tuple(entry for entry in result.entries if entry.audit_passed is False)
        if failed_entries:
            print("")
            print("audit_failures:")
            for entry in failed_entries:
                failed = ", ".join(entry.audit_failed_checks) if entry.audit_failed_checks else "unknown"
                print(f"- {entry.label}: {failed}")


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


def _format_optional_whole_number(value: object) -> str:
    if value is None:
        return "-"
    return f"{float(value):.0f}"


def _format_optional_one_decimal(value: object) -> str:
    if value is None:
        return "-"
    return f"{float(value):.1f}"


def _format_optional_bool(value: bool | None) -> str:
    if value is None:
        return "-"
    return "yes" if value else "no"


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


def _arg_or_default(value, default):
    return default if value is None else value


if __name__ == "__main__":
    raise SystemExit(main())
