"""Command-line promotion gates for experiment manifests."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import glob
import json
from pathlib import Path
import shutil
import shlex
import subprocess
import sys
import time
from typing import Iterable, Mapping

from .evaluation import (
    DEFAULT_MIN_BENCHMARK_GAMES,
    PromotionGateConfig,
    evaluate_promotion_gate,
)
from .evaluation_profiles import EVALUATION_PROFILES, evaluation_profile
from .promotion import load_promotion_registry, record_promotion, verify_promotion_registry
from .run_audit import (
    DEFAULT_AUDIT_CALIBRATION_MARGIN,
    RUN_AUDIT_CONFIG_SCHEMA_VERSION,
    RunAuditConfig,
    audit_run,
    calibrate_run_audit,
    calibrate_run_audits,
    compare_run_manifests_with_threshold,
    load_run_audit_config,
    run_audit_config_from_dict,
    run_audit_config_payload,
    run_audit_config_to_dict,
)
from .source_metadata import collect_source_metadata


CPU_SMOKE_RUN_SUMMARY_SCHEMA_VERSION = "pokezero.cpu_smoke_run_summary.v1"
CPU_PILOT_SUITE_SUMMARY_SCHEMA_VERSION = "pokezero.cpu_pilot_suite_summary.v1"
CPU_SMOKE_SEED_BAND_SPACING = 1_000_000
OPPONENT_POOL_SNAPSHOT_SCHEMA_VERSION = "pokezero.opponent_pool_snapshot.v1"
PROMOTION_RETENTION_PLAN_SCHEMA_VERSION = "pokezero.promotion_retention_plan.v1"
PROMOTION_RETENTION_APPLY_SCHEMA_VERSION = "pokezero.promotion_retention_apply.v1"


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
        "--verify-opponent-pool-only",
        action="store_true",
        help=(
            "With --verify and --opponent-pool-size, make the command exit status depend on "
            "the selected opponent pool and excluded current policy instead of stale entries outside that preview."
        ),
    )
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
    promotions.add_argument(
        "--write-opponent-pool",
        type=Path,
        default=None,
        help=(
            "With --opponent-pool-size, write a compact JSON snapshot of the selected promoted "
            "opponent pool and preflight status."
        ),
    )
    promotions.add_argument(
        "--lifecycle",
        action="store_true",
        help=(
            "Report compact lifecycle counts for promoted checkpoints, including latest, "
            "selected opponent-pool, stale, unselectable, and verification-status buckets."
        ),
    )
    promotions.add_argument(
        "--retention-plan",
        action="store_true",
        help=(
            "With --opponent-pool-size, print a non-destructive retention preview that marks "
            "selected, current, stale, and manual-review promotion entries."
        ),
    )
    promotions.add_argument(
        "--apply-retention-plan",
        action="store_true",
        help=(
            "With --retention-plan, preview archiving verified cleanup candidates. This is a dry run "
            "unless --retention-apply-confirm archive is also passed."
        ),
    )
    promotions.add_argument(
        "--retention-apply-confirm",
        choices=("archive",),
        default=None,
        help=(
            "Actually apply --apply-retention-plan by moving cleanup candidates into the retention "
            "archive and updating registry checkpoint paths. Omit for dry-run output."
        ),
    )
    promotions.add_argument(
        "--retention-archive-dir",
        type=Path,
        default=None,
        help=(
            "Archive directory for --apply-retention-plan. Defaults to a timestamped directory under "
            "<registry-dir>/retention-archive."
        ),
    )
    promotions.add_argument("--json", action="store_true", help="Print the registry as formatted JSON.")
    promotions.set_defaults(func=_promotions)

    profiles = subparsers.add_parser("profiles", help="Print named gate/audit threshold profiles.")
    profiles.add_argument("--json", action="store_true", help="Print profiles as formatted JSON.")
    profiles.set_defaults(func=_profiles)

    audit = subparsers.add_parser("audit", help="Audit a self-play run manifest for regression health.")
    audit.add_argument("path", type=Path, help="Self-play or neural self-play run directory or manifest.json path.")
    audit.add_argument("--profile", choices=profile_choices, default=None, help="Named threshold profile used as defaults for audit checks.")
    audit.add_argument(
        "--audit-config",
        type=Path,
        default=None,
        help="Versioned run-audit config JSON used as defaults. Explicit threshold flags override this file.",
    )
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

    audit_config_report = subparsers.add_parser(
        "audit-config-report",
        help="Inspect a reusable run-audit config and optionally replay it against manifests.",
    )
    audit_config_report.add_argument("audit_config", type=Path, help="Versioned run-audit config JSON path.")
    audit_config_report.add_argument(
        "paths",
        type=Path,
        nargs="*",
        help="Optional self-play or neural self-play run directories or manifest.json paths to audit with this config.",
    )
    audit_config_report.add_argument(
        "--manifest-glob",
        action="append",
        default=None,
        help=(
            "Glob pattern for run directories or manifest.json files to audit with this config. "
            "May be repeated and is expanded in sorted order."
        ),
    )
    audit_config_report.add_argument(
        "--require-source",
        action="store_true",
        help="Return non-zero unless the config includes source provenance metadata.",
    )
    audit_config_report.add_argument(
        "--require-calibration",
        action="store_true",
        help="Return non-zero unless the config includes calibration metadata.",
    )
    audit_config_report.add_argument("--json", action="store_true", help="Print the config report as JSON.")
    audit_config_report.set_defaults(func=_audit_config_report)

    audit_calibrate = subparsers.add_parser("audit-calibrate", help="Suggest audit thresholds from observed self-play runs.")
    audit_calibrate.add_argument(
        "paths",
        type=Path,
        nargs="*",
        help="Self-play or neural self-play run directories or manifest.json paths.",
    )
    audit_calibrate.add_argument(
        "--manifest-glob",
        action="append",
        default=None,
        help=(
            "Glob pattern for run directories or manifest.json files to include in calibration. "
            "May be repeated and is expanded in sorted order."
        ),
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
    audit_calibrate.add_argument(
        "--require-run-count",
        type=int,
        default=0,
        help="Return non-zero unless at least this many pilot runs contributed to the calibration.",
    )
    audit_calibrate.add_argument(
        "--require-benchmark-iterations",
        type=int,
        default=0,
        help="Return non-zero unless at least this many benchmark iterations contributed to the calibration.",
    )
    audit_calibrate.add_argument(
        "--require-min-benchmark-games",
        type=int,
        default=0,
        help="Return non-zero unless calibrated benchmark iterations used at least this many games.",
    )
    audit_calibrate.add_argument(
        "--compare-profile",
        choices=profile_choices,
        default=None,
        help="Also audit the supplied runs against a named profile and report pass/fail status.",
    )
    audit_calibrate.add_argument(
        "--fail-on-profile",
        action="store_true",
        help="With --compare-profile, return non-zero when any calibrated run fails that profile.",
    )
    audit_calibrate.add_argument(
        "--write-config",
        type=Path,
        default=None,
        help=(
            "Write the suggested audit thresholds as a reusable run-audit config JSON. "
            "Requires at least one --require-* sufficiency flag and writes only when sufficiency/profile checks pass."
        ),
    )
    audit_calibrate.add_argument("--json", action="store_true", help="Print the calibration result as JSON.")
    audit_calibrate.set_defaults(func=_audit_calibrate)

    smoke_plan = subparsers.add_parser(
        "cpu-smoke-plan",
        help="Print a tiny CPU-only command recipe for end-to-end bootstrap/self-play validation.",
    )
    _add_cpu_smoke_arguments(smoke_plan)
    smoke_plan.add_argument("--json", action="store_true", help="Print the recipe as JSON.")
    smoke_plan.set_defaults(func=_cpu_smoke_plan)

    smoke_run = subparsers.add_parser(
        "cpu-smoke-run",
        help="Execute the tiny CPU-only bootstrap/self-play validation recipe sequentially.",
    )
    _add_cpu_smoke_arguments(smoke_run)
    smoke_run.add_argument(
        "--summary-path",
        type=Path,
        default=None,
        help="Where to write the smoke-run summary JSON. Defaults to RUN_ROOT/cpu-smoke-run-summary.json.",
    )
    smoke_run.set_defaults(func=_cpu_smoke_run)

    smoke_report = subparsers.add_parser(
        "cpu-smoke-report",
        help="Inspect a cpu-smoke-run summary JSON artifact.",
    )
    smoke_report.add_argument(
        "path",
        type=Path,
        help="Smoke run root or cpu-smoke-run-summary.json path.",
    )
    smoke_report.add_argument("--json", action="store_true", help="Print the summary payload as JSON.")
    smoke_report.set_defaults(func=_cpu_smoke_report)

    pilot_plan = subparsers.add_parser(
        "cpu-pilot-plan",
        help="Print a CPU-only multi-pilot recipe that runs smoke pilots and calibrates audit thresholds.",
    )
    _add_cpu_pilot_arguments(pilot_plan)
    pilot_plan.add_argument("--json", action="store_true", help="Print the pilot recipe as JSON.")
    pilot_plan.set_defaults(func=_cpu_pilot_plan)

    pilot_run = subparsers.add_parser(
        "cpu-pilot-run",
        help="Execute a CPU-only multi-pilot recipe and write a suite summary artifact.",
    )
    _add_cpu_pilot_arguments(pilot_run)
    pilot_run.add_argument(
        "--summary-path",
        type=Path,
        default=None,
        help="Where to write the pilot-suite summary JSON. Defaults to RUN_ROOT/cpu-pilot-suite-summary.json.",
    )
    pilot_run.set_defaults(func=_cpu_pilot_run)

    pilot_report = subparsers.add_parser(
        "cpu-pilot-report",
        help="Inspect a cpu-pilot-run summary JSON artifact.",
    )
    pilot_report.add_argument(
        "path",
        type=Path,
        help="Pilot suite run root or cpu-pilot-suite-summary.json path.",
    )
    pilot_report.add_argument("--json", action="store_true", help="Print the summary payload as JSON.")
    pilot_report.add_argument(
        "--require-ready",
        action="store_true",
        help="Return non-zero unless the derived audit_config_ready verdict is true.",
    )
    pilot_report.set_defaults(func=_cpu_pilot_report)

    compare = subparsers.add_parser("compare", help="Compare self-play run manifests side by side.")
    compare.add_argument("paths", type=Path, nargs="*", help="Self-play or neural self-play run directories or manifest.json paths.")
    compare.add_argument(
        "--manifest-glob",
        action="append",
        default=None,
        help=(
            "Glob pattern for run directories or manifest.json files to include in comparison. "
            "May be repeated and is expanded in sorted order."
        ),
    )
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
        "--audit-config",
        type=Path,
        default=None,
        help="Also evaluate each compared run against a versioned run-audit config JSON.",
    )
    compare.add_argument(
        "--fail-on-audit",
        action="store_true",
        help="With --audit-profile, return non-zero when any compared run fails the selected audit profile.",
    )
    compare.add_argument(
        "--suggest-audit-calibration",
        action="store_true",
        help="Also suggest audit thresholds from the valid compared runs.",
    )
    compare.add_argument(
        "--calibration-margin",
        type=float,
        default=DEFAULT_AUDIT_CALIBRATION_MARGIN,
        help="Fractional safety margin applied to compare audit-calibration suggestions.",
    )
    compare.add_argument(
        "--calibration-aggregate-mode",
        choices=("median", "envelope"),
        default="median",
        help=(
            "How multiple valid compared runs are combined when suggesting audit calibration. "
            "median resists noisy pilots; envelope keeps every valid pilot passable."
        ),
    )
    compare.add_argument(
        "--calibration-require-run-count",
        type=int,
        default=0,
        help=(
            "With --suggest-audit-calibration, return non-zero unless at least this many valid compared "
            "runs contributed to the calibration."
        ),
    )
    compare.add_argument(
        "--calibration-require-benchmark-iterations",
        type=int,
        default=0,
        help=(
            "With --suggest-audit-calibration, return non-zero unless at least this many valid benchmark "
            "iterations contributed to the calibration."
        ),
    )
    compare.add_argument(
        "--calibration-require-min-benchmark-games",
        type=int,
        default=0,
        help=(
            "With --suggest-audit-calibration, return non-zero unless calibrated benchmark iterations "
            "used at least this many games."
        ),
    )
    compare.add_argument(
        "--write-audit-config",
        type=Path,
        default=None,
        help=(
            "With --suggest-audit-calibration and sufficiency requirements, write the suggested "
            "audit config to this versioned JSON path."
        ),
    )
    compare.add_argument("--json", action="store_true", help="Print the comparison result as JSON.")
    compare.set_defaults(func=_compare)
    return parser


def _add_cpu_smoke_arguments(
    parser: argparse.ArgumentParser,
    *,
    run_root_default: Path = Path("runs/cpu-smoke"),
    audit_config_help: str | None = None,
) -> None:
    parser.add_argument("--run-root", type=Path, default=run_root_default, help="Root directory used by the smoke recipe.")
    parser.add_argument(
        "--python-binary",
        default=sys.executable,
        help="Python executable used by the smoke recipe. Defaults to the interpreter running this command.",
    )
    parser.add_argument(
        "--showdown-root",
        type=Path,
        default=None,
        help=(
            "Built Pokemon Showdown checkout root used by the smoke recipe. "
            "If omitted, child commands use their normal Showdown-root resolution."
        ),
    )
    parser.add_argument("--workers", type=int, default=1, help="Worker count used by the smoke recipe.")
    parser.add_argument("--train-games", type=int, default=4, help="Teacher bootstrap training games.")
    parser.add_argument("--validation-games", type=int, default=2, help="Teacher bootstrap validation games.")
    parser.add_argument("--bootstrap-benchmark-games", type=int, default=2, help="Teacher bootstrap benchmark games.")
    parser.add_argument(
        "--teacher-branch-preflight-games",
        type=int,
        default=2,
        help=(
            "Teacher benchmark games used when --require-teacher-branch or "
            "--min-teacher-branch-count is supplied."
        ),
    )
    parser.add_argument(
        "--require-teacher-branch",
        action="append",
        default=None,
        help=(
            "Insert a teacher-benchmark preflight and require this scripted-teacher branch "
            "to appear at least once. May be repeated."
        ),
    )
    parser.add_argument(
        "--min-teacher-branch-count",
        action="append",
        default=None,
        metavar="BRANCH=COUNT",
        help=(
            "Insert a teacher-benchmark preflight and require this scripted-teacher branch "
            "to appear at least COUNT times. May be repeated."
        ),
    )
    parser.add_argument("--selfplay-iterations", type=int, default=2, help="Self-play iterations.")
    parser.add_argument("--selfplay-games", type=int, default=4, help="Self-play collection games per iteration.")
    parser.add_argument("--evaluation-games", type=int, default=2, help="Self-play benchmark games per matchup.")
    parser.add_argument("--feature-count", type=int, default=4096, help="Small linear feature count for the smoke run.")
    parser.add_argument("--window-size", type=int, default=4, help="Temporal window size for bootstrap and self-play.")
    parser.add_argument("--max-decision-rounds", type=int, default=250, help="Decision-round cap used by the smoke recipe.")
    parser.add_argument("--seed-start", type=int, default=1, help="Base deterministic seed used by the smoke recipe.")
    parser.add_argument(
        "--audit-config-path",
        type=Path,
        default=None,
        help=audit_config_help
        or (
            "Where the smoke recipe writes its calibrated audit config. "
            "Defaults to RUN_ROOT/smoke-audit-config.json."
        ),
    )


def _add_cpu_pilot_arguments(parser: argparse.ArgumentParser) -> None:
    _add_cpu_smoke_arguments(
        parser,
        run_root_default=Path("runs/cpu-pilots"),
        audit_config_help=(
            "Where the pilot suite writes its calibrated audit config. "
            "Defaults to RUN_ROOT/pilot-audit-config.json. Each per-pilot smoke run writes "
            "PILOT_ROOT/smoke-audit-config.json."
        ),
    )
    parser.add_argument("--pilot-count", type=int, default=2, help="Number of seeded CPU smoke pilots to run.")
    parser.add_argument(
        "--seed-stride",
        type=int,
        default=10_000,
        help="Seed increment between pilot runs.",
    )
    parser.add_argument(
        "--calibration-require-min-benchmark-games",
        type=int,
        default=1,
        help="Minimum benchmark games each calibrated pilot iteration must include.",
    )


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
    if args.verify_opponent_pool_only and not args.verify:
        raise ValueError("--verify-opponent-pool-only requires --verify.")
    if args.verify_opponent_pool_only and args.opponent_pool_size is None:
        raise ValueError("--verify-opponent-pool-only requires --opponent-pool-size.")
    if args.current_policy_spec is not None and args.opponent_pool_size is None:
        raise ValueError("--current-policy-spec requires --opponent-pool-size.")
    if args.require_opponent_pool_size is not None and args.opponent_pool_size is None:
        raise ValueError("--require-opponent-pool-size requires --opponent-pool-size.")
    if args.write_opponent_pool is not None and args.opponent_pool_size is None:
        raise ValueError("--write-opponent-pool requires --opponent-pool-size.")
    if args.retention_plan and args.opponent_pool_size is None:
        raise ValueError("--retention-plan requires --opponent-pool-size.")
    if args.apply_retention_plan and not args.retention_plan:
        raise ValueError("--apply-retention-plan requires --retention-plan.")
    if args.apply_retention_plan and not (args.verify and args.verify_loadable):
        raise ValueError("--apply-retention-plan requires --verify --verify-loadable.")
    if args.retention_apply_confirm is not None and not args.apply_retention_plan:
        raise ValueError("--retention-apply-confirm requires --apply-retention-plan.")
    if args.retention_archive_dir is not None and not args.apply_retention_plan:
        raise ValueError("--retention-archive-dir requires --apply-retention-plan.")
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
        current_policy_spec=preview_current_policy_spec,
    )
    lifecycle_summary = _promotion_lifecycle_summary(
        entry_statuses,
        verification=verification,
        opponent_pool=opponent_pool,
    )
    opponent_pool_verified = _opponent_pool_verification_passed(
        entry_statuses,
        verification=verification,
        opponent_pool=opponent_pool,
    )
    selected_opponent_pool_verified = _selected_opponent_pool_verification_passed(
        entry_statuses,
        verification=verification,
        opponent_pool=opponent_pool,
    )
    opponent_pool_current_policy_verified = _opponent_pool_current_policy_verification_passed(
        entry_statuses,
        verification=verification,
        opponent_pool=opponent_pool,
    )
    opponent_pool_registry_level_verified = _opponent_pool_registry_level_verification_passed(
        verification=verification,
        opponent_pool=opponent_pool,
    )
    opponent_pool_preflight_verified = (
        None
        if selected_opponent_pool_verified is None or opponent_pool_registry_level_verified is None
        else (
            selected_opponent_pool_verified
            and opponent_pool_registry_level_verified
            and opponent_pool_current_policy_verified is not False
        )
    )
    opponent_pool_snapshot = (
        _opponent_pool_snapshot_payload(
            registry=registry,
            entry_statuses=entry_statuses,
            opponent_pool=opponent_pool,
            available_opponent_pool=available_opponent_pool,
            current_policy_spec=preview_current_policy_spec,
            requested_size=args.opponent_pool_size,
            required_size=args.require_opponent_pool_size,
            verification=verification,
            verify_opponent_pool_only=args.verify_opponent_pool_only,
            opponent_pool_verified=opponent_pool_verified,
            selected_opponent_pool_verified=selected_opponent_pool_verified,
            opponent_pool_current_policy_verified=opponent_pool_current_policy_verified,
            opponent_pool_registry_level_verified=opponent_pool_registry_level_verified,
            opponent_pool_preflight_verified=opponent_pool_preflight_verified,
        )
        if opponent_pool is not None
        else None
    )
    retention_plan = (
        _promotion_retention_plan_payload(
            registry=registry,
            entry_statuses=entry_statuses,
            opponent_pool=opponent_pool,
            available_opponent_pool=available_opponent_pool,
            current_policy_spec=preview_current_policy_spec,
            requested_size=args.opponent_pool_size,
            verification=verification,
            registry_level_verification_passed=opponent_pool_registry_level_verified,
        )
        if args.retention_plan and opponent_pool is not None
        else None
    )
    promotions_exit_code = _promotions_exit_code(
        verification=verification,
        opponent_pool=opponent_pool,
        required_opponent_pool_size=args.require_opponent_pool_size,
        opponent_pool_preflight_verified=opponent_pool_preflight_verified,
        verify_opponent_pool_only=args.verify_opponent_pool_only,
    )
    if args.retention_apply_confirm == "archive" and promotions_exit_code != 0:
        raise ValueError("--retention-apply-confirm archive requires a passing promotions preflight.")
    retention_apply = (
        _promotion_retention_apply_payload(
            registry_path=registry.path,
            retention_plan=retention_plan,
            archive_dir=args.retention_archive_dir,
            confirm_archive=args.retention_apply_confirm == "archive",
        )
        if args.apply_retention_plan and retention_plan is not None
        else None
    )
    if args.write_opponent_pool is not None and opponent_pool_snapshot is not None:
        _write_json_payload(args.write_opponent_pool, opponent_pool_snapshot)
    if args.json:
        payload = registry.to_dict()
        payload["entry_statuses"] = entry_statuses
        if args.lifecycle:
            payload["lifecycle_summary"] = lifecycle_summary
        if opponent_pool is not None:
            payload["opponent_pool_policy_specs"] = list(opponent_pool)
            payload["opponent_pool_excluded_current_policy_spec"] = preview_current_policy_spec
            payload["opponent_pool_verified"] = opponent_pool_verified
            payload["selected_opponent_pool_verified"] = selected_opponent_pool_verified
            payload["opponent_pool_current_policy_verified"] = opponent_pool_current_policy_verified
            payload["opponent_pool_registry_level_verified"] = opponent_pool_registry_level_verified
            payload["opponent_pool_preflight_verified"] = opponent_pool_preflight_verified
            payload["opponent_pool_verification_exit_scope"] = (
                None
                if verification is None
                else "opponent_pool_plus_current" if args.verify_opponent_pool_only else "registry"
            )
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
            payload["opponent_pool_snapshot"] = opponent_pool_snapshot
            if args.write_opponent_pool is not None:
                payload["opponent_pool_snapshot_path"] = str(args.write_opponent_pool)
        if retention_plan is not None:
            payload["retention_plan"] = retention_plan
        if retention_apply is not None:
            payload["retention_apply"] = retention_apply
        if verification is not None:
            payload["verification"] = verification.to_dict()
        print(json.dumps(payload, indent=2, sort_keys=True))
        return promotions_exit_code
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
            pool = f" pool={status['opponent_pool_status']}" if opponent_pool is not None else ""
            print(
                f"- {entry.sequence}: policy={entry.policy_id or '-'} "
                f"checkpoint={entry.checkpoint_path or '-'} promoted_at={entry.promoted_at} "
                f"status={status['verification_status']} selected={selected_as}{pool} "
                f"path={status['checkpoint_path_present']} exists={status['checkpoint_exists']} checksum={status['checksum']} "
                f"loadable={status['loadable']}{label}{source}"
            )
    if args.lifecycle:
        _print_promotion_lifecycle_summary(lifecycle_summary)
    if retention_plan is not None:
        _print_promotion_retention_plan(retention_plan)
    if retention_apply is not None:
        _print_promotion_retention_apply(retention_apply)
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
        if selected_opponent_pool_verified is not None:
            print(f"opponent_pool_verification: {'PASS' if opponent_pool_verified else 'FAIL'}")
            print(f"selected_opponent_pool_verification: {'PASS' if selected_opponent_pool_verified else 'FAIL'}")
            print(
                "opponent_pool_current_policy_verification: "
                f"{_verification_bool_label(opponent_pool_current_policy_verified)}"
            )
            print(
                "opponent_pool_registry_level_verification: "
                f"{'PASS' if opponent_pool_registry_level_verified else 'FAIL'}"
            )
            print(
                "opponent_pool_preflight_verification: "
                f"{'PASS' if opponent_pool_preflight_verified else 'FAIL'}"
            )
            print(
                "opponent_pool_verification_exit_scope: "
                f"{'opponent_pool_plus_current' if args.verify_opponent_pool_only else 'registry'}"
            )
        print("opponent_pool_policy_specs:")
        for spec in opponent_pool:
            print(f"- {spec}")
        if args.write_opponent_pool is not None:
            print(f"opponent_pool_snapshot: {args.write_opponent_pool}")
        if verification is None:
            print("note: pass --verify to confirm the previewed registry is selectable by runtime.")
    if verification is not None:
        _print_registry_verification(verification)
    return promotions_exit_code


def _promotions_exit_code(
    *,
    verification,
    opponent_pool,
    required_opponent_pool_size: int | None,
    opponent_pool_preflight_verified: bool | None,
    verify_opponent_pool_only: bool,
) -> int:
    if verify_opponent_pool_only:
        if opponent_pool_preflight_verified is not True:
            return 2
    elif verification is not None and not verification.passed:
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


def _opponent_pool_snapshot_payload(
    *,
    registry,
    entry_statuses: list[dict[str, object]],
    opponent_pool,
    available_opponent_pool,
    current_policy_spec: str | None,
    requested_size: int | None,
    required_size: int | None,
    verification,
    verify_opponent_pool_only: bool,
    opponent_pool_verified: bool | None,
    selected_opponent_pool_verified: bool | None,
    opponent_pool_current_policy_verified: bool | None,
    opponent_pool_registry_level_verified: bool | None,
    opponent_pool_preflight_verified: bool | None,
) -> dict[str, object]:
    return {
        "schema_version": OPPONENT_POOL_SNAPSHOT_SCHEMA_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "registry_path": str(registry.path),
        "registry_latest_policy_id": registry.latest.policy_id if registry.latest is not None else None,
        "registry_latest_checkpoint_path": registry.latest.checkpoint_path if registry.latest is not None else None,
        "excluded_current_policy_spec": current_policy_spec,
        "requested_size": requested_size,
        "required_size": required_size,
        "selected_size": len(opponent_pool),
        "available_size": len(available_opponent_pool) if available_opponent_pool is not None else None,
        "requirement_passed": _opponent_pool_requirement_passed(opponent_pool, required_size=required_size),
        "verification_enabled": verification is not None,
        "verification_exit_scope": (
            None
            if verification is None
            else "opponent_pool_plus_current" if verify_opponent_pool_only else "registry"
        ),
        "opponent_pool_verified": opponent_pool_verified,
        "selected_opponent_pool_verified": selected_opponent_pool_verified,
        "current_policy_verified": opponent_pool_current_policy_verified,
        "registry_level_verified": opponent_pool_registry_level_verified,
        "preflight_verified": opponent_pool_preflight_verified,
        "policy_specs": list(opponent_pool),
        "selected_entries": [
            _opponent_pool_snapshot_entry(status)
            for status in entry_statuses
            if "opponent_pool" in status["selected_as"]
        ],
    }


def _opponent_pool_snapshot_entry(status: Mapping[str, object]) -> dict[str, object]:
    return {
        "sequence": status["sequence"],
        "policy_id": status["policy_id"],
        "label": status["label"],
        "selection_checkpoint_policy_spec": status["selection_checkpoint_policy_spec"],
        "checkpoint_path": status["checkpoint_path"],
        "source_checkpoint_path": status["source_checkpoint_path"],
        "source_type": status["source_type"],
        "source_iteration": status["source_iteration"],
        "retention_archived_from_checkpoint_path": status["retention_archived_from_checkpoint_path"],
        "retention_archived_at": status["retention_archived_at"],
        "promoted_at": status["promoted_at"],
        "verification_status": status["verification_status"],
        "checkpoint_exists": status["checkpoint_exists"],
        "checksum": status["checksum"],
        "loadable": status["loadable"],
        "policy_id_matches": status["policy_id_matches"],
        "failed_checks": list(status["failed_checks"]),
    }


def _promotion_retention_plan_payload(
    *,
    registry,
    entry_statuses: list[dict[str, object]],
    opponent_pool,
    available_opponent_pool,
    current_policy_spec: str | None,
    requested_size: int | None,
    verification,
    registry_level_verification_passed: bool | None,
) -> dict[str, object]:
    entries = [
        _promotion_retention_plan_entry(
            status,
            verification_enabled=verification is not None,
            registry_level_verification_passed=registry_level_verification_passed,
        )
        for status in entry_statuses
    ]
    action_counts = _count_plan_values(entries, "recommended_action")
    return {
        "schema_version": PROMOTION_RETENTION_PLAN_SCHEMA_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "registry_path": str(registry.path),
        "registry_latest_policy_id": registry.latest.policy_id if registry.latest is not None else None,
        "registry_latest_checkpoint_path": registry.latest.checkpoint_path if registry.latest is not None else None,
        "excluded_current_policy_spec": current_policy_spec,
        "requested_opponent_pool_size": requested_size,
        "selected_opponent_pool_size": len(opponent_pool),
        "available_opponent_pool_size": len(available_opponent_pool) if available_opponent_pool is not None else None,
        "verification_enabled": verification is not None,
        "registry_level_verification_passed": registry_level_verification_passed,
        "summary": {
            "total_entries": len(entries),
            "retain_count": int(action_counts.get("retain", 0)),
            "verify_before_cleanup_count": int(action_counts.get("verify_before_cleanup", 0)),
            "cleanup_candidate_count": int(action_counts.get("cleanup_candidate", 0)),
            "manual_review_count": int(action_counts.get("manual_review", 0)),
            "recommended_action_counts": action_counts,
        },
        "entries": entries,
    }


def _promotion_retention_plan_entry(
    status: Mapping[str, object],
    *,
    verification_enabled: bool,
    registry_level_verification_passed: bool | None,
) -> dict[str, object]:
    action, reason = _promotion_retention_recommendation(
        status,
        verification_enabled=verification_enabled,
        registry_level_verification_passed=registry_level_verification_passed,
    )
    return {
        "sequence": status["sequence"],
        "policy_id": status["policy_id"],
        "label": status["label"],
        "selection_checkpoint_policy_spec": status["selection_checkpoint_policy_spec"],
        "checkpoint_path": status["checkpoint_path"],
        "source_checkpoint_path": status["source_checkpoint_path"],
        "source_type": status["source_type"],
        "source_iteration": status["source_iteration"],
        "promoted_at": status["promoted_at"],
        "selected_as": list(status["selected_as"]),
        "opponent_pool_status": status["opponent_pool_status"],
        "opponent_pool_skip_reason": status["opponent_pool_skip_reason"],
        "verification_status": status["verification_status"],
        "checkpoint_exists": status["checkpoint_exists"],
        "checksum": status["checksum"],
        "loadable": status["loadable"],
        "policy_id_matches": status["policy_id_matches"],
        "failed_checks": list(status["failed_checks"]),
        "recommended_action": action,
        "recommendation_reason": reason,
    }


def _promotion_retention_recommendation(
    status: Mapping[str, object],
    *,
    verification_enabled: bool,
    registry_level_verification_passed: bool | None,
) -> tuple[str, str]:
    selected_as = set(status["selected_as"] if isinstance(status["selected_as"], list) else ())
    if status["failed_checks"]:
        return "manual_review", "verification_failed"
    if status["opponent_pool_status"] == "unselectable":
        return "manual_review", "missing_selection_checkpoint"
    if "latest" in selected_as:
        return "retain", "latest_promotion"
    if "opponent_pool" in selected_as:
        return "retain", "selected_opponent_pool"
    if status["opponent_pool_status"] == "excluded_current_policy":
        return "retain", "current_policy_exclusion"
    if status["retention_archived_at"] is not None:
        return "retain", "already_archived"
    if status["opponent_pool_status"] == "available_outside_requested_size":
        if not verification_enabled:
            return "verify_before_cleanup", "stale_outside_requested_pool_unverified"
        if registry_level_verification_passed is not True:
            return "manual_review", "registry_verification_failed"
        if status["verification_status"] != "pass":
            return "verify_before_cleanup", "stale_outside_requested_pool_partially_verified"
        return "cleanup_candidate", "stale_outside_requested_pool"
    return "retain", "not_stale"


def _count_plan_values(entries: list[Mapping[str, object]], key: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for entry in entries:
        value = str(entry[key])
        counts[value] = counts.get(value, 0) + 1
    return dict(sorted(counts.items()))


def _print_promotion_retention_plan(plan: Mapping[str, object]) -> None:
    summary = plan["summary"] if isinstance(plan["summary"], Mapping) else {}
    print("retention_plan:")
    print(f"- schema_version: {plan['schema_version']}")
    print(f"- verification_enabled: {plan['verification_enabled']}")
    print(f"- selected_opponent_pool_size: {plan['selected_opponent_pool_size']}")
    print(f"- available_opponent_pool_size: {plan['available_opponent_pool_size']}")
    print(f"- registry_level_verification: {_verification_bool_label(plan['registry_level_verification_passed'])}")
    print(f"- retain_count: {summary.get('retain_count', 0)}")
    print(f"- verify_before_cleanup_count: {summary.get('verify_before_cleanup_count', 0)}")
    print(f"- cleanup_candidate_count: {summary.get('cleanup_candidate_count', 0)}")
    print(f"- manual_review_count: {summary.get('manual_review_count', 0)}")
    print("retention_entries:")
    entries = plan["entries"] if isinstance(plan["entries"], list) else []
    if not entries:
        print("- none")
        return
    for entry in entries:
        selected_as = ",".join(entry["selected_as"]) if entry["selected_as"] else "-"
        print(
            f"- {entry['sequence']}: action={entry['recommended_action']} "
            f"reason={entry['recommendation_reason']} pool={entry['opponent_pool_status']} "
            f"selected={selected_as} verification={entry['verification_status']} "
            f"checkpoint={entry['checkpoint_path'] or '-'}"
        )


def _promotion_retention_apply_payload(
    *,
    registry_path: Path,
    retention_plan: Mapping[str, object],
    archive_dir: Path | None,
    confirm_archive: bool,
) -> dict[str, object]:
    generated_at = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    batch_id = _retention_archive_batch_id(generated_at)
    resolved_archive_dir = (
        archive_dir.expanduser().resolve(strict=False)
        if archive_dir is not None
        else registry_path.parent / "retention-archive" / batch_id
    )
    dry_run = not confirm_archive
    entries = [
        _promotion_retention_apply_entry(
            entry,
            registry_path=registry_path,
            archive_dir=resolved_archive_dir,
            dry_run=dry_run,
        )
        for entry in _retention_plan_entries(retention_plan)
    ]
    if not dry_run:
        _apply_retention_archive_entries(
            registry_path=registry_path,
            entries=entries,
            generated_at=generated_at,
        )
    status_counts = _count_plan_values(entries, "apply_status")
    return {
        "schema_version": PROMOTION_RETENTION_APPLY_SCHEMA_VERSION,
        "generated_at": generated_at,
        "registry_path": str(registry_path),
        "retention_plan_schema_version": retention_plan.get("schema_version"),
        "dry_run": dry_run,
        "confirmation": "archive" if confirm_archive else None,
        "archive_dir": str(resolved_archive_dir),
        "registry_updates_checkpoint_paths": confirm_archive,
        "summary": {
            "total_plan_entries": len(entries),
            "cleanup_candidate_count": sum(
                1 for entry in entries if entry["recommended_action"] == "cleanup_candidate"
            ),
            "archive_candidate_count": sum(1 for entry in entries if entry["apply_action"] == "archive"),
            "applied_count": int(status_counts.get("applied", 0)),
            "planned_count": int(status_counts.get("planned", 0)),
            "skipped_count": int(status_counts.get("skipped", 0)),
            "apply_status_counts": status_counts,
        },
        "entries": entries,
    }


def _retention_plan_entries(retention_plan: Mapping[str, object]) -> tuple[Mapping[str, object], ...]:
    entries = retention_plan.get("entries", ())
    if not isinstance(entries, list):
        return ()
    return tuple(entry for entry in entries if isinstance(entry, Mapping))


def _promotion_retention_apply_entry(
    entry: Mapping[str, object],
    *,
    registry_path: Path,
    archive_dir: Path,
    dry_run: bool,
) -> dict[str, object]:
    base = {
        "sequence": entry.get("sequence"),
        "policy_id": entry.get("policy_id"),
        "label": entry.get("label"),
        "checkpoint_path": entry.get("checkpoint_path"),
        "source_checkpoint_path": entry.get("source_checkpoint_path"),
        "retention_archived_from_checkpoint_path": entry.get("retention_archived_from_checkpoint_path"),
        "retention_archived_at": entry.get("retention_archived_at"),
        "recommended_action": entry.get("recommended_action"),
        "recommendation_reason": entry.get("recommendation_reason"),
    }
    if entry.get("recommended_action") != "cleanup_candidate":
        return {
            **base,
            "apply_action": "skip",
            "apply_status": "skipped",
            "apply_reason": "not_cleanup_candidate",
            "resolved_checkpoint_path": None,
            "archive_path": None,
        }
    if not _retention_entry_is_managed_artifact(entry):
        return {
            **base,
            "apply_action": "skip",
            "apply_status": "skipped",
            "apply_reason": "not_managed_artifact_copy",
            "resolved_checkpoint_path": None,
            "archive_path": None,
        }
    resolved_checkpoint = _resolve_retention_checkpoint_path(entry.get("checkpoint_path"), registry_path=registry_path)
    if resolved_checkpoint is None:
        return {
            **base,
            "apply_action": "skip",
            "apply_status": "skipped",
            "apply_reason": "checkpoint_unresolved",
            "resolved_checkpoint_path": None,
            "archive_path": None,
        }
    archive_path = archive_dir / f"{int(entry['sequence']):06d}-{resolved_checkpoint.name}"
    return {
        **base,
        "apply_action": "archive",
        "apply_status": "planned" if dry_run else "pending",
        "apply_reason": "dry_run" if dry_run else "confirmed_archive",
        "resolved_checkpoint_path": str(resolved_checkpoint),
        "archive_path": str(archive_path),
    }


def _retention_entry_is_managed_artifact(entry: Mapping[str, object]) -> bool:
    checkpoint_path = entry.get("checkpoint_path")
    source_checkpoint_path = entry.get("source_checkpoint_path")
    return (
        isinstance(checkpoint_path, str)
        and bool(checkpoint_path)
        and isinstance(source_checkpoint_path, str)
        and bool(source_checkpoint_path)
        and checkpoint_path != source_checkpoint_path
    )


def _resolve_retention_checkpoint_path(value: object, *, registry_path: Path) -> Path | None:
    if not isinstance(value, str) or not value:
        return None
    raw_path = Path(value).expanduser()
    candidates = (raw_path,) if raw_path.is_absolute() else (registry_path.parent / raw_path, raw_path)
    for candidate in _dedupe_cli_paths(candidates):
        if candidate.exists() and candidate.is_file():
            return candidate.resolve(strict=False)
    return None


def _dedupe_cli_paths(paths: Iterable[Path]) -> tuple[Path, ...]:
    deduped: list[Path] = []
    seen: set[str] = set()
    for path in paths:
        key = str(path)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(path)
    return tuple(deduped)


def _apply_retention_archive_entries(
    *,
    registry_path: Path,
    entries: list[dict[str, object]],
    generated_at: str,
) -> None:
    archive_entries = [entry for entry in entries if entry["apply_action"] == "archive"]
    if not archive_entries:
        return
    updates: dict[int, str] = {}
    moved_entries: list[tuple[Path, Path, dict[str, object]]] = []
    archive_root: Path | None = None
    try:
        for entry in archive_entries:
            source_path = Path(str(entry["resolved_checkpoint_path"]))
            archive_path = Path(str(entry["archive_path"]))
            if archive_root is None:
                archive_root = archive_path.parent
                archive_root.mkdir(parents=True, exist_ok=True)
            if archive_path.exists():
                entry["apply_status"] = "skipped"
                entry["apply_reason"] = "archive_path_exists"
                continue
            archive_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(source_path), str(archive_path))
            moved_entries.append((archive_path, source_path, entry))
            updates[int(entry["sequence"])] = str(archive_path)
        if updates:
            _rewrite_retention_archived_checkpoint_paths(
                registry_path=registry_path,
                checkpoint_path_updates=updates,
                generated_at=generated_at,
            )
    except Exception:
        _rollback_retention_archive_moves(moved_entries)
        raise
    for _, _, entry in moved_entries:
        entry["apply_status"] = "applied"
        entry["apply_reason"] = "archived"


def _rollback_retention_archive_moves(moved_entries: list[tuple[Path, Path, dict[str, object]]]) -> None:
    for archive_path, source_path, entry in reversed(moved_entries):
        if archive_path.exists() and not source_path.exists():
            source_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(archive_path), str(source_path))
        entry["apply_status"] = "rolled_back"
        entry["apply_reason"] = "archive_apply_failed_rolled_back"


def _rewrite_retention_archived_checkpoint_paths(
    *,
    registry_path: Path,
    checkpoint_path_updates: Mapping[int, str],
    generated_at: str,
) -> None:
    payload = json.loads(registry_path.read_text(encoding="utf-8"))
    entries = payload.get("entries", ())
    if not isinstance(entries, list):
        raise ValueError("promotion registry entries must be a JSON array.")
    for entry in entries:
        if not isinstance(entry, Mapping):
            continue
        sequence = int(entry["sequence"])
        if sequence not in checkpoint_path_updates:
            continue
        previous_checkpoint_path = entry.get("checkpoint_path")
        entry["checkpoint_path"] = checkpoint_path_updates[sequence]
        entry.setdefault("retention_archived_from_checkpoint_path", previous_checkpoint_path)
        entry["retention_archived_at"] = generated_at
    _write_json_payload(registry_path, payload)


def _retention_archive_batch_id(generated_at: str) -> str:
    return (
        generated_at.replace(":", "")
        .replace("-", "")
        .replace(".", "")
        .replace("+0000", "Z")
    )


def _print_promotion_retention_apply(apply_payload: Mapping[str, object]) -> None:
    summary = apply_payload["summary"] if isinstance(apply_payload["summary"], Mapping) else {}
    print("retention_apply:")
    print(f"- schema_version: {apply_payload['schema_version']}")
    print(f"- dry_run: {apply_payload['dry_run']}")
    print(f"- confirmation: {apply_payload['confirmation'] or '-'}")
    print(f"- archive_dir: {apply_payload['archive_dir']}")
    print(f"- registry_updates_checkpoint_paths: {apply_payload['registry_updates_checkpoint_paths']}")
    print(f"- archive_candidate_count: {summary.get('archive_candidate_count', 0)}")
    print(f"- applied_count: {summary.get('applied_count', 0)}")
    print(f"- planned_count: {summary.get('planned_count', 0)}")
    print(f"- skipped_count: {summary.get('skipped_count', 0)}")
    print("retention_apply_entries:")
    entries = apply_payload["entries"] if isinstance(apply_payload["entries"], list) else []
    if not entries:
        print("- none")
        return
    for entry in entries:
        print(
            f"- {entry['sequence']}: action={entry['apply_action']} status={entry['apply_status']} "
            f"reason={entry['apply_reason']} checkpoint={entry['checkpoint_path'] or '-'} "
            f"archive={entry['archive_path'] or '-'}"
        )


def _promotion_lifecycle_summary(
    entry_statuses: list[dict[str, object]],
    *,
    verification,
    opponent_pool,
) -> dict[str, object]:
    total_entries = len(entry_statuses)
    opponent_pool_requested = opponent_pool is not None
    verification_enabled = verification is not None
    latest_count = sum(1 for status in entry_statuses if "latest" in status["selected_as"])
    selected_opponent_pool_count = sum(1 for status in entry_statuses if "opponent_pool" in status["selected_as"])
    status_counts = _count_status_values(entry_statuses, "opponent_pool_status")
    verification_counts = _count_status_values(entry_statuses, "verification_status")
    unselectable_count = sum(
        1
        for status in entry_statuses
        if status["selection_checkpoint_policy_spec"] is None
    )
    excluded_current_policy_count = int(status_counts.get("excluded_current_policy", 0))
    stale_available_count = int(status_counts.get("available_outside_requested_size", 0))
    selection_eligible_count = sum(
        1
        for status in entry_statuses
        if status["selection_checkpoint_policy_spec"] is not None
    )
    failed_entry_verification_count = int(verification_counts.get("fail", 0))
    registry_level_failed_verification_count = (
        0
        if verification is None
        else sum(1 for check in verification.checks if check.entry_sequence is None and not check.passed)
    )
    selected_opponent_pool_unhealthy_count = sum(
        1
        for status in entry_statuses
        if "opponent_pool" in status["selected_as"] and status["failed_checks"]
    )
    return {
        "total_entries": total_entries,
        "opponent_pool_requested": opponent_pool_requested,
        "verification_enabled": verification_enabled,
        "latest_count": latest_count,
        "selected_opponent_pool_count": selected_opponent_pool_count,
        "selected_opponent_pool_unhealthy_count": selected_opponent_pool_unhealthy_count,
        "selection_eligible_count": selection_eligible_count,
        "unselectable_count": unselectable_count,
        "excluded_current_policy_count": excluded_current_policy_count,
        "stale_available_count": stale_available_count,
        "failed_verification_count": failed_entry_verification_count,
        "registry_level_failed_verification_count": registry_level_failed_verification_count,
        "opponent_pool_status_counts": status_counts,
        "verification_status_counts": verification_counts,
    }


def _count_status_values(entry_statuses: list[dict[str, object]], key: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for status in entry_statuses:
        value = str(status[key])
        counts[value] = counts.get(value, 0) + 1
    return dict(sorted(counts.items()))


def _print_promotion_lifecycle_summary(summary: Mapping[str, object]) -> None:
    print("lifecycle_summary:")
    print(f"- total_entries: {summary['total_entries']}")
    print(f"- opponent_pool_requested: {summary['opponent_pool_requested']}")
    print(f"- verification_enabled: {summary['verification_enabled']}")
    print(f"- latest_count: {summary['latest_count']}")
    print(f"- selected_opponent_pool_count: {summary['selected_opponent_pool_count']}")
    print(f"- selected_opponent_pool_unhealthy_count: {summary['selected_opponent_pool_unhealthy_count']}")
    print(f"- selection_eligible_count: {summary['selection_eligible_count']}")
    print(f"- unselectable_count: {summary['unselectable_count']}")
    print(f"- excluded_current_policy_count: {summary['excluded_current_policy_count']}")
    print(f"- stale_available_count: {summary['stale_available_count']}")
    print(f"- failed_verification_count: {summary['failed_verification_count']}")
    print(f"- registry_level_failed_verification_count: {summary['registry_level_failed_verification_count']}")
    _print_count_mapping("opponent_pool_status_counts", summary["opponent_pool_status_counts"])
    _print_count_mapping("verification_status_counts", summary["verification_status_counts"])


def _print_count_mapping(label: str, value: object) -> None:
    counts = value if isinstance(value, Mapping) else {}
    print(f"{label}:")
    if not counts:
        print("- none: 0")
        return
    for key, count in counts.items():
        print(f"- {key}: {count}")


def _write_json_payload(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(f".{path.name}.tmp")
    temporary_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary_path.replace(path)


def _opponent_pool_verification_passed(
    entry_statuses: list[dict[str, object]],
    *,
    verification,
    opponent_pool,
) -> bool | None:
    if verification is None or opponent_pool is None:
        return None
    return verification.passed


def _selected_opponent_pool_verification_passed(
    entry_statuses: list[dict[str, object]],
    *,
    verification,
    opponent_pool,
) -> bool | None:
    if verification is None or opponent_pool is None:
        return None
    selected_statuses = _entry_status_group(
        entry_statuses,
        lambda status: "opponent_pool" in status["selected_as"],
    )
    return bool(selected_statuses) and len(selected_statuses) == len(opponent_pool) and all(
        not status["failed_checks"]
        for status in selected_statuses
    )


def _opponent_pool_current_policy_verification_passed(
    entry_statuses: list[dict[str, object]],
    *,
    verification,
    opponent_pool,
) -> bool | None:
    if verification is None or opponent_pool is None:
        return None
    current_policy_statuses = _entry_status_group(
        entry_statuses,
        lambda status: status["opponent_pool_status"] == "excluded_current_policy",
    )
    if not current_policy_statuses:
        return None
    return all(
        not status["failed_checks"]
        for status in current_policy_statuses
    )


def _opponent_pool_registry_level_verification_passed(
    *,
    verification,
    opponent_pool,
) -> bool | None:
    if verification is None or opponent_pool is None:
        return None
    return all(check.passed for check in verification.checks if check.entry_sequence is None)


def _verification_bool_label(value: bool | None) -> str:
    if value is None:
        return "N/A"
    return "PASS" if value else "FAIL"


def _entry_status_group(entry_statuses: list[dict[str, object]], predicate) -> tuple[dict[str, object], ...]:
    return tuple(status for status in entry_statuses if predicate(status))


def _promotion_entry_statuses(
    registry,
    *,
    verification,
    opponent_pool,
    current_policy_spec,
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
        opponent_pool_status, opponent_pool_skip_reason = _opponent_pool_entry_status(
            entry,
            selection_checkpoint_policy_spec=selection_checkpoint_policy_spec,
            opponent_pool=opponent_pool,
            opponent_pool_sequences=opponent_pool_sequences,
            current_policy_spec=current_policy_spec,
        )
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
                "retention_archived_from_checkpoint_path": entry.retention_archived_from_checkpoint_path,
                "retention_archived_at": entry.retention_archived_at,
                "promoted_at": entry.promoted_at,
                "selected_as": selected_as,
                "opponent_pool_status": opponent_pool_status,
                "opponent_pool_skip_reason": opponent_pool_skip_reason,
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


def _opponent_pool_entry_status(
    entry,
    *,
    selection_checkpoint_policy_spec,
    opponent_pool,
    opponent_pool_sequences: set[int],
    current_policy_spec,
) -> tuple[str, str | None]:
    if opponent_pool is None:
        return "not_requested", None
    if entry.sequence in opponent_pool_sequences:
        return "selected", None
    if selection_checkpoint_policy_spec is None:
        return "unselectable", "missing_selection_checkpoint"
    if current_policy_spec is not None and selection_checkpoint_policy_spec == current_policy_spec:
        return "excluded_current_policy", "matches_current_policy"
    return "available_outside_requested_size", "outside_requested_pool_size"


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


def _audit_config_report(args: argparse.Namespace) -> int:
    config_payload = _load_audit_config_report_payload(args.audit_config)
    paths = (
        _expanded_manifest_paths(args.paths, args.manifest_glob)
        if args.paths or args.manifest_glob
        else ()
    )
    preflight_runs = tuple(
        _audit_config_preflight_run_payload(path, config=config_payload["config_object"])
        for path in paths
    )
    preflight_passed = (
        None
        if not preflight_runs
        else all(bool(run["passed"]) for run in preflight_runs)
    )
    checks = _audit_config_report_checks(
        config_payload,
        preflight_passed=preflight_passed,
        require_source=args.require_source,
        require_calibration=args.require_calibration,
    )
    passed = all(bool(check["passed"]) for check in checks)
    report = {
        "audit_config_path": str(args.audit_config),
        "schema_version": config_payload["schema_version"],
        "config": run_audit_config_to_dict(config_payload["config_object"]),
        "source": config_payload["source"],
        "calibration": config_payload["calibration"],
        "preflight_requested": bool(preflight_runs),
        "preflight_passed": preflight_passed,
        "preflight_runs": list(preflight_runs),
        "checks": checks,
        "passed": passed,
    }
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        _print_audit_config_report(report)
    return 0 if passed else 2


def _load_audit_config_report_payload(path: Path) -> dict[str, object]:
    raw_payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw_payload, Mapping):
        raise ValueError(f"run audit config must be a JSON object: {path}")
    config = load_run_audit_config(path)
    source = raw_payload.get("source")
    calibration = raw_payload.get("calibration")
    return {
        "schema_version": raw_payload.get("schema_version", RUN_AUDIT_CONFIG_SCHEMA_VERSION),
        "config_object": config,
        "source": dict(source) if isinstance(source, Mapping) else None,
        "calibration": dict(calibration) if isinstance(calibration, Mapping) else None,
    }


def _audit_config_preflight_run_payload(path: Path, *, config: RunAuditConfig) -> dict[str, object]:
    try:
        result = audit_run(path, config=config)
    except Exception as exc:
        return {
            "manifest_path": str(path),
            "source_type": None,
            "latest_iteration": None,
            "latest_benchmark_win_rate": None,
            "latest_benchmark_games": 0,
            "latest_collection_capped_rate": None,
            "latest_benchmark_capped_rate": None,
            "passed": False,
            "failed_checks": ["manifest_error"],
            "error": str(exc),
        }
    else:
        return {
            "manifest_path": str(result.manifest_path),
            "source_type": result.source_type,
            "latest_iteration": result.latest_iteration,
            "latest_benchmark_win_rate": result.latest_benchmark_win_rate,
            "latest_benchmark_games": result.iterations[-1].benchmark_games if result.iterations else 0,
            "latest_collection_capped_rate": result.latest_collection_capped_rate,
            "latest_benchmark_capped_rate": result.latest_benchmark_capped_rate,
            "passed": result.passed,
            "failed_checks": [check.name for check in result.checks if not check.passed],
        }


def _audit_config_report_checks(
    config_payload: Mapping[str, object],
    *,
    preflight_passed: bool | None,
    require_source: bool,
    require_calibration: bool,
) -> list[dict[str, object]]:
    checks: list[dict[str, object]] = [
        {
            "name": "config_loadable",
            "passed": True,
            "observed": config_payload.get("schema_version"),
            "threshold": RUN_AUDIT_CONFIG_SCHEMA_VERSION,
            "message": "audit config loaded successfully",
        }
    ]
    source_present = config_payload.get("source") is not None
    calibration_present = config_payload.get("calibration") is not None
    if require_source:
        checks.append(
            {
                "name": "source_metadata_present",
                "passed": source_present,
                "observed": source_present,
                "threshold": True,
                "message": "source provenance metadata is required",
            }
        )
    if require_calibration:
        checks.append(
            {
                "name": "calibration_metadata_present",
                "passed": calibration_present,
                "observed": calibration_present,
                "threshold": True,
                "message": "calibration metadata is required",
            }
        )
    if preflight_passed is not None:
        checks.append(
            {
                "name": "preflight_audit_passed",
                "passed": preflight_passed is True,
                "observed": preflight_passed,
                "threshold": True,
                "message": "all supplied manifests must pass this audit config",
            }
        )
    return checks


def _print_audit_config_report(report: Mapping[str, object]) -> None:
    print("audit_config_report:")
    print(f"passed: {'PASS' if report['passed'] else 'FAIL'}")
    print(f"config: {report['audit_config_path']}")
    print(f"schema_version: {report['schema_version']}")
    print(f"source_metadata: {'present' if report['source'] is not None else 'missing'}")
    print(f"calibration_metadata: {'present' if report['calibration'] is not None else 'missing'}")
    source = report.get("source")
    if isinstance(source, Mapping):
        print(f"source_branch: {_format_summary_value(source.get('branch'))}")
        print(f"source_head: {_format_summary_value(source.get('head'))}")
        print(f"source_dirty: {_format_summary_value(source.get('dirty'))}")
    calibration = report.get("calibration")
    if isinstance(calibration, Mapping):
        print(f"calibration_source_type: {_format_summary_value(calibration.get('source_type'))}")
        print(f"calibration_run_count: {_format_summary_value(calibration.get('run_count'))}")
        print(f"calibration_benchmark_iterations: {_format_summary_value(calibration.get('benchmark_iteration_count'))}")
        print(f"calibration_aggregate_mode: {_format_summary_value(calibration.get('aggregate_mode'))}")
        calibration_paths = calibration.get("paths")
        if isinstance(calibration_paths, list):
            print("calibration_paths:")
            for path in calibration_paths:
                print(f"- {path}")
    print("thresholds:")
    config = report.get("config")
    if isinstance(config, Mapping):
        for key, value in config.items():
            print(f"- {key}: {_format_summary_value(value)}")
    preflight_passed = report.get("preflight_passed")
    if preflight_passed is None:
        print("preflight: not_requested")
    else:
        print(f"preflight: {'PASS' if preflight_passed else 'FAIL'}")
        print("preflight_runs:")
        for run in report.get("preflight_runs", ()):
            if not isinstance(run, Mapping):
                continue
            run_status = "PASS" if run.get("passed") else "FAIL"
            failed_checks = run.get("failed_checks")
            failed_summary = ", ".join(str(check) for check in failed_checks) if failed_checks else "-"
            print(
                f"- {run_status} {run.get('manifest_path')} "
                f"latest_iteration={_format_summary_value(run.get('latest_iteration'))} "
                f"latest_wr={_format_optional_float(run.get('latest_benchmark_win_rate'))} "
                f"bench_games={_format_summary_value(run.get('latest_benchmark_games'))} "
                f"failed_checks={failed_summary}"
            )
            if run.get("error") is not None:
                print(f"  error: {run.get('error')}")
    print("checks:")
    for check in report.get("checks", ()):
        if not isinstance(check, Mapping):
            continue
        status = "pass" if check.get("passed") else "fail"
        print(
            f"- {status} {check.get('name')}: "
            f"observed={_format_summary_value(check.get('observed'))} "
            f"threshold={_format_summary_value(check.get('threshold'))}"
        )


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
    if args.fail_on_profile and args.compare_profile is None:
        raise ValueError("--fail-on-profile requires --compare-profile.")
    paths = _expanded_manifest_paths(args.paths, args.manifest_glob)
    result = (
        calibrate_run_audit(paths[0], margin=args.margin)
        if len(paths) == 1
        else calibrate_run_audits(paths, margin=args.margin, aggregate_mode=args.aggregate_mode)
    )
    profile_audit = None
    if args.compare_profile is not None:
        profile = evaluation_profile(args.compare_profile)
        profile_audit = _profile_audit_payload(paths, profile_name=profile.name, config=profile.audit_config)
    sufficiency_requested = (
        args.require_run_count > 0
        or args.require_benchmark_iterations > 0
        or args.require_min_benchmark_games > 0
    )
    sufficiency_errors = _calibration_sufficiency_errors(
        result,
        require_run_count=args.require_run_count,
        require_benchmark_iterations=args.require_benchmark_iterations,
        require_min_benchmark_games=args.require_min_benchmark_games,
    )
    profile_failed = bool(profile_audit is not None and not profile_audit["passed"])
    wrote_config_path = None
    if args.write_config is not None:
        if not sufficiency_requested:
            raise ValueError(
                "--write-config requires at least one calibration sufficiency requirement "
                "(--require-run-count, --require-benchmark-iterations, or --require-min-benchmark-games)."
            )
        if sufficiency_errors:
            raise ValueError("--write-config requires calibration sufficiency checks to pass.")
        if profile_failed:
            raise ValueError("--write-config requires the selected profile audit to pass.")
        config_payload = _audit_calibration_config_payload(result)
        _write_json_payload(args.write_config, config_payload)
        wrote_config_path = args.write_config
    if args.json:
        payload = result.to_dict()
        if profile_audit is not None:
            payload["profile_audit"] = profile_audit
        if sufficiency_requested:
            payload["calibration_sufficient"] = not sufficiency_errors
            payload["calibration_sufficiency_errors"] = list(sufficiency_errors)
        if wrote_config_path is not None:
            payload["written_config_path"] = str(wrote_config_path)
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        _print_audit_calibration(result)
        if profile_audit is not None:
            _print_profile_audit(profile_audit)
        if sufficiency_requested:
            _print_calibration_sufficiency(sufficiency_errors)
        if wrote_config_path is not None:
            print(f"written_config: {wrote_config_path}")
    return 2 if sufficiency_errors or (args.fail_on_profile and profile_failed) else 0


def _expanded_manifest_paths(paths: Iterable[Path], manifest_globs: Iterable[str] | None) -> tuple[Path, ...]:
    expanded: list[Path] = []
    seen: set[str] = set()
    unmatched_patterns: list[str] = []
    for path in paths:
        _append_discovered_manifest_path(expanded, seen, path)
    for pattern in manifest_globs or ():
        expanded_pattern = str(Path(pattern).expanduser())
        matches = tuple(Path(match) for match in sorted(glob.glob(expanded_pattern, recursive=True)))
        if not matches:
            unmatched_patterns.append(pattern)
            continue
        for match in matches:
            _append_discovered_manifest_path(expanded, seen, match)
    if not expanded:
        if unmatched_patterns:
            raise ValueError("--manifest-glob matched no paths: " + ", ".join(unmatched_patterns))
        raise ValueError("provide at least one path or --manifest-glob.")
    if unmatched_patterns:
        print(
            "warning: --manifest-glob matched no paths: " + ", ".join(unmatched_patterns),
            file=sys.stderr,
        )
    return tuple(expanded)


def _append_discovered_manifest_path(expanded: list[Path], seen: set[str], path: Path) -> None:
    key = _manifest_identity_key(path)
    if key in seen:
        return
    seen.add(key)
    expanded.append(path)


def _manifest_identity_key(path: Path) -> str:
    expanded = path.expanduser()
    if expanded.exists() and expanded.is_dir():
        expanded = expanded / "manifest.json"
    return str(expanded.resolve(strict=False))


def _audit_calibration_config_payload(result) -> dict[str, object]:
    config = run_audit_config_from_dict(result.suggested_config())
    calibration_paths = (
        tuple(result.paths)
        if hasattr(result, "paths")
        else (result.manifest_path,)
    )
    return run_audit_config_payload(
        config,
        source=collect_source_metadata(),
        calibration={
            "margin": result.margin,
            "source_type": result.source_type,
            "run_count": getattr(result, "run_count", 1),
            "iteration_count": result.iteration_count,
            "benchmark_iteration_count": result.benchmark_iteration_count,
            "aggregate_mode": getattr(result, "aggregate_mode", None),
            "paths": [str(path) for path in calibration_paths],
            "notes": list(getattr(result, "notes", ())),
        },
    )


def _cpu_smoke_plan(args: argparse.Namespace) -> int:
    _validate_cpu_smoke_args(args)
    recipe = _cpu_smoke_recipe(args)
    if args.json:
        print(json.dumps(recipe, indent=2, sort_keys=True))
        return 0
    print("cpu_smoke_plan:")
    print("purpose: tiny CPU-only bootstrap/self-play plumbing validation")
    print("note: smoke-profile thresholds validate command flow, not policy strength.")
    print("note: pass --showdown-root or set the normal Showdown-root environment before running when needed.")
    print("commands:")
    for index, step in enumerate(recipe["steps"], start=1):
        print(f"{index}. {step['name']}")
        print(_shell_join(step["argv"]))
    return 0


def _cpu_smoke_run(args: argparse.Namespace) -> int:
    _validate_cpu_smoke_args(args, validate_showdown_root=True)
    recipe = _cpu_smoke_recipe(args)
    summary_path = args.summary_path if args.summary_path is not None else args.run_root / "cpu-smoke-run-summary.json"
    return _run_recipe_with_summary(
        recipe=recipe,
        summary_path=summary_path,
        schema_version=CPU_SMOKE_RUN_SUMMARY_SCHEMA_VERSION,
        command_name="cpu_smoke_run",
        purpose="tiny CPU-only bootstrap/self-play plumbing validation",
        notes=(
            "smoke-profile thresholds validate command flow, not policy strength.",
            "use a fresh --run-root; this command does not delete existing artifacts.",
        ),
        failure_label="cpu smoke",
        summary_label="cpu smoke",
    )


def _cpu_smoke_report(args: argparse.Namespace) -> int:
    summary_path, summary = _load_cpu_smoke_summary(args.path)
    status = str(summary.get("status", "unknown"))
    recipe = summary.get("recipe") if isinstance(summary.get("recipe"), Mapping) else {}
    teacher_branch_preflight = _cpu_smoke_teacher_branch_preflight_report(recipe)
    if args.json:
        payload = dict(summary)
        payload["summary_source_path"] = str(summary_path)
        payload["teacher_branch_preflight_report"] = teacher_branch_preflight
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0 if status == "passed" else 2
    print("cpu_smoke_report:")
    print(f"summary: {summary_path}")
    print(f"status: {_status_label(status)}")
    print(f"started_at: {_format_summary_value(summary.get('started_at'))}")
    print(f"ended_at: {_format_summary_value(summary.get('ended_at'))}")
    print(f"duration_seconds: {_format_summary_value(summary.get('duration_seconds'))}")
    source = summary.get("source")
    if isinstance(source, Mapping):
        print(f"source_available: {_format_summary_value(source.get('available'))}")
        print(f"source_branch: {_format_summary_value(source.get('branch'))}")
        print(f"source_head: {_format_summary_value(source.get('head'))}")
        print(f"source_dirty: {_format_summary_value(source.get('dirty'))}")
    _print_cpu_smoke_teacher_branch_preflight_report(teacher_branch_preflight)
    failed_step = summary.get("failed_step")
    if isinstance(failed_step, dict):
        print(
            "failed_step: "
            f"{failed_step.get('index')} {failed_step.get('name')} returncode={failed_step.get('returncode')}"
        )
    else:
        print("failed_step: -")
    steps = summary.get("steps")
    if isinstance(steps, list):
        print("steps:")
        for step in steps:
            if not isinstance(step, dict):
                continue
            print(
                f"- {step.get('index')}: {_status_label(str(step.get('status', 'unknown')))} "
                f"{step.get('name')} returncode={_format_summary_value(step.get('returncode'))} "
                f"duration={_format_summary_value(step.get('duration_seconds'))}"
            )
    return 0 if status == "passed" else 2


def _cpu_smoke_teacher_branch_preflight_report(recipe: Mapping[str, object]) -> dict[str, object]:
    requested = recipe.get("teacher_branch_preflight_requested") is True
    path_value = recipe.get("teacher_branch_preflight_output_path")
    report: dict[str, object] = {
        "requested": requested,
        "path": None if path_value is None else str(path_value),
        "available": False,
        "passed": None,
        "schema_version": None,
        "teacher_branch_counts": {},
        "failed_checks": [],
        "error": None,
        "required_teacher_branches": list(recipe.get("required_teacher_branches") or ()),
        "min_teacher_branch_counts": list(recipe.get("min_teacher_branch_counts") or ()),
    }
    if not requested:
        return report
    if path_value is None:
        report["error"] = "teacher_branch_preflight_output_path missing from recipe"
        return report
    path = Path(str(path_value))
    if not path.exists():
        report["error"] = "teacher branch preflight artifact not found"
        return report
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        report["error"] = f"failed to read teacher branch preflight artifact: {exc}"
        return report
    if not isinstance(payload, dict):
        report["error"] = "teacher branch preflight artifact must be a JSON object"
        return report
    report["available"] = True
    report["passed"] = payload.get("passed") if isinstance(payload.get("passed"), bool) else None
    schema_version = payload.get("schema_version")
    report["schema_version"] = schema_version if isinstance(schema_version, str) else None

    teacher_summary = payload.get("teacher_decision_summary")
    if isinstance(teacher_summary, Mapping):
        counts = teacher_summary.get("teacher_branch_counts")
        if isinstance(counts, Mapping):
            report["teacher_branch_counts"] = {
                str(branch): count
                for branch, count in sorted(counts.items(), key=lambda item: str(item[0]))
                if isinstance(count, int)
            }

    checks = payload.get("checks")
    failed_checks: list[dict[str, object]] = []
    if isinstance(checks, list):
        for check in checks:
            if not isinstance(check, Mapping) or check.get("passed") is True:
                continue
            failed_checks.append(
                {
                    key: value
                    for key, value in check.items()
                    if key in {"name", "message", "observed", "threshold", "passed"}
                }
            )
    report["failed_checks"] = failed_checks
    return report


def _print_cpu_smoke_teacher_branch_preflight_report(report: Mapping[str, object]) -> None:
    if report.get("requested") is not True:
        print("teacher_branch_preflight: not_requested")
        return
    status = "UNKNOWN"
    if report.get("available") is not True:
        status = "MISSING"
    elif report.get("passed") is True:
        status = "PASS"
    elif report.get("passed") is False:
        status = "FAIL"
    print(f"teacher_branch_preflight: {status}")
    print(f"teacher_branch_preflight_path: {_format_summary_value(report.get('path'))}")
    error = report.get("error")
    if error:
        print(f"teacher_branch_preflight_error: {error}")
    counts = report.get("teacher_branch_counts")
    if isinstance(counts, Mapping) and counts:
        print("teacher_branch_counts:")
        for branch, count in sorted(counts.items(), key=lambda item: str(item[0])):
            print(f"- {branch}: {count}")
    failed_checks = report.get("failed_checks")
    if isinstance(failed_checks, list) and failed_checks:
        print("teacher_branch_failed_checks:")
        for check in failed_checks:
            if not isinstance(check, Mapping):
                continue
            name = _format_summary_value(check.get("name"))
            message = _format_summary_value(check.get("message"))
            observed = _format_summary_value(check.get("observed"))
            threshold = _format_summary_value(check.get("threshold"))
            print(f"- {name}: observed={observed} threshold={threshold} message={message}")


def _cpu_pilot_plan(args: argparse.Namespace) -> int:
    _validate_cpu_pilot_args(args)
    recipe = _cpu_pilot_recipe(args)
    if args.json:
        print(json.dumps(recipe, indent=2, sort_keys=True))
        return 0
    print("cpu_pilot_plan:")
    print("purpose: CPU-only pilot suite for threshold calibration evidence")
    print("note: runs multiple seeded smoke pilots, then compares and calibrates their manifests.")
    print("note: pass --showdown-root or set the normal Showdown-root environment before running when needed.")
    print("commands:")
    for index, step in enumerate(recipe["steps"], start=1):
        print(f"{index}. {step['name']}")
        print(_shell_join(step["argv"]))
    return 0


def _cpu_pilot_run(args: argparse.Namespace) -> int:
    _validate_cpu_pilot_args(args, validate_showdown_root=True)
    recipe = _cpu_pilot_recipe(args)
    summary_path = args.summary_path if args.summary_path is not None else args.run_root / "cpu-pilot-suite-summary.json"
    return _run_recipe_with_summary(
        recipe=recipe,
        summary_path=summary_path,
        schema_version=CPU_PILOT_SUITE_SUMMARY_SCHEMA_VERSION,
        command_name="cpu_pilot_run",
        purpose="CPU-only pilot suite for threshold calibration evidence",
        notes=("use a fresh --run-root; this command does not delete existing artifacts.",),
        failure_label="cpu pilot",
        summary_label="cpu pilot",
    )


def _run_recipe_with_summary(
    *,
    recipe: dict[str, object],
    summary_path: Path,
    schema_version: str,
    command_name: str,
    purpose: str,
    notes: tuple[str, ...],
    failure_label: str,
    summary_label: str,
) -> int:
    run_started_monotonic = time.perf_counter()
    step_summaries: list[dict[str, object]] = []
    summary: dict[str, object] = {
        "schema_version": schema_version,
        "status": "running",
        "summary_path": str(summary_path),
        "started_at": _utc_timestamp(),
        "ended_at": None,
        "duration_seconds": None,
        "source": recipe["source"],
        "recipe": recipe,
        "steps": step_summaries,
        "failed_step": None,
    }
    _write_json_payload(summary_path, summary)
    summary_update_failed = False
    print(f"{command_name}:")
    print(f"purpose: {purpose}")
    for note in notes:
        print(f"note: {note}")
    print(f"summary: {summary_path}")
    for index, step in enumerate(recipe["steps"], start=1):
        print(f"running_step: {index}/{len(recipe['steps'])} {step['name']}", flush=True)
        print(_shell_join(step["argv"]), flush=True)
        output_json_path = _step_output_json_path(step)
        step_started_monotonic = time.perf_counter()
        step_summary: dict[str, object] = {
            "index": index,
            "name": step["name"],
            "argv": step["argv"],
            "command": step["command"],
            "status": "running",
            "started_at": _utc_timestamp(),
            "ended_at": None,
            "duration_seconds": None,
            "returncode": None,
        }
        if output_json_path is not None:
            step_summary.update(
                {
                    "output_json_path": str(output_json_path),
                    "output_json_written": False,
                    "output_json_valid": False,
                }
            )
        step_summaries.append(step_summary)
        summary_update_failed = _write_run_summary_update(
            summary_path,
            summary,
            summary_label=summary_label,
            previous_failure=summary_update_failed,
        )
        if output_json_path is None:
            completed = subprocess.run(step["argv"])
            output_json_error = None
        else:
            completed = subprocess.run(step["argv"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            output_json_error = _persist_step_json_output(output_json_path, completed, step_summary)
        step_summary["ended_at"] = _utc_timestamp()
        step_summary["duration_seconds"] = round(time.perf_counter() - step_started_monotonic, 6)
        step_summary["returncode"] = int(completed.returncode)
        if completed.returncode == 0 and output_json_error is not None:
            step_summary["returncode"] = 70
            step_summary["output_json_error"] = output_json_error
        if step_summary["returncode"] != 0:
            step_summary["status"] = "failed"
            summary["status"] = "failed"
            summary["failed_step"] = {
                "index": index,
                "name": step["name"],
                "returncode": int(step_summary["returncode"]),
            }
            summary["ended_at"] = _utc_timestamp()
            summary["duration_seconds"] = round(time.perf_counter() - run_started_monotonic, 6)
            _write_run_summary_update(
                summary_path,
                summary,
                summary_label=summary_label,
                previous_failure=summary_update_failed,
            )
            print(
                f"error: {failure_label} step {index} failed with exit code {step_summary['returncode']}: {step['name']}",
                file=sys.stderr,
            )
            if output_json_error is not None:
                print(f"error: {output_json_error}", file=sys.stderr)
            return int(step_summary["returncode"])
        step_summary["status"] = "passed"
        summary_update_failed = _write_run_summary_update(
            summary_path,
            summary,
            summary_label=summary_label,
            previous_failure=summary_update_failed,
        )
    summary["status"] = "passed"
    summary["ended_at"] = _utc_timestamp()
    summary["duration_seconds"] = round(time.perf_counter() - run_started_monotonic, 6)
    _write_run_summary_update(
        summary_path,
        summary,
        summary_label=summary_label,
        previous_failure=summary_update_failed,
    )
    print(f"{command_name}: PASS")
    return 0


def _cpu_pilot_report(args: argparse.Namespace) -> int:
    summary_path, summary = _load_cpu_pilot_summary(args.path)
    status = str(summary.get("status", "unknown"))
    recipe = summary.get("recipe")
    artifact_report = (
        _cpu_pilot_artifact_report(summary, recipe)
        if isinstance(recipe, Mapping)
        else None
    )
    exit_code = _cpu_pilot_report_exit_code(status, artifact_report, require_ready=args.require_ready)
    if args.json:
        payload = dict(summary)
        payload["summary_source_path"] = str(summary_path)
        payload["pilot_artifact_report"] = artifact_report
        print(json.dumps(payload, indent=2, sort_keys=True))
        return exit_code
    print("cpu_pilot_report:")
    print(f"summary: {summary_path}")
    print(f"status: {_status_label(status)}")
    print(f"started_at: {_format_summary_value(summary.get('started_at'))}")
    print(f"ended_at: {_format_summary_value(summary.get('ended_at'))}")
    print(f"duration_seconds: {_format_summary_value(summary.get('duration_seconds'))}")
    if isinstance(recipe, Mapping) and artifact_report is not None:
        calibration_artifact = artifact_report["calibration"]
        replay_artifact = artifact_report["replay"]
        print(f"pilot_count: {_format_summary_value(recipe.get('pilot_count'))}")
        print(f"manifest_glob: {_format_summary_value(recipe.get('manifest_glob'))}")
        print(f"audit_config_path: {_format_summary_value(recipe.get('audit_config_path'))}")
        print(f"calibration_output_path: {_format_summary_value(recipe.get('calibration_output_path'))}")
        print(f"replay_output_path: {_format_summary_value(recipe.get('replay_output_path'))}")
        print(f"calibration_sufficient: {_format_summary_value(calibration_artifact.get('sufficient'))}")
        print(
            "calibration_written_audit_config_path: "
            f"{_format_summary_value(calibration_artifact.get('written_audit_config_path'))}"
        )
        print(
            "calibration_audit_config_write_error: "
            f"{_format_summary_value(calibration_artifact.get('audit_config_write_error'))}"
        )
        print(f"replay_audit_failed: {_format_summary_value(replay_artifact.get('audit_failed'))}")
        print(f"replay_failed_check_count: {_format_summary_value(replay_artifact.get('failed_check_count'))}")
        print(f"audit_config_ready: {_format_optional_bool(artifact_report['audit_config_ready'])}")
        reasons = artifact_report["audit_config_ready_reasons"]
        if isinstance(reasons, list) and reasons:
            print("audit_config_ready_reasons:")
            for reason in reasons:
                print(f"- {reason}")
    failed_step = summary.get("failed_step")
    if isinstance(failed_step, dict):
        print(
            "failed_step: "
            f"{failed_step.get('index')} {failed_step.get('name')} returncode={failed_step.get('returncode')}"
        )
    else:
        print("failed_step: -")
    steps = summary.get("steps")
    if isinstance(steps, list):
        print("steps:")
        for step in steps:
            if not isinstance(step, dict):
                continue
            print(
                f"- {step.get('index')}: {_status_label(str(step.get('status', 'unknown')))} "
                f"{step.get('name')} returncode={_format_summary_value(step.get('returncode'))} "
                f"duration={_format_summary_value(step.get('duration_seconds'))}"
            )
    return exit_code


def _cpu_pilot_report_exit_code(
    status: str,
    artifact_report: Mapping[str, object] | None,
    *,
    require_ready: bool,
) -> int:
    if status != "passed":
        return 2
    if require_ready and (artifact_report is None or artifact_report.get("audit_config_ready") is not True):
        return 2
    return 0


def _step_output_json_path(step: Mapping[str, object]) -> Path | None:
    output_path = step.get("output_json_path")
    if output_path is None:
        return None
    return Path(str(output_path))


def _persist_step_json_output(
    path: Path,
    completed: subprocess.CompletedProcess[str],
    step_summary: dict[str, object],
) -> str | None:
    stdout_text = getattr(completed, "stdout", None) or ""
    stderr_text = getattr(completed, "stderr", None) or ""
    if stdout_text:
        print(stdout_text, end="" if stdout_text.endswith("\n") else "\n")
    if stderr_text:
        print(stderr_text, end="" if stderr_text.endswith("\n") else "\n", file=sys.stderr)
    if not stdout_text.strip():
        return f"expected JSON stdout for artifact step but received no output: {path}"
    try:
        json.loads(stdout_text)
    except json.JSONDecodeError as exc:
        try:
            _write_text_payload(path, stdout_text)
            step_summary["output_json_written"] = True
        except OSError as write_exc:
            return f"failed to write invalid JSON stdout artifact {path}: {write_exc}"
        return f"expected valid JSON stdout for artifact step {path}: {exc}"
    try:
        _write_text_payload(path, stdout_text)
    except OSError as exc:
        return f"failed to write JSON stdout artifact {path}: {exc}"
    step_summary["output_json_written"] = True
    step_summary["output_json_valid"] = True
    return None


def _load_pilot_report_json_summary(path_value: object) -> dict[str, object] | None:
    if path_value is None:
        return None
    path = Path(str(path_value))
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def _cpu_pilot_artifact_report(summary: Mapping[str, object], recipe: Mapping[str, object]) -> dict[str, object]:
    calibration_path = recipe.get("calibration_output_path")
    replay_path = recipe.get("replay_output_path")
    calibration_summary = _load_pilot_report_json_summary(calibration_path)
    replay_summary = _load_pilot_report_json_summary(replay_path)
    calibration = {
        "path": None if calibration_path is None else str(calibration_path),
        "available": calibration_summary is not None,
        "sufficient": None,
        "written_audit_config_path": None,
        "expected_audit_config_path": None if recipe.get("audit_config_path") is None else str(recipe.get("audit_config_path")),
        "audit_config_write_error": None,
    }
    if calibration_summary is not None:
        calibration.update(
            {
                "sufficient": calibration_summary.get("audit_calibration_sufficient"),
                "written_audit_config_path": calibration_summary.get("written_audit_config_path"),
                "audit_config_write_error": calibration_summary.get("audit_config_write_error"),
            }
        )
    replay = {
        "path": None if replay_path is None else str(replay_path),
        "available": replay_summary is not None,
        "audit_failed": None,
        "failed_check_count": None,
    }
    if replay_summary is not None:
        replay.update(
            {
                "audit_failed": replay_summary.get("audit_failed"),
                "failed_check_count": _comparison_failed_check_count(replay_summary),
            }
        )
    reasons = _cpu_pilot_audit_config_not_ready_reasons(summary, calibration, replay)
    return {
        "calibration": calibration,
        "replay": replay,
        "audit_config_ready": not reasons,
        "audit_config_ready_reasons": reasons,
    }


def _cpu_pilot_audit_config_not_ready_reasons(
    summary: Mapping[str, object],
    calibration: Mapping[str, object],
    replay: Mapping[str, object],
) -> list[str]:
    reasons: list[str] = []
    if summary.get("status") != "passed":
        reasons.append("suite_status_not_passed")
    if calibration.get("available") is not True:
        reasons.append("calibration_artifact_missing")
    else:
        if calibration.get("sufficient") is not True:
            reasons.append("calibration_not_sufficient")
        if not calibration.get("written_audit_config_path"):
            reasons.append("calibrated_audit_config_not_written")
        elif (
            calibration.get("expected_audit_config_path")
            and str(calibration.get("written_audit_config_path")) != str(calibration.get("expected_audit_config_path"))
        ):
            reasons.append("calibrated_audit_config_path_mismatch")
        if calibration.get("audit_config_write_error"):
            reasons.append("calibrated_audit_config_write_error")
    if replay.get("available") is not True:
        reasons.append("replay_artifact_missing")
    elif replay.get("audit_failed") is not False:
        reasons.append("replay_audit_failed")
    elif replay.get("failed_check_count") not in (0, None):
        reasons.append("replay_failed_checks_present")
    return reasons


def _comparison_failed_check_count(payload: Mapping[str, object]) -> int:
    failed_count = 0
    entries = payload.get("entries")
    if not isinstance(entries, list):
        return failed_count
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        failed_checks = entry.get("audit_failed_checks")
        if isinstance(failed_checks, list):
            failed_count += len(failed_checks)
    return failed_count


def _cpu_pilot_recipe(args: argparse.Namespace) -> dict[str, object]:
    run_root = args.run_root
    audit_config_path = args.audit_config_path if args.audit_config_path is not None else run_root / "pilot-audit-config.json"
    calibration_output_path = run_root / "pilot-calibration-compare.json"
    replay_output_path = run_root / "pilot-audit-replay.json"
    manifest_glob = run_root / "pilot-*" / "selfplay" / "manifest.json"
    python_binary = args.python_binary
    steps: list[dict[str, object]] = []
    for index in range(args.pilot_count):
        pilot_index = index + 1
        pilot_root = run_root / f"pilot-{pilot_index:04d}"
        pilot_seed_start = args.seed_start + (index * args.seed_stride)
        steps.append(
            {
                "name": f"run CPU smoke pilot {pilot_index}",
                "argv": _cpu_pilot_smoke_run_argv(
                    args,
                    pilot_root=pilot_root,
                    seed_start=pilot_seed_start,
                ),
            }
        )
    benchmark_iterations_required = args.pilot_count * args.selfplay_iterations
    steps.append(
        {
            "name": "compare pilots and write calibrated audit config",
            "argv": [
                python_binary,
                "-m",
                "pokezero.eval_cli",
                "compare",
                "--manifest-glob",
                str(manifest_glob),
                "--suggest-audit-calibration",
                "--calibration-aggregate-mode",
                "envelope",
                "--calibration-require-run-count",
                str(args.pilot_count),
                "--calibration-require-benchmark-iterations",
                str(benchmark_iterations_required),
                "--calibration-require-min-benchmark-games",
                str(args.calibration_require_min_benchmark_games),
                "--write-audit-config",
                str(audit_config_path),
                "--json",
            ],
            "output_json_path": str(calibration_output_path),
        }
    )
    steps.append(
        {
            "name": "compare pilots with calibrated audit config",
            "argv": [
                python_binary,
                "-m",
                "pokezero.eval_cli",
                "compare",
                "--manifest-glob",
                str(manifest_glob),
                "--audit-config",
                str(audit_config_path),
                "--fail-on-audit",
                "--json",
            ],
            "output_json_path": str(replay_output_path),
        }
    )
    return {
        "purpose": "CPU-only pilot suite for threshold calibration evidence",
        "warning": "pilot-suite thresholds are starting evidence, not proof of policy strength",
        "source": collect_source_metadata(),
        "run_root": str(run_root),
        "python_binary": python_binary,
        "showdown_root": None if args.showdown_root is None else str(args.showdown_root),
        "pilot_count": args.pilot_count,
        "seed_start": args.seed_start,
        "seed_stride": args.seed_stride,
        "manifest_glob": str(manifest_glob),
        "audit_config_path": str(audit_config_path),
        "calibration_output_path": str(calibration_output_path),
        "replay_output_path": str(replay_output_path),
        "benchmark_iterations_required": benchmark_iterations_required,
        "calibration_require_min_benchmark_games": args.calibration_require_min_benchmark_games,
        "teacher_branch_preflight_requested": _teacher_branch_preflight_requested(args),
        "teacher_branch_preflight_games": args.teacher_branch_preflight_games,
        "required_teacher_branches": list(args.require_teacher_branch or ()),
        "min_teacher_branch_counts": list(args.min_teacher_branch_count or ()),
        "steps": [
            {
                "name": step["name"],
                "argv": step["argv"],
                "command": _shell_join(step["argv"]),
                **({"output_json_path": step["output_json_path"]} if "output_json_path" in step else {}),
            }
            for step in steps
        ],
    }


def _cpu_pilot_smoke_run_argv(args: argparse.Namespace, *, pilot_root: Path, seed_start: int) -> list[str]:
    argv = [
        args.python_binary,
        "-m",
        "pokezero.eval_cli",
        "cpu-smoke-run",
        "--run-root",
        str(pilot_root),
        "--python-binary",
        args.python_binary,
        "--workers",
        str(args.workers),
        "--train-games",
        str(args.train_games),
        "--validation-games",
        str(args.validation_games),
        "--bootstrap-benchmark-games",
        str(args.bootstrap_benchmark_games),
        "--teacher-branch-preflight-games",
        str(args.teacher_branch_preflight_games),
        "--selfplay-iterations",
        str(args.selfplay_iterations),
        "--selfplay-games",
        str(args.selfplay_games),
        "--evaluation-games",
        str(args.evaluation_games),
        "--feature-count",
        str(args.feature_count),
        "--window-size",
        str(args.window_size),
        "--max-decision-rounds",
        str(args.max_decision_rounds),
        "--seed-start",
        str(seed_start),
        "--audit-config-path",
        str(pilot_root / "smoke-audit-config.json"),
    ]
    argv.extend(_teacher_branch_gate_args(args))
    if args.showdown_root is not None:
        argv.extend(["--showdown-root", str(args.showdown_root)])
    return argv


def _load_cpu_smoke_summary(path: Path) -> tuple[Path, dict[str, object]]:
    summary_path = (
        path / "cpu-smoke-run-summary.json"
        if path.is_dir() or (not path.exists() and path.suffix != ".json")
        else path
    )
    if not summary_path.exists():
        raise FileNotFoundError(f"cpu smoke summary not found: {summary_path}")
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"cpu smoke summary must be a JSON object: {summary_path}")
    if payload.get("schema_version") != CPU_SMOKE_RUN_SUMMARY_SCHEMA_VERSION:
        raise ValueError(
            "Unsupported cpu smoke summary schema: "
            f"{payload.get('schema_version')!r}; expected {CPU_SMOKE_RUN_SUMMARY_SCHEMA_VERSION!r}."
        )
    return summary_path, payload


def _load_cpu_pilot_summary(path: Path) -> tuple[Path, dict[str, object]]:
    summary_path = (
        path / "cpu-pilot-suite-summary.json"
        if path.is_dir() or (not path.exists() and path.suffix != ".json")
        else path
    )
    if not summary_path.exists():
        raise FileNotFoundError(f"cpu pilot summary not found: {summary_path}")
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"cpu pilot summary must be a JSON object: {summary_path}")
    if payload.get("schema_version") != CPU_PILOT_SUITE_SUMMARY_SCHEMA_VERSION:
        raise ValueError(
            "Unsupported cpu pilot summary schema: "
            f"{payload.get('schema_version')!r}; expected {CPU_PILOT_SUITE_SUMMARY_SCHEMA_VERSION!r}."
        )
    return summary_path, payload


def _status_label(status: str) -> str:
    if status == "passed":
        return "PASS"
    if status == "failed":
        return "FAIL"
    if status == "running":
        return "RUNNING"
    return status.upper() if status else "-"


def _format_summary_value(value: object) -> str:
    return "-" if value is None else str(value)


def _validate_cpu_smoke_args(args: argparse.Namespace, *, validate_showdown_root: bool = False) -> None:
    positive_fields = (
        "workers",
        "train_games",
        "validation_games",
        "bootstrap_benchmark_games",
        "selfplay_iterations",
        "selfplay_games",
        "evaluation_games",
        "feature_count",
        "window_size",
    )
    for field_name in positive_fields:
        if getattr(args, field_name) <= 0:
            raise ValueError(f"{field_name.replace('_', '-')} must be positive.")
    if args.max_decision_rounds <= 0:
        raise ValueError("max-decision-rounds must be positive.")
    if _teacher_branch_preflight_requested(args) and args.teacher_branch_preflight_games <= 0:
        raise ValueError("teacher-branch-preflight-games must be positive when teacher branch gates are requested.")
    if validate_showdown_root and args.showdown_root is not None and not args.showdown_root.exists():
        raise ValueError(f"showdown-root does not exist: {args.showdown_root}")


def _teacher_branch_preflight_requested(args: argparse.Namespace) -> bool:
    return bool(args.require_teacher_branch or args.min_teacher_branch_count)


def _teacher_branch_gate_args(args: argparse.Namespace) -> list[str]:
    argv: list[str] = []
    for branch in args.require_teacher_branch or ():
        argv.extend(["--require-teacher-branch", str(branch)])
    for branch_count in args.min_teacher_branch_count or ():
        argv.extend(["--min-teacher-branch-count", str(branch_count)])
    return argv


def _validate_cpu_pilot_args(args: argparse.Namespace, *, validate_showdown_root: bool = False) -> None:
    _validate_cpu_smoke_args(args, validate_showdown_root=validate_showdown_root)
    if args.pilot_count <= 0:
        raise ValueError("pilot-count must be positive.")
    if args.seed_stride <= 0:
        raise ValueError("seed-stride must be positive.")
    if args.pilot_count > 1 and (args.pilot_count - 1) * args.seed_stride >= CPU_SMOKE_SEED_BAND_SPACING:
        raise ValueError(
            "pilot seed offsets must stay below the smoke seed-band spacing; "
            f"reduce pilot-count or seed-stride so (pilot-count - 1) * seed-stride < {CPU_SMOKE_SEED_BAND_SPACING}."
        )
    if args.calibration_require_min_benchmark_games <= 0:
        raise ValueError("calibration-require-min-benchmark-games must be positive.")


def _cpu_smoke_recipe(args: argparse.Namespace) -> dict[str, object]:
    run_root = args.run_root
    teacher_dir = run_root / "teacher-bootstrap"
    selfplay_dir = run_root / "selfplay"
    promotion_registry = run_root / "promotions.json"
    promotion_artifact_dir = run_root / "promoted-checkpoints"
    audit_config_path = args.audit_config_path if args.audit_config_path is not None else run_root / "smoke-audit-config.json"
    teacher_branch_preflight_output_path = run_root / "teacher-branch-preflight.json"
    python_binary = args.python_binary
    showdown_root = None if args.showdown_root is None else str(args.showdown_root)
    showdown_root_args = () if showdown_root is None else ("--showdown-root", showdown_root)
    seed_start = int(args.seed_start)
    validation_seed_start = seed_start + CPU_SMOKE_SEED_BAND_SPACING
    bootstrap_benchmark_seed_start = seed_start + (2 * CPU_SMOKE_SEED_BAND_SPACING)
    preflight_seed_start = seed_start + (3 * CPU_SMOKE_SEED_BAND_SPACING)
    selfplay_seed_start = seed_start + (4 * CPU_SMOKE_SEED_BAND_SPACING)
    evaluation_seed_start = seed_start + (5 * CPU_SMOKE_SEED_BAND_SPACING)
    steps = (
        *(
            (
                (
                    "benchmark scripted teacher branch coverage",
                    [
                        python_binary,
                        "-m",
                        "pokezero.bootstrap_cli",
                        "teacher-benchmark",
                        "--games",
                        str(args.teacher_branch_preflight_games),
                        *showdown_root_args,
                        "--seed-start",
                        str(preflight_seed_start),
                        "--max-decision-rounds",
                        str(args.max_decision_rounds),
                        *_teacher_branch_gate_args(args),
                        "--json",
                    ],
                ),
            )
            if _teacher_branch_preflight_requested(args)
            else ()
        ),
        (
            "bootstrap teacher checkpoint",
            [
                python_binary,
                "-m",
                "pokezero.bootstrap_cli",
                "teacher",
                "--run-dir",
                str(teacher_dir),
                "--train-games",
                str(args.train_games),
                "--validation-games",
                str(args.validation_games),
                "--workers",
                str(args.workers),
                *showdown_root_args,
                "--seed-start",
                str(seed_start),
                "--shuffle-seed",
                str(seed_start),
                "--validation-seed-start",
                str(validation_seed_start),
                "--benchmark-seed-start",
                str(bootstrap_benchmark_seed_start),
                "--preflight-seed-start",
                str(preflight_seed_start),
                "--benchmark-games",
                str(args.bootstrap_benchmark_games),
                "--preflight-games",
                "1",
                "--max-decision-rounds",
                str(args.max_decision_rounds),
                "--window-size",
                str(args.window_size),
                "--feature-count",
                str(args.feature_count),
            ],
        ),
        (
            "run smoke self-play iteration loop",
            [
                python_binary,
                "-m",
                "pokezero.selfplay_cli",
                "iterate",
                "--run-dir",
                str(selfplay_dir),
                "--initial-policy",
                f"linear:{teacher_dir / 'linear-bootstrap.json'}",
                "--validation-data",
                str(teacher_dir / "validation-rollouts.jsonl"),
                "--iterations",
                str(args.selfplay_iterations),
                "--games-per-iteration",
                str(args.selfplay_games),
                "--workers",
                str(args.workers),
                "--evaluation-games",
                str(args.evaluation_games),
                "--seed-start",
                str(selfplay_seed_start),
                "--evaluation-seed-start",
                str(evaluation_seed_start),
                "--shuffle-seed",
                str(seed_start),
                "--promotion-registry",
                str(promotion_registry),
                "--promotion-artifact-dir",
                str(promotion_artifact_dir),
                "--auto-promote",
                "--profile",
                "smoke",
                "--audit-after-iteration",
                "--audit-profile",
                "smoke",
                *showdown_root_args,
                "--max-decision-rounds",
                str(args.max_decision_rounds),
                "--window-size",
                str(args.window_size),
                "--feature-count",
                str(args.feature_count),
            ],
        ),
        (
            "inspect self-play report",
            [
                python_binary,
                "-m",
                "pokezero.selfplay_cli",
                "report",
                "--run-dir",
                str(selfplay_dir),
            ],
        ),
        (
            "audit smoke run",
            [
                python_binary,
                "-m",
                "pokezero.eval_cli",
                "audit",
                str(selfplay_dir),
                "--profile",
                "smoke",
            ],
        ),
        (
            "calibrate smoke audit config",
            [
                python_binary,
                "-m",
                "pokezero.eval_cli",
                "audit-calibrate",
                str(selfplay_dir),
                "--compare-profile",
                "smoke",
                "--fail-on-profile",
                "--require-run-count",
                "1",
                "--require-benchmark-iterations",
                "1",
                "--require-min-benchmark-games",
                "1",
                "--write-config",
                str(audit_config_path),
            ],
        ),
        (
            "audit smoke run with calibrated config",
            [
                python_binary,
                "-m",
                "pokezero.eval_cli",
                "audit",
                str(selfplay_dir),
                "--audit-config",
                str(audit_config_path),
            ],
        ),
    )
    return {
        "purpose": "tiny CPU-only bootstrap/self-play plumbing validation",
        "warning": "smoke-profile thresholds validate command flow, not policy strength",
        "source": collect_source_metadata(),
        "run_root": str(run_root),
        "python_binary": python_binary,
        "showdown_root": showdown_root,
        "seed_start": seed_start,
        "audit_config_path": str(audit_config_path),
        "teacher_branch_preflight_requested": _teacher_branch_preflight_requested(args),
        "teacher_branch_preflight_games": args.teacher_branch_preflight_games,
        "teacher_branch_preflight_output_path": (
            str(teacher_branch_preflight_output_path) if _teacher_branch_preflight_requested(args) else None
        ),
        "required_teacher_branches": list(args.require_teacher_branch or ()),
        "min_teacher_branch_counts": list(args.min_teacher_branch_count or ()),
        "steps": [
            {
                "name": name,
                "argv": argv,
                "command": _shell_join(argv),
                **(
                    {"output_json_path": str(teacher_branch_preflight_output_path)}
                    if name == "benchmark scripted teacher branch coverage"
                    else {}
                ),
            }
            for name, argv in steps
        ],
    }


def _shell_join(argv: list[str]) -> str:
    return " ".join(shlex.quote(part) for part in argv)


def _write_json_payload(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(f".{path.name}.tmp")
    temporary_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary_path.replace(path)


def _write_text_payload(path: Path, payload: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(f".{path.name}.tmp")
    temporary_path.write_text(payload, encoding="utf-8")
    temporary_path.replace(path)


def _write_run_summary_update(
    path: Path,
    payload: dict[str, object],
    *,
    summary_label: str,
    previous_failure: bool,
) -> bool:
    if previous_failure:
        return True
    try:
        _write_json_payload(path, payload)
    except OSError as exc:
        print(f"warning: failed to update {summary_label} summary {path}: {exc}", file=sys.stderr)
        return True
    return False


def _utc_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def _compare(args: argparse.Namespace) -> int:
    if args.audit_profile is not None and args.audit_config is not None:
        raise ValueError("--audit-profile cannot be combined with --audit-config.")
    if args.fail_on_audit and args.audit_profile is None and args.audit_config is None:
        raise ValueError("--fail-on-audit requires --audit-profile or --audit-config.")
    if args.calibration_require_run_count < 0:
        raise ValueError("calibration_require_run_count must be non-negative.")
    if args.calibration_require_benchmark_iterations < 0:
        raise ValueError("calibration_require_benchmark_iterations must be non-negative.")
    if args.calibration_require_min_benchmark_games < 0:
        raise ValueError("calibration_require_min_benchmark_games must be non-negative.")
    if (
        not args.suggest_audit_calibration
        and (
            args.calibration_require_run_count > 0
            or args.calibration_require_benchmark_iterations > 0
            or args.calibration_require_min_benchmark_games > 0
        )
    ):
        raise ValueError("calibration sufficiency requirements require --suggest-audit-calibration.")
    if args.write_audit_config is not None and not args.suggest_audit_calibration:
        raise ValueError("--write-audit-config requires --suggest-audit-calibration.")
    audit_profile = evaluation_profile(args.audit_profile) if args.audit_profile is not None else None
    audit_config = (
        load_run_audit_config(args.audit_config)
        if args.audit_config is not None
        else audit_profile.audit_config if audit_profile is not None else None
    )
    audit_label = (
        str(args.audit_config)
        if args.audit_config is not None
        else audit_profile.name if audit_profile is not None else None
    )
    paths = _expanded_manifest_paths(args.paths, args.manifest_glob)
    result = compare_run_manifests_with_threshold(
        paths,
        min_benchmark_games=args.min_benchmark_games,
        audit_config=audit_config,
        audit_profile=audit_label,
    )
    calibration = None
    calibration_error = None
    if args.suggest_audit_calibration:
        calibration_paths = tuple(entry.manifest_path for entry in result.entries)
        if calibration_paths:
            calibration = (
                calibrate_run_audit(calibration_paths[0], margin=args.calibration_margin)
                if len(calibration_paths) == 1
                else calibrate_run_audits(
                    calibration_paths,
                    margin=args.calibration_margin,
                    aggregate_mode=args.calibration_aggregate_mode,
                )
            )
        else:
            calibration_error = "no valid compared runs were available for audit calibration"
    calibration_sufficiency_requested = (
        args.calibration_require_run_count > 0
        or args.calibration_require_benchmark_iterations > 0
        or args.calibration_require_min_benchmark_games > 0
    )
    calibration_sufficiency_errors = (
        _calibration_sufficiency_errors(
            calibration,
            require_run_count=args.calibration_require_run_count,
            require_benchmark_iterations=args.calibration_require_benchmark_iterations,
            require_min_benchmark_games=args.calibration_require_min_benchmark_games,
        )
        if args.suggest_audit_calibration
        else ()
    )
    wrote_audit_config_path = None
    audit_config_write_error = None
    if args.write_audit_config is not None:
        if not calibration_sufficiency_requested:
            raise ValueError(
                "--write-audit-config requires at least one calibration sufficiency requirement "
                "(--calibration-require-run-count, --calibration-require-benchmark-iterations, "
                "or --calibration-require-min-benchmark-games)."
            )
        if calibration_sufficiency_errors:
            raise ValueError("--write-audit-config requires calibration sufficiency checks to pass.")
        if result.errors:
            audit_config_write_error = "--write-audit-config requires every compared manifest to load successfully."
        elif args.fail_on_audit and result.audit_failed:
            audit_config_write_error = "--write-audit-config requires the selected audit to pass."
        elif calibration is None:
            audit_config_write_error = "--write-audit-config requires audit calibration to be available."
        else:
            _write_json_payload(args.write_audit_config, _audit_calibration_config_payload(calibration))
            wrote_audit_config_path = args.write_audit_config
    if args.json:
        payload = result.to_dict()
        if args.suggest_audit_calibration:
            payload["audit_calibration"] = calibration.to_dict() if calibration is not None else None
            payload["audit_calibration_error"] = calibration_error
            if calibration_sufficiency_requested:
                payload["audit_calibration_sufficient"] = not calibration_sufficiency_errors
                payload["audit_calibration_sufficiency_errors"] = list(calibration_sufficiency_errors)
            if wrote_audit_config_path is not None:
                payload["written_audit_config_path"] = str(wrote_audit_config_path)
            if audit_config_write_error is not None:
                payload["audit_config_write_error"] = audit_config_write_error
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        _print_run_comparison(result)
        if args.suggest_audit_calibration:
            print("")
            print("audit_calibration_suggestion:")
            if calibration is None:
                print(f"unavailable: {calibration_error}")
            else:
                _print_audit_calibration(calibration)
            if calibration_sufficiency_requested:
                _print_calibration_sufficiency(calibration_sufficiency_errors)
            if result.errors:
                print("calibration_excluded_errors:")
                for error in result.errors:
                    print(f"- {error.label}: {error.error}")
            if wrote_audit_config_path is not None:
                print(f"written_audit_config: {wrote_audit_config_path}")
            if audit_config_write_error is not None:
                print(f"audit_config_write_error: {audit_config_write_error}")
    return 2 if result.errors or (args.fail_on_audit and result.audit_failed) or calibration_sufficiency_errors else 0


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
    if getattr(args, "audit_config", None) is not None and getattr(args, "profile", None) is not None:
        raise ValueError("--profile cannot be combined with --audit-config.")
    profile_config = evaluation_profile(args.profile).audit_config
    if getattr(args, "audit_config", None) is not None:
        profile_config = load_run_audit_config(args.audit_config)
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
    _print_source_metadata(result.source_metadata)
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


def _calibration_sufficiency_errors(
    result,
    *,
    require_run_count: int,
    require_benchmark_iterations: int,
    require_min_benchmark_games: int,
) -> tuple[str, ...]:
    if require_run_count < 0:
        raise ValueError("require_run_count must be non-negative.")
    if require_benchmark_iterations < 0:
        raise ValueError("require_benchmark_iterations must be non-negative.")
    if require_min_benchmark_games < 0:
        raise ValueError("require_min_benchmark_games must be non-negative.")
    observed_run_count = int(getattr(result, "run_count", 1)) if result is not None else 0
    observed_benchmark_iterations = int(getattr(result, "benchmark_iteration_count", 0)) if result is not None else 0
    observed_min_benchmark_games = int(getattr(result, "min_latest_benchmark_games", 0)) if result is not None else 0
    errors: list[str] = []
    if observed_run_count < require_run_count:
        errors.append(
            f"calibration_run_count {observed_run_count} is below required {require_run_count}"
        )
    if observed_benchmark_iterations < require_benchmark_iterations:
        errors.append(
            "calibration_benchmark_iterations "
            f"{observed_benchmark_iterations} is below required {require_benchmark_iterations}"
        )
    if require_benchmark_iterations > 0 and result is not None and not getattr(result, "require_benchmark", False):
        errors.append("calibration includes at least one run without benchmark iterations")
    if observed_min_benchmark_games < require_min_benchmark_games:
        errors.append(
            "calibration_min_benchmark_games "
            f"{observed_min_benchmark_games} is below required {require_min_benchmark_games}"
        )
    return tuple(errors)


def _print_calibration_sufficiency(errors: tuple[str, ...]) -> None:
    print(f"calibration_sufficiency: {'FAIL' if errors else 'PASS'}")
    if errors:
        print("calibration_sufficiency_errors:")
        for error in errors:
            print(f"- {error}")


def _profile_audit_payload(
    paths: Iterable[Path],
    *,
    profile_name: str,
    config: RunAuditConfig,
) -> dict[str, object]:
    runs = []
    for path in paths:
        result = audit_run(path, config=config)
        runs.append(
            {
                "manifest_path": str(result.manifest_path),
                "passed": result.passed,
                "failed_checks": [check.name for check in result.checks if not check.passed],
            }
        )
    return {
        "profile": profile_name,
        "passed": all(bool(run["passed"]) for run in runs),
        "runs": runs,
    }


def _print_profile_audit(payload: dict[str, object]) -> None:
    print("")
    print("profile_audit:")
    print(f"profile: {payload['profile']}")
    print(f"status: {'PASS' if payload['passed'] else 'FAIL'}")
    print("runs:")
    for run in payload["runs"]:
        run_status = "PASS" if run["passed"] else "FAIL"
        print(f"- {run_status} {run['manifest_path']}")
        if run["failed_checks"]:
            print(f"  failed_checks: {', '.join(run['failed_checks'])}")


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
    if result.entries:
        print("")
        print("source_provenance:")
        for entry in result.entries:
            print(f"- {entry.label}: {_format_source_metadata(entry.source_metadata)}")
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


def _print_source_metadata(metadata: Mapping[str, object]) -> None:
    if not metadata:
        print("source_metadata: -")
        return
    print("source_metadata:")
    print(f"  available: {_format_source_available(metadata.get('available'))}")
    print(f"  branch: {_format_optional_text(metadata.get('branch'))}")
    print(f"  head: {_format_optional_text(metadata.get('head'))}")
    print(f"  dirty: {_format_source_dirty(metadata.get('dirty'))}")
    print(f"  repo_root: {_format_optional_text(metadata.get('repo_root'))}")
    if metadata.get("error") is not None:
        print(f"  error: {_format_optional_text(metadata.get('error'))}")


def _format_source_metadata(metadata: Mapping[str, object]) -> str:
    if not metadata:
        return "-"
    return (
        f"available={_format_source_available(metadata.get('available'))} "
        f"branch={_format_optional_text(metadata.get('branch'))} "
        f"head={_format_optional_text(metadata.get('head'))} "
        f"dirty={_format_source_dirty(metadata.get('dirty'))}"
        f"{_format_source_error(metadata)}"
    )


def _format_source_available(value: object) -> str:
    return _format_optional_bool(value if isinstance(value, bool) else None)


def _format_source_dirty(value: object) -> str:
    return _format_optional_bool(value if isinstance(value, bool) else None)


def _format_optional_text(value: object) -> str:
    if value is None:
        return "-"
    text = str(value)
    return text if text else "-"


def _format_source_error(metadata: Mapping[str, object]) -> str:
    error = metadata.get("error")
    return "" if error is None else f" error={_format_optional_text(error)}"


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
