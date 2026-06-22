"""Shared CLI helpers for optional run-audit enforcement."""

from __future__ import annotations

import argparse

from .run_audit import RunAuditConfig


DEFAULT_POST_ITERATION_AUDIT_CONFIG = RunAuditConfig(
    max_benchmark_win_rate_drop=0.15,
    max_consecutive_promotion_failures=3,
)


def add_post_iteration_audit_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--audit-after-iteration",
        action="store_true",
        help="Run the self-play run audit after each completed iteration and stop on failure.",
    )
    parser.add_argument("--audit-min-latest-benchmark-win-rate", type=float, default=None)
    parser.add_argument("--audit-min-latest-benchmark-games", type=int, default=None)
    parser.add_argument("--audit-max-latest-collection-capped-rate", type=float, default=None)
    parser.add_argument("--audit-max-latest-benchmark-capped-rate", type=float, default=None)
    parser.add_argument("--audit-max-benchmark-win-rate-drop", type=float, default=None)
    parser.add_argument("--audit-max-consecutive-promotion-failures", type=int, default=None)
    parser.add_argument(
        "--audit-allow-missing-benchmark",
        action="store_true",
        help="With --audit-after-iteration, do not fail solely because the latest benchmark is missing.",
    )
    parser.add_argument(
        "--audit-require-latest-promotion",
        action="store_true",
        help="With --audit-after-iteration, fail unless the latest iteration recorded a promotion.",
    )


def post_iteration_audit_config_from_args(args: argparse.Namespace) -> RunAuditConfig | None:
    if not args.audit_after_iteration:
        return None
    defaults = DEFAULT_POST_ITERATION_AUDIT_CONFIG
    return RunAuditConfig(
        min_latest_benchmark_win_rate=_arg_or_default(
            args.audit_min_latest_benchmark_win_rate,
            defaults.min_latest_benchmark_win_rate,
        ),
        min_latest_benchmark_games=_arg_or_default(
            args.audit_min_latest_benchmark_games,
            defaults.min_latest_benchmark_games,
        ),
        max_latest_collection_capped_rate=_arg_or_default(
            args.audit_max_latest_collection_capped_rate,
            defaults.max_latest_collection_capped_rate,
        ),
        max_latest_benchmark_capped_rate=_arg_or_default(
            args.audit_max_latest_benchmark_capped_rate,
            defaults.max_latest_benchmark_capped_rate,
        ),
        max_benchmark_win_rate_drop=_arg_or_default(
            args.audit_max_benchmark_win_rate_drop,
            defaults.max_benchmark_win_rate_drop,
        ),
        max_consecutive_promotion_failures=_arg_or_default(
            args.audit_max_consecutive_promotion_failures,
            defaults.max_consecutive_promotion_failures,
        ),
        require_benchmark=not args.audit_allow_missing_benchmark,
        require_latest_promotion=bool(args.audit_require_latest_promotion),
    )


def _arg_or_default(value, default):
    return default if value is None else value
