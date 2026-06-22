"""Shared CLI helpers for optional run-audit enforcement."""

from __future__ import annotations

import argparse
from pathlib import Path

from .evaluation_profiles import EVALUATION_PROFILES, evaluation_profile
from .run_audit import RunAuditConfig, load_run_audit_config


DEFAULT_POST_ITERATION_AUDIT_CONFIG = RunAuditConfig(
    max_benchmark_win_rate_drop=0.15,
    max_consecutive_promotion_failures=3,
)


def add_post_iteration_audit_arguments(parser: argparse.ArgumentParser) -> None:
    profile_choices = tuple(sorted(EVALUATION_PROFILES))
    parser.add_argument(
        "--audit-after-iteration",
        action="store_true",
        help="Run the self-play run audit after each completed iteration and stop on failure.",
    )
    parser.add_argument(
        "--audit-profile",
        choices=profile_choices,
        default=None,
        help=(
            "Named audit profile used as defaults for --audit-after-iteration. "
            "Omit to keep the looser post-iteration defaults."
        ),
    )
    parser.add_argument(
        "--audit-config",
        type=Path,
        default=None,
        help=(
            "Versioned run-audit config JSON used as defaults for --audit-after-iteration. "
            "Explicit --audit-* threshold flags override this file."
        ),
    )
    parser.add_argument("--audit-min-latest-benchmark-win-rate", type=float, default=None)
    parser.add_argument("--audit-min-latest-benchmark-games", type=int, default=None)
    parser.add_argument("--audit-max-latest-collection-capped-rate", type=float, default=None)
    parser.add_argument("--audit-max-latest-benchmark-capped-rate", type=float, default=None)
    parser.add_argument("--audit-max-latest-average-decision-rounds", type=float, default=None)
    parser.add_argument("--audit-max-latest-benchmark-average-decision-rounds", type=float, default=None)
    parser.add_argument("--audit-max-latest-process-peak-rss-mb", type=float, default=None)
    parser.add_argument("--audit-max-benchmark-win-rate-drop", type=float, default=None)
    parser.add_argument("--audit-max-consecutive-promotion-failures", type=int, default=None)
    benchmark_group = parser.add_mutually_exclusive_group()
    benchmark_group.add_argument(
        "--audit-require-benchmark",
        dest="audit_require_benchmark",
        action="store_true",
        default=None,
        help="With --audit-after-iteration, fail when the latest benchmark is missing.",
    )
    benchmark_group.add_argument(
        "--audit-allow-missing-benchmark",
        dest="audit_require_benchmark",
        action="store_false",
        default=None,
        help="With --audit-after-iteration, do not fail solely because the latest benchmark is missing.",
    )
    benchmark_opponent_group = parser.add_mutually_exclusive_group()
    benchmark_opponent_group.add_argument(
        "--audit-require-benchmark-opponents",
        dest="audit_require_benchmark_opponent_coverage",
        action="store_true",
        default=None,
        help=(
            "With --audit-after-iteration, fail when the latest benchmark omits "
            "fixed baseline opponents seen in prior benchmark evidence."
        ),
    )
    benchmark_opponent_group.add_argument(
        "--audit-allow-missing-benchmark-opponents",
        dest="audit_require_benchmark_opponent_coverage",
        action="store_false",
        default=None,
        help=(
            "With --audit-after-iteration, do not fail when the latest benchmark omits "
            "fixed baseline opponents seen in prior benchmark evidence."
        ),
    )
    latest_promotion_group = parser.add_mutually_exclusive_group()
    latest_promotion_group.add_argument(
        "--audit-require-latest-promotion",
        dest="audit_require_latest_promotion",
        action="store_true",
        default=None,
        help="With --audit-after-iteration, fail unless the latest iteration recorded a promotion.",
    )
    latest_promotion_group.add_argument(
        "--audit-allow-missing-latest-promotion",
        dest="audit_require_latest_promotion",
        action="store_false",
        default=None,
        help="With --audit-after-iteration, do not fail solely because the latest iteration did not record a promotion.",
    )


def post_iteration_audit_config_from_args(args: argparse.Namespace) -> RunAuditConfig | None:
    if args.audit_profile is not None and args.audit_config is not None:
        raise ValueError("--audit-profile cannot be combined with --audit-config.")
    if not args.audit_after_iteration:
        if args.audit_config is not None:
            raise ValueError("--audit-config requires --audit-after-iteration.")
        return None
    defaults = (
        evaluation_profile(args.audit_profile).audit_config
        if args.audit_profile is not None
        else DEFAULT_POST_ITERATION_AUDIT_CONFIG
    )
    if args.audit_config is not None:
        defaults = load_run_audit_config(args.audit_config)
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
        max_latest_average_decision_rounds=_arg_or_default(
            args.audit_max_latest_average_decision_rounds,
            defaults.max_latest_average_decision_rounds,
        ),
        max_latest_benchmark_average_decision_rounds=_arg_or_default(
            args.audit_max_latest_benchmark_average_decision_rounds,
            defaults.max_latest_benchmark_average_decision_rounds,
        ),
        max_latest_process_peak_rss_mb=_arg_or_default(
            args.audit_max_latest_process_peak_rss_mb,
            defaults.max_latest_process_peak_rss_mb,
        ),
        max_benchmark_win_rate_drop=_arg_or_default(
            args.audit_max_benchmark_win_rate_drop,
            defaults.max_benchmark_win_rate_drop,
        ),
        max_consecutive_promotion_failures=_arg_or_default(
            args.audit_max_consecutive_promotion_failures,
            defaults.max_consecutive_promotion_failures,
        ),
        require_benchmark=_arg_or_default(
            args.audit_require_benchmark,
            defaults.require_benchmark,
        ),
        require_latest_promotion=_arg_or_default(
            args.audit_require_latest_promotion,
            defaults.require_latest_promotion,
        ),
        require_benchmark_opponent_coverage=_arg_or_default(
            args.audit_require_benchmark_opponent_coverage,
            defaults.require_benchmark_opponent_coverage,
        ),
    )


def validate_post_iteration_audit_evaluation_games(
    config: RunAuditConfig | None,
    *,
    evaluation_games: int,
    minimum_benchmark_matchups: int,
) -> None:
    if config is None or not config.require_benchmark:
        return
    if evaluation_games <= 0:
        raise ValueError(
            "--audit-after-iteration requires --evaluation-games > 0 unless "
            "--audit-allow-missing-benchmark is set."
        )
    minimum_benchmark_games = evaluation_games * minimum_benchmark_matchups
    if minimum_benchmark_games < config.min_latest_benchmark_games:
        raise ValueError(
            "--audit-after-iteration requires enough --evaluation-games to satisfy "
            f"the audit benchmark-game floor: at least {config.min_latest_benchmark_games} "
            f"aggregate benchmark games are required, but {evaluation_games} "
            f"evaluation games only guarantees {minimum_benchmark_games}."
        )


def _arg_or_default(value, default):
    return default if value is None else value
