"""Command-line promotion gates for experiment manifests."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from .evaluation import (
    DEFAULT_MAX_BENCHMARK_CAPPED_RATE,
    DEFAULT_MAX_COLLECTION_CAPPED_RATE,
    DEFAULT_MAX_TEACHER_DEGRADATION_RATE,
    DEFAULT_MIN_BENCHMARK_GAMES,
    DEFAULT_MIN_BENCHMARK_WIN_RATE,
    PromotionGateConfig,
    evaluate_promotion_gate,
)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m pokezero.eval_cli")
    subparsers = parser.add_subparsers(dest="command", required=True)

    gate = subparsers.add_parser("gate", help="Evaluate whether a manifest clears promotion thresholds.")
    gate.add_argument("path", type=Path, help="Experiment run directory or manifest.json path.")
    gate.add_argument("--min-benchmark-win-rate", type=float, default=DEFAULT_MIN_BENCHMARK_WIN_RATE)
    gate.add_argument("--min-benchmark-games", type=int, default=DEFAULT_MIN_BENCHMARK_GAMES)
    gate.add_argument("--max-collection-capped-rate", type=float, default=DEFAULT_MAX_COLLECTION_CAPPED_RATE)
    gate.add_argument("--max-benchmark-capped-rate", type=float, default=DEFAULT_MAX_BENCHMARK_CAPPED_RATE)
    gate.add_argument("--max-teacher-degradation-rate", type=float, default=DEFAULT_MAX_TEACHER_DEGRADATION_RATE)
    gate.add_argument(
        "--benchmark-opponent",
        action="append",
        default=None,
        help="Require and gate a specific benchmark opponent policy id. May be repeated. Defaults to every candidate benchmark opponent.",
    )
    gate.add_argument(
        "--opponent-win-rate",
        action="append",
        default=None,
        metavar="POLICY_ID=RATE",
        help="Override the win-rate floor for a specific benchmark opponent. May be repeated.",
    )
    gate.add_argument("--allow-missing-benchmark", action="store_true", help="Do not fail solely because benchmark evidence is missing.")
    gate.add_argument("--json", action="store_true", help="Print the gate result as JSON.")
    gate.set_defaults(func=_gate)
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
        config=PromotionGateConfig(
            min_benchmark_win_rate=args.min_benchmark_win_rate,
            min_benchmark_games=args.min_benchmark_games,
            max_collection_capped_rate=args.max_collection_capped_rate,
            max_benchmark_capped_rate=args.max_benchmark_capped_rate,
            max_teacher_degradation_rate=args.max_teacher_degradation_rate,
            require_benchmark=not args.allow_missing_benchmark,
            required_benchmark_opponents=tuple(args.benchmark_opponent or ()),
            opponent_min_win_rates=_parse_opponent_win_rates(tuple(args.opponent_win_rate or ())),
        ),
    )
    if args.json:
        print(json.dumps(result.to_dict(), indent=2, sort_keys=True))
    else:
        _print_gate_result(result)
    return 0 if result.passed else 2


def _print_gate_result(result) -> None:
    status = "PASS" if result.passed else "FAIL"
    print(f"status: {status}")
    print("mode: absolute benchmark floor; this does not compare against an incumbent checkpoint")
    print(f"source: {result.source_type}")
    print(f"manifest: {result.manifest_path}")
    print(f"candidate_policy: {result.candidate_policy_id or '-'}")
    print(f"checkpoint: {result.checkpoint_path or '-'}")
    if result.source_iteration is not None:
        print(f"iteration: {result.source_iteration}")
    print(f"pooled_benchmark_win_rate: {_format_optional_float(result.benchmark_win_rate)}")
    print(f"collection_capped_rate: {_format_optional_float(result.collection_capped_rate)}")
    print(f"benchmark_capped_rate: {_format_optional_float(result.benchmark_capped_rate)}")
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
