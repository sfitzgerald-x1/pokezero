"""Command-line entry points for G4 refutation mining."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .collection import (
    env_config_with_policy_spec_masks,
    iter_rollout_records,
    policy_from_spec,
    policy_spec_with_showdown_root,
)
from .local_showdown import LocalShowdownConfig, LocalShowdownEnv
from .refutation_mining import (
    DEFAULT_R0_MIN_CERTIFIED_REFUTATIONS,
    DEFAULT_R0_MIN_SAMPLED_WINS,
    RefutationMiningConfig,
    ReplayTerminalBranchEvaluator,
    candidate_count_for_records,
    iter_fragile_states,
    mine_refutations,
    validate_refutation_report_payload,
    write_refutation_report,
)
from .rollout import RolloutConfig


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m pokezero.refutation_cli")
    subparsers = parser.add_subparsers(dest="command", required=True)

    mine = subparsers.add_parser(
        "mine",
        help="Mine certified loser-seat refutations from champion-won rollout records.",
    )
    mine.add_argument("--records", action="append", required=True, type=Path, help="Rollout-record JSONL. May repeat.")
    mine.add_argument("--out-dir", type=Path, required=True, help="Output directory for report + fragile archive.")
    mine.add_argument("--report-name", default="refutation-report.json", help="Report filename under --out-dir.")
    mine.add_argument("--archive-name", default="fragile-states.jsonl", help="Archive filename under --out-dir.")
    _add_common_args(mine)
    mine.add_argument(
        "--p1-policy",
        required=True,
        help="Continuation policy spec for p1. Use the frozen policies that should play after each deviation.",
    )
    mine.add_argument(
        "--p2-policy",
        required=True,
        help="Continuation policy spec for p2. Use the frozen policies that should play after each deviation.",
    )
    mine.add_argument("--showdown-root", type=Path, default=None, help="Built Pokemon Showdown checkout root.")
    mine.add_argument("--node-binary", default="node", help="Node executable used for the BattleStream bridge.")
    mine.add_argument("--format", dest="format_id", default="gen3randombattle", help="Showdown format id.")
    mine.add_argument("--max-decision-rounds", type=int, default=250, help="Continuation decision-round cap.")
    mine.add_argument(
        "--check-prefix-observations",
        action="store_true",
        help="Strictly compare replay prefix observations before branch evaluation. Slower and can fail on history-tail drift.",
    )
    mine.set_defaults(func=_mine)

    plan = subparsers.add_parser(
        "plan",
        help="Count sampled wins/decision points/deviations without running terminal rollouts.",
    )
    plan.add_argument("--records", action="append", required=True, type=Path, help="Rollout-record JSONL. May repeat.")
    _add_common_args(plan)
    plan.set_defaults(func=_plan)

    validate = subparsers.add_parser(
        "validate",
        help="Validate a refutation report and fragile-state archive against the R0 artifact gate.",
    )
    validate.add_argument("--report", type=Path, required=True, help="Refutation report JSON.")
    validate.add_argument(
        "--archive",
        type=Path,
        default=None,
        help="Fragile-state JSONL archive. Defaults to report.archive_path.",
    )
    validate.add_argument(
        "--min-sampled-wins",
        type=int,
        default=DEFAULT_R0_MIN_SAMPLED_WINS,
        help=f"Minimum sampled champion wins required for R0 acceptance (default {DEFAULT_R0_MIN_SAMPLED_WINS}).",
    )
    validate.add_argument(
        "--min-certified-refutations",
        type=int,
        default=DEFAULT_R0_MIN_CERTIFIED_REFUTATIONS,
        help=(
            "Minimum certified fragile-state examples required for R0 acceptance "
            f"(default {DEFAULT_R0_MIN_CERTIFIED_REFUTATIONS})."
        ),
    )
    validate.add_argument(
        "--min-certification-seeds",
        type=int,
        default=20,
        help="Minimum terminal-rollout reseeds per certified example (default 20).",
    )
    validate.set_defaults(func=_validate)
    return parser


def _add_common_args(parser: argparse.ArgumentParser) -> None:
    champion = parser.add_mutually_exclusive_group(required=True)
    champion.add_argument("--champion-policy-id", default=None, help="Policy id whose won games are mined.")
    champion.add_argument("--champion-player-id", default=None, help="Player id whose won games are mined.")
    parser.add_argument("--max-wins", type=int, default=200, help="Maximum champion-won games to sample.")
    parser.add_argument(
        "--max-decision-points-per-game",
        type=int,
        default=None,
        help="Optional cap on loser decision points scanned per sampled win.",
    )
    parser.add_argument(
        "--max-deviations-per-state",
        type=int,
        default=None,
        help="Optional cap on legal loser deviations evaluated per decision point.",
    )
    parser.add_argument(
        "--certification-seeds",
        type=int,
        default=20,
        help="Terminal rollout reseeds per deviation. Must be at least 20.",
    )
    parser.add_argument(
        "--min-flip-rate",
        type=float,
        default=0.60,
        help="Deviation must beat the recorded champion more than this fraction.",
    )
    parser.add_argument(
        "--mode",
        choices=("oracle", "fair"),
        default="oracle",
        help="Refutation mode label. R0 is expected to start with oracle.",
    )


def _config_from_args(args: argparse.Namespace) -> RefutationMiningConfig:
    return RefutationMiningConfig(
        champion_policy_id=args.champion_policy_id,
        champion_player_id=args.champion_player_id,
        max_wins=args.max_wins,
        max_decision_points_per_game=args.max_decision_points_per_game,
        max_deviations_per_state=args.max_deviations_per_state,
        certification_seed_count=args.certification_seeds,
        min_flip_rate=args.min_flip_rate,
        mode=args.mode,
    )


def _load_records(paths: list[Path]) -> tuple:
    records = []
    for path in paths:
        records.extend(iter_rollout_records(path))
    return tuple(records)


def _plan(args: argparse.Namespace) -> int:
    records = _load_records(args.records)
    payload = candidate_count_for_records(records=records, config=_config_from_args(args))
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def _mine(args: argparse.Namespace) -> int:
    records = _load_records(args.records)
    config = _config_from_args(args)
    env_config = LocalShowdownConfig(
        showdown_root=args.showdown_root,
        node_binary=args.node_binary,
    )
    policy_showdown_root = env_config.resolved_showdown_root()
    p1_spec = policy_spec_with_showdown_root(args.p1_policy, policy_showdown_root)
    p2_spec = policy_spec_with_showdown_root(args.p2_policy, policy_showdown_root)
    env_config = env_config_with_policy_spec_masks(
        env_config,
        (p1_spec, p2_spec),
        context="refutation mining",
    )
    rollout_config = RolloutConfig(
        max_decision_rounds=args.max_decision_rounds,
        format_id=args.format_id,
    )
    evaluator = ReplayTerminalBranchEvaluator(
        env_factory=lambda: LocalShowdownEnv(env_config),
        policies={
            "p1": policy_from_spec(p1_spec),
            "p2": policy_from_spec(p2_spec),
        },
        rollout_config=rollout_config,
        check_prefix_observations=args.check_prefix_observations,
    )
    report = mine_refutations(
        records=records,
        config=config,
        evaluator=evaluator,
        archive_path=args.out_dir / args.archive_name,
    )
    write_refutation_report(args.out_dir / args.report_name, report)
    print(json.dumps(report.to_dict(), indent=2, sort_keys=True))
    return 0


def _validate(args: argparse.Namespace) -> int:
    report = json.loads(args.report.read_text(encoding="utf-8"))
    archive_path = args.archive
    if archive_path is None:
        raw_archive_path = report.get("archive_path")
        if not raw_archive_path:
            raise ValueError("--archive is required when report.archive_path is missing.")
        archive_path = Path(str(raw_archive_path))
    payload = validate_refutation_report_payload(
        report=report,
        fragile_states=tuple(iter_fragile_states(archive_path)),
        min_sampled_wins=args.min_sampled_wins,
        min_certified_refutations=args.min_certified_refutations,
        min_certification_seed_count=args.min_certification_seeds,
    )
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if payload["passed"] else 2


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
