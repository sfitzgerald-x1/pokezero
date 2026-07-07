"""Command-line entry points for G4 refutation mining."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable, Mapping

from .admission_guard import AdmissionGuardConfig, validate_admission_guard
from .collection import (
    env_config_with_policy_spec_masks,
    iter_rollout_records,
    policy_from_spec,
    policy_spec_with_showdown_root,
)
from .dataset import TrajectoryDatasetConfig
from .local_showdown import LocalShowdownConfig, LocalShowdownEnv
from .refutation_curriculum import (
    RefutationCurriculumConfig,
    collect_refutation_curriculum_rollouts,
    write_refutation_curriculum_summary,
)
from .refutation_mining import (
    DEFAULT_R0_MIN_CERTIFIED_REFUTATIONS,
    DEFAULT_R0_MIN_FLIP_RATE,
    DEFAULT_R0_MIN_SAMPLED_WINS,
    RefutationMiningConfig,
    ReplayTerminalBranchEvaluator,
    candidate_count_for_records,
    iter_fragile_states,
    mine_refutations,
    reproduce_refutation_archive,
    validate_refutation_report_payload,
    write_refutation_report,
)
from .refutation_population import (
    RefutationBehaviorSeedConfig,
    build_refutation_behavior_seed_manifest,
    write_refutation_behavior_seed_manifest,
)
from .refutation_progress import (
    build_refutation_cycle_report,
    load_refutation_cycle_report_input,
    write_refutation_cycle_report,
)
from .refutation_training import (
    RefutationTrainingConfig,
    write_refutation_behavior_seed_training_cache,
    write_refutation_training_cache,
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
    validate.add_argument(
        "--min-flip-rate",
        type=float,
        default=DEFAULT_R0_MIN_FLIP_RATE,
        help=f"Minimum observed refutation flip rate required for R0 acceptance (default {DEFAULT_R0_MIN_FLIP_RATE}).",
    )
    validate.add_argument(
        "--allow-continuation-only-reseeds",
        action="store_true",
        help=(
            "Treat continuation-policy-only reseeding as acceptable. This is for exploratory/dev "
            "reports only; R0 acceptance requires simulator-RNG reseeding."
        ),
    )
    validate.set_defaults(func=_validate)

    reproduce = subparsers.add_parser(
        "reproduce",
        help="Rerun fragile-state terminal results from replay coordinates and compare to the archive.",
    )
    reproduce.add_argument(
        "--records",
        action="append",
        required=True,
        type=Path,
        help="Source rollout-record JSONL from the mining run, in the same order used by the archive. May repeat.",
    )
    reproduce.add_argument("--archive", type=Path, required=True, help="Certified fragile-state JSONL archive.")
    reproduce.add_argument("--p1-policy", required=True, help="Continuation policy spec for p1.")
    reproduce.add_argument("--p2-policy", required=True, help="Continuation policy spec for p2.")
    reproduce.add_argument("--showdown-root", type=Path, default=None, help="Built Pokemon Showdown checkout root.")
    reproduce.add_argument("--node-binary", default="node", help="Node executable used for the BattleStream bridge.")
    reproduce.add_argument("--format", dest="format_id", default="gen3randombattle", help="Showdown format id.")
    reproduce.add_argument("--max-decision-rounds", type=int, default=250, help="Continuation decision-round cap.")
    reproduce.add_argument("--max-rows", type=int, default=None, help="Optional cap on reproduced archive rows.")
    reproduce.set_defaults(func=_reproduce)

    training_cache = subparsers.add_parser(
        "training-cache",
        help="Build a separate training cache from certified fragile-state refutations.",
    )
    training_cache.add_argument(
        "--records",
        action="append",
        required=True,
        type=Path,
        help="Source rollout-record JSONL from the mining run, in the same order used by the archive. May repeat.",
    )
    training_cache.add_argument("--archive", type=Path, required=True, help="Certified fragile-state JSONL archive.")
    training_cache.add_argument("--out", type=Path, required=True, help="Output training-cache directory.")
    training_cache.add_argument(
        "--target-mode",
        choices=("value", "policy-value", "policy-distribution-value"),
        default="policy-value",
        help=(
            "value: retarget value only and keep the recorded loser action; "
            "use only with PPO/value-only consumers, not BC/RWR. "
            "policy-value: also replace the action target with the certified deviation. "
            "policy-distribution-value: use row.search_policy_distribution as weighted policy targets."
        ),
    )
    training_cache.add_argument(
        "--max-examples",
        type=int,
        default=None,
        help="Optional cap on emitted target examples; multi-target distribution rows are kept or dropped as a unit.",
    )
    training_cache.add_argument(
        "--surprise-weight-scale",
        type=float,
        default=0.0,
        help=(
            "Optional certification-strength weighting scale. 0 leaves every refutation row "
            "at neutral training weight 1.0."
        ),
    )
    training_cache.add_argument(
        "--surprise-weight-max",
        type=float,
        default=4.0,
        help="Maximum per-example training weight when --surprise-weight-scale is enabled.",
    )
    training_cache.add_argument("--window-size", type=int, default=1, help="Training observation window size.")
    training_cache.add_argument("--discount", type=float, default=1.0, help="Dataset discount used while materializing source windows before refutation targets are applied.")
    training_cache.add_argument(
        "--ppo-target-mode",
        choices=("returns", "gae"),
        default="returns",
        help="Dataset target mode used while materializing source windows before refutation targets are applied.",
    )
    training_cache.add_argument("--gae-lambda", type=float, default=0.95, help="GAE lambda for source materialization when --ppo-target-mode=gae.")
    training_cache.add_argument("--overwrite", action="store_true", help="Replace an existing output cache directory.")
    training_cache.set_defaults(func=_training_cache)

    behavior_seed_cache = subparsers.add_parser(
        "behavior-seed-cache",
        help="Build a refutation training cache from an R2 behavior-seed manifest.",
    )
    behavior_seed_cache.add_argument(
        "--records",
        action="append",
        required=True,
        type=Path,
        help="Source rollout-record JSONL from the mining run, in the same order used by the behavior seeds. May repeat.",
    )
    behavior_seed_cache.add_argument("--behavior-seeds", type=Path, required=True, help="R2 behavior-seed manifest JSON.")
    behavior_seed_cache.add_argument("--out", type=Path, required=True, help="Output training-cache directory.")
    behavior_seed_cache.add_argument(
        "--target-mode",
        choices=("value", "policy-value"),
        default="policy-value",
        help=(
            "value: retarget value only and keep the recorded loser action; "
            "policy-value: also replace the action target with the certified deviation."
        ),
    )
    behavior_seed_cache.add_argument("--max-examples", type=int, default=None, help="Optional cap on emitted target examples.")
    behavior_seed_cache.add_argument(
        "--surprise-weight-scale",
        type=float,
        default=0.0,
        help="Optional certification-strength weighting scale. 0 leaves every behavior seed at neutral weight 1.0.",
    )
    behavior_seed_cache.add_argument(
        "--surprise-weight-max",
        type=float,
        default=4.0,
        help="Maximum per-example training weight when --surprise-weight-scale is enabled.",
    )
    behavior_seed_cache.add_argument("--window-size", type=int, default=1, help="Training observation window size.")
    behavior_seed_cache.add_argument("--discount", type=float, default=1.0, help="Dataset discount used while materializing source windows before refutation targets are applied.")
    behavior_seed_cache.add_argument(
        "--ppo-target-mode",
        choices=("returns", "gae"),
        default="returns",
        help="Dataset target mode used while materializing source windows before refutation targets are applied.",
    )
    behavior_seed_cache.add_argument("--gae-lambda", type=float, default=0.95, help="GAE lambda for source materialization when --ppo-target-mode=gae.")
    behavior_seed_cache.add_argument("--overwrite", action="store_true", help="Replace an existing output cache directory.")
    behavior_seed_cache.set_defaults(func=_behavior_seed_cache)

    curriculum = subparsers.add_parser(
        "curriculum",
        help="Collect rollout JSONL from certified fragile states for the R1(d) curriculum slice.",
    )
    curriculum.add_argument(
        "--records",
        action="append",
        required=True,
        type=Path,
        help="Source rollout-record JSONL from the mining run, in the same order used by the archive. May repeat.",
    )
    curriculum.add_argument("--archive", type=Path, required=True, help="Certified fragile-state JSONL archive.")
    curriculum.add_argument("--out", type=Path, required=True, help="Output curriculum rollout-record JSONL.")
    curriculum.add_argument(
        "--summary",
        type=Path,
        default=None,
        help="Optional summary JSON path. Defaults to <out>.summary.json.",
    )
    curriculum.add_argument("--p1-policy", required=True, help="Continuation policy spec for p1.")
    curriculum.add_argument("--p2-policy", required=True, help="Continuation policy spec for p2.")
    curriculum.add_argument("--showdown-root", type=Path, default=None, help="Built Pokemon Showdown checkout root.")
    curriculum.add_argument("--node-binary", default="node", help="Node executable used for the BattleStream bridge.")
    curriculum.add_argument("--format", dest="format_id", default="gen3randombattle", help="Showdown format id.")
    curriculum.add_argument("--max-decision-rounds", type=int, default=250, help="Continuation decision-round cap.")
    curriculum.add_argument(
        "--total-games",
        type=int,
        required=True,
        help="Total collection games in the parent run; multiplied by --curriculum-fraction.",
    )
    curriculum.add_argument(
        "--curriculum-fraction",
        type=float,
        required=True,
        help="Fraction of parent collection games to start from fragile states.",
    )
    curriculum.add_argument("--seed-start", type=int, default=1, help="First continuation-policy RNG seed.")
    curriculum.add_argument("--max-starts", type=int, default=None, help="Optional hard cap on curriculum starts.")
    curriculum.set_defaults(func=_curriculum)

    behavior_seeds = subparsers.add_parser(
        "behavior-seeds",
        help="Build an R2 behavior-seed manifest from certified fragile-state refutations.",
    )
    behavior_seeds.add_argument("--archive", type=Path, required=True, help="Certified fragile-state JSONL archive.")
    behavior_seeds.add_argument("--out", type=Path, required=True, help="Output behavior-seed manifest JSON.")
    behavior_seeds.add_argument("--max-seeds", type=int, default=None, help="Optional cap on emitted behavior seeds.")
    behavior_seeds.add_argument(
        "--min-flip-rate",
        type=float,
        default=0.0,
        help="Minimum certified flip rate required for a row to become a behavior seed.",
    )
    behavior_seeds.add_argument(
        "--mode",
        choices=("oracle", "fair"),
        default=None,
        help="Optional refutation mode filter. Defaults to including both oracle and fair rows.",
    )
    behavior_seeds.set_defaults(func=_behavior_seeds)

    cycle_report = subparsers.add_parser(
        "cycle-report",
        help="Aggregate R0 refutation reports into per-mode trends and oracle/fair gaps.",
    )
    cycle_report.add_argument(
        "--report",
        action="append",
        required=True,
        help=(
            "Refutation report JSON. May be '[cycle_id=]path'. Repeat for each cycle/mode; "
            "cycle ids are naturally sorted for trend calculations."
        ),
    )
    cycle_report.add_argument("--out", type=Path, default=None, help="Optional output JSON path.")
    cycle_report.set_defaults(func=_cycle_report)

    admission_guard = subparsers.add_parser(
        "admission-guard",
        help="Validate that an admission artifact has non-vacuous strength and novelty evidence.",
    )
    admission_guard.add_argument("--input", type=Path, required=True, help="Admission/gauntlet summary JSON.")
    admission_guard.add_argument("--out", type=Path, default=None, help="Optional guard result JSON path.")
    admission_guard.add_argument(
        "--min-win-rate-floor",
        type=float,
        default=0.0,
        help="Required floor is strict: observed win-rate threshold must be greater than this value.",
    )
    admission_guard.add_argument(
        "--min-comparison-vectors",
        type=int,
        default=1,
        help="Minimum comparison vectors or pairwise novelty comparisons required.",
    )
    admission_guard.add_argument(
        "--allow-missing-vector-distance",
        action="store_true",
        help="Only require comparison vectors, not a positive vector-distance threshold.",
    )
    admission_guard.set_defaults(func=_admission_guard)
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
        "--max-line-depth",
        type=int,
        default=1,
        help=(
            "Maximum loser-deviation line depth to evaluate, from 1 to 3. "
            "Depths above 1 force recorded continuation rounds before terminal rollout."
        ),
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
        max_line_depth=args.max_line_depth,
        certification_seed_count=args.certification_seeds,
        min_flip_rate=args.min_flip_rate,
        mode=args.mode,
    )


def _load_records(paths: list[Path]) -> tuple:
    records = []
    for path in paths:
        records.extend(iter_rollout_records(path))
    return tuple(records)


def _iter_records(paths: list[Path]):
    for path in paths:
        yield from iter_rollout_records(path)


def _load_records_at_indices(paths: list[Path], required_indices: Iterable[int]) -> tuple:
    needed = set(required_indices)
    if any(index < 0 for index in needed):
        raise ValueError("source_record_index must be non-negative.")
    if not needed:
        return ()
    max_index = max(needed)
    records: list[Any] = [None] * (max_index + 1)
    remaining = set(needed)
    for record_index, record in enumerate(_iter_records(paths)):
        if record_index in remaining:
            records[record_index] = record
            remaining.remove(record_index)
            if not remaining:
                break
    if remaining:
        missing = ", ".join(str(index) for index in sorted(remaining)[:10])
        raise ValueError(f"source records missing archived source_record_index values: {missing}")
    return tuple(records)


def _source_record_indices_from_fragile_rows(rows: Iterable[Mapping[str, Any]]) -> tuple[int, ...]:
    indices = []
    for row in rows:
        candidate = row.get("candidate")
        if not isinstance(candidate, Mapping):
            continue
        raw_index = candidate.get("source_record_index")
        if raw_index is None:
            continue
        indices.append(int(raw_index))
    return tuple(indices)


def _source_record_indices_from_behavior_seed_manifest(manifest: Mapping[str, Any]) -> tuple[int, ...]:
    seeds = manifest.get("seeds")
    if not isinstance(seeds, list):
        return ()
    indices = []
    for seed in seeds:
        if not isinstance(seed, Mapping):
            continue
        raw_index = seed.get("source_record_index")
        if raw_index is None:
            continue
        indices.append(int(raw_index))
    return tuple(indices)


def _plan(args: argparse.Namespace) -> int:
    records = _iter_records(args.records)
    payload = candidate_count_for_records(records=records, config=_config_from_args(args))
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def _mine(args: argparse.Namespace) -> int:
    records = _iter_records(args.records)
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
        reseed_simulator_rng=True,
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
        min_flip_rate=args.min_flip_rate,
        require_simulator_rng_reseed=not args.allow_continuation_only_reseeds,
    )
    print(json.dumps(payload, indent=2, sort_keys=True))
    if payload["r0_acceptance_eligible"]:
        return 0
    return 3 if payload["passed"] else 2


def _reproduce(args: argparse.Namespace) -> int:
    fragile_states = tuple(iter_fragile_states(args.archive))
    records = _load_records_at_indices(
        args.records,
        _source_record_indices_from_fragile_rows(fragile_states[: args.max_rows] if args.max_rows is not None else fragile_states),
    )
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
        context="refutation reproduction",
    )
    payload = reproduce_refutation_archive(
        records=records,
        fragile_states=fragile_states,
        evaluator=ReplayTerminalBranchEvaluator(
            env_factory=lambda: LocalShowdownEnv(env_config),
            policies={
                "p1": policy_from_spec(p1_spec),
                "p2": policy_from_spec(p2_spec),
            },
            rollout_config=RolloutConfig(
                max_decision_rounds=args.max_decision_rounds,
                format_id=args.format_id,
            ),
            reseed_simulator_rng=True,
        ),
        max_rows=args.max_rows,
    )
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if payload["passed"] else 2


def _training_cache(args: argparse.Namespace) -> int:
    fragile_states = tuple(iter_fragile_states(args.archive))
    records = _load_records_at_indices(args.records, _source_record_indices_from_fragile_rows(fragile_states))
    summary = write_refutation_training_cache(
        records=records,
        fragile_states=fragile_states,
        output_path=args.out,
        dataset_config=TrajectoryDatasetConfig(
            window_size=args.window_size,
            discount=args.discount,
            ppo_target_mode=args.ppo_target_mode,
            gae_lambda=args.gae_lambda,
        ),
        config=RefutationTrainingConfig(
            target_mode=args.target_mode,
            max_examples=args.max_examples,
            surprise_weight_scale=args.surprise_weight_scale,
            surprise_weight_max=args.surprise_weight_max,
        ),
        overwrite=args.overwrite,
    )
    print(json.dumps(summary.to_dict(), indent=2, sort_keys=True))
    return 0


def _behavior_seed_cache(args: argparse.Namespace) -> int:
    manifest = json.loads(args.behavior_seeds.read_text(encoding="utf-8"))
    records = _load_records_at_indices(args.records, _source_record_indices_from_behavior_seed_manifest(manifest))
    summary = write_refutation_behavior_seed_training_cache(
        records=records,
        behavior_seed_manifest=manifest,
        output_path=args.out,
        dataset_config=TrajectoryDatasetConfig(
            window_size=args.window_size,
            discount=args.discount,
            ppo_target_mode=args.ppo_target_mode,
            gae_lambda=args.gae_lambda,
        ),
        config=RefutationTrainingConfig(
            target_mode=args.target_mode,
            max_examples=args.max_examples,
            surprise_weight_scale=args.surprise_weight_scale,
            surprise_weight_max=args.surprise_weight_max,
        ),
        overwrite=args.overwrite,
    )
    print(json.dumps(summary.to_dict(), indent=2, sort_keys=True))
    return 0


def _curriculum(args: argparse.Namespace) -> int:
    fragile_states = tuple(iter_fragile_states(args.archive))
    records = _load_records_at_indices(args.records, _source_record_indices_from_fragile_rows(fragile_states))
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
        context="refutation curriculum",
    )
    rollout_config = RolloutConfig(
        max_decision_rounds=args.max_decision_rounds,
        format_id=args.format_id,
    )
    summary = collect_refutation_curriculum_rollouts(
        records=records,
        fragile_states=fragile_states,
        env_factory=lambda: LocalShowdownEnv(env_config),
        policies={
            "p1": policy_from_spec(p1_spec),
            "p2": policy_from_spec(p2_spec),
        },
        rollout_config=rollout_config,
        output_path=args.out,
        config=RefutationCurriculumConfig(
            total_games=args.total_games,
            curriculum_fraction=args.curriculum_fraction,
            seed_start=args.seed_start,
            max_starts=args.max_starts,
        ),
    )
    summary_path = args.summary or args.out.with_suffix(args.out.suffix + ".summary.json")
    write_refutation_curriculum_summary(summary_path, summary)
    print(json.dumps(summary.to_dict(), indent=2, sort_keys=True))
    return 0


def _behavior_seeds(args: argparse.Namespace) -> int:
    manifest = build_refutation_behavior_seed_manifest(
        tuple(iter_fragile_states(args.archive)),
        config=RefutationBehaviorSeedConfig(
            max_seeds=args.max_seeds,
            min_flip_rate=args.min_flip_rate,
            mode=args.mode,
        ),
    )
    write_refutation_behavior_seed_manifest(args.out, manifest)
    print(json.dumps(manifest.to_dict(), indent=2, sort_keys=True))
    return 0


def _cycle_report(args: argparse.Namespace) -> int:
    inputs = tuple(
        load_refutation_cycle_report_input(spec, default_index=index)
        for index, spec in enumerate(args.report)
    )
    report = build_refutation_cycle_report(inputs)
    if args.out is not None:
        write_refutation_cycle_report(args.out, report)
    print(json.dumps(report.to_dict(), indent=2, sort_keys=True))
    return 0


def _admission_guard(args: argparse.Namespace) -> int:
    try:
        payload = json.loads(args.input.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return _admission_guard_input_error(args, str(exc))
    if not isinstance(payload, dict):
        return _admission_guard_input_error(args, "--input must be a JSON object.")
    result = validate_admission_guard(
        payload,
        config=AdmissionGuardConfig(
            min_win_rate_floor=args.min_win_rate_floor,
            min_comparison_vectors=args.min_comparison_vectors,
            require_vector_distance=not args.allow_missing_vector_distance,
        ),
    )
    result_payload = result.to_dict()
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result_payload, indent=2, sort_keys=True))
    return 0 if result.passed else 2


def _admission_guard_input_error(args: argparse.Namespace, message: str) -> int:
    payload = {
        "schema_version": "pokezero.admission_guard.input_error.v1",
        "passed": False,
        "error": message,
    }
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 1


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
