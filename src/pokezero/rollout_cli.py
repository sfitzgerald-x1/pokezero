"""Command-line rollout collection utilities."""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
from pathlib import Path
import sys
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .observation import ObservationFeatureMasks

from .collection import (
    BenchmarkReport,
    LINEAR_POLICY_SPEC_PREFIX,
    NEURAL_POLICY_SPEC_PREFIX,
    benchmark_rollouts,
    collect_training_cache,
    collect_rollouts,
    env_config_with_policy_spec_masks,
    policy_benchmark_matchups,
    policy_from_spec,
    policy_spec_with_showdown_root,
)
from .dataset import MAX_ACTIVE_TRAINING_CACHE_GB, TrajectoryDatasetConfig, training_cache_paths_byte_size
from .local_showdown import LocalShowdownConfig, LocalShowdownEnv
from .replay_benchmark import ReplayPrefixBenchmarkReport, benchmark_replay_prefixes
from .rollout import RolloutConfig
from .selfplay import OpponentPoolEntry, collect_selfplay_rollouts
from .showdown import observation_schema_version_from_choice, observation_spec_for_schema
from .shaping import parse_shaping_spec
from .neural_selfplay import (
    DEFAULT_COLLECTION_EXPLORATION_EPSILON,
    _with_collection_sampling_options,
)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m pokezero.rollout_cli")
    subparsers = parser.add_subparsers(dest="command", required=True)

    collect = subparsers.add_parser("collect", help="Collect self-play rollout trajectories.")
    collect.add_argument("--games", type=int, required=True, help="Number of games to collect.")
    collect.add_argument("--out", type=Path, required=True, help="JSONL output path.")
    collect.add_argument("--showdown-root", type=Path, default=None, help="Built Pokemon Showdown checkout root.")
    collect.add_argument("--format", dest="format_id", default="gen3randombattle", help="Showdown format id.")
    collect.add_argument("--seed-start", type=int, default=1, help="First deterministic rollout seed.")
    collect.add_argument("--max-decision-rounds", type=int, default=250, help="Rollout decision-round cap.")
    policy_help = (
        "Policy spec. Supports random-legal, simple-legal, scripted-teacher, aggressive-damage, "
        "linear:/path/to/checkpoint.json, or neural:/path/to/checkpoint.pt."
    )
    collect.add_argument("--p1-policy", default="random-legal", help=f"Policy for p1. {policy_help}")
    collect.add_argument("--p2-policy", default="random-legal", help=f"Policy for p2. {policy_help}")
    collect.add_argument("--append", action="store_true", help="Append to the output JSONL instead of replacing it.")
    collect.add_argument("--node-binary", default="node", help="Node executable used for the BattleStream bridge.")
    collect.set_defaults(func=_collect)

    collect_cache = subparsers.add_parser(
        "collect-training-cache",
        help="Collect self-play rollouts directly into a compact neural training cache.",
    )
    collect_cache.add_argument("--games", type=int, required=True, help="Number of games to collect.")
    collect_cache.add_argument("--out", type=Path, required=True, help="Training cache output directory.")
    collect_cache.add_argument("--showdown-root", type=Path, default=None, help="Built Pokemon Showdown checkout root.")
    collect_cache.add_argument("--format", dest="format_id", default="gen3randombattle", help="Showdown format id.")
    collect_cache.add_argument("--seed-start", type=int, default=1, help="First deterministic rollout seed.")
    collect_cache.add_argument("--max-decision-rounds", type=int, default=250, help="Rollout decision-round cap.")
    collect_cache.add_argument("--p1-policy", default="random-legal", help=f"Policy for p1. {policy_help}")
    collect_cache.add_argument("--p2-policy", default="random-legal", help=f"Policy for p2. {policy_help}")
    collect_cache.add_argument("--node-binary", default="node", help="Node executable used for the BattleStream bridge.")
    collect_cache.add_argument("--overwrite", action="store_true", help="Replace an existing training cache directory.")
    collect_cache.add_argument(
        "--max-cache-gb",
        type=float,
        default=MAX_ACTIVE_TRAINING_CACHE_GB,
        help=(
            "Reject the write if existing caches under the output parent plus the new cache "
            f"would exceed this many GiB (default and maximum: {MAX_ACTIVE_TRAINING_CACHE_GB:g})."
        ),
    )
    _add_dataset_config_arguments(collect_cache)
    collect_cache.set_defaults(func=_collect_training_cache)

    collect_selfplay_cache = subparsers.add_parser(
        "collect-selfplay-training-cache",
        help=(
            "Collect the current-policy self-play training perspective directly into compact "
            "neural training cache shards."
        ),
        epilog=(
            "Fleet wrappers must assign each shard a disjoint --seed-start range. Dataset flags "
            "such as --window-size, --discount, --ppo-target-mode, and --gae-lambda are baked "
            "into the cache and must match the central trainer config."
        ),
    )
    collect_selfplay_cache.add_argument("--games", type=int, required=True, help="Number of games to collect.")
    collect_selfplay_cache.add_argument("--out", type=Path, required=True, help="Training cache output directory.")
    collect_selfplay_cache.add_argument(
        "--showdown-root", type=Path, default=None, help="Built Pokemon Showdown checkout root."
    )
    collect_selfplay_cache.add_argument(
        "--format", dest="format_id", default="gen3randombattle", help="Showdown format id."
    )
    collect_selfplay_cache.add_argument(
        "--seed-start",
        type=int,
        default=1,
        help="First deterministic rollout seed. Fleet shards must use non-overlapping seed ranges.",
    )
    collect_selfplay_cache.add_argument(
        "--max-decision-rounds", type=int, default=250, help="Rollout decision-round cap."
    )
    collect_selfplay_cache.add_argument(
        "--current-policy",
        default="random-legal",
        help=f"Current policy spec whose training perspective is recorded. {policy_help}",
    )
    collect_selfplay_cache.add_argument(
        "--opponent-policy",
        action="append",
        default=None,
        help=(
            "Opponent policy spec. May be repeated. When omitted, mirrors --current-policy "
            "for teacher-cut/current-policy self-play."
        ),
    )
    collect_selfplay_cache.add_argument(
        "--opponent-pool",
        type=Path,
        default=None,
        help=(
            "JSON opponent-pool manifest for diversity-tier collection. Accepts a top-level list "
            "or {\"opponents\": [...]} entries with policy_spec/spec or checkpoint_path, optional "
            "weight, and optional member_id/id/name. Cannot be combined with --opponent-policy."
        ),
    )
    collect_selfplay_cache.add_argument(
        "--opponent-pool-self-play-share",
        type=float,
        default=0.5,
        help=(
            "When --opponent-pool is used, reserve this opponent-sampling share for a mirror "
            "match against --current-policy before drawing from the pool weights (default: 0.5)."
        ),
    )
    collect_selfplay_cache.add_argument(
        "--collection-epsilon",
        type=float,
        default=DEFAULT_COLLECTION_EXPLORATION_EPSILON,
        help=(
            "Minimum random legal exploration rate for learned policies during self-play "
            "training-cache collection. Applies only to collection; scripted/random/simple "
            f"policies are unchanged. Default {DEFAULT_COLLECTION_EXPLORATION_EPSILON}."
        ),
    )
    # Ablation-arm feature masks for the checkpoint-less iteration-0 case (the cluster
    # controller collects with current-policy=random-legal before any checkpoint exists).
    # When a neural: policy IS given, masks adopt from the checkpoint and explicit flags
    # must agree or hard-fail (env_config_with_checkpoint_masks conflict semantics). The
    # resolved masks are recorded in the cache metadata either way, so the subsequent
    # train cross-checks them against the model config.
    collect_selfplay_cache.add_argument(
        "--observation-schema",
        choices=("v2.1", "v2.2", "v3"),
        default=None,
        help=(
            "Observation schema for a FRESH (checkpoint-less) collect: v2.1, "
            "v2.2 (default; turn-merged transition tokens; also flips the schema-derived vocabulary), "
            "or v3 (turn-merged grouped layout with the V3 public signals). "
            "With a neural: policy the checkpoint's stamped schema wins and an explicitly "
            "disagreeing flag hard-fails (mask-conflict semantics). Recorded in cache "
            "metadata for the train-side cross-check."
        ),
    )
    collect_selfplay_cache.add_argument(
        "--transition-token-budget",
        type=int,
        default=None,
        help="Most-recent transition-token slots filled at encode time. UNIT IS SCHEMA-DEPENDENT: under v2/v2.1 a token is one declared ACTION (32 = the K=16-turn ablation arm); under --observation-schema v2.2 a token is a whole TURN, so budget 32 covers roughly what 64 action-tokens did.",
    )
    collect_selfplay_cache.add_argument(
        "--no-stats-block",
        action="store_true",
        help="Ablation arm: zero + attention-mask the stats token and per-mon tendency triple.",
    )
    collect_selfplay_cache.add_argument(
        "--no-exact-state",
        action="store_true",
        help="Ablation arm: zero the exact-state layer (PP fractions, counters, expected stats).",
    )
    collect_selfplay_cache.add_argument(
        "--tier2-residuals",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Populate + encode the Tier-2 residual channel (#505). Default: on for checkpoint-less collection; adopted from the checkpoint otherwise.",
    )
    collect_selfplay_cache.add_argument(
        "--tier2-investment",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Populate + encode the defender-side investment channel (#513): the reserved "
            "investment columns (120/139) behind the investment precision gate. A SEPARATE "
            "switch from --tier2-residuals. Default: OFF for checkpoint-less collection "
            "(byte-identical to pre-investment); adopted from the checkpoint otherwise. Only "
            "meaningful under --observation-schema v2.1/v2.2 (no-op under v2). Recorded in "
            "cache metadata for the train-side cross-check."
        ),
    )
    collect_selfplay_cache.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Worker threads within this shard. For fleet collection, prefer more pods over a large value.",
    )
    collect_selfplay_cache.add_argument(
        "--chunk-games",
        type=int,
        default=None,
        help="Optional games per cache chunk. When omitted, writes one cache at --out.",
    )
    collect_selfplay_cache.add_argument(
        "--max-cache-gb",
        type=float,
        default=MAX_ACTIVE_TRAINING_CACHE_GB,
        help=(
            "Reject the write if existing caches under the output parent plus the new cache "
            f"would exceed this many GiB (default and maximum: {MAX_ACTIVE_TRAINING_CACHE_GB:g})."
        ),
    )
    collect_selfplay_cache.add_argument(
        "--node-binary", default="node", help="Node executable used for the BattleStream bridge."
    )
    _add_dataset_config_arguments(collect_selfplay_cache)
    collect_selfplay_cache.set_defaults(func=_collect_selfplay_training_cache)

    benchmark = subparsers.add_parser("benchmark", help="Run rollout benchmarks without writing trajectories.")
    benchmark.add_argument(
        "--games",
        type=int,
        default=20,
        help="Number of games to run per matchup. Default is a smoke size; use hundreds for quality comparisons.",
    )
    benchmark.add_argument("--showdown-root", type=Path, default=None, help="Built Pokemon Showdown checkout root.")
    benchmark.add_argument("--format", dest="format_id", default="gen3randombattle", help="Showdown format id.")
    benchmark.add_argument("--seed-start", type=int, default=1, help="First deterministic rollout seed for every matchup.")
    benchmark.add_argument("--max-decision-rounds", type=int, default=250, help="Rollout decision-round cap.")
    benchmark.add_argument("--node-binary", default="node", help="Node executable used for the BattleStream bridge.")
    benchmark.add_argument(
        "--policy",
        action="append",
        default=None,
        help=(
            "Candidate policy spec to benchmark. May be repeated. When omitted, the command runs "
            "the default baseline throughput matchups."
        ),
    )
    benchmark.add_argument(
        "--opponent-policy",
        action="append",
        default=None,
        help=(
            "Opponent policy spec for custom --policy benchmarks. May be repeated. "
            "Defaults to random-legal and simple-legal."
        ),
    )
    benchmark.add_argument(
        "--include-policy-head-to-head",
        action="store_true",
        help="Also run mirrored candidate-vs-candidate matchups when multiple --policy values are provided.",
    )
    benchmark.add_argument("--json", action="store_true", help="Print benchmark results as JSON.")
    benchmark.set_defaults(func=_benchmark)

    replay_benchmark = subparsers.add_parser(
        "replay-benchmark",
        help="Measure replay-from-root prefix latency for future search/MCTS planning.",
    )
    replay_benchmark.add_argument("--games", type=int, default=3, help="Number of source games to generate.")
    replay_benchmark.add_argument(
        "--prefixes-per-game",
        type=int,
        default=5,
        help="Evenly sampled branch-prefix lengths to replay per source game.",
    )
    replay_benchmark.add_argument("--showdown-root", type=Path, default=None, help="Built Pokemon Showdown checkout root.")
    replay_benchmark.add_argument("--format", dest="format_id", default="gen3randombattle", help="Showdown format id.")
    replay_benchmark.add_argument("--seed-start", type=int, default=1, help="First deterministic rollout seed.")
    replay_benchmark.add_argument("--max-decision-rounds", type=int, default=250, help="Source rollout decision-round cap.")
    replay_benchmark.add_argument("--node-binary", default="node", help="Node executable used for the BattleStream bridge.")
    replay_benchmark.add_argument("--p1-policy", default="random-legal", help=f"Source policy for p1. {policy_help}")
    replay_benchmark.add_argument("--p2-policy", default="random-legal", help=f"Source policy for p2. {policy_help}")
    replay_benchmark.add_argument("--json", action="store_true", help="Print replay benchmark results as JSON.")
    replay_benchmark.set_defaults(func=_replay_benchmark)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


def _collect(args: argparse.Namespace) -> int:
    env_config = LocalShowdownConfig(
        showdown_root=args.showdown_root,
        node_binary=args.node_binary,
    )
    policy_showdown_root = env_config.resolved_showdown_root()
    env_config = env_config_with_policy_spec_masks(
        env_config, (args.p1_policy, args.p2_policy), context="rollout harness"
    )
    policies = {
        "p1": policy_from_spec(policy_spec_with_showdown_root(args.p1_policy, policy_showdown_root)),
        "p2": policy_from_spec(policy_spec_with_showdown_root(args.p2_policy, policy_showdown_root)),
    }
    rollout_config = RolloutConfig(
        max_decision_rounds=args.max_decision_rounds,
        format_id=args.format_id,
    )
    metrics = collect_rollouts(
        output_path=args.out,
        games=args.games,
        env_factory=lambda: LocalShowdownEnv(env_config),
        policies=policies,
        rollout_config=rollout_config,
        seed_start=args.seed_start,
        append=args.append,
    )
    _print_metrics(metrics.to_dict())
    return 0


def _collect_training_cache(args: argparse.Namespace) -> int:
    env_config = LocalShowdownConfig(
        showdown_root=args.showdown_root,
        node_binary=args.node_binary,
    )
    policy_showdown_root = env_config.resolved_showdown_root()
    env_config = env_config_with_policy_spec_masks(
        env_config, (args.p1_policy, args.p2_policy), context="rollout harness"
    )
    policies = {
        "p1": policy_from_spec(policy_spec_with_showdown_root(args.p1_policy, policy_showdown_root)),
        "p2": policy_from_spec(policy_spec_with_showdown_root(args.p2_policy, policy_showdown_root)),
    }
    rollout_config = RolloutConfig(
        max_decision_rounds=args.max_decision_rounds,
        format_id=args.format_id,
    )
    metrics, cache = collect_training_cache(
        output_path=args.out,
        games=args.games,
        env_factory=lambda: LocalShowdownEnv(env_config),
        policies=policies,
        rollout_config=rollout_config,
        dataset_config=_dataset_config_from_args(args),
        seed_start=args.seed_start,
        overwrite=args.overwrite,
        max_cache_root_bytes=_cache_gb_to_bytes(args.max_cache_gb),
        cache_root=args.out.parent,
    )
    _print_metrics(metrics.to_dict())
    print(f"training_cache: {cache.path}")
    print(f"training_cache_examples: {cache.example_count}")
    print(f"training_cache_bytes: {cache.byte_size}")
    return 0


def _explicit_feature_masks_from_args(
    args: argparse.Namespace,
    *,
    base_masks: ObservationFeatureMasks | None = None,
    transition_token_capacity: int | None = None,
) -> ObservationFeatureMasks | None:
    """Explicit encode-time masks when any mask flag was given, else None (use defaults
    or checkpoint adoption). Unspecified fields inherit from ``base_masks`` so one explicit
    flag cannot silently reset a checkpoint-adopted mask on another axis."""
    budget = getattr(args, "transition_token_budget", None)
    tier2 = getattr(args, "tier2_residuals", None)
    investment = getattr(args, "tier2_investment", None)
    no_stats = bool(getattr(args, "no_stats_block", False))
    no_exact = bool(getattr(args, "no_exact_state", False))
    if (
        budget is None
        and tier2 is None
        and investment is None
        and not no_stats
        and not no_exact
    ):
        return None
    from .observation import (
        DEFAULT_OBSERVATION_FEATURE_MASKS,
        TRANSITION_TOKEN_COUNT,
        ObservationFeatureMasks,
    )

    base = base_masks or DEFAULT_OBSERVATION_FEATURE_MASKS
    capacity = transition_token_capacity or TRANSITION_TOKEN_COUNT
    resolved_budget = min(base.transition_token_budget, capacity) if budget is None else budget
    if not 0 < resolved_budget <= capacity:
        raise ValueError(
            f"transition_token_budget must be in 1..{capacity} for the resolved "
            f"observation schema, got {resolved_budget}."
        )

    return ObservationFeatureMasks(
        opponent_tendency_stats_block=(
            False if no_stats else base.opponent_tendency_stats_block
        ),
        exact_state=False if no_exact else base.exact_state,
        transition_token_budget=resolved_budget,
        tier2_residuals=base.tier2_residuals if tier2 is None else bool(tier2),
        tier2_investment=(
            base.tier2_investment if investment is None else bool(investment)
        ),
    )


def _collect_selfplay_training_cache(args: argparse.Namespace) -> int:
    env_config = LocalShowdownConfig(
        showdown_root=args.showdown_root,
        node_binary=args.node_binary,
    )
    policy_showdown_root = env_config.resolved_showdown_root()
    current_policy = _with_collection_sampling_options(
        policy_spec_with_showdown_root(args.current_policy, policy_showdown_root),
        temperature=1.0,
        exploration_epsilon=args.collection_epsilon,
    )
    if args.opponent_pool is not None and args.opponent_policy:
        raise ValueError("--opponent-pool cannot be combined with --opponent-policy.")
    opponent_pool_entries = (
        _load_opponent_pool_manifest(args.opponent_pool, showdown_root=policy_showdown_root)
        if args.opponent_pool is not None
        else None
    )
    if opponent_pool_entries is not None:
        opponent_pool_entries = tuple(
            dataclasses.replace(
                entry,
                policy_spec=_with_collection_sampling_options(
                    entry.policy_spec,
                    temperature=1.0,
                    exploration_epsilon=args.collection_epsilon,
                ),
            )
            for entry in opponent_pool_entries
        )
    opponent_policies = (
        tuple(entry.policy_spec for entry in opponent_pool_entries)
        if opponent_pool_entries is not None
        else tuple(
            _with_collection_sampling_options(
                policy_spec_with_showdown_root(spec, policy_showdown_root),
                temperature=1.0,
                exploration_epsilon=args.collection_epsilon,
            )
            for spec in (args.opponent_policy or (args.current_policy,))
        )
    )
    requested_schema = observation_schema_version_from_choice(args.observation_schema)
    if requested_schema is not None:
        env_config = dataclasses.replace(
            env_config, observation_spec=observation_spec_for_schema(requested_schema)
        )
    # Adopt checkpoint provenance before resolving partial explicit masks. This lets an
    # omitted mask field inherit the checkpoint value instead of resetting to a global
    # default, while the second latch below still rejects any explicit disagreement.
    env_config = env_config_with_policy_spec_masks(
        env_config, (current_policy, *opponent_policies), context="self-play training cache"
    )
    explicit_masks = _explicit_feature_masks_from_args(
        args,
        base_masks=env_config.feature_masks,
        transition_token_capacity=env_config.observation_spec.transition_token_count,
    )
    if explicit_masks is not None:
        env_config = dataclasses.replace(env_config, feature_masks=explicit_masks)
        env_config = env_config_with_policy_spec_masks(
            env_config,
            (current_policy, *opponent_policies),
            context="self-play training cache",
        )
    elif (
        env_config.feature_masks.transition_token_budget
        > env_config.observation_spec.transition_token_count
    ):
        # The schema-agnostic default mask retains the legacy 128-row ceiling. Persist the
        # effective schema capacity so a fresh V3 cache agrees with its train-side config.
        env_config = dataclasses.replace(
            env_config,
            feature_masks=dataclasses.replace(
                env_config.feature_masks,
                transition_token_budget=env_config.observation_spec.transition_token_count,
            ),
        )
    # Explicit flag vs checkpoint schema: hard-fail BOTH directions. (An explicit v2.2
    # against a v2.1 checkpoint already fails inside the latch; an explicit v2.1 equals
    # the default spec, which the latch would silently override with a v2.2 checkpoint's
    # — catch that here.)
    if requested_schema is not None and env_config.observation_spec.schema_version != requested_schema:
        raise ValueError(
            f"--observation-schema {args.observation_schema} conflicts with the checkpoint-"
            f"stamped schema {env_config.observation_spec.schema_version!r}; the schema "
            "cannot change across a resume/adoption (drop the flag to adopt the "
            "checkpoint's schema)."
        )
    rollout_config = RolloutConfig(
        max_decision_rounds=args.max_decision_rounds,
        format_id=args.format_id,
    )
    cache_paths: list[Path] = []
    metrics = collect_selfplay_rollouts(
        output_path=None,
        training_output_path=None,
        training_cache_output_path=args.out,
        training_cache_chunk_games=args.chunk_games,
        training_cache_dataset_config=_dataset_config_from_args(args),
        training_cache_max_root_bytes=_cache_gb_to_bytes(args.max_cache_gb),
        training_cache_root=args.out.parent,
        training_cache_paths_out=cache_paths,
        games=args.games,
        env_factory=lambda: LocalShowdownEnv(env_config),
        rollout_config=rollout_config,
        seed_start=args.seed_start,
        current_policy_spec=current_policy,
        opponent_policy_specs=opponent_policies,
        opponent_pool_entries=opponent_pool_entries,
        opponent_pool_self_play_share=(
            args.opponent_pool_self_play_share if opponent_pool_entries is not None else 0.0
        ),
        worker_count=args.workers,
        training_cache_feature_masks=env_config.feature_masks,
        training_cache_observation_schema=env_config.observation_spec.schema_version,
    )
    _print_metrics(metrics.to_dict())
    if opponent_pool_entries is not None:
        print(f"opponent_pool: {args.opponent_pool}")
        print(f"opponent_pool_members: {len(opponent_pool_entries)}")
        print(f"opponent_pool_self_play_share: {args.opponent_pool_self_play_share:g}")
    for cache_path in cache_paths:
        print(f"training_cache: {cache_path}")
    print(f"training_cache_count: {len(cache_paths)}")
    if cache_paths:
        print(f"training_cache_bytes: {training_cache_paths_byte_size(cache_paths)}")
    return 0


def _load_opponent_pool_manifest(path: Path, *, showdown_root: Path) -> tuple[OpponentPoolEntry, ...]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    raw_entries = payload.get("opponents") if isinstance(payload, dict) else payload
    if raw_entries is None and isinstance(payload, dict):
        raw_entries = payload.get("members")
    if not isinstance(raw_entries, list):
        raise ValueError("opponent pool manifest must be a list or contain an 'opponents' list.")
    entries: list[OpponentPoolEntry] = []
    for index, raw_entry in enumerate(raw_entries):
        if not isinstance(raw_entry, dict):
            raise ValueError(f"opponent pool entry {index} must be an object.")
        policy_spec, checkpoint_path = _opponent_pool_entry_policy_spec(raw_entry, manifest_path=path)
        policy_spec = policy_spec_with_showdown_root(policy_spec, showdown_root)
        weight = float(raw_entry.get("weight", 1.0))
        member_id = raw_entry.get("member_id", raw_entry.get("id", raw_entry.get("name")))
        checkpoint_hash = raw_entry.get("checkpoint_hash", raw_entry.get("hash"))
        if checkpoint_hash is None and checkpoint_path is not None and checkpoint_path.is_file():
            checkpoint_hash = _file_sha256(checkpoint_path)
        entries.append(
            OpponentPoolEntry(
                policy_spec=policy_spec,
                weight=weight,
                member_id=str(member_id) if member_id is not None else None,
                checkpoint_hash=str(checkpoint_hash) if checkpoint_hash is not None else None,
            )
        )
    if not entries:
        raise ValueError("opponent pool manifest must include at least one member.")
    return tuple(entries)


def _opponent_pool_entry_policy_spec(raw_entry: dict, *, manifest_path: Path) -> tuple[str, Path | None]:
    raw_spec = raw_entry.get("policy_spec", raw_entry.get("spec"))
    if raw_spec is not None:
        spec = str(raw_spec).strip()
        if not spec:
            raise ValueError("opponent pool policy_spec must be non-empty.")
        return spec, None
    checkpoint = raw_entry.get("checkpoint_path", raw_entry.get("checkpoint"))
    if checkpoint is None:
        raise ValueError("opponent pool entry must include policy_spec/spec or checkpoint_path/checkpoint.")
    checkpoint_value = str(checkpoint).strip()
    if not checkpoint_value:
        raise ValueError("opponent pool checkpoint_path must be non-empty.")
    checkpoint_path = Path(checkpoint_value)
    if not checkpoint_path.is_absolute():
        checkpoint_path = manifest_path.parent / checkpoint_path
    policy_type = str(raw_entry.get("policy_type", raw_entry.get("type", "neural"))).strip().lower()
    if policy_type == "linear":
        return f"{LINEAR_POLICY_SPEC_PREFIX}{checkpoint_path}", checkpoint_path
    if policy_type == "neural":
        return f"{NEURAL_POLICY_SPEC_PREFIX}{checkpoint_path}", checkpoint_path
    raise ValueError("opponent pool checkpoint entry policy_type must be 'neural' or 'linear'.")


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _benchmark(args: argparse.Namespace) -> int:
    env_config = LocalShowdownConfig(
        showdown_root=args.showdown_root,
        node_binary=args.node_binary,
    )
    policy_showdown_root = env_config.resolved_showdown_root()
    rollout_config = RolloutConfig(
        max_decision_rounds=args.max_decision_rounds,
        format_id=args.format_id,
    )
    matchups = None
    env_config = env_config_with_policy_spec_masks(
        env_config,
        (*(args.policy or ()), *(args.opponent_policy or ())),
        context="rollout benchmark",
    )
    if args.policy:
        matchups = policy_benchmark_matchups(
            policy_specs=args.policy,
            opponent_policy_specs=args.opponent_policy or ("random-legal", "simple-legal"),
            showdown_root=policy_showdown_root,
            include_policy_head_to_head=args.include_policy_head_to_head,
        )
    elif args.opponent_policy:
        raise ValueError("--opponent-policy requires at least one --policy.")
    elif args.include_policy_head_to_head:
        raise ValueError("--include-policy-head-to-head requires at least two --policy values.")

    report = benchmark_rollouts(
        games=args.games,
        env_factory=lambda: LocalShowdownEnv(env_config),
        rollout_config=rollout_config,
        seed_start=args.seed_start,
        matchups=matchups,
    )
    if args.json:
        print(json.dumps(report.to_dict(), indent=2, sort_keys=True))
    else:
        print_benchmark_report(report)
    return 0


def _replay_benchmark(args: argparse.Namespace) -> int:
    env_config = LocalShowdownConfig(
        showdown_root=args.showdown_root,
        node_binary=args.node_binary,
    )
    policy_showdown_root = env_config.resolved_showdown_root()
    env_config = env_config_with_policy_spec_masks(
        env_config, (args.p1_policy, args.p2_policy), context="rollout harness"
    )
    policies = {
        "p1": policy_from_spec(policy_spec_with_showdown_root(args.p1_policy, policy_showdown_root)),
        "p2": policy_from_spec(policy_spec_with_showdown_root(args.p2_policy, policy_showdown_root)),
    }
    rollout_config = RolloutConfig(
        max_decision_rounds=args.max_decision_rounds,
        format_id=args.format_id,
    )
    report = benchmark_replay_prefixes(
        env_factory=lambda: LocalShowdownEnv(env_config),
        policies=policies,
        rollout_config=rollout_config,
        games=args.games,
        prefixes_per_game=args.prefixes_per_game,
        seed_start=args.seed_start,
    )
    if args.json:
        print(json.dumps(report.to_dict(), indent=2, sort_keys=True))
    else:
        print_replay_benchmark_report(report)
    return 0


def _print_metrics(metrics: dict[str, object]) -> None:
    print(f"games: {metrics['games']}")
    print(f"elapsed_seconds: {float(metrics['elapsed_seconds']):.3f}")
    print(f"games_per_second: {float(metrics['games_per_second']):.3f}")
    print(f"decisions_per_second: {float(metrics['decisions_per_second']):.3f}")
    if metrics.get("peak_rss_mb") is not None:
        print(f"peak_rss_mb: {float(metrics['peak_rss_mb']):.2f}")
    print(f"p1_wins: {metrics['p1_wins']}")
    print(f"p2_wins: {metrics['p2_wins']}")
    print(f"ties: {metrics['ties']}")
    print(f"capped_games: {metrics['capped_games']}")
    print(f"average_decision_rounds: {float(metrics['average_decision_rounds']):.2f}")
    print(f"average_simulator_turns: {float(metrics['average_simulator_turns']):.2f}")
    timing = metrics.get("collection_timing")
    if isinstance(timing, dict):
        for key in sorted(timing):
            value = timing[key]
            if isinstance(value, bool) or value is None:
                continue
            if isinstance(value, int):
                print(f"collection_timing_{key}: {value}")
            elif isinstance(value, float):
                print(f"collection_timing_{key}: {value:.6f}")


def _add_dataset_config_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--window-size", type=int, default=1, help="Per-player observation history window (spec v2 default: 1).")
    parser.add_argument("--discount", type=float, default=1.0, help="Terminal return discount per player decision.")
    parser.add_argument("--capped-terminal-value", type=float, default=0.0, help="Return assigned to each player in capped games.")
    parser.add_argument(
        "--hp-delta-return-weight",
        type=float,
        default=0.0,
        help="Optional return-shaping weight for visible player-relative HP differential changes.",
    )
    parser.add_argument(
        "--faint-delta-return-weight",
        type=float,
        default=0.0,
        help="Optional return-shaping weight for visible player-relative faint differential changes.",
    )
    parser.add_argument(
        "--turn-penalty-after",
        type=int,
        default=None,
        help="Optional turn index at which to start applying a per-decision shaped return penalty.",
    )
    parser.add_argument(
        "--turn-penalty",
        type=float,
        default=0.0,
        help="Optional positive per-decision return penalty applied at or after --turn-penalty-after.",
    )
    parser.add_argument(
        "--ppo-target-mode",
        choices=("returns", "gae"),
        default="returns",
        help="PPO advantage/value-target source baked into the cache.",
    )
    parser.add_argument("--gae-lambda", type=float, default=0.95, help="GAE lambda when --ppo-target-mode=gae.")
    parser.add_argument(
        "--shaping-weights",
        default=None,
        help=(
            "Optional dense potential-based reward shaping baked into cache returns/targets: "
            "a preset (wse-arm1), inline JSON weights, or @/path/to.json. Default off — absent "
            "means exactly the unshaped collection behavior (byte-identical caches). The shaping "
            "gamma is --discount; raw rewards and per-step shaping components are stored separately."
        ),
    )


def _dataset_config_from_args(args: argparse.Namespace) -> TrajectoryDatasetConfig:
    return TrajectoryDatasetConfig(
        window_size=args.window_size,
        discount=args.discount,
        capped_terminal_value=args.capped_terminal_value,
        hp_delta_return_weight=args.hp_delta_return_weight,
        faint_delta_return_weight=args.faint_delta_return_weight,
        turn_penalty_after=args.turn_penalty_after,
        turn_penalty=args.turn_penalty,
        ppo_target_mode=args.ppo_target_mode,
        gae_lambda=args.gae_lambda,
        potential_shaping=(
            parse_shaping_spec(args.shaping_weights) if args.shaping_weights is not None else None
        ),
    )


def _cache_gb_to_bytes(value: float | None) -> int:
    resolved = MAX_ACTIVE_TRAINING_CACHE_GB if value is None else value
    if resolved <= 0:
        raise ValueError("--max-cache-gb must be positive.")
    if resolved > MAX_ACTIVE_TRAINING_CACHE_GB:
        raise ValueError(f"--max-cache-gb cannot exceed {MAX_ACTIVE_TRAINING_CACHE_GB:g}.")
    return int(resolved * 1024 * 1024 * 1024)


def print_benchmark_report(report: BenchmarkReport) -> None:
    print(f"format: {report.format_id}")
    print(f"games_per_matchup: {report.games_per_matchup}")
    print(f"max_decision_rounds: {report.max_decision_rounds}")
    print(f"total_games: {report.total_games}")
    print(f"elapsed_seconds: {report.elapsed_seconds:.3f}")
    print(f"games_per_second: {report.games_per_second:.3f}")
    print(f"decisions_per_second: {report.decisions_per_second:.3f}")
    if report.peak_rss_mb is not None:
        print(f"peak_rss_mb: {report.peak_rss_mb:.2f}")
    print("note: default --games is a throughput smoke; use larger N for policy-quality claims.")
    print("")
    header = (
        f"{'matchup':32} {'games':>5} {'p1_wins':>7} {'p2_wins':>7} {'ties':>4} {'capped':>6} "
        f"{'avg_dec':>8} {'avg_turns':>9} {'games/s':>8} {'dec/s':>8}"
    )
    print(header)
    print("-" * len(header))
    for result in report.matchups:
        metrics = result.metrics
        print(
            f"{result.label[:32]:32} "
            f"{metrics.games:5d} "
            f"{metrics.p1_wins:7d} "
            f"{metrics.p2_wins:7d} "
            f"{metrics.ties:4d} "
            f"{metrics.capped_games:6d} "
            f"{metrics.average_decision_rounds:8.2f} "
            f"{metrics.average_simulator_turns:9.2f} "
            f"{metrics.games_per_second:8.3f} "
            f"{metrics.decisions_per_second:8.3f}"
        )
    _print_policy_decision_diagnostics(report)
    if not report.head_to_head_results:
        return
    print("")
    head_to_head_header = (
        f"{'mirror head-to-head':32} {'games':>5} {'first_w':>7} {'second_w':>8} {'ties':>4} {'capped':>6} "
        f"{'first_wr':>8} {'second_wr':>9}"
    )
    print(head_to_head_header)
    print("-" * len(head_to_head_header))
    for result in report.head_to_head_results:
        print(
            f"{result.label[:32]:32} "
            f"{result.games:5d} "
            f"{result.first_policy_wins:7d} "
            f"{result.second_policy_wins:8d} "
            f"{result.ties:4d} "
            f"{result.capped_games:6d} "
            f"{result.first_policy_win_rate:8.3f} "
            f"{result.second_policy_win_rate:9.3f}"
        )


def _print_policy_decision_diagnostics(report: BenchmarkReport) -> None:
    rows = []
    for result in report.matchups:
        summary = result.metrics.policy_decision_summary or {}
        for policy_id, metrics in summary.items():
            if "root_puct_searches" not in metrics and "root_puct_fallbacks" not in metrics:
                continue
            rows.append((result.label, policy_id, metrics))
    if not rows:
        return

    print("")
    print("root-puct diagnostics:")
    header = (
        f"{'matchup':32} {'policy':32} {'dec':>5} {'search':>6} {'visits':>7} {'fallback':>8} "
        f"{'gate':>8} {'cand':>7} {'ms/dec':>8} {'value':>8} {'score':>8}"
    )
    print(header)
    print("-" * len(header))
    fallback_categories = []
    selection_modes = []
    opponent_action_policies = []
    opponent_action_scenario_counts = []
    leaf_rollouts = []
    leaf_rollout_opponents = []
    leaf_actual_rounds = []
    leaf_evaluations = []
    for matchup, policy_id, metrics in rows:
        average_candidate_count = metrics.get("root_puct_average_candidate_count")
        average_elapsed_seconds = metrics.get("root_puct_average_elapsed_seconds")
        average_selected_value = metrics.get("root_puct_average_selected_value")
        average_selected_score = metrics.get("root_puct_average_selected_score")
        value_gate_checks = metrics.get("root_puct_value_gate_checks")
        value_gate_uses = metrics.get("root_puct_value_gate_uses")
        print(
            f"{matchup[:32]:32} "
            f"{policy_id[:32]:32} "
            f"{int(metrics.get('decisions', 0)):5d} "
            f"{int(metrics.get('root_puct_searches', 0)):6d} "
            f"{int(metrics.get('root_puct_total_visits', 0)):7d} "
            f"{int(metrics.get('root_puct_fallbacks', 0)):8d} "
            f"{_optional_report_ratio(value_gate_uses, value_gate_checks):>8} "
            f"{_optional_report_float(average_candidate_count):>7} "
            f"{_optional_report_millis(average_elapsed_seconds):>8} "
            f"{_optional_report_float(average_selected_value):>8} "
            f"{_optional_report_float(average_selected_score):>8}"
        )
        categories = metrics.get("root_puct_fallback_categories")
        if isinstance(categories, dict) and categories:
            formatted = ", ".join(f"{category}={count}" for category, count in categories.items())
            fallback_categories.append(f"{policy_id}: {formatted}")
        modes = metrics.get("root_puct_selection_modes")
        if isinstance(modes, dict) and modes:
            formatted = ", ".join(f"{mode}={count}" for mode, count in modes.items())
            selection_modes.append(f"{policy_id}: {formatted}")
        root_opponent_policies = metrics.get("root_puct_opponent_action_policies")
        if isinstance(root_opponent_policies, dict) and root_opponent_policies:
            formatted = ", ".join(f"{name}={count}" for name, count in root_opponent_policies.items())
            opponent_action_policies.append(f"{policy_id}: {formatted}")
        root_opponent_scenarios = metrics.get("root_puct_opponent_action_scenario_counts")
        if isinstance(root_opponent_scenarios, dict) and root_opponent_scenarios:
            formatted = ", ".join(f"{count}={decisions}" for count, decisions in root_opponent_scenarios.items())
            opponent_action_scenario_counts.append(f"{policy_id}: {formatted}")
        leaf_rounds = metrics.get("root_puct_leaf_rollout_rounds")
        if isinstance(leaf_rounds, dict) and leaf_rounds:
            formatted = ", ".join(f"{rounds}={count}" for rounds, count in leaf_rounds.items())
            leaf_rollouts.append(f"{policy_id}: {formatted}")
        leaf_opponents = metrics.get("root_puct_leaf_rollout_opponent_policies")
        if isinstance(leaf_opponents, dict) and leaf_opponents:
            formatted = ", ".join(f"{name}={count}" for name, count in leaf_opponents.items())
            leaf_rollout_opponents.append(f"{policy_id}: {formatted}")
        actual_rounds = metrics.get("root_puct_leaf_actual_rollout_rounds")
        if isinstance(actual_rounds, dict) and actual_rounds:
            formatted = ", ".join(f"{rounds}={count}" for rounds, count in actual_rounds.items())
            leaf_actual_rounds.append(f"{policy_id}: {formatted}")
        evaluations = metrics.get("root_puct_leaf_evaluations")
        if isinstance(evaluations, dict) and evaluations:
            formatted = ", ".join(f"{name}={count}" for name, count in evaluations.items())
            leaf_evaluations.append(f"{policy_id}: {formatted}")
    if selection_modes:
        print("selection_modes:")
        for mode in selection_modes:
            print(f"  {mode}")
    if opponent_action_policies:
        print("opponent_action_policies:")
        for policy in opponent_action_policies:
            print(f"  {policy}")
    if opponent_action_scenario_counts:
        print("opponent_action_scenario_counts:")
        for scenario_count in opponent_action_scenario_counts:
            print(f"  {scenario_count}")
    if leaf_rollouts:
        print("leaf_rollouts_configured:")
        for leaf_rollout in leaf_rollouts:
            print(f"  {leaf_rollout}")
    if leaf_rollout_opponents:
        print("leaf_rollout_opponents:")
        for leaf_opponent in leaf_rollout_opponents:
            print(f"  {leaf_opponent}")
    if leaf_actual_rounds:
        print("leaf_rollouts_actual:")
        for leaf_rounds in leaf_actual_rounds:
            print(f"  {leaf_rounds}")
    if leaf_evaluations:
        print("leaf_evaluations:")
        for leaf_evaluation in leaf_evaluations:
            print(f"  {leaf_evaluation}")
    if fallback_categories:
        print("fallback_categories:")
        for category in fallback_categories:
            print(f"  {category}")


def _optional_report_float(value: object) -> str:
    if value is None:
        return "-"
    try:
        return f"{float(value):.3f}"
    except (TypeError, ValueError):
        return "-"


def _optional_report_ratio(numerator: object, denominator: object) -> str:
    if numerator is None or denominator is None:
        return "-"
    try:
        return f"{int(numerator)}/{int(denominator)}"
    except (TypeError, ValueError):
        return "-"


def _optional_report_millis(value: object) -> str:
    if value is None:
        return "-"
    try:
        return f"{float(value) * 1000.0:.2f}"
    except (TypeError, ValueError):
        return "-"


def print_replay_benchmark_report(report: ReplayPrefixBenchmarkReport) -> None:
    print(f"format: {report.format_id}")
    print(f"games: {report.games}")
    print(f"prefixes_per_game: {report.prefixes_per_game}")
    print(f"max_decision_rounds: {report.max_decision_rounds}")
    print(f"source_policy_ids: {dict(report.source_policy_ids)}")
    print(f"source_average_decision_rounds: {report.average_source_decision_rounds:.2f}")
    print(f"total_prefixes: {report.total_prefixes}")
    print(f"replay_elapsed_seconds: {report.total_replay_elapsed_seconds:.3f}")
    print(f"avg_replayed_decision_rounds: {report.average_replayed_decision_rounds:.2f}")
    print(f"replayed_decision_rounds_per_second: {report.replayed_decision_rounds_per_second:.1f}")
    print(f"avg_replay_ms: {report.average_replay_seconds * 1000.0:.2f}")
    print(f"median_replay_ms: {report.median_replay_seconds * 1000.0:.2f}")
    print(f"p95_replay_ms: {report.p95_replay_seconds * 1000.0:.2f}")
    print(f"max_replay_ms: {report.max_replay_seconds * 1000.0:.2f}")
    print("note: replay timing excludes source-game generation and includes env.reset plus prefix replay.")


if __name__ == "__main__":
    raise SystemExit(main())
