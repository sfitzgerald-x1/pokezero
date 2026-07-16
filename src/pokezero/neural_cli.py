"""Command-line utilities for optional neural policy experiments."""

from __future__ import annotations

import argparse
import copy
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time
from typing import Any, Callable, Iterable, Mapping, Sequence

from .cli_audit import (
    add_post_iteration_audit_arguments,
    post_iteration_audit_config_from_args,
    validate_post_iteration_audit_evaluation_games,
)
from .collection import (
    cache_observation_schemas_by_path,
    cache_feature_masks_by_path,
    cache_shaping_configs_by_path,
    BenchmarkMatchup,
    benchmark_rollouts,
    distinct_belief_set_source_hashes,
    neural_checkpoint_paths_from_policy_specs,
    policy_from_spec,
    policy_spec_with_showdown_root,
    reject_eval_only_specs,
)
from .shaping import ShapingConfig, parse_shaping_spec
from .dataset import (
    MAX_ACTIVE_TRAINING_CACHE_GB,
    TrajectoryDatasetConfig,
    delete_training_cache_path,
    is_training_cache_path,
    training_cache_paths_byte_size,
    training_cache_root_byte_size,
    write_training_cache_from_rollouts,
)
from .determinization import (
    DEFAULT_BELIEF_WORLD_SAMPLE_CAP,
    gen3_randbat_belief_start_override_planner,
)
from .prior_belief_profile import (
    MINIMUM_PROFILE_DECISIONS,
    PriorBeliefProfileConfig,
    merge_public_corpus_profile_shards,
    profile_public_corpus,
    profile_public_corpus_shard,
)
from .public_decision_corpus import open_public_decision_corpus, sha256_file
from .public_prefix_evaluator import PublicPrefixCandidateValueEvaluator
from .local_showdown import LocalShowdownConfig, LocalShowdownEnv, env_config_with_checkpoint_masks
from .observation import (
    OBSERVATION_SCHEMA_VERSION,
    OBSERVATION_SCHEMA_VERSION_V2_2,
    TRANSITION_TOKEN_COUNT,
)
from .showdown import observation_schema_version_from_choice, observation_spec_for_schema
from .neural_policy import (
    CONSTANT_LEARNING_RATE_SCHEDULE,
    DEFAULT_CATEGORY_OOV_BUCKETS,
    LEARNING_RATE_SCHEDULES,
    MIT_THESIS_LEARNING_RATE_SCHEDULE,
    TransformerPolicyConfig,
    TransformerSoftmaxPolicy,
    TransformerTrainingConfig,
    TransformerTrainingResult,
    checkpoint_file_sha256,
    collect_categorical_ids,
    distributed_training_context,
    evaluate_transformer_action_priors,
    evaluate_transformer_observation_value,
    evaluate_transformer_opponent_action_priors,
    feature_masks_from_model_config,
    initialize_distributed_training,
    load_transformer_checkpoint,
    load_transformer_model_config,
    load_transformer_policy,
    observation_spec_from_model_config,
    transformer_model_configs_from_policies,
    require_torch,
    require_compatible_transformer_value_checkpoint,
    resolve_torch_device,
    save_transformer_checkpoint,
    torch_available,
    train_transformer_policy,
)
from .search_benchmark import (
    RootPUCTCounterfactualBenchmarkReport,
    RootPUCTSearchBenchmarkReport,
    benchmark_root_puct_counterfactual_rollouts,
    benchmark_root_puct_search,
)
from .search_policy import (
    EntropyMarginVisitBudgetSelector,
    FixedExtraVisitBudgetSelector,
    RootPUCTSearchPolicy,
    greedy_opponent_action_planner,
    policy_opponent_action_planner,
    prior_top_k_opponent_action_scenario_planner,
)
from .root_puct_telemetry import root_puct_benchmark_telemetry_report
from .randbat import load_gen3_randbat_source_cached
from .randbat_vocab import gen3_category_vocabulary
from .value_calibration import (
    VALUE_SELECTION_METRICS,
    ValueCalibrationReport,
    evaluate_value_calibration,
    fit_value_calibration_transform,
    value_selection_metric_direction,
    value_selection_metric_value,
    value_selection_score,
)
from .neural_selfplay import (
    COLLECTOR_ADVANCEMENT_MODES,
    DEFAULT_COLLECTION_EXPLORATION_EPSILON,
    NeuralSelfPlayPromotionConfig,
    NeuralValueCalibrationConfig,
    NeuralValueSelectionConfig,
    _mapping,
    _policy_from_spec_for_evaluation,
    _sequence,
    load_neural_selfplay_run_manifest,
    run_neural_selfplay_iterations,
)
from .opponents import (
    CHECKPOINT_POLICY_SPEC_PREFIXES,
    HISTORICAL_OPPONENT_SELECTION_MODES,
    current_family_checkpoint_policy_specs,
    require_current_family_checkpoint_paths,
)
from .policy import Policy, PolicyContext, PolicyDecision, RandomLegalPolicy, SimpleLegalPolicy
from .run_audit import RunAuditFailure
from .rollout import RolloutConfig
from .rollout_cli import print_benchmark_report
from .eval_cli import _add_gate_arguments, _gate_config_from_args
from .refutation_training import REFUTATION_TRAINING_COMPATIBLE_OBJECTIVES
from .source_metadata import collect_source_metadata


MIN_NEURAL_POST_ITERATION_BENCHMARK_MATCHUPS = 4
FOUNDATION_MILESTONE_BENCHMARK_GAMES = 300
NEURAL_ITERATE_EXPERIMENT_PRESETS = ("none", "foundation-arms-race", "recipe-fidelity")
FOUNDATION_ARMS_RACE_PRESET_DEFAULTS: Mapping[str, Any] = {
    "objective": "ppo",
    "mirror_match": True,
    "collector_advancement_mode": "always",
    "collection_temperature": 1.4,
    "historical_opponent_selection": "spread",
    "evaluation_games": 200,
    "value_calibration": True,
    "value_selection": True,
    "value_selection_metric": "pearson_correlation",
    "value_selection_heldout_games": 32,
    "entropy_coef": 0.01,
    "ppo_target_mode": "gae",
}
# MIT thesis PPO hyperparameter table (Table A.3, p.43) — the reference recipe knobs that our
# config can express directly. learning_rate is the thesis base 10^-4.23; the schedule below
# applies the thesis global-progress annealing curve.
MIT_THESIS_REFERENCE_CONFIG: Mapping[str, float | int] = {
    "entropy_coef": 0.0588,
    "epochs": 7,
    "discount": 0.9999,
    "gae_lambda": 0.754,
    "clip_epsilon": 0.0829,
    "value_clip_range": 0.0184,
    "value_loss_weight": 0.4375,
    "max_grad_norm": 0.5430,
    "learning_rate": 5.9e-5,
    "batch_size": 1024,
}
MIT_THESIS_REFERENCE_LEARNING_RATE_SCHEDULE = MIT_THESIS_LEARNING_RATE_SCHEDULE
MIT_THESIS_REFERENCE_TRAINING_GAMES = 3_000_000
# The thesis ran standard temperature-1.0 sampling for self-play collection.
MIT_THESIS_REFERENCE_COLLECTION_TEMPERATURE = 1.0
# Knobs the thesis used that our per-iteration training loop cannot yet express faithfully. Keep
# this structure in the plan/audit payload so future gaps are surfaced explicitly.
RECIPE_FIDELITY_UNSUPPORTED_KNOBS: Mapping[str, str] = {}
RECIPE_FIDELITY_PRESET_DEFAULTS: Mapping[str, Any] = {
    # Loop shape: same arms-race self-play scaffolding (PPO + GAE, mirror self-play, latest-policy
    # collector, held-out Pearson value selection, calibration, max-damage yardstick) ...
    "objective": "ppo",
    "mirror_match": True,
    "collector_advancement_mode": "always",
    # ... but thesis-faithful collection temperature (standard sampling) rather than 1.4.
    "collection_temperature": MIT_THESIS_REFERENCE_COLLECTION_TEMPERATURE,
    "historical_opponent_selection": "spread",
    "evaluation_games": 200,
    "value_calibration": True,
    "value_selection": True,
    "value_selection_metric": "pearson_correlation",
    "value_selection_heldout_games": 32,
    "ppo_target_mode": "gae",
    # Thesis PPO hyperparameter table (the first-order recipe-fidelity knobs).
    "entropy_coef": MIT_THESIS_REFERENCE_CONFIG["entropy_coef"],
    "epochs": MIT_THESIS_REFERENCE_CONFIG["epochs"],
    "discount": MIT_THESIS_REFERENCE_CONFIG["discount"],
    "gae_lambda": MIT_THESIS_REFERENCE_CONFIG["gae_lambda"],
    "clip_epsilon": MIT_THESIS_REFERENCE_CONFIG["clip_epsilon"],
    "value_clip_range": MIT_THESIS_REFERENCE_CONFIG["value_clip_range"],
    "value_loss_weight": MIT_THESIS_REFERENCE_CONFIG["value_loss_weight"],
    "max_grad_norm": MIT_THESIS_REFERENCE_CONFIG["max_grad_norm"],
    "learning_rate": MIT_THESIS_REFERENCE_CONFIG["learning_rate"],
    "learning_rate_schedule": MIT_THESIS_REFERENCE_LEARNING_RATE_SCHEDULE,
    "learning_rate_schedule_total_games": MIT_THESIS_REFERENCE_TRAINING_GAMES,
    "batch_size": MIT_THESIS_REFERENCE_CONFIG["batch_size"],
}


def recipe_fidelity_reference_config() -> dict[str, Any]:
    payload: dict[str, Any] = dict(MIT_THESIS_REFERENCE_CONFIG)
    payload["learning_rate_schedule"] = MIT_THESIS_REFERENCE_LEARNING_RATE_SCHEDULE
    payload["learning_rate_schedule_total_games"] = MIT_THESIS_REFERENCE_TRAINING_GAMES
    return payload


_ITERATE_EXPERIMENT_PRESET_DEFAULTS: Mapping[str, Mapping[str, Any]] = {
    "foundation-arms-race": FOUNDATION_ARMS_RACE_PRESET_DEFAULTS,
    "recipe-fidelity": RECIPE_FIDELITY_PRESET_DEFAULTS,
}
# Loop-shape knobs applied regardless of objective; PPO hyperparameters applied only for objective=ppo.
_ITERATE_PRESET_LOOP_SHAPE_KEYS = (
    "objective",
    "mirror_match",
    "collector_advancement_mode",
    "collection_temperature",
    "historical_opponent_selection",
    "evaluation_games",
    "value_calibration",
    "value_selection",
    "value_selection_metric",
    "value_selection_heldout_games",
)
_ITERATE_PRESET_PPO_KEYS = (
    "entropy_coef",
    "ppo_target_mode",
    "epochs",
    "discount",
    "gae_lambda",
    "clip_epsilon",
    "value_clip_range",
    "value_loss_weight",
    "max_grad_norm",
    "learning_rate",
    "learning_rate_schedule",
    "learning_rate_schedule_total_games",
    "batch_size",
)
NEURAL_FOUNDATION_PLAN_SCHEMA_VERSION = "pokezero.neural_foundation_plan.v1"
NEURAL_FOUNDATION_RUN_SUMMARY_SCHEMA_VERSION = "pokezero.neural_foundation_run_summary.v1"
NEURAL_FOUNDATION_COMPARE_SCHEMA_VERSION = "pokezero.neural_foundation_compare.v1"
NEURAL_FOUNDATION_VALUE_TUNE_PLAN_SCHEMA_VERSION = "pokezero.neural_foundation_value_tune_plan.v1"
NEURAL_FOUNDATION_VALUE_TUNE_SUMMARY_SCHEMA_VERSION = "pokezero.neural_foundation_value_tune_summary.v1"
NEURAL_TRAIN_SUMMARY_SCHEMA_VERSION = "pokezero.neural_train_summary.v1"
FOUNDATION_COMPARE_CANDIDATE_SOURCES = ("latest", "latest-accepted", "best-max-damage")
FOUNDATION_TEACHER_CUT_ALLOWED_INITIAL_POLICY_NAMES = frozenset({"random-legal"})
FOUNDATION_TEACHER_CUT_LEARNED_INITIAL_PREFIXES = ("linear:", "neural:")
NEURAL_FOUNDATION_PROFILES: Mapping[str, Mapping[str, int | None]] = {
    "smoke": {
        "iterations": 2,
        "games_per_iteration": 8,
        "workers": 2,
        "evaluation_games": 8,
        "evaluation_interval_games": None,
        "epochs": 1,
        "max_batches": 2,
        "value_selection_heldout_games": 4,
    },
    "pilot": {
        "iterations": 3,
        "games_per_iteration": 256,
        "workers": 16,
        "evaluation_games": int(FOUNDATION_ARMS_RACE_PRESET_DEFAULTS["evaluation_games"]),
        "evaluation_interval_games": None,
        "epochs": 1,
        "max_batches": None,
        "value_selection_heldout_games": int(FOUNDATION_ARMS_RACE_PRESET_DEFAULTS["value_selection_heldout_games"]),
    },
    "midscale": {
        "iterations": 32,
        "games_per_iteration": 1_600,
        "workers": 128,
        "evaluation_games": FOUNDATION_MILESTONE_BENCHMARK_GAMES,
        "evaluation_interval_games": 10_000,
        "epochs": 1,
        "max_batches": None,
        "value_selection_heldout_games": int(FOUNDATION_ARMS_RACE_PRESET_DEFAULTS["value_selection_heldout_games"]),
    },
}
NEURAL_FOUNDATION_VARIANTS: Mapping[str, Mapping[str, Any]] = {
    "baseline": {
        "description": "Use the foundation-arms-race preset without wrapper-level auxiliary-loss changes.",
        "opponent_action_loss_weight": None,
        "temporal_aggregator": None,
        "opponent_policies": None,
        "teacher_cut": False,
    },
    "teacher-cut": {
        "description": (
            "Run the clean teacher-cut WS-A experiment: PPO self-play with no fixed "
            "teacher/heuristic training opponents after any one-shot initial policy."
        ),
        "opponent_action_loss_weight": None,
        "temporal_aggregator": None,
        "opponent_policies": (),
        "teacher_cut": True,
    },
    "opponent-signal": {
        "description": "Increase opponent-action auxiliary supervision for an H3 foundation ablation.",
        "opponent_action_loss_weight": 1.0,
        "temporal_aggregator": None,
        "opponent_policies": None,
        "teacher_cut": False,
    },
    "temporal-gru": {
        "description": "Use the GRU temporal aggregator as a WS-E/WS-A value/base-net ablation.",
        "opponent_action_loss_weight": None,
        "temporal_aggregator": "gru",
        "opponent_policies": None,
        "teacher_cut": False,
    },
    "opponent-signal-gru": {
        "description": "Combine opponent-action auxiliary supervision with the GRU temporal aggregator.",
        "opponent_action_loss_weight": 1.0,
        "temporal_aggregator": "gru",
        "opponent_policies": None,
        "teacher_cut": False,
    },
    "anti-aggression": {
        "description": "Add aggressive-damage to the fixed opponent pool for targeted counterplay pressure.",
        "opponent_action_loss_weight": None,
        "temporal_aggregator": None,
        "opponent_policies": ("random-legal", "simple-legal", "aggressive-damage"),
        "teacher_cut": False,
    },
    "anti-aggression-gru": {
        "description": "Combine aggressive-damage fixed-opponent pressure with the GRU temporal aggregator.",
        "opponent_action_loss_weight": None,
        "temporal_aggregator": "gru",
        "opponent_policies": ("random-legal", "simple-legal", "aggressive-damage"),
        "teacher_cut": False,
    },
}
_DEFAULT_BENCHMARK_YARDSTICK_POLICY_IDS = frozenset({"max-damage"})
_PLUMBING_BENCHMARK_POLICY_IDS = frozenset({"random-legal", "simple-legal"})
_NAMED_REPORT_POLICY_IDS = frozenset(
    {
        "random-legal",
        "simple-legal",
        "scripted-teacher",
        "max-damage",
        "aggressive-damage",
    }
)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m pokezero.neural_cli")
    subparsers = parser.add_subparsers(dest="command", required=True)

    describe = subparsers.add_parser("describe", help="Print the default neural policy config and torch availability.")
    describe.add_argument("--json", action="store_true", help="Print machine-readable JSON.")
    describe.set_defaults(func=_describe)

    serve = subparsers.add_parser("serve", help="Serve a checkpoint's forward to remote collectors (WS-L1 GPU inference service).")
    serve.add_argument("--checkpoint", required=True, help="Path to the transformer checkpoint to serve.")
    serve.add_argument("--host", default="0.0.0.0", help="Bind host (default 0.0.0.0).")
    serve.add_argument("--port", type=int, default=8600, help="Bind port (default 8600).")
    serve.add_argument("--device", default=None, help="Torch device (default: cuda if available, else cpu).")
    serve.add_argument("--amp", choices=["bf16"], default=None, help="Run the served forward in bf16 autocast (GB200 tensor cores).")
    serve.add_argument("--max-batch", type=int, default=64, help="Max requests coalesced into one batched forward (default 64).")
    serve.add_argument("--batch-window-ms", type=float, default=10.0, help="Dynamic-batching wait window in ms (default 10).")
    serve.set_defaults(func=_serve)

    train = subparsers.add_parser("train", help="Train an entity-token transformer policy from rollout JSONL or training caches.")
    train.add_argument("--data", type=Path, nargs="+", required=True, help="One or more rollout JSONL files or training cache directories.")
    train.add_argument(
        "--refutation-cache",
        type=Path,
        nargs="+",
        default=None,
        help=(
            "Optional certified G4 refutation training-cache directories to mix into "
            "training as capped auxiliary examples. These caches are never deleted by "
            "--delete-cache-after-read."
        ),
    )
    train.add_argument(
        "--refutation-max-fraction",
        type=float,
        default=0.1,
        help=(
            "Maximum refutation examples as a fraction of emitted training examples. "
            "When --refutation-cache is used, this must be >0 and <=0.2."
        ),
    )
    train.add_argument(
        "--refutation-target-mode",
        choices=("policy-value", "policy-distribution-value", "value"),
        default="policy-value",
        help=(
            "Target mode used to build the refutation cache. value mode is accepted "
            "only with objective=ppo or objective=value-only; policy-distribution-value "
            "requires caches built from search_policy_distribution rows."
        ),
    )
    train.add_argument("--out", type=Path, required=True, help="Checkpoint output path.")
    train.add_argument(
        "--summary-out",
        type=Path,
        default=None,
        help="Optional JSON summary path for train timing, checkpoint size, metrics, and cache lifecycle evidence.",
    )
    train.add_argument(
        "--max-cache-gb",
        type=float,
        default=MAX_ACTIVE_TRAINING_CACHE_GB,
        help=(
            f"Reject training-cache inputs whose active cache root exceeds this many GiB "
            f"(default and maximum: {MAX_ACTIVE_TRAINING_CACHE_GB:g})."
        ),
    )
    cache_delete_group = train.add_mutually_exclusive_group()
    cache_delete_group.add_argument(
        "--delete-cache-after-read",
        dest="delete_cache_after_read",
        action="store_true",
        default=True,
        help=(
            "Delete each consumed training cache directory after the checkpoint is safely written. "
            "This is the default for training-cache inputs."
        ),
    )
    cache_delete_group.add_argument(
        "--keep-cache-after-read",
        dest="delete_cache_after_read",
        action="store_false",
        help="Keep consumed training cache directories after training for debugging or audit runs.",
    )
    train.add_argument(
        "--initial-checkpoint",
        type=Path,
        default=None,
        help="Optional checkpoint to warm-start from. Uses that checkpoint's model config; --policy-id can relabel the output.",
    )
    # Ablation-arm feature masks (config, not spec). Fresh train: the flags SET the model
    # config's masks (which downstream harnesses latch back into env encode masks). With
    # --initial-checkpoint the masks come from the checkpoint; explicitly-passed flags
    # must AGREE with it or the command hard-fails (never silently retrain under
    # different observation content — the #492 mismatch class).
    train.add_argument(
        "--observation-schema",
        choices=("v2.1", "v2.2"),
        default=None,
        help=(
            "Observation schema for a FRESH train: v2.1 or v2.2 (default; turn-merged "
            "transition tokens; stamps the model config, sizes the widths, and flips the "
            "schema-derived vocabulary). With --initial-checkpoint the checkpoint's stamped "
            "schema wins and an explicitly disagreeing flag hard-fails (mask-conflict "
            "semantics)."
        ),
    )
    train.add_argument(
        "--transition-token-budget",
        type=int,
        default=None,
        help="Most-recent transition-token slots filled at encode time. UNIT IS SCHEMA-DEPENDENT: under v2/v2.1 a token is one declared ACTION (32 = the K=16-turn ablation arm); under --observation-schema v2.2 a token is a whole TURN, so budget 32 covers roughly what 64 action-tokens did.",
    )
    train.add_argument(
        "--no-stats-block",
        action="store_true",
        help="Ablation arm: zero + attention-mask the stats token and per-mon tendency triple.",
    )
    train.add_argument(
        "--no-exact-state",
        action="store_true",
        help="Ablation arm: zero the exact-state layer (PP fractions, counters, expected stats).",
    )
    train.add_argument(
        "--tier2-residuals",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Tier-2 residual channel (#505). Default for a fresh train: on.",
    )
    train.add_argument(
        "--tier2-investment",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Defender-side investment channel (#513): populate + encode the reserved "
            "investment columns (NUMERIC_TT_INVESTMENT_BIT 120, "
            "NUMERIC_TIER2_INVESTMENT_PINNED 139) behind the investment precision gate. "
            "A SEPARATE switch from --tier2-residuals (different provenance). Default for a "
            "fresh train: OFF (byte-identical to the pre-investment encoder). Only meaningful "
            "under --observation-schema v2.1/v2.2; a no-op under v2. With --initial-checkpoint "
            "the checkpoint's value wins and an explicitly disagreeing flag hard-fails."
        ),
    )
    train.add_argument("--epochs", type=int, default=1, help="Number of training epochs.")
    train.add_argument("--batch-size", type=int, default=64, help="Training batch size.")
    train.add_argument("--learning-rate", type=float, default=3e-4, help="AdamW learning rate.")
    train.add_argument(
        "--learning-rate-schedule",
        choices=LEARNING_RATE_SCHEDULES,
        default=CONSTANT_LEARNING_RATE_SCHEDULE,
        help="Learning-rate schedule. 'mit-thesis' applies base_lr/(8x+1)^1.5 over the supplied progress window.",
    )
    train.add_argument(
        "--learning-rate-schedule-total-games",
        type=int,
        default=None,
        help="Optional total-game denominator used by schedule-aware self-play configs. Standalone train uses progress flags directly.",
    )
    train.add_argument(
        "--learning-rate-progress-start",
        type=float,
        default=0.0,
        help="Global training progress at the start of this standalone train call, in [0, 1].",
    )
    train.add_argument(
        "--learning-rate-progress-end",
        type=float,
        default=0.0,
        help="Global training progress at the end of this standalone train call, in [0, 1].",
    )
    train.add_argument(
        "--learning-rate-warmup-progress",
        type=float,
        default=0.0,
        help="Linear LR warmup fraction of global progress, in [0, 1]. Below this progress the LR ramps "
        "0 -> the scheduled value; 0 disables warmup (default). Stabilizes cold-start at 50M+ scale. "
        "Compares against absolute global progress, so on a resumed run whose progress already starts "
        "past this fraction it never fires.",
    )
    train.add_argument("--weight-decay", type=float, default=0.0, help="AdamW weight decay.")
    train.add_argument("--window-size", type=int, default=1, help="Per-player observation history window (spec v2 default: 1).")
    train.add_argument("--discount", type=float, default=1.0, help="Terminal return discount per player decision.")
    train.add_argument("--capped-terminal-value", type=float, default=0.0, help="Return assigned to each player in capped games.")
    train.add_argument(
        "--hp-delta-return-weight",
        type=float,
        default=0.0,
        help="Optional return-shaping weight for visible player-relative HP differential changes.",
    )
    train.add_argument(
        "--faint-delta-return-weight",
        type=float,
        default=0.0,
        help="Optional return-shaping weight for visible player-relative faint differential changes.",
    )
    train.add_argument(
        "--turn-penalty-after",
        type=int,
        default=None,
        help="Optional turn index at which to start applying a per-decision shaped return penalty.",
    )
    train.add_argument(
        "--turn-penalty",
        type=float,
        default=0.0,
        help="Optional positive per-decision return penalty applied at or after --turn-penalty-after.",
    )
    train.add_argument(
        "--shaping-weights",
        default=None,
        help=(
            "Dense potential-based reward shaping for returns/GAE targets: preset (wse-arm1), "
            "inline JSON, @/path/to.json, or 'none' for explicit-off. Absent: unshaped, or the "
            "initial checkpoint's stamped shaping on resume. Cross-checked against training-cache "
            "provenance (both directions) and stamped into the saved model config."
        ),
    )
    train.add_argument("--value-loss-weight", type=float, default=0.25, help="Scalar value-head MSE loss weight.")
    train.add_argument(
        "--value-clip-range",
        type=float,
        default=None,
        help="Optional PPO value-function update clip range, using recorded rollout values as V_old.",
    )
    train.add_argument(
        "--value-ranking-loss-weight",
        type=float,
        default=0.0,
        help="Optional pairwise value-ranking loss weight. Optimizes leaf ordering for search when positive.",
    )
    train.add_argument(
        "--value-ranking-margin",
        type=float,
        default=0.0,
        help="Non-negative margin for --value-ranking-loss-weight pairwise value ordering.",
    )
    train.add_argument("--opponent-action-loss-weight", type=float, default=0.1, help="Opponent-action auxiliary loss weight.")
    train.add_argument(
        "--switch-action-loss-weight",
        type=float,
        default=1.0,
        help="Multiplier for switch-action policy CE examples under behavior-cloning / reward-weighted objectives.",
    )
    train.add_argument(
        "--action-family-loss-weight",
        type=float,
        default=0.0,
        help="Auxiliary move-vs-switch classification loss weight derived from legal action logits.",
    )
    train.add_argument(
        "--switch-target-loss-weight",
        type=float,
        default=0.0,
        help="Auxiliary conditional switch-target classification loss weight over switch-labeled examples.",
    )
    train.add_argument(
        "--objective",
        choices=("behavior-cloning", "reward-weighted", "ppo", "value-only"),
        default="behavior-cloning",
        help=(
            "Training objective: supervised behavior cloning (default), reward-weighted "
            "behavior cloning, PPO self-play RL, or value-only return prediction."
        ),
    )
    train.add_argument("--clip-epsilon", type=float, default=0.2, help="PPO clipped-surrogate epsilon (objective=ppo).")
    train.add_argument("--entropy-coef", type=float, default=0.0, help="PPO entropy bonus coefficient (objective=ppo).")
    train.add_argument("--no-normalize-advantage", action="store_true", help="Disable PPO advantage normalization (objective=ppo).")
    train.add_argument(
        "--ppo-target-mode",
        choices=("returns", "gae"),
        default="returns",
        help="PPO advantage/value-target source: discounted returns or recorded-value GAE.",
    )
    train.add_argument("--gae-lambda", type=float, default=0.95, help="GAE lambda when --ppo-target-mode=gae.")
    train.add_argument(
        "--max-grad-norm",
        type=float,
        default=None,
        help="Optional global gradient-norm clip applied before each optimizer step (thesis recipe: 0.5430).",
    )
    train.add_argument(
        "--amp",
        choices=["bf16"],
        default=None,
        help="Mixed-precision autocast for forward/loss (WS-A1). 'bf16' keeps fp32 master weights/grads (no GradScaler). Default fp32.",
    )
    train.add_argument(
        "--training-seed",
        type=int,
        default=0,
        help="Base torch seed. Under torchrun, rank r uses base_seed + r for its dropout stream.",
    )
    train.add_argument(
        "--freeze-non-value-parameters",
        action="store_true",
        help="Train only value-head parameters; intended for value-only calibration fine-tunes from --initial-checkpoint.",
    )
    train.add_argument("--max-batches", type=int, default=None, help="Optional max batches per epoch for smoke runs.")
    train.add_argument("--device", default=None, help="Torch device, e.g. cpu, cuda, or mps. Defaults to cuda when available, else cpu.")
    train.add_argument("--embedding-dim", type=int, default=128, help="Transformer embedding width.")
    train.add_argument("--layers", type=int, default=2, help="Transformer encoder layer count. Use 0 for the CPU-fast pooled encoder.")
    train.add_argument("--attention-heads", type=int, default=4, help="Transformer attention head count.")
    train.add_argument("--feedforward-dim", type=int, default=256, help="Transformer feedforward width.")
    train.add_argument("--dropout", type=float, default=0.1, help="Transformer dropout.")
    train.add_argument(
        "--temporal-aggregator",
        choices=("mean", "gru"),
        default="mean",
        help="How to combine encoded observation history for value/opponent heads.",
    )
    train.add_argument("--policy-id", default=None, help="Policy id stored in the checkpoint config.")
    train.add_argument(
        "--category-oov-buckets",
        type=int,
        default=DEFAULT_CATEGORY_OOV_BUCKETS,
        help="Reserved out-of-vocabulary rows in the compact category embedding.",
    )
    train.add_argument(
        "--showdown-root",
        type=Path,
        default=None,
        help="Built Pokemon Showdown checkout root (required: the category vocabulary is the closed Gen 3 randbat universe).",
    )
    train.add_argument(
        "--value-calibration-data",
        type=Path,
        nargs="+",
        default=None,
        help="Optional rollout JSONL path(s) used to write a post-train value calibration artifact.",
    )
    train.add_argument(
        "--value-calibration-out",
        type=Path,
        default=None,
        help="Optional JSON output path for --value-calibration-data. Defaults to printing the report.",
    )
    train.add_argument("--value-calibration-batch-size", type=int, default=128, help="Post-train calibration batch size.")
    train.add_argument("--value-calibration-bins", type=int, default=10, help="Post-train calibration bin count.")
    train.add_argument(
        "--value-selection-data",
        type=Path,
        nargs="+",
        default=None,
        help="Optional held-out rollout JSONL path(s) evaluated after each epoch to restore the best value-calibrated epoch.",
    )
    train.add_argument(
        "--value-selection-metric",
        choices=VALUE_SELECTION_METRICS,
        default="mae",
        help=(
            "Held-out value metric used by --value-selection-data; sign_accuracy and pearson_correlation "
            "are maximized, others are minimized. pearson_correlation measures affine-invariant "
            "linear association, not calibration by itself."
        ),
    )
    train.add_argument(
        "--value-selection-out",
        type=Path,
        default=None,
        help="Optional JSON output path for per-epoch --value-selection-data reports.",
    )
    train.set_defaults(func=_train)

    cache_data = subparsers.add_parser("cache-data", help="Convert rollout JSONL into a compact neural training cache.")
    cache_data.add_argument("--data", type=Path, nargs="+", required=True, help="One or more rollout JSONL files.")
    cache_data.add_argument("--out", type=Path, required=True, help="Training cache output directory.")
    cache_data.add_argument("--overwrite", action="store_true", help="Replace an existing training cache directory.")
    cache_data.add_argument(
        "--max-cache-gb",
        type=float,
        default=MAX_ACTIVE_TRAINING_CACHE_GB,
        help=(
            "Reject the write if existing caches under the output parent plus the new cache "
            f"would exceed this many GiB (default and maximum: {MAX_ACTIVE_TRAINING_CACHE_GB:g})."
        ),
    )
    _add_training_dataset_arguments(cache_data)
    cache_data.set_defaults(func=_cache_data)

    benchmark = subparsers.add_parser("benchmark", help="Benchmark a neural checkpoint against fixed baselines.")
    benchmark.add_argument("--checkpoint", type=Path, required=True, help="Neural checkpoint path.")
    benchmark.add_argument("--policy-id", default=None, help="Optional benchmark policy id alias for the checkpoint.")
    benchmark.add_argument("--games", type=int, default=20, help="Number of games per matchup.")
    benchmark.add_argument("--showdown-root", type=Path, default=None, help="Built Pokemon Showdown checkout root.")
    benchmark.add_argument("--format", dest="format_id", default="gen3randombattle", help="Showdown format id.")
    benchmark.add_argument("--seed-start", type=int, default=1, help="First deterministic rollout seed for every matchup.")
    benchmark.add_argument("--max-decision-rounds", type=int, default=250, help="Rollout decision-round cap.")
    benchmark.add_argument("--node-binary", default="node", help="Node executable used for the BattleStream bridge.")
    benchmark.add_argument("--device", default=None, help="Torch device, e.g. cpu, cuda, or mps. Defaults to checkpoint load behavior.")
    benchmark.add_argument("--sample", action="store_true", help="Sample from the checkpoint policy distribution instead of greedy selection.")
    benchmark.add_argument("--epsilon", type=float, default=0.0, help="Random legal exploration rate during benchmark.")
    benchmark.add_argument("--temperature", type=float, default=1.0, help="Softmax sampling temperature.")
    benchmark.add_argument(
        "--benchmark-reference-policy",
        action="append",
        default=None,
        help=(
            "Additional eval-only or fixed reference policy spec benchmarked against the checkpoint "
            "in both seats, e.g. max-damage. May be repeated."
        ),
    )
    benchmark.add_argument(
        "--benchmark-reference-policy-id",
        action="append",
        default=None,
        help=(
            "Optional policy id alias for each --benchmark-reference-policy. When supplied, "
            "the count must match the number of reference policies."
        ),
    )
    benchmark.add_argument(
        "--allow-legacy-checkpoints",
        action="store_true",
        help=(
            "Allow no-belief/pre-v2 checkpoint families in this benchmark. Use only for "
            "archived historical diagnostics; current strength evals require v2+ current-family checkpoints."
        ),
    )
    benchmark.add_argument("--json", action="store_true", help="Print benchmark results as JSON.")
    benchmark.add_argument(
        "--summary-out",
        type=Path,
        default=None,
        help="Optional JSON path where the benchmark report is persisted for later audit.",
    )
    benchmark.set_defaults(func=_benchmark)

    root_puct_play = subparsers.add_parser(
        "root-puct-play-benchmark",
        help="Benchmark raw checkpoint play against root-PUCT checkpoint play over full games.",
    )
    root_puct_play.add_argument("--checkpoint", type=Path, required=True, help="Neural checkpoint path.")
    root_puct_play.add_argument(
        "--value-checkpoint",
        type=Path,
        default=None,
        help=(
            "Optional compatible checkpoint used only for root-PUCT leaf values. "
            "Use a frozen calibrated copy while --checkpoint continues to provide raw policy priors."
        ),
    )
    root_puct_play.add_argument("--games", type=int, default=20, help="Number of games per matchup.")
    root_puct_play.add_argument("--showdown-root", type=Path, default=None, help="Built Pokemon Showdown checkout root.")
    root_puct_play.add_argument("--format", dest="format_id", default="gen3randombattle", help="Showdown format id.")
    root_puct_play.add_argument("--seed-start", type=int, default=1, help="First deterministic rollout seed for every matchup.")
    root_puct_play.add_argument("--max-decision-rounds", type=int, default=250, help="Rollout decision-round cap.")
    root_puct_play.add_argument("--node-binary", default="node", help="Node executable used for the BattleStream bridge.")
    root_puct_play.add_argument(
        "--opponent-policy",
        action="append",
        default=None,
        help="Fixed opponent policy spec. May be repeated. Defaults to random-legal and simple-legal.",
    )
    root_puct_play.add_argument("--cpuct", type=float, default=1.25, help="PUCT exploration constant.")
    root_puct_play.add_argument(
        "--leaf-rollout-rounds",
        type=int,
        default=0,
        help=(
            "Optional bounded simulator continuation per root candidate before leaf value "
            "evaluation. Zero keeps the default one-ply value-head leaf."
        ),
    )
    root_puct_play.add_argument(
        "--leaf-rollout-rounds-sweep",
        type=int,
        action="append",
        default=None,
        help=(
            "Repeatable leaf-depth sweep value. When supplied, root-puct-play-benchmark "
            "creates one root-PUCT policy variant per unique supplied depth on the same seed "
            "range and ignores --leaf-rollout-rounds."
        ),
    )
    root_puct_play.add_argument(
        "--leaf-rollout-opponent-policy",
        choices=("checkpoint", "benchmark"),
        default="checkpoint",
        help=(
            "Opponent policy used during bounded leaf rollouts. 'checkpoint' preserves the "
            "current checkpoint-vs-checkpoint continuation; 'benchmark' uses the fixed "
            "benchmark opponent for the non-search side. Root simultaneous opponent-action "
            "planning still uses checkpoint opponent-action priors."
        ),
    )
    root_puct_play.add_argument(
        "--root-opponent-action-policy",
        choices=("checkpoint", "benchmark"),
        default="checkpoint",
        help=(
            "Opponent action source used for the simultaneous root branch. 'checkpoint' uses "
            "the checkpoint opponent-action prior head; 'benchmark' asks a separate copy of "
            "the fixed benchmark opponent policy to choose the non-search side's root action "
            "from its private observation. Benchmark mode is privileged evaluation plumbing; "
            "stochastic opponents produce one sampled action, not a modal expectation."
        ),
    )
    root_puct_play.add_argument(
        "--root-opponent-action-scenarios",
        type=int,
        default=1,
        help=(
            "Number of checkpoint-prior opponent root-action scenarios to average per searched "
            "candidate. Values above one require --root-opponent-action-policy checkpoint; "
            "benchmark root-opponent mode intentionally produces one private-observation action."
        ),
    )
    root_puct_play.add_argument(
        "--root-opponent-action-candidate-scenarios",
        type=int,
        default=None,
        help=(
            "Number of checkpoint-prior opponent root-action candidates to try while searching "
            "for replay-legal scenarios. Defaults to --root-opponent-action-scenarios; values "
            "above that provide hidden-mode reserve candidates without averaging more accepted "
            "scenarios."
        ),
    )
    root_puct_play.add_argument(
        "--selection-mode",
        choices=("puct", "value", "visits"),
        default="visits",
        help=(
            "Root candidate selector for the search policy. Defaults to 'visits', which uses PUCT "
            "for root traversal and selects the most-visited action after the root visit budget is "
            "spent. 'value' selects the highest value-evaluated branch; 'puct' selects by final "
            "Q+U score and is diagnostic."
        ),
    )
    root_puct_play.set_defaults(root_visit_budget_explicit=False)
    root_puct_play.add_argument(
        "--root-visit-budget",
        type=int,
        default=16,
        action=_StoreExplicitArgument,
        help=(
            "Root visits per accepted opponent-action scenario; defaults to 16. "
            "With multiple accepted scenarios, total decision visits scale by searched scenario count."
        ),
    )
    root_puct_play.add_argument(
        "--root-extra-visits",
        type=int,
        default=None,
        help=(
            "Fixed visits added after the mandatory legal-action sweep. Mutually exclusive "
            "with adaptive root budgeting; use 0 for the sweep-only arm."
        ),
    )
    root_puct_play.add_argument(
        "--root-time-budget-ms",
        type=int,
        default=None,
        help=(
            "Optional per-decision wall-clock budget after the mandatory legal-action sweep. "
            "Mutually exclusive with fixed and adaptive post-sweep visit budgets."
        ),
    )
    root_puct_play.add_argument(
        "--adaptive-root-contested-extra-visits",
        type=int,
        default=None,
        help=(
            "Enable adaptive root budgeting and add this many visits beyond the mandatory legal-action "
            "sweep when the entropy or initial value-margin threshold marks a decision contested."
        ),
    )
    root_puct_play.add_argument(
        "--adaptive-root-uncontested-extra-visits",
        type=int,
        default=0,
        help="Additional visits beyond the mandatory sweep for non-contested decisions when adaptive budgeting is enabled.",
    )
    root_puct_play.add_argument(
        "--adaptive-root-policy-entropy-threshold",
        type=float,
        default=None,
        help="Mark a decision contested when normalized legal-action policy entropy is at least this value.",
    )
    root_puct_play.add_argument(
        "--adaptive-root-value-margin-threshold",
        type=float,
        default=None,
        help="Mark a decision contested when the mandatory-sweep top-two leaf-value margin is at most this value.",
    )
    root_puct_play.add_argument(
        "--min-value-improvement",
        type=float,
        default=None,
        help=(
            "Optional conservative gate for root-PUCT play: keep the raw-prior action unless "
            "the search-selected action beats it by at least this value margin."
        ),
    )
    root_puct_play.add_argument("--device", default=None, help="Torch device, e.g. cpu, cuda, or mps.")
    root_puct_play.add_argument("--temperature", type=float, default=1.0, help="Softmax temperature for raw policy, leaf policy, and opponent-action priors.")
    root_puct_play.add_argument(
        "--root-prior-temperature",
        type=float,
        default=None,
        help=(
            "Temperature applied only to root-PUCT action priors. Defaults to --temperature, "
            "so existing invocations keep their prior softness unless overridden."
        ),
    )
    root_puct_play.add_argument(
        "--root-dirichlet-alpha",
        type=float,
        default=None,
        help=(
            "Enable an audit-only root Dirichlet-noise arm with this positive concentration. "
            "Omit it to keep the deterministic primary evaluation priors."
        ),
    )
    root_puct_play.add_argument(
        "--root-dirichlet-mix",
        type=float,
        default=0.25,
        help="Fraction of legal root prior mass replaced by Dirichlet noise when enabled.",
    )
    root_puct_play.add_argument(
        "--root-dirichlet-seed",
        type=int,
        default=0,
        help="Base seed for reproducible per-decision root Dirichlet draws.",
    )
    root_puct_play.add_argument(
        "--belief-start-overrides",
        action="store_true",
        help=(
            "Sample player-relative public Gen 3 randbat belief into complete custom-game branch "
            "starts for root-PUCT search. Enables the candidate-set belief source in the benchmark env."
        ),
    )
    root_puct_play.add_argument(
        "--belief-world-sample-cap",
        type=int,
        default=DEFAULT_BELIEF_WORLD_SAMPLE_CAP,
        help=(
            "Maximum PIMC belief worlds per opponent-action scenario when --belief-start-overrides is set. "
            "The resolved count is min(cap, public surviving-variant combinations)."
        ),
    )
    root_puct_play.add_argument(
        "--belief-start-override-attempts",
        type=int,
        default=10,
        help="Replay-consistency attempts per belief-world sample.",
    )
    root_puct_play.add_argument(
        "--belief-start-override-hp-fraction-tolerance",
        type=float,
        default=0.02,
        help="Allowed HP-fraction drift when validating public belief-world replay prefixes.",
    )
    root_puct_play.add_argument(
        "--no-search-fallback",
        action="store_true",
        help="Disable fallback to the raw checkpoint action when root-PUCT branch search fails.",
    )
    root_puct_play.add_argument(
        "--allow-legacy-checkpoints",
        action="store_true",
        help=(
            "Allow no-belief/pre-v2 checkpoint families in this root-PUCT benchmark. Use only "
            "for archived historical diagnostics; current strength evals require v2+ current-family checkpoints."
        ),
    )
    root_puct_play.add_argument("--json", action="store_true", help="Print benchmark results as JSON.")
    root_puct_play.add_argument(
        "--summary-out",
        type=Path,
        default=None,
        help="Optional JSON path where the root-PUCT full-game benchmark report is persisted.",
    )
    root_puct_play.set_defaults(func=_root_puct_play_benchmark)

    root_puct_telemetry = subparsers.add_parser(
        "root-puct-telemetry-report",
        help="Summarize Root-PUCT fallback, scenario-legality, visit, and timing telemetry from a benchmark JSON artifact.",
    )
    root_puct_telemetry.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Telemetry-enabled root-puct-play-benchmark JSON artifact.",
    )
    root_puct_telemetry.add_argument(
        "--policy-id",
        action="append",
        default=None,
        help="Optional Root-PUCT policy id to include. May be repeated; defaults to every telemetry-bearing policy.",
    )
    root_puct_telemetry.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Optional JSON path where the compact telemetry report is persisted.",
    )
    root_puct_telemetry.add_argument("--json", action="store_true", help="Print the telemetry report as JSON.")
    root_puct_telemetry.set_defaults(func=_root_puct_telemetry_report)

    root_puct = subparsers.add_parser(
        "root-puct-benchmark",
        help="Evaluate root-PUCT checkpoint decisions on sampled rollout prefixes.",
    )
    root_puct.add_argument("--checkpoint", type=Path, required=True, help="Neural checkpoint path used for policy priors and value scores.")
    root_puct.add_argument("--games", type=int, default=3, help="Number of source games to generate.")
    root_puct.add_argument(
        "--prefixes-per-game",
        type=int,
        default=5,
        help="Evenly sampled source prefixes evaluated with root PUCT per game.",
    )
    root_puct.add_argument("--showdown-root", type=Path, default=None, help="Built Pokemon Showdown checkout root.")
    root_puct.add_argument("--format", dest="format_id", default="gen3randombattle", help="Showdown format id.")
    root_puct.add_argument("--seed-start", type=int, default=1, help="First deterministic source-game seed.")
    root_puct.add_argument("--max-decision-rounds", type=int, default=250, help="Source rollout decision-round cap.")
    root_puct.add_argument("--node-binary", default="node", help="Node executable used for the BattleStream bridge.")
    root_puct.add_argument("--p1-policy", default="random-legal", help="Source rollout policy for p1.")
    root_puct.add_argument("--p2-policy", default="random-legal", help="Source rollout policy for p2.")
    root_puct.add_argument("--search-player", choices=("p1", "p2"), default="p1", help="Player side whose recorded decisions are re-scored.")
    root_puct.add_argument("--cpuct", type=float, default=1.25, help="PUCT exploration constant.")
    root_puct.add_argument(
        "--value-checkpoint",
        type=Path,
        default=None,
        help="Optional checkpoint used only for leaf values; defaults to --checkpoint for policy priors and values.",
    )
    root_puct.add_argument(
        "--root-extra-visits",
        type=int,
        default=None,
        help="Fixed visits added after the mandatory one-visit-per-legal-action sweep.",
    )
    root_puct.add_argument("--device", default=None, help="Torch device, e.g. cpu, cuda, or mps.")
    root_puct.add_argument("--temperature", type=float, default=1.0, help="Policy-prior softmax temperature.")
    root_puct.add_argument(
        "--allow-legacy-checkpoints",
        action="store_true",
        help=(
            "Allow no-belief/pre-v2 checkpoint families in this root-PUCT benchmark. Use only "
            "for archived historical diagnostics; current strength evals require v2+ current-family checkpoints."
        ),
    )
    root_puct.add_argument("--json", action="store_true", help="Print search benchmark results as JSON.")
    root_puct.set_defaults(func=_root_puct_benchmark)

    root_puct_counterfactual = subparsers.add_parser(
        "root-puct-counterfactual",
        help="Compare recorded vs root-PUCT-selected branch rollout outcomes on sampled prefixes.",
    )
    root_puct_counterfactual.add_argument("--checkpoint", type=Path, required=True, help="Neural checkpoint path used for policy priors and value scores.")
    root_puct_counterfactual.add_argument("--games", type=int, default=3, help="Number of source games to generate.")
    root_puct_counterfactual.add_argument(
        "--prefixes-per-game",
        type=int,
        default=5,
        help="Evenly sampled source prefixes evaluated with root PUCT per game.",
    )
    root_puct_counterfactual.add_argument("--showdown-root", type=Path, default=None, help="Built Pokemon Showdown checkout root.")
    root_puct_counterfactual.add_argument("--format", dest="format_id", default="gen3randombattle", help="Showdown format id.")
    root_puct_counterfactual.add_argument("--seed-start", type=int, default=1, help="First deterministic source-game seed.")
    root_puct_counterfactual.add_argument("--max-decision-rounds", type=int, default=250, help="Source rollout decision-round cap.")
    root_puct_counterfactual.add_argument("--node-binary", default="node", help="Node executable used for the BattleStream bridge.")
    root_puct_counterfactual.add_argument("--p1-policy", default="random-legal", help="Source rollout policy for p1.")
    root_puct_counterfactual.add_argument("--p2-policy", default="random-legal", help="Source rollout policy for p2.")
    root_puct_counterfactual.add_argument(
        "--continuation-p1-policy",
        default=None,
        help="Branch rollout continuation policy for p1. Defaults to --p1-policy.",
    )
    root_puct_counterfactual.add_argument(
        "--continuation-p2-policy",
        default=None,
        help="Branch rollout continuation policy for p2. Defaults to --p2-policy.",
    )
    root_puct_counterfactual.add_argument("--search-player", choices=("p1", "p2"), default="p1", help="Player side whose recorded decisions are re-scored.")
    root_puct_counterfactual.add_argument("--cpuct", type=float, default=1.25, help="PUCT exploration constant.")
    root_puct_counterfactual.add_argument("--device", default=None, help="Torch device, e.g. cpu, cuda, or mps.")
    root_puct_counterfactual.add_argument("--temperature", type=float, default=1.0, help="Policy-prior softmax temperature.")
    root_puct_counterfactual.add_argument(
        "--allow-legacy-checkpoints",
        action="store_true",
        help=(
            "Allow no-belief/pre-v2 checkpoint families in this root-PUCT counterfactual. Use only "
            "for archived historical diagnostics; current strength evals require v2+ current-family checkpoints."
        ),
    )
    root_puct_counterfactual.add_argument("--json", action="store_true", help="Print counterfactual search benchmark results as JSON.")
    root_puct_counterfactual.set_defaults(func=_root_puct_counterfactual)

    prior_belief_profile = subparsers.add_parser(
        "prior-belief-profile",
        help="Profile untempered priors and public-belief uncertainty from a controlled FoulPlay corpus.",
    )
    prior_belief_profile.add_argument(
        "--corpus",
        type=Path,
        required=True,
        help=(
            "pokezero.public-decision-corpus.v1 JSONL. Profiling rejects corpora with fewer than "
            f"{MINIMUM_PROFILE_DECISIONS} valid p1 decisions."
        ),
    )
    prior_belief_profile.add_argument(
        "--max-decisions",
        type=int,
        default=None,
        help=(
            "Optional deterministic prefix cap. The report records a hash of the selected records rather than "
            "the entire source file."
        ),
    )
    prior_belief_profile.add_argument(
        "--start-decision",
        type=int,
        default=0,
        help="Zero-based public-decision offset. Requires --shard and a bounded --max-decisions.",
    )
    prior_belief_profile.add_argument(
        "--source-start-decision",
        type=int,
        default=None,
        help=(
            "Logical source offset recorded by --shard. Use when CORPUS is a verified local snapshot "
            "whose first decision corresponds to a later source-corpus decision."
        ),
    )
    prior_belief_profile.add_argument(
        "--source-corpus-sha256",
        default=None,
        help=(
            "Full-source SHA-256 asserted by the caller when CORPUS is a bounded snapshot. "
            "The caller must verify it before use; matching shards preserve that provenance through merge."
        ),
    )
    prior_belief_profile.add_argument(
        "--shard",
        action="store_true",
        help="Write a bounded map-stage report. Shards are not capstone-eligible until merged.",
    )
    prior_belief_profile.add_argument("--checkpoint", type=Path, required=True, help="Checkpoint used for raw priors and value sweeps.")
    prior_belief_profile.add_argument("--showdown-root", type=Path, required=True, help="Built Pokemon Showdown checkout used for public-world replay.")
    prior_belief_profile.add_argument("--device", default=None, help="Torch device, e.g. cpu, cuda, mps.")
    prior_belief_profile.add_argument("--node-binary", default="node", help="Node executable used by public replay worlds.")
    prior_belief_profile.add_argument(
        "--world-sample-cap",
        type=int,
        default=DEFAULT_BELIEF_WORLD_SAMPLE_CAP,
        help="Maximum public-belief worlds; resolved K=min(cap, public combination count).",
    )
    prior_belief_profile.add_argument(
        "--opponent-scenarios",
        type=int,
        default=1,
        help="Hidden-mode opponent-prior scenarios evaluated per public belief world.",
    )
    prior_belief_profile.add_argument(
        "--entropy-thresholds",
        default="0.25,0.5,0.75,1.0,1.25,1.5,1.75,2.0",
        help="Comma-separated raw Shannon-entropy thresholds for the adaptive gate sweep.",
    )
    prior_belief_profile.add_argument(
        "--margin-thresholds",
        default="0,0.025,0.05,0.1,0.2,0.4",
        help="Comma-separated initial candidate-value top-two margin thresholds for the gate sweep.",
    )
    prior_belief_profile.add_argument(
        "--opponent-legal-mask-mode",
        choices=("hidden", "privileged"),
        default="hidden",
        help="Must remain hidden; privileged mode is rejected because this is a public-only audit.",
    )
    prior_belief_profile.add_argument("--out", type=Path, default=None, help="Optional JSON report path.")
    prior_belief_profile.add_argument("--json", action="store_true", help="Print the full JSON report.")
    prior_belief_profile.set_defaults(func=_prior_belief_profile)

    prior_belief_profile_merge = subparsers.add_parser(
        "prior-belief-profile-merge",
        help="Merge contiguous bounded prior/belief profile shards into one capstone-eligible report.",
    )
    prior_belief_profile_merge.add_argument(
        "--shards",
        type=Path,
        nargs="+",
        required=True,
        help="Shard JSON reports ordered by their zero-based public-decision range.",
    )
    prior_belief_profile_merge.add_argument("--out", type=Path, required=True, help="Merged JSON report path.")
    prior_belief_profile_merge.add_argument("--json", action="store_true", help="Print the merged JSON report.")
    prior_belief_profile_merge.set_defaults(func=_prior_belief_profile_merge)

    value_calibration = subparsers.add_parser(
        "value-calibration",
        help="Evaluate a neural checkpoint value head against rollout return targets.",
    )
    value_calibration.add_argument("--checkpoint", type=Path, required=True, help="Neural checkpoint path.")
    value_calibration.add_argument("--data", type=Path, nargs="+", required=True, help="One or more rollout JSONL files.")
    value_calibration.add_argument(
        "--eval-data",
        type=Path,
        nargs="+",
        default=None,
        help="Optional held-out rollout JSONL files used for the reported metrics after --fit-out.",
    )
    value_calibration.add_argument("--batch-size", type=int, default=128, help="Evaluation batch size.")
    value_calibration.add_argument("--bins", type=int, default=10, help="Number of prediction bins across [-1, 1].")
    value_calibration.add_argument("--device", default=None, help="Torch device, e.g. cpu, cuda, or mps.")
    value_calibration.add_argument("--min-examples", type=int, default=None, help="Fail if fewer calibration examples are evaluated.")
    value_calibration.add_argument("--max-mse", type=float, default=None, help="Fail if calibration MSE exceeds this threshold.")
    value_calibration.add_argument("--max-mae", type=float, default=None, help="Fail if calibration MAE exceeds this threshold.")
    value_calibration.add_argument("--max-abs-bias", type=float, default=None, help="Fail if absolute calibration bias exceeds this threshold.")
    value_calibration.add_argument(
        "--max-expected-calibration-error",
        type=float,
        default=None,
        help="Fail if expected calibration error exceeds this threshold.",
    )
    value_calibration.add_argument("--min-sign-accuracy", type=float, default=None, help="Fail if sign accuracy is below this threshold.")
    value_calibration.add_argument(
        "--min-pearson-correlation",
        type=float,
        default=None,
        help="Fail if linear value-return Pearson correlation is below this threshold or unavailable.",
    )
    value_calibration.add_argument(
        "--fit-out",
        type=Path,
        default=None,
        help="Optional output checkpoint path. Fits a value calibration transform on --data and saves a calibrated checkpoint copy.",
    )
    value_calibration.add_argument(
        "--fit-method",
        choices=("affine", "isotonic"),
        default="affine",
        help="Calibration transform fitted by --fit-out. affine preserves the legacy linear fit; isotonic fits a monotone empirical mapping.",
    )
    value_calibration.add_argument("--json", action="store_true", help="Print calibration results as JSON.")
    value_calibration.set_defaults(func=_value_calibration)

    value_calibration_compare = subparsers.add_parser(
        "value-calibration-compare",
        help="Fit and compare raw, affine, and isotonic value calibration on held-out rollout data.",
    )
    value_calibration_compare.add_argument("--checkpoint", type=Path, required=True, help="Neural checkpoint path.")
    value_calibration_compare.add_argument(
        "--data",
        type=Path,
        nargs="+",
        required=True,
        help="One or more rollout JSONL files used to fit calibration transforms.",
    )
    value_calibration_compare.add_argument(
        "--eval-data",
        type=Path,
        nargs="+",
        required=True,
        help="Held-out rollout JSONL files used to compare raw and calibrated metrics.",
    )
    value_calibration_compare.add_argument("--batch-size", type=int, default=128, help="Evaluation batch size.")
    value_calibration_compare.add_argument("--bins", type=int, default=10, help="Number of prediction bins across [-1, 1].")
    value_calibration_compare.add_argument("--device", default=None, help="Torch device, e.g. cpu, cuda, or mps.")
    value_calibration_compare.add_argument(
        "--selection-metric",
        choices=VALUE_SELECTION_METRICS,
        default="pearson_correlation",
        help="Metric used to select the best reported transform. Defaults to Pearson because search needs value ranking.",
    )
    value_calibration_compare.add_argument("--out", type=Path, default=None, help="Optional JSON output path.")
    value_calibration_compare.add_argument("--json", action="store_true", help="Print comparison results as JSON.")
    value_calibration_compare.set_defaults(func=_value_calibration_compare)

    iterate = subparsers.add_parser("iterate", help="Run neural-policy self-play training iterations.")
    iterate.add_argument("--run-dir", type=Path, required=True, help="Directory for rollouts, checkpoints, and manifests.")
    iterate.add_argument("--iterations", type=int, required=True, help="Number of collect/train/evaluate iterations.")
    iterate.add_argument("--resume", action="store_true", help="Continue an existing neural self-play run directory from its latest manifest.")
    iterate.add_argument(
        "--experiment-preset",
        choices=NEURAL_ITERATE_EXPERIMENT_PRESETS,
        default="none",
        help=(
            "Optional experiment preset. 'foundation-arms-race' fills the current WS-A CPU PPO "
            "arms-race recipe; 'recipe-fidelity' additionally aligns the PPO hyperparameters to "
            "the MIT thesis reference table (entropy 0.0588, 7 epochs, gamma 0.9999, GAE lambda, "
            "clip/value coefficients, grad-norm clip, batch size, base LR, LR annealing). Either preset only "
            "fills options not explicitly supplied on the command line."
        ),
    )
    iterate.add_argument("--games-per-iteration", type=int, required=True, help="Rollout games collected before each train step.")
    iterate.add_argument("--workers", type=int, default=16, help="Parallel rollout collection workers per iteration (capped at the game count).")
    iterate.add_argument("--showdown-root", type=Path, default=None, help="Built Pokemon Showdown checkout root.")
    iterate.add_argument("--format", dest="format_id", default="gen3randombattle", help="Showdown format id.")
    iterate.add_argument("--seed-start", type=int, default=1, help="First deterministic self-play seed.")
    iterate.add_argument("--max-decision-rounds", type=int, default=250, help="Rollout decision-round cap.")
    iterate.add_argument("--node-binary", default="node", help="Node executable used for the BattleStream bridge.")
    iterate.add_argument(
        "--training-cache-root",
        type=Path,
        default=None,
        help=(
            "Optional root for compact per-iteration training cache chunks. When set, neural self-play "
            "trains from array-backed caches instead of raw training-rollouts JSONL."
        ),
    )
    iterate.add_argument(
        "--training-cache-chunk-games",
        type=int,
        default=None,
        help=(
            "When --training-cache-root is set, flush compact training caches every N collected games. "
            "Defaults to one cache directory per iteration."
        ),
    )
    iterate.add_argument(
        "--max-cache-gb",
        type=float,
        default=MAX_ACTIVE_TRAINING_CACHE_GB,
        help=(
            f"Reject compact training-cache writes whose active cache root would exceed this many GiB "
            f"(default and maximum: {MAX_ACTIVE_TRAINING_CACHE_GB:g})."
        ),
    )
    iterate_cache_delete_group = iterate.add_mutually_exclusive_group()
    iterate_cache_delete_group.add_argument(
        "--delete-cache-after-read",
        dest="delete_cache_after_read",
        action="store_true",
        default=True,
        help="Delete per-iteration compact training cache chunks after checkpoint save and calibration uses finish.",
    )
    iterate_cache_delete_group.add_argument(
        "--keep-cache-after-read",
        dest="delete_cache_after_read",
        action="store_false",
        help="Keep compact training cache chunks after training for debugging or audit runs.",
    )
    iterate.add_argument(
        "--omit-rollout-jsonl",
        dest="write_rollout_jsonl",
        action="store_false",
        default=True,
        help=(
            "Do not write full raw rollouts.jsonl during collection. Requires --training-cache-root "
            "so the iteration still has trainable data."
        ),
    )
    iterate.add_argument("--initial-policy", required=True, help="Policy spec used before the first checkpoint exists.")
    iterate.add_argument(
        "--opponent-policy",
        action="append",
        default=None,
        help="Fixed opponent policy spec. May be repeated. Defaults to random-legal and simple-legal.",
    )
    iterate.add_argument(
        "--no-fixed-opponents",
        action="store_true",
        help=(
            "Use no fixed training opponents. Requires --mirror-match so collection can "
            "start from current-vs-current self-play."
        ),
    )
    iterate.add_argument(
        "--benchmark-reference-policy",
        action="append",
        default=None,
        help=(
            "Eval-only policy spec (e.g. max-damage) benchmarked against the candidate each "
            "iteration. May be repeated. Never used for rollout collection or training opponents."
        ),
    )
    iterate.add_argument(
        "--mirror-match",
        action="store_true",
        help=(
            "Add the current policy to the collection opponent pool so it plays copies of "
            "itself (current-vs-current self-play) from iteration 1, rather than self-play only "
            "starting once a checkpoint is promoted into the history pool."
        ),
    )
    iterate.add_argument(
        "--collection-temperature",
        type=float,
        default=1.0,
        help=(
            "Softmax sampling temperature for the self-play collector (>1 explores more). Applies "
            "only to rollout collection; benchmark/advancement use the deterministic policy. "
            "Default 1.0 (unchanged)."
        ),
    )
    iterate.add_argument(
        "--collection-epsilon",
        type=float,
        default=DEFAULT_COLLECTION_EXPLORATION_EPSILON,
        help=(
            "Minimum random legal exploration rate for learned policies during self-play "
            "collection. Applies only to rollout collection; benchmark/advancement stay "
            f"deterministic. Default {DEFAULT_COLLECTION_EXPLORATION_EPSILON}."
        ),
    )
    iterate.add_argument(
        "--collector-advancement-mode",
        choices=COLLECTOR_ADVANCEMENT_MODES,
        default="incumbent-gate",
        help=(
            "How a trained candidate becomes the next rollout collector. 'incumbent-gate' "
            "keeps the default head-to-head gate; 'always' advances every saved candidate for "
            "exploratory arms-race runs and is not promotion evidence; 'yardstick-gate' advances "
            "only candidates that improve over the best accepted max-damage yardstick score."
        ),
    )
    iterate.add_argument(
        "--tensorboard-logdir",
        type=Path,
        default=None,
        help="Write per-iteration TensorBoard scalars (loss, accuracy, win rate vs each benchmarked opponent, advancement) to this directory. Requires the tensorboard package (in the neural extra).",
    )
    iterate.add_argument("--max-historical-opponents", type=int, default=3, help="Number of older checkpoints kept in the opponent pool.")
    iterate.add_argument(
        "--historical-opponent-selection",
        choices=HISTORICAL_OPPONENT_SELECTION_MODES,
        default="recent",
        help=(
            "How to choose historical/promoted checkpoint opponents when more exist than "
            "--max-historical-opponents. 'recent' keeps the latest checkpoints; 'spread' "
            "deterministically spreads across the available history for more diverse league pressure."
        ),
    )
    iterate.add_argument(
        "--promotion-registry",
        type=Path,
        default=None,
        help="Optional promotion registry. When set, historical opponents come from promoted checkpoints instead of raw accepted neural checkpoints.",
    )
    iterate.add_argument(
        "--require-promoted-opponent-pool-size",
        type=int,
        default=None,
        help=(
            "Fail before rollout collection unless at least this many promoted historical opponents "
            "are selectable from the promotion registry after current-policy exclusion. "
            "Cannot exceed --max-historical-opponents."
        ),
    )
    iterate.add_argument(
        "--auto-promote",
        action="store_true",
        help="After each iteration, evaluate the promotion gate and record passing checkpoints in --promotion-registry.",
    )
    iterate.add_argument(
        "--promotion-artifact-dir",
        type=Path,
        default=None,
        help="Optional artifact directory for auto-promoted neural checkpoint copies.",
    )
    iterate.add_argument(
        "--promotion-label-prefix",
        default="neural-selfplay",
        help="Label prefix for auto-promotion entries. Use an empty string to omit labels.",
    )
    iterate.add_argument("--promotion-notes", default=None, help="Optional notes stored on each auto-promotion entry.")
    iterate.add_argument(
        "--allow-duplicate-promotion",
        action="store_true",
        help="Allow auto-promotion to record a checkpoint already present in the registry.",
    )
    _add_gate_arguments(iterate)
    iterate.add_argument(
        "--evaluation-games",
        type=int,
        default=0,
        help="Benchmark games per matchup after each train step. Required to be positive for multi-iteration runs.",
    )
    iterate.add_argument(
        "--evaluation-interval-games",
        type=int,
        default=None,
        help=(
            "Only run full benchmarks on iterations that cross this many collected self-play games. "
            "Use with collector advancement mode 'always' for milestone evals without per-iteration eval cost."
        ),
    )
    iterate.add_argument("--evaluation-seed-start", type=int, default=1_000_000, help="First deterministic benchmark seed.")
    iterate.add_argument("--epochs", type=int, default=1, help="Training epochs per iteration.")
    iterate.add_argument("--batch-size", type=int, default=64, help="Training batch size.")
    iterate.add_argument("--learning-rate", type=float, default=3e-4, help="AdamW learning rate.")
    iterate.add_argument(
        "--learning-rate-schedule",
        choices=LEARNING_RATE_SCHEDULES,
        default=CONSTANT_LEARNING_RATE_SCHEDULE,
        help=(
            "Learning-rate schedule. 'mit-thesis' applies base_lr/(8x+1)^1.5 using self-play "
            "run progress; each iteration's fresh optimizer receives its own progress window."
        ),
    )
    iterate.add_argument(
        "--learning-rate-schedule-total-games",
        type=int,
        default=None,
        help=(
            "Total self-play games mapped to LR progress 1.0 for non-constant schedules. "
            "The recipe-fidelity preset uses the thesis-scale 3,000,000-game denominator."
        ),
    )
    iterate.add_argument(
        "--learning-rate-schedule-completed-games",
        type=int,
        default=None,
        help=(
            "External self-play games already completed before this run starts, used as an LR progress "
            "offset for non-constant schedules. Use when continuing from a checkpoint in a fresh run_dir; "
            "--resume inherits the prior offset from the run manifest."
        ),
    )
    iterate.add_argument(
        "--learning-rate-warmup-progress",
        type=float,
        default=0.0,
        help="Linear LR warmup fraction of global progress, in [0, 1]. Below this progress the LR ramps "
        "0 -> the scheduled value; 0 disables warmup (default). Stabilizes cold-start at 50M+ scale. "
        "Compares against absolute global progress, so on a resumed run whose progress already starts "
        "past this fraction it never fires.",
    )
    iterate.add_argument("--weight-decay", type=float, default=0.0, help="AdamW weight decay.")
    iterate.add_argument("--window-size", type=int, default=1, help="Per-player observation history window (spec v2 default: 1).")
    iterate.add_argument("--discount", type=float, default=1.0, help="Terminal return discount per player decision.")
    iterate.add_argument("--capped-terminal-value", type=float, default=-0.25, help="Return assigned to each player in capped games.")
    iterate.add_argument(
        "--hp-delta-return-weight",
        type=float,
        default=0.0,
        help="Optional return-shaping weight for visible player-relative HP differential changes.",
    )
    iterate.add_argument(
        "--faint-delta-return-weight",
        type=float,
        default=0.0,
        help="Optional return-shaping weight for visible player-relative faint differential changes.",
    )
    iterate.add_argument(
        "--turn-penalty-after",
        type=int,
        default=None,
        help="Optional turn index at which to start applying a per-decision shaped return penalty.",
    )
    iterate.add_argument(
        "--turn-penalty",
        type=float,
        default=0.0,
        help="Optional positive per-decision return penalty applied at or after --turn-penalty-after.",
    )
    iterate.add_argument(
        "--shaping-weights",
        default=None,
        help=(
            "Dense potential-based reward shaping (WS-E arm 1) baked into collected caches and "
            "train targets: preset (wse-arm1), inline JSON, @/path/to.json, or 'none'. Default "
            "off — absent means exactly the unshaped iterate behavior. Stamped into checkpoint "
            "model configs; the shaping gamma is --discount."
        ),
    )
    iterate.add_argument("--value-loss-weight", type=float, default=0.25, help="Scalar value-head MSE loss weight.")
    iterate.add_argument(
        "--value-clip-range",
        type=float,
        default=None,
        help="Optional PPO value-function update clip range, using recorded rollout values as V_old.",
    )
    iterate.add_argument(
        "--value-ranking-loss-weight",
        type=float,
        default=0.0,
        help="Optional pairwise value-ranking loss weight. Optimizes leaf ordering for search when positive.",
    )
    iterate.add_argument(
        "--value-ranking-margin",
        type=float,
        default=0.0,
        help="Non-negative margin for --value-ranking-loss-weight pairwise value ordering.",
    )
    iterate.add_argument("--opponent-action-loss-weight", type=float, default=0.1, help="Opponent-action auxiliary loss weight.")
    iterate.add_argument(
        "--switch-action-loss-weight",
        type=float,
        default=1.0,
        help="Multiplier for switch-action policy CE examples under behavior-cloning / reward-weighted objectives.",
    )
    iterate.add_argument(
        "--action-family-loss-weight",
        type=float,
        default=0.0,
        help="Auxiliary move-vs-switch classification loss weight derived from legal action logits.",
    )
    iterate.add_argument(
        "--switch-target-loss-weight",
        type=float,
        default=0.0,
        help="Auxiliary conditional switch-target classification loss weight over switch-labeled examples.",
    )
    iterate.add_argument(
        "--objective",
        choices=("behavior-cloning", "reward-weighted", "ppo"),
        default="behavior-cloning",
        help=(
            "Training objective: supervised behavior cloning (default), reward-weighted "
            "behavior cloning, or PPO self-play RL."
        ),
    )
    iterate.add_argument("--clip-epsilon", type=float, default=0.2, help="PPO clipped-surrogate epsilon (objective=ppo).")
    iterate.add_argument("--entropy-coef", type=float, default=0.0, help="PPO entropy bonus coefficient (objective=ppo).")
    iterate.add_argument("--no-normalize-advantage", action="store_true", help="Disable PPO advantage normalization (objective=ppo).")
    iterate.add_argument(
        "--ppo-target-mode",
        choices=("returns", "gae"),
        default="returns",
        help="PPO advantage/value-target source: discounted returns or recorded-value GAE.",
    )
    iterate.add_argument("--gae-lambda", type=float, default=0.95, help="GAE lambda when --ppo-target-mode=gae.")
    iterate.add_argument(
        "--max-grad-norm",
        type=float,
        default=None,
        help="Optional global gradient-norm clip applied before each optimizer step (thesis recipe: 0.5430).",
    )
    iterate.add_argument(
        "--amp",
        choices=["bf16"],
        default=None,
        help="Mixed-precision autocast for forward/loss (WS-A1). 'bf16' keeps fp32 master weights/grads (no GradScaler). Default fp32.",
    )
    iterate.add_argument("--max-batches", type=int, default=None, help="Optional max batches per epoch for smoke runs.")
    iterate.add_argument("--device", default=None, help="Torch device, e.g. cpu, cuda, or mps. Defaults to cuda when available, else cpu.")
    iterate.add_argument(
        "--value-calibration",
        action="store_true",
        help="Evaluate value-head calibration after each iteration and store it in the manifest.",
    )
    iterate.add_argument(
        "--value-calibration-scope",
        choices=("iteration", "history"),
        default="iteration",
        help="Calibration data scope: latest iteration training rollouts or full accumulated training history.",
    )
    iterate.add_argument("--value-calibration-batch-size", type=int, default=128, help="Per-iteration calibration batch size.")
    iterate.add_argument("--value-calibration-bins", type=int, default=10, help="Per-iteration calibration bin count.")
    iterate.add_argument(
        "--value-selection",
        action="store_true",
        help="Evaluate value calibration after each training epoch and save the best value-calibrated epoch.",
    )
    iterate.add_argument(
        "--value-selection-scope",
        choices=("iteration", "history"),
        default="iteration",
        help="Selection data scope: latest iteration training rollouts or full accumulated training history.",
    )
    iterate.add_argument(
        "--value-selection-metric",
        choices=VALUE_SELECTION_METRICS,
        default="mae",
        help=(
            "Value metric used by --value-selection; sign_accuracy and pearson_correlation are maximized, "
            "others are minimized. pearson_correlation measures affine-invariant linear association, "
            "not calibration by itself."
        ),
    )
    iterate.add_argument("--value-selection-batch-size", type=int, default=128, help="Per-epoch value-selection batch size.")
    iterate.add_argument("--value-selection-bins", type=int, default=10, help="Per-epoch value-selection bin count.")
    iterate.add_argument(
        "--value-selection-heldout-games",
        type=int,
        default=0,
        help=(
            "Optional held-out self-play games per iteration used only for value selection. "
            "Default 0 selects on training rollouts."
        ),
    )
    iterate.add_argument(
        "--value-selection-seed-start",
        type=int,
        default=2_000_000,
        help="First deterministic seed for optional held-out value-selection games.",
    )
    iterate.add_argument("--embedding-dim", type=int, default=128, help="Transformer embedding width.")
    iterate.add_argument("--layers", type=int, default=2, help="Transformer encoder layer count.")
    iterate.add_argument("--attention-heads", type=int, default=4, help="Transformer attention head count.")
    iterate.add_argument("--feedforward-dim", type=int, default=256, help="Transformer feedforward width.")
    iterate.add_argument("--dropout", type=float, default=0.1, help="Transformer dropout.")
    iterate.add_argument(
        "--temporal-aggregator",
        choices=("mean", "gru"),
        default="mean",
        help="How to combine encoded observation history for value/opponent heads.",
    )
    # Ablation-arm feature masks (config, not spec): recorded on the model config and read back
    # into the env's encode-time masks, so collection, benchmarks, and later eval harnesses all
    # see the same masked observations (never a train/eval mask mismatch).
    iterate.add_argument(
        "--observation-schema",
        choices=("v2.1", "v2.2"),
        default=None,
        help=(
            "Observation schema for a FRESH iterate run: v2.1 or v2.2 (default; "
            "turn-merged transition tokens; sizes the model config, the env spec, and the "
            "schema-derived vocabulary). On --resume the run's stored model config wins; a "
            "disagreeing explicit flag fails the model-config equality validation."
        ),
    )
    iterate.add_argument(
        "--transition-token-budget",
        type=int,
        default=TRANSITION_TOKEN_COUNT,
        help="Most-recent transition-token slots filled at encode time. UNIT IS SCHEMA-DEPENDENT: under v2/v2.1 a token is one declared ACTION (32 = the K=16-turn ablation arm); under --observation-schema v2.2 a token is a whole TURN, so budget 32 covers roughly what 64 action-tokens did.",
    )
    iterate.add_argument(
        "--no-stats-block",
        action="store_true",
        help="Ablation arm: zero + attention-mask the stats token and per-mon tendency triple.",
    )
    iterate.add_argument(
        "--no-exact-state",
        action="store_true",
        help="Ablation arm: zero the exact-state layer (PP fractions, counters, expected stats).",
    )
    iterate.add_argument(
        "--tier2-residuals",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Populate + encode the Tier-2 residual channel (#505). Default: on.",
    )
    iterate.add_argument(
        "--tier2-investment",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Populate + encode the defender-side investment channel (#513): the reserved "
            "investment columns (120/139) behind the investment precision gate. A SEPARATE "
            "switch from --tier2-residuals. Default: OFF (byte-identical to pre-investment). "
            "Only meaningful under v2.1/v2.2 (no-op under v2). On --resume the run's stored "
            "model config wins; a disagreeing explicit flag fails the model-config equality "
            "validation."
        ),
    )
    iterate.add_argument("--policy-id", default="entity-transformer-selfplay", help="Base policy id for generated checkpoints.")
    iterate.add_argument(
        "--category-oov-buckets",
        type=int,
        default=DEFAULT_CATEGORY_OOV_BUCKETS,
        help="Reserved out-of-vocabulary rows in the compact category embedding.",
    )
    add_post_iteration_audit_arguments(iterate)
    iterate.add_argument("--json", action="store_true", help="Print the run manifest as JSON.")
    iterate.set_defaults(func=_iterate)

    foundation_plan = subparsers.add_parser(
        "foundation-plan",
        help="Print a CPU foundation arms-race neural iterate recipe without launching it.",
    )
    _add_foundation_arguments(foundation_plan, include_summary_path=False)
    foundation_plan.add_argument("--json", action="store_true", help="Print the recipe as JSON.")
    foundation_plan.set_defaults(func=_foundation_plan)

    foundation_run = subparsers.add_parser(
        "foundation-run",
        help="Execute a CPU foundation arms-race neural iterate recipe and write a summary artifact.",
    )
    _add_foundation_arguments(foundation_run, include_summary_path=True)
    foundation_run.set_defaults(func=_foundation_run)

    foundation_report = subparsers.add_parser(
        "foundation-report",
        help="Inspect a neural foundation-run summary artifact.",
    )
    foundation_report.add_argument("path", type=Path, help="Foundation run directory or neural-foundation-run-summary.json path.")
    foundation_report.add_argument("--json", action="store_true", help="Print the summary payload as JSON.")
    foundation_report.set_defaults(func=_foundation_report)

    foundation_value_tune_plan = subparsers.add_parser(
        "foundation-value-tune-plan",
        help="Print a value-only fine-tune recipe for a selected foundation checkpoint.",
    )
    _add_foundation_value_tune_arguments(foundation_value_tune_plan, include_summary_path=False)
    foundation_value_tune_plan.add_argument("--json", action="store_true", help="Print the recipe as JSON.")
    foundation_value_tune_plan.set_defaults(func=_foundation_value_tune_plan)

    foundation_value_tune_run = subparsers.add_parser(
        "foundation-value-tune-run",
        help="Run value-only fine-tuning for a selected foundation checkpoint and write a summary.",
    )
    _add_foundation_value_tune_arguments(foundation_value_tune_run, include_summary_path=True)
    foundation_value_tune_run.set_defaults(func=_foundation_value_tune_run)

    foundation_value_tune_report = subparsers.add_parser(
        "foundation-value-tune-report",
        help="Inspect a foundation value-tune summary artifact.",
    )
    foundation_value_tune_report.add_argument("path", type=Path, help="Value-tune output directory or summary JSON path.")
    foundation_value_tune_report.add_argument("--json", action="store_true", help="Print the summary payload as JSON.")
    foundation_value_tune_report.set_defaults(func=_foundation_value_tune_report)

    foundation_compare = subparsers.add_parser(
        "foundation-compare",
        help="Compare neural foundation-run summaries without requiring torch.",
    )
    foundation_compare.add_argument(
        "paths",
        type=Path,
        nargs="+",
        help="Foundation run directory or neural-foundation-run-summary.json path. Pass two or more for a useful comparison.",
    )
    foundation_compare.add_argument("--json", action="store_true", help="Print the comparison payload as JSON.")
    foundation_compare.add_argument(
        "--require-sample-sized",
        action="store_true",
        help="Quality-gate row pass requires foundation_evidence_status=present_and_sample_sized.",
    )
    foundation_compare.add_argument(
        "--candidate-source",
        choices=FOUNDATION_COMPARE_CANDIDATE_SOURCES,
        default="latest",
        help=(
            "Which candidate row drives comparison metrics and quality gates. "
            "'latest' preserves the historical latest-checkpoint view; 'latest-accepted' uses the "
            "collector-retained checkpoint when a gated run rejected newer candidates; "
            "'best-max-damage' uses the best manifest max-damage yardstick row."
        ),
    )
    foundation_compare.add_argument(
        "--min-max-damage-games",
        type=int,
        default=None,
        help="Quality-gate row pass requires at least this many latest max-damage yardstick games.",
    )
    foundation_compare.add_argument(
        "--min-max-damage-win-rate",
        type=float,
        default=None,
        help="Quality-gate row pass requires at least this latest candidate win rate versus max-damage.",
    )
    foundation_compare.add_argument(
        "--min-value-pearson-correlation",
        type=float,
        default=None,
        help="Quality-gate row pass requires at least this value-head Pearson correlation.",
    )
    foundation_compare.add_argument(
        "--min-value-sign-accuracy",
        type=float,
        default=None,
        help="Quality-gate row pass requires at least this value-head sign accuracy.",
    )
    foundation_compare.add_argument(
        "--max-value-expected-calibration-error",
        type=float,
        default=None,
        help="Quality-gate row pass requires value-head ECE at or below this value.",
    )
    foundation_compare.add_argument(
        "--require-quality-pass",
        action="store_true",
        help="Return exit 2 when quality thresholds are configured and no loaded row passes them.",
    )
    foundation_compare.set_defaults(func=_foundation_compare)

    report = subparsers.add_parser("report", help="Print a summary of a neural self-play run manifest.")
    report.add_argument("--run-dir", type=Path, required=True, help="Neural self-play run directory containing manifest.json.")
    report.add_argument("--json", action="store_true", help="Print the raw run manifest as formatted JSON.")
    report.set_defaults(func=_report)

    return parser


def _add_foundation_arguments(parser: argparse.ArgumentParser, *, include_summary_path: bool) -> None:
    profile_choices = tuple(NEURAL_FOUNDATION_PROFILES)
    variant_choices = tuple(NEURAL_FOUNDATION_VARIANTS)
    parser.add_argument("--run-dir", type=Path, required=True, help="Neural self-play run directory.")
    parser.add_argument("--showdown-root", type=Path, required=True, help="Built Pokemon Showdown checkout root.")
    parser.add_argument(
        "--initial-policy",
        default="random-legal",
        help="Initial rollout collector policy spec. Defaults to random-legal for a cold CPU self-play start.",
    )
    parser.add_argument(
        "--continue-from",
        type=Path,
        default=None,
        help=(
            "Continue a fresh foundation run from an existing neural self-play/foundation run directory "
            "or summary. Resolves the latest checkpoint as the initial policy and uses prior collected "
            "games as the LR schedule completed-games offset."
        ),
    )
    parser.add_argument(
        "--profile",
        choices=profile_choices,
        default="smoke",
        help=(
            "Foundation run size profile. smoke is cheap plumbing; pilot matches the current "
            "3x256 CPU recipe; midscale is a ~50k rising-curve gate with 1.6k PPO updates."
        ),
    )
    parser.add_argument(
        "--variant",
        choices=variant_choices,
        default="baseline",
        help="Foundation experiment arm for auxiliary-loss, temporal, and opponent-pool ablations.",
    )
    parser.add_argument("--iterations", type=int, default=None, help="Override profile iterations.")
    parser.add_argument("--games-per-iteration", type=int, default=None, help="Override profile games per iteration.")
    parser.add_argument("--workers", type=int, default=None, help="Override profile rollout workers.")
    parser.add_argument("--evaluation-games", type=int, default=None, help="Override profile benchmark games per matchup.")
    parser.add_argument(
        "--evaluation-interval-games",
        type=int,
        default=None,
        help="Override profile benchmark interval in collected self-play games.",
    )
    parser.add_argument("--epochs", type=int, default=None, help="Override profile training epochs per iteration.")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help=(
            "Override the nested neural iterate training batch size. Recipe-fidelity defaults to "
            "1024; larger CPU-throughput reads should pass this explicitly so the off-recipe "
            "batch size is visible in the generated plan and audit."
        ),
    )
    parser.add_argument("--max-batches", type=int, default=None, help="Override profile max batches per epoch. Use -1 for no cap.")
    parser.add_argument(
        "--learning-rate-schedule-total-games",
        type=int,
        default=None,
        help=(
            "Override the nested neural iterate LR schedule denominator. Use this for midscale "
            "recipe-fidelity reads that should anneal over the read's own total game count."
        ),
    )
    parser.add_argument(
        "--learning-rate-schedule-completed-games",
        type=int,
        default=None,
        help=(
            "External self-play games completed before this foundation run starts. Usually derived "
            "from --continue-from; pass explicitly only when continuing from a checkpoint without a run manifest."
        ),
    )
    parser.add_argument(
        "--value-selection-heldout-games",
        type=int,
        default=None,
        help="Override profile held-out value-selection games per iteration.",
    )
    parser.add_argument("--seed-start", type=int, default=1, help="First rollout collection seed.")
    parser.add_argument("--evaluation-seed-start", type=int, default=1_000_000, help="First benchmark seed.")
    parser.add_argument(
        "--opponent-action-loss-weight",
        type=float,
        default=None,
        help="Override the variant's opponent-action auxiliary loss weight.",
    )
    parser.add_argument(
        "--value-ranking-loss-weight",
        type=float,
        default=None,
        help="Opt into pairwise value-ranking loss for the nested neural iterate command.",
    )
    parser.add_argument(
        "--value-ranking-margin",
        type=float,
        default=None,
        help="Override the pairwise value-ranking margin for the nested neural iterate command.",
    )
    parser.add_argument(
        "--value-clip-range",
        type=float,
        default=None,
        help="Override the nested neural iterate PPO value-function clip range.",
    )
    parser.add_argument(
        "--opponent-policy",
        action="append",
        default=None,
        help=(
            "Override the variant's fixed self-play opponent policy specs. May be repeated. "
            "When omitted, the variant decides whether to use iterate defaults or a custom pool."
        ),
    )
    parser.add_argument(
        "--temporal-aggregator",
        choices=("mean", "gru"),
        default=None,
        help="Override the variant's temporal aggregator for value/opponent heads.",
    )
    parser.add_argument(
        "--embedding-dim",
        type=int,
        default=None,
        help="Override the nested neural iterate transformer embedding width.",
    )
    parser.add_argument(
        "--layers",
        type=int,
        default=None,
        help="Override the nested neural iterate transformer layer count. Use 0 for the CPU-fast pooled encoder.",
    )
    parser.add_argument(
        "--attention-heads",
        type=int,
        default=None,
        help="Override the nested neural iterate transformer attention head count.",
    )
    parser.add_argument(
        "--feedforward-dim",
        type=int,
        default=None,
        help="Override the nested neural iterate transformer feedforward width.",
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=None,
        help="Override the nested neural iterate transformer dropout.",
    )
    parser.add_argument(
        "--category-oov-buckets",
        type=int,
        default=None,
        help="Override compact categorical embedding OOV bucket count for the nested neural iterate command.",
    )
    parser.add_argument(
        "--collector-advancement-mode",
        choices=COLLECTOR_ADVANCEMENT_MODES,
        default=None,
        help="Override the foundation preset's rollout-collector advancement mode.",
    )
    parser.add_argument(
        "--training-cache-root",
        type=Path,
        default=None,
        help=(
            "Optional compact training-cache root passed through to neural iterate. Use this for "
            "mid-scale runs that should avoid raw training-rollouts JSONL."
        ),
    )
    parser.add_argument(
        "--training-cache-chunk-games",
        type=int,
        default=None,
        help=(
            "Flush compact training cache chunks every N collected games in the nested neural iterate command. "
            "Requires --training-cache-root."
        ),
    )
    parser.add_argument(
        "--max-cache-gb",
        type=float,
        default=MAX_ACTIVE_TRAINING_CACHE_GB,
        help=(
            f"Reject compact training-cache writes whose active cache root would exceed this many GiB "
            f"(default and maximum: {MAX_ACTIVE_TRAINING_CACHE_GB:g})."
        ),
    )
    foundation_cache_delete_group = parser.add_mutually_exclusive_group()
    foundation_cache_delete_group.add_argument(
        "--delete-cache-after-read",
        dest="delete_cache_after_read",
        action="store_true",
        default=True,
        help="Delete per-iteration compact training cache chunks after nested PPO train uses finish.",
    )
    foundation_cache_delete_group.add_argument(
        "--keep-cache-after-read",
        dest="delete_cache_after_read",
        action="store_false",
        help="Keep compact training cache chunks after nested PPO training for debugging or audit runs.",
    )
    parser.add_argument(
        "--omit-rollout-jsonl",
        dest="write_rollout_jsonl",
        action="store_false",
        default=True,
        help="Do not write full raw rollouts.jsonl in the nested neural iterate command. Requires --training-cache-root.",
    )
    parser.add_argument(
        "--recipe-fidelity",
        action="store_true",
        help=(
            "Use the recipe-fidelity experiment preset instead of foundation-arms-race: aligns the "
            "PPO hyperparameters to the MIT thesis reference table (entropy 0.0588, 7 epochs, "
            "gamma 0.9999, GAE lambda, clip/value coefficients, grad-norm clip, batch size, base LR, LR annealing)."
        ),
    )
    parser.add_argument("--device", default=None, help="Torch device for the underlying neural iterate command.")
    parser.add_argument("--resume", action="store_true", help="Resume an existing neural foundation run directory.")
    if include_summary_path:
        parser.add_argument(
            "--summary-path",
            type=Path,
            default=None,
            help="Where to write the wrapper summary. Defaults to RUN_DIR/neural-foundation-run-summary.json.",
        )


def _add_foundation_value_tune_arguments(parser: argparse.ArgumentParser, *, include_summary_path: bool) -> None:
    parser.add_argument("path", type=Path, help="Foundation run directory or neural-foundation-run-summary.json path.")
    parser.add_argument(
        "--candidate-source",
        choices=FOUNDATION_COMPARE_CANDIDATE_SOURCES,
        default="latest-accepted",
        help="Foundation candidate to value-tune. Defaults to the retained latest accepted checkpoint.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Output directory. Defaults to RUN_DIR/value-tune/CANDIDATE_SOURCE-iteration-NNNN.",
    )
    parser.add_argument("--epochs", type=int, default=3, help="Value-only fine-tune epochs.")
    parser.add_argument("--batch-size", type=int, default=64, help="Value-only fine-tune batch size.")
    parser.add_argument("--learning-rate", type=float, default=1e-4, help="Value-only fine-tune AdamW learning rate.")
    parser.add_argument(
        "--value-ranking-loss-weight",
        type=float,
        default=0.0,
        help="Optional pairwise value-ranking loss weight for value-only fine-tuning.",
    )
    parser.add_argument(
        "--value-ranking-margin",
        type=float,
        default=0.0,
        help="Non-negative margin for --value-ranking-loss-weight pairwise value ordering.",
    )
    parser.add_argument(
        "--value-selection-metric",
        choices=VALUE_SELECTION_METRICS,
        default="pearson_correlation",
        help="Held-out metric used to select the best value-only epoch.",
    )
    parser.add_argument("--value-calibration-batch-size", type=int, default=128, help="Calibration batch size.")
    parser.add_argument("--value-calibration-bins", type=int, default=10, help="Calibration bin count.")
    parser.add_argument(
        "--calibration-data",
        type=Path,
        nargs="+",
        default=None,
        help="Optional independent rollout JSONL path(s) for final value calibration reporting.",
    )
    parser.add_argument(
        "--require-heldout-selection",
        action="store_true",
        help="Fail unless the selected candidate has value-selection held-out rollout paths.",
    )
    parser.add_argument("--max-batches", type=int, default=None, help="Optional max training batches per epoch for smoke runs.")
    parser.add_argument("--device", default=None, help="Torch device for the underlying train command.")
    if include_summary_path:
        parser.add_argument(
            "--summary-path",
            type=Path,
            default=None,
            help="Where to write the value-tune summary. Defaults to OUT_DIR/neural-foundation-value-tune-summary.json.",
        )


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    args = parser.parse_args(raw_argv)
    args._explicit_cli_options = _explicit_cli_options(raw_argv)
    try:
        if int(os.environ.get("WORLD_SIZE", "1")) > 1 and args.command != "train":
            raise ValueError("torchrun multi-process execution is supported only for the neural_cli train command.")
        return int(args.func(args))
    except RunAuditFailure as exc:
        _print_run_audit_failure(exc)
        return 3
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


def _prior_belief_profile(args: argparse.Namespace) -> int:
    """Run the public-only Step 2 prior/belief uncertainty profile."""

    if args.opponent_legal_mask_mode != "hidden":
        raise ValueError("prior-belief-profile refuses privileged opponent legal-mask mode.")
    shard_mode = bool(getattr(args, "shard", False))
    start_decision = int(getattr(args, "start_decision", 0))
    source_start_decision = getattr(args, "source_start_decision", None)
    source_corpus_sha256 = getattr(args, "source_corpus_sha256", None)
    if start_decision < 0:
        raise ValueError("--start-decision must be non-negative.")
    if shard_mode and args.max_decisions is None:
        raise ValueError("--shard requires a bounded --max-decisions.")
    if not shard_mode and start_decision:
        raise ValueError("--start-decision requires --shard.")
    if source_start_decision is not None:
        if not shard_mode:
            raise ValueError("--source-start-decision requires --shard.")
        if source_start_decision < 0:
            raise ValueError("--source-start-decision must be non-negative.")
        if start_decision:
            raise ValueError("--source-start-decision cannot be combined with --start-decision.")
    if source_corpus_sha256 is not None:
        if not shard_mode:
            raise ValueError("--source-corpus-sha256 requires --shard.")
        if len(source_corpus_sha256) != 64 or any(
            character not in "0123456789abcdef" for character in source_corpus_sha256
        ):
            raise ValueError("--source-corpus-sha256 must be a lowercase SHA-256 digest.")
    corpus = open_public_decision_corpus(
        args.corpus,
        max_decisions=args.max_decisions,
        start_decision=start_decision,
    )
    checkpoint_sha256 = sha256_file(args.checkpoint)
    captured_checkpoint_sha256 = corpus.manifest.get("checkpoint_sha256")
    if captured_checkpoint_sha256 != checkpoint_sha256:
        raise ValueError(
            "checkpoint hash does not match the public corpus manifest; profile the checkpoint that generated "
            "the controlled capture or capture a new corpus."
        )
    model, result = load_transformer_checkpoint(args.checkpoint, map_location=args.device)
    observation_spec = observation_spec_from_model_config(result.model_config)
    vocab = gen3_category_vocabulary(
        args.showdown_root,
        include_turn_merged=observation_spec.schema_version == OBSERVATION_SCHEMA_VERSION_V2_2,
    )
    env_config = LocalShowdownConfig(
        showdown_root=args.showdown_root,
        node_binary=args.node_binary,
        observation_spec=observation_spec,
        category_vocab=vocab,
        feature_masks=feature_masks_from_model_config(result.model_config),
    )
    set_source = load_gen3_randbat_source_cached(args.showdown_root)
    captured_source_hash = corpus.manifest.get("belief_set_source_hash")
    if captured_source_hash != set_source.metadata.source_hash:
        raise ValueError(
            "public corpus belief set-source hash does not match the current Showdown source; "
            "refuse to profile against a different public belief universe."
        )
    profile_config = PriorBeliefProfileConfig(
        entropy_thresholds=_parse_profile_thresholds(args.entropy_thresholds, option="--entropy-thresholds"),
        margin_thresholds=_parse_profile_thresholds(args.margin_thresholds, option="--margin-thresholds"),
        world_sample_cap=args.world_sample_cap,
        opponent_legal_mask_mode="hidden",
        root_noise_enabled=False,
    )

    def prior_evaluator(observations: tuple[PokeZeroObservationV0, ...]) -> tuple[float, ...]:
        # Adaptive gates always see raw checkpoint priors: no collection/root temperature or noise.
        return evaluate_transformer_action_priors(
            model=model,
            result=result,
            observations=observations,
            temperature=1.0,
            device=args.device,
        )

    def value_evaluator(observations: tuple[PokeZeroObservationV0, ...]) -> float:
        return evaluate_transformer_observation_value(
            model=model,
            result=result,
            observations=observations,
            device=args.device,
        )

    def opponent_prior_evaluator(observations: tuple[PokeZeroObservationV0, ...]) -> tuple[float, ...]:
        return evaluate_transformer_opponent_action_priors(
            model=model,
            result=result,
            observations=observations,
            temperature=1.0,
            device=args.device,
        )

    candidate_value_evaluator = PublicPrefixCandidateValueEvaluator(
        env_factory=lambda: LocalShowdownEnv(env_config),
        value_evaluator=value_evaluator,
        opponent_prior_evaluator=opponent_prior_evaluator,
        set_source=set_source,
        world_sample_cap=profile_config.world_sample_cap,
        scenario_count=args.opponent_scenarios,
    )
    profile_kwargs = {
        "prior_evaluator": prior_evaluator,
        "candidate_value_evaluator": candidate_value_evaluator,
        "config": profile_config,
        "belief_set_source": set_source,
        "provenance": {
            "checkpoint": str(args.checkpoint),
            "checkpoint_sha256": checkpoint_sha256,
            "corpus_source_path": str(args.corpus),
            "corpus_max_decisions": args.max_decisions,
            "belief_set_source_hash": set_source.metadata.source_hash,
            "root_noise_enabled": False,
            "opponent_legal_mask_mode": "hidden",
            "opponent_scenarios": args.opponent_scenarios,
        },
        # The replay-from-root candidate sweep can be slow; leave a durable
        # heartbeat in the Job stderr log after every consumed corpus decision.
        "progress_callback": lambda completed, record: print(
            f"prior_belief_profile_progress consumed={completed} decision_id={record.decision_id}",
            file=sys.stderr,
            flush=True,
        ),
    }
    report = (
        profile_public_corpus_shard(
            corpus,
            **profile_kwargs,
            source_start_decision=source_start_decision,
            source_corpus_sha256=source_corpus_sha256,
        )
        if shard_mode
        else profile_public_corpus(corpus, **profile_kwargs)
    )
    if args.out is not None:
        _write_json(args.out, report)
        print(f"prior_belief_profile: {args.out}")
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    elif args.out is None:
        print(
            f"profiled {report['decision_count']} public decisions and "
            f"{report['selection_context_count']} hidden-mode selection contexts"
        )
    return 0


def _prior_belief_profile_merge(args: argparse.Namespace) -> int:
    """Merge validated map-stage reports without reopening the public corpus."""

    shards: list[dict[str, Any]] = []
    for path in args.shards:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError(f"profile shard must be a JSON object: {path}")
        shards.append(payload)
    report = merge_public_corpus_profile_shards(shards)
    _write_json(args.out, report)
    print(f"prior_belief_profile: {args.out}")
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    return 0


def _parse_profile_thresholds(value: str, *, option: str) -> tuple[float, ...]:
    try:
        thresholds = tuple(float(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as exc:
        raise ValueError(f"{option} must be a comma-separated list of finite non-negative numbers.") from exc
    if not thresholds or any(item < 0.0 or not math.isfinite(item) for item in thresholds):
        raise ValueError(f"{option} must be a comma-separated list of finite non-negative numbers.")
    return thresholds


def _explicit_cli_options(argv: Iterable[str]) -> frozenset[str]:
    parser = build_arg_parser()
    _suppress_parser_defaults(parser)
    parsed = parser.parse_args(list(argv))
    return frozenset(vars(parsed))


def _suppress_parser_defaults(parser: argparse.ArgumentParser) -> None:
    parser._defaults.clear()
    for action in parser._actions:
        action.default = argparse.SUPPRESS
        if isinstance(action, argparse._SubParsersAction):
            for subparser in action.choices.values():
                _suppress_parser_defaults(subparser)


def _serve(args: argparse.Namespace) -> int:
    from .inference_service import serve_forever

    print(f"neural serve: loading {args.checkpoint} on device={args.device or 'auto'} amp={args.amp or 'fp32'}")
    print(f"neural serve: batched (max_batch={args.max_batch}, window={args.batch_window_ms}ms); listening on {args.host}:{args.port}")
    serve_forever(
        args.checkpoint, host=args.host, port=args.port, device=args.device, amp=args.amp,
        max_batch=args.max_batch, batch_window_ms=args.batch_window_ms,
    )
    return 0


def _describe(args: argparse.Namespace) -> int:
    # The category embedding is a compact vocabulary built at train time; use a minimal
    # placeholder here just to surface the architecture defaults.
    config = TransformerPolicyConfig.compact_category(category_vocab=("placeholder",), category_oov_buckets=1)
    model_config = config.to_dict()
    for key in ("category_vocab", "categorical_vocab_size", "category_oov_buckets"):
        model_config.pop(key, None)
    model_config["category_embedding"] = "compact vocabulary built at train time (legacy hash embedding retired)"
    payload = {
        "torch_available": torch_available(),
        "model_config": model_config,
    }
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print(f"torch_available: {payload['torch_available']}")
        print(f"policy_id: {config.policy_id}")
        print(f"window_size: {config.window_size}")
        print(f"token_count: {config.token_count}")
        print(f"categorical_feature_count: {config.categorical_feature_count}")
        print(f"numeric_feature_count: {config.numeric_feature_count}")
        print(f"embedding_dim: {config.embedding_dim}")
        print(f"layers: {config.transformer_layers}")
        print(f"attention_heads: {config.attention_heads}")
        print(f"temporal_aggregator: {config.temporal_aggregator}")
    return 0


def _cache_data(args: argparse.Namespace) -> int:
    summary = write_training_cache_from_rollouts(
        args.data,
        args.out,
        config=_training_dataset_config_from_args(args),
        overwrite=args.overwrite,
        max_cache_root_bytes=_cache_gb_to_bytes(args.max_cache_gb),
        cache_root=args.out.parent,
    )
    print(f"training_cache: {summary.path}")
    print(f"training_cache_records: {summary.record_count}")
    print(f"training_cache_examples: {summary.example_count}")
    print(f"training_cache_bytes: {summary.byte_size}")
    return 0


_MASK_FLAG_FIELDS = (
    ("--transition-token-budget", "transition_token_budget"),
    ("--no-stats-block", "stats_block_enabled"),
    ("--no-exact-state", "exact_state_enabled"),
    ("--tier2-residuals/--no-tier2-residuals", "tier2_residuals"),
    ("--tier2-investment/--no-tier2-investment", "tier2_investment"),
)


def _explicit_mask_requests(args: argparse.Namespace) -> dict[str, object]:
    """Model-config mask fields the user explicitly requested on the command line."""
    requested: dict[str, object] = {}
    if args.transition_token_budget is not None:
        requested["transition_token_budget"] = args.transition_token_budget
    if args.no_stats_block:
        requested["stats_block_enabled"] = False
    if args.no_exact_state:
        requested["exact_state_enabled"] = False
    if args.tier2_residuals is not None:
        requested["tier2_residuals"] = bool(args.tier2_residuals)
    if getattr(args, "tier2_investment", None) is not None:
        requested["tier2_investment"] = bool(args.tier2_investment)
    return requested


def _require_mask_flags_agree_with_checkpoint(args: argparse.Namespace, model_config) -> None:
    """With --initial-checkpoint the masks are the checkpoint's; explicit flags must agree.

    Retraining a checkpoint under different observation masks silently changes what the
    already-trained weights see (the #492 mismatch class), so a disagreement hard-fails.
    """
    requested_fields = _explicit_mask_requests(args)
    flags_by_field = {field_name: flag for flag, field_name in _MASK_FLAG_FIELDS}
    for field_name, requested in requested_fields.items():
        current = getattr(model_config, field_name)
        if current != requested:
            raise ValueError(
                f"{flags_by_field[field_name]} requests {field_name}={requested!r} but the "
                f"initial checkpoint was trained with {field_name}={current!r}; masks cannot "
                "change across a resume."
            )


def _require_schema_flag_agrees_with_checkpoint(args: argparse.Namespace, model_config) -> None:
    """With --initial-checkpoint the observation schema is the checkpoint's; an explicit
    --observation-schema must agree or hard-fail (the schema axis of the mask latch —
    retraining under a different schema silently changes every observation column)."""
    requested = observation_schema_version_from_choice(args.observation_schema)
    if requested is None:
        return
    current = model_config.observation_schema_version
    if current != requested:
        raise ValueError(
            f"--observation-schema {args.observation_schema} requests {requested!r} but the "
            f"initial checkpoint was trained under {current!r}; the observation schema "
            "cannot change across a resume (drop the flag to adopt the checkpoint's schema)."
        )


def _require_cache_observation_schema_matches(paths, model_config) -> None:
    """Hard-fail (both directions) when cache schema provenance disagrees with this run.

    The schema-axis twin of the mask cross-check: collection stamps the encoding env's
    observation schema into cache metadata. A cache without the field is LEGACY —
    recorded before schema stamping shipped, which is by definition pre-v2.2 data: it
    passes under a v2/v2.1 model (indistinguishable from today) but REFUSES under a
    v2.2 model (turn-merged rows cannot have come from a legacy collector).
    """
    expected = model_config.observation_schema_version
    for cache_path, recorded in cache_observation_schemas_by_path(paths):
        if recorded is None:
            if expected == OBSERVATION_SCHEMA_VERSION_V2_2:
                raise ValueError(
                    f"training cache {cache_path} records no observation schema (legacy "
                    "collector) but this train run declares "
                    f"{OBSERVATION_SCHEMA_VERSION_V2_2!r}; v2.2 (turn-merged) observations "
                    "must come from a collector run with --observation-schema v2.2."
                )
            continue
        if recorded != expected:
            raise ValueError(
                f"training cache {cache_path} was collected under observation schema "
                f"{recorded!r} but this train run's model config declares {expected!r}; "
                "cross-schema observations must not be trained through."
            )


def _require_cache_masks_match_model_config(paths, model_config) -> None:
    """Hard-fail when a training cache records encode-time masks the model config lacks.

    Collection stamps the resolved masks into cache metadata (the mask-axis twin of the
    belief-provenance hash); a mismatch means the observations in the cache were encoded
    under different masks than this training run declares — never train through it.
    Legacy caches and JSONL inputs record none and cannot be checked.
    """
    expected = {
        "stats_block": model_config.stats_block_enabled,
        "exact_state": model_config.exact_state_enabled,
        "transition_token_budget": model_config.transition_token_budget,
        "tier2_residuals": model_config.tier2_residuals,
        # getattr: duck-typed/legacy config objects without the field are
        # pre-investment by definition (same asymmetric default as the payload latch).
        "tier2_investment": getattr(model_config, "tier2_investment", False),
    }
    for cache_path, masks in cache_feature_masks_by_path(paths):
        if masks is None:
            continue
        # Caches collected before the investment channel record no field; they were
        # encoded with the column constant zero, i.e. tier2_investment=False.
        masks = {"tier2_investment": False, **masks}
        if masks != expected:
            raise ValueError(
                f"training cache {cache_path} was collected under feature masks {masks!r} "
                f"but this train run's model config declares {expected!r}; "
                "mask-mismatched observations must not be trained through."
            )


def _resolved_training_shaping_json(args: argparse.Namespace, initial_training_result) -> str | None:
    """Canonical shaping JSON for this train run (None = unshaped).

    Explicit --shaping-weights wins ('none' spells explicit-off); with an initial
    checkpoint and no flag, the checkpoint's stamped shaping carries forward (same
    adopt-from-checkpoint posture as the mask flags). Unlike masks, retargeting a
    checkpoint under different shaping is a legitimate training operation (targets
    change, observations do not), so an explicit disagreement re-stamps rather than
    hard-fails — with a loud notice.
    """
    checkpoint_json = (
        initial_training_result.model_config.reward_shaping
        if initial_training_result is not None
        else None
    )
    if args.shaping_weights is None:
        return checkpoint_json
    explicit = parse_shaping_spec(args.shaping_weights)
    explicit_json = explicit.canonical_json() if explicit is not None else None
    checkpoint_compare_json = _canonical_shaping_json_for_compare(checkpoint_json)
    if initial_training_result is not None and explicit_json != checkpoint_compare_json:
        print(
            "notice: --shaping-weights re-targets the initial checkpoint "
            f"(checkpoint shaping: {checkpoint_json or 'unshaped'} -> {explicit_json or 'unshaped'}); "
            "the saved checkpoint will be stamped with the new shaping config.",
            file=sys.stderr,
        )
    return explicit_json


def _canonical_shaping_json_for_compare(value: str | None) -> str | None:
    if not value:
        return None
    config = parse_shaping_spec(value)
    return config.canonical_json() if config is not None else None


def _require_cache_shaping_matches_training_config(paths, shaping_json: str | None) -> None:
    """Hard-fail (both directions) when cache shaping provenance disagrees with this run.

    The shaping-axis twin of the mask cross-check (#507): collection bakes shaped
    returns/targets into the cache and stamps the config into metadata. Training a
    shaped cache with the flag off silently keeps dense targets the run never declared;
    training an unshaped/legacy cache with the flag on silently trains WITHOUT the
    requested shaping. Caches whose metadata lacks the field are definitively unshaped
    (their returns predate shaping); only unreadable metadata and JSONL inputs skip
    (JSONL returns are recomputed under this run's config, so they cannot mismatch).
    """
    expected = ShapingConfig.from_json(shaping_json).to_dict() if shaping_json is not None else None
    for cache_path, shaping, checkable in cache_shaping_configs_by_path(paths):
        if not checkable:
            continue
        recorded = ShapingConfig.from_dict(shaping).to_dict() if shaping is not None else None
        if recorded != expected:
            raise ValueError(
                f"training cache {cache_path} was collected with shaping "
                f"{recorded if recorded is not None else 'off'} but this train run requests "
                f"{expected if expected is not None else 'off'}; shaped and unshaped "
                "returns/targets must not be mixed. Re-collect the cache or match --shaping-weights."
            )


def _train(args: argparse.Namespace) -> int:
    command_started_at = datetime.now(timezone.utc)
    command_started = time.perf_counter()
    # Surface the missing-neural-extra message before any file I/O (vocab building reads data).
    require_torch()
    distributed_context_value = initialize_distributed_training(args.device)
    resolved_device = require_torch().device(resolve_torch_device(args.device))
    if distributed_context_value.enabled and resolved_device.type == "cuda":
        resolved_device = require_torch().device("cuda", distributed_context_value.local_rank)
    is_primary_rank = distributed_context_value.is_primary
    if args.value_calibration_out is not None and not args.value_calibration_data:
        raise ValueError("--value-calibration-out requires --value-calibration-data.")
    if args.value_selection_out is not None and not args.value_selection_data:
        raise ValueError("--value-selection-out requires --value-selection-data.")
    refutation_cache_paths = _validate_refutation_cache_args(args)
    cache_lifecycle = _training_cache_lifecycle(args) if is_primary_rank else _TrainingCacheLifecycle()
    input_data_bytes = _input_data_paths_byte_size(args.data) if args.summary_out is not None and is_primary_rank else None
    refutation_cache_bytes = (
        _input_data_paths_byte_size(refutation_cache_paths)
        if args.summary_out is not None and refutation_cache_paths and is_primary_rank
        else None
    )
    initial_model = None
    initial_training_result = None
    if args.initial_checkpoint is not None:
        initial_model, initial_training_result = load_transformer_checkpoint(
            args.initial_checkpoint,
            map_location=str(resolved_device),
        )
    shaping_weights_json = _resolved_training_shaping_json(args, initial_training_result)
    training_config = TransformerTrainingConfig(
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        learning_rate_schedule=args.learning_rate_schedule,
        learning_rate_schedule_total_games=args.learning_rate_schedule_total_games,
        learning_rate_progress_start=args.learning_rate_progress_start,
        learning_rate_progress_end=args.learning_rate_progress_end,
        learning_rate_warmup_progress=args.learning_rate_warmup_progress,
        weight_decay=args.weight_decay,
        window_size=args.window_size,
        discount=args.discount,
        capped_terminal_value=args.capped_terminal_value,
        hp_delta_return_weight=args.hp_delta_return_weight,
        faint_delta_return_weight=args.faint_delta_return_weight,
        turn_penalty_after=args.turn_penalty_after,
        turn_penalty=args.turn_penalty,
        value_loss_weight=args.value_loss_weight,
        value_clip_range=args.value_clip_range,
        value_ranking_loss_weight=args.value_ranking_loss_weight,
        value_ranking_margin=args.value_ranking_margin,
        opponent_action_loss_weight=args.opponent_action_loss_weight,
        switch_action_loss_weight=args.switch_action_loss_weight,
        action_family_loss_weight=args.action_family_loss_weight,
        switch_target_loss_weight=args.switch_target_loss_weight,
        max_batches=args.max_batches,
        device=str(resolved_device),
        objective=args.objective,
        clip_epsilon=args.clip_epsilon,
        entropy_coef=args.entropy_coef,
        normalize_advantage=not args.no_normalize_advantage,
        ppo_target_mode=args.ppo_target_mode,
        gae_lambda=args.gae_lambda,
        max_grad_norm=args.max_grad_norm,
        amp=args.amp,
        random_seed=args.training_seed,
        freeze_non_value_parameters=args.freeze_non_value_parameters,
        shaping_weights=shaping_weights_json,
    )
    if initial_training_result is not None:
        model_config = replace(
            initial_training_result.model_config,
            policy_id=args.policy_id or initial_training_result.model_config.policy_id,
            # Stamp the RESOLVED shaping (inherited, or explicitly re-targeted): the saved
            # checkpoint self-describes the targets this run actually trained under.
            reward_shaping=shaping_weights_json,
        )
        _require_mask_flags_agree_with_checkpoint(args, model_config)
        _require_schema_flag_agrees_with_checkpoint(args, model_config)
        print(
            f"category vocab (from initial checkpoint): {len(model_config.category_vocab):,} tokens + "
            f"{model_config.category_oov_buckets:,} oov -> embedding rows {model_config.categorical_vocab_size:,}",
            file=sys.stderr,
        )
    else:
        model_config_kwargs = dict(
            policy_id=args.policy_id or "entity-transformer",
            window_size=args.window_size,
            embedding_dim=args.embedding_dim,
            transformer_layers=args.layers,
            attention_heads=args.attention_heads,
            feedforward_dim=args.feedforward_dim,
            dropout=args.dropout,
            temporal_aggregator=args.temporal_aggregator,
            stats_block_enabled=not args.no_stats_block,
            exact_state_enabled=not args.no_exact_state,
            transition_token_budget=(
                TRANSITION_TOKEN_COUNT
                if args.transition_token_budget is None
                else args.transition_token_budget
            ),
            tier2_residuals=True if args.tier2_residuals is None else bool(args.tier2_residuals),
            # Asymmetric default vs tier2_residuals: absent flag resolves OFF (no current
            # training run consumes the investment column; a fresh train encodes it
            # byte-identically to the pre-investment encoder). Only meaningful under
            # v2.1/v2.2 — the encoder schema-gates the columns, so under v2 this is a no-op.
            tier2_investment=False if args.tier2_investment is None else bool(args.tier2_investment),
            reward_shaping=shaping_weights_json,
        )
        # Fresh train: --observation-schema SETS the stamped schema + widths (default
        # v2.1, the current default spec); v2.2 also needs the turn-merged vocabulary
        # families (the schema-derived vocab latch).
        schema_version = (
            observation_schema_version_from_choice(args.observation_schema)
            or OBSERVATION_SCHEMA_VERSION
        )
        schema_spec = observation_spec_for_schema(schema_version)
        model_config_kwargs.update(
            observation_schema_version=schema_version,
            categorical_feature_count=schema_spec.categorical_feature_count,
            numeric_feature_count=schema_spec.numeric_feature_count,
            token_count=schema_spec.token_count,
        )
        # The category vocabulary is the closed Gen 3 randbat universe (string->row), the same one
        # the env builds at encode time, so rows align deterministically. (The legacy training-data
        # vocab source is retired: observations now store rows, not collectible hash ids.)
        if args.showdown_root is None:
            raise ValueError("neural training requires --showdown-root for the Gen 3 randbat category vocabulary.")
        from .randbat_vocab import gen3_category_vocabulary

        category_vocab = gen3_category_vocabulary(
            args.showdown_root,
            oov_buckets=args.category_oov_buckets,
            include_turn_merged=schema_version == OBSERVATION_SCHEMA_VERSION_V2_2,
        )
        model_config = TransformerPolicyConfig.compact_category(
            category_vocab=category_vocab.tokens,
            category_oov_buckets=category_vocab.oov_buckets,
            **model_config_kwargs,
        )
        print(
            f"category vocab (randbat-dex universe): {len(category_vocab.tokens):,} tokens + {args.category_oov_buckets:,} oov "
            f"-> embedding rows {model_config.categorical_vocab_size:,}",
            file=sys.stderr,
        )
    if training_config.window_size != model_config.window_size:
        raise ValueError("--window-size must match the model config window_size.")
    training_data_paths = tuple(args.data) + refutation_cache_paths
    _require_cache_masks_match_model_config(training_data_paths, model_config)
    _require_cache_observation_schema_matches(training_data_paths, model_config)
    _require_cache_shaping_matches_training_config(training_data_paths, shaping_weights_json)
    if args.value_selection_data and training_config.objective != "value-only":
        print(
            "warning: --value-selection-data selects by held-out value calibration, not policy quality; "
            "prefer objective=value-only for value-head calibration runs.",
            file=sys.stderr,
        )
    value_selection_payload = None
    train_started = time.perf_counter()
    if args.value_selection_data:
        model, result, value_selection_payload = _train_with_value_selection(
            paths=args.data,
            model_config=model_config,
            training_config=training_config,
            initial_model=initial_model,
            selection_paths=args.value_selection_data,
            selection_metric=args.value_selection_metric,
            batch_size=args.value_calibration_batch_size,
            bins=args.value_calibration_bins,
            consumed_cache_callback=cache_lifecycle.consumed_cache_callback,
            auxiliary_paths=refutation_cache_paths or None,
            auxiliary_max_fraction=args.refutation_max_fraction if refutation_cache_paths else 0.0,
        )
    else:
        train_kwargs: dict[str, object] = {}
        if cache_lifecycle.consumed_cache_callback is not None:
            train_kwargs["consumed_cache_callback"] = cache_lifecycle.consumed_cache_callback
        if refutation_cache_paths:
            train_kwargs["auxiliary_paths"] = refutation_cache_paths
            train_kwargs["auxiliary_max_fraction"] = args.refutation_max_fraction
        if distributed_context_value.enabled:
            train_kwargs["distributed_context_override"] = distributed_context_value
        model, result = train_transformer_policy(
            args.data,
            model_config=model_config,
            training_config=training_config,
            initial_model=initial_model,
            **train_kwargs,
        )
    train_elapsed_seconds = time.perf_counter() - train_started
    if not is_primary_rank:
        # Only rank 0 may create/delete artifacts. Wait until its atomic
        # checkpoint/summary work is finished, then tear down NCCL/Gloo cleanly
        # instead of relying on process exit to release collective resources.
        require_torch().distributed.barrier()
        require_torch().distributed.destroy_process_group()
        return 0
    provenance_hashes = distinct_belief_set_source_hashes(training_data_paths)
    if len(provenance_hashes) == 1 and provenance_hashes[0] is not None:
        result = replace(result, belief_set_source_hash=provenance_hashes[0])
    elif len(provenance_hashes) > 1:
        print(
            "warning: training data mixes belief set-source provenance "
            f"({', '.join(str(h)[:12] for h in provenance_hashes)}); checkpoint records none.",
            file=sys.stderr,
        )
    save_transformer_checkpoint(args.out, model, result=result)
    for metrics in result.epochs:
        line = (
            f"epoch={metrics.epoch} examples={metrics.examples} "
            f"loss={metrics.loss:.6f} policy_loss={metrics.policy_loss:.6f} "
            f"policy_accuracy={metrics.policy_accuracy:.4f}"
        )
        if metrics.value_loss is not None:
            line += f" value_loss={metrics.value_loss:.6f}"
        if metrics.value_ranking_loss is not None:
            line += (
                f" value_ranking_loss={metrics.value_ranking_loss:.6f} "
                f"value_ranking_pairs={metrics.value_ranking_pairs}"
            )
        if metrics.opponent_loss is not None:
            line += (
                f" opponent_loss={metrics.opponent_loss:.6f} "
                f"opponent_accuracy={metrics.opponent_accuracy:.4f}"
            )
        if metrics.elapsed_seconds is not None:
            line += (
                f" elapsed_seconds={metrics.elapsed_seconds:.3f}"
                f" batches={metrics.batches}"
            )
            if metrics.examples_per_second is not None:
                line += f" examples_per_second={metrics.examples_per_second:.1f}"
            if metrics.model_forward_elapsed_seconds is not None:
                line += f" model_forward_seconds={metrics.model_forward_elapsed_seconds:.3f}"
            if metrics.backward_elapsed_seconds is not None:
                line += f" backward_seconds={metrics.backward_elapsed_seconds:.3f}"
            if metrics.batch_load_elapsed_seconds is not None:
                line += f" batch_load_seconds={metrics.batch_load_elapsed_seconds:.3f}"
        print(line)
    print(f"checkpoint: {args.out}")
    if value_selection_payload is not None:
        if args.value_selection_out is not None:
            _write_json(args.value_selection_out, value_selection_payload)
            print(f"value_selection: {args.value_selection_out}")
        else:
            print(
                "value_selection: "
                f"selected_epoch={value_selection_payload['selected_epoch']} "
                f"metric={value_selection_payload['metric']} "
                f"value={value_selection_payload['selected_metric_value']:.6f}"
            )
    if args.value_calibration_data:
        value_calibration = evaluate_value_calibration(
            model=model,
            training_result=result,
            paths=args.value_calibration_data,
            batch_size=args.value_calibration_batch_size,
            bins=args.value_calibration_bins,
            device=resolve_torch_device(args.device),
        )
        payload = {
            "paths": [str(path) for path in args.value_calibration_data],
            "batch_size": args.value_calibration_batch_size,
            "bins": args.value_calibration_bins,
            "report": value_calibration.to_dict(),
        }
        if args.value_calibration_out is not None:
            _write_json(args.value_calibration_out, payload)
            print(f"value_calibration: {args.value_calibration_out}")
        else:
            print("")
            print_value_calibration_report(value_calibration)
    cache_lifecycle.finalize_after_checkpoint()
    if args.summary_out is not None:
        payload = _train_summary_payload(
            args=args,
            model_config=model_config,
            training_config=training_config,
            result=result,
            cache_lifecycle=cache_lifecycle,
            command_started_at=command_started_at,
            elapsed_seconds=time.perf_counter() - command_started,
            train_elapsed_seconds=train_elapsed_seconds,
            input_data_bytes=input_data_bytes,
            refutation_cache_bytes=refutation_cache_bytes,
            value_selection_payload=value_selection_payload,
        )
        _write_json(args.summary_out, payload)
        print(f"train_summary: {args.summary_out}")
    if distributed_context_value.enabled:
        require_torch().distributed.barrier()
        require_torch().distributed.destroy_process_group()
    return 0


def _train_summary_payload(
    *,
    args: argparse.Namespace,
    model_config: TransformerPolicyConfig,
    training_config: TransformerTrainingConfig,
    result: TransformerTrainingResult,
    cache_lifecycle: "_TrainingCacheLifecycle",
    command_started_at: datetime,
    elapsed_seconds: float,
    train_elapsed_seconds: float,
    input_data_bytes: int | None,
    refutation_cache_bytes: int | None,
    value_selection_payload: Mapping[str, Any] | None,
) -> dict[str, Any]:
    checkpoint_bytes = args.out.stat().st_size if args.out.exists() else None
    final_metrics = result.final_metrics
    return {
        "schema_version": NEURAL_TRAIN_SUMMARY_SCHEMA_VERSION,
        "source": collect_source_metadata(),
        "started_at": command_started_at.isoformat().replace("+00:00", "Z"),
        "completed_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "elapsed_seconds": elapsed_seconds,
        "train_elapsed_seconds": train_elapsed_seconds,
        "data_paths": [str(path) for path in args.data],
        "input_data_bytes": input_data_bytes,
        "refutation_cache_bytes": refutation_cache_bytes,
        "checkpoint_path": str(args.out),
        "checkpoint_bytes": checkpoint_bytes,
        "model": {
            "policy_id": model_config.policy_id,
            "window_size": model_config.window_size,
            "embedding_dim": model_config.embedding_dim,
            "transformer_layers": model_config.transformer_layers,
            "attention_heads": model_config.attention_heads,
            "feedforward_dim": model_config.feedforward_dim,
            "dropout": model_config.dropout,
            "temporal_aggregator": model_config.temporal_aggregator,
            "categorical_vocab_size": model_config.categorical_vocab_size,
            "category_oov_buckets": model_config.category_oov_buckets,
        },
        "training_config": training_config.to_dict(),
        "distributed_training": distributed_training_context().to_dict(base_seed=training_config.random_seed),
        "epochs": [metrics.to_dict() for metrics in result.epochs],
        "final_metrics": final_metrics.to_dict(),
        "value_selection": value_selection_payload,
        "refutation_training": {
            "enabled": bool(args.refutation_cache),
            "paths": [str(path) for path in (args.refutation_cache or ())],
            "max_fraction": args.refutation_max_fraction if args.refutation_cache else None,
            "target_mode": args.refutation_target_mode if args.refutation_cache else None,
        },
        "training_cache": cache_lifecycle.to_summary(),
    }


def _input_data_paths_byte_size(paths: Sequence[Path]) -> int | None:
    total = 0
    for path in paths:
        try:
            if is_training_cache_path(path):
                total += training_cache_paths_byte_size([path])
            elif path.is_file() or path.is_symlink():
                total += path.stat().st_size
            elif path.is_dir():
                for child in path.rglob("*"):
                    if child.is_file() or child.is_symlink():
                        total += child.stat().st_size
            else:
                return None
        except OSError:
            return None
    return total


@dataclass
class _TrainingCacheLifecycle:
    delete_after_checkpoint: bool = False
    cache_root: Path | None = None
    cache_footprint_bytes: int | None = None
    cache_footprint_limit_bytes: int | None = None
    consumed_paths: list[Path] = field(default_factory=list)
    deleted_paths: list[Path] = field(default_factory=list)
    deleted_bytes: int = 0

    @property
    def consumed_cache_callback(self) -> Callable[[Path], None] | None:
        return self._record_consumed_cache if self.delete_after_checkpoint else None

    def _record_consumed_cache(self, path: Path) -> None:
        resolved = Path(path)
        if resolved not in self.consumed_paths:
            self.consumed_paths.append(resolved)

    def finalize_after_checkpoint(self) -> None:
        if not self.delete_after_checkpoint:
            return
        for path in self.consumed_paths:
            byte_size = training_cache_paths_byte_size([path])
            delete_training_cache_path(path)
            self.deleted_paths.append(path)
            self.deleted_bytes += byte_size
            print(f"deleted_training_cache: {path} bytes={byte_size}")

    def to_summary(self) -> dict[str, Any]:
        return {
            "root": str(self.cache_root) if self.cache_root is not None else None,
            "footprint_bytes": self.cache_footprint_bytes,
            "footprint_limit_bytes": self.cache_footprint_limit_bytes,
            "delete_after_checkpoint": self.delete_after_checkpoint,
            "consumed_paths": [str(path) for path in self.consumed_paths],
            "deleted_paths": [str(path) for path in self.deleted_paths],
            "deleted_bytes": self.deleted_bytes,
        }


def _training_cache_lifecycle(args: argparse.Namespace) -> _TrainingCacheLifecycle:
    cache_flags = tuple(is_training_cache_path(path) for path in args.data)
    if not any(cache_flags):
        return _TrainingCacheLifecycle()
    if not all(cache_flags):
        raise ValueError("training cache directories cannot be mixed with rollout JSONL paths.")

    max_bytes = _cache_gb_to_bytes(getattr(args, "max_cache_gb", None))
    cache_root = _common_cache_root(tuple(Path(path) for path in args.data))
    cache_bytes = training_cache_root_byte_size(cache_root)
    print(f"training_cache_root: {cache_root}")
    print(f"training_cache_footprint_bytes: {cache_bytes}")
    print(f"training_cache_footprint_limit_bytes: {max_bytes}")
    if cache_bytes > max_bytes:
        raise ValueError(
            f"training cache root footprint {cache_bytes} bytes exceeds --max-cache-gb "
            f"limit of {max_bytes} bytes."
        )

    delete_after_read = bool(getattr(args, "delete_cache_after_read", True))
    if not delete_after_read:
        return _TrainingCacheLifecycle(
            cache_root=cache_root,
            cache_footprint_bytes=cache_bytes,
            cache_footprint_limit_bytes=max_bytes,
        )
    deleted_paths = tuple(Path(path) for path in args.data)
    for name in ("value_calibration_data", "value_selection_data"):
        for path in tuple(getattr(args, name, None) or ()):
            if any(_paths_overlap(path, deleted_path) for deleted_path in deleted_paths):
                flag = "--value-calibration-data" if name == "value_calibration_data" else "--value-selection-data"
                raise ValueError(f"--delete-cache-after-read cannot be used when {flag} overlaps training cache data.")
    return _TrainingCacheLifecycle(
        delete_after_checkpoint=True,
        cache_root=cache_root,
        cache_footprint_bytes=cache_bytes,
        cache_footprint_limit_bytes=max_bytes,
    )


def _training_cache_lifecycle_callback(args: argparse.Namespace) -> Callable[[Path], None] | None:
    return _training_cache_lifecycle(args).consumed_cache_callback


def _validate_refutation_cache_args(args: argparse.Namespace) -> tuple[Path, ...]:
    paths = tuple(Path(path) for path in (getattr(args, "refutation_cache", None) or ()))
    if not paths:
        return ()
    max_fraction = float(getattr(args, "refutation_max_fraction", 0.0))
    if not 0.0 < max_fraction <= 0.2:
        raise ValueError("--refutation-max-fraction must be greater than 0 and at most 0.2.")
    target_mode = getattr(args, "refutation_target_mode", "policy-value")
    objective = getattr(args, "objective", "behavior-cloning")
    for path in paths:
        if not is_training_cache_path(path):
            raise ValueError(f"--refutation-cache path is not a training-cache directory: {path}")
        if any(_paths_overlap(path, primary_path) for primary_path in getattr(args, "data", ())):
            raise ValueError("--refutation-cache paths must not overlap --data paths.")
        cache_target_mode, compatible_objectives = _refutation_cache_training_contract(path)
        if cache_target_mode != target_mode:
            raise ValueError(
                f"--refutation-cache {path} was built with target_mode={cache_target_mode!r} "
                f"but --refutation-target-mode is {target_mode!r}."
            )
        if objective not in compatible_objectives:
            raise ValueError(
                f"--refutation-cache {path} target_mode={cache_target_mode!r} is compatible with "
                f"{', '.join(compatible_objectives)}, not --objective {objective!r}."
            )
    if len({path.expanduser().resolve(strict=False) for path in paths}) != len(paths):
        raise ValueError("--refutation-cache contains duplicate paths.")
    return paths


def _refutation_cache_training_contract(path: Path) -> tuple[str, tuple[str, ...]]:
    metadata_path = path / "metadata.json"
    try:
        payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"--refutation-cache {path} has unreadable metadata.json.") from exc
    refutation_training = payload.get("refutation_training")
    if not isinstance(refutation_training, dict):
        raise ValueError(f"--refutation-cache {path} is missing refutation_training metadata.")
    target_mode = refutation_training.get("target_mode")
    if target_mode not in REFUTATION_TRAINING_COMPATIBLE_OBJECTIVES:
        raise ValueError(f"--refutation-cache {path} records unsupported target_mode {target_mode!r}.")
    compatible = refutation_training.get("compatible_objectives")
    if not isinstance(compatible, list):
        raise ValueError(f"--refutation-cache {path} records invalid compatible_objectives.")
    compatible_objectives = tuple(str(value) for value in compatible)
    expected_compatible_objectives = REFUTATION_TRAINING_COMPATIBLE_OBJECTIVES[str(target_mode)]
    if compatible_objectives != expected_compatible_objectives:
        raise ValueError(
            f"--refutation-cache {path} records compatible_objectives {compatible_objectives!r} "
            f"but target_mode={target_mode!r} requires {expected_compatible_objectives!r}."
        )
    return str(target_mode), expected_compatible_objectives


def _cache_gb_to_bytes(value: float | None) -> int:
    resolved = MAX_ACTIVE_TRAINING_CACHE_GB if value is None else value
    if resolved <= 0:
        raise ValueError("--max-cache-gb must be positive.")
    if resolved > MAX_ACTIVE_TRAINING_CACHE_GB:
        raise ValueError(f"--max-cache-gb cannot exceed {MAX_ACTIVE_TRAINING_CACHE_GB:g}.")
    return int(resolved * 1024 * 1024 * 1024)


def _common_cache_root(paths: tuple[Path, ...]) -> Path:
    if not paths:
        raise ValueError("at least one training cache path is required.")
    resolved_paths = tuple(path.expanduser().resolve(strict=False) for path in paths)
    if len(resolved_paths) == 1:
        return resolved_paths[0].parent
    common = Path(str(resolved_paths[0]))
    for path in resolved_paths[1:]:
        while common != common.parent and common not in (path, *path.parents):
            common = common.parent
    if any(common == path for path in resolved_paths):
        return common.parent
    return common


def _paths_overlap(left: Path, right: Path) -> bool:
    resolved_left = Path(left).expanduser().resolve(strict=False)
    resolved_right = Path(right).expanduser().resolve(strict=False)
    return (
        resolved_left == resolved_right
        or resolved_left in resolved_right.parents
        or resolved_right in resolved_left.parents
    )


def _add_training_dataset_arguments(parser: argparse.ArgumentParser) -> None:
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
            "preset (wse-arm1), inline JSON, or @/path/to.json. Default off (byte-identical "
            "unshaped caches). The shaping gamma is --discount."
        ),
    )


def _training_dataset_config_from_args(args: argparse.Namespace) -> TrajectoryDatasetConfig:
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


def _train_with_value_selection(
    *,
    paths: list[Path],
    model_config: TransformerPolicyConfig,
    training_config: TransformerTrainingConfig,
    initial_model: object | None,
    selection_paths: list[Path],
    selection_metric: str,
    batch_size: int,
    bins: int,
    consumed_cache_callback: Callable[[Path], None] | None = None,
    auxiliary_paths: Sequence[Path] | None = None,
    auxiliary_max_fraction: float = 0.0,
) -> tuple[object, object, dict[str, object]]:
    if batch_size <= 0:
        raise ValueError("value selection batch_size must be positive.")
    if bins <= 0:
        raise ValueError("value selection bins must be positive.")
    value_selection_metric_direction(selection_metric)

    selection_reports = []
    best_state = None
    best_epoch = None
    best_metric_value = None
    best_score = None
    device = resolve_torch_device(training_config.device)

    def evaluate_epoch(model: object, epoch_result: TransformerTrainingResult) -> None:
        nonlocal best_epoch, best_metric_value, best_score, best_state
        epoch_metric = epoch_result.final_metrics
        report = evaluate_value_calibration(
            model=model,
            training_result=epoch_result,
            paths=selection_paths,
            batch_size=batch_size,
            bins=bins,
            device=device,
        )
        try:
            metric_value = value_selection_metric_value(report, selection_metric)
            score = value_selection_score(metric_value, selection_metric)
            metric_unavailable_reason = None
        except ValueError as exc:
            if selection_metric != "pearson_correlation":
                raise
            metric_value = None
            score = None
            metric_unavailable_reason = str(exc)
        epoch = epoch_metric.epoch
        selection_entry: dict[str, object] = {
            "epoch": epoch,
            "metric_value": metric_value,
            "training_metrics": epoch_metric.to_dict(),
            "report": report.to_dict(),
        }
        if metric_unavailable_reason is not None:
            selection_entry["metric_unavailable_reason"] = metric_unavailable_reason
        selection_reports.append(selection_entry)
        if score is None:
            return
        if best_score is None or score > best_score:
            best_score = score
            best_metric_value = metric_value
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())

    train_kwargs: dict[str, object] = {}
    if consumed_cache_callback is not None:
        train_kwargs["consumed_cache_callback"] = consumed_cache_callback
    if auxiliary_paths is not None:
        train_kwargs["auxiliary_paths"] = auxiliary_paths
        train_kwargs["auxiliary_max_fraction"] = auxiliary_max_fraction
    model, full_result = train_transformer_policy(
        paths,
        model_config=model_config,
        training_config=training_config,
        initial_model=initial_model,
        epoch_callback=evaluate_epoch,
        **train_kwargs,
    )
    if best_state is None or best_epoch is None or best_metric_value is None:
        raise ValueError("value selection produced no selectable epoch reports.")
    model.load_state_dict(best_state)
    selected_result = TransformerTrainingResult(
        model_config=model_config,
        training_config=replace(training_config, epochs=best_epoch),
        epochs=tuple(full_result.epochs[:best_epoch]),
    )
    payload = {
        "paths": [str(path) for path in selection_paths],
        "batch_size": batch_size,
        "bins": bins,
        "metric": selection_metric,
        "metric_direction": value_selection_metric_direction(selection_metric),
        "selected_epoch": best_epoch,
        "selected_metric_value": best_metric_value,
        "epochs": selection_reports,
    }
    return model, selected_result, payload

def _benchmark(args: argparse.Namespace) -> int:
    # Benchmark loads arbitrary checkpoints; the env builds the vocabulary from showdown_root
    # (the closed-universe default), which matches any checkpoint trained on the same root.
    env_config = LocalShowdownConfig(
        showdown_root=args.showdown_root,
        node_binary=args.node_binary,
    )
    rollout_config = RolloutConfig(
        max_decision_rounds=args.max_decision_rounds,
        format_id=args.format_id,
    )
    reference_specs = tuple(args.benchmark_reference_policy or ())
    if not args.allow_legacy_checkpoints:
        _require_current_family_benchmark_checkpoints(
            args.checkpoint,
            reference_specs=reference_specs,
            context="neural benchmark",
        )
    deterministic = not bool(args.sample)
    checkpoint_policy = _policy_from_checkpoint(
        args.checkpoint,
        deterministic=deterministic,
        exploration_epsilon=args.epsilon,
        sampling_temperature=args.temperature,
        device=args.device,
    )
    policy_id = _policy_id_alias(args.policy_id, label="--policy-id") if args.policy_id else str(checkpoint_policy.policy_id)
    if args.policy_id:
        checkpoint_policy = _PolicyIdAlias(checkpoint_policy, policy_id=policy_id)
    policy_showdown_root = env_config.resolved_showdown_root()
    reference_policy_ids = tuple(args.benchmark_reference_policy_id or ())
    if reference_policy_ids and len(reference_policy_ids) != len(reference_specs):
        raise ValueError(
            "--benchmark-reference-policy-id must be supplied the same number of times as "
            "--benchmark-reference-policy."
        )
    matchups = [
        BenchmarkMatchup(
            f"{policy_id} vs random-legal",
            checkpoint_policy,
            RandomLegalPolicy(),
        ),
        BenchmarkMatchup(
            f"random-legal vs {policy_id}",
            RandomLegalPolicy(),
            checkpoint_policy,
        ),
        BenchmarkMatchup(
            f"{policy_id} vs simple-legal",
            checkpoint_policy,
            SimpleLegalPolicy(),
        ),
        BenchmarkMatchup(
            f"simple-legal vs {policy_id}",
            SimpleLegalPolicy(),
            checkpoint_policy,
        ),
    ]
    covered_ids = {str(policy_id), "random-legal", "simple-legal"}
    for index, reference_spec in enumerate(reference_specs):
        resolved_reference_spec = policy_spec_with_showdown_root(reference_spec, policy_showdown_root)
        reference_policy = _policy_from_spec_for_evaluation(resolved_reference_spec, device=args.device)
        reference_id = (
            _policy_id_alias(reference_policy_ids[index], label="--benchmark-reference-policy-id")
            if reference_policy_ids
            else str(reference_policy.policy_id)
        )
        if reference_id in covered_ids:
            continue
        if reference_policy_ids:
            reference_policy = _PolicyIdAlias(reference_policy, policy_id=reference_id)
        covered_ids.add(reference_id)
        matchups.append(BenchmarkMatchup(f"{policy_id} vs {reference_id}", checkpoint_policy, reference_policy))
        reverse_reference_policy = _policy_from_spec_for_evaluation(resolved_reference_spec, device=args.device)
        if reference_policy_ids:
            reverse_reference_policy = _PolicyIdAlias(reverse_reference_policy, policy_id=reference_id)
        matchups.append(
            BenchmarkMatchup(
                f"{reference_id} vs {policy_id}",
                reverse_reference_policy,
                checkpoint_policy,
            )
        )
    env_config = _env_config_with_matchup_masks(env_config, matchups, context="neural benchmark")
    report = benchmark_rollouts(
        games=args.games,
        env_factory=lambda: LocalShowdownEnv(env_config),
        rollout_config=rollout_config,
        seed_start=args.seed_start,
        matchups=tuple(matchups),
    )
    payload = report.to_dict()
    if args.summary_out is not None:
        _write_json(args.summary_out, payload)
        print(f"benchmark_summary: {args.summary_out}", file=sys.stderr)
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print_benchmark_report(report)
    return 0


def _require_current_family_benchmark_checkpoints(
    checkpoint_path: Path,
    *,
    reference_specs: Iterable[str],
    context: str,
) -> None:
    require_current_family_checkpoint_paths((checkpoint_path,), context=context)
    checkpoint_reference_specs = tuple(
        spec for spec in reference_specs if _is_checkpoint_policy_spec(spec)
    )
    if checkpoint_reference_specs:
        try:
            current_family_checkpoint_policy_specs(checkpoint_reference_specs, legacy_mode="reject")
        except ValueError as exc:
            raise ValueError(
                f"{context} references require current-family v2+ checkpoints. Legacy no-belief/pre-v2 "
                "checkpoints are historical baselines, not current eval targets. Pass "
                "--allow-legacy-checkpoints only when intentionally reproducing archived historical "
                f"diagnostics. Details: {exc}"
            ) from exc


def _is_checkpoint_policy_spec(policy_spec: str) -> bool:
    body = str(policy_spec).strip().partition("?")[0].strip().lower()
    return any(body.startswith(prefix) for prefix in CHECKPOINT_POLICY_SPEC_PREFIXES)


class _StoreExplicitArgument(argparse.Action):
    """Retain whether an option was supplied instead of only resolving its value."""

    def __call__(
        self,
        parser: argparse.ArgumentParser,
        namespace: argparse.Namespace,
        values: object,
        option_string: str | None = None,
    ) -> None:
        setattr(namespace, self.dest, values)
        setattr(namespace, f"{self.dest}_explicit", True)


@dataclass
class _PolicyIdAlias:
    policy: Policy
    policy_id: str

    def __getattr__(self, name: str) -> Any:
        return getattr(self.policy, name)

    def reset(self) -> None:
        reset = getattr(self.policy, "reset", None)
        if callable(reset):
            reset()

    def select_action(
        self,
        observation,
        *,
        rng,
    ) -> PolicyDecision:
        return replace(self.policy.select_action(observation, rng=rng), policy_id=self.policy_id)

    def select_action_with_context(
        self,
        context: PolicyContext,
        *,
        rng,
    ) -> PolicyDecision:
        contextual_selector = getattr(self.policy, "select_action_with_context", None)
        if callable(contextual_selector):
            decision = contextual_selector(context, rng=rng)
        else:
            decision = self.policy.select_action(context.observation, rng=rng)
        return replace(decision, policy_id=self.policy_id)


def _policy_id_alias(value: str, *, label: str) -> str:
    alias = str(value).strip()
    if not alias:
        raise ValueError(f"{label} must be non-empty when supplied.")
    return alias


def _root_visit_budget_selector(
    args: argparse.Namespace,
) -> FixedExtraVisitBudgetSelector | EntropyMarginVisitBudgetSelector | None:
    root_extra_visits = getattr(args, "root_extra_visits", None)
    if root_extra_visits is not None:
        if root_extra_visits < 0:
            raise ValueError("root extra visits must be non-negative.")
        if (
            args.adaptive_root_contested_extra_visits is not None
            or args.adaptive_root_policy_entropy_threshold is not None
            or args.adaptive_root_value_margin_threshold is not None
            or args.adaptive_root_uncontested_extra_visits != 0
        ):
            raise ValueError("root extra visits cannot be combined with adaptive root budgeting.")
        return FixedExtraVisitBudgetSelector(extra_visits=root_extra_visits)

    configured = args.adaptive_root_contested_extra_visits is not None
    threshold_configured = (
        args.adaptive_root_policy_entropy_threshold is not None
        or args.adaptive_root_value_margin_threshold is not None
    )
    if not configured:
        if threshold_configured or args.adaptive_root_uncontested_extra_visits != 0:
            raise ValueError(
                "adaptive root thresholds and uncontested extra visits require "
                "--adaptive-root-contested-extra-visits."
            )
        return None
    return EntropyMarginVisitBudgetSelector(
        contested_extra_visits=args.adaptive_root_contested_extra_visits,
        uncontested_extra_visits=args.adaptive_root_uncontested_extra_visits,
        minimum_policy_entropy=args.adaptive_root_policy_entropy_threshold,
        maximum_value_margin=args.adaptive_root_value_margin_threshold,
    )


def _adaptive_root_visit_budget_selector(args: argparse.Namespace) -> EntropyMarginVisitBudgetSelector | None:
    """Backward-compatible helper for callers that only need the adaptive selector."""

    selector = _root_visit_budget_selector(args)
    return selector if isinstance(selector, EntropyMarginVisitBudgetSelector) else None


def _root_puct_play_benchmark(args: argparse.Namespace) -> int:
    require_torch()
    if not args.allow_legacy_checkpoints:
        _require_current_family_benchmark_checkpoints(
            args.checkpoint,
            reference_specs=tuple(args.opponent_policy or ()),
            context="root-puct play benchmark",
        )
    if args.root_opponent_action_scenarios <= 0:
        raise ValueError("root opponent action scenarios must be positive.")
    root_opponent_action_candidate_scenarios = (
        args.root_opponent_action_scenarios
        if args.root_opponent_action_candidate_scenarios is None
        else args.root_opponent_action_candidate_scenarios
    )
    if root_opponent_action_candidate_scenarios <= 0:
        raise ValueError("root opponent action candidate scenarios must be positive.")
    if root_opponent_action_candidate_scenarios < args.root_opponent_action_scenarios:
        raise ValueError(
            "root opponent action candidate scenarios must be greater than or equal to "
            "root opponent action scenarios."
        )
    if args.root_opponent_action_policy == "benchmark" and args.root_opponent_action_scenarios != 1:
        raise ValueError(
            "root opponent action scenarios above one require --root-opponent-action-policy checkpoint."
        )
    if args.root_opponent_action_policy == "benchmark" and root_opponent_action_candidate_scenarios != 1:
        raise ValueError(
            "root opponent action candidate scenarios above one require "
            "--root-opponent-action-policy checkpoint."
        )
    root_visit_budget_selector = _root_visit_budget_selector(args)
    if args.root_time_budget_ms is not None:
        if args.root_time_budget_ms <= 0:
            raise ValueError("root time budget must be positive when set.")
        if root_visit_budget_selector is not None:
            raise ValueError("root time budget cannot be combined with fixed or adaptive root budgeting.")
        if args.root_visit_budget_explicit:
            raise ValueError("root time budget cannot be combined with an explicit root visit budget.")
    env_config = LocalShowdownConfig(
        showdown_root=args.showdown_root,
        node_binary=args.node_binary,
    )
    if args.belief_start_overrides:
        if args.format_id != "gen3randombattle":
            raise ValueError("--belief-start-overrides currently supports only gen3randombattle.")
        if args.belief_world_sample_cap <= 0:
            raise ValueError("--belief-world-sample-cap must be positive.")
        if args.belief_start_override_attempts <= 0:
            raise ValueError("--belief-start-override-attempts must be positive.")
        if args.belief_start_override_hp_fraction_tolerance < 0.0:
            raise ValueError("--belief-start-override-hp-fraction-tolerance must be non-negative.")
        # Latch masks through the same belief-enabled environment used for both live and branch play.
        env_config = replace(env_config, set_belief_source=True)
    policy_showdown_root = env_config.resolved_showdown_root()
    rollout_config = RolloutConfig(
        max_decision_rounds=args.max_decision_rounds,
        format_id=args.format_id,
        record_policy_timing=True,
        hide_opponent_legal_action_masks=True,
    )
    leaf_rollout_rounds_values = _root_puct_leaf_rollout_rounds_values(args)
    tag_leaf_policy_ids = args.leaf_rollout_rounds_sweep is not None
    model, result = load_transformer_checkpoint(args.checkpoint, map_location=args.device)
    value_checkpoint = args.value_checkpoint or args.checkpoint
    value_model, value_result = model, result
    value_leaf_provenance: Mapping[str, object] | None = None
    if args.value_checkpoint is not None:
        value_model, value_result = load_transformer_checkpoint(value_checkpoint, map_location=args.device)
        value_leaf_provenance = require_compatible_transformer_value_checkpoint(
            policy_checkpoint=args.checkpoint,
            policy_result=result,
            value_checkpoint=value_checkpoint,
            value_result=value_result,
        )
    # HIGH-1 latch: encode-time masks come from the checkpoint(s) observing through this env.
    env_config = _env_config_with_spec_masks(
        env_config,
        tuple(args.opponent_policy or ()),
        extra_model_configs=(result.model_config, value_result.model_config),
        context="root-puct play benchmark",
    )
    belief_start_override_planner = None
    if args.belief_start_overrides:
        # The planner consumes the same candidate-set source that produced the observation's public belief.
        belief_start_override_planner = gen3_randbat_belief_start_override_planner(
            load_gen3_randbat_source_cached(env_config.resolved_showdown_root()),
            world_sample_cap=args.belief_world_sample_cap,
        )
    raw_policy_id = str(result.model_config.policy_id)
    raw_policy_checkpoint = str(args.checkpoint.resolve(strict=False))
    # Explicit value leaves already prove the raw-file hash through the
    # compatibility latch. Preserve the old no-leaf benchmark seam without
    # forcing synthetic/unit-test checkpoint paths to exist just for reporting.
    raw_policy_checkpoint_sha256 = (
        str(value_leaf_provenance["policy_checkpoint_sha256"])
        if value_leaf_provenance is not None
        else None
    )

    root_dirichlet_enabled = args.root_dirichlet_alpha is not None
    root_puct_variants = (False, True) if root_dirichlet_enabled else (False,)

    def search_policy_id_for(leaf_rollout_rounds: int, *, dirichlet_enabled: bool) -> str:
        root_puct_id = f"{raw_policy_id}+root-puct"
        if tag_leaf_policy_ids:
            root_puct_id = f"{root_puct_id}-leaf{leaf_rollout_rounds}"
        if isinstance(root_visit_budget_selector, EntropyMarginVisitBudgetSelector):
            root_puct_id = f"{root_puct_id}+adaptive-budget"
        if dirichlet_enabled:
            root_puct_id = f"{root_puct_id}+dirichlet"
        return root_puct_id
    search_policy_ids = tuple(
        search_policy_id_for(leaf_rollout_rounds, dirichlet_enabled=dirichlet_enabled)
        for leaf_rollout_rounds in leaf_rollout_rounds_values
        for dirichlet_enabled in root_puct_variants
    )

    def root_search_config_for(leaf_rollout_rounds: int, *, dirichlet_enabled: bool) -> dict[str, object]:
        return {
            "max_decision_rounds": args.max_decision_rounds,
            "temperature": args.temperature,
            "cpuct": args.cpuct,
            "selection_mode": args.selection_mode,
            "root_prior_temperature": (
                args.temperature if args.root_prior_temperature is None else args.root_prior_temperature
            ),
            "minimum_value_improvement": args.min_value_improvement,
            "root_visit_budget": None if args.root_time_budget_ms is not None else args.root_visit_budget,
            "root_extra_visits": args.root_extra_visits,
            "adaptive_root_contested_extra_visits": args.adaptive_root_contested_extra_visits,
            "adaptive_root_uncontested_extra_visits": args.adaptive_root_uncontested_extra_visits,
            "adaptive_root_policy_entropy_threshold": args.adaptive_root_policy_entropy_threshold,
            "adaptive_root_value_margin_threshold": args.adaptive_root_value_margin_threshold,
            "root_time_budget_ms": args.root_time_budget_ms,
            "root_opponent_action_policy": args.root_opponent_action_policy,
            "root_opponent_action_scenarios": args.root_opponent_action_scenarios,
            "root_opponent_action_candidate_scenarios": root_opponent_action_candidate_scenarios,
            "leaf_rollout_rounds": leaf_rollout_rounds,
            "leaf_rollout_sampling": False,
            "leaf_rollout_opponent_policy": args.leaf_rollout_opponent_policy,
            "belief_start_overrides": args.belief_start_overrides,
            "belief_world_sample_cap": args.belief_world_sample_cap,
            "belief_start_override_attempts": args.belief_start_override_attempts,
            "belief_start_override_hp_fraction_tolerance": args.belief_start_override_hp_fraction_tolerance,
            "allow_search_fallback": not args.no_search_fallback,
            "root_dirichlet_alpha": args.root_dirichlet_alpha if dirichlet_enabled else None,
            "root_dirichlet_mix": args.root_dirichlet_mix if dirichlet_enabled else None,
            "root_dirichlet_seed": args.root_dirichlet_seed if dirichlet_enabled else None,
        }

    root_search_policy_configs = {
        search_policy_id_for(leaf_rollout_rounds, dirichlet_enabled=dirichlet_enabled): root_search_config_for(
            leaf_rollout_rounds,
            dirichlet_enabled=dirichlet_enabled,
        )
        for leaf_rollout_rounds in leaf_rollout_rounds_values
        for dirichlet_enabled in root_puct_variants
    }

    def make_raw_policy(policy_id: str | None = None) -> TransformerSoftmaxPolicy:
        return TransformerSoftmaxPolicy(
            model=model,
            result=result,
            deterministic=True,
            sampling_temperature=args.temperature,
            device=args.device,
            policy_id=policy_id,
            checkpoint_path=raw_policy_checkpoint,
            weights_sha256=raw_policy_checkpoint_sha256,
        )

    def make_leaf_rollout_policy(
        *,
        deterministic_search_policy_id: str,
        search_player_id: str,
        benchmark_opponent_policy: Policy | None,
        player_id: str,
    ) -> Policy:
        if benchmark_opponent_policy is not None and player_id != search_player_id:
            return benchmark_opponent_policy
        return make_raw_policy(policy_id=f"{deterministic_search_policy_id}-leaf-{player_id}")

    def value_fn(history):
        return evaluate_transformer_observation_value(
            model=value_model,
            result=value_result,
            observations=history,
            device=args.device,
        )

    def prior_fn(history):
        return evaluate_transformer_action_priors(
            model=model,
            result=result,
            observations=history,
            temperature=1.0,
            device=args.device,
        )

    def opponent_prior_fn(history):
        return evaluate_transformer_opponent_action_priors(
            model=model,
            result=result,
            observations=history,
            temperature=args.temperature,
            device=args.device,
        )

    def make_search_policy(
        *,
        search_policy_id: str,
        leaf_rollout_rounds: int,
        search_player_id: str,
        benchmark_opponent_spec: str,
        dirichlet_enabled: bool,
    ) -> RootPUCTSearchPolicy:
        root_opponent_player_id = "p2" if search_player_id == "p1" else "p1"
        opponent_action_scenario_planner = None
        if args.root_opponent_action_policy == "benchmark":
            opponent_action_planner = policy_opponent_action_planner(
                {root_opponent_player_id: policy_from_spec(benchmark_opponent_spec)},
                planner_id="benchmark",
            )
        else:
            opponent_action_planner = greedy_opponent_action_planner(opponent_prior_fn)
            if root_opponent_action_candidate_scenarios > 1:
                opponent_action_scenario_planner = prior_top_k_opponent_action_scenario_planner(
                    opponent_prior_fn,
                    scenario_count=root_opponent_action_candidate_scenarios,
                )
        leaf_rollout_policy_factory = None
        leaf_rollout_metadata: Mapping[str, object] = {}
        if leaf_rollout_rounds:
            benchmark_opponent_policy = (
                policy_from_spec(benchmark_opponent_spec)
                if args.leaf_rollout_opponent_policy == "benchmark"
                else None
            )
            leaf_rollout_policy_factory = lambda player_id: make_leaf_rollout_policy(
                deterministic_search_policy_id=search_policy_id_for(
                    leaf_rollout_rounds,
                    dirichlet_enabled=False,
                ),
                search_player_id=search_player_id,
                benchmark_opponent_policy=benchmark_opponent_policy,
                player_id=player_id,
            )
            leaf_rollout_metadata = {
                "root_puct_leaf_rollout_opponent_policy": args.leaf_rollout_opponent_policy,
            }
        return RootPUCTSearchPolicy(
            env_factory=lambda: LocalShowdownEnv(env_config),
            rollout_config=rollout_config,
            value_fn=value_fn,
            prior_fn=prior_fn,
            opponent_action_planner=opponent_action_planner,
            opponent_action_scenario_planner=opponent_action_scenario_planner,
            fallback_policy=make_raw_policy(policy_id=f"{raw_policy_id}-fallback"),
            allow_fallback=not args.no_search_fallback,
            policy_id=search_policy_id,
            cpuct=args.cpuct,
            minimum_value_improvement=args.min_value_improvement,
            selection_mode=args.selection_mode,
            # Time-bounded comparisons must not inherit the legacy 16-visit cap.
            root_visit_budget=None if args.root_time_budget_ms is not None else args.root_visit_budget,
            root_visit_budget_selector=root_visit_budget_selector,
            root_time_budget_seconds=(
                None if args.root_time_budget_ms is None else args.root_time_budget_ms / 1000.0
            ),
            root_prior_temperature=(
                args.temperature if args.root_prior_temperature is None else args.root_prior_temperature
            ),
            root_dirichlet_alpha=(args.root_dirichlet_alpha if dirichlet_enabled else None),
            root_dirichlet_mix=args.root_dirichlet_mix,
            root_dirichlet_seed=args.root_dirichlet_seed,
            max_opponent_action_scenarios=args.root_opponent_action_scenarios,
            leaf_rollout_decision_rounds=leaf_rollout_rounds,
            leaf_rollout_policy_factory=leaf_rollout_policy_factory,
            start_override_planner=belief_start_override_planner,
            start_override_attempts=args.belief_start_override_attempts,
            start_override_samples_per_scenario=(None if belief_start_override_planner is not None else 1),
            start_override_hp_fraction_tolerance=args.belief_start_override_hp_fraction_tolerance,
            leaf_rollout_metadata=leaf_rollout_metadata,
        )

    opponent_specs = tuple(args.opponent_policy or ("random-legal", "simple-legal"))
    matchups: list[BenchmarkMatchup] = []
    for opponent_spec in opponent_specs:
        benchmark_opponent_spec = policy_spec_with_showdown_root(opponent_spec, policy_showdown_root)
        opponent_id = policy_from_spec(benchmark_opponent_spec).policy_id
        matchups.extend(
            (
                BenchmarkMatchup(
                    f"{raw_policy_id} vs {opponent_id}",
                    make_raw_policy(),
                    policy_from_spec(benchmark_opponent_spec),
                ),
                BenchmarkMatchup(
                    f"{opponent_id} vs {raw_policy_id}",
                    policy_from_spec(benchmark_opponent_spec),
                    make_raw_policy(),
                ),
            )
        )
        for leaf_rollout_rounds in leaf_rollout_rounds_values:
            for dirichlet_enabled in root_puct_variants:
                search_policy_id = search_policy_id_for(
                    leaf_rollout_rounds,
                    dirichlet_enabled=dirichlet_enabled,
                )
                matchups.extend(
                    (
                        BenchmarkMatchup(
                            f"{search_policy_id} vs {opponent_id}",
                            make_search_policy(
                                search_policy_id=search_policy_id,
                                leaf_rollout_rounds=leaf_rollout_rounds,
                                search_player_id="p1",
                                benchmark_opponent_spec=benchmark_opponent_spec,
                                dirichlet_enabled=dirichlet_enabled,
                            ),
                            policy_from_spec(benchmark_opponent_spec),
                        ),
                        BenchmarkMatchup(
                            f"{opponent_id} vs {search_policy_id}",
                            policy_from_spec(benchmark_opponent_spec),
                            make_search_policy(
                                search_policy_id=search_policy_id,
                                leaf_rollout_rounds=leaf_rollout_rounds,
                                search_player_id="p2",
                                benchmark_opponent_spec=benchmark_opponent_spec,
                                dirichlet_enabled=dirichlet_enabled,
                            ),
                        ),
                    )
                )

    report = benchmark_rollouts(
        games=args.games,
        env_factory=lambda: LocalShowdownEnv(env_config),
        rollout_config=rollout_config,
        seed_start=args.seed_start,
        matchups=tuple(matchups),
    )
    if belief_start_override_planner is not None:
        _require_belief_world_benchmark_coverage(report, search_policy_ids=search_policy_ids)
    payload = _root_puct_play_payload(
        report,
        raw_policy_id=raw_policy_id,
        search_policy_ids=search_policy_ids,
        root_time_budget_ms=args.root_time_budget_ms,
        root_search_policy_configs=root_search_policy_configs,
        root_dirichlet_config=(
            {
                "enabled": True,
                "alpha": args.root_dirichlet_alpha,
                "mix": args.root_dirichlet_mix,
                "base_seed": args.root_dirichlet_seed,
            }
            if root_dirichlet_enabled
            else None
        ),
        value_leaf_provenance=value_leaf_provenance,
        root_visit_budget_selector_config=(
            root_visit_budget_selector.to_dict() if root_visit_budget_selector is not None else None
        ),
    )
    if args.summary_out is not None:
        _write_json(args.summary_out, payload)
        print(f"root_puct_play_benchmark_summary: {args.summary_out}", file=sys.stderr)
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print_benchmark_report(report)
    return 0


def _root_puct_telemetry_report(args: argparse.Namespace) -> int:
    """Render the W1/W2 Root-PUCT telemetry readout without requiring torch."""

    try:
        payload = json.loads(args.input.read_text(encoding="utf-8"))
    except OSError as error:
        raise ValueError(f"could not read root-PUCT benchmark artifact {args.input}: {error}") from error
    except json.JSONDecodeError as error:
        raise ValueError(f"root-PUCT benchmark artifact is not valid JSON: {error}") from error
    if not isinstance(payload, Mapping):
        raise ValueError("root-PUCT benchmark artifact must contain a JSON object.")
    report = root_puct_benchmark_telemetry_report(
        payload,
        policy_ids=tuple(args.policy_id or ()),
    )
    if args.out is not None:
        _write_json(args.out, report)
        print(f"root_puct_telemetry_report: {args.out}", file=sys.stderr)
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        _print_root_puct_telemetry_report(report)
    return 0


def _print_root_puct_telemetry_report(report: Mapping[str, object]) -> None:
    """Keep the human readout compact enough for a 50-game diagnostic run."""

    policies = report.get("policies")
    if not isinstance(policies, Mapping):
        raise ValueError("Root-PUCT telemetry report is missing policies.")
    for policy_id, value in sorted(policies.items()):
        if not isinstance(value, Mapping):
            continue
        decisions = int(value.get("decisions", 0))
        searches = int(value.get("searches", 0))
        fallbacks = int(value.get("fallbacks", 0))
        search_rate = value.get("search_rate")
        fallback_rate = value.get("fallback_rate")
        visits = value.get("visits")
        visit_rate = visits.get("per_root_search_second") if isinstance(visits, Mapping) else None
        print(f"{policy_id}: {searches}/{decisions} searched, {fallbacks} fallbacks")
        print(
            "  rates: "
            f"search={_format_optional_fraction(search_rate)} "
            f"fallback={_format_optional_fraction(fallback_rate)} "
            f"visits/s={_format_optional_float(visit_rate)}"
        )
        categories = value.get("fallback_categories")
        if isinstance(categories, Mapping) and categories:
            print("  fallback categories: " + ", ".join(f"{key}={count}" for key, count in sorted(categories.items())))
        scenario_counts = value.get("scenario_counts")
        if isinstance(scenario_counts, Mapping) and scenario_counts:
            print("  scenarios: " + ", ".join(f"{key}={count}" for key, count in sorted(scenario_counts.items())))
        taxonomy = value.get("scenario_failure_taxonomy")
        if isinstance(taxonomy, Mapping) and taxonomy:
            print("  scenario taxonomy:")
            for key, counts in sorted(taxonomy.items()):
                if isinstance(counts, Mapping) and counts:
                    print("    " + str(key) + ": " + ", ".join(f"{name}={count}" for name, count in sorted(counts.items())))
        wall = value.get("root_search_wall_seconds")
        if isinstance(wall, Mapping):
            print(
                "  root wall: "
                f"mean={_format_optional_float(wall.get('mean'))}s "
                f"p50={_format_optional_float(wall.get('p50'))}s "
                f"p95={_format_optional_float(wall.get('p95'))}s"
            )


def _format_optional_fraction(value: object) -> str:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return f"{float(value):.1%}"
    return "n/a"


def _require_belief_world_benchmark_coverage(
    report: Any,
    *,
    search_policy_ids: Sequence[str],
) -> None:
    """Fail closed when an opt-in belief benchmark fell back before materializing worlds."""

    search_ids = set(search_policy_ids)
    missing: list[str] = []
    for result in report.matchups:
        if result.p1_policy_id not in search_ids and result.p2_policy_id not in search_ids:
            continue
        expected_seeds = set(range(result.seed_start, result.seed_start + result.metrics.games))
        observed_seeds = set((result.root_puct_belief_public_checksums_by_seed or {}).keys())
        if expected_seeds - observed_seeds:
            missing.append(
                f"{result.label}: missing belief-world checksum for seeds "
                f"{', '.join(str(seed) for seed in sorted(expected_seeds - observed_seeds))}"
            )
    if missing:
        raise RuntimeError(
            "belief-start-overrides requested but no public belief world was materialized for every "
            "searched game; refusing to report a fallback-contaminated benchmark: "
            + "; ".join(missing)
        )


def _root_puct_play_payload(
    report: Any,
    *,
    raw_policy_id: str,
    search_policy_ids: Sequence[str],
    root_time_budget_ms: int | None,
    root_search_policy_configs: Mapping[str, Mapping[str, object]],
    root_dirichlet_config: Mapping[str, object] | None = None,
    value_leaf_provenance: Mapping[str, object] | None,
    root_visit_budget_selector_config: Mapping[str, object] | None = None,
) -> dict[str, Any]:
    payload = dict(report.to_dict())
    comparisons = _root_puct_play_comparisons(
        payload,
        raw_policy_id=raw_policy_id,
        search_policy_ids=tuple(search_policy_ids),
    )
    if comparisons:
        payload["root_puct_play_comparisons"] = comparisons
    if root_dirichlet_config is not None:
        payload["root_dirichlet"] = dict(root_dirichlet_config)
    if root_time_budget_ms is not None:
        payload["root_time_budget_ms"] = root_time_budget_ms
    payload["root_puct_policy_configs"] = {
        policy_id: dict(config) for policy_id, config in sorted(root_search_policy_configs.items())
    }
    if len(root_search_policy_configs) == 1:
        payload["root_puct_config"] = dict(next(iter(root_search_policy_configs.values())))
    if value_leaf_provenance is not None:
        payload["value_leaf"] = dict(value_leaf_provenance)
    if root_visit_budget_selector_config is not None:
        if root_visit_budget_selector_config.get("selector_id") == "entropy-or-value-margin":
            # Keep the established adaptive key for existing consumers while fixed-extra
            # experiments use the selector-neutral record above.
            payload["adaptive_root_visit_budget"] = dict(root_visit_budget_selector_config)
        else:
            payload["root_visit_budget_selector"] = dict(root_visit_budget_selector_config)
    return payload


def _root_puct_play_comparisons(
    payload: Mapping[str, Any],
    *,
    raw_policy_id: str,
    search_policy_ids: Sequence[str],
) -> list[dict[str, Any]]:
    head_to_heads = tuple(_mapping(item) for item in _sequence(payload.get("head_to_heads", ())))
    opponents: list[str] = []
    for result in head_to_heads:
        first = str(result.get("first_policy_id", ""))
        second = str(result.get("second_policy_id", ""))
        if first == raw_policy_id and second and second not in opponents:
            opponents.append(second)
        elif second == raw_policy_id and first and first not in opponents:
            opponents.append(first)

    comparisons: list[dict[str, Any]] = []
    for opponent_policy_id in opponents:
        raw = _policy_pair_readout(
            head_to_heads,
            policy_id=raw_policy_id,
            opponent_policy_id=opponent_policy_id,
        )
        if raw is None:
            continue
        for search_policy_id in search_policy_ids:
            search = _policy_pair_readout(
                head_to_heads,
                policy_id=search_policy_id,
                opponent_policy_id=opponent_policy_id,
            )
            if search is None:
                continue
            comparisons.append(
                {
                    "opponent_policy_id": opponent_policy_id,
                    "raw_policy_id": raw_policy_id,
                    "search_policy_id": search_policy_id,
                    "raw": raw,
                    "search": search,
                    "search_minus_raw_win_rate": search["win_rate"] - raw["win_rate"],
                }
            )
    return comparisons


def _policy_pair_readout(
    head_to_heads: Sequence[Mapping[str, Any]],
    *,
    policy_id: str,
    opponent_policy_id: str,
) -> dict[str, Any] | None:
    for result in head_to_heads:
        first = str(result.get("first_policy_id", ""))
        second = str(result.get("second_policy_id", ""))
        if {first, second} != {policy_id, opponent_policy_id}:
            continue
        games = int(result.get("games", 0))
        if first == policy_id:
            wins = int(result.get("first_policy_wins", 0))
            win_rate = float(result.get("first_policy_win_rate", 0.0))
        else:
            wins = int(result.get("second_policy_wins", 0))
            win_rate = float(result.get("second_policy_win_rate", 0.0))
        return {
            "games": games,
            "wins": wins,
            "win_rate": win_rate,
            "ties": int(result.get("ties", 0)),
            "capped_games": int(result.get("capped_games", 0)),
        }
    return None


def _root_puct_leaf_rollout_rounds_values(args: argparse.Namespace) -> tuple[int, ...]:
    raw_values = (
        tuple(args.leaf_rollout_rounds_sweep)
        if args.leaf_rollout_rounds_sweep is not None
        else (args.leaf_rollout_rounds,)
    )
    values: list[int] = []
    for raw_value in raw_values:
        value = int(raw_value)
        if value < 0:
            raise ValueError("leaf rollout rounds must be non-negative.")
        if value not in values:
            values.append(value)
    if not values:
        raise ValueError("at least one leaf rollout round value is required.")
    return tuple(values)


def _root_puct_benchmark(args: argparse.Namespace) -> int:
    require_torch()
    value_checkpoint = args.value_checkpoint or args.checkpoint
    if not args.allow_legacy_checkpoints:
        _require_current_family_benchmark_checkpoints(
            args.checkpoint,
            reference_specs=(args.p1_policy, args.p2_policy),
            context="root-puct benchmark",
        )
        require_current_family_checkpoint_paths(
            (value_checkpoint,),
            context="root-puct benchmark value checkpoint",
        )
    env_config = LocalShowdownConfig(
        showdown_root=args.showdown_root,
        node_binary=args.node_binary,
    )
    policy_showdown_root = env_config.resolved_showdown_root()
    policies = {
        "p1": policy_from_spec(policy_spec_with_showdown_root(args.p1_policy, policy_showdown_root)),
        "p2": policy_from_spec(policy_spec_with_showdown_root(args.p2_policy, policy_showdown_root)),
    }
    rollout_config = RolloutConfig(
        max_decision_rounds=args.max_decision_rounds,
        format_id=args.format_id,
    )
    model, result = load_transformer_checkpoint(args.checkpoint, map_location=args.device)
    if value_checkpoint == args.checkpoint:
        value_model, value_result = model, result
        value_leaf_provenance: Mapping[str, object] | None = None
    else:
        value_model, value_result = load_transformer_checkpoint(value_checkpoint, map_location=args.device)
        # A distinct value leaf is only meaningful for the capstone when it is
        # a calibrated copy of these exact policy priors. Keep ordinary
        # same-checkpoint mechanics benchmarks available, but never silently
        # substitute an unrelated current-family checkpoint for leaf values.
        value_leaf_provenance = require_compatible_transformer_value_checkpoint(
            policy_checkpoint=args.checkpoint,
            policy_result=result,
            value_checkpoint=value_checkpoint,
            value_result=value_result,
        )
    # HIGH-1 latch: encode-time masks come from the checkpoint(s) observing through this env.
    env_config = _env_config_with_spec_masks(
        env_config,
        (args.p1_policy, args.p2_policy),
        extra_model_configs=(result.model_config, value_result.model_config),
        context="root-puct benchmark",
    )

    def value_fn(history):
        return evaluate_transformer_observation_value(
            model=value_model,
            result=value_result,
            observations=history,
            device=args.device,
        )

    def prior_fn(history):
        return evaluate_transformer_action_priors(
            model=model,
            result=result,
            observations=history,
            temperature=args.temperature,
            device=args.device,
        )

    report = benchmark_root_puct_search(
        env_factory=lambda: LocalShowdownEnv(env_config),
        policies=policies,
        rollout_config=rollout_config,
        games=args.games,
        prefixes_per_game=args.prefixes_per_game,
        seed_start=args.seed_start,
        search_player=args.search_player,
        cpuct=args.cpuct,
        root_extra_visits=args.root_extra_visits,
        value_fn=value_fn,
        prior_fn=prior_fn,
    )
    if args.json:
        payload = report.to_dict()
        payload["search_config"] = {
            "prior_temperature": args.temperature,
            "selection_mode": "visits",
        }
        payload["value_leaf"] = (
            {
                **value_leaf_provenance,
                "uses_distinct_value_checkpoint": True,
            }
            if value_leaf_provenance is not None
            else {
                "policy_checkpoint": str(args.checkpoint),
                "value_checkpoint": str(value_checkpoint),
                "uses_distinct_value_checkpoint": False,
            }
        )
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print_root_puct_benchmark_report(report)
    return 0


def _root_puct_counterfactual(args: argparse.Namespace) -> int:
    require_torch()
    if not args.allow_legacy_checkpoints:
        _require_current_family_benchmark_checkpoints(
            args.checkpoint,
            reference_specs=(
                args.p1_policy,
                args.p2_policy,
                args.continuation_p1_policy or args.p1_policy,
                args.continuation_p2_policy or args.p2_policy,
            ),
            context="root-puct counterfactual",
        )
    env_config = LocalShowdownConfig(
        showdown_root=args.showdown_root,
        node_binary=args.node_binary,
    )
    policy_showdown_root = env_config.resolved_showdown_root()
    policies = {
        "p1": policy_from_spec(policy_spec_with_showdown_root(args.p1_policy, policy_showdown_root)),
        "p2": policy_from_spec(policy_spec_with_showdown_root(args.p2_policy, policy_showdown_root)),
    }
    continuation_policies = {
        "p1": policy_from_spec(
            policy_spec_with_showdown_root(args.continuation_p1_policy or args.p1_policy, policy_showdown_root)
        ),
        "p2": policy_from_spec(
            policy_spec_with_showdown_root(args.continuation_p2_policy or args.p2_policy, policy_showdown_root)
        ),
    }
    rollout_config = RolloutConfig(
        max_decision_rounds=args.max_decision_rounds,
        format_id=args.format_id,
    )
    model, result = load_transformer_checkpoint(args.checkpoint, map_location=args.device)

    def value_fn(history):
        return evaluate_transformer_observation_value(
            model=model,
            result=result,
            observations=history,
            device=args.device,
        )

    def prior_fn(history):
        return evaluate_transformer_action_priors(
            model=model,
            result=result,
            observations=history,
            temperature=args.temperature,
            device=args.device,
        )

    env_config = _env_config_with_spec_masks(
        env_config,
        (
            args.p1_policy,
            args.p2_policy,
            args.continuation_p1_policy,
            args.continuation_p2_policy,
        ),
        extra_model_configs=(result.model_config,),
        context="root-puct counterfactual",
    )
    report = benchmark_root_puct_counterfactual_rollouts(
        env_factory=lambda: LocalShowdownEnv(env_config),
        policies=policies,
        continuation_policies=continuation_policies,
        rollout_config=rollout_config,
        games=args.games,
        prefixes_per_game=args.prefixes_per_game,
        seed_start=args.seed_start,
        search_player=args.search_player,
        cpuct=args.cpuct,
        value_fn=value_fn,
        prior_fn=prior_fn,
    )
    if args.json:
        print(json.dumps(report.to_dict(), indent=2, sort_keys=True))
    else:
        print_root_puct_counterfactual_report(report)
    return 0


def _value_calibration(args: argparse.Namespace) -> int:
    require_torch()
    if args.eval_data is not None and args.fit_out is None:
        raise ValueError("--eval-data requires --fit-out.")
    if args.fit_method != "affine" and args.fit_out is None:
        raise ValueError("--fit-method requires --fit-out.")
    _validate_value_calibration_gate_args(args)
    device = resolve_torch_device(args.device)
    model, training_result = load_transformer_checkpoint(args.checkpoint, map_location=device)
    transform = None
    eval_paths = args.eval_data if args.eval_data is not None else args.data
    evaluation_held_out = args.eval_data is not None
    if args.fit_out is not None:
        transform = fit_value_calibration_transform(
            model=model,
            training_result=training_result,
            paths=args.data,
            batch_size=args.batch_size,
            device=device,
            method=args.fit_method,
        )
        training_result = replace(
            training_result,
            value_calibration_transform=transform,
            value_calibration_source_checkpoint_sha256=checkpoint_file_sha256(args.checkpoint),
        )
        save_transformer_checkpoint(args.fit_out, model, result=training_result)
        if not evaluation_held_out:
            print(
                "warning: --fit-out is reporting calibration metrics on the same data used to fit the transform; "
                "pass --eval-data for a held-out calibration read.",
                file=sys.stderr,
            )
        if _value_calibration_transform_value_blind(transform):
            print(
                "warning: fitted value calibration transform is near-constant; the calibrated checkpoint will make "
                "value-head search nearly value-blind.",
                file=sys.stderr,
            )
    report = evaluate_value_calibration(
        model=model,
        training_result=training_result,
        paths=eval_paths,
        batch_size=args.batch_size,
        bins=args.bins,
        device=device,
    )
    quality_gates = _value_calibration_quality_gates(args, report)
    if args.json:
        payload = report.to_dict()
        if args.fit_out is not None:
            payload = {
                "checkpoint": str(args.fit_out),
                "fit_paths": [str(path) for path in args.data],
                "evaluation_paths": [str(path) for path in eval_paths],
                "evaluation_held_out": evaluation_held_out,
                "value_calibration_transform": transform.to_dict() if transform is not None else None,
                "value_calibration_source_checkpoint_sha256": (
                    training_result.value_calibration_source_checkpoint_sha256
                ),
                "report": payload,
            }
        if quality_gates["configured"]:
            payload["quality_gates"] = quality_gates
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        if args.fit_out is not None and transform is not None:
            print(f"value_calibration_transform: {_format_value_calibration_transform(transform)}")
            print(f"calibrated_checkpoint: {args.fit_out}")
            print(f"evaluation_held_out: {_format_bool(evaluation_held_out)}")
            print("")
        print_value_calibration_report(report)
        if quality_gates["configured"]:
            print("")
            _print_value_calibration_quality_gates(quality_gates)
    if quality_gates["configured"] and not quality_gates["passed"]:
        failed = ", ".join(str(check["metric"]) for check in quality_gates["checks"] if not check["passed"])
        print(f"value_calibration_quality_gates_failed: {failed}", file=sys.stderr)
        return 4
    return 0


def _value_calibration_compare(args: argparse.Namespace) -> int:
    require_torch()
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive.")
    if args.bins <= 0:
        raise ValueError("--bins must be positive.")
    device = resolve_torch_device(args.device)
    model, training_result = load_transformer_checkpoint(args.checkpoint, map_location=device)
    uncalibrated_result = replace(training_result, value_calibration_transform=None)
    entries: list[dict[str, Any]] = []

    raw_report = evaluate_value_calibration(
        model=model,
        training_result=uncalibrated_result,
        paths=args.eval_data,
        batch_size=args.batch_size,
        bins=args.bins,
        device=device,
    )
    entries.append(
        _value_calibration_compare_entry(
            method="raw",
            report=raw_report,
            transform=None,
            selection_metric=args.selection_metric,
        )
    )

    for method in ("affine", "isotonic"):
        transform = fit_value_calibration_transform(
            model=model,
            training_result=uncalibrated_result,
            paths=args.data,
            batch_size=args.batch_size,
            device=device,
            method=method,
        )
        calibrated_report = evaluate_value_calibration(
            model=model,
            training_result=replace(uncalibrated_result, value_calibration_transform=transform),
            paths=args.eval_data,
            batch_size=args.batch_size,
            bins=args.bins,
            device=device,
        )
        entries.append(
            _value_calibration_compare_entry(
                method=method,
                report=calibrated_report,
                transform=transform,
                selection_metric=args.selection_metric,
            )
        )

    best = max(entries, key=lambda entry: float(entry["selection_score"]))
    if not math.isfinite(float(best["selection_score"])):
        raise ValueError(f"{args.selection_metric} is unavailable for all calibration methods.")
    warnings = _value_calibration_compare_warnings(
        fit_paths=args.data,
        eval_paths=args.eval_data,
        entries=entries,
        best=best,
        selection_metric=args.selection_metric,
    )
    payload = {
        "checkpoint": str(args.checkpoint),
        "fit_paths": [str(path) for path in args.data],
        "evaluation_paths": [str(path) for path in args.eval_data],
        "evaluation_held_out": True,
        "batch_size": args.batch_size,
        "bins": args.bins,
        "selection_metric": args.selection_metric,
        "selection_direction": value_selection_metric_direction(args.selection_metric),
        "best_method": best["method"],
        "warnings": warnings,
        "methods": entries,
    }
    if args.out is not None:
        _write_json(args.out, payload)
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print_value_calibration_compare(payload)
        if args.out is not None:
            print(f"comparison_json: {args.out}")
    return 0


def _value_calibration_compare_entry(
    *,
    method: str,
    report: ValueCalibrationReport,
    transform: Any | None,
    selection_metric: str,
) -> dict[str, Any]:
    selection_error = None
    try:
        metric_value: float | None = value_selection_metric_value(report, selection_metric)
        selection_score = value_selection_score(metric_value, selection_metric)
    except ValueError as exc:
        metric_value = None
        selection_score = -math.inf
        selection_error = str(exc)
    entry: dict[str, Any] = {
        "method": method,
        "selection_metric_value": metric_value,
        "selection_score": selection_score,
        "value_blind": _value_calibration_transform_value_blind(transform) if transform is not None else False,
        "report": report.to_dict(),
    }
    if selection_error is not None:
        entry["selection_error"] = selection_error
    if transform is not None:
        entry["value_calibration_transform"] = transform.to_dict()
    return entry


def _value_calibration_compare_warnings(
    *,
    fit_paths: Sequence[Path],
    eval_paths: Sequence[Path],
    entries: Sequence[Mapping[str, Any]],
    best: Mapping[str, Any],
    selection_metric: str,
) -> list[dict[str, Any]]:
    warnings: list[dict[str, Any]] = []
    fit_identities = {path.resolve() for path in fit_paths}
    eval_identities = {path.resolve() for path in eval_paths}
    overlap = sorted(str(path) for path in fit_identities & eval_identities)
    if overlap:
        warnings.append(
            {
                "code": "fit_eval_path_overlap",
                "message": "Fit and eval data overlap; reported metrics are not fully held out.",
                "paths": overlap,
            }
        )
    if selection_metric in {"mae", "mse", "expected_calibration_error", "abs_bias"}:
        warnings.append(
            {
                "code": "calibration_only_selection_metric",
                "message": (
                    "Calibration-error metrics can prefer collapsed transforms; inspect ranking metrics "
                    "before using a transform for search."
                ),
            }
        )

    best_report = best["report"]
    best_correlation = best_report.get("pearson_correlation")
    if best_correlation is None or float(best_correlation) < 0.2:
        warnings.append(
            {
                "code": "selected_low_pearson_correlation",
                "message": "Selected method has weak value-return ranking signal.",
                "method": best["method"],
                "value": best_correlation,
                "threshold": 0.2,
            }
        )
    best_sign_accuracy = float(best_report["sign_accuracy"])
    if best_sign_accuracy < 0.55:
        warnings.append(
            {
                "code": "selected_low_sign_accuracy",
                "message": "Selected method has weak outcome-sign accuracy.",
                "method": best["method"],
                "value": best_sign_accuracy,
                "threshold": 0.55,
            }
        )
    if best.get("value_blind"):
        warnings.append(
            {
                "code": "selected_value_blind",
                "message": "Selected method is near-constant and may make value-head search value-blind.",
                "method": best["method"],
            }
        )

    raw = next((entry for entry in entries if entry.get("method") == "raw"), None)
    if raw is not None:
        raw_correlation = raw["report"].get("pearson_correlation")
        if best_correlation is not None and raw_correlation is not None and float(best_correlation) < float(raw_correlation) - 0.05:
            warnings.append(
                {
                    "code": "selected_pearson_regressed_vs_raw",
                    "message": "Selected transform reduced value-return ranking correlation versus raw values.",
                    "method": best["method"],
                    "raw_value": raw_correlation,
                    "selected_value": best_correlation,
                }
            )
    return warnings


def _validate_value_calibration_gate_args(args: argparse.Namespace) -> None:
    if args.min_examples is not None and args.min_examples <= 0:
        raise ValueError("--min-examples must be positive.")
    for name in ("max_mse", "max_mae", "max_abs_bias", "max_expected_calibration_error"):
        value = getattr(args, name)
        if value is not None and (not math.isfinite(value) or value < 0.0):
            raise ValueError(f"--{name.replace('_', '-')} must be finite and non-negative.")
    if args.min_sign_accuracy is not None and (
        not math.isfinite(args.min_sign_accuracy) or not 0.0 <= args.min_sign_accuracy <= 1.0
    ):
        raise ValueError("--min-sign-accuracy must be between 0 and 1.")
    if args.min_pearson_correlation is not None and (
        not math.isfinite(args.min_pearson_correlation) or not -1.0 <= args.min_pearson_correlation <= 1.0
    ):
        raise ValueError("--min-pearson-correlation must be between -1 and 1.")


def _value_calibration_quality_gates(args: argparse.Namespace, report: ValueCalibrationReport) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []
    if not _value_calibration_gate_configured(args):
        return {"configured": False, "passed": True, "checks": checks}

    def add_check(
        *,
        metric: str,
        value: float | int | None,
        threshold: float | int | None,
        operator: str,
        reason: str | None = None,
    ) -> None:
        if threshold is None:
            return
        passed = False
        if value is not None:
            passed = value >= threshold if operator == ">=" else value <= threshold
        check: dict[str, Any] = {
            "metric": metric,
            "operator": operator,
            "threshold": threshold,
            "value": value,
            "passed": passed,
        }
        if reason is not None:
            check["reason"] = reason
        checks.append(check)

    add_check(metric="examples", value=report.examples, threshold=args.min_examples, operator=">=")
    add_check(metric="mse", value=report.mse, threshold=args.max_mse, operator="<=")
    add_check(metric="mae", value=report.mae, threshold=args.max_mae, operator="<=")
    add_check(metric="abs_bias", value=abs(report.bias), threshold=args.max_abs_bias, operator="<=")
    add_check(
        metric="expected_calibration_error",
        value=report.expected_calibration_error,
        threshold=args.max_expected_calibration_error,
        operator="<=",
    )
    add_check(metric="sign_accuracy", value=report.sign_accuracy, threshold=args.min_sign_accuracy, operator=">=")
    add_check(
        metric="pearson_correlation",
        value=report.pearson_correlation,
        threshold=args.min_pearson_correlation,
        operator=">=",
        reason="unavailable" if args.min_pearson_correlation is not None and report.pearson_correlation is None else None,
    )
    return {
        "configured": bool(checks),
        "passed": all(check["passed"] for check in checks),
        "checks": checks,
    }


def _value_calibration_gate_configured(args: argparse.Namespace) -> bool:
    return any(
        getattr(args, name) is not None
        for name in (
            "min_examples",
            "max_mse",
            "max_mae",
            "max_abs_bias",
            "max_expected_calibration_error",
            "min_sign_accuracy",
            "min_pearson_correlation",
        )
    )


def _print_value_calibration_quality_gates(payload: Mapping[str, Any]) -> None:
    print("quality_gates:")
    for check in _sequence(payload.get("checks", ())):
        check_payload = _mapping(check)
        status = "pass" if check_payload.get("passed") is True else "fail"
        reason = check_payload.get("reason")
        reason_text = f" reason={reason}" if reason is not None else ""
        print(
            f"- {check_payload.get('metric')} {check_payload.get('operator')} "
            f"{check_payload.get('threshold')}: {status} "
            f"value={_format_manifest_value(check_payload.get('value'))}{reason_text}"
        )


def _iterate(args: argparse.Namespace) -> int:
    # Surface the missing-neural-extra message before any Showdown file I/O (vocab build).
    require_torch()
    _apply_iterate_experiment_preset(args)
    # Fail fast: eval-only baselines (max-damage) cannot seed self-play training.
    reject_eval_only_specs([args.initial_policy], role="self-play initial policy")
    if args.no_fixed_opponents and args.opponent_policy:
        raise ValueError("--no-fixed-opponents cannot be combined with --opponent-policy.")
    opponent_policy_specs = () if args.no_fixed_opponents else (args.opponent_policy or ("random-legal", "simple-legal"))
    reject_eval_only_specs(opponent_policy_specs, role="self-play training opponent")
    if args.no_fixed_opponents and not args.mirror_match:
        raise ValueError("--no-fixed-opponents requires --mirror-match.")
    if args.auto_promote and args.promotion_registry is None:
        raise ValueError("--auto-promote requires --promotion-registry.")
    if args.auto_promote and args.collector_advancement_mode != "incumbent-gate":
        raise ValueError(
            f"--collector-advancement-mode {args.collector_advancement_mode} cannot be combined with --auto-promote."
        )
    if args.auto_promote and args.evaluation_games <= 0 and args.require_benchmark is not False:
        raise ValueError("--auto-promote requires --evaluation-games > 0 unless --allow-missing-benchmark is set.")
    post_iteration_audit_config = post_iteration_audit_config_from_args(args)
    validate_post_iteration_audit_evaluation_games(
        post_iteration_audit_config,
        evaluation_games=args.evaluation_games,
        minimum_benchmark_matchups=MIN_NEURAL_POST_ITERATION_BENCHMARK_MATCHUPS,
    )
    # Self-play always uses the compact full Gen 3 randbat dex universe embedding. Build the
    # vocabulary ONCE and share it between the env (encode-time rows) and the model config
    # (embedding) so rows can never drift.
    if args.showdown_root is None:
        raise ValueError("neural self-play requires --showdown-root (used for the category vocabulary and the env).")
    from .randbat_vocab import gen3_category_vocabulary

    iterate_schema_version = (
        observation_schema_version_from_choice(args.observation_schema)
        or OBSERVATION_SCHEMA_VERSION
    )
    iterate_schema_spec = observation_spec_for_schema(iterate_schema_version)
    category_vocab = gen3_category_vocabulary(
        args.showdown_root,
        oov_buckets=args.category_oov_buckets,
        include_turn_merged=iterate_schema_version == OBSERVATION_SCHEMA_VERSION_V2_2,
    )
    env_config = LocalShowdownConfig(
        showdown_root=args.showdown_root,
        node_binary=args.node_binary,
        category_vocab=category_vocab,
        observation_spec=iterate_schema_spec,
    )
    rollout_config = RolloutConfig(
        max_decision_rounds=args.max_decision_rounds,
        format_id=args.format_id,
    )
    iterate_shaping = (
        parse_shaping_spec(args.shaping_weights) if args.shaping_weights is not None else None
    )
    iterate_shaping_json = iterate_shaping.canonical_json() if iterate_shaping is not None else None
    training_config = TransformerTrainingConfig(
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        learning_rate_schedule=args.learning_rate_schedule,
        learning_rate_schedule_total_games=args.learning_rate_schedule_total_games,
        learning_rate_warmup_progress=args.learning_rate_warmup_progress,
        weight_decay=args.weight_decay,
        window_size=args.window_size,
        discount=args.discount,
        capped_terminal_value=args.capped_terminal_value,
        hp_delta_return_weight=args.hp_delta_return_weight,
        faint_delta_return_weight=args.faint_delta_return_weight,
        turn_penalty_after=args.turn_penalty_after,
        turn_penalty=args.turn_penalty,
        shaping_weights=iterate_shaping_json,
        value_loss_weight=args.value_loss_weight,
        value_clip_range=args.value_clip_range,
        value_ranking_loss_weight=args.value_ranking_loss_weight,
        value_ranking_margin=args.value_ranking_margin,
        opponent_action_loss_weight=args.opponent_action_loss_weight,
        switch_action_loss_weight=args.switch_action_loss_weight,
        action_family_loss_weight=args.action_family_loss_weight,
        switch_target_loss_weight=args.switch_target_loss_weight,
        max_batches=args.max_batches,
        device=args.device,
        objective=args.objective,
        clip_epsilon=args.clip_epsilon,
        entropy_coef=args.entropy_coef,
        normalize_advantage=not args.no_normalize_advantage,
        ppo_target_mode=args.ppo_target_mode,
        gae_lambda=args.gae_lambda,
        max_grad_norm=args.max_grad_norm,
        amp=args.amp,
    )
    iterate_model_config_kwargs = dict(
        policy_id=args.policy_id,
        window_size=args.window_size,
        embedding_dim=args.embedding_dim,
        transformer_layers=args.layers,
        attention_heads=args.attention_heads,
        feedforward_dim=args.feedforward_dim,
        dropout=args.dropout,
        temporal_aggregator=args.temporal_aggregator,
        stats_block_enabled=not args.no_stats_block,
        exact_state_enabled=not args.no_exact_state,
        transition_token_budget=args.transition_token_budget,
        tier2_residuals=True if args.tier2_residuals is None else bool(args.tier2_residuals),
        # Asymmetric default vs tier2_residuals: absent flag resolves OFF (see the train
        # builder). On --resume this config is equality-checked against the loaded model
        # config, so a disagreeing tier2_investment hard-fails there. Encoder schema-gates
        # the investment columns, so enabling under v2 is a clean no-op.
        tier2_investment=False if args.tier2_investment is None else bool(args.tier2_investment),
        reward_shaping=iterate_shaping_json,
        observation_schema_version=iterate_schema_version,
        categorical_feature_count=iterate_schema_spec.categorical_feature_count,
        numeric_feature_count=iterate_schema_spec.numeric_feature_count,
        token_count=iterate_schema_spec.token_count,
    )
    # Reuse the single vocabulary built above (shared with the env), so the embedding rows the
    # model learns are exactly the rows the env encodes.
    model_config = TransformerPolicyConfig.compact_category(
        category_vocab=category_vocab.tokens,
        category_oov_buckets=category_vocab.oov_buckets,
        **iterate_model_config_kwargs,
    )
    print(
        f"category vocab (randbat-dex universe): {len(category_vocab.tokens):,} tokens + "
        f"{args.category_oov_buckets:,} oov -> embedding rows {model_config.categorical_vocab_size:,}",
        file=sys.stderr,
    )
    initial_policy = policy_spec_with_showdown_root(args.initial_policy, args.showdown_root)
    opponent_policies = tuple(
        policy_spec_with_showdown_root(spec, args.showdown_root)
        for spec in opponent_policy_specs
    )
    # Eval-only references (e.g. max-damage) are allowed here but never seed training above.
    benchmark_references = tuple(
        policy_spec_with_showdown_root(spec, args.showdown_root)
        for spec in (args.benchmark_reference_policy or ())
    )
    value_selection_requested = bool(args.value_selection or args.value_selection_heldout_games > 0)
    if args.value_selection_heldout_games > 0 and not args.value_selection:
        print(
            "warning: --value-selection-heldout-games implies --value-selection.",
            file=sys.stderr,
        )
    if value_selection_requested and args.value_selection_heldout_games <= 0:
        print(
            "warning: --value-selection in neural iterate scores self-play training rollouts, "
            "not held-out validation; use it as value-head calibration plumbing, not policy-strength evidence.",
            file=sys.stderr,
        )
    if value_selection_requested and args.value_selection_heldout_games > 0:
        main_seed_upper_bound = args.seed_start + (args.iterations * args.games_per_iteration)
        if args.value_selection_seed_start < main_seed_upper_bound:
            print(
                "warning: --value-selection-seed-start overlaps the requested training seed range; "
                "held-out value-selection games may not be independent.",
                file=sys.stderr,
            )
    if value_selection_requested and args.value_selection_scope == "history":
        print(
            "warning: --value-selection-scope history re-evaluates the full accumulated selection "
            "history after every epoch and can become expensive.",
            file=sys.stderr,
        )
    auto_promotion_config = _auto_promotion_config_from_args(args)
    # HIGH-1 latch: the collection/benchmark env must encode with the candidate's masks; any
    # neural opponent/reference checkpoint with DIFFERENT masks cannot share this env and
    # refuses loudly here rather than producing silently-mismatched observations.
    env_config = _env_config_with_spec_masks(
        env_config,
        (initial_policy, *opponent_policies, *benchmark_references),
        extra_model_configs=(model_config,),
        context="neural iterate",
    )
    result = run_neural_selfplay_iterations(
        run_dir=args.run_dir,
        iterations=args.iterations,
        games_per_iteration=args.games_per_iteration,
        env_factory=lambda: LocalShowdownEnv(env_config),
        rollout_config=rollout_config,
        model_config=model_config,
        training_config=training_config,
        seed_start=args.seed_start,
        initial_policy_spec=initial_policy,
        fixed_opponent_policy_specs=opponent_policies,
        benchmark_reference_policy_specs=benchmark_references,
        mirror_match=args.mirror_match,
        collection_temperature=args.collection_temperature,
        collection_exploration_epsilon=args.collection_epsilon,
        tensorboard_log_dir=args.tensorboard_logdir,
        max_historical_opponents=args.max_historical_opponents,
        historical_opponent_selection=args.historical_opponent_selection,
        evaluation_games=args.evaluation_games,
        evaluation_interval_games=args.evaluation_interval_games,
        evaluation_seed_start=args.evaluation_seed_start,
        worker_count=args.workers,
        promotion_registry_path=args.promotion_registry,
        required_promoted_opponent_pool_size=args.require_promoted_opponent_pool_size,
        auto_promotion_config=auto_promotion_config,
        post_iteration_audit_config=post_iteration_audit_config,
        post_iteration_audit_failure_mode=args.audit_failure_mode,
        value_calibration_config=_value_calibration_config_from_args(args),
        value_selection_config=_value_selection_config_from_args(args),
        collector_advancement_mode=args.collector_advancement_mode,
        experiment_preset=args.experiment_preset,
        training_cache_root=args.training_cache_root,
        training_cache_chunk_games=args.training_cache_chunk_games,
        training_cache_max_root_bytes=_cache_gb_to_bytes(args.max_cache_gb),
        delete_training_cache_after_train=args.delete_cache_after_read,
        write_rollout_jsonl=args.write_rollout_jsonl,
        learning_rate_schedule_completed_games=args.learning_rate_schedule_completed_games,
        resume=args.resume,
    )
    if args.json:
        print(json.dumps(result.to_dict(), indent=2, sort_keys=True))
    else:
        _print_iterate_summary(result)
    return 0


def _apply_iterate_experiment_preset(args: argparse.Namespace) -> None:
    if args.experiment_preset == "none":
        return
    defaults = _ITERATE_EXPERIMENT_PRESET_DEFAULTS.get(args.experiment_preset)
    if defaults is None:
        raise ValueError(f"unsupported neural iterate experiment preset: {args.experiment_preset!r}.")

    for name in _ITERATE_PRESET_LOOP_SHAPE_KEYS:
        if name in defaults:
            _set_preset_default(args, name, defaults[name])

    # PPO hyperparameters only apply when the resolved objective is PPO, so a user who overrides
    # --objective back to behavior-cloning is not silently handed a PPO-only knob set.
    if args.objective == "ppo":
        for name in _ITERATE_PRESET_PPO_KEYS:
            if name in defaults:
                _set_preset_default(args, name, defaults[name])

    explicit_options = getattr(args, "_explicit_cli_options", frozenset())
    benchmark_references = list(args.benchmark_reference_policy or ())
    if "benchmark_reference_policy" not in explicit_options and not benchmark_references:
        benchmark_references = ["max-damage"]
    elif "max-damage" not in {str(spec).partition("?")[0] for spec in benchmark_references}:
        benchmark_references.append("max-damage")
    args.benchmark_reference_policy = benchmark_references or None


def _set_preset_default(args: argparse.Namespace, name: str, value: Any) -> None:
    if name not in getattr(args, "_explicit_cli_options", frozenset()):
        setattr(args, name, value)


def _print_run_audit_failure(exc: RunAuditFailure) -> None:
    failed = [check.name for check in exc.result.blocking_failed_checks]
    print(f"audit_failed: {exc.result.manifest_path}", file=sys.stderr)
    print(f"failed_checks: {', '.join(failed) if failed else 'unknown'}", file=sys.stderr)


def _foundation_plan(args: argparse.Namespace) -> int:
    recipe = _foundation_recipe(args)
    if args.json:
        print(json.dumps(recipe, indent=2, sort_keys=True))
        return 0
    print("neural_foundation_plan:")
    print(f"purpose: CPU foundation PPO run using the {recipe['experiment_preset']} preset")
    print(f"profile: {recipe['profile']}")
    print(f"variant: {recipe['variant']}")
    print(f"run_dir: {recipe['run_dir']}")
    print(f"manifest: {recipe['manifest_path']}")
    print("command:")
    print(recipe["command"]["shell"])
    return 0


def _foundation_run(args: argparse.Namespace) -> int:
    recipe = _foundation_recipe(args)
    summary_path = args.summary_path if args.summary_path is not None else args.run_dir / "neural-foundation-run-summary.json"
    _validate_foundation_run_paths(args.run_dir, summary_path=summary_path, resume=args.resume)
    started = time.perf_counter()
    summary: dict[str, Any] = {
        "schema_version": NEURAL_FOUNDATION_RUN_SUMMARY_SCHEMA_VERSION,
        "status": "running",
        "summary_path": str(summary_path),
        "started_at": _utc_timestamp(),
        "ended_at": None,
        "duration_seconds": None,
        "source": recipe["source"],
        "recipe": recipe,
        "returncode": None,
        "stdout_tail": None,
        "stderr_tail": None,
        "foundation": None,
        "error": None,
    }
    _write_json(summary_path, summary)
    print("neural_foundation_run:")
    print(f"purpose: CPU foundation PPO run using the {recipe['experiment_preset']} preset")
    print(f"summary: {summary_path}")
    print(recipe["command"]["shell"], flush=True)
    try:
        completed = subprocess.run(recipe["command"]["argv"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    except Exception as exc:
        summary["status"] = "failed"
        summary["ended_at"] = _utc_timestamp()
        summary["duration_seconds"] = round(time.perf_counter() - started, 6)
        summary["error"] = {"type": type(exc).__name__, "message": str(exc)}
        _write_json(summary_path, summary)
        print(f"error: neural foundation run raised {type(exc).__name__}: {exc}", file=sys.stderr)
        raise

    summary["returncode"] = int(completed.returncode)
    summary["stdout_tail"] = _text_tail(completed.stdout)
    summary["stderr_tail"] = _text_tail(completed.stderr)
    summary["foundation"] = _foundation_run_derived_report(args.run_dir, completed.stdout)
    summary["status"] = "passed" if completed.returncode == 0 else "failed"
    summary["ended_at"] = _utc_timestamp()
    summary["duration_seconds"] = round(time.perf_counter() - started, 6)
    _write_json(summary_path, summary)
    if completed.returncode == 0:
        print("neural_foundation_run: PASS")
        print("note: PASS means the wrapper command exited 0; inspect benchmarks and foundation readiness for strength.")
    else:
        print(f"error: neural foundation run failed with exit code {completed.returncode}", file=sys.stderr)
        if completed.stderr:
            print(_text_tail(completed.stderr), file=sys.stderr)
    return int(completed.returncode)


def _foundation_report(args: argparse.Namespace) -> int:
    summary_path, summary = _load_foundation_summary(args.path)
    payload = _foundation_report_payload(summary_path, summary)
    status = str(payload.get("status", "unknown"))
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0 if status == "passed" else 2
    recipe = _optional_mapping(payload.get("recipe"))
    foundation = _optional_mapping(payload.get("foundation"))
    readiness = _optional_mapping(foundation.get("foundation_readiness"))
    print("neural_foundation_report:")
    print("note: wrapper status is process health only, not policy-strength evidence.")
    print(f"summary: {summary_path}")
    print(f"status: {status}")
    print(f"started_at: {_format_manifest_value(payload.get('started_at'))}")
    print(f"ended_at: {_format_manifest_value(payload.get('ended_at'))}")
    print(f"duration_seconds: {_format_manifest_value(payload.get('duration_seconds'))}")
    print(f"returncode: {_format_manifest_value(payload.get('returncode'))}")
    if recipe:
        print(f"profile: {_format_manifest_value(recipe.get('profile'))}")
        print(f"run_dir: {_format_manifest_value(recipe.get('run_dir'))}")
    print(f"manifest: {_format_manifest_value(recipe.get('manifest_path'))}")
    print(f"foundation_manifest_available: {_format_bool(foundation.get('manifest_available'))}")
    print(f"latest_checkpoint: {_format_manifest_value(foundation.get('latest_checkpoint_path'))}")
    print(f"foundation_evidence_status: {_format_manifest_value(readiness.get('foundation_evidence_status'))}")
    max_damage = _optional_mapping(readiness.get("max_damage_yardstick"))
    if max_damage.get("available") is True:
        print(f"max_damage_yardstick: {_format_foundation_yardstick_compact(max_damage)}")
    else:
        print("max_damage_yardstick: missing")
    best_max_damage = _optional_mapping(readiness.get("best_max_damage_yardstick"))
    if best_max_damage.get("available") is True:
        print(f"best_max_damage_yardstick: {_format_foundation_yardstick_compact(best_max_damage)}")
    else:
        print("best_max_damage_yardstick: missing")
    reasons = readiness.get("reasons")
    if isinstance(reasons, list) and reasons:
        print(f"reasons: {', '.join(str(reason) for reason in reasons)}")
    return 0 if status == "passed" else 2


def _foundation_value_tune_plan(args: argparse.Namespace) -> int:
    recipe = _foundation_value_tune_recipe(args)
    if args.json:
        print(json.dumps(recipe, indent=2, sort_keys=True))
        return 0
    print("neural_foundation_value_tune_plan:")
    print("purpose: value-only fine-tune for a selected foundation checkpoint")
    print(f"candidate_source: {recipe['candidate_source']}")
    print(f"candidate_iteration: {recipe['candidate_iteration']}")
    print(f"candidate_checkpoint: {recipe['candidate_checkpoint_path']}")
    print(f"out_dir: {recipe['out_dir']}")
    _print_foundation_value_tune_warnings(recipe)
    print("command:")
    print(recipe["command"]["shell"])
    return 0


def _foundation_value_tune_run(args: argparse.Namespace) -> int:
    recipe = _foundation_value_tune_recipe(args)
    out_dir = Path(str(recipe["out_dir"]))
    summary_path = args.summary_path if args.summary_path is not None else out_dir / "neural-foundation-value-tune-summary.json"
    _validate_foundation_value_tune_paths(out_dir, summary_path=summary_path)
    out_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    summary: dict[str, Any] = {
        "schema_version": NEURAL_FOUNDATION_VALUE_TUNE_SUMMARY_SCHEMA_VERSION,
        "status": "running",
        "summary_path": str(summary_path),
        "started_at": _utc_timestamp(),
        "ended_at": None,
        "duration_seconds": None,
        "source": recipe["source"],
        "recipe": recipe,
        "returncode": None,
        "stdout_tail": None,
        "stderr_tail": None,
        "artifacts": recipe["artifacts"],
        "value_calibration": None,
        "error": None,
    }
    _write_json(summary_path, summary)
    print("neural_foundation_value_tune_run:")
    print("purpose: value-only fine-tune for a selected foundation checkpoint")
    print(f"summary: {summary_path}")
    print(recipe["command"]["shell"], flush=True)
    try:
        completed = subprocess.run(recipe["command"]["argv"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    except Exception as exc:
        summary["status"] = "failed"
        summary["ended_at"] = _utc_timestamp()
        summary["duration_seconds"] = round(time.perf_counter() - started, 6)
        summary["error"] = {"type": type(exc).__name__, "message": str(exc)}
        _write_json(summary_path, summary)
        print(f"error: neural foundation value tune raised {type(exc).__name__}: {exc}", file=sys.stderr)
        raise

    summary["returncode"] = int(completed.returncode)
    summary["stdout_tail"] = _text_tail(completed.stdout)
    summary["stderr_tail"] = _text_tail(completed.stderr)
    summary["value_calibration"] = _load_optional_json_or_error(Path(str(recipe["artifacts"]["value_calibration_path"])))
    summary["status"] = "passed" if completed.returncode == 0 else "failed"
    summary["ended_at"] = _utc_timestamp()
    summary["duration_seconds"] = round(time.perf_counter() - started, 6)
    _write_json(summary_path, summary)
    if completed.returncode == 0:
        print("neural_foundation_value_tune_run: PASS")
        print("note: PASS means value-only fine-tune completed; inspect calibration before using as a search leaf evaluator.")
    else:
        print(f"error: neural foundation value tune failed with exit code {completed.returncode}", file=sys.stderr)
        if completed.stderr:
            print(_text_tail(completed.stderr), file=sys.stderr)
    return int(completed.returncode)


def _foundation_value_tune_report(args: argparse.Namespace) -> int:
    summary_path = args.path / "neural-foundation-value-tune-summary.json" if args.path.is_dir() else args.path
    summary = _load_json_mapping(summary_path)
    status = str(summary.get("status", "unknown"))
    if args.json:
        print(json.dumps(summary, indent=2, sort_keys=True))
        return 0 if status == "passed" else 2
    recipe = _optional_mapping(summary.get("recipe"))
    artifacts = _optional_mapping(summary.get("artifacts"))
    calibration = _optional_mapping(_optional_mapping(summary.get("value_calibration")).get("report"))
    print("neural_foundation_value_tune_report:")
    print("note: wrapper status is process health only; inspect calibration metrics for value quality.")
    print(f"summary: {summary_path}")
    print(f"status: {status}")
    print(f"duration_seconds: {_format_manifest_value(summary.get('duration_seconds'))}")
    print(f"returncode: {_format_manifest_value(summary.get('returncode'))}")
    print(f"candidate_source: {_format_manifest_value(recipe.get('candidate_source'))}")
    print(f"candidate_iteration: {_format_manifest_value(recipe.get('candidate_iteration'))}")
    print(f"candidate_checkpoint: {_format_manifest_value(recipe.get('candidate_checkpoint_path'))}")
    print(f"value_tuned_checkpoint: {_format_manifest_value(artifacts.get('checkpoint_path'))}")
    _print_foundation_value_tune_warnings(recipe)
    if calibration:
        print(
            "value_calibration: "
            f"examples={_format_manifest_value(calibration.get('examples'))} "
            f"sign={_format_optional_float(calibration.get('sign_accuracy'), digits=4)} "
            f"ece={_format_optional_float(calibration.get('expected_calibration_error'), digits=6)} "
            f"corr={_format_optional_float(calibration.get('pearson_correlation'), digits=4)}"
        )
    else:
        print("value_calibration: missing")
    return 0 if status == "passed" else 2


def _foundation_report_payload(summary_path: Path, summary: Mapping[str, Any]) -> dict[str, Any]:
    payload = dict(summary)
    payload["summary_source_path"] = str(summary_path)
    foundation = dict(_optional_mapping(payload.get("foundation")))
    manifest, manifest_source, manifest_error = _foundation_manifest_from_summary(summary_path, summary)
    if manifest is not None:
        iterations = tuple(_mapping(iteration) for iteration in _sequence(manifest.get("iterations", ())))
        foundation["manifest_available"] = True
        foundation["manifest_source"] = manifest_source
        foundation["manifest_error"] = None
        foundation["foundation_readiness"] = _foundation_readiness_report(iterations)
        latest = iterations[-1] if iterations else {}
        if latest:
            foundation["latest_iteration"] = _int_or_none(latest.get("iteration"))
            foundation["latest_checkpoint_path"] = _string_or_none(latest.get("checkpoint_path"))
    elif foundation:
        foundation["manifest_available"] = False
        foundation["manifest_source"] = manifest_source
        foundation["manifest_error"] = manifest_error
    if foundation:
        payload["foundation"] = foundation
    return payload


def _foundation_compare(args: argparse.Namespace) -> int:
    quality_gate_config = _foundation_quality_gate_config_from_args(args)
    entries = [_foundation_compare_entry_or_error(path, candidate_source=args.candidate_source) for path in args.paths]
    for entry in entries:
        entry["quality_gate"] = _foundation_quality_gate(entry, quality_gate_config)
    payload = {
        "schema_version": NEURAL_FOUNDATION_COMPARE_SCHEMA_VERSION,
        "summary_count": len(args.paths),
        "candidate_source": args.candidate_source,
        "quality_gate": quality_gate_config,
        "entries": entries,
    }
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
        return _foundation_compare_exit_code(entries, quality_gate_config)
    _print_foundation_compare(payload)
    return _foundation_compare_exit_code(entries, quality_gate_config)


def _foundation_quality_gate_config_from_args(args: argparse.Namespace) -> dict[str, Any]:
    thresholds = {
        "require_sample_sized": bool(args.require_sample_sized),
        "min_max_damage_games": args.min_max_damage_games,
        "min_max_damage_win_rate": args.min_max_damage_win_rate,
        "min_value_pearson_correlation": args.min_value_pearson_correlation,
        "min_value_sign_accuracy": args.min_value_sign_accuracy,
        "max_value_expected_calibration_error": args.max_value_expected_calibration_error,
    }
    configured = any(value is not None and value is not False for value in thresholds.values())
    if args.require_quality_pass and not configured:
        raise ValueError("--require-quality-pass requires at least one quality threshold.")
    if args.min_max_damage_games is not None and args.min_max_damage_games <= 0:
        raise ValueError("--min-max-damage-games must be positive.")
    _validate_foundation_quality_range(
        thresholds["min_max_damage_win_rate"],
        name="min_max_damage_win_rate",
        lower=0.0,
        upper=1.0,
    )
    _validate_foundation_quality_range(
        thresholds["min_value_pearson_correlation"],
        name="min_value_pearson_correlation",
        lower=-1.0,
        upper=1.0,
    )
    _validate_foundation_quality_range(
        thresholds["min_value_sign_accuracy"],
        name="min_value_sign_accuracy",
        lower=0.0,
        upper=1.0,
    )
    _validate_foundation_quality_range(
        thresholds["max_value_expected_calibration_error"],
        name="max_value_expected_calibration_error",
        lower=0.0,
        upper=None,
    )
    return {
        "configured": configured,
        "require_quality_pass": bool(args.require_quality_pass),
        **thresholds,
    }


def _validate_foundation_quality_range(
    value: object,
    *,
    name: str,
    lower: float,
    upper: float | None,
) -> None:
    if value is None:
        return
    parsed = float(value)
    if parsed < lower or (upper is not None and parsed > upper):
        range_text = f"[{lower}, {upper}]" if upper is not None else f">= {lower}"
        raise ValueError(f"--{name.replace('_', '-')} must be in range {range_text}.")


def _foundation_compare_exit_code(
    entries: Sequence[Mapping[str, Any]],
    quality_gate_config: Mapping[str, Any],
) -> int:
    if any(entry.get("load_error") for entry in entries):
        return 1
    if quality_gate_config.get("require_quality_pass") is True and quality_gate_config.get("configured") is True:
        for entry in entries:
            gate = _optional_mapping(entry.get("quality_gate"))
            if gate.get("status") == "pass":
                return 0
        return 2
    return 0


def _foundation_quality_gate(entry: Mapping[str, Any], config: Mapping[str, Any]) -> dict[str, Any]:
    if config.get("configured") is not True:
        return {"configured": False, "status": "not_configured", "checks": []}
    checks: list[dict[str, Any]] = []
    if entry.get("load_error") is not None:
        checks.append(
            {
                "name": "summary_load_error",
                "passed": False,
                "actual": entry.get("load_error"),
                "threshold": "summary loads",
            }
        )
    if config.get("require_sample_sized") is True:
        actual = _string_or_none(entry.get("foundation_evidence_status"))
        checks.append(
            {
                "name": "foundation_evidence_status",
                "passed": actual == "present_and_sample_sized",
                "actual": actual,
                "threshold": "present_and_sample_sized",
            }
        )
    yardsticks = _optional_mapping(entry.get("yardsticks"))
    max_damage = _optional_mapping(yardsticks.get("max-damage"))
    value = _optional_mapping(entry.get("value_calibration"))
    _append_min_quality_check(
        checks,
        name="min_max_damage_games",
        actual=_float_or_none(max_damage.get("games")) if max_damage.get("available") is True else None,
        threshold=config.get("min_max_damage_games"),
    )
    _append_min_quality_check(
        checks,
        name="min_max_damage_win_rate",
        actual=_float_or_none(max_damage.get("win_rate")) if max_damage.get("available") is True else None,
        threshold=config.get("min_max_damage_win_rate"),
    )
    _append_min_quality_check(
        checks,
        name="min_value_pearson_correlation",
        actual=_float_or_none(value.get("pearson_correlation")) if value.get("available") is True else None,
        threshold=config.get("min_value_pearson_correlation"),
    )
    _append_min_quality_check(
        checks,
        name="min_value_sign_accuracy",
        actual=_float_or_none(value.get("sign_accuracy")) if value.get("available") is True else None,
        threshold=config.get("min_value_sign_accuracy"),
    )
    _append_max_quality_check(
        checks,
        name="max_value_expected_calibration_error",
        actual=_float_or_none(value.get("expected_calibration_error")) if value.get("available") is True else None,
        threshold=config.get("max_value_expected_calibration_error"),
    )
    failed = [check["name"] for check in checks if check.get("passed") is not True]
    return {
        "configured": True,
        "status": "pass" if not failed else "fail",
        "failed_checks": failed,
        "checks": checks,
    }


def _append_min_quality_check(
    checks: list[dict[str, Any]],
    *,
    name: str,
    actual: float | None,
    threshold: object,
) -> None:
    if threshold is None:
        return
    threshold_value = float(threshold)
    checks.append(
        {
            "name": name,
            "passed": actual is not None and actual >= threshold_value,
            "actual": actual,
            "threshold": threshold_value,
        }
    )


def _append_max_quality_check(
    checks: list[dict[str, Any]],
    *,
    name: str,
    actual: float | None,
    threshold: object,
) -> None:
    if threshold is None:
        return
    threshold_value = float(threshold)
    checks.append(
        {
            "name": name,
            "passed": actual is not None and actual <= threshold_value,
            "actual": actual,
            "threshold": threshold_value,
        }
    )


def _foundation_compare_entry_or_error(path: Path, *, candidate_source: str) -> dict[str, Any]:
    try:
        return _foundation_compare_entry(path, candidate_source=candidate_source)
    except Exception as exc:
        return {
            "label": str(path),
            "summary_path": str(path),
            "status": "load_error",
            "profile": "unknown",
            "variant": "unknown",
            "run_dir": None,
            "duration_seconds": None,
            "latest_iteration": None,
            "latest_checkpoint_path": None,
            "candidate_source": candidate_source,
            "candidate_iteration": None,
            "candidate_checkpoint_path": None,
            "candidate_selection_error": str(exc),
            "foundation_evidence_status": "unknown",
            "reasons": [],
            "load_error": str(exc),
            "manifest_loaded": False,
            "manifest_source": None,
            "manifest_error": "summary_load_failed",
            "value_calibration": {"available": False},
            "yardsticks": {
                policy_id: {"available": False, "opponent_policy_id": policy_id}
                for policy_id in ("max-damage", "simple-legal", "random-legal")
            },
            "best_yardsticks": {
                "max-damage": {"available": False, "opponent_policy_id": "max-damage"},
            },
        }


def _foundation_compare_entry(path: Path, *, candidate_source: str) -> dict[str, Any]:
    summary_path, summary = _load_foundation_summary(path)
    recipe = _optional_mapping(summary.get("recipe"))
    foundation = _optional_mapping(summary.get("foundation"))
    readiness = _optional_mapping(foundation.get("foundation_readiness"))
    manifest, manifest_source, manifest_error = _foundation_manifest_from_summary(summary_path, summary)
    iterations = tuple(_mapping(iteration) for iteration in _sequence(manifest.get("iterations", ()))) if manifest else ()
    curves = _benchmark_opponent_curves(iterations) if iterations else {}
    selected_iteration, candidate_error = _select_foundation_candidate_iteration(
        iterations=iterations,
        manifest=manifest,
        candidate_source=candidate_source,
    )
    selected_iterations = (selected_iteration,) if selected_iteration is not None else ()
    selected_curves = _benchmark_opponent_curves(selected_iterations) if selected_iterations else {}
    if iterations:
        readiness = _foundation_readiness_report(selected_iterations) if selected_iteration is not None else _missing_foundation_candidate_readiness(candidate_error)
    elif candidate_source != "latest":
        candidate_error = "candidate source requires a loaded manifest"
        readiness = _missing_foundation_candidate_readiness(candidate_error)
    else:
        candidate_error = None
    candidate_iteration = _int_or_none(selected_iteration.get("iteration")) if selected_iteration is not None else None
    candidate_checkpoint_path = (
        _string_or_none(selected_iteration.get("checkpoint_path")) if selected_iteration is not None else None
    )
    fallback_to_summary_candidate = selected_iteration is None and candidate_error is None
    return {
        "label": _foundation_compare_label(summary_path, recipe),
        "summary_path": str(summary_path),
        "status": str(summary.get("status", "unknown")),
        "profile": str(recipe.get("profile", "unknown")),
        "variant": str(recipe.get("variant", "baseline")),
        "run_dir": _string_or_none(recipe.get("run_dir")),
        "duration_seconds": _float_or_none(summary.get("duration_seconds")),
        "latest_iteration": _coalesce_optional_int(
            iterations[-1].get("iteration") if iterations else readiness.get("latest_iteration"),
            foundation.get("latest_iteration"),
        ),
        "latest_checkpoint_path": _string_or_none(_optional_mapping(manifest).get("latest_checkpoint_path"))
        or _string_or_none(foundation.get("latest_checkpoint_path")),
        "candidate_source": candidate_source,
        "candidate_iteration": candidate_iteration
        if candidate_iteration is not None
        else (
            _coalesce_optional_int(readiness.get("latest_iteration"), foundation.get("latest_iteration"))
            if fallback_to_summary_candidate
            else None
        ),
        "candidate_checkpoint_path": candidate_checkpoint_path
        or (_string_or_none(foundation.get("latest_checkpoint_path")) if fallback_to_summary_candidate else None),
        "candidate_selection_error": candidate_error,
        "foundation_evidence_status": _string_or_none(readiness.get("foundation_evidence_status")) or "unknown",
        "reasons": [str(reason) for reason in _sequence(readiness.get("reasons", ()))],
        "manifest_loaded": manifest is not None,
        "manifest_source": manifest_source,
        "manifest_error": manifest_error,
        "value_calibration": _foundation_compare_value_calibration(readiness),
        "yardsticks": {
            policy_id: _foundation_compare_yardstick(policy_id, selected_curves, readiness)
            for policy_id in ("max-damage", "simple-legal", "random-legal")
        },
        "best_yardsticks": {
            "max-damage": _foundation_compare_best_yardstick("max-damage", curves, readiness),
        },
    }


def _select_foundation_candidate_iteration(
    *,
    iterations: tuple[Mapping[str, Any], ...],
    manifest: Mapping[str, Any] | None,
    candidate_source: str,
) -> tuple[Mapping[str, Any] | None, str | None]:
    if candidate_source not in FOUNDATION_COMPARE_CANDIDATE_SOURCES:
        raise ValueError(f"unsupported foundation candidate source: {candidate_source!r}.")
    if not iterations:
        return None, "manifest has no iterations"
    if candidate_source == "latest":
        return iterations[-1], None
    if candidate_source == "latest-accepted":
        checkpoint_path = _string_or_none(_optional_mapping(manifest).get("latest_accepted_checkpoint_path"))
        if checkpoint_path is None:
            checkpoint_path = _checkpoint_path_from_policy_spec(_string_or_none(_optional_mapping(manifest).get("current_policy_spec")))
        if checkpoint_path is None:
            return None, "latest accepted checkpoint path unavailable"
        selected = _find_foundation_iteration_by_checkpoint(iterations, checkpoint_path)
        if selected is None:
            return None, f"latest accepted checkpoint not found in iterations: {checkpoint_path}"
        return selected, None
    curves = _benchmark_opponent_curves(iterations)
    best = _best_curve_entry(curves, "max-damage")
    if best is None:
        return None, "max-damage yardstick unavailable"
    selected = None
    checkpoint_path = _string_or_none(best.get("checkpoint_path"))
    if checkpoint_path is not None:
        selected = _find_foundation_iteration_by_checkpoint(iterations, checkpoint_path)
    if selected is None:
        best_iteration = _int_or_none(best.get("iteration"))
        selected = _find_foundation_iteration_by_number(iterations, best_iteration)
    if selected is None:
        return None, "best max-damage iteration not found in manifest"
    return selected, None


def _checkpoint_path_from_policy_spec(policy_spec: str | None) -> str | None:
    if policy_spec is None:
        return None
    if policy_spec.startswith("neural:"):
        return policy_spec[len("neural:") :].partition("?")[0]
    return None


def _find_foundation_iteration_by_checkpoint(
    iterations: tuple[Mapping[str, Any], ...],
    checkpoint_path: str,
) -> Mapping[str, Any] | None:
    for iteration in iterations:
        if _foundation_checkpoint_paths_match(_string_or_none(iteration.get("checkpoint_path")), checkpoint_path):
            return iteration
    return None


def _foundation_checkpoint_paths_match(left: str | None, right: str | None) -> bool:
    if left is None or right is None:
        return False
    if left == right:
        return True
    return Path(left).expanduser().resolve(strict=False) == Path(right).expanduser().resolve(strict=False)


def _find_foundation_iteration_by_number(
    iterations: tuple[Mapping[str, Any], ...],
    iteration_number: int | None,
) -> Mapping[str, Any] | None:
    if iteration_number is None:
        return None
    for iteration in iterations:
        if _int_or_none(iteration.get("iteration")) == iteration_number:
            return iteration
    return None


def _missing_foundation_candidate_readiness(reason: str | None) -> dict[str, Any]:
    reasons = ["candidate_selection_failed"]
    if reason:
        reasons.append(reason)
    return {
        "latest_iteration": None,
        "milestone_benchmark_games": FOUNDATION_MILESTONE_BENCHMARK_GAMES,
        "value_calibration": {"available": False},
        "max_damage_yardstick": {"available": False, "opponent_policy_id": "max-damage"},
        "best_max_damage_yardstick": {"available": False, "opponent_policy_id": "max-damage"},
        "foundation_evidence_status": "incomplete",
        "reasons": reasons,
    }


def _foundation_manifest_from_summary(
    summary_path: Path,
    summary: Mapping[str, Any],
) -> tuple[Mapping[str, Any] | None, str | None, str | None]:
    recipe = _optional_mapping(summary.get("recipe"))
    foundation = _optional_mapping(summary.get("foundation"))
    candidates: list[Path] = []
    candidates.append(summary_path.parent / "manifest.json")
    for value in (foundation.get("manifest_source"), recipe.get("manifest_path")):
        if isinstance(value, str) and value:
            candidates.append(Path(value))
    seen: set[str] = set()
    missing: list[str] = []
    for candidate in candidates:
        candidate_key = str(candidate)
        if candidate_key in seen:
            continue
        seen.add(candidate_key)
        if not candidate.exists():
            missing.append(candidate_key)
            continue
        try:
            payload = json.loads(candidate.read_text(encoding="utf-8"))
        except Exception as exc:
            return None, str(candidate), str(exc)
        if not isinstance(payload, Mapping):
            return None, str(candidate), "manifest JSON was not an object"
        return payload, str(candidate), None
    source = missing[0] if missing else None
    error = "manifest not found" if source is not None else "manifest path unavailable"
    return None, source, error


def _foundation_compare_value_calibration(readiness: Mapping[str, Any]) -> dict[str, Any]:
    value = _optional_mapping(readiness.get("value_calibration"))
    if value.get("available") is not True:
        return {"available": False}
    return {
        "available": True,
        "examples": _int_or_none(value.get("examples")),
        "sign_accuracy": _float_or_none(value.get("sign_accuracy")),
        "expected_calibration_error": _float_or_none(value.get("expected_calibration_error")),
        "pearson_correlation": _float_or_none(value.get("pearson_correlation")),
        "mse": _float_or_none(value.get("mse")),
        "mae": _float_or_none(value.get("mae")),
        "bias": _float_or_none(value.get("bias")),
    }


def _foundation_compare_yardstick(
    policy_id: str,
    curves: Mapping[str, list[dict[str, Any]]],
    readiness: Mapping[str, Any],
) -> dict[str, Any]:
    entry = _latest_curve_entry(curves, policy_id)
    if entry is not None:
        return _foundation_compare_yardstick_payload(policy_id, entry, source="manifest")
    if policy_id == "max-damage":
        max_damage = _optional_mapping(readiness.get("max_damage_yardstick"))
        if max_damage.get("available") is True:
            return _foundation_compare_yardstick_payload(policy_id, max_damage, source="summary")
    return {"available": False, "opponent_policy_id": policy_id}


def _foundation_compare_best_yardstick(
    policy_id: str,
    curves: Mapping[str, list[dict[str, Any]]],
    readiness: Mapping[str, Any],
) -> dict[str, Any]:
    entry = _best_curve_entry(curves, policy_id)
    if entry is not None:
        return _foundation_compare_yardstick_payload(policy_id, entry, source="manifest")
    if policy_id == "max-damage":
        best = _optional_mapping(readiness.get("best_max_damage_yardstick"))
        if best.get("available") is True:
            return _foundation_compare_yardstick_payload(policy_id, best, source="summary")
        latest = _optional_mapping(readiness.get("max_damage_yardstick"))
        if latest.get("available") is True:
            return _foundation_compare_yardstick_payload(policy_id, latest, source="summary")
    return {"available": False, "opponent_policy_id": policy_id}


def _foundation_compare_yardstick_payload(
    policy_id: str,
    entry: Mapping[str, Any],
    *,
    source: str,
) -> dict[str, Any]:
    games = _int_or_none(entry.get("games"))
    return {
        "available": True,
        "opponent_policy_id": policy_id,
        "iteration": _int_or_none(entry.get("iteration")),
        "win_rate": _float_or_none(entry.get("win_rate")),
        "games": games,
        "capped_games": _int_or_none(entry.get("capped_games")) or 0,
        "checkpoint_path": _string_or_none(entry.get("checkpoint_path")),
        "checkpoint_policy_spec": _string_or_none(entry.get("checkpoint_policy_spec")),
        "sample_games_ready": games is not None and games >= FOUNDATION_MILESTONE_BENCHMARK_GAMES,
        "source": source,
    }


def _foundation_compare_label(summary_path: Path, recipe: Mapping[str, Any]) -> str:
    run_dir = _string_or_none(recipe.get("run_dir"))
    if run_dir is not None:
        path = Path(run_dir)
    else:
        path = summary_path.parent
    parts = path.parts
    if len(parts) >= 2:
        return str(Path(parts[-2]) / parts[-1])
    return str(path)


def _print_foundation_compare(payload: Mapping[str, Any]) -> None:
    print("neural_foundation_compare:")
    print("note: rates are candidate wins / total games; this is not an MCTS verdict.")
    print(f"candidate_source: {_format_manifest_value(payload.get('candidate_source'))}")
    quality_gate_config = _optional_mapping(payload.get("quality_gate"))
    entries = tuple(_mapping(entry) for entry in _sequence(payload.get("entries", ())))
    if not entries:
        print("entries: 0")
        return
    header = (
        f"{'label':<44} {'status':>7} {'profile':>7} {'variant':>15} {'iter':>4} "
        f"{'evidence':>24} {'gate':>5} {'max_wr':>7} {'max_g':>5} "
        f"{'val_corr':>8} {'val_sign':>8} {'val_ece':>8}"
    )
    print(header)
    print("-" * len(header))
    for entry in entries:
        yardsticks = _optional_mapping(entry.get("yardsticks"))
        max_damage = _optional_mapping(yardsticks.get("max-damage"))
        value = _optional_mapping(entry.get("value_calibration"))
        gate = _optional_mapping(entry.get("quality_gate"))
        print(
            f"{_clip_table_cell(entry.get('label'), 44):<44} "
            f"{_clip_table_cell(entry.get('status'), 7):>7} "
            f"{_clip_table_cell(entry.get('profile'), 7):>7} "
            f"{_clip_table_cell(entry.get('variant'), 15):>15} "
            f"{_format_manifest_value(entry.get('candidate_iteration')):>4} "
            f"{_clip_table_cell(entry.get('foundation_evidence_status'), 24):>24} "
            f"{_foundation_gate_status(gate):>5} "
            f"{_foundation_rate(max_damage):>7} "
            f"{_foundation_games(max_damage):>5} "
            f"{_format_optional_float(value.get('pearson_correlation'), digits=4):>8} "
            f"{_format_optional_float(value.get('sign_accuracy'), digits=4):>8} "
            f"{_format_optional_float(value.get('expected_calibration_error'), digits=4):>8}"
        )
    print("")
    print("checkpoint_sources:")
    for entry in entries:
        manifest_state = "loaded" if entry.get("manifest_loaded") is True else f"missing({_format_manifest_value(entry.get('manifest_error'))})"
        load_error = entry.get("load_error")
        candidate_error = entry.get("candidate_selection_error")
        error_suffix = f" load_error={_format_manifest_value(load_error)}" if load_error is not None else ""
        if candidate_error is not None:
            error_suffix += f" candidate_error={_format_manifest_value(candidate_error)}"
        print(
            f"- {_format_manifest_value(entry.get('label'))}: "
            f"candidate={_format_manifest_value(entry.get('candidate_checkpoint_path'))} "
            f"latest={_format_manifest_value(entry.get('latest_checkpoint_path'))} "
            f"manifest={manifest_state}"
            f"{error_suffix}"
        )
    best_entries = [
        (entry, _optional_mapping(_optional_mapping(entry.get("best_yardsticks")).get("max-damage")))
        for entry in entries
    ]
    if any(best.get("available") is True for _, best in best_entries):
        print("")
        print("best_yardsticks:")
        print("note: best fixed-yardstick rows are selection visibility; quality gates use the selected candidate source.")
        for entry, best in best_entries:
            if best.get("available") is True:
                print(
                    f"- {_format_manifest_value(entry.get('label'))}: "
                    f"max_damage={_format_foundation_yardstick_compact(best)}"
                )
            else:
                print(f"- {_format_manifest_value(entry.get('label'))}: max_damage=missing")
    if quality_gate_config.get("configured") is True:
        print("")
        print("quality_gate:")
        for entry in entries:
            gate = _optional_mapping(entry.get("quality_gate"))
            failed_checks = ", ".join(str(name) for name in _sequence(gate.get("failed_checks", ()))) or "-"
            print(
                f"- {_format_manifest_value(entry.get('label'))}: "
                f"status={_format_manifest_value(gate.get('status'))} "
                f"failed={failed_checks}"
            )


def _foundation_rate(entry: Mapping[str, Any]) -> str:
    if entry.get("available") is not True:
        return "-"
    return _format_optional_float(entry.get("win_rate"), digits=3)


def _foundation_games(entry: Mapping[str, Any]) -> str:
    if entry.get("available") is not True:
        return "-"
    return _format_manifest_value(entry.get("games"))


def _foundation_gate_status(entry: Mapping[str, Any]) -> str:
    if entry.get("configured") is not True:
        return "-"
    status = _string_or_none(entry.get("status"))
    if status == "pass":
        return "pass"
    if status == "fail":
        return "fail"
    return "-"


def _foundation_value_tune_recipe(args: argparse.Namespace) -> dict[str, Any]:
    _validate_foundation_value_tune_args(args)
    summary_path, summary = _load_foundation_summary(args.path)
    manifest, manifest_source, manifest_error = _foundation_manifest_from_summary(summary_path, summary)
    if manifest is None:
        raise ValueError(f"foundation value tune requires a loaded manifest: {manifest_error}")
    iterations = tuple(_mapping(iteration) for iteration in _sequence(manifest.get("iterations", ())))
    selected_iteration, candidate_error = _select_foundation_candidate_iteration(
        iterations=iterations,
        manifest=manifest,
        candidate_source=args.candidate_source,
    )
    if selected_iteration is None:
        raise ValueError(f"foundation candidate unavailable: {candidate_error}")
    candidate_iteration = _int_or_none(selected_iteration.get("iteration"))
    if candidate_iteration is None:
        raise ValueError("foundation candidate is missing its iteration number.")
    candidate_checkpoint = _string_or_none(selected_iteration.get("checkpoint_path"))
    if candidate_checkpoint is None:
        raise ValueError("foundation candidate is missing checkpoint_path.")
    train_paths = _foundation_iteration_paths(
        selected_iteration,
        plural_key="training_rollout_paths",
        singular_key="training_rollout_path",
    )
    if not train_paths:
        raise ValueError("foundation candidate is missing training rollout paths.")
    selection_paths = _foundation_iteration_paths(
        selected_iteration,
        plural_key="value_selection_training_rollout_paths",
        singular_key="value_selection_training_rollout_path",
    )
    if not selection_paths:
        if args.require_heldout_selection:
            raise ValueError("selected foundation candidate has no value-selection held-out rollout paths.")
        selection_paths = train_paths
    calibration_paths = list(args.calibration_data or selection_paths)
    selection_paths_fallback_to_train = selection_paths == train_paths
    calibration_reuses_selection_paths = _foundation_paths_overlap(calibration_paths, selection_paths)
    calibration_overlaps_train_paths = _foundation_paths_overlap(calibration_paths, train_paths)
    warnings = []
    if selection_paths_fallback_to_train:
        warnings.append(
            {
                "code": "selection_paths_fallback_to_train",
                "message": "Value selection is using training rollout paths; provide held-out selection paths for cleaner epoch selection.",
            }
        )
    if calibration_reuses_selection_paths:
        warnings.append(
            {
                "code": "calibration_reuses_value_selection_data",
                "message": "Value calibration is reported on the same paths used for epoch selection; treat it as selection-set calibration, not a final unbiased read.",
            }
        )
    if calibration_overlaps_train_paths and not selection_paths_fallback_to_train:
        warnings.append(
            {
                "code": "calibration_overlaps_training_data",
                "message": "Value calibration overlaps training rollout paths; treat calibration metrics as in-sample and provide independent calibration data for a final read.",
            }
        )
    recipe = _optional_mapping(summary.get("recipe"))
    run_dir = Path(
        _string_or_none(recipe.get("run_dir"))
        or _string_or_none(manifest.get("run_dir"))
        or str(summary_path.parent)
    )
    out_dir = args.out_dir or run_dir / "value-tune" / f"{args.candidate_source}-iteration-{candidate_iteration:04d}"
    artifacts = {
        "checkpoint_path": str(out_dir / "value-tuned-transformer-policy.pt"),
        "value_selection_path": str(out_dir / "value-selection.json"),
        "value_calibration_path": str(out_dir / "value-calibration.json"),
    }
    argv = [
        sys.executable,
        "-m",
        "pokezero.neural_cli",
        "train",
        "--data",
        *[str(path) for path in train_paths],
        "--out",
        artifacts["checkpoint_path"],
        "--initial-checkpoint",
        candidate_checkpoint,
        "--objective",
        "value-only",
        "--freeze-non-value-parameters",
        "--epochs",
        str(args.epochs),
        "--batch-size",
        str(args.batch_size),
        "--learning-rate",
        str(args.learning_rate),
        "--value-ranking-loss-weight",
        str(args.value_ranking_loss_weight),
        "--value-ranking-margin",
        str(args.value_ranking_margin),
        "--value-selection-data",
        *[str(path) for path in selection_paths],
        "--value-selection-metric",
        str(args.value_selection_metric),
        "--value-selection-out",
        artifacts["value_selection_path"],
        "--value-calibration-data",
        *[str(path) for path in calibration_paths],
        "--value-calibration-out",
        artifacts["value_calibration_path"],
        "--value-calibration-batch-size",
        str(args.value_calibration_batch_size),
        "--value-calibration-bins",
        str(args.value_calibration_bins),
    ]
    if args.max_batches is not None:
        argv.extend(["--max-batches", str(args.max_batches)])
    if args.device is not None:
        argv.extend(["--device", str(args.device)])
    return {
        "schema_version": NEURAL_FOUNDATION_VALUE_TUNE_PLAN_SCHEMA_VERSION,
        "source": collect_source_metadata(),
        "summary_path": str(summary_path),
        "manifest_path": manifest_source,
        "candidate_source": args.candidate_source,
        "candidate_iteration": candidate_iteration,
        "candidate_checkpoint_path": candidate_checkpoint,
        "out_dir": str(out_dir),
        "train_paths": [str(path) for path in train_paths],
        "selection_paths": [str(path) for path in selection_paths],
        "calibration_paths": [str(path) for path in calibration_paths],
        "selection_paths_fallback_to_train": selection_paths_fallback_to_train,
        "calibration_reuses_selection_paths": calibration_reuses_selection_paths,
        "calibration_overlaps_train_paths": calibration_overlaps_train_paths,
        "warnings": warnings,
        "config": {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "value_ranking_loss_weight": args.value_ranking_loss_weight,
            "value_ranking_margin": args.value_ranking_margin,
            "value_selection_metric": args.value_selection_metric,
            "value_calibration_batch_size": args.value_calibration_batch_size,
            "value_calibration_bins": args.value_calibration_bins,
            "calibration_data": [str(path) for path in args.calibration_data] if args.calibration_data else None,
            "require_heldout_selection": args.require_heldout_selection,
            "max_batches": args.max_batches,
            "device": args.device,
        },
        "artifacts": artifacts,
        "command": {
            "argv": argv,
            "shell": shlex.join(argv),
        },
    }


def _foundation_iteration_paths(
    iteration: Mapping[str, Any],
    *,
    plural_key: str,
    singular_key: str,
) -> list[Path]:
    paths = []
    for value in _sequence(iteration.get(plural_key, ())):
        path_text = _string_or_none(value)
        if path_text is not None:
            paths.append(Path(path_text))
    if paths:
        return paths
    singular = _string_or_none(iteration.get(singular_key))
    return [Path(singular)] if singular is not None else []


def _foundation_paths_overlap(left: Sequence[Path], right: Sequence[Path]) -> bool:
    return bool(_foundation_path_identities(left) & _foundation_path_identities(right))


def _foundation_path_identities(paths: Sequence[Path]) -> set[str]:
    identities: set[str] = set()
    for path in paths:
        identities.add(str(path))
        identities.add(str(path.expanduser().resolve(strict=False)))
    return identities


def _print_foundation_value_tune_warnings(recipe: Mapping[str, Any]) -> None:
    warnings = tuple(_mapping(warning) for warning in _sequence(recipe.get("warnings", ())))
    if not warnings:
        return
    print("warnings:")
    for warning in warnings:
        print(f"- {_format_manifest_value(warning.get('code'))}: {_format_manifest_value(warning.get('message'))}")


def _validate_foundation_value_tune_args(args: argparse.Namespace) -> None:
    if args.epochs <= 0:
        raise ValueError("--epochs must be positive.")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive.")
    if args.learning_rate <= 0.0 or not math.isfinite(args.learning_rate):
        raise ValueError("--learning-rate must be a positive finite value.")
    if args.value_ranking_loss_weight < 0.0:
        raise ValueError("--value-ranking-loss-weight must be non-negative.")
    if args.value_ranking_margin < 0.0:
        raise ValueError("--value-ranking-margin must be non-negative.")
    if args.value_calibration_batch_size <= 0:
        raise ValueError("--value-calibration-batch-size must be positive.")
    if args.value_calibration_bins <= 0:
        raise ValueError("--value-calibration-bins must be positive.")
    if args.max_batches is not None and args.max_batches <= 0:
        raise ValueError("--max-batches must be positive when provided.")


def _validate_foundation_value_tune_paths(out_dir: Path, *, summary_path: Path) -> None:
    if summary_path.exists():
        raise ValueError(f"summary path already exists: {summary_path}")
    if out_dir.exists() and any(out_dir.iterdir()):
        raise ValueError(f"value tune output directory already exists and is not empty: {out_dir}")


def _load_optional_json(path: Path) -> Mapping[str, Any] | None:
    if not path.exists():
        return None
    return _load_json_mapping(path)


def _load_optional_json_or_error(path: Path) -> Mapping[str, Any] | None:
    try:
        return _load_optional_json(path)
    except Exception as exc:
        return {"load_error": str(exc), "path": str(path)}


def _load_json_mapping(path: Path) -> Mapping[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"JSON file must contain an object: {path}")
    return payload


def _clip_table_cell(value: object, width: int) -> str:
    text = _format_manifest_value(value)
    if len(text) <= width:
        return text
    if width <= 3:
        return text[:width]
    return text[: width - 3] + "..."


def _foundation_recipe(args: argparse.Namespace) -> dict[str, Any]:
    resolved = _foundation_resolved_options(args)
    explicit_options = getattr(args, "_explicit_cli_options", frozenset())
    argv = [
        sys.executable,
        "-m",
        "pokezero.neural_cli",
        "iterate",
        "--run-dir",
        str(args.run_dir),
        "--iterations",
        str(resolved["iterations"]),
        "--games-per-iteration",
        str(resolved["games_per_iteration"]),
        "--workers",
        str(resolved["workers"]),
        "--showdown-root",
        str(args.showdown_root),
        "--initial-policy",
        str(resolved["initial_policy"]),
        "--experiment-preset",
        str(resolved["experiment_preset"]),
        "--epochs",
        str(resolved["epochs"]),
        "--seed-start",
        str(args.seed_start),
        "--evaluation-seed-start",
        str(args.evaluation_seed_start),
        "--json",
    ]
    if args.profile == "smoke" or "evaluation_games" in explicit_options or resolved["evaluation_interval_games"] is not None:
        argv.extend(["--evaluation-games", str(resolved["evaluation_games"])])
    if resolved["evaluation_interval_games"] is not None:
        argv.extend(["--evaluation-interval-games", str(resolved["evaluation_interval_games"])])
    if args.profile == "smoke" or "value_selection_heldout_games" in explicit_options:
        argv.extend(["--value-selection-heldout-games", str(resolved["value_selection_heldout_games"])])
    if resolved["batch_size"] is not None:
        argv.extend(["--batch-size", str(resolved["batch_size"])])
    if resolved["max_batches"] is not None:
        argv.extend(["--max-batches", str(resolved["max_batches"])])
    if resolved["learning_rate_schedule_total_games"] is not None:
        argv.extend(["--learning-rate-schedule-total-games", str(resolved["learning_rate_schedule_total_games"])])
    if resolved["learning_rate_schedule_completed_games"] is not None:
        argv.extend(
            [
                "--learning-rate-schedule-completed-games",
                str(resolved["learning_rate_schedule_completed_games"]),
            ]
        )
    if resolved["opponent_action_loss_weight"] is not None:
        argv.extend(["--opponent-action-loss-weight", str(resolved["opponent_action_loss_weight"])])
    if resolved["value_clip_range"] is not None:
        argv.extend(["--value-clip-range", str(resolved["value_clip_range"])])
    if resolved["value_ranking_loss_weight"] is not None:
        argv.extend(["--value-ranking-loss-weight", str(resolved["value_ranking_loss_weight"])])
    if resolved["value_ranking_margin"] is not None:
        argv.extend(["--value-ranking-margin", str(resolved["value_ranking_margin"])])
    for opponent_policy in _sequence_or_empty(resolved["opponent_policies"]):
        argv.extend(["--opponent-policy", str(opponent_policy)])
    if resolved["no_fixed_opponents"]:
        argv.append("--mirror-match")
        argv.append("--no-fixed-opponents")
    if resolved["temporal_aggregator"] is not None:
        argv.extend(["--temporal-aggregator", str(resolved["temporal_aggregator"])])
    for flag, key in (
        ("--embedding-dim", "embedding_dim"),
        ("--layers", "layers"),
        ("--attention-heads", "attention_heads"),
        ("--feedforward-dim", "feedforward_dim"),
        ("--dropout", "dropout"),
        ("--category-oov-buckets", "category_oov_buckets"),
    ):
        if key in explicit_options:
            argv.extend([flag, str(resolved[key])])
    if resolved["collector_advancement_mode"] is not None:
        argv.extend(["--collector-advancement-mode", str(resolved["collector_advancement_mode"])])
    if resolved["training_cache_root"] is not None:
        argv.extend(["--training-cache-root", str(resolved["training_cache_root"])])
    if resolved["training_cache_chunk_games"] is not None:
        argv.extend(["--training-cache-chunk-games", str(resolved["training_cache_chunk_games"])])
    if "max_cache_gb" in explicit_options:
        argv.extend(["--max-cache-gb", str(resolved["max_cache_gb"])])
    if not resolved["delete_cache_after_read"]:
        argv.append("--keep-cache-after-read")
    if not resolved["write_rollout_jsonl"]:
        argv.append("--omit-rollout-jsonl")
    if args.device is not None:
        argv.extend(["--device", str(args.device)])
    if args.resume:
        argv.append("--resume")
    return {
        "schema_version": NEURAL_FOUNDATION_PLAN_SCHEMA_VERSION,
        "source": collect_source_metadata(),
        "profile": args.profile,
        "variant": args.variant,
        "variant_description": str(NEURAL_FOUNDATION_VARIANTS[args.variant]["description"]),
        "experiment_contract": _foundation_experiment_contract(args, resolved),
        "run_dir": str(args.run_dir),
        "manifest_path": str(args.run_dir / "manifest.json"),
        "showdown_root": str(args.showdown_root),
        "initial_policy": str(resolved["initial_policy"]),
        "experiment_preset": str(resolved["experiment_preset"]),
        "recipe_fidelity": bool(resolved["recipe_fidelity"]),
        "recipe_fidelity_reference": recipe_fidelity_reference_config() if resolved["recipe_fidelity"] else None,
        "recipe_fidelity_unsupported_knobs": (
            dict(RECIPE_FIDELITY_UNSUPPORTED_KNOBS) if resolved["recipe_fidelity"] else None
        ),
        "effective_config_source": "nested neural manifest invocation_config after neural iterate applies the preset",
        "resolved_options": resolved,
        "command": {
            "argv": argv,
            "shell": shlex.join(argv),
        },
    }


def _foundation_resolved_options(args: argparse.Namespace) -> dict[str, Any]:
    explicit_options = getattr(args, "_explicit_cli_options", frozenset())
    profile = NEURAL_FOUNDATION_PROFILES[args.profile]
    variant = NEURAL_FOUNDATION_VARIANTS[args.variant]
    teacher_cut = bool(variant["teacher_cut"])
    continuation = _foundation_continuation_from_args(args, explicit_options=explicit_options)
    if teacher_cut:
        _validate_teacher_cut_foundation_args(args, continuation=continuation)
    opponent_action_loss_weight = (
        args.opponent_action_loss_weight
        if args.opponent_action_loss_weight is not None
        else variant["opponent_action_loss_weight"]
    )
    temporal_aggregator = (
        args.temporal_aggregator
        if args.temporal_aggregator is not None
        else variant["temporal_aggregator"]
    )
    opponent_policies = (
        tuple(str(spec) for spec in args.opponent_policy)
        if args.opponent_policy is not None
        else variant["opponent_policies"]
    )
    recipe_fidelity = bool(getattr(args, "recipe_fidelity", False))
    # Recipe-fidelity runs default to the thesis epoch count; the preset would set it, but the
    # wrapper always emits --epochs explicitly, so derive the default here to stay consistent.
    default_epochs = int(MIT_THESIS_REFERENCE_CONFIG["epochs"]) if recipe_fidelity else profile["epochs"]
    initial_policy = str(continuation["initial_policy"]) if continuation is not None else str(args.initial_policy)
    learning_rate_schedule_completed_games = args.learning_rate_schedule_completed_games
    if continuation is not None and learning_rate_schedule_completed_games is None:
        learning_rate_schedule_completed_games = int(continuation["completed_games"])
    resolved = {
        "iterations": _foundation_option(args.iterations, profile["iterations"]),
        "games_per_iteration": _foundation_option(args.games_per_iteration, profile["games_per_iteration"]),
        "workers": _foundation_option(args.workers, profile["workers"]),
        "evaluation_games": _foundation_option(args.evaluation_games, profile["evaluation_games"]),
        "evaluation_interval_games": _foundation_option(
            args.evaluation_interval_games,
            profile["evaluation_interval_games"],
        ),
        "epochs": _foundation_option(args.epochs, default_epochs),
        "batch_size": args.batch_size,
        "initial_policy": initial_policy,
        "continuation": continuation,
        "recipe_fidelity": recipe_fidelity,
        "experiment_preset": "recipe-fidelity" if recipe_fidelity else "foundation-arms-race",
        "max_batches": _foundation_max_batches(args.max_batches, profile["max_batches"]),
        "learning_rate_schedule_total_games": args.learning_rate_schedule_total_games,
        "learning_rate_schedule_completed_games": learning_rate_schedule_completed_games,
        "value_selection_heldout_games": _foundation_option(
            args.value_selection_heldout_games,
            profile["value_selection_heldout_games"],
        ),
        "opponent_action_loss_weight": opponent_action_loss_weight,
        "value_clip_range": args.value_clip_range,
        "value_ranking_loss_weight": args.value_ranking_loss_weight,
        "value_ranking_margin": args.value_ranking_margin,
        "temporal_aggregator": temporal_aggregator,
        "embedding_dim": args.embedding_dim,
        "layers": args.layers,
        "attention_heads": args.attention_heads,
        "feedforward_dim": args.feedforward_dim,
        "dropout": args.dropout,
        "category_oov_buckets": args.category_oov_buckets,
        "opponent_policies": list(opponent_policies) if opponent_policies is not None else None,
        "no_fixed_opponents": teacher_cut and opponent_policies == (),
        "collector_advancement_mode": args.collector_advancement_mode,
        "teacher_cut": teacher_cut,
        "training_cache_root": str(args.training_cache_root) if args.training_cache_root is not None else None,
        "training_cache_chunk_games": args.training_cache_chunk_games,
        "max_cache_gb": args.max_cache_gb,
        "delete_cache_after_read": bool(args.delete_cache_after_read),
        "write_rollout_jsonl": bool(args.write_rollout_jsonl),
    }
    if recipe_fidelity and resolved["learning_rate_schedule_total_games"] is None:
        completed_games = int(learning_rate_schedule_completed_games or 0)
        requested_games = int(resolved["iterations"]) * int(resolved["games_per_iteration"])
        resolved["learning_rate_schedule_total_games"] = completed_games + requested_games
    for name in ("iterations", "games_per_iteration", "workers", "evaluation_games", "epochs"):
        if int(resolved[name] or 0) <= 0:
            raise ValueError(f"{name.replace('_', '-')} must be positive.")
    if resolved["evaluation_interval_games"] is not None and int(resolved["evaluation_interval_games"]) <= 0:
        raise ValueError("evaluation-interval-games must be positive.")
    if (
        resolved["evaluation_interval_games"] is not None
        and int(resolved["evaluation_interval_games"]) > int(resolved["games_per_iteration"])
        and resolved["collector_advancement_mode"] not in (None, "always")
    ):
        raise ValueError(
            "evaluation-interval-games can skip iteration benchmarks only when "
            "collector-advancement-mode is 'always'."
        )
    if resolved["batch_size"] is not None and int(resolved["batch_size"]) <= 0:
        raise ValueError("batch-size must be positive.")
    if int(resolved["value_selection_heldout_games"] or 0) < 0:
        raise ValueError("value-selection-heldout-games must be non-negative.")
    if (
        resolved["learning_rate_schedule_total_games"] is not None
        and int(resolved["learning_rate_schedule_total_games"]) <= 0
    ):
        raise ValueError("learning-rate-schedule-total-games must be positive.")
    if (
        resolved["learning_rate_schedule_completed_games"] is not None
        and int(resolved["learning_rate_schedule_completed_games"]) < 0
    ):
        raise ValueError("learning-rate-schedule-completed-games must be non-negative.")
    if (
        resolved["learning_rate_schedule_total_games"] is not None
        and resolved["learning_rate_schedule_completed_games"] is not None
        and int(resolved["learning_rate_schedule_completed_games"]) >= int(resolved["learning_rate_schedule_total_games"])
    ):
        raise ValueError(
            "learning-rate-schedule-completed-games must be less than "
            "learning-rate-schedule-total-games; when continuing a run, set the total to the new global game total."
        )
    if (
        resolved["learning_rate_schedule_total_games"] is not None
        and resolved["learning_rate_schedule_completed_games"] is not None
        and (
            int(resolved["learning_rate_schedule_completed_games"])
            + (int(resolved["iterations"]) * int(resolved["games_per_iteration"]))
            > int(resolved["learning_rate_schedule_total_games"])
        )
    ):
        raise ValueError(
            "learning-rate-schedule-total-games must cover completed games plus requested foundation games; "
            "set the total to the new global game total for continuation runs."
        )
    if resolved["opponent_action_loss_weight"] is not None and float(resolved["opponent_action_loss_weight"]) < 0.0:
        raise ValueError("opponent-action-loss-weight must be non-negative.")
    if resolved["value_clip_range"] is not None and float(resolved["value_clip_range"]) <= 0.0:
        raise ValueError("value-clip-range must be positive.")
    if resolved["value_ranking_loss_weight"] is not None and float(resolved["value_ranking_loss_weight"]) < 0.0:
        raise ValueError("value-ranking-loss-weight must be non-negative.")
    if resolved["value_ranking_margin"] is not None and float(resolved["value_ranking_margin"]) < 0.0:
        raise ValueError("value-ranking-margin must be non-negative.")
    if resolved["temporal_aggregator"] is not None and resolved["temporal_aggregator"] not in {"mean", "gru"}:
        raise ValueError("temporal-aggregator must be 'mean' or 'gru'.")
    if resolved["embedding_dim"] is not None and int(resolved["embedding_dim"]) <= 0:
        raise ValueError("embedding-dim must be positive.")
    if resolved["layers"] is not None and int(resolved["layers"]) < 0:
        raise ValueError("layers must be non-negative.")
    if resolved["attention_heads"] is not None and int(resolved["attention_heads"]) <= 0:
        raise ValueError("attention-heads must be positive.")
    if resolved["feedforward_dim"] is not None and int(resolved["feedforward_dim"]) <= 0:
        raise ValueError("feedforward-dim must be positive.")
    if resolved["dropout"] is not None and not 0.0 <= float(resolved["dropout"]) < 1.0:
        raise ValueError("dropout must be >= 0 and < 1.")
    if resolved["category_oov_buckets"] is not None and int(resolved["category_oov_buckets"]) <= 0:
        raise ValueError("category-oov-buckets must be positive.")
    effective_embedding_dim = int(resolved["embedding_dim"] if resolved["embedding_dim"] is not None else 128)
    effective_attention_heads = int(resolved["attention_heads"] if resolved["attention_heads"] is not None else 4)
    if effective_embedding_dim % effective_attention_heads != 0:
        raise ValueError("embedding-dim must be divisible by attention-heads.")
    if (
        resolved["collector_advancement_mode"] is not None
        and resolved["collector_advancement_mode"] not in COLLECTOR_ADVANCEMENT_MODES
    ):
        raise ValueError("collector-advancement-mode is invalid.")
    if resolved["training_cache_chunk_games"] is not None and int(resolved["training_cache_chunk_games"]) <= 0:
        raise ValueError("training-cache-chunk-games must be positive.")
    _cache_gb_to_bytes(float(resolved["max_cache_gb"]))
    if resolved["training_cache_chunk_games"] is not None and resolved["training_cache_root"] is None:
        raise ValueError("--training-cache-chunk-games requires --training-cache-root.")
    if not resolved["write_rollout_jsonl"] and resolved["training_cache_root"] is None:
        raise ValueError("--omit-rollout-jsonl requires --training-cache-root.")
    opponent_policy_values = _sequence_or_empty(resolved["opponent_policies"])
    if any(not str(policy).strip() for policy in opponent_policy_values):
        raise ValueError("opponent-policy entries must be non-empty.")
    return resolved


def _foundation_continuation_from_args(
    args: argparse.Namespace,
    *,
    explicit_options: frozenset[str],
) -> dict[str, Any] | None:
    if args.continue_from is None:
        return None
    if "initial_policy" in explicit_options:
        raise ValueError("--continue-from cannot be combined with explicit --initial-policy.")
    continuation = _load_foundation_continuation(args.continue_from)
    if args.learning_rate_schedule_completed_games is not None and int(args.learning_rate_schedule_completed_games) != int(
        continuation["completed_games"]
    ):
        raise ValueError(
            "--learning-rate-schedule-completed-games must match --continue-from completed games "
            f"({continuation['completed_games']}) or be omitted."
        )
    return continuation


def _load_foundation_continuation(path: Path) -> dict[str, Any]:
    errors: list[str] = []
    try:
        summary_path, summary = _load_foundation_summary(path)
        manifest, manifest_source, manifest_error = _foundation_manifest_from_summary(summary_path, summary)
        if manifest is not None:
            return _foundation_continuation_from_manifest(
                manifest,
                source_path=summary_path,
                source_kind="foundation-summary",
                manifest_source=manifest_source,
            )
        errors.append(f"foundation summary manifest unavailable: {manifest_error}")
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        errors.append(f"foundation summary load failed: {exc}")

    run_dir = path.parent if path.name == "manifest.json" else path
    try:
        manifest = load_neural_selfplay_run_manifest(run_dir)
        return _foundation_continuation_from_manifest(
            manifest,
            source_path=run_dir,
            source_kind="neural-selfplay-run",
            manifest_source=str(run_dir / "manifest.json"),
        )
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        errors.append(f"neural run manifest load failed: {exc}")

    details = "; ".join(errors)
    raise ValueError(
        "--continue-from must point to a neural foundation summary, foundation run directory, "
        f"or neural self-play run directory. {details}"
    )


def _foundation_continuation_from_manifest(
    manifest: Mapping[str, Any],
    *,
    source_path: Path,
    source_kind: str,
    manifest_source: str,
) -> dict[str, Any]:
    iterations = tuple(_mapping(iteration) for iteration in _sequence(manifest.get("iterations", ())))
    if not iterations:
        raise ValueError("--continue-from manifest contains no iterations.")
    current_policy_spec = _string_or_none(manifest.get("current_policy_spec"))
    latest_checkpoint = _string_or_none(manifest.get("latest_checkpoint_path"))
    initial_policy = current_policy_spec if _is_learned_policy_spec(current_policy_spec) else None
    if initial_policy is None and latest_checkpoint is not None:
        initial_policy = f"neural:{latest_checkpoint}"
    if initial_policy is None:
        raise ValueError("--continue-from manifest does not expose a learned current policy or latest checkpoint.")
    checkpoint_path = _checkpoint_path_from_policy_spec(initial_policy) or latest_checkpoint
    completed_games = _manifest_learning_rate_schedule_completed_games(manifest) + sum(
        int(_mapping(iteration.get("collection_metrics", {})).get("games", 0))
        for iteration in iterations
    )
    if completed_games <= 0:
        raise ValueError("--continue-from manifest has no completed collection games.")
    return {
        "source_path": str(source_path),
        "source_kind": source_kind,
        "manifest_source": manifest_source,
        "completed_iterations": len(iterations),
        "completed_games": completed_games,
        "initial_policy": initial_policy,
        "checkpoint_path": checkpoint_path,
    }


def _manifest_learning_rate_schedule_completed_games(manifest: Mapping[str, Any]) -> int:
    for config in reversed(tuple(_mapping(config) for config in _sequence(manifest.get("invocation_configs", ())))):
        value = config.get("learning_rate_schedule_completed_games")
        if value is None:
            continue
        parsed = int(value)
        if parsed < 0:
            raise ValueError("manifest learning_rate_schedule_completed_games must be non-negative.")
        return parsed
    legacy_config = manifest.get("run_config")
    if legacy_config is not None:
        value = _mapping(legacy_config).get("learning_rate_schedule_completed_games")
        if value is not None:
            parsed = int(value)
            if parsed < 0:
                raise ValueError("manifest learning_rate_schedule_completed_games must be non-negative.")
            return parsed
    return 0


def _is_learned_policy_spec(policy_spec: str | None) -> bool:
    if policy_spec is None:
        return False
    return policy_spec.startswith("neural:") or policy_spec.startswith("linear:")


def _validate_teacher_cut_foundation_args(args: argparse.Namespace, *, continuation: Mapping[str, Any] | None = None) -> None:
    if args.opponent_policy is not None:
        raise ValueError("--variant teacher-cut does not allow fixed --opponent-policy training opponents.")
    initial_policy = str(continuation["initial_policy"]) if continuation is not None else str(args.initial_policy)
    initial_body = _policy_spec_name(initial_policy)
    if (
        initial_body in FOUNDATION_TEACHER_CUT_ALLOWED_INITIAL_POLICY_NAMES
        or initial_body.startswith(FOUNDATION_TEACHER_CUT_LEARNED_INITIAL_PREFIXES)
    ):
        return
    allowed = ", ".join(
        sorted(FOUNDATION_TEACHER_CUT_ALLOWED_INITIAL_POLICY_NAMES)
        + [f"{prefix}/path/to/checkpoint" for prefix in FOUNDATION_TEACHER_CUT_LEARNED_INITIAL_PREFIXES]
    )
    raise ValueError(
        f"--variant teacher-cut initial policy must be random-legal or a learned checkpoint spec; "
        f"got {initial_body!r}. Allowed forms: {allowed}."
    )


def _foundation_reward_signal_contract(args: argparse.Namespace, resolved: Mapping[str, Any]) -> str:
    # Foundation-plan does not expose reward-shaping flags. If that changes, keep this contract
    # derived from resolved options rather than letting the recipe overstate the experiment.
    return "game_outcome_only"


def _foundation_eval_yardstick_contract(args: argparse.Namespace, resolved: Mapping[str, Any]) -> str:
    return "max-damage"


def _foundation_teacher_cut_allowed_initial_policy_forms() -> list[str]:
    return (
        sorted(FOUNDATION_TEACHER_CUT_ALLOWED_INITIAL_POLICY_NAMES)
        + [f"{prefix}/path/to/checkpoint" for prefix in FOUNDATION_TEACHER_CUT_LEARNED_INITIAL_PREFIXES]
    )


def _foundation_experiment_contract(args: argparse.Namespace, resolved: Mapping[str, Any]) -> dict[str, Any]:
    teacher_cut = bool(resolved.get("teacher_cut"))
    recipe_fidelity = bool(resolved.get("recipe_fidelity"))
    contract: dict[str, Any] = {
        "name": "teacher-cut" if teacher_cut else "foundation-arms-race",
        "teacher_cut": teacher_cut,
        "recipe_fidelity": recipe_fidelity,
    }
    if not teacher_cut:
        return contract
    return {
        **contract,
        "goal": "test whether PPO self-play can exceed the scripted-teacher ceiling after one-shot initialization",
        "teacher_allowed_as_initial_checkpoint_only": True,
        "live_initial_policy": str(resolved["initial_policy"]),
        "allowed_live_initial_policy_forms": _foundation_teacher_cut_allowed_initial_policy_forms(),
        "continuation": resolved.get("continuation"),
        "fixed_training_opponents": [],
        "uses_mirror_self_play": True,
        "collector_advancement_mode": resolved.get("collector_advancement_mode")
        or FOUNDATION_ARMS_RACE_PRESET_DEFAULTS["collector_advancement_mode"],
        "reward_signal": _foundation_reward_signal_contract(args, resolved),
        "eval_yardstick": _foundation_eval_yardstick_contract(args, resolved),
        "strength_claim_min_games": FOUNDATION_MILESTONE_BENCHMARK_GAMES,
    }


def _policy_spec_name(policy_spec: str) -> str:
    return str(policy_spec).strip().partition("?")[0].strip().lower()


def _sequence_or_empty(value: Any) -> tuple[Any, ...]:
    if value is None:
        return ()
    return tuple(_sequence(value))


def _foundation_option(value: int | None, default: int | None) -> int | None:
    return default if value is None else value


def _foundation_max_batches(value: int | None, default: int | None) -> int | None:
    resolved = default if value is None else value
    if resolved == -1:
        return None
    if resolved is not None and resolved <= 0:
        raise ValueError("max-batches must be positive, or -1 for no cap.")
    return resolved


def _validate_foundation_run_paths(run_dir: Path, *, summary_path: Path, resume: bool) -> None:
    if summary_path.exists() and not resume:
        raise ValueError(f"summary path already exists: {summary_path}")
    if run_dir.exists() and not resume:
        raise ValueError(f"run directory already exists: {run_dir}; use --resume or choose a fresh --run-dir.")


def _recipe_knob_aligned(value: Any, reference: float | int) -> bool:
    if value is None:
        return False
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return False
    if not math.isfinite(numeric):
        return False
    # Tight tolerance on purpose: the thesis distinguishes near-identical values (e.g. gamma
    # 0.9999 vs undiscounted 1.0), so only float-representation noise should be absorbed.
    return math.isclose(numeric, float(reference), rel_tol=1e-6, abs_tol=1e-9)


def recipe_fidelity_audit(
    training_config: Mapping[str, Any] | None,
    *,
    collection_temperature: Any = None,
    learning_rate_schedule_total_games_reference: int = MIT_THESIS_REFERENCE_TRAINING_GAMES,
    learning_rate_schedule_total_games_reference_basis: str = "mit_thesis_training_budget",
) -> dict[str, Any]:
    """Compare an actual training config against the MIT thesis reference recipe.

    This makes "is this run actually recipe-fidelity, or just named that way?" answerable from
    concrete numbers. It checks the knobs our config can express (Table A.3), reports which
    diverge, and always lists the knobs the codebase cannot yet express faithfully so an aligned
    verdict is never read as fully on-recipe.
    """
    config = training_config or {}
    knobs: dict[str, Any] = {}
    off_recipe: list[str] = []

    objective = config.get("objective")
    objective_ok = objective == "ppo"
    knobs["objective"] = {"value": objective, "reference": "ppo", "aligned": objective_ok}
    if not objective_ok:
        off_recipe.append("objective")

    target_mode = config.get("ppo_target_mode")
    target_ok = target_mode == "gae"
    knobs["ppo_target_mode"] = {"value": target_mode, "reference": "gae", "aligned": target_ok}
    if not target_ok:
        off_recipe.append("ppo_target_mode")

    schedule = config.get("learning_rate_schedule")
    schedule_ok = schedule == MIT_THESIS_REFERENCE_LEARNING_RATE_SCHEDULE
    knobs["learning_rate_schedule"] = {
        "value": schedule,
        "reference": MIT_THESIS_REFERENCE_LEARNING_RATE_SCHEDULE,
        "aligned": schedule_ok,
    }
    if not schedule_ok:
        off_recipe.append("learning_rate_schedule")

    schedule_total_games = config.get("learning_rate_schedule_total_games")
    schedule_total_games_ok = _recipe_knob_aligned(
        schedule_total_games, learning_rate_schedule_total_games_reference
    )
    knobs["learning_rate_schedule_total_games"] = {
        "value": schedule_total_games,
        "reference": learning_rate_schedule_total_games_reference,
        "reference_basis": learning_rate_schedule_total_games_reference_basis,
        "aligned": schedule_total_games_ok,
    }
    if not schedule_total_games_ok:
        off_recipe.append("learning_rate_schedule_total_games")

    for name, reference in MIT_THESIS_REFERENCE_CONFIG.items():
        value = config.get(name)
        aligned = _recipe_knob_aligned(value, reference)
        knobs[name] = {"value": value, "reference": reference, "aligned": aligned}
        if not aligned:
            off_recipe.append(name)

    temperature_aligned = _recipe_knob_aligned(
        collection_temperature, MIT_THESIS_REFERENCE_COLLECTION_TEMPERATURE
    )
    knobs["collection_temperature"] = {
        "value": collection_temperature,
        "reference": MIT_THESIS_REFERENCE_COLLECTION_TEMPERATURE,
        "aligned": temperature_aligned,
    }
    if not temperature_aligned:
        off_recipe.append("collection_temperature")

    aligned = not off_recipe
    unsupported_knobs = dict(RECIPE_FIDELITY_UNSUPPORTED_KNOBS)
    return {
        "reference": "mit_thesis_table_a3",
        "aligned": aligned,
        "knobs": knobs,
        "off_recipe": off_recipe,
        "unsupported_knobs": unsupported_knobs,
        "fully_on_recipe": aligned and not unsupported_knobs,
        "note": (
            "aligned=true means the configured Table A.3 PPO knobs match; fully_on_recipe is true "
            "only when no currently unsupported config knobs remain. Recipe scale (~3M battles), "
            "update cadence, and architecture are tracked separately from this config audit."
        ),
    }


def _iteration_training_config(iteration: Mapping[str, Any]) -> Mapping[str, Any] | None:
    training = _optional_mapping(iteration.get("training"))
    if not training:
        return None
    config = _optional_mapping(training.get("config"))
    return config or None


def _latest_invocation_config(manifest: Mapping[str, Any]) -> Mapping[str, Any] | None:
    configs = tuple(_optional_mapping(config) for config in _sequence(manifest.get("invocation_configs", ())))
    for config in reversed(configs):
        if config:
            return config
    return None


def _manifest_configured_training_config(manifest: Mapping[str, Any]) -> Mapping[str, Any] | None:
    invocation = _latest_invocation_config(manifest)
    if invocation is not None:
        config = _optional_mapping(invocation.get("training_config"))
        if config:
            return config
    iterations = tuple(_mapping(iteration) for iteration in _sequence(manifest.get("iterations", ())))
    if not iterations:
        return None
    return _iteration_training_config(iterations[-1])


def _manifest_collection_temperature(manifest: Mapping[str, Any]) -> Any:
    invocation = _latest_invocation_config(manifest)
    if invocation is not None and "collection_temperature" in invocation:
        return invocation.get("collection_temperature")
    return None


def _manifest_learning_rate_schedule_total_games_reference(manifest: Mapping[str, Any]) -> tuple[int, str]:
    invocation = _latest_invocation_config(manifest)
    if invocation is None:
        return MIT_THESIS_REFERENCE_TRAINING_GAMES, "mit_thesis_training_budget"
    try:
        iterations = int(invocation.get("iterations_requested"))
        games_per_iteration = int(invocation.get("games_per_iteration"))
        completed_games = int(invocation.get("learning_rate_schedule_completed_games") or 0)
    except (TypeError, ValueError):
        return MIT_THESIS_REFERENCE_TRAINING_GAMES, "mit_thesis_training_budget"
    scheduled_total_games = completed_games + (iterations * games_per_iteration)
    if scheduled_total_games <= 0:
        return MIT_THESIS_REFERENCE_TRAINING_GAMES, "mit_thesis_training_budget"
    training_config = _manifest_configured_training_config(manifest) or {}
    try:
        configured_total_games = int(training_config.get("learning_rate_schedule_total_games"))
    except (TypeError, ValueError):
        configured_total_games = None
    scheduled_total_candidates = {scheduled_total_games}
    collected_games = _manifest_collected_training_games(manifest)
    if collected_games is not None:
        scheduled_total_candidates.add(completed_games + collected_games)
    if (
        configured_total_games in scheduled_total_candidates
        and configured_total_games != MIT_THESIS_REFERENCE_TRAINING_GAMES
    ):
        return configured_total_games, "scheduled_run_full_sweep"
    return MIT_THESIS_REFERENCE_TRAINING_GAMES, "mit_thesis_training_budget"


def _manifest_collected_training_games(manifest: Mapping[str, Any]) -> int | None:
    iterations = tuple(_mapping(iteration) for iteration in _sequence(manifest.get("iterations", ())))
    if not iterations:
        return None
    total = 0
    for iteration in iterations:
        metrics = _optional_mapping(iteration.get("collection_metrics"))
        if not metrics or "games" not in metrics:
            return None
        try:
            games = int(metrics["games"])
        except (TypeError, ValueError):
            return None
        if games < 0:
            return None
        total += games
    return total


def _manifest_recipe_fidelity_audit(manifest: Mapping[str, Any]) -> dict[str, Any] | None:
    iterations = tuple(_mapping(iteration) for iteration in _sequence(manifest.get("iterations", ())))
    if not iterations:
        return None
    training_config = _manifest_configured_training_config(manifest)
    if training_config is None:
        return None
    schedule_reference, schedule_reference_basis = _manifest_learning_rate_schedule_total_games_reference(manifest)
    audit = recipe_fidelity_audit(
        training_config,
        collection_temperature=_manifest_collection_temperature(manifest),
        learning_rate_schedule_total_games_reference=schedule_reference,
        learning_rate_schedule_total_games_reference_basis=schedule_reference_basis,
    )
    audit["iteration"] = int(iterations[-1].get("iteration", 0))
    return audit


def _foundation_run_derived_report(run_dir: Path, stdout: str) -> dict[str, Any]:
    manifest, source, error = _foundation_manifest_from_run(run_dir, stdout)
    if manifest is None:
        return {
            "manifest_available": False,
            "manifest_source": source,
            "manifest_error": error,
            "latest_checkpoint_path": None,
            "foundation_readiness": None,
            "recipe_fidelity": None,
        }
    iterations = tuple(_mapping(iteration) for iteration in _sequence(manifest.get("iterations", ())))
    return {
        "manifest_available": True,
        "manifest_source": source,
        "manifest_error": None,
        "latest_checkpoint_path": manifest.get("latest_checkpoint_path"),
        "latest_iteration": int(iterations[-1].get("iteration", 0)) if iterations else None,
        "foundation_readiness": _foundation_readiness_report(iterations),
        "recipe_fidelity": _manifest_recipe_fidelity_audit(manifest),
    }


def _foundation_manifest_from_run(run_dir: Path, stdout: str) -> tuple[Mapping[str, Any] | None, str, str | None]:
    manifest_path = run_dir / "manifest.json"
    if manifest_path.exists():
        try:
            return json.loads(manifest_path.read_text(encoding="utf-8")), str(manifest_path), None
        except Exception as exc:
            return None, str(manifest_path), str(exc)
    try:
        payload = json.loads(stdout)
    except Exception as exc:
        return None, "stdout", str(exc)
    if not isinstance(payload, Mapping):
        return None, "stdout", "stdout JSON was not an object"
    return payload, "stdout", None


def _load_foundation_summary(path: Path) -> tuple[Path, Mapping[str, Any]]:
    summary_path = path / "neural-foundation-run-summary.json" if path.is_dir() else path
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"foundation summary must be a JSON object: {summary_path}")
    if payload.get("schema_version") != NEURAL_FOUNDATION_RUN_SUMMARY_SCHEMA_VERSION:
        raise ValueError(f"unsupported foundation summary schema: {payload.get('schema_version')!r}")
    return summary_path, payload


def _text_tail(value: str | None, *, limit: int = 4000) -> str:
    if not value:
        return ""
    return value if len(value) <= limit else value[-limit:]


def _utc_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def _auto_promotion_config_from_args(args: argparse.Namespace) -> NeuralSelfPlayPromotionConfig | None:
    if not args.auto_promote:
        return None
    gate_args = argparse.Namespace(**vars(args))
    gate_args.registry = None
    label_prefix = args.promotion_label_prefix if args.promotion_label_prefix else None
    return NeuralSelfPlayPromotionConfig(
        registry_path=args.promotion_registry,
        gate_config=_gate_config_from_args(gate_args),
        artifact_dir=args.promotion_artifact_dir,
        label_prefix=label_prefix,
        notes=args.promotion_notes,
        allow_duplicate=args.allow_duplicate_promotion,
    )


def _value_calibration_config_from_args(args: argparse.Namespace) -> NeuralValueCalibrationConfig | None:
    if not args.value_calibration:
        return None
    return NeuralValueCalibrationConfig(
        scope=args.value_calibration_scope,
        batch_size=args.value_calibration_batch_size,
        bins=args.value_calibration_bins,
    )


def _value_selection_config_from_args(args: argparse.Namespace) -> NeuralValueSelectionConfig | None:
    if not args.value_selection and args.value_selection_heldout_games <= 0:
        return None
    return NeuralValueSelectionConfig(
        scope=args.value_selection_scope,
        metric=args.value_selection_metric,
        batch_size=args.value_selection_batch_size,
        bins=args.value_selection_bins,
        heldout_games_per_iteration=args.value_selection_heldout_games,
        heldout_seed_start=args.value_selection_seed_start,
    )


def _print_iterate_summary(result) -> None:
    print(f"run_dir: {result.run_dir}")
    for iteration in result.iterations:
        final_epoch = iteration.training.final_metrics
        ppo_diagnostics = _format_live_ppo_diagnostics(final_epoch)
        print(
            f"iteration={iteration.iteration} games={iteration.metrics.games} "
            f"checkpoint={iteration.checkpoint_path} "
            f"loss={final_epoch.loss:.6f} "
            f"policy_accuracy={final_epoch.policy_accuracy:.4f} "
            f"promotion={_promotion_status(getattr(iteration, 'promotion', None))}"
            f"{ppo_diagnostics}"
        )
        value_selection = getattr(iteration, "value_selection", None)
        if value_selection is not None:
            print(
                "value_selection="
                f"epoch={value_selection.get('selected_epoch')} "
                f"metric={value_selection.get('metric')} "
                f"value={float(value_selection.get('selected_metric_value')):.6f} "
                f"artifact={value_selection.get('artifact_path')}"
            )
        if iteration.benchmark is not None:
            print(f"benchmark_total_games={iteration.benchmark.total_games}")
    if result.latest_checkpoint_path is not None:
        print(f"latest_checkpoint: {result.latest_checkpoint_path}")
    print(f"manifest: {result.run_dir / 'manifest.json'}")


def _format_live_ppo_diagnostics(final_epoch: Any) -> str:
    ppo_valid_fraction = getattr(final_epoch, "ppo_valid_fraction", None)
    ppo_clip_fraction = getattr(final_epoch, "ppo_clip_fraction", None)
    ppo_entropy = getattr(final_epoch, "ppo_entropy", None)
    if ppo_valid_fraction is None and ppo_clip_fraction is None and ppo_entropy is None:
        return ""
    return (
        f" ppo_cov={_format_optional_float(ppo_valid_fraction)}"
        f" ppo_clip={_format_optional_float(ppo_clip_fraction)}"
        f" ppo_ent={_format_optional_float(ppo_entropy)}"
    )


def _report(args: argparse.Namespace) -> int:
    manifest = load_neural_selfplay_run_manifest(args.run_dir)
    if args.json:
        print(json.dumps(manifest, indent=2, sort_keys=True))
        return 0
    _print_manifest_report(manifest)
    return 0


def _print_manifest_report(manifest: Mapping[str, Any]) -> None:
    iterations = tuple(_mapping(iteration) for iteration in _sequence(manifest.get("iterations", ())))
    print(f"run_dir: {manifest.get('run_dir')}")
    print(f"current_policy: {_format_manifest_value(manifest.get('current_policy_spec'))}")
    print(f"latest_checkpoint: {_format_manifest_value(manifest.get('latest_checkpoint_path'))}")
    print(f"latest_accepted_checkpoint: {_format_manifest_value(manifest.get('latest_accepted_checkpoint_path'))}")
    _print_source_metadata(_manifest_source_metadata(manifest))
    print(f"iterations: {len(iterations)}")
    if not iterations:
        return
    print("note: gate win rate is the advancement-comparison win rate; blended benchmark win rate is broad health.")
    print("")
    header = (
        f"{'iter':>4} {'games':>5} {'cap':>4} {'bench_wr':>8} {'gate_wr':>8} {'advance':>7} {'promo':>8} "
        f"{'loss':>10} {'pol_acc':>8} {'value':>10} {'sel_ep':>6} {'val_sign':>8} {'val_ece':>10} {'opp_acc':>8} "
        f"{'ppo_cov':>8} {'ppo_clip':>8} {'ppo_ent':>8} checkpoint"
    )
    print(header)
    print("-" * len(header))
    for iteration in iterations:
        metrics = _mapping(iteration.get("collection_metrics", {}))
        final_epoch = _final_epoch_metrics(iteration)
        advancement = _optional_mapping(iteration.get("advancement"))
        calibration_report = _iteration_value_calibration_report(iteration)
        value_selection = _optional_mapping(iteration.get("value_selection"))
        print(
            f"{int(iteration.get('iteration', 0)):4d} "
            f"{int(metrics.get('games', 0)):5d} "
            f"{int(metrics.get('capped_games', 0)):4d} "
            f"{_format_optional_float(_benchmark_win_rate(iteration)):>8} "
            f"{_format_optional_float(_incumbent_win_rate(iteration)):>8} "
            f"{_format_bool(advancement.get('advance_collector')):>7} "
            f"{_manifest_promotion_status(iteration):>8} "
            f"{_format_optional_float(final_epoch.get('loss') if final_epoch else None, digits=6):>10} "
            f"{_format_optional_float(final_epoch.get('policy_accuracy') if final_epoch else None, digits=4):>8} "
            f"{_format_optional_float(final_epoch.get('value_loss') if final_epoch else None, digits=6):>10} "
            f"{_format_manifest_value(value_selection.get('selected_epoch') if value_selection else None):>6} "
            f"{_format_optional_float(calibration_report.get('sign_accuracy') if calibration_report else None, digits=4):>8} "
            f"{_format_optional_float(calibration_report.get('expected_calibration_error') if calibration_report else None, digits=6):>10} "
            f"{_format_optional_float(final_epoch.get('opponent_accuracy') if final_epoch else None, digits=4):>8} "
            f"{_format_optional_float(final_epoch.get('ppo_valid_fraction') if final_epoch else None):>8} "
            f"{_format_optional_float(final_epoch.get('ppo_clip_fraction') if final_epoch else None):>8} "
            f"{_format_optional_float(final_epoch.get('ppo_entropy') if final_epoch else None):>8} "
            f"{iteration.get('checkpoint_path')}"
        )
    _print_benchmark_opponent_curves(iterations)
    _print_benchmark_plumbing_curves(iterations)
    _print_foundation_readiness(iterations)
    _print_recipe_fidelity(manifest)


def _print_recipe_fidelity(manifest: Mapping[str, Any]) -> None:
    audit = _manifest_recipe_fidelity_audit(manifest)
    print("")
    print("recipe_fidelity:")
    if audit is None:
        print("- training_config_unavailable")
        return
    print("note: config-fidelity vs the MIT thesis Table A.3; scale (~3M battles) is tracked separately.")
    print(
        f"- aligned: {_format_bool(audit.get('aligned'))} "
        f"(latest iteration {_format_manifest_value(audit.get('iteration'))})"
    )
    knobs = _optional_mapping(audit.get("knobs"))
    for name in sorted(knobs):
        knob = _optional_mapping(knobs[name])
        flag = "ok" if knob.get("aligned") is True else "OFF"
        reference_suffix = ""
        if knob.get("reference_basis"):
            reference_suffix = f"; basis {_format_manifest_value(knob.get('reference_basis'))}"
        print(
            f"  [{flag:>3}] {name}: {_format_manifest_value(knob.get('value'))} "
            f"(ref {_format_manifest_value(knob.get('reference'))}{reference_suffix})"
        )
    unsupported = _optional_mapping(audit.get("unsupported_knobs"))
    if unsupported:
        print(f"- unsupported (off-recipe by construction): {', '.join(sorted(unsupported))}")


def _final_epoch_metrics(iteration: Mapping[str, Any]) -> Mapping[str, Any] | None:
    training = _mapping(iteration.get("training", {}))
    epochs = tuple(_mapping(epoch) for epoch in _sequence(training.get("epochs", ())))
    return epochs[-1] if epochs else None


def _iteration_value_calibration_report(iteration: Mapping[str, Any]) -> Mapping[str, Any] | None:
    calibration = _optional_mapping(iteration.get("value_calibration"))
    if not calibration:
        return None
    report = _optional_mapping(calibration.get("report"))
    return report if report else None


def _print_foundation_readiness(iterations: tuple[Mapping[str, Any], ...]) -> None:
    report = _foundation_readiness_report(iterations)
    print("")
    print("foundation_readiness:")
    print("note: presence/sample-size only; inspect value quality and strength separately.")
    calibration = _optional_mapping(report.get("value_calibration"))
    if calibration.get("available") is True:
        print(
            "- value_calibration: present "
            f"examples={_format_manifest_value(calibration.get('examples'))} "
            f"sign={_format_optional_float(calibration.get('sign_accuracy'), digits=4)} "
            f"ece={_format_optional_float(calibration.get('expected_calibration_error'), digits=6)} "
            f"corr={_format_optional_float(calibration.get('pearson_correlation'), digits=4)}"
        )
    else:
        print("- value_calibration: missing")
    max_damage = _optional_mapping(report.get("max_damage_yardstick"))
    if max_damage.get("available") is True:
        games = max_damage.get("games")
        sample_state = (
            "milestone"
            if max_damage.get("sample_games_ready") is True
            else f"below_milestone({games}/{FOUNDATION_MILESTONE_BENCHMARK_GAMES})"
        )
        print(
            "- max_damage_yardstick: "
            f"iter={_format_manifest_value(max_damage.get('iteration'))} "
            f"win_rate={_format_optional_float(max_damage.get('win_rate'), digits=3)} "
            f"games={_format_manifest_value(games)} "
            f"cap={_format_manifest_value(max_damage.get('capped_games'))} "
            f"sample={sample_state}"
        )
    else:
        print("- max_damage_yardstick: missing")
    best_max_damage = _optional_mapping(report.get("best_max_damage_yardstick"))
    if best_max_damage.get("available") is True:
        print(f"- best_max_damage_yardstick: {_format_foundation_yardstick_compact(best_max_damage)}")
    else:
        print("- best_max_damage_yardstick: missing")
    print(f"- foundation_evidence_status: {_format_manifest_value(report.get('foundation_evidence_status'))}")
    reasons = report.get("reasons")
    if isinstance(reasons, list) and reasons:
        print(f"  reasons: {', '.join(str(reason) for reason in reasons)}")


def _foundation_readiness_report(iterations: tuple[Mapping[str, Any], ...]) -> dict[str, Any]:
    latest = iterations[-1] if iterations else {}
    calibration_report = _iteration_value_calibration_report(latest)
    curves = _benchmark_opponent_curves((latest,)) if latest else {}
    max_damage_entry = _latest_curve_entry(curves, "max-damage")
    all_curves = _benchmark_opponent_curves(iterations) if iterations else {}
    best_max_damage_entry = _best_curve_entry(all_curves, "max-damage")
    reasons: list[str] = []
    if calibration_report is None:
        reasons.append("value_calibration_missing")
    if max_damage_entry is None:
        reasons.append("max_damage_yardstick_missing")
    elif int(max_damage_entry.get("games", 0)) < FOUNDATION_MILESTONE_BENCHMARK_GAMES:
        reasons.append("max_damage_sample_below_milestone")
    return {
        "latest_iteration": int(latest.get("iteration", 0)) if latest else None,
        "milestone_benchmark_games": FOUNDATION_MILESTONE_BENCHMARK_GAMES,
        "value_calibration": _foundation_value_calibration_payload(calibration_report),
        "max_damage_yardstick": _foundation_yardstick_payload(max_damage_entry),
        "best_max_damage_yardstick": _foundation_yardstick_payload(best_max_damage_entry),
        "foundation_evidence_status": "present_and_sample_sized" if not reasons else "incomplete",
        "reasons": reasons,
    }


def _foundation_value_calibration_payload(report: Mapping[str, Any] | None) -> dict[str, Any]:
    if report is None:
        return {"available": False}
    return {
        "available": True,
        "examples": _int_or_none(report.get("examples")),
        "sign_accuracy": _float_or_none(report.get("sign_accuracy")),
        "expected_calibration_error": _float_or_none(report.get("expected_calibration_error")),
        "pearson_correlation": _float_or_none(report.get("pearson_correlation")),
        "mse": _float_or_none(report.get("mse")),
        "mae": _float_or_none(report.get("mae")),
        "bias": _float_or_none(report.get("bias")),
    }


def _foundation_yardstick_payload(entry: Mapping[str, Any] | None) -> dict[str, Any]:
    if entry is None:
        return {"available": False, "sample_games_ready": False}
    games = _int_or_none(entry.get("games")) or 0
    return {
        "available": True,
        "opponent_policy_id": "max-damage",
        "iteration": _int_or_none(entry.get("iteration")),
        "win_rate": _float_or_none(entry.get("win_rate")),
        "games": games,
        "capped_games": _int_or_none(entry.get("capped_games")) or 0,
        "checkpoint_path": _string_or_none(entry.get("checkpoint_path")),
        "checkpoint_policy_spec": _string_or_none(entry.get("checkpoint_policy_spec")),
        "sample_games_ready": games >= FOUNDATION_MILESTONE_BENCHMARK_GAMES,
    }


def _latest_curve_entry(
    curves: Mapping[str, list[dict[str, Any]]],
    opponent_policy_id: str,
) -> Mapping[str, Any] | None:
    entries = curves.get(opponent_policy_id)
    return entries[-1] if entries else None


def _best_curve_entry(
    curves: Mapping[str, list[dict[str, Any]]],
    opponent_policy_id: str,
) -> Mapping[str, Any] | None:
    entries = curves.get(opponent_policy_id)
    if not entries:
        return None
    sample_sized_entries = [
        entry
        for entry in entries
        if (_int_or_none(entry.get("games")) or 0) >= FOUNDATION_MILESTONE_BENCHMARK_GAMES
    ]
    considered_entries = sample_sized_entries or entries

    def sort_key(entry: Mapping[str, Any]) -> tuple[float, int, int]:
        win_rate = _float_or_none(entry.get("win_rate"))
        return (
            win_rate if win_rate is not None else float("-inf"),
            _int_or_none(entry.get("games")) or 0,
            _int_or_none(entry.get("iteration")) or 0,
        )

    return max(
        considered_entries,
        key=sort_key,
    )


def _format_foundation_yardstick_compact(entry: Mapping[str, Any]) -> str:
    games = _int_or_none(entry.get("games"))
    sample_ready = entry.get("sample_games_ready")
    if sample_ready is not True and sample_ready is not False:
        sample_ready = games is not None and games >= FOUNDATION_MILESTONE_BENCHMARK_GAMES
    sample_state = (
        "milestone"
        if sample_ready is True
        else f"below_milestone({_format_manifest_value(games)}/{FOUNDATION_MILESTONE_BENCHMARK_GAMES})"
    )
    parts = [
        f"iter={_format_manifest_value(entry.get('iteration'))}",
        f"win_rate={_format_optional_float(entry.get('win_rate'), digits=3)}",
        f"games={_format_manifest_value(games)}",
        f"cap={_format_manifest_value(entry.get('capped_games'))}",
        f"sample={sample_state}",
    ]
    checkpoint_path = _string_or_none(entry.get("checkpoint_path"))
    if checkpoint_path is not None:
        parts.append(f"checkpoint={checkpoint_path}")
    return " ".join(parts)


def _print_benchmark_opponent_curves(iterations: tuple[Mapping[str, Any], ...]) -> None:
    curves = _benchmark_opponent_curves(iterations)
    if not curves:
        return
    print("")
    print("benchmark_strength_curves:")
    print("note: max-damage and explicit benchmark references only; rates are candidate wins / total games.")
    for opponent, entries in curves.items():
        cells = " ".join(
            f"{entry['iteration']}:{entry['win_rate']:.3f}/{entry['games']}g"
            f"{',cap=' + str(entry['capped_games']) if entry['capped_games'] else ''}"
            for entry in entries
        )
        print(f"- {opponent}: {cells}")


def _print_benchmark_plumbing_curves(iterations: tuple[Mapping[str, Any], ...]) -> None:
    curves = _benchmark_opponent_curves(iterations, policy_ids=_PLUMBING_BENCHMARK_POLICY_IDS)
    if not curves:
        return
    print("")
    print("benchmark_plumbing_curves:")
    print("note: random/simple are saturated harness-health checks, not strength gradients.")
    for opponent, entries in curves.items():
        cells = " ".join(
            f"{entry['iteration']}:{entry['win_rate']:.3f}/{entry['games']}g"
            f"{',cap=' + str(entry['capped_games']) if entry['capped_games'] else ''}"
            for entry in entries
        )
        print(f"- {opponent}: {cells}")


def _benchmark_opponent_curves(
    iterations: tuple[Mapping[str, Any], ...],
    *,
    policy_ids: frozenset[str] | None = None,
) -> dict[str, list[dict[str, Any]]]:
    curves: dict[str, list[dict[str, Any]]] = {}
    for iteration in iterations:
        candidate_policy_id = _iteration_policy_id(iteration)
        if not candidate_policy_id:
            continue
        yardstick_policy_ids = policy_ids if policy_ids is not None else _benchmark_yardstick_policy_ids(iteration)
        benchmark = _optional_mapping(iteration.get("benchmark"))
        head_to_heads = tuple(_mapping(item) for item in _sequence(benchmark.get("head_to_heads", ())))
        entries = (
            _benchmark_curve_entries_from_head_to_heads(
                head_to_heads,
                candidate_policy_id=candidate_policy_id,
                yardstick_policy_ids=yardstick_policy_ids,
            )
            if head_to_heads
            else _benchmark_curve_entries_from_matchups(
                tuple(_mapping(item) for item in _sequence(benchmark.get("matchups", ()))),
                candidate_policy_id=candidate_policy_id,
                yardstick_policy_ids=yardstick_policy_ids,
            )
        )
        for opponent_policy_id, entry in entries.items():
            curves.setdefault(opponent_policy_id, []).append(
                {
                    "iteration": int(iteration.get("iteration", 0)),
                    "checkpoint_path": _string_or_none(iteration.get("checkpoint_path")),
                    "checkpoint_policy_spec": _string_or_none(iteration.get("checkpoint_policy_spec")),
                    **entry,
                }
            )
    return curves


def _benchmark_yardstick_policy_ids(iteration: Mapping[str, Any]) -> frozenset[str]:
    ids = set(_DEFAULT_BENCHMARK_YARDSTICK_POLICY_IDS)
    for spec in _benchmark_reference_policy_specs(iteration):
        policy_id = _report_policy_id_from_spec(spec)
        if policy_id is not None:
            ids.add(policy_id)
    return frozenset(ids - set(_PLUMBING_BENCHMARK_POLICY_IDS))


def _benchmark_reference_policy_specs(iteration: Mapping[str, Any]) -> tuple[str, ...]:
    specs: list[str] = []
    specs.extend(_string_items(iteration.get("benchmark_reference_policy_specs")))
    invocation_config = _optional_mapping(iteration.get("invocation_config"))
    specs.extend(_string_items(invocation_config.get("benchmark_reference_policy_specs") if invocation_config else ()))
    return tuple(dict.fromkeys(specs))


def _string_items(value: object) -> tuple[str, ...]:
    if isinstance(value, (str, bytes, Mapping)) or not hasattr(value, "__iter__"):
        return ()
    return tuple(item for item in (_string_or_none(item) for item in value) if item is not None)


def _report_policy_id_from_spec(spec: str) -> str | None:
    body = spec.strip().partition("?")[0].strip().lower()
    return body if body in _NAMED_REPORT_POLICY_IDS else None


def _benchmark_curve_entries_from_head_to_heads(
    head_to_heads: Iterable[Mapping[str, Any]],
    *,
    candidate_policy_id: str,
    yardstick_policy_ids: frozenset[str],
) -> dict[str, dict[str, float | int]]:
    entries: dict[str, dict[str, float | int]] = {}
    for head_to_head in head_to_heads:
        first_policy_id = _string_or_none(head_to_head.get("first_policy_id"))
        second_policy_id = _string_or_none(head_to_head.get("second_policy_id"))
        if first_policy_id == candidate_policy_id:
            opponent_policy_id = second_policy_id
            wins = _int_or_none(head_to_head.get("first_policy_wins"))
            win_rate = _float_or_none(head_to_head.get("first_policy_win_rate"))
        elif second_policy_id == candidate_policy_id:
            opponent_policy_id = first_policy_id
            wins = _int_or_none(head_to_head.get("second_policy_wins"))
            win_rate = _float_or_none(head_to_head.get("second_policy_win_rate"))
        else:
            continue
        games = _int_or_none(head_to_head.get("games")) or 0
        if not opponent_policy_id or opponent_policy_id not in yardstick_policy_ids or games <= 0:
            continue
        if win_rate is None:
            win_rate = (wins or 0) / games
        entries[opponent_policy_id] = {
            "win_rate": float(win_rate),
            "games": games,
            "capped_games": _int_or_none(head_to_head.get("capped_games")) or 0,
        }
    return entries


def _benchmark_curve_entries_from_matchups(
    matchups: Iterable[Mapping[str, Any]],
    *,
    candidate_policy_id: str,
    yardstick_policy_ids: frozenset[str],
) -> dict[str, dict[str, float | int]]:
    accumulators: dict[str, dict[str, int]] = {}
    for matchup in matchups:
        p1_policy_id = _string_or_none(matchup.get("p1_policy_id"))
        p2_policy_id = _string_or_none(matchup.get("p2_policy_id"))
        metrics = _mapping(matchup.get("metrics", {}))
        games = _int_or_none(metrics.get("games")) or 0
        if p1_policy_id == candidate_policy_id:
            opponent_policy_id = p2_policy_id
            wins = _int_or_none(metrics.get("p1_wins")) or 0
        elif p2_policy_id == candidate_policy_id:
            opponent_policy_id = p1_policy_id
            wins = _int_or_none(metrics.get("p2_wins")) or 0
        else:
            continue
        if not opponent_policy_id or opponent_policy_id not in yardstick_policy_ids or games <= 0:
            continue
        entry = accumulators.setdefault(opponent_policy_id, {"wins": 0, "games": 0, "capped_games": 0})
        entry["wins"] += wins
        entry["games"] += games
        entry["capped_games"] += _int_or_none(metrics.get("capped_games")) or 0
    return {
        opponent: {
            "win_rate": values["wins"] / values["games"],
            "games": values["games"],
            "capped_games": values["capped_games"],
        }
        for opponent, values in accumulators.items()
        if values["games"] > 0
    }


def _string_or_none(value: object) -> str | None:
    if isinstance(value, str) and value:
        return value
    return None


def _int_or_none(value: object) -> int | None:
    if isinstance(value, bool):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _coalesce_optional_int(*values: object) -> int | None:
    for value in values:
        parsed = _int_or_none(value)
        if parsed is not None:
            return parsed
    return None


def _float_or_none(value: object) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _benchmark_win_rate(iteration: Mapping[str, Any]) -> float | None:
    benchmark = iteration.get("benchmark")
    if benchmark is None:
        return None
    benchmark_payload = _mapping(benchmark)
    policy_id = _iteration_policy_id(iteration)
    if policy_id is None:
        return None
    wins = 0
    games = 0
    for result in tuple(_mapping(result) for result in _sequence(benchmark_payload.get("head_to_heads", ()))):
        result_games = int(result.get("games", 0))
        if result.get("first_policy_id") == policy_id:
            wins += int(result.get("first_policy_wins", 0))
            games += result_games
        elif result.get("second_policy_id") == policy_id:
            wins += int(result.get("second_policy_wins", 0))
            games += result_games
    if games:
        return wins / games
    for result in tuple(_mapping(result) for result in _sequence(benchmark_payload.get("matchups", ()))):
        metrics = _mapping(result.get("metrics", {}))
        result_games = int(metrics.get("games", 0))
        if result.get("p1_policy_id") == policy_id:
            wins += int(metrics.get("p1_wins", 0))
            games += result_games
        elif result.get("p2_policy_id") == policy_id:
            wins += int(metrics.get("p2_wins", 0))
            games += result_games
    return (wins / games) if games else None


def _incumbent_win_rate(iteration: Mapping[str, Any]) -> float | None:
    advancement = _optional_mapping(iteration.get("advancement"))
    candidate_win_rate = advancement.get("candidate_win_rate")
    if candidate_win_rate is not None:
        return float(candidate_win_rate)
    candidate_policy_id = advancement.get("candidate_policy_id")
    incumbent_policy_id = advancement.get("incumbent_policy_id")
    if not isinstance(candidate_policy_id, str) or not isinstance(incumbent_policy_id, str):
        return None
    benchmark = iteration.get("benchmark")
    if benchmark is None:
        return None
    benchmark_payload = _mapping(benchmark)
    for result in tuple(_mapping(result) for result in _sequence(benchmark_payload.get("head_to_heads", ()))):
        ids = {result.get("first_policy_id"), result.get("second_policy_id")}
        if ids != {candidate_policy_id, incumbent_policy_id}:
            continue
        games = int(result.get("games", 0))
        if not games:
            return None
        if result.get("first_policy_id") == candidate_policy_id:
            return int(result.get("first_policy_wins", 0)) / games
        return int(result.get("second_policy_wins", 0)) / games
    return None


def _iteration_policy_id(iteration: Mapping[str, Any]) -> str | None:
    training = _mapping(iteration.get("training", {}))
    model_config = _mapping(training.get("model_config", {}))
    policy_id = model_config.get("policy_id")
    return policy_id if isinstance(policy_id, str) and policy_id else None


def _manifest_promotion_status(iteration: Mapping[str, Any]) -> str:
    promotion = iteration.get("promotion")
    if promotion is None:
        return "-"
    promotion_payload = _optional_mapping(promotion)
    return "yes" if promotion_payload.get("recorded") else "no"


def _manifest_source_metadata(manifest: Mapping[str, Any]) -> Mapping[str, Any]:
    source = manifest.get("source")
    return dict(source) if isinstance(source, Mapping) else {}


def _optional_mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _print_source_metadata(metadata: Mapping[str, Any]) -> None:
    if not metadata:
        print("source_metadata: -")
        return
    print("source_metadata:")
    print(f"  available: {_format_bool(metadata.get('available'))}")
    print(f"  branch: {_format_manifest_value(metadata.get('branch'))}")
    print(f"  head: {_format_manifest_value(metadata.get('head'))}")
    print(f"  dirty: {_format_bool(metadata.get('dirty'))}")
    print(f"  repo_root: {_format_manifest_value(metadata.get('repo_root'))}")
    if metadata.get("error") is not None:
        print(f"  error: {_format_manifest_value(metadata.get('error'))}")


def _promotion_status(promotion) -> str:
    if promotion is None:
        return "-"
    return "recorded" if promotion.recorded else "failed"


def _format_bool(value: Any) -> str:
    if value is True:
        return "yes"
    if value is False:
        return "no"
    return "-"


def _format_manifest_value(value: Any) -> str:
    if value is None:
        return "-"
    return str(value)


def _format_optional_float(value: object, *, digits: int = 3) -> str:
    if value is None:
        return "-"
    return f"{float(value):.{digits}f}"


def _format_value_calibration_transform(transform: Any) -> str:
    if getattr(transform, "method", "affine") == "isotonic":
        point_count = len(getattr(transform, "points", ()))
        return (
            f"method=isotonic points={point_count} "
            f"clip=[{transform.clip_min:.1f},{transform.clip_max:.1f}]"
        )
    return (
        "method=affine "
        f"scale={transform.scale:.6f} bias={transform.bias:.6f} "
        f"clip=[{transform.clip_min:.1f},{transform.clip_max:.1f}]"
    )


def _value_calibration_transform_value_blind(transform: Any) -> bool:
    if getattr(transform, "method", "affine") == "isotonic":
        calibrated_values = tuple(float(value) for _, value in getattr(transform, "points", ()))
        if not calibrated_values:
            return True
        return max(calibrated_values) - min(calibrated_values) <= 1e-6
    return abs(float(getattr(transform, "scale", 0.0))) <= 1e-6


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(f".{path.name}.tmp")
    temporary_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    temporary_path.replace(path)


def print_value_calibration_report(report: ValueCalibrationReport) -> None:
    print(f"examples: {report.examples}")
    print(f"mse: {report.mse:.6f}")
    print(f"mae: {report.mae:.6f}")
    print(f"bias: {report.bias:.6f}")
    print(f"sign_accuracy: {report.sign_accuracy:.4f}")
    print(f"expected_calibration_error: {report.expected_calibration_error:.6f}")
    correlation = "n/a" if report.pearson_correlation is None else f"{report.pearson_correlation:.4f}"
    print(f"pearson_correlation: {correlation}")
    print("")
    header = f"{'bin':>13} {'count':>6} {'pred':>9} {'return':>9} {'cal_err':>9}"
    print(header)
    print("-" * len(header))
    for bin_result in report.bins:
        print(
            f"[{bin_result.lower:5.2f},{bin_result.upper:5.2f}) "
            f"{bin_result.count:6d} "
            f"{bin_result.mean_prediction:9.4f} "
            f"{bin_result.mean_return:9.4f} "
            f"{bin_result.calibration_error:9.4f}"
        )
    if report.slices:
        print("")
        slice_header = f"{'slice':>20} {'count':>6} {'mse':>9} {'mae':>9} {'bias':>9} {'sign':>7} {'ece':>9} {'corr':>7}"
        print(slice_header)
        print("-" * len(slice_header))
        for slice_result in report.slices:
            sign_accuracy = f"{slice_result.sign_accuracy:7.4f}" if slice_result.sign_accuracy_applicable else "    n/a"
            correlation = "    n/a" if slice_result.pearson_correlation is None else f"{slice_result.pearson_correlation:7.4f}"
            print(
                f"{slice_result.name:>20} "
                f"{slice_result.examples:6d} "
                f"{slice_result.mse:9.4f} "
                f"{slice_result.mae:9.4f} "
                f"{slice_result.bias:9.4f} "
                f"{sign_accuracy} "
                f"{slice_result.expected_calibration_error:9.4f} "
                f"{correlation}"
            )


def print_value_calibration_compare(payload: Mapping[str, Any]) -> None:
    print("value_calibration_compare:")
    print(f"checkpoint: {payload['checkpoint']}")
    print(f"selection_metric: {payload['selection_metric']}")
    print(f"selection_direction: {payload['selection_direction']}")
    print(f"best_method: {payload['best_method']}")
    print("")
    header = f"{'method':>10} {'examples':>8} {'mse':>9} {'mae':>9} {'bias':>9} {'sign':>7} {'ece':>9} {'corr':>7} {'metric':>9} {'blind':>6}"
    print(header)
    print("-" * len(header))
    for entry in payload["methods"]:
        report = entry["report"]
        correlation = report.get("pearson_correlation")
        correlation_text = "    n/a" if correlation is None else f"{float(correlation):7.4f}"
        metric_value = entry.get("selection_metric_value")
        metric_text = "      n/a" if metric_value is None else f"{float(metric_value):9.4f}"
        print(
            f"{entry['method']:>10} "
            f"{int(report['examples']):8d} "
            f"{float(report['mse']):9.4f} "
            f"{float(report['mae']):9.4f} "
            f"{float(report['bias']):9.4f} "
            f"{float(report['sign_accuracy']):7.4f} "
            f"{float(report['expected_calibration_error']):9.4f} "
            f"{correlation_text} "
            f"{metric_text} "
            f"{_format_bool(entry.get('value_blind')):>6}"
        )
    warnings = payload.get("warnings")
    if warnings:
        print("")
        print("warnings:")
        for warning in warnings:
            print(f"- {warning['code']}: {warning['message']}")


def print_root_puct_benchmark_report(report: RootPUCTSearchBenchmarkReport) -> None:
    print(f"format: {report.format_id}")
    print(f"games: {report.games}")
    print(f"prefixes_per_game: {report.prefixes_per_game}")
    print(f"max_decision_rounds: {report.max_decision_rounds}")
    print(f"search_player: {report.search_player}")
    print(f"cpuct: {report.cpuct:.3f}")
    print(f"source_policy_ids: {dict(report.source_policy_ids)}")
    print(f"source_average_decision_rounds: {report.average_source_decision_rounds:.2f}")
    print(f"evaluated_prefixes: {report.evaluated_prefixes}")
    print(f"skipped_prefixes: {report.skipped_prefixes}")
    print(f"changed_actions: {report.changed_actions}")
    print(f"action_change_rate: {report.action_change_rate:.3f}")
    print(f"average_candidate_count: {report.average_candidate_count:.2f}")
    print(f"average_search_ms: {report.average_elapsed_seconds * 1000.0:.2f}")
    if not report.decisions:
        return
    print("")
    header = (
        f"{'seed':>6} {'prefix':>6} {'recorded':>8} {'selected':>8} "
        f"{'changed':>7} {'value':>8} {'score':>8} {'cand':>5} {'ms':>8}"
    )
    print(header)
    print("-" * len(header))
    for decision in report.decisions[:20]:
        print(
            f"{decision.seed:6d} "
            f"{decision.prefix_decision_round_count:6d} "
            f"{decision.recorded_action_index:8d} "
            f"{decision.selected_action_index:8d} "
            f"{str(decision.changed_action):>7} "
            f"{decision.selected_value:8.3f} "
            f"{decision.selected_score:8.3f} "
            f"{decision.candidate_count:5d} "
            f"{decision.elapsed_seconds * 1000.0:8.2f}"
        )
    if len(report.decisions) > 20:
        print(f"... {len(report.decisions) - 20} more decisions omitted; use --json for full details.")


def print_root_puct_counterfactual_report(report: RootPUCTCounterfactualBenchmarkReport) -> None:
    print(f"format: {report.format_id}")
    print(f"games: {report.games}")
    print(f"prefixes_per_game: {report.prefixes_per_game}")
    print(f"max_decision_rounds: {report.max_decision_rounds}")
    print(f"search_player: {report.search_player}")
    print(f"cpuct: {report.cpuct:.3f}")
    print(f"source_policy_ids: {dict(report.source_policy_ids)}")
    print(f"continuation_policy_ids: {dict(report.continuation_policy_ids)}")
    print(f"source_average_decision_rounds: {report.average_source_decision_rounds:.2f}")
    print(f"evaluated_prefixes: {report.evaluated_prefixes}")
    print(f"skipped_prefixes: {report.skipped_prefixes}")
    print(f"changed_actions: {report.changed_actions}")
    print(f"action_change_rate: {report.action_change_rate:.3f}")
    print(f"improved_actions: {report.improved_actions}")
    print(f"worsened_actions: {report.worsened_actions}")
    print(f"tied_actions: {report.tied_actions}")
    print(f"average_recorded_rollout_value: {report.average_recorded_rollout_value:.3f}")
    print(f"average_selected_rollout_value: {report.average_selected_rollout_value:.3f}")
    print(f"average_rollout_value_delta: {report.average_rollout_value_delta:.3f}")
    print(f"average_candidate_count: {report.average_candidate_count:.2f}")
    print(f"average_search_ms: {report.average_search_elapsed_seconds * 1000.0:.2f}")
    print(f"average_rollout_ms: {report.average_rollout_elapsed_seconds * 1000.0:.2f}")
    if not report.decisions:
        return
    print("")
    header = (
        f"{'seed':>6} {'prefix':>6} {'recorded':>8} {'selected':>8} "
        f"{'delta':>7} {'rec_v':>7} {'sel_v':>7} {'search_v':>8} {'score':>8} "
        f"{'cand':>5} {'search_ms':>9} {'roll_ms':>8}"
    )
    print(header)
    print("-" * len(header))
    for decision in report.decisions[:20]:
        print(
            f"{decision.seed:6d} "
            f"{decision.prefix_decision_round_count:6d} "
            f"{decision.recorded_action_index:8d} "
            f"{decision.selected_action_index:8d} "
            f"{decision.rollout_value_delta:7.3f} "
            f"{decision.recorded_rollout_value:7.3f} "
            f"{decision.selected_rollout_value:7.3f} "
            f"{decision.selected_search_value:8.3f} "
            f"{decision.selected_search_score:8.3f} "
            f"{decision.candidate_count:5d} "
            f"{decision.search_elapsed_seconds * 1000.0:9.2f} "
            f"{decision.rollout_elapsed_seconds * 1000.0:8.2f}"
        )
    if len(report.decisions) > 20:
        print(f"... {len(report.decisions) - 20} more decisions omitted; use --json for full details.")


def _env_config_with_matchup_masks(env_config, matchups, *, context: str):
    """Adopt the encode-time masks + observation spec the matchup checkpoints trained under
    (HIGH-1 latch + the dual-schema resolution: v2 checkpoints keep the v2 encode)."""
    policies = [policy for matchup in matchups for policy in (matchup.p1_policy, matchup.p2_policy)]
    configs = transformer_model_configs_from_policies(policies)
    return env_config_with_checkpoint_masks(
        env_config,
        [feature_masks_from_model_config(config) for config in configs],
        context=context,
        required_specs=[observation_spec_from_model_config(config) for config in configs],
    )


def _env_config_with_spec_masks(env_config, specs, *, extra_model_configs=(), context: str):
    """Adopt masks + observation spec from ``neural:`` policy specs plus any directly-loaded
    model configs (dual-schema resolution)."""
    configs = [
        load_transformer_model_config(path)
        for path in neural_checkpoint_paths_from_policy_specs(specs)
    ]
    configs.extend(extra_model_configs)
    return env_config_with_checkpoint_masks(
        env_config,
        [feature_masks_from_model_config(config) for config in configs],
        context=context,
        required_specs=[observation_spec_from_model_config(config) for config in configs],
    )


def _policy_from_checkpoint(
    checkpoint: Path,
    *,
    deterministic: bool,
    exploration_epsilon: float,
    sampling_temperature: float,
    device: str | None,
):
    return load_transformer_policy(
        checkpoint,
        deterministic=deterministic,
        exploration_epsilon=exploration_epsilon,
        sampling_temperature=sampling_temperature,
        device=device,
    )


if __name__ == "__main__":
    raise SystemExit(main())
