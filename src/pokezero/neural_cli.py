"""Command-line utilities for optional neural policy experiments."""

from __future__ import annotations

import argparse
import copy
from dataclasses import replace
import json
from pathlib import Path
import sys
from typing import Any, Iterable, Mapping

from .cli_audit import (
    add_post_iteration_audit_arguments,
    post_iteration_audit_config_from_args,
    validate_post_iteration_audit_evaluation_games,
)
from .collection import BenchmarkMatchup, benchmark_rollouts, policy_from_spec, policy_spec_with_showdown_root, reject_eval_only_specs
from .local_showdown import LocalShowdownConfig, LocalShowdownEnv
from .neural_policy import (
    DEFAULT_CATEGORY_OOV_BUCKETS,
    TransformerPolicyConfig,
    TransformerSoftmaxPolicy,
    TransformerTrainingConfig,
    TransformerTrainingResult,
    collect_categorical_ids,
    evaluate_transformer_action_priors,
    evaluate_transformer_observation_value,
    evaluate_transformer_opponent_action_priors,
    load_transformer_checkpoint,
    load_transformer_policy,
    require_torch,
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
    RootPUCTSearchPolicy,
    greedy_opponent_action_planner,
    policy_opponent_action_planner,
    prior_top_k_opponent_action_scenario_planner,
)
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
    NeuralSelfPlayPromotionConfig,
    NeuralValueCalibrationConfig,
    NeuralValueSelectionConfig,
    _mapping,
    _sequence,
    load_neural_selfplay_run_manifest,
    run_neural_selfplay_iterations,
)
from .opponents import HISTORICAL_OPPONENT_SELECTION_MODES
from .policy import Policy, RandomLegalPolicy, SimpleLegalPolicy
from .run_audit import RunAuditFailure
from .rollout import RolloutConfig
from .rollout_cli import print_benchmark_report
from .eval_cli import _add_gate_arguments, _gate_config_from_args


MIN_NEURAL_POST_ITERATION_BENCHMARK_MATCHUPS = 4
FOUNDATION_MILESTONE_BENCHMARK_GAMES = 300
_DEFAULT_BENCHMARK_YARDSTICK_POLICY_IDS = frozenset({"random-legal", "simple-legal"})
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

    train = subparsers.add_parser("train", help="Train an entity-token transformer policy from rollout JSONL.")
    train.add_argument("--data", type=Path, nargs="+", required=True, help="One or more rollout JSONL files.")
    train.add_argument("--out", type=Path, required=True, help="Checkpoint output path.")
    train.add_argument(
        "--initial-checkpoint",
        type=Path,
        default=None,
        help="Optional checkpoint to warm-start from. Uses that checkpoint's model config; --policy-id can relabel the output.",
    )
    train.add_argument("--epochs", type=int, default=1, help="Number of training epochs.")
    train.add_argument("--batch-size", type=int, default=64, help="Training batch size.")
    train.add_argument("--learning-rate", type=float, default=3e-4, help="AdamW learning rate.")
    train.add_argument("--weight-decay", type=float, default=0.0, help="AdamW weight decay.")
    train.add_argument("--window-size", type=int, default=4, help="Per-player observation history window.")
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
    train.add_argument("--value-loss-weight", type=float, default=0.25, help="Scalar value-head MSE loss weight.")
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
        "--freeze-non-value-parameters",
        action="store_true",
        help="Train only value-head parameters; intended for value-only calibration fine-tunes from --initial-checkpoint.",
    )
    train.add_argument("--max-batches", type=int, default=None, help="Optional max batches per epoch for smoke runs.")
    train.add_argument("--device", default=None, help="Torch device, e.g. cpu, cuda, or mps. Defaults to cuda when available, else cpu.")
    train.add_argument("--embedding-dim", type=int, default=128, help="Transformer embedding width.")
    train.add_argument("--layers", type=int, default=2, help="Transformer encoder layer count.")
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
            "are maximized, others are minimized."
        ),
    )
    train.add_argument(
        "--value-selection-out",
        type=Path,
        default=None,
        help="Optional JSON output path for per-epoch --value-selection-data reports.",
    )
    train.set_defaults(func=_train)

    benchmark = subparsers.add_parser("benchmark", help="Benchmark a neural checkpoint against fixed baselines.")
    benchmark.add_argument("--checkpoint", type=Path, required=True, help="Neural checkpoint path.")
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
    benchmark.add_argument("--json", action="store_true", help="Print benchmark results as JSON.")
    benchmark.set_defaults(func=_benchmark)

    root_puct_play = subparsers.add_parser(
        "root-puct-play-benchmark",
        help="Benchmark raw checkpoint play against root-PUCT checkpoint play over full games.",
    )
    root_puct_play.add_argument("--checkpoint", type=Path, required=True, help="Neural checkpoint path.")
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
        "--selection-mode",
        choices=("puct", "value"),
        default="puct",
        help=(
            "Root candidate selector for the search policy. 'puct' preserves current PUCT-score "
            "selection; 'value' selects the highest value-evaluated branch from the same candidates."
        ),
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
    root_puct_play.add_argument("--temperature", type=float, default=1.0, help="Softmax temperature for policy and opponent-action priors.")
    root_puct_play.add_argument(
        "--no-search-fallback",
        action="store_true",
        help="Disable fallback to the raw checkpoint action when root-PUCT branch search fails.",
    )
    root_puct_play.add_argument("--json", action="store_true", help="Print benchmark results as JSON.")
    root_puct_play.set_defaults(func=_root_puct_play_benchmark)

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
    root_puct.add_argument("--device", default=None, help="Torch device, e.g. cpu, cuda, or mps.")
    root_puct.add_argument("--temperature", type=float, default=1.0, help="Policy-prior softmax temperature.")
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
    root_puct_counterfactual.add_argument("--json", action="store_true", help="Print counterfactual search benchmark results as JSON.")
    root_puct_counterfactual.set_defaults(func=_root_puct_counterfactual)

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
    value_calibration.add_argument(
        "--fit-out",
        type=Path,
        default=None,
        help="Optional output checkpoint path. Fits an affine value calibration transform on --data and saves a calibrated checkpoint copy.",
    )
    value_calibration.add_argument("--json", action="store_true", help="Print calibration results as JSON.")
    value_calibration.set_defaults(func=_value_calibration)

    iterate = subparsers.add_parser("iterate", help="Run neural-policy self-play training iterations.")
    iterate.add_argument("--run-dir", type=Path, required=True, help="Directory for rollouts, checkpoints, and manifests.")
    iterate.add_argument("--iterations", type=int, required=True, help="Number of collect/train/evaluate iterations.")
    iterate.add_argument("--resume", action="store_true", help="Continue an existing neural self-play run directory from its latest manifest.")
    iterate.add_argument("--games-per-iteration", type=int, required=True, help="Rollout games collected before each train step.")
    iterate.add_argument("--workers", type=int, default=16, help="Parallel rollout collection workers per iteration (capped at the game count).")
    iterate.add_argument("--showdown-root", type=Path, default=None, help="Built Pokemon Showdown checkout root.")
    iterate.add_argument("--format", dest="format_id", default="gen3randombattle", help="Showdown format id.")
    iterate.add_argument("--seed-start", type=int, default=1, help="First deterministic self-play seed.")
    iterate.add_argument("--max-decision-rounds", type=int, default=250, help="Rollout decision-round cap.")
    iterate.add_argument("--node-binary", default="node", help="Node executable used for the BattleStream bridge.")
    iterate.add_argument("--initial-policy", required=True, help="Policy spec used before the first checkpoint exists.")
    iterate.add_argument(
        "--opponent-policy",
        action="append",
        default=None,
        help="Fixed opponent policy spec. May be repeated. Defaults to random-legal and simple-legal.",
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
        "--collector-advancement-mode",
        choices=COLLECTOR_ADVANCEMENT_MODES,
        default="incumbent-gate",
        help=(
            "How a trained candidate becomes the next rollout collector. 'incumbent-gate' "
            "keeps the default head-to-head gate; 'always' advances every saved candidate for "
            "exploratory arms-race runs and is not promotion evidence."
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
    iterate.add_argument("--evaluation-seed-start", type=int, default=1_000_000, help="First deterministic benchmark seed.")
    iterate.add_argument("--epochs", type=int, default=1, help="Training epochs per iteration.")
    iterate.add_argument("--batch-size", type=int, default=64, help="Training batch size.")
    iterate.add_argument("--learning-rate", type=float, default=3e-4, help="AdamW learning rate.")
    iterate.add_argument("--weight-decay", type=float, default=0.0, help="AdamW weight decay.")
    iterate.add_argument("--window-size", type=int, default=4, help="Per-player observation history window.")
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
    iterate.add_argument("--value-loss-weight", type=float, default=0.25, help="Scalar value-head MSE loss weight.")
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
            "others are minimized."
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

    report = subparsers.add_parser("report", help="Print a summary of a neural self-play run manifest.")
    report.add_argument("--run-dir", type=Path, required=True, help="Neural self-play run directory containing manifest.json.")
    report.add_argument("--json", action="store_true", help="Print the raw run manifest as formatted JSON.")
    report.set_defaults(func=_report)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except RunAuditFailure as exc:
        _print_run_audit_failure(exc)
        return 3
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


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


def _train(args: argparse.Namespace) -> int:
    # Surface the missing-neural-extra message before any file I/O (vocab building reads data).
    require_torch()
    if args.value_calibration_out is not None and not args.value_calibration_data:
        raise ValueError("--value-calibration-out requires --value-calibration-data.")
    if args.value_selection_out is not None and not args.value_selection_data:
        raise ValueError("--value-selection-out requires --value-selection-data.")
    initial_model = None
    initial_training_result = None
    if args.initial_checkpoint is not None:
        initial_model, initial_training_result = load_transformer_checkpoint(
            args.initial_checkpoint,
            map_location=resolve_torch_device(args.device),
        )
    training_config = TransformerTrainingConfig(
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        window_size=args.window_size,
        discount=args.discount,
        capped_terminal_value=args.capped_terminal_value,
        hp_delta_return_weight=args.hp_delta_return_weight,
        faint_delta_return_weight=args.faint_delta_return_weight,
        turn_penalty_after=args.turn_penalty_after,
        turn_penalty=args.turn_penalty,
        value_loss_weight=args.value_loss_weight,
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
        freeze_non_value_parameters=args.freeze_non_value_parameters,
    )
    if initial_training_result is not None:
        model_config = replace(
            initial_training_result.model_config,
            policy_id=args.policy_id or initial_training_result.model_config.policy_id,
        )
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
        )
        # The category vocabulary is the closed Gen 3 randbat universe (string->row), the same one
        # the env builds at encode time, so rows align deterministically. (The legacy training-data
        # vocab source is retired: observations now store rows, not collectible hash ids.)
        if args.showdown_root is None:
            raise ValueError("neural training requires --showdown-root for the Gen 3 randbat category vocabulary.")
        from .randbat_vocab import gen3_category_vocabulary

        category_vocab = gen3_category_vocabulary(args.showdown_root, oov_buckets=args.category_oov_buckets)
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
    if args.value_selection_data and training_config.objective != "value-only":
        print(
            "warning: --value-selection-data selects by held-out value calibration, not policy quality; "
            "prefer objective=value-only for value-head calibration runs.",
            file=sys.stderr,
        )
    value_selection_payload = None
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
        )
    else:
        model, result = train_transformer_policy(
            args.data,
            model_config=model_config,
            training_config=training_config,
            initial_model=initial_model,
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
        if metrics.opponent_loss is not None:
            line += (
                f" opponent_loss={metrics.opponent_loss:.6f} "
                f"opponent_accuracy={metrics.opponent_accuracy:.4f}"
            )
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
    return 0


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
        metric_value = value_selection_metric_value(report, selection_metric)
        score = value_selection_score(metric_value, selection_metric)
        epoch = epoch_metric.epoch
        selection_reports.append(
            {
                "epoch": epoch,
                "metric_value": metric_value,
                "training_metrics": epoch_metric.to_dict(),
                "report": report.to_dict(),
            }
        )
        if best_score is None or score > best_score:
            best_score = score
            best_metric_value = metric_value
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())

    model, full_result = train_transformer_policy(
        paths,
        model_config=model_config,
        training_config=training_config,
        initial_model=initial_model,
        epoch_callback=evaluate_epoch,
    )
    if best_state is None or best_epoch is None or best_metric_value is None:
        raise ValueError("value selection produced no epoch reports.")
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
    deterministic = not bool(args.sample)
    checkpoint_policy = _policy_from_checkpoint(
        args.checkpoint,
        deterministic=deterministic,
        exploration_epsilon=args.epsilon,
        sampling_temperature=args.temperature,
        device=args.device,
    )
    policy_id = checkpoint_policy.policy_id
    report = benchmark_rollouts(
        games=args.games,
        env_factory=lambda: LocalShowdownEnv(env_config),
        rollout_config=rollout_config,
        seed_start=args.seed_start,
        matchups=(
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
        ),
    )
    if args.json:
        print(json.dumps(report.to_dict(), indent=2, sort_keys=True))
    else:
        print_benchmark_report(report)
    return 0


def _root_puct_play_benchmark(args: argparse.Namespace) -> int:
    require_torch()
    if args.root_opponent_action_scenarios <= 0:
        raise ValueError("root opponent action scenarios must be positive.")
    if args.root_opponent_action_policy == "benchmark" and args.root_opponent_action_scenarios != 1:
        raise ValueError(
            "root opponent action scenarios above one require --root-opponent-action-policy checkpoint."
        )
    env_config = LocalShowdownConfig(
        showdown_root=args.showdown_root,
        node_binary=args.node_binary,
    )
    policy_showdown_root = env_config.resolved_showdown_root()
    rollout_config = RolloutConfig(
        max_decision_rounds=args.max_decision_rounds,
        format_id=args.format_id,
    )
    leaf_rollout_rounds_values = _root_puct_leaf_rollout_rounds_values(args)
    tag_leaf_policy_ids = args.leaf_rollout_rounds_sweep is not None
    model, result = load_transformer_checkpoint(args.checkpoint, map_location=args.device)
    raw_policy_id = str(result.model_config.policy_id)

    def search_policy_id_for(leaf_rollout_rounds: int) -> str:
        if tag_leaf_policy_ids:
            return f"{raw_policy_id}+root-puct-leaf{leaf_rollout_rounds}"
        return f"{raw_policy_id}+root-puct"

    def make_raw_policy(policy_id: str | None = None) -> TransformerSoftmaxPolicy:
        return TransformerSoftmaxPolicy(
            model=model,
            result=result,
            deterministic=True,
            sampling_temperature=args.temperature,
            device=args.device,
            policy_id=policy_id,
        )

    def make_leaf_rollout_policy(
        *,
        search_policy_id: str,
        search_player_id: str,
        benchmark_opponent_policy: Policy | None,
        player_id: str,
    ) -> Policy:
        if benchmark_opponent_policy is not None and player_id != search_player_id:
            return benchmark_opponent_policy
        return make_raw_policy(policy_id=f"{search_policy_id}-leaf-{player_id}")

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
            if args.root_opponent_action_scenarios > 1:
                opponent_action_scenario_planner = prior_top_k_opponent_action_scenario_planner(
                    opponent_prior_fn,
                    scenario_count=args.root_opponent_action_scenarios,
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
                search_policy_id=search_policy_id,
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
            fallback_policy=make_raw_policy(policy_id=f"{search_policy_id}-fallback"),
            allow_fallback=not args.no_search_fallback,
            policy_id=search_policy_id,
            cpuct=args.cpuct,
            minimum_value_improvement=args.min_value_improvement,
            selection_mode=args.selection_mode,
            leaf_rollout_decision_rounds=leaf_rollout_rounds,
            leaf_rollout_policy_factory=leaf_rollout_policy_factory,
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
            search_policy_id = search_policy_id_for(leaf_rollout_rounds)
            matchups.extend(
                (
                    BenchmarkMatchup(
                        f"{search_policy_id} vs {opponent_id}",
                        make_search_policy(
                            search_policy_id=search_policy_id,
                            leaf_rollout_rounds=leaf_rollout_rounds,
                            search_player_id="p1",
                            benchmark_opponent_spec=benchmark_opponent_spec,
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
    if args.json:
        print(json.dumps(report.to_dict(), indent=2, sort_keys=True))
    else:
        print_benchmark_report(report)
    return 0


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

    report = benchmark_root_puct_search(
        env_factory=lambda: LocalShowdownEnv(env_config),
        policies=policies,
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
        print_root_puct_benchmark_report(report)
    return 0


def _root_puct_counterfactual(args: argparse.Namespace) -> int:
    require_torch()
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
        )
        training_result = replace(training_result, value_calibration_transform=transform)
        save_transformer_checkpoint(args.fit_out, model, result=training_result)
        if not evaluation_held_out:
            print(
                "warning: --fit-out is reporting calibration metrics on the same data used to fit the transform; "
                "pass --eval-data for a held-out calibration read.",
                file=sys.stderr,
            )
        if abs(transform.scale) <= 1e-6:
            print(
                "warning: fitted value calibration scale is near zero; the calibrated checkpoint will make "
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
    if args.json:
        payload = report.to_dict()
        if args.fit_out is not None:
            payload = {
                "checkpoint": str(args.fit_out),
                "fit_paths": [str(path) for path in args.data],
                "evaluation_paths": [str(path) for path in eval_paths],
                "evaluation_held_out": evaluation_held_out,
                "value_calibration_transform": transform.to_dict() if transform is not None else None,
                "report": payload,
            }
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        if args.fit_out is not None and transform is not None:
            print(
                "value_calibration_transform: "
                f"scale={transform.scale:.6f} bias={transform.bias:.6f} "
                f"clip=[{transform.clip_min:.1f},{transform.clip_max:.1f}]"
            )
            print(f"calibrated_checkpoint: {args.fit_out}")
            print(f"evaluation_held_out: {_format_bool(evaluation_held_out)}")
            print("")
        print_value_calibration_report(report)
    return 0


def _iterate(args: argparse.Namespace) -> int:
    # Surface the missing-neural-extra message before any Showdown file I/O (vocab build).
    require_torch()
    # Fail fast: eval-only baselines (max-damage) cannot seed self-play training.
    reject_eval_only_specs([args.initial_policy], role="self-play initial policy")
    reject_eval_only_specs(args.opponent_policy or (), role="self-play training opponent")
    if args.auto_promote and args.promotion_registry is None:
        raise ValueError("--auto-promote requires --promotion-registry.")
    if args.auto_promote and args.collector_advancement_mode != "incumbent-gate":
        raise ValueError("--collector-advancement-mode always cannot be combined with --auto-promote.")
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

    category_vocab = gen3_category_vocabulary(args.showdown_root, oov_buckets=args.category_oov_buckets)
    env_config = LocalShowdownConfig(
        showdown_root=args.showdown_root,
        node_binary=args.node_binary,
        category_vocab=category_vocab,
    )
    rollout_config = RolloutConfig(
        max_decision_rounds=args.max_decision_rounds,
        format_id=args.format_id,
    )
    training_config = TransformerTrainingConfig(
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        window_size=args.window_size,
        discount=args.discount,
        capped_terminal_value=args.capped_terminal_value,
        hp_delta_return_weight=args.hp_delta_return_weight,
        faint_delta_return_weight=args.faint_delta_return_weight,
        turn_penalty_after=args.turn_penalty_after,
        turn_penalty=args.turn_penalty,
        value_loss_weight=args.value_loss_weight,
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
        for spec in (args.opponent_policy or ("random-legal", "simple-legal"))
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
        tensorboard_log_dir=args.tensorboard_logdir,
        max_historical_opponents=args.max_historical_opponents,
        historical_opponent_selection=args.historical_opponent_selection,
        evaluation_games=args.evaluation_games,
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
        resume=args.resume,
    )
    if args.json:
        print(json.dumps(result.to_dict(), indent=2, sort_keys=True))
    else:
        _print_iterate_summary(result)
    return 0


def _print_run_audit_failure(exc: RunAuditFailure) -> None:
    failed = [check.name for check in exc.result.blocking_failed_checks]
    print(f"audit_failed: {exc.result.manifest_path}", file=sys.stderr)
    print(f"failed_checks: {', '.join(failed) if failed else 'unknown'}", file=sys.stderr)


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
    print("note: incumbent win rate drives advancement; blended benchmark win rate is broad health.")
    print("")
    header = (
        f"{'iter':>4} {'games':>5} {'cap':>4} {'bench_wr':>8} {'inc_wr':>8} {'advance':>7} {'promo':>8} "
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
    _print_foundation_readiness(iterations)


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
    print(f"- foundation_evidence_status: {_format_manifest_value(report.get('foundation_evidence_status'))}")
    reasons = report.get("reasons")
    if isinstance(reasons, list) and reasons:
        print(f"  reasons: {', '.join(str(reason) for reason in reasons)}")


def _foundation_readiness_report(iterations: tuple[Mapping[str, Any], ...]) -> dict[str, Any]:
    latest = iterations[-1] if iterations else {}
    calibration_report = _iteration_value_calibration_report(latest)
    curves = _benchmark_opponent_curves((latest,)) if latest else {}
    max_damage_entry = _latest_curve_entry(curves, "max-damage")
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
        "sample_games_ready": games >= FOUNDATION_MILESTONE_BENCHMARK_GAMES,
    }


def _latest_curve_entry(
    curves: Mapping[str, list[dict[str, Any]]],
    opponent_policy_id: str,
) -> Mapping[str, Any] | None:
    entries = curves.get(opponent_policy_id)
    return entries[-1] if entries else None


def _print_benchmark_opponent_curves(iterations: tuple[Mapping[str, Any], ...]) -> None:
    curves = _benchmark_opponent_curves(iterations)
    if not curves:
        return
    print("")
    print("benchmark_opponent_curves:")
    print("note: fixed yardsticks only; rates are candidate wins / total games.")
    for opponent, entries in curves.items():
        cells = " ".join(
            f"{entry['iteration']}:{entry['win_rate']:.3f}/{entry['games']}g"
            f"{',cap=' + str(entry['capped_games']) if entry['capped_games'] else ''}"
            for entry in entries
        )
        print(f"- {opponent}: {cells}")


def _benchmark_opponent_curves(iterations: tuple[Mapping[str, Any], ...]) -> dict[str, list[dict[str, Any]]]:
    curves: dict[str, list[dict[str, Any]]] = {}
    for iteration in iterations:
        candidate_policy_id = _iteration_policy_id(iteration)
        if not candidate_policy_id:
            continue
        yardstick_policy_ids = _benchmark_yardstick_policy_ids(iteration)
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
    return frozenset(ids)


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
