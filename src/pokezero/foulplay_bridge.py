"""Controlled foul-play benchmark harness for context-aware PokeZero policies.

The existing live-server foul-play benchmark is useful for raw online play, but it cannot exercise
context-aware replay-from-root search: the online client only has protocol lines, while
``RootPUCTSearchPolicy`` needs a deterministic seed, action trajectory, and both players' current
legal requests.

This module keeps foul-play across the GPL boundary by running it as a separate websocket client,
but owns the Showdown ``BattleStream`` process so PokeZero can build the exact ``PolicyContext``
required by root-PUCT.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
from dataclasses import dataclass, field, replace
from pathlib import Path
import random
import sys
from typing import Any, Callable, Mapping, Sequence

from .actions import ACTION_COUNT
from .category_vocab import CategoryVocabulary
from .determinization import gen3_randbat_belief_start_override_planner
from .dex import ShowdownDex, load_showdown_dex_cached
from .env import PlayerId, TerminalState
from .local_showdown import BRIDGE_PATH, LocalShowdownConfig, LocalShowdownEnv, showdown_seed_from_int
from .mcts_diagnostics import root_puct_fallback_category
from .neural_policy import (
    TransformerSoftmaxPolicy,
    evaluate_transformer_action_priors,
    evaluate_transformer_observation_value,
    evaluate_transformer_opponent_action_priors,
    load_transformer_checkpoint,
)
from .observation import PokeZeroObservationV0
from .policy import Policy, PolicyContext, PolicyDecision
from .randbat import load_gen3_randbat_source_cached
from .randbat_vocab import gen3_category_vocabulary
from .rollout import RolloutConfig
from .search_policy import (
    RootPUCTSearchPolicy,
    greedy_opponent_action_planner,
    prior_top_k_opponent_action_scenario_planner,
)
from .showdown import (
    DEFAULT_REPLAY_OBSERVATION_SPEC,
    PlayerRelativeBattleState,
    normalize_for_player,
    observation_from_player_state,
    parse_showdown_replay,
    showdown_choice_for_action,
)
from .teacher_capture import action_index_from_choice_string
from .trajectory import BattleTrajectory, TrajectoryStep


SCHEMA_VERSION = "pokezero.controlled-foulplay-benchmark.v1"
COMPARISON_SCHEMA_VERSION = "pokezero.controlled-foulplay-comparison.v1"
DEFAULT_FOULPLAY_ROOT = Path(__file__).resolve().parents[2] / "third_party" / "foul-play"
DEFAULT_BATTLE_ID_PREFIX = "battle-gen3randombattle-controlled"
_WILSON_95_Z = 1.959963984540054
_MIN_STRENGTH_SAMPLE_GAMES = 300
ControlledFoulPlayProgressCallback = Callable[["ControlledFoulPlayBenchmarkResult"], None]
ControlledFoulPlayComparisonProgressCallback = Callable[["ControlledFoulPlayComparisonResult"], None]
_COMPARISON_MODES = {"per-seed", "per-arm"}


@dataclass(frozen=True)
class ControlledFoulPlayConfig:
    checkpoint: Path
    showdown_root: Path
    foulplay_root: Path = DEFAULT_FOULPLAY_ROOT
    foulplay_python: Path | None = None
    games: int = 1
    seed_start: int = 1
    foulplay_random_seed: int | None = None
    search_time_ms: int = 1000
    max_decision_rounds: int = 250
    format_id: str = "gen3randombattle"
    policy_mode: str = "root-puct"
    device: str | None = None
    temperature: float = 1.0
    cpuct: float = 1.25
    selection_mode: str = "visits"
    root_prior_temperature: float | None = None
    minimum_value_improvement: float | None = None
    minimum_override_prior_ratio: float | None = None
    minimum_score_improvement: float | None = None
    root_visit_budget: int | None = 16
    root_time_budget_ms: int | None = None
    root_opponent_action_scenarios: int = 1
    root_opponent_action_candidate_scenarios: int = ACTION_COUNT
    leaf_rollout_rounds: int = 0
    leaf_rollout_sampling: bool = False
    belief_start_overrides: bool = False
    start_override_attempts: int = 1
    belief_start_override_samples: int = 1
    start_override_hp_fraction_tolerance: float = 0.02
    opponent_legal_mask_mode: str = "hidden"
    allow_search_fallback: bool = True
    node_binary: str = "node"
    pokezero_username: str = "PokeZeroBot"
    foulplay_username: str = "FoulPlayBot"
    websocket_host: str = "127.0.0.1"

    def __post_init__(self) -> None:
        if self.games <= 0:
            raise ValueError("games must be positive.")
        if self.seed_start < 0:
            raise ValueError("seed_start must be non-negative.")
        if self.foulplay_random_seed is not None and self.foulplay_random_seed < 0:
            raise ValueError("foulplay_random_seed must be non-negative when set.")
        if self.search_time_ms <= 0:
            raise ValueError("search_time_ms must be positive.")
        if self.max_decision_rounds <= 0:
            raise ValueError("max_decision_rounds must be positive.")
        if self.policy_mode not in {"raw", "root-puct"}:
            raise ValueError("policy_mode must be 'raw' or 'root-puct'.")
        if self.selection_mode not in {"puct", "value", "visits"}:
            raise ValueError("selection_mode must be 'puct', 'value', or 'visits'.")
        if self.root_prior_temperature is not None and (
            self.root_prior_temperature <= 0.0 or not math.isfinite(self.root_prior_temperature)
        ):
            raise ValueError("root_prior_temperature must be a finite positive value when set.")
        if self.minimum_value_improvement is not None and (
            self.minimum_value_improvement < 0.0 or not math.isfinite(self.minimum_value_improvement)
        ):
            raise ValueError("minimum_value_improvement must be a finite non-negative value when set.")
        if self.minimum_override_prior_ratio is not None and (
            self.minimum_override_prior_ratio < 0.0 or not math.isfinite(self.minimum_override_prior_ratio)
        ):
            raise ValueError("minimum_override_prior_ratio must be a finite non-negative value when set.")
        if self.minimum_score_improvement is not None and (
            self.minimum_score_improvement < 0.0 or not math.isfinite(self.minimum_score_improvement)
        ):
            raise ValueError("minimum_score_improvement must be a finite non-negative value when set.")
        if self.root_visit_budget is not None and self.root_visit_budget <= 0:
            raise ValueError("root_visit_budget must be positive when set.")
        if self.root_time_budget_ms is not None and self.root_time_budget_ms <= 0:
            raise ValueError("root_time_budget_ms must be positive when set.")
        if self.root_opponent_action_scenarios <= 0:
            raise ValueError("root_opponent_action_scenarios must be positive.")
        if self.root_opponent_action_candidate_scenarios <= 0:
            raise ValueError("root_opponent_action_candidate_scenarios must be positive.")
        if self.root_opponent_action_candidate_scenarios < self.root_opponent_action_scenarios:
            raise ValueError(
                "root_opponent_action_candidate_scenarios must be greater than or equal to "
                "root_opponent_action_scenarios."
            )
        if self.leaf_rollout_rounds < 0:
            raise ValueError("leaf_rollout_rounds must be non-negative.")
        if self.leaf_rollout_sampling and self.leaf_rollout_rounds <= 0:
            raise ValueError("leaf_rollout_sampling requires positive leaf_rollout_rounds.")
        if self.start_override_attempts <= 0:
            raise ValueError("start_override_attempts must be positive.")
        if self.belief_start_override_samples <= 0:
            raise ValueError("belief_start_override_samples must be positive.")
        if self.belief_start_override_samples > 1 and not self.belief_start_overrides:
            raise ValueError("belief_start_override_samples requires belief_start_overrides.")
        if self.start_override_hp_fraction_tolerance < 0.0 or not math.isfinite(
            self.start_override_hp_fraction_tolerance
        ):
            raise ValueError("start_override_hp_fraction_tolerance must be a finite non-negative value.")
        if self.opponent_legal_mask_mode not in {"hidden", "privileged"}:
            raise ValueError("opponent_legal_mask_mode must be 'hidden' or 'privileged'.")

    @property
    def resolved_foulplay_python(self) -> Path:
        if self.foulplay_python is not None:
            return self.foulplay_python
        return self.foulplay_root / ".venv" / "bin" / "python"

    @property
    def resolved_foulplay_random_seed(self) -> int:
        if self.foulplay_random_seed is not None:
            return self.foulplay_random_seed
        return self.seed_start

    @property
    def effective_root_prior_temperature(self) -> float:
        if self.root_prior_temperature is not None:
            return self.root_prior_temperature
        return self.temperature


@dataclass(frozen=True)
class ControlledFoulPlayGameResult:
    battle_id: str
    seed: int
    winner: str | None
    pokezero_won: bool
    decision_rounds: int
    pokezero_decisions: int
    root_puct_searches: int
    root_puct_fallbacks: int
    root_puct_total_visits: int = 0
    root_puct_effective_total_visits: int = 0
    root_puct_opponent_action_scenarios_generated: int = 0
    root_puct_opponent_action_scenarios_skipped: int = 0
    root_puct_opponent_action_scenarios_unsearched: int = 0
    root_puct_opponent_action_skip_categories: Mapping[str, int] = field(default_factory=dict)
    root_puct_opponent_action_replay_rejection_decision_rounds: Mapping[str, int] = field(
        default_factory=dict
    )
    root_puct_opponent_action_replay_request_mismatch_decision_rounds: Mapping[str, int] = field(
        default_factory=dict
    )
    root_puct_opponent_action_replay_request_mismatch_players: Mapping[str, int] = field(
        default_factory=dict
    )
    root_puct_opponent_action_start_override_mismatch_decision_rounds: Mapping[str, int] = field(
        default_factory=dict
    )
    root_puct_opponent_action_first_observation_mismatch_paths: Mapping[str, int] = field(default_factory=dict)
    root_puct_opponent_action_groups_generated: int = 0
    root_puct_opponent_action_groups_used: int = 0
    root_puct_opponent_action_groups_skipped: int = 0
    root_puct_opponent_action_groups_unsearched: int = 0
    root_puct_selected_prior_action_changes: int = 0
    root_puct_pre_gate_prior_action_changes: int = 0
    root_puct_time_budget_exhaustions: int = 0
    root_puct_start_override_sources_used: int = 0
    root_puct_start_override_attempts_used: int = 0
    root_puct_start_override_shared_samples: int = 0
    root_puct_start_override_shared_samples_accepted: int = 0
    root_puct_start_override_shared_samples_rejected: int = 0
    root_puct_prior_action_change_details: tuple[Mapping[str, Any], ...] = ()
    root_puct_fallback_reasons: Mapping[str, int] = field(default_factory=dict)
    root_puct_fallback_categories: Mapping[str, int] = field(default_factory=dict)
    root_puct_average_elapsed_seconds: float | None = None

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "battle_id": self.battle_id,
            "seed": self.seed,
            "winner": self.winner,
            "pokezero_won": self.pokezero_won,
            "decision_rounds": self.decision_rounds,
            "pokezero_decisions": self.pokezero_decisions,
            "root_puct_searches": self.root_puct_searches,
            "root_puct_fallbacks": self.root_puct_fallbacks,
            "root_puct_total_visits": self.root_puct_total_visits,
            "root_puct_opponent_action_scenarios_generated": self.root_puct_opponent_action_scenarios_generated,
            "root_puct_opponent_action_scenarios_skipped": self.root_puct_opponent_action_scenarios_skipped,
            "root_puct_opponent_action_scenarios_unsearched": self.root_puct_opponent_action_scenarios_unsearched,
            "root_puct_opponent_action_groups_generated": self.root_puct_opponent_action_groups_generated,
            "root_puct_opponent_action_groups_used": self.root_puct_opponent_action_groups_used,
            "root_puct_opponent_action_groups_skipped": self.root_puct_opponent_action_groups_skipped,
            "root_puct_opponent_action_groups_unsearched": self.root_puct_opponent_action_groups_unsearched,
            "root_puct_selected_prior_action_changes": self.root_puct_selected_prior_action_changes,
            "root_puct_pre_gate_prior_action_changes": self.root_puct_pre_gate_prior_action_changes,
            "root_puct_time_budget_exhaustions": self.root_puct_time_budget_exhaustions,
            "root_puct_start_override_sources_used": self.root_puct_start_override_sources_used,
            "root_puct_start_override_attempts_used": self.root_puct_start_override_attempts_used,
            "root_puct_start_override_shared_samples": self.root_puct_start_override_shared_samples,
            "root_puct_start_override_shared_samples_accepted": (
                self.root_puct_start_override_shared_samples_accepted
            ),
            "root_puct_start_override_shared_samples_rejected": (
                self.root_puct_start_override_shared_samples_rejected
            ),
        }
        if self.root_puct_effective_total_visits:
            payload["root_puct_effective_total_visits"] = self.root_puct_effective_total_visits
        if self.root_puct_opponent_action_skip_categories:
            payload["root_puct_opponent_action_skip_categories"] = dict(
                sorted(self.root_puct_opponent_action_skip_categories.items())
            )
        if self.root_puct_opponent_action_replay_rejection_decision_rounds:
            payload["root_puct_opponent_action_replay_rejection_decision_rounds"] = dict(
                sorted(
                    self.root_puct_opponent_action_replay_rejection_decision_rounds.items(),
                    key=lambda item: int(item[0]),
                )
            )
        if self.root_puct_opponent_action_replay_request_mismatch_decision_rounds:
            payload["root_puct_opponent_action_replay_request_mismatch_decision_rounds"] = dict(
                sorted(
                    self.root_puct_opponent_action_replay_request_mismatch_decision_rounds.items(),
                    key=lambda item: int(item[0]),
                )
            )
        if self.root_puct_opponent_action_replay_request_mismatch_players:
            payload["root_puct_opponent_action_replay_request_mismatch_players"] = dict(
                sorted(self.root_puct_opponent_action_replay_request_mismatch_players.items())
            )
        if self.root_puct_opponent_action_start_override_mismatch_decision_rounds:
            payload["root_puct_opponent_action_start_override_mismatch_decision_rounds"] = dict(
                sorted(
                    self.root_puct_opponent_action_start_override_mismatch_decision_rounds.items(),
                    key=lambda item: int(item[0]),
                )
            )
        if self.root_puct_opponent_action_first_observation_mismatch_paths:
            payload["root_puct_opponent_action_first_observation_mismatch_paths"] = dict(
                sorted(self.root_puct_opponent_action_first_observation_mismatch_paths.items())
            )
        if self.root_puct_average_elapsed_seconds is not None:
            payload["root_puct_average_elapsed_seconds"] = self.root_puct_average_elapsed_seconds
        if self.root_puct_prior_action_change_details:
            payload["root_puct_prior_action_change_details"] = [
                dict(detail)
                for detail in self.root_puct_prior_action_change_details
            ]
        if self.root_puct_fallback_reasons:
            payload["root_puct_fallback_reasons"] = dict(sorted(self.root_puct_fallback_reasons.items()))
        fallback_categories = _fallback_categories_from_reasons(
            self.root_puct_fallback_reasons,
            self.root_puct_fallback_categories,
        )
        if fallback_categories:
            payload["root_puct_fallback_categories"] = dict(sorted(fallback_categories.items()))
        return payload


@dataclass(frozen=True)
class ControlledFoulPlayBenchmarkResult:
    config: ControlledFoulPlayConfig
    policy_id: str
    games: tuple[ControlledFoulPlayGameResult, ...]
    foulplay_random_seed_schedule: tuple[int, ...] | None = None

    @property
    def completed_games(self) -> int:
        return len(self.games)

    @property
    def wins(self) -> int:
        return sum(1 for game in self.games if game.pokezero_won)

    @property
    def win_rate(self) -> float:
        return self.wins / self.completed_games if self.completed_games else 0.0

    def to_dict(self) -> dict[str, Any]:
        root_searches = sum(game.root_puct_searches for game in self.games)
        root_fallbacks = sum(game.root_puct_fallbacks for game in self.games)
        root_total_visits = sum(game.root_puct_total_visits for game in self.games)
        root_effective_total_visits = sum(game.root_puct_effective_total_visits for game in self.games)
        root_scenarios_generated = sum(game.root_puct_opponent_action_scenarios_generated for game in self.games)
        root_scenarios_skipped = sum(game.root_puct_opponent_action_scenarios_skipped for game in self.games)
        root_scenarios_unsearched = sum(game.root_puct_opponent_action_scenarios_unsearched for game in self.games)
        root_scenario_skip_categories: dict[str, int] = {}
        root_replay_rejection_decision_rounds: dict[str, int] = {}
        root_replay_request_mismatch_decision_rounds: dict[str, int] = {}
        root_replay_request_mismatch_players: dict[str, int] = {}
        root_start_override_mismatch_decision_rounds: dict[str, int] = {}
        root_first_observation_mismatch_paths: dict[str, int] = {}
        for game in self.games:
            _merge_count_mapping(
                root_scenario_skip_categories,
                game.root_puct_opponent_action_skip_categories,
            )
            _merge_count_mapping(
                root_replay_rejection_decision_rounds,
                game.root_puct_opponent_action_replay_rejection_decision_rounds,
            )
            _merge_count_mapping(
                root_replay_request_mismatch_decision_rounds,
                game.root_puct_opponent_action_replay_request_mismatch_decision_rounds,
            )
            _merge_count_mapping(
                root_replay_request_mismatch_players,
                game.root_puct_opponent_action_replay_request_mismatch_players,
            )
            _merge_count_mapping(
                root_start_override_mismatch_decision_rounds,
                game.root_puct_opponent_action_start_override_mismatch_decision_rounds,
            )
            _merge_count_mapping(
                root_first_observation_mismatch_paths,
                game.root_puct_opponent_action_first_observation_mismatch_paths,
            )
        root_action_groups_generated = sum(game.root_puct_opponent_action_groups_generated for game in self.games)
        root_action_groups_used = sum(game.root_puct_opponent_action_groups_used for game in self.games)
        root_action_groups_skipped = sum(game.root_puct_opponent_action_groups_skipped for game in self.games)
        root_action_groups_unsearched = sum(game.root_puct_opponent_action_groups_unsearched for game in self.games)
        root_selected_prior_action_changes = sum(game.root_puct_selected_prior_action_changes for game in self.games)
        root_pre_gate_prior_action_changes = sum(game.root_puct_pre_gate_prior_action_changes for game in self.games)
        root_time_budget_exhaustions = sum(game.root_puct_time_budget_exhaustions for game in self.games)
        root_start_override_sources_used = sum(game.root_puct_start_override_sources_used for game in self.games)
        root_start_override_attempts_used = sum(game.root_puct_start_override_attempts_used for game in self.games)
        root_start_override_shared_samples = sum(game.root_puct_start_override_shared_samples for game in self.games)
        root_start_override_shared_samples_accepted = sum(
            game.root_puct_start_override_shared_samples_accepted for game in self.games
        )
        root_start_override_shared_samples_rejected = sum(
            game.root_puct_start_override_shared_samples_rejected for game in self.games
        )
        root_fallback_reasons: dict[str, int] = {}
        root_fallback_categories: dict[str, int] = {}
        for game in self.games:
            for reason, count in game.root_puct_fallback_reasons.items():
                root_fallback_reasons[reason] = root_fallback_reasons.get(reason, 0) + count
            for category, count in _fallback_categories_from_reasons(
                game.root_puct_fallback_reasons,
                game.root_puct_fallback_categories,
            ).items():
                root_fallback_categories[category] = root_fallback_categories.get(category, 0) + count
        elapsed_values = [
            game.root_puct_average_elapsed_seconds
            for game in self.games
            if game.root_puct_average_elapsed_seconds is not None
        ]
        payload: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "checkpoint": str(self.config.checkpoint),
            "format_id": self.config.format_id,
            "policy_id": self.policy_id,
            "policy_mode": self.config.policy_mode,
            "opponent_policy_id": "foul-play",
            "games": self.config.games,
            "completed_games": self.completed_games,
            "complete": self.completed_games >= self.config.games,
            "status": "complete" if self.completed_games >= self.config.games else "partial",
            "wins": self.wins,
            "win_rate": self.win_rate,
            "seed_start": self.config.seed_start,
            "foulplay_random_seed": self.config.resolved_foulplay_random_seed,
            "max_decision_rounds": self.config.max_decision_rounds,
            "root_puct": {
                "cpuct": self.config.cpuct,
                "selection_mode": self.config.selection_mode,
                "root_prior_temperature": self.config.effective_root_prior_temperature,
                "minimum_value_improvement": self.config.minimum_value_improvement,
                "minimum_override_prior_ratio": self.config.minimum_override_prior_ratio,
                "minimum_score_improvement": self.config.minimum_score_improvement,
                "root_visit_budget": self.config.root_visit_budget,
                "root_time_budget_ms": self.config.root_time_budget_ms,
                "root_opponent_action_scenarios": self.config.root_opponent_action_scenarios,
                "root_opponent_action_candidate_scenarios": self.config.root_opponent_action_candidate_scenarios,
                "leaf_rollout_rounds": self.config.leaf_rollout_rounds,
                "leaf_rollout_sampling": self.config.leaf_rollout_sampling,
                "belief_start_overrides": self.config.belief_start_overrides,
                "start_override_attempts": self.config.start_override_attempts,
                "belief_start_override_samples": self.config.belief_start_override_samples,
                "start_override_hp_fraction_tolerance": self.config.start_override_hp_fraction_tolerance,
                "opponent_legal_mask_mode": self.config.opponent_legal_mask_mode,
                "foulplay_search_time_ms": self.config.search_time_ms,
                "allow_search_fallback": self.config.allow_search_fallback,
                "searches": root_searches,
                "fallbacks": root_fallbacks,
                "total_visits": root_total_visits,
                "opponent_action_scenarios_generated": root_scenarios_generated,
                "opponent_action_scenarios_skipped": root_scenarios_skipped,
                "opponent_action_scenarios_unsearched": root_scenarios_unsearched,
                "opponent_action_groups_generated": root_action_groups_generated,
                "opponent_action_groups_used": root_action_groups_used,
                "opponent_action_groups_skipped": root_action_groups_skipped,
                "opponent_action_groups_unsearched": root_action_groups_unsearched,
                "selected_prior_action_changes": root_selected_prior_action_changes,
                "pre_gate_prior_action_changes": root_pre_gate_prior_action_changes,
                "time_budget_exhaustions": root_time_budget_exhaustions,
                "start_override_sources_used": root_start_override_sources_used,
                "start_override_attempts_used": root_start_override_attempts_used,
                "start_override_shared_samples": root_start_override_shared_samples,
                "start_override_shared_samples_accepted": root_start_override_shared_samples_accepted,
                "start_override_shared_samples_rejected": root_start_override_shared_samples_rejected,
            },
            "game_results": [game.to_dict() for game in self.games],
        }
        if self.foulplay_random_seed_schedule is not None:
            payload["foulplay_random_seed_schedule"] = _foulplay_random_seed_schedule_payload(
                self.foulplay_random_seed_schedule
            )
        if elapsed_values:
            payload["root_puct"]["average_elapsed_seconds"] = sum(elapsed_values) / len(elapsed_values)
        if root_effective_total_visits:
            payload["root_puct"]["effective_total_visits"] = root_effective_total_visits
        if root_scenario_skip_categories:
            payload["root_puct"]["opponent_action_skip_categories"] = dict(
                sorted(root_scenario_skip_categories.items())
            )
        if root_replay_rejection_decision_rounds:
            payload["root_puct"]["opponent_action_replay_rejection_decision_rounds"] = dict(
                sorted(root_replay_rejection_decision_rounds.items(), key=lambda item: int(item[0]))
            )
        if root_replay_request_mismatch_decision_rounds:
            payload["root_puct"]["opponent_action_replay_request_mismatch_decision_rounds"] = dict(
                sorted(root_replay_request_mismatch_decision_rounds.items(), key=lambda item: int(item[0]))
            )
        if root_replay_request_mismatch_players:
            payload["root_puct"]["opponent_action_replay_request_mismatch_players"] = dict(
                sorted(root_replay_request_mismatch_players.items())
            )
        if root_start_override_mismatch_decision_rounds:
            payload["root_puct"]["opponent_action_start_override_mismatch_decision_rounds"] = dict(
                sorted(root_start_override_mismatch_decision_rounds.items(), key=lambda item: int(item[0]))
            )
        if root_first_observation_mismatch_paths:
            payload["root_puct"]["opponent_action_first_observation_mismatch_paths"] = dict(
                sorted(root_first_observation_mismatch_paths.items())
            )
        if root_fallback_reasons:
            payload["root_puct"]["fallback_reasons"] = dict(sorted(root_fallback_reasons.items()))
        if root_fallback_categories:
            payload["root_puct"]["fallback_categories"] = dict(sorted(root_fallback_categories.items()))
        return payload


@dataclass(frozen=True)
class ControlledFoulPlayComparisonResult:
    config: ControlledFoulPlayConfig
    raw: ControlledFoulPlayBenchmarkResult | None
    root_puct: ControlledFoulPlayBenchmarkResult | None
    comparison_mode: str = "per-seed"

    @property
    def complete(self) -> bool:
        return (
            self.raw is not None
            and self.root_puct is not None
            and self.raw.completed_games >= self.raw.config.games
            and self.root_puct.completed_games >= self.root_puct.config.games
        )

    @property
    def status(self) -> str:
        if self.raw is None:
            return "pending"
        return "complete" if self.complete else "partial"

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": COMPARISON_SCHEMA_VERSION,
            "checkpoint": str(self.config.checkpoint),
            "format_id": self.config.format_id,
            "opponent_policy_id": "foul-play",
            "games": self.config.games,
            "seed_start": self.config.seed_start,
            "foulplay_random_seed": self.config.resolved_foulplay_random_seed,
            "foulplay_random_seed_schedule": _comparison_foulplay_random_seed_schedule_payload(
                self.config,
                comparison_mode=self.comparison_mode,
                count=self.config.games,
            ),
            "comparison_mode": self.comparison_mode,
            "status": self.status,
            "complete": self.complete,
            "runs": {
                "raw": self.raw.to_dict() if self.raw is not None else None,
                "root_puct": self.root_puct.to_dict() if self.root_puct is not None else None,
            },
            "comparison": _comparison_readout(
                self.raw,
                self.root_puct,
                comparison_mode=self.comparison_mode,
            ),
        }


def _comparison_readout(
    raw: ControlledFoulPlayBenchmarkResult | None,
    root_puct: ControlledFoulPlayBenchmarkResult | None,
    *,
    comparison_mode: str,
) -> dict[str, Any]:
    raw_by_seed = _games_by_seed(raw)
    search_by_seed = _games_by_seed(root_puct)
    matched_seeds = tuple(sorted(raw_by_seed.keys() & search_by_seed.keys()))
    raw_paired_wins = sum(1 for seed in matched_seeds if raw_by_seed[seed].pokezero_won)
    search_paired_wins = sum(1 for seed in matched_seeds if search_by_seed[seed].pokezero_won)
    both_won = sum(
        1
        for seed in matched_seeds
        if raw_by_seed[seed].pokezero_won and search_by_seed[seed].pokezero_won
    )
    raw_only_won = sum(
        1
        for seed in matched_seeds
        if raw_by_seed[seed].pokezero_won and not search_by_seed[seed].pokezero_won
    )
    root_puct_only_won = sum(
        1
        for seed in matched_seeds
        if search_by_seed[seed].pokezero_won and not raw_by_seed[seed].pokezero_won
    )
    neither_won = sum(
        1
        for seed in matched_seeds
        if not raw_by_seed[seed].pokezero_won and not search_by_seed[seed].pokezero_won
    )
    paired_games = len(matched_seeds)
    raw_completed_games = raw.completed_games if raw is not None else 0
    search_completed_games = root_puct.completed_games if root_puct is not None else 0
    raw_wins = raw.wins if raw is not None else 0
    search_wins = root_puct.wins if root_puct is not None else 0

    return {
        "sample_size": {
            "paired_games": paired_games,
            "minimum_strength_games": _MIN_STRENGTH_SAMPLE_GAMES,
            "status": "strength_sized" if paired_games >= _MIN_STRENGTH_SAMPLE_GAMES else "diagnostic_only",
        },
        "aggregate": {
            "analysis_method": "completed_prefix_marginal_rates",
            "raw": _rate_readout(raw_wins, raw_completed_games),
            "root_puct": _rate_readout(search_wins, search_completed_games),
            "root_puct_minus_raw_win_rate": _delta_rate(
                search_wins,
                search_completed_games,
                raw_wins,
                raw_completed_games,
                require_equal_games=True,
            ),
            "delta_interpretation": (
                "descriptive_only_when_both_prefixes_have_equal_nonzero_completed_games"
            ),
        },
        "paired_by_seed": {
            "pairing_method": _pairing_method_for_comparison_mode(comparison_mode),
            "opponent_deterministic": False,
            "paired_counterfactual": False,
            "interval_method": "marginal_wilson_per_arm_not_paired_delta",
            "delta_interpretation": "descriptive_only",
            "games": paired_games,
            "raw": _rate_readout(raw_paired_wins, paired_games),
            "root_puct": _rate_readout(search_paired_wins, paired_games),
            "root_puct_minus_raw_win_rate": _delta_rate(
                search_paired_wins,
                paired_games,
                raw_paired_wins,
                paired_games,
            ),
            "discordant_pairs": {
                "both_won": both_won,
                "raw_only_won": raw_only_won,
                "root_puct_only_won": root_puct_only_won,
                "neither_won": neither_won,
            },
            "first_seed": matched_seeds[0] if matched_seeds else None,
            "last_seed": matched_seeds[-1] if matched_seeds else None,
        },
    }


def _pairing_method_for_comparison_mode(comparison_mode: str) -> str:
    if comparison_mode == "per-seed":
        return "per_seed_shared_battlestream_seed_and_foulplay_start_seed"
    return "shared_battlestream_seed_only"


def _fallback_categories_from_reasons(
    reasons: Mapping[str, int],
    categories: Mapping[str, int],
) -> dict[str, int]:
    result = {str(category): int(count) for category, count in categories.items()}
    if result:
        return result
    for reason, count in reasons.items():
        category = root_puct_fallback_category(reason)
        result[category] = result.get(category, 0) + int(count)
    return result


def _merge_count_mapping(target: dict[str, int], source: object) -> None:
    if not isinstance(source, Mapping):
        return
    for key, value in source.items():
        try:
            count = int(value)
        except (TypeError, ValueError):
            continue
        target[str(key)] = target.get(str(key), 0) + count


def _comparison_foulplay_random_seed_schedule_payload(
    config: ControlledFoulPlayConfig,
    *,
    comparison_mode: str,
    count: int,
) -> dict[str, Any]:
    if comparison_mode == "per-seed":
        return _foulplay_random_seed_schedule_payload(
            _per_seed_foulplay_random_seed_schedule(config, count=count)
        )
    return _foulplay_random_seed_schedule_payload((config.resolved_foulplay_random_seed,))


def _per_seed_foulplay_random_seed_schedule(
    config: ControlledFoulPlayConfig,
    *,
    count: int,
) -> tuple[int, ...]:
    return tuple(
        (
            config.foulplay_random_seed + offset
            if config.foulplay_random_seed is not None
            else config.seed_start + offset
        )
        for offset in range(count)
    )


def _foulplay_random_seed_schedule_payload(seeds: tuple[int, ...]) -> dict[str, Any]:
    return {
        "count": len(seeds),
        "first_seed": seeds[0] if seeds else None,
        "last_seed": seeds[-1] if seeds else None,
        "mode": "constant" if len(set(seeds)) <= 1 else "per_game_incrementing",
        "seeds": list(seeds),
    }


def _games_by_seed(
    result: ControlledFoulPlayBenchmarkResult | None,
) -> dict[int, ControlledFoulPlayGameResult]:
    if result is None:
        return {}
    return {game.seed: game for game in result.games}


def _rate_readout(wins: int, games: int) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "games": games,
        "wins": wins,
        "win_rate": _rate(wins, games),
        "interval_method": "wilson_score_marginal_95",
    }
    if games:
        lower, upper = _wilson_interval(wins, games, z=_WILSON_95_Z)
        payload["wilson_95"] = {"lower": lower, "upper": upper}
    else:
        payload["wilson_95"] = None
    return payload


def _rate(wins: int, games: int) -> float:
    return wins / games if games else 0.0


def _delta_rate(
    first_wins: int,
    first_games: int,
    second_wins: int,
    second_games: int,
    *,
    require_equal_games: bool = False,
) -> float | None:
    if first_games <= 0 or second_games <= 0:
        return None
    if require_equal_games and first_games != second_games:
        return None
    return _rate(first_wins, first_games) - _rate(second_wins, second_games)


def _wilson_interval(wins: int, games: int, *, z: float) -> tuple[float, float]:
    if games <= 0:
        return (0.0, 0.0)
    if z == 0.0:
        rate = wins / games
        return (rate, rate)
    p_hat = wins / games
    z_squared = z * z
    denominator = 1.0 + (z_squared / games)
    center = p_hat + (z_squared / (2.0 * games))
    adjustment = z * math.sqrt(((p_hat * (1.0 - p_hat)) + (z_squared / (4.0 * games))) / games)
    return (
        max(0.0, (center - adjustment) / denominator),
        min(1.0, (center + adjustment) / denominator),
    )


class FoulPlayProtocolError(RuntimeError):
    """Raised when the foul-play websocket client emits an unsupported protocol message."""


@dataclass
class _ProcessLogBuffer:
    stdout: list[str] = field(default_factory=list)
    stderr: list[str] = field(default_factory=list)

    def append_stdout(self, line: str) -> None:
        self.stdout.append(line)
        if len(self.stdout) > 200:
            del self.stdout[: len(self.stdout) - 200]

    def append_stderr(self, line: str) -> None:
        self.stderr.append(line)
        if len(self.stderr) > 200:
            del self.stderr[: len(self.stderr) - 200]

    def tail(self) -> str:
        parts = []
        if self.stderr:
            parts.append("stderr:\n" + "\n".join(self.stderr[-40:]))
        if self.stdout:
            parts.append("stdout:\n" + "\n".join(self.stdout[-40:]))
        return "\n\n".join(parts) or "(no foul-play output captured)"


class _FoulPlayWebsocketServer:
    def __init__(self, *, username: str, host: str) -> None:
        self.username = username
        self.host = host
        self.port: int | None = None
        self.websocket: Any = None
        self.server: Any = None
        self.challenge_queue: asyncio.Queue[str] = asyncio.Queue()
        self.choice_queue: asyncio.Queue[tuple[str, str]] = asyncio.Queue()

    @property
    def uri(self) -> str:
        if self.port is None:
            raise RuntimeError("server has not started.")
        return f"ws://{self.host}:{self.port}/showdown/websocket"

    async def start(self) -> None:
        import websockets

        self.server = await websockets.serve(self._handle_connection, self.host, 0, max_size=None)
        socket = self.server.sockets[0]
        self.port = int(socket.getsockname()[1])

    async def close(self) -> None:
        if self.websocket is not None:
            await self.websocket.close()
            self.websocket = None
        if self.server is not None:
            self.server.close()
            await self.server.wait_closed()
            self.server = None

    async def _handle_connection(self, websocket: Any) -> None:
        self.websocket = websocket
        await websocket.send("|challstr|1|pokezero-controlled")
        try:
            async for message in websocket:
                await self._handle_message(str(message))
        except Exception:
            # The caller monitors the foul-play process and will report its stderr/stdout. Avoid
            # leaking a noisy websocket traceback as the primary error.
            self.websocket = None
            return

    async def _handle_message(self, message: str) -> None:
        room, body = _split_outgoing_showdown_message(message)
        if room and (choice := _choice_body_from_outgoing_message(body)):
            await self.choice_queue.put((room, choice))
            return
        if body.startswith("/trn "):
            await self.send_global(f"|updateuser|{self.username}|1|0|")
            return
        if body.startswith("/challenge "):
            target = body[len("/challenge ") :].split(",", 1)[0].strip()
            await self.challenge_queue.put(target)
            return
        if body.startswith("/leave "):
            battle_id = body[len("/leave ") :].strip()
            await self.send_room_lines(battle_id, ["|deinit|"])
            return
        # /utm, /timer, chat, and /savereplay are accepted no-ops for this controlled harness.

    async def send_global(self, message: str) -> None:
        if self.websocket is None:
            raise FoulPlayProtocolError("foul-play websocket is not connected.")
        await self.websocket.send(message)

    async def send_room_lines(self, battle_id: str, lines: Sequence[str]) -> None:
        if self.websocket is None:
            raise FoulPlayProtocolError("foul-play websocket is not connected.")
        if not lines:
            return
        await self.websocket.send(f">{battle_id}\n" + "\n".join(lines))

    async def wait_for_challenge(self, *, expected_target: str, timeout_seconds: float = 30.0) -> None:
        deadline = asyncio.get_running_loop().time() + timeout_seconds
        while True:
            remaining = deadline - asyncio.get_running_loop().time()
            if remaining <= 0:
                raise TimeoutError("timed out waiting for foul-play challenge.")
            target = await asyncio.wait_for(self.challenge_queue.get(), timeout=remaining)
            if _showdown_id(target) == _showdown_id(expected_target):
                return

    async def wait_for_choice(self, *, battle_id: str, timeout_seconds: float = 120.0) -> str:
        deadline = asyncio.get_running_loop().time() + timeout_seconds
        while True:
            remaining = deadline - asyncio.get_running_loop().time()
            if remaining <= 0:
                raise TimeoutError("timed out waiting for foul-play choice.")
            room, choice = await asyncio.wait_for(self.choice_queue.get(), timeout=remaining)
            if room == battle_id:
                return choice


class _BattleBridge:
    def __init__(self, *, showdown_root: Path, node_binary: str) -> None:
        self.showdown_root = showdown_root
        self.node_binary = node_binary
        self.process: asyncio.subprocess.Process | None = None
        self.events: asyncio.Queue[Mapping[str, Any]] = asyncio.Queue()
        self.stderr_lines: list[str] = []
        self._stdout_task: asyncio.Task[None] | None = None
        self._stderr_task: asyncio.Task[None] | None = None

    async def start(self) -> None:
        self.process = await asyncio.create_subprocess_exec(
            self.node_binary,
            str(BRIDGE_PATH),
            "--showdown-root",
            str(self.showdown_root),
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env={
                "PATH": os.environ.get("PATH", ""),
                "POKEZERO_SHOWDOWN_ROOT": str(self.showdown_root),
            },
        )
        self._stdout_task = asyncio.create_task(self._drain_stdout())
        self._stderr_task = asyncio.create_task(self._drain_stderr())

    async def close(self) -> None:
        process = self.process
        if process is None:
            return
        try:
            if process.returncode is None:
                try:
                    await self.send({"type": "close"})
                    await asyncio.wait_for(process.wait(), timeout=1.0)
                except Exception:
                    process.terminate()
                    try:
                        await asyncio.wait_for(process.wait(), timeout=1.0)
                    except asyncio.TimeoutError:
                        process.kill()
                        await process.wait()
        finally:
            for task in (self._stdout_task, self._stderr_task):
                if task is not None:
                    task.cancel()
            self.process = None

    async def send(self, command: Mapping[str, Any]) -> None:
        if self.process is None or self.process.stdin is None or self.process.returncode is not None:
            raise RuntimeError(self._exit_message())
        self.process.stdin.write(json.dumps(command, separators=(",", ":")).encode("utf-8") + b"\n")
        await self.process.stdin.drain()

    async def next_event(self, *, timeout_seconds: float = 120.0) -> Mapping[str, Any]:
        event = await asyncio.wait_for(self.events.get(), timeout=timeout_seconds)
        if event.get("type") == "error":
            raise RuntimeError(str(event.get("message") or "BattleStream bridge error."))
        return event

    async def _drain_stdout(self) -> None:
        assert self.process is not None and self.process.stdout is not None
        async for raw in self.process.stdout:
            line = raw.decode("utf-8").strip()
            if not line:
                continue
            self.events.put_nowait(json.loads(line))

    async def _drain_stderr(self) -> None:
        assert self.process is not None and self.process.stderr is not None
        async for raw in self.process.stderr:
            self.stderr_lines.append(raw.decode("utf-8", errors="replace").rstrip())
            if len(self.stderr_lines) > 100:
                del self.stderr_lines[: len(self.stderr_lines) - 100]

    def _exit_message(self) -> str:
        if self.process is not None and self.process.returncode is not None:
            stderr = "\n".join(self.stderr_lines[-20:])
            suffix = f" Stderr:\n{stderr}" if stderr else ""
            return f"BattleStream bridge exited with status {self.process.returncode}.{suffix}"
        return "BattleStream bridge is not running."


@dataclass
class _ControlledBattleState:
    battle_id: str
    seed: int
    format_id: str
    public_lines: list[str] = field(default_factory=list)
    request_lines: dict[PlayerId, str] = field(default_factory=dict)
    trajectory: BattleTrajectory | None = None
    decisions: list[PolicyDecision] = field(default_factory=list)
    next_foulplay_rqid: int = 1
    foulplay_terminal_sent: bool = False

    def all_lines(self) -> list[str]:
        return [*self.public_lines, *self.request_lines.values()]


async def run_controlled_foulplay_benchmark(
    config: ControlledFoulPlayConfig,
    *,
    progress_callback: ControlledFoulPlayProgressCallback | None = None,
) -> ControlledFoulPlayBenchmarkResult:
    """Run PokeZero vs foul-play with a known BattleStream seed and context-aware policy."""

    _validate_external_paths(config)
    model, result = load_transformer_checkpoint(config.checkpoint, map_location=config.device)
    policy_id = str(result.model_config.policy_id)
    observation_spec = replace(
        DEFAULT_REPLAY_OBSERVATION_SPEC,
        categorical_feature_count=result.model_config.categorical_feature_count,
        numeric_feature_count=result.model_config.numeric_feature_count,
    )
    vocab = gen3_category_vocabulary(config.showdown_root)
    dex = load_showdown_dex_cached(config.showdown_root)
    env_config = LocalShowdownConfig(
        showdown_root=config.showdown_root,
        node_binary=config.node_binary,
        observation_spec=observation_spec,
        category_vocab=vocab,
    )
    rollout_config = RolloutConfig(
        max_decision_rounds=config.max_decision_rounds,
        format_id=config.format_id,
    )
    policy = _build_policy(
        config=config,
        model=model,
        result=result,
        env_config=env_config,
        rollout_config=rollout_config,
        policy_id=policy_id,
    )
    benchmark_policy_id = policy.policy_id if hasattr(policy, "policy_id") else policy_id

    server = _FoulPlayWebsocketServer(username=config.foulplay_username, host=config.websocket_host)
    bridge = _BattleBridge(showdown_root=config.showdown_root, node_binary=config.node_binary)
    foulplay_process: asyncio.subprocess.Process | None = None
    foulplay_logs = _ProcessLogBuffer()
    foulplay_log_tasks: list[asyncio.Task[None]] = []
    game_results: list[ControlledFoulPlayGameResult] = []
    try:
        await server.start()
        foulplay_process = await _spawn_foulplay(config, server.uri)
        foulplay_log_tasks = [
            asyncio.create_task(_drain_process_stream(foulplay_process.stdout, foulplay_logs.append_stdout)),
            asyncio.create_task(_drain_process_stream(foulplay_process.stderr, foulplay_logs.append_stderr)),
        ]
        await bridge.start()
        for offset in range(config.games):
            seed = config.seed_start + offset
            await _wait_for_foulplay_challenge_or_exit(
                server=server,
                expected_target=config.pokezero_username,
                process=foulplay_process,
                logs=foulplay_logs,
            )
            game_results.append(
                await _run_single_game(
                    config=config,
                    bridge=bridge,
                    server=server,
                    policy=policy,
                    vocab=vocab,
                    dex=dex,
                    observation_spec=observation_spec,
                    seed=seed,
                    foulplay_process=foulplay_process,
                    foulplay_logs=foulplay_logs,
                )
            )
            if progress_callback is not None:
                progress_callback(
                    ControlledFoulPlayBenchmarkResult(
                        config=config,
                        policy_id=benchmark_policy_id,
                        games=tuple(game_results),
                    )
                )
    finally:
        await bridge.close()
        if foulplay_process is not None and foulplay_process.returncode is None:
            foulplay_process.terminate()
            try:
                await asyncio.wait_for(foulplay_process.wait(), timeout=5.0)
            except asyncio.TimeoutError:
                foulplay_process.kill()
                await foulplay_process.wait()
        for task in foulplay_log_tasks:
            task.cancel()
        await server.close()

    return ControlledFoulPlayBenchmarkResult(
        config=config,
        policy_id=benchmark_policy_id,
        games=tuple(game_results),
    )


async def run_controlled_foulplay_comparison(
    config: ControlledFoulPlayConfig,
    *,
    comparison_mode: str = "per-seed",
    progress_callback: ControlledFoulPlayComparisonProgressCallback | None = None,
) -> ControlledFoulPlayComparisonResult:
    """Run raw checkpoint and root-PUCT against foul-play over the same seed band."""

    if comparison_mode not in _COMPARISON_MODES:
        raise ValueError(f"comparison_mode must be one of {sorted(_COMPARISON_MODES)!r}.")

    if comparison_mode == "per-seed":
        return await _run_controlled_foulplay_comparison_per_seed(
            config,
            progress_callback=progress_callback,
        )
    return await _run_controlled_foulplay_comparison_per_arm(
        config,
        progress_callback=progress_callback,
    )


async def _run_controlled_foulplay_comparison_per_arm(
    config: ControlledFoulPlayConfig,
    *,
    progress_callback: ControlledFoulPlayComparisonProgressCallback | None = None,
) -> ControlledFoulPlayComparisonResult:
    raw_result: ControlledFoulPlayBenchmarkResult | None = None
    root_puct_result: ControlledFoulPlayBenchmarkResult | None = None

    def emit_progress() -> None:
        if progress_callback is None:
            return
        progress_callback(
            ControlledFoulPlayComparisonResult(
                config=config,
                raw=raw_result,
                root_puct=root_puct_result,
                comparison_mode="per-arm",
            )
        )

    def raw_progress(result: ControlledFoulPlayBenchmarkResult) -> None:
        nonlocal raw_result
        raw_result = result
        emit_progress()

    def root_puct_progress(result: ControlledFoulPlayBenchmarkResult) -> None:
        nonlocal root_puct_result
        root_puct_result = result
        emit_progress()

    raw_result = await run_controlled_foulplay_benchmark(
        replace(config, policy_mode="raw"),
        progress_callback=raw_progress,
    )
    root_puct_result = await run_controlled_foulplay_benchmark(
        replace(config, policy_mode="root-puct"),
        progress_callback=root_puct_progress,
    )
    return ControlledFoulPlayComparisonResult(
        config=config,
        raw=raw_result,
        root_puct=root_puct_result,
        comparison_mode="per-arm",
    )


async def _run_controlled_foulplay_comparison_per_seed(
    config: ControlledFoulPlayConfig,
    *,
    progress_callback: ControlledFoulPlayComparisonProgressCallback | None = None,
) -> ControlledFoulPlayComparisonResult:
    raw_games: list[ControlledFoulPlayGameResult] = []
    root_puct_games: list[ControlledFoulPlayGameResult] = []
    raw_policy_id: str | None = None
    root_puct_policy_id: str | None = None

    def raw_result() -> ControlledFoulPlayBenchmarkResult | None:
        if raw_policy_id is None:
            return None
        return ControlledFoulPlayBenchmarkResult(
            config=replace(config, policy_mode="raw"),
            policy_id=raw_policy_id,
            games=tuple(raw_games),
            foulplay_random_seed_schedule=_per_seed_foulplay_random_seed_schedule(
                config,
                count=len(raw_games),
            ),
        )

    def root_puct_result() -> ControlledFoulPlayBenchmarkResult | None:
        if root_puct_policy_id is None:
            return None
        return ControlledFoulPlayBenchmarkResult(
            config=replace(config, policy_mode="root-puct"),
            policy_id=root_puct_policy_id,
            games=tuple(root_puct_games),
            foulplay_random_seed_schedule=_per_seed_foulplay_random_seed_schedule(
                config,
                count=len(root_puct_games),
            ),
        )

    def emit_progress() -> None:
        if progress_callback is None:
            return
        progress_callback(
            ControlledFoulPlayComparisonResult(
                config=config,
                raw=raw_result(),
                root_puct=root_puct_result(),
                comparison_mode="per-seed",
            )
        )

    for offset in range(config.games):
        seed = config.seed_start + offset
        single_config = _single_seed_comparison_config(config, seed=seed, offset=offset)

        raw_single = await run_controlled_foulplay_benchmark(replace(single_config, policy_mode="raw"))
        raw_policy_id = raw_single.policy_id
        raw_games.extend(raw_single.games)
        emit_progress()

        root_puct_single = await run_controlled_foulplay_benchmark(
            replace(single_config, policy_mode="root-puct")
        )
        root_puct_policy_id = root_puct_single.policy_id
        root_puct_games.extend(root_puct_single.games)
        emit_progress()

    return ControlledFoulPlayComparisonResult(
        config=config,
        raw=raw_result(),
        root_puct=root_puct_result(),
        comparison_mode="per-seed",
    )


def _single_seed_comparison_config(
    config: ControlledFoulPlayConfig,
    *,
    seed: int,
    offset: int,
) -> ControlledFoulPlayConfig:
    foulplay_seed = config.foulplay_random_seed + offset if config.foulplay_random_seed is not None else seed
    return replace(
        config,
        games=1,
        seed_start=seed,
        foulplay_random_seed=foulplay_seed,
    )


def _validate_external_paths(config: ControlledFoulPlayConfig) -> None:
    if not config.checkpoint.exists():
        raise FileNotFoundError(f"checkpoint not found: {config.checkpoint}")
    if not (config.showdown_root / "dist" / "sim" / "index.js").exists():
        raise FileNotFoundError(
            f"built Pokemon Showdown simulator not found under {config.showdown_root}; "
            "set --showdown-root to a built checkout."
        )
    if not (config.foulplay_root / "run.py").exists():
        raise FileNotFoundError(
            f"foul-play checkout not found at {config.foulplay_root}; initialize third_party/foul-play "
            "or pass --foulplay-root."
        )
    if not config.resolved_foulplay_python.exists():
        raise FileNotFoundError(
            f"foul-play Python not found at {config.resolved_foulplay_python}; run "
            "scripts/setup_foulplay_eval.sh or pass --foulplay-python."
        )


def _build_policy(
    *,
    config: ControlledFoulPlayConfig,
    model: Any,
    result: Any,
    env_config: LocalShowdownConfig,
    rollout_config: RolloutConfig,
    policy_id: str,
) -> Policy:
    def raw_policy(
        policy_id_override: str | None = None,
        *,
        deterministic: bool = True,
    ) -> TransformerSoftmaxPolicy:
        return TransformerSoftmaxPolicy(
            model=model,
            result=result,
            deterministic=deterministic,
            sampling_temperature=config.temperature,
            device=config.device,
            policy_id=policy_id_override,
        )

    if config.policy_mode == "raw":
        return raw_policy(policy_id)

    search_policy_id = f"{policy_id}+root-puct"

    def value_fn(history: tuple[PokeZeroObservationV0, ...]) -> float:
        return evaluate_transformer_observation_value(
            model=model,
            result=result,
            observations=history,
            device=config.device,
        )

    def prior_fn(history: tuple[PokeZeroObservationV0, ...]) -> tuple[float, ...]:
        return evaluate_transformer_action_priors(
            model=model,
            result=result,
            observations=history,
            temperature=1.0,
            device=config.device,
        )

    def opponent_prior_fn(history: tuple[PokeZeroObservationV0, ...]) -> tuple[float, ...]:
        return evaluate_transformer_opponent_action_priors(
            model=model,
            result=result,
            observations=history,
            temperature=config.temperature,
            device=config.device,
        )

    scenario_planner = None
    if config.root_opponent_action_candidate_scenarios > 1:
        scenario_planner = prior_top_k_opponent_action_scenario_planner(
            opponent_prior_fn,
            scenario_count=config.root_opponent_action_candidate_scenarios,
        )

    leaf_rollout_policy_factory = None
    if config.leaf_rollout_rounds:
        leaf_rollout_policy_factory = lambda player_id: raw_policy(
            f"{search_policy_id}-leaf-{player_id}",
            deterministic=not config.leaf_rollout_sampling,
        )

    start_override_planner = None
    if config.belief_start_overrides:
        set_source = load_gen3_randbat_source_cached(config.showdown_root)
        start_override_planner = gen3_randbat_belief_start_override_planner(set_source)

    return RootPUCTSearchPolicy(
        env_factory=lambda: LocalShowdownEnv(env_config),
        rollout_config=rollout_config,
        value_fn=value_fn,
        prior_fn=prior_fn,
        opponent_action_planner=greedy_opponent_action_planner(opponent_prior_fn),
        opponent_action_scenario_planner=scenario_planner,
        fallback_policy=raw_policy(f"{search_policy_id}-fallback"),
        allow_fallback=config.allow_search_fallback,
        policy_id=search_policy_id,
        cpuct=config.cpuct,
        selection_mode=config.selection_mode,
        root_prior_temperature=config.effective_root_prior_temperature,
        minimum_value_improvement=config.minimum_value_improvement,
        minimum_override_prior_ratio=config.minimum_override_prior_ratio,
        minimum_score_improvement=config.minimum_score_improvement,
        root_visit_budget=config.root_visit_budget,
        root_time_budget_seconds=(
            None if config.root_time_budget_ms is None else config.root_time_budget_ms / 1000.0
        ),
        max_opponent_action_scenarios=config.root_opponent_action_scenarios,
        leaf_rollout_decision_rounds=config.leaf_rollout_rounds,
        leaf_rollout_policy_factory=leaf_rollout_policy_factory,
        start_override_planner=start_override_planner,
        start_override_attempts=config.start_override_attempts,
        start_override_samples_per_scenario=config.belief_start_override_samples,
        start_override_hp_fraction_tolerance=config.start_override_hp_fraction_tolerance,
        leaf_rollout_metadata={
            "root_puct_leaf_rollout_opponent_policy": "checkpoint",
            "root_puct_leaf_rollout_sampling": config.leaf_rollout_sampling,
        }
        if config.leaf_rollout_rounds
        else {},
    )


async def _spawn_foulplay(
    config: ControlledFoulPlayConfig,
    websocket_uri: str,
) -> asyncio.subprocess.Process:
    env = _foulplay_env(config)
    return await asyncio.create_subprocess_exec(
        *_foulplay_command(config, websocket_uri),
        cwd=str(config.foulplay_root),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        env=env,
    )


def _foulplay_env(config: ControlledFoulPlayConfig) -> dict[str, str]:
    seed = config.resolved_foulplay_random_seed
    return {
        **os.environ,
        "FOULPLAY_LOCAL_NOSEC": "1",
        "PYTHONPATH": str(config.foulplay_root),
        "POKEZERO_FOULPLAY_RANDOM_SEED": str(seed),
        "PYTHONHASHSEED": str(seed % (2**32)),
    }


def _foulplay_command(config: ControlledFoulPlayConfig, websocket_uri: str) -> tuple[str, ...]:
    run_path = str(config.foulplay_root / "run.py")
    seed_wrapper = (
        "import os, random, runpy, sys; "
        "random.seed(int(os.environ['POKEZERO_FOULPLAY_RANDOM_SEED'])); "
        "script = sys.argv[1]; "
        "sys.argv = sys.argv[1:]; "
        "runpy.run_path(script, run_name='__main__')"
    )
    return (
        str(config.resolved_foulplay_python),
        "-c",
        seed_wrapper,
        run_path,
        "--websocket-uri",
        websocket_uri,
        "--ps-username",
        config.foulplay_username,
        "--bot-mode",
        "challenge_user",
        "--user-to-challenge",
        config.pokezero_username,
        "--pokemon-format",
        config.format_id,
        "--run-count",
        str(config.games),
        "--search-time-ms",
        str(config.search_time_ms),
    )


async def _drain_process_stream(
    stream: asyncio.StreamReader | None,
    append: Any,
) -> None:
    if stream is None:
        return
    async for raw in stream:
        append(raw.decode("utf-8", errors="replace").rstrip())


async def _run_single_game(
    *,
    config: ControlledFoulPlayConfig,
    bridge: _BattleBridge,
    server: _FoulPlayWebsocketServer,
    policy: Policy,
    vocab: CategoryVocabulary,
    dex: ShowdownDex,
    observation_spec: Any,
    seed: int,
    foulplay_process: asyncio.subprocess.Process,
    foulplay_logs: _ProcessLogBuffer,
) -> ControlledFoulPlayGameResult:
    battle_id = f"{DEFAULT_BATTLE_ID_PREFIX}-{seed}"
    state = _ControlledBattleState(
        battle_id=battle_id,
        seed=seed,
        format_id=config.format_id,
        trajectory=BattleTrajectory(
            battle_id=battle_id,
            format_id=config.format_id,
            seed=seed,
            metadata={"opponent_policy_id": "foul-play", "controlled_foulplay_bridge": True},
        ),
    )
    await server.send_room_lines(
        battle_id,
        ["|init|battle", f"|title|{config.pokezero_username} vs. {config.foulplay_username}"],
    )
    await bridge.send(
        {
            "type": "start",
            "battleId": battle_id,
            "formatid": config.format_id,
            "seed": showdown_seed_from_int(seed),
            "players": {
                "p1": config.pokezero_username,
                "p2": config.foulplay_username,
            },
        }
    )

    requested_players: tuple[PlayerId, ...] = ()
    decision_round = 0
    terminal: TerminalState | None = None

    while terminal is None:
        if decision_round >= config.max_decision_rounds:
            terminal = TerminalState(winner=None, turn_count=config.max_decision_rounds, capped=True)
            break
        event = await bridge.next_event()
        if event.get("battleId") != battle_id:
            continue
        event_type = event.get("type")
        if event_type == "stream":
            await _handle_stream_event(state, server, event)
            terminal = _terminal_from_public_lines(state.public_lines, config)
            continue
        if event_type == "ready":
            requested_players = tuple(str(player) for player in event.get("requested") or ())
            if not requested_players:
                continue
            terminal = await _handle_decision_boundary(
                config=config,
                bridge=bridge,
                server=server,
                state=state,
                policy=policy,
                vocab=vocab,
                dex=dex,
                observation_spec=observation_spec,
                decision_round=decision_round,
                requested_players=requested_players,
                foulplay_process=foulplay_process,
                foulplay_logs=foulplay_logs,
            )
            decision_round += 1
            continue
        if event_type == "terminal":
            terminal = _terminal_from_public_lines(state.public_lines, config) or TerminalState(
                winner=None,
                turn_count=decision_round,
            )
            break

    await _notify_foulplay_terminal(
        state=state,
        server=server,
        terminal=terminal,
        config=config,
    )
    winner_name = _winner_name(terminal, config)
    if state.trajectory is not None:
        state.trajectory.record_terminal(terminal)
    elapsed = [
        float(decision.metadata["root_puct_elapsed_seconds"])
        for decision in state.decisions
        if "root_puct_elapsed_seconds" in decision.metadata
    ]
    root_searches = sum(
        1
        for decision in state.decisions
        if decision.metadata.get("policy_family") == "root-puct-search"
        and not decision.metadata.get("root_puct_fallback")
    )
    root_fallbacks = sum(1 for decision in state.decisions if decision.metadata.get("root_puct_fallback"))
    root_fallback_reasons: dict[str, int] = {}
    root_fallback_categories: dict[str, int] = {}
    for decision in state.decisions:
        if not decision.metadata.get("root_puct_fallback"):
            continue
        reason = str(decision.metadata.get("root_puct_fallback_reason") or "unknown")
        root_fallback_reasons[reason] = root_fallback_reasons.get(reason, 0) + 1
        category = str(
            decision.metadata.get("root_puct_fallback_category")
            or root_puct_fallback_category(reason)
        )
        root_fallback_categories[category] = root_fallback_categories.get(category, 0) + 1
    root_total_visits = sum(
        int(decision.metadata.get("root_puct_total_visits") or 0)
        for decision in state.decisions
        if decision.metadata.get("policy_family") == "root-puct-search"
        and not decision.metadata.get("root_puct_fallback")
    )
    root_effective_total_visits = sum(
        int(decision.metadata.get("root_puct_effective_total_visits") or 0)
        for decision in state.decisions
        if decision.metadata.get("policy_family") == "root-puct-search"
        and not decision.metadata.get("root_puct_fallback")
    )
    root_scenarios_generated = sum(
        int(decision.metadata.get("root_puct_opponent_action_scenarios_generated") or 0)
        for decision in state.decisions
        if decision.metadata.get("policy_family") == "root-puct-search"
    )
    root_scenarios_skipped = sum(
        int(decision.metadata.get("root_puct_opponent_action_scenarios_skipped") or 0)
        for decision in state.decisions
        if decision.metadata.get("policy_family") == "root-puct-search"
    )
    root_scenarios_unsearched = sum(
        int(decision.metadata.get("root_puct_opponent_action_scenarios_unsearched") or 0)
        for decision in state.decisions
        if decision.metadata.get("policy_family") == "root-puct-search"
    )
    root_scenario_skip_categories: dict[str, int] = {}
    root_replay_rejection_decision_rounds: dict[str, int] = {}
    root_replay_request_mismatch_decision_rounds: dict[str, int] = {}
    root_replay_request_mismatch_players: dict[str, int] = {}
    root_start_override_mismatch_decision_rounds: dict[str, int] = {}
    root_first_observation_mismatch_paths: dict[str, int] = {}
    for decision in state.decisions:
        if decision.metadata.get("policy_family") != "root-puct-search":
            continue
        _merge_count_mapping(
            root_scenario_skip_categories,
            decision.metadata.get("root_puct_opponent_action_skip_categories"),
        )
        _merge_count_mapping(
            root_replay_rejection_decision_rounds,
            decision.metadata.get("root_puct_opponent_action_replay_rejection_decision_rounds"),
        )
        _merge_count_mapping(
            root_replay_request_mismatch_decision_rounds,
            decision.metadata.get("root_puct_opponent_action_replay_request_mismatch_decision_rounds"),
        )
        _merge_count_mapping(
            root_replay_request_mismatch_players,
            decision.metadata.get("root_puct_opponent_action_replay_request_mismatch_players"),
        )
        _merge_count_mapping(
            root_start_override_mismatch_decision_rounds,
            decision.metadata.get("root_puct_opponent_action_start_override_mismatch_decision_rounds"),
        )
        _merge_count_mapping(
            root_first_observation_mismatch_paths,
            decision.metadata.get("root_puct_opponent_action_first_observation_mismatch_paths"),
        )
    root_action_groups_generated = sum(
        int(decision.metadata.get("root_puct_opponent_action_groups_generated") or 0)
        for decision in state.decisions
        if decision.metadata.get("policy_family") == "root-puct-search"
    )
    root_action_groups_used = sum(
        int(decision.metadata.get("root_puct_opponent_action_groups_used") or 0)
        for decision in state.decisions
        if decision.metadata.get("policy_family") == "root-puct-search"
    )
    root_action_groups_skipped = sum(
        int(decision.metadata.get("root_puct_opponent_action_groups_skipped") or 0)
        for decision in state.decisions
        if decision.metadata.get("policy_family") == "root-puct-search"
    )
    root_action_groups_unsearched = sum(
        int(decision.metadata.get("root_puct_opponent_action_groups_unsearched") or 0)
        for decision in state.decisions
        if decision.metadata.get("policy_family") == "root-puct-search"
    )
    root_selected_prior_action_changes = sum(
        1
        for decision in state.decisions
        if decision.metadata.get("policy_family") == "root-puct-search"
        and not decision.metadata.get("root_puct_fallback")
        and decision.metadata.get("root_puct_selected_changed_prior_action")
    )
    root_pre_gate_prior_action_changes = sum(
        1
        for decision in state.decisions
        if decision.metadata.get("policy_family") == "root-puct-search"
        and not decision.metadata.get("root_puct_fallback")
        and decision.metadata.get("root_puct_pre_gate_changed_prior_action")
    )
    root_time_budget_exhaustions = sum(
        1
        for decision in state.decisions
        if decision.metadata.get("policy_family") == "root-puct-search"
        and not decision.metadata.get("root_puct_fallback")
        and decision.metadata.get("root_puct_time_budget_exhausted")
    )
    root_start_override_sources_used = sum(
        int(decision.metadata.get("root_puct_start_override_sources_used") or 0)
        for decision in state.decisions
        if decision.metadata.get("policy_family") == "root-puct-search"
        and not decision.metadata.get("root_puct_fallback")
    )
    root_start_override_attempts_used = sum(
        int(decision.metadata.get("root_puct_start_override_attempts_used") or 0)
        for decision in state.decisions
        if decision.metadata.get("policy_family") == "root-puct-search"
    )
    root_start_override_shared_samples = sum(
        int(decision.metadata.get("root_puct_start_override_shared_samples") or 0)
        for decision in state.decisions
        if decision.metadata.get("policy_family") == "root-puct-search"
    )
    root_start_override_shared_samples_accepted = sum(
        int(decision.metadata.get("root_puct_start_override_shared_samples_accepted") or 0)
        for decision in state.decisions
        if decision.metadata.get("policy_family") == "root-puct-search"
    )
    root_start_override_shared_samples_rejected = sum(
        int(decision.metadata.get("root_puct_start_override_shared_samples_rejected") or 0)
        for decision in state.decisions
        if decision.metadata.get("policy_family") == "root-puct-search"
    )
    root_prior_action_change_details = _root_puct_prior_action_change_details(state.decisions)
    return ControlledFoulPlayGameResult(
        battle_id=battle_id,
        seed=seed,
        winner=winner_name,
        pokezero_won=winner_name == config.pokezero_username,
        decision_rounds=decision_round,
        pokezero_decisions=len(state.decisions),
        root_puct_searches=root_searches,
        root_puct_fallbacks=root_fallbacks,
        root_puct_total_visits=root_total_visits,
        root_puct_effective_total_visits=root_effective_total_visits,
        root_puct_opponent_action_scenarios_generated=root_scenarios_generated,
        root_puct_opponent_action_scenarios_skipped=root_scenarios_skipped,
        root_puct_opponent_action_scenarios_unsearched=root_scenarios_unsearched,
        root_puct_opponent_action_skip_categories=root_scenario_skip_categories,
        root_puct_opponent_action_replay_rejection_decision_rounds=(
            root_replay_rejection_decision_rounds
        ),
        root_puct_opponent_action_replay_request_mismatch_decision_rounds=(
            root_replay_request_mismatch_decision_rounds
        ),
        root_puct_opponent_action_replay_request_mismatch_players=(
            root_replay_request_mismatch_players
        ),
        root_puct_opponent_action_start_override_mismatch_decision_rounds=(
            root_start_override_mismatch_decision_rounds
        ),
        root_puct_opponent_action_first_observation_mismatch_paths=root_first_observation_mismatch_paths,
        root_puct_opponent_action_groups_generated=root_action_groups_generated,
        root_puct_opponent_action_groups_used=root_action_groups_used,
        root_puct_opponent_action_groups_skipped=root_action_groups_skipped,
        root_puct_opponent_action_groups_unsearched=root_action_groups_unsearched,
        root_puct_selected_prior_action_changes=root_selected_prior_action_changes,
        root_puct_pre_gate_prior_action_changes=root_pre_gate_prior_action_changes,
        root_puct_time_budget_exhaustions=root_time_budget_exhaustions,
        root_puct_start_override_sources_used=root_start_override_sources_used,
        root_puct_start_override_attempts_used=root_start_override_attempts_used,
        root_puct_start_override_shared_samples=root_start_override_shared_samples,
        root_puct_start_override_shared_samples_accepted=root_start_override_shared_samples_accepted,
        root_puct_start_override_shared_samples_rejected=root_start_override_shared_samples_rejected,
        root_puct_prior_action_change_details=root_prior_action_change_details,
        root_puct_fallback_reasons=root_fallback_reasons,
        root_puct_fallback_categories=root_fallback_categories,
        root_puct_average_elapsed_seconds=(sum(elapsed) / len(elapsed) if elapsed else None),
    )


def _root_puct_prior_action_change_details(
    decisions: Sequence[PolicyDecision],
) -> tuple[Mapping[str, Any], ...]:
    details: list[dict[str, Any]] = []
    for decision_index, decision in enumerate(decisions):
        metadata = decision.metadata
        if metadata.get("policy_family") != "root-puct-search":
            continue
        if metadata.get("root_puct_fallback"):
            continue
        if not (
            metadata.get("root_puct_selected_changed_prior_action")
            or metadata.get("root_puct_pre_gate_changed_prior_action")
        ):
            continue
        details.append(
            {
                "decision_index": decision_index,
                "selected_action": decision.action_index,
                "search_action": _optional_int(metadata.get("root_puct_search_action")),
                "prior_action": _optional_int(metadata.get("root_puct_prior_action")),
                "selected_changed_prior_action": bool(metadata.get("root_puct_selected_changed_prior_action")),
                "pre_gate_changed_prior_action": bool(metadata.get("root_puct_pre_gate_changed_prior_action")),
                "selected_value": _optional_float(metadata.get("root_puct_selected_value")),
                "search_value": _optional_float(metadata.get("root_puct_search_action_value")),
                "prior_value": _optional_float(metadata.get("root_puct_prior_value")),
                "selected_score": _optional_float(metadata.get("root_puct_selected_score")),
                "search_score": _optional_float(metadata.get("root_puct_search_action_score")),
                "prior_score": _optional_float(metadata.get("root_puct_prior_score")),
                "selected_action_prior": _optional_float(metadata.get("root_puct_selected_action_prior")),
                "search_action_prior": _optional_float(metadata.get("root_puct_search_action_prior")),
                "prior_action_prior": _optional_float(metadata.get("root_puct_prior_action_prior")),
                "selected_visits": _optional_int(metadata.get("root_puct_selected_action_visits")),
                "search_visits": _optional_int(metadata.get("root_puct_search_action_visits")),
                "prior_visits": _optional_int(metadata.get("root_puct_prior_action_visits")),
                "value_gate_used": bool(metadata.get("root_puct_value_gate_used", False)),
                "prior_ratio_gate_used": bool(metadata.get("root_puct_prior_ratio_gate_used", False)),
                "minimum_override_prior_ratio": _optional_float(
                    metadata.get("root_puct_minimum_override_prior_ratio")
                ),
                "prior_ratio_gate_required_prior": _optional_float(
                    metadata.get("root_puct_prior_ratio_gate_required_prior")
                ),
                "score_gate_used": bool(metadata.get("root_puct_score_gate_used", False)),
                "minimum_score_improvement": _optional_float(
                    metadata.get("root_puct_minimum_score_improvement")
                ),
                "score_gate_required_score": _optional_float(
                    metadata.get("root_puct_score_gate_required_score")
                ),
            }
        )
    return tuple(details)


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    return float(value)


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    return int(value)


async def _handle_stream_event(
    state: _ControlledBattleState,
    server: _FoulPlayWebsocketServer,
    event: Mapping[str, Any],
) -> None:
    stream = event.get("stream")
    raw_lines = event.get("lines")
    if not isinstance(stream, str) or not isinstance(raw_lines, list):
        raise RuntimeError(f"malformed BattleStream event: {event!r}")
    lines = [str(line) for line in raw_lines if str(line)]
    if stream == "omniscient":
        state.public_lines.extend(lines)
    elif stream in {"p1", "p2"}:
        for line in lines:
            if line.startswith("|request|"):
                state.request_lines[stream] = line
        if stream == "p2":
            forwarded = [_line_for_foulplay(state, line) for line in lines]
            for chunk in _line_chunks_safe_for_foulplay(forwarded):
                await server.send_room_lines(state.battle_id, chunk)
            if any(_is_terminal_protocol_line(line) for line in forwarded):
                state.foulplay_terminal_sent = True


async def _notify_foulplay_terminal(
    *,
    state: _ControlledBattleState,
    server: _FoulPlayWebsocketServer,
    terminal: TerminalState,
    config: ControlledFoulPlayConfig,
) -> None:
    if state.foulplay_terminal_sent:
        return
    line = _terminal_line_for_foulplay(terminal, config)
    await server.send_room_lines(state.battle_id, [line])
    state.foulplay_terminal_sent = True


def _terminal_line_for_foulplay(
    terminal: TerminalState,
    config: ControlledFoulPlayConfig,
) -> str:
    winner = _winner_name(terminal, config)
    if winner is None:
        return "|tie|"
    return f"|win|{winner}"


def _is_terminal_protocol_line(line: str) -> bool:
    return line.startswith("|win|") or line == "|tie" or line.startswith("|tie|")


def _line_for_foulplay(state: _ControlledBattleState, line: str) -> str:
    if not line.startswith("|request|"):
        return line
    payload = json.loads(line[len("|request|") :])
    if isinstance(payload, dict) and "rqid" not in payload:
        payload = dict(payload)
        payload["rqid"] = state.next_foulplay_rqid
        state.next_foulplay_rqid += 1
        return "|request|" + json.dumps(payload, separators=(",", ":"))
    return line


def _line_chunks_safe_for_foulplay(lines: Sequence[str]) -> tuple[tuple[str, ...], ...]:
    """Filter and chunk BattleStream lines into messages foul-play can parse.

    foul-play uses the first pipe-delimited command in a websocket message to decide how to parse
    the whole block. BattleStream can put metadata before ``|player|`` or ``|request|`` in the same
    chunk, so force those parser-sensitive lines to the front of their own messages.
    """

    safe_lines = tuple(
        line
        for line in lines
        if line and line != "|" and not line.startswith("|t:|")
    )
    chunks: list[tuple[str, ...]] = []
    current: list[str] = []
    for line in safe_lines:
        if line.startswith("|player|") or line.startswith("|request|"):
            if current:
                chunks.append(tuple(current))
                current = []
            chunks.append((line,))
        else:
            current.append(line)
    if current:
        chunks.append(tuple(current))
    return tuple(chunks)


async def _handle_decision_boundary(
    *,
    config: ControlledFoulPlayConfig,
    bridge: _BattleBridge,
    server: _FoulPlayWebsocketServer,
    state: _ControlledBattleState,
    policy: Policy,
    vocab: CategoryVocabulary,
    dex: ShowdownDex,
    observation_spec: Any,
    decision_round: int,
    requested_players: tuple[PlayerId, ...],
    foulplay_process: asyncio.subprocess.Process,
    foulplay_logs: _ProcessLogBuffer,
) -> TerminalState | None:
    assert state.trajectory is not None
    player_states = {
        player: _player_state(state, player)
        for player in requested_players
    }
    observations = {
        player: _observation_with_search_metadata(
            observation_from_player_state(
                player_states[player],
                category_vocab=vocab,
                spec=observation_spec,
                dex=dex,
            ),
            player_states[player],
        )
        for player in requested_players
    }
    choices: dict[PlayerId, str] = {}
    decisions: dict[PlayerId, PolicyDecision] = {}
    if "p1" in requested_players:
        p1_context = PolicyContext(
            player_id="p1",
            decision_round_index=decision_round,
            battle_id=state.battle_id,
            format_id=config.format_id,
            seed=state.seed,
            observation=observations["p1"],
            requested_players=requested_players,
            trajectory=state.trajectory,
            requested_legal_action_masks=_requested_legal_action_masks_for_context(
                observations,
                acting_player="p1",
                opponent_legal_mask_mode=config.opponent_legal_mask_mode,
            ),
            requested_observations=dict(observations),
        )
        decisions["p1"] = await asyncio.to_thread(
            _select_policy_decision,
            policy,
            observations["p1"],
            p1_context,
            seed=state.seed,
        )
        choices["p1"] = showdown_choice_for_action(player_states["p1"], decisions["p1"].action_index)
    if "p2" in requested_players:
        choice = await _wait_for_foulplay_choice_or_exit(
            server=server,
            battle_id=state.battle_id,
            process=foulplay_process,
            logs=foulplay_logs,
        )
        p2_action = action_index_from_choice_string(player_states["p2"], choice)
        if p2_action is None:
            raise RuntimeError(f"unable to decode foul-play choice {choice!r}.")
        choices["p2"] = choice
        decisions["p2"] = PolicyDecision(
            action_index=p2_action,
            policy_id="foul-play",
            metadata={"raw_choice": choice},
        )

    for player in requested_players:
        decision = decisions.get(player)
        if decision is None:
            continue
        state.trajectory.append(
            TrajectoryStep(
                player_id=player,
                turn_index=decision_round,
                observation=observations[player],
                legal_action_mask=tuple(observations[player].legal_action_mask),
                action_index=decision.action_index,
                metadata={"policy_id": decision.policy_id, **dict(decision.metadata)},
            )
        )
        if player == "p1":
            state.decisions.append(decision)

    await bridge.send({"type": "choices", "battleId": state.battle_id, "choices": choices})
    return None


def _observation_with_search_metadata(
    observation: PokeZeroObservationV0,
    state: PlayerRelativeBattleState,
) -> PokeZeroObservationV0:
    return replace(
        observation,
        metadata={
            **dict(observation.metadata),
            "belief_view": state.belief_view.to_overlay_payload(),
        },
    )


async def _wait_for_foulplay_choice_or_exit(
    *,
    server: _FoulPlayWebsocketServer,
    battle_id: str,
    process: asyncio.subprocess.Process,
    logs: _ProcessLogBuffer,
) -> str:
    if process.returncode is not None:
        raise RuntimeError(f"foul-play exited with status {process.returncode} before choosing.\n{logs.tail()}")
    choice_task = asyncio.create_task(server.wait_for_choice(battle_id=battle_id))
    process_task = asyncio.create_task(process.wait())
    try:
        done, pending = await asyncio.wait(
            {choice_task, process_task},
            return_when=asyncio.FIRST_COMPLETED,
        )
        if choice_task in done:
            return choice_task.result()
        raise RuntimeError(
            f"foul-play exited with status {process.returncode} before choosing.\n{logs.tail()}"
        )
    finally:
        for task in (choice_task, process_task):
            if not task.done():
                task.cancel()


async def _wait_for_foulplay_challenge_or_exit(
    *,
    server: _FoulPlayWebsocketServer,
    expected_target: str,
    process: asyncio.subprocess.Process,
    logs: _ProcessLogBuffer,
) -> None:
    if process.returncode is not None:
        raise RuntimeError(f"foul-play exited with status {process.returncode} before challenging.\n{logs.tail()}")
    challenge_task = asyncio.create_task(server.wait_for_challenge(expected_target=expected_target))
    process_task = asyncio.create_task(process.wait())
    try:
        done, pending = await asyncio.wait(
            {challenge_task, process_task},
            return_when=asyncio.FIRST_COMPLETED,
        )
        if challenge_task in done:
            challenge_task.result()
            return
        raise RuntimeError(
            f"foul-play exited with status {process.returncode} before challenging.\n{logs.tail()}"
        )
    finally:
        for task in (challenge_task, process_task):
            if not task.done():
                task.cancel()


def _select_policy_decision(
    policy: Policy,
    observation: PokeZeroObservationV0,
    context: PolicyContext,
    *,
    seed: int,
) -> PolicyDecision:
    rng = random.Random(f"{seed}:{context.player_id}:{context.decision_round_index}")
    selector = getattr(policy, "select_action_with_context", None)
    if callable(selector):
        return selector(context, rng=rng)
    return policy.select_action(observation, rng=rng)


def _requested_legal_action_masks_for_context(
    observations: Mapping[PlayerId, PokeZeroObservationV0],
    *,
    acting_player: PlayerId,
    opponent_legal_mask_mode: str,
) -> dict[PlayerId, tuple[bool, ...]]:
    masks: dict[PlayerId, tuple[bool, ...]] = {}
    for player, observation in observations.items():
        if player != acting_player and opponent_legal_mask_mode == "hidden":
            continue
        masks[player] = tuple(observation.legal_action_mask)
    return masks


def _player_state(state: _ControlledBattleState, player: PlayerId) -> PlayerRelativeBattleState:
    replay = parse_showdown_replay(state.all_lines(), battle_id=state.battle_id)
    return normalize_for_player(
        replay,
        player_id=player,
        configured_showdown_slot=player,
        format_id=state.format_id,
    )


def _terminal_from_public_lines(
    lines: Sequence[str],
    config: ControlledFoulPlayConfig,
) -> TerminalState | None:
    turn = 0
    winner: PlayerId | None = None
    for line in lines:
        if line.startswith("|turn|"):
            try:
                turn = int(line.split("|", 2)[2])
            except (IndexError, ValueError):
                pass
        elif line.startswith("|win|"):
            winner_name = line.split("|", 2)[2] if len(line.split("|", 2)) >= 3 else ""
            if winner_name == config.pokezero_username:
                winner = "p1"
            elif winner_name == config.foulplay_username:
                winner = "p2"
            return TerminalState(winner=winner, turn_count=turn)
        elif line == "|tie" or line.startswith("|tie|"):
            return TerminalState(winner=None, turn_count=turn)
    return None


def _winner_name(terminal: TerminalState, config: ControlledFoulPlayConfig) -> str | None:
    if terminal.winner == "p1":
        return config.pokezero_username
    if terminal.winner == "p2":
        return config.foulplay_username
    return None


def _split_outgoing_showdown_message(message: str) -> tuple[str, str]:
    if "|" not in message:
        return "", message.strip()
    room, body = message.split("|", 1)
    return room.strip(), body.strip()


def _choice_body_from_outgoing_message(body: str) -> str | None:
    command = body.split("|", 1)[0].strip()
    if command.startswith("/choose "):
        return command[len("/choose ") :].strip()
    if command.startswith("/switch "):
        return f"switch {command[len('/switch ') :].strip()}"
    return None


def _showdown_id(value: str) -> str:
    return "".join(ch for ch in str(value).lower() if ch.isalnum())


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    tmp.replace(path)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a controlled BattleStream benchmark: PokeZero policy vs external foul-play.",
    )
    parser.add_argument("--checkpoint", type=Path, required=True, help="Transformer checkpoint path.")
    parser.add_argument(
        "--showdown-root",
        type=Path,
        default=Path(os.environ.get("POKEZERO_SHOWDOWN_ROOT", "")) if os.environ.get("POKEZERO_SHOWDOWN_ROOT") else None,
        help="Built Pokemon Showdown checkout root, or POKEZERO_SHOWDOWN_ROOT.",
    )
    parser.add_argument("--foulplay-root", type=Path, default=DEFAULT_FOULPLAY_ROOT, help="foul-play checkout path.")
    parser.add_argument("--foulplay-python", type=Path, default=None, help="Python executable for foul-play.")
    parser.add_argument("--games", type=int, default=1, help="Number of games.")
    parser.add_argument("--seed-start", type=int, default=1, help="First deterministic BattleStream seed.")
    parser.add_argument(
        "--foulplay-random-seed",
        type=int,
        default=None,
        help=(
            "Seed for foul-play's Python random/hash startup state. Defaults to --seed-start. "
            "This controls foul-play's random stream but does not make wall-clock MCTS fully deterministic."
        ),
    )
    parser.add_argument("--search-time-ms", type=int, default=1000, help="foul-play search time per move.")
    parser.add_argument("--max-decision-rounds", type=int, default=250, help="Decision-round cap.")
    parser.add_argument("--format", dest="format_id", default="gen3randombattle", help="Showdown format id.")
    parser.add_argument("--policy-mode", choices=("raw", "root-puct"), default="root-puct")
    parser.add_argument("--device", default=None, help="Torch device, e.g. cpu, cuda, mps.")
    parser.add_argument("--temperature", type=float, default=1.0, help="Checkpoint policy softmax temperature.")
    parser.add_argument("--cpuct", type=float, default=1.25, help="Root PUCT exploration constant.")
    parser.add_argument(
        "--selection-mode",
        choices=("puct", "value", "visits"),
        default="visits",
        help=(
            "Root search candidate selection rule. Defaults to 'visits', which uses PUCT's "
            "exploration term for traversal but selects the most-visited root action. 'puct' "
            "selects by final Q+U score and should be treated as diagnostic."
        ),
    )
    parser.add_argument(
        "--root-prior-temperature",
        type=float,
        default=None,
        help=(
            "Temperature applied only to root-PUCT action priors. Defaults to --temperature, "
            "while opponent-action priors and fallback policy continue using --temperature."
        ),
    )
    parser.add_argument(
        "--minimum-value-improvement",
        type=float,
        default=None,
        help=(
            "Require the search-selected action to beat the prior-best action by this value margin; "
            "otherwise use the prior-best action."
        ),
    )
    parser.add_argument(
        "--minimum-override-prior-ratio",
        type=float,
        default=None,
        help=(
            "When search would override the checkpoint prior's greedy legal action, require the "
            "selected action prior to be at least this fraction of the prior-best action prior. "
            "A value of 1.0 only allows max-prior ties to override."
        ),
    )
    parser.add_argument(
        "--minimum-score-improvement",
        type=float,
        default=None,
        help=(
            "When search would override the checkpoint prior's greedy legal action, require the "
            "selected action's root-PUCT score to be at least this much higher than the prior-best "
            "action's score. Use 0.0 to reject lower-score overrides."
        ),
    )
    parser.add_argument(
        "--root-visit-budget",
        type=int,
        default=16,
        help=(
            "Root visits per opponent-action scenario; defaults to 16. "
            "With multiple scenarios, total decision visits scale by the searched scenario count."
        ),
    )
    parser.add_argument(
        "--root-time-budget-ms",
        type=int,
        default=None,
        help=(
            "PokeZero-side wall-clock budget for extra post-sweep root visits. With multiple "
            "opponent-action scenarios, each scenario receives the remaining decision budget at "
            "the time it is searched. The mandatory initial legal-action sweep is always completed "
            "and can exceed the configured budget; --root-visit-budget remains a per-scenario hard cap."
        ),
    )
    parser.add_argument(
        "--root-opponent-action-scenarios",
        type=int,
        default=1,
        help="Number of checkpoint-prior opponent root-action scenarios to average.",
    )
    parser.add_argument(
        "--root-opponent-action-candidate-scenarios",
        type=int,
        default=ACTION_COUNT,
        help=(
            "Number of checkpoint-prior opponent root-action candidates to try while searching "
            "for replay-legal scenarios. Defaults to the full action space; when the opponent "
            "legal mask is hidden, exchangeable switch slots are collapsed into one summed switch "
            "candidate before this cap is applied. The search stops after "
            "--root-opponent-action-scenarios legal scenarios are accepted."
        ),
    )
    parser.add_argument(
        "--leaf-rollout-rounds",
        type=int,
        default=0,
        help="Decision rounds to continue each root branch before leaf value evaluation.",
    )
    parser.add_argument(
        "--leaf-rollout-sampling",
        action="store_true",
        help="Use sampled checkpoint policies, rather than greedy policies, inside leaf rollouts.",
    )
    parser.add_argument(
        "--belief-start-overrides",
        action="store_true",
        help=(
            "Sample public Gen 3 randbat belief into complete custom-game branch starts for "
            "root-PUCT replay search. This is hidden-info safe but experimental."
        ),
    )
    parser.add_argument(
        "--start-override-attempts",
        type=int,
        default=1,
        help=(
            "Replay-consistency attempts per opponent-action scenario when a start-override "
            "planner is enabled. Higher values rejection-sample more hidden worlds before falling "
            "back."
        ),
    )
    parser.add_argument(
        "--belief-start-override-samples",
        type=int,
        default=1,
        help=(
            "Belief start-override samples to average per accepted opponent-action scenario. "
            "Requires --belief-start-overrides. Values above 1 split each opponent-action "
            "scenario across multiple sampled hidden worlds without increasing the accepted "
            "opponent-action cap, increasing search cost."
        ),
    )
    parser.add_argument(
        "--start-override-hp-fraction-tolerance",
        type=float,
        default=0.02,
        help=(
            "Allowed branch-point HP-fraction drift when validating sampled start overrides. "
            "Only self/opponent Pokemon HP-fraction numeric cells use this tolerance; request "
            "shape, legal mask, action candidates, categorical state, status, and all other "
            "numeric features remain exact."
        ),
    )
    parser.add_argument(
        "--opponent-legal-mask-mode",
        choices=("hidden", "privileged"),
        default="hidden",
        help=(
            "Whether root opponent-action planning withholds the opponent's private legal mask "
            "(hidden, default) or uses it as a privileged benchmark safety guard."
        ),
    )
    parser.add_argument(
        "--no-search-fallback",
        action="store_true",
        help="Raise on search failure instead of falling back to the raw checkpoint action.",
    )
    parser.add_argument("--node-binary", default="node", help="Node executable for BattleStream bridge.")
    parser.add_argument("--pokezero-username", default="PokeZeroBot")
    parser.add_argument("--foulplay-username", default="FoulPlayBot")
    parser.add_argument("--summary-out", type=Path, default=None, help="Optional JSON result path.")
    parser.add_argument("--json", action="store_true", help="Print JSON result.")
    return parser


def build_comparison_arg_parser() -> argparse.ArgumentParser:
    parser = build_arg_parser()
    _remove_optional_argument(parser, "--policy-mode")
    parser.set_defaults(policy_mode="root-puct")
    parser.add_argument(
        "--comparison-mode",
        choices=tuple(sorted(_COMPARISON_MODES)),
        default="per-seed",
        help=(
            "Comparison execution order. 'per-seed' runs raw and root-PUCT for each seed before "
            "advancing, restarting foul-play with a matching per-seed startup seed and producing "
            "paired partial progress earlier. 'per-arm' preserves the older raw-all-then-root-PUCT "
            "order and is mainly useful when process startup overhead dominates."
        ),
    )
    parser.description = (
        "Run paired controlled BattleStream benchmarks: raw checkpoint and root-PUCT "
        "against external foul-play over the same seed band."
    )
    parser.epilog = "The comparison runner always runs both raw and root-puct policy modes."
    return parser


def _remove_optional_argument(parser: argparse.ArgumentParser, option: str) -> None:
    for action in tuple(parser._actions):
        if option not in action.option_strings:
            continue
        parser._remove_action(action)
        for group in parser._action_groups:
            if action in group._group_actions:
                group._group_actions.remove(action)
        for option_string in action.option_strings:
            parser._option_string_actions.pop(option_string, None)
        return
    raise AssertionError(f"parser option not found: {option}")


def _config_from_args(
    args: argparse.Namespace,
    *,
    policy_mode: str | None = None,
) -> ControlledFoulPlayConfig:
    return ControlledFoulPlayConfig(
        checkpoint=args.checkpoint,
        showdown_root=args.showdown_root,
        foulplay_root=args.foulplay_root,
        foulplay_python=args.foulplay_python,
        games=args.games,
        seed_start=args.seed_start,
        foulplay_random_seed=args.foulplay_random_seed,
        search_time_ms=args.search_time_ms,
        max_decision_rounds=args.max_decision_rounds,
        format_id=args.format_id,
        policy_mode=policy_mode if policy_mode is not None else args.policy_mode,
        device=args.device,
        temperature=args.temperature,
        cpuct=args.cpuct,
        selection_mode=args.selection_mode,
        root_prior_temperature=args.root_prior_temperature,
        minimum_value_improvement=args.minimum_value_improvement,
        minimum_override_prior_ratio=args.minimum_override_prior_ratio,
        minimum_score_improvement=args.minimum_score_improvement,
        root_visit_budget=args.root_visit_budget,
        root_time_budget_ms=args.root_time_budget_ms,
        root_opponent_action_scenarios=args.root_opponent_action_scenarios,
        root_opponent_action_candidate_scenarios=args.root_opponent_action_candidate_scenarios,
        leaf_rollout_rounds=args.leaf_rollout_rounds,
        leaf_rollout_sampling=args.leaf_rollout_sampling,
        belief_start_overrides=args.belief_start_overrides,
        start_override_attempts=args.start_override_attempts,
        belief_start_override_samples=args.belief_start_override_samples,
        start_override_hp_fraction_tolerance=args.start_override_hp_fraction_tolerance,
        opponent_legal_mask_mode=args.opponent_legal_mask_mode,
        allow_search_fallback=not args.no_search_fallback,
        node_binary=args.node_binary,
        pokezero_username=args.pokezero_username,
        foulplay_username=args.foulplay_username,
    )


async def async_main(argv: Sequence[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    if args.showdown_root is None:
        parser.error("--showdown-root is required, or set POKEZERO_SHOWDOWN_ROOT.")
    config = _config_from_args(args)

    def write_progress(result: ControlledFoulPlayBenchmarkResult) -> None:
        if args.summary_out is not None:
            _write_json(args.summary_out, result.to_dict())

    result = await run_controlled_foulplay_benchmark(
        config,
        progress_callback=write_progress if args.summary_out is not None else None,
    )
    payload = result.to_dict()
    if args.summary_out is not None:
        _write_json(args.summary_out, payload)
        print(f"controlled_foulplay_summary: {args.summary_out}", file=sys.stderr)
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print(
            f"RESULT: {result.policy_id} won {result.wins}/{result.completed_games} "
            f"vs foul-play ({result.win_rate:.1%})"
        )
        root = payload["root_puct"]
        if isinstance(root, Mapping) and root.get("searches"):
            print(
                "root-puct: "
                f"searches={root.get('searches')} fallbacks={root.get('fallbacks')} "
                f"avg_elapsed={root.get('average_elapsed_seconds', 'n/a')}"
            )
    return 0


async def async_comparison_main(argv: Sequence[str] | None = None) -> int:
    parser = build_comparison_arg_parser()
    args = parser.parse_args(argv)
    if args.showdown_root is None:
        parser.error("--showdown-root is required, or set POKEZERO_SHOWDOWN_ROOT.")
    config = _config_from_args(args, policy_mode="root-puct")

    def write_progress(result: ControlledFoulPlayComparisonResult) -> None:
        if args.summary_out is not None:
            _write_json(args.summary_out, result.to_dict())

    result = await run_controlled_foulplay_comparison(
        config,
        comparison_mode=args.comparison_mode,
        progress_callback=write_progress if args.summary_out is not None else None,
    )
    payload = result.to_dict()
    if args.summary_out is not None:
        _write_json(args.summary_out, payload)
        print(f"controlled_foulplay_comparison_summary: {args.summary_out}", file=sys.stderr)
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        comparison = payload["comparison"]
        paired = comparison["paired_by_seed"] if isinstance(comparison, Mapping) else {}
        raw = paired.get("raw", {}) if isinstance(paired, Mapping) else {}
        root_puct = paired.get("root_puct", {}) if isinstance(paired, Mapping) else {}
        sample = comparison["sample_size"] if isinstance(comparison, Mapping) else {}
        result_label = (
            "DIAGNOSTIC RESULT"
            if isinstance(sample, Mapping) and sample.get("status") == "diagnostic_only"
            else "RESULT"
        )
        delta = paired.get("root_puct_minus_raw_win_rate") if isinstance(paired, Mapping) else None
        delta_text = "n/a" if delta is None else f"{float(delta):.1%}"
        print(
            f"{result_label}: root-PUCT "
            f"{int(root_puct.get('wins', 0))}/{int(root_puct.get('games', 0))} "
            "vs raw "
            f"{int(raw.get('wins', 0))}/{int(raw.get('games', 0))} "
            f"on paired foul-play seeds ({args.comparison_mode}) "
            f"(descriptive_delta={delta_text})"
        )
        if isinstance(sample, Mapping) and sample.get("status") == "diagnostic_only":
            print(
                "sample-size: diagnostic_only "
                f"({sample.get('paired_games')}/{sample.get('minimum_strength_games')} paired games)"
            )
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    return asyncio.run(async_main(argv))


def comparison_main(argv: Sequence[str] | None = None) -> int:
    return asyncio.run(async_comparison_main(argv))


if __name__ == "__main__":
    raise SystemExit(main())
