"""Public-safe Root-PUCT benchmark telemetry and reporting helpers.

The search policy already emits detailed metadata while deciding a move.  This
module selects the stable, aggregate-safe subset that ordinary benchmark
artifacts can retain: no observations, action histories, or verbose replay
errors are persisted.  The resulting records are intentionally useful for
both legality diagnosis and wall-clock budget studies.
"""

from __future__ import annotations

import math
from typing import Any, Iterable, Mapping, Sequence

from .mcts_diagnostics import (
    root_puct_fallback_signature,
    sanitize_root_puct_direct_materialization_rejection_categories,
    sanitize_root_puct_missing_sampled_world_reason_categories,
)


ROOT_PUCT_DECISION_TELEMETRY_SCHEMA_VERSION = "pokezero.root_puct_decision_telemetry.v1"
ROOT_PUCT_TELEMETRY_REPORT_SCHEMA_VERSION = "pokezero.root_puct_telemetry_report.v1"

_SCALAR_COUNT_FIELDS = (
    "root_puct_total_visits",
    "root_puct_effective_total_visits",
    # W5 records distinct reconstructed sampled worlds that supplied at least
    # one completed branch search, split by direct public-state construction
    # versus the Tier 1 replay fallback.
    "root_puct_start_override_direct_materializations",
    "root_puct_start_override_replay_materializations",
    "root_puct_opponent_action_scenario_count",
    "root_puct_opponent_action_scenarios_generated",
    "root_puct_opponent_action_scenarios_skipped",
    "root_puct_opponent_action_scenarios_unsearched",
    "root_puct_opponent_action_groups_generated",
    "root_puct_opponent_action_groups_used",
    "root_puct_opponent_action_groups_skipped",
    "root_puct_opponent_action_groups_unsearched",
)

_COUNTER_MAP_FIELDS = (
    # Direct construction is allowed to fail closed to the Tier 1 replay path.
    # Keep only a compact category so W5 diagnostics can identify why without
    # persisting bridge exception text or battle state.
    "root_puct_direct_materialization_rejection_categories",
    "root_puct_direct_materialization_observation_mismatch_paths",
    "root_puct_opponent_action_skip_categories",
    "root_puct_opponent_action_replay_rejection_decision_rounds",
    "root_puct_opponent_action_replay_request_mismatch_decision_rounds",
    "root_puct_opponent_action_replay_request_mismatch_players",
    "root_puct_opponent_action_replay_request_mismatch_shapes",
    "root_puct_opponent_action_start_override_mismatch_decision_rounds",
    "root_puct_opponent_action_first_observation_mismatch_paths",
    "root_puct_opponent_action_missing_sampled_world_reason_categories",
)

_TIMING_KEYS = (
    "prefix_replay_seconds",
    "prefix_replay_count",
    "branch_simulator_step_seconds",
    "branch_simulator_step_count",
    # Nested branch-step slices from bridge-resident sampled worlds. They
    # attribute fused restore-and-step wall without changing additive totals.
    "branch_local_state_restore_seconds",
    "branch_local_state_restore_count",
    "branch_choice_encoding_seconds",
    "branch_choice_encoding_count",
    "branch_bridge_round_trip_seconds",
    "branch_bridge_round_trip_count",
    "branch_bridge_node_processing_seconds",
    "branch_bridge_node_processing_count",
    "branch_bridge_python_orchestration_seconds",
    "branch_bridge_python_orchestration_count",
    "branch_result_projection_seconds",
    "branch_result_projection_count",
    "branch_observation_projection_seconds",
    "branch_observation_projection_count",
    "branch_observation_state_normalization_seconds",
    "branch_observation_state_normalization_count",
    "branch_observation_incremental_sync_seconds",
    "branch_observation_incremental_sync_count",
    "branch_observation_replay_snapshot_seconds",
    "branch_observation_replay_snapshot_count",
    "branch_observation_player_state_normalization_seconds",
    "branch_observation_player_state_normalization_count",
    "branch_observation_state_annotation_seconds",
    "branch_observation_state_annotation_count",
    "branch_observation_encoding_seconds",
    "branch_observation_encoding_count",
    "branch_belief_overlay_projection_seconds",
    "branch_belief_overlay_projection_count",
    "branch_observation_raw_unattributed_seconds",
    "branch_observation_unattributed_seconds",
    "branch_observation_state_normalization_raw_unattributed_seconds",
    "branch_observation_state_normalization_unattributed_seconds",
    "branch_projection_raw_unattributed_seconds",
    "branch_projection_unattributed_seconds",
    "branch_raw_unattributed_seconds",
    "branch_unattributed_seconds",
    "state_snapshot_seconds",
    "state_snapshot_count",
    "state_restore_seconds",
    "state_restore_count",
    # Additive Python orchestration stages. Together with the simulator and
    # evaluator buckets, these shrink residual_seconds to only uncategorized work.
    "root_initial_sweep_orchestration_seconds",
    "root_initial_sweep_orchestration_count",
    "root_search_setup_seconds",
    "root_search_setup_count",
    "root_adaptive_visit_orchestration_seconds",
    "root_adaptive_visit_orchestration_count",
    "root_search_finalization_seconds",
    "root_search_finalization_count",
    "branch_action_validation_seconds",
    "branch_action_validation_count",
    "post_branch_history_seconds",
    "post_branch_history_count",
    # These are nested within snapshot/restore/step stages. They split bridge
    # wall into Node simulator work and local IPC/Python orchestration without
    # changing residual accounting.
    "bridge_round_trip_seconds",
    "bridge_round_trip_count",
    "bridge_node_processing_seconds",
    "bridge_node_processing_count",
    "bridge_python_orchestration_seconds",
    "bridge_python_orchestration_count",
    "belief_world_materialization_seconds",
    "belief_world_materialization_count",
    "opponent_scenario_planning_seconds",
    "opponent_scenario_planning_count",
    "root_policy_setup_seconds",
    "root_policy_setup_count",
    "direct_prefix_construction_seconds",
    "direct_prefix_construction_count",
    "scenario_dispatch_orchestration_seconds",
    "scenario_dispatch_orchestration_count",
    "policy_evaluation_seconds",
    "policy_evaluation_count",
    # These overlap policy/value/scenario timings and are diagnostic W2
    # sub-slices rather than additive components of total_seconds.
    "observation_encoding_seconds",
    "observation_encoding_count",
    "neural_forward_seconds",
    "neural_forward_count",
    "action_prior_neural_forward_seconds",
    "action_prior_neural_forward_count",
    "opponent_action_prior_neural_forward_seconds",
    "opponent_action_prior_neural_forward_count",
    "policy_neural_forward_seconds",
    "policy_neural_forward_count",
    "value_neural_forward_seconds",
    "value_neural_forward_count",
    "value_evaluation_seconds",
    "value_evaluation_count",
    # Adaptive leaves are semantically evaluated one per sampled world. The
    # second counter credits only leaves that actually joined a cross-world
    # batch, so telemetry can distinguish enabled batching from execution.
    "adaptive_value_evaluation_count",
    "adaptive_cross_world_batched_leaf_count",
    "rollout_tail_seconds",
    "rollout_tail_count",
    "policy_value_evaluation_seconds",
    "policy_value_evaluation_count",
    # W5 residual partition. These are diagnostic sub-slices of residual
    # time, not new additive wall-clock stages.
    "puct_search_result_residual_seconds",
    "puct_search_result_residual_count",
    "puct_search_completed_call_seconds",
    "puct_search_completed_call_count",
    "puct_search_retained_completed_call_seconds",
    "puct_search_retained_completed_call_count",
    "puct_search_completed_result_seconds",
    "puct_search_completed_result_count",
    "puct_search_completed_call_overhead_seconds",
    "puct_search_discarded_completed_call_seconds",
    "puct_search_discarded_completed_call_count",
    "puct_search_rejected_call_seconds",
    "puct_search_rejected_call_count",
    "puct_search_unrecorded_call_seconds",
    "puct_search_call_count",
    "raw_residual_seconds",
    "residual_seconds",
    "raw_outer_policy_residual_seconds",
    "outer_policy_residual_seconds",
    "total_seconds",
)

_SCENARIO_COUNT_NAMES = {
    "root_puct_opponent_action_scenario_count": "scenarios_used",
    "root_puct_opponent_action_scenarios_generated": "scenarios_generated",
    "root_puct_opponent_action_scenarios_skipped": "scenarios_skipped",
    "root_puct_opponent_action_scenarios_unsearched": "scenarios_unsearched",
    "root_puct_opponent_action_groups_generated": "groups_generated",
    "root_puct_opponent_action_groups_used": "groups_used",
    "root_puct_opponent_action_groups_skipped": "groups_skipped",
    "root_puct_opponent_action_groups_unsearched": "groups_unsearched",
}

_MATERIALIZATION_COUNT_NAMES = {
    "root_puct_start_override_direct_materializations": "direct",
    "root_puct_start_override_replay_materializations": "replay",
}

_COUNTER_TAXONOMY_NAMES = {
    "root_puct_direct_materialization_rejection_categories": "direct_materialization_rejection_categories",
}


def root_puct_decision_telemetry(
    metadata: Mapping[str, Any],
    *,
    decision_index: int,
    turn_index: int,
) -> dict[str, object] | None:
    """Return the stable diagnostic subset for one Root-PUCT decision.

    The raw fallback reason is deliberately omitted: replay errors can include
    observation values, while the category and request-shape counters are the
    information needed to fix the underlying control-flow issue.
    """

    if metadata.get("policy_family") != "root-puct-search":
        return None
    fallback = bool(metadata.get("root_puct_fallback"))
    payload: dict[str, object] = {
        "schema_version": ROOT_PUCT_DECISION_TELEMETRY_SCHEMA_VERSION,
        "decision_index": decision_index,
        "turn_index": turn_index,
        "outcome": "fallback" if fallback else "searched",
        "fallback": fallback,
    }
    if fallback:
        payload["fallback_category"] = str(metadata.get("root_puct_fallback_category") or "unknown")
        signature = root_puct_fallback_signature(metadata.get("root_puct_fallback_reason"))
        if signature is not None:
            payload["fallback_signature"] = signature
    for field in _SCALAR_COUNT_FIELDS:
        value = _nonnegative_int(metadata.get(field))
        if value is not None:
            payload[field] = value
    for field in ("root_puct_elapsed_seconds", "policy_elapsed_seconds"):
        value = _finite_nonnegative_float(metadata.get(field))
        if value is not None:
            payload[field] = value
    timing = _compact_timing(metadata.get("root_puct_timing"))
    if timing:
        payload["timing"] = timing
    # Dispatch timing is recorded for root-puct-play-benchmark and includes
    # legality failures that return before internal search timing exists.
    full_decision_elapsed = _finite_nonnegative_float(metadata.get("policy_elapsed_seconds"))
    if full_decision_elapsed is None:
        full_decision_elapsed = _finite_nonnegative_float(timing.get("total_seconds"))
    if full_decision_elapsed is not None:
        payload["full_decision_elapsed_seconds"] = full_decision_elapsed
    counters: dict[str, Mapping[str, int]] = {}
    for field in _COUNTER_MAP_FIELDS:
        source = metadata.get(field)
        value = (
            sanitize_root_puct_direct_materialization_rejection_categories(source)
            if field == "root_puct_direct_materialization_rejection_categories"
            else sanitize_root_puct_missing_sampled_world_reason_categories(source)
            if field == "root_puct_opponent_action_missing_sampled_world_reason_categories"
            else _counter_map(source)
        )
        if value:
            counters[field] = value
    if counters:
        payload["counters"] = counters
    return payload


def summarize_root_puct_decision_telemetry(
    decisions: Iterable[Mapping[str, object]],
) -> dict[str, object]:
    """Aggregate compact Root-PUCT decisions into a W1/W2 readout."""

    records = tuple(dict(item) for item in decisions)
    searched = sum(1 for item in records if item.get("outcome") == "searched")
    fallbacks = sum(1 for item in records if item.get("outcome") == "fallback")
    total = len(records)
    fallback_categories: dict[str, int] = {}
    fallback_signatures: dict[str, int] = {}
    counters: dict[str, dict[str, int]] = {field: {} for field in _COUNTER_MAP_FIELDS}
    timing_totals: dict[str, float | int] = {}
    root_elapsed_samples: list[float] = []
    policy_elapsed_samples: list[float] = []
    full_decision_elapsed_samples: list[float] = []
    total_visits = 0
    effective_total_visits = 0
    scenario_counts: dict[str, int] = {}
    materialization_counts = {name: 0 for name in _MATERIALIZATION_COUNT_NAMES.values()}
    for item in records:
        if item.get("outcome") == "fallback":
            category = str(item.get("fallback_category") or "unknown")
            fallback_categories[category] = fallback_categories.get(category, 0) + 1
            signature = item.get("fallback_signature")
            if isinstance(signature, str) and signature:
                fallback_signatures[signature] = fallback_signatures.get(signature, 0) + 1
        for field in _COUNTER_MAP_FIELDS:
            counter_values = item.get("counters")
            source = counter_values.get(field) if isinstance(counter_values, Mapping) else None
            if field == "root_puct_direct_materialization_rejection_categories":
                source = sanitize_root_puct_direct_materialization_rejection_categories(source)
            elif field == "root_puct_opponent_action_missing_sampled_world_reason_categories":
                source = sanitize_root_puct_missing_sampled_world_reason_categories(source)
            _merge_counter_map(counters[field], source)
        total_visits += _nonnegative_int(item.get("root_puct_total_visits")) or 0
        effective_total_visits += _nonnegative_int(item.get("root_puct_effective_total_visits")) or 0
        for field, name in _MATERIALIZATION_COUNT_NAMES.items():
            materialization_counts[name] += _nonnegative_int(item.get(field)) or 0
        for field in _SCALAR_COUNT_FIELDS:
            if not field.startswith("root_puct_opponent_action_"):
                continue
            value = _nonnegative_int(item.get(field))
            if value is not None:
                scenario_counts[field] = scenario_counts.get(field, 0) + value
        root_elapsed = _finite_nonnegative_float(item.get("root_puct_elapsed_seconds"))
        if root_elapsed is not None:
            root_elapsed_samples.append(root_elapsed)
        policy_elapsed = _finite_nonnegative_float(item.get("policy_elapsed_seconds"))
        if policy_elapsed is not None:
            policy_elapsed_samples.append(policy_elapsed)
        full_decision_elapsed = _finite_nonnegative_float(item.get("full_decision_elapsed_seconds"))
        if full_decision_elapsed is not None:
            full_decision_elapsed_samples.append(full_decision_elapsed)
        timing = item.get("timing")
        if isinstance(timing, Mapping):
            for key, value in timing.items():
                if key not in _TIMING_KEYS:
                    continue
                if key.endswith("_count"):
                    parsed_count = _nonnegative_int(value)
                    if parsed_count is not None:
                        timing_totals[key] = int(timing_totals.get(key, 0)) + parsed_count
                else:
                    parsed_seconds = _finite_float(value)
                    if parsed_seconds is not None:
                        timing_totals[key] = float(timing_totals.get(key, 0.0)) + parsed_seconds
    timing_seconds = float(timing_totals.get("total_seconds", 0.0))
    visit_rate = total_visits / timing_seconds if timing_seconds > 0.0 else None
    return {
        "schema_version": ROOT_PUCT_DECISION_TELEMETRY_SCHEMA_VERSION,
        "decisions": total,
        "searches": searched,
        "fallbacks": fallbacks,
        "search_rate": searched / total if total else None,
        "fallback_rate": fallbacks / total if total else None,
        "fallback_categories": dict(sorted(fallback_categories.items())),
        "fallback_signatures": dict(sorted(fallback_signatures.items())),
        "scenario_counts": {
            _SCENARIO_COUNT_NAMES[field]: count
            for field, count in sorted(scenario_counts.items())
        },
        "materialization_counts": materialization_counts,
        "scenario_failure_taxonomy": {
            _COUNTER_TAXONOMY_NAMES.get(
                field,
                field.removeprefix("root_puct_opponent_action_"),
            ): dict(sorted(values.items()))
            for field, values in counters.items()
            if values
        },
        "visits": {
            "total": total_visits,
            "effective_total": effective_total_visits,
            "mean_per_search": total_visits / searched if searched else None,
            "per_root_search_second": visit_rate,
        },
        # ``root_puct_elapsed_seconds`` starts after opponent-scenario planning
        # and prior evaluation. Keep it as a branch-search diagnostic rather
        # than presenting it as the end-to-end Root-PUCT decision cost.
        "branch_search_wall_seconds": _sample_summary(root_elapsed_samples),
        "full_decision_wall_seconds": _sample_summary(full_decision_elapsed_samples),
        "policy_dispatch_wall_seconds": _sample_summary(policy_elapsed_samples),
        "timing_totals": dict(sorted(timing_totals.items())),
    }


def root_puct_benchmark_telemetry_report(
    payload: Mapping[str, object],
    *,
    policy_ids: Sequence[str] = (),
) -> dict[str, object]:
    """Summarize durable Root-PUCT telemetry from a benchmark JSON artifact."""

    requested_ids = set(policy_ids)
    by_policy: dict[str, list[Mapping[str, object]]] = {}
    matchups = payload.get("matchups")
    if not isinstance(matchups, list):
        raise ValueError("benchmark artifact must contain a matchups JSON list.")
    for matchup_index, matchup in enumerate(matchups):
        if not isinstance(matchup, Mapping):
            raise ValueError(f"benchmark artifact matchup {matchup_index} must be a JSON object.")
        player_policies = {
            "p1": matchup.get("p1_policy_id"),
            "p2": matchup.get("p2_policy_id"),
        }
        game_results = matchup.get("game_results")
        if not isinstance(game_results, list):
            raise ValueError(f"benchmark artifact matchup {matchup_index} is missing game_results.")
        for game_index, game in enumerate(game_results):
            if not isinstance(game, Mapping):
                raise ValueError(
                    f"benchmark artifact matchup {matchup_index} game {game_index} must be a JSON object."
                )
            by_player = game.get("root_puct_decision_telemetry_by_player")
            if by_player is None:
                by_player = {}
            if not isinstance(by_player, Mapping):
                raise ValueError(
                    f"benchmark artifact matchup {matchup_index} game {game_index} has invalid "
                    "root_puct_decision_telemetry_by_player."
                )
            root_puct_by_player = game.get("root_puct_by_player")
            if root_puct_by_player is None:
                root_puct_by_player = {}
            if not isinstance(root_puct_by_player, Mapping):
                raise ValueError(
                    f"benchmark artifact matchup {matchup_index} game {game_index} has invalid root_puct_by_player."
                )
            for player, policy_id in player_policies.items():
                if not isinstance(policy_id, str) or not policy_id:
                    continue
                entries = by_player.get(player)
                requested = policy_id in requested_ids
                is_root_puct_seat = (
                    requested
                    or player in root_puct_by_player
                    or _looks_like_root_puct_policy(policy_id)
                )
                if entries is None:
                    if is_root_puct_seat:
                        raise ValueError(
                            f"benchmark artifact matchup {matchup_index} game {game_index} is missing Root-PUCT "
                            f"telemetry for {policy_id} {player}."
                        )
                    continue
                if not isinstance(entries, list):
                    raise ValueError(
                        f"benchmark artifact matchup {matchup_index} game {game_index} has invalid Root-PUCT "
                        f"telemetry entries for {policy_id} {player}; expected a non-empty JSON list."
                    )
                if not entries:
                    raise ValueError(
                        f"benchmark artifact matchup {matchup_index} game {game_index} is missing Root-PUCT "
                        f"telemetry for {policy_id} {player}."
                    )
                if requested_ids and not requested:
                    continue
                validated = tuple(
                    _validated_decision_telemetry(item, policy_id=policy_id, player_id=player, entry_index=index)
                    for index, item in enumerate(entries)
                )
                if validated:
                    by_policy.setdefault(policy_id, []).extend(validated)
    if requested_ids:
        missing = sorted(requested_ids - set(by_policy))
        if missing:
            raise ValueError(
                "benchmark artifact has no Root-PUCT decision telemetry for policy ids: "
                + ", ".join(missing)
            )
    if not by_policy:
        raise ValueError(
            "benchmark artifact has no root_puct_decision_telemetry_by_player entries; rerun with telemetry-enabled code."
        )
    return {
        "schema_version": ROOT_PUCT_TELEMETRY_REPORT_SCHEMA_VERSION,
        "policies": {
            policy_id: summarize_root_puct_decision_telemetry(entries)
            for policy_id, entries in sorted(by_policy.items())
        },
    }


def _compact_timing(value: object) -> dict[str, float | int]:
    if not isinstance(value, Mapping):
        return {}
    result: dict[str, float | int] = {}
    for key in _TIMING_KEYS:
        raw = value.get(key)
        if key.endswith("_count"):
            parsed = _nonnegative_int(raw)
        else:
            parsed = _finite_float(raw)
        if parsed is not None:
            result[key] = parsed
    return result


def _validated_decision_telemetry(
    value: object,
    *,
    policy_id: str,
    player_id: str,
    entry_index: int,
) -> Mapping[str, object]:
    """Reject incompatible records instead of reporting plausible zeroes."""

    context = f"{policy_id} {player_id} telemetry entry {entry_index}"
    if not isinstance(value, Mapping):
        raise ValueError(f"benchmark artifact has invalid {context}; expected a JSON object.")
    if value.get("schema_version") != ROOT_PUCT_DECISION_TELEMETRY_SCHEMA_VERSION:
        raise ValueError(
            f"benchmark artifact has incompatible {context} schema; expected "
            f"{ROOT_PUCT_DECISION_TELEMETRY_SCHEMA_VERSION}."
        )
    if (
        _nonnegative_int(value.get("decision_index")) is None
        or _nonnegative_int(value.get("turn_index")) is None
    ):
        raise ValueError(f"benchmark artifact has invalid {context} decision or turn index.")
    fallback = value.get("fallback")
    outcome = value.get("outcome")
    if not isinstance(fallback, bool) or outcome not in {"searched", "fallback"}:
        raise ValueError(f"benchmark artifact has invalid {context} outcome.")
    if fallback != (outcome == "fallback"):
        raise ValueError(f"benchmark artifact has inconsistent {context} fallback flag.")
    if fallback and not isinstance(value.get("fallback_category"), str):
        raise ValueError(f"benchmark artifact has invalid {context} fallback category.")
    if _finite_nonnegative_float(value.get("full_decision_elapsed_seconds")) is None:
        raise ValueError(f"benchmark artifact has incomplete {context}; missing full decision wall timing.")
    if outcome == "searched":
        if _nonnegative_int(value.get("root_puct_total_visits")) is None:
            raise ValueError(f"benchmark artifact has incomplete {context}; missing root visit count.")
        timing = value.get("timing")
        if not isinstance(timing, Mapping) or _finite_nonnegative_float(timing.get("total_seconds")) is None:
            raise ValueError(f"benchmark artifact has incomplete {context}; missing full decision timing.")
    return value


def _looks_like_root_puct_policy(policy_id: str) -> bool:
    return "root-puct" in policy_id


def _counter_map(value: object) -> dict[str, int]:
    if not isinstance(value, Mapping):
        return {}
    result: dict[str, int] = {}
    for key, count in value.items():
        parsed = _nonnegative_int(count)
        if parsed is not None:
            result[str(key)] = parsed
    return dict(sorted(result.items()))


def _merge_counter_map(target: dict[str, int], value: object) -> None:
    for key, count in _counter_map(value).items():
        target[key] = target.get(key, 0) + count


def _sample_summary(values: Sequence[float]) -> dict[str, float | int | None]:
    if not values:
        return {"samples": 0, "mean": None, "p50": None, "p95": None}
    ordered = sorted(values)
    return {
        "samples": len(ordered),
        "mean": sum(ordered) / len(ordered),
        "p50": _nearest_rank(ordered, 0.50),
        "p95": _nearest_rank(ordered, 0.95),
    }


def _nearest_rank(values: Sequence[float], percentile: float) -> float:
    index = max(0, math.ceil(percentile * len(values)) - 1)
    return values[index]


def _nonnegative_int(value: object) -> int | None:
    if isinstance(value, bool):
        return None
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed >= 0 else None


def _finite_float(value: object) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def _finite_nonnegative_float(value: object) -> float | None:
    parsed = _finite_float(value)
    return parsed if parsed is not None and parsed >= 0.0 else None
