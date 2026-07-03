"""Diagnostics helpers for replay-from-root MCTS."""

from __future__ import annotations

import re

_DECISION_ROUND_RE = re.compile(r"\bdecision round\s+(\d+)\b", re.IGNORECASE)
_REQUEST_MISMATCH_ROUND_RE = re.compile(
    r"\breplay actions for decision round\s+(\d+)\s+do not match environment request\b",
    re.IGNORECASE,
)
_START_OVERRIDE_OBSERVATION_MISMATCH_ROUND_RE = re.compile(
    r"\bstart override does not reproduce recorded replay prefix observations\s+"
    r"for decision round\s+(\d+)\b",
    re.IGNORECASE,
)
_OBSERVATION_MISMATCH_PATH_RE = re.compile(
    r"\((?P<path>[^():]+):\s+actual=.*?\s+expected=.*?\)",
    re.IGNORECASE,
)


def root_puct_fallback_category(reason: object) -> str:
    """Return a stable, compact category for a verbose root-PUCT fallback reason."""

    text = str(reason or "").lower()
    if not text:
        return "unknown"
    if "missing policy context" in text:
        return "missing_policy_context"
    if "player is not requested" in text:
        return "player_not_requested"
    if "opponent_action_planner returned an illegal action" in text:
        return "opponent_planner_illegal_action"
    if "opponent_action_planner returned the acting player's action" in text:
        return "opponent_planner_self_action"
    if "missing opponent actions for" in text:
        return "opponent_planner_missing_actions"
    if "unexpected opponent actions for" in text:
        return "opponent_planner_unexpected_actions"
    if "no branch candidates" in text:
        return "no_branch_candidates"
    if "all opponent action scenarios were replay-illegal" in text:
        has_unexpected_players = "unexpected players" in text
        has_missing_players = "missing requested players" in text
        has_observation_mismatch = "start override does not reproduce" in text
        has_missing_world = "did not produce a sampled world" in text
        if (has_unexpected_players or has_missing_players) and (
            has_observation_mismatch or has_missing_world
        ):
            return "mixed_replay_prefix_divergence"
        if has_observation_mismatch and has_missing_world:
            return "mixed_replay_prefix_divergence"
        if has_unexpected_players and has_missing_players:
            return "mixed_replay_request_mismatch"
        if has_unexpected_players:
            return "replay_request_unexpected_player"
        if has_missing_players:
            return "replay_request_missing_player"
        if has_observation_mismatch:
            return "start_override_observation_mismatch"
        if has_missing_world:
            return "missing_sampled_world"
        return "all_opponent_scenarios_replay_illegal"
    if "action_index " in text and " is not legal for the current request" in text:
        return "illegal_action_for_current_request"
    if "did not produce a sampled world" in text or "sampled world" in text:
        return "missing_sampled_world"
    if "unexpected players" in text:
        return "replay_request_unexpected_player"
    if "missing requested players" in text:
        return "replay_request_missing_player"
    if "start override does not reproduce" in text:
        return "start_override_observation_mismatch"
    if "observation" in text and "mismatch" in text:
        return "replay_observation_mismatch"
    if "replay" in text:
        return "replay_error"
    if "search failed" in text:
        return "search_failed"
    return "other"


def root_puct_replay_rejection_decision_round_counts(reason: object) -> dict[str, int]:
    """Count decision rounds mentioned by a replay rejection reason.

    A single skipped opponent scenario can include multiple retry failures, so this is a rejection
    occurrence histogram rather than a skipped-scenario histogram.
    """

    return _decision_round_counts(_DECISION_ROUND_RE, reason)


def root_puct_replay_request_mismatch_decision_round_counts(reason: object) -> dict[str, int]:
    """Count rounds where replayed actions did not match the environment request shape."""

    return _decision_round_counts(_REQUEST_MISMATCH_ROUND_RE, reason)


def root_puct_start_override_mismatch_decision_round_counts(reason: object) -> dict[str, int]:
    """Count rounds where a sampled world failed branch-point observation validation."""

    return _decision_round_counts(_START_OVERRIDE_OBSERVATION_MISMATCH_ROUND_RE, reason)


def root_puct_first_observation_mismatch_path_counts(reason: object) -> dict[str, int]:
    """Count first-mismatch observation paths embedded in replay rejection reasons.

    Replay observation diagnostics report the first differing observation field only, so this is a
    first-divergence histogram, not a full inventory of every mismatching feature.
    """

    counts: dict[str, int] = {}
    for match in _OBSERVATION_MISMATCH_PATH_RE.finditer(str(reason or "")):
        key = match.group("path").strip()
        if not key:
            continue
        counts[key] = counts.get(key, 0) + 1
    return counts


def _decision_round_counts(pattern: re.Pattern[str], reason: object) -> dict[str, int]:
    counts: dict[str, int] = {}
    for match in pattern.finditer(str(reason or "")):
        key = match.group(1)
        counts[key] = counts.get(key, 0) + 1
    return counts
