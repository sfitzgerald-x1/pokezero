"""Diagnostics helpers for replay-from-root MCTS."""

from __future__ import annotations

import re

_DECISION_ROUND_RE = re.compile(r"\bdecision round\s+(\d+)\b", re.IGNORECASE)
_REQUEST_MISMATCH_ROUND_RE = re.compile(
    r"\breplay actions for decision round\s+(\d+)\s+do not match environment request\b",
    re.IGNORECASE,
)
_REQUEST_MISMATCH_DETAIL_RE = re.compile(
    r"\breplay actions for decision round\s+\d+\s+do not match environment request\s*"
    r"\((?P<details>[^)]*)\)",
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
_MISSING_WORLD_DETAIL_RE = re.compile(
    r"start override planner did not produce a sampled world:\s*(?P<detail>.*?)(?=;|\Z)",
    re.IGNORECASE,
)
_MISSING_WORLD_SOURCE_NONE_RE = re.compile(
    r"start override source did not produce a sampled world(?:\s+\([2-9]\d* attempts\))?",
    re.IGNORECASE,
)
_MISSING_WORLD_PLANNER_NONE_RE = re.compile(
    r"start override planner did not produce a sampled world(?!\s*:)(?:\s+\([2-9]\d* attempts\))?",
    re.IGNORECASE,
)
_REJECTION_ATTEMPT_SUFFIX_RE = re.compile(r"\s+\((?P<count>[2-9]\d*) attempts\)\s*$", re.IGNORECASE)


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
        has_duplicate_override = "sampled start override duplicated" in text
        has_force_switch_illegal_action = (
            "action_index " in text
            and "is not legal for the current request (request_kind=force_switch)." in text
        )
        if (has_unexpected_players or has_missing_players) and (
            has_observation_mismatch or has_missing_world
        ):
            return "mixed_replay_prefix_divergence"
        if has_duplicate_override and (
            has_observation_mismatch or has_missing_world or has_unexpected_players or has_missing_players
        ):
            return "mixed_replay_prefix_divergence"
        if has_observation_mismatch and has_missing_world:
            return "mixed_replay_prefix_divergence"
        if has_duplicate_override:
            return "duplicate_start_override"
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
        if has_force_switch_illegal_action:
            return "force_switch_illegal_action"
        return "all_opponent_scenarios_replay_illegal"
    if (
        "action_index " in text
        and "is not legal for the current request (request_kind=force_switch)." in text
    ):
        return "force_switch_illegal_action"
    if "action_index " in text and " is not legal for the current request" in text:
        return "illegal_action_for_current_request"
    if "sampled start override duplicated" in text:
        return "duplicate_start_override"
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


def root_puct_replay_request_mismatch_player_counts(reason: object) -> dict[str, int]:
    """Count missing/unexpected players in replay request-shape mismatch diagnostics."""

    counts: dict[str, int] = {}
    for details in _request_mismatch_detail_strings(reason):
        for chunk in details.split(";"):
            label, players_text = _request_mismatch_detail_parts(chunk)
            if not label:
                continue
            for player in re.findall(r"\bp[12]\b", players_text, flags=re.IGNORECASE):
                key = f"{label}:{player.lower()}"
                counts[key] = counts.get(key, 0) + 1
    return counts


def root_puct_replay_request_mismatch_shape_counts(reason: object) -> dict[str, int]:
    """Count full replay request shapes, keyed by requested players and recorded action players."""

    counts: dict[str, int] = {}
    for details in _request_mismatch_detail_strings(reason):
        requested_players: str | None = None
        action_players: str | None = None
        for chunk in details.split(";"):
            normalized = chunk.strip().lower()
            if normalized.startswith("requested players:"):
                requested_players = _request_mismatch_players_key(chunk.split(":", 1)[1])
            elif normalized.startswith("action players:"):
                action_players = _request_mismatch_players_key(chunk.split(":", 1)[1])
        if requested_players is None or action_players is None:
            continue
        key = f"requested:{requested_players}|actions:{action_players}"
        counts[key] = counts.get(key, 0) + 1
    return counts


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


def root_puct_missing_sampled_world_reason_counts(reason: object) -> dict[str, int]:
    """Classify public belief-world materialization failures without retaining raw details."""

    counts: dict[str, int] = {}
    text = str(reason or "")
    for match in _MISSING_WORLD_DETAIL_RE.finditer(text):
        category = _missing_sampled_world_reason_category(match.group("detail"))
        counts[category] = counts.get(category, 0) + _rejection_attempt_count(match.group(0))
    for match in _MISSING_WORLD_SOURCE_NONE_RE.finditer(text):
        counts["source_none"] = counts.get("source_none", 0) + _rejection_attempt_count(match.group(0))
    for match in _MISSING_WORLD_PLANNER_NONE_RE.finditer(text):
        counts["planner_none"] = counts.get("planner_none", 0) + _rejection_attempt_count(match.group(0))
    return counts


def _decision_round_counts(pattern: re.Pattern[str], reason: object) -> dict[str, int]:
    counts: dict[str, int] = {}
    for match in pattern.finditer(str(reason or "")):
        key = match.group(1)
        counts[key] = counts.get(key, 0) + 1
    return counts


def _rejection_attempt_count(reason: str) -> int:
    match = _REJECTION_ATTEMPT_SUFFIX_RE.search(reason)
    return int(match.group("count")) if match is not None else 1


def _request_mismatch_detail_strings(reason: object) -> list[str]:
    return [
        match.group("details")
        for match in _REQUEST_MISMATCH_DETAIL_RE.finditer(str(reason or ""))
    ]


def _request_mismatch_detail_parts(chunk: str) -> tuple[str | None, str]:
    normalized = chunk.strip().lower()
    if normalized.startswith("missing requested players:"):
        return "missing", chunk.split(":", 1)[1]
    if normalized.startswith("unexpected players:"):
        return "unexpected", chunk.split(":", 1)[1]
    return None, ""


def _request_mismatch_players_key(players_text: str) -> str:
    players = sorted(
        set(re.findall(r"\bp[12]\b", players_text, flags=re.IGNORECASE)),
    )
    if not players:
        return "none"
    return ",".join(player.lower() for player in players)


def _missing_sampled_world_reason_category(detail: str) -> str:
    normalized = detail.strip().lower()
    if "unsupported format" in normalized:
        return "unsupported_format"
    if "observation metadata is missing" in normalized:
        return "observation_metadata_missing"
    if "belief_view is missing or invalid" in normalized:
        return "belief_view_invalid"
    if "request-known self_team has an unexpected member count" in normalized:
        return "self_team_member_count_invalid"
    if "request-known self_team contains an invalid member" in normalized:
        return "self_team_member_invalid"
    if "request-known self_team member is missing species or moves" in normalized:
        return "self_team_member_identity_incomplete"
    if "request-known self_team fixture stats cannot be reconstructed" in normalized:
        return "self_team_fixture_stats_unavailable"
    if "request-known self_team is missing or inconsistent" in normalized:
        return "self_team_unavailable"
    if "public opponent switch constraints are inconsistent" in normalized:
        return "opponent_switch_constraints_inconsistent"
    if "opponent belief has more revealed pokemon than team slots" in normalized:
        return "revealed_team_overfull"
    if "public constrained hidden opponent species could not be sampled" in normalized:
        return "constrained_hidden_backline_unavailable"
    if "random hidden opponent backline could not be sampled" in normalized:
        return "hidden_backline_unavailable"
    if "sampled opponent team could not satisfy public team slot constraints" in normalized:
        return "team_slot_constraints_unsatisfied"
    if "belief_view player slots are invalid" in normalized:
        return "belief_view_slots_invalid"
    if "could not be sampled from public belief" in normalized:
        return "revealed_opponent_unavailable"
    if "opponent belief could not be materialized" in normalized:
        return "opponent_belief_unavailable"
    return "other"
