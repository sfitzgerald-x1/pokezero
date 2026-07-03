"""Diagnostics helpers for replay-from-root MCTS."""

from __future__ import annotations


def root_puct_fallback_category(reason: object) -> str:
    """Return a stable, compact category for a verbose root-PUCT fallback reason."""

    text = str(reason or "").lower()
    if not text:
        return "unknown"
    if "missing policy context" in text:
        return "missing_policy_context"
    if "player is not requested" in text:
        return "player_not_requested"
    if "did not produce a sampled world" in text or "sampled world" in text:
        return "missing_sampled_world"
    if "no branch candidates" in text:
        return "no_branch_candidates"
    if "all opponent action scenarios were replay-illegal" in text:
        if "unexpected players" in text and "start override does not reproduce" in text:
            return "mixed_replay_prefix_divergence"
        if "missing requested players" in text and "start override does not reproduce" in text:
            return "mixed_replay_prefix_divergence"
        if "unexpected players" in text:
            return "replay_request_unexpected_player"
        if "missing requested players" in text:
            return "replay_request_missing_player"
        if "start override does not reproduce" in text:
            return "start_override_observation_mismatch"
        return "all_opponent_scenarios_replay_illegal"
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
