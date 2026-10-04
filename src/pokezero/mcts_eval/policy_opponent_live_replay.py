"""Read-only live public-root qualification; no policy or native search runs.

The caller supplies a checkpoint-configured live environment. This qualifies
replay inputs only, not a published image, native worlds, decisions, or strength.
Source failures remain in the frozen denominator, never replaced.
"""

from __future__ import annotations

from typing import Any, Mapping

from .policy_opponent_roster import RosterError, load_frozen_roster
from .policy_opponent_source_prefixes import load_source_records, qualify_source_prefixes
from .source_root_replay import source_bound_replay_prefix
from ..public_decision_corpus import PublicObservation
from ..public_replay_materializer import PublicReplayError, replay_public_action_rounds
from ..showdown import showdown_choice_for_action


def validate_live_root(env: Any, record: Any, replay: Any) -> dict[str, Any]:
    """Validate every persisted public input and serialize all legal choices."""
    if replay.terminal is not None or record.acting_player not in replay.requested_players:
        raise PublicReplayError("live_root_request_boundary_differs")
    observation = env.observe(record.acting_player)
    if PublicObservation.from_observation(observation) != record.observation:
        raise PublicReplayError("live_public_observation_differs_from_source")
    belief = observation.metadata.get("belief_view")
    if not isinstance(belief, Mapping) or dict(belief) != dict(record.public_belief_view):
        raise PublicReplayError("live_public_belief_differs_from_source")
    mask = tuple(bool(value) for value in observation.legal_action_mask)
    if mask != record.current_legal_action_mask or not any(mask):
        raise PublicReplayError("live_legal_action_mask_differs_from_source")
    for entry in record.history:
        prior = replay.replay_observations.get(entry.turn_index, {}).get(record.acting_player)
        if prior is None or PublicObservation.from_observation(prior) != entry.observation:
            raise PublicReplayError("live_actor_history_differs_from_source")
    choices = [{"action_index": index,
                "showdown_choice": showdown_choice_for_action(env._state_for_player(record.acting_player), index)}
               for index, legal in enumerate(mask) if legal]
    if any(not isinstance(row["showdown_choice"], str) or not row["showdown_choice"] for row in choices):
        raise PublicReplayError("live_legal_action_cannot_be_serialized")
    return {"requested_players": list(replay.requested_players), "legal_choices": choices,
            "actor_history_observations_verified": len(record.history),
            "simultaneous_continuation_eligible": set(replay.requested_players) == {"p1", "p2"},
            "public_event_canonicalizations": [row.to_dict() for row in replay.event_canonicalizations]}


def qualify_live_replay(roster_path: Any, *, expected_roster_sha256: str,
                        source_root: Any, env: Any) -> dict[str, Any]:
    preflight = qualify_source_prefixes(roster_path, expected_roster_sha256=expected_roster_sha256,
                                        source_root=source_root)
    roster = load_frozen_roster(roster_path, expected_sha256=expected_roster_sha256)
    records = load_source_records(source_root, roster)
    results = []
    for source in preflight["roots"]:
        result = dict(source)
        if source["state"] == "REFUSED":
            result["phase"] = "source_prefix"
        else:
            record = records[source["source_relative_path"]]
            prefix = source_bound_replay_prefix(record, source_records=tuple(records.values()))
            try:
                replay = replay_public_action_rounds(env, seed=record.seed, format_id=record.format_id,
                    public_action_rounds=prefix.public_action_rounds, start_override=None)
                result.update(validate_live_root(env, record, replay), state="LIVE_PUBLIC_REPLAY_VALID")
            except (PublicReplayError, ValueError, RuntimeError) as error:
                result.update(state="REFUSED", phase="live_replay", refusal=str(error),
                              refusal_type=type(error).__name__)
        results.append(result)
    after = qualify_source_prefixes(roster_path, expected_roster_sha256=expected_roster_sha256,
                                    source_root=source_root)
    if after != preflight:
        raise RosterError("source-prefix receipt drift during live replay")
    valid = sum(row["state"] == "LIVE_PUBLIC_REPLAY_VALID" for row in results)
    return {**preflight, "schema_version": "pokezero.paper-policy-opponent-live-replay.v1",
            "status": "LIVE_REPLAY_PREFLIGHT_COMPLETE", "roots": results,
            "live_replay_valid": valid, "refused": len(results) - valid,
            "live_replay_qualified": valid == len(results),
            "native_worlds_qualified": False, "profile_qualified": False,
            "search_invoked": False, "replacement_roots": []}
