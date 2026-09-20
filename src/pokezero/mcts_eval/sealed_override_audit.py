"""Adapt sealed rollout boundaries into matched MCTS override continuations.

This module is deliberately a controller-side adapter.  It reads only the
acting MCTS decision's measured root metadata and the already-selected joint
source actions; it does not expose an actionable snapshot to policy code or
public artifacts.
"""

from __future__ import annotations

import json
from typing import Any, Callable, Mapping

from ..env import PlayerId, PokeZeroEnv
from ..policy import Policy
from ..rollout import RolloutConfig, RolloutSealedPreStepBoundary
from ..sealed_override_continuation import (
    SealedOverrideContinuationError,
    evaluate_sealed_override_pair,
)


class SealedOverrideAuditError(RuntimeError):
    """A source MCTS decision cannot be turned into a sound audit sample."""


def _json_object(value: object, *, label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise SealedOverrideAuditError(f"{label} must be a mapping")
    try:
        copied = json.loads(json.dumps(dict(value), sort_keys=True, separators=(",", ":")))
    except (TypeError, ValueError) as exc:
        raise SealedOverrideAuditError(f"{label} is not JSON-safe") from exc
    if not isinstance(copied, dict):
        raise SealedOverrideAuditError(f"{label} did not serialize to a JSON object")
    return copied


def _action(value: object, *, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise SealedOverrideAuditError(f"{label} must be a non-negative action index")
    return value


def evaluate_measured_override_boundary(
    *,
    boundary: RolloutSealedPreStepBoundary,
    candidate_seat: PlayerId,
    env_factory: Callable[[], PokeZeroEnv],
    continuation_policy_factory: Callable[[], Mapping[PlayerId, Policy]],
    rollout_config: RolloutConfig,
    max_continuation_decision_rounds: int | None = None,
) -> dict[str, Any] | None:
    """Return a paired continuation readout, or ``None`` for a non-override.

    Measured overrides are the only valid denominator: an unmeasured root
    lacks a model action index, and using a guessed action would convert a
    fallback diagnosis into fabricated performance evidence.  The returned
    object contains no snapshot and is suitable for a sealed durable artifact.
    """

    if candidate_seat not in {"p1", "p2"}:
        raise SealedOverrideAuditError("candidate seat must be p1 or p2")
    if tuple(boundary.requested_players) != ("p1", "p2"):
        raise SealedOverrideAuditError("override audit requires a simultaneous p1/p2 boundary")
    if set(boundary.decisions) != {"p1", "p2"}:
        raise SealedOverrideAuditError("override audit boundary must contain exactly p1/p2 decisions")
    candidate = boundary.decisions[candidate_seat]
    metadata = _json_object(candidate.metadata, label="candidate decision metadata")
    measured = metadata.get("model_override")
    if measured is False:
        return None
    if measured is not True:
        if metadata.get("unmeasured_cause") is not None:
            return None
        raise SealedOverrideAuditError(
            "candidate decision has neither a measured override nor an unmeasured cause"
        )
    model_action = _action(metadata.get("model_argmax"), label="model argmax")
    search_action = _action(metadata.get("search_argmax"), label="search argmax")
    if search_action != candidate.action_index:
        raise SealedOverrideAuditError("search argmax does not equal the committed MCTS action")
    if model_action == search_action:
        raise SealedOverrideAuditError("measured override has identical model and search actions")
    opponent_seat: PlayerId = "p2" if candidate_seat == "p1" else "p1"
    opponent_action = _action(
        boundary.decisions[opponent_seat].action_index, label="committed opponent action"
    )
    root_allocation = _json_object(metadata.get("root_allocation"), label="root allocation")
    evidence = {
        "model_argmax": model_action,
        "search_argmax": search_action,
        "root_q_gap": metadata.get("root_q_gap"),
        "root_visit_gap": metadata.get("root_visit_gap"),
        "root_allocation": root_allocation,
        "branch_prior_fallbacks": metadata.get("branch_prior_fallbacks"),
    }
    try:
        readout = evaluate_sealed_override_pair(
            snapshot=boundary.snapshot,
            source_seed=boundary.seed,
            source_decision_round=boundary.decision_round_index,
            subject_player=candidate_seat,
            mcts_action=search_action,
            raw_action=model_action,
            opponent_player=opponent_seat,
            opponent_action=opponent_action,
            search_evidence=evidence,
            env_factory=env_factory,
            continuation_policy_factory=continuation_policy_factory,
            rollout_config=rollout_config,
            max_continuation_decision_rounds=max_continuation_decision_rounds,
        )
    except SealedOverrideContinuationError as exc:
        raise SealedOverrideAuditError(f"sealed continuation failed: {exc}") from exc
    return {
        "schema_version": "pokezero.mcts-sealed-override-audit.v1",
        "seed": boundary.seed,
        "battle_id": boundary.battle_id,
        "candidate_seat": candidate_seat,
        "decision_round_index": boundary.decision_round_index,
        "audit": readout,
    }
