"""Source-bound action-ranking audits at actual MCTS roots.

Unlike the override audit, this controller evaluates up to three actions at an
actual searched root under more than one continuation target.  It is not a
policy input and never serializes a source snapshot or the committed opponent
action.  Its job is deliberately narrower: produce an independently replayed
answer to whether the MCTS choice, raw-policy choice, and leading visited
alternative have different rankings under declared continuation targets.
"""

from __future__ import annotations

import json
from typing import Any, Callable, Mapping, Sequence

from ..actions import ACTION_COUNT
from ..env import PlayerId, PokeZeroEnv
from ..policy import Policy
from ..rollout import RolloutConfig, RolloutSealedPreStepBoundary
from ..sealed_override_continuation import (
    SealedOverrideContinuationError,
    evaluate_sealed_root_action_grid,
)
from .sealed_override_audit import SealedOverrideAuditError, _safe_selection_evidence


class SealedRootActionAuditError(RuntimeError):
    """A sealed MCTS root cannot support an action-ranking audit."""


def _json_object(value: object, *, label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise SealedRootActionAuditError(f"{label} must be a mapping")
    try:
        copied = json.loads(json.dumps(dict(value), sort_keys=True, separators=(",", ":")))
    except (TypeError, ValueError) as exc:
        raise SealedRootActionAuditError(f"{label} is not JSON-safe") from exc
    if not isinstance(copied, dict):
        raise SealedRootActionAuditError(f"{label} did not serialize to a JSON object")
    return copied


def _action(value: object, *, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value < ACTION_COUNT:
        raise SealedRootActionAuditError(f"{label} must be an action index in [0, {ACTION_COUNT})")
    return value


def root_action_candidates(override: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, int]]:
    """Return safe MCTS evidence and the predeclared candidate action set.

    The raw-policy and selected MCTS actions are always retained when distinct.
    The optional third action is the highest-visit root arm not already named.
    It is chosen before any continuation is run, so it cannot be selected on
    outcome information.  Unmeasured roots and non-overrides are intentionally
    excluded: neither supports the central MCTS-minus-raw comparison.
    """

    try:
        evidence = _safe_selection_evidence(override)
    except SealedOverrideAuditError as exc:
        raise SealedRootActionAuditError(f"invalid MCTS root evidence: {exc}") from exc
    if evidence["model_override"] is not True:
        raise SealedRootActionAuditError("root action audit requires a measured MCTS override")
    raw_action = evidence["model_argmax"]
    mcts_action = evidence["search_argmax"]
    if raw_action == mcts_action:
        raise SealedRootActionAuditError("measured MCTS override has identical actions")
    actions = {"raw_policy": raw_action, "mcts_selected": mcts_action}
    arms = evidence["root_allocation"]["arms"]
    leading_other = next(
        (
            arm["action_index"]
            for arm in sorted(
                arms,
                key=lambda arm: (-arm["visit_share"], arm["action_index"]),
            )
            if arm["visit_share"] > 0.0 and arm["action_index"] not in {raw_action, mcts_action}
        ),
        None,
    )
    if leading_other is not None:
        actions["visit_alternative"] = leading_other
    return evidence, actions


def _source_observation_histories(
    boundary: RolloutSealedPreStepBoundary,
) -> Mapping[PlayerId, tuple[Any, ...]]:
    """Return private source histories without serializing or exposing them."""

    raw = getattr(boundary, "policy_observation_histories", None)
    if not isinstance(raw, Mapping) or set(raw) != {"p1", "p2"}:
        raise SealedRootActionAuditError(
            "root action audit requires both source policy observation histories"
        )
    histories: dict[PlayerId, tuple[Any, ...]] = {}
    for player_id in ("p1", "p2"):
        history = raw.get(player_id)
        if not isinstance(history, tuple) or not history:
            raise SealedRootActionAuditError(
                "root action audit source policy history is missing or empty"
            )
        histories[player_id] = history
    return histories


def evaluate_root_action_boundary(
    *,
    boundary: RolloutSealedPreStepBoundary,
    candidate_seat: PlayerId,
    env_factory: Callable[[], PokeZeroEnv],
    continuation_policy_factory_builder: Callable[
        [Mapping[PlayerId, tuple[Any, ...]]],
        Mapping[str, Callable[[], Mapping[PlayerId, Policy]]],
    ],
    continuation_rng_seeds: Sequence[int],
    rollout_config: RolloutConfig,
    max_continuation_decision_rounds: int | None = None,
) -> dict[str, Any] | None:
    """Evaluate an actual MCTS override under paired continuation targets.

    ``None`` means this source boundary is outside the declared override
    stratum.  A one-sided phase is reported as inapplicable rather than being
    silently dropped, because holding the opponent's joint action fixed is an
    explicit requirement of the causal comparison.
    """

    if candidate_seat not in {"p1", "p2"}:
        raise SealedRootActionAuditError("candidate seat must be p1 or p2")
    opponent_seat: PlayerId = "p2" if candidate_seat == "p1" else "p1"
    if candidate_seat not in boundary.decisions:
        if tuple(boundary.requested_players) == (opponent_seat,) and set(boundary.decisions) == {
            opponent_seat
        }:
            return None
        raise SealedRootActionAuditError("root action boundary omits the candidate decision")
    candidate = boundary.decisions[candidate_seat]
    metadata = _json_object(candidate.metadata, label="candidate decision metadata")
    engine_mcts = _json_object(metadata.get("engine_mcts"), label="engine MCTS metadata")
    override = _json_object(engine_mcts.get("override"), label="engine MCTS override telemetry")
    measured = override.get("model_override")
    if measured is False:
        return None
    if measured is not True:
        if override.get("unmeasured_cause") is not None:
            return None
        raise SealedRootActionAuditError(
            "candidate decision has neither a measured override nor an unmeasured cause"
        )
    evidence, actions = root_action_candidates(override)
    if candidate.action_index != actions["mcts_selected"]:
        raise SealedRootActionAuditError("committed MCTS action disagrees with root evidence")
    if tuple(boundary.requested_players) != ("p1", "p2"):
        if tuple(boundary.requested_players) != (candidate_seat,) or set(boundary.decisions) != {
            candidate_seat
        }:
            raise SealedRootActionAuditError("one-sided root action boundary is malformed")
        return {
            "schema_version": "pokezero.sealed-root-action-audit.v1",
            "seed": boundary.seed,
            "battle_id": boundary.battle_id,
            "candidate_seat": candidate_seat,
            "decision_round_index": boundary.decision_round_index,
            "audit_status": "INAPPLICABLE_NON_SIMULTANEOUS",
            "requested_players": [candidate_seat],
            "search_evidence": evidence,
            "actions": actions,
        }
    if set(boundary.decisions) != {"p1", "p2"}:
        raise SealedRootActionAuditError("simultaneous root action boundary must contain p1 and p2")
    opponent_action = _action(
        boundary.decisions[opponent_seat].action_index, label="committed opponent action"
    )
    try:
        histories = _source_observation_histories(boundary)
        grid = evaluate_sealed_root_action_grid(
            snapshot=boundary.snapshot,
            source_battle_id=boundary.battle_id,
            source_seed=boundary.seed,
            source_decision_round=boundary.decision_round_index,
            subject_player=candidate_seat,
            actions=actions,
            opponent_player=opponent_seat,
            opponent_action=opponent_action,
            continuation_policy_factories=continuation_policy_factory_builder(histories),
            continuation_rng_seeds=continuation_rng_seeds,
            search_evidence=evidence,
            env_factory=env_factory,
            rollout_config=rollout_config,
            max_continuation_decision_rounds=max_continuation_decision_rounds,
        )
    except SealedOverrideContinuationError as exc:
        raise SealedRootActionAuditError(f"root action continuation failed: {exc}") from exc
    return {
        "schema_version": "pokezero.sealed-root-action-audit.v1",
        "seed": boundary.seed,
        "battle_id": boundary.battle_id,
        "candidate_seat": candidate_seat,
        "decision_round_index": boundary.decision_round_index,
        "audit_status": "PAIRED",
        "audit": grid,
    }
