"""Adapt sealed rollout boundaries into matched MCTS override continuations.

This module is deliberately a controller-side adapter.  It reads only the
acting MCTS decision's measured root metadata and the already-selected joint
source actions; it does not expose an actionable snapshot to policy code or
public artifacts.
"""

from __future__ import annotations

import json
import math
from typing import Any, Callable, Mapping

from ..actions import ACTION_COUNT
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
    if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value < ACTION_COUNT:
        raise SealedOverrideAuditError(f"{label} must be an action index in [0, {ACTION_COUNT})")
    return value


def _finite_number(value: object, *, label: str) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise SealedOverrideAuditError(f"{label} must be a finite number or null")
    return float(value)


def _safe_selection_evidence(override: Mapping[str, Any]) -> dict[str, Any]:
    """Project engine telemetry to action-indexed, non-private audit evidence.

    Engine root-arm records carry a rendered move label for debugging.  A
    sealed continuation only needs action indices and numeric allocation
    evidence, so deliberately discard that label before producing any durable
    output.  Requiring the complete shape also prevents an older engine image
    from silently producing an incomparable audit row.
    """

    required = {
        "model_argmax",
        "search_argmax",
        "model_override",
        "unmeasured_cause",
        "root_q_gap",
        "root_visit_gap",
        "root_gap_action_indices",
        "root_allocation",
    }
    if not required.issubset(override):
        raise SealedOverrideAuditError("override telemetry is missing selection evidence")
    root = _json_object(override.get("root_allocation"), label="engine root allocation")
    if set(root) != {"worlds", "prior_authority", "prior_cause", "arms"}:
        raise SealedOverrideAuditError("engine root allocation has unsupported fields")
    worlds = root.get("worlds")
    if isinstance(worlds, bool) or not isinstance(worlds, int) or worlds <= 0:
        raise SealedOverrideAuditError("engine root allocation worlds must be positive")
    if not isinstance(root.get("prior_authority"), bool):
        raise SealedOverrideAuditError("engine root allocation prior authority must be boolean")
    prior_cause = root.get("prior_cause")
    if prior_cause is not None and (not isinstance(prior_cause, str) or not prior_cause):
        raise SealedOverrideAuditError("engine root allocation prior cause is malformed")
    raw_arms = root.get("arms")
    if not isinstance(raw_arms, list) or not raw_arms:
        raise SealedOverrideAuditError("engine root allocation must include own-action arms")
    arms: list[dict[str, Any]] = []
    for raw_arm in raw_arms:
        arm = _json_object(raw_arm, label="engine root allocation arm")
        if set(arm) != {
            "move",
            "action_index",
            "visit_share",
            "q",
            "reported_prior",
            "model_prior",
        }:
            raise SealedOverrideAuditError("engine root allocation arm has unsupported fields")
        action_index = _action(arm.get("action_index"), label="root arm action index")
        visit_share = _finite_number(arm.get("visit_share"), label="root arm visit share")
        if visit_share is None or not 0.0 <= visit_share <= 1.0:
            raise SealedOverrideAuditError("root arm visit share must be within [0, 1]")
        arms.append(
            {
                "action_index": action_index,
                "visit_share": visit_share,
                "q": _finite_number(arm.get("q"), label="root arm Q"),
                "reported_prior": _finite_number(
                    arm.get("reported_prior"), label="root arm reported prior"
                ),
                "model_prior": _finite_number(arm.get("model_prior"), label="root arm model prior"),
            }
        )
    if len({arm["action_index"] for arm in arms}) != len(arms):
        raise SealedOverrideAuditError("engine root allocation repeats an action index")
    if not math.isclose(sum(arm["visit_share"] for arm in arms), 1.0, abs_tol=1e-5):
        raise SealedOverrideAuditError("engine root allocation visit shares do not conserve one")
    return {
        "model_argmax": _action(override.get("model_argmax"), label="model argmax"),
        "search_argmax": _action(override.get("search_argmax"), label="search argmax"),
        "model_override": override.get("model_override"),
        "root_q_gap": _finite_number(override.get("root_q_gap"), label="root Q gap"),
        "root_visit_gap": _finite_number(
            override.get("root_visit_gap"), label="root visit gap"
        ),
        "root_gap_action_indices": _safe_action_list(
            override.get("root_gap_action_indices"), label="root gap action indices"
        ),
        "root_allocation": {
            "worlds": worlds,
            "prior_authority": root["prior_authority"],
            "prior_cause": prior_cause,
            "arms": arms,
        },
    }


def _safe_action_list(value: object, *, label: str) -> list[int]:
    if not isinstance(value, list):
        raise SealedOverrideAuditError(f"{label} must be a list")
    return [_action(item, label=label) for item in value]


def _source_observation_histories(
    boundary: RolloutSealedPreStepBoundary,
) -> Mapping[PlayerId, tuple[Any, ...]]:
    """Read trusted source histories without allowing them into a durable readout."""

    raw = getattr(boundary, "policy_observation_histories", None)
    if not isinstance(raw, Mapping) or set(raw) != {"p1", "p2"}:
        raise SealedOverrideAuditError(
            "sealed override continuation requires both source policy observation histories"
        )
    histories: dict[PlayerId, tuple[Any, ...]] = {}
    for player_id in ("p1", "p2"):
        history = raw.get(player_id)
        if not isinstance(history, tuple) or not history:
            raise SealedOverrideAuditError(
                "sealed override continuation source policy history is missing or empty"
            )
        histories[player_id] = history
    return histories


def evaluate_measured_override_boundary(
    *,
    boundary: RolloutSealedPreStepBoundary,
    candidate_seat: PlayerId,
    env_factory: Callable[[], PokeZeroEnv],
    continuation_policy_factory: Callable[[], Mapping[PlayerId, Policy]] | None = None,
    continuation_policy_factory_builder: Callable[
        [Mapping[PlayerId, tuple[Any, ...]]], Callable[[], Mapping[PlayerId, Policy]]
    ] | None = None,
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
    if (continuation_policy_factory is None) == (continuation_policy_factory_builder is None):
        raise SealedOverrideAuditError(
            "sealed override audit requires exactly one continuation policy factory surface"
        )
    if candidate_seat not in boundary.decisions:
        opponent_seat: PlayerId = "p2" if candidate_seat == "p1" else "p1"
        # The hook is installed for the full rollout.  A normal forced phase
        # can request only the raw opponent; that phase cannot contribute a
        # candidate override and has no candidate public-decision record in
        # the denominator, so it must be ignored rather than aborting the
        # source game.
        if tuple(boundary.requested_players) == (opponent_seat,) and set(boundary.decisions) == {
            opponent_seat
        }:
            return None
        raise SealedOverrideAuditError("override audit boundary omits the candidate decision")
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
        raise SealedOverrideAuditError(
            "candidate decision has neither a measured override nor an unmeasured cause"
        )
    evidence = _safe_selection_evidence(override)
    model_action = evidence["model_argmax"]
    search_action = evidence["search_argmax"]
    if search_action != candidate.action_index:
        raise SealedOverrideAuditError("search argmax does not equal the committed MCTS action")
    if model_action == search_action:
        raise SealedOverrideAuditError("measured override has identical model and search actions")
    # A paired continuation requires a committed source action for both seats.
    # A forced, one-sided phase cannot meet that contract, but it remains a
    # measured override that must be durably accounted for so the game-level
    # validator can distinguish inapplicability from an omitted sidecar.
    if tuple(boundary.requested_players) != ("p1", "p2"):
        if tuple(boundary.requested_players) != (candidate_seat,) or set(boundary.decisions) != {
            candidate_seat
        }:
            raise SealedOverrideAuditError(
                "one-sided override boundary must contain exactly the candidate decision"
            )
        return {
            "schema_version": "pokezero.mcts-sealed-override-audit.v1",
            "seed": boundary.seed,
            "battle_id": boundary.battle_id,
            "candidate_seat": candidate_seat,
            "decision_round_index": boundary.decision_round_index,
            "audit_status": "INAPPLICABLE_NON_SIMULTANEOUS",
            "requested_players": [candidate_seat],
            "search_evidence": evidence,
        }
    if set(boundary.decisions) != {"p1", "p2"}:
        raise SealedOverrideAuditError("simultaneous override boundary must contain p1/p2 decisions")
    opponent_seat: PlayerId = "p2" if candidate_seat == "p1" else "p1"
    opponent_action = _action(
        boundary.decisions[opponent_seat].action_index, label="committed opponent action"
    )
    try:
        policies = (
            continuation_policy_factory_builder(_source_observation_histories(boundary))
            if continuation_policy_factory_builder is not None
            else continuation_policy_factory
        )
        if policies is None:
            raise SealedOverrideAuditError("sealed override continuation policy factory is missing")
        readout = evaluate_sealed_override_pair(
            snapshot=boundary.snapshot,
            source_battle_id=boundary.battle_id,
            source_seed=boundary.seed,
            source_decision_round=boundary.decision_round_index,
            subject_player=candidate_seat,
            mcts_action=search_action,
            raw_action=model_action,
            opponent_player=opponent_seat,
            opponent_action=opponent_action,
            search_evidence=evidence,
            env_factory=env_factory,
            continuation_policy_factory=policies,
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
        "audit_status": "PAIRED",
        "audit": readout,
    }
