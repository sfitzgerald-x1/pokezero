"""Optional search labels for an unchanged, on-policy raw actor.

Selection and teacher sampling use separate hash domains. Neither consumes the
actor's RNG. A search fallback is an absent label, never a uniform teacher target.
"""
from __future__ import annotations

from dataclasses import replace
import hashlib
import json
import math
import random
from typing import Any, Callable, Mapping

from .actions import ACTION_COUNT
from .policy import PolicyContext, PolicyDecision


def _seed(domain: str, context: PolicyContext) -> int:
    if (type(context.seed) is not int or type(context.decision_round_index) is not int
            or context.decision_round_index < 0 or context.player_id not in ("p1", "p2")):
        raise ValueError("search labels require a seeded, seat-addressed decision context")
    address = (domain, context.seed, context.player_id, context.decision_round_index)
    encoded = json.dumps(address, separators=(",", ":")).encode("utf-8")
    return int.from_bytes(hashlib.sha256(encoded).digest(), "big")


def select_search_label(context: PolicyContext, fraction: float) -> bool:
    if not math.isfinite(fraction) or not 0.0 <= fraction <= 1.0:
        raise ValueError("search label fraction must be finite and between zero and one")
    return _seed("pokezero-search-label-selection-v1", context) < int(fraction * (1 << 256))


def validate_search_targets(
    policy_target: Any, value_target: Any, legal_mask: Any,
) -> tuple[tuple[float, ...], float]:
    """Validate nine legal probabilities and an acting-seat value in [-1, 1]."""
    if not isinstance(policy_target, (list, tuple)) or len(policy_target) != ACTION_COUNT:
        raise ValueError("search policy target must contain nine probabilities")
    probabilities = tuple(float(p) for p in policy_target)
    if (any(not math.isfinite(p) or p < 0.0 for p in probabilities)
            or not math.isclose(sum(probabilities), 1.0, rel_tol=0.0, abs_tol=1e-7)):
        raise ValueError("search policy target must be a finite probability distribution")
    if len(legal_mask) != ACTION_COUNT or any(p > 0.0 and not legal for p, legal in zip(probabilities, legal_mask)):
        raise ValueError("search policy target contains illegal action mass")
    value = float(value_target)
    if not math.isfinite(value) or not -1.0 <= value <= 1.0:
        raise ValueError("search value target must be finite in acting-seat [-1, 1]")
    return probabilities, value


class SearchLabelPolicy:
    """Run the raw actor once, then query a separate teacher for selected labels.

    The caller attaches this wrapper to the learner's assigned seat only. Teacher
    construction errors remain fatal configuration errors. Ordinary search errors
    produce explicit unlabeled records; capture failures inherit BaseException and
    remain fatal. The real teacher owns mandatory reproduction capture.
    """
    def __init__(self, actor: Any, *, teacher_factory: Callable[[], Any],
                 fraction: float, teacher_identity: Mapping[str, Any]):
        if not math.isfinite(fraction) or not 0.0 < fraction <= 1.0:
            raise ValueError("an active search label wrapper requires fraction in (0, 1]")
        self._actor = actor
        self._teacher_factory = teacher_factory
        self._fraction = fraction
        self._teacher_identity = dict(teacher_identity)

    @property
    def policy_id(self):
        return self._actor.policy_id

    def __getattr__(self, name):
        return getattr(object.__getattribute__(self, "_actor"), name)

    def select_action(self, observation, *, rng: random.Random) -> PolicyDecision:
        raise ValueError("search labels require a live policy context")

    def select_action_with_context(self, context: PolicyContext, *, rng: random.Random) -> PolicyDecision:
        selector = getattr(self._actor, "select_action_with_context", None)
        decision = (selector(context, rng=rng) if callable(selector)
                    else self._actor.select_action(context.observation, rng=rng))
        if not select_search_label(context, self._fraction):
            return decision
        # Deliberately outside the search exception handler: a missing native
        # capability/export is a configuration failure, not a missing data label.
        teacher = self._teacher_factory()
        label: dict[str, Any] = {"status": "UNLABELED", "teacher": dict(self._teacher_identity)}
        teacher_rng = random.Random(_seed("pokezero-search-label-teacher-v1", context))
        try:
            query = teacher.select_action_with_context(context, rng=teacher_rng)
            engine = query.metadata.get("engine_mcts", {})
            target = engine.get("training_targets")
            if not isinstance(target, Mapping) or target.get("status") != "LABELED":
                label["reason"] = "teacher_target_unavailable"
                if isinstance(engine.get("fallback"), str):
                    label.update(reason="teacher_fallback", refusal_code="fallback." + engine["fallback"])
                if isinstance(target, Mapping) and isinstance(target.get("reason"), str):
                    label["reason"] = target["reason"]
            else:
                probabilities, value = validate_search_targets(
                    target["policy_target"], target["value_target"],
                    context.observation.legal_action_mask,
                )
                label.update(status="LABELED", policy_target=list(probabilities), value_target=value)
        except Exception as error:
            # CaptureFailed is intentionally NOT an Exception. The teacher's
            # capture boundary has already persisted any real search refusal.
            label.update(reason="teacher_exception", error_type=type(error).__name__)
        return replace(decision, metadata={**dict(decision.metadata), "search_label": label})


def with_search_labels(actor: Any, *, fraction: float, teacher_factory: Callable[[], Any],
                       teacher_identity: Mapping[str, Any]):
    """The zero-rate path returns the original actor, without another wrapper."""
    if not math.isfinite(fraction) or not 0.0 <= fraction <= 1.0:
        raise ValueError("search label fraction must be finite and between zero and one")
    if fraction == 0.0:
        return actor
    return SearchLabelPolicy(actor, teacher_factory=teacher_factory, fraction=fraction,
                             teacher_identity=teacher_identity)
