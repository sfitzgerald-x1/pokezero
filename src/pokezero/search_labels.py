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


def search_targets_from_reports(world_runs, legal_mask, action_index_for_choice):
    """Read exact visits and unrounded value totals before telemetry rounding.

    ``world_runs`` contains one record per belief draw. Collapsed draws have
    already been expanded by the caller and must not be weighted a second time.
    Native values are side-one win probabilities; ``side_key`` names the
    searching side after materialization and is the authority for conversion.
    No unknown positive mass is dropped or replaced with a uniform target.
    """
    if not world_runs:
        raise ValueError("search targets require a completed world")
    probabilities = [0.0] * ACTION_COUNT
    values = []
    for record in world_runs:
        side = record["side_key"]
        if side not in ("side_one", "side_two"):
            raise ValueError("search target has an unknown native side")
        report = record["report"]
        stats = report["training_root_stats"]
        visits = stats["visits"]
        total = stats["side_one_value_total"]
        if type(visits) is not int or visits <= 0 or type(total) not in (int, float):
            raise ValueError("search target requires positive visits and a numeric value total")
        total = float(total)
        if not math.isfinite(total) or not 0.0 <= total <= visits:
            raise ValueError("search target value total is outside its visit count")
        counts = [0] * ACTION_COUNT
        observed = 0
        for entry in report[side]:
            count = entry["visits"]
            if type(count) is not int or count < 0:
                raise ValueError("search target visits must be nonnegative integers")
            observed += count
            if count == 0:
                continue
            index = action_index_for_choice(entry["move"])
            if (type(index) is not int or not 0 <= index < ACTION_COUNT
                    or len(legal_mask) != ACTION_COUNT or not legal_mask[index]):
                raise ValueError("search target contains unmapped or illegal positive visit mass")
            counts[index] += count
        if observed != visits:
            raise ValueError("search target visit denominator disagrees with the native root")
        for index, count in enumerate(counts):
            probabilities[index] += count / visits / len(world_runs)
        side_one_value = total / visits
        actor_value = side_one_value if side == "side_one" else 1.0 - side_one_value
        values.append(2.0 * actor_value - 1.0)
    distribution, value = validate_search_targets(
        probabilities, sum(values) / len(values), legal_mask,
    )
    return {"status": "LABELED", "policy_target": list(distribution),
            "value_target": value}


class SearchLabelPolicy:
    """Run the raw actor once, then query a separate teacher for selected labels.

    The caller attaches this wrapper to the learner's assigned seat only. Teacher
    construction errors remain fatal configuration errors. Ordinary search errors
    produce explicit unlabeled records; capture failures inherit BaseException and
    remain fatal. The real teacher owns mandatory reproduction capture.
    """
    requires_public_materialization_state = True

    def __init__(self, actor: Any, *, teacher_factory: Callable[[], Any],
                 fraction: float, teacher_identity: Mapping[str, Any]):
        if not math.isfinite(fraction) or not 0.0 < fraction <= 1.0:
            raise ValueError("an active search label wrapper requires fraction in (0, 1]")
        self._actor = actor
        self._teacher_factory = teacher_factory
        self._teacher = None
        self._fraction = fraction
        self._teacher_identity = dict(teacher_identity)

    @property
    def policy_id(self):
        return self._actor.policy_id

    def __getattr__(self, name):
        return getattr(object.__getattribute__(self, "_actor"), name)

    def reset(self):
        reset = getattr(self._actor, "reset", None)
        if callable(reset):
            reset()
        # Do not keep a previous battle's fold or native handle alive.
        self._teacher = None

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
        if self._teacher is None:
            self._teacher = self._teacher_factory()
        teacher = self._teacher
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
