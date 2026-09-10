"""Fail-closed validation for the model-MCTS fixed-deadline qualification.

This is deliberately a small companion to the replay timing adapter, not a new
evaluation framework.  Its input is one durable row per real, replay-backed
decision and its output is the evidence needed before a future MCTS-versus-MCTS
comparison may call itself fixed-wall.  In particular, it refuses to turn a
fallback, a missing native witness, or a zero-work decision into a benign zero.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
import math
from typing import Any


DEADLINE_QUALIFICATION_SCHEMA_VERSION = "pokezero.mcts-deadline-qualification.v1"


class DeadlineQualificationError(ValueError):
    """Evidence does not establish the fixed-deadline contract."""


@dataclass(frozen=True)
class DeadlineQualificationRequirements:
    """The frozen, behavior-bearing allocation for one qualification run."""

    requested_ms: int = 1_000
    sims_per_world: int = 256
    worlds: int = 4
    expected_decisions: int = 16

    def __post_init__(self) -> None:
        if min(
            self.requested_ms,
            self.sims_per_world,
            self.worlds,
            self.expected_decisions,
        ) <= 0:
            raise ValueError("deadline qualification requirements must be positive.")

    def to_payload(self) -> dict[str, int]:
        return asdict(self)


def _mapping(value: Any, field: str, *, decision_id: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise DeadlineQualificationError(f"{decision_id}: {field} must be an object")
    return value


def _sequence(value: Any, field: str, *, decision_id: str) -> Sequence[Any]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise DeadlineQualificationError(f"{decision_id}: {field} must be an array")
    return value


def _int(
    value: Any,
    field: str,
    *,
    decision_id: str,
    minimum: int | None = None,
) -> int:
    if type(value) is not int:
        raise DeadlineQualificationError(f"{decision_id}: {field} must be an integer")
    if minimum is not None and value < minimum:
        raise DeadlineQualificationError(f"{decision_id}: {field} must be >= {minimum}")
    return value


def _finite_number(value: Any, field: str, *, decision_id: str) -> float:
    if type(value) not in (int, float) or not math.isfinite(float(value)):
        raise DeadlineQualificationError(f"{decision_id}: {field} must be a finite number")
    if float(value) < 0.0:
        raise DeadlineQualificationError(f"{decision_id}: {field} must be non-negative")
    return float(value)


def _bool(value: Any, field: str, *, decision_id: str) -> bool:
    if type(value) is not bool:
        raise DeadlineQualificationError(f"{decision_id}: {field} must be a boolean")
    return value


def _require_root_visit_conservation(
    root_visits: Any,
    *,
    completed_iterations: int,
    decision_id: str,
    invocation_index: int,
) -> dict[str, int]:
    visits = _mapping(
        root_visits,
        f"native_invocations[{invocation_index}].root_visits",
        decision_id=decision_id,
    )
    if set(visits) != {"side_one", "side_two"}:
        raise DeadlineQualificationError(
            f"{decision_id}: native_invocations[{invocation_index}].root_visits "
            "must contain exactly side_one and side_two"
        )
    result = {
        side: _int(
            visits[side],
            f"native_invocations[{invocation_index}].root_visits.{side}",
            decision_id=decision_id,
            minimum=0,
        )
        for side in ("side_one", "side_two")
    }
    if any(value != completed_iterations for value in result.values()):
        raise DeadlineQualificationError(
            f"{decision_id}: native_invocations[{invocation_index}] root visits "
            "do not conserve its completed iterations"
        )
    return result


def validate_deadline_decision(
    record: Mapping[str, Any],
    *,
    requirements: DeadlineQualificationRequirements,
) -> dict[str, Any]:
    """Validate and normalize one real deadline-qualified decision record.

    The engine already checks the native report while it is live.  This second
    check is intentionally independent: it validates the durable, decision-level
    evidence rather than trusting a run-level counter or a green process exit.
    """

    decision_id_value = record.get("decision_id")
    if not isinstance(decision_id_value, str) or not decision_id_value:
        raise DeadlineQualificationError("decision_id must be a non-empty string")
    decision_id = decision_id_value
    engine = _mapping(record.get("engine_mcts"), "engine_mcts", decision_id=decision_id)
    if engine.get("leaf_eval") != "model":
        raise DeadlineQualificationError(f"{decision_id}: not a model-MCTS decision")
    if "fallback" in engine:
        raise DeadlineQualificationError(
            f"{decision_id}: fallback {engine['fallback']!r} is not qualification evidence"
        )
    if _int(record.get("invalid_actions"), "invalid_actions", decision_id=decision_id, minimum=0):
        raise DeadlineQualificationError(f"{decision_id}: selected an invalid action")

    constructed = _int(
        engine.get("worlds_constructed"), "worlds_constructed", decision_id=decision_id, minimum=0
    )
    searched = _int(
        engine.get("worlds_searched"), "worlds_searched", decision_id=decision_id, minimum=0
    )
    if constructed != requirements.worlds:
        raise DeadlineQualificationError(
            f"{decision_id}: constructed {constructed} worlds, expected {requirements.worlds}"
        )
    if not 0 < searched <= constructed:
        raise DeadlineQualificationError(
            f"{decision_id}: zero-completed-world decision ({searched}/{constructed} searched)"
        )

    budget = _mapping(engine.get("time_budget"), "engine_mcts.time_budget", decision_id=decision_id)
    if budget.get("scope") != "whole_model_decision":
        raise DeadlineQualificationError(f"{decision_id}: deadline witness has the wrong scope")
    if _int(budget.get("requested_ms"), "time_budget.requested_ms", decision_id=decision_id) != requirements.requested_ms:
        raise DeadlineQualificationError(f"{decision_id}: deadline request differs from frozen contract")
    elapsed_ms = _finite_number(
        budget.get("deadline_elapsed_ms"), "time_budget.deadline_elapsed_ms", decision_id=decision_id
    )
    overshoot_ms = _finite_number(
        budget.get("deadline_overshoot_ms"), "time_budget.deadline_overshoot_ms", decision_id=decision_id
    )
    expected_overshoot = max(0.0, elapsed_ms - requirements.requested_ms)
    if abs(overshoot_ms - expected_overshoot) > 0.001:
        raise DeadlineQualificationError(
            f"{decision_id}: deadline overshoot does not match deadline elapsed time"
        )
    exhausted = _bool(budget.get("exhausted"), "time_budget.exhausted", decision_id=decision_id)
    skipped_worlds = _int(
        budget.get("worlds_budget_skipped"),
        "time_budget.worlds_budget_skipped",
        decision_id=decision_id,
        minimum=0,
    )
    if skipped_worlds > constructed:
        raise DeadlineQualificationError(f"{decision_id}: skipped more worlds than it constructed")

    invocations = _sequence(
        budget.get("native_invocations"), "time_budget.native_invocations", decision_id=decision_id
    )
    if not invocations:
        raise DeadlineQualificationError(f"{decision_id}: has no native deadline invocation")

    normalized_invocations: list[dict[str, Any]] = []
    prefixes = 0
    total_completed = 0
    for index, raw in enumerate(invocations):
        invocation = _mapping(raw, f"native_invocations[{index}]", decision_id=decision_id)
        if invocation.get("status") != "completed":
            refusal = invocation.get("refusal", "missing completed native witness")
            raise DeadlineQualificationError(
                f"{decision_id}: native_invocations[{index}] is not completed ({refusal!r})"
            )
        multiplicity = _int(
            invocation.get("multiplicity"),
            f"native_invocations[{index}].multiplicity",
            decision_id=decision_id,
            minimum=1,
        )
        requested = _int(
            invocation.get("requested_iterations"),
            f"native_invocations[{index}].requested_iterations",
            decision_id=decision_id,
            minimum=1,
        )
        if requested != requirements.sims_per_world * multiplicity:
            raise DeadlineQualificationError(
                f"{decision_id}: native_invocations[{index}] requested {requested}, expected "
                f"{requirements.sims_per_world * multiplicity} for multiplicity {multiplicity}"
            )
        completed = _int(
            invocation.get("completed_iterations"),
            f"native_invocations[{index}].completed_iterations",
            decision_id=decision_id,
            minimum=0,
        )
        remaining = _int(
            invocation.get("remaining_iterations"),
            f"native_invocations[{index}].remaining_iterations",
            decision_id=decision_id,
            minimum=0,
        )
        if completed + remaining != requested:
            raise DeadlineQualificationError(
                f"{decision_id}: native_invocations[{index}] iteration accounting is inconsistent"
            )
        invocation_budget = _int(
            invocation.get("time_budget_ms"),
            f"native_invocations[{index}].time_budget_ms",
            decision_id=decision_id,
            minimum=1,
        )
        if invocation_budget > requirements.requested_ms:
            raise DeadlineQualificationError(
                f"{decision_id}: native_invocations[{index}] exceeded the decision budget"
            )
        native_elapsed = _finite_number(
            invocation.get("time_budget_elapsed_ms"),
            f"native_invocations[{index}].time_budget_elapsed_ms",
            decision_id=decision_id,
        )
        native_overshoot = _finite_number(
            invocation.get("time_budget_batch_overshoot_ms"),
            f"native_invocations[{index}].time_budget_batch_overshoot_ms",
            decision_id=decision_id,
        )
        if abs(native_overshoot - max(0.0, native_elapsed - invocation_budget)) > 0.001:
            raise DeadlineQualificationError(
                f"{decision_id}: native_invocations[{index}] overshoot is inconsistent"
            )
        native_exhausted = _bool(
            invocation.get("time_budget_exhausted"),
            f"native_invocations[{index}].time_budget_exhausted",
            decision_id=decision_id,
        )
        if remaining and not native_exhausted:
            raise DeadlineQualificationError(
                f"{decision_id}: native_invocations[{index}] left work without a deadline witness"
            )
        root_visits = _require_root_visit_conservation(
            invocation.get("root_visits"),
            completed_iterations=completed,
            decision_id=decision_id,
            invocation_index=index,
        )
        total_completed += completed
        prefixes += int(native_exhausted and 0 < completed < requested)
        normalized_invocations.append(
            {
                "multiplicity": multiplicity,
                "requested_iterations": requested,
                "completed_iterations": completed,
                "remaining_iterations": remaining,
                "time_budget_ms": invocation_budget,
                "time_budget_elapsed_ms": native_elapsed,
                "time_budget_batch_overshoot_ms": native_overshoot,
                "time_budget_exhausted": native_exhausted,
                "root_visits": root_visits,
            }
        )
    if total_completed <= 0:
        raise DeadlineQualificationError(f"{decision_id}: no native invocation completed work")

    return {
        "decision_id": decision_id,
        "outer_wall_ms": _finite_number(record.get("outer_wall_ms"), "outer_wall_ms", decision_id=decision_id),
        "deadline_elapsed_ms": elapsed_ms,
        "deadline_overshoot_ms": overshoot_ms,
        "deadline_exhausted": exhausted,
        "worlds_constructed": constructed,
        "worlds_searched": searched,
        "worlds_budget_skipped": skipped_worlds,
        "native_prefixes": prefixes,
        "native_invocations": normalized_invocations,
    }


def validate_deadline_qualification(
    records: Sequence[Mapping[str, Any]],
    *,
    requirements: DeadlineQualificationRequirements,
) -> dict[str, Any]:
    """Validate the complete, bounded qualification corpus and summarize it."""

    if len(records) != requirements.expected_decisions:
        raise DeadlineQualificationError(
            f"qualification has {len(records)} decisions, expected {requirements.expected_decisions}"
        )
    normalized = [validate_deadline_decision(record, requirements=requirements) for record in records]
    identifiers = [row["decision_id"] for row in normalized]
    if len(set(identifiers)) != len(identifiers):
        raise DeadlineQualificationError("qualification has duplicate decision IDs")
    prefix_count = sum(int(row["native_prefixes"]) for row in normalized)
    if not prefix_count:
        raise DeadlineQualificationError(
            "qualification did not observe a nonzero, completed native deadline prefix"
        )
    outer_walls = sorted(row["outer_wall_ms"] for row in normalized)
    overshoots = sorted(row["deadline_overshoot_ms"] for row in normalized)

    def percentile(values: Sequence[float], fraction: float) -> float:
        position = (len(values) - 1) * fraction
        lower = int(position)
        upper = min(lower + 1, len(values) - 1)
        weight = position - lower
        return values[lower] * (1.0 - weight) + values[upper] * weight

    return {
        "schema_version": DEADLINE_QUALIFICATION_SCHEMA_VERSION,
        "requirements": requirements.to_payload(),
        "decision_count": len(normalized),
        "native_prefix_count": prefix_count,
        "deadline_exhausted_decisions": sum(
            int(row["deadline_exhausted"]) for row in normalized
        ),
        "worlds_budget_skipped": sum(row["worlds_budget_skipped"] for row in normalized),
        "outer_wall_ms": {
            "p50": percentile(outer_walls, 0.5),
            "p95": percentile(outer_walls, 0.95),
            "max": max(outer_walls),
        },
        "deadline_overshoot_ms": {
            "p50": percentile(overshoots, 0.5),
            "p95": percentile(overshoots, 0.95),
            "max": max(overshoots),
        },
        "decisions": normalized,
    }
