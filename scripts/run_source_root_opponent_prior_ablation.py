#!/usr/bin/env python3
"""Replay fixed MCTS override roots with only opponent policy priors toggled.

This is a source-bound selection diagnostic, not a strength claim.  It holds
the public root, checkpoint, own priors, search budget, decision RNG, and
native engine fixed.  ``opponent_priors`` changes only the policy priors used
at opponent nodes.  Two independent baseline controls must agree exactly.

Each root is durable on completion and the imported harness owns the manifest,
source receipts, shard receipts, and final validation.  A changed selected
action is only a candidate correction; independent paired continuations are
still required to establish action quality.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence
import re
import time

import run_source_root_leaf_ablation as base


SCHEMA_VERSION = "pokezero.source-root-opponent-prior-ablation.v2"
ARMS = ("model_control_a", "model_control_b", "opponent_priors")
_BASE_MANIFEST = base._manifest
_BASE_VALIDATE_COMPLETED_ROOT = base._validate_completed_root


# The predeclared source-replayable R4 override-audit panel.  Five original
# p2 override roots require a prior *opponent*-owned action that the public
# source record deliberately leaves unresolved; they are explicitly excluded
# below rather than repaired with private action data.  The separate
# sparse-slot diagnostic owns the historical fallback roots.
TARGETS = (
    base.SourceRoot(2026092004, "p1", 19),
    base.SourceRoot(2026092004, "p1", 20),
    base.SourceRoot(2026092005, "p1", 21),
    base.SourceRoot(2026092005, "p1", 25),
    base.SourceRoot(2026092005, "p2", 1),
    base.SourceRoot(2026092005, "p2", 6),
    base.SourceRoot(2026092006, "p1", 2),
    base.SourceRoot(2026092006, "p1", 3),
    base.SourceRoot(2026092007, "p1", 9),
    base.SourceRoot(2026092007, "p1", 19),
    base.SourceRoot(2026092007, "p2", 9),
)

EXCLUDED_UNREPLAYABLE_ROOTS = (
    (base.SourceRoot(2026092004, "p2", 6), "p1 unresolved public event at turn 4"),
    (base.SourceRoot(2026092004, "p2", 10), "p1 unresolved public event at turn 4"),
    (base.SourceRoot(2026092006, "p2", 10), "p1 unresolved public event at turn 3"),
    (base.SourceRoot(2026092006, "p2", 29), "p1 unresolved public event at turn 3"),
    (base.SourceRoot(2026092007, "p2", 19), "p1 unresolved public event at turn 13"),
)


def _opponent_prior_application(telemetry: Mapping[str, Any], *, enabled: bool) -> dict[str, Any]:
    """Fail closed unless native telemetry proves the configured treatment ran."""

    engine = telemetry.get("engine_mcts")
    if not isinstance(engine, Mapping):
        raise base.AblationError("source root has no engine MCTS telemetry")
    raw = engine.get("opponent_prior_application")
    return _validate_opponent_prior_application(raw, enabled=enabled)


def _validate_opponent_prior_application(raw: Any, *, enabled: bool) -> dict[str, Any]:
    """Validate either fresh native telemetry or its durable selection copy."""

    if not isinstance(raw, Mapping) or set(raw) != {
        "enabled", "root_applied", "branch_applied", "total_applied", "digest",
    }:
        raise base.AblationError("source root has malformed opponent-prior application telemetry")
    if raw.get("enabled") is not enabled:
        raise base.AblationError("source root opponent-prior application flag disagrees with arm")
    counts: dict[str, int] = {}
    for field in ("root_applied", "branch_applied", "total_applied"):
        value = raw.get(field)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise base.AblationError(f"source root has invalid opponent-prior {field}")
        counts[field] = value
    if counts["total_applied"] != counts["root_applied"] + counts["branch_applied"]:
        raise base.AblationError("source root opponent-prior applications do not conserve")
    digest = raw.get("digest")
    if enabled:
        if counts["total_applied"] <= 0:
            raise base.AblationError("opponent-prior treatment was not applicable at this root")
        if not isinstance(digest, str) or re.fullmatch(r"[0-9a-f]{64}", digest) is None:
            raise base.AblationError("opponent-prior treatment has no valid applied-vector digest")
    elif counts != {"root_applied": 0, "branch_applied": 0, "total_applied": 0} or digest is not None:
        raise base.AblationError("opponent-prior control unexpectedly applied a treatment vector")
    return {"enabled": enabled, **counts, "digest": digest}


def _control_projection(selection: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "selection": base._control_projection(selection),
        "opponent_prior_application": selection["opponent_prior_application"],
    }


def _new_decider(contract: Any, args: Any, *, use_opponent_priors: bool) -> Any:
    return base._LiveEngineTimingDecider(
        contract,
        args.showdown_root,
        model_decision_time_ms=None,
        model_world_workers=1,
        model_priors=True,
        use_opponent_priors=use_opponent_priors,
        override_telemetry=True,
        rollout_leaf_eval=False,
        rollout_seed=0,
        **base.ROLLOUT,
    )


def _run_root(
    *, root: base.SourceRoot, record: Any, source_records: Sequence[Any],
    historical_fallback: Mapping[str, Any] | None, contract: Any, args: Any,
    manifest_sha256: str,
) -> dict[str, Any]:
    try:
        prefix = base.source_bound_replay_prefix(record, source_records=source_records)
    except base.SourceRootReplayError as error:
        raise base.AblationError(f"{record.decision_id}: source-bound replay repair failed: {error}") from error
    decision_seed = base._decision_seed(record)
    config = base.SearchConfig(**base.SEARCH)
    arm_rows: dict[str, dict[str, Any]] = {}
    for arm in ARMS:
        decider = _new_decider(contract, args, use_opponent_priors=arm == "opponent_priors")
        try:
            started = time.perf_counter()
            telemetry = decider.prepare_public_decision(
                record,
                config,
                public_action_rounds=prefix.public_action_rounds,
                decision_rng_seed=decision_seed,
            )()
            witness = base._selection_witness(telemetry, rollout_leaf_eval=False)
            witness["opponent_prior_application"] = _opponent_prior_application(
                telemetry,
                enabled=arm == "opponent_priors",
            )
            arm_rows[arm] = {
                "wall_seconds": round(time.perf_counter() - started, 6),
                "selection": witness,
            }
        finally:
            decider.close()
    if _control_projection(arm_rows["model_control_a"]["selection"]) != _control_projection(
        arm_rows["model_control_b"]["selection"]
    ):
        raise base.AblationError(f"{record.decision_id}: baseline controls are not deterministic")
    if (
        arm_rows["opponent_priors"]["selection"]["total_iterations"]
        != arm_rows["model_control_a"]["selection"]["total_iterations"]
    ):
        raise base.AblationError(f"{record.decision_id}: opponent-prior arm changed fixed native search work")
    return {
        "schema_version": SCHEMA_VERSION,
        "state": "COMPLETE",
        "manifest_sha256": manifest_sha256,
        "source": root.to_dict(),
        "record_sha256": base._sha256(record.to_dict()),
        "decision_id": record.decision_id,
        "decision_rng_seed": decision_seed,
        "repairs": [repair.to_dict() for repair in prefix.repairs],
        "historical_branch_prior_fallback": (
            None if historical_fallback is None else dict(historical_fallback)
        ),
        "arms": arm_rows,
    }


def _validate_completed_root(
    payload: Any, *, root: base.SourceRoot, record: Any,
    historical_fallback: Mapping[str, Any] | None, manifest_sha256: str,
) -> None:
    """Extend the durable base validator with the treatment's causal receipt."""

    _BASE_VALIDATE_COMPLETED_ROOT(
        payload,
        root=root,
        record=record,
        historical_fallback=historical_fallback,
        manifest_sha256=manifest_sha256,
    )
    assert isinstance(payload, Mapping)  # established by the base validator
    arms = payload["arms"]
    assert isinstance(arms, Mapping)
    selections: dict[str, Mapping[str, Any]] = {}
    for arm in ARMS:
        row = arms[arm]
        assert isinstance(row, Mapping)
        selection = row["selection"]
        assert isinstance(selection, Mapping)
        _validate_opponent_prior_application(
            selection.get("opponent_prior_application"),
            enabled=arm == "opponent_priors",
        )
        selections[arm] = selection
    if _control_projection(selections["model_control_a"]) != _control_projection(
        selections["model_control_b"]
    ):
        raise base.AblationError(f"{root}: durable model controls disagree")
    if selections["opponent_priors"]["total_iterations"] != selections["model_control_a"]["total_iterations"]:
        raise base.AblationError(f"{root}: durable opponent-prior arm changed fixed native search work")


def _manifest(**kwargs: Any) -> dict[str, Any]:
    payload = _BASE_MANIFEST(**kwargs)
    payload["schema_version"] = SCHEMA_VERSION
    payload["model_policy"] = {
        "model_priors": True,
        "baseline_use_opponent_priors": False,
        "treatment_use_opponent_priors": True,
        "override_telemetry": True,
    }
    payload["arms"] = list(ARMS)
    payload["scope"] = {
        "selection_only": True,
        "changed_actions_require_independent_paired_continuations": True,
        "treatment": "opponent-node policy priors only",
        "predeclared_replayable_roots": len(TARGETS),
        "excluded_unreplayable_roots": [
            {"source": root.to_dict(), "reason": reason}
            for root, reason in EXCLUDED_UNREPLAYABLE_ROOTS
        ],
    }
    return payload


def main(argv: Sequence[str] | None = None) -> int:
    # Reuse the hardened durable harness, with all semantic globals replaced in
    # this process before it discovers source records or creates a manifest.
    base.SCHEMA_VERSION = SCHEMA_VERSION
    base.ARMS = ARMS
    base.TARGETS = TARGETS
    base.FALLBACK_TARGETS = frozenset()
    base._run_root = _run_root
    base._manifest = _manifest
    base._validate_completed_root = _validate_completed_root
    base.__doc__ = __doc__
    return base.main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
