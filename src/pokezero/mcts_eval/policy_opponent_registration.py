"""Outcome-independent registration for the bounded three-arm root screen.

Registration describes expected identities; it is not runtime, correctness or
strength evidence. Actual execution must verify them before any policy choice.
The two-hour/96 CPU-hour budget is shared with the later continuation screen.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import re
from typing import Any, Mapping, Sequence

from .policy_opponent_profile import ARMS, MODES, SEED_DOMAIN, root_seed
from .policy_opponent_roster import validate_roster
from .resolver import ContractError


SCHEMA = "pokezero.paper-policy-opponent-profile-registration.v1"
ROSTER_SHA256 = "fa76bc9966bba48f0dd474a2450b4c362e49ac1452db18c39b084a256d30ff7a"
CHAMPION_SHA256 = "0fd095923b4ac7e05d6e2b3ccab9c1e6869dff4893c2dae456caff10dce690be"
SHOWDOWN_SHA256 = "93f81c8eae3d9f769f579a3be5e87bc0d62df39c0c0166bd47a228238188e4b3"
CONFIG = {"depth": 2, "sims": 256, "batch": 16, "worlds": 4, "inference_mode": "local"}
SEMANTICS = {"leaf_eval": "model", "model_priors": True, "use_opponent_priors": False,
    "model_world_workers": 1, "early_stop": False, "c_puct": 1.4,
    "paper_prior_exponent_beta": 1, "complete_paper_reproduction": False,
    "paper_exploration_parameters": "controlled_deviation_not_established_from_publication"}
TIMING = {"deadline_ms": 1000, "native_batch_guard_ms": 64, "tail_p95_limit_ms": 1200,
    "boundary": "request_available_to_validated_showdown_choice",
    "prefix_replay_model_loading_export": "outside_decision_boundary_inside_study_budget",
    "statistics": ["p50", "p95", "p99", "max"], "quantile_method": "nearest_rank"}
BUDGET = {"study_elapsed_seconds": 7200, "study_cpu_hours": 96, "max_concurrent_cpus": 48,
    "shared_with_continuations": True, "profile_concurrency": 1, "cpu_per_runner": 2,
    "torch_threads": 1, "torch_interop_threads": 1, "gpu_count": 0,
    "per_root_worker_seconds": 600, "cleanup_grace_seconds": 5,
    "cap_disposition": "retain_unresolved_no_redraw_no_auto_expansion"}


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ContractError(message)


def canonical_sha256(payload: Any) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":"),
                                   allow_nan=False).encode()).hexdigest()


def freeze_registration(roster: Any, *, roster_sha256: str,
                        expected_runtime: Mapping[str, Any], seed: int = 2026100401) -> dict[str, Any]:
    validate_roster(roster)
    _require(roster_sha256 == ROSTER_SHA256, "profile must bind the original frozen roster bytes")
    _require(isinstance(expected_runtime, Mapping), "profile runtime binding must be an object")
    runtime = dict(expected_runtime)
    _require(set(runtime) == {"source_commit", "source_tree_sha256", "engine_fingerprint",
        "immutable_image", "source_receipt_sha256", "checkpoint_sha256", "model_sha256",
        "tables_sha256", "observation_contract_sha256", "showdown_runtime_sha256"},
        "profile runtime binding is incomplete or has unknown fields")
    for field, value in runtime.items():
        if field == "immutable_image":
            _require(isinstance(value, str) and re.fullmatch(r"[^\s@]+@sha256:[0-9a-f]{64}", value) is not None,
                     "profile requires a digest-qualified immutable image")
        else:
            length = 40 if field == "source_commit" else 64
            _require(isinstance(value, str) and re.fullmatch(rf"[0-9a-f]{{{length}}}", value) is not None,
                     f"profile runtime {field} is malformed")
    _require(runtime["checkpoint_sha256"] == CHAMPION_SHA256, "profile champion changed")
    _require(runtime["showdown_runtime_sha256"] == SHOWDOWN_SHA256, "profile battle oracle changed")
    roots = [{"ordinal": ordinal, "decision_id": row["decision_id"], "seed": row["seed"],
              "seat": row["seat"], "turn_index": row["turn_index"],
              "decision_seed": root_seed(seed, row["decision_id"], purpose="decision"),
              "opponent_seed": root_seed(seed, row["decision_id"], purpose="opponent")}
             for ordinal, row in enumerate(roster["profile_roots"])]
    return {"schema_version": SCHEMA, "status": "REGISTERED_PENDING_RUNTIME_VALIDATION",
            "roster_sha256": roster_sha256, "expected_runtime": runtime,
            "config": dict(CONFIG), "semantics": dict(SEMANTICS), "timing": dict(TIMING),
            "budget": dict(BUDGET), "arms": list(ARMS), "modes": list(MODES), "seed": seed,
            "seed_domain": SEED_DOMAIN.decode().rstrip("\0"), "root_denominator": 32,
            "roots": roots, "proposed_continuation_root_ids": list(roster["proposed_continuation_root_ids"]),
            "analysis": {"all_roots_including_unchanged": True,
                "refusals_keep_original_denominator": True, "replacement_roots": [],
                "pair_counts": "per_native_invocation_without_belief_reweighting_not_global_union",
                "strength_qualified": False}}


def validate_registration(payload: Any, *, roster: Any) -> Mapping[str, Any]:
    _require(isinstance(payload, Mapping), "registration must be an object")
    expected = freeze_registration(roster, roster_sha256=payload.get("roster_sha256"),
        expected_runtime=payload.get("expected_runtime", {}), seed=payload.get("seed"))
    _require(canonical_sha256(dict(payload)) == canonical_sha256(expected),
             "registered profile semantics, budget, roster or analysis drift")
    return payload


def load_registration(path: str | Path, *, expected_sha256: str, roster: Any) -> Mapping[str, Any]:
    """The external byte pin prevents valid-but-different runtime/seed adoption."""
    _require(isinstance(expected_sha256, str) and re.fullmatch(r"[0-9a-f]{64}", expected_sha256) is not None,
             "explicit registration byte hash required")
    raw = Path(path).read_bytes()
    _require(hashlib.sha256(raw).hexdigest() == expected_sha256, "registration byte hash drift")
    return validate_registration(json.loads(raw), roster=roster)


def remaining_study_seconds(*, study_started_unix_s: float, now_unix_s: float,
                            prior_cpu_hours: float, reserved_cpus: int) -> float:
    """Conservative reservation accounting, not sampled process-CPU billing.

    An immutable study start must be shared by profiling and continuations.
    The launcher also bounds each owned worker by monotonic wall time; this
    helper alone is not a timeout or proof of actual scheduler allocations.
    """
    _require(all(type(value) in (int, float) and math.isfinite(value)
                 for value in (study_started_unix_s, now_unix_s, prior_cpu_hours)), "invalid study clock/accounting")
    _require(0 < study_started_unix_s <= now_unix_s and prior_cpu_hours >= 0, "study clock/accounting moved backwards")
    _require(type(reserved_cpus) is int and 0 < reserved_cpus <= BUDGET["max_concurrent_cpus"],
             "study CPU reservation exceeds registered concurrency")
    elapsed = now_unix_s - study_started_unix_s
    return max(0.0, min(BUDGET["study_elapsed_seconds"] - elapsed,
                        (BUDGET["study_cpu_hours"] - prior_cpu_hours) * 3600 / reserved_cpus))


def wall_statistics(values: Sequence[float]) -> dict[str, Any]:
    _require(all(type(value) in (int, float) and math.isfinite(value) and value >= 0 for value in values),
             "invalid completed decision walls")
    ordered = sorted(values)
    if not ordered:
        return {"completed": 0, "p50": None, "p95": None, "p99": None, "max": None}
    return {"completed": len(ordered), "p50": ordered[math.ceil(len(ordered) * .5) - 1],
            "p95": ordered[math.ceil(len(ordered) * .95) - 1],
            "p99": ordered[math.ceil(len(ordered) * .99) - 1], "max": ordered[-1]}
