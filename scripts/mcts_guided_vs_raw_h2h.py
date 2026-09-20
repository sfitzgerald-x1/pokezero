#!/usr/bin/env python3
"""Run one source-bound, durable guided-MCTS-versus-raw-policy study.

This is deliberately a separate contract from ``mcts_mcts_h2h.py``.  The
candidate is context-aware engine MCTS; the baseline is the same checkpoint's
context-free deterministic masked-argmax policy.  Both seats of each seed are
played, and every completed game is atomically persisted before the next game
can begin.  The companion durable launcher owns the process terminal receipt.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, replace
import hashlib
import json
import math
from pathlib import Path
import sys
import time
from typing import Any, Mapping


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from pokezero.mcts_eval.head_to_head import (  # noqa: E402
    HeadToHeadError,
    MctsPolicySpec,
    PublicOnlyMctsPolicy,
    complete_pair,
    load_pair,
    play_mirrored_pair,
    summarize_complete_pairs,
    write_game_immutable,
)
from pokezero.public_decision_corpus import PublicDecisionRecord  # noqa: E402
from pokezero.actions import ACTION_COUNT  # noqa: E402
from pokezero.engine_search import (  # noqa: E402
    BRANCH_PRIOR_FALLBACK_REASON_VALUES,
    OVERRIDE_UNMEASURED_CAUSE_VALUES,
)

# The mature MCTS-versus-MCTS runner owns the source-hash, immutable-write,
# Showdown-binding and durable-launcher primitives.  This runner intentionally
# reuses those primitives rather than carrying a second, subtly weaker version.
from mcts_mcts_h2h import (  # noqa: E402
    _durable_output_root,
    _require_durable_launcher_handoff,
    _runtime_spec,
    _sha256_file,
    _showdown_source_provenance,
    _source_provenance,
    _write_immutable_json,
    _write_progress_json,
)


MANIFEST_SCHEMA_VERSION = "pokezero.mcts-guided-vs-raw-manifest.v1"
PROGRESS_SCHEMA_VERSION = "pokezero.mcts-guided-vs-raw-progress.v1"
COMPLETE_SCHEMA_VERSION = "pokezero.mcts-guided-vs-raw-complete.v1"
PUBLIC_DECISION_EVIDENCE_SCHEMA_VERSION = "pokezero.mcts-guided-vs-raw-public-decision.v1"
BRANCH_PRIOR_LEDGER_EVIDENCE_SCHEMA_VERSION = (
    "pokezero.mcts-guided-vs-raw-branch-prior-ledger.v5"
)
SEALED_OVERRIDE_AUDIT_EVIDENCE_SCHEMA_VERSION = (
    "pokezero.mcts-guided-vs-raw-sealed-override-audit.v2"
)
RAW_SELECTOR = {
    "kind": "deterministic_masked_argmax",
    "deterministic": True,
    "exploration_epsilon": 0.0,
    "sampling_temperature": 1.0,
    "family_gated_selection": False,
    "search": False,
}
REGISTERED_ENGINE_CONFIG = {
    "approximate_hidden_duration_volatiles": True,
    "approximate_partial_trap_turns": True,
    "approximate_sleep_turns": True,
    "approximate_substitute_health": True,
    "c_puct": 1.4,
    "deep_ko_split": True,
    "depth_min": None,
    "early_stop": False,
    "early_stop_min_sims": 64,
    "fold_cross_check": False,
    "fpu_reduction": None,
    "ladder_saturation": 0.9,
    "leaf_batch": 1,
    "leaf_batch_fidelity_loss_ack": False,
    "leaf_eval": "model",
    "model_decision_time_ms": 1_000,
    "model_device": "cpu",
    "model_native_batch_guard_ms": 64,
    "model_priors": True,
    "model_world_workers": 1,
    "override_telemetry": True,
    "rollout_branch_on_damage": False,
    "rollout_count": 32,
    "rollout_leaf_eval": False,
    "rollout_max_plies": 200,
    "rollout_policy": "uniform",
    "rollout_seed": 0,
    "rollout_threads": 1,
    "rollout_threads_cpu_budget_ack": False,
    "root_selector_shadow": False,
    "sample_retry_factor": 4,
    "search_batch": 16,
    "search_depth": 2,
    "search_sims": 256,
    "search_time_ms": 100,
    "strict_fallbacks": True,
    "threads": 1,
    "use_opponent_priors": False,
    "worlds": 4,
    "worlds_min": None,
}
# The sidecar is intentionally not a general-purpose MCTS evaluator: it may
# only certify a configuration whose search semantics have been registered in
# advance.  The CUDA deep protocol is the exact fixed-work configuration used
# for the fallback-concentration investigation.  Keeping both complete
# dictionaries here means a new run cannot accidentally turn an evidence
# replay into a look-alike budget sweep by changing one unreviewed knob.
REGISTERED_DEEP_ENGINE_CONFIG = {
    **REGISTERED_ENGINE_CONFIG,
    "model_decision_time_ms": None,
    "model_device": "cuda",
    "model_native_batch_guard_ms": 0,
    "search_depth": 6,
    "search_sims": 4096,
    "search_time_ms": 1_000,
}
REGISTERED_ENGINE_CONFIGS = (
    REGISTERED_ENGINE_CONFIG,
    REGISTERED_DEEP_ENGINE_CONFIG,
)
SOURCE_BOUND_ENGINE_PATHS = {"checkpoint_path", "model_path", "tables_path"}


def _mapping(value: object, *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise HeadToHeadError(f"{label} must be a JSON object.")
    return value


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _load_manifest(path: str | Path) -> Mapping[str, Any]:
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise HeadToHeadError(f"cannot read guided-vs-raw manifest: {error}") from error
    manifest = _mapping(payload, label="manifest")
    if manifest.get("schema_version") != MANIFEST_SCHEMA_VERSION:
        raise HeadToHeadError(
            f"manifest schema {manifest.get('schema_version')!r} is not "
            f"{MANIFEST_SCHEMA_VERSION!r}."
        )
    return manifest


def _hex(value: object, *, label: str, length: int) -> str:
    text = str(value)
    if len(text) != length or any(character not in "0123456789abcdef" for character in text):
        raise HeadToHeadError(f"{label} must be a {length}-character lowercase SHA-256 value.")
    return text


def _seeds(manifest: Mapping[str, Any]) -> tuple[int, ...]:
    raw = manifest.get("seeds")
    if not isinstance(raw, list) or not raw:
        raise HeadToHeadError("manifest.seeds must be a non-empty JSON list.")
    if any(isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in raw):
        raise HeadToHeadError("manifest.seeds must contain non-negative integer seeds only.")
    seeds = tuple(raw)
    if len(set(seeds)) != len(seeds):
        raise HeadToHeadError("manifest.seeds contains a duplicate mirrored-pair key.")
    return seeds


def _bootstrap(manifest: Mapping[str, Any]) -> tuple[int, int, float]:
    raw = _mapping(manifest.get("bootstrap"), label="manifest.bootstrap")
    resamples = raw.get("resamples")
    seed = raw.get("seed")
    confidence = raw.get("confidence_level")
    if (
        isinstance(resamples, bool)
        or not isinstance(resamples, int)
        or resamples <= 0
        or isinstance(seed, bool)
        or not isinstance(seed, int)
        or seed < 0
        or isinstance(confidence, bool)
        or not isinstance(confidence, (int, float))
        or not 0.0 < float(confidence) < 1.0
    ):
        raise HeadToHeadError("manifest.bootstrap is malformed.")
    return resamples, seed, float(confidence)


def _raw_spec(
    raw: Mapping[str, Any],
    *,
    checkpoint_sha256: str,
    source_commit: str,
    source_tree_sha256: str,
    engine_fingerprint: str,
    showdown_source_sha256: str,
) -> MctsPolicySpec:
    """Freeze the no-search baseline without pretending it is engine MCTS."""

    if dict(_mapping(raw.get("selector"), label="raw.selector")) != RAW_SELECTOR:
        raise HeadToHeadError(
            "raw.selector must exactly declare deterministic masked argmax with no search."
        )
    for field, actual in (
        ("source_commit", source_commit),
        ("source_tree_sha256", source_tree_sha256),
        ("engine_fingerprint", engine_fingerprint),
        ("checkpoint_sha256", checkpoint_sha256),
        ("showdown_source_sha256", showdown_source_sha256),
    ):
        if str(raw.get(field, "")) != actual:
            raise HeadToHeadError(f"raw.{field} does not match the active verified identity.")
    config_id = str(raw.get("config_id", ""))
    policy_id = str(raw.get("policy_id", ""))
    if not config_id or not policy_id:
        raise HeadToHeadError("raw config_id and policy_id must be non-empty.")
    return MctsPolicySpec(
        config_id=config_id,
        policy_id=policy_id,
        source_commit=source_commit,
        source_tree_sha256=source_tree_sha256,
        engine_fingerprint=engine_fingerprint,
        checkpoint_sha256=checkpoint_sha256,
        showdown_source_sha256=showdown_source_sha256,
        config={"policy_kind": "raw_transformer_policy", "selector": dict(RAW_SELECTOR)},
    )


@dataclass
class RawPolicyStats:
    """Explicit zero-search telemetry for the raw-policy baseline."""

    decisions: int = 0
    searched_decisions: int = 0
    fallback_decisions: int = 0
    model_evals: int = 0
    total_iterations: int = 0
    worlds_constructed: int = 0
    worlds_searched: int = 0
    prior_fallbacks: int = 0
    root_prior_fallbacks: int = 0
    branch_prior_fallbacks: int = 0
    opponent_prior_arm_decisions: int = 0
    override_measured_decisions: int = 0
    model_override_decisions: int = 0
    raw_forward_decisions: int = 0
    opponent_request_order_statuses: dict[str, int] | None = None
    opponent_request_order_root_fallback_statuses: dict[str, int] | None = None
    decision_wall_seconds: float = 0.0

    def __post_init__(self) -> None:
        if self.opponent_request_order_statuses is None:
            self.opponent_request_order_statuses = {}
        if self.opponent_request_order_root_fallback_statuses is None:
            self.opponent_request_order_root_fallback_statuses = {}

    def to_payload(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class SealedOverrideAuditConfig:
    """The explicit, source-bound continuation-audit contract.

    The audit is deliberately opt-in: each measured override launches two
    independent continuations, which is appropriate for a focused causal
    study but would silently change the cost of an ordinary strength run.
    """

    max_continuation_decision_rounds: int


class DeterministicRawPolicyAdapter:
    """Give the raw policy a safe context entry point and honest telemetry."""

    requires_public_materialization_state = False

    def __init__(self, policy: Any, *, policy_id: str) -> None:
        self._policy = policy
        self.policy_id = policy_id
        self.stats = RawPolicyStats()

    def select_action(self, observation: Any, *, rng: Any) -> Any:
        started = time.perf_counter()
        try:
            decision = self._policy.select_action(observation, rng=rng)
            if not hasattr(decision, "policy_id"):
                raise HeadToHeadError("raw policy did not return a policy-identified decision.")
            self.stats.raw_forward_decisions += 1
            return replace(decision, policy_id=self.policy_id)
        finally:
            self.stats.decisions += 1
            self.stats.decision_wall_seconds += time.perf_counter() - started

    def select_action_with_context(self, context: Any, *, rng: Any) -> Any:
        # The raw policy only consumes its own observation.  It never receives
        # an opponent request, legal mask, or private history from this wrapper.
        return self.select_action(context.observation, rng=rng)

    def reset(self) -> None:
        self._policy.reset()

    def close(self) -> None:
        return None


def _validated_study(manifest: Mapping[str, Any], *, seeds: tuple[int, ...]) -> Mapping[str, Any]:
    study = _mapping(manifest.get("study"), label="manifest.study")
    if study.get("kind") != "guided_mcts_vs_raw_policy":
        raise HeadToHeadError("study.kind must be guided_mcts_vs_raw_policy.")
    if study.get("pairs") != len(seeds):
        raise HeadToHeadError("study.pairs must equal the exact manifest seed count.")
    if study.get("mirrored_games") != len(seeds) * 2:
        raise HeadToHeadError("study.mirrored_games must be exactly twice study.pairs.")
    if study.get("failure_retry_policy") != {
        "interrupted_before_runner_terminal": "resume_same_root_with_fresh_launcher_attempt",
        "nonzero_runner_exit": "terminal_failed_no_retry",
        "malformed_runner_terminal": "nonbankable_no_retry",
        "completed_game_units": "immutable_reuse_only",
    }:
        raise HeadToHeadError("study.failure_retry_policy is not the registered durable policy.")
    return study


def _sealed_override_audit_config(
    manifest: Mapping[str, Any],
) -> SealedOverrideAuditConfig | None:
    """Parse the optional, fail-closed override-continuation contract.

    A missing block means this remains an ordinary public-evidence study.  If
    the block is present, every *measured* guided override must receive both
    independent raw-policy continuations; sampling only convenient overrides
    would make the causal denominator depend on the search result.
    """

    raw = manifest.get("sealed_override_audit")
    if raw is None:
        return None
    audit = _mapping(raw, label="manifest.sealed_override_audit")
    expected = {
        "schema_version",
        "continuation_selector",
        "max_continuation_decision_rounds",
    }
    if set(audit) != expected:
        raise HeadToHeadError("manifest.sealed_override_audit has unsupported fields.")
    if audit.get("schema_version") != SEALED_OVERRIDE_AUDIT_EVIDENCE_SCHEMA_VERSION:
        raise HeadToHeadError("manifest.sealed_override_audit has an unsupported schema version.")
    if dict(_mapping(audit.get("continuation_selector"), label="sealed audit selector")) != RAW_SELECTOR:
        raise HeadToHeadError(
            "sealed override continuations must use the registered deterministic raw selector."
        )
    maximum = audit.get("max_continuation_decision_rounds")
    if isinstance(maximum, bool) or not isinstance(maximum, int) or maximum <= 0:
        raise HeadToHeadError(
            "sealed override audit max_continuation_decision_rounds must be a positive integer."
        )
    return SealedOverrideAuditConfig(max_continuation_decision_rounds=maximum)


def _require_registered_candidate_config(config: Mapping[str, Any]) -> None:
    """Reject a look-alike MCTS configuration before any game is played."""

    if config.get("model_priors") is not True or config.get("use_opponent_priors") is not False:
        raise HeadToHeadError("candidate must enable own model priors and disable opponent priors.")
    observed = {key: value for key, value in config.items() if key not in SOURCE_BOUND_ENGINE_PATHS}
    if observed not in REGISTERED_ENGINE_CONFIGS:
        raise HeadToHeadError(
            "candidate differs from every registered guided-MCTS evaluation configuration."
        )


def _raw_witness_path(out_root: Path, *, seed: int, candidate_seat: str) -> Path:
    if candidate_seat not in {"p1", "p2"}:
        raise HeadToHeadError("raw selector witness has an invalid candidate seat.")
    return out_root / "raw-selector-witnesses" / f"seed-{seed}-{candidate_seat}.json"


def _public_decision_path(
    out_root: Path,
    *,
    seed: int,
    candidate_seat: str,
    record: PublicDecisionRecord,
) -> Path:
    if candidate_seat not in {"p1", "p2"}:
        raise HeadToHeadError("public decision evidence has an invalid candidate seat.")
    if record.seed != seed or record.acting_player != candidate_seat:
        raise HeadToHeadError("public decision evidence does not match its game identity.")
    return (
        out_root
        / "public-decision-records"
        / f"seed-{seed}-{candidate_seat}"
        / f"turn-{record.turn_index:03d}-{record.decision_id}.json"
    )


def _branch_prior_ledger_path(
    out_root: Path,
    *,
    seed: int,
    candidate_seat: str,
    record: PublicDecisionRecord,
) -> Path:
    """Return the immutable, public-safe native-branch witness address."""

    if record.seed != seed or record.acting_player != candidate_seat:
        raise HeadToHeadError("branch prior ledger does not match its public decision identity.")
    return (
        out_root
        / "branch-prior-fallback-ledgers"
        / f"seed-{seed}-{candidate_seat}"
        / f"turn-{record.turn_index:03d}-{record.decision_id}.json"
    )


def _sealed_override_audit_path(
    out_root: Path,
    *,
    seed: int,
    candidate_seat: str,
    decision_round_index: int,
) -> Path:
    """Return the sealed controller's immutable, non-private readout path."""

    if candidate_seat not in {"p1", "p2"}:
        raise HeadToHeadError("sealed override audit has an invalid candidate seat.")
    if (
        isinstance(decision_round_index, bool)
        or not isinstance(decision_round_index, int)
        or decision_round_index < 0
    ):
        raise HeadToHeadError("sealed override audit has an invalid decision round.")
    return (
        out_root
        / "sealed-override-audits"
        / f"seed-{seed}-{candidate_seat}"
        / f"round-{decision_round_index:03d}.json"
    )


def _sealed_override_audit_payload(
    *,
    candidate: MctsPolicySpec,
    incumbent: MctsPolicySpec,
    candidate_seat: str,
    readout: Mapping[str, Any],
) -> dict[str, Any]:
    """Wrap one fully normalized controller readout in source identities.

    This is deliberately a closed projection *before* the immutable write.
    A later validator cannot retract a private field from an append-only
    evidence root, so arbitrary controller JSON is never an admissible input.
    """

    copied = _validated_sealed_override_readout(readout, candidate_seat=candidate_seat)
    return {
        "schema_version": SEALED_OVERRIDE_AUDIT_EVIDENCE_SCHEMA_VERSION,
        "candidate_provenance_sha256": candidate.provenance_sha256,
        "raw_provenance_sha256": incumbent.provenance_sha256,
        "candidate_seat": candidate_seat,
        "readout": copied,
    }


def _sealed_action(value: object, *, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value < ACTION_COUNT:
        raise HeadToHeadError(f"{label} must be a valid action index.")
    return value


def _validated_sealed_terminal(value: object, *, label: str) -> dict[str, Any]:
    terminal = _mapping(value, label=label)
    if set(terminal) != {"winner", "turn_count", "capped"}:
        raise HeadToHeadError(f"{label} has an unsupported shape.")
    winner = terminal.get("winner")
    if winner not in {"p1", "p2", None} or not isinstance(terminal.get("capped"), bool):
        raise HeadToHeadError(f"{label} has an invalid terminal disposition.")
    if terminal["capped"]:
        raise HeadToHeadError("sealed override continuation must be uncapped.")
    return {
        "winner": winner,
        "turn_count": _nonnegative_int(terminal.get("turn_count"), label=f"{label} turn count"),
        "capped": False,
    }


def _validated_sealed_continuation(value: object, *, label: str) -> dict[str, Any]:
    continuation = _mapping(value, label=label)
    if set(continuation) != {
        "decision_round_count",
        "terminal_after_fixed_joint_step",
        "terminal",
    }:
        raise HeadToHeadError(f"{label} has an unsupported shape.")
    immediate = continuation.get("terminal_after_fixed_joint_step")
    if not isinstance(immediate, bool):
        raise HeadToHeadError(f"{label} immediate-terminal flag must be boolean.")
    count = _nonnegative_int(continuation.get("decision_round_count"), label=f"{label} rounds")
    if immediate != (count == 0):
        raise HeadToHeadError(f"{label} immediate-terminal flag disagrees with its round count.")
    return {
        "decision_round_count": count,
        "terminal_after_fixed_joint_step": immediate,
        "terminal": _validated_sealed_terminal(continuation.get("terminal"), label=f"{label} terminal"),
    }


def _validated_sealed_search_evidence(value: object) -> dict[str, Any]:
    """Return the closed, action-index-only selection evidence safe for a sidecar."""

    evidence = _mapping(value, label="sealed override search evidence")
    expected = {
        "model_argmax",
        "search_argmax",
        "model_override",
        "root_q_gap",
        "root_visit_gap",
        "root_gap_action_indices",
        "root_allocation_missing_action_indices",
        "root_allocation",
    }
    if set(evidence) != expected or evidence.get("model_override") is not True:
        raise HeadToHeadError("sealed override search evidence has an unsupported shape.")
    model_action = _sealed_action(evidence.get("model_argmax"), label="sealed model action")
    search_action = _sealed_action(evidence.get("search_argmax"), label="sealed search action")
    if model_action == search_action:
        raise HeadToHeadError("sealed override actions must differ.")
    root = _mapping(evidence.get("root_allocation"), label="sealed root allocation")
    if set(root) != {"worlds", "prior_authority", "prior_cause", "arms"}:
        raise HeadToHeadError("sealed root allocation has an unsupported shape.")
    worlds = _nonnegative_int(root.get("worlds"), label="sealed root worlds")
    if worlds <= 0 or root.get("prior_authority") is not True or root.get("prior_cause") is not None:
        raise HeadToHeadError("sealed root allocation has invalid prior authority.")
    arms = root.get("arms")
    if not isinstance(arms, list) or not arms:
        raise HeadToHeadError("sealed root allocation must include action-indexed arms.")
    normalized_arms: list[dict[str, Any]] = []
    for raw_arm in arms:
        arm = _mapping(raw_arm, label="sealed root arm")
        if set(arm) != {"action_index", "visit_share", "q", "reported_prior", "model_prior"}:
            raise HeadToHeadError("sealed root arm has an unsupported shape.")
        visit_share = _finite_number(arm.get("visit_share"), label="sealed root visit share")
        if not 0.0 <= visit_share <= 1.0:
            raise HeadToHeadError("sealed root visit share must be within [0, 1].")
        q = arm.get("q")
        normalized_arms.append(
            {
                "action_index": _sealed_action(arm.get("action_index"), label="sealed root action"),
                "visit_share": visit_share,
                "q": None if q is None else _finite_number(q, label="sealed root Q"),
                "reported_prior": (
                    None if arm.get("reported_prior") is None else _finite_number(
                        arm.get("reported_prior"), label="sealed reported prior"
                    )
                ),
                "model_prior": (
                    None if arm.get("model_prior") is None else _finite_number(
                        arm.get("model_prior"), label="sealed model prior"
                    )
                ),
            }
        )
    if len({arm["action_index"] for arm in normalized_arms}) != len(normalized_arms):
        raise HeadToHeadError("sealed root allocation repeats an action index.")
    if not math.isclose(sum(arm["visit_share"] for arm in normalized_arms), 1.0, abs_tol=1e-5):
        raise HeadToHeadError("sealed root visit shares do not conserve one.")
    for field in ("reported_prior", "model_prior"):
        values = [arm[field] for arm in normalized_arms]
        if any(value is not None and not 0.0 <= value <= 1.0 for value in values):
            raise HeadToHeadError(f"sealed {field} must be within [0, 1].")
        if any(value is not None for value in values) and (
            any(value is None for value in values)
            or not math.isclose(sum(value for value in values if value is not None), 1.0, abs_tol=1e-5)
        ):
            raise HeadToHeadError(f"sealed {field} must cover and conserve every root arm.")
    if any(arm["model_prior"] is None for arm in normalized_arms):
        raise HeadToHeadError("sealed model-prior authority must cover every root arm.")
    missing = evidence.get("root_allocation_missing_action_indices")
    if not isinstance(missing, list):
        raise HeadToHeadError("sealed root missing-action coverage must be a list.")
    normalized_missing = [
        _sealed_action(action, label="sealed root missing action") for action in missing
    ]
    if len(set(normalized_missing)) != len(normalized_missing):
        raise HeadToHeadError("sealed root missing-action coverage repeats an action index.")
    if set(normalized_missing) & {arm["action_index"] for arm in normalized_arms}:
        raise HeadToHeadError("sealed root missing-action coverage overlaps its root allocation.")
    gaps = evidence.get("root_gap_action_indices")
    if not isinstance(gaps, list) or len(gaps) > 2:
        raise HeadToHeadError("sealed root gap actions have an unsupported shape.")
    normalized_gaps = [_sealed_action(action, label="sealed root gap action") for action in gaps]
    if len(set(normalized_gaps)) != len(normalized_gaps):
        raise HeadToHeadError("sealed root gap actions repeat an action index.")
    def optional_finite(value: object, *, label: str) -> float | None:
        return None if value is None else _finite_number(value, label=label)
    return {
        "model_argmax": model_action,
        "search_argmax": search_action,
        "model_override": True,
        "root_q_gap": optional_finite(evidence.get("root_q_gap"), label="sealed root Q gap"),
        "root_visit_gap": optional_finite(evidence.get("root_visit_gap"), label="sealed root visit gap"),
        "root_gap_action_indices": normalized_gaps,
        "root_allocation_missing_action_indices": normalized_missing,
        "root_allocation": {
            "worlds": worlds,
            "prior_authority": True,
            "prior_cause": None,
            "arms": normalized_arms,
        },
    }


def _sealed_search_evidence_from_selection(selection: Mapping[str, Any]) -> dict[str, Any]:
    """Project the public decision ledger's measured selection into sidecar shape."""

    return _validated_sealed_search_evidence(
        {
            key: selection[key]
            for key in (
                "model_argmax",
                "search_argmax",
                "model_override",
                "root_q_gap",
                "root_visit_gap",
                "root_gap_action_indices",
                "root_allocation_missing_action_indices",
                "root_allocation",
            )
        }
    )


def _validated_sealed_override_readout(value: object, *, candidate_seat: str) -> dict[str, Any]:
    """Normalize the complete public-safe controller result before persistence."""

    readout = _mapping(value, label="sealed override audit readout")
    base_fields = {
        "schema_version", "seed", "battle_id", "candidate_seat", "decision_round_index", "audit_status"
    }
    if (
        not base_fields.issubset(readout)
        or readout.get("schema_version") != "pokezero.mcts-sealed-override-audit.v1"
    ):
        raise HeadToHeadError("sealed override audit readout has an unsupported shape.")
    if readout.get("candidate_seat") != candidate_seat:
        raise HeadToHeadError("sealed override audit readout has the wrong candidate seat.")
    seed = _nonnegative_int(readout.get("seed"), label="sealed source seed")
    round_index = _nonnegative_int(
        readout.get("decision_round_index"), label="sealed source decision round"
    )
    battle_id = readout.get("battle_id")
    if not isinstance(battle_id, str) or not battle_id:
        raise HeadToHeadError("sealed override audit readout has an invalid battle identity.")
    audit_status = readout.get("audit_status")
    if audit_status == "INAPPLICABLE_NON_SIMULTANEOUS":
        if set(readout) != base_fields | {"requested_players", "search_evidence"}:
            raise HeadToHeadError("inapplicable sealed override audit has an unsupported shape.")
        if readout.get("requested_players") != [candidate_seat]:
            raise HeadToHeadError("inapplicable sealed override audit has the wrong request boundary.")
        return {
            "schema_version": "pokezero.mcts-sealed-override-audit.v1",
            "seed": seed,
            "battle_id": battle_id,
            "candidate_seat": candidate_seat,
            "decision_round_index": round_index,
            "audit_status": audit_status,
            "requested_players": [candidate_seat],
            "search_evidence": _validated_sealed_search_evidence(
                readout.get("search_evidence")
            ),
        }
    if audit_status != "PAIRED" or set(readout) != base_fields | {"audit"}:
        raise HeadToHeadError("sealed override audit readout has an unsupported disposition.")
    audit = _mapping(readout.get("audit"), label="sealed override pair")
    expected = {
        "schema_version", "source_battle_id", "source_seed", "source_decision_round",
        "subject_player", "opponent_player", "mcts_action", "raw_action",
        "opponent_action_held_fixed", "search_evidence", "mcts", "raw",
    }
    if set(audit) != expected or audit.get("schema_version") != "pokezero.sealed-override-pair.v1":
        raise HeadToHeadError("sealed override pair has an unsupported shape.")
    opponent = "p2" if candidate_seat == "p1" else "p1"
    if (
        audit.get("source_battle_id") != battle_id
        or audit.get("source_seed") != seed
        or audit.get("source_decision_round") != round_index
        or audit.get("subject_player") != candidate_seat
        or audit.get("opponent_player") != opponent
        or audit.get("opponent_action_held_fixed") is not True
    ):
        raise HeadToHeadError("sealed override pair does not bind its source boundary.")
    evidence = _validated_sealed_search_evidence(audit.get("search_evidence"))
    mcts_action = _sealed_action(audit.get("mcts_action"), label="sealed MCTS action")
    raw_action = _sealed_action(audit.get("raw_action"), label="sealed raw action")
    if (
        mcts_action != evidence["search_argmax"]
        or raw_action != evidence["model_argmax"]
        or mcts_action == raw_action
    ):
        raise HeadToHeadError("sealed override actions do not bind its selection evidence.")
    return {
        "schema_version": "pokezero.mcts-sealed-override-audit.v1",
        "seed": seed,
        "battle_id": battle_id,
        "candidate_seat": candidate_seat,
        "decision_round_index": round_index,
        "audit_status": "PAIRED",
        "audit": {
            "schema_version": "pokezero.sealed-override-pair.v1",
            "source_battle_id": battle_id,
            "source_seed": seed,
            "source_decision_round": round_index,
            "subject_player": candidate_seat,
            "opponent_player": opponent,
            "mcts_action": mcts_action,
            "raw_action": raw_action,
            "opponent_action_held_fixed": True,
            "search_evidence": evidence,
            "mcts": _validated_sealed_continuation(audit.get("mcts"), label="sealed MCTS continuation"),
            "raw": _validated_sealed_continuation(audit.get("raw"), label="sealed raw continuation"),
        },
    }


def _nonnegative_int(value: object, *, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise HeadToHeadError(f"{label} must be a non-negative integer.")
    return value


def _finite_number(value: object, *, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise HeadToHeadError(f"{label} must be a finite number.")
    return float(value)


def _validated_branch_prior_ledger(value: object) -> dict[str, Any]:
    """Validate the bounded native-tree evidence safe to retain beside a public row."""

    ledger = _mapping(value, label="branch prior fallback ledger")
    expected = {
        "schema_version",
        "native_invocations",
        "belief_worlds",
        "branch_prior_fallbacks",
        "reason_counts",
        "unclassified_branch_prior_fallbacks",
        "reason_ledger_complete",
        "events",
    }
    if set(ledger) != expected or ledger.get("schema_version") != (
        "pokezero.engine-mcts.branch-prior-fallbacks.v1"
    ):
        raise HeadToHeadError("branch prior ledger has an unexpected schema.")
    invocations = _nonnegative_int(ledger.get("native_invocations"), label="native invocations")
    _nonnegative_int(ledger.get("belief_worlds"), label="belief worlds")
    branch_fallbacks = _nonnegative_int(
        ledger.get("branch_prior_fallbacks"), label="branch prior fallbacks"
    )
    unclassified = _nonnegative_int(
        ledger.get("unclassified_branch_prior_fallbacks"), label="unclassified fallbacks"
    )
    if not isinstance(ledger.get("reason_ledger_complete"), bool):
        raise HeadToHeadError("branch prior ledger completeness must be boolean.")
    if bool(ledger["reason_ledger_complete"]) != (unclassified == 0):
        raise HeadToHeadError("branch prior ledger completeness disagrees with unclassified counts.")
    reason_counts = _mapping(ledger.get("reason_counts"), label="branch prior reason counts")
    if set(reason_counts) != BRANCH_PRIOR_FALLBACK_REASON_VALUES:
        raise HeadToHeadError(
            "branch prior reason ledger must use the complete native reason vocabulary."
        )
    normalized_reasons = {
        reason: _nonnegative_int(count, label=f"branch prior reason {reason!r}")
        for reason, count in reason_counts.items()
    }
    if sum(normalized_reasons.values()) + unclassified != branch_fallbacks:
        raise HeadToHeadError("branch prior reasons do not conserve total fallbacks.")
    events = ledger.get("events")
    if not isinstance(events, list) or len(events) != invocations:
        raise HeadToHeadError("branch prior ledger native invocation count does not match events.")
    normalized_events: list[dict[str, Any]] = []
    event_reason_totals = {reason: 0 for reason in normalized_reasons}
    event_unclassified = 0
    for event in events:
        event_mapping = _mapping(event, label="branch prior native invocation")
        if set(event_mapping) != {
            "native_invocation",
            "belief_records",
            "collapse_multiplicity",
            "branch_prior_fallbacks",
            "reason_counts",
        }:
            raise HeadToHeadError("branch prior native invocation has unsupported fields.")
        event_reasons = event_mapping.get("reason_counts")
        if event_reasons is not None:
            event_reasons = _mapping(event_reasons, label="native invocation reason counts")
            if set(event_reasons) != set(normalized_reasons):
                raise HeadToHeadError("native invocation reason vocabulary differs from its ledger.")
            event_reasons = {
                reason: _nonnegative_int(count, label=f"native reason {reason!r}")
                for reason, count in event_reasons.items()
            }
        event_fallbacks = _nonnegative_int(
            event_mapping.get("branch_prior_fallbacks"), label="native invocation fallbacks"
        )
        if event_reasons is not None and sum(event_reasons.values()) != event_fallbacks:
            raise HeadToHeadError("native invocation reasons do not conserve its fallbacks.")
        if event_reasons is None:
            event_unclassified += event_fallbacks
        else:
            for reason, count in event_reasons.items():
                event_reason_totals[reason] += count
        normalized_events.append(
            {
                "native_invocation": _nonnegative_int(
                    event_mapping.get("native_invocation"), label="native invocation identity"
                ),
                "belief_records": _nonnegative_int(
                    event_mapping.get("belief_records"), label="native invocation belief records"
                ),
                "collapse_multiplicity": _nonnegative_int(
                    event_mapping.get("collapse_multiplicity"),
                    label="native invocation collapse multiplicity",
                ),
                "branch_prior_fallbacks": event_fallbacks,
                "reason_counts": event_reasons,
            }
        )
    if [event["native_invocation"] for event in normalized_events] != list(range(1, invocations + 1)):
        raise HeadToHeadError("branch prior native invocation identities must be contiguous.")
    if sum(event["branch_prior_fallbacks"] for event in normalized_events) != branch_fallbacks:
        raise HeadToHeadError("branch prior native invocations do not conserve total fallbacks.")
    if event_reason_totals != normalized_reasons or event_unclassified != unclassified:
        raise HeadToHeadError(
            "branch prior native invocation attribution disagrees with its ledger totals."
        )
    return {
        "schema_version": ledger["schema_version"],
        "native_invocations": invocations,
        "belief_worlds": int(ledger["belief_worlds"]),
        "branch_prior_fallbacks": branch_fallbacks,
        "reason_counts": dict(sorted(normalized_reasons.items())),
        "unclassified_branch_prior_fallbacks": unclassified,
        "reason_ledger_complete": bool(ledger["reason_ledger_complete"]),
        "events": normalized_events,
    }


def _guided_override_from_decision(
    guided_policy: PublicOnlyMctsPolicy,
    record: PublicDecisionRecord,
) -> tuple[Mapping[str, Any], tuple[str, ...]]:
    address = guided_policy.latest_decision_address
    expected_address = {
        "battle_id": record.battle_id,
        "round": record.turn_index,
        "seat": record.acting_player,
        "action_index": record.recorded_action_index,
    }
    if not isinstance(address, Mapping):
        raise HeadToHeadError("guided policy is missing a public decision address.")
    address_fields = set(address)
    if address_fields != set(expected_address) | {"requested_players"}:
        raise HeadToHeadError("guided policy decision address has unsupported fields.")
    if {key: address[key] for key in expected_address} != expected_address:
        raise HeadToHeadError(
            "guided policy metadata is not bound to the public decision being committed."
        )
    requested = address.get("requested_players")
    if not isinstance(requested, list) or any(player not in {"p1", "p2"} for player in requested):
        raise HeadToHeadError("guided policy decision has an invalid request boundary.")
    requested_players = tuple(requested)
    if requested_players not in {(record.acting_player,), ("p1", "p2")}:
        raise HeadToHeadError("guided policy decision has an unsupported request boundary.")
    metadata = _mapping(guided_policy.latest_decision_metadata, label="guided decision metadata")
    engine_mcts = _mapping(metadata.get("engine_mcts"), label="guided engine MCTS metadata")
    return (
        _mapping(engine_mcts.get("override"), label="guided override telemetry"),
        requested_players,
    )


def _branch_prior_ledger_from_override(override: Mapping[str, Any]) -> dict[str, Any]:
    return _validated_branch_prior_ledger(override.get("branch_prior_fallbacks"))


def _validated_selection_evidence(
    selection: Mapping[str, Any],
    *,
    record: PublicDecisionRecord,
) -> dict[str, Any]:
    """Project the safe, decision-local root allocation used by MCTS selection.

    This deliberately excludes opponent arms and world inputs.  It retains the
    candidate's legal action labels, own prior, visit share and Q estimate so a
    later independent continuation audit can test whether an override was
    justified without replaying or exposing the hidden state.
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
    # The producer does not know the public record, so this field is derived
    # here and then bound into the durable sidecar.  Accepting its absence is
    # required for the live producer boundary; accepting a *wrong* persisted
    # value would turn an engine/request action-surface mismatch into an
    # apparently complete allocation.
    coverage_field = "root_allocation_missing_action_indices"
    if set(selection) not in (required, required | {coverage_field}):
        raise HeadToHeadError("guided selection evidence has unsupported fields.")
    search_action = selection["search_argmax"]
    if (
        isinstance(search_action, bool)
        or not isinstance(search_action, int)
        or search_action != record.recorded_action_index
    ):
        raise HeadToHeadError("guided search action does not match its public decision.")
    model_action = selection["model_argmax"]
    legal_action_indices = {
        index for index, is_legal in enumerate(record.current_legal_action_mask) if is_legal
    }
    if model_action is not None and (isinstance(model_action, bool) or model_action not in legal_action_indices):
        raise HeadToHeadError("guided model action must be an integer or null.")
    model_override = selection["model_override"]
    if model_action is None:
        if model_override is not None:
            raise HeadToHeadError("unmeasured guided decision cannot claim a model override.")
    elif not isinstance(model_override, bool) or model_override != (model_action != search_action):
        raise HeadToHeadError("guided model override disagrees with the selected actions.")
    unmeasured_cause = selection["unmeasured_cause"]
    if unmeasured_cause is not None and (
        not isinstance(unmeasured_cause, str) or not unmeasured_cause
    ):
        raise HeadToHeadError("guided unmeasured cause must be a non-empty string or null.")
    def optional_number(value: object, *, label: str) -> float | None:
        return None if value is None else _finite_number(value, label=label)

    root = _mapping(selection["root_allocation"], label="guided root allocation")
    if set(root) != {"worlds", "prior_authority", "prior_cause", "arms"}:
        raise HeadToHeadError("guided root allocation has unsupported fields.")
    worlds = _nonnegative_int(root.get("worlds"), label="guided root allocation worlds")
    if worlds <= 0 or not isinstance(root.get("prior_authority"), bool):
        raise HeadToHeadError("guided root allocation has invalid authority metadata.")
    prior_cause = root.get("prior_cause")
    if prior_cause is not None and (not isinstance(prior_cause, str) or not prior_cause):
        raise HeadToHeadError("guided root allocation prior cause must be a non-empty string or null.")
    if bool(root["prior_authority"]) != (prior_cause is None):
        raise HeadToHeadError("guided root allocation authority disagrees with its cause.")
    if unmeasured_cause != prior_cause:
        raise HeadToHeadError(
            "guided unmeasured cause does not match its root allocation cause."
        )
    if unmeasured_cause is None:
        if model_action is None or not isinstance(model_override, bool):
            raise HeadToHeadError(
                "measured guided selection must include a model action and override verdict."
            )
    elif (
        unmeasured_cause not in OVERRIDE_UNMEASURED_CAUSE_VALUES
        or model_action is not None
        or model_override is not None
    ):
        raise HeadToHeadError(
            "unmeasured guided selection has an unsupported cause or measured fields."
        )
    arms = root.get("arms")
    if not isinstance(arms, list) or not arms:
        raise HeadToHeadError("guided root allocation must include at least one own-action arm.")
    normalized_arms: list[dict[str, Any]] = []
    for arm in arms:
        arm = _mapping(arm, label="guided root allocation arm")
        if set(arm) != {
            "action_index",
            "visit_share",
            "q",
            "reported_prior",
            "model_prior",
        }:
            raise HeadToHeadError("guided root allocation arm has unsupported fields.")
        action_index = arm.get("action_index")
        if isinstance(action_index, bool) or action_index not in legal_action_indices:
            raise HeadToHeadError("guided root allocation arm is not a public legal action.")
        visit_share = _finite_number(arm.get("visit_share"), label="guided root visit share")
        if not 0.0 <= visit_share <= 1.0:
            raise HeadToHeadError("guided root visit share must be within [0, 1].")
        q = optional_number(arm.get("q"), label="guided root Q")
        if q is not None and not -1.0 <= q <= 1.0:
            raise HeadToHeadError("guided root Q must be within [-1, 1] when present.")
        normalized_arms.append(
            {
                "action_index": action_index,
                "visit_share": visit_share,
                "q": q,
                "reported_prior": optional_number(
                    arm.get("reported_prior"), label="guided reported prior"
                ),
                "model_prior": optional_number(arm.get("model_prior"), label="guided model prior"),
            }
        )
    if len({arm["action_index"] for arm in normalized_arms}) != len(normalized_arms):
        raise HeadToHeadError("guided root allocation repeats an own-action arm.")
    covered_action_indices = {arm["action_index"] for arm in normalized_arms}
    missing_action_indices = sorted(legal_action_indices - covered_action_indices)
    # The selected search action and, where measured, the model argmax must
    # still be represented by a native root arm.  Other legal public actions
    # can be absent when a belief world cannot render their engine equivalent.
    # That is diagnostic evidence of an engine/request seam, not a reason to
    # discard all prior durable decisions or crash the evaluation after a long
    # run.  The immutable sidecar names every omitted public action explicitly.
    if search_action not in covered_action_indices:
        raise HeadToHeadError("guided search action is absent from its root allocation.")
    if model_action is not None and model_action not in covered_action_indices:
        raise HeadToHeadError("guided model action is absent from its root allocation.")
    supplied_missing = selection.get(coverage_field)
    if supplied_missing is not None:
        if (
            not isinstance(supplied_missing, list)
            or any(isinstance(index, bool) or not isinstance(index, int) for index in supplied_missing)
            or supplied_missing != missing_action_indices
        ):
            raise HeadToHeadError(
                "guided root allocation missing-action coverage disagrees with its public record."
            )
    if not math.isclose(sum(arm["visit_share"] for arm in normalized_arms), 1.0, abs_tol=1e-5):
        raise HeadToHeadError("guided root visits do not conserve one decision.")
    for key in ("reported_prior", "model_prior"):
        values = [arm[key] for arm in normalized_arms]
        if any(value is not None and not 0.0 <= value <= 1.0 for value in values):
            raise HeadToHeadError(f"guided {key} must be within [0, 1] when present.")
        if any(value is not None for value in values):
            if any(value is None for value in values) or not math.isclose(sum(values), 1.0, abs_tol=1e-5):
                raise HeadToHeadError(f"guided {key} must cover and conserve all own-action arms.")
    if bool(root["prior_authority"]) != all(
        arm["model_prior"] is not None for arm in normalized_arms
    ):
        raise HeadToHeadError("guided root model-prior authority disagrees with its arms.")
    root_q_gap = optional_number(selection["root_q_gap"], label="guided root Q gap")
    root_visit_gap = optional_number(selection["root_visit_gap"], label="guided root visit gap")
    gap_actions = selection["root_gap_action_indices"]
    if not isinstance(gap_actions, list) or len(gap_actions) > 2:
        raise HeadToHeadError("guided root gap witness must name zero to two public actions.")
    if (
        any(isinstance(action, bool) or action not in legal_action_indices for action in gap_actions)
        or len(set(gap_actions)) != len(gap_actions)
    ):
        raise HeadToHeadError("guided root gap witness is not a unique public legal action sequence.")
    by_action = {arm["action_index"]: arm for arm in normalized_arms}
    if any(action not in by_action for action in gap_actions):
        raise HeadToHeadError("guided root gap witness is absent from its root allocation.")
    leaders = [by_action[action] for action in gap_actions]
    if any(arm["visit_share"] <= 0.0 for arm in leaders):
        raise HeadToHeadError("guided root gap witness must exclude zero-visit actions.")
    positive_visits = sorted(
        (arm["visit_share"] for arm in normalized_arms if arm["visit_share"] > 0.0),
        reverse=True,
    )
    expected_gap_count = min(2, len(positive_visits))
    if len(leaders) != expected_gap_count or sorted(
        (arm["visit_share"] for arm in leaders), reverse=True
    ) != positive_visits[:expected_gap_count]:
        raise HeadToHeadError("guided root gap witness does not name the leading visited arms.")
    if len(leaders) < 2:
        if root_visit_gap is not None:
            raise HeadToHeadError("guided root visit gap exists without two leading arms.")
    elif root_visit_gap is None or not 0.0 <= root_visit_gap <= 1.0 or not math.isclose(
        root_visit_gap, leaders[0]["visit_share"] - leaders[1]["visit_share"], abs_tol=1e-5
    ):
        raise HeadToHeadError("guided root visit gap disagrees with its allocation.")
    expected_q_gap = (
        abs(leaders[0]["q"] - leaders[1]["q"])
        if len(leaders) == 2 and leaders[0]["q"] is not None and leaders[1]["q"] is not None
        else None
    )
    if expected_q_gap is None:
        if root_q_gap is not None:
            raise HeadToHeadError("guided root Q gap exists without two valued leading arms.")
    elif root_q_gap is None or not math.isclose(root_q_gap, expected_q_gap, abs_tol=1e-5):
        raise HeadToHeadError("guided root Q gap disagrees with its allocation.")
    return {
        "model_argmax": model_action,
        "search_argmax": search_action,
        "model_override": model_override,
        "unmeasured_cause": unmeasured_cause,
        "root_q_gap": root_q_gap,
        "root_visit_gap": root_visit_gap,
        "root_gap_action_indices": list(gap_actions),
        "root_allocation_missing_action_indices": missing_action_indices,
        "root_allocation": {
            "worlds": worlds,
            "prior_authority": bool(root["prior_authority"]),
            "prior_cause": prior_cause,
            "arms": normalized_arms,
        },
    }


def _selection_evidence_from_override(
    override: Mapping[str, Any], *, record: PublicDecisionRecord
) -> dict[str, Any]:
    """Sanitise engine telemetry into public-only, action-indexed evidence."""

    expected = {
        "model_argmax",
        "search_argmax",
        "model_override",
        "unmeasured_cause",
        "root_q_gap",
        "root_visit_gap",
        "root_gap_action_indices",
        "root_allocation",
    }
    if not expected.issubset(override):
        raise HeadToHeadError("guided override telemetry does not expose complete selection evidence.")
    raw_root = _mapping(override["root_allocation"], label="guided engine root allocation")
    if set(raw_root) != {"worlds", "prior_authority", "prior_cause", "arms"}:
        raise HeadToHeadError("guided engine root allocation has unsupported fields.")
    raw_arms = raw_root.get("arms")
    if not isinstance(raw_arms, list):
        raise HeadToHeadError("guided engine root allocation arms must be a list.")
    public_arms = []
    for arm in raw_arms:
        arm = _mapping(arm, label="guided engine root allocation arm")
        if set(arm) != {
            "move",
            "action_index",
            "visit_share",
            "q",
            "reported_prior",
            "model_prior",
        }:
            raise HeadToHeadError("guided engine root allocation arm has unsupported fields.")
        # The public action index is the complete identity.  Never carry an
        # engine-rendered label (such as typed Hidden Power) beside a public row.
        public_arms.append({key: arm[key] for key in arm if key != "move"})
    return _validated_selection_evidence(
        {
            key: (dict(raw_root, arms=public_arms) if key == "root_allocation" else override[key])
            for key in expected
        },
        record=record,
    )


def _branch_prior_ledger_payload(
    *,
    candidate: MctsPolicySpec,
    incumbent: MctsPolicySpec,
    candidate_seat: str,
    record: PublicDecisionRecord,
    requested_players: tuple[str, ...],
    ledger: Mapping[str, Any],
    selection: Mapping[str, Any],
) -> dict[str, Any]:
    if requested_players not in {(candidate_seat,), ("p1", "p2")}:
        raise HeadToHeadError("branch-prior ledger has an unsupported request boundary.")
    return {
        "schema_version": BRANCH_PRIOR_LEDGER_EVIDENCE_SCHEMA_VERSION,
        "seed": record.seed,
        "candidate_seat": candidate_seat,
        "candidate_provenance_sha256": candidate.provenance_sha256,
        "raw_provenance_sha256": incumbent.provenance_sha256,
        "public_decision": {
            "decision_id": record.decision_id,
            "battle_id": record.battle_id,
            "acting_player": record.acting_player,
            "turn_index": record.turn_index,
            "recorded_action_index": record.recorded_action_index,
        },
        "request_boundary": {"requested_players": list(requested_players)},
        "branch_prior_fallbacks": _validated_branch_prior_ledger(ledger),
        "selection": _validated_selection_evidence(selection, record=record),
    }


def _public_decision_payload(
    *,
    candidate: MctsPolicySpec,
    incumbent: MctsPolicySpec,
    candidate_seat: str,
    record: PublicDecisionRecord,
) -> dict[str, Any]:
    return {
        "schema_version": PUBLIC_DECISION_EVIDENCE_SCHEMA_VERSION,
        "seed": record.seed,
        "candidate_seat": candidate_seat,
        "candidate_provenance_sha256": candidate.provenance_sha256,
        "raw_provenance_sha256": incumbent.provenance_sha256,
        "record": record.to_dict(),
    }


def _public_decision_writer(
    out_root: Path,
    *,
    candidate: MctsPolicySpec,
    incumbent: MctsPolicySpec,
    seed: int,
    candidate_seat: str,
    guided_policy: PublicOnlyMctsPolicy,
):
    """Persist guided decisions as individually immutable public replay units."""

    def write(record: PublicDecisionRecord) -> None:
        # The rollout hook reports both actors. Only the guided actor is in
        # scope for the override audit; retaining raw's private decision view
        # would add storage without adding a search hypothesis.
        if record.acting_player != candidate_seat:
            return
        path = _public_decision_path(
            out_root, seed=seed, candidate_seat=candidate_seat, record=record
        )
        override, requested_players = _guided_override_from_decision(guided_policy, record)
        _write_immutable_json(
            path,
            _public_decision_payload(
                candidate=candidate,
                incumbent=incumbent,
                candidate_seat=candidate_seat,
                record=record,
            ),
        )
        ledger_path = _branch_prior_ledger_path(
            out_root, seed=seed, candidate_seat=candidate_seat, record=record
        )
        _write_immutable_json(
            ledger_path,
            _branch_prior_ledger_payload(
                candidate=candidate,
                incumbent=incumbent,
                candidate_seat=candidate_seat,
                record=record,
                requested_players=requested_players,
                ledger=_branch_prior_ledger_from_override(override),
                selection=_selection_evidence_from_override(override, record=record),
            ),
        )

    return write


def _sealed_override_audit_writer(
    out_root: Path,
    *,
    candidate: MctsPolicySpec,
    incumbent: MctsPolicySpec,
    candidate_seat: str,
    env_factory: Any,
    continuation_policy_factory: Any,
    continuation_rollout_config: Any,
    max_continuation_decision_rounds: int,
):
    """Evaluate and durably retain every measured override at its source boundary.

    ``RolloutDriver`` calls this trusted hook after selecting the simultaneous
    actions but before committing them.  The controller consumes the
    actionable snapshot synchronously and persists only an immutable terminal
    readout.  It never writes the snapshot, opponent action, or observation.
    """

    from pokezero.mcts_eval.sealed_override_audit import (  # noqa: PLC0415
        evaluate_measured_override_boundary,
    )

    def bind_public_root_coverage(readout: Mapping[str, Any], *, boundary: Any) -> dict[str, Any]:
        """Bind the controller's sealed evidence to the already-written public ledger.

        The sealed controller runs at the actionable pre-step boundary and
        deliberately has no public decision record from which to derive the
        native-root coverage vector.  The public-decision hook has already
        atomically written that record and its validated branch ledger.  Copy
        only the derived action-index coverage into the controller readout,
        after proving every controller-provided field agrees with that ledger.
        This keeps an incomplete native root visible without allowing the
        sealed artifact to claim complete action coverage.
        """

        seed = _nonnegative_int(getattr(boundary, "seed", None), label="sealed source seed")
        round_index = _nonnegative_int(
            getattr(boundary, "decision_round_index", None), label="sealed source decision round"
        )
        ledger_root = out_root / "branch-prior-fallback-ledgers" / f"seed-{seed}-{candidate_seat}"
        ledgers = sorted(ledger_root.glob(f"turn-{round_index:03d}-*.json"))
        if len(ledgers) != 1:
            raise HeadToHeadError(
                "sealed override audit requires exactly one public branch ledger at its source round."
            )
        try:
            ledger_payload = json.loads(ledgers[0].read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise HeadToHeadError(f"cannot read sealed audit source ledger: {error}") from error
        ledger = _mapping(ledger_payload, label="sealed audit source ledger")
        selection = _mapping(ledger.get("selection"), label="sealed audit source selection")
        expected_evidence = _sealed_search_evidence_from_selection(selection)
        expected_controller_evidence = dict(expected_evidence)
        expected_controller_evidence.pop("root_allocation_missing_action_indices")

        copied = dict(readout)
        if copied.get("audit_status") == "PAIRED":
            audit = _mapping(copied.get("audit"), label="sealed override pair")
            if audit.get("search_evidence") != expected_controller_evidence:
                raise HeadToHeadError(
                    "sealed override audit controller evidence disagrees with its public source ledger."
                )
            copied["audit"] = {**audit, "search_evidence": expected_evidence}
        elif copied.get("audit_status") == "INAPPLICABLE_NON_SIMULTANEOUS":
            if copied.get("search_evidence") != expected_controller_evidence:
                raise HeadToHeadError(
                    "inapplicable sealed audit controller evidence disagrees with its public source ledger."
                )
            copied["search_evidence"] = expected_evidence
        else:
            raise HeadToHeadError("sealed override audit has an unsupported disposition.")
        return copied

    def write(boundary: Any) -> None:
        readout = evaluate_measured_override_boundary(
            boundary=boundary,
            candidate_seat=candidate_seat,
            env_factory=env_factory,
            continuation_policy_factory=continuation_policy_factory,
            rollout_config=continuation_rollout_config,
            max_continuation_decision_rounds=max_continuation_decision_rounds,
        )
        if readout is None:
            return
        readout = bind_public_root_coverage(readout, boundary=boundary)
        path = _sealed_override_audit_path(
            out_root,
            seed=boundary.seed,
            candidate_seat=candidate_seat,
            decision_round_index=boundary.decision_round_index,
        )
        _write_immutable_json(
            path,
            _sealed_override_audit_payload(
                candidate=candidate,
                incumbent=incumbent,
                candidate_seat=candidate_seat,
                readout=readout,
            ),
        )

    return write


def _raw_witness_payload(game: Any, *, raw_forward_decisions: int) -> dict[str, Any]:
    raw_decisions = game.incumbent_telemetry.decisions
    if raw_forward_decisions != raw_decisions:
        raise HeadToHeadError(
            "raw selector forward count must exactly equal the completed game's raw decision count."
        )
    return {
        "schema_version": "pokezero.mcts-guided-vs-raw-selector-witness.v1",
        "seed": game.seed,
        "candidate_seat": game.candidate_seat,
        "raw_provenance_sha256": game.incumbent.provenance_sha256,
        "raw_policy_id": game.incumbent.policy_id,
        "raw_selector": RAW_SELECTOR,
        "raw_forward_decisions": raw_forward_decisions,
        "raw_telemetry_decisions": raw_decisions,
    }


def _validate_raw_witness(out_root: Path, game: Any) -> Mapping[str, Any]:
    path = _raw_witness_path(out_root, seed=game.seed, candidate_seat=game.candidate_seat)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise HeadToHeadError(f"completed game is missing its readable raw selector witness: {error}") from error
    if not isinstance(payload, Mapping):
        raise HeadToHeadError("raw selector witness is not a JSON object.")
    expected = _raw_witness_payload(
        game, raw_forward_decisions=game.incumbent_telemetry.decisions
    )
    if dict(payload) != expected:
        raise HeadToHeadError("raw selector witness differs from its immutable completed game.")
    return payload


def _validate_public_decision_evidence(out_root: Path, game: Any) -> tuple[PublicDecisionRecord, ...]:
    """Require a complete, source-bound public replay unit for every guided action."""

    root = out_root / "public-decision-records" / f"seed-{game.seed}-{game.candidate_seat}"
    if not root.is_dir():
        raise HeadToHeadError("completed game is missing its public decision evidence directory.")
    records: list[PublicDecisionRecord] = []
    expected_battle_id = f"mcts-h2h-{game.seed}-{game.candidate_seat}"
    for path in sorted(root.glob("turn-*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise HeadToHeadError(f"cannot read public decision evidence {path}: {error}") from error
        if not isinstance(payload, Mapping):
            raise HeadToHeadError("public decision evidence is not a JSON object.")
        if (
            payload.get("schema_version") != PUBLIC_DECISION_EVIDENCE_SCHEMA_VERSION
            or payload.get("seed") != game.seed
            or payload.get("candidate_seat") != game.candidate_seat
            or payload.get("candidate_provenance_sha256") != game.candidate.provenance_sha256
            or payload.get("raw_provenance_sha256") != game.incumbent.provenance_sha256
        ):
            raise HeadToHeadError("public decision evidence does not match its completed game.")
        try:
            record = PublicDecisionRecord.from_dict(_mapping(payload.get("record"), label="public record"))
        except (TypeError, ValueError) as error:
            raise HeadToHeadError(f"public decision evidence has an invalid record: {error}") from error
        expected_path = _public_decision_path(
            out_root,
            seed=game.seed,
            candidate_seat=game.candidate_seat,
            record=record,
        )
        if path != expected_path:
            raise HeadToHeadError("public decision evidence path does not match its canonical record identity.")
        if record.battle_id != expected_battle_id or record.format_id != "gen3randombattle":
            raise HeadToHeadError("public decision evidence does not bind the completed game identity.")
        ledger_path = _branch_prior_ledger_path(
            out_root,
            seed=game.seed,
            candidate_seat=game.candidate_seat,
            record=record,
        )
        try:
            ledger_payload = json.loads(ledger_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise HeadToHeadError(
                f"public decision evidence is missing its readable branch-prior ledger: {error}"
            ) from error
        if not isinstance(ledger_payload, Mapping):
            raise HeadToHeadError("branch-prior ledger evidence is not a JSON object.")
        ledger = _validated_branch_prior_ledger(ledger_payload.get("branch_prior_fallbacks"))
        selection = _validated_selection_evidence(
            _mapping(ledger_payload.get("selection"), label="guided selection evidence"),
            record=record,
        )
        request_boundary = _mapping(
            ledger_payload.get("request_boundary"), label="branch-prior request boundary"
        )
        requested_players = request_boundary.get("requested_players")
        if (
            not isinstance(requested_players, list)
            or tuple(requested_players) not in {(game.candidate_seat,), ("p1", "p2")}
        ):
            raise HeadToHeadError("branch-prior ledger has an invalid request boundary.")
        expected_ledger_payload = _branch_prior_ledger_payload(
            candidate=game.candidate,
            incumbent=game.incumbent,
            candidate_seat=game.candidate_seat,
            record=record,
            requested_players=tuple(requested_players),
            ledger=ledger,
            selection=selection,
        )
        if dict(ledger_payload) != expected_ledger_payload:
            raise HeadToHeadError("branch-prior ledger evidence does not bind its public decision.")
        records.append(record)
    expected_count = game.candidate_telemetry.decisions
    if expected_count <= 0 or len(records) != expected_count:
        raise HeadToHeadError(
            "public decision evidence count must be positive and equal guided decision telemetry."
        )
    decision_ids = {record.decision_id for record in records}
    turn_indices = {record.turn_index for record in records}
    if len(decision_ids) != len(records) or len(turn_indices) != len(records):
        raise HeadToHeadError("public decision evidence contains duplicate guided decision identities.")
    return tuple(records)


def _validate_sealed_override_audit_evidence(out_root: Path, game: Any) -> None:
    """Require a one-to-one sealed terminal readout for measured public overrides."""

    expected_count = game.candidate_telemetry.model_override_decisions
    if isinstance(expected_count, bool) or not isinstance(expected_count, int) or expected_count < 0:
        raise HeadToHeadError("guided telemetry has an invalid measured-override count.")
    # The public decision and native-ledger files are already independently
    # checked by this runner.  They are the source denominator: sidecars may
    # not create, substitute, or silently omit a measured override merely by
    # preserving the aggregate telemetry count.
    records = _validate_public_decision_evidence(out_root, game)
    expected: dict[int, dict[str, Any]] = {}
    for record in records:
        ledger_path = _branch_prior_ledger_path(
            out_root, seed=game.seed, candidate_seat=game.candidate_seat, record=record
        )
        try:
            ledger_payload = json.loads(ledger_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise HeadToHeadError(
                f"cannot read branch-prior ledger needed by sealed override audit: {error}"
            ) from error
        selection = _validated_selection_evidence(
            _mapping(ledger_payload.get("selection"), label="sealed audit source selection"),
            record=record,
        )
        request_boundary = _mapping(
            ledger_payload.get("request_boundary"), label="sealed audit source request boundary"
        )
        requested_players = request_boundary.get("requested_players")
        if (
            not isinstance(requested_players, list)
            or tuple(requested_players) not in {(game.candidate_seat,), ("p1", "p2")}
        ):
            raise HeadToHeadError("sealed audit source ledger has an invalid request boundary.")
        if selection["model_override"] is True:
            if record.turn_index in expected:
                raise HeadToHeadError("measured override source decisions repeat a round identity.")
            expected[record.turn_index] = {
                "selection": selection,
                "requested_players": list(requested_players),
            }
    if len(expected) != expected_count:
        raise HeadToHeadError(
            "guided measured-override telemetry disagrees with its public decision ledger."
        )
    root = out_root / "sealed-override-audits" / f"seed-{game.seed}-{game.candidate_seat}"
    paths = sorted(root.glob("round-*.json")) if root.is_dir() else []
    if len(paths) != len(expected):
        raise HeadToHeadError("sealed override audit count must equal measured public overrides.")
    expected_battle_id = f"mcts-h2h-{game.seed}-{game.candidate_seat}"
    rounds: set[int] = set()
    for path in paths:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise HeadToHeadError(f"cannot read sealed override audit {path}: {error}") from error
        if not isinstance(payload, Mapping) or set(payload) != {
            "schema_version",
            "candidate_provenance_sha256",
            "raw_provenance_sha256",
            "candidate_seat",
            "readout",
        }:
            raise HeadToHeadError("sealed override audit has an unsupported wrapper shape.")
        if (
            payload.get("schema_version") != SEALED_OVERRIDE_AUDIT_EVIDENCE_SCHEMA_VERSION
            or payload.get("candidate_provenance_sha256") != game.candidate.provenance_sha256
            or payload.get("raw_provenance_sha256") != game.incumbent.provenance_sha256
            or payload.get("candidate_seat") != game.candidate_seat
        ):
            raise HeadToHeadError("sealed override audit does not bind the completed game.")
        readout = _validated_sealed_override_readout(
            payload.get("readout"), candidate_seat=game.candidate_seat
        )
        if readout["seed"] != game.seed or readout["battle_id"] != expected_battle_id:
            raise HeadToHeadError("sealed override audit readout does not bind its source decision.")
        round_index = readout.get("decision_round_index")
        expected_path = _sealed_override_audit_path(
            out_root,
            seed=game.seed,
            candidate_seat=game.candidate_seat,
            decision_round_index=round_index,
        )
        if path != expected_path or round_index in rounds or round_index not in expected:
            raise HeadToHeadError("sealed override audit has a duplicate or noncanonical round identity.")
        rounds.add(round_index)
        source = expected[round_index]
        selection = source["selection"]
        expected_evidence = _sealed_search_evidence_from_selection(selection)
        if readout["audit_status"] == "PAIRED":
            audit = _mapping(readout.get("audit"), label="sealed override pair")
            if (
                audit["mcts_action"] != selection["search_argmax"]
                or audit["raw_action"] != selection["model_argmax"]
                or audit["search_evidence"] != expected_evidence
            ):
                raise HeadToHeadError(
                    "sealed override audit does not bind its measured public decision selection."
                )
        elif readout["audit_status"] == "INAPPLICABLE_NON_SIMULTANEOUS":
            if (
                source["requested_players"] != [game.candidate_seat]
                or readout["requested_players"] != source["requested_players"]
                or readout["search_evidence"] != expected_evidence
            ):
                raise HeadToHeadError(
                    "inapplicable sealed override audit does not bind its measured public decision boundary."
                )
        else:  # _validated_sealed_override_readout already rejects this; keep closed on drift.
            raise HeadToHeadError("sealed override audit has an unknown disposition.")
    if rounds != set(expected):
        raise HeadToHeadError("sealed override audit does not cover every measured public override.")


def _validate_completed_game(game: Any) -> None:
    """Reject a persisted game whose live decision evidence is not admissible."""

    candidate = game.candidate_telemetry
    raw = game.incumbent_telemetry
    if candidate.fallback_decisions or raw.fallback_decisions:
        raise HeadToHeadError("a guided-vs-raw game recorded an action fallback.")
    if candidate.root_prior_fallbacks or raw.root_prior_fallbacks:
        raise HeadToHeadError("a guided-vs-raw game recorded a root policy-prior fallback.")
    if (
        raw.searched_decisions
        or raw.model_evals
        or raw.total_iterations
        or raw.worlds_constructed
        or raw.worlds_searched
    ):
        raise HeadToHeadError("raw-policy baseline recorded search work.")


def _validate_summary_evidence(summary: Mapping[str, Any]) -> None:
    """Require aggregate live-root and raw-no-search witnesses before completion."""

    if summary.get("candidate_root_prior_fallbacks") != 0:
        raise HeadToHeadError("guided candidate accumulated a root policy-prior fallback.")
    if summary.get("incumbent_root_prior_fallbacks") != 0:
        raise HeadToHeadError("raw baseline accumulated a root policy-prior fallback.")
    if not isinstance(summary.get("candidate_override_measured_decisions"), int) or summary[
        "candidate_override_measured_decisions"
    ] <= 0:
        raise HeadToHeadError(
            "guided candidate never exposed a live root model-action witness; result is nonbankable."
        )
    if summary.get("incumbent_model_evals") != 0 or summary.get("incumbent_iterations") != 0:
        raise HeadToHeadError("raw baseline summary contains search work.")


def _progress_writer(out_root: Path, *, candidate: MctsPolicySpec, incumbent: MctsPolicySpec):
    def write(event: str, *, seed: int, candidate_seat: str, decision: Any | None = None) -> None:
        payload: dict[str, Any] = {
            "schema_version": PROGRESS_SCHEMA_VERSION,
            "event": event,
            "seed": seed,
            "candidate_seat": candidate_seat,
            "candidate_provenance_sha256": candidate.provenance_sha256,
            "incumbent_provenance_sha256": incumbent.provenance_sha256,
        }
        if decision is not None:
            payload.update(
                {
                    "battle_id": decision.battle_id,
                    "decision_round_index": decision.decision_round_index,
                    "decision_round_count": decision.decision_round_count,
                    "requested_players": list(decision.requested_players),
                    "terminal": decision.terminal,
                    "terminal_capped": decision.terminal_capped,
                    "terminal_winner": decision.terminal_winner,
                }
            )
        _write_progress_json(
            out_root / "progress" / "current.json",
            payload,
            schema_version=PROGRESS_SCHEMA_VERSION,
        )

    return write


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--showdown-root", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--device", default="cpu")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    out_root = _durable_output_root(args.out_dir)
    _require_durable_launcher_handoff(out_root, runner_script=Path(__file__))
    manifest = _load_manifest(args.manifest)
    seeds = _seeds(manifest)
    study = _validated_study(manifest, seeds=seeds)
    sealed_override_audit = _sealed_override_audit_config(manifest)
    resamples, bootstrap_seed, confidence_level = _bootstrap(manifest)
    max_decision_rounds = manifest.get("max_decision_rounds")
    if isinstance(max_decision_rounds, bool) or not isinstance(max_decision_rounds, int) or max_decision_rounds <= 0:
        raise HeadToHeadError("manifest.max_decision_rounds must be a positive integer.")

    from engine_build_fingerprint import assert_fresh, compute_fingerprint  # noqa: PLC0415
    from pokezero.dex import load_showdown_dex_cached  # noqa: PLC0415
    from pokezero.engine_search import EngineMctsPolicy, EnvTier2AnnotationSource  # noqa: PLC0415
    from pokezero.local_showdown import (  # noqa: PLC0415
        LocalShowdownConfig,
        LocalShowdownEnv,
        env_config_from_checkpoint_provenance,
    )
    from pokezero.mcts_eval.lattice import materialize_search_artifacts  # noqa: PLC0415
    from pokezero.mcts_eval.resolver import resolve_checkpoint_contract  # noqa: PLC0415
    from pokezero.neural_policy import (  # noqa: PLC0415
        category_vocab_from_model_config,
        feature_masks_from_model_config,
        load_transformer_model_config,
        load_transformer_policy,
        observation_spec_from_model_config,
    )
    from pokezero.randbat import load_gen3_randbat_source_cached  # noqa: PLC0415
    from pokezero.rollout import RolloutConfig, RolloutDriver  # noqa: PLC0415

    source = _source_provenance()
    source_commit = source["commit"]
    source_tree_sha256 = source["tree_sha256"]
    if _hex(manifest.get("source_tree_sha256"), label="manifest.source_tree_sha256", length=64) != source_tree_sha256:
        raise HeadToHeadError("manifest source-tree identity does not match the executing image.")
    checkpoint_sha256 = _sha256_file(args.checkpoint)
    if _hex(manifest.get("checkpoint_sha256"), label="manifest.checkpoint_sha256", length=64) != checkpoint_sha256:
        raise HeadToHeadError("manifest checkpoint identity does not match --checkpoint.")
    showdown = _showdown_source_provenance(args.showdown_root)
    showdown_source_sha256 = str(showdown["content_sha256"])
    if _hex(manifest.get("showdown_source_sha256"), label="manifest.showdown_source_sha256", length=64) != showdown_source_sha256:
        raise HeadToHeadError("manifest Showdown identity does not match --showdown-root.")
    assert_fresh()
    engine_fingerprint = str(compute_fingerprint()["fingerprint"])

    contract = resolve_checkpoint_contract(
        args.checkpoint,
        model_device=args.device,
        showdown_root=args.showdown_root,
        showdown_source_sha256=showdown_source_sha256,
        expected_showdown_source_sha256=showdown_source_sha256,
    )
    artifacts = materialize_search_artifacts(contract, showdown_root=args.showdown_root)
    candidate_raw = _mapping(manifest.get("candidate"), label="manifest.candidate")
    candidate, candidate_config = _runtime_spec(
        candidate_raw,
        role="guided candidate",
        checkpoint=args.checkpoint,
        checkpoint_sha256=checkpoint_sha256,
        source_commit=source_commit,
        source_tree_sha256=source_tree_sha256,
        engine_fingerprint=engine_fingerprint,
        showdown_source_sha256=showdown_source_sha256,
        model_path=artifacts["model_path"],
        tables_path=artifacts["tables_path"],
        device=args.device,
    )
    incumbent = _raw_spec(
        _mapping(manifest.get("raw"), label="manifest.raw"),
        checkpoint_sha256=checkpoint_sha256,
        source_commit=source_commit,
        source_tree_sha256=source_tree_sha256,
        engine_fingerprint=engine_fingerprint,
        showdown_source_sha256=showdown_source_sha256,
    )
    _require_registered_candidate_config(candidate.config)

    model_config = load_transformer_model_config(args.checkpoint)
    vocabulary = category_vocab_from_model_config(model_config, args.showdown_root)
    env_config = env_config_from_checkpoint_provenance(
        LocalShowdownConfig(
            showdown_root=args.showdown_root,
            set_belief_source=True,
            category_vocab=vocabulary,
        ),
        feature_masks_from_model_config(model_config),
        required_specs=observation_spec_from_model_config(model_config),
        required_vocabs=vocabulary,
        context="guided-MCTS-versus-raw-policy paired runner",
    )
    dex = load_showdown_dex_cached(args.showdown_root)
    set_source = load_gen3_randbat_source_cached(args.showdown_root)
    write_progress = _progress_writer(out_root, candidate=candidate, incumbent=incumbent)

    def continuation_policy_factory() -> Mapping[str, Any]:
        """Allocate two clean raw policies after a fixed source action.

        Both continuation arms use this factory, so the only variable between
        them is the source MCTS-versus-raw action under audit.  Loading fresh
        policies is intentional: policy state, timing, and cached request data
        from either arm must not flow into the other one.
        """

        def raw_policy() -> PublicOnlyMctsPolicy:
            adapter = DeterministicRawPolicyAdapter(
                load_transformer_policy(
                    args.checkpoint,
                    device=args.device,
                    deterministic=True,
                    exploration_epsilon=0.0,
                    sampling_temperature=1.0,
                    family_gated_selection=False,
                ),
                policy_id=incumbent.policy_id,
            )
            return PublicOnlyMctsPolicy(adapter)

        return {"p1": raw_policy(), "p2": raw_policy()}

    continuation_rollout_config = RolloutConfig(
        max_decision_rounds=max_decision_rounds,
        format_id="gen3randombattle",
        record_policy_timing=False,
        hide_opponent_legal_action_masks=True,
    )

    _write_immutable_json(
        out_root / "manifest.json",
        {
            "schema_version": MANIFEST_SCHEMA_VERSION,
            "declared_manifest": manifest,
            "active_source": source,
            "active_engine_fingerprint": engine_fingerprint,
            "active_checkpoint_sha256": checkpoint_sha256,
            "active_showdown_source": showdown,
            "candidate": candidate.to_payload(),
            "raw": incumbent.to_payload(),
            "study": study,
            "seeds": list(seeds),
        },
    )

    raw_adapters: dict[tuple[int, str], DeterministicRawPolicyAdapter] = {}

    def session_factory(seed: int, candidate_seat: str):
        write_progress("game_started", seed=seed, candidate_seat=candidate_seat)
        env = LocalShowdownEnv(env_config)
        annotations = EnvTier2AnnotationSource(env)
        guided = PublicOnlyMctsPolicy(
            EngineMctsPolicy(
                dex=dex,
                set_source=set_source,
                config=candidate_config,
                policy_id=candidate.policy_id,
                annotation_source=annotations,
            )
        )
        raw_adapter = DeterministicRawPolicyAdapter(
            load_transformer_policy(
                args.checkpoint,
                device=args.device,
                deterministic=True,
                exploration_epsilon=0.0,
                sampling_temperature=1.0,
                family_gated_selection=False,
            ),
            policy_id=incumbent.policy_id,
        )
        raw_adapters[(seed, candidate_seat)] = raw_adapter
        raw = PublicOnlyMctsPolicy(raw_adapter)
        other_seat = "p2" if candidate_seat == "p1" else "p1"
        sealed_sink = None
        if sealed_override_audit is not None:
            sealed_sink = _sealed_override_audit_writer(
                out_root,
                candidate=candidate,
                incumbent=incumbent,
                candidate_seat=candidate_seat,
                env_factory=lambda: LocalShowdownEnv(env_config),
                continuation_policy_factory=continuation_policy_factory,
                continuation_rollout_config=continuation_rollout_config,
                max_continuation_decision_rounds=(
                    sealed_override_audit.max_continuation_decision_rounds
                ),
            )
        driver = RolloutDriver(
            env=env,
            policies={candidate_seat: guided, other_seat: raw},
            config=RolloutConfig(
                max_decision_rounds=max_decision_rounds,
                format_id="gen3randombattle",
                record_policy_timing=True,
                hide_opponent_legal_action_masks=True,
                public_decision_sink=_public_decision_writer(
                    out_root,
                    candidate=candidate,
                    incumbent=incumbent,
                    seed=seed,
                    candidate_seat=candidate_seat,
                    guided_policy=guided,
                ),
                decision_sink=lambda decision: write_progress(
                    "decision_committed", seed=seed, candidate_seat=candidate_seat, decision=decision
                ),
                sealed_pre_step_sink=sealed_sink,
            ),
        )
        return driver, guided, raw

    all_games = []
    for seed in seeds:
        completed = load_pair(out_root, seed=seed, candidate=candidate, incumbent=incumbent)
        for game in completed.values():
            _validate_completed_game(game)
            _validate_raw_witness(out_root, game)
            _validate_public_decision_evidence(out_root, game)
            if sealed_override_audit is not None:
                _validate_sealed_override_audit_evidence(out_root, game)

        def on_game(game: Any) -> None:
            _validate_completed_game(game)
            _validate_public_decision_evidence(out_root, game)
            if sealed_override_audit is not None:
                _validate_sealed_override_audit_evidence(out_root, game)
            raw_adapter = raw_adapters.pop((game.seed, game.candidate_seat), None)
            if raw_adapter is None:
                raise HeadToHeadError("completed game has no retained raw-policy selector witness.")
            _write_immutable_json(
                _raw_witness_path(
                    out_root, seed=game.seed, candidate_seat=game.candidate_seat
                ),
                _raw_witness_payload(
                    game, raw_forward_decisions=raw_adapter.stats.raw_forward_decisions
                ),
            )
            write_game_immutable(out_root, game)
            write_progress("game_persisted", seed=game.seed, candidate_seat=game.candidate_seat)

        games = play_mirrored_pair(
            seed=seed,
            candidate=candidate,
            incumbent=incumbent,
            candidate_factory=lambda: None,
            incumbent_factory=lambda: None,
            driver_factory=lambda *_: None,
            completed=completed,
            session_factory=session_factory,
            on_game=on_game,
        )
        complete_pair(games, seed=seed, candidate=candidate, incumbent=incumbent)
        for game in games:
            _validate_completed_game(game)
            _validate_raw_witness(out_root, game)
            _validate_public_decision_evidence(out_root, game)
            if sealed_override_audit is not None:
                _validate_sealed_override_audit_evidence(out_root, game)
        all_games.extend(games)
        write_progress("pair_completed", seed=seed, candidate_seat="both")
        print(f"completed guided-vs-raw mirrored pair seed={seed}", flush=True)

    summary = summarize_complete_pairs(
        all_games,
        seeds=seeds,
        candidate=candidate,
        incumbent=incumbent,
        bootstrap_resamples=resamples,
        bootstrap_seed=bootstrap_seed,
        bootstrap_confidence_level=confidence_level,
    )
    _validate_summary_evidence(summary)
    raw_forward_decisions = sum(
        int(_validate_raw_witness(out_root, game)["raw_forward_decisions"])
        for game in all_games
    )
    raw_telemetry_decisions = sum(game.incumbent_telemetry.decisions for game in all_games)
    if raw_forward_decisions <= 0 or raw_forward_decisions != raw_telemetry_decisions:
        raise HeadToHeadError(
            "aggregate raw forward evidence must be positive and equal raw decision telemetry."
        )
    _write_immutable_json(out_root / "summary.json", summary)
    complete = {
        "schema_version": COMPLETE_SCHEMA_VERSION,
        "status": "COMPLETE",
        "pairs": len(seeds),
        "games": len(all_games),
        "candidate_provenance_sha256": candidate.provenance_sha256,
        "raw_provenance_sha256": incumbent.provenance_sha256,
        "summary_sha256": _sha256_bytes((json.dumps(summary, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")),
        "raw_selector": RAW_SELECTOR,
    }
    _write_immutable_json(out_root / "COMPLETE.json", complete)
    print(json.dumps(complete, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
