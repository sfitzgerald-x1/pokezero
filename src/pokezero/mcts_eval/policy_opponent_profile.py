"""Three-arm source-root measurement, not a playing-strength evaluation.

All arms share the live replay and full decision boundary. Prefix preparation,
model loading, and export are outside that boundary; observation construction,
search/inference, selection, and Showdown serialization are inside it. This
module neither submits work nor changes any durable source record.
"""

from __future__ import annotations

from dataclasses import replace
import hashlib
import json
import math
import random
import time
from types import SimpleNamespace
from typing import Any, Callable, Mapping, Sequence
import warnings

from .lattice import _LiveEngineTimingDecider
from .manifest import SearchConfig
from .resolver import CheckpointContract, ContractError, sha256_file
from .source_root_replay import SourceRootReplayError, source_bound_replay_prefix


ARMS = ("raw_policy", "incumbent_mcts", "own_policy_opponent_mcts")
MODES = ("fixed_work", "matched_deadline")
SEED_DOMAIN = b"pokezero.paper-policy-opponent-profile.rng.v1\0"


def refusal_diagnostic(error: BaseException) -> dict[str, Any] | None:
    """Retain a native refusal witness across the engine's exception wrapper.

    Diagnostics never enter policy context or alter acceptance. Search failures
    still refuse; this records the failing node instead of losing its evidence.
    """
    return _refusal_diagnostic(error, (
        ("policy_opponent_diagnostic", "policy-opponent-refusal-v1"),
        ("raw_policy_leaf_diagnostic", "raw-policy-terminal-refusal-v1"),
    ))


def fallback_refusal_diagnostic(error: BaseException) -> dict[str, Any] | None:
    """Extract strict engine refusal evidence separately from native errors."""
    return _refusal_diagnostic(error, (
        ("engine_search_fallback_diagnostic", "engine-search-fallback-refusal-v1"),
    ))


def _refusal_diagnostic(error: BaseException, attributes: Sequence[tuple[str, str]]) -> dict[str, Any] | None:
    seen: set[int] = set()
    while error is not None and id(error) not in seen:
        seen.add(id(error))
        for attribute, schema in attributes:
            raw = getattr(error, attribute, None)
            if isinstance(raw, str):
                try:
                    payload = json.loads(raw)
                except (TypeError, ValueError):
                    payload = None
                if (isinstance(payload, dict) and payload.get("schema") == schema
                        and payload.get("diagnostic_only_not_policy_input") is True):
                    return payload
        error = error.__cause__
    return None


def _unsigned_seed(value: Any) -> int:
    if type(value) is not int or not 0 <= value < 2**64:
        raise ContractError("profile requires an explicit unsigned 64-bit seed")
    return value


def root_seed(seed: int, decision_id: str, *, purpose: str) -> int:
    """Registered per-root domains; neither arm nor timing mode changes them."""
    _unsigned_seed(seed)
    if (not isinstance(decision_id, str) or len(decision_id) != 64
            or any(char not in "0123456789abcdef" for char in decision_id)
            or purpose not in ("decision", "opponent")):
        raise ContractError("profile RNG identity/domain is invalid")
    payload = SEED_DOMAIN + seed.to_bytes(8, "big") + purpose.encode() + b"\0" + decision_id.encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def _probabilities(values: Any, mask: Sequence[bool]) -> tuple[float, ...]:
    if (not isinstance(values, (list, tuple)) or len(values) != len(mask)
            or not any(mask) or any(type(value) not in (int, float)
                                   or not math.isfinite(value) or value < 0 for value in values)):
        raise ContractError("raw policy returned an invalid probability row")
    row = tuple(float(value) for value in values)
    if (any(row[index] != 0 for index, legal in enumerate(mask) if not legal)
            or not math.isclose(math.fsum(row), 1.0, abs_tol=2e-5)):
        raise ContractError("raw policy probability row has illegal mass or does not conserve one")
    return row


class _RawReplayPolicy:
    """One-snapshot raw policy with actual inference timing and zero tree work."""

    def __init__(self, policy: Any) -> None:
        from ..neural_policy import TransformerInferenceTimingAccumulator

        if (policy.result.model_config.window_size != 1 or not policy.deterministic
                or policy.exploration_epsilon != 0 or policy.sampling_temperature != 1
                or policy.family_gated_selection or policy.forward_fn != policy._default_forward):
            raise ContractError("profile raw arm requires local one-snapshot deterministic masked argmax")
        self.policy = policy
        self.timing = TransformerInferenceTimingAccumulator()
        self.policy.inference_timing = self.timing
        self.policy.record_policy_distribution = True

    @property
    def stats(self) -> Any:
        # These zeros mean this raw arm performs no tree/fold/backup work; they
        # are not inferred search witnesses. Encode/forward work is measured by
        # the canonical policy's real timing sink, including forced choices.
        values = {field: 0 for field in _LiveEngineTimingDecider._STATS_FIELDS}
        values.update(model_evals=self.timing.neural_forward_count,
                      encode_wall_seconds=self.timing.observation_encoding_seconds,
                      model_wall_seconds=self.timing.neural_forward_seconds,
                      depth_reached_histogram={})
        return SimpleNamespace(**values)

    def warm_public_prefix_for_replay(self, **_kwargs: Any) -> None:
        # One-snapshot raw inference has no native incremental fold to warm.
        # Shared live replay still verifies the complete public root before
        # selection. Multi-window models are refused, not truncated silently.
        self.policy.reset()

    def select_action_with_context(self, context: Any, *, rng: random.Random) -> Any:
        from ..neural_policy import observation_spec_from_model_config

        context.observation.validate(observation_spec_from_model_config(self.policy.result.model_config))
        self.policy.reset()
        before = self.timing.neural_forward_count
        decision = self.policy.select_action(context.observation, rng=rng)
        row = _probabilities(decision.metadata.get("policy_distribution"),
                             context.observation.legal_action_mask)
        if self.timing.neural_forward_count - before != 1:
            raise ContractError("raw profile did not witness exactly one neural forward")
        legal = [index for index, enabled in enumerate(context.observation.legal_action_mask) if enabled]
        if decision.action_index != max(legal, key=lambda index: (row[index], -index)):
            raise ContractError("raw profile selection differs from masked argmax")
        return replace(decision, metadata={**decision.metadata, "raw_policy": {
            "selector": "deterministic_masked_argmax", "model_evals": 1,
            "policy_distribution": list(row), "action_index": decision.action_index,
        }})


class _LiveRawPolicyTimingDecider(_LiveEngineTimingDecider):
    """Reuse full source-root replay/validation, never replay persisted tensors."""

    def _policy_for(self, config: SearchConfig) -> Any:
        from ..engine_search import _fence_calibration_seam
        from ..neural_policy import (FreshValueHeadWarning, load_transformer_checkpoint_payload,
                                    load_transformer_policy)

        if config.inference_mode != "local":
            raise ContractError("raw profile supports local inference only")
        if config.config_id in self._policies:
            return self._policies[config.config_id]
        if sha256_file(self._contract.checkpoint_path) != self._contract.checkpoint_sha256:
            raise ContractError("raw profile checkpoint bytes drift")
        payload = load_transformer_checkpoint_payload(self._contract.checkpoint_path)
        _fence_calibration_seam(payload, "raw profile shared champion")
        with warnings.catch_warnings():
            warnings.simplefilter("error", FreshValueHeadWarning)
            policy = load_transformer_policy(self._contract.checkpoint_path,
                device=self._contract.model_device, deterministic=True, exploration_epsilon=0.0,
                sampling_temperature=1.0, family_gated_selection=False)
        if (policy.weights_sha256 != self._contract.checkpoint_sha256
                or tuple(policy.result.model_config.category_vocab) != self._contract.category_vocab
                or policy.result.belief_set_source_hash is None
                or policy.result.belief_set_source_hash != self._set_source.metadata.source_hash):
            raise ContractError("raw profile checkpoint/vocabulary/belief-source binding drift")
        adapter = _RawReplayPolicy(policy)
        self._policies[config.config_id] = adapter
        return adapter


def make_profile_decider(
    contract: CheckpointContract, showdown_root: str, *, arm: str, mode: str,
    opponent_seed: int, deadline_ms: int, native_batch_guard_ms: int,
    model_leaf_override: str | None = None,
    policy_opponent_diagnostics: Any | None = None,
) -> _LiveEngineTimingDecider:
    if policy_opponent_diagnostics is not None:
        from ..policy_opponent_diagnostics import PolicyOpponentDiagnostics
        if (type(policy_opponent_diagnostics) is not PolicyOpponentDiagnostics
                or arm != "incumbent_mcts" or model_leaf_override != "raw_policy_terminal"):
            raise ContractError("callback diagnostics require the explicit incumbent raw-terminal profile")
    _unsigned_seed(opponent_seed)
    if arm not in ARMS or mode not in MODES:
        raise ContractError("unsupported profile arm or timing mode")
    if model_leaf_override not in (None, "hp_fraction", "raw_policy_terminal") or (model_leaf_override is not None
            and arm != "incumbent_mcts"):
        raise ContractError("model leaf override requires an implemented incumbent valuation")
    if (type(deadline_ms) is not int or deadline_ms <= 0
            or type(native_batch_guard_ms) is not int
            or not 0 <= native_batch_guard_ms < deadline_ms):
        raise ContractError("profile deadline/batch guard is invalid")
    if arm == "raw_policy":
        return _LiveRawPolicyTimingDecider(contract, showdown_root)
    return _LiveEngineTimingDecider(contract, showdown_root,
        override_telemetry=True, record_joint_actions=True, model_priors=True, use_opponent_priors=False,
        model_world_workers=1,
        model_decision_time_ms=deadline_ms if mode == "matched_deadline" else None,
        model_native_batch_guard_ms=native_batch_guard_ms if mode == "matched_deadline" else 0,
        policy_opponent=arm == "own_policy_opponent_mcts",
        policy_opponent_seed=opponent_seed if arm == "own_policy_opponent_mcts" else None,
        **({"model_leaf_override": model_leaf_override} if model_leaf_override is not None else {}),
        **({"policy_opponent_diagnostics": policy_opponent_diagnostics}
           if policy_opponent_diagnostics is not None else {}),
        **(dict(rollout_count=1, rollout_max_plies=250, rollout_policy="raw_argmax",
                rollout_seed=opponent_seed, rollout_threads=1, rollout_branch_on_damage=True)
           if model_leaf_override == "raw_policy_terminal" else {}))


def validate_selection(telemetry: Any, *, arm: str, mode: str, config: SearchConfig,
                       mask: Sequence[bool], opponent_seed: int, deadline_ms: int,
                       native_batch_guard_ms: int, model_leaf_override: str | None = None) -> None:
    """Validate arm identity and allocation; configuration alone is not proof."""
    if not isinstance(telemetry, Mapping) or not isinstance(telemetry.get("root_action"), str) or not telemetry["root_action"]:
        raise ContractError("profile has no serialized selected action")
    for key in ("fallbacks", "prior_fallbacks", "invalid_actions"):
        if type(telemetry.get(key)) is not int or telemetry[key] != 0:
            raise ContractError("profile selection has invalid fallback/action evidence")
    for key in ("total_iterations", "model_evals", "max_depth_reached"):
        if type(telemetry.get(key)) is not int or telemetry[key] < 0:
            raise ContractError("profile work/depth witness is malformed")
    engine = telemetry.get("engine_mcts")
    if not isinstance(engine, Mapping):
        raise ContractError("profile engine witness is missing")
    if arm == "raw_policy":
        raw = telemetry.get("raw_policy")
        if (engine or not isinstance(raw, Mapping)
                or raw.get("selector") != "deterministic_masked_argmax"
                or type(raw.get("model_evals")) is not int or raw["model_evals"] != 1
                or telemetry["model_evals"] != 1 or telemetry["total_iterations"] != 0
                or telemetry["max_depth_reached"] != 0):
            raise ContractError("raw profile contains search work or lacks raw inference proof")
        row = _probabilities(raw.get("policy_distribution"), mask)
        legal = [index for index, enabled in enumerate(mask) if enabled]
        if type(raw.get("action_index")) is not int or raw["action_index"] != max(legal, key=lambda index: (row[index], -index)):
            raise ContractError("raw profile selected action differs from its prior row")
        return
    if arm not in ARMS or mode not in MODES or "raw_policy" in telemetry or engine.get("leaf_eval") != "model":
        raise ContractError("profile search arm identity drift")
    from ..engine_search import require_model_leaf_witness
    if model_leaf_override is not None and arm != "incumbent_mcts":
        raise ContractError("model leaf override belongs to the incumbent arm only")
    require_model_leaf_witness({"engine_mcts": engine}, model_leaf_override=model_leaf_override)
    if model_leaf_override is not None:
        rows = engine["model_leaf_override"]["native_invocations"]
        if (sum(row["completed_iterations"] for row in rows) != telemetry["total_iterations"]
                or sum(row["model_evals"] for row in rows) != telemetry["model_evals"]):
            raise ContractError("model-tree leaf work differs from actual native invocations")
    from ..engine_search import EngineSearchWitnessError, validate_native_joint_action_witness
    joint = engine.get("joint_actions")
    if (not isinstance(joint, Mapping)
            or joint.get("scope") != "per_native_invocation_without_belief_reweighting"
            or not isinstance(joint.get("native_invocations"), list) or not joint["native_invocations"]):
        raise ContractError("profile native joint-action witness missing")
    completed = 0
    for invocation in joint["native_invocations"]:
        if (not isinstance(invocation, Mapping) or type(invocation.get("world_seed")) is not int
                or type(invocation.get("belief_multiplicity")) is not int or invocation["belief_multiplicity"] <= 0
                or not isinstance(invocation.get("witness"), Mapping)):
            raise ContractError("profile joint-action invocation identity malformed")
        witness = invocation["witness"]
        try:
            validate_native_joint_action_witness({"iterations": witness.get("root_completed_traversals"),
                                                   "joint_action_witness": witness})
        except EngineSearchWitnessError as error:
            raise ContractError(str(error)) from error
        completed += witness["root_completed_traversals"]
    if completed != telemetry["total_iterations"]:
        raise ContractError("profile joint-action work differs from actual native iterations")
    searched = engine.get("worlds_searched")
    if (type(searched) is not int or not 0 < searched <= config.worlds
            or type(engine.get("worlds_constructed")) is not int
            or engine["worlds_constructed"] != config.worlds):
        raise ContractError("profile belief-world construction/coverage drift")
    if mode == "fixed_work":
        if (searched != config.worlds or "time_budget" in engine
                or engine.get("aggregated_choices_basis") != "full_budget"
                or telemetry["total_iterations"] != config.worlds * config.sims):
            raise ContractError("profile fixed work did not complete its exact allocation")
    else:
        budget = engine.get("time_budget")
        if (not isinstance(budget, Mapping) or budget.get("scope") != "whole_model_decision"
                or type(budget.get("requested_ms")) is not int or budget["requested_ms"] != deadline_ms
                or type(budget.get("native_batch_guard_ms")) is not int
                or budget["native_batch_guard_ms"] != native_batch_guard_ms
                or engine.get("aggregated_choices_basis") not in ("deadline_prefix", "full_budget")
                or not isinstance(budget.get("native_invocations"), list)
                or not budget["native_invocations"]):
            raise ContractError("profile deadline witness is missing or drifted")
        invocations = budget["native_invocations"]
        for invocation in invocations:
            if (not isinstance(invocation, Mapping)
                    or any(type(invocation.get(key)) is not int or invocation[key] < 0
                           for key in ("requested_iterations", "completed_iterations", "remaining_iterations"))
                    or invocation["completed_iterations"] + invocation["remaining_iterations"] != invocation["requested_iterations"]):
                raise ContractError("profile deadline iteration receipt is malformed")
        if sum(row["completed_iterations"] for row in invocations) != telemetry["total_iterations"]:
            raise ContractError("profile deadline productive work differs from native receipts")
    override = engine.get("override")
    allocation = override.get("root_allocation") if isinstance(override, Mapping) else None
    if (not isinstance(allocation, Mapping) or allocation.get("worlds") != searched
            or allocation.get("prior_authority") is not True or allocation.get("prior_cause") is not None
            or not isinstance(allocation.get("arms"), list)):
        raise ContractError("profile lacks authoritative root allocation")
    indices = [row.get("action_index") if isinstance(row, Mapping) else None for row in allocation["arms"]]
    if (any(type(index) is not int for index in indices) or len(indices) != len(set(indices))
            or set(indices) != {index for index, enabled in enumerate(mask) if enabled}):
        raise ContractError("profile root allocation differs from its source legal surface")
    for field in ("visit_share", "model_prior", "reported_prior"):
        values = [row.get(field) for row in allocation["arms"]]
        if (any(type(value) not in (int, float) or not math.isfinite(value) or value < 0 for value in values)
                or not math.isclose(math.fsum(values), 1.0, abs_tol=2e-5)):
            raise ContractError("profile root prior/visit allocation does not conserve one")
    opponent = engine.get("policy_opponent")
    if arm == "incumbent_mcts":
        if opponent is not None:
            raise ContractError("incumbent profile unexpectedly enabled the opponent sampler")
    else:
        if (not isinstance(opponent, Mapping) or opponent.get("mode") != "own_policy_callback"
                or type(opponent.get("seed_root")) is not int or opponent["seed_root"] != opponent_seed
                or opponent.get("seed_derivation") != "sha256-domain-separated-v1"
                or not isinstance(opponent.get("native_invocations"), list)
                or not opponent["native_invocations"]):
            raise ContractError("candidate profile lacks its registered opponent witness")
        for invocation in opponent["native_invocations"]:
            if (not isinstance(invocation, Mapping)
                    or any(type(invocation.get(key)) is not int or invocation[key] < 0
                           for key in ("evaluations", "provider_calls", "samples"))
                    or invocation["evaluations"] > invocation["provider_calls"]):
                raise ContractError("candidate profile opponent work witness is malformed")


def profile_root(
    record: Any, *, source_records: Sequence[Any], contract: CheckpointContract,
    showdown_root: str, config: SearchConfig, seed: int, root_ordinal: int,
    source_requested_players: Sequence[str], deadline_ms: int = 1000,
    native_batch_guard_ms: int = 64,
    decider_factory: Callable[..., Any] = make_profile_decider,
    clock: Callable[[], float] = time.perf_counter,
    on_row: Callable[[str, str, Mapping[str, Any]], None] | None = None,
) -> dict[str, Any]:
    """Measure each arm/mode without swallowing errors or dropping a root.

    Source-owned prefix refusals remain in the denominator. Selection refusals
    retain their elapsed decision wall and all other arm rows. A caller must
    byte-verify the frozen roster and source references BEFORE invoking this
    unit, and must publish its run/resource contract before executing it.
    This is not an experiment launcher or a coverage-qualification receipt.
    """
    if type(root_ordinal) is not int or not 0 <= root_ordinal < 32:
        raise ContractError("profile root ordinal is outside the frozen denominator")
    if tuple(source_requested_players) not in {(record.acting_player,), ("p1", "p2")}:
        raise ContractError("profile source request boundary is invalid")
    decision_seed = root_seed(seed, record.decision_id, purpose="decision")
    opponent_seed = root_seed(seed, record.decision_id, purpose="opponent")
    result: dict[str, Any] = {"decision_id": record.decision_id,
        "source_seed": record.seed, "seat": record.acting_player,
        "turn_index": record.turn_index, "decision_seed": decision_seed,
        "opponent_seed": opponent_seed, "seed_domain": SEED_DOMAIN.decode().rstrip("\0"),
        "timing_boundary": "request_available_to_validated_showdown_choice",
        "qualification": "PENDING_ROSTER_BYTE_BINDING_AND_RUNTIME_QUALIFICATION",
        "modes": {}, "state": "REFUSED"}
    try:
        prefix = source_bound_replay_prefix(record, source_records=source_records)
    except SourceRootReplayError as error:
        result["refusal"] = {"phase": "source_prefix", "type": type(error).__name__, "reason": str(error)}
        return result
    result["prefix_repairs"] = [repair.to_dict() for repair in prefix.repairs]
    for mode_index, mode in enumerate(MODES):
        offset = (root_ordinal + mode_index) % len(ARMS)
        ordered = ARMS[offset:] + ARMS[:offset]
        rows = result["modes"][mode] = {}
        for order, arm in enumerate(ordered):
            decider = None
            decision_started = None
            elapsed = None
            telemetry = None
            phase = "preparation"
            try:
                decider = decider_factory(contract, showdown_root, arm=arm, mode=mode,
                    opponent_seed=opponent_seed, deadline_ms=deadline_ms,
                    native_batch_guard_ms=native_batch_guard_ms)
                prepared = decider.prepare_public_decision(record, config,
                    public_action_rounds=prefix.public_action_rounds,
                    decision_rng_seed=decision_seed, source_requested_players=source_requested_players)
                decision_started = clock()
                phase = "decision"
                telemetry = prepared()
                elapsed = clock() - decision_started
                phase = "validation"
                if not math.isfinite(elapsed) or elapsed < 0:
                    raise ContractError("profile selection has invalid decision wall")
                validate_selection(telemetry, arm=arm, mode=mode, config=config,
                    mask=record.current_legal_action_mask, opponent_seed=opponent_seed,
                    deadline_ms=deadline_ms, native_batch_guard_ms=native_batch_guard_ms)
                rows[arm] = {"state": "COMPLETE", "execution_order": order,
                             "decision_wall_seconds": elapsed, "telemetry": telemetry}
            except (ContractError, ValueError, RuntimeError) as error:
                rows[arm] = {"state": "REFUSED", "execution_order": order,
                    "decision_wall_seconds": elapsed if elapsed is not None else
                        None if decision_started is None else clock() - decision_started,
                    "phase": phase, "type": type(error).__name__, "reason": str(error),
                    **({"telemetry": telemetry} if telemetry is not None else {})}
                diagnostic = refusal_diagnostic(error)
                if diagnostic is not None:
                    rows[arm]["diagnostic"] = {
                        **diagnostic, "decision_id": record.decision_id,
                        "decision_seed": decision_seed, "opponent_seed": opponent_seed,
                    }
            finally:
                if decider is not None:
                    try:
                        decider.close()
                    except (OSError, RuntimeError) as error:
                        rows[arm].update(state="REFUSED", cleanup_error={
                            "type": type(error).__name__, "reason": str(error)})
            if on_row is not None:
                # Outside the decision timer and after cleanup adjudication.
                # A durable sink failure stops execution; never hide it as an
                # eligibility refusal or retry/overwrite its earlier receipt.
                on_row(mode, arm, rows[arm])
    if all(row["state"] == "COMPLETE" for mode in result["modes"].values() for row in mode.values()):
        result["state"] = "COMPLETE"
    return result
