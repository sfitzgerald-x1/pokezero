"""Fresh raw-game source collection, separate from search and outcome audits.

This is a collector building block, not a scientific launch authorization.
Callers must bind their runtime and exposure inventory before using real games.
Only canonical public decision records leave this module. The full trajectory
stays in memory and is never offered to a search adapter.
"""
from __future__ import annotations

from dataclasses import replace
import math
import time
from typing import Any, Mapping

from .search_over_raw import _raw_action, digest, require, select_source_requests


class AuditedRawPolicy:
    """Verify every source choice against the champion's actual masked row.

    Only select_action receives an observation; no privileged PolicyContext or
    simulator is passed through. The one-snapshot checkpoint restriction makes
    the same law usable by the continuation evaluator without losing history.
    """

    def __init__(self, policy: Any, *, checkpoint_sha256: str, public_context_sink=None) -> None:
        require(len(checkpoint_sha256) == 64
            and all(c in "0123456789abcdef" for c in checkpoint_sha256), "invalid checkpoint digest")
        require(policy.weights_sha256 == checkpoint_sha256, "raw checkpoint binding mismatch")
        require(policy.result.model_config.window_size == 1 and policy.deterministic is True
            and policy.exploration_epsilon == 0 and policy.sampling_temperature == 1
            and policy.family_gated_selection is False and policy.history_mask_k is None
            and policy.forward_fn == policy._default_forward,
            "source requires unmodified local one-snapshot masked argmax")
        self.policy = policy
        self.policy.record_policy_distribution = True
        self.policy_id = "search-over-raw-source:" + checkpoint_sha256
        self.verified_decisions = 0
        self.public_context_sink = public_context_sink

    def reset(self) -> None:
        self.policy.reset()
        self.verified_decisions = 0

    def select_action(self, observation: Any, *, rng: Any) -> Any:
        decision = self.policy.select_action(observation, rng=rng)
        row = decision.metadata.get("policy_distribution")
        mask = observation.legal_action_mask
        require(isinstance(row, (list, tuple)) and len(row) == len(mask),
            "raw source omitted its full masked distribution")
        require(all(type(p) in (int, float) and math.isfinite(p) and p >= 0 for p in row)
            and all(legal or row[i] == 0 for i, legal in enumerate(mask))
            and math.isclose(math.fsum(row), 1, abs_tol=2e-5, rel_tol=0),
            "invalid masked source distribution")
        legal = [i for i, enabled in enumerate(mask) if enabled]
        chosen = _raw_action(legal, [row[i] for i in legal])
        require(decision.action_index == chosen, "source decision differs from masked argmax")
        self.verified_decisions += 1
        return replace(decision, policy_id=self.policy_id)

    def select_action_with_context(self, context: Any, *, rng: Any) -> Any:
        # The champion sees only its observation, never the context or auditor.
        decision = self.select_action(context.observation, rng=rng)
        if self.public_context_sink is not None:
            from .head_to_head import public_only_context
            self.public_context_sink(public_only_context(context), decision.action_index)
        return decision


def validate_source_receipt(source, seed, *, contract, panel, require_full=False):
    """Reconcile the fixed priority-selected plus missing source partition."""
    slots = [r for r in contract["panels"][panel]["root_slots"] if r["source_seed"] == seed]
    rows = source["eligible_public_records"]
    indices = [r["source_request_index"] for r in rows]
    require(slots and source["contract_sha256"] == digest(contract) and source["source_seed"] == seed
        and source["panel"] == panel and source["status"] == "COMPLETE"
        and source["source_terminal_complete"] is True and source["source_policy"] == "raw_argmax_both_seats"
        and source["eligible_requests"] == len(rows) and digest(rows) == source["eligible_catalog_sha256"]
        and all(type(i) is int and (i > 0 if contract.get("exclude_opening_requests") else i >= 0)
            for i in indices) and len(set(indices)) == len(indices)
        and source["requested_root_slots"] == len(slots)
        and (not require_full or source["missing_root_ids"] == []),
        "source catalog incomplete or drifted")
    selected = select_source_requests(contract["namespace"], seed, indices, len(slots))
    catalog = {r["source_request_index"]: r["public_record_sha256"] for r in rows}
    require(len(source["roots"]) == len(selected)
        and source["missing_root_ids"] == [r["root_id"] for r in slots[len(selected):]],
        "missing roots; no replacement")
    for slot, (root, index) in enumerate(zip(source["roots"], selected)):
        require(root["root_id"] == slots[slot]["root_id"] and root["source_request_index"] == index
            and digest(root["public_record"]) == root["public_record_sha256"] == catalog[index],
            "source priority selection differs")


def source_roots(contract: Mapping, *, panel: str, source_seed: int,
                 trajectory: Any) -> dict:
    """Freeze the complete catalog before outcome-independent priority sampling.

    Failed/capped games yield no usable roots, not a replacement game. A short
    completed game may fill fewer slots; its remaining slots stay explicit.
    Source wins/losses are not included in search inputs or used for sampling.
    """
    from ..public_decision_corpus import PublicDecisionRecord, public_decision_records_from_trajectory

    require(panel in contract["panels"]
        and source_seed in contract["panels"][panel]["seeds"], "unregistered source seed/panel")
    require(trajectory.seed == source_seed and trajectory.format_id == "gen3randombattle",
        "source trajectory identity mismatch")
    slots = [r for r in contract["panels"][panel]["root_slots"] if r["source_seed"] == source_seed]
    roots = []
    terminal = trajectory.terminal
    complete = terminal is not None and not terminal.capped
    catalog = public_decision_records_from_trajectory(trajectory,
        acting_player=contract["candidate_seat"]) if complete else ()
    if contract.get("exclude_opening_requests", False):
        catalog = tuple(r for r in catalog if r.turn_index > 0)
    require(len({r.turn_index for r in catalog}) == len(catalog), "duplicate source request")
    eligible = [dict(source_request_index=r.turn_index,
        public_record_sha256=digest(PublicDecisionRecord.from_dict(r.to_dict()).to_dict()))
        for r in catalog]
    selected = select_source_requests(contract["namespace"], source_seed,
        [r.turn_index for r in catalog], len(slots))
    records = {r.turn_index: r for r in catalog}
    for slot, request_index in zip(slots, selected):
        record = records[request_index]
        # Canonical parser rejects unknown/private fields rather than serializing
        # arbitrary environment metadata into an ostensibly public artifact.
        public = PublicDecisionRecord.from_dict(record.to_dict()).to_dict()
        roots.append(dict(root_id=slot["root_id"], source_request_index=request_index,
            public_record=public, public_record_sha256=digest(public)))
    return dict(schema="pokezero.search-over-raw.source.v1", panel=panel,
        source_seed=source_seed, contract_sha256=digest(contract),
        status="COMPLETE" if complete else "UNCERTAIN_SOURCE", source_policy="raw_argmax_both_seats",
        eligible_requests=len(catalog), eligible_public_records=eligible,
        eligible_catalog_sha256=digest(eligible), requested_root_slots=len(slots), roots=roots,
        missing_root_ids=[slot["root_id"] for slot in slots[len(roots):]],
        source_terminal_complete=complete, scientific_strength_evidence=False,
        search_invoked=False, replacement_seeds=[])


def collect_raw_source(contract: Mapping, *, panel: str, source_seed: int,
                       env: Any, policies: Mapping[str, AuditedRawPolicy],
                       max_decision_rounds: int = 250, decision_sink=None,
                       sealed_pre_step_sink=None) -> dict:
    """Use the production rollout driver; return only selected public roots.

    The caller owns env lifecycle, durable attempts and runtime qualification.
    Exceptions propagate: no retry, partial-trajectory acceptance or redraw.
    """
    from ..rollout import RolloutConfig, RolloutDriver

    require(panel in contract["panels"]
        and source_seed in contract["panels"][panel]["seeds"], "unregistered source seed/panel")
    require(set(policies) == {"p1", "p2"}
        and all(isinstance(p, AuditedRawPolicy) for p in policies.values()),
        "both seats require audited raw policies")
    require(policies["p1"].policy.weights_sha256 == policies["p2"].policy.weights_sha256,
        "source seats must share the same champion")
    require(type(max_decision_rounds) is int and max_decision_rounds > 0, "invalid source cap")
    started = time.perf_counter()
    result = RolloutDriver(env=env, policies=policies, config=RolloutConfig(
        max_decision_rounds=max_decision_rounds, hide_opponent_legal_action_masks=True,
        decision_sink=decision_sink, sealed_pre_step_sink=sealed_pre_step_sink)).run(seed=source_seed,
            battle_id=f"search-over-raw:{contract['namespace']}:{panel}:{source_seed}")
    receipt = source_roots(contract, panel=panel, source_seed=source_seed,
        trajectory=result.trajectory)
    receipt.update(elapsed_seconds=time.perf_counter() - started,
        decision_boundaries=result.decision_round_count,
        checkpoint_sha256=policies["p1"].policy.weights_sha256,
        verified_raw_decisions=sum(p.verified_decisions for p in {id(p): p for p in policies.values()}.values()))
    return receipt
