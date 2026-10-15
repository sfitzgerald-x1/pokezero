"""Receipt reconciliation for the reconciled collector, without execution.

Consumes a supplied public root, selectors and action-cell continuations. A
caller-supplied independent validator must check actual work/cleanup and terminal
receipts against their durable evidence. Hash equality binds that validation;
it is not a substitute for it. No sources, adapters, games or holdout are opened.
"""
from copy import deepcopy
from dataclasses import asdict
import math

from .search_over_raw import SearchConfiguration, _raw_action, digest, require
from .search_over_raw_screening import action_score, paired_root_interval


def selector_key(configuration):
    return "raw" if configuration.arm == "raw" else configuration.identity


def _sha(value):
    return type(value) is str and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def _accepted_units(configuration, selected):
    evidence = selected["evidence"]
    if configuration.arm == "raw":
        return 0
    count = (evidence.get("total_iterations") if configuration.arm == "incumbent"
        else evidence.get("result", {}).get("trajectories"))
    require(type(count) is int and count >= 0, "invalid accepted-work count")
    # The independent selector validator checks atomic backup witnesses, not
    # just these counters. This function never asserts runtime qualification.
    return count


def reconcile_root_evidence(bundle, *, namespace, slot, configurations,
                            public_record_sha256, runtime_hashes,
                            validate_selector, validate_outcome, allow_unqualified_at_ceiling=False):
    """Derive sharp contrasts from one exact root and shared action-cell bank.

    Validators return the canonical digest of the exact supplied receipt after
    checking its durable evidence, or None for an unvalidated terminal cell.
    Selector failures raise; uncertain outcomes remain [0,1]. Configurations
    are stage-local, never a factorial prerequisite. No input is mutated.
    """
    require(type(allow_unqualified_at_ceiling) is bool and callable(validate_selector) and callable(validate_outcome),
        "independent receipt validators required")
    require(type(configurations) in (tuple, list) and configurations
        and all(type(c) is SearchConfiguration for c in configurations)
        and len({selector_key(c) for c in configurations}) == len(configurations)
        and sum(c.arm == "raw" for c in configurations) == 1,
        "unique stage-local selectors and exactly one shared raw control required")
    require(all(c.leaf != "raw_rollout" for c in configurations),
        "terminal rollout evidence belongs only to the separate fixed-work diagnostic")
    require(type(slot) is dict and set(slot) == {"root_id", "source_seed", "root_slot"}
        and type(slot["root_id"]) is str and slot["root_id"]
        and type(slot["source_seed"]) is int and 0 <= slot["source_seed"] < 2**32
        and type(slot["root_slot"]) is int and slot["root_slot"] >= 0
        and _sha(public_record_sha256), "bound supplied public root required")
    require(type(bundle) is dict and set(bundle) == {
        "root_binding", "continuation_contract", "selections", "outcomes"},
        "exact root evidence bundle required")
    saved = deepcopy(bundle)
    binding = saved["root_binding"]
    require(type(binding) is dict and set(binding) == {
        "root_id", "source_seed", "public_record_sha256", "information_key", "legal_action_mask"}
        and binding["root_id"] == slot["root_id"]
        and type(binding["source_seed"]) is int and binding["source_seed"] == slot["source_seed"]
        and binding["public_record_sha256"] == public_record_sha256
        and type(binding["information_key"]) is str and binding["information_key"],
        "public root binding drift")
    mask = binding["legal_action_mask"]
    require(type(mask) is list and mask and all(type(v) is bool for v in mask) and any(mask),
        "actual public legal action mask required")
    contract = dict(root_binding=binding, root_id=slot["root_id"], namespace=namespace,
        subject="p1", max_boundaries=250, replicates=8, selection_seed=slot["source_seed"],
        policy="raw_argmax_after_initial_sampled_opponent_reply")
    require(saved["continuation_contract"] == contract, "paired continuation law/root/seed drift")
    selectors = saved["selections"]
    names = {selector_key(c) for c in configurations}
    require(type(selectors) is dict and set(selectors) == names
        and type(runtime_hashes) is dict and set(runtime_hashes) == names
        and all(_sha(h) for h in runtime_hashes.values()), "exact stage runtime/selector roster required")
    selected_hashes, accepted, actions, statuses = {}, {}, {}, {}
    for cfg in configurations:
        name = selector_key(cfg)
        selected = selectors[name]
        require(type(selected) is dict and (selected.get("status") == "SELECTED"
            or allow_unqualified_at_ceiling and cfg.arm != "raw"
                and selected.get("status") == "UNQUALIFIED_AT_CEILING")
            and selected.get("root_id") == slot["root_id"] + ":" + name
            and selected.get("statistics_root_id") == slot["root_id"]
            and type(selected.get("selection_seed")) is int and selected["selection_seed"] == slot["source_seed"]
            and selected.get("configuration_sha256") == cfg.identity
            and selected.get("runtime_sha256") == runtime_hashes[name]
            and selected.get("independent_root_statistics") is True,
            "selection root/configuration/runtime/statistics drift")
        action = selected.get("action")
        zero = selected["status"] == "UNQUALIFIED_AT_CEILING"
        require((zero and action is None and selected.get("attempt_consumed") is True
                and selected.get("retry_authorized") is False and selected.get("continuation_authorized") is False)
            or (not zero and type(action) is int and 0 <= action < len(mask) and mask[action]),
            "illegal selected action or zero-work continuation")
        require(type(selected.get("elapsed_seconds")) in (int, float)
            and math.isfinite(selected["elapsed_seconds"]) and selected["elapsed_seconds"] >= 0,
            "actual selection wall time required")
        require("fixed_work_diagnostic" not in selected,
            "fixed-work evidence cannot enter the deployment or A1 lane")
        require(type(selected.get("nominal_search_seconds")) in (int, float)
            and selected["nominal_search_seconds"] == cfg.seconds, "decision ceiling drift")
        evidence = selected.get("evidence")
        require(type(evidence) is dict, "selector work witness required")
        if cfg.arm == "raw":
            require(evidence.get("selector") == "deterministic_masked_argmax"
                and type(evidence.get("model_forwards")) is int and evidence["model_forwards"] == 1
                and evidence.get("legal_actions") == [i for i, enabled in enumerate(mask) if enabled]
                and _raw_action(evidence["legal_actions"], evidence.get("priors", [])) == action,
                "shared raw control differs from actual masked argmax")
        count = _accepted_units(cfg, selected)
        require((zero and count == 0) or (not zero and (cfg.arm == "raw" or sum(mask) == 1 or count > 0)),
            "zero accepted search work cannot supply a scored choice")
        before = digest(selected)
        witness = validate_selector(cfg, deepcopy(selected), deepcopy(binding))
        require(witness == before and digest(selected) == before,
            "selector validation must bind this exact receipt and actual atomic work/cleanup")
        selected_hashes[name], accepted[name], statuses[name] = before, count, selected["status"]
        if not zero: actions[name] = action
    rows = saved["outcomes"]
    require(type(rows) is list, "action-cell continuation list required")
    cells, outcome_hashes, validations = {}, {}, {}
    expected = {(action, rep) for action in set(actions.values()) for rep in range(8)}
    for row in rows:
        require(type(row) is dict and set(row) == {
            "root_id", "action", "replicate", "status", "signed_outcome", "boundaries"},
            "exact terminal action-cell receipt required")
        action, rep = row["action"], row["replicate"]
        require(row["root_id"] == slot["root_id"] and type(action) is int
            and type(rep) is int and (action, rep) in expected and (action, rep) not in cells,
            "duplicate, unexpected or unbound shared outcome cell")
        require(type(row["boundaries"]) is int and 0 <= row["boundaries"] <= 250
            and row["status"] in {"COMPLETE", "CAPPED", "REFUSED", "UNSTARTED"}
            and ((row["status"] == "COMPLETE" and type(row["signed_outcome"]) is int
                and row["signed_outcome"] in (-1, 0, 1))
                or (row["status"] != "COMPLETE" and row["signed_outcome"] is None)),
            "malformed terminal status/score/cost")
        before = digest(row)
        witness = validate_outcome(deepcopy(row), deepcopy(contract))
        require(witness is None or (row["status"] == "COMPLETE" and witness == before),
            "terminal validation must bind this exact complete receipt")
        score = None if row["signed_outcome"] is None else (row["signed_outcome"] + 1) / 2
        cell = dict(status=row["status"], validated=witness is not None, score=score)
        action_score(cell)  # Even action aliases do not excuse malformed evidence.
        cells[action, rep] = cell
        name = f"{action}:{rep}"
        outcome_hashes[name], validations[name] = before, witness
    # Missing records are unknown, not inferred terminal scores or retry grants.
    intervals = {name: list(paired_root_interval(actions[name], actions["raw"], cells)) if name in actions else [-1.,1.]
        for name in names if name != "raw"}
    require(digest(bundle) == digest(saved), "input evidence changed during reconciliation")
    return dict(schema="pokezero.search-over-raw.reconciled-root-evidence.v2" if allow_unqualified_at_ceiling
            else "pokezero.search-over-raw.reconciled-root-evidence.v1",
        **({"selector_statuses":statuses, "unqualified_choice_is_uncertain":True} if allow_unqualified_at_ceiling else {}),
        root_id=slot["root_id"], source_seed=slot["source_seed"],
        public_record_sha256=public_record_sha256, bundle_sha256=digest(saved),
        continuation_contract_sha256=digest(contract),
        configurations={selector_key(c): asdict(c) for c in configurations},
        selected_sha256=selected_hashes, runtime_sha256=deepcopy(runtime_hashes),
        outcome_sha256=outcome_hashes, validated_outcome_sha256=validations,
        actions=actions, root_intervals=intervals, accepted_units=accepted,
        unique_action_cells=len(expected), recorded_action_cells=len(cells),
        validated_action_cells=sum(c["validated"] for c in cells.values()),
        uncertain_action_cells=len(expected)-sum(c["validated"] for c in cells.values()),
        forced_action=sum(mask) == 1, substantive_search=sum(mask) > 1,
        full_world_matching_established=False, runtime_qualified=False,
        collection_authorized=False, holdout_authorized=False, phase_b_authorized=False)
