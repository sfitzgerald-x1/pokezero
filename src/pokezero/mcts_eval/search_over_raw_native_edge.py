"""Prospective native edge support; not independent physics or qualification.

The native helper regenerates instructions from the actual prestate and exact
action slots. Historical captures lack those slots/instructions and remain
uncertain. Neither hashes nor later endpoint agreement upgrade that evidence.
"""
from copy import deepcopy
import json
import math

from .search_over_raw import digest, require
from .search_over_raw_correspondence import boundary_bindings, _state_sha
from .search_over_raw_native_boundary import _choice

EDGE_SCHEMA = "pokezero.native-captured-edge.v1"
SUPPORT_SCHEMA = "pokezero.native-edge-support.v1"
EDGE_FIELDS = {"schema", "root", "branch_index", "branch_on_damage", "actions",
    "instruction_count", "instructions_debug", "instructions_debug_sha256"}
FALSE_FLAGS = {"search_or_evaluator_run", "independent_physics_validated",
    "showdown_transition_validated", "equal_rng_coupling", "runtime_qualified",
    "scientific_collection_authorized"}
SUPPORT_FIELDS = {"schema", "before_utf8_sha256", "after_utf8_sha256", "captured_edge",
    "prestate_terminal", "joint_action_admissible", "regenerated_branches",
    "matching_successor_branch_indices", "successor_in_native_support",
    "recorded_branch_instructions_match", "recorded_edge_supported", "private_controller_only"} | FALSE_FLAGS
SUCCESSOR_SCHEMA="pokezero.native-scoped-successor-support.v1"
SUCCESSOR_SCOPE="captured native prestate, exact joint action and original branch_on_damage law"
ENDPOINT_FIELDS=("active_identities","hp","fainted","status","effective_legal_choices")


def read_native_successor_support(raw,step):
    """Complete prospective branch export, not independent engine semantics."""
    from .search_over_raw_native_boundary import read_native_boundary
    require(type(raw) is str and len(raw.encode())<=8388608,"bounded complete native successor JSON required")
    value=json.loads(raw,object_pairs_hook=_unique_object)
    require(type(value) is dict and set(value)=={"schema","captured_support","prestate_boundary",
        "branch_masses","positive_successors","complete_positive_mass_roster","scope",
        "private_controller_only","independent_physics_validated","showdown_transition_validated",
        "equal_rng_coupling","runtime_qualified","scientific_collection_authorized"}
        and value["schema"]==SUCCESSOR_SCHEMA and value["scope"]==SUCCESSOR_SCOPE
        and value["complete_positive_mass_roster"] is True and value["private_controller_only"] is True
        and all(value[k] is False for k in ("independent_physics_validated","showdown_transition_validated",
            "equal_rng_coupling","runtime_qualified","scientific_collection_authorized")),
        "native successor completeness/scope drift")
    support=read_native_edge_support(json.dumps(value["captured_support"],allow_nan=False),step)
    read_native_boundary(value["prestate_boundary"],state_text=step["before"],root=step["native_edge"]["root"])
    masses,successors=value["branch_masses"],value["positive_successors"]
    require(type(masses) is list and len(masses)==support["regenerated_branches"]
        and type(successors) is list,"full original native branch roster required")
    for index,row in enumerate(masses):
        require(type(row) is dict and set(row)=={"branch_index","percentage"}
            and type(row["branch_index"]) is int and row["branch_index"]==index
            and type(row["percentage"]) in (int,float) and math.isfinite(row["percentage"])
            and row["percentage"]>=0,"native branch index/mass drift")
    total=sum(row["percentage"] for row in masses)
    require(abs(total-100)<.5 if support["joint_action_admissible"] else not masses,
        "complete native mass differs from captured branch law")
    positive=[r["branch_index"] for r in masses if r["percentage"]>0]
    require(len(successors)==len(positive),"positive native successors omitted or added")
    matching=[]
    for index,row in zip(positive,successors):
        require(type(row) is dict and set(row)=={"branch_index","percentage","state_utf8","boundary"}
            and type(row["branch_index"]) is int and row["branch_index"]==index
            and digest(row["percentage"])==digest(masses[index]["percentage"])
            and type(row["state_utf8"]) is str and row["state_utf8"],"positive successor coordinate/mass/state drift")
        read_native_boundary(row["boundary"],state_text=row["state_utf8"],root=False)
        if row["state_utf8"]==step["after"]:matching.append(index)
    require(matching==support["matching_successor_branch_indices"],"captured native after-state differs from full support")
    digest(value)  # All ancillary JSON is finite/canonicalizable too.
    return value


def extract_native_successor_support(capture,*,root_binding,selection_report_sha256,native_successor_support=None):
    """Private, post-selection support for EVERY original arriving coordinate."""
    require(native_successor_support is None or callable(native_successor_support),"bound native successor helper required")
    saved=deepcopy(capture)
    bindings=boundary_bindings(saved,root_binding=root_binding,selection_report_sha256=selection_report_sha256)
    rows=[]
    for binding,step in zip(bindings[1:],saved["steps"]):
        report=None
        if "native_edge" in step and native_successor_support is not None:
            edge=validate_captured_edge(step)
            report=read_native_successor_support(native_successor_support(step["before"],step["after"],
                json.dumps(edge,sort_keys=True)),step)
        rows.append(dict(ordinal=binding["ordinal"],binding=binding,support_report=report,
            status="UNCHECKABLE" if report is None else "COMPLETE_PROJECTED_NATIVE_SUPPORT",
            independent_physics_validated=False,runtime_qualified=False,scientific_collection_authorized=False))
    require(digest(capture)==digest(saved),"captured ancestry changed during successor extraction")
    return rows


def compare_projected_successors(observations,native_boundaries,support_rows,*,native_action_driven_origin_join):
    """Owned Showdown AFTER versus native support, never an RNG/physics claim.

    PRE agreement is only on required endpoint fields; hidden fields outside
    that projection can still differ. Duplicate projected branches are kept,
    not counted as extra transitions. An earlier gap cannot restore a prefix.
    """
    from .search_over_raw_native_boundary import project_native_boundary
    require(type(native_action_driven_origin_join) is bool and type(observations) is list
        and 2<=len(observations)<=65 and type(support_rows) is list
        and len(support_rows)==len(observations)-1 and (native_boundaries is None or
            type(native_boundaries) is list and len(native_boundaries)==len(observations)),
        "original native successor coordinate denominator required")
    before=digest(dict(observations=observations,native=native_boundaries,support=support_rows))
    require(all(type(row["ordinal"]) is int and row["ordinal"]==index for index,row in enumerate(observations))
        and (native_boundaries is None or all(type(row["ordinal"]) is int and row["ordinal"]==index
            for index,row in enumerate(native_boundaries))),"original boundary coordinates changed")
    rows=[];first_gap=None
    for ordinal,source in enumerate(support_rows,1):
        require(type(source["ordinal"]) is int and source["ordinal"]==ordinal
            and type(observations[ordinal]["ordinal"]) is int and observations[ordinal]["ordinal"]==ordinal,
            "projected successor coordinate drift")
        report=source["support_report"];observed=observations[ordinal];previous=observations[ordinal-1]
        if native_boundaries is not None:
            require(digest(source["binding"])==digest(native_boundaries[ordinal]["binding"]),
                "successor support differs from original captured coordinate")
        reason=None;matching=[];mass=None
        if not native_action_driven_origin_join:reason="OWNED_NATIVE_ACTION_REPLAY_UNAVAILABLE"
        elif report is None or native_boundaries is None:reason="COMPLETE_NATIVE_SUCCESSOR_SUPPORT_UNAVAILABLE"
        elif not report["captured_support"]["recorded_edge_supported"]:reason="CAPTURED_NATIVE_EDGE_UNSUPPORTED"
        elif any(previous["observed_required_fields"][k]!="MATCH" for k in ENDPOINT_FIELDS):
            reason="SCOPED_PRESTATE_MISMATCH_OR_UNKNOWN"
        elif observed["observed_required_fields"]["action_identity"]!="MATCH":reason="ACKNOWLEDGED_JOINT_ACTION_MISMATCH_OR_UNKNOWN"
        else:
            pre=project_native_boundary(report["prestate_boundary"])
            actual=native_boundaries[ordinal-1]["native"]
            require(all(digest(pre[k])==digest(actual[k]) for k in ENDPOINT_FIELDS),
                "successor helper prestate differs from actual captured native projection")
            projection=observed["reference_projection"]
            if projection is None or any(projection["showdown"].get(k) is None for k in ENDPOINT_FIELDS):
                reason="SCOPED_SHOWDOWN_SUCCESSOR_UNAVAILABLE"
            else:
                reference=projection["showdown"]
                for branch in report["positive_successors"]:
                    candidate=project_native_boundary(branch["boundary"])
                    if all(digest(candidate[k])==digest(reference[k]) for k in ENDPOINT_FIELDS):
                        matching.append(branch["branch_index"])
                mass=sum(branch["percentage"] for branch in report["positive_successors"]
                    if branch["branch_index"] in matching)
        status="UNCHECKABLE" if reason else "WITHIN_PROJECTED_NATIVE_SUPPORT" if matching else "OUTSIDE_PROJECTED_NATIVE_SUPPORT"
        if status!="WITHIN_PROJECTED_NATIVE_SUPPORT" and first_gap is None:first_gap=ordinal
        rows.append(dict(ordinal=ordinal,status=status,uncheckable_reason=reason,matching_branch_indices=matching,
            matching_native_percentage=mass,checked_positive_native_branches=0 if reason else len(report["positive_successors"]),
            positive_native_branches=None if report is None else len(report["positive_successors"]),
            native_support_sha256=None if report is None else digest(report),
            reference_snapshot_sha256=observed["reference_snapshot_sha256"],
            original_branch_on_damage=None if report is None else report["captured_support"]["captured_edge"]["branch_on_damage"],
            scoped_support_prefix_checked=first_gap is None,first_uncertain_or_outside_support_ordinal=first_gap,
            compared_endpoint_fields=list(ENDPOINT_FIELDS),native_pp_class="engine_pp_model",
            prestate_equality_scope="required endpoint projection only; unchecked hidden fields may differ",
            matching_mass_is_cross_engine_probability=False,independent_physics_validated=False,
            branch_transition_validated=False,equal_rng_coupling=False,full_engine_equivalence=False,
            runtime_qualified=False,scientific_collection_authorized=False))
    require(before==digest(dict(observations=observations,native=native_boundaries,support=support_rows)),
        "projected successor inputs changed during comparison")
    return rows


def validate_captured_edge(step):
    """Validate a prospective record without guessing historical action slots."""
    edge = deepcopy(step.get("native_edge"))
    require(type(edge) is dict and set(edge) == EDGE_FIELDS and edge["schema"] == EDGE_SCHEMA,
        "prospective exact native edge required")
    require(type(step.get("depth")) is int and step["depth"] >= 0
        and type(step.get("key")) is list and len(step["key"]) == 2
        and all(type(v) is int and v >= 0 for v in step["key"])
        and type(edge["root"]) is bool and edge["root"] is (step["depth"] == 0)
        and type(edge["branch_index"]) is int and edge["branch_index"] == step["key"][1]
        and type(edge["branch_on_damage"]) is bool
        and edge["branch_on_damage"] is step["branch_on_damage"],
        "captured edge coordinate/law drift")
    require(type(edge["instruction_count"]) is int and 0 <= edge["instruction_count"] <= 65536
        and type(edge["instructions_debug"]) is str and bool(edge["instructions_debug"])
        and len(edge["instructions_debug"].encode()) <= 8388608
        and edge["instructions_debug_sha256"] == _state_sha(edge["instructions_debug"]),
        "captured edge instruction binding drift")
    require(type(edge["actions"]) is dict and set(edge["actions"]) == {"p1", "p2"},
        "captured edge joint action required")
    for seat, field in (("p1", "side_one_choice"), ("p2", "side_two_choice")):
        action = edge["actions"][seat]
        _choice(action)
        require({k: action[k] for k in ("kind", "engine_id")} == step[field],
            "captured action slot/name drift")
    return edge


def _unique_object(pairs):
    require(len({k for k, _ in pairs}) == len(pairs), "duplicate native support JSON fields")
    return dict(pairs)


def read_native_edge_support(raw, step):
    """Exact helper export and internal laws; callable origin is separately bound."""
    require(type(raw) is str and len(raw.encode()) <= 8388608, "bounded native support JSON required")
    value = json.loads(raw, object_pairs_hook=_unique_object)
    edge = validate_captured_edge(step)
    require(type(value) is dict and set(value) == SUPPORT_FIELDS and value["schema"] == SUPPORT_SCHEMA
        and value["before_utf8_sha256"] == _state_sha(step["before"])
        and value["after_utf8_sha256"] == _state_sha(step["after"])
        and digest(value["captured_edge"]) == digest(edge)
        and value["private_controller_only"] is True
        and all(value[k] is False for k in FALSE_FLAGS), "native support binding/scope drift")
    names = ("prestate_terminal", "joint_action_admissible", "successor_in_native_support",
        "recorded_branch_instructions_match", "recorded_edge_supported")
    require(all(type(value[k]) is bool for k in names), "native support boolean fact drift")
    count, indices = value["regenerated_branches"], value["matching_successor_branch_indices"]
    require(type(count) is int and 0 <= count <= 4096 and type(indices) is list
        and all(type(v) is int and 0 <= v < count for v in indices)
        and indices == sorted(set(indices)), "native support branch roster drift")
    admissible = value["joint_action_admissible"]
    require((not value["prestate_terminal"] or not admissible)
        and (count > 0) is admissible
        and value["successor_in_native_support"] is bool(indices)
        and (not value["recorded_branch_instructions_match"] or (admissible and edge["branch_index"] < count))
        and value["recorded_edge_supported"] is
            (edge["branch_index"] in indices and value["recorded_branch_instructions_match"]),
        "native support transition law drift")
    return value


def extract_native_edge_support(capture, *, root_binding, selection_report_sha256, native_support=None):
    """Retain every edge, including historical/unavailable evidence, without replay."""
    require(native_support is None or callable(native_support), "bound native support helper required")
    saved = deepcopy(capture)
    bindings = boundary_bindings(saved, root_binding=root_binding,
        selection_report_sha256=selection_report_sha256)
    rows = []
    for binding, step in zip(bindings[1:], saved["steps"]):
        report = None
        if "native_edge" in step and native_support is not None:
            edge = validate_captured_edge(step)
            raw = native_support(step["before"], step["after"], json.dumps(edge, sort_keys=True))
            report = read_native_edge_support(raw, step)
        rows.append(dict(ordinal=binding["ordinal"], binding=binding, support_report=report,
            status="UNCHECKABLE" if report is None else
                "SUPPORTED" if report["recorded_edge_supported"] else "UNSUPPORTED",
            independent_physics_validated=False, runtime_qualified=False,
            scientific_collection_authorized=False))
    require(digest(capture) == digest(saved), "captured ancestry mutated during native support extraction")
    return rows


def observe_supported_prefix(boundary_observations, support_rows):
    """Observed PRE/action/AFTER agreement with monotone uncertainty.

    This is not causal trajectory or simulator equivalence. An unknown or
    divergent earlier coordinate prevents later endpoint matches from restoring
    an observed prefix. Branch-transition physics remains explicitly unchecked.
    """
    require(type(boundary_observations) is list and 2 <= len(boundary_observations) <= 65
        and type(support_rows) is list and len(support_rows) == len(boundary_observations)-1,
        "full original transition denominator required")
    endpoint_fields = ("active_identities", "hp", "fainted", "status", "effective_legal_choices")
    result, first_gap, previous_endpoint_match = [], None, False
    for ordinal, observation in enumerate(boundary_observations):
        require(type(observation["ordinal"]) is int and observation["ordinal"] == ordinal,
            "observed prefix coordinate drift")
        fields = observation["observed_required_fields"]
        require(all(fields.get(k) in {"MATCH", "MISMATCH", "UNCHECKABLE"} for k in endpoint_fields),
            "required endpoint observations missing")
        endpoint_match = all(fields[k] == "MATCH" for k in endpoint_fields)
        native_status = "NOT_APPLICABLE"
        local_match = endpoint_match
        reason = None if endpoint_match else "required_endpoint_mismatch_or_unknown"
        if ordinal:
            support = support_rows[ordinal-1]
            require(type(support["ordinal"]) is int and support["ordinal"] == ordinal
                and support["status"] in {"SUPPORTED", "UNSUPPORTED", "UNCHECKABLE"},
                "native support coordinate/status drift")
            native_status = support["status"]
            action_match = fields.get("action_identity") == "MATCH"
            local_match = previous_endpoint_match and endpoint_match and action_match and native_status == "SUPPORTED"
            if reason is None and not previous_endpoint_match: reason = "prestate_mismatch_or_unknown"
            if reason is None and not action_match: reason = "acknowledged_action_mismatch_or_unknown"
            if reason is None and native_status != "SUPPORTED": reason = "native_edge_unsupported_or_unknown"
        if not local_match and first_gap is None: first_gap = ordinal
        result.append(dict(ordinal=ordinal, native_edge_status=native_status,
            local_required_pre_action_after_match=local_match, observed_prefix_match=first_gap is None,
            first_uncertain_or_divergent_ordinal=first_gap, local_gap_reason=reason,
            independently_validated=False, branch_transition_validated=False,
            matched_causal_trajectory=False, runtime_qualified=False, scientific_collection_authorized=False))
        previous_endpoint_match = endpoint_match
    return result
