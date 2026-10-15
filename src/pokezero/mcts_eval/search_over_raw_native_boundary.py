"""Read actual native boundary exports, not hand-authored projection dictionaries.

This reader checks structure/bindings and invokes a caller's bound native
extractor. It does NOT independently certify native semantics, party mapping,
Showdown ownership, branch correspondence, or scientific runtime qualification.
The private projections must remain in the trusted post-selection controller.
"""
from copy import deepcopy
import json

from .search_over_raw import digest, require
from .search_over_raw_correspondence import boundary_bindings, _state_sha
from .search_over_raw_native_boundary_records import (
    SCHEMA, SEATS, SIDE_KEYS, FLAGS, _identity, _choice,
    read_native_boundary, project_native_boundary,
)



def _historical_action(side, captured, *, preview=False):
    """Resolve ONLY unique full-catalog identities; never first-species parsing."""
    kind, engine_id = captured["kind"], captured["engine_id"]
    if kind == "move":
        candidates = [dict(kind=kind,slot=v["slot"],engine_id=v["engine_id"])
            for v in side["active_moves"] if v["engine_id"] == engine_id]
    elif kind == "switch":
        candidates = [dict(kind=kind,slot=v["slot"],engine_id=v["engine_id"])
            for v in side["party_identities"] if v["engine_id"] == engine_id
            and (preview or v["slot"] != side["active_identity"]["slot"])]
    else:
        candidates = [dict(kind=kind,slot=None,engine_id=engine_id)]
    return candidates[0] if len(candidates) == 1 and candidates[0] in side["search_arms"] else None




def extract_native_capture_boundaries(capture, *, root_binding, selection_report_sha256,
                                      native_extract):
    """Bound partial projections at every actual root/after-state coordinate.

    native_extract is the frozen runtime's native_scoped_boundary callable, NOT
    an evaluator or replay callback. These are not accepted boundary receipts:
    action ambiguity and transition/party-join validation remain explicit.
    """
    require(callable(native_extract), "actual bound native extractor required")
    saved = deepcopy(capture)
    bindings = boundary_bindings(saved,root_binding=root_binding,
        selection_report_sha256=selection_report_sha256)
    states = [saved["root_native_state"]] + [v["after"] for v in saved["steps"]]
    reports = []
    for ordinal, state in enumerate(states):
        raw = native_extract(state, ordinal == 0)
        require(type(raw) is str, "native extractor must return its actual JSON export")
        reports.append(read_native_boundary(json.loads(raw),state_text=state,root=ordinal == 0))
    rows = []
    for ordinal, (binding, report) in enumerate(zip(bindings,reports)):
        native = project_native_boundary(report)
        if ordinal:
            step = saved["steps"][ordinal-1]
            previous = reports[ordinal-1]
            actions = {seat:_historical_action(previous[seat],step[key],
                preview=previous["root"] and previous["team_preview"])
                for seat,key in (("p1","side_one_choice"),("p2","side_two_choice"))}
            if "native_edge" in step:
                # Prospective slots distinguish duplicate identities; never
                # infer a slot for a historical name-only action.
                from .search_over_raw_native_edge import validate_captured_edge
                edge=validate_captured_edge(step)
                actions={seat:deepcopy(edge["actions"][seat]) if edge["actions"][seat]
                    in previous[seat]["search_arms"] else None for seat in SEATS}
            if all(v is not None for v in actions.values()): native["action_identity"] = actions
        native["branch_transition"] = None  # Raw flags are NOT independently matched requests/transition.
        rows.append(dict(ordinal=ordinal,binding=binding,native=native,native_export_sha256=digest(report),
            independently_validated=False,party_mapping_validated=False,
            runtime_qualified=False,scientific_collection_authorized=False))
    require(digest(capture) == digest(saved), "captured native path changed during extraction")
    return rows
