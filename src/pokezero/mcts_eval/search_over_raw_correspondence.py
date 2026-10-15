"""Scoped reconciliation of actual searched ancestry and independent boundaries.

The private capture stays in the trusted post-selection controller. This is not
a replay runner and opens no environment. Boundary validators must verify the
actual native projection and independent Showdown receipt; hashes alone do not
prove either. In particular, chosen-action legality is NOT a full legal set.
Historical fixture failures are not modified or promoted by this new schema.
"""
from copy import deepcopy
import hashlib
import math

from .search_over_raw import digest, require

SCHEMA = "pokezero.search-over-raw.scoped-searched-line.v1"
REQUIRED_FIELDS = ("active_identities", "hp", "fainted", "status", "action_identity",
    "effective_legal_choices", "branch_transition")
KNOWN_CLASSES = {"native_pp": "engine_pp_model"}


def _sha(value):
    return type(value) is str and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def _state_sha(text):
    require(type(text) is str and bool(text), "captured serialized state required")
    return hashlib.sha256(text.encode()).hexdigest()


def boundary_bindings(capture, *, root_binding, selection_report_sha256):
    """Every root/transition coordinate from the ACTUAL native ancestry.

    No regenerated branch, dropped prefix or inferred endpoint is accepted.
    Receipt validators additionally bind this capture to its original selection.
    """
    require(_sha(root_binding) and _sha(selection_report_sha256)
        and type(capture) is dict and set(capture) == {"root_native_state", "steps",
            "native_frontier", "model_signed_value", "evaluation_ordinal", "private_controller_only"}
        and capture["private_controller_only"] is True
        and type(capture["evaluation_ordinal"]) is int and capture["evaluation_ordinal"] >= 0
        and type(capture["model_signed_value"]) in (int, float)
        and math.isfinite(capture["model_signed_value"])
        and -1 <= capture["model_signed_value"] <= 1,
        "bound private native capture required")
    steps = capture["steps"]
    require(type(steps) is list and 0 < len(steps) <= 64, "actual bounded searched ancestry required")
    shared = dict(capture_sha256=digest(capture), root_binding=root_binding,
        selection_report_sha256=selection_report_sha256,
        evaluation_ordinal=capture["evaluation_ordinal"], ancestry_sha256=digest(steps))
    root_sha = _state_sha(capture["root_native_state"])
    bindings = [dict(shared, ordinal=0, before_sha256=root_sha, after_sha256=root_sha,
        chosen_actions_sha256=digest(None))]
    expected = capture["root_native_state"]
    parent, seen = None, set()
    for ordinal, step in enumerate(steps, 1):
        fields={"key", "parent", "depth", "before", "after", "branch_on_damage", "side_one_choice", "side_two_choice"}
        require(type(step) is dict and set(step) in (fields,fields | {"native_edge"}), "captured step field drift")
        key = step["key"]
        require(type(key) is list and len(key) == 2 and all(type(v) is int and v >= 0 for v in key)
            and tuple(key) not in seen and digest(step["parent"]) == digest(parent) and step["before"] == expected
            and type(step["depth"]) is int and step["depth"] >= 0
            and type(step["branch_on_damage"]) is bool, "captured ancestry identity/continuity drift")
        actions = {seat: step[field] for seat, field in (
            ("side_one", "side_one_choice"), ("side_two", "side_two_choice"))}
        require(all(type(choice) is dict and set(choice) == {"kind", "engine_id"}
            and choice["kind"] in {"move", "switch", "struggle", "none"}
            and type(choice["engine_id"]) is str and bool(choice["engine_id"])
            for choice in actions.values()), "actual captured joint actions required")
        if "native_edge" in step:
            from .search_over_raw_native_edge import validate_captured_edge
            validate_captured_edge(step)
        bindings.append(dict(shared, ordinal=ordinal, before_sha256=_state_sha(step["before"]),
            after_sha256=_state_sha(step["after"]), chosen_actions_sha256=digest(actions)))
        expected, parent = step["after"], key
        seen.add(tuple(key))
    require(expected == capture["native_frontier"], "captured path endpoint drift")
    return bindings


def reconcile_scoped_searched_line(capture, boundaries, *, root_binding,
                                   selection_report_sha256, validate_boundary):
    """Compare only required direct-check fields, retaining every coordinate.

    A validator returns the canonical boundary receipt digest, or None when it
    cannot independently verify its native/reference provenance and semantics.
    Missing fields remain UNCHECKABLE even in an otherwise validated receipt.
    Both projections must be canonical actual effective fields, not inferred
    from the opposing engine. Differences are observations, not causal proof.
    """
    require(callable(validate_boundary) and type(boundaries) is list,
        "independent boundary receipt validator required")
    bindings = boundary_bindings(capture, root_binding=root_binding,
        selection_report_sha256=selection_report_sha256)
    capture_before = digest(capture)
    saved = deepcopy(boundaries)
    indexed = {}
    for row in saved:
        require(type(row) is dict and set(row) == {"ordinal", "binding", "native", "showdown"}
            and type(row["ordinal"]) is int and 0 <= row["ordinal"] < len(bindings)
            and row["ordinal"] not in indexed, "duplicate or unexpected direct-check boundary")
        ordinal = row["ordinal"]
        require(digest(row["binding"]) == digest(bindings[ordinal])
            and all(type(row[engine]) is dict and set(row[engine]) <= set(REQUIRED_FIELDS) | set(KNOWN_CLASSES)
                for engine in ("native", "showdown")), "direct-check capture/projection binding drift")
        digest(row)  # Reject nonfinite/nonserializable evidence, even if unvalidated.
        indexed[ordinal] = row
    results = []
    for expected in bindings:
        ordinal = expected["ordinal"]
        row = indexed.get(ordinal)
        before = digest(row) if row is not None else None
        witness = None if row is None else validate_boundary(deepcopy(row), deepcopy(expected))
        require(witness is None or (_sha(witness) and witness == before),
            "boundary validator must bind the exact actual receipt")
        fields, known = {}, []
        for field in REQUIRED_FIELDS:
            checked = (witness is not None and row["native"].get(field) is not None
                and row["showdown"].get(field) is not None)
            fields[field] = ("MATCH" if digest(row["native"][field]) == digest(row["showdown"][field])
                else "MISMATCH") if checked else "UNCHECKABLE"
        if witness is not None:
            for field, classification in KNOWN_CLASSES.items():
                if row["native"].get(field) is not None and row["showdown"].get(field) is not None \
                        and digest(row["native"][field]) != digest(row["showdown"][field]):
                    known.append(dict(component=field, classification=classification,
                        native_sha256=digest(row["native"][field]),
                        showdown_sha256=digest(row["showdown"][field]), required_field=False))
        results.append(dict(ordinal=ordinal, binding=expected, receipt_sha256=before,
            independently_validated=witness is not None, required_fields=fields,
            documented_differences=known))
    require(digest(boundaries) == digest(saved) and digest(capture) == capture_before,
        "direct-check evidence changed during validation")
    mismatched = sum(v == "MISMATCH" for row in results for v in row["required_fields"].values())
    unknown = sum(v == "UNCHECKABLE" for row in results for v in row["required_fields"].values())
    return dict(schema=SCHEMA, root_binding=root_binding, capture_sha256=capture_before,
        selection_report_sha256=selection_report_sha256, ancestry_sha256=digest(capture["steps"]),
        path_boundaries=len(capture["steps"]), required_boundary_records=len(bindings),
        recorded_boundary_records=len(indexed), boundaries=results,
        required_field_mismatches=mismatched, uncheckable_required_fields=unknown,
        status="REQUIRED_FIELDS_MATCHED" if not mismatched and not unknown else "LIMITED_COMPARISON",
        required_fields=list(REQUIRED_FIELDS), native_pp_class="engine_pp_model",
        original_failure_reclassified=False, full_engine_equivalence=False,
        exact_rng_coupling_claim=False, causal_mechanism_established=False,
        searched_line_runtime_qualified=False, scientific_collection_authorized=False)
