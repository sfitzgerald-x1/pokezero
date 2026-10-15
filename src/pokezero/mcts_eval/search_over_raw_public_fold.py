"""Prospective full public folds at actual native constructions, never labels.

Materialization overlays cannot reconstruct history-dependent observations.
Keep the exact sanitized request, including pending context, separately from
the draw and overlay. Private source snapshots and set catalogs are excluded.
An internal receipt is not owning-process validation or collection admission.
"""
from copy import deepcopy
import hashlib
from pathlib import Path

from .search_over_raw import digest, require

SCHEMA = "pokezero.search-over-raw.native-construction-public-fold.v1"
MAXIMUM_BYTES = 8 * 1024 * 1024


def source_hashes():
    directory = Path(__file__).resolve().parent
    paths = [Path(__file__).resolve(), directory / "search_over_raw_world_inputs.py",
        directory / "paper_reference_runtime.py", directory.parent / "local_showdown.py"]
    return {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}


def runtime_receipt():
    return dict(schema=SCHEMA, maximum_payload_bytes=MAXIMUM_BYTES,
        at_actual_attempt_and_constructed_callbacks=True, pending_transition_included=True,
        private_source_snapshot_included=False, catalog_included=False,
        labels_produced=False, instrumentation_can_change_selection=True,
        source_hashes=source_hashes(), runtime_qualified=False)


def capture_request(context, pending_transition):
    from .paper_reference_runtime import PublicRootRequest
    from .search_over_raw_world_inputs import _state_payload, public_request_witness
    import json
    request = PublicRootRequest.capture(context.public_materialization_state,
        context.observation, pending_transition=pending_transition)
    payload = _state_payload(request)
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"),
        ensure_ascii=False, allow_nan=False).encode()
    require(len(encoded) <= MAXIMUM_BYTES, "construction public fold retention cap exceeded")
    return dict(public_request_payload=payload, witness=public_request_witness(request))


def construction_capture(attempt, constructed, *, ordinal, construction_receipt):
    require(type(ordinal) is int and 0 <= ordinal < 4 and type(attempt) is dict
        and digest(attempt) == digest(constructed),
        "actual construction public fold changed after sampling attempt")
    return dict(schema=SCHEMA, construction_ordinal=ordinal,
        construction_capture_sha256=digest(construction_receipt),
        public_request_payload=deepcopy(attempt["public_request_payload"]),
        witness=deepcopy(attempt["witness"]), source_hashes=source_hashes(),
        private_controller_only=True, attempt_and_constructed_match=True,
        labels_produced=False, process_ownership_validated=False, runtime_qualified=False)


def read_construction_public_fold(capture, *, ordinal, construction_receipt,
        consumed_binding, public_request_sha256, expected_request=None):
    """Exact independently supplied original bindings; no historical upgrade.

    An optional typed request must match ALL captured fields, not just a compact
    root identity. A later actor reconstructs from that exact request only after
    separate owning-run validation; no materialization projection is decoded.
    """
    from .search_over_raw_world_inputs import _state_payload
    import json
    before = digest(capture)
    require(type(ordinal) is int and 0 <= ordinal < 4 and type(capture) is dict
        and set(capture) == {"schema", "construction_ordinal", "construction_capture_sha256",
            "public_request_payload", "witness", "source_hashes", "private_controller_only",
            "attempt_and_constructed_match", "labels_produced", "process_ownership_validated", "runtime_qualified"}
        and capture["schema"] == SCHEMA and type(capture["construction_ordinal"]) is int
        and capture["construction_ordinal"] == ordinal
        and capture["construction_capture_sha256"] == digest(construction_receipt)
        and capture["private_controller_only"] is True and capture["attempt_and_constructed_match"] is True
        and all(capture[k] is False for k in ("labels_produced", "process_ownership_validated", "runtime_qualified"))
        and capture["source_hashes"] == source_hashes(), "exact actual construction public-fold receipt required")
    payload = capture["public_request_payload"]
    # The digest is the observed full sanitized transport, not the small root
    # binding. Its native owning reader supplies the expected value externally.
    require(type(payload) is dict and payload.get("dataclass") ==
        "pokezero.mcts_eval.paper_reference_runtime.PublicRootRequest"
        and digest(payload) == public_request_sha256
        and len(json.dumps(payload, sort_keys=True, separators=(",", ":"),
            ensure_ascii=False, allow_nan=False).encode()) <= MAXIMUM_BYTES,
        "construction public payload differs from original full request")
    witness = dict(schema="pokezero.search-over-raw.public-request-consumption.v1",
        public_request_sha256=public_request_sha256, consumed_binding=consumed_binding,
        full_supported_public_state_and_pending=True, private_source_snapshot_included=False, runtime_qualified=False)
    require(digest(capture["witness"]) == digest(witness), "construction public witness drift")
    if expected_request is not None:
        from .paper_reference_runtime import PublicRootRequest
        require(type(expected_request) is PublicRootRequest and expected_request.state.replay.requests == {}
            and expected_request.state.belief_engine.set_source is None
            and digest(_state_payload(expected_request)) == public_request_sha256,
            "original typed consumed public fold required; root-only substitutes are insufficient")
    require(digest(capture) == before, "construction public fold changed during read")
    return dict(schema=SCHEMA, capture_sha256=before, public_request_sha256=public_request_sha256,
        construction_ordinal=ordinal, full_public_payload_bound=True,
        process_ownership_validated=False, labels_produced=False, runtime_qualified=False)
