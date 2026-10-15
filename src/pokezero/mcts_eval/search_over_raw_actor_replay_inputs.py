"""Actor-specific suffixes and own requests from retained replay snapshots.

This does not reuse the materialized shell's pre-clone history. The independent
transport reader first joins every raw snapshot to its received byte hash and
actual generation. Exact append-only prefix checks precede seat filtering.
Owning process and construction/source joins remain separate prerequisites.
"""
from copy import deepcopy
import json

from .search_over_raw import digest, require
from .search_over_raw_boundary_transport import read_owned_boundary_transport
from .search_over_raw_source_public_history import channel_lines, _filter_binding

SCHEMA = "pokezero.search-over-raw.constructed-actor-replay-inputs.v1"


def read_actor_replay_inputs(capture, *, process_records, initial_snapshot_json,
        clone_binding, request_ids, actor, runtime):
    """Private read-only transport projection; never terminal labels or actors.

    ``process_records`` must be supplied by the independent owning reader. This
    function proves only their transport consistency. All root/reached slots
    remain present, including waits and terminals without actionable requests.
    """
    require(type(actor) is str and actor in {"p1", "p2"}, "one exact constructed actor required")
    before = digest(dict(capture=capture, records=process_records, initial=initial_snapshot_json,
        clone=clone_binding, requests=request_ids, actor=actor, runtime=runtime))
    filter_binding = _filter_binding(runtime)
    trace = read_owned_boundary_transport(capture, process_records=process_records,
        initial_snapshot_json=initial_snapshot_json, clone_binding=clone_binding, request_ids=request_ids)
    initial = json.loads(initial_snapshot_json)
    previous = initial["battle"].get("log")
    # Validate split boundaries at the initial cut even though the shell prefix
    # is discarded. It may contain information unavailable at the source root.
    channel_lines(previous, channel=0)
    rows = []
    for index, (received, observed) in enumerate(zip(capture["snapshots"], trace["observed_snapshots"])):
        raw = json.loads(received["currentSnapshotJson"])
        log = raw["battle"].get("log")
        channel_lines(log, channel=0)
        require(len(log) >= len(previous) and log[:len(previous)] == previous,
            "LABEL_INPUT_UNAVAILABLE: replay rewrote or truncated its prior log")
        suffix = log[len(previous):]
        require(index != 0 or suffix == [], "clone root log differs from original frozen shell")
        lines = channel_lines(suffix, channel=int(actor[-1]))
        request = raw["boundaryRequests"].get(actor)
        require(request is None or type(request) is dict, "constructed own request malformed")
        if request is not None and "side" in request:
            require(type(request["side"]) is dict and request["side"].get("id") == actor,
                "constructed own request belongs to another seat")
        rows.append(dict(request_id=observed["request_id"], snapshot_index=observed["snapshot_index"],
            observed_generation=observed["observed_generation"], boundary_kind=observed["boundary_kind"],
            snapshot_sha256=observed["snapshot_sha256"],
            current_snapshot_json_sha256=observed["current_snapshot_json_sha256"],
            prior_log_sha256=digest(previous), appended_log_sha256=digest(suffix),
            visible_suffix=lines, visible_suffix_sha256=digest(lines),
            own_request=deepcopy(request), own_request_sha256=digest(request),
            actor_requested=actor in observed["requested"]))
        previous = log
    require(before == digest(dict(capture=capture, records=process_records, initial=initial_snapshot_json,
        clone=clone_binding, requests=request_ids, actor=actor, runtime=runtime))
        and _filter_binding(runtime) == filter_binding, "actor replay inputs changed during projection")
    return dict(schema=SCHEMA, actor=actor, channel=int(actor[-1]),
        capture_sha256=trace["capture_sha256"], transport_sha256=trace["transport_sha256"],
        initial_snapshot_sha256=digest(initial), filter_binding=filter_binding, rows=rows,
        materialized_shell_prefix_used=False, original_source_private_requests_used=False,
        snapshot_transport_consistent=True, process_ownership_validated=False,
        construction_provenance_validated=False, source_prefix_join_validated=False,
        full_actor_observation_validated=False, independent_terminal_labels=False, runtime_qualified=False)
