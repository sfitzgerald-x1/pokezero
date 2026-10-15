"""Private read-only boundary transport for one never-reused owned clone.

This is a distinct contract from eight-replicate continuation banks. Raw
snapshots remain controller-private; the separately observed process trace
contains byte hashes only. Transport consistency is not process ownership,
construction provenance, transition semantics, or runtime qualification.
"""
from contextlib import contextmanager
from copy import deepcopy
import hashlib
import json
import re

from .search_over_raw import digest, require
from .search_over_raw_outcome_run import _redact

SCHEMA = "pokezero.search-over-raw.owned-boundary-transport.v1"
SEATS = ("p1", "p2")


def _identity(value):
    return type(value) is str and re.fullmatch(r"[A-Za-z0-9._:-]{1,200}", value) is not None


def _sha(text):
    require(type(text) is str, "actual private JSON bytes required")
    return hashlib.sha256(text.encode()).hexdigest()


def _requests(snapshot):
    require(type(snapshot) is dict and set(snapshot) == {"battle", "boundaryRequests", "terminalScheduled"}
        and type(snapshot["battle"]) is dict and type(snapshot["battle"].get("ended")) is bool
        and type(snapshot["terminalScheduled"]) is bool and type(snapshot["boundaryRequests"]) is dict
        and set(snapshot["boundaryRequests"]) <= set(SEATS), "paired serialized boundary required")
    if snapshot["battle"]["ended"]:
        require(snapshot["boundaryRequests"] == {}, "terminal snapshot cannot advertise actors")
        return []
    require(snapshot["terminalScheduled"] is False and set(snapshot["boundaryRequests"]) == set(SEATS),
        "nonterminal paired requests missing or unsettled")
    requested = []
    for seat in SEATS:
        row = snapshot["boundaryRequests"][seat]
        require(type(row) is dict, "actual paired request object required")
        if row.get("wait") or row.get("teamPreview"):
            continue
        if (type(row.get("forceSwitch")) is list and any(row["forceSwitch"])) \
                or (type(row.get("active")) is list and row["active"]):
            requested.append(seat)
    require(requested, "nonterminal snapshot has no observed actor")
    return requested


def _binding(binding):
    require(type(binding) is dict and set(binding) == {"battleId", "cloneId", "snapshotJsonSha256"}
        and _identity(binding["battleId"]) and _identity(binding["cloneId"])
        and type(binding["snapshotJsonSha256"]) is str
        and re.fullmatch(r"[a-f0-9]{64}", binding["snapshotJsonSha256"]) is not None,
        "exact owned clone binding required")


class BoundaryTransportCapture:
    """Tap actual send/read seams; do not publish raw snapshots to actors.

    Installing this tap owns neither the bridge process nor its cleanup. Its
    export must be joined to a separate process trace and owning-run receipt.
    """
    schema = SCHEMA
    maximum_snapshots = 65

    def __init__(self, *, clone_binding, request_ids):
        _binding(clone_binding)
        require(type(request_ids) is list and 1 <= len(request_ids) <= self.maximum_snapshots
            and all(_identity(value) for value in request_ids) and len(set(request_ids)) == len(request_ids),
            "prospective root and reached-boundary request roster required")
        self.binding, self.request_ids = deepcopy(clone_binding), list(request_ids)
        self.env, self.records, self.snapshots, self.used = None, [], [], False
        self.failed = False
        self.frozen_inputs = digest(dict(binding=self.binding,request_ids=self.request_ids))

    @contextmanager
    def installed(self, env):
        from ..local_showdown import LocalShowdownEnv
        require(isinstance(env, LocalShowdownEnv) and env._battle_token is None and not self.used
            and getattr(env,"_bridge_boundary_observer",None) is None,
            "empty isolated environment and unused private boundary tap required")
        self.used, self.env = True, env
        env._bridge_boundary_observer = self
        try:
            yield self
        finally:
            require(env._bridge_boundary_observer is self, "boundary tap replaced during run")
            env._bridge_boundary_observer, self.env = None, None

    def _check(self):
        require(digest(dict(binding=self.binding,request_ids=self.request_ids)) == self.frozen_inputs,
            "boundary capture input drift")

    def command(self, payload):
        self._check()
        self.records.append(_redact("command",payload))

    def event(self, payload):
        self._check()
        if payload.get("type") == "owned_boundary_snapshot":
            self.snapshots.append(deepcopy(dict(payload)))
        self.records.append(_redact("event",payload))

    def snapshot(self):
        """Read the next predeclared boundary; no retry, restore or RNG work."""
        self._check()
        require(not self.failed and self.env is not None and self.env._battle_token == self.binding["battleId"],
            "actual active owned token required")
        sent = sum(row["kind"] == "command" and row["payload"].get("type") == "snapshot_owned_boundary"
            for row in self.records)
        require(sent < len(self.request_ids), "declared boundary roster exhausted")
        command = dict(type="snapshot_owned_boundary",**self.binding,requestId=self.request_ids[sent])
        try:
            event = self.env._bridge_request_event(command,"owned_boundary_snapshot")
            require(type(event) is dict and all(event.get(key) == value for key,value in self.binding.items())
                and event.get("requestId") == command["requestId"] and type(event.get("snapshotIndex")) is int
                and event["snapshotIndex"] == sent and _sha(event.get("currentSnapshotJson"))
                    == event.get("currentSnapshotJsonSha256"), "owned boundary response drift")
            raw = json.loads(event["currentSnapshotJson"])
            _requests(raw)
            return raw  # Controller-private input for the scoped projection reader.
        except BaseException:
            self.failed = True
            raise

    def export(self):
        self._check()
        require(self.env is None and not self.failed,
            "failed observation cannot export even after a late acknowledgement")
        return dict(schema=self.schema,clone_binding=deepcopy(self.binding),request_ids=list(self.request_ids),
            records=deepcopy(self.records),snapshots=deepcopy(self.snapshots),private_controller_only=True)


def read_owned_boundary_transport(capture, *, process_records, initial_snapshot_json,
                                 clone_binding, request_ids):
    """Reconcile separately observed commands/events with every raw snapshot.

    Caller inputs are frozen independently, not inferred from the capture.
    This checks ONE released clone, including root and each actual acknowledged
    reached boundary. It cannot prove a complete native path was reached, nor
    promote a caller's construction map or an unowned process trace.
    """
    return _read_owned_boundary_transport(capture, process_records=process_records,
        initial_snapshot_json=initial_snapshot_json, clone_binding=clone_binding, request_ids=request_ids,
        schema=SCHEMA, maximum_snapshots=65)


def _read_owned_boundary_transport(capture, *, process_records, initial_snapshot_json,
        clone_binding, request_ids, schema, maximum_snapshots):
    """Internal parser; public entry points freeze their own schema and limit."""
    _binding(clone_binding)
    saved = deepcopy(capture)
    before = digest(dict(capture=capture,records=process_records,binding=clone_binding,
        requests=request_ids,initial=initial_snapshot_json))
    require(type(request_ids) is list and 1 <= len(request_ids) <= maximum_snapshots
        and all(_identity(v) for v in request_ids) and len(set(request_ids)) == len(request_ids),
        "external boundary request roster required")
    require(type(saved) is dict and set(saved) == {"schema","clone_binding","request_ids","records",
        "snapshots","private_controller_only"} and saved["schema"] == schema
        and saved["private_controller_only"] is True and saved["clone_binding"] == clone_binding
        and saved["request_ids"] == request_ids and type(saved["records"]) is list
        and saved["records"] == process_records and type(saved["snapshots"]) is list
        and len(saved["snapshots"]) == len(request_ids), "exact private/process transport join required")
    require(_sha(initial_snapshot_json) == clone_binding["snapshotJsonSha256"], "external frozen clone bytes drift")
    initial = json.loads(initial_snapshot_json)
    requested = _requests(initial)
    require(initial["battle"]["ended"] is False, "nonterminal clone root required")
    token = clone_binding["battleId"]
    expected_clone = dict(type="clone_snapshot",battleId=token,cloneId=clone_binding["cloneId"],
        snapshotJsonSha256=clone_binding["snapshotJsonSha256"])
    stage, generation, snapshots, pending, acked, choices = "empty", 0, [], None, set(), None
    ends, reseed, boundary_kind = set(), None, None
    for row in process_records:
        require(type(row) is dict and set(row) == {"kind","payload"} and row["kind"] in {"command","event"}
            and type(row["payload"]) is dict, "actual boundary transport row required")
        payload, kind = row["payload"], row["kind"]
        tag = payload.get("type")
        require(payload.get("battleId") == token, "foreign or post-run transport is not this clone")
        if kind == "command":
            if tag == "clone_snapshot":
                require(stage == "empty" and payload == expected_clone, "one exact initial clone command required")
                stage = "cloning"
            elif tag == "snapshot_owned_boundary":
                index = len(snapshots)
                require(stage == "boundary" and pending is None and reseed is None and index < len(request_ids)
                    and payload == dict(type=tag,**clone_binding,requestId=request_ids[index]),
                    "ordered settled owned snapshot command required")
                pending, stage = payload, "snapshot"
            elif tag == "choices":
                require(stage == "observed" and reseed is None and generation + 1 < len(request_ids)
                    and set(payload) == {"type","battleId","choices"} and type(payload["choices"]) is dict
                    and set(payload["choices"]) == set(requested)
                    and all(type(v) is str and v.strip() for v in payload["choices"].values()),
                    "complete declared actor submission after root snapshot required")
                generation += 1
                choices, acked, stage = payload["choices"], set(), "choosing"
            elif tag == "reseed":
                require(stage == "observed" and reseed is None and set(payload) == {"type","battleId","seed"}
                    and type(payload["seed"]) is str and payload["seed"], "explicit settled reseed required")
                reseed = payload["seed"]
            elif tag == "release_clone":
                require(stage == "observed" and reseed is None and len(snapshots) == len(request_ids)
                    and payload == dict(type=tag,**clone_binding), "release only after the full observed roster")
                stage = "releasing"
            else:
                raise ValueError("unreviewed owned-boundary command")
        elif tag == "cloned":
            require(stage == "cloning" and payload == dict(type=tag,**clone_binding,requested=requested),
                "exact clone/root acknowledgement required")
            stage, boundary_kind = "boundary", "root"
        elif tag == "stream":
            require(stage in {"cloning","choosing","boundary","snapshot","observed","releasing"}
                and payload.get("stream") in {"omniscient",*SEATS}, "owned stream outside admitted clone")
            require(payload["stream"] == "omniscient" or stage in {"cloning","choosing","releasing"},
                "late private stream invalidates the settled boundary")
        elif tag == "choice_ack":
            seat = payload.get("player")
            require(stage == "choosing" and seat in choices and seat not in acked
                and payload.get("choice") == choices[seat], "exact unique joint-choice acknowledgement required")
            acked.add(seat)
        elif tag in {"ready","terminal"}:
            require(stage == "choosing" and acked == set(choices), "settled acknowledged transition required")
            if tag == "ready":
                requested = payload.get("requested")
                require(type(requested) is list and requested and len(set(requested)) == len(requested)
                    and requested == [s for s in SEATS if s in requested], "actual ordered boundary actors required")
            else:
                requested = []
            stage, boundary_kind = "boundary", tag
        elif tag == "reseeded":
            require(stage == "observed" and reseed is not None and payload.get("seed") == reseed,
                "same explicit reseed acknowledgement required")
            reseed = None
        elif tag == "owned_boundary_snapshot":
            index = len(snapshots)
            require(stage == "snapshot" and pending is not None and index < len(saved["snapshots"]),
                "snapshot response without its outstanding command")
            raw_event = saved["snapshots"][index]
            require(type(raw_event) is dict and set(raw_event) == {"type",*clone_binding,"requestId",
                "snapshotIndex","boundaryGeneration","currentSnapshotJson","currentSnapshotJsonSha256"}
                and _redact("event",raw_event) == row and all(raw_event.get(k) == v for k,v in clone_binding.items())
                and raw_event["requestId"] == pending["requestId"]
                and type(raw_event["snapshotIndex"]) is int and raw_event["snapshotIndex"] == index
                and type(raw_event["boundaryGeneration"]) is int and raw_event["boundaryGeneration"] == generation,
                "ordered raw snapshot/actual generation/received bytes join required")
            snapshot = json.loads(raw_event["currentSnapshotJson"])
            require(_requests(snapshot) == requested
                and snapshot["battle"]["ended"] is (boundary_kind == "terminal"),
                "serialized boundary differs from observed readiness")
            snapshots.append(dict(request_id=pending["requestId"],snapshot_index=index,
                observed_generation=generation,boundary_kind=boundary_kind,
                current_snapshot_json_sha256=_sha(raw_event["currentSnapshotJson"]),snapshot_sha256=digest(snapshot),
                incoming_choices=deepcopy(choices) if generation else None,requested=list(requested)))
            pending, stage = None, "observed"
        elif tag == "stream_end":
            seat = payload.get("stream")
            require(stage == "releasing" and seat in {"omniscient",*SEATS} and seat not in ends,
                "distinct stream endings during strict release required")
            ends.add(seat)
        elif tag == "clone_released":
            require(stage == "releasing" and ends == {"omniscient",*SEATS}
                and payload == dict(type=tag,**clone_binding), "fully drained exact release required")
            stage = "released"
        else:
            raise ValueError("unreviewed owned-boundary event")
    require(stage == "released" and pending is None and reseed is None
        and len(snapshots) == len(request_ids), "incomplete released boundary trace")
    require(before == digest(dict(capture=capture,records=process_records,binding=clone_binding,
        requests=request_ids,initial=initial_snapshot_json)), "boundary transport inputs changed during read")
    return dict(schema=schema,capture_sha256=digest(saved),transport_sha256=digest(process_records),
        clone_binding=deepcopy(clone_binding),observed_snapshots=snapshots,
        snapshot_transport_consistent=True,process_ownership_validated=False,
        construction_provenance_validated=False,branch_transition_validated=False,
        native_ancestry_coverage_validated=False,independent_terminal_truth=False,
        exact_rng_coupling_claim=False,runtime_qualified=False,scientific_collection_authorized=False)
