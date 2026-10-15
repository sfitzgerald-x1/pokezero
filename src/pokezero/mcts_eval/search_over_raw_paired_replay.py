"""Paired Python/simulator native-path replay, separate from transport-only v2.

No policy is observed during replay. The original materializer context is
restored once; actual stream events advance its parser, requests and belief.
The last context is retained, but an unavailable suffix never supplies the
designated AFTER endpoint. Construction/native ownership remains a caller join.
"""
from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import pickle

from ..local_showdown import LocalShowdownEnv, LocalShowdownSnapshot
from .search_over_raw import digest, require
from .search_over_raw_world_inputs import _snapshot_payload
from .search_over_raw_outcome_run import _redact
from .search_over_raw_boundary_run import UNQUALIFIED, _join_raw_result, RawBoundaryRunValidator
from .search_over_raw_boundary_transport import BoundaryTransportCapture
from .search_over_raw_native_prefix import (
    OwnedNativePrefixBoundaryJob, NativePrefixBoundaryRunContract,
    NativePrefixBoundaryProcessOwner, read_native_prefix_transport, termination_witness,
    _validate_inputs as validate_prefix_inputs, CAPTURE_SCHEMA as PREFIX_SCHEMA,
)
from .search_over_raw_showdown_boundary import NativeReplayUnavailable
from .search_over_raw_supervision import _supervise_owned_job

SCHEMA = "pokezero.search-over-raw.paired-native-prefix.v1"
ARTIFACT = "paired-after-context.pickle"


def replay_law():
    return dict(schema=SCHEMA, initializer="original_retained_paired_materialization",
        transition="LocalShowdownEnv._submit_step_choices", observation_players=[],
        sync_at_each_retained_boundary=True, policy_inference_during_replay=False,
        unavailable_suffix="all_original_unreached_coordinates_uncertain",
        substitute_after_endpoint_permitted=False, retries_permitted=False,
        source_opponent_history_recovered=False, scientific_collection_authorized=False)


def sources():
    paths = (Path(__file__), Path(__file__).with_name("search_over_raw_native_prefix.py"),
        Path(__file__).parents[1]/"local_showdown.py")
    return {str(p.resolve()):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}


def context_identity(snapshot):
    require(type(snapshot) is LocalShowdownSnapshot, "actual retained paired snapshot required")
    return dict(snapshot_sha256=digest(_snapshot_payload(snapshot)),
        battle_token=snapshot.battle_token, battle_id=snapshot.battle_id)


def _prefix_inputs(value):
    return {k:v for k,v in value.items() if k not in
        {"paired_snapshot", "paired_constructor", "paired_replay_law", "paired_replay_sources"}}


def validate_inputs(value):
    validate_prefix_inputs(_prefix_inputs(value))
    require(set(value)==set(_prefix_inputs(value))|{"paired_snapshot","paired_constructor",
        "paired_replay_law","paired_replay_sources"}
        and digest(value["paired_replay_law"])==digest(replay_law())
        and digest(value["paired_replay_sources"])==digest(sources())
        and type(value["paired_snapshot"]) is dict and type(value["paired_constructor"]) is dict
        and value["paired_constructor"].get("snapshot_sha256")==value["paired_snapshot"].get("snapshot_sha256")
        and value["paired_constructor"].get("bridge_snapshot_sha256")==value["raw_root_sha256"],
        "original paired constructor context and replay law required")


def prefix_view(capture):
    require(type(capture) is dict and set(capture)=={"schema","clone_binding","request_ids","records",
        "snapshots","private_controller_only","termination","paired_contexts","replay_streams"}
        and capture["schema"]==SCHEMA, "distinct paired prefix capture required")
    return dict({k:v for k,v in capture.items() if k not in {"paired_contexts","replay_streams"}},schema=PREFIX_SCHEMA)


class PairedPrefixCapture(BoundaryTransportCapture):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.streams, self.contexts = [], []

    def event(self, payload):
        if payload.get("type")=="stream":
            self.streams.append(dict(ordinal=len(self.records),event=deepcopy(dict(payload))))
        super().event(payload)

    def retain(self, raw):
        self.env._sync_incremental_state()
        snapshot=self.env._local_snapshot(bridge_snapshot=deepcopy(raw))
        require(snapshot.terminal is not None or bool(snapshot.latest_requests),
            "nonterminal replay must receive actual paired requests")
        self.contexts.append(context_identity(snapshot))
        return snapshot


def read_paired_prefix(capture, *, process_records, snapshot, expected):
    """Re-fold original, independently observed events; never launch a bridge."""
    validate_inputs(expected)
    require(context_identity(snapshot)==expected["paired_snapshot"]
        and digest(snapshot.bridge_snapshot)==expected["raw_root_sha256"], "paired replay root drift")
    before=digest(dict(capture=capture,records=process_records,expected=expected,snapshot=context_identity(snapshot)))
    trace=read_native_prefix_transport(prefix_view(capture),process_records=process_records,
        raw_root=snapshot.bridge_snapshot,clone_binding=expected["clone_binding"],request_ids=expected["request_ids"],
        native_boundaries=expected["native_boundaries"],party_mapping=expected["party_mapping"])
    streams=capture["replay_streams"]
    require(type(streams) is list and type(capture["paired_contexts"]) is list,
        "original paired replay streams/contexts required")
    indices=[i for i,row in enumerate(process_records) if row["kind"]=="event" and row["payload"].get("type")=="stream"]
    require(len(streams)==len(indices), "complete original replay stream roster required")
    events={}
    for row,index in zip(streams,indices):
        require(type(row) is dict and set(row)=={"ordinal","event"} and type(row["ordinal"]) is int
            and row["ordinal"]==index and digest(_redact("event",row["event"]))==digest(process_records[index]),
            "paired replay stream differs from independent process transport")
        events[index]=row["event"]
    # A pure fold shell: no __init__, filesystem/catalog lookup, process or
    # observation. Restore clones the actual typed parser/belief/trackers.
    env=object.__new__(LocalShowdownEnv)
    env._battle_token=expected["clone_binding"]["battleId"]
    env._restore_local_snapshot_state(snapshot)
    contexts=[];endpoint=None;restored=False
    for index,row in enumerate(process_records):
        payload=row["payload"];tag=payload.get("type")
        if row["kind"]=="command" and tag=="choices":
            env._latest_requests={};env._last_step_had_error=False
        elif row["kind"]=="event":
            if tag=="cloned":
                # The actual producer restores only after clone acknowledgement.
                # Clone-shell streams before that point are discarded by that
                # restore, not appended to the retained source context.
                env._restore_local_snapshot_state(snapshot);restored=True
            elif restored and tag in {"stream","ready","terminal"}:
                env._apply_event(events[index] if tag=="stream" else payload)
            elif tag=="owned_boundary_snapshot":
                raw=json.loads(capture["snapshots"][len(contexts)]["currentSnapshotJson"])
                env._sync_incremental_state()
                endpoint=env._local_snapshot(bridge_snapshot=raw)
                contexts.append(context_identity(endpoint))
    require(digest(contexts)==digest(capture["paired_contexts"]) and len(contexts)==len(capture["snapshots"]),
        "retained paired contexts differ from independently folded actual stream events")
    require(before==digest(dict(capture=capture,records=process_records,expected=expected,snapshot=context_identity(snapshot))),
        "paired replay evidence changed during readback")
    return trace, endpoint


@dataclass(frozen=True)
class PairedPrefixRunContract(NativePrefixBoundaryRunContract):
    def receipt(self):
        return dict(super().receipt(),schema=SCHEMA+".contract",paired_replay_law=replay_law())


class PairedPrefixProcessOwner(NativePrefixBoundaryProcessOwner):
    launch_schema=SCHEMA+".launch"
    close_schema=SCHEMA+".close"

    def __init__(self, *, snapshot, expected, **kwargs):
        super().__init__(**kwargs)
        self.snapshot_context,self.expected=deepcopy(snapshot),deepcopy(expected)

    def _match_capture(self,capture):
        require(getattr(self.env,"_bridge_boundary_observer",None) is None,"paired replay tap still installed")
        read_paired_prefix(capture,process_records=self.records,snapshot=self.snapshot_context,expected=self.expected)


@dataclass(frozen=True)
class OwnedPairedPrefixJob(OwnedNativePrefixBoundaryJob):
    snapshot: object = None
    paired_constructor: object = None

    def input_binding(self):
        value=dict(super().input_binding(),paired_snapshot=context_identity(self.snapshot),
            paired_constructor=deepcopy(self.paired_constructor),paired_replay_law=replay_law(),paired_replay_sources=sources())
        validate_inputs(value)
        require(self.snapshot.terminal is None and digest(self.snapshot.bridge_snapshot)==digest(self.raw_root)
            and self.snapshot.battle_token!=self.clone_binding["battleId"],"original nonterminal paired root required")
        return value

    def __call__(self,ownership,phase):
        phase("startup");before=self.input_binding();env=self._components()
        directory=Path(ownership["directory"]).parent
        tap=PairedPrefixCapture(clone_binding=self.clone_binding,request_ids=self.request_ids)
        owner=PairedPrefixProcessOwner(directory=directory,ownership=ownership,runtime=self.runtime,config=self.config,
            initial_snapshot_json=self.initial_snapshot_json(),clone_binding=self.clone_binding,request_ids=self.request_ids,
            raw_root=self.raw_root,native_boundaries=self.native_boundaries,party_mapping=self.party_mapping,
            snapshot=self.snapshot,expected=before)
        primary=None
        with owner.installed(env):
            try:
                phase("boundaries")
                with tap.installed(env):
                    env._start_bridge();env._battle_token=self.clone_binding["battleId"]
                    event=env._bridge_request_event(dict(type="clone_snapshot",battleId=self.clone_binding["battleId"],
                        cloneId=self.clone_binding["cloneId"],snapshotJson=self.initial_snapshot_json()),"cloned")
                    require(all(event.get(k)==v for k,v in self.clone_binding.items()),"paired clone acknowledgement drift")
                    env._restore_local_snapshot_state(self.snapshot)
                    current=tap.snapshot();self.check_reached_boundary(current);endpoint=tap.retain(current)
                    for ordinal in range(self.transition_count()):
                        try:choices=self.commands_at_boundary(ordinal,current)
                        except NativeReplayUnavailable:break
                        env._submit_step_choices(choices=choices,
                            payload=dict(type="choices",battleId=env._battle_token,choices=choices),observation_players=())
                        current=tap.snapshot();self.check_reached_boundary(current);endpoint=tap.retain(current)
                    termination=termination_witness(current,self.native_boundaries,self.party_mapping,len(tap.snapshots))
                    env._bridge_request_event(dict(type="release_clone",**self.clone_binding),"clone_released")
                    env._battle_token=None
                capture=dict(tap.export(),schema=SCHEMA,termination=termination,
                    paired_contexts=tap.contexts,replay_streams=tap.streams)
                owner.seal(capture)
                payload=pickle.dumps(endpoint,protocol=5)
                with (directory/ARTIFACT).open("xb") as stream:stream.write(payload)
            except BaseException as error:
                primary=error;raise
            finally:
                phase("cleanup")
                if env._process is not None:
                    try:env.close()
                    except BaseException as cleanup:
                        if primary is None:raise
                        primary.add_note("Paired replay cleanup failed: "+type(cleanup).__name__)
        require(digest(self.input_binding())==digest(before) and (directory/ARTIFACT).read_bytes()==payload,
            "paired replay inputs/context changed during cleanup")
        return dict(**before,capture=capture,capture_sha256=digest(capture),
            paired_endpoint=dict(**context_identity(endpoint),artifact_sha256=hashlib.sha256(payload).hexdigest()),
            synthetic_process_fixture=getattr(self,"synthetic_process_fixture",False) is True,**{k:False for k in UNQUALIFIED})


def _join_result(result,expected,capture):
    require(type(result) is dict and "paired_endpoint" in result,"paired replay endpoint metadata required")
    _join_raw_result({k:v for k,v in result.items() if k!="paired_endpoint"},expected,capture)
    require(type(result["paired_endpoint"]) is dict
        and set(result["paired_endpoint"])=={"snapshot_sha256","battle_token","battle_id","artifact_sha256"}
        and {k:v for k,v in result["paired_endpoint"].items() if k!="artifact_sha256"}==capture["paired_contexts"][-1],
        "endpoint must retain the last actual paired context")


def _endpoint_bytes(result,directory):
    path=Path(directory)/ARTIFACT
    require(path==path.resolve() and path.is_file(),"canonical private paired AFTER context required")
    raw=path.read_bytes()
    require(hashlib.sha256(raw).hexdigest()==result["paired_endpoint"]["artifact_sha256"],
        "original paired endpoint file differs from worker result")
    return raw


def supervise_paired_prefix_run(job, *, directory,binding_sha256,verify_admission,validate_result,contract=None,
        _synthetic_watchdog_seconds=None):
    directory=Path(directory).resolve()
    contract=PairedPrefixRunContract() if contract is None else contract
    require(type(contract) is PairedPrefixRunContract and (type(job) is OwnedPairedPrefixJob or
        isinstance(job,OwnedPairedPrefixJob) and getattr(job,"synthetic_process_fixture",False) is True),
        "distinct paired replay job/contract required")
    before=job.input_binding()
    def verify():
        require(digest(job.input_binding())==digest(before),"paired replay admission drift")
        return verify_admission()
    def validate(result):
        _join_result(result,before,result["capture"])
        frozen=digest(result);raw=_endpoint_bytes(result,directory)
        require(validate_result(result) is not False and digest(result)==frozen
            and digest(job.input_binding())==digest(before) and _endpoint_bytes(result,directory)==raw,
            "paired replay acceptance refused or changed")
    return _supervise_owned_job(job,directory=directory,binding_sha256=binding_sha256,verify_admission=verify,
        contract=contract,kind="boundary-run",work_phase="boundaries",validate_result=validate,
        _synthetic_watchdog_seconds=_synthetic_watchdog_seconds)


class PairedPrefixRunValidator(RawBoundaryRunValidator):
    contract_class=PairedPrefixRunContract
    launch_schema=PairedPrefixProcessOwner.launch_schema
    close_schema=PairedPrefixProcessOwner.close_schema
    validate_expected=staticmethod(validate_inputs)

    def __init__(self, *, snapshot,expected_endpoint,**kwargs):
        self.snapshot=deepcopy(snapshot);self.endpoint=deepcopy(expected_endpoint)
        self.paired_inputs=digest(dict(snapshot=context_identity(self.snapshot),endpoint=self.endpoint))
        super().__init__(**kwargs)

    def _check_inputs(self):
        super()._check_inputs()
        require(digest(dict(snapshot=context_identity(self.snapshot),endpoint=self.endpoint))==self.paired_inputs
            and context_identity(self.snapshot)==self.expected["paired_snapshot"],"external paired context drift")

    def _join_result(self,result,capture,runtime_sha):
        require(runtime_sha==self.expected["runtime_sha256"] and digest(result["paired_endpoint"])==digest(self.endpoint),
            "original external endpoint/runtime binding drift")
        _join_result(result,self.expected,capture)

    def _match_capture(self,capture,records):
        _,endpoint=read_paired_prefix(capture,process_records=records,snapshot=self.snapshot,expected=self.expected)
        require(context_identity(endpoint)=={k:v for k,v in self.endpoint.items() if k!="artifact_sha256"},
            "independently folded endpoint differs from original worker")

    def __call__(self,capture):
        self.last_validation=None
        path=self.directory/ARTIFACT
        require(path==path.resolve() and path.is_file(),"canonical private paired AFTER context required")
        raw=path.read_bytes()
        require(hashlib.sha256(raw).hexdigest()==self.endpoint["artifact_sha256"],"paired AFTER context byte drift")
        try:
            witness=super().__call__(capture)
            self._check_inputs()
            require(path==path.resolve() and path.read_bytes()==raw,"paired AFTER context changed during owning callbacks")
            self.last_validation["receipt_file_sha256"][ARTIFACT]=hashlib.sha256(raw).hexdigest()
            return witness
        except BaseException:
            self.last_validation=None;raise

    def retained_endpoint(self,result):
        """Only the full designated nonterminal AFTER endpoint, never a prefix."""
        self.last_validation=None
        try:
            frozen=digest(result)
            require(self(result["capture"])==digest(result["capture"]),"owning paired replay did not clear")
            terminal=(self.directory/"terminal.json").read_bytes()
            require(digest(json.loads(terminal)["result"])==frozen,"paired replay result differs from original terminal")
            require(result["capture"]["termination"]["status"]=="FULL_PATH_OBSERVED",
                "unavailable prefix cannot substitute for designated AFTER endpoint")
            raw=(self.directory/ARTIFACT).read_bytes()
            require(hashlib.sha256(raw).hexdigest()==self.endpoint["artifact_sha256"],"paired endpoint changed before load")
            endpoint=pickle.loads(raw)
            require(context_identity(endpoint)=={k:v for k,v in self.endpoint.items() if k!="artifact_sha256"}
                and endpoint.terminal is None,"actual designated nonterminal paired AFTER context required")
            self._check_inputs()
            require((self.directory/ARTIFACT).read_bytes()==raw and (self.directory/"terminal.json").read_bytes()==terminal
                and digest(result)==frozen,"paired endpoint changed after final readback")
            return endpoint
        except BaseException:
            self.last_validation=None;raise
