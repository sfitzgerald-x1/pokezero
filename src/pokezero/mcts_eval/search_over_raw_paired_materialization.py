"""Prospective actual consumed-world constructor retaining paired Python state.

No native prediction join, replay, labels or scientific admission is supplied
here. The private snapshot is produced by the existing reference initializer,
not synthesized from a raw root or the source opponent's private history.
"""
from copy import deepcopy
from dataclasses import dataclass, replace
import hashlib
import json
from pathlib import Path
import pickle

from ..env import BattleStartOverride
from ..local_showdown import LocalShowdownSnapshot
from .search_over_raw import digest, require
from .search_over_raw_materialization import (
    materialization_inputs, validate_materialization_inputs, materialization_source_hashes,
    _observe_paired_materialization,
    MaterializationCapture,
)
from .search_over_raw_materialization_run import (
    MaterializationRunContract, MaterializationProcessOwner, MaterializationRunValidator,
    OwnedRecordedMaterializationJob, UNQUALIFIED, _join_result,
)
from .search_over_raw_outcome_run import verify_outcome_runtime, _redact
from .search_over_raw_supervision import _supervise_owned_job
from .search_over_raw_world_inputs import _state_payload, _snapshot_payload

SCHEMA = "pokezero.search-over-raw.paired-native-materialization.v2"
ARTIFACT = "paired-context.pickle"
RESET_ARTIFACT = "paired-reset-events.json"


class ContinuationContextUnavailable(ValueError):
    """Well-formed captured overlay not represented by the supplied real context."""
    reason = "CONSUMED_OVERLAY_NOT_REPRESENTED_BY_ORIGINAL_CONTEXT"


def context_law():
    return dict(schema=SCHEMA, initializer="LocalShowdownEnv.materialize_public_world",
        captured_native_draw_and_overlay_required=True, source_actor_request_substituted=True,
        other_requests="actual_materialized_bridge_requests",
        source_public_belief_cloned=True, tier2_and_investment_trackers_reset=True,
        observation_calls_before_snapshot=0, private_snapshot_retained=True,
        original_sampled_p2_history_reconstructed=False, request_correspondence_validated=False,
        native_evaluation_input_parity=False, independent_labels=False, retries_permitted=False)


def context_sources():
    paths = (Path(__file__).resolve(), Path(__file__).with_name("search_over_raw_world_inputs.py").resolve())
    return dict(materialization_source_hashes(), **{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths})


def paired_materialization_inputs(state, recorded_inputs, runtime):
    """No launch or root substitution; require every original constructor input."""
    validate_materialization_inputs(recorded_inputs)
    require("recorded_native_origin" in recorded_inputs, "original consumed-native constructor origin required")
    require(recorded_inputs["public_state_sha256"]==digest(_state_payload(state))
        and recorded_inputs["root_binding"].get("subject")==state.player_id
        and recorded_inputs["materialize"]["publicState"].get("selfPlayer")==state.player_id
        and recorded_inputs["observation_format_id"]==state.observation_format_id
        and (recorded_inputs["draw"]["observation_format_id"] or recorded_inputs["draw"]["format_id"])
            ==state.observation_format_id,"original public context identity/format provenance drift")
    candidate = materialization_inputs(state=state, override=BattleStartOverride(**recorded_inputs["draw"]),
        seed=recorded_inputs["seed"], root_binding=recorded_inputs["root_binding"], options=recorded_inputs["options"])
    expected = {k:v for k,v in recorded_inputs.items() if k != "recorded_native_origin"}
    require(all(digest(candidate[k])==digest(expected[k]) for k in candidate
        if k not in {"materialize","public_payload_sha256"}),"original constructor provenance differs from context")
    if digest(candidate["materialize"]) != digest(expected["materialize"]):
        raise ContinuationContextUnavailable("captured native construction differs from the original paired context")
    return dict(materialization_inputs=deepcopy(recorded_inputs), runtime_sha256=digest(runtime),
        context_law=context_law(), context_sources=context_sources(), original_public_context_sha256=digest(_state_payload(state)))


def _validate_inputs(value, state):
    require(type(value) is dict and set(value)=={"materialization_inputs","runtime_sha256","context_law",
        "context_sources","original_public_context_sha256"}
        and digest(value["context_law"])==digest(context_law())
        and digest(value["context_sources"])==digest(context_sources())
        and value["original_public_context_sha256"]==digest(_state_payload(state)),
        "original source-bound paired context law required")
    validate_materialization_inputs(value["materialization_inputs"])


class PairedMaterializationCapture(MaterializationCapture):
    def __init__(self,inputs):
        super().__init__(inputs);self.reset_events=[]

    def event(self,payload):
        if payload.get("type")=="stream":
            self.reset_events.append(dict(ordinal=len(self.records),event=deepcopy(dict(payload))))
        super().event(payload)


def _reset_history(events,capture):
    """Re-fold actual reset requests at their observed public event cursors."""
    from ..showdown import _ReplayParser
    from ..local_showdown import _decode_request_line
    indices=[i for i,row in enumerate(capture["records"]) if row["kind"]=="event" and row["payload"].get("type")=="stream"]
    require(type(events) is list and len(events)==len(indices),"complete original reset-stream roster required")
    parser=_ReplayParser("paired-reset-readback",complete_prefix=True,hp_visibility={"p1":"exact","p2":"exact"})
    lines=[];parsed=0;history={"p1":[],"p2":[]}
    for row,index in zip(events,indices):
        require(type(row) is dict and set(row)=={"ordinal","event"} and type(row["ordinal"]) is int
            and row["ordinal"]==index and digest(_redact("event",row["event"]))==digest(capture["records"][index]),
            "retained reset event differs from original observed transport")
        event=row["event"];stream=event["stream"]
        require(type(event.get("lines")) is list and all(type(line) is str for line in event["lines"]),
            "actual reset stream lines required")
        for line in (v for v in event["lines"] if v):
            if stream=="omniscient":lines.append(line)
            elif stream in history and line.startswith("|request|"):
                request=_decode_request_line(line)
                if type(request.get("side")) is dict and request["side"].get("id")==stream:
                    parser.feed(lines[parsed:]);parsed=len(lines)
                    retained=deepcopy(request);retained["_pokezero_public_event_cursor"]=len(parser.public_events)
                    history[stream].append(retained);lines.append(line)
    return history


def _context_binding(snapshot, *, state, capture, reset_events, recorded_inputs):
    """Independently check initializer invariants; do not assert request parity."""
    from ..showdown import _ReplayParser
    from ..local_showdown import SnapshotAnnotationCache
    require(type(snapshot) is LocalShowdownSnapshot and snapshot.battle_token=="b1"
        and snapshot.format_id==recorded_inputs["draw"]["format_id"]
        and snapshot.battle_id==f"local-{recorded_inputs['draw']['format_id']}-{recorded_inputs['seed']}"
        and snapshot.observation_format_id==state.observation_format_id
        and snapshot.protocol_lines==() and snapshot.latest_turn==state.replay.turn_number
        and snapshot.terminal is None and snapshot.search_choice_cache=={}
        and type(snapshot.annotation_cache) is SnapshotAnnotationCache
        and snapshot.annotation_cache.tier2_trackers=={} and snapshot.annotation_cache.investment_trackers=={},
        "paired snapshot does not retain the declared original initializer state")
    raw = capture["raw_events"][1]["snapshot"]
    requests = deepcopy(capture["raw_events"][0]["boundaryRequests"])
    requests[state.player_id] = deepcopy(state.self_request)
    initial = state.self_initial_request or requests.get(state.player_id)
    first = {state.player_id:deepcopy(initial)} if initial is not None else deepcopy(requests)
    replay = _ReplayParser.from_snapshot(replace(state.replay,battle_id=snapshot.battle_id,requests=requests)).snapshot()
    require(digest(snapshot.bridge_snapshot)==digest(raw)
        and digest(snapshot.latest_requests)==digest(requests) and digest(snapshot.first_requests)==digest(first)
        and digest(_state_payload(snapshot.replay))==digest(_state_payload(replay))
        and digest(_state_payload(snapshot.belief_engine))==digest(_state_payload(state.belief_engine.clone()))
        and digest(snapshot.request_history)==digest({s:tuple(rows) for s,rows in _reset_history(reset_events,capture).items()}),
        "retained paired state differs from original public context and actual materialized requests")
    return dict(snapshot_sha256=digest(_snapshot_payload(snapshot)),
        bridge_snapshot_sha256=digest(raw), latest_requests_sha256=digest(snapshot.latest_requests),
        actual_bridge_requests_sha256=digest(raw["boundaryRequests"]),
        source_actor_request_sha256=digest(state.self_request),
        request_substitution_changed=digest(snapshot.latest_requests)!=digest(raw["boundaryRequests"]),
        initialization_law_sha256=digest(context_law()), request_correspondence_validated=False,
        paired_replay_validated=False, independent_terminal_labels=False)


@dataclass(frozen=True)
class PairedMaterializationRunContract(MaterializationRunContract):
    def receipt(self):
        return dict(super().receipt(), schema="pokezero.search-over-raw.paired-materialization-run-contract.v2",
            context_law=context_law())


class PairedMaterializationProcessOwner(MaterializationProcessOwner):
    launch_schema="pokezero.search-over-raw.paired-materialization-bridge-launch.v2"
    close_schema="pokezero.search-over-raw.paired-materialization-bridge-close.v2"


@dataclass(frozen=True)
class OwnedPairedMaterializationJob(OwnedRecordedMaterializationJob):
    original_public_context: object = None

    def input_binding(self):
        verify_outcome_runtime(self.runtime,self.config)
        return paired_materialization_inputs(self.original_public_context,self.recorded_inputs,self.runtime)

    def __call__(self,ownership,phase):
        phase("startup"); before=self.input_binding(); env=self._components()
        directory=Path(ownership["directory"]).parent
        owner=PairedMaterializationProcessOwner(directory=directory,ownership=ownership,runtime=self.runtime,
            config=self.config,expected_inputs=before["materialization_inputs"])
        def unchanged():
            require(digest(self.input_binding())==digest(before),"paired constructor inputs changed")
        primary=None
        with owner.installed(env):
            try:
                phase("materialization")
                tap=PairedMaterializationCapture(before["materialization_inputs"])
                observed=_observe_paired_materialization(env,state=self.original_public_context,
                    override=BattleStartOverride(**self.recorded_inputs["draw"]),
                    inputs=before["materialization_inputs"],revalidate=unchanged,tap=tap)
                capture,snapshot=observed["capture"],observed["snapshot"]
                context=_context_binding(snapshot,state=self.original_public_context,capture=capture,reset_events=tap.reset_events,
                    recorded_inputs=before["materialization_inputs"])
                from .search_over_raw_supervision import _save_new
                _save_new(directory/RESET_ARTIFACT,tap.reset_events)
                payload=pickle.dumps(snapshot,protocol=5)
                # Trusted LOCAL private artifact, never an entry-JSON input or
                # a source-opponent history. Exclusive creation forbids reuse.
                with (directory/ARTIFACT).open("xb") as stream:stream.write(payload)
                context.update(artifact=ARTIFACT,artifact_sha256=hashlib.sha256(payload).hexdigest(),
                    reset_events_sha256=digest(tap.reset_events))
                root=deepcopy(snapshot.bridge_snapshot)
                owner.snapshot=root; owner.seal(capture)
            except BaseException as failure:
                primary=failure; raise
            finally:
                phase("cleanup")
                if env._process is not None:
                    try:env.close()
                    except BaseException as cleanup:
                        if primary is None:raise
                        primary.add_note("Paired materialization cleanup failed: "+type(cleanup).__name__)
        unchanged()
        require(hashlib.sha256((directory/ARTIFACT).read_bytes()).hexdigest()==context["artifact_sha256"]
            and digest(_snapshot_payload(snapshot))==context["snapshot_sha256"],"private paired snapshot changed during cleanup")
        return dict(**before,capture=capture,capture_sha256=digest(capture),produced_bridge_snapshot=root,
            produced_bridge_snapshot_sha256=digest(root),paired_context=context,
            synthetic_process_fixture=getattr(self,"synthetic_process_fixture",False) is True,
            **{key:False for key in UNQUALIFIED})


def _base_inputs(value):
    return {key:value[key] for key in ("materialization_inputs","runtime_sha256")}


def _join_paired_result(result,expected,capture,directory,state,expected_context=None):
    require(type(result) is dict and set(result)==set(expected)|set(UNQUALIFIED)|{
        "capture","capture_sha256","produced_bridge_snapshot","produced_bridge_snapshot_sha256",
        "synthetic_process_fixture","paired_context"}
        and all(digest(result[k])==digest(v) for k,v in expected.items()),"exact paired constructor result required")
    base={k:v for k,v in result.items() if k not in set(expected)-set(_base_inputs(expected))|{"paired_context"}}
    _join_result(base,_base_inputs(expected),capture)
    report=result["paired_context"]
    require(type(report) is dict and report.get("artifact")==ARTIFACT,"original private paired artifact required")
    require(expected_context is None or digest(report)==digest(expected_context),
        "paired artifact metadata differs from original externally frozen worker result")
    path=Path(directory)/ARTIFACT
    require(path==path.resolve() and path.is_file(),"canonical private paired artifact required")
    raw=path.read_bytes()
    require(hashlib.sha256(raw).hexdigest()==report.get("artifact_sha256"),"private paired artifact byte drift")
    reset_path=Path(directory)/RESET_ARTIFACT
    require(reset_path==reset_path.resolve() and reset_path.is_file(),"canonical original reset events required")
    reset_bytes=reset_path.read_bytes();events=json.loads(reset_bytes)
    require(digest(events)==report.get("reset_events_sha256"),"retained reset-event drift")
    # Only original locally produced bytes, bound by this owning job/readout,
    # are deserialized. Never load a user-provided artifact path from JSON.
    snapshot=pickle.loads(raw)
    derived=_context_binding(snapshot,state=state,capture=capture,reset_events=events,
        recorded_inputs=expected["materialization_inputs"])
    require(digest(report)==digest(dict(derived,artifact=ARTIFACT,artifact_sha256=hashlib.sha256(raw).hexdigest(),
        reset_events_sha256=digest(events))),
        "private paired context differs from independently read initializer bindings")
    require(path.read_bytes()==raw and reset_path.read_bytes()==reset_bytes,"paired artifacts changed during readback")
    return snapshot


class PairedMaterializationRunValidator(MaterializationRunValidator):
    contract_class=PairedMaterializationRunContract
    launch_schema=PairedMaterializationProcessOwner.launch_schema
    close_schema=PairedMaterializationProcessOwner.close_schema

    def __init__(self,*,directory,job_sha256,binding_sha256,expected_inputs,runtime,original_public_context,expected_context,
            allow_synthetic=False,contract=None):
        from .search_over_raw_outcomes import _sha
        require(_sha(job_sha256) and _sha(binding_sha256) and type(allow_synthetic) is bool,
            "external paired constructor owner bindings required")
        self.directory=Path(directory).resolve();self.job_sha=job_sha256;self.binding=binding_sha256
        self.allow_synthetic=allow_synthetic;self.expected=deepcopy(expected_inputs);self.runtime=deepcopy(runtime)
        self.state=deepcopy(original_public_context)
        require(type(expected_context) is dict and expected_context.get("artifact")==ARTIFACT,
            "original externally frozen paired context metadata required")
        self.context=deepcopy(expected_context)
        _validate_inputs(self.expected,self.state)
        require(digest(paired_materialization_inputs(self.state,self.expected["materialization_inputs"],self.runtime))
            ==digest(self.expected),"external paired constructor inputs differ from original context")
        contract=PairedMaterializationRunContract() if contract is None else contract
        require(type(contract) is PairedMaterializationRunContract,"distinct paired constructor contract required")
        self.contract=contract.receipt();self.inputs=self._input_sha()

    def _input_sha(self):
        return digest(dict(expected=self.expected,runtime=self.runtime,state=_state_payload(self.state),context=self.context))

    def __call__(self,capture):
        self.last_validation=None
        originals={name:(self.directory/name).read_bytes() for name in (ARTIFACT,RESET_ARTIFACT)}
        try:
            witness=super().__call__(capture)
            self._check_inputs()
            require(all((self.directory/name).resolve()==self.directory/name
                and (self.directory/name).read_bytes()==raw for name,raw in originals.items()),
                "private paired artifacts changed during final owning callbacks")
            self.last_validation["receipt_file_sha256"].update({name:hashlib.sha256(raw).hexdigest() for name,raw in originals.items()})
            return witness
        except BaseException:
            self.last_validation=None
            raise

    def _check_inputs(self):
        _validate_inputs(self.expected,self.state)
        require(self._input_sha()==self.inputs,
            "original paired constructor reader inputs changed")

    def _join_result(self,result,capture,runtime_sha):
        require(runtime_sha==self.expected["runtime_sha256"],"paired constructor runtime drift")
        _join_paired_result(result,self.expected,capture,self.directory,self.state,self.context)

    def retained_context(self,result):
        """Fresh original owner read before returning a private in-memory value."""
        import json
        try:
            frozen=digest(result)
            require(self(result["capture"])==digest(result["capture"]),"paired context owning scope did not clear")
            path=self.directory/"terminal.json";terminal=path.read_bytes()
            require(digest(json.loads(terminal)["result"])==frozen,"supplied paired result differs from original terminal")
            artifact=(self.directory/ARTIFACT).read_bytes()
            reset_events=(self.directory/RESET_ARTIFACT).read_bytes()
            snapshot=_join_paired_result(result,self.expected,result["capture"],self.directory,self.state,self.context)
            self._check_inputs()
            require(digest(result)==frozen and path.read_bytes()==terminal
                and (self.directory/ARTIFACT).read_bytes()==artifact
                and (self.directory/RESET_ARTIFACT).read_bytes()==reset_events,
                "paired context changed after original owning read")
            return snapshot
        except BaseException:
            self.last_validation=None
            raise


def supervise_paired_materialization_run(job,*,directory,binding_sha256,verify_admission,validate_result,
        contract=None,_synthetic_watchdog_seconds=None):
    directory=Path(directory).resolve()
    contract=PairedMaterializationRunContract() if contract is None else contract
    require(type(contract) is PairedMaterializationRunContract and (type(job) is OwnedPairedMaterializationJob or
        isinstance(job,OwnedPairedMaterializationJob) and getattr(job,"synthetic_process_fixture",False) is True)
        and callable(validate_result),"original distinct paired constructor job required")
    before=job.input_binding()
    def verify():
        require(digest(job.input_binding())==digest(before),"paired materialization admission drift")
        return verify_admission()
    def validate(result):
        _join_paired_result(result,before,result["capture"],directory,job.original_public_context)
        frozen=digest(result)
        require(validate_result(result) is not False and digest(result)==frozen
            and digest(job.input_binding())==digest(before),"paired materialization acceptance refused or changed")
    return _supervise_owned_job(job,directory=directory,binding_sha256=binding_sha256,verify_admission=verify,
        contract=contract,kind="materialization-run",work_phase="materialization",validate_result=validate,
        _synthetic_watchdog_seconds=_synthetic_watchdog_seconds)
