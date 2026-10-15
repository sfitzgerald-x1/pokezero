"""Private actual reference-constructor observations, never a new source draw.

The existing materializer and snapshot law are unchanged. Raw bridge requests
are retained separately from Python's actor-request substitution. A transport
witness alone proves neither process ownership nor world/engine equivalence.
"""
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import asdict
import hashlib
from pathlib import Path

from .search_over_raw import digest, require
from .search_over_raw_outcome_run import _redact
from .search_over_raw_world_inputs import _state_payload

SCHEMA = "pokezero.search-over-raw.reference-materialization.v1"
BOOL_OPTIONS = ("reference_rest_sleep","reference_consumed_items","reference_turn_clocks",
    "reference_attract","reference_yawn")
MAP_OPTIONS = ("deferred_opponent_actions","deferred_opponent_action_priors",
    "reference_encore_durations","reference_induced_sleep")


def validate_materialization_inputs(inputs):
    """Validate internal input joins, not reconstruct an opaque public-state hash.

    The state digest remains an externally supplied prospective binding. Its
    relationship to the public payload is witnessed by the actual constructor
    seam, not inferred by this pure reader or by matching hashes alone.
    """
    from ..env import BattleStartOverride
    from ..local_showdown import _start_players_payload,showdown_seed_from_int
    from .search_over_raw_outcomes import _sha
    required={"root_binding","public_state_sha256",
        "public_payload_sha256","observation_format_id","draw_sha256","draw","seed","options",
        "start","materialize","source_hashes"}
    require(type(inputs) is dict and set(inputs) in (required,required|{"recorded_native_origin"}),
        "exact external materialization input schema required")
    if "recorded_native_origin" in inputs:
        origin=inputs["recorded_native_origin"]
        require(type(origin) is dict and set(origin)=={"schema","native_selected_sha256",
            "construction_capture_sha256","construction_ordinal","original_public_request_sha256",
            "draw_sha256","public_payload_sha256"}
            and origin["schema"]=="pokezero.search-over-raw.recorded-native-materialization.v1"
            and type(origin["construction_ordinal"]) is int and 0<=origin["construction_ordinal"]<4
            and all(_sha(origin[k]) for k in ("native_selected_sha256","construction_capture_sha256",
                "original_public_request_sha256","draw_sha256","public_payload_sha256"))
            and origin["draw_sha256"]==inputs["draw_sha256"]
            and origin["public_payload_sha256"]==inputs["public_payload_sha256"],
            "exact recorded native constructor origin required")
    require(type(inputs["root_binding"]) is dict and inputs["root_binding"]
        and _sha(inputs["public_state_sha256"]) and _sha(inputs["public_payload_sha256"])
        and _sha(inputs["draw_sha256"]) and type(inputs["seed"]) is int
        and type(inputs["observation_format_id"]) is str and inputs["observation_format_id"]
        and inputs["source_hashes"]==materialization_source_hashes(),"prospective materialization bindings required")
    draw=inputs["draw"]
    require(type(draw) is dict and set(draw)=={"player_teams","format_id","observation_format_id"}
        and type(draw["player_teams"]) is dict and set(draw["player_teams"])=={"p1","p2"}
        and all(type(v) is str and v for v in draw["player_teams"].values())
        and type(draw["format_id"]) is str
        and (draw["observation_format_id"] is None or type(draw["observation_format_id"]) is str
            and draw["observation_format_id"])
        and digest(draw)==inputs["draw_sha256"],"exact canonical sampled draw required")
    override=BattleStartOverride(**draw)
    require(asdict(override)==draw and inputs["start"]==dict(type="start",battleId="b1",
        formatid=override.format_id,seed=showdown_seed_from_int(inputs["seed"]),
        players=_start_players_payload(override)),"actual start must derive from both frozen teams and seed")
    values=inputs["options"]
    require(type(values) is dict and set(values)==set(BOOL_OPTIONS)|set(MAP_OPTIONS)
        and all(type(values[k]) is bool for k in BOOL_OPTIONS)
        and all(values[k] is None or type(values[k]) is dict for k in MAP_OPTIONS),
        "exact complete constructor option schema required")
    command=inputs["materialize"]
    require(type(command) is dict and type(command.get("publicState")) is dict
        and digest(command["publicState"])==inputs["public_payload_sha256"],"frozen public overlay binding required")
    expected=dict(type="materialize",battleId="b1",publicState=command["publicState"],
        referenceRestSleep=values["reference_rest_sleep"],referenceConsumedItems=values["reference_consumed_items"],
        referenceTurnClocks=values["reference_turn_clocks"],referenceAttract=values["reference_attract"],
        referenceYawn=values["reference_yawn"],referenceEncoreDurations=values["reference_encore_durations"],
        referenceInducedSleep=values["reference_induced_sleep"],
        referenceRulesFormat=inputs["observation_format_id"] if values["reference_induced_sleep"] is not None else None)
    require(command==expected and all(type(command[k]) is bool for k in (
        "referenceRestSleep","referenceConsumedItems","referenceTurnClocks","referenceAttract","referenceYawn")),
        "exact options/materializer command join required")
    public=command["publicState"]
    require(public.get("selfPlayer") in {"p1","p2"}
        and public.get("deferredOpponentActions")==dict(values["deferred_opponent_actions"] or {})
        and digest(public.get("deferredOpponentActionPriors"))==digest(dict(values["deferred_opponent_action_priors"] or {})),
        "exact deferred options/public overlay join required")
    return inputs


def recorded_materialization_inputs(*,draw,public_payload,seed,root_binding,public_state_sha256,
        recorded_native_origin,options):
    """Freeze ACTUAL captured overlay; never rebuild it from root state/seed.

    Origin hashes are prospective attempt bindings, not ownership by themselves.
    The producer must separately read the original native owning run.
    """
    from ..env import BattleStartOverride
    from ..local_showdown import _start_players_payload,showdown_seed_from_int
    draw,public,values=deepcopy(draw),deepcopy(public_payload),deepcopy(options)
    require(type(draw) is dict and type(public) is dict and type(values) is dict
        and set(values)==set(BOOL_OPTIONS)|set(MAP_OPTIONS),"explicit complete recorded constructor inputs required")
    override=BattleStartOverride(**draw)
    observation_format=draw["observation_format_id"] or draw["format_id"]
    command=dict(type="materialize",battleId="b1",publicState=public,
        referenceRestSleep=values["reference_rest_sleep"],referenceConsumedItems=values["reference_consumed_items"],
        referenceTurnClocks=values["reference_turn_clocks"],referenceAttract=values["reference_attract"],
        referenceYawn=values["reference_yawn"],referenceEncoreDurations=values["reference_encore_durations"],
        referenceInducedSleep=values["reference_induced_sleep"],
        referenceRulesFormat=observation_format if values["reference_induced_sleep"] is not None else None)
    return validate_materialization_inputs(dict(root_binding=deepcopy(root_binding),
        public_state_sha256=public_state_sha256,public_payload_sha256=digest(public),observation_format_id=observation_format,
        draw=draw,draw_sha256=digest(draw),seed=seed,options=values,materialize=command,
        start=dict(type="start",battleId="b1",formatid=override.format_id,seed=showdown_seed_from_int(seed),
            players=_start_players_payload(override)),source_hashes=materialization_source_hashes(),
        recorded_native_origin=deepcopy(recorded_native_origin)))


def observe_recorded_materialized_world(env,*,inputs):
    """Actual start/materialize/snapshot/end without fictional Python state.

    Returned raw bridge requests stay authoritative. No actor-request substitution
    or archived private simulator snapshot enters the diagnostic constructor.
    """
    from ..env import BattleStartOverride
    import time
    frozen=digest(inputs);validate_materialization_inputs(inputs)
    require("recorded_native_origin" in inputs,"distinct recorded constructor origin required")
    tap=MaterializationCapture(inputs)
    with tap.installed(env):
        env.reset_with_start_override(seed=inputs["seed"],start_override=BattleStartOverride(**inputs["draw"]))
        require(env._battle_token=="b1","recorded constructor must use original fresh owned shell")
        event=env._bridge_request_event(deepcopy(inputs["materialize"]),"materialized")
        require(type(event.get("boundaryRequests")) is dict and event["boundaryRequests"],
            "recorded constructor produced no actionable raw request")
        event=env._bridge_request_event(dict(type="snapshot",battleId="b1"),"snapshot")
        snapshot=deepcopy(event["snapshot"])
        require(type(snapshot) is dict,"actual recorded constructor raw snapshot required")
        env._send_command(dict(type="end",battleId="b1"))
        deadline=time.monotonic()+env.config.read_timeout_seconds;ended=False;ends=set()
        while not ended or ends!={"omniscient","p1","p2"}:
            remaining=deadline-time.monotonic();require(remaining>0,"recorded constructor teardown observation timed out")
            event=env._read_event(timeout=remaining)
            if event is None:continue
            require(event.get("battleId")=="b1","foreign recorded constructor teardown")
            if event.get("type")=="ended":require(not ended,"duplicate recorded ended");ended=True
            elif event.get("type")=="stream_end":
                require(event.get("stream") in {"omniscient","p1","p2"} and event["stream"] not in ends,
                    "duplicate/unknown recorded constructor stream end");ends.add(event["stream"])
            else:require(False,"unexpected recorded constructor teardown event")
        env._battle_token=None
    require(digest(inputs)==frozen,"original recorded overlay changed during actual constructor")
    return dict(inputs=deepcopy(inputs),capture=tap.export(snapshot=snapshot),snapshot=snapshot)


def materialization_source_hashes():
    here = Path(__file__).resolve()
    paths = (here,here.with_name("search_over_raw_outcome_run.py"),here.parents[1]/"local_showdown.py",
        here.parents[3]/"scripts"/"battle_bridge.mjs")
    return {str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}


def materialization_inputs(*, state, override, seed, root_binding, options=None):
    """Freeze public constructor inputs before its actual call, not from outputs."""
    from ..local_showdown import (PublicBattleMaterializationState,_public_materialization_payload,
        _start_players_payload,showdown_seed_from_int)
    require(isinstance(state,PublicBattleMaterializationState) and not state.replay.requests
        and state.replay.winner is None and state.player_id in {"p1","p2"}
        and type(seed) is int and type(root_binding) is dict and root_binding,
        "public-only nonterminal constructor root and exact seed required")
    draw=asdict(override)
    require(set(draw["player_teams"]) == {"p1","p2"}
        and all(type(v) is str and v for v in draw["player_teams"].values()),
        "both frozen explicit sampled teams required; no random source generation")
    options={} if options is None else deepcopy(options)
    require(type(options) is dict and set(options) <= set(BOOL_OPTIONS)|set(MAP_OPTIONS),
        "known materializer options only")
    values={key:options.get(key,False) for key in BOOL_OPTIONS}
    values.update({key:options.get(key) for key in MAP_OPTIONS})
    require(all(type(values[key]) is bool for key in BOOL_OPTIONS)
        and all(values[key] is None or type(values[key]) is dict for key in MAP_OPTIONS),
        "exact constructor option types required")
    public=_public_materialization_payload(state,
        deferred_opponent_actions=values["deferred_opponent_actions"],
        deferred_opponent_action_priors=values["deferred_opponent_action_priors"],
        reference_consumed_items=values["reference_consumed_items"],
        reference_encore=values["reference_encore_durations"] is not None,
        reference_induced_sleep=values["reference_induced_sleep"] is not None,
        reference_turn_clocks=values["reference_turn_clocks"],
        reference_attract=values["reference_attract"],reference_yawn=values["reference_yawn"])
    materialize=dict(type="materialize",battleId="b1",publicState=deepcopy(public),
        referenceRestSleep=values["reference_rest_sleep"],referenceConsumedItems=values["reference_consumed_items"],
        referenceTurnClocks=values["reference_turn_clocks"],referenceAttract=values["reference_attract"],
        referenceYawn=values["reference_yawn"],referenceEncoreDurations=values["reference_encore_durations"],
        referenceInducedSleep=values["reference_induced_sleep"],
        referenceRulesFormat=state.observation_format_id if values["reference_induced_sleep"] is not None else None)
    return validate_materialization_inputs(dict(root_binding=deepcopy(root_binding),public_state_sha256=digest(_state_payload(state)),
        public_payload_sha256=digest(public),observation_format_id=state.observation_format_id,
        draw_sha256=digest(draw),draw=draw,seed=seed,options=values,
        start=dict(type="start",battleId="b1",formatid=override.format_id,
            seed=showdown_seed_from_int(seed),players=_start_players_payload(override)),
        materialize=materialize,source_hashes=materialization_source_hashes()))


class MaterializationCapture:
    """One actual constructor call in an unused shell; no late rehabilitation."""
    def __init__(self, inputs):
        validate_materialization_inputs(inputs)
        self.inputs=deepcopy(inputs);self.input_sha=digest(inputs)
        self.records=[];self.raw=[];self.used=False;self.failed=False;self.env=None

    def _check(self):
        require(not self.failed and digest(self.inputs)==self.input_sha
            and self.inputs["source_hashes"]==materialization_source_hashes(),"materialization witness/input drift")

    @contextmanager
    def installed(self,env):
        from ..local_showdown import LocalShowdownEnv
        self._check()
        require(isinstance(env,LocalShowdownEnv) and env._process is None and env._battle_token is None
            and env._battle_counter==0 and env._bridge_evidence_observer is None and not self.used,
            "fresh exclusively observed hypothetical constructor shell required")
        self.used,self.env=True,env;env._bridge_evidence_observer=self
        try: yield self
        except BaseException:
            self.failed=True
            raise
        finally:
            require(env._bridge_evidence_observer is self,"materialization observer replaced")
            env._bridge_evidence_observer=None;self.env=None

    def _record(self,kind,payload):
        try:
            self._check()
            require(self.env is not None,"materialization observation outside installed seam")
            self.records.append(_redact(kind,payload))
            if kind=="event" and payload.get("type") in {"materialized","snapshot"}:
                self.raw.append(deepcopy(dict(payload)))
        except BaseException:
            self.failed=True
            raise

    def command(self,payload): self._record("command",payload)
    def event(self,payload): self._record("event",payload)

    def export(self,*,snapshot):
        self._check();require(self.used and self.env is None,"constructor tap must be removed before export")
        capture=dict(schema=SCHEMA,input_sha256=self.input_sha,records=deepcopy(self.records),
            raw_events=deepcopy(self.raw),private_controller_only=True)
        read_materialization(capture,expected_inputs=self.inputs,process_records=self.records,
            expected_snapshot=snapshot)
        return capture


def observe_materialized_world(env,*,state,override,seed,root_binding,options=None):
    """One explicitly sampled constructor, root capture and teardown; never retry.

    This helper has no authority or process-owning wrapper of its own. Callers
    must independently own/qualify the process before accepting its output.
    The ordinary snapshot command's request-refresh law is retained explicitly.
    """
    inputs=materialization_inputs(state=state,override=override,seed=seed,root_binding=root_binding,options=options)
    def revalidate():
        require(materialization_inputs(state=state,override=override,seed=seed,root_binding=root_binding,options=options)==inputs,
            "constructor input changed during actual call")
    return _observe_paired_materialization(env,state=state,override=override,inputs=inputs,revalidate=revalidate)


def _observe_paired_materialization(env,*,state,override,inputs,revalidate,tap=None):
    """Shared actual initializer/snapshot/teardown, with caller-specific inputs.

    The caller must rederive its full law before entry and after teardown. This
    seam does not reconstruct Python context from a recorded raw snapshot.
    """
    import time
    revalidate()
    tap=MaterializationCapture(inputs) if tap is None else tap
    require(isinstance(tap,MaterializationCapture) and digest(tap.inputs)==digest(inputs),
        "original constructor capture inputs required")
    with tap.installed(env):
        env.materialize_public_world(state=state,start_override=override,seed=inputs["seed"],**inputs["options"])
        snapshot=env.snapshot()
        env._send_command(dict(type="end",battleId="b1"))
        deadline=time.monotonic()+env.config.read_timeout_seconds
        ended=False;ends=set()
        while not ended or ends!={"omniscient","p1","p2"}:
            remaining=deadline-time.monotonic();require(remaining>0,"constructor teardown observation timed out")
            event=env._read_event(timeout=remaining)
            if event is None:continue
            require(event.get("battleId")=="b1","foreign constructor teardown event")
            if event.get("type")=="ended":require(not ended,"duplicate constructor ended");ended=True
            elif event.get("type")=="stream_end":
                require(event.get("stream") in {"omniscient","p1","p2"} and event["stream"] not in ends,
                    "duplicate/unknown constructor stream end");ends.add(event["stream"])
            else:require(False,"unexpected constructor teardown event")
        env._battle_token=None
    revalidate()
    return dict(inputs=inputs,capture=tap.export(snapshot=snapshot.bridge_snapshot),snapshot=snapshot)


def read_materialization(capture,*,expected_inputs,process_records,expected_snapshot):
    """Read exact actual start/materialize/snapshot/end transport; not ownership."""
    before=digest(dict(capture=capture,inputs=expected_inputs,records=process_records,snapshot=expected_snapshot))
    saved=deepcopy(capture);inputs=deepcopy(expected_inputs)
    validate_materialization_inputs(inputs)
    require(inputs["source_hashes"]==materialization_source_hashes()
        and type(saved) is dict and set(saved)=={"schema","input_sha256","records","raw_events","private_controller_only"}
        and saved["schema"]==SCHEMA and saved["input_sha256"]==digest(inputs)
        and saved["private_controller_only"] is True and saved["records"]==process_records,
        "independent constructor/input/process transport join required")
    token=inputs["start"]["battleId"]
    commands=[v for v in saved["records"] if v["kind"]=="command"]
    expected=[inputs["start"],inputs["materialize"],dict(type="snapshot",battleId=token),dict(type="end",battleId=token)]
    require(commands==[_redact("command",v) for v in expected],"exact one-shot explicit constructor command roster required")
    raw=saved["raw_events"]
    require(type(raw) is list and len(raw)==2 and [v.get("type") for v in raw]==["materialized","snapshot"]
        and all(v.get("battleId")==token for v in raw)
        and digest(raw[1].get("snapshot"))==digest(expected_snapshot),"actual returned root snapshot required")
    stage=0;started=False;ready=False;ended=False;ends=set();raw_index=0
    for record in saved["records"]:
        require(type(record) is dict and set(record)=={"kind","payload"},"constructor transport schema drift")
        value=record["payload"];tag=value.get("type")
        if record["kind"]=="command":
            if tag=="start":require(stage==0,"repeated constructor start");stage=1
            elif tag=="materialize":require(stage==1 and started and ready,"start boundary incomplete");stage=2
            elif tag=="snapshot":require(stage==3,"materialization response absent");stage=4
            elif tag=="end":require(stage==5,"root snapshot absent");stage=6
        else:
            require(record["kind"]=="event" and value.get("battleId")==token,"foreign constructor event")
            if tag=="started":
                require(stage==1 and not started and value.get("formatid")==inputs["start"]["formatid"]
                    and value.get("seed")==inputs["start"]["seed"],"actual start acknowledgement drift");started=True
            elif tag=="ready":require(stage==1 and not ready,"late/duplicate constructor boundary");ready=True
            elif tag in {"materialized","snapshot"}:
                require(stage==(2 if tag=="materialized" else 4)
                    and record==_redact("event",raw[raw_index]),"raw constructor event/process bytes drift")
                stage+=1;raw_index+=1
            elif tag=="stream":
                require(stage==1 and value.get("stream") in {"omniscient","p1","p2"},
                    "late private stream invalidates settled constructor output")
            elif tag=="stream_end":
                require(stage==6 and value.get("stream") in {"omniscient","p1","p2"}
                    and value["stream"] not in ends,"early/duplicate constructor stream end");ends.add(value["stream"])
            elif tag=="ended":require(stage==6 and not ended,"duplicate constructor teardown acknowledgement");ended=True
            else: require(False,"unexpected constructor event")
    require(stage==6 and ended and ends=={"omniscient","p1","p2"},"constructor teardown/stream evidence incomplete")
    from .search_over_raw_boundary_transport import _requests
    require(type(raw[0].get("boundaryRequests")) is dict and type(expected_snapshot) is dict
        and not expected_snapshot["battle"]["ended"],"nonterminal materialized root required")
    _requests(expected_snapshot)
    require(before==digest(dict(capture=capture,inputs=expected_inputs,records=process_records,snapshot=expected_snapshot)),
        "constructor inputs changed during validation")
    return dict(schema=SCHEMA,capture_sha256=digest(saved),snapshot_sha256=digest(expected_snapshot),
        raw_materialized_requests_sha256=digest(raw[0]["boundaryRequests"]),
        snapshot_requests_sha256=digest(expected_snapshot["boundaryRequests"]),
        request_refresh_changed=digest(raw[0]["boundaryRequests"])!=digest(expected_snapshot["boundaryRequests"]),
        snapshot_law="existing_snapshot_command_may_refresh_requests",transport_consistent=True,
        process_ownership_validated=False,construction_provenance_validated=False,
        full_world_equivalence=False,branch_transition_validated=False,runtime_qualified=False,
        scientific_collection_authorized=False)
