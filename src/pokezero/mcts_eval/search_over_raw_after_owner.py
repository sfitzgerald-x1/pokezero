"""Bridge-owned private AFTER cells inside the original supervised auditor.

No actor program or private capture is serialized to another worker. Bridge
lifecycle clearance is separate from the enclosing auditor's eventual exit,
endpoint/history admission, native parity, resource completeness and science.
"""
from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import threading

from ..local_showdown import LocalShowdownEnv
from .search_over_raw import digest, require
from .search_over_raw_after_raw import (AfterRawCellCapture, run_after_raw_cell, read_after_raw_cell)
from .search_over_raw_boundary_run import raw_root_json
from .search_over_raw_outcome_run import ContinuationProcessOwner, verify_outcome_runtime
from .search_over_raw_nested_ownership import current_scope, OwnedScope
from .search_over_raw_supervision import _save_new
from .paper_reference_parallel import owned_process_identity
from .search_over_raw_world_inputs import _state_payload

SCHEMA = "pokezero.search-over-raw.bridge-owned-after-cell.v1"


def capture_identity(capture):
    require(type(capture) is AfterRawCellCapture,"exact original private AFTER capture required")
    return digest(dict(binding=capture.binding,transport=capture.transport,policy_calls=capture.policy_calls))


def _path(value):
    p=Path(value)
    require(p.is_absolute() and p==p.resolve() and not p.is_symlink(),"original canonical AFTER path required")
    return p


def _enclosing(ownership):
    scope=current_scope()
    require(type(scope) is OwnedScope and scope.pid==os.getpid()==os.getpgrp()
        and scope.thread_id==threading.get_ident() and type(ownership) is dict
        and ownership==dict(directory=str(scope.directory/"owned-groups"),controller_pid=scope.pid,
            registration_sha256=scope.binding,protocol="active-generations-v1"),
        "actual original enclosing auditor scope/registration required")
    directory=_path(scope.directory)
    paths=[directory/name for name in ("claim.json","launch.json","job.pickle")]
    raw=[_path(p).read_bytes() for p in paths]
    claim,launch=json.loads(raw[0]),json.loads(raw[1])
    require(claim["binding_sha256"]==scope.binding and claim["job_sha256"]==hashlib.sha256(raw[2]).hexdigest()
        and launch["binding_sha256"]==scope.binding and launch["pid"]==launch["group"]==scope.pid
        and launch["birth_identity"]==owned_process_identity(scope.pid),
        "actual enclosing supervisor claim/job/launch witness required")
    return scope,(digest(_state_payload(vars(scope))),tuple((str(p),b) for p,b in zip(paths,raw))),digest(ownership)


def _enclosing_guard(scope,files,ownership,sha):
    original_scope_sha,original_files=files
    require(current_scope() is scope and scope.pid==os.getpid()==os.getpgrp()
        and scope.thread_id==threading.get_ident() and digest(ownership)==sha
        and digest(_state_payload(vars(scope)))==original_scope_sha
        and all(_path(p).read_bytes()==b for p,b in original_files),"original enclosing AFTER auditor inputs changed")


class AfterRawProcessOwner(ContinuationProcessOwner):
    launch_schema="pokezero.search-over-raw.after-bridge-launch.v1"
    close_schema="pokezero.search-over-raw.after-bridge-close.v1"
    capture_hash_key="after_cell_sha256"

    def __init__(self,*,scope,scope_files,original_ownership,**kwargs):
        self.scope,self.scope_files,self.original_ownership=scope,scope_files,original_ownership
        self.ownership_sha=digest(original_ownership)
        _enclosing_guard(scope,scope_files,original_ownership,self.ownership_sha)
        super().__init__(**kwargs)
        self.original_config=self.config
        self.original_runtime=deepcopy(self.runtime)
        self.original_directory=_path(self.directory)
        self.private_capture=None
        self.private_capture_sha=None
        self.original_claim_bytes=_path(self.directory/"claim.json").read_bytes()
        self.started_refs=None

    def started(self,env,process):
        super().started(env,process)
        self.started_refs=(env,process,self.launch,digest(self.launch))
        self.original_inputs()

    def original_inputs(self):
        _enclosing_guard(self.scope,self.scope_files,self.original_ownership,self.ownership_sha)
        observed_runtime_sha=verify_outcome_runtime(self.runtime,self.config)
        _enclosing_guard(self.scope,self.scope_files,self.original_ownership,self.ownership_sha)
        require(type(self) is AfterRawProcessOwner and self.config is self.original_config
            and self.directory==self.original_directory and _path(self.directory)==self.original_directory
            and _path(self.directory/"claim.json").read_bytes()==self.original_claim_bytes
            and digest(self.runtime)==digest(self.original_runtime)
            and observed_runtime_sha==self.runtime_sha,
            "original AFTER process owner inputs changed")
        require(self.started_refs is None or self.env is self.started_refs[0] and self.process is self.started_refs[1]
            and self.launch is self.started_refs[2] and digest(self.launch)==self.started_refs[3],
            "original AFTER started environment/process/launch replaced")

    def seal(self,capture):
        self.original_inputs()
        require(self.phase=="started" and self.bank_sha is None and self.env._battle_token is None
            and self.env._bridge_boundary_observer is None and self.env._bridge_evidence_observer is None,
            "released AFTER capture before close required")
        require(type(capture) is AfterRawCellCapture and capture.transport["records"]==self.records,
            "separately observed original AFTER transport required")
        self.private_capture=capture
        self.private_capture_sha=capture_identity(capture)
        self.bank_sha=self.private_capture_sha

    def _require_close_ready(self,env):
        self.original_inputs()
        super()._require_close_ready(env)
        require(type(self.private_capture) is AfterRawCellCapture
            and capture_identity(self.private_capture)==self.private_capture_sha==self.bank_sha,
            "sealed original AFTER capture changed before close")


@dataclass(frozen=True)
class OwnedAfterRawCell:
    owner: object
    capture: object
    readout: dict
    original_files: tuple
    original_guard: str
    readout_sha256: str

    def __reduce_ex__(self,protocol):
        raise TypeError("private owning AFTER cell cannot enter durable worker transport")

    def validate(self):
        require(type(self.owner) is AfterRawProcessOwner and self.owner.private_capture is self.capture,
            "original private AFTER cell owner replaced")
        self.owner.original_inputs()
        require(_owner_identity(self.owner)==self.original_guard
            and capture_identity(self.capture)==self.owner.private_capture_sha
            and digest(self.readout)==self.readout_sha256
            and {p.name for p in _path(self.owner.directory).iterdir()}=={Path(p).name for p,b in self.original_files}
            and all(_path(p).read_bytes()==b for p,b in self.original_files),
            "original validated AFTER capture/readout/receipt bytes changed")
        return deepcopy(self.readout)


def _owner_identity(owner):
    return digest(dict(directory=str(owner.directory),ownership=owner.ownership,runtime=owner.runtime,
        scope=_state_payload(vars(owner.scope)),scope_id=id(owner.scope),
        config=_state_payload(owner.config),launch=owner.launch,records=owner.records,phase=owner.phase,
        bank_sha=owner.bank_sha,reaped_exit=None if owner.reaped is None else owner.reaped[0],
        process_returncode=None if owner.process is None else owner.process.returncode,
        process_id=id(owner.process),env_id=id(owner.env),capture_id=id(owner.private_capture),
        config_id=id(owner.config),runtime_id=id(owner.runtime),launch_id=id(owner.launch),records_id=id(owner.records),
        wait_ownership_lost=owner.wait_ownership_lost,private_capture_sha=owner.private_capture_sha))


def read_owned_after_cell(owner,capture,*,expected_claim,raw_after,programs,config,runtime):
    """Distinct in-process reader of actual released/reaped original child."""
    require(type(owner) is AfterRawProcessOwner and owner.private_capture is capture,
        "exact original AFTER process owner/private capture required")
    owner.original_inputs()
    directory=_path(owner.directory)
    names={"claim.json","bridge-launch.json","bridge-transport.json","bridge-close.json"}
    require({p.name for p in directory.iterdir()}==names,"exact original AFTER receipt roster required")
    files=tuple((str(directory/n),_path(directory/n).read_bytes()) for n in sorted(names))
    loaded={Path(p).name:json.loads(b) for p,b in files}
    claim,launch,closed,records=(loaded[n] for n in ("claim.json","bridge-launch.json","bridge-close.json","bridge-transport.json"))
    records=records["records"]
    frozen_inputs=digest(dict(claim=expected_claim,root=raw_after,programs=programs,config=_state_payload(config),runtime=runtime))
    frozen_owner=_owner_identity(owner);frozen_capture=capture_identity(capture)
    refs=(owner.env,owner.process,owner.config,owner.runtime,owner.launch,owner.records,owner.private_capture)
    require(digest(claim)==digest(expected_claim) and claim["enclosing_binding_sha256"]==owner.scope.binding
        and claim["enclosing_directory"]==str(owner.scope.directory)
        and launch==owner.launch and launch["schema"]==owner.launch_schema
        and launch["parent_pid"]==launch["group"]==owner.scope.pid
        and launch["binding_sha256"]==owner.scope.binding and launch["runtime_sha256"]==digest(runtime)
        and type(launch["pid"]) is int and launch["pid"]==owner.process.pid,
        "original AFTER claim/enclosing/actual bridge launch mismatch")
    require(all(digest(claim[k])==digest(v) for k,v in capture.binding.items()
        if k not in {"schema","policy","maximum_boundaries","checkpoint_sha256"}),
        "original AFTER claim differs from actual captured input envelope")
    require(closed["schema"]==owner.close_schema and closed["launch_sha256"]==digest(launch)
        and closed["after_cell_sha256"]==frozen_capture==owner.bank_sha
        and closed["transport_sha256"]==digest(records) and records==owner.records
        and closed["pid"]==launch["pid"] and closed["parent_pid"]==launch["parent_pid"]
        and closed["binding_sha256"]==owner.scope.binding and closed["runtime_sha256"]==digest(runtime)
        and closed["normal_close"] is True and closed["forced_cleanup"] is False and closed["reaped"] is True
        and closed["wait_ownership_lost"] is False and closed["pumps_drained"] is True
        and type(closed["exit_code"]) is int and closed["exit_code"]==0 and closed["error_type"] is None
        and owner.phase=="closed" and owner.reaped is not None and owner.reaped[0]==0
        and owner.process.returncode==0 and owner.wait_ownership_lost is False
        and owner.env._process is None and owner.env._battle_token is None,
        "normal same-child AFTER release/close/reap evidence required")
    require(records[-2:]==[dict(kind="command",payload=dict(type="close")),dict(kind="event",payload=dict(type="closed"))],
        "exact AFTER close trailer required")
    require(capture.transport["records"]==records[:-2],"actual AFTER private capture close segment mismatch")
    result=read_after_raw_cell(capture,process_records=records[:-2],raw_after=raw_after,programs=programs,
        config=config,expected_binding=deepcopy(capture.binding),runtime=runtime)
    owner.original_inputs()
    require(all(a is b for a,b in zip((owner.env,owner.process,owner.config,owner.runtime,owner.launch,owner.records,
        owner.private_capture),refs)) and _owner_identity(owner)==frozen_owner and capture_identity(capture)==frozen_capture
        and digest(dict(claim=expected_claim,root=raw_after,programs=programs,config=_state_payload(config),runtime=runtime))==frozen_inputs
        and {p.name for p in directory.iterdir()}==names and all(_path(p).read_bytes()==b for p,b in files),
        "original AFTER owner/inputs/receipt bytes changed during reconstruction")
    result.update(schema=SCHEMA,bridge_process_ownership_validated=True,enclosing_auditor_terminal_validated=False,
        original_endpoint_provenance_validated=False,actor_history_provenance_validated=False,
        independent_terminal_labels=False,resource_cost_complete=False)
    return OwnedAfterRawCell(owner,capture,result,files,_owner_identity(owner),digest(result))


def execute_owned_after_cell(*,ownership,directory,raw_after,programs,config,runtime,
        sample_seed,event_id,replicate,subject,policies,cell_identity):
    """One create-only private cell in original active auditor; no admission."""
    scope,scope_files,ownership_sha=_enclosing(ownership)
    require(type(cell_identity) is dict and set(cell_identity)=={"leaf_plan_sha256","event_id","cell_id",
            "canonical_construction_ordinal","replicate"}
        and cell_identity["event_id"]==event_id and type(cell_identity["replicate"]) is int
        and cell_identity["replicate"]==replicate and cell_identity["cell_id"]==digest(dict(event_id=event_id,replicate=replicate))
        and type(cell_identity["canonical_construction_ordinal"]) is int
        and 0 <= cell_identity["canonical_construction_ordinal"] < 4
        and type(cell_identity["leaf_plan_sha256"]) is str and len(cell_identity["leaf_plan_sha256"])==64,
        "original logical AFTER leaf-plan/event/canonical alias/replicate identity required")
    directory=_path(directory)
    require(directory!=scope.directory and scope.directory in directory.parents,
        "AFTER cell must remain below its original auditor")
    raw=raw_root_json(raw_after)
    token="after-"+digest(cell_identity)
    clone=dict(battleId=token,cloneId="cell-"+digest(cell_identity),snapshotJsonSha256=hashlib.sha256(raw.encode()).hexdigest())
    request_ids=[token+":"+str(i) for i in range(251)]
    claim=dict(schema=SCHEMA,cell_identity=deepcopy(cell_identity),enclosing_directory=str(scope.directory),
        enclosing_binding_sha256=scope.binding,raw_after_sha256=digest(raw_after),programs_sha256=digest(programs),
        runtime_sha256=verify_outcome_runtime(runtime,config),config_sha256=digest(_state_payload(config)),
        sample_seed=sample_seed,event_id=event_id,replicate=replicate,subject=subject,clone_binding=clone,
        request_ids=request_ids,retries_permitted=False,scientific_collection_authorized=False)
    directory.mkdir(mode=0o700,exist_ok=False)
    _save_new(directory/"claim.json",claim)
    _enclosing_guard(scope,scope_files,ownership,ownership_sha)
    env=LocalShowdownEnv(config)
    owner=AfterRawProcessOwner(scope=scope,scope_files=scope_files,original_ownership=ownership,
        directory=directory,ownership=ownership,runtime=runtime,config=config)
    with owner.installed(env):
        capture=run_after_raw_cell(env,raw_after=raw_after,programs=programs,config=config,clone_binding=clone,
            request_ids=request_ids,sample_seed=sample_seed,event_id=event_id,replicate=replicate,subject=subject,
            policies=policies,checkpoint_sha256=runtime["files"][runtime["checkpoint_path"]],runtime=runtime,process_owner=owner)
    _enclosing_guard(scope,scope_files,ownership,ownership_sha)
    return read_owned_after_cell(owner,capture,expected_claim=claim,raw_after=raw_after,programs=programs,config=config,runtime=runtime)
