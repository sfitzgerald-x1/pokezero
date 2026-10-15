"""Eight create-only bridge-owned AFTER cells, not source-admitted labels.

All cells are consumed together before any launch. This unit preserves a
prospective retained-event roster and bridge receipts; it cannot authenticate
the supplied native inventory, endpoint or actor histories. Those original
owning joins remain mandatory before scored scientific use.
"""
from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

from .search_over_raw import digest,require
from .search_over_raw_native_leaf_cells import NativeLeafCellPlan,leaf_cell_law
from .search_over_raw_after_owner import (
    OwnedAfterRawCell,execute_owned_after_cell,_enclosing,_enclosing_guard,_path,_owner_identity,capture_identity,
)
from .search_over_raw_after_raw import _actors
from .search_over_raw_outcome_run import verify_outcome_runtime,_runtime_inventory
from .search_over_raw_supervision import _save_new
from .search_over_raw_world_inputs import _state_payload

SCHEMA="pokezero.search-over-raw.bridge-owned-after-bank.v1"


def _inventory(directory):
    result=[]
    for p in sorted(_path(directory).rglob("*")):
        p=_path(p)
        require(p.is_dir() or p.is_file(),"unsupported AFTER bank receipt node")
        result.append((str(p),"directory" if p.is_dir() else hashlib.sha256(p.read_bytes()).hexdigest()))
    return tuple(result)


def _cell_state(cell):
    """Callback-free fingerprint of each original private cell and anchors."""
    if cell is None:return None
    require(type(cell) is OwnedAfterRawCell,"exact original AFTER bank cell required")
    return digest(dict(cell_id=id(cell),owner_id=id(cell.owner),capture_id=id(cell.capture),readout_id=id(cell.readout),
        owner=_owner_identity(cell.owner),capture=capture_identity(cell.capture),readout=cell.readout,
        actual_receipts=_inventory(cell.owner.directory),
        files=[(p,hashlib.sha256(b).hexdigest()) for p,b in cell.original_files],
        original_guard=cell.original_guard,readout_sha256=cell.readout_sha256))


def _crosswalk(cells,rows,claim):
    """Every logical cell must join the original claim and actual readout."""
    require(len(cells)==len(rows)==len(claim["cell_roster"])==8,"complete original AFTER bank roster required")
    mutable={"status","signed_outcome","bridge_process_ownership_validated"}
    for rep,(cell,row,original) in enumerate(zip(cells,rows,claim["cell_roster"])):
        require(set(row)==set(original) and all(digest(row[k])==digest(v) for k,v in original.items() if k not in mutable)
            and row["replicate"]==rep,"original AFTER row identity/coordinate changed")
        if cell is None:
            require(row["status"]=="ACTOR_HISTORY_UNAVAILABLE" and row["signed_outcome"] is None
                and row["bridge_process_ownership_validated"] is False
                and not all(claim["actor_history_available"].values()),"unowned AFTER cell cannot claim completion")
        else:
            private_claim=json.loads(dict(cell.original_files)[str(cell.owner.directory/"claim.json")])
            identity={k:row[k] for k in ("leaf_plan_sha256","event_id","cell_id","canonical_construction_ordinal","replicate")}
            require(str(cell.owner.directory)==row["directory"] and digest(private_claim["cell_identity"])==digest(identity)
                and cell.capture.binding["event_id"]==row["event_id"] and cell.capture.binding["replicate"]==rep
                and cell.capture.binding["subject"]==claim["event"]["model_subject"]
                and row["status"]==cell.readout["status"] and row["signed_outcome"]==cell.readout["signed_outcome"]
                and row["bridge_process_ownership_validated"] is cell.readout["bridge_process_ownership_validated"] is True,
                "original AFTER row/cell claim/readout mismatch")


@dataclass(frozen=True)
class OwnedAfterRawBank:
    cells: tuple
    result: dict
    directory: object
    original_files: tuple
    original_sha256: str
    scope: object
    scope_files: tuple
    ownership: dict
    ownership_sha256: str
    original_cells: tuple
    original_cell_states: tuple

    def __reduce_ex__(self,protocol):
        raise TypeError("private AFTER bank cannot enter durable worker transport")

    def _check_original(self):
        """Direct byte/object checks only: no child or runtime callbacks."""
        _enclosing_guard(self.scope,self.scope_files,self.ownership,self.ownership_sha256)
        require(self.cells is self.original_cells and digest(self.result)==self.original_sha256
            and _inventory(self.directory)==self.original_files
            and tuple(_cell_state(c) for c in self.cells)==self.original_cell_states,
            "original AFTER bank/readout/receipt roster changed")
        claim=json.loads((_path(self.directory)/"claim.json").read_bytes())
        _crosswalk(self.cells,self.result["cells"],claim)
        require(self.result["required_cells"]==8 and self.result["scored_terminal_labels"]==0
            and self.result["bridge_owned_cells"]==sum(c is not None for c in self.cells)
            and self.result["complete_transport_results"]==sum(r["status"]=="COMPLETE" for r in self.result["cells"]),
            "actual AFTER bank counts changed")

    def validate(self):
        self._check_original()
        for cell in self.cells:
            if cell is not None:cell.validate()
        # Never let the last child/helper callback establish a new baseline.
        self._check_original()
        return deepcopy(self.result)


def execute_owned_after_bank(*,ownership,plan,expected_plan_sha256,event_id,raw_after,programs,config,runtime,policies,
        _original_admission=None):
    """Actual eight-cell bridge lane in original active auditor, no retry.

    Explicit None histories keep every reserved cell uncertain without launch.
    Malformed histories, process loss, provenance/receipt drift or cleanup error
    stop this auditor's AFTER subsystem, with untouched cells kept unstarted.
    This does not relax the separate collection or holdout admission gates.
    """
    scope,scope_files,ownership_sha=_enclosing(ownership)
    if _original_admission is not None:
        from .search_over_raw_after_admission import OwnedNativeAfterAdmission
        require(type(_original_admission) is OwnedNativeAfterAdmission,"original private AFTER admission required")
        _original_admission.validate()
        require(_original_admission.raw_after is raw_after and _original_admission.programs is programs
            and _original_admission.receipt["event_id"]==event_id
            and _original_admission.receipt["leaf_plan_sha256"]==expected_plan_sha256,
            "bank inputs differ from original private AFTER admission")
    require(type(plan) is NativeLeafCellPlan and plan.sha256==expected_plan_sha256,
        "original externally frozen retained-leaf plan required")
    saved=plan.receipt()
    require(saved["law"]==leaf_cell_law() and type(programs) is dict and set(programs)=={"p1","p2"}
        and all(v is None or type(v) is dict for v in programs.values()),
        "original AFTER law and explicit supported/absent actor histories required")
    matches=[e for e in saved["events"] if e["event_id"]==event_id]
    require(len(matches)==1,"one original retained event required; no replacement frontier")
    event=matches[0]
    require(event["canonical_construction_ordinal"]==min(event["construction_aliases"])
        and len(event["labels"])==8 and all(type(c["replicate"]) is int and c["replicate"]==i
            and c["cell_id"]==digest(dict(event_id=event_id,replicate=i))
            and c["status"]=="UNSTARTED_UNCERTAIN" and c["signed_outcome"] is None
            for i,c in enumerate(event["labels"])),"original canonical alias and complete eight-cell roster required")
    namespace=_path(scope.directory/"after-banks")
    namespace.mkdir(exist_ok=True,mode=0o700)
    require(not (namespace/"operational-stop.json").exists(),"original AFTER subsystem stopped; no retry or continuation")
    directory=_path(namespace/digest(dict(plan_sha256=expected_plan_sha256,event_id=event_id)))
    runtime_sha=verify_outcome_runtime(runtime,config)
    inputs_sha=digest(dict(root=raw_after,programs=programs,config=_state_payload(config),runtime=runtime))
    identities=[dict(leaf_plan_sha256=expected_plan_sha256,event_id=event_id,cell_id=c["cell_id"],
        canonical_construction_ordinal=event["canonical_construction_ordinal"],replicate=i)
        for i,c in enumerate(event["labels"])]
    rows=[dict(**identity,directory=str(directory/("cell-"+identity["cell_id"])),status="UNSTARTED_UNCERTAIN",
        signed_outcome=None,bridge_process_ownership_validated=False) for identity in identities]
    claim=dict(schema=SCHEMA,leaf_plan_sha256=expected_plan_sha256,event=deepcopy(event),cell_roster=deepcopy(rows),
        actor_history_available={s:programs[s] is not None for s in ("p1","p2")},
        runtime_sha256=runtime_sha,private_inputs_sha256=inputs_sha,enclosing_binding_sha256=scope.binding,
        all_eight_cells_consumed=True,alias_fallback_permitted=False,retries_permitted=False,scientific_collection_authorized=False)
    directory.mkdir(mode=0o700,exist_ok=False)
    _save_new(directory/"claim.json",claim)
    claim_bytes=(directory/"claim.json").read_bytes()
    owned=[None]*8
    owned_states=[None]*8
    row_sha=digest(rows);identity_sha=digest(identities)
    result_bytes=None
    original_policies=tuple(policies[s] for s in ("p1","p2")) if type(policies) is dict and set(policies)=={"p1","p2"} else None
    def final_guard():
        # Direct current file bytes and full state, without runtime or child
        # callbacks that could change a previously checked value.
        _enclosing_guard(scope,scope_files,ownership,ownership_sha)
        if _original_admission is not None:_original_admission.validate()
        require(plan.sha256==expected_plan_sha256 and (directory/"claim.json").read_bytes()==claim_bytes
            and _path(directory)==directory and _path(namespace)==namespace
            and not (namespace/"operational-stop.json").exists()
            and inputs_sha==digest(dict(root=raw_after,programs=programs,config=_state_payload(config),runtime=runtime))
            and digest(runtime)==runtime_sha
            and set(runtime["files"])==_runtime_inventory(runtime["argv"],runtime["checkpoint_path"],runtime["extra_files"])
            and all(hashlib.sha256(_path(p).read_bytes()).hexdigest()==sha for p,sha in runtime["files"].items())
            and digest(rows)==row_sha and digest(identities)==identity_sha
            and [_cell_state(c) for c in owned]==owned_states
            and {p.name for p in directory.iterdir()}=={"claim.json"}|({"result.json"} if result_bytes is not None else set())|{
                Path(rows[i]["directory"]).name for i,c in enumerate(owned) if c is not None}
            and (result_bytes is None or (directory/"result.json").read_bytes()==result_bytes)
            and (original_policies is None or all(policies[s] is p for s,p in zip(("p1","p2"),original_policies))),
            "original AFTER bank inputs/claim/policies changed")
    def guard():
        require(verify_outcome_runtime(runtime,config)==runtime_sha,"original AFTER bank runtime changed")
        final_guard()
        for cell in owned:
            if cell is not None:cell.validate()
        final_guard()
    try:
        guard()
        if any(programs[s] is None for s in ("p1","p2")):
            for row in rows:row["status"]="ACTOR_HISTORY_UNAVAILABLE"
            row_sha=digest(rows)
        else:
            _actors(programs,raw_after,config)  # Validate complete cursors BEFORE any cell's bridge launch.
            guard()
            for rep,identity in enumerate(identities):
                guard()
                rows[rep]["status"]="ATTEMPTED_UNCERTAIN"
                row_sha=digest(rows)
                cell=execute_owned_after_cell(ownership=ownership,directory=Path(rows[rep]["directory"]),
                    raw_after=raw_after,programs=programs,config=config,runtime=runtime,sample_seed=saved["sample_seed"],
                    event_id=event_id,replicate=rep,subject=event["model_subject"],policies=policies,cell_identity=identity)
                owned[rep]=cell
                owned_states[rep]=_cell_state(cell)
                guard()
                rows[rep].update(status=cell.readout["status"],signed_outcome=cell.readout["signed_outcome"],
                    bridge_process_ownership_validated=True)
                row_sha=digest(rows)
        guard()
        _crosswalk(owned,rows,claim)
        require({p.name for p in directory.iterdir()}=={"claim.json"}|{
            Path(rows[i]["directory"]).name for i,c in enumerate(owned) if c is not None},
            "unexpected AFTER bank cell/receipt namespace")
        result=dict(schema=SCHEMA,leaf_plan_sha256=expected_plan_sha256,event_id=event_id,
            construction_aliases=deepcopy(event["construction_aliases"]),canonical_construction_ordinal=event["canonical_construction_ordinal"],
            cells=deepcopy(rows),required_cells=8,bridge_owned_cells=sum(c is not None for c in owned),
            complete_transport_results=sum(r["status"]=="COMPLETE" for r in rows),scored_terminal_labels=0,
            native_inventory_owning_validated=False,original_endpoint_provenance_validated=False,
            actor_history_provenance_validated=False,enclosing_auditor_terminal_validated=False,
            independent_terminal_labels=False,resource_cost_complete=False,runtime_qualified=False,
            scientific_collection_authorized=False,holdout_authorized=False,retries_permitted=False)
        _save_new(directory/"result.json",result)
        result_bytes=(directory/"result.json").read_bytes()
        original_cells=tuple(owned)
        bank=OwnedAfterRawBank(original_cells,result,directory,_inventory(directory),digest(result),
            scope,scope_files,ownership,ownership_sha,original_cells,tuple(owned_states))
        bank.validate();guard()
        # guard() itself invokes callbacks. Check the original whole bank AND
        # external inputs after that final callback, without invoking another.
        final_guard();bank._check_original()
        return bank
    except BaseException as error:
        if not (namespace/"operational-stop.json").exists():
            _save_new(namespace/"operational-stop.json",dict(schema=SCHEMA,event_id=event_id,error_type=type(error).__name__,
                retries_permitted=False,scientific_collection_authorized=False))
        _save_new(directory/"failed-bank.json",dict(schema=SCHEMA,status="OPERATIONAL_STOP",cells=rows,
            error_type=type(error).__name__,scored_terminal_labels=0))
        raise
