"""Opt-in preclaim census of concrete supervised jobs, not scientific slots.

Declarations are create-only by canonical directory, including failed spawns.
Each PID/thread-bound writer retains original child bytes through its close;
the worker's actual terminal receipt binds that close. Missing closes remain
partial, never an exhaustive seal or complete-workflow cost assertion.
"""
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import asdict,dataclass
import hashlib
import os
from pathlib import Path
import threading

from .search_over_raw import digest,require
from .search_over_raw_continuation_admission import _path
from .search_over_raw_outcomes import _sha
from .search_over_raw_ledger import _write_new
from .search_over_raw_attempt_cost import AttemptCostSpec,_json,_inventory,_attempt,reconcile_attempt_cost_roster,MAX_COST_ATTEMPTS
from .search_over_raw_nested_ownership import current_scope,MAX_CHILDREN,MAX_DEPTH

SCHEMA="pokezero.search-over-raw.attempt-declaration-journal.v1"
_CURRENT=ContextVar("search_over_raw_attempt_journal",default=None)


def _hash(path):return hashlib.sha256(_path(path).read_bytes()).hexdigest()


def _framework():
    directory=Path(__file__).resolve().parent
    return {str(_path(directory/name)):_hash(directory/name) for name in (
        "search_over_raw_attempt_journal.py","search_over_raw_attempt_cost.py",
        "search_over_raw_supervision.py","search_over_raw_nested_ownership.py","search_over_raw_resources.py",
        "search_over_raw_observer_cost.py","paper_reference_parallel.py")}


@dataclass(frozen=True)
class AttemptJournalHandle:
    directory: str
    metadata_sha256: str
    def __post_init__(self):
        require(type(self.directory) is str and str(_path(self.directory))==self.directory
            and _sha(self.metadata_sha256),"original canonical attempt journal handle required")


def journal_metadata(handle,*,producer=False):
    require(type(handle) is AttemptJournalHandle,"original typed attempt journal handle required")
    path=_path(Path(handle.directory)/"metadata.json")
    require(_hash(path)==handle.metadata_sha256,"original attempt journal metadata drift")
    meta=_json(path)
    require(type(meta) is dict and set(meta)=={"schema","attempt_root","outer_attempt","framework_hashes",
        "scope","work_validated","scientific_collection_authorized","holdout_authorized","resource_cost_complete"}
        and meta["schema"]==SCHEMA and meta["scope"]=="concrete shared-supervisor invocations; not unconstructed work or ordinary actors"
        and all(meta[k] is False for k in ("work_validated","scientific_collection_authorized","holdout_authorized","resource_cost_complete")),
        "original cost-only journal scope required")
    outer=AttemptCostSpec(**meta["outer_attempt"]);root=_path(meta["attempt_root"])
    journal=_path(handle.directory)
    require(outer.parent_directory is None and _path(outer.directory).is_relative_to(root)
        and journal.is_relative_to(root) and journal!=root and _path(outer.directory)!=root
        and not journal.is_relative_to(_path(outer.directory)) and not _path(outer.directory).is_relative_to(journal),
        "journal and original outer attempt must be distinct within fixed root")
    require(type(meta["framework_hashes"]) is dict and all(type(p) is str and _sha(sha)
        and str(_path(p))==p for p,sha in meta["framework_hashes"].items()),"journal framework binding malformed")
    if producer:require(meta["framework_hashes"]==_framework(),"original journal producer framework drift")
    return meta,outer


def create_attempt_journal(directory,*,outer_attempt,attempt_root):
    require(type(outer_attempt) is AttemptCostSpec and outer_attempt.parent_directory is None,
        "original outer attempt descriptor required before journal creation")
    directory=_path(directory);root=_path(attempt_root)
    require(directory.is_relative_to(root) and directory!=root
        and _path(outer_attempt.directory).is_relative_to(root) and _path(outer_attempt.directory)!=root
        and not directory.is_relative_to(_path(outer_attempt.directory))
        and not _path(outer_attempt.directory).is_relative_to(directory),"distinct journal namespace within attempt root required")
    meta=dict(schema=SCHEMA,attempt_root=str(root),outer_attempt=asdict(outer_attempt),framework_hashes=_framework(),
        scope="concrete shared-supervisor invocations; not unconstructed work or ordinary actors",
        work_validated=False,scientific_collection_authorized=False,holdout_authorized=False,resource_cost_complete=False)
    directory.mkdir(mode=0o700,parents=True,exist_ok=False)
    (directory/"declarations").mkdir();(directory/"writer-closes").mkdir()
    _write_new(directory/"metadata.json",meta)
    handle=AttemptJournalHandle(str(directory),_hash(directory/"metadata.json"))
    journal_metadata(handle,producer=True)
    return handle


def _name(directory):return digest(str(_path(directory)))+".json"


def _declaration(handle,path):
    path=_path(path);row=_json(path)
    require(type(row) is dict and set(row)=={"schema","metadata_sha256","spec","spec_sha256","parent_declaration_sha256"}
        and row["schema"]==SCHEMA+".declaration" and row["metadata_sha256"]==handle.metadata_sha256,
        "original attempt declaration metadata drift")
    spec=AttemptCostSpec(**row["spec"])
    require(row["spec_sha256"]==digest(asdict(spec)) and path==Path(handle.directory)/"declarations"/_name(spec.directory),
        "original directory-keyed attempt declaration drift")
    return row,spec


def _permission(handle,path):
    return dict(journal_directory=handle.directory,metadata_sha256=handle.metadata_sha256,
        declaration_path=str(_path(path)),declaration_sha256=_hash(path))


def _permission_handle(permission):
    require(type(permission) is dict and set(permission)=={"journal_directory","metadata_sha256","declaration_path","declaration_sha256"}
        and _sha(permission["declaration_sha256"]),"original closed journal bootstrap reference required")
    handle=AttemptJournalHandle(permission["journal_directory"],permission["metadata_sha256"])
    path=_path(permission["declaration_path"])
    require(path.parent==Path(handle.directory)/"declarations" and _hash(path)==permission["declaration_sha256"],
        "original declaration bootstrap bytes drift")
    return handle,path


class _Writer:
    def __init__(self,handle,permission=None):
        self.handle,self.permission=handle,permission
        self.pid,self.thread=os.getpid(),threading.get_ident()
        self.children={};self.close_sha256=None
    def verify(self):
        require(os.getpid()==self.pid and threading.get_ident()==self.thread,
            "journal writer PID/thread drift; no declaration or spawn")
        journal_metadata(self.handle,producer=True)
        scope=current_scope()
        if self.permission is None:require(scope is None,"outer journal context cannot replace inherited scope")
        else:
            handle,path=_permission_handle(self.permission);_,spec=_declaration(handle,path)
            require(scope is not None and scope.directory==_path(spec.directory) and scope.binding==spec.binding_sha256
                and scope.pid==self.pid,"journal context lost original direct owner")
        require(all(_hash(path)==sha for path,sha in self.children.items()),"original child declaration changed or removed")
    def close(self):
        self.verify()
        key="outer-controller.json" if self.permission is None else _name(current_scope().directory)
        row=dict(schema=SCHEMA+".writer-close",metadata_sha256=self.handle.metadata_sha256,
            owner_declaration_sha256=None if self.permission is None else self.permission["declaration_sha256"],
            writer_pid=self.pid,declared_children=dict(self.children),
            supervised_attempt_population_only=True,work_validated=False,complete_workflow_cost=False)
        path=Path(self.handle.directory)/"writer-closes"/key
        _write_new(path,row);self.close_sha256=_hash(path)


def current_attempt_journal():
    writer=_CURRENT.get()
    if writer is not None:writer.verify()
    return writer


@contextmanager
def outer_attempt_journal_scope(handle):
    require(_CURRENT.get() is None and current_scope() is None,"attempt journal context already installed")
    journal_metadata(handle,producer=True);writer=_Writer(handle)
    token=_CURRENT.set(writer)
    primary=None
    try:yield writer
    except BaseException as error:
        primary=error;raise
    finally:
        try:
            try:writer.close()
            except BaseException as error:
                if primary is None:raise
                primary.add_note("attempt journal writer close failed: "+type(error).__name__)
        finally:_CURRENT.reset(token)


def declare_supervised_attempt(spec):
    writer=current_attempt_journal()
    if writer is None:return None
    require(type(spec) is AttemptCostSpec,"original concrete supervised cost descriptor required")
    meta,outer=journal_metadata(writer.handle,producer=True);scope=current_scope()
    require(_path(spec.directory).is_relative_to(_path(meta["attempt_root"]))
        and not _path(spec.directory).is_relative_to(Path(writer.handle.directory)),"attempt escaped original journal namespace")
    if writer.permission is None:
        require(digest(asdict(spec))==digest(asdict(outer)) and not writer.children,"outer declaration must be original and one-shot")
        parent_sha=None
    else:
        require(scope is not None and spec.parent_directory==str(scope.directory),"nested declaration lost direct parent")
        parent_sha=writer.permission["declaration_sha256"]
    require(not _path(spec.directory).exists(),"occupied attempt cannot be adopted by journal")
    registry=Path(writer.handle.directory)/"declarations"
    require(len(writer.children)<MAX_CHILDREN,"direct writer declaration inventory exhausted")
    path=registry/_name(spec.directory)
    _write_new(path,dict(schema=SCHEMA+".declaration",metadata_sha256=writer.handle.metadata_sha256,
        spec=asdict(spec),spec_sha256=digest(asdict(spec)),parent_declaration_sha256=parent_sha))
    writer.children[str(path)]=_hash(path)
    writer.verify()
    return _permission(writer.handle,path)


@contextmanager
def worker_attempt_journal_scope(permission,directory,binding,payload_sha256):
    """Mandatory original claim/header reference, before trusted job loading."""
    require(_CURRENT.get() is None,"worker journal context already installed")
    claim=_json(Path(directory)/"claim.json")
    require(claim.get("attempt_declaration")==permission,"claim/header journal reference drift")
    if permission is None:
        yield None;return
    handle,path=_permission_handle(permission);meta,outer=journal_metadata(handle,producer=True)
    row,spec=_declaration(handle,path);scope=current_scope()
    require(scope is not None and spec.directory==str(_path(directory)) and spec.binding_sha256==binding
        and spec.job_sha256==payload_sha256==claim["job_sha256"] and digest(spec.contract)==digest(claim["contract"])
        and spec.resource_source_sha256==claim["resource_measurement_source_sha256"],"worker declaration/job/claim binding drift")
    if spec.parent_directory is None:
        require(scope.depth==0 and row["parent_declaration_sha256"] is None
            and digest(asdict(spec))==digest(asdict(outer)),"original outer worker declaration required")
    else:
        parent_path=Path(handle.directory)/"declarations"/_name(spec.parent_directory)
        _,parent=_declaration(handle,parent_path)
        launch=_json(Path(directory)/"launch.json");parent_launch=_json(Path(parent.directory)/"launch.json")
        require(scope.depth>0 and row["parent_declaration_sha256"]==_hash(parent_path)
            and launch["supervisor_pid"]==parent_launch["pid"]==os.getppid()
            and parent_launch["binding_sha256"]==parent.binding_sha256,
            "original journal parent ownership drift")
    writer=_Writer(handle,permission);writer.verify();token=_CURRENT.set(writer)
    primary=None
    try:yield writer
    except BaseException as error:
        primary=error;raise
    finally:
        try:
            try:writer.close()
            except BaseException as error:
                if primary is None:raise
                primary.add_note("attempt journal writer close failed: "+type(error).__name__)
        finally:_CURRENT.reset(token)


def _terminal_writer_close(permission,directory,binding,pid):
    """Strict original writer header; not a normal-return or work witness."""
    handle,path=_permission_handle(permission);row,spec=_declaration(handle,path)
    require(spec.directory==str(_path(directory)) and spec.binding_sha256==binding,"writer-close owning job drift")
    close_path=Path(handle.directory)/"writer-closes"/_name(directory)
    sha=_hash(close_path);value=_json(close_path)
    require(type(value) is dict and set(value)=={"schema","metadata_sha256","owner_declaration_sha256","writer_pid",
        "declared_children","supervised_attempt_population_only","work_validated","complete_workflow_cost"}
        and value["schema"]==SCHEMA+".writer-close" and value["metadata_sha256"]==handle.metadata_sha256
        and value["owner_declaration_sha256"]==permission["declaration_sha256"]
        and type(value["writer_pid"]) is int and type(pid) is int and value["writer_pid"]==pid
        and value["supervised_attempt_population_only"] is True and value["work_validated"] is False
        and value["complete_workflow_cost"] is False and type(value["declared_children"]) is dict,
        "original reaped worker declaration close required")
    require(len(value["declared_children"])<=MAX_CHILDREN,"bounded writer declaration roster required")
    for child_path,child_sha in value["declared_children"].items():
        require(type(child_path) is str and _sha(child_sha)
            and _path(child_path).parent==Path(handle.directory)/"declarations",
            "original bounded writer child reference required")
    return sha,value


def _writer_close_header(permission,directory,binding,worker):
    sha,value=_terminal_writer_close(permission,directory,binding,worker["worker_pid"])
    require(worker["binding_sha256"]==binding and worker["attempt_journal_writer_close_sha256"]==sha,
        "original normal-return writer close binding required")
    return sha,value


def read_job_writer_close(permission,directory,binding,worker):
    sha,value=_writer_close_header(permission,directory,binding,worker)
    handle,path=_permission_handle(permission);_,spec=_declaration(handle,path)
    for child_path,child_sha in value["declared_children"].items():
        child_path=_path(child_path)
        require(_sha(child_sha) and child_path.parent==Path(handle.directory)/"declarations"
            and _hash(child_path)==child_sha,"original writer child declaration bytes drift")
        child_row,child=_declaration(handle,child_path)
        require(child.parent_directory==spec.directory and child_row["parent_declaration_sha256"]==_hash(path),
            "writer child declaration lost original direct parent")
    return sha,value


def read_attempt_journal_cost(handle,*,expected_outer_terminal=None):
    """Census declarations, not successful directories; no historical PID query."""
    meta,outer=journal_metadata(handle);before=_inventory([handle.directory])
    expected_sha=None if expected_outer_terminal is None else digest(expected_outer_terminal)
    registry=Path(handle.directory)/"declarations"
    paths=sorted(registry.iterdir())
    require(len(paths)<=MAX_COST_ATTEMPTS and all(p.is_file() and p.suffix==".json" for p in paths),"bounded exact declaration registry required")
    specs={};declarations={};errors=[];opaque=[]
    for path in paths:
        try:
            row,spec=_declaration(handle,path)
            require(spec.directory not in specs,"duplicate declared directory")
            specs[spec.directory]=spec;declarations[spec.directory]=(path,row)
        except (ValueError,TypeError,KeyError,OSError) as error:
            if path.name==_name(outer.directory):errors.append("MalformedOuterDeclaration")
            else:opaque.append(dict(directory=None,declaration_path=str(path),state="MALFORMED_DECLARATION",evidence_error_type=type(error).__name__))
    if outer.directory not in specs:specs[outer.directory]=outer;errors.append("MissingOuterDeclaration")
    require(digest(asdict(specs[outer.directory]))==digest(asdict(outer)),"journal outer descriptor drift")
    for directory,spec in specs.items():
        if directory not in declarations:continue
        row=declarations[directory][1]
        if spec.parent_directory is None:
            require(directory==outer.directory and row["parent_declaration_sha256"] is None,"unrelated journal root declaration")
        else:
            if spec.parent_directory not in declarations or row["parent_declaration_sha256"]!=_hash(declarations[spec.parent_directory][0]):
                errors.append("OriginalDeclaredParentMissingOrChanged")
    ordered=tuple(specs.values());roster_sha=digest([asdict(s) for s in ordered])
    original_attempt_inventory=_inventory([s.directory for s in ordered])
    forest_valid=True
    for spec in ordered:
        current=spec;seen=set()
        while current.parent_directory is not None:
            if current.directory in seen or len(seen)>=MAX_DEPTH or current.parent_directory not in specs:
                forest_valid=False;break
            seen.add(current.directory);current=specs[current.parent_directory]
    if forest_valid:
        result=reconcile_attempt_cost_roster(ordered,attempt_root=meta["attempt_root"],roster_sha256=roster_sha)
    else:
        # Preserve known individual costs even when a lost declaration means
        # the original forest cannot be joined. Never fabricate a parent spec
        # or a nonoverlapping aggregate from this partial view.
        errors.append("UnjoinableDeclaredForest");rows=[];evidence_errors=[]
        for spec in ordered:
            try:row,_,_=_attempt(spec)
            except (ValueError,TypeError,KeyError,OSError) as error:
                row=dict(directory=spec.directory,parent_directory=spec.parent_directory,state="MALFORMED_EVIDENCE",
                    job_cpu_seconds_including_reaped_children=None,evidence_error_type=type(error).__name__)
                evidence_errors.append(dict(directory=spec.directory,error_type=type(error).__name__))
            rows.append(row)
        result=dict(schema="pokezero.search-over-raw.attempt-roster-cost.v1",roster_sha256=roster_sha,
            attempt_root=meta["attempt_root"],required_attempt_slots=len(ordered),attempts=rows,evidence_errors=evidence_errors,
            delegation_errors=[dict(error_type="UnjoinableDeclaredForest")],measured_cpu_frontier=[],
            known_nonoverlapping_job_cpu_seconds=None,complete_workflow_cost=False,resource_cost_complete=False,
            work_validated=False,scientific_collection_authorized=False,holdout_authorized=False,retries_permitted=False,
            supplied_roster_only=True,attempt_population_exhaustiveness_validated=False)
    writer_children={};close_errors=[];expected_closes={"outer-controller.json"};bound_child_references={}
    terminal_writer_controls=[]
    for spec in ordered:
        directory=Path(spec.directory);terminal_path=directory/"terminal.json"
        try:
            require(spec.directory in declarations or not directory.exists(),"claimed attempt lacks original declaration")
            if spec.directory not in declarations:continue
            permission=_permission(handle,declarations[spec.directory][0])
            if (directory/"claim.json").is_file():
                require(_json(directory/"claim.json").get("attempt_declaration")==permission,
                    "actual claim lost original declaration permission")
            if not terminal_path.is_file():continue
            terminal=_json(terminal_path)
            require(terminal["attempt_declaration_sha256"]==permission["declaration_sha256"],"terminal declaration binding drift")
            if terminal["worker_pid"] is None:continue  # A failed spawn has no worker writer.
            if terminal["supervisor_exit_code"]==124 and spec.parent_directory is not None:
                from .search_over_raw_nested_ownership import _read,_detached,_clean_diagnostic_incomplete,delegated_close_value
                parent=Path(spec.parent_directory);parent_launch=_json(parent/"launch.json")
                parent_spec=specs[spec.parent_directory]
                path=parent/"delegated-groups"/f"child-{terminal['worker_pid']}-{spec.binding_sha256}.json"
                row,original_child=_read(path,parent,parent_launch["pid"],parent_spec.binding_sha256,
                    _json(directory/"nested-detach.json")["depth"]-1)
                closed=_json(path.with_name("closed-"+path.name))
                if closed.get("clean_diagnostic_incomplete") is True:
                    require(original_child==directory and _detached(row,directory,path)
                        and _clean_diagnostic_incomplete(row,directory,terminal)
                        and digest(closed)==digest(delegated_close_value(path,row,directory,terminal,clean_incomplete=True))
                        and not any(child.parent_directory==spec.directory for child in ordered),
                        "bounded killed diagnostic writer cannot hide child declarations")
                    # Actual direct-parent wait ended this writer; the complete
                    # original journal has no child declarations from it. No
                    # normal close, successful work, or future census is claimed.
                    writer_children[spec.directory]=set()
                    terminal_writer_controls.append(spec.directory)
                    close_path=Path(handle.directory)/"writer-closes"/_name(spec.directory)
                    if close_path.exists():
                        sha,close=_terminal_writer_close(permission,directory,spec.binding_sha256,terminal["worker_pid"])
                        require(not close["declared_children"]
                            and terminal["attempt_journal_writer_close_sha256"] in (None,sha),
                            "abnormal terminal writer close cannot conceal declarations")
                        expected_closes.add(close_path.name)
                    continue
            worker=_json(directory/"worker-result.json")
            require(type(terminal["worker_exit_code"]) is int and terminal["worker_exit_code"]==worker["exit_code"]
                and type(worker["worker_pid"]) is int and worker["worker_pid"]==terminal["worker_pid"],"actual writer wait receipt required")
            sha,close=_writer_close_header(permission,directory,spec.binding_sha256,worker)
            require(terminal["attempt_journal_writer_close_sha256"]==sha,"terminal lost original writer close")
            for child_path,child_sha in close["declared_children"].items():
                require(child_path not in bound_child_references,"declaration attributed to two writers")
                bound_child_references[child_path]=(child_sha,spec.directory)
            sha,close=read_job_writer_close(permission,directory,spec.binding_sha256,worker)
            writer_children[spec.directory]=set(close["declared_children"])
            expected_closes.add(_name(spec.directory))
        except (ValueError,TypeError,KeyError,OSError) as error:
            close_errors.append(dict(directory=spec.directory,error_type=type(error).__name__))
    for directory,children in writer_children.items():
        require(children=={str(path) for child,(path,row) in declarations.items() if specs[child].parent_directory==directory},
            "journal child census differs from original worker close")
    controller=Path(handle.directory)/"writer-closes/outer-controller.json"
    controller_valid=False
    if controller.is_file():
        value=_json(controller)
        outer_launch_path=Path(outer.directory)/"launch.json"
        outer_launch=_json(outer_launch_path) if outer_launch_path.is_file() else None
        controller_valid=(type(value) is dict and set(value)=={"schema","metadata_sha256","owner_declaration_sha256",
            "writer_pid","declared_children","supervised_attempt_population_only","work_validated","complete_workflow_cost"}
            and value.get("schema")==SCHEMA+".writer-close" and value.get("metadata_sha256")==handle.metadata_sha256
            and value.get("owner_declaration_sha256") is None and type(value.get("writer_pid")) is int
            and value["writer_pid"]>0 and (outer_launch is None or value["writer_pid"]==outer_launch["supervisor_pid"])
            and value["supervised_attempt_population_only"] is True and value["work_validated"] is False
            and value["complete_workflow_cost"] is False and value.get("declared_children")=={
                str(declarations[outer.directory][0]):_hash(declarations[outer.directory][0])} if outer.directory in declarations else False)
    observed_paths={str(path) for path in paths}
    for child_path,(child_sha,parent) in bound_child_references.items():
        if child_path not in observed_paths:
            opaque.append(dict(directory=None,parent_directory=parent,declaration_path=child_path,
                original_declaration_sha256=child_sha,state="MISSING_DECLARATION",
                job_cpu_seconds_including_reaped_children=None,evidence_error_type="MissingOriginalDeclaration"))
    unknown_closes={p.name for p in (Path(handle.directory)/"writer-closes").iterdir()}-expected_closes
    if unknown_closes:errors.append("UnjoinedWriterClose")
    if expected_outer_terminal is not None:
        require(expected_sha==digest(_json(Path(outer.directory)/"terminal.json")),"original outer returned terminal drift")
    by_directory={r["directory"]:r for r in result["attempts"]}
    launched=[s for s in ordered if (Path(s.directory)/"launch.json").exists()]
    complete=(not errors and not opaque and not close_errors and controller_valid
        and not result["evidence_errors"] and not result["delegation_errors"]
        and by_directory[outer.directory].get("control_descendant_closure_validated") is True
        and all(s.directory in writer_children for s in launched))
    require(before==_inventory([handle.directory]) and original_attempt_inventory==_inventory([s.directory for s in ordered])
        and journal_metadata(handle)[0]==meta
        and (expected_outer_terminal is None or expected_sha==digest(expected_outer_terminal)),
        "original journal bytes or namespace changed during census")
    result["attempts"].extend(opaque);result["required_attempt_slots"]+=len(opaque)
    if opaque or errors:result["known_nonoverlapping_job_cpu_seconds"]=None
    result.update(journal_metadata_sha256=handle.metadata_sha256,journal_declaration_slots_observed=len(paths),
        declaration_errors=errors,writer_close_errors=close_errors,
        declared_supervised_attempt_census_complete=complete,
        all_journal_writers_normal_closed=complete and not terminal_writer_controls,
        cleanly_terminated_diagnostic_writer_directories=terminal_writer_controls,
        future_unconstructed_work_covered=False,
        ordinary_source_and_worker_actors_covered=False,attempt_population_exhaustiveness_validated=False,
        complete_workflow_cost=False,resource_cost_complete=False,work_validated=False)
    return result
