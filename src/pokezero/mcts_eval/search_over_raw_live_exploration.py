"""Fresh fixed-work collection using production live-boundary selectors.

This is a separate exploratory contract. Historical archived-selector receipts
remain historical; partial correspondence can produce diagnostic uncertainty
bounds, never a passed panel or a strength claim.
"""
from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import pickle
import time

from ..root_prior_correspondence import SCHEMA as PRIOR_SCHEMA,input_binding,exact_collector_gate
from .search_over_raw import digest,require
from .search_over_raw_ledger import _write_new as _write_in_existing_parent
from .search_over_raw_world_inputs import _state_payload,public_request_witness
from .search_over_raw_live_capture import shadow_recipe,selector_config,build_selectors,LiveBoundaryCapture

SCHEMA="pokezero.search-over-raw.live-fixed-exploration.v1"


def _write_new(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    _write_in_existing_parent(path,value)


def contract(namespace,seeds,excluded,authority,handoff,checkpoint,total):
    from .search_over_raw_fixed_exploration import contract as fixed_contract
    c=fixed_contract(namespace,seeds,excluded,authority,total)
    c.update(schema=SCHEMA,handoff=deepcopy(handoff),live_shadow=shadow_recipe(checkpoint,
        total=total,leaves=c["leaves"]),raw_control="original live audited source decision",
        work_evidence="production accepted-iteration telemetry; not atomic backup validation",
        correspondence="exact collector-native versus independent live-native before panel interpretation")
    return c


def bindings(config,checkpoint,c):
    return {leaf:input_binding(config,checkpoint=checkpoint.checkpoint_path,model_ts=checkpoint.model_path,
        tables=checkpoint.tables_path,selectors={"sampled4":selector_config(checkpoint,
            total=c["accepted_native_iterations_total"],leaf=leaf)},world_rng="per-decision") for leaf in c["leaves"]}


def read_spec(path,*,verify=True):
    from .search_over_raw_exploration_entry import _decode
    from .search_over_raw_source_run import source_recipe
    s=json.loads(Path(path).read_bytes())
    require(set(s)=={"schema","contract","source_config","checkpoint_contract","source_runtime",
        "exposure_inventory","input_hashes","output_directory","prior_bindings"}
        and s["schema"]==SCHEMA+".entry","exact live exploration entry required")
    c=s["contract"];config,checkpoint=_decode(s["source_config"]),_decode(s["checkpoint_contract"])
    require(digest(c)==digest(contract(c["namespace"],c["panels"]["exploration"]["seeds"],c["excluded_seeds"],
        c["authority"],c["handoff"],checkpoint,c["accepted_native_iterations_total"])),"live contract drift")
    require(str(Path(s["output_directory"]).resolve())==s["output_directory"],"canonical live output required")
    require(s["exposure_inventory"] and s["prior_bindings"].keys()==set(c["leaves"]),"full exposure/binding inventory required")
    expected={**s["source_runtime"]["files"],**s["exposure_inventory"],
        **{c[k]["path"]:c[k]["sha256"] for k in ("authority","handoff")}}
    require(expected==s["input_hashes"],"complete live input inventory required")
    if verify:
        source_recipe(config,checkpoint,s["source_runtime"],live_shadow=c["live_shadow"])
        require(all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==h for p,h in expected.items()),
            "live source/runtime/exposure bytes changed")
        require(bindings(config,checkpoint,c)==s["prior_bindings"],"live correspondence bindings changed")
    return s,config,checkpoint


def prepare(directory,parent_entry,authority_path,handoff_path,total,*,extra_exposures=()):
    import secrets
    import uuid
    from .search_over_raw_exploration_entry import _decode
    from .search_over_raw_outcome_run import outcome_runtime_binding
    sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
    parent_path=Path(parent_entry).resolve();parent=json.loads(parent_path.read_bytes())
    config,checkpoint=_decode(parent["source_config"]),_decode(parent["checkpoint_contract"])
    exposure={**parent["exposure_inventory"],str(parent_path):sha(parent_path)}
    exposure.update({str(Path(p).resolve()):sha(p) for p in extra_exposures})
    excluded=set(parent["contract"]["excluded_seeds"])|set(parent["contract"]["panels"]["exploration"]["seeds"])
    def visit(v,key=""):
        if type(v) is dict:
            for k,x in v.items():visit(x,k)
        elif type(v) is list:
            if key in {"seeds","source_seeds","excluded_seeds","engineering_excluded_seeds"}:
                excluded.update(x for x in v if type(x) is int and 0<=x<2**32)
            for x in v:visit(x,key)
        elif key in {"seed","source_seed","fixture_seed"} and type(v) is int and 0<=v<2**32:excluded.add(v)
    for p,h in exposure.items():
        require(sha(p)==h,"historical exposure bytes changed")
        visit(json.loads(Path(p).read_bytes()))
    seeds=[]
    while len(seeds)<32:
        seed=secrets.randbits(32)
        if seed not in excluded and seed not in seeds:seeds.append(seed)
    authority={k:dict(path=str(Path(p).resolve()),sha256=sha(p))
        for k,p in (("authority",authority_path),("handoff",handoff_path))}
    c=contract(str(uuid.uuid4()),seeds,sorted(excluded),authority["authority"],authority["handoff"],checkpoint,total)
    extras=[p for p in parent["source_runtime"]["extra_files"] if p.endswith((".so",".dylib"))]
    runtime=outcome_runtime_binding(config,checkpoint_path=checkpoint.checkpoint_path,
        extra_files=extras+[checkpoint.model_path,checkpoint.tables_path])
    directory=Path(directory).resolve()
    s=dict(schema=SCHEMA+".entry",contract=c,source_config=_state_payload(config),checkpoint_contract=_state_payload(checkpoint),
        source_runtime=runtime,exposure_inventory=exposure,output_directory=str(directory/"run"),
        prior_bindings=bindings(config,checkpoint,c),input_hashes={**runtime["files"],**exposure,
            **{v["path"]:v["sha256"] for v in authority.values()}})
    directory.mkdir(parents=True,exist_ok=False,mode=0o700)
    path=directory/"entry-spec.json";_write_new(path,s);read_spec(path)
    return dict(entry_spec=str(path),entry_sha256=sha(path),source_seeds=seeds,roots=200,
        accepted_total=total,worker_launched=False,prospective_review_required=True)


def read_live_source(s,source):
    """Actual source lifecycle and persisted live rows, not just manifest syntax."""
    from .search_over_raw_source_run import read_source_owner
    out=Path(s["output_directory"]);seed=source["source_seed"]
    read_source_owner(out/"sources"/str(seed),source,s["source_runtime"],s["contract"],
        live_shadow=s["contract"]["live_shadow"])
    rows={leaf:{} for leaf in s["contract"]["leaves"]}
    for invocation in source["live_shadow"]["invocations"]:
        index=invocation["round"]
        for leaf,h in invocation["rows"].items():
            row=json.loads((out/"live-selections"/str(seed)/leaf/(str(index)+".json")).read_bytes())
            require(digest(row)==h and row["seed"]==seed and row["round"]==index,
                "persisted live selection differs from source manifest")
            rows[leaf][index]=row
    trace=json.loads((out/"sources"/str(seed)/"bridge-transport.json").read_bytes())["records"]
    choices=[r for r in trace if r["kind"]=="command" and r["type"]=="choices"]
    require([r["round"] for r in source["live_shadow"]["invocations"]]
        ==[i for i,r in enumerate(choices) if "p1" in r["players"]],"actual p1 invocation schedule differs")
    expected_paths={out/"live-selections"/str(seed)/leaf/(str(i)+".json") for leaf,rr in rows.items() for i in rr}
    require(set((out/"live-selections"/str(seed)).rglob("*.json"))==expected_paths,
        "foreign or incomplete live selection inventory")
    for root in source["roots"]:
        for leaf in rows:
            row=rows[leaf][root["source_request_index"]]
            require(row["raw_action"]==root["public_record"]["recorded_action_index"]
                and row["legal"]==[i for i,v in enumerate(root["public_record"]["current_legal_action_mask"]) if v],
                "original catalog/live action or legal mapping drift")
    return rows


def gate_reports(collector,live):
    """Unknown rows never become a passed panel or a manufactured prior."""
    whole=exact_collector_gate(collector,live)
    measured=[r for r in collector["rows"] if r["variants"]["sampled4"].get("status")=="MEASURED"]
    subset={**collector,"expected":[dict(seed=r["seed"],round=r["round"],variant="sampled4") for r in measured]}
    known=exact_collector_gate(subset,live)
    require(not measured or known["passed"],"measured collector/live correspondence failed; no contrast admission")
    return dict(status="PASSED" if whole["passed"] and not collector["missing_root_ids"] else "INCOMPLETE",
        whole_selected_roster=whole,measured_rows=known,
        unavailable_selected_roots=len(collector["rows"])-len(measured),missing_root_ids=collector["missing_root_ids"],
        scope="incomplete gates permit diagnostic uncertainty bounds only; not the prescribed passed-panel milestone")


def replay_source(s,source,rows,guard):
    """Invoke the independent production CLI, never reuse captured decisions."""
    import importlib.util
    import sys
    from .search_over_raw_exploration_entry import _decode
    checkpoint=_decode(s["checkpoint_contract"]);config=_decode(s["source_config"]);out=Path(s["output_directory"])
    script=Path(__file__).resolve().parents[3]/"scripts/live_root_prior_check.py"
    module_spec=importlib.util.spec_from_file_location("live_root_prior_check",script)
    module=importlib.util.module_from_spec(module_spec);sys.modules[module_spec.name]=module;module_spec.loader.exec_module(module)
    gates={};seed=source["source_seed"]
    for leaf in s["contract"]["leaves"]:
        selected=[deepcopy(rows[leaf][r["source_request_index"]]) for r in source["roots"]]
        report=dict(schema=PRIOR_SCHEMA,complete=True,bindings=s["prior_bindings"][leaf],rows=selected,
            expected=[dict(seed=seed,round=r["round"],variant="sampled4") for r in selected],
            missing_root_ids=source["missing_root_ids"])
        folder=out/"live-checks"/str(seed)/leaf;folder.mkdir(parents=True,exist_ok=False)
        _write_new(folder/"collector.json",report)
        code=module.main(["--checkpoint",checkpoint.checkpoint_path,"--model-ts",checkpoint.model_path,
            "--tables",checkpoint.tables_path,"--showdown-root",checkpoint.showdown_root,
            "--bridge-path",str(config.resolved_bridge_path()),"--seeds",str(seed),
            "--variants","sampled4","--sims",str(s["contract"]["native_iterations_per_world"]),
            "--leaf",leaf,"--strict-fallbacks","--collector-report",str(folder/"collector.json"),"--out",str(folder/"live.json")],
            boundary_guard=guard)
        live=json.loads((folder/"live.json").read_bytes())
        gate=gate_reports(report,live)
        require(code==(0 if live["collector_gate"]["passed"] else 1),"independent checker exit/verdict drift")
        _write_new(folder/"gate.json",gate);gates[leaf]=gate
    return gates


def read_gates(s,source,rows):
    gates={};out=Path(s["output_directory"]);seed=source["source_seed"]
    for leaf in s["contract"]["leaves"]:
        folder=out/"live-checks"/str(seed)/leaf
        collector=json.loads((folder/"collector.json").read_bytes())
        selected=[rows[leaf][r["source_request_index"]] for r in source["roots"]]
        require(collector==dict(schema=PRIOR_SCHEMA,complete=True,bindings=s["prior_bindings"][leaf],rows=selected,
            expected=[dict(seed=seed,round=r["round"],variant="sampled4") for r in selected],
            missing_root_ids=source["missing_root_ids"]),"collector gate roster differs from original source")
        gates[leaf]=gate_reports(collector,json.loads((folder/"live.json").read_bytes()))
        require(gates[leaf]==json.loads((folder/"gate.json").read_bytes()),"saved exact gate changed")
    return gates


def measure_root(s,config,archive,source,root,rows,guard):
    from .paper_reference_runtime import PublicRootRequest
    from .search_over_raw_archive import retained_archive_root_identity
    from .search_over_raw_fixed_exploration import continuation_cells
    from .search_over_raw_outcome_run import OwnedContinuationJob,supervise_outcome_run,OwningRunValidator
    from .search_over_raw_screening import paired_root_interval
    out=Path(s["output_directory"]);directory=out/"roots"/digest(root["root_id"])
    directory.mkdir(parents=True,exist_ok=False,mode=0o700)
    context,pending,snapshot=archive.selected(root)
    frozen=retained_archive_root_identity(archive,root["source_request_index"])
    witness=public_request_witness(PublicRootRequest.capture(context.public_materialization_state,
        context.observation,pending_transition=pending))
    index=root["source_request_index"];actions={"raw":rows[s["contract"]["leaves"][0]][index]["raw_action"]};unavailable={}
    for leaf in s["contract"]["leaves"]:
        item=rows[leaf][index]["variants"]["sampled4"]
        if item.get("status")=="MEASURED":actions[leaf]=item["action"]
        else:unavailable[leaf]=item.get("error","live selection unavailable")
    binding=dict(root_id=root["root_id"],source_seed=source["source_seed"],public_record_sha256=root["public_record_sha256"],
        information_key=witness["consumed_binding"]["information_key"],legal_action_mask=list(context.observation.legal_action_mask))
    cells={};acceptance=dict(entry_sha256=digest(s),root_binding=binding,archive_sha256=frozen,
        live_rows_sha256={leaf:digest(rows[leaf][index]) for leaf in rows},public_request_witness=witness)
    if len(actions)>1:
        job=OwnedContinuationJob(deepcopy(config),deepcopy(snapshot),binding,deepcopy(s["source_runtime"]),
            "p1",actions,s["contract"]["namespace"])
        job_sha=hashlib.sha256(pickle.dumps(job)).hexdigest();inputs=digest(job.input_binding())
        attempt=digest(dict(entry=digest(s),root=root["root_id"],job_sha256=job_sha,inputs=inputs))
        def admit():
            guard()
            require(hashlib.sha256(pickle.dumps(job)).hexdigest()==job_sha and digest(job.input_binding())==inputs
                and retained_archive_root_identity(archive,index)==frozen,"original live continuation inputs changed")
            return True
        _write_new(directory/"continuation-reservation.json",dict(actions=actions,replicates=8,binding_sha256=attempt,retry_authorized=False))
        terminal=supervise_outcome_run(job,directory=directory/"outcomes",binding_sha256=attempt,
            verify_admission=admit,validate_result=lambda v:True)
        require(terminal["supervisor_exit_code"]==terminal["worker_exit_code"]==0,"continuation operational stop; no retry")
        reader=OwningRunValidator(directory=directory/"outcomes",job_sha256=job_sha,binding_sha256=attempt,
            root_binding=binding,expected_snapshot=snapshot,runtime=s["source_runtime"])
        result=terminal["result"];law=dict(root_binding=binding,root_id=root["root_id"],namespace=s["contract"]["namespace"],
            subject="p1",max_boundaries=250,replicates=8,selection_seed=source["source_seed"],
            policy="raw_argmax_after_initial_sampled_opponent_reply")
        cells=continuation_cells(result["bank"],result,reader,snapshot,binding,s["source_runtime"],law);admit()
        acceptance.update(continuation_job_sha256=job_sha,continuation_binding_sha256=attempt)
    guard()
    row=dict(root_id=root["root_id"],source_seed=source["source_seed"],source_sha256=digest(source),actions=actions,
        input_unavailable=unavailable,accepted_total=s["contract"]["accepted_native_iterations_total"],
        intervals={leaf:list(paired_root_interval(actions[leaf],actions["raw"],cells)) if leaf in actions else [-1.,1.]
            for leaf in s["contract"]["leaves"]},cells=[dict(action=a,replicate=r,**v) for (a,r),v in sorted(cells.items())],
        scientific_strength_claim=False)
    acceptance["row_sha256"]=digest(row);_write_new(directory/"acceptance.json",acceptance)
    return row


def read_completed_root(s,row,source,rows):
    from .search_over_raw_outcome_run import OwnedContinuationJob,OwningRunValidator
    from .search_over_raw_fixed_exploration import continuation_cells,_pins
    from .search_over_raw_screening import paired_root_interval
    require(row["source_seed"]==source["source_seed"] and row["source_sha256"]==digest(source),"completed live source drift")
    if row["root_id"] in source["missing_root_ids"]:
        require(row["actions"]=={} and row["cells"]==[]
            and row["intervals"]=={leaf:[-1.,1.] for leaf in s["contract"]["leaves"]},"missing slot must remain uncertain")
        return
    root=next(r for r in source["roots"] if r["root_id"]==row["root_id"])
    directory=Path(s["output_directory"])/"roots"/digest(row["root_id"])
    require(row["receipt_files"]==_pins(directory),"completed live root receipt bytes changed")
    acceptance=json.loads((directory/"acceptance.json").read_bytes())
    core={k:v for k,v in row.items() if k not in {"receipt_files","source_receipt_files"}}
    require(acceptance["entry_sha256"]==digest(s) and acceptance["row_sha256"]==digest(core),"completed live row changed")
    index=root["source_request_index"];binding=acceptance["root_binding"]
    require(binding["root_id"]==root["root_id"] and binding["source_seed"]==source["source_seed"]
        and binding["public_record_sha256"]==root["public_record_sha256"]
        and binding["legal_action_mask"]==root["public_record"]["current_legal_action_mask"]
        and binding["information_key"]==acceptance["public_request_witness"]["consumed_binding"]["information_key"]
        and acceptance["live_rows_sha256"]=={leaf:digest(rr[index]) for leaf,rr in rows.items()},"original live root join drift")
    actions={"raw":rows[s["contract"]["leaves"][0]][index]["raw_action"]};unavailable={}
    for leaf in s["contract"]["leaves"]:
        item=rows[leaf][index]["variants"]["sampled4"]
        if item.get("status")=="MEASURED":
            require(item["iterations"]==s["contract"]["accepted_native_iterations_total"] and item["worlds_searched"]==4,
                "fixed live work telemetry changed")
            actions[leaf]=item["action"]
        else:unavailable[leaf]=item.get("error","live selection unavailable")
    require(actions==row["actions"] and unavailable==row["input_unavailable"],"live actions or refusals changed")
    cells={}
    if len(actions)>1:
        payload=(directory/"outcomes/job.pickle").read_bytes()
        require(hashlib.sha256(payload).hexdigest()==acceptance["continuation_job_sha256"],"original continuation job changed")
        job=pickle.loads(payload)
        require(type(job) is OwnedContinuationJob and job.root_binding==binding and job.actions==actions
            and job.subject=="p1" and job.namespace==s["contract"]["namespace"] and job.runtime==s["source_runtime"],
            "original live continuation job inputs differ")
        reader=OwningRunValidator(directory=directory/"outcomes",job_sha256=acceptance["continuation_job_sha256"],
            binding_sha256=acceptance["continuation_binding_sha256"],root_binding=binding,expected_snapshot=job.snapshot,runtime=job.runtime)
        result=json.loads((directory/"outcomes/terminal.json").read_bytes())["result"]
        law=dict(root_binding=binding,root_id=root["root_id"],namespace=job.namespace,subject="p1",max_boundaries=250,
            replicates=8,selection_seed=source["source_seed"],policy="raw_argmax_after_initial_sampled_opponent_reply")
        cells=continuation_cells(result["bank"],result,reader,job.snapshot,binding,job.runtime,law)
    require(row["cells"]==[dict(action=a,replicate=r,**v) for (a,r),v in sorted(cells.items())]
        and row["intervals"]=={leaf:list(paired_root_interval(actions[leaf],actions["raw"],cells)) if leaf in actions else [-1.,1.]
            for leaf in s["contract"]["leaves"]},"live continuation intervals differ from independently read cells")


@dataclass(frozen=True)
class LiveExplorationJob:
    spec_path:str
    entry_sha256:str
    deadline_utc_seconds:float
    def __call__(self,ownership,phase):
        from .search_over_raw_bounded_a2 import collect_source
        phase("startup");s,config,checkpoint=read_spec(self.spec_path);out=Path(s["output_directory"])
        consecutive=0
        def guard():
            require(time.time()<self.deadline_utc_seconds,"live exploration calendar cap")
            require(hashlib.sha256(Path(self.spec_path).read_bytes()).hexdigest()==self.entry_sha256,"live entry changed")
            return True
        phase("selection")
        try:
            for seed in s["contract"]["panels"]["exploration"]["seeds"]:
                guard();read_spec(self.spec_path);archive=None
                def factory(env):
                    policies=build_selectors(env,checkpoint,total=s["contract"]["accepted_native_iterations_total"],leaves=s["contract"]["leaves"])
                    def publish(leaf,row):_write_new(out/"live-selections"/str(seed)/leaf/(str(row["round"])+".json"),row)
                    return LiveBoundaryCapture(env,policies,source_seed=seed,total=s["contract"]["accepted_native_iterations_total"],
                        guard=guard,publish=publish)
                try:
                    archive,source,_=collect_source(s,config,checkpoint,seed,out/"sources"/str(seed),guard,
                        capture_actor_history=False,live_capture_factory=factory)
                    rows=read_live_source(s,source)
                    if source["roots"]:replay_source(s,source,rows,guard)
                    read_spec(self.spec_path)
                    for root in source["roots"]:
                        guard();row=measure_root(s,config,archive,source,root,rows,guard)
                        from .search_over_raw_fixed_exploration import _pins
                        row.update(receipt_files=_pins(out/"roots"/digest(root["root_id"])),
                            source_receipt_files=_pins(out/"sources"/str(seed)))
                        _write_new(out/"completed-roots"/(digest(root["root_id"])+".json"),row)
                        print(json.dumps(dict(root_id=root["root_id"],status="LIVE_ROOT_RECORDED")),flush=True)
                        consecutive=consecutive+1 if row["input_unavailable"] else 0
                        if consecutive>=3:return dict(status="THREE_CONSECUTIVE_UNAVAILABLE")
                    for missing in source["missing_root_ids"]:
                        _write_new(out/"completed-roots"/(digest(missing)+".json"),dict(root_id=missing,source_seed=seed,
                            source_sha256=digest(source),actions={},input_unavailable={"source":"missing_fixed_slot"},
                            intervals={leaf:[-1.,1.] for leaf in s["contract"]["leaves"]},cells=[],scientific_strength_claim=False))
                        consecutive+=1
                        if consecutive>=3:return dict(status="THREE_CONSECUTIVE_UNAVAILABLE")
                finally:
                    if archive is not None:archive.close()
            return dict(status="LIVE_FIXED_ROSTER_ATTEMPTED")
        finally:phase("cleanup")


def run(path,review_path):
    from .search_over_raw_supervision import _supervise_owned_job
    from .search_over_raw_exploration_entry import ExplorationAuditorContract
    from .search_over_raw_fixed_exploration import summarize,_pins
    s,_,_=read_spec(path);sha=hashlib.sha256(Path(path).read_bytes()).hexdigest()
    review_bytes=Path(review_path).read_bytes();review=json.loads(review_bytes)
    require(review.get("status")=="CLEAR_LIVE_FIXED_EXPLORATION_ENTRY" and review.get("reviewer")=="Astra"
        and review.get("entry_spec_sha256")==sha and review.get("reviewed_before_first_source_work") is True
        and review.get("holdout_authorized") is False,"prospective exact live entry review required")
    out=Path(s["output_directory"]);out.mkdir(parents=True,exist_ok=False,mode=0o700)
    (out/"completed-roots").mkdir();_write_new(out/"entry.json",s);_write_new(out/"entry-review.json",review)
    def admit():
        read_spec(path)
        require(hashlib.sha256(Path(path).read_bytes()).hexdigest()==sha and Path(review_path).read_bytes()==review_bytes,
            "reviewed live entry changed")
        return True
    start=time.time();deadline=start+s["contract"]["emergency_calendar_seconds"]
    _write_new(out/"launch-clock.json",dict(started_utc_seconds=start,deadline_utc_seconds=deadline,
        scope="calendar emergency cap; never an individual search budget"))
    terminal=_supervise_owned_job(LiveExplorationJob(str(Path(path).resolve()),sha,deadline),directory=out/"auditor",
        binding_sha256=digest(dict(entry=sha,review=hashlib.sha256(review_bytes).hexdigest())),verify_admission=admit,
        contract=ExplorationAuditorContract(s["contract"]["emergency_calendar_seconds"]),kind="staged-exploration-run",
        calendar_deadline_utc_seconds=deadline,
        work_phase="selection",validate_result=lambda v:require(v["status"] in {
            "LIVE_FIXED_ROSTER_ATTEMPTED","THREE_CONSECUTIVE_UNAVAILABLE"},"live exploration terminal drift"))
    records=[json.loads(p.read_bytes()) for p in sorted((out/"completed-roots").glob("*.json"))]
    sources={};gates={}
    for row in records:
        seed=row["source_seed"]
        if seed not in sources:
            source=json.loads((out/"sources"/str(seed)/"source.json").read_bytes())
            rows=read_live_source(s,source);sources[seed]=(source,rows)
            if source["roots"]:gates[str(seed)]=read_gates(s,source,rows)
        source,rows=sources[seed]
        if row["root_id"] not in source["missing_root_ids"]:
            require(row["source_receipt_files"]==_pins(out/"sources"/str(seed)),"completed source receipt bytes changed")
        read_completed_root(s,row,source,rows)
    admit();readout=summarize(s,records,terminal)
    passed=(terminal["supervisor_exit_code"]==terminal["worker_exit_code"]==0 and len(records)==200
        and len(sources)==32 and all(not source["missing_root_ids"] for source,_ in sources.values())
        and len(gates)==32 and all(g[leaf]["status"]=="PASSED" for g in gates.values() for leaf in s["contract"]["leaves"]))
    readout.update(panel_correspondence_gate="PASSED" if passed else "INCOMPLETE",source_gates=gates,
        interpretation="exploratory diagnostic uncertainty bounds; no gain claim" if not passed else "exploratory fixed-work panel; not holdout confirmation",
        work_evidence=s["contract"]["work_evidence"],scientific_strength_claim=False)
    _write_new(out/"readout.json",readout);return readout
