"""October 16 fixed-total-work exploration, never holdout or confirmation.

Reuses the source owner, raw job, actual native dispatch reader and isolated
continuation owner. It does not relax the legacy 64-work diagnostic contract or
admit fixed-work receipts into the old timed-selector ledger.
"""
from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import pickle
import time

from .search_over_raw import SearchConfiguration,digest,require
from .search_over_raw_ledger import _write_new
from .search_over_raw_root_priors import capture_actor_boundary,native_measure,RootInputUnavailable,seed_cluster_interval
from .search_over_raw_world_inputs import _state_payload,public_request_witness,compare_world_inputs

SCHEMA="pokezero.search-over-raw.fixed-total-exploration.v1"
COMPACT_SCHEMA="pokezero.search-over-raw.fixed-total-exploration.compact-v2"


def contract(namespace,seeds,excluded,authority,total,*,compact=False):
    from .search_over_raw_bounded_a2 import contract as bounded_contract
    c=bounded_contract(namespace,seeds,excluded,authority)
    for name in ("schema","configurations","attempt_wall_seconds","execution_ready"):
        c.pop(name)
    require(type(total) is int and total in {320,1024,4096},"reviewed total accepted iteration budget required")
    require(type(compact) is bool and (not compact or total==4096),
        "compact evidence is separately scoped to the fixed4096 exploration")
    result=dict(c,schema=COMPACT_SCHEMA if compact else SCHEMA,accepted_native_iterations_total=total,
        native_iterations_per_world=total//4,native_worlds=4,native_batch=16,
        leaves=["model","hp_fraction"] if total==320 else ["model"],
        wall_clock_search_budget=None,emergency_calendar_seconds=28800,
        source_actor_history_capture=False,consecutive_unavailable_cap=3,
        raw_control="one original archived public-request forward",
        continuation_alias_rule="eight cells per unique selected action, including all-raw aliases",
        complete_case_inference_authorized=False)
    if compact:
        from .search_over_raw_native_atomic_compact import SCHEMA as ENCODING
        result["native_atomic_encoding"]=ENCODING
    return result


def validate_contract(c):
    require(digest(c)==digest(contract(c["namespace"],c["panels"]["exploration"]["seeds"],
        c["excluded_seeds"],c["authority"],c["accepted_native_iterations_total"],
        compact=c.get("schema")==COMPACT_SCHEMA)),"fixed-work exploration contract drift")


def read_spec(path):
    import sys
    from .search_over_raw_exploration_entry import _decode
    from .search_over_raw_source_run import source_recipe
    from .search_over_raw_fixed_work_run import fixed_work_reader_source_hashes
    sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
    s=json.loads(Path(path).read_bytes())
    require(set(s)=={"schema","contract","source_config","checkpoint_contract","source_runtime",
        "exposure_inventory","input_hashes","output_directory","interpretation"}
        and s["schema"]==s["contract"]["schema"]+".entry","exact fixed-work entry schema required")
    validate_contract(s["contract"])
    config,checkpoint=_decode(s["source_config"]),_decode(s["checkpoint_contract"])
    source_recipe(config,checkpoint,s["source_runtime"],capture_actor_history=False)
    cli=Path(__file__).resolve().parents[3]/"scripts/collect_search_over_raw_fixed.py"
    authority=s["contract"]["authority"]
    require(type(s["exposure_inventory"]) is dict and s["exposure_inventory"],"bound prior exposure inventory required")
    expected={**s["source_runtime"]["files"],**fixed_work_reader_source_hashes(),**s["exposure_inventory"],
        authority["path"]:authority["sha256"],str(Path(sys.executable).resolve()):sha(Path(sys.executable).resolve()),str(cli):sha(cli)}
    require(digest(expected)==digest(s["input_hashes"]) and all(sha(p)==h for p,h in expected.items()),
        "fixed-work complete entry inventory drift")
    require(str(Path(s["output_directory"]).resolve())==s["output_directory"],"canonical fixed output path required")
    return s,config,checkpoint


def prepare(directory,parent_entry,authority_path,total,*,compact=False):
    import secrets
    import sys
    import uuid
    from .search_over_raw_exploration_entry import _decode
    from .search_over_raw_outcome_run import outcome_runtime_binding
    from .search_over_raw_fixed_work_run import fixed_work_reader_source_hashes
    sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
    parent_path=Path(parent_entry).resolve();parent=json.loads(parent_path.read_bytes())
    config,checkpoint=_decode(parent["source_config"]),_decode(parent["checkpoint_contract"])
    excluded=sorted(set(parent["contract"]["excluded_seeds"])|set(parent["contract"]["panels"]["exploration"]["seeds"]))
    seeds=[]
    while len(seeds)<32:
        seed=secrets.randbits(32)
        if seed not in excluded and seed not in seeds:seeds.append(seed)
    authority_path=Path(authority_path).resolve()
    c=contract(str(uuid.uuid4()),seeds,excluded,dict(path=str(authority_path),sha256=sha(authority_path)),total,
        compact=compact)
    exposure={**parent["exposure_inventory"],str(parent_path):sha(parent_path)}
    require(all(sha(p)==h for p,h in exposure.items()),"prior exposure evidence drift")
    extras=[p for p in parent["input_hashes"] if p.endswith((".so",".dylib"))]
    runtime=outcome_runtime_binding(config,checkpoint_path=checkpoint.checkpoint_path,
        extra_files=extras+[checkpoint.model_path,checkpoint.tables_path])
    cli=Path(__file__).resolve().parents[3]/"scripts/collect_search_over_raw_fixed.py"
    files={**runtime["files"],**fixed_work_reader_source_hashes(),**exposure,
        str(authority_path):sha(authority_path),str(Path(sys.executable).resolve()):sha(Path(sys.executable).resolve()),
        str(cli):sha(cli)}
    directory=Path(directory).resolve();directory.mkdir(parents=True,exist_ok=False,mode=0o700)
    s=dict(schema=c["schema"]+".entry",contract=c,source_config=_state_payload(config),
        checkpoint_contract=_state_payload(checkpoint),source_runtime=runtime,
        exposure_inventory=exposure,input_hashes=files,output_directory=str(directory/"run"),
        interpretation="Exploratory fixed TOTAL native work, not seconds or per-world nominal sims. Eight paired raw-policy continuations per unique selected action; missing/refused/unvalidated cells stay uncertain. No holdout, strength claim, or historical-roster reuse.")
    path=directory/"entry-spec.json";_write_new(path,s);read_spec(path)
    return dict(entry_spec=str(path),entry_sha256=sha(path),source_seeds=seeds,roots=200,
        accepted_total=total,iterations_per_world=total//4,worker_launched=False)


def continuation_cells(bank,result,reader,snapshot,binding,runtime,law):
    from .search_over_raw_outcomes import OutcomeReceiptValidator
    validate=OutcomeReceiptValidator(bank,expected_snapshot=snapshot,root_binding=binding,
        runtime_sha256=digest(runtime),validate_provenance_and_cleanup=reader)
    cells={}
    for outcome in result["outcomes"]["outcomes"]:
        witness=validate(dict(root_id=binding["root_id"],**outcome),law)
        known=witness is not None and outcome["status"]=="COMPLETE"
        key=(outcome["action"],outcome["replicate"])
        require(key not in cells,"duplicate continuation cell")
        cells[key]=dict(status=outcome["status"],validated=known,
            score=(outcome["signed_outcome"]+1)/2 if known else None)
    return cells


def measure_root(s,config,checkpoint,archive,source,root,boundary,directory,ownership,guard):
    from .paper_reference_runtime import PublicRootRequest
    from .search_over_raw_raw_run import RawSelectionJob,raw_runtime
    from .search_over_raw_archive import retained_archive_root_identity
    from .search_over_raw_outcome_run import OwnedContinuationJob,supervise_outcome_run,OwningRunValidator
    from .search_over_raw_screening import paired_root_interval
    directory.mkdir(parents=True,exist_ok=False,mode=0o700)
    context,pending,snapshot=archive.selected(root)
    frozen=retained_archive_root_identity(archive,root["source_request_index"])
    request=PublicRootRequest.capture(context.public_materialization_state,context.observation,pending_transition=pending)
    boundary.validate(request,checkpoint);witness=public_request_witness(request)
    cfg=SearchConfiguration("raw",seconds=1.)
    raw_job=RawSelectionJob(cfg,checkpoint,checkpoint.showdown_root,context,root["root_id"]+":raw",
        source["source_seed"],root["root_id"],raw_runtime(cfg,checkpoint.checkpoint_sha256,
        checkpoint.showdown_source_sha256),witness["public_request_sha256"],pending)
    raw=raw_job(ownership,lambda _name:None);_write_new(directory/"raw.json",raw)
    selections={};unavailable={};actions={"raw":raw["action"]}
    for leaf in s["contract"]["leaves"]:
        guard()
        try:
            selection=native_measure(context,boundary,checkpoint,source["source_seed"],pending=pending,
                accepted_iterations=s["contract"]["accepted_native_iterations_total"],leaf=leaf,
                evidence_path=directory/(leaf+"-native-inputs.json"),
                **({"compact_native_atomic_work":True} if s["contract"]["schema"]==COMPACT_SCHEMA else {}))
            selections[leaf]=selection;actions[leaf]=selection["action"]
            _write_new(directory/(leaf+"-selected.json"),selection)
        except RootInputUnavailable as error:
            unavailable[leaf]=str(error)
    if len(selections)==2:
        require(compare_world_inputs(selections["model"]["actual_native_evidence"],
            selections["hp_fraction"]["actual_native_evidence"])["entire_population_matched"],
            "model/HP sampled worlds differ")
    binding=dict(root_id=root["root_id"],source_seed=source["source_seed"],
        public_record_sha256=root["public_record_sha256"],
        information_key=witness["consumed_binding"]["information_key"],
        legal_action_mask=list(context.observation.legal_action_mask))
    cells={};outcome_terminal=None
    if selections:
        job=OwnedContinuationJob(deepcopy(config),deepcopy(snapshot),binding,deepcopy(s["source_runtime"]),
            "p1",actions,s["contract"]["namespace"])
        job_sha=hashlib.sha256(pickle.dumps(job)).hexdigest();inputs=digest(job.input_binding())
        attempt=digest(dict(entry=digest(s),root=root["root_id"],job_sha256=job_sha,inputs=inputs))
        def admit():
            guard()
            require(digest(job.input_binding())==inputs and hashlib.sha256(pickle.dumps(job)).hexdigest()==job_sha,
                "original continuation job changed")
            require(retained_archive_root_identity(archive,root["source_request_index"])==frozen,
                "original source root changed")
            return True
        _write_new(directory/"continuation-reservation.json",dict(actions=actions,replicates=8,
            binding_sha256=attempt,retry_authorized=False))
        outcome_terminal=supervise_outcome_run(job,directory=directory/"outcomes",binding_sha256=attempt,
            verify_admission=admit,validate_result=lambda _value:True)
        require(outcome_terminal["supervisor_exit_code"]==outcome_terminal["worker_exit_code"]==0,
            "continuation operational stop; no retry")
        reader=OwningRunValidator(directory=directory/"outcomes",job_sha256=job_sha,binding_sha256=attempt,
            root_binding=binding,expected_snapshot=snapshot,runtime=s["source_runtime"])
        result=outcome_terminal["result"]
        law=dict(root_binding=binding,root_id=root["root_id"],namespace=s["contract"]["namespace"],
            subject="p1",max_boundaries=250,replicates=8,selection_seed=source["source_seed"],
            policy="raw_argmax_after_initial_sampled_opponent_reply")
        cells=continuation_cells(result["bank"],result,reader,snapshot,binding,s["source_runtime"],law)
        admit()
    guard()
    require(retained_archive_root_identity(archive,root["source_request_index"])==frozen,"source changed during root measurement")
    row=dict(root_id=root["root_id"],source_seed=source["source_seed"],source_sha256=digest(source),
        source_pipeline_validated=True,public_request_sha256=witness["public_request_sha256"],
        actions=actions,input_unavailable=unavailable,accepted_total=s["contract"]["accepted_native_iterations_total"],
        intervals={leaf:list(paired_root_interval(actions[leaf],actions["raw"],cells)) if leaf in selections
            else [-1.,1.] for leaf in s["contract"]["leaves"]},
        cells=[dict(action=a,replicate=r,**value) for (a,r),value in sorted(cells.items())],
        native_priors_differ_from_raw={leaf:None if sel["actual_model_argmax"] is None else sel["actual_model_argmax"]!=raw["action"]
            for leaf,sel in selections.items()},scientific_strength_claim=False)
    _write_new(directory/"acceptance.json",dict(entry_sha256=digest(s),row_sha256=digest(row),
        root_binding=binding,public_request_witness=witness,native_statistics_root_id=context.battle_id,
        archive_sha256=frozen,raw_sha256=digest(raw),selected_sha256={leaf:digest(sel) for leaf,sel in selections.items()},
        continuation_job_sha256=job_sha if selections else None,
        continuation_binding_sha256=attempt if selections else None))
    return row


def read_completed_root(s,row,source):
    """Recheck the actual retained source/selection/continuation acceptance.

    Local hashes detect changed bytes, not operator authentication. The private
    continuation job is loaded only from this owned run's pinned job file.
    """
    from .search_over_raw_raw_run import read_raw_selection,raw_runtime
    from .search_over_raw_world_inputs import validate_fixed_work_inputs
    from .search_over_raw_outcome_run import OwnedContinuationJob,OwningRunValidator
    from .search_over_raw_screening import paired_root_interval
    require(row["source_seed"]==source["source_seed"] and row["source_sha256"]==digest(source),"completed source identity drift")
    if row["root_id"] in source["missing_root_ids"]:
        require(row["actions"]=={} and row["intervals"]=={leaf:[-1.,1.] for leaf in s["contract"]["leaves"]},
            "missing source cannot acquire an action or score")
        return row
    roots=[r for r in source["roots"] if r["root_id"]==row["root_id"]]
    require(len(roots)==1,"completed root not in actual source")
    root=roots[0];directory=Path(s["output_directory"])/"roots"/digest(row["root_id"])
    require(row["receipt_files"]==_pins(directory),"accepted root receipt bytes changed")
    accepted=json.loads((directory/"acceptance.json").read_bytes())
    core={k:v for k,v in row.items() if k not in {"receipt_files","source_receipt_files"}}
    require(accepted["entry_sha256"]==digest(s) and accepted["row_sha256"]==digest(core),"accepted completed-root row changed")
    binding=accepted["root_binding"];witness=accepted["public_request_witness"]
    require(binding["root_id"]==root["root_id"] and binding["source_seed"]==source["source_seed"]
        and binding["public_record_sha256"]==root["public_record_sha256"]
        and binding["legal_action_mask"]==root["public_record"]["current_legal_action_mask"]
        and binding["information_key"]==witness["consumed_binding"]["information_key"],"source/selection public root join drift")
    from .search_over_raw_exploration_entry import _decode
    checkpoint=_decode(s["checkpoint_contract"]);cfg=SearchConfiguration("raw",seconds=1.)
    raw=json.loads((directory/"raw.json").read_bytes())
    require(digest(raw)==accepted["raw_sha256"],"raw selection changed")
    read_raw_selection(raw,configuration=cfg,runtime=raw_runtime(cfg,checkpoint.checkpoint_sha256,checkpoint.showdown_source_sha256),
        root_id=root["root_id"]+":raw",statistics_root_id=root["root_id"],selection_seed=source["source_seed"],
        consumed_binding=witness["consumed_binding"],legal_action_mask=binding["legal_action_mask"],
        public_request_sha256=witness["public_request_sha256"])
    actions={"raw":raw["action"]};selections={}
    for leaf,sha in accepted["selected_sha256"].items():
        require(leaf in s["contract"]["leaves"],"unregistered leaf selection")
        sel=json.loads((directory/(leaf+"-selected.json")).read_bytes());selections[leaf]=sel
        require(digest(sel)==sha and sel["accepted_target"]==sel["fixed_iterations"]==s["contract"]["accepted_native_iterations_total"],
            "native selection work or bytes changed")
        validate_fixed_work_inputs(dict(total_iterations=sel["fixed_iterations"],consumed_world_inputs=sel["actual_native_evidence"]),
            arm="incumbent",binding=witness["consumed_binding"],statistics_root_id=accepted["native_statistics_root_id"],
            construction_required=True,expected_public_request_sha256=witness["public_request_sha256"],
            native_public_fold_required=True,iterations_per_world=s["contract"]["native_iterations_per_world"],
            compact_native_atomic_work=s["contract"]["schema"]==COMPACT_SCHEMA,
            compact_fixed_public_fold=s["contract"]["schema"]==COMPACT_SCHEMA)
        actions[leaf]=sel["action"]
    if len(selections)==2:
        require(compare_world_inputs(selections["model"]["actual_native_evidence"],selections["hp_fraction"]["actual_native_evidence"])
            ["entire_population_matched"],"accepted native worlds differ")
    require(actions==row["actions"],"completed actions differ from actual selected actions")
    cells={}
    if selections:
        job_path=directory/"outcomes/job.pickle"
        require(hashlib.sha256(job_path.read_bytes()).hexdigest()==accepted["continuation_job_sha256"],"continuation job changed")
        job=pickle.loads(job_path.read_bytes())
        require(type(job) is OwnedContinuationJob and job.root_binding==binding and job.actions==actions
            and job.namespace==s["contract"]["namespace"] and job.runtime==s["source_runtime"],"original continuation job/root drift")
        reader=OwningRunValidator(directory=directory/"outcomes",job_sha256=accepted["continuation_job_sha256"],
            binding_sha256=accepted["continuation_binding_sha256"],root_binding=binding,expected_snapshot=job.snapshot,runtime=job.runtime)
        result=json.loads((directory/"outcomes/terminal.json").read_bytes())["result"]
        law=dict(root_binding=binding,root_id=root["root_id"],namespace=job.namespace,subject="p1",max_boundaries=250,
            replicates=8,selection_seed=source["source_seed"],policy="raw_argmax_after_initial_sampled_opponent_reply")
        cells=continuation_cells(result["bank"],result,reader,job.snapshot,binding,job.runtime,law)
    require(row["cells"]==[dict(action=a,replicate=r,**value) for (a,r),value in sorted(cells.items())],"completed continuation cells changed")
    expected={leaf:list(paired_root_interval(actions[leaf],actions["raw"],cells)) if leaf in selections else [-1.,1.]
        for leaf in s["contract"]["leaves"]}
    require(row["intervals"]==expected,"completed intervals differ from independently read cells")
    return row


def _pins(directory):
    return {str(p.resolve()):hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(directory).rglob("*") if p.is_file()}


@dataclass(frozen=True)
class FixedExplorationJob:
    spec_path:str
    entry_sha256:str
    def __call__(self,ownership,phase):
        from .search_over_raw_bounded_a2 import collect_source
        phase("startup");s,config,checkpoint=read_spec(self.spec_path)
        out=Path(s["output_directory"]);began=time.time();consecutive=0
        def guard():
            require(time.time()-began<s["contract"]["emergency_calendar_seconds"],"fixed exploration calendar cap")
            require(hashlib.sha256(Path(self.spec_path).read_bytes()).hexdigest()==self.entry_sha256,"fixed entry changed")
            return True
        phase("selection")
        try:
            for seed in s["contract"]["panels"]["exploration"]["seeds"]:
                archive=None;captures={}
                def capture(env,archive,context,action,history):
                    captures[context.decision_round_index]=capture_actor_boundary(env,archive,context,history)
                try:
                    archive,source,_=collect_source(s,config,checkpoint,seed,out/"sources"/str(seed),guard,
                        root_capture=capture,capture_actor_history=False)
                    for root in source["roots"]:
                        guard()
                        row=measure_root(s,config,checkpoint,archive,source,root,
                            captures[root["source_request_index"]],out/"roots"/digest(root["root_id"]),ownership,guard)
                        row["receipt_files"]=_pins(out/"roots"/digest(root["root_id"]))
                        row["source_receipt_files"]=_pins(out/"sources"/str(seed))
                        _write_new(out/"completed-roots"/(digest(root["root_id"])+".json"),row)
                        print(json.dumps(dict(root_id=root["root_id"],actions=row["actions"],intervals=row["intervals"])),flush=True)
                        consecutive=consecutive+1 if row["input_unavailable"] else 0
                        if consecutive>=3:return dict(status="THREE_CONSECUTIVE_UNAVAILABLE")
                    for missing in source["missing_root_ids"]:
                        _write_new(out/"completed-roots"/(digest(missing)+".json"),dict(root_id=missing,
                            source_seed=seed,source_sha256=digest(source),source_receipt_files=_pins(out/"sources"/str(seed)),
                            actions={},input_unavailable={"source":"missing_fixed_slot"},
                            intervals={leaf:[-1.,1.] for leaf in s["contract"]["leaves"]}))
                        consecutive+=1
                        if consecutive>=3:return dict(status="THREE_CONSECUTIVE_UNAVAILABLE")
                finally:
                    if archive is not None:archive.close()
            return dict(status="FIXED_ROSTER_ATTEMPTED")
        finally:phase("cleanup")


def summarize(s,rows,terminal):
    from .search_over_raw_screening import cluster_intervals
    c=s["contract"];slots=c["panels"]["exploration"]["root_slots"];seeds=c["panels"]["exploration"]["seeds"]
    ids={r["root_id"]:r["source_seed"] for r in slots}
    require(len({r["root_id"] for r in rows})==len(rows) and all(ids.get(r["root_id"])==r["source_seed"] for r in rows),
        "duplicate or foreign fixed root")
    roster=tuple((seed,tuple(r["root_id"] for r in slots if r["source_seed"]==seed)) for seed in seeds)
    effects={}
    for leaf in c["leaves"]:
        intervals={r["root_id"]:r["intervals"][leaf] for r in rows}
        clustered=cluster_intervals(roster,intervals)
        known=[r for r in rows if r["intervals"][leaf][0]==r["intervals"][leaf][1]]
        values=[r["intervals"][leaf][0] for r in known]
        effects[leaf]=dict(fixed_seed_weighted_bounds=[math.fsum(v[j] for v in clustered)/32 for j in (0,1)],
            identified_roots=len(known),uncertain_fixed_slots=200-len(known),
            observed_complete_case_mean=math.fsum(values)/len(values) if values else None,
            descriptive_seed_clustered_95=seed_cluster_interval(values,[r["source_seed"] for r in known]) if known else None,
            action_changes=sum(leaf in r["actions"] and r["actions"][leaf]!=r["actions"]["raw"] for r in rows),
            native_prior_disagreements=sum(r.get("native_priors_differ_from_raw",{}).get(leaf) is True for r in rows),
            native_prior_unreported=sum(leaf in r["actions"] and r.get("native_priors_differ_from_raw",{}).get(leaf) is None for r in rows))
    actual_cells=[cell for r in rows for cell in r.get("cells",[])]
    cell_counts={status:sum(cell["status"]==status for cell in actual_cells)
        for status in ("COMPLETE","REFUSED","CAPPED","UNSTARTED")}
    cell_counts["VALIDATED_COMPLETE"]=sum(cell["validated"] is True for cell in actual_cells)
    cell_counts["UNVALIDATED_COMPLETE"]=sum(cell["status"]=="COMPLETE" and cell["validated"] is not True for cell in actual_cells)
    result=dict(schema=c["schema"]+".readout",planned_roots=200,planned_seeds=32,recorded_roots=len(rows),
        accepted_iterations_total=c["accepted_native_iterations_total"],iterations_per_world=c["native_iterations_per_world"],
        effects=effects,unique_observed_continuation_cells=cell_counts,
        supervisor_exit_code=terminal["supervisor_exit_code"],worker_exit_code=terminal["worker_exit_code"],
        terminal_status=terminal["status"],elapsed_seconds=terminal["elapsed_wall_seconds"],
        cpu_seconds=terminal["cpu_seconds_including_reaped_children"],
        complete_case_scope="descriptive observed roots; not full-panel inference",scientific_strength_claim=False,
        holdout_authorized=False,phase_b_authorized=False,retraining_authorized=False)
    if c["schema"]==COMPACT_SCHEMA:
        result["native_atomic_encoding"]=c["native_atomic_encoding"]
    return result


def run(path,review_path):
    from .search_over_raw_supervision import _supervise_owned_job
    from .search_over_raw_exploration_entry import ExplorationAuditorContract
    s,_,_=read_spec(path);sha=hashlib.sha256(Path(path).read_bytes()).hexdigest()
    review_bytes=Path(review_path).read_bytes();review=json.loads(review_bytes)
    require(review.get("status")=="CLEAR_FIXED_TOTAL_EXPLORATION_ENTRY" and review.get("reviewer")=="Astra"
        and review.get("entry_spec_sha256")==sha and review.get("reviewed_before_first_source_work") is True
        and review.get("holdout_authorized") is False,"prospective exact Astra fixed-work review required")
    out=Path(s["output_directory"]);out.mkdir(parents=True,exist_ok=False,mode=0o700)
    (out/"completed-roots").mkdir();_write_new(out/"entry.json",s);_write_new(out/"entry-review.json",review)
    def admit():
        read_spec(path)
        require(hashlib.sha256(Path(path).read_bytes()).hexdigest()==sha and Path(review_path).read_bytes()==review_bytes,
            "reviewed fixed entry changed")
        return True
    terminal=_supervise_owned_job(FixedExplorationJob(str(Path(path).resolve()),sha),directory=out/"auditor",
        binding_sha256=digest(dict(entry=sha,review=hashlib.sha256(review_bytes).hexdigest())),verify_admission=admit,
        contract=ExplorationAuditorContract(s["contract"]["emergency_calendar_seconds"]),
        kind="staged-exploration-run",work_phase="selection",validate_result=lambda v:require(v["status"] in {
            "FIXED_ROSTER_ATTEMPTED","THREE_CONSECUTIVE_UNAVAILABLE"},"fixed exploration terminal drift"))
    rows=[json.loads(p.read_bytes()) for p in sorted((out/"completed-roots").glob("*.json"))]
    from .search_over_raw_source_run import read_source_owner
    sources={}
    for row in rows:
        seed=row["source_seed"];directory=out/"sources"/str(seed)
        if seed not in sources:
            source=json.loads((directory/"source.json").read_bytes())
            require(read_source_owner(directory,source,s["source_runtime"],s["contract"])["source_pipeline_validated"],
                "completed source no longer validates")
            sources[seed]=(source,_pins(directory))
        source,pins=sources[seed]
        require(row["source_receipt_files"]==pins,"completed source receipt bytes changed")
        read_completed_root(s,row,source)
    admit();readout=summarize(s,rows,terminal);_write_new(out/"readout.json",readout)
    return readout
