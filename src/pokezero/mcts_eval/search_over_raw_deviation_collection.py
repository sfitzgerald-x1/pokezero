"""Reviewed 4096 exploration: scan all boundaries, continue deviations only.

Sources use the existing raw-policy/bridge owner and sealed archive unchanged.
Capacity slots cover every possible non-opening boundary; they are NOT missing
statistical observations. The statistical denominator is the durable scan
reservation journal, including failed selections and failed source validation.
"""
from copy import deepcopy
from dataclasses import dataclass
import hashlib
import importlib.util
import json
from pathlib import Path
import secrets
import random
import sys
import time
import uuid

from .search_over_raw import digest, require
from .search_over_raw_deviations import ScannedRoot, LEAVES, stop_reason, summarize
from .search_over_raw_live_capture import LiveBoundaryCapture, shadow_recipe
from .search_over_raw_live_exploration import (
    _write_new, bindings, gate_reports, read_live_source, measure_root, read_completed_root,
)

SCHEMA = "pokezero.search-over-raw.deviation-exploration.v1"
SCAN_POLICY = "chronological_deviation_scan_v1"
SOURCE_ROSTER_CAPACITY = 2000


def contract(namespace, seeds, excluded, authority, handoff, checkpoint):
    require(str(uuid.UUID(namespace)) == namespace, "canonical fresh namespace required")
    require(type(seeds) is list and len(seeds) == SOURCE_ROSTER_CAPACITY
        and len(set(seeds)) == len(seeds) and all(type(s) is int and 0 <= s < 2**32 for s in seeds),
        "fixed ordered fresh source capacity roster required")
    require(type(excluded) is list and excluded == sorted(set(excluded))
        and excluded and all(type(s) is int and 0 <= s < 2**32 for s in excluded)
        and not set(seeds) & set(excluded), "exposure-disjoint sources required")
    return dict(schema=SCHEMA, namespace=namespace, panels={"exploration":dict(seeds=seeds)},
        excluded_seeds=excluded, authority=authority, handoff=handoff, leaves=list(LEAVES),
        accepted_native_iterations_total=4096, native_iterations_per_world=1024,
        work_evidence="production accepted-iteration telemetry; not atomic backup validation",
        native_worlds=4, native_batch=16, candidate_seat="p1", exclude_opening_requests=True,
        source_policy="raw_argmax_both_seats", source_order="frozen seed order; chronological boundaries",
        live_shadow=shadow_recipe(checkpoint,total=4096,leaves=list(LEAVES),scan_policy=SCAN_POLICY),
        deviation_triggered_continuations=True, continuation_replicates=8,
        continuation_policy="raw_argmax_after_initial_sampled_opponent_reply",
        continuation_order="source order then chronological request; unique-action paired bank",
        stopping="100 provisional action deviations per leaf OR 2000 reserved roots OR 28800 seconds",
        provisional_counts_require_independent_gate=True, consecutive_unavailable_cap=3,
        emergency_calendar_seconds=28800, max_source_boundaries=250,
        root_slot_semantics="249 per-source archive capacity only; not an inferential denominator",
        cutoff_cleanup="finish active source raw-only within remaining hard calendar budget",
        primary_weighting="pooled_roots", secondary_weighting="equal_seed",
        source_cluster_minimum=32, minimum_screen_deviations=20, lower_bound_must_be_strictly_positive=True,
        intervals="joint seed bootstrap descriptive; adaptive whole-procedure calibration pending",
        collection_authorized=True, retries_permitted=False, complete_case_inference_authorized=False,
        holdout_authorized=False, phase_b_authorized=False, retraining_authorized=False)


def source_spec(s, seed):
    """Deterministic adapter to the unchanged owner, not a sampled root quota."""
    require(seed in s["contract"]["panels"]["exploration"]["seeds"], "unregistered source")
    c = deepcopy(s["contract"])
    c["panels"]["exploration"]["root_slots"] = [dict(root_id=f"exploration:{seed}:{slot}",
        source_seed=seed,root_slot=slot) for slot in range(249)]
    return {**s, "contract": c}


def read_spec(path):
    from .search_over_raw_exploration_entry import _decode
    from .search_over_raw_source_run import source_recipe
    s = json.loads(Path(path).read_bytes())
    require(set(s)=={"schema","contract","source_config","checkpoint_contract","source_runtime",
        "exposure_inventory","input_hashes","output_directory","prior_bindings"}
        and s["schema"] == SCHEMA+".entry", "exact deviation entry required")
    config,checkpoint=(_decode(s[k]) for k in ("source_config","checkpoint_contract"))
    c=s["contract"]
    require(c == contract(c["namespace"],c["panels"]["exploration"]["seeds"],c["excluded_seeds"],
        c["authority"],c["handoff"],checkpoint), "deviation contract changed")
    source_recipe(config,checkpoint,s["source_runtime"],live_shadow=c["live_shadow"])
    expected={**s["source_runtime"]["files"],**s["exposure_inventory"],
        **{c[k]["path"]:c[k]["sha256"] for k in ("authority","handoff")}}
    require(s["exposure_inventory"] and expected==s["input_hashes"]
        and all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==h for p,h in expected.items()),
        "source/runtime/authority/exposure drift")
    require(str(Path(s["output_directory"]).resolve())==s["output_directory"]
        and bindings(config,checkpoint,c)==s["prior_bindings"], "output/prior bindings changed")
    return s,config,checkpoint


def prepare(directory,parent_entry,authority,handoff,*,extra_exposures=()):
    from .search_over_raw_exploration_entry import _decode
    from .search_over_raw_outcome_run import outcome_runtime_binding
    from .search_over_raw_world_inputs import _state_payload
    sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
    parent_path=Path(parent_entry).resolve();parent=json.loads(parent_path.read_bytes())
    config,checkpoint=(_decode(parent[k]) for k in ("source_config","checkpoint_contract"))
    exposure={**parent["exposure_inventory"],str(parent_path):sha(parent_path)}
    exposure.update({str(Path(p).resolve()):sha(p) for p in extra_exposures})
    excluded=set(parent["contract"]["excluded_seeds"]) | set(parent["contract"]["panels"]["exploration"]["seeds"])
    def visit(value,key=""):
        if type(value) is dict:
            for k,v in value.items():visit(v,k)
        elif type(value) is list:
            if key in {"seeds","source_seeds","excluded_seeds","engineering_excluded_seeds"}:
                excluded.update(v for v in value if type(v) is int and 0<=v<2**32)
            for v in value:visit(v,key)
        elif key in {"seed","source_seed","fixture_seed"} and type(value) is int and 0<=value<2**32:
            excluded.add(value)
    for p,h in exposure.items():
        require(sha(p)==h,"historical evidence changed");visit(json.loads(Path(p).read_bytes()))
    seeds=[]
    while len(seeds)<SOURCE_ROSTER_CAPACITY:
        seed=secrets.randbits(32)
        if seed not in excluded and seed not in seeds:seeds.append(seed)
    authority,handoff=(dict(path=str(Path(p).resolve()),sha256=sha(p)) for p in (authority,handoff))
    c=contract(str(uuid.uuid4()),seeds,sorted(excluded),authority,handoff,checkpoint)
    # Preserve the parent's external native/dependency pins, not just .so files.
    extras=list(parent["source_runtime"]["extra_files"])
    cli=Path(__file__).resolve().parents[3]/"scripts/collect_search_over_raw_deviations.py"
    extras.extend([checkpoint.model_path,checkpoint.tables_path,str(cli),str(Path(sys.executable).resolve())])
    runtime=outcome_runtime_binding(config,checkpoint_path=checkpoint.checkpoint_path,extra_files=extras)
    directory=Path(directory).resolve()
    s=dict(schema=SCHEMA+".entry",contract=c,source_config=_state_payload(config),
        checkpoint_contract=_state_payload(checkpoint),source_runtime=runtime,exposure_inventory=exposure,
        output_directory=str(directory/"run"),prior_bindings=bindings(config,checkpoint,c),
        input_hashes={**runtime["files"],**exposure,authority["path"]:authority["sha256"],handoff["path"]:handoff["sha256"]})
    directory.mkdir(parents=True,exist_ok=False,mode=0o700)
    path=directory/"entry-spec.json";_write_new(path,s);read_spec(path)
    return dict(entry_spec=str(path),entry_sha256=sha(path),source_capacity=len(seeds),max_scanned_roots=2000,
        worker_launched=False,prospective_review_required=True)


class DeviationCapture(LiveBoundaryCapture):
    """Reserve before selection; a cutoff is raw-only cleanup, not more scan."""
    def __init__(self,*args,state,output,started,**kwargs):
        super().__init__(*args,**kwargs)
        self.state,self.output,self.started=state,Path(output),started

    def __call__(self,context,decision):
        from ..root_prior_correspondence import boundary_fingerprint
        index=context.decision_round_index
        self._guard()
        if self.state["stop"] is not None:
            require(context.seed==self.seed and context.player_id=="p1"
                and all(index not in rows for rows in self.rows.values()), "raw cleanup identity/retry")
            legal=[i for i,v in enumerate(context.observation.legal_action_mask) if v]
            for leaf in self.selectors:
                row=dict(seed=self.seed,round=index,boundary_sha256=boundary_fingerprint(context),legal=legal,
                    raw_action=decision.action_index,raw_prior=list(decision.metadata["policy_distribution"]),
                    variants={"sampled4":dict(status="NOT_SCANNED",reason=self.state["stop"])})
                self.rows[leaf][index]=row;self.publish(leaf,deepcopy(row))
            return
        if index==0:
            return super().__call__(context,decision)
        reason=stop_reason(self.state["roots"],time.time()-self.started)
        require(reason is None,"scan continued after stopping boundary")
        ordinal=len(self.state["roots"])
        _write_new(self.output/"scan-reservations"/(f"{ordinal:04d}.json"),dict(ordinal=ordinal,
            seed=self.seed,request_index=index,raw=decision.action_index,
            boundary_sha256=boundary_fingerprint(context),both_leaves_required=True,retry_authorized=False))
        super().__call__(context,decision)
        actions={leaf:self.rows[leaf][index]["variants"]["sampled4"].get("action")
            if self.rows[leaf][index]["variants"]["sampled4"]["status"]=="MEASURED" else None for leaf in LEAVES}
        root=ScannedRoot(self.seed,index,decision.action_index,**actions)
        self.state["roots"].append(root)
        unavailable=any(getattr(root,leaf) is None for leaf in LEAVES)
        self.state["unavailable"]=self.state["unavailable"]+1 if unavailable else 0
        reason="THREE_CONSECUTIVE_UNAVAILABLE" if self.state["unavailable"]>=3 else stop_reason(
            self.state["roots"],time.time()-self.started)
        if reason:
            self.state["stop"]=reason
            _write_new(self.output/"scan-stop.json",dict(reason=reason,scanned_roots=len(self.state["roots"]),
                last_seed=self.seed,last_request_index=index,counts_provisional_until_gate=True))


def _selected_source(source,reservations):
    indices={r["request_index"] for r in reservations}
    roots=sorted((r for r in source["roots"] if r["source_request_index"] in indices),key=lambda r:r["source_request_index"])
    require({r["source_request_index"] for r in roots}==indices,"closed source did not retain every scanned boundary")
    return {**source,"roots":roots,"missing_root_ids":[]}


def _check_prefix(rows,reservations):
    indices=[r["request_index"] for r in reservations]
    require(indices==sorted(set(indices)),"chronological unique reserved prefix required")
    for leaf in LEAVES:
        scanned=sorted(i for i,row in rows[leaf].items() if i>0
            and row["variants"]["sampled4"]["status"]!="NOT_SCANNED")
        require(scanned==indices,"scan journal omits or adds an attempted eligible boundary")
        require(not indices or not any(i<=indices[-1] and i>0
            and row["variants"]["sampled4"]["status"]=="NOT_SCANNED" for i,row in rows[leaf].items()),
            "scan resumed after a skipped boundary")


def replay(s,source,rows,reservations,checkpoint,config,guard):
    from ..root_prior_correspondence import SCHEMA as PRIOR_SCHEMA
    _check_prefix(rows,reservations)
    selected=_selected_source(source,reservations);seed=source["source_seed"]
    script=Path(__file__).resolve().parents[3]/"scripts/live_root_prior_check.py"
    module_spec=importlib.util.spec_from_file_location("live_root_prior_check",script)
    module=importlib.util.module_from_spec(module_spec);sys.modules[module_spec.name]=module;module_spec.loader.exec_module(module)
    for leaf in LEAVES:
        original=[rows[leaf][r["source_request_index"]] for r in selected["roots"]]
        report=dict(schema=PRIOR_SCHEMA,complete=True,bindings=s["prior_bindings"][leaf],rows=original,
            expected=[dict(seed=seed,round=r["round"],variant="sampled4") for r in original],missing_root_ids=[])
        folder=Path(s["output_directory"])/"live-checks"/str(seed)/leaf
        _write_new(folder/"collector.json",report)
        code=module.main(["--checkpoint",checkpoint.checkpoint_path,"--model-ts",checkpoint.model_path,
            "--tables",checkpoint.tables_path,"--showdown-root",checkpoint.showdown_root,
            "--bridge-path",str(config.resolved_bridge_path()),"--seeds",str(seed),"--variants","sampled4",
            "--sims","1024","--leaf",leaf,"--strict-fallbacks","--collector-prefix-only",
            "--collector-report",str(folder/"collector.json"),"--out",str(folder/"live.json")],boundary_guard=guard)
        live=json.loads((folder/"live.json").read_bytes());gate=gate_reports(report,live)
        require(code==(0 if live["collector_gate"]["passed"] else 1),"gate exit/verdict drift")
        _check_actions(original,live)
        _write_new(folder/"gate.json",gate)


def _check_actions(original,live):
    by_index={r["round"]:r for r in live["rows"]}
    require(len(by_index)==len(live["rows"]),"duplicate independent live boundary")
    for row in original:
        item=row["variants"]["sampled4"]
        if item["status"]=="MEASURED":
            require(item.get("iterations")==4096 and item.get("worlds_searched")==4,
                "declared fixed native work telemetry changed")
            require(by_index[row["round"]]["raw_action"]==row["raw_action"]
                and by_index[row["round"]]["variants"]["sampled4"].get("engine_action")==item["action"],
                "independent live selected action changed")


def _reservations(out):
    rows=[json.loads(p.read_bytes()) for p in sorted((Path(out)/"scan-reservations").glob("*.json"))]
    require([r["ordinal"] for r in rows]==list(range(len(rows))) and len(rows)<=2000,
        "complete contiguous scan journal required")
    require(len({(r["seed"],r["request_index"]) for r in rows})==len(rows)
        and all(r["both_leaves_required"] is True and r["retry_authorized"] is False for r in rows),
        "scan boundary repeated or recipe changed")
    for row in rows:
        require(set(row)=={"ordinal","seed","request_index","raw","boundary_sha256",
            "both_leaves_required","retry_authorized"} and type(row["ordinal"]) is int
            and type(row["boundary_sha256"]) is str and len(row["boundary_sha256"])==64,
            "exact original scan reservation required")
        ScannedRoot(row["seed"],row["request_index"],row["raw"],None,None)
    return rows


def _attempted_sources(s):
    out=Path(s["output_directory"]);roster=s["contract"]["panels"]["exploration"]["seeds"]
    directories=list((out/"sources").iterdir()) if (out/"sources").exists() else []
    require(all(p.is_dir() for p in directories),"foreign source artifact")
    names={p.name for p in directories}
    attempted=roster[:len(names)]
    require(names=={str(seed) for seed in attempted},"actual sources are not the frozen source prefix")
    return attempted


def _final_binding(s,status):
    from .search_over_raw_fixed_exploration import _pins
    out=Path(s["output_directory"])
    return dict(status=status,reservations_sha256=digest(_reservations(out)),
        attempted_sources=_attempted_sources(s),
        live_rows_inventory_sha256=digest(_pins(out/"live-selections")),
        sources_inventory_sha256=digest(_pins(out/"sources")),
        completed_rows_inventory_sha256=digest(_pins(out/"completed-roots")),
        scan_stop_sha256=hashlib.sha256((out/"scan-stop.json").read_bytes()).hexdigest()
            if (out/"scan-stop.json").exists() else None)


def _check_actual_scan_inventory(s,reserved):
    """A deleted final source cannot vanish with its journal reservations."""
    out=Path(s["output_directory"]);attempted=_attempted_sources(s)
    declared={(r["seed"],r["request_index"]) for r in reserved}
    require(all(seed in attempted for seed,_ in declared),"reserved boundary lacks an actual source attempt")
    live_seeds={p.name for p in (out/"live-selections").iterdir()} if (out/"live-selections").exists() else set()
    require(live_seeds<={str(seed) for seed in attempted},"foreign live source folder")
    for seed in attempted:
        for path in (out/"live-selections"/str(seed)).glob("*/*.json"):
            row=json.loads(path.read_bytes())
            require(row["seed"]==seed and path.parent.name in LEAVES
                and path.name==str(row["round"])+".json","live row inventory identity changed")
            if row["round"]>0 and row["variants"]["sampled4"]["status"]!="NOT_SCANNED":
                require((seed,row["round"]) in declared,"actual live scan missing from reservation denominator")
    for path in (out/"completed-roots").glob("*.json"):
        row=json.loads(path.read_bytes())
        require(row["source_seed"] in attempted,"foreign completed source root")
        source=json.loads((out/"sources"/str(row["source_seed"])/"source.json").read_bytes())
        index=next(r["source_request_index"] for r in source["roots"] if r["root_id"]==row["root_id"])
        require((row["source_seed"],index) in declared,"completed root omitted from scan denominator")
    return attempted


def _check_successful_stop(s,reserved,status):
    """Read the first action-only stopping boundary, never continuation scores."""
    out=Path(s["output_directory"]);roots=[];unavailable=0;first=None
    for reservation in reserved:
        seed,index=reservation["seed"],reservation["request_index"]
        actions={}
        for leaf in LEAVES:
            row=json.loads((out/"live-selections"/str(seed)/leaf/(str(index)+".json")).read_bytes())
            require(row["raw_action"]==reservation["raw"] and
                row["boundary_sha256"]==reservation["boundary_sha256"],"stop-law boundary drift")
            item=row["variants"]["sampled4"]
            require(item["status"] in {"MEASURED","UNAVAILABLE"},"successful scan lacks both selector attempts")
            actions[leaf]=item["action"] if item["status"]=="MEASURED" else None
        root=ScannedRoot(seed,index,reservation["raw"],**actions);roots.append(root)
        unavailable=unavailable+1 if any(getattr(root,leaf) is None for leaf in LEAVES) else 0
        reason="THREE_CONSECUTIVE_UNAVAILABLE" if unavailable>=3 else stop_reason(roots,0.)
        if reason:
            require(first is None,"scan continued after its first stopping boundary")
            first=dict(reason=reason,scanned_roots=len(roots),last_seed=seed,last_request_index=index,
                counts_provisional_until_gate=True)
        elif first is not None:
            require(False,"scan continued after its first stopping boundary")
    path=out/"scan-stop.json"
    require((first is None and not path.exists()) or
        (first is not None and path.exists() and json.loads(path.read_bytes())==first),
        "saved scan stop disagrees with original provisional action prefix")
    if status=="SOURCE_CAPACITY_EXHAUSTED":
        require(first is None and _attempted_sources(s)==s["contract"]["panels"]["exploration"]["seeds"],
            "source capacity was not exhausted")
    elif status=="UNCERTAIN_SOURCE_STOP":
        attempted=_attempted_sources(s)
        require(attempted and json.loads((out/"sources"/str(attempted[-1])/"source.json").read_bytes())["status"]
            =="UNCERTAIN_SOURCE","source-stop terminal lacks its actual uncertain source")
    else:
        require(first is not None and status==first["reason"],"claimed terminal does not satisfy stopping law")


def _complete_case_diagnostics(roots,intervals,seed):
    """Required descriptive identified subset, never the inferential denominator."""
    result={}
    for leaf in LEAVES:
        groups={}
        for root,bounds in zip(roots,intervals):
            if bounds[leaf][0]==bounds[leaf][1]:groups.setdefault(root.seed,[]).append(bounds[leaf][0])
        sufficient=[(len(values),sum(values)) for values in groups.values()]
        count=sum(n for n,_ in sufficient)
        means={};ci={}
        if sufficient:
            def estimate(sample,weight):
                return sum(total for _,total in sample)/sum(n for n,_ in sample) if weight=="pooled_roots" else (
                    sum(total/n for n,total in sample)/len(sample))
            boot={weight:[] for weight in ("pooled_roots","equal_seed")};rng=random.Random(seed)
            for _ in range(10000):
                sample=rng.choices(sufficient,k=len(sufficient))
                for weight in boot:boot[weight].append(estimate(sample,weight))
            means={weight:estimate(sufficient,weight) for weight in boot}
            ci={weight:[sorted(values)[250],sorted(values)[9750]] for weight,values in boot.items()}
        result[leaf]=dict(identified_roots=count,uncertain_roots=len(roots)-count,
            identified_source_clusters=len(groups),mean=means or None,descriptive_seed_bootstrap_95=ci or None,
            interpretation="outcome-selected identified subset diagnostic; not fixed-denominator inference",
            complete_case_inference_authorized=False)
    return result


@dataclass(frozen=True)
class DeviationJob:
    spec_path:str
    entry_sha256:str
    started:float
    deadline:float

    def __call__(self,ownership,phase):
        from .search_over_raw_bounded_a2 import collect_source
        from .search_over_raw_fixed_exploration import _pins
        phase("startup");s,config,checkpoint=read_spec(self.spec_path);out=Path(s["output_directory"])
        state=dict(roots=[],stop=None,unavailable=0)
        def guard():
            require(time.time()<self.deadline,"deviation exploration hard calendar cap")
            require(hashlib.sha256(Path(self.spec_path).read_bytes()).hexdigest()==self.entry_sha256,"entry changed")
        try:
            phase("selection")
            for seed in s["contract"]["panels"]["exploration"]["seeds"]:
                guard();read_spec(self.spec_path)
                if state["stop"]:return _final_binding(s,state["stop"])
                ss=source_spec(s,seed);archive=None
                def factory(env):
                    from .search_over_raw_live_capture import build_selectors
                    def publish(leaf,row):_write_new(out/"live-selections"/str(seed)/leaf/(str(row["round"])+".json"),row)
                    return DeviationCapture(env,build_selectors(env,checkpoint,total=4096,leaves=list(LEAVES)),
                        source_seed=seed,total=4096,guard=guard,publish=publish,state=state,output=out,started=self.started)
                try:
                    archive,source,_=collect_source(ss,config,checkpoint,seed,out/"sources"/str(seed),guard,
                        capture_actor_history=False,live_capture_factory=factory)
                    reservations=[r for r in _reservations(out) if r["seed"]==seed]
                    if source["status"]!="COMPLETE":
                        return _final_binding(s,"UNCERTAIN_SOURCE_STOP")
                    if not reservations:continue
                    rows=read_live_source(ss,source)
                    replay(s,source,rows,reservations,checkpoint,config,guard)
                    for root in _selected_source(source,reservations)["roots"]:
                        guard();row=measure_root(ss,config,archive,source,root,rows,guard)
                        row.update(receipt_files=_pins(out/"roots"/digest(root["root_id"])),
                            source_receipt_files=_pins(out/"sources"/str(seed)))
                        _write_new(out/"completed-roots"/(digest(root["root_id"])+".json"),row)
                        print(json.dumps(dict(seed=seed,request_index=root["source_request_index"],
                            actions=row["actions"],status="DEVIATION_SCAN_ROOT_RECORDED")),flush=True)
                finally:
                    if archive is not None:archive.close()
            return _final_binding(s,state["stop"] or "SOURCE_CAPACITY_EXHAUSTED")
        finally:phase("cleanup")


def readout(s,terminal):
    """Rebuild every reserved root; failed or unfinished work stays uncertain."""
    from .search_over_raw_fixed_exploration import _pins
    out=Path(s["output_directory"]);reserved=_reservations(out);by_seed={}
    attempted=_check_actual_scan_inventory(s,reserved)
    if terminal["supervisor_exit_code"]==0 and terminal["worker_exit_code"]==0:
        result=terminal.get("result")
        require(type(result) is dict and result==_final_binding(s,result["status"]),
            "original successful worker scan/source inventories changed")
        _check_successful_stop(s,reserved,result["status"])
    seed_order={seed:i for i,seed in enumerate(s["contract"]["panels"]["exploration"]["seeds"])}
    positions=[seed_order[r["seed"]] for r in reserved]
    require(positions==sorted(positions),"scan source order changed")
    for r in reserved:by_seed.setdefault(r["seed"],[]).append(r)
    roots=[];intervals=[];failures={};validated_sources=0
    for seed in attempted:
        rr=by_seed.get(seed,[])
        ss=source_spec(s,seed);known={};source=None;rows=None
        try:
            source=json.loads((out/"sources"/str(seed)/"source.json").read_bytes())
            rows=read_live_source(ss,source)
            require(source["status"]=="COMPLETE","source did not close complete")
            _check_prefix(rows,rr)
            selected=_selected_source(source,rr)
            for leaf in LEAVES if rr else ():
                folder=out/"live-checks"/str(seed)/leaf
                collector=json.loads((folder/"collector.json").read_bytes());live=json.loads((folder/"live.json").read_bytes())
                original=[rows[leaf][r["source_request_index"]] for r in selected["roots"]]
                require(collector["rows"]==original and collector["bindings"]==s["prior_bindings"][leaf]
                    and collector["missing_root_ids"]==[] and collector["expected"]==[
                        dict(seed=seed,round=r["round"],variant="sampled4") for r in original],"gate roster/bindings changed")
                require(gate_reports(collector,live)==json.loads((folder/"gate.json").read_bytes()),"saved gate changed")
                _check_actions(original,live)
            validated_sources+=bool(rr)
            for r in selected["roots"]:known[r["source_request_index"]]=r
        except (ValueError,KeyError,OSError,TypeError) as error:
            # A successful worker cannot turn failed independent validation
            # into a clean experiment merely by bounding away the evidence.
            if terminal["supervisor_exit_code"]==0 and terminal["worker_exit_code"]==0:
                require(terminal.get("result",{}).get("status")=="UNCERTAIN_SOURCE_STOP"
                    and source is not None and source.get("status")=="UNCERTAIN_SOURCE",
                    "successful worker has invalid source/gate evidence: "+type(error).__name__)
            failures[str(seed)]=type(error).__name__
        for reservation in rr:
            index=reservation["request_index"];actions={leaf:None for leaf in LEAVES};bounds={leaf:(-1.,1.) for leaf in LEAVES}
            if index in known:
                for leaf in LEAVES:
                    live_row=rows[leaf][index];item=live_row["variants"]["sampled4"]
                    require(live_row["raw_action"]==reservation["raw"]
                        and live_row["boundary_sha256"]==reservation["boundary_sha256"],"scan journal/source boundary drift")
                    if item["status"]=="MEASURED":
                        actions[leaf]=item["action"]
                        bounds[leaf]=(0.,0.) if actions[leaf]==reservation["raw"] else (-1.,1.)
                root=known[index];path=out/"completed-roots"/(digest(root["root_id"])+".json")
                if path.exists():
                    row=json.loads(path.read_bytes())
                    require(row["source_receipt_files"]==_pins(out/"sources"/str(seed)),"source bytes changed")
                    read_completed_root(ss,row,source,rows);bounds=row["intervals"]
            roots.append(ScannedRoot(seed,index,reservation["raw"],**actions));intervals.append(bounds)
    summary={weight:summarize(roots,intervals,weighting=weight,bootstrap_seed=int(digest(s)[:16],16))
        for weight in ("pooled_roots","equal_seed")} if roots else {}
    return dict(schema=SCHEMA+".readout",entry_sha256=digest(s),scanned_roots=len(roots),
        validated_source_clusters=validated_sources,source_validation_failures=failures,summaries=summary,
        complete_case_diagnostics=_complete_case_diagnostics(roots,intervals,int(digest(s)[:16],16)),
        status="INCOMPLETE" if failures or terminal["supervisor_exit_code"]!=0 or
            terminal["worker_exit_code"]!=0 else "DIAGNOSTIC_INSUFFICIENT_SOURCE_COVERAGE" if
            validated_sources<32 else "EXPLORATORY_SCAN_RECORDED",
        terminal_status=terminal["status"],supervisor_exit_code=terminal["supervisor_exit_code"],
        worker_exit_code=terminal["worker_exit_code"],worker_result=terminal.get("result"),
        interpretation="stopped exploratory scan; joint bootstrap descriptive pending adaptive calibration",
        work_evidence=s["contract"].get("work_evidence","production telemetry; not atomic backup validation"),
        holdout_authorized=False,phase_b_authorized=False,retraining_authorized=False,scientific_strength_claim=False)


def run(path,review_path):
    from .search_over_raw_supervision import _supervise_owned_job
    from .search_over_raw_exploration_entry import ExplorationAuditorContract
    s,_,_=read_spec(path);entry_sha=hashlib.sha256(Path(path).read_bytes()).hexdigest()
    review_bytes=Path(review_path).read_bytes();review=json.loads(review_bytes)
    require(review.get("status")=="CLEAR_DEVIATION_EXPLORATION_ENTRY" and review.get("reviewer")=="Astra"
        and review.get("entry_spec_sha256")==entry_sha and review.get("reviewed_before_first_source_work") is True
        and review.get("holdout_authorized") is False,"prospective exact deviation entry review required")
    out=Path(s["output_directory"]);out.mkdir(parents=True,exist_ok=False,mode=0o700)
    _write_new(out/"entry.json",s);_write_new(out/"entry-review.json",review)
    def admit():
        read_spec(path)
        require(hashlib.sha256(Path(path).read_bytes()).hexdigest()==entry_sha
            and Path(review_path).read_bytes()==review_bytes,"reviewed deviation entry changed")
        return True
    start=time.time();deadline=start+28800
    _write_new(out/"launch-clock.json",dict(started_utc_seconds=start,deadline_utc_seconds=deadline))
    terminal=_supervise_owned_job(DeviationJob(str(Path(path).resolve()),entry_sha,start,deadline),
        directory=out/"auditor",binding_sha256=digest(dict(entry=entry_sha,review=hashlib.sha256(review_bytes).hexdigest())),
        verify_admission=admit,contract=ExplorationAuditorContract(28800),kind="staged-exploration-run",
        calendar_deadline_utc_seconds=deadline,work_phase="selection",validate_result=lambda v:require(
            v["status"] in {"DEVIATION_TARGET","ROOT_CAP","THREE_CONSECUTIVE_UNAVAILABLE",
                "UNCERTAIN_SOURCE_STOP","SOURCE_CAPACITY_EXHAUSTED"},"unrecognized deviation terminal"))
    admit();result=readout(s,terminal);_write_new(out/"readout.json",result);return result
