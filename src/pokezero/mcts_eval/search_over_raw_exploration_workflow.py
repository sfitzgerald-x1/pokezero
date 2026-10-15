"""Retained-panel diagnostics followed by unchanged staged selection.

No source allocation or admission. Diagnostic values never enter the selector
recipes. Existing local fixed-work incompletion is not a replay retry policy.
"""
from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import pickle

from .search_over_raw import digest,require
from .search_over_raw_world_inputs import _state_payload
from .search_over_raw_reconciled import ReconciledPhaseALedger,controller_input_hashes
from .search_over_raw_source_run import TrustedExplorationPanel
from .search_over_raw_diagnostic_producer import (FixedWorkRecipe,diagnostic_configurations,
    diagnostic_manifest,diagnostic_review_template,produce_owned_fixed_work_diagnostics,reference_value_recipe)
from .search_over_raw_native_chain import native_chain_review_template,produce_owned_native_diagnostic_chain
from .search_over_raw_consumed_materialization import _original_diagnostic_owner_guard
from .search_over_raw_attempt_cost import _inventory
from .search_over_raw_ledger import _write_new
from .search_over_raw_stage_run import run_owned_exploration_stages
from .search_over_raw_outcome_run import verify_outcome_runtime


@dataclass(frozen=True,kw_only=True)
class NativeChainRecipe:
    config: object=None
    runtime: dict | None=None
    options: dict | None=None
    helper_files: dict | None=None
    review_path: str | None=None
    unqualified_reason: str | None=None
    preserve_replay_prefix: bool=False
    paired_continuation_labels: bool=False


def preflight_diagnostic_workflow(ledger,*,recipes,chain_recipe,checkpoint_contract,
        showdown_root,file_hashes,review_path):
    """No-load, no-launch prospective recipe checks before source work."""
    require(type(ledger) is ReconciledPhaseALedger and type(chain_recipe) is NativeChainRecipe,
        "original ledger and typed native-chain recipe required")
    require(type(chain_recipe.preserve_replay_prefix) is bool and type(chain_recipe.paired_continuation_labels) is bool
        and (not chain_recipe.paired_continuation_labels or chain_recipe.preserve_replay_prefix),
        "explicit prospective paired labels require prefix-preserving replay")
    ledger.verify()
    manifest=diagnostic_manifest(ledger.contract,recipes,checkpoint_contract=checkpoint_contract,
        showdown_root=showdown_root,file_hashes=file_hashes)
    path=Path(review_path)
    require(path.is_absolute() and path==path.resolve(),"canonical diagnostic review required")
    original=path.read_bytes();review=json.loads(original)
    expected=diagnostic_review_template(manifest)
    expected.update(status="CLEAR_FIXED_WORK_DIAGNOSTIC_ROSTER",independent_review_completed=True,
        reviewed_before_first_source_work=True)
    require(digest(review)==digest(expected) and ledger.contract["input_hashes"].get(str(path))
        ==hashlib.sha256(original).hexdigest(),"original prospective diagnostic review required")
    if chain_recipe.unqualified_reason is not None:
        require(type(chain_recipe.unqualified_reason) is str and chain_recipe.unqualified_reason.strip()
            and all(getattr(chain_recipe,k) is None for k in ("config","runtime","options","helper_files","review_path")),
            "unqualified native chain cannot carry executable inputs")
    else:
        r=chain_recipe
        from .search_over_raw_materialization import BOOL_OPTIONS,MAP_OPTIONS
        require(type(r.options) is dict and set(r.options)==set(BOOL_OPTIONS)|set(MAP_OPTIONS)
            and all(type(r.options[k]) is bool for k in BOOL_OPTIONS)
            and all(r.options[k] is None or type(r.options[k]) is dict for k in MAP_OPTIONS)
            and r.options["deferred_opponent_actions"] is None and r.options["deferred_opponent_action_priors"] is None,
            "complete prospective chain options required; pending fields come from actual capture")
        verify_outcome_runtime(r.runtime,r.config)
        require(r.runtime["checkpoint_path"]==checkpoint_contract.checkpoint_path
            and r.runtime["files"][r.runtime["checkpoint_path"]]==checkpoint_contract.checkpoint_sha256
            and str(r.config.resolved_showdown_root())==showdown_root==checkpoint_contract.showdown_root,
            "native chain must retain original champion and simulator")
        require(type(r.helper_files) is dict and r.helper_files and all(type(p) is str
            and str(Path(p).resolve())==p and hashlib.sha256(Path(p).read_bytes()).hexdigest()==sha
            and ledger.contract["input_hashes"].get(p)==sha for p,sha in r.helper_files.items()),
            "native chain helper files must be prospectively pinned")
        chain_path=Path(r.review_path)
        require(chain_path.is_absolute() and chain_path==chain_path.resolve(),"canonical native chain review required")
        raw=chain_path.read_bytes()
        expected_chain=native_chain_review_template(ledger.contract,config=r.config,runtime=r.runtime,
            options=r.options,helper_files=r.helper_files,preserve_prefix=r.preserve_replay_prefix,
            paired_labels=r.paired_continuation_labels)
        expected_chain.update(status="CLEAR_NATIVE_DIAGNOSTIC_CHAIN_RECIPE",independent_review_completed=True,
            reviewed_before_first_source_work=True)
        require(digest(json.loads(raw))==digest(expected_chain) and ledger.contract["input_hashes"].get(str(chain_path))
            ==hashlib.sha256(raw).hexdigest(),"original prospective native chain review required")
    require(path.read_bytes()==original and all(ledger.contract["input_hashes"].get(p)==h
        for p,h in controller_input_hashes().items()),"diagnostic preflight inputs changed or code unpinned")
    return manifest


def run_owned_exploration_workflow(ledger,panel,*,diagnostic_recipes,native_chain_recipe,
        diagnostic_review_path,diagnostic_file_hashes,recipes,checkpoint_contract,showdown_root,file_hashes,
        raw_runtime,history_review_path,continuation_config,continuation_runtime,set_source,
        source_config=None,source_review_path=None,
        native_after_recipe=None,
        _synthetic_diagnostic_transport=None,_synthetic_chain_helpers=None,_synthetic_chain_transport=None,
        _synthetic_selector_transport=None,_synthetic_continuation_job_class=None):
    """One original panel; diagnostics cannot select a deployment recipe.

    The existing staged runner owns archive closure. If diagnostics fail before
    staging, this coordinator closes the same archives without retrying work.
    """
    require(type(ledger) is ReconciledPhaseALedger and type(panel) is TrustedExplorationPanel
        and panel.ledger is ledger,"original complete retained panel required")
    require(not native_chain_recipe.paired_continuation_labels or native_after_recipe is None,
        "paired continuation labels and legacy original-history AFTER population are distinct protocols")
    if native_after_recipe is not None:
        from .search_over_raw_after_population import _freeze_after_recipe
        after_recipe_guard=_freeze_after_recipe(native_after_recipe)
    else:after_recipe_guard=lambda:None
    staged_objects=tuple((key,value) for key,value in recipes.items())
    staged=dict(recipes=recipes,checkpoint_contract=checkpoint_contract,showdown_root=showdown_root,
        file_hashes=file_hashes,raw_runtime=raw_runtime,history_review_path=history_review_path,
        continuation_config=continuation_config,continuation_runtime=continuation_runtime,set_source=set_source)
    staged.update(source_config=source_config,source_review_path=source_review_path)
    def inputs():
        return digest(dict(staged=_state_payload(staged),diagnostic_files=diagnostic_file_hashes,diagnostics=_state_payload(diagnostic_recipes),
            chain=_state_payload(native_chain_recipe),after=_state_payload(native_after_recipe),review=str(diagnostic_review_path)))
    original_inputs=inputs();source_objects=tuple(panel.owners)
    sources=controller_input_hashes()
    try:
        manifest=preflight_diagnostic_workflow(ledger,recipes=diagnostic_recipes,chain_recipe=native_chain_recipe,
            checkpoint_contract=checkpoint_contract,showdown_root=checkpoint_contract.showdown_root,file_hashes=diagnostic_file_hashes,
            review_path=diagnostic_review_path)
        after_recipe_guard()
        if native_after_recipe is not None:
            from .search_over_raw_after_population import preflight_after_population
            preflight_after_population(ledger,native_after_recipe,checkpoint_contract=checkpoint_contract)
        panel.validate()
        require(inputs()==original_inputs,"original workflow inputs changed during panel validation")
        owners,population=produce_owned_fixed_work_diagnostics(ledger,panel,recipes=diagnostic_recipes,
            checkpoint_contract=checkpoint_contract,showdown_root=checkpoint_contract.showdown_root,file_hashes=diagnostic_file_hashes,
            review_path=diagnostic_review_path,_synthetic_job_transport=_synthetic_diagnostic_transport)
        # Freeze BEFORE any chain or later source/owning-reader callback.
        fixed_paths=[ledger.directory/"fixed-work-diagnostics"]
        fixed_inventory=_inventory(fixed_paths)
        population_bytes=(ledger.directory/"fixed-work-diagnostics.json").read_bytes()
        population_sha=digest(population)
        owner_objects=tuple((key,owner,owner.job,owner.reader,owner.selected) for key,owner in owners.items())
        owner_types=tuple((type(o),type(job),type(reader)) for _,o,job,reader,_ in owner_objects)
        def owner_inputs():
            return digest([dict(key=key,job=hashlib.sha256(pickle.dumps(o.job)).hexdigest(),
                selected=o.selected,selected_sha=o.selected_sha256,reader=o.reader._inputs(),readout=o.reader.last_validation)
                for key,o in owners.items()])
        frozen_owner_inputs=owner_inputs();chain_guards=[]
        def original_diagnostics():
            after_recipe_guard()
            # Check exact classes BEFORE dispatching any potentially replaced
            # reader method, including for reference/HP owners without chains.
            require(tuple(type(o) for o in owners.values())==tuple(types[0] for types in owner_types)
                and tuple((type(o),type(o.job),type(o.reader)) for o in owners.values())==owner_types,
                "original diagnostic owner/job/reader class changed")
            require(tuple(owners)==tuple(k for k,*_ in owner_objects)
                and all(owners[k] is o and o.job is job and o.reader is reader and o.selected is selected
                    for k,o,job,reader,selected in owner_objects)
                and owner_inputs()==frozen_owner_inputs and digest(population)==population_sha
                and _inventory(fixed_paths)==fixed_inventory
                and (ledger.directory/"fixed-work-diagnostics.json").read_bytes()==population_bytes,
                "original diagnostic evidence changed during later work")
            for guard in chain_guards:guard()
        native_cfg=next(c for c in diagnostic_configurations() if c.arm=="incumbent" and c.leaf=="model")
        native_slots=[s for s in population["slots"] if s["configuration_sha256"]==native_cfg.identity]
        chains=[];private_chains=[]
        for slot in native_slots:
            root_id=slot["root_id"]
            row=dict(root_id=root_id,source_seed=slot["source_seed"],
                native_status=slot["status"],status=slot["status"],diagnostic=None)
            if slot["status"]=="DIAGNOSTIC_COMPLETE":
                if native_chain_recipe.unqualified_reason is not None:
                    row.update(status="SKIPPED_UNQUALIFIED_CHAIN",reason=native_chain_recipe.unqualified_reason)
                else:
                    r=native_chain_recipe;owner=owners[(root_id,native_cfg.identity)]
                    private,report=produce_owned_native_diagnostic_chain(ledger,owner,config=r.config,runtime=r.runtime,
                        options=r.options,helper_files=r.helper_files,review_path=r.review_path,
                        preserve_prefix=r.preserve_replay_prefix,paired_labels=r.paired_continuation_labels,
                        _synthetic_helpers=_synthetic_chain_helpers,_synthetic_job_transport=_synthetic_chain_transport)
                    # Existing callback-free owning guard; no new owner layer.
                    def freeze_chain(private,report,owner):
                        mats,replays=private["materializations"],private["replays"]
                        guard=_original_diagnostic_owner_guard(owner.selected,mats,replays,owner.reader)
                        readers=[owner.reader]+[row["validator"] for row in mats+replays]
                        readouts={str(reader.directory):deepcopy(reader.last_validation) for reader in readers}
                        root=ledger.directory/"native-diagnostic-chains"/digest(owner.job.statistics_root_id)
                        inventory=_inventory([root]);report_sha=digest(report)
                        objects=tuple((k,v) for k,v in private.items())
                        plans=tuple((private[k],private[k].payload_utf8) for k in ("materialization_plan","replay_plan"))
                        def check():
                            require(tuple(private)==tuple(k for k,_ in objects)
                                and all(private[k] is obj for k,obj in objects)
                                and all(plan.payload_utf8==raw for plan,raw in plans)
                                and digest(report)==report_sha and _inventory([root])==inventory,
                                "original earlier native chain changed during later work")
                            guard(readouts)
                        return check
                    chain_guards.append(freeze_chain(private,report,owner))
                    original_diagnostics()
                    private_chains.append(private)
                    row.update(status="CHAIN_RECORDED",diagnostic=report)
            chains.append(row)
            original_diagnostics()
        require(len(chains)==len(manifest["slots"])//6,"native chain roster cannot shrink")
        for owner in owners.values():owner.validate(ledger);original_diagnostics()
        panel.validate();original_diagnostics()
        after_population=None
        if native_after_recipe is not None:
            from .search_over_raw_after_population import produce_owned_after_population
            from .search_over_raw_nested_ownership import current_scope
            scope=current_scope()
            require(scope is not None,"actual enclosing AFTER auditor required")
            if _synthetic_chain_helpers is not None:extract=_synthetic_chain_helpers.extract
            elif private_chains:
                import pokezero_search
                module=str(Path(pokezero_search.__file__).resolve())
                require(module in native_chain_recipe.helper_files and pokezero_search.MODEL_FEATURE_ENABLED is True,
                    "original declared native AFTER extraction required")
                extract=pokezero_search.native_scoped_boundary
            else:extract=None
            after_population=produce_owned_after_population(ledger,manifest=manifest,chain_rows=chains,
                private_chains=private_chains,recipe=native_after_recipe,checkpoint_contract=checkpoint_contract,
                ownership=dict(directory=str(scope.directory/"owned-groups"),controller_pid=scope.pid,
                    registration_sha256=scope.binding,protocol="active-generations-v1"),native_extract=extract)
            original_diagnostics();after_population.validate_live()
        reports_sha=digest(dict(population=population,chains=chains))
        paths=[ledger.directory/"fixed-work-diagnostics",ledger.directory/"native-diagnostic-chains"]
        # Absent namespaces are retained as absence, not discovered successful attempts.
        original_inventory=_inventory(paths)
        def final_guard():
            ledger.verify()
            require(inputs()==original_inputs and tuple(recipes)==tuple(k for k,_ in staged_objects)
                and all(recipes[k] is obj for k,obj in staged_objects)
                and controller_input_hashes()==sources,"original diagnostic/staged workflow inputs changed")
            original_diagnostics()
            if after_population is not None:after_population.check_closed()
            require(digest(dict(population=population,chains=chains))==reports_sha
                and _inventory(paths)==original_inventory
                and (ledger.directory/"fixed-work-diagnostics.json").read_bytes()==population_bytes,
                "original diagnostic evidence changed during staging")
        final_guard()
        result=run_owned_exploration_stages(ledger,panel,**staged,
            _synthetic_selector_transport=_synthetic_selector_transport,
            _synthetic_continuation_job_class=_synthetic_continuation_job_class)
        # Archives are now CLOSED by the original scheduler. Recheck original
        # data/bytes without invoking a live-archive validator after closure.
        final_guard()
        require(tuple(panel.owners)==source_objects and all(not o.archive._public and not o.archive._private
            for o in source_objects),"original panel archives must close after staged work")
        report=dict(schema="pokezero.search-over-raw.exploration-workflow.v1",
            diagnostic_population=population,native_chains=chains,staged=result,
            diagnostic_results_used_for_selection=False,private_archive_returned=False,
            runtime_qualified=False,resource_cost_complete=False,scientific_measurement_validated=False,
            collection_authorized=False,holdout_authorized=False,phase_b_authorized=False)
        if after_population is not None:
            report.update(schema="pokezero.search-over-raw.exploration-workflow.v2",
                native_after_population=deepcopy(after_population.result))
        _write_new(ledger.directory/"exploration-workflow.json",report)
        return report
    except BaseException as error:
        try:
            if not (ledger.directory/"operational-stop.json").exists():ledger.stop(type(error).__name__)
        except BaseException as stop:error.add_note("Workflow stop persistence also failed: "+type(stop).__name__)
        raise
    finally:
        for owner in source_objects:owner.close()


def read_recorded_workflow(ledger,report,*,diagnostic_recipes,native_chain_recipe,diagnostic_review_path,
        diagnostic_file_hashes,checkpoint_contract,auditor_directory,auditor_pid,allow_synthetic=False,
        _synthetic_helpers=None,native_after_recipe=None,expected_outer_terminal=None):
    """Independent completed child readback; never poll historical processes.

    Job pickles below are original trusted LOCAL implementation artifacts,
    anchored by the caller's original outer terminal and immutable readouts.
    No pickle or dynamic class name is accepted from external entry JSON.
    Source attachment was checked while archives were live; this reader does
    not reconstruct a closed private archive or claim simulator equivalence.
    """
    from dataclasses import asdict
    from .search_over_raw_fixed_work import FixedWorkDiagnosticContract
    from .search_over_raw_fixed_work_run import FixedWorkRunValidator
    from .search_over_raw_supervision import FixedWorkAdapterJob
    from .search_over_raw_attempt_cost import AttemptCostSpec,_attempt
    from .search_over_raw_resources import resource_source_hash
    from .search_over_raw_materialization_run import OwnedRecordedMaterializationJob,MaterializationRunValidator,MaterializationRunContract
    from .search_over_raw_boundary_run import OwnedNativePathBoundaryJob,NativePathBoundaryRunValidator,NativePathBoundaryRunContract
    from .search_over_raw_consumed_materialization import reconcile_owned_replay_diagnostic
    from .search_over_raw_nested_ownership import DiagnosticIncompletePolicy
    require(type(allow_synthetic) is bool and type(auditor_pid) is int and auditor_pid>0,
        "original outer owner and explicit readback scope required")
    if native_after_recipe is not None:
        from .search_over_raw_after_population import _freeze_after_recipe
        after_recipe_guard=_freeze_after_recipe(native_after_recipe)
    else:after_recipe_guard=lambda:None
    if allow_synthetic:
        require(ledger.contract["execution_ready"] is False and ledger.contract["collection_authorized"] is False
            and _synthetic_helpers is not None and getattr(_synthetic_helpers,"synthetic_process_fixture",False) is True,
            "authored completed readback cannot qualify admitted science")
    else:require(_synthetic_helpers is None,"authored helpers cannot enter production readback")
    recipe_objects=tuple((key,obj,type(obj)) for key,obj in diagnostic_recipes.items())
    chain_fields=tuple((key,getattr(native_chain_recipe,key),type(getattr(native_chain_recipe,key)))
        for key in ("config","runtime","options","helper_files","review_path","unqualified_reason","preserve_replay_prefix",
            "paired_continuation_labels"))
    original_types=(type(diagnostic_recipes),type(native_chain_recipe),type(diagnostic_file_hashes),type(checkpoint_contract),
        type(_synthetic_helpers))
    def helper_methods():
        return tuple((id(getattr(fn,"__func__",fn)),id(getattr(fn,"__self__",None)))
            for fn in (getattr(_synthetic_helpers,key,None) for key in ("extract","support","successor")))
    original_helpers=helper_methods()
    def external_inputs():
        return digest(dict(recipes=_state_payload(diagnostic_recipes),chain=_state_payload(native_chain_recipe),
            files=diagnostic_file_hashes,checkpoint=_state_payload(vars(checkpoint_contract)),
            review=str(diagnostic_review_path),auditor_directory=str(auditor_directory),auditor_pid=auditor_pid,
            synthetic=allow_synthetic,after=_state_payload(native_after_recipe),outer_terminal=expected_outer_terminal))
    external_sha=external_inputs();original_controller=controller_input_hashes()
    external_paths=set(original_controller)|set(diagnostic_file_hashes)|{str(Path(diagnostic_review_path))}
    if native_chain_recipe.unqualified_reason is None:
        external_paths.update(native_chain_recipe.runtime["files"])
        external_paths.update(native_chain_recipe.helper_files)
        external_paths.add(native_chain_recipe.review_path)
    if native_after_recipe is not None and native_after_recipe.unqualified_reason is None:
        external_paths.update(native_after_recipe.runtime["files"])
        external_paths.add(native_after_recipe.review_path)
    def external_bytes():
        data={}
        for value in external_paths:
            path=Path(value)
            require(type(value) is str and path.is_absolute() and path==path.resolve() and path.is_file(),
                "original canonical external workflow inputs required")
            data[value]=hashlib.sha256(path.read_bytes()).hexdigest()
        return data
    original_external_bytes=external_bytes()
    def external_guard():
        after_recipe_guard()
        require((type(diagnostic_recipes),type(native_chain_recipe),type(diagnostic_file_hashes),type(checkpoint_contract),
                type(_synthetic_helpers))==original_types
            and tuple(diagnostic_recipes)==tuple(k for k,_,_ in recipe_objects)
            and all(diagnostic_recipes[k] is obj and type(obj) is kind for k,obj,kind in recipe_objects)
            and all(getattr(native_chain_recipe,k) is obj and type(obj) is kind for k,obj,kind in chain_fields),
            "original external workflow inputs or classes changed")
        require(external_inputs()==external_sha and helper_methods()==original_helpers
            and controller_input_hashes()==original_controller and external_bytes()==original_external_bytes,
            "original external workflow inputs or bytes changed after callbacks")
    def load(path):
        path=Path(path)
        require(path==path.resolve() and path.is_file(),"original canonical workflow evidence required")
        return json.loads(path.read_bytes())
    def attempt(path,kind,phase,contract):
        path=Path(path);claim=load(path/"claim.json")
        spec=AttemptCostSpec(str(path),claim["job_sha256"],claim["binding_sha256"],kind,phase,contract.receipt(),
            resource_source_hash(),contract.emergency_watchdog_seconds,str(Path(auditor_directory).resolve()))
        cost,launch,terminal=_attempt(spec)
        require(launch is not None and launch["supervisor_pid"]==auditor_pid and terminal is not None,
            "diagnostic child must belong to original actual auditor")
        return claim,terminal
    def original_job(path,claim,cls):
        raw=(Path(path)/"job.pickle").read_bytes()
        require(hashlib.sha256(raw).hexdigest()==claim["job_sha256"],"original local job bytes changed")
        job=pickle.loads(raw)
        require(type(job) is cls or allow_synthetic and isinstance(job,cls)
            and getattr(job,"synthetic_process_fixture",False) is True,"original typed local diagnostic job required")
        return job
    frozen=digest(report);inventory=_inventory([ledger.directory])
    has_after=native_after_recipe is not None
    require(type(report) is dict and set(report)=={"schema","diagnostic_population","native_chains","staged",
        "diagnostic_results_used_for_selection","private_archive_returned","runtime_qualified","resource_cost_complete",
        "scientific_measurement_validated","collection_authorized","holdout_authorized","phase_b_authorized"}
            | ({"native_after_population"} if has_after else set())
        and report["schema"]=="pokezero.search-over-raw.exploration-workflow.v"+("2" if has_after else "1")
        and all(report[k] is False for k in report if k not in {"schema","diagnostic_population","native_chains","staged","native_after_population"})
        and digest(load(ledger.directory/"exploration-workflow.json"))==frozen,
        "exact original nonqualifying workflow result required")
    manifest=preflight_diagnostic_workflow(ledger,recipes=diagnostic_recipes,chain_recipe=native_chain_recipe,
        checkpoint_contract=checkpoint_contract,showdown_root=checkpoint_contract.showdown_root,
        file_hashes=diagnostic_file_hashes,review_path=diagnostic_review_path)
    if has_after:
        from .search_over_raw_after_population import preflight_after_population
        preflight_after_population(ledger,native_after_recipe,checkpoint_contract=checkpoint_contract)
    population=report["diagnostic_population"]
    require(digest(population)==digest(load(ledger.directory/"fixed-work-diagnostics.json"))
        and population["manifest_sha256"]==digest(manifest) and population["required_slots"]==len(manifest["slots"])
        and len(population["slots"])==len(manifest["slots"]),"complete original fixed diagnostic population required")
    cfgs={c.identity:c for c in diagnostic_configurations()};fixed=FixedWorkDiagnosticContract();validated={};attempt_paths=[]
    for slot,row in zip(manifest["slots"],population["slots"]):
        root=slot["root"];cfg=cfgs[slot["configuration_sha256"]];recipe=diagnostic_recipes[cfg.identity]
        require(digest({k:row[k] for k in ("root_id","source_seed","configuration_sha256")})
            ==digest(dict(root_id=root["root_id"],source_seed=root["source_seed"],configuration_sha256=cfg.identity)),
            "original diagnostic slot order/identity changed")
        source=load(ledger.directory/"source-attempts"/str(root["source_seed"])/"source.json")
        available=any(r["root_id"]==root["root_id"] for r in source["roots"])
        expected_status="UNSTARTED" if not available else "SKIPPED_UNQUALIFIED_RECIPE" if recipe.unqualified_reason else None
        path=ledger.directory/"fixed-work-diagnostics"/digest(slot)
        if expected_status is not None:
            require(row["status"]==expected_status and row["attempt_directory"] is None
                and all(row[k] is None for k in ("accepted_work","selected_sha256","cost")) and not path.exists(),
                "unavailable original slot cannot gain work or lose uncertainty")
            continue
        require(row["attempt_directory"]==str(path),"original fixed attempt namespace changed")
        attempt_paths.append(path)
        claim,terminal=attempt(path,"fixed-work","selection",fixed)
        if terminal["supervisor_exit_code"]==124:
            expected=DiagnosticIncompletePolicy(digest(manifest),digest(slot),claim["job_sha256"],claim["binding_sha256"])
            require(row["status"]=="DIAGNOSTIC_INCOMPLETE" and row["accepted_work"] is None
                and digest(claim.get("diagnostic_incomplete_policy"))==digest(expected.receipt()),
                "only original prospectively bound clean fixed-work incompletion is local")
            continue
        require(terminal["supervisor_exit_code"]==0 and row["status"]=="DIAGNOSTIC_COMPLETE"
            and row["accepted_work"]==fixed.accepted_boundary(cfg.arm),"original complete fixed job required")
        job=original_job(path,claim,FixedWorkAdapterJob)
        require(digest(_state_payload(job.configuration))==digest(_state_payload(cfg))
            and digest(_state_payload(job.checkpoint_contract))==digest(_state_payload(checkpoint_contract))
            and job.showdown_root==checkpoint_contract.showdown_root and job.statistics_root_id==root["root_id"]
            and job.root_id==root["root_id"]+":"+cfg.identity and type(job.selection_seed) is int
            and job.capture_native_public_fold is recipe.capture_native_public_fold
            and digest(_state_payload(job.owned_reference_value_recipe))==digest(_state_payload(reference_value_recipe(recipe)))
            and job.selection_seed==root["source_seed"] and job.expected_runtime_sha256==digest(recipe.runtime)
            and job.expected_public_request_sha256==row["owning_readout"]["full_public_request_sha256"],
            "completed fixed job differs from original prospective slot/recipe")
        from .search_over_raw_reference_values import ReferenceValueRunValidator
        labels=reference_value_recipe(recipe)
        reader_class=FixedWorkRunValidator if labels is None else ReferenceValueRunValidator
        reader=reader_class(**({"recipe":labels} if labels is not None else {}),
            directory=path,job_sha256=claim["job_sha256"],binding_sha256=claim["binding_sha256"],
            configuration=cfg,root_id=root["root_id"]+":"+cfg.identity,statistics_root_id=root["root_id"],
            selection_seed=root["source_seed"],runtime_sha256=digest(recipe.runtime),consumed_binding=job._consumed_binding,
            public_request_sha256=row["owning_readout"]["full_public_request_sha256"],file_hashes=diagnostic_file_hashes,
            checkpoint_path=checkpoint_contract.checkpoint_path,checkpoint_sha256=checkpoint_contract.checkpoint_sha256,
            showdown_root=checkpoint_contract.showdown_root,allow_synthetic=allow_synthetic,
            native_public_fold_required=recipe.capture_native_public_fold)
        selected=terminal["result"]
        require(reader(selected)==row["selected_sha256"]==digest(selected)
            and digest(reader.last_validation)==digest(row["owning_readout"]),"independent fixed child readback differs")
        validated[(root["root_id"],cfg.identity)]=(selected,reader,job)
    fixed_root=ledger.directory/"fixed-work-diagnostics"
    require({p for p in fixed_root.iterdir()}==set(attempt_paths) if fixed_root.exists() else not attempt_paths,
        "unexpected or missing fixed-work attempt directory")
    native=next(c for c in cfgs.values() if c.arm=="incumbent" and c.leaf=="model")
    native_slots=[r for r in population["slots"] if r["configuration_sha256"]==native.identity]
    require(len(report["native_chains"])==len(native_slots),"full original native-chain roster required")
    total_materializers=total_replays=0;chain_paths=[];after_contexts={}
    total_label_jobs=total_label_cells=total_complete_labels=0
    for row,slot in zip(report["native_chains"],native_slots):
        require(row["root_id"]==slot["root_id"] and type(row["source_seed"]) is int
            and row["source_seed"]==slot["source_seed"] and row["native_status"]==slot["status"],"native chain slot drift")
        root=ledger.directory/"native-diagnostic-chains"/digest(row["root_id"])
        expected=slot["status"] if slot["status"]!="DIAGNOSTIC_COMPLETE" else (
            "SKIPPED_UNQUALIFIED_CHAIN" if native_chain_recipe.unqualified_reason else "CHAIN_RECORDED")
        require(row["status"]==expected,"original chain uncertainty/status changed")
        if expected!="CHAIN_RECORDED":
            require(row["diagnostic"] is None and not root.exists(),"uncertain chain cannot acquire results")
            continue
        chain_paths.append(root);r=native_chain_recipe;recorded=row["diagnostic"]
        require(digest(recorded)==digest(load(root/"diagnostic.json")),"original native-chain diagnostic changed")
        selected,reader,job=validated[(row["root_id"],native.identity)]
        chain_claim=load(root/"claim.json")
        plan=chain_claim["materialization_plan"];replay_plan=load(root/"replay-plan.json")["private_plan"]
        from .search_over_raw_native_leaf_cells import plan_native_leaf_cells,native_leaf_run_binding
        leaf_plan=plan_native_leaf_cells(selected,run_binding=native_leaf_run_binding(reader),
            consumed_binding=reader.consumed,ancestry_contract=job.native_ancestry_contract,
            paired_labels=r.paired_continuation_labels)
        def private_sha(value):
            return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(",",":"),ensure_ascii=False,allow_nan=False).encode()).hexdigest()
        require(private_sha(plan)==recorded["materialization_plan_sha256"]
            and private_sha(replay_plan)==recorded["replay_plan_sha256"],"original immutable native plans changed")
        require(leaf_plan.payload_utf8==json.dumps(chain_claim["leaf_cell_plan"],sort_keys=True,
            separators=(",",":"),ensure_ascii=False,allow_nan=False)
            and leaf_plan.sha256==chain_claim["leaf_cell_plan_sha256"]==recorded["leaf_cell_plan_sha256"]
            and recorded["retained_unique_leaf_events"]==leaf_plan.receipt()["retained_unique_evaluation_events"]
            and recorded["required_leaf_label_cells"]==leaf_plan.receipt()["required_label_cells"]
            and type(recorded["leaf_label_process_launches"]) is int
            and type(recorded["completed_leaf_label_cells"]) is int
            and (r.paired_continuation_labels or recorded["leaf_label_process_launches"]==recorded["completed_leaf_label_cells"]==0),
            "original unique retained leaf-cell roster changed or acquired unowned labels")
        require(plan["native_selected_sha256"]==digest(selected) and len(plan["records"])==4,
            "all four original constructor coordinates required")
        mats=[];replays=[];reservations=[]
        recipe_sha=digest(native_chain_review_template(ledger.contract,config=r.config,runtime=r.runtime,
            options=r.options,helper_files=r.helper_files,preserve_prefix=r.preserve_replay_prefix,
            paired_labels=r.paired_continuation_labels)|dict(status="CLEAR_NATIVE_DIAGNOSTIC_CHAIN_RECIPE",
                independent_review_completed=True,reviewed_before_first_source_work=True))
        def reservation(path,claim,role,coordinate,plan_sha,expected_inputs,contract):
            binding=digest(dict(recipe_sha256=recipe_sha,plan_sha256=plan_sha,coordinate=coordinate,
                job_sha256=claim["job_sha256"],expected=expected_inputs))
            require(claim["binding_sha256"]==binding,"original prospective native-chain attempt binding changed")
            kind,work=("materialization-run","materialization") if role=="materialization" else (
                ("outcome-run","continuations") if role=="labels" else ("boundary-run","boundaries"))
            spec=AttemptCostSpec(str(path),claim["job_sha256"],binding,kind,work,contract.receipt(),resource_source_hash(),
                contract.emergency_watchdog_seconds,str(Path(auditor_directory).resolve()))
            name=digest(dict(path=str(path),binding=binding))+".json"
            require(digest(load(root/"reservations"/name))==digest(dict(role=role,attempt=vars(spec),
                plan_sha256=plan_sha,retry_authorized=False)),"original native-chain reservation differs from planned attempt")
            reservations.append(name)
        paired=r.paired_continuation_labels
        from .search_over_raw_paired_materialization import (OwnedPairedMaterializationJob,PairedMaterializationRunContract,
            PairedMaterializationRunValidator)
        from .search_over_raw_native_chain import plan_owned_native_materializations
        from .search_over_raw_paired_chain import pin_original_result
        def paired_original(path):
            entries=[item["owner"] for item in recorded["owning_jobs"] if item["owner"]["directory"]==str(path)]
            require(len(entries)==1,"unique original paired owner inventory required")
            original=entries[0]
            return pin_original_result(path,job_sha256=original["job_sha256"],readout=original["readout"])
        if paired:
            fresh=plan_owned_native_materializations(selected,native_validator=reader,config=r.config,runtime=r.runtime,
                options=r.options,attempt_root=Path(plan["attempt_root"]),public_state_sha256=plan["public_state_sha256"],
                paired_context=job.context.public_materialization_state).receipt()
            for original,current in zip(plan["records"],fresh["records"]):
                if original["status"]=="PLANNED_NOT_ADMITTED" and current["status"]=="ATTEMPT_ALREADY_CLAIMED":
                    current["status"]="PLANNED_NOT_ADMITTED"
            require(digest(fresh)==digest(plan),"paired constructor plan differs from original native/context inputs")
        materialization_contract=PairedMaterializationRunContract() if paired else MaterializationRunContract()
        materialization_cls=OwnedPairedMaterializationJob if paired else OwnedRecordedMaterializationJob
        for record in plan["records"]:
            path=Path(record["attempt_directory"])
            require(path.parent==root/"materializations","original constructor namespace required")
            if paired and record["status"]=="CONSUMED_OVERLAY_NOT_REPRESENTED_BY_ORIGINAL_CONTEXT":
                require(not path.exists(),"unavailable paired context cannot acquire a constructor attempt")
                continue
            claim,terminal=attempt(path,"materialization-run","materialization",materialization_contract)
            require(terminal["supervisor_exit_code"]==0,"failed materializer cannot clear completed workflow")
            pinned=paired_original(path) if paired else None
            expected_job=materialization_cls(r.config,record["recorded_inputs"],r.runtime,
                **(dict(original_public_context=job.context.public_materialization_state) if paired else {}))
            require(digest(expected_job.input_binding())==record["expected_inputs_sha256"],"original constructor plan input drift")
            actual=original_job(path,claim,materialization_cls)
            require(digest(actual.input_binding())==digest(expected_job.input_binding()),"actual constructor job differs from plan")
            reservation(path,claim,"materialization",dict(construction_ordinal=record["construction_ordinal"]),
                private_sha(plan),expected_job.input_binding(),materialization_contract)
            mr=(PairedMaterializationRunValidator if paired else MaterializationRunValidator)(
                directory=path,job_sha256=claim["job_sha256"],binding_sha256=claim["binding_sha256"],
                expected_inputs=expected_job.input_binding(),runtime=r.runtime,allow_synthetic=allow_synthetic,
                **(dict(original_public_context=job.context.public_materialization_state,
                    expected_context=pinned["paired_context"]) if paired else {}))
            require(mr(terminal["result"]["capture"])==digest(terminal["result"]["capture"]),"constructor owner revalidation required")
            mats.append(dict(construction_ordinal=record["construction_ordinal"],result=terminal["result"],validator=mr))
        from .search_over_raw_native_prefix import (OwnedNativePrefixBoundaryJob,
            NativePrefixBoundaryRunContract, NativePrefixBoundaryRunValidator)
        from .search_over_raw_paired_replay import OwnedPairedPrefixJob,PairedPrefixRunContract,PairedPrefixRunValidator
        prefix=r.preserve_replay_prefix
        expected_schema="pokezero.search-over-raw.private-native-replay-plan.v3" if paired else \
            "pokezero.search-over-raw.private-native-replay-plan.v2" if prefix else \
            "pokezero.search-over-raw.private-native-replay-plan.v1"
        require(replay_plan["schema"]==expected_schema, "replay law differs from original prospective recipe")
        replay_cls=OwnedPairedPrefixJob if paired else OwnedNativePrefixBoundaryJob if prefix else OwnedNativePathBoundaryJob
        replay_contract=PairedPrefixRunContract() if paired else NativePrefixBoundaryRunContract() if prefix else NativePathBoundaryRunContract()
        replay_reader=PairedPrefixRunValidator if paired else NativePrefixBoundaryRunValidator if prefix else NativePathBoundaryRunValidator
        for record in replay_plan["records"]:
            path=Path(record["attempt_directory"]);spec=record["job_spec"]
            require(path.parent==root/"replays","original replay namespace required")
            if spec is None:
                require(not path.exists(),"blocked replay cannot acquire an attempt")
                continue
            extra={}
            if paired:
                material=next(m for m in mats if m["construction_ordinal"]==record["construction_ordinal"])
                extra["snapshot"]=material["validator"].retained_context(material["result"])
            expected_job=replay_cls(config=r.config,**{k:v for k,v in spec.items() if k!="expected_inputs_sha256"},**extra)
            require(digest(expected_job.input_binding())==spec["expected_inputs_sha256"],"original replay plan inputs changed")
            claim,terminal=attempt(path,"boundary-run","boundaries",replay_contract)
            require(terminal["supervisor_exit_code"]==0,"failed replay cannot clear completed workflow")
            pinned=paired_original(path) if paired else None
            actual=original_job(path,claim,replay_cls)
            require(digest(actual.input_binding())==digest(expected_job.input_binding()),"actual replay job differs from plan")
            reservation(path,claim,"replay",{k:record[k] for k in ("construction_ordinal","evaluation_ordinal")},
                private_sha(replay_plan),expected_job.input_binding(),replay_contract)
            rr=replay_reader(directory=path,job_sha256=claim["job_sha256"],binding_sha256=claim["binding_sha256"],
                expected_inputs=expected_job.input_binding(),expected_raw_root=expected_job.raw_root,runtime=r.runtime,
                allow_synthetic=allow_synthetic,**(dict(snapshot=expected_job.snapshot,
                    expected_endpoint=pinned["paired_endpoint"]) if paired else {}))
            require(rr(terminal["result"]["capture"]) is not None,"replay owner revalidation required")
            replays.append(dict(construction_ordinal=record["construction_ordinal"],evaluation_ordinal=record["evaluation_ordinal"],
                result=terminal["result"],validator=rr))
        if paired:
            from .search_over_raw_paired_chain import paired_label_descriptors,paired_label_report
            from .search_over_raw_paired_labels import OwnedPairedLabelJob,PairedLabelRunContract,PairedLabelRunValidator
            descriptors=paired_label_descriptors(leaf_plan,replays,config=r.config,runtime=r.runtime,attempt_root=root/"labels")
            require(type(recorded.get("paired_labels")) is list and len(recorded["paired_labels"])==len(descriptors),
                "full original unique paired-label event roster required")
            label_count=complete_count=0
            for descriptor,reported in zip(descriptors,recorded["paired_labels"]):
                descriptor["plan_sha256"]=leaf_plan.sha256
                path=Path(descriptor["attempt_directory"]);expected_job=descriptor["job"]
                if expected_job is None:
                    require(not path.exists() and digest(reported)==digest(paired_label_report(descriptor)),
                        "unavailable canonical paired event must retain all eight null cells")
                    continue
                contract=PairedLabelRunContract()
                claim,terminal=attempt(path,"outcome-run","continuations",contract)
                require(terminal["supervisor_exit_code"]==0,"failed paired label attempt cannot clear workflow")
                pin_original_result(path,job_sha256=reported["job_sha256"],readout=reported["owning_readout"])
                actual=original_job(path,claim,OwnedPairedLabelJob)
                expected_inputs=expected_job.input_binding()
                require(digest(actual.input_binding())==digest(expected_inputs),"paired label job differs from original event/prediction/RNG")
                reservation(path,claim,"labels",{k:descriptor[k] for k in ("construction_ordinal","evaluation_ordinal","event_id")},
                    leaf_plan.sha256,expected_inputs,contract)
                lr=PairedLabelRunValidator(directory=path,job_sha256=claim["job_sha256"],binding_sha256=claim["binding_sha256"],
                    root_binding=expected_job.root_binding,expected_snapshot=expected_job.snapshot,runtime=r.runtime,
                    expected_inputs=expected_inputs,allow_synthetic=allow_synthetic)
                observed=paired_label_report(descriptor,dict(result=terminal["result"],validator=lr))
                require(digest(observed)==digest(reported),"paired labels differ from original independently read outcomes")
                label_count+=1;complete_count+=sum(c["status"]=="COMPLETE" for c in observed["cells"])
            require(recorded["leaf_label_process_launches"]==label_count and recorded["completed_leaf_label_cells"]==complete_count,
                "paired label counts must follow original unique event owners")
            total_label_jobs+=label_count;total_label_cells+=len(descriptors)*8;total_complete_labels+=complete_count
        require({p.name for p in (root/"reservations").iterdir()}==set(reservations),"original complete reservation roster required")
        if allow_synthetic:helpers=_synthetic_helpers
        else:
            import pokezero_search
            module=str(Path(pokezero_search.__file__).resolve())
            require(module in r.helper_files and pokezero_search.MODEL_FEATURE_ENABLED is True,"original declared native helpers required")
            from types import SimpleNamespace
            helpers=SimpleNamespace(extract=pokezero_search.native_scoped_boundary,support=pokezero_search.native_edge_support,
                successor=pokezero_search.native_scoped_successor_support)
        joined=reconcile_owned_replay_diagnostic(selected,mats,replays,native_validator=reader,
            ancestry_contract=job.native_ancestry_contract,native_extract=helpers.extract,
            native_support=getattr(helpers,"support",None),native_successor_support=getattr(helpers,"successor",None))
        require(digest(joined["owned_replay_join"])==digest(recorded["owned_replay_join"])
            and recorded["actual_constructor_runs"]==len(mats) and recorded["actual_replay_runs"]==len(replays),
            "completed native-chain joins differ from independently read children")
        if paired:
            from .search_over_raw_paired_chain import paired_value_diagnostics
            require(digest(recorded["paired_value_diagnostics"])==digest(paired_value_diagnostics(
                leaf_plan,recorded["paired_labels"],joined["owned_replay_join"])),
                "paired prediction/outcome diagnostics differ from original scoped replay evidence")
        total_materializers+=len(mats);total_replays+=len(replays)
        after_contexts[row["root_id"]]=dict(plan=leaf_plan,replays=replays)
    chain_root=ledger.directory/"native-diagnostic-chains"
    require({p for p in chain_root.iterdir()}==set(chain_paths) if chain_root.exists() else not chain_paths,
        "unexpected or missing native-chain namespace")
    external_guard()
    after_readback=None
    if has_after:
        from .search_over_raw_after_population import read_recorded_after_population
        after_readback=read_recorded_after_population(ledger,report,manifest=manifest,chain_rows=report["native_chains"],
            contexts=after_contexts,recipe=native_after_recipe,auditor_directory=auditor_directory,auditor_pid=auditor_pid,
            expected_outer_terminal=expected_outer_terminal,allow_synthetic=allow_synthetic)
        external_guard()
    require(digest(report)==frozen and _inventory([ledger.directory])==inventory,
        "original workflow evidence changed during completed readback")
    answer=dict(status="DIAGNOSTIC_CHILD_OWNERS_REVALIDATED",fixed_slots=len(manifest["slots"]),native_chain_rows=len(native_slots),
        completed_fixed_jobs=len(validated),materializer_jobs=total_materializers,replay_jobs=total_replays,
        closed_source_archive_reconstructed=False,full_world_equivalence=False,runtime_qualified=False,
        resource_cost_complete=False,scientific_measurement_validated=False,holdout_authorized=False)
    if has_after:answer["native_after_readback"]=after_readback
    if native_chain_recipe.paired_continuation_labels:
        answer["paired_value_readback"]=dict(label_jobs=total_label_jobs,known_required_cells=total_label_cells,
            complete_labels=total_complete_labels,uncertain_labels=total_label_cells-total_complete_labels,
            unknown_native_slots=sum(slot["status"]!="DIAGNOSTIC_COMPLETE" for slot in native_slots),
            scoped_diagnostics_revalidated=True,labels_used_for_selection=False,
            native_prediction_input_parity=False,runtime_qualified=False)
    if any(reference_value_recipe(recipe) is not None for recipe in diagnostic_recipes.values()):
        summaries=[reader.last_validation["reference_value_labels"] for _,reader,_ in validated.values()
            if "reference_value_labels" in reader.last_validation]
        answer["reference_value_readback"]=dict(completed_populations=len(summaries),
            labels=sum(row["labels"] for row in summaries),complete_labels=sum(row["complete_labels"] for row in summaries),
            uncertain_labels=sum(row["uncertain_labels"] for row in summaries),
            terminal_populations=sum(row["independent_terminal_labels"] for row in summaries),
            labels_used_for_selection=False,runtime_qualified=False,resource_cost_complete=False)
    return answer
