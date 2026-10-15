"""Explicitly admitted one-shot exploration entry; no allocation or holdout.

External JSON is data, never a pickle or dynamic class import. An existing v3
ledger binds the complete prospective recipes and separate entry review.
Private archives stay inside one externally owned auditor. Successful staging
does not establish direct checks, complete costs or scientific improvement.
"""
from copy import deepcopy
from dataclasses import asdict, dataclass, fields
import hashlib
import json
import math
import os
from pathlib import Path
import pickle
import sys
import time
import threading

from .search_over_raw import SearchConfiguration, digest, require
from .search_over_raw_continuation_admission import _path, _fresh_review, initialize_fresh_namespace
from .search_over_raw_ledger import _write_new
from .search_over_raw_reconciled import (ReconciledPhaseALedger, controller_input_hashes,
    a1_configurations, a2_configurations, deployment_configurations, selector_runtime_recipe, configuration_review_template,
    HOLDOUT_ONLY_GATE)
from .search_over_raw_source_run import (collect_owned_exploration_panel, source_recipe,
    source_review_template, read_source_owner)
from .search_over_raw_stage_run import (StageSelectorRecipe, ProspectivelyUnavailableStageRecipe,
    exploration_recipe_keys, run_owned_exploration_stages, validate_unavailable_recipe,
    validate_shared_availability, require_unavailable_candidate_absence)
from .search_over_raw_world_inputs import _state_payload
from .search_over_raw_outcome_run import verify_outcome_runtime
from .search_over_raw_raw_run import raw_runtime
from .search_over_raw_supervision import _supervise_owned_job, _verify_closed_generations
from .search_over_raw_nested_ownership import verify_delegated_closure, current_scope
from .search_over_raw_resources import (read_completed_process_cost, validate_memory_receipt,
    measurement_protocol, resource_source_hash)

SCHEMA="pokezero.search-over-raw.exploration-entry.v1"
WORKFLOW_SCHEMA="pokezero.search-over-raw.exploration-entry.v2"
ACTOR_WORKFLOW_SCHEMA="pokezero.search-over-raw.exploration-entry.v3"
AFTER_WORKFLOW_SCHEMA="pokezero.search-over-raw.exploration-entry.v4"
REFERENCE_VALUE_WORKFLOW_SCHEMA="pokezero.search-over-raw.exploration-entry.v5"
PAIRED_VALUE_WORKFLOW_SCHEMA="pokezero.search-over-raw.exploration-entry.v6"
AFTER_WORKFLOW_SCHEMAS=(AFTER_WORKFLOW_SCHEMA,REFERENCE_VALUE_WORKFLOW_SCHEMA)
WORKFLOW_SCHEMAS=(WORKFLOW_SCHEMA,ACTOR_WORKFLOW_SCHEMA,*AFTER_WORKFLOW_SCHEMAS,PAIRED_VALUE_WORKFLOW_SCHEMA)
REVIEW_SCHEMA="pokezero.search-over-raw.exploration-entry-review.v1"


def _entry_diagnostic_recipe(schema,recipe):
    from .search_over_raw_diagnostic_producer import FixedWorkRecipe,ReferenceValueFixedWorkRecipe,reference_value_recipe
    require(type(recipe) is FixedWorkRecipe or schema in (REFERENCE_VALUE_WORKFLOW_SCHEMA,PAIRED_VALUE_WORKFLOW_SCHEMA)
        and type(recipe) is ReferenceValueFixedWorkRecipe,
        "typed prospectively versioned fixed-work recipe required")
    return reference_value_recipe(recipe)


def _capture_actor_history(spec):
    """An explicit prospective v3 recipe option, never a command override."""
    if spec.get("schema") in (ACTOR_WORKFLOW_SCHEMA,*AFTER_WORKFLOW_SCHEMAS,PAIRED_VALUE_WORKFLOW_SCHEMA):
        require(type(spec.get("capture_actor_history")) is bool,
            "exact prospective actor capture boolean required")
        return spec["capture_actor_history"]
    require("capture_actor_history" not in spec,
        "historical entry cannot acquire actor capture")
    return False


def _source_recipe(spec,config,checkpoint):
    return source_recipe(config,checkpoint,spec["source_runtime"],
        capture_actor_history=_capture_actor_history(spec))


def _collect_entry_panel(ledger,spec,prepared):
    """Forward only the capture option that passed the prospective review."""
    capture = _capture_actor_history(spec)
    require(type(prepared.get("capture_actor_history")) is bool
        and prepared["capture_actor_history"] is capture,
        "actor capture differs from prepared source recipe")
    return collect_owned_exploration_panel(ledger,stage="A1",configuration=a1_configurations()[0].identity,
        config=prepared["source_config"],checkpoint_contract=prepared["checkpoint"],
        bridge_runtime=spec["source_runtime"],source_review_path=spec["source_review_path"],
        history_review_path=spec["history_review_path"],capture_actor_history=capture)


def _json(raw):
    def pairs(rows):
        require(len({k for k,_ in rows}) == len(rows),"duplicate JSON field")
        return dict(rows)
    def invalid(value):raise ValueError("nonfinite JSON value")
    return json.loads(raw,object_pairs_hook=pairs,parse_constant=invalid)


def _decode(value):
    """Small closed typed data codec; no code imports selected by input."""
    from ..local_showdown import LocalShowdownConfig
    from ..observation import ObservationSpec, ObservationFeatureMasks
    from ..category_vocab import CategoryVocabulary
    from .resolver import CheckpointContract
    from .paper_reference_runtime import ShowdownWorkerFactory
    from .search_over_raw_ancestry import NativeAncestryContract
    from .search_over_raw_diagnostic_producer import FixedWorkRecipe,ReferenceValueFixedWorkRecipe
    from .search_over_raw_reference_values import ReferenceValueLabelRecipe
    from .search_over_raw_exploration_workflow import NativeChainRecipe
    from .search_over_raw_after_population import NativeAfterRecipe
    allowed={cls.__module__+"."+cls.__qualname__:cls for cls in
        (LocalShowdownConfig,ObservationSpec,ObservationFeatureMasks,CategoryVocabulary,
         CheckpointContract,ShowdownWorkerFactory,NativeAncestryContract,FixedWorkRecipe,ReferenceValueFixedWorkRecipe,
         ReferenceValueLabelRecipe,NativeChainRecipe,NativeAfterRecipe)}
    if value is None or type(value) in (str,int,float,bool):
        require(type(value) is not float or math.isfinite(value),"nonfinite typed data value")
        return value
    require(type(value) is dict,"typed data object required")
    if set(value)=={"path"}:
        require(type(value["path"]) is str,"path data must be text")
        return Path(value["path"])
    if set(value)=={"mapping"}:
        rows=value["mapping"]
        require(type(rows) is list and all(type(r) is list and len(r)==2 for r in rows),"mapping data malformed")
        keys=[_decode(r[0]) for r in rows]
        require(len(set(keys))==len(keys),"duplicate mapping key")
        return {k:_decode(r[1]) for k,r in zip(keys,rows)}
    if set(value)=={"container","items"}:
        require(value["container"] in {"list","tuple","set","frozenset"} and type(value["items"]) is list,
            "container data malformed")
        return {"list":list,"tuple":tuple,"set":set,"frozenset":frozenset}[value["container"]](
            _decode(x) for x in value["items"])
    require(set(value)=={"dataclass","fields"} and value["dataclass"] in allowed,
        "unknown typed recipe class; no dynamic imports")
    cls=allowed[value["dataclass"]]
    require(type(value["fields"]) is dict and set(value["fields"])=={f.name for f in fields(cls)},
        "exact typed recipe fields required")
    result=cls(**{k:_decode(v) for k,v in value["fields"].items()})
    require(_state_payload(result)==value,"typed recipe did not round trip")
    return result


def _core(contract):
    return {k:v for k,v in contract.items() if k not in
        {"input_hashes","execution_ready","collection_authorized","remaining_gates"}}


def _selector_recipe_law(cfg,runtime,factory,checkpoint):
    """No-load prospective runtime law; future oracle truth is not invented."""
    require(type(runtime) is dict and not {"team_oracle","oracle_payload_sha256"}&set(runtime),
        "future root truth cannot enter entry recipes")
    if cfg.arm=="reference":
        from .paper_reference_runtime import ShowdownWorkerFactory
        from .search_over_raw_reference_timed_run import _runtime, _runtime_law
        from .search_over_raw_world_inputs import TIMED_REFERENCE_PUBLIC_CAPTURE
        if cfg.belief=="oracle":
            from .search_over_raw_reference_oracle_run import _configuration, REFERENCE_ORACLE_CAPTURE
            _runtime_law(cfg,runtime,digest(runtime),configuration_check=_configuration)
            require(digest(runtime.get("reference_team_oracle_observation"))==digest(REFERENCE_ORACLE_CAPTURE)
                and type(runtime.get("oracle_reconstruction_catalog_sha256")) is str
                and len(runtime["oracle_reconstruction_catalog_sha256"])==64
                and all(c in "0123456789abcdef" for c in runtime["oracle_reconstruction_catalog_sha256"]),
                "prospective reference oracle observation/catalog required")
        else:_runtime(cfg,runtime,digest(runtime))
        require(type(factory) is ShowdownWorkerFactory
            and digest(asdict(factory))==digest(runtime.get("reference_factory"))
            and (factory.checkpoint,factory.checkpoint_sha256,factory.showdown_root,factory.set_source_hash)
                ==(checkpoint.checkpoint_path,checkpoint.checkpoint_sha256,checkpoint.showdown_root,
                   checkpoint.showdown_source_sha256)
            and str(_path(factory.checkpoint))==factory.checkpoint
            and str(_path(factory.showdown_root))==factory.showdown_root,
            "actual reference factory differs from reviewed runtime/champion/simulator")
        require(digest(runtime.get("full_public_request_observation"))==digest(TIMED_REFERENCE_PUBLIC_CAPTURE),
            "full reference public request observation required")
    else:
        from .search_over_raw_timed_run import _runtime, _runtime_law
        from .search_over_raw_world_inputs import TIMED_NATIVE_PUBLIC_CAPTURE
        require(factory is None,"native entry cannot carry reference factory")
        if cfg.belief=="oracle":
            from .search_over_raw_native_oracle_run import _runtime as oracle_runtime
            oracle_runtime(cfg,runtime,digest(runtime))
        else:
            from .search_over_raw_actor_timed_run import RUNTIME_KEY,_actor_runtime
            (_actor_runtime if RUNTIME_KEY in runtime else _runtime)(cfg,runtime,digest(runtime))
        require(digest(runtime.get("full_public_request_observation"))==digest(TIMED_NATIVE_PUBLIC_CAPTURE),
            "full native public request observation required")


def _selector_files(files,checkpoint):
    """Existing owning-reader inventory law, checked BEFORE any source work."""
    from .search_over_raw_fixed_work_run import fixed_work_reader_source_paths
    from .search_over_raw_outcomes import _sha
    required=fixed_work_reader_source_paths()
    dist=_path(Path(checkpoint.showdown_root)/"dist")
    require((dist/"sim/index.js").is_file() and (dist/"sim/state.js").is_file(),
        "built selector simulator binding required")
    required.update(str(_path(p)) for p in dist.rglob("*") if p.is_file())
    required.update((checkpoint.checkpoint_path,str(Path(sys.executable).resolve())))
    exports={checkpoint.checkpoint_path:checkpoint.checkpoint_sha256}
    for path,sha in ((checkpoint.model_path,checkpoint.model_sha256),(checkpoint.tables_path,checkpoint.tables_sha256)):
        require(path is None and sha is None or type(path) is str and _sha(sha),"exported selector model/table binding malformed")
        if path is not None:required.add(path);exports[path]=sha
    require(type(files) is dict and required<=set(files) and exports.items()<=files.items()
        and all(type(p) is str and str(_path(p))==p and _sha(sha)
            and hashlib.sha256(Path(p).read_bytes()).hexdigest()==sha for p,sha in files.items()),
        "complete current selector file inventory/champion bindings required")


@dataclass(frozen=True)
class ExplorationAuditorContract:
    emergency_watchdog_seconds: float
    def __post_init__(self):
        require(type(self.emergency_watchdog_seconds) in (int,float)
            and math.isfinite(self.emergency_watchdog_seconds) and self.emergency_watchdog_seconds>0,
            "explicit positive whole-auditor watchdog required")
    def receipt(self):
        return dict(schema="pokezero.search-over-raw.exploration-auditor-contract.v1",
            emergency_watchdog_seconds=self.emergency_watchdog_seconds,
            scope="whole exploration auditor including startup, sources, nested work and cleanup",
            per_decision_ceiling_changed=False,retries_permitted=False,
            holdout_authorized=False,phase_b_authorized=False,resource_cost_complete=False)


def entry_review_template(contract,spec):
    """Prospective data review; does not clear ledger admission or run work."""
    outer=ExplorationAuditorContract(spec["emergency_watchdog_seconds"])
    return dict(schema=REVIEW_SCHEMA,status="UNREVIEWED_EXPLORATION_ENTRY",
        contract_recipe_sha256=digest(_core(contract)),entry_spec_sha256=digest(spec),
        controller_input_hashes=controller_input_hashes(),outer_contract=outer.receipt(),
        reviewed_before_first_source_work=False,complete_recipe_registry_reviewed=False,
        whole_auditor_lifecycle_reviewed=False,collection_authority_from_ledger_only=True,
        holdout_authorized=False,phase_b_authorized=False,retry_authorized=False)


def _prepared(ledger,spec):
    from ..local_showdown import LocalShowdownConfig
    from .resolver import CheckpointContract
    from .paper_reference_runtime import ShowdownWorkerFactory
    require(type(spec) is dict and spec.get("schema") in (SCHEMA,*WORKFLOW_SCHEMAS)
        and set(spec)=={"schema","ledger_directory","contract_recipe_sha256",
        "review_path","source_review_path","history_review_path","emergency_watchdog_seconds",
        "source_config","checkpoint_contract","source_runtime","selector_files","raw_runtime",
        "continuation_config","continuation_runtime","recipes"}
            | ({"diagnostics"} if spec.get("schema") in WORKFLOW_SCHEMAS else set())
            | ({"capture_actor_history"} if spec.get("schema") in (ACTOR_WORKFLOW_SCHEMA,*AFTER_WORKFLOW_SCHEMAS,PAIRED_VALUE_WORKFLOW_SCHEMA) else set()),
        "exact data-only exploration entry specification required")
    require(spec["ledger_directory"]==str(ledger.directory)
        and spec["contract_recipe_sha256"]==digest(_core(ledger.contract)),"entry ledger/roster recipe drift")
    source_config,checkpoint,continuation_config=(_decode(spec[k]) for k in
        ("source_config","checkpoint_contract","continuation_config"))
    require(type(source_config) is LocalShowdownConfig and type(continuation_config) is LocalShowdownConfig
        and type(checkpoint) is CheckpointContract,"exact source/checkpoint/continuation recipes required")
    _source_recipe(spec,source_config,checkpoint)
    verify_outcome_runtime(spec["continuation_runtime"],continuation_config)
    require(spec["continuation_runtime"]["files"].get(spec["continuation_runtime"]["checkpoint_path"])
        ==checkpoint.checkpoint_sha256 and str(continuation_config.resolved_showdown_root())==checkpoint.showdown_root,
        "continuations must retain the source champion and simulator")
    require(digest(spec["raw_runtime"])==digest(raw_runtime(SearchConfiguration("raw",seconds=1.),
        checkpoint.checkpoint_sha256,checkpoint.showdown_source_sha256)),"exact shared raw champion recipe required")
    for key in ("review_path","source_review_path","history_review_path"):
        require(type(spec[key]) is str and str(_path(spec[key]))==spec[key],"canonical review path required")
    _selector_files(spec["selector_files"],checkpoint)
    recipes={};configs={c.identity:c for c in a1_configurations()+a2_configurations()+deployment_configurations()}
    require(type(spec["recipes"]) is list,"explicit prospective recipe registry required")
    for row in spec["recipes"]:
        require(type(row) is dict and set(row) in (
            {"stage","configuration","runtime","review_path","reference_factory"},
            {"stage","configuration","unavailable_reason","review_path"}),
            "exact selector recipe fields required")
        key=(row["stage"],row["configuration"])
        require(key in exploration_recipe_keys() and key not in recipes,"unknown/duplicate prospective recipe")
        if "unavailable_reason" in row:
            recipe=ProspectivelyUnavailableStageRecipe(reason=row["unavailable_reason"],review_path=row["review_path"])
            validate_unavailable_recipe(ledger,key[0],key[1],recipe)
            recipes[key]=recipe
            continue
        cfg=configs[key[1]];factory=_decode(row["reference_factory"])
        require(row["runtime"].get("checkpoint_sha256")==checkpoint.checkpoint_sha256
            and row["runtime"].get("set_source_hash")==checkpoint.showdown_source_sha256,
            "selector champion/catalog differs from source")
        require(factory is None if cfg.arm=="incumbent" else type(factory) is ShowdownWorkerFactory,
            "exact reference factory recipe required")
        _selector_recipe_law(cfg,row["runtime"],factory,checkpoint)
        require(str(_path(row["review_path"]))==row["review_path"],"canonical configuration review path required")
        path=_path(row["review_path"]);raw=path.read_bytes();review=_json(raw)
        require(ledger.contract["input_hashes"].get(str(path))==hashlib.sha256(raw).hexdigest(),"configuration review unbound")
        # Candidate A3 recipes are prospective data, not A3 eligibility. Do
        # not ask the ledger for A3 configurations before the actual A2 freeze.
        expected=configuration_review_template(ledger.contract,stage=key[0],configuration=key[1],
            raw_runtime_sha256=review.get("raw_runtime_sha256"),selector_recipe_sha256=review.get("selector_recipe_sha256"),
            qualification_evidence_hashes=review.get("qualification_evidence_hashes"),cost_scope=review.get("cost_scope"))
        expected.update(status="CLEAR_CONFIGURATION_RECIPE",reviewed_before_first_source_work=True,
            atomic_work_and_cleanup_validated=True,resource_cost_qualification_reviewed=True)
        require(digest(review)==digest(expected) and all(ledger.contract["input_hashes"].get(p)==sha
            for p,sha in {**review["controller_input_hashes"],**review["qualification_evidence_hashes"]}.items()),
            "exact prospective configuration/code/qualification review required")
        require(not {"team_oracle","oracle_payload_sha256"}&set(row["runtime"]),"future root truth cannot enter entry recipes")
        require(review["raw_runtime_sha256"]==digest(spec["raw_runtime"])
            and review["selector_recipe_sha256"]==digest(selector_runtime_recipe(cfg,row["runtime"])),
            "prospective selector recipe differs from review")
        recipes[key]=StageSelectorRecipe(runtime=deepcopy(row["runtime"]),
            qualification_review_path=row["review_path"],reference_factory=factory)
    require(set(recipes)==set(exploration_recipe_keys()),"complete A1/A2 and candidate A3 recipes required")
    validate_shared_availability(recipes)
    history,_=_fresh_review(ledger,spec["history_review_path"])
    require(history["raw_runtime_sha256"]==digest(spec["raw_runtime"])
        and history["continuation_runtime_sha256"]==digest(spec["continuation_runtime"]),"fresh history runtime drift")
    path=_path(spec["source_review_path"]);raw=path.read_bytes();review=_json(raw)
    require(ledger.contract["input_hashes"].get(str(path))==hashlib.sha256(raw).hexdigest(),"prospective source review unbound")
    expected=source_review_template(ledger.contract,_source_recipe(spec,source_config,checkpoint),
        qualification_evidence_hashes=review["qualification_evidence_hashes"])
    expected.update(status="CLEAR_SOURCE_RECIPE",reviewed_before_first_source_work=True,
        source_runtime_and_cleanup_reviewed=True,source_policy_and_terminal_reviewed=True)
    require(digest(review)==digest(expected) and all(ledger.contract["input_hashes"].get(p)==sha
        for p,sha in {**review["controller_input_hashes"],**review["qualification_evidence_hashes"]}.items()),
        "exact source recipe/code/qualification review required")
    prepared=dict(source_config=source_config,checkpoint=checkpoint,continuation_config=continuation_config,recipes=recipes,
        capture_actor_history=_capture_actor_history(spec))
    if spec["schema"] in WORKFLOW_SCHEMAS:
        from .search_over_raw_diagnostic_producer import (FixedWorkRecipe,ReferenceValueFixedWorkRecipe,
            diagnostic_configurations,fixed_work_runtime_recipe,reference_value_recipe)
        from .search_over_raw_exploration_workflow import NativeChainRecipe,preflight_diagnostic_workflow
        d=spec["diagnostics"]
        require(type(d) is dict and set(d)=={"recipes","native_chain_recipe","review_path"}
            | ({"native_after_recipe"} if spec["schema"] in AFTER_WORKFLOW_SCHEMAS else set())
            and type(d["recipes"]) is list,"exact prospective diagnostic registry required")
        fixed={};cfgs={c.identity:c for c in diagnostic_configurations()}
        for row in d["recipes"]:
            require(type(row) is dict and set(row)=={"configuration","recipe"}
                and row["configuration"] in cfgs and row["configuration"] not in fixed,"unknown/duplicate diagnostic recipe")
            recipe=_decode(row["recipe"]);cfg=cfgs[row["configuration"]]
            labels=_entry_diagnostic_recipe(spec["schema"],recipe)
            if recipe.unqualified_reason is None:
                expected_runtime=fixed_work_runtime_recipe(cfg,checkpoint,reference_factory=recipe.reference_factory,
                    native_ancestry_contract=recipe.native_ancestry_contract,
                    capture_native_public_fold=recipe.capture_native_public_fold,
                    owned_reference_value_recipe=labels)
                require(digest(recipe.runtime)==digest(expected_runtime),"complete fixed-work diagnostic runtime recipe drift")
            fixed[row["configuration"]]=recipe
        chain=_decode(d["native_chain_recipe"])
        require(type(chain) is NativeChainRecipe,"typed native-chain recipe required")
        require(chain.paired_continuation_labels is (spec["schema"]==PAIRED_VALUE_WORKFLOW_SCHEMA),
            "paired native labels require their explicit prospective v6 entry")
        preflight_diagnostic_workflow(ledger,recipes=fixed,chain_recipe=chain,checkpoint_contract=checkpoint,
            showdown_root=checkpoint.showdown_root,file_hashes=spec["selector_files"],review_path=d["review_path"])
        prepared.update(diagnostic_recipes=fixed,native_chain_recipe=chain,diagnostic_review_path=d["review_path"],
            diagnostic_file_hashes=spec["selector_files"])
        if spec["schema"] in AFTER_WORKFLOW_SCHEMAS:
            from .search_over_raw_after_population import NativeAfterRecipe,preflight_after_population
            after=_decode(d["native_after_recipe"])
            require(type(after) is NativeAfterRecipe,"typed prospective AFTER recipe required")
            if after.unqualified_reason is None:
                require(digest(_state_payload(after.config))==digest(_state_payload(source_config)),
                    "AFTER history encoder must retain the original source configuration")
            preflight_after_population(ledger,after,checkpoint_contract=checkpoint)
            prepared["native_after_recipe"]=after
    return prepared


def _entry_inputs(ledger_directory,spec_path):
    """Original immutable authority/input checks, valid before AND after work."""
    ledger=ReconciledPhaseALedger(_path(ledger_directory))
    require(ledger.contract["execution_ready"] is True and ledger.contract["collection_authorized"] is True
        and set(ledger.contract["remaining_gates"])<={HOLDOUT_ONLY_GATE},
        "exploration source/runtime/collection admission is incomplete")
    ledger.verify()
    require(all(ledger.contract["input_hashes"].get(p)==sha for p,sha in controller_input_hashes().items()),
        "current controller code must be bound before exploration execution")
    spec_path=_path(spec_path);raw=spec_path.read_bytes();spec=_json(raw)
    require(ledger.contract["input_hashes"].get(str(spec_path))==hashlib.sha256(raw).hexdigest(),"entry spec must be prospectively bound")
    prepared=_prepared(ledger,spec)
    review_path=_path(spec["review_path"]);review_bytes=review_path.read_bytes();review=_json(review_bytes)
    require(ledger.contract["input_hashes"].get(str(review_path))==hashlib.sha256(review_bytes).hexdigest(),"entry review unbound")
    expected=entry_review_template(ledger.contract,spec)
    expected.update(status="CLEAR_EXPLORATION_ENTRY",reviewed_before_first_source_work=True,
        complete_recipe_registry_reviewed=True,whole_auditor_lifecycle_reviewed=True)
    require(digest(review)==digest(expected),"whole-auditor entry review required")
    ledger.verify()
    require(spec_path.read_bytes()==raw and review_path.read_bytes()==review_bytes,"entry inputs changed during preflight")
    return ExplorationAuditorJob(str(ledger.directory),str(spec_path),hashlib.sha256(raw).hexdigest(),
        ledger.contract_sha256,hashlib.sha256(review_bytes).hexdigest()),prepared,ledger,spec


def _original_claim(job,spec):
    directory=_path(Path(job.ledger_directory)/"exploration-entry-run")
    claim=_json((directory/"claim.json").read_bytes())
    payload=pickle.dumps(job)
    require(set(claim)=={"job_sha256","binding_sha256","contract","retry_authorized",
        "synthetic_process_fixture","resource_measurement_protocol","resource_measurement_source_sha256"}
        | ({"attempt_declaration"} if "attempt_declaration" in claim else set())
        and claim["job_sha256"]==hashlib.sha256(payload).hexdigest()
        ==hashlib.sha256((directory/"job.pickle").read_bytes()).hexdigest()
        and claim["binding_sha256"]==digest(job.__dict__)
        and claim["contract"]==ExplorationAuditorContract(spec["emergency_watchdog_seconds"]).receipt()
        and digest(claim["resource_measurement_protocol"])==digest(measurement_protocol())
        and claim["resource_measurement_source_sha256"]==resource_source_hash()
        and claim["retry_authorized"] is False and claim["synthetic_process_fixture"] is False,
        "original production entry claim required")
    if "attempt_declaration" in claim:
        from .search_over_raw_attempt_journal import _permission_handle,_declaration,journal_metadata
        handle,path=_permission_handle(claim["attempt_declaration"]);_,outer=journal_metadata(handle)
        _,declared=_declaration(handle,path)
        require(handle.directory==str(_path(Path(job.ledger_directory)/"attempt-cost-journal"))
            and digest(asdict(outer))==digest(asdict(declared))==digest(asdict(exploration_attempt_cost_spec(job))),
            "original outer entry declaration binding drift")


def prepare_exploration_entry(ledger_directory,spec_path,*,_owned_job=None,_attempt_journal=None):
    """Read-only fresh admission before claims, model loads, or source spawn."""
    job,prepared,ledger,spec=_entry_inputs(ledger_directory,spec_path)
    ledger.verify_exploration("A1",a1_configurations()[0].identity)
    run_directory=_path(ledger.directory/"exploration-entry-run")
    journal_directory=_path(ledger.directory/"attempt-cost-journal")
    if _attempt_journal is not None:
        from .search_over_raw_attempt_journal import current_attempt_journal,journal_metadata
        writer=current_attempt_journal();_,outer=journal_metadata(_attempt_journal,producer=True)
        require(_owned_job is None and writer is not None and writer.permission is None
            and writer.handle==_attempt_journal and writer.handle.directory==str(journal_directory)
            and digest(asdict(outer))==digest(asdict(exploration_attempt_cost_spec(job))),
            "original current entry journal context required")
    elif _owned_job is None:require(not journal_directory.exists(),"exploration entry journal already occupied; no resumption")
    if _owned_job is None:require(not run_directory.exists(),"exploration entry already occupied; no resumption")
    else:
        scope=current_scope()
        require(type(_owned_job) is ExplorationAuditorJob and scope is not None
            and scope.directory==run_directory and scope.binding==digest(_owned_job.__dict__),"original owned entry scope required")
        require(job==_owned_job,"original owned entry inputs changed")
        _original_claim(job,spec)
    for name in ("continuation-admission","shared-raw-runs","source-attempts","staged-run-claim.json",
                 "a2-freeze.json","selection.json"):
        require(not _path(ledger.directory/name).exists(),"exploration entry already occupied; no resumption")
    require(all(not any((ledger.directory/stage).iterdir()) for stage in ("A1","A2","A3")),"stage work already exists")
    ledger.verify()
    return job,prepared


@dataclass(frozen=True)
class ExplorationAuditorJob:
    ledger_directory: str
    spec_path: str
    spec_sha256: str
    contract_sha256: str
    review_sha256: str
    def __call__(self,ownership,phase):
        phase("startup")
        job,prepared=prepare_exploration_entry(self.ledger_directory,self.spec_path,_owned_job=self)
        require(job==self,"original exploration job inputs changed")
        ledger=ReconciledPhaseALedger(self.ledger_directory);spec=_json(Path(self.spec_path).read_bytes())
        panel=None
        try:
            initialize_fresh_namespace(ledger,history_review_path=spec["history_review_path"])
            phase("selection")
            from ..randbat import load_gen3_randbat_source_cached
            catalog=load_gen3_randbat_source_cached(prepared["checkpoint"].showdown_root)
            require(catalog.metadata.source_hash==prepared["checkpoint"].showdown_source_sha256,"entry source catalog drift")
            panel=_collect_entry_panel(ledger,spec,prepared)
            runner=run_owned_exploration_stages
            extra={}
            if spec["schema"] in WORKFLOW_SCHEMAS:
                from .search_over_raw_exploration_workflow import run_owned_exploration_workflow
                runner=run_owned_exploration_workflow
                extra={k:prepared[k] for k in ("diagnostic_recipes","native_chain_recipe","diagnostic_review_path","diagnostic_file_hashes")}
                if spec["schema"] in AFTER_WORKFLOW_SCHEMAS:extra["native_after_recipe"]=prepared["native_after_recipe"]
            result=runner(ledger,panel,recipes=prepared["recipes"],
                checkpoint_contract=prepared["checkpoint"],showdown_root=prepared["checkpoint"].showdown_root,
                file_hashes=spec["selector_files"],raw_runtime=spec["raw_runtime"],history_review_path=spec["history_review_path"],
                continuation_config=prepared["continuation_config"],continuation_runtime=spec["continuation_runtime"],set_source=catalog,
                source_config=prepared["source_config"],source_review_path=spec["source_review_path"],**extra)
            ledger.verify()
            require(hashlib.sha256(Path(self.spec_path).read_bytes()).hexdigest()==self.spec_sha256
                and hashlib.sha256(Path(spec["review_path"]).read_bytes()).hexdigest()==self.review_sha256
                and ledger.contract_sha256==self.contract_sha256,"entry inputs changed during staged work")
            return result
        except BaseException as error:
            if not (ledger.directory/"operational-stop.json").exists():
                try:ledger.stop(type(error).__name__)
                except BaseException as stop_error:error.add_note("entry stop receipt failed: "+type(stop_error).__name__)
            raise
        finally:
            phase("cleanup")
            if panel is not None:
                for owner in panel.owners:owner.close()


def exploration_attempt_cost_spec(job):
    """Freeze the original outer cost descriptor before launch, not admission."""
    from .search_over_raw_attempt_cost import AttemptCostSpec
    require(type(job) is ExplorationAuditorJob,"original typed exploration cost job required")
    raw=_path(job.spec_path).read_bytes()
    require(hashlib.sha256(raw).hexdigest()==job.spec_sha256,"original entry cost spec drift")
    spec=_json(raw);contract=ExplorationAuditorContract(spec["emergency_watchdog_seconds"])
    require(spec["ledger_directory"]==job.ledger_directory,"original entry cost namespace drift")
    return AttemptCostSpec(str(_path(Path(job.ledger_directory)/"exploration-entry-run")),
        hashlib.sha256(pickle.dumps(job)).hexdigest(),digest(job.__dict__),"staged-exploration-run",
        "selection",contract.receipt(),resource_source_hash(),contract.emergency_watchdog_seconds)


def read_exploration_attempt_cost(job,attempts,*,roster_sha256,expected_terminal=None):
    """Read supplied original attempts even after the scientific ledger stops.

    This is not completed-work validation, fresh admission, or proof that the
    supplied roster exhausts every dynamically constructed or ordinary actor.
    No attempt is recovered from discovered successful receipt directories.
    """
    from .search_over_raw_attempt_cost import AttemptCostSpec,reconcile_attempt_cost_roster
    require(type(job) is ExplorationAuditorJob and type(attempts) is tuple
        and attempts and all(type(row) is AttemptCostSpec for row in attempts),
        "original entry job and supplied typed cost roster required")
    job_sha=digest(asdict(job));terminal_sha=None if expected_terminal is None else digest(expected_terminal)
    root=_path(job.ledger_directory);spec_path=_path(job.spec_path)
    raw=spec_path.read_bytes();spec=_json(raw);review_path=_path(spec["review_path"])
    envelope_path=root/"contract.json";envelope_raw=envelope_path.read_bytes();envelope=_json(envelope_raw)
    review_raw=review_path.read_bytes()
    require(hashlib.sha256(raw).hexdigest()==job.spec_sha256
        and hashlib.sha256(review_raw).hexdigest()==job.review_sha256
        and set(envelope)=={"contract","contract_sha256"}
        and digest(envelope["contract"])==envelope["contract_sha256"]==job.contract_sha256
        and spec["ledger_directory"]==job.ledger_directory,
        "original entry cost inputs/contract drift")
    outer_directory=str(_path(root/"exploration-entry-run"))
    outer=[row for row in attempts if row.directory==outer_directory]
    require(len(outer)==1 and outer[0].parent_directory is None
        and all(row.directory==outer_directory or row.parent_directory is not None for row in attempts),
        "supplied entry roster must retain its original outer owner")
    contract=ExplorationAuditorContract(spec["emergency_watchdog_seconds"])
    require(outer[0].job_sha256==hashlib.sha256(pickle.dumps(job)).hexdigest()
        and outer[0].binding_sha256==digest(job.__dict__) and outer[0].kind=="staged-exploration-run"
        and outer[0].work_phase=="selection" and digest(outer[0].contract)==digest(contract.receipt())
        and outer[0].effective_watchdog_seconds==contract.emergency_watchdog_seconds,
        "supplied cost roster lost original outer job/watchdog binding")
    result=reconcile_attempt_cost_roster(attempts,attempt_root=root,roster_sha256=roster_sha256)
    if expected_terminal is not None:
        require(terminal_sha==digest(_json((Path(outer_directory)/"terminal.json").read_bytes())),
            "original returned outer terminal differs from cost evidence")
    # Original-byte checks after the callback-bearing reconciliation. Do not
    # call ledger.verify(): a stopped ledger must not hide already spent cost.
    require(job_sha==digest(asdict(job)) and raw==spec_path.read_bytes()
        and review_raw==review_path.read_bytes() and envelope_raw==envelope_path.read_bytes()
        and roster_sha256==digest([asdict(row) for row in attempts])
        and (expected_terminal is None or terminal_sha==digest(expected_terminal)),
        "original entry cost inputs changed during readback")
    result.update(entry_job_binding_sha256=digest(job.__dict__),original_entry_inputs_bound=True,
        source_actor_and_preclaim_costs_exhaustive=False,completed_entry_work_validated=False,
        runtime_qualified=False,phase_b_authorized=False)
    return result


def run_exploration_entry(ledger_directory,spec_path):
    from .search_over_raw_observer_cost import ObserverWindow
    window=ObserverWindow("exploration_entry")
    try:
        with window:
            result=_run_exploration_entry(ledger_directory,spec_path,_observer_window=window)
    except BaseException as primary:
        try:
            primary.observer_cost_measurement=window.snapshot()
            primary.observer_publication_measurement=_publish_entry_observer_window(window)
        except BaseException as failure:
            if hasattr(failure,"observer_publication_measurement"):
                primary.observer_publication_measurement=failure.observer_publication_measurement
            primary.add_note("entry observer evidence failed: "+type(failure).__name__)
        raise
    _publish_entry_observer_window(window)
    return result


def _publish_entry_observer_window(window):
    """Only an existing original admitted journal can receive cost evidence."""
    from .search_over_raw_observer_cost import ObserverWindow,read_observer_window,publication_cost
    from .search_over_raw_attempt_journal import AttemptJournalHandle,journal_metadata
    trace=window.snapshot();original_trace=deepcopy(trace);original_trace_sha=digest(trace);read_observer_window(trace)
    require(type(window) is ObserverWindow and trace["observer_pid"]==os.getpid()
        and trace["observer_thread_id"]==threading.get_ident(),"publication must retain original observer PID/thread")
    binding=trace["bindings"].get("journal_handle")
    if binding is None:return  # Failed admission must create no output namespace.
    handle=AttemptJournalHandle(**binding);path=_path(Path(handle.directory)/"entry-observer-window.json")
    try:
        _,outer=journal_metadata(handle)
        require(trace["bindings"].get("entry_job") is not None
            and outer.binding_sha256==digest(trace["bindings"]["entry_job"]),"original observer entry/journal binding drift")
        returned=trace["bindings"].get("outer_terminal")
        terminal_path=Path(outer.directory)/"terminal.json"
        if returned is not None:
            require(terminal_path.is_file() and digest(_json(terminal_path.read_bytes()))==digest(returned),
                "observer lost original actual outer terminal")
        _write_new(path,trace)
    except BaseException as failure:
        ended_wall=time.perf_counter();ended_cpu=time.thread_time()
        try:
            failure.observer_publication_measurement=publication_cost(original_trace,ended_wall=ended_wall,ended_cpu=ended_cpu,
                trace_path=str(path),trace_bytes_sha256=None,error_type=type(failure).__name__)
        except BaseException as measurement_error:
            failure.add_note("publication failure measurement unavailable: "+type(measurement_error).__name__)
        raise
    # Capture the boundary immediately after the original trace write closes.
    # The supplementary receipt's own validation/hash/write stays excluded.
    ended_wall=time.perf_counter();ended_cpu=time.thread_time()
    raw=path.read_bytes();raw_sha=hashlib.sha256(raw).hexdigest()
    require(original_trace_sha==digest(trace)==digest(window.snapshot())==digest(_json(raw)),
        "published original observer trace changed")
    value=publication_cost(original_trace,ended_wall=ended_wall,ended_cpu=ended_cpu,
        trace_path=str(path),trace_bytes_sha256=raw_sha)
    original_publication=deepcopy(value);publication_sha=digest(value)
    publication_path=_path(Path(handle.directory)/"entry-observer-publication-cost.json")
    try:
        journal_metadata(handle)
        require(path==_path(Path(handle.directory)/"entry-observer-window.json") and raw==path.read_bytes()
            and original_trace_sha==digest(trace)==digest(window.snapshot()),
            "original observer trace changed before publication supplement")
        _write_new(publication_path,value)
        require(publication_path==_path(Path(handle.directory)/"entry-observer-publication-cost.json")
            and publication_sha==digest(value)==digest(_json(publication_path.read_bytes()))
            and original_trace_sha==digest(trace)==digest(window.snapshot()) and raw==path.read_bytes(),
            "original observer publication changed during final write")
    except BaseException as failure:
        failure.observer_publication_measurement=original_publication
        raise
    return original_publication


def _run_exploration_entry(ledger_directory,spec_path,*,_observer_window):
    job,_=prepare_exploration_entry(ledger_directory,spec_path)
    _observer_window.bind("entry_job",asdict(job))
    spec=_json(Path(job.spec_path).read_bytes());contract=ExplorationAuditorContract(spec["emergency_watchdog_seconds"])
    directory=Path(job.ledger_directory)/"exploration-entry-run"
    from .search_over_raw_attempt_journal import create_attempt_journal,outer_attempt_journal_scope
    journal=create_attempt_journal(Path(job.ledger_directory)/"attempt-cost-journal",
        outer_attempt=exploration_attempt_cost_spec(job),attempt_root=job.ledger_directory)
    _observer_window.bind("journal_handle",asdict(journal))
    admission_calls=0
    def admission():
        nonlocal admission_calls
        if admission_calls==0:
            actual,_=prepare_exploration_entry(job.ledger_directory,job.spec_path,_attempt_journal=journal)
        else:
            # This is the same invocation's post-wait4 recheck, NOT a new
            # admission or a resume path. Expected completed outputs exist.
            require(admission_calls==1,"unexpected repeated entry admission callback")
            actual,_,_,actual_spec=_entry_inputs(job.ledger_directory,job.spec_path)
            _original_claim(job,actual_spec)
        require(actual==job,"original entry authority/inputs changed")
        admission_calls+=1
        return True
    # Worker preflight permits only this already-published external run claim;
    # no namespace/source/stage work is resumable.
    with outer_attempt_journal_scope(journal):
        terminal=_supervise_owned_job(job,directory=directory,binding_sha256=digest(job.__dict__),
            verify_admission=admission,contract=contract,kind="staged-exploration-run",work_phase="selection",
            validate_result=lambda result:require(result["status"]=="STAGED_EXPLORATION_RECORDED"
                and result["private_archive_returned"] is False and result["holdout_authorized"] is False,"staged result shape drift"))
        _observer_window.bind("outer_terminal",terminal)
    require(terminal["supervisor_exit_code"]==0,"exploration auditor failed; no retry")
    read_exploration_entry(job,terminal)
    return terminal


def read_entry_observer_cost(job,expected_outer_terminal=None):
    """Original entry/journal/window join; historical workers are never polled."""
    from .search_over_raw_attempt_journal import AttemptJournalHandle,journal_metadata,read_attempt_journal_cost
    from .search_over_raw_observer_cost import read_observer_window,read_publication_cost
    require(type(job) is ExplorationAuditorJob,"original typed observer entry job required")
    directory=_path(Path(job.ledger_directory)/"attempt-cost-journal")
    root=_path(job.ledger_directory)
    before=_evidence_inventory(root);job_sha=digest(asdict(job))
    expected_sha=None if expected_outer_terminal is None else digest(expected_outer_terminal)
    path=directory/"entry-observer-window.json";raw=path.read_bytes();trace=_json(raw)
    summary=read_observer_window(trace)
    publication_path=_path(directory/"entry-observer-publication-cost.json")
    publication_raw=publication_path.read_bytes() if publication_path.exists() else None
    publication=None if publication_raw is None else read_publication_cost(_json(publication_raw),trace,
        trace_path=str(_path(path)),trace_bytes_sha256=hashlib.sha256(raw).hexdigest())
    if publication is not None:require(publication["status"]=="TRACE_PUBLISHED","failed publication cannot enter durable cost subtotal")
    require(trace["label"]=="exploration_entry" and trace["bindings"].get("entry_job")==asdict(job),
        "observer window lost original typed entry")
    handle=AttemptJournalHandle(**trace["bindings"]["journal_handle"]);_,outer=journal_metadata(handle)
    require(handle.directory==str(directory) and outer.binding_sha256==digest(asdict(job)),"observer original journal drift")
    spec_path=_path(job.spec_path);spec_raw=spec_path.read_bytes();spec=_json(spec_raw)
    review_path=_path(spec["review_path"]);review_raw=review_path.read_bytes()
    envelope_path=_path(root/"contract.json");envelope_raw=envelope_path.read_bytes();envelope=_json(envelope_raw)
    require(hashlib.sha256(spec_raw).hexdigest()==job.spec_sha256
        and hashlib.sha256(review_raw).hexdigest()==job.review_sha256
        and set(envelope)=={"contract","contract_sha256"}
        and spec["ledger_directory"]==job.ledger_directory
        and envelope["contract_sha256"]==digest(envelope["contract"])==job.contract_sha256,
        "original observer entry inputs changed")
    contract=ExplorationAuditorContract(spec["emergency_watchdog_seconds"])
    require(outer.directory==str(_path(root/"exploration-entry-run")) and outer.parent_directory is None
        and outer.job_sha256==hashlib.sha256(pickle.dumps(job)).hexdigest()
        and outer.binding_sha256==digest(job.__dict__) and outer.kind=="staged-exploration-run"
        and outer.work_phase=="selection" and digest(outer.contract)==digest(contract.receipt())
        and outer.effective_watchdog_seconds==contract.emergency_watchdog_seconds,
        "observer cost lost original outer job/watchdog binding")
    returned=trace["bindings"].get("outer_terminal")
    if expected_outer_terminal is not None:require(returned is not None and digest(returned)==digest(expected_outer_terminal),
        "observer original returned outer terminal drift")
    costs=read_attempt_journal_cost(handle,expected_outer_terminal=returned)
    root_row=next(row for row in costs["attempts"] if row.get("directory")==outer.directory)
    launch_path=Path(outer.directory)/"launch.json"
    if launch_path.is_file():
        launch=_json(launch_path.read_bytes())
        require(type(launch["supervisor_pid"]) is int and launch["supervisor_pid"]==trace["observer_pid"],
            "observer calling process differs from actual outer parent")
        # A bare reused PID cannot prove distinct child generations. Refuse
        # this cost join as ambiguous, never change scientific admission.
        require(all(row["query_pid"]!=launch["pid"] for row in trace["own_process_queries"]),
            "query/outer PID attribution ambiguous; no disjoint cost subtotal")
    known_job_cpu=root_row.get("job_cpu_seconds_including_reaped_children")
    known=summary["calling_thread_cpu_seconds"]+summary["known_own_query_cpu_seconds"]
    if publication is not None:known+=publication["calling_thread_cpu_seconds"]
    if known_job_cpu is not None:known+=known_job_cpu
    require(raw==path.read_bytes() and job_sha==digest(asdict(job))
        and spec_path==_path(job.spec_path) and spec_raw==spec_path.read_bytes()
        and review_path==_path(spec["review_path"]) and review_raw==review_path.read_bytes()
        and envelope_path==_path(root/"contract.json") and envelope_raw==envelope_path.read_bytes()
        and (expected_outer_terminal is None or expected_sha==digest(expected_outer_terminal))
        and before==_evidence_inventory(_path(job.ledger_directory)),"original observer/attempt evidence changed during final readback")
    return dict(schema="pokezero.search-over-raw.entry-observer-cost.v1",observer_window=summary,
        publication_measurement=publication,publication_usage_unknown=publication is None,
        actual_outer_job_cpu_seconds=known_job_cpu,known_disjoint_cpu_subtotal_seconds=known,
        narrower_supervisor_and_sampler_counters_added_again=False,nested_costs_added_again=False,
        campaign_elapsed_wall_seconds=summary["elapsed_wall_seconds"] if publication is None else
            publication["ended_monotonic_seconds"]-trace["started_monotonic_seconds"],job_wall_exposure_added_again=False,
        final_measurement_validation_and_serialization_included=False,
        all_declared_supervised_attempts_closed=costs["declared_supervised_attempt_census_complete"],
        incomplete_components=(["original observer trace publication usage unknown"] if publication is None else [])+
            ["supplemental measurement receipt validation/serialization/write","non-calling-thread external work",
            "pending or lost query usage","unreaped or unknown job descendants"],
        total_workflow_cpu_seconds=None,complete_workflow_cost=False,resource_cost_complete=False,
        work_validated=False,runtime_qualified=False,scientific_collection_authorized=False,holdout_authorized=False)


def read_exploration_entry(job,expected_terminal):
    """Independent completed owner/ledger join; never polls historical PIDs."""
    require(type(job) is ExplorationAuditorJob,"original typed entry job required")
    ledger=ReconciledPhaseALedger(_path(job.ledger_directory));ledger.verify()
    before=_evidence_inventory(ledger.directory)
    raw=Path(job.spec_path).read_bytes();spec=_json(raw)
    require(hashlib.sha256(raw).hexdigest()==job.spec_sha256 and ledger.contract_sha256==job.contract_sha256
        and hashlib.sha256(Path(spec["review_path"]).read_bytes()).hexdigest()==job.review_sha256,
        "original completed entry bindings changed")
    actual,prepared,_,_=_entry_inputs(job.ledger_directory,job.spec_path)
    require(actual==job,"completed entry lost original authority or review")
    directory=_path(ledger.directory/"exploration-entry-run")
    _original_claim(job,spec)
    claim,launch,worker,terminal=(_json((directory/name).read_bytes()) for name in
        ("claim.json","launch.json","worker-result.json","terminal.json"))
    binding=digest(job.__dict__);contract=ExplorationAuditorContract(spec["emergency_watchdog_seconds"])
    require(claim["job_sha256"]==hashlib.sha256(pickle.dumps(job)).hexdigest()
        ==hashlib.sha256((directory/"job.pickle").read_bytes()).hexdigest()
        and claim["binding_sha256"]==binding and claim["contract"]==contract.receipt()
        and claim["retry_authorized"] is False and claim["synthetic_process_fixture"] is False,
        "original outer job/claim drift")
    pid=launch["pid"]
    require(type(pid) is int and pid>0 and launch["group"]==pid
        and type(launch["supervisor_pid"]) is int and launch["supervisor_pid"]>0 and launch["supervisor_pid"]!=pid
        and type(launch["birth_identity"]) is str and launch["birth_identity"]
        and launch["binding_sha256"]==binding and worker["binding_sha256"]==terminal["binding_sha256"]==binding
        and worker["worker_pid"]==terminal["worker_pid"]==pid,
        "outer launch/worker/wait identity drift")
    require(digest(terminal)==digest(expected_terminal)
        and terminal["schema"]=="pokezero.search-over-raw.staged-exploration-run-supervision.v1"
        and terminal["status"]=="STAGED_EXPLORATION_RUN_COMPLETE"
        and all(type(v) is int and v==0 for v in
            (terminal["supervisor_exit_code"],terminal["worker_exit_code"],worker["exit_code"]))
        and terminal["error_type"] is None and worker["error_type"] is None
        and terminal["cleanup_error_type"] is None and terminal["watchdog_reached"] is False
        and terminal["watchdog_seconds"]==contract.emergency_watchdog_seconds,
        "normal externally reaped exploration auditor required")
    closure=verify_delegated_closure(directory,pid,binding,_verify_closed_generations)
    costs=read_completed_process_cost(terminal,worker,work_phase="selection")
    if terminal["memory_measurement"] is not None:
        validate_memory_receipt(directory,binding=binding,main_launch=launch,expected_summary=terminal["memory_measurement"])
        require(terminal["resource_measurement_error_type"] is None,"conflicting entry RSS failure")
    else:require(type(terminal["resource_measurement_error_type"]) is str and terminal["resource_measurement_error_type"],
        "missing entry RSS measurement must be explicit")
    result=terminal["result"]
    require(digest(worker["result"])==digest(result),"original outer worker/workflow result drift")
    diagnostic_readback=None
    if spec["schema"] in WORKFLOW_SCHEMAS:
        from .search_over_raw_exploration_workflow import read_recorded_workflow
        diagnostic_readback=read_recorded_workflow(ledger,result,diagnostic_recipes=prepared["diagnostic_recipes"],
            native_chain_recipe=prepared["native_chain_recipe"],diagnostic_review_path=prepared["diagnostic_review_path"],
            diagnostic_file_hashes=spec["selector_files"],checkpoint_contract=prepared["checkpoint"],
            auditor_directory=directory,auditor_pid=pid,
            native_after_recipe=prepared.get("native_after_recipe"),expected_outer_terminal=expected_terminal)
        result=result["staged"]
    staged_keys={"status","reports","evaluator_freeze","selection","launches",
        "source_panel","synthetic_process_fixture","elapsed_seconds","auditor_cpu_seconds","resource_cost_complete",
        "cost_scope","private_archive_returned","runtime_qualified_by_this_code","collection_authorized",
        "holdout_authorized","phase_b_authorized"}
    require(type(result) is dict and set(result) in (staged_keys,staged_keys|{"input_unavailable_by_root_configuration"})
        and result["status"]=="STAGED_EXPLORATION_RECORDED"
        and all(result[k] is False for k in
            ("synthetic_process_fixture","resource_cost_complete","private_archive_returned","runtime_qualified_by_this_code",
             "collection_authorized","holdout_authorized","phase_b_authorized")),"exact nonprivate staged result required")
    reports={stage:ledger.report_stage(stage) for stage in ("A1","A2","A3")}
    require(digest(reports)==digest(result["reports"]) and digest(ledger._a2())==digest(result["evaluator_freeze"])
        and digest(ledger._selection())==digest(result["selection"]),"returned summaries differ from durable staged evidence")
    panel=result["source_panel"];seeds=ledger.contract["panels"]["exploration"]["seeds"]
    source_recipe_sha=digest(_source_recipe(spec,prepared["source_config"],prepared["checkpoint"]))
    source_review_sha=hashlib.sha256(Path(spec["source_review_path"]).read_bytes()).hexdigest()
    require(type(panel) is dict and set(panel)=={"schema","contract_sha256","source_count",
        "source_recipe_sha256","source_runtime_sha256","source_review_sha256","source_sha256_by_seed",
        "observed_root_ids","missing_root_ids","source_roster_complete","source_pipeline_validated",
        "runtime_qualified","collection_authorized","holdout_authorized"}
        and panel["schema"]=="pokezero.search-over-raw.retained-exploration-panel.v1"
        and panel["contract_sha256"]==job.contract_sha256
        and type(panel["source_count"]) is int and panel["source_count"]==len(seeds)
        and panel["source_recipe_sha256"]==source_recipe_sha
        and panel["source_runtime_sha256"]==digest(spec["source_runtime"])
        and panel["source_review_sha256"]==source_review_sha
        and panel["source_roster_complete"] is True and panel["source_pipeline_validated"] is True
        and all(panel[k] is False for k in ("runtime_qualified","collection_authorized","holdout_authorized")),
        "returned panel lost original recipe/review/authority scope")
    require({p.name for p in (ledger.directory/"source-attempts").iterdir()}=={str(s) for s in seeds},"completed source roster drift")
    observed=[];missing=[];source_hashes={}
    for seed in seeds:
        source_directory=_path(ledger.directory/"source-attempts"/str(seed))
        source=_json((source_directory/"source.json").read_bytes())
        source_claim=_json((source_directory/"claim.json").read_bytes())
        require(source_claim["auditor_pid"]==pid and source_claim["auditor_binding_sha256"]==binding
            and source_claim["synthetic_process_fixture"] is False
            and source_claim["recipe_sha256"]==source_recipe_sha
            and source_claim["source_review_sha256"]==source_review_sha
            and type(source_claim["source_seed"]) is int and source_claim["source_seed"]==seed,
            "source belongs to another auditor, seed or reviewed recipe")
        read_source_owner(source_directory,source,spec["source_runtime"],ledger.contract)
        observed.extend(r["root_id"] for r in source["roots"]);missing.extend(source["missing_root_ids"])
        source_hashes[str(seed)]=digest(source)
    require(panel["source_count"]==len(seeds) and panel["observed_root_ids"]==observed
        and panel["missing_root_ids"]==missing and panel["source_sha256_by_seed"]==source_hashes,
        "returned panel differs from fixed durable source roster")
    slots={r["root_id"] for r in ledger.contract["panels"]["exploration"]["root_slots"]}
    require(len(observed)+len(missing)==len(slots) and len(set(observed))==len(observed)
        and len(set(missing))==len(missing) and not set(observed)&set(missing)
        and set(observed)|set(missing)==slots,"completed sources must partition fixed root slots")
    gaps={}
    for stage in ("A1","A2","A3"):
        for cfg in ledger._stage_configs(stage):
            evidence_path=ledger.directory/stage/(cfg.identity+"-root-evidence.json")
            recipe=prepared["recipes"].get((stage,cfg.identity))
            if type(recipe) is ProspectivelyUnavailableStageRecipe:
                require(evidence_path.exists(),"prospective unavailable stage evidence missing")
            if not evidence_path.exists():continue
            evidence=_json(evidence_path.read_bytes())
            if type(recipe) is ProspectivelyUnavailableStageRecipe:
                from .search_over_raw_stage_run import read_unavailable_stage_evidence
                actual=read_unavailable_stage_evidence(ledger,stage,cfg.identity,evidence)
                require(digest(actual)==digest(validate_unavailable_recipe(ledger,stage,cfg.identity,recipe)),
                    "completed unavailability differs from prospective registry")
            else:
                require("prospective_configuration_unavailable" not in evidence,
                    "executable recipe cannot become prospectively unavailable after work")
            handoff=evidence.get("owned_input_handoff",{})
            for root,receipt in handoff.get("input_unavailable_by_root",evidence.get("input_unavailable_by_root",{})).items():
                key=root+":"+cfg.identity
                require(root in observed and (key not in gaps or digest(gaps[key])==digest(receipt)),
                    "completed unavailable input coordinate/reuse drift")
                gaps[key]=receipt
    require(type(result.get("input_unavailable_by_root_configuration",{})) is dict
        and digest(result.get("input_unavailable_by_root_configuration",{}))==digest(gaps),
        "completed staged result differs from durable unavailable partition")
    final_closure=verify_delegated_closure(directory,pid,binding,_verify_closed_generations)
    journal_costs=None
    if "attempt_declaration" in claim:
        from .search_over_raw_attempt_journal import _permission_handle,read_attempt_journal_cost
        handle,_=_permission_handle(claim["attempt_declaration"])
        journal_costs=read_attempt_journal_cost(handle,expected_outer_terminal=expected_terminal)
        require(journal_costs["declared_supervised_attempt_census_complete"] is True,
            "completed entry lacks closed declared supervised-attempt census")
    ledger.verify()  # Callback-free checks AFTER the final callback-bearing read.
    for (stage_name,identity),recipe in prepared["recipes"].items():
        if type(recipe) is ProspectivelyUnavailableStageRecipe:
            validate_unavailable_recipe(ledger,stage_name,identity,recipe)
            require_unavailable_candidate_absence(ledger,identity)
    require(hashlib.sha256(Path(job.spec_path).read_bytes()).hexdigest()==job.spec_sha256
        and before==_evidence_inventory(ledger.directory) and final_closure==closure,
        "entry owner evidence changed during independent validation")
    return dict(status="EXPLORATION_WORKFLOW_OWNER_VALIDATED" if diagnostic_readback is not None else "STAGED_EXPLORATION_OWNER_VALIDATED",
        **({"diagnostic_readback":diagnostic_readback} if diagnostic_readback is not None else {}),
        costs=costs,attempt_journal_costs=journal_costs,source_count=len(seeds),
        full_resource_cost_complete=False,runtime_qualified=False,holdout_authorized=False,phase_b_authorized=False)


def _evidence_inventory(directory):
    """Snapshot original files AND entries; reject symlink/special-file aliases."""
    rows={}
    for path in _path(directory).rglob("*"):
        path=_path(path)
        require(path.is_file() or path.is_dir(),"unsupported entry evidence node")
        rows[str(path)]="directory" if path.is_dir() else hashlib.sha256(path.read_bytes()).hexdigest()
    return rows
