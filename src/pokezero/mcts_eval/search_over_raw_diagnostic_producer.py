"""Fixed six-recipe slot-zero diagnostic producer on a retained owned panel.

No source allocation, collection admission, holdout, deployment selection or
archive closure. All original slots remain, including unavailable recipes,
missing roots and clean emergency terminations. A terminal rollout's accepted
backup count is not independently validated terminal truth.
"""
from copy import deepcopy
from dataclasses import asdict, dataclass, fields, replace
import hashlib
import json
from pathlib import Path
import pickle

from .search_over_raw import SearchConfiguration, digest, require
from .search_over_raw_ledger import _write_new
from .search_over_raw_reconciled import ReconciledPhaseALedger, panel_roster, controller_input_hashes, a1_configurations
from .search_over_raw_source_run import TrustedExplorationPanel, TrustedOwnedSource
from .search_over_raw_fixed_work import FixedWorkDiagnosticContract
from .search_over_raw_fixed_work_run import FixedWorkRunValidator
from .search_over_raw_supervision import FixedWorkAdapterJob, supervise_fixed_work
from .search_over_raw_nested_ownership import DiagnosticIncompletePolicy, current_scope
from .search_over_raw_attempt_cost import AttemptCostSpec, _attempt, _inventory
from .search_over_raw_resources import resource_source_hash
from .search_over_raw_archive import (prepare_archived_fixed_work_job, prepare_archived_reference_fixed_work_job,
    archived_exploration_root_binding)
from .search_over_raw_world_inputs import _state_payload, compare_world_inputs


def diagnostic_configurations():
    return tuple(SearchConfiguration(arm, "public", leaf, 1., 20 if arm == "reference" else 1)
        for arm in ("incumbent", "reference") for leaf in ("model", "hp_fraction", "raw_rollout"))


def fixed_work_runtime_recipe(configuration,checkpoint_contract,*,reference_factory=None,native_ancestry_contract=None,
                              capture_native_public_fold=False,native_fixed_value_contract=None,
                              owned_reference_value_recipe=None):
    """No-load exact ordinary fixed adapter law; no future root fields.

    Mirrors the explicitly scoped WorldInputFixedWorkAdapter recipe. The
    separate native-value lane requires its explicit contract; historical
    manifests cannot acquire it through an ordinary recipe. Timed/callback
    lanes remain different runtimes and are not admitted here.
    """
    from .search_over_raw_world_inputs import SCHEMA,OWNERSHIP_METADATA,TIMED_REFERENCE_PUBLIC_CAPTURE
    from .search_over_raw_atomic_work import SCHEMA as REFERENCE_ATOMIC_SCHEMA
    from .search_over_raw_native_atomic_work import SCHEMA as NATIVE_ATOMIC_SCHEMA
    from .search_over_raw_construction import SCHEMA as CONSTRUCTION_SCHEMA,construction_source_hashes
    from .search_over_raw_ancestry import SCHEMA as ANCESTRY_SCHEMA,NativeAncestryContract
    from .paper_reference_runtime import ShowdownWorkerFactory
    cfg=configuration
    require(type(cfg) is SearchConfiguration and cfg.identity in {c.identity for c in diagnostic_configurations()},
        "canonical fixed diagnostic configuration required")
    if native_fixed_value_contract is not None:
        from .search_over_raw_native_fixed_values import NativeFixedWorkValueContract
        require(type(native_fixed_value_contract) is NativeFixedWorkValueContract
            and cfg.arm=="incumbent" and cfg.leaf=="model" and cfg.belief=="public"
            and cfg.workers==1 and native_ancestry_contract is None
            and capture_native_public_fold is False and reference_factory is None,
            "native fixed values need a separate prospective model-only recipe")
    require(type(capture_native_public_fold) is bool and (not capture_native_public_fold or
        cfg.arm=="incumbent" and cfg.leaf=="model" and native_ancestry_contract is not None),
        "prospective fixed-model full public-fold capture scope required")
    if cfg.arm=="reference":
        require(type(reference_factory) is ShowdownWorkerFactory and native_ancestry_contract is None
            and (reference_factory.checkpoint,reference_factory.checkpoint_sha256,reference_factory.showdown_root,
                reference_factory.set_source_hash)==(checkpoint_contract.checkpoint_path,checkpoint_contract.checkpoint_sha256,
                    checkpoint_contract.showdown_root,checkpoint_contract.showdown_source_sha256),
            "original reference champion/simulator/catalog factory required")
    else:
        native_scope = (native_fixed_value_contract is not None or
            type(native_ancestry_contract) is NativeAncestryContract
            and native_ancestry_contract.capture_native_edges is True) if cfg.leaf == "model" else native_ancestry_contract is None
        require(reference_factory is None and native_scope, "original native ancestry scope required")
    value=dict(configuration=asdict(cfg),checkpoint_sha256=checkpoint_contract.checkpoint_sha256,
        set_source_hash=checkpoint_contract.showdown_source_sha256,root_statistics="independent_per_root",
        incumbent=dict(depth=6,sims=16,batch=16,worlds=4,native_batch_guard_ms=0) if cfg.arm=="incumbent" else None,
        reference_factory=asdict(reference_factory) if cfg.arm=="reference" else None,
        initial_dispatch_workers=6 if cfg.arm=="reference" else None,
        fixed_work_diagnostic=FixedWorkDiagnosticContract().receipt(),
        consumed_world_inputs=dict(schema=SCHEMA,initial_inputs_only=True,snapshot_fields_excluded=[],
            ownership_metadata_excluded=OWNERSHIP_METADATA if cfg.arm=="reference" else [],
            instrumentation_can_change_deadlines=True,qualifies_uninstrumented_runtime=False))
    if cfg.arm=="reference":
        value.update(owned_process_receipts=dict(enabled=True,publishes_before_detachment=True,qualifies_uninstrumented_runtime=False),
            reference_atomic_work=dict(schema=REFERENCE_ATOMIC_SCHEMA,scope="actual_own_backup_and_cleanup; not independent simulator labels",
                instrumentation_can_change_deadlines=True,runtime_qualified=False),
            full_public_request_observation=deepcopy(TIMED_REFERENCE_PUBLIC_CAPTURE))
        if cfg.leaf!="model":value["reference_leaf"]=dict(leaf=cfg.leaf,rollout_cap=250,
            rollout_policy="raw_argmax_both_seats",tree="unchanged_trajectory_reference",priors="unchanged_champion")
    else:
        value.update(native_atomic_work=dict(schema=NATIVE_ATOMIC_SCHEMA,scope="observed_float32_backup_consistency; not independent terminal labels",
                instrumentation_can_change_deadlines=True,runtime_qualified=False),
            native_construction=dict(schema=CONSTRUCTION_SCHEMA,source_hashes=construction_source_hashes(),
                private_controller_only=True,instrumentation_can_change_deadlines=True,runtime_qualified=False),
            full_public_request_observation=dict(schema="pokezero.search-over-raw.public-request-consumption.v1",
                at_each_actual_native_sampling_attempt=True,pending_transition_included=True,
                private_source_snapshot_included=False,runtime_qualified=False))
        if cfg.leaf!="model":
            value["incumbent_leaf"]=dict(leaf=cfg.leaf,tree="unchanged_encoded_model_tree",priors="unchanged_champion",
                model_forwards="retained",value_frame="side_one_absolute")
            if cfg.leaf=="raw_rollout":value["incumbent_leaf"].update(policy="both_seats_own_raw_masked_argmax",rollout_cap=250,
                cap="refusal_without_value_fallback",deadline="none; external emergency watchdog only",rollout_count=1,
                rollout_threads=1,branch_on_damage=True,seed="selection_seed_then_sha256_world_domain_v1")
        if native_ancestry_contract is not None:
            value["native_searched_ancestry"]=dict(schema=ANCESTRY_SCHEMA,contract=asdict(native_ancestry_contract),
                private_controller_only=True,decision_deadline=None,emergency_watchdog="external_only",
                event_population="actual_model_evaluations_including_cancelled_rounds",labels_used_for_backup=False,
                instrumentation_can_change_selection=True,runtime_qualified=False)
    if capture_native_public_fold:
        from .search_over_raw_public_fold import runtime_receipt
        value["native_public_fold"]=runtime_receipt()
    if native_fixed_value_contract is not None:
        from .search_over_raw_native_fixed_value_run import runtime_receipt
        value["native_fixed_values"]=runtime_receipt(native_fixed_value_contract)
    if owned_reference_value_recipe is not None:
        _check_reference_values(cfg,owned_reference_value_recipe,checkpoint_contract)
        value["owned_reference_value_recipe"]=owned_reference_value_recipe.receipt()
    return value


@dataclass(frozen=True, kw_only=True)
class FixedWorkRecipe:
    runtime: dict | None = None
    reference_factory: object = None
    native_ancestry_contract: object = None
    unqualified_reason: str | None = None
    capture_native_public_fold: bool = False


@dataclass(frozen=True, kw_only=True)
class ReferenceValueFixedWorkRecipe(FixedWorkRecipe):
    """Distinct prospective recipe; ordinary/historical recipes gain no labels."""
    owned_reference_value_recipe: object = None

    def __post_init__(self):
        from .search_over_raw_reference_values import ReferenceValueLabelRecipe
        require(type(self.owned_reference_value_recipe) is ReferenceValueLabelRecipe
            and self.unqualified_reason is None and self.capture_native_public_fold is False
            and self.native_ancestry_contract is None,
            "owned reference diagnostic needs an explicit ready label recipe")


def reference_value_recipe(recipe):
    require(type(recipe) in (FixedWorkRecipe,ReferenceValueFixedWorkRecipe),
        "exact prospective diagnostic recipe required")
    return recipe.owned_reference_value_recipe if type(recipe) is ReferenceValueFixedWorkRecipe else None


def _check_reference_values(cfg,recipe,checkpoint):
    from .search_over_raw_reference_values import ReferenceValueLabelRecipe
    from .search_over_raw_outcome_run import verify_outcome_runtime
    require(type(recipe) is ReferenceValueLabelRecipe and cfg.arm=="reference"
        and cfg.belief=="public" and cfg.leaf in {"model","hp_fraction"} and cfg.workers==20
        and recipe.runtime["checkpoint_path"]==checkpoint.checkpoint_path
        and recipe.runtime["files"][checkpoint.checkpoint_path]==checkpoint.checkpoint_sha256
        and recipe.runtime["argv"][3]==checkpoint.showdown_root,
        "reference labels require original public model/HP champion and simulator")
    verify_outcome_runtime(recipe.runtime)


def diagnostic_manifest(contract, recipes, *, checkpoint_contract, showdown_root, file_hashes):
    """Pure prospective data; only EXISTING exploration identities are used."""
    roster = panel_roster(contract["panels"]["exploration"])
    configurations = diagnostic_configurations()
    require(type(recipes) is dict and set(recipes) == {c.identity for c in configurations}
        and all(type(r) in (FixedWorkRecipe,ReferenceValueFixedWorkRecipe) for r in recipes.values()),
        "all six diagnostic recipe states required")
    fixed = FixedWorkDiagnosticContract()
    require(digest(contract["fixed_work_diagnostic"]) == digest(dict(fixed.receipt(),
        root_subset="root_slot zero of every exploration seed")), "original fixed-work diagnostic law required")
    rows=[]
    for cfg in configurations:
        recipe=recipes[cfg.identity]
        labels=reference_value_recipe(recipe)
        if labels is not None:
            _check_reference_values(cfg,labels,checkpoint_contract)
            require(type(recipe.runtime) is dict
                and digest(recipe.runtime.get("owned_reference_value_recipe"))==digest(labels.receipt())
                and all(file_hashes.get(p)==h for p,h in labels.runtime["files"].items()),
                "reference label law and complete runtime inventory must be prospectively pinned")
        else:
            require(recipe.runtime is None or "owned_reference_value_recipe" not in recipe.runtime,
                "ordinary diagnostic recipe cannot silently acquire reference labels")
        require(type(recipe.capture_native_public_fold) is bool and (not recipe.capture_native_public_fold or
            recipe.unqualified_reason is None and cfg.arm=="incumbent" and cfg.leaf=="model"),
            "public-fold capture is an explicit ready native-model recipe only")
        if recipe.unqualified_reason is not None:
            require(type(recipe.unqualified_reason) is str and recipe.unqualified_reason.strip()
                and recipe.runtime is None and recipe.reference_factory is None and recipe.native_ancestry_contract is None,
                "unqualified diagnostic recipe requires a reason, not executable inputs")
        else:
            require(type(recipe.runtime) is dict and digest(recipe.runtime.get("fixed_work_diagnostic")) == digest(fixed.receipt()),
                "ready diagnostic recipe must use fixed accepted work, not a decision ceiling")
            if recipe.capture_native_public_fold:
                from .search_over_raw_public_fold import runtime_receipt
                require(digest(recipe.runtime.get("native_public_fold")) == digest(runtime_receipt()),
                    "public-fold runtime must match explicit prospective capture scope")
            else:
                require("native_public_fold" not in recipe.runtime,
                    "public-fold runtime cannot silently enable unreviewed capture")
            if cfg.arm == "reference":
                from .paper_reference_runtime import ShowdownWorkerFactory
                require(type(recipe.reference_factory) is ShowdownWorkerFactory and recipe.native_ancestry_contract is None,
                    "reference diagnostic needs its original twenty-worker factory")
            else:
                require(recipe.reference_factory is None, "native diagnostic cannot substitute a reference factory")
                from .search_over_raw_ancestry import NativeAncestryContract
                require(type(recipe.native_ancestry_contract) is NativeAncestryContract
                    and recipe.native_ancestry_contract.capture_native_edges is True if cfg.leaf == "model"
                    else recipe.native_ancestry_contract is None,
                    "native model diagnostic requires prospectively sampled actual ancestry")
        rows.append(dict(configuration=asdict(cfg), configuration_sha256=cfg.identity,
            status="UNQUALIFIED_RECIPE" if recipe.unqualified_reason is not None else "READY",
            reason=recipe.unqualified_reason, runtime=deepcopy(recipe.runtime),
            factory=_state_payload(recipe.reference_factory), ancestry=_state_payload(recipe.native_ancestry_contract),
            capture_native_public_fold=recipe.capture_native_public_fold))
        if labels is not None:rows[-1]["owned_reference_value_recipe"]=_state_payload(labels)
    zero={r["source_seed"]:r for r in contract["panels"]["exploration"]["root_slots"] if r["root_slot"] == 0}
    slots=[dict(root=deepcopy(zero[seed]),configuration_sha256=cfg.identity) for seed,_ in roster for cfg in configurations]
    # Review bytes themselves are subsequently pinned in input_hashes. Exclude
    # that provenance map here to avoid an impossible review/hash fixed point;
    # execution still verifies the complete original ledger and controller map.
    labeled=any(reference_value_recipe(r) is not None for r in recipes.values())
    return dict(schema="pokezero.search-over-raw.fixed-diagnostic-manifest.v"+("2" if labeled else "1"),
        contract_design_sha256=digest({k:v for k,v in contract.items() if k != "input_hashes"}),
        fixed_work_contract=fixed.receipt(), recipes=rows, slots=slots,
        checkpoint=_state_payload(vars(checkpoint_contract)), showdown_root=showdown_root,
        file_hashes=deepcopy(file_hashes), decision_ceiling=None,
        diagnostic_results_used_for_selection=False, all_recipes_must_qualify=False,
        source_allocation=False, collection_authorized=False, holdout_authorized=False)


def diagnostic_review_template(manifest):
    return dict(schema="pokezero.search-over-raw.fixed-diagnostic-review.v1",manifest_sha256=digest(manifest),
        status="REVIEW_REQUIRED", independent_review_completed=False, reviewed_before_first_source_work=False,
        no_deployment_selection_use=True, local_clean_timeout_only=True, collection_authorized=False)


@dataclass(frozen=True)
class OwnedFixedWorkDiagnostic:
    """Private live-archive handoff to the consumed-draw/replay coordinator."""
    source_owner: object
    job: object
    reader: object
    selected: dict
    selected_sha256: str

    def __post_init__(self):
        object.__setattr__(self,"_objects",(self.source_owner,self.job,self.reader,self.selected))
        object.__setattr__(self,"_frozen",self._inputs())

    def __getstate__(self):raise TypeError("private diagnostic owner cannot be serialized")

    def _inputs(self):
        return digest(dict(job_sha256=hashlib.sha256(pickle.dumps(self.job)).hexdigest(),selected=self.selected,
            selected_sha256=self.selected_sha256,reader_inputs=self.reader._inputs(),readout=self.reader.last_validation,
            inventory=_inventory([self.reader.directory]),source_inputs=self.source_owner._identity()))

    def _original_inputs(self):
        from .search_over_raw_reference_values import ReferenceValueRunValidator
        expected_reader=ReferenceValueRunValidator if self.job.owned_reference_value_recipe is not None else FixedWorkRunValidator
        require(all(actual is original for actual,original in zip(
            (self.source_owner,self.job,self.reader,self.selected),self._objects))
            and type(self.source_owner) is TrustedOwnedSource
            and type(self.reader) is expected_reader
            and (expected_reader is FixedWorkRunValidator or
                digest(_state_payload(self.reader.recipe))==digest(_state_payload(self.job.owned_reference_value_recipe)))
            and (type(self.job) is FixedWorkAdapterJob or self.reader.allow_synthetic is True
                and isinstance(self.job,FixedWorkAdapterJob) and getattr(self.job,"synthetic_process_fixture",False) is True)
            and self._inputs()==self._frozen,"original private diagnostic owner changed")
        self.source_owner._original_owner()

    def validate(self, ledger):
        self._original_inputs()
        self.source_owner.require_inputs(ledger,self.source_owner.archive,self.source_owner.source)
        require(self.reader(self.selected) == self.selected_sha256, "original diagnostic owner evidence drift")
        self.source_owner.require_inputs(ledger,self.source_owner.archive,self.source_owner.source)
        self._original_inputs()
        return deepcopy(self.reader.last_validation)


def _job_envelope(job):
    return dict(configuration=asdict(job.configuration),checkpoint=_state_payload(vars(job.checkpoint_contract)),
        showdown_root=job.showdown_root,context=_state_payload(job.context),pending=_state_payload(job.pending_transition),
        root_id=job.root_id,statistics_root_id=job.statistics_root_id,selection_seed=job.selection_seed,
        runtime_sha256=job.expected_runtime_sha256,public_request_sha256=job.expected_public_request_sha256,
        factory=_state_payload(job.reference_factory),ancestry=_state_payload(job.native_ancestry_contract),
        capture_native_public_fold=job.capture_native_public_fold,
        owned_reference_value_recipe=_state_payload(job.owned_reference_value_recipe),
        consumed=job._consumed_binding,set_source_hash=job._set_source_hash)


def produce_owned_fixed_work_diagnostics(ledger, panel, *, recipes, checkpoint_contract,
        showdown_root, file_hashes, review_path, _synthetic_job_transport=None):
    """One-shot producer; original archives remain caller-owned and live.

    Authored transport is confined to unadmitted disposable panels. It may add
    typed job-subclass fields, never replace source bindings or owning readers.
    This function does not grant scientific execution authority.
    """
    require(type(ledger) is ReconciledPhaseALedger and type(panel) is TrustedExplorationPanel
        and panel.ledger is ledger, "exact original retained panel/ledger required")
    ledger.verify()
    synthetic=_synthetic_job_transport is not None
    if synthetic:
        require(callable(_synthetic_job_transport) and getattr(_synthetic_job_transport,"synthetic_process_fixture",False) is True
            and ledger.contract["execution_ready"] is False and ledger.contract["collection_authorized"] is False,
            "authored diagnostic jobs cannot enter admitted science")
    else: ledger.verify_exploration("A1",a1_configurations()[0].identity)
    manifest=diagnostic_manifest(ledger.contract,recipes,checkpoint_contract=checkpoint_contract,
        showdown_root=showdown_root,file_hashes=file_hashes)
    manifest_sha=digest(manifest)
    review_path=Path(review_path).resolve();review_bytes=review_path.read_bytes()
    review=json.loads(review_bytes)
    expected=diagnostic_review_template(manifest)
    expected.update(status="CLEAR_FIXED_WORK_DIAGNOSTIC_ROSTER",independent_review_completed=True,
        reviewed_before_first_source_work=True)
    require(digest(review)==digest(expected) and ledger.contract["input_hashes"].get(str(review_path))
        ==hashlib.sha256(review_bytes).hexdigest(), "prospective independently reviewed diagnostic roster required")
    recipe_objects=tuple((k,id(v)) for k,v in recipes.items())
    require(all(ledger.contract["input_hashes"].get(p)==h for p,h in controller_input_hashes().items()),
        "diagnostic producer code must be prospectively pinned")
    panel.validate()
    claim_path=ledger.directory/"fixed-work-diagnostic-claim.json"
    _write_new(claim_path,dict(manifest=manifest,manifest_sha256=manifest_sha,contract_sha256=ledger.contract_sha256,
        review_sha256=hashlib.sha256(review_bytes).hexdigest(),panel_inputs_sha256=panel.frozen,
        synthetic_process_fixture=synthetic,retry_authorized=False))
    claim_bytes=claim_path.read_bytes()
    def guard():
        ledger.verify();panel._original_inputs()
        require(tuple((k,id(v)) for k,v in recipes.items())==recipe_objects
            and digest(diagnostic_manifest(ledger.contract,recipes,checkpoint_contract=checkpoint_contract,
                showdown_root=showdown_root,file_hashes=file_hashes))==manifest_sha
            and review_path.read_bytes()==review_bytes and claim_path.read_bytes()==claim_bytes,
            "original diagnostic manifest/review/claim changed")
        require(all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==h for p,h in file_hashes.items()),
            "original diagnostic declared files changed")
    owners={};rows=[];original_attempts={};configurations={c.identity:c for c in diagnostic_configurations()}
    def final_originals():
        guard()
        for owner in owners.values():owner._original_inputs()  # No late reader/source callback after this pass.
        require(all(_inventory([path])==snapshot for path,snapshot in original_attempts.items()),
            "earlier diagnostic attempt receipts or ownership inventory changed")
    fixed=FixedWorkDiagnosticContract()
    try:
        guard()
        for slot in manifest["slots"]:
            root_id=slot["root"]["root_id"];identity=slot["configuration_sha256"]
            cfg=configurations[identity];recipe=recipes[identity];source_owner=panel.by_root.get(root_id)
            row=dict(root_id=root_id,source_seed=slot["root"]["source_seed"],configuration_sha256=identity,
                status="UNSTARTED",attempt_directory=None,accepted_work=None,selected_sha256=None,cost=None,
                independent_terminal_truth_validated=False,deployable=False)
            if source_owner is None: rows.append(row);continue
            if recipe.unqualified_reason is not None:
                row.update(status="SKIPPED_UNQUALIFIED_RECIPE",reason=recipe.unqualified_reason)
                rows.append(row);continue
            source_owner.require_inputs(ledger,source_owner.archive,source_owner.source);guard()
            before=archived_exploration_root_binding(source_owner.archive,source_owner.source,
                contract=ledger.contract,root_id=root_id)
            root=next(r for r in source_owner.source["roots"] if r["root_id"]==root_id)
            context,pending,_=source_owner.archive.selected(root)
            from .head_to_head import public_only_context
            from .paper_reference_runtime import PublicRootRequest
            public=public_only_context(context)
            request=PublicRootRequest.capture(public.public_materialization_state,public.observation,pending_transition=pending)
            sanitized=replace(public,observation=request.observation,requested_observations={public.player_id:request.observation},
                public_materialization_state=request.state)
            witness=before["public_request_witness"]
            expected_envelope=dict(configuration=asdict(cfg),checkpoint=_state_payload(vars(checkpoint_contract)),
                showdown_root=showdown_root,context=_state_payload(sanitized),pending=_state_payload(pending),
                root_id=root_id+":"+identity,statistics_root_id=root_id,selection_seed=slot["root"]["source_seed"],
                runtime_sha256=digest(recipe.runtime),public_request_sha256=witness["public_request_sha256"],
                factory=_state_payload(recipe.reference_factory),ancestry=_state_payload(recipe.native_ancestry_contract),
                capture_native_public_fold=recipe.capture_native_public_fold,
                owned_reference_value_recipe=_state_payload(reference_value_recipe(recipe)),
                consumed=witness["consumed_binding"],set_source_hash=witness["consumed_binding"]["set_source_hash"])
            def original_job():
                require(type(job.checkpoint_contract) is type(checkpoint_contract)
                    and digest(_job_envelope(job))==digest(expected_envelope)
                    and digest(bound)==digest(before),"returned diagnostic job differs from original prospective envelope")
            kwargs=dict(contract=ledger.contract,root_id=root_id,configuration=cfg,checkpoint_contract=checkpoint_contract,
                showdown_root=showdown_root,runtime_sha256=digest(recipe.runtime))
            job,bound=(prepare_archived_reference_fixed_work_job(source_owner.archive,source_owner.source,
                reference_factory=recipe.reference_factory,
                owned_reference_value_recipe=reference_value_recipe(recipe),**kwargs) if cfg.arm=="reference" else
                prepare_archived_fixed_work_job(source_owner.archive,source_owner.source,
                    native_ancestry_contract=recipe.native_ancestry_contract,
                    capture_native_public_fold=recipe.capture_native_public_fold,**kwargs))
            original_job();guard()
            if synthetic:
                extra=_synthetic_job_transport(configuration=cfg,job=job,source_owner=source_owner)
                require(type(extra) is dict and set(extra)=={"job_class","job_kwargs"}
                    and isinstance(extra["job_class"],type) and issubclass(extra["job_class"],FixedWorkAdapterJob)
                    and getattr(extra["job_class"],"synthetic_process_fixture",False) is True
                    and type(extra["job_kwargs"]) is dict
                    and not set(extra["job_kwargs"]) & {f.name for f in fields(FixedWorkAdapterJob)},
                    "authored transport adds fields only on an explicitly synthetic typed diagnostic job")
                original=job
                original_root=next(r for r in source_owner.source["roots"] if r["root_id"]==root_id)
                original_context,_,_=source_owner.archive.selected(original_root)
                args={f.name:getattr(original,f.name) for f in fields(FixedWorkAdapterJob)}
                args["context"]=deepcopy(original_context)  # Re-sanitize ORIGINAL capture, never resample a stripped request.
                job=extra["job_class"](**args,**extra["job_kwargs"])
                original_job();guard()
            payload=pickle.dumps(job);job_sha=hashlib.sha256(payload).hexdigest()
            binding=digest(dict(manifest_sha256=manifest_sha,slot=slot,job_sha256=job_sha,source_binding=bound))
            directory=(ledger.directory/"fixed-work-diagnostics"/digest(slot)).resolve()
            scope=current_scope()
            spec=AttemptCostSpec(str(directory),job_sha,binding,"fixed-work","selection",fixed.receipt(),
                resource_source_hash(),fixed.emergency_watchdog_seconds,None if scope is None else str(scope.directory))
            labels=reference_value_recipe(recipe)
            from .search_over_raw_reference_values import ReferenceValueRunValidator
            reader_class=FixedWorkRunValidator if labels is None else ReferenceValueRunValidator
            reader=reader_class(**({"recipe":labels} if labels is not None else {}),
                directory=directory,job_sha256=job_sha,binding_sha256=binding,
                configuration=cfg,root_id=expected_envelope["root_id"],statistics_root_id=root_id,
                selection_seed=expected_envelope["selection_seed"],runtime_sha256=expected_envelope["runtime_sha256"],
                consumed_binding=witness["consumed_binding"],
                file_hashes=file_hashes,checkpoint_path=checkpoint_contract.checkpoint_path,
                checkpoint_sha256=checkpoint_contract.checkpoint_sha256,showdown_root=showdown_root,
                public_request_sha256=witness["public_request_sha256"],allow_synthetic=synthetic,
                native_public_fold_required=recipe.capture_native_public_fold)
            def admit():
                guard();source_owner.require_inputs(ledger,source_owner.archive,source_owner.source);guard()
                original_job()
                require(hashlib.sha256(pickle.dumps(job)).hexdigest()==job_sha,
                    "original diagnostic job changed before/after dispatch")
                if not synthetic:ledger.verify_exploration("A1",a1_configurations()[0].identity)
            policy=None if synthetic else DiagnosticIncompletePolicy(manifest_sha,digest(slot),job_sha,binding)
            terminal=supervise_fixed_work(job,directory=directory,binding_sha256=binding,verify_admission=admit,
                diagnostic_incomplete_policy=policy)
            admit()
            cost,_,observed=_attempt(spec)
            require(digest(observed)==digest(terminal) and cost["wait4_usage_join_validated"] is True,
                "original diagnostic cost/terminal wait join required")
            row.update(attempt_directory=str(directory),cost=cost)
            if terminal["supervisor_exit_code"]==124:
                require(terminal["status"]=="DIAGNOSTIC_INCOMPLETE" and terminal["cleanup_error_type"] is None
                    and terminal["result"] is None,"only clean unscored watchdog is local diagnostic incompletion")
                row["status"]="DIAGNOSTIC_INCOMPLETE"
            else:
                require(terminal["supervisor_exit_code"]==0,"diagnostic operational failure stops original producer")
                selected=terminal["result"];selected_sha=reader(selected)
                require(selected_sha==digest(selected),"independent original fixed-work owning reader required")
                owner=OwnedFixedWorkDiagnostic(source_owner,job,reader,selected,selected_sha)
                validation=owner.validate(ledger);guard();owners[(root_id,identity)]=owner
                row.update(status="DIAGNOSTIC_COMPLETE",accepted_work=fixed.accepted_boundary(cfg.arm),
                    selected_sha256=selected_sha,owning_readout=validation)
            rows.append(row);original_attempts[directory]=_inventory([directory]);final_originals()
        comparisons=[]
        for seed,_ in panel_roster(ledger.contract["panels"]["exploration"]):
            root_id=next(s["root"]["root_id"] for s in manifest["slots"] if s["root"]["source_seed"]==seed)
            for arm in ("incumbent","reference"):
                available=[(cfg,owners[(root_id,cfg.identity)]) for cfg in configurations.values()
                    if cfg.arm==arm and (root_id,cfg.identity) in owners]
                for index,(left,lo) in enumerate(available):
                    for right,ro in available[index+1:]:
                        lo.validate(ledger);ro.validate(ledger);guard()
                        comparison=compare_world_inputs(lo.selected["evidence"]["consumed_world_inputs"],
                            ro.selected["evidence"]["consumed_world_inputs"])
                        comparisons.append(dict(root_id=root_id,left=left.identity,right=right.identity,
                            actual_consumed_input_comparison=comparison))
        result=dict(schema="pokezero.search-over-raw.fixed-diagnostic-population.v1",manifest_sha256=manifest_sha,
            required_slots=len(manifest["slots"]),slots=rows,within_arm_world_comparisons=comparisons,
            archives_still_caller_owned=True,diagnostic_results_used_for_selection=False,
            complete_resource_cost=False,terminal_truth_validated=False,scientific_measurement_validated=False,
            runtime_qualified=False,holdout_authorized=False,retries_permitted=False)
        for owner in owners.values():owner.validate(ledger)
        final_originals();_write_new(ledger.directory/"fixed-work-diagnostics.json",result)
        return owners,result
    except BaseException as failure:
        try:
            if not (ledger.directory/"operational-stop.json").exists():ledger.stop(type(failure).__name__)
        except BaseException as stop_failure:
            failure.add_note("Operational stop persistence also failed: "+type(stop_failure).__name__)
        raise
