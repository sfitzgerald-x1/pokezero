"""Owned full-panel A1/A2 -> evaluator freeze -> eligible A3 orchestration.

Consumes an already collected, retained panel. This is deliberately not a CLI,
source allocator, collection authorization, fixed-work diagnostic or holdout
entry. Production delegates admission and ownership to the existing producers.
The caller relinquishes the private panel: every archive is closed on return.
"""
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
import hashlib
import json
import time

from .search_over_raw import SearchConfiguration, digest, require
from .search_over_raw_ledger import _write_new
from .search_over_raw_reconciled import (
    ReconciledPhaseALedger, a1_configurations, a2_configurations,
    deployment_configurations, selector_runtime_recipe, controller_input_hashes,
)
from .search_over_raw_source_run import TrustedExplorationPanel
from .search_over_raw_selector_producer import (produce_archived_selector,read_actor_input_unavailable,
    _actor_unavailable_path)
from .search_over_raw_actor_timed_run import RUNTIME_KEY
from .search_over_raw_continuation_producer import produce_reserved_continuation
from .search_over_raw_continuation_admission import ContinuationAdmissionJournal
from .search_over_raw_root_plan import ArchivedRootEvaluationPlan, _owner_inputs
from .search_over_raw_world_inputs import _state_payload


@dataclass(frozen=True, kw_only=True)
class StageSelectorRecipe:
    runtime: dict
    qualification_review_path: str
    reference_factory: object = None


@dataclass(frozen=True, kw_only=True)
class ProspectivelyUnavailableStageRecipe:
    reason: str
    review_path: str


def unavailable_configuration_review_template(contract, *, stage, configuration, reason):
    """Review a declared unavailable candidate, never fabricate qualification."""
    from .search_over_raw_reconciled import validate_contract
    validate_contract(contract)
    require((stage, configuration) in exploration_recipe_keys()
        and type(reason) is str and reason == reason.strip() and bool(reason),
        "explicit stage-local prospective unavailability reason required")
    core={k:deepcopy(v) for k,v in contract.items() if k not in
        {"input_hashes","execution_ready","collection_authorized","remaining_gates"}}
    return dict(schema="pokezero.search-over-raw.configuration-unavailable-review.v1",
        status="UNREVIEWED_PROSPECTIVE_CONFIGURATION_UNAVAILABLE",stage=stage,
        configuration=configuration,reason=reason,contract_recipe_sha256=digest(core),
        controller_input_hashes=controller_input_hashes(),independent_review_completed=False,
        reviewed_before_first_source_work=False,configuration_available=False,
        selector_attempted=False,observed_zero_work=False,runtime_qualified=False,
        collection_authorized=False,holdout_authorized=False,phase_b_authorized=False)


def validate_unavailable_recipe(ledger, stage, configuration, recipe):
    require(type(recipe) is ProspectivelyUnavailableStageRecipe,
        "exact typed prospective unavailability required")
    path=Path(recipe.review_path)
    require(path.is_absolute() and str(path.resolve()) == recipe.review_path,
        "canonical prospective availability review required")
    raw=path.read_bytes();review=json.loads(raw)
    expected=unavailable_configuration_review_template(ledger.contract,stage=stage,
        configuration=configuration,reason=recipe.reason)
    expected.update(status="CLEAR_PROSPECTIVE_CONFIGURATION_UNAVAILABLE",
        independent_review_completed=True,reviewed_before_first_source_work=True)
    require(digest(review)==digest(expected)
        and ledger.contract["input_hashes"].get(str(path))==hashlib.sha256(raw).hexdigest()
        and all(ledger.contract["input_hashes"].get(p)==sha
            for p,sha in review["controller_input_hashes"].items()),
        "original prospectively reviewed unavailability/code binding required")
    return dict(reason=recipe.reason,review_path=recipe.review_path,
        review_sha256=hashlib.sha256(raw).hexdigest())


def validate_shared_availability(recipes):
    by_identity={}
    for (_,identity),recipe in recipes.items():
        state=(type(recipe) is ProspectivelyUnavailableStageRecipe,
               recipe.reason if type(recipe) is ProspectivelyUnavailableStageRecipe else None)
        require(identity not in by_identity or by_identity[identity]==state,
            "shared A1/A2 selector identity requires consistent prospective availability")
        by_identity[identity]=state


def require_unavailable_candidate_absence(ledger, configuration):
    """Also covers unavailable A3 candidates that selection never executes."""
    for slot in ledger.contract["panels"]["exploration"]["root_slots"]:
        token=digest(dict(root_id=slot["root_id"],configuration=configuration))
        require(not (ledger.directory/"selector-producer-claims"/(token+".json")).exists()
            and not (ledger.directory/"selector-runs"/token).exists(),
            "prospectively unavailable configuration acquired candidate work")


def read_unavailable_stage_evidence(ledger, stage, configuration, evidence):
    binding=evidence["prospective_configuration_unavailable"]
    require(type(binding) is dict and set(binding)=={"reason","review_path","review_sha256"},
        "exact prospective unavailability binding required")
    current=validate_unavailable_recipe(ledger,stage,configuration,
        ProspectivelyUnavailableStageRecipe(reason=binding["reason"],review_path=binding["review_path"]))
    require(digest(binding)==digest(current)
        and evidence["schema"]=="pokezero.search-over-raw.empty-panel-stage.v1"
        and evidence["stage"]==stage and evidence["configuration"]==configuration
        and evidence["contract_sha256"]==ledger.contract_sha256
        and evidence["status"]=="UNSTARTED" and evidence["selector_attempted"] is False
        and evidence.get("observed_zero_work") is False and evidence["accepted_units"]==0
        and not evidence["root_intervals"] and not evidence["root_evidence"]
        and "owned_input_handoff" not in evidence and "input_unavailable_by_root" not in evidence,
        "prospective unavailability cannot adopt outcomes, owners or observed zero work")
    require_unavailable_candidate_absence(ledger,configuration)
    return binding


def exploration_recipe_keys():
    """All prospective candidates; A3 execution still follows actual A2."""
    return tuple((stage, cfg.identity) for stage, configs in (
        ("A1", a1_configurations()), ("A2", a2_configurations()),
        ("A3", tuple(c for c in deployment_configurations() if c.seconds != 1.)))
        for cfg in configs)


def run_owned_exploration_stages(ledger, panel, *, recipes, checkpoint_contract,
        showdown_root, file_hashes, raw_runtime, history_review_path,
        continuation_config, continuation_runtime, set_source,
        source_config=None, source_review_path=None,
        _synthetic_selector_transport=None, _synthetic_continuation_job_class=None):
    """One irreversible staged run, with no directory-inferred resumption.

    Synthetic transport authors only extra fields on typed job subclasses;
    actual production preparation, supervision and independent readers still
    own acceptance. It is rejected for admitted scientific contracts. Zero
    accepted selector work is local, whereas integrity failures stop the run.
    """
    require(type(ledger) is ReconciledPhaseALedger
        and type(panel) is TrustedExplorationPanel and panel.ledger is ledger,
        "exact retained full exploration panel and original ledger required")
    started, cpu_started = time.monotonic(), time.process_time()
    synthetic = _synthetic_selector_transport is not None
    all_plans, all_banks, selectors, journals, banks_by_root = [], [], {}, {}, {}
    unavailable = {}
    unavailable_files = {}
    artifacts, reports = {}, {}
    launches = dict(raw=0, search=0, continuation=0, reused_selectors=0, reused_a3_records=0)
    try:
        ledger.verify()
        require(type(recipes) is dict and set(recipes) == set(exploration_recipe_keys())
            and all(type(r) in (StageSelectorRecipe,ProspectivelyUnavailableStageRecipe) for r in recipes.values()),
            "complete prospective A1/A2 and candidate A3 recipe registry required; no silent omissions")
        require(type(file_hashes) is dict and type(raw_runtime) is dict
            and type(continuation_runtime) is dict, "explicit common runtime/file recipes required")
        if synthetic:
            require(callable(_synthetic_selector_transport)
                and getattr(_synthetic_selector_transport, "synthetic_process_fixture", False) is True
                and ledger.contract["execution_ready"] is False
                and ledger.contract["collection_authorized"] is False
                and _synthetic_continuation_job_class is not None,
                "explicit unadmitted authored full-panel scope required")
        else:
            require(_synthetic_continuation_job_class is None,
                "authored continuation cannot enter production scheduler")
            ledger.verify_exploration("A1",a1_configurations()[0].identity)
            require(continuation_runtime.get("files",{}).get(continuation_runtime.get("checkpoint_path"))
                ==checkpoint_contract.checkpoint_sha256
                and str(continuation_config.resolved_showdown_root())==checkpoint_contract.showdown_root,
                "production continuations must retain source champion and simulator")
        require(all(ledger.contract["input_hashes"].get(p) == h
            for p, h in controller_input_hashes().items()), "staged controller code must be pinned")
        validate_shared_availability(recipes)
        for (stage,identity),recipe in recipes.items():
            if type(recipe) is ProspectivelyUnavailableStageRecipe:
                validate_unavailable_recipe(ledger,stage,identity,recipe)
        recipe_objects = tuple((key, id(value)) for key, value in recipes.items())
        def input_identity():
            return digest(dict(recipes=[dict(stage=k[0], configuration=k[1],
                recipe=_state_payload(v)) for k,v in recipes.items()],
                checkpoint=_state_payload(checkpoint_contract), showdown=showdown_root,
                files=file_hashes, raw=raw_runtime, history=str(Path(history_review_path).resolve()),
                continuation_config=_state_payload(continuation_config), continuation=continuation_runtime,
                source_config=_state_payload(source_config),source_review_path=str(source_review_path),
                catalog=_state_payload(set_source)))
        input_sha = input_identity()
        claim_path = ledger.directory / "staged-run-claim.json"
        _write_new(claim_path, dict(contract_sha256=ledger.contract_sha256,
            panel_inputs_sha256=panel.frozen, recipe_inputs_sha256=input_sha,
            synthetic_process_fixture=synthetic, retry_authorized=False,
            collection_authorized=False, holdout_authorized=False))
        claim_bytes = claim_path.read_bytes()
        def guard():
            ledger.verify()
            require(tuple((key,id(value)) for key,value in recipes.items()) == recipe_objects
                and input_identity() == input_sha and claim_path.read_bytes() == claim_bytes,
                "original staged recipes or create-only claim changed")
            panel._original_inputs()  # Callback-free global source identity check.
        panel.validate(); guard()

        def select(stage, cfg, root_id, role):
            source_owner = panel.by_root[root_id]
            spec = recipes[(stage, cfg.identity)]
            runtime = deepcopy(raw_runtime if role == "raw" else spec.runtime)
            if role == "search" and cfg.belief == "oracle":
                from .search_over_raw_archive import capture_archived_team_oracle
                oracle, _ = capture_archived_team_oracle(source_owner.archive, source_owner.source,
                    contract=ledger.contract, root_id=root_id, set_source=set_source)
                runtime["team_oracle"] = oracle.receipt()
                if cfg.arm == "reference": runtime["oracle_payload_sha256"] = digest(_state_payload(oracle))
                require(digest(selector_runtime_recipe(cfg,runtime))
                    == digest(selector_runtime_recipe(cfg,spec.runtime)), "root-specific oracle recipe drift")
                guard()
            key = (root_id, "raw" if role == "raw" else cfg.identity)
            prior = selectors.get(key)
            authored = {}
            if synthetic:
                authored = _synthetic_selector_transport(stage=stage, configuration=cfg,
                    role=role, root_id=root_id, source_owner=source_owner,
                    runtime=deepcopy(runtime), prior_owner=prior)
                require(type(authored) is dict and set(authored) ==
                    {"_synthetic_job_class", "_synthetic_job_kwargs"}, "typed authored transport fields only")
                guard()
                if RUNTIME_KEY in runtime:
                    require(authored["_synthetic_job_class"] is None and authored["_synthetic_job_kwargs"] is None,
                        "actor route requires actual authored work, not supplied selection packets")
                    authored["_synthetic_actual_actor"]=True
            owner, receipt = produce_archived_selector(ledger, stage, cfg.identity,
                archive=source_owner.archive, source=source_owner.source, source_owner=source_owner,
                root_id=root_id, checkpoint_contract=checkpoint_contract, showdown_root=showdown_root,
                runtime=runtime, file_hashes=file_hashes,
                qualification_review_path=spec.qualification_review_path,
                history_review_path=history_review_path, selector=role,
                reference_factory=spec.reference_factory if role == "search" else None,
                set_source=set_source if role == "search" and cfg.belief == "oracle" else None,
                prior_owner=prior,source_config=source_config,source_review_path=source_review_path, **authored)
            guard()
            if owner is None:
                require(role == "search" and RUNTIME_KEY in runtime and receipt["status"] == "INPUT_UNAVAILABLE",
                    "only typed prelaunch actor input gaps may omit an owner")
                read_actor_input_unavailable(ledger,source_owner,receipt,configuration=cfg)
                if key in unavailable: require(digest(unavailable[key]) == digest(receipt),
                    "earlier unavailable input cannot be replaced")
                unavailable[key]=receipt
                path=_actor_unavailable_path(ledger,root_id,cfg.identity)
                current=path.read_bytes()
                require(key not in unavailable_files or unavailable_files[key] == current,
                    "earlier unavailable receipt bytes changed")
                unavailable_files[key]=current
                guard()
                return None
            require(key not in unavailable,"unavailable actor input cannot later acquire a selector")
            require(prior is None or owner is prior, "reuse must retain the actual selector owner")
            selectors[key] = owner
            launches[role if receipt["selector_launched"] else "reused_selectors"] += 1
            return owner

        def empty_configuration(stage, cfg, gaps=None, prospective_unavailable=None):
            # No available root means no selector was attempted. It is NOT
            # observed zero-work at a ceiling and cannot qualify any arm.
            panel.validate(); guard()
            evidence = dict(schema="pokezero.search-over-raw.empty-panel-stage.v1",
                contract_sha256=ledger.contract_sha256, stage=stage, configuration=cfg.identity,
                root_intervals={}, root_evidence={}, accepted_units=0, source_panel=panel.receipt(),
                staged_run_claim_sha256=hashlib.sha256(claim_bytes).hexdigest(),
                status="UNSTARTED", selector_attempted=False, holdout_authorized=False)
            if gaps:
                panel.require_stage_inputs(ledger,{},input_unavailable=gaps,configuration=cfg)
                evidence["input_unavailable_by_root"]=deepcopy(gaps)
            if prospective_unavailable is not None:
                evidence["prospective_configuration_unavailable"]=deepcopy(prospective_unavailable)
                evidence["observed_zero_work"]=False
                read_unavailable_stage_evidence(ledger,stage,cfg.identity,evidence)
            guard()
            _write_new(ledger.directory / stage / f"{cfg.identity}-root-evidence.json", evidence)
            return ledger.record_stage(stage,cfg.identity,{},accepted_units=0,
                status="UNSTARTED",evidence_sha256=digest(evidence))

        def execute_configuration(stage, cfg):
            guard()
            recipe=recipes[(stage,cfg.identity)]
            if type(recipe) is ProspectivelyUnavailableStageRecipe:
                binding=validate_unavailable_recipe(ledger,stage,cfg.identity,recipe)
                return empty_configuration(stage,cfg,prospective_unavailable=binding)
            if not panel.by_root: return empty_configuration(stage,cfg)
            plans, supplied_banks, gaps = {}, {}, {}
            for root_id, source_owner in panel.by_root.items():
                raw, search = select(stage,cfg,root_id,"raw"), select(stage,cfg,root_id,"search")
                if search is None:
                    gaps[root_id]=unavailable[(root_id,cfg.identity)]
                    continue
                plan = ArchivedRootEvaluationPlan(source_owner.archive,source_owner.source,
                    contract=ledger.contract,root_id=root_id,
                    configurations=(SearchConfiguration("raw",seconds=1.),cfg),
                    selections={"raw":raw.selection,cfg.identity:search.selection},
                    selector_readers={"raw":raw.reader,cfg.identity:search.reader},
                    source_owner=source_owner,allow_synthetic=synthetic,
                    oracle_set_source=set_source if cfg.belief == "oracle" else None)
                guard(); all_plans.append(plan); plans[root_id] = plan
                # No continuation at all for an unqualified search selector;
                # in particular, do not consume a raw-only bank in its name.
                if search.selection["status"] != "UNQUALIFIED_AT_CEILING":
                    if root_id not in journals:
                        journals[root_id] = ContinuationAdmissionJournal.create(ledger,plan,
                            history_review_path=history_review_path)
                        guard()
                    bank, receipt = produce_reserved_continuation(ledger,stage,cfg.identity,
                        plan=plan,journal=journals[root_id],config=continuation_config,
                        runtime=continuation_runtime,
                        _synthetic_job_class=_synthetic_continuation_job_class)
                    guard()
                    if bank is not None:
                        all_banks.append(bank); banks_by_root.setdefault(root_id,[]).append(bank)
                    launches["continuation"] += int(receipt["continuation_launched"])
                supplied_banks[root_id] = tuple(banks_by_root.get(root_id,())) or None
            if not plans:return empty_configuration(stage,cfg,gaps)
            extra = {"input_unavailable":gaps} if gaps else {}
            artifact = ledger.record_owned_stage_evidence(stage,cfg.identity,plans=plans,
                continuations=supplied_banks,qualification_review_path=recipes[(stage,cfg.identity)].qualification_review_path,
                source_panel=panel,**extra)
            guard()
            return artifact

        for stage in ("A1","A2"):
            for cfg in ledger._stage_configs(stage):
                artifacts[(stage,cfg.identity)] = execute_configuration(stage,cfg)
            reports[stage] = ledger.report_stage(stage); guard()
        evaluator_freeze = ledger.freeze_evaluators(); guard()
        for cfg in ledger._stage_configs("A3"):
            if cfg.seconds == 1.:
                prior = artifacts[("A2",cfg.identity)]
                artifact = ledger.record_stage("A3",cfg.identity,prior["root_intervals"],
                    accepted_units=prior["summary"]["accepted_units"],status=prior["summary"]["status"],
                    evidence_sha256=prior["evidence_sha256"])
                launches["reused_a3_records"] += 1
            else: artifact = execute_configuration("A3",cfg)
            artifacts[("A3",cfg.identity)] = artifact; guard()
        reports["A3"] = ledger.report_stage("A3"); guard()
        # Validate every retained earlier owner, then take callback-free global
        # inventory after the last callback. A later root cannot rewrite an
        # already accepted source, selector, or continuation bank.
        frozen_owners = _owner_inputs(tuple(all_plans),tuple(all_banks))
        for (root_id,_), owner in selectors.items():
            source_owner=panel.by_root[root_id]
            owner.validate(source_owner.archive,source_owner.source); guard()
        def unavailable_identity():
            return digest([[root,cfg,receipt] for (root,cfg),receipt in sorted(unavailable.items())])
        unavailable_sha=unavailable_identity()
        for (root_id,cfgid),receipt in unavailable.items():
            read_actor_input_unavailable(ledger,panel.by_root[root_id],receipt)
            guard()
        for plan in all_plans: plan.revalidate(); guard()
        for bank in all_banks:
            bank.plan.accept_continuation_result(bank.job,bank.result,owning_reader=bank.reader); guard()
        panel.validate(); guard()
        require(_owner_inputs(tuple(all_plans),tuple(all_banks)) == frozen_owners,
            "earlier owned evidence changed during final staged callbacks")
        require(all(owner._inputs() == owner.frozen for owner in selectors.values()),
            "earlier retained selector changed during final staged callbacks")
        require(unavailable_identity()==unavailable_sha,"earlier unavailable inputs changed during final staged callbacks")
        for stage in ("A1","A2","A3"):
            require(digest(ledger.report_stage(stage)) == digest(reports[stage]),
                "earlier stage report changed during later callbacks")
        panel_receipt=panel.receipt(); guard()
        selection = ledger.freeze(); guard()
        # No callback-bearing operation follows this global check. In
        # particular, freeze/report/source callbacks cannot create candidate
        # work or rewrite a previously accepted unavailable receipt.
        require(unavailable_identity()==unavailable_sha and set(unavailable_files)==set(unavailable),
            "earlier unavailable partition changed after staged freeze")
        for (root,cfg),raw in unavailable_files.items():
            path=_actor_unavailable_path(ledger,root,cfg)
            claim=path.with_name(path.name.replace("-input-unavailable.json",".json"))
            directory=ledger.directory/"selector-runs"/digest(dict(root_id=root,configuration=cfg))
            require(path.read_bytes()==raw and not claim.exists() and not directory.exists(),
                "unavailable input bytes or candidate-attempt absence changed after staged freeze")
        for (stage,identity),recipe in recipes.items():
            if type(recipe) is ProspectivelyUnavailableStageRecipe:
                validate_unavailable_recipe(ledger,stage,identity,recipe)
                require_unavailable_candidate_absence(ledger,identity)
                path=ledger.directory/stage/(identity+"-root-evidence.json")
                if path.exists():
                    from .search_over_raw_ledger import _load
                    read_unavailable_stage_evidence(ledger,stage,identity,_load(path))
        return dict(status="STAGED_EXPLORATION_RECORDED", reports=reports,
            evaluator_freeze=evaluator_freeze, selection=selection, launches=launches,
            source_panel=panel_receipt, synthetic_process_fixture=synthetic,
            input_unavailable_by_root_configuration={root+":"+cfg:deepcopy(value)
                for (root,cfg),value in unavailable.items()},
            elapsed_seconds=time.monotonic()-started, auditor_cpu_seconds=time.process_time()-cpu_started,
            resource_cost_complete=False, cost_scope="Auditor wall time and own CPU only; nested costs not summed.",
            private_archive_returned=False, runtime_qualified_by_this_code=False,
            collection_authorized=False, holdout_authorized=False, phase_b_authorized=False)
    except BaseException as error:
        if (ledger.directory / "staged-run-claim.json").exists() and not (ledger.directory / "operational-stop.json").exists():
            try: ledger.stop(type(error).__name__)
            except BaseException as stop_error: error.add_note("Unable to persist staged stop: "+type(stop_error).__name__)
        raise
    finally:
        for owner in panel.owners: owner.close()
