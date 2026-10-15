"""Private consumed-native constructor planning; no redraws or source games."""
from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json
import pickle
from pathlib import Path

from .search_over_raw import digest,require
from .search_over_raw_fixed_work_run import FixedWorkRunValidator
from .search_over_raw_construction import read_native_construction,construction_source_hashes
from .search_over_raw_materialization import BOOL_OPTIONS,MAP_OPTIONS,recorded_materialization_inputs
from .search_over_raw_materialization_run import OwnedRecordedMaterializationJob
from .search_over_raw_outcome_run import verify_outcome_runtime
from .search_over_raw_reconciled import controller_input_hashes
from .search_over_raw_world_inputs import _state_payload
from .search_over_raw_outcomes import _sha
from .search_over_raw_native_leaf_cells import leaf_cell_law, freeze_owned_native_leaf_cells


def _private_json(value):
    return json.dumps(value,sort_keys=True,separators=(",",":"),ensure_ascii=False,allow_nan=False)


@dataclass(frozen=True)
class OwnedNativeMaterializationPlan:
    payload_utf8: str

    def receipt(self):
        require(type(self.payload_utf8) is str,"private canonical materialization plan bytes required")
        value=json.loads(self.payload_utf8)
        require(_private_json(value)==self.payload_utf8,"canonical materialization plan required")
        return value

    @property
    def sha256(self):return hashlib.sha256(self.payload_utf8.encode()).hexdigest()


def plan_owned_native_materializations(native_selected,*,native_validator,config,runtime,
        options,attempt_root,public_state_sha256,paired_context=None):
    """All four ACTUAL constructions; original chance seed is transport only.

    The original full public request and captured overlay remain different
    bindings. Neither a seed match nor observed constructor input matching
    establishes common RNG, physics or full-engine equivalence.
    """
    require(type(native_validator) is FixedWorkRunValidator
        and native_validator.configuration.arm=="incumbent" and native_validator.configuration.leaf=="model"
        and _sha(native_validator.public_request_sha256) and _sha(public_state_sha256),
        "original full-public native model owning reader required")
    attempt_root=Path(attempt_root)
    require(attempt_root.is_absolute() and attempt_root==attempt_root.resolve(),"original canonical materialization namespace required")
    require(type(options) is dict and set(options)==set(BOOL_OPTIONS)|set(MAP_OPTIONS)
        and all(type(options[k]) is bool for k in BOOL_OPTIONS)
        and all(options[k] is None or type(options[k]) is dict for k in MAP_OPTIONS)
        and options["deferred_opponent_actions"] is None and options["deferred_opponent_action_priors"] is None,
        "prospective complete constructor options; deferred fields come from ACTUAL captured overlay")
    selected_sha=digest(native_selected);config_sha=digest(_state_payload(config));runtime_sha=verify_outcome_runtime(runtime,config)
    require(str(config.resolved_showdown_root())==native_validator.showdown_root
        and runtime["checkpoint_path"]==native_validator.checkpoint_path
        and runtime["files"][runtime["checkpoint_path"]]==native_validator.checkpoint_sha,
        "planned native materializations must retain original champion/simulator")
    option_sha=digest(options);sources=controller_input_hashes()
    require(native_validator(native_selected)==selected_sha,"independently owned original native diagnostic required")
    original_readout=deepcopy(native_validator.last_validation)
    from .search_over_raw_attempt_cost import _inventory
    original_files=_inventory([native_validator.directory])
    manifest=native_selected["evidence"]["consumed_world_inputs"]
    rows=sorted(manifest["rows"],key=lambda row:row["ordinal"])
    calls={row["call_ordinal"]:row for row in manifest["calls"]}
    require([r["ordinal"] for r in rows]==list(range(4)),"all four original native construction ordinals required")
    records=[]
    for row in rows:
        ordinal=row["ordinal"];call=calls[row["call_ordinal"]]
        checked=read_native_construction(row["construction_receipt"],binding=native_validator.consumed,
            ordinal=ordinal,draw_sha256=row["draw_sha256"],constructed_sha256=row["constructed_sha256"],
            state_utf8_sha256=row["native_state_utf8_sha256"],source_hashes=construction_source_hashes())
        world=row["construction_receipt"]["events"][0]["value"]
        seed=call["native_call_parameters"]["chance_seed"]
        require(type(seed) is int,"actual original native chance seed required")
        actual_options=deepcopy(options)
        actual_options["deferred_opponent_actions"]=deepcopy(world["payload"].get("deferredOpponentActions",{}))
        actual_options["deferred_opponent_action_priors"]=deepcopy(world["payload"].get("deferredOpponentActionPriors",{}))
        origin=dict(schema="pokezero.search-over-raw.recorded-native-materialization.v1",
            native_selected_sha256=selected_sha,construction_capture_sha256=checked["capture_sha256"],
            construction_ordinal=ordinal,original_public_request_sha256=native_validator.public_request_sha256,
            draw_sha256=digest(world["override"]),public_payload_sha256=digest(world["payload"]))
        inputs=recorded_materialization_inputs(draw=world["override"],public_payload=world["payload"],seed=seed,
            root_binding=native_validator.consumed,public_state_sha256=public_state_sha256,
            recorded_native_origin=origin,options=actual_options)
        unavailable=None
        if paired_context is not None:
            from .search_over_raw_paired_materialization import OwnedPairedMaterializationJob,ContinuationContextUnavailable
            require(digest(_state_payload(paired_context))==public_state_sha256,"original paired public context required")
            job=OwnedPairedMaterializationJob(deepcopy(config),inputs,deepcopy(runtime),deepcopy(paired_context))
            try:expected=job.input_binding()
            except ContinuationContextUnavailable as missing:
                unavailable=missing.reason;expected=dict(materialization_inputs=inputs,runtime_sha256=runtime_sha)
        else:
            job=OwnedRecordedMaterializationJob(deepcopy(config),inputs,deepcopy(runtime));expected=job.input_binding()
        path=attempt_root/("native-materialization-"+digest(dict(native=selected_sha,ordinal=ordinal,inputs=expected)))
        require(path==path.resolve(),"materialization attempt namespace redirected")
        records.append(dict(construction_ordinal=ordinal,native_call_ordinal=row["call_ordinal"],
            collapsed_construction_ordinals=deepcopy(call["construction_ordinals"]),native_call_multiplicity=call["multiplicity"],
            attempt_directory=str(path),status=unavailable or ("ATTEMPT_ALREADY_CLAIMED" if path.exists() else "PLANNED_NOT_ADMITTED"),
            expected_inputs_sha256=digest(expected),recorded_inputs=inputs))
    require(native_validator(native_selected)==selected_sha and digest(native_validator.last_validation)==digest(original_readout)
        and digest(native_selected)==selected_sha and _inventory([native_validator.directory])==original_files
        and config_sha==digest(_state_payload(config)) and runtime_sha==verify_outcome_runtime(runtime,config)
        and digest(options)==option_sha and controller_input_hashes()==sources,
        "original native/materialization planning inputs changed during callbacks")
    # This is after the LAST reader/runtime callback. Do not adopt a redirected
    # namespace or a newly occupied attempt as a replacement baseline.
    require(attempt_root==attempt_root.resolve() and all(Path(r["attempt_directory"])==Path(r["attempt_directory"]).resolve()
        and (r["status"]=="CONSUMED_OVERLAY_NOT_REPRESENTED_BY_ORIGINAL_CONTEXT" or
            Path(r["attempt_directory"]).exists()==(r["status"]=="ATTEMPT_ALREADY_CLAIMED")) for r in records),
        "original materialization paths or occupancy changed during final callbacks")
    return OwnedNativeMaterializationPlan(_private_json(dict(
        schema="pokezero.search-over-raw.private-native-materialization-plan.v2" if paired_context is not None else
            "pokezero.search-over-raw.private-native-materialization-plan.v1",private_controller_only=True,
        native_selected_sha256=selected_sha,native_job_sha256=native_validator.job_sha,
        full_public_request_sha256=native_validator.public_request_sha256,public_state_sha256=public_state_sha256,
        runtime_sha256=runtime_sha,config_sha256=config_sha,options=deepcopy(options),controller_sources=sources,
        attempt_root=str(attempt_root),records=records,required_construction_ordinals=4,
        actual_native_calls=len(calls),source_generation=False,world_redraw=False,process_launches=0,
        seed_law="actual native invocation chance_seed; not shared RNG equivalence",
        full_world_equivalence=False,runtime_qualified=False,scientific_collection_authorized=False,retries_permitted=False)))


def build_owned_native_materialization_jobs(plan,native_selected,*,native_validator,config,runtime,options,paired_context=None):
    require(type(plan) is OwnedNativeMaterializationPlan,"original immutable materialization plan required")
    saved=plan.receipt();frozen=plan.sha256
    def check():
        fresh=plan_owned_native_materializations(native_selected,native_validator=native_validator,
            config=config,runtime=runtime,options=options,attempt_root=saved["attempt_root"],
            public_state_sha256=saved["public_state_sha256"],paired_context=paired_context)
        require(fresh.payload_utf8==plan.payload_utf8 and frozen==plan.sha256,
            "materialization plan differs from original owning run or claims")
    check();jobs=[]
    for row in saved["records"]:
        if row["status"]!="PLANNED_NOT_ADMITTED":continue
        if paired_context is not None:
            from .search_over_raw_paired_materialization import OwnedPairedMaterializationJob
            job=OwnedPairedMaterializationJob(deepcopy(config),deepcopy(row["recorded_inputs"]),deepcopy(runtime),deepcopy(paired_context))
        else:job=OwnedRecordedMaterializationJob(deepcopy(config),deepcopy(row["recorded_inputs"]),deepcopy(runtime))
        require(digest(job.input_binding())==row["expected_inputs_sha256"],"planned constructor job differs from originals")
        jobs.append(dict(construction_ordinal=row["construction_ordinal"],attempt_directory=row["attempt_directory"],
            plan_sha256=frozen,job=job))
    check();return jobs


def native_chain_review_template(contract,*,config,runtime,options,helper_files,preserve_prefix=False,paired_labels=False):
    """Review the prospective recipe, not outcome-informed per-root jobs."""
    require(type(preserve_prefix) is bool and type(paired_labels) is bool and (not paired_labels or preserve_prefix),
        "explicit prospective paired labels require prefix-preserving replay")
    from .search_over_raw_native_prefix import prefix_law
    from .search_over_raw_paired_chain import paired_chain_law
    return dict(schema="pokezero.search-over-raw.native-chain-review.v3" if paired_labels else
            "pokezero.search-over-raw.native-chain-review.v2" if preserve_prefix else
            "pokezero.search-over-raw.native-chain-review.v1",
        **(dict(replay_prefix_law=prefix_law()) if preserve_prefix else {}),
        **(dict(paired_chain_law=paired_chain_law()) if paired_labels else {}),
        contract_design_sha256=digest({k:v for k,v in contract.items() if k!="input_hashes"}),
        config_sha256=digest(_state_payload(config)),runtime_sha256=digest(runtime),options=deepcopy(options),
        helper_files=deepcopy(helper_files),materialization_count=4,
        materialization_seed_law="actual native invocation chance_seed; not shared RNG equivalence",
        original_consumed_draw_and_overlay_only=True,complete_sampled_coordinate_roster=True,
        retained_after_leaf_cell_law=paired_chain_law()["leaf_cells"] if paired_labels else leaf_cell_law(),
        status="REVIEW_REQUIRED",independent_review_completed=False,reviewed_before_first_source_work=False,
        collection_authorized=False,holdout_authorized=False,retries_permitted=False)


def produce_owned_native_diagnostic_chain(ledger,native_owner,*,config,runtime,options,helper_files,review_path,
        preserve_prefix=False,paired_labels=False,_synthetic_helpers=None,_synthetic_job_transport=None):
    """Actual constructor/replay chain on one original live diagnostic owner.

    The collection caller must retain its panel through this function. No new
    source, sampler, scientific admission, fallback action or retry is provided.
    Recipe review is fixed before source work; observed coordinates merely bind
    dynamic one-shot jobs to that already reviewed recipe.
    """
    from .search_over_raw_diagnostic_producer import OwnedFixedWorkDiagnostic
    from .search_over_raw_reconciled import ReconciledPhaseALedger,a1_configurations
    from .search_over_raw_ledger import _write_new
    from .search_over_raw_attempt_cost import _inventory,AttemptCostSpec,_attempt
    from .search_over_raw_nested_ownership import current_scope
    from .search_over_raw_resources import resource_source_hash
    from .search_over_raw_materialization_run import (MaterializationRunContract,MaterializationRunValidator,
        supervise_materialization_run)
    from .search_over_raw_boundary_run import (NativePathBoundaryRunContract,NativePathBoundaryRunValidator,
        OwnedNativePathBoundaryJob,supervise_native_path_boundary_run)
    from .search_over_raw_native_prefix import (NativePrefixBoundaryRunContract, NativePrefixBoundaryRunValidator,
        supervise_native_prefix_boundary_run)
    from .search_over_raw_consumed_materialization import (plan_owned_native_replays,build_owned_native_replay_jobs,
        reconcile_archived_exploration_diagnostic,_original_diagnostic_owner_guard)
    from dataclasses import fields
    require(type(ledger) is ReconciledPhaseALedger and type(native_owner) is OwnedFixedWorkDiagnostic,
        "original private native diagnostic/ledger required")
    native_owner.validate(ledger)
    job=native_owner.job
    require(job.configuration.arm=="incumbent" and job.configuration.leaf=="model"
        and job.native_ancestry_contract is not None and job.native_ancestry_contract.capture_native_edges is True,
        "actual prospectively sampled native model diagnostic required")
    synthetic=_synthetic_helpers is not None or _synthetic_job_transport is not None
    if synthetic:
        require(_synthetic_helpers is not None and _synthetic_job_transport is not None
            and getattr(_synthetic_helpers,"synthetic_process_fixture",False) is True
            and callable(_synthetic_job_transport) and getattr(_synthetic_job_transport,"synthetic_process_fixture",False) is True
            and ledger.contract["execution_ready"] is False and ledger.contract["collection_authorized"] is False,
            "authored native chains cannot enter admitted scientific work")
    else:ledger.verify_exploration("A1",a1_configurations()[0].identity)
    verify_outcome_runtime(runtime,config)
    require(runtime["checkpoint_path"]==job.checkpoint_contract.checkpoint_path
        and runtime["files"][runtime["checkpoint_path"]]==job.checkpoint_contract.checkpoint_sha256
        and str(config.resolved_showdown_root())==job.showdown_root==job.checkpoint_contract.showdown_root,
        "native diagnostic chain must preserve original champion and simulator")
    require(type(helper_files) is dict and helper_files and all(type(p) is str and str(Path(p).resolve())==p
        and _sha(h) and hashlib.sha256(Path(p).read_bytes()).hexdigest()==h for p,h in helper_files.items()),
        "prospective native helper file bindings required")
    review_path=Path(review_path).resolve();review_bytes=review_path.read_bytes();review=json.loads(review_bytes)
    expected_review=native_chain_review_template(ledger.contract,config=config,runtime=runtime,options=options,
        helper_files=helper_files,preserve_prefix=preserve_prefix,paired_labels=paired_labels)
    expected_review.update(status="CLEAR_NATIVE_DIAGNOSTIC_CHAIN_RECIPE",independent_review_completed=True,
        reviewed_before_first_source_work=True)
    require(digest(review)==digest(expected_review) and ledger.contract["input_hashes"].get(str(review_path))
        ==hashlib.sha256(review_bytes).hexdigest(),"original prospective native-chain recipe review required")
    if synthetic:
        extract=_synthetic_helpers.extract;support=getattr(_synthetic_helpers,"support",None)
        successor=getattr(_synthetic_helpers,"successor",None)
    else:
        import pokezero_search
        module_path=str(Path(pokezero_search.__file__).resolve())
        require(module_path in helper_files and pokezero_search.MODEL_FEATURE_ENABLED is True,
            "original declared model-native helper binary required")
        extract=pokezero_search.native_scoped_boundary;support=pokezero_search.native_edge_support
        successor=pokezero_search.native_scoped_successor_support
    require(callable(extract) and (support is None or callable(support)) and (successor is None or callable(successor)),
        "original scoped native helper interfaces required")
    helper_objects=(extract,support,successor)
    sources=controller_input_hashes();recipe_sha=digest(expected_review)
    require(all(ledger.contract["input_hashes"].get(p)==h for p,h in sources.items()),"native chain controller sources must be pinned")
    root=ledger.directory/"native-diagnostic-chains"/digest(job.statistics_root_id)
    require(root==root.resolve(),"original native-chain namespace cannot be redirected")
    materializations=[];replays=[];costs=[];originals={};materialization_plan=None;replay_plan=None;leaf_cell_plan=None
    descriptor_guards=[];owner_guards=[]
    claim_bytes=None;replay_bytes=None;materialization_plan_bytes=None;replay_plan_bytes=None;leaf_cell_plan_bytes=None
    reservation_bytes={};materialization_objects=[];replay_objects=[]
    def track_jobs(descriptors):
        for d in descriptors:
            descriptor_guards.append((d,d["job"],digest({k:v for k,v in d.items() if k!="job"}),
                hashlib.sha256(pickle.dumps(d["job"])).hexdigest(),digest(d["job"].input_binding())))
    def guard():
        ledger.verify();native_owner._original_inputs()
        require(root==root.resolve() and all(Path(d["attempt_directory"])==Path(d["attempt_directory"]).resolve()
            for d,_,_,_,_ in descriptor_guards),"original native-chain namespace/attempt path redirected")
        require(native_owner.job is job and recipe_sha==digest(native_chain_review_template(ledger.contract,
            config=config,runtime=runtime,options=options,helper_files=helper_files,preserve_prefix=preserve_prefix,
            paired_labels=paired_labels)|dict(
                status="CLEAR_NATIVE_DIAGNOSTIC_CHAIN_RECIPE",independent_review_completed=True,reviewed_before_first_source_work=True))
            and review_path.read_bytes()==review_bytes and controller_input_hashes()==sources
            and all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==h for p,h in helper_files.items()),
            "original native chain recipe/review/helper bytes changed")
        verify_outcome_runtime(runtime,config)
        require(all(_inventory([p])==frozen for p,frozen in originals.items()),"earlier native-chain attempt evidence changed")
        for descriptor,original,metadata_sha,sha,inputs_sha in descriptor_guards:
            require(descriptor["job"] is original and digest({k:v for k,v in descriptor.items() if k!="job"})==metadata_sha
                and hashlib.sha256(pickle.dumps(original)).hexdigest()==sha
                and digest(original.input_binding())==inputs_sha,"earlier or future planned native-chain job changed")
        for original_guard,readouts in owner_guards:original_guard(readouts)
        require(len(materializations)==len(materialization_objects) and all(a is b for a,b in zip(materializations,materialization_objects))
            and len(replays)==len(replay_objects) and all(a is b for a,b in zip(replays,replay_objects)),
            "original native-chain result wrappers replaced or reordered")
        reservations=root/"reservations"
        if root.exists() or reservation_bytes:
            require(reservations.is_dir() and reservations==reservations.resolve()
                and {p.name for p in reservations.iterdir()}==set(reservation_bytes)
                and all((reservations/name)==(reservations/name).resolve()
                    and (reservations/name).read_bytes()==original for name,original in reservation_bytes.items()),
                "original native-chain reservation bytes or roster changed")
        require(claim_bytes is None or (root/"claim.json").read_bytes()==claim_bytes,"original native-chain claim changed")
        require(replay_bytes is None or (root/"replay-plan.json").read_bytes()==replay_bytes,"original replay plan bytes changed")
        require(materialization_plan_bytes is None or materialization_plan.payload_utf8==materialization_plan_bytes,
            "original private materialization plan replaced")
        require(replay_plan_bytes is None or replay_plan.payload_utf8==replay_plan_bytes,"original private replay plan replaced")
        require(leaf_cell_plan_bytes is None or leaf_cell_plan.payload_utf8==leaf_cell_plan_bytes,
            "original retained leaf-cell plan replaced")
    def admit():
        guard();native_owner.validate(ledger);guard()
        if not synthetic:ledger.verify_exploration("A1",a1_configurations()[0].identity)
    def authored(role,original):
        if not synthetic:return original
        extra=_synthetic_job_transport(role=role,job=original,native_owner=native_owner)
        base=type(original)
        require(type(extra) is dict and set(extra)=={"job_class","job_kwargs"}
            and isinstance(extra["job_class"],type) and issubclass(extra["job_class"],base)
            and getattr(extra["job_class"],"synthetic_process_fixture",False) is True
            and type(extra["job_kwargs"]) is dict and not set(extra["job_kwargs"]) & {f.name for f in fields(base)},
            "authored chain adds fields only to typed original job")
        value=extra["job_class"](**{f.name:getattr(original,f.name) for f in fields(base)},**extra["job_kwargs"])
        require(digest(value.input_binding())==digest(original.input_binding()),"authored chain changed prospective inputs")
        guard();return value
    def run(role,descriptor,contract,reader_cls,supervise):
        guard()
        original=descriptor["job"];expected=deepcopy(original.input_binding());running=authored(role,original)
        job_sha=hashlib.sha256(pickle.dumps(running)).hexdigest();path=Path(descriptor["attempt_directory"])
        binding=digest(dict(recipe_sha256=recipe_sha,plan_sha256=descriptor["plan_sha256"],
            coordinate={k:descriptor[k] for k in ("construction_ordinal","evaluation_ordinal","event_id") if k in descriptor},
            job_sha256=job_sha,expected=expected))
        reader_kwargs=dict(directory=path,job_sha256=job_sha,binding_sha256=binding,expected_inputs=expected,
            runtime=runtime,allow_synthetic=synthetic)
        if role=="replay":reader_kwargs["expected_raw_root"]=deepcopy(original.raw_root)
        if role=="labels":reader_kwargs.update(root_binding=original.root_binding,expected_snapshot=original.snapshot)
        # Paired readers pin metadata from the actual original returned worker,
        # never from a caller's predicted snapshot or rehashed durable wrapper.
        reader=None
        def verify():
            admit()
            require(hashlib.sha256(pickle.dumps(running)).hexdigest()==job_sha
                and digest(running.input_binding())==digest(expected),"original native-chain job changed before/after dispatch")
        scope=current_scope()
        kind,work=("materialization-run","materialization") if role=="materialization" else (
            ("outcome-run","continuations") if role=="labels" else ("boundary-run","boundaries"))
        spec=AttemptCostSpec(str(path),job_sha,binding,kind,work,contract.receipt(),resource_source_hash(),
            contract.emergency_watchdog_seconds,None if scope is None else str(scope.directory))
        reservation_path=root/"reservations"/(digest(dict(path=str(path),binding=binding))+".json")
        require(reservation_path==reservation_path.resolve(),"original reservation namespace redirected")
        _write_new(reservation_path,dict(
            role=role,attempt=vars(spec),plan_sha256=descriptor["plan_sha256"],retry_authorized=False))
        reservation_bytes[reservation_path.name]=reservation_path.read_bytes()
        guard()
        terminal=supervise(running,directory=path,binding_sha256=binding,verify_admission=verify,
            # The wrapper already independently checks its strict capture/input
            # transport law. Full ownership needs terminal.json, which does not
            # exist until this supervisor returns; it is checked below.
            validate_result=lambda _:True,contract=contract)
        verify();cost,_,observed=_attempt(spec);costs.append(cost)
        require(digest(observed)==digest(terminal) and terminal["supervisor_exit_code"]==0,
            "native-chain failed attempt remains unscored; no retry or fallback")
        result=terminal["result"]
        if paired_labels and role=="materialization":
            reader_kwargs.update(original_public_context=original.original_public_context,expected_context=result["paired_context"])
        if paired_labels and role=="replay":
            reader_kwargs.update(snapshot=original.snapshot,expected_endpoint=result["paired_endpoint"])
        reader=reader_cls(**reader_kwargs)
        capture=result["bank"] if role=="labels" else result["capture"]
        require(reader(capture)==digest(capture),"actual independent native-chain owning reader required")
        originals[path]=_inventory([path]);guard()
        return dict(**{k:descriptor[k] for k in ("construction_ordinal","evaluation_ordinal","event_id") if k in descriptor},
            result=result,validator=reader)
    try:
        admit()
        materialization_plan=plan_owned_native_materializations(native_owner.selected,native_validator=native_owner.reader,
            config=config,runtime=runtime,options=options,attempt_root=root/"materializations",
            public_state_sha256=digest(_state_payload(job.context.public_materialization_state)),
            paired_context=job.context.public_materialization_state if paired_labels else None)
        materialization_jobs=build_owned_native_materialization_jobs(materialization_plan,native_owner.selected,
            native_validator=native_owner.reader,config=config,runtime=runtime,options=options,
            paired_context=job.context.public_materialization_state if paired_labels else None)
        require(all(r["status"] in {"PLANNED_NOT_ADMITTED","CONSUMED_OVERLAY_NOT_REPRESENTED_BY_ORIGINAL_CONTEXT"}
            for r in materialization_plan.receipt()["records"]) if paired_labels else len(materialization_jobs)==4,
            "occupied original constructor attempt cannot be resumed or replaced")
        materialization_plan_bytes=materialization_plan.payload_utf8;track_jobs(materialization_jobs)
        leaf_cell_plan=freeze_owned_native_leaf_cells(native_owner,ledger=ledger,paired_labels=paired_labels)
        leaf_cell_plan_bytes=leaf_cell_plan.payload_utf8
        root.mkdir(parents=True,exist_ok=False,mode=0o700)
        (root/"reservations").mkdir(mode=0o700)
        _write_new(root/"claim.json",dict(recipe_sha256=recipe_sha,native_selected_sha256=native_owner.selected_sha256,
            materialization_plan=materialization_plan.receipt(),materialization_plan_sha256=materialization_plan.sha256,
            leaf_cell_plan=leaf_cell_plan.receipt(),leaf_cell_plan_sha256=leaf_cell_plan.sha256,
            retry_authorized=False))
        claim_bytes=(root/"claim.json").read_bytes()
        from .search_over_raw_paired_materialization import (PairedMaterializationRunContract,PairedMaterializationRunValidator,
            supervise_paired_materialization_run)
        from .search_over_raw_paired_replay import PairedPrefixRunContract,PairedPrefixRunValidator,supervise_paired_prefix_run
        for descriptor in materialization_jobs:
            materializations.append(run("materialization",descriptor,
                PairedMaterializationRunContract() if paired_labels else MaterializationRunContract(),
                PairedMaterializationRunValidator if paired_labels else MaterializationRunValidator,
                supervise_paired_materialization_run if paired_labels else supervise_materialization_run))
            materialization_objects.append(materializations[-1])
            frozen_rows=list(materializations)
            original_guard=_original_diagnostic_owner_guard(native_owner.selected,frozen_rows,[],native_owner.reader)
            readouts={str(r.directory):deepcopy(r.last_validation) for r in
                [native_owner.reader]+[row["validator"] for row in frozen_rows]}
            owner_guards.append((original_guard,readouts));guard()
        replay_plan=plan_owned_native_replays(native_owner.selected,materializations,native_validator=native_owner.reader,
            ancestry_contract=job.native_ancestry_contract,native_extract=extract,config=config,attempt_root=root/"replays",
            preserve_prefix=preserve_prefix,paired_contexts=paired_labels)
        replay_jobs=build_owned_native_replay_jobs(replay_plan,native_owner.selected,materializations,
            native_validator=native_owner.reader,ancestry_contract=job.native_ancestry_contract,native_extract=extract,config=config)
        _write_new(root/"replay-plan.json",dict(private_plan=replay_plan.receipt(),plan_sha256=replay_plan.sha256))
        replay_bytes=(root/"replay-plan.json").read_bytes()
        replay_plan_bytes=replay_plan.payload_utf8;track_jobs(replay_jobs)
        for descriptor in replay_jobs:
            replays.append(run("replay",descriptor,
                PairedPrefixRunContract() if paired_labels else NativePrefixBoundaryRunContract() if preserve_prefix else NativePathBoundaryRunContract(),
                PairedPrefixRunValidator if paired_labels else NativePrefixBoundaryRunValidator if preserve_prefix else NativePathBoundaryRunValidator,
                supervise_paired_prefix_run if paired_labels else supervise_native_prefix_boundary_run if preserve_prefix else supervise_native_path_boundary_run))
            replay_objects.append(replays[-1])
            frozen_replays=list(replays)
            original_guard=_original_diagnostic_owner_guard(native_owner.selected,list(materializations),frozen_replays,native_owner.reader)
            readouts={str(r.directory):deepcopy(r.last_validation) for r in
                [native_owner.reader]+[row["validator"] for row in materializations+frozen_replays]}
            owner_guards.append((original_guard,readouts));guard()
        original_guard=_original_diagnostic_owner_guard(native_owner.selected,materializations,replays,native_owner.reader)
        label_runs=[];label_reports=[]
        if paired_labels:
            from .search_over_raw_paired_chain import paired_label_descriptors,paired_label_report
            from .search_over_raw_paired_labels import (PairedLabelRunContract,PairedLabelRunValidator,
                supervise_paired_label_run)
            descriptors=paired_label_descriptors(leaf_cell_plan,replays,config=config,runtime=runtime,attempt_root=root/"labels")
            guard()
            for descriptor in descriptors:descriptor["plan_sha256"]=leaf_cell_plan.sha256
            track_jobs([d for d in descriptors if d["job"] is not None])
            for descriptor in descriptors:
                if descriptor["job"] is None:label_reports.append(paired_label_report(descriptor))
                else:
                    owned=run("labels",descriptor,PairedLabelRunContract(),PairedLabelRunValidator,supervise_paired_label_run)
                    label_runs.append(owned);label_reports.append(paired_label_report(descriptor,owned));guard()
        result=reconcile_archived_exploration_diagnostic(native_owner.selected,materializations,replays,
            archive=native_owner.source_owner.archive,source=native_owner.source_owner.source,contract=ledger.contract,
            root_id=job.statistics_root_id,native_validator=native_owner.reader,ancestry_contract=job.native_ancestry_contract,
            native_extract=extract,native_support=support,native_successor_support=successor)
        readouts={row["owner"]["directory"]:deepcopy(row["owner"]["readout"]) for row in result["owning_jobs"]}
        native_owner.validate(ledger);original_guard(readouts);guard()
        require((root/"claim.json").read_bytes()==claim_bytes and (root/"replay-plan.json").read_bytes()==replay_bytes
            and (extract,support,successor)==helper_objects,"original private native-chain plans/helpers changed")
        report=dict(result,materialization_plan_sha256=materialization_plan.sha256,replay_plan_sha256=replay_plan.sha256,
            leaf_cell_plan_sha256=leaf_cell_plan.sha256,
            retained_unique_leaf_events=leaf_cell_plan.receipt()["retained_unique_evaluation_events"],
            required_leaf_label_cells=leaf_cell_plan.receipt()["required_label_cells"],
            leaf_label_process_launches=len(label_runs),completed_leaf_label_cells=sum(c["status"]=="COMPLETE"
                for row in label_reports for c in row["cells"]),
            actual_constructor_runs=len(materializations),actual_replay_runs=len(replays),attempt_costs=costs,
            diagnostic_results_used_for_selection=False,holdout_authorized=False)
        if paired_labels:
            from .search_over_raw_paired_chain import paired_value_diagnostics
            report["paired_labels"]=label_reports
            report["paired_value_diagnostics"]=paired_value_diagnostics(leaf_cell_plan,label_reports,result["owned_replay_join"])
        _write_new(root/"diagnostic.json",report)
        return dict(native_owner=native_owner,materializations=materializations,replays=replays,
            leaf_cell_plan=leaf_cell_plan,
            materialization_plan=materialization_plan,replay_plan=replay_plan),report
    except BaseException as failure:
        try:
            if not (ledger.directory/"operational-stop.json").exists():ledger.stop(type(failure).__name__)
        except BaseException as stop_failure:failure.add_note("Native chain stop persistence also failed: "+type(stop_failure).__name__)
        raise
