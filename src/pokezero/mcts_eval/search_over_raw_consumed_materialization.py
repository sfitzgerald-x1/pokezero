"""Owned native construction/dispatch to owned reference-constructor joins.

No sampler, native call, replay, simulator launch or universal equivalence gate.
One consumed ordinal is joined only after BOTH distinct owning-run readers
validate their actual receipts. Every original construction ordinal remains in
the denominator, including collapsed multiplicity and unmaterialized worlds.
Matching drawn teams/public overlays is not matching engines or transitions.
"""
from copy import deepcopy
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path

from .search_over_raw import digest, require
from .search_over_raw_construction import construction_source_hashes, read_native_construction
from .search_over_raw_fixed_work_run import FixedWorkRunValidator
from .search_over_raw_materialization_run import MaterializationRunValidator, _join_result
from .search_over_raw_paired_materialization import PairedMaterializationRunValidator
from .search_over_raw_paired_replay import PairedPrefixRunValidator, PairedPrefixRunContract, read_paired_prefix
from .search_over_raw_world_inputs import _state_payload

SCHEMA = "pokezero.search-over-raw.owned-consumed-materialization-join.v1"


def _private_json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


@dataclass(frozen=True)
class OwnedNativeReplayPlan:
    """Immutable PRIVATE specifications, not admission or an executable run.

    A fresh owning-evidence join is required to build jobs. This object alone
    certifies neither a caller's projections nor scientific launch authority.
    """
    payload_utf8: str

    def receipt(self):
        require(type(self.payload_utf8) is str, "frozen private replay plan bytes required")
        result = json.loads(self.payload_utf8)
        require(_private_json(result) == self.payload_utf8, "canonical replay plan bytes required")
        return result

    @property
    def sha256(self):
        return hashlib.sha256(self.payload_utf8.encode()).hexdigest()


def plan_owned_native_replays(native_selected, reference_runs, *, native_validator,
                              ancestry_contract, native_extract, config, attempt_root, preserve_prefix=False,
                              paired_contexts=False):
    """Plan ALL original coordinates without launching, sampling or loading.

    Attempt paths are deterministic within the externally frozen attempt root.
    Any existing path is a consumed claim, not evidence of success or a reason
    to choose a replacement path. The later supervisor must claim it create-only.
    Replanning after failure never offers a retry or alternative action.
    """
    from .search_over_raw_boundary_run import OwnedNativePathBoundaryJob, NativePathBoundaryRunContract, raw_root_json
    from .search_over_raw_ancestry import NativeAncestryContract
    from .search_over_raw_outcome_run import verify_outcome_runtime
    from .search_over_raw_reconciled import controller_input_hashes
    from .search_over_raw_showdown_boundary import project_showdown_boundary, commands_for_native_actions
    from .search_over_raw_world_inputs import _state_payload
    from .search_over_raw_native_prefix import OwnedNativePrefixBoundaryJob, NativePrefixBoundaryRunContract
    from .search_over_raw_showdown_boundary import NativeReplayUnavailable

    attempt_root = Path(attempt_root)
    require(type(preserve_prefix) is bool and type(paired_contexts) is bool and (not paired_contexts or preserve_prefix)
        and type(ancestry_contract) is NativeAncestryContract
        and attempt_root.is_absolute() and attempt_root == attempt_root.resolve(),
        "externally frozen canonical absolute replay attempt root required")
    config_before, sources = digest(_state_payload(config)), controller_input_hashes()
    roots = join_owned_ancestry_roots(native_selected, reference_runs, native_validator=native_validator,
        ancestry_contract=ancestry_contract, native_extract=native_extract)
    materializers = {row["construction_ordinal"]: row for row in reference_runs}
    for row in reference_runs: verify_outcome_runtime(row["validator"].runtime, config)
    records = []
    for root in roots["rows"]:
        constructor = root["constructor"]
        ordinal = constructor["construction_ordinal"]
        for event in root["sampled_events"]:
            coordinate = dict(construction_ordinal=ordinal, evaluation_ordinal=event["evaluation_ordinal"])
            identity = digest(dict(native_selected_sha256=roots["native_selected_sha256"],
                native_capture_sha256=event["capture_sha256"], **coordinate))
            directory = attempt_root / ("native-replay-" + identity)
            require(directory == directory.resolve(), "replay attempt path redirected")
            blockers, spec = [], None
            if directory.exists() or directory.is_symlink(): blockers.append("ATTEMPT_ALREADY_CLAIMED")
            if not constructor["actual_constructor_join_validated"]:
                blockers.append(constructor["status"])
            if root["mapping"] is None: blockers.append("UNIQUE_PARTY_MAPPING_UNAVAILABLE")
            boundaries = event["boundaries"]
            if boundaries is None: blockers.append("NATIVE_BOUNDARIES_UNAVAILABLE")
            elif any(row["native"].get("action_identity") is None for row in boundaries[1:]):
                blockers.append("NATIVE_ACTION_AMBIGUOUS")
            if not blockers:
                material = materializers[ordinal]
                raw_root = material["result"]["produced_bridge_snapshot"]
                projection = project_showdown_boundary(raw_root, root["mapping"])
                if projection["reference_terminal"]:
                    blockers.append("REFERENCE_ROOT_TERMINAL")
                else:
                    # Structural projection checks remain outside this catch.
                    # An unsupported but valid initial request (e.g. preview)
                    # blocks its row before constructing the nonterminal job.
                    try:
                        commands_for_native_actions(raw_root, root["mapping"], boundaries[1]["native"]["action_identity"])
                    except NativeReplayUnavailable:
                        if not preserve_prefix: blockers.append("INITIAL_NATIVE_ACTION_UNAVAILABLE_OR_UNSUPPORTED")
                    except ValueError:
                        if preserve_prefix: raise
                        blockers.append("INITIAL_NATIVE_ACTION_UNAVAILABLE_OR_UNSUPPORTED")
                if not blockers:
                    origin = dict(materialization_capture_sha256=constructor["reference_capture_sha256"],
                        produced_bridge_snapshot_sha256=constructor["produced_bridge_snapshot_sha256"],
                        native_selected_sha256=roots["native_selected_sha256"],
                        native_capture_sha256=event["capture_sha256"], **coordinate)
                    spec = dict(raw_root=deepcopy(raw_root), root_binding=deepcopy(native_validator.consumed),
                        root_origin=origin, runtime=deepcopy(material["validator"].runtime),
                        clone_binding=dict(battleId="nr-"+identity, cloneId="np-"+identity,
                            snapshotJsonSha256=hashlib.sha256(raw_root_json(raw_root).encode()).hexdigest()),
                        request_ids=["nr-"+identity+":"+str(index) for index in range(event["required_boundaries"])],
                        native_boundaries=deepcopy(boundaries), party_mapping=deepcopy(root["mapping"]))
                    if paired_contexts:
                        from .search_over_raw_paired_replay import OwnedPairedPrefixJob
                        require(type(material["validator"]) is PairedMaterializationRunValidator,
                            "paired replay requires original paired constructor ownership")
                        snapshot=material["validator"].retained_context(material["result"])
                        spec["paired_constructor"]=deepcopy(material["result"]["paired_context"])
                        job=OwnedPairedPrefixJob(config=config,snapshot=snapshot,**spec)
                    else:
                        job = (OwnedNativePrefixBoundaryJob if preserve_prefix else OwnedNativePathBoundaryJob)(config=config, **spec)
                    # Malformed evidence/configuration is a failure, not an
                    # unsupported path relabeling. Only legal resolution can
                    # become an explicit unavailable initial-action row.
                    spec["expected_inputs_sha256"] = digest(job.input_binding())
            records.append(dict(**coordinate, native_capture_sha256=event["capture_sha256"],
                required_boundaries=event["required_boundaries"], attempt_directory=str(directory),
                status="PREREQUISITES_UNAVAILABLE" if blockers else "PLANNED_NOT_ADMITTED",
                blockers=blockers, job_spec=spec))
    # Native extraction is callback-bearing. Preserve the frozen configuration,
    # sources, owning receipts and exact plan across its final recheck.
    require(digest(roots) == digest(join_owned_ancestry_roots(native_selected, reference_runs,
        native_validator=native_validator, ancestry_contract=ancestry_contract, native_extract=native_extract))
        and digest(_state_payload(config)) == config_before and controller_input_hashes() == sources,
        "private replay planning inputs or owning evidence changed")
    for row in reference_runs: verify_outcome_runtime(row["validator"].runtime, config)
    require(attempt_root == attempt_root.resolve() and all(Path(row["attempt_directory"])
        == Path(row["attempt_directory"]).resolve() for row in records),
        "replay attempt paths redirected during final extraction")
    require(all(not Path(row["attempt_directory"]).exists() and not Path(row["attempt_directory"]).is_symlink()
        for row in records if row["job_spec"] is not None), "attempt claimed during replay planning")
    return OwnedNativeReplayPlan(_private_json(dict(
        schema="pokezero.search-over-raw.private-native-replay-plan.v3" if paired_contexts else
            "pokezero.search-over-raw.private-native-replay-plan.v2" if preserve_prefix else
            "pokezero.search-over-raw.private-native-replay-plan.v1", private_controller_only=True,
        attempt_root=str(attempt_root), config_sha256=config_before, controller_sources=sources,
        ancestry_contract=asdict(ancestry_contract),
        replay_contract=(PairedPrefixRunContract
            if paired_contexts else NativePrefixBoundaryRunContract if preserve_prefix else NativePathBoundaryRunContract)().receipt(),
        ancestry_root_join=roots, records=records, required_construction_ordinals=4,
        actual_native_calls=roots["actual_native_calls"], actual_model_evaluations=roots["actual_model_evaluations"],
        actual_sampled_events=roots["actual_sampled_events"],
        materialization_ordinal_records=len(reference_runs),
        actual_materialization_runs=len({(str(row["validator"].directory), row["validator"].job_sha)
            for row in reference_runs}),
        required_construction_sample_records=len(records),
        required_boundary_records=sum(row["required_boundaries"] for row in records),
        required_arriving_action_records=sum(row["required_boundaries"]-1 for row in records),
        planned_job_records=sum(row["job_spec"] is not None for row in records),
        synthetic_scope=roots["synthetic_scope"], process_launches=0, source_generation=False,
        retries_permitted=False, branch_transition_validated=False, resource_cost_complete=False,
        runtime_qualified=False, scientific_collection_authorized=False)))


def build_owned_native_replay_jobs(plan, native_selected, reference_runs, *, native_validator,
                                   ancestry_contract, native_extract, config):
    """Revalidate frozen PRIVATE specifications; return jobs, never run them.

    Future collection must separately bind this exact plan/attempt root in its
    reviewed admission. This is not a reservation or scientific authorization.
    """
    from .search_over_raw_boundary_run import OwnedNativePathBoundaryJob
    from .search_over_raw_native_prefix import OwnedNativePrefixBoundaryJob
    require(type(plan) is OwnedNativeReplayPlan, "distinct frozen native replay plan required")
    saved, before = plan.receipt(), plan.sha256
    require(saved["schema"] in {"pokezero.search-over-raw.private-native-replay-plan.v1",
        "pokezero.search-over-raw.private-native-replay-plan.v2","pokezero.search-over-raw.private-native-replay-plan.v3"}, "explicit original native replay plan version required")
    paired=saved["schema"].endswith(".v3");prefix=paired or saved["schema"].endswith(".v2")
    def check():
        fresh = plan_owned_native_replays(native_selected, reference_runs, native_validator=native_validator,
            ancestry_contract=ancestry_contract, native_extract=native_extract, config=config,
            attempt_root=saved["attempt_root"], preserve_prefix=prefix,paired_contexts=paired)
        require(fresh.payload_utf8 == plan.payload_utf8 and plan.sha256 == before,
            "replay plan differs from current owning evidence, sources or claims")
    check()
    jobs = []
    for row in saved["records"]:
        if row["job_spec"] is None: continue
        spec = deepcopy(row["job_spec"])
        expected = spec.pop("expected_inputs_sha256")
        if paired:
            from .search_over_raw_paired_replay import OwnedPairedPrefixJob
            material=next(r for r in reference_runs if r["construction_ordinal"]==row["construction_ordinal"])
            snapshot=material["validator"].retained_context(material["result"])
            job=OwnedPairedPrefixJob(config=deepcopy(config),snapshot=snapshot,**spec)
        else:
            job = (OwnedNativePrefixBoundaryJob if prefix else OwnedNativePathBoundaryJob)(config=deepcopy(config), **spec)
        require(digest(job.input_binding()) == expected, "planned native replay job input drift")
        jobs.append(dict(construction_ordinal=row["construction_ordinal"], evaluation_ordinal=row["evaluation_ordinal"],
            attempt_directory=row["attempt_directory"], plan_sha256=before, job=job))
    check()
    return jobs


def reconcile_owned_replay_diagnostic(native_selected, reference_runs, replay_runs, *,
                                     native_validator, ancestry_contract, native_extract,
                                     native_support=None,native_successor_support=None):
    """Attach observed replay/work/cost evidence, NEVER scored panel evidence.

    Recomputes the full owned join; no supplied summary or hash is accepted as
    validation. The fixed-work selection has a distinct identity from a timed
    deployment selector. No observed seam yet binds it to a registered panel
    record, so panel attachment remains explicitly unvalidated. Only supplied
    completed owning jobs are costed; missing/failed attempts and controller
    work outside those jobs remain uncovered, not zero or a retry permission.
    """
    from .search_over_raw_reconciled import controller_input_hashes
    sources = controller_input_hashes()
    joined = join_owned_raw_replays(native_selected, reference_runs, replay_runs,
        native_validator=native_validator, ancestry_contract=ancestry_contract, native_extract=native_extract,
        native_support=native_support,native_successor_support=native_successor_support)

    def collect():
        owners, generations = {}, {}
        def add(role, coordinate, reader, capture):
            witness = reader(capture)  # Fresh durable read, never stale last_validation.
            readout = deepcopy(reader.last_validation)
            require(type(readout) is dict and readout["process_receipts_validated"] is True,
                "actual completed owning-process readout required")
            expected = digest(capture)
            require(readout.get("selected_sha256", readout.get("capture_sha256")) == expected
                and (witness is None or witness == expected), "exact diagnostic owning readout required")
            path, job = str(reader.directory), reader.job_sha
            require(path == str(Path(path).resolve()), "diagnostic cost owner redirected")
            launch = readout["main_launch"]
            generation = (launch["pid"], launch["birth_identity"], launch["supervisor_pid"])
            # PID reuse is legitimate; the birth and owning parent remain part
            # of identity. The same observed generation cannot be two jobs.
            require(generation not in generations or generations[generation] == path,
                "overlapping diagnostic owning launch generations")
            generations[generation] = path
            record = dict(role=role, directory=path, job_sha256=job, binding_sha256=reader.binding,
                capture_sha256=expected, validation_witness=witness, readout=readout,
                declared_runtime_sha256=reader.runtime_sha if role == "native-fixed-work" else digest(reader.runtime))
            if path in owners:
                require(digest(owners[path]["owner"]) == digest(record),
                    "conflicting diagnostic job or receipt aliases")
                owners[path]["coordinates"].append(coordinate)
            else:
                owners[path] = dict(owner=record, coordinates=[coordinate])
        add("native-fixed-work", None, native_validator, native_selected)
        for row in reference_runs:
            add("materialization", dict(construction_ordinal=row["construction_ordinal"]),
                row["validator"], row["result"]["capture"])
        for row in replay_runs:
            add("replay", dict(construction_ordinal=row["construction_ordinal"],
                evaluation_ordinal=row["evaluation_ordinal"]), row["validator"], row["result"]["capture"])
        return [owners[path] for path in sorted(owners)]

    owners = collect()
    # All callback-bearing work precedes the final fresh cost/owner recheck.
    require(digest(joined) == digest(join_owned_raw_replays(native_selected, reference_runs, replay_runs,
        native_validator=native_validator, ancestry_contract=ancestry_contract, native_extract=native_extract,
        native_support=native_support,native_successor_support=native_successor_support)),
        "diagnostic projections, rosters or owning evidence changed")
    require(digest(owners) == digest(collect()) and sources == controller_input_hashes(),
        "diagnostic resource readouts or reader sources changed")
    native = next(row["owner"] for row in owners if row["owner"]["role"] == "native-fixed-work")
    costs = [row["owner"]["readout"]["costs"] for row in owners]
    sampler_complete = all(row["sampling_child_cpu_seconds"] is not None for row in costs)
    job_cpu = sum(row["job_cpu_seconds_including_reaped_children"] for row in costs)
    supervisor_cpu = sum(row["supervisor_thread_cpu_seconds"] for row in costs)
    known_sampler_cpu = sum(row["sampling_child_cpu_seconds"] for row in costs
        if row["sampling_child_cpu_seconds"] is not None)
    return dict(schema="pokezero.search-over-raw.private-owned-replay-diagnostic.v1",
        private_controller_only=True, status="OBSERVED_DIAGNOSTIC_NOT_PANEL_ADMISSION",
        diagnostic_selection_sha256=digest(native_selected),
        diagnostic_configuration_sha256=native_validator.configuration.identity,
        diagnostic_runtime_sha256=native_validator.runtime_sha,
        diagnostic_statistics_root_id=native_validator.statistics_root_id,
        diagnostic_selection_seed=native_validator.seed,
        consumed_root_binding=deepcopy(native_validator.consumed),
        fixed_work_contract=native_validator.contract.receipt(),
        actual_fixed_work=native["readout"]["actual_work"], owned_replay_join=joined,
        owning_jobs=owners, costed_unique_jobs=len(owners),
        covered_cost_subtotals=dict(job_cpu_seconds_including_reaped_children=job_cpu,
            supervisor_thread_cpu_seconds=supervisor_cpu,
            known_sampling_child_cpu_seconds=known_sampler_cpu,
            sampling_child_cpu_seconds=known_sampler_cpu if sampler_complete else None,
            all_costed_jobs_sampler_cpu_known=sampler_complete,
            known_separate_cpu_components_seconds=job_cpu+supervisor_cpu+known_sampler_cpu,
            accumulated_job_wall_exposure_seconds=sum(row["elapsed_wall_seconds"] for row in costs),
            campaign_elapsed_wall_seconds=None, concurrent_tree_peak_bytes=None,
            descendant_close_cpu_added_again=False),
        uncovered_cost_scope=["unsupplied, unstarted, failed or interrupted attempts",
            "missing materialization/replay coordinates (not inferred zero cost)",
            "controller/planning/validation and source costs outside supervised jobs",
            "unobserved instantaneous or physical tree-memory peak"],
        controller_sources=sources, synthetic_scope=joined["synthetic_scope"],
        root_to_panel_attachment_validated=False, deployment_accepted_units_added=0,
        scored_root_intervals_changed=False, qualification_changed=False,
        branch_transition_validated=False, independent_terminal_labels=False,
        complete_attempt_population_cost=False, resource_cost_complete=False,
        runtime_qualified=False, scientific_collection_authorized=False, holdout_authorized=False)


def _original_diagnostic_owner_guard(native_selected,reference_runs,replay_runs,native_validator):
    """Freeze original owners before archive callbacks; final guard invokes none.

    Receipt trees include added/removed entries. Declared model/code/runtime
    files outside those trees remain bound too. This is not an atomic snapshot
    or an owning-reader replacement; the actual readers must still validate.
    """
    from .search_over_raw_continuation_admission import _path
    from .search_over_raw_boundary_run import RawBoundaryRunValidator,NativePathBoundaryRunValidator
    from .search_over_raw_native_prefix import NativePrefixBoundaryRunValidator
    from .search_over_raw_world_inputs import _state_payload
    readers=[native_validator]+[row["validator"] for row in reference_runs+replay_runs]
    reader_types=tuple(type(reader) for reader in readers)
    require(all(type(reader) in (FixedWorkRunValidator,MaterializationRunValidator,PairedMaterializationRunValidator,
        RawBoundaryRunValidator,NativePathBoundaryRunValidator,NativePrefixBoundaryRunValidator,PairedPrefixRunValidator) for reader in readers),"original diagnostic owning readers required")
    rows=tuple(reference_runs+replay_runs)
    def inputs():
        owners=[]
        for reader in readers:
            keys=("directory","job_sha","binding","allow_synthetic","contract","expected")
            if type(reader) is FixedWorkRunValidator:
                keys+=("configuration","root_id","statistics_root_id","seed","runtime_sha","consumed",
                    "file_hashes","checkpoint_path","checkpoint_sha","showdown_root","public_request_sha256",
                    "native_public_fold_required","frozen")
            else:
                keys+=("runtime","inputs")
                if type(reader) in (RawBoundaryRunValidator,NativePathBoundaryRunValidator,NativePrefixBoundaryRunValidator):keys+=("raw_root",)
                if type(reader) is PairedMaterializationRunValidator:keys+=("state","context")
                if type(reader) is PairedPrefixRunValidator:keys+=("raw_root","snapshot","endpoint","paired_inputs")
            owners.append(_state_payload({key:getattr(reader,key) for key in keys}))
        return digest(dict(native=native_selected,owners=owners,
            rows=[dict(coordinates={k:r[k] for k in r if k!="validator"},reader_index=readers.index(r["validator"])) for r in rows]))
    def files():
        snapshot={}
        for reader in readers:
            directory=_path(reader.directory)
            require(directory.is_dir(),"original diagnostic owner directory unavailable")
            for path in directory.rglob("*"):
                path=_path(path)
                require(path.is_file() or path.is_dir(),"unsupported diagnostic receipt node")
                snapshot[str(path)]="directory" if path.is_dir() else hashlib.sha256(path.read_bytes()).hexdigest()
            declared=reader.file_hashes if type(reader) is FixedWorkRunValidator else reader.runtime["files"]
            for path in declared:
                path=_path(path)
                snapshot[str(path)]=hashlib.sha256(path.read_bytes()).hexdigest()
        return snapshot
    original_inputs,original_files=inputs(),files()
    def guard(readouts):
        current=reference_runs+replay_runs
        require(len(current)==len(rows) and all(a is b for a,b in zip(current,rows))
            and tuple(type(reader) for reader in readers)==reader_types
            and native_validator is readers[0] and all(row["validator"] is reader for row,reader in zip(rows,readers[1:]))
            and inputs()==original_inputs and files()==original_files,
            "original diagnostic owner inputs or receipt files changed after archive callbacks")
        require(all(digest(reader.last_validation)==digest(readouts[str(reader.directory)]) for reader in readers),
            "original diagnostic owning readouts changed after archive callback")
    return guard


def reconcile_archived_exploration_diagnostic(native_selected, reference_runs, replay_runs, *,
        archive, source, contract, root_id, native_validator, ancestry_contract, native_extract,
        native_support=None,native_successor_support=None):
    """Private observed panel-to-request attachment, NOT scored admission.

    Only an owning reader prospectively bound to the full request may attach.
    Missing slots/coordinates stay visible and the source receipt's structural
    partition is never upgraded to source-policy or terminal qualification.
    """
    from .search_over_raw_archive import archived_exploration_root_binding,retained_archive_root_identity
    from .search_over_raw_fixed_work_run import FixedWorkRunValidator
    require(type(native_validator) is FixedWorkRunValidator,
        "actual fixed-work owning reader required for archive attachment")
    owner_guard=_original_diagnostic_owner_guard(native_selected,reference_runs,replay_runs,native_validator)
    selected_roots=[r for r in source["roots"] if r["root_id"]==root_id]
    require(len(selected_roots)==1,"original diagnostic archived root required")
    index=selected_roots[0]["source_request_index"]
    root_before=retained_archive_root_identity(archive,index)
    source_before,contract_before=digest(source),digest(contract)
    before = archived_exploration_root_binding(archive, source, contract=contract, root_id=root_id)
    witness = before["public_request_witness"]
    require(before["slot"]["root_slot"] == 0
        and native_validator.statistics_root_id == root_id and type(native_validator.seed) is int
        and native_validator.seed == before["slot"]["source_seed"]
        and native_validator.configuration.arm == "incumbent"
        and digest(native_validator.consumed) == digest(witness["consumed_binding"])
        and native_validator.public_request_sha256 == witness["public_request_sha256"],
        "prospective owning diagnostic/public panel request mismatch")
    require(native_validator(native_selected) == digest(native_selected),
        "uncleared owning diagnostic cannot validate archive attachment")
    result = reconcile_owned_replay_diagnostic(native_selected, reference_runs, replay_runs,
        native_validator=native_validator, ancestry_contract=ancestry_contract, native_extract=native_extract,
        native_support=native_support,native_successor_support=native_successor_support)
    # The native extract callback is finished before re-reading the archive and
    # all owning receipts. Caller labels or stale readouts cannot clear a join.
    require(digest(before) == digest(archived_exploration_root_binding(archive, source,
        contract=contract, root_id=root_id)), "archived source or private boundary changed during attachment")
    require(digest(result) == digest(reconcile_owned_replay_diagnostic(native_selected, reference_runs, replay_runs,
        native_validator=native_validator, ancestry_contract=ancestry_contract, native_extract=native_extract,
        native_support=native_support,native_successor_support=native_successor_support)),
        "owning diagnostic changed during archived attachment")
    result_before=digest(result)
    readouts={row["owner"]["directory"]:deepcopy(row["owner"]["readout"]) for row in result["owning_jobs"]}
    require(digest(before) == digest(archived_exploration_root_binding(archive, source,
        contract=contract, root_id=root_id)), "archived root changed after final owning callbacks")
    # The archive extractor above is the LAST callback. Do not alternate it
    # with another owner read: check the ORIGINAL frozen objects/bytes directly.
    owner_guard(readouts)
    require(root_before==retained_archive_root_identity(archive,index) and source_before==digest(source)
        and contract_before==digest(contract) and result_before==digest(result),
        "retained diagnostic root or result changed after final archive callback")
    return dict(result, archive_crosswalk=before, root_to_panel_attachment_validated=True,
        panel_attachment_scope="exploration identity/full public request only; never scored admission",
        source_policy_validated=False, source_terminal_validated=False, source_freshness_validated=False)


def join_consumed_materializations(native_selected, reference_runs, *, native_validator):
    """Return scoped observations, not collection or simulator equivalence.

    reference_runs rows carry a constructor ordinal, its actual job result and
    its separately prospectively bound MaterializationRunValidator. Those
    inputs must not be fabricated from the native result after the fact. The
    explicit synthetic opt-in on each reader only clears engineering fixtures.
    Nothing implicitly recovers or repeats a draw to fill a missing row.
    """
    require(type(native_validator) is FixedWorkRunValidator
        and native_validator.configuration.arm == "incumbent"
        and type(reference_runs) is list and len(reference_runs) <= 4,
        "distinct native ownership reader and bounded reference roster required")
    native_before = digest(native_selected)
    native = deepcopy(native_selected)
    indexed, originals = {}, []
    for row in reference_runs:
        require(type(row) is dict and set(row) == {"construction_ordinal", "result", "validator"}
            and type(row["construction_ordinal"]) is int and 0 <= row["construction_ordinal"] < 4
            and row["construction_ordinal"] not in indexed
            and type(row["validator"]) in (MaterializationRunValidator,PairedMaterializationRunValidator),
            "unique original ordinal and distinct materialization reader required")
        originals.append((row, row["validator"], digest(row["result"])))
        indexed[row["construction_ordinal"]] = dict(result=deepcopy(row["result"]), validator=row["validator"])
    native_witness = native_validator(native)
    require(native_witness is None or native_witness == native_before, "exact independent native ownership required")
    # Even non-clearing synthetic scope has its full allocation inspected by
    # the owning reader; do not silently turn it into production evidence.
    manifest = native["evidence"]["consumed_world_inputs"]
    rows = sorted(manifest["rows"], key=lambda row: row["ordinal"])
    calls = {row["call_ordinal"]:row for row in manifest["calls"]}
    result = []
    for row in rows:
        ordinal = row["ordinal"]
        call = calls[row["call_ordinal"]]
        checked = read_native_construction(row["construction_receipt"], binding=native_validator.consumed,
            ordinal=ordinal, draw_sha256=row["draw_sha256"], constructed_sha256=row["constructed_sha256"],
            state_utf8_sha256=row["native_state_utf8_sha256"], source_hashes=construction_source_hashes())
        item = dict(construction_ordinal=ordinal, native_call_ordinal=call["call_ordinal"],
            collapsed_construction_ordinals=deepcopy(call["construction_ordinals"]),
            native_call_multiplicity=call["multiplicity"], native_input_sha256=call["input_sha256"],
            native_state_utf8_sha256=row["native_state_utf8_sha256"],
            native_report_utf8_sha256=call["report_utf8_sha256"],
            native_construction_capture_sha256=checked["capture_sha256"],
            native_selected_sha256=native_before, native_run_validated=native_witness is not None,
            reference_capture_sha256=None, reference_run_validated=False,
            produced_bridge_snapshot_sha256=None, materializer_options_sha256=None,
            status="UNMATERIALIZED_UNCERTAIN", mismatched_inputs=[],
            actual_constructor_join_validated=False)
        supplied = indexed.get(ordinal)
        if supplied is not None:
            reference, validator = supplied["result"], supplied["validator"]
            if type(validator) is PairedMaterializationRunValidator:
                validator._join_result(reference,reference["capture"],validator.expected["runtime_sha256"])
            else:_join_result(reference, validator.expected, reference["capture"])
            witness = validator(reference["capture"])
            require(witness is None or witness == digest(reference["capture"]),
                "exact independently owned materialization capture required")
            # Compare the separately frozen, actually validated input object,
            # not a caller wrapper that could alias numeric types via Python ==.
            require(digest(reference["materialization_inputs"])
                == digest(validator.expected["materialization_inputs"]), "canonical materialization input join required")
            inputs = validator.expected["materialization_inputs"]
            world = row["construction_receipt"]["events"][0]["value"]
            mismatches = []
            for name, left, right in (
                ("root_binding", native_validator.consumed, inputs["root_binding"]),
                ("draw", world["override"], inputs["draw"]),
                ("public_overlay", world["payload"], inputs["materialize"]["publicState"]),
            ):
                if digest(left) != digest(right): mismatches.append(name)
            if "recorded_native_origin" in inputs:
                expected_origin=dict(schema="pokezero.search-over-raw.recorded-native-materialization.v1",
                    native_selected_sha256=native_before,construction_capture_sha256=checked["capture_sha256"],
                    construction_ordinal=ordinal,original_public_request_sha256=native_validator.public_request_sha256,
                    draw_sha256=digest(world["override"]),public_payload_sha256=digest(world["payload"]))
                if digest(inputs["recorded_native_origin"])!=digest(expected_origin):mismatches.append("recorded_native_origin")
                actual_seed=call["native_call_parameters"]["chance_seed"]
                if type(actual_seed) is not int or type(inputs["seed"]) is not int or inputs["seed"]!=actual_seed:
                    mismatches.append("recorded_native_chance_seed")
            both = native_witness is not None and witness is not None
            item.update(reference_capture_sha256=digest(reference["capture"]),
                reference_run_validated=witness is not None,
                produced_bridge_snapshot_sha256=reference["produced_bridge_snapshot_sha256"],
                materializer_options_sha256=digest(inputs["options"]), mismatched_inputs=mismatches,
                status="UNVALIDATED_SCOPE" if not both else "INPUT_MISMATCH" if mismatches
                    else "MATCHED_CONSTRUCTOR_INPUTS_NOT_FULL_WORLD",
                actual_constructor_join_validated=both and not mismatches)
        result.append(item)
    require(digest(native_selected) == native_before and all(row["validator"] is validator
        and digest(row["result"]) == before for row, validator, before in originals)
        and [row["construction_ordinal"] for row in reference_runs] == list(indexed),
        "owned constructor join inputs changed during independent validation")
    return dict(schema=SCHEMA, native_selected_sha256=native_before,
        required_construction_ordinals=4, actual_native_calls=len(calls), rows=result,
        matched_constructor_ordinals=sum(row["actual_constructor_join_validated"] for row in result),
        materialization_ordinal_records=len(indexed),
        materialization_runs_recorded=len({(str(row["validator"].directory), row["validator"].job_sha)
            for row in indexed.values()}),
        synthetic_scope=native_validator.last_validation["synthetic_process_fixture"]
            or any(row["result"]["synthetic_process_fixture"] for row in indexed.values()),
        private_state_exported=False, native_ancestry_coverage_validated=False,
        party_mapping_validated=False, branch_transition_validated=False,
        full_world_equivalence=False, resource_cost_complete=False, runtime_qualified=False,
        scientific_collection_authorized=False)


def _observe_raw_replay_boundaries(native_boundaries, reference_boundaries, mapping, required_boundaries):
    """Compare observed fields, never supply an independent semantic validator.

    Incoming identities use actual acknowledged commands and the PRE-state
    request catalog. Later ordinal agreement after divergence is not causal
    trajectory correspondence. Root actions have no arriving command.
    """
    from .search_over_raw_correspondence import REQUIRED_FIELDS, KNOWN_CLASSES
    from .search_over_raw_showdown_boundary import project_showdown_boundary
    require(type(required_boundaries) is int and 1 <= required_boundaries <= 65,
        "full original boundary denominator required")
    for rows in (native_boundaries, reference_boundaries):
        require(rows is None or type(rows) is list and len(rows) == required_boundaries,
            "observed boundary roster cannot be shortened")
    before = digest(dict(native=native_boundaries, reference=reference_boundaries, mapping=mapping))
    observations = []
    for ordinal in range(required_boundaries):
        native = None if native_boundaries is None else native_boundaries[ordinal]
        reference = None if reference_boundaries is None else reference_boundaries[ordinal]
        require(native is None or type(native["ordinal"]) is int and native["ordinal"] == ordinal,
            "actual native projection coordinate drift")
        projection = None
        if reference is not None and mapping is not None:
            projection = project_showdown_boundary(reference["snapshot"], mapping,
                before_snapshot=None if not ordinal else reference_boundaries[ordinal-1]["snapshot"],
                incoming_choices=None if not ordinal else reference["observed"]["incoming_choices"])
        fields = {}
        for field in REQUIRED_FIELDS:
            left = None if native is None else native["native"].get(field)
            right = None if projection is None else projection["showdown"].get(field)
            fields[field] = "NOT_APPLICABLE" if field == "action_identity" and not ordinal else \
                "UNCHECKABLE" if left is None or right is None else \
                "MATCH" if digest(left) == digest(right) else "MISMATCH"
        documented = []
        if native is not None and projection is not None:
            for field, classification in KNOWN_CLASSES.items():
                left, right = native["native"].get(field), projection["showdown"].get(field)
                if left is not None and right is not None and digest(left) != digest(right):
                    documented.append(dict(component=field, classification=classification,
                        native_sha256=digest(left), showdown_sha256=digest(right), required_field=False))
        observations.append(dict(ordinal=ordinal, observed_required_fields=fields,
            native_projection_sha256=None if native is None else digest(native),
            reference_snapshot_sha256=None if reference is None else digest(reference["snapshot"]),
            incoming_choices_sha256=None if not ordinal or reference is None else digest(reference["observed"]["incoming_choices"]),
            reference_projection=projection, documented_differences=documented,
            independently_validated=False, branch_transition_validated=False,
            matched_causal_trajectory=False, runtime_qualified=False, scientific_collection_authorized=False))
    require(before == digest(dict(native=native_boundaries, reference=reference_boundaries, mapping=mapping)),
        "observed replay projection inputs changed")
    return observations


def join_owned_raw_replays(native_selected, reference_runs, replay_runs, *, native_validator,
                           ancestry_contract, native_extract, native_support=None, native_successor_support=None):
    """Join prior owners and full coordinate rosters, never certify transitions.

    Every constructor/sample pair remains visible, including unreplayed pairs.
    An origin digest from a replay is not evidence: it must match the separately
    validated native and materializer witnesses. Commands are separately frozen
    and observed. Scoped action/field comparisons are observations, not
    independently certified projection semantics or transition correspondence.
    """
    from .search_over_raw_boundary_run import RawBoundaryRunValidator, NativePathBoundaryRunValidator, _join_raw_result
    from .search_over_raw_native_prefix import NativePrefixBoundaryRunValidator, read_native_prefix_transport
    from .search_over_raw_boundary_transport import read_owned_boundary_transport
    def reader_binding(reader):
        return digest(dict(directory=str(reader.directory), job_sha=reader.job_sha, binding=reader.binding,
            allow_synthetic=reader.allow_synthetic, contract=reader.contract, expected=reader.expected,
            raw_root=reader.raw_root, runtime=reader.runtime, inputs=reader.inputs,
            **(dict(paired_snapshot=reader.paired_inputs) if type(reader) is PairedPrefixRunValidator else {})))
    require(type(replay_runs) is list and len(replay_runs) <= 32, "bounded original replay roster required")
    indexed, originals = {}, []
    for row in replay_runs:
        require(type(row) is dict and set(row) == {"construction_ordinal", "evaluation_ordinal", "result", "validator"}
            and type(row["construction_ordinal"]) is int and 0 <= row["construction_ordinal"] < 4
            and type(row["evaluation_ordinal"]) is int and row["evaluation_ordinal"] >= 0
            and type(row["validator"]) in (RawBoundaryRunValidator, NativePathBoundaryRunValidator, NativePrefixBoundaryRunValidator,PairedPrefixRunValidator),
            "original replay coordinates and distinct independent raw owner required")
        key = (row["construction_ordinal"], row["evaluation_ordinal"])
        require(key not in indexed, "duplicate constructor/sample replay record")
        indexed[key] = dict(result=deepcopy(row["result"]), validator=row["validator"])
        originals.append((row, key, row["validator"], digest(row["result"]), reader_binding(row["validator"])))
    roots = join_owned_ancestry_roots(native_selected, reference_runs, native_validator=native_validator,
        ancestry_contract=ancestry_contract, native_extract=native_extract, native_support=native_support,
        native_successor_support=native_successor_support)
    materializers = {r["construction_ordinal"]: r for r in reference_runs}
    required = {(r["constructor"]["construction_ordinal"], e["evaluation_ordinal"])
        for r in roots["rows"] for e in r["sampled_events"]}
    require(set(indexed) <= required, "replay not in the actual sampled-event roster")
    records = []
    for root in roots["rows"]:
        constructor = root["constructor"]
        ordinal = constructor["construction_ordinal"]
        for event in root["sampled_events"]:
            key = (ordinal, event["evaluation_ordinal"])
            item = dict(construction_ordinal=ordinal, evaluation_ordinal=key[1],
                native_capture_sha256=event["capture_sha256"], required_boundaries=event["required_boundaries"],
                status="UNREPLAYED_UNCERTAIN", replay_capture_sha256=None, replay_run_validated=False,
                origin_mismatches=[], reference_boundaries=None, initial_roundtrip_equal=None,
                owned_root_origin_and_coordinates_join=False, native_joint_actions_matched=None,
                native_action_driven_origin_join=False)
            supplied = indexed.get(key)
            if supplied is not None:
                result, validator = supplied["result"], supplied["validator"]
                if type(validator) is PairedPrefixRunValidator:
                    validator._join_result(result,result["capture"],validator.expected["runtime_sha256"])
                else:_join_raw_result(result, validator.expected, result["capture"])
                witness = validator(result["capture"])
                require(witness is None or witness == digest(result["capture"]), "exact independent raw replay capture required")
                supplied["witness"] = witness
                mismatches = []
                expected_origin = dict(materialization_capture_sha256=constructor["reference_capture_sha256"],
                    produced_bridge_snapshot_sha256=constructor["produced_bridge_snapshot_sha256"],
                    native_selected_sha256=roots["native_selected_sha256"], native_capture_sha256=event["capture_sha256"],
                    construction_ordinal=ordinal, evaluation_ordinal=key[1])
                for field, value in expected_origin.items():
                    if digest(validator.expected["root_origin"][field]) != digest(value): mismatches.append(field)
                if digest(validator.expected["root_binding"]) != digest(native_validator.consumed):
                    mismatches.append("consumed_root_binding")
                material = materializers.get(ordinal)
                if material is None:
                    mismatches.append("missing_prior_materialization")
                else:
                    if digest(validator.raw_root) != digest(material["result"]["produced_bridge_snapshot"]):
                        mismatches.append("actual_materialized_raw_root")
                    # Source/version/config/file bindings must remain the same
                    # through construction and cloning; a new root wrapper must
                    # not silently move to a different simulator or checkpoint.
                    if digest(validator.runtime) != digest(material["validator"].runtime):
                        mismatches.append("materialization_replay_runtime")
                if len(validator.expected["request_ids"]) != event["required_boundaries"]:
                    mismatches.append("full_native_coordinate_roster")
                native_driven = type(validator) in (NativePathBoundaryRunValidator, NativePrefixBoundaryRunValidator,PairedPrefixRunValidator)
                if native_driven:
                    if digest(validator.expected["native_boundaries"]) != digest(event["boundaries"]):
                        mismatches.append("actual_native_boundary_projections")
                    if digest(validator.expected["party_mapping"]) != digest(root["mapping"]):
                        mismatches.append("actual_unique_party_mapping")
                both = witness is not None and constructor["actual_constructor_join_validated"]
                status = "UNVALIDATED_REPLAY_SCOPE" if not both else "ORIGIN_MISMATCH" if mismatches \
                    else "OWNED_RAW_ROOT_AND_COORDINATES_NOT_TRANSITIONS"
                item.update(status=status, replay_capture_sha256=digest(result["capture"]),
                    replay_run_validated=witness is not None, origin_mismatches=mismatches,
                    owned_root_origin_and_coordinates_join=both and not mismatches,
                    native_action_driven_origin_join=native_driven and both and not mismatches)
                if both and not mismatches:
                    capture = result["capture"]
                    from .search_over_raw_boundary_run import raw_root_json
                    if type(validator) is PairedPrefixRunValidator:
                        require(type(material["validator"]) is PairedMaterializationRunValidator
                            and validator.expected["paired_constructor"]==material["result"]["paired_context"],
                            "paired replay must retain its original constructor context")
                        original=material["validator"].retained_context(material["result"])
                        require(digest(_state_payload(original))==digest(_state_payload(validator.snapshot)),
                            "paired replay snapshot differs from original constructor")
                        trace,_=read_paired_prefix(capture,process_records=capture["records"],snapshot=validator.snapshot,
                            expected=validator.expected)
                        item["replay_termination"]=deepcopy(trace["termination"])
                        if not trace["full_native_path_observed"]:item["status"]="OWNED_PREFIX_AND_UNAVAILABLE_SUFFIX"
                    elif type(validator) is NativePrefixBoundaryRunValidator:
                        trace = read_native_prefix_transport(capture, process_records=capture["records"], raw_root=validator.raw_root,
                            clone_binding=validator.expected["clone_binding"], request_ids=validator.expected["request_ids"],
                            native_boundaries=validator.expected["native_boundaries"], party_mapping=validator.expected["party_mapping"])
                        item["replay_termination"] = deepcopy(trace["termination"])
                        if not trace["full_native_path_observed"]: item["status"] = "OWNED_PREFIX_AND_UNAVAILABLE_SUFFIX"
                    else:
                        trace = read_owned_boundary_transport(capture, process_records=capture["records"],
                            initial_snapshot_json=raw_root_json(validator.raw_root),
                            clone_binding=validator.expected["clone_binding"], request_ids=validator.expected["request_ids"])
                    item["reference_boundaries"] = [dict(observed=observed,
                        snapshot=json.loads(raw["currentSnapshotJson"]))
                        for observed, raw in zip(trace["observed_snapshots"], capture["snapshots"])]
                    item["initial_roundtrip_equal"] = digest(item["reference_boundaries"][0]["snapshot"]) == digest(validator.raw_root)
                    item["reference_boundaries"].extend([None] * (event["required_boundaries"] - len(item["reference_boundaries"])))
                    # Exact roundtrip equality is reported, not a universal
                    # simulator-equivalence gate or independent terminal label.
            item["boundary_observations"] = _observe_raw_replay_boundaries(event["boundaries"],
                item["reference_boundaries"], root["mapping"], event["required_boundaries"])
            from .search_over_raw_native_edge import observe_supported_prefix
            item["native_edge_support"] = deepcopy(event["native_edge_support"])
            item["supported_prefix_observations"] = observe_supported_prefix(
                item["boundary_observations"], item["native_edge_support"])
            if native_successor_support is not None:
                from .search_over_raw_native_edge import compare_projected_successors
                item["projected_successor_support"] = compare_projected_successors(
                    item["boundary_observations"],event["boundaries"],event["native_successor_support"],
                    native_action_driven_origin_join=item["native_action_driven_origin_join"])
            records.append(item)
    # The last native extraction is callback-bearing. All replay checks must
    # follow it, including external reader bindings and the exact scope witness.
    require(digest(roots) == digest(join_owned_ancestry_roots(native_selected, reference_runs,
        native_validator=native_validator, ancestry_contract=ancestry_contract, native_extract=native_extract,
        native_support=native_support,native_successor_support=native_successor_support)),
        "prior native/materializer owners or native projections changed")
    require(len(replay_runs) == len(originals) and all(current is original
        and set(current) == {"construction_ordinal", "evaluation_ordinal", "result", "validator"}
        and type(current["construction_ordinal"]) is int and type(current["evaluation_ordinal"]) is int
        and (current["construction_ordinal"], current["evaluation_ordinal"]) == key
        and current["validator"] is validator and digest(current["result"]) == before
        and reader_binding(validator) == reader_before
        for current, (original, key, validator, before, reader_before) in zip(replay_runs, originals)),
        "original replay inputs changed during joining")
    # Re-read actual owners after every controller operation; no closed handle
    # is polled and no exit code is inferred from process absence.
    for row in indexed.values():
        if type(row["validator"]) is PairedPrefixRunValidator:
            row["validator"]._join_result(row["result"],row["result"]["capture"],row["validator"].expected["runtime_sha256"])
        else:_join_raw_result(row["result"], row["validator"].expected, row["result"]["capture"])
        require(digest(row["validator"](row["result"]["capture"])) == digest(row["witness"]),
            "independent raw replay ownership witness changed")
    return dict(schema=("pokezero.search-over-raw.private-owned-raw-replay-join.v3"
        if any(type(row["validator"]) is PairedPrefixRunValidator for row in replay_runs)
        else "pokezero.search-over-raw.private-owned-raw-replay-join.v2"
        if any(type(row["validator"]) is NativePrefixBoundaryRunValidator for row in replay_runs)
        else "pokezero.search-over-raw.private-owned-raw-replay-join.v1"), private_controller_only=True,
        **(dict(projected_successor_support_counts={status:sum(row["status"]==status for record in records
            for row in record["projected_successor_support"]) for status in
            ("WITHIN_PROJECTED_NATIVE_SUPPORT","OUTSIDE_PROJECTED_NATIVE_SUPPORT","UNCHECKABLE")})
            if native_successor_support is not None else {}),
        ancestry_root_join=roots, records=records, required_construction_sample_records=len(required),
        synthetic_scope=roots["synthetic_scope"] or any(r["result"]["synthetic_process_fixture"] for r in indexed.values()),
        replay_ordinal_records=len(indexed), actual_replay_runs=len({(str(r["validator"].directory), r["validator"].job_sha)
            for r in indexed.values()}),
        owned_root_origin_and_coordinate_joins=sum(r["owned_root_origin_and_coordinates_join"] for r in records),
        native_action_driven_origin_joins=sum(r["native_action_driven_origin_join"] for r in records),
        required_boundary_records=sum(r["required_boundaries"] for r in records),
        required_arriving_action_records=sum(r["required_boundaries"]-1 for r in records),
        observed_required_field_counts={status: sum(value == status for record in records
            for boundary in record["boundary_observations"] for value in boundary["observed_required_fields"].values())
            for status in ("MATCH", "MISMATCH", "UNCHECKABLE", "NOT_APPLICABLE")},
        observed_arriving_action_counts={status: sum(boundary["ordinal"] > 0
            and boundary["observed_required_fields"]["action_identity"] == status for record in records
            for boundary in record["boundary_observations"]) for status in ("MATCH", "MISMATCH", "UNCHECKABLE")},
        native_edge_support_counts={status: sum(edge["status"] == status for record in records
            for edge in record["native_edge_support"]) for status in ("SUPPORTED", "UNSUPPORTED", "UNCHECKABLE")},
        observed_supported_prefix_boundaries=sum(boundary["observed_prefix_match"] for record in records
            for boundary in record["supported_prefix_observations"]),
        native_joint_actions_matched=False, branch_transition_validated=False, independent_terminal_labels=False,
        full_world_equivalence=False, resource_cost_complete=False, runtime_qualified=False, scientific_collection_authorized=False)


def join_owned_ancestry_roots(native_selected, reference_runs, *, native_validator,
                              ancestry_contract, native_extract, native_support=None,native_successor_support=None):
    """Private structural join; no replay, labels, source draws or qualification.

    The callable is the post-selection native boundary extractor, not a value
    evaluator. Its returned reports are read and bound to actual owned state
    bytes, but this does not authenticate its loaded runtime or certify physics.
    Every construction ordinal and sampled event remains present. Collapsed
    ordinals share an actual call's sample, not additional evaluation events.
    """
    from .search_over_raw_ancestry import validate_owned_native_ancestry
    from .search_over_raw_construction import derive_party_mapping
    from .search_over_raw_native_boundary import read_native_boundary, extract_native_capture_boundaries
    from .search_over_raw_native_edge import extract_native_edge_support
    from .search_over_raw_reconciled import controller_input_hashes

    require(callable(native_extract), "post-selection boundary extractor required")
    require(native_support is None or callable(native_support), "post-selection native support helper required")
    require(native_successor_support is None or callable(native_successor_support),"post-selection successor helper required")
    require(type(reference_runs) is list and len(reference_runs) <= 4 and all(type(row) is dict
        and set(row) == {"construction_ordinal", "result", "validator"} for row in reference_runs),
        "bounded original materialization roster required")
    before = digest(native_selected)
    sources = controller_input_hashes()
    originals = [(row, row["validator"], row["construction_ordinal"], digest(row["result"]))
        for row in reference_runs]
    # Both readers independently inspect the original, prospective owning runs.
    constructors = join_consumed_materializations(native_selected, reference_runs,
        native_validator=native_validator)
    ancestry = validate_owned_native_ancestry(native_selected,
        run_validator=native_validator, contract=ancestry_contract)
    saved = deepcopy(native_selected)
    manifest = saved["evidence"]["consumed_world_inputs"]
    indexed = {row["construction_ordinal"]: deepcopy(row["result"]) for row in reference_runs}
    calls = {row["call_ordinal"]: row for row in manifest["calls"]}
    exports = saved["evidence"]["native_searched_ancestry"]
    # Repeated shared roots are extracted once; the full root/interior distinction
    # is part of the key. Cache exact UTF-8 privately and return its hash at every
    # boundary; do not substitute the parsed/canonical JSON hash for raw bytes.
    cache = {}

    def unique_object(pairs):
        result = {}
        for key, value in pairs:
            require(key not in result, "duplicate native boundary JSON key")
            result[key] = value
        return result

    def extract(state, root):
        key = (state, root)
        if key not in cache:
            raw = native_extract(state, root)
            require(type(raw) is str and len(raw.encode()) <= ancestry_contract.maximum_payload_bytes,
                "bounded actual native boundary UTF-8 export required")
            checked = read_native_boundary(json.loads(raw, object_pairs_hook=unique_object),
                state_text=state, root=root)
            cache[key] = (raw, checked)
        return cache[key][0]

    rows = []
    for world, constructor in zip(sorted(manifest["rows"], key=lambda r: r["ordinal"]), constructors["rows"]):
        ordinal, call_id = world["ordinal"], world["call_ordinal"]
        readout = ancestry["invocations"][call_id]
        payload = json.loads(exports[call_id]["export_utf8"])
        captures = {leaf["capture"]["evaluation_ordinal"]: leaf["capture"] for leaf in payload["evaluations"]}
        item = dict(constructor=deepcopy(constructor), ancestry_export_utf8_sha256=readout["export_utf8_sha256"],
            actual_model_evaluations=readout["actual_model_evaluations"],
            sampled_events=[dict(evaluation_ordinal=event["evaluation_ordinal"],
                capture_sha256=event["capture_sha256"], required_boundaries=len(event["boundaries"]),
                boundaries=None) for event in readout["retained_evaluations"]],
            mapping=None, mapping_sha256=None, native_root_export_utf8_sha256=None,
            status=constructor["status"], limitations=[], owned_unique_party_mapping_join=False)
        for event in item["sampled_events"]:
            event["native_edge_support"] = extract_native_edge_support(captures[event["evaluation_ordinal"]],
                root_binding=native_validator.consumed["root_binding"],
                selection_report_sha256=calls[call_id]["report_utf8_sha256"], native_support=None)
            if native_successor_support is not None:
                from .search_over_raw_native_edge import extract_native_successor_support
                event["native_successor_support"]=extract_native_successor_support(captures[event["evaluation_ordinal"]],
                    root_binding=native_validator.consumed["root_binding"],
                    selection_report_sha256=calls[call_id]["report_utf8_sha256"],native_successor_support=None)
        if constructor["actual_constructor_join_validated"] and ancestry["owning_run_validated"]:
            checked = read_native_construction(world["construction_receipt"], binding=native_validator.consumed,
                ordinal=ordinal, draw_sha256=world["draw_sha256"], constructed_sha256=world["constructed_sha256"],
                state_utf8_sha256=world["native_state_utf8_sha256"], source_hashes=construction_source_hashes())
            state = world["construction_receipt"]["events"][1]["value"]["native_state"]
            raw = extract(state, True)
            mapping = derive_party_mapping(indexed[ordinal]["produced_bridge_snapshot"], cache[(state, True)][1], checked)
            item.update(mapping=deepcopy(mapping["mapping"]), limitations=mapping["limitations"],
                native_root_export_utf8_sha256=hashlib.sha256(raw.encode()).hexdigest(),
                status="OWNED_ROOT_UNIQUE_SET_JOIN_NOT_TRANSITION" if mapping["mapping"] is not None
                    else "PARTY_IDENTITY_UNCERTAIN",
                mapping_sha256=digest(mapping["mapping"]) if mapping["mapping"] is not None else None,
                owned_unique_party_mapping_join=mapping["mapping"] is not None)
            # Extract ALL boundaries, never only the retained endpoint. Mapping
            # failure does not erase the event denominator or select a new leaf.
            for event in item["sampled_events"]:
                capture = captures[event["evaluation_ordinal"]]
                event["native_edge_support"] = extract_native_edge_support(capture,
                    root_binding=native_validator.consumed["root_binding"],
                    selection_report_sha256=calls[call_id]["report_utf8_sha256"], native_support=native_support)
                if native_successor_support is not None:
                    event["native_successor_support"]=extract_native_successor_support(capture,
                        root_binding=native_validator.consumed["root_binding"],
                        selection_report_sha256=calls[call_id]["report_utf8_sha256"],
                        native_successor_support=native_successor_support)
                event["boundaries"] = extract_native_capture_boundaries(capture,
                    root_binding=native_validator.consumed["root_binding"],
                    selection_report_sha256=calls[call_id]["report_utf8_sha256"], native_extract=extract)
                states = [capture["root_native_state"]] + [step["after"] for step in capture["steps"]]
                for boundary, text in zip(event["boundaries"], states):
                    boundary["native_export_utf8_sha256"] = hashlib.sha256(
                        cache[(text, boundary["ordinal"] == 0)][0].encode()).hexdigest()
        rows.append(item)
    require(before == digest(native_selected) and sources == controller_input_hashes()
        and len(reference_runs) == len(originals) and all(current is row and row["validator"] is validator
            and set(row) == {"construction_ordinal", "result", "validator"}
            and row["construction_ordinal"] == ordinal and type(row["construction_ordinal"]) is int
            and digest(row["result"]) == sha
            for current, (row, validator, ordinal, sha) in zip(reference_runs, originals)),
        "private ancestry/root join inputs or reader sources changed")
    require(digest(constructors) == digest(join_consumed_materializations(native_selected, reference_runs,
        native_validator=native_validator)) and digest(ancestry) == digest(validate_owned_native_ancestry(
            native_selected, run_validator=native_validator, contract=ancestry_contract)),
        "owning receipts or runtime inputs changed during native extraction")
    return dict(schema="pokezero.search-over-raw.private-owned-ancestry-root-join.v1",
        private_controller_only=True, native_selected_sha256=before, rows=rows,
        required_construction_ordinals=4, actual_native_calls=ancestry["actual_native_calls"],
        actual_model_evaluations=sum(r["actual_model_evaluations"] for r in ancestry["invocations"]),
        actual_sampled_events=sum(len(r["retained_evaluations"]) for r in ancestry["invocations"]),
        construction_sample_records=sum(len(r["sampled_events"]) for r in rows),
        unique_native_boundary_exports=len(cache), synthetic_scope=constructors["synthetic_scope"],
        owned_unique_party_mapping_joins=sum(r["owned_unique_party_mapping_join"] for r in rows),
        native_extractor_runtime_qualified=False, party_mapping_validated=False,
        branch_transition_validated=False, independent_terminal_labels=False,
        full_world_equivalence=False, resource_cost_complete=False, runtime_qualified=False,
        scientific_collection_authorized=False)
