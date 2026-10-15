"""Full retained-event AFTER roster; unavailable histories remain uncertain.

This explicitly null-only diagnostic consumes the complete original roster.
It is not a substitute for paired terminal labels or a collection admission.
Live owner validation and post-archive direct guards are deliberately distinct.
"""
from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

from .search_over_raw import digest, require
from .search_over_raw_after_admission import execute_original_native_after_bank, _payload
from .search_over_raw_after_bank import OwnedAfterRawBank, _inventory as bank_inventory
from .search_over_raw_after_owner import _enclosing, _enclosing_guard
from .search_over_raw_attempt_cost import _inventory
from .search_over_raw_consumed_materialization import _original_diagnostic_owner_guard
from .search_over_raw_diagnostic_producer import diagnostic_configurations
from .search_over_raw_native_leaf_cells import NativeLeafCellPlan, leaf_cell_law
from .search_over_raw_outcome_run import verify_outcome_runtime, _runtime_inventory
from .search_over_raw_supervision import _save_new
from .search_over_raw_world_inputs import _state_payload

SCHEMA = "pokezero.search-over-raw.native-after-population.v1"


@dataclass(frozen=True, kw_only=True)
class NativeAfterRecipe:
    config: object = None
    runtime: dict | None = None
    review_path: str | None = None
    unqualified_reason: str | None = None


def _freeze_after_recipe(recipe):
    """Original typed objects, not only equal serialized recipe values."""
    require(type(recipe) is NativeAfterRecipe,"original typed AFTER recipe required")
    kind=type(recipe)
    keys=("config","runtime","review_path","unqualified_reason")
    fields=tuple((key,getattr(recipe,key),type(getattr(recipe,key))) for key in keys)
    original_sha=digest(_state_payload(recipe))
    def guard():
        require(type(recipe) is kind and set(vars(recipe)) == set(keys)
            and all(getattr(recipe,k) is obj and type(getattr(recipe,k)) is typ for k,obj,typ in fields)
            and digest(_state_payload(recipe)) == original_sha,
            "original AFTER recipe class/field references or values changed")
    guard()
    return guard


def population_law():
    return dict(schema=SCHEMA, leaf_cell_law=leaf_cell_law(), null_only=True,
        population="all_original_incumbent_model_slots_and_unique_retained_events",
        missing_event_count="unknown_not_zero", paired_history_launch_permitted=False,
        diagnostic_results_used_for_selection=False, retries_permitted=False,
        scientific_collection_authorized=False, holdout_authorized=False)


def after_population_review_template(contract, *, config, runtime):
    return dict(schema=SCHEMA, contract_design_sha256=digest({k:v for k,v in contract.items() if k != "input_hashes"}),
        config_sha256=digest(_state_payload(config)), runtime_sha256=digest(runtime),
        law=population_law(), status="PENDING_INDEPENDENT_REVIEW",
        independent_review_completed=False, reviewed_before_first_source_work=False)


def preflight_after_population(ledger, recipe, *, checkpoint_contract):
    """No source, clone, policy load or inference; reviewed before first source."""
    guard=_freeze_after_recipe(recipe)
    ledger.verify()
    guard()
    if recipe.unqualified_reason is not None:
        require(type(recipe.unqualified_reason) is str and recipe.unqualified_reason.strip()
            and all(getattr(recipe, k) is None for k in ("config", "runtime", "review_path")),
            "unqualified AFTER recipe cannot carry executable inputs")
        return
    verify_outcome_runtime(recipe.runtime, recipe.config)
    guard()
    require(recipe.runtime["checkpoint_path"] == checkpoint_contract.checkpoint_path
        and recipe.runtime["files"][recipe.runtime["checkpoint_path"]] == checkpoint_contract.checkpoint_sha256
        and str(recipe.config.resolved_showdown_root()) == checkpoint_contract.showdown_root,
        "original champion/simulator required for AFTER recipe")
    path = Path(recipe.review_path)
    require(path.is_absolute() and path == path.resolve(), "canonical AFTER review required")
    raw = path.read_bytes()
    expected = after_population_review_template(ledger.contract, config=recipe.config, runtime=recipe.runtime)
    expected.update(status="CLEAR_NULL_ONLY_AFTER_POPULATION_RECIPE", independent_review_completed=True,
        reviewed_before_first_source_work=True)
    require(digest(json.loads(raw)) == digest(expected)
        and ledger.contract["input_hashes"].get(str(path)) == hashlib.sha256(raw).hexdigest(),
        "original prospective null-only AFTER review required")
    guard()


def plan_after_population(manifest, chain_rows, leaf_plans, *, recipe):
    """Structural denominator builder, not an owning-evidence validator."""
    require(type(recipe) is NativeAfterRecipe and type(chain_rows) is list and type(leaf_plans) is dict,
        "explicit original AFTER roster inputs required")
    cfg = next(c for c in diagnostic_configurations() if c.arm == "incumbent" and c.leaf == "model")
    expected = [s["root"] for s in manifest["slots"] if s["configuration_sha256"] == cfg.identity]
    require(len(expected) >= 32 and len({r["source_seed"] for r in expected}) >= 32 and len(expected) == len(chain_rows)
        and len({r["root_id"] for r in expected}) == len(expected), "complete original native slot roster required")
    slots = []; events = []; known = set()
    for root, row in zip(expected, chain_rows):
        require(type(row) is dict and row["root_id"] == root["root_id"]
            and type(row["source_seed"]) is int and row["source_seed"] == root["source_seed"],
            "original native slot order/seed/root changed")
        recorded = row["status"] == "CHAIN_RECORDED"
        require(recorded is (row["diagnostic"] is not None), "native chain availability is contradictory")
        count = None; cells = None; plan_sha = None
        if recorded:
            known.add(root["root_id"])
            plan = leaf_plans.get(root["root_id"])
            require(type(plan) is NativeLeafCellPlan, "every original recorded chain needs its leaf plan")
            saved = plan.receipt(); report = row["diagnostic"]
            require(plan.sha256 == report["leaf_cell_plan_sha256"] and saved["law"] == leaf_cell_law()
                and saved["run_binding"]["statistics_root_id"] == root["root_id"]
                and saved["run_binding"]["selection_seed"] == root["source_seed"]
                and saved["retained_unique_evaluation_events"] == len(saved["events"])
                and saved["required_label_cells"] == len(saved["events"])*8
                and report["retained_unique_leaf_events"] == len(saved["events"])
                and report["required_leaf_label_cells"] == len(saved["events"])*8,
                "original event denominator differs from native chain")
            count = len(saved["events"]); cells = count*8; plan_sha = plan.sha256
            for event in saved["events"]:
                require(event["canonical_construction_ordinal"] == min(event["construction_aliases"])
                    and len(event["labels"]) == 8 and all(type(c["replicate"]) is int
                        and c["replicate"] == i and c["cell_id"] == digest(dict(event_id=event["event_id"], replicate=i))
                        and c["status"] == "UNSTARTED_UNCERTAIN" and c["signed_outcome"] is None
                        for i,c in enumerate(event["labels"])), "original full eight-cell population required")
                events.append(dict(root_id=root["root_id"], leaf_plan_sha256=plan.sha256,
                    event_id=event["event_id"], canonical_construction_ordinal=event["canonical_construction_ordinal"],
                    evaluation_ordinal=event["evaluation_ordinal"], after_snapshot_index=event["required_after_snapshot_index"],
                    cell_ids=[c["cell_id"] for c in event["labels"]]))
        slots.append(dict(root_id=root["root_id"], source_seed=root["source_seed"], native_status=row["status"],
            status="UNSTARTED_UNCERTAIN" if recorded and recipe.unqualified_reason is None else
                "SKIPPED_UNQUALIFIED_AFTER" if recorded else "NATIVE_EVENT_POPULATION_UNKNOWN",
            retained_events=count, required_cells=cells, leaf_plan_sha256=plan_sha))
    require(set(leaf_plans) == known and len({e["event_id"] for e in events}) == len(events),
        "extra/duplicate original native leaf population")
    return dict(schema=SCHEMA, law=population_law(), slots=slots, events=events,
        required_native_slots=len(slots), known_retained_events=len(events), known_required_cells=len(events)*8,
        unknown_native_slots=sum(s["retained_events"] is None for s in slots),
        unqualified_reason=recipe.unqualified_reason)


@dataclass(frozen=True)
class OwnedAfterPopulation:
    result: dict
    original_result_sha256: str
    pairs: tuple
    original_pairs: tuple
    original_states: tuple
    direct_guard: object

    def __reduce_ex__(self, protocol):
        raise TypeError("private AFTER population cannot enter worker transport")

    def check_closed(self):
        """No live-archive callback; original payloads/refs/receipts only."""
        self.direct_guard()
        require(self.pairs is self.original_pairs and digest(self.result) == self.original_result_sha256
            and tuple(_pair_state(a,b) for a,b in self.pairs) == self.original_states,
            "original AFTER population changed after later callbacks")
        for _, bank in self.pairs: bank._check_original()

    def validate_live(self):
        self.check_closed()
        for admission, bank in self.pairs:
            admission.validate(); bank.validate(); self.check_closed()
        self.check_closed()
        return deepcopy(self.result)


def _pair_state(admission, bank):
    from .search_over_raw_after_admission import OwnedNativeAfterAdmission
    require(type(admission) is OwnedNativeAfterAdmission and type(bank) is OwnedAfterRawBank,
        "original typed AFTER admission/bank required")
    return digest(dict(admission_id=id(admission), bank_id=id(bank), raw_id=id(admission.raw_after),
        programs_id=id(admission.programs), receipt_id=id(admission.receipt), guard_id=id(admission.original_guard),
        payload=_payload(admission.raw_after, admission.programs, admission.receipt), original=admission.original_sha256,
        result=bank.result, files=bank_inventory(bank.directory)))


def _original_row_crosswalk(rows, pairs, plan, original_hashes):
    require(len(rows) == len(pairs) == len(original_hashes)
        and [digest(row) for row in rows] == original_hashes
        and all(digest(row["admission"]) == digest(a.receipt) and digest(row["bank"]) == digest(b.result)
            and digest({k:row[k] for k in plan["events"][i]}) == digest(plan["events"][i])
            for i,(row,(a,b)) in enumerate(zip(rows,pairs))),
        "original copied AFTER rows/coordinates differ from their owning pairs")


def produce_owned_after_population(ledger, *, manifest, chain_rows, private_chains, recipe,
        checkpoint_contract, ownership, native_extract):
    """Claim ALL slots/events/cells before the first original event admission."""
    recipe_guard=_freeze_after_recipe(recipe)
    preflight_after_population(ledger, recipe, checkpoint_contract=checkpoint_contract)
    recipe_guard()
    scope, scope_files, ownership_sha = _enclosing(ownership)
    require(type(private_chains) is list, "original private chain list required")
    chain_refs = tuple(private_chains); guards = []; plans = {}
    for chain in private_chains:
        owner = chain["native_owner"]; root_id = owner.job.statistics_root_id
        require(root_id not in plans, "duplicate original AFTER native root")
        plans[root_id] = chain["leaf_cell_plan"]
        direct = _original_diagnostic_owner_guard(owner.selected, chain["materializations"], chain["replays"], owner.reader)
        readers = [owner.reader]+[r["validator"] for r in chain["materializations"]+chain["replays"]]
        readouts = {str(r.directory):deepcopy(r.last_validation) for r in readers}
        objects = tuple(chain.items()); plan_bytes = tuple((chain[k],chain[k].payload_utf8)
            for k in ("leaf_cell_plan", "materialization_plan", "replay_plan"))
        guards.append((chain, objects, plan_bytes, direct, readouts, owner, owner.job, owner.reader, owner.selected))
    plan = plan_after_population(manifest, chain_rows, plans, recipe=recipe)
    root = ledger.directory/"native-after-population"
    require(root == root.resolve() and not root.exists() and not (scope.directory/"after-banks").exists(),
        "AFTER population is create-only; no prior bank, retry or continuation")
    original_inputs = digest(dict(manifest=manifest, chains=chain_rows, recipe=_state_payload(recipe), plan=plan))
    original_controller = ledger.contract["input_hashes"]
    declared = {p:sha for p,sha in original_controller.items()}
    if recipe.runtime is not None: declared.update(recipe.runtime["files"])
    root.mkdir(mode=0o700)
    claim = dict(schema=SCHEMA, plan=deepcopy(plan), plan_sha256=digest(plan), enclosing_binding_sha256=scope.binding)
    _save_new(root/"claim.json", claim)
    claim_bytes = (root/"claim.json").read_bytes()
    pairs = []; states = []; rows = []; row_states = []
    files = _inventory([root]); bank_files = _inventory([scope.directory/"after-banks"])
    def check():
        recipe_guard()
        _enclosing_guard(scope, scope_files, ownership, ownership_sha)
        require(len(private_chains) == len(chain_refs) and all(a is b for a,b in zip(private_chains,chain_refs))
            and digest(dict(manifest=manifest, chains=chain_rows, recipe=_state_payload(recipe), plan=plan)) == original_inputs
            and (root/"claim.json").read_bytes() == claim_bytes and _inventory([root]) == files
            and _inventory([scope.directory/"after-banks"]) == bank_files
            and tuple(_pair_state(a,b) for a,b in pairs) == tuple(states)
            and all(Path(p).is_file() and Path(p) == Path(p).resolve()
                and hashlib.sha256(Path(p).read_bytes()).hexdigest() == sha for p,sha in declared.items()),
            "original full AFTER population/inputs/receipts changed")
        _original_row_crosswalk(rows,pairs,plan,row_states)
        for chain, objects, byte_plans, direct, readouts, owner, job, reader, selected in guards:
            require(tuple(chain) == tuple(k for k,_ in objects) and all(chain[k] is obj for k,obj in objects)
                and owner.job is job and owner.reader is reader and owner.selected is selected
                and all(p.payload_utf8 == raw for p,raw in byte_plans), "original private AFTER chain changed")
            direct(readouts)
        for _, bank in pairs: bank._check_original()
    try:
        check()
        if recipe.unqualified_reason is None:
            for event in plan["events"]:
                check()
                chain = next(c for c in private_chains if c["native_owner"].job.statistics_root_id == event["root_id"])
                admission, bank = execute_original_native_after_bank(chain, ledger=ledger, ownership=ownership,
                    event_id=event["event_id"], expected_plan_sha256=event["leaf_plan_sha256"], source_config=recipe.config,
                    native_extract=native_extract, runtime=recipe.runtime, policies=None, null_only=True)
                # Preserve prior banks across the last admission callback; only
                # this newly expected create-only bank may change the namespace.
                require(tuple(_pair_state(a,b) for a,b in pairs) == tuple(states), "earlier AFTER bank changed")
                require({p.name for p in (scope.directory/"after-banks").iterdir()} == {
                    digest(dict(plan_sha256=e["leaf_plan_sha256"],event_id=e["event_id"]))
                    for e in plan["events"][:len(pairs)+1]}, "unexpected AFTER bank or retry namespace")
                pairs.append((admission,bank)); states.append(_pair_state(admission,bank))
                rows.append(dict(**deepcopy(event), admission=deepcopy(admission.receipt), bank=deepcopy(bank.result)))
                row_states.append(digest(rows[-1]))
                bank_files = _inventory([scope.directory/"after-banks"])
                check()
        result = dict(schema=SCHEMA, plan_sha256=digest(plan), slots=completed_slots(plan), events=rows,
            required_native_slots=plan["required_native_slots"], known_retained_events=plan["known_retained_events"],
            known_required_cells=plan["known_required_cells"], unknown_native_slots=plan["unknown_native_slots"],
            consumed_events=len(rows), consumed_cells=len(rows)*8, after_process_launches=0,
            scored_terminal_labels=0, independent_terminal_labels=False, resource_cost_complete=False,
            runtime_qualified=False, diagnostic_results_used_for_selection=False, scientific_collection_authorized=False,
            holdout_authorized=False, unqualified_reason=recipe.unqualified_reason)
        _save_new(root/"result.json",result); files = _inventory([root])
        check()
        original_pairs = tuple(pairs)
        owned = OwnedAfterPopulation(result,digest(result),original_pairs,original_pairs,tuple(states),check)
        owned.validate_live(); owned.check_closed()
        return owned
    except BaseException as error:
        if not (root/"operational-stop.json").exists():
            _save_new(root/"operational-stop.json",dict(schema=SCHEMA, error_type=type(error).__name__,
                retries_permitted=False, scientific_collection_authorized=False))
        raise


def completed_slots(plan):
    slots=deepcopy(plan["slots"])
    for slot in slots:
        if slot["status"] == "UNSTARTED_UNCERTAIN":slot["status"] = "ALL_RESERVED_CELLS_UNCERTAIN"
    return slots


def validate_null_admission_reason(raw, programs, status):
    """Exact current producer reasons; missing anchors are normal nulls."""
    expected = "CANONICAL_PATH_UNAVAILABLE" if raw is None else (
        "TERMINAL_AFTER_REQUIRES_SEPARATE_LAW" if raw["battle"]["ended"] else None)
    if expected is not None:
        require(status == expected and programs["p1"] is None,"unavailable/terminal AFTER cannot acquire an actor program")
    elif programs["p1"] is not None:
        require(status == "SAMPLED_ACTOR_HISTORY_UNAVAILABLE","known p1 history must retain the sampled-p2 null reason")
    else:
        require(status in {"ORIGINAL_ACTOR_HISTORY_ABSENT","ORIGINAL_SOURCE_ENCODING_UNAVAILABLE",
            "ACTOR_NOT_REQUESTED","ACTUAL_OWN_REQUEST_UNAVAILABLE","EARLIER_ACTIONABLE_OWN_REQUEST_UNAVAILABLE",
            "OWN_REQUEST_ANCHOR_UNAVAILABLE","OWN_KNOWLEDGE_ANCHOR_MISMATCH"},
            "unavailable p1 history needs its original null reason")


def read_recorded_after_population(ledger, workflow, *, manifest, chain_rows, contexts, recipe,
        auditor_directory, auditor_pid, expected_outer_terminal, allow_synthetic=False):
    """Completed null-only join, called after independent native child readers.

Contexts are reconstructed by read_recorded_workflow, not private archive
recovery. p1 program hashes remain reported in-auditor checks, not externally
reconstructed actor provenance. No receipt here can admit a scored label.
"""
    from .search_over_raw_after_admission import SCHEMA as ADMISSION_SCHEMA
    from .search_over_raw_after_bank import SCHEMA as BANK_SCHEMA
    from .search_over_raw_resources import (read_completed_process_cost,measurement_protocol,resource_source_hash,
        validate_memory_receipt)
    from .search_over_raw_nested_ownership import verify_delegated_closure
    from .search_over_raw_supervision import _verify_closed_generations
    recipe_guard=_freeze_after_recipe(recipe)
    directory = Path(auditor_directory)
    require(directory.is_absolute() and directory == directory.resolve() and type(auditor_pid) is int
        and auditor_pid > 0 and type(expected_outer_terminal) is dict and type(allow_synthetic) is bool,
        "actual original enclosing auditor terminal required for completed AFTER readback")
    paths = [ledger.directory,directory]
    inventory = _inventory(paths)
    before = digest(dict(workflow=workflow,manifest=manifest,chains=chain_rows,recipe=_state_payload(recipe),
        terminal=expected_outer_terminal))
    def load(path):
        require(path == path.resolve() and path.is_file(), "canonical completed AFTER receipt required")
        return json.loads(path.read_bytes())
    claim,launch,worker,terminal = (load(directory/name) for name in
        ("claim.json","launch.json","worker-result.json","terminal.json"))
    require(digest(terminal) == digest(expected_outer_terminal)
        and hashlib.sha256((directory/"job.pickle").read_bytes()).hexdigest() == claim["job_sha256"]
        and claim["retry_authorized"] is False
        and claim["synthetic_process_fixture"] is allow_synthetic
        and launch["pid"] == launch["group"] == worker["worker_pid"] == terminal["worker_pid"] == auditor_pid
        and type(launch["supervisor_pid"]) is int and launch["supervisor_pid"] > 0
        and launch["supervisor_pid"] != auditor_pid
        and type(launch["birth_identity"]) is str and launch["birth_identity"]
        and claim["binding_sha256"] == launch["binding_sha256"] == worker["binding_sha256"] == terminal["binding_sha256"]
        and all(type(v) is int and v == 0 for v in (terminal["supervisor_exit_code"],terminal["worker_exit_code"],worker["exit_code"]))
        and terminal["error_type"] is None and worker["error_type"] is None and terminal["cleanup_error_type"] is None
        and terminal["watchdog_reached"] is False and digest(worker["result"]) == digest(terminal["result"]),
        "completed AFTER population needs the original normal outer launch/result/wait join")
    require(digest(claim["resource_measurement_protocol"]) == digest(measurement_protocol())
        and claim["resource_measurement_source_sha256"] == resource_source_hash(),
        "original enclosing resource protocol/code binding required")
    if terminal["memory_measurement"] is not None:
        validate_memory_receipt(directory,binding=claim["binding_sha256"],main_launch=launch,
            expected_summary=terminal["memory_measurement"])
        require(terminal["resource_measurement_error_type"] is None,"conflicting AFTER enclosing resource failure")
    else:
        require(type(terminal["resource_measurement_error_type"]) is str and terminal["resource_measurement_error_type"],
            "missing AFTER enclosing resource measurement must be explicit")
    if allow_synthetic:
        require(terminal["schema"] == "pokezero.search-over-raw.fixed-work-supervision.v1"
            and terminal["status"] == "DIAGNOSTIC_COMPLETE"
            and digest(terminal["result"]["authored_external_readback_inputs"]["report"]) == digest(workflow),
            "authored outer fixture must retain this exact completed workflow")
    else:
        require(terminal["schema"] == "pokezero.search-over-raw.staged-exploration-run-supervision.v1"
            and terminal["status"] == "STAGED_EXPLORATION_RUN_COMPLETE"
            and digest(terminal["result"]) == digest(workflow), "original production workflow terminal required")
    verify_delegated_closure(directory,auditor_pid,claim["binding_sha256"],_verify_closed_generations)
    cost = read_completed_process_cost(terminal,worker,work_phase="selection")
    recipe_guard()
    plan = plan_after_population(manifest,chain_rows,{k:v["plan"] for k,v in contexts.items()},recipe=recipe)
    root = ledger.directory/"native-after-population"
    original_claim = dict(schema=SCHEMA,plan=plan,plan_sha256=digest(plan),enclosing_binding_sha256=claim["binding_sha256"])
    require(digest(load(root/"claim.json")) == digest(original_claim)
        and {p.name for p in root.iterdir()} == {"claim.json","result.json"},
        "complete original AFTER population claim required; no failed or extra attempts")
    result = workflow["native_after_population"]
    require(digest(load(root/"result.json")) == digest(result), "original AFTER result differs from completed workflow")
    ready = recipe.unqualified_reason is None
    expected_events = plan["events"] if ready else []
    require(type(result["events"]) is list and len(result["events"]) == len(expected_events),
        "complete original AFTER event denominator required")
    bank_paths = []
    for planned,row in zip(expected_events,result["events"]):
        require(set(row) == set(planned)|{"admission","bank"}
            and digest({k:row[k] for k in planned}) == digest(planned), "original AFTER event coordinates changed")
        context = contexts[row["root_id"]]; event = next(e for e in context["plan"].receipt()["events"]
            if e["event_id"] == row["event_id"])
        matches = [r for r in context["replays"] if (r["construction_ordinal"],r["evaluation_ordinal"])
            == (row["canonical_construction_ordinal"],row["evaluation_ordinal"])]
        raw = None
        if matches:
            require(len(matches) == 1, "one original canonical AFTER replay required")
            snapshots = matches[0]["result"]["capture"]["snapshots"]
            index = row["after_snapshot_index"]
            from .search_over_raw_native_prefix import NativePrefixBoundaryRunValidator
            prefix_unreached=(type(matches[0]["validator"]) is NativePrefixBoundaryRunValidator
                and type(index) is int and 0 < len(snapshots) <= index
                and len(matches[0]["validator"].expected["request_ids"])==index+1)
            if not prefix_unreached:
                require(type(index) is int and len(snapshots) == index+1 and 0 <= index < len(snapshots)
                    and snapshots[index]["snapshotIndex"] == snapshots[index]["boundaryGeneration"] == index,
                    "actual evaluated AFTER coordinate required, not PRE or another alias")
                raw = json.loads(snapshots[index]["currentSnapshotJson"])
        admission = row["admission"]; programs = admission["actor_program_sha256"]
        valid_sha = lambda s: type(s) is str and len(s) == 64 and all(c in "0123456789abcdef" for c in s)
        require(type(programs) is dict and set(programs) == {"p1","p2"} and programs["p2"] is None
            and (programs["p1"] is None or valid_sha(programs["p1"])),
            "sampled-p2 must remain unavailable in the null-only completed law")
        validate_null_admission_reason(raw,programs,admission["status"])
        expected_admission = dict(schema=ADMISSION_SCHEMA,event_id=row["event_id"],leaf_plan_sha256=row["leaf_plan_sha256"],
            construction_ordinal=row["canonical_construction_ordinal"],evaluation_ordinal=row["evaluation_ordinal"],
            after_snapshot_index=row["after_snapshot_index"],status=admission["status"],required_cells=8,
            raw_after_sha256=None if raw is None else digest(raw),actor_program_sha256=programs,
            native_inventory_owning_validated=True,original_endpoint_provenance_validated=raw is not None,
            original_p1_history_provenance_validated=programs["p1"] is not None,
            sampled_p2_history_provenance_validated=False,complete_actor_histories_validated=False,
            independent_terminal_labels=False,scored_terminal_labels=0,native_evaluation_input_parity=False,
            resource_cost_complete=False,runtime_qualified=False,scientific_collection_authorized=False,holdout_authorized=False,
            alias_fallback_permitted=False,retries_permitted=False)
        require(digest(admission) == digest(expected_admission), "completed AFTER admission acquired unsupported qualification")
        bank_directory = directory/"after-banks"/digest(dict(plan_sha256=row["leaf_plan_sha256"],event_id=row["event_id"]))
        bank_paths.append(bank_directory)
        cells = [dict(leaf_plan_sha256=row["leaf_plan_sha256"],event_id=row["event_id"],cell_id=c["cell_id"],
            canonical_construction_ordinal=row["canonical_construction_ordinal"],replicate=i,
            directory=str(bank_directory/("cell-"+c["cell_id"])),status="ACTOR_HISTORY_UNAVAILABLE",
            signed_outcome=None,bridge_process_ownership_validated=False) for i,c in enumerate(event["labels"])]
        bank_claim = load(bank_directory/"claim.json")
        initial_cells = [dict(c,status="UNSTARTED_UNCERTAIN") for c in cells]
        require(valid_sha(bank_claim["private_inputs_sha256"]), "reported private AFTER input hash required")
        expected_bank_claim = dict(schema=BANK_SCHEMA,leaf_plan_sha256=row["leaf_plan_sha256"],event=event,
            cell_roster=initial_cells,actor_history_available={"p1":programs["p1"] is not None,"p2":False},
            runtime_sha256=digest(recipe.runtime),private_inputs_sha256=bank_claim["private_inputs_sha256"],
            enclosing_binding_sha256=claim["binding_sha256"],all_eight_cells_consumed=True,
            alias_fallback_permitted=False,retries_permitted=False,scientific_collection_authorized=False)
        expected_bank = dict(schema=BANK_SCHEMA,leaf_plan_sha256=row["leaf_plan_sha256"],event_id=row["event_id"],
            construction_aliases=event["construction_aliases"],canonical_construction_ordinal=row["canonical_construction_ordinal"],
            cells=cells,required_cells=8,bridge_owned_cells=0,complete_transport_results=0,scored_terminal_labels=0,
            native_inventory_owning_validated=False,original_endpoint_provenance_validated=False,
            actor_history_provenance_validated=False,enclosing_auditor_terminal_validated=False,
            independent_terminal_labels=False,resource_cost_complete=False,runtime_qualified=False,
            scientific_collection_authorized=False,holdout_authorized=False,retries_permitted=False)
        require(digest(bank_claim) == digest(expected_bank_claim) and digest(row["bank"]) == digest(expected_bank)
            and digest(load(bank_directory/"result.json")) == digest(expected_bank)
            and {p.name for p in bank_directory.iterdir()} == {"claim.json","result.json"},
            "null-only AFTER bank cannot gain bridge launches, nonnull outcomes or missing cells")
    namespace = directory/"after-banks"
    require({p for p in namespace.iterdir()} == set(bank_paths) if namespace.exists() else not bank_paths,
        "extra/missing AFTER bank namespace")
    expected_result = dict(schema=SCHEMA,plan_sha256=digest(plan),slots=completed_slots(plan),events=result["events"],
        required_native_slots=plan["required_native_slots"],known_retained_events=plan["known_retained_events"],
        known_required_cells=plan["known_required_cells"],unknown_native_slots=plan["unknown_native_slots"],
        consumed_events=len(expected_events),consumed_cells=len(expected_events)*8,after_process_launches=0,
        scored_terminal_labels=0,independent_terminal_labels=False,resource_cost_complete=False,runtime_qualified=False,
        diagnostic_results_used_for_selection=False,scientific_collection_authorized=False,holdout_authorized=False,
        unqualified_reason=recipe.unqualified_reason)
    require(digest(result) == digest(expected_result) and _inventory(paths) == inventory
        and before == digest(dict(workflow=workflow,manifest=manifest,chains=chain_rows,recipe=_state_payload(recipe),
            terminal=expected_outer_terminal)), "original AFTER population changed during final outer callbacks")
    recipe_guard()
    return dict(status="COMPLETE_NULL_ONLY_AFTER_ROSTER_REVALIDATED",required_native_slots=plan["required_native_slots"],
        known_retained_events=plan["known_retained_events"],known_required_cells=plan["known_required_cells"],
        unknown_native_slots=plan["unknown_native_slots"],consumed_cells=len(expected_events)*8,
        after_process_launches=0,scored_terminal_labels=0,enclosing_auditor_terminal_validated=True,
        p1_history_externally_reconstructed=False,private_input_hash_externally_reconstructed=False,
        reported_in_auditor_p1_history_checks=True,outer_process_cost=cost,resource_cost_complete=False,
        runtime_qualified=False,scientific_measurement_validated=False,holdout_authorized=False)
