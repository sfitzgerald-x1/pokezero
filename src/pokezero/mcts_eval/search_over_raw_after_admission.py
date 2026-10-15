"""Original evaluated Showdown endpoint and private actor-history admission.

No clone, inference, source generation or scored label. Native input parity and
physics remain separate from this original-owner and observation-history join.
"""
from copy import deepcopy
from dataclasses import dataclass
import json

from .search_over_raw import digest,require
from .search_over_raw_after_owner import _path
from .search_over_raw_attempt_cost import _inventory
from .search_over_raw_diagnostic_producer import OwnedFixedWorkDiagnostic
from .search_over_raw_consumed_materialization import _original_diagnostic_owner_guard
from .search_over_raw_native_leaf_cells import NativeLeafCellPlan,freeze_owned_native_leaf_cells
from .search_over_raw_native_chain import OwnedNativeMaterializationPlan
from .search_over_raw_consumed_materialization import OwnedNativeReplayPlan
from .search_over_raw_actor_replay_projection import read_owned_original_actor_replay,projection_identity
from .search_over_raw_actor_projection import compile_actor_program
from .search_over_raw_world_inputs import _state_payload
from .search_over_raw_boundary_transport import _requests
from .search_over_raw_materialization_run import _join_result
from .search_over_raw_boundary_run import _join_raw_result

SCHEMA="pokezero.search-over-raw.original-native-after-admission.v1"


def _payload(raw,programs,receipt):
    return digest(dict(raw_after=raw,programs=programs,receipt=receipt))


@dataclass(frozen=True)
class OwnedNativeAfterAdmission:
    raw_after: object
    programs: dict
    receipt: dict
    original_sha256: str
    original_guard: object

    def __reduce_ex__(self,protocol):
        raise TypeError("private original AFTER admission cannot enter worker transport")

    def validate(self):
        self.original_guard()
        require(_payload(self.raw_after,self.programs,self.receipt)==self.original_sha256,
            "original AFTER endpoint/program/admission changed")
        return deepcopy(self.receipt)


def admit_owned_native_after(chain,*,ledger,event_id,expected_plan_sha256,source_config,native_extract):
    """Select only the original canonical AFTER coordinate, without fallback.

    Contradictory ownership/rosters/receipts fail closed. An originally blocked
    canonical path or genuinely unavailable actor history stays explicit null.
    Sampled p2 never receives the original source opponent's private history.
    """
    require(type(chain) is dict and set(chain)=={"native_owner","materializations","replays",
        "leaf_cell_plan","materialization_plan","replay_plan"},"original complete private native chain required")
    owner=chain["native_owner"];plan=chain["leaf_cell_plan"]
    require(type(owner) is OwnedFixedWorkDiagnostic and type(plan) is NativeLeafCellPlan
        and plan.sha256==expected_plan_sha256 and callable(native_extract)
        and type(chain["materialization_plan"]) is OwnedNativeMaterializationPlan
        and type(chain["replay_plan"]) is OwnedNativeReplayPlan,"original owning native leaf plan required")
    mats,replays=chain["materializations"],chain["replays"]
    require(type(mats) is list and type(replays) is list,"original complete constructor/replay lists required")
    refs=tuple(chain.values());mat_refs=tuple(mats);replay_refs=tuple(replays)
    owner_guard=_original_diagnostic_owner_guard(owner.selected,mats,replays,owner.reader)
    readers=[owner.reader]+[r["validator"] for r in mats+replays]
    readouts={str(r.directory):deepcopy(r.last_validation) for r in readers}
    root=_path(ledger.directory/"native-diagnostic-chains"/digest(owner.job.statistics_root_id))
    files=_inventory([root]);config_sha=digest(_state_payload(source_config));plan_bytes=plan.payload_utf8
    other_plans=tuple((chain[k],chain[k].payload_utf8) for k in ("materialization_plan","replay_plan"))
    claim=json.loads((root/"claim.json").read_bytes())
    replay_claim=json.loads((root/"replay-plan.json").read_bytes())
    require(claim["leaf_cell_plan_sha256"]==expected_plan_sha256
        and digest(claim["leaf_cell_plan"])==digest(plan.receipt())
        and claim["native_selected_sha256"]==owner.selected_sha256
        and claim["materialization_plan_sha256"]==chain["materialization_plan"].sha256
        and digest(claim["materialization_plan"])==digest(chain["materialization_plan"].receipt())
        and replay_claim["plan_sha256"]==chain["replay_plan"].sha256
        and digest(replay_claim["private_plan"])==digest(chain["replay_plan"].receipt()),
        "original prospective leaf/constructor/replay claims differ from private plans")
    declared_mats=chain["materialization_plan"].receipt()["records"]
    declared_replays=chain["replay_plan"].receipt()["records"]
    require(sorted(r["construction_ordinal"] for r in mats)==list(range(4))
        and len(mats)==len(declared_mats)==4
        and all(type(r["construction_ordinal"]) is int for r in mats+declared_mats)
        and all(type(r["construction_ordinal"]) is int and type(r["evaluation_ordinal"]) is int
            for r in replays+declared_replays)
        and all(any(r["construction_ordinal"]==d["construction_ordinal"]
            and str(r["validator"].directory)==d["attempt_directory"] for r in mats) for d in declared_mats)
        and sorted((r["construction_ordinal"],r["evaluation_ordinal"],str(r["validator"].directory)) for r in replays)
            ==sorted((d["construction_ordinal"],d["evaluation_ordinal"],d["attempt_directory"])
                for d in declared_replays if d["job_spec"] is not None),
        "complete original planned constructor/replay roster required before canonical selection")
    projection=None;projection_sha=None;raw=None;raw_sha=None;programs={"p1":None,"p2":None};programs_sha=digest(programs)
    def guard():
        # Direct original-object/file checks after the last reader/encoder.
        owner._original_inputs();owner_guard(readouts)
        require(len(chain)==len(refs) and all(a is b for a,b in zip(chain.values(),refs))
            and len(mats)==len(mat_refs) and all(a is b for a,b in zip(mats,mat_refs))
            and len(replays)==len(replay_refs) and all(a is b for a,b in zip(replays,replay_refs))
            and plan.payload_utf8==plan_bytes and _inventory([root])==files
            and digest(_state_payload(source_config))==config_sha
            and all(p.payload_utf8==b for p,b in other_plans)
            and (raw is None or digest(raw)==raw_sha) and digest(programs)==programs_sha
            and (projection is None or projection_identity(projection)==projection_sha),
            "original AFTER admission roster/owners/inputs changed")
    guard()
    # Every original row must clear its actual worker evidence, including rows
    # not selected for this event. Entry-time hashes cannot promote a modified
    # noncanonical result/reader into an "original" baseline.
    for row in mats:
        declared=next(d for d in declared_mats if d["construction_ordinal"]==row["construction_ordinal"])
        reader=row["validator"]
        require(digest(reader.expected)==declared["expected_inputs_sha256"],
            "original constructor reader differs from frozen plan inputs")
        _join_result(row["result"],reader.expected,row["result"]["capture"])
        require(reader(row["result"]["capture"])==digest(row["result"]["capture"]),
            "every original constructor owning reader must clear")
        guard()
    for row in replays:
        declared=next(d for d in declared_replays if (d["construction_ordinal"],d["evaluation_ordinal"])
            ==(row["construction_ordinal"],row["evaluation_ordinal"]))
        reader=row["validator"]
        require(digest(reader.expected)==declared["job_spec"]["expected_inputs_sha256"],
            "original replay reader differs from frozen plan inputs")
        _join_raw_result(row["result"],reader.expected,row["result"]["capture"])
        require(reader(row["result"]["capture"])==digest(row["result"]["capture"]),
            "every original replay owning reader must clear")
        guard()
    fresh=freeze_owned_native_leaf_cells(owner,ledger=ledger)
    require(fresh.payload_utf8==plan_bytes,"original AFTER plan differs from independently owned native evidence")
    guard()
    events=[e for e in fresh.receipt()["events"] if e["event_id"]==event_id]
    require(len(events)==1,"one original retained native event required")
    event=events[0];ordinal=event["canonical_construction_ordinal"]
    coordinate=(ordinal,event["evaluation_ordinal"])
    planned=[d for d in declared_replays if (d["construction_ordinal"],d["evaluation_ordinal"])==coordinate]
    require(len(planned)==1,"original canonical evaluation coordinate missing from complete replay plan")
    reason="CANONICAL_PATH_UNAVAILABLE";endpoint_valid=False
    endpoint_unreached=False
    if planned[0]["job_spec"] is not None:
        from .search_over_raw_native_prefix import NativePrefixBoundaryRunValidator
        original_replay=next(r for r in replays if (r["construction_ordinal"],r["evaluation_ordinal"])==coordinate)
        if type(original_replay["validator"]) is NativePrefixBoundaryRunValidator:
            required=event["required_after_snapshot_index"]+1
            require(len(original_replay["validator"].expected["request_ids"])==required,
                "original prefix replay must retain the evaluated endpoint denominator")
            endpoint_unreached=len(original_replay["result"]["capture"]["snapshots"])<required
    if planned[0]["job_spec"] is not None and not endpoint_unreached:
        material=next(r for r in mats if r["construction_ordinal"]==ordinal)
        replay=next(r for r in replays if (r["construction_ordinal"],r["evaluation_ordinal"])==coordinate)
        projection=read_owned_original_actor_replay(owner,material,replay,ledger=ledger,
            source_config=source_config,native_extract=native_extract)
        projection_sha=projection_identity(projection)
        guard()
        snapshots=replay["result"]["capture"]["snapshots"]
        index=event["required_after_snapshot_index"]
        require(type(index) is int and 0 <= index < len(snapshots) and len(snapshots)==index+1
            and len(projection.observations)==index+1,"exact full replay AFTER endpoint roster required")
        final=snapshots[index];raw=json.loads(final["currentSnapshotJson"])
        raw_sha=digest(raw)
        require(final["snapshotIndex"]==index and final["boundaryGeneration"]==index,
            "actual evaluated AFTER transport index/generation required")
        _requests(raw);endpoint_valid=True
        if raw["battle"]["ended"]:reason="TERMINAL_AFTER_REQUIRES_SEPARATE_LAW"
        else:
            programs["p1"]=projection.actor_program(index)
            programs_sha=digest(programs)
            if programs["p1"] is not None:
                compiled=compile_actor_program(programs["p1"],config=source_config,
                    expected_program_sha256=digest(programs["p1"]),expected_config_sha256=config_sha)
                require(digest(_state_payload(compiled.observations[-1]))==digest(_state_payload(projection.observations[index]))
                    and digest(_state_payload(compiled.materialization_states[-1]))
                        ==digest(_state_payload(projection.materialization_states[index]))
                    and digest(compiled.materialization_states[-1].self_request)==digest(raw["boundaryRequests"]["p1"]),
                    "actual AFTER program/input/request differs from original owning actor projection")
                reason="SAMPLED_ACTOR_HISTORY_UNAVAILABLE"
            else:reason=projection.receipt["unavailable_reasons"][index]
        guard()
    receipt=dict(schema=SCHEMA,event_id=event_id,leaf_plan_sha256=expected_plan_sha256,
        construction_ordinal=ordinal,evaluation_ordinal=event["evaluation_ordinal"],
        after_snapshot_index=event["required_after_snapshot_index"],
        status=reason,required_cells=8,raw_after_sha256=None if raw is None else digest(raw),
        actor_program_sha256={s:None if p is None else digest(p) for s,p in programs.items()},
        native_inventory_owning_validated=True,original_endpoint_provenance_validated=endpoint_valid,
        original_p1_history_provenance_validated=programs["p1"] is not None,
        sampled_p2_history_provenance_validated=False,complete_actor_histories_validated=False,
        independent_terminal_labels=False,scored_terminal_labels=0,native_evaluation_input_parity=False,
        resource_cost_complete=False,runtime_qualified=False,scientific_collection_authorized=False,holdout_authorized=False,
        alias_fallback_permitted=False,retries_permitted=False)
    result=OwnedNativeAfterAdmission(raw,programs,receipt,_payload(raw,programs,receipt),guard)
    result.validate()
    return result


def execute_original_native_after_bank(chain,*,ledger,ownership,event_id,expected_plan_sha256,
        source_config,native_extract,runtime,policies,null_only=False):
    """Original-owner entry; no caller-supplied programs/endpoint admission.

    Current sampled-p2 history is unavailable, so all original eight cells are
    consumed as null without launches. This does not authorize collection or
    turn a bridge-owned terminal observation into an independent scored label.
    """
    from .search_over_raw_after_bank import execute_owned_after_bank
    from .search_over_raw_outcome_run import verify_outcome_runtime
    admission=admit_owned_native_after(chain,ledger=ledger,event_id=event_id,
        expected_plan_sha256=expected_plan_sha256,source_config=source_config,native_extract=native_extract)
    require(type(null_only) is bool,"explicit AFTER execution scope required")
    if null_only:
        require(policies is None and admission.programs["p2"] is None
            and admission.receipt["complete_actor_histories_validated"] is False,
            "null-only population cannot launch newly available paired histories")
    owner=chain["native_owner"]
    verify_outcome_runtime(runtime,source_config)
    require(runtime["checkpoint_path"]==owner.job.checkpoint_contract.checkpoint_path
        and runtime["files"][runtime["checkpoint_path"]]==owner.job.checkpoint_contract.checkpoint_sha256,
        "original champion checkpoint required for AFTER bank")
    admission.validate()
    bank=execute_owned_after_bank(ownership=ownership,plan=chain["leaf_cell_plan"],
        expected_plan_sha256=expected_plan_sha256,event_id=event_id,raw_after=admission.raw_after,
        programs=admission.programs,config=source_config,runtime=runtime,policies=policies,_original_admission=admission)
    bank.validate();admission.validate();bank._check_original()
    return admission,bank
