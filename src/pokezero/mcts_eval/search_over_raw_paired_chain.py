"""Original native event -> canonical paired endpoint -> raw/raw label jobs.

This is the opt-in collector join, not a new sampler or eligibility-by-outcome
alias selector. Every retained event and all eight cells remain in the report.
"""
from copy import deepcopy
from pathlib import Path
import hashlib
import json

from .search_over_raw import digest,require
from .search_over_raw_paired_materialization import context_law
from .search_over_raw_paired_replay import replay_law,PairedPrefixRunValidator
from .search_over_raw_paired_labels import prepare_paired_label_job
from .search_over_raw_outcomes import paired_player_names,_root_requested
from .search_over_raw_native_leaf_cells import leaf_cell_law,NativeLeafCellPlan

NAMESPACE="paired-native-after.v1"


def paired_leaf_law():
    return dict(leaf_cell_law(),schema="pokezero.search-over-raw.native-after-leaf-cell-plan.v2",
        randomness=dict(namespace=NAMESPACE,root_id="sha256(event_id,sample_seed,paired_label_schema)",
            per_step="rng_seed(namespace,root_id,replicate,step,'chance')",
            stream_key="rng_seed(namespace,root_id,replicate)"))


def paired_chain_law():
    return dict(schema="pokezero.search-over-raw.paired-native-chain.v1",
        context=context_law(),replay=replay_law(),leaf_cells=paired_leaf_law(),
        canonical_alias_only=True,unavailable_cells="all_eight_null_no_fallback",
        request_correspondence="actual_requested_seat_action_catalogs",
        value_diagnostics="all_events_unknown_bounds_and_separate_scoped_field_strata_not_native_state_calibration",
        native_prediction_input_parity=False,branch_transition_validated=False,
        scientific_collection_authorized=False)


def paired_label_coordinates(event,sample_seed):
    return digest(dict(event_id=event["event_id"],sample_seed=sample_seed,
        paired_label_schema="pokezero.search-over-raw.paired-native-after-labels.v1")),NAMESPACE


def request_correspondence(snapshot):
    """Only the direct check needed for policy-index -> real request choices.

    The normal observation and direct-request paths share this layout
    translator. No claim of engine equivalence or native input equality.
    """
    from ..local_showdown import requested_players_from_requests
    from ..showdown import showdown_choices_for_request
    requested=_root_requested(snapshot)
    if list(requested_players_from_requests(snapshot.latest_requests))!=requested:return None
    catalogs={}
    for seat in requested:
        raw=snapshot.bridge_snapshot["boundaryRequests"][seat]
        paired=snapshot.latest_requests.get(seat);parsed=snapshot.replay.requests.get(seat)
        if not all(type(row) is dict and row.get("side",{}).get("id")==seat for row in (raw,paired,parsed)):return None
        actual=showdown_choices_for_request(raw,seat)
        if not actual or digest(actual)!=digest(showdown_choices_for_request(paired,seat)) \
                or digest(actual)!=digest(showdown_choices_for_request(parsed,seat)):return None
        def identities(request):
            active=request.get("active",[])
            moves=active[0].get("moves",[]) if active else []
            require(all(type(move) is dict and type(move.get("id")) is str and move["id"] for move in moves),
                "actual requested move-slot identities required")
            party=request["side"].get("pokemon")
            require(type(party) is list and all(type(mon) is dict for mon in party),
                "actual ordered request party required")
            # Full original seat-private rows distinguish reordered bench
            # members even with duplicate species/nicknames. This compares
            # three views of ONE actual request, not native/Showdown physics.
            return dict(moves=[move["id"] for move in moves],party=party)
        if digest(identities(raw))!=digest(identities(paired)) or digest(identities(raw))!=digest(identities(parsed)):return None
        catalogs[seat]=actual
    return dict(requested=requested,action_catalog_sha256=digest(catalogs),
        native_prediction_input_parity=False,full_engine_equivalence=False)


def pin_original_result(directory,*,job_sha256,readout):
    """Original report's file inventory, BEFORE any private job/context load."""
    directory=Path(directory)
    require(directory==directory.resolve() and type(readout) is dict
        and readout.get("process_receipts_validated") is True,"original paired owning readout required")
    hashes=readout["receipt_file_sha256"]
    require(type(hashes) is dict and "terminal.json" in hashes
        and all(type(name) is str and Path(name).name==name
            and (directory/name)==(directory/name).resolve()
            and hashlib.sha256((directory/name).read_bytes()).hexdigest()==sha for name,sha in hashes.items())
        and hashlib.sha256((directory/"job.pickle").read_bytes()).hexdigest()==job_sha256,
        "paired files differ from original independently frozen owning readout")
    raw=(directory/"terminal.json").read_bytes()
    require(hashlib.sha256(raw).hexdigest()==hashes["terminal.json"],"original paired terminal byte drift")
    return json.loads(raw)["result"]


def paired_label_descriptors(plan,replays,*,config,runtime,attempt_root):
    """Fresh original replay reads; never select a different successful alias."""
    require(type(plan) is NativeLeafCellPlan,"original immutable native leaf plan required")
    frozen=plan.payload_utf8;saved=plan.receipt()
    require(saved["schema"]=="pokezero.search-over-raw.native-after-leaf-cell-plan.v2"
        and digest(saved["law"])==digest(paired_leaf_law()),"explicit paired continuation RNG/context law required")
    indexed={}
    for row in replays:
        require(type(row["validator"]) is PairedPrefixRunValidator,"paired replay owning reader required")
        key=(row["construction_ordinal"],row["evaluation_ordinal"])
        require(key not in indexed,"duplicate original paired replay coordinate")
        indexed[key]=row
    records=[]
    for event in saved["events"]:
        key=(event["canonical_construction_ordinal"],event["evaluation_ordinal"])
        path=Path(attempt_root)/event["event_id"]
        require(path.is_absolute() and path==path.resolve(),"original canonical paired label namespace required")
        require(key[0]==min(event["construction_aliases"]),"frozen original canonical construction alias required")
        record=dict(event_id=event["event_id"],construction_ordinal=key[0],evaluation_ordinal=key[1],
            event=deepcopy(event),status="UNSTARTED_UNCERTAIN",reason=None,attempt_directory=str(path),
            request_correspondence=None,job=None)
        replay=indexed.get(key)
        if replay is None:record["reason"]="CANONICAL_REPLAY_UNAVAILABLE"
        else:
            reader=replay["validator"];result=replay["result"]
            require(reader(result["capture"])==digest(result["capture"]),"original paired replay ownership required")
            expected=reader.expected;origin=expected["root_origin"]
            require(origin["native_selected_sha256"]==event["native_selected_sha256"]
                and origin["native_capture_sha256"]==event["native_capture_sha256"]
                and origin["construction_ordinal"]==key[0] and origin["evaluation_ordinal"]==key[1]
                and len(expected["request_ids"])==event["required_after_snapshot_index"]+1,
                "paired endpoint must be the exact original native prediction coordinate")
            if result["capture"]["termination"]["status"]!="FULL_PATH_OBSERVED":record["reason"]="DESIGNATED_AFTER_UNREACHED"
            else:
                # Terminal native/replay disagreement cannot become a made-up
                # continuation label. Keep this event's full eight cells null.
                from .search_over_raw_paired_replay import read_paired_prefix,context_identity
                _,snapshot=read_paired_prefix(result["capture"],process_records=result["capture"]["records"],
                    snapshot=reader.snapshot,expected=reader.expected)
                if snapshot.terminal is not None:record["reason"]="DESIGNATED_AFTER_TERMINAL"
                elif event["model_subject"] not in _root_requested(snapshot):record["reason"]="SUBJECT_REQUEST_UNAVAILABLE"
                else:
                    paired_player_names(snapshot)
                    correspondence=request_correspondence(snapshot)
                    if correspondence is None:record["reason"]="PAIRED_REQUEST_CORRESPONDENCE_UNAVAILABLE"
                    else:
                        root_id,namespace=paired_label_coordinates(event,saved["sample_seed"])
                        root=dict(root_id=root_id,source_seed=saved["run_binding"]["selection_seed"],
                            event_id=event["event_id"],leaf_cell_plan_sha256=plan.sha256,
                            model_signed_value=event["model_signed_value"],model_self_value01_bits=event["model_self_value01_bits"],
                            native_event=deepcopy({k:v for k,v in event.items() if k!="labels"}),
                            request_correspondence=correspondence)
                        record["job"]=prepare_paired_label_job(reader,result,config=config,runtime=runtime,
                            root_binding=root,subject=event["model_subject"],namespace=namespace)
                        require(context_identity(record["job"].snapshot)==context_identity(snapshot),
                            "paired label preparation differs from original folded endpoint")
                        record.update(status="PLANNED_NOT_ADMITTED",request_correspondence=correspondence)
        records.append(record)
    require(plan.payload_utf8==frozen,"original paired leaf plan changed during owning callbacks")
    return records


def paired_label_report(descriptor,run=None):
    report={k:deepcopy(v) for k,v in descriptor.items() if k!="job"}
    report["cells"]=deepcopy(descriptor["event"]["labels"])
    report["owning_readout"]=None
    if run is not None:
        result=run["result"];reader=run["validator"]
        require(reader(result["bank"])==digest(result["bank"]),"original paired label owner required")
        report.update(status="LABELS_RECORDED",owning_readout=deepcopy(reader.last_validation),
            job_sha256=reader.job_sha,binding_sha256=reader.binding,expected_inputs_sha256=digest(reader.expected))
        for cell,row in zip(report["cells"],result["outcomes"]):
            require(cell["replicate"]==row["replicate"],"original paired replicate order drift")
            cell.update(status="COMPLETE" if row["status"]=="COMPLETE" else "CAPPED_UNCERTAIN",
                signed_outcome=row["signed_outcome"],boundaries=row["boundaries"])
    return report


def paired_value_diagnostics(plan,label_reports,replay_join):
    """Describe all frozen predictions versus outcomes at paired coordinates.

    Call only after original owning readers have validated BOTH inputs. Scoped
    field agreement is a stratum, never an eligibility filter or physics claim.
    The unstarted/capped cells stay in every applicable eight-cell denominator.
    """
    from .search_over_raw_native_edge import ENDPOINT_FIELDS
    from .search_over_raw_value_diagnostics import calibration_bounds
    require(type(plan) is NativeLeafCellPlan,"original paired leaf plan required")
    saved=plan.receipt()
    require(saved["schema"]=="pokezero.search-over-raw.native-after-leaf-cell-plan.v2"
        and digest(saved["law"])==digest(paired_leaf_law())
        and replay_join["ancestry_root_join"]["native_selected_sha256"]==saved["native_selected_sha256"],
        "paired prediction and scoped replay populations differ")
    before=digest(dict(reports=label_reports,replay=replay_join));frozen=plan.payload_utf8
    events=saved["events"]
    require(type(label_reports) is list and len(label_reports)==len(events),"all original paired events required")
    indexed={}
    for row in replay_join["records"]:
        key=(row["construction_ordinal"],row["evaluation_ordinal"])
        require(key not in indexed,"duplicate scoped replay coordinate")
        indexed[key]=row
    require(set(indexed)=={(alias,event["evaluation_ordinal"]) for event in events
        for alias in event["construction_aliases"]},"full original scoped alias/event roster required")
    names=("SCOPED_FIELDS_MATCH","SCOPED_FIELDS_MISMATCH","SCOPED_FIELDS_UNCHECKABLE")
    strata={name:[] for name in names};leaves=[];records=[]
    for event,report in zip(events,label_reports):
        require(report["event_id"]==event["event_id"] and digest(report["event"])==digest(event)
            and report["construction_ordinal"]==event["canonical_construction_ordinal"]
            and report["evaluation_ordinal"]==event["evaluation_ordinal"]
            and type(report["cells"]) is list and len(report["cells"])==8,
            "original paired prediction/canonical coordinate/eight-cell roster required")
        labels=[]
        for original,cell in zip(event["labels"],report["cells"]):
            require(all(type(cell[k]) is type(original[k]) and cell[k]==original[k]
                for k in ("cell_id","replicate","chance_stream_key")),"paired calibration cell identity drift")
            status=cell["status"];value=cell["signed_outcome"]
            require(status in {"COMPLETE","CAPPED_UNCERTAIN","UNSTARTED_UNCERTAIN"}
                and (type(value) in (int,float) and value in (-1,0,1) if status=="COMPLETE" else value is None),
                "paired uncertainty cannot be dropped or scored")
            labels.append(dict(replicate=cell["replicate"],signed_outcome=value,
                status="DEADLINE_UNCERTAIN" if status=="UNSTARTED_UNCERTAIN" else status))
        leaf=dict(model_signed_value=event["model_signed_value"],labels=labels)
        replay=indexed[(event["canonical_construction_ordinal"],event["evaluation_ordinal"])]
        count=event["required_after_snapshot_index"]+1
        require(replay["native_capture_sha256"]==event["native_capture_sha256"]
            and replay["required_boundaries"]==count and len(replay["boundary_observations"])==count,
            "scoped diagnostics differ from original designated AFTER coordinate")
        counts={status:0 for status in ("MATCH","MISMATCH","UNCHECKABLE")}
        for ordinal,boundary in enumerate(replay["boundary_observations"]):
            require(type(boundary["ordinal"]) is int and boundary["ordinal"]==ordinal,
                "original scoped boundary roster required")
            for field in ENDPOINT_FIELDS+(("action_identity",) if ordinal else ()):
                status=boundary["observed_required_fields"][field]
                require(status in counts,"scoped direct-field classification drift")
                counts[status]+=1
        name=names[1] if counts["MISMATCH"] else names[2] if counts["UNCHECKABLE"] else names[0]
        strata[name].append(leaf);leaves.append(leaf)
        calibration=next(row for row in calibration_bounds([leaf]) if row["leaves"])
        records.append(dict(event_id=event["event_id"],canonical_construction_ordinal=event["canonical_construction_ordinal"],
            evaluation_ordinal=event["evaluation_ordinal"],native_capture_sha256=event["native_capture_sha256"],
            model_signed_value=event["model_signed_value"],canonical_replay_record_sha256=digest(replay),
            label_status=report["status"],unavailable_reason=report["reason"],
            required_labels=8,unknown_labels=calibration["unknown_labels"],
            win_score_interval=calibration["win_score_interval"],squared_error_interval=calibration["squared_error_interval"],
            scoped_field_stratum=name,scoped_field_counts=counts,
            supported_prefix_observations=deepcopy(replay["supported_prefix_observations"]),
            projected_successor_support=deepcopy(replay.get("projected_successor_support")),
            branch_transition_validated=False,native_evaluated_state_equivalence=False))
    require(before==digest(dict(reports=label_reports,replay=replay_join)) and plan.payload_utf8==frozen,
        "paired descriptive inputs changed")
    return dict(schema="pokezero.search-over-raw.paired-value-diagnostics.v1",leaf_cell_plan_sha256=plan.sha256,
        population="all_frozen_unique_native_evaluation_events_canonical_alias_only",
        target="raw_raw_win_score_from_designated_Showdown_replay_coordinate_not_proven_native_state",
        weighting="event_weighted_not_independent_trials",required_events=len(events),required_labels=8*len(events),
        unknown_labels=sum(row["unknown_labels"] for row in records),events=records,
        calibration=calibration_bounds(leaves),scoped_strata={name:dict(events=len(rows),required_labels=8*len(rows),
            calibration=calibration_bounds(rows)) for name,rows in strata.items()},
        compared_fields=list(ENDPOINT_FIELDS)+["action_identity"],native_pp_class="engine_pp_model",
        scoped_matching_is_not_physics_validation=True,complete_case_filter_applied=False,
        branch_transition_validated=False,native_evaluated_state_equivalence=False,
        scientific_calibration_evidence=False,labels_used_for_selection=False,runtime_qualified=False)
