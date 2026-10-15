"""Original-subject history into owned Showdown replay input projections.

No source, simulator step, model, native-evaluation parity or terminal label.
The pure projection is distinct from the owning reader. Unknown sampled-p2
history remains unavailable, never copied from the original source opponent.
"""
from copy import deepcopy
from dataclasses import dataclass
import json

from .search_over_raw import digest, require
from .search_over_raw_actor_projection import ActorObservationFold
from .search_over_raw_actor_replay_inputs import SCHEMA as INPUT_SCHEMA, read_actor_replay_inputs
from .search_over_raw_source_actor_history import RetainedSourceActorProgram, retained_program_identity
from .search_over_raw_world_inputs import _state_payload

SCHEMA = "pokezero.search-over-raw.original-actor-replay-projection.v1"


@dataclass(frozen=True)
class OriginalActorReplayProjection:
    observations: tuple
    materialization_states: tuple
    receipt: dict
    program: object
    program_cutoffs: tuple

    def __reduce_ex__(self, protocol):
        raise TypeError("private actor replay projection cannot enter worker/durable transport")

    def actor_program(self,index):
        require(type(index) is int and 0 <= index < len(self.observations)
            and len(self.program_cutoffs)==len(self.observations),"original actor boundary cutoff required")
        cutoff=self.program_cutoffs[index]
        if cutoff is None:return None
        require(type(self.program) is dict and type(cutoff) is int and 0 < cutoff <= len(self.program["events"])
            and self.program["events"][cutoff-1]["kind"]=="observe","original already-observed actor cutoff required")
        return dict(deepcopy(self.program),events=deepcopy(self.program["events"][:cutoff]))


def projection_identity(value):
    require(type(value) is OriginalActorReplayProjection, "exact actor replay projection required")
    return digest(dict(observations=_state_payload(value.observations),
        states=_state_payload(value.materialization_states), receipt=value.receipt,
        program=value.program,program_cutoffs=value.program_cutoffs))


def _own_anchors(request):
    if type(request) is not dict or type(request.get("side")) is not dict:
        return None
    side = request["side"]
    if side.get("id") != "p1" or type(side.get("pokemon")) is not list or not 1 <= len(side["pokemon"]) <= 6:
        return None
    # Current actor-known moves/items/base stats may not change silently.
    # Condition/active flags and active PP remain in the ACTUAL requests used
    # below, not in an immutable party anchor. This is not a native-PP gate.
    fields = ("ident", "details", "moves", "baseAbility", "ability", "item", "stats")
    if not all(type(mon) is dict and type(mon.get("ident")) is str
            and type(mon.get("details")) is str for mon in side["pokemon"]):
        return None
    if len({mon["ident"] for mon in side["pokemon"]}) != len(side["pokemon"]):
        return None
    return [{key: deepcopy(mon.get(key)) for key in fields} for mon in side["pokemon"]]


def _active_moves(request):
    active = request.get("active")
    if type(active) is not list or len(active) != 1 or type(active[0]) is not dict:
        return []
    moves = active[0].get("moves")
    return moves if type(moves) is list and all(type(move) is dict for move in moves) else []


def project_original_actor_replay(source, replay_inputs, *, config,
        expected_source_identity, expected_replay_sha256):
    """Pure actual-encoder projection; external owning joins are NOT inferred.

    Caller freezes both inputs independently. Every original replay coordinate
    remains present. Inactive/terminal rows do not call the encoder or advance
    its annotations. An actionable row without its own request stays null.
    """
    require(source is None or type(source) is RetainedSourceActorProgram,
        "original private source actor program or explicit absence required")
    require((source is None and expected_source_identity is None or source is not None
        and retained_program_identity(source) == expected_source_identity)
        and type(replay_inputs) is dict and digest(replay_inputs) == expected_replay_sha256
        and replay_inputs.get("schema") == INPUT_SCHEMA and replay_inputs.get("actor") in {"p1", "p2"}
        and replay_inputs.get("materialized_shell_prefix_used") is False
        and replay_inputs.get("snapshot_transport_consistent") is True,
        "independently bound source and actor replay inputs required")
    rows = deepcopy(replay_inputs.get("rows"))
    require(type(rows) is list and rows and all(type(row) is dict
        and type(row.get("snapshot_index")) is int and row["snapshot_index"] == index
        and type(row.get("observed_generation")) is int and row["observed_generation"] == index
        and type(row.get("actor_requested")) is bool
        and digest(row.get("visible_suffix")) == row.get("visible_suffix_sha256")
        and digest(row.get("own_request")) == row.get("own_request_sha256")
        for index, row in enumerate(rows)), "complete ordered original actor-boundary roster required")
    require(rows[0]["visible_suffix"] == [], "materialized shell prefix cannot enter actor history")
    config_sha = digest(_state_payload(config))
    nulls = [None] * len(rows)
    reasons, values, states, pp_changes = [None] * len(rows), list(nulls), list(nulls), []
    output_bindings = {}
    program=None;cutoffs=[None]*len(rows)
    if replay_inputs["actor"] == "p2":
        reasons = ["SAMPLED_ACTOR_HISTORY_UNAVAILABLE"] * len(rows)
    elif source is None:
        reasons = ["ORIGINAL_ACTOR_HISTORY_ABSENT"] * len(rows)
    elif source.receipt["status"] != "INPUT_CORRESPONDENCE_MATCHED" or not source.receipt["complete_actor_history_validated"]:
        reasons = ["ORIGINAL_SOURCE_ENCODING_UNAVAILABLE"] * len(rows)
    else:
        require(source.program["actor"] == "p1" and source.receipt["config_sha256"] == config_sha,
            "original source actor/config changed")
        before_state = source.compiled.materialization_states[-1]
        known = _own_anchors(before_state.self_request) if before_state is not None else None
        root = _own_anchors(rows[0]["own_request"])
        if known is None or root is None:
            reasons = ["OWN_REQUEST_ANCHOR_UNAVAILABLE"] * len(rows)
        elif digest(known) != digest(root):
            reasons = ["OWN_KNOWLEDGE_ANCHOR_MISMATCH"] * len(rows)
        else:
            program=deepcopy(source.program)
            next_coordinate=max(e["coordinate"] for e in program["events"] if e["kind"]=="observe")+1
            fold = ActorObservationFold(config=config, actor="p1", opening_request=source.program["opening_request"],
                expected_opening_sha256=digest(source.program["opening_request"]),
                battle_id=source.program.get("battle_id","actor-projection"))
            rebuilt = []
            for event in source.program["events"]:
                if event["kind"] == "protocol": fold.feed_visible(event["lines"])
                elif event["kind"] == "own_request": fold.feed_own_request(event["request"])
                else: rebuilt.append(fold.observe())
            require(digest(_state_payload(tuple(rebuilt))) == digest(_state_payload(source.compiled.observations)),
                "original source encoding changed while rebuilding chronology")
            source_moves = _active_moves(before_state.self_request)
            actual_moves = _active_moves(rows[0]["own_request"])
            for index, (old, new) in enumerate(zip(source_moves, actual_moves)):
                if old.get("pp") != new.get("pp") or old.get("maxpp") != new.get("maxpp"):
                    pp_changes.append(dict(slot=index, source_pp=old.get("pp"), actual_pp=new.get("pp"),
                        source_maxpp=old.get("maxpp"), actual_maxpp=new.get("maxpp")))
            history_available = True
            for index, row in enumerate(rows):
                fold.feed_visible(row["visible_suffix"])
                program["events"].append(dict(kind="protocol",lines=deepcopy(row["visible_suffix"])))
                request = row["own_request"]
                if type(request) is dict and type(request.get("side")) is dict:
                    fold.feed_own_request(request)
                    program["events"].append(dict(kind="own_request",request=deepcopy(request)))
                if not row["actor_requested"]:
                    reasons[index] = "TERMINAL" if row["boundary_kind"] == "terminal" else "ACTOR_NOT_REQUESTED"
                elif type(request) is not dict or type(request.get("side")) is not dict:
                    history_available = False
                    reasons[index] = "ACTUAL_OWN_REQUEST_UNAVAILABLE"
                elif not history_available:
                    reasons[index] = "EARLIER_ACTIONABLE_OWN_REQUEST_UNAVAILABLE"
                else:
                    value = fold.observe()
                    require(value is not None, "actual requested actor has no actionable encoded request")
                    values[index], states[index] = value, fold.public_state()
                    program["events"].append(dict(kind="observe",coordinate=next_coordinate))
                    next_coordinate+=1;cutoffs[index]=len(program["events"])
                    output_bindings[index] = digest(_state_payload((values[index], states[index],
                        dict(program,events=program["events"][:cutoffs[index]]))))
            fold.check()
    require((source is None and expected_source_identity is None or source is not None
        and retained_program_identity(source) == expected_source_identity)
        and digest(replay_inputs) == expected_replay_sha256
        and digest(_state_payload(config)) == config_sha
        and all(digest(_state_payload((values[index], states[index],
            dict(program,events=program["events"][:cutoffs[index]])))) == binding
            for index, binding in output_bindings.items()),
        "original actor projection inputs or prior outputs changed during callbacks")
    return OriginalActorReplayProjection(tuple(values), tuple(states), dict(schema=SCHEMA,
        actor=replay_inputs["actor"], source_identity=expected_source_identity,
        replay_inputs_sha256=expected_replay_sha256, config_sha256=config_sha,
        coordinates=[row["snapshot_index"] for row in rows], unavailable_reasons=reasons,
        encoded_boundaries=sum(value is not None for value in values),
        unavailable_boundaries=sum(value is None for value in values),
        root_pp_differences=pp_changes, pp_equality_used_as_identity_gate=False,
        materialized_shell_prefix_used=False, original_source_p2_history_used=False,
        source_process_join_validated=False, constructor_process_join_validated=False,
        replay_process_join_validated=False, native_evaluation_input_parity=False,
        complete_paired_continuation_qualified=False, independent_terminal_labels=False,
        runtime_qualified=False, collection_authorized=False),program,tuple(cutoffs))


def read_owned_original_actor_replay(native_owner, materialization, replay, *, ledger, source_config,
        native_extract):
    """Original live source to exact recorded constructor/released replay.

    Owning readers and constructor-origin input checks are required separately
    from party anchors. This does not establish native model input parity or a
    complete paired raw continuation. No source/constructor/replay is launched.
    """
    from .search_over_raw_diagnostic_producer import OwnedFixedWorkDiagnostic
    from .search_over_raw_materialization_run import MaterializationRunValidator, _join_result
    from .search_over_raw_boundary_run import NativePathBoundaryRunValidator, raw_root_json, _join_raw_result
    from .search_over_raw_native_prefix import NativePrefixBoundaryRunValidator, strict_transport_view
    from .search_over_raw_consumed_materialization import (
        _original_diagnostic_owner_guard, join_owned_raw_replays)
    from .search_over_raw_archive import archived_exploration_root_binding
    require(type(native_owner) is OwnedFixedWorkDiagnostic and type(materialization) is dict
        and type(replay) is dict and type(materialization.get("validator")) is MaterializationRunValidator
        and type(replay.get("validator")) in (NativePathBoundaryRunValidator,NativePrefixBoundaryRunValidator) and callable(native_extract),
        "original native/source and distinct exact constructor/replay owning readers required")
    objects = (native_owner.source_owner, native_owner.job, native_owner.reader, native_owner.selected,
        materialization["validator"], replay["validator"], materialization["result"], replay["result"])
    owner_guard = _original_diagnostic_owner_guard(native_owner.selected, [materialization], [replay], native_owner.reader)
    config_sha = digest(_state_payload(source_config))
    native_owner.validate(ledger)
    readouts = {str(native_owner.reader.directory): deepcopy(native_owner.reader.last_validation)}
    source_owner = native_owner.source_owner
    root_id = native_owner.job.statistics_root_id
    crosswalk = archived_exploration_root_binding(source_owner.archive, source_owner.source,
        contract=ledger.contract, root_id=root_id)
    require(crosswalk["public_request_witness"]["public_request_sha256"] == native_owner.reader.public_request_sha256
        and digest(crosswalk["public_request_witness"]["consumed_binding"]) == digest(native_owner.reader.consumed),
        "original source root differs from consumed native public request")
    ctor, path = materialization["validator"], replay["validator"]
    # A capture-only reader validates its durable worker result. Also join the
    # supplied result wrapper to that exact expected/capture law; otherwise a
    # caller can substitute its root or runtime while retaining a good capture.
    _join_result(materialization["result"], ctor.expected, materialization["result"]["capture"])
    _join_raw_result(replay["result"], path.expected, replay["result"]["capture"])
    require(ctor(materialization["result"]["capture"]) == digest(materialization["result"]["capture"]),
        "independently clearing constructor ownership required")
    readouts[str(ctor.directory)] = deepcopy(ctor.last_validation)
    require(path(replay["result"]["capture"]) == digest(replay["result"]["capture"]),
        "independently clearing replay ownership required")
    readouts[str(path.directory)] = deepcopy(path.last_validation)
    owner_guard(readouts)
    inputs = ctor.expected["materialization_inputs"]
    origin = inputs.get("recorded_native_origin")
    require(type(origin) is dict and origin["native_selected_sha256"] == native_owner.selected_sha256
        and origin["original_public_request_sha256"] == native_owner.reader.public_request_sha256
        and origin["construction_ordinal"] == materialization["construction_ordinal"]
        and digest(inputs["root_binding"]) == digest(native_owner.reader.consumed),
        "exact recorded native constructor/source origin required")
    root = materialization["result"]["produced_bridge_snapshot"]
    replay_origin = path.expected["root_origin"]
    require(digest(root) == digest(path.raw_root) and replay_origin["produced_bridge_snapshot_sha256"] == digest(root)
        and replay_origin["materialization_capture_sha256"] == materialization["result"]["capture_sha256"]
        and replay_origin["construction_ordinal"] == materialization["construction_ordinal"] == replay["construction_ordinal"]
        and replay_origin["evaluation_ordinal"] == replay["evaluation_ordinal"]
        and replay_origin["native_selected_sha256"] == native_owner.selected_sha256
        and digest(path.expected["root_binding"]) == digest(native_owner.reader.consumed),
        "actual produced root and frozen native replay origin mismatch")
    # Reuse the full structural law, not just caller-declared origin hashes.
    # It joins actual draw/overlay/chance seed, selected evaluation capture,
    # full boundary roster, native projections, unique party map and runtime.
    # The extractor is still not an authenticated physics/input-parity oracle.
    joined = join_owned_raw_replays(native_owner.selected, [materialization], [replay],
        native_validator=native_owner.reader, ancestry_contract=native_owner.job.native_ancestry_contract,
        native_extract=native_extract)
    coordinate = (replay["construction_ordinal"], replay["evaluation_ordinal"])
    matched = [row for row in joined["records"] if
        (row["construction_ordinal"], row["evaluation_ordinal"]) == coordinate]
    require(len(matched) == 1 and matched[0]["native_action_driven_origin_join"] is True,
        "actual selected native constructor/evaluation path crosswalk required")
    owner_guard(readouts)
    source = source_owner.actor_program(root_id, config=source_config) if source_owner._actor_history is not None else None
    before = retained_program_identity(source) if source is not None else None
    records_path = path.directory / "bridge-transport.json"
    records_bytes = records_path.read_bytes()
    process_records = json.loads(records_bytes)["records"]
    # OwningRunValidator has already checked normal same-child close. Its
    # private capture was sealed before that handshake; mirror the validated
    # reader's exact boundary, not an arbitrary filtered/shortened trace.
    require(process_records[-2:] == [dict(kind="command", payload=dict(type="close")),
        dict(kind="event", payload=dict(type="closed"))], "actual replay close transport required")
    actor_capture=replay["result"]["capture"]
    if type(path) is NativePrefixBoundaryRunValidator:
        require(len(actor_capture["snapshots"])==len(path.expected["request_ids"]),
            "unreached original AFTER cannot substitute a reachable actor-history prefix")
        actor_capture=strict_transport_view(actor_capture)  # Original v2 owning reader already checked the full path.
    actor_inputs = read_actor_replay_inputs(actor_capture,
        process_records=process_records[:-2], initial_snapshot_json=raw_root_json(root),
        clone_binding=path.expected["clone_binding"], request_ids=path.expected["request_ids"], actor="p1", runtime=path.runtime)
    actor_sha = digest(actor_inputs)
    result = project_original_actor_replay(source, actor_inputs, config=source_config,
        expected_source_identity=before, expected_replay_sha256=actor_sha)
    result_before = projection_identity(result)
    native_owner.validate(ledger)
    require(digest(crosswalk) == digest(archived_exploration_root_binding(source_owner.archive, source_owner.source,
        contract=ledger.contract, root_id=root_id)), "original source crosswalk changed after actor projection")
    # Last archive callback has returned. Direct checks protect the original
    # objects/bytes/readouts and full materialization states, not a new baseline.
    native_owner._original_inputs()
    owner_guard(readouts)
    require(all(actual is original for actual, original in zip((native_owner.source_owner, native_owner.job,
        native_owner.reader, native_owner.selected, materialization["validator"], replay["validator"],
        materialization["result"], replay["result"]), objects))
        and records_path.read_bytes() == records_bytes and projection_identity(result) == result_before
        and digest(actor_inputs) == actor_sha and digest(_state_payload(source_config)) == config_sha
        and (source is None or retained_program_identity(source) == before),
        "original source/constructor/replay projection changed through final callbacks")
    result.receipt.update(source_process_join_validated=True, constructor_process_join_validated=True,
        replay_process_join_validated=True, source_root_id=root_id,
        synthetic_scope=any(reader.last_validation["synthetic_process_fixture"]
            for reader in (native_owner.reader, ctor, path)),
        actual_native_path_crosswalk_validated=True,
        construction_ordinal=replay["construction_ordinal"], evaluation_ordinal=replay["evaluation_ordinal"])
    return result
