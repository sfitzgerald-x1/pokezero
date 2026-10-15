"""Private AFTER raw/raw cell execution; no forced initial action or source.

This low-level worker stays inside the original auditor with private actor
programs. Its owning reader is separate from this low-level consistency law. Caller-provided
histories cannot establish sampled-opponent provenance. Its separate reader
checks transport/input/terminal consistency, not process or physics ownership.
"""
from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json
import math
import random
import time

from ..local_showdown import LocalShowdownEnv, showdown_seed_from_int
from .search_over_raw import digest, require, rng_seed, _raw_action
from .search_over_raw_actor_projection import ActorObservationFold, compile_actor_program
from .search_over_raw_boundary_transport import (
    BoundaryTransportCapture, _read_owned_boundary_transport, _requests, _binding,
)
from .search_over_raw_boundary_run import raw_root_json
from .search_over_raw_source import AuditedRawPolicy
from .search_over_raw_source_public_history import channel_lines, _filter_binding
from .search_over_raw_world_inputs import _state_payload
from .search_over_raw_outcome_run import verify_outcome_runtime

SCHEMA = "pokezero.search-over-raw.after-raw-cell.v1"
TRANSPORT_SCHEMA = "pokezero.search-over-raw.after-raw-prefix-transport.v1"
MAXIMUM_BOUNDARIES = 250


def after_step_seed(sample_seed, event_id, replicate, step):
    require(type(sample_seed) is int and 0 <= sample_seed < 2**64
        and type(event_id) is str and len(event_id) == 64 and all(c in "0123456789abcdef" for c in event_id)
        and type(replicate) is int and 0 <= replicate < 8
        and type(step) is int and 0 <= step < MAXIMUM_BOUNDARIES,
        "exact AFTER cell/step chance coordinates required")
    return rng_seed("native-after-label.v1", sample_seed, event_id, replicate, step)


def _validate_binding(b):
    require(type(b) is dict and set(b) == {"schema","raw_after_sha256","programs_sha256","config_sha256",
        "clone_binding","request_ids","sample_seed","event_id","replicate","subject","checkpoint_sha256",
        "policy","maximum_boundaries","runtime_sha256"} and b["schema"] == SCHEMA
        and b["policy"] == "raw_argmax_both_seats_from_after_step_zero"
        and type(b["maximum_boundaries"]) is int and b["maximum_boundaries"] == MAXIMUM_BOUNDARIES
        and type(b["subject"]) is str and b["subject"] in {"p1","p2"}
        and all(type(b[k]) is str and len(b[k]) == 64 and all(c in "0123456789abcdef" for c in b[k])
            for k in ("raw_after_sha256","programs_sha256","config_sha256","checkpoint_sha256","runtime_sha256"))
        and type(b["request_ids"]) is list and len(b["request_ids"]) == MAXIMUM_BOUNDARIES+1,
        "exact frozen AFTER raw/raw cell law required")
    _binding(b["clone_binding"])
    after_step_seed(b["sample_seed"],b["event_id"],b["replicate"],0)


class AfterRawTransportCapture(BoundaryTransportCapture):
    schema = TRANSPORT_SCHEMA
    maximum_snapshots = MAXIMUM_BOUNDARIES + 1

    def export(self):
        result = super().export()
        # This distinct law declares a maximum prefix, not a native path that
        # must reach every supplied coordinate. The reader proves its stop.
        result["request_ids"] = result["request_ids"][:len(result["snapshots"])]
        return result


def read_after_raw_transport(capture, *, process_records, raw_after, clone_binding, request_ids):
    require(type(request_ids) is list and len(request_ids) == MAXIMUM_BOUNDARIES+1,
        "full prospective AFTER maximum request roster required")
    count = len(capture["snapshots"])
    require(2 <= count <= len(request_ids) and capture["request_ids"] == request_ids[:count],
        "exact observed AFTER prefix required; no arbitrary shortening")
    trace = _read_owned_boundary_transport(capture, process_records=process_records,
        initial_snapshot_json=raw_root_json(raw_after), clone_binding=clone_binding,
        request_ids=request_ids[:count], schema=TRANSPORT_SCHEMA, maximum_snapshots=MAXIMUM_BOUNDARIES+1)
    require(trace["observed_snapshots"][-1]["boundary_kind"] == "terminal" or count == MAXIMUM_BOUNDARIES+1,
        "AFTER continuation stopped without terminal or fixed cap")
    return trace


def _fold(program, config):
    # Validate exact schema/order and execute every prior observation call.
    compiled = compile_actor_program(program, config=config, expected_program_sha256=digest(program),
        expected_config_sha256=digest(_state_payload(config)))
    fold = ActorObservationFold(config=config, actor=program["actor"], opening_request=program["opening_request"],
        expected_opening_sha256=digest(program["opening_request"]), battle_id=program.get("battle_id", "actor-projection"))
    last = None
    for event in program["events"]:
        if event["kind"] == "protocol": fold.feed_visible(event["lines"])
        elif event["kind"] == "own_request": fold.feed_own_request(event["request"])
        else: last = fold.observe()
    require(program["events"][-1]["kind"] == "observe"
        and digest(_state_payload(last)) == digest(_state_payload(compiled.observations[-1])),
        "AFTER history must end at its already encoded original observation")
    return fold, last


def _actors(programs, raw_after, config):
    require(type(programs) is dict and set(programs) == {"p1", "p2"}
        and all(type(programs[s]) is dict and programs[s].get("actor") == s for s in programs),
        "LABEL_INPUT_UNAVAILABLE: both separately frozen actor histories required before clone")
    folds, values = {}, {}
    for seat in ("p1", "p2"):
        folds[seat], values[seat] = _fold(programs[seat], config)
        require(digest(folds[seat].env._latest_requests.get(seat)) == digest(raw_after["boundaryRequests"].get(seat)),
            "actual AFTER own request differs from original actor history")
        require((values[seat] is not None) is (seat in _requests(raw_after)),
            "actual AFTER actor roster differs from encoded cursor")
    return folds, values


def _feed(folds, raw, previous):
    log = raw["battle"].get("log")
    channel_lines(log, channel=0)
    require(len(log) >= len(previous) and log[:len(previous)] == previous,
        "LABEL_INPUT_UNAVAILABLE: continuation rewrote prior protocol")
    requested = _requests(raw)
    for seat, fold in folds.items():
        fold.feed_visible(channel_lines(log[len(previous):], channel=int(seat[-1])))
        request = raw["boundaryRequests"].get(seat)
        require(seat not in requested or type(request) is dict and type(request.get("side")) is dict,
            "LABEL_INPUT_UNAVAILABLE: requested actor lacks actual full own request")
        if type(request) is dict and type(request.get("side")) is dict: fold.feed_own_request(request)
    return deepcopy(log)


def _choice(fold, seat, action):
    return fold.env._choices_for_actions({seat: action})[seat]


def _terminal(records, subject, initial_log, final_log):
    players, labels, choosing, choices, acked = {}, [], False, set(), set()
    terminal_line = lambda line: line.startswith("|win|") or line == "|tie" or line.startswith("|tie|")
    require(not any(terminal_line(line) for line in initial_log), "nonterminal AFTER root contains terminal protocol")
    for line in initial_log:
        if line.startswith("|player|"):
            fields = line.split("|"); players[fields[2]] = fields[3]
    for record in records:
        payload = record["payload"]
        if record["kind"] == "command" and payload.get("type") == "choices":
            choosing = True
            choices, acked = set(payload["choices"]), set()
        elif record["kind"] == "event" and payload.get("type") == "choice_ack":
            acked.add(payload["player"])
        elif record["kind"] == "event" and payload.get("type") in {"ready","terminal"}:
            require(choosing and (payload["type"] == "terminal" or not labels),
                "terminal protocol contradicts actual acknowledged transition")
            choosing = False
        if record["kind"] != "event" or payload.get("type") != "stream" or payload.get("stream") != "omniscient":
            continue
        for line in payload.get("lines", []):
            require(not line.startswith("|error|"), "AFTER simulator protocol error")
            require(not terminal_line(line) or choosing and acked == choices,
                "AFTER terminal protocol outside actual acknowledged transition")
            if line.startswith("|player|"):
                fields = line.split("|"); players[fields[2]] = fields[3]
            elif line.startswith("|win|"):
                winner = line.split("|", 2)[2]
                require(set(players) == {"p1", "p2"} and winner in players.values()
                    and len(set(players.values())) == 2, "unknown AFTER terminal winner")
                labels.append(1 if players[subject] == winner else -1)
            elif line == "|tie" or line.startswith("|tie|"): labels.append(0)
    require(len(labels) <= 1, "duplicate/conflicting AFTER terminal protocol")
    appended = final_log[len(initial_log):]
    recorded_lines = [line for r in records if r["kind"] == "event" and r["payload"].get("type") == "stream"
        and r["payload"].get("stream") == "omniscient" for line in r["payload"].get("lines", []) if terminal_line(line)]
    require([line for line in appended if terminal_line(line)] == recorded_lines,
        "actual final snapshot and transition terminal protocol disagree")
    return labels[0] if labels else None


@dataclass(frozen=True)
class AfterRawCellCapture:
    binding: dict
    transport: dict
    policy_calls: tuple

    def __reduce_ex__(self, protocol):
        raise TypeError("private AFTER actor/cell capture cannot enter durable worker transport")


def run_after_raw_cell(env, *, raw_after, programs, config, clone_binding, request_ids,
        sample_seed, event_id, replicate, subject, policies, checkpoint_sha256, runtime, process_owner=None):
    """One fresh clone in its original auditor; caller owns bridge/process reap.

    Private complete programs stay in process. This is not a job serialized to
    another worker and cannot infer source/history/clone ownership from hashes.
    Unknown histories refuse before bridge start. No retry or fallback exists.
    """
    _binding(clone_binding)
    if process_owner is not None:
        from .search_over_raw_after_owner import AfterRawProcessOwner
        require(type(process_owner) is AfterRawProcessOwner and env._bridge_process_observer is process_owner,
            "exact installed AFTER process owner required")
    after_step_seed(sample_seed, event_id, replicate, 0)
    require(type(subject) is str and subject in {"p1", "p2"} and type(env) is LocalShowdownEnv
        and env._process is None and env._battle_token is None and digest(_state_payload(env.config)) == digest(_state_payload(config))
        and getattr(env,"_after_raw_cell_consumed",False) is False
        and type(request_ids) is list and len(request_ids) == MAXIMUM_BOUNDARIES+1
        and type(policies) is dict and set(policies) == {"p1", "p2"},
        "empty original AFTER environment and fixed continuation bindings required")
    env._after_raw_cell_consumed = True
    runtime_sha = verify_outcome_runtime(runtime,config)
    require(runtime["files"][runtime["checkpoint_path"]] == checkpoint_sha256,
        "AFTER raw policy must retain original declared checkpoint bytes")
    original_filter = _filter_binding(runtime)
    initial = raw_root_json(raw_after)
    require(hashlib.sha256(initial.encode()).hexdigest() == clone_binding["snapshotJsonSha256"],
        "exact frozen AFTER clone bytes required")
    binding = dict(schema=SCHEMA, raw_after_sha256=digest(raw_after), programs_sha256=digest(programs),
        config_sha256=digest(_state_payload(config)), clone_binding=deepcopy(clone_binding), request_ids=list(request_ids),
        sample_seed=sample_seed, event_id=event_id, replicate=replicate, subject=subject,
        checkpoint_sha256=checkpoint_sha256, runtime_sha256=runtime_sha,
        policy="raw_argmax_both_seats_from_after_step_zero", maximum_boundaries=250)
    _validate_binding(binding)
    before = digest(binding)
    env_config = env.config
    folds, values = _actors(programs, raw_after, config)
    auditors = {s:AuditedRawPolicy(policies[s], checkpoint_sha256=checkpoint_sha256) for s in policies}
    objects = tuple(policies[s] for s in ("p1", "p2"))
    for auditor in auditors.values(): auditor.reset()
    tap = AfterRawTransportCapture(clone_binding=clone_binding, request_ids=request_ids)
    calls, originals, primary, captured = [], [], None, None
    def final_guard():
        require(verify_outcome_runtime(runtime,config) == runtime_sha and _filter_binding(runtime) == original_filter,
            "original AFTER runtime/filter bytes changed during execution")
        require(before == digest(dict(binding, raw_after_sha256=digest(raw_after), programs_sha256=digest(programs),
            config_sha256=digest(_state_payload(config)),clone_binding=clone_binding,request_ids=request_ids))
            and env.config is env_config and digest(_state_payload(env.config)) == binding["config_sha256"]
            and all(policies[s] is original for s,original in zip(("p1","p2"),objects))
            and all(digest(_state_payload(value)) == sha for value,sha in originals),
            "original AFTER cell inputs, policies or previous observations changed during execution")
        for seat in ("p1","p2"):
            AuditedRawPolicy(policies[seat],checkpoint_sha256=checkpoint_sha256)
    try:
        with tap.installed(env):
            env._start_bridge()
            env._battle_token = clone_binding["battleId"]
            env._bridge_request_event(dict(type="clone_snapshot", battleId=clone_binding["battleId"],
                cloneId=clone_binding["cloneId"], snapshotJson=initial), "cloned")
            current = tap.snapshot()
            require(digest(current) == digest(raw_after), "AFTER clone roundtrip changed original endpoint")
            previous = deepcopy(current["battle"]["log"])
            channel_lines(previous, channel=0)  # Validate, never import shell history into actors.
            for step in range(MAXIMUM_BOUNDARIES):
                if current["battle"]["ended"]: break
                choices = {}
                for seat in _requests(current):
                    observation = values[seat] if step == 0 else folds[seat].observe()
                    require(observation is not None, "requested AFTER actor encoding unavailable")
                    identity = digest(_state_payload(observation))
                    decision = auditors[seat].select_action(observation, rng=random.Random(0))
                    require(digest(_state_payload(observation)) == identity, "actual raw input mutated during inference")
                    originals.append((observation, identity))
                    action = decision.action_index
                    choice = _choice(folds[seat], seat, action)
                    calls.append(dict(step=step, seat=seat, observation_sha256=identity,
                        observation_object_id=id(observation), mask=list(observation.legal_action_mask),
                        distribution=list(decision.metadata["policy_distribution"]), action=action, choice=choice))
                    choices[seat] = choice
                seed = showdown_seed_from_int(after_step_seed(sample_seed, event_id, replicate, step))
                ack = env._bridge_request_event(dict(type="reseed", battleId=clone_binding["battleId"], seed=seed), "reseeded")
                require(ack.get("seed") == seed, "AFTER chance acknowledgement mismatch")
                env._send_command(dict(type="choices", battleId=clone_binding["battleId"], choices=choices))
                deadline = time.monotonic()+config.read_timeout_seconds
                while True:
                    require(time.monotonic() < deadline, "AFTER continuation boundary timed out")
                    event = env._read_event(timeout=max(.001, deadline-time.monotonic()))
                    if event is None: continue
                    require(event.get("battleId") == clone_binding["battleId"] and event.get("type") != "error",
                        "foreign or failed AFTER continuation event")
                    if event.get("type") in {"ready", "terminal"}: break
                current = tap.snapshot()
                previous = _feed(folds, current, previous)
            env._bridge_request_event(dict(type="release_clone", **clone_binding), "clone_released")
            env._battle_token = None
        final_guard()
        captured = AfterRawCellCapture(binding, tap.export(), tuple(calls))
        if process_owner is not None:
            require(env._bridge_process_observer is process_owner,"original AFTER process observer replaced")
            process_owner.seal(captured)
    except BaseException as error:
        primary = error
        raise
    finally:
        if env._process is not None:
            try: env.close()
            except BaseException as cleanup:
                if primary is None: raise
                primary.add_note("AFTER cell cleanup failed: "+str(cleanup))
    final_guard()
    return captured


def read_after_raw_cell(capture, *, process_records, raw_after, programs, config, expected_binding, runtime):
    """Independent reconstruction; consistency only, no owning-process claim."""
    require(type(capture) is AfterRawCellCapture and digest(capture.binding) == digest(expected_binding),
        "original independently frozen AFTER cell binding required")
    before = digest(dict(binding=capture.binding, transport=capture.transport, calls=capture.policy_calls,
        expected=expected_binding,records=process_records, root=raw_after, programs=programs, config=_state_payload(config), runtime=runtime))
    b = expected_binding
    _validate_binding(b)
    require(b["raw_after_sha256"] == digest(raw_after) and b["programs_sha256"] == digest(programs)
        and b["config_sha256"] == digest(_state_payload(config))
        and b["runtime_sha256"] == verify_outcome_runtime(runtime,config)
        and b["checkpoint_sha256"] == runtime["files"][runtime["checkpoint_path"]], "original AFTER inputs changed")
    original_filter = _filter_binding(runtime)
    trace = read_after_raw_transport(capture.transport, process_records=process_records, raw_after=raw_after,
        clone_binding=b["clone_binding"], request_ids=b["request_ids"])
    folds, values = _actors(programs, raw_after, config)
    previous, index = deepcopy(raw_after["battle"]["log"]), 0
    reseeds = [r["payload"]["seed"] for r in process_records if r["kind"] == "command" and r["payload"]["type"] == "reseed"]
    boundaries = len(trace["observed_snapshots"])-1
    require(reseeds == [showdown_seed_from_int(after_step_seed(b["sample_seed"], b["event_id"], b["replicate"], n))
        for n in range(boundaries)], "exact AFTER step chance streams required")
    for step, received in enumerate(capture.transport["snapshots"][:-1]):
        raw = json.loads(received["currentSnapshotJson"])
        choices = {}
        for seat in _requests(raw):
            require(index < len(capture.policy_calls), "missing actual AFTER policy call")
            call = capture.policy_calls[index]; index += 1
            observation = values[seat] if step == 0 else folds[seat].observe()
            require(call["step"] == step and type(call["step"]) is int and call["seat"] == seat
                and call["observation_sha256"] == digest(_state_payload(observation))
                and digest(call["mask"]) == digest(list(observation.legal_action_mask)), "actual AFTER raw input correspondence mismatch")
            legal = [i for i, allowed in enumerate(call["mask"]) if allowed]
            dist = call["distribution"]
            require(type(dist) is list and len(dist) == len(call["mask"])
                and all(type(p) in (int,float) and math.isfinite(p) and p >= 0 for p in dist)
                and math.isclose(math.fsum(dist),1,abs_tol=2e-5,rel_tol=0)
                and all(type(allowed) is bool and (allowed or dist[i] == 0) for i,allowed in enumerate(call["mask"])),
                "actual AFTER masked distribution mismatch")
            action = _raw_action(legal, [dist[i] for i in legal])
            require(type(call["action"]) is int and call["action"] == action and call["choice"] == _choice(folds[seat], seat, action),
                "AFTER action/translation differs from raw argmax")
            choices[seat] = call["choice"]
        require(choices == trace["observed_snapshots"][step+1]["incoming_choices"], "actual AFTER command differs from policy")
        previous = _feed(folds, json.loads(capture.transport["snapshots"][step+1]["currentSnapshotJson"]), previous)
    require(index == len(capture.policy_calls), "extra AFTER policy calls cannot supply labels")
    terminal = trace["observed_snapshots"][-1]["boundary_kind"] == "terminal"
    final_log = json.loads(capture.transport["snapshots"][-1]["currentSnapshotJson"])["battle"]["log"]
    outcome = _terminal(process_records, b["subject"], raw_after["battle"]["log"], final_log)
    require(terminal or outcome is None, "terminal protocol followed by capped/nonterminal continuation")
    require(verify_outcome_runtime(runtime,config) == b["runtime_sha256"] and _filter_binding(runtime) == original_filter,
        "original AFTER runtime/filter bytes changed during reconstruction")
    _validate_binding(expected_binding)
    require(digest(expected_binding) == digest(capture.binding)
        and before == digest(dict(binding=capture.binding, transport=capture.transport, calls=capture.policy_calls,
        expected=expected_binding,records=process_records, root=raw_after, programs=programs, config=_state_payload(config), runtime=runtime)),
        "original AFTER inputs/capture changed during independent reconstruction")
    return dict(schema=SCHEMA, status="COMPLETE" if terminal and outcome is not None else
        "TERMINAL_UNVALIDATED" if terminal else "CAPPED_UNCERTAIN", signed_outcome=outcome,
        boundaries=boundaries, policy_input_correspondence_validated=True, forced_initial_action=False,
        sampled_initial_opponent=False, process_ownership_validated=False, actor_history_provenance_validated=False,
        independent_terminal_labels=False, native_input_parity=False, runtime_qualified=False, scientific_collection_authorized=False)
