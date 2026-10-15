"""Private prospective source-subject stream and actual raw-input chronology.

The original source owner supplies transport records. Retained seat payloads
never enter durable receipts or a sampled opponent's history. This compares a
seat-visible reconstruction with actual historical-policy inputs; it does not
change the policy to make that comparison pass. A mismatch stays unavailable.
"""
from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

from .search_over_raw import digest, require
from .search_over_raw_actor_projection import (
    CONTEXT_SCHEMA as PROGRAM_SCHEMA, ENCODING_LAW, MAXIMUM_BYTES,
    compile_actor_program, projection_source_hashes)
from .search_over_raw_world_inputs import _state_payload

SCHEMA = "pokezero.search-over-raw.source-actor-history.v1"


def _code():
    paths = [Path(__file__).resolve(), *(Path(__file__).with_name(name) for name in (
        "search_over_raw.py", "search_over_raw_source.py", "search_over_raw_source_run.py",
        "search_over_raw_world_inputs.py"))]
    return {**projection_source_hashes(), **{str(path): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in paths}}


def _observation_sha(observation):
    # Full actual object, including masks, schema, vocabulary and metadata.
    # Do not erase fields to excuse a mismatch with the historical encoding.
    return digest(_state_payload(observation))


class SourceActorHistoryCapture:
    """In-memory observer only, enabled prospectively for original source p1."""
    def __init__(self, config):
        self.config_sha = digest(_state_payload(config))
        self.sources = _code()
        self._owner = self._env = self._records = None
        self._original_refs = None
        self._events, self._observations, self._policy_inputs = [], [], []
        self._bytes = 0
        self._sealed = None

    def __reduce_ex__(self, protocol):
        raise TypeError("private source actor history cannot be serialized")

    def bind(self, owner, env, records):
        require(self._owner is None and type(records) is list
            and owner.env is env and owner.records is records
            and digest(_state_payload(env.config)) == self.config_sha,
            "one original source owner/environment/transport required")
        self._owner, self._env, self._records = owner, env, records
        self._original_refs = (owner, env, records)

    def _check(self, *, writing=False):
        require(self._owner is not None and self._owner.env is self._env
            and self._original_refs is not None and all(actual is original for actual, original in zip(
                (self._owner, self._env, self._records), self._original_refs))
            and self._owner.records is self._records
            and self._owner.actor_history is self
            and digest(_state_payload(self._env.config)) == self.config_sha
            and self.sources == _code() and (not writing or self._sealed is None),
            "original source actor capture changed or is sealed")

    def _retain(self, value):
        self._bytes += len(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode())
        require(self._bytes <= MAXIMUM_BYTES, "source actor private retention cap exceeded")

    def event(self, payload, record_index):
        if payload.get("type") != "stream" or payload.get("stream") != "p1":
            return
        self._check(writing=True)
        from .search_over_raw_source_run import _source_trace
        require(type(record_index) is int and 0 <= record_index < len(self._records)
            and self._records[record_index] == _source_trace("event", payload),
            "actual received source actor payload differs from transport")
        retained = dict(record_index=record_index, payload=deepcopy(dict(payload)))
        self._retain(retained)
        self._events.append(retained)

    def observation(self, env, actor, observation):
        self._check(writing=True)
        require(env is self._env, "source actor observation came from another environment")
        if actor != "p1":
            return
        ordinal, index = len(self._observations), len(self._records)
        sha = _observation_sha(observation)
        battle_id = env._parser.snapshot().battle_id
        require(type(battle_id) is str and observation.metadata.get("battle_id") == battle_id,
            "actual source observation origin differs from original parser")
        row = dict(kind="actor_observation", type="source_actor_observation", actor=actor,
            record_index=index, transport_cutoff=index, observation_ordinal=ordinal,
            observation_sha256=sha,battle_id=battle_id)
        self._retain(dict(receipt=row, observation=_state_payload(observation)))
        # Retain actual object identity as well as bytes. A newly constructed
        # equal observation cannot pretend to be the one returned by _observe.
        self._observations.append((deepcopy(row), observation))
        self._records.append(row)

    def policy_input(self, observation):
        self._check(writing=True)
        matches = [(row, actual) for row, actual in self._observations if actual is observation]
        require(matches, "actual raw input lacks its original source encoding call")
        observed, _ = matches[-1]
        sha = _observation_sha(observation)
        require(sha == observed["observation_sha256"], "raw policy input changed after source encoding")
        index = len(self._records)
        row = dict(kind="actor_policy_input", type="source_actor_policy_input", actor="p1",
            record_index=index, transport_cutoff=index,
            observation_ordinal=observed["observation_ordinal"], observation_sha256=sha)
        self._retain(row)
        self._policy_inputs.append(deepcopy(row))
        self._records.append(row)
        return deepcopy(row)

    def check_policy_input(self, token, observation):
        self._check(writing=True)
        require(token in self._policy_inputs and self._observations[token["observation_ordinal"]][1] is observation
            and _observation_sha(observation) == token["observation_sha256"],
            "actual raw policy mutated/substituted its source input")

    def identity(self):
        self._check()
        return digest(dict(config=self.config_sha, code=self.sources,
            owner_id=id(self._owner), env_id=id(self._env), records_id=id(self._records), events=self._events,
            observations=[row for row, _ in self._observations], policies=self._policy_inputs,
            bytes=self._bytes))

    def seal(self):
        self._check(writing=True)
        self._sealed = self.identity()

    def verify(self, records):
        self._check()
        require(self._sealed is not None and self.identity() == self._sealed
            and digest(records) == digest(self._records), "original sealed source actor history/transport changed")
        from .search_over_raw_source_run import _source_trace
        expected = [i for i, row in enumerate(records) if row.get("kind") == "event"
            and row.get("type") == "stream" and row.get("stream") == "p1"]
        require([row["record_index"] for row in self._events] == expected,
            "source actor stream roster incomplete")
        for row in self._events:
            require(records[row["record_index"]] == _source_trace("event", row["payload"]),
                "retained source actor stream differs from received transport")
        for kind, rows in (("actor_observation", [row for row, _ in self._observations]),
                ("actor_policy_input", self._policy_inputs)):
            require([row["record_index"] for row in rows] == [i for i, r in enumerate(records)
                if r.get("kind") == kind], "source actor call roster incomplete")
            for row in rows:
                require(records[row["record_index"]] == row
                    and row["record_index"] == row["transport_cutoff"], "source actor chronology drift")
        require(all(_observation_sha(value) == row["observation_sha256"]
            for row, value in self._observations), "original returned source observation changed")
        return self._sealed


@dataclass(frozen=True)
class RetainedSourceActorProgram:
    # Private, in-process input for a later original-subject continuation join.
    program: dict
    compiled: object
    receipt: dict

    def __reduce_ex__(self, protocol):
        raise TypeError("source actor program cannot enter worker/durable transport")


def retained_program_identity(result):
    """Callback-free complete guard, including mutable materialization state."""
    require(type(result) is RetainedSourceActorProgram, "exact retained source actor program required")
    return digest(dict(program=result.program, observations=_state_payload(result.compiled.observations),
        states=_state_payload(result.compiled.materialization_states), compiler_receipt=result.compiled.receipt,
        receipt=result.receipt))


def read_source_actor_program(capture, *, records, cutoff_record_index, config):
    """Read original p1 only; caller independently binds the selected boundary.

    The returned receipt proves transport/encoding correspondence only when
    every actual encoding agrees. It never certifies owning-process lifecycle,
    source-to-construction correspondence, independent labels or runtime.
    """
    require(type(capture) is SourceActorHistoryCapture and type(records) is list
        and type(cutoff_record_index) is int and 0 <= cutoff_record_index < len(records)
        and digest(_state_payload(config)) == capture.config_sha,
        "original source actor capture/config/cutoff required")
    frozen, records_sha, config_sha = capture.verify(records), digest(records), capture.config_sha
    cutoff = records[cutoff_record_index]
    require(cutoff.get("kind") == "event" and cutoff.get("type") == "snapshot_actionable_boundary"
        and cutoff.get("available") is True and "p1" in cutoff.get("requested", []),
        "original p1 pre-step boundary cutoff required")
    starts = [i for i, row in enumerate(records[:cutoff_record_index])
        if row.get("kind") == "event" and row.get("type") == "started"]
    require(len(starts) == 1, "LABEL_INPUT_UNAVAILABLE: source start chronology missing")
    timeline = [(row["record_index"], "stream", row["payload"]) for row in capture._events]
    timeline += [(row["record_index"], "observe", row) for row, _ in capture._observations]
    events, opening, visible, expected = [], None, [], []
    for index, kind, payload in sorted(timeline):
        if index > cutoff_record_index:
            break
        require(index > starts[0], "actor history precedes original source start")
        if kind == "observe":
            require(opening is not None and any(line in {"|start", "|start|"} for line in visible)
                and all(any(line.startswith("|player|" + seat + "|") for line in visible)
                    for seat in ("p1", "p2")), "LABEL_INPUT_UNAVAILABLE: reset-to-actor-observation history incomplete")
            events.append(dict(kind="observe", coordinate=payload["observation_ordinal"]))
            expected.append(payload["observation_sha256"])
            continue
        lines = payload.get("lines")
        require(type(lines) is list and all(type(line) is str and not any(c in line for c in (
                "\n", "\r", "\x00", "\u2028", "\u2029")) and not line.startswith("|split") for line in lines),
            "LABEL_INPUT_UNAVAILABLE: source seat stream malformed/unfiltered")
        for line in lines:
            if line.startswith("|request|"):
                request = json.loads(line[len("|request|"):])
                if opening is None:
                    require(not any(line.startswith("|turn|") and line not in {"|turn|1", "|turn|0"}
                        for line in visible), "LABEL_INPUT_UNAVAILABLE: first retained request is mid-battle")
                    opening = deepcopy(request)
                events.append(dict(kind="own_request", request=request))
            else:
                require(not line.startswith("|request"), "malformed source request protocol")
                visible.append(line)
                events.append(dict(kind="protocol", lines=[line]))
    require(opening is not None and expected, "LABEL_INPUT_UNAVAILABLE: source own opening/observation history missing")
    observed = [row for row, _ in capture._observations if row["record_index"] <= cutoff_record_index]
    require(observed and type(observed[0].get("battle_id")) is str
        and all(row.get("battle_id") == observed[0]["battle_id"] for row in observed),
        "original source observation origin changed or is unavailable")
    program = dict(schema=PROGRAM_SCHEMA, actor="p1", encoding_law=ENCODING_LAW,
        battle_id=observed[0]["battle_id"],
        opening_request=opening, events=events)
    compiled = compile_actor_program(program, config=config, expected_program_sha256=digest(program),
        expected_config_sha256=config_sha)
    actual = [_observation_sha(value) if value is not None else None for value in compiled.observations]
    mismatched = [coordinate for coordinate, old, new in zip(compiled.receipt["coordinates"], expected, actual) if old != new]
    policies = [row for row in capture._policy_inputs if row["record_index"] <= cutoff_record_index]
    require(policies, "LABEL_INPUT_UNAVAILABLE: actual source raw inputs missing")
    for row in policies:
        require(row["observation_ordinal"] in compiled.receipt["coordinates"]
            and row["observation_sha256"] == expected[compiled.receipt["coordinates"].index(row["observation_ordinal"])],
            "source raw input lacks its earlier actual encoding")
    require(capture.verify(records) == frozen and digest(records) == records_sha
        and digest(_state_payload(config)) == config_sha, "source actor inputs changed during encoder callbacks")
    receipt = dict(schema=SCHEMA, actor="p1", capture_sha256=frozen,
        source_transport_sha256=records_sha, cutoff_record_index=cutoff_record_index,
        source_snapshot_sha256=cutoff["snapshot_sha256"], program_sha256=digest(program),
        config_sha256=config_sha, source_hashes=deepcopy(capture.sources),
        observation_coordinates=compiled.receipt["coordinates"], raw_input_count=len(policies),
        mismatched_observation_coordinates=mismatched,
        status="ENCODING_MISMATCH" if mismatched else "INPUT_CORRESPONDENCE_MATCHED",
        actual_raw_input_binding_validated=True, complete_actor_history_validated=not mismatched,
        historical_encoding_unchanged=True, original_source_p2_history_used=False,
        process_ownership_validated=False, source_prefix_join_validated=False,
        construction_provenance_validated=False, independent_terminal_labels=False, runtime_qualified=False)
    return RetainedSourceActorProgram(program, compiled, receipt)
