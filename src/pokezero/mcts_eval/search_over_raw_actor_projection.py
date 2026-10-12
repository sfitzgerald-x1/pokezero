"""Process-free actor history folding through the existing checkpoint encoder.

This consumes independently bound, complete actor-visible history and actual
own requests. It cannot establish their source/construction provenance. It
never resets a battle, invents a paired Python snapshot, or evaluates a model.
Missing own opening knowledge/history must remain unavailable upstream; a
spectator prefix plus latest request is not a substitute for this program.
"""
from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass, fields, is_dataclass
from enum import Enum
import hashlib
import json
from pathlib import Path, PurePath

from ..local_showdown import LocalShowdownConfig, LocalShowdownEnv
from ..showdown import _ReplayParser


def digest(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
        allow_nan=False).encode()).hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


STATE_OBJECT_TYPES = frozenset({"pokezero.belief.PublicBattleBeliefEngine",
    "pokezero.tier2.Tier2LiveTracker", "pokezero.tier2._IncrementalContextFold",
    "pokezero.investment.InvestmentLiveTracker"})


def _state_payload(value, active=None):
    """Deterministic full state, never repr/pickle or a partial public projection.

    All fields of known mutable state objects are retained, including caches.
    This can conservatively report a mismatch for non-causal cache differences;
    it cannot certify equality by silently deleting one. Unsupported objects or
    cycles refuse the witness. Hashing the full catalogs/dex costs real time.
    """
    active = set() if active is None else active
    kind = type(value)
    name = kind.__module__ + "." + kind.__qualname__
    if isinstance(value, Enum):
        return dict(enum=name, value=_state_payload(value.value, active))
    if value is None or kind in (str, int, float, bool):
        return value
    if isinstance(value, PurePath):
        return dict(path=str(value))
    require(id(value) not in active, "cyclic consumed input state")
    active.add(id(value))
    try:
        if isinstance(value, Mapping):
            entries = [[_state_payload(key, active), _state_payload(item, active)]
                for key, item in value.items()]
            entries.sort(key=lambda pair: json.dumps(pair[0], sort_keys=True, allow_nan=False))
            return dict(mapping=entries)
        if kind in (list, tuple, set, frozenset):
            items = [_state_payload(item, active) for item in value]
            if kind in (set, frozenset):
                items.sort(key=lambda item: json.dumps(item, sort_keys=True, allow_nan=False))
            return dict(container=kind.__name__, items=items)
        if is_dataclass(value) and not isinstance(value, type):
            return dict(dataclass=name, fields={field.name: _state_payload(
                getattr(value, field.name), active) for field in fields(value)})
        require(name in STATE_OBJECT_TYPES and hasattr(value, "__dict__")
            and not any("__slots__" in cls.__dict__ for cls in kind.__mro__),
            "unsupported consumed input state type")
        return dict(object=name, fields=_state_payload(vars(value), active))
    finally:
        active.remove(id(value))


SCHEMA = "pokezero.search-over-raw.actor-projection-program.v1"
MAXIMUM_BYTES = 8 * 1024 * 1024
ENCODING_LAW = "seat_visible_protocol_and_actual_own_requests"


def projection_source_hashes():
    directory = Path(__file__).resolve().parents[1]
    paths = [Path(__file__).resolve(), *(directory/name for name in (
        "local_showdown.py", "showdown.py", "belief.py", "tier2.py", "investment.py",
        "observation.py", "dex.py", "randbat.py", "randbat_vocab.py"))]
    return {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}


class _ProjectionEnv(LocalShowdownEnv):
    """Only normal parser, belief, annotation and observation methods are used."""
    def _no_bridge(self, *args, **kwargs):
        raise ValueError("actor projection cannot launch/reset/restore/step a simulator")
    _start_bridge = _send_command = _read_event = reset = reset_with_start_override = _no_bridge
    step = snapshot = snapshot_actionable_boundary = restore = observe_search_snapshot = _no_bridge


def _own_request(request, actor, *, opening=False):
    require(type(request) is dict and type(request.get("side")) is dict
        and request["side"].get("id") == actor and type(request["side"].get("pokemon")) is list
        and 1 <= len(request["side"]["pokemon"]) <= 6
        and not any(key.startswith("_pokezero") for key in request),
        "LABEL_INPUT_UNAVAILABLE: actual own request/team required without internal cursor aliases")
    if opening:
        require(not request.get("wait") and not request.get("teamPreview") and not request.get("forceSwitch")
            and type(request.get("active")) is list and len(request["active"]) == 1
            and all(type(mon) is dict and type(mon.get("moves")) is list and mon["moves"]
                and all(type(move) is str and move for move in mon["moves"])
                for mon in request["side"]["pokemon"]),
            "LABEL_INPUT_UNAVAILABLE: genuine battle-start own moveset anchor required")
    return deepcopy(request)


class ActorObservationFold:
    """Reusable pure history fold for a separately owned continuation controller.

    Opening data must be an actual observed own start request, not a mid-battle
    fallback. Its hash alone is not that provenance proof. This interface does
    not claim complete history; the owning program reader supplies chronology.
    """
    def __init__(self, *, config, actor, opening_request, expected_opening_sha256):
        require(type(config) is LocalShowdownConfig and actor in {"p1", "p2"}
            and type(actor) is str and digest(opening_request) == expected_opening_sha256,
            "exact actor/config/externally frozen own opening request required")
        self.actor, self.config_sha = actor, digest(_state_payload(config))
        self.opening_sha = expected_opening_sha256
        self.sources = projection_source_hashes()
        opening = _own_request(opening_request, actor, opening=True)
        self.env = _ProjectionEnv(deepcopy(config))
        other = "p2" if actor == "p1" else "p1"
        self.env._parser = _ReplayParser("actor-projection", complete_prefix=True,
            hp_visibility={actor: "exact", other: "percentage"})
        # First knowledge is not guessed from requests[0] at a mid-battle leaf.
        self.env._first_requests = {actor: opening}
        self._encoded_bytes = 0
        self.check()

    def check(self):
        require(type(self.env) is _ProjectionEnv and self.env._process is None
            and self.env._battle_token is None and self.env._stdout_thread is None
            and self.env._stderr_thread is None and digest(_state_payload(self.env.config)) == self.config_sha
            and set(self.env._latest_requests) <= {self.actor}
            and set(self.env._first_requests) == {self.actor}
            and digest(self.env._first_requests[self.actor]) == self.opening_sha
            and not self.env._request_history["p2" if self.actor == "p1" else "p1"]
            and set(self.env._parser.snapshot().requests) <= {self.actor}
            and projection_source_hashes() == self.sources,
            "original pure actor projection inputs/state changed")

    def feed_visible(self, lines):
        self.check()
        require(type(lines) is list and all(type(line) is str and not any(c in line for c in (
                "\n", "\r", "\x00", "\u2028", "\u2029"))
            and not line.startswith(("|request", "|split")) for line in lines),
            "only already filtered actor-visible protocol may enter projection")
        self._encoded_bytes += sum(len(line.encode()) + 1 for line in lines)
        require(self._encoded_bytes <= MAXIMUM_BYTES, "actor history retention cap exceeded")
        # The ordinary env's protocol fold accepts lines on its public channel;
        # the independently checked seat-filtered content is supplied here.
        self.env._apply_event(dict(type="stream", stream="omniscient", lines=deepcopy(lines)))
        self.env._sync_incremental_state()
        self.check()

    def feed_own_request(self, request):
        self.check()
        retained = _own_request(request, self.actor)
        text = "|request|" + json.dumps(retained, sort_keys=True, separators=(",", ":"), allow_nan=False)
        self._encoded_bytes += len(text.encode()) + 1
        require(self._encoded_bytes <= MAXIMUM_BYTES, "actor history retention cap exceeded")
        # This preserves the existing public-event cursor, PP history,
        # initial-anchor and recharge/Struggle handling instead of copying them.
        self.env._apply_event(dict(type="stream", stream=self.actor, lines=[text]))
        self.env._sync_incremental_state()
        self.check()

    def observe(self):
        self.check()
        if self.env._terminal is not None or self.actor not in self.env.requested_players():
            return None
        value = LocalShowdownEnv._observe(self.env, self.actor)
        self.check()
        return value

    def public_state(self):
        self.check()
        require(self.env._terminal is None and self.actor in self.env.requested_players(),
            "terminal/waiting actor has no actionable materialization state")
        value = LocalShowdownEnv.public_materialization_state(self.env, self.actor)
        self.check()
        return value


@dataclass(frozen=True)
class CompiledActorProgram:
    observations: tuple
    materialization_states: tuple
    receipt: dict


def compile_actor_program(program, *, config, expected_program_sha256, expected_config_sha256):
    """Execute exact request/annotation chronology without loading a model.

    The independent owning reader must supply the complete program and hashes.
    No caller declaration here promotes source visibility or history coverage.
    Waiting and terminal observation coordinates are returned as None.
    """
    require(type(program) is dict and set(program) == {"schema", "actor", "encoding_law", "opening_request", "events"}
        and program["schema"] == SCHEMA and program["encoding_law"] == ENCODING_LAW
        and type(program["events"]) is list and program["events"]
        and digest(program) == expected_program_sha256
        and digest(_state_payload(config)) == expected_config_sha256,
        "externally bound complete actor projection program/config required")
    source_hashes = projection_source_hashes()
    frozen = deepcopy(program)
    fold = ActorObservationFold(config=config, actor=frozen["actor"], opening_request=frozen["opening_request"],
        expected_opening_sha256=digest(frozen["opening_request"]))
    observations, states, coordinates = [], [], []
    for row in frozen["events"]:
        require(type(row) is dict and row.get("kind") in {"protocol", "own_request", "observe"},
            "exact actor history event kind required")
        if row["kind"] == "protocol":
            require(set(row) == {"kind", "lines"}, "actor protocol event schema drift")
            fold.feed_visible(row["lines"])
        elif row["kind"] == "own_request":
            require(set(row) == {"kind", "request"}, "actor request event schema drift")
            fold.feed_own_request(row["request"])
        else:
            require(set(row) == {"kind", "coordinate"} and type(row["coordinate"]) is int
                and row["coordinate"] >= 0 and (not coordinates or row["coordinate"] > coordinates[-1]),
                "original strictly ordered actor observation coordinates required")
            value = fold.observe()
            observations.append(value)
            states.append(fold.public_state() if value is not None else None)
            coordinates.append(row["coordinate"])
    require(coordinates and digest(program) == expected_program_sha256
        and digest(_state_payload(config)) == expected_config_sha256 and projection_source_hashes() == source_hashes,
        "actor projection inputs changed during actual encoder callbacks")
    fold.check()
    return CompiledActorProgram(tuple(observations), tuple(states), dict(schema=SCHEMA,
        program_sha256=expected_program_sha256, config_sha256=expected_config_sha256,
        source_hashes=source_hashes, encoding_law=ENCODING_LAW, coordinates=coordinates,
        observed_requests=len(fold.env._request_history[frozen["actor"]]),
        encoded_observations=sum(value is not None for value in observations),
        unavailable_observations=sum(value is None for value in observations),
        bridge_process_started=False, model_loaded=False, fabricated_snapshot_used=False,
        source_prefix_join_validated=False, construction_provenance_validated=False,
        complete_actor_history_validated=False, independent_terminal_labels=False, runtime_qualified=False))
