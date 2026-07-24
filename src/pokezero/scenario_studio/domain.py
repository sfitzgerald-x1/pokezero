"""Canonical, portable endgame-scenario data model.

The browser deliberately edits these typed values instead of a serialized Pokemon Showdown
battle. A scenario describes the allowed authoring surface only; the local bridge owns the
simulator snapshot and applies the separately validated battle-boundary patch.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from typing import Any, Mapping, Sequence

from ..dex import normalize_id
from ..showdown_fixture import FixturePokemon


ENDGAME_SCENARIO_SCHEMA_VERSION = "endgame-scenario-v1"
_SIDES = ("p1", "p2")
_CONSTRUCTION_MODES = frozenset({"generated", "source-composed"})
_OBJECTIVE_KINDS = frozenset({"forced_win", "best_move", "survival", "custom"})
_VERIFICATION_STATUSES = frozenset({"unverified", "manual", "engine_exhaustive", "probabilistic"})
SCENARIO_STATUS_IDS = ("", "brn", "par", "psn", "tox", "slp", "frz")
SCENARIO_WEATHER_IDS = ("", "raindance", "sunnyday", "sandstorm", "hail")
SCENARIO_SIDE_CONDITION_LIMITS = {
    "spikes": 3,
    "reflect": 5,
    "lightscreen": 5,
    "safeguard": 5,
    "mist": 5,
}
SCENARIO_VOLATILE_IDS = (
    "confusion",
    "substitute",
    "leechseed",
    "taunt",
    "encore",
    "disable",
    "torment",
    "attract",
    "nightmare",
    "curse",
    "ingrain",
    "focusenergy",
    "yawn",
    "perishsong",
    "flashfire",
    "mudsport",
    "watersport",
)
_VOLATILE_TURN_LIMITS = {
    "confusion": (1, 4),
    "taunt": (1, 2),
    "encore": (1, 6),
    "disable": (1, 5),
    "yawn": (1, 1),
    "perishsong": (1, 3),
}
_VOLATILE_MOVE_IDS = frozenset({"encore", "disable"})


class ScenarioValidationError(ValueError):
    """A user-correctable scenario error with a JSON-pointer-style location."""

    def __init__(self, message: str, *, path: str = "") -> None:
        self.path = path
        super().__init__(f"{path}: {message}" if path else message)


def _mapping(value: Any, *, path: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ScenarioValidationError("must be an object", path=path)
    return value


def _list(value: Any, *, path: str) -> Sequence[Any]:
    if not isinstance(value, list):
        raise ScenarioValidationError("must be an array", path=path)
    return value


def _string(value: Any, *, path: str, allow_empty: bool = False) -> str:
    if not isinstance(value, str) or (not allow_empty and not value.strip()):
        raise ScenarioValidationError("must be a non-empty string", path=path)
    return value.strip()


def _integer(value: Any, *, path: str, minimum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ScenarioValidationError("must be an integer", path=path)
    if minimum is not None and value < minimum:
        raise ScenarioValidationError(f"must be at least {minimum}", path=path)
    return value


def _optional_integer(
    value: Any,
    *,
    path: str,
    minimum: int | None = None,
    maximum: int | None = None,
) -> int | None:
    if value is None:
        return None
    result = _integer(value, path=path, minimum=minimum)
    if maximum is not None and result > maximum:
        raise ScenarioValidationError(f"must be at most {maximum}", path=path)
    return result


def _string_mapping(value: Any, *, path: str) -> dict[str, int]:
    if value is None:
        return {}
    mapping = _mapping(value, path=path)
    result: dict[str, int] = {}
    for key, item in mapping.items():
        result[_string(key, path=path)] = _integer(item, path=f"{path}/{key}", minimum=0)
    return result


def _condition_mapping(value: Any, *, path: str) -> dict[str, int]:
    result = _string_mapping(value, path=path)
    for raw_name, count in result.items():
        name = normalize_id(raw_name)
        maximum = SCENARIO_SIDE_CONDITION_LIMITS.get(name)
        if maximum is None:
            raise ScenarioValidationError("has an unsupported Gen 3 side condition", path=f"{path}/{raw_name}")
        if count > maximum:
            raise ScenarioValidationError(f"must be at most {maximum}", path=f"{path}/{raw_name}")
        if count == 0:
            raise ScenarioValidationError("must be omitted instead of set to zero", path=f"{path}/{raw_name}")
    return {normalize_id(name): count for name, count in result.items()}


@dataclass(frozen=True)
class ScenarioMove:
    move_id: str
    pp: int
    max_pp: int

    @classmethod
    def from_payload(cls, payload: Any, *, path: str) -> "ScenarioMove":
        value = _mapping(payload, path=path)
        return cls(
            move_id=normalize_id(_string(value.get("id"), path=f"{path}/id")),
            pp=_integer(value.get("pp"), path=f"{path}/pp", minimum=0),
            max_pp=_integer(value.get("max_pp"), path=f"{path}/max_pp", minimum=1),
        )

    def to_payload(self) -> dict[str, Any]:
        return {"id": self.move_id, "pp": self.pp, "max_pp": self.max_pp}


@dataclass(frozen=True)
class ScenarioStatus:
    status_id: str = ""
    sleep_turns_remaining: int | None = None
    toxic_stage: int | None = None

    @classmethod
    def from_payload(cls, payload: Any, *, path: str) -> "ScenarioStatus":
        if payload is None:
            return cls()
        value = _mapping(payload, path=path)
        status_id = normalize_id(_string(value.get("id", ""), path=f"{path}/id", allow_empty=True))
        if status_id not in SCENARIO_STATUS_IDS:
            raise ScenarioValidationError("has an unsupported Gen 3 status", path=f"{path}/id")
        sleep_turns = _optional_integer(
            value.get("sleep_turns_remaining"),
            path=f"{path}/sleep_turns_remaining",
            minimum=1,
            maximum=4,
        )
        toxic_stage = _optional_integer(
            value.get("toxic_stage"),
            path=f"{path}/toxic_stage",
            minimum=0,
            maximum=15,
        )
        if status_id == "slp":
            if sleep_turns is None:
                raise ScenarioValidationError(
                    "is required for sleep", path=f"{path}/sleep_turns_remaining"
                )
        elif sleep_turns is not None:
            raise ScenarioValidationError(
                "is only valid for sleep", path=f"{path}/sleep_turns_remaining"
            )
        if status_id == "tox":
            if toxic_stage is None:
                raise ScenarioValidationError("is required for toxic", path=f"{path}/toxic_stage")
        elif toxic_stage is not None:
            raise ScenarioValidationError("is only valid for toxic", path=f"{path}/toxic_stage")
        return cls(
            status_id=status_id,
            sleep_turns_remaining=sleep_turns,
            toxic_stage=toxic_stage,
        )

    def to_payload(self) -> dict[str, Any]:
        payload: dict[str, Any] = {"id": self.status_id}
        if self.sleep_turns_remaining is not None:
            payload["sleep_turns_remaining"] = self.sleep_turns_remaining
        if self.toxic_stage is not None:
            payload["toxic_stage"] = self.toxic_stage
        return payload


@dataclass(frozen=True)
class ScenarioVolatile:
    volatile_id: str
    turns_remaining: int | None = None
    turns_elapsed: int | None = None
    hp: int | None = None
    move_id: str | None = None

    @classmethod
    def from_payload(cls, payload: Any, *, path: str) -> "ScenarioVolatile":
        value = _mapping(payload, path=path)
        volatile_id = normalize_id(_string(value.get("id"), path=f"{path}/id"))
        if volatile_id not in SCENARIO_VOLATILE_IDS:
            raise ScenarioValidationError("has an unsupported reconstructable volatile", path=f"{path}/id")
        turns_remaining = _optional_integer(
            value.get("turns_remaining"),
            path=f"{path}/turns_remaining",
            minimum=1,
        )
        turns_elapsed = _optional_integer(
            value.get("turns_elapsed"),
            path=f"{path}/turns_elapsed",
            minimum=0,
        )
        hp = _optional_integer(value.get("hp"), path=f"{path}/hp", minimum=1)
        move_id = value.get("move_id")
        if move_id is not None:
            move_id = normalize_id(_string(move_id, path=f"{path}/move_id"))

        limits = _VOLATILE_TURN_LIMITS.get(volatile_id)
        if limits is not None:
            if turns_remaining is None:
                raise ScenarioValidationError("is required", path=f"{path}/turns_remaining")
            if not limits[0] <= turns_remaining <= limits[1]:
                raise ScenarioValidationError(
                    f"must be between {limits[0]} and {limits[1]}",
                    path=f"{path}/turns_remaining",
                )
        elif turns_remaining is not None:
            raise ScenarioValidationError(
                "is not used by this volatile", path=f"{path}/turns_remaining"
            )

        if volatile_id in {"confusion", "encore"}:
            maximum = 5 if volatile_id == "confusion" else 6
            if turns_elapsed is None or turns_elapsed > maximum:
                raise ScenarioValidationError(
                    f"must be between 0 and {maximum}", path=f"{path}/turns_elapsed"
                )
        elif turns_elapsed is not None:
            raise ScenarioValidationError(
                "is not used by this volatile", path=f"{path}/turns_elapsed"
            )

        if volatile_id == "substitute":
            if hp is None:
                raise ScenarioValidationError("is required", path=f"{path}/hp")
        elif hp is not None:
            raise ScenarioValidationError("is only valid for Substitute", path=f"{path}/hp")

        if volatile_id in _VOLATILE_MOVE_IDS:
            if not move_id:
                raise ScenarioValidationError("is required", path=f"{path}/move_id")
        elif move_id is not None:
            raise ScenarioValidationError(
                "is only valid for Encore or Disable", path=f"{path}/move_id"
            )
        return cls(
            volatile_id=volatile_id,
            turns_remaining=turns_remaining,
            turns_elapsed=turns_elapsed,
            hp=hp,
            move_id=move_id,
        )

    def to_payload(self) -> dict[str, Any]:
        payload: dict[str, Any] = {"id": self.volatile_id}
        if self.turns_remaining is not None:
            payload["turns_remaining"] = self.turns_remaining
        if self.turns_elapsed is not None:
            payload["turns_elapsed"] = self.turns_elapsed
        if self.hp is not None:
            payload["hp"] = self.hp
        if self.move_id is not None:
            payload["move_id"] = self.move_id
        return payload


@dataclass(frozen=True)
class ScenarioField:
    weather_id: str = ""
    turns_remaining: int = 0
    permanent: bool = False

    @classmethod
    def from_payload(cls, payload: Any, *, path: str) -> "ScenarioField":
        if payload is None:
            return cls()
        value = _mapping(payload, path=path)
        weather_id = normalize_id(
            _string(value.get("weather", ""), path=f"{path}/weather", allow_empty=True)
        )
        if weather_id not in SCENARIO_WEATHER_IDS:
            raise ScenarioValidationError("has an unsupported Gen 3 weather", path=f"{path}/weather")
        turns_remaining = _integer(
            value.get("turns_remaining", 0),
            path=f"{path}/turns_remaining",
            minimum=0,
        )
        permanent = value.get("permanent", False)
        if not isinstance(permanent, bool):
            raise ScenarioValidationError("must be a boolean", path=f"{path}/permanent")
        if not weather_id:
            if turns_remaining or permanent:
                raise ScenarioValidationError(
                    "clear weather cannot have duration or permanence", path=path
                )
        elif permanent:
            if weather_id not in {"raindance", "sunnyday", "sandstorm"}:
                raise ScenarioValidationError(
                    "only ability-backed rain, sun, or sand can be permanent in Gen 3",
                    path=f"{path}/permanent",
                )
            if turns_remaining != 5:
                raise ScenarioValidationError(
                    "permanent weather must use the pinned five-turn feature value",
                    path=f"{path}/turns_remaining",
                )
        elif not 1 <= turns_remaining <= 5:
            raise ScenarioValidationError(
                "timed weather must have one to five turns remaining",
                path=f"{path}/turns_remaining",
            )
        return cls(
            weather_id=weather_id,
            turns_remaining=turns_remaining,
            permanent=permanent,
        )

    def to_payload(self) -> dict[str, Any]:
        return {
            "weather": self.weather_id,
            "turns_remaining": self.turns_remaining,
            "permanent": self.permanent,
        }


@dataclass(frozen=True)
class ScenarioPokemon:
    variant_id: str
    species: str
    level: int
    ability: str
    item: str
    moves: tuple[ScenarioMove, ...]
    current_hp: int
    max_hp: int
    nature: str = ""
    gender: str | None = None
    evs: Mapping[str, int] = field(default_factory=dict)
    ivs: Mapping[str, int] = field(default_factory=dict)
    status: ScenarioStatus = field(default_factory=ScenarioStatus)

    @classmethod
    def from_payload(cls, payload: Any, *, path: str) -> "ScenarioPokemon":
        value = _mapping(payload, path=path)
        raw_moves = _list(value.get("moves"), path=f"{path}/moves")
        gender = value.get("gender")
        if gender is not None and gender not in {"M", "F", "N"}:
            raise ScenarioValidationError("must be M, F, N, or null", path=f"{path}/gender")
        return cls(
            variant_id=_string(value.get("variant_id"), path=f"{path}/variant_id"),
            species=_string(value.get("species"), path=f"{path}/species"),
            level=_integer(value.get("level"), path=f"{path}/level", minimum=1),
            ability=_string(value.get("ability"), path=f"{path}/ability", allow_empty=True),
            item=_string(value.get("item"), path=f"{path}/item", allow_empty=True),
            moves=tuple(
                ScenarioMove.from_payload(move, path=f"{path}/moves/{index}")
                for index, move in enumerate(raw_moves)
            ),
            current_hp=_integer(value.get("current_hp"), path=f"{path}/current_hp", minimum=0),
            max_hp=_integer(value.get("max_hp"), path=f"{path}/max_hp", minimum=1),
            nature=_string(value.get("nature", ""), path=f"{path}/nature", allow_empty=True),
            gender=gender,
            evs=_string_mapping(value.get("evs"), path=f"{path}/evs"),
            ivs=_string_mapping(value.get("ivs"), path=f"{path}/ivs"),
            status=ScenarioStatus.from_payload(value.get("status"), path=f"{path}/status"),
        )

    def to_payload(self) -> dict[str, Any]:
        payload = {
            "variant_id": self.variant_id,
            "species": self.species,
            "level": self.level,
            "ability": self.ability,
            "item": self.item,
            "moves": [move.to_payload() for move in self.moves],
            "current_hp": self.current_hp,
            "max_hp": self.max_hp,
            "nature": self.nature,
            "gender": self.gender,
            "evs": dict(sorted(self.evs.items())),
            "ivs": dict(sorted(self.ivs.items())),
        }
        if self.status.status_id:
            payload["status"] = self.status.to_payload()
        return payload

    def to_fixture(self) -> FixturePokemon:
        return FixturePokemon(
            species=self.species,
            moves=tuple(move.move_id for move in self.moves),
            ability=self.ability or None,
            item=self.item or None,
            level=self.level,
            nature=self.nature,
            gender=self.gender,
            evs=dict(self.evs) or None,
            ivs=dict(self.ivs) or None,
        )


@dataclass(frozen=True)
class ScenarioSide:
    construction_mode: str
    generated_team_seed: int | None
    active_slot: int
    pokemon: tuple[ScenarioPokemon, ...]
    side_conditions: Mapping[str, int] = field(default_factory=dict)
    active_volatiles: tuple[ScenarioVolatile, ...] = ()

    @classmethod
    def from_payload(cls, payload: Any, *, path: str) -> "ScenarioSide":
        value = _mapping(payload, path=path)
        construction_mode = _string(value.get("construction_mode"), path=f"{path}/construction_mode")
        if construction_mode not in _CONSTRUCTION_MODES:
            raise ScenarioValidationError("must be generated or source-composed", path=f"{path}/construction_mode")
        generated_seed = value.get("generated_team_seed")
        if generated_seed is not None:
            generated_seed = _integer(generated_seed, path=f"{path}/generated_team_seed")
        raw_pokemon = _list(value.get("pokemon"), path=f"{path}/pokemon")
        return cls(
            construction_mode=construction_mode,
            generated_team_seed=generated_seed,
            active_slot=_integer(value.get("active_slot"), path=f"{path}/active_slot", minimum=0),
            pokemon=tuple(
                ScenarioPokemon.from_payload(pokemon, path=f"{path}/pokemon/{index}")
                for index, pokemon in enumerate(raw_pokemon)
            ),
            side_conditions=_condition_mapping(
                value.get("side_conditions"), path=f"{path}/side_conditions"
            ),
            active_volatiles=tuple(
                ScenarioVolatile.from_payload(item, path=f"{path}/active_volatiles/{index}")
                for index, item in enumerate(
                    _list(value.get("active_volatiles", []), path=f"{path}/active_volatiles")
                )
            ),
        )

    def to_payload(self) -> dict[str, Any]:
        payload = {
            "construction_mode": self.construction_mode,
            "generated_team_seed": self.generated_team_seed,
            "active_slot": self.active_slot,
            "pokemon": [pokemon.to_payload() for pokemon in self.pokemon],
        }
        if self.side_conditions:
            payload["side_conditions"] = dict(sorted(self.side_conditions.items()))
        if self.active_volatiles:
            payload["active_volatiles"] = [item.to_payload() for item in self.active_volatiles]
        return payload


@dataclass(frozen=True)
class ScenarioObjective:
    kind: str
    expected_root_actions: tuple[str, ...] = ()
    principal_variation: tuple[str, ...] = ()
    max_plies: int = 1
    verification_status: str = "unverified"
    verification_engine: str | None = None
    verification_artifact: str | None = None

    @classmethod
    def from_payload(cls, payload: Any, *, path: str) -> "ScenarioObjective":
        value = _mapping(payload, path=path)
        kind = _string(value.get("kind"), path=f"{path}/kind")
        if kind not in _OBJECTIVE_KINDS:
            raise ScenarioValidationError("has an unsupported objective kind", path=f"{path}/kind")
        verification = _mapping(value.get("verification", {}), path=f"{path}/verification")
        status = _string(verification.get("status", "unverified"), path=f"{path}/verification/status")
        if status not in _VERIFICATION_STATUSES:
            raise ScenarioValidationError("has an unsupported status", path=f"{path}/verification/status")
        expected = _list(value.get("expected_root_actions", []), path=f"{path}/expected_root_actions")
        variation = _list(value.get("principal_variation", []), path=f"{path}/principal_variation")
        return cls(
            kind=kind,
            expected_root_actions=tuple(
                _string(item, path=f"{path}/expected_root_actions/{index}")
                for index, item in enumerate(expected)
            ),
            principal_variation=tuple(
                _string(item, path=f"{path}/principal_variation/{index}")
                for index, item in enumerate(variation)
            ),
            max_plies=_integer(value.get("max_plies", 1), path=f"{path}/max_plies", minimum=1),
            verification_status=status,
            verification_engine=(
                _string(verification["engine"], path=f"{path}/verification/engine")
                if verification.get("engine") is not None
                else None
            ),
            verification_artifact=(
                _string(verification["artifact"], path=f"{path}/verification/artifact")
                if verification.get("artifact") is not None
                else None
            ),
        )

    def to_payload(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "expected_root_actions": list(self.expected_root_actions),
            "principal_variation": list(self.principal_variation),
            "max_plies": self.max_plies,
            "verification": {
                "status": self.verification_status,
                "engine": self.verification_engine,
                "artifact": self.verification_artifact,
            },
        }


@dataclass(frozen=True)
class EndgameScenario:
    scenario_id: str
    title: str
    description: str
    tags: tuple[str, ...]
    seed: int
    randbat_source_hash: str
    teams: Mapping[str, ScenarioSide]
    objective: ScenarioObjective
    perspective: str = "p1"
    side_to_move: str = "p1"
    knowledge_mode: str = "fully_revealed"
    author_notes: str = ""
    replay_proven: bool = False
    turn_number: int = 1
    battle_field: ScenarioField = field(default_factory=ScenarioField)
    schema_version: str = ENDGAME_SCENARIO_SCHEMA_VERSION

    @classmethod
    def from_payload(cls, payload: Any) -> "EndgameScenario":
        value = _mapping(payload, path="")
        schema_version = _string(value.get("schema_version"), path="/schema_version")
        if schema_version != ENDGAME_SCENARIO_SCHEMA_VERSION:
            raise ScenarioValidationError(
                f"unsupported schema version {schema_version!r}", path="/schema_version"
            )
        if value.get("format_id") != "gen3customgame":
            raise ScenarioValidationError(
                "MVP scenarios must materialize in gen3customgame", path="/format_id"
            )
        if value.get("source_format_id") != "gen3randombattle":
            raise ScenarioValidationError(
                "MVP scenarios must use the Gen 3 randbats source catalog", path="/source_format_id"
            )
        provenance = _mapping(value.get("provenance"), path="/provenance")
        raw_teams = _mapping(value.get("teams"), path="/teams")
        if set(raw_teams) != set(_SIDES):
            raise ScenarioValidationError("must contain exactly p1 and p2", path="/teams")
        tags = _list(value.get("tags", []), path="/tags")
        perspective = _string(value.get("perspective", "p1"), path="/perspective")
        side_to_move = _string(value.get("side_to_move", perspective), path="/side_to_move")
        if perspective not in _SIDES or side_to_move not in _SIDES:
            raise ScenarioValidationError("must be p1 or p2", path="/perspective")
        knowledge_mode = _string(value.get("knowledge_mode", "fully_revealed"), path="/knowledge_mode")
        if knowledge_mode != "fully_revealed":
            raise ScenarioValidationError("MVP supports fully_revealed only", path="/knowledge_mode")
        return cls(
            scenario_id=_string(value.get("scenario_id"), path="/scenario_id"),
            title=_string(value.get("title"), path="/title"),
            description=_string(value.get("description", ""), path="/description", allow_empty=True),
            tags=tuple(_string(tag, path=f"/tags/{index}") for index, tag in enumerate(tags)),
            seed=_integer(value.get("seed", 0), path="/seed"),
            randbat_source_hash=_string(provenance.get("randbat_source_hash"), path="/provenance/randbat_source_hash"),
            teams={side: ScenarioSide.from_payload(raw_teams[side], path=f"/teams/{side}") for side in _SIDES},
            objective=ScenarioObjective.from_payload(value.get("objective"), path="/objective"),
            perspective=perspective,
            side_to_move=side_to_move,
            knowledge_mode=knowledge_mode,
            author_notes=_string(value.get("author_notes", ""), path="/author_notes", allow_empty=True),
            replay_proven=bool(provenance.get("replay_proven", False)),
            turn_number=_integer(value.get("turn", 1), path="/turn", minimum=1),
            battle_field=ScenarioField.from_payload(value.get("field"), path="/field"),
            schema_version=schema_version,
        )

    def to_payload(self) -> dict[str, Any]:
        payload = {
            "schema_version": self.schema_version,
            "scenario_id": self.scenario_id,
            "title": self.title,
            "description": self.description,
            "tags": list(self.tags),
            "format_id": "gen3customgame",
            "source_format_id": "gen3randombattle",
            "seed": self.seed,
            "provenance": {
                "randbat_source_hash": self.randbat_source_hash,
                "replay_proven": self.replay_proven,
            },
            "knowledge_mode": self.knowledge_mode,
            "perspective": self.perspective,
            "side_to_move": self.side_to_move,
            "teams": {side: self.teams[side].to_payload() for side in _SIDES},
            "objective": self.objective.to_payload(),
            "author_notes": self.author_notes,
        }
        if self.battle_field.weather_id:
            payload["field"] = self.battle_field.to_payload()
        if self.turn_number != 1:
            payload["turn"] = self.turn_number
        return payload

    def canonical_json(self) -> str:
        return json.dumps(self.to_payload(), sort_keys=True, indent=2) + "\n"

    @classmethod
    def from_payload_json(cls, payload: str) -> "EndgameScenario":
        try:
            decoded = json.loads(payload)
        except json.JSONDecodeError as exc:
            raise ScenarioValidationError("invalid JSON") from exc
        return cls.from_payload(decoded)

    def side(self, player: str) -> ScenarioSide:
        try:
            return self.teams[player]
        except KeyError as exc:
            raise ScenarioValidationError("must be p1 or p2", path="/teams") from exc
