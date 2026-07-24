"""Canonical, portable endgame-scenario data model.

The browser deliberately edits these typed values instead of a serialized Pokemon Showdown
battle.  A scenario describes the allowed authoring surface only; the local bridge owns the
simulator snapshot and applies the separately validated HP/PP patch.
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


def _string_mapping(value: Any, *, path: str) -> dict[str, int]:
    if value is None:
        return {}
    mapping = _mapping(value, path=path)
    result: dict[str, int] = {}
    for key, item in mapping.items():
        result[_string(key, path=path)] = _integer(item, path=f"{path}/{key}", minimum=0)
    return result


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
        )

    def to_payload(self) -> dict[str, Any]:
        return {
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
        )

    def to_payload(self) -> dict[str, Any]:
        return {
            "construction_mode": self.construction_mode,
            "generated_team_seed": self.generated_team_seed,
            "active_slot": self.active_slot,
            "pokemon": [pokemon.to_payload() for pokemon in self.pokemon],
        }


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
            schema_version=schema_version,
        )

    def to_payload(self) -> dict[str, Any]:
        return {
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
