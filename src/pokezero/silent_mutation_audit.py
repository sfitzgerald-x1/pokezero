"""Bounded audit of public-relevant simulator mutations without protocol backing.

The bridge snapshot is an oracle and contains hidden set data.  This module
uses it only in memory, projects revealed Pokemon and public field families,
and serializes field paths plus protocol tags -- never raw simulator values.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Mapping

from .local_showdown import LocalShowdownSnapshot
from .showdown import _normalize_identifier


_POKEMON_FIELDS = (
    "active",
    "fainted",
    "hp",
    "status",
    "boosts",
    "volatiles",
    "types",
)
_FIELD_PROTOCOL_TAGS: Mapping[str, frozenset[str]] = {
    "active": frozenset({"switch", "drag", "replace", "faint"}),
    "fainted": frozenset({"faint", "switch", "drag", "replace"}),
    "hp": frozenset({"-damage", "-heal", "-sethp", "faint", "switch", "drag", "replace"}),
    "status": frozenset({"-status", "-curestatus", "-cureteam", "faint"}),
    "boosts": frozenset(
        {"-boost", "-unboost", "-setboost", "-clearallboost", "-clearboost", "switch", "drag", "replace", "faint"}
    ),
    "volatiles": frozenset(
        {"-start", "-end", "-activate", "-singleturn", "-singlemove", "switch", "drag", "replace", "faint"}
    ),
    "types": frozenset({"-start", "-end", "-formechange", "-detailschange", "-transform"}),
    "weather": frozenset({"-weather"}),
    "side_conditions": frozenset({"-sidestart", "-sideend", "-swapsideconditions"}),
}


@dataclass(frozen=True)
class SilentMutationAggregate:
    """A public-safe, aggregated mutation classification.

    ``entity`` contains only a side and an already-revealed species.  Values are
    intentionally omitted: a bridge snapshot must never become an audit artifact.
    """

    entity: str
    field: str
    classification: str
    protocol_tags: tuple[str, ...]
    count: int
    example_game_id: str
    example_turn: int
    detail: str

    def to_json_dict(self) -> dict[str, object]:
        return {
            "entity": self.entity,
            "field": self.field,
            "classification": self.classification,
            "protocol_tags": list(self.protocol_tags),
            "count": self.count,
            "example_game_id": self.example_game_id,
            "example_turn": self.example_turn,
            "detail": self.detail,
        }


@dataclass
class SilentMutationAuditReport:
    """Aggregate silent-mutation audit evidence for one or more bounded games."""

    steps_audited: int = 0
    public_entities_observed: int = 0
    ambiguous_entities_skipped: int = 0
    _counts: Counter[tuple[str, str, str, tuple[str, ...], str]] = field(default_factory=Counter)
    _examples: dict[tuple[str, str, str, tuple[str, ...], str], tuple[str, int]] = field(default_factory=dict)

    def record_transition(
        self,
        before: LocalShowdownSnapshot,
        after: LocalShowdownSnapshot,
        *,
        game_id: str,
    ) -> None:
        """Diff one decision transition, retaining only public-safe evidence."""

        before_surface, before_ambiguous = _public_mutation_surface(before)
        after_surface, after_ambiguous = _public_mutation_surface(after)
        self.steps_audited += 1
        self.public_entities_observed += len(set(before_surface) | set(after_surface))
        self.ambiguous_entities_skipped += before_ambiguous + after_ambiguous
        protocol_tags = _protocol_tags(after.protocol_lines[len(before.protocol_lines) :])
        turn = int(after.replay.turn_number)

        for entity in sorted((set(before_surface) & set(after_surface)) - {"field"}):
            before_fields = before_surface[entity]
            after_fields = after_surface[entity]
            for field in _POKEMON_FIELDS:
                if before_fields[field] == after_fields[field]:
                    continue
                classification, detail = _classify_pokemon_mutation(
                    field=field,
                    before_fields=before_fields,
                    after_fields=after_fields,
                    protocol_tags=protocol_tags,
                    after=after,
                    entity=entity,
                )
                self._record(
                    entity=entity,
                    field=field,
                    classification=classification,
                    protocol_tags=protocol_tags,
                    game_id=game_id,
                    turn=turn,
                    detail=detail,
                )

        for field in ("weather", "side_conditions"):
            if before_surface.get("field", {}).get(field) == after_surface.get("field", {}).get(field):
                continue
            classification = "protocol-backed" if _has_protocol_backing(field, protocol_tags) else "silent-candidate"
            detail = (
                "A matching public protocol family was emitted."
                if classification == "protocol-backed"
                else "No matching public protocol family was emitted; adjudicate belief tracking or accepted loss."
            )
            self._record(
                entity="field",
                field=field,
                classification=classification,
                protocol_tags=protocol_tags,
                game_id=game_id,
                turn=turn,
                detail=detail,
            )

    def _record(
        self,
        *,
        entity: str,
        field: str,
        classification: str,
        protocol_tags: tuple[str, ...],
        game_id: str,
        turn: int,
        detail: str,
    ) -> None:
        # Keep one stable example while counting every observed occurrence.
        key = (entity, field, classification, protocol_tags, detail)
        self._counts[key] += 1
        self._examples.setdefault(key, (game_id, turn))

    def aggregates(self) -> tuple[SilentMutationAggregate, ...]:
        return tuple(
            SilentMutationAggregate(
                entity=entity,
                field=field,
                classification=classification,
                protocol_tags=protocol_tags,
                count=count,
                example_game_id=game_id,
                example_turn=turn,
                detail=detail,
            )
            for (entity, field, classification, protocol_tags, detail), count in sorted(self._counts.items())
            for game_id, turn in (self._examples[(entity, field, classification, protocol_tags, detail)],)
        )

    def to_json_dict(self) -> dict[str, object]:
        aggregates = self.aggregates()
        classifications = Counter(item.classification for item in aggregates for _ in range(item.count))
        return {
            "schema_version": "pokezero.silent-mutation-audit.v1",
            "steps_audited": self.steps_audited,
            "public_entities_observed": self.public_entities_observed,
            "ambiguous_entities_skipped": self.ambiguous_entities_skipped,
            "classification_counts": dict(sorted(classifications.items())),
            "silent_candidate_count": classifications["silent-candidate"],
            "aggregates": [item.to_json_dict() for item in aggregates],
            "limitations": [
                "Artifacts omit raw bridge values and unobserved Pokemon entirely.",
                "A silent-candidate is an audit shortlist entry, not an encoder defect verdict.",
                "Internal simulator bookkeeping outside the public-relevant field families is out of scope.",
            ],
        }


def _public_mutation_surface(
    snapshot: LocalShowdownSnapshot,
) -> tuple[dict[str, dict[str, object]], int]:
    """Project only revealed Pokemon and public field families from a bridge snapshot."""

    battle = snapshot.bridge_snapshot.get("battle")
    if not isinstance(battle, Mapping):
        return {"field": {"weather": None, "side_conditions": ()}}, 0
    surface: dict[str, dict[str, object]] = {
        "field": {
            "weather": _normalized_value(_mapping_value(battle.get("field"), "weather")),
            "side_conditions": (),
        }
    }
    ambiguous = 0
    for side in battle.get("sides", ()):
        if not isinstance(side, Mapping):
            continue
        side_id = side.get("id")
        if side_id not in {"p1", "p2"}:
            continue
        raw_pokemon = tuple(item for item in side.get("pokemon", ()) if isinstance(item, Mapping))
        known_species = {
            _normalized_value(pokemon.species)
            for pokemon in snapshot.replay.public_revealed.get(str(side_id), ())
        }
        counts = Counter(_raw_species(pokemon) for pokemon in raw_pokemon)
        for pokemon in raw_pokemon:
            species = _raw_species(pokemon)
            if not species or species not in known_species:
                continue
            if counts[species] != 1:
                ambiguous += 1
                continue
            replay_volatiles = getattr(snapshot.replay, "volatiles", {})
            public_volatiles = replay_volatiles.get(str(side_id), ()) if pokemon.get("isActive") else ()
            surface[f"{side_id}:{species}"] = _pokemon_surface(pokemon, public_volatiles=public_volatiles)
        surface["field"]["side_conditions"] = tuple(
            sorted(
                (
                    str(candidate.get("id")),
                    tuple(sorted(str(key) for key in candidate.get("sideConditions", {}).keys())),
                )
                for candidate in battle.get("sides", ())
                if isinstance(candidate, Mapping) and candidate.get("id") in {"p1", "p2"}
            )
        )
    return surface, ambiguous


def _pokemon_surface(
    pokemon: Mapping[str, Any],
    *,
    public_volatiles: tuple[str, ...] | list[str],
) -> dict[str, object]:
    """Return in-memory values used only for equality checks, never artifact payloads."""

    boosts = pokemon.get("boosts")
    volatiles = pokemon.get("volatiles")
    types = pokemon.get("types")
    return {
        "active": bool(pokemon.get("isActive")),
        "fainted": bool(pokemon.get("fainted")),
        "hp": _numeric_pair(pokemon.get("hp"), pokemon.get("maxhp")),
        "status": _normalized_value(pokemon.get("status")),
        "boosts": tuple(sorted((str(key), int(value)) for key, value in boosts.items())) if isinstance(boosts, Mapping) else (),
        # Choice locking and related action constraints live in the raw simulator but are
        # request-private. Keep only volatile ids already exposed by the public replay fold.
        "volatiles": tuple(
            sorted(
                _normalized_value(key)
                for key in volatiles
                if _normalized_value(key) in {_normalized_value(value) for value in public_volatiles}
            )
        )
        if isinstance(volatiles, Mapping)
        else (),
        "types": tuple(sorted(_normalized_value(value) for value in types)) if isinstance(types, list) else (),
    }


def _classify_pokemon_mutation(
    *,
    field: str,
    before_fields: Mapping[str, object],
    after_fields: Mapping[str, object],
    protocol_tags: tuple[str, ...],
    after: LocalShowdownSnapshot,
    entity: str,
) -> tuple[str, str]:
    if _has_protocol_backing(field, protocol_tags):
        return "protocol-backed", "A matching public protocol family was emitted."
    # A switch line itself exposes the incoming active Pokemon's condition. Treat that direct
    # public request boundary as backing for its status/HP, but never for a switched-out mon.
    if (
        field in {"hp", "status"}
        and bool(after_fields["active"])
        and "switch" in protocol_tags
    ):
        return "protocol-backed", "The switch boundary directly reported the incoming Pokemon condition."
    if field == "status" and _belief_status_matches(after, entity, after_fields["status"]):
        return "belief-inferred", "The public belief state matches the post-mutation status without a direct status tag."
    return "silent-candidate", "No matching public protocol family or belief inference covered this mutation."


def _belief_status_matches(snapshot: LocalShowdownSnapshot, entity: str, status: object) -> bool:
    side, _, species = entity.partition(":")
    if side not in {"p1", "p2"} or not species:
        return False
    for belief in snapshot.belief_engine.snapshot().side(side):
        if _normalized_value(belief.species) == species:
            return _normalized_value(belief.status) == status
    return False


def _has_protocol_backing(field: str, protocol_tags: tuple[str, ...]) -> bool:
    return bool(_FIELD_PROTOCOL_TAGS[field] & set(protocol_tags))


def _protocol_tags(lines: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(sorted({parts[1] for line in lines if (parts := line.split("|")) and len(parts) > 1 and parts[1]}))


def _mapping_value(value: object, key: str) -> object:
    return value.get(key) if isinstance(value, Mapping) else None


def _numeric_pair(numerator: object, denominator: object) -> tuple[float, float] | None:
    if not isinstance(numerator, (int, float)) or not isinstance(denominator, (int, float)):
        return None
    return float(numerator), float(denominator)


def _raw_species(pokemon: Mapping[str, Any]) -> str:
    species = str(pokemon.get("species") or "")
    if species.startswith("[Species:") and species.endswith("]"):
        species = species[len("[Species:") : -1]
    return _normalized_value(species)


def _normalized_value(value: object) -> str:
    return _normalize_identifier(str(value or ""))
