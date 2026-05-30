"""Public battle belief tracking for replay, overlays, and training inputs.

The engine in this module only consumes public information. It is intentionally
format-agnostic: random-battle set sources can be plugged in later without
changing the public-state tracking API.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import re
from typing import Any, Mapping, Optional, Protocol, Sequence


@dataclass(frozen=True)
class CandidateSetSummary:
    species: str
    candidate_count: Optional[int] = None
    uncertainty: float = 1.0
    notes: tuple[str, ...] = ()


class PokemonSetSource(Protocol):
    def summarize(
        self,
        *,
        format_id: Optional[str],
        species: str,
        revealed_moves: tuple[str, ...],
    ) -> CandidateSetSummary | None:
        ...


@dataclass(frozen=True)
class RevealedPokemonBelief:
    showdown_slot: str
    species: str
    condition: Optional[str] = None
    status: Optional[str] = None
    active: bool = False
    revealed_moves: tuple[str, ...] = ()
    revealed_ability: Optional[str] = None
    revealed_item: Optional[str] = None
    candidate_set_count: Optional[int] = None
    uncertainty: float = 1.0

    @property
    def key(self) -> str:
        return belief_key(self.showdown_slot, self.species)

    def to_overlay_payload(self) -> dict[str, Any]:
        return {
            "showdown_slot": self.showdown_slot,
            "species": self.species,
            "condition": self.condition,
            "status": self.status,
            "active": self.active,
            "revealed_moves": list(self.revealed_moves),
            "revealed_ability": self.revealed_ability,
            "revealed_item": self.revealed_item,
            "candidate_set_count": self.candidate_set_count,
            "uncertainty": self.uncertainty,
        }


@dataclass(frozen=True)
class PlayerBeliefView:
    self_slot: str
    opponent_slot: str
    self_pokemon: tuple[RevealedPokemonBelief, ...]
    opponent_pokemon: tuple[RevealedPokemonBelief, ...]

    def opponent_by_species(self) -> Mapping[str, RevealedPokemonBelief]:
        return {_normalize_species(pokemon.species): pokemon for pokemon in self.opponent_pokemon}

    def to_overlay_payload(self) -> dict[str, Any]:
        return {
            "self_slot": self.self_slot,
            "opponent_slot": self.opponent_slot,
            "self_pokemon": [pokemon.to_overlay_payload() for pokemon in self.self_pokemon],
            "opponent_pokemon": [pokemon.to_overlay_payload() for pokemon in self.opponent_pokemon],
        }


@dataclass(frozen=True)
class BattleBeliefSnapshot:
    format_id: Optional[str]
    event_count: int
    sides: Mapping[str, tuple[RevealedPokemonBelief, ...]]

    def side(self, showdown_slot: str) -> tuple[RevealedPokemonBelief, ...]:
        return self.sides.get(showdown_slot, ())

    def for_player(self, showdown_slot: str) -> PlayerBeliefView:
        opponent_slot = _opponent_slot(showdown_slot)
        return PlayerBeliefView(
            self_slot=showdown_slot,
            opponent_slot=opponent_slot,
            self_pokemon=self.side(showdown_slot),
            opponent_pokemon=self.side(opponent_slot),
        )

    def to_overlay_payload(self) -> dict[str, Any]:
        return {
            "format_id": self.format_id,
            "event_count": self.event_count,
            "sides": {
                slot: [pokemon.to_overlay_payload() for pokemon in pokemon_list]
                for slot, pokemon_list in self.sides.items()
            },
        }


class PublicBattleBeliefEngine:
    def __init__(
        self,
        *,
        format_id: Optional[str] = None,
        set_source: PokemonSetSource | None = None,
    ) -> None:
        self.format_id = format_id
        self.set_source = set_source
        self._event_count = 0
        self._sides: dict[str, list[RevealedPokemonBelief]] = {"p1": [], "p2": []}

    @classmethod
    def from_events(
        cls,
        events: Sequence[Any],
        *,
        format_id: Optional[str] = None,
        set_source: PokemonSetSource | None = None,
    ) -> "PublicBattleBeliefEngine":
        engine = cls(format_id=format_id, set_source=set_source)
        for event in events:
            engine.ingest_event(event)
        return engine

    def ingest_event(self, event: Any) -> None:
        event_type = _event_value(event, "event_type")
        actor_slot = _event_value(event, "actor_slot")
        actor_ident = _event_value(event, "actor_ident")
        target_slot = _event_value(event, "target_slot")
        target_ident = _event_value(event, "target_ident")
        primary = _event_value(event, "primary")
        secondary = _event_value(event, "secondary")
        self._event_count += 1

        if event_type in {"switch", "drag", "replace"} and actor_slot and primary:
            self._mark_side_inactive(actor_slot)
            self._upsert(
                showdown_slot=actor_slot,
                species=str(primary),
                condition=_string_or_none(secondary),
                active=True,
            )
            return

        if event_type == "move" and actor_slot and primary:
            species = self._active_species(actor_slot) or _species_from_ident(actor_ident)
            if species:
                belief = self._upsert(showdown_slot=actor_slot, species=species)
                self._replace_belief(
                    belief,
                    revealed_moves=_append_unique(belief.revealed_moves, str(primary)),
                )
            return

        if event_type in {"-damage", "-heal"} and target_slot:
            belief = self._target_belief(target_slot, target_ident)
            if belief is not None:
                self._replace_belief(belief, condition=_string_or_none(primary))
            return

        if event_type == "-status" and target_slot:
            belief = self._target_belief(target_slot, target_ident)
            if belief is not None:
                self._replace_belief(belief, status=_string_or_none(primary))
            return

        if event_type == "-curestatus" and target_slot:
            belief = self._target_belief(target_slot, target_ident)
            if belief is not None:
                self._replace_belief(belief, status=None)
            return

        if event_type == "faint" and target_slot:
            belief = self._target_belief(target_slot, target_ident)
            if belief is not None:
                self._replace_belief(belief, condition="0 fnt", active=False)
            return

        if event_type == "-ability" and target_slot and primary:
            belief = self._target_belief(target_slot, target_ident)
            if belief is not None:
                self._replace_belief(belief, revealed_ability=str(primary))
            return

        if event_type == "-item" and target_slot and primary:
            belief = self._target_belief(target_slot, target_ident)
            if belief is not None:
                self._replace_belief(belief, revealed_item=str(primary))

    def snapshot(self) -> BattleBeliefSnapshot:
        return BattleBeliefSnapshot(
            format_id=self.format_id,
            event_count=self._event_count,
            sides={slot: tuple(pokemon) for slot, pokemon in self._sides.items()},
        )

    def _upsert(
        self,
        *,
        showdown_slot: str,
        species: str,
        condition: Optional[str] = None,
        active: Optional[bool] = None,
    ) -> RevealedPokemonBelief:
        normalized_species = _normalize_species(species)
        side = self._sides.setdefault(showdown_slot, [])
        for index, belief in enumerate(side):
            if _normalize_species(belief.species) == normalized_species:
                updated = belief
                if condition is not None:
                    updated = replace(updated, condition=condition)
                if active is not None:
                    updated = replace(updated, active=active)
                updated = self._with_set_summary(updated)
                side[index] = updated
                return updated
        created = self._with_set_summary(
            RevealedPokemonBelief(
                showdown_slot=showdown_slot,
                species=species,
                condition=condition,
                active=bool(active),
            )
        )
        side.append(created)
        return created

    def _replace_belief(self, belief: RevealedPokemonBelief, **changes: Any) -> RevealedPokemonBelief:
        side = self._sides.get(belief.showdown_slot, [])
        for index, candidate in enumerate(side):
            if candidate.key == belief.key:
                updated = self._with_set_summary(replace(candidate, **changes))
                side[index] = updated
                return updated
        return belief

    def _mark_side_inactive(self, showdown_slot: str) -> None:
        self._sides[showdown_slot] = [
            replace(pokemon, active=False)
            for pokemon in self._sides.get(showdown_slot, [])
        ]

    def _active_species(self, showdown_slot: str) -> Optional[str]:
        active = self._active_belief(showdown_slot)
        return active.species if active is not None else None

    def _active_belief(self, showdown_slot: str) -> RevealedPokemonBelief | None:
        return next((pokemon for pokemon in self._sides.get(showdown_slot, []) if pokemon.active), None)

    def _target_belief(
        self,
        showdown_slot: str,
        target_ident: Optional[str],
    ) -> RevealedPokemonBelief | None:
        active = self._active_belief(showdown_slot)
        if active is not None:
            return active
        species = _species_from_ident(target_ident)
        if species is None:
            return None
        return self._upsert(showdown_slot=showdown_slot, species=species)

    def _with_set_summary(self, belief: RevealedPokemonBelief) -> RevealedPokemonBelief:
        if self.set_source is None:
            return belief
        summary = self.set_source.summarize(
            format_id=self.format_id,
            species=belief.species,
            revealed_moves=belief.revealed_moves,
        )
        if summary is None:
            return belief
        return replace(
            belief,
            candidate_set_count=summary.candidate_count,
            uncertainty=summary.uncertainty,
        )


def belief_key(showdown_slot: str, species: str) -> str:
    return f"{showdown_slot}:{_normalize_species(species)}"


def _event_value(event: Any, name: str) -> Optional[str]:
    if isinstance(event, Mapping):
        value = event.get(name)
    else:
        value = getattr(event, name, None)
    return str(value) if value is not None else None


def _append_unique(values: tuple[str, ...], value: str) -> tuple[str, ...]:
    normalized = _normalize_identifier(value)
    if any(_normalize_identifier(existing) == normalized for existing in values):
        return values
    return (*values, value)


def _species_from_ident(ident: Optional[str]) -> Optional[str]:
    if not ident:
        return None
    species = str(ident).split(":", 1)[-1].strip()
    return species or None


def _opponent_slot(showdown_slot: str) -> str:
    if showdown_slot == "p1":
        return "p2"
    if showdown_slot == "p2":
        return "p1"
    raise ValueError(f"Unsupported Showdown slot: {showdown_slot!r}.")


def _string_or_none(value: Optional[str]) -> Optional[str]:
    return value if value not in {"", None} else None


def _normalize_species(species: str) -> str:
    return _normalize_identifier(species)


def _normalize_identifier(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value).lower())
