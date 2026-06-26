"""Public battle belief tracking for replay, overlays, and training inputs.

The engine in this module only consumes public information. It is intentionally
format-agnostic: random-battle set sources can be plugged in later without
changing the public-state tracking API.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, replace
import math
import random
import re
from typing import Any, Mapping, Optional, Protocol, Sequence


@dataclass(frozen=True)
class BeliefEvidence:
    kind: str
    detail: str
    source_line: Optional[str] = None

    def to_payload(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "detail": self.detail,
            "source_line": self.source_line,
        }


@dataclass(frozen=True)
class CandidateSetSummary:
    species: str
    candidate_count: Optional[int] = None
    uncertainty: float = 1.0
    notes: tuple[str, ...] = ()
    possible_abilities: tuple[str, ...] = ()
    possible_items: tuple[str, ...] = ()
    possible_moves: tuple[str, ...] = ()
    candidate_variants: tuple[Mapping[str, Any], ...] = ()
    source_metadata: Mapping[str, Any] | None = None


class PokemonSetSource(Protocol):
    def summarize(
        self,
        *,
        format_id: Optional[str],
        species: str,
        revealed_moves: tuple[str, ...],
        revealed_ability: Optional[str] = None,
        revealed_item: Optional[str] = None,
        ruled_out_abilities: tuple[str, ...] = (),
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
    ruled_out_abilities: tuple[str, ...] = ()
    candidate_set_count: Optional[int] = None
    uncertainty: float = 1.0
    possible_abilities: tuple[str, ...] = ()
    possible_items: tuple[str, ...] = ()
    possible_moves: tuple[str, ...] = ()
    candidate_variants: tuple[Mapping[str, Any], ...] = ()
    source_metadata: Mapping[str, Any] | None = None
    evidence: tuple[BeliefEvidence, ...] = ()

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
            "ruled_out_abilities": list(self.ruled_out_abilities),
            "candidate_set_count": self.candidate_set_count,
            "uncertainty": self.uncertainty,
            "possible_abilities": list(self.possible_abilities),
            "possible_items": list(self.possible_items),
            "possible_moves": list(self.possible_moves),
            "candidate_variants": [dict(variant) for variant in self.candidate_variants],
            "source_metadata": dict(self.source_metadata) if self.source_metadata else None,
            "evidence": [item.to_payload() for item in self.evidence],
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
class DeterminizedOpponentPokemon:
    showdown_slot: str
    species: str
    active: bool
    condition: Optional[str] = None
    status: Optional[str] = None
    revealed_moves: tuple[str, ...] = ()
    variant_id: Optional[str] = None
    source_set_id: Optional[str] = None
    role: Optional[str] = None
    level: Optional[int] = None
    moves: tuple[str, ...] = ()
    ability: Optional[str] = None
    item: Optional[str] = None
    candidate_count: Optional[int] = None
    uncertainty: float = 1.0
    possible_abilities: tuple[str, ...] = ()
    possible_items: tuple[str, ...] = ()
    possible_moves: tuple[str, ...] = ()
    source_metadata: Mapping[str, Any] | None = None

    @property
    def resolved(self) -> bool:
        return bool(self.variant_id or self.source_set_id or self.moves or self.ability or self.item)

    def to_payload(self) -> dict[str, Any]:
        return {
            "showdown_slot": self.showdown_slot,
            "species": self.species,
            "active": self.active,
            "condition": self.condition,
            "status": self.status,
            "revealed_moves": list(self.revealed_moves),
            "resolved": self.resolved,
            "variant_id": self.variant_id,
            "source_set_id": self.source_set_id,
            "role": self.role,
            "level": self.level,
            "moves": list(self.moves),
            "ability": self.ability,
            "item": self.item,
            "candidate_count": self.candidate_count,
            "uncertainty": self.uncertainty,
            "possible_abilities": list(self.possible_abilities),
            "possible_items": list(self.possible_items),
            "possible_moves": list(self.possible_moves),
            "source_metadata": dict(self.source_metadata) if self.source_metadata else None,
        }


@dataclass(frozen=True)
class OpponentBeliefDeterminization:
    """One sampled opponent hidden-set realization from player-knowable belief state."""

    self_slot: str
    opponent_slot: str
    sample_index: int
    combination_count: int
    opponent_pokemon: tuple[DeterminizedOpponentPokemon, ...]

    @property
    def unresolved_count(self) -> int:
        return sum(1 for pokemon in self.opponent_pokemon if not pokemon.resolved)

    def to_payload(self) -> dict[str, Any]:
        return {
            "self_slot": self.self_slot,
            "opponent_slot": self.opponent_slot,
            "sample_index": self.sample_index,
            "combination_count": self.combination_count,
            "unresolved_count": self.unresolved_count,
            "opponent_pokemon": [pokemon.to_payload() for pokemon in self.opponent_pokemon],
        }


def sample_opponent_determinizations(
    view: PlayerBeliefView,
    *,
    sample_count: int = 1,
    rng: random.Random | None = None,
) -> tuple[OpponentBeliefDeterminization, ...]:
    """Sample bounded opponent hidden-set realizations for search.

    The sampler only uses public, player-relative belief. Candidate variants remain in source order
    for deterministic enumeration; passing ``rng`` switches to unweighted random sampling. No
    probabilities are invented, and unsourced/unknown Pokemon stay unresolved.
    """

    if sample_count <= 0:
        raise ValueError("sample_count must be positive.")
    choices_by_pokemon = tuple(tuple(pokemon.candidate_variants) for pokemon in view.opponent_pokemon)
    choice_counts = tuple(max(1, len(choices)) for choices in choices_by_pokemon)
    combination_count = math.prod(choice_counts) if choice_counts else 1
    result_count = sample_count if rng is not None else min(sample_count, combination_count)
    if not choices_by_pokemon:
        result_count = 1

    results: list[OpponentBeliefDeterminization] = []
    for sample_index in range(result_count):
        if rng is None:
            selected_variants = _deterministic_variant_selection(choices_by_pokemon, sample_index)
        else:
            selected_variants = tuple(
                choices[rng.randrange(len(choices))] if choices else None
                for choices in choices_by_pokemon
            )
        results.append(
            OpponentBeliefDeterminization(
                self_slot=view.self_slot,
                opponent_slot=view.opponent_slot,
                sample_index=sample_index,
                combination_count=combination_count,
                opponent_pokemon=tuple(
                    _determinized_pokemon(pokemon, variant)
                    for pokemon, variant in zip(view.opponent_pokemon, selected_variants, strict=True)
                ),
            )
        )
    return tuple(results)


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
        self._pending_switches: list[_PendingSwitch] = []

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
        raw_line = _event_value(event, "raw_line")
        self._event_count += 1

        if event_type not in {"switch", "drag", "replace"}:
            self._resolve_pending_switches_for_event(event)
        elif self._pending_switches:
            self._resolve_pending_switches_as_no_trigger(raw_line)
        self._record_raw_ability_reveal(event)

        if event_type in {"switch", "drag", "replace"} and actor_slot and primary:
            self._mark_side_inactive(actor_slot)
            belief = self._upsert(
                showdown_slot=actor_slot,
                species=str(primary),
                condition=_string_or_none(secondary),
                active=True,
            )
            if self._can_queue_intimidate_non_trigger(belief):
                self._pending_switches.append(
                    _PendingSwitch(
                        showdown_slot=actor_slot,
                        ident=actor_ident,
                        species=belief.species,
                    )
                )
            return

        if event_type == "move" and actor_slot and primary:
            species = self._active_species(actor_slot) or _species_from_ident(actor_ident)
            if species:
                belief = self._upsert(showdown_slot=actor_slot, species=species)
                revealed_moves = _append_unique(belief.revealed_moves, str(primary))
                evidence = belief.evidence
                if revealed_moves != belief.revealed_moves:
                    evidence = _append_evidence(
                        evidence,
                        BeliefEvidence(
                            kind="revealed-move",
                            detail=f"Observed {primary}; incompatible set variants were removed.",
                            source_line=raw_line,
                        ),
                    )
                self._replace_belief(
                    belief,
                    revealed_moves=revealed_moves,
                    evidence=evidence,
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

        if event_type in {"-ability", "ability"} and target_slot and primary:
            belief = self._target_belief(target_slot, target_ident)
            if belief is not None:
                self._replace_belief(
                    belief,
                    revealed_ability=str(primary),
                    evidence=_append_evidence(
                        belief.evidence,
                        BeliefEvidence(
                            kind="confirmed-ability",
                            detail=f"Confirmed ability {primary}; incompatible set variants were removed.",
                            source_line=raw_line,
                        ),
                    ),
                )
            return

        if event_type == "-item" and target_slot and primary:
            belief = self._target_belief(target_slot, target_ident)
            if belief is not None:
                self._replace_belief(
                    belief,
                    revealed_item=str(primary),
                    evidence=_append_evidence(
                        belief.evidence,
                        BeliefEvidence(
                            kind="revealed-item",
                            detail=f"Observed item {primary}; incompatible set variants were removed.",
                            source_line=raw_line,
                        ),
                    ),
                )

    def resolve_pending_switches_at_boundary(self) -> None:
        self._resolve_pending_switches_as_no_trigger(None)

    def resolved_player_view(self, showdown_slot: str) -> "PlayerBeliefView":
        """Boundary-resolved belief view for a slot WITHOUT mutating this engine.

        Resolving pending switches at a boundary is destructive, so a persistent engine
        (fed incrementally across observations) cannot resolve in place. We deepcopy only the
        small per-battle state (``_sides``/``_pending_switches``) — sharing the immutable, heavy
        ``set_source`` — and resolve the twin. Equivalent to a throwaway
        ``from_events`` engine's resolve+snapshot, but O(belief-state) instead of O(events).
        """
        twin = PublicBattleBeliefEngine(format_id=self.format_id, set_source=self.set_source)
        twin._event_count = self._event_count
        twin._sides = copy.deepcopy(self._sides)
        twin._pending_switches = copy.deepcopy(self._pending_switches)
        twin.resolve_pending_switches_at_boundary()
        return twin.snapshot().for_player(showdown_slot)

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
        try:
            summary = self.set_source.summarize(
                format_id=self.format_id,
                species=belief.species,
                revealed_moves=belief.revealed_moves,
                revealed_ability=belief.revealed_ability,
                revealed_item=belief.revealed_item,
                ruled_out_abilities=belief.ruled_out_abilities,
            )
        except TypeError:
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
            possible_abilities=summary.possible_abilities,
            possible_items=summary.possible_items,
            possible_moves=summary.possible_moves,
            candidate_variants=summary.candidate_variants,
            source_metadata=summary.source_metadata,
        )

    def _record_raw_ability_reveal(self, event: Any) -> None:
        event_type = _event_value(event, "event_type")
        if event_type in {"-ability", "ability"}:
            return
        raw_line = _event_value(event, "raw_line")
        ability_ident, ability_name = _confirmed_ability_from_event(event)
        ability_slot = _slot_from_ident(ability_ident)
        if not ability_slot or not ability_name:
            return
        belief = self._target_belief(ability_slot, ability_ident)
        if belief is None or _normalize_identifier(belief.revealed_ability or "") == _normalize_identifier(ability_name):
            return
        self._replace_belief(
            belief,
            revealed_ability=ability_name,
            evidence=_append_evidence(
                belief.evidence,
                BeliefEvidence(
                    kind="confirmed-ability",
                    detail=f"Confirmed ability {ability_name} from public protocol effect.",
                    source_line=raw_line,
                ),
            ),
        )

    def _resolve_pending_switches_for_event(self, event: Any) -> None:
        if not self._pending_switches:
            return
        raw_line = _event_value(event, "raw_line")
        ability_ident, ability_name = _confirmed_ability_from_event(event)
        if ability_name:
            remaining: list[_PendingSwitch] = []
            for pending in self._pending_switches:
                if _ident_matches_pending(ability_ident, pending) and _normalize_identifier(ability_name) == "intimidate":
                    belief = self._find_belief(pending.showdown_slot, pending.species)
                    if belief is not None:
                        self._replace_belief(
                            belief,
                            revealed_ability="Intimidate",
                            evidence=_append_evidence(
                                belief.evidence,
                                BeliefEvidence(
                                    kind="confirmed-ability",
                                    detail="Confirmed Intimidate from switch-in trigger.",
                                    source_line=raw_line,
                                ),
                            ),
                        )
                else:
                    remaining.append(pending)
            self._pending_switches = remaining
            return
        if _is_pending_switch_boundary(event):
            self._resolve_pending_switches_as_no_trigger(raw_line)

    def _resolve_pending_switches_as_no_trigger(self, source_line: Optional[str]) -> None:
        pending_switches = self._pending_switches
        self._pending_switches = []
        for pending in pending_switches:
            belief = self._find_belief(pending.showdown_slot, pending.species)
            if belief is None or not self._can_rule_out_intimidate(belief):
                continue
            self._replace_belief(
                belief,
                ruled_out_abilities=_append_unique(belief.ruled_out_abilities, "Intimidate"),
                evidence=_append_evidence(
                    belief.evidence,
                    BeliefEvidence(
                        kind="ruled-out-ability",
                        detail="No public Intimidate trigger occurred on switch-in, so Intimidate was ruled out.",
                        source_line=source_line,
                    ),
                ),
            )

    def _can_queue_intimidate_non_trigger(self, belief: RevealedPokemonBelief) -> bool:
        abilities = {_normalize_identifier(ability) for ability in belief.possible_abilities}
        return "intimidate" in abilities and any(ability != "intimidate" for ability in abilities)

    def _can_rule_out_intimidate(self, belief: RevealedPokemonBelief) -> bool:
        abilities = {_normalize_identifier(ability) for ability in belief.possible_abilities}
        if "intimidate" not in abilities or not any(ability != "intimidate" for ability in abilities):
            return False
        other_active = self._active_belief(_opponent_slot(belief.showdown_slot))
        if other_active is None:
            return False
        blockers = {"clearbody", "hypercutter", "whitesmoke"}
        if other_active.revealed_ability and _normalize_identifier(other_active.revealed_ability) in blockers:
            return False
        if not other_active.revealed_ability:
            possible = {_normalize_identifier(ability) for ability in other_active.possible_abilities}
            if possible.intersection(blockers):
                return False
        return True

    def _find_belief(self, showdown_slot: str, species: str) -> RevealedPokemonBelief | None:
        normalized_species = _normalize_species(species)
        return next(
            (
                pokemon
                for pokemon in self._sides.get(showdown_slot, [])
                if _normalize_species(pokemon.species) == normalized_species
            ),
            None,
        )


@dataclass(frozen=True)
class _PendingSwitch:
    showdown_slot: str
    ident: Optional[str]
    species: str


def belief_key(showdown_slot: str, species: str) -> str:
    return f"{showdown_slot}:{_normalize_species(species)}"


def _deterministic_variant_selection(
    choices_by_pokemon: tuple[tuple[Mapping[str, Any], ...], ...],
    sample_index: int,
) -> tuple[Mapping[str, Any] | None, ...]:
    selected: list[Mapping[str, Any] | None] = []
    radix = 1
    for choices in choices_by_pokemon:
        if not choices:
            selected.append(None)
            continue
        selected.append(choices[(sample_index // radix) % len(choices)])
        radix *= len(choices)
    return tuple(selected)


def _determinized_pokemon(
    pokemon: RevealedPokemonBelief,
    variant: Mapping[str, Any] | None,
) -> DeterminizedOpponentPokemon:
    if variant is None:
        return DeterminizedOpponentPokemon(
            showdown_slot=pokemon.showdown_slot,
            species=pokemon.species,
            active=pokemon.active,
            condition=pokemon.condition,
            status=pokemon.status,
            revealed_moves=pokemon.revealed_moves,
            candidate_count=pokemon.candidate_set_count,
            uncertainty=pokemon.uncertainty,
            possible_abilities=pokemon.possible_abilities,
            possible_items=pokemon.possible_items,
            possible_moves=pokemon.possible_moves,
            source_metadata=pokemon.source_metadata,
        )
    return DeterminizedOpponentPokemon(
        showdown_slot=pokemon.showdown_slot,
        species=pokemon.species,
        active=pokemon.active,
        condition=pokemon.condition,
        status=pokemon.status,
        revealed_moves=pokemon.revealed_moves,
        variant_id=_optional_variant_string(variant.get("variant_id")),
        source_set_id=_optional_variant_string(variant.get("source_set_id")),
        role=_optional_variant_string(variant.get("role")),
        level=_optional_variant_int(variant.get("level")),
        moves=_variant_string_tuple(variant.get("moves")),
        ability=_optional_variant_string(variant.get("ability")),
        item=_optional_variant_string(variant.get("item")),
        candidate_count=pokemon.candidate_set_count,
        uncertainty=pokemon.uncertainty,
        possible_abilities=pokemon.possible_abilities,
        possible_items=pokemon.possible_items,
        possible_moves=pokemon.possible_moves,
        source_metadata=pokemon.source_metadata,
    )


def _variant_string_tuple(value: Any) -> tuple[str, ...]:
    if isinstance(value, str) or not isinstance(value, (list, tuple)):
        return ()
    return tuple(str(item) for item in value if str(item))


def _optional_variant_string(value: Any) -> Optional[str]:
    if value is None or value == "":
        return None
    return str(value)


def _optional_variant_int(value: Any) -> Optional[int]:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


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


def _append_evidence(
    values: tuple[BeliefEvidence, ...],
    evidence: BeliefEvidence,
) -> tuple[BeliefEvidence, ...]:
    signature = (evidence.kind, evidence.detail, evidence.source_line)
    if any((item.kind, item.detail, item.source_line) == signature for item in values):
        return values
    return (*values, evidence)


def _confirmed_ability_from_event(event: Any) -> tuple[Optional[str], Optional[str]]:
    event_type = _event_value(event, "event_type")
    target_ident = _event_value(event, "target_ident")
    primary = _event_value(event, "primary")
    raw_line = _event_value(event, "raw_line") or ""
    if event_type in {"-ability", "ability"} and primary:
        return target_ident, primary
    ability_match = re.search(r"\[from\] ability: ([^|\]]+)", raw_line)
    ident_match = re.search(r"\[of\] ([^|]+)", raw_line)
    if ability_match:
        return (
            ident_match.group(1).strip() if ident_match else target_ident,
            ability_match.group(1).strip(),
        )
    return None, None


def _ident_matches_pending(ident: Optional[str], pending: _PendingSwitch) -> bool:
    if pending.ident and ident:
        return _normalize_identifier(ident) == _normalize_identifier(pending.ident)
    if ident:
        return _normalize_identifier(_species_from_ident(ident) or "") == _normalize_identifier(pending.species)
    return False


def _is_pending_switch_boundary(event: Any) -> bool:
    event_type = _event_value(event, "event_type")
    if event_type in {"-boost", "-unboost", "-damage", "-heal", "-status", "-curestatus", "-item"}:
        return False
    return True


def _species_from_ident(ident: Optional[str]) -> Optional[str]:
    if not ident:
        return None
    species = str(ident).split(":", 1)[-1].strip()
    return species or None


def _slot_from_ident(ident: Optional[str]) -> Optional[str]:
    if not ident:
        return None
    match = re.match(r"^(p[12])", str(ident))
    return match.group(1) if match else None


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
