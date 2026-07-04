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
    # True when the reveals matched no known set (Showdown randbats drift, an unfiltered called or
    # copied move, ...) and we fell back to the unconstrained species pool. The state is "maximally
    # uncertain", NOT "certain": uncertainty is forced to 1.0 in that case.
    inconsistent: bool = False


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
    # Transform (Ditto): while transformed the mon fights as ``transform_species`` — its stats,
    # types and moves are the copied target's, NOT its own set. Consumers should read stats/types
    # from ``transform_species`` and must not treat moves used while transformed as its real set.
    transformed: bool = False
    transform_species: Optional[str] = None
    # Exact-state ledger (observation_compression_design.md, exact-state class).
    # ``move_uses`` counts PP charged per revealed move id (Pressure double-charges included;
    # Sleep-Talk-called moves charge the caller; moves used while transformed charge nothing).
    move_uses: tuple[tuple[str, int], ...] = ()
    # Sleep bookkeeping: observed |cant …|slp turns since the status landed; ``rest_sleep``
    # marks Rest self-sleep (wake deterministic modulo Early Bird candidates).
    sleep_turns: int = 0
    rest_sleep: bool = False
    # Turns this mon has been active in its current stint (reset on entry).
    turns_active: int = 0
    # Deterministic non-proc pruning results (Leftovers / Lum / pinch berries). Frozen once the
    # held item is mutated (Trick / Knock Off): pruning applies to the original assignment only.
    ruled_out_items: tuple[str, ...] = ()
    item_mutated: bool = False
    # Natural Cure detection: status carried out on switch + the side's cure-all (Heal Bell /
    # Aromatherapy) counter at exit; a clean re-entry with an unchanged counter confirms.
    status_on_exit: Optional[str] = None
    cure_all_count_on_exit: int = -1

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
            "transformed": self.transformed,
            "transform_species": self.transform_species,
            "move_uses": [list(pair) for pair in self.move_uses],
            "sleep_turns": self.sleep_turns,
            "rest_sleep": self.rest_sleep,
            "turns_active": self.turns_active,
            "ruled_out_items": list(self.ruled_out_items),
            "item_mutated": self.item_mutated,
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
        # Exact-state engine bookkeeping (all protocol-tautological; no mechanics model).
        self._turn_number = 0
        self._cure_all_count: dict[str, int] = {"p1": 0, "p2": 0}
        # Sleep Clause Mod (live semantics): the belief key of the opposing mon this side put to
        # sleep, cleared on its wake or faint. Rest self-sleep never engages the clause.
        self._sleep_clause_holder: dict[str, Optional[str]] = {"p1": None, "p2": None}
        # Per-turn proc tracking for the non-proc pruning family.
        self._leftovers_healed_this_turn: set[str] = set()
        self._berry_ate_this_turn: set[str] = set()
        # Pending Mud Shot Shield-Dust check: (target_key, saw_damage, cancelled).
        self._pending_mudshot: Optional[dict[str, Any]] = None

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
        self._record_item_reveal(event)

        if event_type in {"switch", "drag", "replace"} and actor_slot and primary:
            self._record_switch_out_state(actor_slot)
            self._mark_side_inactive(actor_slot)
            belief = self._upsert(
                showdown_slot=actor_slot,
                species=str(primary),
                condition=_string_or_none(secondary),
                active=True,
            )
            belief = self._on_switch_in(belief, condition=_string_or_none(secondary), raw_line=raw_line)
            if self._can_queue_intimidate_non_trigger(belief):
                self._pending_switches.append(
                    _PendingSwitch(
                        showdown_slot=actor_slot,
                        ident=actor_ident,
                        species=belief.species,
                    )
                )
            return

        if event_type == "-transform" and actor_slot and primary:
            # ``|-transform|p1a: Ditto|p2a: Blissey`` — the actor now fights as the target. Record
            # the copied identity so consumers read stats/types from it; moves used while
            # transformed are suppressed below (they are the target's, not the actor's set).
            species = self._active_species(actor_slot) or _species_from_ident(actor_ident)
            target_species = _species_from_ident(primary)
            if species and target_species:
                belief = self._upsert(showdown_slot=actor_slot, species=species)
                self._replace_belief(
                    belief,
                    transformed=True,
                    transform_species=target_species,
                    evidence=_append_evidence(
                        belief.evidence,
                        BeliefEvidence(
                            kind="transform",
                            detail=f"Transformed into {target_species}; copied moves are not its set.",
                            source_line=raw_line,
                        ),
                    ),
                )
            return

        if event_type == "move" and actor_slot and primary:
            self._resolve_pending_mudshot()
            species = self._active_species(actor_slot) or _species_from_ident(actor_ident)
            if species:
                belief = self._upsert(showdown_slot=actor_slot, species=species)
                move_id = _normalize_identifier(str(primary))
                caller = _called_move_source(raw_line)
                if move_id in {"healbell", "aromatherapy"}:
                    self._cure_all_count[actor_slot] = self._cure_all_count.get(actor_slot, 0) + 1
                if move_id == "mudshot":
                    target_belief = self._active_belief(_other_side(actor_slot))
                    if target_belief is not None:
                        self._pending_mudshot = {
                            "target_key": target_belief.key,
                            "target_side": _other_side(actor_slot),
                            "saw_damage": False,
                            "cancelled": False,
                        }
                # PP ledger: called moves charge the CALLER's PP (they spend none of their own);
                # transformed mons charge nothing (copied moves are instance-scoped, 5 PP,
                # discarded on switch-out — never the real set's ledger); Struggle has no PP.
                if belief.transformed:
                    return
                if caller in _CALLER_MOVES:
                    # The called execution spends no PP of its own; the caller was already
                    # charged on its own |move| line (Showdown always emits it first).
                    return
                if move_id != "struggle":
                    belief = self._charge_move_use(belief, move_id)
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
                if event_type == "-heal" and raw_line and "[from] item: Leftovers" in raw_line:
                    self._leftovers_healed_this_turn.add(belief.key)
                if (
                    event_type == "-damage"
                    and self._pending_mudshot is not None
                    and not self._pending_mudshot.get("cancelled")
                    and belief.key == self._pending_mudshot.get("target_key")
                    and not (raw_line and "[from]" in raw_line)
                ):
                    self._pending_mudshot["saw_damage"] = True
                self._replace_belief(belief, condition=_string_or_none(primary))
            return

        if event_type == "-unboost" and target_slot:
            if (
                self._pending_mudshot is not None
                and not self._pending_mudshot.get("cancelled")
            ):
                belief = self._target_belief(target_slot, target_ident)
                if belief is not None and belief.key == self._pending_mudshot.get("target_key"):
                    self._pending_mudshot["cancelled"] = True
            return

        if event_type == "-status" and target_slot:
            belief = self._target_belief(target_slot, target_ident)
            if belief is not None:
                rest = bool(raw_line and "move: Rest" in raw_line)
                status_value = _string_or_none(primary)
                changes: dict[str, Any] = {"status": status_value}
                if status_value == "slp":
                    changes["sleep_turns"] = 0
                    changes["rest_sleep"] = rest
                    # Sleep Clause Mod engages only for opponent-inflicted sleep (never Rest,
                    # never Synchronize-style reflections, which carry a [from] ability tag).
                    if not rest and not (raw_line and "[from] ability:" in raw_line):
                        self._sleep_clause_holder[_other_side(target_slot)] = belief_key(
                            belief.showdown_slot, belief.species
                        )
                self._replace_belief(belief, **changes)
            return

        if event_type == "-curestatus" and target_slot:
            belief = self._target_belief(target_slot, target_ident)
            if belief is not None:
                if belief.status == "slp" and belief.rest_sleep and belief.sleep_turns == 1:
                    # Rest sleeps exactly 2 turns in gen 3; a 1-turn Rest wake is deterministic
                    # Early Bird identification (5 reachable carriers).
                    if not belief.revealed_ability:
                        belief = self._replace_belief(
                            belief,
                            revealed_ability="Early Bird",
                            evidence=_append_evidence(
                                belief.evidence,
                                BeliefEvidence(
                                    kind="confirmed-ability",
                                    detail="Woke from Rest after 1 turn; only Early Bird halves Rest sleep.",
                                    source_line=raw_line,
                                ),
                            ),
                        )
                self._clear_sleep_clause_for(belief)
                self._replace_belief(belief, status=None, sleep_turns=0, rest_sleep=False)
            return

        if event_type == "cant" and raw_line and "|slp" in raw_line:
            # ``|cant|p2a: Snorlax|slp`` — the parser does not decompose cant lines, so read the
            # ident from the raw line. Each observed sleeping turn ticks the counter.
            parts = raw_line.split("|")
            cant_ident = parts[2] if len(parts) > 2 else None
            cant_slot = _slot_from_ident(cant_ident)
            if cant_slot:
                species = self._active_species(cant_slot) or _species_from_ident(cant_ident)
                if species:
                    belief = self._upsert(showdown_slot=cant_slot, species=species)
                    self._replace_belief(belief, sleep_turns=belief.sleep_turns + 1)
            return

        if event_type == "faint" and target_slot:
            belief = self._target_belief(target_slot, target_ident)
            if belief is not None:
                self._clear_sleep_clause_for(belief)
                self._replace_belief(
                    belief, condition="0 fnt", active=False, status_on_exit=None, cure_all_count_on_exit=-1
                )
            return

        if event_type == "turn":
            self._turn_number += 1
            for side in ("p1", "p2"):
                active = self._active_belief(side)
                if active is not None:
                    self._replace_belief(active, turns_active=active.turns_active + 1)
            return

        if event_type == "upkeep":
            # The |upkeep| line follows all residuals (Leftovers heals, pinch-berry eats), so
            # end-of-turn non-proc pruning runs here with this turn's proc sets fully populated.
            self._sweep_end_of_turn_non_procs()
            self._resolve_pending_mudshot()
            self._leftovers_healed_this_turn = set()
            self._berry_ate_this_turn = set()
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
                changes: dict[str, Any] = {
                    "revealed_item": str(primary),
                    "evidence": _append_evidence(
                        belief.evidence,
                        BeliefEvidence(
                            kind="revealed-item",
                            detail=f"Observed item {primary}; incompatible set variants were removed.",
                            source_line=raw_line,
                        ),
                    ),
                }
                if raw_line and "move: Trick" in raw_line:
                    changes["item_mutated"] = True
                self._replace_belief(belief, **changes)

    def _record_item_reveal(self, event: Any) -> None:
        """Record an item reveal that the explicit ``-item`` branch misses.

        Items are revealed three ways in the protocol: ``|-item|`` (Frisk/Trick/Trace — handled in
        ingest_event), ``|-enditem|`` (a berry is eaten, or the item is knocked off / consumed), and
        inline ``[from] item: X`` tags on other events (``|-heal|...|[from] item: Leftovers``,
        ``|-damage|...|[from] item: Life Orb``). The last two are how the most common Gen 3 items
        (Leftovers, Life Orb, berries) actually surface, so without this they never register."""
        event_type = _event_value(event, "event_type")
        raw_line = _event_value(event, "raw_line") or ""
        primary = _event_value(event, "primary")

        item: Optional[str] = None
        if event_type == "-enditem" and primary:
            item = primary  # the ended/consumed/removed item names itself
        else:
            marker = "[from] item:"
            if marker in raw_line:
                item = raw_line.split(marker, 1)[1].split("|")[0].strip()
        if not item:
            return

        # The item belongs to the mon the effect applies to (target), else the acting mon. In Gen 3
        # every "[from] item:" surface (Leftovers -heal, Life Orb -damage, -enditem berries/Knock
        # Off) owns to that mon, so the "[of]" tag is deliberately not consulted; revisit if a later
        # gen introduces items whose "[from]" effect owns to the "[of]" mon.
        slot = _event_value(event, "target_slot") or _event_value(event, "actor_slot")
        ident = _event_value(event, "target_ident") or _event_value(event, "actor_ident")
        if not slot:
            return
        belief = self._target_belief(slot, ident)
        if belief is None:
            return
        if event_type == "-enditem":
            if "[eat]" in raw_line:
                self._berry_ate_this_turn.add(belief.key)
            if "move: Knock Off" in raw_line or "move: Trick" in raw_line:
                # Held-item mutation: non-proc pruning applies to the ORIGINAL assignment only.
                belief = self._replace_belief(belief, item_mutated=True)
        if _normalize_identifier(belief.revealed_item or "") == _normalize_identifier(item):
            return  # already known
        self._replace_belief(
            belief,
            revealed_item=item,
            evidence=_append_evidence(
                belief.evidence,
                BeliefEvidence(
                    kind="revealed-item",
                    detail=f"Observed item {item} via {event_type}; incompatible set variants were removed.",
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
        twin._turn_number = self._turn_number
        twin._cure_all_count = dict(self._cure_all_count)
        twin._sleep_clause_holder = dict(self._sleep_clause_holder)
        twin._leftovers_healed_this_turn = set(self._leftovers_healed_this_turn)
        twin._berry_ate_this_turn = set(self._berry_ate_this_turn)
        twin._pending_mudshot = copy.deepcopy(self._pending_mudshot)
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
        # Leaving the field ends Transform: the mon reverts to itself, so clear the copied identity
        # (and the known-stats it implied). A fainted mon keeps its last state — we stop caring once
        # it is KO'd.
        self._sides[showdown_slot] = [
            replace(pokemon, active=False, transformed=False, transform_species=None)
            if pokemon.transformed
            else replace(pokemon, active=False)
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

    @property
    def sleep_clause_holders(self) -> dict[str, Optional[str]]:
        """Per side: the belief key of the opposing mon this side currently has asleep (live)."""
        return dict(self._sleep_clause_holder)

    @property
    def turn_number(self) -> int:
        return self._turn_number

    def _charge_move_use(self, belief: RevealedPokemonBelief, move_id: str) -> RevealedPokemonBelief:
        # Pressure on the OPPOSING active doubles the PP spent; gen 3 announces Pressure on
        # entry, so the opposing ability is public whenever the double applies.
        opposing = self._active_belief(_other_side(belief.showdown_slot))
        charge = 2 if opposing is not None and _normalize_identifier(opposing.revealed_ability or "") == "pressure" else 1
        normalized = _normalize_identifier(move_id)
        uses = dict(belief.move_uses)
        uses[normalized] = uses.get(normalized, 0) + charge
        return self._replace_belief(belief, move_uses=tuple(sorted(uses.items())))

    def _record_switch_out_state(self, showdown_slot: str) -> None:
        outgoing = self._active_belief(showdown_slot)
        if outgoing is None or outgoing.condition == "0 fnt":
            return
        self._replace_belief(
            outgoing,
            status_on_exit=outgoing.status,
            cure_all_count_on_exit=self._cure_all_count.get(showdown_slot, 0),
        )

    def _on_switch_in(
        self,
        belief: RevealedPokemonBelief,
        *,
        condition: Optional[str],
        raw_line: Optional[str],
    ) -> RevealedPokemonBelief:
        changes: dict[str, Any] = {"turns_active": 0}
        condition_status = _status_token_from_condition(condition)
        if belief.status_on_exit and condition_status is None:
            # Natural Cure elimination: carried a status out, returned clean, and no public
            # cure-all (Heal Bell / Aromatherapy) happened in between. All cure paths in this
            # pool are public events, so this is deterministic identification.
            if belief.cure_all_count_on_exit == self._cure_all_count.get(belief.showdown_slot, 0):
                if not belief.revealed_ability:
                    changes["revealed_ability"] = "Natural Cure"
                    changes["evidence"] = _append_evidence(
                        belief.evidence,
                        BeliefEvidence(
                            kind="confirmed-ability",
                            detail="Returned status-free with no public cure-all between exits; only Natural Cure explains it.",
                            source_line=raw_line,
                        ),
                    )
            changes["status"] = None
            changes["sleep_turns"] = 0
            changes["rest_sleep"] = False
        elif condition_status is not None:
            changes["status"] = condition_status
        changes["status_on_exit"] = None
        changes["cure_all_count_on_exit"] = -1
        return self._replace_belief(belief, **changes)

    def _clear_sleep_clause_for(self, belief: RevealedPokemonBelief) -> None:
        for side, holder in list(self._sleep_clause_holder.items()):
            if holder == belief.key:
                self._sleep_clause_holder[side] = None

    def _rule_out_items(
        self,
        belief: RevealedPokemonBelief,
        items: tuple[str, ...],
        detail: str,
    ) -> RevealedPokemonBelief:
        new_items = tuple(item for item in items if item not in belief.ruled_out_items)
        if not new_items:
            return belief
        return self._replace_belief(
            belief,
            ruled_out_items=belief.ruled_out_items + new_items,
            evidence=_append_evidence(
                belief.evidence,
                BeliefEvidence(kind="ruled-out-item", detail=detail, source_line=None),
            ),
        )

    def _sweep_end_of_turn_non_procs(self) -> None:
        for side in ("p1", "p2"):
            belief = self._active_belief(side)
            if belief is None or belief.item_mutated or belief.revealed_item:
                continue
            hp_fraction = _hp_fraction_from_condition(belief.condition)
            if hp_fraction is None or hp_fraction <= 0.0:
                continue
            if hp_fraction < 1.0 and belief.key not in self._leftovers_healed_this_turn:
                belief = self._rule_out_items(
                    belief,
                    ("leftovers",),
                    "Ended a damaged turn with no Leftovers heal; Leftovers variants removed.",
                )
            if belief.status:
                belief = self._rule_out_items(
                    belief,
                    ("lumberry",),
                    "Status persisted without an instant Lum cure; Lum variants removed.",
                )
            if hp_fraction <= 0.25 and belief.key not in self._berry_ate_this_turn:
                belief = self._rule_out_items(
                    belief,
                    ("salacberry", "petayaberry", "liechiberry"),
                    "Ended a turn at or below 25% HP with no pinch-berry activation; pinch variants removed.",
                )

    def _resolve_pending_mudshot(self) -> None:
        pending = self._pending_mudshot
        self._pending_mudshot = None
        if not pending or pending.get("cancelled") or not pending.get("saw_damage"):
            return
        side = pending.get("target_side")
        for belief in self._sides.get(str(side), []):
            if belief.key != pending.get("target_key"):
                continue
            # Mud Shot is the pool's only 100% target secondary: damage landed (not on a sub —
            # sub hits report no plain -damage) with no spe drop and no tagged blocker ⇒ the
            # only remaining explanation is Shield Dust. Conservative: candidates must allow it.
            if belief.revealed_ability:
                return
            candidates = {_normalize_identifier(a) for a in belief.possible_abilities}
            if candidates and "shielddust" not in candidates:
                return
            self._replace_belief(
                belief,
                revealed_ability="Shield Dust",
                evidence=_append_evidence(
                    belief.evidence,
                    BeliefEvidence(
                        kind="confirmed-ability",
                        detail="Mud Shot's guaranteed Speed drop did not fire on a clean hit; only Shield Dust explains it.",
                        source_line=None,
                    ),
                ),
            )
            return

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
                ruled_out_items=belief.ruled_out_items if not belief.item_mutated else (),
            )
        except TypeError:
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


# Moves that invoke ANOTHER move. The invoked move is not part of the caller's own set, so it must
# not be recorded as a revealed move (e.g. Metronome -> Fissure, Sleep Talk -> Spore).
# (copycat is Gen 4+; harmless to list for a Gen 3 engine — it just never matches.)
_CALLER_MOVES = frozenset(
    {"metronome", "mirrormove", "sleeptalk", "assist", "naturepower", "copycat"}
)


def _other_side(showdown_slot: str) -> str:
    return "p2" if showdown_slot == "p1" else "p1"


def _hp_fraction_from_condition(condition: Optional[str]) -> Optional[float]:
    """Parse an HP fraction from a protocol condition string ('155/307 par', '0 fnt')."""
    if not condition:
        return None
    head = condition.split()[0]
    if head == "0" or "fnt" in condition:
        return 0.0
    if "/" not in head:
        return None
    try:
        current, maximum = head.split("/", 1)
        maximum_value = float(maximum)
        if maximum_value <= 0:
            return None
        return max(0.0, min(1.0, float(current) / maximum_value))
    except ValueError:
        return None


def _status_token_from_condition(condition: Optional[str]) -> Optional[str]:
    """Status token from a condition string ('250/250 slp' -> 'slp'), None when healthy."""
    if not condition:
        return None
    parts = condition.split()
    if len(parts) < 2 or parts[1] == "fnt":
        return None
    return parts[1]


def _called_move_source(raw_line: Optional[str]) -> Optional[str]:
    """Normalized caller move if a ``|move|`` line was invoked by another move, else None.

    Handles both protocol forms — ``[from]move: Sleep Talk`` and the bare ``[from] Sleep Talk`` —
    and deliberately does NOT match ``[from]lockedmove`` (Thrash/Outrage continuations ARE the
    mon's own move) or other non-caller effects."""
    if not raw_line:
        return None
    marker = raw_line.find("[from]")
    if marker == -1:
        return None
    tag = raw_line[marker + len("[from]"):].split("|")[0].strip()
    lowered = tag.lower()
    if lowered.startswith("move:"):
        tag = tag[len("move:"):].strip()
    return _normalize_identifier(tag)


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
