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
    gender: Optional[str] = None
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
    # gen3 sleep ``skippedTime``: the count of TRAILING contiguous Sleep-Talk/Snore (``sleepUsable``)
    # turns immediately before this mon last switched out. Those turns did not advance the wake
    # timer once the mon returns — gen3 refunds them on switch-in (``time += skippedTime``) — so
    # ``_on_switch_in`` subtracts this from ``sleep_turns``. Resets to 0 on a non-sleepUsable sleep
    # turn (matching the sim) and on switch-in (once consumed).
    sleep_skipped_turns: int = 0
    # Turns this mon has been active in its current stint (reset on entry).
    turns_active: int = 0
    # Deterministic non-proc pruning results (Leftovers / Lum / pinch berries / Choice Band).
    # The first three are end-of-turn NON-PROCS — an opportunity the item had and did not take;
    # Choice Band is the same shape read off the move stream instead, since two different moves
    # in one stay are an opportunity the ``choicelock`` volatile would have denied. Frozen once
    # the held item is mutated (Trick / Knock Off): pruning applies to the original assignment.
    ruled_out_items: tuple[str, ...] = ()
    item_mutated: bool = False
    # Removal vs swap, the distinction ``item_mutated`` alone cannot carry: True while the
    # mon's CURRENT public item state is "holds nothing" — the last mutation stripped it
    # (Knock Off, or a Trick that took the item and returned none) OR the mon publicly
    # consumed it (a berry ``[eat]``, White Herb, ...; consumption does NOT set
    # ``item_mutated``: the consumed item still identifies the original assignment).
    # "Holds nothing" is exactly representable in a determinized world (clear the sampled
    # item); consumers treat any ``item_removed`` mon that way regardless of mutation.
    item_removed: bool = False
    # The mon's CURRENT held item as positively named by the protocol line that PUT it
    # there — gen3: Trick's ``|-item|SLOT|ITEM|[from] move: Trick`` announcement (probed
    # verbatim; the holder is the line's own target). Set ONLY by that audited surface,
    # cleared by any subsequent ``-enditem`` on the mon. While ``item_mutated`` is True
    # and ``item_removed`` is False, a non-None value lets determinized worlds substitute
    # the revealed current item instead of failing closed; None in that state means the
    # current item is NOT protocol-confirmed (unexpected mutation source) — fail closed.
    # NOTE: do not read ``revealed_item`` for this purpose — its post-mutation semantics
    # are surface-dependent (a Knock Off ``-enditem`` leaves it naming the REMOVED item).
    current_public_item: Optional[str] = None
    # The mon's ORIGINAL held item — the randbats generator's own assignment — once the protocol
    # has named it with CERTAINTY, even though the mon no longer holds it. This is the fact
    # ``item_mutated`` alone cannot carry: "what it holds NOW is not the assignment" and "what it
    # WAS holding is now known" are different statements, and only the first suppresses variant
    # matching. ``revealed_item`` cannot express the second either, because its post-mutation
    # meaning is surface-dependent (Knock Off leaves it naming the removed ORIGINAL, Trick's
    # ``-item`` leaves it naming the RECEIVED item, which is somebody else's assignment).
    #
    # Two audited surfaces, both verified against the vendored engine:
    #   * ``|-enditem|SLOT|ITEM|[from] move: Knock Off|[of] ATTACKER`` (``data/moves.ts``
    #     knockoff.onAfterHit) and Trick's silent ``|-enditem|SLOT|ITEM|[silent]|[from] move:
    #     Trick`` — the subject's OWN item;
    #   * Trick's ``|-item|SLOT|ITEM|[from] move: Trick``, which is CROSS-ATTRIBUTED: trick.onHit
    #     gives the SOURCE's item to the TARGET and the TARGET's to the SOURCE, so the item named
    #     on each of the two lines is the PARTNER's assignment (see ``_PendingTrick``).
    # Written only while the mon's held item was still its own assignment (a mon already carrying
    # a Tricked item names that item, not the generator's), and never overwritten once set.
    original_public_item: Optional[str] = None
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
            "gender": self.gender,
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
            "sleep_skipped_turns": self.sleep_skipped_turns,
            "turns_active": self.turns_active,
            "ruled_out_items": list(self.ruled_out_items),
            "item_mutated": self.item_mutated,
            "item_removed": self.item_removed,
            "current_public_item": self.current_public_item,
            "original_public_item": self.original_public_item,
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
    gender: Optional[str] = None
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
            "gender": self.gender,
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


def variant_identity(variant: Mapping[str, Any]) -> tuple[Any, ...]:
    """Stable identity of one candidate-variant mapping.

    Used to intersect an evidence producer's SURVIVOR list against the engine's own
    candidate list (:meth:`PublicBattleBeliefEngine.narrow_candidate_variants`). Both
    sides read the SAME mappings out of the same ``CandidateSetSummary``, so the fields
    are taken RAW: normalizing here would add a divergence risk (two normalizers drifting
    apart silently drops every pin) without adding any discrimination between variants
    that the generator already emits distinctly.
    """

    raw_moves = variant.get("moves")
    moves = tuple(str(move) for move in raw_moves) if isinstance(raw_moves, (list, tuple)) else ()
    level = variant.get("level")
    return (
        int(level) if isinstance(level, int) and level > 0 else 0,
        moves,
        str(variant.get("item") or ""),
        str(variant.get("ability") or ""),
    )


def _variant_field_values(
    variants: Sequence[Mapping[str, Any]], field_name: str, *, plural: bool
) -> set[str]:
    values: set[str] = set()
    for variant in variants:
        raw = variant.get(field_name)
        if plural:
            if isinstance(raw, (list, tuple)):
                values.update(str(item) for item in raw)
        elif raw:
            values.add(str(raw))
    return values


def _narrowed_possible_values(
    values: tuple[str, ...],
    variants: Sequence[Mapping[str, Any]],
    kept: Sequence[Mapping[str, Any]],
    field_name: str,
    *,
    plural: bool,
) -> tuple[str, ...]:
    """Drop ``possible_*`` entries no SURVIVING variant carries. Filter, never recompute.

    The randbats source builds ``possible_moves``/``_items``/``_abilities`` as exactly this
    projection over the surviving variant list (``_stable_unique`` over the same fields
    ``Gen3RandbatVariant.to_summary`` writes into ``candidate_variants``), so filtering the
    already-emitted tuple reproduces what the source itself would have emitted for the
    narrowed set — while preserving ITS ordering and ITS string forms. Recomputing here
    would introduce a second normalizer that could silently drift.

    Guarded for sources that build these surfaces some other way: if any emitted value is
    absent from the projection over ALL variants, the two are not the same vocabulary, and
    the surface is returned untouched rather than filtered down to something wrong. A filter
    that would empty the surface is also declined — same refusal asymmetry as the pin itself.
    """

    if not values:
        return values
    if not set(values) <= _variant_field_values(variants, field_name, plural=plural):
        return values
    survivors = _variant_field_values(kept, field_name, plural=plural)
    narrowed = tuple(value for value in values if value in survivors)
    return narrowed or values


class PublicBattleBeliefEngine:
    def __init__(
        self,
        *,
        format_id: Optional[str] = None,
        set_source: PokemonSetSource | None = None,
        item_belief_narrowing: bool = False,
    ) -> None:
        self.format_id = format_id
        self.set_source = set_source
        # ObservationFeatureMasks.item_belief_narrowing, carried down by the env. When True the
        # protocol-CERTAIN item facts this engine records are allowed to narrow candidate sets:
        # the ORIGINAL assignment named by a Knock Off / Trick (``original_public_item``) becomes
        # a variant-matching key. Default False because narrowing moves NUMERIC_CANDIDATE_SET_COUNT
        # and NUMERIC_UNCERTAINTY, which exist in EVERY schema — turning it on shifts the input
        # distribution of every checkpoint ever trained, so it must be opted into by a fresh arm.
        self.item_belief_narrowing = bool(item_belief_narrowing)
        self._event_count = 0
        self._sides: dict[str, list[RevealedPokemonBelief]] = {"p1": [], "p2": []}
        self._pending_switches: list[_PendingSwitch] = []
        # Exact-state engine bookkeeping (all protocol-tautological; no mechanics model).
        self._turn_number = 0
        self._cure_all_count: dict[str, int] = {"p1": 0, "p2": 0}
        # Sleep Clause Mod (live semantics): the belief key of the opposing mon this side put to
        # sleep, cleared on its wake or faint. Rest self-sleep never engages the clause.
        self._sleep_clause_holder: dict[str, Optional[str]] = {"p1": None, "p2": None}
        # Per-turn proc tracking for the non-proc pruning family. Leftovers pruning keys off the
        # PRE-RESIDUAL damage state: gen3's Leftovers slot precedes status/Leech chip, so a mon
        # chipped only during residuals gives no Leftovers evidence (its slot ran at full HP).
        self._leftovers_healed_this_turn: set[str] = set()
        self._berry_ate_this_turn: set[str] = set()
        # Belief keys whose Shed Skin ``-activate`` fired this turn. A Shed Skin
        # carrier that Rests can proc its 33% cure on the first upkeep and wake in
        # exactly 1 turn — indistinguishable by sleep-count from an Early Bird
        # Rest wake — so this set suppresses the false Early Bird pin (Fix C). The
        # ``-activate`` precedes its ``-curestatus`` in the same residual phase.
        self._shed_skin_activated_this_turn: set[str] = set()
        # Belief keys with an unresolved ``|cant|…|slp`` this turn — awaiting a following
        # ``sleepUsable`` (Sleep Talk / Snore) move that would mark the turn as ``skippedTime``.
        # Any key still unresolved at ``|upkeep`` was a plain sleep turn, which resets skip to 0.
        self._sleep_cant_pending: set[str] = set()
        self._hp_after_actions: dict[str, Optional[float]] = {}
        # Choice-lock bookkeeping: belief key -> the FIRST freely selected move of the mon's
        # current stay on the field, which is the move a Choice Band would have locked it into.
        # Cleared on switch-in. See ``_note_choice_lock_selection``.
        self._stay_locked_move: dict[str, str] = {}
        # Pending Mud Shot Shield-Dust check: (target_key, saw_damage, cancelled).
        self._pending_mudshot: Optional[dict[str, Any]] = None
        # The Trick pairing published by the ``-activate`` line, consumed by the two ``-item``
        # lines that follow it in the same protocol chunk (see ``_PendingTrick``). Cleared by the
        # first event that is neither ``-item`` nor ``-enditem``, so it can never span two Tricks.
        self._pending_trick: Optional[_PendingTrick] = None
        # Variant narrowing from EXTERNAL damage-evidence producers (defender-side investment
        # inference, pokezero.investment). Belief key -> the identities still viable. Monotone:
        # each call intersects, never widens, so a narrowing can only be undone by rebuilding
        # the engine. Kept as identities rather than mappings so the filter survives the
        # set source re-summarizing on every reveal.
        #
        # This module stays a stdlib LEAF: computing which variants survive needs the dex and
        # the gen3 spread/damage core, so the producer computes survivors and hands them in
        # rather than belief.py importing that stack.
        self._variant_pins: dict[str, frozenset[tuple[Any, ...]]] = {}
        # Diagnostic only: keys where an incoming survivor list contradicted the standing pin
        # (empty intersection). The pin is LEFT INTACT and the contradiction counted — a false
        # narrowing that eliminates the true variant is unrecoverable and corrupts every
        # downstream derived feature and search world, so the conflict path never widens.
        self._variant_pin_conflicts: dict[str, int] = {}

    @classmethod
    def from_events(
        cls,
        events: Sequence[Any],
        *,
        format_id: Optional[str] = None,
        set_source: PokemonSetSource | None = None,
        item_belief_narrowing: bool = False,
    ) -> "PublicBattleBeliefEngine":
        engine = cls(
            format_id=format_id,
            set_source=set_source,
            item_belief_narrowing=item_belief_narrowing,
        )
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
        self._track_trick_pairing(event_type, raw_line)

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
                gender=_gender_from_switch_line(raw_line),
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
                # gen3 ``skippedTime``: a mon that ``|cant|…|slp``s this turn but still MOVES did
                # so via a ``sleepUsable`` move (Sleep Talk / Snore are the only ones that let a
                # sleeping mon act). That turn does not count toward waking once it pivots, so
                # accumulate it here; the following ``[from] Sleep Talk`` called move (caller set)
                # is not a fresh selection and must not double-count.
                if (
                    caller is None
                    and move_id in _SLEEP_USABLE_MOVES
                    and belief.key in self._sleep_cant_pending
                ):
                    self._sleep_cant_pending.discard(belief.key)
                    belief = self._replace_belief(
                        belief, sleep_skipped_turns=belief.sleep_skipped_turns + 1
                    )
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
                # Sleep Talk (unlike Metronome/Assist/Nature Power/...) can only call the mon's OWN
                # set members, so the executed callee is a GENUINE reveal even though it spends no PP
                # of its own. The other _CALLER_MOVES call random, non-set moves — those never reveal.
                sleep_talk_called = caller == "sleeptalk"
                if caller in _CALLER_MOVES and not sleep_talk_called:
                    # The called execution spends no PP of its own; the caller was already
                    # charged on its own |move| line (Showdown always emits it first).
                    return
                if caller == "lockedmove":
                    # Locked continuations (Solar Beam release) already paid on initiation.
                    # _called_move_source normalizes both the spaced and unspaced [from] forms.
                    return
                if move_id == "struggle":
                    # Struggle is a forced pseudo-move, never a set member (cf. showdown.py's
                    # exclusion from the determinized move list). Recording it as revealed makes the
                    # belief inconsistent with every real variant, collapsing the mon to the
                    # max-entropy fallback (full pool, uncertainty 1.0) and wiping a hard-won endgame
                    # read for the rest of the game. It spends none of the mon's own PP either.
                    return
                # Everything that reaches here without ``sleep_talk_called`` is a move the
                # PLAYER chose: called moves, locked continuations and Struggle have all
                # returned above. That is exactly the set the Choice lock constrains.
                if not sleep_talk_called:
                    belief = self._note_choice_lock_selection(belief, move_id)
                # A Sleep-Talk-called move spends none of the callee's PP (the Sleep Talk |move| line
                # already charged the caller); everything else that reaches here charges normally.
                if not sleep_talk_called:
                    foe_targeted = bool(target_slot) and target_slot != actor_slot
                    belief = self._charge_move_use(belief, move_id, foe_targeted=foe_targeted)
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
                if _is_action_phase_hp_change(raw_line):
                    self._hp_after_actions[belief.key] = _hp_fraction_from_condition(_string_or_none(primary))
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
                    # gen3 ``slp.onStart`` sets ``skippedTime = 0`` — a fresh sleep never inherits a
                    # prior stint's Sleep-Talk refund.
                    changes["sleep_skipped_turns"] = 0
                    # Sleep Clause Mod engages only for opponent-inflicted sleep (never Rest,
                    # never Synchronize-style reflections, which carry a [from] ability tag).
                    if not rest and not (raw_line and "[from] ability:" in raw_line):
                        self._sleep_clause_holder[_other_side(target_slot)] = belief_key(
                            belief.showdown_slot, belief.species
                        )
                self._replace_belief(belief, **changes)
            return

        if event_type == "-curestatus" and target_slot:
            # Heal Bell / Aromatherapy cure EVERY team member; a benched ally serializes without a
            # position letter (``|-curestatus|p2: Snorlax|par``), so ``_target_belief`` — which
            # returns the ACTIVE mon whenever one is present — would misattribute the cure to the
            # active mon and leave the benched ally's status stale. Resolve the benched ally by
            # species instead. Active-target cures keep their position letter and take the
            # unchanged shared path below (no change to active-target behavior).
            benched_target = target_ident is not None and not _ident_has_position(target_ident)
            if benched_target:
                belief = self._benched_target_belief(target_slot, target_ident)
            else:
                belief = self._target_belief(target_slot, target_ident)
            if belief is not None:
                # The 1-turn-Rest-wake Early Bird identification only applies to an ACTIVE mon
                # waking from Rest; a benched ally cleared by a team cure is not a Rest wake, and
                # its sleep_turns never ticked, so gate the pin to the active-target path.
                if (
                    not benched_target
                    and belief.status == "slp"
                    and belief.rest_sleep
                    and belief.sleep_turns == 1
                ):
                    # Rest sleeps exactly 2 turns in gen 3; a 1-turn Rest wake is deterministic
                    # Early Bird identification (5 reachable carriers) — EXCEPT a Shed Skin
                    # carrier that Rests and procs its 33% cure on the first upkeep also wakes in
                    # 1 turn. That mon's Shed Skin ``-activate`` fired this turn (recorded above),
                    # so it is not Early Bird — suppress the false pin (Fix C).
                    shed_skin_wake = belief.key in self._shed_skin_activated_this_turn
                    if not belief.revealed_ability and not shed_skin_wake:
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
                changes = self._status_cure_changes(belief)
                if benched_target:
                    # A benched cure (Heal Bell's per-mon ``[silent]`` -curestatus) publicly
                    # and fully accounts for this off-field mon's status change, so retire its
                    # switch-out status ledger. Otherwise a stale ``status_on_exit`` survives to
                    # drive the Natural Cure re-entry inference (``_on_switch_in``) — the
                    # amplifier that lets an already-cured benched mon re-enter mis-flagged.
                    changes["status_on_exit"] = None
                    changes["cure_all_count_on_exit"] = -1
                self._replace_belief(belief, **changes)
            return

        if event_type == "-cureteam" and actor_slot:
            # Aromatherapy cures EVERY living team member and (gen3 inherits the gen4 mod)
            # emits a SINGLE ``|-cureteam|SOURCE`` line with NO per-mon ``-curestatus`` — it
            # calls ``clearStatus()`` (silent), not ``cureStatus()``. Heal Bell DOES emit
            # per-mon ``[silent]`` curestatus, which is why census #762 — exercising Heal Bell
            # only — wrongly declared ``-cureteam`` dead code. The source ident is the active
            # user, so its side is ``actor_slot``. Clear status + condition-suffix + sleep
            # counters (and retire the switch-out ledger) for every LIVING member, mirroring
            # ``clearStatus()``'s ``hp && status`` gate so unrelated members stay byte-identical.
            for member in list(self._sides.get(actor_slot, [])):
                if member.condition == "0 fnt":
                    continue  # clearStatus() no-ops on a fainted (hp == 0) mon
                if member.status is None and _status_token_from_condition(member.condition) is None:
                    continue  # nothing to clear — do not churn an already-healthy member
                self._clear_sleep_clause_for(member)
                changes = self._status_cure_changes(member)
                changes["status_on_exit"] = None
                changes["cure_all_count_on_exit"] = -1
                self._replace_belief(member, **changes)
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
                    # Mark the cant unresolved: a following Sleep-Talk/Snore ``|move|`` this turn
                    # reclassifies it as a ``skippedTime`` turn; otherwise ``|upkeep`` clears it as
                    # a plain (skip-resetting) sleep turn.
                    self._sleep_cant_pending.add(belief.key)
            return

        if event_type == "faint" and target_slot:
            belief = self._target_belief(target_slot, target_ident)
            if belief is not None:
                if self._pending_mudshot is not None and self._pending_mudshot.get("target_key") == belief.key:
                    # A KO'd target never runs the secondary; no Shield Dust evidence.
                    self._pending_mudshot = None
                self._clear_sleep_clause_for(belief)
                self._replace_belief(
                    belief,
                    condition="0 fnt",
                    active=False,
                    status_on_exit=None,
                    cure_all_count_on_exit=-1,
                    # Transform ends on faint just as it does on a regular switch. Keeping the
                    # copied identity after a faint leaks stale target attributes into the token.
                    transformed=False,
                    transform_species=None,
                )
            return

        if event_type == "-sethp" and raw_line:
            # Pain Split: ``|-sethp|p1a: X|155/307`` — keep condition (and the pre-residual
            # snapshot) current or later pinch-berry sweeps read stale HP.
            parts = raw_line.split("|")
            sethp_ident = parts[2] if len(parts) > 2 else None
            sethp_slot = _slot_from_ident(sethp_ident)
            sethp_condition = parts[3].strip() if len(parts) > 3 else None
            if sethp_slot and sethp_condition:
                belief = self._target_belief(sethp_slot, sethp_ident)
                if belief is not None:
                    self._replace_belief(belief, condition=sethp_condition)
                    self._hp_after_actions[belief.key] = _hp_fraction_from_condition(sethp_condition)
            return

        if event_type == "turn":
            self._turn_number += 1
            for side in ("p1", "p2"):
                active = self._active_belief(side)
                if active is not None:
                    self._replace_belief(active, turns_active=active.turns_active + 1)
            return

        if event_type == "upkeep" and raw_line == "|upkeep":
            # The |upkeep line follows all residuals (Leftovers heals, pinch-berry eats), so
            # end-of-turn non-proc pruning runs here with this turn's proc sets fully populated.
            self._sweep_end_of_turn_non_procs()
            self._resolve_pending_mudshot()
            # A ``|cant|…|slp`` still unresolved at upkeep saw no ``sleepUsable`` move this turn: a
            # plain sleep turn, which resets ``skippedTime`` to 0 in the sim (only a trailing run of
            # Sleep-Talk/Snore turns is refundable on the next pivot).
            for pending_key in self._sleep_cant_pending:
                belief = self._belief_by_key(pending_key)
                if belief is not None and belief.sleep_skipped_turns:
                    self._replace_belief(belief, sleep_skipped_turns=0)
            self._sleep_cant_pending = set()
            self._leftovers_healed_this_turn = set()
            self._berry_ate_this_turn = set()
            self._shed_skin_activated_this_turn = set()
            return

        if event_type in {"-ability", "ability"} and target_slot and primary:
            # Trace names the TRACED mon's ability on the TRACER's line; redirect to the
            # ``[of]`` mon, and record nothing at all when it is absent. See _trace_copy_source.
            is_trace, traced_ident = _trace_copy_source(raw_line)
            if is_trace:
                traced_slot = _slot_from_ident(traced_ident) if traced_ident else None
                if not traced_slot:
                    return
                target_slot, target_ident = traced_slot, traced_ident
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
                tricked = bool(raw_line) and "[from] move: Trick" in raw_line
                if tricked:
                    changes["item_mutated"] = True
                    # The mon RECEIVED an item: whatever its removal history, it
                    # now publicly holds one that is not the sampled assignment.
                    changes["item_removed"] = False
                    # Trick's -item line names the CURRENT item on its own target
                    # (probed verbatim both directions) — the audited surface the
                    # world-construction override consumes.
                    changes["current_public_item"] = str(primary)
                elif raw_line and "[from] move:" in raw_line:
                    # Hardening (PR #741 review): an -item line from an UNAUDITED
                    # move source (a pool change to Thief/Covet, Recycle, ...)
                    # also mutates the held item, but its exact semantics are not
                    # modeled here — mark the mutation WITHOUT a confirmed current
                    # item so world construction fails closed instead of silently
                    # treating it as a plain reveal of the original assignment.
                    changes["item_mutated"] = True
                    changes["item_removed"] = False
                    changes["current_public_item"] = None
                self._replace_belief(belief, **changes)
                if tricked:
                    # ... and the item it names is the PARTNER's original assignment.
                    self._record_tricked_partner_original_item(
                        subject_slot=target_slot, item=str(primary), raw_line=raw_line
                    )

    def _track_trick_pairing(self, event_type: Optional[str], raw_line: Optional[str]) -> None:
        """Latch ``|-activate|<source>|move: Trick|[of] <target>`` for the ``-item`` lines it precedes.

        trick.onHit emits the ``-activate`` first and then, contiguously, one ``-item`` or
        ``-enditem`` line per participant, so the pairing is cleared by the first event that is
        neither. Nothing else can consume a stale pairing, and a Trick whose ``-activate`` never
        arrived leaves ``_pending_trick`` None — the redirect then records nothing.
        """

        if event_type not in {"-item", "-enditem"}:
            self._pending_trick = None
        if event_type != "-activate" or not raw_line:
            return
        if not _TRICK_EFFECT.search(raw_line):
            return
        fields = raw_line.split("|")
        source_ident = fields[2].strip() if len(fields) > 2 else ""
        of_match = re.search(r"\[of\] ([^|\]]+)", raw_line)
        target_ident = of_match.group(1).strip() if of_match else ""
        source_slot = _slot_from_ident(source_ident)
        target_slot = _slot_from_ident(target_ident)
        if not source_slot or not target_slot or source_slot == target_slot:
            return
        self._pending_trick = _PendingTrick(
            source_slot=source_slot,
            source_ident=source_ident,
            source_original_known=self._holds_own_assigned_item(source_slot),
            target_slot=target_slot,
            target_ident=target_ident,
            target_original_known=self._holds_own_assigned_item(target_slot),
        )

    def _holds_own_assigned_item(self, showdown_slot: str) -> bool:
        belief = self._active_belief(showdown_slot)
        return belief is not None and not belief.item_mutated

    def _record_tricked_partner_original_item(
        self, *, subject_slot: str, item: str, raw_line: Optional[str]
    ) -> None:
        """Attribute a Trick ``-item`` line's item to the PARTNER, whose assignment it is.

        See ``_PendingTrick``: the subject is who holds the item now, the partner is who was
        given it by the generator. With no pairing available, or with a partner that was already
        carrying somebody else's item, this records nothing — crediting the subject would be the
        cross-attribution bug rather than a conservative fallback.
        """

        pending = self._pending_trick
        if pending is None:
            return
        partner_slot, partner_ident, partner_original_known = pending.partner_of(subject_slot)
        if not partner_slot or not partner_original_known:
            return
        partner = self._target_belief(partner_slot, partner_ident)
        if partner is None or partner.original_public_item:
            return
        self._replace_belief(
            partner,
            original_public_item=item,
            evidence=_append_evidence(
                partner.evidence,
                BeliefEvidence(
                    kind="original-item",
                    detail=(
                        f"Trick handed {item} to the other side: it is this mon's own "
                        "assigned item, whatever it holds now."
                    ),
                    source_line=raw_line,
                ),
            ),
        )

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
            if "[from] move: Knock Off" in raw_line or "[from] move: Trick" in raw_line:
                # Held-item mutation: non-proc pruning applies to the ORIGINAL assignment
                # only. Either surface here ends with the mon holding NOTHING (Knock Off
                # removal, or a Trick that took the item and returned none — probed:
                # ``|-enditem|SLOT|ITEM|[silent]|[from] move: Trick``) — a public item
                # state determinized worlds can express, unlike a live swap.
                mutation: dict[str, Any] = {
                    "item_mutated": True,
                    "item_removed": True,
                    "current_public_item": None,
                }
                # Both of these ``-enditem`` surfaces name the SUBJECT's own item (unlike
                # Trick's ``-item`` lines, which are cross-attributed), so if the mon was
                # still holding its own assignment the removal states that assignment with
                # certainty. Recording it keeps the fact that ``item_mutated`` would
                # otherwise discard: the item is gone, but it is now KNOWN.
                if not belief.item_mutated and not belief.original_public_item:
                    mutation["original_public_item"] = item
                belief = self._replace_belief(belief, **mutation)
            elif "[from] move:" in raw_line:
                # Hardening (PR #741 review): an -enditem from an UNAUDITED move source
                # (a pool change to Thief/Covet, ...) is not extended removal semantics —
                # the giving/taking halves of such moves are unmodeled. Mark the mutation
                # with NO removal and NO confirmed current item: world construction fails
                # closed loudly (public_effect_blocked) instead of silently handing the
                # sampled item back or guessing.
                belief = self._replace_belief(
                    belief, item_mutated=True, item_removed=False, current_public_item=None
                )
            else:
                # Consumption (a berry ``[eat]``, White Herb, gen4+ ``[from] stealeat``):
                # the item is publicly GONE from the holder. Unlike Knock Off/Trick this
                # does NOT set item_mutated — a self-consumed item was (unless a prior
                # mutation says otherwise) the original assignment, so revealed_item must
                # keep pinning variant matching. Worlds express the current state by
                # clearing the sampled item, instead of handing the eaten berry back.
                belief = self._replace_belief(
                    belief, item_removed=True, current_public_item=None
                )
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
        twin = self.clone()
        twin.resolve_pending_switches_at_boundary()
        return twin.snapshot().for_player(showdown_slot)

    def narrow_candidate_variants(
        self,
        key: str,
        survivors: Sequence[Mapping[str, Any]],
        *,
        reason: str = "",
    ) -> bool:
        """Restrict ``key``'s candidate variants to ``survivors``. Monotone.

        The narrowing is stored by identity and re-applied every time the set source
        re-summarizes, so it persists for the rest of the battle. Returns True when the
        standing pin actually changed.

        Refuses two ways, both of which LEAVE THE STANDING PIN INTACT rather than widening:
        an empty ``survivors`` (no evidence is not evidence for nothing), and a survivor list
        disjoint from the standing pin (counted in ``variant_pin_conflicts``). The asymmetry is
        deliberate: dropping a true variant is unrecoverable, while declining a narrowing only
        costs precision.
        """

        del reason  # accepted for call-site readability; conflicts are counted, not attributed
        incoming = frozenset(variant_identity(variant) for variant in survivors)
        if not incoming:
            return False
        standing = self._variant_pins.get(key)
        if standing is None:
            self._variant_pins[key] = incoming
            return True
        merged = standing & incoming
        if not merged:
            self._variant_pin_conflicts[key] = self._variant_pin_conflicts.get(key, 0) + 1
            return False
        if merged == standing:
            return False
        self._variant_pins[key] = merged
        return True

    @property
    def variant_pin_conflicts(self) -> Mapping[str, int]:
        """Per-key count of contradicted narrowings — a precision alarm for the producer."""

        return dict(self._variant_pin_conflicts)

    def _apply_variant_pin(
        self, key: str, summary_fields: dict[str, Any]
    ) -> dict[str, Any]:
        """Filter a summary's variants through ``key``'s standing pin, in place.

        ``uncertainty`` is ``count / species_pool_total`` (see the randbats source), so scaling
        it by the surviving fraction keeps that denominator implicit and correct. An
        ``inconsistent`` summary is left alone: its variants are the unconstrained fallback pool
        with uncertainty forced to 1.0, and narrowing a fallback would manufacture confidence
        the reveals do not support.

        The ``possible_*`` surfaces are narrowed too, so the snapshot cannot end up internally
        inconsistent (one surviving variant, five "possible" moves). They are FILTERED, never
        recomputed: see :func:`_narrowed_possible_values`.
        """

        pin = self._variant_pins.get(key)
        variants = summary_fields.get("candidate_variants") or ()
        if not pin or not variants or summary_fields.get("_inconsistent"):
            return summary_fields
        kept = tuple(v for v in variants if variant_identity(v) in pin)
        if not kept or len(kept) == len(variants):
            # Empty means the pin and this summary disagree entirely (the producer pinned
            # against a stale variant list). Keep the unfiltered set: see the refusal
            # asymmetry in narrow_candidate_variants.
            if not kept:
                self._variant_pin_conflicts[key] = self._variant_pin_conflicts.get(key, 0) + 1
            return summary_fields
        previous = len(variants)
        summary_fields["candidate_variants"] = kept
        summary_fields["candidate_set_count"] = len(kept)
        prior_uncertainty = summary_fields.get("uncertainty")
        if isinstance(prior_uncertainty, (int, float)):
            summary_fields["uncertainty"] = float(prior_uncertainty) * (len(kept) / previous)
        for field_name, variant_field, plural in (
            ("possible_abilities", "ability", False),
            ("possible_items", "item", False),
            ("possible_moves", "moves", True),
        ):
            summary_fields[field_name] = _narrowed_possible_values(
                summary_fields.get(field_name) or (),
                variants,
                kept,
                variant_field,
                plural=plural,
            )
        return summary_fields

    def clone(self) -> "PublicBattleBeliefEngine":
        """Copy only the public-evidence state needed to continue a sampled world."""

        twin = PublicBattleBeliefEngine(
            format_id=self.format_id,
            set_source=self.set_source,
            item_belief_narrowing=self.item_belief_narrowing,
        )
        twin._event_count = self._event_count
        twin._sides = copy.deepcopy(self._sides)
        twin._pending_switches = copy.deepcopy(self._pending_switches)
        twin._turn_number = self._turn_number
        twin._cure_all_count = dict(self._cure_all_count)
        twin._sleep_clause_holder = dict(self._sleep_clause_holder)
        twin._leftovers_healed_this_turn = set(self._leftovers_healed_this_turn)
        twin._hp_after_actions = dict(self._hp_after_actions)
        twin._stay_locked_move = dict(self._stay_locked_move)
        twin._berry_ate_this_turn = set(self._berry_ate_this_turn)
        twin._shed_skin_activated_this_turn = set(self._shed_skin_activated_this_turn)
        twin._sleep_cant_pending = set(self._sleep_cant_pending)
        twin._pending_mudshot = copy.deepcopy(self._pending_mudshot)
        twin._pending_trick = self._pending_trick
        # Narrowings are public evidence like any other reveal: a sampled world that forgot
        # them would re-admit variants the strikes already excluded.
        twin._variant_pins = dict(self._variant_pins)
        twin._variant_pin_conflicts = dict(self._variant_pin_conflicts)
        return twin

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
        gender: Optional[str] = None,
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
                if gender is not None:
                    updated = replace(updated, gender=gender)
                updated = self._with_set_summary(updated)
                side[index] = updated
                return updated
        created = self._with_set_summary(
            RevealedPokemonBelief(
                showdown_slot=showdown_slot,
                species=species,
                condition=condition,
                active=bool(active),
                gender=gender,
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
        # (and the known stats it implied). The faint handler applies the same reset.
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

    def _benched_target_belief(
        self,
        showdown_slot: str,
        target_ident: Optional[str],
    ) -> RevealedPokemonBelief | None:
        """Resolve a benched target (no position letter) by species, NOT the active mon.

        Used only for the ``-curestatus`` team-cure (Heal Bell / Aromatherapy) path, where a cured
        benched ally is identified purely by species. Species Clause makes this unambiguous, and a
        statused benched mon was necessarily seen on the field, so it already exists in beliefs."""
        species = _species_from_ident(target_ident)
        if species is None:
            return None
        normalized = _normalize_species(species)
        side = self._sides.get(showdown_slot, [])
        if not any(_normalize_species(belief.species) == normalized for belief in side):
            # No exact match: a cosmetic-forme benched mon serializes under its BASE name in
            # the cure ident (gen3 randbats name an Unown-Z simply "Unown") while beliefs track
            # the lettered forme from the switch details ("Unown-Z"). Redirect the cure onto the
            # existing forme belief so its status still clears; without this the benched cure
            # spawns a phantom base-species entry and the real forme keeps its stale status.
            for belief in side:
                if _base_species_id(belief.species) == normalized:
                    return belief
        return self._upsert(showdown_slot=showdown_slot, species=species)

    @property
    def sleep_clause_holders(self) -> dict[str, Optional[str]]:
        """Per side: the belief key of the opposing mon this side currently has asleep (live)."""
        return dict(self._sleep_clause_holder)

    @property
    def turn_number(self) -> int:
        return self._turn_number

    def _charge_move_use(
        self, belief: RevealedPokemonBelief, move_id: str, *, foe_targeted: bool = True
    ) -> RevealedPokemonBelief:
        # Pressure on the OPPOSING active doubles PP spent, but only for FOE-TARGETED moves
        # (gen3 engine behavior — self-targeted Rest/Swords Dance are never pressured). Gen 3
        # announces Pressure on entry, so the opposing ability is public when the double applies.
        opposing = self._active_belief(_other_side(belief.showdown_slot))
        charge = (
            2
            if foe_targeted
            and opposing is not None
            and _normalize_identifier(opposing.revealed_ability or "") == "pressure"
            else 1
        )
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
        self._hp_after_actions[belief.key] = _hp_fraction_from_condition(condition)
        # ``choiceband.onStart`` removes the ``choicelock`` volatile, so entering the field
        # clears the lock and the next move selected becomes the new locked one.
        self._stay_locked_move.pop(belief.key, None)
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
            if condition_status == "slp":
                # gen3 refunds the sleep turns spent on Sleep Talk / Snore before this pivot
                # (``time += skippedTime`` in ``slp.onSwitchIn``): those turns did not advance the
                # wake timer, so subtract them from the observed sleep count on re-entry.
                changes["sleep_turns"] = max(0, belief.sleep_turns - belief.sleep_skipped_turns)
        # skippedTime is consumed on switch-in in the sim; clear it regardless of the return status.
        changes["sleep_skipped_turns"] = 0
        changes["status_on_exit"] = None
        changes["cure_all_count_on_exit"] = -1
        return self._replace_belief(belief, **changes)

    def _clear_sleep_clause_for(self, belief: RevealedPokemonBelief) -> None:
        for side, holder in list(self._sleep_clause_holder.items()):
            if holder == belief.key:
                self._sleep_clause_holder[side] = None

    def _status_cure_changes(self, belief: RevealedPokemonBelief) -> dict[str, Any]:
        """Every belief field a full non-volatile status cure clears: the ``status``
        itself, the sleep counters, AND the status suffix carried on ``condition`` (else
        the encoder's condition-suffix status fallback silently re-derives the cleared
        status — the training-encoder divergence this shared helper closes). Used by both
        the per-mon ``-curestatus`` path and the team-wide ``-cureteam`` path so they clear
        status + suffix + counters identically."""
        changes: dict[str, Any] = {
            "status": None,
            "sleep_turns": 0,
            "rest_sleep": False,
            "sleep_skipped_turns": 0,
        }
        stripped = strip_condition_status(belief.condition)
        if stripped != belief.condition:
            changes["condition"] = stripped
        return changes

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

    def _note_choice_lock_selection(
        self, belief: RevealedPokemonBelief, move_id: str
    ) -> RevealedPokemonBelief:
        """Two different FREELY SELECTED moves in one stay rule Choice Band out.

        ``data/conditions.ts`` choicelock (gen3 inherits it — ``data/mods/gen3`` overrides
        neither the volatile nor the item) records ``activeMove.id`` in ``onStart`` and fails
        any later move whose id differs in ``onBeforeMove``. ``choiceband.onModifyMove`` adds
        the volatile on the holder's first move and ``choiceband.onStart`` removes it on
        switch-in, so the lock spans exactly one stay on the field. A mon seen selecting two
        different moves without leaving therefore cannot have been holding one.

        The negative direction only, and the same evidence class as the Leftovers / Lum /
        pinch-berry non-proc pruning it joins: certain, free, and derived from public
        ``|move|`` lines with no damage analysis and so no precision gate. Choice Band is the
        pool's second most common item (160 variants, behind Leftovers' 1371), so this is the
        only non-proc rule that moves a large share of the pool.

        "Freely selected" is what the caller has already established: called moves, locked
        continuations (Solar Beam's release — the pool's only two-turn move), Struggle and
        moves used while transformed have all returned before this point, and Sleep Talk's
        callee is excluded by the caller because Sleep Talk chose it, not the player. The one
        remaining way a differing move could reach the protocol under an active lock is
        choicelock's own failure line (``|move|<mon>|<other>|[still]`` + ``|-fail|``), and it
        is unreachable here: ``onDisableMove`` removes the other moves from the request, so
        neither a bot nor a ladder client can select one.

        The bookkeeping runs in every switch state; only the CONCLUSION is gated, so a
        switch-off engine is byte-identical. Frozen once the held item is mutated, exactly
        like the rest of the family: after a Trick the mon is locked (or not) by somebody
        else's item, which says nothing about its own assignment.
        """

        locked = self._stay_locked_move.get(belief.key)
        if locked is None:
            self._stay_locked_move[belief.key] = move_id
            return belief
        if locked == move_id or not self.item_belief_narrowing:
            return belief
        if belief.item_mutated or belief.revealed_item:
            return belief
        return self._rule_out_items(
            belief,
            ("choiceband",),
            "Selected two different moves in one stay on the field; the Choice Band lock "
            "forbids that, so Choice Band variants were removed.",
        )

    def _sweep_end_of_turn_non_procs(self) -> None:
        for side in ("p1", "p2"):
            belief = self._active_belief(side)
            if belief is None or belief.item_mutated or belief.revealed_item:
                continue
            hp_fraction = _hp_fraction_from_condition(belief.condition)
            if hp_fraction is None or hp_fraction <= 0.0:
                continue
            # Leftovers evidence keys off the PRE-RESIDUAL state: gen3 runs the Leftovers slot
            # before status/Leech chip, so a mon damaged only by later residuals gave its
            # Leftovers no chance to fire. No snapshot => no evidence (conservative).
            hp_pre_residual = self._hp_after_actions.get(belief.key)
            if (
                hp_pre_residual is not None
                and hp_pre_residual < 1.0
                and belief.key not in self._leftovers_healed_this_turn
            ):
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
            # Pinch berries (Salac/Petaya/Liechi) activate on an HP DROP during the action
            # phase (being hit), NOT on an end-of-turn residual crossing: in gen3 a mon that
            # first falls to/below 25% from a later residual (Toxic/burn/sand/Leech chip) got
            # no berry-activation opportunity at this boundary. Gate on the action-phase HP
            # snapshot, exactly like the Leftovers slot above, so only a genuine action-phase
            # non-proc — HP already at/below threshold after actions with no berry eaten —
            # rules the pinch variants out. No snapshot => no evidence (conservative).
            if (
                hp_pre_residual is not None
                and hp_pre_residual <= 0.25
                and belief.key not in self._berry_ate_this_turn
            ):
                belief = self._rule_out_items(
                    belief,
                    ("salacberry", "petayaberry", "liechiberry"),
                    "Action-phase HP at or below 25% with no pinch-berry activation; pinch variants removed.",
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

    def _variant_matching_item(self, belief: RevealedPokemonBelief) -> Optional[str]:
        """The item key variant matching may use: the GENERATOR's assignment, never a swapped one.

        Un-mutated, ``revealed_item`` IS that assignment (a consumed berry still identifies it).
        Post-mutation (Trick / Knock Off) the current item is somebody else's, so the channel is
        suppressed — that part is unconditional and unchanged. What the mutation does NOT
        justify discarding is the ORIGINAL assignment when the same protocol line named it, and
        ``original_public_item`` is exactly that fact.

        Gated on ``item_belief_narrowing`` because honouring it narrows candidate sets that
        previously stayed wide: 36% of gen3 item reveals pin a single variant outright, which
        moves NUMERIC_CANDIDATE_SET_COUNT / NUMERIC_UNCERTAINTY on every schema. Note the
        RECEIVED item is never a key in either state — a Choice Band Tricked onto a mon that
        could legitimately carry one must not narrow it — because it lands in ``revealed_item``
        while ``original_public_item`` stays with the mon the generator gave it to.
        """

        if not belief.item_mutated:
            return belief.revealed_item
        if self.item_belief_narrowing:
            return belief.original_public_item
        return None

    def _with_set_summary(self, belief: RevealedPokemonBelief) -> RevealedPokemonBelief:
        if self.set_source is None:
            return belief
        try:
            summary = self.set_source.summarize(
                format_id=self.format_id,
                species=belief.species,
                revealed_moves=belief.revealed_moves,
                revealed_ability=belief.revealed_ability,
                revealed_item=self._variant_matching_item(belief),
                ruled_out_abilities=belief.ruled_out_abilities,
                ruled_out_items=belief.ruled_out_items,
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
        fields: dict[str, Any] = {
            "candidate_set_count": summary.candidate_count,
            "uncertainty": summary.uncertainty,
            "candidate_variants": summary.candidate_variants,
            "possible_abilities": summary.possible_abilities,
            "possible_items": summary.possible_items,
            "possible_moves": summary.possible_moves,
            "_inconsistent": summary.inconsistent,
        }
        # Re-applied on EVERY summarize (each reveal re-derives the set), which is what makes a
        # narrowing persist for the rest of the battle rather than being undone by the next reveal.
        fields = self._apply_variant_pin(belief_key(belief.showdown_slot, belief.species), fields)
        fields.pop("_inconsistent", None)
        return replace(
            belief,
            source_metadata=summary.source_metadata,
            **fields,
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
        if belief is None:
            return
        if event_type == "-activate" and _normalize_identifier(ability_name) == "shedskin":
            # Record the proc for the Early Bird Rest-wake guard (Fix C), before
            # the already-confirmed early return below so it fires on every proc.
            self._shed_skin_activated_this_turn.add(belief.key)
        if _normalize_identifier(belief.revealed_ability or "") == _normalize_identifier(ability_name):
            return
        if belief.revealed_ability:
            # A gen3 mon has exactly one ability, so a second, DIFFERENT claim
            # means one of the two attributions is wrong. Keep the earlier
            # confirmation and flag: the earlier reveal named this mon on its
            # own line (e.g. ``|-ability|p1a: Zapdos|Pressure|[silent]`` on
            # entry) while later conflicting claims come from tag-attribution
            # heuristics — exactly the class of the live-captured bug where a
            # ``[of]`` misread swapped Zapdos's confirmed Pressure for Volt
            # Absorb and silently destroyed the belief. Overwriting would also
            # collapse the candidate variants to an impossible set (off-script
            # fallback, uncertainty 1.0).
            self._replace_belief(
                belief,
                evidence=_append_evidence(
                    belief.evidence,
                    BeliefEvidence(
                        kind="conflicting-ability-evidence",
                        detail=(
                            f"Ignored conflicting ability claim {ability_name}; "
                            f"keeping earlier confirmed {belief.revealed_ability}."
                        ),
                        source_line=raw_line,
                    ),
                ),
            )
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

    def _belief_by_key(self, key: str) -> RevealedPokemonBelief | None:
        for side in self._sides.values():
            for pokemon in side:
                if pokemon.key == key:
                    return pokemon
        return None


@dataclass(frozen=True)
class _PendingSwitch:
    showdown_slot: str
    ident: Optional[str]
    species: str


@dataclass(frozen=True)
class _PendingTrick:
    """The Trick pairing published by ``|-activate|<source>|move: Trick|[of] <target>``.

    ``data/moves.ts`` trick.onHit takes ``yourItem`` from the target and ``myItem`` from the
    source, then announces the SWAP::

        |-item|<target>|<myItem>|[from] move: Trick      <- the SOURCE's original assignment
        |-item|<source>|<yourItem>|[from] move: Trick    <- the TARGET's original assignment

    Both lines are therefore cross-attributed with respect to the ORIGINAL item, exactly like
    the Trace ``-ability`` line (see ``_trace_copy_source``): the subject is who holds it NOW,
    the mon it identifies is the partner. Unlike Trace the ``-item`` lines carry no ``[of]``
    tag of their own, so the pairing has to come from the ``-activate`` line that always
    precedes them; with no pairing in hand the redirect records nothing rather than crediting
    the subject, which is precisely the bug the redirect exists to avoid.

    ``*_original_known`` snapshots, at ``-activate`` time, whether that participant's held item
    was still its own generator assignment. A mon carrying an item from an EARLIER Trick names
    that item on the swap line, not the generator's, so its partner learns nothing.
    """

    source_slot: str
    source_ident: Optional[str]
    source_original_known: bool
    target_slot: str
    target_ident: Optional[str]
    target_original_known: bool

    def partner_of(self, subject_slot: str) -> tuple[Optional[str], Optional[str], bool]:
        """``(slot, ident, original_known)`` of the mon whose item a line ON ``subject_slot`` names."""

        if subject_slot == self.source_slot:
            return self.target_slot, self.target_ident, self.target_original_known
        if subject_slot == self.target_slot:
            return self.source_slot, self.source_ident, self.source_original_known
        return None, None, False


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
            gender=pokemon.gender,
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
        gender=pokemon.gender,
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

# gen3 ``move.sleepUsable`` — the only moves a sleeping mon may execute (its ``|cant|…|slp`` still
# fires first). Selecting one accrues ``skippedTime`` (see ``sleep_skipped_turns``).
_SLEEP_USABLE_MOVES = frozenset({"sleeptalk", "snore"})

# ``move: Trick`` on an ``-activate`` line, refusing the later-gen ``Trick-or-Treat`` and
# ``Trick Room`` prefixes that a plain substring test would swallow.
_TRICK_EFFECT = re.compile(r"move: Trick(?![\w-])")


_RESIDUAL_HP_TAGS = (
    "[from] psn",
    "[from] brn",
    "[from] Sandstorm",
    "[from] Hail",
    "[from] Leech Seed",
    "[from] item: Leftovers",
    "[from] ability: Rain Dish",
    "[from] Curse",
    "[from] Nightmare",
    "[from] move: Wrap",
    "[from] partiallytrapped",
    # Wish's landing heal is an end-of-turn RESIDUAL heal (gen3 slotCondition, residual order 7),
    # exactly like the Leftovers / Rain Dish residual heals above: it must NOT overwrite the
    # action-phase HP snapshot the non-proc item pruning reads (same #769 mechanism as the psn/brn
    # residual-DAMAGE tags). Without this, a Wish landing on a mon that fell to <=25% during the
    # action phase (no berry eaten) would overwrite the low pre-residual snapshot with the healed
    # value and MASK the action-phase non-proc, wrongly leaving the pinch variants un-pruned.
    # (Ingrain is deliberately NOT listed — 0 gen3-randbats pool carriers, unreachable.)
    "[from] move: Wish",
)


def _is_action_phase_hp_change(raw_line: Optional[str]) -> bool:
    """True for HP changes that land before the end-of-turn residual slots.

    Untagged damage/heals are action-phase; Spikes switch-in chip is action-phase (it fires on
    entry, before any residual). Residual-tagged sources must NOT update the pre-residual
    snapshot or gen3's residual order (Leftovers before status/Leech chip) manufactures false
    Leftovers evidence.
    """
    if not raw_line:
        return True
    if "[from] Spikes" in raw_line or "[from] drain" in raw_line or "Recoil" in raw_line:
        return True
    return not any(tag in raw_line for tag in _RESIDUAL_HP_TAGS)


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


def strip_condition_status(condition: Optional[str]) -> Optional[str]:
    """Drop the non-volatile status suffix (slp/par/brn/psn/tox/frz) from a condition
    string, preserving the HP token and the ``fnt`` marker: ``'387/387 slp' -> '387/387'``,
    ``'0 fnt' -> '0 fnt'``, ``'283/301' -> '283/301'``.

    A cured mon whose ``condition`` still carries its old status suffix makes the
    encoder's condition-suffix status fallback (showdown.py, ``status = belief.status
    if belief.status is not None else condition.status``) silently re-derive the cleared
    status. Every status-cure surface — per-mon ``-curestatus`` and team-wide
    ``-cureteam`` on the belief side, and the public-condition update on the showdown
    side — must strip it in lockstep with clearing ``belief.status``."""
    if not condition:
        return condition
    parts = condition.split()
    if not parts:
        return condition
    return " ".join([parts[0]] + [part for part in parts[1:] if part == "fnt"])


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


def _trace_copy_source(raw_line: str) -> tuple[bool, Optional[str]]:
    """``(is_a_Trace_copy, ident of the mon COPIED FROM)`` for an ``-ability`` line.

    ``sim/pokemon.ts setAbility`` emits
    ``|-ability|<tracer>|<COPIED>|<tracer's own ability>|[from] ability: Trace|[of] <traced>``,
    so the line's SUBJECT is the tracer while the ability named belongs to the TRACED mon.
    Attributing it to the subject is wrong in both directions at once: it corrupts the tracer's
    candidate set with an ability it cannot have, and it throws away a CERTAIN reveal that
    would narrow the traced mon.

    The corruption is sticky. Trace re-fires on every switch-in and the conflicting-evidence
    guard keeps the FIRST claim, so one early copy persists for the rest of the battle — the
    live case being a Gardevoir left holding ``levitate`` and silently granted Spikes immunity.

    ``(True, None)`` means a Trace line whose ``[of]`` is absent: the copy cannot be attributed
    to anyone, and falling back to the subject is precisely the bug, so callers record nothing.

    Reachability: Trace is live in the gen3 randbats pool (Gardevoir, Porygon2). Both carry
    Trace on every variant, so the tracer's OWN ability — the third field on the line — is
    deliberately not recorded anywhere: it could never narrow their candidate sets.
    """

    if not re.search(r"\[from\] ability: Trace\b", raw_line or ""):
        return False, None
    traced = re.search(r"\[of\] ([^|\]]+)", raw_line or "")
    return True, traced.group(1).strip() if traced else None


def _confirmed_ability_from_event(event: Any) -> tuple[Optional[str], Optional[str]]:
    event_type = _event_value(event, "event_type")
    actor_ident = _event_value(event, "actor_ident")
    target_ident = _event_value(event, "target_ident")
    primary = _event_value(event, "primary")
    raw_line = _event_value(event, "raw_line") or ""
    if event_type in {"-ability", "ability"} and primary:
        is_trace, traced_ident = _trace_copy_source(raw_line)
        if is_trace:
            return traced_ident, (primary if traced_ident else None)
        return target_ident, primary
    if event_type == "-start" and primary and primary.strip().lower().startswith("ability:"):
        # ``|-start|<holder>|ability: X`` (Flash Fire activation): the holder
        # is the mon on the line, which the event parser surfaces as
        # ``actor_ident`` (``-start`` is outside its target-ident group).
        return actor_ident or target_ident, primary.split(":", 1)[1].strip()
    if event_type == "-activate" and primary and primary.strip().lower().startswith("ability:"):
        # ``|-activate|<holder>|ability: X`` — Shed Skin's ONLY public tell
        # (abilities.ts shedskin onResidual: ``add('-activate', pokemon,
        # 'ability: Shed Skin')``), plus Synchronize/Own Tempo/Limber/etc. The
        # mon on the line is ALWAYS the holder (parts[2]); for the few shapes
        # that also carry ``[of] <other>`` — Forewarn's warn-target, Commander's
        # ally — that ``[of]`` names a different mon, never the holder, so it is
        # deliberately ignored here. ``-activate`` sits inside the parser's
        # target-ident group, so the holder arrives as ``target_ident``. Guarded
        # to ``ability:``-bearing lines only, so item/move/Substitute ``-activate``
        # shapes never reach this. Goes through the ``_record_raw_ability_reveal``
        # guard, so it cannot overwrite an authoritative earlier ``-ability`` pin.
        return target_ident or actor_ident, primary.split(":", 1)[1].strip()
    ability_match = re.search(r"\[from\] ability: ([^|\]]+)", raw_line)
    if not ability_match:
        return None, None
    ability_name = ability_match.group(1).strip()
    if event_type == "-heal":
        # ``|-heal|<healed>|<cond>|[from] ability: X|[of] <attacker>`` — for
        # heals Showdown's ``[of]`` names the MOVE SOURCE, not the ability
        # holder (sim/battle.ts:2311 appends ``[of] ${source}`` whenever
        # source !== target). The holder is always the healed mon (Volt/Water
        # Absorb). Reading ``[of]`` here pinned the ability on the ATTACKER
        # (live capture: Zapdos's protocol-confirmed Pressure was overwritten
        # by Lanturn's Volt Absorb).
        return target_ident, ability_name
    ident_match = re.search(r"\[of\] ([^|]+)", raw_line)
    if ident_match:
        # Damage/status shapes (Rough Skin, Static): ``[of]`` is the holder.
        return ident_match.group(1).strip(), ability_name
    # No ``[of]``: the holder is the mon on the line itself. ``-immune`` sits
    # outside the parser's target-ident group, so its ident arrives as
    # ``actor_ident``; keep ``target_ident`` first for the shapes that do
    # populate it.
    return target_ident or actor_ident, ability_name


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


def _gender_from_switch_line(raw_line: Optional[str]) -> Optional[str]:
    parts = str(raw_line or "").split("|")
    if len(parts) < 4 or parts[1] not in {"switch", "drag", "replace"}:
        return None
    for part in parts[3].split(","):
        token = part.strip().upper()
        if token in {"M", "F", "N"}:
            return token
    return None


def _base_species_id(species: str) -> str:
    """Normalized BASE species id, forme suffix dropped: ``'Unown-Z' -> 'unown'``. Used only
    to reconcile a cosmetic forme's base-name cure ident (gen3 randbats' Unown) with the
    lettered-forme species the belief tracks."""
    return _normalize_species(str(species).split("-", 1)[0])


def _slot_from_ident(ident: Optional[str]) -> Optional[str]:
    if not ident:
        return None
    match = re.match(r"^(p[12])", str(ident))
    return match.group(1) if match else None


def _ident_has_position(ident: Optional[str]) -> bool:
    """True for an ACTIVE-slot ident (``p2a: Snorlax``); False for a benched ident (``p2: Snorlax``).

    Showdown appends a field-position letter (``a`` in singles) only to on-field Pokemon; a benched
    mon referenced by a team-wide effect (Heal Bell curing every ally) carries just ``pN:``."""
    return bool(re.match(r"^p[12][a-z]", str(ident or "")))


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
