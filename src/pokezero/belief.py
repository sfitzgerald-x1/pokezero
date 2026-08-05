"""Public battle belief tracking for replay, overlays, and training inputs.

The engine in this module only consumes public information. It is intentionally
format-agnostic: random-battle set sources can be plugged in later without
changing the public-state tracking API.
"""

from __future__ import annotations

from collections import Counter
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
        # belief key -> the ability the mon is currently RUNNING (Trace copies onto the tracer).
        # Distinct from `revealed_ability`, which stays the mon's own set ability.
        self._running_ability: dict[str, str] = {}
        # Reachability instrumentation for the Pressure branch of `_charge_move_use`. A PP
        # differential that reports "0 violations" proves nothing unless the branch it is meant to
        # cover actually ran, and the branch this counts -- a foe-targeted move whose wire target
        # slot the engine blanked -- is exactly the one that used to lose its double charge.
        self._pressure_charge_counts: Counter[str] = Counter()
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
        # HP each active mon was at when the gen3 ITEM RESIDUAL SLOT (order 10 / subOrder 4, where
        # Leftovers and the pinch berries both fire) ran, as far as the public log can determine it.
        # ``None`` for a key means "not determinable this turn" and is treated as NO EVIDENCE, which
        # is different from an absent key only in intent. Maintained by ``_apply_hp_observation``.
        self._hp_after_actions: dict[str, Optional[float]] = {}
        # True from the sim's action->residual boundary marker until the next turn's first action.
        # ``sim/battle.ts`` emits a BARE ``|`` line at the top of the ``residual`` queue action
        # (``case 'residual': this.add('')``), which is the only in-band signal separating the two
        # phases -- and the phase, not the ``[from]`` tag, is what actually decides whether an HP
        # change happened before the item slot. See ``_track_residual_phase``.
        self._in_residual_phase: bool = False
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
        self._track_residual_phase(event_type, raw_line)
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
                # Transform copies the target's ABILITY as well as its stats/moves
                # (`sim/pokemon.ts:1353` in `transformInto`, `if (this.battle.gen > 2)
                # this.setAbility(pokemon.ability, ...)`; nothing
                # in the gen3 -> gen4 -> ... chain overrides `transformInto`). So a Ditto that
                # copies a Pressure mon pressures for real. Second producer of a running ability
                # after Trace; missing it cost Ho-Oh's Sacred Fire a double-charge.
                target_belief = self._active_belief(_slot_from_ident(primary) or "")
                if target_belief is not None:
                    # The target's RUNNING ability, not its revealed one: transforming into a mon
                    # that has itself Traced copies what it is running now.
                    copied = self._running_ability.get(
                        target_belief.key
                    ) or target_belief.revealed_ability
                    if copied:
                        self._running_ability[belief.key] = str(copied)
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
                    belief = self._charge_move_use(
                        belief, move_id, wire_target_slot=target_slot or ""
                    )
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
                self._apply_hp_observation(
                    belief.key, _string_or_none(primary), raw_line=raw_line
                )
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
            # Pain Split: ``|-sethp|p1a: X|155/307`` — keep condition (and the pre-slot HP
            # snapshot) current or later pinch-berry sweeps read stale HP. Pain Split is a move, so
            # this is always action-phase; it goes through the same classifier anyway rather than
            # writing the snapshot directly, so there is one place that decides.
            parts = raw_line.split("|")
            sethp_ident = parts[2] if len(parts) > 2 else None
            sethp_slot = _slot_from_ident(sethp_ident)
            sethp_condition = parts[3].strip() if len(parts) > 3 else None
            if sethp_slot and sethp_condition:
                belief = self._target_belief(sethp_slot, sethp_ident)
                if belief is not None:
                    self._replace_belief(belief, condition=sethp_condition)
                    self._apply_hp_observation(
                        belief.key, sethp_condition, raw_line=raw_line
                    )
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
            # The pre-slot HP snapshot is PER-TURN evidence, like everything else cleared here: a mon
            # that takes NO HP line at all on a turn published nothing about that turn, and the phase
            # classifier cannot help, because it only sees lines that exist. The live reproducer was
            # Togetic (true item Leftovers): no HP change on the turn, snapshot stale at 135/264 from
            # two turns earlier, toxic chip at 10/6 leaving it at full when its 10/4 slot ran, and
            # Leftovers ruled out from the stale value.
            #
            # Scope of the claim, stated precisely because the previous version of this comment
            # overstated it: with the classifier fixed, a stale snapshot can only ever be HIGHER than
            # the true HP at the next slot (HP rises only through lines that update the snapshot,
            # discard it, or reveal the item outright), so leaving it would cost narrowing, not
            # soundness. The clear stays because the policy is that these rules reason from what THIS
            # turn published rather than from a monotonicity argument across turns. Its guard is
            # ``test_the_pre_slot_snapshot_is_cleared_at_upkeep``, which asserts the invariant
            # directly; that test's own docstring explains why no behavioural fixture can.
            self._hp_after_actions = {}
            return

        if event_type in {"-ability", "ability"} and target_slot and primary:
            # Trace names the TRACED mon's ability on the TRACER's line; redirect to the
            # ``[of]`` mon, and record nothing at all when it is absent. See _trace_copy_source.
            is_trace, traced_ident = _trace_copy_source(raw_line)
            if is_trace:
                traced_slot = _slot_from_ident(traced_ident) if traced_ident else None
                if not traced_slot:
                    return
                # The TRACER is now RUNNING the copied ability, even though its own set ability
                # stays Trace. `revealed_ability` is redirected to the `[of]` mon below and must
                # stay that way -- writing the copied ability onto the tracer's set would be a
                # fact it cannot have. But the copy is live for engine purposes, and Pressure is
                # the case that bites: a traced Pressure really does double-charge the foe's PP.
                tracer = self._target_belief(target_slot, target_ident)
                if tracer is not None:
                    self._running_ability[tracer.key] = str(primary)
                target_slot, target_ident = traced_slot, traced_ident
                traced_belief = self._target_belief(target_slot, target_ident)
                # Trace copies whatever the target is CURRENTLY running, which for a transformed
                # mon is its copy target's ability, not its own. Recording it as a certain reveal
                # writes a fact the mon cannot have -- a Ditto transformed into Claydol yields
                # `revealed_ability=Levitate` on Ditto, whose only pool ability is Limber, and the
                # set collapses to the inconsistent fallback. It widens rather than dropping the
                # true variant, so it is safe, but it is a wrong and sticky fact. The engine
                # already tracks the flag, so simply decline the reveal.
                if traced_belief is not None and traced_belief.transformed:
                    return
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
        twin._running_ability = dict(self._running_ability)
        twin._pressure_charge_counts = Counter(self._pressure_charge_counts)
        twin._in_residual_phase = self._in_residual_phase
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

    @property
    def pressure_charge_counts(self) -> dict[str, int]:
        """Reachability tallies for the Pressure branch of the PP ledger (see `__init__`)."""
        return dict(self._pressure_charge_counts)

    def _charge_move_use(
        self, belief: RevealedPokemonBelief, move_id: str, *, wire_target_slot: str = ""
    ) -> RevealedPokemonBelief:
        # Pressure on the OPPOSING active doubles PP spent, but only when a FOE is in the engine's
        # `pressureTargets` -- self-targeted Rest / Swords Dance are never pressured.
        #
        # Decided from the MOVE, never from the protocol line's target slot: the engine blanks that
        # slot for any move whose animation is suppressed, so it is not a usable signal. See
        # _NEVER_PRESSURED_POOL_MOVES.
        normalized_for_target = _normalize_identifier(move_id)
        foe_targeted = normalized_for_target not in _NEVER_PRESSURED_POOL_MOVES
        opposing = self._active_belief(_other_side(belief.showdown_slot))
        # The ability the opposing mon is currently RUNNING, which is not always the one it was
        # revealed with: Trace copies the target's ability onto the TRACER, and the tracer then
        # pressures for real. `revealed_ability` deliberately stays the tracer's own set ability
        # (Trace), so consulting only that missed every traced Pressure -- V3 measured Absol's
        # Shadow Ball at 22 believed vs 21 true, one of two uses charged single.
        opposing_ability = ""
        if opposing is not None:
            opposing_ability = _normalize_identifier(
                self._running_ability.get(opposing.key) or opposing.revealed_ability or ""
            )
        charge = 2 if foe_targeted and opposing_ability == "pressure" else 1
        # `wire_target_slot` is instrumentation ONLY -- it never feeds `charge`. It exists so the
        # differential can report how often the old wire-slot proxy would have disagreed.
        self._pressure_charge_counts["charges"] += 1
        if opposing_ability == "pressure":
            self._pressure_charge_counts["vs_pressure"] += 1
            if charge == 2:
                self._pressure_charge_counts["doubled"] += 1
                wire_says_foe = bool(wire_target_slot) and wire_target_slot != belief.showdown_slot
                if not wire_says_foe:
                    self._pressure_charge_counts["doubled_despite_blank_or_self_wire_slot"] += 1
            else:
                self._pressure_charge_counts["single_self_targeted_vs_pressure"] += 1
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
        # A traced ability does not survive leaving the field.
        self._running_ability.pop(belief.key, None)
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

    def _track_residual_phase(self, event_type: Optional[str], raw_line: Optional[str]) -> None:
        """Follow the sim's own action -> residual boundary, which is a BARE ``|`` protocol line.

        ``sim/battle.ts:2836`` -- ``case 'residual': this.add('')`` -- emits it at the top of the
        residual queue action, immediately before ``fieldEvent('Residual')`` runs every residual
        handler and then adds ``|upkeep``. It survives the parser intact: a bare line yields
        ``ShowdownPublicEvent(event_type='unknown', raw_line='|')`` (``_public_event_from_line``
        maps the empty type token to ``'unknown'``), verified live on the local env.

        See ``_ACTION_PHASE_EVENT_TYPES`` for why the marker from ``turnLoop`` does not need to be
        told apart from this one.
        """
        if raw_line is not None and raw_line.strip() == "|":
            self._in_residual_phase = True
            return
        if event_type == "turn" or event_type in _ACTION_PHASE_EVENT_TYPES:
            self._in_residual_phase = False

    def _apply_hp_observation(
        self, key: str, condition: Optional[str], *, raw_line: Optional[str]
    ) -> None:
        """Fold one HP line into the pre-item-slot snapshot, per ``_hp_snapshot_action``."""
        action = _hp_snapshot_action(raw_line, in_residual_phase=self._in_residual_phase)
        if action == _HP_SNAPSHOT_UPDATE:
            self._hp_after_actions[key] = _hp_fraction_from_condition(condition)
        elif action == _HP_SNAPSHOT_DISCARD:
            self._hp_after_actions[key] = None

    def _sweep_end_of_turn_non_procs(self) -> None:
        for side in ("p1", "p2"):
            belief = self._active_belief(side)
            if belief is None or belief.item_mutated or belief.revealed_item:
                continue
            hp_fraction = _hp_fraction_from_condition(belief.condition)
            if hp_fraction is None or hp_fraction <= 0.0:
                continue
            # Both rules below read ONE value: the HP this mon was at when its own item residual
            # slot (order 10 / subOrder 4) ran. ``_hp_snapshot_action`` maintains it; the ordering
            # facts and their reachability live there.
            #
            # ``None`` -- absent key or an explicitly discarded snapshot -- means NO EVIDENCE, and
            # both rules decline. The snapshot is also cleared per turn at ``|upkeep``, so a value
            # from an earlier turn cannot masquerade as this turn's.
            #
            # A mon at FULL HP when its slot runs needs no separate guard: a pre-slot heal to full
            # (Wish 7, or an action-phase Rest/Recover) UPDATES the snapshot to 1.0, and
            # ``hp_pre_residual < 1.0`` then declines on its own. A dedicated
            # ``_healed_to_full_this_turn`` set used to sit here doing that job a second time. It was
            # removed because it was both redundant and WRONG: it keyed off "healed to full at any
            # point this turn", not "was full when the slot ran", so it suppressed sound
            # eliminations -- Recover to full and then take a hit down to 180/267, or Wish to full
            # and then eat Sandstorm chip (field order 8, still before the slot) down to 255/272. In
            # both the mon was demonstrably below full at 10/4 with no heal line, which is exactly
            # the evidence this rule exists to use. It was also a surviving mutant: deleting its
            # clause left the whole suite green while three docstrings claimed to isolate it.
            #
            # The Octillery shape that motivated the original guard (565 violations / 250 games) is
            # still covered, by the snapshot rather than by a second set:
            #
            #   |-damage|p2a: Octillery|169/272
            #   |-heal|p2a: Octillery|272/272|[from] move: Wish|[wisher] Umbreon
            #   |upkeep
            #
            # Wish is pre-slot, so the snapshot ends at 1.0 and the rule declines. Octillery really
            # does hold Leftovers, and because every Octillery variant holds it the rule-out emptied
            # the candidate set; on a mixed-item species it would drop the true variant instead.
            #
            # An earlier version of this fix required the mon to END the turn below full. That proxy
            # is UNSOUND and is gone: the slot is at 10/4 while brn/psn/tox chip at 10/6, so a mon
            # can be full when Leftovers runs and below full at ``|upkeep``. It let the defect
            # survive in live data (21 violations / 400 games) on exactly that shape.
            #
            # This rule has now been wrong FOUR TIMES, each time because a fix was written as if it
            # had closed the family: the end-of-turn proxy, the stale snapshot, Liquid Ooze damage on
            # the drainer, and the untagged Leech Seed drain heal on the drainer. Three of the four
            # were the same mistake -- an HP change that lands outside the action phase read as
            # though it described the pre-slot HP. So the family is NOT declared closed here. What
            # changed is that the classifier's default flipped: an HP change it cannot place now
            # DISCARDS the snapshot instead of being assumed action-phase, so the next unenumerated
            # source costs narrowing rather than soundness. That is the only claim being made.
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
            # Pinch berries (Salac/Petaya/Liechi) fire at the RESIDUAL item slot in gen3 --
            # `data/mods/gen3/items.ts` sets `onUpdate: undefined` (no inherit) plus
            # `onResidualOrder: 10, onResidualSubOrder: 4`, i.e. the SAME slot as Leftovers. The
            # comment here used to say the opposite -- that they activate on an action-phase HP
            # drop and never on a residual crossing -- and that premise was backwards. It is what
            # made this rule eliminate the TRUE berry: any heal ordered before 10/4 (Wish 7, weather
            # field 8) lifts the mon back over the 25% line before the berry is ever offered, so its
            # silence is not evidence.
            #
            # The gate is therefore the same pre-slot HP snapshot the Leftovers rule uses, and the
            # correctness rests on ``_hp_snapshot_action`` classifying by PHASE first and gen3 order
            # second: pre-slot changes UPDATE the snapshot, at/after-slot changes leave it, and an
            # unclassifiable one discards it.
            # No snapshot => no evidence (conservative).
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


# Everything below classifies an HP change against ONE reference point: the gen3 item residual
# slot, order 10 / subOrder 4, where Leftovers (`data/mods/gen4/items.ts` leftovers) and the pinch
# berries (`data/mods/gen3/items.ts` salacberry/petayaberry/liechiberry) both fire. Both non-proc
# rules ask the same question -- "what HP was this mon at when its own item slot ran" -- so the only
# thing that matters about an HP line is whether it landed before that slot, at/after it, or at an
# UNDETERMINED position.
#
# Orders below are the gen3-EFFECTIVE ones, walked through the mod chain
# (`data/mods/gen3/scripts.ts` is `inherit: 'gen4'`). The shared `data/moves.ts` / `data/items.ts`
# numbers are gen5+ and different; citing them is the "generalized from the shared engine file"
# mistake the Hidden Power premise already made once.
#
#   Wish 7  <  weather field 8 (Sandstorm/Hail)
#     <  Ingrain 10/1  <  Rain Dish 10/3  <  [ Leftovers and pinch berries 10/4 ]
#     <  Leech Seed 10/5  <  brn/psn/tox 10/6  <  Nightmare 10/7  <  Curse 10/8
#     <  partiallytrapped 10/9  <  Future Sight 11 (unreachable: no pool set carries it)
#
# CAUTION, and this is why the lists below are not the whole story: that is NOT a total order over
# handlers. `Battle.comparePriority` (`sim/battle.ts:404-411`) sorts order, then priority, then
# SPEED DESCENDING, then subOrder -- so among the many order-10 handlers, speed dominates subOrder
# and the subOrder chain only holds WITHIN ONE POKEMON (same effect holder => same speed => subOrder
# decides). Every entry in the at/after list is an effect on the mon whose own item slot is in
# question, which is what makes its position guaranteed. A CROSS-POKEMON residual -- the Leech Seed
# drain heal on the drainer, Liquid Ooze damage on the drainer -- can land on either side of the
# other mon's slot depending on the speed tie, so it is not classifiable at all and gets the third
# treatment: the snapshot is DISCARDED and the turn yields no evidence.

# Residual sources that resolve strictly BEFORE the item slot, so their result IS the HP the item
# saw. Missing an entry here only costs narrowing (it degrades to "undetermined"), never soundness.
#
# Reachability in gen3 randbats, measured against the pool (220 species) rather than assumed --
# THREE OF THE FIVE ARE DEAD ENTRIES, kept for correctness-by-construction, and a reader has to be
# able to tell which. An earlier revision of this list deleted the reachability notes entirely,
# which left live and dead entries indistinguishable:
#   * Wish       -- LIVE. 16 pool species carry it.
#   * Sandstorm  -- LIVE. Reachable via Sand Stream, carried by all 15 Tyranitar sets (the pool's
#     only Sand Stream holder); no pool set carries the Sandstorm move itself.
#   * Hail       -- unreachable. No pool set carries Hail and no pool ability summons it.
#   * Rain Dish  -- unreachable. No pool set has the ability. Order 10/3, from
#     `data/mods/gen3/abilities.ts` raindish, which sets `onWeather: undefined, // no inherit` and
#     replaces it with `onResidualOrder: 10, onResidualSubOrder: 3` plus its own `onResidual`.
#     Neither gen4 nor gen5 overrides it, so the gen3 entry is what runs.
#
#     A previous revision of this comment "corrected" 10/3 to "order 8, because it is an `onWeather`
#     handler in `data/abilities.ts`". That was wrong, and it was wrong in the exact way this file
#     warns about three times: the shared file's `onWeather` form is gen5+, and the gen3 MOD replaces
#     it. The mechanical cause is worth naming because it is repeatable -- the lookup was written as
#     `grep gen4/abilities.ts && echo ... && grep gen3/abilities.ts`, and grep exits non-zero when it
#     finds nothing, so the FAILED gen4 lookup short-circuited the `&&` chain and the gen3 lookup
#     never ran. Empty output was then read as "no mod override". Never gate a mod-chain lookup on
#     `&&`: check every layer unconditionally and make a miss print something.
#
#     Behaviourally inert here (10/3 and 8 are both pre-slot, and the ability is pool-unreachable),
#     but it was presented as the corrected reading, which is how a wrong engine fact gets inherited.
#   * Ingrain    -- unreachable. No pool set carries the move.
_PRE_ITEM_SLOT_RESIDUAL_HP_TAGS = (
    "[from] move: Wish",
    "[from] Sandstorm",
    "[from] Hail",
    "[from] ability: Rain Dish",
    "[from] move: Ingrain",
)

# Residual sources on the SAME Pokemon whose subOrder puts them at or after the item slot, so the
# snapshot must survive them unchanged. Missing an entry here also only costs narrowing.
#
# Every one of these is a RESIDUAL-ONLY effect in gen3 -- none of them can produce an HP line during
# the action phase -- so the tag alone settles the classification and it is applied without
# consulting the phase. That is deliberate belt-and-braces: it keeps the at/after cases correct on
# any event stream that reached this engine with the phase markers stripped, and it is what the
# ``|`` marker is NOT relied on for.
#
#   * Leftovers 10/4        -- LIVE (the slot itself; also latches `_leftovers_healed_this_turn`).
#   * Leech Seed 10/5       -- LIVE. Damage on the SEEDED mon, i.e. same-Pokemon. 12 pool species
#     carry the move. (The paired drain HEAL on the drainer is cross-Pokemon; see below.)
#   * brn / psn / tox 10/6  -- LIVE. `data/mods/gen4/conditions.ts` brn/psn/tox. Toxic chip is
#     emitted as `[from] psn` (measured: 6,695 `[from] psn` lines and zero `[from] tox` lines over
#     400 games / 2 seeds), so two tags cover all three conditions. A third `"[from] tox"` entry was
#     dropped rather than kept as belt: it never matches, and as a bare substring it would also
#     match any future `[from] tox…` effect name, which is a way to be wrong for free.
#   * Nightmare 10/7        -- unreachable. `data/mods/gen4/moves.ts` nightmare sets 10/7; no pool
#     set carries the move.
#   * Curse 10/8            -- unreachable. `data/mods/gen4/moves.ts` curse sets 10/8, but only the
#     GHOST-type branch inflicts the residual, and none of the 5 pool Curse carriers (Dunsparce,
#     Miltank, Muk, Regirock, Snorlax) is Ghost -- they all get the stat-drop branch. The 6 pool
#     Ghost species never carry Curse.
#   * partiallytrapped 10/9 -- LIVE, barely. `data/mods/gen4/conditions.ts` partiallytrapped sets
#     10/9; Shuckle's Wrap is the pool's only source.
#
# An earlier version of this list said the sub-orders of Curse, Nightmare and Wrap "are not
# established here". They are, at the three sites cited above; the uncertainty was in the reading,
# not in the engine.
_AT_OR_AFTER_ITEM_SLOT_RESIDUAL_HP_TAGS = (
    "[from] item: Leftovers",
    "[from] Leech Seed",
    "[from] psn",
    "[from] brn",
    "[from] Nightmare",
    "[from] Curse",
    "[from] move: Wrap",
    "[from] partiallytrapped",
)

# Sources that are ACTION-PHASE ONLY in gen3, so the tag settles it the other way.
#
# Confusion self-damage is dealt from ``confusion.onBeforeMove`` (`data/mods/gen4/conditions.ts`,
# gen3 has no override), i.e. strictly inside move execution. It needs the exception because it is
# the one HP line in the format that can be the FIRST thing in a turn: a confused mon that hits
# itself emits ``|-activate|…|confusion`` and ``|-damage|…|[from] confusion`` and NO ``|move|`` line,
# so on that turn nothing has yet re-opened the action phase after ``turnLoop``'s bare marker.
# Measured against the raw protocol, with the POLICY NAMED in each case. The policy matters and an
# unqualified line count invites over-generalizing: the residual-phase share of HP lines moves from
# 66% to 59% between the two policies below, so "N lines checked" means different coverage under each.
# Zero classification disagreements and zero unpairable lines in every run:
#   * uniform-random-legal, 150 games seed 31337: 24,316 HP lines, 16,073 residual-phase;
#   * move-bias 0.75 (the gate's own policy), 150 games seed 31337: 18,364 lines, 10,872 residual;
#   * move-bias 0.75, 150 games seed 555001: 17,026 lines, 10,228 residual.
# (An earlier round measured 67,583 lines over 400 games / 2 seeds, uniform policy only.)
# Without this exception, exactly the confusion lines diverge -- 1 of 35,283 on seed 555001 in that
# earlier uniform run -- and the divergence direction is a discarded snapshot, i.e. declined
# evidence, not a wrong belief. Closed anyway so the differential can assert zero.
#
# The same runs establish that the residual-phase HP-line vocabulary is CLOSED in this format:
# `[from] item: Leftovers`, `[from] psn`, `[from] Sandstorm`, `[from] brn`,
# `[from] move: Wrap|[partiallytrapped]`, `[silent]`, `[from] move: Wish`, `[from] Leech Seed`.
# `[silent]` is in NO tuple and lands on the discard default deliberately -- it is the Leech Seed
# drain heal, whose position against the 10/4 slot is not determinable from the line, and
# discarding is the whole fix. So do NOT read the tuples as exhaustive over this vocabulary and
# "tighten" the default: that reintroduces the defect this classifier exists to remove. Every OTHER
# listed tag is handled explicitly; the default is what keeps the next addition safe rather than
# silent.
_ACTION_PHASE_ONLY_HP_TAGS = ("[from] confusion",)

# Sources whose position against the holder's own item slot is UNDETERMINABLE in either phase, so
# they always discard the snapshot.
#
# Liquid Ooze turns a drain into damage ON THE DRAINER (`data/mods/gen4/abilities.ts` liquidooze,
# `canOoze = ['drain', 'leechseed']`; gen3 inherits gen4), and the protocol line does not name which
# of the two it was:
#   * from an action-phase drain move it is action-phase, i.e. pre-slot;
#   * from the Leech Seed residual it is cross-Pokemon at order 10, where SPEED outranks subOrder, so
#     it can land on either side of the drainer's own 10/4 slot.
# Neither the tag nor the phase resolves it, so it is declined unconditionally. Pool-reachable via
# Swalot and Tentacruel. Found by the V1 sweep at 400 games, and it broke CONTAINMENT rather than
# merely widening:
#   |switch|p2a: Flygon|Flygon, L78, F|253/253      <- full when Leftovers ran, so no heal line
#   |-damage|p1a: Swalot|0 fnt|[from] Leech Seed|[of] p2a: Flygon
#   |-damage|p2a: Flygon|220/253|[from] ability: Liquid Ooze|[of] p1a: Swalot
# Flygon's true item IS Leftovers, and Flygon is a mixed-item species, so the rule-out dropped the
# true variant and left a confidently wrong single-candidate pin.
_UNORDERED_HP_TAGS = ("[from] ability: Liquid Ooze",)

# What to do with the pre-slot HP snapshot when an HP line arrives.
_HP_SNAPSHOT_UPDATE = "update"
_HP_SNAPSHOT_KEEP = "keep"
_HP_SNAPSHOT_DISCARD = "discard"


def _hp_snapshot_action(raw_line: Optional[str], *, in_residual_phase: bool) -> str:
    """How an HP line bears on the pre-item-slot HP snapshot.

    ``update`` -- the line landed before the item slot, so its new HP IS what the item saw.
    ``keep``   -- the line landed at or after the slot, so the snapshot still describes it.
    ``discard``-- the line's position relative to the slot is not determinable, so the turn yields
    no evidence for this mon. Declining is the only safe answer: the audit requires beliefs
    "degrade to absent, never to plausible", and a wrong snapshot rules out the TRUE item.

    The PHASE is what closes the classification, and it comes from the engine rather than from a tag
    list. Classifying by ``[from]`` tag alone required the at/after list to be exhaustive over every
    residual source that might ever emit an HP line -- an open-ended obligation, and it was already
    unmet: the Leech Seed drain heal on the DRAINER is emitted with ``[silent]`` and no ``[from]``
    tag at all (``sim/battle.ts:2293-2296``, ``case 'leechseed': case 'rest': this.add('-heal',
    target, target.getHealth, '[silent]')``), so it read as action-phase, overwrote the snapshot with
    a residual-phase HP, and ruled out the true Leftovers on a mon that was at full HP when its slot
    ran. The pre-fix magnitude -- 85 violations at 400 games on seed 31337 and 65 on 555001, all of
    them family 5 ``ruled_out_item`` / "true item 'leftovers' was ruled out" -- is the REVIEW's
    measurement of the pre-fix code, recorded here with that attribution because this round ran only
    the post-fix sweeps, which are 0 on both seeds.

    A blanket "``[silent]`` means residual" rule does not work either, because that same engine
    branch emits ``[silent]`` for REST, which is an ACTION-phase heal and is carried by 46 pool
    species. The phase separates the two exactly, with no tag involved -- and, more to the point, an
    unenumerated residual source now lands on ``discard`` instead of silently reading as
    action-phase, so the list no longer has to be exhaustive to be safe.
    """
    if not raw_line:
        # No line text, so there is no tag to read -- but the PHASE is still known, and that is
        # enough on the action side: everything there precedes every residual regardless of source.
        # In the residual phase nothing is left to classify with, so it declines.
        #
        # This branch used to return ``update`` unconditionally, which made the one case with the
        # LEAST information the most permissive -- the opposite of the rule the rest of this function
        # exists to enforce, and the same shape as the four defects that preceded it. It was also a
        # surviving mutant: flipping it to ``discard`` left all 90 tests green. Not reachable from the
        # parser (``ShowdownPublicEvent.raw_line`` is always populated), but ``ingest_event`` also
        # accepts plain mappings, so it is reachable by construction.
        return _HP_SNAPSHOT_DISCARD if in_residual_phase else _HP_SNAPSHOT_UPDATE
    if any(tag in raw_line for tag in _UNORDERED_HP_TAGS):
        return _HP_SNAPSHOT_DISCARD
    if any(tag in raw_line for tag in _ACTION_PHASE_ONLY_HP_TAGS):
        return _HP_SNAPSHOT_UPDATE
    if any(tag in raw_line for tag in _AT_OR_AFTER_ITEM_SLOT_RESIDUAL_HP_TAGS):
        # Residual-only effects, so the tag settles it without the phase (see that tuple's note).
        return _HP_SNAPSHOT_KEEP
    if not in_residual_phase:
        # Action phase: recoil, drain, Spikes chip on entry, Rest, Recover, direct damage. All of it
        # precedes every residual, so all of it updates the snapshot.
        return _HP_SNAPSHOT_UPDATE
    if any(tag in raw_line for tag in _PRE_ITEM_SLOT_RESIDUAL_HP_TAGS):
        return _HP_SNAPSHOT_UPDATE
    # Residual phase, unrecognized source. Reached today by the ``[silent]`` Leech Seed drain heal on
    # the drainer -- cross-Pokemon at order 10, where speed outranks subOrder, so it has no
    # determinable position against the drainer's own 10/4 slot. It is also the landing spot for any
    # residual source not yet enumerated, which is the point: the default is to decline evidence
    # rather than to invent a phase for it.
    return _HP_SNAPSHOT_DISCARD


# Event types that can only occur in the ACTION phase, used to close the residual phase.
#
# The bare ``|`` marker is emitted from three sites in ``sim/battle.ts``: ``turnLoop`` (2963, always
# immediately followed by ``|t:|``), the ``residual`` queue action (2836, the one that matters), and
# ``win`` (1526). The parser DROPS ``|t:|`` lines before the belief engine sees them
# (``showdown.py`` ``_feed_line``: they are wall-clock noise that would make replay-from-root
# observations differ), so the two markers are indistinguishable at this layer by lookahead. They do
# not need to be: ``turnLoop``'s marker is followed by the turn's actions, and every action opens
# with one of these lines, so the first of them re-opens the action phase. Differentially verified
# against the raw protocol -- where ``|t:|`` IS visible and the phase is therefore exactly known --
# over 400 games on two seeds: zero disagreements on any HP line.
_ACTION_PHASE_EVENT_TYPES = frozenset({"move", "switch", "drag", "replace", "cant"})


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


# Pool moves that Pressure NEVER charges double, derived from the ENGINE'S TARGET rather than from
# the protocol line.
#
# The wire slot cannot be used for this at all: `sim/battle.ts:3155-3159` BLANKS it whenever the
# move's animation is suppressed --
#     } else if (args.includes('[still]')) {
#         // If no animation plays, the target should never be known
#         const parts = this.log[this.lastMoveLine].split('|'); parts[4] = '';
# -- so a foe-targeted status move reads as untargeted and silently loses its Pressure double.
#
# Size of the class, re-runnable rather than quoted: `scripts/blank_target_slot_census.py`.
#     PYTHONPATH=src .venv/bin/python scripts/blank_target_slot_census.py 60 999
# 60 games, BLANK/total, foe-targeted rows only (self-targeted moves blank too -- Recover 31/113,
# Refresh 31/39 -- but they are in this set and were never pressured anyway):
#
#     seed 999    toxic 70/344  spikes 39/78  whirlwind 26/38  thunderwave 12/50
#                 solarbeam 9/13  hypnosis 9/40  willowisp 9/26  leechseed 7/28
#     seed 4711   toxic 28/286  solarbeam 10/11  encore 7/24  spikes 6/35  leechseed 5/23
#
# An earlier fix here enumerated the four `target: "all"` moves and called that "the" broken class;
# toxic alone blanks several times more often than raindance does. (The first version of this
# comment carried a third table that no re-run reproduces -- it was measured once, early, and then
# copied forward. Numbers that cannot be re-run do not belong in a comment; hence the script.)
#
# The engine rule (`sim/pokemon.ts:792` `getMoveTargets`, resolving at `:853-860`, not overridden
# anywhere in the gen3 chain): `pressureTargets = targets`, zeroed only for `foeSide` and re-filled
# by `mustpressure`. So a foe is present for every target kind
# EXCEPT `self` and the ally-side ones. Enumerated over all 125 pool moves from
# `data/random-battles/gen3/sets.json`: 23 `self` + 2 `allyTeam`, everything else pressured
# (`all` 4, `allAdjacent` 3, `allAdjacentFoes` 3, `any` 2, `foeSide` 1, `normal` 85, `scripted` 2).
#
# Targets resolved along the MOD CHAIN, not read off shared `data/moves.ts`, because two pool moves
# declare theirs in a mod: `surf` is `allAdjacentFoes` in `data/mods/gen3/moves.ts` where the shared
# file says `allAdjacent`, and `curse` is re-declared `normal` in `data/mods/gen7/moves.ts`. Neither
# changes which side of this set the move lands on -- both still reach a foe -- but the first
# derivation here did read only the shared file, and "it happened not to matter this time" is not a
# method.
#
# Curse is the one move whose dex target lies: `sim/pokemon.ts:998-1004` retargets it to
# `nonGhostTarget` (self) unless the user is Ghost, and all five pool carriers -- Muk (Poison),
# Snorlax, Dunsparce, Miltank (Normal), Regirock (Rock) -- are non-Ghost, so in this pool it is
# always self-targeted. Listed with that reachability note rather than resolved from the user's
# type, which the belief engine does not carry.
_NEVER_PRESSURED_POOL_MOVES = frozenset({
    # target: "self"
    "agility", "batonpass", "bellydrum", "bulkup", "calmmind", "destinybond", "dragondance",
    "endure", "milkdrink", "moonlight", "morningsun", "protect", "recover", "refresh", "rest",
    "slackoff", "sleeptalk", "softboiled", "substitute", "swordsdance", "synthesis", "tailglow",
    "wish",
    # target: "allyTeam"
    "aromatherapy", "healbell",
    # retargeted to self for every pool carrier (see above)
    "curse",
})


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
