"""Fixture-only one-turn outcome comparison: poke-engine vs. Showdown.

This is a narrow, optional diagnostic for the poke-engine evaluation spike (step 4
of ``docs/poke_engine_assessment.md``). It builds a :class:`~pokezero.poke_engine_adapter.BattleSpec`
from a Showdown one-turn fixture **result** (using the real opening request data --
species/level from ``details``, hp/maxhp from ``condition``, stats from
``side.pokemon[].stats``, moves from the request, abilities/items when present, and
types from the existing Showdown dex loader), enumerates the engine's instruction
branches for the same two move choices, applies each branch to read its final
active HP, and compares those outcomes against the HP Showdown actually produced.

It is **honest feasibility evidence, not an adoption gate.** As of poke-engine
0.0.47 the engine's damage numbers differ slightly from Showdown's seeded turn, so
:func:`compare_one_turn_outcome` reports ``matched=False`` for the curated
Charmander/Squirtle damage smoke. That mismatch is surfaced, not hidden.

When a mismatch is found, :func:`build_one_turn_damage_diagnostic` gathers structured,
serializable evidence about *where* it lives -- observed vs. engine per-side damage
deltas, the Python request-derived active-state summary fed into state construction
(stats/hp/moves from the Showdown request, types from the dex), an active-state summary
read back from the *built* ``poke_engine.State`` and compared field-by-field against the
request-derived spec, and the engine's direct ``calculate_damage`` output when the
binding exposes it -- and reports a conservative ``likely_mismatch_surface`` without
claiming a root cause it has not proven. The request-derived summaries describe the
*input* to ``build_poke_engine_state``; the engine-state summaries describe what the
engine actually built from it, so a matching comparison narrows the mismatch off the
spec->engine state-translation path. When both sides' comparisons match, the surface
narrows to ``engine damage/data path``; if either mismatches or cannot be inspected, the
broad ``engine damage/data or state-translation path`` stands with notes explaining why.
Either way the *exact* cause is left explicitly unresolved.

Scope is **singles, move-vs-move only.** The module is optional and lazy: importing
it never requires the native wheel (the real engine is imported only when a
comparison actually runs and no fake ``module`` is supplied). It is deliberately
disconnected from rollout, training, search, benchmarks, and self-play.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
import re
from typing import TYPE_CHECKING, Any, Mapping, Sequence

from .dex import normalize_id
from .poke_engine_adapter import (
    TYPELESS,
    BattleSpec,
    MoveSpec,
    PokemonSpec,
    SideSpec,
    build_poke_engine_state,
)
from .poke_engine_backend import require_poke_engine

if TYPE_CHECKING:
    from .dex import ShowdownDex
    from .local_showdown import LocalShowdownConfig
    from .showdown_fixture import OneTurnFixtureResult

# The two seats of a singles battle, in (side_one, side_two) == (p1, p2) order.
SIDE_PLAYER_IDS = ("p1", "p2")
# Active-slot prefixes Showdown uses in the omniscient protocol for each seat.
ACTIVE_SLOT_PREFIXES = ("p1a", "p2a")
# Protocol tags that report a (possibly new) HP value for a slot.
_HP_BEARING_TAGS = ("switch", "drag", "replace", "-damage", "-heal", "-sethp")
# Default PP for moves whose request entry omits it; PP is irrelevant to a single
# turn's damage outcome, so a fixed value keeps the spec construction simple.
_DEFAULT_PP = 32

# Conservative ``likely_mismatch_surface`` values. ``NARROW`` is only claimed once
# the built engine state has been inspected and matches the request-derived spec
# for every compared field; otherwise the broad surface stands.
NO_MISMATCH_SURFACE = "none"
NARROW_MISMATCH_SURFACE = "engine damage/data path"
BROAD_MISMATCH_SURFACE = "engine damage/data or state-translation path"

# Active-Pokemon fields compared between the request-derived spec and the built
# engine state, in display order.
ENGINE_STATE_COMPARED_FIELDS = (
    "species",
    "level",
    "hp",
    "maxhp",
    "types",
    "ability",
    "item",
    "attack",
    "defense",
    "special_attack",
    "special_defense",
    "speed",
    "moves",
)
# Marker recorded as the engine-side value of a field the engine state could not
# expose, so an unreadable field surfaces as an explicit mismatch (never a crash).
ENGINE_FIELD_UNREAD = "<unread>"


@dataclass(frozen=True)
class EngineBranchOutcome:
    """One poke-engine instruction branch's final active HP and metadata."""

    percentage: float
    # Final active HP after applying this branch, in (side_one, side_two) order.
    final_hp: tuple[int, int]
    description: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "percentage": self.percentage,
            "final_hp": list(self.final_hp),
            "description": self.description,
        }


@dataclass(frozen=True)
class OutcomeComparison:
    """Result of comparing Showdown's seeded turn to poke-engine branches.

    ``supported`` is ``False`` only when the engine outcomes could not be
    enumerated at all; in that case ``reason`` explains why and the engine fields
    are empty. ``matched`` means the exact HP tuple Showdown produced appears among
    the engine's branch outcomes -- the bar for true one-turn equivalence on this
    fixture. A ``supported=True, matched=False`` result is the honest "engine ran
    but disagrees" signal, not a failure to run.
    """

    supported: bool
    matched: bool
    showdown_final_hp: tuple[int, int] | None
    engine_final_hp_outcomes: tuple[EngineBranchOutcome, ...]
    p1_move: str | None = None
    p2_move: str | None = None
    reason: str | None = None
    notes: tuple[str, ...] = field(default_factory=tuple)

    def engine_final_hp_tuples(self) -> tuple[tuple[int, int], ...]:
        """Distinct final HP tuples across engine branches, in first-seen order."""

        seen: list[tuple[int, int]] = []
        for outcome in self.engine_final_hp_outcomes:
            if outcome.final_hp not in seen:
                seen.append(outcome.final_hp)
        return tuple(seen)

    def to_dict(self) -> dict[str, Any]:
        return {
            "supported": self.supported,
            "matched": self.matched,
            "showdown_final_hp": list(self.showdown_final_hp) if self.showdown_final_hp is not None else None,
            "engine_final_hp_outcomes": [outcome.to_dict() for outcome in self.engine_final_hp_outcomes],
            "engine_final_hp_tuples": [list(t) for t in self.engine_final_hp_tuples()],
            "p1_move": self.p1_move,
            "p2_move": self.p2_move,
            "reason": self.reason,
            "notes": list(self.notes),
        }

    def summary(self) -> str:
        if not self.supported:
            return f"outcome comparison UNSUPPORTED: {self.reason or 'unknown reason'}"
        status = "MATCH" if self.matched else "MISMATCH"
        return (
            f"one-turn outcome {status}: showdown {self.showdown_final_hp} vs. "
            f"engine {list(self.engine_final_hp_tuples())} "
            f"({len(self.engine_final_hp_outcomes)} branches)"
        )


# --- Showdown request -> BattleSpec ---------------------------------------


def parse_details(details: str) -> tuple[str, int]:
    """Parse a Showdown ``details`` string into ``(species_id, level)``.

    ``details`` looks like ``"Charmander, M"``, ``"Charmander, L78"``, or
    ``"Charmander, L50, M"``. The species is the first comma-separated field; the
    level is the ``L<n>`` token if present, defaulting to 100 (Showdown omits the
    ``L`` marker at level 100).
    """

    if not isinstance(details, str) or not details.strip():
        raise ValueError(f"details must be a non-empty string, got {details!r}")
    parts = [part.strip() for part in details.split(",")]
    species = normalize_id(parts[0])
    if not species:
        raise ValueError(f"could not parse species from details {details!r}")
    level = 100
    for token in parts[1:]:
        match = re.fullmatch(r"[Ll](\d+)", token)
        if match:
            level = int(match.group(1))
            break
    return species, level


def parse_condition(condition: str) -> tuple[int, int | None]:
    """Parse a Showdown ``condition`` string into ``(hp, maxhp_or_None)``.

    Handles ``"219/219"``, ``"127/219 par"``, ``"0 fnt"``, and bare ``"0"``. A
    fainted/HP-only condition carries no max, so ``maxhp`` is ``None`` there.
    """

    if not isinstance(condition, str) or not condition.strip():
        raise ValueError(f"condition must be a non-empty string, got {condition!r}")
    head = condition.strip().split()[0]
    if "/" in head:
        hp_text, max_text = head.split("/", 1)
        return int(hp_text), int(max_text)
    return int(head), None


def _member_moves(member: Mapping[str, Any], active_row: Mapping[str, Any] | None) -> tuple[MoveSpec, ...]:
    """Move slots for a party member.

    Prefers the active row's move list (which carries pp) for the active Pokemon;
    otherwise falls back to the bench member's ``moves`` id list with default pp.
    """

    if active_row is not None:
        rows = active_row.get("moves")
        if isinstance(rows, list) and rows:
            specs = []
            for row in rows:
                if not isinstance(row, Mapping):
                    continue
                move_id = _request_move_id(row)
                pp = row.get("pp")
                specs.append(MoveSpec(id=move_id, pp=int(pp) if isinstance(pp, int) else _DEFAULT_PP))
            if specs:
                return tuple(specs)
    move_ids = member.get("moves")
    if not isinstance(move_ids, list) or not move_ids:
        raise ValueError(f"party member {member.get('ident')!r} has no usable move list")
    return tuple(MoveSpec(id=normalize_id(move_id), pp=_DEFAULT_PP) for move_id in move_ids)


def _request_move_id(row: Mapping[str, Any]) -> str:
    for key in ("id", "move"):
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            return normalize_id(value)
    raise ValueError(f"request move is missing an id/move name: {row!r}")


def _stat(stats: Mapping[str, Any], key: str, ident: Any) -> int:
    value = stats.get(key)
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"party member {ident!r} is missing integer stat {key!r}")
    return value


def pokemon_spec_from_request_member(
    member: Mapping[str, Any],
    dex: "ShowdownDex",
    *,
    active_row: Mapping[str, Any] | None = None,
) -> PokemonSpec:
    """Build a :class:`PokemonSpec` from one Showdown request party member.

    Pulls species/level from ``details``, hp/maxhp from ``condition``, the five
    battle stats from ``stats``, moves from the request, ability from
    ``baseAbility``/``ability``, item from ``item`` (empty string -> none), and
    types from the Showdown dex (lowercased to poke-engine id style).
    """

    if not isinstance(member, Mapping):
        raise TypeError(f"member must be a mapping, got {type(member).__name__}")
    ident = member.get("ident")
    species, level = parse_details(str(member.get("details", "")))
    hp, maxhp = parse_condition(str(member.get("condition", "")))
    if maxhp is None:
        raise ValueError(f"party member {ident!r} condition has no max HP to build a spec from")

    stats = member.get("stats")
    if not isinstance(stats, Mapping):
        raise ValueError(f"party member {ident!r} is missing a 'stats' block")

    species_info = dex.species_info(species)
    if species_info is None or not species_info.types:
        raise ValueError(f"Showdown dex has no types for species {species!r}")
    types = tuple(type_name.lower() for type_name in species_info.types)

    ability = member.get("baseAbility") or member.get("ability")
    item = member.get("item")
    return PokemonSpec(
        id=species,
        level=level,
        types=types,
        hp=hp,
        maxhp=maxhp,
        attack=_stat(stats, "atk", ident),
        defense=_stat(stats, "def", ident),
        special_attack=_stat(stats, "spa", ident),
        special_defense=_stat(stats, "spd", ident),
        speed=_stat(stats, "spe", ident),
        ability=normalize_id(ability) if ability else None,
        item=normalize_id(item) if item else None,
        moves=_member_moves(member, active_row),
    )


def side_spec_from_request(request: Mapping[str, Any], dex: "ShowdownDex") -> SideSpec:
    """Build a singles :class:`SideSpec` from one seat's Showdown request payload."""

    if not isinstance(request, Mapping):
        raise TypeError(f"request must be a mapping, got {type(request).__name__}")
    side = request.get("side")
    pokemon = side.get("pokemon") if isinstance(side, Mapping) else None
    if not isinstance(pokemon, list) or not pokemon:
        raise ValueError("request side has no pokemon list")

    active_row = _active_row(request)
    active_index = 0
    for index, member in enumerate(pokemon):
        if isinstance(member, Mapping) and member.get("active"):
            active_index = index
            break

    specs = []
    for index, member in enumerate(pokemon):
        # Only the active member gets the pp-bearing active row; bench members fall
        # back to their own move-id list.
        row = active_row if index == active_index else None
        specs.append(pokemon_spec_from_request_member(member, dex, active_row=row))
    return SideSpec(pokemon=tuple(specs), active_index=active_index)


def build_battle_spec_from_result(result: "OneTurnFixtureResult", dex: "ShowdownDex") -> BattleSpec:
    """Build a :class:`BattleSpec` from a one-turn fixture result's opening requests."""

    if result.p1_request is None or result.p2_request is None:
        raise ValueError("fixture result is missing an opening request for one or both seats")
    return BattleSpec(
        side_one=side_spec_from_request(result.p1_request, dex),
        side_two=side_spec_from_request(result.p2_request, dex),
    )


def _active_row(request: Mapping[str, Any]) -> Mapping[str, Any] | None:
    active_rows = request.get("active")
    if not isinstance(active_rows, list) or not active_rows:
        return None
    if len(active_rows) > 1:
        raise ValueError("singles only: request carries more than one active slot")
    row = active_rows[0]
    return row if isinstance(row, Mapping) else None


# --- choice -> engine move id ---------------------------------------------


def engine_move_for_choice(request: Mapping[str, Any] | None, choice: str) -> str:
    """Resolve a Showdown ``"move N"``/``"move <name>"`` choice to a poke-engine move id.

    A numeric index selects from the request's active move list (1-based, as the
    Showdown client uses); a name is normalized directly. Only move choices are
    supported here -- switches raise ``ValueError`` since this comparator is
    move-vs-move only.
    """

    if not isinstance(choice, str) or not choice.strip():
        raise ValueError(f"choice must be a non-empty string, got {choice!r}")
    tokens = choice.strip().split()
    if tokens[0] != "move":
        raise ValueError(f"only 'move' choices are supported by this comparator, got {choice!r}")
    if len(tokens) < 2:
        raise ValueError(f"move choice {choice!r} is missing a move selector")
    selector = tokens[1]
    if selector.isdigit():
        active_row = _active_row(request) if request is not None else None
        moves = active_row.get("moves") if isinstance(active_row, Mapping) else None
        if not isinstance(moves, list) or not moves:
            raise ValueError(f"cannot resolve numeric move choice {choice!r}: no active move list")
        index = int(selector) - 1
        if not 0 <= index < len(moves):
            raise ValueError(f"move choice {choice!r} is out of range for {len(moves)} moves")
        return _request_move_id(moves[index])
    # A named choice is the remainder after "move" (Showdown drops spaces/case on the id).
    move_id = normalize_id(" ".join(tokens[1:]))
    active_row = _active_row(request) if request is not None else None
    moves = active_row.get("moves") if isinstance(active_row, Mapping) else None
    if isinstance(moves, list) and moves:
        legal = {_request_move_id(row) for row in moves if isinstance(row, Mapping)}
        if move_id not in legal:
            raise ValueError(f"move choice {choice!r} is not present in the active move list")
    return move_id


# --- engine branch enumeration --------------------------------------------


def _active_hp(state: Any, side_attr: str) -> int:
    side = getattr(state, side_attr)
    return int(side.pokemon[int(side.active_index)].hp)


def _branch_description(branch: Any) -> str:
    instruction_list = getattr(branch, "instruction_list", None)
    if instruction_list is not None:
        return str(instruction_list)
    return repr(branch)


def enumerate_engine_outcomes(
    engine: Any,
    state: Any,
    move_one: str,
    move_two: str,
) -> tuple[EngineBranchOutcome, ...]:
    """Enumerate engine instruction branches and their final active HP.

    Each branch is applied to the *original* ``state`` via ``apply_instructions``
    (which returns a fresh state and leaves ``state`` untouched), and the resulting
    active HP for both seats is read off the returned state.
    """

    branches = tuple(engine.generate_instructions(state, move_one, move_two))
    outcomes = []
    for branch in branches:
        after = state.apply_instructions(branch)
        outcomes.append(
            EngineBranchOutcome(
                percentage=float(getattr(branch, "percentage", 0.0)),
                final_hp=(_active_hp(after, "side_one"), _active_hp(after, "side_two")),
                description=_branch_description(branch),
            )
        )
    return tuple(outcomes)


# --- observed Showdown outcome --------------------------------------------


def observed_final_active_hp(result: "OneTurnFixtureResult") -> tuple[int, int]:
    """Parse the final active HP tuple ``(p1, p2)`` from the omniscient protocol.

    Tracks the most recent HP value reported for each seat's active slot
    (``p1a``/``p2a``) across HP-bearing protocol tags, and treats a ``|faint|`` on
    the slot as HP 0. Raises if either seat never reports an HP value.
    """

    latest: dict[str, int] = {}
    for line in result.protocol_lines:
        fields = line.split("|")
        if len(fields) < 2:
            continue
        tag = fields[1]
        if tag == "faint":
            slot = fields[2] if len(fields) > 2 else ""
            seat = _seat_for_slot(slot)
            if seat is not None:
                latest[seat] = 0
            continue
        if tag not in _HP_BEARING_TAGS:
            continue
        seat = _seat_for_slot(fields[2])
        if seat is None:
            continue
        hp_index = 4 if tag in {"switch", "drag", "replace"} else 3
        if len(fields) <= hp_index:
            continue
        hp_value = _parse_hp(fields[hp_index])
        if hp_value is not None:
            latest[seat] = hp_value

    missing = [seat for seat in SIDE_PLAYER_IDS if seat not in latest]
    if missing:
        raise ValueError(f"protocol never reported active HP for {missing}")
    return latest["p1"], latest["p2"]


def _seat_for_slot(slot: str) -> str | None:
    text = str(slot).strip()
    for prefix, seat in zip(ACTIVE_SLOT_PREFIXES, SIDE_PLAYER_IDS):
        if text.startswith(prefix + ":") or text == prefix:
            return seat
    return None


def _parse_hp(token: str) -> int | None:
    head = str(token).strip().split()[0] if str(token).strip() else ""
    if not head:
        return None
    number = head.split("/", 1)[0]
    try:
        return int(number)
    except ValueError:
        return None


# --- comparison ------------------------------------------------------------


def compare_outcomes(
    showdown_final_hp: tuple[int, int],
    engine_outcomes: Sequence[EngineBranchOutcome],
    *,
    p1_move: str | None = None,
    p2_move: str | None = None,
    notes: Sequence[str] = (),
) -> OutcomeComparison:
    """Compare an observed Showdown HP tuple against enumerated engine outcomes.

    Pure matching: ``matched`` is true iff the exact Showdown HP tuple appears among
    the engine branches' final HP tuples.
    """

    engine_outcomes = tuple(engine_outcomes)
    engine_tuples = {outcome.final_hp for outcome in engine_outcomes}
    return OutcomeComparison(
        supported=True,
        matched=tuple(showdown_final_hp) in engine_tuples,
        showdown_final_hp=tuple(showdown_final_hp),
        engine_final_hp_outcomes=engine_outcomes,
        p1_move=p1_move,
        p2_move=p2_move,
        notes=tuple(notes),
    )


def compare_one_turn_outcome(
    result: "OneTurnFixtureResult",
    dex: "ShowdownDex",
    *,
    module: Any | None = None,
    p1_choice: str | None = None,
    p2_choice: str | None = None,
) -> OutcomeComparison:
    """Compare a Showdown one-turn result against poke-engine branch outcomes.

    Builds a :class:`BattleSpec` from the fixture's opening requests, resolves the
    two move choices (defaulting to the choices the fixture actually submitted),
    enumerates engine branches, and compares final active HP. Pass a fake
    ``module`` to keep tests off the native wheel; ``None`` loads the engine lazily.
    """

    p1_choice = p1_choice if p1_choice is not None else result.choices.get("p1")
    p2_choice = p2_choice if p2_choice is not None else result.choices.get("p2")
    if not p1_choice or not p2_choice:
        raise ValueError("both p1 and p2 choices are required to compare outcomes")

    move_one = engine_move_for_choice(result.p1_request, p1_choice)
    move_two = engine_move_for_choice(result.p2_request, p2_choice)
    observed = observed_final_active_hp(result)

    engine = require_poke_engine() if module is None else module
    spec = build_battle_spec_from_result(result, dex)
    state = build_poke_engine_state(spec, module=engine)
    engine_outcomes = enumerate_engine_outcomes(engine, state, move_one, move_two)

    return compare_outcomes(
        observed,
        engine_outcomes,
        p1_move=move_one,
        p2_move=move_two,
    )


# --- damage-mismatch diagnostic -------------------------------------------
#
# The comparator above answers "does any engine branch reproduce Showdown's exact
# HP?" (no, on 0.0.47). The diagnostic below answers the *next* question -- given
# a mismatch, what surface is it on? It gathers structured, serializable evidence
# (request-derived state summaries, observed vs. engine per-side damage deltas, and
# the engine's direct calculate_damage output when the binding exposes it) so the
# difference can be narrowed without guessing. It is deliberately conservative: it
# reports where the evidence points, not a root cause it has not proven.


@dataclass(frozen=True)
class ActiveStateSummary:
    """Summary of one seat's active Pokemon as fed into state construction.

    Fields come from the Python :class:`SideSpec` the comparator builds: stats, hp,
    level, ability, item, and moves are taken from the Showdown opening request, and
    types are dex-derived. This is the *input* to ``build_poke_engine_state``. Whether
    the engine actually built a faithful state from it is checked separately by
    :func:`engine_active_state_summary` / :func:`compare_engine_active_state`; this
    summary on its own says nothing about the engine's damage math.
    """

    species: str
    level: int
    hp: int
    maxhp: int
    types: tuple[str, ...]
    ability: str | None
    item: str | None
    attack: int
    defense: int
    special_attack: int
    special_defense: int
    speed: int
    moves: tuple[tuple[str, int], ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "species": self.species,
            "level": self.level,
            "hp": self.hp,
            "maxhp": self.maxhp,
            "types": list(self.types),
            "ability": self.ability,
            "item": self.item,
            "attack": self.attack,
            "defense": self.defense,
            "special_attack": self.special_attack,
            "special_defense": self.special_defense,
            "speed": self.speed,
            "moves": [{"id": move_id, "pp": pp} for move_id, pp in self.moves],
        }


@dataclass(frozen=True)
class EngineActiveStateSummary:
    """One seat's active Pokemon read back from a *built* ``poke_engine.State``.

    Unlike :class:`ActiveStateSummary` (which summarizes the request-derived *input*
    to ``build_poke_engine_state``), this reads the engine object that construction
    actually produced. Values are normalized for comparison against the request
    summary: ids are lower-cased, the Gen 3 ``typeless`` padding slot is dropped from
    ``types``, and an ``ability``/``item`` of ``"none"`` (the engine's no-value
    sentinel) becomes ``None``. Any field the engine could not expose is left ``None``
    and named in ``missing_fields`` so it surfaces as an explicit comparison mismatch
    rather than a crash.
    """

    species: str | None
    level: int | None
    hp: int | None
    maxhp: int | None
    types: tuple[str, ...] | None
    ability: str | None
    item: str | None
    attack: int | None
    defense: int | None
    special_attack: int | None
    special_defense: int | None
    speed: int | None
    moves: tuple[tuple[str, int], ...] | None
    missing_fields: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "species": self.species,
            "level": self.level,
            "hp": self.hp,
            "maxhp": self.maxhp,
            "types": list(self.types) if self.types is not None else None,
            "ability": self.ability,
            "item": self.item,
            "attack": self.attack,
            "defense": self.defense,
            "special_attack": self.special_attack,
            "special_defense": self.special_defense,
            "speed": self.speed,
            "moves": (
                [{"id": move_id, "pp": pp} for move_id, pp in self.moves]
                if self.moves is not None
                else None
            ),
            "missing_fields": list(self.missing_fields),
        }


@dataclass(frozen=True)
class FieldMismatch:
    """One field that differs between the request-derived spec and the engine state."""

    field: str
    request_value: Any
    engine_value: Any

    def to_dict(self) -> dict[str, Any]:
        return {
            "field": self.field,
            "request": _jsonable_value(self.request_value),
            "engine": _jsonable_value(self.engine_value),
        }


@dataclass(frozen=True)
class ActiveStateComparison:
    """Field-level comparison of one seat's request spec vs. its built engine state.

    ``matched`` is ``True`` only when every compared field agrees. ``mismatches``
    lists each disagreeing field (an unreadable engine field is reported with the
    :data:`ENGINE_FIELD_UNREAD` marker, never silently skipped). ``reason`` is set
    when the engine state could not be fully inspected.
    """

    side: str
    matched: bool
    mismatches: tuple[FieldMismatch, ...]
    request_summary: ActiveStateSummary | None
    engine_summary: EngineActiveStateSummary | None
    reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "side": self.side,
            "matched": self.matched,
            "mismatches": [mismatch.to_dict() for mismatch in self.mismatches],
            "request_summary": self.request_summary.to_dict() if self.request_summary is not None else None,
            "engine_summary": self.engine_summary.to_dict() if self.engine_summary is not None else None,
            "reason": self.reason,
        }

    def summary(self) -> str:
        if self.matched:
            return f"{self.side} engine-state MATCH (all {len(ENGINE_STATE_COMPARED_FIELDS)} fields)"
        fields = ", ".join(mismatch.field for mismatch in self.mismatches)
        return f"{self.side} engine-state MISMATCH on: {fields}"


@dataclass(frozen=True)
class ObservedDamage:
    """Showdown's observed per-side damage, derived from the opening request HP."""

    opening_hp: tuple[int, int]
    final_hp: tuple[int, int]
    deltas: tuple[int, int]

    def to_dict(self) -> dict[str, Any]:
        return {
            "opening_hp": list(self.opening_hp),
            "final_hp": list(self.final_hp),
            "deltas": list(self.deltas),
        }


@dataclass(frozen=True)
class EngineBranchDamageSummary:
    """One engine branch's final HP plus its per-side damage deltas from opening HP."""

    percentage: float
    final_hp: tuple[int, int]
    deltas: tuple[int, int]
    description: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "percentage": self.percentage,
            "final_hp": list(self.final_hp),
            "deltas": list(self.deltas),
            "description": self.description,
        }


@dataclass(frozen=True)
class DirectCalculateDamageDiagnostic:
    """Direct ``calculate_damage`` output for both turn orders, or why it is unavailable.

    ``supported=False`` is the expected, non-fatal outcome when the binding lacks
    ``calculate_damage``, raises, or returns a shape we cannot serialize; ``reason``
    then explains which. ``supported=True`` carries the coerced JSON-able output for
    ``side_one_moves_first=True`` and ``=False``.
    """

    supported: bool
    output_side_one_first: Any | None
    output_side_two_first: Any | None
    reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "supported": self.supported,
            "side_one_moves_first": self.output_side_one_first,
            "side_two_moves_first": self.output_side_two_first,
            "reason": self.reason,
        }


@dataclass(frozen=True)
class OneTurnDamageDiagnostic:
    """Serializable evidence bundle for a compared one-turn damage fixture.

    This intentionally records a mismatch as a mismatch (``matched=False``). The
    ``likely_mismatch_surface`` is a *conservative* read of the gathered evidence,
    not a proven root cause.
    """

    supported: bool
    matched: bool
    p1_move: str | None
    p2_move: str | None
    observed: ObservedDamage | None
    side_one_state: ActiveStateSummary | None
    side_two_state: ActiveStateSummary | None
    engine_branches: tuple[EngineBranchDamageSummary, ...]
    direct_calculate_damage: DirectCalculateDamageDiagnostic | None
    likely_mismatch_surface: str
    side_one_engine_state: EngineActiveStateSummary | None = None
    side_two_engine_state: EngineActiveStateSummary | None = None
    side_one_comparison: ActiveStateComparison | None = None
    side_two_comparison: ActiveStateComparison | None = None
    notes: tuple[str, ...] = field(default_factory=tuple)
    reason: str | None = None

    def engine_delta_tuples(self) -> tuple[tuple[int, int], ...]:
        seen: list[tuple[int, int]] = []
        for branch in self.engine_branches:
            if branch.deltas not in seen:
                seen.append(branch.deltas)
        return tuple(seen)

    def to_dict(self) -> dict[str, Any]:
        return {
            "supported": self.supported,
            "matched": self.matched,
            "p1_move": self.p1_move,
            "p2_move": self.p2_move,
            "observed": self.observed.to_dict() if self.observed is not None else None,
            "side_one_state": self.side_one_state.to_dict() if self.side_one_state is not None else None,
            "side_two_state": self.side_two_state.to_dict() if self.side_two_state is not None else None,
            "side_one_engine_state": (
                self.side_one_engine_state.to_dict() if self.side_one_engine_state is not None else None
            ),
            "side_two_engine_state": (
                self.side_two_engine_state.to_dict() if self.side_two_engine_state is not None else None
            ),
            "side_one_comparison": (
                self.side_one_comparison.to_dict() if self.side_one_comparison is not None else None
            ),
            "side_two_comparison": (
                self.side_two_comparison.to_dict() if self.side_two_comparison is not None else None
            ),
            "engine_branches": [branch.to_dict() for branch in self.engine_branches],
            "engine_delta_tuples": [list(t) for t in self.engine_delta_tuples()],
            "direct_calculate_damage": (
                self.direct_calculate_damage.to_dict() if self.direct_calculate_damage is not None else None
            ),
            "likely_mismatch_surface": self.likely_mismatch_surface,
            "notes": list(self.notes),
            "reason": self.reason,
        }

    def summary(self) -> str:
        if not self.supported:
            return f"damage diagnostic UNSUPPORTED: {self.reason or 'unknown reason'}"
        status = "MATCH" if self.matched else "MISMATCH"
        observed = self.observed.deltas if self.observed is not None else None
        return (
            f"one-turn damage {status}: observed deltas {observed} vs. engine deltas "
            f"{list(self.engine_delta_tuples())}; {self._engine_state_phrase()}; "
            f"likely surface: {self.likely_mismatch_surface}"
        )

    def _engine_state_phrase(self) -> str:
        comparisons = (self.side_one_comparison, self.side_two_comparison)
        if any(comparison is None for comparison in comparisons):
            return "engine-state not inspected"
        if all(comparison.matched for comparison in comparisons):  # type: ignore[union-attr]
            return "engine-state matches request spec (both sides)"
        mismatched = [c.side for c in comparisons if not c.matched]  # type: ignore[union-attr]
        return f"engine-state mismatch on {', '.join(mismatched)}"


def active_state_summary_from_side(side: SideSpec) -> ActiveStateSummary:
    """Summarize the active Pokemon of a request-derived :class:`SideSpec`."""

    mon = side.pokemon[int(side.active_index)]
    return ActiveStateSummary(
        species=mon.id,
        level=mon.level,
        hp=mon.hp,
        maxhp=mon.maxhp,
        types=tuple(mon.types),
        ability=mon.ability,
        item=mon.item,
        attack=mon.attack,
        defense=mon.defense,
        special_attack=mon.special_attack,
        special_defense=mon.special_defense,
        speed=mon.speed,
        moves=tuple((move.id, int(move.pp)) for move in mon.moves),
    )


# --- engine-state inspection ----------------------------------------------
#
# The summary above describes the request-derived *input* to state construction.
# The helpers below read the active Pokemon back off the *built* engine state and
# compare it field-by-field, so a mismatch can be attributed to (or cleared from)
# the spec->engine state-translation path. They never raise on a malformed/opaque
# state: an unreadable field is recorded as missing and reported as a mismatch.


def _jsonable_value(value: Any) -> Any:
    """Coerce a comparison value (scalars, tuples, nested tuples) to JSON shape."""

    if isinstance(value, (list, tuple)):
        return [_jsonable_value(item) for item in value]
    return value


def _lower_id(value: Any) -> str:
    return str(value).strip().lower()


def _normalize_optional_id(value: Any) -> str | None:
    """Lower-case an id, treating ``None``/``""``/``"none"`` as no-value (``None``)."""

    if value is None:
        return None
    text = str(value).strip().lower()
    if text in ("", "none"):
        return None
    return text


def _coerce_engine_int(value: Any) -> int:
    if isinstance(value, bool):
        raise TypeError("bool is not a valid engine int field")
    return int(value)


def _normalize_engine_types(raw: Any) -> tuple[str, ...]:
    """Lower-case engine type slots and drop the Gen 3 ``typeless`` padding slot."""

    if isinstance(raw, str):
        raw = (raw,)
    out: list[str] = []
    for entry in raw:
        text = str(entry).strip().lower()
        if text and text != TYPELESS:
            out.append(text)
    return tuple(out)


def _coerce_request_moves(raw: Any) -> tuple[tuple[str, int], ...]:
    out: list[tuple[str, int]] = []
    for move_id, pp in raw:
        out.append((_lower_id(move_id), _coerce_engine_int(pp)))
    return tuple(out)


def _coerce_engine_moves(raw: Any) -> tuple[tuple[str, int], ...]:
    out: list[tuple[str, int]] = []
    for move in raw:
        move_id = _lower_id(getattr(move, "id"))
        pp = getattr(move, "pp")
        out.append((move_id, _coerce_engine_int(pp)))
    return tuple(out)


def _read_engine_field(obj: Any, attr: str, field: str, missing: list[str], coerce) -> Any:
    """Read and normalize one engine attribute, recording ``field`` if it cannot be read."""

    try:
        raw = getattr(obj, attr)
    except Exception:  # noqa: BLE001 - a diagnostic must survive opaque/foreign states
        missing.append(field)
        return None
    try:
        return coerce(raw)
    except Exception:  # noqa: BLE001 - normalization failure is a missing field, not a crash
        missing.append(field)
        return None


def engine_active_state_summary(state: Any, side_attr: str) -> EngineActiveStateSummary:
    """Summarize the active Pokemon of a built engine ``state`` for one seat.

    ``side_attr`` is ``"side_one"`` or ``"side_two"``. Reads species/level/hp/maxhp,
    types (normalized), ability/item (``"none"`` -> ``None``), the five battle stats,
    and move ids/pp off the engine objects, normalizing for comparison against an
    :class:`ActiveStateSummary`. Never raises: a field that cannot be read is left
    ``None`` and named in ``missing_fields``; if the active Pokemon itself cannot be
    located, every field is reported missing.
    """

    missing: list[str] = []
    mon = None
    try:
        side = getattr(state, side_attr)
        mon = side.pokemon[int(side.active_index)]
    except Exception:  # noqa: BLE001 - opaque/foreign state must not crash the diagnostic
        mon = None
    if mon is None:
        return EngineActiveStateSummary(
            species=None,
            level=None,
            hp=None,
            maxhp=None,
            types=None,
            ability=None,
            item=None,
            attack=None,
            defense=None,
            special_attack=None,
            special_defense=None,
            speed=None,
            moves=None,
            missing_fields=ENGINE_STATE_COMPARED_FIELDS,
        )

    # ``ability``/``item`` use the optional normalizer (engine "none" -> None) and are
    # only "missing" when the attribute itself is absent; the others are missing when
    # absent or un-coercible.
    ability = _read_engine_field(mon, "ability", "ability", missing, _normalize_optional_id)
    item = _read_engine_field(mon, "item", "item", missing, _normalize_optional_id)
    return EngineActiveStateSummary(
        species=_read_engine_field(mon, "id", "species", missing, _lower_id),
        level=_read_engine_field(mon, "level", "level", missing, _coerce_engine_int),
        hp=_read_engine_field(mon, "hp", "hp", missing, _coerce_engine_int),
        maxhp=_read_engine_field(mon, "maxhp", "maxhp", missing, _coerce_engine_int),
        types=_read_engine_field(mon, "types", "types", missing, _normalize_engine_types),
        ability=ability,
        item=item,
        attack=_read_engine_field(mon, "attack", "attack", missing, _coerce_engine_int),
        defense=_read_engine_field(mon, "defense", "defense", missing, _coerce_engine_int),
        special_attack=_read_engine_field(mon, "special_attack", "special_attack", missing, _coerce_engine_int),
        special_defense=_read_engine_field(mon, "special_defense", "special_defense", missing, _coerce_engine_int),
        speed=_read_engine_field(mon, "speed", "speed", missing, _coerce_engine_int),
        moves=_read_engine_field(mon, "moves", "moves", missing, _coerce_engine_moves),
        missing_fields=tuple(missing),
    )


def compare_engine_active_state(
    request_summary: ActiveStateSummary,
    engine_summary: EngineActiveStateSummary,
    side: str,
) -> ActiveStateComparison:
    """Compare a request-derived summary against an engine-state summary, field by field.

    A field the engine could not read (named in ``engine_summary.missing_fields``) is
    reported as a mismatch carrying the :data:`ENGINE_FIELD_UNREAD` marker. ``matched``
    is ``True`` only when no field disagrees.
    """

    missing = set(engine_summary.missing_fields)
    checks: tuple[tuple[str, Any, Any], ...] = (
        ("species", _lower_id(request_summary.species), engine_summary.species),
        ("level", request_summary.level, engine_summary.level),
        ("hp", request_summary.hp, engine_summary.hp),
        ("maxhp", request_summary.maxhp, engine_summary.maxhp),
        ("types", _normalize_engine_types(request_summary.types), engine_summary.types),
        ("ability", _normalize_optional_id(request_summary.ability), engine_summary.ability),
        ("item", _normalize_optional_id(request_summary.item), engine_summary.item),
        ("attack", request_summary.attack, engine_summary.attack),
        ("defense", request_summary.defense, engine_summary.defense),
        ("special_attack", request_summary.special_attack, engine_summary.special_attack),
        ("special_defense", request_summary.special_defense, engine_summary.special_defense),
        ("speed", request_summary.speed, engine_summary.speed),
        ("moves", _coerce_request_moves(request_summary.moves), engine_summary.moves),
    )

    mismatches: list[FieldMismatch] = []
    for field_name, request_value, engine_value in checks:
        if field_name in missing:
            mismatches.append(FieldMismatch(field_name, request_value, ENGINE_FIELD_UNREAD))
            continue
        if request_value != engine_value:
            mismatches.append(FieldMismatch(field_name, request_value, engine_value))

    reason = None
    if missing:
        reason = "engine state could not expose fields: " + ", ".join(sorted(missing))
    return ActiveStateComparison(
        side=side,
        matched=not mismatches,
        mismatches=tuple(mismatches),
        request_summary=request_summary,
        engine_summary=engine_summary,
        reason=reason,
    )


def _active_request_member(request: Mapping[str, Any]) -> Mapping[str, Any]:
    side = request.get("side") if isinstance(request, Mapping) else None
    pokemon = side.get("pokemon") if isinstance(side, Mapping) else None
    if not isinstance(pokemon, list) or not pokemon:
        raise ValueError("request side has no pokemon list")
    for member in pokemon:
        if isinstance(member, Mapping) and member.get("active"):
            return member
    first = pokemon[0]
    if not isinstance(first, Mapping):
        raise ValueError("request side pokemon[0] is not a mapping")
    return first


def opening_active_hp(result: "OneTurnFixtureResult") -> tuple[int, int]:
    """Opening active HP ``(p1, p2)`` parsed from each seat's request condition."""

    if result.p1_request is None or result.p2_request is None:
        raise ValueError("fixture result is missing an opening request for one or both seats")
    p1_hp, _ = parse_condition(str(_active_request_member(result.p1_request).get("condition", "")))
    p2_hp, _ = parse_condition(str(_active_request_member(result.p2_request).get("condition", "")))
    return p1_hp, p2_hp


def observed_damage_from_result(result: "OneTurnFixtureResult") -> ObservedDamage:
    """Compute Showdown's observed per-side damage from opening request HP."""

    opening = opening_active_hp(result)
    final = observed_final_active_hp(result)
    deltas = (opening[0] - final[0], opening[1] - final[1])
    return ObservedDamage(opening_hp=opening, final_hp=final, deltas=deltas)


def _coerce_jsonable(value: Any) -> tuple[bool, Any]:
    """Coerce ``value`` to a JSON-serializable shape, reporting success.

    Accepts ``None``, scalars, and (recursively) lists/tuples/mappings of them.
    Anything else (an opaque binding object, say) is rejected so the caller can
    record an explicit unsupported reason instead of emitting a non-serializable
    blob.
    """

    if value is None or isinstance(value, (str, bool)):
        return True, value
    if isinstance(value, (int, float)):
        # Reject non-finite floats (NaN/inf): they are not strict-JSON-safe and
        # would only round-trip under json.dumps(..., allow_nan=True).
        if isinstance(value, float) and not math.isfinite(value):
            return False, None
        return True, value
    if isinstance(value, Mapping):
        out: dict[str, Any] = {}
        for key, item in value.items():
            ok, coerced = _coerce_jsonable(item)
            if not ok:
                return False, None
            out[str(key)] = coerced
        return True, out
    if isinstance(value, (list, tuple)):
        items: list[Any] = []
        for item in value:
            ok, coerced = _coerce_jsonable(item)
            if not ok:
                return False, None
            items.append(coerced)
        return True, items
    return False, None


def direct_calculate_damage_diagnostic(
    engine: Any,
    state: Any,
    move_one: str,
    move_two: str,
) -> DirectCalculateDamageDiagnostic:
    """Probe ``engine.calculate_damage`` for both turn orders, never raising.

    Signature probed: ``calculate_damage(state, side_one_move, side_two_move,
    side_one_moves_first)``. Returns ``supported=False`` with a reason when the
    function is absent, raises, or returns a shape we cannot serialize.
    """

    calc = getattr(engine, "calculate_damage", None)
    if not callable(calc):
        return DirectCalculateDamageDiagnostic(
            supported=False,
            output_side_one_first=None,
            output_side_two_first=None,
            reason="engine exposes no callable calculate_damage",
        )

    coerced: dict[bool, Any] = {}
    for side_one_moves_first in (True, False):
        try:
            raw = calc(state, move_one, move_two, side_one_moves_first)
        except Exception as exc:  # noqa: BLE001 - a diagnostic must survive engine quirks
            return DirectCalculateDamageDiagnostic(
                supported=False,
                output_side_one_first=None,
                output_side_two_first=None,
                reason=(
                    f"calculate_damage raised for side_one_moves_first="
                    f"{side_one_moves_first}: {type(exc).__name__}: {exc}"
                ),
            )
        ok, value = _coerce_jsonable(raw)
        if not ok:
            return DirectCalculateDamageDiagnostic(
                supported=False,
                output_side_one_first=None,
                output_side_two_first=None,
                reason=(
                    f"calculate_damage returned an unrecognized/unserializable shape for "
                    f"side_one_moves_first={side_one_moves_first}: {type(raw).__name__}"
                ),
            )
        coerced[side_one_moves_first] = value

    return DirectCalculateDamageDiagnostic(
        supported=True,
        output_side_one_first=coerced[True],
        output_side_two_first=coerced[False],
        reason=None,
    )


def _mismatch_surface_and_notes(
    *,
    matched: bool,
    observed: ObservedDamage,
    engine_branches: Sequence[EngineBranchDamageSummary],
    side_one_comparison: ActiveStateComparison | None,
    side_two_comparison: ActiveStateComparison | None,
) -> tuple[str, tuple[str, ...]]:
    """Conservatively describe where a damage mismatch appears to live.

    The built engine state is read back and compared field-by-field against the
    request-derived spec (``side_one_comparison``/``side_two_comparison``). When both
    sides match, the exposed/stored engine active fields line up with the request
    spec, so the surface narrows to the engine's damage/data path for those inspected
    fields. When either side mismatches or could not be inspected, the broad
    engine-damage-or-translation surface stands. Either way the *exact* cause (base
    power, category, type chart, stat usage, rounding, or a translation defect) is
    left explicitly unresolved -- a matching state-comparison narrows *where* the bug
    is, it does not name it.
    """

    if matched:
        return NO_MISMATCH_SURFACE, ("engine reproduces the observed one-turn damage outcome",)

    engine_deltas: list[tuple[int, int]] = []
    for branch in engine_branches:
        if branch.deltas not in engine_deltas:
            engine_deltas.append(branch.deltas)
    deltas_note = (
        f"observed per-side damage deltas {list(observed.deltas)} are not reproduced by any "
        f"engine branch ({[list(d) for d in engine_deltas]})"
    )

    comparisons = (side_one_comparison, side_two_comparison)
    inspectable = all(comparison is not None for comparison in comparisons)
    both_match = inspectable and all(comparison.matched for comparison in comparisons)  # type: ignore[union-attr]

    if both_match:
        notes = (
            "built engine-state active summaries were read back and match the request-derived "
            "spec on BOTH sides for every inspected exposed/stored field (species/level/hp/maxhp/"
            "types/ability/item/stats/moves; types are dex-derived)",
            deltas_note,
            "the surviving mismatch is therefore on the engine's damage/data path, but the exact "
            "cause (move base power/category, type-effectiveness table, stat usage, or rounding) "
            "remains UNRESOLVED -- this narrows the surface, it does not prove a cause",
        )
        return NARROW_MISMATCH_SURFACE, notes

    # Either a comparison disagreed or the engine state could not be fully inspected:
    # the spec->engine state-translation path cannot be cleared, so keep the broad surface.
    notes_list = [
        "active-state stats/hp/moves are request-derived and types are dex-derived (see "
        "side_one_state/side_two_state)",
    ]
    if not inspectable:
        notes_list.append(
            "the built engine state could not be inspected for at least one side, so engine-state "
            "fidelity is unproven and the spec->engine state-translation path cannot be cleared"
        )
    else:
        notes_list.append(
            "the built engine state was inspected but did not clear the request-derived "
            "comparison: "
            + "; ".join(
                _comparison_mismatch_phrase(comparison)
                for comparison in comparisons
                if comparison is not None and not comparison.matched
            )
        )
    notes_list.append(deltas_note)
    notes_list.append(
        "exact root cause (move base power/category, type-effectiveness table, stat usage, "
        "rounding, or a spec->engine state-translation defect) remains UNRESOLVED -- this "
        "records the surface, not a proven cause"
    )
    return BROAD_MISMATCH_SURFACE, tuple(notes_list)


def _comparison_mismatch_phrase(comparison: ActiveStateComparison) -> str:
    parts = []
    for mismatch in comparison.mismatches:
        parts.append(
            f"{mismatch.field} (request={_jsonable_value(mismatch.request_value)!r} "
            f"vs engine={_jsonable_value(mismatch.engine_value)!r})"
        )
    if comparison.reason:
        parts.append(f"reason={comparison.reason}")
    if not parts:
        parts.append("no field-level mismatch recorded")
    return f"{comparison.side}: " + ", ".join(parts)


def build_one_turn_damage_diagnostic(
    result: "OneTurnFixtureResult",
    dex: "ShowdownDex",
    *,
    module: Any | None = None,
    p1_choice: str | None = None,
    p2_choice: str | None = None,
) -> OneTurnDamageDiagnostic:
    """Build a serializable damage diagnostic for a compared one-turn fixture.

    Mirrors :func:`compare_one_turn_outcome` (same request->state path and move
    resolution) and layers on observed/engine per-side damage deltas, request-derived
    active-state summaries, and a (non-raising) direct ``calculate_damage`` probe.
    Pass a fake ``module`` to keep tests off the native wheel; ``None`` loads it lazily.
    """

    p1_choice = p1_choice if p1_choice is not None else result.choices.get("p1")
    p2_choice = p2_choice if p2_choice is not None else result.choices.get("p2")
    if not p1_choice or not p2_choice:
        raise ValueError("both p1 and p2 choices are required to build a damage diagnostic")

    move_one = engine_move_for_choice(result.p1_request, p1_choice)
    move_two = engine_move_for_choice(result.p2_request, p2_choice)

    observed = observed_damage_from_result(result)
    spec = build_battle_spec_from_result(result, dex)
    side_one_state = active_state_summary_from_side(spec.side_one)
    side_two_state = active_state_summary_from_side(spec.side_two)

    engine = require_poke_engine() if module is None else module
    state = build_poke_engine_state(spec, module=engine)
    branch_outcomes = enumerate_engine_outcomes(engine, state, move_one, move_two)
    engine_branches = tuple(
        EngineBranchDamageSummary(
            percentage=outcome.percentage,
            final_hp=outcome.final_hp,
            deltas=(observed.opening_hp[0] - outcome.final_hp[0], observed.opening_hp[1] - outcome.final_hp[1]),
            description=outcome.description,
        )
        for outcome in branch_outcomes
    )

    # Read the active Pokemon back off the built state and compare it field-by-field
    # against the request-derived spec, so the spec->engine translation path can be
    # cleared (or implicated) rather than left to guesswork.
    side_one_engine_state = engine_active_state_summary(state, "side_one")
    side_two_engine_state = engine_active_state_summary(state, "side_two")
    side_one_comparison = compare_engine_active_state(side_one_state, side_one_engine_state, "side_one")
    side_two_comparison = compare_engine_active_state(side_two_state, side_two_engine_state, "side_two")

    direct = direct_calculate_damage_diagnostic(engine, state, move_one, move_two)

    # Reuse the comparator's matching rule rather than recomputing it inline.
    matched = compare_outcomes(
        observed.final_hp, branch_outcomes, p1_move=move_one, p2_move=move_two
    ).matched
    surface, notes = _mismatch_surface_and_notes(
        matched=matched,
        observed=observed,
        engine_branches=engine_branches,
        side_one_comparison=side_one_comparison,
        side_two_comparison=side_two_comparison,
    )

    return OneTurnDamageDiagnostic(
        supported=True,
        matched=matched,
        p1_move=move_one,
        p2_move=move_two,
        observed=observed,
        side_one_state=side_one_state,
        side_two_state=side_two_state,
        engine_branches=engine_branches,
        direct_calculate_damage=direct,
        likely_mismatch_surface=surface,
        side_one_engine_state=side_one_engine_state,
        side_two_engine_state=side_two_engine_state,
        side_one_comparison=side_one_comparison,
        side_two_comparison=side_two_comparison,
        notes=notes,
    )


def run_charmander_squirtle_outcome_comparison(
    *,
    config: "LocalShowdownConfig | None" = None,
    module: Any | None = None,
    dex: "ShowdownDex | None" = None,
    seed: int = 7,
) -> OutcomeComparison:
    """Run the Charmander/Ember vs. Squirtle/Water Gun damage smoke end-to-end.

    Runs one deterministic Showdown turn (both seats pick move 1), loads the Gen 3
    dex for typing, builds the engine state from the real opening requests, and
    compares outcomes. Requires both a built local Showdown checkout (node) and a
    poke-engine wheel; intended for the optional real integration test and manual
    diagnostics only. Not wired into any rollout/training/search path.
    """

    from .local_showdown import LocalShowdownConfig
    from .dex import load_showdown_dex_cached
    from .showdown_fixture import charmander_squirtle_fixture, run_one_turn_fixture

    config = config or LocalShowdownConfig()
    p1_team, p2_team = charmander_squirtle_fixture()
    result = run_one_turn_fixture(
        p1_team=p1_team,
        p2_team=p2_team,
        p1_choice="move 1",
        p2_choice="move 1",
        seed=seed,
        config=config,
    )
    if dex is None:
        dex = load_showdown_dex_cached(config.resolved_showdown_root())
    return compare_one_turn_outcome(result, dex, module=module)


def run_charmander_squirtle_damage_diagnostic(
    *,
    config: "LocalShowdownConfig | None" = None,
    module: Any | None = None,
    dex: "ShowdownDex | None" = None,
    seed: int = 7,
) -> OneTurnDamageDiagnostic:
    """Build the damage diagnostic for the curated Charmander/Squirtle fixture.

    Runs the same deterministic Showdown turn as
    :func:`run_charmander_squirtle_outcome_comparison`, then returns a
    :class:`OneTurnDamageDiagnostic` instead of a bare match/mismatch. Requires a
    built local Showdown checkout (node) and a poke-engine wheel; intended for the
    optional real integration test and manual diagnostics only. Not wired into any
    rollout/training/search path.
    """

    from .local_showdown import LocalShowdownConfig
    from .dex import load_showdown_dex_cached
    from .showdown_fixture import charmander_squirtle_fixture, run_one_turn_fixture

    config = config or LocalShowdownConfig()
    p1_team, p2_team = charmander_squirtle_fixture()
    result = run_one_turn_fixture(
        p1_team=p1_team,
        p2_team=p2_team,
        p1_choice="move 1",
        p2_choice="move 1",
        seed=seed,
        config=config,
    )
    if dex is None:
        dex = load_showdown_dex_cached(config.resolved_showdown_root())
    return build_one_turn_damage_diagnostic(result, dex, module=module)
