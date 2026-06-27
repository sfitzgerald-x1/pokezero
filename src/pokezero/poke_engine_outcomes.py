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

Scope is **singles, move-vs-move only.** The module is optional and lazy: importing
it never requires the native wheel (the real engine is imported only when a
comparison actually runs and no fake ``module`` is supplied). It is deliberately
disconnected from rollout, training, search, benchmarks, and self-play.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import re
from typing import TYPE_CHECKING, Any, Mapping, Sequence

from .dex import normalize_id
from .poke_engine_adapter import (
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
