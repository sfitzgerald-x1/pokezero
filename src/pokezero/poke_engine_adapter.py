"""Curated Showdown Gen 3 fixture -> ``poke_engine.State`` adapter.

This is a narrow, optional seam for the poke-engine evaluation spike. It maps a
small, hand-curated battle fixture into the constructor surface proven by
``doctor --smoke`` (``State``/``Side``/``Pokemon``/``Move``) and offers a local
reversible smoke that builds the state and checks apply/reverse round-trips.

It is intentionally disconnected from rollout, training, search, and benchmarks.
The real ``poke_engine`` module is imported lazily; importing this module never
requires the Rust-backed wheel. Pass an explicit ``module`` (e.g. a fake) to
keep CI off the native dependency, or ``None`` to use the installed engine via
:func:`~pokezero.poke_engine_backend.require_poke_engine`.

This adapter only constructs a state; it does **not** prove Showdown or Gen 3
random-battle mechanics equivalence. Legal-action equivalence against Showdown
request payloads lives in :mod:`pokezero.poke_engine_legal_actions` (currently
gated on a poke-engine root-option export).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from .poke_engine_backend import (
    PokeEngineReversibleSmokeResult,
    PokeEngineUnavailableError,
    require_poke_engine,
    run_reversible_smoke_on_state,
)

# Gen 3 stores types as an exactly-two-slot pair; a mono-type Pokemon fills the
# empty slot with ``typeless`` (mirrors the serialized state from doctor --smoke).
TYPELESS = "typeless"
TYPE_SLOTS = 2

# Module attributes the adapter needs to construct a state.
ADAPTER_CONSTRUCTION_API = ("State", "Side", "Pokemon", "Move")


@dataclass(frozen=True)
class MoveSpec:
    """A single move slot on a curated Pokemon."""

    id: str
    pp: int = 32
    disabled: bool = False


@dataclass(frozen=True)
class PokemonSpec:
    """A curated Gen 3 Pokemon set.

    ``id`` is the poke-engine species id (lowercase, no spaces, e.g.
    ``"charmander"``). ``types`` may carry one or two entries; a single type is
    padded to the Gen 3 two-slot pair with ``typeless``.
    """

    id: str
    level: int
    types: Sequence[str]
    hp: int
    maxhp: int
    attack: int
    defense: int
    special_attack: int
    special_defense: int
    speed: int
    moves: Sequence[MoveSpec]
    status: str = "none"
    ability: str | None = None
    item: str | None = None
    nature: str | None = None
    rest_turns: int = 0
    sleep_turns: int = 0
    weight_kg: float | None = None


@dataclass(frozen=True)
class SideSpec:
    """One seat: an ordered party plus which slot is active."""

    pokemon: Sequence[PokemonSpec]
    active_index: int = 0
    # Optional Gen 3 side conditions, keyed by ``poke_engine.SideConditions``
    # field name (snake_case, e.g. ``"spikes"``, ``"reflect"``).
    side_conditions: Mapping[str, int] = field(default_factory=dict)
    # Active Pokemon stat stages, keyed by ``poke_engine.Side`` boost field
    # prefix (``"attack"``, ``"defense"``, ``"special_attack"``,
    # ``"special_defense"``, ``"speed"``, ``"accuracy"``, ``"evasion"``).
    boosts: Mapping[str, int] = field(default_factory=dict)
    # Active Pokemon volatile statuses, engine ids (e.g. ``"leechseed"``).
    volatile_statuses: Sequence[str] = ()
    # Substitute HP behind a ``"substitute"`` volatile (0 = no substitute).
    substitute_health: int = 0
    # This side must choose a replacement (its active fainted mid/end of turn).
    force_switch: bool = False
    # Pending Wish as (turns_counter, heal_amount); (0, 0) = none. The engine
    # decrements the counter each end-of-turn and heals when it reaches zero.
    wish: tuple[int, int] = (0, 0)
    # Engine last-used-move token ("move:<slot>" / "switch:<idx>"); "" = unset.
    # SHARP EDGE: the engine accepts only slot INDICES here — a move id is
    # accepted at construction then panics inside generate_instructions.
    last_used_move: str = ""
    # Volatile duration counters by poke_engine.VolatileStatusDurations field
    # name (e.g. {"encore": 1}); empty = engine defaults.
    volatile_status_durations: Mapping[str, int] = field(default_factory=dict)


@dataclass(frozen=True)
class BattleSpec:
    """A curated two-sided battle fixture."""

    side_one: SideSpec
    side_two: SideSpec
    weather: str = "none"
    terrain: str = "none"
    trick_room: bool = False
    # Turns of weather left; -1 means indefinite (ability-set Gen 3 weather).
    # Only forwarded to the engine when ``weather`` is not ``"none"``.
    weather_turns_remaining: int = -1


def minimal_gen3_fixture() -> BattleSpec:
    """The curated Charmander/Ember vs. Squirtle/Water Gun Gen 3 fixture.

    Matches the minimal state proven reversible by ``doctor --smoke`` so the
    adapter path and the backend smoke exercise the same mechanics surface.
    """

    charmander = PokemonSpec(
        id="charmander",
        level=100,
        types=("fire",),
        hp=100,
        maxhp=100,
        attack=100,
        defense=100,
        special_attack=100,
        special_defense=100,
        speed=100,
        status="none",
        moves=(MoveSpec(id="ember", pp=32), MoveSpec(id="tackle", pp=32)),
    )
    squirtle = PokemonSpec(
        id="squirtle",
        level=100,
        types=("water",),
        hp=100,
        maxhp=100,
        attack=100,
        defense=100,
        special_attack=100,
        special_defense=100,
        speed=100,
        status="none",
        moves=(MoveSpec(id="watergun", pp=32), MoveSpec(id="tackle", pp=32)),
    )
    return BattleSpec(
        side_one=SideSpec(pokemon=(charmander,), active_index=0),
        side_two=SideSpec(pokemon=(squirtle,), active_index=0),
        weather="none",
        terrain="none",
        trick_room=False,
    )


def build_poke_engine_state(spec: BattleSpec, module: Any | None = None) -> Any:
    """Build a ``poke_engine.State`` from a curated :class:`BattleSpec`.

    When ``module`` is ``None`` the installed engine is loaded lazily via
    :func:`~pokezero.poke_engine_backend.require_poke_engine`; pass a fake module
    to keep tests off the native dependency. Invalid fixtures raise ``ValueError``
    (out-of-range/empty data) or ``TypeError`` (wrong field types) with a path
    pointing at the offending field.
    """

    if not isinstance(spec, BattleSpec):
        raise TypeError(f"spec must be a BattleSpec, got {type(spec).__name__}")

    engine = require_poke_engine() if module is None else module
    missing = tuple(name for name in ADAPTER_CONSTRUCTION_API if not hasattr(engine, name))
    if missing:
        raise PokeEngineUnavailableError("Missing construction API: " + ", ".join(missing))

    side_one = _build_side(engine, spec.side_one, "side_one")
    side_two = _build_side(engine, spec.side_two, "side_two")

    if not isinstance(spec.trick_room, bool):
        raise TypeError(f"trick_room must be a bool, got {type(spec.trick_room).__name__}")

    kwargs: dict[str, Any] = {
        "side_one": side_one,
        "side_two": side_two,
        "weather": str(spec.weather),
        "terrain": str(spec.terrain),
        "trick_room": spec.trick_room,
    }
    if str(spec.weather) != "none":
        kwargs["weather_turns_remaining"] = _require_int(
            spec.weather_turns_remaining, "weather_turns_remaining"
        )
    return engine.State(**kwargs)


def _build_side(engine: Any, side: SideSpec, path: str) -> Any:
    if not isinstance(side, SideSpec):
        raise TypeError(f"{path} must be a SideSpec, got {type(side).__name__}")
    if not side.pokemon:
        raise ValueError(f"{path} must contain at least one Pokemon")

    # Validate the cheap active_index before constructing every Pokemon.
    active = side.active_index
    if isinstance(active, bool) or not isinstance(active, int):
        raise TypeError(f"{path}.active_index must be an int, got {type(active).__name__}")
    if not 0 <= active < len(side.pokemon):
        raise ValueError(
            f"{path}.active_index {active} is out of range for {len(side.pokemon)} Pokemon"
        )

    party = [
        _build_pokemon(engine, member, f"{path}.pokemon[{index}]")
        for index, member in enumerate(side.pokemon)
    ]

    kwargs: dict[str, Any] = {"pokemon": party, "active_index": str(active)}
    if side.side_conditions:
        kwargs["side_conditions"] = _build_side_conditions(engine, side.side_conditions, path)
    for stat, stage in dict(side.boosts).items():
        if stat not in SIDE_BOOST_FIELDS:
            raise ValueError(f"{path}.boosts has unknown stat {stat!r}")
        stage = _require_int(stage, f"{path}.boosts[{stat!r}]")
        if not -6 <= stage <= 6:
            raise ValueError(f"{path}.boosts[{stat!r}] must be within [-6, 6], got {stage}")
        if stage:
            kwargs[f"{stat}_boost"] = stage
    if side.volatile_statuses:
        volatiles = [str(name) for name in side.volatile_statuses]
        if any(not name for name in volatiles):
            raise ValueError(f"{path}.volatile_statuses entries must be non-empty")
        kwargs["volatile_statuses"] = set(volatiles)
    if _require_non_negative_int(side.substitute_health, f"{path}.substitute_health"):
        kwargs["substitute_health"] = side.substitute_health
    if not isinstance(side.force_switch, bool):
        raise TypeError(f"{path}.force_switch must be a bool, got {type(side.force_switch).__name__}")
    if side.force_switch:
        kwargs["force_switch"] = True
    wish_counter, wish_amount = side.wish
    _require_non_negative_int(wish_counter, f"{path}.wish[0]")
    _require_non_negative_int(wish_amount, f"{path}.wish[1]")
    if wish_counter:
        kwargs["wish"] = (wish_counter, wish_amount)
    if side.last_used_move:
        token = str(side.last_used_move)
        prefix, _, index = token.partition(":")
        if prefix not in ("move", "switch") or not index.isdigit():
            raise ValueError(
                f"{path}.last_used_move must be 'move:<slot>' or 'switch:<idx>' with a numeric "
                f"index (engine panics on move ids), got {token!r}"
            )
        kwargs["last_used_move"] = token
    if side.volatile_status_durations:
        factory = getattr(engine, "VolatileStatusDurations", None)
        if factory is None:
            raise PokeEngineUnavailableError(
                f"{path}.volatile_status_durations requested but engine lacks VolatileStatusDurations"
            )
        for name, turns in side.volatile_status_durations.items():
            _require_non_negative_int(turns, f"{path}.volatile_status_durations[{name!r}]")
        kwargs["volatile_status_durations"] = factory(**dict(side.volatile_status_durations))
    return engine.Side(**kwargs)


def _build_side_conditions(engine: Any, conditions: Mapping[str, int], path: str) -> Any:
    factory = getattr(engine, "SideConditions", None)
    if factory is None:
        raise PokeEngineUnavailableError(
            f"{path}.side_conditions requested but engine has no SideConditions type"
        )
    if not isinstance(conditions, Mapping):
        raise TypeError(
            f"{path}.side_conditions must be a mapping, got {type(conditions).__name__}"
        )
    for key, value in conditions.items():
        _require_non_negative_int(value, f"{path}.side_conditions[{key!r}]")
    return factory(**dict(conditions))


def _require_int(value: Any, label: str) -> int:
    """Reject bools and non-ints; bools are ints in Python and never valid here."""

    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{label} must be an int, got {type(value).__name__}")
    return value


def _require_positive_int(value: Any, label: str) -> int:
    if _require_int(value, label) <= 0:
        raise ValueError(f"{label} must be positive, got {value}")
    return value


def _require_non_negative_int(value: Any, label: str) -> int:
    if _require_int(value, label) < 0:
        raise ValueError(f"{label} must be non-negative, got {value}")
    return value


# Battle stats that must each be a positive int.
POKEMON_STAT_FIELDS = ("attack", "defense", "special_attack", "special_defense", "speed")

# Boost fields accepted by ``poke_engine.Side`` (suffixed ``_boost`` on build).
SIDE_BOOST_FIELDS = POKEMON_STAT_FIELDS + ("accuracy", "evasion")


def _build_pokemon(engine: Any, member: PokemonSpec, path: str) -> Any:
    if not isinstance(member, PokemonSpec):
        raise TypeError(f"{path} must be a PokemonSpec, got {type(member).__name__}")
    if not member.id:
        raise ValueError(f"{path}.id must be a non-empty species id")
    if not member.moves:
        raise ValueError(f"{path}.moves must contain at least one move")

    _require_positive_int(member.level, f"{path}.level")
    _require_positive_int(member.maxhp, f"{path}.maxhp")
    _require_non_negative_int(member.hp, f"{path}.hp")
    if member.hp > member.maxhp:
        raise ValueError(
            f"{path}.hp {member.hp} exceeds {path}.maxhp {member.maxhp}"
        )
    for stat in POKEMON_STAT_FIELDS:
        _require_positive_int(getattr(member, stat), f"{path}.{stat}")

    kwargs: dict[str, Any] = {
        "id": member.id,
        "level": member.level,
        "types": _normalize_types(member.types, path),
        "hp": member.hp,
        "maxhp": member.maxhp,
        "attack": member.attack,
        "defense": member.defense,
        "special_attack": member.special_attack,
        "special_defense": member.special_defense,
        "speed": member.speed,
        "status": member.status,
        "moves": [_build_move(engine, move, f"{path}.moves[{i}]") for i, move in enumerate(member.moves)],
    }
    if member.ability is not None:
        kwargs["ability"] = member.ability
    if member.item is not None:
        kwargs["item"] = member.item
    if member.nature is not None:
        kwargs["nature"] = member.nature
    if _require_non_negative_int(member.rest_turns, f"{path}.rest_turns"):
        kwargs["rest_turns"] = member.rest_turns
    if _require_non_negative_int(member.sleep_turns, f"{path}.sleep_turns"):
        kwargs["sleep_turns"] = member.sleep_turns
    if member.weight_kg is not None:
        weight = float(member.weight_kg)
        if weight <= 0.0:
            raise ValueError(f"{path}.weight_kg must be positive, got {weight}")
        kwargs["weight_kg"] = weight
    return engine.Pokemon(**kwargs)


def _build_move(engine: Any, move: MoveSpec, path: str) -> Any:
    if not isinstance(move, MoveSpec):
        raise TypeError(f"{path} must be a MoveSpec, got {type(move).__name__}")
    if not move.id:
        raise ValueError(f"{path}.id must be a non-empty move id")
    _require_non_negative_int(move.pp, f"{path}.pp")
    if not isinstance(move.disabled, bool):
        raise TypeError(f"{path}.disabled must be a bool, got {type(move.disabled).__name__}")
    if move.disabled:
        return engine.Move(id=move.id, pp=move.pp, disabled=True)
    return engine.Move(id=move.id, pp=move.pp)


def _normalize_types(types: Sequence[str], path: str) -> tuple[str, ...]:
    """Pad/validate a type list into the Gen 3 two-slot pair."""

    if isinstance(types, str):
        raise TypeError(f"{path}.types must be a sequence of type names, not a bare string")
    slots = [str(entry) for entry in types]
    if not slots:
        raise ValueError(f"{path}.types must contain at least one type")
    if len(slots) > TYPE_SLOTS:
        raise ValueError(f"{path}.types accepts at most {TYPE_SLOTS} types, got {len(slots)}")
    while len(slots) < TYPE_SLOTS:
        slots.append(TYPELESS)
    return tuple(slots)


def run_adapter_reversible_smoke(
    spec: BattleSpec | None = None,
    *,
    module: Any | None = None,
    move_one: str = "ember",
    move_two: str = "watergun",
    max_instruction_checks: int = 8,
) -> PokeEngineReversibleSmokeResult:
    """Build a fixture into a state and run the reversible apply/reverse smoke.

    Defaults to :func:`minimal_gen3_fixture` and the Ember/Water Gun pairing it
    was curated for. Reuses the backend round-trip core so this stays a thin
    fixture-aware wrapper, not a duplicate of the smoke logic.
    """

    engine = require_poke_engine() if module is None else module
    fixture = minimal_gen3_fixture() if spec is None else spec
    state = build_poke_engine_state(fixture, module=engine)
    # build_poke_engine_state has already validated the fixture (and active_index
    # range), so checking the smoke moves against the active Pokemon here turns an
    # opaque "generated no instructions" failure into a clear, actionable error.
    _require_move_on_active(fixture.side_one, move_one, "side_one")
    _require_move_on_active(fixture.side_two, move_two, "side_two")
    return run_reversible_smoke_on_state(
        engine,
        state,
        move_one,
        move_two,
        max_instruction_checks=max_instruction_checks,
    )


def _require_move_on_active(side: SideSpec, move_id: str, path: str) -> None:
    """Reject a smoke move the active Pokemon does not actually carry."""

    active = side.pokemon[side.active_index]
    available = [move.id for move in active.moves]
    if move_id not in available:
        raise ValueError(
            f"smoke move {move_id!r} is not on the active {path} Pokemon {active.id!r} "
            f"(available: {', '.join(available)})"
        )
