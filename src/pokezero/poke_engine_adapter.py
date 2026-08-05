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
from functools import lru_cache
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

# The search entrypoint `pokezero.engine_search` drives. `monte_carlo_tree_search`
# is a pure-PYTHON wrapper in poke-engine-py's `python/poke_engine/__init__.py`
# that adapts the native `mcts` pyfunction, so an install carrying the compiled
# extension WITHOUT that wrapper imports cleanly, exposes `mcts`, and then fails
# only at call time with a bare AttributeError.
MCTS_ENTRYPOINT_API = ("monte_carlo_tree_search",)

# The one supported way to produce a wheel with the gen3 patch set applied.
POKE_ENGINE_BUILD_COMMAND = "scripts/setup_poke_engine.sh /path/to/venv/bin/python"
# Any mid-Rest value works; 2 is the middle of the reachable 1..3 range.
_REST_TURNS_PROBE_VALUE = 2
# Not 1: a refund of 1 could also be produced by a bool coerced to int or an
# off-by-one, so it cannot distinguish 'carried' from 'accidentally 1'.
_REST_SLEEP_REFUND_PROBE_VALUE = 2
# Gen 3 Rest sets a 3-turn counter, and the engine's wake match PANICS above it.
_REST_SLEEP_TURNS_MAX = 3


class PokeEngineMoveTrapUnsupportedError(PokeEngineUnavailableError):
    """Raised when a state needs move-trapping but the native patch is absent.

    Upstream has no TRAPPED volatile at all, and the binding's volatile parser
    resolves an unknown token to NONE rather than rejecting it — so an
    unpatched wheel accepts ``"trapped"`` and SILENTLY DROPS it. That is the
    worst possible failure for this particular effect: the sampled world would
    hand the trapped seat its switch options back, and search would confidently
    plan an escape Showdown will refuse. Construction must fail closed.
    """


class PokeEngineTransformRevertUnsupportedError(PokeEngineUnavailableError):
    """Raised when a spec carries a pre-transform base form the binding cannot take.

    ``Pokemon.pre_transform`` is what lets a CONSTRUCTED Transform end: without
    it the engine has no base form to restore and the copy is stuck for the rest
    of the search. A wheel built before that field existed rejects the keyword
    outright, so this is a loud failure rather than a silent drop — but the check
    exists so the failure names the cause and the fix instead of surfacing as a
    bare TypeError from deep inside construction.
    """


class PokeEngineMctsEntrypointMissingError(PokeEngineUnavailableError):
    """Raised when the installed binding has no `monte_carlo_tree_search`.

    Unlike the other capability errors here, this is not about a missing gen3
    patch — it is about which HALF of the binding got installed.
    `monte_carlo_tree_search` is a pure-Python wrapper around the native `mcts`
    pyfunction, shipped in poke-engine-py's `python/poke_engine/__init__.py`. An
    install that carries the compiled extension without that wrapper imports
    fine and exposes `mcts`, so nothing notices until search actually runs and
    dies on a bare AttributeError several layers down.
    """


class PokeEngineRestTurnsUnsupportedError(PokeEngineUnavailableError):
    """Raised when a world carries a Rest sleep the binding cannot express.

    ``rest_turns`` is not merely the Rest wake timer, it is how the engine spells
    gen3's Sleep Clause exemption: ``has_alive_non_rested_sleeping_pkmn`` counts a
    sleeper only while ``rest_turns == 0``. A binding that takes the field and drops
    it therefore does not decline the world -- it builds one where a Rest-asleep mon
    re-arms a clause the real battle exempts it from, so search plans around a sleep
    move Showdown would let through (or, on the other side, believes its own sleep
    move is blocked). The damage is worst exactly where this fix aims: a benched
    Rest-sleeper, whose mis-modelled state nothing else in the position reveals.
    """


class PokeEngineRestSleepRefundUnsupportedError(PokeEngineUnavailableError):
    """Raised when a world carries a pending Rest refund the binding cannot express.

    Distinct from :class:`PokeEngineRestTurnsUnsupportedError` in the direction of
    the error. Dropping ``rest_turns`` re-arms the Sleep Clause; dropping
    ``rest_sleep_pending_refund`` loses turns the sim would credit back, so the mon
    wakes EARLY on every branch that switches it in. Both are silent, which is why
    each gets a probe rather than a version check.
    """


class PokeEngineChargeStateUnsupportedError(PokeEngineUnavailableError):
    """Raised when a world carries a mid-charge state the binding cannot express.

    The charge volatile IS the commitment: the engine keys `active_is_charging_move`
    off it to lock the side to that one move, and releases it in
    `generate_instructions`. A binding that drops the volatile does not decline the
    world -- it builds one where the charging mon is FREE, so search lets it pick any
    move and, if it picks Solar Beam, starts a fresh charge instead of releasing. That
    is the silent wrongness this fails closed on.
    """


class PokeEngineAttractUnsupportedError(PokeEngineUnavailableError):
    """Raised when a state needs Attract but the native patch is absent.

    Upstream accepts the volatile token while silently omitting its Gen 3
    immobilization branch. Treating that as a supported state would make search
    optimistically price every attracted turn, so construction must fail closed.
    """


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
    gender: str | None = None
    rest_turns: int = 0
    # Rest sleep turns already skipped by Sleep Talk/Snore that gen 3 credits back
    # on SWITCH-IN. Only an ACTIVE sleeper needs it: for a benched one a switch-in
    # necessarily precedes the next attempt, so construction folds the refund
    # straight into ``rest_turns``. An active mon may never switch, and search
    # explores the stay-in branch too, where pre-applying would wake it early.
    rest_sleep_pending_refund: int = 0
    sleep_turns: int = 0
    weight_kg: float | None = None
    # BASE IDENTITY: what the engine restores when a temporary change to the
    # current identity ends (`ability_on_switch_out` for the ability, the
    # TYPECHANGE switch-out arm for the types). ``None`` means "whatever the
    # current value is", which is what every untransformed Pokemon wants — only
    # a spec describing an already-Transformed active needs them to differ from
    # ``ability``/``types``, because there the CURRENT identity is the donor's
    # and the base identity is still the transformer's own.
    base_ability: str | None = None
    base_types: Sequence[str] | None = None
    # The Pokemon's OWN base form, set only when this spec describes an already
    # Transformed active (the belief-world constructor bakes the copied species/
    # stats/moves straight into the spec). The engine restores it when the
    # transformer leaves the field; without it the copy is stuck for the rest of
    # the search. Nested pre_transform is ignored — a base form has none.
    pre_transform: "PokemonSpec | None" = None


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
    # Mid-turn Baton Pass boundary: this side chose Baton Pass and is now
    # picking the recipient (engine restricts it to switch choices).
    baton_passing: bool = False
    # The OTHER side already committed its move this turn (engine resolves
    # ``switch_out_move_second_saved_move`` after the replacement enters).
    slow_uturn_move: bool = False
    switch_out_move_second_saved_move: str = ""
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

    if _spec_requires_attract(spec):
        _require_attract_immobilization(engine)
    if _spec_requires_move_trap(spec):
        require_move_trap_support(engine)
    if _spec_requires_pre_transform(spec):
        require_pre_transform_support(engine)
    if _spec_requires_rest_turns(spec):
        require_rest_turns_support(engine)

    return _build_poke_engine_state_unchecked(spec, engine)


def _build_poke_engine_state_unchecked(spec: BattleSpec, engine: Any) -> Any:
    """Build a state after any mechanic-specific capability checks have run."""

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


def _spec_requires_move_trap(spec: BattleSpec) -> bool:
    return any(
        str(volatile).casefold() == "trapped"
        for side in (spec.side_one, spec.side_two)
        for volatile in side.volatile_statuses
    )


def require_move_trap_support(engine: Any | None = None) -> None:
    """Prove that the installed binding carries the gen3 move-trapping patch.

    There is no binding-level version marker for a local patch, so probe the
    capability itself. Unlike Attract — which the wheel at least stores before
    ignoring — an unpatched wheel does not know TRAPPED exists, and
    ``PokemonVolatileStatus::from_str`` is generated with ``default = NONE``, so
    the token is accepted and quietly discarded. Round-tripping a minimal state
    through ``to_string`` catches exactly that: on a patched wheel the volatile
    survives serialization, on an unpatched one it vanishes.

    Exposed (rather than private) because the world constructor has to gate on
    it before it builds a side, not only when it renders one.
    """

    engine = engine if engine is not None else require_poke_engine()
    try:
        supported = _cached_move_trap_supported(engine)
    except TypeError:
        # Explicit test doubles can be unhashable; production modules are not.
        supported = _move_trap_supported(engine)
    if not supported:
        raise PokeEngineMoveTrapUnsupportedError(
            "Move-trapping (Mean Look / Spider Web / Block) requires the patched "
            "poke-engine wheel; the installed engine dropped the TRAPPED volatile "
            "instead of round-tripping it, which would silently hand a trapped "
            "Pokemon its switch options back. Rebuild with "
            "scripts/setup_poke_engine.sh (patch list entry "
            "third_party/poke-engine-gen3-move-trapping.patch)."
        )


@lru_cache(maxsize=32)
def _cached_move_trap_supported(engine: Any) -> bool:
    return _move_trap_supported(engine)


def _move_trap_supported(engine: Any) -> bool:
    """Return whether the binding preserves the TRAPPED volatile end to end."""

    state_type = getattr(engine, "State", None)
    side_type = getattr(engine, "Side", None)
    if state_type is None or side_type is None:
        return False
    try:
        state = state_type(
            side_one=side_type(volatile_statuses={"trapped"}),
            side_two=side_type(),
        )
        serialized = str(state.to_string())
        if "TRAPPED" not in serialized.upper():
            return False
        # #878 made serialization a fixed point; a volatile that survives the
        # first write but not the read would still corrupt a search root.
        return str(state_type.from_string(serialized).to_string()) == serialized
    except Exception:  # noqa: BLE001 - capability checks must fail closed
        return False


def require_charge_state_support(engine: Any | None = None) -> None:
    """Prove the installed binding round-trips a two-turn charge volatile.

    Same house pattern as :func:`require_move_trap_support`, and for the same reason:
    `PokemonVolatileStatus::from_str` is generated with ``default = NONE``, so an
    engine that does not know SOLARBEAM ACCEPTS the token and silently discards it.
    Probing the round trip is the only way to tell the two apart.
    """

    engine = engine if engine is not None else require_poke_engine()
    try:
        supported = _cached_charge_state_supported(engine)
    except TypeError:
        # Explicit test doubles can be unhashable; production modules are not.
        supported = _charge_state_supported(engine)
    if not supported:
        raise PokeEngineChargeStateUnsupportedError(
            "A mid-charge (Solar Beam) world requires a poke-engine that models the "
            "charge volatile; the installed engine dropped SOLARBEAM instead of "
            "round-tripping it, which would build the charging Pokemon FREE and let "
            "search start a fresh charge instead of releasing. Rebuild with: "
            f"{POKE_ENGINE_BUILD_COMMAND}"
        )


@lru_cache(maxsize=32)
def _cached_charge_state_supported(engine: Any) -> bool:
    return _charge_state_supported(engine)


def _charge_state_supported(engine: Any) -> bool:
    """Return whether the binding preserves the SOLARBEAM volatile end to end."""

    state_type = getattr(engine, "State", None)
    side_type = getattr(engine, "Side", None)
    if state_type is None or side_type is None:
        return False
    try:
        state = state_type(
            side_one=side_type(volatile_statuses={"solarbeam"}),
            side_two=side_type(),
        )
        serialized = str(state.to_string())
        if "SOLARBEAM" not in serialized.upper():
            return False
        return str(state_type.from_string(serialized).to_string()) == serialized
    except Exception:  # noqa: BLE001 - capability checks must fail closed
        return False


def _spec_requires_rest_turns(spec: BattleSpec) -> bool:
    return any(
        int(member.rest_turns) > 0
        for side in (spec.side_one, spec.side_two)
        for member in side.pokemon
    )


def require_rest_turns_support(engine: Any | None = None) -> None:
    """Prove the installed binding preserves a Rest counter end to end.

    Same house pattern as :func:`require_move_trap_support`, and needed for the same
    reason even though ``rest_turns`` is an UPSTREAM field rather than a patched-in
    one: what this fix depends on is not that the keyword is accepted but that the
    value survives, and a binding whose serialization drops it accepts the state and
    then quietly hands search a Rest-asleep mon with a zeroed counter — an ordinary
    sleeper, clause re-armed. Probing the capability is the only way to tell those
    apart, so probe the round trip rather than the install.

    The probe compares a resting serialization against a non-resting one instead of
    reading a fixed CSV column: the Pokemon record has gained fields before (the
    gen3 Transform patch appended ``pre_transform``), and an index would silently
    start measuring the wrong one. Difference proves the value is carried; the
    ``from_string``/``to_string`` fixed point (per #878) proves it survives the read
    as well as the write, since a field that made it out but not back in would still
    corrupt a search root.
    """

    engine = engine if engine is not None else require_poke_engine()
    try:
        supported = _cached_rest_turns_supported(engine)
    except TypeError:
        # Explicit test doubles can be unhashable; production modules are not.
        supported = _rest_turns_supported(engine)
    if not supported:
        raise PokeEngineRestTurnsUnsupportedError(
            "A Rest-asleep world requires a poke-engine that round-trips "
            "Pokemon.rest_turns; the installed engine dropped it, which would build "
            "the Rest-sleeper as an ordinary sleeper and re-arm the gen3 Sleep Clause "
            "the Rest is exempt from. Rebuild with: "
            f"{POKE_ENGINE_BUILD_COMMAND}"
        )


@lru_cache(maxsize=32)
def _cached_rest_turns_supported(engine: Any) -> bool:
    return _rest_turns_supported(engine)


@lru_cache(maxsize=32)
def _cached_rest_sleep_refund_supported(engine: Any) -> bool:
    return _rest_sleep_refund_supported(engine)


def require_rest_sleep_refund_support(engine: Any | None = None) -> None:
    """Prove the installed binding preserves a pending Rest refund end to end.

    Same house pattern as :func:`require_rest_turns_support`, and needed for a
    sharper reason. The refund only ever reaches search through the pyo3
    conversion, and that direction is the one the Rust compiler does NOT protect:
    adding the field to ``PyPokemon`` forces ``E0063`` in ``From<Pokemon>`` and in
    ``#[new]``, but ``impl Into<Pokemon>`` compiles perfectly well while writing a
    literal 0. A binding built that way accepts the keyword, drops the value, and
    hands search a sleeper whose skipped turns are simply gone -- an UNDER-credit
    that no test on this side can see. Probe the round trip, not the install.
    """

    engine = engine if engine is not None else require_poke_engine()
    try:
        supported = _cached_rest_sleep_refund_supported(engine)
    except TypeError:
        # Explicit test doubles can be unhashable; production modules are not.
        supported = _rest_sleep_refund_supported(engine)
    if not supported:
        raise PokeEngineRestSleepRefundUnsupportedError(
            "An active Rest-sleeper with skipped Sleep Talk/Snore turns requires a "
            "poke-engine that round-trips Pokemon.rest_sleep_pending_refund; the "
            "installed engine dropped it, which would build the sleeper with its "
            "skipped turns simply gone and wake it EARLY on every branch that "
            "switches it back in. Rebuild with: "
            f"{POKE_ENGINE_BUILD_COMMAND}"
        )


def _rest_sleep_refund_supported(engine: Any) -> bool:
    """Return whether the binding preserves a pending Rest refund end to end."""

    state_type = getattr(engine, "State", None)
    side_type = getattr(engine, "Side", None)
    pokemon_type = getattr(engine, "Pokemon", None)
    if state_type is None or side_type is None or pokemon_type is None:
        return False

    def serialize(pending: int) -> str:
        state = state_type(
            side_one=side_type(
                pokemon=[
                    pokemon_type(
                        id="snorlax",
                        status="sleep",
                        rest_turns=1,
                        rest_sleep_pending_refund=pending,
                    )
                ]
            ),
            side_two=side_type(),
        )
        return str(state.to_string())

    try:
        # Difference proves the write is carried; the fixed point proves it
        # survives the read as well, since a field that made it out but not back
        # in would still corrupt a search root.
        pending = serialize(_REST_SLEEP_REFUND_PROBE_VALUE)
        if pending == serialize(0):
            return False
        return str(state_type.from_string(pending).to_string()) == pending
    except Exception:  # noqa: BLE001 - capability checks must fail closed
        return False


def _rest_turns_supported(engine: Any) -> bool:
    """Return whether the binding preserves a Rest counter end to end."""

    state_type = getattr(engine, "State", None)
    side_type = getattr(engine, "Side", None)
    pokemon_type = getattr(engine, "Pokemon", None)
    if state_type is None or side_type is None or pokemon_type is None:
        return False

    def serialize(rest_turns: int) -> str:
        state = state_type(
            side_one=side_type(
                pokemon=[pokemon_type(id="snorlax", status="sleep", rest_turns=rest_turns)]
            ),
            side_two=side_type(),
        )
        return str(state.to_string())

    try:
        resting = serialize(_REST_TURNS_PROBE_VALUE)
        if resting == serialize(0):
            return False
        return str(state_type.from_string(resting).to_string()) == resting
    except Exception:  # noqa: BLE001 - capability checks must fail closed
        return False


def require_mcts_entrypoint(engine: Any | None = None) -> None:
    """Prove the installed binding exposes the search entrypoint engine_search calls.

    Same house pattern as :func:`require_move_trap_support`, but a plain symbol
    check rather than a round trip: there is nothing behavioural to probe, the
    name is either present or it is not. Exposed so callers and tests can turn a
    half-installed binding into a skip that names the build command, instead of
    an AttributeError from inside the search loop.
    """

    engine = engine if engine is not None else require_poke_engine()
    missing = tuple(name for name in MCTS_ENTRYPOINT_API if not hasattr(engine, name))
    if missing:
        raise PokeEngineMctsEntrypointMissingError(
            "The installed poke-engine is missing "
            + ", ".join(missing)
            + ". That name is a pure-Python wrapper in poke-engine-py's "
            "python/poke_engine/__init__.py, so this is an install carrying the "
            "compiled extension without its Python half — it imports fine and "
            "exposes the native `mcts`, then fails only when search runs. "
            f"Rebuild with: {POKE_ENGINE_BUILD_COMMAND}"
        )


def _spec_requires_pre_transform(spec: BattleSpec) -> bool:
    return any(
        member.pre_transform is not None
        for side in (spec.side_one, spec.side_two)
        for member in side.pokemon
    )


def require_pre_transform_support(engine: Any | None = None) -> None:
    """Prove that the installed binding carries the gen3 Transform patch's snapshot.

    Same shape as :func:`require_move_trap_support`: there is no version marker
    for a local patch, so probe the CAPABILITY rather than the install. The
    round trip matters as much as the keyword being accepted — a binding that
    took the field and dropped it on serialization would leave a constructed
    Transform unrevertible in exactly the way the field exists to prevent.
    """

    engine = engine if engine is not None else require_poke_engine()
    try:
        supported = _cached_pre_transform_supported(engine)
    except TypeError:
        # Explicit test doubles can be unhashable; production modules are not.
        supported = _pre_transform_supported(engine)
    if not supported:
        raise PokeEngineTransformRevertUnsupportedError(
            "A constructed Transform carries its pre-transform base form, which "
            "requires the patched poke-engine wheel; the installed engine does "
            "not round-trip Pokemon.pre_transform, so a transformed Pokemon "
            "could never revert on switch-out. Rebuild with "
            "scripts/setup_poke_engine.sh (patch list entry "
            "third_party/poke-engine-gen3-transform.patch)."
        )


@lru_cache(maxsize=32)
def _cached_pre_transform_supported(engine: Any) -> bool:
    return _pre_transform_supported(engine)


def _pre_transform_supported(engine: Any) -> bool:
    """Return whether the binding preserves a pre-transform record end to end."""

    state_type = getattr(engine, "State", None)
    side_type = getattr(engine, "Side", None)
    pokemon_type = getattr(engine, "Pokemon", None)
    if state_type is None or side_type is None or pokemon_type is None:
        return False
    record = "ditto;1;2;3;4;5;transform:8;none:0;none:0;none:0"
    try:
        state = state_type(
            side_one=side_type(pokemon=[pokemon_type(id="ditto", pre_transform=record)]),
            side_two=side_type(),
        )
        serialized = str(state.to_string())
        if record.upper() not in serialized.upper():
            return False
        return str(state_type.from_string(serialized).to_string()) == serialized
    except Exception:  # noqa: BLE001 - capability checks must fail closed
        return False


def _spec_requires_attract(spec: BattleSpec) -> bool:
    return any(
        str(volatile).casefold() == "attract"
        for side in (spec.side_one, spec.side_two)
        for volatile in side.volatile_statuses
    )


def _require_attract_immobilization(engine: Any) -> None:
    """Prove that the installed binding has the Gen 3 Attract patch.

    There is no binding-level version marker for this local patch. Exercise the
    exact adapter rendering instead: an attracted Swords Dance user must split
    into equal move and immobilization branches. The result is cached per
    module, so a search decision never repeats the probe for each sampled world.
    """

    try:
        supported = _cached_attract_immobilization_supported(engine)
    except TypeError:
        # Explicit test doubles can be unhashable. They remain safe to probe;
        # production extension modules are hashable and use the cache above.
        supported = _attract_immobilization_supported(engine)
    if not supported:
        raise PokeEngineAttractUnsupportedError(
            "Attract requires the patched poke-engine wheel; the installed engine did not "
            "produce the required Gen 3 50/50 immobilization branches. Rebuild with "
            "third_party/poke-engine-gen3-attract.patch."
        )


@lru_cache(maxsize=32)
def _cached_attract_immobilization_supported(engine: Any) -> bool:
    return _attract_immobilization_supported(engine)


def _attract_immobilization_supported(engine: Any) -> bool:
    """Return whether a minimal adapter-rendered Attract state branches 50/50."""

    generate = getattr(engine, "generate_instructions", None)
    if not callable(generate):
        return False
    try:
        state = _build_poke_engine_state_unchecked(_attract_probe_spec(), engine)
        branches = tuple(generate(state, "swordsdance", "splash"))
        total = sum(float(branch.percentage) for branch in branches)
        moved = sum(
            float(branch.percentage)
            for branch in branches
            if any("Boost SideOne" in str(instruction) for instruction in branch.instruction_list)
        )
    except Exception:  # noqa: BLE001 - capability checks must fail closed
        return False
    immobilized = total - moved
    return (
        abs(total - 100.0) < 1e-6
        and abs(moved - 50.0) < 1e-6
        and abs(immobilized - 50.0) < 1e-6
    )


def _attract_probe_spec() -> BattleSpec:
    """A deterministic move whose only missing branch is Attract immobilization."""

    snorlax = PokemonSpec(
        id="snorlax",
        level=80,
        types=("normal",),
        hp=300,
        maxhp=300,
        attack=180,
        defense=180,
        special_attack=180,
        special_defense=180,
        speed=120,
        ability="innerfocus",
        item="none",
        moves=(MoveSpec(id="swordsdance", pp=16), MoveSpec(id="bodyslam", pp=16)),
    )
    wobbuffet = PokemonSpec(
        id="wobbuffet",
        level=80,
        types=("psychic",),
        hp=300,
        maxhp=300,
        attack=180,
        defense=180,
        special_attack=180,
        special_defense=180,
        speed=120,
        ability="shadowtag",
        item="none",
        moves=(MoveSpec(id="splash", pp=16), MoveSpec(id="tackle", pp=16)),
    )
    return BattleSpec(
        side_one=SideSpec(pokemon=(snorlax,), volatile_statuses=("attract",)),
        side_two=SideSpec(pokemon=(wobbuffet,)),
    )


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
    if side.baton_passing:
        kwargs["baton_passing"] = True
    if side.slow_uturn_move:
        kwargs["slow_uturn_move"] = True
        if side.switch_out_move_second_saved_move:
            kwargs["switch_out_move_second_saved_move"] = str(side.switch_out_move_second_saved_move)
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


# Wire format of the engine's `PreTransform` record (src/state.rs), which
# `Pokemon::deserialize` parses back:
#   id;attack;defense;special_attack;special_defense;speed;M0;M1;M2;M3
# with each move slot rendered "<move id>:<pp>". Ids are the same lowercase
# strings this adapter already hands the binding for `Pokemon.id` / `Move.id`
# (the engine's FromStr uppercases), and an unused slot is "none:0" — exactly
# what an in-search Transform writes. Pinned by
# tests/test_poke_engine_adapter.py::PreTransformSerializationTest.
_PRE_TRANSFORM_MOVE_SLOTS = 4


def _serialize_pre_transform(base: PokemonSpec, path: str) -> str:
    """Render a base-form spec into the engine's PreTransform wire string."""

    if not isinstance(base, PokemonSpec):
        raise TypeError(f"{path} must be a PokemonSpec, got {type(base).__name__}")
    if not base.id:
        raise ValueError(f"{path}.id must be a non-empty species id")
    for stat in POKEMON_STAT_FIELDS:
        _require_positive_int(getattr(base, stat), f"{path}.{stat}")
    if len(base.moves) > _PRE_TRANSFORM_MOVE_SLOTS:
        raise ValueError(
            f"{path}.moves has {len(base.moves)} slots, engine limit is "
            f"{_PRE_TRANSFORM_MOVE_SLOTS}"
        )

    slots = []
    for index in range(_PRE_TRANSFORM_MOVE_SLOTS):
        if index < len(base.moves):
            move = base.moves[index]
            if not move.id:
                raise ValueError(f"{path}.moves[{index}].id must be a non-empty move id")
            slots.append(f"{move.id}:{int(move.pp)}")
        else:
            # An absent slot is NONE at 0 PP, which keeps it out of the engine's
            # move enumeration on the way back.
            slots.append("none:0")

    fields = [base.id] + [str(getattr(base, stat)) for stat in POKEMON_STAT_FIELDS] + slots
    return ";".join(fields)


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
        # Always passed: the binding's own default is a flat
        # ("normal", "typeless"), which is wrong for every non-Normal Pokemon.
        # Nothing in the gen3 build READ base_types until Transform's switch-out
        # revert, which is why the default went unnoticed; sending the real types
        # makes the field mean what its name says without changing any search
        # decision. See BaseIdentityTest.
        "base_types": _normalize_types(
            member.types if member.base_types is None else member.base_types,
            f"{path}.base_types",
        ),
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
    if member.base_ability is not None:
        # Left unset the binding copies `ability`, which is exactly the "same as
        # current" default — so an untransformed Pokemon is byte-identical either
        # way and only a constructed Transform passes this.
        kwargs["base_ability"] = member.base_ability
    if member.item is not None:
        kwargs["item"] = member.item
    if member.nature is not None:
        kwargs["nature"] = member.nature
    if member.gender is not None:
        gender = str(member.gender).strip().upper()
        mapped_gender = {"M": "male", "F": "female", "N": "none"}.get(gender)
        if mapped_gender is None:
            raise ValueError(f"{path}.gender must be M, F, N, or None, got {member.gender!r}")
        kwargs["gender"] = mapped_gender
    if _require_non_negative_int(member.rest_turns, f"{path}.rest_turns"):
        kwargs["rest_turns"] = member.rest_turns
    if _require_non_negative_int(
        member.rest_sleep_pending_refund, f"{path}.rest_sleep_pending_refund"
    ):
        # Fails closed on a sum the engine cannot represent. The engine clamps to 3
        # because a wake match on rest_turns > 3 PANICS and search has no refusal
        # channel at that point -- but the constructor does have one, and refusing
        # a decision beats silently truncating a refund into a plausible counter.
        total = int(member.rest_turns) + int(member.rest_sleep_pending_refund)
        if total > _REST_SLEEP_TURNS_MAX:
            raise ValueError(
                f"{path}: rest_turns + rest_sleep_pending_refund must be "
                f"<= {_REST_SLEEP_TURNS_MAX}, got {total}"
            )
        kwargs["rest_sleep_pending_refund"] = member.rest_sleep_pending_refund
    if _require_non_negative_int(member.sleep_turns, f"{path}.sleep_turns"):
        kwargs["sleep_turns"] = member.sleep_turns
    if member.pre_transform is not None:
        kwargs["pre_transform"] = _serialize_pre_transform(
            member.pre_transform, f"{path}.pre_transform"
        )
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
