"""Belief-sampled world -> poke-engine battle state (engine swap plan v3, track A).

This module is the world constructor of ``docs/test_time_search_plan_v3.md``:
it turns the exact pair the direct Node materialization consumes — a
:class:`~pokezero.local_showdown.PublicBattleMaterializationState` (the public
branch point) plus a :class:`~pokezero.env.BattleStartOverride` (the
belief-sampled determinized world as packed teams) — into a
:class:`~pokezero.poke_engine_adapter.BattleSpec` and, from there, a native
``poke_engine.State``.

Design rules (frozen in the v3 plan):

- **Pure function of its inputs.** No env, bridge, or live-battle access. The
  public overlay is produced by the same ``_public_materialization_payload``
  helper the Node direct path uses, so the two paths cannot drift on the
  public half of the construction.
- **Anti-leakage by construction.** ``PublicBattleMaterializationState``
  strips all request payloads except the acting player's own, and the
  opponent's team comes exclusively from the belief-sampled packed team. The
  P-1 checksum gate upstream is unaffected.
- **Fail closed.** Any public effect this mapping cannot express exactly
  raises :class:`EngineWorldUnsupported` with a stable ``reason`` slug; the
  caller falls back to the sim-backed path. No approximations are silently
  substituted (approximations, when accepted, must be explicit exemptions in
  the golden-corpus sense).

GPL note: ``third_party/foul-play`` was used strictly as behavioral reference
for poke-engine's construction conventions; no code is copied from it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from .dex import ShowdownDex, normalize_id
from .env import BattleStartOverride
from .gen3_damage import gen3_hp_stat, gen3_stat
from .poke_engine_adapter import BattleSpec, MoveSpec, PokemonSpec, SideSpec, build_poke_engine_state
from .showdown_fixture import FixturePokemon, _STAT_ORDER

_MAX_IV = 31
_PLAYER_SLOTS = ("p1", "p2")
_NEUTRAL_NATURES = frozenset({"", "serious", "hardy", "docile", "bashful", "quirky"})

# Showdown replay weather ids -> poke-engine weather names (Gen 3 set).
_WEATHER_IDS = {
    "raindance": "rain",
    "rain": "rain",
    "sunnyday": "sun",
    "sun": "sun",
    "sandstorm": "sand",
    "sand": "sand",
    "hail": "hail",
}

# Public volatile ids this construction expresses exactly today. Everything
# else fails closed (substitute needs public sub-health bookkeeping; confusion
# and kin need duration state the public replay does not carry yet).
_SUPPORTED_VOLATILES = frozenset({"leechseed"})

# Showdown boost keys -> adapter SideSpec boost keys.
_BOOST_KEYS = {
    "atk": "attack",
    "def": "defense",
    "spa": "special_attack",
    "spd": "special_defense",
    "spe": "speed",
    "accuracy": "accuracy",
    "evasion": "evasion",
}

# Showdown side-condition ids -> poke_engine.SideConditions field names (Gen 3).
_SIDE_CONDITION_IDS = {
    "spikes": "spikes",
    "reflect": "reflect",
    "lightscreen": "light_screen",
    "safeguard": "safeguard",
    "mist": "mist",
}

# Gen 3 timed side conditions (5 turns, no extension items in Gen 3). The
# public payload stores these as presence flags plus a set turn; poke-engine's
# SideConditions fields for them are TURNS-REMAINING counters, so the count
# must be derived — copying the flag through would make every screen expire
# after one search turn.
_TIMED_SIDE_CONDITIONS = frozenset({"reflect", "lightscreen", "safeguard", "mist"})
_TIMED_SIDE_CONDITION_TURNS = 5

# Showdown status codes -> poke-engine status names. ``slp`` is deliberately
# absent from the strict map: public state does not carry sleep/rest turn
# counts yet, and guessing them biases wake-up odds (fail closed by default).
# ``approximate_sleep_turns=True`` opts into mapping slp with sleep_turns=0
# ("just fell asleep") — a documented approximation for search POCs; the real
# fix is public sleep-counter tracking in the replay state.
_STATUS_CODES = {
    "": "none",
    "brn": "burn",
    "par": "paralyze",
    "psn": "poison",
    "tox": "toxic",
    "frz": "freeze",
}
_SLEEP_STATUS_CODE = "slp"

_MOVE_SLOT_LIMIT = 4
_MANUAL_WEATHER_TURNS = 5

# Gen 3 Hidden Power derivation (type from IV low bits, BP from IV second bits).
# poke-engine's gen3 move table only knows fully-qualified ids like
# ``hiddenpowergrass70``; the randbats set pool stores ``hiddenpowergrass`` and
# Showdown requests report plain ``hiddenpower``, so both must be translated.
_HP_TYPE_ORDER = (
    "fighting", "flying", "poison", "ground", "rock", "bug", "ghost", "steel",
    "fire", "water", "grass", "electric", "psychic", "ice", "dragon", "dark",
)
_HP_STAT_BITS = ("hp", "atk", "def", "spe", "spa", "spd")


def hidden_power_engine_id(move_id: str, ivs: Mapping[str, int] | None) -> str:
    """Translate a hiddenpower id into poke-engine's typed+BP gen3 id.

    Raises :class:`EngineWorldUnsupported` when the id carries a type that the
    IVs do not produce (an inconsistent sampled set must not be silently
    reinterpreted).
    """

    suffix = move_id[len("hiddenpower"):]
    iv = lambda stat: int((ivs or {}).get(stat, _MAX_IV))
    type_bits = sum(((iv(stat) & 1) << index) for index, stat in enumerate(_HP_STAT_BITS))
    iv_type = _HP_TYPE_ORDER[type_bits * 15 // 63]
    bp_bits = sum((((iv(stat) >> 1) & 1) << index) for index, stat in enumerate(_HP_STAT_BITS))
    base_power = 30 + bp_bits * 40 // 63
    if suffix and suffix != iv_type:
        raise EngineWorldUnsupported(
            "hidden_power_iv_mismatch",
            f"move {move_id!r} disagrees with IV-derived type {iv_type!r}",
        )
    return f"hiddenpower{iv_type}{base_power}"


def _engine_species_id(species_id: str) -> str:
    """Collapse cosmetic formes to the id the dex/engine know (Unown letters).

    Applied to ENGINE-facing ids only (`PokemonSpec.id`, stats/dex lookups).
    ``EngineWorld.party_species`` deliberately keeps the sampled team's OWN
    species ids (protocol/request convention, e.g. ``unownc``): the leaf
    path's event context contract is "display species in engine party order"
    — synthesized protocol lines and md-team matching must land on the same
    species keys the real protocol and the request use, not the engine's
    collapsed base id.
    """

    if species_id.startswith("unown"):
        return "unown"
    return species_id


class EngineWorldUnsupported(ValueError):
    """A public effect the engine-world construction cannot express exactly.

    ``reason`` is a stable slug for fallback telemetry; ``detail`` carries the
    human-readable specifics.
    """

    def __init__(self, reason: str, detail: str) -> None:
        super().__init__(f"{reason}: {detail}")
        self.reason = reason
        self.detail = detail


# ---------------------------------------------------------------------------------------------
# Packed-team parsing (inverse of showdown_fixture.pack_pokemon / pack_team).
# ---------------------------------------------------------------------------------------------


def unpack_pokemon(packed: str) -> FixturePokemon:
    """Parse one Showdown packed set back into a :class:`FixturePokemon`.

    Mirrors ``showdown_fixture.pack_pokemon`` field for field: empty EV slots
    mean 0, empty IV slots mean 31, empty level means 100, and the species is
    recovered from the name field when the species field is blank.
    """

    parts = packed.split("|")
    if len(parts) < 12:
        raise ValueError(f"packed set has {len(parts)} fields, expected at least 12: {packed!r}")
    name, species, item, ability, moves, nature, evs, gender, ivs, _shiny, level, _tail = parts[:12]
    resolved_species = species or name
    if not resolved_species:
        raise ValueError(f"packed set is missing a species: {packed!r}")
    move_ids = tuple(move for move in moves.split(",") if move)
    if not move_ids:
        raise ValueError(f"packed set for {resolved_species!r} has no moves")
    return FixturePokemon(
        species=resolved_species,
        moves=move_ids,
        ability=ability or None,
        item=item or None,
        level=int(level) if level else 100,
        nature=nature or "",
        gender=gender or None,
        evs=_unpack_spread(evs, default=0),
        ivs=_unpack_spread(ivs, default=_MAX_IV),
    )


def unpack_team(packed: str) -> tuple[FixturePokemon, ...]:
    """Parse a ``]``-delimited packed team string."""

    if not packed:
        raise ValueError("packed team string is empty")
    return tuple(unpack_pokemon(entry) for entry in packed.split("]"))


def _unpack_spread(packed: str, *, default: int) -> dict[str, int]:
    if not packed:
        return {stat: default for stat in _STAT_ORDER}
    values = packed.split(",")
    if len(values) != len(_STAT_ORDER):
        raise ValueError(f"packed spread has {len(values)} slots, expected {len(_STAT_ORDER)}: {packed!r}")
    return {
        stat: int(value) if value else default
        for stat, value in zip(_STAT_ORDER, values)
    }


# ---------------------------------------------------------------------------------------------
# Payload -> BattleSpec construction.
# ---------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class EngineWorld:
    """A constructed engine-side world plus the identity maps search needs.

    Party order is the SAMPLED OVERRIDE order, not the request's active-first
    ``selfTeamOrder`` permutation. Consumers must map switch choices through
    ``party_species`` — never through raw request indices. (The species sets
    are checked for consistency at construction; only the ordering differs.)
    """

    spec: BattleSpec
    # Which BattleSpec side ("side_one"/"side_two") each player slot landed on.
    slot_sides: Mapping[str, str]
    # Party order per player slot, as normalized species ids (engine party order).
    party_species: Mapping[str, tuple[str, ...]]


def battle_spec_from_payload(
    payload: Mapping[str, Any],
    override: BattleStartOverride,
    *,
    dex: ShowdownDex,
    approximate_sleep_turns: bool = False,
    approximate_substitute_health: bool = False,
    blocked_slots: Mapping[str, str] | None = None,
    encored_moves: Mapping[str, str] | None = None,
    recharging_slots: Sequence[str] = (),
    truant_slots: Sequence[str] = (),
    rng: Any | None = None,
) -> EngineWorld:
    """Pure construction: public materialization payload + sampled teams -> spec.

    ``payload`` must be the dict produced by
    ``local_showdown._public_materialization_payload`` (or a test literal of
    the same shape); ``override`` supplies both sides' belief-sampled packed
    teams. Raises :class:`EngineWorldUnsupported` whenever the position holds
    public state this construction cannot express exactly.
    """

    _reject_unsupported_globals(payload)

    sides_payload = payload.get("sides")
    if not isinstance(sides_payload, Mapping):
        raise EngineWorldUnsupported("payload_malformed", "payload has no sides mapping")

    self_player = str(payload.get("selfPlayer") or "")
    if self_player not in _PLAYER_SLOTS:
        raise EngineWorldUnsupported("payload_malformed", f"selfPlayer {self_player!r} is not a player slot")
    request_kind = str(payload.get("selfRequestKind") or "")
    if request_kind not in ("move", "force-switch"):
        raise EngineWorldUnsupported(
            "boundary_not_move_request",
            f"self request kind {request_kind!r} is not supported",
        )
    self_force_switch = request_kind == "force-switch"
    pending_baton_pass = tuple(str(s) for s in payload.get("pendingBatonPassSides") or ())
    self_baton_passing = False
    if pending_baton_pass:
        # Supported shape: OUR Baton Pass is pending and we are choosing the
        # recipient (boosts/whitelisted volatiles pass; the engine restricts
        # us to switch choices). The opponent's committed-but-hidden action
        # is sampled per world into switch_out_move_second_saved_move, but
        # review probes show the gen3 engine build does NOT resolve it after
        # the switch (root values are invariant to the sample) — the
        # recipient enters without eating the committed move, a fail-soft
        # optimistic under-model. The field is populated for forward
        # compatibility; treat the commitment as UNMODELED today. Any other
        # pending shape stays fail-closed.
        if set(pending_baton_pass) != {self_player} or not self_force_switch:
            raise EngineWorldUnsupported(
                "pending_baton_pass",
                f"unsupported pending Baton Pass shape: {pending_baton_pass!r} (kind {request_kind!r})",
            )
        self_baton_passing = True
    request_state = payload.get("selfActiveRequestState")
    if isinstance(request_state, Mapping):
        raised = sorted(flag for flag, value in request_state.items() if value)
        if raised:
            raise EngineWorldUnsupported(
                "self_request_state_unsupported",
                f"self active request flags {raised} constrain legality beyond this construction",
            )

    turn = payload.get("turn")
    if not isinstance(turn, int):
        raise EngineWorldUnsupported("payload_malformed", "payload has no integer turn")
    for blocked_slot, block_reason in (blocked_slots or {}).items():
        raise EngineWorldUnsupported(
            "public_effect_blocked",
            f"slot {blocked_slot!r}: {block_reason} (caller-declared unexpressible public effect)",
        )

    built_sides: dict[str, SideSpec] = {}
    party_species: dict[str, tuple[str, ...]] = {}
    for slot in _PLAYER_SLOTS:
        side_payload = sides_payload.get(slot)
        if not isinstance(side_payload, Mapping):
            raise EngineWorldUnsupported("payload_malformed", f"side {slot!r} is missing")
        packed = override.player_teams.get(slot)
        if not packed:
            raise EngineWorldUnsupported("override_side_missing", f"override has no packed team for {slot!r}")
        team = unpack_team(packed)
        is_self_slot = slot == self_player
        built_sides[slot], species_order = _build_side_spec(
            slot=slot,
            side_payload=side_payload,
            team=team,
            dex=dex,
            is_self=is_self_slot,
            turn=turn,
            self_benched_move_history=bool(payload.get("selfBenchedMoveHistory")),
            approximate_sleep_turns=approximate_sleep_turns,
            approximate_substitute_health=approximate_substitute_health,
            force_switch=is_self_slot and self_force_switch,
            baton_passing=is_self_slot and self_baton_passing,
            opponent_committed_pending=(not is_self_slot) and self_baton_passing,
            wish_set_turn=_wish_set_turn(payload, slot),
            encored_move=(encored_moves or {}).get(slot),
            must_recharge=slot in (recharging_slots or ()),
            truant_loafs=slot in (truant_slots or ()),
            rng=rng,
        )
        party_species[slot] = species_order

    self_order = payload.get("selfTeamOrder")
    if isinstance(self_order, Sequence) and not isinstance(self_order, str):
        order_ids = {normalize_id(str(species)) for species in self_order}
        if order_ids and order_ids != set(party_species[self_player]):
            raise EngineWorldUnsupported(
                "self_world_mismatch",
                f"request team {sorted(order_ids)} != sampled world {sorted(party_species[self_player])}",
            )

    weather, weather_turns = _weather_fields(payload)
    spec = BattleSpec(
        side_one=built_sides["p1"],
        side_two=built_sides["p2"],
        weather=weather,
        weather_turns_remaining=weather_turns,
    )
    return EngineWorld(
        spec=spec,
        slot_sides={"p1": "side_one", "p2": "side_two"},
        party_species=party_species,
    )


def world_battle_spec(
    state: Any,
    override: BattleStartOverride,
    *,
    dex: ShowdownDex,
    approximate_sleep_turns: bool = False,
    approximate_substitute_health: bool = False,
    blocked_slots: Mapping[str, str] | None = None,
    encored_moves: Mapping[str, str] | None = None,
    recharging_slots: Sequence[str] = (),
    truant_slots: Sequence[str] = (),
    rng: Any | None = None,
) -> EngineWorld:
    """Construct the engine world for a live public branch point.

    ``state`` is a ``PublicBattleMaterializationState``; the public overlay is
    computed by the same payload helper the Node direct path uses. Deferred
    opponent actions are deliberately not forwarded: boundaries that need them
    fail closed (``boundary_not_move_request`` / ``pending_baton_pass``).
    """

    from .local_showdown import _public_materialization_payload

    payload = _public_materialization_payload(state)
    return battle_spec_from_payload(
        payload,
        override,
        dex=dex,
        approximate_sleep_turns=approximate_sleep_turns,
        approximate_substitute_health=approximate_substitute_health,
        blocked_slots=blocked_slots,
        encored_moves=encored_moves,
        recharging_slots=recharging_slots,
        truant_slots=truant_slots,
        rng=rng,
    )


def build_engine_world(
    state: Any,
    override: BattleStartOverride,
    *,
    dex: ShowdownDex,
    module: Any | None = None,
) -> tuple[EngineWorld, Any]:
    """World constructor end point: returns ``(EngineWorld, poke_engine.State)``."""

    world = world_battle_spec(state, override, dex=dex)
    return world, build_poke_engine_state(world.spec, module=module)


# ---------------------------------------------------------------------------------------------
# Internals.
# ---------------------------------------------------------------------------------------------


def _reject_unsupported_globals(payload: Mapping[str, Any]) -> None:
    if payload.get("deferredOpponentActions") or payload.get("deferredOpponentActionPriors"):
        raise EngineWorldUnsupported("deferred_opponent_action", "deferred opponent actions are not supported")
    future_sight = payload.get("futureSight")
    if isinstance(future_sight, Mapping) and any(int(v) for v in future_sight.values()):
        raise EngineWorldUnsupported("future_sight_pending", "a Future Sight strike is pending")


def _wish_set_turn(payload: Mapping[str, Any], slot: str) -> int | None:
    wish_turns = payload.get("wishSetTurns")
    if not isinstance(wish_turns, Mapping):
        return None
    value = wish_turns.get(slot)
    return value if isinstance(value, int) else None


def _weather_fields(payload: Mapping[str, Any]) -> tuple[str, int]:
    raw = payload.get("weather")
    weather_id = normalize_id(str(raw)) if raw else ""
    if not weather_id or weather_id == "none":
        return "none", -1
    weather = _WEATHER_IDS.get(weather_id)
    if weather is None:
        raise EngineWorldUnsupported("weather_unsupported", f"weather {raw!r} has no Gen 3 engine mapping")
    if payload.get("weatherFromAbility"):
        return weather, -1
    turn = payload.get("turn")
    set_turn = payload.get("weatherSetTurn")
    if not isinstance(turn, int) or not isinstance(set_turn, int):
        raise EngineWorldUnsupported("weather_turns_unknown", "manual weather without turn bookkeeping")
    remaining = _MANUAL_WEATHER_TURNS - (turn - set_turn)
    if remaining <= 0:
        raise EngineWorldUnsupported(
            "weather_turns_inconsistent",
            f"manual weather set on turn {set_turn} would have expired by turn {turn}",
        )
    return weather, remaining


def _build_side_spec(
    *,
    slot: str,
    side_payload: Mapping[str, Any],
    team: Sequence[FixturePokemon],
    dex: ShowdownDex,
    is_self: bool,
    turn: int,
    self_benched_move_history: bool,
    approximate_sleep_turns: bool = False,
    approximate_substitute_health: bool = False,
    force_switch: bool = False,
    wish_set_turn: int | None = None,
    encored_move: str | None = None,
    must_recharge: bool = False,
    truant_loafs: bool = False,
    baton_passing: bool = False,
    opponent_committed_pending: bool = False,
    rng: Any | None = None,
) -> tuple[SideSpec, tuple[str, ...]]:
    blockers = side_payload.get("materializationBlockers")
    if blockers:
        raise EngineWorldUnsupported("materialization_blocker", f"{slot}: {', '.join(map(str, blockers))}")

    rows = side_payload.get("pokemon")
    if not isinstance(rows, Sequence) or isinstance(rows, str):
        raise EngineWorldUnsupported("payload_malformed", f"side {slot!r} has no pokemon rows")
    rows_by_species: dict[str, Mapping[str, Any]] = {}
    for row in rows:
        if not isinstance(row, Mapping):
            raise EngineWorldUnsupported("payload_malformed", f"side {slot!r} has a non-mapping pokemon row")
        rows_by_species[normalize_id(str(row.get("species") or ""))] = row

    party: list[PokemonSpec] = []
    species_order: list[str] = []
    active_index: int | None = None
    for mon in team:
        species_id = normalize_id(mon.species)
        row = rows_by_species.pop(species_id, None)
        member = _build_pokemon_spec(
            mon,
            row,
            dex=dex,
            slot=slot,
            is_self=is_self,
            self_benched_move_history=self_benched_move_history,
            approximate_sleep_turns=approximate_sleep_turns,
        )
        if row is not None and bool(row.get("active")):
            if active_index is not None:
                raise EngineWorldUnsupported("payload_malformed", f"side {slot!r} has two active rows")
            active_index = len(party)
        party.append(member)
        # The sampled team's own id, NOT the engine-collapsed base id: this is
        # the ctx/party_species surface (request/protocol naming convention).
        # Pre-fix the collapse here made an Unown team trip the
        # self_world_mismatch guard on every decision (request "unownc" vs
        # world "unown") and broke ctx→md species matching on the leaf path.
        species_order.append(species_id)
    if rows_by_species:
        raise EngineWorldUnsupported(
            "public_species_not_in_world",
            f"side {slot!r} public rows not covered by the sampled world: {sorted(rows_by_species)}",
        )
    if active_index is None:
        raise EngineWorldUnsupported("payload_malformed", f"side {slot!r} has no active row")

    volatiles = [normalize_id(str(v)) for v in side_payload.get("volatiles") or ()]
    supported = _SUPPORTED_VOLATILES | ({"substitute"} if approximate_substitute_health else set())
    if "encore" in volatiles:
        supported = supported | {"encore"}
    if must_recharge:
        # Publicly-forced recharge turn (Hyper Beam landed last round): the
        # engine's MUSTRECHARGE volatile restricts the side to "No Move" —
        # without it, searched worlds hand the recharging mon a free action.
        volatiles = volatiles + ["mustrecharge"]
        supported = supported | {"mustrecharge"}
    if truant_loafs:
        # Truant phase is publicly derivable (acted last round -> loafs now).
        # The engine models the alternation once seeded with the volatile;
        # without it every sampled world has Slaking about to act.
        volatiles = volatiles + ["truant"]
        supported = supported | {"truant"}
    unsupported = sorted(set(volatiles) - supported)
    if unsupported:
        raise EngineWorldUnsupported("volatile_unsupported", f"side {slot!r}: {unsupported}")
    substitute_health = 0
    if "substitute" in volatiles:
        # Public info does not carry the sub's remaining HP; a fresh sub costs
        # maxhp/4, so that is the documented upper-bound approximation.
        substitute_health = party[active_index].maxhp // 4

    boosts: dict[str, int] = {}
    for key, value in (side_payload.get("boosts") or {}).items():
        mapped = _BOOST_KEYS.get(str(key))
        if mapped is None:
            raise EngineWorldUnsupported("boost_unsupported", f"side {slot!r} boost key {key!r}")
        if int(value):
            boosts[mapped] = int(value)

    set_turns = side_payload.get("sideConditionSetTurns") or {}
    side_conditions: dict[str, int] = {}
    for key, value in (side_payload.get("sideConditions") or {}).items():
        condition_id = normalize_id(str(key))
        mapped = _SIDE_CONDITION_IDS.get(condition_id)
        if mapped is None:
            raise EngineWorldUnsupported("side_condition_unsupported", f"side {slot!r} condition {key!r}")
        if not int(value):
            continue
        if condition_id in _TIMED_SIDE_CONDITIONS:
            # The payload stores a presence flag; the engine field counts turns
            # remaining. Derive it or refuse — never copy the flag through.
            set_turn = set_turns.get(key, set_turns.get(condition_id))
            if not isinstance(set_turn, int):
                raise EngineWorldUnsupported(
                    "side_condition_turns_unknown",
                    f"side {slot!r} timed condition {key!r} has no set turn",
                )
            remaining = _TIMED_SIDE_CONDITION_TURNS - (turn - set_turn)
            if remaining <= 0:
                raise EngineWorldUnsupported(
                    "side_condition_turns_inconsistent",
                    f"side {slot!r} condition {key!r} set on turn {set_turn} would have expired by turn {turn}",
                )
            side_conditions[mapped] = remaining
        else:
            side_conditions[mapped] = int(value)
    toxic_stage = side_payload.get("toxicStage")
    if isinstance(toxic_stage, int) and toxic_stage > 0:
        side_conditions["toxic_count"] = toxic_stage

    wish = (0, 0)
    if wish_set_turn is not None:
        remaining = 2 - (turn - wish_set_turn)
        if remaining not in (1, 2):
            raise EngineWorldUnsupported(
                "wish_turns_inconsistent",
                f"side {slot!r} wish set on turn {wish_set_turn} at turn {turn}",
            )
        # Timing verified against the engine (counter=1 heals end of this
        # turn). The amount is IGNORED by poke-engine, which heals the
        # resolving active's maxhp/2 — a known low-severity deviation from
        # gen3 (true heal = the CASTER's maxhp/2); we pass the active's
        # value for forward compatibility should the engine start using it.
        wish = (remaining, party[active_index].maxhp // 2)

    last_used_move = ""
    volatile_durations: dict[str, int] = {}
    if "encore" in volatiles:
        active_specs = party[active_index].moves
        encored_index = _resolve_encored_move_index(
            active_specs,
            rows_for_active=(
                _active_row_moves(side_payload) if is_self else None
            ),
            encored_move=encored_move,
        )
        if encored_index is None:
            raise EngineWorldUnsupported(
                "encore_move_unknown",
                f"side {slot!r} is encored but the locked move cannot be determined",
            )
        # The engine restricts the side to last_used_move while ENCORE is set
        # (verified empirically); it does not decrement the duration, so the
        # mon is modeled as locked for the search horizon. Conservative for an
        # OPPONENT's encore; pessimistic for our own near expiry (real gen3
        # encores run 3-8 turns) — acceptable over short MCTS horizons.
        last_used_move = f"move:{encored_index}"
        volatile_durations["encore"] = 1

    slow_uturn_move = False
    saved_move = ""
    if opponent_committed_pending:
        active_specs = [spec for spec in party[active_index].moves if spec.id != "none" and not spec.disabled]
        if not active_specs:
            raise EngineWorldUnsupported(
                "pending_baton_pass", f"side {slot!r} has no sampleable committed move"
            )
        if rng is None:
            raise EngineWorldUnsupported(
                "pending_baton_pass", "committed-move sampling requires an rng"
            )
        slow_uturn_move = True
        saved_move = active_specs[rng.randrange(len(active_specs))].id

    return (
        SideSpec(
            pokemon=tuple(party),
            active_index=active_index,
            side_conditions=side_conditions,
            boosts=boosts,
            volatile_statuses=tuple(volatiles),
            substitute_health=substitute_health,
            force_switch=force_switch,
            baton_passing=baton_passing,
            slow_uturn_move=slow_uturn_move,
            switch_out_move_second_saved_move=saved_move,
            wish=wish,
            last_used_move=last_used_move,
            volatile_status_durations=volatile_durations,
        ),
        tuple(species_order),
    )


def _active_row_moves(side_payload: Mapping[str, Any]) -> list[Mapping[str, Any]] | None:
    rows = side_payload.get("pokemon")
    if not isinstance(rows, Sequence) or isinstance(rows, str):
        return None
    for row in rows:
        if isinstance(row, Mapping) and bool(row.get("active")):
            moves = row.get("moves")
            if isinstance(moves, Sequence) and not isinstance(moves, str):
                return [move for move in moves if isinstance(move, Mapping)]
    return None


def _resolve_encored_move_index(
    move_specs: Sequence[Any],
    *,
    rows_for_active: Sequence[Mapping[str, Any]] | None,
    encored_move: str | None,
) -> int | None:
    """Index of the encored move in the constructed move order, or None.

    Self side: the request marks every non-encored move disabled, so exactly
    one enabled move identifies the lock. Opponent side: the caller passes the
    publicly-observed last move (``encored_move``).
    """

    if encored_move:
        target = normalize_id(encored_move)
        for index, spec in enumerate(move_specs):
            spec_id = normalize_id(spec.id)
            if spec_id == target:
                return index
            if target.startswith("hiddenpower") and spec_id.startswith("hiddenpower"):
                return index
        return None
    if rows_for_active:
        enabled = [
            normalize_id(str(move.get("id")))
            for move in rows_for_active
            if isinstance(move.get("id"), str) and not bool(move.get("disabled"))
        ]
        if len(enabled) == 1:
            target = enabled[0]
            for index, spec in enumerate(move_specs):
                spec_id = normalize_id(spec.id)
                if spec_id == target:
                    return index
                if target.startswith("hiddenpower") and spec_id.startswith("hiddenpower"):
                    return index
    return None


def _build_pokemon_spec(
    mon: FixturePokemon,
    row: Mapping[str, Any] | None,
    *,
    dex: ShowdownDex,
    slot: str,
    is_self: bool,
    self_benched_move_history: bool = False,
    approximate_sleep_turns: bool = False,
) -> PokemonSpec:
    species_id = _engine_species_id(normalize_id(mon.species))
    info = dex.species_info(species_id)
    if info is None:
        raise EngineWorldUnsupported("species_unknown", f"{slot}: {mon.species!r} is not in the Gen 3 dex")
    nature = normalize_id(mon.nature) if mon.nature else ""
    if nature not in _NEUTRAL_NATURES:
        raise EngineWorldUnsupported(
            "nature_not_neutral",
            f"{slot}: {mon.species!r} has nature {mon.nature!r} (Gen 3 randbats sets are neutral)",
        )

    evs = mon.evs or {}
    ivs = mon.ivs or {}
    base_hp = int(info.base_stats.get("hp", 0))
    # Shedinja: the generator (and gen3_damage.randbats_spread_details) pin
    # max HP to 1 when base HP is 1; the raw formula would give ~164 and a
    # "1/1" public condition would then fraction-scale into an unkillable
    # Shedinja in searched worlds (audit finding, 2026-07-18).
    maxhp = 1 if base_hp == 1 else gen3_hp_stat(base_hp, int(ivs.get("hp", _MAX_IV)), int(evs.get("hp", 0)), mon.level)
    stats = {
        stat: gen3_stat(
            int(info.base_stats.get(stat, 0)),
            int(ivs.get(stat, _MAX_IV)),
            int(evs.get(stat, 0)),
            mon.level,
        )
        for stat in ("atk", "def", "spa", "spd", "spe")
    }

    hp, status = _hp_and_status(
        row,
        maxhp=maxhp,
        slot=slot,
        species=mon.species,
        is_self=is_self,
        approximate_sleep_turns=approximate_sleep_turns,
    )
    moves = _move_specs(
        mon,
        row,
        dex=dex,
        slot=slot,
        is_self=is_self,
        self_benched_move_history=self_benched_move_history,
    )

    return PokemonSpec(
        id=species_id,
        level=mon.level,
        types=info.types,
        hp=hp,
        maxhp=maxhp,
        attack=stats["atk"],
        defense=stats["def"],
        special_attack=stats["spa"],
        special_defense=stats["spd"],
        speed=stats["spe"],
        moves=moves,
        status=status,
        ability=normalize_id(mon.ability) if mon.ability else None,
        item=normalize_id(mon.item) if mon.item else None,
        weight_kg=info.weight_kg if info.weight_kg > 0 else None,
    )


def _hp_and_status(
    row: Mapping[str, Any] | None,
    *,
    maxhp: int,
    slot: str,
    species: str,
    is_self: bool,
    approximate_sleep_turns: bool = False,
) -> tuple[int, str]:
    if row is None:
        return maxhp, "none"
    condition = str(row.get("condition") or "")
    if not condition:
        raise EngineWorldUnsupported("payload_malformed", f"{slot}: {species!r} row has no condition")
    hp_part, _, status_part = condition.partition(" ")
    status_code = status_part.strip()
    if status_code == "fnt" or hp_part == "0":
        return 0, "none"
    current_raw, _, max_raw = hp_part.partition("/")
    try:
        current = int(current_raw)
        denominator = int(max_raw) if max_raw else maxhp
    except ValueError as error:
        raise EngineWorldUnsupported(
            "payload_malformed", f"{slot}: {species!r} condition {condition!r} is not parseable"
        ) from error
    if denominator <= 0 or not 0 <= current <= denominator:
        raise EngineWorldUnsupported(
            "payload_malformed", f"{slot}: {species!r} condition {condition!r} is out of range"
        )
    if is_self and denominator != maxhp:
        # The acting player's request reports exact max HP. A mismatch means the
        # stat computation disagrees with the sim — never scale over it.
        raise EngineWorldUnsupported(
            "self_maxhp_mismatch",
            f"{slot}: {species!r} request max HP {denominator} != computed {maxhp}",
        )
    if denominator == maxhp:
        hp = current
    else:
        # Public opponent HP is fraction-of-100; scale onto the sampled set's
        # computed max HP. Rounding here is a documented exemption candidate.
        hp = max(1, round(current * maxhp / denominator)) if current else 0
    status = _STATUS_CODES.get(status_code)
    if status is None:
        if status_code == _SLEEP_STATUS_CODE and approximate_sleep_turns:
            # Documented approximation: model the mon as freshly asleep
            # (sleep_turns=0). Biases wake-up odds late in a sleep; the exact
            # fix is public sleep-counter tracking in the replay state.
            return hp, "sleep"
        raise EngineWorldUnsupported(
            "status_unsupported",
            f"{slot}: {species!r} status {status_code!r} (sleep needs public turn counts)",
        )
    return hp, status


def _move_specs(
    mon: FixturePokemon,
    row: Mapping[str, Any] | None,
    *,
    dex: ShowdownDex,
    slot: str,
    is_self: bool,
    self_benched_move_history: bool = False,
) -> tuple[MoveSpec, ...]:
    if len(mon.moves) > _MOVE_SLOT_LIMIT:
        raise EngineWorldUnsupported(
            "payload_malformed", f"{slot}: {mon.species!r} has {len(mon.moves)} moves"
        )
    known_pp: dict[str, tuple[int, bool]] = {}
    if is_self and row is not None:
        for entry in row.get("moves") or ():
            if not isinstance(entry, Mapping) or not isinstance(entry.get("id"), str):
                continue
            pp = entry.get("pp")
            if isinstance(pp, int):
                known_pp[normalize_id(entry["id"])] = (pp, bool(entry.get("disabled")))

    if is_self and known_pp:
        sampled_ids = {normalize_id(move) for move in mon.moves}
        sampled_has_hp = any(m.startswith("hiddenpower") for m in sampled_ids)
        for request_move in known_pp:
            if request_move in sampled_ids:
                continue
            if request_move.startswith("hiddenpower") and sampled_has_hp:
                continue
            raise EngineWorldUnsupported(
                "self_moveset_mismatch",
                f"{slot}: request-known move {request_move!r} is absent from the sampled "
                f"moveset (Transform/Mimic-class desync)",
            )

    specs: list[MoveSpec] = []
    for move in mon.moves:
        move_id = normalize_id(move)
        # Request-known PP rows report Hidden Power as plain "hiddenpower";
        # match on that base before translating to the engine's typed+BP id.
        pp_keys = (move_id, "hiddenpower") if move_id.startswith("hiddenpower") else (move_id,)
        pp_key = next((key for key in pp_keys if key in known_pp), None)
        if pp_key is not None:
            pp, disabled = known_pp[pp_key]
        else:
            if is_self and self_benched_move_history:
                # A benched self mon has spent PP somewhere and this slot has no
                # cached PP snapshot — catalog full PP would be wrong for our
                # own side, where exactness is available. Fail closed.
                raise EngineWorldUnsupported(
                    "self_pp_unknown",
                    f"{slot}: {mon.species!r} move {move!r} has no request-known PP",
                )
            info = dex.move_info(move_id)
            max_pp = info.max_pp if info is not None else 0
            if max_pp <= 0:
                raise EngineWorldUnsupported(
                    "move_unknown", f"{slot}: {mon.species!r} move {move!r} has no catalog PP"
                )
            # Opponent PP decrements are not tracked publicly yet: full PP is a
            # documented exemption (see the v3 plan's exemption rule).
            pp, disabled = max_pp, False
        if move_id.startswith("hiddenpower"):
            move_id = hidden_power_engine_id(move_id, mon.ivs)
        specs.append(MoveSpec(id=move_id, pp=pp, disabled=disabled))
    while len(specs) < _MOVE_SLOT_LIMIT:
        specs.append(MoveSpec(id="none", pp=0, disabled=True))
    return tuple(specs)
