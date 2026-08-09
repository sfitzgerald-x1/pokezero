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

from dataclasses import dataclass, replace
from typing import Any, Mapping, Sequence

from .dex import ShowdownDex, normalize_id
from .env import BattleStartOverride
from .gen3_damage import gen3_hp_stat, gen3_stat
from .poke_engine_adapter import (
    BattleSpec,
    MoveSpec,
    PokemonSpec,
    SideSpec,
    build_poke_engine_state,
    require_charge_state_support,
    require_move_trap_support,
    require_rest_sleep_refund_support,
    require_rest_turns_support,
)
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
# ``flashfire`` needs neither: the parser sets it on the public ``-start``
# line and clears it on ``-end``/switch, and the engine models it as
# until-switch (1.5x own-fire boost; the immunity lives in the ability hook,
# so the volatile alone is boost-only — never wrong, at worst incomplete if a
# sampled world lacked the ability, which cannot happen for the mono-ability
# Gen 3 randbats carriers nor for the request-known self side).
# ``attract`` needs no duration either (Gen 3 infatuation runs until the holder
# switches or the source leaves — no countdown): the parser sets it on the
# public ``-start``/``-activate move: Attract`` line and clears it on
# ``-end``/switch. The patched Gen 3 engine prices the 50%-per-turn move
# immobilization as a chance branch and, in singles, clears the relationship
# when either active switches.
# Volatiles the sampled world can express EXACTLY from public information.
# ``destinybond`` qualifies on the vendored gen3 engine: it is a presence-only
# volatile (gen3/choice_effects.rs clears it on any non-Destiny-Bond move, per
# the gens 2-6 rule) that triggers on KO, so seeding it from the public
# ``|-singlemove|` line reproduces the engine's state with no hidden component.
# ``perish1``-``perish4`` qualify for the same reason from the other direction:
# the counter is the ONLY state Perish Song carries, Showdown publishes it on
# every ``|-start|<mon>|perishN`` line, and the vendored engine runs the
# countdown itself (gen3/generate_instructions.rs decrements PERISH3 -> PERISH2
# -> PERISH1 and faints at zero). Seeding the current count reproduces the
# engine's state exactly, so a Perish-Song endgame is searched rather than
# guessed — precisely the position where search matters most.
# ``taunt`` qualifies too, and it is the only entry here needing a separate
# duration field seeded (done below, next to Yawn's). It qualifies because gen 3's
# clock is FIXED, so the remaining count is not hidden information:
# `data/mods/gen3/moves.ts` pins the condition at `duration: 2` with
# `durationCallback: undefined`, and the `onStart` that bumps duration by one
# against an already-moved target belongs to modern Showdown -- gen4 overrides
# `onStart` with a plain one and gen3 inherits gen4 (`data/mods/gen3/scripts.ts`:
# `inherit: 'gen4'`). So there is no roll to guess and no dependence on which
# side moved first. Measured on the local simulator both ways round rather than
# read off: with a faster Taunt user and with a slower one, EXACTLY ONE
# subsequent request carries the volatile.
#
# ⚠ THAT ARGUMENT ALONE DOES NOT SEPARATE IT FROM YAWN, which is gated on
# `approximate_hidden_duration_volatiles` below and whose gen3 clock is ALSO a
# fixed `duration: 2` with no `durationCallback` and a plain `onStart`. So "the
# clock is fixed, therefore nothing is hidden" is true of both and is NOT a
# discriminator. An earlier revision of this branch claimed a structural one and
# it was false; see the withdrawal note at the seeding site.
#
# The honest statement is narrower, and it is what the entry rests on: the count
# is exact at an ORDINARY boundary, which is measured, and at the one boundary
# where it is not -- a mid-turn replacement, where the answer depends on the
# Taunt's age and the payload does not carry it -- `taunt` is REMOVED from this
# set again and the world fails closed. Exact where admitted, refused where not.
# Yawn takes the other option at the same seam (approximate rather than refuse);
# that is a policy difference, it is not re-measured here, and nothing about Yawn
# is changed by this.
_SUPPORTED_VOLATILES = frozenset({
    "leechseed", "flashfire", "attract", "destinybond", "taunt",
    "perish1", "perish2", "perish3", "perish4",
})

# Mid-charge state of a two-turn move, keyed by the MOVE id, matching the parser
# (showdown._CHARGE_MOVE_VOLATILES) and the engine's own charge volatile. Behind a
# capability guard rather than in the set above: an engine that does not know the
# volatile ACCEPTS the token and drops it (`from_str` defaults to NONE), which builds
# the charging Pokemon FREE instead of declining the world.
_CHARGE_VOLATILES = frozenset({"solarbeam"})

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
# gen3 Rest sets the engine's counter to 3 (gen3/choice_effects.rs); it is decremented
# once per move attempt and the mon wakes at 1.
_REST_SLEEP_TURNS = 3

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


# Showdown request flags that withhold hidden information rather than restrict the
# action set. Each belief world resolves them by committing to a concrete opponent
# hypothesis.
#
# ``maybeTrapped`` is the one that actually fires here, and it covers more than
# "the foe MIGHT have a trapping ability": an ability trap sets trapped='hidden'
# (sim/pokemon.ts), and getMoveRequestData only reports ``trapped`` when the value
# is exactly True — so a REVEALED, actively-trapping Arena Trap or Shadow Tag also
# arrives as maybeTrapped. That is why it dominated the fallback count.
#
# ``maybeDisabled``/``maybeLocked`` are listed for completeness but are dead in this
# format: Imprison is their only producer (maybeLocked is derived from maybeDisabled)
# and no gen3 randbats set carries Imprison.
_HIDDEN_INFORMATION_REQUEST_FLAGS = frozenset({"maybeTrapped", "maybeDisabled", "maybeLocked"})


def _undischarged_materialization_blockers(
    blockers: Any,
    *,
    removed_item_species: frozenset[str],
    item_overrides: Mapping[str, str],
) -> tuple[str, ...]:
    """Return the payload blockers the caller has NOT positively expressed.

    ``materializationBlockers`` is written by the payload producer, which has no
    view of the item signals a caller derives independently from the same belief
    engine. Two producers therefore describe one fact: a publicly itemless mon
    arrives here both as an ``item-state-removed:<species>`` blocker AND (from a
    caller that resolves it) in ``removed_item_species``, whose whole purpose is
    to clear the sampled set's item. Vetoing on the blocker discarded a world the
    caller had already made exact — the single largest fallback source in the
    2026-07-26 depth study (2811 world failures).

    A blocker is discharged ONLY by the matching positive signal for the SAME
    species: a removal by ``removed_item_species``, a mutation by a confirmed
    ``item_overrides`` entry. Everything else — ambiguous rows, mutations with no
    protocol-confirmed item, Baton-Pass volatiles, unknown Leech Seed source —
    stays fail-closed, because nothing in the caller's signals expresses it and
    guessing would search a mechanically false world.
    """

    if not blockers:
        return ()
    undischarged: list[str] = []
    for raw in blockers:
        token = str(raw)
        kind, _, species = token.partition(":")
        species_id = normalize_id(species)
        if kind == "item-state-removed" and species_id in removed_item_species:
            continue
        if kind == "item-state-unconfirmed" and species_id in item_overrides:
            continue
        undischarged.append(token)
    return tuple(undischarged)


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
    approximate_partial_trap_turns: bool = False,
    approximate_hidden_duration_volatiles: bool = False,
    blocked_slots: Mapping[str, str] | None = None,
    encored_moves: Mapping[str, str] | None = None,
    removed_item_species: Mapping[str, Sequence[str]] | None = None,
    current_item_overrides: Mapping[str, Mapping[str, str]] | None = None,
    recharging_slots: Sequence[str] = (),
    truant_slots: Sequence[str] = (),
    transformed_slots: Mapping[str, str] | None = None,
    rng: Any | None = None,
) -> EngineWorld:
    """Pure construction: public materialization payload + sampled teams -> spec.

    ``payload`` must be the dict produced by
    ``local_showdown._public_materialization_payload`` (or a test literal of
    the same shape); ``override`` supplies both sides' belief-sampled packed
    teams. ``removed_item_species`` names, per slot, normalized species whose
    held item is publicly gone (Knock Off / an item-taking Trick / a public
    consumption) — the built world clears the sampled set's item for them,
    because the current public item state is exactly "no item" while the
    sampled item only reflects the set's battle-start assignment.
    ``current_item_overrides`` maps, per slot, species -> the CURRENT item the
    protocol positively revealed on them (a Trick swap's ``-item`` line) — the
    built world substitutes it for the sampled assignment's item (spread,
    moves and ability keep the sampled assignment's values: Trick moves only
    the item). A species named by BOTH signals is contradictory belief state
    and fails closed. Raises :class:`EngineWorldUnsupported` whenever the
    position holds public state this construction cannot express exactly.

    ``substituteHealthState`` is the replay's public Substitute provenance:
    ``"full"`` and an exact ``"exact"`` + ``substituteDepletion`` pair can
    build an active Substitute. Exact depletion is subtracted from this
    sampled world's initial Substitute HP; it is not replay-scale remaining HP.
    ``"unknown"`` is the sole public-information limit; missing or invalid
    active provenance is an instrumentation contradiction.
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
        # Only flags that BIND legality may fail construction. Showdown's
        # ``maybe*`` flags are the opposite of a constraint: they are the sim
        # DECLINING to tell us whether we are trapped or disabled, because
        # answering would leak the opponent's hidden ability or move set
        # (pokemon.ts getMoveRequestData). ``maybeTrapped`` in particular fires
        # whenever the foe's unrevealed ability could be Arena Trap, Magnet
        # Pull, or Shadow Tag — all in the gen3 randbats pool, which made this
        # the second-largest fallback source in the 2026-07-26 depth study.
        #
        # Refusing to search on a hidden-information marker is backwards for a
        # belief searcher: each sampled world commits to a concrete opponent
        # set, so the engine derives that world's trapped/disabled truth exactly
        # and consistently. Sampling the hypothesis IS the designed resolution,
        # and it strictly dominates falling back to a uniform-legal guess.
        #
        # ``trapped`` stays fail-closed: it is a hard, already-disclosed "you
        # cannot switch", and a world that let us switch anyway would search
        # illegal actions.
        binding = [
            flag
            for flag in raised
            if flag not in _HIDDEN_INFORMATION_REQUEST_FLAGS and flag != "trapped"
        ]
        if binding:
            raise EngineWorldUnsupported(
                "self_request_state_unsupported",
                f"self active request flags {binding} constrain legality beyond this construction",
            )
    self_trapped = bool(isinstance(request_state, Mapping) and request_state.get("trapped"))

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
    # Encore locks that could not be expressed as a slot index yet, because the
    # active's moveset is about to be replaced by `_apply_transform`.
    pending_encore_locks: dict[str, str] = {}
    self_active_request_moves = payload.get("selfActiveMoves")
    if not isinstance(self_active_request_moves, Sequence) or isinstance(
        self_active_request_moves, str
    ):
        self_active_request_moves = None
    for slot in _PLAYER_SLOTS:
        side_payload = sides_payload.get(slot)
        if not isinstance(side_payload, Mapping):
            raise EngineWorldUnsupported("payload_malformed", f"side {slot!r} is missing")
        packed = override.player_teams.get(slot)
        if not packed:
            raise EngineWorldUnsupported("override_side_missing", f"override has no packed team for {slot!r}")
        team = unpack_team(packed)
        is_self_slot = slot == self_player
        built_sides[slot], species_order, pending_encore = _build_side_spec(
            slot=slot,
            side_payload=side_payload,
            team=team,
            dex=dex,
            is_self=is_self_slot,
            turn=turn,
            self_benched_move_history=bool(payload.get("selfBenchedMoveHistory")),
            approximate_sleep_turns=approximate_sleep_turns,
            approximate_substitute_health=approximate_substitute_health,
            approximate_partial_trap_turns=approximate_partial_trap_turns,
            approximate_hidden_duration_volatiles=approximate_hidden_duration_volatiles,
            force_switch=is_self_slot and self_force_switch,
            world_owes_replacement=self_force_switch,
            baton_passing=is_self_slot and self_baton_passing,
            opponent_committed_pending=(not is_self_slot) and self_baton_passing,
            wish_set_turn=_wish_set_turn(payload, slot),
            encored_move=(encored_moves or {}).get(slot),
            removed_item_species=frozenset(
                normalize_id(str(species))
                for species in (removed_item_species or {}).get(slot, ())
            ),
            current_item_overrides={
                normalize_id(str(species)): normalize_id(str(item))
                for species, item in ((current_item_overrides or {}).get(slot) or {}).items()
            },
            must_recharge=slot in (recharging_slots or ()),
            truant_loafs=slot in (truant_slots or ()),
            transformed_active=slot in (transformed_slots or {}),
            self_active_request_moves=(
                self_active_request_moves if is_self_slot else None
            ),
            rng=rng,
        )
        party_species[slot] = species_order
        if pending_encore is not None:
            pending_encore_locks[slot] = pending_encore

    self_order = payload.get("selfTeamOrder")
    if isinstance(self_order, Sequence) and not isinstance(self_order, str):
        order_ids = {normalize_id(str(species)) for species in self_order}
        if order_ids and order_ids != set(party_species[self_player]):
            raise EngineWorldUnsupported(
                "self_world_mismatch",
                f"request team {sorted(order_ids)} != sampled world {sorted(party_species[self_player])}",
            )

    if transformed_slots:
        built_sides = _apply_transform(
            built_sides,
            transformed_slots,
            self_player=self_player,
            self_active_request_moves=self_active_request_moves,
            self_request_struggle_only=_self_request_is_struggle_only(
                sides_payload.get(self_player),
                self_active_request_moves,
            ),
        )

    # AFTER the copy, never before: the slot index an Encore lock resolves to is
    # only meaningful against the moveset the engine will actually read.
    if pending_encore_locks:
        built_sides = _apply_encore_locks(built_sides, pending_encore_locks)

    if self_trapped:
        _require_world_reproduces_trap(built_sides, dex=dex, self_player=self_player)

    weather, weather_turns = _weather_fields(payload)
    built_sides = _apply_forecast_types(built_sides, weather=weather)
    # LAST of the three retype arms, deliberately. Precedence, from sim source:
    #
    #   * `_apply_transform` (above) mirrors Transform, which copies the donor's types.
    #   * `_apply_forecast_types` (above) DERIVES Castform's type from public weather,
    #     mirroring Forecast's `onUpdate`.
    #   * this arm applies an OBSERVED `|-start|...|typechange|<type>|` -- Color Change's
    #     `onAfterMoveSecondary` calling `setType(type)` (data/abilities.ts:554-562).
    #
    # Observation beats derivation, so it goes last: the other two reconstruct what the
    # types SHOULD be from a rule, while this one is the sim telling us what they ARE. If a
    # transformed mon is then Color Change'd, Showdown's later `setType` wins because both
    # mutate `pokemon.types` in event order; applying the observation last reproduces that
    # without having to model the ordering.
    #
    # Only the `type:` form is consumed. `forme:` (Castform Forecast) is deliberately left
    # to `_apply_forecast_types`: that arm already derives the same answer from the same
    # public weather, and Forecast is `onUpdate`, so it cannot lag the observation. Handling
    # it here too would give Castform two writers that must agree -- the shape that made the
    # encoder-vocabulary bug survive for months.
    built_sides = _apply_live_typechange(built_sides, payload)
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


# Showdown gives every copied slot `pp = Math.min(5, move.pp)` where `move` is the
# DEX entry (sim/pokemon.ts transformInto), i.e. the move's BASE PP — not whatever
# the donor has left. No gen3 move has a base PP below 5, so a copied slot is
# always exactly 5, even off a donor that has drained the move to 1. This used to
# read `min(5, donor_remaining_pp)`, which under-filled the copy and disagreed with
# what the engine itself writes when Transform is CLICKED
# (third_party/poke-engine-gen3-transform.patch); the two now use one rule.
_TRANSFORM_MOVE_PP = 5


def _self_request_is_struggle_only(
    side_payload: Any,
    self_active_request_moves: Sequence[Mapping[str, Any]] | None,
) -> bool:
    """Whether the acting seat's request is Showdown's Struggle branch.

    ``local_showdown._request_reports_only_struggle`` is the primary reader of
    that branch, but it reads the RAW request, which does not reach here. Both
    halves of its verdict do, and this is the payload-side mirror:

      * ``selfActiveMoves`` is empty. ``_request_active_moves`` keeps only rows
        carrying int ``pp``/``maxpp``, and the substituted Struggle row carries
        neither, so the branch always publishes ``[]``.
      * EVERY move on the acting side's active row is ``disabled``. That is the
        marking ``_apply_struggle_only_move_state`` writes at exactly this
        branch and nowhere else, and it is what separates Struggle from its
        pp-less siblings -- ``mustrecharge`` and the two-turn charge lock also
        publish ``selfActiveMoves: []``, because they too come off a request
        whose one move row has no ``pp``, but they leave the row's own
        ``disabled`` flags alone.

    The second clause cannot be satisfied by an ordinary request: the rows come
    from ``_request_active_moves``, and a request whose every slot was disabled
    would have had ``getMoves`` return ``[]`` and been replaced by the Struggle
    row before any pp-bearing row could be retained. So an all-disabled
    non-empty row list only ever comes from the marking.

    ``None`` rather than ``[]`` for ``selfActiveMoves`` means the payload has no
    such key at all (a bare test literal), which is not evidence of anything.
    """

    if self_active_request_moves is None or self_active_request_moves:
        return False
    if not isinstance(side_payload, Mapping):
        return False
    rows = _active_row_moves(side_payload)
    return bool(rows) and all(bool(row.get("disabled")) for row in rows)


def _request_move_state_by_id(
    self_active_request_moves: Sequence[Mapping[str, Any]] | None,
) -> dict[str, tuple[int, bool]]:
    """``normalized move id -> (pp, disabled)`` from the acting seat's request."""

    state: dict[str, tuple[int, bool]] = {}
    for row in self_active_request_moves or ():
        if not isinstance(row, Mapping) or not isinstance(row.get("id"), str):
            continue
        pp = row.get("pp")
        if not isinstance(pp, int):
            continue
        state[normalize_id(row["id"])] = (pp, bool(row.get("disabled")))
    return state


def _apply_transform(
    sides: Mapping[str, SideSpec],
    transformed_slots: Mapping[str, str],
    *,
    self_player: str | None = None,
    self_active_request_moves: Sequence[Mapping[str, Any]] | None = None,
    self_request_struggle_only: bool = False,
) -> dict[str, SideSpec]:
    """Re-express a publicly Transformed active as the mon it copied.

    The vendored gen3 engine has no TRANSFORM volatile at all, so this used to
    fail closed on every Ditto — the largest remaining fallback source. But
    Transform needs no volatile to express: gen3 copies species, types, the five
    non-HP stats, the ability and the moveset (at 5 PP), and leaves HP alone
    (sim/pokemon.ts transformInto). Every one of those is already a field on
    PokemonSpec, so the copy can simply be BAKED into the active's spec.

    The donor is read out of the SAME sampled world rather than the dex: in
    singles Transform targets the opposing active, so the world being built
    already contains the exact mon that was copied, with that world's own
    belief-sampled spread. Reading it here keeps the two sides consistent — a
    dex lookup would invent a spread the opponent's own party contradicts.

    The copied moveset's IDENTITY comes from that sampled donor, but its
    per-slot USABILITY does not: on our own seat the request reports the copy's
    live PP and disable state, and ``_copied_move_spec`` overlays it. Read that
    function for why re-seeding it was wrong.

    Two things are deliberately NOT copied, because gen3 does not copy them:
    HP/maxhp (the transformer keeps its own, which is why a transformed Ditto
    stays frail), and the stat stages — Showdown already reports the
    transformer's own boosts post-copy, so the payload's boost block is correct
    as-is and re-applying the donor's would double-count.

    Reversion IS reproduced now. The engine gained a real TRANSFORMED volatile
    and a `pre_transform` record of the transformer's own base form
    (third_party/poke-engine-gen3-transform.patch), so a constructed Transform
    can end the way a clicked one does. Expressing that takes three things
    beyond the copy itself, and getting any of them wrong is worse than not
    reverting at all:

      * `pre_transform` — the pre-copy spec, which the engine reads back to
        restore species, the five stats and the moveset.
      * BASE IDENTITY — `base_ability` and `base_types` stay the TRANSFORMER's
        own while `ability`/`types` become the donor's. This is what the engine
        restores (`ability_on_switch_out`, and the TYPECHANGE switch-out arm),
        and it is the half that used to be silently wrong: PokemonSpec plumbed
        neither field, so the binding defaulted `base_ability` to the ability
        this function had just copied FROM THE DONOR. A bridge-built transformed
        Ditto therefore reverted into a Ditto with the donor's Immunity, and its
        types into a flat Normal.
      * The TRANSFORMED and TYPECHANGE volatiles, which are what make the engine
        run those two restores at all.
    """

    request_move_state = _request_move_state_by_id(self_active_request_moves)
    updated = dict(sides)
    for slot, target_species in transformed_slots.items():
        is_self = self_player is not None and slot == self_player
        side = updated.get(slot)
        donor_side = updated.get("p2" if slot == "p1" else "p1")
        if side is None or donor_side is None or not side.pokemon:
            raise EngineWorldUnsupported(
                "transform_unexpressible", f"side {slot!r} has no built side to transform"
            )
        target_id = _engine_species_id(normalize_id(str(target_species)))
        donor = next(
            (mon for mon in donor_side.pokemon if _engine_species_id(normalize_id(mon.id)) == target_id),
            None,
        )
        if donor is None:
            # The copied mon is not in this world's opposing party, so its
            # stats/moves would have to be invented. Fail closed instead.
            raise EngineWorldUnsupported(
                "transform_unexpressible",
                f"side {slot!r} copied {target_species!r}, absent from the sampled opposing party",
            )
        active = side.pokemon[side.active_index]
        copied = replace(
            active,
            id=donor.id,
            types=donor.types,
            # The transformer's own identity, kept for the engine to restore.
            # `active` is the PRE-copy spec, so its ability/types are the real
            # base ones; an explicit base already on the spec wins (nothing sets
            # one today, but silently overwriting it would be a trap).
            base_ability=(
                active.base_ability if active.base_ability is not None else active.ability
            ),
            base_types=(
                active.base_types if active.base_types is not None else tuple(active.types)
            ),
            pre_transform=active,
            attack=donor.attack,
            defense=donor.defense,
            special_attack=donor.special_attack,
            special_defense=donor.special_defense,
            speed=donor.speed,
            ability=donor.ability,
            weight_kg=donor.weight_kg,
            moves=tuple(
                _copied_move_spec(
                    move,
                    request_move_state=request_move_state if is_self else {},
                    struggle_only=is_self and self_request_struggle_only,
                )
                for move in donor.moves
            ),
        )
        party = list(side.pokemon)
        party[side.active_index] = copied
        # TRANSFORMED drives the species/stats/moveset restore off `pre_transform`;
        # TYPECHANGE is the existing arm that restores `types -> base_types`. Both
        # are dropped by the same switch-out that consumes them.
        volatiles = list(side.volatile_statuses)
        for volatile in ("transformed", "typechange"):
            if volatile not in volatiles:
                volatiles.append(volatile)
        updated[slot] = replace(
            side, pokemon=tuple(party), volatile_statuses=tuple(volatiles)
        )
    return updated


def _copied_move_spec(
    move: Any,
    *,
    request_move_state: Mapping[str, tuple[int, bool]],
    struggle_only: bool,
) -> Any:
    """One slot of the copied moveset, with its LIVE usability.

    THE DEFECT THIS CLOSES. ``_TRANSFORM_MOVE_PP`` is what Showdown writes at
    the instant Transform resolves, and re-writing it on every world build says
    the copy is always as fresh as it was on turn one. It is not: the
    transformer spends that PP like any other, and Showdown reports the spend to
    us on our own seat, in ``selfActiveMoves`` -- the RAW request, which for a
    transformed active lists the COPIED moves (the same source
    ``_build_side_spec`` already reads Encore's disable pattern off, and NOT the
    active ROW, which stays the pre-Transform snapshot on purpose so the
    transformer's own PP survives reversion). Every world therefore believed a
    Ditto that had drained Surf to 0 still had five of them.

    Measured on ``fb3h1-960111`` p2, a Ditto transformed into Suicune: at round
    63 the request read ``surf 0/24 disabled, substitute 1/10, icebeam 0/10
    disabled, calmmind 0/20 disabled`` while all eight worlds built
    ``surf 5, rest 5, icebeam 5, calmmind 5``, none disabled. One round later
    the real Ditto ran out entirely, Showdown substituted the Struggle
    pseudo-move, and the engine -- still holding four full moves -- proposed
    four of them against a request offering only ``struggle``. Nothing mapped:
    ``choices_unmapped`` / ``all_unmapped_legality_mismatch``, on all 40
    remaining rounds of the battle.

    ``struggle_only`` is that last state, where the request publishes no move
    rows at all to overlay and the only fact available is the one Showdown
    already computed for every slot: none of them is usable. It is applied to
    EVERY copied slot, including slots the sampled donor has and the real donor
    does not, because the verdict is about the transformer, not the moveset.
    PP is left pinned there rather than zeroed, exactly as
    ``_apply_struggle_only_move_state`` leaves it: the Struggle request carries
    no PP, and ``Pokemon::add_available_moves`` already refuses a disabled slot.

    A slot the request does not name keeps ``_TRANSFORM_MOVE_PP``, and the reason
    is NOT that the request has nothing to say about it. It says the opposite:
    on our own seat the request enumerates the copy's ENTIRE moveset, so a slot
    the sampled donor carries and the request does not name is PROVABLY not in
    the real copy (``rest`` above, in five of the eight worlds -- the donor comes
    from the belief-SAMPLED opposing party, which can carry a move the real
    Suicune never had). Disabling it would remove a fiction, not invent a fact.

    It is left usable so that the fiction stays COUNTABLE. An unusable slot is
    silently absent from the engine's options; a usable one gets proposed, misses
    the request, and lands in ``EngineSearchStats.unmapped_choices`` -- 16 on the
    exemplar battle and 10 on the 60-game batch, which is the belief sampler's
    moveset error made visible on a counter rather than swallowed by the world
    builder. This class was found by that counter, and the burndown's stop
    condition is read off it.

    Note the asymmetry with the non-transformed self path, which does the other
    thing: it refuses outright rather than invent PP for a slot with no
    request-known snapshot (``self_pp_unknown``, :2490). The difference is who
    owns the error. There the missing PP is OUR OWN Pokemon's and a guess would
    be silently wrong; here the extra slot belongs to a sampled OPPONENT variant
    that this world invented, and refusing every Transform world whose sampled
    donor over-covers the real one would trade a counted miss for a refusal.
    """

    move_id = normalize_id(move.id)
    # Request rows report Hidden Power as plain `hiddenpower`; the spec id is the
    # engine's typed+BP form. Same tolerance `_move_specs` applies on the
    # non-transformed path, and gen3 has at most one Hidden Power slot.
    keys = (move_id, "hiddenpower") if move_id.startswith("hiddenpower") else (move_id,)
    known = next(
        (request_move_state[key] for key in keys if key in request_move_state), None
    )
    # The two inputs are mutually exclusive as things stand -- the struggle-only
    # branch is admitted only on an EMPTY `selfActiveMoves`, so there is nothing
    # to overlay there -- and the `or` is how they compose rather than a claim
    # that both can be true at once.
    pp, disabled = known if known is not None else (_TRANSFORM_MOVE_PP, False)
    return replace(move, pp=pp, disabled=disabled or struggle_only)


def _require_world_reproduces_trap(
    sides: Mapping[str, SideSpec],
    *,
    dex: ShowdownDex,
    self_player: str,
) -> None:
    """Fail closed unless the BUILT world independently traps our active.

    ``trapped`` is a disclosed hard constraint: Showdown is telling us we cannot
    switch. Refusing to search on it was over-strict, because the sampled world
    usually reproduces the trap on its own — the flag is disclosed precisely
    when its cause is public, and the belief filter carries a revealed trapping
    ability into every sample. But it is not ALWAYS reproduced: the vendored
    gen3 engine models no Mean Look / Block / Spider Web, so a move-trapped mon
    would be free to switch in search and the tree would explore illegal lines.

    So this verifies instead of assuming, by transcribing the engine's own
    ``Side::trapped`` conditions (gen3/state.rs): a partial trap or locked move
    on us, Shadow Tag on the foe, Arena Trap against a grounded target, or
    Magnet Pull against a Steel type. If none holds, the trap has a cause the
    world cannot express and construction fails closed as before.
    """

    foe_player = "p2" if self_player == "p1" else "p1"
    self_side, foe_side = sides.get(self_player), sides.get(foe_player)
    if self_side is None or foe_side is None:
        raise EngineWorldUnsupported("payload_malformed", "trap check is missing a built side")

    self_volatiles = {normalize_id(str(v)) for v in self_side.volatile_statuses}
    # ``mustrecharge`` is not one of Side::trapped's conditions, but it reaches the
    # same place: the engine's option builder skips the switch branch entirely
    # while it is set, so the world does refuse to switch. Showdown reports the
    # hard lock as ``trapped``, so without this the recharge turn after every
    # Hyper Beam fell back needlessly.
    # ``trapped`` is Showdown's move-trap (Mean Look / Spider Web / Block), which
    # the engine now models as its own volatile and honours in Side::trapped, so
    # a world carrying it does refuse to switch. It reaches this set only through
    # the allowlist below, which fails closed on a wheel that predates
    # third_party/poke-engine-gen3-move-trapping.patch.
    # A mid-charge two-turn move is the same shape of hard lock: the engine's
    # `active_is_charging_move` restricts get_all_options to that one move, so the
    # side has no switch to analyse and the extra trap reasoning below is moot.
    # Consistent with `lockedmove` beside it -- both are commitments the engine
    # already enforces.
    if self_volatiles & (
        {"trapped", "partiallytrapped", "lockedmove", "mustrecharge"} | _CHARGE_VOLATILES
    ):
        return

    active = self_side.pokemon[self_side.active_index] if self_side.pokemon else None
    foe_active = foe_side.pokemon[foe_side.active_index] if foe_side.pokemon else None
    if active is None or foe_active is None:
        raise EngineWorldUnsupported("payload_malformed", "trap check is missing an active mon")

    foe_ability = normalize_id(str(foe_active.ability or ""))
    types = {str(t).lower() for t in (active.types or ())}
    if foe_ability == "shadowtag":
        return
    if foe_ability == "magnetpull" and "steel" in types:
        return
    if foe_ability == "arenatrap":
        # The engine's grounded test: Flying types and Levitate are exempt.
        ability = normalize_id(str(active.ability or ""))
        if "flying" not in types and ability != "levitate":
            return

    raise EngineWorldUnsupported(
        "self_request_state_unsupported",
        "self active request flags ['trapped'] constrain legality beyond this construction "
        f"(sampled world does not trap: foe ability {foe_ability!r})",
    )


def _truant_volatile_decision(
    side_payload: Mapping[str, Any], truant_loafs: bool
) -> bool:
    """Whether this side's active mon carries the TRUANT (loafing) volatile.

    The PAYLOAD wins whenever it is a bool. The parser tracks the sim's own free-running
    toggle; `truant_loafs` is the caller-side "acted last round -> loafs now" proxy that this
    replaces, so a payload `False` must OVERRIDE a proxy `True` rather than OR with it.

    `None` means no phase assertion: no holder, a truncated prefix, or a full-prefix Trace
    acquisition whose residual event-queue membership is not public-derivable. It falls back
    to the legacy proxy; this preserves previous behaviour but is not a fail-closed
    materialization block.
    """
    phase = side_payload.get("truantPhase")
    if isinstance(phase, bool):
        return phase
    return bool(truant_loafs)


def _apply_live_typechange(
    sides: Mapping[str, SideSpec], payload: Mapping[str, Any]
) -> dict[str, SideSpec]:
    """Stamp an observed Color Change retype onto the active mon.

    Mono-type by construction: Showdown's Color Change calls `setType(type)`, which
    REPLACES the type list rather than appending, so a retyped Kecleon is single-typed.

    Reverts on switch-out are already handled upstream -- the parser clears
    `live_type_override` in its switch block -- so an empty value here means "base types",
    not "unknown", and this is a no-op.
    """
    result = dict(sides)
    for slot, side in tuple(result.items()):
        raw = ((payload.get("sides") or {}).get(slot) or {}).get("liveTypeOverride")
        if not raw or not str(raw).startswith("type:"):
            continue
        live_type = str(raw).split(":", 1)[1].strip()
        if not live_type:
            continue
        mon = side.pokemon[side.active_index]
        party = list(side.pokemon)
        party[side.active_index] = replace(mon, types=(live_type,))
        result[slot] = replace(side, pokemon=tuple(party))
    return result


def _apply_forecast_types(sides: Mapping[str, SideSpec], *, weather: str) -> dict[str, SideSpec]:
    """Latch Castform's current type into the engine root state.

    Poke-engine updates Forecast after weather changes inside a searched line,
    but the initial world is reconstructed from base Pokédex types. Apply the
    current public weather here so root and leaf evaluations agree.
    """

    result = dict(sides)
    active = tuple(side.pokemon[side.active_index] for side in result.values())
    weather_suppressed = any(
        mon.hp > 0 and mon.ability in {"airlock", "cloudnine"} for mon in active
    )
    forecast_type = {
        "rain": "Water",
        "sun": "Fire",
        "hail": "Ice",
    }.get(weather, "Normal")
    if weather_suppressed:
        forecast_type = "Normal"

    for slot, side in tuple(result.items()):
        mon = side.pokemon[side.active_index]
        if mon.id != "castform" or mon.ability != "forecast":
            continue
        party = list(side.pokemon)
        party[side.active_index] = replace(mon, types=(forecast_type,))
        result[slot] = replace(side, pokemon=tuple(party))
    return result


def world_battle_spec(
    state: Any,
    override: BattleStartOverride,
    *,
    dex: ShowdownDex,
    approximate_sleep_turns: bool = False,
    approximate_substitute_health: bool = False,
    approximate_partial_trap_turns: bool = False,
    approximate_hidden_duration_volatiles: bool = False,
    blocked_slots: Mapping[str, str] | None = None,
    encored_moves: Mapping[str, str] | None = None,
    removed_item_species: Mapping[str, Sequence[str]] | None = None,
    current_item_overrides: Mapping[str, Mapping[str, str]] | None = None,
    recharging_slots: Sequence[str] = (),
    truant_slots: Sequence[str] = (),
    transformed_slots: Mapping[str, str] | None = None,
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
        approximate_partial_trap_turns=approximate_partial_trap_turns,
        approximate_hidden_duration_volatiles=approximate_hidden_duration_volatiles,
        blocked_slots=blocked_slots,
        encored_moves=encored_moves,
        removed_item_species=removed_item_species,
        current_item_overrides=current_item_overrides,
        recharging_slots=recharging_slots,
        truant_slots=truant_slots,
        transformed_slots=transformed_slots,
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
    approximate_partial_trap_turns: bool = False,
    approximate_hidden_duration_volatiles: bool = False,
    force_switch: bool = False,
    # WORLD-level, unlike `force_switch`, which is per-side (`is_self_slot and
    # self_force_switch`). The deferred residual block runs on the replacement ply
    # for BOTH sides, so a volatile whose clock that block advances is ambiguous on
    # either seat -- the opponent's Taunt is the measured case, and the opponent's
    # own `force_switch` is always False.
    world_owes_replacement: bool = False,
    wish_set_turn: int | None = None,
    encored_move: str | None = None,
    removed_item_species: frozenset[str] = frozenset(),
    current_item_overrides: Mapping[str, str] | None = None,
    must_recharge: bool = False,
    truant_loafs: bool = False,
    baton_passing: bool = False,
    opponent_committed_pending: bool = False,
    transformed_active: bool = False,
    self_active_request_moves: Sequence[Mapping[str, Any]] | None = None,
    rng: Any | None = None,
) -> tuple[SideSpec, tuple[str, ...], str | None]:
    """Build one side. The third return value is a DEFERRED Encore lock.

    A transformed active's moveset is replaced by ``_apply_transform`` after
    this function returns, so its Encore lock cannot be expressed as a slot
    index yet. The move ID is handed back instead, for ``_apply_encore_locks``
    to bind against the final moveset. ``None`` for every other side.
    """

    item_overrides = dict(current_item_overrides or {})
    blockers = _undischarged_materialization_blockers(
        side_payload.get("materializationBlockers"),
        removed_item_species=removed_item_species,
        item_overrides=item_overrides,
    )
    if blockers:
        raise EngineWorldUnsupported(
            "materialization_blocker",
            f"{slot}: {', '.join(blockers)}",
        )

    conflicted = sorted(removed_item_species & set(item_overrides))
    if conflicted:
        # One signal says "publicly holds nothing", the other "publicly holds
        # X": contradictory belief state for the same mon. Never guess which
        # is right — a wrong item in a searched world is silent wrongness.
        raise EngineWorldUnsupported(
            "item_state_conflict",
            f"side {slot!r}: {conflicted} carry both a removal and a current-item override",
        )

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
            item_removed=species_id in removed_item_species,
            item_override=item_overrides.get(species_id),
            is_transformed_active=bool(transformed_active and row is not None and bool(row.get("active"))),
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
    if side_payload.get("meanlookTrap") is True and "trapped" not in volatiles:
        # Showdown's move-trap arrives on its OWN payload key, not in `volatiles`: the parser
        # tracks `|-activate|SLOT|trapped` in `meanlook_trap`, separate from the
        # TRACKED_VOLATILES-gated bag, because that bag is also the observation encoder's
        # vocabulary and the Node bridge's materialization allowlist. See the producer note in
        # `local_showdown._public_materialization_payload`.
        #
        # Joined HERE, before the allowlist below, so the move trap takes exactly the path a
        # payload-carried `trapped` already takes -- the same `require_move_trap_support()`
        # wheel gate, the same `_SUPPORTED_VOLATILES` admission, and the same
        # `_require_world_reproduces_trap` discharge. Nothing downstream learns there are two
        # producers.
        volatiles = volatiles + ["trapped"]
    supported = _SUPPORTED_VOLATILES | ({"substitute"} if approximate_substitute_health else set())
    if world_owes_replacement and "taunt" in supported:
        # TAUNT IS EXACT AT AN ORDINARY BOUNDARY AND AMBIGUOUS AT THIS ONE, so it
        # is withdrawn from the allow-list here rather than seeded with a guess.
        #
        # A replacement boundary is taken BEFORE the faint turn's residual has
        # run (gen <= 3 replaces after every move), and the engine RUNS that
        # deferred residual on the replacement ply: `end_of_turn_triggered`
        # returns true whenever either side's `force_switch` is set, which is the
        # flag `battle_spec_from_payload` sets from `selfRequestKind`. Measured
        # through the real path -- the replacement ply's instruction list carries
        # `Heal SideTwo` (Leftovers) and `RemoveVolatileStatus SideTwo: TAUNT`.
        #
        # So the seed has to say how many ticks are ALREADY elapsed, and at this
        # boundary that depends on the Taunt's AGE, which the payload does not
        # carry. Both ages are reachable and they disagree, measured live on the
        # simulator by
        # tests/test_struggle_only_move_state.py::TauntReplacementBoundaryAgeTests
        # -- which asserts the DISAGREEMENT itself, so if a future change ever
        # made the two ages behave alike that pin goes red and this refusal
        # should be revisited:
        #
        #   age 0 (Taunt landed on the faint turn)   -> 1 taunted move phase left
        #                                               => the engine needs seed 0
        #   age 1 (Taunt landed the turn before)     -> 0 taunted move phases left
        #                                               => the engine needs seed 1
        #
        # and at a `force_switch` world the engine gives 0 phases for seed 1 and
        # 1 phase for seed 0. There is therefore NO single seed that is right
        # here; picking either trades one silent error for the opposite one. The
        # age IS publicly derivable -- Showdown announces `-start ... move: Taunt`
        # on a known turn -- but the parser does not track it today, so deriving
        # it is follow-up work and not a guess to make here. Note the size of that
        # follow-up before assuming it is small: `TRACKED_VOLATILES` doubles as the
        # observation encoder's vocabulary and the bridge's materialization
        # allowlist, so an age field lands in the observation spec too.
        #
        # Withdrawing from `supported` rather than minting a new reason is
        # deliberate: the resulting `volatile_unsupported: side 'pN': ['taunt']`
        # is exactly what this boundary is, it keeps the fail-closed shape
        # identical to `origin/main`'s, and it adds no counter key to the census.
        supported = supported - {"taunt"}
    if approximate_partial_trap_turns:
        # Gen 3 Wrap/Bind/Clamp/Fire Spin/Whirlpool run 2-5 RANDOM turns, and
        # the public replay never sees the roll. The vendored engine models
        # PARTIALLYTRAPPED with no duration counter at all: it traps and deals
        # maxhp/16 per turn until the TRAPPER switches out
        # (gen3/generate_instructions.rs). So the volatile has no exact
        # expression, and this opt-in accepts the engine's own shape.
        #
        # The bias is ONE-SIDED IN FAVOUR OF THE TRAPPER, not simply
        # "pessimistic": this gate applies to whichever side holds the volatile.
        # When our mon is trapped the search under-rates escaping; when WE are
        # the trapper (Shuckle is the pool's only Wrap user) it over-rates the
        # lock. And the trap is UNBOUNDED, not merely long — the trapped mon has
        # no switch option at all (Side::trapped), so a deep line can grind it
        # down over far more turns than the real 2-5 roll allows.
        #
        # That is the reason it is a named approximation rather than an
        # allowlist entry: it belongs with approximate_sleep_turns and
        # approximate_substitute_health, visible in the config and attributable
        # in a run's provenance, not silently folded into "expressed exactly".
        supported = supported | {"partiallytrapped"}
    if approximate_hidden_duration_volatiles:
        # Two more public volatiles whose REMAINING duration is not public, and
        # whose engine model does not match the Gen 3 rule either:
        #   confusion — Gen 3 runs 2-5 random turns; the engine prices the 50%
        #     self-hit but never expires it inside a search, so a searched world
        #     is PESSIMISTIC about shaking it off.
        #   yawn — seeded at duration 1 above, which is EXACT at an ordinary
        #     move boundary: Showdown applies Yawn (duration 2) mid-turn and
        #     burns the first tick at that same turn's residual, and singles
        #     offers no request in between, so an observable Yawn always has one
        #     tick left. It rides here only for the residual case a mid-turn
        #     force-switch snapshot could produce, where the real count is 2 and
        #     the sleep would land a turn early.
        #
        # Both are still far better than the alternative they replace: without
        # them the whole decision falls back to a uniform-legal guess, which is
        # wrong about every move rather than about one effect's clock.
        supported = supported | {"confusion", "yawn"}
    if "trapped" in volatiles:
        # Showdown's move-trap (Mean Look / Spider Web / Block). Expressed
        # EXACTLY, not approximated: the gen3 trap carries no duration and no
        # residual — it lasts until the trapper leaves the field — so there is no
        # hidden clock to guess, unlike partiallytrapped above.
        #
        # Gated on the wheel actually carrying the patch. An unpatched binding
        # resolves the unknown volatile token to NONE and drops it silently,
        # which would be strictly worse than the fallback this replaces: instead
        # of declining the decision, search would hand the trapped seat its
        # switch options back and confidently plan an escape Showdown refuses.
        require_move_trap_support()
        supported = supported | {"trapped"}
    if volatiles_set := (set(volatiles) & _CHARGE_VOLATILES):
        # Mid-charge (Solar Beam): the commitment the public protocol announced with
        # `-prepare`. Without it the world is built with the charging mon free and the
        # engine starts a FRESH charge instead of releasing -- silently wrong rather
        # than declined, which is why this is expressed rather than approximated.
        require_charge_state_support()
        supported = supported | volatiles_set
    if "encore" in volatiles:
        supported = supported | {"encore"}
    if must_recharge:
        # Publicly-forced recharge turn (Hyper Beam landed last round): the
        # engine's MUSTRECHARGE volatile restricts the side to "No Move" —
        # without it, searched worlds hand the recharging mon a free action.
        volatiles = volatiles + ["mustrecharge"]
        supported = supported | {"mustrecharge"}
    # Truant phase. The PAYLOAD value wins when present: the parser tracks the sim's own
    # free-running toggle (`onResidual` flips `truantTurn` every turn unconditionally, seeded
    # at switch-in by `this.turn !== 0`), whereas the `truant_loafs` argument is a caller-side
    # "acted last round -> loafs now" proxy. The proxy agrees with the bit until the first
    # turn a holder is stopped from moving by something OTHER than Truant -- sleep, paralysis,
    # flinch, freeze, recharge, a switch -- after which it is inverted for the rest of the
    # stint. That single failure produced the 48-row loaf-phase family.
    #
    # `None` means no phase assertion (no holder, a truncated prefix, or a full-prefix Trace
    # acquisition whose residual event-queue membership is ambiguous). It falls back to the
    # caller's legacy proxy; this is intentionally compatible, not fail-closed.
    if _truant_volatile_decision(side_payload, truant_loafs):
        volatiles = volatiles + ["truant"]
        supported = supported | {"truant"}
    unsupported = sorted(set(volatiles) - supported)
    if unsupported:
        raise EngineWorldUnsupported("volatile_unsupported", f"side {slot!r}: {unsupported}")
    substitute_health = 0
    raw_substitute_health_state = side_payload.get("substituteHealthState")
    substitute_health_state = (
        raw_substitute_health_state
        if isinstance(raw_substitute_health_state, str)
        else None
    )
    raw_substitute_depletion = side_payload.get("substituteDepletion")
    if "substitute" in volatiles:
        # A freshly-created Substitute is public-exact at floor(maxhp / 4).
        # Only canonical ``unknown`` provenance is a named public-information
        # limit. Validate the state/value PAIR before construction or limit
        # accounting so a malformed companion cannot hide behind a valid name.
        initial_substitute_health = party[active_index].maxhp // 4
        if substitute_health_state in {"full", "unknown"} and not (
            raw_substitute_depletion is None
            or (
                not isinstance(raw_substitute_depletion, bool)
                and isinstance(raw_substitute_depletion, int)
                and raw_substitute_depletion == 0
            )
        ):
            raise EngineWorldUnsupported(
                "substitute_health_provenance_contradiction",
                f"side {slot!r} Substitute state {substitute_health_state!r} "
                f"cannot carry depletion {raw_substitute_depletion!r}",
            )
        if substitute_health_state == "full":
            substitute_health = initial_substitute_health
        elif substitute_health_state == "exact":
            if (
                isinstance(raw_substitute_depletion, bool)
                or not isinstance(raw_substitute_depletion, int)
                or raw_substitute_depletion <= 0
            ):
                raise EngineWorldUnsupported(
                    "substitute_health_provenance_contradiction",
                    f"side {slot!r} has invalid exact Substitute depletion "
                    f"{raw_substitute_depletion!r}",
                )
            substitute_health = initial_substitute_health - raw_substitute_depletion
            if substitute_health <= 0:
                raise EngineWorldUnsupported(
                    "substitute_depletion_world_incompatible",
                    f"side {slot!r} sampled max HP {party[active_index].maxhp} gives "
                    f"initial Substitute HP {initial_substitute_health}, which could not "
                    f"survive exact public depletion {raw_substitute_depletion}",
                )
        elif substitute_health_state == "unknown":
            # SAMPLE, do not refuse. This was 396 killed decisions in era 59 -- 48.6% of the
            # construction channel and its largest class -- and GOAL.md §0.2 names the reason
            # it should not be a refusal at all: "Hidden information is not a refusal
            # category. The belief machinery's entire design is to sample any consistent
            # hypothesis." It already does exactly that for unrevealed sets, items and
            # abilities; a Substitute's remaining HP is one more belief dimension.
            #
            # THE RANGE, stated without overselling it. `unknown` arises only from a
            # NON-BREAKING hit whose damage the public record does not reveal
            # (`_update_substitute_health_state`: gen 3's four fixed-damage moves give
            # `exact`, everything else gives `unknown`). Each such hit proves two things: it
            # removed at least 1 HP, and the Substitute SURVIVED. Adding the depletion already
            # PROVEN by fixed-damage hits, remaining health lies in
            # `[1, initial - min_depletion]`.
            #
            # ONE ACCUMULATOR, order-independent. Each hit is charged exactly once: its
            # proven damage when the public record resolves it, the minimum 1 HP when it does
            # not. Seismic Toss then Ice Beam and Ice Beam then Seismic Toss both give 78.
            # Two accumulators produced an ordering asymmetry that review measured twice.
            #
            # HOW WIDE THAT ACTUALLY IS. With a single UNRESOLVED hit and nothing proven the
            # range is `[1, initial - 1]` -- for the measured `initial = 162` case that is
            # 161/162, and it is nearer 98% at a small `initial` like 50, so the figure is
            # instance-specific rather than a constant. An earlier version called the range
            # narrow and said it "tightens as more hits land", which oversold a per-hit effect
            # of 1 HP. What does real work is a RESOLVED hit, which can remove 100 at a stroke.
            #
            # Review measured the HIT distribution on era 59's seed band at 75% one hit / 19.6%
            # two / 5.4% three (n=56) -- but that was on the retired hit-COUNT field, so it does
            # not by itself say how often `min_depletion` is 1. That needs one hit, unresolved,
            # with nothing resolved before it, and the resolved/unresolved mix was not measured.
            # Stated as the separate facts they are rather than fused into a claim about this
            # bound.
            #
            # A ~10x TIGHTER BOUND IS AVAILABLE and is the obvious follow-up rather than a
            # hypothetical: `gen3_damage.gen3_damage_rolls()` is engine-exact and already used
            # in production, and in a sampled world the attacker's set, spread, item and
            # boosts are all committed, so each unknown hit's damage is one of 16 consecutive
            # integers. That needs per-hit move identity in the payload, which this counter
            # does not carry.
            #
            # Sampling per world, not once: each world commits to its own hypothesis and
            # search averages over them, the same treatment trapped and disabled received.
            # Committing to one value globally would be the guess.
            raw_min = side_payload.get("substituteMinDepletion")
            # `bool` is excluded explicitly because `True` is an `int` in Python and would
            # otherwise become a bound of 1 from a flag. No `> 0` clause: the `< 1` gate below
            # already refuses zero and negatives, so adding one here was redundant rather than
            # protective -- mutation showed removing it changed nothing, which is how a
            # redundant guard reads as a tested one.
            min_depletion = (
                raw_min if isinstance(raw_min, int) and not isinstance(raw_min, bool) else 0
            )
            if min_depletion < 1:
                # NO BOUND SUPPLIED -> refuse exactly as before, under the SAME reason code.
                #
                # The justification is COMPATIBILITY, not range width. An earlier version said
                # sampling needs the bound because `[1, initial]` is "wide enough that a
                # sampled value would be a guess" -- incoherent, since the range this DOES
                # sample is 99.4% as wide when one unknown hit has landed. The real reason is
                # that a producer which has not been taught to accumulate must behave
                # bit-for-bit as before, and minting a different code here would rename a
                # refusal rather than fix it.
                #
                # UNREACHABLE from every live producer in this repo, which review established
                # and I had not: the `"unknown"` assignment and the accumulation are adjacent
                # statements and all four reset paths zero the accumulator, so
                # `state == "unknown"` implies `min_depletion >= 1`. Regenerating real protocol
                # on era 59's own seed band gave 187/187 observations with at least one hit and
                # none with zero (n=187, a different and larger sample than the n=56 hit
                # distribution cited above). "At least one hit implies a bound of at least 1" is
                # then arithmetic, not a second measurement, since the bound sums
                # `damage or 1` per hit. Defence in depth against a future producer, not a live
                # path.
                raise EngineWorldUnsupported(
                    "substitute_health_unknown",
                    f"side {slot!r} has explicit unknown Substitute health provenance "
                    f"with no bounded minimum depletion",
                )
            upper = initial_substitute_health - min_depletion
            if upper < 1:
                # More proven depletion than the sampled max HP can absorb: this WORLD is
                # inconsistent with the public record, not the record with itself. Refusing one
                # world is correct and is not a fallback -- the retry budget samples another.
                #
                # NOTE this DOES reclassify: a case that used to raise
                # `substitute_health_unknown` now raises
                # `substitute_depletion_world_incompatible`. Said plainly because the comments
                # above argue against renaming refusal classes, and this is a narrow instance
                # of exactly that. The existing exact-depletion path already treats the
                # analogous case the same way.
                raise EngineWorldUnsupported(
                    "substitute_depletion_world_incompatible",
                    f"side {slot!r} sampled max HP {party[active_index].maxhp} gives initial "
                    f"Substitute HP {initial_substitute_health}, which cannot absorb a proven "
                    f"minimum depletion of {min_depletion}",
                )
            if rng is None:
                # RAISE rather than default to `upper`. Taking the maximum silently biases
                # every world toward a near-full Substitute, and review found the first
                # version of the consumer test was itself taking that branch -- so the
                # `randint` path was not exercised at all.
                #
                # The SHARED reason code is deliberate: `engine_search`'s telemetry key
                # interpolates the detail string, so this and the no-bound refusal above
                # already separate in the raw sub-keys while sharing one class. Do not "tidy"
                # these messages into one, or a wiring bug merges into a hidden-information
                # class. There is precedent -- `pending_baton_pass` raises the same way for a
                # missing rng.
                raise EngineWorldUnsupported(
                    "substitute_health_unknown",
                    f"side {slot!r} needs an rng to sample bounded Substitute health",
                )
            # `randint(1, 1)` is 1, so no `upper > 1` special case: that guard was an
            # equivalent mutant -- dead code no test could distinguish.
            substitute_health = rng.randint(1, upper)
        else:
            raise EngineWorldUnsupported(
                "substitute_health_provenance_contradiction",
                f"side {slot!r} has active Substitute with invalid public state "
                f"{raw_substitute_health_state!r}",
            )
    else:
        if raw_substitute_depletion is not None:
            raise EngineWorldUnsupported(
                "substitute_health_provenance_contradiction",
                f"side {slot!r} has no Substitute volatile but carries depletion "
                f"{raw_substitute_depletion!r}",
            )
        if substitute_health_state not in {None, "", "absent", "broken"}:
            raise EngineWorldUnsupported(
                "substitute_health_provenance_contradiction",
                f"side {slot!r} has no Substitute volatile but health state "
                f"{raw_substitute_health_state!r}",
            )

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
    # `toxicStage` is a bridge-only pre-tick counter, not the public multiplier.  The engine
    # charges `toxic_count + 1`; 15 would therefore create an illegal stage-16 residual.
    toxic_stage = side_payload.get("toxicStage")
    if party[active_index].status == "toxic":
        if (
            isinstance(toxic_stage, bool)
            or not isinstance(toxic_stage, int)
            or not 0 <= toxic_stage <= 14
        ):
            raise EngineWorldUnsupported(
                "toxic_stage_unknown",
                f"side {slot!r} has active Toxic without a public toxicStage",
            )
        if toxic_stage > 0:
            side_conditions["toxic_count"] = toxic_stage
    elif toxic_stage is not None and (
        isinstance(toxic_stage, bool) or not isinstance(toxic_stage, int) or toxic_stage != 0
    ):
        raise EngineWorldUnsupported(
            "toxic_stage_inconsistent",
            f"side {slot!r} has toxicStage {toxic_stage!r} without active Toxic",
        )
    # Consecutive-Protect decay. The engine prices the NEXT stall attempt at
    # CONSECUTIVE_PROTECT_CHANCE ** side_conditions.protect (0.5 ** k), and only
    # branches at all when k > 0 — so an unseeded world says "this is a first
    # Protect" and returns a single 100%-success branch. Showdown's gen3 ladder
    # is 1, 1/2, 1/4, 1/8 by attempt, so k is exactly the count of consecutive
    # SUCCESSFUL stall uses immediately preceding this decision, with no offset.
    #
    # Publicly derivable, so this leaks nothing: the parser builds the count from
    # `-singleturn` (success) and resets it on a failed stall, any non-stall
    # move, `cant`, switch-out/drag or faint — the five public mirrors of
    # Showdown deleting the volatile. Same justification as the two-turn charge
    # state, which is likewise announced before it is used.
    stall_counter = side_payload.get("stallCounter")
    if isinstance(stall_counter, int) and stall_counter > 0:
        side_conditions["protect"] = stall_counter

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
    # Public last EXECUTED move for this side, straight off the payload (the parser derives it
    # from ``|move|`` / ``|cant|`` / switch lines under the engine's own truth table). Falls
    # back to the caller-supplied ``encored_move`` so the search lane, which derives the same
    # fact by scanning recent public events, keeps working unchanged.
    raw_last_used = side_payload.get("lastUsedMove")
    last_used_move_id = normalize_id(raw_last_used) if raw_last_used else None
    if last_used_move_id == "switch":
        last_used_move_id = "switch"
    volatile_durations: dict[str, int] = {}
    pending_encore_move: str | None = None
    if "encore" in volatiles and transformed_active:
        # DEFERRED, because this active's moveset is about to be REPLACED.
        #
        # Showdown locks Encore by move ID; the engine locks by move SLOT INDEX
        # (`last_used_move = move:<i>`). Resolving that index here would resolve
        # it against the PRE-Transform moveset, and `_apply_transform` (called
        # from `battle_spec_from_payload` right after this function returns)
        # then swaps the donor's moveset in underneath it -- so the surviving
        # index names a move nobody encored. Resolve an ID now; `_apply_encore_locks`
        # binds it to a slot once the final moveset exists.
        #
        # `_active_row_moves` must NOT be consulted for a transformed active.
        # That row is deliberately the pre-Transform snapshot:
        # `local_showdown.actor_move_states_from_request_history` skips requests
        # taken while transformed so that PP stays honest. For a gen3 randbats
        # Ditto it is the single move `transform` -- and a ONE-move list
        # satisfies the self-seat "exactly one enabled move identifies the lock"
        # rule SPURIOUSLY, yielding slot 0 for every such Encore. Measured on
        # holdout `19100170/71-72`: Showdown Encored Protect (donor slot 3), the
        # world locked donor slot 0 (Body Slam), and the phantom KO made
        # `end_of_turn_is_deferred` suppress the whole residual block.
        #
        # The id-keyed sources, in preference order:
        #   * `encored_move` -- the caller's publicly-observed lock (opponent seat).
        #   * `selfActiveMoves` -- the RAW request's usable moveset, which for a
        #     transformed active lists the COPIED moves with Encore's disable
        #     pattern already applied, so its single enabled entry IS the lock.
        #   * `sides[slot]["lastUsedMove"]` -- the parser's public last executed
        #     move, which under an active Encore is the encored move.
        #
        # The last two are SELF-SEAT ONLY, but note that deferral itself is not:
        # a transformed OPPONENT takes this path too, and that CHANGES ITS
        # COVERAGE. Before deferral its `encored_move` was matched against the
        # transformer's own moveset -- Ditto's `[transform]` -- so a real lock
        # like `protect` was absent and construction raised, a counted
        # `encore_move_unknown` skip. It now resolves against the copy and
        # builds. That is the correct world rather than a refusal, and an id the
        # copy does not contain still fails closed, but it can turn a skip into
        # a measured boundary. Unobserved in the dev/holdout windows; pinned by
        # tests/test_engine_world_encore_transform.py
        # ::EncoreOnATransformedOpponentTests.
        pending_encore_move = normalize_id(encored_move) if encored_move else None
        if pending_encore_move is None and is_self:
            pending_encore_move = _sole_enabled_move_id(self_active_request_moves)
            if pending_encore_move is None and last_used_move_id not in (None, "switch"):
                pending_encore_move = last_used_move_id
        if pending_encore_move is None:
            raise EngineWorldUnsupported(
                "encore_move_unknown",
                f"side {slot!r} is encored while transformed but the locked move "
                "cannot be determined",
            )
        # `last_used_move` stays empty here on purpose: `_apply_encore_locks` is
        # the single writer for a deferred lock, and leaving a placeholder would
        # give the field two writers that must agree.
        volatile_durations["encore"] = 1
    elif "encore" in volatiles:
        active_specs = party[active_index].moves
        encored_index = _resolve_encored_move_index(
            active_specs,
            rows_for_active=(
                _active_row_moves(side_payload) if is_self else None
            ),
            encored_move=encored_move,
            # THIRD source, lowest precedence, both seats — the same
            # `sides[slot]["lastUsedMove"]` the transformed branch above already
            # consults. The resolver, not this call site, owns the `"switch"`
            # sentinel: it is part of that field's vocabulary, so every caller
            # of the resolver should get the same reading of it.
            public_last_used_move=last_used_move_id,
        )
        if encored_index is None:
            raise EngineWorldUnsupported(
                "encore_move_unknown",
                f"side {slot!r} is encored but the locked move cannot be determined",
            )
        # The engine restricts the side to last_used_move while ENCORE is set.
        #
        # The duration IS now modelled, as of
        # third_party/poke-engine-gen3-encore-duration.patch: gen3 Showdown rolls
        # `this.random(3, 7)` (data/mods/gen3/moves.ts), so an Encore lasts 3-6
        # locked turns — not the 3-8 this comment used to claim, which came from
        # gen4's `this.random(4, 9)` and is the wrong side of the
        # gen3-inherits-gen4 boundary. The engine burns one tick per end-of-turn
        # and expires the volatile on a hazard ladder.
        #
        # The counter means "locked turns already elapsed", so seeding it at 1
        # says this Encore has been running for one turn. That stays a deliberate
        # floor: the true elapsed count is not observable from the request, and
        # under-counting keeps the lock modelled for at least as long as it really
        # lasts rather than freeing the mon early. Deriving the real value from
        # observation history is follow-up work, and it only becomes measurable
        # once the wheel carries the patch — same sequencing as the move-trap
        # wiring.
        last_used_move = f"move:{encored_index}"
        volatile_durations["encore"] = 1
    elif last_used_move_id:
        # Seed it for EVERY side, not only an already-encored one. Until 2026-07-29 this
        # block was the only writer, so a mon that had visibly just moved still reached the
        # engine as LastUsedMove::None -- and Encore's onStart reads it, so
        # `move_has_no_effect` fired `LastUsedMove::None => true` and Encore failed outright.
        # The engine was correct at every step; the world simply never told it. 11 rows of the
        # cycle-nine census were this, mislabelled as a missing same-turn redirect that the
        # engine has implemented all along (generate_instructions.rs, the onOverrideAction
        # mirror).
        if last_used_move_id == "switch":
            # A POSITIVE fact, not ignorance: a fresh switch-in genuinely has no lastMove
            # (`Pokemon.clearVolatile()`), and Encore correctly fails against it. The engine
            # has a distinct variant for exactly this.
            last_used_move = "switch:0"
        else:
            index = _resolve_encored_move_index(
                party[active_index].moves,
                rows_for_active=None,
                encored_move=last_used_move_id,
            )
            # Unresolvable means the observed move is not in the constructed moveset (an
            # unrevealed slot on a sampled world). Leave it None rather than guess: None is
            # the honest "this world does not know", and it reproduces exactly today's
            # behaviour for that side instead of inventing a lock.
            if index is not None:
                last_used_move = f"move:{index}"

    if "taunt" in volatiles:
        # Seed the counter at 1 for the same reason Yawn is seeded at 1 below,
        # and on the same arithmetic. Showdown applies Taunt (gen3 duration 2)
        # during a turn's move phase and burns the first tick at that same turn's
        # residual; singles offers no request in between, so every Taunt a
        # decision boundary can observe already has exactly one tick left.
        #
        # The engine's counter is TICKS ELAPSED, not turns remaining
        # (gen3/generate_instructions.rs 10.15: `0 => taunt += 1`,
        # `1 => remove the volatile`, anything else panics). Seeding 1 therefore
        # says "this is the last taunted turn", which is what the observation
        # means; the struct default of 0 would hold the searched mon taunted for
        # a second turn that Showdown has already freed. Read off the built wheel
        # rather than argued: at duration 1 the first end-of-turn emits
        # `RemoveVolatileStatus TAUNT`, at duration 0 it emits
        # `ChangeVolatileStatusDuration TAUNT: 1` instead
        # (tests/test_engine_world_taunt.py pins both).
        #
        # THIS SEED IS ONLY REACHED AT AN ORDINARY BOUNDARY. The replacement
        # boundary, where the count is ambiguous, is withdrawn from the allow-list
        # further up in this function and never gets here -- see that block for
        # the measurement.
        #
        # ⚠ A PREVIOUS REVISION OF THIS COMMENT CLAIMED THE OPPOSITE AND WAS
        # WRONG. It said a replacement turn "contributes NO move phase and NO
        # residual", so the seam "cannot reach Taunt", and offered that as the
        # structural discriminator against Yawn. The measurement behind it built
        # the engine state with `hp = 0` and NO `force_switch` flag, which is not
        # what `_build_side_spec` emits: gen3 `get_all_options` checks the
        # explicit flag FIRST, and `end_of_turn_triggered` returns true on that
        # flag, so production takes the arm that DOES run the residual. Withdrawn
        # rather than reworded.
        #
        # There is no structural discriminator against Yawn, and none is claimed.
        # The difference is a POLICY one, stated plainly: at the ambiguous
        # boundary Taunt fails closed and Yawn is approximated behind
        # `approximate_hidden_duration_volatiles`. Whether Yawn should fail closed
        # too is not measured here and nothing about Yawn is changed.
        volatile_durations["taunt"] = 1

    if "yawn" in volatiles:
        # Seed the counter at 1, NOT the struct default of 0. Showdown applies
        # Yawn (duration 2) during a turn's move phase and burns the first tick
        # at that same turn's residual; in singles there is no request between
        # those two points, so EVERY Yawn a decision boundary can observe is
        # already one tick old. Leaving the engine's 0 would push the sleep a
        # full turn late in the normal case, not just an edge case.
        volatile_durations["yawn"] = 1

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
        pending_encore_move,
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


def _sole_enabled_move_id(rows: Sequence[Mapping[str, Any]] | None) -> str | None:
    """The one enabled move id in a request-shaped move list, or None.

    Encore's request signature: every move except the locked one is reported
    ``disabled``. Exactly one enabled entry therefore names the lock, and any
    other count means these rows do not identify it. Callers must satisfy
    themselves that the rows describe the CURRENT usable moveset -- a
    pre-Transform snapshot can hold a single move for reasons that have nothing
    to do with Encore, which is precisely how this rule was once satisfied
    spuriously.
    """

    if not rows:
        return None
    enabled = [
        normalize_id(str(row.get("id")))
        for row in rows
        if isinstance(row, Mapping)
        and isinstance(row.get("id"), str)
        and not bool(row.get("disabled"))
    ]
    if len(enabled) == 1:
        return enabled[0]
    return None


def _move_index_by_id(move_specs: Sequence[Any], move_id: str) -> int | None:
    """Slot of ``move_id`` in the constructed move order, or None.

    The ``hiddenpower*`` prefix tolerance is deliberate: the request names the
    typed variant (``hiddenpowerground70``) while a sampled spec may carry a
    differently-typed one, and gen3 has at most one Hidden Power slot, so the
    prefix is unambiguous within a single moveset.
    """

    target = normalize_id(move_id)
    for index, spec in enumerate(move_specs):
        spec_id = normalize_id(spec.id)
        if spec_id == target:
            return index
        if target.startswith("hiddenpower") and spec_id.startswith("hiddenpower"):
            return index
    return None


def _resolve_encored_move_index(
    move_specs: Sequence[Any],
    *,
    rows_for_active: Sequence[Mapping[str, Any]] | None,
    encored_move: str | None,
    public_last_used_move: str | None = None,
) -> int | None:
    """Index of the encored move in the constructed move order, or None.

    Three id sources, in precedence order — the same ladder the transformed
    branch of ``_build_side_spec`` already walks:

    1. ``encored_move`` -- the caller's publicly-observed lock. On the opponent
       seat the search lane derives this by scanning the observation's
       ``recent_public_events``.
    2. ``rows_for_active`` -- the self request's move rows, where Encore's
       signature is "exactly one enabled entry" (see ``_sole_enabled_move_id``).
    3. ``public_last_used_move`` -- the parser's public last EXECUTED move for
       this side (``sides[slot]["lastUsedMove"]``).

    Source 3 exists because sources 1 and 2 are both WINDOWED or absent, and
    source 3 is neither:

    * ``recent_public_events`` is ``replay.public_events[-24:]``
      (``showdown.py`` ``recent_event_limit``). Once an Encored target is
      immobilised for a few turns — asleep, frozen, recharging — its last
      ``|move|`` line scrolls out of that window and only ``|cant|`` lines
      remain, so source 1 goes silent while the Encore is still live. This was
      95/1682 gen3 randbat variants' worth of `encore_move_unknown` refusals.
    * Source 2 does not exist on the opponent seat at all: there is no opposing
      self-request to read a disable pattern from.
    * ``replay.last_used_move`` is a per-slot latch with NO window. It is
      written only by a non-``[from]`` ``|move|`` line and reset to the
      ``"switch"`` sentinel on switch/drag, i.e. it is the identical fact source
      1 scans for, retained indefinitely instead of for 24 lines.

    Soundness of source 3 as an ENCORE lock, which is why it is admissible here
    and not a guess: Showdown's Encore locks ``target.lastMove`` and refuses to
    start when there is none (``!move``), so the lock and the latch agree at
    onset; while the volatile holds, the target can only execute the encored
    move, and an immobilised turn emits ``|cant|`` which does not touch the
    latch; Encore is ``noCopy`` and cannot survive a switch, which is exactly
    what the ``"switch"`` sentinel filtered below would represent; and
    Showdown's ``encore`` condition removes itself at the residual once the
    encored move's PP hits zero, so an Encore can never be observed alongside a
    Struggle substitution. A called move (Sleep Talk's callee, Metronome's) is
    excluded from the latch by the same ``[from]`` discriminator source 1 uses.

    Fail-closed is preserved end to end. An empty/absent latch returns None, the
    ``switch`` sentinel returns None, and an id the SAMPLED moveset does not
    contain returns None from ``_move_index_by_id``; all three reach the
    caller's ``encore_move_unknown`` raise. Nothing here invents a slot.

    This is the NON-transformed path only. A transformed active defers to
    ``_apply_encore_locks``, because ``move_specs`` here predates the copy.
    """

    if encored_move:
        return _move_index_by_id(move_specs, encored_move)
    target = _sole_enabled_move_id(rows_for_active)
    if target is None:
        # Strictly additive: only consulted where the two windowed sources
        # already returned nothing, so no world that constructs today changes.
        # Left un-normalized otherwise -- ``_move_index_by_id`` normalizes its
        # target, and a second call here would be dead defensive code.
        latch = public_last_used_move or None
        # ``"switch"`` is the one member of the latch's vocabulary that is not a
        # move id: the parser writes it on switch/drag to say "this mon has no
        # last move" (``Pokemon.clearVolatile()`` nulls ``lastMove``). The rule
        # lives here, not at the call site, because it is a fact about the FIELD
        # -- and because relying on "no move is named switch, so the lookup
        # misses anyway" would make the guard behaviour-inert and untestable.
        if latch is not None and normalize_id(latch) != "switch":
            target = latch
    if target is None:
        return None
    return _move_index_by_id(move_specs, target)


def _apply_encore_locks(
    sides: Mapping[str, SideSpec],
    pending: Mapping[str, str],
) -> dict[str, SideSpec]:
    """Bind each DEFERRED Encore lock to a slot in the final moveset.

    Runs after ``_apply_transform``, so ``active.moves`` is the donor's copied
    set and the index this writes is the one the engine will read.

    Fail-closed on purpose, and identically to the non-deferred path: an id that
    is not in the copied moveset means the world cannot express the lock, so it
    refuses to build rather than inventing one. Falling back to slot 0 here is
    exactly the defect this function exists to remove.

    ORDERING, which deferral does change for a transformed side. The
    "no id at all" refusal still fires inside ``_build_side_spec``, before
    ``self_world_mismatch`` and ``transform_unexpressible``, exactly as it always
    did. But the "id absent from the moveset" refusal now fires HERE, i.e. AFTER
    both of those. So a transformed side that would fail both checks is now
    attributed to the earlier one. Non-transformed sides are unaffected, and the
    measured skip histogram is unchanged on both 200-game windows.
    """

    updated = dict(sides)
    for slot, move_id in pending.items():
        side = updated.get(slot)
        if side is None or not side.pokemon:
            raise EngineWorldUnsupported(
                "encore_move_unknown", f"side {slot!r} has no built side to encore"
            )
        active = side.pokemon[side.active_index]
        index = _move_index_by_id(active.moves, move_id)
        if index is None:
            raise EngineWorldUnsupported(
                "encore_move_unknown",
                f"side {slot!r} is encored on {move_id!r}, absent from the "
                f"post-Transform moveset {[spec.id for spec in active.moves]}",
            )
        updated[slot] = replace(side, last_used_move=f"move:{index}")
    return updated


def _resolved_ability(mon: Any, row: Mapping[str, Any] | None) -> str | None:
    """The ability the world should carry: protocol-confirmed if there is one, else sampled."""
    if row is not None:
        revealed = row.get("revealedAbility")
        if isinstance(revealed, str) and revealed.strip():
            return normalize_id(revealed)
    return normalize_id(mon.ability) if mon.ability else None


def _build_pokemon_spec(
    mon: FixturePokemon,
    row: Mapping[str, Any] | None,
    *,
    dex: ShowdownDex,
    slot: str,
    is_self: bool,
    self_benched_move_history: bool = False,
    approximate_sleep_turns: bool = False,
    item_removed: bool = False,
    item_override: str | None = None,
    is_transformed_active: bool = False,
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

    resolved_ability = _resolved_ability(mon, row)
    hp, status, rest_turns, rest_sleep_pending_refund = _hp_and_status(
        row,
        maxhp=maxhp,
        slot=slot,
        species=mon.species,
        is_self=is_self,
        approximate_sleep_turns=approximate_sleep_turns,
        rest_sleep_early_bird=resolved_ability == "earlybird",
    )
    if rest_turns:
        # Gated on the wheel actually preserving the counter, for the same reason the
        # trapped volatile is: a binding that accepts ``rest_turns`` and drops it builds
        # this mon as an ORDINARY sleeper, which re-arms the Sleep Clause the Rest is
        # exempt from and hands search a sleep move Showdown would refuse. Failing the
        # decision is strictly better than searching a rule the sim does not have.
        require_rest_turns_support()
    if rest_sleep_pending_refund:
        # Separate probe from rest_turns, because the failure is the opposite
        # direction: a binding that drops this field does not re-arm Sleep Clause,
        # it loses turns the sim credits back and wakes the mon EARLY on every
        # branch that switches it in. Both are silent, so both get probed.
        require_rest_sleep_refund_support()
    moves = _move_specs(
        mon,
        row,
        dex=dex,
        slot=slot,
        is_self=is_self,
        self_benched_move_history=self_benched_move_history,
        is_transformed_active=is_transformed_active,
    )
    public_gender = _gender_from_details(str(row.get("details") or "")) if row else None

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
        # ``sleep_turns`` is deliberately left at its 0 default alongside a non-zero
        # ``rest_turns``: the engine reads the two as alternatives, branching on
        # ``rest_turns`` first and consulting ``sleep_turns`` only in its 0 arm
        # (gen3/generate_instructions.rs), so a Rest sleep has no elapsed-turn count
        # to carry and inventing one would be read by nothing.
        rest_turns=rest_turns,
        rest_sleep_pending_refund=rest_sleep_pending_refund,
        # A protocol-CONFIRMED ability beats the sampled battle-start assignment, for the
        # same reason the item does below: Trace publicly replaces the holder's ability
        # mid-battle, and a world rebuilt from the sampled set hands the engine TRACE, which
        # plays the mon without the copied ability entirely (damaging through a traced Flash
        # Fire immunity, seed 1500248 steps 77-78).
        #
        # This also retires a claim in _SUPPORTED_VOLATILES above, which held that the
        # boost-only `flashfire` volatile could be wrong only "if a sampled world lacked the
        # ability, which cannot happen for the mono-ability Gen 3 randbats carriers". True of
        # NATIVE carriers; a Trace user that ACQUIRED Flash Fire is exactly that case.
        #
        # Ability field only -- gen3 does not fire the copied ability's Start event on
        # acquisition (#962 patch 32), so no activation is simulated here.
        ability=resolved_ability,
        # The CURRENT public item state beats the sampled battle-start
        # assignment: a publicly-stripped/consumed item is gone
        # (item_removed), and a Trick-swapped mon holds exactly the item the
        # protocol named (item_override). Stats above deliberately keep the
        # original assignment's spread — Trick moves only the item. The two
        # signals are mutually exclusive (item_state_conflict guard upstream).
        item=(
            item_override
            if item_override
            else (None if item_removed else (normalize_id(mon.item) if mon.item else None))
        ),
        gender=public_gender or mon.gender,
        weight_kg=info.weight_kg if info.weight_kg > 0 else None,
    )


def _gender_from_details(details: str) -> str | None:
    for part in details.split(","):
        token = part.strip().upper()
        if token in {"M", "F", "N"}:
            return token
    return None


def _hp_and_status(
    row: Mapping[str, Any] | None,
    *,
    maxhp: int,
    slot: str,
    species: str,
    is_self: bool,
    approximate_sleep_turns: bool = False,
    rest_sleep_early_bird: bool = False,
) -> tuple[int, str, int, int]:
    """Return ``(hp, engine status, rest_turns, rest_sleep_pending_refund)`` for one row.

    ``rest_turns`` is non-zero only for a mon whose sleep the public protocol attributed
    to its own Rest; see :func:`_rest_turns_from_row`.
    """

    if row is None:
        return maxhp, "none", 0, 0
    condition = str(row.get("condition") or "")
    if not condition:
        raise EngineWorldUnsupported("payload_malformed", f"{slot}: {species!r} row has no condition")
    hp_part, _, status_part = condition.partition(" ")
    status_code = status_part.strip()
    if status_code == "fnt" or hp_part == "0":
        return 0, "none", 0, 0
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
        if status_code == _SLEEP_STATUS_CODE:
            if bool(row.get("restSleepProvenanceUnrepresentable")):
                raise EngineWorldUnsupported(
                    "rest_sleep_provenance_unrepresentable",
                    f"{slot}: {species!r} has malformed public Rest provenance",
                )
            # These two were ONE code (`rest_sleep_skipped_time_pending`) and ONE row
            # flag (`restSleepRefundPending`) until this split. They have different
            # causes and different owners, and conflating them made the class
            # unsizeable: era-57 could not say how much of its 607 decisions an engine
            # field would actually recover.
            #
            # BOTH old names are retired, and the flag rename is not cosmetic. Stored
            # corpora keep `public_materialization` rows verbatim (`golden_corpus.py`),
            # and several scripts re-feed those rows straight back through here. A
            # pre-split row carrying `restSleepRefundPending` may have come from EITHER
            # producer, so reusing that flag for B would silently bank producer-A rows
            # as B's share -- the exact misattribution the code retirement prevents on
            # the reason axis. Legacy rows therefore get their own third code: they
            # still refuse, and they are never counted as either producer.
            if bool(row.get("restSleepAttemptUnsettled")):
                # Harness/observation: the attempt is unclassified because the snapshot
                # landed mid-turn. Not an engine limitation.
                raise EngineWorldUnsupported(
                    "rest_sleep_attempt_unsettled",
                    f"{slot}: {species!r} has an unsettled public Rest sleep attempt",
                )
            if bool(row.get("restSleepActiveRefundPending")):
                # NO LONGER A REFUSAL. The engine now carries the pending refund on
                # the mon (`Pokemon.rest_sleep_pending_refund`) and spends it only on
                # the branch where a switch-in actually happens, so the value that
                # used to have nowhere to go now has somewhere to go.
                #
                # The refund is deliberately NOT folded into rest_turns here. Folding
                # it in is right for a BENCHED mon -- a switch-in necessarily precedes
                # its next attempt -- and wrong for an active one, which may never
                # switch while search explores the stay-in branch. Doing both would
                # credit the same turns twice.
                if "restSleepAttempts" not in row:
                    # A row written by the PRE-write-side harness: it set the flags
                    # and withheld the counts. Fail closed, but under its OWN code --
                    # falling through to `provenance_unrepresentable` would blame
                    # malformed public data for what is really a stale corpus, and
                    # silently inflate a different counted class in replay telemetry.
                    # The legacy canary cannot catch it either: these rows carry the
                    # producer flag and return before that check.
                    raise EngineWorldUnsupported(
                        "rest_sleep_refund_pending_precounts_legacy",
                        f"{slot}: {species!r} carries a pending Rest refund with no "
                        "attempt counts, so it predates the write side",
                    )
                skipped = row.get("restSleepSkippedTime", 0)
                rest_turns = _rest_turns_from_row(
                    row, early_bird=rest_sleep_early_bird, fold_skipped=False
                )
                if rest_turns is None or not isinstance(skipped, int) or isinstance(skipped, bool):
                    raise EngineWorldUnsupported(
                        "rest_sleep_provenance_unrepresentable",
                        f"{slot}: {species!r} has unrepresentable public Rest provenance "
                        "for a pending refund",
                    )
                return hp, "sleep", rest_turns, skipped
            # LAST, and load-bearing that it is last. A live row sets this flag too
            # (see `_mark_legacy_rest_refund_pending`) so that a pre-split checkout
            # replaying it still refuses instead of silently approximating. Reaching
            # HERE therefore means the row carries the old flag and NEITHER producer
            # flag -- i.e. it really was written before the split, and its producer is
            # not recoverable. Moving this check earlier would swallow every live row.
            #
            # CANARY: in a fresh post-split era this code must count exactly ZERO.
            # Every live row carries a producer flag, so anything landing here came
            # from a replayed pre-split corpus or a mixed-version fleet. A nonzero
            # count in new telemetry is a signal to act on, not noise.
            if bool(row.get("restSleepRefundPending")):
                raise EngineWorldUnsupported(
                    "rest_sleep_refund_pending_unsplit_legacy",
                    f"{slot}: {species!r} carries the pre-split Rest refund flag alone, "
                    "whose producer is not recoverable from the row",
                )
            if "restSleepAttempts" in row:
                rest_turns = _rest_turns_from_row(row, early_bird=rest_sleep_early_bird)
                if rest_turns is not None:
                    # EXACT, not approximated: the public attempt count reconstructs the
                    # engine's own Rest counter with nothing left to guess.
                    return hp, "sleep", rest_turns, 0
                # A present annotation is positive Rest provenance, even when its
                # counter is invalid. Never reinterpret it as induced sleep: that
                # would zero rest_turns and re-arm Sleep Clause.
                raise EngineWorldUnsupported(
                    "rest_sleep_provenance_unrepresentable",
                    f"{slot}: {species!r} has unrepresentable public Rest provenance",
                )
            if approximate_sleep_turns:
                # Documented approximation only for an unannotated, induced sleep.
                # Model the mon as freshly asleep (sleep_turns=0); this biases wake-up
                # odds late in a sleep. The exact fix is public sleep-counter tracking.
                return hp, "sleep", 0, 0
        raise EngineWorldUnsupported(
            "status_unsupported",
            f"{slot}: {species!r} status {status_code!r} (sleep needs public turn counts)",
        )
    return hp, status, 0, 0


def _rest_turns_from_row(
    row: Mapping[str, Any], *, early_bird: bool = False, fold_skipped: bool = True
) -> int | None:
    """Rebuild the engine's Rest counter from the row's public attempt count.

    ``restSleepAttempts`` (k) is written by ``local_showdown._apply_rest_sleep_provenance``
    for exactly those mons whose ``slp`` the protocol attributed to their OWN Rest and
    that the opposing side never put to sleep; an induced sleeper never carries it, so
    the field's presence IS the provenance and no second lookup is needed here.

    The arithmetic is exact rather than approximate because both clocks are the same
    clock. The engine sets ``rest_turns = 3`` on Rest and decrements it once per move
    ATTEMPT, waking the mon when it reaches 1 (gen3/generate_instructions.rs); k counts
    those same attempts off the public ``|cant|SLOT|slp`` lines, which gen3 emits on
    precisely the attempts that tick and on no others. Early Bird consumes two timer units
    per attempt. ``restSleepSkippedTime`` is the trailing public Sleep Talk/Snore refund that
    the next switch-in will restore; ``restSleepRefundedTime`` records prior switch-in refunds.
    Each restores one unit. The exact remaining counter is therefore
    ``3 - k * (2 if Early Bird else 1) + refunded + skipped``.

    For a non-Early-Bird world with no refund this is ``3 - k``, NOT ``4 - k``. The
    off-by-one argument goes: the engine wakes at ``rest_turns == 1`` while Showdown wakes
    at ``time == 0``, so the engine must run one ahead. That compares Showdown's counter
    AFTER its decrement against the engine's BEFORE its own. Both decrement on the attempt,
    and both wake on the attempt whose PRE-decrement counter is 1::

        showdown  data/mods/gen3/conditions.ts:  time--;  if (time <= 0) cure and move
        engine    gen3/generate_instructions.rs: match rest_turns { 1 => wake, 2|3 => stay }

    Measured rather than argued — an ordinary Rest (``time``/``rest_turns`` = 3) costs
    three attempts on both sides, of which exactly the first two are the non-acting ones
    that emit ``|cant|``:

        k=0  ->  rest_turns 3  ->  2 more cants, then it acts
        k=1  ->  rest_turns 2  ->  1 more cant,  then it acts
        k=2  ->  rest_turns 1  ->  0 more cants; the very next attempt acts

    ``scripts/gen3_switch_differential.py --only restattemptclock`` walks that ladder at
    the real sim; ``gen3_rest_sleep_clause.rs`` walks the engine's and asserts the table
    above row by row.

    ``4 - k`` gets every ordinary row wrong, k=0 included. NOTHING CLAMPS IT — not here
    (the range check below is on the INPUT k, never on the returned counter) and not in
    the adapter (which validates only non-negativity). What differs between the rows is
    how the wrongness surfaces: k=0 would build ``rest_turns = 4``, a value the engine has
    no match arm for and panics on (``Invalid rest_turns value: 4``, pinned by
    ``a_rest_counter_of_four_is_not_a_representable_state``), while k=1 and k=2 build 3
    and 2 — legal counters, silently one attempt late.

    A k=0 check WOULD have caught ``4 - k``, loudly. Every ordinary row is pinned for
    REACHABILITY: a fresh, un-attempted Rest dominates the corpus, so an error confined to
    k=1/k=2 can ride along in production data. Early Bird and skippedTime remain separate
    pinned cases because their arithmetic is not the ordinary ``3 - k`` table.

    Why this matters beyond the wake timer: gen3's Sleep Clause Mod exempts a Rest sleep
    (``rulesets.ts`` skips a sleeper whose ``statusState.source`` is its own ally), and
    the engine spells that exemption as ``rest_turns == 0`` in
    ``has_alive_non_rested_sleeping_pkmn``. A Rest-sleeper built with ``rest_turns = 0``
    is therefore not merely mis-timed, it silently re-arms a clause the real battle does
    not have — and on the BENCH, where nothing else would ever reveal the error.
    """

    attempts = row.get("restSleepAttempts")
    refunded = row.get("restSleepRefundedTime", 0)
    skipped = row.get("restSleepSkippedTime", 0)
    # Bools are ints in Python; an accidental True must not read as one attempt.
    if (
        isinstance(attempts, bool)
        or not isinstance(attempts, int)
        or isinstance(refunded, bool)
        or not isinstance(refunded, int)
        or isinstance(skipped, bool)
        or not isinstance(skipped, int)
    ):
        return None
    if (
        attempts < 0
        or refunded < 0
        or skipped < 0
        or refunded + skipped > attempts
    ):
        # Refunds cannot outnumber public attempts. Anything else means the tracker
        # and this arithmetic have drifted apart, which is exactly the silent
        # wrongness the constructor fails closed on rather than clamping into.
        return None
    if early_bird and attempts > 1:
        # A Rest starts at three units, so the second Early Bird attempt wakes before it
        # can emit another sleep cant. Treat a larger public count as inconsistent.
        return None
    # ``fold_skipped=False`` is for an ACTIVE sleeper, whose refund rides on the mon
    # as ``rest_sleep_pending_refund`` and is spent only on the branch where a
    # switch-in actually happens. Folding it in here as well would credit the same
    # turns TWICE -- with attempts=2, refunded=0, skipped=1 the true counter is 1
    # with 1 pending, but a fold plus a pending field gives the engine 2 + 1 and it
    # clamps to 3. Benched mons keep the fold, because for them a switch-in
    # necessarily precedes the next attempt.
    rest_turns = (
        _REST_SLEEP_TURNS
        - attempts * (2 if early_bird else 1)
        + refunded
        + (skipped if fold_skipped else 0)
    )
    if not fold_skipped:
        # Still floored at 1, NOT 0. Two independent reasons, and both bite silently:
        # the engine's wake match treats rest_turns == 0 as ordinary sleep and reads
        # sleep_turns instead, and its switch-in consumer is guarded on
        # rest_turns > 0, so a 0 counter would never be credited the refund at all.
        # A row implying 0 remaining with turns still banked is therefore not
        # representable, and refusing beats building a Sleep-Clause-re-armed mon.
        if not 1 <= rest_turns <= _REST_SLEEP_TURNS:
            return None
        # The SUM is what the engine holds once the refund lands, so it is the sum
        # that has to fit. A BACKSTOP, not the decisive check, and review proved it:
        # the `refunded + skipped > attempts` rejection above already forces
        # 3 - k*(1|2) + refunded + skipped <= 3, and an exhaustive sweep over
        # (k, refunded, skipped) x Early Bird found zero inputs where this line is
        # the one doing the rejecting. Kept because the fail-closed contract should
        # not depend on that derivation staying true.
        return rest_turns if rest_turns + skipped <= _REST_SLEEP_TURNS else None
    return rest_turns if 1 <= rest_turns <= _REST_SLEEP_TURNS else None


def _move_specs(
    mon: FixturePokemon,
    row: Mapping[str, Any] | None,
    *,
    dex: ShowdownDex,
    slot: str,
    is_self: bool,
    self_benched_move_history: bool = False,
    is_transformed_active: bool = False,
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

    if is_self and known_pp and not is_transformed_active:
        sampled_ids = {normalize_id(move) for move in mon.moves}
        sampled_has_hp = any(m.startswith("hiddenpower") for m in sampled_ids)
        for request_move in known_pp:
            if request_move in sampled_ids:
                continue
            if request_move.startswith("hiddenpower") and sampled_has_hp:
                continue
            # Scalars FIRST, list LAST, AND the prose kept short enough that both
            # scalars fit. deployment/mcts/analyze_probe.py collapses each reason on
            # `split(": ")[-1]` and then prints only `reason[:88]`, so anything past that
            # is cut off entirely. Two earlier versions of this message failed here: one
            # put a flag after the moveset, and the reorder alone was still too long for
            # `has_transform` to survive. Verified against the real truncation rather than
            # assumed -- the one case that had looked legible was Ditto, whose set is a
            # single move and happened to fit.
            #
            # WHAT THESE THREE FIELDS SEPARATE, and what an earlier version got wrong:
            #
            # There is NO belief-sampling step for our own moveset. `mon.moves` here is
            # the battle-START request-known team, taken verbatim from the root snapshot
            # (`determinization._self_team_from_metadata_result` canonicalises
            # `request_moves` and nothing else). So "our sample is wrong" was never the
            # alternative hypothesis, and a `mimic` flag was dead on arrival: gen 3
            # randbats has no Mimic in its sets at all, and our own team could not
            # contain one even if it did. That field could only ever read False.
            #
            # The live causes, which these fields do separate:
            #   * `active` -- a mismatch on a BENCHED mon cannot be in-flight Transform
            #     or Mimic, because both revert on switch-out. It is root-snapshot vs
            #     current-request divergence, and failing closed is correct.
            #   * `sampled_has_transform` -- transform really is in gen 3 randbats, on
            #     ditto and mew only, so this names the population the guard at
            #     `is_transformed_active` is supposed to have suppressed. If it is True
            #     AND active is True, that suppression missed.
            #   * `species` -- sibling refusals in this same function already carry it;
            #     `slot` is only the side, so the old message never named the mon.
            sampled_sorted = ",".join(sorted(sampled_ids))
            raise EngineWorldUnsupported(
                "self_moveset_mismatch",
                f"{slot}: {mon.species!r} "
                f"active={bool((row or {}).get('active'))} "
                f"has_transform={'transform' in sampled_ids} "
                f"move {request_move!r} absent from root self_team [{sampled_sorted}]",
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
