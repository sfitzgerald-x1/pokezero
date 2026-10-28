"""Device public encounter/HP/status/boost ledger for the encoder input fold.

All arguments are publicly resolved facts. This module cannot read private
requests or physical engine state. Species IDs belong to the caller's frozen
public dex catalog; they are distinct from set-source IDs and vocabulary rows.
Full protocol attribution, active-state clocks and observation projection are
still separate required consumers of the typed public stream.
"""
from typing import NamedTuple

import jax
import jax.numpy as jnp


SCHEMA = "pokezero.public_mon_state.v1"
NONE, SLEEP, PARALYSIS, BURN, FREEZE, POISON, TOXIC = range(7)
SWITCH, DRAG, REPLACE, BATON_PASS = range(4)
SET, DELTA, CLEAR, CLEAR_POSITIVE, CLEAR_NEGATIVE = range(5)
STATS = ("atk", "def", "spa", "spd", "spe", "accuracy", "evasion")


class MonState(NamedTuple):
    species_binding: jax.Array      # [B,8], frozen public dex namespace, zero = legacy unbound
    encounter: jax.Array            # [B,2,6] stable public encounter identity
    species: jax.Array              # opaque public dex namespace; zero unknown
    source_species: jax.Array       # set-source namespace; zero unavailable
    level: jax.Array
    gender: jax.Array               # 0 unknown/genderless, 1 male, 2 female
    hp_numerator: jax.Array
    hp_denominator: jax.Array
    hp_known: jax.Array
    status: jax.Array
    fainted: jax.Array
    active: jax.Array               # [B,2] public encounter slot, -1 unknown
    boosts: jax.Array               # [B,2,7] current active's public boosts
    arrived_by_baton: jax.Array     # [B,2]


def namespace_binding(batch, binding=None):
    if binding is None:
        return jnp.zeros((batch,8),jnp.uint32)
    if getattr(binding,'shape',None)!=(8,) or getattr(binding,'dtype',None)!=jnp.dtype(jnp.uint32):
        raise ValueError('public species namespace requires eight uint32 words')
    return jnp.broadcast_to(binding,(batch,8))


def initialize(batch: int, *, species_binding=None) -> MonState:
    if type(batch) is not int or batch <= 0:
        raise ValueError("positive public-state batch required")
    mons = (batch, 2, 6)
    z = lambda: jnp.zeros(mons, jnp.int32)
    return MonState(namespace_binding(batch,species_binding), z(), z(), z(), z(), z(), z(), z(), jnp.zeros(mons, bool),
        z(), jnp.zeros(mons, bool), jnp.full((batch, 2), -1, jnp.int32),
        jnp.zeros((batch, 2, len(STATS)), jnp.int32), jnp.zeros((batch, 2), bool))


def _int32(value):
    """Typed transport must not alias floats, booleans or wide IDs by casting."""
    if type(value) is int:
        if not -(2**31) <= value < 2**31:
            raise ValueError("public state word exceeds int32")
        return jnp.asarray(value, jnp.int32)
    if getattr(value, "dtype", None) != jnp.dtype(jnp.int32):
        raise ValueError("public state words must be int32 without coercion")
    return jnp.asarray(value)


def _address(state, side, slot):
    side, slot = _int32(side), _int32(slot)
    batch = state.encounter.shape[0]
    side = jnp.broadcast_to(side, (batch,))
    slot = jnp.broadcast_to(slot, (batch,))
    return jnp.arange(batch), jnp.clip(side, 0, 1), jnp.clip(slot, 0, 5), (side >= 0) & (side < 2) & (slot >= 0) & (slot < 6)


def _write(array, address, value, admitted):
    before = array[address]
    mask = admitted.reshape(admitted.shape + (1,) * (before.ndim - admitted.ndim))
    return array.at[address].set(jnp.where(mask, value, before))


def arrive(state: MonState, *, side, slot, encounter, species, source_species,
           level, gender, hp_numerator, hp_denominator, status, kind=SWITCH, valid=True):
    """Commit a resolved public arrival, retaining persistent identity per slot.

    A different encounter cannot overwrite an occupied slot; a returning known
    encounter reuses its slot. Equal species are never an identity join. Only
    explicit Baton Pass carries active boost stages; switch/drag/replace reset
    them exactly as the Python parser's arrival block does. Other Baton Pass
    volatile/clock consumers must be ported before a full arrival is admitted.
    """
    row, seat, target, address_ok = _address(state, side, slot)
    encounter, species, source_species, level, gender, hp_numerator, hp_denominator, status, kind = (
        _int32(x) for x in (encounter, species, source_species, level, gender, hp_numerator, hp_denominator, status, kind))
    old_identity = state.encounter[row, seat, target]
    # Identity is unique on a side, including across two duplicate species.
    existing = state.encounter[row, seat]
    unique = ~jnp.any((existing == encounter[..., None]) & (jnp.arange(6) != target[:, None]), axis=-1)
    typed = (encounter > 0) & (species >= 0) & (source_species >= 0) & (level >= 0) & (level <= 100) & (gender >= 0) & (gender <= 2)
    hp_ok = (hp_numerator >= 0) & (hp_denominator > 0) & (hp_numerator <= hp_denominator)
    fields_ok = typed & hp_ok & (status >= NONE) & (status <= TOXIC) & (kind >= SWITCH) & (kind <= BATON_PASS)
    admitted = address_ok & unique & ((old_identity == 0) | (old_identity == encounter)) & fields_ok & jnp.asarray(valid)
    address = row, seat, target
    output = state._replace(**{name: _write(getattr(state, name), address, value, admitted) for name, value in (
        ("encounter", encounter), ("species", species), ("source_species", source_species),
        ("level", level), ("gender", gender), ("hp_numerator", hp_numerator),
        ("hp_denominator", hp_denominator), ("status", status), ("hp_known", True), ("fainted", False))})
    side_address = row, seat
    output = output._replace(
        active=_write(state.active, side_address, target, admitted),
        boosts=_write(state.boosts, side_address,
            jnp.where((kind == BATON_PASS)[..., None], state.boosts[side_address], 0), admitted),
        arrived_by_baton=_write(state.arrived_by_baton, side_address, kind == BATON_PASS, admitted))
    return output, admitted


def hp_update(state: MonState, *, side, slot, encounter, numerator, denominator, valid=True):
    """HP-only consumer; apply condition status/faint facts separately as needed.

    Retain the wire denominator; a rounded percentage is not exact engine HP.
    Full damage/heal messages should use condition_update below.
    """
    row, seat, target, address_ok = _address(state, side, slot)
    encounter, numerator, denominator = (_int32(x) for x in (encounter, numerator, denominator))
    admitted = address_ok & (encounter > 0) & (state.encounter[row, seat, target] == encounter) & (
        numerator >= 0) & (denominator > 0) & (numerator <= denominator) & jnp.asarray(valid)
    address = row, seat, target
    return state._replace(hp_numerator=_write(state.hp_numerator, address, numerator, admitted),
        hp_denominator=_write(state.hp_denominator, address, denominator, admitted),
        hp_known=_write(state.hp_known, address, True, admitted)), admitted


def condition_update(state: MonState, *, side, slot, encounter, numerator, denominator,
                     status, fainted, valid=True):
    """Commit all condition fields from a public damage/heal/sethp message.

    A bare '0 fnt' has no denominator; preserve that absence as zero rather than
    manufacturing a max-HP fact. Its zero fraction is nevertheless known.
    """
    row, seat, target, address_ok = _address(state, side, slot)
    encounter, numerator, denominator, status = (_int32(x) for x in (encounter, numerator, denominator, status))
    fainted = jnp.asarray(fainted)
    if fainted.dtype != jnp.bool_:
        raise ValueError("public faint fact must be boolean")
    hp_ok = ((denominator > 0) & (numerator >= 0) & (numerator <= denominator)) | (
        fainted & (numerator == 0) & (denominator == 0))
    admitted = address_ok & (encounter > 0) & (state.encounter[row, seat, target] == encounter) & (
        status >= NONE) & (status <= TOXIC) & hp_ok & jnp.asarray(valid)
    address = row, seat, target
    return state._replace(**{name: _write(getattr(state, name), address, value, admitted) for name, value in (
        ("hp_numerator", numerator), ("hp_denominator", denominator), ("status", status),
        ("fainted", fainted), ("hp_known", True))}), admitted


def status_update(state: MonState, *, side, slot, encounter, status, valid=True):
    """Support an explicitly resolved bench cure without touching the active mon."""
    row, seat, target, address_ok = _address(state, side, slot)
    encounter, status = _int32(encounter), _int32(status)
    admitted = address_ok & (encounter > 0) & (state.encounter[row, seat, target] == encounter) & (
        status >= NONE) & (status <= TOXIC) & jnp.asarray(valid)
    return state._replace(status=_write(state.status, (row, seat, target), status, admitted)), admitted


def faint(state: MonState, *, side, slot, encounter, valid=True):
    row, seat, target, address_ok = _address(state, side, slot)
    encounter = _int32(encounter)
    admitted = address_ok & (encounter > 0) & (state.encounter[row, seat, target] == encounter) & jnp.asarray(valid)
    address = row, seat, target
    return state._replace(hp_numerator=_write(state.hp_numerator, address, 0, admitted),
        hp_denominator=_write(state.hp_denominator, address, 0, admitted),
        hp_known=_write(state.hp_known, address, True, admitted),
        status=_write(state.status, address, NONE, admitted),
        fainted=_write(state.fainted, address, True, admitted)), admitted


def boost_update(state: MonState, *, side, stat, amount=0, operation=DELTA, valid=True):
    """Apply active public stage changes, with the source's [-6,6] cap."""
    side, stat, amount, operation = (_int32(x) for x in (side, stat, amount, operation))
    row = jnp.arange(state.active.shape[0])
    seat, index = jnp.clip(side, 0, 1), jnp.clip(stat, 0, len(STATS) - 1)
    admitted = (side >= 0) & (side < 2) & (stat >= 0) & (stat < len(STATS)) & (
        operation >= SET) & (operation <= CLEAR_NEGATIVE) & (
        state.active[row, seat] >= 0) & jnp.asarray(valid)
    before = state.boosts[row, seat]
    old_stat = before[jnp.arange(before.shape[0]), index]
    # Bound the addend first so int32 transport extremes cannot wrap before
    # the same final cap used by Python's unbounded integer arithmetic.
    new_stat = jnp.clip(jnp.where(operation == SET, jnp.clip(amount, -6, 6),
                                  old_stat + jnp.clip(amount, -12, 12)), -6, 6)
    changed = before.at[jnp.arange(before.shape[0]), index].set(new_stat)
    changed = jnp.where((operation == CLEAR)[..., None], 0, changed)
    changed = jnp.where((operation == CLEAR_POSITIVE)[..., None], jnp.minimum(before, 0), changed)
    changed = jnp.where((operation == CLEAR_NEGATIVE)[..., None], jnp.maximum(before, 0), changed)
    return state._replace(boosts=_write(state.boosts, (row, seat), changed, admitted)), admitted


def copy_boosts(state: MonState, *, source_side, target_side, valid=True):
    """Psych Up/Transform copy public stages, preserving both identities."""
    source, target = _int32(source_side), _int32(target_side)
    row = jnp.arange(state.active.shape[0])
    source_safe, target_safe = jnp.clip(source, 0, 1), jnp.clip(target, 0, 1)
    admitted = (source >= 0) & (source < 2) & (target >= 0) & (target < 2) & (
        state.active[row, source_safe] >= 0) & (state.active[row, target_safe] >= 0) & jnp.asarray(valid)
    return state._replace(boosts=_write(state.boosts, (row, source_safe),
        state.boosts[row, target_safe], admitted)), admitted


def clear_all_boosts(state: MonState, *, valid=True):
    """Haze's public clear applies to both sides, including empty active slots."""
    valid=jnp.asarray(valid)
    if valid.dtype!=jnp.bool_:raise ValueError("global boost validity must be boolean")
    admitted=jnp.broadcast_to(valid,(state.active.shape[0],))
    return state._replace(boosts=jnp.where(admitted[:,None,None],0,state.boosts)),admitted


def cure_team(state: MonState, *, side, valid=True):
    """Clear named nonvolatile status on known living members of one side.

    Fainted conditions remain fainted. This is the status-field consumer only;
    sleep/Rest clocks, sleep-clause ownership and cure inference remain separate.
    """
    side=_int32(side);valid=jnp.asarray(valid)
    if valid.dtype!=jnp.bool_:raise ValueError("team cure validity must be boolean")
    batch=state.active.shape[0];side=jnp.broadcast_to(side,(batch,));row=jnp.arange(batch)
    seat=jnp.clip(side,0,1);admitted=(side>=0)&(side<2)&valid
    known_living=(state.encounter[row,seat]>0)&~state.fainted[row,seat]
    updated=jnp.where(known_living,0,state.status[row,seat])
    return state._replace(status=_write(state.status,(row,seat),updated,admitted)),admitted
