"""Gen 3 Transform (Showdown f76228a1): the copy, its revert and the three follow-ups.

The move (data/moves.ts transform; data/mods/gen4/moves.ts:1443-1446, inherited by Gen 3): accuracy true, no protect
flag, bypasssub, so neither an accuracy roll, Protect nor a Substitute stops it, and it makes no PRNG call.
transformInto (sim/pokemon.ts:1271-1384) fails on a target that is already transformed (gen >= 2); in Gen 3 it does not
fail on a target's Substitute or on a user that is itself transformed (both gen >= 5). It copies, in place: Atk..Spe
(storedStats; HP and max HP stay), the current types (a Color Change result included), the moves with pp
min(5, move.pp) and nothing disabled, all seven stages, the Hidden Power type and power (gen < 5) and the ability
(setAbility(..., isTransform), whose Start runs only for gen > 3, so a copied Intimidate, Drizzle or Drought does
nothing). `species` keeps the original: every engine reader of species wants it (item gates, identity, labels); the
donor's original species is kept in transform_species. clearVolatile (:1509-1561) reverts at switch-out (a drag and a
Baton Pass included) and at a faint (faintMessages -> clearVolatile(false)): moveSlots <- baseMoveSlots (the base PP
as last paid, Transform's own cost included), ability <- baseAbility, Hidden Power <- base, setSpecies(baseSpecies)
(types and stats). Stages are zeroed there by the existing departure and faint code.

Only an active can be transformed, so the saves the revert needs are per side (state.transform_*).
"""
from typing import NamedTuple

import jax
import jax.numpy as jnp

from . import mechanics as m


class TransformTable(NamedTuple):
    move: jax.Array       # [M] bool, the vocabulary's transform
    copied_pp: jax.Array  # [M] int32, a copied slot's pp: min(5, move.pp) (5 for every Gen 3 move), 0 for the empty id


def from_metadata(metadata):
    names = metadata["vocabulary"]["moves"]
    rows = metadata.get("moves") or [None] * len(names)
    if len(rows) != len(names):
        raise ValueError("move metadata/vocabulary mismatch")
    pp = [0 if i == 0 or name == "_none" else min(5, int((row or {}).get("pp", 5)))
          for i, (name, row) in enumerate(zip(names, rows))]
    return TransformTable(jnp.asarray([name == "transform" for name in names]), jnp.asarray(pp, jnp.int32))


def _sides(state, side, eligible):
    """[B,2] the given side (a scalar or one per row) where `eligible` ([B] or scalar)."""
    b = state.hp.shape[0]
    side = jnp.broadcast_to(jnp.asarray(side, jnp.int32), (b,))
    eligible = jnp.broadcast_to(jnp.asarray(eligible), (b,))
    return (jnp.arange(2)[None, :] == side[:, None]) & eligible[:, None]


def _at_active(state, sides):
    """[B,2,6] the active member of each side in `sides` ([B,2])."""
    return sides[:, :, None] & (jnp.arange(6)[None, None, :] == state.active[:, :, None])


def _cells(state, x):
    """[B,2,...] each side's active cell of a [B,2,6,...] leaf."""
    index = state.active.reshape(state.active.shape + (1,) * (x.ndim - 2))
    return jnp.take_along_axis(x, index, axis=2)[:, :, 0]


def _put(mask, value, leaf):
    """`leaf` with `value` (broadcast over the six members) written where `mask` ([B,2,6]) holds."""
    return jnp.where(mask.reshape(mask.shape + (1,) * (leaf.ndim - 3)), jnp.expand_dims(value, 2), leaf)


def _keep(mask, value, leaf):
    """A per-side leaf ([B,2,...]) set to `value` where `mask` ([B,2]) holds."""
    return jnp.where(mask.reshape(mask.shape + (1,) * (leaf.ndim - 2)), value, leaf)


def apply(state, side, eligible, table):
    """The executed Transform of `side`'s active (eligible: it acts and the dispatched move is Transform; Sleep Talk's
    called Transform included). Returns (state, copied): copied is the successful copy (the foe active alive and not
    itself transformed). Every executed Transform sets the side's action-history slot to -1 (kind stays MOVE): Showdown's
    lastMove is `transform`, whose failencore makes a later Encore fail (data/moves.ts encore), while the engine keys
    Encore on the slot, which now names a copied move (encore.apply, encore.override, chance_consumers' closure all read
    "no encorable last move"). last_move (the caller identity) stays transform. A user already transformed keeps its
    saves: a re-transform (a Ditto that copied a Mew with Transform) must restore the true base. (A Sleep Talk call of
    Transform also clears the slot, while Showdown's lastMove is then Sleep Talk; no pool set holds both moves, so a
    later Encore of a copied Sleep Talk is unreachable.)

    Written as whole-leaf selects over side and member masks (no per-row scatter), so its traced size stays small."""
    user = _sides(state, side, eligible)                            # [B,2]
    alive = _cells(state, state.hp) > 0
    copied = user & alive & alive[:, ::-1] & ~state.transformed[:, ::-1]
    first = copied & ~state.transformed
    donor = lambda x: _cells(state, x)[:, ::-1]                     # each side's foe active cell
    member = _at_active(state, copied)
    own_cells = lambda x: _cells(state, x)
    saved_hp = jnp.stack((own_cells(state.hidden_power_type), own_cells(state.hidden_power_power)), axis=-1)
    donor_moves = donor(state.moves)
    copied_stats = jnp.concatenate((own_cells(state.stats)[..., :1], donor(state.stats)[..., 1:]), axis=-1)
    state = state._replace(
        transform_stats=_keep(first, own_cells(state.stats)[..., 1:], state.transform_stats),
        transform_moves=_keep(first, own_cells(state.moves), state.transform_moves),
        transform_pp=_keep(first, own_cells(state.pp), state.transform_pp),
        transform_hidden_power=_keep(first, saved_hp, state.transform_hidden_power),
        stats=_put(member, copied_stats, state.stats),
        types=_put(member, donor(state.types), state.types),
        moves=_put(member, donor_moves, state.moves),
        pp=_put(member, table.copied_pp[donor_moves], state.pp),
        disabled=_put(member, jnp.zeros_like(donor(state.disabled)), state.disabled),
        ability=_put(member, donor(state.ability), state.ability),
        hidden_power_type=_put(member, donor(state.hidden_power_type), state.hidden_power_type),
        hidden_power_power=_put(member, donor(state.hidden_power_power), state.hidden_power_power),
        boosts=_put(member, donor(state.boosts), state.boosts),
        transformed=state.transformed | copied,
        transform_species=_keep(copied, donor(state.species), state.transform_species),
        last_action_index=_keep(user, jnp.full_like(state.last_action_index, -1), state.last_action_index))
    return state, jnp.any(copied, axis=1)


def _revert(state, use):
    """clearVolatile's revert of each side's active where `use` ([B,2], already gated on the flag)."""
    member = _at_active(state, use)
    saved = state.transform_hidden_power
    stats = jnp.pad(state.transform_stats, ((0, 0), (0, 0), (1, 0)))[:, :, None, :]      # HP column untouched below
    state = state._replace(
        stats=jnp.where(member[..., None] & (jnp.arange(6) >= 1), stats, state.stats),
        moves=_put(member, state.transform_moves, state.moves),
        pp=_put(member, state.transform_pp, state.pp),
        disabled=jnp.where(member[..., None], False, state.disabled),
        hidden_power_type=_put(member, saved[..., 0], state.hidden_power_type),
        hidden_power_power=_put(member, saved[..., 1], state.hidden_power_power),
        ability=jnp.where(member, state.base_ability, state.ability),
        types=jnp.where(member[..., None], state.base_types, state.types))
    clear = lambda leaf: _keep(use, jnp.zeros_like(leaf), leaf)
    return state._replace(transformed=state.transformed & ~use, transform_species=clear(state.transform_species),
        transform_stats=clear(state.transform_stats), transform_moves=clear(state.transform_moves),
        transform_pp=clear(state.transform_pp), transform_hidden_power=clear(state.transform_hidden_power))


def revert(state, side, eligible):
    """clearVolatile's revert of `side`'s current active where it is transformed: Atk..Spe, moves, PP (the base PP as
    last paid), Hidden Power, ability <- base_ability, types <- base_types; nothing disabled; the flag and saves clear.
    Stages are the caller's (departure and the faint clear zero them). Returns (state, reverted [B])."""
    use = _sides(state, side, eligible) & state.transformed
    return _revert(state, use), jnp.any(use, axis=1)


def faint_revert(state):
    """faintMessages -> clearVolatile(false): revert each side whose transformed active has fainted.

    transition calls it once, at the step's end with the other faint clears. Nothing in a step reads a fainted active's
    copied stats, moves, PP, ability, types or Hidden Power after its faint (it never acts again; the residual handlers,
    Pressure, Trace, Intimidate and weather suppression skip a fainted holder; the residual order was frozen before the
    residuals), and a fainted active only leaves the field at a later step's replacement, so one revert at the step's end
    gives every step boundary Showdown's faint-time state, the next replacement's outgoing Speed included."""
    dead = jnp.any((state.hp <= 0) & (jnp.arange(6)[None, None, :] == state.active[:, :, None]), axis=-1)
    return _revert(state, state.transformed & dead)


def update_cure(state, side, eligible, entry_table):
    """Gen < 5 runs eachEvent('Update') after every action (sim/battle.ts:2938). Right after a successful Transform
    (eligible) a copied Immunity, Limber, Insomnia, Vital Spirit, Water Veil or Magma Armor cures the matching status
    (-activate, cureStatus), and a copied Own Tempo ends confusion (data/abilities.ts:2051, 2322, 3098; the same tables as
    entry.enter's). No other path gives an active a curing ability it did not hold at entry. Returns (state, cured,
    unconfused), each [B]."""
    sides = _sides(state, side, eligible) & (_cells(state, state.hp) > 0)
    ability = _cells(state, state.ability)
    status = _cells(state, state.status)
    cured = jnp.zeros_like(sides)
    unconfused = jnp.zeros_like(sides)
    if entry_table is not None and entry_table.status_cure is not None:
        cured = sides & (status != m.NONE) & entry_table.status_cure[ability, status]
    if entry_table is not None and entry_table.own_tempo is not None:
        unconfused = sides & entry_table.own_tempo[ability] & (_cells(state, state.confusion_counter) != 0)
    cure, calm = _at_active(state, cured), _at_active(state, unconfused)
    state = state._replace(status=jnp.where(cure, m.NONE, state.status), status_counter=jnp.where(cure, 0, state.status_counter),
        sleep_skipped=jnp.where(cure, 0, state.sleep_skipped), rest_origin=jnp.where(cure, False, state.rest_origin),
        confusion_counter=jnp.where(calm, 0, state.confusion_counter))
    return state, jnp.any(cured, axis=1), jnp.any(unconfused, axis=1)


def choice_unlock(state, eligible):
    """endTurn's DisableMove: choicelock's onDisableMove drops a lock whose move the holder no longer has
    (data/conditions.ts:350, `!pokemon.hasMove(this.effectState.move)`). Only a transformed active can hold such a lock
    (a Choice holder's Transform locks on `transform`, then holds the copied moves; it keeps the lock when the donor
    itself knows Transform), so this is gated on the flag and leaves every other lane bitwise."""
    lock = _cells(state, state.choice_move)
    known = jnp.any(_cells(state, state.moves) == lock[..., None], axis=-1)
    eligible = jnp.broadcast_to(jnp.asarray(eligible), (state.hp.shape[0],))
    drop = eligible[:, None] & state.transformed & (lock > 0) & ~known
    return state._replace(choice_move=jnp.where(_at_active(state, drop), 0, state.choice_move))
