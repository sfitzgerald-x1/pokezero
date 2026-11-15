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


def _rows(state):
    return jnp.arange(state.hp.shape[0])


def apply(state, side, eligible, table):
    """The executed Transform of `side`'s active (eligible: it acts and the dispatched move is Transform; Sleep Talk's
    called Transform included). Returns (state, copied): copied is the successful copy (the foe active alive and not
    itself transformed). Every executed Transform sets the side's action-history slot to -1 (kind stays MOVE): Showdown's
    lastMove is `transform`, whose failencore makes a later Encore fail (data/moves.ts encore), while the engine keys
    Encore on the slot, which now names a copied move (encore.apply, encore.override, chance_consumers' closure all read
    "no encorable last move"). last_move (the caller identity) stays transform. A user already transformed keeps its
    saves: a re-transform (a Ditto that copied a Mew with Transform) must restore the true base."""
    rows = _rows(state)
    side = jnp.broadcast_to(jnp.asarray(side, jnp.int32), rows.shape)
    foe = 1 - side
    a = state.active[rows, side]
    f = state.active[rows, foe]
    eligible = jnp.broadcast_to(jnp.asarray(eligible), rows.shape)
    copied = eligible & ~state.transformed[rows, foe] & (state.hp[rows, foe, f] > 0) & (state.hp[rows, side, a] > 0)
    first = copied & ~state.transformed[rows, side]
    def put(leaf, update, gate):
        old = leaf[rows, side]
        return leaf.at[rows, side].set(jnp.where(gate.reshape(gate.shape + (1,) * (old.ndim - 1)), update, old))
    def put_active(leaf, update, gate):
        old = leaf[rows, side, a]
        return leaf.at[rows, side, a].set(jnp.where(gate.reshape(gate.shape + (1,) * (old.ndim - 1)), update, old))
    saved_hp = jnp.stack((state.hidden_power_type[rows, side, a], state.hidden_power_power[rows, side, a]), axis=-1)
    state = state._replace(
        transform_stats=put(state.transform_stats, state.stats[rows, side, a, 1:], first),
        transform_moves=put(state.transform_moves, state.moves[rows, side, a], first),
        transform_pp=put(state.transform_pp, state.pp[rows, side, a], first),
        transform_hidden_power=put(state.transform_hidden_power, saved_hp, first))
    donor_moves = state.moves[rows, foe, f]
    stats = state.stats[rows, side, a].at[:, 1:].set(state.stats[rows, foe, f, 1:])
    state = state._replace(
        stats=put_active(state.stats, stats, copied),
        types=put_active(state.types, state.types[rows, foe, f], copied),
        moves=put_active(state.moves, donor_moves, copied),
        pp=put_active(state.pp, table.copied_pp[donor_moves], copied),
        disabled=put_active(state.disabled, jnp.zeros_like(state.disabled[rows, side, a]), copied),
        ability=put_active(state.ability, state.ability[rows, foe, f], copied),
        hidden_power_type=put_active(state.hidden_power_type, state.hidden_power_type[rows, foe, f], copied),
        hidden_power_power=put_active(state.hidden_power_power, state.hidden_power_power[rows, foe, f], copied),
        boosts=put_active(state.boosts, state.boosts[rows, foe, f], copied),
        transformed=put(state.transformed, jnp.ones_like(copied), copied),
        transform_species=put(state.transform_species, state.species[rows, foe, f], copied),
        last_action_index=state.last_action_index.at[rows, side].set(
            jnp.where(eligible, -1, state.last_action_index[rows, side])))
    return state, copied


def revert(state, side, eligible):
    """clearVolatile's revert of `side`'s current active where it is transformed: Atk..Spe, moves, PP (the base PP as
    last paid), Hidden Power, ability <- base_ability, types <- base_types; nothing disabled; the flag and saves clear.
    Stages are the caller's (departure and the faint clear zero them). Returns (state, reverted)."""
    rows = _rows(state)
    side = jnp.broadcast_to(jnp.asarray(side, jnp.int32), rows.shape)
    a = state.active[rows, side]
    use = jnp.broadcast_to(jnp.asarray(eligible), rows.shape) & state.transformed[rows, side]
    def put_active(leaf, update):
        old = leaf[rows, side, a]
        return leaf.at[rows, side, a].set(jnp.where(use.reshape(use.shape + (1,) * (old.ndim - 1)), update, old))
    def clear(leaf):
        old = leaf[rows, side]
        return leaf.at[rows, side].set(jnp.where(use.reshape(use.shape + (1,) * (old.ndim - 1)), jnp.zeros_like(old), old))
    stats = state.stats[rows, side, a].at[:, 1:].set(state.transform_stats[rows, side])
    saved_hp = state.transform_hidden_power[rows, side]
    state = state._replace(
        stats=put_active(state.stats, stats),
        moves=put_active(state.moves, state.transform_moves[rows, side]),
        pp=put_active(state.pp, state.transform_pp[rows, side]),
        disabled=put_active(state.disabled, jnp.zeros_like(state.disabled[rows, side, a])),
        hidden_power_type=put_active(state.hidden_power_type, saved_hp[:, 0]),
        hidden_power_power=put_active(state.hidden_power_power, saved_hp[:, 1]),
        ability=put_active(state.ability, state.base_ability[rows, side, a]),
        types=put_active(state.types, state.base_types[rows, side, a]))
    state = state._replace(transformed=clear(state.transformed), transform_species=clear(state.transform_species),
        transform_stats=clear(state.transform_stats), transform_moves=clear(state.transform_moves),
        transform_pp=clear(state.transform_pp), transform_hidden_power=clear(state.transform_hidden_power))
    return state, use


def faint_revert(state, fainted):
    """faintMessages -> clearVolatile(false): revert each side whose active is in `fainted` ([B,2,6] newly fainted)."""
    rows = _rows(state)
    for side in range(2):
        state, _ = revert(state, side, fainted[rows, side, state.active[:, side]])
    return state


def update_cure(state, side, eligible, entry_table):
    """Gen < 5 runs eachEvent('Update') after every action (sim/battle.ts:2938). Right after a successful Transform
    (eligible) a copied Immunity, Limber, Insomnia, Vital Spirit, Water Veil or Magma Armor cures the matching status
    (-activate, cureStatus), and a copied Own Tempo ends confusion (data/abilities.ts:2051, 2322, 3098; the same tables as
    entry.enter's). No other path gives an active a curing ability it did not hold at entry. Returns (state, cured,
    unconfused)."""
    rows = _rows(state)
    side = jnp.broadcast_to(jnp.asarray(side, jnp.int32), rows.shape)
    a = state.active[rows, side]
    eligible = jnp.broadcast_to(jnp.asarray(eligible), rows.shape) & (state.hp[rows, side, a] > 0)
    ability = state.ability[rows, side, a]
    status = state.status[rows, side, a]
    cured = jnp.zeros_like(eligible)
    unconfused = jnp.zeros_like(eligible)
    if entry_table is not None and entry_table.status_cure is not None:
        cured = eligible & (status != m.NONE) & entry_table.status_cure[ability, status]
    if entry_table is not None and entry_table.own_tempo is not None:
        unconfused = eligible & entry_table.own_tempo[ability] & (state.confusion_counter[rows, side, a] > 0)
    def put(leaf, value, gate):
        return leaf.at[rows, side, a].set(jnp.where(gate, value, leaf[rows, side, a]))
    state = state._replace(status=put(state.status, m.NONE, cured), status_counter=put(state.status_counter, 0, cured),
        sleep_skipped=put(state.sleep_skipped, 0, cured), rest_origin=put(state.rest_origin, False, cured),
        confusion_counter=put(state.confusion_counter, 0, unconfused))
    return state, cured, unconfused


def choice_unlock(state, eligible):
    """endTurn's DisableMove: choicelock's onDisableMove drops a lock whose move the holder no longer has
    (data/conditions.ts:350, `!pokemon.hasMove(this.effectState.move)`). Only a transformed active can hold such a lock
    (a Choice holder's Transform locks on `transform`, then holds the copied moves; it keeps the lock when the donor
    itself knows Transform), so this is gated on the flag and leaves every other lane bitwise."""
    rows = _rows(state)
    eligible = jnp.broadcast_to(jnp.asarray(eligible), rows.shape)
    for side in range(2):
        a = state.active[:, side]
        lock = state.choice_move[rows, side, a]
        known = jnp.any(state.moves[rows, side, a] == lock[:, None], axis=-1)
        drop = eligible & state.transformed[:, side] & (lock > 0) & ~known
        state = state._replace(choice_move=state.choice_move.at[rows, side, a].set(jnp.where(drop, 0, lock)))
    return state
