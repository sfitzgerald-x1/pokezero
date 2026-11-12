"""Selected singles Leech Seed and Wish events, not wider healing fidelity.

The seed's source slot is always the opposite singles slot; its occupant can
switch without losing the recipient. Baton Pass and Liquid Ooze remain outside
this increment. Wish belongs to a side slot, not its original party member.
"""
from typing import NamedTuple

import jax
import jax.numpy as jnp

from .state import LEECH_SEED


class SustainTable(NamedTuple):
    seed: jax.Array
    wish: jax.Array
    grass_type: jax.Array
    spin: jax.Array


def from_metadata(metadata):
    vocab = metadata["vocabulary"]
    return SustainTable(jnp.asarray([name == "leechseed" for name in vocab["moves"]]),
        jnp.asarray([name == "wish" for name in vocab["moves"]]),
        jnp.asarray(vocab["types"].index("grass")),
        jnp.asarray([name=="rapidspin" for name in vocab["moves"]]))


def spin_cleanup(state,side,*,eligible):
    """Clear represented seed/Spikes and the spinner's partial trap (Wrap), never Mean Look/ability/Ingrain traps.

    Do not erase anonymous external or linked trapping flags to pretend they are Bind.
    """
    rows=jnp.arange(state.hp.shape[0]);active=state.active[rows,side]
    applies=jnp.asarray(eligible) & (state.hp[rows,side,active]>0)
    old=state.volatile[rows,side,active]
    state=state._replace(volatile=state.volatile.at[rows,side,active].set(
        jnp.where(applies,old & ~jnp.uint32(LEECH_SEED),old)),
        spikes=state.spikes.at[rows,side].set(jnp.where(applies,0,state.spikes[rows,side])))
    # Fidelity M2: Rapid Spin also ends the spinner's partial trap (data/mods/gen4/moves.ts:1096-1098 self onHit;
    # data/moves.ts onAfterHit/onAfterSubDamage), a Substitute hit included: removeVolatile prints -end ...
    # [partiallytrapped]. States without the leaves (fixtures) skip it.
    if hasattr(state,'partial_trap_counter'):
        state=state._replace(partial_trap_counter=state.partial_trap_counter.at[rows,side,active].set(
            jnp.where(applies,0,state.partial_trap_counter[rows,side,active])),
            partial_trap_source=state.partial_trap_source.at[rows,side,active].set(
                jnp.where(applies,-1,state.partial_trap_source[rows,side,active])))
    return state


def seed_target(table,move,types,hp,sub_hp,volatile,*,eligible):
    applies = (jnp.asarray(eligible) & table.seed[move] & (hp > 0) & (sub_hp == 0)
        & ~jnp.any(types == table.grass_type,axis=-1))
    return volatile | jnp.where(applies,jnp.uint32(LEECH_SEED),jnp.uint32(0))


class LeechResult(NamedTuple):
    hp: jax.Array
    recipient_hp: jax.Array


def leech(hp,max_hp,volatile,recipient_hp,recipient_max_hp,*,eligible):
    active = (jnp.asarray(eligible) & (hp > 0) & (recipient_hp > 0)
        & ((volatile & LEECH_SEED) != 0))
    amount = jnp.where(active,jnp.minimum(hp,jnp.maximum(1,max_hp//8)),0)
    # Drain and healing finish within one handler, before faintMessages/checkWin.
    return LeechResult(hp-amount,jnp.where(active,
        jnp.minimum(recipient_max_hp,recipient_hp+amount),recipient_hp))


class WishResult(NamedTuple):
    hp: jax.Array
    counter: jax.Array


def wish_residual(hp,max_hp,counter,*,eligible):
    ticking = jnp.asarray(eligible) & (counter > 0)
    expires = ticking & (counter == 1)
    healed = jnp.minimum(max_hp,hp+jnp.maximum(1,max_hp//2))
    return WishResult(jnp.where(expires & (hp > 0),healed,hp),
        jnp.where(ticking,counter-1,counter))
