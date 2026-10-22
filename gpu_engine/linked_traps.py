"""Selected direct Gen 3 move traps, distinct from ability/partial traps."""
from typing import NamedTuple

import jax
import jax.numpy as jnp


class TrapTable(NamedTuple):
    move: jax.Array


def from_metadata(metadata):
    return TrapTable(jnp.asarray([name in {"meanlook","spiderweb","block"}
        for name in metadata["vocabulary"]["moves"]]))


def apply(state,side,move,table,*,eligible):
    rows=jnp.arange(state.hp.shape[0]);foe=1-side
    actor=state.active[rows,side];target=state.active[rows,foe]
    # All three are blocked by Substitute in the pinned Gen 3 move surface.
    hit=(eligible & table.move[move] & (state.hp[rows,side,actor]>0)
        & (state.hp[rows,foe,target]>0) & (state.substitute_hp[rows,foe,target]==0))
    old=state.trap_source[rows,foe,target]
    return state._replace(trap_source=state.trap_source.at[rows,foe,target].set(
        jnp.where(hit,actor,old)))


def prune(state):
    """Faint cleanup releases linked volatiles before a replacement request.

    Ordinary departure handles live switches immediately. This final mask also
    clears dead target/owner links and cannot reactivate a stale party identity.
    """
    source=state.trap_source
    owner_hp=jnp.take_along_axis(state.hp[:,::-1],jnp.clip(source,0,5),axis=2)
    active_target=jnp.arange(6)[None,None,:]==state.active[...,None]
    valid=((source>=0) & (source==state.active[:,::-1,None]) & (owner_hp>0)
        & (state.hp>0) & active_target)
    return state._replace(trap_source=jnp.where(valid,source,-1))
