"""Direct Gen 3 Roar/Whirlwind subset; not full move/event qualification."""
from typing import NamedTuple

import jax
import jax.numpy as jnp

from . import entry
from .state import ROOTED


class PhazingTable(NamedTuple):
    move: jax.Array
    roar: jax.Array
    soundproof: jax.Array
    suction_cups: jax.Array


def from_metadata(metadata):
    moves=metadata["vocabulary"]["moves"];abilities=metadata["vocabulary"]["abilities"]
    return PhazingTable(jnp.asarray([x in {"roar","whirlwind"} for x in moves]),
        jnp.asarray([x=="roar" for x in moves]),jnp.asarray([x=="soundproof" for x in abilities]),
        jnp.asarray([x=="suctioncups" for x in abilities]))


def choose_bench(hp,active,roll):
    """One uniform [0,60) roll selects uniformly among 1..5 living benches.

    60 is divisible by every possible bench count. No biased modulo of a
    percentile roll, rejection loop or data-dependent host branch is needed.
    """
    candidates=(hp>0) & (jnp.arange(6)[None,:]!=active[:,None])
    count=jnp.sum(candidates,axis=1,dtype=jnp.int32)
    rank=jnp.cumsum(candidates,axis=1)-1
    chosen=jnp.argmax(candidates & (rank==(roll % jnp.maximum(1,count))[:,None]),axis=1)
    return chosen.astype(jnp.int32),count>0


def drag(state,side,move,roll,table,entry_table,*,eligible):
    """Install a random living bench, clear outgoing state, then run entry.

    The caller cancels the displaced actor's queued action. Hazard KO is
    returned separately because Gen 3 cancels all remaining moves after a KO.
    """
    rows=jnp.arange(state.hp.shape[0]);old=state.active[rows,side]
    target,exists=choose_bench(state.hp[rows,side],old,roll)
    ability=state.ability[rows,side,old]
    blocked=table.suction_cups[ability] | (table.roar[move] & table.soundproof[ability])
    blocked |= (state.volatile[rows,side,old] & jnp.uint32(ROOTED))!=0
    changed=eligible & table.move[move] & exists & ~blocked & (state.hp[rows,side,old]>0)
    state=entry.depart(state,side,entry_table,eligible=changed)
    state=state._replace(active=state.active.at[rows,side].set(jnp.where(changed,target,old)),
        active_turns=state.active_turns.at[rows,side].set(jnp.where(changed,0,state.active_turns[rows,side])),
        trapped=state.trapped.at[rows,side].set(state.trapped[rows,side] & ~changed),
        baton_pass=state.baton_pass.at[rows,side].set(state.baton_pass[rows,side] & ~changed))
    knocked_out=jnp.zeros_like(changed)
    if entry_table is not None:
        state,knocked_out=entry.enter(state,side,entry_table,eligible=changed)
    return state,changed,knocked_out
