"""Gen 3 Encore's queued override and residual clock, not general move locks."""
from typing import NamedTuple

import jax
import jax.numpy as jnp


class EncoreTable(NamedTuple):
    move: jax.Array
    allowed_last: jax.Array


def from_metadata(metadata):
    names=metadata["vocabulary"]["moves"];rows=metadata["moves"]
    if len(names)!=len(rows):raise ValueError("move metadata/vocabulary mismatch")
    return EncoreTable(jnp.asarray([name=="encore" for name in names]),
        jnp.asarray([name not in {"_none","struggle","encore"}
            and not ({} if row is None else row.get("flags",{})).get("failencore")
            for name,row in zip(names,rows)]))


def override(state,side,move,slot,*,struggle):
    """runMove OverrideAction: preserve original queue priority, change PP slot."""
    rows=jnp.arange(state.hp.shape[0]);active=state.active[rows,side]
    forced=state.encore_move[rows,side,active]
    matches=state.moves[rows,side,active]==forced[:,None]
    changed=(state.encore_counter[rows,side,active]>0) & ~struggle & jnp.any(matches,axis=1)
    return jnp.where(changed,forced,move),jnp.where(changed,jnp.argmax(matches,axis=1),slot)


def apply(state,side,move,roll,table,*,eligible,queued):
    """Target last normal caller, not Sleep Talk's called move; do not refresh.

    Gen3 chooses [3,7), adding one if no target move remains queued. Slot PP is
    checked at application, and Encore itself/failencore moves are rejected.
    """
    rows=jnp.arange(state.hp.shape[0]);active=state.active[rows,side]
    last=state.last_move[rows,side,active]
    matches=state.moves[rows,side,active]==last[:,None]
    slot=jnp.argmax(matches,axis=1)
    current=state.encore_counter[rows,side,active]
    starts=(eligible & table.move[move] & (state.hp[rows,side,active]>0)
        & (current==0) & table.allowed_last[last] & jnp.any(matches,axis=1)
        & (state.pp[rows,side,active,slot]>0))
    duration=3+roll+(~jnp.asarray(queued)).astype(jnp.int32)
    return state._replace(encore_move=state.encore_move.at[rows,side,active].set(
        jnp.where(starts,last,state.encore_move[rows,side,active])),
        encore_counter=state.encore_counter.at[rows,side,active].set(jnp.where(starts,duration,current)))


def residual(state,side,*,eligible):
    rows=jnp.arange(state.hp.shape[0]);active=state.active[rows,side]
    move=state.encore_move[rows,side,active]
    matches=state.moves[rows,side,active]==move[:,None]
    slot=jnp.argmax(matches,axis=1)
    current=state.encore_counter[rows,side,active]
    remains=jnp.maximum(0,current-1)
    valid=jnp.any(matches,axis=1) & (state.pp[rows,side,active,slot]>0)
    remaining=jnp.where(eligible & (state.hp[rows,side,active]>0),jnp.where(valid,remains,0),current)
    return state._replace(encore_counter=state.encore_counter.at[rows,side,active].set(remaining),
        encore_move=state.encore_move.at[rows,side,active].set(jnp.where(remaining>0,move,0)))
