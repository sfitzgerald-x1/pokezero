"""Gen 3 Sleep Talk selection and sleep-clock bookkeeping, not all callers."""
from typing import NamedTuple

import jax
import jax.numpy as jnp

from . import mechanics as m


class SleepTalkTable(NamedTuple):
    move: jax.Array
    callable: jax.Array
    early_bird: jax.Array


def from_metadata(metadata):
    vocab=metadata["vocabulary"];names=vocab["moves"];rows=metadata["moves"]
    if len(rows)!=len(names):raise ValueError("move metadata/vocabulary mismatch")
    allowed=[]
    for name,row in zip(names,rows):
        flags={} if row is None else row.get("flags",{})
        allowed.append(name not in {"_none","sleeptalk","struggle"}
            and not flags.get("nosleeptalk") and not flags.get("charge"))
    return SleepTalkTable(jnp.asarray([name=="sleeptalk" for name in names]),
        jnp.asarray(allowed),jnp.asarray([name=="earlybird" for name in vocab["abilities"]]))


def choose(table,moves,pp,roll):
    """Choose before testing PP; zero-PP selection fails without another draw.

    A four-slot moveset containing Sleep Talk has at most three callable slots.
    Counts 1..3 divide 60; the fixed [0,60) rank roll adds no modulo bias.
    Disabled/Choice-locked called slots are not excluded from the source pool.
    """
    eligible=table.callable[moves]
    count=jnp.sum(eligible,axis=1,dtype=jnp.int32)
    ranks=jnp.cumsum(eligible,axis=1)-1
    slot=jnp.argmax(eligible & (ranks==(roll%jnp.maximum(1,count))[:,None]),axis=1)
    chosen=jnp.take_along_axis(moves,slot[:,None],axis=1)[:,0]
    has_pp=jnp.take_along_axis(pp,slot[:,None],axis=1)[:,0]>0
    return chosen,(count>0)&has_pp


def before_move(state,side,before_status,*,eligible,sleep_usable):
    """The status handler runs before flinch/confusion, even if they then stop use."""
    rows=jnp.arange(state.hp.shape[0]);active=state.active[rows,side]
    old=state.sleep_skipped[rows,side,active]
    sleeping=(state.status[rows,side,active]==m.SLP) & (before_status==m.SLP)
    skipped=jnp.where(sleeping & sleep_usable,old+1,0)
    return state._replace(sleep_skipped=state.sleep_skipped.at[rows,side,active].set(
        jnp.where(eligible,skipped,old)))
