"""Selected Gen3 Focus Punch before-turn preparation and TryMove gate."""
from typing import NamedTuple

import jax
import jax.numpy as jnp

from .state import FOCUS_PUNCH,FOCUS_LOST


class FocusTable(NamedTuple):
    move: jax.Array
    damaging: jax.Array


def from_metadata(metadata):
    names=metadata['vocabulary']['moves'];rows=metadata['moves']
    if len(names)!=len(rows):raise ValueError('move metadata/vocabulary mismatch')
    return FocusTable(jnp.asarray([name=='focuspunch' for name in names]),
        jnp.asarray([bool(row and row.get('category','Status')!='Status') for row in rows]))


def prepare(state,moves,table,*,ordinary,move_requests):
    """Prepare original queued move before switches/OverrideAction/BeforeMove.

    An interrupted Baton request retains its original preparation; a new
    ordinary turn discards any old single-turn flags before preparing actives.
    """
    rows=jnp.arange(state.hp.shape[0])[:,None];sides=jnp.arange(2)[None,:]
    clear=jnp.uint32(~(FOCUS_PUNCH|FOCUS_LOST)&0xffffffff)
    bits=jnp.where(ordinary[:,None,None],state.volatile & clear,state.volatile)
    active=state.active;old=bits[rows,sides,active]
    charge=(ordinary[:,None] & move_requests & table.move[moves] & (state.hp[rows,sides,active]>0))
    return state._replace(volatile=bits.at[rows,sides,active].set(
        old | jnp.where(charge,jnp.uint32(FOCUS_PUNCH),jnp.uint32(0))))


def failed(state,side,move,table):
    rows=jnp.arange(state.hp.shape[0]);bits=state.volatile[rows,side,state.active[rows,side]]
    return table.move[move] & ((bits & FOCUS_LOST)!=0)


def hit(state,side,move,table,*,eligible):
    """Target onHit: non-status direct hit, even zero Endure damage, not Sub.

    Caller excludes miss/Protect/absorption/type immunity and Substitute hits.
    A direct hit need not deal positive HP damage to break focus in Gen3.
    """
    rows=jnp.arange(state.hp.shape[0]);foe=1-side;target=state.active[rows,foe]
    bits=state.volatile[rows,foe,target]
    lost=eligible & table.damaging[move] & ((bits & FOCUS_PUNCH)!=0)
    return state._replace(volatile=state.volatile.at[rows,foe,target].set(
        bits | jnp.where(lost,jnp.uint32(FOCUS_LOST),jnp.uint32(0))))
