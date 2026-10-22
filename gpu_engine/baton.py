"""Direct Baton Pass subset; not a general volatile/event-copy engine."""
from typing import NamedTuple

import jax
import jax.numpy as jnp

from .state import LEECH_SEED


class BatonTable(NamedTuple):
    move: jax.Array


def from_metadata(metadata):
    return BatonTable(jnp.asarray([
        name == "batonpass" for name in metadata["vocabulary"]["moves"]]))


def copy_to_recipient(state, side, target, *, eligible):
    """Copy supported passable state BEFORE outgoing cleanup and entry events.

    HP, status, species, stats, moves, PP, item and ability belong to the
    recipient. Direct move-trap links transfer at either end. Flash Fire and
    Choice lock do not pass. A successful direct
    Baton cannot retain flinch/Protect/Endure; wider volatile copy is unclaimed.
    """
    rows = jnp.arange(state.hp.shape[0])
    old = state.active[rows, side]
    # Gen 3 copies both `trapped` and `trapper` linked volatiles. Stable party
    # indices identify the owner; a passing owner redirects every linked target.
    own=state.trap_source[rows,side,old]
    state=state._replace(trap_source=state.trap_source.at[rows,side,target].set(
        jnp.where(eligible,own,state.trap_source[rows,side,target])))
    foe=1-side;links=state.trap_source[rows,foe]
    state=state._replace(trap_source=state.trap_source.at[rows,foe].set(
        jnp.where(eligible[:,None] & (links==old[:,None]),target[:,None],links)))
    for name in ("boosts", "substitute_hp", "confusion_counter", "volatile",
                 "protect_counter", "choice_move", "last_move", "encore_move", "encore_counter", "charge_move", "charge_counter"):
        value = getattr(state, name)
        source = value[rows, side, old]
        if name == "volatile":
            source &= jnp.uint32(LEECH_SEED)
        elif name in ("protect_counter", "choice_move", "last_move", "encore_move", "encore_counter", "charge_move", "charge_counter"):
            source = jnp.zeros_like(source)
        gate = eligible[:, None] if source.ndim == 2 else eligible
        value = value.at[rows, side, target].set(jnp.where(
            gate, source, value[rows, side, target]))
        state = state._replace(**{name: value})
    return state
