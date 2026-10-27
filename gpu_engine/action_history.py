"""Side-owned Gen3 action history, matching Rust LastUsedMove variants.

Record at dispatch/committed switch, never reconstruct a slot from move identity.
Initialization, failed BeforeMove, faint cleanup and entry callbacks preserve it.
The per-Pokemon last_move field remains a separate caller-identity observation.
"""
import jax.numpy as jnp

NONE, SWITCH, MOVE, STRUGGLE = range(4)


def move_slot(state, side):
    """Return a safe gather index and validity, without identity-based guessing."""
    rows = jnp.arange(state.hp.shape[0])
    index = state.last_action_index[rows,side]
    valid = (state.last_action_kind[rows,side] == MOVE) & (index >= 0) & (index < 4)
    return jnp.clip(index,0,3),valid


def record_move(state, side, slot, *, eligible, struggle=False):
    rows = jnp.arange(state.hp.shape[0])
    kind = jnp.where(struggle, STRUGGLE, MOVE)
    index = jnp.where(struggle, -1, slot)
    return state._replace(
        last_action_kind=state.last_action_kind.at[rows,side].set(jnp.where(
            eligible,kind,state.last_action_kind[rows,side])),
        last_action_index=state.last_action_index.at[rows,side].set(jnp.where(
            eligible,index,state.last_action_index[rows,side])))


def record_switch(state, side, target, *, eligible):
    rows = jnp.arange(state.hp.shape[0])
    return state._replace(
        last_action_kind=state.last_action_kind.at[rows,side].set(jnp.where(
            eligible,SWITCH,state.last_action_kind[rows,side])),
        last_action_index=state.last_action_index.at[rows,side].set(jnp.where(
            eligible,target,state.last_action_index[rows,side])))
