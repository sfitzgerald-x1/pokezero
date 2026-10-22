"""Selected Gen3 Solar Beam charge and weather rules, not generic charge moves."""
from typing import NamedTuple

import jax
import jax.numpy as jnp

from . import weather


class SolarTable(NamedTuple):
    move: jax.Array


def from_metadata(metadata):
    return SolarTable(jnp.asarray([name=='solarbeam' for name in metadata['vocabulary']['moves']]))


def clear(state,side,*,eligible):
    rows=jnp.arange(state.hp.shape[0]);active=state.active[rows,side]
    return state._replace(**{name:getattr(state,name).at[rows,side,active].set(
        jnp.where(eligible,0,getattr(state,name)[rows,side,active]))
        for name in ('charge_move','charge_counter')})


def try_move(state,side,move,table,effective_weather,*,eligible):
    """TryMove follows the caller's PP/Choice bookkeeping, before target gates."""
    rows=jnp.arange(state.hp.shape[0]);active=state.active[rows,side]
    solar=eligible & table.move[move]
    release=solar & (state.charge_move[rows,side,active]==move)
    starts=solar & ~release & (effective_weather!=weather.SUN)
    state=clear(state,side,eligible=release)
    state=state._replace(charge_move=state.charge_move.at[rows,side,active].set(
        jnp.where(starts,move,state.charge_move[rows,side,active])),
        charge_counter=state.charge_counter.at[rows,side,active].set(
            jnp.where(starts,2,state.charge_counter[rows,side,active])))
    return state,starts


def base_power(table,move,power,effective_weather):
    weak=(effective_weather==weather.RAIN)|(effective_weather==weather.SAND)|(effective_weather==weather.HAIL)
    return jnp.where(table.move[move]&weak,power//2,power)


def residual(state,*,eligible):
    """Duration2 includes the charge turn; deferred replacements do not tick."""
    counter=jnp.where(eligible[:,None,None],jnp.maximum(0,state.charge_counter-1),state.charge_counter)
    return state._replace(charge_counter=counter,charge_move=jnp.where(counter>0,state.charge_move,0))
