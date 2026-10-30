"""Gen 3 weather-move subset; Solar Beam charge is integrated separately.

IDs come from the pool, not hard-coded vocabulary positions. Weather expiry
precedes its order-8 upkeep callback, which precedes order-10 held/status events.
This component does not qualify any move or complete-game coverage.
"""
from typing import NamedTuple

import jax
import jax.numpy as jnp

NONE, RAIN, SUN, SAND, HAIL = range(5)


class WeatherTable(NamedTuple):
    move_weather: jax.Array
    thunder: jax.Array
    fire_type: jax.Array
    water_type: jax.Array
    sand_immune_type: jax.Array
    ice_type: jax.Array
    suppressor: jax.Array
    sand_veil: jax.Array


def from_metadata(metadata):
    vocab = metadata["vocabulary"]
    types = vocab["types"]
    flags = lambda key,names: jnp.asarray([name in names for name in vocab[key]])
    weather = {"raindance":RAIN,"sunnyday":SUN,"sandstorm":SAND,"hail":HAIL}
    return WeatherTable(jnp.asarray([weather.get(name,NONE) for name in vocab["moves"]]),
        flags("moves",{"thunder"}),jnp.asarray(types.index("fire")),
        jnp.asarray(types.index("water")),flags("types",{"rock","ground","steel"}),
        jnp.asarray(types.index("ice")),flags("abilities",{"airlock","cloudnine"}),
        flags("abilities",{"sandveil"}))


def effective(table,weather,abilities,hp):
    suppressed = jnp.any(table.suppressor[abilities] & (hp > 0),axis=-1)
    return jnp.where(suppressed,NONE,weather)


def set_move(weather,counter,requested,*,eligible=True,_weather_trace=None,
             _actor_side=None,_actor_physical=None,_move_id=None):
    # Gen 3 same-weather moves fail without refreshing either temporary or
    # permanent weather. A different weather replaces the current condition.
    change = jnp.asarray(eligible) & (requested != NONE) & (requested != weather)
    if _weather_trace is not None:
        from . import engine_weather_progress_callbacks as callbacks
        from . import engine_weather_effect_callbacks as effects
        new_weather,new_counter=jnp.where(change,requested,weather),jnp.where(change,5,counter)
        trace=callbacks.move_weather(_weather_trace,_actor_side,_actor_physical,_move_id,
            weather,new_weather,counter,new_counter,eligible=change)
        # The real source emits a failed-move outcome on an attempted
        # same-weather use. Until that control is registered, keep it unknown.
        trace=effects.same_weather_failure(trace,_actor_side,_actor_physical,_move_id,
            weather,new_weather,requested,
            eligible=jnp.asarray(eligible)&(requested!=NONE)&(requested==weather))
        return new_weather,new_counter,trace
    return jnp.where(change,requested,weather),jnp.where(change,5,counter)


def residual_clock(weather,counter,*,eligible=True,_weather_trace=None):
    ticking = jnp.asarray(eligible) & (weather != NONE) & (counter > 0)
    next_counter = jnp.where(ticking,counter-1,counter)
    if _weather_trace is not None:
        from . import engine_weather_progress_callbacks as callbacks
        new_weather=jnp.where(ticking & (next_counter == 0),NONE,weather)
        trace=callbacks.weather_clock(_weather_trace,weather,new_weather,counter,next_counter,eligible=eligible)
        return new_weather,next_counter,trace
    return jnp.where(ticking & (next_counter == 0),NONE,weather),next_counter


def damage_modifier(table,weather,move_type):
    boost = ((weather == RAIN) & (move_type == table.water_type)
             | (weather == SUN) & (move_type == table.fire_type))
    weaken = ((weather == RAIN) & (move_type == table.fire_type)
              | (weather == SUN) & (move_type == table.water_type))
    return jnp.where(boost,3,1),jnp.where(boost | weaken,2,1)


def thunder_accuracy(table,weather,move,accuracy,always_hit):
    thunder = table.thunder[move]
    return (jnp.where(thunder & (weather == SUN),50,accuracy),
            always_hit | (thunder & (weather == RAIN)))


def chip(table,weather,types,ability,hp,max_hp,*,eligible=True):
    sand_immune = jnp.any(table.sand_immune_type[types],axis=-1) | table.sand_veil[ability]
    hail_immune = jnp.any(types == table.ice_type,axis=-1)
    damage = ((weather == SAND) & ~sand_immune | (weather == HAIL) & ~hail_immune)
    damage &= jnp.asarray(eligible) & (hp > 0)
    return jnp.where(damage,jnp.maximum(0,hp-jnp.maximum(1,max_hp//16)),hp)
