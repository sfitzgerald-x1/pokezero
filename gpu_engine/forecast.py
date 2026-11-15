"""Gen 3 Forecast (Castform): the forme follows the effective weather at each WeatherChange event.

Showdown f76228a1 (Gen 3 inherits data/abilities.ts through the gen4 and gen5 mods):
- data/abilities.ts forecast: onStart runs singleEvent('WeatherChange') on the holder itself; onWeatherChange returns
  unless pokemon.baseSpecies.baseSpecies is Castform and the mon is not transformed, then takes the forme of
  pokemon.effectiveWeather(): sunnyday -> Castform-Sunny (Fire), raindance -> Castform-Rainy (Water), hail ->
  Castform-Snowy (Ice), anything else (sandstorm, none, suppressed) -> Castform (Normal). It changes only an active mon
  whose species differs: formeChange(forme, effect, false, '0', '[msg]') prints |-formechange|...|[msg]|[from] ability:
  Forecast (sim/pokemon.ts formeChange) and setSpecies replaces the types (setType(species.types, true)). The formes
  share base stats, so nothing else changes. data/mods/gen3/abilities.ts sets flags {} (Gen 4 adds notrace), so Trace
  can copy it, inertly: the copier is not Castform.
- WeatherChange fires (eachEvent, every non-fainted active in Speed order) at setWeather's success (sim/field.ts: a move
  that changes the weather; an ability Start that sets a new weather or makes a temporary one permanent), at clearWeather
  (the residual clock ends it), and at a Cloud Nine / Air Lock holder's End (data/abilities.ts onEnd: ending = true, then
  eachEvent): its switch-out (sim/battle-actions.ts switchIn, for an unfainted old active) and its faint (sim/battle.ts
  faintMessages). Gen 4 (and so Gen 3) removes their onSwitchIn and makes onStart only clear `ending`
  (data/mods/gen4/abilities.ts airlock, cloudnine): a suppressor's arrival fires no WeatherChange, so a Castform keeps its
  forme until the next event (a Cloud Nine -> Air Lock handoff exposes the raw weather in between).
- sim/field.ts suppressingWeather: a non-fainted active whose ability suppresses weather and is not `ending`. An
  installed holder suppresses before its own Start runs.
- A switch-out's clearVolatile -> setSpecies(baseSpecies) reverts silently: entry.depart restores base_types.

The engine stores no forme: the types are its proxy (Castform's four formes have four distinct type rows, and nothing
else retypes a Castform in Gen 3). `species` is the constructor species (Transform keeps it), so it stands for
baseSpecies: a Ditto or Mew copying Castform, or a Trace copy of Forecast, is inert. No Gen 3 random-battle set sets hail,
so Castform-Snowy is unreachable there; the table still carries it. Forecast draws no chance.
"""
from typing import NamedTuple

import jax
import jax.numpy as jnp

from . import weather as climate


class ForecastTable(NamedTuple):
    forecast: jax.Array   # [A] bool, the ability is Forecast
    castform: jax.Array   # () int32, Castform's species id; -1 when the vocabulary lacks it (no holder)
    forme: jax.Array      # [5,2] int32, per weather id (NONE, RAIN, SUN, SAND, HAIL) its forme's types
    retyped: jax.Array    # [5] bool, the weather has its own forme (rain, sun, hail); otherwise Castform's own types


def from_metadata(metadata):
    vocab=metadata["vocabulary"];types=vocab["types"]
    forme=[[0,0] for _ in range(5)]
    for weather,name in ((climate.RAIN,"water"),(climate.SUN,"fire"),(climate.HAIL,"ice")):
        forme[weather]=[types.index(name),0]
    species=vocab["species"]
    return ForecastTable(jnp.asarray([name=="forecast" for name in vocab["abilities"]]),
        jnp.asarray(species.index("castform") if "castform" in species else -1,jnp.int32),
        jnp.asarray(forme,jnp.int32),jnp.asarray([w in (climate.RAIN,climate.SUN,climate.HAIL) for w in range(5)]))


def _actives(state,name):
    rows=jnp.arange(state.hp.shape[0])[:,None];sides=jnp.arange(2)[None,:]
    return getattr(state,name)[rows,sides,state.active]


def update(state,table,weather_table,*,eligible,weather=None,ending=None):
    """One WeatherChange event: every eligible living active Castform holding Forecast takes the forme of the effective
    weather. eligible: [B] (both actives) or [B,2]; weather: the field weather the event sees (default state.weather);
    ending: [B,2] actives whose suppressor ability no longer counts (an End in progress). Returns (state, changed [B,2])."""
    b=state.hp.shape[0];rows=jnp.arange(b)[:,None];sides=jnp.arange(2)[None,:]
    eligible=jnp.asarray(eligible,bool)
    eligible=jnp.broadcast_to(eligible[:,None] if eligible.ndim==1 else eligible,(b,2))
    ability=_actives(state,'ability');hp=_actives(state,'hp')
    suppressing=weather_table.suppressor[ability] & (hp>0)
    if ending is not None:suppressing &= ~jnp.asarray(ending,bool)
    raw=state.weather if weather is None else weather
    effective=jnp.where(jnp.any(suppressing,axis=1),climate.NONE,raw)
    old=state.types[rows,sides,state.active]
    target=jnp.where(table.retyped[effective][:,None,None],table.forme[effective][:,None,:],_actives(state,'base_types'))
    holder=eligible & (_actives(state,'species')==table.castform) & table.forecast[ability] & (hp>0)
    # Transform (when the state carries it): onWeatherChange returns for a transformed mon.
    if hasattr(state,'transformed'):holder &= ~state.transformed
    new=jnp.where(holder[...,None],target,old)
    return state._replace(types=state.types.at[rows,sides,state.active].set(new)),jnp.any(new!=old,axis=-1)


def suppressor_fainted(state,before_hp,weather_table):
    """[B] an active Cloud Nine / Air Lock holder (its current ability: a Trace copy counts) fainted since before_hp: its
    End at faintMessages fires WeatherChange."""
    fainted=(before_hp[jnp.arange(state.hp.shape[0])[:,None],jnp.arange(2)[None,:],state.active]>0) & (_actives(state,'hp')<=0)
    return jnp.any(fainted & weather_table.suppressor[_actives(state,'ability')],axis=1)


def suppressor_departs(before,side,weather_table):
    """[B] `side`'s active in `before` (its pre-departure ability) is a living suppressor: its End at the switch-out fires
    WeatherChange. A fainted old active runs no End there (switchIn's unfaintedActive)."""
    rows=jnp.arange(before.hp.shape[0]);active=before.active[rows,side]
    return weather_table.suppressor[before.ability[rows,side,active]] & (before.hp[rows,side,active]>0)


def entry_events(before,after,side,weather_table):
    """[B,2] the WeatherChange events of one entry (gen4 runSwitch: hazards, then the Ability Start): the entrant's own
    Forecast Start (its own side), a weather its Start set or made permanent (setWeather: both sides), and a suppressor
    entrant's hazard faint (its End: both sides)."""
    rows=jnp.arange(before.hp.shape[0]);sides=jnp.arange(2)[None,:];own=sides==jnp.asarray(side)[...,None]
    weather=(after.weather!=before.weather) | (after.weather_counter!=before.weather_counter)
    return own | (weather | suppressor_fainted(after,before.hp,weather_table))[:,None]
