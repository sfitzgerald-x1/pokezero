"""Selected Gen 3 ability tables and entry effects; not full coverage.

The inherited Gen 4 runSwitch applies hazards before Ability Start. Ordinary
switches and their runSwitch callbacks resolve one at a time, not together.
Pursuit is dispatched by transition before entry. Direct Baton and phazing
reuse these selected entry/departure events; broader ability events are pending.
"""
from typing import NamedTuple

import jax
import jax.numpy as jnp

from . import weather
from . import mechanics as m


class EntryTable(NamedTuple):
    weather_ability: jax.Array
    intimidate: jax.Array
    natural_cure: jax.Array
    levitate: jax.Array
    attack_drop_immune: jax.Array
    swift_swim: jax.Array
    chlorophyll: jax.Array
    flying_type: jax.Array
    ground_type: jax.Array
    spikes_move: jax.Array
    speed_boost: jax.Array


def from_metadata(metadata):
    vocab = metadata["vocabulary"]
    flag = lambda names: jnp.asarray([name in names for name in vocab["abilities"]])
    climates = {"drizzle":weather.RAIN,"drought":weather.SUN,"sandstream":weather.SAND}
    return EntryTable(jnp.asarray([climates.get(name,weather.NONE) for name in vocab["abilities"]]),
        flag({"intimidate"}),flag({"naturalcure"}),flag({"levitate"}),
        flag({"clearbody","whitesmoke","hypercutter"}),flag({"swiftswim"}),flag({"chlorophyll"}),
        jnp.asarray(vocab["types"].index("flying")),jnp.asarray(vocab["types"].index("ground")),
        jnp.asarray([name == "spikes" for name in vocab["moves"]]),flag({"speedboost"}))


def speed_modifier(table,ability,effective_weather,speed):
    doubled = ((table.swift_swim[ability] & (effective_weather == weather.RAIN))
        | (table.chlorophyll[ability] & (effective_weather == weather.SUN)))
    return speed * jnp.where(doubled,2,1)


def depart(state,side,table,*,eligible):
    """Selected outgoing cleanup shared by voluntary, Baton and drag switches.

    Baton copies its supported fields before calling this. This does not run
    BeforeSwitchOut/Pursuit, which does not apply to a drag or direct Baton.
    """
    rows=jnp.arange(state.hp.shape[0]);active=state.active[rows,side]
    hp=state.hp[rows,side,active];status=state.status[rows,side,active]
    cure=hp<=0
    if table is not None:
        cure |= table.natural_cure[state.ability[rows,side,active]]
    counter=state.status_counter[rows,side,active]
    state=state._replace(status=state.status.at[rows,side,active].set(
        jnp.where(eligible & cure,m.NONE,status)),
        status_counter=state.status_counter.at[rows,side,active].set(
            jnp.where(eligible & (cure | (status==m.TOX)),0,counter)),
        sleep_skipped=state.sleep_skipped.at[rows,side,active].set(jnp.where(
            eligible & (cure | (status!=m.SLP)),0,state.sleep_skipped[rows,side,active])))
    for name in ("boosts","volatile","confusion_counter","protect_counter","substitute_hp","choice_move",
                 "last_move","encore_move","encore_counter"):
        value=getattr(state,name);old=value[rows,side,active]
        gate=eligible[:,None] if old.ndim==2 else eligible
        state=state._replace(**{name:value.at[rows,side,active].set(jnp.where(gate,0,old))})
    # Remove both ends of linked move traps. A direct Baton remaps/copies its
    # links to the recipient BEFORE departure, so those transferred links stay.
    own=state.trap_source[rows,side,active]
    state=state._replace(trap_source=state.trap_source.at[rows,side,active].set(
        jnp.where(eligible,-1,own)))
    foe=1-side;linked=state.trap_source[rows,foe]
    return state._replace(trap_source=state.trap_source.at[rows,foe].set(
        jnp.where(eligible[:,None] & (linked==active[:,None]),-1,linked)))


def enter(state,side,table,*,eligible,hazards=True):
    """One entry event; the caller owns ordering and queued-action cancellation."""
    rows = jnp.arange(state.hp.shape[0])
    active = state.active[rows,side]
    foe = 1-side
    foe_active = state.active[rows,foe]
    hp = state.hp[rows,side,active]
    ability = state.ability[rows,side,active]
    grounded = (~jnp.any(state.types[rows,side,active] == table.flying_type,axis=-1)
        & ~table.levitate[ability])
    layers = jnp.clip(state.spikes[rows,side],0,3)
    numerator = jnp.array([0,3,4,6],jnp.int32)[layers]
    amount = jnp.maximum(1,state.max_hp[rows,side,active]*numerator//24)
    hit = eligible & (hp > 0) & grounded & (layers > 0) & hazards
    next_hp = jnp.where(hit,jnp.maximum(0,hp-amount),hp)
    knocked_out = hit & (next_hp == 0)
    state = state._replace(hp=state.hp.at[rows,side,active].set(next_hp))
    # Gen3 runs status SwitchIn even after a hazard KO, before faintMessages.
    state=restore_sleep(state,side,eligible=eligible)
    starts = eligible & (next_hp > 0)
    requested = table.weather_ability[ability]
    change = (starts & (requested != weather.NONE)
        & ((state.weather != requested) | (state.weather_counter > 0)))
    state = state._replace(weather=jnp.where(change,requested,state.weather),
        weather_counter=jnp.where(change,0,state.weather_counter))
    foe_hp = state.hp[rows,foe,foe_active]
    blocked = ((state.substitute_hp[rows,foe,foe_active] > 0)
        | table.attack_drop_immune[state.ability[rows,foe,foe_active]])
    drops = starts & table.intimidate[ability] & (foe_hp > 0) & ~blocked
    old = state.boosts[rows,foe,foe_active,0]
    boosts = state.boosts.at[rows,foe,foe_active,0].set(jnp.where(drops,jnp.maximum(-6,old-1),old))
    return state._replace(boosts=boosts),knocked_out


def restore_sleep(state,side,*,eligible):
    """Gen3 status SwitchIn after hazards, before Ability Start."""
    rows=jnp.arange(state.hp.shape[0]);active=state.active[rows,side]
    skipped=state.sleep_skipped[rows,side,active]
    sleeping=state.status[rows,side,active]==m.SLP
    counter=state.status_counter[rows,side,active]
    return state._replace(status_counter=state.status_counter.at[rows,side,active].set(
        jnp.where(eligible & sleeping,counter+skipped,counter)),
        sleep_skipped=state.sleep_skipped.at[rows,side,active].set(jnp.where(eligible,0,skipped)))
