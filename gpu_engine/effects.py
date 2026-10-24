"""Static tables for the initial major-status and self-effect move subset.

These are explicit Gen 3 rules, not inference from names or base power. Items,
abilities and the wider move/event surface still need integration. The full
transition conservatively flags every move as a placeholder until qualification.
"""
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from . import mechanics as m
from . import weather as climate


class MoveEffects(NamedTuple):
    status: jax.Array             # [M], major status applied to foe
    self_boosts: jax.Array        # [M,7]
    heal_half: jax.Array          # [M], floor(max HP / 2)
    rest: jax.Array               # [M], full heal and exactly three sleep counters
    haze: jax.Array               # [M], clear both active mons' boosts
    thunderwave: jax.Array        # [M], Ground immunity belongs to this move
    status_immunity: jax.Array    # [7,18], type-based major-status immunity
    ground_type: jax.Array
    substitute: jax.Array
    protect: jax.Array
    endure: jax.Array
    protectable: jax.Array         # actual pinned move flags, not category inference
    bypass_sub: jax.Array
    heal_bell: jax.Array
    soundproof: jax.Array
    weather_heal: jax.Array       # [M], Gen 3 Synthesis/Moonlight/Morning Sun
    pain_split: jax.Array         # [M], live HP average assigned through own caps
    belly_drum: jax.Array         # [M], direct half-max HP cost and Attack +12
    hit_spa_drop: jax.Array       # [M], Overheat/Psycho Boost self drop on successful hit


def from_metadata(metadata):
    names = metadata["vocabulary"]["moves"]
    types = metadata["vocabulary"]["types"]
    index = {name:i for i,name in enumerate(names)}
    status = np.zeros(len(names),np.int32)
    for name, value in {"toxic":m.TOX,"poisonpowder":m.PSN,"thunderwave":m.PAR,
            "stunspore":m.PAR,"willowisp":m.BRN,"sleeppowder":m.SLP,
            "hypnosis":m.SLP,"spore":m.SLP,"sing":m.SLP}.items():
        if name in index: status[index[name]] = value
    boosts = np.zeros((len(names),7),np.int32)
    for name, values in {
        "swordsdance":[2,0,0,0,0,0,0], "calmmind":[0,0,1,1,0,0,0],
        "bulkup":[1,1,0,0,0,0,0], "agility":[0,0,0,0,2,0,0],
        "dragondance":[1,0,0,0,1,0,0],
        "tailglow":[0,0,2,0,0,0,0],
        "amnesia":[0,0,0,2,0,0,0], "irondefense":[0,2,0,0,0,0,0],
        "barrier":[0,2,0,0,0,0,0], "meditate":[1,0,0,0,0,0,0],
        "howl":[1,0,0,0,0,0,0], "doubleteam":[0,0,0,0,0,0,1],
    }.items():
        if name in index: boosts[index[name]] = values
    flags = lambda chosen: jnp.asarray([name in chosen for name in names])
    immunity = np.zeros((7,len(types)),bool)
    for value, blocked in ((m.BRN,["fire"]),(m.FRZ,["ice"]),
                           (m.PSN,["poison","steel"]),(m.TOX,["poison","steel"])):
        for name in blocked: immunity[value,types.index(name)] = True
    # Electric types can be paralyzed in Gen 3. Ground only blocks Thunder Wave,
    # not every route to paralysis (for example Stun Spore).
    rows = metadata.get("moves")
    if rows is not None and len(rows) != len(names):
        raise ValueError("move metadata/vocabulary mismatch")
    move_flag = lambda flag: jnp.asarray([
        bool(row and row.get("flags",{}).get(flag)) for row in rows
    ]) if rows is not None else jnp.zeros(len(names),bool)
    return MoveEffects(jnp.asarray(status),jnp.asarray(boosts),
        flags({"recover","softboiled","milkdrink","slackoff"}),flags({"rest"}),
        flags({"haze"}),flags({"thunderwave"}),jnp.asarray(immunity),
        jnp.asarray(types.index("ground"),jnp.int32),flags({"substitute"}),
        flags({"protect","detect"}),flags({"endure"}),move_flag("protect"),move_flag("bypasssub"),
        flags({"healbell"}),jnp.asarray([name=="soundproof" for name in
            metadata["vocabulary"].get("abilities",["_none"])]),
        flags({"synthesis","moonlight","morningsun"}),flags({"painsplit"}),flags({"bellydrum"}),
        flags({"overheat","psychoboost"}))


def cure_team(state,side,table,*,eligible):
    """Gen3 Heal Bell cures living own-party status except each Soundproof holder.

    The user is not exempt from Soundproof in this generation. Confusion and
    other volatiles are not major status and are deliberately preserved.
    """
    rows=jnp.arange(state.hp.shape[0])
    status=state.status[rows,side]
    cured=(jnp.asarray(eligible)[:,None] & (state.hp[rows,side]>0) & (status!=m.NONE)
        & ~table.soundproof[state.ability[rows,side]])
    updates={}
    for name in ("status","status_counter","sleep_skipped"):
        value=getattr(state,name)
        updates[name]=value.at[rows,side].set(jnp.where(cured,0,value[rows,side]))
    return state._replace(**updates)
def weather_healing(max_hp, weather):
    """Pinned Gen 3 fractional healing, truncated once (not modern 0.667)."""
    amount = jnp.where(weather == climate.SUN,max_hp * 2 // 3,
        jnp.where((weather == climate.RAIN) | (weather == climate.SAND)
            | (weather == climate.HAIL),max_hp // 4,max_hp // 2))
    return jnp.maximum(1,amount)


def stall_success(counter, roll):
    """Gen 3 consecutive Protect/Detect/Endure: 1, 1/2, 1/4, then 1/8.

    Counter is the next attempt's denominator (zero means no active stall).
    Roll is uniform integer [0,8); a successful attempt stores min(8,2*counter).
    """
    return roll < 8 // jnp.maximum(1,counter)
