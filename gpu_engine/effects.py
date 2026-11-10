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
    # Fidelity patch A1 (appended; every existing field keeps its position and value).
    curse: jax.Array              # [M], non-Ghost Curse; its self_boosts row is +1 Atk, +1 Def, -1 Spe
    ghost_type: jax.Array         # [], a Ghost-type user's Curse is the unported Ghost form (no boost)
    hit_self_boosts: jax.Array    # [M,7], Superpower self stages after a successful hit
    target_boosts: jax.Array      # [M,7], Charm's primary stage change on the target
    drop_block: jax.Array         # [A,7], foe-caused drops an ability blocks (Clear Body, White Smoke, Hyper Cutter)
    aromatherapy: jax.Array       # [M], gen4-inherited party clearStatus, Soundproof included
    refresh: jax.Array            # [M], cure the user's par/brn/psn/tox
    # Fidelity patch A2 (appended).
    destinybond: jax.Array        # [M], volatileStatus destinybond (state.DESTINY_BOND)


# Gen 3 inherits gen4's Curse (data/mods/gen4/moves.ts:268-281): a non-Ghost user
# gets self {atk:1, def:1, spe:-1} and targets itself; Superpower's self boosts
# (data/moves.ts:18494-18499); Charm's boosts (data/moves.ts:2345-2347).
CURSE_BOOSTS = [1, 1, 0, 0, -1, 0, 0]
HIT_SELF_BOOSTS = {"superpower": [-1, -1, 0, 0, 0, 0, 0]}
TARGET_BOOSTS = {"charm": [-2, 0, 0, 0, 0, 0, 0]}
# onTryBoost of a foe-caused drop (data/abilities.ts clearbody/whitesmoke delete
# every negative stage, hypercutter only Attack; gen3 inherits them unchanged).
DROP_BLOCK = {"clearbody": [True] * 7, "whitesmoke": [True] * 7,
              "hypercutter": [True, False, False, False, False, False, False]}


def from_metadata(metadata):
    names = metadata["vocabulary"]["moves"]
    types = metadata["vocabulary"]["types"]
    index = {name:i for i,name in enumerate(names)}
    status = np.zeros(len(names),np.int32)
    for name, value in {"toxic":m.TOX,"poisonpowder":m.PSN,"thunderwave":m.PAR,
            "stunspore":m.PAR,"willowisp":m.BRN,"sleeppowder":m.SLP,
            "hypnosis":m.SLP,"spore":m.SLP,"sing":m.SLP,"lovelykiss":m.SLP}.items():
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
        "curse":CURSE_BOOSTS,
    }.items():
        if name in index: boosts[index[name]] = values
    hit_self = np.zeros((len(names),7),np.int32)
    for name, values in HIT_SELF_BOOSTS.items():
        if name in index: hit_self[index[name]] = values
    target = np.zeros((len(names),7),np.int32)
    for name, values in TARGET_BOOSTS.items():
        if name in index: target[index[name]] = values
    abilities = metadata["vocabulary"].get("abilities",["_none"])
    drop_block = np.asarray([DROP_BLOCK.get(name,[False]*7) for name in abilities],bool).reshape(len(abilities),7)
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
        flags({"overheat","psychoboost"}),flags({"curse"}),
        jnp.asarray(types.index("ghost") if "ghost" in types else -1,jnp.int32),
        jnp.asarray(hit_self),jnp.asarray(target),jnp.asarray(drop_block),
        flags({"aromatherapy"}),flags({"refresh"}),flags({"destinybond"}))


def cure_team(state,side,table,*,eligible,_return_write_mask=False):
    """Gen3 Heal Bell cures living own-party status except each Soundproof holder.

    The user is not exempt from Soundproof in this generation. Confusion and
    other volatiles are not major status and are deliberately preserved.
    """
    rows=jnp.arange(state.hp.shape[0])
    status=state.status[rows,side]
    cured=(jnp.asarray(eligible)[:,None] & (state.hp[rows,side]>0) & (status!=m.NONE)
        & ~table.soundproof[state.ability[rows,side]])
    updates={}
    for name in ("status","status_counter","sleep_skipped","rest_origin"):
        value=getattr(state,name)
        updates[name]=value.at[rows,side].set(jnp.where(cured,jnp.zeros_like(value[rows,side]),value[rows,side]))
    result=state._replace(**updates)
    return (result,cured) if _return_write_mask else result


def cure_party(state,side,table,*,eligible):
    """Gen3 Aromatherapy: gen4's onHit (data/mods/gen4/moves.ts:26-35) runs
    clearStatus() on every member of the user's side. clearStatus (sim/pokemon.ts:1755)
    skips a fainted or unstatused mon; nothing else is exempt (no Soundproof skip,
    unlike Heal Bell), and Protect or a Substitute never interacts. Same writes as
    cure_team, whose Soundproof column is cleared here rather than changed."""
    return cure_team(state,side,table._replace(soundproof=jnp.zeros_like(table.soundproof)),eligible=eligible)


def sleep_clause_blocked(state,side):
    """[B] Sleep Clause Mod (data/rulesets.ts:1378-1401) holds against a foe-made
    sleep on `side`: another living member is asleep from a non-ally source.

    Rest (the only ally-sourced sleep, statusState.source is the user) is
    rest_origin and does not count. The target itself is excluded: its sleep
    attempt already failed on 'already statused' before the SetStatus event.
    """
    rows=jnp.arange(state.hp.shape[0])
    other=jnp.arange(state.hp.shape[2])[None,:]!=state.active[rows,side][:,None]
    return jnp.any((state.hp[rows,side]>0) & (state.status[rows,side]==m.SLP)
        & ~state.rest_origin[rows,side] & other,axis=1)


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
