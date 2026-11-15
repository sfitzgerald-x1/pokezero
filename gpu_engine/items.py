"""Explicit Gen 3 held-item subset for the isolated transition.

No broader item/ability fidelity claim. IDs come from the real pool vocabulary.
"""
from typing import NamedTuple

import jax
import jax.numpy as jnp

from . import mechanics as m


class ItemTable(NamedTuple):
    leftovers: jax.Array
    lum: jax.Array
    choice_band: jax.Array
    thick_club: jax.Array
    light_ball: jax.Array
    club_species: jax.Array
    ball_species: jax.Array
    soul_dew: jax.Array
    dew_species: jax.Array
    pinch_stat: jax.Array
    # Fidelity patch A2 (appended): Knock Off removes the target's item; Sticky Hold blocks it.
    knock_off: jax.Array       # [M], gen4 moves.ts knockoff onAfterHit (inherited by Gen 3)
    sticky_hold: jax.Array     # [A], gen4 abilities.ts stickyhold onTakeItem
    # Fidelity patch M1 (appended): Trick swaps the actives' items (data/moves.ts trick, Gen 3 = base).
    trick: jax.Array           # [M]
    # Fidelity WH (appended): White Herb (data/items.ts whiteherb; no gen3/gen4/gen5-8 override, so Gen 3 is the base item).
    white_herb: jax.Array      # [I]
    # Damage modifiers (appended): Gen 3's 1.1x type items in the pool, Silk Scarf and Twisted Spoon (see attack_modifier).
    type_boost: jax.Array      # [I], the type index the item boosts (0: none)


# data/mods/gen3/items.ts (f76228a1) moves the type items from BasePower to the attack stat; the Gen 3 category follows
# the move's type, so the one hook (onModifyAtk for Normal, onModifySpA for the special types) is the move's category.
# getItem (data/random-battles/gen3/teams.ts) assigns only these two: Linoone's Silk Scarf, Unown's Twisted Spoon.
TYPE_BOOST_ITEMS = {"silkscarf": "normal", "twistedspoon": "psychic"}


def from_metadata(metadata):
    vocab = metadata["vocabulary"]
    flag = lambda key,names: jnp.asarray([name in names for name in vocab[key]])
    return ItemTable(flag("items",{"leftovers"}),flag("items",{"lumberry"}),
        flag("items",{"choiceband"}),flag("items",{"thickclub"}),flag("items",{"lightball"}),
        flag("species",{"cubone","marowak"}),flag("species",{"pikachu"}),
        flag("items",{"souldew"}),flag("species",{"latias","latios"}),
        jnp.asarray([{"liechiberry":0,"petayaberry":2,"salacberry":4}.get(name,-1) for name in vocab["items"]]),
        jnp.asarray([name=="knockoff" for name in vocab.get("moves",["_none"])]),
        jnp.asarray([name=="stickyhold" for name in vocab.get("abilities",["_none"])]),
        jnp.asarray([name=="trick" for name in vocab.get("moves",["_none"])]),
        flag("items",{"whiteherb"}),_type_boost(vocab))


def _type_boost(vocab):
    types = vocab.get("types")
    if types is None and any(name in TYPE_BOOST_ITEMS for name in vocab["items"]):
        raise ValueError("a type item needs the type vocabulary")
    return jnp.asarray([types.index(TYPE_BOOST_ITEMS[name]) if name in TYPE_BOOST_ITEMS else 0
        for name in vocab["items"]],jnp.int32)


def attack_modifier(table,item,species,physical,*,move_type=None):
    """The held item's Atk/SpA handler as (num, den): one item, so one factor of the ModifyAtk/ModifySpA chain.

    Damage modifiers: Silk Scarf (data/mods/gen3/items.ts:337-345, onModifyAtkPriority 1) is chainModify(1.1) on Atk for
    a Normal move, Twisted Spoon (data/mods/gen3/items.ts:406-415, onModifySpAPriority 1) on SpA for a Psychic move (Gen 3
    clears both items' base onBasePower): 1.1*4096 truncates to 4505, returned as 4505/4096 (combat.attack_modifier chains
    it with the ability in Q12). move_type is the move's resolved type: Hidden Power's own (a Hidden Power Psychic takes
    Twisted Spoon; Hidden Power is never Normal), Struggle and the confusion self-hit the typeless 0. None (the confusion
    self-hit, Pursuit's Dark intercept) is no type item."""
    physical = jnp.asarray(physical,jnp.bool_)
    band = physical & table.choice_band[item]
    doubled = ((physical & table.thick_club[item] & table.club_species[species])
        | (~physical & table.light_ball[item] & table.ball_species[species]))
    dew = ~physical & table.soul_dew[item] & table.dew_species[species]
    num,den = jnp.where(band | dew,3,jnp.where(doubled,2,1)),jnp.where(band | dew,2,1)
    if move_type is None:
        return num,den
    boosted = (table.type_boost[item] > 0) & (jnp.asarray(move_type) == table.type_boost[item])
    return jnp.where(boosted,4505,num),jnp.where(boosted,4096,den)


def defense_modifier(table,item,species,physical):
    """Gen3 Soul Dew modifies SpD, not physical Defense or final damage."""
    dew = ~jnp.asarray(physical,jnp.bool_) & table.soul_dew[item] & table.dew_species[species]
    return jnp.where(dew,3,1),jnp.where(dew,2,1)


class LumResult(NamedTuple):
    item: jax.Array
    status: jax.Array
    counter: jax.Array
    confusion: jax.Array


def consume_lum(table,item,status,counter,confusion,hp,*,eligible=True):
    consume = (jnp.asarray(eligible) & (hp > 0) & table.lum[item]
        & ((status != m.NONE) | (confusion > 0)))
    return LumResult(jnp.where(consume,0,item),jnp.where(consume,m.NONE,status),
        jnp.where(consume,0,counter),jnp.where(consume,0,confusion))


def leftovers_heal(table,item,hp,max_hp,*,eligible=True):
    active = jnp.asarray(eligible) & (hp > 0) & table.leftovers[item]
    amount = jnp.maximum(1,max_hp // 16)
    return jnp.where(active,jnp.minimum(max_hp,hp+amount),hp)


def white_herb(table,item,boosts,hp,*,eligible=True):
    """White Herb (data/items.ts:7653-7707, the base item; Gen 3 has no override).

    Its handler (onStart, which onAnySwitchIn, onAnyAfterMove and onResidual all
    call) finds a negative stage, so the holder runs useItem() (sim/pokemon.ts:1811-1849):
    refused at 0 HP or off the field, else -enditem|X|White Herb, onUse resets every
    negative stage to 0 (-clearnegativeboost|X|[silent]) and the item is gone
    (lastItem keeps it; it is no mutation). Positive stages stay. The caller owns
    the trigger points. Returns (item, boosts, used).
    """
    used=jnp.asarray(eligible) & (hp>0) & table.white_herb[item] & jnp.any(boosts<0,axis=-1)
    return jnp.where(used,0,item),jnp.where(used[...,None],jnp.maximum(boosts,0),boosts),used


def consume_pinch(table,item,boosts,hp,max_hp,*,eligible=True):
    """Selected Gen3 pinch berries: residual order10/suborder4, not onUpdate.

    Consume even at the boost cap, but never on a fainted/ineligible holder.
    Integer cross-multiplication retains the exact odd-max-HP threshold.
    """
    stat=table.pinch_stat[item]
    consume=jnp.asarray(eligible) & (hp>0) & (hp*4<=max_hp) & (stat>=0)
    raised=jnp.minimum(6,boosts+(jnp.arange(7)==stat[...,None]))
    return jnp.where(consume,0,item),jnp.where(consume[...,None],raised,boosts)
