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


def from_metadata(metadata):
    vocab = metadata["vocabulary"]
    flag = lambda key,names: jnp.asarray([name in names for name in vocab[key]])
    return ItemTable(flag("items",{"leftovers"}),flag("items",{"lumberry"}),
        flag("items",{"choiceband"}),flag("items",{"thickclub"}),flag("items",{"lightball"}),
        flag("species",{"cubone","marowak"}),flag("species",{"pikachu"}))


def attack_modifier(table,item,species,physical):
    physical = jnp.asarray(physical,jnp.bool_)
    band = physical & table.choice_band[item]
    doubled = ((physical & table.thick_club[item] & table.club_species[species])
        | (~physical & table.light_ball[item] & table.ball_species[species]))
    return jnp.where(band,3,jnp.where(doubled,2,1)),jnp.where(band,2,1)


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
