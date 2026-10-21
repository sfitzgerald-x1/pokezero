"""Selected Gen 3 combat and ability-trapping rules for the sampled spike.

Move traps, Baton Pass, ability replacement/suppression and the wider ability
surface remain separate mechanics. IDs come from the accepted pool vocabulary.
"""
from typing import NamedTuple

import jax
import jax.numpy as jnp

from . import mechanics as m


class CombatTable(NamedTuple):
    guts: jax.Array
    thick_fat: jax.Array
    arena_trap: jax.Array
    magnet_pull: jax.Array
    shadow_tag: jax.Array
    levitate: jax.Array
    flying_type: jax.Array
    steel_type: jax.Array
    fire_type: jax.Array
    ice_type: jax.Array


def from_metadata(metadata):
    vocab = metadata["vocabulary"]
    flag = lambda name:jnp.asarray([value == name for value in vocab["abilities"]])
    typ = lambda name:jnp.asarray(vocab["types"].index(name))
    return CombatTable(flag("guts"),flag("thickfat"),flag("arenatrap"),flag("magnetpull"),
        flag("shadowtag"),flag("levitate"),typ("flying"),typ("steel"),typ("fire"),typ("ice"))


def attack_modifier(table,ability,status,physical,*,item_num=1,item_den=1):
    """One Atk-event chain, not sequentially rounded Guts/item damage.

The selected multipliers are exact Q12 fractions (1, 1.5, 2). Their rational
product is the same chain as the pinned event; damage applies it once to Atk.
"""
    activated = table.guts[ability] & (status != m.NONE) & jnp.asarray(physical)
    return item_num*jnp.where(activated,3,1),item_den*jnp.where(activated,2,1)


def base_power(table,ability,move_type,power):
    # Gen 3 Thick Fat modifies BasePower, not Atk/SpA or final damage.
    halved = table.thick_fat[ability] & ((move_type == table.fire_type) | (move_type == table.ice_type))
    return m.modify(power,1,jnp.where(halved,2,1))


def switch_trapped(table,ability,types,hp):
    """Resolved active singles ability traps, shaped [B,2].

    Gen 3 Shadow Tag traps other Shadow Tag holders too; Ghost immunity to
    trapping is later-generation behavior. The caller permits forced switches.
    """
    foe = ability[:,::-1]
    grounded = ~jnp.any(types == table.flying_type,axis=-1) & ~table.levitate[ability]
    steel = jnp.any(types == table.steel_type,axis=-1)
    affected = (table.shadow_tag[foe] | (table.magnet_pull[foe] & steel)
        | (table.arena_trap[foe] & grounded))
    return affected & (hp > 0) & (hp[:,::-1] > 0)
