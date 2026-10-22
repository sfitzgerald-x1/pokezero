"""Selected fixed-damage and recoil events for Gen 3 singles.

Counter/Mirror Coat, variable base powers and other special hits remain
placeholders. These tables do not qualify complete move or game coverage.
"""
from typing import NamedTuple

import jax
import jax.numpy as jnp


class DamageMoveTable(NamedTuple):
    level_damage: jax.Array
    constant_damage: jax.Array
    recoil_den: jax.Array
    struggle: jax.Array
    rock_head: jax.Array


def from_metadata(metadata):
    vocab=metadata["vocabulary"]
    names=vocab["moves"]
    constants={"sonicboom":20,"dragonrage":40}
    recoil={"doubleedge":3,"volttackle":3,"takedown":4,"submission":4,"struggle":4}
    return DamageMoveTable(jnp.asarray([name in {"seismictoss","nightshade"} for name in names]),
        jnp.asarray([constants.get(name,0) for name in names]),
        jnp.asarray([recoil.get(name,0) for name in names]),
        jnp.asarray([name=="struggle" for name in names]),
        jnp.asarray([name=="rockhead" for name in vocab["abilities"]]))


def resolve_fixed(table,move,level,ordinary_damage,*,immune):
    fixed=table.level_damage[move] | (table.constant_damage[move]>0)
    amount=jnp.where(table.level_damage[move],level,table.constant_damage[move])
    return jnp.where(fixed,jnp.where(jnp.asarray(immune),0,amount),ordinary_damage)


def recoil_damage(table,move,ability,dealt):
    den=table.recoil_den[move]
    applies=(dealt>0) & (den>0) & (~table.rock_head[ability] | table.struggle[move])
    # Pinned Gen 3 calcRecoilDamage floors, unlike later generations' round.
    return jnp.where(applies,jnp.maximum(1,dealt//jnp.maximum(1,den)),0)
