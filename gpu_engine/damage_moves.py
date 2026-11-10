"""Selected fixed-damage, recoil, drain and retaliation events for Gen 3 singles.

Variable base powers and other special hits remain placeholders. These tables
do not qualify complete move or game coverage. Fidelity A2 adds Giga Drain's
drain with Liquid Ooze, and Counter/Mirror Coat (see transition.py).
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
    # Fidelity patch A2 (appended; existing fields keep their position and value).
    drain_den: jax.Array      # [M], move.drain [1,den]: Giga Drain (data/moves.ts drain:[1,2])
    liquid_ooze: jax.Array    # [A], gen4 abilities.ts liquidooze onSourceTryHeal (drain -> damage)
    counter_kind: jax.Array   # [M], 1 Counter, 2 Mirror Coat (gen3 moves.ts counter/mirrorcoat)
    hidden_power: jax.Array   # [M], counted by Counter, never by Mirror Coat (gen3 onDamage)


def from_metadata(metadata):
    vocab=metadata["vocabulary"]
    names=vocab["moves"]
    constants={"sonicboom":20,"dragonrage":40}
    recoil={"doubleedge":3,"volttackle":3,"takedown":4,"submission":4,"struggle":4}
    return DamageMoveTable(jnp.asarray([name in {"seismictoss","nightshade"} for name in names]),
        jnp.asarray([constants.get(name,0) for name in names]),
        jnp.asarray([recoil.get(name,0) for name in names]),
        jnp.asarray([name=="struggle" for name in names]),
        jnp.asarray([name=="rockhead" for name in vocab["abilities"]]),
        jnp.asarray([{"gigadrain":2}.get(name,0) for name in names],jnp.int32),
        jnp.asarray([name=="liquidooze" for name in vocab["abilities"]]),
        jnp.asarray([{"counter":1,"mirrorcoat":2}.get(name,0) for name in names],jnp.int32),
        jnp.asarray([name.startswith("hiddenpower") for name in names]))


def resolve_fixed(table,move,level,ordinary_damage,*,immune):
    fixed=table.level_damage[move] | (table.constant_damage[move]>0)
    amount=jnp.where(table.level_damage[move],level,table.constant_damage[move])
    return jnp.where(fixed,jnp.where(jnp.asarray(immune),0,amount),ordinary_damage)


def recoil_damage(table,move,ability,dealt):
    den=table.recoil_den[move]
    applies=(dealt>0) & (den>0) & (~table.rock_head[ability] | table.struggle[move])
    # Pinned Gen 3 calcRecoilDamage floors, unlike later generations' round.
    return jnp.where(applies,jnp.maximum(1,dealt//jnp.maximum(1,den)),0)


def drain_amount(dealt,hits_sub):
    """Gen 3 drain [1,2] from the damage actually dealt. On the Pokemon: spreadDamage
    (sim/battle.ts, gen<=4) heals clampIntRange(floor(dealt/2),1); on a Substitute the
    gen4 substitute condition (data/mods/gen4/moves.ts) heals ceil(sub damage/2)."""
    return jnp.where(hits_sub,(dealt+1)//2,jnp.maximum(1,dealt//2))


def counter_damage(taken):
    """damageCallback: volatiles[counter|mirrorcoat].damage || 1, the recorded 2x damage."""
    return jnp.maximum(1,2*taken)
