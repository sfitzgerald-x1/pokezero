"""Selected Gen 3 TryHit abilities, not general ability/event coverage.

The caller resolves accuracy, Protect and whether a move targets a Pokemon.
These handlers precede Substitute, and immunity still applies at full HP.
"""
from typing import NamedTuple

import jax
import jax.numpy as jnp

from . import mechanics as m

FLASH_FIRE = 2  # volatile bit 1; ordinary turns preserve it, switching clears it


class AbsorptionTable(NamedTuple):
    water_absorb: jax.Array
    volt_absorb: jax.Array
    flash_fire: jax.Array
    water_type: jax.Array
    electric_type: jax.Array
    fire_type: jax.Array
    thunderwave: jax.Array
    willowisp: jax.Array
    fire_thaws: jax.Array


def from_metadata(metadata):
    vocab = metadata["vocabulary"]
    flag = lambda key,name: jnp.asarray([value == name for value in vocab[key]])
    rows = metadata["moves"]
    if len(rows) != len(vocab["moves"]):
        raise ValueError("move metadata/vocabulary mismatch")
    # Gen 3 thaw uses the move's original catalog type, not its dynamic type:
    # Hidden Power Fire and Fire Weather Ball do NOT thaw the target.
    fire_thaws = jnp.asarray([bool(row and row["source_type"] == "Fire"
        and row["category"] != "Status") for row in rows])
    return AbsorptionTable(flag("abilities","waterabsorb"),flag("abilities","voltabsorb"),
        flag("abilities","flashfire"),jnp.asarray(vocab["types"].index("water")),
        jnp.asarray(vocab["types"].index("electric")),jnp.asarray(vocab["types"].index("fire")),
        flag("moves","thunderwave"),flag("moves","willowisp"),fire_thaws)


class AbsorptionResult(NamedTuple):
    blocked: jax.Array
    hp: jax.Array
    volatile: jax.Array


def try_hit(table,ability,move,move_type,status,types,hp,max_hp,substitute_hp,volatile,*,eligible):
    eligible = jnp.asarray(eligible) & (hp > 0)
    heals = eligible & ((table.water_absorb[ability] & (move_type == table.water_type))
        | (table.volt_absorb[ability] & (move_type == table.electric_type) & ~table.thunderwave[move]))
    failed_wisp = table.willowisp[move] & ((status != m.NONE) | (substitute_hp > 0)
        | jnp.any(types == table.fire_type,axis=-1))
    flashes = (eligible & table.flash_fire[ability] & (move_type == table.fire_type)
        & (status != m.FRZ) & ~failed_wisp)
    healed = jnp.minimum(max_hp,hp + jnp.maximum(1,max_hp // 4))
    return AbsorptionResult(heals | flashes,jnp.where(heals,healed,hp),
        jnp.where(flashes,volatile | jnp.uint32(FLASH_FIRE),volatile))


def damage_phase1(table,move_type,volatile):
    active = (move_type == table.fire_type) & ((volatile & FLASH_FIRE) != 0)
    return jnp.where(active,3,1),jnp.where(active,2,1)
