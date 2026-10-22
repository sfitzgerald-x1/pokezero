"""Selected Gen 3 target secondaries, not general move-effect coverage.

The caller supplies one fixed [0,100) roll and excludes Substitute damage even
when that hit broke the substitute. No chance is drawn conditionally here.
"""
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from . import mechanics as m
from .effects import from_metadata as effects_from_metadata
from .state import FLINCH
from .weather import SUN


# chance, major status, SpD drop, flinch; pinned Gen 3 overrides matter (Crunch).
EFFECTS = {
    "rockslide": (30, m.NONE, False, True),
    "icebeam": (10, m.FRZ, False, False),
    "icepunch": (10, m.FRZ, False, False),
    "thunderbolt": (10, m.PAR, False, False),
    "thunderpunch": (10, m.PAR, False, False),
    "thunder": (30, m.PAR, False, False),
    "bodyslam": (30, m.PAR, False, False),
    "sludgebomb": (30, m.PSN, False, False),
    "fireblast": (10, m.BRN, False, False),
    "flamethrower": (10, m.BRN, False, False),
    "firepunch": (10, m.BRN, False, False),
    "shadowball": (20, m.NONE, True, False),
    "crunch": (20, m.NONE, True, False),
    "psychic": (10, m.NONE, True, False),
}


class SecondaryTable(NamedTuple):
    chance: jax.Array
    status: jax.Array
    spd_drop: jax.Array
    flinch: jax.Array
    grace: jax.Array
    dust: jax.Array
    clear_body: jax.Array
    inner_focus: jax.Array
    ability_immunity: jax.Array   # [A,7], selected status-admission abilities
    type_immunity: jax.Array


def from_metadata(metadata):
    vocab = metadata["vocabulary"]
    rows = [EFFECTS.get(name, (0, m.NONE, False, False)) for name in vocab["moves"]]
    flags = lambda names: jnp.asarray([name in names for name in vocab["abilities"]])
    immunity = np.zeros((len(vocab["abilities"]), 7), bool)
    blocked = {"limber": (m.PAR,), "magmaarmor": (m.FRZ,),
        "waterveil": (m.BRN,), "immunity": (m.PSN, m.TOX)}
    for index, name in enumerate(vocab["abilities"]):
        for status in blocked.get(name, ()): immunity[index, status] = True
    return SecondaryTable(jnp.asarray([row[0] for row in rows], jnp.int32),
        jnp.asarray([row[1] for row in rows], jnp.int32),
        jnp.asarray([row[2] for row in rows]), jnp.asarray([row[3] for row in rows]),
        flags({"serenegrace"}), flags({"shielddust"}), flags({"clearbody", "whitesmoke"}),
        flags({"innerfocus"}), jnp.asarray(immunity), effects_from_metadata(metadata).status_immunity)


def status_allowed(table, status, types, ability, *, effective_weather=0):
    immune = jnp.any(table.type_immunity[status[:, None], types], axis=1)
    immune |= table.ability_immunity[ability, status]
    immune |= (status == m.FRZ) & (jnp.asarray(effective_weather) == SUN)
    return ~immune


def apply(state, side, move, roll, table, *, eligible, effective_weather=0):
    rows = jnp.arange(state.hp.shape[0]); foe = 1 - side
    own, target = state.active[rows, side], state.active[rows, foe]
    get = lambda value: value[rows, foe, target]
    ability = get(state.ability)
    chance = table.chance[move] * jnp.where(table.grace[state.ability[rows, side, own]], 2, 1)
    proc = (jnp.asarray(eligible) & (get(state.hp) > 0) & ~table.dust[ability]
        & (jnp.asarray(roll) < chance))
    status = table.status[move]
    changes = (proc & (status != m.NONE) & (get(state.status) == m.NONE)
        & status_allowed(table, status, get(state.types), ability, effective_weather=effective_weather))
    boosts = get(state.boosts)
    drops = proc & table.spd_drop[move] & ~table.clear_body[ability]
    boosts = boosts.at[:, 3].set(jnp.where(drops, jnp.maximum(-6, boosts[:, 3] - 1), boosts[:, 3]))
    flinch = proc & table.flinch[move] & ~table.inner_focus[ability]
    return state._replace(
        status=state.status.at[rows, foe, target].set(jnp.where(changes, status, get(state.status))),
        status_counter=state.status_counter.at[rows, foe, target].set(
            jnp.where(changes, 0, get(state.status_counter))),
        boosts=state.boosts.at[rows, foe, target].set(boosts),
        volatile=state.volatile.at[rows, foe, target].set(
            get(state.volatile) | jnp.where(flinch, jnp.uint32(FLINCH), jnp.uint32(0))))
