"""Selected Gen 3 target secondaries and Static, not full event coverage.

The caller supplies one fixed [0,100) roll and excludes Substitute damage even
when that hit broke the substitute. No chance is drawn conditionally here.
"""
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from . import mechanics as m
from .effects import from_metadata as effects_from_metadata
from .state import FLINCH,FOCUS_PUNCH
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
    contact: jax.Array           # [M], pinned catalog flag, not damage category
    static: jax.Array            # [A], Gen3 one-third contact paralysis


def from_metadata(metadata):
    vocab = metadata["vocabulary"]
    rows = [EFFECTS.get(name, (0, m.NONE, False, False)) for name in vocab["moves"]]
    flags = lambda names: jnp.asarray([name in names for name in vocab["abilities"]])
    immunity = np.zeros((len(vocab["abilities"]), 7), bool)
    blocked = {"limber": (m.PAR,), "magmaarmor": (m.FRZ,),
        "waterveil": (m.BRN,), "immunity": (m.PSN, m.TOX)}
    for index, name in enumerate(vocab["abilities"]):
        for status in blocked.get(name, ()): immunity[index, status] = True
    move_rows = metadata.get('moves')
    if move_rows is not None and len(move_rows) != len(vocab['moves']):
        raise ValueError('move metadata/vocabulary mismatch')
    contact = [bool(row and row.get('flags', {}).get('contact')) for row in move_rows] \
        if move_rows is not None else [False]*len(vocab['moves'])
    return SecondaryTable(jnp.asarray([row[0] for row in rows], jnp.int32),
        jnp.asarray([row[1] for row in rows], jnp.int32),
        jnp.asarray([row[2] for row in rows]), jnp.asarray([row[3] for row in rows]),
        flags({"serenegrace"}), flags({"shielddust"}), flags({"clearbody", "whitesmoke"}),
        flags({"innerfocus"}), jnp.asarray(immunity), effects_from_metadata(metadata).status_immunity,
        jnp.asarray(contact), flags({'static'}))


def apply_contact(state, side, move, roll, table, *, eligible):
    """Static's Gen3 onDamagingHit, after damage/secondaries, before Lum.

    The existing uniform [0,60) auxiliary draw gives exactly 20/60 = 1/3.
    It is shared with phazing only across mutually exclusive dispatched moves:
    the pinned Gen3 Roar/Whirlwind have no contact flag. A source plan rejects
    a table claiming both consumers for one move. Saved phazing draws already
    survive interrupted turns. No percentile approximation or extra key draw.

    Caller excludes original Substitute damage (even if it broke), misses,
    immunity, Protect and zero damage. A KO of the Static holder still triggers;
    a fainted attacking source cannot receive status. Shield Dust and Serene
    Grace do not affect this ability event; Limber and existing status do.
    """
    rows = jnp.arange(state.hp.shape[0]); foe = 1-side
    own, target = state.active[rows, side], state.active[rows, foe]
    ability = state.ability[rows, side, own]
    proc = (jnp.asarray(eligible) & table.contact[move]
        & table.static[state.ability[rows, foe, target]] & (jnp.asarray(roll) < 20)
        & (state.hp[rows, side, own] > 0) & (state.status[rows, side, own] == m.NONE)
        & ~table.ability_immunity[ability, m.PAR])
    return state._replace(
        status=state.status.at[rows, side, own].set(jnp.where(proc, m.PAR, state.status[rows, side, own])),
        status_counter=state.status_counter.at[rows, side, own].set(
            jnp.where(proc, 0, state.status_counter[rows, side, own])))


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
    flinch = (proc & table.flinch[move] & ~table.inner_focus[ability]
        & ((get(state.volatile) & FOCUS_PUNCH)==0))
    return state._replace(
        status=state.status.at[rows, foe, target].set(jnp.where(changes, status, get(state.status))),
        status_counter=state.status_counter.at[rows, foe, target].set(
            jnp.where(changes, 0, get(state.status_counter))),
        boosts=state.boosts.at[rows, foe, target].set(boosts),
        volatile=state.volatile.at[rows, foe, target].set(
            get(state.volatile) | jnp.where(flinch, jnp.uint32(FLINCH), jnp.uint32(0))))
