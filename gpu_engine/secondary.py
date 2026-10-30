"""Selected Gen 3 self/target secondaries and Static, not full event coverage.

The caller supplies one fixed [0,100) roll and the original Substitute-hit
flag. Self effects survive Substitute damage and a target KO; target effects
do not. No chance is drawn conditionally here.
"""
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from . import mechanics as m
from .effects import from_metadata as effects_from_metadata
from .items import consume_lum
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
    "sacredfire": (50, m.BRN, False, False),
}

EXTRA_CHANCES = {"silverwind": 10, "ancientpower": 10, "mudshot": 100}


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
    speed_drop: jax.Array        # [M], selected target Speed secondary
    self_omni: jax.Array         # [M], five user stages; not target Dust/Sub gated
    synchronize: jax.Array       # [A], deterministic reflection of fresh status


def from_metadata(metadata):
    vocab = metadata["vocabulary"]
    rows = [EFFECTS.get(name, (EXTRA_CHANCES.get(name, 0), m.NONE, False, False)) for name in vocab["moves"]]
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
        jnp.asarray(contact), flags({'static'}),
        jnp.asarray([name == 'mudshot' for name in vocab['moves']]),
        jnp.asarray([name in {'silverwind', 'ancientpower'} for name in vocab['moves']]),
        flags({'synchronize'}))


def after_status_event(state, source, afflicted, status, table, *, applied, item_table=None, _status_trace=None):
    """Gen3 AfterSetStatus: reflect first, then consume each event's Lum.

    The caller has just installed a fresh status and supplies that exact mask.
    Synchronize uses no new chance and bypasses the source's Substitute. Toxic
    reflects as ordinary poison; sleep/freeze and self-infliction do not reflect.
    Existing type/ability immunity and a living, unstatused source still apply.
    """
    rows=jnp.arange(state.hp.shape[0])
    own,target=state.active[rows,source],state.active[rows,afflicted]
    reflected=jnp.where(status==m.TOX,m.PSN,status)
    reflects=(jnp.asarray(applied) & (source!=afflicted)
        & table.synchronize[state.ability[rows,afflicted,target]]
        & ((status==m.PAR)|(status==m.BRN)|(status==m.PSN)|(status==m.TOX))
        & (state.hp[rows,source,own]>0) & (state.status[rows,source,own]==m.NONE)
        & status_allowed(table,reflected,state.types[rows,source,own],state.ability[rows,source,own]))
    if _status_trace is not None:
        from . import engine_status_callbacks as callbacks
        _status_trace=callbacks.unknown(_status_trace,reflects)
    updates={}
    for name,value in [('status',reflected),('status_counter',0),('sleep_skipped',0),('rest_origin',False)]:
        old=getattr(state,name)
        updates[name]=old.at[rows,source,own].set(jnp.where(reflects,value,old[rows,source,own]))
    state=state._replace(**updates)
    if item_table is not None:
        # A nested reflection finishes before the original holder's Lum event.
        # Cure now, before a later contact event can inflict/reflect again.
        for side,changed in ((source,reflects),(afflicted,jnp.asarray(applied))):
            active=state.active[rows,side]
            get=lambda a:a[rows,side,active]
            old_item=get(state.item)
            result=consume_lum(item_table,old_item,get(state.status),get(state.status_counter),
                get(state.confusion_counter),get(state.hp),eligible=changed)
            consumed=result.item!=old_item
            if _status_trace is not None:
                _status_trace=callbacks.unknown(_status_trace,consumed & (get(state.status)!=m.NONE))
            updates={}
            for name,value in zip(('item','status','status_counter','confusion_counter'),result):
                updates[name]=getattr(state,name).at[rows,side,active].set(value)
            for name in ('sleep_skipped','rest_origin'):
                old=getattr(state,name)
                updates[name]=old.at[rows,side,active].set(jnp.where(consumed,jnp.zeros_like(get(old)),get(old)))
            state=state._replace(**updates)
    if _status_trace is not None:return state,_status_trace
    return state


def apply_contact(state, side, move, roll, table, *, eligible, item_table=None, _status_trace=None):
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
    if _status_trace is not None:
        from . import engine_status_callbacks as callbacks
        _status_trace=callbacks.unknown(_status_trace,proc)
    state=state._replace(
        status=state.status.at[rows, side, own].set(jnp.where(proc, m.PAR, state.status[rows, side, own])),
        status_counter=state.status_counter.at[rows, side, own].set(
            jnp.where(proc, 0, state.status_counter[rows, side, own])))
    if _status_trace is not None:
        return after_status_event(state,foe,side,jnp.full(state.hp.shape[0],m.PAR),table,
            applied=proc,item_table=item_table,_status_trace=_status_trace)
    return after_status_event(state,foe,side,jnp.full(state.hp.shape[0],m.PAR),table,
        applied=proc,item_table=item_table)


def status_allowed(table, status, types, ability, *, effective_weather=0):
    immune = jnp.any(table.type_immunity[status[:, None], types], axis=1)
    immune |= table.ability_immunity[ability, status]
    immune |= (status == m.FRZ) & (jnp.asarray(effective_weather) == SUN)
    return ~immune


def apply(state, side, move, roll, table, *, eligible, effective_weather=0, substitute_hit=False,
          item_table=None, _status_trace=None):
    rows = jnp.arange(state.hp.shape[0]); foe = 1 - side
    own, target = state.active[rows, side], state.active[rows, foe]
    get = lambda value: value[rows, foe, target]
    ability = get(state.ability)
    chance = table.chance[move] * jnp.where(table.grace[state.ability[rows, side, own]], 2, 1)
    trigger = jnp.asarray(eligible) & (jnp.asarray(roll) < chance)
    proc = (trigger & (get(state.hp) > 0) & ~table.dust[ability]
        & ~jnp.asarray(substitute_hit))
    status = table.status[move]
    changes = (proc & (status != m.NONE) & (get(state.status) == m.NONE)
        & status_allowed(table, status, get(state.types), ability, effective_weather=effective_weather))
    if _status_trace is not None:
        from . import engine_status_callbacks as callbacks
        _status_trace=callbacks.unknown(_status_trace,changes)
    boosts = get(state.boosts)
    drops = proc & table.spd_drop[move] & ~table.clear_body[ability]
    boosts = boosts.at[:, 3].set(jnp.where(drops, jnp.maximum(-6, boosts[:, 3] - 1), boosts[:, 3]))
    speed = proc & table.speed_drop[move] & ~table.clear_body[ability]
    boosts = boosts.at[:, 4].set(jnp.where(speed, jnp.maximum(-6, boosts[:, 4] - 1), boosts[:, 4]))
    self_proc = trigger & table.self_omni[move] & (state.hp[rows, side, own] > 0)
    own_boosts = state.boosts[rows, side, own]
    own_boosts = own_boosts.at[:, :5].set(jnp.where(self_proc[:, None],
        jnp.clip(own_boosts[:, :5] + 1, -6, 6), own_boosts[:, :5]))
    flinch = (proc & table.flinch[move] & ~table.inner_focus[ability]
        & ((get(state.volatile) & FOCUS_PUNCH)==0))
    state=state._replace(
        status=state.status.at[rows, foe, target].set(jnp.where(changes, status, get(state.status))),
        status_counter=state.status_counter.at[rows, foe, target].set(
            jnp.where(changes, 0, get(state.status_counter))),
        boosts=state.boosts.at[rows, foe, target].set(boosts).at[rows, side, own].set(own_boosts),
        volatile=state.volatile.at[rows, foe, target].set(
            get(state.volatile) | jnp.where(flinch, jnp.uint32(FLINCH), jnp.uint32(0))))
    if _status_trace is not None:
        return after_status_event(state,side,foe,status,table,applied=changes,item_table=item_table,_status_trace=_status_trace)
    return after_status_event(state,side,foe,status,table,applied=changes,item_table=item_table)
