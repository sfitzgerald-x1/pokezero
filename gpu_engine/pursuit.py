"""Gen 3 singles Pursuit interception; not a generic early-move dispatcher.

Dark is always Special in Gen 3. Shared damage arithmetic and item/PP helpers
retain the selected interactions without running BeforeMove at all.
"""
from typing import NamedTuple

import jax
import jax.numpy as jnp

from . import items as held
from . import mechanics as m
from . import pressure as pp_cost


class PursuitTable(NamedTuple):
    pursuit: jax.Array


def from_metadata(metadata):
    return PursuitTable(jnp.asarray([name == "pursuit" for name in metadata["vocabulary"]["moves"]]))


def intercept(state, side, slot, move, chance, table, *, eligible):
    """Hit the outgoing target before switch clearing/entry, consuming one move.

    Caller snapshots beforeTurn registration (sleep/freeze exclude it), excludes
    forced replacements and retains the chosen switch after KO in Gen 3.
    Interception bypasses paralysis/confusion/flinch and their counters. Wider
    contact/item events outside the spike subset remain placeholders.
    """
    rows = jnp.arange(state.hp.shape[0])
    foe = 1-side
    own, target = state.active[:,side],state.active[:,foe]
    get = lambda values,seat,active:values[rows,seat,active]
    hp = get(state.hp,foe,target)
    used = jnp.asarray(eligible) & table.pursuit.pursuit[move] & (get(state.hp,side,own)>0) & (hp>0)
    attack,defense = get(state.stats,side,own),get(state.stats,foe,target)
    attack_boosts,defense_boosts = get(state.boosts,side,own),get(state.boosts,foe,target)
    typ = table.type[move]
    own_types,target_types = get(state.types,side,own),get(state.types,foe,target)
    num = table.type_num[typ,target_types[:,0]]*table.type_num[typ,target_types[:,1]]
    den = table.type_den[typ,target_types[:,0]]*table.type_den[typ,target_types[:,1]]
    attack_num,attack_den = (1,1)
    if table.items is not None:
        attack_num,attack_den = held.attack_modifier(table.items,get(state.item,side,own),
            get(state.species,side,own),False)
    damage = m.gen3_damage(level=get(state.level,side,own),power=80,
        attack=attack[:,3],defense=defense[:,4],attack_stage=attack_boosts[:,2],
        defense_stage=defense_boosts[:,3],physical=False,critical=chance[:,4]==0,
        attack_num=attack_num,attack_den=attack_den,
        stab=jnp.any(own_types==typ[:,None],axis=1),effectiveness_num=num,
        effectiveness_den=den,immune=num==0,roll=85+chance[:,5])
    # beingCalledBack makes accuracy unconditional, but Substitute still caps
    # actual damage and prevents spill into the outgoing Pokemon's HP.
    sub = get(state.substitute_hp,foe,target)
    dealt = jnp.where(used,jnp.minimum(jnp.where(sub>0,sub,hp),damage),0)
    after_hp = hp-jnp.where(sub>0,0,dealt)
    state = state._replace(hp=state.hp.at[rows,foe,target].set(after_hp),
        substitute_hp=state.substitute_hp.at[rows,foe,target].set(sub-jnp.where(sub>0,dealt,0)))
    cost = 1
    if table.pressure is not None:
        cost = pp_cost.cost(table.pressure,move,own_types,get(state.ability,foe,target),hp)
    pp = state.pp[rows,side,own,slot]
    state = state._replace(pp=state.pp.at[rows,side,own,slot].set(jnp.where(used,jnp.maximum(0,pp-cost),pp)))
    # The inherited interception handler locks only after successful useMove.
    # Keep this explicit even though Dark has no type immunity in Gen 3.
    successful = used & (dealt>0)
    if table.items is not None:
        old_lock = get(state.choice_move,side,own)
        lock = successful & table.items.choice_band[get(state.item,side,own)] & (old_lock==0)
        state = state._replace(choice_move=state.choice_move.at[rows,side,own].set(
            jnp.where(lock,move,old_lock)))
    return state,used,used & (after_hp==0)
