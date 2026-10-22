"""Device-native cost-shape placeholder, NOT PokeZero's production v4 encoder.

The champion's token sections/shapes/dtypes are represented, but the columns and
category IDs are deliberately NOT its vocabulary/feature mapping. Opposing
private state is visible. This must not feed strength evaluations or training;
it is only the handoff's encode/forward throughput workload.
"""
from typing import NamedTuple

import jax
import jax.numpy as jnp

from .state import bench_indices
from .transition import action_mask

TOKENS,NUMERIC,CATEGORICAL=23,132,41
TOKEN_TYPES=(0,)+(1,)*6+(2,)*6+(3,)*9+(5,)
IS_PLACEHOLDER=True


class Observation(NamedTuple):
    numeric_features: jax.Array   # [B,2,23,132] float32
    categorical_ids: jax.Array    # [B,2,23,41] int32; raw engine IDs, not v4 rows
    token_type_ids: jax.Array     # [B,2,23] int32
    attention_mask: jax.Array     # [B,2,23] bool
    legal_action_mask: jax.Array  # [B,2,9] bool


def encode(state,table):
    """Build both seat perspectives entirely on device, without RNG or mutation.

    Layout: field0, own party1..6, opposing party7..12, actions13..21,
    tendency22. The table-backed request mask includes current ability traps.
    No host callback, strings, per-battle Python objects, or production encoder.
    """
    b=state.hp.shape[0]
    mon_num=jnp.zeros((b,2,6,NUMERIC),jnp.float32)
    hp_fraction=state.hp/jnp.maximum(1,state.max_hp)
    mon_num=mon_num.at[...,0].set(hp_fraction)
    mon_num=mon_num.at[...,1].set(state.level/100)
    mon_num=mon_num.at[...,2:8].set(state.stats/1000)
    mon_num=mon_num.at[...,8:15].set(state.boosts/6)
    mon_num=mon_num.at[...,15:19].set(state.pp/64)
    mon_num=mon_num.at[...,19].set(jnp.arange(6)==state.active[...,None])
    mon_num=mon_num.at[...,20].set(state.status_counter/16)
    mon_num=mon_num.at[...,21].set(state.substitute_hp/jnp.maximum(1,state.max_hp))
    mon_num=mon_num.at[...,22].set(state.sleep_skipped/16)
    mon_num=mon_num.at[...,23].set(state.confusion_counter/5)
    mon_cat=jnp.zeros((b,2,6,CATEGORICAL),jnp.int32)
    for column,value in ((0,state.species),(1,state.item),(2,state.ability),(3,state.status)):
        mon_cat=mon_cat.at[...,column].set(value)
    mon_cat=mon_cat.at[...,4:6].set(state.types)
    mon_cat=mon_cat.at[...,6:10].set(state.moves)
    num=jnp.zeros((b,2,TOKENS,NUMERIC),jnp.float32)
    cat=jnp.zeros((b,2,TOKENS,CATEGORICAL),jnp.int32)
    num=num.at[:,:,1:7].set(mon_num).at[:,:,7:13].set(mon_num[:,::-1])
    cat=cat.at[:,:,1:7].set(mon_cat).at[:,:,7:13].set(mon_cat[:,::-1])
    num=num.at[:,:,0,0].set(state.turn[:,None]/100)
    num=num.at[:,:,0,1].set(state.spikes/3).at[:,:,0,2].set(state.spikes[:,::-1]/3)
    num=num.at[:,:,0,3].set(state.weather_counter[:,None]/5)
    num=num.at[:,:,0,4].set(state.done[:,None])
    cat=cat.at[:,:,0,0].set(state.weather[:,None])
    legal=action_mask(state,table)
    active=lambda x:jnp.take_along_axis(x,state.active[...,None,None],axis=2)[...,0,:]
    move_ids,pp=active(state.moves),active(state.pp)
    bench=bench_indices(state.active)
    target_ids=jnp.take_along_axis(state.species,bench,axis=2)
    target_hp=jnp.take_along_axis(hp_fraction,bench,axis=2)
    cat=cat.at[:,:,13:22,0].set(jnp.concatenate((move_ids,target_ids),axis=2))
    cat=cat.at[:,:,13:22,1].set(jnp.where(jnp.arange(9)<4,1,2))
    num=num.at[:,:,13:22,0].set(legal)
    num=num.at[:,:,13:22,1].set(jnp.arange(9)<4)
    num=num.at[:,:,13:22,2].set(jnp.arange(9)/8)
    num=num.at[:,:,13:17,3].set(pp/64)
    num=num.at[:,:,17:22,5].set(target_hp)
    num=num.at[:,:,22,0].set(jnp.sum(state.hp>0,axis=2)/6)
    num=num.at[:,:,22,1].set(jnp.sum(state.hp[:,::-1]>0,axis=2)/6)
    present=state.species>0
    attention=jnp.ones((b,2,TOKENS),jnp.bool_)
    attention=attention.at[:,:,1:7].set(present).at[:,:,7:13].set(present[:,::-1])
    types=jnp.broadcast_to(jnp.asarray(TOKEN_TYPES,jnp.int32),(b,2,TOKENS))
    return Observation(num,cat,types,attention,legal)
