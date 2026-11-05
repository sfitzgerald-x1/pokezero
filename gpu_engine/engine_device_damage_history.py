"""Original v4 previous-public-turn damage meaning, from actual HP writes.

No protocol/event tape or admission is built. Exact victim fractions remain
private to that victim's owning readout; the other seat sees shown-percent
losses. Current accumulators settle only when the caller opens a public TURN.
"""
from typing import NamedTuple
import jax
import jax.numpy as j


class History(NamedTuple):
    hp: object                   # [B,2] previous actually observed active HP
    maximum: object              # [B,2] previous owning denominator
    taken: object                # [B,2,2,2]: side,(shown,exact),(high,low)
    dealt: object                # same, explicit ordinary-hit attribution only


def initialize(state):
    r=j.arange(len(state.hp))[:,None];s=j.arange(2)[None,:]
    hp=state.hp[r,s,state.active];maximum=state.max_hp[r,s,state.active]
    zero=j.zeros((len(state.hp),2,2,2),j.float32)
    return History(hp,maximum,zero,zero)


def _add(pair,value):
    # Original public_damage._add TwoSum algebra: device float32, without
    # long-prefix drift or dependence on JAX x64. No validity is cached.
    high,low=pair[...,0],pair[...,1]
    total=jax.lax.optimization_barrier(high+value)
    recovered=jax.lax.optimization_barrier(total-high)
    error=jax.lax.optimization_barrier((high-(total-recovered))+(value-recovered))
    correction=jax.lax.optimization_barrier(low+error)
    new_high=jax.lax.optimization_barrier(total+correction)
    new_low=jax.lax.optimization_barrier(correction-(new_high-total))
    return j.where((value>0)[...,None],j.stack((new_high,new_low),-1),pair)


def arrive(history,state,side,use):
    r=j.arange(len(state.hp));p=state.active[r,side]
    hp=history.hp.at[r,side].set(j.where(use,state.hp[r,side,p],history.hp[r,side]))
    maximum=history.maximum.at[r,side].set(j.where(use,state.max_hp[r,side,p],history.maximum[r,side]))
    def clear(value):return value.at[r,side].set(j.where(use[:,None,None],0,value[r,side]))
    return History(hp,maximum,clear(history.taken),clear(history.dealt))


def observe(history,state,use,percent,*,record_damage=True):
    r=j.arange(len(state.hp))[:,None];s=j.arange(2)[None,:]
    hp=state.hp[r,s,state.active];maximum=state.max_hp[r,s,state.active]
    exact=j.maximum(0,history.hp-hp).astype(j.float32)/j.maximum(history.maximum,1)
    shown=j.maximum(0,percent(history.hp,history.maximum)-percent(hp,maximum)).astype(j.float32)/100
    values=j.stack((shown,exact),-1)*j.asarray(record_damage,j.bool_)
    values=j.where(use[:,None,None],values,0)
    taken=_add(history.taken,values)
    return history._replace(hp=j.where(use[:,None],hp,history.hp),
        maximum=j.where(use[:,None],maximum,history.maximum),taken=taken)


def direct(history,side,shown,exact,use):
    r=j.arange(len(use));value=j.stack((shown,exact),-1)
    old=history.dealt[r,side]
    return history._replace(dealt=history.dealt.at[r,side].set(
        j.where(use[:,None,None],_add(old,value),old)))


def settle(history,use):
    """Return original previous-round values, then clear current rows only."""
    taken=j.sum(history.taken,axis=-1);dealt=j.sum(history.dealt,axis=-1)
    new=history._replace(taken=j.where(use[:,None,None,None],0,history.taken),
        dealt=j.where(use[:,None,None,None],0,history.dealt))
    return new,taken,dealt
