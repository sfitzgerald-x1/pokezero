"""Source public-event ledgers for existing v4 field/tendency descriptors.

No sampled/private opponent history or future selected action is an input.
Truant is anchored by native public species/actual MOVE/CANT, not engine clock.
Spikes stores shown and owning-exact victim fractions separately.
"""
from typing import NamedTuple
import jax
import jax.numpy as j
from pokezero.dex import normalize_id
from .engine_device_damage_history import _add


class Tables(NamedTuple):
    native_truant:object
    truant:object


def compile_tables(metadata):
    v=metadata['vocabulary']
    return Tables(j.asarray([normalize_id(n)in ('slaking','slakoth')for n in v['species']],bool),
        j.asarray([normalize_id(n)=='truant'for n in v['abilities']],bool))


class State(NamedTuple):
    holder:object            # B,inducing side -> victim public slot, -1 empty
    victims:object           # B,victim side,public slot; ability sleep included
    known:object             # B,side active Truant phase known
    loaf:object
    traced:object            # current publicly copied Truant holder
    skip_flip:object
    post_upkeep:object
    hazards:object           # B,side,(shown,exact),(high,low); exact owning only
    blocked:object           # B,attacking side Protect/Detect-blocked windows
    pursuit:object           # B,attacking side actual public Pursuit activation
    weather:object           # B,setter side,(rain,sun,sand),(seen,ability)


def initialize(batch):
    pair=j.zeros((batch,2),bool)
    return State(j.full((batch,2),-1,j.int32),j.zeros((batch,2,6),bool),pair,pair,pair,pair,j.zeros(batch,bool),
        j.zeros((batch,2,2,2),j.float32),j.zeros((batch,2),j.int32),j.zeros((batch,2),j.int32),j.zeros((batch,2,3,2),bool))


def _seat(side,b):return j.broadcast_to(j.asarray(side,j.int32),(b,))
def _set(old,side,value,use):
    r=j.arange(len(use));s=_seat(side,len(use));before=old[r,s]
    mask=use.reshape(use.shape+(1,)*(before.ndim-1))
    return old.at[r,s].set(j.where(mask,value,before))


def arrival(t,tables,state,side,use):
    h=t.public_port
    if h is None:return t
    r=j.arange(len(use));s=_seat(side,len(use));p=state.active[r,s]
    species=state.species[r,s,p]
    native=j.zeros_like(use)if tables is None else (species>0)&(species<len(tables.native_truant))&tables.native_truant[j.clip(species,0,len(tables.native_truant)-1)]
    h=h._replace(known=_set(h.known,s,native,use),loaf=_set(h.loaf,s,native&(t.turn!=0),use),
        traced=_set(h.traced,s,j.zeros_like(use),use),skip_flip=_set(h.skip_flip,s,native&h.post_upkeep,use))
    return t._replace(public_port=h)


def anchor(t,side,use,*,loaf=False):
    h=t.public_port
    if h is None:return t
    r=j.arange(len(use));s=_seat(side,len(use));holder=h.known[r,s]|h.traced[r,s]
    yes=use&holder
    return t._replace(public_port=h._replace(known=_set(h.known,s,j.ones_like(use),yes),loaf=_set(h.loaf,s,j.full_like(use,loaf),yes)))


def trace(t,tables,side,ability,use):
    h=t.public_port
    if h is None or tables is None:return t
    is_truant=(ability>0)&(ability<len(tables.truant))&tables.truant[j.clip(ability,0,len(tables.truant)-1)]
    return t._replace(public_port=h._replace(known=_set(h.known,side,j.zeros_like(use),use),loaf=_set(h.loaf,side,j.zeros_like(use),use),
        traced=_set(h.traced,side,is_truant,use),skip_flip=_set(h.skip_flip,side,j.zeros_like(use),use)))


def upkeep(t,use):
    h=t.public_port
    return t if h is None else t._replace(public_port=h._replace(post_upkeep=h.post_upkeep|use))


def turn(t,use,number):
    h=t.public_port
    if h is None:return t
    flip=use[:,None]&(number[:,None]>=2)&h.known&~h.skip_flip
    return t._replace(public_port=h._replace(loaf=j.where(flip,~h.loaf,h.loaf),skip_flip=j.where(use[:,None],False,h.skip_flip),
        post_upkeep=j.where(use,False,h.post_upkeep)))


def condition(t,side,physical,old,new,use,*,rest=False,ability_source=False,faint=False):
    h=t.public_port
    if h is None:return t
    r=j.arange(len(use));s=_seat(side,len(use));slot=t.physical_to_public[r,s,j.clip(physical,0,5)]
    yes=use&(slot>=0)&(slot<6);at=j.clip(slot,0,5);induced=yes&(old!=5)&(new==5)&~j.asarray(rest)
    holder=_set(h.holder,1-s,at,induced&~j.asarray(ability_source))
    previous=h.victims[r,s,at]
    victims=h.victims.at[r,s,at].set(j.where(induced,True,previous))
    cure=yes&((old!=new)&(new==0)|j.asarray(faint))
    held=holder[r,1-s];holder=_set(holder,1-s,j.full_like(at,-1),cure&(held==slot))
    clear=yes&(((old==5)&(new==0))|j.asarray(faint))
    victims=victims.at[r,s,at].set(j.where(clear,False,victims[r,s,at]))
    return t._replace(public_port=h._replace(holder=holder,victims=victims))


def team_cure(t,side,use):
    h=t.public_port
    if h is None:return t
    s=_seat(side,len(use));r=j.arange(len(use))
    held=h.holder[r,1-s];at=j.clip(held,0,5)
    living=(held>=0)&~t.fainted[r,s,at]&(t.status[r,s,at]!=0)
    return t._replace(public_port=h._replace(holder=_set(h.holder,1-s,j.full_like(held,-1),use&living),
        victims=_set(h.victims,s,j.zeros_like(h.victims[r,s]),use)))


def spikes(t,before,after,side,use,percent):
    h=t.public_port
    if h is None:return t
    r=j.arange(len(use));s=_seat(side,len(use));p=before.active[r,s]
    hp=before.hp[r,s,p];maximum=before.max_hp[r,s,p];new=after.hp[r,s,p]
    yes=use&(maximum>0)&(hp>new)&(hp>=0)&(hp<=maximum)&(new>=0)
    shown=j.maximum(0,percent(hp,maximum)-percent(new,maximum)).astype(j.float32)/100
    exact=j.maximum(0,hp-new).astype(j.float32)/j.maximum(maximum,1)
    value=j.where(yes[:,None],j.stack((shown,exact),-1),0)
    old=h.hazards[r,s];new=_add(old,value)
    return t._replace(public_port=h._replace(hazards=_set(h.hazards,s,new,yes)))


def count(t,side,use,name):
    h=t.public_port
    if h is None:return t
    s=_seat(side,len(use));r=j.arange(len(use));old=getattr(h,name)
    return t._replace(public_port=h._replace(**{name:_set(old,s,old[r,s]+1,use)}))


def protect(t,side,acts,accuracy_pass,protected,protectable):
    # Effective Gen3 scripts.ts: Accuracy passes BEFORE TryHit/Protect.
    # Natural type immunity is reported only after TryHit, so is not a gate.
    return count(t,side,acts&accuracy_pass&protected&protectable,'blocked')


def weather(t,side,kind,use,*,ability=False):
    h=t.public_port
    if h is None:return t
    r=j.arange(len(use));s=_seat(side,len(use));at=j.clip(kind-1,0,2);yes=use&(kind>=1)&(kind<=3)
    old=h.weather[r,s,at];value=old|j.stack((yes,yes&j.asarray(ability)),-1)
    mask=yes[:,None];out=h.weather.at[r,s,at].set(j.where(mask,value,old))
    return t._replace(public_port=h._replace(weather=out))


def redact_for_owner(state,owner):
    """D reply neural prior exact hazard history is unavailable, never borrowed."""
    if state is None:return None
    return state._replace(hazards=state.hazards.at[:,1-owner,1,:].set(0))
