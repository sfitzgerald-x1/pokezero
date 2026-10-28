"""Public damage attribution and retained v4 damage/item credit columns.

Read old HP before the condition update. This is observation bookkeeping, not
battle simulation. Separate compensated float32 sums avoid long-prefix drift
without enabling device float64. Hazard expected exposure is a separate writer.
"""
import hashlib
import json
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from pokezero import showdown as reference
from .private_team import vocabulary_binding
from .public_abilities import _bool
from .public_field import FieldFeaturePatch
from .public_mon_state import _int32

SCHEMA='pokezero.public_damage.v1'
ARRIVE,TURN,MOVE,CLOSE,REMOVED,DAMAGE=range(6)
MON_CORE=('LAST_DAMAGE_DEALT','LAST_DAMAGE_TAKEN')
FIELD_CORE=('SELF_HAZARD_CREDIT','OPP_HAZARD_CREDIT','SELF_ITEMS_REMOVED_CREDIT','OPP_ITEMS_REMOVED_CREDIT')
MON_COLUMNS=tuple(reference.v4_numeric_index(getattr(reference,'NUMERIC_'+n)) for n in MON_CORE)
FIELD_COLUMNS=tuple(reference.v4_numeric_index(getattr(reference,'NUMERIC_'+n)) for n in FIELD_CORE)
SUM_FIELDS=('current_dealt','current_taken','last_dealt','last_taken','hazard_suffered')


class DamageTables(NamedTuple):
    vocabulary_binding: jax.Array
    binding: jax.Array


def compile_damage(vocabulary):
    vb=vocabulary_binding(vocabulary)
    payload=dict(schema=SCHEMA,vocabulary=np.asarray(vb).tolist(),mon=MON_COLUMNS,field=FIELD_COLUMNS,
        team_size=reference._TEAM_SIZE,tags=['[from] Spikes','[from] move: Knock Off'])
    raw=hashlib.sha256(json.dumps(payload,sort_keys=True,separators=(',',':')).encode()).digest()
    return DamageTables(vb,jnp.asarray(np.frombuffer(raw,dtype='>u4').astype(np.uint32)))


class DamageState(NamedTuple):
    binding: jax.Array
    species_binding: jax.Array
    turn: jax.Array
    window_actor: jax.Array      # public side, -1 no open move window
    current_dealt: jax.Array    # each sum [B,2,2]: public side by high/low parts
    current_taken: jax.Array
    last_dealt: jax.Array
    last_taken: jax.Array
    hazard_suffered: jax.Array
    items_removed: jax.Array    # victim-side counts, not giver-side counts


def _binding(value):
    if getattr(value,'shape',None)!=(8,) or getattr(value,'dtype',None)!=jnp.dtype(jnp.uint32):
        raise ValueError('damage inputs require eight uint32 namespace words')


def initialize(batch,tables,*,species_binding):
    if type(batch) is not int or batch<=0:raise ValueError('positive damage batch required')
    _binding(species_binding)
    z=jnp.zeros(batch,jnp.int32);s=jnp.zeros((batch,2,2),jnp.float32)
    return DamageState(jnp.broadcast_to(tables.binding,(batch,8)),jnp.broadcast_to(species_binding,(batch,8)),
        z,jnp.full(batch,-1,jnp.int32),s,s,s,s,s,jnp.zeros((batch,2),jnp.int32))


def _shape(state,mons):
    if state.turn.ndim!=1 or not state.turn.shape[0]:raise ValueError('nonempty damage batch required')
    batch=state.turn.shape[0]
    for name in state._fields:
        a=getattr(state,name)
        shape=(batch,8) if name in ('binding','species_binding') else (batch,2,2) if name in SUM_FIELDS else (batch,2) if name=='items_removed' else (batch,)
        dtype=jnp.uint32 if name in ('binding','species_binding') else jnp.float32 if name in SUM_FIELDS else jnp.int32
        if a.shape!=shape or a.dtype!=dtype:raise ValueError('damage state shape or dtype: '+name)
    for name,shape,dtype in (('species_binding',(batch,8),jnp.uint32),('active',(batch,2),jnp.int32),
        ('encounter',(batch,2,6),jnp.int32),('hp_numerator',(batch,2,6),jnp.int32),
        ('hp_denominator',(batch,2,6),jnp.int32),('hp_known',(batch,2,6),jnp.bool_)):
        a=getattr(mons,name)
        if a.shape!=shape or a.dtype!=dtype:raise ValueError('damage mon shape or dtype: '+name)
    return batch


def _row(value,batch,*,boolean=False):
    a=_bool(value) if boolean else _int32(value)
    if a.shape not in ((),(batch,)):raise ValueError('damage fact requires scalar or exact batch vector')
    return jnp.broadcast_to(a,(batch,))


def _coherent(state,tables,mons):
    ok=jnp.all(state.binding==tables.binding,axis=-1)&jnp.all(state.species_binding==mons.species_binding,axis=-1)
    ok &= (state.turn>=0)&(state.window_actor>=-1)&(state.window_actor<2)
    ok &= jnp.all(state.items_removed>=0,axis=-1)&jnp.all((mons.active>=-1)&(mons.active<6),axis=-1)
    for name in SUM_FIELDS:
        a=getattr(state,name);ok &= jnp.all(jnp.isfinite(a),axis=(-1,-2))&jnp.all(a[...,0]+a[...,1]>=0,axis=-1)
    return ok


def _add(pair,value):
    """Compensated TwoSum with explicit barriers against reassociation."""
    high,low=pair[...,0],pair[...,1]
    total=jax.lax.optimization_barrier(high+value)
    recovered=jax.lax.optimization_barrier(total-high)
    error=jax.lax.optimization_barrier((high-(total-recovered))+(value-recovered))
    correction=jax.lax.optimization_barrier(low+error)
    new_high=jax.lax.optimization_barrier(total+correction)
    new_low=jax.lax.optimization_barrier(correction-(new_high-total))
    return jnp.where((value>0)[...,None],jnp.stack((new_high,new_low),axis=-1),pair)


def apply(state,tables,mons,*,operation,side=-1,slot=-1,encounter=0,value=0,new_hp=0,
          tagged=False,spikes=False,producer_binding,valid=True):
    """Apply a resolved fact; DAMAGE must run before its mon condition update.

    ARRIVE resets only the incoming side's per-mon sums and closes the global
    window. Faint and blank action separators intentionally do neither: those
    are the frozen parser's rules. REMOVED counts literal Knock Off victim sides.
    """
    batch=_shape(state,mons);_binding(producer_binding)
    op,side,slot,enc,value,new_hp=(_row(x,batch) for x in (operation,side,slot,encounter,value,new_hp))
    tagged,spikes,valid=(_row(x,batch,boolean=True) for x in (tagged,spikes,valid))
    row=jnp.arange(batch);seat=jnp.clip(side,0,1);target=jnp.clip(slot,0,5)
    ok=_coherent(state,tables,mons)&jnp.all(producer_binding==tables.binding)&(op>=ARRIVE)&(op<=DAMAGE)
    turn=op==TURN;targeted=(op==ARRIVE)|(op==MOVE)|(op==DAMAGE);removed=op==REMOVED
    address=(row,seat,target);active=mons.active[row,seat]
    ok &= jnp.where(targeted,(side>=0)&(side<2)&(slot>=0)&(slot<6)&(active==slot)&
        (enc>0)&(mons.encounter[address]==enc),jnp.where(removed,(side>=0)&(side<2)&(slot==-1)&(enc==0),
        (side==-1)&(slot==-1)&(enc==0)))
    ok &= jnp.where(turn,(state.turn<2**31-1)&(value==state.turn+1),value==0)
    ok &= jnp.where(op==DAMAGE,(new_hp>=0)&(~spikes|tagged), (new_hp==0)&~tagged&~spikes)
    ok &= ~removed|(state.items_removed[row,seat]<2**31-1)
    hp,maximum=mons.hp_numerator[address],mons.hp_denominator[address]
    known=mons.hp_known[address]&(maximum>0)
    ok &= (op!=DAMAGE)|((hp>=0)&(maximum>=0)&jnp.where(maximum>0,hp<=maximum,hp==0))
    fraction=jnp.where((op==DAMAGE)&known,jnp.maximum(0,hp-new_hp).astype(jnp.float32)/jnp.maximum(maximum,1),0.)
    victim=jnp.arange(2)==seat[:,None];actor=jnp.arange(2)==state.window_actor[:,None]
    delta=fraction[:,None]*victim
    dealt=fraction[:,None]*actor*((state.window_actor>=0)&(state.window_actor!=side)&~tagged)[:,None]
    current_dealt=_add(state.current_dealt,dealt);current_taken=_add(state.current_taken,delta)
    hazard=_add(state.hazard_suffered,delta*spikes[:,None])
    clear=((op==ARRIVE)[:,None]&victim)[...,None]
    current_dealt=jnp.where(clear,0.,current_dealt);current_taken=jnp.where(clear,0.,current_taken)
    last_dealt=jnp.where(turn[:,None,None],state.current_dealt,jnp.where(clear,0.,state.last_dealt))
    last_taken=jnp.where(turn[:,None,None],state.current_taken,jnp.where(clear,0.,state.last_taken))
    current_dealt=jnp.where(turn[:,None,None],0.,current_dealt);current_taken=jnp.where(turn[:,None,None],0.,current_taken)
    window=jnp.where((op==ARRIVE)|turn|(op==CLOSE),-1,jnp.where(op==MOVE,side,state.window_actor))
    candidate=state._replace(turn=jnp.where(turn,value,state.turn),window_actor=window,
        current_dealt=current_dealt,current_taken=current_taken,last_dealt=last_dealt,last_taken=last_taken,
        hazard_suffered=hazard,items_removed=state.items_removed.at[row,seat].add(removed.astype(jnp.int32)))
    ok &= _coherent(candidate,tables,mons)
    use=ok&valid
    return jax.tree.map(lambda before,after:jnp.where(use.reshape((batch,)+(1,)*(before.ndim-1)),after,before),state,candidate),ok|~valid


def condition_tags(line):
    """Three exact typed condition metadata bits, consumed before updating HP."""
    if not isinstance(line,str) or not line.startswith('|'):raise ValueError('public damage line required')
    p=line.split('|')
    damage=len(p)>=4 and p[1]=='-damage'
    tagged=damage and any(part.strip().startswith('[from]') for part in p[4:])
    return int(damage),int(tagged),int(tagged and '[from] Spikes' in line)


def control_facts(line):
    """Global confusion/cant window closure and literal Knock Off side credit."""
    if not isinstance(line,str) or not line.startswith('|'):raise ValueError('public damage line required')
    p=line.split('|');kind=p[1] if len(p)>1 else ''
    if kind=='request':raise ValueError('private request in public damage stream')
    if kind=='cant' or reference._is_confusion_self_hit(p):return ((CLOSE,-1),)
    if kind=='-enditem' and len(p)>=3 and '[from] move: Knock Off' in line:
        seat=reference._slot_from_ident(p[2])
        if seat not in ('p1','p2'):raise ValueError('Knock Off credit requires public victim side')
        return ((REMOVED,int(seat[1])-1),)
    return ()


class DamageFeaturePatch(NamedTuple):
    numeric: jax.Array
    categorical: jax.Array
    numeric_columns: jax.Array
    categorical_columns: jax.Array
    admitted: jax.Array


def project_active(state,tables,mons,*,expected_vocabulary):
    batch=_shape(state,mons);_binding(expected_vocabulary)
    ok=_coherent(state,tables,mons)&jnp.all(expected_vocabulary==tables.vocabulary_binding)
    values=jnp.stack((state.last_dealt.sum(axis=-1),state.last_taken.sum(axis=-1)),axis=-1)
    slots=jnp.clip(mons.active,0,5)
    enc=jnp.take_along_axis(mons.encounter,slots[...,None],axis=-1)[...,0]
    present=(mons.active>=0)&(enc>0)
    numeric=jnp.zeros((batch,2,132),jnp.float32).at[...,jnp.asarray(MON_COLUMNS)].set(
        jnp.where((ok[:,None]&present)[...,None],jnp.minimum(1.,values),0.))
    return DamageFeaturePatch(numeric,jnp.zeros((batch,2,41),jnp.int32),
        jnp.zeros(132,bool).at[jnp.asarray(MON_COLUMNS)].set(True),jnp.zeros(41,bool),ok)


def project_credits(state,tables,mons,*,observer_side,expected_vocabulary):
    batch=_shape(state,mons);_binding(expected_vocabulary)
    observer=_row(observer_side,batch);row=jnp.arange(batch);own=jnp.clip(observer,0,1);opp=1-own
    ok=_coherent(state,tables,mons)&(observer>=0)&(observer<2)&jnp.all(expected_vocabulary==tables.vocabulary_binding)
    hazard=state.hazard_suffered.sum(axis=-1)
    values=jnp.stack((hazard[row,own],hazard[row,opp],state.items_removed[row,own],state.items_removed[row,opp]),axis=-1)
    numeric=jnp.zeros((batch,132),jnp.float32).at[:,jnp.asarray(FIELD_COLUMNS)].set(jnp.where(ok[:,None],
        jnp.minimum(1.,values/reference._TEAM_SIZE),0.))
    return FieldFeaturePatch(numeric,jnp.zeros((batch,41),jnp.int32),
        jnp.zeros(132,bool).at[jnp.asarray(FIELD_COLUMNS)].set(True),jnp.zeros(41,bool),ok)
