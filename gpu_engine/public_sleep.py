"""Public sleep-attempt and active-age state for the champion's exact columns.

This ports the belief engine's counters, not the replay world's separate Rest
materialization ledger. Pending attempts, skippedTime refunds, explicit status,
and condition suffixes are distinct source facts. Wake certainty and trapper
ability features need the public ability/ruled-out-set reducer and are not
claimed by this partial writer.
"""
import hashlib
import json
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from pokezero import showdown as reference
from pokezero.belief import _called_move_source
from .private_team import vocabulary_binding
from .public_mon_state import _int32, NONE, SLEEP, TOXIC
from .public_abilities import _bool

SCHEMA='pokezero.public_sleep.v1'
ARRIVE, FAINT, TURN, STATUS, CURE, CURE_TEAM, CANT_SLEEP, USABLE_MOVE, UPKEEP, CONDITION=range(10)
CORE=('SLEEP_TURNS','REST_SLEEP','TURNS_ACTIVE')
COLUMNS=tuple(reference.v4_numeric_index(getattr(reference,'NUMERIC_'+n)) for n in CORE)


class SleepTables(NamedTuple):
    species_binding: jax.Array
    vocabulary_binding: jax.Array
    binding: jax.Array
    species_count: jax.Array


def compile_sleep(species_catalog,vocabulary):
    vb=vocabulary_binding(vocabulary)
    if not np.array_equal(np.asarray(vb),np.asarray(species_catalog.device.vocabulary_binding)):
        raise ValueError('sleep inputs and public species require the same frozen vocabulary')
    payload=dict(schema=SCHEMA,species=np.asarray(species_catalog.device.binding).tolist(),
        vocabulary=np.asarray(vb).tolist(),columns=COLUMNS,sleep_divisor=5,age_divisor=64)
    digest=hashlib.sha256(json.dumps(payload,sort_keys=True,separators=(',',':')).encode()).digest()
    return SleepTables(species_catalog.device.binding,vb,jnp.asarray(np.frombuffer(digest,dtype='>u4').astype(np.uint32)),
        jnp.asarray(len(species_catalog.names),jnp.int32))


class SleepState(NamedTuple):
    binding: jax.Array
    species_binding: jax.Array
    turn: jax.Array
    encounter: jax.Array
    active: jax.Array
    live: jax.Array
    status: jax.Array
    condition_status: jax.Array
    literal_faint: jax.Array
    normalized_faint: jax.Array
    status_on_exit: jax.Array
    count: jax.Array
    rest: jax.Array
    skipped: jax.Array
    pending: jax.Array
    age: jax.Array


def _binding(value):
    if getattr(value,'shape',None)!=(8,) or getattr(value,'dtype',None)!=jnp.dtype(jnp.uint32):
        raise ValueError('sleep namespace requires eight uint32 words')
    return value


def initialize(batch,tables,*,species_binding):
    if type(batch) is not int or batch<=0:raise ValueError('positive sleep batch required')
    _binding(species_binding)
    if not np.array_equal(np.asarray(species_binding),np.asarray(tables.species_binding)):
        raise ValueError('sleep inputs require their frozen public species namespace')
    z=jnp.zeros((batch,2,6),jnp.int32);b=jnp.zeros((batch,2,6),bool)
    return SleepState(jnp.broadcast_to(tables.binding,(batch,8)),jnp.broadcast_to(species_binding,(batch,8)),
        jnp.zeros(batch,jnp.int32),z,jnp.full((batch,2),-1,jnp.int32),jnp.zeros((batch,2),bool),
        z,z,b,b,b,z,b,z,b,z)


def _shape(state,tables,mons):
    if state.turn.ndim!=1 or not state.turn.shape[0]:raise ValueError('nonempty sleep batch required')
    b=state.turn.shape[0]
    for name in state._fields:
        shape=(b,8) if name in ('binding','species_binding') else (b,) if name=='turn' else (
            (b,2) if name in ('active','live') else (b,2,6))
        dtype=jnp.uint32 if name in ('binding','species_binding') else jnp.bool_ if name in (
            'live','literal_faint','normalized_faint','status_on_exit','rest','pending') else jnp.int32
        x=getattr(state,name)
        if x.shape!=shape or x.dtype!=dtype:raise ValueError('sleep state shape/dtype: '+name)
    for name,shape,dtype in [('encounter',(b,2,6),jnp.int32),('species',(b,2,6),jnp.int32),('active',(b,2),jnp.int32),
        ('status',(b,2,6),jnp.int32),('species_binding',(b,8),jnp.uint32)]:
        x=getattr(mons,name)
        if x.shape!=shape or x.dtype!=dtype:raise ValueError('public mon shape/dtype: '+name)
    for x in (tables.species_binding,tables.vocabulary_binding,tables.binding):_binding(x)
    if tables.species_count.shape!=() or tables.species_count.dtype!=jnp.int32:
        raise ValueError('sleep public species count requires scalar int32')
    return b


def _coherent(state,tables,mons):
    ok=jnp.all(state.binding==tables.binding,axis=-1)&jnp.all(state.species_binding==tables.species_binding,axis=-1)
    ok &= jnp.all(mons.species_binding==tables.species_binding,axis=-1)
    ok &= jnp.all((mons.status>=NONE)&(mons.status<=TOXIC),axis=(-1,-2))
    ok &= (state.turn>=0)&jnp.all((state.active>=-1)&(state.active<6),axis=-1)
    ok &= jnp.all((mons.active>=-1)&(mons.active<6),axis=-1)
    ok &= jnp.all(state.encounter>=0,axis=(-1,-2))
    # The source belief key is side+species. Do not silently split its sleep
    # history across two distinct public encounters with the same species.
    present_mon=mons.encounter>0
    ok &= jnp.all(~present_mon|((mons.species>0)&(mons.species<tables.species_count)),axis=(-1,-2))
    duplicate=(mons.species[...,None]==mons.species[...,None,:])&present_mon[...,None]&present_mon[...,None,:]
    duplicate &= ~jnp.eye(6,dtype=bool)
    ok &= ~jnp.any(duplicate,axis=(-1,-2,-3))
    for x in (state.status,state.condition_status):ok &= jnp.all((x>=NONE)&(x<=TOXIC),axis=(-1,-2))
    for x in (state.count,state.skipped,state.age):ok &= jnp.all(x>=0,axis=(-1,-2))
    present=state.encounter>0
    ok &= jnp.all(~state.literal_faint|state.normalized_faint,axis=(-1,-2))
    for name in ('status','condition_status','literal_faint','normalized_faint','status_on_exit','count','rest','skipped','pending','age'):
        ok &= ~jnp.any(~present&(getattr(state,name)!=0),axis=(-1,-2))
    row=jnp.arange(state.turn.shape[0])[:,None];seat=jnp.arange(2)[None,:]
    active_seen=state.encounter[row,seat,jnp.clip(state.active,0,5)]>0
    ok &= jnp.all(~state.live|((state.active>=0)&active_seen),axis=-1)
    return ok


def apply(state,tables,mons,*,operation,side=-1,slot=-1,encounter=0,first=0,second=0,
          producer_binding,valid=True):
    """Apply resolved public facts; literal_faint means the exact source `0 fnt`.

    ARRIVE follows the mon ledger arrival and receives its literal condition
    flag in first. STATUS carries status/Rest; CONDITION carries status and
    two raw condition flags (bit0 exact `0 fnt`, bit1 normalizes to `0 fnt`).
    CURE first distinguishes a benched cure.
    The canonical producer must jointly commit all consumers of a raw line.
    """
    b=_shape(state,tables,mons);_binding(producer_binding)
    op,side,slot,enc,x,y=(jnp.broadcast_to(_int32(v),(b,)) for v in (operation,side,slot,encounter,first,second))
    valid=jnp.broadcast_to(_bool(valid),(b,));row=jnp.arange(b)
    seat=jnp.clip(side,0,1);target=jnp.clip(slot,0,5);address=(row,seat,target)
    arrive=op==ARRIVE;turn=op==TURN;upkeep=op==UPKEEP;team=op==CURE_TEAM;global_fact=turn|upkeep
    ok=_coherent(state,tables,mons)&jnp.all(producer_binding==tables.binding)&(op>=ARRIVE)&(op<=CONDITION)
    targeted=~global_fact&~team
    address_ok=(side>=0)&(side<2)&(slot>=0)&(slot<6)&(enc>0)&(mons.encounter[address]==enc)
    ok &= jnp.where(global_fact,(side==-1)&(slot==-1)&(enc==0),
        jnp.where(team,(side>=0)&(side<2)&(slot==-1)&(enc==0),address_ok))
    same=state.encounter==mons.encounter
    target_mask=(jnp.arange(2)[None,:,None]==seat[:,None,None])&(jnp.arange(6)[None,None,:]==target[:,None,None])
    same=jnp.where(arrive[:,None,None]&target_mask,True,same)
    active_same=state.active==mons.active
    active_same=jnp.where(arrive[:,None]&(jnp.arange(2)==seat[:,None]),True,active_same)
    ok &= jnp.all(same,axis=(-1,-2))&jnp.all(active_same,axis=-1)
    ok &= ~targeted|jnp.where(arrive,(mons.active[row,seat]==slot)&((state.encounter[address]==0)|(state.encounter[address]==enc)),state.encounter[address]==enc)
    # Belief attribution is active-first while a live occupant exists. The
    # positionless team-cure spelling is the explicit species-target exception.
    bench_cure=(op==CURE)&(x==1)
    ok &= ~targeted|arrive|~state.live[row,seat]|bench_cure|(state.active[row,seat]==slot)
    bo=(x==0)|(x==1);by=(y==0)|(y==1)
    ok &= jnp.where(turn,(state.turn<2**31-1)&(x==state.turn+1)&(y==0),
        jnp.where((op==STATUS)|(op==CONDITION),(x>=NONE)&(x<=TOXIC)&jnp.where(op==STATUS,by&((x==SLEEP)|(y==0)),
            (y>=0)&(y<=3)&(((y&1)==0)|((y&2)!=0))),
        jnp.where(arrive|(op==CURE),bo&(y==0),(x==0)&(y==0))))
    ok &= ~turn|jnp.all(~state.live|(state.age[row[:,None],jnp.arange(2)[None,:],jnp.clip(state.active,0,5)]<2**31-1),axis=-1)
    ok &= ~((op==CANT_SLEEP)|((op==USABLE_MOVE)&state.pending[address]))|jnp.where(op==CANT_SLEEP,state.count[address],state.skipped[address])<2**31-1
    out=state._asdict();put=lambda arr,val:arr.at[address].set(val)
    # The outgoing explicit status is saved before the incoming stint resets.
    old_slot=jnp.clip(state.active[row,seat],0,5);old_address=(row,seat,old_slot)
    leaving=arrive&state.live[row,seat]&~state.literal_faint[old_address]
    exits=state.status_on_exit.at[old_address].set(jnp.where(leaving,state.status[old_address]!=NONE,state.status_on_exit[old_address]))
    incoming=mons.status[address]
    clear_arrival=arrive&exits[address]&(incoming==NONE)
    status=jnp.where(arrive&((incoming!=NONE)|clear_arrival),incoming,state.status[address])
    status=jnp.where(op==STATUS,x,status)
    count=jnp.where(arrive&(incoming==SLEEP),jnp.maximum(0,state.count[address]-state.skipped[address]),state.count[address])
    count=jnp.where((op==STATUS)&(x==SLEEP),0,jnp.where(op==CANT_SLEEP,count+1,count))
    rest=jnp.where((op==STATUS)&(x==SLEEP),y==1,state.rest[address])
    skipped=jnp.where(arrive|((op==STATUS)&(x==SLEEP)),0,state.skipped[address])
    skipped=jnp.where((op==USABLE_MOVE)&state.pending[address],skipped+1,skipped)
    pending=jnp.where(op==CANT_SLEEP,True,jnp.where(op==USABLE_MOVE,False,state.pending[address]))
    cure=(op==CURE)|clear_arrival
    status=jnp.where(cure,NONE,status);count=jnp.where(cure,0,count);rest=jnp.where(cure,False,rest)
    skipped=jnp.where(op==CURE,0,skipped)
    cond=jnp.where(arrive,incoming,jnp.where(op==CONDITION,x,state.condition_status[address]))
    cond=jnp.where(op==CURE,NONE,cond)
    normalized=jnp.where(arrive,x==1,jnp.where(op==CONDITION,(y&2)!=0,jnp.where(op==FAINT,True,state.normalized_faint[address])))
    literal=jnp.where(arrive,x==1,jnp.where(op==CONDITION,(y&1)!=0,jnp.where(op==FAINT,True,
        jnp.where(op==CURE,normalized,state.literal_faint[address]))))
    cond=jnp.where(op==FAINT,NONE,cond)
    for name,val in [('status',status),('count',count),('rest',rest),('skipped',skipped),('pending',pending),
        ('condition_status',cond),('literal_faint',literal),('normalized_faint',normalized),('age',jnp.where(arrive,0,state.age[address]))]:
        out[name]=jnp.where(targeted[:,None,None],put(getattr(state,name),val),getattr(state,name))
    exits=exits.at[address].set(jnp.where(arrive|((op==CURE)&(x==1))|(op==FAINT),False,exits[address]))
    out['status_on_exit']=jnp.where(targeted[:,None,None],exits,state.status_on_exit)
    out['encounter']=jnp.where(arrive[:,None,None],put(state.encounter,enc),state.encounter)
    out['active']=state.active.at[row,seat].set(jnp.where(arrive,slot,state.active[row,seat]))
    out['live']=state.live.at[row,seat].set(jnp.where(arrive,True,jnp.where(op==FAINT,False,state.live[row,seat])))
    team_mask=team[:,None,None]&(jnp.arange(2)[None,:,None]==seat[:,None,None])&~state.literal_faint&((state.status!=NONE)|(state.condition_status!=NONE))
    for name in ('status','condition_status','count','rest','skipped','status_on_exit'):
        out[name]=jnp.where(team_mask,0,out[name]).astype(getattr(state,name).dtype)
    out['literal_faint']=jnp.where(team_mask,state.normalized_faint,out['literal_faint'])
    out['skipped']=jnp.where(upkeep[:,None,None]&state.pending,0,out['skipped'])
    out['pending']=jnp.where(upkeep[:,None,None],False,out['pending'])
    age_next=state.age.at[row[:,None],jnp.arange(2)[None,:],jnp.clip(state.active,0,5)].add(state.live.astype(jnp.int32))
    out['age']=jnp.where(turn[:,None,None],age_next,out['age']);out['turn']=jnp.where(turn,x,state.turn)
    use=ok&valid
    output=SleepState(**out)
    output=jax.tree.map(lambda new,old:jnp.where(use.reshape((b,)+(1,)*(old.ndim-1)),new,old),output,state)
    return output,ok|~valid


class SleepPatch(NamedTuple):
    numeric: jax.Array           # [B,2,6,132], absolute public side/encounter order
    numeric_columns: jax.Array
    admitted: jax.Array


class OwnStatusFacts(NamedTuple):
    owner_side: jax.Array
    vocabulary_binding: jax.Array
    species_binding: jax.Array
    species: jax.Array            # [6] or [B,6], private request order
    present: jax.Array
    status: jax.Array
    active: jax.Array


def parse_own_status(request,*,owner_side,species_catalog,vocabulary):
    """Parse only the owner's request facts; never an opponent private view."""
    from .private_requests import parse_action_request
    owner=parse_action_request(request,owner_side=owner_side).owner_side
    vb=vocabulary_binding(vocabulary)
    if not np.array_equal(np.asarray(vb),np.asarray(species_catalog.device.vocabulary_binding)):
        raise ValueError('own sleep status requires the frozen public vocabulary')
    team=reference._self_team_from_request(request,f'p{owner_side+1}')
    species=np.zeros(6,np.int32);present=np.zeros(6,bool);status=np.zeros(6,np.int32);active=np.zeros(6,bool)
    ids={'none':NONE,'slp':SLEEP,'par':2,'brn':3,'frz':4,'psn':5,'tox':TOXIC}
    for i,mon in enumerate(team):
        if mon.showdown_slot!=f'p{owner_side+1}':raise ValueError('private status row belongs to another seat')
        token=reference._condition_features(mon.condition).status
        if token not in ids:raise ValueError('unknown owning request status')
        species[i]=species_catalog.species_id(mon.species);present[i]=True;status[i]=ids[token];active[i]=mon.active
    if len(set(species[present].tolist()))!=int(present.sum()):
        raise ValueError('own species must identify one request row without ambiguity')
    return OwnStatusFacts(owner,vb,species_catalog.device.binding,
        jnp.asarray(species),jnp.asarray(present),jnp.asarray(status),jnp.asarray(active))


def project(state,tables,mons,*,observer_side,expected_vocabulary,own_status,exact_state=True):
    """Write sleep count/Rest/age; wake certainty is deliberately unclaimed."""
    b=_shape(state,tables,mons);_binding(expected_vocabulary)
    observer=jnp.broadcast_to(_int32(observer_side),(b,))
    exact=jnp.broadcast_to(_bool(exact_state),(b,))
    if not isinstance(own_status,OwnStatusFacts):raise ValueError('owner-bound request status facts required')
    def packet(value,tail,dtype):
        if getattr(value,'dtype',None)!=jnp.dtype(dtype) or value.shape not in (tail,(b,)+tail):
            raise ValueError('own status packet shape/dtype mismatch')
        return jnp.broadcast_to(value,(b,)+tail)
    owner=packet(own_status.owner_side,(),jnp.int32)
    vb=packet(own_status.vocabulary_binding,(8,),jnp.uint32)
    sb=packet(own_status.species_binding,(8,),jnp.uint32)
    species=packet(own_status.species,(6,),jnp.int32)
    present_own=packet(own_status.present,(6,),jnp.bool_)
    request_status=packet(own_status.status,(6,),jnp.int32)
    request_active=packet(own_status.active,(6,),jnp.bool_)
    ok=_coherent(state,tables,mons)&jnp.all(state.encounter==mons.encounter,axis=(-1,-2))&jnp.all(state.active==mons.active,axis=-1)
    ok &= (observer>=0)&(observer<=1)&jnp.all(expected_vocabulary==tables.vocabulary_binding)
    ok &= (owner==observer)&jnp.all(vb==tables.vocabulary_binding,axis=-1)&jnp.all(sb==tables.species_binding,axis=-1)
    ok &= jnp.all(jnp.where(present_own,(species>0)&(species<tables.species_count)&(request_status>=NONE)&(request_status<=TOXIC),
        (species==0)&(request_status==NONE)&~request_active),axis=-1)
    duplicate=(species[:,:,None]==species[:,None,:])&present_own[:,:,None]&present_own[:,None,:]&~jnp.eye(6,dtype=bool)
    ok &= ~jnp.any(duplicate,axis=(-1,-2))
    match=(mons.species[...,None]==species[:,None,None,:])&present_own[:,None,None,:]
    own_side=jnp.arange(2)[None,:,None]==observer[:,None,None]
    ok &= jnp.all(~own_side|(mons.encounter==0)|(match.sum(axis=-1)==1),axis=(-1,-2))
    private_status=jnp.sum(jnp.where(match,request_status[:,None,None,:],0),axis=-1)
    private_active=jnp.any(match&request_active[:,None,None,:],axis=-1)
    opponent=jnp.arange(2)[None,:,None]!=(observer[:,None,None])
    status=jnp.where(opponent,jnp.where(state.status!=NONE,state.status,mons.status),private_status)
    sleeping=status==SLEEP
    active=jnp.where(opponent,jnp.arange(6)[None,None,:]==mons.active[:,:,None],private_active)
    values=jnp.stack((jnp.where(sleeping,jnp.minimum(1.,state.count/5.),0.),
        (sleeping&state.rest).astype(jnp.float32),jnp.where(active,jnp.minimum(1.,state.age/64.),0.)),axis=-1)
    present=state.encounter>0
    numeric=jnp.zeros((b,2,6,132),jnp.float32).at[...,jnp.asarray(COLUMNS)].set(jnp.where(present[...,None]&(ok&exact)[:,None,None,None],values,0.))
    return SleepPatch(numeric,jnp.zeros(132,bool).at[jnp.asarray(COLUMNS)].set(True),ok)


def lexical_facts(line):
    """Facts for an already-resolved canonical producer; no identity inference."""
    if not isinstance(line,str):raise ValueError('sleep source line must be text')
    p=line.split('|');kind=p[1] if len(p)>1 else '';facts=[]
    if kind=='cant' and '|slp' in line:facts.append((CANT_SLEEP,0,0))
    elif kind=='move' and len(p)>=4 and _called_move_source(line) is None and reference._normalize_identifier(p[3]) in ('sleeptalk','snore'):
        facts.append((USABLE_MOVE,0,0))
    return tuple(facts)
