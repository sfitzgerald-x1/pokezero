"""Bounded public action-window and per-stay tendency aggregates.

Lexical adapters supply facts only. The device owns occupants, opportunities,
mutable open-window outcomes and counters. This component does not admit a raw
stream or assemble a complete observation by itself.
"""
from dataclasses import dataclass
import hashlib
import json
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from pokezero import showdown as reference, transitions as source
from pokezero.belief import _CALLER_MOVES, _called_move_source
from .private_team import vocabulary_binding
from .public_moves import _static_category
from .public_mon_state import _int32
from .public_abilities import _bool

SCHEMA='pokezero.public_tendency.v1'
OTHER,SWITCH,DRAG,REPLACE,FAINT,TURN,BOUNDARY,MOVE,CANT,PREPARE,WEATHER,OUTCOME,PURSUIT,WIN,UPKEEP=range(15)
MON_COLUMNS=tuple(reference.v4_numeric_index(x) for x in (
    reference.NUMERIC_MON_SWITCHED_BEFORE_ATTACK,reference.NUMERIC_MON_STAYED_AND_ATTACKED,
    reference.NUMERIC_MON_TURNS_ACTIVE_TOTAL,reference.NUMERIC_MON_SWITCHED_VS_ACTIVE,
    reference.NUMERIC_MON_STAYED_VS_ACTIVE))
STAT_COLUMNS=tuple(reference.v4_numeric_index(x) for x in (
    reference.NUMERIC_STAT_OPP_SWITCH_COUNT,reference.NUMERIC_STAT_OPP_DECISION_OPPORTUNITIES,
    reference.NUMERIC_STAT_BLOCKED_ON_OUR_ATTACK,reference.NUMERIC_STAT_PURSUIT_INTERCEPT_PREDICT,
    reference.NUMERIC_STAT_MY_SWITCH_TURNS,*range(reference.NUMERIC_STAT_WEATHER_REVEAL_OFFSET,
                                               reference.NUMERIC_STAT_WEATHER_REVEAL_OFFSET+6)))


class TendencyTables(NamedTuple):
    species_binding: jax.Array
    move_binding: jax.Array
    vocabulary_binding: jax.Array
    binding: jax.Array
    species_count: jax.Array
    move_count: jax.Array
    baton_move: jax.Array
    pursuit_move: jax.Array
    stats_category: jax.Array


@dataclass(frozen=True)
class TendencyCatalog:
    species: object
    moves: object
    device: TendencyTables


def compile_tendency(species_catalog,moves,vocabulary):
    vb=vocabulary_binding(vocabulary)
    if not np.array_equal(np.asarray(vb),np.asarray(species_catalog.device.vocabulary_binding)):
        raise ValueError('tendency requires the frozen public vocabulary')
    if not np.array_equal(np.asarray(species_catalog.device.source_binding),np.asarray(moves.device.source_binding)):
        raise ValueError('tendency species and moves require the same set source')
    payload=dict(schema=SCHEMA,species=np.asarray(species_catalog.device.binding).tolist(),
        moves=np.asarray(moves.device.binding).tolist(),vocabulary=np.asarray(vb).tolist(),
        mon_columns=MON_COLUMNS,stats_columns=STAT_COLUMNS,capacity=6,divisors=[64,8],
        category=_static_category(vocabulary,'stats'))
    digest=hashlib.sha256(json.dumps(payload,sort_keys=True,separators=(',',':')).encode()).digest()
    tables=TendencyTables(species_catalog.device.binding,moves.device.binding,vb,
        jnp.asarray(np.frombuffer(digest,dtype='>u4').astype(np.uint32)),
        jnp.asarray(len(species_catalog.names),jnp.int32),jnp.asarray(len(moves.names),jnp.int32),
        jnp.asarray(moves.ids.get('batonpass',0),jnp.int32),jnp.asarray(moves.ids.get('pursuit',0),jnp.int32),
        jnp.asarray(payload['category'],jnp.int32))
    return TendencyCatalog(species_catalog,moves,tables)


class TendencyFact(NamedTuple):
    kind: jax.Array
    side: jax.Array
    species: jax.Array
    action: jax.Array
    target: jax.Array
    turn: jax.Array
    called: jax.Array
    locked: jax.Array
    recharge: jax.Array
    baton: jax.Array
    weather: jax.Array
    ability: jax.Array
    outcome: jax.Array


class TendencyState(NamedTuple):
    binding: jax.Array
    species_binding: jax.Array
    complete: jax.Array
    turn: jax.Array
    species: jax.Array              # [B,2,6], unique base-species counter keys
    active: jax.Array               # [B,2], local key index, -1 if absent
    moved: jax.Array
    lead: jax.Array
    faint: jax.Array
    baton: jax.Array
    charge: jax.Array
    mon: jax.Array                  # [B,2,6,3], switched/stayed/turns
    matchup: jax.Array              # [B,2,6,6,2], switched/stayed
    switches: jax.Array
    opportunities: jax.Array
    opportunity_turn: jax.Array
    blocked: jax.Array              # finalized windows; open window added virtually
    pursuits: jax.Array
    weather: jax.Array              # [B,2,3,2], seen/ability OR bits
    marker: jax.Array
    window_side: jax.Array
    window_move: jax.Array
    window_target: jax.Array
    window_outcome: jax.Array


def _binding(x):
    if getattr(x,'shape',None)!=(8,) or getattr(x,'dtype',None)!=jnp.dtype(jnp.uint32):
        raise ValueError('tendency namespace requires eight uint32 words')


def initialize(batch,tables,*,species_binding,complete_prefix=False):
    if type(batch) is not int or batch<=0 or type(complete_prefix) is not bool:
        raise ValueError('positive tendency batch and explicit prefix provenance required')
    _binding(species_binding)
    if not np.array_equal(np.asarray(species_binding),np.asarray(tables.species_binding)):
        raise ValueError('tendency requires its public species namespace')
    z=lambda *shape:jnp.zeros((batch,*shape),jnp.int32)
    f=lambda *shape:jnp.zeros((batch,*shape),bool)
    return TendencyState(jnp.broadcast_to(tables.binding,(batch,8)),jnp.broadcast_to(species_binding,(batch,8)),
        jnp.full(batch,complete_prefix,bool),z(),z(2,6),jnp.full((batch,2),-1,jnp.int32),f(2),f(2),f(2),f(2),
        z(2),z(2,6,3),z(2,6,6,2),z(2),z(2),jnp.full((batch,2),-1,jnp.int32),z(2),z(2),f(2,3,2),f(2),
        jnp.full(batch,-1,jnp.int32),f(),jnp.full(batch,-1,jnp.int32),jnp.full(batch,9,jnp.int32))


def _shape(s,t):
    b=s.turn.shape[0] if s.turn.ndim==1 else 0
    if not b:raise ValueError('nonempty tendency batch required')
    shapes={'binding':(8,),'species_binding':(8,),'species':(2,6),'active':(2,),
        'mon':(2,6,3),'matchup':(2,6,6,2),'weather':(2,3,2)}
    pairs=('moved','lead','faint','baton','charge','switches','opportunities','opportunity_turn','blocked','pursuits','marker')
    flags=('complete','moved','lead','faint','baton','weather','marker','window_move')
    for n in s._fields:
        shape=shapes.get(n,(2,) if n in pairs else ())
        dtype=jnp.uint32 if n in ('binding','species_binding') else jnp.bool_ if n in flags else jnp.int32
        if getattr(s,n).shape!=(b,*shape) or getattr(s,n).dtype!=dtype:raise ValueError('tendency state shape/dtype: '+n)
    for x in t[:4]:_binding(x)
    for x in t[4:]:
        if x.shape!=() or x.dtype!=jnp.int32:raise ValueError('tendency scalar table shape/dtype')
    return b


def _coherent(s,t):
    ok=jnp.all(s.binding==t.binding,axis=-1)&jnp.all(s.species_binding==t.species_binding,axis=-1)&(s.turn>=0)
    ok &= jnp.all((s.active>=-1)&(s.active<6),axis=-1)
    ok &= jnp.all((s.species>=0)&(s.species<t.species_count),axis=(1,2))
    equal=(s.species[:,:,:,None]==s.species[:,:,None,:])&(s.species[:,:,:,None]>0)
    ok &= ~jnp.any(equal&~jnp.eye(6,dtype=bool)[None,None,:,:],axis=(1,2,3))
    current=jnp.take_along_axis(s.species,jnp.clip(s.active,0,5)[:,:,None],axis=-1)[:,:,0]
    ok &= jnp.all((s.active<0)|(current>0),axis=-1)
    ok &= jnp.all((s.opportunity_turn>=-1)&(s.opportunity_turn<=s.turn[:,None]),axis=-1)
    ok &= jnp.all((s.charge>=0)&(s.charge<t.move_count),axis=-1)
    for n in ('mon','switches','opportunities','blocked','pursuits'):
        x=getattr(s,n);ok &= jnp.all((x>=0)&(x<=64),axis=tuple(range(1,x.ndim)))
    ok &= jnp.all((s.matchup>=0)&(s.matchup<=8),axis=(1,2,3,4))
    ok &= (s.window_side>=-1)&(s.window_side<=1)&(s.window_target>=-1)&(s.window_target<=1)
    ok &= jnp.any(s.window_outcome[:,None]==jnp.array([0,1,2,3,4,5,9]),axis=-1)
    return ok


def apply(state,tables,fact,*,producer_binding,valid=True):
    """Apply one lexical fact; bad rows return their entire previous state."""
    b=_shape(state,tables);_binding(producer_binding)
    if not isinstance(fact,TendencyFact):raise ValueError('typed tendency fact required')
    flags=('called','locked','recharge','baton','ability')
    d={n:jnp.broadcast_to(_bool(x) if n in flags else _int32(x),(b,)) for n,x in zip(fact._fields,fact)}
    k,side=d['kind'],d['side'];rows=jnp.arange(b);seat=jnp.clip(side,0,1);other=1-seat
    named=(side>=0)&(side<=1);arrive=(k>=SWITCH)&(k<=REPLACE);move=k==MOVE;cant=k==CANT
    ok=_coherent(state,tables)&jnp.all(producer_binding==tables.binding)&(k>=OTHER)&(k<=UPKEEP)
    ok &= (side>=-1)&(side<=1)&(d['target']>=-1)&(d['target']<=1)&(d['species']>=0)&(d['species']<tables.species_count)
    ok &= (d['action']>=0)&(d['action']<tables.move_count)&(d['weather']>=0)&(d['weather']<=3)&(d['turn']>=0)
    ok &= jnp.any(d['outcome'][:,None]==jnp.array([0,1,2,3,4,5,9]),axis=-1)
    ok &= ~(arrive|move|cant|(k==FAINT)|(k==PREPARE)|(k==PURSUIT))|named
    ok &= ~arrive|(d['species']>0)
    ok &= ~(move|(k==PREPARE))|(d['action']>0)
    ok &= ~(k==TURN)|((d['turn']>=1)&((state.turn==0)|
        ((state.turn<2**31-1)&(d['turn']==state.turn+1))))
    s=state
    old=s.active[rows,seat];old_safe=jnp.clip(old,0,5);facing=s.active[rows,other];face_safe=jnp.clip(facing,0,5)
    voluntary=(k==SWITCH)&s.lead[rows,seat]&~s.faint[rows,seat]&~s.baton[rows,seat]&~d['baton']
    first_move=move&(old>=0)&~s.moved[rows,seat]
    bail=voluntary&(old>=0)&~s.moved[rows,seat]
    mon=s.mon.at[rows,seat,old_safe,0].set(jnp.minimum(64,s.mon[rows,seat,old_safe,0]+bail))
    mon=mon.at[rows,seat,old_safe,1].set(jnp.minimum(64,mon[rows,seat,old_safe,1]+first_move))
    match=s.matchup.at[rows,seat,old_safe,face_safe,0].set(jnp.minimum(8,s.matchup[rows,seat,old_safe,face_safe,0]+(bail&(facing>=0))))
    match=match.at[rows,seat,old_safe,face_safe,1].set(jnp.minimum(8,match[rows,seat,old_safe,face_safe,1]+(first_move&(facing>=0))))
    # The first move of a stay counts even when called or locked; only the
    # opportunity denominator excludes those declarations.
    locked=d['locked']|((s.charge[rows,seat]>0)&(s.charge[rows,seat]==d['action']))
    decision=voluntary|(move&~d['called']&~locked)|(cant&~d['recharge'])
    fresh=decision&(s.opportunity_turn[rows,seat]!=s.turn)
    opportunities=s.opportunities.at[rows,seat].set(jnp.minimum(64,s.opportunities[rows,seat]+fresh))
    opturn=s.opportunity_turn.at[rows,seat].set(jnp.where(decision,s.turn,s.opportunity_turn[rows,seat]))
    switches=s.switches.at[rows,seat].set(jnp.minimum(64,s.switches[rows,seat]+voluntary))
    intercept=move&(d['action']==tables.pursuit_move)&(tables.pursuit_move>0)&(d['target']>=0)&s.marker[rows,jnp.clip(d['target'],0,1)]
    pursuits=s.pursuits.at[rows,seat].set(jnp.minimum(64,s.pursuits[rows,seat]+intercept))
    close=arrive|move|cant|(k==TURN)|(k==BOUNDARY)|(k==UPKEEP)|(k==WIN)
    winseat=jnp.clip(s.window_side,0,1)
    blocked=s.blocked.at[rows,winseat].set(jnp.minimum(64,s.blocked[rows,winseat]+(
        close&s.window_move&(s.window_side>=0)&(s.window_outcome==2))))
    wside=jnp.where(close,-1,s.window_side);wmove=s.window_move&~close
    wtarget=jnp.where(close,-1,s.window_target);outcome=jnp.where(close,9,s.window_outcome)
    action=move|cant|(k==SWITCH)
    wside=jnp.where(action,side,wside);wmove=jnp.where(action,move,wmove)
    wtarget=jnp.where(move,d['target'],wtarget)
    outcome=jnp.where((k==OUTCOME)&(d['target']==wtarget)&(wside>=0),jnp.minimum(outcome,d['outcome']),outcome)
    # Markers survive blanks/faints/wins; evaluate before clearing on MOVE.
    marker=jnp.where((arrive|move|cant|(k==TURN)|(k==UPKEEP))[:,None],False,s.marker)
    marker=marker.at[rows,seat].set(marker[rows,seat]|(k==PURSUIT))
    moved=s.moved.at[rows,seat].set(jnp.where(arrive,False,s.moved[rows,seat]|move))
    lead=s.lead.at[rows,seat].set(s.lead[rows,seat]|arrive)
    faint=s.faint.at[rows,seat].set(jnp.where(arrive,False,s.faint[rows,seat]|(k==FAINT)))
    baton=s.baton.at[rows,seat].set(jnp.where(arrive,False,jnp.where(move,d['action']==tables.baton_move,s.baton[rows,seat])))
    charge=s.charge.at[rows,seat].set(jnp.where(arrive|move|cant,0,jnp.where(k==PREPARE,d['action'],s.charge[rows,seat])))
    keys=s.species[rows,seat];found=(keys==d['species'][:,None])&(keys>0)
    empty=keys==0;slot=jnp.where(jnp.any(found,axis=-1),jnp.argmax(found,axis=-1),jnp.argmax(empty,axis=-1))
    ok &= ~arrive|(jnp.any(found|empty,axis=-1))
    species=s.species.at[rows,seat,slot].set(jnp.where(arrive,d['species'],s.species[rows,seat,slot]))
    active=s.active.at[rows,seat].set(jnp.where(arrive,slot,s.active[rows,seat]))
    # TURN increments stale fainted occupants too, exactly as the source map.
    for st in (0,1):
        idx=jnp.clip(s.active[:,st],0,5)
        mon=mon.at[rows,st,idx,2].set(jnp.minimum(64,mon[rows,st,idx,2]+((k==TURN)&(s.active[:,st]>=0))))
    setter=jnp.where(named,side,wside);ws=jnp.clip(setter,0,1);weather_idx=jnp.clip(d['weather']-1,0,2)
    reveal=(k==WEATHER)&(d['weather']>0)&(setter>=0)
    weather=s.weather.at[rows,ws,weather_idx,0].set(s.weather[rows,ws,weather_idx,0]|reveal)
    weather=weather.at[rows,ws,weather_idx,1].set(weather[rows,ws,weather_idx,1]|(reveal&d['ability']))
    candidate=s._replace(turn=jnp.where(k==TURN,d['turn'],s.turn),species=species,active=active,moved=moved,
        lead=lead,faint=faint,baton=baton,charge=charge,mon=mon,matchup=match,switches=switches,
        opportunities=opportunities,opportunity_turn=opturn,blocked=blocked,pursuits=pursuits,weather=weather,
        marker=marker,window_side=wside,window_move=wmove,window_target=wtarget,window_outcome=outcome)
    enabled=jnp.broadcast_to(_bool(valid),(b,));accepted=ok&enabled
    result=jax.tree.map(lambda old,new:jnp.where(accepted.reshape((b,)+(1,)*(old.ndim-1)),new,old),s,candidate)
    return result,~enabled|ok


def decode_line(line,*,catalog):
    """Translate source lexical predicates, without folding counters on host."""
    if not isinstance(line,str):raise ValueError('tendency input requires a raw protocol string')
    p=line.split('|');name=p[1] if len(p)>1 else '';ident=p[2] if len(p)>2 else ''
    seat=reference._slot_from_ident(ident);side=int(seat[1])-1 if seat in ('p1','p2') else -1
    k=OTHER;species=action=turn=weather=0;target=-1;called=locked=recharge=baton=ability=False;outcome=9
    if name in ('switch','drag','replace'):
        if len(p)<4 or side<0:raise ValueError('tendency arrival requires public side and species')
        k={'switch':SWITCH,'drag':DRAG,'replace':REPLACE}[name]
        label=reference._species_from_details(p[3]) or reference._species_from_ident(ident)
        species=catalog.species.species_id(label)
        # The source keys counters by exact display spelling before its writer
        # normalizes them. Refuse alias collisions rather than merge raw keys.
        if label!=catalog.species.names[species]:
            raise ValueError('tendency counter keys require canonical catalog species spelling')
        baton=reference._line_mentions_baton_pass(p)
    elif name=='move':
        if len(p)<4 or side<0:raise ValueError('tendency move requires public side and action')
        k=MOVE;action=catalog.moves.move_id(p[3]);caller=_called_move_source(line)
        called=caller in _CALLER_MOVES;locked=caller=='lockedmove'
        defender=(reference._slot_from_ident(p[4]) if len(p)>4 else None) or f'p{2-side}'
        target=int(defender[1])-1
    elif name=='cant':
        if len(p)<4 or side<0:raise ValueError('tendency cant requires public side and reason')
        k=CANT;recharge=reference._side_condition_identifier(p[3]) in source._CANT_NO_CHOICE_REASONS
    elif name=='-prepare':
        if len(p)<4 or side<0:raise ValueError('tendency prepare requires side and action')
        k=PREPARE;action=catalog.moves.move_id(p[3])
    elif name=='faint':k=FAINT
    elif name=='turn':
        number=reference._canonical_turn_number(line)
        if number is None or number>=2**31:raise ValueError('tendency requires an exact monotone int32 turn')
        k=TURN;turn=number
    elif line=='|upkeep':k=UPKEEP
    elif name=='':k=BOUNDARY
    elif name=='win':k=WIN
    elif name=='-weather' and len(p)>=3:
        value=reference._normalize_identifier(p[2])
        if value and value!='none' and '[upkeep]' not in line:
            k=WEATHER;weather={'raindance':1,'sunnyday':2,'sandstorm':3}.get(value,0)
            setter=source._of_tag_slot(line);side=int(setter[1])-1 if setter else -1
            payload=source._from_tag_payload(line)
            ability=payload is not None and payload.lower().startswith('ability:')
    if name in ('-immune','-heal','-start','-activate','-end'):
        payload=source._from_tag_payload(line);label=reference._side_condition_identifier(p[3]) if len(p)>3 else ''
        rank=None
        if name=='-immune':rank=0 if source._is_absorb_signature(payload) else 1
        elif name in ('-heal','-start') and (source._is_absorb_signature(payload) or source._is_absorb_start(name,p)):rank=0
        elif name=='-activate' and label in ('protect','detect'):rank=2
        elif name=='-activate' and label=='substitute':rank=4
        elif name=='-activate' and label=='endure':rank=5
        elif name=='-end' and label=='substitute':rank=3
        elif name=='-activate' and label=='pursuit':k=PURSUIT
        if rank is not None:k=OUTCOME;target=side;outcome=rank
    if k in (SWITCH,DRAG,REPLACE,FAINT,MOVE,CANT,PREPARE,OUTCOME,PURSUIT) and side<0:
        raise ValueError('tendency action facts require a public side')
    values=(k,side,species,action,target,turn,called,locked,recharge,baton,weather,ability,outcome)
    bools={6,7,8,9,11}
    return TendencyFact(*(jnp.asarray(v,jnp.bool_ if i in bools else jnp.int32) for i,v in enumerate(values)))


class TendencyProjection(NamedTuple):
    numeric: jax.Array
    stats_numeric: jax.Array
    stats_role: jax.Array
    numeric_columns: jax.Array
    stats_columns: jax.Array
    stats_attention: jax.Array
    admitted: jax.Array


def project(state,tables,*,observer_side,opponent_species,own_active_species,expected_vocabulary,enabled=True):
    """Write opponent cells and token22; own active base species is private input.

    The caller supplies opponent token order and the first active own-request
    candidate's base species, never a sampled world or a public-pointer fallback.
    """
    b=_shape(state,tables);_binding(expected_vocabulary)
    observer=jnp.broadcast_to(_int32(observer_side),(b,));opp=1-jnp.clip(observer,0,1);row=jnp.arange(b)
    species=_int32(opponent_species);own=jnp.broadcast_to(_int32(own_active_species),(b,))
    if species.shape!=(b,6):raise ValueError('tendency token species must be B by six')
    on=jnp.broadcast_to(_bool(enabled),(b,));admitted=_coherent(state,tables)&state.complete
    admitted &= jnp.all(expected_vocabulary==tables.vocabulary_binding)&(observer>=0)&(observer<=1)
    admitted &= jnp.all((species>=0)&(species<tables.species_count),axis=-1)&(own>=0)&(own<tables.species_count)
    on &= admitted
    matches=(state.species[row,opp,None,:]==species[:,:,None])&(species[:,:,None]>0)
    indices=jnp.argmax(matches,axis=-1);known=jnp.any(matches,axis=-1)
    values=state.mon[row[:,None],opp[:,None],indices].astype(jnp.float32)/64
    own_matches=(state.species[row,jnp.clip(observer,0,1)]==own[:,None])&(own[:,None]>0)
    own_idx=jnp.argmax(own_matches,axis=-1)
    matchup=state.matchup[row[:,None],opp[:,None],indices,own_idx[:,None]].astype(jnp.float32)/8
    matchup=jnp.where(jnp.any(own_matches,axis=-1)[:,None,None],matchup,0)
    numeric=jnp.zeros((b,6,132),jnp.float32)
    numeric=numeric.at[:,:,jnp.array(MON_COLUMNS)].set(jnp.where((known&on[:,None])[:,:,None],jnp.concatenate([values,matchup],axis=-1),0))
    provisional=(state.window_move&(state.window_side==observer)&(state.window_outcome==2)).astype(jnp.int32)
    blocked=jnp.minimum(64,state.blocked[row,jnp.clip(observer,0,1)]+provisional)
    counts=jnp.stack([state.switches[row,opp],state.opportunities[row,opp],blocked,state.pursuits[row,opp],
        state.switches[row,jnp.clip(observer,0,1)]],axis=-1).astype(jnp.float32)/64
    stats=jnp.zeros((b,132),jnp.float32).at[:,jnp.array(STAT_COLUMNS)].set(jnp.concatenate([
        counts,state.weather[row,opp].reshape((b,6)).astype(jnp.float32)],axis=-1))
    stats=stats.at[:,reference.v4_numeric_index(reference.NUMERIC_PRESENT)].set(1)
    stats=jnp.where(on[:,None],stats,0)
    nmask=jnp.zeros(132,bool).at[jnp.array(MON_COLUMNS)].set(True)
    smask=jnp.zeros(132,bool).at[jnp.array((*STAT_COLUMNS,reference.v4_numeric_index(reference.NUMERIC_PRESENT)))].set(True)
    return TendencyProjection(numeric,stats,jnp.where(on,tables.stats_category,0),nmask,smask,on,
        admitted)
