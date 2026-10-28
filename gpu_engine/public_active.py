"""Public active volatile bags and elapsed clocks for retained v4 columns.

This observes the champion parser's rules, including its documented Mean Look
Baton Pass clearing and the different `-end Wrap`/`-end partiallytrapped` clocks.
Optional species-bound Truant tracks known versus unknown public phases. Toxic,
sleep, damage and live types have separate consumers.
Lifecycle facts require the canonical, resolved encounter from the caller; the
raw adapter below deliberately cannot declare a whole protocol line consumed.
"""
from dataclasses import dataclass
import hashlib
import json
from types import MappingProxyType
from typing import Mapping, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from pokezero import showdown as reference
from .private_team import vocabulary_binding
from .public_abilities import _bool
from .public_mon_state import _int32
from .public_moves import _static_category

SCHEMA = 'pokezero.public_active.v3'
ARRIVE, FAINT, TURN, EXPIRE_ACTION, ANIMATION, ADD, REMOVE, WRAP_END, STALL_ATTEMPT, STALL_SUCCESS, STALL_FAIL, STALL_BREAK, RECHARGE, CLEAR_RECHARGE, TRAPPED, PERISH_START, ITEM_SET, ITEM_END, CHOICE_MOVE = range(19)
UPKEEP,TRUANT_TRACE,TRUANT_CANT=range(19,22)
MAX_OPERATION=TRUANT_CANT
CLOCK_NAMES = ('confusion', 'encore', 'partiallytrapped')
CORE = ('STALL_COUNTER', 'CONFUSION_TURNS', 'ENCORE_TURNS', 'WRAP_TRAP_TURNS', 'MEANLOOK_TRAP',
        'CHOICE_LOCKED', 'ITEM_SWAPPED')
NUMERIC_COLUMNS = tuple(reference.v4_numeric_index(getattr(reference, 'NUMERIC_'+n)) for n in CORE)
CATEGORY_COLUMNS = tuple(range(reference.CATEGORY_VOLATILE_OFFSET,
    reference.CATEGORY_VOLATILE_OFFSET+reference.VOLATILE_BUCKET_COUNT))
NAMES = ('',)+tuple(sorted(reference.TRACKED_VOLATILES | {reference.MUST_RECHARGE_VOLATILE}))
RECHARGE_ID = NAMES.index(reference.MUST_RECHARGE_VOLATILE)
TRUANT_COLUMN=reference.v4_numeric_index(reference.NUMERIC_TRUANT_LOAF)


class ActiveTables(NamedTuple):
    category: jax.Array
    transferable: jax.Array
    expires_action: jax.Array
    charge: jax.Array
    perish: jax.Array
    clocks: jax.Array
    vocabulary_binding: jax.Array
    binding: jax.Array
    native_truant: jax.Array | None = None
    public_species_binding: jax.Array | None = None


@dataclass(frozen=True)
class ActiveCatalog:
    names: tuple[str, ...]
    ids: Mapping[str, int]
    device: ActiveTables


def compile_active(vocabulary,*,species_catalog=None):
    names = NAMES
    ids = {n:i for i,n in enumerate(names)}
    category = [0]+[_static_category(vocabulary, 'volatile:'+n) for n in names[1:]]
    masks = [[n in values for n in names] for values in (
        reference._BATON_PASS_TRANSFERRED_VOLATILES,
        reference._SINGLEMOVE_VOLATILES | reference._CHARGE_MOVE_VOLATILES,
        reference._CHARGE_MOVE_VOLATILES, reference._PERISH_COUNTERS)]
    clocks = [ids[n] for n in CLOCK_NAMES]
    vb = vocabulary_binding(vocabulary)
    payload = dict(schema=SCHEMA,names=names,category=category,masks=masks,clocks=clocks,
        vocabulary=np.asarray(vb).tolist(),stall_cap=reference._STALL_COUNTER_CAP,
        choice_items=sorted(reference._CHOICE_ITEMS))
    native=sb=None
    if species_catalog is not None:
        if not np.array_equal(np.asarray(vb),np.asarray(species_catalog.device.vocabulary_binding)):
            raise ValueError('Truant and public species require the same frozen vocabulary')
        native=jnp.asarray([reference._normalize_identifier(n) in reference._TRUANT_SPECIES
            for n in species_catalog.names],bool)
        sb=species_catalog.device.binding
        payload.update(native_truant=np.asarray(native).tolist(),species=np.asarray(sb).tolist())
    digest = hashlib.sha256(json.dumps(payload,sort_keys=True,separators=(',',':')).encode()).digest()
    binding = jnp.asarray(np.frombuffer(digest,dtype='>u4').astype(np.uint32))
    return ActiveCatalog(names,MappingProxyType(ids),ActiveTables(jnp.asarray(category,jnp.int32),
        *(jnp.asarray(m,bool) for m in masks),jnp.asarray(clocks,jnp.int32),vb,binding,native,sb))


class ActiveState(NamedTuple):
    binding: jax.Array
    species_binding: jax.Array
    encounter: jax.Array       # [B,2], current public encounter; zero before an arrival
    turn: jax.Array
    volatiles: jax.Array       # [B,2,V], recharge is kept in its own flag below
    elapsed: jax.Array         # [B,2,3], public turns-so-far, not hidden duration
    stall: jax.Array
    stall_pending: jax.Array
    recharge: jax.Array
    meanlook: jax.Array
    choice_public: jax.Array    # active slot's publicly announced Choice Band
    choice_locked: jax.Array
    item_swapped: jax.Array     # parser's public Trick provenance, not inferred mutation
    post_upkeep: jax.Array      # [B], true only after the canonical upkeep marker
    truant_known: jax.Array
    truant_loaf: jax.Array
    truant_skip_flip: jax.Array
    traced_truant: jax.Array


def _binding(value):
    if getattr(value,'shape',None)!=(8,) or getattr(value,'dtype',None)!=jnp.dtype(jnp.uint32):
        raise ValueError('active inputs require eight uint32 namespace words')
    return value


def initialize(batch, tables, *, species_binding):
    if type(batch) is not int or batch<=0: raise ValueError('positive active batch required')
    sb = _binding(species_binding)
    if (tables.native_truant is None)!=(tables.public_species_binding is None):
        raise ValueError('Truant tables require the frozen public species namespace')
    if tables.native_truant is not None and not np.array_equal(np.asarray(sb),np.asarray(tables.public_species_binding)):
        raise ValueError('Truant initialization requires its compiled public species namespace')
    z = jnp.zeros((batch,2),jnp.int32);b = jnp.zeros((batch,2),bool)
    return ActiveState(jnp.broadcast_to(tables.binding,(batch,8)),jnp.broadcast_to(sb,(batch,8)),
        z,jnp.zeros(batch,jnp.int32),jnp.zeros((batch,2,tables.category.shape[0]),bool),
        jnp.zeros((batch,2,3),jnp.int32),z,b,b,b,b,b,b,jnp.zeros(batch,bool),b,b,b,b)


def _shape(state, tables, mons):
    if state.turn.ndim!=1 or not state.turn.shape[0]: raise ValueError('nonempty active batch required')
    batch=state.turn.shape[0]
    for name in state._fields:
        arr=getattr(state,name)
        shape=(batch,8) if name in ('binding','species_binding') else (batch,) if name in ('turn','post_upkeep') else (
            (batch,2,tables.category.shape[0]) if name=='volatiles' else (batch,2,3) if name=='elapsed' else (batch,2))
        dtype=jnp.uint32 if name in ('binding','species_binding') else jnp.bool_ if name in (
            'volatiles','stall_pending','recharge','meanlook','choice_public','choice_locked','item_swapped',
            'post_upkeep','truant_known','truant_loaf','truant_skip_flip','traced_truant') else jnp.int32
        if arr.shape!=shape or arr.dtype!=dtype: raise ValueError('active state shape or dtype: '+name)
    for name,shape,dtype in (('species_binding',(batch,8),jnp.uint32),
        ('active',(batch,2),jnp.int32),('encounter',(batch,2,6),jnp.int32)):
        arr=getattr(mons,name)
        if arr.shape!=shape or arr.dtype!=dtype: raise ValueError('public encounter shape or dtype: '+name)
    if (tables.native_truant is None)!=(tables.public_species_binding is None):
        raise ValueError('Truant tables require the frozen public species namespace')
    if tables.native_truant is not None:
        _binding(tables.public_species_binding)
        if tables.native_truant.ndim!=1 or not tables.native_truant.shape[0] or tables.native_truant.dtype!=jnp.bool_:
            raise ValueError('Truant species membership must be a nonempty boolean vector')
        if mons.species.shape!=(batch,2,6) or mons.species.dtype!=jnp.int32:
            raise ValueError('Truant requires exact public species words')
    return batch


def _row(value,batch,*,boolean=False):
    arr=_bool(value) if boolean else _int32(value)
    if arr.shape not in ((),(batch,)): raise ValueError('active fact must be scalar or exact batch vector')
    return jnp.broadcast_to(arr,(batch,))


def _current(mons):
    batch=mons.active.shape[0]
    ids=jnp.take_along_axis(mons.encounter,jnp.clip(mons.active,0,5)[...,None],axis=-1)[...,0]
    ok=jnp.all((mons.active>=-1)&(mons.active<6)&((mons.active<0)|(ids>0)),axis=-1)
    return jnp.where(mons.active>=0,ids,0),ok


def _coherent(state,tables,mons):
    current,ok=_current(mons)
    ok &= jnp.all(state.binding==tables.binding,axis=-1)&jnp.all(state.species_binding==mons.species_binding,axis=-1)
    ok &= (state.turn>=0)&jnp.all(state.encounter>=0,axis=-1)
    ok &= jnp.all((state.stall>=0)&(state.stall<=reference._STALL_COUNTER_CAP),axis=-1)
    ok &= jnp.all(state.elapsed>=0,axis=(-1,-2))&~jnp.any(state.volatiles[...,0],axis=-1)
    ok &= ~jnp.any(state.volatiles[...,RECHARGE_ID],axis=-1)
    ok &= ~jnp.any(state.choice_locked&~state.choice_public,axis=-1)
    if tables.native_truant is not None:
        species=jnp.take_along_axis(mons.species,jnp.clip(mons.active,0,5)[...,None],axis=-1)[...,0]
        present=current>0
        native=tables.native_truant[jnp.clip(species,0,len(tables.native_truant)-1)]&present
        ok &= jnp.all(state.species_binding==tables.public_species_binding,axis=-1)
        ok &= jnp.all(~present|((species>0)&(species<len(tables.native_truant))),axis=-1)
        # A just-arrived mon is already in mons but the active consumer still
        # owns its predecessor until ARRIVE commits. Other operations reject
        # that encounter mismatch below; do not reject a legitimate lifecycle.
        same=state.encounter==current
        ok &= ~jnp.any(same&state.truant_known&~(present&(native|state.traced_truant)),axis=-1)
        ok &= ~jnp.any(state.truant_loaf&~state.truant_known,axis=-1)
        ok &= ~jnp.any(state.truant_skip_flip&~state.post_upkeep[:,None],axis=-1)
    else:
        ok &= ~state.post_upkeep&~jnp.any(state.truant_known|state.truant_loaf|state.truant_skip_flip|state.traced_truant,axis=-1)
    return current,ok


def apply(state,tables,mons,*,operation,side=-1,encounter=0,value=0,baton_pass=False,
          producer_binding,valid=True):
    """Apply a resolved public fact; malformed rows leave every field unchanged.

    ARRIVE runs after the public encounter ledger arrival. Other targeted facts
    require that same current encounter. TURN requires all lifecycle changes
    already committed; chronology is contiguous, as in the ordered event fold.
    """
    batch=_shape(state,tables,mons);_binding(producer_binding)
    op,side,enc,value=(_row(v,batch) for v in (operation,side,encounter,value))
    bp,valid=(_row(v,batch,boolean=True) for v in (baton_pass,valid))
    current,ok=_coherent(state,tables,mons)
    row=jnp.arange(batch);seat=jnp.clip(side,0,1);address=(row,seat)
    turn=op==TURN;upkeep=op==UPKEEP;arrive=op==ARRIVE;faint=op==FAINT
    global_fact=turn|upkeep
    vcount=tables.category.shape[0]
    ok &= jnp.all(producer_binding==tables.binding)&(op>=ARRIVE)&(op<=MAX_OPERATION)
    ok &= jnp.where(global_fact,(side==-1)&(enc==0)&jnp.where(turn,
        (state.turn<2**31-1)&(value==state.turn+1),value==0),
        (side>=0)&(side<2)&(enc>0)&(current[address]==enc))
    # Only the targeted arrival may account for an encounter-ledger change.
    same=state.encounter==current
    same=jnp.where(arrive[:,None]&(jnp.arange(2)==seat[:,None]),True,same)
    ok &= jnp.all(same,axis=-1)&(~bp|arrive)
    named=(op==ADD)|(op==REMOVE)|(op==PERISH_START)
    ok &= jnp.where(named,(value>0)&(value<vcount),jnp.where((op==STALL_ATTEMPT)|(op==CHOICE_MOVE)|(op==TRUANT_TRACE),
        (value==0)|(value==1),jnp.where(op==ITEM_SET,(value>=0)&(value<=3),jnp.where(turn,True,value==0))))
    # Recharge enters only through its own public producer, not a generic bag write.
    ok &= ~named|(value!=RECHARGE_ID)
    ok &= (op!=PERISH_START)|tables.perish[jnp.clip(value,0,vcount-1)]
    if tables.native_truant is None:ok &= op<UPKEEP
    increments=state.volatiles[...,tables.clocks]
    ok &= ~turn|jnp.all(~increments|(state.elapsed<2**31-1),axis=(-1,-2))
    use=ok&valid
    old=state.volatiles[address];mask=jnp.arange(vcount)==value[:,None]
    bag=jnp.where(arrive[:,None],old&bp[:,None]&tables.transferable,old)
    bag=jnp.where(faint[:,None],False,bag)
    bag &= ~( ((op==EXPIRE_ACTION)[:,None]&tables.expires_action) |
              ((op==ANIMATION)[:,None]&tables.charge))
    bag=jnp.where((op==PERISH_START)[:,None],bag&~tables.perish,bag)
    bag=jnp.where(((op==ADD)|(op==PERISH_START))[:,None],bag|mask,jnp.where((op==REMOVE)[:,None],bag&~mask,bag))
    elapsed=state.elapsed[address]
    reset=(faint[:,None]|(arrive[:,None]&~bag[:,tables.clocks])|
        ((op==REMOVE)[:,None]&(value[:,None]==tables.clocks)&(jnp.arange(3)<2))|
        ((op==WRAP_END)[:,None]&(jnp.arange(3)==2)))
    elapsed=jnp.where(reset,0,elapsed)
    elapsed_all=state.elapsed.at[address].set(elapsed)
    elapsed_all=jnp.where(turn[:,None,None],state.elapsed+increments.astype(jnp.int32),elapsed_all)
    break_stall=arrive|faint|(op==STALL_BREAK)|((op==STALL_ATTEMPT)&(value==0))|((op==STALL_FAIL)&state.stall_pending[address])
    stall=jnp.where(break_stall,0,jnp.where(op==STALL_SUCCESS,jnp.minimum(reference._STALL_COUNTER_CAP,state.stall[address]+1),state.stall[address]))
    pending=jnp.where(break_stall|(op==STALL_SUCCESS),False,
        jnp.where(op==STALL_ATTEMPT,value==1,state.stall_pending[address]))
    recharge=jnp.where(arrive|faint|(op==CLEAR_RECHARGE),False,jnp.where(op==RECHARGE,True,state.recharge[address]))
    mean=jnp.where((arrive|faint)[:,None],False,state.meanlook)
    mean=mean.at[address].set(jnp.where(op==TRAPPED,True,mean[address]))
    reset_item=arrive|faint|(op==ITEM_END)
    choice=jnp.where(reset_item,False,jnp.where(op==ITEM_SET,(value&1)==1,state.choice_public[address]))
    locked=jnp.where(reset_item|(op==ITEM_SET),False,
        state.choice_locked[address]|((op==CHOICE_MOVE)&(value==1)&state.choice_public[address]))
    swapped=jnp.where(reset_item,False,jnp.where(op==ITEM_SET,(value&2)==2,state.item_swapped[address]))
    post=state.post_upkeep;known=state.truant_known;loaf=state.truant_loaf
    skip=state.truant_skip_flip;traced=state.traced_truant
    if tables.native_truant is not None:
        native=tables.native_truant[jnp.clip(mons.species[row,seat,jnp.clip(mons.active[address],0,5)],0,len(tables.native_truant)-1)]
        holder=native|traced[address]
        anchor_act=(op==CHOICE_MOVE)&(value==1)&holder
        anchor_loaf=(op==TRUANT_CANT)&holder
        unknown=op==TRUANT_TRACE
        known_value=jnp.where(arrive,native,jnp.where(unknown,False,jnp.where(anchor_act|anchor_loaf,True,known[address])))
        loaf_value=jnp.where(arrive,native&(state.turn!=0),jnp.where(unknown|anchor_act,False,jnp.where(anchor_loaf,True,loaf[address])))
        skip_value=jnp.where(arrive&native,post,jnp.where(unknown&(value==1),False,skip[address]))
        known=known.at[address].set(known_value);loaf=loaf.at[address].set(loaf_value)
        skip=skip.at[address].set(skip_value)
        traced=traced.at[address].set(jnp.where(arrive,False,jnp.where(unknown,value==1,traced[address])))
        flip=turn[:,None]&(value[:,None]>=2)&known&~skip
        loaf=jnp.where(flip,~loaf,loaf)
        skip=jnp.where(turn[:,None],False,skip)
        post=jnp.where(turn,False,jnp.where(upkeep,True,post))
    candidate=state._replace(encounter=state.encounter.at[address].set(jnp.where(arrive,enc,state.encounter[address])),
        turn=jnp.where(turn,value,state.turn),volatiles=state.volatiles.at[address].set(bag),elapsed=elapsed_all,
        stall=state.stall.at[address].set(stall),stall_pending=state.stall_pending.at[address].set(pending),
        recharge=state.recharge.at[address].set(recharge),meanlook=mean,
        choice_public=state.choice_public.at[address].set(choice),
        choice_locked=state.choice_locked.at[address].set(locked),item_swapped=state.item_swapped.at[address].set(swapped),
        post_upkeep=post,truant_known=known,truant_loaf=loaf,truant_skip_flip=skip,traced_truant=traced)
    return jax.tree.map(lambda before,after:jnp.where(use.reshape((batch,)+(1,)*(before.ndim-1)),after,before),state,candidate),use


@dataclass(frozen=True)
class ActiveProjection:
    raw_line: str
    facts: tuple[tuple[int,int,int], ...]  # operation, public side, value
    binding: tuple[int, ...]
    full_line_consumers_complete: bool=False


def project_line(line,*,catalog):
    """Named non-lifecycle consumers only; caller must resolve current identity.

    Canonical arrivals and faints are separate joint lifecycle operations. A
    `-end partiallytrapped` removes the bag entry but intentionally retains the
    elapsed clock, exactly like the frozen Python writer's current rules.
    """
    if not isinstance(line,str) or not line.startswith('|'): raise ValueError('public protocol line required')
    p=line.split('|');kind=p[1] if len(p)>1 else '';facts=[]
    if kind=='request': raise ValueError('private request in public active stream')
    if kind=='turn':
        n=reference._canonical_turn_number(line)
        if n is None or n>=2**31: raise ValueError('canonical int32 turn required')
        facts.append((TURN,-1,n))
    elif len(p)>=3:
        slot=reference._slot_from_ident(p[2]);side=int(slot[1])-1 if slot in ('p1','p2') else -1
        name=reference._side_condition_identifier(p[3]) if len(p)>=4 else ''
        def add(op,val=0):
            if side<0: raise ValueError('active fact requires public side')
            facts.append((op,side,val))
        if (kind=='move' and len(p)>=4) or kind=='cant' or reference._is_confusion_self_hit(p):
            add(EXPIRE_ACTION)
        if kind=='move' and len(p)>=4:
            add(STALL_ATTEMPT,int(reference._normalize_identifier(p[3]) in reference._STALL_MOVE_IDS))
            add(CHOICE_MOVE,int(not any(part.startswith('[from]') for part in p[4:])))
        elif kind=='cant':
            add(STALL_BREAK)
            if name=='recharge': add(CLEAR_RECHARGE)
            if catalog.device.native_truant is not None and 'Truant' in line:add(TRUANT_CANT)
        elif kind=='-anim': add(ANIMATION)
        elif kind=='-mustrecharge': add(RECHARGE)
        elif kind=='-singleturn' and len(p)>=4 and reference._is_stall_singleturn(p[3]): add(STALL_SUCCESS)
        elif kind=='-fail': add(STALL_FAIL)
        elif kind=='-item' and len(p)>=4:
            add(ITEM_SET,int(reference._normalize_identifier(p[3]) in reference._CHOICE_ITEMS)+2*int('[from] move: Trick' in line))
        elif kind=='-enditem':add(ITEM_END)
        elif kind=='-ability' and len(p)>=4 and 'ability: Trace' in line and p[3].strip() and catalog.device.native_truant is not None:
            add(TRUANT_TRACE,int(reference._normalize_identifier(p[3].strip())=='truant'))
        if kind=='-activate' and name=='trapped': add(TRAPPED)
        if kind=='-activate' and name in reference._PARTIAL_TRAP_MOVES: add(ADD,catalog.ids['partiallytrapped'])
        elif kind=='-end' and name in reference._PARTIAL_TRAP_MOVES:
            add(REMOVE,catalog.ids['partiallytrapped']);add(WRAP_END)
        elif ((kind in ('-start','-end','-singlemove') and name in reference.TRACKED_VOLATILES) or
              (kind=='-prepare' and name in reference._CHARGE_MOVE_VOLATILES)):
            op=REMOVE if kind=='-end' else PERISH_START if kind=='-start' and name in reference._PERISH_COUNTERS else ADD
            add(op,catalog.ids[name])
    return ActiveProjection(line,tuple(facts),tuple(int(x) for x in np.asarray(catalog.device.binding)))


class ActiveFeaturePatch(NamedTuple):
    numeric: jax.Array           # [B,2,132], public side order; active mon only
    categorical: jax.Array      # [B,2,41]
    numeric_columns: jax.Array
    categorical_columns: jax.Array
    overflow: jax.Array         # [B,2], source truncates sorted bags at six entries
    admitted: jax.Array


def project(state,tables,mons,*,expected_vocabulary):
    batch=_shape(state,tables,mons);_binding(expected_vocabulary)
    current,ok=_coherent(state,tables,mons)
    ok &= jnp.all(state.encounter==current,axis=-1)&jnp.all(expected_vocabulary==tables.vocabulary_binding)
    present=current>0
    numbers=jnp.concatenate((jnp.minimum(1.,state.stall[...,None]/8.),
        jnp.minimum(1.,state.elapsed/jnp.asarray([5.,6.,5.])),state.meanlook[...,None].astype(jnp.float32),
        state.choice_locked[...,None].astype(jnp.float32),state.item_swapped[...,None].astype(jnp.float32)),axis=-1)
    numeric=jnp.zeros((batch,2,132),jnp.float32).at[...,jnp.asarray(NUMERIC_COLUMNS)].set(jnp.where(present[...,None],numbers,0.))
    if tables.native_truant is not None:
        numeric=numeric.at[...,TRUANT_COLUMN].set(jnp.where(present&state.truant_known,state.truant_loaf,False).astype(jnp.float32))
    bag=state.volatiles.at[...,RECHARGE_ID].set(state.recharge)
    order=jnp.sort(jnp.where(bag,jnp.arange(tables.category.shape[0]),tables.category.shape[0]),axis=-1)[...,:reference.VOLATILE_BUCKET_COUNT]
    cats=jnp.where(order<tables.category.shape[0],tables.category[jnp.minimum(order,tables.category.shape[0]-1)],0)
    categorical=jnp.zeros((batch,2,41),jnp.int32).at[...,jnp.asarray(CATEGORY_COLUMNS)].set(jnp.where(present[...,None],cats,0))
    nm=jnp.zeros(132,bool).at[jnp.asarray(NUMERIC_COLUMNS)].set(True)
    if tables.native_truant is not None:nm=nm.at[TRUANT_COLUMN].set(True)
    cm=jnp.zeros(41,bool).at[jnp.asarray(CATEGORY_COLUMNS)].set(True)
    return ActiveFeaturePatch(jnp.where(ok[:,None,None],numeric,0.),jnp.where(ok[:,None,None],categorical,0),
        nm,cm,(jnp.sum(bag,axis=-1)>reference.VOLATILE_BUCKET_COUNT)&present,ok)
