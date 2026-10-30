"""Diagnostic kind20 -> ORIGINAL FULL raw words, BRN/PSN only.

This is a one-receipt format/BEFORE-ledger validator, never registration or
admission. All eligible rows retain UNKNOWN256, including coherent words.
The original emitter, aggregate refusals and every shared consumer are untouched.
TOX is ALWAYS unsupported until actual owning counter history/source-stage
mapping is proved. Its pinned lexical templates are unqualified controls only.

Private counter words are bounded/unchanged for BRN/PSN and NEVER enter public
words. No damage/counter formula, final BattleState join, faint, ordering or
lifetime-once attestation. Zero HP and incomplete public prefixes refuse.
Only format_device is a trusted pure cached-JIT primitive. Host APIs check the
original immutable helper/FULL handles outside each owning invocation.
"""
from dataclasses import dataclass
import hashlib
from typing import NamedTuple
import weakref

import jax
import jax.numpy as jnp
import numpy as np

from . import engine_status_residual_callbacks as residual
from . import engine_full_v4_event_device as full, engine_lifecycle_callbacks as life
from . import public_inputs as inputs, public_wire, v4_encoder, public_toxic
from . import public_mon_state as mon

SCHEMA = 'pokezero.status_residual_full_words.diagnostic.v1'
_STATUS = (0,2,3,5,6,1,4)
_EXTRA = ('nonproc','refresh','mudshot','lifecycle','sleep_clause','switches')
AuxBindings = NamedTuple('AuxBindings',[(n,object) for n in _EXTRA])


class Tables(NamedTuple):
    helper: residual.Data
    templates: object
    length: object
    namespaces: full.Bindings
    encoder_binding: object
    species_map: object
    extra: AuxBindings
    binding: object


@dataclass(frozen=True, eq=False)
class Compiled:
    device: Tables


class Result(NamedTuple):
    chunk: inputs.EventChunk
    format_coherent: object
    refusal: object


_COMPILED = weakref.WeakKeyDictionary()


def _host():
    if isinstance(jnp.zeros((),jnp.int32),jax.core.Tracer):
        raise ValueError('checked status formatter host boundary cannot be traced')


def compile_formatter(status_compiled, *, full_adapter):
    """Original FULL writer templates; strings/host work ONLY at registration."""
    _host()
    helper = residual.check_compiled(status_compiled)
    full.check_adapter(full_adapter)
    if full_adapter.causal.projector.catalog is not residual._COMPILED[status_compiled][4]:
        raise ValueError('same original status/engine/FULL catalog edge required')
    catalog = full_adapter.catalog;kwargs = v4_encoder.wire_kwargs(catalog)
    name = catalog.species.names[1];ident = 'p1a: '+name
    prior,_ = public_wire.project_lines(public_wire.WireState(),
        [f'|switch|{ident}|{name}, L80|100/100', '|turn|1', '|'],**kwargs)
    rows={};length=np.zeros((4,7),np.int32)
    tokens=('', 'slp','par','brn','frz','psn','tox')
    allowed={int(k) for k in (inputs.Kind.RAW_BEGIN,inputs.Kind.SWITCH_PRELUDE,
        inputs.Kind.MUDSHOT,inputs.Kind.CONDITION,inputs.Kind.TOXIC,
        inputs.Kind.NONPROC,inputs.Kind.TENDENCY,inputs.Kind.SLEEP)}
    for reason,tag in ((residual.BURN,'brn'),(residual.POISON,'psn'),(residual.TOXIC,'psn')):
        for status,token in enumerate(tokens):
            _,projection=public_wire.project_line(prior,f'|-damage|{ident}|71/100'+
                (' '+token if token else '')+'|[from] '+tag,**kwargs)
            words=np.array([e.words() for e in projection.events],np.int32)
            if (not 1<=len(words)<=64 or sum(words[:,0]==int(inputs.Kind.CONDITION))!=1
                    or any(int(k) not in allowed for k in words[:,0])):
                raise ValueError('original FULL residual schema unsupported; no truncation')
            rows[reason,status]=words;length[reason,status]=len(words)
    width=max(map(len,rows.values()));packed=np.zeros((4,7,width,16),np.int32)
    for (reason,status),words in rows.items():packed[reason,status,:len(words)]=words
    extra=[]
    for key in _EXTRA:
        table=getattr(catalog.context,key)
        if table is None:raise ValueError('original FULL context required')
        extra.append(table.binding)
    namespaces=full_adapter.device.namespaces
    species_map=full_adapter.causal.device.species
    blob=packed.tobytes()+length.tobytes()+np.asarray(helper.binding).tobytes()+SCHEMA.encode()
    blob+=np.asarray(species_map).tobytes()+np.asarray(full_adapter.device.binding).tobytes()
    blob+=b''.join(np.asarray(v).tobytes() for v in (*namespaces,catalog.binding,*extra))
    binding=jnp.asarray(np.frombuffer(hashlib.sha256(blob).digest(),dtype='>u4').astype(np.uint32))
    tables=Tables(helper,jnp.array(packed),jnp.array(length),namespaces,catalog.binding,
        species_map,AuxBindings(*extra),binding)
    compiled=Compiled(tables)
    _COMPILED[compiled]=(tables,tuple(jax.tree.leaves(tables)),status_compiled,full_adapter)
    return compiled


def check_compiled(compiled):
    _host()
    record=_COMPILED.get(compiled) if type(compiled) is Compiled else None
    if record is None or compiled.device is not record[0]:
        raise ValueError('original immutable status formatter required')
    leaves=tuple(jax.tree.leaves(compiled.device))
    if len(leaves)!=len(record[1]) or any(a is not b or not isinstance(a,jax.Array)
            or isinstance(a,jax.core.Tracer) for a,b in zip(leaves,record[1])):
        raise ValueError('original immutable status formatter edges required')
    residual.check_compiled(record[2]);full.check_adapter(record[3])
    return compiled.device


def _before(stream, public, tables, b):
    if type(stream) is not full.StreamState or type(public) is not v4_encoder.State:
        raise ValueError('typed BEFORE FULL/public state required')
    cursor=stream.causal.cursor;life._typed(cursor.stream_id,(b,),jnp.int32);life._valid(cursor)
    for value,shape,dtype in ((stream.sequence,(b,),jnp.int32),(stream.causal.sequence,(b,),jnp.int32),
            (stream.causal.owner,(b,),jnp.int32),(stream.causal.admitted,(b,),jnp.bool_),
            (stream.live,(b,2),jnp.bool_),(stream.species,(b,2,6),jnp.int32),
            (stream.status,(b,2,6),jnp.int32),(stream.hp,(b,2,6),jnp.int32),
            (stream.maximum,(b,2,6),jnp.int32),(public.binding,(b,8),jnp.uint32),
            (public.mons.species_binding,(b,8),jnp.uint32),(public.context.binding,(b,8),jnp.uint32),
            (public.cursor.source_binding,(b,8),jnp.uint32),
            (public.cursor.stream_id,(b,),jnp.int32),(public.cursor.sequence,(b,),jnp.int32),
            (public.cursor.turn,(b,),jnp.int32),(public.cursor.phase,(b,),jnp.int32),
            (public.cursor.needs_more,(b,),jnp.bool_),(public.mons.active,(b,2),jnp.int32),
            (public.mons.encounter,(b,2,6),jnp.int32),(public.mons.hp_known,(b,2,6),jnp.bool_),
            (public.mons.fainted,(b,2,6),jnp.bool_),(public.context.nonproc.residual,(b,),jnp.bool_),
            (public.context.refresh.family,(b,),jnp.int32),(public.context.toxic.visibility,(b,2),jnp.int32)):
        life._typed(value,shape,dtype)
    ok=(jnp.all(public.binding==tables.encoder_binding,axis=1)
        &jnp.all(public.cursor.source_binding==tables.namespaces.source_binding,axis=1)
        &jnp.all(public.mons.species_binding==tables.namespaces.species_binding,axis=1)
        &jnp.all(public.context.binding==tables.namespaces.context_binding,axis=1))
    for key,field in (('ability','abilities'),('move','moves'),('item','items'),('field','field'),
            ('active','active'),('damage','damage'),('type','types'),('sleep','sleep'),
            ('toxic','toxic'),('tendency','tendency')):
        value=getattr(public.context,field).binding;life._typed(value,(b,8),jnp.uint32)
        ok &= jnp.all(value==getattr(tables.namespaces,key+'_binding'),axis=1)
    for key in _EXTRA:
        value=(public.context.nonproc_binding if key=='nonproc' else
            public.context.refresh.controller.binding if key=='refresh' else getattr(public.context,key).binding)
        life._typed(value,(b,8),jnp.uint32);ok &= jnp.all(value==getattr(tables.extra,key),axis=1)
    visibility=jnp.where(jnp.arange(2)[None,:]==stream.causal.owner[:,None],public_toxic.EXACT,public_toxic.PERCENTAGE)
    ok &= (life._valid(cursor)&cursor.initialized&stream.causal.admitted
        &(stream.causal.owner>=0)&(stream.causal.owner<2)&(cursor.phase==0)
        &(public.cursor.stream_id==cursor.stream_id)&(public.cursor.sequence==stream.sequence)
        &(stream.causal.sequence==stream.sequence)&(stream.sequence>=0)
        &(public.cursor.turn==cursor.turn)&(public.cursor.phase==cursor.phase)
        &~public.cursor.needs_more&(public.context.refresh.family==0)&public.context.nonproc.residual
        &jnp.all(public.context.toxic.visibility==visibility,axis=1)
        &jnp.all(public.mons.active==cursor.public_slot,axis=1))
    for name,field in (('species','species'),('status','status'),('hp','hp_numerator'),('maximum','hp_denominator')):
        value=getattr(public.mons,field);life._typed(value,(b,2,6),jnp.int32)
        ok &= jnp.all(getattr(stream,name)==value,axis=(1,2))
    return ok


def format_device(tape,index,stream,public,tables,*,eligible):
    """Trusted one-receipt diagnostic; current BEFORE state, no final-state API."""
    if type(tables) is not Tables or not residual._check(tape,tables.helper):
        raise ValueError('actual lifecycle storage and original formatter tables required')
    b,c=tape.words.shape[:2];eligible=residual._eligible(eligible,tape.count)
    for value in (*tables.namespaces,tables.encoder_binding,*tables.extra,tables.binding):life._typed(value,(8,),jnp.uint32)
    if tables.templates.ndim!=4 or tables.templates.shape[:2]!=(4,7) or not 1<=tables.templates.shape[2]<=64:
        raise ValueError('bounded original lexical templates required')
    width=tables.templates.shape[2];life._typed(tables.templates,(4,7,width,16),jnp.int32)
    life._typed(tables.length,(4,7),jnp.int32)
    bound=_before(stream,public,tables,b)
    if tables.species_map.ndim!=1 or not tables.species_map.shape[0]:raise ValueError('original species map required')
    life._typed(tables.species_map,tables.species_map.shape,jnp.int32)
    try:index=mon._int32(index)
    except (TypeError,ValueError):index=jnp.full((b,),-1,jnp.int32)
    if index.shape not in ((),(b,)):index=jnp.full((b,),-1,jnp.int32)
    index=jnp.broadcast_to(index,(b,));rows=jnp.arange(b);at=jnp.clip(index,0,c-1)
    raw=tape.words[rows,at];side=raw[:,3];safe=jnp.clip(side,0,1)
    cursor=stream.causal.cursor;slot=cursor.public_slot[rows,safe];safe_slot=jnp.clip(slot,0,5)
    encounter=cursor.encounter[rows,safe];status=raw[:,8];reason=raw[:,9]
    revealed=cursor.revealed_species_id[rows,safe]
    canonical=(jnp.all(tables.helper.reason_by_status==jnp.array(residual._REASON_BY_STATUS,jnp.int32))
        &jnp.all(tables.helper.damage_tag_by_reason==jnp.array(residual._TAG_BY_REASON,jnp.int32)))
    hp=lambda value:jnp.clip(value,0,1_000_000)
    denominator=jnp.clip(raw[:,7],1,1_000_000)
    def percent(value):
        value=hp(value);amount=(100*value+denominator-1)//denominator
        return jnp.where((value<denominator)&(amount==100),99,amount)
    own=stream.causal.owner==side
    old_visible=jnp.where(own,raw[:,5],percent(raw[:,5]));new_visible=jnp.where(own,raw[:,6],percent(raw[:,6]))
    maximum=jnp.where(own,raw[:,7],100);public_status=jnp.array(_STATUS,jnp.int32)[jnp.clip(status,0,6)]
    size=tables.length[jnp.clip(reason,0,3),public_status]
    good=(bound&canonical&(tape.stream_id==cursor.stream_id)&(index>=0)&(index<tape.count)
        &(tape.count>=0)&(tape.count<=c)&((tape.refusal&~(1|256))==0)
        &(raw[:,0]==residual.STATUS_RESIDUAL)&(side>=0)&(side<2)
        &(raw[:,4]==cursor.physical_active[rows,safe])&(slot>=0)&(slot<6)&(encounter>0)
        &stream.live[rows,safe]&~public.mons.fainted[rows,safe,safe_slot]&public.mons.hp_known[rows,safe,safe_slot]
        &(public.mons.encounter[rows,safe,safe_slot]==encounter)
        &(revealed>0)&(revealed<tables.species_map.shape[0])
        &(public.mons.species[rows,safe,safe_slot]==tables.species_map[jnp.clip(revealed,0,tables.species_map.shape[0]-1)])
        &(tape.event_turn[rows,at]==cursor.turn)&(tape.event_phase[rows,at]==cursor.phase)
        &jnp.all(tape.event_generation[rows,at]==cursor.generation,axis=1)
        &jnp.all(tape.switch_details[rows,at]==0,axis=1)
        &((reason==residual.BURN)|(reason==residual.POISON))
        &(reason==tables.helper.reason_by_status[jnp.clip(status,0,6)])
        &(raw[:,1]>=0)&(raw[:,1]<=15)&(raw[:,2]==raw[:,1])
        &(raw[:,5]>raw[:,6])&(raw[:,6]>0)&(raw[:,7]>=raw[:,5])&(raw[:,7]<=1_000_000)
        &(public.mons.status[rows,safe,safe_slot]==public_status)
        &(public.mons.hp_numerator[rows,safe,safe_slot]==old_visible)
        &(public.mons.hp_denominator[rows,safe,safe_slot]==maximum)
        &(size>0)&(size<=64)&(stream.sequence<=2**31-1-size))
    for key in ('engine_binding','source_binding','move_binding','producer_binding'):
        good &= jnp.all(getattr(tape,key)==getattr(tables.helper,key))
    good &= jnp.all(cursor.engine_binding==tables.helper.engine_binding,axis=1)
    schema=tables.templates[jnp.clip(reason,0,3),public_status];kind=schema[:,:,0]
    words=schema.at[:,:,1].set(stream.sequence[:,None]+jnp.arange(width)[None,:]+1)
    words=words.at[:,:,2].set(cursor.turn[:,None]);words=words.at[:,:,3].set(cursor.phase[:,None])
    for side_col,slot_col in ((4,5),(6,7)):
        addressed=schema[:,:,side_col]>=0
        words=words.at[:,:,side_col].set(jnp.where(addressed,side[:,None],-1))
        words=words.at[:,:,slot_col].set(jnp.where(schema[:,:,slot_col]>=0,slot[:,None],-1))
    condition=kind==int(inputs.Kind.CONDITION);non=kind==int(inputs.Kind.NONPROC)
    toxic=kind==int(inputs.Kind.TOXIC);raw_begin=kind==int(inputs.Kind.RAW_BEGIN)
    sleep=kind==int(inputs.Kind.SLEEP);mud=kind==int(inputs.Kind.MUDSHOT)
    for col,value in ((8,encounter),(9,new_visible),(10,maximum)):
        words=words.at[:,:,col].set(jnp.where(condition,value[:,None],words[:,:,col]))
    for col,value in ((9,side),(10,slot),(11,encounter),(12,new_visible),(13,maximum)):
        words=words.at[:,:,col].set(jnp.where(non,value[:,None],words[:,:,col]))
    for col,value in ((9,encounter),(11,new_visible),(12,maximum)):
        words=words.at[:,:,col].set(jnp.where(toxic,value[:,None],words[:,:,col]))
    words=words.at[:,:,8].set(jnp.where(sleep,encounter[:,None],words[:,:,8]))
    for col,value in ((9,side),(10,slot),(11,encounter)):
        words=words.at[:,:,col].set(jnp.where(mud&(schema[:,:,9]>=0),value[:,None],words[:,:,col]))
    for col,value in ((13,new_visible),(14,maximum),(15,public_status)):
        words=words.at[:,:,col].set(jnp.where(raw_begin,value[:,None],words[:,:,col]))
    use=eligible&good;valid=use[:,None]&(jnp.arange(width)[None,:]<size[:,None])
    padded=jnp.zeros((b,64,16),jnp.int32).at[:,:width].set(jnp.where(valid[:,:,None],words,0))
    valid=jnp.pad(valid,((0,0),(0,64-width)))
    chunk=inputs.EventChunk(tape.stream_id,padded,valid,jnp.zeros(b,bool),*tables.namespaces)
    return Result(chunk,use,tape.refusal|jnp.where(eligible,256,0))


def format_receipt(tape,index,stream,public,*,compiled,eligible):
    _host()
    return format_device(tape,index,stream,public,check_compiled(compiled),eligible=eligible)
