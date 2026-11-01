"""Bounded literal HIT_OUTCOME observations, NEVER an admission input.

Two original action-loop cutpoints fit one frame. Overflow/missing storage is
only diagnostic. No event, ticket, refusal or risk bit is produced here. Host
serialization includes refused rows only; these private operands are not model
features, public facts or training labels.
"""
from typing import NamedTuple
import jax.numpy as j
import numpy as np

OUTCOME_FIELDS=('power','critical','hit','immune','numerator','denominator','absorbed',
    'substitute','protected','enduring','recoil','explodes')
META_FIELDS=('move','actor_side','actor_physical','target_side','target_physical','eligible',
    'stream_id','frame','raw_count','turn','phase','generation0','generation1','prior_tape_refusal',
    'prior_execution_refusal','prior_kernel_risk')
FIELDS=OUTCOME_FIELDS+META_FIELDS
CAPACITY=2
RESIDUAL_FIELDS=('side','physical','profile_qualified','cached_speed_valid','end_speed0','end_speed1',
    'guard_valid','stream_id','frame','raw_count','turn','phase','generation0','generation1',
    'prior_tape_refusal','prior_execution_refusal','prior_kernel_risk')
BEFORE_MOVE_FIELDS=('actor_side','actor_physical','move','caller_move','eligible','hp','maximum','ability',
    'status','counter','confusion_counter','flinched','loafing','sleep_usable','early_bird',
    'thaw_roll','paralysis_roll','confusion_roll','result_status','result_counter',
    'result_confusion_counter','can_act','confusion_self_hit','stream_id','frame','raw_count',
    'turn','phase','generation0','generation1','prior_tape_refusal','prior_execution_refusal','prior_kernel_risk')

BATON_START_FIELDS=('actor_side', 'actor_physical', 'move', 'caller_move', 'eligible', 'acts', 'begin_valid', 'registered_bound', 'data_bound', 'transaction_present', 'history_side', 'public_known', 'public_pending', 'public_substitute', 'hp', 'maximum', 'status', 'counter', 'ability', 'base_ability', 'item', 'type0', 'type1', 'base_type0', 'base_type1', 'substitute_hp', 'confusion_counter', 'volatile_uint32', 'protect_counter', 'choice_move', 'encore_move', 'encore_counter', 'charge_move', 'charge_counter', 'spikes', 'any_trap_source', 'pending_residual', 'any_forced_switch', 'any_baton_pass', 'ability_namespace_size', 'item_namespace_size', 'quiet_ability_lookup', 'quiet_item_lookup', 'boost0', 'boost1', 'boost2', 'boost3', 'boost4', 'boost5', 'boost6', 'public_boost0', 'public_boost1', 'public_boost2', 'public_boost3', 'public_boost4', 'public_boost5', 'public_boost6', 'stream_id', 'frame', 'raw_count', 'turn', 'phase', 'generation0', 'generation1', 'prior_tape_refusal', 'prior_execution_refusal', 'prior_kernel_risk', 'quiet_status_lookup')

class Capture(NamedTuple):
    words:object
    valid:object
    count:object
    overflow:object
    residual_words:object=None
    residual_valid:object=None
    residual_count:object=None
    residual_overflow:object=None
    before_move_words:object=None
    before_move_valid:object=None
    before_move_count:object=None
    before_move_overflow:object=None
    baton_start_words:object=None
    baton_start_known:object=None
    baton_start_valid:object=None
    baton_start_count:object=None
    baton_start_overflow:object=None

def empty(batch):
    return Capture(j.zeros((batch,CAPACITY,len(FIELDS)),j.int32),j.zeros((batch,CAPACITY),bool),
        j.zeros(batch,j.int32),j.zeros(batch,bool),
        j.zeros((batch,CAPACITY,len(RESIDUAL_FIELDS)),j.int32),j.zeros((batch,CAPACITY),bool),
        j.zeros(batch,j.int32),j.zeros(batch,bool),
        j.zeros((batch,CAPACITY,len(BEFORE_MOVE_FIELDS)),j.int32),j.zeros((batch,CAPACITY),bool),
        j.zeros(batch,j.int32),j.zeros(batch,bool),
        j.zeros((batch,CAPACITY,len(BATON_START_FIELDS)),j.int32),j.zeros((batch,CAPACITY,len(BATON_START_FIELDS)),bool),
        j.zeros((batch,CAPACITY),bool),j.zeros(batch,j.int32),j.zeros(batch,bool))

def _valid(value,b):
    return type(value) is Capture and all(getattr(a,'shape',None)==shape and getattr(a,'dtype',None)==j.dtype(dtype)
        for a,shape,dtype in zip(value,((b,CAPACITY,len(FIELDS)),(b,CAPACITY),(b,),(b,)),(j.int32,j.bool_,j.int32,j.bool_)))

def record_on_tape(tape,outcome,move,side,physical,target,*,eligible,active):
    """Store the SAME literal Outcome before its original risk observer.

    A missing or malformed optional diagnostic leaves the entire tape unchanged.
    Only active loop rows consume slots; eligible is recorded rather than used
    to erase blocked/nonacting outcome operands.
    """
    context=getattr(tape,'execution',None);capture=getattr(context,'diagnostics',None)
    b=len(tape.count)
    if not _valid(capture,b) or getattr(context,'risks',None) is None:return tape
    values=tuple(outcome)+(move,side,physical,1-side,target,eligible,tape.stream_id,context.ledger.frame,tape.count,tape.cursor.turn,
        tape.cursor.phase,tape.cursor.generation[:,0],tape.cursor.generation[:,1],tape.refusal,
        context.ledger.refusal,context.risks.carry.risk)
    if len(values)!=len(FIELDS) or any(getattr(v,'shape',None)!=(b,) for v in values):return tape
    if getattr(active,'shape',None)!=(b,) or getattr(active,'dtype',None)!=j.dtype(bool):return tape
    words=j.stack([v.astype(j.int32) for v in values],axis=1)
    rows=j.arange(b);slot=j.clip(capture.count,0,CAPACITY-1);room=(capture.count>=0)&(capture.count<CAPACITY)
    write=active&room
    updated=capture._replace(words=capture.words.at[rows,slot].set(j.where(write[:,None],words,capture.words[rows,slot])),
        valid=capture.valid.at[rows,slot].set(capture.valid[rows,slot]|write),
        count=j.where(write,capture.count+1,capture.count),overflow=capture.overflow|(active&~room))
    return tape._replace(execution=context._replace(diagnostics=updated))


def _valid_residual(value,b):
    return _valid(value,b) and all(getattr(a,'shape',None)==shape and getattr(a,'dtype',None)==j.dtype(dtype)
        for a,shape,dtype in zip(value[4:8],((b,CAPACITY,len(RESIDUAL_FIELDS)),(b,CAPACITY),(b,),(b,)),(j.int32,j.bool_,j.int32,j.bool_)))

def _valid_before_move(value,b):
    return _valid(value,b) and all(getattr(a,'shape',None)==shape and getattr(a,'dtype',None)==j.dtype(dtype)
        for a,shape,dtype in zip(value[8:12],((b,CAPACITY,len(BEFORE_MOVE_FIELDS)),(b,CAPACITY),(b,),(b,)),(j.int32,j.bool_,j.int32,j.bool_)))

def _valid_baton_start(value,b):
    return _valid(value,b) and all(getattr(a,'shape',None)==shape and getattr(a,'dtype',None)==j.dtype(dtype)
        for a,shape,dtype in zip(value[12:17],((b,CAPACITY,len(BATON_START_FIELDS)),(b,CAPACITY,len(BATON_START_FIELDS)),(b,CAPACITY),(b,),(b,)),(j.int32,j.bool_,j.bool_,j.int32,j.bool_)))

def record_baton_start_on_tape(tape,values,*,active):
    """Literal source operands only; unknown fields have explicit known=False.

    Missing/malformed optional storage is inert. No gate consumes this data.
    Uint32 volatile bits are stored losslessly as int32 and restored on host.
    """
    context=getattr(tape,'execution',None);capture=getattr(context,'diagnostics',None);b=len(tape.count)
    if not _valid_baton_start(capture,b)or type(values)is not dict:return tape
    if getattr(active,'shape',None)!=(b,)or getattr(active,'dtype',None)!=j.dtype(bool):return tape
    words=[];known=[]
    for name in BATON_START_FIELDS:
        value=values.get(name);mask=j.ones(b,bool)
        if type(value)is tuple and len(value)==2:value,mask=value
        valid=(getattr(value,'shape',None)==(b,)and getattr(value,'dtype',None)in(j.dtype(j.int32),j.dtype(j.uint32),j.dtype(bool))and getattr(mask,'shape',None)==(b,)and getattr(mask,'dtype',None)==j.dtype(bool))
        words.append(value.astype(j.int32)if valid else j.zeros(b,j.int32));known.append(mask if valid else j.zeros(b,bool))
    words=j.stack(words,axis=1);known=j.stack(known,axis=1);rows=j.arange(b);slot=j.clip(capture.baton_start_count,0,CAPACITY-1);room=(capture.baton_start_count>=0)&(capture.baton_start_count<CAPACITY);write=active&room
    updated=capture._replace(baton_start_words=capture.baton_start_words.at[rows,slot].set(j.where(write[:,None],words,capture.baton_start_words[rows,slot])),
        baton_start_known=capture.baton_start_known.at[rows,slot].set(j.where(write[:,None],known,capture.baton_start_known[rows,slot])),
        baton_start_valid=capture.baton_start_valid.at[rows,slot].set(capture.baton_start_valid[rows,slot]|write),
        baton_start_count=j.where(write,capture.baton_start_count+1,capture.baton_start_count),baton_start_overflow=capture.baton_start_overflow|(active&~room))
    return tape._replace(execution=context._replace(diagnostics=updated))

def record_before_move_on_tape(tape,before,side,physical,move,caller_move,status,counter,confusion_counter,
        *,hp,maximum,ability,thaw_roll,paralysis_roll,confusion_roll,flinched,
        sleep_usable,early_bird,loafing,eligible,active):
    """SAME original BeforeMove inputs/result before any status assignment.

    The original call is pure and work is unchanged at this immediate cutpoint.
    No blocker classification, source claim, public event or risk is produced.
    Missing/malformed/overflow storage is diagnostic-only and never a refusal.
    """
    context=getattr(tape,'execution',None);capture=getattr(context,'diagnostics',None);b=len(tape.count)
    if not _valid_before_move(capture,b) or getattr(context,'risks',None)is None:return tape
    if any(not hasattr(before,name)for name in ('status','counter','confusion_counter','can_act','confusion_self_hit')):return tape
    integers=(side,physical,move,caller_move,hp,maximum,ability,status,counter,confusion_counter,
        thaw_roll,paralysis_roll,confusion_roll,before.status,before.counter,before.confusion_counter)
    booleans=(flinched,loafing,sleep_usable,early_bird,eligible,active,before.can_act,before.confusion_self_hit)
    if any(getattr(v,'shape',None)!=(b,)or getattr(v,'dtype',None)!=j.dtype(j.int32)for v in integers):return tape
    if any(getattr(v,'shape',None)!=(b,)or getattr(v,'dtype',None)!=j.dtype(bool)for v in booleans):return tape
    values=(side,physical,move,caller_move,eligible,hp,maximum,ability,status,counter,confusion_counter,
        flinched,loafing,sleep_usable,early_bird,thaw_roll,paralysis_roll,confusion_roll,
        before.status,before.counter,before.confusion_counter,before.can_act,before.confusion_self_hit,
        tape.stream_id,context.ledger.frame,tape.count,tape.cursor.turn,tape.cursor.phase,
        tape.cursor.generation[:,0],tape.cursor.generation[:,1],tape.refusal,context.ledger.refusal,context.risks.carry.risk)
    if any(getattr(v,'shape',None)!=(b,)for v in values):return tape
    words=j.stack([v.astype(j.int32)for v in values],axis=1)
    rows=j.arange(b);slot=j.clip(capture.before_move_count,0,CAPACITY-1)
    room=(capture.before_move_count>=0)&(capture.before_move_count<CAPACITY);write=active&room
    updated=capture._replace(before_move_words=capture.before_move_words.at[rows,slot].set(j.where(write[:,None],words,capture.before_move_words[rows,slot])),
        before_move_valid=capture.before_move_valid.at[rows,slot].set(capture.before_move_valid[rows,slot]|write),
        before_move_count=j.where(write,capture.before_move_count+1,capture.before_move_count),
        before_move_overflow=capture.before_move_overflow|(active&~room))
    return tape._replace(execution=context._replace(diagnostics=updated))

def record_residual_on_tape(tape,side,physical,qualified,cached_speed_valid,end_speed,guard_valid,*,active):
    """SAME original residual guard operands, before emitter/ledger changes.

    No recomputation, source ticket, refusal, risk or policy input. A missing or
    malformed optional store is inert; overflow is diagnostic-only. Exactly
    the actually requested residual sites consume the two bounded slots.
    """
    context=getattr(tape,'execution',None);capture=getattr(context,'diagnostics',None);b=len(tape.count)
    if not _valid_residual(capture,b) or getattr(context,'risks',None)is None:return tape
    if getattr(end_speed,'shape',None)!=(b,2) or getattr(end_speed,'dtype',None)!=j.dtype(j.int32):return tape
    if any(getattr(v,'shape',None)!=(b,)or getattr(v,'dtype',None)!=j.dtype(bool)for v in(qualified,cached_speed_valid,guard_valid,active)):return tape
    if any(getattr(v,'shape',None)!=(b,)or getattr(v,'dtype',None)!=j.dtype(j.int32)for v in(side,physical)):return tape
    values=(side,physical,qualified,cached_speed_valid,end_speed[:,0],end_speed[:,1],guard_valid,
        tape.stream_id,context.ledger.frame,tape.count,tape.cursor.turn,tape.cursor.phase,
        tape.cursor.generation[:,0],tape.cursor.generation[:,1],tape.refusal,context.ledger.refusal,context.risks.carry.risk)
    if any(getattr(v,'shape',None)!=(b,)for v in values):return tape
    words=j.stack([v.astype(j.int32)for v in values],axis=1)
    rows=j.arange(b);slot=j.clip(capture.residual_count,0,CAPACITY-1)
    room=(capture.residual_count>=0)&(capture.residual_count<CAPACITY);write=active&room
    updated=capture._replace(residual_words=capture.residual_words.at[rows,slot].set(j.where(write[:,None],words,capture.residual_words[rows,slot])),
        residual_valid=capture.residual_valid.at[rows,slot].set(capture.residual_valid[rows,slot]|write),
        residual_count=j.where(write,capture.residual_count+1,capture.residual_count),
        residual_overflow=capture.residual_overflow|(active&~room))
    return tape._replace(execution=context._replace(diagnostics=updated))

def baton_start_guard_values(record):
    """Named nonexclusive guard facts from literal operands; None is unknown."""
    def calc(names,fn):
        values=[record.get(n)for n in names]
        return None if any(type(v)not in(int,bool)for v in values)else bool(fn(*values))
    result={}
    for key,field,want in [('acts','acts',1),('registered_bound','registered_bound',1),('data_bound','data_bound',1),('transaction_present','transaction_present',1),('public_known','public_known',1),('public_not_pending','public_pending',0),('pending_residual_clear','pending_residual',0),('forced_switches_clear','any_forced_switch',0),('baton_flags_clear','any_baton_pass',0),('spikes_clear','spikes',0),('trap_sources_clear','any_trap_source',0)]:
        result[key]=calc((field,),lambda x,want=want:x==want)
    result['history_empty']=calc(('history_side',),lambda x:x<0)
    result['ordinary_caller']=calc(('caller_move','move'),lambda x,y:x==y)
    result['quiet_status']=(calc(('quiet_status_lookup',),lambda x:x==1)if 'quiet_status_lookup'in record else calc(('status','counter'),lambda s,c:(s==0 and c==0)or(s==4 and 0<=c<=15)or(s==1 and c==0)))
    result['living_actor']=calc(('hp',),lambda x:x>0)
    result['quiet_ability']=calc(('ability','ability_namespace_size','quiet_ability_lookup'),lambda i,n,v:0<i<n and v==1)
    result['quiet_item']=calc(('item','item_namespace_size','quiet_item_lookup'),lambda i,n,v:0<=i<n and v==1)
    result['original_ability']=calc(('ability','base_ability'),lambda x,y:x==y)
    result['original_types']=calc(('type0','type1','base_type0','base_type1'),lambda a,b,c,d:a==c and b==d)
    result['substitute_range']=calc(('substitute_hp','maximum'),lambda x,m:0<=x<=m<=1000000)
    for field in ('confusion_counter','volatile_uint32','protect_counter','choice_move','encore_move','encore_counter','charge_move','charge_counter'):
        result[field+'_clear']=calc((field,),lambda x:x==0)
    result['public_substitute_equal']=calc(('public_substitute','substitute_hp'),lambda p,x:p==int(x>0))
    fields=tuple('boost'+str(i)for i in range(7))+tuple('public_boost'+str(i)for i in range(7))
    result['public_boosts_equal']=calc(fields,lambda *v:v[:7]==v[7:])
    return result

def refused_rows(capture,refused,*,_host_cache=None):
    """Copy each used array once, then serialize ONLY refused host rows.

    The ephemeral identity cache is local to this serialization call. No JAX
    row indexing/gather, extra barrier, or policy/public consumer is introduced.
    Empty/missing/malformed diagnostic storage retains its original semantics.
    """
    refused=np.asarray(refused)
    if refused.ndim!=1 or refused.dtype!=np.dtype(bool):raise ValueError('bool[B] refused rows required')
    indices=np.flatnonzero(refused);result=[]
    available=_valid(capture,len(refused));residual_available=_valid_residual(capture,len(refused))if available else False
    before_available=_valid_before_move(capture,len(refused))if available else False
    baton_available=_valid_baton_start(capture,len(refused))if available else False
    baton_present=type(capture)is Capture and any(x is not None for x in capture[12:17])
    cache={}if _host_cache is None else _host_cache
    def host(value):
        key=id(value)
        if key not in cache:cache[key]=(value,np.asarray(value))
        return cache[key][1]
    copied=tuple(host(value)for value in capture[:4])if available and len(indices)else None
    residual_copied=tuple(host(value)for value in capture[4:8])if residual_available and len(indices)else None
    before_copied=tuple(host(value)for value in capture[8:12])if before_available and len(indices)else None
    baton_copied=tuple(host(value)for value in capture[12:17])if baton_available and len(indices)else None
    for row in indices:
        item={'row':int(row),'available':available}
        if available:
            words,valid,count,overflow=copied
            item.update(count=int(count[row]),overflow=bool(overflow[row]),
                records=[dict(zip(FIELDS,map(int,record)))for record in words[row][valid[row]]])
            residual={'available':residual_available,'capacity':CAPACITY}
            if residual_available:
                words,valid,count,overflow=residual_copied
                residual.update(count=int(count[row]),overflow=bool(overflow[row]),
                    records=[dict(zip(RESIDUAL_FIELDS,map(int,record)))for record in words[row][valid[row]]])
            item['residual']=residual
            if before_available:
                words,valid,count,overflow=before_copied
                item['before_move']=dict(available=True,capacity=CAPACITY,count=int(count[row]),overflow=bool(overflow[row]),
                    records=[dict(zip(BEFORE_MOVE_FIELDS,map(int,record)))for record in words[row][valid[row]]])
        if baton_present:
            item['baton_start']=dict(available=baton_available,capacity=CAPACITY,scope='Literal same-call operands; nonexclusive guard facts, never causal admission.')
            if baton_available:
                words,known,valid,count,overflow=baton_copied;records=[]
                for record,mask in zip(words[row][valid[row]],known[row][valid[row]],strict=True):
                    values={name:(int(value)if yes else None)for name,value,yes in zip(BATON_START_FIELDS,record,mask,strict=True)}
                    if values['volatile_uint32']is not None:values['volatile_uint32']&=0xffffffff
                    values['guard_values']=baton_start_guard_values(values)
                    records.append(values)
                item['baton_start'].update(count=int(count[row]),overflow=bool(overflow[row]),records=records)
            else:item['baton_start']['reason']='MISSING_OR_MALFORMED_STORAGE'
        result.append(item)
    return dict(schema='pokezero.literal-refused-outcomes.v1',capacity=CAPACITY,rows=result)
