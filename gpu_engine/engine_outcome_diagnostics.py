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

class Capture(NamedTuple):
    words:object
    valid:object
    count:object
    overflow:object
    residual_words:object=None
    residual_valid:object=None
    residual_count:object=None
    residual_overflow:object=None

def empty(batch):
    return Capture(j.zeros((batch,CAPACITY,len(FIELDS)),j.int32),j.zeros((batch,CAPACITY),bool),
        j.zeros(batch,j.int32),j.zeros(batch,bool),
        j.zeros((batch,CAPACITY,len(RESIDUAL_FIELDS)),j.int32),j.zeros((batch,CAPACITY),bool),
        j.zeros(batch,j.int32),j.zeros(batch,bool))

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
        for a,shape,dtype in zip(value[4:],((b,CAPACITY,len(RESIDUAL_FIELDS)),(b,CAPACITY),(b,),(b,)),(j.int32,j.bool_,j.int32,j.bool_)))

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
    cache={}if _host_cache is None else _host_cache
    def host(value):
        key=id(value)
        if key not in cache:cache[key]=(value,np.asarray(value))
        return cache[key][1]
    copied=tuple(host(value)for value in capture[:4])if available and len(indices)else None
    residual_copied=tuple(host(value)for value in capture[4:])if residual_available and len(indices)else None
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
        result.append(item)
    return dict(schema='pokezero.literal-refused-outcomes.v1',capacity=CAPACITY,rows=result)
