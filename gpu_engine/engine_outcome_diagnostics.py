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

class Capture(NamedTuple):
    words:object
    valid:object
    count:object
    overflow:object

def empty(batch):
    return Capture(j.zeros((batch,CAPACITY,len(FIELDS)),j.int32),j.zeros((batch,CAPACITY),bool),
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
    updated=Capture(capture.words.at[rows,slot].set(j.where(write[:,None],words,capture.words[rows,slot])),
        capture.valid.at[rows,slot].set(capture.valid[rows,slot]|write),
        j.where(write,capture.count+1,capture.count),capture.overflow|(active&~room))
    return tape._replace(execution=context._replace(diagnostics=updated))

def refused_rows(capture,refused):
    """Copy only refused row slices, including explicit missing/overflow state."""
    refused=np.asarray(refused)
    if refused.ndim!=1 or refused.dtype!=np.dtype(bool):raise ValueError('bool[B] refused rows required')
    result=[]
    for row in np.flatnonzero(refused):
        item={'row':int(row),'available':_valid(capture,len(refused))}
        if item['available']:
            valid=np.asarray(capture.valid[row],bool)
            item.update(count=int(np.asarray(capture.count[row])),overflow=bool(np.asarray(capture.overflow[row])),
                records=[dict(zip(FIELDS,map(int,record))) for record in np.asarray(capture.words[row])[valid]])
        result.append(item)
    return dict(schema='pokezero.literal-refused-outcomes.v1',capacity=CAPACITY,rows=result)
