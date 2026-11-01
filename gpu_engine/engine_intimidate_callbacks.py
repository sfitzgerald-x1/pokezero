"""Actual positive Intimidate writes, partial input coverage only.

Source raw29 is one composite receipt for the literal ability announcement
followed by the literal target atk-1 line. Original typed constructor/cursor
and accounting/risk transport are required at this owning source cutpoint;
these device values do not attest copied/fabricated constructor origin. Existing
host constructor/source-witness guards establish that actual callsite. Later
entries require their immediately preceding literal installation. Blocked,
capped/no-output attempts stay UNKNOWN. No boost carry is added.
"""
from typing import NamedTuple
import jax.numpy as jnp
from . import engine_public_events as public

INTIMIDATE=29
UNKNOWN_INTIMIDATE=256

class Write(NamedTuple):
    actor_ability:object
    target_ability:object
    target_hp:object
    target_substitute:object
    old_stage:object
    new_stage:object


def ordinary_prefix(tape,index,cursor):
    """Same-call RAW6 -> RAW29; current cursor is AFTER that installation.

    Used before emission at index=count and by the original lower scan at the
    actual receipt index. No final-state reconstruction or generic entry token.
    """
    from . import engine_lifecycle_callbacks as life
    b=tape.count.shape[0];rows=jnp.arange(b);c=tape.words.shape[1]
    index=jnp.broadcast_to(jnp.asarray(index,jnp.int32),(b,));at=jnp.clip(index-1,0,c-1)
    raw=tape.words[rows,at];side=raw[:,3];safe=jnp.clip(side,0,1)
    generation=tape.event_generation[rows,at].at[rows,safe].add(1)
    return (life._valid(cursor)&cursor.initialized&(cursor.turn>0)&(cursor.phase==0)
        &jnp.all(cursor.established,axis=1)&(tape.stream_id==cursor.stream_id)
        &(index>0)&(index<=tape.count)&(index<=c)
        &(raw[:,0]==life.SWITCH_INSTALLED)&(raw[:,1]==side)&(side>=0)&(side<2)
        &(raw[:,2]>=0)&(raw[:,2]<6)&(raw[:,4]>=0)&(raw[:,4]<6)&(raw[:,2]!=raw[:,4])
        &(raw[:,4]==cursor.physical_active[rows,safe])
        &((raw[:,5]==0)|(raw[:,5]==1))&jnp.all(raw[:,6:]==0,axis=1)
        &(tape.event_turn[rows,at]==cursor.turn)&(tape.event_phase[rows,at]==0)
        &jnp.all(generation==cursor.generation,axis=1))


def _ordinary_installation(tape,index,cursor):
    """Old immediate prefix OR exactly one owned/claimed entry-Spikes span."""
    from . import engine_entry_spikes_callbacks as spikes
    b=len(tape.count);index=jnp.broadcast_to(jnp.asarray(index,jnp.int32),(b,))
    hazard=spikes.claimed(tape,index-1,cursor)
    return ordinary_prefix(tape,index,cursor)|hazard,index-jnp.where(hazard,2,1)


def ordinary_owned(tape,index,cursor):
    """Exact positive ordinary raw29 schema/clock/installation identity."""
    b=tape.count.shape[0];rows=jnp.arange(b);index=jnp.broadcast_to(jnp.asarray(index,jnp.int32),(b,))
    at=jnp.clip(index,0,tape.words.shape[1]-1);raw=tape.words[rows,at]
    side=raw[:,1];safe=jnp.clip(side,0,1);prefix,installation=_ordinary_installation(tape,index,cursor)
    previous=tape.words[rows,jnp.clip(installation,0,tape.words.shape[1]-1)]
    return (prefix&(index<tape.count)
        &(raw[:,0]==INTIMIDATE)&(side>=0)&(side<2)&(raw[:,3]==1-side)
        &(previous[:,3]==side)&(previous[:,4]==raw[:,2])
        &(raw[:,2]==cursor.physical_active[rows,safe])&(raw[:,4]==cursor.physical_active[rows,1-safe])
        &(raw[:,5]>0)&(raw[:,6]>-6)&(raw[:,6]<=6)&(raw[:,7]==raw[:,6]-1)
        &jnp.all(raw[:,8:]==0,axis=1)&jnp.all(tape.switch_details[rows,at]==0,axis=1)
        &(tape.event_turn[rows,at]==cursor.turn)&(tape.event_phase[rows,at]==0)
        &jnp.all(tape.event_generation[rows,at]==cursor.generation,axis=1))


def record_ordinary_device(tape,ticket,side,physical,target_physical,write,table,*,requested):
    """Observe the SAME positive old->old-1 write; no numerical assignment."""
    from . import engine_execution_completeness as execution, engine_kernel_frame_risk as risks
    if type(write) is not Write:raise ValueError('literal actual Intimidate write required')
    b=tape.count.shape[0];rows=jnp.arange(b)
    if type(side) is int:side=jnp.full(b,side,jnp.int32)
    for x in (side,physical,target_physical,*write):execution._shape(x,(b,),jnp.int32)
    execution._shape(requested,(b,),jnp.bool_)
    for name in ('engine_binding','source_binding','move_binding','producer_binding'):
        execution._shape(getattr(tape,name),(8,),jnp.uint32)
    context=execution._context(tape)
    if context is None:return tape
    ledger=context.ledger;slot,same=execution._ticket(ledger,ticket)
    bound=jnp.zeros(b,bool)
    if type(context.risks) is risks.Context:
        bound,_=risks._bound(context.risks.carry,tape,context.risks.tables)
        bound &= context.risks.carry.opened&(context.risks.carry.frame==ledger.frame)
    safe=jnp.clip(side,0,1);n=table.intimidate.shape[0]
    prefix,installation=_ordinary_installation(tape,tape.count,tape.cursor)
    valid=(bound&same&execution._validate(ledger,context.tables)&(ledger.refusal==0)
        &((tape.refusal&~public.UNINSTRUMENTED)==0)
        &(ledger.site[rows,slot]==int(execution.Site.INTIMIDATE_ATTEMPT))
        &(ledger.side[rows,slot]==side)&(ledger.start[rows,slot]==tape.count)
        &(ledger.turn[rows,slot]==tape.cursor.turn)&(ledger.phase[rows,slot]==tape.cursor.phase)
        &jnp.all(ledger.generation[rows,slot]==tape.cursor.generation,axis=1)
        &prefix&(side>=0)&(side<2)
        &(physical==tape.cursor.physical_active[rows,safe])
        &(target_physical==tape.cursor.physical_active[rows,1-safe])
        &(tape.words[rows,jnp.clip(installation,0,tape.words.shape[1]-1),3]==side)
        &(write.actor_ability>0)&(write.actor_ability<n)&table.intimidate[jnp.clip(write.actor_ability,0,n-1)]
        &(write.target_ability>=0)&(write.target_ability<n)&~table.attack_drop_immune[jnp.clip(write.target_ability,0,n-1)]
        &(write.target_hp>0)&(write.target_substitute==0)&(write.old_stage>-6)&(write.old_stage<=6)
        &(write.new_stage==write.old_stage-1))
    inside=(tape.count>=0)&(tape.count<tape.words.shape[1]);at=jnp.clip(tape.count,0,tape.words.shape[1]-1)
    valid &= ~inside|jnp.all(tape.switch_details[rows,at]==0,axis=1)
    z=jnp.zeros_like(tape.count)
    tape=public._append(tape,INTIMIDATE,side,physical,1-side,target_physical,
        (write.actor_ability,write.old_stage,write.new_stage,z,z),requested&valid)
    ledger=execution.unhandled(ledger,ticket,active=requested&~valid,tables=context.tables)
    ledger=execution.emitted(ledger,ticket,tape,active=requested&valid,tables=context.tables)
    tape=execution._put_context(tape,ledger,context)
    return tape._replace(refusal=tape.refusal|jnp.where(requested&(~valid|~inside),UNKNOWN_INTIMIDATE,0))


def _constructor(tape):
    from . import engine_kernel_frame_risk as risks,engine_execution_completeness as execution
    from . import engine_lifecycle_callbacks as life
    context=execution._context(tape)
    if context is None or type(context.risks) is not risks.Context:return jnp.zeros_like(tape.count,bool)
    carry=context.risks.carry;tables=context.risks.tables
    bound,_=risks._bound(carry,tape,tables)
    # _valid performs exact schema checks, but intentionally returns False
    # during this AFTER-reveals/BEFORE-completion constructor transition.
    life._valid(tape.cursor)
    if tape.words.shape[1]<2:return jnp.zeros_like(tape.count,bool)
    cursor=tape.cursor
    mapping=jnp.full_like(cursor.physical_to_public,-1).at[:,:,0].set(0)
    initial=tape.words[:,:2]
    prefix=(jnp.all(initial[:,:,0]==life.INITIAL_REVEAL,axis=1)
        &jnp.all(initial[:,:,1:3]==-1,axis=(1,2))&jnp.all(initial[:,:,3]==jnp.arange(2),axis=1)
        &jnp.all(initial[:,:,4:]==0,axis=(1,2))
        &jnp.all(tape.event_turn[:,:2]==0,axis=1)&jnp.all(tape.event_phase[:,:2]==0,axis=1)
        &jnp.all(tape.event_generation[:,:2]==0,axis=(1,2))&(tape.count>=2))
    transition=(cursor.admitted&jnp.all(cursor.public_slot==0,axis=1)&jnp.all(cursor.encounter==1,axis=1)
        &jnp.all(cursor.next_slot==1,axis=1)&jnp.all(cursor.physical_to_public==mapping,axis=(1,2)))
    return (bound&carry.opened&(carry.frame==1)&(context.ledger.frame==1)
        &(carry.counts[:,int(risks.Branch.CONSTRUCTOR)-1]==1)
        &((carry.observed&(1<<(int(risks.Branch.CONSTRUCTOR)-1)))!=0)
        &prefix&transition&~tape.cursor.initialized&(tape.cursor.turn==0)&(tape.cursor.phase==0)
        &jnp.all(tape.cursor.generation==0,axis=1)&jnp.all(tape.cursor.established,axis=1)
        &jnp.all(tape.cursor.physical_active==0,axis=1))


def record_device(tape,ticket,side,physical,target_physical,write,table,*,requested):
    from . import engine_execution_completeness as execution
    if not hasattr(tape,'event_turn'):return tape
    if type(write) is not Write:raise ValueError('literal actual Intimidate write required')
    b=tape.count.shape[0];rows=jnp.arange(b)
    if type(side) is int:side=jnp.full(b,side,jnp.int32)
    for x in (side,physical,target_physical,*write):execution._shape(x,(b,),jnp.int32)
    execution._shape(requested,(b,),jnp.bool_)
    n=table.intimidate.shape[0];safe=jnp.clip(side,0,1)
    valid=(_constructor(tape)&(side>=0)&(side<2)&(physical==0)&(target_physical==0)
        &(physical==tape.cursor.physical_active[rows,safe])&(target_physical==tape.cursor.physical_active[rows,1-safe])
        &(write.actor_ability>0)&(write.actor_ability<n)&table.intimidate[jnp.clip(write.actor_ability,0,n-1)]
        &(write.target_ability>=0)&(write.target_ability<n)&~table.attack_drop_immune[jnp.clip(write.target_ability,0,n-1)]
        &(write.target_hp>0)&(write.target_substitute==0)&(write.old_stage==0)&(write.new_stage==-1))
    inside=(tape.count>=0)&(tape.count<tape.words.shape[1]);at=jnp.clip(tape.count,0,tape.words.shape[1]-1)
    valid &= ~inside|jnp.all(tape.switch_details[rows,at]==0,axis=1)
    context=execution._context(tape)
    if context is not None:
        ledger=context.ledger
        attempts=(jnp.arange(ledger.site.shape[1])[None,:]<ledger.count[:,None])
        attempts &= (ledger.site==int(execution.Site.INTIMIDATE_INITIAL))&(ledger.side==side[:,None])
        valid &= jnp.sum(attempts,axis=1)==1
    z=jnp.zeros_like(tape.count)
    tape=public._append(tape,INTIMIDATE,side,physical,1-side,target_physical,
        (write.actor_ability,write.old_stage,write.new_stage,z,z),requested&valid)
    context=execution._context(tape)
    if context is not None:
        ledger=context.ledger;slot,same=execution._ticket(ledger,ticket)
        source=same&(ledger.site[rows,slot]==int(execution.Site.INTIMIDATE_INITIAL))
        ledger=execution.unhandled(ledger,ticket,active=requested&~(valid&source),tables=context.tables)
        ledger=execution.emitted(ledger,ticket,tape,active=requested&valid&source,tables=context.tables)
        tape=execution._put_context(tape,ledger,context)
    return tape._replace(refusal=tape.refusal|jnp.where(requested&(~valid|~inside),UNKNOWN_INTIMIDATE,0))
