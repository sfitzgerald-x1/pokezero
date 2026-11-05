"""Bounded literal HIT_OUTCOME observations, NEVER an admission input.

Two original action-loop cutpoints fit one frame. Overflow/missing storage is
only diagnostic. No event, ticket, refusal or risk bit is produced here. Host
serialization includes refused rows only; these private operands are not model
features, public facts or training labels.
"""
from typing import NamedTuple
import jax
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
SLEEP_GUARD_FIELDS=('sleep_rest_origin','sleep_skipped','sleep_volatile_uint32',
    'sleep_block_requested','sleep_block_valid','sleep_block_move_ok')

BATON_START_FIELDS=('actor_side', 'actor_physical', 'move', 'caller_move', 'eligible', 'acts', 'begin_valid', 'registered_bound', 'data_bound', 'transaction_present', 'history_side', 'public_known', 'public_pending', 'public_substitute', 'hp', 'maximum', 'status', 'counter', 'ability', 'base_ability', 'item', 'type0', 'type1', 'base_type0', 'base_type1', 'substitute_hp', 'confusion_counter', 'volatile_uint32', 'protect_counter', 'choice_move', 'encore_move', 'encore_counter', 'charge_move', 'charge_counter', 'spikes', 'any_trap_source', 'pending_residual', 'any_forced_switch', 'any_baton_pass', 'ability_namespace_size', 'item_namespace_size', 'quiet_ability_lookup', 'quiet_item_lookup', 'boost0', 'boost1', 'boost2', 'boost3', 'boost4', 'boost5', 'boost6', 'public_boost0', 'public_boost1', 'public_boost2', 'public_boost3', 'public_boost4', 'public_boost5', 'public_boost6', 'stream_id', 'frame', 'raw_count', 'turn', 'phase', 'generation0', 'generation1', 'prior_tape_refusal', 'prior_execution_refusal', 'prior_kernel_risk', 'quiet_status_lookup')

NO_DAMAGE_FIELDS=('actor_side', 'actor_physical', 'target_physical', 'move', 'caller_move', 'requested', 'accuracy_hit', 'move_type', 'ability', 'status', 'hp', 'maximum', 'volatile_uint32', 'substitute_hp', 'actor_hp', 'explodes', 'actor_hp_type', 'actor_hp_power', 'target_type0', 'target_type1', 'item', 'absorption_blocked', 'absorption_hp', 'absorption_volatile_uint32', 'write_power', 'write_critical', 'write_hit', 'write_immune', 'write_numerator', 'write_denominator', 'write_substitute', 'write_protected', 'write_enduring', 'write_old_hp', 'write_new_hp', 'write_maximum', 'current_actor_hp', 'current_volatile_uint32', 'current_substitute_hp', 'current_actor_hp_type', 'current_actor_hp_power', 'current_type0', 'current_type1', 'current_item', 'current_ability', 'move_namespace_size', 'ability_namespace_size', 'item_namespace_size', 'candidate', 'selected_type', 'dynamic_type', 'levitate_source', 'levitate_move', 'ground_type', 'flying_type', 'levitate_lookup', 'allowed_item_lookup', 'finish_scheduled', 'header_code', 'finish_matches', 'room', 'use', 'stream_id', 'frame', 'raw_count', 'turn', 'phase', 'generation0', 'generation1', 'prior_tape_refusal', 'prior_execution_refusal', 'prior_kernel_risk')

# Two independent original call phases, each bounded to the two seats/frame.
DIRECT_PHASES=('emitter','span_capture')
DIRECT_FIELDS=('actor_side', 'actor_physical', 'target_physical', 'move', 'requested', 'registered_bound', 'effective_eligible', 'write_power', 'write_critical', 'write_hit', 'write_immune', 'write_numerator', 'write_denominator', 'write_substitute', 'write_protected', 'write_enduring', 'write_old_hp', 'write_new_hp', 'write_maximum', 'critical_annotation', 'effect_annotation', 'need', 'room', 'valid_before_details', 'valid', 'use', 'overflow', 'capacity', 'move_namespace_size', 'seismic_toss_move', 'cursor_valid', 'cursor_initialized', 'stream_match', 'engine_namespace', 'source_namespace', 'move_namespace', 'producer_namespace', 'cursor_namespace', 'clock_match', 'phase_zero', 'actor_nonnegative', 'actor_current', 'target_current', 'write_valid', 'detail0_inside', 'detail0_zero', 'detail1_inside', 'detail1_zero', 'ticket_same', 'ticket_slot', 'ticket_serial', 'ticket_frame', 'ledger_slot', 'ledger_count', 'ledger_site', 'ledger_side', 'ledger_stage', 'ledger_start', 'ledger_end', 'ledger_opened', 'ledger_turn', 'ledger_phase', 'ledger_generation0', 'ledger_generation1', 'namespace_valid', 'namespace_tape', 'declared_header', 'declared_current', 'declared_stamp', 'declared', 'length', 'controls', 'exact_hp', 'site_match', 'side_match', 'span_count', 'semantic', 'declaration_word0', 'declaration_word1', 'declaration_word2', 'declaration_word3', 'declaration_word4', 'declaration_word5', 'declaration_word6', 'declaration_word7', 'declaration_word8', 'declaration_word9', 'first_word0', 'first_word1', 'first_word2', 'first_word3', 'first_word4', 'first_word5', 'first_word6', 'first_word7', 'first_word8', 'first_word9', 'second_word0', 'second_word1', 'second_word2', 'second_word3', 'second_word4', 'second_word5', 'second_word6', 'second_word7', 'second_word8', 'second_word9', 'hp_word0', 'hp_word1', 'hp_word2', 'hp_word3', 'hp_word4', 'hp_word5', 'hp_word6', 'hp_word7', 'hp_word8', 'hp_word9', 'stream_id', 'frame', 'raw_count', 'turn', 'phase', 'generation0', 'generation1', 'prior_tape_refusal', 'prior_execution_refusal', 'prior_kernel_risk')

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
    no_damage_words:object=None
    no_damage_known:object=None
    no_damage_malformed:object=None
    no_damage_valid:object=None
    no_damage_count:object=None
    no_damage_overflow:object=None
    sleep_guard_words:object=None
    sleep_guard_known:object=None
    direct_words:object=None
    direct_known:object=None
    direct_malformed:object=None
    direct_valid:object=None
    direct_count:object=None
    direct_overflow:object=None
    direct_unsigned:object=None

def empty(batch):
    return Capture(j.zeros((batch,CAPACITY,len(FIELDS)),j.int32),j.zeros((batch,CAPACITY),bool),
        j.zeros(batch,j.int32),j.zeros(batch,bool),
        j.zeros((batch,CAPACITY,len(RESIDUAL_FIELDS)),j.int32),j.zeros((batch,CAPACITY),bool),
        j.zeros(batch,j.int32),j.zeros(batch,bool),
        j.zeros((batch,CAPACITY,len(BEFORE_MOVE_FIELDS)),j.int32),j.zeros((batch,CAPACITY),bool),
        j.zeros(batch,j.int32),j.zeros(batch,bool),
        j.zeros((batch,CAPACITY,len(BATON_START_FIELDS)),j.int32),j.zeros((batch,CAPACITY,len(BATON_START_FIELDS)),bool),
        j.zeros((batch,CAPACITY),bool),j.zeros(batch,j.int32),j.zeros(batch,bool),
        j.zeros((batch,CAPACITY,len(NO_DAMAGE_FIELDS)),j.int32),j.zeros((batch,CAPACITY,len(NO_DAMAGE_FIELDS)),bool),
        j.zeros((batch,CAPACITY,len(NO_DAMAGE_FIELDS)),bool),j.zeros((batch,CAPACITY),bool),j.zeros(batch,j.int32),j.zeros(batch,bool),
        j.zeros((batch,CAPACITY,len(SLEEP_GUARD_FIELDS)),j.int32),j.zeros((batch,CAPACITY,len(SLEEP_GUARD_FIELDS)),bool),
        j.zeros((batch,len(DIRECT_PHASES),CAPACITY,len(DIRECT_FIELDS)),j.int32),j.zeros((batch,len(DIRECT_PHASES),CAPACITY,len(DIRECT_FIELDS)),bool),
        j.zeros((batch,len(DIRECT_PHASES),CAPACITY,len(DIRECT_FIELDS)),bool),j.zeros((batch,len(DIRECT_PHASES),CAPACITY),bool),
        j.zeros((batch,len(DIRECT_PHASES)),j.int32),j.zeros((batch,len(DIRECT_PHASES)),bool),
        j.zeros((batch,len(DIRECT_PHASES),CAPACITY,len(DIRECT_FIELDS)),bool))

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

def _valid_no_damage(value,b):
    return _valid(value,b) and all(getattr(a,'shape',None)==shape and getattr(a,'dtype',None)==j.dtype(dtype)
        for a,shape,dtype in zip(value[17:23],((b,CAPACITY,len(NO_DAMAGE_FIELDS)),)*3+((b,CAPACITY),(b,),(b,)),(j.int32,j.bool_,j.bool_,j.bool_,j.int32,j.bool_)))

def direct_enabled(tape):
    """Storage-shape capability only, never an eligibility/admission predicate."""
    capture=getattr(getattr(tape,'execution',None),'diagnostics',None);b=len(tape.count)
    return _valid_direct(capture,b) and all(isinstance(v,(jax.Array,jax.core.Tracer))for v in capture[25:32])

def _valid_direct(value,b):
    shape=(b,len(DIRECT_PHASES),CAPACITY,len(DIRECT_FIELDS))
    return _valid(value,b) and getattr(value.direct_unsigned,'shape',None)==shape and getattr(value.direct_unsigned,'dtype',None)==j.dtype(bool) and all(getattr(a,'shape',None)==sh and getattr(a,'dtype',None)==j.dtype(dt)
        for a,sh,dt in zip(value[25:31],(shape,shape,shape,(b,len(DIRECT_PHASES),CAPACITY),(b,len(DIRECT_PHASES)),(b,len(DIRECT_PHASES))),(j.int32,j.bool_,j.bool_,j.bool_,j.int32,j.bool_)))

def direct_meta(tape):
    """Literal current cutpoint metadata; absent values stay unavailable."""
    c=getattr(tape,'execution',None);cursor=getattr(tape,'cursor',None);ledger=getattr(c,'ledger',None);risks=getattr(c,'risks',None)
    generation=getattr(cursor,'generation',None);b=len(tape.count)
    col=lambda n:generation[:,n]if getattr(generation,'shape',None)==(b,2)else None
    return dict(stream_id=getattr(tape,'stream_id',None),frame=getattr(ledger,'frame',None),raw_count=tape.count,
        turn=getattr(cursor,'turn',None),phase=getattr(cursor,'phase',None),generation0=col(0),generation1=col(1),
        prior_tape_refusal=getattr(tape,'refusal',None),prior_execution_refusal=getattr(ledger,'refusal',None),prior_kernel_risk=getattr(getattr(risks,'carry',None),'risk',None))

def record_direct_on_tape(tape,values,*,phase,active):
    """Observe already-evaluated original operands/results, no guard evaluation.

    Each phase has two slots. Known/malformed masks distinguish unavailable
    legacy operands from literal false/zero; no field is read by admission.
    """
    c=getattr(tape,'execution',None);capture=getattr(c,'diagnostics',None);b=len(tape.count)
    if not direct_enabled(tape)or type(values)is not dict or type(phase)is not int or not 0<=phase<len(DIRECT_PHASES):return tape
    if getattr(active,'shape',None)!=(b,)or getattr(active,'dtype',None)!=j.dtype(bool):return tape
    words=[];known=[];malformed=[];unsigned=[]
    for name in DIRECT_FIELDS:
        v=values.get(name);missing=v is None;good=getattr(v,'shape',None)==(b,)and getattr(v,'dtype',None)in(j.dtype(j.int32),j.dtype(j.uint32),j.dtype(bool))
        words.append(v.astype(j.int32)if good else j.zeros(b,j.int32));known.append(j.full(b,good,bool));malformed.append(j.full(b,not missing and not good,bool));unsigned.append(j.full(b,good and getattr(v,'dtype',None)==j.dtype(j.uint32),bool))
    words=j.stack(words,axis=1);known=j.stack(known,axis=1);malformed=j.stack(malformed,axis=1);unsigned=j.stack(unsigned,axis=1);rows=j.arange(b)
    count=capture.direct_count[:,phase];slot=j.clip(count,0,CAPACITY-1);room=(count>=0)&(count<CAPACITY);write=active&room
    def put(a,v):return a.at[rows,phase,slot].set(j.where(write[:,None],v,a[rows,phase,slot]))
    updated=capture._replace(direct_words=put(capture.direct_words,words),direct_known=put(capture.direct_known,known),direct_malformed=put(capture.direct_malformed,malformed),direct_unsigned=put(capture.direct_unsigned,unsigned),
        direct_valid=capture.direct_valid.at[rows,phase,slot].set(capture.direct_valid[rows,phase,slot]|write),
        direct_count=capture.direct_count.at[:,phase].set(j.where(write,count+1,count)),direct_overflow=capture.direct_overflow.at[:,phase].set(capture.direct_overflow[:,phase]|(active&~room)))
    return tape._replace(execution=c._replace(diagnostics=updated))

def record_no_damage_on_tape(tape,p,w,*,scheduled,code,matched,room,use,actor_hp,volatile,sub_hp,actor_hp_type,actor_hp_power,target_types,item,ability):
    """Original finish operands and already-evaluated results, never admission.

    Missing optional values and malformed shapes have separate diagnostic
    masks. Storage absence/overflow changes no public/source/risk fact.
    """
    c=getattr(tape,'execution',None);capture=getattr(c,'diagnostics',None);b=len(tape.count)
    if not _valid_no_damage(capture,b):return tape
    active=getattr(p,'requested',None)
    if getattr(active,'shape',None)!=(b,)or getattr(active,'dtype',None)!=j.dtype(bool):return tape
    before=p.before;data=p.context.data.no_damage;values={}
    for name,field in [('actor_side','actor'),('actor_physical','physical'),('target_physical','target'),('move','move'),('caller_move','caller_move'),('requested','requested')]:values[name]=getattr(p,field,None)
    for name,field in [('accuracy_hit','accuracy_hit'),('move_type','move_type'),('ability','ability'),('status','status'),('hp','hp'),('maximum','maximum'),('volatile_uint32','volatile'),('substitute_hp','sub_hp'),('actor_hp','actor_hp'),('explodes','explodes'),('actor_hp_type','actor_hp_type'),('actor_hp_power','actor_hp_power'),('item','item')]:values[name]=getattr(before,field,None)
    def column(value,i):
        if value is None:return None
        return value[:,i]if getattr(value,'shape',None)==(b,2)else j.zeros((b,1),j.int32)
    for name,value in [('target_type',before.target_types),('current_type',target_types)]:
        for i in range(2):values[name+str(i)]=column(value,i)
    values.update(absorption_blocked=p.absorbed.blocked,absorption_hp=p.absorbed.hp,absorption_volatile_uint32=p.absorbed.volatile)
    for name,value in zip(w._fields,w):values['write_'+name]=value
    values.update(current_actor_hp=actor_hp,current_volatile_uint32=volatile,current_substitute_hp=sub_hp,current_actor_hp_type=actor_hp_type,current_actor_hp_power=actor_hp_power,current_item=item,current_ability=ability)
    def lookup(table,ids):
        if table is None or ids is None:return None
        if getattr(table,'ndim',None)!=1 or not len(table)or getattr(ids,'shape',None)!=(b,)or getattr(ids,'dtype',None)!=j.dtype(j.int32):return j.zeros((b,1),j.int32)
        return table[j.clip(ids,0,len(table)-1)],(ids>=0)&(ids<len(table))
    for name,table in [('candidate',data.candidates),('selected_type',data.move_types),('dynamic_type',data.dynamic_type)]:values[name]=lookup(table,p.move)
    values.update(move_namespace_size=j.full(b,len(data.candidates),j.int32),ability_namespace_size=j.full(b,len(data.levitate),j.int32),item_namespace_size=j.full(b,len(data.levitate_items),j.int32))
    values.update(levitate_source=j.broadcast_to(data.levitate_source,(b,)),levitate_move=j.broadcast_to(data.levitate_move,(b,)),ground_type=j.broadcast_to(data.ground_type,(b,)),flying_type=j.broadcast_to(data.flying_type,(b,)),levitate_lookup=lookup(data.levitate,before.ability),allowed_item_lookup=lookup(data.levitate_items,before.item))
    values.update(finish_scheduled=scheduled,header_code=code,finish_matches=matched,room=room,use=use,stream_id=tape.stream_id,frame=c.ledger.frame,raw_count=tape.count,turn=tape.cursor.turn,phase=tape.cursor.phase,generation0=tape.cursor.generation[:,0],generation1=tape.cursor.generation[:,1],prior_tape_refusal=tape.refusal,prior_execution_refusal=c.ledger.refusal,prior_kernel_risk=None if c.risks is None else c.risks.carry.risk)
    words=[];known=[];malformed=[]
    for name in NO_DAMAGE_FIELDS:
        value=values.get(name);mask=j.ones(b,bool);missing=value is None
        if type(value)is tuple and len(value)==2:value,mask=value
        valid=getattr(value,'shape',None)==(b,)and getattr(value,'dtype',None)in(j.dtype(j.int32),j.dtype(j.uint32),j.dtype(bool))and getattr(mask,'shape',None)==(b,)and getattr(mask,'dtype',None)==j.dtype(bool)
        words.append(value.astype(j.int32)if valid else j.zeros(b,j.int32));known.append(mask if valid else j.zeros(b,bool));malformed.append(j.full(b,not missing and not valid,bool))
    words=j.stack(words,axis=1);known=j.stack(known,axis=1);malformed=j.stack(malformed,axis=1);rows=j.arange(b);slot=j.clip(capture.no_damage_count,0,CAPACITY-1);room=(capture.no_damage_count>=0)&(capture.no_damage_count<CAPACITY);write=active&room
    updated=capture._replace(no_damage_words=capture.no_damage_words.at[rows,slot].set(j.where(write[:,None],words,capture.no_damage_words[rows,slot])),no_damage_known=capture.no_damage_known.at[rows,slot].set(j.where(write[:,None],known,capture.no_damage_known[rows,slot])),no_damage_malformed=capture.no_damage_malformed.at[rows,slot].set(j.where(write[:,None],malformed,capture.no_damage_malformed[rows,slot])),no_damage_valid=capture.no_damage_valid.at[rows,slot].set(capture.no_damage_valid[rows,slot]|write),no_damage_count=j.where(write,capture.no_damage_count+1,capture.no_damage_count),no_damage_overflow=capture.no_damage_overflow|(active&~room))
    return tape._replace(execution=c._replace(diagnostics=updated))

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

def record_sleep_guard_on_tape(tape,rest_origin,skipped,volatile,requested,valid,move_ok,*,eligible):
    """Literal begin operands/results aligned to the next BeforeMove slot.

    No count, ticket, word, risk or admission value changes. Missing storage or
    malformed operands stay unknown; no original private fact is inferred.
    """
    context=getattr(tape,'execution',None);capture=getattr(context,'diagnostics',None);b=len(tape.count)
    if not _valid_before_move(capture,b)or not _valid_sleep_guards(capture,b):return tape
    if getattr(eligible,'shape',None)!=(b,)or getattr(eligible,'dtype',None)!=j.dtype(bool):return tape
    supplied=(rest_origin,skipped,volatile,requested,valid,move_ok);expected=(j.bool_,j.int32,j.uint32,j.bool_,j.bool_,j.bool_)
    values=[];known=[]
    for value,dtype in zip(supplied,expected,strict=True):
        good=getattr(value,'shape',None)==(b,)and getattr(value,'dtype',None)==j.dtype(dtype)
        values.append(value.astype(j.int32)if good else j.zeros(b,j.int32))
        known.append(j.full((b,),good,bool))
    extra=j.stack(values,axis=1);mask=j.stack(known,axis=1);rows=j.arange(b)
    slot=j.clip(capture.before_move_count,0,CAPACITY-1)
    write=eligible&(capture.before_move_count>=0)&(capture.before_move_count<CAPACITY)
    updated=capture._replace(
        sleep_guard_words=capture.sleep_guard_words.at[rows,slot].set(j.where(write[:,None],extra,capture.sleep_guard_words[rows,slot])),
        sleep_guard_known=capture.sleep_guard_known.at[rows,slot].set(j.where(write[:,None],mask,capture.sleep_guard_known[rows,slot])))
    return tape._replace(execution=context._replace(diagnostics=updated))


def _valid_sleep_guards(value,b):
    return type(value)is Capture and all(getattr(v,'shape',None)==(b,CAPACITY,len(SLEEP_GUARD_FIELDS))
        and getattr(v,'dtype',None)==j.dtype(dtype)
        for v,dtype in ((value.sleep_guard_words,j.int32),(value.sleep_guard_known,j.bool_)))

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

def no_damage_groups(record):
    """Nonexclusive diagnostic groups from stored literals, not cause labels."""
    def calc(names,fn):
        values=[record.get(n)for n in names]
        return None if any(type(v)not in(int,bool)for v in values)else bool(fn(*values))
    groups={}
    for name in ('candidate','levitate_source','levitate_lookup','allowed_item_lookup','finish_scheduled','finish_matches','room','use'):
        groups[name]=calc((name,),lambda x:x==1)
    groups['ordinary_caller']=calc(('move','caller_move'),lambda x,y:x==y)
    groups['canonical_levitate_move']=calc(('move','levitate_move'),lambda x,y:y>0 and x==y)
    groups['non_flying_target']=calc(('target_type0','target_type1','flying_type'),lambda x,y,f:x!=f and y!=f)
    groups['static_type_join']=calc(('move_type','selected_type','dynamic_type'),lambda x,y,d:x==y)if record.get('dynamic_type')==0 else None
    groups['zero_volatile']=calc(('volatile_uint32',),lambda x:x==0)
    groups['target_private_conserved']=calc(('hp','write_old_hp','write_new_hp','maximum','write_maximum'),lambda h,o,n,m,w:h==o==n and m==w)
    groups['current_type_item_ability_join']=calc(('target_type0','target_type1','item','ability','current_type0','current_type1','current_item','current_ability'),lambda *v:v[:4]==v[4:])
    return groups


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
    no_damage_available=_valid_no_damage(capture,len(refused))if available else False
    no_damage_present=type(capture)is Capture and any(x is not None for x in capture[17:23])
    direct_available=_valid_direct(capture,len(refused))if available else False
    direct_present=type(capture)is Capture and any(x is not None for x in capture[25:32])
    cache={}if _host_cache is None else _host_cache
    def host(value):
        key=id(value)
        if key not in cache:cache[key]=(value,np.asarray(value))
        return cache[key][1]
    copied=tuple(host(value)for value in capture[:4])if available and len(indices)else None
    residual_copied=tuple(host(value)for value in capture[4:8])if residual_available and len(indices)else None
    before_copied=tuple(host(value)for value in capture[8:12])if before_available and len(indices)else None
    baton_copied=tuple(host(value)for value in capture[12:17])if baton_available and len(indices)else None
    no_damage_copied=tuple(host(value)for value in capture[17:23])if no_damage_available and len(indices)else None
    sleep_guard_copied=tuple(host(value)for value in capture[23:25])if before_available and _valid_sleep_guards(capture,len(refused))and len(indices)else None
    direct_copied=tuple(host(value)for value in capture[25:32])if direct_available and len(indices)else None
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
                if sleep_guard_copied is not None:
                    item['before_move']['sleep_guard_operands_available']=True
                    extra,known=sleep_guard_copied
                    for record,values,mask in zip(item['before_move']['records'],extra[row][valid[row]],known[row][valid[row]],strict=True):
                        record.update({name:(int(value)if yes else None)for name,value,yes in zip(SLEEP_GUARD_FIELDS,values,mask,strict=True)})
                        if record['sleep_volatile_uint32']is not None:record['sleep_volatile_uint32']&=0xffffffff
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
        if no_damage_present:
            result_data=dict(available=no_damage_available,capacity=CAPACITY,scope='Actual finish operands/header code/scheduled/final result. Nonexclusive diagnostic groups, never causes or admission.')
            if no_damage_available:
                words,known,malformed,valid,count,overflow=no_damage_copied;records=[]
                for record,mask,bad in zip(words[row][valid[row]],known[row][valid[row]],malformed[row][valid[row]],strict=True):
                    values={name:int(value)if yes else None for name,value,yes in zip(NO_DAMAGE_FIELDS,record,mask,strict=True)}
                    for n in ('volatile_uint32','absorption_volatile_uint32','current_volatile_uint32'):
                        if values[n]is not None:values[n]&=0xffffffff
                    values['malformed_fields']=[n for n,v in zip(NO_DAMAGE_FIELDS,bad,strict=True)if v]
                    values['unavailable_fields']=[n for n,v in values.items()if v is None]
                    values['diagnostic_groups']=no_damage_groups(values);records.append(values)
                result_data.update(count=int(count[row]),overflow=bool(overflow[row]),records=records)
            else:result_data['reason']='MISSING_OR_MALFORMED_STORAGE'
            item['no_damage_finish']=result_data
        if direct_present:
            direct_result=dict(available=direct_available,capacity_per_phase=CAPACITY,scope='Literal SAME-call emitter and capture checks. Independent bounded phases; false/missing/overflow are observations, not causes or admission.')
            if direct_available:
                words,known,malformed,valid,count,overflow,unsigned=direct_copied
                for phase,name in enumerate(DIRECT_PHASES):
                    records=[]
                    for record,mask,bad,types in zip(words[row,phase][valid[row,phase]],known[row,phase][valid[row,phase]],malformed[row,phase][valid[row,phase]],unsigned[row,phase][valid[row,phase]],strict=True):
                        values={field:((int(v)&0xffffffff)if uns else int(v))if yes else None for field,v,yes,uns in zip(DIRECT_FIELDS,record,mask,types,strict=True)}
                        values['malformed_fields']=[field for field,v in zip(DIRECT_FIELDS,bad,strict=True)if v]
                        values['unavailable_fields']=[field for field,v in values.items()if v is None];records.append(values)
                    direct_result[name]=dict(count=int(count[row,phase]),overflow=bool(overflow[row,phase]),records=records)
            else:direct_result['reason']='MISSING_OR_MALFORMED_STORAGE'
            item['direct_hit']=direct_result
        result.append(item)
    return dict(schema='pokezero.literal-refused-outcomes.v1',capacity=CAPACITY,rows=result)
