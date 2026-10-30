"""INTERNAL literal burn/poison/toxic residual writes; no public admission.

Kind20 words1/2 are BEFORE/AFTER engine status_counter, words3/4 current target
side/physical, and words5..9 OLD_HP, NEW_HP, MAX_HP, OLD_STATUS, canonical reason
(BURN1/POISON2/TOXIC3). These counter words are PRIVATE diagnostics, not actors or
public Toxic evidence. Toxic remains a distinct reason even though the original
source damage tag is '[from] psn'. The caller must capture the actual numerical
HP/counter assignment. No damage, counter progression or faint is reconstructed.

Diagnostic attempts always latch UNKNOWN256. A PRIVATE registered owning
context may defer ONLY this new emitter latch on supported nonlethal BRN/PSN
writes after separately proven source hooks/history as well as sticky birth
and cached-speed checks. The actual registered caller currently has no original
hook compiler: its missing hook_context stays CLOSED/unknown, never proofTrue. Prior bits and
invalid/no-output/TOX/lethal/overflow claims remain unknown. Positive lethal writes are explicitly
UNQUALIFIED diagnostics; this helper emits no faint and cannot prove its queue
order. Original source Damage/status/ability hooks, no-output paths, owning
history and public consumer registration remain unproved. Legacy tapes are
unchanged. Only the pure device seam is suitable inside a cached JIT; original
host compiler/metadata/catalog guards must run outside every owning invocation.
"""
from dataclasses import dataclass
import hashlib
import json
from typing import NamedTuple
import weakref

import jax
import jax.numpy as jnp
import numpy as np

from . import engine_public_events as public, engine_lifecycle_callbacks as life
from . import engine_private_team as private_team, mechanics, public_mon_state

STATUS_RESIDUAL = 20
UNKNOWN_STATUS_RESIDUAL = 256
BURN, POISON, TOXIC = 1, 2, 3
SCHEMA = 'pokezero.engine_status_residual_write.partial.v1'
# Exact existing engine mechanics enum: NONE,PAR,BRN,PSN,TOX,SLP,FRZ.
_REASON_BY_STATUS = (0, 0, BURN, POISON, TOXIC, 0, 0)
# Canonical wire tags for reasons0..3: absent, brn, psn, psn.
_TAG_BY_REASON = (0, 1, 2, 2)


class Data(NamedTuple):
    binding: object
    engine_binding: object
    source_binding: object
    move_binding: object
    producer_binding: object
    reason_by_status: object
    damage_tag_by_reason: object


class RegisteredStatusResidualContext(NamedTuple):
    """PRIVATE statics; not an accepted boolean or constructor attestation.

    The owning transition also requires actual BirthRecord and the SAME original
    ProfileData object. Host driver must check both original compiler holders.
    """
    data: object
    weather_profile_data: object
    # Original source hook/history proof is deliberately NOT supplied by the
    # speed profile. No implementation currently registers a non-None value.
    hook_context: object = None


def retain_profile_device(record,state,streams,data,*,eligible):
    """Retain intermediate physical state; inactive rows preserve old carry."""
    from . import engine_weather_speed_profile as speed_profile
    retained=speed_profile.retain_device(record,state,streams,data)
    mask=_eligible(eligible,record.qualified_birth);b=mask.shape[0]
    return jax.tree.map(lambda old,new:jnp.where(mask.reshape((b,)+(1,)*(old.ndim-1)),new,old),record,retained)


@dataclass(frozen=True, eq=False)
class Compiled:
    device: Data


_COMPILED = weakref.WeakKeyDictionary()


def _json(value):
    return json.dumps(value,sort_keys=True,separators=(',', ':'),allow_nan=False)


def _host():
    # Reject tracing even when only cursor/chance or an unused dummy is dynamic.
    if isinstance(jnp.zeros((),jnp.int32),jax.core.Tracer):
        raise ValueError('checked status residual host boundary cannot be traced')


def compile_status_residual(metadata, *, engine_catalog):
    """Original metadata/catalog namespaces plus fixed canonical engine enum."""
    _host()
    public.check_catalog(engine_catalog,engine_catalog.engine_table,
        expected_binding=engine_catalog.binding,
        expected_engine_binding=private_team.engine_metadata_binding(metadata))
    if (mechanics.NONE,mechanics.PAR,mechanics.BRN,mechanics.PSN,mechanics.TOX,mechanics.SLP,mechanics.FRZ)!=tuple(range(7)):
        raise ValueError('original engine status enum required')
    payload=_json(dict(schema=SCHEMA,metadata=metadata,producer=engine_catalog.binding,
        engine=engine_catalog.engine_binding,source=engine_catalog.source_binding,
        moves=engine_catalog.move_binding,status_names=('none','par','brn','psn','tox','slp','frz'),
        reasons=_REASON_BY_STATUS,damage_tags=_TAG_BY_REASON,source_revision=public.SHOWDOWN,
        scope='literal-write-only-unknown-no-faint')).encode()
    digest=hashlib.sha256(payload).digest()
    words=lambda a:jnp.asarray(a,jnp.uint32)
    data=Data(words(np.frombuffer(digest,dtype='>u4').astype(np.uint32)),
        words(engine_catalog.engine_binding),words(engine_catalog.source_binding),
        words(engine_catalog.move_binding),
        words(np.frombuffer(bytes.fromhex(engine_catalog.binding),dtype='>u4').astype(np.uint32)),
        jnp.array(_REASON_BY_STATUS,jnp.int32),jnp.array(_TAG_BY_REASON,jnp.int32))
    result=Compiled(data)
    _COMPILED[result]=(data,tuple(data),metadata,_json(metadata),engine_catalog)
    return result


def check_compiled(compiled):
    """Original immutable compiler/metadata, outside EVERY owning invocation."""
    _host()
    record=_COMPILED.get(compiled) if type(compiled) is Compiled else None
    if (record is None or compiled.device is not record[0]
            or any(a is not b for a,b in zip(compiled.device,record[1]))
            or _json(record[2])!=record[3]
            or any(not isinstance(a,jax.Array) or isinstance(a,jax.core.Tracer) for a in compiled.device)):
        raise ValueError('original immutable status residual compiler/metadata required')
    catalog=record[4]
    public.check_catalog(catalog,catalog.engine_table,expected_binding=catalog.binding,
        expected_engine_binding=compiled.device.engine_binding)
    return compiled.device


def _check(tape,data):
    if type(tape) is public.CallbackTape:return False
    if type(tape) is not life.LifecycleTape or type(tape.cursor) is not life.Cursor:
        raise ValueError('typed actual lifecycle tape/cursor required')
    if getattr(tape.count,'ndim',None)!=1 or not tape.count.shape[0]:
        raise ValueError('nonempty status callback batch required')
    b=tape.count.shape[0]
    if tape.words.ndim!=3 or tape.words.shape[0]!=b or not 1<=tape.words.shape[1]<=64 or tape.words.shape[2]!=10:
        raise ValueError('bounded actual status callback storage required')
    c=tape.words.shape[1]
    for value,shape,dtype in ((tape.words,(b,c,10),jnp.int32),(tape.count,(b,),jnp.int32),
            (tape.refusal,(b,),jnp.int32),(tape.turn,(b,),jnp.int32),(tape.stream_id,(b,),jnp.int32),
            (tape.event_turn,(b,c),jnp.int32),(tape.event_phase,(b,c),jnp.int32),
            (tape.event_generation,(b,c,2),jnp.int32),(tape.switch_details,(b,c,6),jnp.int32)):
        life._typed(value,shape,dtype)
    if type(data) is not Data:raise ValueError('original checked status residual mapping required')
    for value in (tape.engine_binding,tape.source_binding,tape.move_binding,tape.producer_binding,
            data.binding,data.engine_binding,data.source_binding,data.move_binding,data.producer_binding):
        life._typed(value,(8,),jnp.uint32)
    life._typed(data.reason_by_status,(7,),jnp.int32)
    life._typed(data.damage_tag_by_reason,(4,),jnp.int32)
    life._valid(tape.cursor)
    return True


def _eligible(value,count):
    if type(value) is bool:return jnp.full(count.shape,value)
    if getattr(value,'dtype',None)!=jnp.dtype(bool) or getattr(value,'shape',None) not in ((),count.shape):
        raise ValueError('actual residual eligibility must be boolean scalar/B')
    return jnp.broadcast_to(jnp.asarray(value),count.shape)


def _unknown(tape,eligible):
    return tape._replace(refusal=tape.refusal|jnp.where(eligible,UNKNOWN_STATUS_RESIDUAL,0))


def record_status_residual_device(tape,side,physical,old_hp,new_hp,maximum,old_status,
                                  old_counter,new_counter,data,*,eligible,_defer_own_unknown=False):
    """Trusted actual-write seam; all payload/counters are literal int32 values.

    No-op/dead/unsupported/invalid eligible claims remain unknown-only. Valid
    strict HP loss may reach0 as a diagnostic, without generating a FAINT.
    Counter bounds/unchanged non-Toxic counters are validity checks only; this
    primitive NEVER computes damage, stage+1, or a status from HP/counters.
    """
    if not _check(tape,data):return tape
    eligible=_eligible(eligible,tape.count)
    defer=_eligible(_defer_own_unknown,tape.count)
    try:
        values=tuple(public_mon_state._int32(v) for v in
            (side,physical,old_hp,new_hp,maximum,old_status,old_counter,new_counter))
    except (TypeError,ValueError):return _unknown(tape,eligible)
    if any(v.shape not in ((),tape.count.shape) for v in values):return _unknown(tape,eligible)
    side,physical,old_hp,new_hp,maximum,status,old_counter,new_counter=(jnp.broadcast_to(v,tape.count.shape) for v in values)
    rows=jnp.arange(tape.count.shape[0]);safe=jnp.clip(side,0,1)
    index=jnp.clip(tape.count,0,tape.words.shape[1]-1);room=tape.count<tape.words.shape[1]
    reason=data.reason_by_status[jnp.clip(status,0,6)]
    canonical=jnp.all(data.reason_by_status==jnp.array(_REASON_BY_STATUS,jnp.int32))
    canonical &= jnp.all(data.damage_tag_by_reason==jnp.array(_TAG_BY_REASON,jnp.int32))
    counters=(old_counter>=0)&(old_counter<=15)&(new_counter>=0)&(new_counter<=15)
    counters &= jnp.where(status==mechanics.TOX,new_counter>=old_counter,new_counter==old_counter)
    valid=(life._valid(tape.cursor)&tape.cursor.initialized
        &(tape.stream_id==tape.cursor.stream_id)
        &jnp.all(tape.engine_binding==data.engine_binding)&jnp.all(tape.source_binding==data.source_binding)
        &jnp.all(tape.move_binding==data.move_binding)&jnp.all(tape.producer_binding==data.producer_binding)
        &jnp.all(tape.cursor.engine_binding==data.engine_binding,axis=1)
        &(tape.turn==tape.cursor.turn)&(tape.cursor.phase==0)
        &(side>=0)&(side<=1)&(physical>=0)&(physical<6)
        &(physical==tape.cursor.physical_active[rows,safe])
        &(tape.count>=0)&(tape.count<=tape.words.shape[1])
        &(old_hp>0)&(new_hp>=0)&(new_hp<old_hp)&(maximum>=old_hp)&(maximum>0)
        &(status>=0)&(status<7)&(reason>0)&canonical&counters
        &(~room|jnp.all(tape.switch_details[rows,index]==0,axis=1)))
    # Internal deferral is not public acceptance: only this emitter's NEW bit
    # is omitted. Existing refusals are never cleared or reclassified.
    can_defer=(defer&valid&room&(new_hp>0)&((status==mechanics.BRN)|(status==mechanics.PSN))
        &((tape.refusal&~public.UNINSTRUMENTED)==0))
    tape=public._append(tape,STATUS_RESIDUAL,old_counter,new_counter,side,physical,
        (old_hp,new_hp,maximum,status,reason),eligible&valid)
    return _unknown(tape,eligible&~can_defer)


def record_status_residual(tape,side,physical,old_hp,new_hp,maximum,old_status,
                          old_counter,new_counter,*,compiled,eligible):
    """Checked host call, never an outer-JIT/cacheable source boundary."""
    _host()
    return record_status_residual_device(tape,side,physical,old_hp,new_hp,maximum,old_status,
        old_counter,new_counter,check_compiled(compiled),eligible=eligible)
