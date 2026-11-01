"""INTERNAL literal burn/poison/toxic residual writes; no public admission.

Kind20 words1/2 are BEFORE/AFTER engine status_counter, words3/4 current target
side/physical, and words5..9 OLD_HP, NEW_HP, MAX_HP, OLD_STATUS, canonical reason
(BURN1/POISON2/TOXIC3). These counter words are PRIVATE diagnostics, not actors or
public Toxic evidence. Toxic remains a distinct reason even though the original
source damage tag is '[from] psn'. The caller must capture the actual numerical
HP/counter assignment. No damage, counter progression or faint is reconstructed.

Diagnostic attempts always latch UNKNOWN256. A PRIVATE registered owning
kernel-event observer may defer ONLY this new emitter latch on literal,
nonlethal BRN/PSN/TOX writes after sticky birth and actual cached-speed checks.
This observes this kernel's executed write; it does not prove original
Showdown hooks, physical equivalence, exhaustive source events or no-output
paths. Original-hook/history diagnostics remain separate. Prior bits and
invalid/no-output/lethal/overflow claims remain unknown. Positive lethal
writes are explicitly UNQUALIFIED diagnostics; this helper emits no faint.
Legacy tapes are unchanged. Only the pure device seam belongs inside a cached
JIT; original host compiler/metadata/catalog guards must run outside every
owning invocation.
"""
from dataclasses import dataclass
import hashlib
import json
import subprocess
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
    singleton_order: object


class RegisteredStatusResidualContext(NamedTuple):
    """PRIVATE kernel-event statics, not original-physics qualification.

    The owning transition requires actual BirthRecord and the SAME original
    ProfileData object. Host driver must check both original compiler holders.
    Registration enables only selected literal kernel writes and transport;
    original hook/history diagnostics do not gate or establish that observation.
    """
    data: object
    weather_profile_data: object
    # Reserved original-hook diagnostic context, not a kernel-event opener.
    # This observer registers no non-None value and rejects caller proof values.
    hook_context: object = None


def retain_profile_device(record,state,streams,data,*,eligible,hypothetical=False):
    """Retain intermediate physical state; inactive rows preserve old carry."""
    from . import engine_weather_speed_profile as speed_profile
    if hypothetical:
        from . import v5_hypothetical_profile as hypothesis
        retained=hypothesis.retain_device(record,state,streams,data)
        mask=_eligible(eligible,record.qualified_origin)
    else:
        retained=speed_profile.retain_device(record,state,streams,data)
        mask=_eligible(eligible,record.qualified_birth)
    b=mask.shape[0]
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


def compile_status_residual(metadata, *, engine_catalog, source_rules=None):
    """Original metadata/catalog namespaces plus fixed canonical engine enum."""
    _host()
    public.check_catalog(engine_catalog,engine_catalog.engine_table,
        expected_binding=engine_catalog.binding,
        expected_engine_binding=private_team.engine_metadata_binding(metadata))
    if (mechanics.NONE,mechanics.PAR,mechanics.BRN,mechanics.PSN,mechanics.TOX,mechanics.SLP,mechanics.FRZ)!=tuple(range(7)):
        raise ValueError('original engine status enum required')
    source_descriptor=None
    if source_rules is not None:
        from . import engine_request_source as source
        origin=source._rules_origin(source_rules)
        if not np.array_equal(source_rules.device.engine_binding,engine_catalog.engine_binding):
            raise ValueError('same original status residual source rules required')
        script=r"""const path=require('path'),c=require('crypto'),r=process.argv[1],{Battle}=require(path.join(r,'dist/sim/battle.js')),{Pokemon}=require(path.join(r,'dist/sim/pokemon.js')),d=require(path.join(r,'dist/sim/dex.js')).Dex.mod('gen3'),h=f=>c.createHash('sha256').update(f.toString()).digest('hex'),g=xs=>xs.filter(x=>typeof x.onResidual==='function'&&x.onResidualOrder===10&&!(x.onResidualPriority||0)&&x.onResidualSubOrder===6).map(x=>x.id).sort();console.log(JSON.stringify({functions:Object.fromEntries(Object.entries({comparePriority:Battle.prototype.comparePriority,speedSort:Battle.prototype.speedSort,fieldEvent:Battle.prototype.fieldEvent,resolvePriority:Battle.prototype.resolvePriority,runAction:Battle.prototype.runAction,updateSpeed:Battle.prototype.updateSpeed,pokemonUpdateSpeed:Pokemon.prototype.updateSpeed}).map(([n,f])=>[n,h(f)])),conditions:Object.fromEntries(['brn','psn','tox'].map(n=>{const x=d.conditions.get(n);return [n,[x.onResidualOrder,x.onResidualPriority||0,x.onResidualSubOrder,h(x.onResidual)]]})),tie_group:{abilities:g(d.abilities.all().filter(x=>x.gen<=3)),items:g(d.items.all().filter(x=>x.gen<=3)),conditions:g([...new Set([...Object.keys(d.data.Conditions),...Object.keys(d.data.Moves)])].map(n=>d.conditions.get(n)))}}));"""
        source_descriptor=json.loads(subprocess.run([origin.node,'-e',script,origin.root],capture_output=True,text=True,check=True,timeout=30).stdout)
        if source_descriptor!=SINGLETON_SOURCE:raise ValueError('original residual queue/source changed')
    payload=_json(dict(schema=SCHEMA,singleton_source=source_descriptor,metadata=metadata,producer=engine_catalog.binding,
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
        jnp.array(_REASON_BY_STATUS,jnp.int32),jnp.array(_TAG_BY_REASON,jnp.int32),jnp.array(source_descriptor is not None,jnp.bool_))
    result=Compiled(data)
    _COMPILED[result]=(data,tuple(data),metadata,_json(metadata),engine_catalog,source_rules)
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
    if record[5] is not None:
        from . import engine_request_source as source
        source._rules_origin(record[5])
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
    life._typed(data.singleton_order,(),jnp.bool_)
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
    selected=((status==mechanics.BRN)|(status==mechanics.PSN)
        |((status==mechanics.TOX)&(new_counter==jnp.minimum(old_counter+1,15))))
    can_defer=(defer&valid&room&(new_hp>0)&selected
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


# Positive source capability only for the absence of a status-handler tie.
# The original physical scheduler and all of its chance coordinates are unchanged.
SINGLETON_SOURCE=dict(functions=dict(comparePriority='6037e6f1ce0144cd4075a939b5e14f1a9829fc760056a4eda58e0afda57f4015',speedSort='bb743bd89b65e5fa2270d62c2011946312a66bfbabab1f0b1ba923e6943c26bb',fieldEvent='5ada727300458c9350861313d56328944a13360b80efba54eca9bcdee979a879',resolvePriority='0c1208b08dc0bcf44d85ce8508b199508a11b52d996935fe472fc7f5d2049f4f',runAction='6c36c083886cd9f924699a934ba43e3aee101c4c435fdb00a811b476bc9f23df',updateSpeed='3dc9a9eb6c457cd99bb2ffd92ac112eb44fc43772e75aba5ba9f4046a3a9f3b4',pokemonUpdateSpeed='4c8fa2dab90a627d300b7d9a5d377e70b097cf942912c1b1f0ce9c2533372074'),tie_group=dict(abilities=[],items=[],conditions=['brn','psn','tox']),conditions=dict(brn=[10,0,6,'e2d98cce06adcb06823fc4b11301d78d413186ccd89e7477705672670ff283a8'],psn=[10,0,6,'e2d98cce06adcb06823fc4b11301d78d413186ccd89e7477705672670ff283a8'],tox=[10,0,6,'5bbd74e4fc26c19c8d842a4188dc3ca503b36ccf8da69cf9fa66f84e3e461919']))

class QueueSnapshot(NamedTuple):
    binding: object
    stream_id: object
    frame: object
    turn: object
    phase: object
    generation: object
    physical: object
    hp: object
    status: object
    speeds: object
    eligible: object


def capture_queue_device(tape,data,hp,status,speeds,*,eligible):
    """SAME pre-Wish/pre-weather queue snapshot, never a later-state estimate."""
    from . import engine_execution_completeness as execution
    if not _check(tape,data):raise ValueError('owning residual queue tape required')
    b=len(tape.count)
    for x in (hp,status,speeds):life._typed(x,(b,2),jnp.int32)
    life._typed(eligible,(b,),jnp.bool_)
    context=execution._context(tape)
    frame=jnp.full(b,-1,jnp.int32)if context is None else context.ledger.frame
    return QueueSnapshot(data.binding,tape.stream_id,frame,tape.cursor.turn,tape.cursor.phase,
        tape.cursor.generation,tape.cursor.physical_active,hp,status,speeds,eligible)


def singleton_queue_matches(tape,data,queue,side,physical,current_hp,current_status):
    """One queued order10/sub6 status handler cannot have a status-speed tie.

    This joins actual immutable call-local operands and current source lifetime;
    it does not consume a tie coin or explain two tied status handlers.
    """
    from . import engine_execution_completeness as execution
    if not _check(tape,data):raise ValueError('owning residual queue tape required')
    if type(queue)is not QueueSnapshot:raise ValueError('same-call typed residual queue required')
    b=len(tape.count);rows=jnp.arange(b)
    life._typed(queue.binding,(8,),jnp.uint32)
    for x in (queue.stream_id,queue.frame,queue.turn,queue.phase,side,physical,current_hp,current_status):life._typed(x,(b,),jnp.int32)
    for x in (queue.generation,queue.physical,queue.hp,queue.status,queue.speeds):life._typed(x,(b,2),jnp.int32)
    life._typed(queue.eligible,(b,),jnp.bool_)
    context=execution._context(tape)
    if context is None:return jnp.zeros(b,bool)
    ledger=context.ledger;safe=jnp.clip(side,0,1)
    handlers=(queue.hp>0)&((queue.status==mechanics.BRN)|(queue.status==mechanics.PSN)|(queue.status==mechanics.TOX))
    good=execution._validate(ledger,context.tables)&execution._tape(ledger,tape,context.tables)&life._valid(tape.cursor)&tape.cursor.initialized
    good &= (data.singleton_order&jnp.all(queue.binding==data.binding)&queue.eligible&ledger.opened
        &(ledger.frame>0)&(ledger.frame==queue.frame)&(ledger.refusal==0)&((tape.refusal&~public.UNINSTRUMENTED)==0)
        &(queue.stream_id==tape.stream_id)&(queue.turn==tape.cursor.turn)&(queue.phase==0)&(queue.phase==tape.cursor.phase)
        &jnp.all(queue.generation==tape.cursor.generation,axis=1)&jnp.all(queue.physical==tape.cursor.physical_active,axis=1)
        &jnp.all((queue.hp>0)&(queue.status>=0)&(queue.status<=6),axis=1)
        &jnp.all((queue.speeds>0)&(queue.speeds<=10000),axis=1)&(queue.speeds[:,0]==queue.speeds[:,1])
        &(jnp.sum(handlers,axis=1)==1)&(side>=0)&(side<2)&handlers[rows,safe]
        &(physical==queue.physical[rows,safe])&(current_status==queue.status[rows,safe])&(current_hp>0))
    for name in ('engine_binding','source_binding','move_binding','producer_binding'):good &= jnp.all(getattr(tape,name)==getattr(data,name))
    return good
