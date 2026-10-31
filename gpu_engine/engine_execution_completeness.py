"""PRIVATE, bounded accounting for residual/entry source obligations.

An owning callsite must enter BEFORE an observable attempt/write, including a
blocked/no-write attempt. It either records its exact appended LifecycleTape
span or marks the obligation unhandled immediately. Only consume_frame(), which
actually runs the original full projector and all retained consumers, credits
an emitted span. No caller accepted/proof boolean, state difference, zero mask,
or final-state equality can discharge an obligation.

This inventory is a PARTIAL source-bound list, NOT an exhaustive source/Dex-hook
inventory or an attestation of constructor/session origin. Registration of every
actual callsite, exact semantic site-to-emitter mapping, source no-output proofs,
birth/history and original hook support remain UNPROVED integration obligations.
scoped_drained means only the entered obligations drained. This module exposes
NO full-source, encoder, B4, fidelity or scientific qualification API. In
particular, not calling enter() cannot certify that no observable branch ran.
"""
from dataclasses import dataclass
from enum import IntEnum
import hashlib
import json
from pathlib import Path
from typing import NamedTuple
import weakref

import jax
import jax.numpy as jnp
import numpy as np

from . import engine_lifecycle_callbacks as life
from . import engine_full_v4_event_device as full

SCHEMA = 'pokezero.private.execution_obligations.partial.v1'


class Site(IntEnum):
    WISH = 1
    SPEED_BOOST = 2
    LEFTOVERS = 3
    PINCH_ITEM = 4
    STATUS_RESIDUAL = 5
    LEECH_DRAIN = 6
    LEECH_HEAL = 7
    ENTRY_HAZARD = 8
    INTIMIDATE_ATTEMPT = 9
    DIRECT_HIT = 10
    SUBSTITUTE_SELF = 11
    TOXIC_ATTEMPT = 12
    TRACE_COPY = 13
    RECOIL_HP = 14
    INTIMIDATE_INITIAL = 15


UNKNOWN_SITE, ORDER, OVERFLOW, NAMESPACE, RECEIPT, CONSUMPTION, UNRESOLVED = (1, 2, 4, 8, 16, 32, 64)
INTEGRATION_REQUIRED = ('actual-constructor/session-origin', 'reviewed-exhaustive-site/dispatch-inventory',
    'exact-site/emitter-semantics', 'source-no-public-output-proofs', 'sticky-birth/history',
    'original-source-hooks-outside-engine-mutations', 'owning-SOURCE/PUBLIC_COVERAGE-wiring')


class Tables(NamedTuple):
    engine_binding: object             # GLOBAL (8,), never indexed as B rows
    source_binding: object             # GLOBAL (8,)
    producer_binding: object           # GLOBAL (8,)
    inventory_binding: object          # GLOBAL (8,)


@dataclass(frozen=True, eq=False)
class Compiled:
    device: Tables


_COMPILED = weakref.WeakKeyDictionary()


def compile_accounting(adapter):
    """Bind a partial accounting inventory to the ORIGINAL full adapter.

    No caller can declare source support or inventory exhaustiveness here.
    Actual callsite integration must be separately reviewed against these exact
    source hashes; this compiler does not claim that integration has occurred.
    """
    if isinstance(jnp.zeros((),jnp.int32),jax.core.Tracer):
        raise ValueError('checked accounting compiler must remain outside outer JIT')
    full.check_adapter(adapter)
    lower = adapter.causal.projector.catalog
    payload = dict(schema=SCHEMA, sites={s.name: int(s) for s in Site},
        obligations=INTEGRATION_REQUIRED, files={name: hashlib.sha256(
            Path(__file__).with_name(name).read_bytes()).hexdigest() for name in
            ('transition.py', 'entry.py', 'engine_execution_completeness.py')},
        adapter=np.asarray(adapter.device.binding).tolist())
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(',', ':')).encode()).digest()
    tables = Tables(jnp.asarray(lower.engine_binding, jnp.uint32),
        jnp.asarray(lower.source_binding, jnp.uint32),
        jnp.asarray(np.frombuffer(bytes.fromhex(lower.binding), dtype='>u4').astype(np.uint32)),
        jnp.asarray(np.frombuffer(digest, dtype='>u4').astype(np.uint32)))
    compiled = Compiled(tables)
    _COMPILED[compiled] = (tables, tuple(tables), adapter)
    return compiled


def _checked(compiled):
    if isinstance(jnp.zeros((),jnp.int32),jax.core.Tracer):
        raise ValueError('checked accounting compiler must remain outside outer JIT')
    record = _COMPILED.get(compiled) if type(compiled) is Compiled else None
    if record is None or compiled.device is not record[0] or any(a is not b for a, b in zip(compiled.device, record[1])):
        raise ValueError('original immutable accounting compiler required')
    full.check_adapter(record[2])
    return record


def _shape(a, shape, dtype):
    if not isinstance(a, (jax.Array, jax.core.Tracer)) or a.shape != shape or a.dtype != jnp.dtype(dtype):
        raise ValueError('exact immutable execution-accounting shape/dtype required')


class Ledger(NamedTuple):
    stream_id: object
    inventory_binding: object
    frame: object
    opened: object
    next_serial: object
    refusal: object
    upstream_refusal: object
    count: object
    site: object
    side: object
    serial: object
    stage: object                     # 1 entered, 2 exact emission, 3 consumed
    start: object
    end: object
    turn: object
    phase: object
    generation: object
    claimed: object                   # tape index -> obligation slot+1
    words: object
    event_turn: object
    event_phase: object
    event_generation: object
    no_output_kind: object            # 0 positive span; 1 typed full-HP Leftovers
    no_output_facts: object           # physical,item,hp,max,status,residual_event
    no_output_binding: object         # original source-specific compiler binding


class Ticket(NamedTuple):
    stream_id: object
    inventory_binding: object
    frame: object
    slot: object
    serial: object


class Report(NamedTuple):
    scoped_drained: object
    refusal: object
    upstream_refusal: object
    entered: object
    unresolved: object


class Context(NamedTuple):
    """PRIVATE actual lifecycle carry; never an exhaustive-source certificate."""
    ledger: Ledger
    tables: Tables
    risks: object = None  # PRIVATE actual-kernel diagnostics; never a coverage certificate


UNKNOWN_EXECUTION = 512
UNKNOWN_KERNEL_RISK = 1024


def commit_rows(active, before, after):
    """Ledger has one GLOBAL eight-word binding, including when B happens to8."""
    if type(before) is not Ledger or type(after) is not Ledger:
        raise ValueError('typed private ledgers required')
    _shape(active,before.stream_id.shape,jnp.bool_)
    fields={name:jnp.where(active.reshape((active.shape[0],)+(1,)*(getattr(before,name).ndim-1)),
        getattr(after,name),getattr(before,name)) for name in before._fields if name!='inventory_binding'}
    return before._replace(**fields)


def begin_on_tape(tape, ledger, tables, *, active):
    """Begin BEFORE the actual constructor/step source handlers execute."""
    if not hasattr(tape,'execution'):
        raise ValueError('owning lifecycle accounting field required')
    ledger=begin_frame(ledger,tape,frame=ledger.frame+1,active=active,tables=tables)
    return tape._replace(execution=Context(ledger,tables))


def _context(tape):
    context=getattr(tape,'execution',None)
    if context is not None and type(context) is not Context:
        raise ValueError('original private lifecycle accounting context required')
    return context


def _put_context(tape, ledger, context):
    # Immediate sticky source refusal; no successful later emitter/consumer
    # clears a previously unsupported or malformed attempt.
    return tape._replace(execution=context._replace(ledger=ledger),
        refusal=tape.refusal | jnp.where(ledger.refusal!=0,UNKNOWN_EXECUTION,0))


def open_attempt(tape, site, side, *, active):
    """Actual callsite BEFORE dispatch/write. False loop masks enter nothing."""
    context=_context(tape)
    if context is None:return tape,None
    b=tape.count.shape
    _shape(active,b,jnp.bool_)
    side=jnp.broadcast_to(jnp.asarray(side,jnp.int32),b)
    site=jnp.full(b,int(site),jnp.int32)
    ledger,ticket=enter(context.ledger,tape,site,side,active=active,tables=context.tables)
    return _put_context(tape,ledger,context),ticket


def unhandled_attempt(tape, site, side, *, active):
    """No no-op exemption: record unsupported blocked/capped attempts too."""
    tape,ticket=open_attempt(tape,site,side,active=active)
    context=_context(tape)
    if context is None:return tape
    ledger=unhandled(context.ledger,ticket,active=active,tables=context.tables)
    return _put_context(tape,ledger,context)


def capture_leftovers_write(tape, ticket, *, requested, positive_write):
    """Reviewed LEFTOVERS site maps ONLY to its single literal kind17 receipt.

    The owning caller supplies its actual numerical write guard. This checks
    exact source-site/emitter identity, then snapshots the original stamped
    span. Full-HP/blocked/no-output attempts remain unsupported; the consumer
    must later drain this span before credit is granted.
    """
    context=_context(tape)
    if context is None:return tape
    ledger=context.ledger;slot,same=_ticket(ledger,ticket);rows=jnp.arange(tape.count.shape[0])
    _shape(requested,tape.count.shape,jnp.bool_);_shape(positive_write,tape.count.shape,jnp.bool_)
    start=ledger.start[rows,slot];index=jnp.clip(start,0,tape.words.shape[1]-1)
    raw=tape.words[rows,index]
    semantic=(same & (ledger.site[rows,slot]==int(Site.LEFTOVERS)) & (tape.count==start+1)
        & (raw[:,0]==17)&(raw[:,3]==ledger.side[rows,slot]))
    accepted=requested & positive_write & semantic
    ledger=unhandled(ledger,ticket,active=requested&~(positive_write&semantic),tables=context.tables)
    ledger=emitted(ledger,ticket,tape,active=accepted,tables=context.tables)
    return _put_context(tape,ledger,context)


def capture_trace_copy(tape,ticket,*,requested):
    """Snapshot only this actual entry copy's one raw27; no consume credit."""
    context=_context(tape)
    if context is None:return tape
    from . import engine_trace_callbacks as trace
    ledger=context.ledger;slot,same=_ticket(ledger,ticket);rows=jnp.arange(tape.count.shape[0])
    _shape(requested,tape.count.shape,jnp.bool_)
    start=ledger.start[rows,slot];raw=tape.words[rows,jnp.clip(start,0,tape.words.shape[1]-1)]
    side=raw[:,3];safe=jnp.clip(side,0,1)
    semantic=(same&(ledger.site[rows,slot]==int(Site.TRACE_COPY))&(tape.count==start+1)
        &(raw[:,0]==trace.TRACE_COPY)&(side==ledger.side[rows,slot])&(side>=0)&(side<2)
        &(raw[:,1]>0)&(raw[:,2]>0)&(raw[:,2]==raw[:,7])&(raw[:,5]==1-side)
        &(raw[:,4]==tape.cursor.physical_active[rows,safe])
        &(raw[:,6]==tape.cursor.physical_active[rows,1-safe])&jnp.all(raw[:,8:]==0,axis=1))
    ledger=unhandled(ledger,ticket,active=requested&~semantic,tables=context.tables)
    ledger=emitted(ledger,ticket,tape,active=requested&semantic,tables=context.tables)
    return _put_context(tape,ledger,context)


def capture_status_residual_write(tape, ticket, *, requested, positive_write):
    """Map this actual STATUS attempt only to one literal nonlethal BRN/PSN/TOX kind20.

    This is exact span capture, not consumption credit. No-output, lethal,
    malformed and missing spans stay unhandled. Genuine full consumption by
    both owning perspectives is still required; prior refusals never clear.
    """
    context=_context(tape)
    if context is None:return tape
    ledger=context.ledger;slot,same=_ticket(ledger,ticket);rows=jnp.arange(tape.count.shape[0])
    _shape(requested,tape.count.shape,jnp.bool_);_shape(positive_write,tape.count.shape,jnp.bool_)
    start=ledger.start[rows,slot];index=jnp.clip(start,0,tape.words.shape[1]-1)
    raw=tape.words[rows,index]
    brn_psn=((raw[:,8]==2)&(raw[:,9]==1))|((raw[:,8]==3)&(raw[:,9]==2))
    toxic=(raw[:,8]==4)&(raw[:,9]==3)&(raw[:,2]==jnp.minimum(raw[:,1]+1,15))
    semantic=(same & (ledger.site[rows,slot]==int(Site.STATUS_RESIDUAL)) & (tape.count==start+1)
        & (raw[:,0]==20)&(raw[:,3]==ledger.side[rows,slot])
        & (brn_psn|toxic)&jnp.where(brn_psn,raw[:,2]==raw[:,1],True)
        & (raw[:,1]>=0)&(raw[:,1]<=15)&(raw[:,2]>=0)&(raw[:,2]<=15)
        & (raw[:,4]>=0)&(raw[:,4]<6)
        & (raw[:,4]==tape.cursor.physical_active[rows,jnp.clip(raw[:,3],0,1)])
        & (raw[:,5]>raw[:,6])&(raw[:,6]>0)&(raw[:,7]>=raw[:,5]))
    accepted=requested & positive_write & semantic
    ledger=unhandled(ledger,ticket,active=requested&~(positive_write&semantic),tables=context.tables)
    ledger=emitted(ledger,ticket,tape,active=accepted,tables=context.tables)
    return _put_context(tape,ledger,context)


def _direct_hit_span_matches(tape,ticket,actor,actor_physical,target_physical,move,write,*,data=None):
    """Read-only literal Site10 span check shared by capture and risk observer.

    This computes facts from the SAME source ticket/write/tape, not a caller
    accepted flag. Stage/credit belongs to its owning capture/consumer caller.
    """
    from . import engine_direct_hit_controls as direct
    context=_context(tape)
    if context is None:return jnp.zeros_like(tape.count,dtype=jnp.bool_)
    ledger=context.ledger;slot,same=_ticket(ledger,ticket);b=tape.count.shape[0];rows=jnp.arange(b)
    if type(write) is not direct.HitWrite:raise ValueError('literal direct-hit write required')
    for value in (actor,actor_physical,target_physical,move):_shape(value,(b,),jnp.int32)
    valid=direct.validate_write_device(write,b,data=data,move=move)
    start=ledger.start[rows,slot]
    declaration=tape.words[rows,jnp.clip(start-1,0,tape.words.shape[1]-1)]
    safe_actor=jnp.clip(actor,0,1)
    declared=((start>0)&(declaration[:,0]==direct.public.MOVE)
        &(declaration[:,1]==actor)&(declaration[:,2]==actor_physical)
        &(declaration[:,3]==1-actor)&(declaration[:,4]==target_physical)
        &(declaration[:,5]==move)&jnp.all(declaration[:,6:]==0,axis=1))
    declared &= (life._valid(tape.cursor)&(actor>=0)&(actor<2)
        &(actor_physical==tape.cursor.physical_active[rows,safe_actor])
        &(target_physical==tape.cursor.physical_active[rows,1-safe_actor])
        &(tape.cursor.turn==ledger.turn[rows,slot])&(tape.cursor.phase==ledger.phase[rows,slot])
        &jnp.all(tape.cursor.generation==ledger.generation[rows,slot],axis=1))
    prefix=jnp.clip(start-1,0,tape.words.shape[1]-1)
    declared &= ((tape.event_turn[rows,prefix]==ledger.turn[rows,slot])
        &(tape.event_phase[rows,prefix]==ledger.phase[rows,slot])
        &jnp.all(tape.event_generation[rows,prefix]==ledger.generation[rows,slot],axis=1))
    critical,effect=direct.annotation_flags_device(write,data=data,move=move)
    critical=jnp.broadcast_to(critical,(b,));effect=jnp.broadcast_to(effect,(b,))
    num=jnp.broadcast_to(write.numerator,(b,));den=jnp.broadcast_to(write.denominator,(b,))
    length=critical.astype(jnp.int32)+effect.astype(jnp.int32)+1
    first=tape.words[rows,jnp.clip(start,0,tape.words.shape[1]-1)]
    second=tape.words[rows,jnp.clip(start+critical.astype(jnp.int32),0,tape.words.shape[1]-1)]
    hp=tape.words[rows,jnp.clip(start+length-1,0,tape.words.shape[1]-1)]
    expected=jnp.stack((actor,actor_physical,1-actor,target_physical,move,
        jnp.broadcast_to(write.old_hp,(b,)),jnp.broadcast_to(write.new_hp,(b,)),num,den),axis=1)
    controls=(~critical|((first[:,0]==direct.CRITICAL)&jnp.all(first[:,1:]==expected,axis=1)))
    controls &= ~effect|((second[:,0]==jnp.where(num>den,direct.SUPER_EFFECTIVE,direct.RESISTED))
        & jnp.all(second[:,1:]==expected,axis=1))
    exact_hp=((hp[:,0]==direct.public.MAIN_HP)&jnp.all(hp[:,1:3]==-1,axis=1)&(hp[:,3]==1-actor)
        &(hp[:,4]==target_physical)&(hp[:,5]==write.new_hp)&(hp[:,6]==write.maximum)
        &(hp[:,7]>=0)&(hp[:,7]<=6)&jnp.all(hp[:,8:]==0,axis=1))
    semantic=(_validate(ledger,context.tables) & _tape(ledger,tape,context.tables) & same & valid & declared & (ledger.site[rows,slot]==int(Site.DIRECT_HIT))
        &(ledger.side[rows,slot]==1-actor)&(tape.count==start+length)&controls&exact_hp)
    return semantic


def capture_direct_hit_write(tape,ticket,actor,actor_physical,target_physical,move,write,*,requested,data=None):
    """Capture the complete literal positive span; BOTH owners must drain it."""
    context=_context(tape)
    if context is None:return tape
    _shape(requested,tape.count.shape,jnp.bool_)
    semantic=_direct_hit_span_matches(tape,ticket,actor,actor_physical,target_physical,move,write,data=data)
    ledger=context.ledger
    ledger=unhandled(ledger,ticket,active=requested&~semantic,tables=context.tables)
    ledger=emitted(ledger,ticket,tape,active=requested&semantic,tables=context.tables)
    return _put_context(tape,ledger,context)


def _substitute_span_matches(tape,ticket,side,physical,move,write):
    """Actual self-write after its own declaration, distinct from Site10."""
    from . import engine_substitute_callbacks as substitute
    context=_context(tape)
    if context is None:return jnp.zeros_like(tape.count,bool)
    ledger=context.ledger;slot,same=_ticket(ledger,ticket);b=len(tape.count);rows=jnp.arange(b)
    for x in (side,physical,move):_shape(x,(b,),jnp.int32)
    valid=substitute.validate_write_device(write,b)
    start=ledger.start[rows,slot];at=jnp.clip(start,0,tape.words.shape[1]-1)
    prefix=jnp.clip(start-1,0,tape.words.shape[1]-1)
    raw=tape.words[rows,at];decl=tape.words[rows,prefix];safe=jnp.clip(side,0,1)
    expected=jnp.stack((jnp.full_like(side,substitute.SUBSTITUTE_CREATED),write.old_sub_hp,
        write.new_sub_hp,side,physical,write.old_hp,write.new_hp,write.maximum,write.status,move),axis=1)
    same &= (_validate(ledger,context.tables)&_tape(ledger,tape,context.tables)&valid
        &(ledger.site[rows,slot]==int(Site.SUBSTITUTE_SELF))&(ledger.side[rows,slot]==side)
        &(tape.count==start+1)&jnp.all(raw==expected,axis=1)&(start>0)
        &(decl[:,0]==substitute.public.MOVE)&(decl[:,1]==side)&(decl[:,2]==physical)
        &(decl[:,3]==1-side)&(decl[:,4]==tape.cursor.physical_active[rows,1-safe])&(decl[:,5]==move)&jnp.all(decl[:,6:]==0,axis=1)
        &life._valid(tape.cursor)&(physical==tape.cursor.physical_active[rows,safe])
        &(tape.cursor.turn==ledger.turn[rows,slot])&(tape.cursor.phase==ledger.phase[rows,slot])
        &jnp.all(tape.cursor.generation==ledger.generation[rows,slot],axis=1))
    for index in (prefix,at):
        same &= ((tape.event_turn[rows,index]==ledger.turn[rows,slot])
            &(tape.event_phase[rows,index]==ledger.phase[rows,slot])
            &jnp.all(tape.event_generation[rows,index]==ledger.generation[rows,slot],axis=1))
    return same


def capture_substitute_write(tape,ticket,side,physical,move,write,*,requested):
    context=_context(tape)
    if context is None:return tape
    _shape(requested,tape.count.shape,jnp.bool_)
    same=_substitute_span_matches(tape,ticket,side,physical,move,write)
    ledger=unhandled(context.ledger,ticket,active=requested&~same,tables=context.tables)
    ledger=emitted(ledger,ticket,tape,active=requested&same,tables=context.tables)
    return _put_context(tape,ledger,context)


def _toxic_span_matches(tape,ticket,side,physical,target,move,attempt):
    from . import engine_toxic_callbacks as toxic
    context=_context(tape)
    if context is None:return jnp.zeros_like(tape.count,bool)
    ledger=context.ledger;slot,same=_ticket(ledger,ticket);b=len(tape.count);rows=jnp.arange(b)
    expected=toxic.words_device(side,physical,target,move,attempt)
    applied=toxic.outcome_device(expected)==toxic.STATUS_WRITE
    start=ledger.start[rows,slot];first=jnp.clip(start,0,tape.words.shape[1]-1)
    at=jnp.clip(start+applied.astype(jnp.int32),0,tape.words.shape[1]-1)
    prefix=jnp.clip(start-1,0,tape.words.shape[1]-1)
    raw=tape.words[rows,at];decl=tape.words[rows,prefix];safe=jnp.clip(side,0,1)
    from . import engine_status_callbacks as status
    absent=jnp.full_like(side,-1);zero=jnp.zeros_like(side)
    status_words=jnp.stack((jnp.full_like(side,status.STATUS_APPLIED),absent,absent,1-side,target,
        attempt.old_status,attempt.new_status,jnp.full_like(side,status.PRIMARY),zero,zero),axis=1)
    same &= (~applied|jnp.all(tape.words[rows,first]==status_words,axis=1))
    same &= (_validate(ledger,context.tables)&_tape(ledger,tape,context.tables)&toxic.supported_raw(expected)
        &(attempt.old_counter>=0)&(attempt.old_counter<=15)&(attempt.new_counter>=0)&(attempt.new_counter<=15)
        &(ledger.site[rows,slot]==int(Site.TOXIC_ATTEMPT))&(ledger.side[rows,slot]==1-side)
        &(tape.count==start+1+applied.astype(jnp.int32))&jnp.all(raw==expected,axis=1)&(start>0)
        &(decl[:,0]==toxic.public.MOVE)&(decl[:,1]==side)&(decl[:,2]==physical)
        &(decl[:,3]==1-side)&(decl[:,4]==target)&(decl[:,5]==move)&jnp.all(decl[:,6:]==0,axis=1)
        &life._valid(tape.cursor)&(physical==tape.cursor.physical_active[rows,safe])
        &(target==tape.cursor.physical_active[rows,1-safe])
        &(tape.cursor.turn==ledger.turn[rows,slot])&(tape.cursor.phase==ledger.phase[rows,slot])
        &jnp.all(tape.cursor.generation==ledger.generation[rows,slot],axis=1))
    for index in (prefix,first,at):
        same &= ((tape.event_turn[rows,index]==ledger.turn[rows,slot])&(tape.event_phase[rows,index]==ledger.phase[rows,slot])
            &jnp.all(tape.event_generation[rows,index]==ledger.generation[rows,slot],axis=1))
    return same


def capture_toxic_attempt(tape,ticket,side,physical,target,move,attempt,*,requested):
    context=_context(tape)
    if context is None:return tape
    _shape(requested,tape.count.shape,jnp.bool_);same=_toxic_span_matches(tape,ticket,side,physical,target,move,attempt)
    ledger=unhandled(context.ledger,ticket,active=requested&~same,tables=context.tables)
    ledger=emitted(ledger,ticket,tape,active=requested&same,tables=context.tables)
    return _put_context(tape,ledger,context)


def capture_leftovers_full_hp(tape,receipt,*,no_output_tables):
    """Capture ONLY the typed actual full-HP Leftovers branch, never a generic empty span.

    The source-specific producer must run before/after the SAME numerical write.
    This records branch facts at stage2; both genuine owner consumers still have
    to drain the entire frame before stage3. Prior failures are never cleared.
    """
    from . import engine_leftovers_no_output as no_output
    if type(receipt) is not no_output.Receipt or type(receipt.before) is not no_output.Before:
        raise ValueError('source-specific full-HP Leftovers receipt required')
    context,namespace=no_output._context(tape,no_output_tables)
    ledger=context.ledger;before=receipt.before;slot,same=_ticket(ledger,before.ticket)
    b=ledger.stream_id.shape[0];rows=jnp.arange(b)
    if not no_output.leftovers._check(before.tape,no_output_tables.leftovers):
        raise ValueError('actual retained full-HP BEFORE tape required')
    _shape(before.binding,(8,),jnp.uint32);_shape(before.requested,(b,),jnp.bool_)
    for name in ('ledger_count','next_serial','residual_event','side','physical','item','hp','maximum','status','reason'):
        _shape(getattr(before,name),(b,),jnp.int32)
    for name in ('after_count','after_hp','after_maximum','after_status','after_item','after_physical',
                 'upstream_refusal','reason'):
        _shape(getattr(receipt,name),(b,),jnp.int32)
    namespace &= jnp.all(before.binding==no_output_tables.binding)
    same &= ((before.tape.stream_id==tape.stream_id)&(before.tape.turn==tape.turn)
        &(before.tape.refusal==tape.refusal))
    for name in ('words','event_turn','event_phase','event_generation','switch_details'):
        old,new=getattr(before.tape,name),getattr(tape,name)
        _shape(old,new.shape,new.dtype)
        same &= jnp.all(old==new,axis=tuple(range(1,new.ndim)))
    for old,new in zip(before.tape.cursor,tape.cursor):
        same &= jnp.all(old==new,axis=tuple(range(1,new.ndim))) if new.ndim>1 else old==new
    for name in ('engine_binding','source_binding','move_binding','producer_binding'):
        namespace &= jnp.all(getattr(before.tape,name)==getattr(tape,name))
    same &= ((ledger.site[rows,slot]==int(Site.LEFTOVERS))&(ledger.side[rows,slot]==before.side)
        &(ledger.stage[rows,slot]==1)&(ledger.start[rows,slot]==before.tape.count)
        &(ledger.end[rows,slot]==before.tape.count)&(tape.count==before.tape.count)
        &(ledger.count==before.ledger_count)&(ledger.next_serial==before.next_serial)
        &(ledger.turn[rows,slot]==tape.cursor.turn)&(ledger.phase[rows,slot]==tape.cursor.phase)
        &jnp.all(ledger.generation[rows,slot]==tape.cursor.generation,axis=1))
    same &= ((receipt.after_count==tape.count)&(receipt.after_hp==before.hp)
        &(receipt.after_maximum==before.maximum)&(receipt.after_status==before.status)
        &(receipt.after_item==before.item)&(receipt.after_physical==before.physical))
    facts=jnp.stack((before.physical,before.item,before.hp,before.maximum,before.status,before.residual_event),axis=1)
    specific=((before.residual_event>=2)&(before.residual_event<4)&(before.side==before.residual_event%2)
        &(before.physical>=0)&(before.physical<6)&(before.item==no_output_tables.leftovers.leftovers_item)
        &(before.physical==tape.cursor.physical_active[rows,jnp.clip(before.side,0,1)])
        &(before.item>0)&(before.hp>0)&(before.hp==before.maximum)&(before.status>=0)&(before.status<7))
    good=namespace&same&specific&(before.reason==0)&(receipt.reason==0)
    use=before.requested&good
    ledger=unhandled(ledger,before.ticket,active=before.requested&~good,tables=context.tables)
    def put(array,value):
        old=array[rows,slot]
        return array.at[rows,slot].set(jnp.where(use.reshape((b,)+(1,)*(old.ndim-1)),value,old))
    ledger=ledger._replace(stage=put(ledger.stage,jnp.full(b,2,jnp.int32)),
        no_output_kind=put(ledger.no_output_kind,jnp.ones(b,jnp.int32)),
        no_output_facts=put(ledger.no_output_facts,facts),
        no_output_binding=put(ledger.no_output_binding,jnp.broadcast_to(no_output_tables.binding,(b,8))),
        upstream_refusal=ledger.upstream_refusal|jnp.where(before.requested,receipt.upstream_refusal|tape.refusal,0))
    return _put_context(tape,ledger,context)


def join_consumed_owners(first, second, *, tables):
    """Credit comes from TWO genuine consume_frame calls, never an input bool."""
    a,b=report(first,tables=tables),report(second,tables=tables)
    same=((first.frame==second.frame)&(first.stream_id==second.stream_id)
        & (first.count==second.count)&(first.next_serial==second.next_serial))
    for name in ('site','side','serial','start','end','turn','phase','generation','claimed','words',
                 'event_turn','event_phase','event_generation','no_output_kind','no_output_facts',
                 'no_output_binding'):
        same &= jnp.all((getattr(first,name)==getattr(second,name)).reshape((first.stream_id.shape[0],-1)),axis=1)
    good=a.scoped_drained & b.scoped_drained & same
    joined=first._replace(stage=jnp.minimum(first.stage,second.stage),opened=first.opened|second.opened,
        refusal=first.refusal|second.refusal|jnp.where(good,0,CONSUMPTION),
        upstream_refusal=first.upstream_refusal|second.upstream_refusal)
    return joined,good


def initialize(stream_ids, *, compiled, capacity=16, receipt_capacity=64):
    """Private allocation only. Does NOT establish actual source origin."""
    tables, _, _ = _checked(compiled)
    if getattr(stream_ids, 'ndim', None) != 1:
        raise ValueError('B-leading stream vector required')
    b = stream_ids.shape[0]
    _shape(stream_ids, (b,), jnp.int32)
    if not b or np.any(np.asarray(stream_ids) <= 0) or len(set(np.asarray(stream_ids).tolist())) != b:
        raise ValueError('unique positive stream IDs required')
    if type(capacity) is not int or not 1 <= capacity <= 32 or type(receipt_capacity) is not int or not 1 <= receipt_capacity <= 64:
        raise ValueError('bounded obligation/receipt capacities required')
    z = lambda *shape: jnp.zeros(shape, jnp.int32)
    return Ledger(stream_ids, tables.inventory_binding, z(b), jnp.zeros(b, bool), z(b), z(b), z(b), z(b),
        *[z(b, capacity) for _ in range(8)], z(b, capacity, 2), z(b, receipt_capacity),
        z(b, receipt_capacity, 10), z(b, receipt_capacity), z(b, receipt_capacity), z(b, receipt_capacity, 2),
        z(b,capacity),z(b,capacity,6),jnp.zeros((b,capacity,8),jnp.uint32))


def _validate(ledger, tables):
    if type(ledger) is not Ledger or type(tables) is not Tables:
        raise ValueError('typed private accounting ledger/tables required')
    b = ledger.stream_id.shape[0]; k = ledger.site.shape[1]; c = ledger.claimed.shape[1]
    for name in ('engine_binding', 'source_binding', 'producer_binding', 'inventory_binding'):
        _shape(getattr(tables, name), (8,), jnp.uint32)
    _shape(ledger.inventory_binding, (8,), jnp.uint32)
    for name in ('stream_id', 'frame', 'next_serial', 'refusal', 'upstream_refusal', 'count'):
        _shape(getattr(ledger, name), (b,), jnp.int32)
    _shape(ledger.opened, (b,), jnp.bool_)
    for name in ('site', 'side', 'serial', 'stage', 'start', 'end', 'turn', 'phase'):
        _shape(getattr(ledger, name), (b, k), jnp.int32)
    _shape(ledger.generation, (b, k, 2), jnp.int32)
    _shape(ledger.claimed, (b, c), jnp.int32); _shape(ledger.words, (b, c, 10), jnp.int32)
    _shape(ledger.event_turn, (b, c), jnp.int32); _shape(ledger.event_phase, (b, c), jnp.int32)
    _shape(ledger.event_generation, (b, c, 2), jnp.int32)
    _shape(ledger.no_output_kind,(b,k),jnp.int32)
    _shape(ledger.no_output_facts,(b,k,6),jnp.int32)
    _shape(ledger.no_output_binding,(b,k,8),jnp.uint32)
    return (jnp.all(ledger.inventory_binding == tables.inventory_binding)
        & (ledger.stream_id>0) & (ledger.frame>=0) & (ledger.next_serial>=0)
        & (ledger.count>=0) & (ledger.count<=k)
        & jnp.all((ledger.stage>=0)&(ledger.stage<=3),axis=1)
        & jnp.all((ledger.no_output_kind>=0)&(ledger.no_output_kind<=1),axis=1)
        & jnp.all((ledger.claimed>=0)&(ledger.claimed<=k),axis=1))


def _tape(ledger, tape, tables):
    if type(tape) is not life.LifecycleTape:
        raise ValueError('exact stamped lifecycle tape required')
    b, c = ledger.claimed.shape
    _shape(tape.words, (b, c, 10), jnp.int32)
    for name in ('count', 'stream_id', 'refusal'):
        _shape(getattr(tape, name), (b,), jnp.int32)
    for name in ('engine_binding', 'source_binding', 'producer_binding'):
        _shape(getattr(tape, name), (8,), jnp.uint32)
    _shape(tape.event_turn, (b, c), jnp.int32); _shape(tape.event_phase, (b, c), jnp.int32)
    _shape(tape.event_generation, (b, c, 2), jnp.int32)
    _shape(tape.cursor.stream_id,(b,),jnp.int32)
    _shape(tape.cursor.engine_binding,(b,8),jnp.uint32)
    _shape(tape.cursor.turn,(b,),jnp.int32); _shape(tape.cursor.phase,(b,),jnp.int32)
    _shape(tape.cursor.generation,(b,2),jnp.int32)
    return ((tape.stream_id == ledger.stream_id) & (tape.cursor.stream_id == ledger.stream_id)
        & (tape.count >= 0) & (tape.count <= c)
        & jnp.all(tape.cursor.engine_binding==tables.engine_binding[None,:],axis=1)
        & jnp.all(tape.engine_binding == tables.engine_binding)
        & jnp.all(tape.source_binding == tables.source_binding)
        & jnp.all(tape.producer_binding == tables.producer_binding))


def _commit(mask, old, new):
    # inventory_binding is GLOBAL (8,) even when B happens to equal8.
    values = {name: jnp.where(mask.reshape((mask.shape[0],)+(1,)*(a.ndim-1)), b, a)
        for name, a, b in zip(old._fields, old, new) if name != 'inventory_binding'}
    return old._replace(**values)


def _fail(ledger, eligible, good, bit):
    return ledger._replace(refusal=ledger.refusal | jnp.where(eligible & ~good, bit, 0))


def begin_frame(ledger, tape, *, frame, active, tables):
    """Fresh exact+1 private frame; never resets lifetime refusal/history."""
    valid = _validate(ledger, tables); b = ledger.stream_id.shape[0]
    _shape(frame, (b,), jnp.int32); _shape(active, (b,), jnp.bool_)
    ns = valid & _tape(ledger, tape, tables)
    good = ns & ~ledger.opened & (ledger.frame < 2**31-1) & (frame == ledger.frame+1)
    candidate = jax.tree.map(jnp.zeros_like, ledger)._replace(stream_id=ledger.stream_id,
        inventory_binding=ledger.inventory_binding, frame=frame, opened=jnp.ones(b, bool),
        next_serial=ledger.next_serial, refusal=ledger.refusal, upstream_refusal=ledger.upstream_refusal | tape.refusal)
    result = _commit(active & good, ledger, candidate)
    result = _fail(result, active, ns, NAMESPACE)
    return _fail(result, active, good, ORDER)


def enter(ledger, tape, site, side, *, active, tables):
    """Call at the actual executed attempt BEFORE its receipt/write, even if blocked."""
    valid = _validate(ledger, tables); b, k = ledger.site.shape
    for a, dt in ((site,jnp.int32),(side,jnp.int32),(active,jnp.bool_)):_shape(a, (b,), dt)
    ns = valid & _tape(ledger, tape, tables)
    room = (ledger.count < k) & (ledger.next_serial < 2**31-1)
    good = ns & ledger.opened & room & (site >= 1) & (site <= len(Site)) & ((side == 0) | (side == 1))
    slot = jnp.minimum(ledger.count, k-1); rows = jnp.arange(b); use = active & good
    def put(a, v):
        old = a[rows, slot]; return a.at[rows, slot].set(jnp.where(use.reshape((b,)+(1,)*(old.ndim-1)), v, old))
    serial = ledger.next_serial+1
    result = ledger._replace(site=put(ledger.site,site), side=put(ledger.side,side), serial=put(ledger.serial,serial),
        stage=put(ledger.stage,jnp.ones(b,jnp.int32)), start=put(ledger.start,tape.count),
        end=put(ledger.end,tape.count), turn=put(ledger.turn,tape.cursor.turn),
        phase=put(ledger.phase,tape.cursor.phase), generation=put(ledger.generation,tape.cursor.generation),
        count=ledger.count+use.astype(jnp.int32), next_serial=ledger.next_serial+use.astype(jnp.int32))
    result = _fail(result, active, ns, NAMESPACE)
    result = _fail(result, active, room, OVERFLOW)
    result = _fail(result, active, good, ORDER)
    ticket = Ticket(ledger.stream_id, ledger.inventory_binding, ledger.frame,
        jnp.where(use,slot,-1), jnp.where(use,serial,-1))
    return result, ticket


def _ticket(ledger, ticket):
    if type(ticket) is not Ticket:raise ValueError('private obligation ticket required')
    b, k = ledger.site.shape; rows = jnp.arange(b); slot = jnp.clip(ticket.slot,0,k-1)
    for name in ('stream_id','frame','slot','serial'):_shape(getattr(ticket,name),(b,),jnp.int32)
    _shape(ticket.inventory_binding,(8,),jnp.uint32)
    return slot, (ledger.opened & (ticket.stream_id==ledger.stream_id) & (ticket.frame==ledger.frame)
        & jnp.all(ticket.inventory_binding==ledger.inventory_binding) & (ticket.slot>=0) & (ticket.slot<ledger.count)
        & (ticket.serial==ledger.serial[rows,slot]))


def unhandled(ledger, ticket, *, active, tables):
    """Sticky immediately at an unsupported actual attempt; never resolve by no-op."""
    valid = _validate(ledger,tables); _, same = _ticket(ledger,ticket)
    _shape(active,ledger.stream_id.shape,jnp.bool_)
    result = _fail(ledger,active,valid & same,ORDER)
    return result._replace(refusal=result.refusal | jnp.where(active,UNKNOWN_SITE,0))


def emitted(ledger, ticket, tape, *, active, tables):
    """Capture actual positive appended span, exact words/stamps; never state diffs.

    This captures transport identity, NOT semantic site-to-kind correctness.
    That mapping belongs to the reviewed actual callsite/emitter integration.
    A no-public-output attempt has no generic exemption: keep it unhandled
    until its source-specific no-output proof is implemented and reviewed.
    """
    valid = _validate(ledger,tables); slot, same = _ticket(ledger,ticket)
    b,c = ledger.claimed.shape; rows=jnp.arange(b); _shape(active,(b,),jnp.bool_)
    start=ledger.start[rows,slot]; span=(jnp.arange(c)[None,:]>=start[:,None]) & (jnp.arange(c)[None,:]<tape.count[:,None])
    ns=valid & _tape(ledger,tape,tables)
    stamps=jnp.all(~span | ((tape.words[:,:,0]>0) & (tape.event_turn==ledger.turn[rows,slot,None])
        & (tape.event_phase==ledger.phase[rows,slot,None])
        & jnp.all(tape.event_generation==ledger.generation[rows,slot,None,:],axis=2)),axis=1)
    good=ns & same & (ledger.stage[rows,slot]==1) & (tape.count>start) & stamps
    good &= jnp.all(~span | (ledger.claimed==0),axis=1)
    use=active & good; mask=use[:,None]&span
    stage=ledger.stage.at[rows,slot].set(jnp.where(use,2,ledger.stage[rows,slot]))
    end=ledger.end.at[rows,slot].set(jnp.where(use,tape.count,ledger.end[rows,slot]))
    result=ledger._replace(stage=stage,end=end,claimed=jnp.where(mask,slot[:,None]+1,ledger.claimed),
        words=jnp.where(mask[:,:,None],tape.words,ledger.words),event_turn=jnp.where(mask,tape.event_turn,ledger.event_turn),
        event_phase=jnp.where(mask,tape.event_phase,ledger.event_phase),
        event_generation=jnp.where(mask[:,:,None],tape.event_generation,ledger.event_generation),
        upstream_refusal=ledger.upstream_refusal|jnp.where(active,tape.refusal,0))
    result=_fail(result,active,ns,NAMESPACE)
    return _fail(result,active,good,RECEIPT)


def consume_frame_device(ledger, frame, stream, public, *, active, tables, adapter_tables, catalog,
                         weather_hp=None, leftovers=None, status_residual=None,leftovers_no_output=None):
    """Trusted owning closure: actually project/fold before resolving ANY span.

    Dynamic objects remain trusted same-core inputs, as in the original full
    adapter. The host consume_frame checks the genuine compiled adapter edges.
    There is deliberately no public resolve(accepted=True) API.
    """
    valid=_validate(ledger,tables); b=ledger.stream_id.shape[0]; _shape(active,(b,),jnp.bool_)
    tape=frame.callbacks; ns=valid & _tape(ledger,tape,tables)
    claimed=ledger.claimed>0
    exact=jnp.all(~claimed | ((jnp.all(ledger.words==tape.words,axis=2))
        & (ledger.event_turn==tape.event_turn) & (ledger.event_phase==tape.event_phase)
        & jnp.all(ledger.event_generation==tape.event_generation,axis=2)
        & (jnp.arange(claimed.shape[1])[None,:]<tape.count[:,None])),axis=1)
    # These are the SAME typed contexts guarded by the actual owning driver,
    # not an alternate resolver or a caller-provided consumption result.
    consumed=full.project_and_fold(stream,frame,adapter_tables,public,catalog,
        weather_hp=weather_hp,leftovers=leftovers,status_residual=status_residual)
    projection=consumed.projection
    emitted_count=jnp.sum(projection.frames.valid,axis=(0,2),dtype=jnp.int32)
    # A genuine nonempty consumed frame advances the SAME public/transport cursor
    # by exactly every expanded word; no caller-supplied prefix or accepted flag.
    drained=(projection.error==0) & projection.component_admitted
    drained &= (projection.state.sequence==stream.sequence+emitted_count)
    drained &= (consumed.public.cursor.sequence==projection.state.sequence)
    drained &= (consumed.public.cursor.stream_id==ledger.stream_id)
    drained &= (projection.state.causal.cursor.stream_id==ledger.stream_id)
    drained &= jnp.where(jnp.any(claimed,axis=1),emitted_count>0,True)
    present=jnp.arange(ledger.stage.shape[1])[None,:]<ledger.count[:,None]
    positive=(ledger.no_output_kind==0)&(ledger.end>ledger.start)
    silent=jnp.zeros_like(positive)
    if leftovers_no_output is not None:
        from . import engine_leftovers_no_output as no_output
        if type(leftovers_no_output) is not no_output.Tables:
            raise ValueError('original typed Leftovers no-output tables required')
        _shape(leftovers_no_output.binding,(8,),jnp.uint32)
        source_same=jnp.array(True)
        for actual,expected in zip(leftovers_no_output.accounting,tables):source_same &= jnp.all(actual==expected)
        facts=ledger.no_output_facts
        silent=((ledger.no_output_kind==1)&(ledger.site==int(Site.LEFTOVERS))
            &(ledger.end==ledger.start)&source_same
            &(facts[:,:,0]==jnp.take_along_axis(tape.cursor.physical_active,jnp.clip(ledger.side,0,1),axis=1))
            &jnp.all(ledger.generation==tape.cursor.generation[:,None,:],axis=2)
            &jnp.all(ledger.no_output_binding==leftovers_no_output.binding[None,None,:],axis=2)
            &(facts[:,:,0]>=0)&(facts[:,:,0]<6)&(facts[:,:,1]==leftovers_no_output.leftovers.leftovers_item)
            &(facts[:,:,1]>0)&(facts[:,:,2]>0)&(facts[:,:,2]==facts[:,:,3])
            &(facts[:,:,4]>=0)&(facts[:,:,4]<7)&(facts[:,:,5]>=2)&(facts[:,:,5]<4)
            &(ledger.side==facts[:,:,5]%2))
    captured=(ledger.stage==2)&(ledger.start>=0)&(ledger.end<=tape.count[:,None])&(positive|silent)
    all_emitted=jnp.all(~present|captured,axis=1)
    good=ns & ledger.opened & exact & all_emitted & drained
    candidate=ledger._replace(stage=jnp.where(present,3,ledger.stage),opened=jnp.zeros(b,bool),
        upstream_refusal=ledger.upstream_refusal|tape.refusal)
    result=_commit(active & good,ledger,candidate)
    result=_fail(result,active,ns,NAMESPACE)
    result=_fail(result,active,exact,RECEIPT)
    result=_fail(result,active,all_emitted,UNRESOLVED)
    result=_fail(result,active,ledger.opened & drained,CONSUMPTION)
    # Even a refused attempted frame closes its private accounting window;
    # lifetime refusal survives the next frame. Roll back ALL consumer state
    # on accounting failure, as on an actual full-consumer failure.
    # Rejected attempts retain their newly observed source diagnostics too.
    # Public rollback must not erase them; inactive private history is frozen.
    result=result._replace(opened=jnp.where(active,False,result.opened),
        upstream_refusal=result.upstream_refusal|jnp.where(active,tape.refusal,0))
    use=active & good & (result.refusal==0)
    error=jnp.where(active & ~use,jnp.where(projection.error!=0,projection.error,int(full.inputs.Error.PAYLOAD)),0)
    safe_projection=projection._replace(state=full._commit(use,stream,projection.state),error=error,
        component_admitted=use,chunk=projection.chunk._replace(
            words=jnp.where(use[:,None,None],projection.chunk.words,0),
            valid=projection.chunk.valid & use[:,None],more=projection.chunk.more & use),
        frames=full.FrameChunks(jnp.where(use[None,:,None,None],projection.frames.words,0),
            projection.frames.valid & use[None,:,None],projection.frames.more & use[None,:]))
    return result,full.Consumed(safe_projection,full._commit_public(use,public,consumed.public))


def consume_frame(ledger, frame, stream, public, *, active, compiled):
    tables,_,adapter=_checked(compiled)
    return consume_frame_device(ledger,frame,stream,public,active=active,tables=tables,
        adapter_tables=adapter.device,catalog=adapter.catalog)


def report(ledger, *, tables):
    """Partial accounting only; never feed scoped_drained alone to admission."""
    valid=_validate(ledger,tables)
    present=jnp.arange(ledger.stage.shape[1])[None,:]<ledger.count[:,None]
    unresolved=jnp.sum(present & (ledger.stage!=3),axis=1,dtype=jnp.int32)
    return Report(valid & ~ledger.opened & (ledger.frame>0) & (ledger.refusal==0) & (unresolved==0),
        ledger.refusal,ledger.upstream_refusal,ledger.count,unresolved)
