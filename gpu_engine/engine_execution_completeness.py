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


UNKNOWN_EXECUTION = 512


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


def join_consumed_owners(first, second, *, tables):
    """Credit comes from TWO genuine consume_frame calls, never an input bool."""
    a,b=report(first,tables=tables),report(second,tables=tables)
    same=((first.frame==second.frame)&(first.stream_id==second.stream_id)
        & (first.count==second.count)&(first.next_serial==second.next_serial))
    for name in ('site','side','serial','start','end','turn','phase','generation','claimed','words',
                 'event_turn','event_phase','event_generation'):
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
        z(b, receipt_capacity, 10), z(b, receipt_capacity), z(b, receipt_capacity), z(b, receipt_capacity, 2))


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
    return (jnp.all(ledger.inventory_binding == tables.inventory_binding)
        & (ledger.stream_id>0) & (ledger.frame>=0) & (ledger.next_serial>=0)
        & (ledger.count>=0) & (ledger.count<=k)
        & jnp.all((ledger.stage>=0)&(ledger.stage<=3),axis=1)
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
                         weather_hp=None, leftovers=None):
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
        weather_hp=weather_hp,leftovers=leftovers)
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
    all_emitted=jnp.all(~present | (ledger.stage==2),axis=1)
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
