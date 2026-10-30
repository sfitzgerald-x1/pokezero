"""Specific actual-kernel FULL-HP Leftovers no-public-output witness.

Capture BEFORE held.leftovers_heal at residual event2/3 and finish immediately
AFTER the same HP assignment. This is a literal kernel branch receipt, not a
generic empty-span exemption, HP-delta inference or original-physics proof.
Original Battle.heal runs TryHeal BEFORE its full-HP return; those original
hooks remain diagnostic. RootSession must own the actual cutpoint and ticket;
bare copied snapshots cannot attest that a numerical call actually happened.

This module never edits tape/accounting/public state. The owning accounting
integration must separately recognize this exact typed path, prevent spending
one ticket twice, and require genuine both-owner full-frame drains. No caller
accepted/proof boolean, original-hook-history gate or qualification API exists.
"""
from dataclasses import dataclass
import hashlib
from pathlib import Path
from typing import NamedTuple
import weakref

import jax
import jax.numpy as jnp
import numpy as np

from . import engine_execution_completeness as accounting
from . import engine_leftovers_callbacks as leftovers
from . import engine_weather_speed_profile as speed
from . import engine_lifecycle_callbacks as life, public_mon_state

SCHEMA='pokezero.actual_kernel.leftovers_full_hp_no_output.v1'
NAMESPACE, ATTEMPT, BRANCH, BEFORE, AFTER, OUTPUT, PROFILE, INACTIVE = (1<<n for n in range(8))
_COMPILED=weakref.WeakKeyDictionary()


class Tables(NamedTuple):
    binding: object
    leftovers: leftovers.Data
    profile: speed.ProfileData
    accounting: accounting.Tables


@dataclass(frozen=True,eq=False)
class Compiled:
    device: Tables


class Before(NamedTuple):
    binding: object                    # GLOBAL8
    tape: life.LifecycleTape            # exact BEFORE storage; execution stripped
    ticket: accounting.Ticket
    ledger_count: object
    next_serial: object
    engine_turn: object                # literal kernel counter; NOT public turn
    residual_event: object              # actual event2/3, never inferred HP delta
    side: object
    physical: object
    item: object
    hp: object
    maximum: object
    status: object
    reason: object
    requested: object


class Receipt(NamedTuple):
    before: Before
    after_count: object
    after_hp: object
    after_maximum: object
    after_status: object
    after_item: object
    after_physical: object
    upstream_refusal: object
    reason: object


def _host():
    if isinstance(jnp.zeros((),jnp.int32),jax.core.Tracer):
        raise ValueError('checked full-HP Leftovers boundary cannot be traced')


def compile_no_output(leftovers_compiled,*,profile,accounting_compiled):
    _host()
    data=leftovers.check_compiled(leftovers_compiled);profile_data=speed.check_profile(profile)
    accounting_data,_,adapter=accounting._checked(accounting_compiled)
    engine=leftovers._COMPILED[leftovers_compiled][4]
    if (adapter.causal.projector.catalog is not engine or speed._PROFILES[profile][5] is not engine
            or not np.array_equal(np.asarray(data.engine_binding),np.asarray(profile_data.engine_binding))
            or not np.array_equal(np.asarray(data.source_binding),np.asarray(profile_data.source_binding))):
        raise ValueError('same original Leftovers/profile/accounting catalogs required')
    payload=leftovers._json(dict(schema=SCHEMA,leftovers=np.asarray(data.binding).tolist(),
        profile=np.asarray(profile_data.binding).tolist(),accounting=np.asarray(accounting_data.inventory_binding).tolist(),
        producer=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        kernel=hashlib.sha256(Path(__file__).with_name('items.py').read_bytes()).hexdigest(),
        branch='actual residual event2/3; BEFORE live canonical item and HP==MAX; AFTER same write and no outputs'))
    digest=np.frombuffer(hashlib.sha256(payload.encode()).digest(),dtype='>u4').astype(np.uint32)
    tables=Tables(jnp.asarray(digest),data,profile_data,accounting_data);value=Compiled(tables)
    _COMPILED[value]=(tables,tuple(jax.tree.leaves(tables)),leftovers_compiled,profile,accounting_compiled)
    return value


def check_compiled(compiled):
    _host();record=_COMPILED.get(compiled) if type(compiled) is Compiled else None
    if record is None or compiled.device is not record[0]:raise ValueError('original no-output compiler required')
    leaves=tuple(jax.tree.leaves(compiled.device))
    if (len(leaves)!=len(record[1]) or any(a is not b for a,b in zip(leaves,record[1]))
            or any(not isinstance(a,jax.Array) or isinstance(a,jax.core.Tracer) for a in leaves)
            or leftovers.check_compiled(record[2]) is not compiled.device.leftovers
            or speed.check_profile(record[3]) is not compiled.device.profile
            or accounting._checked(record[4])[0] is not compiled.device.accounting):
        raise ValueError('original immutable no-output compiler edges required')
    return compiled.device


def _vector(value,b):
    value=public_mon_state._int32(value)
    if value.shape not in ((),(b,)):raise ValueError('scalar/B actual residual event required')
    return jnp.broadcast_to(value,(b,))


def _context(tape,tables):
    if type(tables) is not Tables or type(tables.leftovers) is not leftovers.Data or type(tables.profile) is not speed.ProfileData:
        raise ValueError('typed checked no-output tables required')
    life._typed(tables.binding,(8,),jnp.uint32)
    if not leftovers._check(tape,tables.leftovers):raise ValueError('actual lifecycle tape required')
    context=accounting._context(tape)
    if context is None:raise ValueError('actual BEFORE accounting attempt required')
    same=accounting._validate(context.ledger,tables.accounting)&accounting._tape(context.ledger,tape,tables.accounting)
    for a,b in zip(context.tables,tables.accounting):same &= jnp.all(a==b)
    for name in ('engine_binding','source_binding','move_binding','producer_binding'):
        same &= jnp.all(getattr(tape,name)==getattr(tables.leftovers,name))
    same &= life._valid(tape.cursor)&tape.cursor.initialized&(tape.cursor.phase==0)
    return context,same


def _active_values(state,tape,side):
    b=len(tape.count);speed._state_shape(state,tape.stream_id)
    life._typed(state.active,(b,2),jnp.int32)
    rows=jnp.arange(b);safe=jnp.clip(side,0,1);physical=state.active[rows,safe]
    p=jnp.clip(physical,0,5)
    return (physical,*(getattr(state,name)[rows,safe,p] for name in ('item','hp','max_hp','status')))


def capture_before_device(tape,ticket,state,residual_event,record,tables,*,requested):
    """Trusted exact pre-dispatch seam. requested is actual attempt eligibility.

    Full HP is an explicit BEFORE branch condition, not inferred from a later
    no-change result. The original typed entered LEFTOVERS ticket is mandatory.
    """
    context,namespace=_context(tape,tables);b=len(tape.count)
    requested=leftovers._eligible(requested,tape.count)
    malformed=jnp.zeros(b,jnp.bool_)
    try:event=_vector(residual_event,b)
    except (TypeError,ValueError):event=jnp.zeros(b,jnp.int32);malformed=jnp.ones(b,jnp.bool_)
    side=event%2;physical,item,hp,maximum,status=_active_values(state,tape,side)
    retained=speed.retain_device(record,state,tape.stream_id,tables.profile)
    ledger=context.ledger;slot,same=accounting._ticket(ledger,ticket);rows=jnp.arange(b)
    attempt=(same&(ledger.site[rows,slot]==int(accounting.Site.LEFTOVERS))
        &(ledger.side[rows,slot]==side)&(ledger.stage[rows,slot]==1)
        &(ledger.start[rows,slot]==tape.count)&(ledger.end[rows,slot]==tape.count)
        &(ledger.turn[rows,slot]==tape.cursor.turn)&(ledger.phase[rows,slot]==tape.cursor.phase)
        &jnp.all(ledger.generation[rows,slot]==tape.cursor.generation,axis=1))
    branch=(event>=2)&(event<4)&~malformed
    identity=(physical>=0)&(physical<6)&(physical==tape.cursor.physical_active[rows,side])
    identity &= (state.turn>=0)&~state.done
    identity &= state.species[rows,side,jnp.clip(physical,0,5)]==tape.cursor.revealed_species_id[rows,side]
    before=(identity&(hp>0)&(hp==maximum)&(maximum>0)&(item==tables.leftovers.leftovers_item)
        &(tables.leftovers.leftovers_item>0)&(status>=0)&(status<7))
    reason=(jnp.where(namespace,0,NAMESPACE)|jnp.where(attempt,0,ATTEMPT)
        |jnp.where(branch,0,BRANCH)|jnp.where(before,0,BEFORE)
        |jnp.where(retained.qualified_birth,0,PROFILE))
    # Keep this temporary witness finite: do not recursively retain the owning
    # accounting context inside a snapshot later passed back to accounting.
    return Before(tables.binding,tape._replace(execution=None),ticket,ledger.count,ledger.next_serial,state.turn,
        event,side,physical,item,hp,maximum,status,
        jnp.where(requested,reason,INACTIVE),requested)


def finish_device(before,tape,state,record,tables):
    """Trusted AFTER-SAME-write seam; returns only this typed branch witness.

    Unchanged output storage is necessary but NOT sufficient: full-HP BEFORE,
    entered attempt identity, canonical item, exact active target/stamps and
    existing profile carry are all independently checked. No amount formula.
    """
    if type(before) is not Before:raise ValueError('typed full-HP BEFORE snapshot required')
    context,namespace=_context(tape,tables);prior=before.tape;b=len(tape.count)
    if not leftovers._check(prior,tables.leftovers):raise ValueError('typed retained BEFORE lifecycle tape required')
    if prior.words.shape!=tape.words.shape:raise ValueError('same retained callback storage required')
    for name in ('ledger_count','next_serial','engine_turn','residual_event','side','physical','item','hp','maximum','status','reason'):
        life._typed(getattr(before,name),(b,),jnp.int32)
    life._typed(before.requested,(b,),jnp.bool_);life._typed(before.binding,(8,),jnp.uint32)
    namespace &= jnp.all(before.binding==tables.binding)
    physical,item,hp,maximum,status=_active_values(state,tape,before.side)
    retained=speed.retain_device(record,state,tape.stream_id,tables.profile)
    # Recheck the original entered attempt has not been emitted/spent/replaced.
    ledger=context.ledger;slot,same=accounting._ticket(ledger,before.ticket);rows=jnp.arange(b)
    same &= ((ledger.site[rows,slot]==int(accounting.Site.LEFTOVERS))&(ledger.side[rows,slot]==before.side)
        &(ledger.stage[rows,slot]==1)&(ledger.start[rows,slot]==prior.count)
        &(ledger.end[rows,slot]==prior.count)&(ledger.count==before.ledger_count)
        &(ledger.next_serial==before.next_serial))
    same &= ((ledger.turn[rows,slot]==prior.cursor.turn)&(ledger.phase[rows,slot]==prior.cursor.phase)
        &jnp.all(ledger.generation[rows,slot]==prior.cursor.generation,axis=1))
    cursor=jnp.ones(b,jnp.bool_)
    for a,z in zip(prior.cursor,tape.cursor):
        cursor &= jnp.all(a==z,axis=tuple(range(1,a.ndim))) if a.ndim>1 else a==z
    namespace &= (tape.stream_id==prior.stream_id)&(tape.turn==prior.turn)
    for name in ('engine_binding','source_binding','move_binding','producer_binding'):
        namespace &= jnp.all(getattr(prior,name)==getattr(tape,name))
    equal=lambda a,z:jnp.all(a==z,axis=tuple(range(1,a.ndim)))
    # Full retained storage equality catches appends, overwritten prefix,
    # hidden padding writes and altered causal stamps; no truncation/empty-span.
    output=(tape.count==prior.count)&equal(tape.words,prior.words)&equal(tape.event_turn,prior.event_turn)
    output &= equal(tape.event_phase,prior.event_phase)&equal(tape.event_generation,prior.event_generation)
    output &= equal(tape.switch_details,prior.switch_details)&(tape.refusal==prior.refusal)
    after=(cursor&(physical==before.physical)&(item==before.item)&(hp==before.hp)
        &(maximum==before.maximum)&(status==before.status)&(state.turn==before.engine_turn)&~state.done)
    after &= state.species[rows,jnp.clip(before.side,0,1),jnp.clip(physical,0,5)]==prior.cursor.revealed_species_id[rows,jnp.clip(before.side,0,1)]
    # These are branch facts, not an accepted bit carried from capture.
    branch=((before.residual_event>=2)&(before.residual_event<4)
        &(before.side==before.residual_event%2)&(before.physical>=0)&(before.physical<6))
    bound_before=((before.engine_turn>=0)&(before.hp>0)&(before.hp==before.maximum)&(before.maximum>0)
        &(before.item==tables.leftovers.leftovers_item)&(before.item>0)
        &(before.status>=0)&(before.status<7)
        &(before.physical==prior.cursor.physical_active[rows,jnp.clip(before.side,0,1)]))
    reason=(before.reason|jnp.where(namespace,0,NAMESPACE)|jnp.where(same,0,ATTEMPT)
        |jnp.where(branch,0,BRANCH)|jnp.where(bound_before,0,BEFORE)
        |jnp.where(after,0,AFTER)|jnp.where(output,0,OUTPUT)|jnp.where(retained.qualified_birth,0,PROFILE))
    return Receipt(before,tape.count,hp,maximum,status,item,physical,prior.refusal|tape.refusal,reason)


def capture_before(tape,ticket,state,residual_event,record,*,compiled,requested):
    _host()
    return capture_before_device(tape,ticket,state,residual_event,record,check_compiled(compiled),requested=requested)


def finish(before,tape,state,record,*,compiled):
    _host()
    return finish_device(before,tape,state,record,check_compiled(compiled))
