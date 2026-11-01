"""Opt-in actual engine -> retained173 writer driver, no qualification.

The normal placeholder benchmark remains unchanged. This driver consumes each
new lifecycle frame once, in the same call that invokes the real core. Two
owners retain separate B-row observer states; physical stream IDs are never
duplicated into an ambiguous2B observer batch.

The default basic path reports its actual retained-format refusal. An explicit
retained-full path consumes source-derived raw controls and the full retained
fold. Neither path copies basic public state into the full context or reuses an
old observation after refusal. Raw unwritten-mask totals include intentional
role-specific zero defaults; they are descriptive, not a completeness verdict.
The explicit source-witness mode can establish logical input coverage for one
actual frame, after both full folds. Global B4/physics qualification stays false.
Only a pure FAKE forward interface is exposed for source integration tests.
"""
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import time
from typing import NamedTuple
import weakref

import jax
import jax.numpy as jnp
import numpy as np

from . import engine_request_source as source, engine_private_actions as actions
from . import engine_private_team as team, engine_request_order as order
from . import engine_lifecycle_callbacks as life, engine_public_events as events
from . import engine_public_event_device as base, engine_lifecycle_event_device as public
from . import engine_own_facts as own, public_context, public_mon_state, public_belief
from . import private_actions, v4_encoder, transition
from . import engine_weather_speed_profile as weather_speed
from . import engine_execution_completeness as execution
from . import engine_kernel_frame_risk as kernel_risks
from . import engine_outcome_diagnostics as outcome_diagnostics
from . import jax_compile_observations as compile_observations
from . import engine_direct_hit_controls as direct_hit_controls

SCHEMA = 'pokezero.engine_retained173_driver.partial.v1'
SOURCE, PUBLIC_COMPONENT, FULL_PUBLIC, REQUEST, OWN, ENCODER, PENDING, CELLS, PUBLIC_COVERAGE = (1 << n for n in range(9))
REASONS = {SOURCE:'source-lifecycle-unknown', PUBLIC_COMPONENT:'basic-public-consumer-refusal',
    FULL_PUBLIC:'retained-public-consumer-refusal', REQUEST:'actual-own-request-refusal',
    OWN:'own-facts-refusal', ENCODER:'retained-logical-writer-refusal', PENDING:'pending-numeric-columns',
    CELLS:'nonzero-cell-without-writer', PUBLIC_COVERAGE:'incomplete-public-line-consumers'}


class Core(NamedTuple):
    engine: object
    cursor: object
    orders: tuple
    originals: tuple
    constructors: tuple
    unsafe: tuple
    requests: tuple
    admitted: object
    frame_number: object
    weather_profile: object = None
    execution: object = None  # PRIVATE joined closed ledger
    risks: object = None  # PRIVATE sticky actual-kernel diagnostics
    frame_inputs: object = None  # PRIVATE original per-frame witness carry
    frame_input_report: object = None  # PRIVATE current frame input evidence
    terminal_closeout: object = None  # Actual final direct-KO receipt AND both owner drains
    baton_history: object = None # row-only genuine consumed source transaction


class BasicPublic(NamedTuple):
    cursor: object
    beliefs: object
    mons: object
    context: object


class Observer(NamedTuple):
    owner: object
    projection: object
    basic: BasicPublic
    retained: v4_encoder.State
    retained_healthy: object
    fake_calls: object
    accounting: object = None  # PRIVATE genuine consume diagnostics


class FakeInputs(NamedTuple):
    numeric_features: object
    categorical_ids: object
    token_type_ids: object
    attention_mask: object
    legal_action_mask: object


class FakeScores(NamedTuple):
    logits: object
    calls: object


class OwnerResult(NamedTuple):
    own: object
    encoded: object
    basic_error: object
    retained_error: object
    reason: object
    unwritten_numeric: object
    unwritten_categorical: object
    policy_admitted: object
    scored: object
    fake_logits: object
    fake_calls: object
    full_line_consumers_complete: object
    full_encoder_admitted: object
    B4_qualified: object


class DriverState(NamedTuple):
    core: Core
    observers: tuple


class Result(NamedTuple):
    state: DriverState
    owners: tuple


class FrameDiagnostics(NamedTuple):
    invalid_actions: object
    placeholder_moves: object
    active_rows: object
    executed_rows: object
    # Optional host-only measurements; never part of Core or a device kernel.
    host_timings: object = None


class AdvanceWithDiagnostics(NamedTuple):
    """Exact successful result and diagnostics from one checked transaction."""
    result: Result
    diagnostics: FrameDiagnostics


@dataclass(frozen=True, eq=False)
class Prepared:
    binding: object


@dataclass(eq=False)
class Session:
    """Original host session; advance owns its one current device carry."""
    prepared: Prepared
    state: DriverState
    result: Result


_PREPARED = weakref.WeakKeyDictionary()
_SESSIONS = weakref.WeakKeyDictionary()
_REQUEST_PROOF_ORIGINS = weakref.WeakKeyDictionary()
_LEECH_HISTORY_ORIGINS = weakref.WeakKeyDictionary()
_HOST_GUARD_TIMING = ContextVar('pokezero_driver_host_guard_timing', default=None)


class _HostGuardTimingScope:
    """Shared lexical lifetime, including copies of the current Context."""
    __slots__ = ('timings','group','active')
    def __init__(self,timings,group):
        self.timings,self.group,self.active = timings,group,True


def _basic_initialize(catalog, context, streams):
    b = streams.shape[0]
    beliefs = public_belief.initialize(jnp.zeros((b, 2, 6), jnp.int32), catalog.source.device)
    mons = public_mon_state.initialize(b, species_binding=catalog.species.device.binding)
    return BasicPublic(v4_encoder.public_inputs.initialize(streams, source_binding=catalog.source.device.binding),
        beliefs, mons, public_context.initialize(beliefs, context, species_binding=catalog.species.device.binding))


def _rows(ok, before, after):
    """Select explicitly row-only trees. Global namespaces must be detached."""
    def choose(a, b):
        if a.ndim == 0 or a.shape[0] != ok.shape[0] or a.shape != b.shape:
            raise ValueError('declared row tree contains a non-row leaf')
        return jnp.where(ok.reshape((ok.shape[0],)+(1,)*(a.ndim-1)), b, a)
    return jax.tree.map(choose, before, after)


def _public_rows(ok, before, after):
    from . import public_refresh_stream
    old_refresh, new_refresh = before.context.refresh, after.context.refresh
    refresh = None if old_refresh is None and new_refresh is None else public_refresh_stream.commit(
        ok, old_refresh, new_refresh)
    selected = _rows(ok, before._replace(context=before.context._replace(refresh=None)),
        after._replace(context=after.context._replace(refresh=None)))
    return selected._replace(context=selected.context._replace(refresh=refresh))


def _core_rows(ok, before, after):
    ledger=None if before.execution is None else execution.commit_rows(ok,before.execution,after.execution)
    risks=_risk_rows(ok,before.risks,after.risks)
    witness=None
    if before.frame_inputs is not None or after.frame_inputs is not None:
        from . import engine_source_frame_witness as frame_witness
        witness=frame_witness.commit_rows(ok,before.frame_inputs,after.frame_inputs)
    return _rows(ok,before._replace(execution=None,risks=None,frame_inputs=None),
        after._replace(execution=None,risks=None,frame_inputs=None))._replace(
        execution=ledger,risks=risks,frame_inputs=witness)


def _risk_rows(active,before,after):
    if before is None and after is None:return None
    if type(before) is not kernel_risks.Carry or type(after) is not kernel_risks.Carry:
        raise ValueError('actual typed risk carry must survive every owning phase')
    b=before.stream_id.shape[0]
    execution._shape(active,(b,),jnp.bool_)
    for carry in (before,after):
        for name in ('stream_id','frame','risk','observed'):
            execution._shape(getattr(carry,name),(b,),jnp.int32)
        execution._shape(carry.opened,(b,),jnp.bool_)
        execution._shape(carry.counts,(b,len(kernel_risks.Branch)),jnp.int32)
    selected=kernel_risks.commit_rows(active,before,after)
    same=jnp.array(True)
    for name in ('binding','accounting_binding'):
        old,new=getattr(before,name),getattr(after,name)
        execution._shape(old,(8,),jnp.uint32);execution._shape(new,(8,),jnp.uint32)
        same &= jnp.all(old==new)
    same &= before.stream_id==after.stream_id
    return selected._replace(stream_id=before.stream_id,
        risk=before.risk|selected.risk|jnp.where(active&~same,kernel_risks.NAMESPACE,0))


def _observer_rows(ok, before, after):
    basic = _public_rows(ok, before.basic, after.basic)
    retained = _public_rows(ok, before.retained, after.retained)
    accounting=None if before.accounting is None else execution.commit_rows(ok,before.accounting,after.accounting)
    selected = _rows(ok, before._replace(basic=None, retained=None,accounting=None),
        after._replace(basic=None, retained=None,accounting=None))
    return selected._replace(basic=basic, retained=retained,accounting=accounting)


def _report_rows(ok, before, after):
    # These two leaves are schema-wide, including when B happens to equal8.
    def detach(report):
        return report._replace(encoded=report.encoded._replace(pending_numeric_columns=None,
            observation=report.encoded.observation._replace(category_count=None)))
    selected = _rows(ok, detach(before), detach(after))
    return selected._replace(encoded=selected.encoded._replace(
        pending_numeric_columns=before.encoded.pending_numeric_columns,
        observation=selected.encoded.observation._replace(category_count=before.encoded.observation.category_count)))


def _frame(core, statics, engine_actions, chance, active_rows=None, *, weather_profile_data=None,
           leftovers_context=None,execution_tables=None,status_residual_context=None,kernel_risk_tables=None,
           direct_hit_context=None,substitute_context=None,toxic_context=None,recoil_context=None,calm_mind_context=None,swords_dance_context=None,spikes_context=None,thunder_wave_context=None,sleep_powder_context=None,hypothetical_profile=None,terminal_observer=False,leech_history=None,baton_public=None):
    """One real core invocation; no host registration/read in the hot path."""
    catalog, bindings, engine_table = statics
    del catalog  # the real constructor's host catalog is never checked in JIT
    b = core.cursor.stream_id.shape[0]
    before = life.snapshot(core.cursor)
    tape = life.LifecycleTape(jnp.zeros((b, 64, 10), jnp.int32), jnp.zeros(b, jnp.int32),
        jnp.full((b,), events.UNINSTRUMENTED, jnp.int32), core.cursor.turn,
        bindings.engine_binding, bindings.source_binding, bindings.move_binding, bindings.producer_binding,
        core.cursor.stream_id, jnp.zeros((b, 64), jnp.int32), jnp.zeros((b, 64), jnp.int32),
        jnp.zeros((b, 64, 2), jnp.int32), jnp.zeros((b, 64, 6), jnp.int32), core.cursor)
    # Refused source rows are frozen, never advanced using an unproved request.
    source_ready = core.admitted & core.requests[0].actions.admitted & core.requests[1].actions.admitted
    if active_rows is not None:
        source_ready &= active_rows
    if execution_tables is not None:
        tape=execution.begin_on_tape(tape,core.execution,execution_tables,active=source_ready)
        tape=tape._replace(execution=tape.execution._replace(diagnostics=outcome_diagnostics.empty(b),party_orders=core.orders))
    if kernel_risk_tables is not None:
        if execution_tables is None or type(core.risks) is not kernel_risks.Carry:
            raise ValueError('actual original accounting and retained risk carry required')
        carry=kernel_risks.begin_frame_device(core.risks,tape,kernel_risk_tables,active=source_ready)
        tape=tape._replace(execution=tape.execution._replace(risks=kernel_risks.Context(carry,kernel_risk_tables)),
            refusal=tape.refusal|jnp.where(source_ready&(carry.risk!=0),execution.UNKNOWN_KERNEL_RISK,0))
    if direct_hit_context is not None and type(direct_hit_context) is direct_hit_controls.RegisteredContext and direct_hit_context.data.leech_seed is not None and execution_tables is not None:
        from . import engine_leech_seed_callbacks as leech_callbacks
        if leech_history is None:leech_history=leech_callbacks.empty_history(tape.cursor,direct_hit_context.data.leech_seed)
        leech_callbacks.history_shape(leech_history,b)
        tape=tape._replace(execution=tape.execution._replace(leech_history=leech_history))
    if type(direct_hit_context) is direct_hit_controls.RegisteredContext and direct_hit_context.data.baton_pass is not None and baton_public is not None and execution_tables is not None and kernel_risk_tables is not None:
        from . import engine_baton_pass_callbacks as baton_callbacks
        if core.baton_history is None:raise ValueError('owning Baton history unavailable')
        tape=baton_callbacks.prepare(tape,core.engine,core.baton_history,baton_public,core.requests,core.orders,engine_actions,direct_hit_context.data.baton_pass)
    if terminal_observer:
        from . import engine_terminal_callbacks as terminal
        if execution_tables is None or kernel_risk_tables is None or direct_hit_context is None:
            raise ValueError('terminal observer requires original registered direct/accounting/risk contexts')
        tape=terminal.begin_on_tape(tape,core.engine,direct_hit_context,active=source_ready,
            status_data=None if status_residual_context is None else status_residual_context.data)
    masked = core.engine._replace(done=core.engine.done | ~source_ready)
    result = transition.step_with_chance(masked, engine_actions, engine_table, chance,
        _public_trace=tape, _lifecycle_trace=True,
        _weather_hp_context=None if weather_profile_data is None else (core.weather_profile,weather_profile_data),
        _leftovers_context=leftovers_context,_status_residual_context=status_residual_context,
        _direct_hit_context=direct_hit_context,_substitute_context=substitute_context,_toxic_context=toxic_context,_recoil_context=recoil_context,_calm_mind_context=calm_mind_context,_swords_dance_context=swords_dance_context,_spikes_context=spikes_context,_thunder_wave_context=thunder_wave_context,_sleep_powder_context=sleep_powder_context,_hypothetical_profile=hypothetical_profile)
    engine = _rows(source_ready, core.engine, result.step.state)
    cursor = _rows(source_ready, core.cursor, result.callbacks.cursor)
    # Bindings are GLOBAL [8], never row-select them by a coincidental B=8.
    globals_ = {'engine_binding', 'source_binding', 'move_binding', 'producer_binding','execution'}
    callbacks = result.callbacks._replace(**{name: _rows(source_ready, getattr(tape, name),
        getattr(result.callbacks, name)) for name in tape._fields if name not in globals_})
    callbacks = callbacks._replace(cursor=cursor)
    if execution_tables is not None:
        ledger=execution.commit_rows(source_ready,tape.execution.ledger,result.callbacks.execution.ledger)
        # This immutable context is minted by the SAME secondary setter call.
        # Retain it until both FULL owners validate its claimed RAW9+36 span;
        # rebuilding solely from the pre-call context loses that source edge.
        from . import engine_leech_seed_callbacks as leech_callbacks
        from . import engine_thunder_wave_callbacks as thunder_callbacks
        from . import engine_natural_cure_callbacks as natural_cure_callbacks
        context=tape.execution._replace(ledger=ledger,
            leech_history=leech_callbacks.commit_history(source_ready,tape.execution.leech_history,result.callbacks.execution.leech_history),
            secondary_status=result.callbacks.execution.secondary_status,
            baton_transaction=None if tape.execution.baton_transaction is None else baton_callbacks.rows(source_ready,tape.execution.baton_transaction,result.callbacks.execution.baton_transaction),
            self_destruct=result.callbacks.execution.self_destruct,
            natural_cure=natural_cure_callbacks.commit(source_ready,result.callbacks.execution.natural_cure),
            thunder_wave_immunity=thunder_callbacks.commit_ground(source_ready,result.callbacks.execution.thunder_wave_immunity),
            diagnostics=_rows(source_ready,
            tape.execution.diagnostics,result.callbacks.execution.diagnostics))
        if kernel_risk_tables is not None:
            after=result.callbacks.execution.risks
            if type(after) is not kernel_risks.Context:
                raise ValueError('actual kernel discarded its owning risk context')
            carry=_risk_rows(source_ready,tape.execution.risks.carry,after.carry)
            context=context._replace(risks=tape.execution.risks._replace(carry=carry))
            callbacks=callbacks._replace(refusal=callbacks.refusal|jnp.where(
                source_ready&(carry.risk!=0),execution.UNKNOWN_KERNEL_RISK,0))
        if terminal_observer:
            context=context._replace(terminal=terminal.commit_rows(source_ready,
                tape.execution.terminal,result.callbacks.execution.terminal))
        callbacks=callbacks._replace(execution=context)
    profile=None if weather_profile_data is None and hypothetical_profile is None else _rows(source_ready,
        core.weather_profile,result.weather_profile)
    return life.LifecycleResult(engine, cursor, callbacks, before,
        life.snapshot(cursor), jnp.where(source_ready, result.step.invalid_actions, False),
        jnp.where(source_ready, result.step.placeholder_moves, 0),profile)


def _leech_history(runtime,core,publics,healthy):
    context=runtime['direct_hit_context']
    if context is None or context.data.leech_seed is None or runtime['direct_hit_mode']!='kernel-event':return None
    from . import engine_leech_seed_callbacks as leech_callbacks
    data=context.data.leech_seed;active_tables=runtime['catalog'].active.device
    proof=_checked_leech_history_proof(runtime.get('leech_history_proof'),
        leech_callbacks.history_from_public,context,data,active_tables)
    if proof is None:
        return leech_callbacks.history_from_public(core.engine,core.cursor,publics,
            healthy=healthy,data=data,active_tables=active_tables)
    return proof(core.engine,core.cursor,publics,healthy,data,active_tables)


def _request(engine, ledger, original, retained, unsafe, arrival, admitted, statics, current_types=None,current_encore=None):
    """The existing source request + private adapter PURE lower primitives."""
    rules, adapter, private, action_tables, engine_table = statics
    packet, reason = source._produce(engine, ledger, arrival, original, retained, unsafe,
        rules, adapter.binding, engine_table,current_types,current_encore)
    reason |= jnp.where(admitted, 0, source.ARRIVAL)
    packet = packet._replace(complete=packet.complete & admitted,
        mode=jnp.where(admitted, packet.mode, actions.UNKNOWN))
    owned = team.select_owned_team(engine, ledger.owner_side, engine_binding=ledger.engine_binding)
    reordered = order.reorder_owned_team(owned, ledger, boundary=packet.boundary)
    tf = team._team_facts_device(reordered.owned, private, expected_binding=private.binding)
    selected = actions._select(engine, ledger.owner_side)
    request, facts, primary, mechanics, exact, num, den, ok = actions._device(selected, owned, tf.facts,
        ledger, packet, retained, adapter, action_tables, reordered.admitted & tf.admitted)
    rows = jnp.arange(ledger.owner_side.shape[0])
    species = owned.species[rows, jnp.clip(owned.active, 0, 5)]
    ghost = adapter.ghost[jnp.clip(species, 0, adapter.ghost.shape[0]-1)]
    af = private_actions.ActionFacts(request, facts, jnp.broadcast_to(action_tables.binding, packet.binding.shape),
        primary, mechanics, ghost, ok, exact & ok, num.astype(jnp.int32), den.astype(jnp.int32))
    features = private_actions.project_actions(af, action_tables, observer_side=ledger.owner_side)
    accepted = ok & features.admitted
    mapping = jnp.where(accepted[:, None] & request.move_present, jnp.arange(4, dtype=jnp.int32), -1)
    result = actions.Result(actions._sanitize(af, accepted), features, mapping,
        packet.engine_legal & accepted[:, None], packet.boundary, accepted)
    return source.Produced(result, reason | jnp.where((reason == 0) & ~accepted, source.ADAPTER, 0))


def _public(observer, frame, tables, context, catalog):
    p = public.project_and_fold(observer.projection, frame, tables, observer.basic.cursor,
        observer.basic.beliefs, catalog.source.device, mon_state=observer.basic.mons,
        context=observer.basic.context, context_tables=context)
    basic = BasicPublic(p.fold.cursor, p.fold.beliefs, p.fold.mon_state, p.fold.context)
    # Real full context is attempted; missing producers are not filled from the
    # basic component or hidden engine state to make this pass.
    retained, error = v4_encoder.fold_chunk(observer.retained, p.projection.chunk, catalog)
    healthy = observer.retained_healthy & (error == 0) & p.projection.component_admitted
    return observer._replace(projection=p.projection.state, basic=basic, retained=retained,
        retained_healthy=healthy), p.projection, error


def _full_public(observer, frame, tables, catalog, consume, *, weather_hp=None, leftovers=None, active=None,
                 execution_tables=None,status_residual=None,leftovers_no_output=None):
    """Consume genuine full-format controls without relabeling basic state."""
    kwargs = dict(weather_hp=weather_hp)
    if leftovers is not None:
        kwargs['leftovers'] = leftovers
    if status_residual is not None:
        kwargs['status_residual'] = status_residual
    accounting=observer.accounting
    if execution_tables is None:
        if leftovers_no_output is not None:
            raise ValueError('Leftovers no-output requires actual owning accounting')
        result = consume(observer.projection, frame, tables, observer.retained, catalog, **kwargs)
    else:
        if type(frame.callbacks.execution) is not execution.Context:
            raise ValueError('actual accounting frame context required')
        context=frame.callbacks.execution
        if type(context.tables) is not execution.Tables:
            raise ValueError('typed original accounting namespaces required')
        same=jnp.array(True)
        for actual,expected in zip(context.tables,execution_tables,strict=True):
            execution._shape(actual,(8,),jnp.uint32)
            same &= jnp.all(actual==expected)
        ledger=context.ledger._replace(refusal=context.ledger.refusal|jnp.where(same,0,execution.NAMESPACE))
        if active is None:active=jnp.ones_like(observer.owner,dtype=bool)
        accounting,result=execution.consume_frame_device(ledger,frame,
            observer.projection,observer.retained,active=active,tables=execution_tables,
            adapter_tables=tables,catalog=catalog,weather_hp=weather_hp,leftovers=leftovers,
            status_residual=status_residual,leftovers_no_output=leftovers_no_output)
    projection = result.projection
    healthy = observer.retained_healthy & (projection.error == 0) & projection.component_admitted
    return observer._replace(projection=projection.state, retained=result.public,
        retained_healthy=healthy,accounting=accounting), projection, projection.error


def _own_encode(core, observer, catalog, own_tables, *, mask_source=False):
    facts = own._device(core.engine, core.orders, core.requests, observer.owner, own_tables)
    if mask_source:
        mask=core.admitted
        packet=jax.tree.map(lambda x:jnp.where(mask.reshape((mask.shape[0],)+(1,)*(x.ndim-1)),x,jnp.zeros_like(x)),facts.facts)
        facts=facts._replace(facts=packet,admitted=facts.admitted & mask,
            reason=facts.reason|jnp.where(mask,0,own.SOURCE))
    encoded = own._encode(observer.retained, facts.facts, observer.owner, core.cursor.stream_id, catalog)
    return facts, encoded


def _score(observer, facts, encoded, projection, public_error, core_admitted, fake_forward, active_rows=None):
    a = encoded.observation
    attended = a.attention_mask[:, :, None]
    numeric_missing = jnp.sum(attended & ~a.numeric_written, axis=(1, 2))
    categorical_missing = jnp.sum(attended & ~a.categorical_written, axis=(1, 2))
    # These masks track explicit role-specific writes, not dense tensor
    # completeness. The retained source initializes every row to zero; fields
    # outside that token's role intentionally remain unwritten zero defaults.
    # Column implementation is checked separately by pending columns/parity;
    # never count these descriptive mask totals as missing encoder columns.
    invalid_numeric = jnp.any(attended & ~a.numeric_written & (a.numeric_features != 0), axis=(1, 2))
    invalid_categorical = jnp.any(attended & ~a.categorical_written & (a.categorical_ids != 0), axis=(1, 2))
    pending = jnp.full_like(core_admitted, encoded.pending_numeric_columns.shape[0] != 0)
    gates = ((SOURCE, ~core_admitted), (PUBLIC_COMPONENT, projection.error != 0),
        (FULL_PUBLIC, (public_error != 0) | ~observer.retained_healthy),
        (REQUEST, (facts.reason & own.SOURCE) != 0), (OWN, ~facts.admitted),
        (ENCODER, ~a.admitted), (PENDING, pending),
        (CELLS, invalid_numeric | invalid_categorical),
        (PUBLIC_COVERAGE, ~projection.full_line_consumers_complete))
    reason = jnp.zeros_like(observer.owner)
    for bit, missing in gates:
        reason |= jnp.where(missing, bit, 0)
    allowed = reason == 0
    if active_rows is not None:
        allowed &= active_rows
    # This is FAKE-only component eligibility. Scientific qualification stays
    # separate and false; it must not become a permanent false logical gate.
    inputs = FakeInputs(a.numeric_features, a.categorical_ids, a.token_type_ids, a.attention_mask, a.legal_action_mask)
    def score(row):
        use, values = row
        return jax.lax.cond(use, fake_forward,
            lambda _: FakeScores(jnp.zeros((9,), jnp.float32), jnp.array(0, jnp.int32)), values)
    scores = jax.lax.map(score, (allowed, inputs))
    # Even an accidentally ignored admission flag exposes no stale observation.
    safe = a._replace(numeric_features=jnp.where(allowed[:, None, None], a.numeric_features, 0),
        categorical_ids=jnp.where(allowed[:, None, None], a.categorical_ids, 0),
        attention_mask=a.attention_mask & allowed[:, None], legal_action_mask=a.legal_action_mask & allowed[:, None],
        switch_targets=jnp.where(allowed[:, None], a.switch_targets, -1), admitted=allowed)
    encoded = encoded._replace(observation=safe, posterior_mask=jnp.where(allowed[:, None, None], encoded.posterior_mask, 0),
        posterior_weights=jnp.where(allowed[:, None, None], encoded.posterior_weights, 0))
    calls = observer.fake_calls + scores.calls
    observer = observer._replace(fake_calls=calls)
    no = jnp.zeros_like(allowed)
    return observer, OwnerResult(facts, encoded, projection.error, public_error, reason,
        numeric_missing, categorical_missing, allowed, scores.calls != 0, scores.logits, calls,
        projection.full_line_consumers_complete, encoded.full_encoder_admitted, no)


def _prepare_request_proof_pair(function, *, _context=None, **static):
    """Keep only compiled code per owner; every proof input remains current.

    The source/catalog guards at the original phase boundaries remain outside
    these functions. No accepted result or dynamic public history is retained.
    """
    static=tuple(static.items())
    def bind(owner):
        def proof(engine,before,public_state,tape,healthy):
            return function(engine,before,public_state,owner=owner,tape=tape,
                healthy=healthy,**dict(static))
        compiled=jax.jit(proof)
        _REQUEST_PROOF_ORIGINS[compiled]=(owner,function,_context,static)
        return compiled
    return tuple(bind(owner) for owner in (0,1))


def _checked_request_proof_pair(pair,function,context,**static):
    """Current enabled source holders must still match the retained code."""
    if type(pair) is not tuple or len(pair)!=2:
        raise ValueError('original prepared request proof pair required')
    for owner,proof in enumerate(pair):
        original=_REQUEST_PROOF_ORIGINS.get(proof)
        if (original is None or original[0]!=owner or original[1] is not function
                or original[2] is not context or tuple(static)!=tuple(k for k,_ in original[3])
                or any(static[k] is not value for k,value in original[3])):
            raise ValueError('current request proof source holders changed')
    return pair


def _prepare_leech_history_proof(function,context,data,active_tables):
    """Retain code only; all public, physical and table arrays stay inputs."""
    def proof(engine,cursor,publics,healthy,current_data,current_active_tables):
        return function(engine,cursor,publics,healthy=healthy,data=current_data,
            active_tables=current_active_tables)
    compiled=jax.jit(proof)
    _LEECH_HISTORY_ORIGINS[compiled]=(function,context,data,active_tables)
    return compiled


def _checked_leech_history_proof(proof,function,context,data,active_tables):
    """Cache provenance selects code reuse, never runtime admission."""
    try:
        original=_LEECH_HISTORY_ORIGINS.get(proof) if callable(proof) else None
    except TypeError:
        return None
    if (original is None or original[0] is not function or original[1] is not context
            or original[2] is not data or original[3] is not active_tables):
        return None
    return proof


def prepare(request_compiled, own_compiled, *, fake_forward, public_mode='basic', leftovers_compiled=None,
            leftovers_mode='diagnostic',account_source_sites=False,status_residual_compiled=None,
            status_residual_mode='diagnostic',track_kernel_risks=False,
            direct_hit_compiled=None,direct_hit_mode='diagnostic',frame_input_mode='closed',
            substitute_compiled=None,substitute_mode='diagnostic',toxic_compiled=None,toxic_mode='diagnostic',
            recoil_compiled=None,recoil_mode='closed',calm_mind_compiled=None,calm_mind_mode='closed',swords_dance_compiled=None,swords_dance_mode='closed',spikes_compiled=None,spikes_mode='closed',thunder_wave_compiled=None,thunder_wave_mode='closed',sleep_powder_compiled=None,sleep_powder_mode='closed'):
    """Host-only immutable preparation. fake_forward is a pure source-test spy.

    The returned compiled closures contain no registration calls or host reads.
    The checked host API reruns guards before every cached device invocation.
    The explicit retained-full mode consumes the full-format device adapter;
    its remaining coverage and scientific qualification flags stay unchanged.
    Optional Leftovers diagnostics retain UNKNOWN. Explicit retained mode uses
    the original helper/profile and full-consumer contract; all scientific
    qualification flags remain false.
    Explicit kernel-event status mode additionally binds the original status
    helper and actual accounting carry. Only genuine full consumption by both
    observers can close selected BRN/PSN spans. It establishes no original
    physics equivalence or exhaustive event coverage.
    Explicit source-witness mode derives per-frame logical input coverage from
    the actual pinned source and both consumers before encoding. Its default is
    closed; it grants no global B4, physics or strength qualification.
    """
    if type(calm_mind_mode) is not str or calm_mind_mode not in ('closed','kernel-event'):
        raise ValueError('explicit closed or kernel-event Calm Mind mode required')
    if calm_mind_mode=='kernel-event' and (calm_mind_compiled is None or not account_source_sites or public_mode!='retained-full'):
        raise ValueError('kernel-event Calm Mind requires original helper/full/accounting runtime')
    if calm_mind_compiled is not None and calm_mind_mode!='kernel-event':raise ValueError('Calm Mind helper requires explicit kernel-event mode')
    if calm_mind_compiled is not None and isinstance(jnp.zeros((),jnp.int32),jax.core.Tracer):raise ValueError('checked Calm Mind driver must remain outside outer JIT')
    if type(swords_dance_mode) is not str or swords_dance_mode not in ('closed','kernel-event'):
        raise ValueError('explicit closed or kernel-event Swords Dance mode required')
    if swords_dance_mode=='kernel-event' and (swords_dance_compiled is None or not account_source_sites or public_mode!='retained-full'):
        raise ValueError('kernel-event Swords Dance requires original helper/full/accounting runtime')
    if swords_dance_compiled is not None and swords_dance_mode!='kernel-event':raise ValueError('Swords Dance helper requires explicit kernel-event mode')
    if swords_dance_compiled is not None and isinstance(jnp.zeros((),jnp.int32),jax.core.Tracer):raise ValueError('checked Swords Dance driver must remain outside outer JIT')
    if type(thunder_wave_mode) is not str or thunder_wave_mode not in ('closed','kernel-event'):
        raise ValueError('explicit closed or kernel-event Thunder Wave mode required')
    if thunder_wave_mode=='kernel-event' and (thunder_wave_compiled is None or not account_source_sites or public_mode!='retained-full' or not track_kernel_risks):
        raise ValueError('kernel-event Thunder Wave requires original full/accounting/kernel-risk runtime')
    if thunder_wave_compiled is not None and thunder_wave_mode!='kernel-event':raise ValueError('Thunder Wave helper requires explicit kernel-event mode')
    if thunder_wave_compiled is not None and isinstance(jnp.zeros((),jnp.int32),jax.core.Tracer):raise ValueError('checked Thunder Wave driver must remain outside outer JIT')
    if type(sleep_powder_mode) is not str or sleep_powder_mode not in ('closed','kernel-event'):
        raise ValueError('explicit closed or kernel-event Sleep Powder mode required')
    if sleep_powder_mode=='kernel-event' and (sleep_powder_compiled is None or not account_source_sites or public_mode!='retained-full' or not track_kernel_risks):
        raise ValueError('kernel-event Sleep Powder requires original full/accounting/kernel-risk runtime')
    if sleep_powder_compiled is not None and sleep_powder_mode!='kernel-event':raise ValueError('Sleep Powder helper requires explicit kernel-event mode')
    if sleep_powder_compiled is not None and isinstance(jnp.zeros((),jnp.int32),jax.core.Tracer):raise ValueError('checked Sleep Powder driver must remain outside outer JIT')
    if type(spikes_mode) is not str or spikes_mode not in ('closed','kernel-event'):
        raise ValueError('explicit closed or kernel-event Spikes mode required')
    if spikes_mode=='kernel-event' and (spikes_compiled is None or not account_source_sites or public_mode!='retained-full' or not track_kernel_risks):
        raise ValueError('kernel-event Spikes requires original full/accounting/kernel-risk runtime')
    if spikes_compiled is not None and spikes_mode!='kernel-event':raise ValueError('Spikes helper requires explicit kernel-event mode')
    if spikes_compiled is not None and isinstance(jnp.zeros((),jnp.int32),jax.core.Tracer):raise ValueError('checked Spikes driver must remain outside outer JIT')
    if type(recoil_mode) is not str or recoil_mode not in ('closed','kernel-event'):
        raise ValueError('explicit closed or kernel-event recoil mode required')
    if recoil_mode=='kernel-event' and (recoil_compiled is None or direct_hit_compiled is None
            or direct_hit_mode!='kernel-event' or not account_source_sites or public_mode!='retained-full'):
        raise ValueError('kernel-event recoil requires original direct-hit/full/accounting runtime')
    if recoil_compiled is not None and recoil_mode!='kernel-event':
        raise ValueError('original recoil helper requires explicit kernel-event mode')
    if recoil_compiled is not None and isinstance(jnp.zeros((),jnp.int32),jax.core.Tracer):
        raise ValueError('checked recoil driver must remain outside outer JIT')
    if type(frame_input_mode) is not str or frame_input_mode not in ('closed','source-witness'):
        raise ValueError('explicit closed or source-witness frame input mode required')
    if frame_input_mode=='source-witness' and (public_mode!='retained-full'
            or not account_source_sites or not track_kernel_risks):
        raise ValueError('source witness requires actual full/accounting/kernel-risk runtime')
    if type(track_kernel_risks) is not bool:
        raise ValueError('explicit kernel risk tracking mode required')
    if track_kernel_risks and isinstance(jnp.zeros((),jnp.int32),jax.core.Tracer):
        raise ValueError('checked kernel risk driver must remain outside outer JIT')
    if track_kernel_risks and not account_source_sites:
        raise ValueError('kernel risk tracking requires actual source-site accounting')
    if type(toxic_mode) is not str or toxic_mode not in ('diagnostic','kernel-event'):
        raise ValueError('explicit diagnostic or kernel-event Toxic mode required')
    if toxic_mode=='kernel-event' and (toxic_compiled is None or not account_source_sites):
        raise ValueError('kernel-event Toxic requires original helper and actual accounting')
    if toxic_compiled is not None:
        if isinstance(jnp.zeros((),jnp.int32),jax.core.Tracer):raise ValueError('checked Toxic driver must remain outside outer JIT')
        if public_mode!='retained-full':raise ValueError('Toxic requires retained-full mode')
    if type(substitute_mode) is not str or substitute_mode not in ('diagnostic','kernel-event'):
        raise ValueError('explicit diagnostic or kernel-event Substitute mode required')
    if substitute_mode=='kernel-event' and (substitute_compiled is None or not account_source_sites):
        raise ValueError('kernel-event Substitute requires original helper and actual accounting')
    if substitute_compiled is not None:
        if isinstance(jnp.zeros((),jnp.int32),jax.core.Tracer):
            raise ValueError('checked Substitute driver must remain outside outer JIT')
        if public_mode!='retained-full':raise ValueError('Substitute requires retained-full mode')
    if type(direct_hit_mode) is not str or direct_hit_mode not in ('diagnostic','kernel-event'):
        raise ValueError('explicit diagnostic or kernel-event direct-hit mode required')
    if direct_hit_mode=='kernel-event' and (direct_hit_compiled is None or not account_source_sites):
        raise ValueError('kernel-event direct hits require original helper and actual source accounting')
    if direct_hit_compiled is not None:
        if isinstance(jnp.zeros((),jnp.int32),jax.core.Tracer):
            raise ValueError('checked direct-hit driver must remain outside outer JIT')
        if public_mode!='retained-full':raise ValueError('direct hits require retained-full mode')
    if type(status_residual_mode) is not str or status_residual_mode not in ('diagnostic','kernel-event'):
        raise ValueError('explicit diagnostic or kernel-event status residual mode required')
    if status_residual_mode=='kernel-event' and (status_residual_compiled is None or not account_source_sites):
        raise ValueError('kernel-event status requires original helper and actual source-site accounting')
    if status_residual_compiled is not None:
        if isinstance(jnp.zeros((),jnp.int32),jax.core.Tracer):
            raise ValueError('checked status residual driver must remain outside outer JIT')
        if public_mode!='retained-full':
            raise ValueError('status residual diagnostics require explicit retained-full mode')
    if type(account_source_sites) is not bool:
        raise ValueError('explicit private source-site accounting mode required')
    if account_source_sites and (public_mode!='retained-full' or leftovers_mode!='retained' or leftovers_compiled is None):
        raise ValueError('source-site accounting requires retained-full and registered original Leftovers')
    if type(public_mode) is not str or public_mode not in ('basic', 'retained-full'):
        raise ValueError('explicit basic or retained-full public mode required')
    if type(leftovers_mode) is not str or leftovers_mode not in ('diagnostic', 'retained'):
        raise ValueError('explicit diagnostic or retained Leftovers mode required')
    if leftovers_mode == 'retained' and (public_mode != 'retained-full' or leftovers_compiled is None):
        raise ValueError('retained Leftovers requires retained-full and original compiled helper')
    record = own._checked(own_compiled)
    if record['request_compiled'] is not request_compiled or not callable(fake_forward):
        raise ValueError('same original request/OwnFacts producer and pure fake forward required')
    request = source._checked(request_compiled)
    catalog = record['catalog']
    leftovers_context=None;status_residual_context=None;direct_hit_context=None;substitute_context=None;toxic_context=None;recoil_context=None;calm_mind_context=None;swords_dance_context=None;spikes_context=None;thunder_wave_context=None;sleep_powder_context=None
    if thunder_wave_compiled is not None:
        from . import engine_thunder_wave_callbacks as thunder_wave_callbacks
        thunder_wave_data=thunder_wave_callbacks.check_compiled(thunder_wave_compiled)
        if thunder_wave_callbacks._COMPILED[thunder_wave_compiled][2] is not request['lifecycle']:raise ValueError('same original Thunder Wave/request engine catalog required')
    if sleep_powder_compiled is not None:
        from . import engine_sleep_powder_callbacks as sleep_powder_callbacks
        sleep_powder_data=sleep_powder_callbacks.check_compiled(sleep_powder_compiled)
        if sleep_powder_callbacks._COMPILED[sleep_powder_compiled][2] is not request['lifecycle']:raise ValueError('same original Sleep Powder/request engine catalog required')
        sleep_powder_callbacks.check_lexical_catalog(sleep_powder_compiled,catalog)
    if spikes_compiled is not None:
        from . import engine_spikes_callbacks as spikes_callbacks
        spikes_data=spikes_callbacks.check_compiled(spikes_compiled)
        if spikes_callbacks._COMPILED[spikes_compiled][2] is not request['lifecycle']:raise ValueError('same original Spikes/request engine catalog required')
    if recoil_compiled is not None:
        from . import engine_recoil_callbacks as recoil_callbacks
        recoil_data=recoil_callbacks.check_compiled(recoil_compiled)
        if recoil_callbacks._COMPILED[recoil_compiled][2] is not request['lifecycle']:
            raise ValueError('same original request engine catalog required for recoil caller')
    if calm_mind_compiled is not None:
        from . import engine_calm_mind_callbacks as calm_callbacks
        calm_data=calm_callbacks.check_compiled(calm_mind_compiled)
        if calm_callbacks._COMPILED[calm_mind_compiled][2] is not request['lifecycle']:raise ValueError('same original Calm Mind/request engine catalog required')
    if swords_dance_compiled is not None:
        from . import engine_swords_dance_callbacks as swords_callbacks
        swords_data=swords_callbacks.check_compiled(swords_dance_compiled)
        if swords_callbacks._COMPILED[swords_dance_compiled][2] is not request['lifecycle']:raise ValueError('same original Swords Dance/request engine catalog required')
    if toxic_compiled is not None:
        from . import engine_toxic_callbacks as toxic_callbacks
        toxic_data=toxic_callbacks.check_compiled(toxic_compiled)
        if toxic_callbacks._COMPILED[toxic_compiled][2] is not request['lifecycle']:
            raise ValueError('same original request engine catalog required for Toxic caller')
        toxic_context=toxic_callbacks.Context(toxic_data)
    if substitute_compiled is not None:
        from . import engine_substitute_callbacks as substitute_callbacks
        substitute_data=substitute_callbacks.check_compiled(substitute_compiled)
        if substitute_callbacks._COMPILED[substitute_compiled][2] is not request['lifecycle']:
            raise ValueError('same original request engine catalog required for Substitute caller')
        substitute_context=substitute_callbacks.Context(substitute_data)
    if direct_hit_compiled is not None:
        from . import engine_direct_hit_controls as direct_hit_controls
        direct_hit_data=direct_hit_controls.check_compiled(direct_hit_compiled)
        if direct_hit_controls._COMPILED[direct_hit_compiled][2] is not request['lifecycle']:
            raise ValueError('same original request engine catalog required for direct-hit caller')
        direct_hit_context=direct_hit_controls.Context(direct_hit_data)
    if status_residual_compiled is not None:
        from . import engine_status_residual_callbacks as status_residual_callbacks
        status_data=status_residual_callbacks.check_compiled(status_residual_compiled)
        if status_residual_callbacks._COMPILED[status_residual_compiled][4] is not request['lifecycle']:
            raise ValueError('same original request engine catalog required for status residual caller')
        status_residual_context=life.StatusResidualContext(status_data)
    if leftovers_compiled is not None:
        from . import engine_leftovers_callbacks as leftovers_callbacks
        data=leftovers_callbacks.check_compiled(leftovers_compiled)
        if leftovers_callbacks._COMPILED[leftovers_compiled][4] is not request['lifecycle']:
            raise ValueError('same original request engine catalog required for Leftovers caller')
        leftovers_context=life.LeftoversContext(data)
    base_handle = base.compile_projector(request['lifecycle'], expected_binding=request['lifecycle'].binding,
        expected_engine_binding=request['rules'].device.engine_binding)
    context = public_context.compile_context(catalog.source, catalog.abilities, request['lifecycle'].public_moves,
        items=catalog.items, field_catalog=catalog.field, damage_tables=catalog.damage)
    public_handle = public.compile_projector(json.loads(request['metadata']), base_handle,
        species_catalog=catalog.species, source=catalog.source, context_tables=context)
    check_public, register_public = public.check_projector, public.register
    public_fast = lambda obs, frame, profile: _public(obs, frame, public_handle.device, context, catalog)
    speed_profile=None;execution_compiled=None;no_output_compiled=None;kernel_risk_compiled=None
    if public_mode == 'retained-full':
        from . import engine_full_v4_event_device as full
        public_handle = full.compile_adapter(json.loads(request['metadata']), base_handle, catalog=catalog)
        check_public, register_public = full.check_adapter, full.register
        speed_profile=weather_speed.compile_profile(json.loads(request['metadata']),catalog=catalog,
            engine_catalog=request['lifecycle'],rules=request['rules'])
        if status_residual_mode=='kernel-event':
            status_residual_context=status_residual_callbacks.RegisteredStatusResidualContext(
                status_data,speed_profile.device)
        if leftovers_mode == 'retained':
            leftovers_context=leftovers_callbacks.RegisteredLeftoversContext(data,speed_profile.device)
        if account_source_sites:
            from . import engine_leftovers_no_output as no_output
            execution_compiled=execution.compile_accounting(public_handle)
            no_output_compiled=no_output.compile_no_output(leftovers_compiled,
                profile=speed_profile,accounting_compiled=execution_compiled)
            leftovers_context=leftovers_context._replace(no_output_tables=no_output_compiled.device)
            if track_kernel_risks:
                kernel_risk_compiled=kernel_risks.compile_risks(request['lifecycle'],execution_compiled,
                    direct_hit_compiled=direct_hit_compiled if direct_hit_mode=='kernel-event' else None,
                    substitute_compiled=substitute_compiled if substitute_mode=='kernel-event' else None,
                    toxic_compiled=toxic_compiled if toxic_mode=='kernel-event' else None,
                    recoil_compiled=recoil_compiled if recoil_mode=='kernel-event' else None,
                    calm_mind_compiled=calm_mind_compiled if calm_mind_mode=='kernel-event' else None,
                    swords_dance_compiled=swords_dance_compiled if swords_dance_mode=='kernel-event' else None,
                    spikes_compiled=spikes_compiled if spikes_mode=='kernel-event' else None,
                    thunder_wave_compiled=thunder_wave_compiled if thunder_wave_mode=='kernel-event' else None,sleep_powder_compiled=sleep_powder_compiled if sleep_powder_mode=='kernel-event' else None)
            if calm_mind_mode=='kernel-event':calm_mind_context=calm_callbacks.Context(calm_data,execution_compiled.device)
            if swords_dance_mode=='kernel-event':swords_dance_context=swords_callbacks.Context(swords_data,execution_compiled.device)
            if spikes_mode=='kernel-event':spikes_context=spikes_callbacks.Context(spikes_data,execution_compiled.device)
            if thunder_wave_mode=='kernel-event':thunder_wave_context=thunder_wave_callbacks.Context(thunder_wave_data,execution_compiled.device)
            if sleep_powder_mode=='kernel-event':sleep_powder_context=sleep_powder_callbacks.Context(sleep_powder_data,execution_compiled.device)
            if toxic_mode=='kernel-event':
                toxic_context=toxic_callbacks.RegisteredContext(toxic_data,execution_compiled.device)
            if substitute_mode=='kernel-event':
                substitute_context=substitute_callbacks.RegisteredContext(substitute_data,execution_compiled.device)
            if direct_hit_mode=='kernel-event':
                direct_hit_context=direct_hit_controls.RegisteredContext(direct_hit_data,execution_compiled.device)
            if recoil_mode=='kernel-event':
                recoil_context=recoil_callbacks.Context(recoil_data,direct_hit_context)
        public_fast = lambda obs, frame, profile, active=None: _full_public(obs, frame, public_handle.device,
            catalog, full.project_and_fold,weather_hp=(profile,speed_profile.device),
            leftovers=None if leftovers_mode == 'diagnostic' else leftovers_context.data,active=active,
            execution_tables=None if execution_compiled is None else execution_compiled.device,
            status_residual=None if status_residual_mode=='diagnostic' else status_residual_context.data,
            leftovers_no_output=None if no_output_compiled is None else no_output_compiled.device)
    static = (request['rules'].device, request['adapter'].device,
        team._checked_tables(request['private_team']), request['catalog'].device, request['engine_table'])
    payload = dict(schema=SCHEMA, metadata=json.loads(request['metadata']), lifecycle=request['lifecycle'].binding,
        request=np.asarray(request_compiled.binding).tolist(), own=np.asarray(own_compiled.device.binding).tolist(),
        encoder=np.asarray(catalog.binding).tolist(), forward='FAKE_ONLY', public_mode=public_mode,
        weather_speed_profile=None if speed_profile is None else np.asarray(speed_profile.device.binding).tolist())
    if leftovers_context is not None:
        payload['leftovers_binding']=np.asarray(leftovers_context.data.binding).tolist()
        payload['leftovers_mode']=leftovers_mode
    if execution_compiled is not None:payload['execution_inventory']=np.asarray(execution_compiled.device.inventory_binding).tolist()
    if no_output_compiled is not None:payload['leftovers_no_output_binding']=np.asarray(no_output_compiled.device.binding).tolist()
    if kernel_risk_compiled is not None:payload['kernel_risk_binding']=np.asarray(kernel_risk_compiled.device.binding).tolist()
    if status_residual_context is not None:
        payload['status_residual_binding']=np.asarray(status_residual_context.data.binding).tolist()
        payload['status_residual_mode']=status_residual_mode
    if direct_hit_context is not None:
        payload['direct_hit_binding']=np.asarray(direct_hit_context.data.binding).tolist()
        payload['direct_hit_mode']=direct_hit_mode
    if toxic_context is not None:
        payload['toxic_binding']=np.asarray(toxic_context.data.binding).tolist();payload['toxic_mode']=toxic_mode
    if calm_mind_context is not None:
        payload['calm_mind_binding']=np.asarray(calm_mind_context.data.binding).tolist();payload['calm_mind_mode']=calm_mind_mode
    if swords_dance_context is not None:
        payload['swords_dance_binding']=np.asarray(swords_dance_context.data.binding).tolist();payload['swords_dance_mode']=swords_dance_mode
    if substitute_context is not None:
        payload['substitute_binding']=np.asarray(substitute_context.data.binding).tolist()
        payload['substitute_mode']=substitute_mode
    if recoil_context is not None:
        payload['recoil_binding']=np.asarray(recoil_context.data.binding).tolist();payload['recoil_mode']=recoil_mode
    if spikes_context is not None:
        payload['spikes_binding']=np.asarray(spikes_context.data.binding).tolist();payload['spikes_mode']=spikes_mode
    if thunder_wave_context is not None:
        payload['thunder_wave_binding']=np.asarray(thunder_wave_context.data.binding).tolist();payload['thunder_wave_mode']=thunder_wave_mode
    if sleep_powder_context is not None:
        payload['sleep_powder_binding']=np.asarray(sleep_powder_context.data.binding).tolist();payload['sleep_powder_mode']=sleep_powder_mode
    if frame_input_mode=='source-witness':
        from . import engine_source_frame_witness as frame_witness
        payload['frame_input_mode']=frame_input_mode
        payload['frame_witness_source']=hashlib.sha256(Path(frame_witness.__file__).read_bytes()).hexdigest()
    binding = hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    result = Prepared(binding)
    request_type_proofs=None;request_encore_proofs=None
    leech_history_proof=None
    if direct_hit_context is not None and direct_hit_context.data.leech_seed is not None and direct_hit_mode=='kernel-event':
        from . import engine_leech_seed_callbacks as leech_callbacks
        leech_history_proof=_prepare_leech_history_proof(leech_callbacks.history_from_public,
            direct_hit_context,direct_hit_context.data.leech_seed,catalog.active.device)
    if direct_hit_context is not None and direct_hit_context.data.color_change is not None and direct_hit_mode=='kernel-event':
        from . import engine_color_change_callbacks as color
        request_type_proofs=_prepare_request_proof_pair(color.expected_owned_types,
            _context=direct_hit_context,
            data=direct_hit_context.data.color_change,categories=public_handle.device.causal.color_categories,
            type_tables=catalog.types.device)
    if direct_hit_context is not None and direct_hit_context.data.encore is not None and direct_hit_mode=='kernel-event':
        from . import engine_encore_callbacks as encore_callbacks
        request_encore_proofs=_prepare_request_proof_pair(encore_callbacks.expected_request,
            _context=direct_hit_context,
            data=direct_hit_context.data.encore,active_tables=catalog.active.device,rules=request['rules'].device)
    _PREPARED[result] = dict(binding=binding, prepared=weakref.ref(result),request_compiled=request_compiled, own_compiled=own_compiled,
        request_type_proofs=request_type_proofs,request_encore_proofs=request_encore_proofs,
        leech_history_proof=leech_history_proof,
        own=record, request=request, catalog=catalog, context=context, public=public_handle, fake=fake_forward,
        public_mode=public_mode, check_public=check_public, register_public=register_public,
        speed_profile=speed_profile,leftovers_compiled=leftovers_compiled,
        leftovers_mode=leftovers_mode,leftovers_context=leftovers_context,execution_compiled=execution_compiled,
        no_output_compiled=no_output_compiled,
        kernel_risk_compiled=kernel_risk_compiled,
        frame_input_mode=frame_input_mode,frame_witness_compiled=None,frame_witness_fast=None,
        direct_hit_compiled=direct_hit_compiled,direct_hit_context=direct_hit_context,direct_hit_mode=direct_hit_mode,
        substitute_compiled=substitute_compiled,substitute_context=substitute_context,substitute_mode=substitute_mode,
        toxic_compiled=toxic_compiled,toxic_context=toxic_context,toxic_mode=toxic_mode,
        recoil_compiled=recoil_compiled,recoil_context=recoil_context,recoil_mode=recoil_mode,
        calm_mind_compiled=calm_mind_compiled,calm_mind_context=calm_mind_context,calm_mind_mode=calm_mind_mode,
        swords_dance_compiled=swords_dance_compiled,swords_dance_context=swords_dance_context,swords_dance_mode=swords_dance_mode,
        spikes_compiled=spikes_compiled,spikes_context=spikes_context,spikes_mode=spikes_mode,
        thunder_wave_compiled=thunder_wave_compiled,thunder_wave_context=thunder_wave_context,thunder_wave_mode=thunder_wave_mode,
        sleep_powder_compiled=sleep_powder_compiled,sleep_powder_context=sleep_powder_context,sleep_powder_mode=sleep_powder_mode,
        status_residual_compiled=status_residual_compiled,status_residual_context=status_residual_context,
        status_residual_mode=status_residual_mode,
        retain_speed=None if speed_profile is None else jax.jit(lambda profile,state,streams:
            weather_speed.retain_device(profile,state,streams,speed_profile.device)),
        frame=jax.jit(lambda core, act, chance, active=None,leech_history=None,baton_public=None:
            _frame(core, (None, base_handle.device, request['engine_table']), act, chance, active,
                weather_profile_data=None if speed_profile is None else speed_profile.device,
                leftovers_context=leftovers_context,status_residual_context=status_residual_context,
                execution_tables=None if execution_compiled is None else execution_compiled.device,
                kernel_risk_tables=None if kernel_risk_compiled is None else kernel_risk_compiled.device,
                direct_hit_context=direct_hit_context,substitute_context=substitute_context,toxic_context=toxic_context,recoil_context=recoil_context,calm_mind_context=calm_mind_context,swords_dance_context=swords_dance_context,spikes_context=spikes_context,thunder_wave_context=thunder_wave_context,sleep_powder_context=sleep_powder_context,
                terminal_observer=frame_input_mode=='source-witness' and direct_hit_mode=='kernel-event',leech_history=leech_history,baton_public=baton_public)),
        request_fast=jax.jit(lambda engine, ledger, original, retained, unsafe, arrival, admitted,current_types=None,current_encore=None:
            _request(engine, ledger, original, retained, unsafe, arrival, admitted, static,current_types,current_encore)),
        installations=jax.jit(life.apply_installations),
        unsafe=jax.jit(lambda unsafe, state: unsafe | source._unsafe_history(state, request['rules'].device)),
        public_fast=jax.jit(public_fast),
        encode_fast=jax.jit(lambda core, obs: _own_encode(core, obs, catalog, own_compiled.device,mask_source=account_source_sites)),
        join_execution=None if execution_compiled is None else jax.jit(lambda first,second:
            execution.join_consumed_owners(first,second,tables=execution_compiled.device)),
        score_fast=jax.jit(lambda obs, facts, encoded, projection, error, admitted, active=None:
            _score(obs, facts, encoded, projection, error, admitted, fake_forward, active)))
    if frame_input_mode=='source-witness':
        # The base Prepared registry must exist before the witness compiler
        # checks its original edges. Keep _checked base-only: no recursion.
        compiled=frame_witness.compile_witness(result)
        _PREPARED[result]['frame_witness_compiled']=compiled
        _PREPARED[result]['frame_witness_fast']=jax.jit(lambda carry,core,frame,streams,publics,active:
            frame_witness.consume_pair_device(carry,core,frame,streams,publics,
                compiled.device,catalog,active=active))
    return result


def _checked(prepared):
    with v4_encoder._driver_guard_pass():
        r = _PREPARED.get(prepared) if type(prepared) is Prepared else None
        if r is None or prepared.binding is not r['binding']:
            raise ValueError('original prepared driver required')
        with _guard_span('own'):
            own._checked(r['own_compiled'])
        with _guard_span('request_source'):
            source._checked(r['request_compiled'])
        with _guard_span('public'):
            r['check_public'](r['public'])
        if r['execution_compiled'] is not None:
            with _guard_span('accounting'):
                execution._checked(r['execution_compiled'])
        if r['no_output_compiled'] is not None:
            from . import engine_leftovers_no_output as no_output
            with _guard_span('no_output'):
                no_output.check_compiled(r['no_output_compiled'])
        if r['kernel_risk_compiled'] is not None:
            with _guard_span('risk'):
                kernel_risks.check_compiled(r['kernel_risk_compiled'])
        if r['calm_mind_compiled'] is not None:
            from . import engine_calm_mind_callbacks as calm_callbacks
            with _guard_span('calm_mind'):
                calm_callbacks.check_compiled(r['calm_mind_compiled'])
        if r['swords_dance_compiled'] is not None:
            from . import engine_swords_dance_callbacks as swords_callbacks
            with _guard_span('swords_dance'):
                swords_callbacks.check_compiled(r['swords_dance_compiled'])
        if r['recoil_compiled'] is not None:
            from . import engine_recoil_callbacks as recoil_callbacks
            with _guard_span('recoil'):
                recoil_callbacks.check_compiled(r['recoil_compiled'])
        if r['toxic_compiled'] is not None:
            from . import engine_toxic_callbacks as toxic_callbacks
            with _guard_span('toxic'):
                toxic_callbacks.check_compiled(r['toxic_compiled'])
        if r['substitute_compiled'] is not None:
            from . import engine_substitute_callbacks as substitute_callbacks
            with _guard_span('substitute'):
                substitute_callbacks.check_compiled(r['substitute_compiled'])
        if r['direct_hit_compiled'] is not None:
            from . import engine_direct_hit_controls as direct_hit_controls
            with _guard_span('direct_hit'):
                direct_hit_controls.check_compiled(r['direct_hit_compiled'])
        if r['spikes_compiled'] is not None:
            from . import engine_spikes_callbacks as spikes_callbacks
            with _guard_span('spikes'):
                spikes_callbacks.check_compiled(r['spikes_compiled'])
        if r['thunder_wave_compiled'] is not None:
            from . import engine_thunder_wave_callbacks as thunder_wave_callbacks
            with _guard_span('thunder_wave'):
                thunder_wave_callbacks.check_compiled(r['thunder_wave_compiled'])
        if r['sleep_powder_compiled'] is not None:
            from . import engine_sleep_powder_callbacks as sleep_powder_callbacks
            with _guard_span('sleep_powder'):
                sleep_powder_callbacks.check_compiled(r['sleep_powder_compiled'])
        if r['speed_profile'] is not None:
            with _guard_span('speed_profile'):
                weather_speed.check_profile(r['speed_profile'])
        if r['leftovers_compiled'] is not None:
            from . import engine_leftovers_callbacks as leftovers_callbacks
            with _guard_span('leftovers'):
                leftovers_callbacks.check_compiled(r['leftovers_compiled'])
        if r['status_residual_compiled'] is not None:
            from . import engine_status_residual_callbacks as status_residual_callbacks
            with _guard_span('status_residual'):
                status_residual_callbacks.check_compiled(r['status_residual_compiled'])
        return r

@contextmanager
def _host_span(timings, name):
    """Read-only host attribution, separate from the original phase callback."""
    if timings is None:
        yield
        return
    start = time.perf_counter_ns()
    scope = _HostGuardTimingScope(timings,name) if name in ('initial_guards_ms','extra_guards_ms') else None
    token = _HOST_GUARD_TIMING.set(scope)
    try:
        yield
    finally:
        if scope is not None:scope.active = False
        try:
            timings[name] = timings.get(name, 0.) + (time.perf_counter_ns()-start)/1e6
        finally:
            _HOST_GUARD_TIMING.reset(token)


@contextmanager
def _guard_span(name):
    """Overlapping actual host call cost, only inside an enabled guard span.

    Absent keys mean unexecuted/unmeasured. Counts include entered calls that
    raise; timings never cache validity or influence the original result.
    """
    scope = _HOST_GUARD_TIMING.get()
    if scope is None or not scope.active:
        yield
        return
    timings,group = scope.timings,scope.group
    prefix = 'guard_call.'+group+'.'+name
    start = time.perf_counter_ns()
    try:
        yield
    finally:
        if scope.active:
            elapsed = (time.perf_counter_ns()-start)/1e6
            if scope.active:
                timings[prefix+'_calls'] = timings.get(prefix+'_calls',0) + 1
                timings[prefix+'_ms'] = timings.get(prefix+'_ms',0.) + elapsed


def _timed(callback, name, function, *args, _host_timings=None):
    if callback is None:
        with compile_observations.phase('driver.'+name):
            return function(*args)
    with _host_span(_host_timings, f'input_block_wait.{name}_ms'):
        jax.block_until_ready(args)
    start = time.perf_counter_ns()
    with compile_observations.phase('driver.'+name):
        result = function(*args)
        jax.block_until_ready(result)
    callback(name, (time.perf_counter_ns()-start)/1e6)
    return result


def _checked_witness(r, *, constructed=None, session=None):
    """Separate outer guard; the witness's base driver check cannot recurse."""
    compiled=r['frame_witness_compiled']
    if compiled is not None:
        from . import engine_source_frame_witness as frame_witness
        if constructed is None and session is None:
            with _guard_span('witness_compiled'):
                frame_witness.check_compiled(compiled)
        else:
            with _guard_span('witness_host_origin'):
                frame_witness.check_host_origin(compiled,constructed=constructed,session=session)


def _consume_accounted(r,state,frame,active,timing_callback,previous_reports,host_timings=None):
    # BOTH original full folds finish before any own encoding or fake scoring.
    candidates=[];projections=[];errors=[]
    if r['frame_witness_compiled'] is not None:
        with _host_span(host_timings, 'extra_guards_ms'):
            _checked_witness(r)
        paired=_timed(timing_callback,'public_pair',r['frame_witness_fast'],state.core.frame_inputs,
            state.core,frame,tuple(x.projection for x in state.observers),
            tuple(x.retained for x in state.observers),active, _host_timings=host_timings)
        with _host_span(host_timings, 'extra_guards_ms'):
            _checked_witness(r)
        joint=paired.report.frame_inputs_complete
        joined=paired.ledger
        evidence=paired.report if state.core.frame_input_report is None else _rows(
            active,state.core.frame_input_report,paired.report)
        core=state.core._replace(execution=joined,admitted=state.core.admitted & (~active|joint),
            frame_inputs=paired.carry,frame_input_report=evidence,
            terminal_closeout=None if state.core.terminal_closeout is None else jnp.where(
                active,paired.terminal_closeout,state.core.terminal_closeout))
        for before,consumed in zip(state.observers,paired.owners):
            projection=consumed.projection._replace(full_line_consumers_complete=joint)
            candidate=before._replace(projection=projection.state,retained=consumed.public,
                retained_healthy=before.retained_healthy & (~active|joint),accounting=joined)
            candidates.append(_observer_rows(active,before,candidate))
            projections.append(projection);errors.append(projection.error)
    else:
        for side,before in enumerate(state.observers,1):
            with _host_span(host_timings, 'extra_guards_ms'):
                _checked(r['prepared']())
            def public_phase(observer,frame,active):
                after,projection,error=r['public_fast'](observer,frame,state.core.weather_profile,active)
                return _observer_rows(active,observer,after),projection,error
            candidate,projection,error=_timed(timing_callback,f'public_p{side}',public_phase,before,frame,active, _host_timings=host_timings)
            candidates.append(candidate);projections.append(projection);errors.append(error)
        with _host_span(host_timings, 'extra_guards_ms'):
            _checked(r['prepared']())
        joined,joint=r['join_execution'](candidates[0].accounting,candidates[1].accounting)
        ledger=execution.commit_rows(active,state.core.execution,joined)
        core=state.core._replace(execution=ledger,admitted=state.core.admitted & (~active|joint))
    observers=[];reports=[]
    for index,(before,candidate,projection,error) in enumerate(zip(state.observers,candidates,projections,errors)):
        observer=_observer_rows(active & joint,before,candidate)
        observer=observer._replace(accounting=execution.commit_rows(active,before.accounting,joined),
            retained_healthy=observer.retained_healthy & (~active|joint))
        with _host_span(host_timings, 'extra_guards_ms'):
            _checked(r['prepared']())
        facts,encoded=_timed(timing_callback,f'encode_p{index+1}',r['encode_fast'],core,observer, _host_timings=host_timings)
        def admission_phase(obs,facts,encoded,projection,error,admitted,active,previous):
            after,report=r['score_fast'](obs,facts,encoded,projection,error,admitted,active)
            if previous is not None:report=_report_rows(active,previous,report)
            return after,report
        previous=None if previous_reports is None else previous_reports[index]
        with _host_span(host_timings, 'extra_guards_ms'):
            _checked(r['prepared']())
        observer,report=_timed(timing_callback,f'admission_p{index+1}',admission_phase,
            observer,facts,encoded,projection,error,core.admitted,active,previous, _host_timings=host_timings)
        observers.append(observer);reports.append(report)
    if core.baton_history is not None and frame.callbacks.execution is not None and frame.callbacks.execution.baton_transaction is not None:
        from . import engine_baton_pass_callbacks as baton_callbacks
        core=core._replace(baton_history=_rows(active,core.baton_history,baton_callbacks.mature(core.baton_history,frame,tuple(o.accounting for o in observers),tuple(o.retained for o in observers),core.requests)))
    return Result(DriverState(core,tuple(observers)),tuple(reports))


def _consume(r, state, frame, active_rows=None, timing_callback=None, previous_reports=None,host_timings=None):
    if active_rows is None:
        active_rows = jnp.ones_like(state.core.admitted)
    if r['execution_compiled'] is not None:
        return _consume_accounted(r,state,frame,active_rows,timing_callback,previous_reports,host_timings)
    observers, reports = [], []
    for side, observer in enumerate(state.observers, 1):
        def public_phase(before, frame, active):
            candidate, projection, error = r['public_fast'](before, frame,state.core.weather_profile)
            return _observer_rows(active, before, candidate), projection, error
        observer, projection, error = _timed(timing_callback, f'public_p{side}', public_phase,
            observer, frame, active_rows, _host_timings=host_timings)
        facts, encoded = _timed(timing_callback, f'encode_p{side}', r['encode_fast'], state.core, observer, _host_timings=host_timings)
        def admission_phase(obs, facts, encoded, projection, error, admitted, active, previous):
            candidate, report = r['score_fast'](obs, facts, encoded, projection, error, admitted, active)
            if previous is not None:
                report = _report_rows(active, previous, report)
            return candidate, report
        previous = None if previous_reports is None else previous_reports[side-1]
        observer, report = _timed(timing_callback, f'admission_p{side}', admission_phase, observer,
            facts, encoded, projection, error, state.core.admitted, active_rows, previous, _host_timings=host_timings)
        observers.append(observer); reports.append(report)
    return Result(state._replace(observers=tuple(observers)), tuple(reports))


def construct(pool, team_indices, *, prepared, stream_ids, seed=0, gender_initialization=None):
    """Actual constructor once, then consume its genuine initial frame once."""
    r = _checked(prepared)
    _checked_witness(r)
    snapshot = source.construct_from_pool(pool, team_indices, compiled=r['request_compiled'],
        stream_ids=stream_ids, seed=seed, gender_initialization=gender_initialization,
        _execution_compiled=r['execution_compiled'],_kernel_risk_compiled=r['kernel_risk_compiled'])
    b = snapshot.state.hp.shape[0]
    new = life.register(stream_ids, catalog=r['request']['lifecycle'], expected_binding=r['request']['lifecycle'].binding,
        expected_engine_binding=r['request']['rules'].device.engine_binding)
    frame = life.LifecycleResult(snapshot.state, snapshot.cursor, snapshot.callbacks, life.snapshot(new),
        snapshot.registry, jnp.zeros(b, bool), jnp.zeros(b, jnp.int32))
    admitted = snapshot.cursor.admitted & ((snapshot.callbacks.refusal & ~events.UNINSTRUMENTED) == 0)
    # Only this actual constructor callsite mints the birth record. The sealed
    # pre-entry state retains unsafe items even if later callbacks consume them.
    profile=None
    if r['speed_profile'] is not None:
        if snapshot.birth_state is None:raise ValueError('actual pre-entry birth state required')
        profile=weather_speed.qualify_birth(snapshot.birth_state,stream_ids,profile=r['speed_profile'])
        profile=weather_speed.retain_frame_device(profile,snapshot.state,stream_ids,r['speed_profile'].device,snapshot.callbacks)
    core = Core(snapshot.state, snapshot.cursor, snapshot.orders,
        tuple(reg.original for reg in snapshot.registrations),
        tuple(actions._constructor(reg.constructor) for reg in snapshot.registrations),
        tuple(reg.unsafe_origin for reg in snapshot.registrations), snapshot.requests, admitted, jnp.zeros(b, jnp.int32),profile,
        None if snapshot.callbacks.execution is None else snapshot.callbacks.execution.ledger,
        None if snapshot.callbacks.execution is None or snapshot.callbacks.execution.risks is None
            else snapshot.callbacks.execution.risks.carry)
    if r['direct_hit_mode']=='kernel-event' and r['execution_compiled'] is not None and r['direct_hit_context'] is not None and r['direct_hit_context'].data.baton_pass is not None:
        from . import engine_baton_pass_callbacks as baton_callbacks
        core=core._replace(baton_history=baton_callbacks.empty_history(core.engine,core.cursor,jnp.stack(tuple(o.request_to_physical for o in core.orders),axis=1)))
    if r['frame_witness_compiled'] is not None:
        from . import engine_source_frame_witness as frame_witness
        _checked_witness(r,constructed=snapshot)
        core=core._replace(frame_inputs=frame_witness.initialize_from_constructed(
            r['frame_witness_compiled'],snapshot),terminal_closeout=jnp.zeros(b,bool))
    observers = []
    for side in (0, 1):
        owners = jnp.full((b,), side, jnp.int32)
        observers.append(Observer(owners, r['register_public'](stream_ids, owners, compiled=r['public']),
            _basic_initialize(r['catalog'], r['context'], stream_ids),
            v4_encoder.initialize(r['catalog'], stream_ids=stream_ids, owner_sides=owners),
            jnp.ones(b, bool), jnp.zeros(b, jnp.int32),core.execution))
    result = _consume(r, DriverState(core, tuple(observers)), frame)
    session = Session(prepared, result.state, result)
    _SESSIONS[session] = dict(prepared=prepared, state=result.state, result=result, constructed=snapshot,
        advancing=False, diagnostics=FrameDiagnostics(jnp.zeros(b, bool), jnp.zeros(b, jnp.int32),
            jnp.zeros(b, bool), jnp.zeros(b, bool)))
    return session


def _session(session):
    record = _SESSIONS.get(session) if type(session) is Session else None
    if (record is None or session.prepared is not record['prepared'] or session.state is not record['state']
            or session.result is not record['result']):
        raise ValueError('original current driver session required')
    return record


def terminal_complete(core):
    """Literal final source receipt + BOTH full drains; no next request needed.

    A terminal OwnFacts/encoder refusal is expected because no decision follows.
    It cannot stand in for this closeout or waive failed public/accounting/risk.
    """
    value=getattr(core,'terminal_closeout',None)
    if value is None:return jnp.zeros_like(core.engine.done)
    execution._shape(value,core.engine.done.shape,jnp.bool_)
    return value & core.engine.done & core.admitted


def latest_diagnostics(session):
    record = _session(session)
    _checked(session.prepared)
    return record['diagnostics']


def advance(session, engine_actions, *, active_rows=None, timing_callback=None, chance_inputs=None, collect_timings=False, return_diagnostics=False):
    """Source component advance; never a model rollout or implicit CLI enable.

    The API takes no external frame: its sole core result is consumed here once.
    A host Session retains one current carry; cached kernels cannot bypass the
    original static guards or let a replaced/stale carry resume silently.
    collect_timings records the original synchronized phase intervals internally;
    it cannot be combined with an arbitrary external timing observer.
    """
    if type(return_diagnostics) is not bool:
        raise ValueError('return_diagnostics must be an explicit bool')
    if type(collect_timings) is not bool:
        raise ValueError('collect_timings must be an explicit bool')
    if collect_timings and timing_callback is not None:
        raise ValueError('built-in timing and external timing callback are mutually exclusive')
    host_timings = {} if collect_timings or timing_callback is not None else None
    advance_start = time.perf_counter_ns() if host_timings is not None else None
    with _host_span(host_timings, 'initial_guards_ms'):
        record = _session(session)
        if record['advancing']:
            raise ValueError('driver advance is not reentrant')
        if isinstance(engine_actions, jax.core.Tracer) or isinstance(active_rows, jax.core.Tracer):
            raise ValueError('checked host driver cannot be wrapped in an outer JIT')
        if timing_callback is not None and not callable(timing_callback):
            raise ValueError('host timing callback must be callable')
        r = _checked(session.prepared)
        source._constructed(record['constructed'])
        _checked_witness(r,session=session)
    core = session.state.core
    b = core.frame_number.shape[0]
    if active_rows is None:
        active = jnp.ones(b, bool)
    else:
        if getattr(active_rows, 'shape', None) != (b,) or getattr(active_rows, 'dtype', None) != jnp.dtype(bool):
            raise ValueError('exact bool[B] active rows required')
        active = jnp.asarray(active_rows)
    executed = active & core.admitted & ~core.engine.done
    for request in core.requests:
        executed &= request.actions.admitted
    for observer in session.state.observers:
        if r['public_mode'] == 'basic':
            executed &= observer.projection.admitted
        else:
            executed &= observer.projection.causal.admitted & observer.retained_healthy
    old_state, old_result = session.state, session.result
    record['advancing'] = True
    try:
        def checked_timing(name, milliseconds):
            with _host_span(host_timings, 'timing_observer_ms'):
                timing_callback(name, milliseconds)
            # A host observer runs between cached phases. Revalidate at that
            # boundary so a mutated static holder cannot reach the next phase.
            with _host_span(host_timings, 'post_observer_guards_ms'):
                _session(session)
                _checked(session.prepared)
                source._constructed(record['constructed'])
                _checked_witness(r,session=session)
        def built_in_timing(name, milliseconds):
            # Original synchronized _timed interval, with no external observer.
            # Normal phase/source/session guards remain in their original sites.
            host_timings['phase.'+name+'_ms']=milliseconds
        observe = built_in_timing if collect_timings else (None if timing_callback is None else checked_timing)
        chance = transition.sample_chance_inputs(core.engine) if chance_inputs is None else chance_inputs
        leech_history=_leech_history(r,core,tuple(o.retained for o in old_state.observers),
            core.admitted & old_state.observers[0].retained_healthy & old_state.observers[1].retained_healthy)
        baton_public=None
        if r['direct_hit_mode']=='kernel-event' and r['execution_compiled'] is not None and r['direct_hit_context'] is not None and r['direct_hit_context'].data.baton_pass is not None:
            from . import engine_baton_pass_callbacks as baton_callbacks
            baton_public=baton_callbacks.public_proof(core.engine,core.cursor,tuple(o.retained for o in old_state.observers),
                core.admitted & old_state.observers[0].retained_healthy & old_state.observers[1].retained_healthy,r['direct_hit_context'].data.baton_pass)
        frame = _timed(observe, 'engine', r['frame'], core, engine_actions, chance, executed,leech_history,baton_public, _host_timings=host_timings)
        def requests_phase(core, frame):
            orders = tuple(r['installations'](ledger, frame) for ledger in core.orders)
            unsafe = tuple(r['unsafe'](value, frame.state) for value in core.unsafe)
            admitted = core.admitted & frame.cursor.admitted & ((frame.callbacks.refusal & ~events.UNINSTRUMENTED) == 0)
            current_types=(None,None)
            direct_context=r['direct_hit_context']
            if direct_context is not None and direct_context.data.color_change is not None and r['direct_hit_mode']=='kernel-event':
                from . import engine_color_change_callbacks as color
                proofs=_checked_request_proof_pair(r['request_type_proofs'],color.expected_owned_types,direct_context,
                    data=direct_context.data.color_change,categories=r['public'].device.causal.color_categories,
                    type_tables=r['catalog'].types.device)
                current_types=tuple(proof(frame.state,core.engine,observer.retained,frame.callbacks,observer.retained_healthy)
                    for proof,observer in zip(proofs,old_state.observers))
            current_encore=(None,None)
            if direct_context is not None and direct_context.data.encore is not None and r['direct_hit_mode']=='kernel-event':
                from . import engine_encore_callbacks as encore_callbacks
                proofs=_checked_request_proof_pair(r['request_encore_proofs'],encore_callbacks.expected_request,direct_context,
                    data=direct_context.data.encore,active_tables=r['catalog'].active.device,rules=r['request']['rules'].device)
                current_encore=tuple(proof(frame.state,core.engine,observer.retained,frame.callbacks,observer.retained_healthy)
                    for proof,observer in zip(proofs,old_state.observers))
            requests = tuple(r['request_fast'](frame.state, ledger, original, retained, history,
                frame.after_registry, admitted,types,encore) for ledger, original, retained, history,types,encore in
                zip(orders, core.originals, core.constructors, unsafe,current_types,current_encore))
            profile=core.weather_profile if r['retain_speed'] is None else r['retain_speed'](
                frame.weather_profile,frame.state,frame.cursor.stream_id)
            candidate = core._replace(engine=frame.state, cursor=frame.cursor, orders=orders, unsafe=unsafe,
                requests=requests, admitted=admitted, frame_number=core.frame_number+executed.astype(jnp.int32),
                baton_history=core.baton_history if frame.callbacks.execution is None or frame.callbacks.execution.baton_transaction is None else frame.callbacks.execution.baton_transaction.history,
                weather_profile=profile,execution=None if frame.callbacks.execution is None else frame.callbacks.execution.ledger)
            candidate=candidate._replace(risks=None if frame.callbacks.execution is None
                or frame.callbacks.execution.risks is None else frame.callbacks.execution.risks.carry)
            # Originals and constructor snapshots are immutable source constants.
            chosen = _core_rows(executed, core._replace(originals=None, constructors=None),
                candidate._replace(originals=None, constructors=None))
            return chosen._replace(originals=core.originals, constructors=core.constructors)
        if r['execution_compiled'] is not None:
            with _host_span(host_timings, 'extra_guards_ms'):
                _checked(session.prepared)
        next_core = _timed(observe, 'requests', requests_phase, core, frame, _host_timings=host_timings)
        result = _consume(r, old_state._replace(core=next_core), frame, executed, observe, old_result.owners,host_timings)
        # Callbacks can observe timing but cannot replace or replay the carry.
        with _host_span(host_timings, 'final_session_guard_ms'):
            if _session(session) is not record or session.state is not old_state or session.result is not old_result:
                raise ValueError('timing callback changed the driver session')
        session.state, session.result = result.state, result
        if host_timings is not None:
            host_timings['advance_wall_ms'] = (time.perf_counter_ns()-advance_start)/1e6
        diagnostics=FrameDiagnostics(frame.invalid_actions, frame.placeholder_moves, active, executed,host_timings)
        record.update(state=result.state, result=result, diagnostics=diagnostics)
        return AdvanceWithDiagnostics(result, diagnostics) if return_diagnostics else result
    finally:
        record['advancing'] = False
