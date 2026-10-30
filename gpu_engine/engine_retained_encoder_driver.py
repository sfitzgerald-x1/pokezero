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
All current full/B4 flags remain false. Only a pure FAKE forward interface is
exposed for source integration tests.
"""
from dataclasses import dataclass
import hashlib
import json
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


def _observer_rows(ok, before, after):
    basic = _public_rows(ok, before.basic, after.basic)
    retained = _public_rows(ok, before.retained, after.retained)
    selected = _rows(ok, before._replace(basic=None, retained=None), after._replace(basic=None, retained=None))
    return selected._replace(basic=basic, retained=retained)


def _report_rows(ok, before, after):
    # These two leaves are schema-wide, including when B happens to equal8.
    def detach(report):
        return report._replace(encoded=report.encoded._replace(pending_numeric_columns=None,
            observation=report.encoded.observation._replace(category_count=None)))
    selected = _rows(ok, detach(before), detach(after))
    return selected._replace(encoded=selected.encoded._replace(
        pending_numeric_columns=before.encoded.pending_numeric_columns,
        observation=selected.encoded.observation._replace(category_count=before.encoded.observation.category_count)))


def _frame(core, statics, engine_actions, chance, active_rows=None):
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
    masked = core.engine._replace(done=core.engine.done | ~source_ready)
    result = transition.step_with_chance(masked, engine_actions, engine_table, chance,
        _public_trace=tape, _lifecycle_trace=True)
    engine = _rows(source_ready, core.engine, result.step.state)
    cursor = _rows(source_ready, core.cursor, result.callbacks.cursor)
    # Bindings are GLOBAL [8], never row-select them by a coincidental B=8.
    globals_ = {'engine_binding', 'source_binding', 'move_binding', 'producer_binding'}
    callbacks = result.callbacks._replace(**{name: _rows(source_ready, getattr(tape, name),
        getattr(result.callbacks, name)) for name in tape._fields if name not in globals_})
    callbacks = callbacks._replace(cursor=cursor)
    return life.LifecycleResult(engine, cursor, callbacks, before,
        life.snapshot(cursor), jnp.where(source_ready, result.step.invalid_actions, False),
        jnp.where(source_ready, result.step.placeholder_moves, 0))


def _request(engine, ledger, original, retained, unsafe, arrival, admitted, statics):
    """The existing source request + private adapter PURE lower primitives."""
    rules, adapter, private, action_tables, engine_table = statics
    packet, reason = source._produce(engine, ledger, arrival, original, retained, unsafe,
        rules, adapter.binding, engine_table)
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


def _full_public(observer, frame, tables, catalog, consume):
    """Consume genuine full-format controls without relabeling basic state."""
    result = consume(observer.projection, frame, tables, observer.retained, catalog)
    projection = result.projection
    healthy = observer.retained_healthy & (projection.error == 0) & projection.component_admitted
    return observer._replace(projection=projection.state, retained=result.public,
        retained_healthy=healthy), projection, projection.error


def _own_encode(core, observer, catalog, own_tables):
    facts = own._device(core.engine, core.orders, core.requests, observer.owner, own_tables)
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


def prepare(request_compiled, own_compiled, *, fake_forward, public_mode='basic'):
    """Host-only immutable preparation. fake_forward is a pure source-test spy.

    The returned compiled closures contain no registration calls or host reads.
    The checked host API reruns guards before every cached device invocation.
    The explicit retained-full mode consumes the full-format device adapter;
    its remaining coverage and scientific qualification flags stay unchanged.
    """
    if type(public_mode) is not str or public_mode not in ('basic', 'retained-full'):
        raise ValueError('explicit basic or retained-full public mode required')
    record = own._checked(own_compiled)
    if record['request_compiled'] is not request_compiled or not callable(fake_forward):
        raise ValueError('same original request/OwnFacts producer and pure fake forward required')
    request = source._checked(request_compiled)
    catalog = record['catalog']
    base_handle = base.compile_projector(request['lifecycle'], expected_binding=request['lifecycle'].binding,
        expected_engine_binding=request['rules'].device.engine_binding)
    context = public_context.compile_context(catalog.source, catalog.abilities, request['lifecycle'].public_moves,
        items=catalog.items, field_catalog=catalog.field, damage_tables=catalog.damage)
    public_handle = public.compile_projector(json.loads(request['metadata']), base_handle,
        species_catalog=catalog.species, source=catalog.source, context_tables=context)
    check_public, register_public = public.check_projector, public.register
    public_fast = lambda obs, frame: _public(obs, frame, public_handle.device, context, catalog)
    if public_mode == 'retained-full':
        from . import engine_full_v4_event_device as full
        public_handle = full.compile_adapter(json.loads(request['metadata']), base_handle, catalog=catalog)
        check_public, register_public = full.check_adapter, full.register
        public_fast = lambda obs, frame: _full_public(obs, frame, public_handle.device,
            catalog, full.project_and_fold)
    static = (request['rules'].device, request['adapter'].device,
        team._checked_tables(request['private_team']), request['catalog'].device, request['engine_table'])
    payload = dict(schema=SCHEMA, metadata=json.loads(request['metadata']), lifecycle=request['lifecycle'].binding,
        request=np.asarray(request_compiled.binding).tolist(), own=np.asarray(own_compiled.device.binding).tolist(),
        encoder=np.asarray(catalog.binding).tolist(), forward='FAKE_ONLY', public_mode=public_mode)
    binding = hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    result = Prepared(binding)
    _PREPARED[result] = dict(binding=binding, request_compiled=request_compiled, own_compiled=own_compiled,
        own=record, request=request, catalog=catalog, context=context, public=public_handle, fake=fake_forward,
        public_mode=public_mode, check_public=check_public, register_public=register_public,
        frame=jax.jit(lambda core, act, chance, active=None:
            _frame(core, (None, base_handle.device, request['engine_table']), act, chance, active)),
        request_fast=jax.jit(lambda engine, ledger, original, retained, unsafe, arrival, admitted:
            _request(engine, ledger, original, retained, unsafe, arrival, admitted, static)),
        installations=jax.jit(life.apply_installations),
        unsafe=jax.jit(lambda unsafe, state: unsafe | source._unsafe_history(state, request['rules'].device)),
        public_fast=jax.jit(public_fast),
        encode_fast=jax.jit(lambda core, obs: _own_encode(core, obs, catalog, own_compiled.device)),
        score_fast=jax.jit(lambda obs, facts, encoded, projection, error, admitted, active=None:
            _score(obs, facts, encoded, projection, error, admitted, fake_forward, active)))
    return result


def _checked(prepared):
    r = _PREPARED.get(prepared) if type(prepared) is Prepared else None
    if r is None or prepared.binding is not r['binding']:
        raise ValueError('original prepared driver required')
    own._checked(r['own_compiled'])
    source._checked(r['request_compiled'])
    r['check_public'](r['public'])
    return r


def _timed(callback, name, function, *args):
    if callback is None:
        return function(*args)
    jax.block_until_ready(args)
    start = time.perf_counter_ns()
    result = function(*args)
    jax.block_until_ready(result)
    callback(name, (time.perf_counter_ns()-start)/1e6)
    return result


def _consume(r, state, frame, active_rows=None, timing_callback=None, previous_reports=None):
    if active_rows is None:
        active_rows = jnp.ones_like(state.core.admitted)
    observers, reports = [], []
    for side, observer in enumerate(state.observers, 1):
        def public_phase(before, frame, active):
            candidate, projection, error = r['public_fast'](before, frame)
            return _observer_rows(active, before, candidate), projection, error
        observer, projection, error = _timed(timing_callback, f'public_p{side}', public_phase,
            observer, frame, active_rows)
        facts, encoded = _timed(timing_callback, f'encode_p{side}', r['encode_fast'], state.core, observer)
        def admission_phase(obs, facts, encoded, projection, error, admitted, active, previous):
            candidate, report = r['score_fast'](obs, facts, encoded, projection, error, admitted, active)
            if previous is not None:
                report = _report_rows(active, previous, report)
            return candidate, report
        previous = None if previous_reports is None else previous_reports[side-1]
        observer, report = _timed(timing_callback, f'admission_p{side}', admission_phase, observer,
            facts, encoded, projection, error, state.core.admitted, active_rows, previous)
        observers.append(observer); reports.append(report)
    return Result(state._replace(observers=tuple(observers)), tuple(reports))


def construct(pool, team_indices, *, prepared, stream_ids, seed=0, gender_initialization=None):
    """Actual constructor once, then consume its genuine initial frame once."""
    r = _checked(prepared)
    snapshot = source.construct_from_pool(pool, team_indices, compiled=r['request_compiled'],
        stream_ids=stream_ids, seed=seed, gender_initialization=gender_initialization)
    b = snapshot.state.hp.shape[0]
    new = life.register(stream_ids, catalog=r['request']['lifecycle'], expected_binding=r['request']['lifecycle'].binding,
        expected_engine_binding=r['request']['rules'].device.engine_binding)
    frame = life.LifecycleResult(snapshot.state, snapshot.cursor, snapshot.callbacks, life.snapshot(new),
        snapshot.registry, jnp.zeros(b, bool), jnp.zeros(b, jnp.int32))
    admitted = snapshot.cursor.admitted & ((snapshot.callbacks.refusal & ~events.UNINSTRUMENTED) == 0)
    core = Core(snapshot.state, snapshot.cursor, snapshot.orders,
        tuple(reg.original for reg in snapshot.registrations),
        tuple(actions._constructor(reg.constructor) for reg in snapshot.registrations),
        tuple(reg.unsafe_origin for reg in snapshot.registrations), snapshot.requests, admitted, jnp.zeros(b, jnp.int32))
    observers = []
    for side in (0, 1):
        owners = jnp.full((b,), side, jnp.int32)
        observers.append(Observer(owners, r['register_public'](stream_ids, owners, compiled=r['public']),
            _basic_initialize(r['catalog'], r['context'], stream_ids),
            v4_encoder.initialize(r['catalog'], stream_ids=stream_ids, owner_sides=owners),
            jnp.ones(b, bool), jnp.zeros(b, jnp.int32)))
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


def latest_diagnostics(session):
    record = _session(session)
    _checked(session.prepared)
    return record['diagnostics']


def advance(session, engine_actions, *, active_rows=None, timing_callback=None, chance_inputs=None):
    """Source component advance; never a model rollout or implicit CLI enable.

    The API takes no external frame: its sole core result is consumed here once.
    A host Session retains one current carry; cached kernels cannot bypass the
    original static guards or let a replaced/stale carry resume silently.
    """
    record = _session(session)
    if record['advancing']:
        raise ValueError('driver advance is not reentrant')
    if isinstance(engine_actions, jax.core.Tracer) or isinstance(active_rows, jax.core.Tracer):
        raise ValueError('checked host driver cannot be wrapped in an outer JIT')
    if timing_callback is not None and not callable(timing_callback):
        raise ValueError('host timing callback must be callable')
    r = _checked(session.prepared)
    source._constructed(record['constructed'])
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
            timing_callback(name, milliseconds)
            # A host observer runs between cached phases. Revalidate at that
            # boundary so a mutated static holder cannot reach the next phase.
            _session(session)
            _checked(session.prepared)
            source._constructed(record['constructed'])
        observe = None if timing_callback is None else checked_timing
        chance = transition.sample_chance_inputs(core.engine) if chance_inputs is None else chance_inputs
        frame = _timed(observe, 'engine', r['frame'], core, engine_actions, chance, executed)
        def requests_phase(core, frame):
            orders = tuple(r['installations'](ledger, frame) for ledger in core.orders)
            unsafe = tuple(r['unsafe'](value, frame.state) for value in core.unsafe)
            admitted = core.admitted & frame.cursor.admitted & ((frame.callbacks.refusal & ~events.UNINSTRUMENTED) == 0)
            requests = tuple(r['request_fast'](frame.state, ledger, original, retained, history,
                frame.after_registry, admitted) for ledger, original, retained, history in
                zip(orders, core.originals, core.constructors, unsafe))
            candidate = core._replace(engine=frame.state, cursor=frame.cursor, orders=orders, unsafe=unsafe,
                requests=requests, admitted=admitted, frame_number=core.frame_number+executed.astype(jnp.int32))
            # Originals and constructor snapshots are immutable source constants.
            chosen = _rows(executed, core._replace(originals=None, constructors=None),
                candidate._replace(originals=None, constructors=None))
            return chosen._replace(originals=core.originals, constructors=core.constructors)
        next_core = _timed(observe, 'requests', requests_phase, core, frame)
        result = _consume(r, old_state._replace(core=next_core), frame, executed, observe, old_result.owners)
        # Callbacks can observe timing but cannot replace or replay the carry.
        if _session(session) is not record or session.state is not old_state or session.result is not old_result:
            raise ValueError('timing callback changed the driver session')
        session.state, session.result = result.state, result
        record.update(state=result.state, result=result,
            diagnostics=FrameDiagnostics(frame.invalid_actions, frame.placeholder_moves, active, executed))
        return result
    finally:
        record['advancing'] = False
