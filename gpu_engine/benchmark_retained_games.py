"""Explicit, unqualified B4 retained173/bf16 finite-cohort producer.

This path owns one actual retained driver session, two separate owner observers,
and one physical engine step per advancing request. It never uses observation.py.
The driver's fake-forward boundary receives only a no-op spy. Actual champion
inference is downstream, on compacted, admitted, currently requested rows only.

No runtime is enabled by default. There is deliberately no checkpoint-loading
CLI: the reviewed remote runner supplies original compiled catalogs, a bound
constructor pool, and explicit enablement. Current full public admission remains
closed, so the current actual driver produces no model calls or completions.
Scientific use requires independent review of an immutable source freeze.
"""
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
import statistics
import subprocess
import time
from types import MappingProxyType
import weakref

import jax
import jax.numpy as j
import numpy as np

from . import champion, precision_benchmark, source_provenance
from . import engine_retained_encoder_driver as driver
from . import engine_own_facts, engine_private_actions, private_requests
from .benchmark_games import sample_actions
from .benchmark import sole_gpu

SCHEMA = 'pokezero.retained173_bf16_finite_cohort.unqualified.v1'
_RUNTIMES = weakref.WeakKeyDictionary()


@dataclass(frozen=True, eq=False)
class Runtime:
    binding: str


def _noop_fake(_):
    return driver.FakeScores(j.zeros(9, j.float32), j.array(0, j.int32))


def _hex(value):
    return type(value) is str and re.fullmatch('[a-f0-9]{64}', value) is not None


def prepare(request_compiled, own_compiled, weights_path, *,
            expected_weights_sha256, expected_checkpoint_sha256,
            enable_unqualified=False):
    """Bind an exact existing champion; opt-in precedes checkpoint loading.

    Runtime preparation is host-only. Loading does not execute the model. The
    original request/OwnFacts/catalog/checkpoint identities remain mandatory.
    """
    if enable_unqualified is not True:
        raise ValueError('explicit unqualified retained benchmark enablement required')
    if not all(_hex(x) for x in (expected_weights_sha256, expected_checkpoint_sha256)):
        raise ValueError('exact weights and checkpoint SHA256 required')
    record = engine_own_facts._checked(own_compiled)
    if (record['request_compiled'] is not request_compiled
            or record['catalog'].checkpoint_sha256 != expected_checkpoint_sha256):
        raise ValueError('original request/OwnFacts and exact encoder checkpoint required')
    # Hash before loading, then check the loader's own immutable byte snapshot.
    if precision_benchmark.digest(weights_path) != expected_weights_sha256:
        raise ValueError('weights archive hash mismatch')
    device = sole_gpu()
    with jax.default_device(device):
        params, config, metadata = champion.load(weights_path)
    if (config != champion.Config()
            or metadata['weights_archive_sha256'] != expected_weights_sha256
            or metadata['checkpoint_sha256'] != expected_checkpoint_sha256):
        raise ValueError('exact default champion/config/archive/checkpoint required')
    prepared = driver.prepare(request_compiled, own_compiled,
        public_mode='retained-full', fake_forward=_noop_fake)
    return _bind(prepared, jax.device_put(params, device), config, metadata)


def _bind(prepared, params, config, metadata):
    """Private model binding also used with tiny synthetic weights in tests."""
    champion.validate_weights(params, config)
    # JAX arrays are immutable; copy the mapping so caller replacement cannot
    # change a cached executable or its model identity.
    params = MappingProxyType({k: j.asarray(v) for k, v in params.items()})
    payload = dict(driver=prepared.binding, config=config.__dict__, weights=metadata)
    binding = hashlib.sha256(json.dumps(payload, sort_keys=True,
        separators=(',', ':')).encode()).hexdigest()
    runtime = Runtime(binding)
    _RUNTIMES[runtime] = dict(binding=binding, prepared=prepared, params=params,
        config=config, metadata=json.loads(json.dumps(metadata)), networks={})
    return runtime


def _checked(runtime):
    record = _RUNTIMES.get(runtime) if type(runtime) is Runtime else None
    if record is None or runtime.binding is not record['binding']:
        raise ValueError('original retained benchmark runtime required')
    driver._checked(record['prepared'])
    return record


def _keys(seed, stream_ids, frames, owners):
    """Independent of compaction order, survivor count, and other games."""
    root = jax.random.PRNGKey(seed)
    return jax.vmap(lambda stream, frame, owner:
        jax.random.fold_in(jax.random.fold_in(jax.random.fold_in(root, stream), owner), frame))(
            j.asarray(stream_ids, j.uint32), j.asarray(frames, j.uint32), j.asarray(owners, j.uint32))


def _requested(state):
    return np.stack([np.asarray(private_requests.action_inputs(
        request.actions.facts.request, observer_side=ledger.owner_side).ready, bool)
        for request, ledger in zip(state.core.requests, state.core.orders)], axis=1)


def _admission(result, active, config):
    """Validate both owner observations before gathering any model inputs."""
    batch = len(active)
    allowed = np.ones((batch, 2), bool)
    reasons = np.zeros((batch, 2), np.int64)
    for side, report in enumerate(result.owners):
        obs = report.encoded.observation
        reason = np.asarray(report.reason)
        if reason.shape != (batch,):
            raise ValueError('one retained admission reason per physical row/owner required')
        reasons[:, side] = reason
        allowed[:, side] = (np.asarray(report.policy_admitted)
            & np.asarray(obs.admitted) & np.asarray(report.full_line_consumers_complete)
            & (reason == 0))
        specs = ((obs.numeric_features, (batch, config.tokens, config.numeric), np.float32),
            (obs.categorical_ids, (batch, config.tokens, config.categorical), np.int32),
            (obs.token_type_ids, (batch, config.tokens), np.int32),
            (obs.attention_mask, (batch, config.tokens), np.bool_),
            (obs.legal_action_mask, (batch, 9), np.bool_))
        for value, shape, dtype in specs:
            if value.shape != shape or value.dtype != dtype:
                raise ValueError('retained encoder/model tensor contract differs')
        # Host validation is deliberately included in the advancing clock.
        numeric, categorical, types, attention, legal = [np.asarray(x[0]) for x in specs]
        valid = (np.isfinite(numeric).all(axis=(1, 2))
            & ((categorical >= 0) & (categorical < config.vocabulary)).all(axis=(1, 2))
            & ((types >= 0) & (types < config.token_types)).all(axis=1)
            & attention.any(axis=1))
        actual = result.state.core.requests[side].actions
        valid &= (legal == np.asarray(actual.features.legal)).all(axis=1)
        request_inputs = private_requests.action_inputs(actual.facts.request,
            observer_side=result.state.core.orders[side].owner_side)
        valid &= (np.asarray(obs.switch_targets) == np.asarray(request_inputs.switch_targets)).all(axis=1)
        valid &= ~np.asarray(request_inputs.ready) | legal.any(axis=1)
        allowed[:, side] &= valid
    requested = _requested(result.state)
    # Waiting is legitimate. A live game with no requested owner is unknown.
    refused = active & (~allowed.all(axis=1) | ~requested.any(axis=1))
    eligible = requested & allowed & (active & ~refused)[:, None]
    return eligible, refused, reasons


def _network(record, args):
    rows = args[0].shape[0]
    compile_ms = 0.
    if rows not in record['networks']:
        executable, seconds = precision_benchmark._compile(record['params'],
            record['config'], args, 'bfloat16')
        record['networks'][rows] = executable
        compile_ms = seconds * 1000
    start = time.perf_counter_ns()
    heads = jax.block_until_ready(record['networks'][rows](*args))
    return heads, compile_ms, (time.perf_counter_ns() - start) / 1e6


def _decide(record, result, eligible, sampling_seed):
    """One vectorized bf16 call, then use the existing action routing exactly."""
    batch = eligible.shape[0]
    flat = np.flatnonzero(eligible.reshape(-1))
    if not len(flat):
        raise ValueError('no eligible owner rows: model invocation forbidden')
    arrays = []
    for field in ('numeric_features', 'categorical_ids', 'token_type_ids', 'attention_mask'):
        both = j.stack([getattr(r.encoded.observation, field) for r in result.owners], axis=1)
        arrays.append(both.reshape((2 * batch,) + both.shape[2:])[flat])
    heads, compile_ms, forward_ms = _network(record, tuple(arrays))
    if (heads.policy_logits.shape != (len(flat), 9) or heads.value.shape != (len(flat),)
            or heads.opponent_action_logits.shape != (len(flat), 9)):
        raise ValueError('all three champion output shapes required')
    finite = np.logical_and.reduce([np.isfinite(np.asarray(x)).reshape(len(flat), -1).all(axis=1)
        for x in heads])
    owner = flat % 2
    row = flat // 2
    core = result.state.core
    keys = _keys(sampling_seed, np.asarray(core.cursor.stream_id)[row],
        np.asarray(core.frame_number)[row], owner)
    legal = j.stack([r.encoded.observation.legal_action_mask for r in result.owners], axis=1)
    legal = legal.reshape(2 * batch, 9)[flat]
    draws = np.asarray(sample_actions(heads.policy_logits, legal, keys))
    choices = np.full((batch, 2), -1, np.int32)
    choices[row, owner] = np.where(finite, draws, -1)
    refused = np.zeros(batch, bool)
    refused[row[~finite]] = True
    routed = []
    for side in (0, 1):
        route = engine_private_actions.route_choice(core.requests[side].actions,
            core.orders[side], j.asarray(choices[:, side]))
        route = jax.block_until_ready(route)
        refused |= eligible[:, side] & ~np.asarray(route.admitted)
        # Zero is an ignored transport value only for waiting/done/frozen seats.
        # A requested row with no valid route is frozen before the core call.
        routed.append(j.where(route.admitted, route.engine_action, 0))
    return j.stack(routed, axis=1), refused, dict(model_rows=len(flat),
        nonfinite_owner_rows=int((~finite).sum()), compile_ms=compile_ms, forward_ms=forward_ms)


def _stats(values):
    return dict(samples_ms=values, mean_ms=statistics.fmean(values) if values else None)


def _source_snapshot():
    root = Path(__file__).resolve().parents[1]
    def git(*args):
        return subprocess.check_output(['git', '-C', str(root), *args], text=True).strip()
    if git('status', '--porcelain', '--', 'gpu_engine'):
        raise RuntimeError('immutable clean GPU source freeze required')
    return dict(commit=git('rev-parse', 'HEAD'), package_sha256={
        path.name: precision_benchmark.digest(path) for path in sorted((root / 'gpu_engine').glob('*.py'))})


def measure(runtime, pool, team_indices, *, stream_ids, constructor_seed=719,
            sampling_seed=9771, max_requests=1000, enable_unqualified=False):
    """Advance one immutable finite cohort; no refill, replay, or warmup games.

    Host compaction/validation, cold shape compilations, per-phase barriers and
    callbacks are included in advancing wall time. This is an instrumented cost,
    never an unprofiled speed estimate or a sum-of-components speed claim.
    """
    if enable_unqualified is not True:
        raise ValueError('explicit unqualified retained benchmark enablement required')
    if (type(max_requests) is not int or max_requests < 1
            or any(type(x) is not int or not 0 <= x < 2**32
                for x in (constructor_seed, sampling_seed))):
        raise ValueError('positive finite request cap and uint32 seeds required')
    record = _checked(runtime)
    source = _source_snapshot()
    receipt = source_provenance._pool(pool, rehash=True)
    indices = np.asarray(team_indices)
    ids = np.asarray(stream_ids)
    if (indices.ndim != 2 or indices.shape[1] != 2 or not len(indices)
            or indices.dtype.kind not in 'iu' or len(np.unique(indices)) != indices.size
            or np.any(indices < 0) or np.any(indices >= len(pool.arrays['species']))
            or ids.shape != (len(indices),) or ids.dtype != np.int32
            or np.any(ids <= 0) or len(np.unique(ids)) != len(ids)):
        raise ValueError('distinct fixed team pairs and unique positive int32 stream IDs required')
    batch = len(indices)
    start = time.perf_counter_ns()
    session = driver.construct(pool, indices, prepared=record['prepared'],
        stream_ids=j.asarray(ids), seed=constructor_seed)
    jax.block_until_ready(session.result)
    construction_ms = (time.perf_counter_ns() - start) / 1e6
    completed, refused = np.zeros(batch, bool), np.zeros(batch, bool)
    details = [dict(stream_id=int(x), team_indices=pair.tolist(), refusal=None) for x, pair in zip(ids, indices)]
    full_steps, forward, compiles, phases, decisions = [], [], [], {}, []
    encoder_steps, valid_steps, model_rows = [], 0, 0
    placeholder_moves = np.zeros(batch, np.int64)
    executed_steps = np.zeros(batch, np.int64)
    start = time.perf_counter_ns()
    status = 'REQUEST_CAP'
    for index in range(max_requests):
        step_start = time.perf_counter_ns()
        _checked(runtime)  # original static/catalog guards run before every step
        active = ~completed & ~refused
        # An initially terminal or externally replaced state is never a measured
        # completion. The checked driver separately guards current-carry identity.
        initially_done = active & np.asarray(session.state.core.engine.done)
        refused |= initially_done
        for row in np.flatnonzero(initially_done):
            details[row]['refusal'] = dict(kind='terminal-without-observed-transition', request=index)
        active &= ~initially_done
        eligible, failed, reasons = _admission(session.result, active, record['config'])
        refused |= failed
        for row in np.flatnonzero(failed):
            details[row]['refusal'] = dict(kind='retained-or-request-admission', request=index,
                owner_reason=reasons[row].tolist())
        active &= ~failed
        if not active.any():
            status = 'COMPLETE_COHORT' if completed.all() else 'NO_ADMITTED_LIVE_ROWS'
            break
        actions, failed, cost = _decide(record, session.result, eligible, sampling_seed)
        model_rows += cost['model_rows']
        forward.append(cost['forward_ms']); compiles.append(cost['compile_ms'])
        decisions.append(cost)
        refused |= failed
        for row in np.flatnonzero(failed):
            details[row]['refusal'] = dict(kind='nonfinite-head-or-unroutable-choice', request=index)
        active &= ~failed
        if not active.any():
            status = 'NO_ADMITTED_LIVE_ROWS'
            break
        step_phases = {}
        def timing_callback(name, milliseconds):
            if (type(name) is not str or not np.isfinite(milliseconds) or milliseconds < 0):
                raise ValueError('finite synchronized driver phase timing required')
            step_phases[name] = step_phases.get(name, 0.) + float(milliseconds)
        before_frames = np.asarray(session.state.core.frame_number).copy()
        result = driver.advance(session, actions, active_rows=j.asarray(active), timing_callback=timing_callback)
        jax.block_until_ready(result)
        diagnostics = driver.latest_diagnostics(session)
        jax.block_until_ready(diagnostics)
        if not np.array_equal(np.asarray(diagnostics.active_rows), active):
            raise RuntimeError('driver advanced a different physical row set')
        executed = np.asarray(diagnostics.executed_rows)
        if executed.shape != (batch,) or executed.dtype != np.bool_ or np.any(executed & ~active):
            raise RuntimeError('driver executed outside the requested physical row set')
        if not np.array_equal(np.asarray(result.state.core.frame_number) - before_frames, executed.astype(np.int32)):
            raise RuntimeError('physical core frame must advance exactly once per active row')
        executed_steps += executed
        placeholder_moves += np.where(executed, np.asarray(diagnostics.placeholder_moves), 0)
        # A terminal transition has no further owning request to score, but its
        # public frame must still be consumed successfully by both observers.
        # Do not treat an expected terminal OwnFacts refusal as public failure.
        public_ok = np.logical_and.reduce([
            (np.asarray(report.basic_error) == 0) & (np.asarray(report.retained_error) == 0)
            & np.asarray(observer.retained_healthy)
            for report, observer in zip(result.owners, result.state.observers)])
        invalid = active & (~executed | np.asarray(diagnostics.invalid_actions)
            | ~np.asarray(result.state.core.admitted) | ~public_ok)
        refused |= invalid
        for row in np.flatnonzero(invalid):
            details[row]['refusal'] = dict(kind='engine-or-source-refusal', request=index,
                invalid_action=bool(np.asarray(diagnostics.invalid_actions)[row]),
                executed=bool(executed[row]), public_frame_admitted=bool(public_ok[row]))
        completed |= executed & ~invalid & np.asarray(result.state.core.engine.done)
        required = {'engine', 'requests', 'public_p1', 'public_p2', 'encode_p1', 'encode_p2',
            'admission_p1', 'admission_p2'}
        if set(step_phases) != required:
            raise RuntimeError('complete synchronized driver timing phases required')
        for name, value in step_phases.items():
            phases.setdefault(name, []).append(value)
        encoder_steps.append(sum(step_phases[x] for x in
            ('public_p1', 'public_p2', 'encode_p1', 'encode_p2', 'admission_p1', 'admission_p2')))
        full_steps.append((time.perf_counter_ns() - step_start) / 1e6)
        valid_steps += 1
        if completed.all():
            status = 'COMPLETE_COHORT'
            break
    # Classify new encoder refusals even when the cap lands on that boundary.
    active = ~completed & ~refused
    _, failed, reasons = _admission(session.result, active, record['config'])
    refused |= failed
    for row in np.flatnonzero(failed):
        details[row]['refusal'] = dict(kind='retained-or-request-admission-at-cap',
            request=valid_steps, owner_reason=reasons[row].tolist())
    elapsed = (time.perf_counter_ns() - start) / 1e9
    censored = ~completed & ~refused
    for row, detail in enumerate(details):
        detail.update(status='COMPLETED' if completed[row] else 'UNKNOWN_REFUSED' if refused[row]
            else 'UNKNOWN_CENSORED', final_turn=int(np.asarray(session.state.core.engine.turn)[row]),
            engine_placeholder_moves=int(placeholder_moves[row]), executed_steps=int(executed_steps[row]),
            final_owner_full_line_consumers_complete=[bool(np.asarray(owner.full_line_consumers_complete)[row])
                for owner in session.result.owners])
    source_provenance._pool(pool, rehash=True)
    _checked(runtime)
    if _source_snapshot() != source:
        raise RuntimeError('bound source changed during retained measurement')
    return dict(schema=SCHEMA, status=status, runtime_binding=runtime.binding,
        source=source,
        full_B4_qualified=False, fidelity_qualified=False, strength_qualified=False,
        compute_dtype='bfloat16', model_outputs=['policy_logits', 'value', 'opponent_action_logits'],
        weights=record['metadata'], pool_receipt=receipt, batch=batch, original_denominator=batch,
        completed_games=int(completed.sum()), refused_games=int(refused.sum()),
        censored_games=int(censored.sum()), unknown_games=int((~completed).sum()),
        completed_fraction=float(completed.mean()),
        games_per_second=int(completed.sum()) / elapsed if valid_steps else None,
        advancing_elapsed_seconds=elapsed, construction_ms=construction_ms,
        advancing_steps=valid_steps, max_requests=max_requests, model_owner_rows=model_rows,
        engine_placeholder_moves=int(placeholder_moves.sum()), executed_game_steps=int(executed_steps.sum()),
        full_step=_stats(full_steps), bf16_forward=_stats(forward), cold_compile=_stats(compiles),
        driver_components={name: _stats(values) for name, values in phases.items()},
        retained_encoder=_stats(encoder_steps),
        encoder_fraction_of_instrumented_full_step=(sum(encoder_steps) / sum(full_steps) if full_steps else None),
        observation_trigger_25_percent=(sum(encoder_steps) / sum(full_steps) > .25 if batch == 2048 and full_steps else None),
        constructor_seed=constructor_seed, sampling_seed=sampling_seed, cohort=details, decisions=decisions,
        scope='Unqualified retained173/bf16 actual finite cohort; no fidelity/strength/B4 qualification.',
        accounting='Original denominator retained; actual observed valid terminal transitions only complete; refused and unfinished rows unknown; no refill/redraw/pooling.',
        timing='Instrumented synchronized host wall; construction excluded; compaction/validation/cold recompiles/phase barriers/callbacks included. Components separately scoped; no sum-of-components speed claim.',
        encoder_scope='Both owner public folds, own facts/retained encode and admission; excludes source requests, model and routing. Fraction uses the same advancing steps.',
        cold_shape_policy='One compact vectorized forward per request; no waiting/done/refused inputs forwarded; no warmup/replay or prefix discarded.')
