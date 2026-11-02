"""Opt-in repeated fixed-input component profiles, not played games.

The caller supplies a genuine immutable boundary, its actual actions/chance,
and the original loaded public29 actor. This module constructs/samples nothing,
does not advance a measured trajectory, and cannot report games/second. Every
timed invocation reuses the same input arrays and returns the full outputs.
"""
from dataclasses import dataclass
import hashlib
import statistics
import time

import numpy as np

SCHEMA = 'pokezero.device-selfplay-fixed-component-profile.v1'


@dataclass(frozen=True)
class FixedInputs:
    state: object
    trackers: object
    actions: object
    chance: object
    active_rows: object


def _validate(snapshot):
    from .state import BattleState
    from .transition import ChanceInputs
    from .engine_device_trackers import Trackers
    if (type(snapshot) is not FixedInputs or type(snapshot.state) is not BattleState
            or type(snapshot.trackers) is not Trackers or type(snapshot.chance) is not ChanceInputs):
        raise ValueError('actual typed state/public/chance boundary required')
    import jax
    b = snapshot.state.done.shape[0]
    if b < 1 or snapshot.state.done.shape != (b,) or snapshot.state.done.dtype != np.bool_:
        raise ValueError('nonempty current done flags required')
    if snapshot.actions.shape != (b, 2) or snapshot.actions.dtype != np.int32:
        raise ValueError('same-boundary int32 Bx2 physical actions required')
    if (getattr(snapshot.active_rows, 'shape', None) != (b,)
            or snapshot.active_rows.dtype != np.bool_
            or not np.array_equal(np.asarray(snapshot.active_rows), ~np.asarray(snapshot.state.done))):
        raise ValueError('explicit actual active mask must match this boundary current done flags')
    for value in jax.tree.leaves((snapshot.state, snapshot.trackers)):
        if value.ndim < 1 or value.shape[0] != b:
            raise ValueError('every state/public carry leaf must have the same batch')
    for key, value in zip(snapshot.chance._fields, snapshot.chance):
        shape = (b, 2, 12) if key == 'draws' else (b,) if key == 'initial_tie' else (b, 2)
        if value.shape != shape or value.dtype != np.int32:
            raise ValueError('original complete int32 chance capsule required: '+key)
    if snapshot.trackers.observed_moves is None or snapshot.trackers.mon_features is None:
        raise ValueError('actual initialize_with_public tracker carry required')
    return b


def gather_tail(snapshot, *, capacity=16):
    """Gather current living rows, pad only from a genuine original done row.

    No actual row is replaced or advanced. The returned diagnostic copy retains
    every active row exactly once. Padded actor rows still execute and are
    explicitly counted; they are not extra games or qualified observations.
    """
    if type(capacity) is not int or not 1 <= capacity <= 16:
        raise ValueError('bounded tail capacity1..16 required')
    b = _validate(snapshot)
    active = np.flatnonzero(np.asarray(snapshot.active_rows))
    if not len(active): return None, dict(status='UNMEASURED', reason='NO_ACTIVE_ROWS', active_rows=0)
    if len(active) > capacity:
        return None, dict(status='UNMEASURED', reason='ACTIVE_ROWS_EXCEED_TAIL_CAPACITY', active_rows=len(active))
    import jax
    import jax.numpy as j
    n = min(capacity, b)
    inactive = np.flatnonzero(~np.asarray(snapshot.active_rows))
    rows = np.full(n, inactive[0] if len(inactive) else active[0], np.int32)
    rows[:len(active)] = active
    index = j.asarray(rows, j.int32)
    state, trackers, actions, chance, active_rows = jax.tree.map(lambda value: value[index],
        (snapshot.state, snapshot.trackers, snapshot.actions, snapshot.chance, snapshot.active_rows))
    padding = j.arange(n) >= len(active)
    return FixedInputs(state, trackers, actions, chance, active_rows), dict(status='READY',
        source_batch=b, physical_batch=n, active_rows=len(active), padding_rows=n-len(active),
        source_rows=rows.tolist(), padding_mask=np.asarray(padding).tolist(),
        scope='Diagnostic gathered copy only; each active source row once, padding copies one original already-done row. No invented state, trajectory or policy change.')


def make_components(engine_tables, observation_tables, config, *, include_bare=True, dense_public_updates=False):
    """Pure original hot paths. No construction, chance sampling, or actor init."""
    if type(include_bare) is not bool: raise ValueError('strict optional bare-step flag required')
    if type(dense_public_updates) is not bool: raise ValueError('static dense public-update flag required')
    import jax.numpy as j
    from . import transition, engine_device_observation as encoder, engine_device_trackers as public
    from .benchmark_device_selfplay import run_actor

    def encode(state, trackers):
        b = state.done.shape[0]
        masks = transition.action_mask(state, engine_tables)
        return tuple(encoder.encode(state, public.readout(trackers, j.full((b,), side, j.int32)),
            observation_tables, observer_sides=j.full((b,), side, j.int32),
            legal_action_mask=masks[:, side]) for side in (0, 1))

    def forward(params, observations):
        return run_actor(params, config, observations, actor_kind='v5-public29')

    def public_step(state, actions, trackers, chance):
        options = dict(dense_public_updates=True) if dense_public_updates else {}
        return transition.step_with_public(state, actions, engine_tables, trackers,
            belief_tables=observation_tables.belief, chance_inputs=chance, **options)

    def bare_step(state, actions, chance):
        return transition.step_with_chance(state, actions, engine_tables, chance)

    def joint(state, actions, trackers, chance, params):
        observations = encode(state, trackers)
        actor = forward(params, observations)
        result = public_step(state, actions, trackers, chance)
        # Return ALL29 tensors, real policy/value results, and full state/public
        # output: XLA cannot silently prune columns or hooks from this profile.
        return observations, actor, result

    result = dict(encoder_both_seats=encode, public29_bf16_forward=forward,
        step_with_public=public_step, fixed_actions_joint=joint)
    if include_bare: result['bare_step_same_chance'] = bare_step
    return result


def _duration(clock, call, *, block=None):
    start = clock()
    result = call()
    if block is not None: result = block(result)
    ns = clock()-start
    if not isinstance(ns, (int, float)) or not np.isfinite(ns) or ns < 0:
        raise ValueError('monotonic finite host interval required')
    return result, ns/1e6


def _measure(function, args, *, repeats, warmup, jit, block, clock):
    """One explicit lower/compile, then SAME AOT executable/signature only."""
    block(args)  # input readiness outside every component clock
    cached = jit(function)
    lowered, lowering = _duration(clock, lambda: cached.lower(*args))
    compiled, compilation = _duration(clock, lowered.compile)
    result, first = _duration(clock, lambda: compiled(*args), block=block)
    warm = []
    for _ in range(warmup): result, value = _duration(clock, lambda: compiled(*args), block=block); warm.append(value)
    samples = []
    for _ in range(repeats): result, value = _duration(clock, lambda: compiled(*args), block=block); samples.append(value)
    return result, dict(status='MEASURED_FIXED_INPUT', lowering_wall_ms=lowering,
        compile_or_cache_load_wall_ms=compilation, first_execute_blocked_ms=first,
        warmup_execute_blocked_ms=warm, warm_execute_blocked_ms=samples,
        warm_median_ms=statistics.median(samples), warm_mean_ms=statistics.fmean(samples),
        warm_min_ms=min(samples), warm_max_ms=max(samples), explicit_compile_calls=1,
        warmed_executable='SAME_AOT_EXECUTABLE_AND_SIGNATURE', measured_calls=repeats,
        synchronization='jax.block_until_ready on inputs before clock; on whole output before end of every execution interval',
        compile_scope='Explicit lower/compile-or-cache-load host wall; not an exclusive backend compiler-service measurement')


def _geometry(snapshot):
    from .benchmark_device_selfplay import request_seats
    active = np.asarray(snapshot.active_rows)
    requested = np.asarray(request_seats(snapshot.state)) & active[:, None]
    return dict(physical_engine_rows=len(active), active_engine_rows=int(active.sum()),
        physical_encoder_owner_rows=2*len(active), actual_requested_owner_rows=int(requested.sum()),
        physical_bf16_owner_rows=2*len(active), actor_tokens=29,
        padded_or_terminal_owner_rows=int(2*(~active).sum()))


def _physical_difference(left, right):
    # Outside all timing intervals. Same-code bare/public outputs, not an
    # engine/Showdown comparison or new game outcome/strength measurement.
    return [name for name in left._fields
        if not np.array_equal(np.asarray(getattr(left, name)), np.asarray(getattr(right, name)))]


def _payload_summary(value):
    """Observe every returned leaf once, outside all component timing clocks."""
    import jax
    digest = hashlib.sha256(); count = 0; size = 0; nonfinite = 0
    for leaf in jax.tree.leaves(value):
        array = np.asarray(leaf)
        if array.dtype.kind not in 'biufc': raise ValueError('primitive array outputs required')
        digest.update((str(array.dtype)+':'+str(array.shape)+'\0').encode())
        digest.update(array.tobytes())
        count += 1; size += array.nbytes
        if array.dtype.kind in 'fc': nonfinite += int(np.count_nonzero(~np.isfinite(array)))
    return dict(sha256=digest.hexdigest(), returned_array_leaves=count, returned_bytes=size,
        nonfinite_elements=nonfinite, scope='Last actual returned whole payload; host transfer/hash outside ALL component timing intervals. No output pruning or game credit.')


def profile_fixed(snapshot, *, params, config, engine_tables, observation_tables,
                  bindings, enable_profile=False, repeats=20, warmup=3, tail_capacity=16, include_bare=True, dense_public_updates=False):
    """Profile a caller-supplied declared fixed boundary, opt-in before work.

    ``bindings`` names actual source/pool/actor/declaration/snapshot artifacts;
    the caller loads/verifies these once. No archive or runtime is fabricated
    here. A different fixed boundary needs an explicit separate profile call.
    """
    if enable_profile is not True: raise ValueError('explicit fixed-input profiling enablement required')
    if (type(repeats) is not int or not 2 <= repeats <= 1000
            or type(warmup) is not int or not 1 <= warmup <= 100
            or type(include_bare) is not bool or type(dense_public_updates) is not bool or type(bindings) is not dict or not bindings):
        raise ValueError('bounded repeats/warmup and actual input provenance required')
    if type(tail_capacity) is not int or not 1 <= tail_capacity <= 16:
        raise ValueError('tail capacity must be1..16')
    _validate(snapshot)
    import jax
    from . import v5_model
    from pokezero.v5_model import V5ModelConfig
    if type(config) is not V5ModelConfig:
        raise ValueError('actual original public29 model configuration required')
    v5_model.validate_weights(params, config)
    components = make_components(engine_tables, observation_tables, config, include_bare=include_bare,
        **(dict(dense_public_updates=True) if dense_public_updates else {}))
    tail, tail_meta = gather_tail(snapshot, capacity=tail_capacity)
    variants = [('full', snapshot, dict(status='READY', source_batch=len(snapshot.state.done),
        physical_batch=len(snapshot.state.done), padding_rows=0))]
    if tail is not None: variants.append(('gathered_tail', tail, tail_meta))
    rows = []
    for label, fixed, meta in variants:
        outputs = {}; phases = {}
        args = dict(encoder_both_seats=(fixed.state, fixed.trackers),
            step_with_public=(fixed.state, fixed.actions, fixed.trackers, fixed.chance),
            bare_step_same_chance=(fixed.state, fixed.actions, fixed.chance),
            fixed_actions_joint=(fixed.state, fixed.actions, fixed.trackers, fixed.chance, params))
        for name in ('encoder_both_seats', 'public29_bf16_forward', 'step_with_public',
                     'bare_step_same_chance', 'fixed_actions_joint'):
            if name not in components: continue
            arguments = (params, outputs['encoder_both_seats']) if name == 'public29_bf16_forward' else args[name]
            outputs[name], phases[name] = _measure(components[name], arguments,
                repeats=repeats, warmup=warmup, jit=jax.jit, block=jax.block_until_ready, clock=time.perf_counter_ns)
        original = outputs['step_with_public'].step
        same = None if not include_bare else _physical_difference(original.state, outputs['bare_step_same_chance'].state)
        errors = [int(np.count_nonzero(np.asarray(o.error))) for o in outputs['encoder_both_seats']]
        actor = outputs['public29_bf16_forward']
        rows.append(dict(variant=label, gathering=meta, geometry=_geometry(fixed), phases=phases,
            public_vs_bare_differing_state_fields=same,
            fixed_step_invalid_rows=int(np.count_nonzero(np.asarray(original.invalid_actions))),
            encoder_error_physical_rows_by_seat=errors,
            actor_nonfinite_physical_owner_rows=int(np.count_nonzero(~np.asarray(actor.finite))),
            returned_payloads={name: _payload_summary(value) for name, value in outputs.items()},
            public_trackers_returned=True))
    return dict(schema=SCHEMA, source_bindings=dict(bindings), profiles=rows, dense_public_updates=dense_public_updates,
        gathered_tail_status=tail_meta, repeats=repeats, warmup=warmup,
        actor_kind='v5-public29', actor_compute_dtype='bfloat16', model_initialization_calls=0,
        model_config=config.metadata(), runtime=dict(jax=jax.__version__, numpy=np.__version__,
            backend=jax.default_backend()),
        sampled_chance_calls=0, constructor_calls=0, played_games=0, full_game_throughput=None,
        scope='REPEATED_FIXED_INPUT_KERNEL_PROFILE_ONLY; not a new game/cohort, refill, replay trajectory, strength or qualification.',
        attribution='Components and joint intervals overlap and must not be summed. Bare/public difference is descriptive same fixed chance, not an exclusive hook-time attribution.',
        selection='Caller declares fixed input boundaries before profiling; no outcome-informed resampling or stopping. Full and tail are two compiled shapes over the same active source rows.',
        policy='Supplied fixed actions are reused; real forward outputs are retained but never choose a new trajectory action here.',
        synchronization='Blocked component intervals; no sampling/retirement/storage cost included. First execution and compilation reported separately.',
        qualification=False)
