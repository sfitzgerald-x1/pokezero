"""Finite full-game timing of the direct public device observation path.

The actual public29 encoder drives the champion's explicit23-token adapter and
bf16 policy, both seats. Missing hooks stay zero in its column manifest. This
first-cut speed estimate does not establish parity, strength or B4 passage.
No protocol capture, event proof, host fold or placeholder observation is used.
"""
from typing import NamedTuple
import time

import jax
import jax.numpy as j
import numpy as np

from . import champion, transition


class Carry(NamedTuple):
    state: object
    trackers: object
    ids: object
    requests: object
    cursor: object
    completed: object
    capped: object
    failed: object
    turns: object
    decisions: object
    error_counts: object
    requested_views: object
    physical_views: object
    ticks: object
    observation_checksum: object


def select_rows(mask, new, old):
    """Only current lanes change; None deferred tracker leaves remain None."""
    return jax.tree.map(lambda n, o: j.where(
        mask.reshape(mask.shape + (1,) * (o.ndim - 1)), n, o), new, old)


def request_seats(state):
    hp = j.take_along_axis(state.hp, state.active[..., None], axis=2)[..., 0]
    forced = state.forced_switch | (hp <= 0)
    return (~state.done[:, None]) & (~j.any(forced, axis=1)[:, None] | forced)


def physical_actions(actions, targets, active):
    """Map the encoder's five request-order switches to engine bench lanes."""
    target = j.take_along_axis(targets, j.clip(actions - 4, 0, 4)[..., None], axis=-1)[..., 0]
    lane = target - (target > active)
    return j.where(actions < 4, actions, 4 + lane).astype(j.int32)


def sample_keys(root_ids, requests, *, sampling_seed):
    base = jax.random.PRNGKey(sampling_seed)
    def one(root, request):
        key = jax.random.fold_in(jax.random.fold_in(base, root), request)
        return jax.vmap(lambda side: jax.random.fold_in(key, side))(j.arange(2))
    return jax.vmap(one)(root_ids, requests)


def retire(carry, after, updated_trackers, initial, *, failure, requested, errors,
           max_requests, tracker_initialize, observation_checksum=0):
    """Retain every declared cell once, including failures/caps; no redraw."""
    assigned = carry.ids >= 0
    count = carry.requests + assigned.astype(j.int32)
    done = assigned & after.done & ~failure
    failed = assigned & failure
    capped = assigned & ~done & ~failed & (count >= max_requests)
    ended = done | failed | capped
    n = carry.completed.shape[0]
    # Sentinel n uses scatter-drop, so padded lanes cannot overwrite real IDs.
    index = j.where(ended, carry.ids, n)
    completion = carry.completed.at[index].set(done, mode='drop')
    caps = carry.capped.at[index].set(capped, mode='drop')
    failures = carry.failed.at[index].set(failed, mode='drop')
    turns = carry.turns.at[index].set(j.where(ended, after.turn, 0), mode='drop')
    decisions = carry.decisions.at[j.where(assigned, carry.ids, n)].add(
        j.sum(requested, axis=1, dtype=j.int32), mode='drop')
    available = ended | ~assigned
    next_ids = carry.cursor + j.cumsum(available.astype(j.int32)) - 1
    fill = available & (next_ids < n)
    safe = j.clip(next_ids, 0, n - 1)
    fresh = jax.tree.map(lambda x: x[safe], initial)
    state = select_rows(fill, fresh, after)
    trackers = select_rows(fill, tracker_initialize(fresh), updated_trackers)
    ids = j.where(fill, next_ids, j.where(ended, -1, carry.ids))
    state = state._replace(done=state.done | (ids < 0))
    return Carry(state, trackers, ids, j.where(fill, 0, count),
        carry.cursor + j.sum(fill, dtype=j.int32), completion, caps, failures,
        turns, decisions, carry.error_counts + errors,
        carry.requested_views + j.sum(requested, dtype=j.int32),
        carry.physical_views + 2 * len(ids), carry.ticks + 1,
        carry.observation_checksum + j.asarray(observation_checksum, j.uint32))


def make_driver(config, engine_tables, observation_tables, *,
                sampling_seed, max_requests=4096, chunk_steps=16):
    """One compiled worker; engine and tracker writes share the same device step."""
    from . import engine_device_observation as encoder
    from . import engine_device_trackers as public
    def advance(c, initial, root_ids, params):
        state = c.state
        masks = transition.action_mask(state, engine_tables)
        observations = [encoder.encode(state, public.readout(c.trackers,
            j.full((len(c.ids),), side, j.int32)), observation_tables,
            observer_sides=j.full((len(c.ids),), side, j.int32),
            legal_action_mask=masks[:, side]) for side in range(2)]
        inputs = [encoder.champion_inputs(o) for o in observations]
        paired = {key: j.stack((inputs[0][key], inputs[1][key]), axis=1)
                  for key in inputs[0]}
        output = champion.forward(params, **paired, c=config, compute_dtype='bfloat16')
        # All three real heads are computed and checked, not a discarded shadow.
        finite = (j.all(j.isfinite(output.policy_logits), axis=-1)
            & j.isfinite(output.value) & j.all(j.isfinite(output.opponent_action_logits), axis=-1))
        logits = j.nan_to_num(output.policy_logits, nan=0., posinf=0., neginf=0.)
        safe_output = output._replace(policy_logits=logits)
        legal = j.stack([o.legal_action_mask for o in observations], axis=1)
        keys = sample_keys(root_ids[j.clip(c.ids, 0, len(root_ids)-1)], c.requests,
                           sampling_seed=sampling_seed)
        chosen = champion.sample(safe_output, legal, keys)
        targets = j.stack([o.switch_targets for o in observations], axis=1)
        actions = physical_actions(j.maximum(chosen, 0), targets, state.active)
        requested = request_seats(state) & (c.ids >= 0)[:, None]
        flags = j.stack([o.error for o in observations], axis=1)
        errors = j.sum(((flags[..., None] & (1 << j.arange(len(encoder.ERROR_KINDS)))) != 0)
            & requested[..., None], axis=(0, 1), dtype=j.int32)
        out = transition.step(state, actions, engine_tables)
        failure = out.invalid_actions | j.any(requested & ~finite, axis=1)
        trackers = public.update(c.trackers, state, out.state, actions,
                                 active_rows=(c.ids >= 0) & ~state.done & ~out.invalid_actions)
        # Retain ALL29 outputs, including public belief tails not consumed by
        # the champion23 adapter. Otherwise XLA could prune encoder work. This
        # diagnostic checksum wraps uint32 intentionally; it is not parity.
        checksum = j.asarray(0, j.uint32)
        for o in observations:
            checksum += j.sum(jax.lax.bitcast_convert_type(o.numeric_features, j.uint32), dtype=j.uint32)
            for value in (o.categorical_ids, o.token_type_ids, o.attention_mask):
                checksum += j.sum(value.astype(j.uint32), dtype=j.uint32)
        return retire(c, out.state, trackers, initial, failure=failure,
            requested=requested, errors=errors, max_requests=max_requests,
            tracker_initialize=public.initialize, observation_checksum=checksum)
    def chunk(c, initial, root_ids, params):
        def body(c, _):
            # No policy/engine work after the fixed finite cohort is exhausted.
            c = jax.lax.cond(j.any(c.ids >= 0), lambda x: advance(x, initial, root_ids, params), lambda x: x, c)
            return c, None
        return jax.lax.scan(body, c, None, length=chunk_steps)[0]
    return jax.jit(chunk)


def initialize(initial, batch):
    from . import engine_device_trackers as public
    n = initial.done.shape[0]
    if not 1 <= batch <= n:
        raise ValueError('positive batch within the fixed declared cohort')
    state = jax.tree.map(lambda x: x[:batch], initial)
    zero = j.zeros(n, j.int32)
    return Carry(state, public.initialize(state), j.arange(batch, dtype=j.int32),
        j.zeros(batch, j.int32), j.asarray(batch, j.int32),
        j.zeros(n, bool), j.zeros(n, bool), j.zeros(n, bool), zero, zero,
        j.zeros(7, j.int32), j.asarray(0, j.int32), j.asarray(0, j.int32), j.asarray(0, j.int32),
        j.asarray(0, j.uint32))


def summary(carry, *, seconds, batch, compile_seconds, warmup_seconds):
    complete, capped, failed, turns, decisions = map(np.asarray,
        (carry.completed, carry.capped, carry.failed, carry.turns, carry.decisions))
    n = len(complete)
    if seconds <= 0 or np.any(complete.astype(int) + capped + failed > 1):
        raise ValueError('positive wall and disjoint whole-roster outcomes required')
    unattempted = n - int(np.asarray(carry.cursor))
    inflight = int(np.sum(np.asarray(carry.ids) >= 0))
    if int(complete.sum()+capped.sum()+failed.sum()) + unattempted + inflight != n:
        raise ValueError('full fixed cohort accounting differs')
    return dict(batch=batch, declared_games=n, completed_games=int(complete.sum()),
        capped_games=int(capped.sum()), failed_games=int(failed.sum()),
        unattempted_games=unattempted, inflight_games=inflight,
        full_games_per_second=float(complete.sum()/seconds), warm_wall_seconds=seconds,
        compile_seconds=compile_seconds, warmup_seconds=warmup_seconds,
        actual_requested_views=int(np.asarray(carry.requested_views)),
        physical_encoder_forward_rows=int(np.asarray(carry.physical_views)),
        error_counts=np.asarray(carry.error_counts).tolist(),
        completed_turns=turns[complete].tolist(), all_final_turns=turns.tolist(),
        decisions_by_game=decisions.tolist(), completed=complete.tolist(),
        capped=capped.tolist(), failed=failed.tolist(), ticks=int(np.asarray(carry.ticks)),
        all29_observation_checksum_uint32=int(np.asarray(carry.observation_checksum)),
        scope='First-cut real direct encoder; zero manifest retained. No parity/coverage/strength/canary/B4 qualification.')


def run_finite(compiled, start, initial, root_ids, params, *, progress=None, stall_floor_seconds=600):
    """Whole finite cohort including tail/dispatch/scalar accounting wall time.

    Stall guard grows with the longest observed completed game so far; it is
    not a duration cap. Caller supplies a separate disjoint warmup cohort.
    """
    c = start
    begin = last_completion = time.perf_counter()
    longest = 0.
    birth = np.full(c.completed.shape, np.nan)
    previous_cursor = int(np.asarray(c.cursor)); birth[:previous_cursor] = begin
    seen = np.zeros(c.completed.shape, bool)
    previous_boundary = begin
    while True:
        c = compiled(c, initial, root_ids, params)
        completed_mask, retired, active, cursor = jax.device_get((c.completed,
            j.sum(c.completed | c.capped | c.failed), j.sum(c.ids >= 0), c.cursor))
        completed_mask = np.asarray(completed_mask)
        completed = int(completed_mask.sum()); retired, active, cursor = map(int, (retired, active, cursor))
        now = time.perf_counter()
        # New assignments/completions can occur inside a chunk. The preceding
        # boundary is a conservative start and this boundary a conservative end.
        birth[previous_cursor:cursor] = previous_boundary
        new = completed_mask & ~seen
        if np.any(new):
            longest = max(longest, float(np.max(now-birth[new])))
            last_completion = now
        seen = completed_mask; previous_cursor = cursor; previous_boundary = now
        if progress:
            progress(dict(completed=completed, retired=retired, active=active,
                elapsed_seconds=now-begin, ticks=int(c.ticks)))
        if active == 0:
            return c, time.perf_counter()-begin
        if now-last_completion > max(stall_floor_seconds, 5*longest):
            raise RuntimeError('no completed game within adaptive stall window; preserve fixed next cells')


def measure_pair(params, config, engine_tables, observation_tables, warm_initial,
                 initial, warm_roots, roots, *, batch, sampling_seed,
                 max_requests=4096, chunk_steps=16, progress=None):
    """Disjoint full warmup cohort followed by the whole declared measured one."""
    warm_roots, roots = np.asarray(warm_roots), np.asarray(roots)
    n = initial.done.shape[0]
    if (warm_initial.done.shape != initial.done.shape or warm_roots.shape != (n,)
            or roots.shape != (n,) or len(np.unique(np.concatenate((warm_roots, roots)))) != 2*n):
        raise ValueError('same shapes and disjoint per-game warm/measured root IDs required')
    initial, warm_initial, params = jax.block_until_ready((initial, warm_initial, params))
    start = jax.block_until_ready(initialize(initial, batch))
    warm_start = jax.block_until_ready(initialize(warm_initial, batch))
    roots = j.asarray(roots, j.uint32); warm_roots = j.asarray(warm_roots, j.uint32)
    driver = make_driver(config, engine_tables, observation_tables,
        sampling_seed=sampling_seed, max_requests=max_requests, chunk_steps=chunk_steps)
    before = time.perf_counter()
    lowered = driver.lower(start, initial, roots, params)
    lowering = time.perf_counter()-before
    if progress:progress(dict(phase='LOWERED', batch=batch, lowering_seconds=lowering))
    before = time.perf_counter()
    compiled = lowered.compile()
    compilation = time.perf_counter()-before
    if progress:progress(dict(phase='COMPILED', batch=batch, compile_seconds=compilation))
    def report(phase):
        return (lambda value: progress(dict(value, phase=phase, batch=batch))) if progress else None
    warm, warm_seconds = run_finite(compiled, warm_start, warm_initial, warm_roots, params,
        progress=report('DISJOINT_FULL_WARMUP'))
    finished, seconds = run_finite(compiled, start, initial, roots, params,
        progress=report('WARM_FULL_MEASURED'))
    result = summary(finished, seconds=seconds, batch=batch,
        compile_seconds=compilation, warmup_seconds=warm_seconds)
    result.update(lowering_seconds=lowering, chunk_steps=chunk_steps,
        max_requests_per_game=max_requests,
        warmup_summary=summary(warm, seconds=warm_seconds, batch=batch,
            compile_seconds=compilation, warmup_seconds=0),
        requested_views_producer_error_denominator=result['actual_requested_views'])
    return result
