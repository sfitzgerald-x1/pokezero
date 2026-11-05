"""Finite full-game timing of the direct public device observation path.

The actual public29 encoder drives the champion's explicit23-token adapter and
bf16 policy, both seats. Missing hooks stay zero in its column manifest. Timing
alone does not establish parity, strength or B4 passage.
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
    error_views: object


class ActorResult(NamedTuple):
    policy_logits: object
    value: object
    finite: object


def run_actor(params, config, observations, *, actor_kind):
    """Actual selected actor; no fabricated auxiliary heads for v5 inference."""
    from . import engine_device_observation as encoder
    if actor_kind == 'champion23':
        inputs = [encoder.champion_inputs(o) for o in observations]
        paired = {key: j.stack((inputs[0][key], inputs[1][key]), axis=1)
                  for key in inputs[0]}
        out = champion.forward(params, **paired, c=config, compute_dtype='bfloat16')
        finite = (j.all(j.isfinite(out.policy_logits), axis=-1) & j.isfinite(out.value)
            & j.all(j.isfinite(out.opponent_action_logits), axis=-1))
        return ActorResult(out.policy_logits, out.value, finite)
    if actor_kind != 'v5-public29':
        raise ValueError('explicit champion23 or v5-public29 actor required')
    from . import v5_model
    inputs = [encoder.actor_inputs(o) for o in observations]
    b = observations[0].numeric_features.shape[0]
    paired = {key: j.stack((inputs[0][key], inputs[1][key]), axis=1).reshape(
        (2*b,) + inputs[0][key].shape[1:]) for key in inputs[0]}
    out = v5_model.forward(params, paired, config, compute_dtype='bfloat16')
    finite = out.valid & j.all(j.isfinite(out.policy_logits), axis=-1) & j.isfinite(out.value)
    if out.value_logits is not None:finite &= j.all(j.isfinite(out.value_logits), axis=-1)
    return ActorResult(out.policy_logits.reshape(b, 2, 9), out.value.reshape(b, 2), finite.reshape(b, 2))


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


def retire(carry, after, updated_trackers, initial, initial_public, *, failure, requested, errors,
           max_requests, observation_checksum=0, error_views=0):
    """Retain every declared cell once, including failures/caps; no redraw."""
    assigned = carry.ids >= 0
    requested = requested & assigned[:, None]
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
    fresh_public = jax.tree.map(lambda x: x[safe], initial_public)
    state = select_rows(fill, fresh, after)
    # Reuse the SAME constructor's actual Ability Start/Trace/entry knowledge,
    # never reconstruct public state by polling an already initialized battle.
    trackers = select_rows(fill, fresh_public, updated_trackers)
    ids = j.where(fill, next_ids, j.where(ended, -1, carry.ids))
    state = state._replace(done=state.done | (ids < 0))
    return Carry(state, trackers, ids, j.where(fill, 0, count),
        carry.cursor + j.sum(fill, dtype=j.int32), completion, caps, failures,
        turns, decisions, carry.error_counts + errors,
        carry.requested_views + j.sum(requested, dtype=j.int32),
        carry.physical_views + 2 * len(ids), carry.ticks + 1,
        carry.observation_checksum + j.asarray(observation_checksum, j.uint32),
        carry.error_views + j.asarray(error_views, j.int32))


def make_driver(config, engine_tables, observation_tables, *,
                sampling_seed, max_requests=4096, chunk_steps=16, actor_kind='champion23',
                donate_carry=False):
    """One compiled worker; engine and tracker writes share the same device step."""
    if type(donate_carry) is not bool:
        raise ValueError('literal carry donation opt-in required')
    from . import engine_device_observation as encoder
    from . import engine_device_trackers as public
    def advance(c, initial, initial_public, root_ids, params):
        state = c.state
        masks = transition.action_mask(state, engine_tables)
        observations = [encoder.encode(state, public.readout(c.trackers,
            j.full((len(c.ids),), side, j.int32)), observation_tables,
            observer_sides=j.full((len(c.ids),), side, j.int32),
            legal_action_mask=masks[:, side]) for side in range(2)]
        output = run_actor(params, config, observations, actor_kind=actor_kind)
        finite = output.finite
        logits = j.nan_to_num(output.policy_logits, nan=0., posinf=0., neginf=0.)
        safe_output = output._replace(policy_logits=logits)
        legal = j.stack([o.legal_action_mask for o in observations], axis=1)
        keys = sample_keys(root_ids[j.clip(c.ids, 0, len(root_ids)-1)], c.requests,
                           sampling_seed=sampling_seed)
        chosen = champion.sample(safe_output, legal, keys)
        actions = j.stack([encoder.map_actions(observations[side], state,
            j.full((len(c.ids),), side, j.int32), j.maximum(chosen[:, side], 0))
            for side in range(2)], axis=1)
        requested = request_seats(state) & (c.ids >= 0)[:, None]
        flags = j.stack([o.error for o in observations], axis=1)
        encoder_errors = j.sum(((flags[..., None] & (1 << j.arange(len(encoder.ERROR_KINDS)))) != 0)
            & requested[..., None], axis=(0, 1), dtype=j.int32)
        actor_errors = j.sum(requested & ~finite, dtype=j.int32)
        errors = j.concatenate((encoder_errors, actor_errors[None]))
        error_views = j.sum(requested & ((flags != 0) | ~finite), dtype=j.int32)
        public_out = transition.step_with_public(state, actions, engine_tables, c.trackers,
            belief_tables=observation_tables.belief)
        out = public_out.step
        # An observation/actor refusal keeps a real view and legal fallback;
        # retain its kind/union count and continue the game. Only an actual
        # engine advancement refusal ends that declared game as failed.
        failure = out.invalid_actions
        trackers = public_out.trackers
        # Retain ALL29 outputs, including public belief tails not consumed by
        # the champion23 adapter. Otherwise XLA could prune encoder work. This
        # diagnostic checksum wraps uint32 intentionally; it is not parity.
        checksum = j.asarray(0, j.uint32)
        for o in observations:
            checksum += j.sum(jax.lax.bitcast_convert_type(o.numeric_features, j.uint32), dtype=j.uint32)
            for value in (o.categorical_ids, o.token_type_ids, o.attention_mask):
                checksum += j.sum(value.astype(j.uint32), dtype=j.uint32)
        return retire(c, out.state, trackers, initial, initial_public, failure=failure,
            requested=requested, errors=errors, max_requests=max_requests,
            observation_checksum=checksum, error_views=error_views)
    def chunk(c, initial, initial_public, root_ids, params):
        def body(c, _):
            # No policy/engine work after the fixed finite cohort is exhausted.
            c = jax.lax.cond(j.any(c.ids >= 0), lambda x: advance(x, initial, initial_public, root_ids, params), lambda x: x, c)
            return c, None
        return jax.lax.scan(body, c, None, length=chunk_steps)[0]
    return jax.jit(chunk, donate_argnums=(0,)) if donate_carry else jax.jit(chunk)


def initialize(initial, initial_public, batch):
    from . import engine_device_observation as encoder
    n = initial.done.shape[0]
    if not 1 <= batch <= n:
        raise ValueError('positive batch within the fixed declared cohort')
    state = jax.tree.map(lambda x: x[:batch], initial)
    public = jax.tree.map(lambda x: x[:batch], initial_public)
    zero = j.zeros(n, j.int32)
    return Carry(state, public, j.arange(batch, dtype=j.int32),
        j.zeros(batch, j.int32), j.asarray(batch, j.int32),
        j.zeros(n, bool), j.zeros(n, bool), j.zeros(n, bool), zero, zero,
        j.zeros(len(encoder.ERROR_KINDS)+1, j.int32), j.asarray(0, j.int32), j.asarray(0, j.int32), j.asarray(0, j.int32),
        j.asarray(0, j.uint32), j.asarray(0, j.int32))


def own_carry_buffers(carry):
    """One private device copy PER leaf, before any donated execution.

    Initialization may share bank slices, counters and tracker feature arrays.
    Copy each occurrence separately: neither the immutable bank/params nor two
    donated leaves may share a buffer. None remains a structural empty leaf.
    The supported JAX device_put(may_alias=False) contract forces each copy.
    """
    return jax.block_until_ready(jax.tree.map(
        lambda value: jax.device_put(value, may_alias=False), carry))


def summary(carry, *, seconds, batch, compile_seconds, warmup_seconds):
    from . import engine_device_observation as encoder
    complete, capped, failed, turns, decisions = map(np.asarray,
        (carry.completed, carry.capped, carry.failed, carry.turns, carry.decisions))
    n = len(complete)
    names=list(encoder.ERROR_KINDS)+['actor_nonfinite']
    if (np.asarray(carry.error_counts).shape != (len(names),)
            or int(np.asarray(carry.decisions).sum()) != int(np.asarray(carry.requested_views))):
        raise ValueError('typed error kinds and actual decision/view accounting required')
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
        error_kind_names=names,
        producer_error_views=int(np.asarray(carry.error_views)),
        producer_error_fraction=float(np.asarray(carry.error_views))/max(1,int(np.asarray(carry.requested_views))),
        completed_turns=turns[complete].tolist(), all_final_turns=turns.tolist(),
        decisions_by_game=decisions.tolist(), completed=complete.tolist(),
        capped=capped.tolist(), failed=failed.tolist(), ticks=int(np.asarray(carry.ticks)),
        all29_observation_checksum_uint32=int(np.asarray(carry.observation_checksum)),
        scope='Real direct encoder; zero manifest retained. Timing alone is not parity/coverage/strength/canary/B4 qualification.')


def run_finite(compiled, start, initial, initial_public, root_ids, params, *, progress=None,
               stall_floor_seconds=600, donate_carry=False):
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
        c = compiled(c, initial, initial_public, root_ids, params)
        completed_mask, retired, active, cursor = jax.device_get((c.completed,
            j.sum(c.completed | c.capped | c.failed), j.sum(c.ids >= 0), c.cursor))
        # CPU device_get can expose a borrowed view. A later donated call may
        # reuse its buffer; the host lifetime/stall snapshot must own its bytes.
        completed_mask = (np.array(completed_mask, copy=True) if donate_carry
                          else np.asarray(completed_mask))
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
                 initial, warm_public, initial_public, warm_roots, roots, *, batch, sampling_seed,
                 max_requests=4096, chunk_steps=16, progress=None, actor_kind='champion23',
                 donate_carry=False):
    """Disjoint full warmup cohort followed by the whole declared measured one."""
    if type(donate_carry) is not bool:
        raise ValueError('literal carry donation opt-in required')
    warm_roots, roots = np.asarray(warm_roots), np.asarray(roots)
    n = initial.done.shape[0]
    if (warm_initial.done.shape != initial.done.shape or warm_roots.shape != (n,)
            or roots.shape != (n,) or len(np.unique(np.concatenate((warm_roots, roots)))) != 2*n):
        raise ValueError('same shapes and disjoint per-game warm/measured root IDs required')
    initial, warm_initial, warm_public, initial_public, params = jax.block_until_ready(
        (initial, warm_initial, warm_public, initial_public, params))
    start = jax.block_until_ready(initialize(initial, initial_public, batch))
    warm_start = jax.block_until_ready(initialize(warm_initial, warm_public, batch))
    ownership_seconds = None
    if donate_carry:
        before = time.perf_counter()
        start, warm_start = own_carry_buffers(start), own_carry_buffers(warm_start)
        ownership_seconds = time.perf_counter()-before
    roots = j.asarray(roots, j.uint32); warm_roots = j.asarray(warm_roots, j.uint32)
    driver = make_driver(config, engine_tables, observation_tables,
        sampling_seed=sampling_seed, max_requests=max_requests, chunk_steps=chunk_steps, actor_kind=actor_kind,
        donate_carry=donate_carry)
    before = time.perf_counter()
    lowered = driver.lower(start, initial, initial_public, roots, params)
    lowering = time.perf_counter()-before
    if progress:progress(dict(phase='LOWERED', batch=batch, lowering_seconds=lowering))
    before = time.perf_counter()
    compiled = lowered.compile()
    compilation = time.perf_counter()-before
    if progress:progress(dict(phase='COMPILED', batch=batch, compile_seconds=compilation))
    def report(phase):
        return (lambda value: progress(dict(value, phase=phase, batch=batch))) if progress else None
    warm, warm_seconds = run_finite(compiled, warm_start, warm_initial, warm_public, warm_roots, params,
        progress=report('DISJOINT_FULL_WARMUP'), donate_carry=donate_carry)
    finished, seconds = run_finite(compiled, start, initial, initial_public, roots, params,
        progress=report('WARM_FULL_MEASURED'), donate_carry=donate_carry)
    result = summary(finished, seconds=seconds, batch=batch,
        compile_seconds=compilation, warmup_seconds=warm_seconds)
    result.update(lowering_seconds=lowering, chunk_steps=chunk_steps, actor_kind=actor_kind,
        donate_carry=donate_carry, carry_ownership_setup_seconds=ownership_seconds,
        carry_ownership_scope=('INDEPENDENT_COPY_PER_LEAF_BEFORE_WARMUP_AND_MEASUREMENT'
            if donate_carry else 'LEGACY_NO_DONATION'),
        max_requests_per_game=max_requests,
        warmup_summary=summary(warm, seconds=warm_seconds, batch=batch,
            compile_seconds=compilation, warmup_seconds=0),
        requested_views_producer_error_denominator=result['actual_requested_views'])
    return result
