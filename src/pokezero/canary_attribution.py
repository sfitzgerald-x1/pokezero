"""Finite canary-versus-9375 attribution; no dispatch or training on import.

The caller binds both heads, the frozen engine checkout, original public
tables and a declared seat-swapped roster. Engine and Showdown evidence remain
separate conditions. Refusals and interruptions never become observed draws.
"""
import math
import random
import time


def _quantile(values, p):
    values = sorted(values)
    at = (len(values) - 1) * p
    lo, hi = math.floor(at), math.ceil(at)
    return values[lo] + (values[hi] - values[lo]) * (at - lo)


def summarize(rows, *, pairs, bootstrap_seed, replicates=10000):
    """One fixed two-seat pair per seed; unknown scores retain both extrema."""
    if type(pairs) is not int or pairs < 1000 or type(replicates) is not int or replicates < 1:
        raise ValueError('at least1000 declared seat-swapped pairs required')
    if len(rows) != 2 * pairs:
        raise ValueError('every original cell, including unknowns, required')
    grouped = []
    for index in range(pairs):
        seats = rows[2 * index:2 * index + 2]
        if [(r['pair'], r['candidate_seat']) for r in seats] != [(index, 0), (index, 1)]:
            raise ValueError('original pair/seat order required exactly once')
        scores = []
        for row in seats:
            if type(row['pair']) is not int or type(row['candidate_seat']) is not int:
                raise ValueError('literal integer pair/seat coordinates required')
            if row['status'] not in ('COMPLETE','CAPPED','REFUSED','FAILED','INTERRUPTED_UNKNOWN','NOT_ATTEMPTED'):
                raise ValueError('explicit observed or unknown disposition required')
            value = row.get('score')
            if row['status'] == 'COMPLETE':
                if type(value) not in (int, float) or value not in (0., .5, 1.):
                    raise ValueError('literal terminal win/draw/loss score required')
            elif value is not None:
                raise ValueError('unknown cell cannot carry a score')
            scores.append(value)
        grouped.append(scores)
    known = [x for group in grouped for x in group if x is not None]
    complete = [sum(g) / 2 for g in grouped if None not in g]
    lower = [sum(0. if x is None else x for x in g) / 2 for g in grouped]
    upper = [sum(1. if x is None else x for x in g) / 2 for g in grouped]
    rng = random.Random(bootstrap_seed)
    lows, highs = [], []
    for _ in range(replicates):
        draw = [rng.randrange(pairs) for _ in range(pairs)]
        lows.append(sum(lower[i] for i in draw) / pairs)
        highs.append(sum(upper[i] for i in draw) / pairs)
    interval = [_quantile(lows, .05), _quantile(highs, .95)]
    return dict(declared_pairs=pairs, declared_games=2 * pairs,
                known_games=len(known), unknown_games=2 * pairs - len(known),
                complete_pairs=len(complete), unknown_pairs=pairs-len(complete),
                completed_only_score=None if not known else sum(known) / len(known),
                complete_pair_score=None if not complete else sum(complete) / len(complete),
                identification_bounds=[sum(lower) / pairs, sum(upper) / pairs],
                paired_seed_90_missing_cell_envelope=interval,
                direction='WIN' if interval[0] > .5 else 'LOSS' if interval[1] < .5 else 'UNRESOLVED',
                bootstrap_seed=bootstrap_seed, bootstrap_replicates=replicates,
                equivalence_established=False)


def attribute(engine, showdown, *, foulplay_failed):
    """A Foul Play loss alone is not evidence of engine fidelity failure."""
    a, b = engine['direction'], showdown['direction']
    if type(foulplay_failed) is not bool or any(x not in ('WIN','LOSS','UNRESOLVED') for x in (a,b)):
        raise ValueError('actual closed direct comparisons and typed FoulPlay verdict required')
    # Scott applies attribution only to a failed fidelity canary. A passing
    # canary selects V2 regardless of the descriptive direct comparisons.
    if not foulplay_failed:
        return dict(cause='FOULPLAY_CANARY_NOT_FAILED', training_route='V2')
    if a == 'WIN' and b == 'LOSS':
        return dict(cause='ENGINE_OR_OBSERVATION_SKEW', training_route='PYTHON_A_B')
    if a == b == 'LOSS':
        return dict(cause='FINETUNE_DEGRADATION', training_route='V2')
    if a == b == 'WIN' and foulplay_failed:
        return dict(cause='SELFPLAY_DRIFT', training_route='V2')
    return dict(cause='UNRESOLVED_DIRECT_COMPARISONS', training_route=None)


def make_engine_duel(prepared, config, *, max_requests=4096, chunk_steps=8):
    """Same frozen device step/trackers/encoder, two deterministic public heads.

    This module is loaded alongside the externally pinned engine checkout.
    It changes neither engine code nor tracker knowledge. An initial physical
    state is duplicated for the two seats before calling this chunk.
    """
    import jax
    import jax.numpy as j
    from gpu_engine import champion, transition, engine_device_trackers as trackers
    from gpu_engine import engine_device_observation as encoder
    from gpu_engine import benchmark_device_selfplay as sampling
    from gpu_engine.b4_device_canary_collection import Carry
    if config.tokens != 23 or config.numeric != 132 or config.categorical != 41:
        raise ValueError('original champion23 config required')
    if type(max_requests) is not int or max_requests <= 0 or type(chunk_steps) is not int or chunk_steps <= 0:
        raise ValueError('finite declared request cap and chunk required')
    table, observations = prepared.engine_tables, prepared.observation_tables

    def advance(c, candidate_params, champion_params, candidate_seats):
        state = c.state; batch = len(c.streams); live = c.status == 0
        state = state._replace(done=state.done | ~live)
        requested = sampling.request_seats(state) & live[:, None]
        masks = transition.action_mask(state, table)
        views = [encoder.encode(state, trackers.readout(c.public, j.full(batch, side, j.int32)),
                    observations, observer_sides=j.full(batch, side, j.int32), legal_action_mask=masks[:, side])
                 for side in (0, 1)]
        inputs = [encoder.champion_inputs(view) for view in views]
        paired = {name: j.stack([x[name] for x in inputs], axis=1) for name in inputs[0]}
        candidate = champion.forward(candidate_params, **paired, c=config, compute_dtype='bfloat16')
        control = champion.forward(champion_params, **paired, c=config, compute_dtype='bfloat16')
        pick = j.arange(2)[None, :] == candidate_seats[:, None]
        logits = j.where(pick[..., None], candidate.policy_logits, control.policy_logits)
        value = j.where(pick, candidate.value, control.value)
        opponent = j.where(pick[..., None], candidate.opponent_action_logits, control.opponent_action_logits)
        finite = j.all(j.isfinite(logits), -1) & j.isfinite(value) & j.all(j.isfinite(opponent), -1)
        legal = j.stack([v.legal_action_mask for v in views], 1)
        errors = j.stack([v.error for v in views], 1)
        bad = requested & ((errors != 0) | ~finite | ~j.any(legal, -1))
        refused = live & j.any(bad, 1)
        actions = j.argmax(j.where(legal, j.nan_to_num(logits), -j.inf), -1).astype(j.int32)
        actual = j.stack([encoder.map_actions(views[side], state, j.full(batch, side, j.int32), actions[:, side])
                          for side in (0, 1)], 1)
        result = transition.step_with_public(state._replace(done=state.done | refused), actual,
                    table, c.public, belief_tables=observations.belief)
        after = result.step.state
        refused |= live & result.step.invalid_actions
        complete = live & ~refused & after.done
        frames = c.frames + live.astype(j.int32)
        capped = live & ~refused & ~complete & (frames >= max_requests)
        status = j.where(refused, 3, j.where(complete, 1, j.where(capped, 2, c.status)))
        carry = Carry(after, result.trackers, c.streams, frames, status,
                      j.where(complete, after.winner, c.winner))
        return carry, dict(requested=j.sum(requested), producer_errors=j.sum(bad),
                           invalid_games=j.sum(live & result.step.invalid_actions))

    return jax.jit(lambda c, candidate, champion, seats:
        jax.lax.scan(lambda x, _: advance(x, candidate, champion, seats), c, None, length=chunk_steps))


def showdown_game(env, *, factories, runtimes, seed, stream, candidate_seat,
                  max_requests=4096, decision_seconds=60., on_decision=lambda **kw: None,
                  on_oracle=None, start_override=None):
    """Both v4 models act on B2 owner streams; no Foul Play process is used.

    The owning supervisor must enforce on_decision's deadline for a blocked
    library call. A returned late action is rejected here as well. Optional
    simulator snapshots are delivered only to a separate validation sink AFTER
    both public actions are selected and are never passed to either actor.
    """
    from .v5_python_collection import OwningEnv, _terminal_winner
    if not isinstance(env, OwningEnv) or set(factories) != {0, 1} or set(runtimes) != {0, 1}:
        raise ValueError('literal owning streams and two separately bound raw factories required')
    if candidate_seat not in (0, 1) or decision_seconds != 60.:
        raise ValueError('declared candidate seat and original60-second learner required')
    actors = {}; positions = {0: 0, 1: 0}; frames = 0
    def feed():
        for side, actor in actors.items():
            seat = 'p' + str(side + 1)
            lines = tuple(env.owner_lines[seat][positions[side]:])
            if lines:
                actor.accept_owner_stream(battle_id=actor.battle_id, owner=seat,
                    generation=actor.generation, prefix_sha256=actor.prefix_sha256, lines=lines)
                positions[side] += len(lines)
    try:
        if start_override is None:env.reset(seed=seed, format_id='gen3randombattle')
        else:env.reset_with_start_override(seed=seed, start_override=start_override)
        battle_id = 'canary-attribution-' + str(stream)
        for side in (0, 1):
            actor = factories[side].actor(runtimes[side]); actors[side] = actor
            actor.begin_game(battle_id=battle_id, owner='p' + str(side+1), seed=seed, stream_id=stream)
        while frames < max_requests and env.terminal() is None:
            feed(); decisions = {}; choices = {}
            for seat in env.requested_players():
                side = int(seat[1])-1; actor = actors[side]; snapshot = actor.snapshot_boundary()
                begin = time.monotonic(); on_decision(frame=frames, side=side, deadline_epoch=time.time()+decision_seconds)
                decision = actor.decide(snapshot, rng=random.Random(seed ^ stream ^ (frames << 1) ^ side))
                if time.monotonic()-begin > decision_seconds:
                    raise TimeoutError('original learner decision limit exceeded')
                decisions[seat] = decision.action_index; choices[seat] = actor.choice(snapshot)
            if not decisions or env._choices_for_actions(decisions) != choices:
                raise ValueError('owning B2 decision routing disagrees with original Showdown request')
            if on_oracle is not None:
                on_oracle(frame=frames, snapshot=env.snapshot(), owner_lines={s: tuple(env.owner_lines[s]) for s in ('p1','p2')})
            env.step(decisions); frames += 1
        terminal = env.terminal()
        if terminal is None or terminal.capped:
            return dict(status='CAPPED', score=None, frames=frames, reason='original finite request/turn cap')
        feed()
        markers = [tuple(x for x in env.owner_lines[s] if x.startswith('|win|') or x in ('|tie','|tie|')) for s in ('p1','p2')]
        if len(markers[0]) != 1 or markers[0] != markers[1]:
            raise ValueError('literal two owning terminal streams must agree')
        winner = _terminal_winner(env.owner_lines, markers[0][0], terminal)
        score = .5 if winner == 2 else float(winner == candidate_seat)
        return dict(status='COMPLETE', score=score, frames=frames, terminal_marker=markers[0][0], winner=winner)
    finally:
        for actor in actors.values():actor.close_game()
