"""Bounded fresh-corpus capture around an unchanged stock device collector.

This is capture evidence, never a parity qualification. The caller supplies a
clean, pinned integration runtime, its original plan/actor, and a sealed launch
declaration. Native reference and candidate replay are separate consumers of
the recorded prefixes. There is no refill or selection by comparison result.
"""
from dataclasses import replace
import json

import numpy as np

SCHEMA = 'pokezero.fresh200.capture.v1'
GAMES = 256
MAX_STEPS = 64
FIRST_FRAME = 8
ROOTS = 200
P1_ROOTS = 107
CHUNK_STEPS = 4


def selection_policy():
    return dict(games=GAMES, max_steps=MAX_STEPS, first_pre_frame=FIRST_FRAME,
        roots=ROOTS, owner_counts=[P1_ROOTS, ROOTS-P1_ROOTS],
        order='frame,original_ordinal', physical_address='frame,original_ordinal',
        owner_schedule='p1 iff floor((i+1)*107/200)>floor(i*107/200)',
        eligibility='PRE RUNNING and requested owner only; no error/parity filter',
        refill=False)


def leading_batch(plan):
    """Keep the actual leading rows and all original RNG coordinates."""
    batch = plan.iterations[0][0]
    if len(batch.games) < GAMES:raise ValueError('original first batch has fewer than 256 games')
    if type(plan.max_requests) is not int or plan.max_requests <= FIRST_FRAME:
        raise ValueError('original request horizon cannot supply declared PRE frames')
    return replace(batch, games=batch.games[:GAMES], gender_root_ids=batch.gender_root_ids[:GAMES])


def original_assignments(plan, batch, snapshot, events, runs):
    """Require the recorded first full bank, then preserve its leading slice."""
    first = plan.iterations[0][0]
    streams = [game.stream_id for game in first.games]
    matches = [event['payload']['assignments'] for event in events
        if event['kind'] == 'BATCH_STARTED' and list(event['payload']['streams']) == streams]
    if len(matches) != 1:raise ValueError('one original first-bank assignment receipt required')
    ordinals = {game.stream_id: i for i, game in enumerate(plan.games)}
    if len(ordinals) != len(plan.games):raise ValueError('original plan repeats stream identities')
    assignments = [runs.assignment(plan, game, ordinals[game.stream_id], snapshot, ()) for game in batch.games]
    recorded = json.loads(runs._json(assignments))
    if recorded != matches[0][:GAMES]:raise ValueError('original leading actor assignments differ')
    return recorded, [ordinals[game.stream_id] for game in batch.games]


def select_roots(addresses):
    """Select exactly the predeclared 200 physical roots, or fail without refill.

    Every address, including refused/error-bearing views, is enumerated before
    this function. Eligibility does not inspect observation or parity fields.
    Both owners at one physical address cannot count as two distinct roots.
    """
    rows = sorted(addresses, key=lambda row: (row['frame'], row['original_ordinal']))
    seen = set()
    for row in rows:
        address = (row['frame'], row['original_ordinal'])
        if address in seen:raise ValueError('duplicate physical address')
        seen.add(address)
        if (type(row['frame']) is not int or not 0 <= row['frame'] < MAX_STEPS
                or type(row['original_ordinal']) is not int or row['original_ordinal'] < 0
                or type(row['live']) is not bool or len(row['requested']) != 2
                or any(type(value) is not bool for value in row['requested'])):
            raise ValueError('malformed physical address')
    used = set();selected = []
    for index in range(ROOTS):
        owner = 0 if (index+1)*P1_ROOTS//ROOTS > index*P1_ROOTS//ROOTS else 1
        row = next((row for row in rows if row['live'] and row['frame'] >= FIRST_FRAME
            and row['requested'][owner] and (row['frame'], row['original_ordinal']) not in used), None)
        if row is None:raise ValueError('declared universe lacks 200 roots with the fixed owner schedule')
        used.add((row['frame'], row['original_ordinal']))
        selected.append(dict(row, owner=owner, root_index=index))
    return selected


def assert_tree_bits(expected, actual, path='root'):
    """Full tree/None/dtype/shape/bit equality; no numeric tolerance."""
    if expected is None or actual is None:
        if expected is not actual:raise ValueError('tree None mismatch at '+path)
        return
    if isinstance(expected, tuple) and hasattr(expected, '_fields'):
        if type(expected) is not type(actual) or expected._fields != actual._fields:
            raise ValueError('tree fields mismatch at '+path)
        for name in expected._fields:assert_tree_bits(getattr(expected, name), getattr(actual, name), path+'.'+name)
    elif isinstance(expected, dict):
        if not isinstance(actual, dict) or list(expected) != list(actual):raise ValueError('tree keys mismatch at '+path)
        for name in expected:assert_tree_bits(expected[name], actual[name], path+'.'+name)
    elif isinstance(expected, (tuple, list)):
        if type(expected) is not type(actual) or len(expected) != len(actual):raise ValueError('tree sequence mismatch at '+path)
        for index, (old, new) in enumerate(zip(expected, actual)):assert_tree_bits(old, new, path+'.'+str(index))
    else:
        old, new = np.asarray(expected), np.asarray(actual)
        if old.dtype != new.dtype or old.shape != new.shape or old.tobytes() != new.tobytes():
            raise ValueError('tree bits mismatch at '+path)


def replay_inputs(record):
    """Retain complete physical/history trees without an actor-shaped template.

    The original complete collector record is retained separately. These are
    references to its actual arrays, not reconstructed history or observations.
    Leading scan dimensions, every NamedTuple field and None are unchanged.
    """
    return dict(pre_state=record['pre'].state, pre_public=record['pre'].public,
        step_state=record['step_state'], post_state=record['post'].state, post_public=record['post'].public,
        actions=record['frames'].engine_actions, requested=record['frames'].requested,
        streams=record['pre'].streams, frames=record['pre'].frames, status=record['pre'].status,
        chance=record['chance'], actor_keys=record['actor_keys'])


def make_capture(backend, bindings, bank, model, *, sampling_seed, max_requests):
    """Materialize stateless chance coordinates beside the stock collector.

    The collector still owns actor inference, sampling and all numerical
    transitions. The recorded ChanceInputs are independently materialized at
    the same input state, not obtained by replacing a transition function.
    An explicit-chance replay below must reproduce every recorded POST bit.
    """
    import jax
    import jax.numpy as j
    from gpu_engine import transition, v5_device_rollout as rollout
    from gpu_engine import benchmark_device_selfplay as sampling
    prepared = backend.prepared
    one = rollout.make_collector(prepared.engine_tables, prepared.observation_tables,
        model, backend.labels, bindings, sampling_seed=sampling_seed, chunk_steps=1,
        max_requests=max_requests, actor_bank=bank)

    def advance(carry, params):
        out = one(carry, params)
        frames = jax.tree.map(lambda value: value[0], out.frames)
        live = carry.status == rollout.RUNNING
        valid = (frames.actor_finite & (frames.errors == 0) & j.any(frames.legal_action_mask, axis=-1)
            & j.isfinite(frames.probabilities) & (frames.probabilities > 0))
        refused = live & (j.any(frames.requested & ~valid, axis=1)
            | (carry.streams != j.asarray(bank.streams, j.int32)))
        step_state = carry.state._replace(done=carry.state.done | ~live | refused)
        chance = transition.sample_chance_inputs(step_state)
        actor_keys = sampling.sample_keys(carry.streams, carry.frames, sampling_seed=sampling_seed)
        return out.carry, dict(pre=carry, step_state=step_state, post=out.carry,
            frames=frames, chance=chance, actor_keys=actor_keys)

    def chunk(carry, params, *, steps):
        return jax.lax.scan(lambda current, _: advance(current, params), carry, None, length=steps)

    return jax.jit(chunk, static_argnames=('steps',))


def collect_capsule(root, *, context, snapshot, params, events, source, launch, attestation):
    """Record one finite prefix universe; never run scoring/finalization.

    Launch is a previously sealed attestation manifest with this exact policy
    and source. A captured capsule deliberately leaves parity gates OPEN.
    """
    import jax
    import jax.numpy as j
    from pokezero import v5_run as runs
    from gpu_engine import transition, v5_device_rollout as rollout
    from gpu_engine.v5_device_backend import RuntimeBackend
    launch_doc = json.loads(attestation.verify_artifact(root, launch))
    if (launch_doc.get('schema') != attestation.SCHEMA or launch_doc.get('kind') != 'manifest'
            or launch_doc.get('source') != source
            or launch_doc.get('payload', {}).get('selection_policy') != selection_policy()):
        raise ValueError('sealed launch source/selection policy differs')
    plan = context['plan'];prepared = context['prepared'];batch = leading_batch(plan)
    assignments, ordinals = original_assignments(plan, batch, snapshot, events, runs)
    backend = RuntimeBackend(prepared, context['pool'], gender_policy=context['gender'])
    backend.check_plan(plan)
    bindings = rollout.Bindings(plan.model.category_vocabulary_sha256, plan.model.label_vocabulary,
        prepared.source_sha256, (snapshot.archive_sha256,)*2, (True, True), backend.declaration_sha256)
    bank = rollout.actor_bank_from_assignments(bindings, (snapshot.archive_sha256,),
        current_archive=snapshot.archive_sha256, assignments=assignments)
    captured = {};initializer = backend.initialize

    def retain_initial(raw):
        captured['raw'] = raw
        captured['initialized'] = initializer(raw)
        return captured['initialized']

    backend.initialize = retain_initial
    try:carry = jax.block_until_ready(backend._construct(batch))
    finally:backend.initialize = initializer
    try:
        if tuple(np.asarray(carry.streams).tolist()) != bank.streams:
            raise ValueError('constructed bank stream order differs')
        initial_tie = jax.vmap(lambda key: jax.random.randint(jax.random.fold_in(key, 314159), (), 0, 2))(captured['raw'].keys)
        explicit_initial = jax.jit(lambda raw, tie: transition.initialize_with_public(raw,
            prepared.engine_tables, belief_tables=prepared.observation_tables.belief, tie_coin=tie))
        initial_replayed = jax.block_until_ready(explicit_initial(captured['raw'], initial_tie))
        assert_tree_bits(captured['initialized'], initial_replayed, 'initial')
        initial = attestation.save_snapshot(root, 'constructor', jax.device_get(dict(captured,
            initial_tie=initial_tie, carry=carry)), source=source)
        initial_input = attestation.save_snapshot(root, 'constructor-input', jax.device_get(dict(
            raw=captured['raw'], initial_tie=initial_tie, streams=carry.streams)), source=source)
        initial_output = attestation.save_snapshot(root, 'constructor-output',
            jax.device_get(captured['initialized']), source=source)
        games = [dict(original_ordinal=ordinal, stream_id=game.stream_id, battle_id=game.battle_id,
            seed=game.seed, team_indices=list(game.team_indices)) for ordinal, game in zip(ordinals, batch.games)]
        construction = attestation.seal_manifest(root, 'construction', dict(games=games,
            constructor_seed=batch.constructor_seed, sampling_seed=batch.sampling_seed,
            gender_seed=batch.gender_seed, gender_domain=batch.gender_domain,
            gender_root_ids=list(batch.gender_root_ids), assignments=assignments,
            assignment_sha256=bank.assignment_sha256, actor_archive_sha256=snapshot.archive_sha256,
            actor_checkpoint_sha256=snapshot.checkpoint_sha256, declaration_sha256=backend.declaration_sha256,
            initial_explicit_chance_replay='all State/Trackers bits equal'),
            parents=[launch, initial, initial_input, initial_output], source=source)
        execute = make_capture(backend, bindings, bank, plan.model, sampling_seed=batch.sampling_seed,
            max_requests=plan.max_requests)
        replay = jax.jit(lambda state, public, actions, chance: transition.step_with_public(state, actions,
            prepared.engine_tables, public, belief_tables=prepared.observation_tables.belief, chance_inputs=chance))
        horizon = min(MAX_STEPS, plan.max_requests)
        parents = [construction];chunks = [];addresses = [];previous = jax.device_get(carry)
        for start in range(0, horizon, CHUNK_STEPS):
            steps = min(CHUNK_STEPS, horizon-start)
            carry, records = jax.block_until_ready(execute(carry, (params,), steps=steps))
            host = jax.device_get(records)
            for offset in range(steps):
                record = jax.tree.map(lambda value: value[offset], host)
                assert_tree_bits(previous, record['pre'], 'continuity')
                if record['chance'].effect is None:raise ValueError('complete seven-field ChanceInputs required')
                checked = jax.block_until_ready(replay(record['step_state'], record['pre'].public,
                    record['frames'].engine_actions, record['chance']))
                assert_tree_bits(record['post'].state, checked.step.state, 'explicit POST State')
                assert_tree_bits(record['post'].public, checked.trackers, 'explicit POST Trackers')
                for row, game in enumerate(games):
                    if int(record['pre'].status[row]) != rollout.RUNNING:continue
                    addresses.append(dict(game, row=row, step=start+offset, frame=int(record['pre'].frames[row]),
                        live=True, requested=[bool(value) for value in record['frames'].requested[row]],
                        observation_errors=[int(value) for value in record['frames'].errors[row]],
                        next_status=rollout.STATUS[int(record['post'].status[row])]))
                previous = record['post']
            chunk_ref = attestation.save_snapshot(root, 'prefix-%03d' % start, host, source=source)
            inputs_ref = attestation.save_snapshot(root, 'prefix-%03d-inputs' % start, replay_inputs(host), source=source)
            chunk = attestation.seal_manifest(root, 'prefix-%03d-link' % start,
                dict(start_step=start, steps=steps, explicit_replay='all State/Trackers POST bits equal'),
                parents=[*parents, chunk_ref, inputs_ref], source=source)
            chunks.append(dict(start_step=start, steps=steps, arrays=chunk_ref, replay_inputs=inputs_ref, link=chunk))
            parents = [chunk]
        selected = [];selection_error = None
        try:selected = select_roots(addresses)
        except ValueError as error:selection_error = str(error)
        dispositions = [dict(game, status=rollout.STATUS[int(previous.status[row])],
            last_frame=int(previous.frames[row])) for row, game in enumerate(games)]
        return attestation.seal_manifest(root, 'capture', dict(schema=SCHEMA,
            status='CAPTURED_ONLY' if selection_error is None else 'INCOMPLETE', passed=False,
            selection_policy=selection_policy(), actual_horizon=horizon, selected=selected,
            selection_error=selection_error, addresses=addresses, dispositions=dispositions,
            chunks=chunks, constructor=initial, constructor_input=initial_input, constructor_output=initial_output,
            chance_recording='stateless same-coordinate materialization beside unchanged stock collector; explicit replay verified',
            open_gates=['candidate continuing State/Trackers replay', 'strict Native same-prefix physical replay',
                'literal owning Python/B2 references', 'actual-device strict parity']), parents=parents, source=source)
    finally:backend.close()
