"""Actual-device candidate replay of one frozen fresh200 prefix universe.

The candidate initializes and advances its own Trackers from the saved raw
constructor. Integration Trackers are read only to verify the recorded prefix
chain; they are never passed to candidate initialize/step/encode. No actor,
policy, reference generation, extra chance draw or root replacement occurs.
"""
import hashlib
import json
import re

import numpy as np


def verify_capture(root, reference, *, source, capture, attestation):
    """Check the sealed DAG and the exact, predeclared distinct-root roster."""
    visited = set();active = set()

    def visit(ref):
        key = (ref['path'], ref['sha256'])
        if key in active:raise ValueError('cyclic capture attestation')
        if key in visited:return
        if len(visited)+len(active) >= 1024:raise ValueError('capture attestation exceeds bound')
        active.add(key);document = json.loads(attestation.verify_artifact(root, ref))
        if document.get('schema') != attestation.SCHEMA or document.get('source') != source:
            raise ValueError('capture DAG source/schema differs')
        if document.get('kind') == 'manifest':
            for parent in document['parents']:visit(parent)
        elif document.get('kind') == 'snapshot':attestation.verify_artifact(root, document['arrays'])
        else:raise ValueError('unknown capture DAG node')
        active.remove(key);visited.add(key)

    visit(reference)
    document = json.loads(attestation.verify_artifact(root, reference));payload = document['payload']
    if (payload.get('schema') != capture.SCHEMA or payload.get('status') != 'CAPTURED_ONLY'
            or payload.get('passed') is not False or payload.get('selection_error') is not None
            or payload.get('selection_policy') != capture.selection_policy()):
        raise ValueError('incomplete or changed fresh-corpus capture')
    if payload['selected'] != capture.select_roots(payload['addresses']):raise ValueError('sealed root roster changed')
    position = 0
    for chunk in payload['chunks']:
        if chunk['start_step'] != position or type(chunk['steps']) is not int or not 0 < chunk['steps'] <= capture.CHUNK_STEPS:
            raise ValueError('captured prefix has a gap or changed chunk bound')
        position += chunk['steps']
    if position != payload['actual_horizon'] or not capture.FIRST_FRAME < position <= capture.MAX_STEPS:
        raise ValueError('captured prefix horizon differs')
    return payload


def input_template(state, public, *, rows, chance_type):
    """Trusted runtime types/shapes only; all values are replaced on load."""
    chance = chance_type(np.zeros((rows, 2, 12), np.int32), np.zeros((rows, 2), np.int32),
        np.zeros((rows, 2), np.int32), np.zeros((rows, 2), np.int32), np.zeros((rows, 2), np.int32),
        np.zeros(rows, np.int32), np.zeros((rows, 2, 5), np.int32))
    return dict(pre_state=state, pre_public=public, step_state=state, post_state=state, post_public=public,
        actions=np.zeros((rows, 2), np.int32), requested=np.zeros((rows, 2), bool),
        streams=np.zeros(rows, np.int32), frames=np.zeros(rows, np.int32), status=np.zeros(rows, np.int32),
        chance=chance, actor_keys=np.zeros((rows, 2, 2), np.uint32))


def constructor_template(state, *, rows):
    # Capture used jax.device_get(dict(...)), whose dictionary tree order is
    # sorted. Keep that recorded order without relaxing transport validation.
    return dict(initial_tie=np.zeros(rows, np.int32), raw=state, streams=np.zeros(rows, np.int32))


def scanned_template(template, *, steps):
    """Preserve the replay-input mapping order while adding the scan axis.

    Capture created replay_inputs AFTER device_get. Mapping this outer dict
    with JAX would sort it again and change the sealed descriptor's key order.
    Values already came through JAX; mapping each value preserves their trees.
    """
    import jax
    return {name:jax.tree.map(lambda value: np.broadcast_to(np.asarray(value), (steps, *value.shape)), value)
        for name, value in template.items()}


def replay(root, reference, *, prepared, integration_source, candidate_source, capture, attestation, runtime_parent=None,
        artifact_prefix='candidate'):
    """Save candidate device observations and complete corrected history.

    Successful replay is not Python/B2 parity. Every captured State field must
    agree physically; any divergence is sealed and stops this replay. Native
    same-prefix physical and literal-owner reference gates stay open.
    """
    import jax
    devices = jax.devices()
    if len(devices) != 1 or devices[0].platform != 'gpu':raise ValueError('one actual GPU required for candidate observations')
    if not re.fullmatch('[a-z0-9][a-z0-9_-]{0,47}', artifact_prefix):raise ValueError('explicit bounded replay artifact prefix required')
    import jax.numpy as j
    from gpu_engine import transition, state as state_module, v5_device_rollout as rollout
    from gpu_engine import engine_device_observation as encoder, engine_device_trackers as trackers
    from gpu_engine import benchmark_device_selfplay as sampling
    if runtime_parent is None:raise ValueError('sealed candidate runtime identity required')
    runtime = json.loads(attestation.verify_artifact(root, runtime_parent))
    if (runtime.get('schema') != attestation.SCHEMA or runtime.get('kind') != 'manifest'
            or runtime.get('source') != candidate_source
            or hashlib.sha256((json.dumps(runtime['payload'], sort_keys=True, separators=(',', ':'), allow_nan=False)+'\n').encode()).hexdigest()
                != candidate_source['runtime_sha256']):
        raise ValueError('sealed candidate runtime identity differs')
    payload = verify_capture(root, reference, source=integration_source, capture=capture, attestation=attestation)
    rows = capture.GAMES
    # empty_state supplies only the trusted complete State schema. Loading
    # replaces EVERY leaf, including keys/genders; it is never a played root.
    constructor = attestation.load_snapshot(root, payload['constructor_input'], constructor_template(state_module.empty_state(rows), rows=rows),
        expected_source=integration_source)
    initialize = jax.jit(lambda raw, tie: transition.initialize_with_public(raw, prepared.engine_tables,
        belief_tables=prepared.observation_tables.belief, tie_coin=tie))
    initialized = jax.block_until_ready(initialize(constructor['raw'], constructor['initial_tie']))
    original = attestation.load_snapshot(root, payload['constructor_output'], initialized, expected_source=integration_source)
    state, public = initialized.state, initialized.trackers
    roots = {};parents = [reference, runtime_parent];completed_steps = 0;phase = 'initial State';current = None
    originals_state, originals_public = original.state, original.trackers
    selected = {(row['step'], row['row'], row['owner']): row for row in payload['selected']}
    if len(selected) != capture.ROOTS:raise ValueError('duplicate selected device root')
    fields = ('numeric_features', 'categorical_ids', 'token_type_ids', 'attention_mask', 'legal_action_mask', 'error')

    def advance(state, public, done, actions, chance):
        masks = transition.action_mask(state, prepared.engine_tables)
        observations = [encoder.encode(state, trackers.readout(public, j.full(rows, side, j.int32)),
            prepared.observation_tables, observer_sides=j.full(rows, side, j.int32), legal_action_mask=masks[:, side])
            for side in (0, 1)]
        observed = {name: j.stack([getattr(value, name) for value in observations], axis=1) for name in fields}
        out = transition.step_with_public(state._replace(done=done), actions, prepared.engine_tables, public,
            belief_tables=prepared.observation_tables.belief, chance_inputs=chance)
        return out, observed, sampling.request_seats(state)

    execute = jax.jit(advance)
    try:
        capture.assert_tree_bits(original.state, state, phase)
        for chunk in payload['chunks']:
            template = input_template(state, public, rows=rows, chance_type=transition.ChanceInputs)
            template = scanned_template(template, steps=chunk['steps'])
            records = attestation.load_snapshot(root, chunk['replay_inputs'], template, expected_source=integration_source)
            candidate_records = []
            for offset in range(chunk['steps']):
                step = chunk['start_step']+offset;current = dict(step=step, chunk=chunk['replay_inputs'])
                record = jax.tree.map(lambda value: value[offset], records)
                phase = 'recorded prefix continuity'
                capture.assert_tree_bits(originals_state, record['pre_state'], phase+'.State')
                capture.assert_tree_bits(originals_public, record['pre_public'], phase+'.Trackers')
                capture.assert_tree_bits(constructor['streams'], record['streams'], 'recorded stream identities')
                phase = 'candidate PRE State'
                capture.assert_tree_bits(record['pre_state'], state, phase)
                # The stock collector's recorded stop/refusal mask is part of
                # the fixed replay input, not a new policy/actor decision.
                phase = 'candidate STEP input State'
                step_state = state._replace(done=record['step_state'].done)
                capture.assert_tree_bits(record['step_state'], step_state, phase)
                before_state, before_public = state, public
                result, observations, requested = jax.block_until_ready(execute(state, public,
                    record['step_state'].done, record['actions'], record['chance']))
                phase = 'requested owner identities'
                capture.assert_tree_bits(record['requested'], np.asarray(requested) & (record['status']==rollout.RUNNING)[:, None], phase)
                state, public = result.step.state, result.trackers
                phase = 'candidate POST State'
                capture.assert_tree_bits(record['post_state'], state, phase)
                host = jax.device_get(observations)
                for row in range(rows):
                    for owner in (0, 1):
                        address = selected.get((step, row, owner))
                        if address is None:continue
                        if (int(record['streams'][row]) != address['stream_id'] or int(record['frames'][row]) != address['frame']
                                or not bool(record['requested'][row, owner])):
                            raise ValueError('selected literal device address diverged')
                        roots[address['root_index']] = {name: host[name][row, owner].copy() for name in fields}
                candidate_records.append(jax.device_get(dict(pre_state=before_state, pre_public=before_public,
                    post_state=state, post_public=public)))
                originals_state, originals_public = record['post_state'], record['post_public'];completed_steps += 1
            arrays = jax.tree.map(lambda *values: np.stack(values), *candidate_records)
            receipt = attestation.save_snapshot(root, artifact_prefix+'-prefix-%03d' % chunk['start_step'], arrays, source=candidate_source)
            link = attestation.seal_manifest(root, artifact_prefix+'-prefix-%03d-link' % chunk['start_step'],
                dict(start_step=chunk['start_step'], steps=chunk['steps'], physical_State='all fields bitwise equal',
                    Trackers='candidate initialize/step only; original history never supplied'), parents=[*parents, receipt], source=candidate_source)
            parents = [link]
        if sorted(roots) != list(range(capture.ROOTS)):raise ValueError('missing selected actual-device observations; no replacement')
        observations = {name: np.stack([roots[index][name] for index in range(capture.ROOTS)]) for name in fields}
        observation_ref = attestation.save_snapshot(root, artifact_prefix+'-observations', observations, source=candidate_source)
        return attestation.seal_manifest(root, artifact_prefix, dict(status='DEVICE_REPLAYED_ONLY', passed=False,
            selected=payload['selected'], roots=capture.ROOTS, owner_counts=[capture.P1_ROOTS, capture.ROOTS-capture.P1_ROOTS],
            completed_steps=completed_steps, observation_errors=int(np.count_nonzero(observations['error'])),
            observations=observation_ref, device=str(devices[0]), physical_State='complete bitwise agreement',
            history='candidate continuing Trackers, never copied from integration snapshots',
            open_gates=['strict Native same-prefix physical replay', 'literal owning Python/B2 references', 'strict actual-device parity']),
            parents=[*parents, observation_ref], source=candidate_source)
    except ValueError as error:
        failed = attestation.save_snapshot(root, artifact_prefix+'-failure-state', jax.device_get(dict(state=state, public=public)), source=candidate_source)
        return attestation.seal_manifest(root, artifact_prefix+'-failure', dict(status='FAILED', passed=False,
            phase=phase, detail=str(error), current=current, completed_steps=completed_steps,
            measured_selected_roots=len(roots), refill=False), parents=[*parents, failed], source=candidate_source)
