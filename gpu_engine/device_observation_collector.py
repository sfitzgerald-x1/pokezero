"""Prospective finite public29 parity collection, never a selfplay dependency.

The legal policy is independent of both encoders. A current PRE physical
boundary determines eligibility; its subsequent POST/chance result controls
only the next boundary. Protocol refusals are data, not trajectory filters.
All large evidence is written beside the explicitly remote output directory.
No model, team generator, old capture trace, or event proof is used here.
"""
from collections import Counter
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import tempfile

import numpy as np

SCHEMA = 'pokezero.device-observation-collector.v1'
ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024*1024), b''): h.update(block)
    return h.hexdigest()


def _json(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)


def _hex(value, length=64):
    return type(value) is str and re.fullmatch('[0-9a-f]{'+str(length)+'}', value) is not None


def _uint(value):
    return type(value) is int and 0 <= value < 2**32


def checked_document(reference):
    if (type(reference) is not dict or set(reference) != {'path', 'sha256'}
            or type(reference['path']) is not str or not _hex(reference['sha256'])):
        raise ValueError('literal path/SHA256 reference required')
    payload = Path(reference['path']).read_bytes()
    if hashlib.sha256(payload).hexdigest() != reference['sha256']:
        raise ValueError('referenced bytes changed: '+reference['path'])
    return json.loads(payload)


def validate_plan(plan):
    required = {'schema', 'source_commit', 'pool', 'vocabulary', 'labels', 'gender_policy',
        'showdown_root', 'node', 'node_sha256', 'constructor_seed', 'sampling_seed',
        'gender_seed', 'gender_domain', 'max_requests', 'max_turns', 'cells', 'policy'}
    if type(plan) is not dict or set(plan) != required or plan['schema'] != SCHEMA:
        raise ValueError('complete explicit finite collection plan required')
    if not _hex(plan['source_commit'], 40) or not _hex(plan['node_sha256']):
        raise ValueError('source/Node identities required')
    if any(type(plan[k]) is not str or not plan[k] for k in ('showdown_root', 'node')):
        raise ValueError('actual source and Node paths required')
    for key in ('pool', 'vocabulary', 'labels', 'gender_policy'):
        ref = plan[key]
        if (type(ref) is not dict or set(ref) != {'path', 'sha256'}
                or type(ref['path']) is not str or not ref['path'] or not _hex(ref['sha256'])):
            raise ValueError('path/hash required: '+key)
    if (any(not _uint(plan[k]) for k in ('constructor_seed', 'sampling_seed', 'gender_seed', 'gender_domain'))
            or type(plan['max_requests']) is not int or not 1 <= plan['max_requests'] <= 4096
            or type(plan['max_turns']) is not int or not 1 <= plan['max_turns'] <= 1000
            or plan['policy'] != 'PCG64_SEEDSEQUENCE_UNIFORM_PHYSICAL_LEGAL_V1'):
        raise ValueError('finite limits and declared independent legal policy required')
    cells = plan['cells']
    if type(cells) is not list or not cells or len(cells) > 10000:
        raise ValueError('nonempty bounded declared order required')
    seen = {k: set() for k in ('stream_id', 'root_id', 'native_seed', 'team_index')}
    for cell in cells:
        if (type(cell) is not dict or set(cell) != {'stream_id', 'root_id', 'native_seed', 'team_indices'}
                or any(not _uint(cell[k]) for k in ('stream_id', 'root_id', 'native_seed'))
                or cell['native_seed'] >= 2**31
                or type(cell['team_indices']) is not list or len(cell['team_indices']) != 2
                or any(not _uint(x) for x in cell['team_indices'])):
            raise ValueError('literal finite cell addresses required')
        for key in ('stream_id', 'root_id', 'native_seed'):
            if cell[key] in seen[key]: raise ValueError('reused declared '+key)
            seen[key].add(cell[key])
        for index in cell['team_indices']:
            if index in seen['team_index']: raise ValueError('fresh distinct team rows required')
            seen['team_index'].add(index)
    return plan


def choose_legal(mask, requested, *, seed, stream, step):
    """Independent diagnostic RNG, never engine keys or tensor outputs."""
    mask, requested = np.asarray(mask), np.asarray(requested)
    if mask.shape != (2, 9) or mask.dtype != np.bool_ or requested.shape != (2,) or requested.dtype != np.bool_:
        raise ValueError('actual physical legal request required')
    actions = np.zeros((1, 2), np.int32)
    for side in (0, 1):
        if not requested[side]: continue
        legal = np.flatnonzero(mask[side])
        if not len(legal): raise ValueError('ENGINE_REQUEST_HAS_NO_LEGAL_ACTION')
        rng = np.random.Generator(np.random.PCG64(np.random.SeedSequence([seed, stream, step, side])))
        actions[0, side] = rng.choice(legal)
    return actions


def _prefix(previous, differences=None, *, unresolved=False):
    if previous is False or differences: return False
    if previous is None or unresolved: return None
    return True


def _unresolved(audit):
    return any(row.get('kind') in ('UNRESOLVED', 'INITIAL_RANDOM') for row in audit)


def _error(error):
    return dict(type=type(error).__name__, message=str(error)[:4000])


class Evidence:
    """Fresh remote files; arrays and physical bindings never enter stdout."""
    def __init__(self, out):
        self.root = Path(out)
        self.root.mkdir()  # no resume/reconciliation/replacement
        (self.root/'rows').mkdir()
        self.stream = (self.root/'decisions.jsonl').open('x')
        self.physical = (self.root/'physical.jsonl').open('x')
        self.events = (self.root/'journal.jsonl').open('x')
        self.counts = Counter()

    def event(self, value):
        self.events.write(_json(value)+'\n'); self.events.flush()

    def boundary(self, value):
        payload = _json(value)
        self.physical.write(payload+'\n'); self.physical.flush()
        return hashlib.sha256(payload.encode()).hexdigest()

    def decision(self, metadata, candidate, reference):
        arrays = {}
        for label, record in (('candidate', candidate), ('reference', reference)):
            if record is not None:
                for key, value in record.items(): arrays[label+'__'+key] = np.asarray(value)
        path = self.root/'rows'/(metadata['decision_id']+'.npz')
        # Binary tensors retain exact dtypes, signedness, nonfinite values.
        with path.open('xb') as stream: np.savez_compressed(stream, **arrays)
        row = dict(metadata, tensor_path=str(path.relative_to(self.root)), tensor_sha256=digest(path))
        self.stream.write(_json(row)+'\n'); self.stream.flush()
        self.counts['decisions'] += 1

    def close(self):
        for stream in (self.stream, self.physical, self.events): stream.close()


@dataclass
class Runtime:
    plan: dict
    pool: object
    original_ivs: object
    vocabulary: dict
    factory: object
    config: object
    tables: object
    observation_tables: object
    initialize: object
    observe: object
    advance: object
    sample: object
    gender_policy: dict
    bindings: dict

    def construct(self, cell):
        import jax
        import jax.numpy as j
        from . import transition, gender
        raw = transition.state_from_pool(self.pool.arrays, np.asarray([cell['team_indices']], np.int32),
            seed=self.plan['constructor_seed'])
        key = jax.random.fold_in(jax.random.PRNGKey(self.plan['constructor_seed']), j.uint32(cell['stream_id']))
        raw = raw._replace(keys=key[None])
        explicit = self.pool.arrays.get('gender')
        raw = gender.initialize_genders(raw, self.gender_policy, self.vocabulary['species'],
            seed=self.plan['gender_seed'], domain=self.plan['gender_domain'],
            root_ids=np.asarray([cell['root_id']], np.uint32),
            explicit_genders=None if explicit is None else explicit[np.asarray([cell['team_indices']])])
        initialized = jax.block_until_ready(self.initialize(raw))
        return initialized.state, initialized.trackers, raw

    def parties(self, raw, cell):
        from .finite_stream_comparison import pool_sets
        return pool_sets(self.pool, np.asarray(cell['team_indices']), np.asarray(raw.gender)[0],
            np.asarray(raw.happiness)[0], original_ivs=self.original_ivs)

    def boundary(self, state):
        from .fidelity import requested_seats
        return dict(state={k: np.asarray(getattr(state, k)).tolist() for k in state._fields},
            requested=requested_seats(state).tolist())

    def differences(self, state, snapshot, env, parties):
        from .finite_gameplay_comparison import state_differences
        layout = {k: dict(shape=list(np.asarray(getattr(state, k)).shape[1:]),
            dtype=str(np.asarray(getattr(state, k)).dtype)) for k in state._fields}
        return state_differences(self.boundary(state), snapshot, env, parties, self.vocabulary, layout)

    def chance_control(self, coupling, state, chance, cell, request):
        layout = {k: dict(shape=list(np.asarray(getattr(state, k)).shape[1:]),
            dtype=str(np.asarray(getattr(state, k)).dtype)) for k in state._fields}
        packet = dict(before=self.boundary(state), chance={k: np.asarray(getattr(chance, k)).tolist()
            for k in chance._fields}, stream_id=cell['stream_id'], request=request)
        return coupling.begin(packet, layout)

    def selection(self, state, actions, requested, usable):
        # Literal chosen move identity, including canonical Struggle and hard
        # charge. This describes the selected request, never a called child.
        selected = []
        for side in (0, 1):
            action = int(actions[0, side]); p = int(state.active[0, side])
            value = dict(requested=bool(requested[side]), engine_action=action)
            if requested[side]:
                if action < 4:
                    move = int(state.moves[0, side, p, action])
                    if int(state.charge_move[0, side, p]) > 0: move = int(state.charge_move[0, side, p])
                    if not np.any(usable[side]): move = int(self.tables.struggle)
                    value.update(kind='move', move=self.vocabulary['moves'][move], mechanics=move)
                else: value.update(kind='switch', physical=[x for x in range(6) if x != p][action-4])
            selected.append(value)
        return selected


def prepare(plan):
    """One startup compiler/factory, no battle/model construction here."""
    validate_plan(plan)
    actual = subprocess.check_output(['git', '-C', str(ROOT), 'rev-parse', 'HEAD'], text=True).strip()
    if actual != plan['source_commit']: raise ValueError('collection source commit differs')
    subprocess.run(['git', '-C', str(ROOT), 'diff', '--quiet', 'HEAD', '--', 'gpu_engine', 'src', 'scripts'], check=True)
    if digest(plan['node']) != plan['node_sha256']: raise ValueError('Node binary differs')
    from . import source_provenance, finite_stream_comparison as original
    pool = source_provenance.load_constructor_pool(plan['pool']['path'])
    if source_provenance._pool(pool, rehash=True)['pool_sha256'] != plan['pool']['sha256']:
        raise ValueError('constructor pool differs')
    if max(i for c in plan['cells'] for i in c['team_indices']) >= len(pool.arrays['species']):
        raise ValueError('declared team row outside actual pool')
    ivs = original.load_original_ivs(plan['pool']['path'], pool, expected_pool_sha256=plan['pool']['sha256'])
    vocab_doc = checked_document(plan['vocabulary']); labels = checked_document(plan['labels'])
    gender_policy = checked_document(plan['gender_policy'])
    metadata = pool.metadata()
    from . import gender
    gender.validate_policy(gender_policy, metadata['vocabulary']['species'])
    provenance = gender_policy.get('provenance', {})
    if (provenance.get('pool_sha256') != plan['pool']['sha256']
            or provenance.get('compiled_tree') != metadata['compiled_tree']
            or provenance.get('showdown_commit') != metadata['source_commit_label']):
        raise ValueError('gender policy must bind this actual original pool/source')
    import jax
    import jax.numpy as j
    from pokezero.category_vocab import CategoryVocabulary
    from pokezero.dex import load_showdown_dex_cached
    from pokezero.randbat import load_gen3_randbat_source_cached
    from pokezero.recorded_mirror_roots import CHAMPION_CONTRACT
    from pokezero.v5_contract import LabelVocabulary
    from pokezero.v5_model import V5ModelConfig
    from pokezero.v5_phasec import category_binding
    from pokezero.local_showdown import LocalShowdownConfig
    from pokezero.observation import ObservationFeatureMasks
    from pokezero.showdown import V4_REPLAY_OBSERVATION_SPEC
    from . import public_belief, v4_encoder, v5_observation, transition, engine_request_source
    from . import engine_device_observation as encoder, engine_device_trackers as public
    from .device_protocol_reference import Factory
    from .benchmark_device_selfplay import request_seats
    from .state import usable_moves
    # Original loader verifies the real source revision and actual compiled
    # Dex tree against this pool. No source rule is used to waive a state row.
    rules = engine_request_source.load_source_rules(plan['showdown_root'], metadata, node=plan['node'])
    origin = engine_request_source._rules_origin(rules)
    vocabulary = CategoryVocabulary(tuple(vocab_doc['tokens']), vocab_doc['oov_buckets'], vocab_doc['aliases'])
    belief = load_gen3_randbat_source_cached(plan['showdown_root'])
    if (vocab_doc['category_vocab_sha256'] != CHAMPION_CONTRACT['category_vocab_sha256']
            or belief.metadata.source_hash != CHAMPION_CONTRACT['belief_set_source_hash']):
        raise ValueError('original category/source namespace differs')
    catalog = v4_encoder.compile_encoder(public_belief.compile_source(belief),
        load_showdown_dex_cached(plan['showdown_root']), vocabulary,
        checkpoint_sha256=CHAMPION_CONTRACT['checkpoint_sha256'], feature_masks=CHAMPION_CONTRACT['feature_masks'])
    model_config = V5ModelConfig(category_vocab_size=vocabulary.size,
        category_vocabulary_sha256=category_binding(vocabulary), label_vocabulary=LabelVocabulary(**labels))
    if (category_binding(vocabulary) != model_config.category_vocabulary_sha256
            or model_config.label_vocabulary.moves != tuple(catalog.moves.names[1:])):
        raise ValueError('actual original ordered category/label namespace differs')
    adapter = v5_observation.compile_adapter(catalog, model_config=model_config)
    tables = encoder.compile_tables(metadata, catalog=catalog)
    engine = transition.tables_from_metadata(metadata)
    masks = dict(catalog.masks.__dict__); masks['opponent_tendency_stats_block'] = masks.pop('stats_block')
    config = LocalShowdownConfig(showdown_root=plan['showdown_root'], node_binary=plan['node'],
        observation_spec=V4_REPLAY_OBSERVATION_SPEC, feature_masks=ObservationFeatureMasks(**masks),
        category_vocab=catalog.vocabulary, set_belief_source=True)

    def observe(state, trackers):
        mask = transition.action_mask(state, engine)
        views = tuple(encoder.encode(state, public.readout(trackers, j.full((1,), side, j.int32)), tables,
            observer_sides=j.full((1,), side, j.int32), legal_action_mask=mask[:, side]) for side in (0, 1))
        return mask, request_seats(state), usable_moves(state), views

    source_files = subprocess.check_output(['git', '-C', str(ROOT), 'ls-files', 'gpu_engine', 'src', 'scripts'], text=True).splitlines()
    bindings = dict(source_commit=actual, source_files={p: digest(ROOT/p) for p in source_files},
        pool=plan['pool'], vocabulary=plan['vocabulary'], labels=plan['labels'], gender_policy=plan['gender_policy'],
        showdown_source=json.loads(origin.source_tree_json), compiled_tree=metadata['compiled_tree'],
        node_sha256=plan['node_sha256'], jax_version=jax.__version__, numpy_version=np.__version__,
        device_platform=jax.default_backend(), policy=plan['policy'])
    factory = Factory(catalog, adapter)
    factory.functions.require_catalog(catalog)  # existing B2 dtype/precision/source contract before any cell
    return Runtime(plan, pool, ivs, metadata['vocabulary'], factory, config, engine, tables,
        jax.jit(lambda s: transition.initialize_with_public(s, engine, belief_tables=tables.belief)),
        jax.jit(observe), jax.jit(lambda s, a, t, chance: transition.step_with_public(s, a, engine, t,
            belief_tables=tables.belief, chance_inputs=chance)), jax.jit(transition.sample_chance_inputs),
        gender_policy, bindings)


def _candidate(observation):
    from .device_observation_parity import SHAPES
    return {key: np.asarray(getattr(observation, key))[0] for key in SHAPES}


def collect_game(runtime, cell, writer, reducer, *, env_factory=None, coupling_factory=None, folder=None):
    """One fixed cell once. Only output/storage failures abort the whole cohort."""
    from . import finite_gameplay_comparison as comparison, finite_stream_comparison as original
    env_factory = original.OwningShowdownEnv if env_factory is None else env_factory
    coupling_factory = comparison.ChanceControl if coupling_factory is None else coupling_factory
    env = None; native_live = False; prefix = None; references = [None, None]
    offsets = [0, 0]; lines = [[], []]; snapshot = None; evidence = None
    root = dict(cell, status='ATTEMPTED', executed_steps=0, requested_views=0, candidate_error_views=0,
        candidate_error_kind_views={})
    writer.event(dict(kind='START', **root))
    try:
        state, trackers, raw = runtime.construct(cell)
        parties = runtime.parties(raw, cell)
        identities = [[comparison.fidelity.toid(mon['species']) for mon in party] for party in parties]
        writer.boundary(dict(kind='CONSTRUCTOR', stream_id=cell['stream_id'], raw=runtime.boundary(raw),
            initialized=runtime.boundary(state), literal_sets=parties))
        try:
            packed = original.pack_sets(parties, showdown_root=runtime.plan['showdown_root'], node=runtime.plan['node'])
            coupling = coupling_factory(Path(folder)/str(cell['stream_id']), runtime.config)
            env = env_factory(coupling.config)
            env.reset_pool_sets(packed, seed=cell['native_seed'])
            audit = coupling.initialized()
            snapshot = env.snapshot(); evidence = comparison.native_evidence(snapshot, env)
            differences = runtime.differences(state, snapshot, env, identities)
            prefix = _prefix(True, differences, unresolved=_unresolved(audit))
            root.update(initial_differences=differences, initial_chance_audit=audit)
            for side in (0, 1):
                references[side] = runtime.factory.game(owner_side=side, stream_id=cell['stream_id'],
                    battle_id='device-parity-'+str(cell['stream_id']))
                lines[side] = list(env.owner_transcripts[side]); references[side].feed(lines[side])
            offsets = [len(x) for x in env.owner_transcripts]; native_live = True
            writer.boundary(dict(kind='INITIAL_NATIVE', stream_id=cell['stream_id'],
                engine=runtime.boundary(state), native=evidence, differences=differences,
                chance_audit=audit, owning_lines=lines, prefix_agreement=prefix))
        except OSError: raise
        except Exception as error:
            root['native_initialization_refusal'] = _error(error); prefix = None
        for request in range(runtime.plan['max_requests']):
            if bool(np.asarray(state.done)[0]): root['status'] = 'TERMINAL'; break
            if int(np.asarray(state.turn)[0]) >= runtime.plan['max_turns']:
                root['status'] = 'TURN_CAP_UNKNOWN'; break
            mask, requested, usable, observations = runtime.observe(state, trackers)
            mask, requested, usable = np.asarray(mask)[0], np.asarray(requested)[0], np.asarray(usable)[0]
            before = runtime.boundary(state); pre = None
            if native_live:
                try:
                    pre = runtime.differences(state, snapshot, env, identities)
                    prefix = _prefix(prefix, pre)
                except Exception as error:
                    root['native_precheck_refusal'] = _error(error); prefix = None; native_live = False
            # Freeze PRE eligibility and views before any advancement. Future
            # chance/state disagreement cannot remove this actual current view.
            eligible = prefix
            candidates = [_candidate(o) for o in observations]
            ref_values = [None, None]; ref_errors = [None, None]; ref_meta = [None, None]
            for side in (0, 1):
                if requested[side]:
                    from .engine_device_observation import ERROR_KINDS
                    root['requested_views'] += 1
                    error_bits = int(np.asarray(observations[side].error)[0])
                    root['candidate_error_views'] += int(error_bits != 0)
                    for bit, kind in enumerate(ERROR_KINDS):
                        if error_bits & (1 << bit):
                            root['candidate_error_kind_views'][kind] = root['candidate_error_kind_views'].get(kind, 0)+1
                    if references[side] is not None:
                        ref_values[side], ref_errors[side] = references[side].view()
                        ref_meta[side] = references[side].metadata()
                    else: ref_errors[side] = dict(source='B2_PROTOCOL', kind='ALL', column='ALL', reason='NO_NATIVE_REFERENCE')
            actions = choose_legal(mask, requested, seed=runtime.plan['sampling_seed'], stream=cell['stream_id'], step=request)
            selected = runtime.selection(state, actions, requested, usable)
            chance = runtime.sample(state)  # exactly one original draw capsule
            chance_packet = {key: np.asarray(getattr(chance, key)).tolist() for key in chance._fields}
            native_before = evidence
            # Engine executes once with the SAME sampled object, never redraw.
            try:
                advanced = runtime.advance(state, actions, trackers, chance)
                after = advanced.step.state
                if bool(np.asarray(advanced.step.invalid_actions)[0]):
                    raise ValueError('ENGINE_ADVANCEMENT_INVALID_ACTION')
                # Force actual completion here even if a future adapter defers
                # transfer of a leaf not used in physical boundary evidence.
                after_packet = runtime.boundary(after)
            except OSError: raise
            except Exception as error:
                root.update(status='ENGINE_ADVANCEMENT_FAILED', engine_error=_error(error))
                physical_sha = writer.boundary(dict(kind='STEP', stream_id=cell['stream_id'], request=request,
                    before=before, before_native=native_before, pre_differences=pre, pre_agreement=eligible,
                    actions=actions.tolist(), selected=selected, chance=chance_packet, engine_error=_error(error)))
                _record_views(writer, reducer, cell, request, int(np.asarray(state.turn)[0]), requested,
                    candidates, ref_values, ref_errors, ref_meta, lines, eligible, physical_sha, observations)
                break
            native_actions = None; control = None; audit = []; post = None; native_error = None
            if native_live:
                try:
                    native_actions = comparison.native_actions(env, snapshot, selected, identities)
                    control = runtime.chance_control(coupling, state, chance, cell, request)
                    try: env.step(native_actions)  # exactly one original advancement
                    finally: audit = coupling.end()
                    snapshot = env.snapshot(); evidence = comparison.native_evidence(snapshot, env)
                    post = runtime.differences(after, snapshot, env, identities)
                    prefix = _prefix(prefix, post, unresolved=_unresolved(audit))
                except OSError: raise
                except Exception as error:
                    native_error = _error(error); prefix = None; native_live = False
            after_lines = [list(env.owner_transcripts[s][offsets[s]:]) for s in (0, 1)] if env is not None else [[], []]
            physical_sha = writer.boundary(dict(kind='STEP', stream_id=cell['stream_id'], request=request,
                before=before, after=after_packet, before_native=native_before,
                after_native=evidence if native_error is None and native_actions is not None else None,
                pre_differences=pre, post_differences=post, pre_agreement=eligible, next_prefix=prefix,
                actions=actions.tolist(), selected=selected, native_actions=native_actions,
                chance=chance_packet, chance_control=control, chance_audit=audit, native_error=native_error,
                owning_lines_pre=lines, owning_lines_post=after_lines))
            _record_views(writer, reducer, cell, request, int(np.asarray(state.turn)[0]), requested,
                candidates, ref_values, ref_errors, ref_meta, lines, eligible, physical_sha, observations)
            state, trackers = after, advanced.trackers
            root['executed_steps'] += 1
            if native_live:
                for side in (0, 1):
                    lines[side] = after_lines[side]
                    references[side].feed(lines[side])
                offsets = [len(x) for x in env.owner_transcripts]
            else: lines = [[], []]
        else: root['status'] = 'REQUEST_CAP_UNKNOWN'
        if bool(np.asarray(state.done)[0]): root['status'] = 'TERMINAL'
        elif root['status'] == 'ATTEMPTED': root['status'] = 'UNKNOWN'
        root.update(final_turn=int(np.asarray(state.turn)[0]), final_done=bool(np.asarray(state.done)[0]),
            final_winner=int(np.asarray(state.winner)[0]), final_physical_prefix=prefix)
    except OSError: raise
    except Exception as error: root.update(status='COLLECTION_CELL_FAILED', error=_error(error))
    finally:
        if env is not None: env.close()
    writer.event(dict(kind='END', **root))
    return root


def _record_views(writer, reducer, cell, request, turn, requested, candidates, refs, errors,
                  ref_meta, lines, agreement, physical_sha, observations):
    for side in (0, 1):
        if not requested[side]: continue
        kinds = sorted({line.split('|', 2)[1] if line.startswith('|') else 'MALFORMED' for line in lines[side]})
        decision_id = f"{cell['stream_id']}-{request}-{side}"
        metadata = dict(decision_id=decision_id, stream_id=cell['stream_id'], root_id=cell['root_id'],
            request=request, owner_side=side, turn=turn, line_kinds=kinds, agreement=agreement,
            refusal=errors[side], reference_metadata=ref_meta[side], physical_binding_sha256=physical_sha,
            candidate_error=int(np.asarray(observations[side].error)[0]))
        writer.decision(metadata, candidates[side], refs[side])
        reducer.add(decision_id=decision_id, turn=turn, line_kinds=kinds, agreement=agreement,
            reference=refs[side], candidate=candidates[side], refusal=errors[side])


def collect(plan, *, out, enable_collection=False):
    if enable_collection is not True: raise ValueError('explicit prospective collection enablement required')
    validate_plan(plan)
    out = Path(out)
    if '..' in out.parts or not any(out.is_relative_to(p) for p in (Path('/shared/scott'), Path('/store'))):
        raise ValueError('large evidence must remain at an explicit remote output path')
    if out.exists(): raise ValueError('fresh output required; never resume/refill a declared bank')
    runtime = prepare(plan)
    from .device_observation_parity import AgreementParity
    reducer = AgreementParity(source_bindings=runtime.bindings)
    writer = Evidence(out); cells = []
    try:
        (out/'plan.json').write_text(_json(plan)+'\n')
        (out/'source-bindings.json').write_text(_json(runtime.bindings)+'\n')
        with tempfile.TemporaryDirectory(prefix='native-control-', dir=out) as folder:
            for cell in plan['cells']:
                cells.append(collect_game(runtime, cell, writer, reducer, folder=folder))
                pending = out/'progress.tmp'
                pending.write_text(_json(dict(completed_cells=len(cells),
                    declared_cells=len(plan['cells']), actual_steps=sum(c['executed_steps'] for c in cells)))+'\n')
                os.replace(pending, out/'progress.json')
    finally: writer.close()
    report = reducer.report()
    report.update(collection_schema=SCHEMA, declared_cells=len(plan['cells']), cells=cells,
        cell_status_counts=dict(Counter(c['status'] for c in cells)),
        requested_candidate_views=sum(c['requested_views'] for c in cells),
        candidate_error_views=sum(c['candidate_error_views'] for c in cells),
        candidate_error_kind_views=dict(sum((Counter(c['candidate_error_kind_views']) for c in cells), Counter())),
        policy=plan['policy'], model_calls=0, team_generation_calls=0,
        observation_timing_scope='OFFLINE_COMPARISON; not selfplay throughput',
        physical_agreement_scope='Continuous PRE original represented12-party state/dynamic types/requested/terminal agreement and resolved prior chance; not full volatile/queue/RNG identity. POST affects NEXT eligibility only.',
        artifact_files={p.name: dict(bytes=p.stat().st_size, sha256=digest(p))
            for p in (out/'plan.json', out/'source-bindings.json', out/'decisions.jsonl', out/'physical.jsonl', out/'journal.jsonl')})
    (out/'summary.json').write_text(_json(report)+'\n')
    return dict(summary_path=str(out/'summary.json'), summary_sha256=digest(out/'summary.json'),
        declared_cells=len(cells), qualification=report['qualification'], full_B4_pass=False)


def main(argv=None):
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True); parser.add_argument('--expected-config-sha256', required=True)
    parser.add_argument('--out', required=True); parser.add_argument('--enable-collection', action='store_true')
    args = parser.parse_args(argv)
    if not args.enable_collection: parser.error('--enable-collection is required before any work')
    plan = checked_document(dict(path=args.config, sha256=args.expected_config_sha256))
    print(_json(collect(plan, out=args.out, enable_collection=True)), flush=True)


if __name__ == '__main__': main()
