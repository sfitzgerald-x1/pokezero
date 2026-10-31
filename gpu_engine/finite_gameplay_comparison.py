"""Prospective saved gameplay -> original Native owning stream comparison.

No engine/model is replayed. One original Native battle per declared root is
advanced once along captured semantic actions. The existing fidelity bridge
couples the SAME literal chance operands; unmapped roles remain UNKNOWN.
Only a continuous prefix of measured pre/post represented-state agreement is
eligible for stream/column equality; physics divergence is never a stream bug.
Artifacts are streamed/spooled next to the explicit (normally remote) output.
"""
from collections import Counter
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import tempfile
import time

import numpy as np

from . import engine_gameplay_trace as trace
from . import finite_stream_comparison as original, owning_python_reference, fidelity, public_wire, v4_encoder
from .state import BattleState
from pokezero.showdown import showdown_choice_for_action

SCHEMA = 'pokezero.B4-conditional-gameplay-stream-comparison.v1'
MAX_RECORD_BYTES = 16*1024**2


class ChanceControl:
    """Existing reviewed fidelity bridge/hooks, no new mechanics interception."""
    def __init__(self, folder, config):
        self.folder=folder; folder.mkdir()
        launcher=folder/'bridge.mjs'
        launcher.write_text('await import('+json.dumps(Path(__file__).with_name('fidelity_bridge.mjs').as_uri())+');\n')
        self.config=replace(config,bridge_path=launcher)
        self.control=folder/'chance.json';self.audit=folder/'chance-audit.jsonl'
        self.initializing=folder/'initializing.json'
        # All actual genders came from the original accepted pool/start packet.
        self.initializing.write_text(json.dumps(dict(allow_inert_gender=False)))

    def take_audit(self):
        rows=[]
        if self.audit.exists():
            if self.audit.stat().st_size>MAX_RECORD_BYTES:raise ValueError('bounded chance audit exceeded')
            rows=[json.loads(line)for line in self.audit.read_text().splitlines()]
            self.audit.unlink()
        return rows

    def initialized(self):
        self.initializing.unlink(missing_ok=True)
        return self.take_audit()

    def begin(self, captured, layout):
        state=decode_state(captured['before']['state'],layout);raw=captured['chance']
        if set(raw)!= {'draws','phazing','secondary','sleep_talk','entry_ties','initial_tie'}:
            raise ValueError('complete original ChanceInputs required')
        # These are the original kernel's saved-array selectors on a suspended
        # action turn; not another RNG sample or reconstructed outcome.
        pending=bool(state.pending_residual[0])
        arrays={name:np.asarray(raw[name]) for name in raw}
        draws=state.residual_draws[0] if pending else arrays['draws'][0]
        secondary=state.secondary_draws[0] if pending else arrays['secondary'][0]
        sleep=state.sleep_talk_draws[0] if pending else arrays['sleep_talk'][0]
        bounds=np.array([2,100,100,100,16,16,100,4,8,2,2,2])
        if (draws.shape!=(2,12) or draws.dtype.kind not in 'iu' or np.any(draws<0) or np.any(draws>=bounds)
                or any(v.shape!=(2,) or v.dtype.kind not in 'iu' or np.any(v<0) or np.any(v>=limit)
                    for v,limit in ((secondary,100),(sleep,60)))):
            raise ValueError('actual original finite chance operands required')
        value=dict(schema='pokezero.gpu_chance_tape.v1',nonce=f"{captured['stream_id']}:{captured['request']}",
            draws=draws.tolist(),secondary=secondary.tolist(),sleep_talk=sleep.tolist())
        self.control.write_text(json.dumps(value));return value

    def end(self):
        self.control.unlink(missing_ok=True)
        return self.take_audit()


def decode_state(packet, layout):
    if set(packet) != set(BattleState._fields) or set(layout) != set(packet):
        raise ValueError('literal complete actual state packet required')
    values = {}
    for name in BattleState._fields:
        dtype = np.dtype(layout[name]['dtype']); value = np.asarray(packet[name])
        if (dtype.kind not in 'iub' or value.shape != (1, *layout[name]['shape'])
                or value.dtype.kind not in ('b' if dtype.kind == 'b' else 'iu')):
            raise ValueError('actual state shape/type changed: '+name)
        if dtype.kind != 'b' and (np.any(value < np.iinfo(dtype).min) or np.any(value > np.iinfo(dtype).max)):
            raise ValueError('state integer outside original dtype: '+name)
        values[name] = value.astype(dtype)
    return BattleState(**values)


def captured_words(packet):
    if packet.get('available') is not True:
        raise ValueError(packet.get('reason', 'MISSING_ENGINE_FULL_CAPTURE'))
    shape = packet.get('shape'); packets = packet.get('packets')
    if (type(shape) is not list or len(shape) != 3 or shape[1:] != [64, 16]
            or type(shape[0]) is not int or not 0 < shape[0] <= 1024
            or type(packets) is not list or len(packets) != shape[0]):
        raise ValueError('actual bounded FULL packet shape required')
    words = np.zeros(shape, np.int32); valid = np.zeros(shape[:2], bool); more = np.zeros(shape[0], bool)
    for index, packet in enumerate(packets):
        mask = np.asarray(packet['valid']); rows = np.asarray(packet['words'])
        if mask.dtype != np.bool_ or mask.shape != (64,) or type(packet['more']) is not bool:
            raise ValueError('actual FULL masks required')
        if rows.size == 0: rows = np.empty((0, 16), np.int32)
        if (rows.shape != (int(mask.sum()), 16) or rows.dtype.kind not in 'iu'
                or np.any(rows < -2**31) or np.any(rows >= 2**31)):
            raise ValueError('actual integer FULL words required')
        words[index, mask] = rows; valid[index] = mask; more[index] = packet['more']
    return original.engine_words(words, valid, more)


def observation_arrays(values):
    """Decode the original five device dtypes lost by JSON, never re-encode."""
    if set(values) != set(original.SHAPES): raise ValueError('exact five saved tensors required')
    result = {}
    for name, shape in original.SHAPES.items():
        value = np.asarray(values[name]); mask = name.endswith('mask')
        kind = 'b' if mask else 'iu' if name != 'numeric_features' else 'iuf'
        if value.shape != shape or value.dtype.kind not in kind or not np.isfinite(value).all():
            raise ValueError('saved tensor type/shape changed:'+name)
        if kind == 'iu' and (np.any(value < 0) or np.any(value >= 2**31)):
            raise ValueError('saved categorical int32 range changed')
        result[name] = value.astype(np.bool_ if mask else np.int32 if kind == 'iu' else np.float32)
    return original._arrays(result, device=True)


class OwnerReference:
    """Original writer continuation and original incremental PythonBackend."""
    def __init__(self, catalog, config, *, owner, root_id, seed):
        self.catalog, self.owner, self.root_id, self.seed = catalog, owner, root_id, seed
        self.wire = public_wire.WireState(); self.poisoned = False
        self.history = []; self.request_count = 0; self.last_request = None
        self.prefix = hashlib.sha256(); self.backend = None; self.backend_error = None
        try:
            self.contract = owning_python_reference.ActorContract.champion()
            self.backend = owning_python_reference.PythonBackend(self.contract, config=config,
                battle_id=f'gameplay-{root_id}', owner=owner, stream_id=root_id)
        except Exception as error:
            self.backend_error = str(error)

    def append(self, lines, *, constructor=False):
        if any(type(line) is not str for line in lines): raise ValueError('literal owner lines required')
        # Validate ownership/visibility before retaining or ingesting anything.
        owning_python_reference._lines([*self.history, *lines], self.owner)
        self.history.extend(lines)
        for line in lines:
            self.prefix.update((line+'\n').encode())
            if line.startswith('|request|'):
                self.last_request = line; self.request_count += 1
        if self.backend is not None and self.backend_error is None:
            try:
                self.backend.ingest(lines)
            except Exception as error: self.backend_error = str(error)
        start = next((i for i, line in enumerate(lines) if line.startswith('|switch|')), len(lines)) if constructor else 0
        kinds, private, refused, words = Counter(), Counter(), [], []
        for offset, line in enumerate(lines[start:], start):
            kind = original.line_kind(line)
            if kind == 'request': private[kind] += 1; continue
            kinds[kind] += 1
            try:
                if self.poisoned: raise ValueError('PRIOR_REFERENCE_PROJECTION_REFUSAL')
                self.wire, packet = public_wire.project_line(self.wire, line, **v4_encoder.wire_kwargs(self.catalog))
                words.extend(list(event.words()) for event in packet.events)
            except ValueError as error:
                self.poisoned = True
                refused.append(dict(kind=kind, offset=offset, reason=str(error),
                    line_sha256=hashlib.sha256(line.encode()).hexdigest()))
        if constructor and start == len(lines):
            self.poisoned = True; refused.append(dict(kind='MISSING_INITIAL_SWITCH', reason='no Native arrival'))
        return dict(words=None if self.poisoned else words, raw_kinds=dict(kinds),
            refused_lines=refused, private_request_lines=dict(private),
            literal_owner_lines_sha256=original._sha(lines))

    def observation(self):
        if self.backend_error is not None or self.backend is None or self.last_request is None:
            raise ValueError(self.backend_error or 'MISSING_OWN_REQUEST')
        snapshot = owning_python_reference.OwnerSnapshot(self.contract.namespace,
            f'gameplay-{self.root_id}', self.owner, self.seed, self.root_id, 0,
            tuple(self.history), self.prefix.hexdigest(), self.request_count, self.last_request)
        observation, _ = self.backend.encode(snapshot)
        return {name:getattr(observation, name) for name in original.SHAPES}

    def close(self):
        if self.backend is not None: self.backend.close()


def compare_owner(boundary, side, reference, owner_reference, *, eligible):
    refusal = list(reference['refused_lines']); actual = None
    try: actual = captured_words(boundary['full'][side])
    except (ValueError, KeyError, TypeError) as error:
        refusal.append(dict(kind='ENGINE_FULL_CAPTURE', reason=str(error)))
    status = 'UNKNOWN'
    if eligible and not refusal and reference['words'] is not None:
        status = 'MATCH' if actual == reference['words'] else 'MISMATCH'
    result = dict(status=status, reference=reference, refusals=refusal,
        columns=dict(status='UNKNOWN', reason='STATE_NOT_AGREED' if not eligible else 'OBSERVATION_NOT_ADMITTED'),
        engine_words_sha256=None if actual is None else original._sha(actual),
        reference_words_sha256=None if reference['words'] is None else original._sha(reference['words']))
    if status == 'MISMATCH':
        words = reference['words']
        at = next((i for i, (a, b) in enumerate(zip(actual, words)) if a != b), min(len(actual), len(words)))
        result['first_difference'] = dict(index=at, engine=actual[at] if at < len(actual) else None,
            showdown=words[at] if at < len(words) else None)
    observation = boundary['observations'][side]
    if eligible and not refusal and observation.get('admitted') is True:
        try:
            a = observation_arrays(observation['arrays'])
            b = original._arrays(owner_reference.observation(), device=False)
            delta = np.abs(a['numeric_features'].astype(np.float64)-b['numeric_features'])
            result['columns'] = dict(status='COMPARED', numeric_bad=(delta > 1e-6).sum(axis=0).tolist(),
                numeric_max=delta.max(axis=0).tolist(),
                categorical_bad=(a['categorical_ids'] != b['categorical_ids']).sum(axis=0).tolist(),
                other_bad={k:int(np.count_nonzero(a[k] != b[k])) for k in
                    ('token_type_ids', 'attention_mask', 'legal_action_mask')})
        except Exception as error:
            result['columns'] = dict(status='UNKNOWN', reason='OWNING_PYTHON_REFERENCE_REFUSED', detail=str(error))
    return result


def state_differences(boundary, snapshot, env, original_parties, vocabulary, layout):
    state = decode_state(boundary['state'], layout)
    # The original represented-state comparison's full initial contract is
    # reused at EVERY boundary: all12 HP/maxHP/status/faints/stats/boosts/
    # ability/item/move/PP + active identities + weather. Not PRNG equality.
    differences = fidelity.compare(state, snapshot, original_parties, vocabulary, initial=True)
    requested = [f'p{i+1}' for i, value in enumerate(boundary['requested']) if value]
    native_requested = list(env.requested_players())
    if requested != native_requested:
        differences.append(dict(field='requested_players', gpu=requested, showdown=native_requested))
    terminal = env.terminal(); done = bool(state.done[0])
    if done != (terminal is not None):
        differences.append(dict(field='terminal', gpu=done, showdown=terminal is not None))
    elif done:
        winner = {'p1':0, 'p2':1, None:2}[terminal.winner]
        if terminal.capped or winner != int(state.winner[0]):
            differences.append(dict(field='winner_or_cap', gpu=int(state.winner[0]), showdown=[winner, terminal.capped]))
    return differences


def native_evidence(snapshot, env):
    """Literal original fields read by the comparison; no derived mechanics."""
    battle = snapshot.bridge_snapshot['battle']
    fields = ('hp','maxhp','status','fainted','storedStats','boosts','ability','item','moveSlots')
    sides = [dict(pokemon=[dict(set={'species':mon['set']['species']},
        **{key:mon[key] for key in fields}) for mon in side['pokemon']]) for side in battle['sides']]
    terminal = env.terminal()
    value = dict(battle=dict(sides=sides,field={'weather':battle['field']['weather']},turn=battle.get('turn')),
        requested_players=list(env.requested_players()),
        terminal=None if terminal is None else dict(winner=terminal.winner,capped=terminal.capped),
        original_snapshot_sha256=original._sha(snapshot.bridge_snapshot))
    # Freeze the literal subset while the live Native boundary is unchanged.
    return json.loads(json.dumps(value,allow_nan=False))


def native_actions(env, snapshot, selected, original_parties):
    """Map literal selected semantics through original Native legal choices."""
    players = tuple(f'p{i+1}' for i, value in enumerate(selected) if value['requested'])
    if tuple(env.requested_players()) != players: raise ValueError('REQUESTED_PLAYERS_DIVERGED')
    battle = snapshot.bridge_snapshot['battle']; result = {}
    for side, player in enumerate(('p1', 'p2')):
        if player not in players: continue
        wanted = selected[side]; matches = []
        local = env._state_for_player(player)
        request = snapshot.latest_requests[player]
        for action, legal in enumerate(env.legal_actions(player)):
            if not legal: continue
            kind, number = showdown_choice_for_action(local, action).split(); index = int(number)-1
            if kind == 'move' and wanted['kind'] == 'move':
                identity = fidelity.canonical(fidelity.toid(request['active'][0]['moves'][index]['id']))
                if identity == fidelity.canonical(fidelity.toid(wanted['move'])): matches.append(action)
            elif kind == 'switch' and wanted['kind'] == 'switch':
                identity = fidelity.toid(battle['sides'][side]['pokemon'][index]['set']['species'])
                physical = wanted['physical']
                if type(physical) is int and 0 <= physical < 6 and identity == original_parties[side][physical]:
                    matches.append(action)
        if len(matches) != 1: raise ValueError('SELECTED_SEMANTIC_ACTION_NOT_UNIQUE_OR_LEGAL:'+player)
        result[player] = matches[0]
    if not result: raise ValueError('NO_REFERENCE_REQUEST')
    return result


class Reducer:
    def __init__(self, roots):
        self.roots = roots; self.states = Counter(); self.streams = Counter(); self.refusals = Counter()
        self.raw = Counter(); self.native = Counter(); self.views = 0; self.eligible = 0; self.columns = 0
        self.nb = np.zeros(132, np.int64); self.cb = np.zeros(41, np.int64); self.maximum = np.zeros(132)
        self.unknown_columns = Counter(); self.decisions = 0
        self.other_bad = Counter()

    def add(self, row):
        self.decisions += 1; self.views += 2; eligible = row.get('state_status') == 'AGREE'
        self.states[row.get('state_status', 'UNKNOWN')] += 1
        self.raw.update(str(word[0]) for word in (row.get('engine_raw') or {}).get('words', []))
        for side in (0, 1):
            owner = row.get('owners', [{}]*2)[side]
            self.native.update(owner.get('reference', {}).get('raw_kinds', {}))
            self.refusals.update(item['kind'] for item in owner.get('refusals', []))
            if eligible:
                self.eligible += 1; self.streams[owner.get('status', 'UNKNOWN')] += 1
            columns = owner.get('columns', {}) if eligible else {}
            if columns.get('status') == 'COMPARED':
                nb, cb, maximum = (np.asarray(columns[k]) for k in ('numeric_bad', 'categorical_bad', 'numeric_max'))
                if (nb.shape != (132,) or cb.shape != (41,) or maximum.shape != (132,)
                        or nb.dtype.kind not in 'iu' or cb.dtype.kind not in 'iu' or np.any(maximum < 0)
                        or not np.isfinite(maximum).all() or np.any((nb < 0)|(nb > 23)) or np.any((cb < 0)|(cb > 23))):
                    raise ValueError('original173 column reduction shape/range required')
                self.columns += 1; self.nb += nb.astype(np.int64); self.cb += cb.astype(np.int64)
                self.maximum = np.maximum(self.maximum, maximum)
                self.other_bad.update(columns.get('other_bad', {}))
            else: self.unknown_columns[columns.get('reason', 'STATE_NOT_AGREED')] += 1

    def summary(self):
        match, mismatch, unknown = (self.streams[k] for k in ('MATCH', 'MISMATCH', 'UNKNOWN'))
        return dict(original_roots=len(self.roots), actual_engine_decisions=self.decisions,
            original_owner_decision_views=self.views, state_counts=dict(self.states),
            conditional_on_continuous_measured_state_agreement=dict(owner_views=self.eligible,
                matched=match, mismatched=mismatch, unknown=unknown,
                identified_match_rate=match/(match+mismatch) if match+mismatch else None,
                match_fraction_bounds=[match/self.eligible, (match+unknown)/self.eligible] if self.eligible else [None, None]),
            nonagreement_or_unknown_state_views=self.views-self.eligible,
            refusals_by_event_kind=dict(self.refusals), engine_raw_kind_counts=dict(self.raw),
            actual_native_line_kind_counts=dict(self.native), unknown_column_views_by_reason=dict(self.unknown_columns),
            other_tensor_mismatched_cells=dict(self.other_bad),
            numeric_columns=[dict(index=i, expected_cells=23*self.views, compared_cells=23*self.columns,
                unknown_cells=23*(self.views-self.columns), mismatched_cells=int(self.nb[i]),
                maximum_absolute_error=float(self.maximum[i]) if self.columns else None) for i in range(132)],
            categorical_columns=[dict(index=i, expected_cells=23*self.views, compared_cells=23*self.columns,
                unknown_cells=23*(self.views-self.columns), mismatched_cells=int(self.cb[i])) for i in range(41)],
            full_B4_qualified=False, fidelity_qualified=False)


def _spool(path, measured, expected_sha256, folder):
    descriptor = measured['gameplay_trace']
    if (expected_sha256 != descriptor['sha256'] or trace.digest_file(path) != expected_sha256
            or Path(path).stat().st_size != descriptor['bytes']):
        raise ValueError('original caller-bound trace bytes changed')
    roster = [dict(stream_id=row['stream_id'], team_indices=row['team_indices']) for row in measured['cohort']]
    ids = [row['stream_id'] for row in roster]
    if not ids or len(ids) != len(set(ids)) or len(ids) != measured['original_denominator']:
        raise ValueError('complete unique fixed root roster required')
    if any(type(value) is not int or not 0 < value < 2**31 for value in ids):
        raise ValueError('original positive int32 stream IDs required')
    counts = dict.fromkeys(ids, 0); initials = set(); chain = '0'*64; header = end = None; last = {}; terminals = {}
    parsed_digest = hashlib.sha256()
    with Path(path).open('rb') as source:
        while raw := source.readline(MAX_RECORD_BYTES+1):
            parsed_digest.update(raw)
            if len(raw) > MAX_RECORD_BYTES or not raw.endswith(b'\n'): raise ValueError('bounded complete JSONL record required')
            row = json.loads(raw)
            if end is not None or row['previous_sha256'] != chain: raise ValueError('trace order/hash chain changed')
            chain = hashlib.sha256(raw).hexdigest(); kind, payload = row['kind'], row['payload']
            if header is None:
                if kind != 'HEADER': raise ValueError('original trace header missing')
                header = payload
                if (header['schema'] != trace.SCHEMA or header['cohort'] != roster
                        or header['runtime_binding'] != measured['runtime_binding']
                        or header['source'] != measured['source'] or header['pool_receipt'] != measured['pool_receipt']
                        or any(header[k] != measured[k] for k in ('max_requests', 'constructor_seed', 'sampling_seed'))):
                    raise ValueError('same actual measurement/source/pool/seeds required')
                continue
            if kind == 'END': end = payload; continue
            stream = payload.get('stream_id')
            if stream not in counts: raise ValueError('foreign stream in actual trace')
            if kind == 'INITIAL':
                if stream in initials or counts[stream]: raise ValueError('duplicate/migrated constructor')
                decode_state(payload['boundary']['state'], header['state_layout'])
                initials.add(stream); last[stream] = original._sha(payload['boundary'])
            elif kind == 'DECISION':
                if (stream not in initials or payload['request'] != counts[stream]
                        or payload['request'] >= header['max_requests']
                        or original._sha(payload['before']) != last[stream]):
                    raise ValueError('actual decision history gap/reorder/before-state join changed')
                before, after = payload['before'], payload['after']
                for boundary in (before, after): decode_state(boundary['state'], header['state_layout'])
                if (after['frame'] != before['frame']+int(payload['executed'])
                        or before['cursor']['stream_id'] != stream or after['cursor']['stream_id'] != stream):
                    raise ValueError('actual executed frame/stream join changed')
                counts[stream] += 1; last[stream] = original._sha(after)
                if payload['executed'] and not payload['invalid'] and after['state']['done'][0]:
                    terminals[stream] = dict(stream_id=stream,frame=after['frame'],executed=True,invalid=False,
                        core_admitted=after['core_admitted'],done=after['state']['done'][0],winner=after['state']['winner'][0],
                        terminal_closeout=after['terminal_closeout'],
                        hp=after['state']['hp'][0],max_hp=after['state']['max_hp'][0],engine_turn=after['state']['turn'][0],
                        owner_admission=after['owners'],runtime_binding=header['runtime_binding'],source=header['source'])
            else: raise ValueError('unknown trace record')
            with (folder/str(stream)).open('ab') as target: target.write(raw)
    if (parsed_digest.hexdigest() != expected_sha256 or header is None or end is None or end['cohort'] != measured['cohort']
            or {int(k):v for k,v in end['written'].items()} != counts
            or end['attempts'] != {str(k):v for k,v in descriptor['attempts'].items()}):
        raise ValueError('original trace completion/count binding missing')
    if descriptor['complete']:
        recorded = {row['stream_id']:row for row in descriptor['terminals']}
        completed = {row['stream_id'] for row in measured['cohort'] if row['status'] == 'COMPLETED'}
        if recorded != terminals or completed != set(terminals):
            raise ValueError('COMPLETED needs the same literal executed terminal/winner/HP/admission evidence')
    return header, counts


def compare_trace(measured, path, pool, catalog, *, expected_trace_sha256, native_seed_start,
                  showdown_root, node, env_config, original_ivs, out_path):
    """EXPLICIT new prospective Native cohort; one root/process at a time."""
    reference_start=time.perf_counter_ns()
    ids = [row['stream_id'] for row in measured['cohort']]
    if type(native_seed_start) is not int or not 0 <= native_seed_start < native_seed_start+len(ids) <= 2**31:
        raise ValueError('finite declared original Native seed bank required')
    receipt = original.source_provenance._pool(pool, rehash=True)
    if (receipt['pool_sha256'] != measured['pool_receipt']['pool_sha256']
            or pool.metadata()['vocabulary'] is None or original_ivs.pool_sha256 != measured['pool_receipt']['pool_sha256']):
        raise ValueError('same original pool/IV archive required')
    out_path = Path(out_path); reducer = Reducer(ids); roots = []
    with tempfile.TemporaryDirectory(prefix='gameplay-spool-', dir=out_path.parent) as temporary:
        folder = Path(temporary); header, counts = _spool(path, measured, expected_trace_sha256, folder)
        with out_path.open('x') as output:
            for ordinal, cell in enumerate(measured['cohort']):
                stream = cell['stream_id']; env = None; references = []; stopped = None; continuous = True
                processed = 0
                root = dict(stream_id=stream, native_seed=native_seed_start+ordinal, native_started=False,
                    engine_status=cell['status'], trace_decisions=counts[stream])
                source = folder/str(stream)
                try:
                    if not source.exists(): raise ValueError('MISSING_ACTUAL_CONSTRUCTOR_CAPTURE')
                    records = source.open('r')
                    with records:
                        first = json.loads(next(records))['payload']; initial = first['boundary']
                        state = decode_state(initial['state'], header['state_layout'])
                        parties = original.pool_sets(pool, cell['team_indices'], state.gender[0], state.happiness[0], original_ivs=original_ivs)
                        packed = original.pack_sets(parties, showdown_root=showdown_root, node=node)
                        original_parties = [[fidelity.toid(mon['species']) for mon in party] for party in parties]
                        coupling=ChanceControl(folder/f'native-{stream}',env_config)
                        env = original.OwningShowdownEnv(coupling.config); root['native_started'] = True
                        env.reset_pool_sets(packed, seed=native_seed_start+ordinal)
                        initial_audit=coupling.initialized();root['initial_chance_audit']=initial_audit
                        initial_unknown=any(row['kind']in('UNRESOLVED','INITIAL_RANDOM')for row in initial_audit)
                        root['exact_pool_sets_sha256'] = original._sha(parties)
                        snapshot = env.snapshot()
                        evidence = native_evidence(snapshot,env)
                        output.write(json.dumps(dict(kind='INITIAL',stream_id=stream,native_seed=root['native_seed'],
                            native_compared_state=evidence,native_owner_lines=[list(lines) for lines in env.owner_transcripts]),sort_keys=True)+'\n')
                        output.flush()
                        differences = state_differences(initial, snapshot, env, original_parties, pool.metadata()['vocabulary'], header['state_layout'])
                        continuous = not differences and not initial_unknown
                        root.update(initial_state_status='UNKNOWN' if initial_unknown else 'AGREE' if continuous else 'DIVERGE', initial_state_differences=differences)
                        references = [OwnerReference(catalog, env_config, owner=f'p{s+1}', root_id=stream, seed=root['native_seed']) for s in (0,1)]
                        offsets = [len(lines) for lines in env.owner_transcripts]
                        for side in (0,1): references[side].append(list(env.owner_transcripts[side]), constructor=True)
                        for raw in records:
                            captured = json.loads(raw)['payload']; row = dict(stream_id=stream, request=captured['request'],
                                state_status='UNKNOWN', owners=[{},{}], engine_raw=captured['after']['raw'])
                            try:
                                if stopped: raise ValueError('PRIOR_NATIVE_REFUSAL:'+stopped)
                                pre = state_differences(captured['before'], snapshot, env, original_parties, pool.metadata()['vocabulary'], header['state_layout'])
                                row['native_before_state_sha256'] = original._sha(evidence)
                                actions = native_actions(env, snapshot, captured['selected'], original_parties)
                                row['actual_native_actions'] = actions
                                row['chance_control']=coupling.begin(captured,header['state_layout'])
                                try:
                                    env.step(actions)  # ORIGINAL actual Native advancement, exactly once
                                finally:
                                    row['chance_audit']=coupling.end()
                                snapshot = env.snapshot()
                                evidence = native_evidence(snapshot,env)
                                post = state_differences(captured['after'], snapshot, env, original_parties, pool.metadata()['vocabulary'], header['state_layout'])
                                unresolved=any(item['kind']in('UNRESOLVED','INITIAL_RANDOM')for item in row['chance_audit'])
                                eligible = continuous and not pre and not post and captured['executed'] and not unresolved
                                row.update(state_status='UNKNOWN' if unresolved else 'AGREE' if eligible else 'DIVERGE', before_state_differences=pre,
                                    after_state_differences=post, prior_state_agreement=continuous,
                                    native_compared_state=evidence,
                                    native_owner_lines=[list(lines[offset:]) for lines, offset in zip(env.owner_transcripts, offsets)])
                                continuous = continuous and not pre and not post and not unresolved
                                for side in (0,1):
                                    projected = references[side].append(row['native_owner_lines'][side])
                                    row['owners'][side] = compare_owner(captured['after'], side, projected, references[side], eligible=eligible)
                                    if unresolved:
                                        row['owners'][side]['refusals'].append(dict(kind='UNRESOLVED_CHANCE',reason='original coupling audit'))
                                        row['owners'][side]['columns']=dict(status='UNKNOWN',reason='UNRESOLVED_CHANCE')
                                offsets = [len(lines) for lines in env.owner_transcripts]
                            except Exception as error:
                                stopped = str(error); continuous = False
                                row['error'] = dict(type=type(error).__name__, message=stopped)
                                row['owners'] = [dict(status='UNKNOWN', refusals=[dict(kind='NATIVE_ADVANCEMENT_REFUSAL', reason=stopped)],
                                    columns=dict(status='UNKNOWN', reason='NATIVE_ADVANCEMENT_REFUSAL')) for _ in (0,1)]
                                row['native_owner_lines'] = [list(lines[offset:]) for lines, offset in zip(env.owner_transcripts, offsets)]
                                for side in (0,1):
                                    row['owners'][side]['reference'] = dict(raw_kinds=dict(Counter(original.line_kind(line)
                                        for line in row['native_owner_lines'][side] if original.line_kind(line) != 'request')))
                                offsets = [len(lines) for lines in env.owner_transcripts]
                            reducer.add(row); processed += 1
                            output.write(json.dumps(row, sort_keys=True, allow_nan=False)+'\n'); output.flush()
                except OSError:
                    raise  # An output transport failure cannot be a scientific row.
                except Exception as error:
                    root['error'] = dict(type=type(error).__name__, message=str(error))
                    if source.exists():
                        with source.open() as remaining:
                            next(remaining)
                            for raw in remaining:
                                captured = json.loads(raw)['payload']
                                if captured['request'] < processed: continue
                                row = dict(stream_id=stream, request=captured['request'], state_status='UNKNOWN',
                                    engine_raw=captured['after']['raw'], owners=[dict(status='UNKNOWN',
                                        refusals=[dict(kind='NATIVE_CONSTRUCTOR_REFUSAL', reason=str(error))]) for _ in (0,1)])
                                reducer.add(row); output.write(json.dumps(row, sort_keys=True)+'\n')
                finally:
                    for reference in references:
                        try: reference.close()
                        except Exception as error: root['python_close_error'] = str(error)
                    if env is not None:
                        try: env.close()
                        except Exception as error: root['close_error'] = str(error)
                missing = int(measured['gameplay_trace']['attempts'].get(str(stream), measured['gameplay_trace']['attempts'].get(stream, 0)))-counts[stream]
                if missing < 0: raise ValueError('negative missing decision count')
                for request in range(counts[stream], counts[stream]+missing):
                    row = dict(stream_id=stream, request=request, state_status='UNKNOWN', owners=[dict(status='UNKNOWN',
                        refusals=[dict(kind='ENGINE_TRACE_CAPTURE_MISSING', reason='bounded/failed capture')]) for _ in (0,1)])
                    reducer.add(row); output.write(json.dumps(row, sort_keys=True)+'\n')
                roots.append(root)
    return dict(schema=SCHEMA, trace_sha256=expected_trace_sha256, runtime_binding=measured['runtime_binding'],
        pool_sha256=measured['pool_receipt']['pool_sha256'], native_seed_start=native_seed_start,
        roots=roots, decisions_artifact=dict(path=str(out_path), sha256=trace.digest_file(out_path)),
        reference_elapsed_seconds=(time.perf_counter_ns()-reference_start)/1e9,
        existing_chance_hook_sources={name:trace.digest_file(Path(__file__).with_name(name))for name in('fidelity_bridge.mjs','chance_hooks.cjs')},
        summary=reducer.summary(), full_B4_qualified=False,
        scope='GAMEPLAY_ONLY; actual same actions and literal original ChanceInputs through existing fidelity coupling; unmapped roles UNKNOWN. Full existing represented-state fields plus requested/terminal, not internal RNG/volatile/counter/queue equivalence. Continuous disagreement never counted as a stream bug. Original173 public Python encoder, no model or JAX replay. Reference/capture costs are separate from throughput.')


def main(argv=None):
    """Explicit prospective collector; no default invocation or old-root retry."""
    import argparse
    from .finite_stream_comparison_saved import champion_env_config
    from pokezero.randbat import load_gen3_randbat_source_cached
    from pokezero.dex import load_showdown_dex_cached
    from pokezero.recorded_mirror_roots import CHAMPION_CONTRACT
    from pokezero.category_vocab import CategoryVocabulary
    from . import public_belief, source_provenance
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('measurement','trace','team-pool','vocabulary','showdown-root','out','decisions-out'):
        parser.add_argument('--'+name, required=True, type=Path)
    for name in ('measurement-sha256','trace-sha256'):
        parser.add_argument('--'+name, required=True)
    parser.add_argument('--native-seed-start', type=int, required=True)
    parser.add_argument('--node', default='node'); args=parser.parse_args(argv)
    if args.out.exists() or args.decisions_out.exists(): raise FileExistsError('exclusive comparison outputs required')
    raw = args.measurement.read_bytes()
    if hashlib.sha256(raw).hexdigest() != args.measurement_sha256:
        raise ValueError('original caller-bound measurement bytes changed')
    measured = json.loads(raw); values=json.loads(args.vocabulary.read_text())
    vocabulary=CategoryVocabulary(tuple(values['tokens']), values['oov_buckets'], values['aliases'])
    masks=CHAMPION_CONTRACT['feature_masks']
    catalog=v4_encoder.compile_encoder(public_belief.compile_source(load_gen3_randbat_source_cached(args.showdown_root)),
        load_showdown_dex_cached(args.showdown_root), vocabulary,
        checkpoint_sha256=values['checkpoint_sha256'], feature_masks=masks)
    config=champion_env_config(showdown_root=args.showdown_root,node=args.node,vocabulary=vocabulary,masks=masks)
    pool=source_provenance.load_constructor_pool(args.team_pool)
    ivs=original.load_original_ivs(args.team_pool,pool,expected_pool_sha256=measured['pool_receipt']['pool_sha256'])
    result=compare_trace(measured,args.trace,pool,catalog,expected_trace_sha256=args.trace_sha256,
        native_seed_start=args.native_seed_start,showdown_root=args.showdown_root,node=args.node,
        env_config=config,original_ivs=ivs,out_path=args.decisions_out)
    result['measurement_sha256']=args.measurement_sha256
    with args.out.open('x') as output: json.dump(result,output,sort_keys=True,allow_nan=False)
    print(json.dumps(result['summary'],sort_keys=True)); return 0


if __name__=='__main__': raise SystemExit(main())
