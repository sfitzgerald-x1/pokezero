"""Finite nonfatal engine/REAL Showdown owning-stream measurements.

This diagnostic never calls a model, generates teams, repairs state, or qualifies
B4. The constructor collector starts actual pinned Showdown battles from the
same already-generated pool sets. Stream comparison uses the original FULL
wire projection of actual p1/p2 bridge messages, never writer templates as the
reference. A refused projection/encoder cell remains UNKNOWN; later roots run.
"""
from collections import Counter
from dataclasses import dataclass
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess

import numpy as np

from pokezero.env import BattleStartOverride
from pokezero.local_showdown import LocalShowdownConfig, LocalShowdownEnv
from pokezero.recorded_encoder_parity import SHAPES, _arrays
from .public_inputs import EVENT_WORDS
from . import fidelity, public_wire, source_provenance, v4_encoder
from . import owning_python_reference

SCHEMA = 'pokezero.B4-finite-real-showdown-stream-comparison.v1'


@dataclass(frozen=True)
class OriginalPoolIVs:
    """Comparator-only original archive snapshot, never constructor-pool fields."""
    pool_sha256: str
    archive_bytes: bytes

    def values(self, pool):
        receipt = source_provenance._pool(pool, rehash=True)
        if (type(self.archive_bytes) is not bytes
                or hashlib.sha256(self.archive_bytes).hexdigest() != self.pool_sha256
                or receipt['pool_sha256'] != self.pool_sha256):
            raise ValueError('ORIGINAL_IV_POOL_BINDING_CHANGED')
        with np.load(io.BytesIO(self.archive_bytes), allow_pickle=False) as archive:
            if 'ivs' not in archive.files:
                raise ValueError('ORIGINAL_POOL_IVS_MISSING')
            ivs = archive['ivs']
        if (ivs.shape != (len(pool.arrays['species']), 6, 6) or ivs.dtype.kind not in 'iu'
                or np.any(ivs < 0) or np.any(ivs > 31)):
            raise ValueError('LITERAL_ORIGINAL_POOL_IVS_REQUIRED')
        return np.frombuffer(ivs.tobytes(), dtype=ivs.dtype).reshape(ivs.shape)


def load_original_ivs(path, pool, *, expected_pool_sha256):
    """Read one original NPZ snapshot bound to the saved measurement's receipt."""
    with Path(path).open('rb') as stream:
        before = os.fstat(stream.fileno()); payload = stream.read(); after = os.fstat(stream.fileno())
    identity = lambda s:(s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns)
    if (identity(before) != identity(after) or len(payload) != after.st_size
            or hashlib.sha256(payload).hexdigest() != expected_pool_sha256):
        raise ValueError('ORIGINAL_IV_POOL_ARCHIVE_CHANGED')
    result = OriginalPoolIVs(expected_pool_sha256, payload)
    result.values(pool)
    return result


def _sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
        allow_nan=False).encode()).hexdigest()


def line_kind(line):
    return line.split('|', 2)[1] if isinstance(line, str) and line.startswith('|') else 'MALFORMED'


class OwningShowdownEnv(LocalShowdownEnv):
    """Retain ORIGINAL getPlayerStreams output before the base discards it.

    snapshot.protocol_lines is omniscient plus private requests and is NOT an
    owning stream. Only actual p1/p2 messages for the current battle enter here.
    This subclass changes neither the native simulator nor its stream writer.
    """
    def __init__(self, config):
        self.owner_transcripts = [[], []]
        super().__init__(config)

    def _apply_event(self, event):
        if (event.get('type') == 'stream' and event.get('stream') in ('p1', 'p2')
                and self._battle_token is not None and event.get('battleId') == self._battle_token):
            lines = event.get('lines')
            if not isinstance(lines, list) or any(type(line) is not str for line in lines):
                raise ValueError('literal actual owning bridge lines required')
            self.owner_transcripts[int(event['stream'][1])-1].extend(lines)
        return super()._apply_event(event)

    def reset_pool_sets(self, packed, *, seed):
        """Original Battle.getTeam honors provided sets before random generation.

        The public start-override convenience API restricts format to customgame.
        Use its same original _reset transport with explicit randombattle format
        so native HP Percentage Mod/rules remain the engine exposure's format.
        No generator, source patch, or omniscient HP conversion is involved.
        """
        self.owner_transcripts = [[], []]
        override = BattleStartOverride(player_teams=dict(zip(('p1', 'p2'), packed)),
            observation_format_id='gen3randombattle')
        self._reset(seed=seed, format_id='gen3randombattle', start_override=override)


def pool_sets(pool, team_indices, genders, happiness=None, *, original_ivs=None):
    """Decode original Nature/EV/IV/moves/items; do NOT regenerate from seeds."""
    source_provenance._pool(pool, rehash=True)
    arrays, vocabulary = pool.arrays, pool.metadata()['vocabulary']
    if original_ivs is not None:
        if type(original_ivs) is not OriginalPoolIVs:
            raise ValueError('ORIGINAL_POOL_IV_SNAPSHOT_REQUIRED')
        ivs = original_ivs.values(pool)
    else:
        ivs = arrays.get('ivs')
        if ivs is None:
            raise ValueError('ORIGINAL_POOL_IV_SNAPSHOT_REQUIRED')
    indices, genders = np.asarray(team_indices), np.asarray(genders)
    if (indices.shape != (2,) or indices.dtype.kind not in 'iu'
            or len(set(indices.tolist())) != 2 or np.any(indices < 0)
            or np.any(indices >= len(arrays['species']))
            or genders.shape != (2, 6) or genders.dtype.kind not in 'iu'
            or not np.isin(genders, [0, 1, 2]).all()):
        raise ValueError('two original team rows and literal initialized genders required')
    happy = np.full((2, 6), 255) if happiness is None else np.asarray(happiness)
    if happy.shape != (2, 6) or happy.dtype.kind not in 'iu' or np.any((happy < 0) | (happy > 255)):
        raise ValueError('literal original happiness required')
    stats = ('hp', 'atk', 'def', 'spa', 'spd', 'spe')
    parties = []
    for side, index in enumerate(indices):
        party = []
        for slot in range(6):
            value = lambda field: int(arrays[field][index, slot])
            name = lambda table, field: vocabulary[table][value(field)]
            party.append(dict(species=name('species', 'species'), level=value('level'),
                moves=[vocabulary['moves'][int(move)] for move in arrays['moves'][index, slot] if move],
                item='' if not value('item') else name('items', 'item'),
                ability=name('abilities', 'ability'), nature=name('natures', 'nature'),
                ivs=dict(zip(stats, map(int, ivs[index, slot]))),
                evs=dict(zip(stats, map(int, arrays['evs'][index, slot]))),
                gender={0: 'N', 1: 'M', 2: 'F'}[int(genders[side, slot])],
                happiness=int(happy[side, slot])))
        parties.append(party)
    return parties


def pack_sets(parties, *, showdown_root, node):
    """Call ONLY the original Teams.pack; Teams.generate is never called."""
    code = ('const fs=require("node:fs"),path=require("node:path");'
        'const {Teams}=require(path.join(process.argv[1],"dist/sim/teams.js"));'
        'process.stdout.write(JSON.stringify(JSON.parse(fs.readFileSync(0,"utf8")).map(p=>Teams.pack(p))));')
    packed = json.loads(subprocess.check_output([node, '-e', code, str(showdown_root)],
        input=json.dumps(parties), text=True))
    if not isinstance(packed, list) or len(packed) != 2 or any(type(p) is not str or not p for p in packed):
        raise ValueError('two actual original packed teams required')
    return packed


def reference_words(lines, catalog, *, constructor=False):
    """Project every line in the exposed frame; refusals never abort a cohort.

    Constructor frames start at the first native switch, after battle setup.
    Count the excluded setup kinds explicitly. Private requests are a separate
    encoder input, never a public event. No line inside the frame is skipped.
    A refusal poisons this projection: subsequent lines are diagnostic only.
    """
    lines = tuple(lines)
    if any(type(line) is not str for line in lines):
        raise ValueError('literal actual player stream required')
    start = 0
    if constructor:
        start = next((i for i, line in enumerate(lines) if line.startswith('|switch|')), len(lines))
    outside = Counter(line_kind(line) for line in lines[:start])
    private = Counter()
    kinds, refusals, words = Counter(), [], []
    state = public_wire.WireState()
    tables = v4_encoder.wire_kwargs(catalog)
    for index, line in enumerate(lines[start:], start):
        kind = line_kind(line)
        if kind == 'request':
            private[kind] += 1
            continue
        kinds[kind] += 1
        try:
            state, projected = public_wire.project_line(state, line, **tables)
            for event in projected.events:
                row = np.asarray(event.words())
                if (row.shape != (EVENT_WORDS,) or row.dtype.kind not in 'iu'
                        or np.any(row < np.iinfo(np.int32).min) or np.any(row > np.iinfo(np.int32).max)):
                    raise ValueError('literal complete integer public event words required')
                words.append(row.tolist())
        except ValueError as error:
            refusals.append(dict(kind=kind, offset=index, line_sha256=hashlib.sha256(line.encode()).hexdigest(),
                reason=str(error)))
    if constructor and start == len(lines):
        refusals.append(dict(kind='MISSING_INITIAL_SWITCH', offset=None, reason='no original arrival'))
    return dict(words=None if refusals else words, projected_word_count=len(words),
        raw_kinds=dict(kinds), refused_lines=refusals, setup_kinds_outside_frame=dict(outside),
        private_request_lines=dict(private), literal_owning_stream_sha256=_sha(lines))


def engine_words(words, valid, more):
    """Flatten captured ORIGINAL FULL chunks for ONE owner, without refolding."""
    words, valid, more = np.asarray(words), np.asarray(valid), np.asarray(more)
    if (words.ndim != 3 or words.shape[1:] != (64, 16) or words.dtype.kind not in 'iu'
            or valid.shape != words.shape[:2] or valid.dtype != np.bool_
            or more.shape != (len(words),) or more.dtype != np.bool_
            or np.any(words < np.iinfo(np.int32).min) or np.any(words > np.iinfo(np.int32).max)):
        raise ValueError('literal captured FULL words/valid/more required')
    rows, ended = [], False
    for packet, mask, remaining in zip(words, valid, more):
        count = int(mask.sum())
        if (not np.array_equal(mask, np.arange(64) < count)
                or ended and (count or remaining) or remaining and count != 64):
            raise ValueError('original contiguous completely drained FULL chunks required')
        rows.extend(packet[:count].tolist())
        if not remaining:
            ended = True
    if not ended or not rows:
        raise ValueError('captured FULL stream missing or unfinished')
    return rows


def compare_owner(lines, captured, catalog, *, constructor=False, actual_observation=None,
                  reference_observation=None):
    """One nonfatal captured prefix; this collector measures constructors only."""
    native = reference_words(lines, catalog, constructor=constructor)
    refusal = list(native['refused_lines'])
    emitted = None
    try:
        emitted = engine_words(*captured)
    except (ValueError, TypeError) as error:
        refusal.append(dict(kind='ENGINE_FULL_CAPTURE', reason=str(error)))
    status = 'UNKNOWN'
    if not refusal:
        status = 'MATCH' if emitted == native['words'] else 'MISMATCH'
    result = dict(status=status, reference=native, refusals=refusal,
        engine_word_count=0 if emitted is None else len(emitted),
        engine_words_sha256=None if emitted is None else _sha(emitted),
        reference_words_sha256=None if native['words'] is None else _sha(native['words']))
    if status == 'MISMATCH':
        first = next((i for i, (a, b) in enumerate(zip(emitted, native['words'])) if a != b),
            min(len(emitted), len(native['words'])))
        result['first_difference'] = dict(index=first, engine=emitted[first] if first < len(emitted) else None,
            showdown=native['words'][first] if first < len(native['words']) else None)
    # Without both actual observations every one of the 173 columns is UNKNOWN.
    result['columns'] = dict(status='UNKNOWN', reason='OBSERVATIONS_NOT_CAPTURED')
    if actual_observation is not None and reference_observation is not None and not refusal:
        try:
            actual, reference = _arrays(actual_observation, device=True), _arrays(reference_observation, device=False)
            delta = np.abs(actual['numeric_features'].astype(np.float64)-reference['numeric_features'])
            result['columns'] = dict(status='COMPARED', numeric_bad=(delta > 1e-6).sum(axis=0).tolist(),
                numeric_max=delta.max(axis=0).tolist(), categorical_bad=(actual['categorical_ids'] !=
                reference['categorical_ids']).sum(axis=0).tolist(), other_bad={k:int(np.count_nonzero(actual[k] != reference[k]))
                    for k in ('token_type_ids', 'attention_mask', 'legal_action_mask')})
        except ValueError as error:
            result['columns'] = dict(status='UNKNOWN', reason='ENCODER_REFUSED', detail=str(error))
    return result


def collect_constructors(pool, team_indices, engine_state, captured, catalog, *, root_ids,
                         showdown_root, node, native_seed_start, env_config=None,
                         engine_observations=None, original_ivs=None):
    """Actual finite paired constructor exposure; per-root failure is nonfatal.

    Root supplies the same actual post-entry engine state and original captured
    FULL words/valid/more arrays with B,owner,chunk leading axes. This does not
    run/replay an engine, generate teams, choose actions or call a model. No
    constructor row may be replaced after a native/projection failure.
    """
    indices, ids = np.asarray(team_indices), np.asarray(root_ids)
    if (ids.ndim != 1 or indices.shape != (len(ids), 2) or ids.dtype.kind not in 'iu'
            or len(set(ids.tolist())) != len(ids) or not len(ids)
            or type(native_seed_start) is not int or not 0 <= native_seed_start < native_seed_start+len(ids) <= 2**31):
        raise ValueError('fixed finite same-cohort root IDs, team pairs and native seeds required')
    config = env_config or LocalShowdownConfig(showdown_root=Path(showdown_root), node_binary=node)
    rows = []
    for index, root_id in enumerate(ids):
        row = dict(root_id=int(root_id), team_indices=indices[index].tolist(),
            native_seed=native_seed_start+index, state_status='UNKNOWN', owners=[])
        env = None
        try:
            state = __import__('jax').tree.map(lambda x:x[index:index+1], engine_state)
            if int(np.asarray(state.turn)[0]) != 0:
                raise ValueError('constructor-only comparison requires the actual turn0 engine boundary')
            parties = pool_sets(pool, indices[index], np.asarray(state.gender)[0], np.asarray(state.happiness)[0],
                original_ivs=original_ivs)
            packed = pack_sets(parties, showdown_root=showdown_root, node=node)
            row['exact_pool_sets_sha256'] = _sha(parties)
            env = OwningShowdownEnv(config)
            env.reset_pool_sets(packed, seed=native_seed_start+index)
            # Preserve the actual Native reference before any state comparison,
            # observation or projection can fail. This remote row is evidence,
            # never permission to replay its already-attempted constructor.
            row['native_owner_lines'] = [list(lines) for lines in env.owner_transcripts]
            snapshot = env.snapshot()
            original = [[pool.metadata()['vocabulary']['species'][int(v)] for v in
                pool.arrays['species'][team]] for team in indices[index]]
            differences = fidelity.compare(state, snapshot, original, pool.metadata()['vocabulary'], initial=True)
            row.update(state_status='AGREE' if not differences else 'DIVERGE', state_differences=differences)
            for owner in range(2):
                owner_capture = tuple(np.asarray(part)[index, owner] for part in captured)
                actual = reference = None
                reference_receipt = None
                observation_error = None
                if engine_observations is not None:
                    try:
                        actual = {name:np.asarray(engine_observations[owner][name])[index] for name in SHAPES}
                        reference, reference_receipt = owning_python_reference.observe_owner(config,
                            row['native_owner_lines'][owner], owner=('p1', 'p2')[owner],
                            battle_id=f'saved-constructor-{int(root_id)}', stream_id=int(root_id),
                            seed=row['native_seed'])
                    except Exception as error:
                        observation_error = dict(type=type(error).__name__, message=str(error))
                # Always collect stream/refusal diagnostics; conditional reducer
                # alone decides which actually measured state-agreement rows enter.
                try:
                    compared = compare_owner(row['native_owner_lines'][owner], owner_capture,
                        catalog, constructor=True, actual_observation=actual, reference_observation=reference)
                except Exception as error:
                    compared = dict(status='UNKNOWN', error=dict(type=type(error).__name__, message=str(error)),
                        columns=dict(status='UNKNOWN', reason='REFERENCE_COMPARISON_ERROR'))
                if observation_error is not None:
                    compared['columns'] = dict(status='UNKNOWN', reason='REFERENCE_OR_ENGINE_ENCODER_REFUSED',
                        detail=observation_error)
                if reference_receipt is not None:
                    compared['python_reference'] = reference_receipt
                row['owners'].append(compared)
        except Exception as error:
            row['error'] = dict(type=type(error).__name__, message=str(error))
        finally:
            if env is not None:
                try:
                    env.close()
                except Exception as error:
                    row['close_error'] = dict(type=type(error).__name__, message=str(error))
        rows.append(row)
    return dict(schema=SCHEMA, scope='ACTUAL_CONSTRUCTOR_ONLY_NO_ACTION_OR_MODEL', rows=rows,
        summary=reduce_cohort(rows, root_ids=ids.tolist()))


def reduce_cohort(rows, *, root_ids):
    """All fixed roots/2 owners, unknown bounds and all 173 column denominators."""
    if (not isinstance(root_ids, (list, tuple)) or not root_ids
            or any(type(v) is not int for v in root_ids) or len(set(root_ids)) != len(root_ids)):
        raise ValueError('fixed unique integer root roster required')
    indexed = {}
    for row in rows:
        key = row.get('root_id')
        if key not in root_ids or key in indexed:
            raise ValueError('foreign or duplicate root result')
        indexed[key] = row
    states, streams, refused, raw_kinds, unknown_columns = Counter(), Counter(), Counter(), Counter(), Counter()
    numeric_bad, categorical_bad = np.zeros(132, np.int64), np.zeros(41, np.int64)
    numeric_max = np.zeros(132)
    column_views = 0
    agreement_views = 0
    for root_id in root_ids:
        row = indexed.get(root_id, {})
        state = row.get('state_status', 'UNKNOWN')
        if state not in ('AGREE', 'DIVERGE', 'UNKNOWN'):
            raise ValueError('measured native state status required')
        states[state] += 1
        owners = row.get('owners', [])
        if len(owners) > 2:
            raise ValueError('at most the two actual owners required')
        for side in range(2):
            owner = owners[side] if side < len(owners) else {}
            for item in owner.get('refusals', []):refused[item['kind']] += 1
            raw_kinds.update(owner.get('reference', {}).get('raw_kinds', {}))
            if state == 'AGREE':
                agreement_views += 1
                status = owner.get('status', 'UNKNOWN')
                if status not in ('MATCH', 'MISMATCH', 'UNKNOWN'):
                    raise ValueError('explicit stream comparison status required')
                streams[status] += 1
            columns = owner.get('columns', {})
            if state == 'AGREE' and columns.get('status') == 'COMPARED':
                nb, cb, maximum = (np.asarray(columns[k]) for k in
                    ('numeric_bad', 'categorical_bad', 'numeric_max'))
                if (nb.shape != (132,) or cb.shape != (41,) or maximum.shape != (132,)
                        or nb.dtype.kind not in 'iu' or cb.dtype.kind not in 'iu'
                        or np.any((nb < 0) | (nb > 23)) or np.any((cb < 0) | (cb > 23))
                        or maximum.dtype.kind not in 'iuf' or not np.isfinite(maximum).all()
                        or np.any(maximum < 0)):
                    raise ValueError('exact per-column compared cell counts required')
                numeric_bad += nb
                categorical_bad += cb
                numeric_max = np.maximum(numeric_max, maximum)
                column_views += 1
            else:
                unknown_columns[columns.get('reason', 'STATE_NOT_AGREED_OR_NOT_ATTEMPTED')] += 1
    identified = streams['MATCH'] + streams['MISMATCH']
    total_views = 2*len(root_ids)
    return dict(original_roots=len(root_ids), original_owner_views=total_views, attempted_roots=len(indexed),
        not_attempted_roots=len(root_ids)-len(indexed), state_counts=dict(states),
        conditional_on_measured_state_agreement=dict(owner_views=agreement_views,
            matched=streams['MATCH'], mismatched=streams['MISMATCH'], unknown=streams['UNKNOWN'],
            identified_match_rate=streams['MATCH']/identified if identified else None,
            match_fraction_bounds=[streams['MATCH']/agreement_views,
                (streams['MATCH']+streams['UNKNOWN'])/agreement_views] if agreement_views else [None, None]),
        nonagreement_or_unknown_state_views=total_views-agreement_views,
        refusals_by_event_kind=dict(refused), actual_native_line_kind_counts=dict(raw_kinds),
        numeric_columns=[dict(index=i, expected_cells=23*total_views, compared_cells=23*column_views,
            unknown_cells=23*(total_views-column_views), mismatched_cells=int(numeric_bad[i]),
            maximum_absolute_error=float(numeric_max[i]) if column_views else None) for i in range(132)],
        categorical_columns=[dict(index=i, expected_cells=23*total_views, compared_cells=23*column_views,
            unknown_cells=23*(total_views-column_views), mismatched_cells=int(categorical_bad[i])) for i in range(41)],
        unknown_column_views_by_reason=dict(unknown_columns), full_B4_qualified=False,
        interpretation='Canonical FULL event-word stream match, not literal text equality; actual native owning lines retained by hash/kind. Unknowns retained, no physics or B4 qualification.')
