"""Opt-in remote JSONL capture of actual retained decisions, never policy input.

One row is copied at a time, after its original core call. An exact immutable
prior after-packet may supply the next before-packet; all checks stay outside
this measurement-data reuse. No engine, model,
writer/fold, RNG sampling or simulator runs here. Capacity/IO failures disable
capture and retain dropped counts; they never alter the experiment's decisions.
The caller chooses an explicit remote path. No default artifact is created.
"""
import hashlib
import json
from pathlib import Path
import time
from collections import namedtuple

import jax
import numpy as np

try:
    from jaxlib import _jax as _jax_extension
    _CANONICAL_JAX_ARRAY_IMPL = _jax_extension.ArrayImpl
except (ImportError, AttributeError):
    _CANONICAL_JAX_ARRAY_IMPL = None  # Unavailable implementation: fresh capture.

from .state import BattleState
from pokezero.recorded_encoder_parity import SHAPES

SCHEMA = 'pokezero.actual-gameplay-trace.v1'
_TUPLE_GETTER = type(namedtuple('_TraceTupleDescriptor', 'field').field)


def _boundary_proof(result):
    """Host-only proof of the immutable inputs read by boundary(), not a gate.

    Unserialized private constructor/static holders are deliberately excluded.
    Every traversed container must be an ordinary immutable tuple record; every
    captured leaf must be the exact live native JAX ArrayImpl or an immutable scalar.
    NumPy fixtures, mutable wrappers and unfamiliar types always use fresh data.
    """
    from . import engine_retained_encoder_driver as driver
    edges = []
    def record(value):
        cls = type(value)
        fields = getattr(cls, '_fields', None)
        if (cls.__bases__ != (tuple,) or type(fields) is not tuple
                or len(fields) != len(value) or cls.__getattribute__ is not tuple.__getattribute__
                or getattr(value, '__dict__', None) is not None
                or any(type(name) is not str or type(getattr(cls, name, None)) is not _TUPLE_GETTER for name in fields)):
            raise TypeError('unproven boundary container')
        if any(getattr(value, name) is not tuple.__getitem__(value, index) for index, name in enumerate(fields)):
            raise TypeError('changed boundary field descriptor')
        edges.append(('record', value, fields))
        return fields
    def field(value, name):
        return tuple.__getitem__(value, record(value).index(name))
    def leaf(value):
        if type(value) is _CANONICAL_JAX_ARRAY_IMPL:
            if value.is_deleted():raise ValueError('deleted boundary array')
            edges.append(('array', value, (tuple(value.shape), str(value.dtype))))
        elif type(value) in (type(None), bool, int, str, bytes):
            edges.append(('scalar', value, type(value)))
        elif type(value) is float and np.isfinite(value):
            edges.append(('scalar', value, type(value)))
        else:raise TypeError('unproven boundary leaf')
    def fields(value, names):
        for name in names:leaf(field(value, name))
    def sequence(value):
        if type(value) is not tuple:raise TypeError('mutable boundary sequence')
        edges.append(('tuple', value, len(value)))
        return value
    try:
        if type(result) is not driver.Result: return None
        state = field(result, 'state')
        if type(state) is not driver.DriverState:return None
        core = field(state, 'core')
        if type(core) is not driver.Core:return None
        engine = field(core, 'engine')
        if type(engine) is not BattleState:return None
        fields(engine, BattleState._fields)
        unsafe = field(core, 'unsafe')
        if unsafe is None:leaf(unsafe)
        else:
            for value in sequence(unsafe):leaf(value)
        fields(core, ('frame_number', 'admitted', 'terminal_closeout'))
        fields(field(core, 'cursor'), ('stream_id', 'turn', 'phase', 'generation'))
        for request in sequence(field(core, 'requests')):
            fields(field(field(field(request, 'actions'), 'facts'), 'request'), ('present', 'wait'))
        for owner in sequence(field(result, 'owners')):
            fields(owner, ('reason', 'basic_error', 'retained_error', 'full_line_consumers_complete'))
            fields(field(field(owner, 'encoded'), 'observation'), ('admitted', *SHAPES))
        report = field(core, 'frame_input_report')
        if report is None:leaf(report)
        else:fields(report, ('raw_count', 'raw_words', 'event_turn', 'event_phase', 'event_generation',
            'switch_details', 'refusal', 'owner_full_words', 'owner_full_valid', 'owner_full_more'))
        for name, selected in (('risks', 'risk'), ('execution', 'refusal')):
            value = field(core, name)
            if value is None:leaf(value)
            else:fields(value, (selected,))
        return tuple(edges)
    except Exception:
        return None


def _same_boundary_proof(before, after):
    if before is None or after is None or len(before) != len(after):return False
    for old, new in zip(before, after, strict=True):
        if old[0] != new[0] or old[2] != new[2]:return False
        if old[0] == 'scalar':
            if old[1] != new[1]:return False
        elif old[1] is not new[1]:return False
    return True


def encoded(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def digest_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def _row(value, index):
    # Slice BEFORE transfer, including B=8 where GLOBAL8 namespaces exist.
    return np.asarray(value[index:index+1])


def state_packet(state, row):
    if tuple(state._fields) != BattleState._fields:
        raise ValueError('actual complete BattleState required')
    return {name: _row(getattr(state, name), row).tolist() for name in BattleState._fields}


def _unsafe(core, row):
    values = getattr(core, 'unsafe', None)
    if values is None:
        return None
    if type(values) is not tuple or len(values) != 2:
        raise ValueError('two original per-owner unsafe carries required')
    result = []
    for value in values:
        scalar = np.asarray(value[row])
        if scalar.shape != () or scalar.dtype != np.bool_:
            raise ValueError('literal bool unsafe carry required')
        result.append(bool(scalar))
    return result


def request_history_rules(compiled):
    """Original static lookup namespace, for diagnostics only; never admission."""
    from . import engine_request_source as request
    from .engine_private_team import engine_metadata_binding
    record = request._checked(compiled)
    rules = record['rules']; origin = request._rules_origin(rules)
    metadata = json.loads(record['metadata'])
    if not np.array_equal(np.asarray(engine_metadata_binding(metadata)), np.asarray(rules.device.engine_binding)):
        raise ValueError('original request history vocabulary binding differs')
    flags = np.asarray(rules.device.unsupported_history)
    names = metadata['vocabulary']
    if flags.dtype != np.bool_ or flags.shape != (len(names['moves']),):
        raise ValueError('original bool unsupported-history namespace required')
    return dict(schema='pokezero.request-history-diagnostic.v1',
        request_binding=np.asarray(compiled.binding).tolist(),
        rules_binding=np.asarray(rules.device.binding).tolist(),
        engine_binding=np.asarray(rules.device.engine_binding).tolist(),
        source_revision=request.SHOWDOWN, generator_sha256=origin.generator_sha256,
        source_tree_sha256=hashlib.sha256(origin.source_tree_json.encode()).hexdigest(),
        compiled_tree_sha256=hashlib.sha256(origin.compiled_tree_json.encode()).hexdigest(),
        unsupported_history=flags.tolist(), unsupported_history_sha256=hashlib.sha256(flags.tobytes()).hexdigest(),
        engine_move_names=list(names['moves']), engine_species_names=list(names['species']),
        namespace='Literal BattleState indices, not public or request-mechanics IDs')


def _history_lookup(packet, rules):
    """Static lookup only; actual unsafe carries are copied separately."""
    flags = rules['unsupported_history']; state = packet['state']
    members = []
    for side in (0, 1):
        for physical, species in enumerate(state['species'][0][side]):
            if species <= 0:
                continue
            move = state['last_move'][0][side][physical]
            valid = 0 <= move < len(flags)
            members.append(dict(side=side, physical=physical, species=species, last_move=move,
                last_move_valid=valid, unsupported_history=flags[move] if valid else None))
    return members


def _full(report, row, owner):
    if report is None:
        return dict(available=False, reason='MISSING_ACTUAL_FULL_REPORT')
    words = np.asarray(report.owner_full_words[row, owner])
    valid = np.asarray(report.owner_full_valid[row, owner])
    more = np.asarray(report.owner_full_more[row, owner])
    if (words.ndim != 3 or words.shape[1:] != (64, 16) or words.dtype != np.int32
            or valid.shape != words.shape[:2] or valid.dtype != np.bool_
            or more.shape != words.shape[:1] or more.dtype != np.bool_):
        return dict(available=False, reason='MALFORMED_ACTUAL_FULL_REPORT')
    # Store original packet masks/more and only present words; never reproject.
    return dict(available=True, shape=list(words.shape),
        packets=[dict(valid=mask.tolist(), more=bool(remaining), words=packet[mask].tolist())
            for packet, mask, remaining in zip(words, valid, more, strict=True)])


def boundary(result, row):
    core = result.state.core
    report = getattr(core, 'frame_input_report', None)
    observations, owners = [], []
    for side, owner in enumerate(result.owners):
        observation = owner.encoded.observation
        admitted = bool(np.asarray(observation.admitted[row]))
        observations.append(dict(admitted=admitted,
            arrays={name: np.asarray(getattr(observation, name)[row]).tolist() for name in SHAPES}
                if admitted else None))
        owners.append({name: int(np.asarray(getattr(owner, name)[row])) for name in
            ('reason', 'basic_error', 'retained_error', 'full_line_consumers_complete')})
    raw = None
    if report is not None:
        count = int(np.asarray(report.raw_count[row]))
        if 0 <= count <= report.raw_words.shape[1]:
            raw = dict(words=np.asarray(report.raw_words[row, :count]).tolist(),
                count=count, turn=np.asarray(report.event_turn[row, :count]).tolist(),
                phase=np.asarray(report.event_phase[row, :count]).tolist(),
                generation=np.asarray(report.event_generation[row, :count]).tolist(),
                switch_details=np.asarray(report.switch_details[row, :count]).tolist(),
                witness_refusal=int(np.asarray(report.refusal[row])))
    return dict(state=state_packet(core.engine, row),
        unsafe=_unsafe(core, row),
        frame=int(np.asarray(core.frame_number[row])),
        cursor={name: np.asarray(getattr(core.cursor, name)[row]).tolist() for name in
            ('stream_id', 'turn', 'phase', 'generation')},
        core_admitted=bool(np.asarray(core.admitted[row])),
        terminal_closeout=None if getattr(core,'terminal_closeout',None) is None else bool(np.asarray(core.terminal_closeout[row])),
        requested=[bool(np.asarray(request.actions.facts.request.present[row])
            and not np.asarray(request.actions.facts.request.wait[row])) for request in core.requests],
        owners=owners, observations=observations, full=[_full(report, row, side) for side in (0, 1)], raw=raw,
        kernel_risk=None if core.risks is None else int(np.asarray(core.risks.risk[row])),
        execution_refusal=None if core.execution is None else int(np.asarray(core.execution.refusal[row])))


def selected_actions(result, actions, row, action_names):
    """Literal original request mechanics and fixed physical switch routing."""
    core = result.state.core
    selected = []
    for side, request in enumerate(core.requests):
        value = request.actions
        packet = value.facts.request
        ready = bool(np.asarray(packet.present[row]) and not np.asarray(packet.wait[row]))
        action = int(np.asarray(actions[row, side]))
        entry = dict(requested=ready, engine_action=action)
        if ready:
            if not 0 <= action < 9:
                raise ValueError('actual selected engine action outside nine-action layout')
            if action < 4:
                matches = np.flatnonzero((np.asarray(value.move_action_map[row]) == action)
                    & np.asarray(packet.move_present[row]) & np.asarray(value.features.legal[row, :4]))
                if len(matches) != 1:
                    raise ValueError('actual request-to-mechanics mapping is not unique')
                slot = int(matches[0]); mechanics = int(np.asarray(value.facts.mechanics[row, slot]))
                if not 0 < mechanics < len(action_names):
                    raise ValueError('actual bound mechanics namespace missing')
                entry.update(kind='move', request_slot=slot, mechanics=mechanics, move=action_names[mechanics])
            else:
                active = int(np.asarray(core.engine.active[row, side]))
                physical = [slot for slot in range(6) if slot != active][action - 4]
                entry.update(kind='switch', physical=physical)
        selected.append(entry)
    return selected


class TraceWriter:
    """Exclusive bounded writer. Methods are nonfatal after explicit creation."""
    def __init__(self, path, *, max_bytes=1024**3, max_record_bytes=4*1024**2, reuse_boundary_data=True):
        if (type(max_bytes) is not int or max_bytes < 4096
                or type(max_record_bytes) is not int or not 1024 <= max_record_bytes <= 16*1024**2):
            raise ValueError('explicit finite trace byte bounds required')
        if type(reuse_boundary_data) is not bool:raise ValueError('literal trace data reuse mode required')
        self.path = Path(path)
        self.stream = self.path.open('xb')
        self.max_bytes, self.max_record_bytes = max_bytes, max_record_bytes
        self.bytes = 0; self.chain = '0'*64; self.records = 0
        self.failures = []; self.disabled = False; self.closed = False
        self.header = None; self.attempts = {}; self.written = {}; self.terminals = {}
        self.initial_ms = 0.; self.advancing_ms = 0.
        self.unsafe_first = set()
        self._reuse_boundary_data = reuse_boundary_data
        self._boundary_result = None
        self._boundary_packets = {}

    def _before_boundary(self, result, row):
        if type(row) is not int and not isinstance(row, np.integer):return boundary(result, row)
        cached = self._boundary_packets.get(int(row))
        runtime = None if self.header is None else self.header.get('runtime_binding')
        if (self._reuse_boundary_data and result is self._boundary_result and cached is not None
                and type(runtime) is str and runtime == cached[0]
                and _same_boundary_proof(cached[1], _boundary_proof(result))):
            # Canonical bytes cannot be mutated by a payload consumer. Return a
            # fresh ordinary packet; never reuse an admission/consumer result.
            return json.loads(cached[2])
        return boundary(result, row)

    def _remember_boundary(self, result, packet):
        if not self._reuse_boundary_data:return None
        runtime = None if self.header is None else self.header.get('runtime_binding')
        if type(runtime) is not str:return None
        proof = _boundary_proof(result)
        if proof is None:return None
        try:return runtime, proof, encoded(packet)
        except Exception:return None  # Optimization cannot change capture errors.

    def _write(self, kind, payload, *, final=False):
        if self.disabled and not final:
            return False
        try:
            row = dict(kind=kind, previous_sha256=self.chain, payload=payload)
            raw = encoded(row) + b'\n'
            if len(raw) > self.max_record_bytes or (not final and self.bytes + len(raw) > self.max_bytes):
                raise ValueError('TRACE_CAPACITY')
            if self.stream.write(raw) != len(raw):
                raise OSError('short trace write')
            self.stream.flush()
            self.bytes += len(raw); self.chain = hashlib.sha256(raw).hexdigest(); self.records += 1
            return True
        except Exception as error:
            self.disabled = True
            if not self.failures: self.failures.append(dict(type=type(error).__name__, message=str(error)))
            return False

    def begin(self, *, initial, runtime_binding, source, pool_receipt, team_indices,
              stream_ids, max_requests, constructor_seed, sampling_seed, action_names, request_compiled=None):
        start = time.perf_counter_ns()
        try:
            if self.header is not None or self.closed:
                raise ValueError('trace begins exactly once')
            layout = {name: dict(dtype=str(getattr(initial.state.core.engine, name).dtype),
                shape=list(getattr(initial.state.core.engine, name).shape[1:])) for name in BattleState._fields}
            self.header = dict(schema=SCHEMA, runtime_binding=runtime_binding, source=source,
                pool_receipt=pool_receipt, cohort=[dict(stream_id=int(s), team_indices=list(map(int, pair)))
                    for s, pair in zip(stream_ids, team_indices, strict=True)],
                max_requests=max_requests, constructor_seed=constructor_seed, sampling_seed=sampling_seed,
                action_names=list(action_names), state_layout=layout,
                reference_chance='ORIGINAL_CAPTURED_CHANCE_INPUTS; existing fidelity semantic coupling; unmapped roles UNKNOWN',
                max_bytes=self.max_bytes, max_record_bytes=self.max_record_bytes)
            if request_compiled is not None:
                self.header['request_history_rules'] = request_history_rules(request_compiled)
            self.attempts = {int(s): 0 for s in stream_ids}; self.written = dict(self.attempts)
            self._write('HEADER', self.header)
            packets = {}
            for row, stream in enumerate(stream_ids):
                packet = boundary(initial, row)
                if self._write('INITIAL', dict(stream_id=int(stream), boundary=packet)):
                    remembered = self._remember_boundary(initial, packet)
                    if remembered is not None:packets[row] = remembered
            self._boundary_result = initial if packets and not self.disabled else None
            self._boundary_packets = packets if not self.disabled else {}
        except Exception as error:
            self.disabled = True; self.failures.append(dict(type=type(error).__name__, message=str(error)))
        self.initial_ms = (time.perf_counter_ns()-start)/1e6

    def decision(self, *, request, before, after, actions, active, executed, invalid, chance):
        start = time.perf_counter_ns()
        packets = {}
        for row in np.flatnonzero(active):
            try:
                stream = int(np.asarray(before.state.core.cursor.stream_id[row]))
                self.attempts[stream] += 1
                if self.attempts[stream] > self.header['max_requests']:
                    raise ValueError('declared decision bound exceeded')
                # Terminal evidence is bounded by the original roster and copied
                # before benchmark completion publication, even if IO is capped.
                engine = after.state.core.engine
                if bool(executed[row] and not invalid[row] and np.asarray(engine.done[row])):
                    self.terminals[stream] = dict(stream_id=stream,
                        frame=int(np.asarray(after.state.core.frame_number[row])), executed=True,
                        invalid=False, core_admitted=bool(np.asarray(after.state.core.admitted[row])),
                        terminal_closeout=None if getattr(after.state.core,'terminal_closeout',None) is None else bool(np.asarray(after.state.core.terminal_closeout[row])),
                        done=bool(np.asarray(engine.done[row])), winner=int(np.asarray(engine.winner[row])),
                        hp=np.asarray(engine.hp[row]).tolist(), max_hp=np.asarray(engine.max_hp[row]).tolist(),
                        engine_turn=int(np.asarray(engine.turn[row])),
                        owner_admission=[{name:np.asarray(getattr(owner,name)[row]).item() for name in
                            ('reason','basic_error','retained_error','full_line_consumers_complete')}
                            for owner in after.owners],
                        runtime_binding=self.header['runtime_binding'], source=self.header['source'])
                if self.disabled:
                    continue
                payload = dict(stream_id=stream, request=int(request),
                    before=self._before_boundary(before, row), after=boundary(after, row),
                    chance={name:_row(value,row).tolist() for name,value in chance._asdict().items()},
                    selected=selected_actions(before, actions, row, self.header['action_names']),
                    executed=bool(executed[row]), invalid=bool(invalid[row]))
                first = []
                old, new = payload['before']['unsafe'], payload['after']['unsafe']
                if old is not None and new is not None:
                    first = [owner for owner in (0, 1) if not old[owner] and new[owner]
                        and (stream, owner) not in self.unsafe_first]
                rules = self.header.get('request_history_rules')
                if first:
                    payload['unsafe_history_first_transition'] = dict(owners=first,
                        before_lookup=None if rules is None else _history_lookup(payload['before'], rules),
                        after_lookup=None if rules is None else _history_lookup(payload['after'], rules),
                        rules_available=rules is not None)
                if self._write('DECISION', payload):
                    self.written[stream] += 1
                    self.unsafe_first.update((stream, owner) for owner in first)
                    remembered = self._remember_boundary(after, payload['after'])
                    if remembered is not None:packets[int(row)] = remembered
            except Exception as error:
                self.disabled = True
                if not self.failures: self.failures.append(dict(type=type(error).__name__, message=str(error)))
        # Only the current Result is retained, never inactive/history carries.
        self._boundary_result = after if packets and not self.disabled else None
        self._boundary_packets = packets if not self.disabled else {}
        milliseconds = (time.perf_counter_ns()-start)/1e6
        self.advancing_ms += milliseconds
        return milliseconds

    def finish(self, cohort):
        start = time.perf_counter_ns()
        if self.closed: raise ValueError('trace closes exactly once')
        self._write('END', dict(cohort=cohort, attempts=self.attempts, written=self.written,
            failures=self.failures), final=True)
        digest = size = None
        try:
            self.stream.close()
            digest, size = digest_file(self.path), self.path.stat().st_size
        except Exception as error:
            self.disabled = True
            self.failures.append(dict(type=type(error).__name__, message=str(error)))
        self.closed = True
        self._boundary_result = None
        self._boundary_packets = {}
        return dict(schema=SCHEMA, path=str(self.path), sha256=digest,
            bytes=size, maximum_total_bytes=self.max_bytes+self.max_record_bytes,
            complete=not self.disabled, failures=self.failures, attempts=self.attempts, written=self.written,
            records=self.records, runtime_binding=self.header.get('runtime_binding') if self.header else None,
            terminals=list(self.terminals.values()), initial_capture_ms=self.initial_ms,
            advancing_capture_ms=self.advancing_ms,
            finalize_ms=(time.perf_counter_ns()-start)/1e6,
            timing='Decision capture INCLUDED in advancing/full-step wall; initial/finalize separate. Never uninstrumented throughput.',
            full_B4_qualified=False)
