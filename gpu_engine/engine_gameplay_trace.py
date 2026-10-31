"""Opt-in remote JSONL capture of actual retained decisions, never policy input.

One row is copied at a time, after its original core call. No engine, model,
writer/fold, RNG sampling or simulator runs here. Capacity/IO failures disable
capture and retain dropped counts; they never alter the experiment's decisions.
The caller chooses an explicit remote path. No default artifact is created.
"""
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from .state import BattleState
from pokezero.recorded_encoder_parity import SHAPES

SCHEMA = 'pokezero.actual-gameplay-trace.v1'


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
    def __init__(self, path, *, max_bytes=1024**3, max_record_bytes=4*1024**2):
        if (type(max_bytes) is not int or max_bytes < 4096
                or type(max_record_bytes) is not int or not 1024 <= max_record_bytes <= 16*1024**2):
            raise ValueError('explicit finite trace byte bounds required')
        self.path = Path(path)
        self.stream = self.path.open('xb')
        self.max_bytes, self.max_record_bytes = max_bytes, max_record_bytes
        self.bytes = 0; self.chain = '0'*64; self.records = 0
        self.failures = []; self.disabled = False; self.closed = False
        self.header = None; self.attempts = {}; self.written = {}; self.terminals = {}
        self.initial_ms = 0.; self.advancing_ms = 0.

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
              stream_ids, max_requests, constructor_seed, sampling_seed, action_names):
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
            self.attempts = {int(s): 0 for s in stream_ids}; self.written = dict(self.attempts)
            self._write('HEADER', self.header)
            for row, stream in enumerate(stream_ids):
                self._write('INITIAL', dict(stream_id=int(stream), boundary=boundary(initial, row)))
        except Exception as error:
            self.disabled = True; self.failures.append(dict(type=type(error).__name__, message=str(error)))
        self.initial_ms = (time.perf_counter_ns()-start)/1e6

    def decision(self, *, request, before, after, actions, active, executed, invalid, chance):
        start = time.perf_counter_ns()
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
                    before=boundary(before, row), after=boundary(after, row),
                    chance={name:_row(value,row).tolist() for name,value in chance._asdict().items()},
                    selected=selected_actions(before, actions, row, self.header['action_names']),
                    executed=bool(executed[row]), invalid=bool(invalid[row]))
                if self._write('DECISION', payload): self.written[stream] += 1
            except Exception as error:
                self.disabled = True
                if not self.failures: self.failures.append(dict(type=type(error).__name__, message=str(error)))
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
        return dict(schema=SCHEMA, path=str(self.path), sha256=digest,
            bytes=size, maximum_total_bytes=self.max_bytes+self.max_record_bytes,
            complete=not self.disabled, failures=self.failures, attempts=self.attempts, written=self.written,
            records=self.records, runtime_binding=self.header.get('runtime_binding') if self.header else None,
            terminals=list(self.terminals.values()), initial_capture_ms=self.initial_ms,
            advancing_capture_ms=self.advancing_ms,
            finalize_ms=(time.perf_counter_ns()-start)/1e6,
            timing='Decision capture INCLUDED in advancing/full-step wall; initial/finalize separate. Never uninstrumented throughput.',
            full_B4_qualified=False)
