"""Closed-command artifact verification for the sealed Native prefix consumer.

This layer accepts independently admitted source/method/command bindings and
verifies the saved original IO. It does not launch processes, stage source,
decode device State/Trackers, construct references, or qualify physical parity.
The fixed-input program builder and live recorder remain separate work.
"""
import base64
import copy
from dataclasses import dataclass, field
import hashlib
import json
import re
import struct

from . import fresh200_witness_audit as native
from .fresh200_hook_admission import HookAdmissionBinding, HookCursor, HookPolicy

CHANNELS = ('omniscient', 'spectator', 'p1', 'p2', 'p3', 'p4')
READERS = ('omniscient', 'p1', 'p2')
METHODS = frozenset(('source.iterator', 'source.next', 'source.write', 'source.push', 'source.end',
    'source.error', 'stream.push', 'stream.end', 'stream.error'))
EVENT = frozenset(('schema', 'session', 'battle_id', 'source_epoch', 'command_id', 'sequence',
    'previous_hash', 'operation', 'evidence', 'hash'))
COMPLETION = frozenset(('schema', 'session', 'battle_id', 'source_epoch', 'command_id', 'phase',
    'source_registry_hash', 'stdin_sha256', 'control_hash', 'native_receipt_hash', 'snapshot_hash',
    'previous_completion_hash', 'event_range', 'event_head', 'producer_frontier', 'fanout_ack_frontier',
    'streams', 'stdout_frontier', 'stdout_hash', 'source_eof', 'require_eof', 'poisoned', 'qualification', 'hash'))
MAX_BYTES = 64 * 1024**2
MAX_EVENTS = 100000
EVIDENCE = {
    'SOURCE_REGISTERED': {'methods'}, 'STREAMS_REGISTERED': {'names', 'required_readers'},
    'SOURCE_PUSH': {'producer_sequence', 'type', 'side', 'request_call_id', 'raw'}, 'SOURCE_WRITE_RETURN': set(),
    'SOURCE_DELIVERED': {'producer_sequence'}, 'STREAM_PUSH': {'stream', 'stream_sequence', 'producer_sequence', 'raw'},
    'FANOUT_ACK': {'producer_sequence'}, 'REQUEST_CALL_BEGIN': {'call_id', 'side'}, 'REQUEST_CALL_END': {'call_id'},
    'READER_BEGIN': {'stream', 'stream_sequence'}, 'STDOUT_WRITE': {'ordinal', 'wire'},
    'REQUEST_CONSUMED': {'owner', 'generation', 'emit_call_id', 'emit_command_id', 'producer_sequence', 'stream_sequence', 'line_sha256', 'classification'},
    'READER_END': {'stream', 'stream_sequence', 'protocol_lines'}, 'SOURCE_PUSH_END': set(), 'SOURCE_EOF': set(),
    'STREAM_PUSH_END': {'stream'}, 'READER_EOF': {'stream'}, 'CONTROL_SCHEDULED': {'id', 'kind'}, 'CONTROL_FINISHED': {'id', 'discarded'},
}
ORDINALS = frozenset(('producer_sequence', 'stream_sequence', 'call_id', 'ordinal', 'generation', 'emit_call_id', 'protocol_lines', 'id'))


def _need(value, message):
    if not value:raise ValueError(message)


def _canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False).encode('utf8')


def _line(stream):
    wire = stream.readline(MAX_BYTES+1)
    _need(type(wire) is bytes and 0 < len(wire) <= MAX_BYTES and wire.endswith(b'\n')
        and not wire.endswith(b'\r\n'), 'bounded exact UTF8 JSON line required')
    return wire, native.document(wire.decode('utf8'))


def _framed(digest, value):
    digest.update(struct.pack('<Q', len(value)));digest.update(value)


def _raw(value):
    _need(type(value) is dict and set(value) == {'utf8_base64', 'bytes', 'sha256', 'split_segments'},
        'complete original chunk identity required')
    try:data = base64.b64decode(value['utf8_base64'], validate=True)
    except (ValueError, TypeError) as error:raise ValueError('literal chunk base64 differs') from error
    text = data.decode('utf8')
    _need(type(value['bytes']) is int and len(data) == value['bytes'] and native.digest(data) == value['sha256']
        and type(value['split_segments']) is int and len(text.split('\n')) == value['split_segments'],
        'literal chunk bytes/hash/segments differ')
    return text, data


@dataclass(frozen=True)
class CommandSpec:
    """Immutable bytes from the independent sealed input/source admission."""
    stdin_wire: bytes
    native_control_wire: bytes
    transport_control_wire: bytes
    battle_id: str
    hook_identity_digest: object = None
    hook_admission: object = None

    def documents(self):
        _need(all(type(x) is bytes and x.endswith(b'\n') and b'\r' not in x and len(x) <= MAX_BYTES
            for x in (self.stdin_wire, self.native_control_wire, self.transport_control_wire)),
            'exact independently sealed control/stdin bytes required')
        _need(b'\n' not in self.stdin_wire[:-1], 'one original stdin JSON line required')
        command = native.document(self.stdin_wire.decode('utf8'))
        control = native.control(native.document(self.native_control_wire.decode('utf8')))
        bound = native.document(self.transport_control_wire.decode('utf8'))
        _need(type(command) is dict and type(self.battle_id) is str and self.battle_id
            and command.get('battleId') == self.battle_id and command.get('type') == control['phase']
            and ((native._hex(self.hook_identity_digest) and self.hook_admission is None)
                or self.hook_identity_digest is None and type(self.hook_admission) is HookAdmissionBinding),
            'independent Native command/source address differs')
        keys = {'schema', 'native_nonce', 'stdin_sha256', 'command', 'previous_completion_hash'}
        _need(type(bound) is dict and set(bound) == keys and bound['schema'] == 'pokezero.native_transport_control.v1'
            and bound['native_nonce'] == control['nonce'] and native._same(bound['command'], command)
            and bound['stdin_sha256'] == native.digest(self.stdin_wire)
            and native._hex(bound['previous_completion_hash']), 'independent transport control binding differs')
        return command, control, bound


@dataclass
class _Stream:
    chunks: list = field(default_factory=list)
    consumed: int = 0
    open: object = None
    ended: bool = False
    reader_ended: bool = False
    eof_stdout: bool = False
    produced_hash: object = field(default_factory=hashlib.sha256)
    consumed_hash: object = field(default_factory=hashlib.sha256)
    prefix: object = field(default_factory=hashlib.sha256)
    lines: int = 0
    generation: int = 0
    actionable: bool = False

    def fork(self):
        result = copy.copy(self);result.chunks = copy.deepcopy(self.chunks);result.open = copy.deepcopy(self.open)
        for name in ('produced_hash', 'consumed_hash', 'prefix'):setattr(result, name, getattr(self, name).copy())
        return result

    def summary(self):
        return dict(produced=len(self.chunks), consumed=self.consumed, produced_hash=self.produced_hash.hexdigest(),
            consumed_hash=self.consumed_hash.hexdigest(), protocol_lines=self.lines, prefix_hash=self.prefix.hexdigest(),
            request_generation=self.generation, actionable=self.actionable, ended=self.ended, reader_ended=self.reader_ended)


class TransportAudit:
    """Replay the source-bound IO evidence without executing Native again."""
    def __init__(self, *, session, battle_id, source_registry_hash, expected_methods):
        _need(type(session) is str and session and type(battle_id) is str and battle_id
            and native._hex(source_registry_hash), 'independent transport identity required')
        _need(type(expected_methods) is dict and set(expected_methods) == METHODS
            and all(native._hex(x) for x in expected_methods.values()), 'independent full method-source roster required')
        self.session = session;self.battle_id = battle_id;self.source_registry_hash = source_registry_hash
        self.expected_methods = dict(expected_methods);self.sequence = 0;self.head = native.ZERO;self.event_bytes = 0;self.next_callback = 0
        self.previous_completion = native.ZERO;self.producers = [];self.delivered = 0;self.ack = 0;self.current = None
        self.source_registered = False;self.streams_registered = False;self.source_ended = False;self.source_eof = False
        self.streams = {name:_Stream() for name in CHANNELS};self.requests = {};self.request_stack = [];self.callbacks = {}
        self.stdout_count = 0;self.stdout_hash = hashlib.sha256()

    def fork(self):
        result = copy.copy(self);result.producers = copy.deepcopy(self.producers);result.requests = copy.deepcopy(self.requests)
        result.request_stack = list(self.request_stack);result.callbacks = dict(self.callbacks)
        result.streams = {name:value.fork() for name, value in self.streams.items()};result.stdout_hash = self.stdout_hash.copy()
        return result

    def _request(self, owner, chunk, line, command_id):
        key = (command_id, chunk['request'])
        _need(key in self.requests, 'request has no actual Native emitRequest source join')
        request = self.requests[key]
        _need(request['side'] == owner and request['returned'] and not request['joined']
            and request['producer'] == chunk['producer'], 'request generation/source call join differs')
        packet = native.document(line[len('|request|'):])
        _need(packet is None or type(packet) is dict and type(packet.get('side')) is dict
            and packet['side'].get('id') == owner, 'literal owning request identity differs')
        if packet is not None:
            _need('wait' not in packet or type(packet['wait']) is bool, 'literal owning wait bit required')
            _need('active' not in packet or type(packet['active']) is list and len(packet['active']) <= 1
                and all(type(x) is dict for x in packet['active']), 'literal singles active request required')
            _need('forceSwitch' not in packet or type(packet['forceSwitch']) is list and len(packet['forceSwitch']) <= 1
                and all(type(x) is bool for x in packet['forceSwitch']), 'literal singles replacement request required')
        state = self.streams[owner];state.generation += 1
        state.actionable = bool(packet is not None and not packet.get('wait', False)
            and (packet.get('active') or any(packet.get('forceSwitch', []))))
        request['joined'] = True
        return dict(owner=owner, generation=state.generation, emit_call_id=chunk['request'], emit_command_id=command_id,
            producer_sequence=chunk['producer'], stream_sequence=chunk['sequence'], line_sha256=native.digest(line.encode('utf8')),
            classification='null' if packet is None else 'wait' if packet.get('wait') else 'actionable' if state.actionable else 'inactive')

    def _event(self, operation, e, *, command_id, stdout, joined, snapshots):
        _need(operation in EVIDENCE and set(e) == EVIDENCE[operation], 'complete closed transport event evidence required')
        _need(all(type(e[k]) is int and e[k] >= 0 for k in ORDINALS & e.keys()), 'literal transport ordinals required')
        if operation == 'SOURCE_REGISTERED':
            _need(not self.source_registered, 'Native source was registered more than once')
            wanted = {k:v for k, v in self.expected_methods.items() if k.startswith('source.')}
            _need(type(e['methods']) is list and all(type(x) is dict and set(x) == {'name', 'sha256'} for x in e['methods']),
                'complete original source method identities required')
            actual = {x['name']:x['sha256'] for x in e['methods']}
            _need(len(actual) == len(e['methods']) and actual == wanted, 'observed source methods differ from independent admission')
            self.source_registered = True
        elif operation == 'STREAMS_REGISTERED':
            _need(self.source_registered and not self.streams_registered and e == dict(names=list(CHANNELS), required_readers=list(READERS)),
                'exact original six-stream/three-reader registration required');self.streams_registered = True
        elif operation == 'SOURCE_PUSH':
            _need(self.source_registered and not self.source_ended and e['producer_sequence'] == len(self.producers)+1,
                'Native producer sequence differs')
            value, _ = _raw(e['raw']);kind, separator, body = value.partition('\n')
            _need(separator and kind == e['type'] and kind in ('update', 'sideupdate', 'end'), 'Native producer type differs')
            side = None;request = None
            if kind == 'sideupdate':
                side, separator, payload = body.partition('\n');_need(separator and side in CHANNELS[2:], 'Native sideupdate address differs')
                if any(x.startswith('|request|') for x in payload.split('\n')):
                    _need(len(payload.split('\n')) == 1 and self.request_stack, 'unjoined Native request emission')
                    request = self.request_stack[-1];entry = self.requests[(command_id, request)]
                    _need(entry['side'] == side and entry['producer'] is None, 'Native request producer join differs')
                    entry['producer'] = e['producer_sequence']
            _need(e['side'] == side and native._same(e['request_call_id'], request), 'Native emitted call/address metadata differs')
            self.producers.append(dict(value=value, type=kind, side=side, request=request, fanout=[]))
        elif operation == 'SOURCE_WRITE_RETURN':_need(e == {}, 'closed original write return required')
        elif operation == 'SOURCE_DELIVERED':
            _need(self.current is None and e == dict(producer_sequence=self.delivered+1) and self.delivered < len(self.producers),
                'actual Native producer delivery FIFO differs');self.delivered += 1;self.current = self.delivered
        elif operation == 'STREAM_PUSH':
            _need(self.streams_registered and self.current == e['producer_sequence'], 'owner output lacks actual fanout interval')
            producer = self.producers[self.current-1];order = CHANNELS if producer['type'] == 'update' else (producer['side'],) if producer['type'] == 'sideupdate' else ()
            _need(len(producer['fanout']) < len(order) and order[len(producer['fanout'])] == e['stream'], 'Native fanout order/cardinality differs')
            state = self.streams[e['stream']];_need(not state.ended and e['stream_sequence'] == len(state.chunks)+1, 'owner chunk sequence differs')
            value, data = _raw(e['raw']);producer['fanout'].append(e['stream']);_framed(state.produced_hash, data)
            state.chunks.append(dict(sequence=e['stream_sequence'], producer=self.current, request=producer['request'], value=value, stdout=False))
        elif operation == 'FANOUT_ACK':
            _need(self.current == e['producer_sequence'] == self.ack+1, 'subsequent-next fanout acknowledgement differs')
            producer = self.producers[self.current-1];wanted = list(CHANNELS) if producer['type'] == 'update' else [producer['side']] if producer['type'] == 'sideupdate' else []
            _need(producer['fanout'] == wanted, 'incomplete original Native fanout');self.ack = self.current;self.current = None
        elif operation == 'REQUEST_CALL_BEGIN':
            key = (command_id, e['call_id']);_need(e['side'] in ('p1', 'p2') and key not in self.requests, 'duplicate or foreign Native request call')
            self.requests[key] = dict(side=e['side'], producer=None, returned=False, joined=False);self.request_stack.append(e['call_id'])
        elif operation == 'REQUEST_CALL_END':
            _need(self.request_stack and self.request_stack.pop() == e['call_id'], 'Native request call stack differs')
            entry = self.requests[(command_id, e['call_id'])];_need(entry['producer'] is not None and not entry['returned'], 'Native request return/source differs');entry['returned'] = True
        elif operation == 'READER_BEGIN':
            _need(e['stream'] in READERS, 'unregistered original bridge reader');state = self.streams[e['stream']]
            _need(state.open is None and not state.reader_ended and state.consumed < len(state.chunks)
                and state.chunks[state.consumed]['sequence'] == e['stream_sequence'], 'reader chunk FIFO differs')
            state.open = copy.deepcopy(state.chunks[state.consumed])
        elif operation == 'STDOUT_WRITE':
            value, data = _raw(e['wire']);_need(e['ordinal'] == self.stdout_count+1, 'original stdout ordinal differs')
            captured, packet = _line(stdout);_need(captured == data and value.endswith('\n'), 'original captured stdout wire differs')
            _need(type(packet) is dict and ('battleId' not in packet or packet['battleId'] == self.battle_id), 'stdout battle identity differs')
            if packet.get('type') == 'stream':
                _need(packet.get('stream') in READERS, 'stdout packet is not an actual bridge reader');state = self.streams[packet['stream']]
                expected = [x for x in state.open['value'].split('\n') if x] if state.open else None
                _need(expected and not state.open['stdout'] and native._same(packet.get('lines'), expected), 'original stdout/private stream join differs')
                state.open['stdout'] = True
            elif packet.get('type') == 'stream_end':
                _need(packet.get('stream') in READERS, 'stdout EOF is not an actual reader');state = self.streams[packet['stream']]
                _need(state.ended and not state.reader_ended and not state.eof_stdout, 'original stdout EOF join differs');state.eof_stdout = True
            elif packet.get('type') == 'snapshot':snapshots.append(packet)
            self.stdout_count += 1;_framed(self.stdout_hash, data)
        elif operation == 'REQUEST_CONSUMED':
            _need(e['owner'] in ('p1', 'p2'), 'actual owning request required');state = self.streams[e['owner']];chunk = state.open
            _need(chunk is not None, 'request has no original reader chunk')
            lines = [x for x in chunk['value'].split('\n') if x.startswith('|request|')]
            _need(len(lines) == 1, 'one exact literal request emission required')
            wanted = self._request(e['owner'], chunk, lines[0], command_id)
            _need(native._same(e, wanted), 'actual request generation/classification differs');joined.add(e['emit_call_id'])
        elif operation == 'READER_END':
            _need(e['stream'] in READERS, 'actual bridge reader required');state = self.streams[e['stream']];chunk = state.open
            _need(chunk and chunk['sequence'] == e['stream_sequence'], 'reader completion token differs')
            lines = [x for x in chunk['value'].split('\n') if x]
            _need(e['protocol_lines'] == len(lines) and (bool(chunk['stdout']) if lines else not chunk['stdout']), 'original stdout completion differs')
            for line in lines:
                state.prefix.update((line+'\n').encode('utf8'));state.lines += 1
                if line == '|tie' or line.startswith('|win|'):state.actionable = False
            _framed(state.consumed_hash, chunk['value'].encode('utf8'));state.consumed += 1;state.open = None
        elif operation == 'SOURCE_PUSH_END':
            _need(not self.source_ended and e == {}, 'source closure observed twice');self.source_ended = True
        elif operation == 'SOURCE_EOF':
            _need(self.source_ended and not self.source_eof and self.delivered == len(self.producers) and e == {}, 'Native source EOF differs');self.source_eof = True
        elif operation == 'STREAM_PUSH_END':
            _need(e['stream'] in CHANNELS, 'actual Native stream EOF required');state = self.streams[e['stream']]
            _need(self.source_eof and not state.ended, 'owner EOF lacks original source closure');state.ended = True
        elif operation == 'READER_EOF':
            _need(e['stream'] in READERS, 'actual reader EOF required');state = self.streams[e['stream']]
            _need(state.ended and state.eof_stdout and not state.reader_ended and state.open is None
                and state.consumed == len(state.chunks), 'reader EOF/transcript differs');state.reader_ended = True
        elif operation == 'CONTROL_SCHEDULED':
            _need(e['id'] == self.next_callback and e['kind'] in ('ready', 'terminal'), 'original scheduled callback differs')
            self.next_callback += 1;self.callbacks[e['id']] = e['kind']
        elif operation == 'CONTROL_FINISHED':
            _need(e['id'] in self.callbacks and type(e['discarded']) is bool, 'original callback completion differs');del self.callbacks[e['id']]
        else:raise ValueError('unbound Native transport operation: '+operation)

    def verify_command(self, events, receipt_wire, spec, *, stdout, native_receipt_hash, native_receipt, native_rows):
        """Mutates this candidate only; ClosedCommandVerifier commits atomically."""
        _need(type(receipt_wire) is bytes and receipt_wire.endswith(b'\n') and not receipt_wire.endswith(b'\r\n')
            and len(receipt_wire) <= MAX_BYTES, 'exact transport completion bytes required')
        command, control, bound = spec.documents();receipt = native.document(receipt_wire.decode('utf8'))
        _need(type(receipt) is dict and set(receipt) == COMPLETION, 'complete closed transport receipt schema required')
        _need(native._hex(receipt.get('hash')), 'transport completion hash required')
        body = {k:v for k, v in receipt.items() if k != 'hash'}
        _need(native.digest(_canonical(body)) == receipt['hash'], 'transport completion bytes/hash differ')
        _need(receipt.get('schema') == 'pokezero.native_transport_completion.v1' and receipt.get('session') == self.session
            and receipt.get('battle_id') == spec.battle_id == self.battle_id and receipt.get('source_registry_hash') == self.source_registry_hash
            and receipt.get('command_id') == control['nonce'] and receipt.get('phase') == control['phase']
            and receipt.get('stdin_sha256') == bound['stdin_sha256'] and receipt.get('control_hash') == native.digest(spec.transport_control_wire)
            and receipt.get('previous_completion_hash') == bound['previous_completion_hash'] == self.previous_completion
            and receipt.get('native_receipt_hash') == native_receipt_hash and receipt.get('source_epoch') == 1
            and type(receipt.get('source_epoch')) is int and receipt.get('poisoned') is False and receipt.get('qualification') is False,
            'independent command/source/receipt completion binding differs')
        span = receipt.get('event_range');_need(type(span) is list and len(span) == 2 and all(type(x) is int for x in span)
            and span[0] == self.sequence < span[1] <= MAX_EVENTS, 'bounded complete transport event span required')
        started = ended = False;joined = set();snapshot_hash = None;snapshots = []
        while self.sequence < span[1]:
            wire, row = _line(events)
            _need(type(row) is dict and set(row) == EVENT and row['schema'] == 'pokezero.native_transport_event.v1'
                and row['session'] == self.session and row['battle_id'] == self.battle_id and type(row['source_epoch']) is int and row['source_epoch'] == 1
                and row['command_id'] == control['nonce'] and type(row['sequence']) is int and row['sequence'] == self.sequence
                and row['previous_hash'] == self.head and native._hex(row['hash'])
                and row['hash'] == native.digest(_canonical({k:v for k, v in row.items() if k != 'hash'})),
                'exact source-bound transport event chain differs')
            self.event_bytes += len(_canonical({k:v for k, v in row.items() if k != 'hash'}))
            _need(self.event_bytes <= MAX_BYTES, 'bounded cumulative transport evidence required')
            operation = row['operation'];e = row['evidence'];_need(type(e) is dict, 'literal transport evidence required')
            if operation == 'COMMAND_RECEIVED':
                _need(not started and self.sequence == span[0] and e == dict(phase=control['phase'], received_line=spec.stdin_wire[:-1].decode('utf8'),
                    stdin_sha256=bound['stdin_sha256'], control_hash=receipt['control_hash']), 'original received command bytes differ');started = True
            elif operation == 'COMMAND_RETURN':
                _need(started and not ended and set(e) == {'original_threw', 'native_receipt_hash', 'snapshot_hash'}
                    and e['original_threw'] is False and e['native_receipt_hash'] == native_receipt_hash
                    and (e['snapshot_hash'] is None or native._hex(e['snapshot_hash'])), 'original command outcome differs')
                snapshot_hash = e['snapshot_hash'];ended = True
            else:
                _need(started, 'transport evidence precedes exact command admission')
                self._event(operation, e, command_id=control['nonce'], stdout=stdout, joined=joined, snapshots=snapshots)
            self.sequence += 1;self.head = row['hash']
        emitted = {}
        for row in native_rows:
            frame = row.get('frame')
            if row['operation'] == 'CALL_ENTER' and frame and frame['operation'] == 'Side.emitRequest':
                subject = row['evidence']['subject'];side = subject.get('n')
                _need(subject.get('kind') == 'Side' and type(side) is int and side in (0, 1), 'literal causal emitRequest side required')
                emitted[frame['callId']] = 'p'+str(side+1)
        observed = {key[1]:value for key, value in self.requests.items() if key[0] == control['nonce']}
        _need(set(emitted) == set(observed) == joined and all(observed[k]['side'] == value for k, value in emitted.items()),
            'transport and actual causal emitRequest rosters differ')
        eof = control['phase'] in ('end', 'close')
        _need(started and ended and self.source_registered and self.streams_registered and not self.request_stack and not self.callbacks
            and self.current is None and self.ack == len(self.producers) and all(x['returned'] and x['joined'] for x in self.requests.values())
            and all(self.streams[name].open is None and self.streams[name].consumed == len(self.streams[name].chunks) for name in READERS),
            'incomplete original Native producer/reader/request/control drain')
        _need(not eof or self.source_eof and all(x.ended for x in self.streams.values()) and all(self.streams[name].reader_ended for name in READERS),
            'final source/owner/reader EOF proof is incomplete')
        if control['phase'] == 'snapshot':
            _need(len(snapshots) == 1 and type(command.get('requestId')) is str and command['requestId']
                and snapshots[0].get('requestId') == command['requestId'], 'one exact addressed Native snapshot required')
            snapshot = snapshots[0]['snapshot'];witness = snapshot['witness'];boundary = native_receipt['boundary']
            _need(native._same(witness['receipt'], native_receipt) and boundary['kind'] == 'snapshot'
                and boundary['requestId'] == command['requestId'] and native._same(boundary['wire'], witness['wire'])
                and snapshot_hash == witness['wire']['state_sha256'], 'early snapshot and final command completion differ')
            native.verify_snapshot_wire(witness['wire'], snapshot)
        else:_need(not snapshots and snapshot_hash is None, 'unbound snapshot outside snapshot command')
        wanted = dict(body, event_head=self.head, producer_frontier=len(self.producers), fanout_ack_frontier=self.ack,
            streams={name:value.summary() for name, value in self.streams.items()}, stdout_frontier=self.stdout_count,
            stdout_hash=self.stdout_hash.hexdigest(), source_eof=self.source_eof, require_eof=eof, snapshot_hash=snapshot_hash)
        _need(native._same(body, wanted), 'transport completion summary differs from actual IO evidence')
        self.previous_completion = receipt['hash']
        return receipt


class ClosedCommandVerifier:
    """Source-admitted causal+transport verification with atomic cursors.

    The caller preserves exact artifact files before invoking this verifier.
    I/O/verification retries reuse these artifacts; no Native call is made.
    Supplying independent hook/source admission is mandatory, never replaced by
    a digest copied from the running witness. No physical/reference admission
    or fixed-input recording program is implemented by this class.
    """
    def __init__(self, *, session, battle_id, source_registry_hash, expected_methods, call_operations, hook_policy=None):
        _need(isinstance(call_operations, (set, frozenset)) and call_operations
            and all(type(x) is str and x for x in call_operations), 'independent causal operation roster required')
        self.call_operations = frozenset(call_operations);self.source_registry_hash = source_registry_hash
        _need(hook_policy is None or type(hook_policy) is HookPolicy and self.call_operations==hook_policy.operations,
            'independent complete catalog operation roster required')
        self.hook_policy=hook_policy;self.hook_cursor=HookCursor() if hook_policy is not None else None
        self.native_cursor = native.Cursor();self.transport_cursor = TransportAudit(session=session, battle_id=battle_id,
            source_registry_hash=source_registry_hash, expected_methods=expected_methods);self.completed = []

    def admit_spec(self,spec):
        _need(isinstance(spec,CommandSpec),'independently sealed command specification required')
        documents=spec.documents()
        if self.hook_policy is None:_need(spec.hook_admission is None,'registration policy was not independently admitted')
        else:
            _need(spec.hook_identity_digest is None,'registration mode cannot adopt a runtime exact digest')
            self.hook_cursor.admit_binding(spec.hook_admission,self.hook_policy)
        return documents

    def verify(self, spec, *, audit, native_receipt_wire, events, transport_receipt_wire, stdout):
        _need(isinstance(spec, CommandSpec), 'independently sealed command specification required')
        _, control, _ = self.admit_spec(spec);candidate_native = copy.deepcopy(self.native_cursor);candidate_transport = self.transport_cursor.fork()
        candidate_hooks=None
        expected = dict(control=control, battle_id=spec.battle_id, hook_identity_digest=spec.hook_identity_digest)
        positions = [(stream, stream.tell()) for stream in (audit, events, stdout)]
        try:
            extras=('SNAPSHOT_BEGIN','SNAPSHOT_SERIALIZED')
            if self.hook_policy is not None:
                raw=native.document(native_receipt_wire.decode('utf8'));count=raw.get('rowCount') if type(raw) is dict else None
                _need(type(count) is int and 2<=count<=native.MAX_ROWS,'bounded registration command scan required')
                start=audit.tell();scanned=[];size=0
                for _ in range(count):
                    wire,row=native._line(audit);size+=len(wire)
                    _need(size<=native.MAX_COMMAND_BYTES,'bounded registration command bytes required');scanned.append((wire,row))
                candidate_hooks=self.hook_cursor.candidate_command(scanned,policy=self.hook_policy,phase=control['phase'])
                audit.seek(start);expected['hook_identity_digest']=candidate_hooks.digest;extras=(*extras,'HOOK_REGISTERED')
            rows = native.verify_command(audit, native_receipt_wire, expected, cursor=candidate_native,
                source_registry_hash=self.source_registry_hash, call_operations=self.call_operations,
                extra_operations=extras)
            receipt = candidate_transport.verify_command(events, transport_receipt_wire, spec, stdout=stdout,
                native_receipt_hash=native.digest(native_receipt_wire), native_receipt=native.document(native_receipt_wire.decode('utf8')), native_rows=rows)
        except BaseException:
            for stream, position in positions:stream.seek(position)
            raise
        # Publish cursor movement only after both independently bound checks.
        self.native_cursor = candidate_native;self.transport_cursor = candidate_transport
        if candidate_hooks is not None:self.hook_cursor=candidate_hooks
        result = dict(status='RAW_CLOSED_COMMAND_VERIFIED_ONLY', command_id=control['nonce'], phase=control['phase'],
            native_receipt_sha256=native.digest(native_receipt_wire), transport_receipt_hash=receipt['hash'],
            native_rows=len(rows), transport_event_head=candidate_transport.head, qualification=False,
            open_gates=['full fixed-prefix closure', 'independent73-field physical replay', 'literal owning references', 'actual-device parity'])
        self.completed.append(result);return result

    def finish_artifacts(self, *, audit, events, stdout, expected_command_ids):
        """Close saved IO files only; the fixed scientific program is a separate gate."""
        _need(type(expected_command_ids) in (list, tuple) and expected_command_ids
            and all(type(x) is str and x for x in expected_command_ids)
            and len(set(expected_command_ids)) == len(expected_command_ids)
            and [x['command_id'] for x in self.completed] == list(expected_command_ids),
            'independently declared complete command roster required')
        _need(self.completed[-1]['phase'] == 'close' and self.transport_cursor.source_eof
            and not self.native_cursor.frames and all(self.transport_cursor.streams[n].reader_ended for n in READERS),
            'explicit final close and original source/reader EOF required')
        positions = [(stream, stream.tell()) for stream in (audit, events, stdout)]
        try:
            _need(not any(stream.read(1) for stream, _ in positions), 'unaccounted saved Native IO tail')
        finally:
            for stream, position in positions:stream.seek(position)
        return dict(status='RAW_CLOSED_ARTIFACT_SET_VERIFIED_ONLY', qualification=False,
            commands=len(self.completed), native_rows=self.native_cursor.sequence, transport_events=self.transport_cursor.sequence,
            open_gates=['sealed fixed-input program applicability', 'independent73-field physical replay', 'literal owning references', 'actual-device parity'])
