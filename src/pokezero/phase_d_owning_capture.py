"""Opt-in append-only capture of genuine B2 owning streams for Phase D.

The hook receives one already isolated actor, never a bridge, opponent request,
model, RNG or outcome. It records accepted literal wire and prospective decision
boundaries without changing the actor's return values. Keep these files remote.
Capture failures stop the owning cell; they never drop a packet and continue.
"""
from contextlib import contextmanager
from contextvars import ContextVar
import hashlib
import json
import os
from pathlib import Path

SCHEMA = 'pokezero.phase-d.owning-wire.v1'
_SCOPE = ContextVar('phase_d_owning_capture', default=None)


def encoded(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


@contextmanager
def capture_to(directory, *, cell, declaration_sha256):
    """Wrap exactly one original endpoint cell; no runtime work at import."""
    if (type(cell) is not dict or type(cell.get('ordinal')) is not int or cell['ordinal'] < 0
            or type(declaration_sha256) is not str or len(declaration_sha256) != 64
            or any(c not in '0123456789abcdef' for c in declaration_sha256)):
        raise ValueError('original endpoint cell and declaration digest required')
    token = _SCOPE.set((Path(directory), json.loads(encoded(cell)), declaration_sha256))
    try:
        yield
    finally:
        _SCOPE.reset(token)


def capture_actor(actor):
    """Install on the SAME actor after construction; inactive scope is a no-op.

    Call at the original factory's return of LiveProtocolActor. Capture begins
    before begin_game, but a stream packet is saved only after its original B2
    acceptance succeeds. Each prospective decide call is saved before encoding
    or selection. No call reads a policy result or modifies the supplied RNG.
    """
    scope = _SCOPE.get()
    if scope is None:
        return actor
    from .live_protocol_actor import LiveProtocolActor
    if not isinstance(actor, LiveProtocolActor) or actor.status != 'IDLE' or hasattr(actor, '_phase_d_capture'):
        raise ValueError('one fresh original B2 actor required')
    directory, cell, declaration = scope
    directory.mkdir(parents=True, exist_ok=True)
    stream = (directory / (str(cell['ordinal']) + '.jsonl')).open('xb', buffering=0)
    prefix, seq, previous = hashlib.sha256(), 0, '0' * 64
    state = {'started': False, 'closed': False, 'decisions': set(), 'turn': 0, 'lines': 0}
    originals = {k: getattr(actor, k) for k in ('begin_game', 'accept_owner_stream', 'decide', 'close_game')}

    def emit(kind, **fields):
        nonlocal seq, previous
        row = dict(schema=SCHEMA, sequence=seq, previous_sha256=previous, event=kind, **fields)
        blob = encoded(row)
        # A partial write is a capture failure, never an omitted successful row.
        if stream.write(blob + b'\n') != len(blob) + 1:
            raise OSError('short owning capture write')
        previous = hashlib.sha256(blob).hexdigest()
        seq += 1

    def begin_game(**identity):
        result = originals['begin_game'](**identity)
        emit('BEGIN', cell=cell, declaration_sha256=declaration, namespace=actor.contract.namespace, **identity)
        state['started'] = True
        return result

    def accept_owner_stream(**packet):
        result = originals['accept_owner_stream'](**packet)
        lines = packet['lines']
        for line in lines:
            prefix.update(line.encode() + b'\n')
            if line.startswith('|turn|'):
                state['turn'] = int(line.split('|')[2])
        if prefix.hexdigest() != actor.prefix_sha256:
            raise ValueError('capture literal prefix differs from original actor')
        emit('WIRE', generation=actor.generation, start=state['lines'], end=state['lines'] + len(lines),
             prefix_sha256=prefix.hexdigest(), lines=lines)
        state['lines'] += len(lines)
        return result

    def decide(snapshot, *, rng):
        key = (snapshot.generation, snapshot.request_count)
        if key in state['decisions']:
            raise ValueError('captured original request cannot be retried')
        state['decisions'].add(key)
        emit('ROOT', decision_index=len(state['decisions']) - 1, turn=state['turn'],
             generation=snapshot.generation, request_count=snapshot.request_count,
             prefix_sha256=snapshot.prefix_sha256, own_request_line=snapshot.own_request_line)
        return originals['decide'](snapshot, rng=rng)

    def close_game():
        try:
            result = originals['close_game']()
            emit('CLOSE', lifecycle=result, line_count=state['lines'], root_count=len(state['decisions']))
            os.fsync(stream.fileno())
            return result
        finally:
            state['closed'] = True
            stream.close()

    actor._phase_d_capture = state
    actor.begin_game, actor.accept_owner_stream, actor.decide, actor.close_game = (
        begin_game, accept_owner_stream, decide, close_game)
    return actor


def read_capture(path):
    """Validate a remote append-only file and yield roots with literal prefixes.

    Interrupted files may yield their durable ROOT records. CLOSE and terminal
    outcomes are never required for eligibility; selection must not favor games
    that finished. No adjacent/synthetic request or missing prefix is repaired.
    """
    prior, prefix, lines, begin, last_generation = '0' * 64, hashlib.sha256(), [], None, 0
    decision = 0
    with Path(path).open('rb') as stream:
        for sequence, raw in enumerate(stream):
            if not raw.endswith(b'\n'):
                raise ValueError('partial capture row; preserve file, no repair')
            row = json.loads(raw)
            if (encoded(row) + b'\n' != raw or row.get('schema') != SCHEMA
                    or row.get('sequence') != sequence or row.get('previous_sha256') != prior):
                raise ValueError('owning capture chain differs')
            prior = hashlib.sha256(raw[:-1]).hexdigest()
            kind = row['event']
            if kind == 'BEGIN':
                if begin is not None or sequence != 0 or row['owner'] not in ('p1', 'p2'):
                    raise ValueError('one original owning capture registration required')
                begin = row
            elif begin is None:
                raise ValueError('original capture registration missing')
            elif kind == 'WIRE':
                delta = row['lines']
                if (type(delta) is not list or not delta or any(type(x) is not str or '\n' in x or '\r' in x for x in delta)
                        or row['start'] != len(lines) or row['end'] != len(lines) + len(delta)
                        or row['generation'] != last_generation + 1):
                    raise ValueError('literal capture delta/cursor differs')
                for line in delta:
                    if line.startswith('|request|'):
                        request = json.loads(line[9:])
                        if request.get('side', {}).get('id') != begin['owner']:
                            raise ValueError('foreign request in isolated owner capture')
                    if line.startswith('|split|'):
                        raise ValueError('omniscient split is not owning wire')
                    prefix.update(line.encode() + b'\n')
                if row['prefix_sha256'] != prefix.hexdigest():
                    raise ValueError('literal capture prefix hash differs')
                lines.extend(delta)
                last_generation = row['generation']
            elif kind == 'ROOT':
                requests = [x for x in lines if x.startswith('|request|')]
                if (row['generation'] != last_generation or row['prefix_sha256'] != prefix.hexdigest()
                        or not requests or row['own_request_line'] != requests[-1]
                        or row['request_count'] != len(requests) or row['decision_index'] != decision):
                    raise ValueError('root request/prefix differs from original capture')
                decision += 1
                yield dict(cell=begin['cell'], declaration_sha256=begin['declaration_sha256'],
                           battle_id=begin['battle_id'], owner=begin['owner'], seed=begin['seed'],
                           stream_id=begin['stream_id'], namespace=begin['namespace'],
                           **{k: row[k] for k in ('decision_index', 'turn', 'generation', 'request_count', 'prefix_sha256', 'own_request_line')},
                           lines=tuple(lines), capture_path=str(path), capture_row=sequence)
            elif kind != 'CLOSE':
                raise ValueError('unsupported owning capture event')
