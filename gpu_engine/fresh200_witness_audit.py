"""Verify exact Native witness bytes/command roster; never physical admission.

The caller supplies independently bound source/hook identities and the sealed
command roster. This module cannot certify that a witness covers every Native
operation. Native physics and literal owning-reference gates remain separate.
"""
import base64
from dataclasses import dataclass, field
import hashlib
import json
import math
import re

ZERO = '0'*64
MAX_LINE = 1024**2
MAX_COMMAND_BYTES = 64*1024**2
MAX_ROWS = 100000
CONTROL = frozenset(('schema','run','stream','reset','command','phase','request','nonce','chance_nonce','previous_receipt_hash'))
ROW = frozenset(('schema','run','stream','reset','command','nonce','phase','battleId','sequence','previousHash','operation','frame','evidence'))
RECEIPT = frozenset(('schema','control','battleId','firstSequence','lastSequence','rowCount','byteCount','chainHash',
    'openFrameCount','poisoned','faultCode','sourceRegistryHash','hookIdentityDigest','boundary','qualification'))


def digest(value):return hashlib.sha256(value).hexdigest()


def _hex(value):return type(value) is str and re.fullmatch('[a-f0-9]{64}', value) is not None


def _integer(value, minimum=0):return type(value) is int and minimum <= value < 2**53


def _pairs(items):
    result = {}
    for key, value in items:
        if key in result:raise ValueError('duplicate witness JSON key')
        result[key] = value
    return result


def _int(text):
    value = int(text)
    if text == '-0':raise ValueError('nonliteral witness integer')
    return value


def _float(text):
    value = float(text)
    if not math.isfinite(value) or value == 0 and math.copysign(1, value) < 0:
        raise ValueError('nonliteral witness number')
    return value


def document(blob):
    return json.loads(blob, object_pairs_hook=_pairs, parse_int=_int, parse_float=_float,
        parse_constant=lambda _: (_ for _ in ()).throw(ValueError('nonfinite witness JSON')))


def _same(left, right):
    if type(left) is not type(right):return False
    if isinstance(left, dict):return set(left) == set(right) and all(_same(v, right[k]) for k, v in left.items())
    if isinstance(left, list):return len(left) == len(right) and all(_same(a, b) for a, b in zip(left, right))
    return left == right


def control(value):
    if (type(value) is not dict or set(value) != CONTROL or value['schema'] != 'pokezero.native_witness_control.v1'
            or any(not _integer(value[k]) for k in ('stream','reset','command','request'))
            or any(type(value[k]) is not str or not value[k] for k in ('run','nonce'))
            or value['phase'] not in ('start','choices','snapshot','end','close')
            or not (value['chance_nonce'] is None or type(value['chance_nonce']) is str)
            or not _hex(value['previous_receipt_hash'])):raise ValueError('complete literal witness control required')
    return value


def _line(stream):
    value = stream.readline(MAX_LINE+1)
    if len(value) > MAX_LINE or not value.endswith(b'\n') or value.endswith(b'\r\n'):
        raise ValueError('missing or oversized exact UTF8 witness frame')
    # Decode explicitly; json.loads(bytes) also accepts UTF16/32, which Native
    # appendAudit never emits and which would invalidate this wire contract.
    return value, document(value.decode('utf8'))


@dataclass
class Cursor:
    sequence: int = 0
    chain_hash: str = ZERO
    receipt_hash: str = ZERO
    next_call: int = 0
    frames: list = field(default_factory=list)
    commands: dict = field(default_factory=dict)
    nonces: set = field(default_factory=set)


def verify_command(audit, receipt_wire, expected, *, cursor, source_registry_hash, call_operations, extra_operations=()):
    """Consume one closed command, preserving its source-bound evidence rows."""
    if (type(receipt_wire) is not bytes or len(receipt_wire) > MAX_LINE or not receipt_wire.endswith(b'\n')
            or receipt_wire.endswith(b'\r\n')):raise ValueError('exact UTF8 receipt frame required')
    if set(expected) != {'control','battle_id','hook_identity_digest'} or not _hex(source_registry_hash):
        raise ValueError('independent complete command/source/hook binding required')
    wanted = control(expected['control']);receipt = document(receipt_wire.decode('utf8'))
    if (type(receipt) is not dict or set(receipt) != RECEIPT
            or receipt['schema'] != 'pokezero.fresh200.native_witness_receipt.v1'
            or not _same(control(receipt['control']), wanted)
            or receipt['battleId'] != expected['battle_id'] or type(expected['battle_id']) is not str
            or not expected['battle_id'] or receipt['sourceRegistryHash'] != source_registry_hash
            or not _hex(expected['hook_identity_digest']) or receipt['hookIdentityDigest'] != expected['hook_identity_digest']
            or receipt['poisoned'] is not False or receipt['faultCode'] is not None
            or receipt['qualification'] is not False or type(receipt['openFrameCount']) is not int
            or receipt['openFrameCount'] != 0):raise ValueError('unclean or incorrectly bound witness receipt')
    if (not _integer(receipt['rowCount'], 2) or receipt['rowCount'] > MAX_ROWS
            or not _integer(receipt['byteCount'], 1) or receipt['byteCount'] > MAX_COMMAND_BYTES
            or not _integer(receipt['firstSequence']) or not _integer(receipt['lastSequence'])
            or receipt['firstSequence'] != cursor.sequence
            or receipt['lastSequence'] != cursor.sequence+receipt['rowCount']-1
            or not _hex(receipt['chainHash'])):raise ValueError('bounded contiguous witness receipt required')
    key = (wanted['run'], wanted['stream'], wanted['reset'])
    if (wanted['previous_receipt_hash'] != cursor.receipt_hash or wanted['nonce'] in cursor.nonces
            or wanted['command'] != cursor.commands.get(key, -1)+1 or cursor.frames):
        raise ValueError('witness command gap, replay or unclosed parent')
    roles = frozenset(call_operations);extras = frozenset(extra_operations)
    if (not roles or any(type(x) is not str or not x for x in roles | extras)
            or extras & {'COMMAND_BEGIN','COMMAND_END','CALL_ENTER','CALL_RETURN','CALL_THROW'}):
        raise ValueError('bound witness operation roster required')
    rows = [];byte_count = 0
    for index in range(receipt['rowCount']):
        wire, row = _line(audit);byte_count += len(wire)
        if byte_count > receipt['byteCount']:raise ValueError('witness command byte bound exceeded')
        if (type(row) is not dict or set(row) != ROW or row['schema'] != 'pokezero.fresh200.native_witness_row.v1'
                or not _integer(row['sequence']) or row['sequence'] != cursor.sequence
                or row['previousHash'] != cursor.chain_hash or row['battleId'] != expected['battle_id']
                or any(not _same(row[k], wanted[k]) for k in ('run','stream','reset','command','nonce','phase'))):
            raise ValueError('witness row context, sequence or exact-byte hash chain differs')
        operation = row['operation'];frame = row['frame']
        if type(operation) is not str:raise ValueError('literal witness operation required')
        if index == 0:
            if operation != 'COMMAND_BEGIN' or frame is not None or row['evidence'] is not None:
                raise ValueError('explicit witness begin missing')
        elif index == receipt['rowCount']-1:
            evidence = row['evidence']
            if (operation != 'COMMAND_END' or frame is not None or cursor.frames
                    or type(evidence) is not dict or set(evidence) != {'originalThrew','boundary'}
                    or evidence['originalThrew'] is not False or not _same(evidence['boundary'], receipt['boundary'])):
                raise ValueError('complete successful witness end missing')
        elif operation in ('CALL_ENTER','CALL_RETURN','CALL_THROW'):
            if (type(frame) is not dict or set(frame) != {'callId','parentCall','operation'}
                    or not _integer(frame['callId']) or type(frame['operation']) is not str or frame['operation'] not in roles
                    or not (frame['parentCall'] is None or _integer(frame['parentCall']))):
                raise ValueError('bound native call frame required')
            if operation == 'CALL_ENTER':
                parent = cursor.frames[-1]['callId'] if cursor.frames else None
                if frame['callId'] != cursor.next_call or not _same(frame['parentCall'], parent):
                    raise ValueError('witness call gap or parent mismatch')
                cursor.next_call += 1;cursor.frames.append(frame)
            else:
                if not cursor.frames or not _same(cursor.frames[-1], frame):raise ValueError('witness call frames unbalanced')
                if operation == 'CALL_THROW':raise ValueError('original Native exception makes reference unsuccessful')
                cursor.frames.pop()
        elif operation not in extras:raise ValueError('unbound witness material operation')
        elif ((frame is None and cursor.frames) or frame is not None and
                (not cursor.frames or not _same(frame, cursor.frames[-1]))):
            raise ValueError('witness material event call scope differs')
        rows.append(row);cursor.sequence += 1;cursor.chain_hash = digest(wire)
    if byte_count != receipt['byteCount'] or cursor.chain_hash != receipt['chainHash']:
        raise ValueError('witness command exact-byte count or terminal hash differs')
    cursor.commands[key] = wanted['command'];cursor.nonces.add(wanted['nonce']);cursor.receipt_hash = digest(receipt_wire)
    return rows


def verify_files(audit_path, receipts_path, expected_commands, *, source_registry_hash, call_operations, extra_operations=()):
    if not isinstance(expected_commands, (list, tuple)) or not expected_commands:
        raise ValueError('nonempty sealed witness command roster required')
    cursor = Cursor();commands = []
    with open(audit_path, 'rb') as audit, open(receipts_path, 'rb') as receipts:
        for expected in expected_commands:
            wire, _ = _line(receipts)
            commands.append(verify_command(audit, wire, expected, cursor=cursor, source_registry_hash=source_registry_hash,
                call_operations=call_operations, extra_operations=extra_operations))
        if audit.read(1) or receipts.read(1):raise ValueError('unaccounted witness audit or receipt bytes')
    return dict(schema='pokezero.fresh200.witness_audit_validation.v1', command_count=len(commands),
        row_count=cursor.sequence, chain_hash=cursor.chain_hash, receipt_hash=cursor.receipt_hash,
        source_registry_hash=source_registry_hash, audit_rules_passed=True, qualification=False,
        open_gates=['complete Native causal coverage','same-prefix physical agreement','literal owning references','strict actual-device parity']), commands


def verify_snapshot_wire(packet, snapshot):
    """Bind exact Node serializer bytes without re-encoding floats in Python."""
    keys = {'state_wire_base64','state_sha256','requests_wire_base64','requests_sha256'}
    if type(packet) is not dict or set(packet) != keys:raise ValueError('complete literal snapshot wire binding required')
    for name, field in (('state','battle'), ('requests','boundaryRequests')):
        try:wire = base64.b64decode(packet[name+'_wire_base64'], validate=True)
        except (TypeError, ValueError) as error:raise ValueError('invalid literal snapshot wire encoding') from error
        if len(wire) > 16*1024**2 or digest(wire) != packet[name+'_sha256']:
            raise ValueError('snapshot wire SHA or bound differs')
        if not _same(document(wire.decode('utf8')), snapshot[field]):raise ValueError('literal Native snapshot differs from its wire')
    return packet
