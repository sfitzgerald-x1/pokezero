"""Committed report prefix for compact fused runs."""

from contextlib import contextmanager
import fcntl
import json
import os
from pathlib import Path


SCHEMA = 'pokezero.v2-fused-state.v1'
REPORTS = 'reports.jsonl'
PARENT_RECEIPT = 'cooldown-parent.json'


def _directory_identity(path):
    try:
        info = path.stat()
    except FileNotFoundError:
        return None
    return info.st_dev, info.st_ino


def validate_output(out, parent):
    """Refuse a fork that writes to either physical tree of the parent."""
    if parent is None:
        if (out / PARENT_RECEIPT).exists():
            raise ValueError('existing cooldown child requires its original parent arguments')
        return
    out, parent = out.resolve(), parent.resolve()
    parent_id = _directory_identity(parent)
    if parent_id is None:
        raise ValueError('cooldown parent output does not exist')
    out_id = _directory_identity(out)
    if (parent_id in (_directory_identity(path) for path in (out, *out.parents)) or out_id is not None
            and out_id in (_directory_identity(path) for path in (parent, *parent.parents))):
        raise ValueError('cooldown child and parent need distinct, non-nested output roots')


@contextmanager
def commit_guard(out, committed_bytes, completed_rounds):
    """Only the current pointer may append and then publish the next pointer."""
    with (out / '.round-ledger.lock').open('a+b') as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        path = out / 'state.json'
        if path.exists():
            ledger = json.loads(path.read_bytes()).get('round_ledger') or {}
            current = ledger.get('committed_bytes'), ledger.get('completed_rounds')
        else:
            current = (0, 0)
        if current != (committed_bytes, completed_rounds):
            raise ValueError('another writer advanced the fused round ledger')
        yield


def append_round(out, committed_bytes, report, encoding_receipt):
    """Write one round. Any tail past the previous state pointer is uncommitted."""
    record = json.dumps(dict(report=report, encoding_receipt=encoding_receipt), default=str)
    data = record.encode() + b'\n'
    with (out / REPORTS).open('a+b') as stream:
        if stream.tell() < committed_bytes:
            raise ValueError('committed fused report prefix is missing')
        if stream.tell() > committed_bytes:
            stream.truncate(committed_bytes)
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())
    return committed_bytes + len(data)


def read_state(out, *, include_full_reports=True):
    """Read legacy state or reconstruct the committed prefix from a compact pointer."""
    out = Path(out)
    state = json.loads((out / 'state.json').read_bytes())
    if type(state) is not dict:
        raise ValueError('fused resume manifest must be a JSON object')
    if 'round_ledger' not in state:
        return state
    ledger = state['round_ledger']
    if ledger['schema'] != SCHEMA:
        raise ValueError('unknown fused round ledger schema')
    snapshots = [ledger['initializer']]
    reports = []
    receipts = {}
    end = ledger['committed_bytes']
    count = ledger['completed_rounds']
    if type(end) is not int or type(count) is not int or end <= 0 or count <= 0:
        raise ValueError('invalid committed fused report prefix')
    with (out / REPORTS).open('rb') as stream:
        for index in range(count):
            remaining = end - stream.tell()
            if remaining <= 0:
                raise ValueError('truncated committed fused report prefix')
            line = stream.readline(remaining)
            if not line.endswith(b'\n'):
                raise ValueError('truncated committed fused report')
            body = json.loads(line)
            report = body['report']
            snapshot = report['snapshot']
            if report['iteration'] != index or snapshot['iteration'] != index:
                raise ValueError('fused report prefix is out of order')
            if not include_full_reports:
                report.pop('ranks', None)
            reports.append(report)
            snapshots.append(snapshot)
            receipts[snapshot['checkpoint_sha256']] = body['encoding_receipt']
        if stream.tell() != end or snapshots[-1] != ledger['latest_snapshot']:
            raise ValueError('fused state pointer differs from the report prefix')
    return dict(state, snapshots=snapshots, reports=reports, checkpoint_encoding_receipts=receipts)
