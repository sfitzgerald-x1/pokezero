"""Committed report prefix for compact fused runs."""

from bisect import bisect_left
from contextlib import contextmanager
import fcntl
import hashlib
from itertools import accumulate
import json
import os
from pathlib import Path
import re


SCHEMA = 'pokezero.v2-fused-state.v1'
REPORTS = 'reports.jsonl'
PARENT_RECEIPT = 'cooldown-parent.json'


def verify_campaign_prefix(original, successor, saved):
    """Verify the committed S0a declaration prefix before admitting its successor."""
    from .v5_run import _json

    rounds = 100
    ledger = saved.get('round_ledger', {})
    if ledger.get('completed_rounds') != rounds or ledger.get('schema') != SCHEMA:
        raise ValueError('campaign adoption requires exactly 100 committed compact rounds')
    if hashlib.sha256(_json(original).encode()).hexdigest() != saved.get('declaration_sha256'):
        raise ValueError('original campaign declaration differs from the committed state')
    if len(original['iterations']) != 250 or len(successor['iterations']) != 5000:
        raise ValueError('campaign adoption requires the original 5M and successor 100M declarations')
    if _json(original['iterations'][:rounds]) != _json(successor['iterations'][:rounds]):
        raise ValueError('successor changes a committed game, seed or training roster address')
    # The successor may extend the physical pool. Its used prefix bytes are
    # checked separately from the roster addresses in these documents.
    def immutable_fields(document):
        return {key: value for key, value in document.items() if key not in ('iterations', 'pool_sha256')}

    if _json(immutable_fields(original)) != _json(immutable_fields(successor)):
        raise ValueError('successor changes the campaign identity or learner policy')


def campaign_batch_addresses(batch):
    """Return the ranges in the compact declaration's six address spaces."""
    games = batch['games']
    count = games['game_count']
    return (
        (games['stream_start'], count),
        (games['identity_seed_start'], count),
        (games['team_start'], 2 * count),
        (batch['constructor_seed'], 1),
        (batch['sampling_seed'], 1),
        (batch['gender_root_start'], count),
    )


def verify_disjoint_campaign_addresses(parent_rounds, child):
    """Refuse reuse of any committed or reserved parent address."""
    parent = (campaign_batch_addresses(batch) for batches in parent_rounds for batch in batches)
    continuation = (campaign_batch_addresses(batch) for batches in child for batch in batches)
    for occupied, requested in zip(zip(*parent, strict=True), zip(*continuation, strict=True), strict=True):
        intervals = sorted(occupied)
        starts = [start for start, _ in intervals]
        frontier = list(accumulate((start + size for start, size in intervals), max))
        for start, size in requested:
            index = bisect_left(starts, start + size)
            if index and frontier[index - 1] > start:
                raise ValueError('cooldown child reuses a committed or reserved parent roster or RNG address')


def _directory_identity(path):
    try:
        info = path.stat()
    except FileNotFoundError:
        return None
    return info.st_dev, info.st_ino


def _mount_paths():
    """Map Linux mount points to superblock-relative roots, including bind mounts."""
    try:
        lines = Path('/proc/self/mountinfo').read_text().splitlines()
    except FileNotFoundError:  # local development outside Linux
        return {}

    def decode(value):
        return Path(re.sub(r'\\([0-7]{3})', lambda match: chr(int(match[1], 8)), value))

    mounts = {}
    for line in lines:
        _, _, device, root, point, *_ = line.split()
        mounts[decode(point)] = device, decode(root)
    return mounts


def _backing_path(path, mounts):
    for candidate in (path, *path.parents):
        if candidate in mounts:
            device, root = mounts[candidate]
            return device, root / path.relative_to(candidate)
    return None


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
    same_tree = (parent_id in (_directory_identity(path) for path in (out, *out.parents)) or out_id is not None
                 and out_id in (_directory_identity(path) for path in (parent, *parent.parents)))
    mounts = _mount_paths()
    backing_out, backing_parent = _backing_path(out, mounts), _backing_path(parent, mounts)
    shared_backing_tree = bool(backing_out and backing_parent and backing_out[0] == backing_parent[0] and (
        backing_out[1].is_relative_to(backing_parent[1]) or backing_parent[1].is_relative_to(backing_out[1])))
    if same_tree or shared_backing_tree:
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
