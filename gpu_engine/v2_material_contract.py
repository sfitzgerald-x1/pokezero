"""Immutable address and material bindings for segmented V2 campaigns."""

import hashlib
import json
from pathlib import Path
import re


def encode(document):
    return json.dumps(document, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def digest(data):
    return hashlib.sha256(data).hexdigest()


def reference(path, data):
    return {'path': str(path), 'sha256': digest(data)}


def read(ref, read_bytes):
    data = read_bytes(ref['path'])
    if digest(data) != ref['sha256']:
        raise ValueError(f"material checksum differs for {ref['path']}")
    return json.loads(data)


def artifact(ref):
    if (set(ref) != {'path', 'sha256'} or not Path(ref['path']).is_absolute()
            or not re.fullmatch('[0-9a-f]{64}', ref['sha256'])):
        raise ValueError('expected an absolute, digest-bound material artifact')
    return ref


def shard_end(shard):
    start, stop = shard['seeds']
    return shard['row_start'] + stop - start


def load_plan(ref, read_bytes):
    plan = read(artifact(ref), read_bytes)
    validate_plan(plan)
    if plan['protected_seeds'] != protected_ranges(plan['protected_seed_inventories'], read_bytes):
        raise ValueError('protected ranges differ from their pinned inventories')
    return plan



def protected_ranges(inventories, read_bytes):
    if not inventories:
        raise ValueError('pinned protected seed inventories are required')
    ranges = set()
    for ref in inventories:
        document = read(artifact(ref), read_bytes)
        if document['complete'] is not True or not document['seed_ranges']:
            raise ValueError('complete, nonempty protected seed inventory required')
        for row in document['seed_ranges']:
            start, stop = row['start'], row['stop']
            if type(start) is not int or type(stop) is not int or not 0 <= start < stop <= 2**32:
                raise ValueError('invalid protected evaluation seed range')
            ranges.add((start, stop))
    return [list(pair) for pair in sorted(ranges)]

def validate_plan(plan):
    if plan['version'] != 1 or not Path(plan['root']).is_absolute():
        raise ValueError('unsupported material plan')
    artifact(plan['parent_pool'])
    artifact(plan['gender_policy'])
    artifact(plan['showdown']['node'])
    row = 0
    paths = set()
    for shard in plan['shards']:
        start, stop = shard['seeds']
        if shard['row_start'] != row or not 0 <= start < stop <= 2**32:
            raise ValueError(f'invalid or missing planned shard at row {row}')
        if 'existing' in shard:
            ref = artifact(shard['existing'])
            if ref['path'] in paths:
                raise ValueError('an existing shard is assigned to two address ranges')
            paths.add(ref['path'])
        row = shard_end(shard)
    seed_ranges = sorted(shard['seeds'] for shard in plan['shards'])
    if any(left[1] > right[0] for left, right in zip(seed_ranges, seed_ranges[1:])):
        raise ValueError('planned constructor seeds overlap')
    if (not 0 < plan['preserved_team_rows'] <= row or not plan['windows']
            or plan['preserved_team_rows'] != plan['windows'][0]['rows'][0]):
        raise ValueError('preserved roster must extend to the first new training row')
    protected = plan['protected_seeds']
    if not protected or any(not 0 <= start < stop <= 2**32 for start, stop in protected):
        raise ValueError('invalid protected evaluation seed range')
    fresh = sorted(shard['seeds'] for shard in plan['shards']
                   if shard['row_start'] >= plan['preserved_team_rows'])
    cursor = 0
    for start, stop in fresh:
        while cursor < len(protected) and protected[cursor][1] <= start:
            cursor += 1
        if cursor < len(protected) and protected[cursor][0] < stop:
            raise ValueError('planned constructor seed overlaps protected evaluation')
    prefix = [shard for shard in plan['shards'] if shard['row_start'] < plan['preserved_team_rows']]
    if shard_end(prefix[-1]) != plan['preserved_team_rows'] or any('existing' not in shard for shard in prefix):
        raise ValueError('the original roster must be covered by whole, digest-bound shards')
    spans = []
    for window in plan['windows']:
        start, stop = window['rounds']
        lo, hi = window['rows']
        if not (0 <= start < stop and plan['preserved_team_rows'] <= lo < hi <= row):
            raise ValueError('invalid planned round or row window')
        spans.append((lo, hi))
    spans.sort()
    if any(left[1] > right[0] for left, right in zip(spans, spans[1:])):
        raise ValueError('planned team addresses overlap across segments')


def round_rows(iterations, start, stop):
    if not 0 <= start < stop <= len(iterations):
        raise ValueError('material window is outside the declaration')
    spans = []
    for batches in iterations[start:stop]:
        for batch in batches:
            games = batch['games']
            lo = games['team_start']
            spans.append((lo, lo + 2 * games['game_count']))
    if not spans or any(lo >= hi for lo, hi in spans) or any(
        left[1] != right[0] for left, right in zip(spans, spans[1:])
    ):
        raise ValueError('expected a contiguous training roster for the segment')
    return [spans[0][0], spans[-1][1]]


def window_shards(plan, index):
    lo, hi = plan['windows'][index]['rows']
    return [shard for shard in plan['shards'] if shard['row_start'] < hi and shard_end(shard) > lo]


def shard_path(plan, shard):
    if 'existing' in shard:
        return Path(shard['existing']['path'])
    return Path(plan['root']) / f"{shard['row_start']:010d}" / 'a.npz'


def segment_directory(plan, index):
    return Path(plan['root']) / f'{index:05d}'


def manifest_path(plan, index):
    return segment_directory(plan, index) / 'm.json'


def verify_manifest(plan, plan_sha, index, previous, document):
    """Check that an immutable manifest covers exactly the planned window."""
    window = plan['windows'][index]
    shards = window_shards(plan, index)
    entries = document['shards']
    if (document['plan_sha256'] != plan_sha or document['segment'] != index
            or document['previous_sha256'] != previous or document['window'] != window
            or len(entries) != len(shards)):
        raise ValueError('segment manifest does not match its address plan and ledger position')
    for shard, entry in zip(shards, entries, strict=True):
        ref = artifact(entry['artifact'])
        if (entry['row_start'] != shard['row_start'] or entry['seeds'] != shard['seeds']
                or entry['rows'] != shard_end(shard) - shard['row_start']
                or ref['path'] != str(shard_path(plan, shard))
                or ('existing' in shard and ref != shard['existing'])):
            raise ValueError('segment manifest binds an unplanned shard')
    base = segment_directory(plan, index)
    for key, name in (('pool', 'b.npz'), ('gender_policy', 'c.json')):
        if artifact(document[key])['path'] != str(base / name):
            raise ValueError(f'segment manifest binds an unplanned {key} artifact')
    offset = shards[0]['row_start']
    if document['row_offset'] != offset or document['team_rows'] != shard_end(shards[-1]) - offset:
        raise ValueError('segment manifest binds an incorrect material span')
    return document


def bound_manifest(plan, plan_sha, index, previous, read_bytes):
    path = manifest_path(plan, index)
    data = read_bytes(str(path))
    document = verify_manifest(plan, plan_sha, index, previous, json.loads(data))
    return reference(path, data), document


def material_artifacts(document):
    return [*(entry['artifact'] for entry in document['shards']), document['pool'], document['gender_policy']]


def gender_semantics(document, pool_sha):
    provenance = dict(document['provenance'])
    if provenance.pop('pool_sha256') != pool_sha:
        raise ValueError('gender policy is bound to a different constructor material')
    return dict(document, provenance=provenance)



def verification_path(plan, index):
    return Path(plan['root']) / f'v{index:05d}' / 'd.json'


def verification(plan_sha, index, ref):
    return dict(plan_sha256=plan_sha, segment=index, manifest_sha256=ref['sha256'])


def require_verification(plan, plan_sha, index, ref, read_bytes):
    document = json.loads(read_bytes(str(verification_path(plan, index))))
    if document != verification(plan_sha, index, ref):
        raise ValueError('material verification differs from the candidate segment manifest')

def load_segment(arguments, stop_round, admitted_round):
    plan_path, plan_sha, index, path, manifest_sha = arguments
    read_bytes = lambda name: Path(name).read_bytes()
    plan = load_plan({'path': plan_path, 'sha256': plan_sha}, read_bytes)
    index = int(index)
    if not 0 <= index < len(plan['windows']) or str(manifest_path(plan, index)) != path:
        raise ValueError('unplanned material manifest path')
    previous = None if index == 0 else digest(manifest_path(plan, index - 1).read_bytes())
    ref = artifact({'path': path, 'sha256': manifest_sha})
    document = read(ref, read_bytes)
    require_verification(plan, plan_sha, index, ref, read_bytes)
    verify_manifest(plan, plan_sha, index, previous, document)
    start, stop = document['window']['rounds']
    if stop != stop_round or not start <= admitted_round <= stop:
        raise ValueError('material manifest does not match the admitted segment boundaries')
    return plan_sha, plan, document
