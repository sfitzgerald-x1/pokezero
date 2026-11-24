"""Assemble a digest-bound constructor roster from ordered shared-storage shards."""

from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import subprocess
from tempfile import TemporaryDirectory

import numpy as np

from gpu_engine.source_provenance import verify_constructor_pool_prefix
from gpu_engine.team_pool import FIELDS


POPULATION_FIELDS = {
    'seed_start', 'team_count', 'completed_teams', 'move_slots', 'move_frequency',
    'target85_move_count', 'tensor_move_slots',
}


def file_sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def verify_coverage(shards, required_rows):
    row = 0
    for shard in shards:
        start, stop = shard['seeds']
        if shard['row_start'] != row or not 0 <= start < stop <= 2**32:
            raise ValueError(f'invalid or missing shard at roster row {row}')
        row += stop - start
    if row != required_rows:
        raise ValueError(f'roster ends at row {row}, expected {required_rows}')

    last_stop = 0
    for start, stop in sorted(shard['seeds'] for shard in shards):
        if start < last_stop:
            raise ValueError(f'constructor team seeds overlap at {start}')
        last_stop = stop


def summarize_moves(metadata, counts, rows):
    slots = sum(counts.values())
    if slots <= 0:
        raise ValueError('constructor roster has no recorded moves')
    positions = {name: index for index, name in enumerate(metadata['vocabulary']['moves'])}
    frequency = []
    cumulative = 0
    for move, count in sorted(counts.items(), key=lambda item: (-item[1], positions[item[0]])):
        cumulative += count
        frequency.append(dict(id=move, count=count, share=count / slots,
                              cumulative_share=cumulative / slots, implemented=False))
    return dict(move_slots=slots, move_frequency=frequency, tensor_move_slots=rows * 24,
                target85_move_count=next(index + 1 for index, move in enumerate(frequency)
                                         if move['cumulative_share'] >= 0.85))


def pool_metadata(path):
    with np.load(path, allow_pickle=False) as archive:
        return json.loads(archive['metadata_utf8'].tobytes())


def static_metadata(metadata):
    return {key: value for key, value in metadata.items() if key not in POPULATION_FIELDS}


def check_shard(shard, static, destinations=None, *, offset=0):
    start, stop = shard['seeds']
    count = stop - start
    row = shard['row_start']
    if file_sha(shard['path']) != shard['sha256']:
        raise ValueError(f'constructor shard checksum differs at row {row}')
    with np.load(shard['path'], allow_pickle=False) as source:
        metadata = json.loads(source['metadata_utf8'].tobytes())
        if static_metadata(metadata) != static:
            raise ValueError(f'constructor shard static metadata differs at row {row}')
        if (metadata['seed_start'] != start or metadata['completed_teams'] != count
                or not np.array_equal(source['seeds'], np.arange(start, stop, dtype=np.uint32))):
            raise ValueError(f'constructor shard seed ledger differs at row {row}')
        for name, (shape, dtype) in POOL_FIELDS.items():
            values = source[name]
            if values.shape != (count, *shape) or values.dtype != dtype:
                raise ValueError(f'constructor shard array layout differs for {name} at row {row}')
            if destinations is not None:
                local = row - offset
                destinations[name][local:local + count] = values
    return {move['id']: move['count'] for move in metadata['move_frequency']}


POOL_FIELDS = dict(FIELDS, seeds=((), np.uint32))


def assemble(manifest, output):
    output = Path(output)
    shards = manifest['shards']
    rows = manifest['required_team_rows']
    parent = manifest['parent_pool']
    verify_coverage(shards, rows)
    if file_sha(parent['path']) != parent['sha256']:
        raise ValueError('parent pool checksum differs')
    return assemble_span(shards, 0, rows, pool_metadata(parent['path']), output,
                         prefix=(parent, manifest['preserved_team_rows']))


def assemble_span(shards, offset, rows, metadata, output, *, prefix=None):
    """Write a contiguous span with the same field and metadata encoding as the full pool."""
    if output.exists():
        raise FileExistsError(output)
    static = static_metadata(metadata)
    counts = Counter()
    with TemporaryDirectory(dir=output.parent) as scratch:
        arrays = {
            name: np.lib.format.open_memmap(Path(scratch) / f'{name}.npy', mode='w+',
                                            dtype=dtype, shape=(rows, *shape))
            for name, (shape, dtype) in POOL_FIELDS.items()
        }
        for shard in shards:
            counts.update(check_shard(shard, static, arrays, offset=offset))
        merged = dict(static, seed_start=metadata['seed_start'], team_count=rows,
                      completed_teams=rows, **summarize_moves(static, counts, rows))
        arrays['metadata_utf8'] = np.frombuffer(json.dumps(merged, sort_keys=True).encode(), dtype=np.uint8)
        temporary = Path(scratch) / 'pool.npz'
        np.savez_compressed(temporary, **arrays)
        digest = file_sha(temporary)
        if prefix is not None:
            parent, preserved = prefix
            verify_constructor_pool_prefix(parent, dict(path=temporary, sha256=digest),
                                           preserved, successor_teams=rows)
        with temporary.open('rb') as stream:
            os.fsync(stream.fileno())
        os.link(temporary, output)
    return dict(path=str(output), sha256=digest, team_rows=rows, bytes=output.stat().st_size)


def generate_gender_policy(pool_result, showdown, output):
    output = Path(output)
    node = showdown['node']
    if file_sha(node['path']) != node['sha256']:
        raise ValueError('Showdown Node binary differs from the bound runtime')
    with np.load(pool_result['path'], allow_pickle=False) as archive:
        metadata = json.loads(archive['metadata_utf8'].tobytes())
    if metadata['completed_teams'] != pool_result['team_rows']:
        raise ValueError('constructor metadata differs from the assembled pool receipt')
    if metadata['compiled_tree']['sha256'] != showdown['compiled_tree_sha256']:
        raise ValueError('constructor metadata differs from the pinned Showdown tree')
    metadata['pool_sha256'] = pool_result['sha256']
    program = Path(__file__).with_name('gender_policy.cjs')
    with TemporaryDirectory(dir=output.parent) as directory:
        spec = Path(directory) / 'pool.json'
        spec.write_text(json.dumps(metadata, sort_keys=True))
        temporary = Path(directory) / 'policy.json'
        with temporary.open('xb') as stream:
            subprocess.run([node['path'], str(program), showdown['root'], str(spec), showdown['commit']],
                           stdout=stream, check=True, timeout=120)
        policy = json.loads(temporary.read_bytes())
        if policy['provenance']['pool_sha256'] != pool_result['sha256']:
            raise ValueError('generated gender policy is bound to a different constructor pool')
        policy_sha256 = file_sha(temporary)
        os.link(temporary, output)
    return dict(path=str(output), sha256=policy_sha256)
