"""Assemble a digest-bound constructor roster from ordered shared-storage shards."""

import argparse
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


def assemble(manifest, output):
    output = Path(output)
    shards = manifest['shards']
    rows = manifest['required_team_rows']
    parent = manifest['parent_pool']
    verify_coverage(shards, rows)
    if output.exists():
        raise FileExistsError(output)
    if file_sha(parent['path']) != parent['sha256']:
        raise ValueError('parent pool checksum differs')

    with np.load(parent['path'], allow_pickle=False) as original:
        parent_metadata = json.loads(original['metadata_utf8'].tobytes())
    static = {key: value for key, value in parent_metadata.items() if key not in POPULATION_FIELDS}
    shapes = dict(FIELDS, seeds=((), np.uint32))
    counts = Counter()
    with TemporaryDirectory(dir=output.parent) as scratch:
        arrays = {name: np.lib.format.open_memmap(Path(scratch) / f'{name}.npy', mode='w+',
                  dtype=dtype, shape=(rows, *shape)) for name, (shape, dtype) in shapes.items()}
        for shard in shards:
            start, stop = shard['seeds']
            count = stop - start
            row = shard['row_start']
            if file_sha(shard['path']) != shard['sha256']:
                raise ValueError(f'constructor shard checksum differs at row {row}')
            with np.load(shard['path'], allow_pickle=False) as source:
                metadata = json.loads(source['metadata_utf8'].tobytes())
                if {key: value for key, value in metadata.items() if key not in POPULATION_FIELDS} != static:
                    raise ValueError(f'constructor shard static metadata differs at row {row}')
                if (metadata['seed_start'] != start or metadata['completed_teams'] != count
                        or not np.array_equal(source['seeds'], np.arange(start, stop, dtype=np.uint32))):
                    raise ValueError(f'constructor shard seed ledger differs at row {row}')
                for name, destination in arrays.items():
                    values = source[name]
                    if values.shape != (count, *destination.shape[1:]) or values.dtype != destination.dtype:
                        raise ValueError(f'constructor shard array layout differs for {name} at row {row}')
                    destination[row:row+count] = values
                counts.update({move['id']: move['count'] for move in metadata['move_frequency']})

        metadata = dict(static, seed_start=parent_metadata['seed_start'],
                        team_count=rows, completed_teams=rows, **summarize_moves(static, counts, rows))
        arrays['metadata_utf8'] = np.frombuffer(json.dumps(metadata, sort_keys=True).encode(), dtype=np.uint8)
        temporary = Path(scratch) / 'pool.npz'
        np.savez_compressed(temporary, **arrays)
        digest = file_sha(temporary)
        verify_constructor_pool_prefix(
            parent, dict(path=temporary, sha256=digest),
            manifest['preserved_team_rows'], successor_teams=rows,
        )
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
    program = Path(__file__).resolve().parents[1] / 'gpu_engine' / 'gender_policy.cjs'
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--gender-policy-out', type=Path)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_bytes())
    result = assemble(manifest, args.out)
    if args.gender_policy_out is not None:
        result['gender_policy'] = generate_gender_policy(result, manifest['showdown'], args.gender_policy_out)
    peak = Path('/sys/fs/cgroup/memory.peak')
    if peak.exists():
        result['memory_peak_bytes'] = int(peak.read_text())
    print(json.dumps(result, sort_keys=True))


if __name__ == '__main__':
    main()
