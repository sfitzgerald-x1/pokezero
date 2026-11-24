"""Publish an ordered window of immutable V2 material on shared storage."""

import argparse
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from functools import partial
import fcntl
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Event, Thread

import numpy as np

from . import gender, team_pool
from . import v2_material_contract as contract
from .v2_pool_assembly import (
    POOL_FIELDS, assemble, assemble_span, check_shard, file_sha, generate_gender_policy,
    pool_metadata, static_metadata,
)


def read_bytes(path):
    return Path(path).read_bytes()


def sync_directory(path):
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def write_document(path, document):
    with path.open('xb') as stream:
        stream.write(contract.encode(document))
        stream.flush()
        os.fsync(stream.fileno())


@contextmanager
def publish_directory(path):
    """Expose all files of a shard or segment at one durable commit point."""
    if path.exists():
        raise FileExistsError(path)
    with TemporaryDirectory(dir=path.parent) as temporary:
        staged = Path(temporary) / 'data'
        staged.mkdir()
        yield staged
        sync_directory(staged)
        staged.rename(path)
        sync_directory(path.parent)


def shard_receipt(plan, row):
    return Path(plan['root']) / 'i' / f'{row:010d}' / 'd.json'


def record_shard_receipt(path, binding):
    with publish_directory(path.parent) as staged:
        write_document(staged / path.name, binding)


def produce_shard(plan, shard, static, generate, published):
    target = contract.shard_path(plan, shard)
    start, stop = shard['seeds']
    row = shard['row_start']
    receipt = shard_receipt(plan, row)
    local = target.parent / 'd.json'
    if 'existing' in shard:
        binding = shard['existing']
    elif receipt.exists():
        binding = json.loads(receipt.read_bytes())
    elif row in published:
        binding = published[row]
    elif target.parent.exists():
        binding = json.loads(local.read_bytes())
    else:
        show = plan['showdown']
        with publish_directory(target.parent) as staged:
            generated = staged / target.name
            result = generate(generated, teams=stop - start, seed_start=start,
                              node=show['node']['path'], showdown_root=show['root'],
                              source_commit_label=show['commit'])
            binding = {'path': str(target), 'sha256': result['sha256']}
            check_shard(dict(binding, path=generated, seeds=shard['seeds'], row_start=row), static)
            write_document(staged / local.name, binding)
        record_shard_receipt(receipt, binding)
        return dict(binding, seeds=shard['seeds'], row_start=row)
    if (contract.artifact(binding)['path'] != str(target)
            or row in published and binding != published[row]):
        raise ValueError('published shard disagrees with its plan or admitted binding')
    result = dict(binding, seeds=shard['seeds'], row_start=row)
    check_shard(result, static)
    if 'existing' not in shard:
        if json.loads(local.read_bytes()) != binding:
            raise ValueError('shard disagrees with its own publication receipt')
        if not receipt.exists():
            record_shard_receipt(receipt, binding)
    return result


def verify_preserved_plan(plan, static):
    """Authorize the original prefix and prohibit reuse anywhere in its source pool."""
    original = plan['parent_pool']
    if file_sha(original['path']) != original['sha256']:
        raise ValueError('parent pool checksum differs')
    preserved = plan['preserved_team_rows']
    prefix = [shard for shard in plan['shards'] if shard['row_start'] < preserved]
    for shard in prefix:
        check_shard(dict(shard, **shard['existing']), static)
    with np.load(original['path'], allow_pickle=False) as pool:
        if set(pool.files) != {*POOL_FIELDS, 'metadata_utf8'}:
            raise ValueError('parent constructor pool has an incompatible array inventory')
        seeds = np.sort(pool['seeds'])
        if len(seeds) < preserved:
            raise ValueError('parent pool does not cover the preserved roster')
        for shard in plan['shards'][len(prefix):]:
            start, stop = shard['seeds']
            lo, hi = np.searchsorted(seeds, [start, stop])
            if lo != hi:
                raise ValueError('fresh constructor seed reuses the parent roster')
        for name, (shape, dtype) in POOL_FIELDS.items():
            values = pool[name]
            if values.shape != (len(seeds), *shape) or values.dtype != dtype or not values.flags.c_contiguous:
                raise ValueError(f'parent constructor pool array layout differs for {name}')
            for shard in prefix:
                lo, hi = shard['row_start'], contract.shard_end(shard)
                with np.load(shard['existing']['path'], allow_pickle=False) as archive:
                    if not np.array_equal(values[lo:hi], archive[name]):
                        raise ValueError(f'preserved constructor bytes differ for {name} at row {lo}')


def verify_publication(plan, plan_sha, index, ref, document, static):
    """Reopen the published shards and bind the structural check to their manifest."""
    for expected, shard in zip(contract.window_shards(plan, index), document['shards'], strict=True):
        if 'existing' not in expected:
            receipt = shard_receipt(plan, shard['row_start'])
            local = Path(shard['artifact']['path']).parent / 'd.json'
            if any(json.loads(path.read_bytes()) != shard['artifact'] for path in (receipt, local)):
                raise ValueError('shard disagrees with its own publication receipt')
    path = contract.verification_path(plan, index).parent
    if path.exists():
        contract.require_verification(plan, plan_sha, index, ref, read_bytes)
        verify_files(document)
        return
    for shard in document['shards']:
        check_shard(dict(shard, **shard['artifact']), static)
    for key in ('pool', 'gender_policy'):
        artifact = document[key]
        if file_sha(artifact['path']) != artifact['sha256']:
            raise ValueError(f"material checksum differs for {artifact['path']}")
    original = contract.gender_semantics(contract.read(plan['gender_policy'], read_bytes),
                                         plan['parent_pool']['sha256'])
    current = contract.gender_semantics(contract.read(document['gender_policy'], read_bytes),
                                        document['pool']['sha256'])
    if original != current:
        raise ValueError('the material changes the original gender assignment policy')
    gender.validate_policy(current, static['vocabulary']['species'])
    with np.load(document['pool']['path'], allow_pickle=False) as pool:
        if set(pool.files) != {*POOL_FIELDS, 'metadata_utf8'}:
            raise ValueError('unexpected field in segment constructor pool')
        metadata = json.loads(pool['metadata_utf8'].tobytes())
        if static_metadata(metadata) != static or metadata['completed_teams'] != document['team_rows']:
            raise ValueError('segment pool metadata differs from its verified material')
        for name, (shape, dtype) in POOL_FIELDS.items():
            values = pool[name]
            if values.shape != (document['team_rows'], *shape) or values.dtype != dtype:
                raise ValueError(f'segment pool array layout differs for {name}')
            for shard in document['shards']:
                row = shard['row_start'] - document['row_offset']
                with np.load(shard['artifact']['path'], allow_pickle=False) as archive:
                    if not np.array_equal(values[row:row + shard['rows']], archive[name]):
                        raise ValueError(f'segment pool differs from its verified shards for {name}')
    with publish_directory(path) as staged:
        write_document(staged / 'd.json', contract.verification(plan_sha, index, ref))

def verify_files(document):
    for ref in contract.material_artifacts(document):
        if file_sha(ref['path']) != ref['sha256']:
            raise ValueError(f"material checksum differs for {ref['path']}")


def generate_segment(plan, plan_sha, index, previous, shards, metadata, policy_writer):
    path = contract.segment_directory(plan, index)
    offset = shards[0]['row_start']
    rows = contract.shard_end(shards[-1]) - offset
    with publish_directory(path) as staged:
        pool = assemble_span(shards, offset, rows, metadata, staged / 'b.npz')
        policy = policy_writer(pool, plan['showdown'], staged / 'c.json')
        document = dict(
            plan_sha256=plan_sha, segment=index, previous_sha256=previous, window=plan['windows'][index],
            row_offset=offset, team_rows=rows, shards=[
                dict(row_start=shard['row_start'], seeds=shard['seeds'],
                     rows=contract.shard_end(shard) - shard['row_start'],
                     artifact={key: shard[key] for key in ('path', 'sha256')})
                for shard in shards
            ], pool={'path': str(path / 'b.npz'), 'sha256': pool['sha256']},
            gender_policy={'path': str(path / 'c.json'), 'sha256': policy['sha256']},
        )
        contract.verify_manifest(plan, plan_sha, index, previous, document)
        write_document(staged / 'm.json', document)



def existing_manifests(plan, plan_sha):
    """Check the surviving chain before any recovery can write material."""
    manifests = []
    previous = None
    for index in range(len(plan['windows'])):
        if not contract.manifest_path(plan, index).exists():
            if contract.segment_directory(plan, index).exists() or contract.verification_path(plan, index).exists():
                raise ValueError('a published manifest is missing; restore the original bytes')
            continue
        if index != len(manifests):
            raise ValueError('a published manifest is missing; restore the original bytes')
        ref, document = contract.bound_manifest(plan, plan_sha, index, previous, read_bytes)
        if contract.verification_path(plan, index).exists():
            contract.require_verification(plan, plan_sha, index, ref, read_bytes)
        manifests.append((ref, document))
        previous = ref['sha256']
    final = Path(plan['root']) / 'z'
    closed = final.exists()
    if closed:
        if len(manifests) != len(plan['windows']):
            raise ValueError('a published manifest is missing; restore the original bytes')
        closing = json.loads((final / 'd.json').read_bytes())
        if (closing['plan_sha256'] != plan_sha or closing['ledger_sha256'] != previous
                or closing['team_rows'] != contract.shard_end(plan['shards'][-1])
                or file_sha(final / 't.npz') != closing['pool_sha256']):
            raise ValueError('final material receipt differs from its plan, ledger or pool')
    return manifests, closed

def publish(plan_ref, start, stop, *, workers=1, generate=team_pool.generate,
            policy_writer=generate_gender_policy):
    plan = contract.load_plan(plan_ref, read_bytes)
    plan_sha = plan_ref['sha256']
    root = Path(plan['root'])
    if not 0 <= start < stop <= len(plan['windows']) or workers < 1:
        raise ValueError('invalid material window or worker count')
    root.mkdir(parents=True, exist_ok=True)
    # The root is the writer namespace. Hold its lock across every publication,
    # including validation of a reused shard and completion of the digest.
    with (root / 'a.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        manifests, closed = existing_manifests(plan, plan_sha)
        if start > len(manifests):
            raise ValueError('a prior segment must publish before generating this window')
        marker = root / 'p'
        metadata = pool_metadata(plan['parent_pool']['path'])
        static = static_metadata(metadata)
        if not marker.exists():
            verify_preserved_plan(plan, static)
            with publish_directory(marker) as staged:
                write_document(staged / 'd.json', plan_ref)
        if json.loads((marker / 'd.json').read_bytes())['sha256'] != plan_sha:
            raise ValueError('shared material root already belongs to another plan')

        (root / 'i').mkdir(exist_ok=True)
        published = {}
        previous = None
        for index, (ref, document) in enumerate(manifests[:start]):
            contract.require_verification(plan, plan_sha, index, ref, read_bytes)
            remember_shards(published, document)
            previous = ref['sha256']

        if file_sha(plan['showdown']['node']['path']) != plan['showdown']['node']['sha256']:
            raise ValueError('Node binary differs from the bound generator')
        make_shard = partial(produce_shard, plan, static=static, generate=generate, published=published)
        with ThreadPoolExecutor(max_workers=workers) as pool:
            for index in range(start, stop):
                if index < len(manifests):
                    ref, document = manifests[index]
                else:
                    shards = list(pool.map(make_shard, contract.window_shards(plan, index)))
                    generate_segment(plan, plan_sha, index, previous, shards, metadata, policy_writer)
                    ref, document = contract.bound_manifest(plan, plan_sha, index, previous, read_bytes)
                remember_shards(published, document)
                verify_publication(plan, plan_sha, index, ref, document, static)
                previous = ref['sha256']
            if stop == len(plan['windows']) and not closed:
                finalize(plan, plan_sha, previous, published, pool, make_shard)
    return previous


def remember_shards(published, document):
    for entry in document['shards']:
        row = entry['row_start']
        artifact = entry['artifact']
        if published.setdefault(row, artifact) != artifact:
            raise ValueError('segment manifests disagree about a shared shard')


def finalize(plan, plan_sha, previous, admitted, pool, make_shard):
    target = Path(plan['root']) / 'z'
    pending = [shard for shard in plan['shards'] if shard['row_start'] not in admitted]
    all_refs = dict(admitted)
    all_refs.update((shard['row_start'], shard) for shard in pool.map(make_shard, pending))
    shards = [dict(path=all_refs[shard['row_start']]['path'], sha256=all_refs[shard['row_start']]['sha256'],
                   seeds=shard['seeds'], row_start=shard['row_start']) for shard in plan['shards']]
    manifest = dict(parent_pool=plan['parent_pool'], preserved_team_rows=plan['preserved_team_rows'],
                    required_team_rows=contract.shard_end(shards[-1]), shards=shards)
    with publish_directory(target) as staged:
        result = assemble(manifest, staged / 't.npz')
        write_document(staged / 'd.json', dict(plan_sha256=plan_sha, ledger_sha256=previous,
                                              pool_sha256=result['sha256'], team_rows=result['team_rows']))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', required=True)
    parser.add_argument('--sha256', required=True)
    parser.add_argument('--start', required=True, type=int)
    parser.add_argument('--stop', required=True, type=int)
    parser.add_argument('--workers', default=1, type=int)
    args = parser.parse_args()
    peak = Path('/sys/fs/cgroup/memory.peak')
    current = Path('/sys/fs/cgroup/memory.current')
    finished = Event()
    samples = []

    def sample_memory():
        while not finished.is_set():
            samples.append(int(current.read_text()))
            finished.wait(1)

    sampler = None
    if not peak.exists() and current.exists():
        sampler = Thread(target=sample_memory, daemon=True)
        sampler.start()
    try:
        ledger_sha = publish({'path': args.plan, 'sha256': args.sha256}, args.start, args.stop,
                             workers=args.workers)
    finally:
        finished.set()
        if sampler is not None:
            sampler.join()
    peak_bytes = int(peak.read_text()) if peak.exists() else max(samples, default=None)
    print(json.dumps(dict(ledger_sha256=ledger_sha, memory_peak_bytes=peak_bytes), sort_keys=True), flush=True)


if __name__ == '__main__':
    main()
