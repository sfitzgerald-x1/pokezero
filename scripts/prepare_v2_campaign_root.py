"""Stage the immutable 2M parent into a fresh campaign output on shared storage."""

import argparse
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory

from pokezero.v2_fused_state import REPORTS, read_state, validate_output


def source_file(root, relative):
    relative = Path(relative)
    if relative.is_absolute() or '..' in relative.parts:
        raise ValueError(f'expected a relative parent artifact path: {relative}')
    path = root / relative
    if path.resolve() != path or not path.is_file():
        raise ValueError(f'expected an owned parent artifact: {relative}')
    return path


def checked_bytes(path, expected):
    data = path.read_bytes()
    if hashlib.sha256(data).hexdigest() != expected:
        raise ValueError(f'parent digest differs for {path.name}')
    return data


def copy_file(source, target, expected=None, *, prefix_bytes=None):
    target.parent.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256()
    remaining = prefix_bytes
    with source.open('rb') as reader, target.open('xb') as writer:
        while remaining is None or remaining > 0:
            size = 1 << 20 if remaining is None else min(remaining, 1 << 20)
            block = reader.read(size)
            if not block:
                break
            writer.write(block)
            digest.update(block)
            if remaining is not None:
                remaining -= len(block)
    if (remaining is not None and remaining != 0) or (expected and digest.hexdigest() != expected):
        raise ValueError(f'parent artifact differs for {source.name}')
    return digest.hexdigest()


def copy_retained_checkpoint(source, staging, snapshot):
    checkpoint = snapshot['checkpoint']
    proof = checkpoint + '.encoding.json'
    copy_file(source_file(source, checkpoint), staging / checkpoint, snapshot['checkpoint_sha256'])
    if (source / proof).exists():
        copy_file(source_file(source, proof), staging / proof)


def prepare(source, output, binding):
    source = Path(source).resolve()
    output = Path(output)
    validate_output(output, source)
    if output.exists() or output.is_symlink():
        raise FileExistsError(output)

    pointer = source_file(source, 'state.json')
    pointer_data = checked_bytes(pointer, binding['state_sha256'])
    saved = read_state(source, include_full_reports=False)
    ledger = saved.get('round_ledger', {})
    if (ledger.get('completed_rounds') != 100 or saved.get('declaration_sha256') != binding['declaration_sha256']
            or saved.get('adopted_prefix') is not None or saved.get('cooldown_parent') is not None):
        raise ValueError('fresh-root staging requires the bound uncooled 2M parent')
    latest = ledger['latest_snapshot']
    if latest['checkpoint_sha256'] != binding['checkpoint_sha256']:
        raise ValueError('parent checkpoint differs from the selected Arm A snapshot')
    if set(binding['milestone_sha256s']) != {'1000000', '2000000'}:
        raise ValueError('both original milestone dispatch digests are required')

    with TemporaryDirectory(dir=output.parent) as directory:
        staging = Path(directory) / 'root'
        staging.mkdir()
        report_digest = copy_file(source_file(source, REPORTS), staging / REPORTS,
                                  prefix_bytes=ledger['committed_bytes'])
        for relative in ('declaration.json', 'route-stop-authority.json'):
            expected = binding['declaration_sha256'] if relative == 'declaration.json' else None
            copy_file(source_file(source, relative), staging / relative, expected)
        stop = 'evaluation-stop.json'
        if (source / stop).exists():
            copy_file(source_file(source, stop), staging / stop)
        for relative, digest in binding.get('additional_inputs', {}).items():
            copy_file(source_file(source, relative), staging / relative, digest)

        for snapshot in saved['snapshots'][1:]:
            actor = snapshot['archive']
            copy_file(source_file(source, actor), staging / actor, snapshot['archive_sha256'])
        for games, digest in sorted(binding['milestone_sha256s'].items()):
            relative = Path('artifacts') / 'cpu-panels' / games / 'request.json'
            data = checked_bytes(source_file(source, relative), digest)
            dispatch = json.loads(data)
            request = dispatch['request']
            snapshot = saved['snapshots'][int(games) // 20_000]
            if (request['snapshot'] != snapshot or request['declared_games'] != int(games)
                    or request['declaration_sha256'] != binding['declaration_sha256']):
                raise ValueError(f'milestone {games} differs from the committed parent')
            full = dispatch['full_checkpoint']
            copy_retained_checkpoint(source, staging, snapshot)
            if full['path'] != snapshot['checkpoint']:
                copy_file(source_file(source, full['path']), staging / full['path'], full['sha256'])
            elif full['sha256'] != snapshot['checkpoint_sha256']:
                raise ValueError(f'milestone {games} has a conflicting full checkpoint digest')
            target = staging / relative
            target.parent.mkdir(parents=True)
            target.write_bytes(data)
            void = relative.with_name('evaluation-void.json')
            if (source / void).exists():
                copy_file(source_file(source, void), staging / void)

        if checked_bytes(pointer, binding['state_sha256']) != pointer_data:
            raise ValueError('parent state changed during staging')
        (staging / 'state.json').write_bytes(pointer_data)
        if read_state(staging, include_full_reports=False) != saved:
            raise ValueError('staged state differs from the original committed prefix')
        if output.exists() or output.is_symlink():
            raise FileExistsError(output)
        staging.rename(output)
    return dict(output=str(output.resolve()), committed_rounds=100, state_sha256=binding['state_sha256'],
                report_sha256=report_digest)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--binding', type=Path, required=True)
    args = parser.parse_args()
    result = prepare(args.source_root, args.output, json.loads(args.binding.read_bytes()))
    print(json.dumps(result, sort_keys=True))


if __name__ == '__main__':
    main()
