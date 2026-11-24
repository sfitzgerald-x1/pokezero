"""Bind assembled pool, runtime and declaration artifacts into a V2 release packet."""

import argparse
import hashlib
import json
from pathlib import Path
import re

from pokezero.v5_phasec import CAMPAIGN_GAMES_PER_ROUND
from pokezero.v5_run import _json


def digest(data):
    return hashlib.sha256(data).hexdigest()


def read_reference(ref):
    data = Path(ref['path']).read_bytes()
    if digest(data) != ref['sha256']:
        raise ValueError('input artifact differs from its binding')
    return json.loads(data)


def rebind_runtime(previous, old_policy, new_policy, pool_ref, policy_ref, source_commit):
    def semantic_policy(document, pool):
        provenance = dict(document['provenance'])
        if provenance.pop('pool_sha256') != pool['sha256']:
            raise ValueError('gender policy differs from its constructor pool')
        return dict(document, provenance=provenance)

    if not re.fullmatch(r'[0-9a-f]{40}', source_commit):
        raise ValueError('production runtime needs a pinned source commit')
    if semantic_policy(old_policy, previous['constructor_pool']) != semantic_policy(new_policy, pool_ref):
        raise ValueError('the production pool changes the original gender constructor semantics')
    return dict(previous, source_commit=source_commit, constructor_pool=pool_ref, gender_policy=policy_ref)


def build_packet(config, declarations, declarations_root, runtime_ref):
    expected = {row['parent_games']: row for row in declarations['children']}
    declared = {row['parent_games']: row for row in config['children']}
    if (len(expected) != len(declarations['children']) or len(declared) != len(config['children'])
            or expected.keys() != declared.keys()):
        raise ValueError('production packet must cover every declared cooldown exactly once')

    def reference(row):
        relative = Path(row['path'])
        if relative.is_absolute() or len(relative.parts) != 1 or relative.name == '..':
            raise ValueError('expected a declaration filename in the sealed bundle')
        return {'path': str(declarations_root / relative), 'sha256': row['sha256'],
                'effective_sha256': row['effective_sha256']}

    main = dict(config['main'], stops=declarations['main']['stops'],
                declaration=reference(declarations['main']))
    children = []
    for parent_games, row in expected.items():
        child = declared[parent_games]
        if parent_games % CAMPAIGN_GAMES_PER_ROUND:
            raise ValueError('cooldown parent is outside a committed round boundary')
        children.append(dict(child, parent_round=parent_games // CAMPAIGN_GAMES_PER_ROUND,
                             stop=row['stop'], declaration=reference(row)))
    fields = ('checkout', 'source_commit', 'launcher', 'resilience', 'cpu', 'memory',
              'reference_run', 'stop_key_reference')
    return dict({key: config[key] for key in fields}, main=main, children=children,
                previous_runtime=config['previous_runtime']['path'], runtime=runtime_ref)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--pool-result', type=Path, required=True)
    parser.add_argument('--gender-policy', type=Path, required=True)
    parser.add_argument('--declarations', type=Path, required=True, help='declaration manifest')
    parser.add_argument('--declarations-root', type=Path, required=True, help='shared path used by training ranks')
    parser.add_argument('--runtime-path', type=Path, required=True, help='shared runtime path used by training ranks')
    parser.add_argument('--root-binding', type=Path, required=True)
    parser.add_argument('--restore-result', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_bytes())
    result = json.loads(args.pool_result.read_bytes())
    declarations = json.loads(args.declarations.read_bytes())
    if (result['team_rows'] != declarations['required_team_rows']
            or result['sha256'] != declarations['pool_sha256']):
        parser.error('roster receipt differs from the declaration bundle')
    if not args.declarations_root.is_absolute() or not args.runtime_path.is_absolute():
        parser.error('published declaration and runtime references must be absolute')
    pool_ref = {key: result[key] for key in ('path', 'sha256')}
    previous = read_reference(config['previous_runtime'])
    policy = args.gender_policy.read_bytes()
    policy_ref = dict(path=config['gender_policy_path'], sha256=digest(policy))
    runtime = rebind_runtime(previous, read_reference(previous['gender_policy']), json.loads(policy),
                             pool_ref, policy_ref, config['source_commit'])
    runtime_data = _json(runtime).encode()
    runtime_ref = dict(path=str(args.runtime_path), sha256=digest(runtime_data))
    packet = build_packet(config, declarations, args.declarations_root, runtime_ref)
    receipt = args.restore_result.read_bytes()
    restored = json.loads(receipt)
    binding = json.loads(args.root_binding.read_bytes())
    if (restored['output'] != f"/shared/scott/v2p/{packet['main']['output']}"
            or restored['committed_rounds'] != 100
            or restored['state_sha256'] != binding['state_sha256']
            or declarations['parent_declaration_sha256'] != binding['declaration_sha256']):
        parser.error('restored root differs from the selected 2M prefix and declared output')
    packet['main']['restore'] = dict(
        state_sha256=binding['state_sha256'], checkpoint_sha256=binding['checkpoint_sha256'],
        declaration_sha256=binding['declaration_sha256'], receipt_sha256=digest(receipt),
    )
    args.out.mkdir()
    (args.out / 'runtime.json').write_bytes(runtime_data)
    (args.out / 'packet.json').write_text(_json(packet) + '\n')


if __name__ == '__main__':
    main()
