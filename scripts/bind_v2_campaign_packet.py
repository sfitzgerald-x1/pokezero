"""Bind assembled pool, runtime and declaration artifacts into a V2 release packet."""

import argparse
import hashlib
import json
from pathlib import Path
import re

from gpu_engine import v2_material_contract as material
from pokezero.v5_phasec import CAMPAIGN_GAMES_PER_ROUND
from pokezero.v5_run import _json
from scripts.prepare_v2_campaign_declarations import material_windows


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



def bind_material(config, declarations, local_root, plan_ref, runtime):
    from scripts.supervise_v2_campaign import material_bytes, verify_material_files

    plan = material.load_plan(plan_ref, material_bytes)
    if (plan_ref['sha256'] != declarations['pool_sha256']
            or plan['protected_seed_inventories'] != config['protected_seed_inventories']
            or material.shard_end(plan['shards'][-1]) != declarations['required_team_rows']
            or plan['parent_pool'] != runtime['constructor_pool']
            or plan['gender_policy'] != runtime['gender_policy'] or plan['showdown'] != runtime['showdown']):
        raise ValueError('material plan differs from the runtime or full declaration')

    def declared(row):
        member = Path(row['path'])
        if len(member.parts) != 1 or member.is_absolute() or member.name == '..':
            raise ValueError('expected a declaration filename in the bundle')
        document = read_reference(dict(path=local_root / member, sha256=row['sha256']))
        if document['pool_sha256'] != plan_ref['sha256']:
            raise ValueError('declaration does not bind its material plan')
        return document

    main = declared(declarations['main'])
    children = {row['parent_games']: declared(row) for row in declarations['children']}
    if material_windows(main, children, declarations['main']['stops']) != plan['windows']:
        raise ValueError('material windows differ from the declared segment schedule')
    ready = []
    previous = None
    for index in range(len(plan['windows'])):
        try:
            ref, document = material.bound_manifest(plan, plan_ref['sha256'], index, previous, material_bytes)
            material.require_verification(plan, plan_ref['sha256'], index, ref, material_bytes)
        except FileNotFoundError:
            break
        verify_material_files(document)
        ready.append(ref)
        previous = ref['sha256']
    if not ready:
        raise ValueError('publish and verify at least the first segment before binding the packet')
    return dict(plan=plan_ref, ready=ready, producer=config['producer'])

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--pool-result', type=Path)
    mode.add_argument('--material-plan', type=Path, help='published physical address plan')
    parser.add_argument('--gender-policy', type=Path)
    parser.add_argument('--declarations', type=Path, required=True, help='declaration manifest')
    parser.add_argument('--declarations-root', type=Path, required=True, help='shared path used by training ranks')
    parser.add_argument('--runtime-path', type=Path, required=True, help='shared runtime path used by training ranks')
    parser.add_argument('--root-binding', type=Path, required=True)
    parser.add_argument('--restore-result', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_bytes())
    declarations = json.loads(args.declarations.read_bytes())
    if not args.declarations_root.is_absolute() or not args.runtime_path.is_absolute():
        parser.error('published declaration and runtime references must be absolute')
    previous = read_reference(config['previous_runtime'])
    if args.material_plan:
        runtime = dict(previous, source_commit=config['source_commit'])
        plan_ref = dict(path=str(args.material_plan), sha256=declarations['pool_sha256'])
        stream = bind_material(config, declarations, args.declarations.parent, plan_ref, runtime)
    else:
        if args.gender_policy is None:
            parser.error('--pool-result requires --gender-policy')
        result = json.loads(args.pool_result.read_bytes())
        if (result['team_rows'] != declarations['required_team_rows']
                or result['sha256'] != declarations['pool_sha256']):
            parser.error('roster receipt differs from the declaration bundle')
        pool_ref = {key: result[key] for key in ('path', 'sha256')}
        policy = args.gender_policy.read_bytes()
        policy_ref = dict(path=config['gender_policy_path'], sha256=digest(policy))
        runtime = rebind_runtime(previous, read_reference(previous['gender_policy']), json.loads(policy),
                                 pool_ref, policy_ref, config['source_commit'])
    runtime_data = _json(runtime).encode()
    runtime_ref = dict(path=str(args.runtime_path), sha256=digest(runtime_data))
    packet = build_packet(config, declarations, args.declarations_root, runtime_ref)
    if args.material_plan:
        packet['material'] = stream
    receipt = args.restore_result.read_bytes()
    restored = json.loads(receipt)
    binding = json.loads(args.root_binding.read_bytes())
    if (restored['output'] != f"/shared/scott/v2p/{packet['main']['output']}"
            or restored['committed_rounds'] != 100
            or restored['state_sha256'] != binding['state_sha256']
            or declarations['parent_declaration_sha256'] != binding['declaration_sha256']):
        parser.error('restored root differs from the selected 2M prefix and declared output')
    milestones = binding['milestone_sha256s']
    if set(milestones) != {'1000000', '2000000'}:
        parser.error('the original 1M and 2M dispatch digests are required')
    packet['main']['restore'] = dict(
        state_sha256=binding['state_sha256'], checkpoint_sha256=binding['checkpoint_sha256'],
        declaration_sha256=binding['declaration_sha256'], receipt_sha256=digest(receipt),
        milestone_sha256s=milestones,
    )
    args.out.mkdir()
    (args.out / 'runtime.json').write_bytes(runtime_data)
    (args.out / 'packet.json').write_text(_json(packet) + '\n')


if __name__ == '__main__':
    main()
