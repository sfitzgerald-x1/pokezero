"""Prepare the pinned 100M main declaration and its ten base cooldown declarations."""

import argparse
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path

from gpu_engine import v2_material_contract as material
from pokezero.v2_fused_state import campaign_batch_addresses
from pokezero.v5_phasec import (
    CAMPAIGN_COOLDOWN_GAMES_BY_PARENT,
    CAMPAIGN_GAMES_PER_ROUND,
    campaign_iterations,
)
from pokezero.v5_run import _json, _sha, manifest_iteration_game_counts
from pokezero.v5_training import TrainConfig, train_config_metadata


COMMITTED_ROUNDS = 100
RESERVED_ROUNDS = 120
MAIN_STOPS = (1250, 2500, 3750, 5000)
FORK_GAMES = (13_500_000, *(9_000_000 * index for index in range(1, 10)))
ADDRESS_KEYS = (
    'stream_start', 'identity_seed_start', 'team_start', 'constructor_seed_start',
    'sampling_seed_start', 'gender_root_start',
)


def address_frontiers(iterations):
    ranges = (campaign_batch_addresses(batch) for round_ in iterations for batch in round_)
    return {
        key: max(start + size for start, size in column)
        for key, column in zip(ADDRESS_KEYS, zip(*ranges, strict=True), strict=True)
    }


def allocate(run_id, rounds, start_round, frontiers, gender):
    return [
        [asdict(batch) for batch in round_]
        for round_ in campaign_iterations(run_id, rounds, start_round=start_round, **frontiers, **gender)
    ]


def build_declarations(parent, raw_parent, pool_sha256):
    prefix = parent['iterations']
    train = parent['train']
    if (
        len(prefix) != 250
        or parent.get('train_both_seats') is not True
        or parent['max_requests'] != 1024
        or (train['batch_size'], train['epochs'], train['learning_rate']) != (2048, 1, 1.18e-4)
        or parent['evaluation']['execution'] != 'deferred'
    ):
        raise ValueError('expected the effective both-seat 5M declaration with the selected R2 recipe')
    reference = TrainConfig(**raw_parent['train'])
    expected = train_config_metadata(replace(reference, batch_size=2048, epochs=1, learning_rate=1.18e-4,
                                             learning_rate_schedule='constant', precision='float32'))
    if ((reference.batch_size, reference.epochs) != (512, 5)
            or dict(raw_parent, train=expected) != parent):
        raise ValueError('the original Phase C declaration must produce the bound effective recipe')
    reserved = prefix[:RESERVED_ROUNDS]
    if any(count != CAMPAIGN_GAMES_PER_ROUND for count in manifest_iteration_game_counts(
        dict(parent, iterations=reserved)
    )):
        raise ValueError('the 2M prefix and reserved S0 rounds must use 20,000 games per round')

    template = dict(raw_parent, pool_sha256=pool_sha256)
    gender = {key: reserved[-1][-1][key] for key in ('gender_seed', 'gender_domain')}
    frontiers = address_frontiers(reserved)
    extension = allocate(parent['run_id'], MAIN_STOPS[-1], COMMITTED_ROUNDS, frontiers, gender)
    main = dict(template, iterations=prefix[:COMMITTED_ROUNDS] + extension)
    frontiers = address_frontiers(extension)

    children = {}
    for parent_games in sorted(FORK_GAMES):
        child_games = CAMPAIGN_COOLDOWN_GAMES_BY_PARENT[parent_games]
        endpoint = parent_games + child_games
        run_id = f"{parent['run_id']}-c{endpoint}"
        rounds = child_games // CAMPAIGN_GAMES_PER_ROUND
        iterations = allocate(run_id, rounds, 0, frontiers, gender)
        policy = dict(parent['evaluation'], interval_games=child_games)
        children[parent_games] = dict(template, run_id=run_id, iterations=iterations, evaluation=policy)
        frontiers = address_frontiers(iterations)

    return main, children, frontiers['team_start']


def material_windows(main, children, stops):
    windows = []

    def add(document, start, stop):
        windows.append(dict(rounds=[start, stop], rows=material.round_rows(document['iterations'], start, stop)))

    remaining = sorted(children)
    start = COMMITTED_ROUNDS
    for stop in stops:
        add(main, start, stop)
        start = stop
        while remaining and remaining[0] <= stop * CAMPAIGN_GAMES_PER_ROUND:
            child = children[remaining.pop(0)]
            add(child, 0, len(child['iterations']))
    if remaining:
        raise ValueError('a cooldown parent is beyond the main-line horizon')
    return windows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--parent', type=Path, required=True, help='original effective declaration')
    parser.add_argument('--parent-sha256', required=True)
    parser.add_argument('--raw-parent', type=Path, required=True, help='original Phase C launch declaration')
    parser.add_argument('--raw-parent-sha256', required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--pool-sha256', help='physical combined constructor pool digest')
    mode.add_argument('--material-template', type=Path, help='physical shard plan without the round windows')
    parser.add_argument('--protected-inventories', type=Path, help='release-approved protected inventory references')
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    parent = json.loads(args.parent.read_bytes())
    if _sha(parent) != args.parent_sha256:
        parser.error('effective parent declaration digest differs')
    raw = args.raw_parent.read_bytes()
    if hashlib.sha256(raw).hexdigest() != args.raw_parent_sha256:
        parser.error('original Phase C declaration digest differs')
    raw_parent = json.loads(raw)
    pool_sha = args.pool_sha256 or '0' * 64
    if len(pool_sha) != 64 or any(c not in '0123456789abcdef' for c in pool_sha):
        parser.error('pool digest must be a lowercase SHA-256')

    successor, children, rows = build_declarations(parent, raw_parent, pool_sha)
    if args.material_template:
        from scripts.supervise_v2_campaign import material_bytes

        if args.protected_inventories is None:
            parser.error('--material-template requires --protected-inventories')
        plan = json.loads(args.material_template.read_bytes())
        if {'windows', 'preserved_team_rows', 'protected_seeds', 'protected_seed_inventories'} & plan.keys():
            parser.error('reserved rows, round windows and protected seeds are derived from bound inputs')
        plan['preserved_team_rows'] = address_frontiers(parent['iterations'][:RESERVED_ROUNDS])['team_start']
        plan['windows'] = material_windows(successor, children, MAIN_STOPS)
        plan['protected_seed_inventories'] = json.loads(args.protected_inventories.read_bytes())
        plan['protected_seeds'] = material.protected_ranges(plan['protected_seed_inventories'], material_bytes)
        material.validate_plan(plan)
        if material.shard_end(plan['shards'][-1]) != rows:
            parser.error('the material plan must cover the full declared pool')
        plan_data = material.encode(plan)
        pool_sha = material.digest(plan_data)
        for document in (successor, *children.values()):
            document['pool_sha256'] = pool_sha
    args.out.mkdir()
    if args.material_template:
        (args.out / 'material.json').write_bytes(plan_data)
    manifest = {
        'parent_declaration_sha256': args.parent_sha256,
        'pool_sha256': pool_sha,
        'required_team_rows': rows,
        'main': {'path': 'main.json', 'sha256': _sha(successor),
                 'effective_sha256': _sha(dict(successor, train=parent['train'])), 'stops': MAIN_STOPS},
        'children': [],
    }
    (args.out / 'main.json').write_text(_json(successor))
    for parent_games, child in children.items():
        path = f'fork-{parent_games}.json'
        (args.out / path).write_text(_json(child))
        manifest['children'].append({
            'path': path, 'sha256': _sha(child), 'parent_games': parent_games,
            'effective_sha256': _sha(dict(child, train=parent['train'])),
            'stop': len(child['iterations']),
        })
    (args.out / 'manifest.json').write_text(_json(manifest) + '\n')


if __name__ == '__main__':
    main()
