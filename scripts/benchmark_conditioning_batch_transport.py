"""Bounded transport benchmark, explicitly not study or strength evidence."""
import argparse
from dataclasses import fields
import hashlib
import json
from pathlib import Path
import statistics
import time

from pokezero.env import BattleStartOverride
from pokezero.local_showdown import LocalShowdownConfig, LocalShowdownEnv
from pokezero.showdown_fixture import FixturePokemon, pack_team


def normalized(value):
    if isinstance(value, dict):
        return {key: normalized(row) for key, row in value.items()}
    if isinstance(value, (list, tuple)):
        return [normalized(row) for row in value
            if not (isinstance(row, str) and row.startswith('|t:|'))]
    return value


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def contract(env):
    result = {}
    for player in ('p1', 'p2'):
        state = env.public_materialization_state(player)
        result[player] = {field.name: (state.belief_engine.snapshot() if field.name == 'belief_engine'
            else getattr(state, field.name)) for field in fields(state)}
    return (result, {player: env.observe(player) for player in env.requested_players()},
        normalized(env.snapshot().bridge_snapshot))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--showdown-root', type=Path, required=True)
    parser.add_argument('--rounds', type=int, default=5)
    parser.add_argument('--trials', type=int, default=128)
    args = parser.parse_args()
    if args.output.exists() or not 1 <= args.rounds <= 10 or not 8 <= args.trials <= 512 or args.trials % 8:
        raise ValueError('create-only output and bounded complete eight-trial batches required')
    root = Path(__file__).resolve().parents[1]
    files = [Path(__file__), root/'src/pokezero/local_showdown.py', root/'scripts/battle_bridge.mjs',
        root/'scripts/battle_bridge_conditioning_batch.mjs']
    hashes = {str(path):digest(path) for path in files}
    config = LocalShowdownConfig(showdown_root=args.showdown_root, set_belief_source=True)
    override = BattleStartOverride(observation_format_id='gen3randombattle', player_teams={
        'p1':pack_team((FixturePokemon('Tauros',('Body Slam','Earthquake'),ability='Intimidate',level=76),)),
        'p2':pack_team((FixturePokemon('Snorlax',('Rest','Body Slam'),ability='Immunity',level=71),))})
    rows = []
    with LocalShowdownEnv(config) as env:
        env.reset_with_start_override(seed=17, start_override=override)
        env.step({'p1':0,'p2':1})
        resident = env.snapshot_for_search()
        prefix = tuple(event.raw_line for event in resident.replay.public_events
            if event.raw_line not in ('','|',"|message|The battle's RNG was reset."))
        # An impossible target forces all identical seeds through both paths.
        target = prefix + ('|message|not observed benchmark sentinel',)
        actions = {'p1':0,'p2':1}
        for round_id in range(args.rounds):
            seeds = [round_id*1000+index for index in range(args.trials)]
            results = {}
            # Alternate ordering to reduce warmup/temporal bias.
            for mode in (('single','batch8') if round_id%2==0 else ('batch8','single')):
                before = env.root_puct_bridge_timing_snapshot()['bridge_round_trip_count']
                began = time.perf_counter()
                if mode == 'single':
                    for seed in seeds:
                        env.step_from_search_snapshot_for_conditioning(resident, actions, chance_seed=seed)
                        if env.terminal() is None:
                            env.public_materialization_state('p1')
                else:
                    for start in range(0,len(seeds),8):
                        result = env.conditioning_batch_from_search_snapshot(resident,actions,
                            chance_seeds=seeds[start:start+8],expected_history=target,
                            deadline_at=time.perf_counter()+10)
                        if result['consumed'] != 8 or result['matched'] or result['deadline_reached']:
                            raise RuntimeError('benchmark failed exact eight-trial consumption')
                elapsed = time.perf_counter()-began
                results[mode] = dict(seconds=elapsed, trials=len(seeds),
                    round_trips=env.root_puct_bridge_timing_snapshot()['bridge_round_trip_count']-before,
                    trials_per_second=len(seeds)/elapsed)
                results[mode]['contract'] = contract(env)
            if results['single'].pop('contract') != results['batch8'].pop('contract'):
                raise RuntimeError('benchmark final hidden/public/lazy-view contract drift')
            rows.append(dict(round=round_id,results=results,
                speedup=results['single']['seconds']/results['batch8']['seconds'], final_contract_equal=True))
        env.release_search_snapshot(resident)
    if any(digest(path)!=expected for path,expected in hashes.items()):
        raise RuntimeError('benchmark source changed during measurement')
    receipt = dict(schema='pokezero.conditioning-batch-transport-benchmark.v1',
        status='TRANSPORT_EQUIVALENCE_AND_THROUGHPUT_MEASURED_NOT_STRENGTH',
        rows=rows,median_speedup=statistics.median(row['speedup'] for row in rows),
        source_hashes=hashes,study_resumed=False,strength_inference=False,
        limitation='fixture retry transport only; no claim about full posterior/search throughput or root400')
    with args.output.open('x') as stream:
        json.dump(receipt,stream,indent=2,sort_keys=True)
        stream.write('\n')
    print(json.dumps({key:receipt[key] for key in ('status','median_speedup')},sort_keys=True))


if __name__ == '__main__':
    main()
