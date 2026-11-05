"""Paired head-to-head of two v5 actors on the GPU engine (refill collector).

Each evaluation game is played twice on identical initial states (same teams,
constructor and gender seeds): once with actor A in seat 0 and once with A in
seat 1. The pair score is A's mean result over the two games (win 1, draw 0.5,
loss 0). Pairs with a capped or refused game are excluded and counted. Reports
A's paired score, its standard error and a normal 90% interval.

Evaluation games come from a declared diagnostic config's roster (`--config`,
smoke schema, `--iteration` selects which declared iteration's blocks). Actors
are named by run root and iteration: `<run>/state.json` (V2 runs) or the run's
journal SNAPSHOT events (file-pipeline runs).

All paths are command-line arguments.
"""
import argparse
from dataclasses import replace
import json
import math
from pathlib import Path
import time

import numpy as np

from gpu_engine.v2_bench_collector import log
from gpu_engine.v2_fused import journal

SWAP_OFFSET = 1_000_000_000


def snapshot(run, iteration):
    from pokezero import v5_run as runs
    state = Path(run, 'state.json')
    if state.exists():
        found = [x for x in json.loads(state.read_text())['snapshots'] if x['iteration'] == iteration]
    else:
        found = [e['payload'] for e in journal(run) if e['kind'] == 'SNAPSHOT' and e['payload']['iteration'] == iteration]
    if len(found) < 1:raise ValueError(f'no snapshot for iteration {iteration} in {run}')
    return runs.Snapshot(**found[-1])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True);parser.add_argument('--iteration', type=int, default=1)
    parser.add_argument('--blocks', type=int, default=3, help='declared 2,048-game blocks; each is played twice')
    parser.add_argument('--a-run', required=True);parser.add_argument('--a-iteration', type=int, required=True)
    parser.add_argument('--b-run', required=True);parser.add_argument('--b-iteration', type=int, required=True)
    parser.add_argument('--lanes', type=int, default=2048);parser.add_argument('--out', required=True)
    parser.add_argument('--label', default='head-to-head')
    args = parser.parse_args(argv)
    import jax
    from pokezero import v5_run as runs
    from gpu_engine import v5_device_rollout as rollout, v2_collector as v2
    from gpu_engine.v5_device_backend import RuntimeBackend
    from gpu_engine.v2_fused import prepare
    begin = time.perf_counter()
    prep = argparse.Namespace(config=args.config, arm='B', iterations=2 if args.iteration else 1)
    context = prepare(prep);plan = context['plan']
    a = snapshot(args.a_run, args.a_iteration);b = snapshot(args.b_run, args.b_iteration)
    if a.archive_sha256 == b.archive_sha256:raise ValueError('two distinct actors required')
    def load(snapshot, root):
        # Same architecture as the evaluation forward; `replicate` (seed label) may differ.
        from gpu_engine import v5_model
        params, config, _ = v5_model.load(Path(root, snapshot.archive), expected_archive_sha256=snapshot.archive_sha256,
            expected_checkpoint_sha256=snapshot.checkpoint_sha256, expected_category_vocabulary_sha256=plan.model.category_vocabulary_sha256,
            expected_label_vocabulary_sha256=plan.model.label_vocabulary.sha256, enable_v5=True)
        if replace(config, replicate=plan.model.replicate) != plan.model:raise ValueError('actor architecture differs from evaluation model')
        return params
    params = (load(a, args.a_run), load(b, args.b_run))
    backend = RuntimeBackend(context['prepared'], context['pool'], gender_policy=context['gender']);backend.check_plan(plan)
    bindings = rollout.Bindings(plan.model.category_vocabulary_sha256, plan.model.label_vocabulary, context['prepared'].source_sha256,
        (a.archive_sha256,) * 2, (True, True), backend.declaration_sha256)
    blocks = [];pairs = []
    for batch in plan.iterations[args.iteration][:args.blocks]:
        swapped = replace(batch, games=tuple(replace(g, stream_id=g.stream_id - SWAP_OFFSET, battle_id=g.battle_id + '-swap')
            for g in batch.games))
        for side, declared in ((0, batch), (1, swapped)):
            assignments = [dict(stream_id=g.stream_id, learner_side=side, opponent=dict(archive_sha256=b.archive_sha256))
                for g in declared.games]
            bank = rollout.actor_bank_from_assignments(bindings, (a.archive_sha256, b.archive_sha256),
                current_archive=a.archive_sha256, assignments=assignments)
            blocks.append((jax.block_until_ready(backend._construct(declared)), bank, declared.sampling_seed))
        pairs.extend((g.stream_id, g.stream_id - SWAP_OFFSET) for g in batch.games)
    roster = v2.roster_from_blocks(blocks, archives=(a.archive_sha256, b.archive_sha256), groups=((0, 1), (1, 0)))
    collector = v2.RefillCollector(context['prepared'].engine_tables, context['prepared'].observation_tables, plan.model,
        backend.labels, roster, lanes=args.lanes, max_requests=plan.max_requests, rows_per_game=64)
    compile_seconds = collector.compile(params, threads=16)
    result = collector.run(params);records = v2.game_records(result)
    index = {int(s): i for i, s in enumerate(records['streams'])}
    from gpu_engine.v5_device_rollout import STATUS
    def score(stream, a_seat):
        i = index[stream];status = STATUS[int(records['status'][i])]
        if status != 'COMPLETE':return None
        winner = int(records['winner'][i])
        return 0.5 if winner == 2 else 1.0 if winner == a_seat else 0.0
    paired = [];excluded = 0;seat = {0: [], 1: []}
    for first, second in pairs:
        x, y = score(first, 0), score(second, 1)
        if x is not None:seat[0].append(x)
        if y is not None:seat[1].append(y)
        if x is None or y is None:excluded += 1;continue
        paired.append((x + y) / 2)
    values = np.asarray(paired, np.float64);n = len(values)
    mean = float(values.mean());se = float(values.std(ddof=1) / math.sqrt(n))
    report = dict(schema='v2-head-to-head.v1', label=args.label, a=dict(run=args.a_run, iteration=a.iteration, archive_sha256=a.archive_sha256),
        b=dict(run=args.b_run, iteration=b.iteration, archive_sha256=b.archive_sha256), pairs=n, excluded_pairs=excluded,
        a_paired_score=mean, standard_error=se, interval_90=[mean - 1.645 * se, mean + 1.645 * se],
        a_minus_b_points=200 * (mean - 0.5), a_minus_b_interval_90_points=[200 * (mean - 1.645 * se - 0.5), 200 * (mean + 1.645 * se - 0.5)],
        a_score_by_a_seat={str(k): float(np.mean(v)) if v else None for k, v in seat.items()},
        dispositions={name: int((np.asarray(records['status']) == code).sum()) for code, name in enumerate(STATUS)},
        collector=dict(occupancy=result.stats['occupancy'], kernel_seconds=result.stats['kernel_seconds']),
        compile_seconds=compile_seconds, wall_seconds=time.perf_counter() - begin,
        scope='Paired engine head-to-head, same teams with seats swapped; not a Foul Play result.')
    out = Path(args.out);out.mkdir(parents=True, exist_ok=True)
    (out / f'{args.label}.json').write_text(json.dumps(report, indent=1, default=str) + '\n')
    log('HEAD_TO_HEAD', **{k: v for k, v in report.items() if k not in ('compile_seconds',)})
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
