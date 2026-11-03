"""V2-a kernel-only refill collection benchmark on one GPU.

Mirrors v5_device_smoke.execute up to the plan, runtime preparation and the
original per-block construction, then collects one declared roster with the
refill collector, including on-device row compaction and row-store
finalization. No training and no files other than the report. Optional
reference: the measured run's journal (per-game terminal records, initial
actor snapshot, assignments) and its v5 caches (row-level equality).

All paths are command-line arguments.
"""
import argparse
import json
import math
import os
from pathlib import Path
import time

import numpy as np


def log(message, **values):
    print(json.dumps(dict(message=message, time=time.strftime('%H:%M:%S'), **values), default=str), flush=True)


def journal(run):
    events = []
    for path in sorted(Path(run, 'journal').glob('*.json')):
        events.append(json.loads(path.read_text()))
    return events


def prepare(args):
    from pokezero import v5_worker as worker, v5_run as runs, v5_contract as contract, v5_device_smoke as smoke
    config_path = Path(args.config);blob = config_path.read_bytes()
    import hashlib
    config = json.loads(blob);config_sha = hashlib.sha256(blob).hexdigest()
    runtime = worker._record(config['runtime']);labels = contract.LabelVocabulary(**worker._record(config['labels']))
    profile, vocabulary, gender, category = worker._runtime_metadata(runtime, expected_pool_sha256=config['bank']['pool_sha256'])
    if gender is None:raise ValueError('original gender metadata for declared pool required')
    plan = smoke.build_plan(config['bank'], profile, labels, category, arm=args.arm, iterations=1)
    # Same precision/runtime probe as the original worker; the memory policy
    # is the launcher's (this benchmark holds the row store, no trainer).
    try:worker._probe(runtime)
    except ValueError as error:
        # Record a runtime difference instead of hiding it; keep the precision policy.
        import jax
        log('PROBE_DIFFERS', error=str(error))
        jax.config.update('jax_enable_x64', False);jax.config.update('jax_default_matmul_precision', 'highest')
    inputs = worker.Inputs(plan, profile, json.dumps(runtime), json.dumps(vocabulary), json.dumps(gender), runs._sha(plan.manifest()))
    begin = time.perf_counter();prepared, pool, _, _ = worker.prepare_device_runtime(inputs)
    return dict(config=config, config_sha256=config_sha, runtime=runtime, plan=plan, prepared=prepared, pool=pool,
        gender=gender, preparation_seconds=time.perf_counter() - begin)


def actor(args, plan, events):
    """Iteration-0 actor: the run's own exported initial snapshot, verified."""
    from pokezero import v5_run as runs
    from gpu_engine import v5_model
    snapshots = [e['payload'] for e in events if e['kind'] == 'SNAPSHOT' and e['payload']['iteration'] == -1]
    if len(snapshots) != 1:raise ValueError('one initial actor snapshot required in the reference journal')
    snapshot = runs.Snapshot(**snapshots[0])
    if snapshot.model_sha256 != runs._sha(plan.model.metadata()):raise ValueError('initial snapshot model differs from plan')
    params, config, metadata = v5_model.load(Path(args.reference_run, snapshot.archive), expected_archive_sha256=snapshot.archive_sha256,
        expected_checkpoint_sha256=snapshot.checkpoint_sha256, expected_category_vocabulary_sha256=plan.model.category_vocabulary_sha256,
        expected_label_vocabulary_sha256=plan.model.label_vocabulary.sha256, enable_v5=True)
    if config != plan.model:raise ValueError('loaded initial actor configuration differs')
    return snapshot, params, metadata


def construct(context, snapshot, events, *, games=None):
    """Original per-block constructor calls and planner assignments."""
    import jax
    from pokezero import v5_run as runs
    from gpu_engine import v5_device_rollout as rollout, v2_collector as v2
    from gpu_engine.v5_device_backend import RuntimeBackend
    plan = context['plan'];prepared = context['prepared']
    backend = RuntimeBackend(prepared, context['pool'], gender_policy=context['gender']);backend.check_plan(plan)
    ordinal = {g.stream_id: i for i, g in enumerate(plan.games)}
    started = {tuple(e['payload']['streams']): e['payload']['assignments'] for e in events if e['kind'] == 'BATCH_STARTED'}
    bindings = rollout.Bindings(plan.model.category_vocabulary_sha256, plan.model.label_vocabulary, prepared.source_sha256,
        (snapshot.archive_sha256,)*2, (True, True), backend.declaration_sha256)
    blocks = [];timings = [];matched = 0;total = 0
    for batch in plan.iterations[0]:
        if games is not None and total >= games:break
        assignments = tuple(runs.assignment(plan, g, ordinal[g.stream_id], snapshot, ()) for g in batch.games)
        recorded = json.loads(runs._json(assignments))
        reference = started.get(tuple(g.stream_id for g in batch.games))
        if reference is not None:
            if reference != recorded:raise ValueError('recomputed planner assignments differ from the reference run')
            matched += 1
        bank = rollout.actor_bank_from_assignments(bindings, (snapshot.archive_sha256,), current_archive=snapshot.archive_sha256,
            assignments=recorded)
        begin = time.perf_counter();carry = jax.block_until_ready(backend._construct(batch))
        timings.append(time.perf_counter() - begin)
        blocks.append((carry, bank, batch.sampling_seed));total += len(batch.games)
    begin = time.perf_counter();roster = v2.roster_from_blocks(blocks);jax.block_until_ready(roster.device)
    del blocks
    nbytes = sum(x.nbytes for x in jax.tree.leaves(roster.device))
    return backend, roster, dict(blocks=len(timings), construct_seconds=timings, concat_seconds=time.perf_counter() - begin,
        assignment_blocks_matching_reference=matched, roster_device_bytes=nbytes)


def compare_games(result, events):
    from gpu_engine import v2_collector as v2
    from gpu_engine.v5_device_rollout import STATUS
    records = v2.game_records(result)
    ends = {e['payload']['stream_id']: e['payload'] for e in events if e['kind'] == 'GAME_END'}
    same = dict(status=0, terminal_frame=0, winner=0, all=0);missing = 0;differ = []
    for i, stream in enumerate(records['streams']):
        ref = ends.get(int(stream))
        if ref is None:missing += 1;continue
        status = STATUS[int(records['status'][i])];frame = int(records['terminal_frame'][i])
        winner = int(records['winner'][i]) if status == 'COMPLETE' else None
        checks = dict(status=status == ref['status'], terminal_frame=frame == ref['terminal_frame'], winner=winner == ref['terminal_winner'])
        for key, value in checks.items():same[key] += value
        if all(checks.values()):same['all'] += 1
        elif len(differ) < 20:differ.append(dict(stream=int(stream), v2=[status, frame, winner], reference=[ref['status'], ref['terminal_frame'], ref['terminal_winner']]))
    return dict(games=len(records['streams']), reference_missing=missing, equal=same, first_differences=differ)


def compare_cache(store, cache_path, model_config, *, stream_range):
    """Exact row equality vs one original v5 cache, keyed by (stream, frame, owner)."""
    import jax
    import jax.numpy as j
    import torch
    from pokezero import v5_data
    from pokezero.v5_loss import v5_batch_to_torch, INTEGER_FIELDS
    begin = time.perf_counter()
    cache = v5_data.VerifiedCache(cache_path)
    batch = next(cache.batches(batch_size=1 << 30))
    reference = v5_batch_to_torch(batch, model_config, training=True)
    ref_keys = np.asarray([(a['stream'], a['frame'], a['owner']) for a in batch.v5_columns['addresses']], np.int64)
    read_seconds = time.perf_counter() - begin
    lo, hi = stream_range
    streams = store.arrays['streams'][:store.n]
    index = j.nonzero((streams >= lo) & (streams <= hi))[0]
    get = lambda name: np.asarray(jax.device_get(j.take(store.arrays[name], index, axis=0)))
    keys = np.stack([get('streams'), get('frames'), get('owners')], axis=1).astype(np.int64)
    ours = {name: get(name) for name in store.arrays if name not in ('streams', 'frames', 'owners', 'games', 'turns', 'terminal_status')}
    ours.update({k: v[:len(keys)] for k, v in store._constants(np).items()})
    status = np.asarray(jax.device_get(j.take(store.arrays['terminal_status'], index)))
    from gpu_engine.v5_device_rollout import STATUS
    def table(k):return {tuple(row): i for i, row in enumerate(k)}
    ref_index, our_index = table(ref_keys), table(keys)
    shared = sorted(set(ref_index) & set(our_index))
    r = np.asarray([ref_index[k] for k in shared], np.int64);o = np.asarray([our_index[k] for k in shared], np.int64)
    fields = {};row_equal = np.ones(len(shared), bool)
    for name, tensor in reference.items():
        if name in ('schema', 'observation_schema', 'category_vocabulary_sha256', 'label_vocabulary_sha256'):
            fields[name] = dict(equal=tensor == store.binding[name]);continue
        if name == 'terminal_status':
            want = np.asarray(tensor, dtype=object)[r];got = np.asarray(STATUS, dtype=object)[status[o]]
            eq = want == got
        else:
            want = tensor.numpy()[r];got = ours[name][o]
            if name in INTEGER_FIELDS:got = got.astype(np.int64)
            if got.dtype != want.dtype:fields[name] = dict(equal=False, dtype=[str(got.dtype), str(want.dtype)]);row_equal[:] = False;continue
            eq = np.all((got == want).reshape(len(shared), -1), axis=1)
        entry = dict(rows_equal=int(eq.sum()), rows=len(shared))
        if name != 'terminal_status' and want.dtype.kind == 'f' and len(shared):
            entry['max_abs_diff'] = float(np.max(np.abs(got.astype(np.float64) - want.astype(np.float64))))
        fields[name] = entry;row_equal &= eq
    games_ref = set(ref_keys[:, 0].tolist());games_ours = set(keys[:, 0].tolist())
    bad_games = sorted(set(np.asarray(shared)[~row_equal][:, 0].tolist()) if len(shared) else set())
    only_ref = set(ref_index) - set(our_index);only_ours = set(our_index) - set(ref_index)
    divergent = sorted(set(bad_games) | {k[0] for k in only_ref} | {k[0] for k in only_ours})
    first = {}
    if divergent:
        sk = np.asarray(shared, np.int64) if shared else np.zeros((0, 3), np.int64)
        for game in divergent[:20]:
            frames = sk[(sk[:, 0] == game) & ~row_equal][:, 1] if len(sk) else []
            extra = [k[1] for k in only_ref | only_ours if k[0] == game]
            first[int(game)] = int(min(list(frames) + extra)) if (len(frames) or extra) else None
    return dict(cache=str(cache_path), reference_rows=len(ref_keys), v2_rows=len(keys), shared_rows=len(shared),
        only_reference=len(only_ref), only_v2=len(only_ours), rows_fully_equal=int(row_equal.sum()),
        games_reference=len(games_ref), games_v2=len(games_ours), divergent_games=len(divergent),
        first_divergent_frame=first, fields=fields, read_seconds=read_seconds, compare_seconds=time.perf_counter() - begin)


def measure(args, context, backend, roster, params, events, lanes):
    import jax
    from gpu_engine import v2_collector as v2
    plan = context['plan'];prepared = context['prepared']
    collector = v2.RefillCollector(prepared.engine_tables, prepared.observation_tables, plan.model, backend.labels, roster,
        lanes=lanes, max_requests=plan.max_requests, chunk_steps=args.chunk_steps, shrink_factor=args.shrink_factor,
        min_lanes=args.min_lanes, rows_per_game=args.rows_per_game)
    log('LEVELS', lanes=lanes, levels=[x.lanes for x in collector.levels], row_capacity=collector.row_capacity(),
        row_buffer_gib=collector.row_capacity() * v2.ROW_BYTES / 2**30)
    begin = time.perf_counter()
    compile_seconds = collector.compile(params, threads=args.compile_threads, progress=lambda x: log('COMPILE', **x))
    compile_wall = time.perf_counter() - begin
    log('COMPILED', lanes=lanes, wall_seconds=compile_wall, **compile_seconds)
    timeline = [];last = [0.]
    def progress(x):
        timeline.append([round(x['elapsed_seconds'], 3), x['level'], x['live'], x['started'], x['rows']])
        if x['elapsed_seconds'] - last[0] >= args.log_every:
            last[0] = x['elapsed_seconds'];log('PROGRESS', lanes=lanes, **x)
    result = collector.run(params, progress=progress)
    stats = result.stats
    log('COLLECTED', lanes=lanes, kernel_seconds=stats['kernel_seconds'], occupancy=stats['occupancy'], rows=stats['rows'],
        tail_seconds=stats['tail_seconds'])
    store = v2.finalize(result, loss_config=plan.loss, discount=plan.discount)
    handoff = time.perf_counter();tensors = store.to_torch(exact_dtypes=False, status_strings=False)
    handoff = dict(seconds=time.perf_counter() - handoff, device=str(tensors['numeric_features'].device),
        numeric_shape=list(tensors['numeric_features'].shape), keys=len(tensors))
    del tensors
    dispositions = store.stats['dispositions'];complete = dispositions['COMPLETE']
    seconds = stats['kernel_seconds'];final = store.stats['finalize_seconds']
    report = dict(lanes=lanes, levels=[x.lanes for x in collector.levels], games=roster.games,
        compile_wall_seconds=compile_wall, compile_seconds=compile_seconds,
        kernel_seconds=seconds, finalize_seconds=final,
        completed_games=complete, dispositions=dispositions,
        completed_games_per_kernel_second=complete / seconds,
        completed_games_per_second_including_finalize=complete / (seconds + final),
        occupancy=stats['occupancy'], live_lane_steps=stats['live_lane_steps'], physical_lane_steps=stats['physical_lane_steps'],
        ticks=stats['ticks'], tail_seconds=stats['tail_seconds'], tail_ticks=stats['tail_ticks'], queue_exhausted=stats['queue_exhausted'],
        per_level=stats['levels'], compactions=stats['compactions'], rows=store.n, rows_per_completed_game=store.n / max(1, complete),
        ppo_rows=store.stats['ppo_rows'], trajectories=store.stats['trajectories'], row_capacity=stats['row_capacity'],
        row_segments=stats['row_segments'], requested_views=stats['requested_views'], torch_handoff=handoff,
        chunk_steps=args.chunk_steps, shrink_factor=args.shrink_factor, min_lanes=args.min_lanes,
        bar=dict(occupancy_min=.85, games_per_second_min=250.,
            passed=bool(stats['occupancy'] >= .85 and complete / seconds >= 250.)))
    log('MEASURED', **{k: v for k, v in report.items() if k not in ('per_level', 'compactions', 'compile_seconds')})
    if events:
        report['reference_games'] = compare_games(result, events);log('GAME_RECORDS', **report['reference_games'])
        caches = sorted((e['payload'] for e in events if e['kind'] == 'CACHE'), key=lambda x: x['batch'])
        report['reference_caches'] = []
        for entry in caches[:args.compare_caches]:
            batch = plan.iterations[0][entry['batch']];lo, hi = batch.games[0].stream_id, batch.games[-1].stream_id
            if hi > int(roster.streams.max()):continue
            value = compare_cache(store, Path(args.reference_run, entry['path']), plan.model, stream_range=(lo, hi))
            value['batch'] = entry['batch'];report['reference_caches'].append(value)
            log('CACHE_ROWS', **{k: v for k, v in value.items() if k != 'fields'},
                fields_not_equal={k: v for k, v in value['fields'].items() if not v.get('equal', v.get('rows_equal') == v.get('rows'))})
    report['timeline'] = timeline
    del store, result, collector
    jax.clear_caches() if args.clear_caches else None
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True, help='measured run config.json (v5 device smoke schema)')
    parser.add_argument('--reference-run', required=True, help='measured run root: journal, artifacts, caches')
    parser.add_argument('--out', default=os.environ.get('V2_OUT'))
    parser.add_argument('--arm', default='B', choices=('A', 'B'))
    parser.add_argument('--lanes', type=int, nargs='+', default=[2048])
    parser.add_argument('--games', type=int, default=None, help='first N declared games (whole blocks); default all')
    parser.add_argument('--chunk-steps', type=int, default=8)
    parser.add_argument('--shrink-factor', type=int, default=4)
    parser.add_argument('--min-lanes', type=int, default=128)
    parser.add_argument('--rows-per-game', type=int, default=112)
    parser.add_argument('--compile-threads', type=int, default=4)
    parser.add_argument('--compare-caches', type=int, default=1)
    parser.add_argument('--log-every', type=float, default=10.)
    parser.add_argument('--clear-caches', action='store_true')
    args = parser.parse_args(argv)
    if not args.out:raise ValueError('explicit output directory required')
    out = Path(args.out);out.mkdir(parents=True, exist_ok=True)
    begin = time.perf_counter()
    events = journal(args.reference_run);log('JOURNAL', events=len(events), seconds=time.perf_counter() - begin)
    context = prepare(args);log('PREPARED', seconds=context['preparation_seconds'], config_sha256=context['config_sha256'])
    import jax
    log('DEVICE', devices=[str(d) for d in jax.devices()], kind=jax.devices()[0].device_kind)
    snapshot, params, metadata = actor(args, context['plan'], events)
    log('ACTOR', archive_sha256=snapshot.archive_sha256, checkpoint_sha256=snapshot.checkpoint_sha256, parameters=metadata['parameter_count'])
    backend, roster, construction = construct(context, snapshot, events, games=args.games)
    log('ROSTER', games=roster.games, groups=[list(g) for g in roster.groups], counts=roster.counts.tolist(), **construction)
    report = dict(schema='v2a-refill-collector-benchmark.v1', config=args.config, config_sha256=context['config_sha256'],
        source_commit=os.environ.get('V2_SOURCE_COMMIT'), run_name=os.environ.get('V2_RUN_NAME'),
        device_kind=jax.devices()[0].device_kind, preparation_seconds=context['preparation_seconds'],
        actor=dict(archive_sha256=snapshot.archive_sha256, checkpoint_sha256=snapshot.checkpoint_sha256), roster=construction,
        max_requests=context['plan'].max_requests, runs=[])
    for lanes in args.lanes:
        report['runs'].append(measure(args, context, backend, roster, (params,), events, lanes))
        (out / 'report.json').write_text(json.dumps(report, indent=1, default=str) + '\n')
    log('DONE', report=str(out / 'report.json'), passed=[r['bar']['passed'] for r in report['runs']])
    return 0


if __name__ == '__main__':raise SystemExit(main())
