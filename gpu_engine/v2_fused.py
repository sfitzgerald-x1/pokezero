"""V2-b fused worker: refill collection -> in-memory rows -> GPU training -> export.

One process, one GPU, no spool/assembly/merge and no training cache files. Per
iteration: construct the declared blocks with the original constructor calls
and planner assignments, collect the whole roster with the refill collector,
finalize the row store on device, hand it to the V2 trainer by DLPack, train
the declared epochs under the original learning-rate window, then checkpoint
and export the actor for the next iteration. Historical opponents follow the
original planner (`v5_run.assignment` with the snapshots exported so far).

Audit: every `--audit-every`-th global construction block's rows are written
compressed to `<out>/audit/` for offline comparison with the file pipeline.

All paths are command-line arguments.
"""
import argparse
import gc
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from gpu_engine.v2_bench_collector import journal, log


def prepare(config_path, *, arm, iterations):
    from pokezero import v5_worker as worker, v5_run as runs, v5_contract as contract, v5_device_smoke as smoke
    blob = Path(config_path).read_bytes();config = json.loads(blob)
    runtime = worker._record(config['runtime']);labels = contract.LabelVocabulary(**worker._record(config['labels']))
    profile, vocabulary, gender, category = worker._runtime_metadata(runtime, expected_pool_sha256=config['bank']['pool_sha256'])
    if gender is None:raise ValueError('original gender metadata for declared pool required')
    plan = smoke.build_plan(config['bank'], profile, labels, category, arm=arm, iterations=iterations)
    try:worker._probe(runtime)
    except ValueError as error:
        import jax
        log('PROBE_DIFFERS', error=str(error))
        jax.config.update('jax_enable_x64', False);jax.config.update('jax_default_matmul_precision', 'highest')
    inputs = worker.Inputs(plan, profile, json.dumps(runtime), json.dumps(vocabulary), json.dumps(gender), runs._sha(plan.manifest()))
    begin = time.perf_counter();prepared, pool, _, _ = worker.prepare_device_runtime(inputs)
    return dict(config=config, config_sha256=hashlib.sha256(blob).hexdigest(), plan=plan, prepared=prepared, pool=pool,
        gender=gender, preparation_seconds=time.perf_counter() - begin)


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for part in iter(lambda: f.read(1 << 20), b''):h.update(part)
    return h.hexdigest()


def load_actor(plan, snapshot, root):
    from gpu_engine import v5_model
    params, config, _ = v5_model.load(Path(root, snapshot.archive), expected_archive_sha256=snapshot.archive_sha256,
        expected_checkpoint_sha256=snapshot.checkpoint_sha256, expected_category_vocabulary_sha256=plan.model.category_vocabulary_sha256,
        expected_label_vocabulary_sha256=plan.model.label_vocabulary.sha256, enable_v5=True)
    if config != plan.model:raise ValueError('loaded actor configuration differs from plan')
    return params


def construct(context, backend, iteration, current, past, ordinal):
    """Original per-block constructor calls and planner assignments for one iteration."""
    import jax
    from pokezero import v5_run as runs
    from gpu_engine import v5_device_rollout as rollout, v2_collector as v2
    plan = context['plan'];prepared = context['prepared'];blocks = [];kinds = dict(SELFPLAY=0, HISTORICAL=0)
    for batch in plan.iterations[iteration]:
        assignments = tuple(runs.assignment(plan, g, ordinal[g.stream_id], current, past) for g in batch.games)
        recorded = json.loads(runs._json(assignments))
        for a in recorded:kinds[a['opponent_kind']] += 1
        required = tuple(dict.fromkeys([current.archive_sha256, *(a['opponent']['archive_sha256'] for a in recorded)]))
        bindings = rollout.Bindings(plan.model.category_vocabulary_sha256, plan.model.label_vocabulary, prepared.source_sha256,
            (current.archive_sha256,) * 2, (True, True), backend.declaration_sha256)
        bank = rollout.actor_bank_from_assignments(bindings, required, current_archive=current.archive_sha256, assignments=recorded)
        blocks.append((jax.block_until_ready(backend._construct(batch)), bank, batch.sampling_seed))
    roster = v2.roster_from_blocks(blocks);jax.block_until_ready(roster.device)
    return roster, kinds


def audit(store, roster, blocks, out):
    """Compressed host copy of the audited blocks' rows (keys, targets and inputs)."""
    import jax
    import jax.numpy as j
    streams = np.asarray(jax.device_get(store.arrays['streams'][:store.n]))
    written = []
    for index, lo, hi in blocks:
        rows = np.flatnonzero((streams >= lo) & (streams <= hi))
        take = j.asarray(rows)
        arrays = {name: np.asarray(jax.device_get(j.take(value[:store.n], take, axis=0))) for name, value in store.arrays.items()}
        path = Path(out, f'audit-block-{index:06d}.npz');np.savez_compressed(path, **arrays)
        written.append(dict(block=index, streams=[int(lo), int(hi)], rows=int(len(rows)), path=str(path), bytes=path.stat().st_size))
    return written


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True, help='declared plan config (the measured run config.json)')
    parser.add_argument('--reference-run', required=True, help='run root holding the initial snapshot (journal + artifacts)')
    parser.add_argument('--out', required=True)
    parser.add_argument('--arm', default='B')
    parser.add_argument('--iterations', type=int, default=2)
    parser.add_argument('--lanes', type=int, default=2048)
    parser.add_argument('--mode', default='cudagraphs')
    parser.add_argument('--order', default='ordered')
    parser.add_argument('--check-interval', type=int, default=64)
    parser.add_argument('--compile-threads', type=int, default=16)
    parser.add_argument('--audit-every', type=int, default=50)
    args = parser.parse_args(argv)
    out = Path(args.out);(out / 'artifacts').mkdir(parents=True, exist_ok=True);(out / 'audit').mkdir(exist_ok=True)
    wall_start = time.perf_counter()
    import jax
    import torch
    from pokezero import v5_run as runs, v2_trainer as v2t
    from gpu_engine import v5_model, v2_collector as v2c, v2_bench_trainer as bench
    from gpu_engine.v5_device_backend import RuntimeBackend
    context = prepare(args.config, arm=args.arm, iterations=args.iterations);plan = context['plan']
    log('PREPARED', seconds=context['preparation_seconds'], iterations=len(plan.iterations), games=len(plan.games))
    events = journal(args.reference_run)
    initial = [e['payload'] for e in events if e['kind'] == 'SNAPSHOT' and e['payload']['iteration'] == -1]
    if len(initial) != 1:raise ValueError('one initial snapshot required in the reference run')
    snapshots = [runs.Snapshot(**initial[0])];roots = {snapshots[0].archive_sha256: Path(args.reference_run)}
    if snapshots[0].model_sha256 != runs._sha(plan.model.metadata()):raise ValueError('initial snapshot model differs from plan')
    params = {snapshots[0].archive_sha256: load_actor(plan, snapshots[0], args.reference_run)}
    model, loss, train, state, _ = bench.configurations(Path(args.reference_run, snapshots[0].checkpoint))
    if model != plan.model or loss != plan.loss or train != plan.train:raise ValueError('initial checkpoint recipe differs from plan')
    runtime = v2t.create(plan.model, plan.loss, plan.train, v2t.V2Config(mode=args.mode, order=args.order, optimizer='fused',
        check_interval=args.check_interval), enable_v5=True, device='cuda')
    runtime.model.load_state_dict(state);trainer = v2t.Trainer(runtime)
    backend = RuntimeBackend(context['prepared'], context['pool'], gender_policy=context['gender']);backend.check_plan(plan)
    ordinal = {g.stream_id: i for i, g in enumerate(plan.games)}
    block_base = 0;reports = []
    reference_ends = {e['payload']['stream_id']: e['payload'] for e in events if e['kind'] == 'GAME_END'}
    for iteration in range(len(plan.iterations)):
        phase = {};clock = time.perf_counter();start = clock
        def mark(name):
            nonlocal clock
            now = time.perf_counter();phase[name] = now - clock;clock = now
        current = snapshots[-1];past = tuple(s for s in snapshots if s.iteration < current.iteration)
        roster, kinds = construct(context, backend, iteration, current, past, ordinal);mark('construct')
        collector = v2c.RefillCollector(context['prepared'].engine_tables, context['prepared'].observation_tables, plan.model,
            backend.labels, roster, lanes=args.lanes, max_requests=plan.max_requests)
        actor_params = tuple(params[digest] for digest in roster.actor_archives)
        collector.compile(actor_params, threads=args.compile_threads);mark('collector_compile')
        result = collector.run(actor_params);mark('collect')
        rows = v2c.finalize(result, loss_config=plan.loss, discount=plan.discount);mark('finalize')
        audited = []
        for k, batch in enumerate(plan.iterations[iteration]):
            index = block_base + k
            if index % args.audit_every == 0:
                ids = [g.stream_id for g in batch.games];audited.append((index, min(ids), max(ids)))
        audit_records = audit(rows, roster, audited, out / 'audit');mark('audit')
        records = v2c.game_records(result)
        same = None
        if iteration == 0 and reference_ends:
            from gpu_engine.v5_device_rollout import STATUS
            same = 0
            for i, stream in enumerate(records['streams']):
                ref = reference_ends.get(int(stream))
                status = STATUS[int(records['status'][i])]
                if ref and ref['status'] == status and ref['terminal_frame'] == int(records['terminal_frame'][i]) and (
                        status != 'COMPLETE' or ref['terminal_winner'] == int(records['winner'][i])):same += 1
        tensors = rows.to_torch(device='cuda', exact_dtypes=True)
        store = v2t.row_store(tensors, plan.model, plan.loss, store_id=f'v2-iteration-{iteration}');mark('handoff_validate')
        train_report = trainer.train(store, window=runs.learning_rate_window(plan, iteration))
        torch.cuda.synchronize();mark('train')
        del store, tensors;gc.collect()
        checkpoint = out / 'artifacts' / f'train-{iteration}.pt';v2t.save_checkpoint(checkpoint, runtime)
        archive = out / 'artifacts' / f'actor-{iteration}.npz'
        checkpoint_sha = sha(checkpoint)
        metadata = v5_model.export(checkpoint, archive, expected_checkpoint_sha256=checkpoint_sha, enable_v5=True)
        snapshot = runs.Snapshot(iteration=iteration, checkpoint=f'artifacts/train-{iteration}.pt', checkpoint_sha256=checkpoint_sha,
            archive=f'artifacts/actor-{iteration}.npz', archive_sha256=metadata['weights_archive_sha256'],
            model_sha256=runs._sha(plan.model.metadata()))
        params[snapshot.archive_sha256] = load_actor(plan, snapshot, out);snapshots.append(snapshot);mark('checkpoint_export_reload')
        stats = result.stats if hasattr(result, 'stats') else {}
        games = roster.games;total = time.perf_counter() - start
        updates = train_report['updates']
        report = dict(iteration=iteration, games=games, rows=rows.n, dispositions=rows.stats['dispositions'], opponents=kinds,
            actor_archives=len(roster.actor_archives), lane_groups=len(roster.groups), phase_seconds=phase, iteration_seconds=total,
            games_per_second=games / total,
            games_per_second_excluding_compile=games / (total - phase['collector_compile']),
            training=dict(updates=len(updates), steps=runtime.steps, rows=runtime.rows, mode=trainer.mode_report(),
                last_loss=updates[-1].get('loss') if updates else None),
            collector=dict(levels=[x.lanes for x in collector.levels], **{k: v for k, v in stats.items() if isinstance(v, (int, float, str))}),
            audit=audit_records, games_matching_reference=same, snapshot=dict(archive_sha256=snapshot.archive_sha256))
        reports.append(report);log('ITERATION', **{k: v for k, v in report.items() if k != 'audit'})
        block_base += len(plan.iterations[iteration])
        del rows, result, collector, roster, actor_params;gc.collect()
        for digest in list(params):
            if digest not in {s.archive_sha256 for s in snapshots[-1 - plan.historical_window:]}:del params[digest]
    total = time.perf_counter() - wall_start
    summary = dict(schema='v2-fused-measurement.v1', config_sha256=context['config_sha256'], lanes=args.lanes, mode=args.mode,
        order=args.order, iterations=reports, preparation_seconds=context['preparation_seconds'], wall_seconds=total,
        games=sum(r['games'] for r in reports), games_per_second_wall=sum(r['games'] for r in reports) / total,
        scope='Throughput of the fused V2 pipeline; not a strength result.')
    (out / 'report.json').write_text(json.dumps(summary, indent=1, default=str) + '\n')
    log('DONE', wall_seconds=total, games_per_second_wall=summary['games_per_second_wall'])
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
