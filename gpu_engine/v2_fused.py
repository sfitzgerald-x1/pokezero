"""V2-b fused worker: refill collection -> in-memory rows -> GPU training -> export.

One process, one GPU, no spool/assembly/merge and no training cache files. Per
iteration: construct the declared blocks with the original constructor calls
and planner assignments, collect the whole roster with the refill collector,
finalize the row store on device, hand it to the V2 trainer by DLPack, train
the declared epochs under the original learning-rate window, then checkpoint
and export the actor for the next iteration. Historical opponents follow the
original planner (`v5_run.assignment` with the snapshots exported so far).

The declaration is either a 20k diagnostic config (`--config`, the smoke
schema) or an original arm declaration (`--declaration`/`--runtime` with their
SHA-256 and `--plan-arm`). The initial snapshot comes from `--reference-run`.
Each completed iteration is recorded in `<out>/state.json`; a restarted job
resumes from the last checkpoint with its exact optimizer, RNG and cursor.

Audit: every `--audit-every`-th global construction block's rows are written
compressed to `<out>/audit/` for offline comparison with the file pipeline.

All paths are command-line arguments.
"""
import argparse
from dataclasses import asdict
import gc
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np

from gpu_engine.v2_bench_collector import log


def journal(run):
    """Reference run journal; tolerant of an event being written by a live run."""
    events = []
    for path in sorted(Path(run, 'journal').glob('*.json')):
        try:events.append(json.loads(path.read_text()))
        except (json.JSONDecodeError, OSError):continue
    return events


def prepare(args):
    from pokezero import v5_worker as worker, v5_run as runs, v5_contract as contract, v5_device_smoke as smoke
    if args.config:
        blob = Path(args.config).read_bytes();config = json.loads(blob)
        runtime = worker._record(config['runtime']);labels = contract.LabelVocabulary(**worker._record(config['labels']))
        profile, vocabulary, gender, category = worker._runtime_metadata(runtime, expected_pool_sha256=config['bank']['pool_sha256'])
        if gender is None:raise ValueError('original gender metadata for declared pool required')
        plan = smoke.build_plan(config['bank'], profile, labels, category, arm=args.arm, iterations=args.iterations)
        inputs = worker.Inputs(plan, profile, json.dumps(runtime), json.dumps(vocabulary), json.dumps(gender), runs._sha(plan.manifest()))
        declared = dict(config_sha256=hashlib.sha256(blob).hexdigest())
    else:
        inputs = worker.load_inputs(declaration=args.declaration, expected_declaration_sha256=args.declaration_sha256,
            runtime=args.runtime, expected_runtime_sha256=args.runtime_sha256, arm=args.plan_arm, enable_v5=True)
        plan = inputs.plan;runtime = inputs.runtime;gender = inputs.gender_policy
        declared = dict(declaration_sha256=args.declaration_sha256, runtime_sha256=args.runtime_sha256, plan_arm=args.plan_arm)
    try:worker._probe(runtime)
    except ValueError as error:
        import jax
        log('PROBE_DIFFERS', error=str(error))
        jax.config.update('jax_enable_x64', False);jax.config.update('jax_default_matmul_precision', 'highest')
    begin = time.perf_counter();prepared, pool, _, _ = worker.prepare_device_runtime(inputs)
    return dict(declared=declared, plan=plan, prepared=prepared, pool=pool, gender=gender,
        preparation_seconds=time.perf_counter() - begin)


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


def history_window(plan, past):
    """The planner's eligible history (v5_run.assignment): last `historical_window`, none for A."""
    if plan.model.arm == 'A' or plan.historical_fraction == 0:return ()
    return tuple(past[-plan.historical_window:])


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
    window = history_window(plan, past)
    archives, groups = v2.canonical_groups(current.archive_sha256, [s.archive_sha256 for s in window],
        slots=v2.history_slots(len(window)))
    roster = v2.roster_from_blocks(blocks, archives=archives, groups=groups);jax.block_until_ready(roster.device)
    return roster, kinds, len(window)


def audit(store, blocks, out):
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


def training_summary(updates):
    """Mean of every numeric metric over the iteration and over its final epoch."""
    numeric = lambda rows: {key: float(np.mean([r[key] for r in rows])) for key in rows[0]
        if all(isinstance(r.get(key), (int, float)) and not isinstance(r.get(key), bool) for r in rows)}
    if not updates:return {}
    last_epoch = max(r.get('learning_rate_epoch', 0) for r in updates)
    final = [r for r in updates if r.get('learning_rate_epoch', 0) == last_epoch]
    return dict(iteration_mean=numeric(updates), final_epoch_mean=numeric(final), final_epoch=last_epoch)


def file_pipeline_order(keys, batches):
    """Permutation and segments reproducing the file pipeline's cache order.

    Each declared 2,048-game block was one cache, read in block order; inside a
    cache rows follow ``np.argwhere(trainable)`` over (chunk step, bank row,
    owner), and a live game's chunk step equals its frame. So the order is
    (block, frame, row-in-block, owner), one segment per block.
    """
    position = {}
    for block, batch in enumerate(batches):
        for row, game in enumerate(batch.games):position[game.stream_id] = (block, row)
    streams, frames, owners = keys[:, 0], keys[:, 1], keys[:, 2]
    block = np.asarray([position[int(s)][0] for s in streams], np.int64)
    row = np.asarray([position[int(s)][1] for s in streams], np.int64)
    order = np.lexsort((owners, row, frames, block))
    counts = np.bincount(block[order], minlength=len(batches))
    return order, tuple(int(c) for c in counts if c)


def write_json(path, value):
    temporary = Path(str(path) + '.tmp');temporary.write_text(json.dumps(value, indent=1, default=str) + '\n')
    os.replace(temporary, path)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', help='20k diagnostic config (smoke schema)')
    parser.add_argument('--declaration');parser.add_argument('--declaration-sha256')
    parser.add_argument('--runtime');parser.add_argument('--runtime-sha256');parser.add_argument('--plan-arm')
    parser.add_argument('--reference-run', required=True, help='run root holding the initial snapshot (journal + artifacts)')
    parser.add_argument('--out', required=True)
    parser.add_argument('--arm', default='B', help='model shape for --config')
    parser.add_argument('--iterations', type=int, default=2, help='iterations for --config (1 or 2)')
    parser.add_argument('--max-iterations', type=int, default=None, help='stop after this many declared iterations')
    parser.add_argument('--lanes', type=int, default=2048)
    parser.add_argument('--mode', default='cudagraphs')
    parser.add_argument('--order', default='ordered')
    parser.add_argument('--row-order', choices=('tick', 'file'), default='tick',
        help='tick: collection order; file: the file pipeline cache order (block, frame, row, owner)')
    parser.add_argument('--check-interval', type=int, default=64)
    parser.add_argument('--compile-threads', type=int, default=16)
    parser.add_argument('--audit-every', type=int, default=50)
    parser.add_argument('--torch-memory-fraction', type=float, default=0.45,
        help='Torch per-process GPU memory cap after the runtime probe; JAX keeps its own fraction')
    args = parser.parse_args(argv)
    if bool(args.config) == bool(args.declaration):raise SystemExit('exactly one of --config or --declaration')
    out = Path(args.out);(out / 'artifacts').mkdir(parents=True, exist_ok=True);(out / 'audit').mkdir(exist_ok=True)
    wall_start = time.perf_counter()
    import torch
    from pokezero import v5_run as runs, v2_trainer as v2t
    from gpu_engine import v5_model, v2_collector as v2c, v2_bench_trainer as bench
    from gpu_engine.v5_device_backend import RuntimeBackend
    context = prepare(args);plan = context['plan']
    if args.torch_memory_fraction is not None:
        # The runtime probe caps Torch for the file pipeline; V2 holds the row store in Torch too.
        torch.cuda.set_per_process_memory_fraction(args.torch_memory_fraction, 0)
    log('PREPARED', seconds=context['preparation_seconds'], iterations=len(plan.iterations), games=len(plan.games), **context['declared'])
    events = journal(args.reference_run)
    initial = [e['payload'] for e in events if e['kind'] == 'SNAPSHOT' and e['payload']['iteration'] == -1]
    if len(initial) != 1:raise ValueError('one initial snapshot required in the reference run')
    first = runs.Snapshot(**initial[0])
    if first.model_sha256 != runs._sha(plan.model.metadata()):raise ValueError('initial snapshot model differs from plan')
    roots = {first.archive_sha256: Path(args.reference_run)}
    v2_config = v2t.V2Config(mode=args.mode, order=args.order, optimizer='fused', check_interval=args.check_interval)
    state_path = out / 'state.json'
    if state_path.exists():
        saved = json.loads(state_path.read_text())
        snapshots = [runs.Snapshot(**x) for x in saved['snapshots']];reports = saved['reports']
        for snapshot in snapshots[1:]:roots[snapshot.archive_sha256] = out
        checkpoint = out / snapshots[-1].checkpoint
        runtime = v2t.load_checkpoint(checkpoint, enable_v5=True, v2_config=v2_config, expected_model_config=plan.model,
            expected_loss_config=plan.loss, expected_train_config=plan.train, device='cuda')
        log('RESUMED', completed_iterations=len(reports), checkpoint=str(checkpoint), steps=runtime.steps)
    else:
        snapshots = [first];reports = []
        model, loss, train, weights, _ = bench.configurations(Path(args.reference_run, first.checkpoint))
        if model != plan.model or loss != plan.loss or train != plan.train:raise ValueError('initial checkpoint recipe differs from plan')
        runtime = v2t.create(plan.model, plan.loss, plan.train, v2_config, enable_v5=True, device='cuda')
        runtime.model.load_state_dict(weights)
    trainer = v2t.Trainer(runtime)
    params = {}
    def actor_params(digest):
        if digest not in params:
            snapshot = next(s for s in snapshots if s.archive_sha256 == digest)
            params[digest] = load_actor(plan, snapshot, roots[digest])
        return params[digest]
    backend = RuntimeBackend(context['prepared'], context['pool'], gender_policy=context['gender']);backend.check_plan(plan)
    ordinal = {g.stream_id: i for i, g in enumerate(plan.games)}
    collectors = {}
    reference_ends = {e['payload']['stream_id']: e['payload'] for e in events if e['kind'] == 'GAME_END'}
    last = len(plan.iterations) if args.max_iterations is None else min(len(plan.iterations), args.max_iterations)
    for iteration in range(len(reports), last):
        phase = {};clock = time.perf_counter();start = clock
        def mark(name):
            nonlocal clock
            now = time.perf_counter();phase[name] = now - clock;clock = now
        current = snapshots[-1];past = tuple(s for s in snapshots if s.iteration < current.iteration)
        roster, kinds, history = construct(context, backend, iteration, current, past, ordinal);mark('construct')
        current_params = actor_params(current.archive_sha256)
        tree = tuple(current_params if digest.startswith(v2c.PAD_PREFIX) else actor_params(digest) for digest in roster.actor_archives)
        key = (len(roster.groups), roster.games);compile_timings = None
        if key in collectors:
            collector = collectors[key].rebind(roster);reused = True
        else:
            collector = v2c.RefillCollector(context['prepared'].engine_tables, context['prepared'].observation_tables, plan.model,
                backend.labels, roster, lanes=args.lanes, max_requests=plan.max_requests,
                layout_counts=v2c.expected_counts(roster.groups, roster.games, plan.historical_fraction))
            compile_timings = collector.compile(tree, threads=args.compile_threads);reused = False
            collectors.clear();collectors[key] = collector  # history only grows: older layouts never recur
        mark('collector_compile')
        result = collector.run(tree);mark('collect')
        rows = v2c.finalize(result, loss_config=plan.loss, discount=plan.discount);mark('finalize')
        base = sum(len(plan.iterations[i]) for i in range(iteration));audited = []
        for k, batch in enumerate(plan.iterations[iteration]):
            if (base + k) % args.audit_every == 0:
                ids = [g.stream_id for g in batch.games];audited.append((base + k, min(ids), max(ids)))
        audit_records = audit(rows, audited, out / 'audit');mark('audit')
        records = v2c.game_records(result);same = None
        if reference_ends:
            from gpu_engine.v5_device_rollout import STATUS
            same = dict(compared=0, equal=0)
            for i, stream in enumerate(records['streams']):
                ref = reference_ends.get(int(stream))
                if ref is None:continue
                status = STATUS[int(records['status'][i])];same['compared'] += 1
                if ref['status'] == status and ref['terminal_frame'] == int(records['terminal_frame'][i]) and (
                        status != 'COMPLETE' or ref['terminal_winner'] == int(records['winner'][i])):same['equal'] += 1
        segments = None
        if args.row_order == 'file':
            # Permute in JAX memory, one field at a time, before the zero-copy hand-off.
            import jax.numpy as jnp
            order, segments = file_pipeline_order(rows.keys(), plan.iterations[iteration])
            index = jnp.asarray(order)
            for name in list(rows.arrays):
                # Gather straight from the full buffer (indices < n): no slice copy; free each buffer before the next.
                value = rows.arrays[name];rows.arrays[name] = None
                rows.arrays[name] = jnp.take(value, index, axis=0).block_until_ready();del value
            del index
        tensors = rows.to_torch(device='cuda', exact_dtypes=True)
        store = v2t.row_store(tensors, plan.model, plan.loss, segments=segments, store_id=f'v2-iteration-{iteration}');mark('handoff_validate')
        train_report = trainer.train(store, window=runs.learning_rate_window(plan, iteration))
        torch.cuda.synchronize();mark('train')
        del store, tensors;gc.collect()
        checkpoint = out / 'artifacts' / f'train-{iteration}.pt';v2t.save_checkpoint(checkpoint, runtime)
        archive = out / 'artifacts' / f'actor-{iteration}.npz';checkpoint_sha = sha(checkpoint)
        metadata = v5_model.export(checkpoint, archive, expected_checkpoint_sha256=checkpoint_sha, enable_v5=True)
        snapshot = runs.Snapshot(iteration=iteration, checkpoint=f'artifacts/train-{iteration}.pt', checkpoint_sha256=checkpoint_sha,
            archive=f'artifacts/actor-{iteration}.npz', archive_sha256=metadata['weights_archive_sha256'],
            model_sha256=runs._sha(plan.model.metadata()))
        roots[snapshot.archive_sha256] = out;snapshots.append(snapshot);actor_params(snapshot.archive_sha256)
        mark('checkpoint_export_reload')
        games = roster.games;total = time.perf_counter() - start;steady = total - phase['collector_compile'] - phase['audit']
        stats = result.stats;updates = train_report['updates']
        report = dict(iteration=iteration, games=games, rows=rows.n, dispositions=rows.stats['dispositions'], opponents=kinds,
            history=history, history_slots=v2c.history_slots(history), lane_groups=len(roster.groups), collector_reused=reused,
            compile_timings=compile_timings, phase_seconds=phase, iteration_seconds=total, steady_seconds=steady,
            games_per_second=games / total, games_per_second_steady=games / steady,
            training=dict(updates=len(updates), steps=runtime.steps, rows=runtime.rows, mode=trainer.mode_report(),
                **training_summary(updates)),
            collector={k: v for k, v in stats.items() if isinstance(v, (int, float, str))},
            audit=audit_records, games_matching_reference=same, snapshot=asdict(snapshot))
        reports.append(report)
        log('ITERATION', iteration=iteration, games_per_second=round(report['games_per_second'], 2),
            games_per_second_steady=round(report['games_per_second_steady'], 2), phase_seconds={k: round(v, 1) for k, v in phase.items()},
            history=history, reused=reused, occupancy=round(stats.get('occupancy', 0), 3), dispositions=report['dispositions'],
            final_epoch_loss=report['training'].get('final_epoch_mean', {}).get('loss'), matching=same)
        write_json(state_path, dict(snapshots=[asdict(s) for s in snapshots], reports=reports))
        del rows, result, roster, tree;gc.collect()
        keep = {s.archive_sha256 for s in snapshots[-1 - plan.historical_window:]}
        for digest in list(params):
            if digest not in keep:del params[digest]
    total = time.perf_counter() - wall_start
    games = sum(r['games'] for r in reports)
    summary = dict(schema='v2-fused-measurement.v2', declared=context['declared'], lanes=args.lanes, mode=args.mode,
        order=args.order, row_order=args.row_order, iterations=reports, preparation_seconds=context['preparation_seconds'], wall_seconds_this_attempt=total,
        games=games, games_per_second_steady_mean=games / max(1e-9, sum(r['steady_seconds'] for r in reports)),
        scope='Throughput of the fused V2 pipeline; not a strength result.')
    write_json(out / 'report.json', summary)
    log('DONE', wall_seconds=total, iterations=len(reports), games=games)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
