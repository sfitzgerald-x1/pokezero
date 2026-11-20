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
from dataclasses import asdict, replace
import gc
import hashlib
import json
import os
from pathlib import Path
import time
import uuid

import numpy as np
from pokezero.v2_lr_start import LRStart, SCALING_RULES, scaled_peak_rate

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
        # V2 self_moves: the config's optional declaration (absent = off); Plan refuses a non-bool value.
        plan = smoke.build_plan(config['bank'], profile, labels, category, arm=args.arm, iterations=args.iterations,
            self_moves=config.get('self_moves', False))
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
    params, config, meta = v5_model.load(Path(root, snapshot.archive), expected_archive_sha256=snapshot.archive_sha256,
        expected_checkpoint_sha256=snapshot.checkpoint_sha256, expected_category_vocabulary_sha256=plan.model.category_vocabulary_sha256,
        expected_label_vocabulary_sha256=plan.model.label_vocabulary.sha256, enable_v5=True)
    if config != plan.model:raise ValueError('loaded actor configuration differs from plan')
    from gpu_engine.private_self_moves import check_run_actor,checked_actor_initializer
    initializer=checked_actor_initializer(meta,plan_self_moves=plan.self_moves,snapshot=snapshot,root=root)
    check_run_actor(meta, plan_self_moves=plan.self_moves, iteration=snapshot.iteration, context='v2 fused actor',
        initializer=initializer)
    return params


def history_window(plan, current, past):
    """The same finite eligible bank used by the declared assignment."""
    from pokezero.v5_run import opponent_history
    if plan.model.arm == 'A' or plan.historical_fraction == 0:return ()
    return opponent_history(plan,current,past)


def construct(context, backend, iteration, current, past, ordinal, *, shard=None):
    """Original per-block constructor calls and planner assignments for one iteration."""
    import jax
    from pokezero import v5_run as runs
    from gpu_engine import v5_device_rollout as rollout, v2_collector as v2
    plan = context['plan'];prepared = context['prepared'];blocks = [];kinds = dict(SELFPLAY=0, HISTORICAL=0)
    window = history_window(plan, current, past)
    offset = 0
    for batch in plan.iterations[iteration]:
        begin, end = (0, len(batch.games)) if shard is None else (
            max(0, shard[0] - offset), min(len(batch.games), shard[1] - offset))
        offset += len(batch.games)
        if end <= begin:
            continue
        assignments = tuple(runs.assignment(plan, g, ordinal[g.stream_id], current, past, history=window) for g in batch.games)
        recorded = json.loads(runs._json(assignments[begin:end]))
        for a in recorded:kinds[a['opponent_kind']] += 1
        required = tuple(dict.fromkeys([current.archive_sha256, *(a['opponent']['archive_sha256'] for a in recorded)]))
        bindings = rollout.Bindings(plan.model.category_vocabulary_sha256, plan.model.label_vocabulary, prepared.source_sha256,
            (current.archive_sha256,) * 2, (True, True), backend.declaration_sha256)
        bank = rollout.actor_bank_from_assignments(bindings, required, current_archive=current.archive_sha256, assignments=recorded)
        # Construct the original full block BEFORE slicing. Reconstructing a
        # smaller batch would change shape/index-dependent constructor RNG.
        carry = jax.block_until_ready(backend._construct(batch))
        if begin or end != len(batch.games):
            carry = jax.tree.map(lambda value: value[begin:end], carry)
        blocks.append((carry, bank, batch.sampling_seed))
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


def write_json(path, value, *, durable=False):
    temporary = Path(str(path) + '.tmp')
    with temporary.open('w') as stream:
        stream.write(json.dumps(value, indent=1, default=str) + '\n')
        if durable:
            stream.flush()
            os.fsync(stream.fileno())
    os.replace(temporary, path)
    if durable:
        descriptor = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)


def select_learner_precision(context, precision):
    """Record an explicit learner override without changing the initial binding."""
    plan = context['plan']
    if precision is None:
        return plan
    from pokezero import v5_run as runs
    plan = replace(plan, train=replace(plan.train, precision=precision))
    context['plan'] = plan
    context['declared'] = dict(context['declared'], learner_precision=precision,
        effective_plan_sha256=runs._sha(plan.manifest()))
    return plan


def select_learner_lr_start(context, lr_start):
    """Override the effective learner recipe after binding the original initializer."""
    plan = context['plan']
    if lr_start is None:
        return plan
    from pokezero import v5_run as runs
    peak = scaled_peak_rate(plan.train.batch_size, lr_start.scaling)
    plan = replace(plan, train=replace(plan.train, learning_rate=peak, learning_rate_schedule='constant'))
    context['plan'] = plan
    context['declared'] = dict(context['declared'], lr_start=asdict(lr_start),
        effective_plan_sha256=runs._sha(plan.manifest()))
    return plan


def _validate_lr_start_reports(runtime, reports, plan, policy):
    """Bind the fused round ledger and declared games to the loaded checkpoint."""
    if not reports or len(reports) > len(plan.iterations):
        raise ValueError('resuming LR start requires its committed fused rounds')
    start_update = start_row = start_game = 0
    for iteration, report in enumerate(reports):
        games = sum(len(batch.games) for batch in plan.iterations[iteration])
        end_game = start_game + games
        window = dict(start_game=start_game, end_game=end_game, total_games=len(plan.games))
        training = report['training']
        start = training.get('lr_start')
        steps, rows = training['steps'], training['rows']
        expected = dict(policy=policy, declared_window=window, start_update=start_update,
            end_update=steps, start_row=start_row, end_row=rows)
        if (start != expected or report.get('iteration') != iteration or report.get('games') != games
                or type(steps) is not int or type(rows) is not int or steps <= start_update
                or rows <= start_row or training['updates'] != steps - start_update):
            raise ValueError('LR-start report progress differs from the declaration')
        origin = (start_update, start_row)
        start_update, start_row, start_game = steps, rows, end_game
    if (runtime.cursor is None or not runtime.cursor['complete'] or runtime.window is None
            or asdict(runtime.window) != window or runtime.lr_start_origin != origin
            or (runtime.steps, runtime.rows) != (start_update, start_row)):
        raise ValueError('LR-start round ledger differs from the committed checkpoint')


def argument_parser():
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
    parser.add_argument('--native-embedding-backward', action='store_true',
        help='opt in to native category-embedding backward; must match on resume')
    parser.add_argument('--fallback-random', action='store_true',
        help='opt in to compiler native random/dropout (compiled modes); must match on resume')
    parser.add_argument('--learner-precision', choices=('float32', 'bfloat16'), default=None,
        help='explicit learner precision override; recorded in the effective plan and must match on resume')
    parser.add_argument('--lr-start', choices=SCALING_RULES,
        help='opt in to global-batch peak LR scaling and optional update warmup')
    parser.add_argument('--lr-warmup-updates', type=int,
        help='number of global optimizer updates to reach peak LR (default 0 with --lr-start)')
    parser.add_argument('--order', default='ordered')
    parser.add_argument('--row-order', choices=('tick', 'file'), default=None,
        help='tick: collection order; file (default for new runs): the file pipeline cache order (block, frame, row, owner). '
             'A resumed run keeps the order recorded in its state.json')
    parser.add_argument('--check-interval', type=int, default=64)
    parser.add_argument('--recipe', choices=('R5', 'R2', 'R1'), default=None,
        help='explicit batch/epochs/LR package from the validated R5 reference; records the effective plan hash')
    parser.add_argument('--compile-threads', type=int, default=16)
    parser.add_argument('--audit-every', type=int, default=50)
    parser.add_argument('--torch-memory-fraction', type=float, default=0.45,
        help='Torch per-process GPU memory cap after the runtime probe (V2 also holds the row store in Torch); '
             '0 keeps the runtime declaration\'s torch_fraction. JAX keeps its own fraction')
    return parser


def _rank_zero(distribution, phase, action):
    """Complete an owned phase and share its outcome before peers advance."""
    if distribution is None:
        return action()
    result = error = None
    if distribution.rank == 0:
        try:
            result = action()
        except Exception as exc:
            error = exc
    result, failure = distribution.broadcast((result, None if error is None else (type(error).__name__, str(error))))
    if failure is not None:
        kind, message = failure
        error_type = ValueError if kind == 'ValueError' else RuntimeError
        raise error_type(f'{phase} failed on rank 0: {kind}: {message}') from error
    return result


def _validated_on_all_ranks(distribution, action):
    if distribution is None:
        return action()
    error = None
    try:
        result = action()
    except (Exception, SystemExit) as exc:
        error = exc
    failures = distribution.gather(None if error is None else (type(error).__name__, str(error)))
    for rank, failure in enumerate(failures):
        if failure is not None:
            kind, message = failure
            error_type = SystemExit if kind == 'SystemExit' else ValueError
            raise error_type(f'resume inputs failed on rank {rank}: {message}') from error
    return result

def main(argv=None, *, distribution=None):
    parser = argument_parser()
    args = parser.parse_args(argv)
    if args.lr_start is None and args.lr_warmup_updates is not None:
        parser.error('--lr-warmup-updates requires --lr-start')
    try:
        lr_start = None if args.lr_start is None else LRStart(args.lr_start, args.lr_warmup_updates or 0)
    except ValueError as error:
        parser.error(str(error))
    if bool(args.config) == bool(args.declaration):raise SystemExit('exactly one of --config or --declaration')
    out = Path(args.out);(out / 'artifacts').mkdir(parents=True, exist_ok=True);(out / 'audit').mkdir(exist_ok=True)
    wall_start = time.perf_counter()
    import torch
    from pokezero import v5_run as runs, v2_trainer as v2t
    from gpu_engine import v5_model, v2_collector as v2c, v2_bench_trainer as bench
    from pokezero import checkpoint_self_moves as encoding
    from gpu_engine.v5_device_backend import RuntimeBackend
    context = prepare(args);plan = context['plan']
    initial_train = plan.train
    if args.recipe is not None:
        from pokezero.v2_distributed import recipe_train
        plan = replace(plan, train=recipe_train(initial_train, args.recipe));context['plan'] = plan
        context['declared'] = dict(context['declared'], recipe=args.recipe,
            effective_plan_sha256=runs._sha(plan.manifest()))
    plan = select_learner_precision(context, args.learner_precision)
    plan = select_learner_lr_start(context, lr_start)
    lr_start_policy = context['declared'].get('lr_start') if lr_start is not None else None
    if distribution is not None:
        distribution.bind(context, args)
    declaration_sha256 = runs._sha(plan.manifest())
    deferred = plan.evaluation is not None and plan.evaluation.deferred
    dispatch = None
    state_path = out / 'state.json'
    if deferred:
        from pokezero.v5_deferred_evaluation import MilestoneDispatch, retain_full_checkpoints, sync_actor, sync_directory

        def setup_dispatch():
            nonlocal dispatch
            declaration = out / 'declaration.json'
            if state_path.exists() and not declaration.is_file():
                raise ValueError('original fused evaluation declaration is missing')
            if declaration.exists():
                if runs._sha(json.loads(declaration.read_bytes())) != declaration_sha256:
                    raise ValueError('original fused evaluation declaration differs')
            else:
                runs._publish(declaration, runs._json(plan.manifest()).encode())
                sync_directory(out)
            dispatch = MilestoneDispatch(out, plan.evaluation, plan.evaluation_milestones(), declaration_sha256,
                report_observer_error=lambda reason: log('EVALUATION_STOP_READ_ERROR', reason=reason))

        _rank_zero(distribution, 'evaluation declaration', setup_dispatch)

    if args.torch_memory_fraction > 0:
        # The runtime probe caps Torch for the file pipeline; V2 holds the row store in Torch too.
        torch.cuda.set_per_process_memory_fraction(args.torch_memory_fraction, 0)
    log('PREPARED', seconds=context['preparation_seconds'], iterations=len(plan.iterations), games=len(plan.games), **context['declared'])
    _, _, events = encoding.verified_run(args.reference_run)
    initial = [e['payload'] for e in events if e['kind'] == 'SNAPSHOT' and e['payload']['iteration'] == -1]
    if len(initial) != 1:raise ValueError('one initial snapshot required in the reference run')
    first = runs.Snapshot(**initial[0])
    if first.model_sha256 != runs._sha(plan.model.metadata()):raise ValueError('initial snapshot model differs from plan')
    roots = {first.archive_sha256: Path(args.reference_run)}
    v2_config = v2t.V2Config(mode=args.mode, order=args.order, optimizer='fused', check_interval=args.check_interval,
        distributed_rows=distribution is not None, native_embedding_backward=args.native_embedding_backward,
        fallback_random=args.fallback_random, lr_start=lr_start)
    placement = {} if distribution is None else dict(process_group=distribution.process_group)
    def read_resume_manifest():
        try:
            saved = json.loads(state_path.read_bytes())
        except FileNotFoundError:
            return None
        if type(saved) is not dict:
            raise ValueError('fused resume manifest must be a JSON object')
        return saved

    saved = _rank_zero(distribution, 'resume manifest', read_resume_manifest)
    if saved is not None:
        def resume_inputs():
            if saved.get('declaration_sha256') != declaration_sha256:
                raise ValueError('original fused run declaration binding required')
            if saved.get('lr_start') != lr_start_policy:
                raise ValueError('resume requires the original fused LR-start declaration')
            if distribution is not None:
                distribution.check_resume(saved)
            elif 'distribution' in saved:
                raise ValueError('distributed fused run requires its original placement on resume')
            recorded = saved.get('row_order')
            if recorded is None and args.row_order is None:
                raise SystemExit('state.json does not record its row order (older runs defaulted to tick); pass --row-order explicitly')
            if recorded is not None and args.row_order not in (None, recorded):
                raise SystemExit(f'run was recorded with --row-order {recorded}; a resume cannot switch to {args.row_order}')
            receipts = saved.get('checkpoint_encoding_receipts', {})
            restored = [runs.Snapshot(**x) for x in saved['snapshots']]
            digest = restored[-1].checkpoint_sha256
            proof = None
            if not plan.self_moves and digest in receipts:
                proof = encoding.off_from_receipt(receipts[digest])
            return restored, receipts, recorded or args.row_order, proof

        snapshots, encoding_receipts, row_order, off_provenance = _validated_on_all_ranks(distribution, resume_inputs)
        reports = saved['reports']
        for snapshot in snapshots[1:]:roots[snapshot.archive_sha256] = out
        checkpoint = out / snapshots[-1].checkpoint
        runtime = v2t.load_checkpoint(checkpoint, enable_v5=True, v2_config=v2_config, expected_model_config=plan.model,
            expected_loss_config=plan.loss, expected_train_config=plan.train, device='cuda',
            expected_self_moves=plan.self_moves, expected_checkpoint_sha256=snapshots[-1].checkpoint_sha256,
            off_provenance=off_provenance, **placement)
        if lr_start is not None:
            _validated_on_all_ranks(distribution, lambda: _validate_lr_start_reports(
                runtime, reports, plan, lr_start_policy))
        log('RESUMED', completed_iterations=len(reports), checkpoint=str(checkpoint), steps=runtime.steps, row_order=row_order)
    else:
        snapshots = [first];reports = [];row_order = args.row_order or 'file'
        encoding_receipts = {}
        initializer, proof = encoding.original_initializer(args.reference_run, first)
        model, loss, train, weights, _ = bench.configurations(Path(args.reference_run, first.checkpoint),
            expected_checkpoint_sha256=first.checkpoint_sha256,expected_self_moves=plan.self_moves,
            off_provenance=proof,weights_only_initializer=initializer)
        if model != plan.model or loss != plan.loss or train != initial_train:raise ValueError('initial checkpoint recipe differs from plan')
        runtime = v2t.create(plan.model, plan.loss, plan.train, v2_config, enable_v5=True, device='cuda',
            self_moves=plan.self_moves, **placement)
        runtime.model.load_state_dict(weights)
    if distribution is None:
        trainer = v2t.Trainer(runtime)
    else:
        from pokezero.v2_distributed import DistributedTrainer
        trainer = DistributedTrainer(runtime)
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
    def publish_evaluation(snapshot):
        dispatch.publish_committed(snapshot, report_error=lambda reason: log(
            'EVALUATION_REQUEST_ERROR', iteration=snapshot.iteration, reason=reason))

    if deferred and saved is not None:
        def recover_evaluations():
            for snapshot in snapshots[1:]:
                if snapshot.iteration in dispatch.milestones and dispatch.request_missing(snapshot):
                    publish_evaluation(snapshot)
            retain_full_checkpoints(out, [
                dict(iteration=item.iteration, path=item.checkpoint) for item in snapshots[1:]
            ], dispatch.milestones)

        _rank_zero(distribution, 'evaluation recovery and retention', recover_evaluations)
    route_stop = None
    for iteration in range(len(reports), last):
        if deferred:
            route_stop = _rank_zero(distribution, 'evaluation stop', lambda: dispatch.stop_at_boundary(iteration))
            if route_stop is not None:
                if distribution is None or distribution.rank == 0:
                    log('STOP', iteration=iteration, reason='S0_TRANSFER_FAIL', stop_request=route_stop)
                break
        phase = {};clock = time.perf_counter();start = clock
        def mark(name):
            nonlocal clock
            now = time.perf_counter();phase[name] = now - clock;clock = now
        current = snapshots[-1];past = tuple(s for s in snapshots if s.iteration < current.iteration)
        shard = {} if distribution is None else dict(shard=distribution.shard(plan, iteration))
        roster, kinds, history = construct(context, backend, iteration, current, past, ordinal, **shard);mark('construct')
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
        audit_out = out / 'audit' if distribution is None else out / 'audit' / f'rank-{distribution.rank}'
        audit_out.mkdir(parents=True, exist_ok=True)
        audit_records = audit(rows, audited, audit_out);mark('audit')
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
        # The row arrays hold their own references; dropping the collector's lets each reordered field free its original.
        result.segments.clear()
        segments = None
        row_keys = rows.keys() if distribution is not None else None
        if row_order == 'file' and distribution is None:
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
        store = v2t.row_store(tensors, plan.model, plan.loss, segments=segments, store_id=f'v2-iteration-{iteration}')
        del tensors
        if distribution is not None:
            from pokezero.v2_distributed import distribute_store
            rows.arrays.clear()  # DLPack tensors own the buffers through exchange.
            batches = [tuple(g.stream_id for g in b.games) for b in plan.iterations[iteration]]
            store = distribute_store(store, row_keys, batches, runtime)
        mark('handoff_validate')
        train_report = trainer.train(store, window=runs.learning_rate_window(plan, iteration))
        torch.cuda.synchronize();mark('train')
        del store;gc.collect()
        # An attempt interrupted before the state commit owns its artifacts.
        suffix = _rank_zero(distribution, 'checkpoint attempt', lambda: f'-{uuid.uuid4().hex}') if deferred else ''
        checkpoint = out / 'artifacts' / f'train-{iteration}{suffix}.pt'
        receipt = v2t.save_checkpoint(checkpoint, runtime)
        archive = out / 'artifacts' / f'actor-{iteration}{suffix}.npz'

        def export_checkpoint():
            checkpoint_sha = receipt['sha256']
            encoding_receipt = receipt['encoding_receipt']
            encoding_receipts[checkpoint_sha] = encoding_receipt
            metadata = v5_model.export(checkpoint, archive, expected_checkpoint_sha256=checkpoint_sha, enable_v5=True,
                self_moves=plan.self_moves,
                off_provenance=encoding.off_from_receipt(encoding_receipt) if not plan.self_moves else None)
            if deferred:
                sync_actor(archive)
            return checkpoint_sha, metadata

        checkpoint_sha, metadata = _rank_zero(distribution, 'checkpoint export', export_checkpoint)
        snapshot = runs.Snapshot(iteration=iteration, checkpoint=str(checkpoint.relative_to(out)), checkpoint_sha256=checkpoint_sha,
            archive=str(archive.relative_to(out)), archive_sha256=metadata['weights_archive_sha256'],
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
        if lr_start is not None:
            report['training']['lr_start'] = train_report['lr_start']
        if distribution is not None:
            report = distribution.reduce_report(report)
        reports.append(report)
        log('ITERATION', iteration=iteration, games_per_second=round(report['games_per_second'], 2),
            games_per_second_steady=round(report['games_per_second_steady'], 2), phase_seconds={k: round(v, 1) for k, v in phase.items()},
            history=history, reused=reused, occupancy=round(stats.get('occupancy', 0), 3), dispositions=report['dispositions'],
            final_epoch_loss=report['training'].get('final_epoch_mean', {}).get('loss'), matching=same)
        def commit_round():
            state = dict(snapshots=[asdict(s) for s in snapshots], reports=reports, row_order=row_order,
                declaration_sha256=declaration_sha256, checkpoint_encoding_receipts=encoding_receipts)
            if lr_start is not None:
                state['lr_start'] = lr_start_policy
            if distribution is not None:
                state['distribution'] = distribution.contract
            write_json(state_path, state, durable=deferred)
            if deferred:
                publish_evaluation(snapshot)
                # The reference initializer is outside this run's retention.
                retain_full_checkpoints(out, [
                    dict(iteration=item.iteration, path=item.checkpoint) for item in snapshots[-2:]
                    if item.iteration >= 0], dispatch.milestones)

        _rank_zero(distribution, 'round commit, outbox and retention', commit_round)
        del rows, result, roster, tree;gc.collect()
        if getattr(plan, 'historical_anchor_slots', 0):
            keep = {s.archive_sha256 for s in (*history_window(plan, snapshot, snapshots[:-1]), snapshot)}
        else:
            keep = {s.archive_sha256 for s in snapshots[-1 - plan.historical_window:]}
        for digest in list(params):
            if digest not in keep:del params[digest]
    if deferred and route_stop is None:
        route_stop = _rank_zero(distribution, 'evaluation stop', lambda: dispatch.stop_at_boundary(len(reports)))
    total = time.perf_counter() - wall_start
    games = sum(r['games'] for r in reports)
    summary = dict(schema='v2-fused-measurement.v2', declared=context['declared'], lanes=args.lanes, mode=args.mode,
        order=args.order, row_order=row_order, torch_memory_fraction_override=args.torch_memory_fraction if args.torch_memory_fraction > 0 else None, iterations=reports, preparation_seconds=context['preparation_seconds'], wall_seconds_this_attempt=total,
        games=games, games_per_second_steady_mean=games / max(1e-9, sum(r['steady_seconds'] for r in reports)),
        scope='Throughput of the fused V2 pipeline; not a strength result.')
    if route_stop is not None:
        summary.update(status='STOPPED', reason='S0_TRANSFER_FAIL', stop_request=route_stop)
    _rank_zero(distribution, 'final report', lambda: write_json(out / 'report.json', summary))
    log('DONE', wall_seconds=total, iterations=len(reports), games=games)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
