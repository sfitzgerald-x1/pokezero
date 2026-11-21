"""One D3 bank chunk, labeled by the JAX teacher and by flagged variants: records must be equal.

Rebuilds the D3 bank plan from its declared inputs (checked against the run's declaration hash),
loads its iteration -1 actor, constructs the first batch and collects --chunks chunks with teacher
roots retained. Each variant Provider (--variant: comma-separated extra Provider flags set True, e.g.
`batched_prep`) labels every chunk; the baseline Provider (frame_batch, chunk_batch, vectorized: the
D3 JAX teacher, about 0.5 s per root) labels the first --baseline-chunks chunks, and on those every
variant's label_chunk records must equal the baseline's exactly (same process, same programs; the
saved-root gate's float tolerance is reported as a diagnostic only). Per provider and chunk it reports label wall time and host
preparation vs pricer device time per priced root; chunk 0 includes compilation, so steady-state
numbers come from later chunks. All paths are command-line arguments; the output directory must be new.
"""
import argparse
import collections
import hashlib
import json
import os
from pathlib import Path
import time


def log(message, **values):
    print(json.dumps(dict(message=message, time=time.strftime('%H:%M:%S'), **values), default=str), flush=True)


def canonical(records):
    return json.loads(json.dumps(records, default=str, sort_keys=True))


def timing(provider, records, label_seconds):
    calls = provider.batch_timings;priced = sum(x['roots'] for x in calls)
    prepare = sum(getattr(provider, 'root_prepare_seconds', {}).values())
    world_prepare = sum(x['world_prepare_seconds'] for x in calls)
    device = sum(x['device_seconds'] for x in calls);compile_ = sum(x['lower_seconds'] + x['compile_seconds'] for x in calls)
    per = lambda seconds: None if not priced else 1e3 * seconds / priced
    return dict(label_seconds=label_seconds, statuses=dict(collections.Counter(r['status'] for r in records)), priced_roots=priced,
        pricer_calls=len(calls), compile_seconds=compile_, root_prepare_seconds=prepare, world_prepare_seconds=world_prepare,
        device_seconds=device, host_prepare_ms_per_root=per(prepare + world_prepare), device_ms_per_root=per(device),
        label_ms_per_root=per(label_seconds - compile_), joint_leaves=162 * priced,
        leaves_per_second=None if label_seconds <= compile_ else 162 * priced / (label_seconds - compile_))


def initializer_proof(origin, snapshot):
    from pokezero import checkpoint_self_moves as encoding
    from pokezero.v5_run import Snapshot

    if origin is None:
        raise ValueError('an OFF actor in an ON plan requires --initializer-run with original initializer evidence')
    root, declaration, events = encoding.verified_run(origin)
    if declaration.get('self_moves', False):
        raise ValueError('the original initializer run must declare self_moves off')
    identity = ('iteration', 'checkpoint_sha256', 'archive_sha256', 'model_sha256')
    matches = [
        Snapshot(**event['payload']) for event in events
        if event['kind'] == 'SNAPSHOT'
        and all(event['payload'].get(field) == getattr(snapshot, field) for field in identity)
    ]
    if len(matches) != 1:
        raise ValueError('exactly one original snapshot matching the destination initializer is required')
    proof, _ = encoding.original_initializer(root, matches[0])
    return proof


def load_actor(plan, snapshot, run_root, initializer_run):
    from gpu_engine import v5_model
    from gpu_engine.private_self_moves import check_run_actor
    from pokezero.checkpoint_self_moves import archive_self_moves
    from pokezero.v5_run import _sha

    if snapshot.model_sha256 != _sha(plan.model.metadata()):
        raise ValueError('snapshot model digest differs from the plan')
    params, config, meta = v5_model.load(run_root / snapshot.archive, expected_archive_sha256=snapshot.archive_sha256,
        expected_checkpoint_sha256=snapshot.checkpoint_sha256,
        expected_category_vocabulary_sha256=plan.model.category_vocabulary_sha256,
        expected_label_vocabulary_sha256=plan.model.label_vocabulary.sha256, enable_v5=True)
    if config != plan.model:
        raise SystemExit('actor configuration differs')
    proof = None
    if plan.self_moves and not archive_self_moves(meta):
        proof = initializer_proof(initializer_run, snapshot)
    check_run_actor(meta, plan_self_moves=plan.self_moves, iteration=snapshot.iteration,
                    context='D3 chunk equal', initializer=proof)
    return params


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('--d3-root', required=True, help='the D3 bank run root (prespec.json, runtime.json, run/)')
    parser.add_argument('--initializer-run', type=Path,
        help='original OFF run directory with the matching iteration -1 snapshot, required only for an OFF initializer in an ON plan')
    parser.add_argument('--out', required=True, help='new output directory')
    parser.add_argument('--chunks', type=int, default=2)
    parser.add_argument('--baseline-chunks', type=int, default=1, help='chunks the baseline teacher labels and the variants are compared on')
    parser.add_argument('--variant', action='append', default=[], help='comma-separated Provider flags set True, e.g. batched_prep')
    parser.add_argument('--memory-policy', action='store_true', help="apply the bank runtime's declared JAX memory policy (else the caller's environment)")
    args = parser.parse_args(argv)
    if not args.variant:raise SystemExit('at least one --variant required')
    if not 1 <= args.baseline_chunks <= args.chunks:raise SystemExit('need 1 <= --baseline-chunks <= --chunks')
    out = Path(args.out);out.mkdir(parents=True, exist_ok=False)
    root = Path(args.d3_root)
    run_root = root / 'run'
    from pokezero import v5_worker as worker, v5_run as runs
    runtime_path = root / 'runtime.json'
    runtime = worker._record(dict(path=str(runtime_path), sha256=hashlib.sha256(runtime_path.read_bytes()).hexdigest()))
    if args.memory_policy:worker._memory(runtime['memory_policy'], apply=True)   # must precede the first JAX import
    import jax
    from pokezero.v5_model import V5ModelConfig
    from pokezero.v5_loss import V5LossConfig
    from pokezero import v5_training as training
    from gpu_engine import v5_device_rollout as rollout
    from gpu_engine.v5_device_backend import RuntimeBackend
    from gpu_engine.v5_device_teacher import Provider
    from gpu_engine.d3_saved_root_gate import differences
    profile, vocabulary, gender, _ = worker._runtime_metadata(runtime, expected_pool_sha256=runtime['constructor_pool']['sha256'])
    d = json.loads((run_root / 'declaration.json').read_text())
    if 'self_moves' in d and d['self_moves'] is not True:
        raise ValueError('self_moves must be omitted or declared as true in the plan')
    loss_fields = {k: v for k, v in d['loss'].items() if k in V5LossConfig.__dataclass_fields__}
    batches = tuple(tuple(runs.Batch(tuple(runs.Game(**g) for g in x['games']), x['constructor_seed'], x['sampling_seed'],
        x.get('gender_seed'), x.get('gender_domain'), tuple(x.get('gender_root_ids', ()))) for x in it) for it in d['iterations'])
    plan = runs.Plan(d['run_id'], V5ModelConfig.from_metadata(d['model']), V5LossConfig(**loss_fields), training.TrainConfig(**d['train']),
        batches, d['pool_sha256'], d['selection_seed'], d['historical_fraction'], d['historical_window'], d['max_requests'],
        runs.Watchdog(**d['watchdog']), checkpoint_updates=d['checkpoint_updates'], discount=d['discount'],
        self_moves=d.get('self_moves', False))
    declared = json.loads((run_root / 'journal' / '000000000.json').read_text())['payload']['declaration_sha256']
    if runs._sha(plan.manifest()) != declared:raise SystemExit('rebuilt plan differs from the D3 declaration')
    log('PLAN', declaration_sha256=declared, arm=plan.model.arm, games=len(d['iterations'][0][0]['games']),
        memory={k: os.environ.get(k) for k in ('XLA_PYTHON_CLIENT_PREALLOCATE', 'XLA_PYTHON_CLIENT_MEM_FRACTION', 'XLA_PYTHON_CLIENT_ALLOCATOR')})
    try:worker._probe(runtime)
    except ValueError as error:
        log('PROBE_DIFFERS', error=str(error));jax.config.update('jax_enable_x64', False);jax.config.update('jax_default_matmul_precision', 'highest')
    inputs = worker.Inputs(plan, profile, json.dumps(runtime), json.dumps(vocabulary), json.dumps(gender), runs._sha(plan.manifest()))
    prepared, pool, _, _ = worker.prepare_device_runtime(inputs)
    events = (json.loads(p.read_text()) for p in sorted((run_root / 'journal').glob('*.json')))
    snap = next(runs.Snapshot(**e['payload']) for e in events if e['kind'] == 'SNAPSHOT' and e['payload']['iteration'] == -1)
    params = load_actor(plan, snap, run_root, args.initializer_run)
    base = dict(frame_batch=True, chunk_batch=True, vectorized=True)
    names = ['baseline'] + args.variant
    providers = {'baseline': Provider(prepared, plan, **base)}
    for variant in args.variant:
        providers[variant] = Provider(prepared, plan, **base, **{flag: True for flag in variant.split(',')})
    backend = RuntimeBackend(prepared, pool, gender_policy=gender, teacher=providers['baseline']);backend.check_plan(plan)
    batch = plan.iterations[0][0];ordinal = {g.stream_id: i for i, g in enumerate(plan.games)}
    assignments = json.loads(runs._json(tuple(runs.assignment(plan, g, ordinal[g.stream_id], snap, ()) for g in batch.games)))
    archive = snap.archive_sha256
    bindings = rollout.Bindings(plan.model.category_vocabulary_sha256, plan.model.label_vocabulary, prepared.source_sha256,
        (archive,) * 2, (True, True), backend.declaration_sha256)
    actor_bank = rollout.actor_bank_from_assignments(bindings, (archive,), current_archive=archive, assignments=assignments)
    collector = rollout.make_collector(prepared.engine_tables, prepared.observation_tables, plan.model, backend.labels, bindings,
        sampling_seed=batch.sampling_seed, chunk_steps=8, max_requests=plan.max_requests, actor_bank=actor_bank, retain_teacher_roots=True)
    carry = jax.block_until_ready(backend._construct(batch));log('CONSTRUCTED', games=len(batch.games))
    report = dict(d3_root=str(root), declaration_sha256=declared, baseline=base, variants=args.variant, chunks=[])
    equal = True
    for chunk in range(args.chunks):
        device = jax.block_until_ready(collector(carry, (params,)));carry = device.carry
        host = rollout.boundary(device._replace(carry=carry._replace(state=None, public=None, truth=None), roots=None))
        records, timings = {}, {}
        for name in names if chunk < args.baseline_chunks else args.variant:
            provider = providers[name];provider.reset_metrics()
            clock = time.perf_counter();result = provider.label_chunk(device.roots, host.frames, actor_bank, (params,))
            timings[name] = timing(provider, result, time.perf_counter() - clock);records[name] = canonical(result)
            log('LABELED', chunk=chunk, provider=name, **timings[name])
        comparisons = {}
        for name in args.variant if chunk < args.baseline_chunks else ():
            a, b = records['baseline'], records[name]
            found = [] if len(a) != len(b) else [(i, d) for i, (x, y) in enumerate(zip(a, b)) for d in [differences(x, y)] if d]
            comparisons[name] = dict(records=[len(a), len(b)], exact=a == b, within_tolerance=len(a) == len(b) and not found,
                differing_records=len(found), first_differences=[dict(record=i, differences=d[:5]) for i, d in found[:3]])
            equal = equal and comparisons[name]['exact']
            log('COMPARED', chunk=chunk, variant=name, **comparisons[name])
        report['chunks'].append(dict(chunk=chunk, timings=timings, comparisons=comparisons))
    report['exact'] = equal;report['compared_chunks'] = args.baseline_chunks
    (out / 'chunk-equality.json').write_text(json.dumps(report, indent=1, default=str) + '\n')
    log('CHUNK_EQUALITY_COMPLETE', exact=equal, out=str(out / 'chunk-equality.json'))
    return 0 if equal else 1


if __name__ == '__main__':raise SystemExit(main())
