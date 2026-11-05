"""One D3 bank chunk, labeled by the JAX teacher and by flagged variants: records must be equal.

Rebuilds the D3 bank plan from its declared inputs (checked against the run's declaration hash),
loads its iteration -1 actor, constructs the first batch and collects --chunks chunks with teacher
roots retained. Each variant Provider (--variant: comma-separated extra Provider flags set True, e.g.
`batched_prep`) labels every chunk; the baseline Provider (frame_batch, chunk_batch, vectorized: the
D3 JAX teacher, about 0.5 s per root) labels the first --baseline-chunks chunks, and on those every
variant's label_chunk records must equal the baseline's: exactly, or (reported separately) within the
saved-root gate's float tolerance. Per provider and chunk it reports label wall time and host
preparation vs pricer device time per priced root; chunk 0 includes compilation, so steady-state
numbers come from later chunks. All paths are command-line arguments; the output directory must be new.
"""
import argparse
import collections
import hashlib
import json
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


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('--d3-root', required=True, help='the D3 bank run root (prespec.json, runtime.json, run/)')
    parser.add_argument('--labels', required=True)
    parser.add_argument('--labels-sha256', required=True)
    parser.add_argument('--out', required=True, help='new output directory')
    parser.add_argument('--chunks', type=int, default=2)
    parser.add_argument('--baseline-chunks', type=int, default=1, help='chunks the baseline teacher labels and the variants are compared on')
    parser.add_argument('--variant', action='append', default=[], help='comma-separated Provider flags set True, e.g. batched_prep')
    args = parser.parse_args(argv)
    if not args.variant:raise SystemExit('at least one --variant required')
    if not 1 <= args.baseline_chunks <= args.chunks:raise SystemExit('need 1 <= --baseline-chunks <= --chunks')
    out = Path(args.out);out.mkdir(parents=True, exist_ok=False)
    root = Path(args.d3_root)
    import jax
    import numpy as np  # noqa: F401  (the teacher's numpy path)
    from pokezero import v5_worker as worker, v5_run as runs, v5_contract as contract
    from pokezero.v5_model import V5ModelConfig
    from pokezero.v5_loss import V5LossConfig
    from pokezero import v5_training as training
    from gpu_engine import v5_device_rollout as rollout, v5_model
    from gpu_engine.v5_device_backend import RuntimeBackend
    from gpu_engine.v5_device_teacher import Provider
    from gpu_engine.d3_saved_root_gate import differences
    runtime_path = root / 'runtime.json'
    runtime = worker._record(dict(path=str(runtime_path), sha256=hashlib.sha256(runtime_path.read_bytes()).hexdigest()))
    worker._record(dict(path=args.labels, sha256=args.labels_sha256))
    profile, vocabulary, gender, _ = worker._runtime_metadata(runtime, expected_pool_sha256=runtime['constructor_pool']['sha256'])
    d = json.loads((root / 'run' / 'declaration.json').read_text())
    loss_fields = {k: v for k, v in d['loss'].items() if k in V5LossConfig.__dataclass_fields__}
    batches = tuple(tuple(runs.Batch(tuple(runs.Game(**g) for g in x['games']), x['constructor_seed'], x['sampling_seed'],
        x.get('gender_seed'), x.get('gender_domain'), tuple(x.get('gender_root_ids', ()))) for x in it) for it in d['iterations'])
    plan = runs.Plan(d['run_id'], V5ModelConfig.from_metadata(d['model']), V5LossConfig(**loss_fields), training.TrainConfig(**d['train']),
        batches, d['pool_sha256'], d['selection_seed'], d['historical_fraction'], d['historical_window'], d['max_requests'],
        runs.Watchdog(**d['watchdog']), checkpoint_updates=d['checkpoint_updates'], discount=d['discount'])
    declared = json.loads((root / 'run' / 'journal' / '000000000.json').read_text())['payload']['declaration_sha256']
    if runs._sha(plan.manifest()) != declared:raise SystemExit('rebuilt plan differs from the D3 declaration')
    log('PLAN', declaration_sha256=declared, arm=plan.model.arm, games=len(d['iterations'][0][0]['games']))
    try:worker._probe(runtime)
    except ValueError as error:
        log('PROBE_DIFFERS', error=str(error));jax.config.update('jax_enable_x64', False);jax.config.update('jax_default_matmul_precision', 'highest')
    inputs = worker.Inputs(plan, profile, json.dumps(runtime), json.dumps(vocabulary), json.dumps(gender), runs._sha(plan.manifest()))
    prepared, pool, _, _ = worker.prepare_device_runtime(inputs)
    base = dict(frame_batch=True, chunk_batch=True, vectorized=True)
    names = ['baseline'] + args.variant
    providers = {'baseline': Provider(prepared, plan, **base)}
    for variant in args.variant:
        providers[variant] = Provider(prepared, plan, **base, **{flag: True for flag in variant.split(',')})
    backend = RuntimeBackend(prepared, pool, gender_policy=gender, teacher=providers['baseline']);backend.check_plan(plan)
    events = [json.loads(p.read_text()) for p in sorted((root / 'run' / 'journal').glob('*.json'))[:40]]
    snapshot = next(e['payload'] for e in events if e['kind'] == 'SNAPSHOT' and e['payload']['iteration'] == -1)
    params, config, _ = v5_model.load(root / 'run' / snapshot['archive'], expected_archive_sha256=snapshot['archive_sha256'],
        expected_checkpoint_sha256=snapshot['checkpoint_sha256'], expected_category_vocabulary_sha256=plan.model.category_vocabulary_sha256,
        expected_label_vocabulary_sha256=plan.model.label_vocabulary.sha256, enable_v5=True)
    if config != plan.model:raise SystemExit('actor configuration differs')
    batch = plan.iterations[0][0];ordinal = {g.stream_id: i for i, g in enumerate(plan.games)};snap = runs.Snapshot(**snapshot)
    assignments = json.loads(runs._json(tuple(runs.assignment(plan, g, ordinal[g.stream_id], snap, ()) for g in batch.games)))
    archive = snapshot['archive_sha256']
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
            equal = equal and comparisons[name]['within_tolerance']
            log('COMPARED', chunk=chunk, variant=name, **comparisons[name])
        report['chunks'].append(dict(chunk=chunk, timings=timings, comparisons=comparisons))
    report['equal'] = equal;report['compared_chunks'] = args.baseline_chunks
    (out / 'chunk-equality.json').write_text(json.dumps(report, indent=1, default=str) + '\n')
    log('CHUNK_EQUALITY_COMPLETE', equal=equal, out=str(out / 'chunk-equality.json'))
    return 0 if equal else 1


if __name__ == '__main__':raise SystemExit(main())
