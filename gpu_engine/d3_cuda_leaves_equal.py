"""Byte-equality of the D3 frame pricer with generated CUDA leaves, on real D3 frames.

Rebuilds the D3 bank plan from its declared inputs (checked against the run's declaration hash),
loads its iteration -1 actor, collects chunks of the first block with teacher roots retained,
prepares each chunk's selected roots as the vectorized Provider does (one batched detach per scan
step) and prices every frame twice: FramePricer(cuda_leaves=False) and FramePricer(cuda_leaves=True),
same requests, same actor. The frame outputs (pre, peer_logits, peer_legal, requested, values,
errors, done) and the per-root results must be byte-equal; with --capture-inputs the encoded root
and leaf actor inputs (the model's inputs) are compared too. Per-frame device times of both pricers
are reported (--repeat re-prices each frame for steadier numbers). All paths are command-line
arguments; the output directory must be new.
"""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import time

import numpy as np

OUTPUT_KEYS = ('pre', 'peer_logits', 'peer_legal', 'requested', 'values', 'errors', 'done')
INPUT_KEYS = ('own_inputs', 'peer_inputs', 'leaf_inputs')


def log(message, **values):
    print(json.dumps(dict(message=message, time=time.strftime('%H:%M:%S'), **values), default=str), flush=True)


def same_bytes(a, b):
    a, b = np.asarray(a), np.asarray(b)
    return a.dtype == b.dtype and a.shape == b.shape and a.tobytes() == b.tobytes()


def compare_outputs(a, b, keys):
    """{key: lanes differing} over the frame output arrays (axis 0 = root slot)."""
    import jax
    report = {}
    for key in keys:
        if (key in a) != (key in b):report[key] = 'missing';continue
        if key not in a:continue
        la, ta = jax.tree_util.tree_flatten_with_path(a[key]);lb, tb = jax.tree_util.tree_flatten_with_path(b[key])
        if ta != tb:report[key] = 'structure';continue
        for (path, x), (_, y) in zip(la, lb):
            x, y = np.asarray(x), np.asarray(y)
            if x.dtype != y.dtype or x.shape != y.shape:report[key + jax.tree_util.keystr(path)] = f'geometry {x.shape}{x.dtype} vs {y.shape}{y.dtype}';continue
            d = (x.view(np.uint8).reshape(x.shape[0], -1) != y.view(np.uint8).reshape(y.shape[0], -1)).any(axis=1) if x.ndim else x.tobytes() != y.tobytes()
            report[key + jax.tree_util.keystr(path)] = int(np.count_nonzero(d))
    return report


def comparable(result):
    """The parts of a FramePricer result that depend on the frame computation (worlds excluded)."""
    reduction = None if result.reduction is None else asdict(result.reduction)
    return dict(status=result.status, reason=result.reason, missing_worlds=result.missing_worlds, reduction=reduction,
        world_reports=result.world_reports, actual_teacher_calls=result.actual_teacher_calls)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--d3-root', required=True, help='D3 bank run root (prespec.json, runtime.json, run/)')
    parser.add_argument('--labels', required=True);parser.add_argument('--labels-sha256', required=True)
    parser.add_argument('--out', required=True, help='new output directory')
    parser.add_argument('--chunks', type=int, default=2)
    parser.add_argument('--repeat', type=int, default=0, help='extra pricings of every frame (timing)')
    parser.add_argument('--max-roots', type=int, default=0, help='price only the first N selected roots of each frame (0: all)')
    parser.add_argument('--capture-inputs', action='store_true', help='also compare the encoded root and leaf actor inputs')
    parser.add_argument('--steps', type=int, default=0, help='only the first N scan steps (frames) of each chunk (0: all)')
    parser.add_argument('--joint-arena-budget', type=int, default=0, help='override JOINT_ARENA_BUDGET_BYTES (bytes; 0: default), to exercise world slices')
    args = parser.parse_args(argv)
    out = Path(args.out);out.mkdir(parents=True, exist_ok=False)
    root = Path(args.d3_root)
    import jax
    from pokezero import v5_worker as worker, v5_run as runs, v5_contract as contract
    from gpu_engine import v5_device_rollout as rollout, v5_model
    from gpu_engine.v5_device_backend import RuntimeBackend
    from gpu_engine.v5_device_teacher import Provider
    from gpu_engine import device_public_teacher_batch as batch
    from gpu_engine.device_public_teacher_batch import FramePricer
    if args.joint_arena_budget:batch.JOINT_ARENA_BUDGET_BYTES = args.joint_arena_budget
    log('JOINT_ARENA_BUDGET_BYTES', bytes=batch.JOINT_ARENA_BUDGET_BYTES)
    prespec = json.loads((root / 'prespec.json').read_text())
    runtime_path = root / 'runtime.json'
    runtime = worker._record(dict(path=str(runtime_path), sha256=hashlib.sha256(runtime_path.read_bytes()).hexdigest()))
    labels = contract.LabelVocabulary(**worker._record(dict(path=args.labels, sha256=args.labels_sha256)))
    pool_sha = runtime['constructor_pool']['sha256']
    profile, vocabulary, gender, category = worker._runtime_metadata(runtime, expected_pool_sha256=pool_sha)
    d = json.loads((root / 'run' / 'declaration.json').read_text())
    from pokezero.v5_model import V5ModelConfig
    from pokezero.v5_loss import V5LossConfig
    from pokezero import v5_training as training
    loss_fields = {k: v for k, v in d['loss'].items() if k in V5LossConfig.__dataclass_fields__}
    batches = tuple(tuple(runs.Batch(tuple(runs.Game(**g) for g in x['games']), x['constructor_seed'], x['sampling_seed'],
        x.get('gender_seed'), x.get('gender_domain'), tuple(x.get('gender_root_ids', ()))) for x in it) for it in d['iterations'])
    plan = runs.Plan(d['run_id'], V5ModelConfig.from_metadata(d['model']), V5LossConfig(**loss_fields), training.TrainConfig(**d['train']),
        batches, d['pool_sha256'], d['selection_seed'], d['historical_fraction'], d['historical_window'], d['max_requests'],
        runs.Watchdog(**d['watchdog']), checkpoint_updates=d['checkpoint_updates'], discount=d['discount'])
    started = json.loads((root / 'run' / 'journal' / '000000000.json').read_text())['payload']['declaration_sha256']
    if runs._sha(plan.manifest()) != started:raise SystemExit('rebuilt plan differs from the D3 declaration')
    log('PLAN', declaration_sha256=started, batch_rows=len(d['iterations'][0][0]['games']), arm=plan.model.arm)
    try:worker._probe(runtime)
    except ValueError as error:
        log('PROBE_DIFFERS', error=str(error));jax.config.update('jax_enable_x64', False);jax.config.update('jax_default_matmul_precision', 'highest')
    inputs = worker.Inputs(plan, profile, json.dumps(runtime), json.dumps(vocabulary), json.dumps(gender), runs._sha(plan.manifest()))
    clock = time.perf_counter();prepared, pool, _, _ = worker.prepare_device_runtime(inputs);log('PREPARED', seconds=time.perf_counter() - clock)
    provider = Provider(prepared, plan, frame_batch=True, chunk_batch=True, vectorized=True, batched_prep=True)
    backend = RuntimeBackend(prepared, pool, gender_policy=gender, teacher=provider);backend.check_plan(plan)
    events = [json.loads(p.read_text()) for p in sorted((root / 'run' / 'journal').glob('*.json'))[:40]]
    snapshot = next(e['payload'] for e in events if e['kind'] == 'SNAPSHOT' and e['payload']['iteration'] == -1)
    params, config, _ = v5_model.load(root / 'run' / snapshot['archive'], expected_archive_sha256=snapshot['archive_sha256'],
        expected_checkpoint_sha256=snapshot['checkpoint_sha256'], expected_category_vocabulary_sha256=plan.model.category_vocabulary_sha256,
        expected_label_vocabulary_sha256=plan.model.label_vocabulary.sha256, enable_v5=True)
    if config != plan.model:raise SystemExit('actor configuration differs')
    batch = plan.iterations[0][0];ordinal = {g.stream_id: i for i, g in enumerate(plan.games)}
    snap = runs.Snapshot(**snapshot)
    assignments = json.loads(runs._json(tuple(runs.assignment(plan, g, ordinal[g.stream_id], snap, ()) for g in batch.games)))
    archive = snapshot['archive_sha256']
    bindings = rollout.Bindings(plan.model.category_vocabulary_sha256, plan.model.label_vocabulary, prepared.source_sha256,
        (archive,) * 2, (True, True), backend.declaration_sha256)
    actor_bank = rollout.actor_bank_from_assignments(bindings, (archive,), current_archive=archive, assignments=assignments)
    provider.bind(actor_bank, (params,))
    collector = rollout.make_collector(prepared.engine_tables, prepared.observation_tables, plan.model, backend.labels, bindings,
        sampling_seed=batch.sampling_seed, chunk_steps=8, max_requests=plan.max_requests, actor_bank=actor_bank, retain_teacher_roots=True)
    carry = jax.block_until_ready(backend._construct(batch));log('CONSTRUCTED', games=len(batch.games))
    options = dict(actor_capacity=plan.historical_window + 1, capacity=4096, vectorized=True, host_prep=True)
    pricers = dict(off=FramePricer(provider.executor, plan.model, **options), on=FramePricer(provider.executor, plan.model, cuda_leaves=True, **options))
    for pricer in pricers.values():pricer.capture_inputs = args.capture_inputs
    keys = OUTPUT_KEYS + (INPUT_KEYS if args.capture_inputs else ())
    frames_report = [];totals = dict(frames=0, roots=0, worlds=0, branches=0, differing=0, result_differences=0)
    for chunk in range(args.chunks):
        clock = time.perf_counter();device = jax.block_until_ready(collector(carry, (params,)));collect_s = time.perf_counter() - clock
        carry = device.carry
        host = rollout.boundary(device._replace(carry=carry._replace(state=None, public=None, truth=None), roots=None))
        provider.reset_metrics();records, grouped = provider._prepare_batched(device.roots, host.frames, actor_bank)
        log('CHUNK_PREPARED', chunk=chunk, collect_seconds=collect_s, frames=sorted(grouped), roots_by_frame={str(k): len(v) for k, v in sorted(grouped.items())})
        for frame in sorted(grouped)[:args.steps or None]:
            requests = [request for _, request in grouped[frame]][:args.max_roots or None]
            if not requests:continue
            outputs = {};results = {};timings = {}
            for name, pricer in pricers.items():
                runs_ = []
                for attempt in range(1 + args.repeat):
                    priced = pricer.price(requests, provider.materializer, provider.recipe, (params,))
                    t = dict(pricer.last_timing);runs_.append(t)
                    if attempt == 0:outputs[name] = pricer.last_output;results[name] = priced
                timings[name] = dict(first=runs_[0], device_seconds=[t['device_seconds'] for t in runs_],
                    device_min=min(t['device_seconds'] for t in runs_), device_mean=sum(t['device_seconds'] for t in runs_) / len(runs_))
            report = compare_outputs(outputs['off'], outputs['on'], keys)
            same_results = [json.dumps(comparable(a), sort_keys=True, default=str) == json.dumps(comparable(b), sort_keys=True, default=str)
                for a, b in zip(results['off'], results['on'])]
            count = len(requests);capacity = timings['off']['first']['root_capacity']
            differing = sum(v for v in report.values() if isinstance(v, int)) + sum(1 for v in report.values() if not isinstance(v, int))
            slice_worlds = batch.joint_slice_worlds(pricers['on'].generated['joint'], 2 * capacity)
            entry = dict(chunk=chunk, frame=int(frame), roots=count, root_capacity=capacity, worlds=2 * count, branches=162 * count,
                joint_slices=2 * capacity // slice_worlds, joint_slice_worlds=slice_worlds,
                equal=differing == 0 and all(same_results), differing_by_output=report, result_differences=count - sum(same_results),
                statuses={name: dict(zip(*np.unique([r.status for r in rs], return_counts=True))) for name, rs in results.items()},
                device_seconds={name: t['device_seconds'] for name, t in timings.items()},
                device_min_seconds={name: t['device_min'] for name, t in timings.items()},
                compile_seconds={name: t['first']['compile_seconds'] for name, t in timings.items()},
                lower_seconds={name: t['first']['lower_seconds'] for name, t in timings.items()},
                generate_seconds=timings['on']['first'].get('generate_seconds'),
                frame_wall_seconds={name: t['first']['frame_wall_seconds'] for name, t in timings.items()})
            entry['statuses'] = {name: {str(k): int(v) for k, v in s.items()} for name, s in entry['statuses'].items()}
            frames_report.append(entry);log('FRAME', **entry)
            totals['frames'] += 1;totals['roots'] += count;totals['worlds'] += 2 * count;totals['branches'] += 162 * count
            totals['differing'] += differing;totals['result_differences'] += entry['result_differences']
            (out / 'equality.json').write_text(json.dumps(dict(totals=totals, equal=totals['differing'] == 0 and totals['result_differences'] == 0,
                compared_keys=keys, frames=frames_report), indent=1, default=str) + '\n')
    generated = pricers['on'].generated or {}
    modules = {name: dict(g.module.stats, arena_words=g.module.arena_words, inputs=len(g.module.inputs), outputs=len(g.module.outputs),
        library=str(g.path)) for name, g in generated.items()}
    equal = totals['differing'] == 0 and totals['result_differences'] == 0 and totals['frames'] > 0
    (out / 'equality.json').write_text(json.dumps(dict(totals=totals, equal=equal, compared_keys=keys, frames=frames_report, modules=modules),
        indent=1, default=str) + '\n')
    log('EQUALITY', equal=equal, **totals)
    log('MODULES', modules=modules)
    log('DONE')
    return 0 if equal else 1


if __name__ == '__main__':raise SystemExit(main())
