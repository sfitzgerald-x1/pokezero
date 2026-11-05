"""Retrain one file-pipeline iteration with the V2 trainer on its exact caches.

Reads the reference run's `train-<iteration>` checkpoint metadata to recover
its ordered cache roster (progress manifest digests), loads those caches into
a V2 row store (one ordered segment per cache, as `train_caches` reads them),
starts from the reference initial weights and trains the same epochs under
the learning-rate window recorded in the reference checkpoint. Writes a V2
checkpoint, exports the actor and records `state.json` so `v2_head_to_head`
can name it.

Iteration 0 only: it starts from a fresh optimizer, so same rows, same order
and same initial weights isolate the trainer. Later iterations would also need
the previous AdamW state, step and row counters and RNG, which this tool does
not restore.

All paths are command-line arguments.
"""
import argparse
from dataclasses import asdict
import glob
import json
from pathlib import Path
import time

from gpu_engine.v2_bench_collector import log
from gpu_engine.v2_fused import journal, sha, write_json


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference-run', required=True)
    parser.add_argument('--iteration', type=int, default=0, help='must be 0 (see the module docstring)')
    parser.add_argument('--mode', default='cudagraphs');parser.add_argument('--order', default='ordered')
    parser.add_argument('--optimizer', default='fused')
    parser.add_argument('--extra-cache-root', action='append', default=[], help='more directories to search for caches')
    parser.add_argument('--out', required=True)
    args = parser.parse_args(argv)
    if args.iteration != 0:raise SystemExit('only iteration 0 is supported: later iterations need optimizer state this tool does not restore')
    import torch
    from pokezero import v5_run as runs, v5_training as v5, v2_trainer as v2t
    from gpu_engine import v5_model
    from gpu_engine.v2_bench_trainer import configurations
    run = Path(args.reference_run);out = Path(args.out);(out / 'artifacts').mkdir(parents=True, exist_ok=True)
    events = journal(run)
    def checkpoint_of(iteration):
        found = [e['payload'] for e in events if e['kind'] == 'SNAPSHOT' and e['payload']['iteration'] == iteration]
        if not found:raise ValueError(f'no reference snapshot for iteration {iteration}')
        return runs.Snapshot(**found[-1])
    target, start = checkpoint_of(args.iteration), checkpoint_of(args.iteration - 1)
    payload = torch.load(run / target.checkpoint, map_location='cpu', weights_only=True)
    manifest = payload['metadata']['progress']['manifest'] if payload['metadata'].get('progress') else None
    if not manifest:raise ValueError('reference checkpoint carries no cache manifest')
    digests = [record['sha256'] for record in manifest]
    by_digest = {}
    for root in [run, *(Path(x) for x in args.extra_cache_root)]:
        for meta in root.rglob('metadata.json'):
            if not meta.parent.name.startswith(('cache-', 'recovered-')):continue  # recovered banks keep their own prefix
            try:by_digest.setdefault(json.loads(meta.read_text()).get('metadata_sha256'), meta.parent)
            except (OSError, json.JSONDecodeError):continue
    missing = [d for d in digests if d not in by_digest]
    if missing:log('MISSING_CACHES', digests=missing, searched=[str(run), *args.extra_cache_root])
    if missing:raise ValueError(f'{len(missing)} reference caches not found')
    paths = [by_digest[d] for d in digests]
    model, loss, train, weights, _ = configurations(run / start.checkpoint)
    reference_train = v5.TrainConfig(**payload['metadata']['training'])
    if reference_train != train:raise ValueError('reference iteration recipe differs from its starting checkpoint')
    begin = time.perf_counter()
    store = v2t.load_cache_store(paths, model, loss, device='cuda')
    load_seconds = time.perf_counter() - begin
    runtime = v2t.create(model, loss, train, v2t.V2Config(mode=args.mode, order=args.order, optimizer=args.optimizer),
        enable_v5=True, device='cuda')
    runtime.model.load_state_dict(weights);trainer = v2t.Trainer(runtime)
    saved_rate = payload['metadata'].get('learning_rate_state')
    if train.learning_rate_schedule == 'constant':
        if saved_rate is not None:raise ValueError('constant reference recipe carries scheduled state')
        window = None
    else:
        if not saved_rate:raise ValueError('scheduled reference checkpoint carries no learning-rate window')
        window = v5.DeclaredGameWindow(**saved_rate['window'])
    begin = time.perf_counter();report = trainer.train(store, window=window);torch.cuda.synchronize()
    train_seconds = time.perf_counter() - begin
    checkpoint = out / 'artifacts' / f'train-{args.iteration}.pt';v2t.save_checkpoint(checkpoint, runtime)
    archive = out / 'artifacts' / f'actor-{args.iteration}.npz';digest = sha(checkpoint)
    metadata = v5_model.export(checkpoint, archive, expected_checkpoint_sha256=digest, enable_v5=True)
    snapshot = runs.Snapshot(iteration=args.iteration, checkpoint=f'artifacts/train-{args.iteration}.pt', checkpoint_sha256=digest,
        archive=f'artifacts/actor-{args.iteration}.npz', archive_sha256=metadata['weights_archive_sha256'], model_sha256=target.model_sha256)
    summary = dict(reference_run=str(run), iteration=args.iteration, caches=len(paths), rows=store.rows,
        reference_steps=payload['metadata'].get('steps'), reference_rows=payload['metadata'].get('rows'),
        v2_steps=runtime.steps, v2_rows=runtime.rows, mode=trainer.mode_report(), order=args.order,
        load_seconds=load_seconds, train_seconds=train_seconds)
    write_json(out / 'state.json', dict(snapshots=[asdict(snapshot)], reports=[summary]))
    log('RETRAINED', **summary)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
