"""Retrain one file-pipeline iteration with the V2 trainer on its exact caches.

Reads the reference run's `train-<iteration>` checkpoint metadata to recover
its ordered cache roster (progress manifest digests), loads those caches into
a V2 row store (one ordered segment per cache, as `train_caches` reads them),
starts from the reference `train-<iteration-1>` weights and trains the same
epochs under the same declared learning-rate window. Writes a V2 checkpoint,
exports the actor and records `state.json` so `v2_head_to_head` can name it.

This isolates the trainer: same rows, same order, same initial weights.

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
    parser.add_argument('--iteration', type=int, default=0)
    parser.add_argument('--total-games', type=int, required=True, help='declared games in the reference plan')
    parser.add_argument('--games-per-iteration', type=int, default=20000)
    parser.add_argument('--mode', default='cudagraphs');parser.add_argument('--order', default='ordered')
    parser.add_argument('--optimizer', default='fused')
    parser.add_argument('--out', required=True)
    args = parser.parse_args(argv)
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
    for path in glob.glob(str(run / 'caches' / 'cache-*')):
        meta = Path(path, 'metadata.json')
        if meta.exists():by_digest[json.loads(meta.read_text()).get('metadata_sha256')] = Path(path)
    missing = [d for d in digests if d not in by_digest]
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
    window = v5.DeclaredGameWindow(args.iteration * args.games_per_iteration, (args.iteration + 1) * args.games_per_iteration, args.total_games)
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
