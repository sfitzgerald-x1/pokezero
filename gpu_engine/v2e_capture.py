"""Record real engine-step inputs from the JAX V2 collector (oracle capture).

Runs the unmodified refill collector over the declared roster and, through a
host callback around transition.step_with_public, records every --every-th
tick's exact (state, actions, trackers) step inputs: all lanes, including
empty/done lanes. Each record is one .npz shard of flattened leaves plus the
leaf paths, so gpu_engine.v2e_diff can rebuild the pytrees and replay any
module (readout, encoder, step, trackers) on real states. The callback only
copies arrays; compression and writing run on background threads.

All paths are command-line arguments.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import threading
import time

import numpy as np

from gpu_engine.v2_bench_collector import log, journal, prepare, actor, construct


def leaf_paths(tree):
    import jax
    return [jax.tree_util.keystr(path) for path, _ in jax.tree_util.tree_flatten_with_path(tree)[0]]


def _save_atomic(path, **arrays):
    """Write an npz shard under a temporary name and rename it, so a crash never leaves a partial shard."""
    tmp = path.with_name(path.name + '.part')
    with open(tmp, 'wb') as f:np.savez_compressed(f, **arrays)
    os.replace(tmp, path)


class Sink:
    def __init__(self, out, *, every, skip, limit, workers=8):
        self.out = Path(out);self.out.mkdir(parents=True, exist_ok=True)
        if any(self.out.glob('shard-*.npz')):raise SystemExit(f'{self.out} already holds shards; capture into an empty directory')
        self.every = every;self.skip = skip;self.limit = limit;self.calls = 0;self.saved = 0;self.lanes = 0
        self.pool = ThreadPoolExecutor(max_workers=workers);self.lock = threading.Lock();self.futures = []
        self.paths = None;self.level_lanes = {}

    def __call__(self, level, *leaves):
        with self.lock:
            call = self.calls;self.calls += 1
            if call < self.skip or (call - self.skip) % self.every or (self.limit and self.saved >= self.limit):return
            index = self.saved;self.saved += 1
        arrays = {f'l{i:03d}': np.asarray(x) for i, x in enumerate(leaves)}
        lanes = int(arrays['l000'].shape[0])
        future = self.pool.submit(_save_atomic, self.out / f'shard-{index:06d}.npz', tick=np.int64(call), level=np.int64(level), **arrays)
        with self.lock:
            self.lanes += lanes;self.level_lanes[lanes] = self.level_lanes.get(lanes, 0) + 1;self.futures.append(future)

    def close(self):
        for f in self.futures:f.result()
        self.pool.shutdown()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--reference-run', required=True)
    parser.add_argument('--out', default=os.environ.get('V2_OUT'))
    parser.add_argument('--arm', default='B', choices=('A', 'B'))
    parser.add_argument('--lanes', type=int, default=2048)
    parser.add_argument('--games', type=int, default=None)
    parser.add_argument('--every', type=int, default=2, help='record every k-th tick')
    parser.add_argument('--skip', type=int, default=0, help='ticks to skip before recording')
    parser.add_argument('--limit', type=int, default=0, help='maximum shards (0: no limit)')
    parser.add_argument('--chunk-steps', type=int, default=8)
    args = parser.parse_args(argv)
    out = Path(args.out);out.mkdir(parents=True, exist_ok=True)
    import jax
    from gpu_engine import transition, v2_collector as v2
    events = journal(args.reference_run);context = prepare(args)
    snapshot, params, _ = actor(args, context['plan'], events)
    backend, roster, construction = construct(context, snapshot, events, games=args.games)
    plan = context['plan'];prepared = context['prepared']
    sink = Sink(out / 'shards', every=args.every, skip=args.skip, limit=args.limit)
    original = transition.step_with_public;meta = {}
    def recorded(state, actions, table, trackers, **kwargs):
        tree = (state, actions, trackers)
        if 'paths' not in meta:meta['paths'] = leaf_paths(tree)
        jax.debug.callback(sink, np.int32(state.hp.shape[0]), *jax.tree.leaves(tree), ordered=False)
        return original(state, actions, table, trackers, **kwargs)
    transition.step_with_public = recorded
    try:
        collector = v2.RefillCollector(prepared.engine_tables, prepared.observation_tables, plan.model, backend.labels, roster,
            lanes=args.lanes, max_requests=plan.max_requests, chunk_steps=args.chunk_steps, rows_per_game=112)
        begin = time.perf_counter();collector.compile((params,), threads=4);log('COMPILED', seconds=time.perf_counter() - begin)
        begin = time.perf_counter();result = collector.run((params,))
        jax.effects_barrier()   # every recorded tick's callback has run
        sink.close()
    finally:
        transition.step_with_public = original
    summary = dict(source_commit=os.environ.get('V2_SOURCE_COMMIT'), lanes=args.lanes, games=roster.games, every=args.every,
        skip=args.skip, ticks_seen=sink.calls, shards=sink.saved, lane_steps=sink.lanes, shards_by_lane_count=sink.level_lanes,
        paths=meta['paths'], collect_seconds=time.perf_counter() - begin, kernel_seconds=result.stats['kernel_seconds'])
    (out / 'capture.json').write_text(json.dumps(summary, indent=1) + '\n')
    log('CAPTURED', **{k: v for k, v in summary.items() if k != 'paths'}, leaves=len(meta['paths']))
    return 0


if __name__ == '__main__':raise SystemExit(main())
