"""Bitwise before/after check of the V2 collector on a fixed roster block.

Collects the first --games declared games (whole construction blocks) with the
refill collector, finalizes the row store and writes a record: per-game
terminal records, the (stream, frame, owner) key of every row, one digest per
row over every row-store field, and one digest per field over the key-sorted
rows, and an on-device checksum of every leaf of the lane carry (engine
state, public trackers, truth) after every chunk. --compare loads an earlier record (for example from the unmodified
engine) and reports exact equality, the differing fields and the first
differing rows. Run both sides with the same XLA autotune cache
(--xla_gpu_per_fusion_autotune_cache_dir) so the unchanged forward keeps the
same GEMM choices. All paths are command-line arguments.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np

from gpu_engine.v2_bench_collector import log, journal, prepare, actor, construct

GAME_FIELDS = ('status', 'winner', 'terminal_frame', 'terminal_hp', 'terminal_max_hp', 'bad')


def carry_checksum(carry):
    """Two uint32 position-weighted checksums per array leaf, on device."""
    import jax
    import jax.numpy as j
    def one(x):
        x = j.asarray(x)
        if x.dtype == j.bool_ or x.dtype.itemsize < 4:u = x.astype(j.uint32)
        elif x.dtype.itemsize == 4:u = jax.lax.bitcast_convert_type(x, j.uint32)
        else:raise ValueError('unexpected 64-bit carry leaf')
        u = u.reshape(-1);i = j.arange(u.shape[0], dtype=j.uint32)
        return j.stack([j.sum(u * (2 * i + 1), dtype=j.uint32), jax.lax.reduce(u ^ (i * j.uint32(2654435761)), j.uint32(0), jax.lax.bitwise_xor, (0,))])
    return j.stack([one(x) for x in jax.tree.leaves(carry)])


def watch(collector):
    """Record carry checksums after every chunk kernel call."""
    import jax
    original = collector.kernel;digests = [];checksum = jax.jit(carry_checksum)
    def kernel(level):
        executable = original(level)
        def call(lanes, book, rows, device, params):
            out = executable(lanes, book, rows, device, params)
            digests.append(np.asarray(jax.device_get(checksum(out[0].carry))));return out
        return call
    collector.kernel = kernel
    return digests


def record(args):
    import jax
    from gpu_engine import v2_collector as v2
    events = journal(args.reference_run);context = prepare(args)
    snapshot, params, _ = actor(args, context['plan'], events)
    backend, roster, construction = construct(context, snapshot, events, games=args.games)
    plan = context['plan'];prepared = context['prepared']
    collector = v2.RefillCollector(prepared.engine_tables, prepared.observation_tables, plan.model, backend.labels, roster,
        lanes=args.lanes, max_requests=plan.max_requests, chunk_steps=args.chunk_steps, rows_per_game=args.rows_per_game)
    begin = time.perf_counter();collector.compile((params,), threads=4);compile_s = time.perf_counter() - begin
    log('COMPILED', seconds=compile_s, levels=[x.lanes for x in collector.levels])
    digests = watch(collector)
    result = collector.run((params,))
    log('COLLECTED', **{k: result.stats[k] for k in ('kernel_seconds', 'ticks', 'occupancy', 'rows')},
        seconds_per_tick=[x['seconds_per_tick'] for x in result.stats['levels']])
    store = v2.finalize(result, loss_config=plan.loss, discount=plan.discount)
    book = jax.device_get(result.book)
    games = {name: np.asarray(getattr(book, name)) for name in GAME_FIELDS}
    games['streams'] = np.asarray(roster.streams)
    n = store.n;keys = store.keys().astype(np.int64)
    order = np.lexsort((keys[:, 2], keys[:, 1], keys[:, 0]))
    names = sorted(store.arrays)
    row_hash = [hashlib.sha256() for _ in range(n)] if args.row_digests else None
    field_digest = {}
    for name in names:
        value = np.asarray(jax.device_get(store.arrays[name]))[:n][order]
        flat = np.ascontiguousarray(value).reshape(n, -1).view(np.uint8)
        field_digest[name] = dict(sha256=hashlib.sha256(flat.tobytes()).hexdigest(), dtype=str(value.dtype), shape=list(value.shape))
        if row_hash is not None:
            for i in range(n):row_hash[i].update(flat[i].tobytes())
    rows = np.asarray([h.digest()[:16] for h in row_hash], dtype='S16') if row_hash is not None else None
    out = Path(args.out);out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / 'equivalence.npz', keys=keys[order], carry=np.stack(digests), **({'rows': rows} if rows is not None else {}),
        **{f'game_{k}': v for k, v in games.items()})
    summary = dict(source_commit=os.environ.get('V2_SOURCE_COMMIT'), xla_flags=os.environ.get('XLA_FLAGS'), games=roster.games,
        lanes=args.lanes, rows=n, compile_seconds=compile_s, stats={k: result.stats[k] for k in ('kernel_seconds', 'ticks', 'occupancy')},
        seconds_per_tick=[x['seconds_per_tick'] for x in result.stats['levels']], fields=field_digest,
        dispositions=store.stats['dispositions'])
    (out / 'equivalence.json').write_text(json.dumps(summary, indent=1, default=str) + '\n')
    log('RECORDED', out=str(out), rows=n, games=roster.games)
    return out


def compare(a, b):
    ja = json.loads(Path(a, 'equivalence.json').read_text());jb = json.loads(Path(b, 'equivalence.json').read_text())
    na = np.load(Path(a, 'equivalence.npz'));nb = np.load(Path(b, 'equivalence.npz'))
    games = {k[5:]: bool(np.array_equal(na[k], nb[k])) for k in na.files if k.startswith('game_')}
    keys = na['keys'].shape == nb['keys'].shape and bool(np.array_equal(na['keys'], nb['keys']))
    fields = {k: ja['fields'][k]['sha256'] == jb['fields'].get(k, {}).get('sha256') for k in ja['fields']}
    carry = na['carry'].shape == nb['carry'].shape and bool(np.array_equal(na['carry'], nb['carry']))
    first_carry = None
    if not carry and na['carry'].shape[1:] == nb['carry'].shape[1:]:
        m = min(len(na['carry']), len(nb['carry']));bad = np.argwhere(np.any(na['carry'][:m] != nb['carry'][:m], axis=-1))
        first_carry = dict(chunk=int(bad[0, 0]), leaves=sorted({int(x) for x in bad[bad[:, 0] == bad[0, 0], 1]})) if len(bad) else dict(chunks=[len(na['carry']), len(nb['carry'])])
    rows = None;first = []
    if keys and 'rows' in na.files and 'rows' in nb.files:
        differ = np.flatnonzero(na['rows'] != nb['rows']);rows = dict(differ=int(len(differ)), total=int(len(na['rows'])))
        first = na['keys'][differ[:20]].tolist()
    equal = all(games.values()) and keys and all(fields.values()) and carry
    report = dict(equal=equal, games=games, keys_equal=keys, fields_equal=fields, rows=rows, first_differing_keys=first,
        carry_equal=carry, carry_chunks=int(len(na['carry'])), carry_leaves=int(na['carry'].shape[1]), first_carry_difference=first_carry,
        a=dict(commit=ja['source_commit'], rows=ja['rows'], seconds_per_tick=ja['seconds_per_tick'], kernel_seconds=ja['stats']['kernel_seconds']),
        b=dict(commit=jb['source_commit'], rows=jb['rows'], seconds_per_tick=jb['seconds_per_tick'], kernel_seconds=jb['stats']['kernel_seconds']))
    log('COMPARED', **report)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config')
    parser.add_argument('--reference-run')
    parser.add_argument('--out', default=os.environ.get('V2_OUT'))
    parser.add_argument('--arm', default='B', choices=('A', 'B'))
    parser.add_argument('--games', type=int, default=4000)
    parser.add_argument('--lanes', type=int, default=2048)
    parser.add_argument('--chunk-steps', type=int, default=8)
    parser.add_argument('--rows-per-game', type=int, default=112)
    parser.add_argument('--row-digests', action='store_true')
    parser.add_argument('--compare', nargs=2, metavar=('A', 'B'), help='compare two recorded directories and exit')
    args = parser.parse_args(argv)
    if args.compare:
        report = compare(*args.compare)
        if args.out:Path(args.out).mkdir(parents=True, exist_ok=True);Path(args.out, 'compare.json').write_text(json.dumps(report, indent=1) + '\n')
        return 0 if report['equal'] else 1
    if not (args.config and args.reference_run and args.out):raise ValueError('config, reference run and output required')
    record(args)
    return 0


if __name__ == '__main__':raise SystemExit(main())
