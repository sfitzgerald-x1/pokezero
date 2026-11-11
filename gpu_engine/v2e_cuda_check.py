"""GPU checks for the V2 CUDA engine phases P0 and P2.

p0: the real collector chunk with every tick's lane carry passed through the
    in-place CUDA round trip, against the unmodified collector from the same
    start: all lanes, book and row buffers bitwise after each chunk, ms per
    tick, kernels and host graph launches per chunk.
p2: CUDA chance inputs and key advance against transition.sample_chance_inputs
    and jax.random.split on random and edge keys (>= 100M draws). The appended
    `effect` field differs on every lane until the CUDA kernel draws it
    (deferred to CUDA option A, design 13 item 6).

All paths are command-line arguments.
"""
import argparse
import json
import os
from pathlib import Path
import statistics
import time

import numpy as np

from gpu_engine.v2_bench_collector import log, journal, prepare, actor, construct


def tree_equal(a, b):
    import jax
    import jax.numpy as j
    la, lb = jax.tree.leaves(a), jax.tree.leaves(b)
    if len(la) != len(lb):return False, 'leaf count'
    for i, (x, y) in enumerate(zip(la, lb)):
        if x.shape != y.shape or x.dtype != y.dtype:return False, f'leaf {i} shape/dtype'
        if x.dtype == j.bool_:same = bool(j.array_equal(x, y))
        else:same = bool(j.array_equal(jax.lax.bitcast_convert_type(x, j.uint8) if x.dtype.itemsize == 1 else
            jax.lax.bitcast_convert_type(x, {2: j.uint16, 4: j.uint32, 8: j.uint32}[x.dtype.itemsize]),
            jax.lax.bitcast_convert_type(y, j.uint8) if y.dtype.itemsize == 1 else
            jax.lax.bitcast_convert_type(y, {2: j.uint16, 4: j.uint32, 8: j.uint32}[y.dtype.itemsize])))
        if not same:return False, f'leaf {i}'
    return True, None


def p0(args):
    import jax
    import jax.numpy as j
    from gpu_engine import v2_collector as v2, v2e_cuda
    from gpu_engine.v2_engine_profile import device_events, is_kernel
    v2e_cuda.load(args.build_dir)
    events = journal(args.reference_run);context = prepare(args)
    snapshot, params, _ = actor(args, context['plan'], events)
    backend, roster, _ = construct(context, snapshot, events, games=args.games)
    plan = context['plan'];prepared = context['prepared'];params = (params,)
    original = v2._make_tick
    def patched(*a, **k):
        tick, refill = original(*a, **k)
        def tick_rt(lanes, book, rows, roster, params):
            lanes, book, rows, record = tick(lanes, book, rows, roster, params)
            carry, _ = v2e_cuda.roundtrip(lanes.carry)
            return lanes._replace(carry=carry), book, rows, record
        return tick_rt, refill
    report = []
    for lanes in args.lanes:
        collectors = {}
        for name in ('jax', 'roundtrip'):
            v2._make_tick = patched if name == 'roundtrip' else original
            c = v2.RefillCollector(prepared.engine_tables, prepared.observation_tables, plan.model, backend.labels, roster,
                lanes=lanes, max_requests=plan.max_requests, chunk_steps=args.chunk_steps, min_lanes=lanes, rows_per_game=1, donate=False)
            begin = time.perf_counter();kernel = c.kernel(0)
            start = c.start()(roster.device);rows = v2.allocate_rows(c.row_capacity())
            compiled = kernel.lower(*start, rows, roster.device, params).compile()
            collectors[name] = (c, compiled, time.perf_counter() - begin)
            log('P0_COMPILED', lanes=lanes, variant=name, seconds=collectors[name][2])
        v2._make_tick = original
        state = {}
        for name, (c, compiled, _) in collectors.items():
            lanes_state, book = c.start()(roster.device);state[name] = [lanes_state, book, v2.allocate_rows(c.row_capacity())]
        times = {name: [] for name in collectors};equal = [];reason = None
        for chunk in range(args.chunks):
            outs = {}
            for name, (c, compiled, _) in collectors.items():
                ls, bk, rw = state[name];bk = bk._replace(rows=j.zeros((), j.int32))
                begin = time.perf_counter();out = jax.block_until_ready(compiled(ls, bk, rw, roster.device, params))
                times[name].append(time.perf_counter() - begin);state[name] = list(out[:3]);outs[name] = out[:4]
            same, why = tree_equal(outs['jax'], outs['roundtrip']);equal.append(same)
            if not same and reason is None:reason = f'chunk {chunk}: {why}'
        traces = {}
        for name, (c, compiled, _) in collectors.items():
            ls, bk, rw = state[name];bk = bk._replace(rows=j.zeros((), j.int32))
            with jax.profiler.trace(None) as session:
                out = jax.block_until_ready(compiled(ls, bk, rw, roster.device, params))
            evs, _, host = device_events(session.profile_data)
            traces[name] = dict(kernels_per_tick=sum(1 for e in evs if is_kernel(e)) / args.chunk_steps,
                touch_kernels=sum(1 for e in evs if is_kernel(e) and 'touch' in e['name']) / args.chunk_steps,
                touch_us_per_tick=sum(e['dur'] for e in evs if is_kernel(e) and 'touch' in e['name']) / 1e3 / args.chunk_steps,
                host_api=host)
        ms = {name: 1e3 * statistics.median(t[1:]) / args.chunk_steps for name, t in times.items()}
        entry = dict(lanes=lanes, chunks=args.chunks, all_chunks_bitwise_equal=all(equal), first_difference=reason,
            ms_per_tick=ms, overhead_ms_per_tick=ms['roundtrip'] - ms['jax'], traces=traces,
            compile_seconds={k: v[2] for k, v in collectors.items()})
        log('P0', **entry);report.append(entry)
    return report


def p2(args):
    import jax
    import jax.numpy as j
    from gpu_engine import transition, v2e_cuda
    from gpu_engine.state import empty_state
    v2e_cuda.load(args.build_dir)
    names = ('draws', 'phazing', 'secondary', 'sleep_talk', 'entry_ties', 'initial_tie')
    # The appended `effect` field (fidelity R, design 2.1) is not drawn by the hand-written CUDA kernel yet
    # (deferred to CUDA option A, design 13 item 6): every lane counts as differing on it until it is.
    reference = jax.jit(lambda st: (transition.sample_chance_inputs(st), jax.vmap(lambda k: jax.random.split(k, 2))(st.keys)[:, 0]))
    cuda = jax.jit(v2e_cuda.chance)
    rng = np.random.default_rng(args.seed);b = args.lanes_per_batch;total = 0;bad = {n: 0 for n in names + ('next_keys',)}
    edge = np.array([0, 1, 2, 0x7FFFFFFF, 0x80000000, 0xFFFFFFFE, 0xFFFFFFFF, 0x1BD11BDA, 271828, 314159, 982451653], np.uint64)
    edge_counts = np.array([np.iinfo(np.int32).min, -1, 0, 1, 2, 63, 64, np.iinfo(np.int32).max], np.int64)
    for batch in range(args.batches):
        keys = rng.integers(0, 1 << 32, (b, 2), dtype=np.uint64)
        counts = rng.integers(-2**31, 2**31, b, dtype=np.int64)
        if batch == 0:
            pairs = np.array([(x, y) for x in edge for y in edge], np.uint64)
            n = len(pairs) * len(edge_counts)
            keys[:n] = np.repeat(pairs, len(edge_counts), axis=0);counts[:n] = np.tile(edge_counts, len(pairs))
        keys = j.asarray(keys.astype(np.uint32));counts = j.asarray(counts.astype(np.int32))
        st = empty_state(b)._replace(keys=keys, replacement_count=counts)
        ref, ref_keys = reference(st);got = cuda(keys, counts)
        for n, g in zip(names, got[:6]):bad[n] += int(j.sum(j.any((getattr(ref, n) != g).reshape(b, -1), axis=1)))
        bad['next_keys'] += int(j.sum(j.any(ref_keys != got[6], axis=1)))
        bad['effect'] = bad.get('effect', 0) + (b if len(got) < 8 else int(j.sum(j.any((ref.effect != got[7]).reshape(b, -1), axis=1))))
        total += b
    draws = total * (24 + 2 * 4 + 1)
    begin = time.perf_counter()
    for _ in range(20):jax.block_until_ready(cuda(keys, counts))
    cuda_ms = (time.perf_counter() - begin) / 20 * 1e3
    begin = time.perf_counter()
    for _ in range(20):jax.block_until_ready(reference(st))
    jax_ms = (time.perf_counter() - begin) / 20 * 1e3
    entry = dict(lanes=total, draws=draws, lanes_differing=bad, equal=all(v == 0 for v in bad.values()),
        timing=dict(lanes=b, cuda_ms=cuda_ms, jax_ms=jax_ms))
    log('P2', **entry)
    return entry


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('p0', 'p2'))
    parser.add_argument('--config');parser.add_argument('--reference-run')
    parser.add_argument('--out', default=os.environ.get('V2_OUT'))
    parser.add_argument('--arm', default='B', choices=('A', 'B'))
    parser.add_argument('--build-dir', default=None)
    parser.add_argument('--lanes', type=int, nargs='+', default=[128, 2048])
    parser.add_argument('--games', type=int, default=None)
    parser.add_argument('--chunk-steps', type=int, default=8)
    parser.add_argument('--chunks', type=int, default=6)
    parser.add_argument('--lanes-per-batch', type=int, default=1 << 20)
    parser.add_argument('--batches', type=int, default=4)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args(argv)
    out = Path(args.out);out.mkdir(parents=True, exist_ok=True)
    result = p0(args) if args.mode == 'p0' else p2(args)
    (out / f'{args.mode}.json').write_text(json.dumps(result, indent=1, default=str) + '\n')
    log('DONE', mode=args.mode)
    return 0


if __name__ == '__main__':raise SystemExit(main())
