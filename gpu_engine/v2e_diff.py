"""Differential harness: a JAX oracle module against its CUDA port, bitwise.

Replays captured real engine-step inputs (gpu_engine.v2e_capture shards) and
optional synthetic lanes through both implementations of one module and
compares every output leaf bit for bit, per lane. Reports lane-steps compared,
differing lane-steps per output leaf, and the first differing (leaf, lane);
the first differing lanes' inputs are written as a repro shard. The report
records a digest of the oracle's imported engine sources, so a re-run against
a newer JAX commit is distinguishable.

Modules are registered in MODULES; each maps (state, actions, trackers,
context) to outputs for both sides. Paths are command-line arguments.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import numpy as np

from gpu_engine.v2_bench_collector import log, journal, prepare, actor, construct


def oracle_digest():
    """sha256 over every imported gpu_engine module source, excluding the CUDA side and tools."""
    h = hashlib.sha256();names = []
    for name, module in sorted(sys.modules.items()):
        path = getattr(module, '__file__', None)
        if not name.startswith('gpu_engine.') or not path or name.startswith(('gpu_engine.v2e_', 'gpu_engine.v2_')):continue
        names.append(name);h.update(name.encode());h.update(Path(path).read_bytes())
    return h.hexdigest(), names


def unsigned(x):
    import jax
    import jax.numpy as j
    if x.dtype == j.bool_:return x
    width = {1: j.uint8, 2: j.uint16, 4: j.uint32}[x.dtype.itemsize]
    return x if x.dtype == width else jax.lax.bitcast_convert_type(x, width)


def lane_differences(a, b):
    """Per output leaf: bool [L] lanes whose bits differ (structure must match)."""
    import jax
    import jax.numpy as j
    pa, ta = jax.tree_util.tree_flatten_with_path(a);pb, tb = jax.tree_util.tree_flatten_with_path(b)
    if ta != tb:raise ValueError(f'output structures differ: {ta} vs {tb}')
    out = {}
    for (path, x), (_, y) in zip(pa, pb):
        if x.shape != y.shape or x.dtype != y.dtype:raise ValueError(f'{jax.tree_util.keystr(path)}: {x.shape}{x.dtype} vs {y.shape}{y.dtype}')
        d = unsigned(x) != unsigned(y)
        out[jax.tree_util.keystr(path)] = j.any(d.reshape(d.shape[0], -1), axis=1) if d.ndim else d[None]
    return out


# ---------------------------------------------------------------------------
# Modules: (jax_fn, cuda_fn) over (state, actions, trackers, context).

def _identity():
    from gpu_engine import v2e_cuda
    return (lambda s, a, t, c: (s, t)), (lambda s, a, t, c: v2e_cuda.roundtrip((s, t))[0])


def _chance():
    import jax
    from gpu_engine import transition, v2e_cuda
    def reference(s, a, t, c):
        x = transition.sample_chance_inputs(s)
        return (x.draws, x.phazing, x.secondary, x.sleep_talk, x.entry_ties, x.initial_tie,
            jax.vmap(lambda k: jax.random.split(k, 2))(s.keys)[:, 0])
    return reference, (lambda s, a, t, c: v2e_cuda.chance(s.keys, s.replacement_count))


def _negative_control():
    """Harness self-check: the CUDA side flips one bit of state.hp at lane 3, seat 1, slot 2."""
    from gpu_engine import v2e_cuda
    def flipped(s, a, t, c):
        s2, t2 = v2e_cuda.roundtrip((s, t))[0]
        return s2._replace(hp=s2.hp.at[3, 1, 2].set(s2.hp[3, 1, 2] ^ 1)), t2
    return (lambda s, a, t, c: (s, t)), flipped


def _observe(ctx):
    from gpu_engine import v2e_gen, v2e_modules
    fn = v2e_modules.observe(ctx['engine_tables'], ctx['observation_tables'])
    state, _, public = ctx['template']
    generated = v2e_gen.compile_function('observe', fn, (state, public))
    log('GENERATED', module='observe', arena_words=generated.module.arena_words, lane_literals=generated.module.lane_literals,
        source_lines=generated.module.source_lines, consts=len(generated.module.consts), library=str(generated.path))
    def cuda(s, a, t, c):
        out, err = generated(s, t)
        return out, err
    def reference(s, a, t, c):
        return fn(s, t), j_zero_err()
    return reference, cuda


def j_zero_err():
    import jax.numpy as j
    return j.zeros((1,), j.uint32)


MODULES = dict(identity=_identity, chance=_chance, negative_control=_negative_control, observe=_observe)
GENERATED = {'observe'}


# ---------------------------------------------------------------------------

def load_shard(path):
    with np.load(path) as z:
        leaves = [z[k] for k in sorted(k for k in z.files if k[0] == 'l' and k[1:].isdigit())]
        return leaves, int(z['tick']), int(z['level'])


def run(module, shard_paths, template, context, *, out, workers=8, repro_lanes=16):
    import jax
    jax_fn, cuda_fn = MODULES[module](dict(context, template=template)) if module in GENERATED else MODULES[module]()
    treedef = jax.tree.structure(template)
    jitted = {}
    def call(fn_name, fn, *args):
        key = (fn_name, args[0].hp.shape[0])
        if key not in jitted:jitted[key] = jax.jit(lambda s, a, t: fn(s, a, t, context))
        return jitted[key](*args)
    totals = {};compared = 0;differing = 0;first = None;begin = time.perf_counter()
    def prefetched(paths):
        # Bounded read-ahead: at most `workers` shards in flight.
        with ThreadPoolExecutor(max_workers=workers) as pool:
            pending = [pool.submit(load_shard, p) for p in paths[:workers]]
            for i, path in enumerate(paths):
                loaded = pending[i].result();pending[i] = None
                if i + workers < len(paths):pending.append(pool.submit(load_shard, paths[i + workers]))
                yield path, loaded
    if True:
        for path, (leaves, tick, level) in prefetched(list(shard_paths)):
            state, actions, trackers = jax.tree.unflatten(treedef, [jax.numpy.asarray(x) for x in leaves])
            want = call('jax', jax_fn, state, actions, trackers);got = call('cuda', cuda_fn, state, actions, trackers)
            diffs = lane_differences(want, got);lanes = state.hp.shape[0];compared += lanes
            any_lane = np.zeros(lanes, bool)
            for name, d in diffs.items():
                d = np.asarray(d);totals[name] = totals.get(name, 0) + int(d.sum());any_lane |= d if d.shape == any_lane.shape else d.any()
            differing += int(any_lane.sum())
            if any_lane.any() and first is None:
                lane = int(np.flatnonzero(any_lane)[0]);leaf = next(n for n, d in diffs.items() if np.asarray(d).reshape(-1)[min(lane, np.asarray(d).size - 1)])
                first = dict(shard=str(path), tick=tick, lane=lane, leaf=leaf)
                rows = np.flatnonzero(any_lane)[:repro_lanes]
                np.savez_compressed(Path(out) / f'repro-{module}.npz', lanes=rows, **{f'l{i:03d}': x[rows] for i, x in enumerate(leaves)})
                log('FIRST_DIFFERENCE', module=module, **first)
    seconds = time.perf_counter() - begin
    return dict(module=module, shards=len(shard_paths), lane_steps=compared, lane_steps_differing=differing,
        equal=differing == 0, differing_by_leaf={k: v for k, v in totals.items() if v}, output_leaves=len(totals),
        first_difference=first, seconds=seconds)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('modules', nargs='+', choices=tuple(MODULES))
    parser.add_argument('--config', required=True)
    parser.add_argument('--reference-run', required=True)
    parser.add_argument('--shards', required=True, help='directory of capture shards')
    parser.add_argument('--out', default=os.environ.get('V2_OUT'))
    parser.add_argument('--arm', default='B', choices=('A', 'B'))
    parser.add_argument('--limit', type=int, default=0, help='first N shards (0: all)')
    parser.add_argument('--build-dir', default=None)
    args = parser.parse_args(argv)
    out = Path(args.out);out.mkdir(parents=True, exist_ok=True)
    import jax
    import jax.numpy as j
    from gpu_engine import v2e_cuda
    from gpu_engine.v2e_capture import leaf_paths
    v2e_cuda.load(args.build_dir)
    events = journal(args.reference_run);context = prepare(args)
    snapshot, params, _ = actor(args, context['plan'], events)
    backend, roster, _ = construct(context, snapshot, events, games=2048)
    first = lambda tree: jax.tree.map(lambda x: x[:1], tree)
    template = (first(roster.device.state), j.zeros((1, 2), j.int32), first(roster.device.public))
    capture = json.loads(Path(args.shards).parent.joinpath('capture.json').read_text())
    if capture['paths'] != leaf_paths(template):raise ValueError('capture leaf paths differ from the oracle pytrees')
    shards = sorted(Path(args.shards).glob('shard-*.npz'))
    if args.limit:shards = shards[:args.limit]
    digest, modules = oracle_digest()
    ctx = dict(engine_tables=context['prepared'].engine_tables, observation_tables=context['prepared'].observation_tables)
    report = dict(oracle_digest=digest, oracle_modules=len(modules), source_commit=os.environ.get('V2_SOURCE_COMMIT'),
        capture_commit=capture.get('source_commit'), cuda_library=str(v2e_cuda._loaded['path']), results=[])
    for module in args.modules:
        result = run(module, shards, template, ctx, out=out);report['results'].append(result);log('DIFF', **result)
        (out / 'diff.json').write_text(json.dumps(report, indent=1, default=str) + '\n')
    log('DONE', equal=all(r['equal'] for r in report['results']))
    return 0 if all(r['equal'] for r in report['results']) else 1


if __name__ == '__main__':raise SystemExit(main())
