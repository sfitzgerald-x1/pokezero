"""Engine acceleration, stage 0: profile one warm V2 collector chunk.

For each lane count a single-level RefillCollector (refill on, no tail shrink)
over the declared roster is compiled, warmed and timed, then one chunk is
traced with the JAX profiler. Device kernels are attributed to tick components
through named scopes added around the collector's calls (engine step, action
mask, tracker readout, encoder, actor inputs, forward, sampling keys, action
mapping); the rest is collector bookkeeping (book, row compaction, refill,
categorical sampling). Each component is also timed in isolation on the warm
lane state. Named scopes change HLO metadata only, never the computation.

All paths are command-line arguments; nothing is written but --out.
"""
import argparse
import collections
import functools
import gzip
import json
import os
from pathlib import Path
import re
import statistics
import time

import numpy as np

from gpu_engine.v2_bench_collector import log, journal, prepare, actor, construct

SCOPES = (('transition', 'step_with_public', 'C_step'), ('transition', 'action_mask', 'C_mask'),
    ('trackers', 'readout', 'C_readout'), ('encoder', 'encode', 'C_encode'), ('encoder', 'actor_inputs', 'C_inputs'),
    ('encoder', 'map_actions', 'C_map_actions'), ('v5_model', 'forward', 'C_forward'), ('v2', 'sample_keys', 'C_keys'),
    ('sampling', 'request_seats', 'C_request'), ('rollout', '_physical', 'C_physical'), ('sampling', 'select_rows', 'C_select'))


def install_scopes():
    import jax
    from gpu_engine import transition, engine_device_trackers as trackers, engine_device_observation as encoder
    from gpu_engine import v5_model, v2_collector as v2, benchmark_device_selfplay as sampling, v5_device_rollout as rollout
    modules = dict(transition=transition, trackers=trackers, encoder=encoder, v5_model=v5_model, v2=v2,
        sampling=sampling, rollout=rollout)
    def scoped(name, fn):
        @functools.wraps(fn)
        def wrapper(*a, **k):
            with jax.named_scope(name):return fn(*a, **k)
        return wrapper
    for module, attr, name in SCOPES:
        setattr(modules[module], attr, scoped(name, getattr(modules[module], attr)))


# --------------------------------------------------------------------------
# Optimized HLO index: instruction name -> opcode, shapes, scope, source.

INSTRUCTION = re.compile(r'^\s*(?:ROOT\s+)?%?([\w.\-]+)\s*=\s*(.+?)\s+([a-z][\w\-]*)\((.*)$')
SHAPE = re.compile(r'\b(pred|s8|u8|s16|u16|s32|u32|s64|u64|f16|bf16|f32|f64)\[([\d,]*)\]')
BYTES = dict(pred=1, s8=1, u8=1, s16=2, u16=2, f16=2, bf16=2, s32=4, u32=4, f32=4, s64=8, u64=8, f64=8)


def shape_bytes(text):
    total = 0
    for dtype, dims in SHAPE.findall(text):
        n = 1
        for d in dims.split(','):
            if d:n *= int(d)
        total += n * BYTES[dtype]
    return total


def scope_of(op_name):
    for part in op_name.split('/'):
        if part.startswith('C_'):return part[2:]
    return None


FRAME_SECTIONS = ('FileNames', 'FunctionNames', 'FileLocations', 'StackFrames')
LIBRARY = ('jax', 'numpy', 'runpy', 'v2_engine_profile', 'contextlib', 'functools', 'threading')
HELPERS = {'_active', '_set_active', '_get', '_set', '_write', '_select', 'select_rows', 'copy', '_seat', '_r'}


def _frames(lines):
    sect = None;files = {};funcs = {};locs = {};frames = {}
    for line in lines:
        if line in FRAME_SECTIONS:sect = line;continue
        if sect and (not line or line.startswith(('%', 'ENTRY', 'HloModule'))):sect = None
        if not sect:continue
        key, value = line.split(' ', 1);key = int(key)
        if sect == 'FileNames':files[key] = value.strip('"')
        elif sect == 'FunctionNames':funcs[key] = value.strip('"')
        else:(locs if sect == 'FileLocations' else frames)[key] = {k: int(v) for k, v in re.findall(r'(\w+)=(\d+)', value)}
    def stack(fid):
        out = [];seen = set()
        while fid in frames and fid not in seen:
            seen.add(fid);loc = locs.get(frames[fid].get('file_location_id', 0), {})
            out.append((Path(files.get(loc.get('file_name_id', 0), '?')).name, funcs.get(loc.get('function_name_id', 0), '?'), loc.get('line', 0)))
            fid = frames[fid].get('parent_frame_id', 0)
        return out
    return stack


def hlo_index(text):
    lines = text.splitlines();stack = _frames(lines)
    index = {};computation = None;counts = collections.Counter();per_computation = collections.defaultdict(collections.Counter)
    for line in lines:
        stripped = line.strip()
        if not stripped or stripped.startswith('//'):continue
        if not line.startswith(' ') and stripped.endswith('{'):
            computation = stripped.split()[1 if stripped.startswith('ENTRY') else 0].lstrip('%');continue
        m = INSTRUCTION.match(line)
        if not m:continue
        name, shape, opcode, rest = m.groups()
        op_name = re.search(r'op_name="([^"]*)"', rest);frame = re.search(r'stack_frame_id=(\d+)', rest)
        kind = re.search(r'kind=(k\w+)', rest)
        engine = [f for f in (stack(int(frame.group(1))) if frame else []) if not any(x in f[0] for x in LIBRARY)]
        caller = next((f for f in engine if f[1].split('.')[-1] not in HELPERS), engine[0] if engine else None)
        operands = rest.split('),', 1)[0] if '),' in rest else rest
        index[name] = dict(opcode=opcode, kind=kind.group(1) if kind else None, computation=computation,
            scope=scope_of(op_name.group(1)) if op_name else None, op_name=op_name.group(1)[-120:] if op_name else None,
            source=f'{engine[0][0]}:{engine[0][2]} {engine[0][1]}' if engine else None,
            module=caller[0] if caller else None, function=f'{caller[0]}:{caller[1]}' if caller else None,
            out_bytes=shape_bytes(shape), in_bytes=shape_bytes(operands))
        per_computation[computation][opcode] += 1;counts[opcode] += 1
    return index, counts, per_computation


def normalize(name):
    return re.sub(r'[.\-]', '_', name)


# --------------------------------------------------------------------------
# Trace analysis.

def device_events(profile_data):
    planes = list(profile_data.planes);events = [];lines = {}
    for plane in planes:
        if not plane.name.startswith('/device:GPU'):continue
        for line in plane.lines:
            evs = list(line.events);lines[f'{plane.name}|{line.name}'] = len(evs)
            for ev in evs:
                stats = {}
                for item in ev.stats:
                    try:k, v = item
                    except (TypeError, ValueError):continue
                    stats[k] = v
                if not stats:
                    try:stats = dict(ev.stats)
                    except Exception:pass
                events.append(dict(name=ev.name, start=ev.start_ns, end=ev.end_ns, dur=ev.duration_ns, line=line.name, stats=stats))
    host = collections.Counter()
    for plane in planes:
        if plane.name.startswith('/host'):
            for line in plane.lines:
                for ev in line.events:
                    if ev.name.startswith(('cuLaunch', 'cudaLaunch', 'cuGraphLaunch', 'cudaGraphLaunch', 'cuMemcpy', 'cuStreamSynchronize', 'cuEvent')):
                        host[ev.name] += 1
    return events, lines, dict(host)


def is_kernel(event):
    line = event['line'].lower();name = event['name'].lower()
    if not line.startswith('stream'):return False
    return not name.startswith(('memcpy', 'memset')) and 'memcpy' not in name and 'memset' not in name


def union_ns(intervals):
    total = 0;end = None;start = None
    for s, e in sorted(intervals):
        if end is None or s > end:
            if end is not None:total += end - start
            start, end = s, e
        else:end = max(end, e)
    if end is not None:total += end - start
    return total


def analyze(events, index, steps, *, wall_ns, top=20):
    kernels = [e for e in events if is_kernel(e)]
    other = [e for e in events if not is_kernel(e)]
    by_norm = {normalize(k): k for k in index}
    def lookup(e):
        for key in ('hlo_op', 'tf_op', 'name'):
            value = e['stats'].get(key) if key != 'name' else e['name']
            if not value:continue
            value = str(value).split(':')[-1].strip().lstrip('%')
            if value in index:return value
            if normalize(value) in by_norm:return by_norm[normalize(value)]
        return None
    hist = collections.Counter();bins = (2_000, 5_000, 10_000, 50_000, 200_000, 1_000_000)
    labels = ('<2us', '2-5us', '5-10us', '10-50us', '50-200us', '200us-1ms', '>1ms')
    comp_time = collections.Counter();comp_count = collections.Counter()
    per_op = collections.defaultdict(lambda: dict(count=0, ns=0))
    mod_time = collections.Counter();mod_count = collections.Counter();fn_time = collections.Counter();fn_count = collections.Counter()
    for e in kernels:
        hist[labels[next((i for i, b in enumerate(bins) if e['dur'] < b), len(bins))]] += 1
        name = lookup(e);info = index.get(name, {}) if name else {}
        scope = info.get('scope') or ('collector' if name else 'unmapped')
        comp_time[scope] += e['dur'];comp_count[scope] += 1
        mod = f"{scope}/{info.get('module')}";mod_time[mod] += e['dur'];mod_count[mod] += 1
        fn = info.get('function') or '?';fn_time[fn] += e['dur'];fn_count[fn] += 1
        op = per_op[name or e['name']];op['count'] += 1;op['ns'] += e['dur'];op['scope'] = scope;op['info'] = info
        op['kernel'] = e['name'][:80]
    starts = [e['start'] for e in kernels];ends = [e['end'] for e in kernels]
    span = (max(ends) - min(starts)) if kernels else 0
    busy = union_ns([(e['start'], e['end']) for e in kernels])
    gaps = []
    srt = sorted(kernels, key=lambda e: e['start'])
    for a, b in zip(srt, srt[1:]):gaps.append(max(0, b['start'] - a['end']))
    ranked = sorted(per_op.items(), key=lambda kv: -kv[1]['ns'])[:top]
    total_ns = sum(e['dur'] for e in kernels)
    return dict(kernels=len(kernels), kernels_per_tick=len(kernels) / steps, other_events=len(other),
        other_names=dict(collections.Counter(e['name'][:40] for e in other).most_common(8)),
        kernel_ns_sum=total_ns, gpu_busy_ns=busy, kernel_span_ns=span, wall_ns=wall_ns,
        busy_fraction_of_span=busy / span if span else None, busy_fraction_of_wall=busy / wall_ns if wall_ns else None,
        gap_ns_median=statistics.median(gaps) if gaps else None, gap_ns_sum=sum(gaps),
        kernel_ns_median=statistics.median([e['dur'] for e in kernels]) if kernels else None,
        histogram={k: hist.get(k, 0) for k in labels},
        mapped_kernels=sum(c for s, c in comp_count.items() if s != 'unmapped'),
        components={s: dict(kernels=comp_count[s], kernels_per_tick=comp_count[s] / steps, ms_per_tick=comp_time[s] / steps / 1e6,
            share=comp_time[s] / total_ns if total_ns else None) for s in sorted(comp_time, key=lambda s: -comp_time[s])},
        modules={m: dict(kernels_per_tick=mod_count[m] / steps, ms_per_tick=mod_time[m] / steps / 1e6)
            for m in sorted(mod_time, key=lambda m: -mod_time[m])[:25]},
        functions={f: dict(kernels_per_tick=fn_count[f] / steps, ms_per_tick=fn_time[f] / steps / 1e6)
            for f in sorted(fn_time, key=lambda f: -fn_time[f])[:30]},
        durations_us=dict(zip(('p10', 'p50', 'p90', 'p99'), (np.percentile([e['dur'] for e in kernels], (10, 50, 90, 99)) / 1e3).round(3).tolist())) if kernels else None,
        top=[dict(op=name, kernel=v['kernel'], scope=v['scope'], count=v['count'], us_total=v['ns'] / 1e3,
            us_per_tick=v['ns'] / steps / 1e3, us_mean=v['ns'] / v['count'] / 1e3, opcode=v['info'].get('opcode'),
            kind=v['info'].get('kind'), source=v['info'].get('source'), function=v['info'].get('function'),
            bytes=v['info'].get('in_bytes', 0) + v['info'].get('out_bytes', 0),
            gbps=(v['info'].get('in_bytes', 0) + v['info'].get('out_bytes', 0)) / (v['ns'] / v['count']) if v['info'] else None)
            for name, v in ranked])


def stats_sample(events, k=3):
    return [dict(name=e['name'][:80], line=e['line'], dur=e['dur'], stats={a: str(b)[:120] for a, b in e['stats'].items()}) for e in events[:k]]


# --------------------------------------------------------------------------

def timed(fn, *args, repeats=20, warm=3):
    import jax
    for _ in range(warm):jax.block_until_ready(fn(*args))
    times = []
    for _ in range(repeats):
        begin = time.perf_counter();jax.block_until_ready(fn(*args));times.append(time.perf_counter() - begin)
    return dict(ms_median=1e3 * statistics.median(times), ms_min=1e3 * min(times))


def trace_call(fn, *args, steps=1, index=None, calls=3):
    import jax
    jax.block_until_ready(fn(*args))
    with jax.profiler.trace(None) as session:
        begin = time.perf_counter()
        for _ in range(calls):jax.block_until_ready(fn(*args))
        wall = time.perf_counter() - begin
    events, lines, host = device_events(session.profile_data)
    kernels = [e for e in events if is_kernel(e)]
    busy = union_ns([(e['start'], e['end']) for e in kernels])
    return dict(kernels_per_call=len(kernels) / calls, kernel_ms_per_call=sum(e['dur'] for e in kernels) / calls / 1e6,
        busy_ms_per_call=busy / calls / 1e6, wall_ms_per_call=wall / calls * 1e3, host_api=host,
        **({'analysis': analyze(events, index, steps * calls, wall_ns=wall * 1e9, top=10)} if index is not None else {}))


def components(collector, lanes_state, params, out, lanes):
    """Isolated per-component timings on the warm lane state (one tick's work)."""
    import jax
    import jax.numpy as j
    from gpu_engine import transition, engine_device_trackers as trackers, engine_device_observation as encoder, v5_model
    from gpu_engine.v5_device_rollout import RUNNING
    et, ot, cfg = collector.engine_tables, collector.observation_tables, collector.model_config
    c = lanes_state.carry;b = len(c.streams)
    state = c.state._replace(done=c.state.done | (c.status != RUNNING))
    side = lambda s: j.full(b, s, j.int32)
    f_mask = jax.jit(lambda st: transition.action_mask(st, et))
    f_readout = jax.jit(lambda pub: tuple(trackers.readout(pub, side(s)) for s in range(2)))
    f_encode = jax.jit(lambda st, views, m: tuple(encoder.encode(st, views[s], ot, observer_sides=side(s), legal_action_mask=m[:, s]) for s in range(2)))
    def inputs_of(obs):
        return jax.tree.map(lambda *xs: j.stack(xs, axis=1).reshape((2*b,) + xs[0].shape[1:]), *(encoder.actor_inputs(o) for o in obs))
    f_inputs = jax.jit(inputs_of)
    f_forward = jax.jit(lambda p, x: v5_model.forward(p, x, cfg, compute_dtype='bfloat16'))
    def pick(obs, st, key):
        legal = j.stack([o.legal_action_mask for o in obs], axis=1)
        logits = j.where(legal, jax.random.uniform(key, legal.shape), -j.inf)
        acts = j.argmax(logits, axis=-1).astype(j.int32)
        return j.stack([encoder.map_actions(obs[s], st, side(s), acts[:, s]) for s in range(2)], axis=1)
    f_pick = jax.jit(pick)
    f_step = jax.jit(lambda st, a, pub: transition.step_with_public(st, a, et, pub, belief_tables=ot.belief))
    masks = f_mask(state);views = f_readout(c.public);obs = f_encode(state, views, masks);inputs = f_inputs(obs)
    actions = f_pick(obs, state, jax.random.PRNGKey(0))
    result = {}
    for name, fn, args in (('mask', f_mask, (state,)), ('readout_x2', f_readout, (c.public,)), ('encode_x2', f_encode, (state, views, masks)),
            ('forward', f_forward, (params[0], inputs)), ('step', f_step, (state, actions, c.public))):
        begin = time.perf_counter();jax.block_until_ready(fn(*args));first = time.perf_counter() - begin
        entry = timed(fn, *args);entry['first_call_s'] = first
        try:entry.update(trace_call(fn, *args))
        except Exception as error:entry['trace_error'] = repr(error)[:300]
        result[name] = entry;log('COMPONENT', lanes=lanes, component=name, **{k: v for k, v in entry.items() if k != 'host_api'})
    return result


def profile_lanes(args, context, backend, roster, params, lanes, out):
    import jax
    import jax.numpy as j
    from gpu_engine import v2_collector as v2
    plan = context['plan'];prepared = context['prepared'];steps = args.chunk_steps
    collector = v2.RefillCollector(prepared.engine_tables, prepared.observation_tables, plan.model, backend.labels, roster,
        lanes=lanes, max_requests=plan.max_requests, chunk_steps=steps, min_lanes=lanes, rows_per_game=1)
    assert len(collector.levels) == 1
    device = roster.device;capacity = collector.row_capacity()
    lanes_state, book = collector.start()(device);rows = v2.allocate_rows(capacity)
    begin = time.perf_counter();lowered = collector.kernel(0).lower(lanes_state, book, rows, device, params);lower_s = time.perf_counter() - begin
    begin = time.perf_counter();compiled = lowered.compile();compile_s = time.perf_counter() - begin
    text = compiled.as_text();index, opcodes, per_computation = hlo_index(text)
    with gzip.open(out / f'hlo_chunk_{lanes}.txt.gz', 'wt') as f:f.write(text)
    try:cost = compiled.cost_analysis();cost = cost[0] if isinstance(cost, (list, tuple)) else cost
    except Exception as error:cost = dict(error=repr(error))
    cost = {k: float(v) for k, v in (cost or {}).items() if isinstance(v, (int, float)) and (k in ('flops', 'bytes accessed', 'transcendentals', 'optimal_seconds') or k.startswith('bytes accessed'))}
    heavy = ('fusion', 'custom-call', 'copy', 'gather', 'scatter', 'dynamic-update-slice', 'dynamic-slice', 'sort', 'reduce', 'convolution', 'dot', 'while', 'conditional', 'call')
    computations = {c: {k: v for k, v in cnt.items() if k in heavy} for c, cnt in per_computation.items() if not c.startswith(('fused_', 'fused', 'region'))}
    hlo = dict(instructions=sum(opcodes.values()), opcodes={k: opcodes[k] for k in heavy if k in opcodes},
        launching_computations={c: v for c, v in sorted(computations.items(), key=lambda kv: -sum(kv[1].values()))[:12]},
        fusion_kinds=dict(collections.Counter(v['kind'] for v in index.values() if v['opcode'] == 'fusion')),
        command_buffer_calls=sum(1 for v in index.values() if v['opcode'] == 'call' and 'command_buffer' in (v.get('op_name') or '')),
        text_bytes=len(text))
    log('COMPILED_PROFILE', lanes=lanes, lower_s=lower_s, compile_s=compile_s, cost=cost, **{k: v for k, v in hlo.items() if k != 'launching_computations'})
    log('HLO_COMPUTATIONS', lanes=lanes, computations=hlo['launching_computations'])
    state = dict(lanes=lanes_state, book=book, rows=rows)
    def chunk():
        st = state;bk = st['book']._replace(rows=j.zeros((), j.int32))
        a, bb, r, o, _ = compiled(st['lanes'], bk, st['rows'], device, params)
        st.update(lanes=a, book=bb, rows=r);return jax.device_get(o)
    for _ in range(args.warm_chunks):chunk()
    times = []
    for _ in range(args.timed_chunks):
        begin = time.perf_counter();live, cursor, _ = chunk();times.append(time.perf_counter() - begin)
    live_lanes = int(np.asarray(live).sum())
    timing = dict(ms_per_tick_median=1e3 * statistics.median(times) / steps, ms_per_tick_min=1e3 * min(times) / steps,
        chunks=len(times), live_lanes_after=live_lanes, started=int(np.asarray(cursor).sum()))
    log('TIMED', lanes=lanes, **timing)
    traces = []
    for t in range(args.trace_chunks):
        options = jax.profiler.ProfileOptions()
        try:options.python_tracer_level = 0;options.host_tracer_level = 2
        except Exception:pass
        with jax.profiler.trace(str(out / f'trace_{lanes}') if args.save_trace and t == 0 else None, profiler_options=options) as session:
            begin = time.perf_counter();chunk();wall = time.perf_counter() - begin
        events, lines, host = device_events(session.profile_data)
        if t == 0:
            log('TRACE_LINES', lanes=lanes, lines=lines, host_api=host)
            log('TRACE_SAMPLE', lanes=lanes, sample=stats_sample([e for e in events if is_kernel(e)]))
        analysis = analyze(events, index, steps, wall_ns=wall * 1e9, top=args.top)
        analysis['host_api'] = host;traces.append(analysis)
        log('TRACE', lanes=lanes, trace=t, **{k: v for k, v in analysis.items() if k not in ('top', 'components', 'modules', 'functions')})
        log('TRACE_COMPONENTS', lanes=lanes, trace=t, components=analysis['components'])
        if t == 0:log('TRACE_MODULES', lanes=lanes, modules=analysis['modules']);log('TRACE_FUNCTIONS', lanes=lanes, functions=analysis['functions'])
        if t == 0:
            for row in analysis['top']:log('TOP', lanes=lanes, **row)
    comp = components(collector, state['lanes'], params, out, lanes) if args.components else None
    return dict(lanes=lanes, chunk_steps=steps, lower_s=lower_s, compile_s=compile_s, cost=cost, hlo=hlo, timing=timing,
        traces=traces, components=comp)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--reference-run', required=True)
    parser.add_argument('--out', default=os.environ.get('V2_OUT'))
    parser.add_argument('--arm', default='B', choices=('A', 'B'))
    parser.add_argument('--lanes', type=int, nargs='+', default=[128, 2048])
    parser.add_argument('--games', type=int, default=None)
    parser.add_argument('--chunk-steps', type=int, default=8)
    parser.add_argument('--warm-chunks', type=int, default=3)
    parser.add_argument('--timed-chunks', type=int, default=6)
    parser.add_argument('--trace-chunks', type=int, default=2)
    parser.add_argument('--top', type=int, default=25)
    parser.add_argument('--save-trace', action='store_true')
    parser.add_argument('--components', action='store_true')
    parser.add_argument('--no-scopes', action='store_true')
    parser.add_argument('--frames', type=int, default=40, help='traceback frames kept in HLO locations (attribution depth)')
    parser.add_argument('--xla-flags', default=None, help='appended to XLA_FLAGS before JAX initializes')
    args = parser.parse_args(argv)
    if not args.out:raise ValueError('explicit output directory required')
    out = Path(args.out);out.mkdir(parents=True, exist_ok=True)
    if args.xla_flags:os.environ['XLA_FLAGS'] = (os.environ.get('XLA_FLAGS', '') + ' ' + args.xla_flags).strip()
    import jax
    jax.config.update('jax_traceback_in_locations_limit', args.frames)
    if not args.no_scopes:install_scopes()
    events = journal(args.reference_run)
    context = prepare(args)
    import jax
    log('DEVICE', devices=[str(d) for d in jax.devices()], kind=jax.devices()[0].device_kind, xla_flags=os.environ.get('XLA_FLAGS'))
    snapshot, params, _ = actor(args, context['plan'], events)
    backend, roster, construction = construct(context, snapshot, events, games=args.games)
    log('ROSTER', games=roster.games)
    report = dict(schema='v2-engine-profile.v1', source_commit=os.environ.get('V2_SOURCE_COMMIT'), xla_flags=os.environ.get('XLA_FLAGS'),
        device_kind=jax.devices()[0].device_kind, games=roster.games, runs=[])
    for lanes in args.lanes:
        report['runs'].append(profile_lanes(args, context, backend, roster, (params,), lanes, out))
        (out / 'profile.json').write_text(json.dumps(report, indent=1, default=str) + '\n')
    log('DONE', report=str(out / 'profile.json'))
    return 0


if __name__ == '__main__':raise SystemExit(main())
