"""Option-A first GPU measurement (design note section 5): the engine step's time by phase, generated and JAX.

Phases are v2e_census.PHASES (design section 1.1), the one line table every subcommand uses. Each subcommand writes
JSON into a new --out directory; every path is an argument. Shards are gpu_engine.v2e_capture shards (--shards is the
shards/ directory beside capture.json, checked against the oracle pytrees as gpu_engine.v2e_diff checks them); tables
and the template load from --config/--reference-run as v2e_diff loads them.

- marks (step 2.1): compile_function('step', ..., profile=True, mode='thread') and run --limit shards at each --lanes.
  Each profile mark is lane 0's clock64 after one equation that emitted code, relative to its chunk kernel's start; a
  mark's cycles are its delta from the previous mark of the same chunk. Cycles are summed per phase (with the action
  iteration and residual event), per group A..G, per group x category and module, and per chunk (marks.json, marks.npz).
- walls (step 2.2): the normal generated step (thread mode; the cached .so when --build-dir holds it) at --lanes
  (default 128, 2,048, 8,192; a lane count above the largest shard concatenates consecutive shards: 8,192 = four
  2,048-lane shards): per-call wall of --calls warm calls, the jitted JAX oracle on the same inputs (--no-jax skips it)
  and, unless --no-trace, per-chunk kernel times from a JAX profiler (CUPTI) trace of --trace-calls calls. Prints the
  nsys command to run around it (with --no-trace: two CUPTI clients do not mix).
- ncu-targets / ncu-driver (step 4): ncu-targets prints the ncu kernel regex for the heaviest chunk of each of the
  --top phase groups (from marks at --lanes), plus chance_kernel and touch_kernel; ncu-driver makes one call of the
  normal step and (--hand-kernels) of the P2 chance and P0 round-trip kernels at --lanes, for `ncu --set full` to wrap.
- jax-attrib (step 3): offline, from a `gpu_engine.v2_engine_profile --save-trace --frames 40` output directory: device
  kernel time per tick of the step scope (C_step) attributed to phases by the outermost transition.step_with_chance
  frame in each kernel's HLO stack frames (v2_engine_profile's kernel lookup; frames through hlo_stacks, which corrects jax
  0.12's parent-id offset). A fused kernel is charged to its root op's phase; the step time in fusions spanning more
  than one phase is reported beside it.
- divergence (step 5, CPU): per lane of --limit shards, the request kind, each side's action kind, the chosen move's
  effect flags (the move-indexed columns of the engine tables), the active statuses, abilities and items; per phase
  group the 32-lane union of path classes (distinct classes per warp, union vs lane feature counts) and a union factor
  under a block cost model weighted by the census (features equal weight), at uniform share = the group's census hook
  share and at the design's 0.7.
- pricer-split: the live DepthN pricer (depthn_live_search.make_pricers at bucket 1: capacity 1 x W/2 = 8, depth 2, k 9,
  k1 9/9, prior, block_lanes 2187, blocks_per_call 16, the champion) split into the leaf (turn-2) forward and the engine
  work by gpu_engine.d3_depth2_throughput's --profile-split method: (i) the normal call, (ii) the same pricer with every
  leaf forward replaced by zeros of its output shapes (constant_turn2_forward, recognized by its static batch size),
  (iii) D3's depth-1 FramePricer; forward = (i) - (ii), engine (stepping, observing, P1's extra work such as the
  opponent-view child-prior forward) = (ii) - (iii); and (iv) every forward zeroed, P1's included: engine total = (iv),
  all forwards = (i) - (iv). On the D3 bank chunk's saved roots (d3_depthn_throughput.prepare: the gate's request builder
  and champion binding); the bank executor's table digest and every variant's program keys are recorded, and the tables
  are compared with the live executor's when --live-metadata or --live-tables-digest is given.
- report: merges the outputs into phase_attribution.json and one markdown table (phase x device lines, equations,
  generated ms at 128 and 2,048, JAX ms at 128 and 2,048, hook share, union factor; plus the live pricer split line)
  and applies the section 5 decision rule (recommended M2 and M4). Generated ms per phase split each chunk's kernel time by
  its mark cycles; the profile build chunks exactly as the plain build (v2e_j2c excludes mark lines from the chunk size),
  and a chunk mismatch falls back to step-wide cycle shares with a warning, never silently.
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
import sys
import time
from types import SimpleNamespace

import numpy as np

from gpu_engine import v2e_census as cz

SCHEMA = 'pokezero.v2e-phase-profile.v1'
CHUNK = re.compile(r'chunk_(\d+)(?!\d)')   # demangled (chunk_12(...)) or mangled (8chunk_12ENS_6ParamsE) kernel names
MARK_WRITE = re.compile(r'err\[1 \+ (\d+)\]')
HAND_KERNELS = ('chance_kernel', 'touch_kernel')
PRIOR_GENERATED_MS_2048 = 9.28      # design section 5: generated step at 2,048 lanes (P4)
PRIOR_OBSERVE_MS_2048 = 4.42        # generated observe at 2,048 lanes (P3)
FG_MIN_SHARE, E_SPLIT_SHARE = .15, .40
M6_STATUS = 'not evaluated: observe is not measured in this job'
DESIGN_UNIFORM_SHARE = .7          # design section 3.1: the tracker hooks are about 10k of the ~14-15k typical ops


def log(message, **values):
    print(json.dumps(dict(message=message, time=time.strftime('%H:%M:%S'), **values), default=str), flush=True)


def new_out(path):
    out = Path(path);out.mkdir(parents=True, exist_ok=False)
    return out


def write_json(path, data):
    Path(path).write_text(json.dumps(data, indent=1, default=str) + '\n')


def pct(values, q):
    return float(np.percentile(np.asarray(values, float), q)) if len(values) else None


# ---------------------------------------------------------------------------
# Shards and the generated step.

@functools.lru_cache(maxsize=None)
def shard_lanes(path):
    with np.load(path) as z:return int(z['l000'].shape[0])


def shards_with(paths, lanes, count):
    """The first `count` shards with at least `lanes` lanes (their own order)."""
    out = []
    for path in paths:
        if len(out) >= count:break
        if shard_lanes(path) >= lanes:out.append(path)
    return out


def batch_for(paths, template, lanes):
    """(tree of numpy leaves with `lanes` lanes, source shards): the first shard holding that many lanes, sliced; otherwise
    consecutive shards of the largest lane count concatenated (8,192 lanes = four 2,048-lane shards), sliced."""
    import jax
    from gpu_engine.v2e_diff import load_shard
    sizes = [(p, shard_lanes(p)) for p in paths]
    one = next((p for p, n in sizes if n >= lanes), None)
    if one is not None:return cz.shard_tree(one, template, lanes), [str(one)]
    widest = max(n for _, n in sizes);chosen = []
    for p, n in sizes:
        if n == widest:chosen.append(p)
        if len(chosen) * widest >= lanes:break
    if len(chosen) * widest < lanes:raise SystemExit(f'{lanes} lanes need more than the {len(chosen)} {widest}-lane shards available')
    parts = [load_shard(p)[0] for p in chosen]
    leaves = [np.concatenate(xs)[:lanes] for xs in zip(*parts)]
    return jax.tree.unflatten(jax.tree.structure(template), leaves), [str(p) for p in chosen]


def step_function(ctx):
    from gpu_engine import v2e_modules
    return v2e_modules.step(ctx['engine_tables'], ctx['observation_tables'])


def compile_step(ctx, args, *, profile):
    from gpu_engine import v2e_gen
    fn = step_function(ctx);begin = time.perf_counter()
    generated = v2e_gen.compile_function('step', fn, ctx['template'], build_dir=args.build_dir, profile=profile, mode='thread')
    log('GENERATED', profile=profile, seconds=time.perf_counter() - begin, library=str(generated.path), marks=len(generated.module.marks),
        digest=generated.module.digest, **generated.module.stats)
    return fn, generated


def device_tree(tree):
    import jax
    return jax.tree.map(lambda x: jax.numpy.asarray(x), tree)


# ---------------------------------------------------------------------------
# marks

def mark_chunks(module):
    """Chunk index of every profile mark, read from the generated chunk sources (mark i writes err[1 + i])."""
    chunk = np.full(len(module.marks), -1, np.int64);name = re.compile(re.escape(f'v2e_gen_{module.name}_') + r'(\d+)\.cu')
    for file, text in module.files.items():
        m = name.fullmatch(file)
        if m:chunk[[int(x) for x in MARK_WRITE.findall(text)]] = int(m.group(1))
    if len(chunk) and (chunk < 0).any():raise ValueError(f'{int((chunk < 0).sum())} profile marks found in no chunk source')
    if len(chunk) and (np.diff(chunk) < 0).any():raise ValueError('profile marks are not in chunk order')
    return chunk


def mark_cycles(values, chunk):
    """Per-mark lane-0 cycles: each mark's clock64 offset minus the previous mark's of the same chunk (the first mark of a
    chunk keeps its offset from the chunk kernel's start). The deltas of a chunk sum to its last mark's offset."""
    v = np.asarray(values, np.int64);d = v.copy()
    same = np.zeros(len(v), bool);same[1:] = chunk[1:] == chunk[:-1]
    d[same] = v[same] - v[np.flatnonzero(same) - 1]
    return d


def mark_table(marks):
    """(labels, modules) per mark: the census labels of the marks' phase lines (program order) and their modules."""
    labels = cz.label_lines([m[3] if len(m) > 3 else None for m in marks])
    modules = [(m[4] if len(m) > 4 else None) or 'none' for m in marks]
    return labels, modules


def aggregate(cycles, labels, modules, chunk, top=15):
    """Cycle sums by label, phase, group, group x category, group x module (top) and chunk x group."""
    by_label, by_phase, by_group = collections.Counter(), collections.Counter(), collections.Counter()
    by_category = collections.defaultdict(collections.Counter);by_module = collections.defaultdict(collections.Counter)
    by_chunk = collections.defaultdict(collections.Counter)
    for c, label, module, k in zip(cycles.tolist(), labels, modules, chunk.tolist()):
        g = cz.group_of(label);by_label[label] += c;by_phase[cz.base(label)] += c;by_group[g] += c
        by_category[g][cz.category(module)] += c;by_module[g][module] += c;by_chunk[k][g] += c
    total = int(sum(by_group.values()))
    return dict(cycles_total=total, by_group=dict(by_group), share_by_group={g: v / total if total else None for g, v in by_group.items()},
        by_phase=dict(by_phase), by_label=dict(by_label), by_group_category={g: dict(v) for g, v in by_category.items()},
        by_group_module={g: dict(v.most_common(top)) for g, v in by_module.items()},
        by_chunk={str(k): dict(v) for k, v in sorted(by_chunk.items())})


def cmd_marks(args):
    out = new_out(args.out);ctx = cz.load_context(args);import jax
    _, generated = compile_step(ctx, args, profile=True);module = generated.module
    chunk = mark_chunks(module);labels, modules = mark_table(module.marks)
    paths = cz.shard_paths(args.shards, ctx['template']);call = jax.jit(lambda s, a, t: generated(s, a, t))
    report = dict(schema=SCHEMA, kind='marks', source_commit=os.environ.get('V2_SOURCE_COMMIT'), module=dict(digest=module.digest,
        marks=len(module.marks), chunks=module.stats.get('chunks'), marked_chunks=int(chunk.max()) + 1 if len(chunk) else 0,
        library=str(generated.path), stats=module.stats),
        method='lane 0 (block 0, thread 0) clock64 per mark; per-mark cycles = delta from the previous mark of its chunk', lanes={})
    arrays = dict(chunk=chunk, line=np.asarray([m[3] if m[3] is not None else -1 for m in module.marks], np.int64))
    for lanes in args.lanes:
        chosen = shards_with(paths, lanes, args.limit)
        if not chosen:raise SystemExit(f'no shard with {lanes} lanes')
        total = np.zeros(len(module.marks), np.int64);per_shard = [];errors = []
        for path in chosen:
            tree = device_tree(cz.shard_tree(path, ctx['template'], lanes))
            _, err = jax.block_until_ready(call(*tree));err = np.asarray(err)
            cycles = mark_cycles(err[1:], chunk);total += cycles;per_shard.append(int(cycles.sum()));errors.append(int(err[0]))
        entry = aggregate(total, labels, modules, chunk)
        entry.update(shards=[str(p) for p in chosen], cycles_per_shard=per_shard, error_words=errors, negative_marks=int((total < 0).sum()))
        report['lanes'][str(lanes)] = entry;arrays[f'cycles_{lanes}'] = total
        log('MARKS', lanes=lanes, shards=len(chosen), cycles_total=entry['cycles_total'], share_by_group=entry['share_by_group'], error_words=sorted(set(errors)))
        write_json(out / 'marks.json', report)
    np.savez_compressed(out / 'marks.npz', modules=np.asarray(modules), labels=np.asarray(labels), **arrays)
    log('DONE', out=str(out / 'marks.json'))
    return 0 if all(not e for r in report['lanes'].values() for e in r['error_words']) else 1


# ---------------------------------------------------------------------------
# walls

def timed_calls(fn, args, *, calls, warm):
    import jax
    for _ in range(warm):jax.block_until_ready(fn(*args))
    times = []
    for _ in range(calls):
        begin = time.perf_counter();jax.block_until_ready(fn(*args));times.append(1e3 * (time.perf_counter() - begin))
    return dict(calls=calls, warm=warm, ms_median=statistics.median(times), ms_min=min(times), ms_p90=pct(times, 90), ms_mean=statistics.fmean(times), ms=times)


def chunk_kernels(events):
    """Per chunk index: kernel durations (ns) from device events; other kernels by name."""
    from gpu_engine.v2_engine_profile import is_kernel
    chunks, other = collections.defaultdict(list), collections.defaultdict(list)
    for e in events:
        if not is_kernel(e):continue
        m = CHUNK.search(e['name'])
        (chunks[int(m.group(1))] if m else other[e['name'][:80]]).append(e['dur'])
    return chunks, other


def traced_chunks(fn, args, *, calls):
    """Per-chunk kernel time of `calls` warm calls under the JAX profiler (CUPTI)."""
    import jax
    from gpu_engine.v2_engine_profile import device_events, is_kernel, union_ns
    jax.block_until_ready(fn(*args))
    with jax.profiler.trace(None) as session:
        begin = time.perf_counter()
        for _ in range(calls):jax.block_until_ready(fn(*args))
        wall = time.perf_counter() - begin
    events, _, _ = device_events(session.profile_data);chunks, other = chunk_kernels(events)
    kernels = [e for e in events if is_kernel(e)]
    median = {str(c): statistics.median(v) for c, v in sorted(chunks.items())}
    return dict(calls=calls, chunk_kernels_per_call=sum(len(v) for v in chunks.values()) / calls, chunk_ns_median=median,
        chunk_ms_sum_of_medians=sum(median.values()) / 1e6, kernel_ms_per_call=sum(e['dur'] for e in kernels) / calls / 1e6,
        busy_ms_per_call=union_ns([(e['start'], e['end']) for e in kernels]) / calls / 1e6, wall_ms_per_call=1e3 * wall / calls,
        other_kernels={k: dict(count=len(v), ns_median=statistics.median(v)) for k, v in other.items()},
        chunk_calls_seen={str(c): len(v) for c, v in sorted(chunks.items())})


def profiler_commands(args, out_root='$OUT'):
    """The nsys and ncu invocations for this job (printed by walls and run by scripts/v2e_prof_a1.sh)."""
    run = f'{sys.executable} -m gpu_engine.v2e_phase_profile'
    common = f'--config {args.config} --reference-run {args.reference_run} --shards {args.shards} --build-dir {args.build_dir}'
    return dict(
        nsys=f'nsys profile --trace=cuda,nvtx --cuda-graph-trace=node --sample=none --cpuctxsw=none --force-overwrite=false -o {out_root}/nsys/walls '
            f'{run} walls {common} --lanes 128 2048 8192 --calls 50 --no-trace --no-jax --out {out_root}/nsys/walls-run',
        nsys_stats=f'nsys stats --report cuda_gpu_kern_sum,cuda_gpu_trace --format csv --output {out_root}/nsys/walls {out_root}/nsys/walls.nsys-rep',
        ncu=f'ncu --set full --kernel-name-base demangled --kernel-name "$(' + f'{run} ncu-targets --marks {out_root}/marks --lanes 128 --top 2)" '
            f'--launch-count 8 -o {out_root}/ncu/step128 {run} ncu-driver {common} --lanes 128 --hand-kernels --marks {out_root}/marks --out {out_root}/ncu/driver')


def cmd_walls(args):
    out = new_out(args.out);ctx = cz.load_context(args);import jax
    commands = profiler_commands(args);log('PROFILER_COMMANDS', **commands)
    fn, generated = compile_step(ctx, args, profile=False);paths = cz.shard_paths(args.shards, ctx['template'])
    call = jax.jit(lambda s, a, t: generated(s, a, t));oracle = jax.jit(fn)
    report = dict(schema=SCHEMA, kind='walls', source_commit=os.environ.get('V2_SOURCE_COMMIT'), library=str(generated.path),
        digest=generated.module.digest, chunks=generated.module.stats.get('chunks'), profiler_commands=commands, lanes={})
    for lanes in args.lanes:
        tree, sources = batch_for(paths, ctx['template'], lanes);tree = device_tree(tree)
        entry = dict(sources=sources);_, err = jax.block_until_ready(call(*tree));entry['error_word'] = int(np.asarray(err)[0])
        entry['generated'] = timed_calls(call, tree, calls=args.calls, warm=args.warm)
        if not args.no_trace:entry['generated_trace'] = traced_chunks(call, tree, calls=args.trace_calls)
        if not args.no_jax:
            begin = time.perf_counter();jax.block_until_ready(oracle(*tree));entry['jax_first_call_s'] = time.perf_counter() - begin
            entry['jax'] = timed_calls(oracle, tree, calls=args.calls, warm=args.warm)
        report['lanes'][str(lanes)] = entry
        log('WALLS', lanes=lanes, generated_ms=entry['generated']['ms_median'], jax_ms=(entry.get('jax') or {}).get('ms_median'),
            chunk_ms=(entry.get('generated_trace') or {}).get('chunk_ms_sum_of_medians'), error_word=entry['error_word'])
        write_json(out / 'walls.json', report)
    log('DONE', out=str(out / 'walls.json'))
    return 0


# ---------------------------------------------------------------------------
# ncu

def heaviest_chunks(marks_report, lanes, top=2):
    """[(group, chunk, cycles)]: for each of the `top` groups with the most cycles, its chunk with the most cycles of it."""
    entry = marks_report['lanes'][str(lanes)]
    groups = [g for g in sorted(entry['by_group'], key=lambda g: -entry['by_group'][g]) if g in cz.GROUPS][:top]
    out = []
    for g in groups:
        k, c = max(((int(k), v.get(g, 0)) for k, v in entry['by_chunk'].items()), key=lambda kv: kv[1])
        out.append((g, k, c))
    return out


def ncu_regex(chunks, hand=HAND_KERNELS):
    names = sorted({f'chunk_{k}' for _, k, _ in chunks}) + list(hand)
    return 'regex:(^|::)(' + '|'.join(names) + r')\('


def cmd_ncu_targets(args):
    report = json.loads(Path(args.marks, 'marks.json').read_text());chunks = heaviest_chunks(report, args.lanes, args.top)
    print(ncu_regex(chunks, () if args.no_hand_kernels else HAND_KERNELS))
    print(json.dumps(dict(targets=[dict(group=g, chunk=k, cycles=c) for g, k, c in chunks])), file=sys.stderr)
    return 0


def cmd_ncu_driver(args):
    out = new_out(args.out);ctx = cz.load_context(args);import jax
    _, generated = compile_step(ctx, args, profile=False);paths = cz.shard_paths(args.shards, ctx['template'])
    if args.marks:   # ncu-targets names profile-build chunks: they must be this plain build's chunks
        marked = json.loads(Path(args.marks, 'marks.json').read_text())['module'].get('chunks')
        if marked != generated.module.stats.get('chunks'):
            raise SystemExit(f"the profile build has {marked} chunks, this plain build {generated.module.stats.get('chunks')}: chunk_<k> would name another region")
    tree, sources = batch_for(paths, ctx['template'], args.lanes);tree = device_tree(tree)
    call = jax.jit(lambda s, a, t: generated(s, a, t))
    for _ in range(args.calls):jax.block_until_ready(call(*tree))
    hand = {}
    if args.hand_kernels:
        from gpu_engine import v2e_cuda
        v2e_cuda.load(args.build_dir);state, _, trackers = tree
        jax.block_until_ready(v2e_cuda.chance(state.keys, state.replacement_count));jax.block_until_ready(v2e_cuda.roundtrip((state, trackers)))
        hand = dict(library=str(v2e_cuda._loaded['path']), kernels=list(HAND_KERNELS))
    write_json(out / 'driver.json', dict(schema=SCHEMA, kind='ncu-driver', lanes=args.lanes, calls=args.calls, sources=sources, library=str(generated.path), hand=hand))
    log('DONE', out=str(out / 'driver.json'))
    return 0


# ---------------------------------------------------------------------------
# jax-attrib

FRAME_SECTIONS = ('FileNames', 'FunctionNames', 'FileLocations', 'StackFrames')


def hlo_stacks(lines):
    """stack(frame id) -> [(file basename, function, line)], innermost first, from the HLO text's frame tables.

    jax/XLA 0.12 print every StackFrames parent_frame_id one above the parent's id (a root prints 1 = itself), which
    v2_engine_profile._frames follows literally, so its stacks stop after a frame or two. A self-parent cannot occur
    otherwise, so its presence selects the -1 correction; stacks then reach the outermost recorded frame."""
    sect = None;files = {};funcs = {};locs = {};frames = {}
    for line in lines:
        if line in FRAME_SECTIONS:sect = line;continue
        if sect and (not line or line.startswith(('%', 'ENTRY', 'HloModule'))):sect = None
        if not sect:continue
        key, value = line.split(' ', 1);key = int(key)
        if sect == 'FileNames':files[key] = value.strip('"')
        elif sect == 'FunctionNames':funcs[key] = value.strip('"')
        else:(locs if sect == 'FileLocations' else frames)[key] = {k: int(v) for k, v in re.findall(r'(\w+)=(\d+)', value)}
    offset = 1 if any(f.get('parent_frame_id') == k for k, f in frames.items()) else 0
    cache = {}
    def stack(fid):
        if fid not in cache:
            out = [];seen = set();walk = fid
            while walk in frames and walk not in seen:
                seen.add(walk);loc = locs.get(frames[walk].get('file_location_id', 0), {})
                out.append((Path(files.get(loc.get('file_name_id', 0), '?')).name, funcs.get(loc.get('function_name_id', 0), '?'), loc.get('line', 0)))
                walk = frames[walk].get('parent_frame_id', 0) - offset
            cache[fid] = out
        return cache[fid]
    stack.offset = offset
    return stack


CALLS_ATTR = re.compile(r'\bcalls=(\{[^}]*\}|%?[\w.\-]+)')
ATTRIB_CAVEAT = ('a kernel is charged to the phase of its HLO instruction\'s own metadata (a fusion\'s: its root op\'s); a fusion whose '
    'fused instructions come from several phases is charged whole to that one phase (multi_phase_fusion_share says how much step time that is)')


def phase_hlo_index(text):
    """Optimized HLO instruction name -> dict(opcode, scope, line, module, phases): the C_ scope, the outermost
    transition.py step_with_chance frame's line and the innermost engine module of its stack frames; for an instruction that
    calls computations (a fusion), `phases`: the phases of every instruction it reaches, its own included."""
    from gpu_engine import v2_engine_profile as vp
    lines = text.splitlines();stack = hlo_stacks(lines);index = {};members = collections.defaultdict(list);calls = {};computation = None
    for line in lines:
        stripped = line.strip()
        if stripped and not line.startswith(' ') and stripped.endswith('{') and not stripped.startswith('HloModule'):
            computation = stripped.split()[1 if stripped.startswith('ENTRY') else 0].lstrip('%');continue
        m = vp.INSTRUCTION.match(line)
        if not m:continue
        name, _, opcode, rest = m.groups()
        op_name = re.search(r'op_name="([^"]*)"', rest);frame = re.search(r'stack_frame_id=(\d+)', rest)
        frames = stack(int(frame.group(1))) if frame else []
        phase = next((ln for f, fn, ln in reversed(frames) if f == 'transition.py' and fn.split('.')[-1] == 'step_with_chance'), None)
        engine = [f for f in frames if not any(x in f[0] for x in vp.LIBRARY)]
        index[name] = dict(opcode=opcode, scope=vp.scope_of(op_name.group(1)) if op_name else None, line=phase,
            module=engine[0][0].rsplit('.', 1)[0] if engine else None)
        members[computation].append(name);called = CALLS_ATTR.search(rest)
        if called:calls[name] = re.findall(r'%?([\w.\-]+)', called.group(1))
    memo = {}
    def reach(computation, path):
        if computation not in memo:
            memo[computation] = set()   # a cycle (none in valid HLO) contributes nothing
            memo[computation] = set().union(*(phases(n, path) for n in members.get(computation, ())))
        return memo[computation]
    def phases(name, path=()):
        own = {cz.phase_of(index[name]['line'])} if index[name]['line'] is not None else set()
        return own.union(*(reach(c, path + (name,)) for c in calls.get(name, ())))
    for name in calls:index[name]['phases'] = sorted(phases(name))
    return index


def attribute(events, index, steps, scope='step'):
    """Kernel ns per tick of `scope` by label, phase and group (and group x category); other scopes and unmapped kernels apart."""
    from gpu_engine import v2_engine_profile as vp
    by_norm = {vp.normalize(k): k for k in index}
    def lookup(e):
        for key in ('hlo_op', 'tf_op', 'name'):
            value = e['stats'].get(key) if key != 'name' else e['name']
            if not value:continue
            value = str(value).split(':')[-1].strip().lstrip('%')
            if value in index:return value
            if vp.normalize(value) in by_norm:return by_norm[vp.normalize(value)]
        return None
    phase, group, scopes = collections.Counter(), collections.Counter(), collections.Counter()
    category = collections.defaultdict(collections.Counter);count = collections.Counter();multi = collections.Counter()
    for e in events:
        if not vp.is_kernel(e):continue
        name = lookup(e);info = index.get(name, {}) if name else {}
        s = info.get('scope') or ('collector' if name else 'unmapped');scopes[s] += e['dur']
        if s != scope:continue
        p = cz.phase_of(info.get('line')) or 'outside';g = cz.group_of(p)
        phase[p] += e['dur'];group[g] += e['dur'];count[g] += 1;category[g][cz.category(info.get('module'))] += e['dur']
        spans = info.get('phases') or ()
        if len(spans) > 1:multi['phase'] += e['dur']
        if len({cz.group_of(x) for x in spans}) > 1:multi['group'] += e['dur']
    ms = lambda c: {k: v / steps / 1e6 for k, v in c.items()}
    total = sum(group.values())
    return dict(steps=steps, step_ms_per_tick=total / steps / 1e6, ms_per_tick_by_group=ms(group), ms_per_tick_by_phase=ms(phase),
        kernels_per_tick_by_group={k: v / steps for k, v in count.items()}, share_by_group={k: v / total if total else None for k, v in group.items()},
        ms_per_tick_by_group_category={g: ms(c) for g, c in category.items()}, ms_per_tick_by_scope=ms(scopes),
        multi_phase_fusion_ms_per_tick=multi['phase'] / steps / 1e6, multi_phase_fusion_share=multi['phase'] / total if total else None,
        multi_group_fusion_share=multi['group'] / total if total else None, caveat=ATTRIB_CAVEAT)


def cmd_jax_attrib(args):
    out = new_out(args.out);import jax
    from gpu_engine.v2_engine_profile import device_events
    root = Path(args.profile_dir);profile = json.loads((root / 'profile.json').read_text()) if (root / 'profile.json').exists() else {}
    steps_of = {int(r['lanes']): int(r.get('chunk_steps') or args.chunk_steps) for r in profile.get('runs', ())}
    report = dict(schema=SCHEMA, kind='jax-attrib', profile_dir=str(root), source_commit=profile.get('source_commit'), caveat=ATTRIB_CAVEAT, lanes={})
    for lanes in args.lanes:
        with gzip.open(root / f'hlo_chunk_{lanes}.txt.gz', 'rt') as f:index = phase_hlo_index(f.read())
        traces = sorted((root / f'trace_{lanes}').rglob('*.xplane.pb'))
        if not traces:raise SystemExit(f'no saved trace under {root}/trace_{lanes} (run v2_engine_profile with --save-trace)')
        events = []
        for path in traces:events += device_events(jax.profiler.ProfileData.from_file(str(path)))[0]
        entry = attribute(events, index, steps_of.get(lanes, args.chunk_steps))
        entry.update(traces=[str(p) for p in traces], hlo_instructions=len(index),
            hlo_with_phase=sum(1 for v in index.values() if v['line'] is not None and v['scope'] == 'step'))
        report['lanes'][str(lanes)] = entry
        log('JAX_ATTRIB', lanes=lanes, step_ms_per_tick=entry['step_ms_per_tick'], by_group=entry['ms_per_tick_by_group'],
            multi_phase_fusion_share=entry['multi_phase_fusion_share'], multi_group_fusion_share=entry['multi_group_fusion_share'])
    write_json(out / 'jax_attrib.json', report)
    log('DONE', out=str(out / 'jax_attrib.json'))
    return 0


# ---------------------------------------------------------------------------
# divergence

BASE_COLUMNS = {'.type', '.priority', '.accuracy', '.type_num', '.type_den', '.struggle'}
RESIDUAL_ABILITIES = ('speed_boost', 'truant')


def table_columns(table):
    """Path-class columns of the engine tables: move-indexed (length = move vocabulary, outside .items/.entry/.combat/
    .public_port and the base value columns), ability-indexed (.entry/.combat, length = .entry.intimidate's) and
    item-indexed (.items, length = .items.leftovers'). {kind: {path: numpy array}}."""
    import jax
    leaves = jax.tree_util.tree_flatten_with_path(table)[0];n_moves = int(np.shape(table.power)[0])
    n_abilities = int(np.shape(table.entry.intimidate)[0]) if getattr(table, 'entry', None) is not None else None
    n_items = int(np.shape(table.items.leftovers)[0]) if getattr(table, 'items', None) is not None else None
    out = dict(move={}, ability={}, item={})
    for path, x in leaves:
        key = jax.tree_util.keystr(path);x = np.asarray(x)
        if x.ndim == 0 or x.dtype.kind not in 'biu':continue
        if key.startswith('.public_port'):continue
        if key.startswith('.items'):
            if x.shape[0] == n_items:out['item'][key] = x
        elif key.startswith(('.entry', '.combat')):
            if x.ndim == 1 and x.shape[0] == n_abilities:out['ability'][key] = x
            elif x.shape[0] == n_moves:out['move'][key] = x
        elif x.shape[0] == n_moves and key not in BASE_COLUMNS:out['move'][key] = x
    return out


def lookup_flags(columns, ids, mask, prefix):
    """{prefix + path: bool [B]}: lanes in `mask` whose id has a nonzero entry in the column (any along trailing axes)."""
    out = {}
    for key, col in columns.items():
        v = col[np.clip(ids, 0, col.shape[0] - 1)]
        v = v.reshape(v.shape[0], -1).any(1) if v.ndim > 1 else v != 0
        out[prefix + key] = mask & v
    return out


def merge_sides(per_side):
    """OR of two sides' {feature: bool [B]} dicts."""
    out = {}
    for d in per_side:
        for k, v in d.items():out[k] = out[k] | v if k in out else v.copy()
    return out


def lane_features(state, actions, table, columns, mask, usable):
    """Per group A..G: {feature: bool [B]} path-class features of every lane, from the step inputs (approximate: chance,
    hits and the turn's own changes are unknown before the step). mask: action_mask [B,2,9]; usable: usable_moves [B,2,4]."""
    from gpu_engine import mechanics as m
    from gpu_engine.state import LEECH_SEED
    s = SimpleNamespace(**{k: np.asarray(v) for k, v in state._asdict().items()});actions = np.asarray(actions)
    B = s.hp.shape[0];rows = np.arange(B)[:, None];sides = np.arange(2)[None, :];at = lambda x: x[rows, sides, s.active]
    forced = s.forced_switch.astype(bool) | (at(s.hp) <= 0);replacement = forced.any(1);requested = ~replacement[:, None] | forced
    valid = np.take_along_axis(np.asarray(mask), np.clip(actions, 0, 8)[..., None], axis=2)[..., 0];done = s.done.astype(bool)
    invalid = ~done & (requested & (~valid | (actions < 0) | (actions > 8))).any(1);live = ~done & ~invalid
    resuming = live & s.pending_residual.astype(bool);ordinary = live & ~replacement
    kinds = dict(done=done, invalid=invalid, resuming=resuming & ~invalid, replacement=live & replacement & ~resuming, ordinary=ordinary & ~resuming)
    kind = {f'kind:{k}': v for k, v in kinds.items()}
    switches = live[:, None] & requested & (actions >= 4);entering = switches | ((live & replacement)[:, None] & forced)
    saved = np.where(live & s.pending_residual.astype(bool), s.queued_actor, -1)
    move_actions = np.where(saved[:, None] == sides, s.queued_actions, actions);slot = np.clip(move_actions, 0, 3)
    struggle = ~np.asarray(usable).any(-1);moves = s.moves[rows, sides, s.active, slot];charge = at(s.charge_move)
    moves = np.where(charge > 0, charge, moves);moves = np.where(struggle, int(np.asarray(table.struggle)), moves)
    acting = ordinary[:, None] & ~switches
    ability, item, status = at(s.ability), at(s.item), at(s.status)
    bench = np.arange(5)[None, None, :] + (np.arange(5)[None, None, :] >= s.active[..., None])
    target = np.take_along_axis(bench, np.clip(actions - 4, 0, 4)[..., None], axis=2)[..., 0];incoming = s.ability[rows, sides, target]
    flags = lambda kind, ids, mask, prefix: lookup_flags(columns[kind], ids, mask, prefix)
    side = lambda k, x: x[:, k]
    out = dict(A=dict(kind, initialize=live & s.entry_pending.astype(bool).any(1)))
    focus = getattr(getattr(table, 'focus_punch', None), 'move', None);pursuit = getattr(getattr(table, 'pursuit', None), 'pursuit', None)
    b = [dict(struggle=side(k, acting & struggle), charge=side(k, acting & (charge > 0))) for k in range(2)]
    if focus is not None:
        for k in range(2):b[k]['focus_punch'] = side(k, acting) & np.asarray(focus)[side(k, moves)]
    out['B'] = dict(kind, **merge_sides(b))
    out['C'] = {'pursuit': np.zeros(B, bool)}
    if pursuit is not None:
        p = np.asarray(pursuit)
        out['C']['pursuit'] = (side(0, acting) & p[side(0, moves)] & side(1, switches)) | (side(1, acting) & p[side(1, moves)] & side(0, switches))
    d = []
    for k in range(2):
        e = side(k, entering);row = dict(switch=e, baton=e & side(k, s.baton_pass.astype(bool)), spikes=e & (side(k, s.spikes) > 0))
        row.update(flags('ability', side(k, incoming), e, 'enter:'));row.update(flags('ability', side(k, ability), e, 'leave:'))
        d.append(row)
    out['D'] = merge_sides(d)
    e = []
    for k in range(2):
        a = side(k, acting);row = flags('move', side(k, moves), a, 'move:')
        row.update(flags('ability', side(k, ability), a, 'ability:'));row.update(flags('item', side(k, item), a, 'item:'))
        for code in range(1, 7):row[f'status:{code}'] = a & (side(k, status) == code)
        row.update(foe_substitute=a & (side(1 - k, at(s.substitute_hp)) > 0), confused=a & (side(k, at(s.confusion_counter)) > 0),
            encore=a & (side(k, at(s.encore_counter)) > 0), switched=side(k, switches))
        e.append(row)
    out['E'] = merge_sides(e)
    turn = ordinary | resuming;f = []
    for k in range(2):
        row = {f'status:{c}': turn & (side(k, status) == c) for c in (m.BRN, m.PSN, m.TOX)}
        row.update(flags('item', side(k, item), turn, 'item:'))
        row.update({key: v for key, v in flags('ability', side(k, ability), turn, 'ability:').items() if key.endswith(RESIDUAL_ABILITIES)})
        row.update(leech_seed=turn & ((side(k, at(s.volatile)).astype(np.uint32) & LEECH_SEED) != 0), wish=turn & (side(k, s.wish_counter) > 0),
            encore=turn & (side(k, at(s.encore_counter)) > 0), charge=turn & (side(k, at(s.charge_counter)) > 0))
        f.append(row)
    out['F'] = dict(merge_sides(f), weather=turn & (s.weather > 0))
    out['G'] = dict(kind)
    return out


class UnionStats:
    """Per group: 32-lane warp unions of lane path classes, accumulated over shards."""
    def __init__(self, warp=32):
        self.warp = warp;self.groups = collections.defaultdict(lambda: dict(warps=0, lanes=0, lane_features=0, union_features=0,
            mean_lane_features_sum=0., distinct=[], frequency=collections.Counter()))

    def add(self, features):
        for g, named in features.items():
            names = sorted(named);stats = self.groups[g]
            if not names:continue
            x = np.stack([np.asarray(named[n], bool) for n in names], axis=1);B = x.shape[0]
            for n, c in zip(names, x.sum(0).tolist()):stats['frequency'][n] += c
            for start in range(0, B, self.warp):
                w = x[start:start + self.warp];lanes = w.shape[0]
                stats['warps'] += 1;stats['lanes'] += lanes;stats['lane_features'] += int(w.sum())
                stats['union_features'] += int(w.any(0).sum());stats['mean_lane_features_sum'] += float(w.sum()) / lanes
                stats['distinct'].append(len({row.tobytes() for row in np.packbits(w, axis=1)}))

    def summary(self, uniform=None, design=DESIGN_UNIFORM_SHARE):
        """Per group: distinct classes per warp, union vs lane features and the block-model union factor (warp cost
        u + (1-u) |union| / F over the mean lane cost u + (1-u) |lane| / F; F = features seen), at u = the group's `uniform`
        share (the census hook share) and at u = `design` (the design's ~10k of 14-15k ops)."""
        out = {}
        for g, s in self.groups.items():
            if not s['warps']:continue
            F = sum(1 for v in s['frequency'].values() if v) or 1;u = (uniform or {}).get(g, 0.) or 0.
            lane = s['mean_lane_features_sum'] / s['warps'];union = s['union_features'] / s['warps']
            cost = lambda share, features: share + (1 - share) * features / F
            warp_cost, lane_cost = cost(u, union), cost(u, lane)
            out[g] = dict(warps=s['warps'], lanes=s['lanes'], features_seen=F, uniform_share=u, design_uniform_share=design,
                union_factor_design_uniform=cost(design, union) / cost(design, lane) if cost(design, lane) else None,
                distinct_classes_per_warp=dict(mean=statistics.fmean(s['distinct']), p50=pct(s['distinct'], 50), p90=pct(s['distinct'], 90),
                    max=max(s['distinct'])), single_class_warps=sum(1 for d in s['distinct'] if d == 1) / s['warps'],
                lane_features_mean=lane, union_features_mean=union, union_factor=warp_cost / lane_cost if lane_cost else None,
                masked_uniform_factor=1 / lane_cost if lane_cost else None,
                feature_lane_share={k: v / s['lanes'] for k, v in s['frequency'].most_common()})
        return out


def census_weights(path):
    """(uniform share = hook share, equations) per group from a census.json, or (None, None)."""
    if not path:return None, None
    groups = json.loads(Path(path).read_text())['census']['groups']
    return {g: r['hook_share'] for g, r in groups.items()}, {g: r['n'] for g, r in groups.items()}


def weighted_factor(summary, equations, key='union_factor'):
    use = {g: r[key] for g, r in summary.items() if g in cz.GROUPS and r.get(key) is not None}
    if not use:return None
    weights = {g: (equations or {}).get(g, 1) for g in use};total = sum(weights.values())
    return sum(use[g] * weights[g] for g in use) / total if total else None


def cmd_divergence(args):
    out = new_out(args.out);ctx = cz.load_context(args);import jax
    from gpu_engine import transition
    from gpu_engine.state import usable_moves
    table = ctx['engine_tables'];columns = table_columns(table);host_table = jax.device_get(table)
    mask_fn = jax.jit(lambda st: transition.action_mask(st, table));usable_fn = jax.jit(usable_moves)
    paths = cz.shard_paths(args.shards, ctx['template'], args.limit);stats = UnionStats(args.warp);begin = time.perf_counter()
    for i, path in enumerate(paths):
        state, actions, _ = cz.shard_tree(path, ctx['template'])
        device_state = jax.tree.map(jax.numpy.asarray, state)
        mask, usable = np.asarray(mask_fn(device_state)), np.asarray(usable_fn(device_state))
        stats.add(lane_features(state, actions, host_table, columns, mask, usable))
        if (i + 1) % 10 == 0:log('DIVERGENCE_PROGRESS', shards=i + 1, seconds=time.perf_counter() - begin)
    uniform, equations = census_weights(args.census);summary = stats.summary(uniform, args.design_uniform_share)
    report = dict(schema=SCHEMA, kind='divergence', shards=len(paths), warp=args.warp, census=args.census, columns={k: sorted(v) for k, v in columns.items()},
        groups=summary, weighted_union_factor=weighted_factor(summary, equations), design_uniform_share=args.design_uniform_share,
        weighted_union_factor_design_uniform=weighted_factor(summary, equations, 'union_factor_design_uniform'),
        uniform_shares=dict(census='each group\'s census hook share (about 0.9)' if uniform else 'none given: 0', design=args.design_uniform_share),
        weights='census equations per group' if equations else 'equal',
        method=('approximate (design section 5 step 5): classes from the step inputs; union factor = mean warp cost / mean lane cost with cost '
            'u + (1 - u) x features / F (u = the group\'s census hook share, and again at the design\'s u = 0.7; F = features seen in the group, '
            'equal feature weights); '
            'distinct_classes_per_warp is the raw union of path classes'))
    write_json(out / 'divergence.json', report)
    log('DIVERGENCE', **{g: dict(union=r['union_factor'], union_design=r['union_factor_design_uniform'], distinct=r['distinct_classes_per_warp']['mean'])
        for g, r in summary.items()}, weighted=report['weighted_union_factor'], weighted_design=report['weighted_union_factor_design_uniform'])
    return 0


# ---------------------------------------------------------------------------
# pricer-split

LIVE = dict(worlds=16, bucket=1, depth=2, k=9, k1=9, selection='prior', block_lanes=2187, blocks_per_call=16)


def split_shares(normal, constant, depth1, zero_all=None):
    """The live split per call in ms. The depth-2 method: leaf forward = (i) - (ii), engine = (ii) - (iii) (stepping, observing
    and P1's extra work over D3 depth 1, e.g. the opponent-view child-prior forward), depth 1 = (iii), total = (i). With the
    every-forward-zeroed variant (iv): all forwards = (i) - (iv), engine total = (iv). Shares of (i)."""
    from gpu_engine import d3_depth2_throughput as t2
    est = t2.split_estimate(normal, constant, depth1)
    forward, engine = est.get('turn2_forward_seconds_estimate'), est.get('turn2_step_observe_and_child_prior_seconds_estimate')
    every = None if zero_all is None or normal is None else normal - zero_all
    ms = lambda x: None if x is None else 1e3 * x;share = lambda x: None if x is None or not normal else x / normal
    return dict(est, forward_ms=ms(forward), engine_ms=ms(engine), depth1_ms=ms(depth1), total_ms=ms(normal),
        engine_and_depth1_ms=ms(constant), forward_share=share(forward), engine_share=share(engine), depth1_share=share(depth1),
        zero_all_forwards_seconds=zero_all, all_forwards_ms=ms(every), engine_total_ms=ms(zero_all), all_forwards_share=share(every),
        engine_total_share=share(zero_all))


def zero_forward(real, census):
    """forward= seam replacing EVERY forward (P1's priors and the leaf values) by zeros of the real output shapes (valid True) that
    consume every input: d3_depth2_throughput.constant_turn2_forward's construction at every batch size. At k = 9 full width the
    enumeration does not depend on logits or values, so this variant does the pricer's engine work without the network."""
    import jax
    import jax.numpy as j
    def forward(params, inputs):
        census.append(int(jax.tree.leaves(inputs)[0].shape[0]))
        shapes = jax.eval_shape(real, params, inputs)
        out = jax.tree.map(lambda s: j.ones(s.shape, s.dtype) if s.dtype == j.bool_ else j.zeros(s.shape, s.dtype), shapes)
        keep = sum(j.sum(x.astype(j.float32)) for x in jax.tree.leaves(inputs)) * 0.
        return out._replace(value=out.value + keep.astype(out.value.dtype))
    return forward


def pricer_args(args):
    """The namespace d3_depthn_throughput.prepare reads, at the live shape: capacity = bucket x W/2 roots per call, one block."""
    return SimpleNamespace(**vars(args), capacity=LIVE['bucket'] * LIVE['worlds'] // 2, w_values=(2,), k1_own=LIVE['k1'], k1_opp=LIVE['k1'],
        block_lanes=LIVE['block_lanes'])


def pricer_tables(prepared, args):
    """The bank executor's compiled tables (constructor_worlds.engine_tables_digest of prepared.engine_tables / observation_tables) and,
    when given, their comparison with the live executor's: --live-metadata (the live server's --metadata, read as it reads it) compiled
    through constructor_worlds.engine_tables with the bank's catalog (the live binding requires an equal catalog), entry for entry
    (constructor_worlds.tables_differences), and/or a digest recorded elsewhere (--live-tables-digest, e.g. the depth-N gate's TABLES_BRIDGE)."""
    from gpu_engine import constructor_worlds as cw
    bank = (prepared.engine_tables, prepared.observation_tables);names = ('engine', 'observation')
    out = dict(bank_digest=cw.engine_tables_digest(bank), digest_function='gpu_engine.constructor_worlds.engine_tables_digest',
        entries=len(cw.table_leaves(dict(zip(names, bank)))), live_comparison=None)
    if args.live_metadata:
        from gpu_engine.phase_d_protocol_constructor import read_bound
        live = cw.engine_tables(read_bound(args.live_metadata, args.live_metadata_sha256), prepared.catalog)
        found = [d for x, y, name in zip(bank, live, names) for d in cw.tables_differences({name: x}, {name: y})]
        out.update(live_metadata=args.live_metadata, live_metadata_sha256=args.live_metadata_sha256, live_digest=cw.engine_tables_digest(live),
            live_equal=not found, live_differences=found[:20], live_comparison='equal (recompiled live metadata)' if not found else f'DIFFERENT: {found[0]}')
    if args.live_tables_digest:
        same = args.live_tables_digest == out['bank_digest'];out.update(recorded_live_digest=args.live_tables_digest, recorded_live_digest_equal=same)
        note = 'equal to the recorded live digest' if same else 'DIFFERENT from the recorded live digest'
        out['live_comparison'] = note if out['live_comparison'] is None else out['live_comparison'] + '; ' + note
    return out


def cmd_pricer_split(args):
    found = os.environ.get('JAX_DEFAULT_MATMUL_PRECISION')
    if found != args.matmul_precision:raise SystemExit(f'JAX_DEFAULT_MATMUL_PRECISION must be {args.matmul_precision!r} (the live pin); found {found!r}')
    out = new_out(args.out)
    from gpu_engine import d3_depth2_gate as gate, d3_depth2_throughput as t2, d3_depthn_throughput as t3   # no JAX at import
    ns = pricer_args(args);ctx = t3.prepare(ns)   # bank chunk requests, the champion, block 0; applies --memory-policy before JAX loads
    import jax
    from gpu_engine import depthn_live_search as live, device_depthn_pricer as dn   # these import JAX: only after the memory policy
    if str(jax.config.jax_default_matmul_precision) != args.matmul_precision:raise SystemExit('JAX matmul precision differs from the pin')
    if args.compilation_cache:
        from pokezero.owning_worker_runtime import configure_compilation_cache
        configure_compilation_cache(args.compilation_cache)
    if ctx.forward is None:raise SystemExit('the live pricer prices with the champion: --champion-archive and its sha256s required')
    tables = pricer_tables(ctx.provider.prepared, args);log('PRICER_TABLES', **{k: v for k, v in tables.items() if k != 'live_differences'})
    rows = dn.block_nodes(LIVE['block_lanes'], LIVE['k'], LIVE['k']) * LIVE['k'] ** 2;reason = t2.split_check(rows, ns.capacity);census, every = [], []
    def pricer(forward):   # the live server's own constructor (model None: unused with forward=)
        return live.make_pricers(ctx.provider.executor, forward, ctx.params, worlds=LIVE['worlds'], buckets=(LIVE['bucket'],),
            block_lanes=LIVE['block_lanes'], blocks_per_call=LIVE['blocks_per_call'])[LIVE['bucket']]
    def caller(name, p=None):
        c = t3.Caller('depth1' if name == 'depth1' else 'depth2-k9', ns, ctx);c.name = name
        if p is not None:c.pricer = p
        return c
    configs = [('depth1', None), ('live', pricer(ctx.forward))]
    if reason is None:configs.append(('constant_leaf_forward', pricer(t2.constant_turn2_forward(ctx.forward, rows, census))))
    configs.append(('zero_all_forwards', pricer(zero_forward(ctx.forward, every))))
    sections = {}
    for name, p in configs:
        c = caller(name, p);blocks = t3.measure(c, ctx.planned[:1], args.repeats, memory=t2.memory_stats, progress=gate.log)
        keys = getattr(c.pricer, 'programs', None)
        sections[name] = dict(blocks=blocks, program_keys=None if keys is None else sorted(repr(k) for k in keys),
            options=None if p is None else dict(capacity=p.capacity, depth=p.depth, k_own=p.k_own, k_opp=p.k_opp, k1_own=p.k1_own, k1_opp=p.k1_opp,
                block_lanes=p.block_lanes, blocks_per_call=p.blocks_per_call, selection=p.selection, fixed_capacity=p.fixed_capacity, digest=p.digest,
                evaluator_suffix=repr(p.evaluator_suffix)))
    median = lambda name: sections[name]['blocks'][0]['warm_device_seconds_median'] if name in sections else None
    skipped = reason or t2.split_check(rows, ns.capacity, census)
    split = split_shares(median('live'), None if skipped else median('constant_leaf_forward'), median('depth1'), median('zero_all_forwards'))
    live_block = sections['live']['blocks'][0]
    shape_keys = lambda name: [k.split(", ('evaluator'")[0] for k in sections[name]['program_keys'] or ()]
    report = dict(schema=SCHEMA, kind='pricer-split', evaluator=ctx.evaluator, live_shape=LIVE, roots_per_call=ns.capacity, roots_available=len(ctx.available),
        cycled=len(ctx.available) < ns.capacity, chunk=args.chunk, repeats=args.repeats, leaf_forward_rows=rows, traced_forward_rows=sorted(set(census)),
        traced_forward_rows_zero_all=sorted(set(every)), skipped_constant_variant=skipped, matmul_precision=str(jax.config.jax_default_matmul_precision),
        devices=[d.device_kind for d in jax.devices()], split=split, tables=tables,
        program_shapes_match=dict(constant_leaf_forward=None if 'constant_leaf_forward' not in sections else shape_keys('constant_leaf_forward') == shape_keys('live'),
            zero_all_forwards=shape_keys('zero_all_forwards') == shape_keys('live')),
        pricer_timing_split=dict(p1_ms=None if live_block.get('warm_p1_seconds_median') is None else 1e3 * live_block['warm_p1_seconds_median'],
            plies_ms_lower_bound=None if live_block.get('warm_plies_seconds_median') is None else 1e3 * live_block['warm_plies_seconds_median']),
        sections=sections, method=('d3_depth2_throughput --profile-split: warm median device seconds of separately compiled programs; leaf forwards '
            'recognized by static batch size (leaf_forward_rows) and replaced by zeros of their output shapes that consume every input; '
            'zero_all_forwards does the same for every forward (P1\'s too)'),
        scope=('live pricer config via depthn_live_search.make_pricers at bucket 1 (8 requests of 2 worlds = one W=16 root) on the bank executor; '
            'roots are D3 bank chunk requests (D3 worlds), not constructor worlds. Checked, not assumed: the bank executor\'s compiled tables '
            '(tables: digest, and against the live executor\'s with --live-metadata / --live-tables-digest) and the program keys per variant; '
            'the live server\'s keys are these plus (\'worlds\', source, carry signature) for constructor worlds'),
        caveat=('estimates from separate programs; XLA may schedule the variants differently; engine = (ii) - (iii) includes P1\'s extra work over '
            'D3 depth 1 (e.g. the opponent-view child-prior forward), as in the depth-2 split; engine_total (every forward zeroed, P1\'s included) '
            'is the direct engine time at k = 9 full width'))
    write_json(out / 'pricer_split.json', report)
    log('PRICER_SPLIT', **{k: v for k, v in split.items() if k.endswith(('_ms', '_share'))}, skipped=skipped, tables=tables.get('live_comparison'))
    return 0


# ---------------------------------------------------------------------------
# report

def nsys_chunks(path):
    """{lanes: {chunk: median ns}} from an `nsys stats --report cuda_gpu_trace --format csv` file (lanes = grid x 32)."""
    import csv
    per = collections.defaultdict(lambda: collections.defaultdict(list))
    with open(path, newline='') as f:
        for row in csv.DictReader(f):
            name = row.get('Name') or row.get('Kernel Name') or '';m = CHUNK.search(name)
            if not m:continue
            dur = row.get('Duration (ns)') or row.get('Duration') or row.get('Duration [ns]');grid = row.get('GrdX') or row.get('Grid X')
            if dur is None or grid is None:continue
            per[int(float(grid)) * 32][int(m.group(1))].append(float(dur))
    return {lanes: {str(k): statistics.median(v) for k, v in chunks.items()} for lanes, chunks in per.items()}


NCU_DETAILS = ('Warp Cycles Per Issued Instruction', 'Warp Cycles Per Executed Instruction', 'L1/TEX Hit Rate', 'L2 Hit Rate', 'Duration',
    'Executed Instructions', 'Achieved Occupancy', 'Avg. Active Threads Per Warp', 'Registers Per Thread', 'No Eligible')
NCU_STALL = re.compile(r'smsp__average_warps?_issue_stalled_(\w+?)_per_issue_active\.ratio$')


def ncu_number(text):
    try:return float(str(text).replace(',', ''))
    except ValueError:return None


def ncu_summary(details=None, raw=None):
    """Per profiled kernel (by ncu ID): its name, the details page's CPI, hit rates and size metrics, and the raw page's stall
    reasons in cycles per issued instruction (largest first). Best effort over `ncu --import ... --page details|raw --csv`."""
    import csv
    out = collections.OrderedDict()
    def kernel(row):
        key = row.get('ID') or str(len(out));name = row.get('Kernel Name') or row.get('Function Name') or '?'
        return out.setdefault(key, dict(kernel=name[:120], chunk=(lambda m: int(m.group(1)) if m else None)(CHUNK.search(name)), metrics={}, stalls={}))
    if details and Path(details).exists():
        with open(details, newline='') as f:
            for row in csv.DictReader(f):
                if row.get('Metric Name') in NCU_DETAILS:
                    kernel(row)['metrics'][row['Metric Name']] = dict(value=ncu_number(row.get('Metric Value')), unit=row.get('Metric Unit'))
    if raw and Path(raw).exists():
        with open(raw, newline='') as f:
            rows = list(csv.DictReader(f))
        for row in rows[1:] if rows and not ncu_number(rows[0].get('ID')) else rows:   # the raw page's second line holds units
            entry = kernel(row)
            for name, value in row.items():
                m = NCU_STALL.match(name or '')
                if m and ncu_number(value):entry['stalls'][m.group(1)] = ncu_number(value)
            entry['stalls'] = dict(sorted(entry['stalls'].items(), key=lambda kv: -kv[1]))
    return out


def ncu_by_group(ncu, marks, lanes=128):
    """ncu kernels with the phase group holding most of each chunk's mark cycles at `lanes` (hand kernels: None)."""
    by_chunk = ((marks or {}).get('lanes', {}).get(str(lanes)) or {}).get('by_chunk', {})
    for k in ncu.values():
        groups = by_chunk.get(str(k['chunk'])) if k['chunk'] is not None else None
        k['group'] = max(groups, key=groups.get) if groups else None
    return dict(ncu)


def generated_ms(marks_entry, wall_ms, chunk_ns=None):
    """(ms per group, how): generated ms summing to the warm wall per call, each chunk's kernel time split by its mark cycle
    shares when the timed chunk ids are exactly the marked ones, else the step-wide lane-0 cycle shares, with `how` saying
    which and why (a mismatch is named, never silently relabelled). (None, reason) without a wall or cycles."""
    by_chunk = {k: v for k, v in marks_entry['by_chunk'].items() if sum(v.values()) > 0}
    if chunk_ns and set(chunk_ns) == set(by_chunk):timed, how = True, 'walls x mark cycle shares by chunk kernel time'
    elif chunk_ns:
        timed = False;missing, extra = sorted(set(by_chunk) - set(chunk_ns), key=int), sorted(set(chunk_ns) - set(by_chunk), key=int)
        how = (f'walls x step-wide lane-0 cycle shares (FALLBACK: chunk ids do not line up; marked chunks without a time {missing[:8]}'
            f'{"..." if len(missing) > 8 else ""}, timed chunks without marks {extra[:8]}{"..." if len(extra) > 8 else ""})')
    else:timed, how = False, 'walls x step-wide lane-0 cycle shares (no per-chunk times)'
    weights = collections.Counter()
    for k, groups in by_chunk.items():
        scale = chunk_ns[k] / sum(groups.values()) if timed else 1.
        for g, c in groups.items():weights[g] += c * scale
    total = sum(weights.values())
    if not total or wall_ms is None:return None, 'no walls' if wall_ms is None else 'no mark cycles'
    return {g: wall_ms * w / total for g, w in weights.items()}, how


def decide(gen_2048, census_groups):
    """The design's section 5 decision rule over generated ms per group at 2,048 lanes (fallback: equation shares)."""
    if gen_2048:
        total = sum(gen_2048.values());share = {g: v / total for g, v in gen_2048.items()};basis = 'generated ms at 2,048 lanes'
    else:
        total = sum(r['n'] for r in census_groups.values());share = {g: r['n'] / total for g, r in census_groups.items()};basis = 'census equations (no timing)'
    fg = share.get('F', 0.) + share.get('G', 0.);e = share.get('E', 0.)
    m2 = 'F + G' if fg >= FG_MIN_SHARE else 'D'
    return dict(basis=basis, shares=share, fg_share=fg, e_share=e, M1='hooks (first, regardless)',
        M2=m2, M2_reason=f'F + G = {100 * fg:.1f}% of the step ' + ('>= 15%: keep F + G' if fg >= FG_MIN_SHARE else '< 15%: M2 becomes D'),
        M3='A-D before E (drag needs the entry library; Pursuit needs the damage pipeline)',
        M4='E split (E1-E2 | E3-E4)' if e > E_SPLIT_SHARE else 'E whole',
        M4_reason=f'E = {100 * e:.1f}% ' + ('> 40%: split' if e > E_SPLIT_SHARE else '<= 40%: one milestone'))


def fmt(x, digits=2, percent=False):
    if x is None:return '-'
    return f'{100 * x:.0f}%' if percent else (f'{x:,.{digits}f}' if isinstance(x, float) else f'{x:,}')


def cmd_report(args):
    out = new_out(args.out)
    load = lambda d, name: json.loads(Path(d, name).read_text()) if d and Path(d, name).exists() else None
    census, marks, walls = load(args.census, 'census.json'), load(args.marks, 'marks.json'), load(args.walls, 'walls.json')
    jx, div, pricer = load(args.jax, 'jax_attrib.json'), load(args.divergence, 'divergence.json'), load(args.pricer, 'pricer_split.json')
    nsys = nsys_chunks(args.nsys_csv) if args.nsys_csv and Path(args.nsys_csv).exists() else {}
    ncu = ncu_summary(args.ncu_details, args.ncu_raw) if args.ncu_details or args.ncu_raw else {}
    groups = census['census']['groups'] if census else {}
    lines = (census or {}).get('device_lines') or {}
    gen, jax_ms, sources, warnings, jax_fused = {}, {}, {}, [], {}
    chunk_counts = dict(marks=((marks or {}).get('module') or {}).get('chunks'), walls=(walls or {}).get('chunks'))
    for lanes in (128, 2048):
        entry = (marks or {}).get('lanes', {}).get(str(lanes));w = (walls or {}).get('lanes', {}).get(str(lanes))
        wall = (w or {}).get('generated', {}).get('ms_median')
        chunk_ns = ((w or {}).get('generated_trace') or {}).get('chunk_ns_median') or nsys.get(lanes)
        if chunk_ns and chunk_counts.get('marks') is not None and chunk_counts.get('walls') is not None and chunk_counts['marks'] != chunk_counts['walls']:
            warnings.append(f"{lanes} lanes: per-chunk times not used: the profile build has {chunk_counts['marks']} chunks, the timed build "
                f"{chunk_counts['walls']}");chunk_ns = None
        if entry:
            gen[lanes], sources[lanes] = generated_ms(entry, wall, chunk_ns)
            if 'FALLBACK' in (sources[lanes] or ''):warnings.append(f'{lanes} lanes: {sources[lanes]}')
        j = (jx or {}).get('lanes', {}).get(str(lanes))
        if j:jax_ms[lanes] = j['ms_per_tick_by_group'];jax_fused[lanes] = (j.get('multi_phase_fusion_share'), j.get('multi_group_fusion_share'))
    hook_cycles = {}
    if marks and '2048' in marks['lanes']:
        e = marks['lanes']['2048']
        hook_cycles = {g: c.get('tracker', 0) / sum(c.values()) for g, c in e['by_group_category'].items() if sum(c.values())}
    union = (div or {}).get('groups', {})
    total_eq = sum(r['n'] for r in groups.values()) if groups else None
    measured_2048 = sum((gen.get(2048) or {}).values()) or None
    rows = [];extra = sorted({k for d in [groups, *gen.values(), *jax_ms.values()] if d for k in d} - set(cz.GROUPS))
    for g in cz.GROUPS + tuple(extra):   # extra: equations outside step_with_chance (the chance block, the tracker wrap) or 'pre'
        r = groups.get(g, {})
        # Device lines: A counts the surviving host-check lines before 459 too, as the design's 33 does.
        dl = sum(v for k, v in lines.items() if cz.group_of(k) == g or (g == 'A' and cz.group_of(k) == 'pre')) if lines and g != 'pre' else None
        rows.append(dict(phase=g, device_lines=dl, equations=r.get('n'),
            generated_ms_128=(gen.get(128) or {}).get(g), generated_ms_2048=(gen.get(2048) or {}).get(g),
            jax_ms_128=(jax_ms.get(128) or {}).get(g), jax_ms_2048=(jax_ms.get(2048) or {}).get(g),
            hook_share_equations=r.get('hook_share'), hook_share_cycles_2048=hook_cycles.get(g), union_factor=(union.get(g) or {}).get('union_factor'),
            union_factor_design_uniform=(union.get(g) or {}).get('union_factor_design_uniform'),
            prior_ms_2048=(r['n'] / total_eq * (measured_2048 or PRIOR_GENERATED_MS_2048)) if r.get('n') and total_eq else None))
    decision = decide(gen.get(2048), {g: r for g, r in groups.items() if g in cz.GROUPS}) if (gen.get(2048) or groups) else None
    split = (pricer or {}).get('split')
    report = dict(schema=SCHEMA, kind='phase_attribution', inputs=dict(census=args.census, marks=args.marks, walls=args.walls, jax=args.jax,
        divergence=args.divergence, pricer=args.pricer, nsys_csv=args.nsys_csv, ncu_details=args.ncu_details, ncu_raw=args.ncu_raw),
        generated_ms_source=sources, chunk_counts=chunk_counts, warnings=warnings, rows=rows, decision=decision,
        weighted_union_factor=(div or {}).get('weighted_union_factor'), weighted_union_factor_design_uniform=(div or {}).get('weighted_union_factor_design_uniform'),
        union_uniform_shares=dict(census='each group\'s census hook share', design=(div or {}).get('design_uniform_share')),
        live_pricer_split=split, live_pricer_caveat=(pricer or {}).get('caveat'), ncu=ncu_by_group(ncu, marks),
        jax_attrib=dict(caveat=(jx or {}).get('caveat'), multi_phase_fusion_share={l: v[0] for l, v in jax_fused.items()},
            multi_group_fusion_share={l: v[1] for l, v in jax_fused.items()}),
        walls={l: dict(generated_ms=(v.get('generated') or {}).get('ms_median'), jax_ms=(v.get('jax') or {}).get('ms_median'))
            for l, v in ((walls or {}).get('lanes') or {}).items()},
        observe=dict(status=M6_STATUS, prior_generated_observe_ms_2048=PRIOR_OBSERVE_MS_2048, measured_generated_step_ms_2048=measured_2048,
            rule='M6 (observe) is independent of the step milestones: start it alongside M1 once M0 lands if observe stays a large share'))
    write_json(out / 'phase_attribution.json', report)
    md = ['| Phase | Device lines | Equations | Generated ms (128) | Generated ms (2,048) | JAX ms (128) | JAX ms (2,048) | Hook share (eq / cycles) | '
        f"Union factor (census u / u={fmt((div or {}).get('design_uniform_share'))}) | Prior ms (2,048) |",
        '| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
    for r in rows:
        md.append(f"| {r['phase']} | {fmt(r['device_lines'])} | {fmt(r['equations'])} | {fmt(r['generated_ms_128'])} | {fmt(r['generated_ms_2048'])} | "
            f"{fmt(r['jax_ms_128'])} | {fmt(r['jax_ms_2048'])} | {fmt(r['hook_share_equations'], percent=True)} / {fmt(r['hook_share_cycles_2048'], percent=True)} | "
            f"{fmt(r['union_factor'])} / {fmt(r['union_factor_design_uniform'])} | {fmt(r['prior_ms_2048'])} |")
    tot = lambda key: sum(r[key] for r in rows if r[key] is not None) if any(r[key] is not None for r in rows) else None
    md.append(f"| **Total** | {fmt(tot('device_lines'))} | {fmt(tot('equations'))} | {fmt(tot('generated_ms_128'))} | {fmt(tot('generated_ms_2048'))} | "
        f"{fmt(tot('jax_ms_128'))} | {fmt(tot('jax_ms_2048'))} | | {fmt(report['weighted_union_factor'])} / {fmt(report['weighted_union_factor_design_uniform'])} | "
        f"{fmt(tot('prior_ms_2048'))} |")
    md += [f'Generated ms ({l} lanes): {s}.' for l, s in sources.items() if s] + [f'WARNING: {w}' for w in warnings]
    md += [f'JAX ms ({l} lanes): each fused kernel is charged to its root op\'s phase; {fmt(a, percent=True)} of step kernel time is in fusions '
        f'spanning more than one phase ({fmt(b, percent=True)} more than one group).' for l, (a, b) in jax_fused.items()]
    for key, k in ncu.items():
        cpi = (k['metrics'].get('Warp Cycles Per Issued Instruction') or {}).get('value');l1 = (k['metrics'].get('L1/TEX Hit Rate') or {}).get('value')
        l2 = (k['metrics'].get('L2 Hit Rate') or {}).get('value');top = ', '.join(f'{r} {v:.1f}' for r, v in list(k['stalls'].items())[:3])
        md.append(f"ncu {k['kernel'][:60]} ({k.get('group') or '-'}): CPI {fmt(cpi)}, L1 hit {fmt(l1)}%, L2 hit {fmt(l2)}%; stalls {top or '-'}.")
    if split:
        md += ['', f"Live pricer split (bucket 1, W=16, capacity 8): leaf forward {fmt(split.get('forward_ms'))} ms ({fmt(split.get('forward_share'), percent=True)}), "
            f"engine {fmt(split.get('engine_ms'))} ms ({fmt(split.get('engine_share'), percent=True)}; includes P1's extra work over D3 depth 1, e.g. the "
            f"opponent-view child-prior forward), depth 1 {fmt(split.get('depth1_ms'))} ms ({fmt(split.get('depth1_share'), percent=True)}), "
            f"total {fmt(split.get('total_ms'))} ms per call. Every forward zeroed (P1's included): all forwards {fmt(split.get('all_forwards_ms'))} ms "
            f"({fmt(split.get('all_forwards_share'), percent=True)}), engine total {fmt(split.get('engine_total_ms'))} ms "
            f"({fmt(split.get('engine_total_share'), percent=True)})."]
        tables = (pricer or {}).get('tables') or {}
        if tables:md.append(f"Pricer tables: bank executor digest {str(tables.get('bank_digest'))[:16]}; live comparison: {tables.get('live_comparison') or 'not run'}.")
    if decision:
        md += ['', f"Decision ({decision['basis']}): M1 {decision['M1']}; M2 = {decision['M2']} ({decision['M2_reason']}); M3 {decision['M3']}; "
            f"M4 = {decision['M4']} ({decision['M4_reason']}); M6 (observe): {M6_STATUS}."]
    (out / 'phase_attribution.md').write_text('\n'.join(md) + '\n')
    print('\n'.join(md))
    if decision:print(f"RECOMMENDED M2: {decision['M2']}\nRECOMMENDED M4: {decision['M4']}\nM6: {M6_STATUS}")
    for w in warnings:print(f'WARNING: {w}', file=sys.stderr)
    return 0


# ---------------------------------------------------------------------------

def parser():
    p = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    sub = p.add_subparsers(dest='command', required=True)
    def run_args(q, lanes):
        q.add_argument('--config', required=True);q.add_argument('--reference-run', required=True)
        q.add_argument('--arm', default='B', choices=('A', 'B'));q.add_argument('--shards', required=True, help='capture shards/ directory')
        q.add_argument('--games', type=int, default=1, help='games constructed for the template (structure only)')
        q.add_argument('--build-dir', default=None);q.add_argument('--out', required=True, help='new output directory')
        if lanes is not None:q.add_argument('--lanes', type=int, nargs='+', default=lanes)
    q = sub.add_parser('marks', help='profile-build lane-0 cycles per phase');run_args(q, [128, 2048])
    q.add_argument('--limit', type=int, default=20, help='shards per lane count')
    q = sub.add_parser('walls', help='warm per-call wall of the normal build, JAX and per-chunk kernel times');run_args(q, [128, 2048, 8192])
    q.add_argument('--calls', type=int, default=50);q.add_argument('--warm', type=int, default=5);q.add_argument('--trace-calls', type=int, default=5)
    q.add_argument('--no-trace', action='store_true');q.add_argument('--no-jax', action='store_true')
    q = sub.add_parser('ncu-targets', help='print the ncu kernel regex from marks');q.add_argument('--marks', required=True)
    q.add_argument('--lanes', type=int, default=128);q.add_argument('--top', type=int, default=2);q.add_argument('--no-hand-kernels', action='store_true')
    q = sub.add_parser('ncu-driver', help='one call of the normal step (and the hand kernels) for ncu');run_args(q, None)
    q.add_argument('--lanes', type=int, default=128);q.add_argument('--calls', type=int, default=1);q.add_argument('--hand-kernels', action='store_true')
    q.add_argument('--marks', help='marks output directory: refuse when the profile build chunked differently from this build')
    q = sub.add_parser('jax-attrib', help='JAX kernel time per phase from a v2_engine_profile --save-trace directory')
    q.add_argument('--profile-dir', required=True);q.add_argument('--lanes', type=int, nargs='+', default=[128, 2048])
    q.add_argument('--chunk-steps', type=int, default=8);q.add_argument('--out', required=True)
    q = sub.add_parser('divergence', help='32-lane union of path classes per phase (CPU)');run_args(q, None)
    q.add_argument('--limit', type=int, default=100);q.add_argument('--warp', type=int, default=32);q.add_argument('--census', help='census.json (weights)')
    q.add_argument('--design-uniform-share', type=float, default=DESIGN_UNIFORM_SHARE, help='second uniform share for the union factor (design: 0.7)')
    q = sub.add_parser('pricer-split', help='live DepthN pricer: leaf forward vs engine (d3_depth2_throughput --profile-split method)')
    q.add_argument('--d3-root', required=True);q.add_argument('--chunk', type=int, required=True);q.add_argument('--memory-policy', action='store_true')
    q.add_argument('--max-roots', type=int, default=None);q.add_argument('--repeats', type=int, default=3)
    q.add_argument('--matmul-precision', default='highest', choices=('highest', 'high', 'default'))
    q.add_argument('--compilation-cache', help='absolute directory for JAX\'s persistent compilation cache');q.add_argument('--out', required=True)
    q.add_argument('--live-metadata', help='the live server\'s engine metadata (its --metadata): compare its compiled tables with the bank executor\'s')
    q.add_argument('--live-metadata-sha256', help='sha256 of --live-metadata (required with it)')
    q.add_argument('--live-tables-digest', help='a recorded constructor_worlds.engine_tables_digest of the live executor\'s tables to compare with')
    from gpu_engine.d3_depth2_gate import add_head_arguments
    from gpu_engine import champion_evaluator as ce
    add_head_arguments(q);ce.add_arguments(q)
    q = sub.add_parser('report', help='phase_attribution.json and the table, with the decision rule')
    for name in ('census', 'marks', 'walls', 'jax', 'divergence', 'pricer'):q.add_argument('--' + name, help=f'{name} output directory')
    q.add_argument('--nsys-csv', help='nsys stats cuda_gpu_trace CSV (per-chunk times when walls has no trace)')
    q.add_argument('--ncu-details', help='ncu --page details --csv export');q.add_argument('--ncu-raw', help='ncu --page raw --csv export')
    q.add_argument('--out', required=True)
    return p


COMMANDS = {'marks': cmd_marks, 'walls': cmd_walls, 'ncu-targets': cmd_ncu_targets, 'ncu-driver': cmd_ncu_driver, 'jax-attrib': cmd_jax_attrib,
    'divergence': cmd_divergence, 'pricer-split': cmd_pricer_split, 'report': cmd_report}


def main(argv=None):
    p = parser();args = p.parse_args(argv)
    if args.command == 'pricer-split':
        from gpu_engine.d3_depth2_gate import check_head_arguments
        from gpu_engine import champion_evaluator as ce
        check_head_arguments(p, args);ce.check_arguments(p, args)
        if not args.champion_archive:p.error('pricer-split prices with the champion, as the live pricer: --champion-archive and its sha256s required')
        if bool(args.live_metadata) != bool(args.live_metadata_sha256):p.error('--live-metadata and --live-metadata-sha256 go together')
    return COMMANDS[args.command](args)


if __name__ == '__main__':raise SystemExit(main())
