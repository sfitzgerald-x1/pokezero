"""DepthNPricer throughput on one D3 bank chunk: depth N (default 3), k in --ks, W in --worlds, prior and searched.

Requests are the chunk's, built as d3_depth2_gate --d3-root --chunk builds them (imported, not copied), with the gate's
head override and evaluator header. Every call prices C roots (--capacity, the chunk's requests in order, cycled when it
has fewer) with fixed_capacity=True and digest=False, so one compiled program serves every call. W = 2K worlds per root:
K recipe blocks (U1's world_block_seeds, quarter selection bypassed after block 0, as the gate), each block one call;
per block 1 cold call (compile) and --repeats warm calls. Configs (each measured at every W):
- depth1: D3's FramePricer at the same C (P1 is D3's turn 1: the split's P1 reference);
- depth2-k{k}: DepthNPricer(depth=2, k, prior), the split's per-lane leaf cost;
- depth2-fw: DepthNPricer(depth=2, k=9), the full-width depth-2 pass of `searched`;
- depth{N}-prior-k{k}: DepthNPricer(depth=N, k at every ply, prior);
- depth{N}-searched-k{k}: per call the full-width depth-2 pass, the public searched_maps(frame, k, k) on the host, then
  DepthNPricer(depth=N, selection='searched') at the widths of the maps it returned (never assumed k + 1). Its device
  seconds are the pass's plus depth N's; the maps' host seconds and every part are reported.
Per config and W: device ms per root (the sum over blocks of each block's warm median device seconds / C, the pricer's
timing) and wall ms per root (frame_wall_seconds: the host loop over plies and packing included) against the 1,000 ms
budget, leaves enumerated/evaluated, compile seconds (cold) and memory (process peak and peak over the config's
baseline; --isolate gives each config its own process). Per-ply device split (an ESTIMATE, labelled): P1 = depth1's
device seconds; the leaf ply = (depth2-k{k} - depth1) / its evaluated leaf lanes x depth N's evaluated leaf lanes;
the expand plies = the rest, divided by their evaluated lanes. Progress is JSON lines; the result is throughput.json in
a new --out directory.
"""
import argparse
from contextlib import nullcontext
from dataclasses import replace
import importlib
import json
from pathlib import Path
from types import SimpleNamespace
import time
from unittest.mock import patch

import numpy as np

from . import d3_depth2_throughput as t2
from .d3_depthn_gate import split_searched

SCHEMA = 'pokezero.depthn-throughput.v1'
BUDGET_MS = 1000.
MODES = ('prior', 'searched')


def config_names(depth, ks, modes):
    names = ['depth1'] + [f'depth2-k{k}' for k in ks] + (['depth2-fw'] if 'searched' in modes else [])
    return names + [f'depth{depth}-{mode}-k{k}' for mode in modes for k in ks]


def parse_config(name):
    """'depth1' | 'depth2-fw' | 'depth{N}-{mode}-k{k}' (depth2-k{k} is prior) -> dict(kind, depth, mode, k)."""
    if name == 'depth1':return dict(kind='depth1', depth=1, mode=None, k=None)
    if name == 'depth2-fw':return dict(kind='depthn', depth=2, mode='prior', k=9)
    head, *rest = name.split('-');depth = int(head[len('depth'):])
    if len(rest) == 1:return dict(kind='depthn', depth=depth, mode='prior', k=int(rest[0][1:]))
    return dict(kind='searched' if rest[0] == 'searched' else 'depthn', depth=depth, mode=rest[0], k=int(rest[1][1:]))


def ints(text):
    return tuple(int(x) for x in str(text).split(','))


def parse(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('--d3-root', required=True, help='the D3 bank run root (prespec.json, runtime.json, run/)')
    parser.add_argument('--chunk', type=int, required=True, help='bank chunk index; chunks 0..N are collected')
    parser.add_argument('--memory-policy', action='store_true', help="apply the bank runtime's JAX memory policy")
    parser.add_argument('--out', required=True, help='new output directory')
    parser.add_argument('--capacity', type=int, required=True, help='roots per call C (fixed_capacity=True)')
    parser.add_argument('--depth', type=int, default=3, help='search depth N >= 3 (default 3)')
    parser.add_argument('--ks', default='2,3,4', help='k values (k_own = k_opp = k at every ply)')
    parser.add_argument('--worlds', default='2,8', help='W values: worlds per root, even (W = 2K recipe blocks)')
    parser.add_argument('--modes', default='prior,searched', help='selections to time')
    parser.add_argument('--k1-own', type=int, default=9, help='turn-1 learner width');parser.add_argument('--k1-opp', type=int, default=9, help='turn-1 opponent width')
    parser.add_argument('--block-lanes', type=int, default=2187, help='DepthNPricer block_lanes')
    parser.add_argument('--repeats', type=int, default=3, help='warm calls after the cold one, per block')
    parser.add_argument('--max-roots', type=int, help='use only the first N requests of the chunk')
    parser.add_argument('--isolate', action='store_true', help='measure each config in its own subprocess (each reloads the chunk)')
    parser.add_argument('--only', help=argparse.SUPPRESS)   # one config; used by --isolate
    from gpu_engine.d3_depth2_gate import add_head_arguments, check_head_arguments   # numpy only: no JAX before the memory policy
    add_head_arguments(parser)
    args = parser.parse_args(argv)
    check_head_arguments(parser, args)
    try:args.k_values, args.w_values, args.mode_values = ints(args.ks), ints(args.worlds), tuple(args.modes.split(','))
    except ValueError:parser.error('--ks/--worlds: comma-separated integers')
    if args.chunk < 0 or args.capacity < 1 or args.repeats < 1 or args.block_lanes < 1:parser.error('--chunk >= 0 and positive --capacity/--repeats/--block-lanes')
    if args.depth < 3:parser.error('--depth >= 3 required (depth 2 is d3_depth2_throughput)')
    if any(not 1 <= k <= 9 for k in args.k_values + (args.k1_own, args.k1_opp)):parser.error('widths must be 1..9')
    if any(w < 2 or w % 2 for w in args.w_values):parser.error('--worlds: even W >= 2 (two worlds per recipe block)')
    if any(m not in MODES for m in args.mode_values):parser.error(f'--modes from {MODES}')
    if args.max_roots is not None and args.max_roots < 1:parser.error('positive --max-roots required')
    args.configs = config_names(args.depth, args.k_values, args.mode_values)
    if args.only is not None and args.only not in args.configs:parser.error(f'--only one of {args.configs}')
    if args.isolate and args.only:parser.error('--isolate runs --only itself')
    return args


def prepare(args):
    """The chunk's requests (the gate's builder), the evaluator and the planned blocks for the largest W (W = 2 uses block 0,
    W = 2K the first K blocks: world_block_seeds is prefix-stable)."""
    from gpu_engine import d3_depth2_gate as gate
    with gate.stage(gate.log, 'load_bank_chunk', chunk=args.chunk):loaded = gate.bank_chunk_requests(args)
    provider, plan, params = loaded.provider, loaded.plan, loaded.params
    head = gate.load_head(args, plan=plan, labels=loaded.labels, category_sha256=loaded.category_sha256) if args.head_archive else None
    if head is not None:params = (head[0],)
    groups, root_values, dropped, arm_checked = gate.bind_roots(loaded.groups, params, plan, head_config=None if head is None else head[1])
    evaluator = dict(gate.evaluator_header(args, loaded, head), arm_checked=arm_checked, dropped_roots=dropped, weights_affect_speed=t2.WEIGHTS_NOTE)
    gate.log('EVALUATOR_RESOLVED', evaluator=evaluator, root_value_scale=gate.percentiles(root_values))
    from pokezero import v5_gumbel as original
    from gpu_engine import device_public_worlds as worlds
    available = [r for g in groups for r in g][:args.max_roots];roots = t2.cycle_roots(available, args.capacity)
    seeds = gate.world_block_seeds(provider.recipe.seed, max(args.w_values) // 2);planned = []
    for block, seed in enumerate(seeds):
        recipe = provider.recipe if block == 0 else original.Recipe(seed)
        context = (lambda: patch.object(original.Recipe, 'selected', lambda self, address: True)) if block else nullcontext
        with context():
            requests = roots if block == 0 else [replace(r, plans=worlds.plan_starts(r.plans.root, provider.source, recipe)) for r in roots]
        planned.append((block, recipe, requests, context))
    return SimpleNamespace(provider=provider, plan=plan, params=params, source=loaded.source, evaluator=evaluator, root_values=root_values,
        available=available, seeds=seeds, planned=planned, actor_capacity=len(params) if head is not None else plan.historical_window + 1)


def compiled_seconds(pricer):
    return sum(c[1] + c[2] for c in getattr(pricer, 'compiles', ()))


class Caller:
    """One config's call: price C roots, return the record (device/wall/compile seconds, leaves, plies, parts, memory)."""
    def __init__(self, config, args, ctx, module=None, frame_pricer_class=None):
        self.config, self.args, self.ctx = parse_config(config), args, ctx;self.name = config;self.cache = {}
        self.module = module or importlib.import_module('gpu_engine.device_depthn_pricer')
        kind = self.config['kind'];executor, model = ctx.provider.executor, ctx.plan.model
        if kind == 'depth1':
            if frame_pricer_class is None:from gpu_engine.device_public_teacher_batch import FramePricer as frame_pricer_class
            self.pricer = frame_pricer_class(executor, model, capacity=args.capacity, block_rows=args.capacity, actor_capacity=ctx.actor_capacity,
                vectorized=True, host_prep=True)
        elif kind == 'depthn':self.pricer = self.depthn(self.config['depth'], self.config['k'], self.config['k'], 'prior')
        else:self.pass_pricer = self.depthn(2, 9, 9, 'prior')

    def depthn(self, depth, k_own, k_opp, selection):
        key = (depth, k_own, k_opp, selection)
        if key not in self.cache:
            self.cache[key] = self.module.DepthNPricer(self.ctx.provider.executor, self.ctx.plan.model, capacity=self.args.capacity, depth=depth,
                k_own=k_own, k_opp=k_opp, k1_own=self.args.k1_own, k1_opp=self.args.k1_opp, block_lanes=self.args.block_lanes, selection=selection,
                fixed_capacity=True, digest=False, actor_capacity=self.ctx.actor_capacity)
        return self.cache[key]

    def _depthn(self, pricer, requests, recipe, **kw):
        before = compiled_seconds(pricer);frame = pricer.price(requests, self.ctx.provider.materializer, recipe, self.ctx.params, **kw);timing = frame.timing
        return frame, dict(device_seconds=timing['device_seconds'], wall_seconds=timing.get('frame_wall_seconds'), compile_seconds=compiled_seconds(pricer) - before,
            leaves_enumerated=timing.get('enumerated_leaves'), leaves_evaluated=timing.get('evaluated_leaf_lanes'),
            plies=json.loads(json.dumps(timing.get('plies', ()), default=str)), bytes_per_node_state=timing.get('bytes_per_node_state'))

    def __call__(self, requests, recipe):
        kind = self.config['kind']
        if kind == 'depth1':
            results = self.pricer.price(requests, self.ctx.provider.materializer, recipe, self.ctx.params);timing = self.pricer.last_timing
            pairs = sum(r.get('required_pairs') or 0 for result in results for r in result.world_reports)
            return dict(device_seconds=timing['device_seconds'], wall_seconds=timing.get('frame_wall_seconds'),
                compile_seconds=(timing.get('lower_seconds') or 0.) + (timing.get('compile_seconds') or 0.), leaves_enumerated=pairs,
                leaves_evaluated=timing.get('evaluated_joint_slots'))
        if kind == 'depthn':return self._depthn(self.pricer, requests, recipe)[1]
        raw, first = self._depthn(self.pass_pricer, requests, recipe)
        clock = time.perf_counter();own, opp, _ = split_searched(self.module.searched_maps(self.module.as_depth2_frame(raw), self.config['k'], self.config['k']))
        host = time.perf_counter() - clock
        widths = (int(own.shape[-1]), int(opp.shape[-1]));rest = (self.config['k'],) * (self.config['depth'] - 2)
        pricer = self.depthn(self.config['depth'], widths[:1] + rest, widths[1:] + rest, 'searched')
        _, second = self._depthn(pricer, requests, recipe, selection_maps={2: (own, opp)})
        return dict(second, device_seconds=first['device_seconds'] + second['device_seconds'],
            wall_seconds=None if None in (first['wall_seconds'], second['wall_seconds']) else first['wall_seconds'] + host + second['wall_seconds'],
            compile_seconds=first['compile_seconds'] + second['compile_seconds'], searched_maps_host_seconds=host, maps_widths=list(widths),
            parts=dict(depth2_full_width_pass=first, searched_maps_host_seconds=host, depthn=second))


def measure(caller, planned, repeats, *, memory, progress):
    """Per block: 1 cold + `repeats` warm calls; the memory baseline is taken before the config's first call."""
    from gpu_engine.d3_depth2_gate import stage
    blocks, baseline = [], memory() or {}
    for block, recipe, requests, context in planned:
        calls = []
        for i in range(1 + repeats):
            with stage(progress, f'{caller.name}_call', block=block, call=i, warm=i > 0, roots=len(requests)) as info, context():
                record = dict(caller(requests, recipe), memory=memory());calls.append(record)
                info.update({k: v for k, v in record.items() if k not in ('plies', 'parts')})
        warm = calls[1:];device = [c['device_seconds'] for c in warm];wall = [c['wall_seconds'] for c in warm if c.get('wall_seconds') is not None]
        peaks = [c['memory'].get('process_peak_bytes_so_far') for c in calls if c.get('memory') and c['memory'].get('process_peak_bytes_so_far') is not None]
        peak = max(peaks) if peaks else None
        blocks.append(dict(block=block, cold_compile_seconds=calls[0]['compile_seconds'], warm_device_seconds=device, warm_device_seconds_median=t2.median(device),
            warm_device_seconds_min=min(device) if device else None, warm_wall_seconds_median=t2.median(wall), leaves_enumerated=calls[0].get('leaves_enumerated'),
            leaves_evaluated=calls[0].get('leaves_evaluated'), plies=calls[0].get('plies'), maps_widths=calls[0].get('maps_widths'),
            searched_maps_host_seconds_median=t2.median([c['searched_maps_host_seconds'] for c in warm if 'searched_maps_host_seconds' in c]),
            parts_warm_median=None if 'parts' not in calls[0] else dict(depth2_full_width_pass=t2.median([c['parts']['depth2_full_width_pass']['device_seconds'] for c in warm]),
                depthn=t2.median([c['parts']['depthn']['device_seconds'] for c in warm])),
            process_peak_bytes_so_far=peak, memory_baseline=baseline,
            peak_over_baseline_in_use=None if peak is None or baseline.get('bytes_in_use') is None else peak - baseline['bytes_in_use'], calls=calls))
    return blocks


def derive(blocks, capacity, worlds):
    """One W: the first W/2 blocks' warm medians summed, per root, against the budget."""
    used = blocks[:worlds // 2];total = lambda key: None if any(b[key] is None for b in used) else sum(b[key] for b in used)
    device, wall = total('warm_device_seconds_median'), total('warm_wall_seconds_median')
    ms = lambda seconds: None if seconds is None else 1e3 * seconds / capacity
    leaves = lambda key: None if any(b.get(key) is None for b in used) else sum(b[key] for b in used)
    out = dict(worlds=worlds, blocks=len(used), roots_per_call=capacity, device_ms_per_root=ms(device), wall_ms_per_root=ms(wall), budget_ms_per_root=BUDGET_MS,
        device_budget_ratio=None if device is None else ms(device) / BUDGET_MS, wall_budget_ratio=None if wall is None else ms(wall) / BUDGET_MS,
        within_budget_device=None if device is None else ms(device) <= BUDGET_MS, within_budget_wall=None if wall is None else ms(wall) <= BUDGET_MS,
        leaves_enumerated=leaves('leaves_enumerated'), leaves_evaluated=leaves('leaves_evaluated'),
        device_ms_per_root_min=ms(total('warm_device_seconds_min')), cold_compile_seconds=used[0]['cold_compile_seconds'] if used else None,
        peak_over_baseline_in_use=max((b['peak_over_baseline_in_use'] for b in used if b['peak_over_baseline_in_use'] is not None), default=None))
    if used and used[0].get('parts_warm_median'):
        parts = [b['parts_warm_median'] for b in used]
        out['parts_ms_per_root'] = {k: ms(sum(p[k] for p in parts)) for k in parts[0]}
        out['parts_ms_per_root']['searched_maps_host'] = ms(sum(b['searched_maps_host_seconds_median'] or 0. for b in used))
    return out


def ply_split(depthn, depth2, depth1):
    """ESTIMATE of one W's depth-N device ms/root by stage from derived sections and the depth-N blocks' ply work:
    P1 = depth1's; leaf ply = (depth2-k - depth1) / depth2-k's evaluated leaf lanes x depth N's evaluated leaf lanes; the
    expand plies the remainder, apportioned by their evaluated lanes. None where an input is missing."""
    d, two, one = depthn['derived'], depth2['derived'], depth1['derived']
    if None in (d['device_ms_per_root'], two['device_ms_per_root'], one['device_ms_per_root']) or not two['leaves_evaluated'] or d['leaves_evaluated'] is None:return None
    total = d['device_ms_per_root'] - (d.get('parts_ms_per_root') or {}).get('depth2_full_width_pass', 0.)
    per_lane = (two['device_ms_per_root'] - one['device_ms_per_root']) / two['leaves_evaluated'];leaf = per_lane * d['leaves_evaluated']
    plies = depthn['blocks'][0].get('plies') or ();expand = [p for p in plies if int(p.get('ply', 0)) < max((int(x.get('ply', 0)) for x in plies), default=0)]
    rest = total - one['device_ms_per_root'] - leaf;lanes = sum(float(p.get('lanes') or 0) for p in expand)
    return dict(estimate=True, method=ply_split.__doc__.split('\n')[0].strip() + ' ' + ' '.join(x.strip() for x in ply_split.__doc__.split('\n')[1:3]),
        p1_ms_per_root=one['device_ms_per_root'], leaf_ply_ms_per_root=leaf, expand_plies_ms_per_root=rest,
        expand_by_ply={str(p.get('ply')): None if not lanes else rest * float(p.get('lanes') or 0) / lanes for p in expand},
        depth2_full_width_pass_ms_per_root=(d.get('parts_ms_per_root') or {}).get('depth2_full_width_pass'),
        searched_maps_host_ms_per_root=(d.get('parts_ms_per_root') or {}).get('searched_maps_host'),
        caveat='separate programs and batch shapes; XLA may schedule them differently; host time (packing, the ply loop) is in wall_ms_per_root, not here')


def measure_config(config, args, ctx, progress, *, module=None):
    caller = Caller(config, args, ctx, module=module);blocks = measure(caller, ctx.planned, args.repeats, memory=t2.memory_stats, progress=progress)
    return dict(config=parse_config(config), blocks=blocks, derived={str(w): derive(blocks, args.capacity, w) for w in args.w_values})


def isolated(args, argv, out):
    """--isolate: each config in its own subprocess (`--only CONFIG --out OUT/CONFIG`); returns ({config: section}, contexts)."""
    import subprocess, sys
    argv = list(sys.argv[1:] if argv is None else argv);clean, skip = [], False
    for token in argv:
        if skip:skip = False;continue
        if token == '--isolate':continue
        if token == '--out':skip = True;continue
        if token.startswith('--out='):continue
        clean.append(token)
    sections, contexts = {}, {}
    for config in args.configs:
        target = out / config
        code = subprocess.call([sys.executable, '-m', 'gpu_engine.d3_depthn_throughput', *clean, '--only', config, '--out', str(target)])
        if code != 0:raise SystemExit(f'--isolate: the {config} subprocess exited {code}')
        child = json.loads((target / 'throughput.json').read_text());sections[config] = child['sections'][config];contexts[config] = child.get('context')
    return sections, contexts


def report_of(args, sections, context, *, isolated_run):
    from gpu_engine import d3_depth2_gate as gate
    by_w = {}
    for w in map(str, args.w_values):
        rows = {name: s['derived'][w] for name, s in sections.items()}
        splits = {}
        for name, s in sections.items():
            c = s['config']
            if c['depth'] >= 3 and f'depth2-k{c["k"]}' in sections and 'depth1' in sections:
                d = dict(blocks=s['blocks'], derived=s['derived'][w])
                splits[name] = ply_split(d, dict(derived=sections[f'depth2-k{c["k"]}']['derived'][w]), dict(derived=sections['depth1']['derived'][w]))
        by_w[w] = dict(configs=rows, ply_split_estimate=splits)
    return dict(evaluator=context['evaluator'], value_scale=gate.value_scale(context['root_values'], dict(count=0), args.value_reference),
        weights_affect_speed=t2.WEIGHTS_NOTE, schema=SCHEMA, source=context['source'], capacity=args.capacity, depth=args.depth, ks=list(args.k_values),
        worlds=list(args.w_values), modes=list(args.mode_values), block_lanes=args.block_lanes, k1_own=args.k1_own, k1_opp=args.k1_opp, repeats=args.repeats,
        roots_available=context['roots_available'], cycled=context['roots_available'] < args.capacity, devices=context['devices'], measured_order=list(sections),
        isolated=isolated_run, memory_scope=t2.MEMORY_SCOPE + (' (isolated: one process per config)' if isolated_run else ''), budget_ms_per_root=BUDGET_MS,
        by_worlds=by_w, sections=sections,
        scope='device_seconds = DepthNPricer/FramePricer timing; wall = frame_wall_seconds (host ply loop and packing included); searched = the '
            'full-width depth-2 pass + searched_maps (host) + depth N at the maps\' widths; ms per root = sum over the W/2 blocks of warm medians / C')


def main(argv=None):
    args = parse(argv)
    import importlib.util   # fail before loading the bank; find_spec does not import JAX
    if importlib.util.find_spec('gpu_engine.device_depthn_pricer') is None:raise SystemExit('gpu_engine.device_depthn_pricer is not on this tree')
    from gpu_engine import d3_depth2_gate as gate
    gate.log('EVALUATOR', override=bool(args.head_archive), archive=args.head_archive, archive_sha256=args.head_archive_sha256,
        checkpoint_sha256=args.head_checkpoint_sha256, value_reference=args.value_reference, weights_affect_speed=t2.WEIGHTS_NOTE,
        resolved='EVALUATOR_RESOLVED follows once the bank is loaded')
    out = Path(args.out);out.mkdir(parents=True, exist_ok=False)
    if args.isolate:
        gate.log('THROUGHPUT_START', configs=args.configs, roots_per_call=args.capacity, isolate=True)
        sections, contexts = isolated(args, argv, out);context = next(c for c in contexts.values() if c)
    else:
        ctx = prepare(args)
        import jax
        context = dict(evaluator=ctx.evaluator, root_values=list(ctx.root_values), roots_available=len(ctx.available), source=ctx.source,
            devices=[d.device_kind for d in jax.devices()], block_seeds=list(ctx.seeds))
        configs = [args.only] if args.only else args.configs
        gate.log('THROUGHPUT_START', configs=configs, roots_available=context['roots_available'], roots_per_call=args.capacity, worlds=list(args.w_values), isolate=False)
        sections = {config: measure_config(config, args, ctx, gate.log) for config in configs}
        if args.only:
            (out / 'throughput.json').write_text(json.dumps(dict(evaluator=ctx.evaluator, only=args.only, sections=sections, context=context),
                indent=1, allow_nan=False, default=str) + '\n')
            return 0
    report = report_of(args, sections, context, isolated_run=args.isolate)
    (out / 'throughput.json').write_text(json.dumps(report, indent=1, allow_nan=False, default=str) + '\n')   # evaluator first
    gate.log('THROUGHPUT_COMPLETE', out=str(out / 'throughput.json'),
        device_ms_per_root={w: {n: r['device_ms_per_root'] for n, r in v['configs'].items()} for w, v in report['by_worlds'].items()})
    return 0


if __name__ == '__main__':raise SystemExit(main())
