"""Depth2Pricer throughput on one D3 bank chunk, optionally next to D3's depth-1 FramePricer.

Requests are the chunk's, built exactly as d3_depth2_gate --d3-root --chunk builds them (imported, not
copied). Every call prices C roots (--capacity), taken from the chunk in order and cycled when it has
fewer, with Depth2Pricer(fixed_capacity=True, capacity=C) (pricer d5150eac or later), so one compiled
program serves every call. Each of the K recipe blocks (--worlds-blocks; U1's world_block_seeds, quarter
selection bypassed after block 0, as the gate) runs 1 cold call and --repeats warm calls. Per block:
compile seconds (cold), device seconds (min and median over warm calls), turn-2 leaves enumerated and
evaluated (the pricer's timing) and device memory after each call. Derived at W = 2K: device ms per root
= the sum over blocks of each block's warm device seconds / C, and turn-2 leaves per second.

--d1 runs FramePricer(vectorized=True, host_prep=True) at the same capacity and blocks (device ms per
root, depth-1 pairs per second). --profile-split runs three warm variants of block 0's call: (i) the
normal call, (ii) the same call through Depth2Pricer's forward= seam with every turn-2 leaf forward
replaced by zeros of its output shapes, (iii) the D3 depth-1 call; (i)-(ii) estimates the turn-2
forward and (ii)-(iii) the rest of depth 2 beyond D3's turn 1. Turn-2 leaf forwards are recognized by
their static batch size at trace time (child_block*k_own*k_opp rows). If that size can coincide with a
turn-1 forward's, (ii) is skipped and the reason reported; the sizes actually traced are checked too.
Progress is JSON lines; the result is throughput.json in a new --out directory.

--head-archive/--head-archive-sha256/--head-checkpoint-sha256 price with that v5 head instead of the bank snapshot's
actor (the gate's override: U1's load, arm checks, the head's root logits, one actor). throughput.json and the first
JSON line carry the evaluator; device time does not depend on the weight values, so the evaluator is a label only.
--champion-archive/--champion-archive-sha256/--champion-checkpoint-sha256 [--champion-precision] (exclusive with
--head-archive) price with champion 9375 through the forward= seam (gpu_engine.champion_evaluator): a different network, so
here the evaluator does change the time. Depth2Pricer, --d1's FramePricer and --profile-split's three variants all take it
(the constant variant wraps it: turn-1 forwards stay the champion's, turn-2 leaf forwards become zeros of its output shapes).
"""
import argparse
from contextlib import nullcontext
from dataclasses import replace
import inspect
import json
from types import SimpleNamespace
from pathlib import Path
import statistics
import time
from unittest.mock import patch

import numpy as np

SCHEMA = 'pokezero.depth2-throughput.v1'
LANES = 81   # D3's fixed joint lanes per world
WEIGHTS_NOTE = 'false (device time independent of the weight values; label only)'
CHAMPION_SPEED_NOTE = ('true: champion 9375 is a different network from the V2 heads (layer norm, dense 2,048 FFN, 23 tokens), so the evaluator '
    'changes the device time; its weight values still do not')


def weights_note(args):
    """weights_affect_speed for the run's evaluator: a V2 head (bank actor or override) is a label only; the champion is not."""
    return CHAMPION_SPEED_NOTE if getattr(args, 'champion_archive', None) else WEIGHTS_NOTE
CONFIGS = ('depth1', 'depth2', 'constant')
MEMORY_SCOPE = ('process_peak_bytes_so_far is jax.local_devices()[0].memory_stats()[\'peak_bytes_in_use\'] after the call: the PROCESS '
    'lifetime peak so far, so it includes the bank load and any config measured earlier in the same process (see measured_order); '
    'use --isolate for one process per config. memory_baseline is taken just before a config\'s first measured call; '
    'peak_over_baseline_in_use = peak - baseline bytes_in_use, and peak_masked_by_earlier marks a config whose own peak is hidden '
    'below an earlier one (its peak did not exceed the baseline peak)')


def parse(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('--d3-root', required=True, help='the D3 bank run root (prespec.json, runtime.json, run/)')
    parser.add_argument('--chunk', type=int, required=True, help='bank chunk index; chunks 0..N are collected')
    parser.add_argument('--memory-policy', action='store_true', help="apply the bank runtime's JAX memory policy")
    parser.add_argument('--out', required=True, help='new output directory')
    parser.add_argument('--capacity', type=int, required=True, help='roots per call C: Depth2Pricer(fixed_capacity=True, capacity=C)')
    for name in ('k-own', 'k-opp', 'k1-own', 'k1-opp'):
        parser.add_argument('--' + name, type=int, default=9, help='width 1..9 (9 = full width)')
    parser.add_argument('--child-block', type=int, help='Depth2Pricer child_block (default: the pricer default)')
    parser.add_argument('--worlds-blocks', type=int, default=1, help='K recipe blocks, W = 2K worlds per root')
    parser.add_argument('--repeats', type=int, default=3, help='warm calls after the cold one, per block')
    parser.add_argument('--max-roots', type=int, help='use only the first N requests of the chunk')
    parser.add_argument('--d1', action='store_true', help="also time D3's depth-1 FramePricer at the same capacity and blocks")
    parser.add_argument('--profile-split', action='store_true', help='split block 0 device time: normal / constant turn-2 forward / depth 1')
    parser.add_argument('--isolate', action='store_true', help='measure each config (depth1, depth2, constant) in its own subprocess, so each '
        'memory peak is that config\'s own (each subprocess reloads the bank chunk)')
    parser.add_argument('--only', choices=CONFIGS, help=argparse.SUPPRESS)   # one config; used by --isolate
    from gpu_engine.d3_depth2_gate import add_head_arguments, check_head_arguments   # numpy only: no JAX before the memory policy
    from gpu_engine import champion_evaluator as ce
    add_head_arguments(parser);ce.add_arguments(parser)
    args = parser.parse_args(argv)
    check_head_arguments(parser, args);ce.check_arguments(parser, args)
    if args.chunk < 0:parser.error('--chunk N >= 0 required')
    if args.capacity < 1 or args.repeats < 1 or args.worlds_blocks < 1:parser.error('positive --capacity, --repeats and --worlds-blocks required')
    if any(not 1 <= v <= 9 for v in (args.k_own, args.k_opp, args.k1_own, args.k1_opp)):parser.error('widths must be 1..9')
    if args.child_block is not None and args.child_block < 1:parser.error('positive --child-block required')
    if args.max_roots is not None and args.max_roots < 1:parser.error('positive --max-roots required')
    if args.isolate and args.only:parser.error('--isolate runs --only itself')
    return args


def cycle_roots(requests, capacity):
    """C requests in chunk order, cycling from the start when the chunk has fewer."""
    requests = list(requests)
    if not requests:raise ValueError('the chunk has no priced request')
    return [requests[i % len(requests)] for i in range(capacity)]


def median(values):
    return None if not values else statistics.median(values)


def summarize_block(calls):
    """One block's calls (cold first): compile (cold), device min/median (warm), leaves and memory."""
    cold, warm = calls[0], calls[1:]
    device = [c['device_seconds'] for c in warm]
    peaks = [c['memory']['process_peak_bytes_so_far'] for c in calls if c.get('memory') and c['memory'].get('process_peak_bytes_so_far') is not None]
    return dict(cold_compile_seconds=cold['compile_seconds'], cold_device_seconds=cold['device_seconds'], warm_device_seconds=device,
        warm_device_seconds_min=min(device) if device else None, warm_device_seconds_median=median(device),
        leaves_enumerated=cold.get('leaves_enumerated'), leaves_evaluated=cold.get('leaves_evaluated'),
        leaves_constant=all(c.get('leaves_enumerated') == cold.get('leaves_enumerated') for c in warm),
        process_peak_bytes_so_far=max(peaks) if peaks else None, bytes_in_use_after=(calls[-1].get('memory') or {}).get('bytes_in_use'), calls=calls)


def derive(blocks, capacity):
    """W = 2K totals: device ms per root (sum of per-block warm device seconds / C) and leaves per second."""
    out = {}
    for stat in ('min', 'median'):
        seconds = [b[f'warm_device_seconds_{stat}'] for b in blocks]
        total = None if any(s is None for s in seconds) else sum(seconds)
        out[f'device_ms_per_root_{stat}'] = None if total is None else 1e3 * total / capacity
        for kind in ('enumerated', 'evaluated'):
            leaves = [b[f'leaves_{kind}'] for b in blocks]
            out[f'leaves_{kind}_per_second_{stat}'] = None if total is None or not total or any(x is None for x in leaves) else sum(leaves) / total
    out.update(worlds=2 * len(blocks), roots_per_call=capacity, leaves_enumerated=sum(b['leaves_enumerated'] or 0 for b in blocks),
        leaves_evaluated=sum(b['leaves_evaluated'] or 0 for b in blocks))
    return out


def depth2_call(pricer, requests, materializer, recipe, params, memory):
    frame = pricer.price(requests, materializer, recipe, params);timing = frame.timing
    t1, status = getattr(frame, 't1_value', None), getattr(frame, 't1_status', None)
    abs_t1 = None if t1 is None or status is None else np.abs(np.asarray(t1, np.float64)[np.asarray(status) == 2])
    return dict(compile_seconds=(timing.get('lower_seconds') or 0.) + (timing.get('compile_seconds') or 0.), device_seconds=timing['device_seconds'],
        leaves_enumerated=timing.get('turn2_enumerated_leaves'), leaves_evaluated=timing.get('turn2_evaluated_leaf_lanes'),
        frame_wall_seconds=timing.get('frame_wall_seconds'), memory=memory(), _abs_t1=abs_t1)


def depth1_call(pricer, requests, materializer, recipe, params, memory):
    """FramePricer: leaves are D's required turn-1 pairs (enumerated) and its evaluated joint lanes."""
    results = pricer.price(requests, materializer, recipe, params);timing = pricer.last_timing
    pairs = sum(r.get('required_pairs') or 0 for result in results for r in result.world_reports)
    return dict(compile_seconds=(timing.get('lower_seconds') or 0.) + (timing.get('compile_seconds') or 0.), device_seconds=timing['device_seconds'],
        leaves_enumerated=pairs, leaves_evaluated=timing.get('evaluated_joint_slots'), frame_wall_seconds=timing.get('frame_wall_seconds'),
        memory=memory())


def measure(name, call, pricer, blocks, materializer, params, repeats, *, memory, progress, collect=None):
    """For each (block, recipe, requests): 1 cold + `repeats` warm calls; returns per-block summaries.
    collect gets each block's cold-call |t1_value| at expanded cells (the depth-2 value scale)."""
    from gpu_engine.d3_depth2_gate import stage
    summaries = [];baseline = memory() or {}
    for block, recipe, requests, context in blocks:
        calls = []
        for i in range(1 + repeats):
            with stage(progress, f'{name}_call', block=block, call=i, warm=i > 0, roots=len(requests)) as info, context():
                record = call(pricer, requests, materializer, recipe, params, memory);abs_t1 = record.pop('_abs_t1', None)
                if i == 0 and collect is not None and abs_t1 is not None:collect.append(abs_t1)
                calls.append(record);info.update(record)
        summary = summarize_block(calls);peak = summary['process_peak_bytes_so_far']
        summary.update(memory_baseline=baseline, peak_over_baseline_in_use=None if peak is None or baseline.get('bytes_in_use') is None
            else peak - baseline['bytes_in_use'], peak_masked_by_earlier=None if peak is None or baseline.get('process_peak_bytes_so_far') is None
            else peak <= baseline['process_peak_bytes_so_far'])
        summaries.append(dict(block=block, **summary))
    return summaries


def turn2_rows(child_block, k_own, k_opp):
    return child_block * k_own * k_opp


def turn1_rows(capacity):
    """Every batch size a non-turn-2 forward can have at this capacity: an archive group's rows (2C, or a
    power-of-two history group up to 2C) for root priors, times LANES for turn-1 leaves and child priors."""
    groups = {2 * capacity} | {1 << i for i in range((2 * capacity).bit_length())}
    return {g * m for g in groups for m in (1, LANES)}


def constant_turn2_forward(real, rows, census):
    """forward= seam: real outputs, except zeros of the real output shapes (valid True) for batches of `rows`.

    The zeros consume every input through a sum times 0, so the turn-2 leaf encoding stays in the program
    and the difference is the forward's own cost. census gets each traced batch size."""
    import jax
    import jax.numpy as j
    def forward(params, inputs):
        size = int(jax.tree.leaves(inputs)[0].shape[0]);census.append(size)   # every input leads with the batch
        if size != rows:return real(params, inputs)
        shapes = jax.eval_shape(real, params, inputs)
        out = jax.tree.map(lambda s: j.ones(s.shape, s.dtype) if s.dtype == j.bool_ else j.zeros(s.shape, s.dtype), shapes)
        keep = sum(j.sum(x.astype(j.float32)) for x in jax.tree.leaves(inputs)) * 0.
        return out._replace(value=out.value + keep.astype(out.value.dtype))
    return forward


def split_check(rows, capacity, census=None):
    """None when turn-2 forwards are identifiable by size (a priori, and in the traced census if given), else the reason."""
    others = turn1_rows(capacity)
    if rows in others:return f'turn-2 leaf batch ({rows} rows) can equal a turn-1 forward batch at capacity {capacity}'
    if census is not None:
        if rows not in census:return f'no forward of {rows} rows was traced: turn-2 forwards were not seen at the seam'
        unknown = sorted(set(census) - others - {rows})
        if unknown:return f'unexpected traced forward batch sizes {unknown}: the seam layout is not the assumed one'
    return None


def split_estimate(normal, constant, depth1):
    """Differences of warm median device seconds: (i)-(ii) the turn-2 forward, (ii)-(iii) the rest of depth 2."""
    out = dict(normal_seconds=normal, constant_turn2_forward_seconds=constant, depth1_seconds=depth1)
    if constant is not None:out['turn2_forward_seconds_estimate'] = normal - constant
    if constant is not None and depth1 is not None:out['turn2_step_observe_and_child_prior_seconds_estimate'] = constant - depth1
    if depth1 is not None:out['depth2_minus_depth1_seconds'] = normal - depth1
    return out


def memory_stats():
    import jax
    stats = jax.local_devices()[0].memory_stats() or {}
    return dict(process_peak_bytes_so_far=stats.get('peak_bytes_in_use'), bytes_in_use=stats.get('bytes_in_use'))


def make_depth2(executor, model, *, capacity, widths, child_block, actor_capacity, forward=None):
    from gpu_engine.device_depth2_pricer import Depth2Pricer
    parameters = inspect.signature(Depth2Pricer).parameters
    if 'fixed_capacity' not in parameters:raise SystemExit('Depth2Pricer has no fixed_capacity option: pricer d5150eac or later required')
    if forward is not None and 'forward' not in parameters:raise SystemExit('Depth2Pricer has no forward= seam')
    options = dict(capacity=capacity, fixed_capacity=True, actor_capacity=actor_capacity, **widths)
    if child_block is not None:options['child_block'] = child_block
    if forward is not None:options['forward'] = forward
    pricer = Depth2Pricer(executor, model, **options)
    return pricer, dict({k: v for k, v in options.items() if k != 'forward'}, child_block=getattr(pricer, 'child_block', child_block))


def prepare(args):
    """The bank chunk's requests (the gate's builder), the evaluator (head override, champion 9375 or the snapshot actor) and the
    planned blocks. `forward` is the champion evaluator (None: the V2 default)."""
    from gpu_engine import d3_depth2_gate as gate
    with gate.stage(gate.log, 'load_bank_chunk', chunk=args.chunk):
        loaded = gate.bank_chunk_requests(args)
    provider, plan, params, source = loaded.provider, loaded.plan, loaded.params, loaded.source
    head = gate.load_head(args, plan=plan, labels=loaded.labels, category_sha256=loaded.category_sha256) if args.head_archive else None
    champion = gate.load_champion(args, loaded) if getattr(args, 'champion_archive', None) else None
    if head is not None:params = (head[0],)
    if champion is not None:params = (champion.params,)
    groups, root_values, dropped, arm_checked = gate.bind_roots(loaded.groups, params, plan, head_config=None if head is None else head[1], evaluator=champion)
    evaluator = dict(gate.evaluator_header(args, loaded, head, champion=champion), arm_checked=arm_checked, dropped_roots=dropped,
        weights_affect_speed=weights_note(args))
    gate.log('EVALUATOR_RESOLVED', evaluator=evaluator, root_value_scale=gate.percentiles(root_values))
    from pokezero import v5_gumbel as original
    from gpu_engine import device_public_worlds as worlds
    available = [r for g in groups for r in g][:args.max_roots]
    roots = cycle_roots(available, args.capacity);widths = dict(k_own=args.k_own, k_opp=args.k_opp, k1_own=args.k1_own, k1_opp=args.k1_opp)
    seeds = gate.world_block_seeds(provider.recipe.seed, args.worlds_blocks)
    def blocks():
        for block, seed in enumerate(seeds):
            recipe = provider.recipe if block == 0 else original.Recipe(seed)
            context = (lambda: patch.object(original.Recipe, 'selected', lambda self, address: True)) if block else nullcontext
            with context():
                requests = roots if block == 0 else [replace(r, plans=worlds.plan_starts(r.plans.root, provider.source, recipe)) for r in roots]
            yield block, recipe, requests, context
    return SimpleNamespace(provider=provider, plan=plan, params=params, source=source, evaluator=evaluator, root_values=root_values, available=available,
        widths=widths, seeds=seeds, planned=list(blocks()), forward=champion,
        actor_capacity=len(params) if head is not None or champion is not None else plan.historical_window + 1)


def depth1_blocks(args, planned):
    """Depth 1 times every block with --d1; for --profile-split alone only block 0 (the split's baseline), isolated or not."""
    return list(planned) if args.d1 else list(planned)[:1]


def run_context(ctx, devices):
    """What the report needs from a prepared process: the evaluator, root values and counts (an --isolate child's too)."""
    return dict(evaluator=ctx.evaluator, root_values=list(ctx.root_values), roots_available=len(ctx.available), source=ctx.source,
        widths=ctx.widths, block_seeds=list(ctx.seeds), devices=devices)


def measure_config(config, args, ctx, progress):
    """One measured config in this process: 'depth1' (FramePricer), 'depth2' (Depth2Pricer, every block) or 'constant'
    (block 0 with the turn-2 forward replaced by zeros). Every pricer takes ctx.forward, the run's evaluator (the champion's, or
    none: the V2 default), and the constant variant wraps it. Returns its report section."""
    from gpu_engine import v5_model
    from gpu_engine.device_public_teacher_batch import FramePricer
    forward = getattr(ctx, 'forward', None);seam = {} if forward is None else dict(forward=forward)
    if config == 'depth1':
        d1 = FramePricer(ctx.provider.executor, ctx.plan.model, capacity=args.capacity, block_rows=args.capacity, actor_capacity=ctx.actor_capacity,
            vectorized=True, host_prep=True, **seam)
        planned = depth1_blocks(args, ctx.planned)
        blocks = measure('depth1', depth1_call, d1, planned, ctx.provider.materializer, ctx.params, args.repeats, memory=memory_stats, progress=progress)
        return dict(blocks=blocks, derived=derive(blocks, args.capacity), leaves='enumerated = D required turn-1 pairs; evaluated = FramePricer evaluated_joint_slots')
    pricer, options = make_depth2(ctx.provider.executor, ctx.plan.model, capacity=args.capacity, widths=ctx.widths, child_block=args.child_block,
        actor_capacity=ctx.actor_capacity, forward=forward)
    if config == 'depth2':
        t1 = []
        blocks = measure('depth2', depth2_call, pricer, ctx.planned, ctx.provider.materializer, ctx.params, args.repeats, memory=memory_stats,
            progress=progress, collect=t1)
        return dict(blocks=blocks, derived=derive(blocks, args.capacity), options=options, abs_t1=np.concatenate(t1).tolist() if t1 else [])
    rows = turn2_rows(options['child_block'], args.k_own, args.k_opp);reason = split_check(rows, args.capacity);census = []
    if reason is not None:return dict(skipped=reason, rows=rows, census=[])
    real = forward if forward is not None else lambda p, inputs: v5_model.forward(p, inputs, ctx.plan.model, compute_dtype='bfloat16')
    variant, _ = make_depth2(ctx.provider.executor, ctx.plan.model, capacity=args.capacity, widths=ctx.widths, child_block=args.child_block,
        actor_capacity=ctx.actor_capacity, forward=constant_turn2_forward(real, rows, census))
    blocks = measure('split_constant_turn2_forward', depth2_call, variant, ctx.planned[:1], ctx.provider.materializer, ctx.params, args.repeats,
        memory=memory_stats, progress=progress)
    return dict(blocks=blocks, rows=rows, census=sorted(set(census)), skipped=split_check(rows, args.capacity, census))


def isolated(args, argv, out):
    """--isolate: each config in its own subprocess (`--only CONFIG --out OUT/CONFIG`); returns {config: its section}."""
    import subprocess, sys
    argv = list(sys.argv[1:] if argv is None else argv);clean = []
    skip = False
    for i, token in enumerate(argv):
        if skip:skip = False;continue
        if token == '--isolate':continue
        if token == '--out':skip = True;continue
        if token.startswith('--out='):continue
        clean.append(token)
    configs = (['depth1'] if args.d1 or args.profile_split else []) + ['depth2'] + (['constant'] if args.profile_split else [])
    sections, contexts = {}, {}
    for config in configs:
        target = out / config
        code = subprocess.call([sys.executable, '-m', 'gpu_engine.d3_depth2_throughput', *clean, '--only', config, '--out', str(target)])
        if code != 0:raise SystemExit(f'--isolate: the {config} subprocess exited {code}')
        child = json.loads((target / 'throughput.json').read_text());sections[config] = child['sections'][config]
        contexts[config] = child.get('context')
    return sections, configs, contexts


def main(argv=None):
    args = parse(argv)
    import importlib.util   # fail before loading the bank; find_spec does not import JAX
    if importlib.util.find_spec('gpu_engine.device_depth2_pricer') is None:raise SystemExit('gpu_engine.device_depth2_pricer is not on this tree')
    from gpu_engine import d3_depth2_gate as gate, champion_evaluator as ce
    gate.log('EVALUATOR', override=bool(args.head_archive or args.champion_archive), archive=args.head_archive, archive_sha256=args.head_archive_sha256,
        checkpoint_sha256=args.head_checkpoint_sha256, value_reference=args.value_reference,
        weights_affect_speed=weights_note(args), **ce.log_fields(args),
        resolved='EVALUATOR_RESOLVED follows once the bank is loaded')
    out = Path(args.out);out.mkdir(parents=True, exist_ok=False)
    if args.isolate:
        # The parent never loads the bank or touches the device: every child starts with the whole device to itself.
        gate.log('THROUGHPUT_START', roots_per_call=args.capacity, repeats=args.repeats, isolate=True)
        sections, order, contexts = isolated(args, argv, out);context = contexts['depth2']
    else:
        ctx = prepare(args)
        import jax
        context = run_context(ctx, [d.device_kind for d in jax.devices()])
        if args.only:   # one config (an --isolate subprocess)
            section = measure_config(args.only, args, ctx, gate.log)
            (out / 'throughput.json').write_text(json.dumps(dict(evaluator=ctx.evaluator, only=args.only, sections={args.only: section}, context=context),
                indent=1, allow_nan=False, default=str) + '\n')
            return 0
        gate.log('THROUGHPUT_START', roots_available=context['roots_available'], roots_per_call=args.capacity,
            cycled=context['roots_available'] < args.capacity, worlds=2 * len(ctx.seeds), repeats=args.repeats, devices=context['devices'], isolate=False)
        # Depth 1 first: in one process the device peak only grows, so depth 1's number is its own.
        order = (['depth1'] if args.d1 or args.profile_split else []) + ['depth2'] + (['constant'] if args.profile_split else [])
        sections = {config: measure_config(config, args, ctx, gate.log) for config in order}
    devices = context['devices'];available = context['roots_available']
    depth2 = sections['depth2'];abs_t1 = depth2.pop('abs_t1', [])
    report = dict(evaluator=context['evaluator'], value_scale=gate.value_scale(context['root_values'], gate.percentiles(abs_t1), args.value_reference),
        weights_affect_speed=weights_note(args), schema=SCHEMA, source=context['source'], capacity=args.capacity, widths=context['widths'],
        child_block=depth2['options']['child_block'], worlds_blocks=args.worlds_blocks, block_seeds=context['block_seeds'], repeats=args.repeats,
        roots_available=available, roots_per_call=args.capacity, cycled=available < args.capacity, pricer_options=depth2['options'], devices=devices,
        measured_order=order, isolated=args.isolate, memory_scope=MEMORY_SCOPE + (' (isolated: one process per config)' if args.isolate else ''),
        depth2=dict(blocks=depth2['blocks'], derived=depth2['derived']),
        scope='device_seconds = the pricer timing (one jitted call to block_until_ready plus its device_get); compile = lower + compile of the cold call')
    if args.d1:report['depth1'] = sections['depth1']
    if args.isolate:report['isolated_contexts'] = {k: {f: v.get(f) for f in ('roots_available', 'devices')} for k, v in contexts.items() if v}
    if args.profile_split:
        constant = sections['constant'];normal = depth2['blocks'][0]['warm_device_seconds_median']
        constant_seconds = None if constant.get('skipped') else constant['blocks'][0]['warm_device_seconds_median']
        report['profile_split'] = dict(split_estimate(normal, constant_seconds, sections['depth1']['blocks'][0]['warm_device_seconds_median']),
            turn2_forward_rows=constant['rows'], traced_forward_rows=constant['census'], skipped_constant_variant=constant.get('skipped'),
            method='turn-2 leaf forwards recognized by static batch size at trace time; zeros of the real output shapes consume every input',
            caveat='estimates from warm medians of separate programs; XLA may schedule the variants differently')
    (out / 'throughput.json').write_text(json.dumps(report, indent=1, allow_nan=False, default=str) + '\n')   # evaluator first
    gate.log('THROUGHPUT_COMPLETE', out=str(out / 'throughput.json'), depth2=report['depth2']['derived'],
        depth1=report.get('depth1', {}).get('derived'), split=report.get('profile_split'))
    return 0


if __name__ == '__main__':raise SystemExit(main())
