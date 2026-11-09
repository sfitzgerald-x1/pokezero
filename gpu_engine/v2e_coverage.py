"""Coverage report over the capture shards: per phase group, how many lane-steps take each path (design note section 2.2).

For every shard the oracle step is replayed once under the attribute-wrapping taps of gpu_engine.v2e_hook_taps, over
the public-tracker hooks and the mechanics callees alike, and every tapped call's per-lane mask (its `use`, `eligible`,
`acts`, `ordinary`, `active`, ... argument) is returned from the same jitted step. Those masks are exact: they are the
oracle's own values. Counted per phase group (v2e_census.PHASES; a call site inside a hook or mechanics module is
attributed to the phase of the last step_with_chance-line call before it):
- `hook:<name>@<site>`: every device_public hook call with its mask true (switch_decision, install, baton_copy, drag,
  the Pursuit `move`/`port_count` calls, ...);
- `mech:<module>.<function>@<site>`: every mechanics callee with its eligibility mask true (pursuit.intercept,
  phazing.drag, baton.copy_to_recipient, the effect functions, the residual handlers, ...);
- `effect:<table column>@<site>`: each move-indexed table flag of the move the `move` hook declared (exact, after Encore,
  Sleep Talk and Struggle), per declaring site (the caller move, the Sleep Talk child, Pursuit, Focus Punch);
- `step:*`: deferral (the output's pending_residual), resume, request kinds and the approximate input-derived path
  classes of v2e_phase_profile.lane_features (`input:*`).
coverage.json and coverage.md list every path's count and lane share, flag the paths with fewer than --rare (1,000)
occurrences (the synthetic-lane list for M2..M4) and give the warp union factor per phase group over the exact masks
(v2e_phase_profile.UnionStats: mean warp-union path count over mean per-lane path count, and the design's cost-model
factor). Paths are arguments.
"""
import argparse
import collections
import inspect
import json
import os
from pathlib import Path
import re
import time

import numpy as np

from gpu_engine import v2e_census as cz, v2e_hook_taps as taps

SCHEMA = 'pokezero.v2e-coverage.v1'
MECHANICS = ('mechanics', 'action_history', 'items', 'weather', 'absorption', 'entry', 'combat', 'pressure', 'sustain', 'damage_moves',
    'variable_moves', 'pursuit', 'baton', 'phazing', 'linked_traps', 'critical', 'secondary', 'sleep_talk', 'self_destruct', 'encore',
    'solar_beam', 'focus_punch', 'effects')
MODULES = dict(taps.TAP_MODULES, **{m: f'gpu_engine.{m}' for m in MECHANICS})
SCAN = tuple(dict.fromkeys(taps.SCAN_MODULES + tuple(MODULES.values())))
MASK_PARAMS = ('use', 'eligible', 'acts', 'ordinary', 'active', 'rows_active', 'applied', 'changes', 'proc')
# `move` hook sites whose declared move's table flags are counted (transition lines; see v2e_hook_taps sites).
MOVE_SITES = {'transition:807': 'caller', 'transition:838': 'child', 'transition:561': 'pursuit', 'transition:534': 'focus_punch'}
RARE = 1000


def mechanics_names():
    """Every function of the mechanics modules as '<module>.<function>' (tapped generically; only calls with a mask count)."""
    import importlib
    out = []
    for m in MECHANICS:
        module = importlib.import_module(f'gpu_engine.{m}')
        for attr, value in sorted(vars(module).items()):
            if inspect.isfunction(value) and value.__module__ == module.__name__:out.append(f'{m}.{attr}')
    return out


def hook_names():
    from gpu_engine import v2e_hooks
    return list(v2e_hooks.ALL_HOOKS)


def bound_arguments(call, modules):
    """{parameter: value} of a recorded call bound to the oracle signature (the carry/Context parameter included)."""
    fn = taps.original(call.name, modules);sig = inspect.signature(fn)
    try:bound = sig.bind(None, *call.args, **call.kwargs)
    except TypeError:return {}
    bound.apply_defaults()
    return dict(bound.arguments)


def lane_mask(value, b):
    """A per-lane bool [B] from a hook mask argument (array, Python bool, or a broadcastable array), or None."""
    import jax.numpy as j
    if isinstance(value, (bool, np.bool_)):return j.full((b,), bool(value))
    if not hasattr(value, 'shape') or not hasattr(value, 'dtype'):return None
    if value.dtype != j.bool_:return None
    if value.ndim == 0:return j.broadcast_to(value, (b,))
    if value.shape[0] != b:return None
    return value if value.ndim == 1 else j.any(value.reshape(b, -1), axis=1)


def group_of_line(line):
    if line is None:return None
    name = cz.phase_of(line)
    return cz.group_of(name) if name and not name.startswith('other') else None


class Features:
    """Builds the per-group feature dict inside one traced step; records the feature names per lane count."""
    def __init__(self, ctx, columns):
        from gpu_engine import v2e_modules
        self.step = v2e_modules.step(ctx['engine_tables'], ctx['observation_tables']);self.table = ctx['engine_tables'];self.columns = columns
        self.names = {};self.sites = {}

    def __call__(self, state, actions, trackers):
        import jax
        import jax.numpy as j
        from gpu_engine import transition
        from gpu_engine.state import usable_moves
        b = state.hp.shape[0];names = hook_names() + mechanics_names()
        with taps.tap(names, modules=MODULES, scan=SCAN) as calls:out = self.step(state, actions, trackers)
        feats = {g: {} for g in cz.GROUPS};group = 'A';sites = collections.Counter()
        for call in calls:
            kind = 'hook' if call.name.split('.')[0] in taps.TAP_MODULES or '.' not in call.name else 'mech'
            if call.site.startswith('transition:'):
                g = group_of_line(int(call.site.split(':')[1]))
                if g is not None:group = g
            args = bound_arguments(call, MODULES);mask = None
            for param in MASK_PARAMS:
                if param in args:
                    mask = lane_mask(args[param], b)
                    if mask is not None:break
            if mask is None:continue
            feats[group][f'{kind}:{call.key}'] = mask;sites[(kind, call.name)] += 1
            if call.name == 'move' and call.site in MOVE_SITES and 'move' in args:
                move = j.clip(j.asarray(args['move'], j.int32), 0, None)
                for path, column in self.columns.items():
                    flag = j.asarray(column)[j.clip(move, 0, column.shape[0] - 1)]
                    flag = flag.reshape(b, -1).any(axis=1) if flag.ndim > 1 else flag != 0
                    feats[group][f'effect:{path}@{MOVE_SITES[call.site]}'] = mask & flag
        after = out.step.state
        live = ~state.done & ~out.step.invalid_actions
        feats['A']['step:resume'] = live & state.pending_residual;feats['A']['step:invalid'] = out.step.invalid_actions
        feats['F']['step:defer'] = after.pending_residual;feats['G']['step:done_after'] = after.done & ~state.done
        feats['G']['step:winner_decided'] = (after.winner >= 0) & ~state.done
        mask = transition.action_mask(state, self.table);usable = usable_moves(state)
        stacked = {g: (j.stack([feats[g][n] for n in sorted(feats[g])], axis=1) if feats[g] else j.zeros((b, 0), bool)) for g in cz.GROUPS}
        names_by_group = {g: sorted(feats[g]) for g in cz.GROUPS}
        if b in self.names and self.names[b] != names_by_group:raise RuntimeError('tapped feature names differ between traces of the same lane count')
        self.names[b] = names_by_group;self.sites[b] = {f'{k}:{n}': c for (k, n), c in sites.items()}
        return stacked, mask, usable


def path_of(name):
    """The path a per-occurrence feature belongs to: the site without its `#<occurrence>` suffix (a hook site reached in
    both action orders or in every residual event is one path)."""
    return re.sub(r'#\d+$', '', name)


def summarize(counts, lanes, union, rare):
    """Per group: {feature: count, share} per tapped occurrence, the per-path aggregation over occurrences, the rare
    paths (aggregated count under `rare`) and the union factors."""
    out = {}
    for g in cz.GROUPS:
        features = {name: dict(count=int(c), lane_share=int(c) / lanes if lanes else None) for name, c in sorted(counts[g].items(), key=lambda kv: (-kv[1], kv[0]))}
        paths = collections.Counter()
        for name, c in counts[g].items():paths[path_of(name)] += int(c)
        paths = {name: dict(count=c, lane_share=c / lanes if lanes else None) for name, c in sorted(paths.items(), key=lambda kv: (-kv[1], kv[0]))}
        out[g] = dict(features=features, paths=paths, rare=[name for name, r in paths.items() if r['count'] < rare], union=union.get(g))
    return out


def markdown(report):
    out = [f"# Coverage over {report['shards']} shards, {report['lane_steps']:,} lane-steps", '',
        '| Group | Tapped occurrences | Paths (occurrences merged) | Paths under {rare} | Union factor (exact masks) | Union factor (design cost model) | Distinct classes per warp (mean / p90) |'.format(rare=report['rare']),
        '| --- | ---: | ---: | ---: | ---: | ---: | --- |']
    for g, row in report['groups'].items():
        u = row['union'] or {}
        out.append(f"| {g} | {len(row['features'])} | {len(row['paths'])} | {len(row['rare'])} | {u.get('raw_union_factor', float('nan')):.3f} | "
            f"{(u.get('union_factor_design_uniform') or float('nan')):.3f} | {u.get('distinct_classes_per_warp', {}).get('mean', float('nan')):.1f} / {u.get('distinct_classes_per_warp', {}).get('p90', float('nan'))} |")
    out += ['', f"Weighted union factor (exact masks, census weights when given): {report.get('weighted_raw_union_factor')}", '',
        '## Paths with fewer than {rare} occurrences (synthetic-lane candidates)'.format(rare=report['rare']), '', '| Group | Path | Count |', '| --- | --- | ---: |']
    for g, row in report['groups'].items():
        for name in row['rare']:out.append(f"| {g} | `{name}` | {row['paths'][name]['count']} |")
    out += ['', '## Key paths', '', '| Group | Path | Count | Lane share |', '| --- | --- | ---: | ---: |']
    for g, row in report['groups'].items():
        for name, r in row['paths'].items():
            if name.startswith(('hook:switch_decision', 'hook:install', 'hook:baton_copy', 'hook:drag', 'hook:port_count', 'mech:pursuit', 'mech:phazing', 'mech:baton', 'step:')):
                out.append(f"| {g} | `{name}` | {r['count']:,} | {100 * (r['lane_share'] or 0):.2f}% |")
    out += ['', '## All paths (occurrences merged; per-occurrence counts in coverage.json `features`)', '']
    for g, row in report['groups'].items():
        out += [f'### {g}', '', '| Path | Count | Lane share |', '| --- | ---: | ---: |']
        out += [f"| `{name}` | {r['count']:,} | {100 * (r['lane_share'] or 0):.2f}% |" for name, r in row['paths'].items()]
        out.append('')
    return '\n'.join(out) + '\n'


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    p.add_argument('--config', required=True);p.add_argument('--reference-run', required=True);p.add_argument('--arm', default='B', choices=('A', 'B'))
    p.add_argument('--shards', required=True);p.add_argument('--out', required=True);p.add_argument('--limit', type=int, default=0)
    p.add_argument('--census', help='census.json: hook shares per group for the design cost-model union factor')
    p.add_argument('--warp', type=int, default=32);p.add_argument('--rare', type=int, default=RARE)
    args = p.parse_args(argv)
    out = Path(args.out);out.mkdir(parents=True, exist_ok=True)
    import jax
    from gpu_engine import v2e_phase_profile as pp
    from gpu_engine.v2_bench_collector import log
    ctx = cz.load_context(args);table = ctx['engine_tables'];columns = pp.table_columns(table);host_table = jax.device_get(table)
    features = Features(ctx, {k: np.asarray(v) for k, v in columns['move'].items()})
    jitted = {}
    paths = cz.shard_paths(args.shards, ctx['template'], args.limit)
    counts = {g: collections.Counter() for g in cz.GROUPS};lanes = 0;union = pp.UnionStats(args.warp);begin = time.perf_counter()
    for i, path in enumerate(paths):
        state, actions, trackers = cz.shard_tree(path, ctx['template']);b = state.hp.shape[0]
        if b not in jitted:
            t0 = time.perf_counter();jitted[b] = jax.jit(features);stacked, mask, usable = jitted[b](*jax.tree.map(jax.numpy.asarray, (state, actions, trackers)))
            jax.block_until_ready(stacked);log('COMPILED', lanes=b, seconds=time.perf_counter() - t0, features={g: len(n) for g, n in features.names[b].items()})
        else:stacked, mask, usable = jitted[b](*jax.tree.map(jax.numpy.asarray, (state, actions, trackers)))
        exact = {g: np.asarray(stacked[g]) for g in cz.GROUPS};names = features.names[b]
        approx = pp.lane_features(state, actions, host_table, columns, np.asarray(mask), np.asarray(usable))
        per_group = {}
        for g in cz.GROUPS:
            d = {n: exact[g][:, k] for k, n in enumerate(names[g])}
            for n, v in approx.get(g, {}).items():d[f'input:{n}'] = np.asarray(v, bool)
            per_group[g] = d
            for n, v in d.items():counts[g][n] += int(v.sum())
        union.add(per_group);lanes += b
        if (i + 1) % 20 == 0 or i + 1 == len(paths):log('COVERAGE_PROGRESS', shards=i + 1, lane_steps=lanes, seconds=time.perf_counter() - begin)
    uniform, equations = pp.census_weights(args.census);summary = union.summary(uniform)
    for g, s in summary.items():
        s['raw_union_factor'] = s['union_features_mean'] / s['lane_features_mean'] if s['lane_features_mean'] else None
    groups = summarize(counts, lanes, summary, args.rare)
    report = dict(schema=SCHEMA, source_commit=os.environ.get('V2_SOURCE_COMMIT'), shards=len(paths), lane_steps=lanes, warp=args.warp, rare=args.rare,
        census=args.census, seconds=time.perf_counter() - begin, calls_per_step={str(b): s for b, s in features.sites.items()},
        groups=groups, weighted_raw_union_factor=pp.weighted_factor(summary, equations, 'raw_union_factor'),
        weighted_union_factor_design_uniform=pp.weighted_factor(summary, equations, 'union_factor_design_uniform'),
        rare_total=sum(len(g['rare']) for g in groups.values()),
        method='exact per-lane masks of every tapped hook and mechanics call (the oracle\'s own values) plus the move-table flags of each declared move; '
            'input:* are the approximate input-derived classes of v2e_phase_profile.lane_features; union factors as v2e_phase_profile.UnionStats')
    (out / 'coverage.json').write_text(json.dumps(report, indent=1, default=str) + '\n')
    (out / 'coverage.md').write_text(markdown(report))
    log('COVERAGE', shards=len(paths), lane_steps=lanes, rare=report['rare_total'], union={g: (s.get('raw_union_factor')) for g, s in summary.items()}, out=str(out / 'coverage.json'))
    return 0


if __name__ == '__main__':raise SystemExit(main())
