"""Equation census of the collector's engine step by phase (option-A design note, section 1.3 and section 5 step 1).

Traces v2e_modules.step(engine_tables, observation_tables), the closure the collector runs, exactly as the generator
does (v2e_gen.trace under v2e_modules.trace_substitutions: lane_index.LIMIT = 0, LANES lanes, DCE), on the real tables
of a declared run, loaded as gpu_engine.v2e_diff loads them (--config, --reference-run). The example is the capture
template, or with --shards the first lane of the first capture shard (its leaf paths checked against the template).
Each top-level equation is attributed, with v2e_j2c.mark_site (the rule the generator's profile marks use), to the
outermost transition.step_with_chance frame (the phase line) and the innermost gpu_engine module; the leaves of a call
equation inherit its attribution.

Reports, into census.json and census.md in a new --out directory:
- leaf equations by phase (with the action iteration of E1..E4 and the residual event of F2) x innermost module, and by
  phase x category (tracker modules / mechanics / inline transition), plus the select-only cone share (an equation whose
  every use is a select_n case operand or another select-only equation);
- the live carry at every phase change (lane vars and per-lane elements of the values defined before and used at or
  after it), with the design's boundaries b0..b8 tagged;
- the generator's post-CSE/fold equation count per phase: v2e_j2c.generate(profile=True, mode='thread') emits a mark
  for every equation that emitted code (calls excluded); --no-generate skips it;
- device lines per phase from the pruned step_with_chance listing (--listing, lines 'NNN: source').

PHASES is the one phase line table (design section 1.1) every phase tool uses (v2e_phase_profile's subcommands too);
tests check it against transition.step_with_chance's source, so line drift fails loudly. Paths are arguments.
"""
import argparse
import collections
import json
import os
from pathlib import Path
import re
import time

import numpy as np

SCHEMA = 'pokezero.v2e-census.v1'
# transition.step_with_chance: def line and last line (design note section 1.1; tree 5bc2fe8c, shifted by fidelity A1, A2, R, B, M1, M2, R1 and WH).
SPAN = (400, 2706)
# (phase, first line, last line, group). 'pre' is the host-side argument checks; boundary statements open the next phase.
PHASES = (
    ('pre', 400, 544, 'pre'),
    ('A', 545, 599, 'A'),      # request kind, validity, resuming draws, key split, age, initialize, Lum
    ('B', 600, 634, 'B'),      # speeds, switches, saved actor, slot, Struggle, charge, recharge, Focus Punch preparation
    ('C', 635, 672, 'C'),      # Pursuit per side
    ('D', 673, 792, 'D'),      # switch order, Baton, depart/install/enter, forced replacement entries, post-switch Lum
    ('E0', 793, 815, 'E'),     # ordering
    ('E1', 816, 1034, 'E'),     # per action: pre-move (the recharge turn included)
    ('E2', 1035, 1410, 'E'),    # damage
    ('E3', 1411, 2038, 'E'),   # effects (Perish Song, Yawn, the Substitute's trap release and Wrap's trap included)
    ('E4', 2039, 2166, 'E'),   # post (recoil, move weather, Spikes, drag, Baton, fainted boosts)
    ('F1', 2167, 2278, 'F'),   # defer/ready, end speeds, Wish, weather
    ('F2', 2279, 2621, 'F'),   # residual events (argsort at 2286, the 14-event loop at 2287), the Perish pass, White Herb's order 29 and the recharge tick
    ('G', 2622, 2706, 'G'),    # end of turn
)
GROUPS = ('A', 'B', 'C', 'D', 'E', 'F', 'G')
GROUP = {name: group for name, _, _, group in PHASES}
# Source text each line must start with (stripped): a line shift anywhere above an anchor fails the drift test.
ANCHORS = {
    400: 'def step_with_chance(state: BattleState, actions: jax.Array, table: MoveTable,',
    545: 'rows = jnp.arange(state.hp.shape[0])',
    600: '# All action speeds are frozen before ordinary switches.',
    602: 'speed = _speeds(work,table)',
    635: 'pursuit_used = jnp.zeros_like(switches)',
    673: 'bench = bench_indices(state.active)',
    792: 'else:work = _lum_update(work,table,live,_device_public=_device_public)',
    793: 'priority = table.priority[moves]',
    816: 'for order in range(2):',
    817: '_faint_before_hp=work.hp',
    1035: 'attack_stats = _active(work.stats, work, side)',
    1411: 'if table.effects is not None:',
    2039: '# Gen3 applies contact DamagingHit',
    2131: 'work,changed,hazard_ko=dragging.drag(work,foe,move,phazing_draws[rows,side],',
    2167: 'turn_context = live & (~replacement | state.pending_residual)',
    2279: 'event_sides = jnp.array(residual_order.EVENT_SIDES)',
    2286: 'event_order = jnp.argsort(event_keys,axis=1)',
    2287: 'for order in range(residual_order.EVENTS):',
    2302: 'if _status_residual_context is not None:',
    2622: 'work=linking.prune(work)',
    2697: 'work = jax.tree.map(lambda old, new: jnp.where(live.reshape(',
    2706: 'return result if _public_trace is None else public_callbacks.TracedStep(result,_public_trace)',
}
# Iteration rules over lines in program order (the scratch census's): an E1..E4 line <= E_RESTART after one >= E_LATE
# starts the second action; in F2 a line <= F2_HEAD after one > F2_BODY starts the next of the 14 residual events.
E_LATE, E_RESTART = 2057, 902
F2_LOOP, F2_HEAD, F2_BODY = 2287, 2302, 2317
# Design boundaries: the phase label whose first equation each one is (b0: the first step_with_chance equation; b8: the end).
BOUNDARIES = (('b0', 'A'), ('b1', 'B'), ('b2', 'D'), ('b3', 'E0'), ('b4', 'E1.0'), ('b5', 'E1.1'), ('b6', 'F1'), ('b7', 'G'), ('b8', 'END'))
TRACKERS = {'engine_device_public_hooks', 'engine_device_trackers', 'engine_device_public_port', 'engine_device_visible_belief',
    'engine_device_negative_items', 'engine_device_damage_history', 'lane_index', 'public_active', 'v2e_modules'}
CALLS = {'jit', 'pjit', 'closed_call', 'custom_jvp_call', 'custom_vjp_call'}


def phase_of(line):
    """PHASES name of a step_with_chance line; None for no line (outside step_with_chance)."""
    if line is None:return None
    for name, first, last, _ in PHASES:
        if first <= line <= last:return name
    return f'other{line}'


def label_lines(lines):
    """Per-item labels for step_with_chance lines in program order: the phase name, 'E1.0'..'E4.1' with the action
    iteration, 'F2.pre' before the residual loop and 'F2.0'..'F2.13' per event, 'outside' for None."""
    labels = [];action = 0;last_e = None;event = -1;last_f = None
    for line in lines:
        name = phase_of(line)
        if name is None:labels.append('outside');continue
        if name in ('E1', 'E2', 'E3', 'E4'):
            if last_e is not None and last_e >= E_LATE and line <= E_RESTART:action += 1
            last_e = line;name = f'{name}.{action}'
        elif name == 'F2':
            if line < F2_LOOP:name = 'F2.pre'
            else:
                if event < 0 or (last_f is not None and last_f > F2_BODY and line <= F2_HEAD):event += 1
                last_f = line;name = f'F2.{event}'
        labels.append(name)
    return labels


def base(label):
    """'E3.1' -> 'E3', 'F2.4' -> 'F2'."""
    return label.split('.', 1)[0]


def group_of(label):
    """The design group (A..G) of a label; 'outside', 'pre' and 'otherN' stay themselves."""
    return GROUP.get(base(label), base(label))


def category(module):
    if module in TRACKERS or (module or '').startswith('public_belief'):return 'tracker'
    return 'inline' if module == 'transition' else 'mechanics'


def listing_lines(path):
    """Device lines per PHASES name from a pruned listing ('  NNN: source' per surviving line)."""
    counts = collections.Counter()
    for text in Path(path).read_text().splitlines():
        m = re.match(r'\s*(\d+):', text)
        if m:counts[phase_of(int(m.group(1))) or 'outside'] += 1
    return dict(counts)


# ---------------------------------------------------------------------------
# Loading: the tables and capture template exactly as gpu_engine.v2e_diff loads them.

def load_context(args, *, games=None):
    """dict(engine_tables, observation_tables, template, context) for --config/--reference-run/--arm (v2e_diff's loading;
    the template's structure and per-lane shapes do not depend on how many games are constructed)."""
    import jax
    import jax.numpy as j
    from gpu_engine.v2_bench_collector import journal, prepare, actor, construct
    events = journal(args.reference_run);context = prepare(args)
    snapshot, params, _ = actor(args, context['plan'], events)
    _, roster, _ = construct(context, snapshot, events, games=games if games is not None else getattr(args, 'games', 1))
    first = lambda tree: jax.tree.map(lambda x: x[:1], tree)
    template = (first(roster.device.state), j.zeros((1, 2), j.int32), first(roster.device.public))
    prepared = context['prepared']
    return dict(engine_tables=prepared.engine_tables, observation_tables=prepared.observation_tables, template=template, context=context)


def shard_paths(directory, template, limit=0):
    """Capture shards under `directory` in order, after v2e_diff's check of capture.json's leaf paths against the template."""
    from gpu_engine.v2e_capture import leaf_paths
    capture = json.loads(Path(directory).parent.joinpath('capture.json').read_text())
    if capture['paths'] != leaf_paths(template):raise ValueError('capture leaf paths differ from the oracle pytrees')
    shards = sorted(Path(directory).glob('shard-*.npz'))
    if not shards:raise SystemExit(f'no capture shards under {directory}')
    return shards[:limit] if limit else shards


def shard_tree(path, template, lanes=None):
    """One shard as the (state, actions, trackers) pytree of numpy leaves, optionally its first `lanes` lanes."""
    import jax
    from gpu_engine.v2e_diff import load_shard
    leaves, _, _ = load_shard(path)
    if lanes:leaves = [x[:lanes] for x in leaves]
    return jax.tree.unflatten(jax.tree.structure(template), leaves)


# ---------------------------------------------------------------------------
# Census of a closed jaxpr.

def _leaves(jaxpr):
    from jax.extend import core as jcore
    return sum((sum(_leaves(s) for s in subs) if (subs := list(jcore.jaxprs_in_params(e.params))) else 1) for e in jaxpr.eqns)


def census(closed, lanes):
    """The census of a traced step (dict; see the module docstring). `lanes` is the trace's lane size."""
    from jax.extend import core as jcore
    from gpu_engine import v2e_j2c
    jx = closed.jaxpr;ids = {}
    vid = lambda v: ids.setdefault(v, len(ids))
    prim, names, lines, modules, nleaf, ins, outs = [], [], [], [], [], [], []
    for e in jx.eqns:
        line, module = v2e_j2c.mark_site(e);subs = list(jcore.jaxprs_in_params(e.params))
        prim.append(e.primitive.name);names.append(e.params.get('name') if subs else None)
        lines.append(line);modules.append(module or 'none');nleaf.append(sum(_leaves(s) for s in subs) if subs else 1)
        ins.append([(vid(v), i) for i, v in enumerate(e.invars) if not isinstance(v, jcore.Literal)]);outs.append([vid(v) for v in e.outvars])
    for v in list(jx.invars) + list(jx.constvars):vid(v)
    outset = {vid(v) for v in jx.outvars if not isinstance(v, jcore.Literal)}
    n = len(prim);nleaf = np.asarray(nleaf, np.int64);labels = label_lines(lines)
    # Select-only cones: every use is a select_n case operand (position >= 1) or another select-only equation.
    uses = collections.defaultdict(list)
    for i, r in enumerate(ins):
        for v, pos in r:uses[v].append((i, pos))
    select = np.zeros(n, bool)
    for i in range(n - 1, -1, -1):
        ok = True;used = False
        for v in outs[i]:
            if v in outset:ok = False;break
            for k, pos in uses.get(v, ()):
                used = True
                if not ((pos >= 1 and (prim[k] == 'select_n' or names[k] == '_where')) or select[k]):ok = False;break
            if not ok:break
        select[i] = ok and used
    by_label = collections.OrderedDict();by_module = collections.Counter()
    for label, module, count, sel in zip(labels, modules, nleaf, select):
        row = by_label.setdefault(label, dict(n=0, tracker=0, mechanics=0, inline=0, select_only=0, modules=collections.Counter()))
        row['n'] += int(count);row[category(module)] += int(count);row['select_only'] += int(count) * bool(sel);row['modules'][module] += int(count)
        by_module[module] += int(count)
    total = int(nleaf.sum())
    def merged(key):
        out = collections.OrderedDict()
        for label, row in by_label.items():
            o = out.setdefault(key(label), dict(n=0, tracker=0, mechanics=0, inline=0, select_only=0, modules=collections.Counter()))
            for k in ('n', 'tracker', 'mechanics', 'inline', 'select_only'):o[k] += row[k]
            o['modules'].update(row['modules'])
        for o in out.values():
            o.update(share=o['n'] / total if total else None, hook_share=o['tracker'] / o['n'] if o['n'] else None,
                select_only_share=o['select_only'] / o['n'] if o['n'] else None, modules=dict(o['modules'].most_common()))
        return out
    # Live carry at every label change (values defined before it and used at or after it).
    nv = len(ids);per = np.ones(nv, np.int64);lane = np.zeros(nv, bool)
    for v, i in ids.items():
        shape = tuple(getattr(v.aval, 'shape', ()));size = int(np.prod(shape)) if shape else 1
        if shape and shape[0] == lanes:per[i] = size // lanes;lane[i] = True
        else:per[i] = size
    defined = np.full(nv, -1);last = np.full(nv, -2)
    for i in range(n):
        for v in outs[i]:defined[v] = i
        for v, _ in ins[i]:last[v] = i
    for v in outset:last[v] = n
    changes = [(i, l) for i, l in enumerate(labels) if l != (labels[i - 1] if i else None)] + [(n, 'END')]
    tags = {};seen = set()
    for i, label in changes:
        for tag, start in BOUNDARIES:
            if label == start and tag not in seen:tags[i] = tag;seen.add(tag)
    carry = []
    for i, label in changes:
        live = (defined < i) & (last >= i);a = live & lane;b = live & ~lane;small = a & (per > 1) & (per <= 32);large = a & (per > 32)
        carry.append(dict(label=label, index=i, boundary=tags.get(i), lane_vars=int(a.sum()), lane_elements=int(per[a].sum()),
            scalars=int((a & (per == 1)).sum()), small=[int(small.sum()), int(per[small].sum())], large=[int(large.sum()), int(per[large].sum())],
            non_lane=[int(b.sum()), int(per[b].sum())]))
    for row in by_label.values():row['modules'] = dict(row['modules'].most_common())
    from gpu_engine import v2e_gen
    return dict(lanes=lanes, top_level=n, walk=sum(1 for _ in v2e_gen._walk(jx)), leaf=total, select_only_share=float(nleaf[select].sum() / total) if total else None,
        labels=by_label, phases=merged(base), groups=merged(group_of), modules=dict(by_module.most_common()), carry=carry)


def generated_counts(marks):
    """Post-CSE/fold equations per label, phase and group from profile marks (primitive, where, shapes, line, module): one
    mark per equation that emitted code; call equations (whose inner equations carry their own marks) are excluded."""
    keep = [m for m in marks if m[0] not in CALLS];labels = label_lines([m[3] if len(m) > 3 else None for m in keep])
    by_label = collections.Counter(labels)
    phases, groups = collections.Counter(), collections.Counter()
    for label, count in by_label.items():phases[base(label)] += count;groups[group_of(label)] += count
    return dict(marks=len(marks), equations=len(keep), labels=dict(by_label), phases=dict(phases), groups=dict(groups))


def markdown(report):
    out = ['| Phase | Equations | Share | Tracker | Mechanics | Inline | Hook share | Select-only | Device lines | Generated (post-CSE) |',
        '| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
    lines = report.get('device_lines') or {};post = (report.get('generator') or {}).get('groups') or {}
    for name, row in report['census']['groups'].items():
        dl = sum(v for k, v in lines.items() if group_of(k) == name or (name == 'A' and group_of(k) == 'pre')) if lines else None   # A: with the host checks, as the design
        out.append(f"| {name} | {row['n']:,} | {100 * row['share']:.1f}% | {row['tracker']:,} | {row['mechanics']:,} | {row['inline']:,} | "
            f"{100 * row['hook_share']:.0f}% | {100 * row['select_only_share']:.0f}% | {dl if dl is not None else '-'} | "
            f"{post.get(name, '-') if post else '-'} |")
    c = report['census']
    out += ['', f"Total leaf equations {c['leaf']:,} (top-level {c['top_level']:,}); select-only {100 * c['select_only_share']:.1f}%.", '',
        '| Boundary | At label | Lane vars | Per-lane elements |', '| --- | --- | ---: | ---: |']
    for row in c['carry']:
        if row['boundary']:out.append(f"| {row['boundary']} | {row['label']} | {row['lane_vars']} | {row['lane_elements']:,} |")
    top = list(c['modules'].items())[:12]
    out += ['', 'By innermost module: ' + ', '.join(f'{m} {n:,}' for m, n in top) + '.']
    return '\n'.join(out) + '\n'


def parser():
    p = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    p.add_argument('--config', required=True);p.add_argument('--reference-run', required=True)
    p.add_argument('--arm', default='B', choices=('A', 'B'))
    p.add_argument('--shards', help='capture shard directory: trace on the first shard\'s first lane (leaf paths checked)')
    p.add_argument('--listing', help='pruned step_with_chance listing (lines "NNN: source") for device lines per phase')
    p.add_argument('--no-generate', action='store_true', help='skip the generator\'s post-CSE count')
    p.add_argument('--games', type=int, default=1, help='games constructed for the template (structure only)')
    p.add_argument('--out', required=True, help='new output directory')
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    out = Path(args.out);out.mkdir(parents=True, exist_ok=False)
    from gpu_engine.v2_bench_collector import log
    begin = time.perf_counter();ctx = load_context(args);loaded = time.perf_counter() - begin
    import jax
    from gpu_engine import v2e_gen, v2e_j2c, v2e_modules
    example = ctx['template']
    if args.shards:example = shard_tree(shard_paths(args.shards, example)[0], example, lanes=1)
    fn = v2e_modules.step(ctx['engine_tables'], ctx['observation_tables'])
    begin = time.perf_counter()
    with v2e_modules.trace_substitutions():
        closed, avals, _, _ = v2e_gen.trace(lambda *a: fn(*a), tuple(example))
    traced = time.perf_counter() - begin;log('TRACED', seconds=traced)
    report = dict(schema=SCHEMA, source_commit=os.environ.get('V2_SOURCE_COMMIT'), jax=jax.__version__, example='shard' if args.shards else 'template',
        seconds=dict(load=loaded, trace=traced), phases_table=[list(p) for p in PHASES])
    begin = time.perf_counter();report['census'] = census(closed, v2e_j2c.LANES);report['seconds']['census'] = time.perf_counter() - begin
    log('CENSUS', leaf=report['census']['leaf'], groups={g: r['n'] for g, r in report['census']['groups'].items()})
    if args.listing:report['device_lines'] = listing_lines(args.listing)
    if not args.no_generate:
        begin = time.perf_counter()
        module = v2e_j2c.generate('step', closed, avals, profile=True, mode='thread')
        report['generator'] = dict(generated_counts(module.marks), stats=module.stats);report['seconds']['generate'] = time.perf_counter() - begin
        log('GENERATED', equations=report['generator']['equations'], groups=report['generator']['groups'])
    (out / 'census.json').write_text(json.dumps(report, indent=1, default=str) + '\n')
    (out / 'census.md').write_text(markdown(report))
    log('DONE', out=str(out / 'census.json'))
    return 0


if __name__ == '__main__':raise SystemExit(main())
