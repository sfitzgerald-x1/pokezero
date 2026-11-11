"""Monte Carlo-truth ladder, the backline: how the sampled worlds' unrevealed opponent Pokemon enter depth-1 search, as rungs and
diagnostics on the re-pricer's rows (CPU only; mc_ladder_worlds' and depthn_rr_score's operators, no engine, model or device).

Inputs as gpu_engine.mc_ladder_worlds: --sampled (run C / E, or a --depth1-cells replay of the same configuration), --true (run A /
B's true_private replay, optional), --weights (fp_reply_evidence weights for --sampled, optional), --manifest (self roots). R3's
reply prior throughout: the reply model's logits on Foul Play roots, the champion's on self roots.

Lane labels. A world's opponent reply lanes (the depth-1 grid's columns, peer.lanes) are labelled move:<id>, switch:<species> for a
seen Pokemon (in every sampled world: the revealed ones) or switch:? for a switch into an unseen slot (its species is the world's own
backline draw). Unseen-switch lanes of a world: its kept lanes labelled switch:?.

Rungs (mc_ladder_score's rung format; guards: the sampled row's, mc_ladder_rungs.guard; R1sp: the true row's):
- R3sw: R3 recomputed through this module's grid path (parity: equals R3 exactly).
- R3swavg_lane / R3swavg_pool / R3swavg_seen: each world keeps its prior pi_w, and its values for unseen-switch replies G_w[a][b]
  are replaced by the mean over worlds of that lane's value where it is an unseen switch (lane), by the mean over every unseen-switch
  cell of every world (pool), or by the lane mean over the worlds whose seen sets equal this world's (seen). This drops the
  covariance between pi_w(b) and G_w(a, b) on those replies: a decorrelation test, not a re-expression of averaging.
- R3noswu: unseen-switch replies dropped from pi_w and the prior renormalized over the rest (a world with nothing else keeps its
  prior): an ablation that changes the opponent policy, not a bound.
- R3pa / R3pa_rw_b1: each world keeps its grid; its prior becomes the cross-world average pi_bar = sum_w p_w pi_w over labels
  (p_w uniform, or depthn_rr_score.world_weights(loglik, 1)), mapped onto the world's lanes: a label the world lacks drops, switch:?
  mass is split equally over the world's own unseen-switch lanes, then renormalized over its kept lanes.
- R3tp / R1sp (with --true): world w's sampled grid with true world w's prior mapped by label (R3tp), and true world w's grid with
  sampled world w's prior (R1sp); the same kernel seed, so the same chance draw.
- R3rv (with --sampled rows carrying depth1.cells, depthn_rr_reprice --depth1-cells): each world keeps its prior; a cell where the
  opponent is forced to switch at the child (cells.opp_forced: its active fainted this turn, a replacement pending) takes the mean
  over worlds of the same (a, label) cell where it is also forced.
Undecided reasons: mc_ladder_worlds' (depth1_missing, refused_at_depth1, ...), true_row_missing, world_sets_missing, cells_missing,
unpriced_cell, world_weights_missing.

Diagnostics (summary.json): roots, the reply prior's mean mass on unseen-switch replies, backline coverage (per unseen true Pokemon:
the share of sampled worlds holding its species / its true set; the share in no world), and with --true the mean |G_true - G_sampled|
per reply category (move / switch_seen / switch_unseen; unseen: category means), weighted by the true prior; with cells, the share of
priced cells forced for the opponent / us and terminal, and the same |dG| split by whether the sampled cell is opponent-forced.

usage: python -m gpu_engine.mc_ladder_backline --kind foulplay|self --sampled ROWS ... [--true ROWS ...] [--weights W.jsonl]
           [--manifest MANIFEST.jsonl] --out NEW_DIR
Writes NEW_DIR/rung-<NAME>.jsonl and summary.json. Score with `python -m gpu_engine.mc_ladder_worlds pairs`.
"""
import argparse
import collections
import json
import math
import statistics
from pathlib import Path

from scripts import depthn_rr_score as rr   # first: it puts src/ on sys.path for pokezero
from gpu_engine import fp_reply_evidence as ev
from gpu_engine import mc_ladder_rungs as L
from gpu_engine import mc_ladder_worlds as M
from pokezero import v5_gumbel as original

SCHEMA = 'mc-ladder-backline.v1'
UNSEEN = 'switch:?'
CATEGORIES = ('move', 'switch_seen', 'switch_unseen')


def seen_species(worlds):
    """The species in every world's opponent party (fp_reply_evidence.world_sets), or None when a world has no party."""
    sets = [ev.world_sets(w) if type(w) is dict else None for w in worlds]
    if not sets or any(s is None for s in sets):return None
    return set.intersection(*[set(s) for s in sets])


def labels(world, seen):
    """The world's nine opponent reply lanes as labels (module docstring); None for a lane without one."""
    out = []
    for x in ((world.get('peer') or {}).get('lanes') or [None] * 9):
        if type(x) is str and x.startswith('switch:'):out.append(('switch:' + ev.species_key(x[7:])) if ev.species_key(x[7:]) in seen else UNSEEN)
        else:out.append(x if type(x) is str else None)
    return out


def category(label):
    if label is None:return None
    return 'switch_unseen' if label == UNSEEN else 'switch_seen' if label.startswith('switch:') else 'move'


def kept(p):
    return [b for b in range(9) if p.opponent_requested and p.opponent_legal[b]]


def prior(p):
    """{lane: probability} over a requested world's kept lanes (v5_gumbel._probabilities: R3's prior)."""
    probs = original._probabilities(p.opponent_logits, p.opponent_legal)
    return {b: probs[b] for b in kept(p)}


def with_grid(p, grid=None, legal=None):
    return original.WorldPrices(p.opponent_requested, tuple(legal) if legal is not None else p.opponent_legal, p.opponent_logits,
                                tuple(tuple(r) for r in grid) if grid is not None else p.joint_values)


def q_with(prices, priors, legal):
    """argmax over the legal own actions of mean_w sum_b priors[w][b] G_w[a][b] (a waiting world: column 0), lowest index on ties;
    raises rr.Unscorable('unpriced_cell') on a missing cell."""
    acts = [a for a in range(9) if legal[a]];q = {}
    for a in acts:
        terms = []
        for p, pr in zip(prices, priors):
            cells = [(1., p.joint_values[a][0])] if not p.opponent_requested else [(pr[b], p.joint_values[a][b]) for b in pr]
            if any(v is None for _, v in cells):raise rr.Unscorable('unpriced_cell')
            terms.append(math.fsum(w * v for w, v in cells))
        q[a] = math.fsum(terms) / len(terms)
    return rr.argmax(q, sorted(q))


def mapped_prior(source, source_labels, target, target_labels):
    """The source world's prior moved onto the target world's kept lanes by label (switch:? mass split equally over the target's own
    unseen-switch lanes; labels the target lacks drop), renormalized; None when no mass lands."""
    mass = collections.defaultdict(float)
    for b, v in prior(source).items():mass[source_labels[b]] += v
    return spread(mass, target, target_labels)


def spread(mass, target, target_labels):
    lanes = kept(target);unseen = sum(target_labels[b] == UNSEEN for b in lanes)
    raw = {b: (mass.get(UNSEEN, 0.) / unseen if target_labels[b] == UNSEEN else mass.get(target_labels[b], 0.)) for b in lanes}
    total = math.fsum(raw.values())
    return None if not total > 0 else {b: v / total for b, v in raw.items()}


def lane_means(prices, unseen, legal, groups):
    """{(world, a, b): mean of G_v[a][b] over v in groups[world] where b is an unseen-switch lane of v and priced}."""
    out = {}
    for w, u in enumerate(unseen):
        for b in u:
            for a in range(9):
                if not legal[a]:continue
                vals = [prices[v].joint_values[a][b] for v in groups[w] if b in unseen[v] and prices[v].joint_values[a][b] is not None]
                if vals:out[w, a, b] = math.fsum(vals) / len(vals)
    return out


def replaced(prices, means):
    out = []
    for w, p in enumerate(prices):
        grid = [list(r) for r in p.joint_values]
        for (v, a, b), m in means.items():
            if v == w:grid[a][b] = m
        out.append(with_grid(p, grid=grid))
    return out


def switch_rungs(prices, labs, legal, worlds):
    """{R3sw, R3swavg_lane, R3swavg_pool, R3swavg_seen, R3noswu: WorldPrices list} (module docstring)."""
    unseen = [{b for b in kept(p) if lab[b] == UNSEEN} for p, lab in zip(prices, labs)]
    everyone = [list(range(len(prices)))] * len(prices)
    keys = [tuple(sorted(ev.world_sets(w).items(), key=lambda kv: kv[0])) for w in worlds]
    seen = seen_species(worlds)
    keys = [tuple((m, s) for m, s in k if m in seen) for k in keys]
    bygroup = [[v for v in range(len(prices)) if keys[v] == keys[w]] for w in range(len(prices))]
    pooled = {}
    for a in range(9):
        if not legal[a]:continue
        vals = [prices[v].joint_values[a][c] for v in range(len(prices)) for c in unseen[v] if prices[v].joint_values[a][c] is not None]
        if vals:
            m = math.fsum(vals) / len(vals)
            for w, u in enumerate(unseen):
                for b in u:pooled[w, a, b] = m
    drop = []
    for p, u in zip(prices, unseen):
        rest = [k and b not in u for b, k in enumerate(p.opponent_legal)]
        drop.append(with_grid(p, legal=rest) if p.opponent_requested and any(rest) else p)
    return dict(R3sw=list(prices), R3swavg_lane=replaced(prices, lane_means(prices, unseen, legal, everyone)),
                R3swavg_pool=replaced(prices, pooled), R3swavg_seen=replaced(prices, lane_means(prices, unseen, legal, bygroup)),
                R3noswu=drop)


def prior_average(prices, labs, weights):
    """Each requested world's pi_bar (module docstring: R3pa) as {lane: probability}; a waiting world None."""
    mass = collections.defaultdict(float)
    for p, lab, w in zip(prices, labs, weights):
        if not p.opponent_requested:continue
        for b, v in prior(p).items():mass[lab[b]] += w * v
    out = []
    for p, lab in zip(prices, labs):
        if not p.opponent_requested:out.append(None);continue
        out.append(spread(mass, p, lab) or prior(p))
    return out


def reveal_average(prices, labs, cells, legal):
    """R3rv's grids: opponent-forced cells take the mean over worlds of the same (a, label) cell where it is also forced."""
    pools = collections.defaultdict(list)
    for p, lab, c in zip(prices, labs, cells):
        for a in range(9):
            if not legal[a]:continue
            for b in kept(p):
                if c['opp_forced'][a][b] and p.joint_values[a][b] is not None:pools[a, lab[b]].append(p.joint_values[a][b])
    out = []
    for p, lab, c in zip(prices, labs, cells):
        grid = [list(r) for r in p.joint_values]
        for a in range(9):
            if not legal[a]:continue
            for b in kept(p):
                vals = pools.get((a, lab[b]))
                if c['opp_forced'][a][b] and vals:grid[a][b] = math.fsum(vals) / len(vals)
        out.append(with_grid(p, grid=grid))
    return out


def grid_differences(sampled, s_labs, true, t_labs, legal, cells=None):
    """[(category, |dG|, true prior mass of the category, sampled cell opponent-forced or None)] per world pair and own action
    (module docstring)."""
    out = []
    for w, (s, sl, t, tl) in enumerate(zip(sampled, s_labs, true, t_labs)):
        if not (s.opponent_requested and t.opponent_requested):continue
        pt = prior(t);mass = collections.defaultdict(float)
        for b, v in pt.items():mass[category(tl[b])] += v
        for a in range(9):
            if not legal[a]:continue
            for cat in CATEGORIES:
                tb = [b for b in kept(t) if category(tl[b]) == cat and t.joint_values[a][b] is not None]
                if not tb:continue
                if cat == 'switch_unseen':
                    sb = [b for b in kept(s) if sl[b] == UNSEEN and s.joint_values[a][b] is not None]
                    if sb:out.append((cat, abs(statistics.fmean(t.joint_values[a][b] for b in tb) - statistics.fmean(s.joint_values[a][b] for b in sb)), mass[cat], None))
                    continue
                for b in tb:
                    x = next((c for c in kept(s) if sl[c] == tl[b] and s.joint_values[a][c] is not None), None)
                    if x is None:continue
                    forced = None if cells is None else bool(cells[w]['opp_forced'][a][x])
                    out.append((cat, abs(t.joint_values[a][b] - s.joint_values[a][x]), mass[cat], forced))
    return out


def decide(raw, true_raw, kind, weights):
    """({rung: (action or None, reason)}, normalized sampled row, normalized true row or None, per-root diagnostics)."""
    reply = M.REPLY[kind];d = M.normalized(raw);out = {};diag = {}
    names = ['R3sw', 'R3swavg_lane', 'R3swavg_pool', 'R3swavg_seen', 'R3noswu', 'R3pa'] + (['R3pa_rw_b1'] if weights is not None else []) \
        + (['R3tp', 'R1sp'] if true_raw is not False else []) + ['R3rv']
    try:
        legal, _ = L.root_prior(d);worlds = M.depth1_worlds(raw);prices = M.prices_of(worlds, reply)
        seen = seen_species(worlds)
        if seen is None:raise rr.Unscorable('world_sets_missing')
    except rr.Unscorable as error:return {n: (None, error.reason) for n in names}, d, None, diag
    labs = [labels(w, seen) for w in worlds]

    def run(name, fn):
        try:out[name] = fn(), None
        except rr.Unscorable as error:out[name] = None, error.reason
        except ValueError:out[name] = None, 'averaging_error'

    for name, ps in switch_rungs(prices, labs, legal, worlds).items():run(name, lambda ps=ps: L.average(ps, legal)[0])
    run('R3pa', lambda: q_with(prices, prior_average(prices, labs, [1. / len(prices)] * len(prices)), legal))
    if weights is not None:
        try:ll = M.logliks(weights, (raw['ordinal'], raw['decision_index']), len(raw['worlds']));ll_reason = None
        except rr.Unscorable as error:ll, ll_reason = None, error.reason
        if ll is None:out['R3pa_rw_b1'] = None, ll_reason
        else:run('R3pa_rw_b1', lambda: q_with(prices, prior_average(prices, labs, rr.world_weights(ll, 1.)), legal))
    masses = [math.fsum(v for b, v in prior(p).items() if lab[b] == UNSEEN) for p, lab in zip(prices, labs) if p.opponent_requested]
    diag['unseen_switch_mass'] = statistics.fmean(masses) if masses else None
    cells = [((w.get('depth1') or {}).get('cells')) for w in worlds]
    have_cells = all(type(c) is dict for c in cells)
    if have_cells:
        run('R3rv', lambda: L.average(reveal_average(prices, labs, cells, legal), legal)[0])
        n = sum(1 for p in prices for a in range(9) if legal[a] for b in kept(p) if p.joint_values[a][b] is not None)
        diag['cells'] = dict(priced=n, **{k: sum(1 for p, c in zip(prices, cells) for a in range(9) if legal[a] for b in kept(p)
                                                   if p.joint_values[a][b] is not None and c[k][a][b]) for k in ('opp_forced', 'own_forced', 'terminal')})
    else:out['R3rv'] = None, 'cells_missing'
    t = None
    if true_raw is not False:
        try:
            if true_raw is None:raise rr.Unscorable('true_row_missing')
            t = M.normalized(true_raw);tworlds = M.depth1_worlds(true_raw)
            if len(tworlds) != len(worlds):raise rr.Unscorable('true_row_missing')
            tprices = M.prices_of(tworlds, reply);tlabs = [labels(w, seen) for w in tworlds]
            truth = ev.world_sets(tworlds[0])
            if truth is None:raise rr.Unscorable('world_sets_missing')
        except rr.Unscorable as error:
            out['R3tp'] = out['R1sp'] = (None, error.reason)
        else:
            def swapped(grids, glabs, priors, plabs):
                mapped = []
                for g, gl, p, pl in zip(grids, glabs, priors, plabs):
                    if not g.opponent_requested:mapped.append(None);continue
                    if not p.opponent_requested:raise rr.Unscorable('prior_unmapped')
                    m = mapped_prior(p, pl, g, gl)
                    if m is None:raise rr.Unscorable('prior_unmapped')
                    mapped.append(m)
                return q_with(grids, mapped, legal)
            run('R3tp', lambda: swapped(prices, labs, tprices, tlabs))
            run('R1sp', lambda: swapped(tprices, tlabs, prices, labs))
            unseen_true = [m for m in truth if m not in seen];sets = [ev.world_sets(w) for w in worlds]
            diag['backline'] = [dict(species=sum(m in s for s in sets) / len(sets), set=sum(s.get(m) == truth[m] for s in sets) / len(sets)) for m in unseen_true]
            diag['grid_diff'] = grid_differences(prices, labs, tprices, tlabs, legal, cells if have_cells else None)
    return out, d, t, diag


def summarize(diags):
    """Means of the per-root diagnostics (module docstring)."""
    mean = lambda xs: statistics.fmean(xs) if xs else None
    out = dict(roots=len(diags), unseen_switch_mass=mean([x['unseen_switch_mass'] for x in diags if x.get('unseen_switch_mass') is not None]))
    back = [m for x in diags for m in x.get('backline', ())]
    if back:out['backline'] = dict(unseen_mons=len(back), species_share=mean([m['species'] for m in back]), set_share=mean([m['set'] for m in back]),
                                   in_no_world=mean([float(m['species'] == 0) for m in back]))
    gd = [g for x in diags for g in x.get('grid_diff', ())]
    if gd:
        out['grid_diff'] = {c: dict(cells=len(v), mean_abs=mean([g[1] for g in v]), prior_weighted=mean([g[1] * g[2] for g in v]))
                            for c in CATEGORIES for v in [[g for g in gd if g[0] == c]] if v}
        forced = [g for g in gd if g[3] is not None]
        if forced:out['grid_diff_by_forced'] = {str(f): dict(cells=len(v), mean_abs=mean([g[1] for g in v]), prior_weighted=mean([g[1] * g[2] for g in v]))
                                                for f in (True, False) for v in [[g for g in forced if g[3] is f]] if v}
    cells = [x['cells'] for x in diags if 'cells' in x]
    if cells:
        total = sum(c['priced'] for c in cells)
        out['cells'] = dict(priced=total, **{k: sum(c[k] for c in cells) / total for k in ('opp_forced', 'own_forced', 'terminal')})
    return out


def build(sampled_rows, kind, true_rows=None, weights=None, manifest=None):
    """({rung: [rung rows]}, summary)."""
    manifest = manifest or {};counts = collections.Counter();true_counts = collections.Counter()
    sampled = M.load(sampled_rows, kind, manifest, M.SAMPLED_RUN[kind], counts)
    truths = M.load(true_rows, kind, manifest, M.TRUE_RUN[kind], true_counts) if true_rows is not None else None
    rungs = collections.defaultdict(list);stats = {};diags = []
    for index in sorted(sampled):
        key, raw, _ = sampled[index]
        if truths is None:true_raw = False
        else:
            entry = truths.get(index);true_raw = None if entry is None or entry[2] is not None else entry[1]
        chosen, d, t, diag = decide(raw, true_raw, kind, weights);diags.append(diag)
        try:legal, greedy = L.root_prior(d)
        except rr.Unscorable:legal = greedy = None
        for name, (action, reason) in chosen.items():
            row = dict(key, action=action, greedy_action=greedy, guard_tripped=False, actions=None if legal is None else [a for a in range(9) if legal[a]])
            s = stats.setdefault(name, dict(rows=0, decided=0, undecided=collections.Counter(), guard_tripped=collections.Counter()));s['rows'] += 1
            if action is None:row['reason'] = reason;s['undecided'][reason] += 1
            else:
                s['decided'] += 1;tripped, why = L.guard(t if name == 'R1sp' else d, action, greedy, M.REPLY[kind])
                if tripped:row.update(guard_tripped=True, guard_reason=why);s['guard_tripped'][why] += 1
            rungs[name].append(row)
    plain = lambda c: dict(sorted(c.items()))
    summary = dict(schema=SCHEMA, kind=kind, sampled=plain(counts), true=plain(true_counts) if true_rows is not None else None, roots=len(sampled),
                   rungs={n: dict(s, undecided=plain(s['undecided']), guard_tripped=plain(s['guard_tripped'])) for n, s in stats.items()},
                   diagnostics=summarize(diags))
    return dict(rungs), summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--kind', choices=sorted(M.REPLY), required=True);parser.add_argument('--sampled', nargs='+', required=True)
    parser.add_argument('--true', nargs='+');parser.add_argument('--weights');parser.add_argument('--manifest');parser.add_argument('--out', required=True)
    args = parser.parse_args(argv)
    if (args.kind == 'self') != (args.manifest is not None):raise SystemExit('--manifest is required for self roots and refused for Foul Play roots')
    rungs, summary = build(L.read_rows(args.sampled), args.kind, L.read_rows(args.true) if args.true else None,
                           rr.load_world_weights(args.weights) if args.weights else None, L.load_manifest(args.manifest) if args.manifest else None)
    directory = Path(args.out);directory.mkdir(parents=True, exist_ok=False)
    for name, rows in rungs.items():(directory / f'rung-{name}.jsonl').write_text(''.join(json.dumps(r, sort_keys=True) + '\n' for r in rows))
    (directory / 'summary.json').write_text(json.dumps(dict(summary, arguments=vars(args)), indent=1) + '\n')
    print(json.dumps(dict(kind=summary['kind'], roots=summary['roots'], decided={n: s['decided'] for n, s in summary['rungs'].items()}), sort_keys=True))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
