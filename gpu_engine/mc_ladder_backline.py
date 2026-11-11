"""Monte Carlo-truth ladder, the backline: how the sampled worlds' unrevealed opponent Pokemon enter depth-1 search, as rungs and
diagnostics on the re-pricer's rows (CPU only; mc_ladder_worlds' and depthn_rr_score's operators, no engine, model or device).

Inputs as gpu_engine.mc_ladder_worlds: --sampled (run C / E, or a --depth1-cells replay of the same configuration), --true (run A /
B's true_private replay, optional), --weights (fp_reply_evidence weights for --sampled, optional), --manifest (self roots),
--check-r3 (that run's rung-R3.jsonl: R3sw is held to it). R3's reply prior throughout: the reply model's logits on Foul Play
roots, the champion's on self roots.

Lane labels. A world's opponent reply lanes (the depth-1 grid's columns, peer.lanes) are labelled move:<id>, switch:<species> for a
seen Pokemon (in every sampled world: in practice the revealed ones; an unrevealed species drawn into all W worlds counts as seen
too) or switch:? for a switch into an unseen slot (its species is the world's own backline draw). Unseen-switch lanes of a world:
its kept lanes labelled switch:?. A requested world with a kept lane without a label, or whose kept lanes are not its peer view's
legal lanes (kept_matches_peer_legal false), refuses the root (lane_unlabelled).

Priors mapped by label (R3pa, R3tp, R1sp): a target world's kept lane takes its label's mass from the source mass (a label the
target lacks drops); switch:? mass is split over the target's own unseen-switch lanes in proportion to the target world's own
prior over them (equally where that prior puts nothing there), so a world mapped onto itself keeps its prior; then renormalized
over its kept lanes. With no mass landing, R3pa keeps the world's own prior (counted: prior_fallback_worlds); R3tp / R1sp refuse
(prior_unmapped). q is depthn_rr_score.prior_q on the worlds' grids: averaging's own summation, so its own priors give R3's q.

Rungs (mc_ladder_score's rung format; guards: the sampled row's, mc_ladder_rungs.guard; R1sp: the true row's):
- R3sw: R3 recomputed through this module (held to --check-r3).
- R3swavg_lane / R3swavg_pool / R3swavg_seen: each world keeps its prior pi_w, and its priced values for unseen-switch replies
  G_w[a][b] are replaced by the mean over worlds of that lane's priced value where it is an unseen switch (lane), by the mean over
  every priced unseen-switch cell of every world (pool), or by the lane mean over the worlds whose seen Pokemon's sets equal this
  world's (seen; group sizes in the summary). This drops the covariance between pi_w(b) and G_w(a, b) on those replies: a
  decorrelation test, not a re-expression of averaging. An unpriced cell stays unpriced.
- R3noswu: unseen-switch replies dropped from pi_w and the prior renormalized over the rest (a world with nothing else keeps its
  prior): an ablation that changes the opponent policy, not a bound.
- R3pa / R3pa_rw_b1: each world keeps its grid; its prior becomes the cross-world average pi_bar = sum_w p_w pi_w over labels
  (p_w uniform, or depthn_rr_score.world_weights(loglik, 1)), mapped onto the world's lanes as above.
- R3tp / R1sp (with --true): world w's sampled grid with true world w's prior (R3tp), and true world w's grid with sampled world
  w's prior (R1sp). World w of the two rows must share its kernel seed (the same chance draw; kernel_seed_mismatch otherwise) and
  the two rows their root prior (mc_ladder_worlds.same_root; root_prior_differs).
- R3rv (with --sampled rows carrying depth1.cells, depthn_rr_reprice --depth1-cells): each world keeps its prior; a priced cell
  where the opponent is forced to switch at the child (cells.opp_forced: after a faint or a Baton Pass, a replacement pending)
  takes the mean over worlds of each world's mean over its own forced priced cells with the same (a, label).
Undecided reasons: mc_ladder_worlds' (depth1_missing[:cause], refused_at_depth1, block_shape, prefix_too_long, root_prior_missing,
...), world_sets_missing, lane_unlabelled, unpriced_cell, averaging_error, nonfinite, world_weights_missing / world_weights_mismatch,
cells_missing, and for R3tp / R1sp: true_row_missing (no row, or another world count), the true row's own true_worlds_failed /
true_worlds_not_sampled, kernel_seed_mismatch, root_prior_differs, prior_unmapped.

Diagnostics (summary.json): roots, the reply prior's mean mass on unseen-switch replies, the seen-group sizes, backline coverage
(per unseen true Pokemon: the share of sampled worlds holding its species / its true set; the share in no world), and with --true
the value differences per reply category (move / switch_seen / switch_unseen) between true world w and sampled world w, per
(world, own action) pair: for each true lane its |G_true - G_sampled| at the label-matched sampled lane, weighted by the lane's
true prior (switch_unseen: the category means' difference, weighted by the category's true mass). Reported per category: cells,
mean_abs (unweighted), weighted_mean (sum |dG| pi / sum pi) and contribution (sum |dG| pi per (world, action) pair: the category's
share of the expected |dQ|); with cells, the same split by whether the sampled cell is opponent-forced, and the share of priced
cells forced for either seat or terminal.

usage: python -m gpu_engine.mc_ladder_backline --kind foulplay|self --sampled ROWS ... [--true ROWS ...] [--weights W.jsonl]
           [--manifest MANIFEST.jsonl] [--check-r3 rung-R3.jsonl] --out NEW_DIR
Writes NEW_DIR/rung-<NAME>.jsonl and summary.json (exit 1 when --check-r3 finds a disagreement or compares nothing). Score with
`python -m gpu_engine.mc_ladder_worlds pairs`.
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
from gpu_engine import search_hedge as hedge
from pokezero import v5_gumbel as original

SCHEMA = 'mc-ladder-backline.v1'
UNSEEN = 'switch:?'
CATEGORIES = ('move', 'switch_seen', 'switch_unseen')
EXAMPLES = 20


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


def check_labels(worlds, prices, labs):
    """Refuse (lane_unlabelled) a requested world with an unlabelled kept lane or kept lanes other than its peer view's legal ones."""
    for w, p, lab in zip(worlds, prices, labs):
        if not p.opponent_requested:continue
        if w.get('kept_matches_peer_legal') is False or any(lab[b] is None for b in kept(p)):raise rr.Unscorable('lane_unlabelled')


def with_grid(p, grid=None, legal=None):
    return original.WorldPrices(p.opponent_requested, tuple(legal) if legal is not None else p.opponent_legal, p.opponent_logits,
                                tuple(tuple(r) for r in grid) if grid is not None else p.joint_values)


def q_with(prices, priors, legal):
    """argmax of depthn_rr_score.prior_q (averaging's summation) with priors[w] ({lane: probability}; None: R3's own prior, or a
    waiting world's column 0), lowest index on ties; raises rr.Unscorable."""
    try:gs = hedge.grids(prices, legal)
    except ValueError:raise rr.Unscorable('averaging_error') from None
    if gs is None:raise rr.Unscorable('unpriced_cell')
    full = []
    for p, g, pr in zip(prices, gs, priors):
        if not p.opponent_requested:full.append(rr.WAITING);continue
        if pr is None:full.append(tuple(g.prior[g.replies.index(b)] if b in g.replies else 0. for b in range(9)));continue
        if any(pr.get(b, 0.) > 0 and b not in g.replies for b in range(9)):raise rr.Unscorable('unpriced_cell')
        full.append(tuple(pr.get(b, 0.) for b in range(9)))
    q = rr.prior_q(gs, full)
    if any(not math.isfinite(v) for v in q.values()):raise rr.Unscorable('nonfinite')
    return rr.argmax(q, sorted(q))


def spread(mass, target, target_labels):
    """Label mass onto a target world's kept lanes (module docstring); None when nothing lands."""
    lanes = kept(target);own = prior(target);unseen = [b for b in lanes if target_labels[b] == UNSEEN]
    own_unseen = math.fsum(own[b] for b in unseen)
    raw = {}
    for b in lanes:
        if target_labels[b] == UNSEEN:raw[b] = mass.get(UNSEEN, 0.) * (own[b] / own_unseen if own_unseen > 0 else 1. / len(unseen))
        else:raw[b] = mass.get(target_labels[b], 0.)
    total = math.fsum(raw.values())
    return None if not total > 0 else {b: v / total for b, v in raw.items()}


def label_mass(p, lab):
    mass = collections.defaultdict(float)
    for b, v in prior(p).items():mass[lab[b]] += v
    return mass


def unseen_lanes(prices, labs):
    return [{b for b in kept(p) if lab[b] == UNSEEN} for p, lab in zip(prices, labs)]


def replaced(prices, means):
    """The worlds with priced cells (w, a, b) of `means` replaced; an unpriced cell stays unpriced."""
    out = []
    for w, p in enumerate(prices):
        grid = [list(r) for r in p.joint_values]
        for (v, a, b), m in means.items():
            if v == w and grid[a][b] is not None:grid[a][b] = m
        out.append(with_grid(p, grid=grid))
    return out


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


def seen_groups(worlds):
    """Per world, the worlds whose seen Pokemon's sets equal its own."""
    seen = seen_species(worlds);keys = [tuple(sorted((m, s) for m, s in ev.world_sets(w).items() if m in seen)) for w in worlds]
    return [[v for v in range(len(worlds)) if keys[v] == keys[w]] for w in range(len(worlds))]


def switch_rungs(prices, labs, legal, groups):
    """{R3sw, R3swavg_lane, R3swavg_pool, R3swavg_seen, R3noswu: WorldPrices list} (module docstring)."""
    unseen = unseen_lanes(prices, labs);pooled = {}
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
    everyone = [list(range(len(prices)))] * len(prices)
    return dict(R3sw=list(prices), R3swavg_lane=replaced(prices, lane_means(prices, unseen, legal, everyone)),
                R3swavg_pool=replaced(prices, pooled), R3swavg_seen=replaced(prices, lane_means(prices, unseen, legal, groups)), R3noswu=drop)


def prior_average(prices, labs, weights, counts=None):
    """Each requested world's pi_bar ({lane: probability}) mapped onto its lanes (module docstring: R3pa); a waiting world None."""
    mass = collections.defaultdict(float)
    for p, lab, w in zip(prices, labs, weights):
        if not p.opponent_requested:continue
        for label, v in label_mass(p, lab).items():mass[label] += w * v
    out = []
    for p, lab in zip(prices, labs):
        if not p.opponent_requested:out.append(None);continue
        mapped = spread(mass, p, lab)
        if mapped is None and counts is not None:counts['prior_fallback_worlds'] += 1
        out.append(mapped or prior(p))
    return out


def forced_set(cells, key='opp_forced'):
    return {tuple(x) for x in (cells or {}).get(key) or ()}


def reveal_average(prices, labs, cells, legal):
    """R3rv's grids (module docstring): per (a, label), the mean over worlds of each world's mean over its forced priced cells."""
    forced = [forced_set(c) for c in cells];pools = collections.defaultdict(list)
    for p, lab, f in zip(prices, labs, forced):
        mine = collections.defaultdict(list)
        for a, b in f:
            if legal[a] and b in kept(p) and p.joint_values[a][b] is not None:mine[a, lab[b]].append(p.joint_values[a][b])
        for k, vals in mine.items():pools[k].append(math.fsum(vals) / len(vals))
    out = []
    for p, lab, f in zip(prices, labs, forced):
        grid = [list(r) for r in p.joint_values]
        for a, b in f:
            vals = pools.get((a, lab[b])) if legal[a] and b in kept(p) else None
            if vals and grid[a][b] is not None:grid[a][b] = math.fsum(vals) / len(vals)
        out.append(with_grid(p, grid=grid))
    return out


def grid_differences(sampled, s_labs, true, t_labs, legal, cells=None):
    """[(category, |dG|, weight, sampled cell opponent-forced or None)] per world pair and own action (module docstring), and the
    number of (world, own action) pairs."""
    out, pairs = [], 0
    for w, (s, sl, t, tl) in enumerate(zip(sampled, s_labs, true, t_labs)):
        if not (s.opponent_requested and t.opponent_requested):continue
        pt = prior(t);forced = forced_set(cells[w]) if cells is not None else None
        for a in range(9):
            if not legal[a]:continue
            pairs += 1
            for cat in CATEGORIES:
                tb = [b for b in kept(t) if category(tl[b]) == cat and t.joint_values[a][b] is not None]
                if not tb:continue
                if cat == 'switch_unseen':
                    sb = [b for b in kept(s) if sl[b] == UNSEEN and s.joint_values[a][b] is not None]
                    if sb:out.append((cat, abs(statistics.fmean(t.joint_values[a][b] for b in tb) - statistics.fmean(s.joint_values[a][b] for b in sb)),
                                      math.fsum(pt[b] for b in tb), None))
                    continue
                for b in tb:
                    x = next((c for c in kept(s) if sl[c] == tl[b] and s.joint_values[a][c] is not None), None)
                    if x is None:continue
                    out.append((cat, abs(t.joint_values[a][b] - s.joint_values[a][x]), pt[b], None if forced is None else (a, x) in forced))
    return out, pairs


def true_pairing(raw, true_entry, worlds):
    """The true row's depth-1 worlds paired with `worlds` (module docstring: R3tp / R1sp), or raises rr.Unscorable."""
    if true_entry is None:raise rr.Unscorable('true_row_missing')
    if true_entry[2] is not None:raise rr.Unscorable(true_entry[2])
    tworlds = M.depth1_worlds(true_entry[1])
    if len(tworlds) != len(worlds):raise rr.Unscorable('true_row_missing')
    for s, t in zip(worlds, tworlds):
        seed = (t or {}).get('kernel_seed')
        if type(seed) is not int or seed != (s or {}).get('kernel_seed'):raise rr.Unscorable('kernel_seed_mismatch')
    if not M.same_root(raw, true_entry[1]):raise rr.Unscorable('root_prior_differs')
    return tworlds


def decide(raw, true_entry, kind, weights, counts, with_true):
    """({rung: (action or None, reason)}, normalized sampled row, normalized true row or None, per-root diagnostics)."""
    reply = M.REPLY[kind];d = M.normalized(raw);out = {};diag = {}
    names = ['R3sw', 'R3swavg_lane', 'R3swavg_pool', 'R3swavg_seen', 'R3noswu', 'R3pa'] + (['R3pa_rw_b1'] if weights is not None else []) \
        + (['R3tp', 'R1sp'] if with_true else []) + ['R3rv']

    def undecided(reason):
        cause = d['refusal_cause']
        return None, reason + (f':{cause}' if cause and reason == 'depth1_missing' else '')

    try:
        legal, _ = L.root_prior(d);worlds = M.depth1_worlds(raw);prices = M.prices_of(worlds, reply)
        seen = seen_species(worlds)
        if seen is None:raise rr.Unscorable('world_sets_missing')
        labs = [labels(w, seen) for w in worlds];check_labels(worlds, prices, labs)
    except rr.Unscorable as error:return {n: undecided(error.reason) for n in names}, d, None, diag

    def run(name, fn):
        try:out[name] = fn(), None
        except rr.Unscorable as error:out[name] = undecided(error.reason)

    def averaged(ps):
        try:return L.average(ps, legal)[0]
        except ValueError:raise rr.Unscorable('averaging_error') from None

    groups = seen_groups(worlds);diag['seen_group_size'] = statistics.fmean(len(g) for g in groups)
    for name, ps in switch_rungs(prices, labs, legal, groups).items():run(name, lambda ps=ps: averaged(ps))
    run('R3pa', lambda: q_with(prices, prior_average(prices, labs, [1. / len(prices)] * len(prices), counts), legal))
    if weights is not None:
        try:ll = M.logliks(weights, (raw['ordinal'], raw['decision_index']), len(raw['worlds']))
        except rr.Unscorable as error:out['R3pa_rw_b1'] = undecided(error.reason)
        else:run('R3pa_rw_b1', lambda: q_with(prices, prior_average(prices, labs, rr.world_weights(ll, 1.), counts), legal))
    masses = [math.fsum(v for b, v in prior(p).items() if lab[b] == UNSEEN) for p, lab in zip(prices, labs) if p.opponent_requested]
    diag['unseen_switch_mass'] = statistics.fmean(masses) if masses else None
    cells = [(w.get('depth1') or {}).get('cells') for w in worlds]
    have_cells = all(type(c) is dict for c in cells)
    if have_cells:
        run('R3rv', lambda: averaged(reveal_average(prices, labs, cells, legal)))
        priced = [(w, a, b) for w, p in enumerate(prices) for a in range(9) if legal[a] for b in kept(p) if p.joint_values[a][b] is not None]
        sets = [{k: forced_set(c, k) for k in ('opp_forced', 'own_forced', 'terminal')} for c in cells]
        diag['cells'] = dict(priced=len(priced), **{k: sum((a, b) in sets[w][k] for w, a, b in priced) for k in ('opp_forced', 'own_forced', 'terminal')})
    else:out['R3rv'] = None, 'cells_missing'
    t = None
    if with_true:
        try:
            tworlds = true_pairing(raw, true_entry, worlds);t = M.normalized(true_entry[1])
            tprices = M.prices_of(tworlds, reply);tlabs = [labels(w, seen) for w in tworlds];check_labels(tworlds, tprices, tlabs)
            truth = ev.world_sets(tworlds[0])
            if truth is None:raise rr.Unscorable('world_sets_missing')
        except rr.Unscorable as error:
            t = None;out['R3tp'] = out['R1sp'] = undecided(error.reason)
        else:
            def swapped(grids, glabs, priors, plabs):
                mapped = []
                for g, gl, p, pl in zip(grids, glabs, priors, plabs):
                    if not g.opponent_requested:mapped.append(None);continue
                    m = spread(label_mass(p, pl), g, gl) if p.opponent_requested else None
                    if m is None:raise rr.Unscorable('prior_unmapped')
                    mapped.append(m)
                return q_with(grids, mapped, legal)
            run('R3tp', lambda: swapped(prices, labs, tprices, tlabs))
            run('R1sp', lambda: swapped(tprices, tlabs, prices, labs))
            unseen_true = [m for m in truth if m not in seen];sets = [ev.world_sets(w) for w in worlds]
            diag['backline'] = [dict(species=sum(m in s for s in sets) / len(sets), set=sum(s.get(m) == truth[m] for s in sets) / len(sets)) for m in unseen_true]
            diag['grid_diff'], diag['grid_pairs'] = grid_differences(prices, labs, tprices, tlabs, legal, cells if have_cells else None)
    return out, d, t, diag


def summarize(diags):
    """Means of the per-root diagnostics (module docstring)."""
    mean = lambda xs: statistics.fmean(xs) if xs else None
    out = dict(roots=len(diags), unseen_switch_mass=mean([x['unseen_switch_mass'] for x in diags if x.get('unseen_switch_mass') is not None]),
               seen_group_size=mean([x['seen_group_size'] for x in diags if 'seen_group_size' in x]))
    back = [m for x in diags for m in x.get('backline', ())]
    if back:out['backline'] = dict(unseen_mons=len(back), species_share=mean([m['species'] for m in back]), set_share=mean([m['set'] for m in back]),
                                   in_no_world=mean([float(m['species'] == 0) for m in back]))
    gd = [g for x in diags for g in x.get('grid_diff', ())];pairs = sum(x.get('grid_pairs', 0) for x in diags)

    def stats(v):
        w = math.fsum(g[2] for g in v)
        return dict(cells=len(v), mean_abs=mean([g[1] for g in v]), weighted_mean=math.fsum(g[1] * g[2] for g in v) / w if w > 0 else None,
                    contribution=math.fsum(g[1] * g[2] for g in v) / pairs if pairs else None)
    if gd:
        out['grid_diff'] = dict(pairs=pairs, **{c: stats(v) for c in CATEGORIES for v in [[g for g in gd if g[0] == c]] if v})
        forced = [g for g in gd if g[3] is not None]
        if forced:out['grid_diff_by_forced'] = {str(f): stats(v) for f in (True, False) for v in [[g for g in forced if g[3] is f]] if v}
    cells = [x['cells'] for x in diags if 'cells' in x]
    if cells:
        total = sum(c['priced'] for c in cells)
        out['cells'] = dict(priced=total, **{k: sum(c[k] for c in cells) / total for k in ('opp_forced', 'own_forced', 'terminal')})
    return out


def build(sampled_rows, kind, true_rows=None, weights=None, manifest=None, check_r3=None):
    """({rung: [rung rows]}, summary)."""
    manifest = manifest or {};counts = collections.Counter();true_counts = collections.Counter();notes = collections.Counter()
    sampled = M.load(sampled_rows, kind, manifest, M.SAMPLED_RUN[kind], counts)
    with_true = true_rows is not None
    truths = M.load(true_rows, kind, manifest, M.TRUE_RUN[kind], true_counts) if with_true else {}
    rungs = collections.defaultdict(list);stats = {};diags = []
    for index in sorted(sampled):
        key, raw, _ = sampled[index]
        chosen, d, t, diag = decide(raw, truths.get(index), kind, weights, notes, with_true);diags.append(diag)
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
    check = None
    if check_r3 is not None:
        mine = {tuple(r[f] for f in L.KEY_FIELDS[kind]): r['action'] for r in rungs.get('R3sw', ())};check = dict(compared=0, disagreements=0, examples=[])
        for r in check_r3:
            index = tuple(r[f] for f in L.KEY_FIELDS[kind])
            if index not in mine or mine[index] is None or r.get('action') is None:continue
            check['compared'] += 1
            if mine[index] != r['action']:
                check['disagreements'] += 1
                if len(check['examples']) < EXAMPLES:check['examples'].append(dict(root=list(index), mine=mine[index], theirs=r['action']))
    plain = lambda c: dict(sorted(c.items()))
    summary = dict(schema=SCHEMA, kind=kind, sampled=plain(counts), true=plain(true_counts) if with_true else None, roots=len(sampled), notes=plain(notes),
                   rungs={n: dict(s, undecided=plain(s['undecided']), guard_tripped=plain(s['guard_tripped'])) for n, s in stats.items()},
                   r3_check=check, diagnostics=summarize(diags))
    return dict(rungs), summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--kind', choices=sorted(M.REPLY), required=True);parser.add_argument('--sampled', nargs='+', required=True)
    parser.add_argument('--true', nargs='+');parser.add_argument('--weights');parser.add_argument('--manifest');parser.add_argument('--check-r3')
    parser.add_argument('--out', required=True)
    args = parser.parse_args(argv)
    if (args.kind == 'self') != (args.manifest is not None):raise SystemExit('--manifest is required for self roots and refused for Foul Play roots')
    rungs, summary = build(L.read_rows(args.sampled), args.kind, L.read_rows(args.true) if args.true else None,
                           rr.load_world_weights(args.weights) if args.weights else None, L.load_manifest(args.manifest) if args.manifest else None,
                           M.read_jsonl(args.check_r3) if args.check_r3 else None)
    directory = Path(args.out);directory.mkdir(parents=True, exist_ok=False)
    for name, rows in rungs.items():(directory / f'rung-{name}.jsonl').write_text(''.join(json.dumps(r, sort_keys=True) + '\n' for r in rows))
    (directory / 'summary.json').write_text(json.dumps(dict(summary, arguments=vars(args)), indent=1) + '\n')
    print(json.dumps(dict(kind=summary['kind'], roots=summary['roots'], r3_check=summary['r3_check'],
                          decided={n: s['decided'] for n, s in summary['rungs'].items()}), sort_keys=True))
    check = summary['r3_check']
    return 1 if check is not None and (check['disagreements'] or not check['compared']) else 0


if __name__ == '__main__':
    raise SystemExit(main())
